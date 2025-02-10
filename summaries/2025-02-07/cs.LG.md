New uploads on arXiv(cs.CL)

### Can Grammarly and ChatGPT accelerate language change? AI-powered technologies and their impact on the English language: wordiness vs. conciseness (https://arxiv.org/abs/2502.04324)
Comments:
          10 pages, article

- **What's New**: 이 논문은 자연어 처리(NLP) 기반의 언어 기술과 AI 기반의 자연어 생성 모델이 영어 언어에 미치는 영향을 연구합니다. 특히 Grammarly와 ChatGPT가 단어의 잉여성과 간결성에 대해 어떻게 영향을 미치는지를 분석합니다. 그 과정에서 영어 문장 구조를 간소화하길 권장하는 두 도구의 차이를 강조합니다.

- **Technical Details**: 이 연구는 목적 부사(subordinator)인 'in order to'를 중심으로 한 사례 연구를 통해, 원어민이 작성한 문장들도 Grammarly와 ChatGPT가 더 짧고 간결한 문장 구조를 추천하는 방법을 보여줍니다. 분석된 문장은 현대 영어 코퍼스에서 추출된 것이며 완벽하게 문법적으로 맞습니다. 그러나 두 도구는 상대적으로 짧은 문장에 대해서도 간결성을 우선시하는 경향을 보입니다.

- **Performance Highlights**: 결과적으로 논문은 Grammarly와 같은 기술이 단순히 언어 변화를 반영하는 것을 넘어 언어의 변화를 촉진하거나 가속화할 가능성이 있음을 주장합니다. 이는 언어 기술의 발전이 실제 소통 방식에 중대한 영향을 미칠 수 있음을 나타내는 중요한 발견으로, 언어학자에게는 흥미로운 결과입니다.



### Variation of sentence length across time and genr (https://arxiv.org/abs/2502.04321)
Comments:
          20 pages

- **What's New**: 이 논문은 역사적 미국 영어 코퍼스(Corpus of Historical American English)라는 대규모 다층 장르 코퍼스를 활용하여 언어 변화의 경향을 연구하는 실용적인 측면을 제시합니다. 또한, 지난 몇 세기 동안 영어 문장의 길이가 지속적으로 감소하고 있다는 widely held assumption (널리 퍼진 가정)을 검증합니다. 마지막으로 문장 길이의 변화와 영어 구문 사용의 변화 사이의 가능한 연관성을 강조합니다.

- **Technical Details**: 연구는 문장 길이와 장르(genre) 그리고 비한정 목적 종속 접속사(in order to)의 빈도 감소 사이의 상관관계(interrelation)를 분석합니다. 특히, 비한정 목적 부사절의 사용 감소가 문장 길이의 변화와 어떻게 연결되는지에 대한 실증적 증거(empirical proof of concept)를 제시합니다. 이는 영어 문헌에서 문장 구조의 진화(evolution of syntactic structure)와 관련된 중요한 통찰(insight)을 제공합니다.

- **Performance Highlights**: 이러한 연구 결과는 영어 문장 길이에 대한 새로운 이해를 제공하며, 언어 변화 연구의 중요한 기초 자료로 작용할 수 있습니다. 문장 길이와 내용에서 보이는 장르별 차이 점의 분석은 향후 언어학적(metalinguistic) 연구나 교육에 있어 유용하게 활용될 수 있을 것으로 기대됩니다.



### ChamaleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters (https://arxiv.org/abs/2502.04315)
- **What's New**: 최근 큰 발전을 이루고 있는 대규모 언어 모델(LLMs)의 연구는 다양한 작업에서 놀라운 성능을 보여주고 있습니다. 이 논문에서는 ChamaleonLLM이라는 새로운 프레임워크를 소개하며, 이는 추론(inference) 시 모델이 동적으로 적응할 수 있도록 지원합니다. 전통적인 방법들과는 달리, ChamaleonLLM은 배치(batch) 기반 클러스터링(clustering)과 저차원(low-rank) 업데이트를 실시간으로 생성하는 방식을 활용하여 성능을 극대화합니다.

- **Technical Details**: ChamaleonLLM의 핵심은 배치 통계치를 기반으로 저차원(modification) 업데이트를 동적으로 생성하는 것입니다. 입력은 의미적 및 구문적 유사성에 따라 클러스터로 그룹화되어, 동질적인 입력들로 이루어진 미니 배치(batch)가 생성됩니다. 또한, 하이퍼 네트워크(hyper-network)를 통해 모델의 디코더(decoder) 가중치를 실시간으로 적응시키는 방식을 채택하여, 추론 과정에서 입력 데이터의 세부 사항을 보다 유연하게 반영합니다.

- **Performance Highlights**: 실험 결과, ChamaleonLLM은 기존의 LoRA 방식에 비해 향상된 성능을 보여 주었으며, 동적인 데이터 상황에서도 유연하게 대처할 수 있는 잠재력이 있습니다. 이 접근법은 메모리 및 계산 요구 사항을 줄이면서도, 고성능의 언어 모델 추론을 가능하게 하여 다양한 작업에 적응할 수 있는 잠재력을 지니고 있습니다. ChamaleonLLM은 오픈 소스로 제공되어 실험의 재현성을 보장하며, 연구자들이 이 프레임워크의 이점을 쉽게 활용할 수 있도록 하고 있습니다.



### BOUQuET: dataset, Benchmark and Open initiative for Universal Quality Evaluation in Translation (https://arxiv.org/abs/2502.04314)
- **What's New**: 이 논문은 다중 중심 및 다중 등록/영역 데이터 세트 및 벤치마크인 BOUQuET를 제시합니다. 이 데이터 세트는 비영어 언어로 처음 수작업으로 만들어졌으며, 전 세계 인구의 절반이 일반적으로 사용하는 23개 언어가 포함되어 있습니다. 이는 보다 정확한 번역을 가능하게 하는 중심 언어(pivot languages)로 기능할 수 있습니다. 데이터 세트는 문장 수준을 넘어서 여러 길이의 단락으로 구성되어 있으며, 다국어 언어 특성을 강제하기 위해 멀티센트릭을 고려하여 설계되었습니다.

- **Technical Details**: BOUQuET 데이터 세트는 프랑스어, 독일어, 힌디어, 인도네시아어, 중국어(만다린), 러시아어 및 스페인어 등 사용이 빈번한 언어로 구성되어 있습니다. 각 언어는 최종 데이터 세트에 동일한 수의 문장을 제공하며, 이는 상위 20개 언어에 포함됩니다. 또한, 데이터 세트는 다양한 언어적 특성을 포함하여, 다르게 등록(register)되며, 동적 확장성(dynamic extensibility)을 내장하고 있습니다.

- **Performance Highlights**: BOUQuET는 기계 번역(Machine Translation) 데이터 세트와 비교할 때 광범위한 도메인을 포괄하며 비전문가에게 번역 작업을 단순화한다고 주장합니다. BOUQuET는 다중 언어 발전에 기여하기 위해 지역 사회의 참여를 통해 지속적으로 발전할 수 있는 가능성을 가지고 있습니다. 이러한 데이터 세트는 번역 참여를 위한 공개 이니셔티브에 특히 적합합니다.



### ScoreFlow: Mastering LLM Agent Workflows via Score-based Preference Optimization (https://arxiv.org/abs/2502.04306)
Comments:
          Project: this https URL

- **What's New**: 최근의 연구는 대규모 언어 모델(multi-agent systems)을 활용하여 복잡한 문제 해결을 시도하고, 이를 구축하는 데 필요한 수작업을 줄이는 방향으로 나아가고 있습니다. 새로운 프레임워크인 ScoreFlow는 이러한 도전 과제에 대응하기 위해 마련되었으며, 효율적인 그래디언트 기반 최적화(gradient-based optimization)를 연속 공간에서 활용합니다.

- **Technical Details**: ScoreFlow는 Score-DPO라는 새로운 변형을 포함하고 있으며, 이는 정량적 피드백(quantitative feedback)을 고려한 직접 선호 최적화(direct preference optimization) 방법입니다. 기존의 불규칙한 최적화 기법들의 한계로 인해 유연성이 부족하고, 적응성이 떨어지며 확장성에서도 문제를 겪고 있던 기존 방법들과는 차별화되는 특징을 지닌다고 할 수 있습니다.

- **Performance Highlights**: ScoreFlow는 질문 응답(question answering), 코딩(coding), 수학적 추론(mathematical reasoning) 등의 여섯 개 벤치마크에서 기존 기준선 대비 8.2%의 성능 향상을 달성했습니다. 또한, 이 프레임워크는 더 작은 모델이 더 큰 모델보다 낮은 추론 비용(inference costs)으로 우수한 성능을 발휘할 수 있도록 지원합니다.



### Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization (https://arxiv.org/abs/2502.04295)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 성능 개선을 위한 새로운 방법론인 콘텐츠-형식 통합 프롬프트 최적화(Content-Format Integrated Prompt Optimization, CFPO)를 제안합니다. CFPO는 프롬프트의 내용과 형식을 동시에 최적화하는 혁신적인 접근 방식을 포함하고, 반복적인 정련 과정을 통해 접근합니다. 또한, 이 방법은 자연어 변형을 활용하여 콘텐츠 변형을 탐색하고, 다양한 형식 옵션을 체계적으로 평가하는 동적 형식 탐색 전략을 사용합니다.

- **Technical Details**: CFPO는 두 가지 주요 차원에서 작업하는 구조적인 사고의 원칙을 활용합니다. 첫째, 프롬프트 렌더러(Prompt Renderer)는 프롬프트 내 모든 구성 요소의 조직 구조를 관리하며, 둘째, 쿼리 형식(Query Format)은 인스턴스 학습 예제와 쿼리의 발표 방식을 결정합니다. 이 두 가지 차원을 통합하여 CFPO는 콘텐츠와 형식 유형 간의 효과적인 구분을 가능하게 하는 구조적 템플릿을 정의합니다.

- **Performance Highlights**: 다양한 작업에 대해 여러 오픈 소스 LLM을 사용하여 CFPO의 성능을 평가한 결과, CFPO는 콘텐츠 전용 최적화 방법에 비해 측정 가능한 성능 개선 효과를 보여주었습니다. 이는 통합된 콘텐츠-형식 최적화의 중요성을 강조하며, LLM 성능 향상을 위한 실질적이고 모델 독립적인 접근 방식을 제공합니다.



### A Methodology for Studying Linguistic and Cultural Change in China, 1900-1950 (https://arxiv.org/abs/2502.04286)
Comments:
          14 pages, 4 figures

- **What's New**: 이 논문은 20세기 전반기, 특히 중국의 언어와 문화 변화에 대한 정량적 접근 방식을 제시합니다. 이 시기는 컴퓨터 인문학(Computational Humanities) 연구에서 다루어진 적이 적으며, 새로운 연구 기법의 필요성을 강조합니다.

- **Technical Details**: 이 연구는 19세기 후반과 20세기 중국 텍스트 분석을 위한 프레임워크를 제공합니다. 단어 수 집계(word counts)와 단어 임베딩(word embeddings)과 같은 기존 방법론이 어떻게 역사적인 통찰을 제공할 수 있는지를 시연합니다.

- **Performance Highlights**: 이 연구를 통해 서구 현대성과 중국 문화 담론 간의 복잡한 협상이 어떻게 이루어졌는지를 이해할 수 있는 새로운 역사적 통찰력을 얻게 됩니다. 이 방법론은 기존의 텍스트 분석 기법에 대한 새로운 반성을 제시합니다.



### How does a Multilingual LM Handle Multiple Languages? (https://arxiv.org/abs/2502.04269)
Comments:
          10 pages, 8 figures

- **What's New**: 이번 연구는 BLOOM 1.7B와 같은 다언어 모델들의 언어 처리 능력을 면밀히 평가하며, 저자원 언어에 대한 모델의 효과성을 중점적으로 다룹니다. 이 논문에서는 고자원 언어와 저자원 언어 간의 의미적 유사성을 연구하며, 모델의 내적 문법 및 의미 인코딩을 검토합니다. 특히, 이 연구는 다언어 NLP 모델의 개선을 목표로 하여 이를 통해 언어 기술의 포괄성을 제고하려고 합니다.

- **Technical Details**: 이 연구에서는 다언어 단어 임베딩 분석과 예측 작업을 통해 다언어 모델의 성능과 특성을 분석합니다. Google Cloud Translation API를 활용한 고품질 번역을 통해 5,000개의 영어 단어로 구성된 코퍼스를 생성하고, 이를 사용하여 cosine similarity를 통해 의미적 유사성을 평가합니다. 또한, 모델의 내부 동작을 이해하기 위해 숨겨진 상태 분석을 수행하며, 각 레이어의 역할과 과제별 기능을 평가합니다.

- **Performance Highlights**: BLOOM 모델의 성능은 인도유럽어 계열 언어인 힌디어와 타밀어가 초기 레이어에서 0.92-0.95로 높은 유사성을 보인 반면, 아랍어는 초기 유사성이 0.50으로 낮았음을 보여줍니다. 심층 레이어에서는 모든 언어가 유사성 점수가 감소하였으며, 특히 아랍어는 상대적으로 낮은 유사성 점수를 유지했습니다. 이러한 결과는 모델이 저자원 언어 및 특정 언어 과제를 처리하는데 있어 문제를 드러냅니다.



### TriNER: A Series of Named Entity Recognition Models For Hindi, Bengali & Marath (https://arxiv.org/abs/2502.04245)
- **What's New**: 이번 연구에서는 인도의 세 가지 주요 언어인 힌디어(Hindi), 벵골어(Bengali), 마라티어(Marathi)에 대한 다국어(NER) 모델을 개발하였습니다. 이러한 언어의 복잡성과 다양성 때문에 발생하는 문제를 해결하기 위해, 단일 모델을 통해 다양한 엔티티 그룹을 통합적으로 식별하는 방법을 제시합니다.

- **Technical Details**: 커스텀 트랜스포머 모델을 훈련하고 몇 가지 사전 훈련된(pretrained) 모델을 미세 조정(fine-tune)하여, 총 6개의 엔티티 그룹에 대해 F1 점수 92.11을 달성했습니다. 이 모델은 자연어 처리의 핵심 작업 중 하나인 개체 인식(NER)의 성능을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: 이 논문에 제시된 모델은 서로 다른 언어 간의 엔티티 그룹 및 태그 이름의 불일치를 크게 줄이는 데 기여할 것으로 기대됩니다. 다국어 NER 모델의 도입은 인도의 다양한 언어 환경에서 더욱 효과적으로 활용될 수 있는 가능성을 제공합니다.



### MAGA: MAssive Genre-Audience Reformulation to Pretraining Corpus Expansion (https://arxiv.org/abs/2502.04235)
Comments:
          Dataset released url this https URL

- **What's New**: 본 연구에서는 기존 말뭉치에서 다양한 맥락적 요소를 포함한 고품질의 프리트레인 데이터(pretraining data)를 생성하기 위해, 	extbf{MAGA} (MAssive Genre-Audience) 재구성 방법을 제안합니다. 이 방법은 770B 토큰에 달하는 MAGACorpus를 구축하여 모델 크기(134M-13B)에 따라 일관된 향상을 보여줍니다. 또한, 새로운 신Synthetic 데이터 생성 기술을 통해 언어 모델의 한계를 극복하는 데 필수적인 요소임을 입증합니다.

- **Technical Details**: MAGA는 기존 고품질 텍스트 컬렉션을 기반으로 770억 개의 토큰을 포함한 MAGACorpus를 구축하고, 원래 말뭉치와 비교할 때 다양한 모델 크기에서 우수한 성능을 나타냅니다. 이 기술은 3.3B MoE 모델을 활용하여 원시 문서만을 입력으로 사용하며, 데이터셋을 적응적으로 확대하는 경량적이고 확장 가능한 접근 방식을 채택하고 있습니다. 또한, 프롬프트 엔지니어링(prompt engineering)이 합성 훈련 붕괴(synthetic training collapse)에 미치는 영향을 분석하고, 기존 검증 손실(validation loss) 메트릭의 한계를 밝혀냅니다.

- **Performance Highlights**: MAGA는 데이터 반복(data repetition) 및 업샘플링(upsampling) 전략에 비해 일관된 성능 개선을 보여주었으며, 다음 세대 언어 모델 훈련에 있어 합성 데이터의 필요성을 입증합니다. MAGA 방법론을 통해 가볍고 효율적인 데이터 생성이 가능해졌으며, 훈련 데이터셋을 질을 유지하면서도 대규모로 확장할 수 있는 신뢰할 수 있는 경로를 제공합니다. 이 연구는 다양한 장르와 청중 대상을 위한 유형을 조합하여 3.9배의 토큰 확장을 가능하게 했습니다.



### A Classification System Approach in Predicting Chinese Censorship (https://arxiv.org/abs/2502.04234)
- **What's New**: 이번 논문은 중국 인터넷에서 소셜 미디어인 Weibo 게시물이 검열될지 여부를 예측하기 위해 분류기(classifier)를 사용하는 연구입니다. 연구는 랜덤 샘플링을 통해 정제된 중국어 구문 데이터셋을 구축하고, 이 데이터셋에서 바이너리 검열 마킹을 이용해 4개의 로지스틱 회귀 모델을 분류하는 데 활용되었습니다. 또한, 사전 훈련된 트랜스포머(transformer) 모델을 사용해 유사한 분류 작업을 수행하여 Fined-Tuned BERT 모델이 다른 방법들보다 우수한 성능을 보였음을 밝혔습니다.

- **Technical Details**: 중국어 언어의 특성상, 의미 있는 구문을 토큰화하기 위해 Jieba라는 검증된 NLP 라이브러리를 사용하였습니다. 논문에서는 TF-IDF 접근법에서 영감을 받아 4가지 정보 검색기(information retriever)를 훈련 데이터에 적용하였으며, 이를 통해 각각의 특성 벡터(feature vector)를 추출하였습니다. 로지스틱 회귀 모델에서 가장 좋은 성능을 나타낸 Fined-Tuned BERT 모델의 활용이 강조되었고, 이를 통해 NLP 기법을 통해 검열 라벨 시스템의 가능성을 탐구하고 있습니다.

- **Performance Highlights**: 평가 지표로는 매크로 F1 점수와 ROC-AUC를 사용하였으며, Fined-Tuned BERT 모델이 다른 분류 방법보다 우수한 성능을 기록했습니다. 또한, 2012년 Weibo의 검열 키워드 리스트를 활용하여 모델링을 진행하였고, 모델링 결과 검열되지 않은 데이터에서의 불균형도 고려하여 3%인 검열 데이터를 포함한 전체 데이터의 효과적으로 분석했습니다. 이 연구는 Weibo에서의 검열 이해를 더 깊이 있게 할 것으로 기대됩니다.



### Sports and Women's Sports: Gender Bias in Text Generation with Olympic Data (https://arxiv.org/abs/2502.04218)
Comments:
          NAACL 2025

- **What's New**: 이번 연구에서는 올림픽 경기의 남녀 이벤트 데이터를 활용하여 언어 모델에서의 성별 편향을 조사합니다. 이전 연구에서 보여진 성별 편향이 문맥에 따라 어떻게 나타나는지를 정량적으로 분석하는 새로운 방법론을 제시합니다. 연구팀은 성별 모호성이 있는 프롬프트에서 여성을 차별하는 경향을 발견하며, 이는 LLM의 편향적인 성향을 드러냅니다.

- **Technical Details**: 연구는 올림픽 게임의 1988년부터 2021년까지의 데이터를 활용하여 LLM의 성별 편향을 측정하기 위한 세 가지 메트릭스를 설정합니다. 이들은 성별이 명확하게 명시된 경우와 모호한 경우로 나뉘어 있으며, 각 경우에서 생성된 텍스트를 평가하여 성별에 대한 편향을 수치화합니다. 모델 비교는 닫힌 및 개방형 모델Weights를 포함하여 다양한 LLM에 대해 수행됩니다.

- **Performance Highlights**: 결과에 따르면, 모든 모델이 일반적으로 성별 편향을 보였으며, 이 편향의 양상은 모델에 따라 다르게 나타났습니다. 성별이 불분명한 프롬프트에서 남성 결과를 우선적으로 찾는 경향을 확인하게 되었고, 이 연구는 앞으로 LLM의 사용 시 성별 관련 편향을 어떻게 줄일 수 있는지에 대한 중요한 통찰력을 제공합니다.



### The Best Instruction-Tuning Data are Those That F (https://arxiv.org/abs/2502.04194)
- **What's New**: 이번 논문에서는 GRAPE라는 새로운 SFT(Supervised Fine-Tuning) 프레임워크를 제안합니다. 이는 타겟 모델의 사전 훈련 분포에 가장 가까운 응답을 선택하여, 데이터 수집 과정에서 발생할 수 있는 성능 저하 문제를 해결하는 데 중점을 둡니다. GRAPE는 다양한 LLM(Language Model)에서 응답을 수집하고, 그 중 타겟 모델에 대해 높은 확률을 보이는 응답을 선택하여 데이터의 질을 향상시킵니다.

- **Technical Details**: GRAPE는 여러 모델에서 수집된 응답 중 타겟 모델과 가장 유사한 응답을 선택하여 SFT를 진행합니다. 이 과정은 타겟 모델의 확률을 이용해 이루어지며, 기존의 일률적인 응답 대신 모델에 적합한 응답을 사용합니다. 특히, GRAPE의 접근 방식은 데이터 분포의 이동에 따른 문제를 최소화하여 성능 향상을 도모합니다.

- **Performance Highlights**: GRAPE는 LLaMA3.1-8B, Mistral-7B 및 Qwen2.5-7B와 같은 일반적으로 사용되는 LLM에서 테스트되었으며, 기존 베이스라인보다 최대 17.3%의 성능 향상을 기록했습니다. 또한, GRAPE 선택 데이터를 사용하여 Tulu3 및 Olmo2에 대한 후속 데이터에서도 강력한 성능 개선을 보여주었습니다. GRAPE는 데이터의 양을 줄이고 학습 에폭수를 절반으로 줄여도 높은 성능을 유지함으로써 SFT 과정에서 높은 효율을 입증했습니다.



### Lexical Substitution is not Synonym Substitution: On the Importance of Producing Contextually Relevant Word Substitutes (https://arxiv.org/abs/2502.04173)
Comments:
          Accepted to ICAART 2025

- **What's New**: 본 논문에서는 Lexical Substitution(LS) 기술을 통해 문장에서 특정 단어를 유사한 단어로 교체하는 방법을 제시합니다. 특히, ConCat이라는 새로운 방법론을 소개하여, 원래 문장을 사용하여 모델에 전달되는 컨텍스트 정보를 증대시키는 접근 방식을 채택하였습니다. 기존 방법과 비교하여, 이 방법은 문맥적으로 관련 있는 예측을 생성하는 데 매우 효과적임을 입증합니다.

- **Technical Details**: ConCat은 원래 문장과 마스킹된(target word masked) 문장을 결합하여 문맥과 의미 사이의 균형을 개선합니다. 이 방법은 BERT와 XL-Net 모델을 실험하여 RoBERTa 모델에 최적화되었으며, 하이브리드 접근을 통해 WordNet을 사용하여 생성된 후보 단어 중 부적절한 단어를 필터링하는 절차를 포함합니다. 이러한 방법은 데이터 처리의 효율성을 높이면서도, 문맥적으로 적합한 대체 단어를 제공하게 됩니다.

- **Performance Highlights**: 정량적 및 정성적 평가를 통해 ConCat의 성능을 LS07, CoInCo, Swords라는 세 가지 표준 벤치마크 데이터셋에서 평가하였습니다. 연구 결과, 사용자들은 ConCat로 생성된 대체 단어를 골드 대체 단어보다 선호하는 경향을 보였으며, 이는 인간 평가의 중요성을 강조합니다. 마지막으로, LS와 관련한 기존 평가 방법의 잠재적 문제점에 대한 비판적 논의가 진행되었으며, ConCat이 텍스트의 의미적 유용성을 얼마나 잘 보존하는지를 조사하여, 문장 의미를 유지하며 효과적인 성과를 달성하는 것으로 나타났습니다.



### UltraIF: Advancing Instruction Following from the Wild (https://arxiv.org/abs/2502.04153)
- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)의 복잡한 지시사항을 따를 수 있는 능력을 향상시키기 위한 새로운 접근법인 UltraIF를 제안합니다. Open-source(오픈 소스) 데이터를 활용하여 LLM을 설계함으로써, 공개된 데이터와 대기업에 의해 훈련된 모델 간의 격차를 해소하고자 합니다.

- **Technical Details**: UltraIF는 실제 사용자 요청을 간단한 쿼리와 제약조건, 그리고 그 제약조건에 대한 평가 질문으로 분해합니다. 이후 UltraComposer라는 모델을 훈련시켜 제약조건과 관련된 프롬프트를 평가 질문과 함께 구성합니다. 이를 통해 복잡한 지시사항을 합성하고, 평가 질문을 이용해 응답을 필터링할 수 있습니다.

- **Performance Highlights**: 실험 결과, UltraIF는 LLaMA-3.1-8B-Base 모델을 성공적으로 조정하여, 5가지 지시사항 처리 기준에서 인스트럭트 버전과 비교해 동등한 성능을 발휘했습니다. 또한, UltraIF는 LLaMA-3.1-8B-Instruct 모델을 self-alignment를 통해 더욱 개선할 수 있음을 보여주었습니다. 이 접근법은 다양한 사용 사례에 대한 가능성을 제시합니다.



### The Order Effect: Investigating Prompt Sensitivity in Closed-Source LLMs (https://arxiv.org/abs/2502.04134)
Comments:
          The first 3 authors have contributed equally

- **What's New**: 최근 대규모 언어 모델(LLMs)의 신뢰성을 보장하는 것이 중요해지고 있습니다. 이 논문에서는 LLM의 입력 순서에 대한 민감도를 다루며, 다중 선택 질문과 같은 다양한 작업에서 수행한 실험 결과를 제공합니다. 매우 미세한 입력의 변화가 성능에 중대한 영향을 미친다는 점을 강조하고 있으며, LLM의 신뢰성을 높이기 위한 추가 연구가 필요하다는 결론을 내리고 있습니다.

- **Technical Details**: 이 연구는 LLM의 입력 형식에 대한 민감성과 순서 의존성이 어떤 영향을 미치는지 분석합니다. 특히 사용된 모델은 비공식 모델인 GPT 시리즈이며, 동일한 입력에 대한 응답이 입력 순서에 따라 달라질 수 있다는 것을 보여주고자 합니다. 기존의 연구들에서 논의된 내용들과는 달리, 모델의 응답 일관성 문제를 통합적으로 다루어 새로운 시각을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 입력의 순서 변화가 전반적인 성능에 긍정적인 영향을 미치지 않는 것으로 나타났습니다. 몇 가지 예외를 제외하고는, 충분한 프롬프트 제공에도 불구하고 여전히 응답의 안정성이 결여된 것을 확인했습니다. 이러한 결과는 임상적 결정이나 중요한 비즈니스 결정에서 LLM을 사용할 경우 큰 위험 요소가 될 수 있음을 시사합니다.



### LLMs to Support a Domain Specific Knowledge Assistan (https://arxiv.org/abs/2502.04095)
- **What's New**: 이 연구는 지속 가능성 보고를 위한 도메인 특화 지식 어시스턴트 개발에 대한 새로운 접근법을 제시합니다. 국제 재무 보고 기준(IFRS)를 기반으로 한 최초의 고품질 합성 질문-답변(QA) 데이터셋을 생성하여 기업의 IFRS 보고 지원을 위한 기초를 마련하였습니다. 이는 1,063개의 다양한 QA 쌍으로 구성되어 지속 가능성 보고에서의 여러 사용자 쿼리를 충족합니다.

- **Technical Details**: 이 프로젝트에서는 두 가지 아키텍처를 사용하여 지속 가능성 보고 영역에서 질문-답변 시스템을 설계하였습니다. RAG(검색 증강 생성) 파이프라인과 완전 LLM 기반 파이프라인이 포함되며, 두 아키텍처 모두 QA 데이터셋에 대해 실험하고 미세 조정을 거쳐 개발되었습니다. 최종 파이프라인은 도메인 특화 데이터를 기반으로 미세 조정된 LLM과 복잡한 쿼리를 처리하기 위한 산업 분류 기능을 포함하고 있습니다.

- **Performance Highlights**: RAG 아키텍처는 단일 산업 질문에서 85.32%, 교차 산업 질문에서 72.15%의 정확도를 달성하며 기준 방법론 대비 각각 4.67 및 19.21 퍼센트 포인트 향상되었습니다. 또한 LLM 기반 파이프라인은 단일 산업 질문에서 93.45%, 교차 산업 질문에서 80.30%의 정확도를 기록, 기준 대비 각각 12.80 및 27.36 퍼센트 포인트 개선을 보였습니다.



### AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inferenc (https://arxiv.org/abs/2502.04077)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 인해 Key-Value(KV) 캐시 압축을 통한 효율적인 추론이 주목받고 있습니다. 하지만 기존 방법들은 주로 휴리스틱 순위를 사용하여 중요 KV 토큰을 파악하는 데 한계가 있어 LLM 성능 저하를 초래합니다. 이에 본 연구에서는 AttentionPredictor라는 학습 기반의 중요 토큰 식별 접근법을 제안하고 있습니다.

- **Technical Details**: AttentionPredictor는 경량의 합성곱 모델을 활용하여 시공간 패턴(spatiotemporal patterns)을 포착하고 다음 토큰의 attention score를 예측합니다. 이 방법은 기존의 휴리스틱 방식으로는 캡처하기 어려운 동적인 시간 패턴을 학습 기반으로 해결하는 데 초점을 맞추고 있습니다. 알고리즘의 메모리 소비가 거의 없으면서도 정확한 예측을 제공하는 점이 주요 특징입니다.

- **Performance Highlights**: 제안된 방법은 LongBench 데이터셋에서 16배의 KV 캐시 압축을 달성하면서 기존 최고 성능보다 41% 향상된 결과를 보여주었습니다. 또한 32K 컨텍스트에서의 지연 시간을 1.4배 단축시킬 수 있었습니다. 이러한 성과는 AttentionPredictor의 정확한 중요한 토큰 식별 덕분으로, LLM 성능을 유지하며 효율적인 캐시 관리가 가능하다는 것을 시사합니다.



### Controllable Emotion Generation with Emotion Vectors (https://arxiv.org/abs/2502.04075)
Comments:
          15 pages, 5 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)을 기반으로 한 기술들이 고객 서비스, 콘텐츠 생성 및 체화된 지능 분야에서 큰 발전을 이루었으나, 감정을 적절한 톤과 타이밍으로 표현하는 능력은 아직 부족하다. 본 연구에서는 감정 표현을 제어할 수 있는 방법을 제안하며, 이는 유니버설하고 유연하며 효과적으로 검증되었다. 제안한 방법은 지능형 고객 서비스와 문학적 창작, 가정용 로봇과 같은 감정을 필요로 하는 분야에서 널리 응용될 수 있다.

- **Technical Details**: 연구에서는 감정 벡터(Emotion Vector)를 사용하여 언어 모델의 출력에서 감정 톤을 조정하는 두 단계 방법론을 제안한다. EmotionQuery라는 전문화된 데이터셋을 구축하여 5가지 기본 감정 상태에 따라 500개의 쿼리를 생성하였다. 모델은 감정 조건 없이 중립적인 설정과 특정 감정을 반영하는 감정 설정에서 각각 반응을 생성하며, 이 차이를 통해 감정 벡터를 정의하는 방식으로 진행된다.

- **Performance Highlights**: 제안한 방법은 다양한 LLM 아키텍처에서 많은 실험을 통해 효과성을 입증하였으며, 기존 방법이 특정 모델 또는 훈련 세트에 얽매여 있는 한계를 극복하였다. 연구 결과는 LLM이 기본적으로 감정을 표현할 수 있는 능력을 갖고 있음을 보여주며, 이를 통해 모델의 출력 품질을 높일 수 있는 가능성을 제시하였다. 감정 표현 능력의 향상은 고객 서비스 및 정신 건강 관리와 같은 여러 분야에서 사용자 경험을 크게 개선할 수 있는 잠재력을 가지고 있다.



### Predicting Large Language Model Capabilities on Closed-Book QA Tasks Using Only Information Available Prior to Training (https://arxiv.org/abs/2502.04066)
- **What's New**: 이 논문에서는 OpenAI의 GPT-4 기술 보고서를 기반으로 특정 작업에 대한 모델의 성능을 훈련 전에 예측할 수 있다는 새로운 접근 방식을 제시합니다. 이 접근 방식은 리소스 할당을 최적화하고 데이터가 목표 작업과 일치하도록 보장하는 데 중요한 역할을 합니다. 특히, 약 1.5조 토큰의 데이터를 사용하여 세 가지 대형 언어 모델(1.6B, 7B, 13B)을 사전 훈련하며, 사전 훈련 데이터와 지식 보유, 작업별 지식 보유를 예측하는 데 집중합니다.

- **Technical Details**: 이 연구는 560,000달러와 520,000 GPU 시간을 투입하여 세 가지 대형 언어 모델을 사전 훈련했습니다. 사전 훈련 데이터 분석을 위해 지식 트리플(knowledge triples)을 사용하고, 모델의 지식 보유를 평가하는 데 SMI 지표(Size-dependent Mutual Information)를 도입합니다. 이 지표는 사전 훈련 데이터, 모델 크기, 작업별 지식 보유 간의 관계를 정량화하며, 여러 크기의 모델 기반으로 ACC와 SMI 간의 강한 선형 상관관계를 발견했습니다.

- **Performance Highlights**: 실험 결과 SMI 지표와 다양한 크기의 모델의 CBQA 작업에서의 정확도 간에 강한 선형 상관관계가 있음을 확인했습니다. Coefficient of Determination(R²) 값은 0.84를 초과하여 모형이 작업 특정 지식을 얼마나 잘 보유하고 있는지를 효과적으로 예측하는 데 매우 유용함을 시사합니다. 또한, 연구진은 1.6B 규모의 모델에 대한 사전 훈련 데이터와 가중치를 공개하여 후속 연구에 기여하고 있습니다.



### Simulating the Emergence of Differential Case Marking with Communicating Neural-Network Agents (https://arxiv.org/abs/2502.04038)
- **What's New**: 이번 연구는 Differential Case Marking (DCM)이라는 현상을 인공지능 에이전트의 상호작용을 통해 재현하려고 하며, DCM의 발생에 있어 의사소통의 중요성을 강조하고 있습니다. 이전 연구인 Smith & Culbertson (2020)의 결과를 바탕으로, 언어 학습만으로는 DCM이 발생하지 않지만, 에이전트 간의 의사소통이 이루어질 때 DCM이 나타난다는 점을 밝혔습니다. 더욱이, 이번 연구는 신경망 기반의 시뮬레이션 모델이 언어 진화에 관한 실험적 연구를 보완하는 가능성을 보여줍니다.

- **Technical Details**:  연구에서는 Neural agent Language Learning and Communication (NeLLCom) 프레임워크를 사용하여 에이전트가 인공 언어를 먼저 학습한 후 의사소통을 하도록 설정하였습니다. 이 과정에서 기존 언어 체계와의 비교를 통해 DCM 발생에 있어서 학습과 의사소통의 효과를 전체적으로 평가했습니다. 다소 일반적인 통신 최적화 알고리즘과 신경망 학습자를 사용하였으며, 그들은 언어 또는 의미적 선호에 대한 사전 경험이 없는 상태에서 학습을 진행했습니다.

- **Performance Highlights**:  연구 결과, DCM 발생의 중요한 요인은 에이전트 간의 의사소통으로 확인되었으며, 이는 통신 최적화 과정에서 나타났습니다. 특히, 에이전트의 상호작용이 DCM의 발생을 촉진하며, 이 과정에서 언어의 효율성이 극대화된다는 것을 보여주었습니다. 이러한 결과는 언어가 어떻게 발전하고 변화할 수 있는지를 이해하는 데 중요한 단서가 될 것입니다.



### Exploring Imbalanced Annotations for Effective In-Context Learning (https://arxiv.org/abs/2502.04037)
- **What's New**: 본 연구는 불균형(class imbalance) 클래스 분포가 주석 데이터셋의 In-context learning (ICL) 수행에 미치는 영향을 처음으로 연구했습니다. 실험을 통해 기존의 재조정(rebalance) 방법이 ICL의 불균형 문제를 완화하지 못하며, 오히려 문제를 악화시킬 수 있음을 발견하였습니다. 이러한 배경에서 저자들은 클래스 가중치(class weighting)를 이용하여 원래의 점수 함수(original scoring functions)를 수정하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 저자들은 주석 데이터셋과 테스트 데이터셋 간의 분포 차이를 두 가지 구성 요소인 클래스별 가중치(class-wise weights)와 조건적 바이어스(conditional bias)로 나누어 설명합니다. 이들은 균형 잡힌 검증 데이터셋에서의 실험적 오류를 최소화하여 조건적 바이어스를 추정하고, 이를 통해 원래의 점수 함수를 수정하여 ICL의 성능을 개선할 수 있습니다. 저자들은 효과적 수(number)와 같은 클래스별 가중치를 사용하고, 조건적 바이어스를 추정하는 방식을 채택하였습니다.

- **Performance Highlights**: 저자들은 Amazon, AgNews, Yelp 등 7개의 다양한 데이터셋에서 extensive한 평가를 통해 제안한 방법의 효과성을 입증하였습니다. 예를 들어, 100 비율의 불균형 데이터셋에서 ICL의 테스트 정확도가 37.83%에서 43.29%로 증가하여 무려 5.46%의 개선을 보였습니다. 이러한 성능 향상은 생성(generation) 작업에서도 ICL의 성능을 개선하는 데 유효함을 확인하였습니다.



### Quantification of Biodiversity from Historical Survey Text with LLM-based Best-Worst Scaling (https://arxiv.org/abs/2502.04022)
Comments:
          NoDaLiDa 2025, EcoNLP Workshop

- **What's New**: 이번 연구에서는 역사적 조사 텍스트에서 종의 빈도를 결정하는 방법을 평가하고, 이를 분류 작업으로 형식화하고 최종적으로 Best-Worst Scaling (BWS)을 사용한 회귀 작업으로 프레임화할 수 있음을 보여줍니다. 연구진은 Ministral-8B, DeepSeek-V3, GPT-4와 같은 대규모 언어 모델(LLMs)을 테스트하여 후자의 두 모델이 사람들과의 합의가 비교적 잘 이루어진다는 결과를 도출했습니다.

- **Technical Details**: 주된 작업은 특정 지역에서 동물 종의 발생 빈도를 나타내는 수량 레이블을 텍스트에 할당하는 것입니다. 이 작업은 이진 분류, 7개 클래스의 다중 클래스 설정 및 연속 값으로 스케일링하는 방법으로 수행됩니다. 연구진은 이진 분류 작업의 유효성을 테스트하기 위해 BERT, 로지스틱 회귀, SVM 및 랜덤 포레스트 모델을 사용하여 성능을 평가했습니다.

- **Performance Highlights**: 모델 훈련 결과, F1-macro 점수가 90대 초반에 도달하는 데 몇 백 개의 텍스트만 필요하며, 전체 데이터셋에서 모델은 일반적인 텍스트에 대해 높은 정확도를 보였습니다. 특히, LaBSE 모델이 더 나은 성능을 보였고, GPT-4와 DeepSeek-V3의 제로샷 분류 실험 결과는 각기 다른 강점을 보여주었습니다. 최종적으로, 연구진은 회귀 문제로의 전환을 통해 수량 추정 문제를 보다 일반적인 카테고리로 확장할 수 있는 가능성을 모색하고 있습니다.



### Ontology-Guided, Hybrid Prompt Learning for Generalization in Knowledge Graph Question Answering (https://arxiv.org/abs/2502.03992)
Comments:
          Accepted By ICSC 2025

- **What's New**: OntoSCPrompt는 여러 Knowledge Graph(KG) 간의 일반화 능력을 향상시키기 위한 혁신적인 두 단계 구조의 KGQA 시스템입니다. 첫 번째 단계에서는 특정 KG에 의존하지 않는 SPARQL 쿼리 구조를 생성하고, 두 번째 단계에서는 KG 특유의 정보를 채워 넣습니다. 또한, 온톨로지 기반의 하이브리드 프롬프트 학습 전략을 통해 KG에 대한 이해도를 증대시킵니다.

- **Technical Details**: KGQA 시스템은 질문을 SPARQL 쿼리 구조로 변환하는 두 가지 단계로 구성됩니다. 첫 번째 단계는 일반적인 SPARQL 구조를 예측하는 것이고, 두 번째 단계는 특정 KG에 대한 개념, 관계 및 엔티티로 이 구조를 채우는 것입니다. 이 과정에서, 다양한 KG의 이질성을 처리하기 위해 개념 및 복잡한 SPARQL 절에 대한 새로운 자리 표시자를 추가하였습니다.

- **Performance Highlights**: OntoSCPrompt는 CWQ, WebQSP 및 LC-QuAD 1.0과 같은 다양한 KGQA 데이터 세트에서 SOTA 접근 방식과 동등한 성과를 냈습니다. 실험 결과는 리트레이닝 없이도 효율적으로 작동하며, DBLP-QuAD 및 CoyPu KG와 같은 보지 못한 도메인 특정 KG에 대해서도 잘 일반화된다는 것을 보여줍니다.



### PGB: One-Shot Pruning for BERT via Weight Grouping and Permutation (https://arxiv.org/abs/2502.03984)
- **What's New**: 이 논문에서는 BERT의 비효율적인 구조를 개선하기 위해 새로운 반구조화 일회성 가지치기 방법인 'Permutation and Grouping for BERT' (PGB)를 제안합니다. PGB는 중요 그룹을 식별하고, 불필요한 가중치를 제거하여 높은 압축 효율성과 희소성을 유지하면서 정확성을 보존합니다. 이 방법은 기존의 반복 가지치기 및 지식 증류(knowledge distillation) 기법보다 간단하고 계산 비용이 적습니다.

- **Technical Details**: PGB 접근 방식은 BERT의 다중 헤드 어텐션(multi-head attention)과 피드포워드 레이어(feed-forward layers)에서 작동하여 개별 가중치의 중요 그룹을 구성하고, 중요하지 않은 모든 가중치를 구조적으로 가지치기합니다. 중요한 그룹이 형성되지 않은 경우 해당 레이어를 통째로 삭제하여 모델을 더욱 압축할 수 있습니다. 논문에서는 GLUE와 SQuAD 벤치마크에서 PGB를 BERT_BASE에 적용한 실험 결과를 통해 기법의 효과성을 입증하였습니다.

- **Performance Highlights**: PGB는 기존의 고급 구조적 가지치기 기법에 비해 계산 비용과 정확도 유지 측면에서 우수한 성능을 나타냈습니다. 실험 결과 PGB는 BERT_BASE 모델을 사용하여 효율성과 정확성 모두에서 최신 기술(SOTA) 가지치기 방법을 능가함을 보여주었습니다. 이를 통해 PGB는 자연어 처리(NLP) 작업에서 더 작은 모델을 보다 효과적으로 사용할 수 있는 가능성을 제시합니다.



### MAQInstruct: Instruction-based Unified Event Relation Extraction (https://arxiv.org/abs/2502.03954)
Comments:
          Accepted by WWW 2025 short

- **What's New**: 이 논문에서는 MAQInstruct라는 개선된 Instruction-based Event Relation Extraction 프레임워크를 제안합니다. 기존의 방법들이 다중 클래스 분류, MASK 예측, 또는 프로토타입 매칭에 기반해 있었던 것과 달리, MAQInstruct는 주어진 event-relation 지침을 사용하여 이벤트를 선택하는 방식으로 작업을 전환합니다. 이러한 접근은 추론에 필요한 샘플 수를 대폭 줄이면서 성능을 향상시키는 데 기여합니다.

- **Technical Details**: MAQInstruct 프레임워크는 bipartite matching loss를 포함하여 기존의 instruction-based 방법의 생성 순서에 대한 의존성을 줄이는 데 초점을 맞추고 있습니다. 흥미롭게도, 이 프레임워크는 순서의 변화가 이벤트 관계에 미치는 영향을 최소화하여 여러 대규모 언어 모델(LLMs)에서의 성능을 향상시킵니다. 구체적으로, 모델은 이벤트 관계 유형이 event mentions의 수보다 현저히 적기 때문에 훈련 및 추론 샘플을 k×n으로 감소시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 MAQInstruct는 다양한 LLM에서 이벤트 관계 추출 작업을 획기적으로 개선하고 있습니다. 특히, 이 방법은 대규모 언어 모델의 능력을 활용하여 이전의 분류 기반 방법을 초월하는 성능을 보여주었습니다. 또한, bipartite matching loss를 통해 사건 관계 추출 작업을 수행할 때 적절성을 높이고 올바른 답변을 더 잘 생성하도록 돕습니다.



### Afrispeech-Dialog: A Benchmark Dataset for Spontaneous English Conversations in Healthcare and Beyond (https://arxiv.org/abs/2502.03945)
Comments:
          19 pages, 5 figures

- **What's New**: 본 연구에서는 Afrispeech-Dialog라는 새로운 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 아프리카 억양의 영어로 이루어진 50개의 의료 및 비의료 대화로 구성되어 있으며, 자동 음성 인식(ASR)과 관련 기술을 평가하기 위해 설계되었습니다. 연구진은 상태-of-the-art(SOTA) 스피커 다이얼리제이션과 ASR 시스템을 아프리카 억양 대화에서 평가하여, 원주율 억양과 비교해 10% 이상의 성능 저하를 발견하였습니다.

- **Technical Details**: Afrispeech-Dialog 데이터셋은 아프리카 의료 및 비의료 대화의 7시간 분량이 포함되어 있습니다. 연구팀은 다양한 ASR 시스템과 LLM 기반 요약 기법들의 성능을 벤치마크하여 이 시스템들이 아프리카 억양 대화에 어떻게 적용될 수 있는지를 평가했습니다. 또한, 의료 대화 요약에 대한 대규모 언어 모델(LLM)의 능력을 탐구하고, ASR 오류가 선행된 의류 초록에 미치는 영향을 실험적으로 분석했습니다.

- **Performance Highlights**: 연구 결과, 전반적인 ASR 성능은 기존의 원주율 억양에 비해 유의하게 저하되었습니다. 이러한 결과는 아프리카 억양이 의료 분야의 음성 인식 기술에서 가장 큰 도전 과제가 되고 있음을 시사합니다. 데이터셋의 개발은 다양한 언어적 배경을 가진 대화형 AI 기술의 발전을 위한 중요한 초석이 될 것입니다.



### Experiments with Large Language Models on Retrieval-Augmented Generation for Closed-Source Simulation Softwar (https://arxiv.org/abs/2502.03916)
Comments:
          11 pages, 6 tables

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 접근법을 통해 클로즈드 소스 시뮬레이션 소프트웨어에 LLM(대형 언어 모델)의 응용 가능성을 탐구합니다. RAG 시스템을 통해 LLM의 응답 작성 시 사용자 요청에 따라 관련 문서를 제공하여, 지식 집약적인 작업에 필요한 정보를 지원할 수 있습니다. 이는 기존의 오픈소스 데이터에 교육받지 않은 클로즈드 소스 환경에서도 LLM의 능력을 확장할 가능성을 보여줍니다.

- **Technical Details**: 스무디드 입자 유체역학(SPH)과 같은 시뮬레이션 방법론을 사용하여 LLM의 지식 수준을 테스트합니다. 연구는 Pasimodo라는 클로즈드 소스 소프트웨어에 대한 LLM의 이해도를 평가하며, 관련 내부 지식을 제공한 후 RAG 시스템을 통해 모델을 생성하는 방법을 다룹니다. 이 과정에서 RAG는 LLM이 처리할 수 있는 정보의 품질을 향상시키고, 정보 격차를 줄이는 데 도움이 되는 것으로 나타났습니다.

- **Performance Highlights**: 제시된 초기 실험 결과는 RAG 시스템이 클로즈드 소스 시뮬레이션 소프트웨어에 효과적으로 정보를 접근할 수 있도록 돕는 가능성을 보여줍니다. 다수의 실험 예제를 통해 LLM의 응답에서 나타난 정보 격차와 데이터 불완전성을 발견하였으며, 이러한 결과는 향후 추가 연구의 필요성을 강조합니다. 기능적으로, RAG 접근법은 LLM의 환각(hallucination) 위험을 감소시키는 잠재력을 보입니다.



### BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation (https://arxiv.org/abs/2502.03860)
Comments:
          36 pages

- **What's New**: 이 논문은 최근 OpenAI의 o1 모델이 혁신적으로 나타낸 LongCoT(긴 사고의 연쇄) 기능을 체계적으로 개발할 수 있는 새로운 접근 방식을 제시합니다. 특히, 기존 LongCoT 모델이나 비싼 인력 주석 없이도 LLM(대형 언어 모델)의 LongCoT 능력을 개발할 수 있는 방식인 Bootstrapping LongCoT (BOLT)를 도입합니다. 이 BOLT 방법은 몇 가지 단계로 구성되어 있으며, 10개의 예제를 통해 실제로 가능한 접근 방식을 입증하였습니다.

- **Technical Details**: BOLT 방법은 세 가지 주요 단계로 나뉩니다: 첫째, ShortCoT LLM(짧은 사고의 연쇄 모델)을 활용한 LongCoT 데이터 부트스트래핑, 둘째, LongCoT 감독 세부 조정, 셋째, 온라인 교육을 통한 LongCoT 능력 향상입니다. 이를 통해 Llama-3.1-70B-Instruct 모델을 사용하여 다양한 모델 스케일(7B, 8B, 70B)에 적용하였으며, 실험에서 강력한 성능을 보여주었습니다. 이 연구는 효율성과 실용성을 증명하여 대규모 데이터 수집 없이 LongCoT를 개발할 수 있도록 합니다.

- **Performance Highlights**: BOLT 방법은 Arena-Hard, MT-Bench, WildBench, ZebraLogic, MATH500와 같은 다양한 벤치마크에서 뛰어난 성과를 거두었습니다. 이 벤치마크들은 정보 탐색, 창의적 글쓰기, 코딩, 계획 및 수학 문제 해결을 포함한 여러 분야에서 모델의 사고 및 작업 해결 능력을 평가합니다. BOLT는 기존의 블랙박스 접근 방식과 달리 투명한 화이트박스 접근 방식인 점에서 주목할 만하며, 앞으로도 연구자들을 위해 훈련 데이터와 모델을 오픈 소스화할 계획입니다.



### Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis (https://arxiv.org/abs/2502.03843)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 정렬을 위한 고품질의 대규모 지침이 얼마나 중요한지를 강조합니다. 기존의 자연어 이해(NLU) 지침 구축 작업은 정보 추출(IE) 중심으로 진행되어, 기계 독해, 질문 응답, 텍스트 분류와 같은 과제를 간과했습니다. 이로 인해 NLU 데이터의 다양성이 떨어지고 훈련된 LLM의 일반화 능력이 감소하는 문제가 발생했습니다. 이를 해결하기 위해, 'Hum'이라는 고품질의 합성 지침 데이터셋을 제안합니다.

- **Technical Details**: Hum은 정보 추출(IE), 기계 독해, 텍스트 분류, 그리고 지침 일반화 과제를 포함하여 다양한 NLU 작업을 위한 지침을 수집하도록 설계되었습니다. 이 데이터셋은 인간-LLMs 협업 메커니즘을 통해 지침 다양성을 더욱 풍부하게 만드는 방식으로 지침을 합성합니다. 연구는 5개의 NLU 작업과 28개의 LLM 일반 능력 평가 데이터셋에서 수행되어, Hum의 효과를 입증했습니다.

- **Performance Highlights**: 실험 결과, Hum은 여섯 개의 LLM의 NLU 능력을 평균 3.1% 향상시키는데 기여했습니다. 다른 일반 능력에는 유의미한 감소가 없었으며, 이는 Hum의 효과적인 지침 차원에서 각 작업의 다양성을 크게 증가시켰음을 보여줍니다. 이 연구는 NLU 작업의 품질 향상을 위한 중요한 진전을 의미합니다.



### A comprehensive survey of contemporary Arabic sentiment analysis: Methods, Challenges, and Future Directions (https://arxiv.org/abs/2502.03827)
Comments:
          Paper accepted to NAACL 2025

- **What's New**: 이 논문은 아랍어 감정 분석(Arabic Sentiment Analysis, ASA)에 대한 최신 연구 동향을 체계적으로 정리하며, 기존 문헌의 한계와 과제를 규명합니다. 특히 딥 러닝(Deep Learning)을 활용한 방법론에 중점을 두고 아랍어 감정 분석의 연구 격차를 일반 감정 분석과 비교하여 강조하고 있습니다. 또한, 향후 연구를 위한 세부적인 방향성을 제시합니다.

- **Technical Details**: 논문에서는 아랍어 감정 분석의 발전 과정을 전통적인 렉시콘 기반 방법(Lexicon-based methods)에서 딥 러닝 기반 방법(Deep Learning-based methods)으로 설명합니다. 또한, 감정 점수(Sentiment Scores)와 단어의 의미를 제공하는 패턴을 학습할 수 있는 기계 학습 방법(Machine Learning Methods)의 소개가 이루어지며, 다양한 피처 엔지니어링(Feature Engineering) 방법이 언급됩니다. 이 섹션에서는 감정 분석 모델의 적용에 있어 아랍어 렉시콘이 중요한 역할을 할 수 있음을 설명합니다.

- **Performance Highlights**: 아랍어 감정 분석 모델의 성능 향상에는 다양한 요소들이 기여합니다. 아랍어 렉시콘을 활용한 데이터 전처리와 감정 가중치 조정이 중요하게 다루어졌으며, 이는 특히 저자원 환경에서 모델 성능을 향상시키는 데 효과적입니다. 실제 사례로 딥 러닝 모델에 렉시콘을 통합하여 성능을 개선한 연구가 소개되며, 복잡한 모델의 해석 가능성(Interpretability) 또한 향상되는 결과가 나타났습니다.



### Syntriever: How to Train Your Retriever with Synthetic Data from LLMs (https://arxiv.org/abs/2502.03824)
Comments:
          the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted

- **What's New**: 이 논문에서는 Syntriever라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 최신 블랙박스 LLMs의 합성 데이터를 활용하여 리트리버 모델을 훈련 및 미세조정할 수 있도록 설계되었습니다. Syntriever는 두 단계로 구성되어 있으며, 첫 번째 단계에서는 LLM 생성 합성 데이터를 사용하여 리트리버를 훈련하고, 두 번째 단계에서는 LLM의 선호도와 리트리버를 정렬합니다.

- **Technical Details**: Syntriever는 LLM의 지식을 효과적으로 추출하는 데 필요한 두 가지 주요 단계를 포함합니다. 첫 번째 단계인 distillation 단계에서는 chain-of-thoughts를 이용해 합성 쿼리와 관련된 문서를 생성하고 LLM이 자체 검증을 통해 환각 가능성을 최소화합니다. 두 번째 단계인 alignment 단계에서는 리트리버를 LLM의 선호도에 맞추기 위해 partial Plackett-Luce ranking 방법을 사용하여 훈련합니다.

- **Performance Highlights**: Syntriever는 다양한 도메인의 벤치마크 데이터셋에서 모든 성능 기준을 크게 초과하는 결과를 보였습니다. 특히, nDCG@10 기준으로 이전의 최고 성능보다 최대 18.6% 향상이 있었습니다. 이 프레임워크는 다양한 기본 리트리버 및 LLM와 결합이 가능하여 검색 정확도 증가를 가져옵니다.



### PsyPlay: Personality-Infused Role-Playing Conversational Agents (https://arxiv.org/abs/2502.03821)
- **What's New**: 이번 연구에서는 Personality-infused Role-Playing Conversational Agents (RPCAs)를 제안하며, 이를 통해 대화 중에 고유한 성격 특성을 정확하게 묘사하도록 LLM 에이전트를 유도합니다. 이를 위해 새로운 대화 생성 프레임워크인 PsyPlay를 소개하며, 에이전트가 특정 주제를 중심으로 활발한 논의를 진행할 수 있도록 합니다. PsyPlay는 4745개의 성격이 주입된 대화 샘플로 구성된 PsyPlay-Bench 데이터 코퍼스를 제공하여 개인화된 역할극과 대화 성격 탐지 연구에 기여할 것입니다.

- **Technical Details**: PsyPlay 프레임워크는 Role Card Creation, Topic Extraction, Dialogue Generation의 세 가지 단계로 구성됩니다. Role Card Creation 단계에서는 고유한 성격과 특성을 가진 에이전트 역할을 대량으로 생성하고, Topic Extraction 단계에서는 현실적인 주제를 공공 데이터 세트에서 추출하여 대화가 비현실적인 잡담이 아닌 실제 문제에 집중되도록 합니다. 마지막으로 Dialogue Generation 단계에서는 에이전트가 지정된 성격 특성에 따라 주제에 대해 포괄적인 논의를 하도록 유도합니다.

- **Performance Highlights**: PsyPlay의 효과성을 평가하기 위해 GPT-3.5를 사용한 Personality Back-Testing을 수행했습니다. 그 결과, PsyPlay는 80.31%의 성공률로 의도된 성격 특성을 정확하게 묘사할 수 있음을 확인했습니다. 긍정적인 가치에 alignment된 LLM이 부정적인 역할보다 긍정적인 성격 역할을 더 성공적으로 묘사하는 경향이 있음을 발견했습니다.



### Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspectiv (https://arxiv.org/abs/2502.03805)
- **What's New**: 이 논문은 Key-Value (KV) 캐시에서 핵심 항목을 식별하기 위한 형식적 연구를 제공합니다. 기존 방법들이 주로 attention weights에만 의존한 데 반해, 이 연구는 출력 변동(output perturbation) 분석을 통해 핵심 항목을 찾는 새로운 알고리즘을 제안합니다. 이는 KV 항목 내의 value states와 pretrained parameter matrix의 중요성을 강조하며, 이에 따라 두 가지 최신 캐시 퇴출(Cache Eviction) 방법에 통합되어 효과를 보여줍니다.

- **Technical Details**: 저자들은 출력 변동을 최소화하는 것을 목표로 하고, L1 distance를 사용하여 worst-case perturbation을 정량화합니다. 제안된 알고리즘은 두 단계의 greedy 방식을 기반으로 하며, 기존의 SOTA 캐시 퇴출 방법과의 통합 과정을 통해 성능 향상을 이룹니다. 연구는 Needle-in-a-Haystack 테스트와 Longbench 벤치마크를 활용하여 제안된 방법의 효과를 검증합니다.

- **Performance Highlights**: 제안된 알고리즘은 Llama 모델에서 92% 이상의 attention heads에서 출력 변동을 효과적으로 줄이는 성과를 보였습니다. 또한 다양한 캐시 크기에서 좋은 성능을 유지하며, 실제 응용 프로그램에서 자원 제약 하에서도 품질 손실을 완화하는 데 기여합니다. 결과적으로, 알고리즘은 긴 시퀀스 추론 시 KV 캐시의 효율성을 극대화하고 있습니다.



### Enhancing Hallucination Detection through Noise Injection (https://arxiv.org/abs/2502.03799)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 잘못된 응답인 'hallucination'을 탐지하는 새로운 방법을 제안합니다. 기존의 샘플링 방식에서 발생하는 문제를 해결하기 위해, Bayesian 관점에서 모델의 불확실성을 고려하여 탐지 성능을 향상시키는 방안을 소개합니다. 저자는 모델 파라미터의 적절한 하위 집합을 변형하여 추론의 정확성을 개선했다고 주장합니다.

- **Technical Details**: 저자들은 hallucination 탐지를 위해 aleatoric(관측) 불확실성과 epistemic(모델) 불확실성을 함께 고려해야 한다고 주장합니다. 이들 불확실성을 기반으로하여, 샘플링할 수 있는 신뢰할 수 있는 모델의 분포를 추정하기 위한 새로운 접근법을 제안합니다. 이 접근법은 사전 훈련된 모델의 파라미터를 변형함으로써 수행됩니다.

- **Performance Highlights**: 논문에서는 제안된 방법이 다양한 데이터셋과 모델 아키텍처에서 효과적으로 작동함을 실험적으로 입증하고 있습니다. 특히, Gemma-2B-it, Phi-3-mini-4k-instruct, Mistral-7B-Instruct 등 여러 모델에 걸쳐 모델 불확실성을 결합하여 hallucination 탐지의 효과성을 입증했습니다. 이러한 방식은 언어 모델의 적합성을 향상시키는 데 큰 기여를 할 것으로 기대됩니다.



### It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers (https://arxiv.org/abs/2502.03793)
- **What's New**: 이 연구에서는 전통적인 태스크 전용 분류 헤드에 의존하지 않고 현대적인 encoder-only 모델인 ModernBERT-Large-Instruct를 도입합니다. 이 모델은 마스킹된 언어 모델링(MLM) 헤드를 활용하여 제너레이티브(classification를 위한 생성적) 기능을 발휘합니다. 특히, 이 모델은 기존의 LLM(large language models)보다 60% 더 적은 매개변수로 MMLU에서 Llama3-1B의 93% 성능을 달성했습니다.

- **Technical Details**: 이 연구에서 사용된 ModernBERT-Large-Instruct 모델은 단순한 훈련 루프와 추론 메커니즘을 채택하여 복잡한 전처리나 엔지니어링된 프롬프트 없이도 강력한 제로샷(Zero-shot) 성능을 보여줍니다. 대량의 현대화된 데이터 믹스를 통해 훈련된 이 모델은 다양한 NLU(자연어 이해) 작업에서 파인튜닝(fine-tuned) 과정 후에도 강력한 성능을 나타낸다고 밝혔습니다. 기존의 encoder-only 모델들에 비해 낮은 오버헤드를 지니면서도 효과적입니다.

- **Performance Highlights**: 제로샷 및 파인튜닝 설정 모두에서 ModernBERT-Large-Instruct는 이전 접근 방식과 경쟁력을 보여줍니다. 특히 뉴스 주제 감지, 텍스트 일관성, 포럼 게시물 주제 식별에서 기존의 분류 헤드 메소드와 어깨를 나란히 하거나 성능을 능가하는 결과를 보였습니다. 이러한 결과는 현대 architecture에 기반한 모델의 중요성과 다각적 데이터 믹스의 가치를 강조합니다.



### Hierarchical Contextual Manifold Alignment for Structuring Latent Representations in Large Language Models (https://arxiv.org/abs/2502.03766)
- **What's New**: 이 연구는 Hierarchical Contextual Manifold Alignment (HCMA)라는 새로운 방법론을 소개하여, 대규모 언어 모델의 토큰 임베딩을 비모수적(non-parametric) 변환을 통해 재구성하면서 언어적 관계를 유지하고 표현 클러스터링의 일관성을 높였습니다. 기존의 파라미터 조정 및 강화 학습 접근 방식과는 달리, HCMA는 모델의 기본 네트워크 가중치를 변경하지 않고도 효율성을 유지하며 구조적 개선을 도입하는 것을 목표로 합니다. 실험 결과, 이 방법이 희귀 토큰 검색 및 장기 의존성 추적에서 개선을 보여주었고, 일부 언어적 작업에서의 맥락적 안정성을 향상시키는 것으로 나타났습니다.

- **Technical Details**: HCMA는 토큰 분포의 내재적(topological) 지형을 추론하고 구조적 정렬 과정을 통해 토큰 표현에서의 조각화(fragmentation)와 중복성을 줄이는 방식으로 작동합니다. 이는 토큰 간의 계층적 관계를 암시할 수 있는 명확한 메커니즘을 만들어 내며, 사전 학습된 지식을 손상시키지 않으면서도 표현 품질에서 측정 가능한 향상을 달성합니다. 연구는 열린 소스 LLM에 대한 광범위한 실험 세트를 통해 HCMA의 임베딩 공간 조직 개선을 평가했습니다.

- **Performance Highlights**: 본 연구의 결과는 HCMA가 다른 기존 세밀 조정 및 임베딩 변동화 방법들과 비교해도 관찰된 개선과 더불어 계산 효율성을 유지한다는 점을 강조합니다. 또한, 맥락의 일관성을 향상시키며 모델의 해석 가능성을 높인 것으로 나타났습니다. 이 연구는 토큰 근접 관계에서의 불일치를 줄여 언어 생성에서의 해석성을 향상시키는 방향으로, 구조적 표현 학습의 더 넓은 중요성을 reinforced합니다.



### Rethinking the Residual Distribution of Locate-then-Editing Methods in Model Editing (https://arxiv.org/abs/2502.03748)
Comments:
          Preprint

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 지식을 편집하는 새로운 기법인 Boundary Layer UpdatE (BLUE)를 제안합니다. 기존의 locate-then-edit 방법의 한계인 레지듀얼(distribution residual) 문제를 해결하기 위해 BLUE는 첫 번째와 마지막 중요 레이어만 업데이트하여 성능 개선을 이루고 있습니다. 본 연구는 BLUE가 평균 35.59%의 성능 향상을 가져오며 LLM의 일반적인 기능 보존 능력을 높인다는 점을 강조합니다.

- **Technical Details**: 이 논문은 locate-then-edit 방법의 레지듀얼 분포(residual distribution)가 편집 오류를 유발하여 원래의 LLM 지식을 저해하는 방식에 대해 이론적 및 경험적으로 분석합니다. BLUE 전략은 레지듀얼 분포 대신 레지듀얼의 직접 계산을 통해 첫 번째 및 마지막 중요 레이어만 업데이트함으로써 모델 편집의 정확성을 높입니다. 이를 통해 12개의 연속 편집 실험에서 성능 향상과 일반 기능 보존을 보여주었습니다.

- **Performance Highlights**: 경험적 분석에 따르면, BLUE가 적용된 모델 편집 방법은 기존의 방법에 비해 35.59%의 성능 개선을 가져왔습니다. 이 연구에서는 BLUE가 MEMIT, AlphaEdit, PRUNE, RECT 같은 모델 편집 기법에 효과적으로 적용되었으며, 후속 작업의 성능도 향상된다고 설명합니다. 결론적으로, BLUE는 기존의 locate-then-edit 방법이 가지는 성능 저하 문제를 해결할 수 있는 혁신적인 대안임을 증명하고 있습니다.



### MultiQ&A: An Analysis in Measuring Robustness via Automated Crowdsourcing of Question Perturbations and Answers (https://arxiv.org/abs/2502.03711)
Comments:
          AAAI 2025 Workshop on Preventing and Detecting LLM Misinformation (PDLM) (Oral)

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 답변에서 발생할 수 있는 hallucination 문제를 해결하기 위한 시스템인 MultiQ&A를 제안합니다. MultiQ&A는 LLM이 생성한 답변의 일관성과 견고성을 평가하는 체계적인 접근 방식으로, 다양한 질문의 변형을 만들고 이에 대한 답변을 crowdsourcing하고 있습니다. 실험을 통해 1.9백만 개의 질문 변형과 2.3백만 개의 답변을 검토하였으며, gpt-3.5-turbo 모델이 변수에 대해 비교적 일관성을 유지함을 보여줍니다.

- **Technical Details**: MultiQ&A는 세 가지 구성 요소로 이루어진 강력한 다단계 파이프라인을 갖고 있습니다: Query Rewriter, Answer Generator, Aggregator. Query Rewriter는 원래의 쿼리(q0)를 다양한 의미적으로 일관된 변형으로 변환하며, Answer Generator는 이러한 변수들에 대해 독립적인 gpt-3.5-turbo 에이전트를 사용하여 다수의 답변을 생성합니다. Aggregator는 개별 답변을 통합하여 최종 결과를 도출하는 역할을 수행합니다.

- **Performance Highlights**: MultiQ&A의 실험 결과, 1.9 백만 개의 질문과 2.3 백만 개의 답변을 분석하여 실제 시나리오를 모방한 결과, gpt-3.5-turbo 모델이 의미적으로 안정적이면서도 다양한 표현을 생성함을 추가적으로 확인하였습니다. MultiQ&A는 LLMs의 변동성을 강조하며, 각 질문에 대한 모델의 변화를 보여주는 것을 목표로 한다는 점에서 큰 의의가 있습니다. 따라서, 이 시스템은 기관에서 LLM를 채택하기 위한 신뢰성 있는 프레임워크로 작용할 가능성을 제공합니다.



### Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers (https://arxiv.org/abs/2502.03708)
- **What's New**: 이 논문에서는 Large Language Model (LLM)의 내부 활성화를 통해 의미론적 개념을 감지하고, 원하는 출력으로 유도하는 일반적인 방법을 제안합니다. 특히, 비선형 특성 학습 방법을 사용하여 각 레이어에서 개념을 예측하는 데 중요한 선형 방향을 식별하고, 레이어 간의 특징을 집계하여 강력한 개념 탐지기와 유도 메커니즘을 구축합니다. 이를 통해 환각, 유해성, 독성 및 진실성 결여를 탐지하는 데 최신 결과를 달성하였습니다.

- **Technical Details**: 이 연구에서는 Recursive Feature Machines (RFMs)를 통한 비선형 방법을 사용하여 레이어별 내부 활성화에서 특정 개념을 감지하고 이를 유도하는 프레임워크를 설명합니다. LLM의 활성화로부터 개념 벡터를 집계함으로써 감지 및 유도 프로세스를 개선하며, 데이터 효율성을 높이는 동시에 적은 수의 라벨이 있는 훈련 요청으로도 높은 성능을 보여줍니다. 본 방법은 표준 LLM 추론 파이프라인에 통합할 수 있어 별도의 개념별 튜닝 모델이 필요하지 않습니다.

- **Performance Highlights**: 본 접근 방식은 기존의 탐지 방법보다 향상된 성능을 보여줍니다. 일곱 개의 벤치마크에서 잇따른 실험을 통해 우리의 탐지 방법이 환각과 유해성 개념을 포함하여 다양한 개념을 감지하는 데 효과적임을 입증하였습니다. 또한, 상대적으로 적은 자원을 가진 공개 언어 모델에서도 GPT-4o와 같은 최신 LLM을 초내기 위한 성능 향상도 목격되었습니다.



### LLM Alignment as Retriever Optimization: An Information Retrieval Perspectiv (https://arxiv.org/abs/2502.03699)
Comments:
          26 pages

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 정렬(alignment)을 향상시키기 위한 새로운 직접 최적화 방법인 LarPO(LLM Alignment as Retriever Preference Optimization)를 소개합니다. 기존의 RL(강화 학습) 기반 접근법의 복잡성을 해결하기 위해, 정보 검색(IR) 원리를 활용하여 LLM과 IR 방법론을 연결하는 시스템적인 프레임워크를 제안합니다. 이를 통해 LLM의 생성 모델과 보상 모델을 IR의 검색기-재정렬기(retriever-reranker) 패러다임에 매핑합니다.

- **Technical Details**: LarPO는 LLM의 선호도를 직접 최적화하는 새로운 방법으로, 정보 검색의 세 가지 핵심 원리인 검색기 최적화 목적, 하드 네거티브 마이닝(hard negative mining), 후보 목록 구축(candidate list construction)을 활용하여 정렬 품질을 개선합니다. LLM은 검색기의 역할을 하고, 보상 모델은 재정렬기의 역할을 수행함으로써 효과적인 정렬 성능을 구현합니다. 이러한 접근은 LLM의 생성 및 IR 검색기에서 일반적으로 사용되는 이중 인코더 아키텍처를 활용하여 성과를 도출합니다.

- **Performance Highlights**: 실험 결과, LarPO 방법은 AlpacaEval2에서 38.9%와 MixEval-Hard에서 13.7%의 평균적인 성능 향상을 보여주었습니다. 이러한 결과는 LarPO의 효과를 증명하며, 정보 검색 기법이 LLM 정렬의 품질을 높일 수 있는 새로운 가능성을 제시합니다. 본 연구는 LLM 정렬과 정보 검색 간의 중요한 연결고리를 확립함으로써, 이 분야의 발전을 위한 실용적인 방법을 제공합니다.



### A Comparison of DeepSeek and Other LLMs (https://arxiv.org/abs/2502.03688)
Comments:
          21 pages, 5 figures, 6 tables

- **What's New**: 최근 DeepSeek(딥시크)는 AI 커뮤니티 내외에서 많은 주목을 받고 있으며, 본 논문에서는 DeepSeek과 다른 대형 언어 모델(LLM) 간의 비교를 다룬다. 본 연구는 두 가지 설정에서 작업을 수행하며, 첫 번째는 저자 분류(authorship classification), 두 번째는 인용 분류(citation classification)다. 각 실험에서 DeepSeek은 Claude, Gemini, GPT 및 Llama와 같은 4개의 인기 LLM과 비교된다.

- **Technical Details**: 이 논문에서는 저자 분류와 인용 분류를 통해 고유한 데이터 세트를 활용하여 LLM의 예측 정확도를 분석한다. 저자 분류는 문서가 인간에 의해 작성되었는지 AI에 의해 생성되었는지를 판단하는 작업이다. 인용 분류에서는 작은 텍스트 조각과 그에 대한 인용 유형을 매칭하는 정밀한 분류를 수행한다.

- **Performance Highlights**: DeepSeek은 대부분의 경우 Gemini, GPT 및 Llama에 비해 분류 정확도에서 우위를 보였으나, Claude에는 미치지 못했다. DeepSeek은 상대적으로 느리지만 사용 비용이 낮고, Claude보다 상당히 비쌌다. 출력 유사성 면에서 DeepSeek은 Gemini 및 Claude와 가장 유사한 결과를 보였다.



### Controlled LLM Decoding via Discrete Auto-regressive Biasing (https://arxiv.org/abs/2502.03685)
- **What's New**: 이번 논문에서는 기존의 에너지 기반 디코딩 방식을 개선하여, 텍스트 생성의 제어 가능성을 높이는 새로운 접근 방식을 제시합니다. 기존의 기법들은 매개 변수를 조정하더라도 유창성(fluency)과 제약(constraint) 만족 간의 균형이 부족한 문제를 가지고 있었습니다. 이에 따라, 우리는 연속 공간에서 샘플링하는 것이 아니라, 텍스트 토큰의 자연스러운 이산(discrete) 공간에서 작동하는 Discrete Auto-regressive Biasing이라는 새로운 디코딩 알고리즘을 제안합니다.

- **Technical Details**: 우리가 제안한 방법은 생성된 시퀀스와 보조 바이어스 시퀀스에 대한 결합 분포를 정의하여 제어된 텍스트 생성을 위한 새로운 수식을 도입합니다. 이를 통해, 기울기(gradient)를 활용하면서 완전히 이산 텍스트 도메인에서 작동하는 Langevin-within-Gibbs 샘플링 알고리즘을 제시합니다. 이 접근 방식은 제약 만족도를 효과적으로 향상시키면서도, 유창성을 유지하거나 심지어 개선하는 데 기여합니다.

- **Performance Highlights**: 우리의 제어된 디코딩 방법은 감정 제어(sentiment control), 언어 해독(language detoxification), 키워드 유도 생성(keyword-guided generation) 등의 작업에서 두드러진 장점을 나타냅니다. 제안된 방법은 이전의 기술에 비해 계산 비용이 낮으면서도 성능을 유지하거나 향상시키는 것을 보여줍니다. 이러한 결과는 대규모 언어 모델의 출력에 대해 사용자 정의 제약을 효과적으로 적용할 수 있는 가능성을 제시합니다.



### Reflection-Window Decoding: Text Generation with Selective Refinemen (https://arxiv.org/abs/2502.03678)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 텍스트 생성에서 자기 회귀 해독의 단점을 이론적으로 규명하고, 생성된 내용을 정제하기 위한 슬라이딩 리플렉션 윈도우와 일시정지 기준을 포함한 프레임워크를 제안합니다. 이는 정제(refinement)와 생성을 교환적으로 수행할 수 있는 방법을 통해 효율성과 최적성을 동시에 충족시킬 수 있습니다. 이 접근 방식은 기존의 해독 방식보다 상당한 개선을 보여줍니다.

- **Technical Details**: 연구는 텍스트 생성에서 자기 회귀 방식의 단점을 강조합니다. 자기 회귀 방식은 이전에 생성된 내용을 수정하는 자연스러운 메커니즘이 부족하여 최적의 응답을 보장하지 못합니다. 본 논문에서는 슬라이딩 리플렉션 윈도우와 일시정지 기준을 도입하여 다수의 토큰을 병렬로 예측할 수 있도록 하고, 이로 인해 해독 과정 중 독립적으로 정제와 생성을 오갈 수 있는 구조를 제시합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 접근 방식이 기존 해독 방법보다 성능이 개선되었음을 보여줍니다. 이 방법은 성능 및 효율성 면에서 빔 탐색(beam search)과 유사하거나 더 나은 결과를 제공합니다. 따라서 새로운 접근은 텍스트 생성을 위한 효율적이고 최적화된 해법을 제공하는 데 기여합니다.



### Advancing Reasoning in Large Language Models: Promising Methods and Approaches (https://arxiv.org/abs/2502.03671)
Comments:
          9 Pages, 1 Figure, IEEE Format

- **What's New**: 이번 논문은 최근의 연구가 진행되고 있는 Large Language Models (LLMs)의 추론 능력 향상 방법들을 종합적으로 리뷰합니다. LLMs는 자연어 처리(NLP) 분야에서 큰 성공을 거두었지만, 심층적인 추론 능력에서 여전히 제한이 있습니다. 저자들은 특히 Chain-of-Thought, Self-Consistency, Tree-of-Thought와 같은 다양한 Prompting Strategies, retrieval-augmented models와 neuro-symbolic integration 같은 Architectural Innovations, 그리고 specialized datasets를 활용한 Learning Paradigms를 통해 LLMs의 추론 능력 강화를 시도하고 있습니다.

- **Technical Details**: LLMs의 추론 능력은 논리적 추론, 수학적 문제 해결 및 상식적 추론 등 다양한 인지 프로세스 유형으로 나눌 수 있습니다. 이 논문에서는 Deductive, Inductive, Abductive 및 Commonsense Reasoning을 소개하며, 전통적인 AI 접근 방식과 LLMs의 차이를 설명합니다. 또한, Chain-of-Thought(CoT) 추론을 포함한 다양한 Prompting Techniques의 중요성을 강조하며, Self-Consistency와 같은 기법을 통해 LLMs의 응답 품질을 개선할 수 있는 방법을 조명합니다.

- **Performance Highlights**: DeepSeek-R1과 같은 최신 LLM은 특히 수학 및 코딩 분야에서 뛰어난 추론 능력을 보여주고 있습니다. 이 모델은 사람과 유사한 분석적 사고를 시뮬레이션하여 복잡한 문제 해결 및 논리적 추론에서 성과를 나타냅니다. CoT와 Self-Consistency 기법은 LLM이 다양한 문제에 대해 더 정확한 추론을 할 수 있도록 돕는 데 기여하고 있으며, 이러한 방법들을 활용한 연구가 지속될 것으로 기대됩니다.



### Looking for the Inner Music: Probing LLMs' Understanding of Literary Sty (https://arxiv.org/abs/2502.03647)
- **What's New**: 최근 연구에 따르면, 언어 모델이 전통적인 스타일 분석에서 생각했던 것보다 훨씬 동안의 짧은 문학 구절의 저자를 식별하는 데 훈련될 수 있다는 것을 보여줍니다. 우리는 이러한 결과를 저자 식별로 재현하고 새로운 데이터셋을 통해 소설 장르 분석으로 확장합니다. 흥미롭게도, LLM들은 저자와 장르를 구별할 수 있지만 서로 다른 방식으로 작동하며 이는 메모리 기반 접근과 특성 학습 방식의 차이를 보여줍니다.

- **Technical Details**: 우리는 훈련된 LLM이 저자 스타일을 정의하는 특징을 찾기 위해 세 가지 방법을 사용합니다. 여기에는 입력 텍스트에 대한 직접적인 구문적 제거(syntactic ablations) 및 모델의 내부 구조를 분석하는 방법이 포함됩니다. 결과적으로, 저자 스타일은 장르 레벨 스타일보다 정의하기가 더 쉽고, 작은 구문적 결정 및 문맥적 단어 사용에 더 큰 영향을 받는 것으로 나타났습니다.

- **Performance Highlights**: 모델들은 매우 짧은 텍스트(20~50 단어)에서 저자와 장르를 인식하는 데 있어 무작위 정확도를 초과하는 성과를 보였습니다. 가장 큰 LLM인 Llama-3와 Flan-T5는 27명의 저자 및 5개의 장르에 대한 텍스트를 각각 50% 이상의 정확도로 분류하여 가장 높은 성능을 기록했습니다. 이러한 결과는 문학적 신호가 이 규모에서도 존재함을 확인시켜 주며, 저자 스타일과 장르 스타일을 구별하는 데 사용되는 특징이 다름을 보여줍니다.



### Context-Preserving Gradient Modulation for Large Language Models: A Novel Approach to Semantic Consistency in Long-Form Text Generation (https://arxiv.org/abs/2502.03643)
- **What's New**: 이 논문에서는 긴 텍스트 생성 과정에서 의미적 일관성을 유지하는 문제를 해결하기 위해 Context-Preserving Gradient Modulation (CPGM)이라는 새로운 접근 방식을 제안합니다. CPGM은 모든 상황에 적합한 크기가 결정된 gradient 업데이트를 동적으로 조정하여 장기적인 맥락 유지를 우선시하도록 설계되었습니다. 이 방법은 기존의 훈련 프로세스에 비해 심리적 일관성을 향상시킴으로써 텍스트 생성의 품질을 높입니다.

- **Technical Details**: CPGM의 기본 개념은 맥락에 따라 gradient를 제어하고 모듈화하여 일관성을 높이는 것입니다. 이를 통해 모델은 문맥과 관련된 정보를 우선시하고, 주제 이탈 및 일관성 저하 문제를 완화합니다. 포괄적인 맥락과 일치시키기 위해 예측된 현재 토큰의 중요성을 평가하며, 이러한 동적 조정은 더 일관된 내러티브 흐름을 생성합니다.

- **Performance Highlights**: 비교 평가 결과, CPGM이 개별 모델의 일관성, 맥락 유지 및 장기 의존성 추적에 긍정적인 영향을 미친 것으로 나타났습니다. 기존 모델에 비해 반복적인 구문을 완화하고 다양한 언어적 맥락에서의 적응성을 향상시켰으며, 통계적 검증을 통해 일관성 지표의 개선이 입증되었습니다. CPGM은 기존 아키텍처 변경 없이도 저비용으로 이러한 이점을 제공하는 효율성을 확보하고 있습니다.



### Sorting the Babble in Babel: Assessing the Performance of Language Detection Algorithms on the OpenAlex Databas (https://arxiv.org/abs/2502.03627)
Comments:
          33 pages, 4 figures

- **What's New**: 이번 연구는 OpenAlex의 언어 메타데이터 품질을 최적화하기 위한 다양한 언어 분류 절차들을 설계하고, 이를 평가하는 데 중점을 두고 있습니다. 최신 자동 언어 탐지 알고리즘을 활용하여 다수의 기사 샘플을 수집하고, 이를 기반으로 한 다양한 분류 절차를 제안합니다. 연구의 주요 목적은 이러한 알고리즘을 적용하여 최적의 성능을 발휘하는 방법을 찾는 것입니다.

- **Technical Details**: 이 연구에서는 메타데이터를 바탕으로 기사 언어를 추론하는 과정을 두 가지 주요 과제로 나누어 수행합니다. 첫째, 메타데이터 언어를 탐지하고, 둘째, 해당 메타데이터에 따라 기사의 언어를 유추합니다. 이 과정은 알고리즘의 성능을 높이기 위해 조합된 데이터세트를 필수적으로 사용하며, 성능 평가는 정밀도(precision), 재현율(recall), 처리 시간(processing time) 등을 통해 이루어집니다.

- **Performance Highlights**: 연구 결과, 알고리즘의 성능은 구현된 측정값의 중요성에 따라 크게 달라짐을 알 수 있었습니다. LangID 알고리즘은 제목, 초록, 저널명 등 다양한 메타데이터에서 최상의 결과를 보여주었으며, 반면에 FastSpell 알고리즘은 제목만 사용하는 경우에 가장 효과적임을 입증했습니다. 이러한 연구 결과는 OpenAlex 데이터베이스의 다국어 정보 검색 및 분석의 가능성을 더욱 높여줄 것으로 기대됩니다.



### Can Cross Encoders Produce Useful Sentence Embeddings? (https://arxiv.org/abs/2502.03552)
- **What's New**: 본 연구에서는 Cross Encoder (CE)로부터 추출한 초기 레이어의 hidden states가 정보 검색(Information Retrieval)에서 유용하게 활용될 수 있다는 흥미로운 발견을 보고합니다. 일반적으로 Dual Encoder (DE)가 협코딩 방식으로 메모리를 세분화하여 정보를 추출하도록 훈련되지만, CEs의 초기 레이어에서 얻은 임베딩이 DE보다 정보 검색에 더 효과적임을 보여주고 있습니다. 이로 인해 CE를 활용하여 더 가벼운 DE를 생성하고, 추론 속도를 5.15배 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 본 논문에서는 CE에서 문장 쌍을 위한 implicit한 임베딩을 추출하는 방법을 설명합니다. CE의 각 레이어에서 token을 평균 풀링(mean pooling)하여 임베딩을 생성하며, 초기 레이어의 가중치를 DE 모델에 주입하여 새로운 지식 주입(knowledge infusion) 모델을 구축합니다. 제안하는 DE-2 CE 모델은 기존 DE 모델과 비교하여 최대 1%의 정확도 차이를 보이며, 추론 시간이 약 5배 더 빠른 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, CE의 초기 레이어 hidden states가 DE의 첫 번째 레이어보다 정보 검색에 더 강한 신호를 전달한다는 것을 확인했습니다. 총 12개의 다양한 데이터셋을 활용하여 IR 성능을 측정했으며, CE 기반의 지식 주입이 정보 검색에서 유용하다는 것을 입증했습니다. 최종적으로, 제안된 DE-2 CE 모델은 유사한 DE 모델에 비해 성능 저하 없이 현저히 빠른 추론 속도를 기록함으로써 효율성을 높였습니다.



### Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignmen (https://arxiv.org/abs/2502.04328)
- **What's New**: 최근 GPT-4o의 발전에 따라 다중 모달 언어 모델에 대한 관심이 증가하고 있습니다. 이러한 배경에서 소개된 Ola 모델은 이미지, 비디오 및 오디오를 이해하는데 있어 전문적인 모델들과 경쟁할 수 있는 성능을 달성했습니다. Ola는 언어 모델의 지원 모달리티를 점진적으로 확장하는 Progressive Modality Alignment 전략이 핵심 설계로 도입되었습니다.

- **Technical Details**: Ola 모델은 초기 이미지와 텍스트의 기본 모달리티에서 시작하여 모델의 스킬 세트를 점진적으로 확장합니다. 이는 음성 데이터와 비디오는 모든 모달리티의 연결을 강화하기 위한 중요한 단계입니다. 또한, Ola는 다양한 모달리티를 처리할 수 있는 아키텍처를 지원하며, 고품질 음성 합성을 위한 Sentence-wise Streaming Decoding 모듈을 통합하여 실시간으로 사용자와 상호작용할 수 있습니다.

- **Performance Highlights**: Ola는 7억 개의 매개변수를 통해 이미지, 비디오 및 오디오 벤치마크에서 경쟁력 있는 성능을 보입니다. OpenCompass 기준에서 평균 정확도 72.6%, VideoMME 기준에서 68.4%의 인상적인 정확성을 달성하며, 오디오 이해 과제에서도 평균 WER 3.1을 기록했습니다. 이러한 결과는 기존의 오픈 옴니 모달 LLM을 크게 초과하며, 최신 전문 LLM과 비슷한 규모의 모델 대비 우수한 성능을 입증합니다.



### Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions (https://arxiv.org/abs/2502.04322)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 안전성 문제를 다루며, 기존의 연구들이 주로 기술적인 지식을 요구하는 공격 방법에 집중하고 있다는 점을 지적합니다. 연구자들은 jailbreak된 응답이 일반 사용자에게 해로운 행동을 유도하는 데 진정으로 유용한지와 간단한 상호작용에서 안전성 취약점이 존재하는지를 탐구합니다. 이를 통해 저자들은 HarmScore라는 새로운 메트릭을 제안하며, 다중 단계, 다국어 상호작용에서 해로운 행동을 유도하기 위한 새로운 프레임워크인 Speak Easy를 소개합니다.

- **Technical Details**: Speak Easy는 일반 사용자가 해로운 내용을 찾기 위해 사용할 수 있는 두 가지 유형의 인간-LM 상호작용인 다중 단계 추론(multi-step reasoning)과 다국어 질문(multilingual querying)을 모사합니다. 연구자들은 이 프레임워크를 통해 사용자가 해로운 쿼리를 여러 개의 무해한 하위 쿼리로 나누어 LLM의 안전 장치를 쉽게 우회할 수 있도록 하는 방법을 제안합니다. 이 논문은 GPT-4o, Qwen2-72B-Instruct 등 여러 안전 정렬된 LLM을 대상으로 Speak Easy의 효과를 체계적으로 평가하며, HarmScore는 인간 평가와 잘 일치함을 보여줍니다.

- **Performance Highlights**: Speak Easy는 여러 기준을 통해 GPT-4o의 공격 성공률(Attack Success Rate, ASR)을 평균 0.319 증가시키고, HarmScore를 0.426 증가시키는 결과를 도출했습니다. 또 다른 분석 연구를 통해 더 많은 분해 단계와 언어를 사용하는 것이 응답의 해로움을 증가시킨다는 것을 입증했습니다. 특이하게도, Speak Easy는 기존의 jailbreak 방법들에도 통합되어 성능 향상을 가져오는 것으로 나타났습니다.



### Great Models Think Alike and this Undermines AI Oversigh (https://arxiv.org/abs/2502.04313)
Comments:
          60 pages, 20 figures

- **What's New**: 본 논문은 AI Oversight의 맥락에서, 언어 모델(LM)의 유사성이 이들 평가 및 감독에서 어떻게 작용하는지를 탐구합니다. 이를 위해, 모델의 실수(overlap in model mistakes)를 기반으로 한 확률적 유사성 측정 지표인 CAPA를 제안합니다. 연구 결과, 유사한 모델들이 상호작용할 때 더 나은 평가를 수행하게 된다라는 사실을 발견하였습니다.

- **Technical Details**: CAPA는 모델의 정확도를 고려하여 유사성을 측정하기 위해 고안된 메트릭입니다. 논문의 기술적 세부사항에서는 CAPA의 수학적 유도와 다수 모델 설정에의 확장 방법이 포함되어 있습니다. 이 메트릭은 전통적인 정합성은 물론, Scatter π와 Fleiss κ 같은 다른 메트릭들과 비교하여 새로운 계산 방식을 도입하고 있습니다.

- **Performance Highlights**: 모델의 능력이 증가함에 따라, 유사한 오류를 내는 경향이 발견되었습니다. 이는 AI Oversight의 위험 요소를 부각시키며, 모델 유사성을 모니터링할 필요성을 강조합니다. 결론적으로, 모델 간의 유사성을 보도하고 수정하는 것이 AI 감독의 새로운 패러다임에서 필수적임을 언급하고 있습니다.



### Multi-agent Architecture Search via Agentic Supern (https://arxiv.org/abs/2502.04180)
- **What's New**: 본 논문에서는 Large Language Model (LLM) 기반의 다중 에이전트 시스템을 발전시키기 위한 새로운 접근 방식인 MaAS를 소개합니다. MaAS는 기존의 고정된 설계 방식에서 벗어나, 다양한 쿼리에 맞춰 동적으로 리소스를 할당하는 에이전틱 슈퍼넷(agentic supernet)을 최적화합니다. 이러한 방법은 에이전트 간의 협업과 상호 작용을 통해 에이전트의 인지 경계를 확장하도록 돕습니다.

- **Technical Details**: MaAS는 쿼리에 따라 의존적인 에이전트 시스템을 슈퍼넷에서 샘플링하여 고품질 솔루션을 제공하고 맞춤형 리소스 할당을 가능하게 합니다. 이는 LLM 호출, 도구 호출, 토큰 비용 등을 포함한 복잡한 리소스 관리 기능을 갖추고 있습니다. 이 접근법은 기계 학습(ML) 및 자동화 시스템(AI systems)의 설계를 효율적으로 구현할 수 있도록 도와줍니다.

- **Performance Highlights**: MaAS는 기존의 수작업이나 자동화된 다중 에이전트 시스템보다 6%에서 45%까지 적은 추론 비용을 요구하며, 성능은 0.54%에서 11.82%까지 향상되었습니다. 또한, 다양한 데이터셋과 LLM 백본 간의 전이 가능성이 우수하여 적용 범위가 넓습니다. 이러한 성능은 MaAS가 다중 에이전트 시스템 설계의 혁신적 대안이 될 수 있음을 보여줍니다.



### Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis (https://arxiv.org/abs/2502.04128)
- **What's New**: 최근 언어 모델(GPT 시리즈 및 o1 모델)의 발전은 훈련 및 추론 시 컴퓨팅 리소스를 확장하는 것이 효과적임을 보여주고 있습니다. 이 논문은 확장된 훈련 및 추론 성능을 갖춘 단일 레이어 벡터 양자화(VQ) 코덱과 Transformer 아키텍처를 활용한 TTS 시스템 LLaSA를 제안합니다. LLaSA는 TTS의 자연스러운 음성과 복잡한 프러소디 패턴 생성을 개선하며, 기존의 다단계 TTS 시스템의 복잡성을 줄입니다.

- **Technical Details**: LLaSA는 LLaMA 모델을 기반으로 하여 음성 토큰을 포함하는 확장된 어휘를 사용합니다. 훈련 시, 자연스러운 음성과 프러소디 정확성을 높이기 위해 훈련 시간을 늘리는 것이 중요하다고 설명하고 있습니다. 또한, LLaSA는 음성 이해 모델을 확인자로 사용하여 추론 시 생성 출력을 특정 확인자 바이어스에 맞추어 더 감정적으로 표현할 수 있도록 하는 최적화 기법을 적용합니다.

- **Performance Highlights**: 리브리Speech(LibriSpeech) 테스트 세트 실험에서 LLaSA는 최첨단 성능을 달성하였으며, 감정 표현, 음색 일관성 및 콘텐츠 정확도를 크게 향상시켰습니다. 이 연구에서 제안된 모델은 오픈소스가 되어, TTS 커뮤니티의 혁신을 촉진할 것으로 기대됩니다. 실험 결과, 훈련 및 추론 단계의 확장이 TTS 성능을 크게 향상시키는 것을 입증했습니다.



### Leveraging Reasoning with Guidelines to Elicit and Utilize Knowledge for Enhancing Safety Alignmen (https://arxiv.org/abs/2502.04040)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문에서는 대규모 언어 모델의 안전성을 확보하기 위한 기존의 Refusal Training (RT) 방법의 한계를 분석하고, OOD(Out-Of-Distribution) 공격에 대한 일반화 성능을 향상시키기 위해 새로운 접근법을 제안합니다. 연구 결과, RT는 안전과 관련된 잠재적 지식을 일관되게 이끌어내지 못하는 문제를 드러냈고, 이를 해결하기 위해 Safety Reasoning with Guidelines (SRG)라는 방법을 소개합니다. SRG는 각 쿼리에 대해 단계별 이유 추론을 수행하도록 모델을 훈련시킵니다.

- **Technical Details**: 논문에서는 RT가 OOD 상황에서 일반화 능력이 부족하다고 지적하며, 훈련에서 직접적인 거부에 의존할 경우 모델이 피상적인 단축 경로에 의존하게 되고 탄력적인 표현 매핑을 학습하지 못한다고 설명합니다. 새로운 방법인 SRG는 3가지 필수 구성 요소를 포함하여, 지침에 따라 쿼리에 대한 단계적 추론을 수행하도록 모델을 훈련시키며, 이는 안전 관련 지식 활용을 효과적으로 증진시킵니다. 이 방법은 처리 과정에서 지침의 맥락 조화를 내재화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, SRG 방법이 기존의 RT 방식에 비해 OOD 공격에 대한 일반화 성능을 유의미하게 개선시키는 것으로 나타났습니다. BoN(Best-of-N) 평가 방식을 적용한 결과, N이 증가함에 따라 OOD 공격에 대한 실패 비율(ASR)이 감소하는 현상이 관찰되었습니다. 이 연구는 모델이 안전 관련 잠재 지식을 충분히 보유하고 있지만, 기존 방법이 이를 일관되게 이끌어내지 못했음을 입증하고 있습니다.



### Enhancing Online Learning Efficiency Through Heterogeneous Resource Integration with a Multi-Agent RAG System (https://arxiv.org/abs/2502.03948)
- **What's New**: 이번 포스터 논문은 학습 효율성을 높이기 위한 Multi-Agent Retrieval-Augmented Generation (RAG) 시스템을 도입합니다. 이 시스템은 YouTube 튜토리얼, GitHub 저장소, 문서 웹사이트와 같은 다양한 온라인 리소스를 통합하여 정보 검색 및 합성 과정을 자동화합니다. 초기 사용자 연구에서 시스템의 사용성과 유용성이 높게 평가되어, 지식 습득의 효율성을 크게 향상시킬 가능성을 보여주었습니다.

- **Technical Details**: Multi-Agent RAG 시스템은 GPT-4o를 기반으로 하며, 중앙의 Manager Agent가 YouTube Video, GitHub Repository, Documentation Website 및 Generic Search Engine과 같은 네 가지 특화된 에이전트 간의 작업을 조정합니다. 각 에이전트는 독립적으로 운영되며, 관련 정보 검색을 위해 API와 도구를 활용합니다. 시스템의 모듈형 디자인은 확장성과 적응성을 보장하여, 필요에 맞춰 추가 에이전트나 도구를 쉽게 통합할 수 있습니다.

- **Performance Highlights**: 예비 평가 결과, 사용자 설문조사에 참여한 15명의 대학원생들은 Perceived Usefulness (PU) 점수가 평균 75로 중간 높은 유용성을 나타냈지만, 각기 다른 사용자들의 요구에 따라 유용성 인식에 차이가 있음을 보여주었습니다. 반면, Perceived Ease of Use (PEU) 점수는 평균 91.11로 강력한 사용성을 나타내어 대부분 사용자들이 시스템의 유용성을 높이 평가했습니다. 이처럼, 시스템의 높은 사용성과 더불어 유용성을 개선하기 위한 특정적 노력의 필요성이 제기되었습니다.



### DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation (https://arxiv.org/abs/2502.03930)
Comments:
          16 pages, 8 figures

- **What's New**: 이번 연구에서는 Diffusion Transformer Autoregressive Modeling (DiTAR)이라는 패치 기반의 오토회귀(autoregressive) 프레임워크를 제안하며, 언어 모델(language model)과 확산 변환기(diffusion transformer)를 결합합니다. 이를 통해 연속적인 매개변수 생성을 위한 효율성을 크게 향상시키고, 계산 요구 사항을 줄이는 효과를 보여줍니다. 특히, DiTAR는 패치 생성을 위한 분할 정복 전략을 활용하여, 언어 모델이 집계된 패치 임베딩을 처리하고, 확산 변환기가 다음 패치를 생성할 수 있도록 합니다.

- **Technical Details**: DiTAR는 연속 토큰의 예측을 위해 causal attention과 bidirectional attention의 강점을 결합한 오토회귀 모델입니다. 이를 통해 DiTAR는 패치를 여러 개로 나누고, 언어 모델이 패치 간 예측을 담당하며, 확산 변환기가 패치 내 예측을 수행하며 효율성을 높이고 있습니다. 새로운 온도(temperature) 정의를 통해 역확산 ODE에서 노이즈의 도입 시점을 정하고, 이를 활용한 샘플링 기법을 제안하여 연속값 언어 모델의 탐색과 활용을 조율합니다.

- **Performance Highlights**: DiTAR는 제로샷 텍스트 투 스피치(zero-shot text-to-speech) 작업에서 뛰어난 성능을 발휘하며, 특히 강인성(robustness), 화자 유사성(speaker similarity), 자연스러움(naturalness) 분야에서 최첨단(SOTA) 성과를 기록했습니다. 이러한 성과는 기존의 모델들이 요구하는 계산량보다 현저히 낮은 비용으로 이루어졌습니다. 특히, 여러 단계를 거치는 복잡한 파이프라인 대신, DiTAR는 언어 모델이 최종 특성을 직접 예측하는 단순화된 방식을 적용하여 우수한 결과를 도출했습니다.



### Adaptive Semantic Prompt Caching with VectorQ (https://arxiv.org/abs/2502.03771)
- **What's New**: 이번 연구는 기존의 정적 유사도 임계값(static threshold) 접근 방식이 다양한 프롬프트(prompt)를 효과적으로 분류하는 데 부족하다는 것을 보여줍니다. 사용자는 이제 VectorQ라는 새로운 프레임워크를 통해 임베딩(embedding) 별로 특정한 임계값 지역(threshold regions)을 학습하여, 각 임베딩의 복잡성과 불확실성에 적응할 수 있습니다. 이는 LLM(generated response) 응답 재사용의 정확도를 높이고, 비용을 절감할 수 있는 기회를 제공합니다.

- **Technical Details**: VectorQ는 후보(candidate)와 가장 가까운 이웃(nearest neighbor) 및 해당 이웃의 캐시된 LLM 응답, 그리고 이와 관련된 임계값 지역을 바탕으로 작동합니다. 시스템은 후보와 가장 가까운 이웃 간의 유사성을 분석하여 캐시 응답 재사용 여부를 결정합니다. 이 프레임워크는 베이지안 샘플링(Bayesian sampling) 기반의 접근 방식을 사용하여, 가장 불확실한 지역을 우선하여 재평가하고, 잘못된 캐시 적중을 최소화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: VectorQ는 다양한 데이터 세트에 대한 평가를 통해 기존의 정적의 임계값보다 12배 더 높은 캐시 적중(rate)과 최대 92%의 오류율 감소를 기록했습니다. 이는 LLM 응답 생성을 위한 비용을 대폭 절감할 수 있음을 의미합니다. 결과적으로 VectorQ는 최신 시스템에 비해 모든 정적 임계값에서 일관되게 우수한 성능을 보여주며, 향후 메모리 관리와 프롬프트 캐싱에 대한 새로운 방향성을 제시합니다.



### DocMIA: Document-Level Membership Inference Attacks against DocVQA Models (https://arxiv.org/abs/2502.03692)
Comments:
          ICLR 2025

- **What's New**: 이 논문에서는 Document Visual Question Answering (DocVQA) 모델을 겨냥한 신종 멤버십 추론 공격(Membership Inference Attack, MIA) 두 가지를 소개합니다. 이는 모델 아키텍처와 매개변수에 대한 완전한 접근 권한을 가진 화이트박스(white-box) 설정과, 모델의 출력만을 이용하는 블랙박스(black-box) 설정을 위해 설계되었습니다. 공격자는 추가 데이터에 접근할 수 없는 상황을 가정하였으며, 이는 더 현실적인 시나리오임에도 불구하고 더 많은 도전 과제를 제공합니다.

- **Technical Details**: 제안된 공격 방법인 Document-level Membership Inference Attack (DocMIA)은 동일 문서가 훈련 세트에 여러 번 등장하는 상황에서 발생하는 문제를 해결합니다. 자동 회귀(auto-regressive) 출력에서 logit 기반의 전통적인 지표를 추출하는 것이 어려운 점을 극복하기 위해, DocMIA를 위한 구별 가능한(feature discriminative) 메트릭을 생성하는 새로운 방법론을 제안합니다. 이 연구는 기존 모델에 대한 보편적인 공격 기술 외에도, 화이트박스 및 블랙박스 환경에서 추가 데이터 없이 수행 가능합니다.

- **Performance Highlights**: 세 가지 멀티모달 DocVQA 모델과 두 개의 데이터셋에서 비교 평가를 수행한 결과, 본 연구의 공격 방법은 기존의 최첨단 멤버십 추론 공격을 능가하는 성능을 보여준 바 있습니다. 특히, 세 가지 접근 방식(가장 간단한 레이어 미세 조정(Fine-tuning), LoRA 방식을 적용한 미세 조정, 이미지 그래디언트 사용)을 통해 DocMIA의 효과를 입증했습니다. 이와 같은 혁신적인 접근법은 DocVQA 모델의 개인 정보 보호 위험을 강조하면서, 해당 분야의 기존 연구와 차별화되는 점을 부각시킵니다.



### REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations (https://arxiv.org/abs/2502.03629)
- **What's New**: 기존의 이미지 편집 모델들이 실제 사용자 요구를 충족시키지 못하는 문제를 다룬 REALEDIT(Real Edit)라는 새로운 대규모 데이터셋을 소개합니다. 이 데이터셋은 Reddit에서 수집된 진짜 사용자 요청과 인간에 의해 편집된 이미지를 포함하고 있으며, 9300개의 평가 예제를 포함하고 있어 다양한 실제 요구를 테스트할 수 있습니다. REALEDIT는 인간의 편향을 감소시키고 다양한 사용자 요구를 반영하는 구조로 설계되었습니다. 이 연구의 결과는 기존 모델이 이러한 작업에서 부족한 점이 있음을 강조합니다.

- **Technical Details**: REALEDIT 데이터셋은 사람에 의해 편집된 이미지와 그에 대한 요청을 기반으로 한 48K의 훈련 예제와 9300개의 테스트 예제를 포함하고 있습니다. 이를 위해 데이터 수집 파이프라인을 구성하였으며, 두 개의 주요 서브레딧인 r/PhotoshopRequest와 r/estoration에서 받은 요청을 기반으로 하여 데이터를 구성하였습니다. 편집된 이미지들은 원본 이미지와 편집 지침을 포함하여 사용자의 실제 요청을 반영하는 형태로 수집되었습니다. 이 데이터셋은 진짜 편집 요구 사항을 더 효과적으로 반영하며, 기존의 합성 데이터세트의 한계를 극복하고자 합니다.

- **Performance Highlights**: REALEDIT 모델은 기존의 최고 성능 모델보다 165포인트 높은 Elo 스코어를 기록하며 주목받았습니다. 또한 VIEScore와 같은 자동화된 메트릭에서도 92%의 향상을 보여줍니다. 모델은 Reddit에서 새로운 요청에 대해 긍정적인 피드백을 받았으며, 편집 외에도 진짜 편집된 이미지의 탐지 성능이 향상될 가능성도 확인되었습니다. 이 연구는 이미지 편집 작업 외에도 다양한 AI 기반 응용 분야에 대한 데이터셋의 유용성을 강조합니다.



### An Empirical Exploration of ChatGPT's Ability to Support Problem Formulation Tasks for Mission Engineering and a Documentation of its Performance Variability (https://arxiv.org/abs/2502.03511)
Comments:
          10 pages, 3 figures, submitted to Conference on Systems Engineering Research (CSER)

- **What's New**: 이 논문은 시스템 공학(Systems Engineering, SE)과 미션 엔지니어링(Mission Engineering, ME) 내에서 생성적 인공지능(generative AI)의 활용 가능성을 탐구합니다. 특히, 메가 시스템 관점에서 문제를 formulat(e)하는 데 있어서 AI의 역할에 주목하고 있습니다. 주목할 만한 것은 NASA 우주 임무 설계 과제를 사례로 들어, ChatGPT-3.5의 이해관계자(stakeholder) 식별 능력을 평가하고 있다는 점입니다.

- **Technical Details**: 본 연구에서는 Large Language Models (LLMs)가 ME 문제 formulat(e) 작업을 지원하는 품질과 일관성을 분석합니다. 구체적으로는 이해관계자 식별 작업에서의 성능을 여러 차례 병행하여 시험하고, 출력물의 품질과 변동성을 질적으로 평가했습니다. 결과적으로, LLM은 사람 중심의 이해관계자를 잘 식별하지만 외부 시스템이나 환경 요인 식별에서는 낮은 성과를 보였습니다.

- **Performance Highlights**: 연구 결과, LLM의 출력물은 일관성이 부족하고, 문제 formulat(e)에는 적합하지 않은 솔루션 특정(output) 경향이 있음을 확인했습니다. 그럼에도 불구하고, ChatGPT는 일부 전문가의 업무 부담을 줄일 수 있는 가능성을 보여주었습니다. 그러나 다양한 병행 시도에서 출력물 간의 큰 변동성을 관찰하여, LLM 사용 시 주의를 기울여야 함을 강조했습니다.



### Teaching Language Models to Critique via Reinforcement Learning (https://arxiv.org/abs/2502.03492)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 출력 품질을 비판하고 개선하기 위해, 인간의 개입 없이도 피드백을 생성하도록 비평가 모델을 훈련하는 프레임워크인 CTRL (Critic Training via Reinforcement Learning)을 제안합니다. CTRL 프레임워크를 통해 코드 생성 과정에서 비평가가 생성자 모델의 성능을 크게 향상시킬 수 있음을 발견했습니다. 특히, 비평가 모델은 더 강력한 생성 모델과 결합하여 놀라운 성능 향상을 이끌어낼 수 있는 능력을 보여주었습니다.

- **Technical Details**: CTRL 프레임워크는 critic 모델을 task-performing 모델에서 분리하여, 비평이 반복되는 과정에서 최적의 솔루션 생성을 유도합니다. 이 과정에서, Group Relative Policy Optimization (GRPO)을 통해 두 단계의 훈련 파이프라인을 구현합니다. 이 방법은 주어진 코드 생성 문제에서 비평가가 직접적인 피드백을 제공하고, 이를 통해 더 나은 솔루션으로 나아갈 수 있도록 돕습니다.

- **Performance Highlights**: CTRL을 사용한 훈련은 CodeContests, LiveCodeBench, MBPP+, JudgeBench 등 여러 벤치마크에서 자가 비평 방법이나 기존의 강력한 비평 모델보다 눈에 띄게 우수한 성과를 나타냈습니다. 이러한 결과는 상대적으로 약한 비평가 모델이 강력한 생성 모델을 효과적으로 안내할 수 있는 가능성을 보여줍니다. 추가적으로, CTRL은 iterative critique-revision을 통해 테스트 시간 성능을 개선할 수 있으며, CodeContests 벤치마크에서 106.1%의 상대적 향상을 이루어냈습니다.



New uploads on arXiv(cs.IR)

### Counterfactual Query Rewriting to Use Historical Relevance Feedback (https://arxiv.org/abs/2502.03891)
- **What's New**: 이 연구에서는 사용자 쿼리의 역사적 관련 피드백(historical relevance feedback)을 활용하여 쿼리를 재작성하는 접근 방식을 제안합니다. 이전에 관련성이 있었던 문서들을 새로운 쿼리의 강도를 높이는 신호로 사용하며, 이를 통해 다양한 시나리오에서 향상된 검색 결과를 제공합니다. 특히, 이 방법이 복잡한 transformer 기반 접근 방식보다 더 효과적임을 보여 주었습니다.

- **Technical Details**: 사용자 쿼리를 수정하기 위한 세 가지 접근 방식이 제안되었습니다: (1) boosting, (2) 명시적 관련 피드백(explicit relevance feedback), (3) keyqueries. 이 과정에서 과거의 관련성을 조건부로 가정하여 문서 평가를 진행하며, 각 접근법은 진화하는 문서 집합에서 더욱 효과적으로 작동합니다. 또한, 이러한 방법들은 과거 데이터에서 파생된 tf-idf 점수를 기반으로 쿼리를 확장하는 방식으로 적용됩니다.

- **Performance Highlights**: CLEF LongEval 시나리오에서 수행한 평가 결과, 역사적 관련 피드백을 이용한 쿼리 재작성 방식이 boosting과 동등한 효과를 나타냈지만, 새로운 문서에 대해서도 일반화가 가능하다는 장점을 가지고 있습니다. 이 접근 방식은 neural transformers를 크게 능가하고, 사전 계산된 쿼리를 사용하기 때문에 효율성 또한 매우 뛰어납니다.



### Boosting Knowledge Graph-based Recommendations through Confidence-Aware Augmentation with Large Language Models (https://arxiv.org/abs/2502.03715)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)을 활용하여 추천 시스템을 위한 지식 그래프(KG)의 품질과 관련성을 향상시키기 위한 새로운 프레임워크인 CKG-LLMA(Confidence-aware KG-based Recommendation Framework with LLM Augmentation)를 제안합니다. CKG-LLMA는 KG를 고품질 정보로 풍부하게 하는 LLM 기반 서브그래프 증강기, 노이즈 트리플을 필터링하는 신뢰 기반 메시지 전파 메커니즘, 사용자-아이템 상호작용과 KG 데이터를 통합하는 이중 뷰 대조 학습 방법을 포함합니다.

- **Technical Details**: CKG-LLMA의 구조는 LLM의 도움을 받아 KG 기반 추천 시스템 내의 대조 학습, 메시지 전파 및 서브그래프 증강을 결합하여 설계되었습니다. 이 프레임워크는 비록 기존 KG에서 발생하는 노이즈와 구식 관계들을 해결하고, 사용자 및 아이템 표현을 강화하기 위해 두 개의 뷰를 사용하는 대조 학습 방법을 채택합니다. 또한, LLM을 통해 사용자 행동에 대한 설명을 생성하는 과정을 통해 추천 시스템의 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, CKG-LLMA는 여러 공개 데이터 세트에서 다양한 기초 모델보다 우수한 성능을 보였습니다. 우리의 모델은 신뢰할 수 있고 설명 가능하며, 사용자에게 유익한 추천 결과를 생성하는 능력을 보여주었습니다. 특히, LLM의 도입을 통해 KG 기반 추천 시스템의 전반적인 성능을 크게 향상시켰습니다.



### Contrastive Learning for Cold Start Recommendation with Adaptive Feature Fusion (https://arxiv.org/abs/2502.03664)
- **What's New**: 이 논문에서는 대조 학습(contrastive learning)을 통합한 콜드 스타트 추천 모델을 제안합니다. 콜드 스타트(Cold Start) 시나리오에서 사용자 및 항목 상호작용 데이터가 부족하여 추천 시스템의 성능 저하 문제를 해결하는 것을 목표로 합니다. 이 모델은 적응형 특성 선택 모듈을 통해 주요 특성의 가중치를 동적으로 조정하며 추천 성능을 향상시킵니다.

- **Technical Details**: 모델은 사용자 속성(user attributes), 항목 메타 정보(item meta-information), 그리고 맥락적 특징(contextual features)을 효과적으로 통합하기 위해 다중 모달(feature fusion) 메커니즘을 결합합니다. 대조 학습 메커니즘을 도입하여 양성 및 음성 샘플 쌍을 구성함으로써 특성 표현의 강인성과 일반화 능력을 향상시킵니다. 실험은 MovieLens-1M 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 제안된 모델은 HR, NDCG, MRR, Recall 등에서 Matrix Factorization, LightGBM, DeepFM, AutoRec와 같은 주류 추천 방법보다 현저히 우수한 성능을 보였으며, 특히 콜드 스타트 시나리오에서 두드러진 효과를 나타냈습니다. Ablation 실험을 통해 각 모듈의 모델 성능 향상에 대한 중요성을 검증하였고, 학습률 민감도 분석에서는 적절한 학습률이 모델의 최적화 효과에 필수적이라는 것을 보여주었습니다.



### Digital Gatekeeping: An Audit of Search Engine Results shows tailoring of queries on the Israel-Palestine Conflic (https://arxiv.org/abs/2502.04266)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 사용자 맞춤화가 검색 엔진 결과에 미치는 영향을 심층적으로 분석하였으며, 이스라엘-팔레스타인 분쟁과 같은 논란이 많은 주제에 대한 첫 번째 실증 연구를 제시했습니다. DuckDuckGo, Google, Yahoo와 같은 주요 검색 엔진에서의 결과를 감사하기 위해 개인 정보 보호를 고려한 도구를 개발했습니다. 연구 결과, 사용자 위치와 브라우징 이력이 검색 결과를 상당히 맞춤화했고, 특히 분쟁 관련 쿼리에서 더 강하게 나타났습니다.

- **Technical Details**: 이 연구에서는 자동화된 크롤링 방법을 사용하여 검색 엔진의 결과를 분석했습니다. Type 1, Type 2, Type 3의 세 가지 봇 유형을 사용하여 사용자 프로필을 단계적으로 확장하며, 이를 통해 위치, 브라우저 언어 및 브라우징 이력을 포함한 다양한 변수를 조작했습니다. 분석 방법론으로는 Rank-Biased Overlap (RBO) 등을 활용하여 검색 결과의 변화를 세심하게 측정하였습니다.

- **Performance Highlights**: 연구 결과, 이스라엘-팔레스타인 분쟁과 같은 민감한 주제에 대해서는 검색 엔진의 맞춤화가 예전 연구보다 훨씬 더 높았습니다. Google과 DuckDuckGo는 브라우징 이력에 기반한 개인화가 제한적이라는 주장을 했지만, 결과적으로는 significant한 차이를 보였으며, 반면 Yahoo는 상대적으로 낮은 개인화를 보여주었습니다. 이러한 결과는 검색 엔진의 알고리즘이 실제로 어떻게 작동하는지에 대한 의문을 제기하며, 투명성 부족 문제를 강조합니다.



### MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation (https://arxiv.org/abs/2502.04176)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 Multimodal Retrieval-Augmented Multimodal Generation (MRAMG)이라는 새로운 태스크를 도입하여, 텍스트와 이미지를 결합한 답변 생성을 목표로 하고 있습니다. 기존의 Retrieval-Augmented Generation (RAG) 방법들이 텍스트 기반의 출력에 집중했던 반면, MRAMG는 다양한 모드의 정보를 최대한 활용하는 데 중점을 두고 있습니다. 논문에서는 또한 MRAMG 성능을 평가하기 위한 MRAMG-Bench라는 새로운 벤치마크를 제안합니다.

- **Technical Details**: MRAMG-Bench는 총 4,346개의 문서와 14,190개의 이미지, 4,800개의 질문-답변(QA) 쌍으로 구성된 데이터셋입니다. 이 데이터셋은 웹 데이터, 학술 논문, 라이프스타일의 세 가지 카테고리에서 수집되었습니다. 새로운 벤치마크는 LLMs와 MLLMs의 성능을 rigorously(엄격히) 평가할 수 있는 통계적 및 LLM 기반 홀드를 벤치마크에 통합하고 있습니다.

- **Performance Highlights**: 아울러, 본 연구는 텍스트와 이미지 모두를 생성할 수 있는 효율적인 멀티모달 답변 생성 프레임워크를 제안하여 기존의 LLM 기반 접근법에 대한 한계를 극복하고 있습니다. MRAMG-Bench를 통해 11개의 고급 생성 모델의 성능을 포괄적으로 평가한 결과, 각 모델의 능력과 제한 사항에 대한 귀중한 통찰을 제공합니다.



### LLM Alignment as Retriever Optimization: An Information Retrieval Perspectiv (https://arxiv.org/abs/2502.03699)
Comments:
          26 pages

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 정렬(alignment)을 향상시키기 위한 새로운 직접 최적화 방법인 LarPO(LLM Alignment as Retriever Preference Optimization)를 소개합니다. 기존의 RL(강화 학습) 기반 접근법의 복잡성을 해결하기 위해, 정보 검색(IR) 원리를 활용하여 LLM과 IR 방법론을 연결하는 시스템적인 프레임워크를 제안합니다. 이를 통해 LLM의 생성 모델과 보상 모델을 IR의 검색기-재정렬기(retriever-reranker) 패러다임에 매핑합니다.

- **Technical Details**: LarPO는 LLM의 선호도를 직접 최적화하는 새로운 방법으로, 정보 검색의 세 가지 핵심 원리인 검색기 최적화 목적, 하드 네거티브 마이닝(hard negative mining), 후보 목록 구축(candidate list construction)을 활용하여 정렬 품질을 개선합니다. LLM은 검색기의 역할을 하고, 보상 모델은 재정렬기의 역할을 수행함으로써 효과적인 정렬 성능을 구현합니다. 이러한 접근은 LLM의 생성 및 IR 검색기에서 일반적으로 사용되는 이중 인코더 아키텍처를 활용하여 성과를 도출합니다.

- **Performance Highlights**: 실험 결과, LarPO 방법은 AlpacaEval2에서 38.9%와 MixEval-Hard에서 13.7%의 평균적인 성능 향상을 보여주었습니다. 이러한 결과는 LarPO의 효과를 증명하며, 정보 검색 기법이 LLM 정렬의 품질을 높일 수 있는 새로운 가능성을 제시합니다. 본 연구는 LLM 정렬과 정보 검색 간의 중요한 연결고리를 확립함으로써, 이 분야의 발전을 위한 실용적인 방법을 제공합니다.



### Can Cross Encoders Produce Useful Sentence Embeddings? (https://arxiv.org/abs/2502.03552)
- **What's New**: 본 연구에서는 Cross Encoder (CE)로부터 추출한 초기 레이어의 hidden states가 정보 검색(Information Retrieval)에서 유용하게 활용될 수 있다는 흥미로운 발견을 보고합니다. 일반적으로 Dual Encoder (DE)가 협코딩 방식으로 메모리를 세분화하여 정보를 추출하도록 훈련되지만, CEs의 초기 레이어에서 얻은 임베딩이 DE보다 정보 검색에 더 효과적임을 보여주고 있습니다. 이로 인해 CE를 활용하여 더 가벼운 DE를 생성하고, 추론 속도를 5.15배 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 본 논문에서는 CE에서 문장 쌍을 위한 implicit한 임베딩을 추출하는 방법을 설명합니다. CE의 각 레이어에서 token을 평균 풀링(mean pooling)하여 임베딩을 생성하며, 초기 레이어의 가중치를 DE 모델에 주입하여 새로운 지식 주입(knowledge infusion) 모델을 구축합니다. 제안하는 DE-2 CE 모델은 기존 DE 모델과 비교하여 최대 1%의 정확도 차이를 보이며, 추론 시간이 약 5배 더 빠른 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, CE의 초기 레이어 hidden states가 DE의 첫 번째 레이어보다 정보 검색에 더 강한 신호를 전달한다는 것을 확인했습니다. 총 12개의 다양한 데이터셋을 활용하여 IR 성능을 측정했으며, CE 기반의 지식 주입이 정보 검색에서 유용하다는 것을 입증했습니다. 최종적으로, 제안된 DE-2 CE 모델은 유사한 DE 모델에 비해 성능 저하 없이 현저히 빠른 추론 속도를 기록함으로써 효율성을 높였습니다.



New uploads on arXiv(cs.CV)

### SMART: Advancing Scalable Map Priors for Driving Topology Reasoning (https://arxiv.org/abs/2502.04329)
Comments:
          Accepted by ICRA 2025. Project page: this https URL

- **What's New**: 이 논문은 자율 주행에서의 lane topology(차선 구조) 추론의 중요성을 강조하며, 기존의 센서 데이터에 의존하지 않고 스케일이 가능한 프로그램인 SMART를 제안합니다. SMART는 스탠다드 디피니션(SD) 및 위성 지도를 활용하여 매핑 사전 모델을 학습하고, 대규모의 HD(고해상도) 지도에 의해 감독됩니다. 이러한 접근은 자율 주행 및 운전 보조 시스템을 위한 도로 요소의 구조와 연결 관계 이해를 향상시킵니다.

- **Technical Details**: SMART는 두 단계의 드라이빙 토폴로지 추론 파이프라인을 특징으로 합니다. 첫 번째 단계에서는 SD 및 위성 지도를 사용하여 lane topology를 추론하는 맵 프라이어 모델을 도입하고, 두 번째 단계에서는 이 모델을 기존의 온라인 토폴로지 추론 모델과 통합합니다. 이러한 두 단계의 프로세스는 고품질 센서 데이터에 대한 의존성을 줄이고 매우 접근 가능한 지리공간 맵을 통해 적응 가능한 맵 표현의 학습을 가능하게 합니다.

- **Performance Highlights**: SMART를 단독으로 사용했을 때도 기존 오프라인 차선 토폴로지 이해에 있어 뛰어난 성능을 보였으며, SD 및 위성 입력만 이용하여 오픈레인(OpenLane-V2) 기준에서 성능이 최대 28% 향상되었습니다. 이러한 성과는 SMART의 강력한 훈련으로 이루어진 보편성이 높은 맵 프라이어가 온라인 토폴로지 추론의 일반화를 크게 향상시킬 수 있음을 보여줍니다.



### Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignmen (https://arxiv.org/abs/2502.04328)
- **What's New**: 최근 GPT-4o의 발전에 따라 다중 모달 언어 모델에 대한 관심이 증가하고 있습니다. 이러한 배경에서 소개된 Ola 모델은 이미지, 비디오 및 오디오를 이해하는데 있어 전문적인 모델들과 경쟁할 수 있는 성능을 달성했습니다. Ola는 언어 모델의 지원 모달리티를 점진적으로 확장하는 Progressive Modality Alignment 전략이 핵심 설계로 도입되었습니다.

- **Technical Details**: Ola 모델은 초기 이미지와 텍스트의 기본 모달리티에서 시작하여 모델의 스킬 세트를 점진적으로 확장합니다. 이는 음성 데이터와 비디오는 모든 모달리티의 연결을 강화하기 위한 중요한 단계입니다. 또한, Ola는 다양한 모달리티를 처리할 수 있는 아키텍처를 지원하며, 고품질 음성 합성을 위한 Sentence-wise Streaming Decoding 모듈을 통합하여 실시간으로 사용자와 상호작용할 수 있습니다.

- **Performance Highlights**: Ola는 7억 개의 매개변수를 통해 이미지, 비디오 및 오디오 벤치마크에서 경쟁력 있는 성능을 보입니다. OpenCompass 기준에서 평균 정확도 72.6%, VideoMME 기준에서 68.4%의 인상적인 정확성을 달성하며, 오디오 이해 과제에서도 평균 WER 3.1을 기록했습니다. 이러한 결과는 기존의 오픈 옴니 모달 LLM을 크게 초과하며, 최신 전문 LLM과 비슷한 규모의 모델 대비 우수한 성능을 입증합니다.



### WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs (https://arxiv.org/abs/2502.04326)
- **What's New**: 이 논문에서는 시각, 청각, 텍스트 입력을 동시에 평가하는 첫 번째 멀티모달 비디오 이해 벤치마크인 WorldSense를 소개합니다. 기존 벤치마크와 달리, WorldSense는 오디오와 비디오의 긴밀한 연결을 강조하고 다양하고 양질의 비디오 및 다중 선택 QA 쌍을 통해 종합적인 평가를 가능하게 합니다. 이를 통해 실제 세계 맥락에서의 멀티모달 이해 능력을 평가할 수 있는 플랫폼을 제공합니다.

- **Technical Details**: WorldSense는 1662개의 오디오-비디오 동기화 비디오와 3172개의 다중 선택 질문으로 구성되어 있습니다. 이 자료들은 8개의 주요 도메인과 67개의 세분화된 하위 범주로 체계적으로 분류되어 있으며, 질문은 음성 및 비주얼 정보의 동시 처리를 요구하여 모델의 멀티모달 처리 능력을 rigorously 평가합니다. QA 쌍은 80명의 전문가에 의해 수차례 검증된 고품질 주석을 통해 보장됩니다.

- **Performance Highlights**: 전반적인 실험 결과, 기존 모델들이 실제 세계의 맥락을 이해하는 데 어려움을 겪고 있음을 보여줍니다. 오픈소스 비디오-오디오 모델은 약 25%의 정확성을 보이는 반면, Gemini 1.5 Pro와 같은 상용 모델은 48%의 정확성을 달성하지만 단일 모드에서 성능이 15% 감소합니다. 이러한 결과는 WorldSense의 모달리티 간 강한 결합을 강조하며, 실제 세계 이해 능력에서 상당한 격차가 있음을 드러냅니다.



### ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features (https://arxiv.org/abs/2502.04320)
- **What's New**: 이 논문에서는 multi-modal diffusion transformers(DiT)가 가지는 데이터 표현이 해석 가능성을 증대시키는 독특한 속성을 가지고 있는지를 탐구합니다. 이를 위해 저자들은 ConceptAttention이라는 새로운 방법을 소개하며, 이 방법은 DiT의 attention layers에서 생성된 saliency maps를 활용하여 이미지 내 텍스트 개념을 정확하게 위치시키는 역할을 합니다. 기존의 cross-attention 메커니즘에 비해 선형 프로젝션을 통해 더 선명한 saliency maps를 생성할 수 있다는 주요 발견을 제공합니다.

- **Technical Details**: ConceptAttention은 DiT의 attention layers의 파라미터를 재사용하여 매우 맥락화된 concept embeddings를 생성하는 방식을 사용합니다. 특히 각 텍스트 개념은 그에 해당하는 시각적 요소(예: '용', '태양')와 연결될 수 있습니다. 이 방법은 추가 학습을 필요로 하지 않으며, 주어진 이미지와 concept embeddings 간의 선형 프로젝션을 수행하여 고품질의 saliency maps를 생성합니다.

- **Performance Highlights**: ConceptAttention은 제로샷(Zero-shot) 이미지 분할 작업에서 최고 성능을 달성하며, ImageNet-Segmentation 데이터셋과 Pascal VOC의 단일 클래스 서브셋에서 총 11가지의 다른 제로샷 해석성 방법을 초월하는 성과를 입증합니다. 이러한 성과는 DiT 모델의 표현이 분할과 같은 비전 과제로의 이전 가능성을 잘 보여줍니다. 저자들은 ConceptAttention을 통해 연구자들이 텍스트-이미지 생성 모델의 복잡한 역학 관계를 해석하고 탐구할 수 있도록 하는 기회를 제공합니다.



### sshELF: Single-Shot Hierarchical Extrapolation of Latent Features for 3D Reconstruction from Sparse-Views (https://arxiv.org/abs/2502.04318)
Comments:
          Joint first authorship

- **What's New**: 이 논문에서는 sshELF라는 새로운 빠른 단일 샷 파이프라인을 소개합니다. 이는 드문 외향 카메라 뷰에서의 3D 장면 복원을 위한 계층적 특징 외삽(hierarchical extrapolation of latent features) 방법을 활용합니다. 기존의 방식들이 놓쳤던 전경과 후경 간의 정보 교환을 가능하게 하여 보이지 않는 영역의 복원을 개선합니다.

- **Technical Details**: sshELF는 두 개의 주요 단계로 나뉘며, 첫 번째 단계는 가상 뷰를 생성하는 백본(backbone) 네트워크이고, 두 번째 단계는 참조 및 가상 뷰를 명시적 Gaussian primitives로 해석하는 변환자(translator)입니다. 이러한 구조적 분리는 훈련 중 연산 요구 사항을 줄이고 더 많은 정보를 전달할 수 있도록 합니다. 또한, 사전 훈련된 기초 모델을 통합하여 깊이 예측과 텍스처를 공동 추론할 수 있도록 했습니다.

- **Performance Highlights**: sshELF는 여섯 개의 희소 입력 뷰에서 360도 장면을 복원하며, 합성 및 실제 데이터 세트에서 경쟁력 있는 결과를 보여줍니다. 이 방법은 빠른 실시간 렌더링을 지원하며, 보이지 않는 지역의 재구성이 가능하여 downstream 응용에서도 풍부한 잠재적 특성을 제공합니다. 또한, 기존 방법들보다 더 나은 공간적 의미와 점유, 기하학에 대한 통찰력을 얻을 수 있습니다.



### Factorized Implicit Global Convolution for Automotive Computational Fluid Dynamics Prediction (https://arxiv.org/abs/2502.04317)
- **What's New**: 이 연구에서는 자동차 설계를 위한 Computational Fluid Dynamics (CFD) 문제를 해결하기 위해 새로운 아키텍처인 Factorized Implicit Global Convolution (FIGConv)를 제안합니다. 기존의 3D 신경망 CFD 모델들은 계산 복잡도가 O(N^3)으로 높은 해상도의 3D 데이터를 처리하는 데 어려움을 겪고 있는 반면, FIGConv는 O(N^2)으로 계산 효율적인 개선을 제공합니다. 이를 통해 대규모 3D 메쉬를 효과적으로 처리할 수 있습니다.

- **Technical Details**: FIGConv는 높은 해상도를 근사하기 위해 Factorized Implicit Grids를 사용할 뿐만 아니라, 2D 재매개변화를 통한 효율적인 전역 합성을 결합하여 효과적인 정보 수집 및 통합을 위한 U-형 아키텍처를 구현합니다. 이로 인해 기존 방법들에 비해 약 40%의 상대 평균 제곱 오차 감소와 70%의 절대 평균 제곱 오차 감소를 달성합니다. 주어진 데이터셋에서 FIGConv는 드래그 예측을 위해 0.95의 R^2 값을 기록하며, 이는 이전의 최고 성능 모델보다 현저한 개선을 보여줍니다.

- **Performance Highlights**: 본 연구에서는 DrivAerNet과 Ahmed body 데이터셋에서 FIGConv의 성능을 검증하며, 드래그 계수와 표면 압력을 예측하는 작업에서 기존 방법보다 훨씬 빠른 속도를 자랑합니다. 두 데이터셋 모두에서 FIGConv는 최신 기술을 초월하는 성능을 보였으며, CFD 문제 해결을 위한 새로운 기준을 제시합니다. 이러한 결과는 자동차 산업에서의 고해상도 유체 역학 시뮬레이션의 가능성을 크게 확장합니다.



### MotionCanvas: Cinematic Shot Design with Controllable Image-to-Video Generation (https://arxiv.org/abs/2502.04299)
Comments:
          It is best viewed in Acrobat. Project page: this https URL

- **What's New**: 본 논문은 이미지에서 비디오로의 생성(context of image-to-video generation) 과정에서 작업자가 시네마틱 비디오 샷을 설계할 수 있는 방법인 MotionCanvas를 제안합니다. 해당 시스템은 사용자 주도의 제어를 통합하여 장면 인식을 통해 객체와 카메라의 움직임을 동시에 조정할 수 있게 합니다. 고전적 컴퓨터 그래픽스와 현대 비디오 생성 기술의 통찰을 연결하여 비싼 3D 학습 데이터를 필요로 하지 않고도 3D 인식 모션 제어를 가능하게 합니다.

- **Technical Details**: MotionCanvas는 세 가지 주요 구성요소로 이루어져 있습니다: (1) 다양한 장면 인식 모션 의도를 포착하는 모션 디자인 모듈, (2) 이러한 의도를 화면 공간 모션 신호로 변환하는 변환 모듈, (3) 모션이 조건화된 비디오 생성 모델입니다. 사용자로부터의 3D 장면 공간에서의 모션 계획을 효율적으로 변환하여 비디오 생성 모델로 전달할 수 있도록 Motion Signal Translation 모듈을 설계하였습니다. 이를 통해 작업자는 장면 공간에서 모션을 구상하고, 해당 의도를 해석하여 비디오 생성 프로세스에 적합한 신호를 생성할 수 있습니다.

- **Performance Highlights**: MotionCanvas는 다양한 실사 이미지와 샷 디자인 시나리오에서 그 효과를 입증하며, 디지털 콘텐츠 제작에서의 창의적인 작업 흐름을 개선하는 잠재력을 보여줍니다. 사용자들은 최종 비디오 소재에 대한 자신의 창의적 비전을 표현하기 위해 카메라 이동 방식 및 객체 배열을 조절할 수 있습니다. 이는 시네마틱 샷 디자인을 통합한 이미지에서 비디오로의 생성 작업에서 유의미한 향상을 가져올 수 있음을 나타냅니다.



### GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation (https://arxiv.org/abs/2502.04293)
- **What's New**: 이 논문에서는 GCE-Pose라는 새로운 방법론을 제안하여, 모델 프리(category-level) 객체 포즈 추정에서 불완전한 관측에도 불구하고 범주 간의 컨텍스트 정보를 효과적으로 활용합니다. 특히, Semantic Shape Reconstruction (SSR) 모듈을 통해 입력된 부분 RGB-D 객체의 글로벌 기하학 및 의미를 재구성하여 더 나은 추정을 가능하게 합니다. 또한, Global Context Enhanced (GCE) 기능 융합 모듈을 도입하여 부분 관측데이터와 재구성된 글로벌 컨텍스트를 효과적으로 통합합니다.

- **Technical Details**: GCE-Pose의 주요 구성 요소는 두 가지로, SSR 모듈과 GCE 기능 융합 모듈이 있습니다. SSR 모듈은 입력된 부분 포인트를 완전한 형태로 재구성하면서 카테고리별 3D 의미 프로토타입을 매끄럽게 통합합니다. GCE 기능 융합 모듈은 재구성된 글로벌 컨텍스트를 지역적 정보와 효과적으로 융합하여 포즈 추정을 개선합니다.

- **Performance Highlights**: 실험 결과, GCE-Pose는 HouseCat6D 및 NOCS-REAL275와 같은 복잡한 실제 데이터셋에서 기존 방법들보다 월등한 성능을 보였습니다. 새로운 인스턴스에 대한 포즈 추정의 강력함을 입증하였으며, 특히 형태 변형이나 차폐가 발생하는 상황에서도 안정적인 결과를 달성했습니다. 이러한 성능 향상은 글로벌 컨텍스트 우선 정보와 GCE 융합 모듈의 효과를 잘 나타냅니다.



### Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances (https://arxiv.org/abs/2502.04268)
Comments:
          11 pages, 5 figures, 10 tables

- **What's New**: 오리엔티드 객체 탐지(ood)는 자율주행, 항공 이미지, 소매 장면 등에서 필수적인 작업으로 자리잡고 있다. 최근 RBox 라벨링의 비용이 높아지는 현실 속에서, 본 연구는 포인트 주도의 OOD를 통해 효율적으로 객체 감지를 수행할 수 있는 Point2RBox-v2를 소개한다. 이 방법은 객체 간의 공간적 레이아웃을 활용하여 새로운 시너지를 창출하는 데 초점을 맞추고 있다.

- **Technical Details**: Point2RBox-v2의 핵심 원리는 3가지 손실 함수에 기반한다: 첫째, Gaussian overlap loss로 객체를 2D Gaussian 분포로 보고 겹침을 최소화하여 각 인스턴스의 상한을 학습한다. 둘째, Voronoi watershed loss는 Voronoi 분할을 통해 각 인스턴스의 하한을 학습한다. 셋째, consistency loss를 통해 입력 이미지와 증강된 뷰 사이의 크기 및 회전 변화를 학습한다.

- **Performance Highlights**: Point2RBox-v2는 DOTA, HRSC, FAIR1M 벤치마크에서 각각 62.61%, 86.15%, 34.71%의 성능을 기록하며, 밀집된 장면에서 기존 기술보다 우수한 결과를 보인다. 제안된 알고리즘은 경량화되었고, 효율적인 객체 탐지를 제공할 것으로 기대된다. 제공된 코드 또한 연구자들이 사용할 수 있게 공개된다.



### Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion (https://arxiv.org/abs/2502.04263)
Comments:
          Accepted for publication at ICLR 2025

- **What's New**: 이 논문에서는 CLIP와 같은 다중 모달 (multi-modal) Vision-Language Models (VLMs)에서 텍스트와 이미지 인코더를 개별적으로 사용하는 관행이 비효율적임을 보여줍니다. 특히, 이미지 간 유사성 (intra-image similarity) 및 텍스트 간 유사성 (intra-text similarity) 문제를 'intra-modal misalignment'라는 개념으로 정의하여, 이러한 비효율성이 발생하는 원인을 설명합니다.

- **Technical Details**: 저자들은 'modality inversion'을 활용하여 입력 모달리티에서 보완 모달리티로의 표현 변환을 수행합니다. 이를 위해 Optimization-based Textual Inversion (OTI) 및 Optimization-based Visual Inversion (OVI) 기법을 도입하여 고정된 인코더를 사용하여 특징을 변환합니다. 실험을 통해 15개 이상의 데이터셋에서 intra-modal 작업에 대한 inter-modal 방식 접근의 성과 향상을 입증합니다.

- **Performance Highlights**: 실험 결과, inter-modal 접근 방식이 intra-modal 기초 성능을 초과하는 것으로 나타났습니다. CLIP의 인코더 간 intra-modal 유사성을 이용하는 기존의 방법이 성능 저하를 일으킨다는 점을 강조하며, VLM 사전 훈련 과정에서 intra-modal 손실 항을 포함하거나 텍스트와 이미지 임베딩 공간 간의 갭을 줄이면 intra-modal misalignment 문제를 완화할 수 있음을 시사합니다.



### An object detection approach for lane change and overtake detection from motion profiles (https://arxiv.org/abs/2502.04244)
Comments:
          6 pages, 3 figures

- **What's New**: 본 논문은 대시캠 영상을 이용한 차량의 추월 및 차선 변경 조작을 식별하기 위해 새로운 객체 탐지 접근 방식을 제안합니다. 이를 위해 차량 동작을 단일 이미지로 압축한 motion profile을 활용합니다. 기존의 방법들에 비해 낮은 계산 요구 사항으로 실시간 처리와 경량화를 달성할 수 있습니다.

- **Technical Details**: motion profile의 응용은 비디오 스트림을 H×W×T 차원에서 W×T 차원으로 압축하여 사운드 패턴을 식별하는 데 초점을 맞추고 있습니다. 우리가 제안한 YOLOv3 알고리즘은 CoordConv 레이어를 활용하여 객체를 탐지하고, 이 질량의 패턴을 시간적 공간 영역에서 인식합니다. 이러한 아키텍처는 최첨단 성능을 달성하며, 실시간 탐지 성능을 보장합니다.

- **Performance Highlights**: 우리의 모델은 다른 문헌의 기준에 비해 mAP 및 F1 점수에서 우수한 성과를 보이며, computational efficiency 면에서도 뛰어난 성능을 발휘합니다. motion profile 생성 시간과 모델 추론 시간을 보고하여 이 솔루션의 실시간 탐지가 가능함을 증명합니다. 또한 이 연구는 edge-computing 환경에서도 효과적으로 작동할 수 있는 가능성을 보여줍니다.



### Keep It Light! Simplifying Image Clustering Via Text-Free Adapters (https://arxiv.org/abs/2502.04226)
- **What's New**: 이 논문에서는 장애물인 대량의 텍스트 모드와 복잡한 학습 과정을 사용하지 않고도 경쟁력 있는 성능을 달성할 수 있는 심층 클러스터링(deep clustering) 방법론인 SCP(Simple Clustering via Pre-trained models)를 제안합니다. 이 방법은 텍스트 최적화 없이 미리 훈련된 비전 모델의 기능 표현을 활용하여 소규모 클러스터 헤드만 훈련합니다. 이를 통해 실세계 응용 프로그램에 더 쉽게 적용할 수 있는 간단한 클러스터링 파이프라인을 구현합니다.

- **Technical Details**: SCP는 주로 CIFAR-10, CIFAR-20, CIFAR-100, STL-10, ImageNet-10, ImageNet-Dogs와 같은 벤치마크 데이터셋을 이용하여 성능을 실험합니다. SCP의 성능은 기존의 복잡한 방법들과 비교할 때 경쟁력이 있으며, 이 방식은 전통적인 클러스터링 기술보다 훨씬 효율적입니다. 그 이론적 근거로는 이미지와 텍스트 정보가 결합되지 않아도 효율적인 클러스터링이 이루어질 수 있다는 점을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과, SCP는 기존 최첨단 성능(SOTA)과 비슷한 수준의 경쟁력을 보이며, 특히 비전 모델의 대표성만으로도 성능을 발휘할 수 있다는 것을 강조합니다. 이 파이프라인은 표준 L4 GPU에서 실행이 가능하여 널리 사용할 수 있는 가능성을 나타내며, 이러한 접근 방식은 클러스터링 성능 향상에 기여할 것으로 기대됩니다.



### \'Eclair -- Extracting Content and Layout with Integrated Reading Order for Documents (https://arxiv.org/abs/2502.04223)
- **What's New**: 본 논문은 Éclair라는 새로운 다목적 텍스트 추출 도구를 소개합니다. 이는 OCR(Optical Character Recognition) 기술을 넘어 문서의 구조와 의미적 정보를 이해하여 복잡한 문서에서 고품질의 텍스트를 추출하는 데 중점을 둡니다. Éclair는 여러 페이지에 걸쳐 다양한 형식을 가진 문서들을 처리하는 능력을 갖추고 있으며, 형식화된 텍스트와 그에 대한 경계 상자(bounding box), 그리고 의미적 클래스(semantic classes)를 동시에 추출할 수 있습니다.

- **Technical Details**: Éclair는 ViT 생김새의 인코더와 자동 회귀 디코더로 구성된 트랜스포머 아키텍처를 사용합니다. 이 모델은 LaTeX 소스에서 직접 생성된 진리 레이블(ground truth labels)을 이용하여 훈련되어, 다양한 설명 주제를 포함하는 arXiv-5M이라는 대규모 데이터셋을 생성했습니다. 모델은 읽기 순서를 유지하면서 형식화된 텍스트와 그에 대한 경계 상자 및 의미적 클래스를 추출할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: Éclair는 새롭게 출시된 DROBS 벤치마크에서 최첨단(state-of-the-art) 정확도를 달성했으며, 복잡한 문서 구조를 이해하는 여러 기존 벤치마크에서도 경쟁력 있는 결과를 보여주었습니다. 이는 텍스트 밀도가 높은 문서에서 효과적으로 고품질의 텍스트 토큰을 추출하고, 문서 질문 응답(document question answering) 및 LLM 훈련에 필요한 데이터를 정제하는 데 기여할 수 있음을 보여줍니다.



### Enhanced Feature-based Image Stitching for Endoscopic Videos in Pediatric Eosinophilic Esophagitis (https://arxiv.org/abs/2502.04207)
- **What's New**: 이번 연구에서는 위내시경 영상의 스티칭 품질을 개선하기 위해 새로운 전처리 파이프라인을 제안합니다. 특히 식도 내시경 영상의 특수성을 고려하여, 기존 기술로는 해결할 수 없는 스티칭 문제를 해결하고자 합니다. 이 방법은 내시경 비디오 데이터를 연속적인 2D 이미지로 변환하며, 의료영상의 효과적인 진단을 지원합니다.

- **Technical Details**: 이 파이프라인은 네 가지 주요 단계로 구성됩니다: (1) 키프레임 선택, (2) 이미지 회전 조정, (3) 표면 펼치기, (4) 특징점 매칭입니다. AHE(Adaptive Histogram Equalization) 기술을 사용하여 이미지에서 더 많은 특징점을 탐지하고, SIFT(Scale-Invariant Feature Transform) 알고리즘을 통해 특징점을 매칭합니다. 마지막으로, RANSAC(Random Sample Consensus) 알고리즘을 활용하여 적합한 매칭 쌍을 정제합니다.

- **Performance Highlights**: 20개의 소아 내시경 비디오에 대한 실험 결과, 제안된 방법이 기존 기법에 비해 이미지 정렬 및 스티칭 품질을 유의미하게 향상시킴을 보여주었습니다. 유효한 특징점 매칭 쌍의 수가 증가함에 따라 스티칭 품질이 향상되었으며, 이는 전체적인 영상의 정확성과 신뢰성을 높이는 데 기여하였습니다.



### PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models? (https://arxiv.org/abs/2502.04192)
Comments:
          Under Review

- **What's New**: 이 연구는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 픽셀 수준 이해를 개선하기 위해 새로운 벤치마크를 제안합니다. 연구자들은 픽셀 수준의 기초(supervision) 없이 학습한 MLLMs 또한 높은 성능을 발휘할 수 있음을 입증하고 있습니다. 이러한 MLLMs가 시각적 질문 응답(Visual Question Answering, VQA)에서 기존의 MLLMs보다 더 나은 결과를 내는 경우도 있다는 점이 주목됩니다.

- **Technical Details**: 연구에서는 두 가지 새로운 벤치마크인 PixMMVP와 PixCV-Bench를 제시하며, 이 벤치마크들은 특정 질문에 대한 객체 관심 표현을 수동으로 주석 처리하는 방법을 포함합니다. PixFoundation이라고 불리는 단순한 기법을 통해 기본 모델에서 픽셀 수준 정보의 추출을 용이하게 합니다. 또한, 미세 조정 없이도 MLLMs가 꽤 높은 성능을 발휘하는 것에 대한 연구를 진행하였습니다.

- **Performance Highlights**: 연구 결과, 높은 전통적 기초(supervision)를 가진 MLLMs들이 도전적인 기준에서 여전히 성능이 저하되고 있음을 보여줍니다. 많은 모델이 시각적 기초(input grounding)에서 성능이 부족한 반면, 픽셀 수준 기초 없이도 높은 성능을 나타내는 모델들이 존재합니다. 이러한 결과는 MLLMs가 지속적으로 기능을 발전시킬 수 있는 가능성을 보여줍니다.



### YOLOv4: A Breakthrough in Real-Time Object Detection (https://arxiv.org/abs/2502.04161)
- **What's New**: 이 논문에서는 YOLOv4의 획기적인 성능을 상세히 탐구하고 있습니다. YOLOv4는 Darknet 프레임워크를 활용하여 COCO 데이터셋에서 최상의 성능을 달성하였으며, 다양한 최신 기법을 결합하여 정확성과 적응력을 향상시켰습니다. Cross mini-Batch Normalization, Self-Adversarial Training 등과 같은 기법들은 YOLOv4의 주요한 특성으로, 실시간 객체 인식에 있어서 매우 효과적입니다.

- **Technical Details**: YOLOv4는 CSPDarkNet53을 기반으로 하며, Path Aggregation Network (PANet)과 Spatial Pyramid Pooling (SPP)을 통합하여 전체적인 처리 효율성과 정확성을 크게 향상시켰습니다. 새로운 데이터 보강 기법인 Mosaic augmentation은 여러 이미지 맥락에서 학습하여 다양한 객체 크기와 환경에 일반화 능력을 높입니다. 이 모델은 꾸준한 속도와 성능 향상에도 불구하고 상대적으로 낮은 연산 복잡성을 유지하며 효율적인 실시간 처리를 가능하게 합니다.

- **Performance Highlights**: YOLOv4는 약 65fps의 처리 속도로 43.5%의 AP(average precision)를 달성하였습니다. 이는 실시간 환경에서도 매우 효율적이고 경제적인 성능을 발휘할 수 있음을 보여줍니다. 이전 버전들에 비해 속도와 정확성 모두에서 약 10%와 12%의 향상을 보이며, 다양한 응용 분야에서도 강력한 도구로 자리매김하였습니다.



### HD-EPIC: A Highly-Detailed Egocentric Video Datas (https://arxiv.org/abs/2502.04144)
Comments:
          29 pages. Project Webpage and Dataset: this http URL

- **What's New**: 이 논문에서는 비주얼 질문 응답(Visual Question Answering, VQA) 벤치마크와 함께 고도로 세부적으로 주석이 달린 새로운 주방 기반의 인지적 비디오 데이터셋 HD-EPIC을 소개합니다. HD-EPIC은 3D 공간의 디지털 트윈을 통해 모든 주석이 기초 데이터로 뒷받침되며, 사람의 시선과 객체의 움직임까지 추적합니다. 이 데이터셋은 다양한 가정 환경에서 자발적으로 녹화된 비디오로 구성되어 있으며, 실험실 환경에서 수집된 데이터셋과 비교할 때 현실감이 뛰어납니다.

- **Technical Details**: HD-EPIC은 41시간 분량의 비디오로 구성되어 있으며, 9개의 주방에서 69개의 레시피와 59,000개의 세부 행동, 51,000개의 오디오 이벤트, 20,000개의 물체 움직임 및 37,000개의 3D 객체 마스크를 캡처합니다. 특히, 각 주방은 라벨이 붙은 fixture의 디지털 트윈으로 구성되어 있으며, 각 행동은 시작 및 종료 시간과 함께 설명됩니다. 또한, 이 데이터셋은 다양한 비디오 기반 및 비디오-언어 모델을 평가하는 데 매우 적합한 주석 세트를 제공합니다.

- **Performance Highlights**: HD-EPIC을 통해 설계된 도전적인 VQA 벤치마크는 26,000개의 질문으로 구성되며, 이는 레시피 인식, 영양, 세부 행동, 3D 인식, 객체 움직임 및 시선 방향을 평가합니다. Gemini Pro 모델은 이 벤치마크에서 38.5%라는 낮은 성과를 기록하여 현재 비디오 언어 모델의 한계를 드러냅니다. 이 논문은 HD-EPIC 데이터셋이 제공하는 다층적 주석 덕분에 비디오 인식 및 행동 인식 작업에서 향후 발전 가능성을 열어준다는 점에서 중요한 기여를 하고 있습니다.



### Beyond the Final Layer: Hierarchical Query Fusion Transformer with Agent-Interpolation Initialization for 3D Instance Segmentation (https://arxiv.org/abs/2502.04139)
Comments:
          Under review

- **What's New**: 이 논문에서는 3D 인스턴스 분할을 위한 새로운 방법인 BFL(Beyond the Final Layer)을 제안합니다. BFL은 Agent-Interpolation Initialization Module(AI2M)을 통해 쿼리 초기화를 향상시키며, Hierarchical Query Fusion Decoder(HQFD)를 통해 계층 간 인식 유지 문제를 해결합니다. 이를 통해 전통적인 트랜스포머 기반 방법의 한계를 극복하고 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: BFL은 두 가지 주요 모듈을 통합한 모델입니다. AI2M은 위치 쿼리와 내용 쿼리의 집합을 초기화하고, 이를 통해 포인트 클라우드의 표본 내용 쿼리를 interpolating하여 고른 픽셀 분포를 제공합니다. HQFD는 각 레이어의 예측 간 Intersection over Union(IoU)를 계산하고, 낮은 오버랩 쿼리들을 통합하여 다음 레이어에 전달함으로써 안정적인 최적화를 도모합니다.

- **Performance Highlights**: ScanNetV2, ScanNet200, ScanNet++, S3DIS 데이터셋에서 BFL의 성능은 기존 방법에 비해 우수한 결과를 보여주었습니다. 특히, 쿼리 초기화 및 계층 간 기억 유지 문제를 효과적으로 다룸으로써 모델의 총 성능을 7.8% 향상시키며, 물체 인식을 개선하는 데 성공했습니다.



### Adaptive Margin Contrastive Learning for Ambiguity-aware 3D Semantic Segmentation (https://arxiv.org/abs/2502.04111)
- **What's New**: 본 논문은 3D 포인트 클라우드의 의미적 분할을 위한 적응형 마진 대조 학습 방법인 AMContrast3D를 제안합니다. 기존 방법들이 동일한 목적 함수를 사용하여 각 포인트의 애매함과 판별력 부족을 무시하는 반면, AMContrast3D는 개별 포인트의 애매함 수준에 따라 적응형 목표를 설정합니다. 이를 통해 낮은 애매함을 가진 포인트의 정확성을 보장하고 높은 애매함을 가진 포인트에서는 실수를 허용함으로써 모델의 최적성을 향상시킵니다.

- **Technical Details**: AMContrast3D는 포지션 임베딩을 기반으로 포인트의 애매함을 추정하고, 이후 마진 생성기를 개발하여 대조적 특성 임베딩의 결정 경계를 조정합니다. 특히, 높은 애매함을 가진 포인트에 대해 심지어는 부정적인 마진을 설정하여 경계가 줄어들도록 합니다. 이렇게 조정된 마진은 각 포인트의 분포에 따라 훈련 난이도를 동적으로 조절하며, 이는 모델의 과최적화를 방지하고 특성 판별력을 강화하는 데 기여합니다.

- **Performance Highlights**: 대규모 데이터셋인 S3DIS와 ScanNet에 대한 실험 결과에서, AMContrast3D는 최첨단 방법들을 능가하는 성능을 보여주었습니다. 연구 결과는 제안한 방법의 유효성을 입증하며, 특히 애매한 포인트에 대한 embedded learning의 강점을 강조합니다. Ablation study를 통해 제안한 방법이 기존 방법들보다 보완적인 효과를 제공함을 추가적으로 확인했습니다.



### Efficient Few-Shot Continual Learning in Vision-Language Models (https://arxiv.org/abs/2502.04098)
- **What's New**: 이 논문에서는 LoRSU(Low-Rank Adaptation with Structured Updates)라는 새로운 방법을 제안합니다. 이 방법은 VLMs(Vision-Language Models) 내에서 이미지 인코더를 선택적으로 업데이트함으로써 효율적이고 강력한 성능 개선을 목표로 합니다. LoRSU는 구조적 및 지역화된 매개변수 업데이트를 도입하여 기존의 이미지 인코더가 자주 발생하는 오류를 효과적으로 수정하고, 모델의 일반적인 강건성을 유지합니다.

- **Technical Details**: LoRSU 방법은 사전 훈련된 이미지 인코더(CLIP 변형)를 사용하여 고정된 상태에서 VLM을 비주얼 어시스턴트로 배포할 때의 한계를 극복하고자 합니다. 이 방법은 기존 모델의 대부분 오버헤드를 줄이면서 모든 매개변수에 대한 일반적인 업데이트를 수행하여 파라미터 선택성을 극대화 합니다. LoRSU는 매개변수 선택을 위해 이론적 통찰력을 활용하여 중요 매개변수만 업데이트하여 자원 효율성을 달성합니다.

- **Performance Highlights**: 실험 결과, LoRSU는 VQA(Visual Question Answering) 작업에서 25배 이상의 계산 효율성을 보이며 성능을 희생하지 않고 이미지 인코더를 업데이트하는 데 있어 우수한 성능을 발휘하고 있습니다. 또한 이 방법은 특정 도메인으로의 전환에서도 기존 지식을 보존하며, 파라미터 수가 적은 이미지 인코더의 업데이트가 VLM의 성능 개선에 필수적임을 보여줍니다.



### Automatic quantification of breast cancer biomarkers from multiple 18F-FDG PET image segmentation (https://arxiv.org/abs/2502.04083)
Comments:
          Submit soon to EJNMMI Research

- **What's New**: 본 연구는 유방암 환자에서 선행항암요법(neoadjuvant chemotherapy, NAC) 후 18F-FDG PET 이미지를 이용하여 종양 분할을 자동으로 수행하는 시스템을 개발했습니다. 연구팀은 243개의 초기 18F-FDG PET 스캔(PET_Bl)과 180개의 추적 18F-FDG PET 스캔(PET_Fu)을 분석하여, 자동으로 종양 영역을 세분화하고 핵심 바이오마커를 추출했습니다. 이는 유방암의 치료 반응을 평가하고 진단의 정확성을 높이기 위한 중요한 첫걸음으로 평가됩니다.

- **Technical Details**: 연구에서는 nnUNet 딥러닝 모델을 사용하여 PET_Bl 스캔에서 종양을 정확히 분할했습니다. PET_Fu 스캔의 경우, 모델을 15개의 후속 검사로 미세 조정하여 NAC 후의 종양 진화를 평가할 수 있도록 하였습니다. 이 과정에서 최대 표준 섭취량(maximum standardized uptake value, SUVmax), 대사 종양 부피(metabolic tumor volume, MTV), 총 병변 당화(total lesion glycolysis, TLG) 등의 바이오마커를 계산하여 종양의 변화를 파악했습니다.

- **Performance Highlights**: nnUNet 모델은 PET_Bl 스캔에서 0.89의 Dice similarity coefficient (DSC)와 3.52 mm의 Hausdorff distance (HD)를 기록하며 우수한 성능을 보였습니다. PET_Fu 스캔에서 미세 조정 후에는 DSC 0.78과 HD 4.95 mm를 달성했습니다. 전반적으로 SUVmax, MTV 및 TLG의 중요한 평균 감소가 관찰되어, 수집된 바이오마커가 유방암의 진행 상황을 평가하는 데 유의미하다는 것을 입증했습니다.



### Content-Rich AIGC Video Quality Assessment via Intricate Text Alignment and Motion-Aware Consistency (https://arxiv.org/abs/2502.04076)
- **What's New**: 이번 논문에서는 새로운 세대의 비디오 생성 모델인 Sora의 등장이 AI 생성 콘텐츠(Artificial Intelligence Generated Content, AIGC) 비디오 품질 평가(Video Quality Assessment, VQA)에서 발생하는 새로운 과제를 다룹니다. 기존 평가 방식이 긴 텍스트 프롬프트와 복잡한 동작 패턴을 가진 비디오를 이해하는 데 어려움을 겪는 가운데, CRAVE(Content-Rich AIGC Video Evaluator)라는 새로운 평가 모델을 제안합니다. CRAVE는 텍스트와 비디오의 복합성을 효과적으로 통합하는 기능을 가지고 있습니다.

- **Technical Details**: CRAVE는 세 가지 주요 관점에서 AIGC 비디오를 평가합니다. 첫 번째로, 전통적인 시각적 조화(visual harmony)를 고려하고, 이는 기존 VQA 방법에서 미적 요소와 왜곡을 측정하는 데 중점을 두었습니다. 두 번째로, CRAVE는 다양한 세분화된 텍스트-시간 융합 모듈(multi-granularity text-temporal fusion module)을 사용하여 복잡한 텍스트와 비디오 동작을 정렬합니다. 마지막으로, 혼합 운동-충실도 모델(hybrid motion-fidelity modeling)을 활용하여 시간적 품질을 평가합니다.

- **Performance Highlights**: CRAVE는 T2V-DB 및 제안된 CRAVE-DB와 같은 여러 AIGC VQA 벤치마크에서 뛰어난 결과를 도출했습니다. 이 모델은 length of prompts와 다양한 소스에서의 비디오 품질 평가에서 인간 인지와 높은 일치를 보여주었습니다. 이를 바탕으로, CRAVE는 새로운 세대의 AIGC 비디오에서 시간적 및 비디오-텍스트 일관성을 평가하는 효과적인 도구로 자리잡았습니다.



### 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation (https://arxiv.org/abs/2502.04074)
- **What's New**: 본 논문은 3D와 2D 시선 추정이 서로 다른 연구 분야로 여겨지는 경향을 극복하고, 사전 훈련된 3D 시선 추정 네트워크를 2D 시선 예측에 적용하는 혁신적인 크로스-태스크 소수 샷(few-shot) 접근 방식을 소개합니다. 이 방법은 새로운 장치에서 몇 장의 훈련 이미지만으로 2D 시선을 예측할 수 있도록 고안되었습니다. 3D와 2D 시선 간의 도메인 차이(domain gap)와 화면 포즈(screen poses)의 불확실성, 제한된 훈련 데이터로 인해 이 작업이 매우 도전적이라는 점이 강조됩니다.

- **Technical Details**: 제안된 프레임워크는 화면 포즈를 모델링하고 3D 시선을 2D 시선으로 투영하는 물리 기반(physics-based) 미분 가능(differentiable) 투영 모듈을 포함합니다. 이 모듈은 학습 가능한 매개변수(parameters)를 가지고 있으며, 기존의 3D 시선 네트워크와의 통합이 가능하므로 아키텍처를 수정할 필요가 없습니다. 또한, 역 투영(reverse projection) 과정을 사용하여 2D 레이블을 3D 공간으로 변환하고 플리핑(flipping) 문제를 해결하는 동적 유사 레이블링(dynamic pseudo-labelling) 전략을 도입합니다.

- **Performance Highlights**: 제안된 방법은 MPIIGaze, EVE, GazeCapture 데이터셋에서 평가되었으며, 각각 노트북, 데스크탑 컴퓨터, 모바일 장치에서 수집된 데이터입니다. 이 성능의 우수성은 제안된 접근 방식의 효과를 명확히 하고, 실제 애플리케이션에서 강력한 가능성을 보여줍니다. 결론적으로, 본 연구는 2D 시선 추정 기술에 대한 새로운 시각을 제공하며, 다양한 기기에서의 적용 가능성을 증명하고 있습니다.



### Inteligencia artificial para la multi-clasificaci\'on de fauna en fotograf\'ias autom\'aticas utilizadas en investigaci\'on cient\'ifica (https://arxiv.org/abs/2502.04064)
Comments:
          in Spanish language, XXIV Workshop de Investigadores en Ciencias de la Computación (WICC 2022, Mendoza)

- **What's New**: 이번 연구에서는 자연 환경 관리를 위해 카메라 트랩(camera traps)을 활용하여 야생 동물의 행동 및 분포를 이해하고자 합니다. 아르헨티나 Tierra del Fuego 지역에서 다양한 초식동물(guanacos, cows, sheep)의 임목 이용 연구를 통해 생태계 보호와 관리 최적화에 기여하고 있습니다. 특히, 대량의 이미지 처리 문제를 해결하기 위해 Neural Networks와 Deep Learning 기술을 활용하여 동물 종 분류의 효율성을 높이는 방안을 모색합니다.

- **Technical Details**: 이 연구는 카메라 트랩을 사용하여 야생 동물의 이미지를 수집하고, 이러한 방대한 데이터를 Neural Networks와 Deep Learning 알고리즘으로 분석하여 분류하는 방법에 초점을 맞추고 있습니다. 수집된 사진들은 생태학적 관장에서 중요한 정보를 제공할 수 있으며, 이러한 정보는 야생 지역 관리에 효과적으로 적용될 수 있습니다. 대규모 이미지 데이터를 처리하기 위한 자동화된 솔루션 구축이 주요 기술적 과제입니다.

- **Performance Highlights**: 본 프로젝트는 야생 동물의 종 분류를 위한 신경망 모델 개발을 목표로 하고 있으며, 이는 기존의 수작업 이미지 분석 방식의 한계를 극복하는 데 기여할 것입니다. 시간과 비용 효율성을 극대화하여, 생태계 연구에 있어 중요한 통찰력을 제공할 것으로 기대됩니다. 또한, 이 연구는 야생 동물 보존 분야에서 AI 기술의 응용 가능성을 확대하는 데 중요한 기여를 할 것입니다.



### PartEdit: Fine-Grained Image Editing using Pre-Trained Diffusion Models (https://arxiv.org/abs/2502.04050)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 사전 훈련된 diffusion 모델을 기반으로 객체 부분에 대한 텍스트 기반 이미지 편집 접근 방식을 제안합니다. 기존의 diffusion 모델은 이미지의 세부적인 객체 부분을 충분히 이해하지 못해 사용자가 요청한 세밀한 편집이 어렵습니다. 이를 해결하기 위해 각 대체 편집 지역을 정확히 위치시키는 특별한 텍스트 토큰을 학습하여, 모델의 지식을 확장합니다.

- **Technical Details**: 제안하는 방식은 특수한 텍스트 토큰을 학습하여 다양한 객체 부분에 대한 이해를 향상시킵니다. 이 토큰들은 각 추론 단계에서 신뢰할 수 있는 로컬라이제이션 마스크를 생성하도록 최적화되어, 편집이 필요한 지역을 정확하게 위치시킵니다. 또한, feature blending 및 adaptive thresholding 전략을 설계하여 편집을 매끄럽게 수행합니다.

- **Performance Highlights**: 우리의 방법은 모든 측정 지표에서 기존의 편집 방법들보다 우수한 성과를 보여 주며, 실시된 사용자 연구에서도 77%에서 90%의 사용자들이 우리 방법을 선호하는 결과를 나타냅니다. 이를 통해 우리의 접근 방식이 세밀한 이미지 편집에 적합하다는 것을 입증하였습니다.



### Enhancing people localisation in drone imagery for better crowd management by utilising every pixel in high-resolution images (https://arxiv.org/abs/2502.04014)
Comments:
          This is the pre-print. The article is submitted to the Engineering Applications of Artificial Intelligence journal

- **What's New**: 이번 논문에서는 드론을 이용한 사람 위치 식별의 정확성을 높이기 위한 새로운 접근법이 제안되었습니다. 이는 특히 대규모 행사와 공공 집회의 군중 관리에 중요한 역할을 할 수 있습니다. 새로운 Pixel Distill 모듈과 함께, 동적 환경에서의 사람 수 세기와 위치 파악을 개선하는 새로운 데이터셋인 UP-COUNT가 공개되었습니다. 이 접근법은 기존의 방법들보다 더 나은 성능을 발휘하며, 실제 환경에서의 적용 가능성을 보여줍니다.

- **Technical Details**: 제안된 Dot localisation 기법은 UAV 촬영 이미지에서 포인트 레이블을 가진 밀집 객체를 감지하기 위해 설계되었습니다. 이 방법은 세부 정보를 잃지 않도록 이미지의 해상도를 조정하는 대신, 모든 픽셀 값을 동시에 처리하는 Pixel Distill 모듈을 활용하여 높은 처리 효율성을 달성합니다. 이러한 접근 방식은 알림 상자가 필요하지 않은 경우, 객체의 중심 좌표를 추정하는 데 유리합니다. 실험 결과, 이 새로운 방법은 기존 알고리즘보다 우수한 성능을 발휘하였습니다.

- **Performance Highlights**: 새롭게 소개된 UP-COUNT 데이터셋은 다양한 사람 수의 시나리오를 포함하여, 드론을 활용한 이미지 촬영 시의 도전 과제를 해결하고 있습니다. 제안된 방법은 기존의 DroneCrowd 데이터셋 및 UP-COUNT 데이터셋에서 기존 기법보다 뛰어난 성능을 발휘하여, 군중 객체 위치 파악 업무에 있어서 나은 효율성을 제공합니다. 이러한 성과는 향후 실제 환경에서의 드론 기반 감시 및 분석을 위한 방향성을 제시합니다.



### CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing (https://arxiv.org/abs/2502.03997)
- **What's New**: 이번 연구에서는 Text-based CAD editing이라는 개념을 도입하며, CAD 모델을 텍스트 기반으로 자동 수정하는 최초의 프레임워크인 CAD-Editor를 제안합니다. 이는 기존의 CAD 모델을 입력으로 활용하지 않고는 불가능한 의미 있는 편집을 가능하게 합니다. CAD-Editor는 설계 вариации 모델과 Large Vision-Language Models (LVLMs)를 활용한 자동화된 데이터 합성 파이프라인을 통해 고유한 데이터 쌍을 생성합니다.

- **Technical Details**: CAD-Editor는 seq2seq (sequence-to-sequence) 생성 문제로 구성되며, 입력으로 표시된 수정 지침과 원래 CAD 모델의 시퀀스를 조합합니다. 이 작업은 두 개의 하위 작업으로 분해되어, 수정이 필요한 영역을 찾아내고 해당 영역에 적절한 수정 사항을 삽입합니다. 각 단계에서는 Large Language Models (LLMs)을 사용하여 자연어 이해 및 CAD 지식을 활용하여 복잡한 편집 문제를 처리합니다.

- **Performance Highlights**: 실험 결과, CAD-Editor는 CAD 모델 생성의 유효성, 텍스트와 CAD의 정합성 및 전체적인 품질에서 기초 사례들보다 뛰어난 성과를 보였습니다. 특히, CAD-Editor는 사용자 요구에 맞춘 정밀한 수정을 가능하게 하여 디자인 전문 지식이 적은 사람들도 더욱 효과적으로 CAD 모델을 생성할 수 있도록 지원합니다.



### RWKV-UI: UI Understanding with Enhanced Perception and Reasoning (https://arxiv.org/abs/2502.03971)
Comments:
          10 pages, 5figures, conference

- **What's New**: 본 연구에서는 기존의 Vision Language Models(VLMs)가 고해상도 웹 인터페이스에서 정보 손실과 제한된 추리 능력으로 어려움을 겪고 있다는 문제를 해결하고자 RWKV-UI라는 새로운 모델을 제안합니다. 이 모델은 고해상도 UI 이미지를 효과적으로 처리하기 위해 RWKV 아키텍처에 기반하여 설계되었으며, 레이아웃 검출(layout detection)을 시각적 프롬프트로 도입하여 웹페이지 레이아웃 구조를理解합니다.

- **Technical Details**: RWKV-UI는 SIGLIP, DINO, SAM의 세 가지 비주얼 인코더를 포함하는 아키텍처를 채택하였으며, 최대 해상도 4096×4096을 지원하여 세밀한 UI 구조를 이해합니다. 입력 이미지는 네 부분으로 나뉘어 각 인코더를 통해 특성을 추출하고, 이 특성들은 합쳐져 avgpool2d를 통해 통합됩니다. 이러한 방식으로 모델은 원본 이미지의 모든 세부 정보를 손실 없이 유지하며, 해상도가 높아져도 추론 비용은 증가하지 않습니다.

- **Performance Highlights**: 실험 결과, RWKV-UI는 고해상도 UI 이해 및 인터랙티브 추리 작업에서 기존 모델보다 우수한 성능을 보였습니다. 특히, 웹 인터페이스의 다양한 요소들을 효과적으로 인식하고, 사용자 상호작용 경로를 예측하는 능력이 뛰어난 것으로 나타났습니다. 이러한 성과는 고해상도 시나리오에서의 추론 능력을 크게 향상시키는 데 기여합니다.



### MultiFloodSynth: Multi-Annotated Flood Synthetic Dataset Generation (https://arxiv.org/abs/2502.03966)
Comments:
          6 pages, 6 figures. Accepted as Oral Presentation to AAAI 2025 Workshop on Good-Data

- **What's New**: 이 논문에서는 홍수 위험 감지 시스템을 위한 합성 데이터 생성 프레임워크인 MultiFloodSynth를 소개합니다. 이 프레임워크는 다양한 실제 속성을 가상 세계로 옮겨와 홍수 상황을 시뮬레이션하며, 기존 데이터 수집의 한계를 극복하기 위해 최신 generative models를 활용합니다. MultiFloodSynth는 5단계의 다양한 주석을 포함한 풍부한 합성 데이터셋을 제공하여, 실제 데이터셋과의 유사성을 유지하면서도 효율적인 모델 학습 환경을 조성합니다.

- **Technical Details**: 프레임워크는 3D 엔진을 이용하여 가상 도시 홍수 상황을 합성합니다. 이 과정에서 레이아웃(layout), 조명(lighting), 홍수 높이(flood-level)와 같은 다양한 속성을 고려하며, 여러 종류의 주석(annotation) 정보(예: normal map, 세분화(segmentation) 맵, 2D/3D 바운딩 박스)를 생성합니다. MultiFloodSynth는 사용자가 원하는 구조를 설정하고 조정할 수 있는 유연성을 제공하여 최종 가상 장면을 구성할 수 있게 합니다.

- **Performance Highlights**: MultiFloodSynth는 총 70,117장의 이미지를 생성하였으며, 그 중 14,593장이 홍수 장면에 해당합니다. 실제 데이터와 비교했을 때, YOLOv10 모델을 사용한 홍수 감지 성능이 높아졌으며, 향상된 정확성을 보여줍니다. 또한 다양한 주석 형태를 활용하여 다양한 컴퓨터 비전 작업에 대한 응용 가능성을 높였습니다.



### Improving the Perturbation-Based Explanation of Deepfake Detectors Through the Use of Adversarially-Generated Samples (https://arxiv.org/abs/2502.03957)
Comments:
          Accepted for publication, AI4MFDD Workshop @ IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025), Tucson, AZ, USA, Feb. 2025. This is the authors' "accepted version"

- **What's New**: 이번 논문에서는 감지된 딥페이크로 분류된 입력 이미지의 적대적으로 생성된 샘플을 사용하여, 다양한 입력 특성의 중요성을 추론하고 시각적 설명을 생성하는 새로운 접근 방식을 제안합니다. 이 과정은 Natural Evolution Strategies(NES)를 기반으로 하며, 초기 딥페이크 탐지기의 결정을 반전시키는 샘플을 생성하는 데 초점을 맞추고 있습니다. 이렇게 생성된 샘플을 통해 기존의 설명 방법을 개선하고 보다 신뢰할 수 있는 설명을 제공합니다.

- **Technical Details**: 제안된 접근법은 적대적으로 생성된 샘플을 사용하여 perturbation masks를 형성하며, 이는 특정 입력 특성의 중요성을 추론하는 데 도움을 줍니다. 네 가지 SOTA 설명 방법인 RISE, SHAP, LIME, SOBOL을 수정하여 이 새로운 perturbation 방식을 통합하고 데이터베이스로는 FaceForensics++를 활용하여 성능을 평가합니다. 실험 과정에서 적은 양의 노이즈만을 추가하여 시각적으로 유사한 샘플을 생성하며, OOD(out-of-distribution) 문제를 회피합니다.

- **Performance Highlights**: 수량적 및 정성적 평가를 통해 제안된 perturbation 접근 방식이 기존 설명 방법의 성능에 긍정적인 기여를 했음을 입증하였습니다. 특히, 수정된 설명 방법이 조작된 이미지 영역을 더 정확하게 구별하는 능력을 갖추고 있음을 보여줍니다. 이를 통해, 딥페이크 탐지기에서의 결정에 대한 보다 유용하고 의미 있는 시각적 설명을 제공합니다.



### LR0.FM: Low-Resolution Zero-shot Classification Benchmark For Foundation Models (https://arxiv.org/abs/2502.03950)
Comments:
          Accepted to ICLR 2025

- **What's New**: 본 논문은 비주얼-언어 기초 모델(Foundation Models, FMs)에 대한 새로운 벤치마크, LR0.FM을 제시하여 저해상도(LR) 이미지에서의 제로샷 분류 성능을 평가합니다. 10개의 FM과 66개의 백본(backbone), 그리고 15개의 다양한 이미지 분류 데이터셋을 통해 저해상도가 모델 성능에 미치는 영향을 분석하였습니다. 이 연구는 기존 측정 지표의 한계를 극복하기 위해 Weighted Aggregated Robustness(WAR)라는 새로운 지표를 도입합니다.

- **Technical Details**: 비주얼-언어 기초 모델들은 CLIP, LLaMA 등 다양한 모델을 포함하며, 이러한 모델들은 대규모 멀티모달 데이터셋을 통한 광범위한 프리트레이닝(pre-training)을 통해 제로샷(zero-shot) 능력을 발휘합니다. 저해상도 이미지는 보안 감시 영상 및 위성 이미지 등 여러 실제 시나리오에서 자주 발생하며, 이는 매우 도전적인 과제가 됩니다. 연구 결과에 따르면 모델의 크기가 저해상도 감소에 대한 저항성과 긍정적으로 관련이 있으며, 프리트레이닝 데이터셋의 품질이 크기보다 더 중요하다는 사실을 발견했습니다.

- **Performance Highlights**: 제안된 모델 LR-TK0는 저해상도 환경에서 FMs의 성능을 향상시키는 간단하지만 효과적인 방법으로, 기존 프리트레이닝 가중치를 변경하지 않고도 저해상도에 대한 모델의 강건성을 높일 수 있습니다. 그 결과, 다양한 데이터셋에서 저해상도 제로샷 분류 작업의 성능이 향상되었습니다. 전체적으로 본 연구는 저해상도 이미지에서 FMs의 성능 저하를 탐색하고 이 분야에 대한 새로운 통찰력을 제공합니다.



### No Free Lunch in Annotation either: An objective evaluation of foundation models for streamlining annotation in animal tracking (https://arxiv.org/abs/2502.03907)
Comments:
          \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 동물 추적을 위한 주석 생성의 번거로운 작업을 다루는 기초 모델의 능력을 분석합니다. 대량의 데이터 주석화는 추적 모델의 강건성에 있어 중요한 요소이며, 이론적 신뢰성을 높이는 데 기여할 수 있습니다. 자동 주석 생성은 시간 소모적인 주석 작업을 간소화할 가능성이 높지만, 주석의 품질이 정확도에 미치는 영향이 크다는 점을 강조합니다.

- **Technical Details**: 제안된 SAM-QA (SAM Quality Annotation) 접근법은 Segment Anything Model을 기반으로 하여 자동으로 생성된 프롬프트를 순차적으로 적용합니다. 이 접근법은 초기 프롬프트를 수동으로 제공하고 바이너리 마스크를 생성하여 품질 검사를 진행하는 세 단계로 구성됩니다. 데이터셋은 쥐와 생쥐의 비디오 자료로 구성되며, 주로 경량화된 TinyViT 모델을 활용하고 교차 엔트로피 및 Dice 손실을 사용하여 세분화를 진행합니다.

- **Performance Highlights**: SAM-QA의 신뢰성 높은 주석 생산 전략은 기존의 SAM2 비디오 사용 시 얻었던 IDF1 점수 65.6에 비해 80.8로 향상된 결과를 나타냈습니다. 이는 수동 주석과 자동 주석을 적절히 결합하는 것이 동물 추적 모델의 성능을 극대화할 수 있음을 시사합니다. 전통적인 세분화 기법과 비교하여, SAM-QA는 고품질 주석을 생산하는 데 있어 효과적임을 입증했습니다.



### LeAP: Consistent multi-domain 3D labeling using Foundation Models (https://arxiv.org/abs/2502.03901)
Comments:
          9 pages, 4 figures. ICRA25 preprint

- **What's New**: 이번 연구에서는 Label Any Pointcloud (LeAP)라는 새로운 도구를 소개합니다. 이 도구는 2D Vision Foundation Models (VFMs)을 활용하여 3D 데이터를 자동으로 레이블링 할 수 있도록 합니다. 특히, 다양한 클래스 리스트에 대해 일관된 레이블을 생성할 수 있는 기능을 제공합니다.

- **Technical Details**: LeAP는 Bayesian 업데이트를 사용하여 포인트 레이블을 복합적으로 결합하고, 이를 통해 voxel에 적용하여 시공간 일관성을 개선합니다. 3D Consistency Network (3D-CN)는 3D 정보를 활용하여 레이블의 품질을 더욱 향상시킵니다. 이러한 방법들은 기존의 레이블링 기법들과 비교하여 더 높은 레이블 일관성을 제공합니다.

- **Performance Highlights**: 실험 결과, LeAP는 다양한 분야에서 높은 품질의 3D 시맨틱 레이블을 자동으로 생성할 수 있음을 보여주었습니다. 새로운 도메인에 적응한 모델은 시맨틱 분할 작업에서 최대 34.2 mIoU의 향상을 기록했습니다. 따라서, 이 연구는 3D 데이터 레이블링의 효율성을 크게 개선할 가능성이 있습니다.



### Rule-Based Modeling of Low-Dimensional Data with PCA and Binary Particle Swarm Optimization (BPSO) in ANFIS (https://arxiv.org/abs/2502.03895)
Comments:
          41 pages, 9 figures

- **What's New**: 이 논문에서는 저차원 해석 가능성을 위한 Fuzzy rule-based 시스템의 장점을 살리고, Deep Learning 기술의 문제를 해결하기 위한 새로운 방법론을 제안합니다. 특히, Principal Component Analysis (PCA)와 Binary Particle Swarm Optimization (BPSO)를 활용하여 ANFIS의 규칙 수를 대폭 줄이는 전략적인 규칙 축소 모델을 개발했습니다. 이 접근법은 해석 가능한 AI를 위한 강력한 프레임워크를 제공하며, 다양한 분야에서의 응용 가능성을 강조합니다.

- **Technical Details**: ANFIS는 Fuzzy Logic 시스템과 Neural Networks를 통합하여 데이터의 패턴을 배울 수 있는 강력한 추정 모델입니다. 하지만, 높은 차원의 데이터에 대한 계산 비용과 규칙 기반 복잡성의 문제로 인해 한계가 있습니다. 본 연구에서는 PCA를 통해 데이터의 차원을 줄이고, BPSO를 통해 가장 중요한 구성 요소를 선택하여 규칙 수를 최소화하면서도 결정을 내리는 정확성을 유지합니다.

- **Performance Highlights**: 본 연구에서 제안한 모델은 UCI 호흡기, 키일 분류, 회귀 데이터셋 및 실제 허혈성 뇌졸중 데이터셋에 대한 검증을 통해 적응성과 실용성을 나타냈습니다. 규칙 수의 감소, 훈련 시간의 단축 및 높은 정확도를 보여줌으로써 복잡한 데이터 시나리오에서도 효과적인 성능을 입증했습니다. 이 방법은 Fuzzy Logic과 최적화를 결합하여 튼튼한 솔루션을 제공하며, 해석 가능한 AI 구현에 기여합니다.



### Advanced Object Detection and Pose Estimation with Hybrid Task Cascade and High-Resolution Networks (https://arxiv.org/abs/2502.03877)
- **What's New**: 이번 연구는 6D 객체 감지(object detection)와 자세 추정(pose estimation)의 정확도를 크게 향상시키기 위해 Hybrid Task Cascade (HTC)와 High-Resolution Network (HRNet) 백본을 통합한 새로운 파이프라인을 제안합니다. 기존 6D-VNet 프레임워크를 개선하였으며, 각 단계에서 객체 제안이 반복적으로 정제되는 HTC의 장점과 HRNet의 고해상도 표현 유지 능력을 활용하였습니다. 이러한 혁신을 통해 공공 및 개인 벤치마크에서 우수한 성능을 발휘함을 입증하였습니다.

- **Technical Details**: 제안된 방법론은 HTC와 HRNet 백본을 기반으로 한 3단계 Hybrid Task Cascade 아키텍처를 활용하여 6D 객체 감지 및 자세 추정을 수행합니다. HTC 프레임워크는 의미론적 분할(semantic segmentation)과 객체 감지 작업을 통합하여 각 단계에서 객체 제안을 정제하여 정확도를 향상시킵니다. HRNet은 다양한 해상도에서 특징을 추출할 수 있는 다중 병렬 합성곱을 사용하여, 자세 추정 및 분할을 위한 세부적인 공간 정보 유지에 필수적입니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존 최첨단 모델들보다 상당한 성능 향상을 보여주었습니다. 공공 벤치마크에서의 성능은 물론 개인 리더보드에서 발생하는 다양한 복잡한 환경 속에서도 정확한 3D 감지 및 자세 추정이 가능하다는 것을 입증하였습니다. 이를 통해 높은 정밀성과 신뢰성을 요구하는 실제 응용 분야에서의 활용 가능성이 커졌습니다.



### Taking A Closer Look at Interacting Objects: Interaction-Aware Open Vocabulary Scene Graph Generation (https://arxiv.org/abs/2502.03856)
- **What's New**: INOVA는 상호작용하는 객체를 인식하는 새로운 프레임워크로, 기존의 OVSGG 방법들이 상호작용하는 객체들 간의 관계를 간과함으로써 발생하는 문제를 해결하고자 합니다. 이 방법은 이미지에서 상호작용하는 객체를 식별하기 위한 명확한 생성 전략을 사용하고, 이 후 훈련 단계에서 흥미로운 객체들에 대한 쿼리 선택을 우선시합니다. 또한, 지식 증류(knowledge distillation) 과정을 통해 모델의 강인성을 강화하는 데 중점을 두고 있습니다.

- **Technical Details**: INOVA는 이중 인코더-단일 디코더 아키텍처를 따르며, 시각적 및 텍스트 인코더, 교차 모달 디코더, 그리고 개체 및 관계 분류기로 구성되어 있습니다. 사전 훈련 단계에서 INOVA는 상호작용하는 객체 쌍의 바닥에 강조하는 쌍방향 상호작용 프롬프트를 활용하여, 비활성 객체의 간섭을 줄이고 관계 예측의 정확성을 높입니다. 이 프레임워크는 INOVA의 강력한 성능을 보장하는 다양한 전략적 접근법을 도입하고 있습니다.

- **Performance Highlights**: INOVA는 VG 및 GQA 데이터셋에서의 광범위한 실험을 통해 기존의 상호작용 인식 기법과 비교해 획기적인 성능을 달성했습니다. 본 연구는 상호작용에 대한 명확한 모델링을 통해 객체 간의 관계 오류를 감소시키면서, 실세계 응용의 가능성을 증명했습니다. 다양한 벤치마크에서의 결과는 INOVA의 효과가 기존 방법들보다 월등함을 시사합니다.



### Semi-rPPG: Semi-Supervised Remote Physiological Measurement with Curriculum Pseudo-Labeling (https://arxiv.org/abs/2502.03855)
Comments:
          Accepted by IEEE Transactions on Instrumentation and Measurement (TIM)

- **What's New**: 이 논문에서는 Remote Photoplethysmography (rPPG) 신호를 학습하기 위한 새로운 반지도 학습 방법인 Semi-rPPG를 제안합니다. 이 방법은 소량의 레이블이 있는 데이터와 풍부한 레이블이 없는 데이터를 결합하여 모델 성능을 향상시킵니다. 특히 커리큘럼 가짜 레이블링(curriculum pseudo-labeling) 전략과 준주기 신호(quasi-periodic signals)를 위한 일관성 정규화(consistency regularization) 용어를 통해 노이즈의 영향을 최소화합니다.

- **Technical Details**: Semi-rPPG는 첫 번째 에폭(epoch) 동안 레이블이 있는 데이터로 모델을 훈련하고, 그 후 커리큘럼 가짜 레이블링 전략을 통해 레이블이 없는 데이터에 대한 가짜 레이블을 생성하는 과정을 포함합니다. 이 방법은 레이블이 부정확한 데이터를 필터링하면서 고품질의 레이블을 활용하여 최적의 학습 환경을 조성합니다. 또한, 휘트와 스트롱 증강 클립(augmented clips)을 사용하여 준주기 신호의 특정 시간적 특성(temporal features)을 추출하는 새로운 일관성 손실을 도입합니다.

- **Performance Highlights**: 이 연구에서 제안한 Semi-rPPG 방법은 세 가지 전통적인 반지도 학습 방법과 비교하여 다양한 프로토콜에서 가장 우수한 성능을 나타냈습니다. 여러 공공 데이터셋에 대한 intra-dataset 및 cross-dataset 평가를 통해 반지도 rPPG 측정을 위한 새로운 기준이 수립되었습니다. 제안한 방법의 유효성을 입증하기 위해 ablation study도 수행되었습니다.



### Pursuing Better Decision Boundaries for Long-Tailed Object Detection via Category Information Amoun (https://arxiv.org/abs/2502.03852)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: 이 논문에서는 객체 탐지(object detection)에서 카테고리 정보량(category information amount) 개념을 도입하고 이를 측정하는 방법을 제안합니다. 이 연구는 모델이 인스턴스 수가 비교적 균형 잡힌 데이터셋에서도 카테고리 편향(category bias)을 나타냄을 보여 주며, 단순히 인스턴스 수만으로는 이러한 현상을 설명할 수 없음을 시사합니다. 따라서 카테고리 정보량이 각 카테고리의 학습 난이도를 더 잘 반영한다고 주장합니다.

- **Technical Details**: 제안된 정보량 기반 각도 여유(Information Amount-Guided Angular Margin, IGAM) 손실 함수는 카테고리의 정보량에 따라 각 카테고리의 결정 공간을 동적으로 조정하여 장기간의 데이터셋에서 카테고리 편향을 줄이는 것을 목표로 합니다. IGAM Loss는 저비용의 엔드 투 엔드 훈련 전략을 통해 동적으로 정보량을 업데이트할 수 있습니다. 실험 결과는 LVIS v1.0 및 COCO-LT와 같은 장기간 테스트 데이터셋에서 IGAM 방법이 우수한 성과를 나타내고 있음을 보여줍니다.

- **Performance Highlights**: IGAM Loss를 적용한 결과, 기존의 많은 방법들을 초월하여 장기 데이터셋에서 성능을 개선했습니다. 특히, 희소 카테고리에 대한 모델의 정확성을 크게 향상시켰습니다. 상대적으로 균형 잡힌 Pascal VOC 데이터셋에서도 우리의 방법은 도전적인 카테고리에서 다른 접근법보다 현저히 우수한 성과를 보였습니다.



### Adapting Human Mesh Recovery with Vision-Language Feedback (https://arxiv.org/abs/2502.03836)
Comments:
          6 pages, 7 figures

- **What's New**: 이 연구는 인간 메쉬 복원(human mesh recovery)을 위한 새로운 접근 방법을 제안합니다. 주요 아이디어는 대형 비전-언어 모델(large vision-language models, VLMs)을 활용하여 2D 이미지 관찰을 보완하는 것입니다. 이를 통해 3D 인식을 개선하고 최적화 공간을 제한하는 방식으로, 텍스트와 3D 포즈 신호 간의 간극을 메꾸는 새로운 방법론을 개발합니다.

- **Technical Details**: 연구에서는 SMPL 모델을 사용하여 3D 사람을 6D 표현으로 나타냅니다. 초기 포즈 추정치는 회귀 기반(regression-based) 접근을 통해 계산하며, 이후 비전-언어 모델(VLM)을 통한 상호작용 설명을 이용하여 포즈를 개선합니다. 또한, VQ-VAE를 통해 텍스트와 포즈 간의 정렬을 정의하여 다중 모드 피드백(multi-modal feedback)을 결합하게 됩니다.

- **Performance Highlights**: 여러 벤치마크에서 실험 결과, 제안된 방법이 정확한 3D 포즈 추정(3D pose estimation) 및 모델-이미지 정렬(model-image alignment) 성능에서 우수한 결과를 보였습니다. 또한, 세부적인 텍스트 기반 상호작용 설명(interactive descriptions)이 인간 메쉬 복원 성능을 향상시키는 데 기여하였습니다. 본 연구의 코드도 공개될 예정입니다.



### Single-Domain Generalized Object Detection by Balancing Domain Diversity and Invarianc (https://arxiv.org/abs/2502.03835)
- **What's New**: 이 논문에서는 단일 도메인 일반화(object detection, S-DGOD)를 위한 새로운 모델인 Diversity Invariance Detection Model (DIDM)을 제안합니다. 기존 모델들은 주로 특징 불변성(feature invariance)에 집중했으나, 이는 도메인 간의 실제 차이를 간과할 수 있는 문제를 초래할 수 있습니다. DIDM은 도메인 특화 정보의 다양성과 도메인 간 불변성을 모두 고려하여 모델의 Robustness를 향상시키는 데 중점을 둡니다.

- **Technical Details**: DIDM은 두 가지 주요 모듈, Diversity Learning Module (DLM)과 Weighted Aligning Module (WAM)으로 구성됩니다. DLM은 도메인 특화 정보의 다양성을 유지하고 동시에 의미적 제한을 줄이기 위해 설계되었으며, 엔트로피 최대화 손실(entropy maximization loss)과 특징 다양성 손실(feature diversity loss)을 도입합니다. WAM은 도메인 간 특징 정렬(feature alignment)을 통해 도메인 불변성(domain invariance)을 높이면서도 특징 다양성을 보존하도록 설계되었습니다.

- **Performance Highlights**: 제안된 DIDM 모델은 다섯 가지 고유한 데이터 세트에서 실험을 수행하여 우수한 성능을 입증했습니다. 실험 결과는 모델이 다양한 기상 조건에서 효과적인 탐지 성능을 유지하며, 학습 과정에서의 정보 손실을 최소화하는 데 성공했음을 보여줍니다. 이러한 결과는 DIDM의 접근 방식이 단일 도메인 일반화 과제에서 효율적으로 작동함을 입증합니다.



### FE-UNet: Frequency Domain Enhanced U-Net with Segment Anything Capability for Versatile Image Segmentation (https://arxiv.org/abs/2502.03829)
- **What's New**: 이 논문은 이미지 분할(image segmentation) 작업에서 Convolutional Neural Networks (CNNs)와 Transformers의 차별적인 주파수 감도 기능을 비교하며, 이를 바탕으로 Wavelet-Guided Spectral Pooling Module (WSPM)과 Frequency Domain Enhanced Receptive Field Block (FE-RFB)를 도입합니다. 이러한 혁신을 통해 새로운 모델인 FE-UNet을 개발하여 일반화 능력을 증대시키면서 세분화 정확성을 높였습니다. 이 연구는 다양한 해양 동물 분할 및 용종(segmentation) 작업에서 최첨단 성능을 달성함을 보여줍니다.

- **Technical Details**: FE-UNet은 저주파(low-frequency) 정보를 강화하는 Deep Wavelet Convolution (DWTConv) 기법을 통합하여 고주파(high-frequency)와 저주파 성분 간의 균형을 맞춥니다. 또한, FE-RFB는 다중 스케일(multi-scale) 수용체 필드의 집합을 통해 인간 시각 시스템(human visual system)의 중심성과 경도(eccentricity)를 반영하는 기능을 제공합니다. 이러한 요소는 CNN과 인간 시각 시스템의 보완적인 강점을 활용하여 세분화 성능을 효과적으로 향상시키는 데 기여합니다.

- **Performance Highlights**: 다양한 해양 동물과 용종 데이터셋에 대한 실험 결과, FE-UNet은 뛰어난 유연성과 효과성을 입증하며 여러 작업에서 최첨단 성능을 기록했습니다. 실험에서는 특히 자연 이미지에서 고주파와 저주파 성분의 특징을 균형 있게 추출함으로써 세분화의 정확성을 크게 향상시킨 결과를 보였습니다. 이러한 높은 성능은 FE-UNet이 복잡한 주파수 특성을 효과적으로 처리할 수 있음을 시사합니다.



### FairT2I: Mitigating Social Bias in Text-to-Image Generation via Large Language Model-Assisted Detection and Attribute Rebalancing (https://arxiv.org/abs/2502.03826)
- **What's New**: FairT2I는 Text-to-Image (T2I) 생성에서 사회적 편견을 탐지하고 완화하는 새로운 프레임워크입니다. 이 프레임워크는 LLM(대형 언어 모델)을 활용하여 생성된 이미지에서의 잠재적 편견을 식별하고, T2I 모델 내에서 민감한 속성을 조정하여 이를 완화합니다. FairT2I는 이미지 생성의 질을 유지하면서도 편견을 상당히 줄일 수 있는 능력을 보여줍니다.

- **Technical Details**: FairT2I는 두 가지 주요 구성 요소로 구성되어 있습니다: 1) LLM 기반의 편견 탐지 모듈은 텍스트 프롬프트에 따라 생성된 이미지에서 사회적 편견을 식별합니다. 2) 속성 재조정 모듈은 식별된 편견을 완화하기 위해 T2I 모델 내에서 민감한 속성을 미세 튜닝합니다. 이러한 과정은 지능형 편견 탐지와 속성 샘플링을 통해 이루어지며, 기존의 T2I 디바이싱 접근법과는 달리 훈련 없이 추론 시에 동적으로 적용할 수 있습니다.

- **Performance Highlights**: FairT2I는 다양한 T2I 모델과 데이터셋에 걸쳐 실험을 수행한 결과, 이미지 생성을 위한 출력의 다양성을 크게 향상시키고 불리한 고정 관념을 줄이는 데 성공하였습니다. 사용자 연구와 비모수 분석을 통해, FairT2I가 단순한 비편향 기준선에 비해 현저히 개선된 결과를 도출했음을 입증했습니다. 또한 P2 데이터셋을 이용하여 인간 관찰자가 인지하기 어려운 미세한 편견도 감지할 수 있는 능력을 보여주었습니다.



### Optimized Unet with Attention Mechanism for Multi-Scale Semantic Segmentation (https://arxiv.org/abs/2502.03813)
- **What's New**: 이 논문은 기존의 Unet 모델에 Attention mechanism을 결합하여 개선된 Unet 모델을 제안합니다. 특히, 채널 Attention과 공간 Attention 모듈을 도입하여 모델이 중요한 특징에 집중할 수 있는 능력을 강화합니다. 다중 스케일 기능 융합 전략을 통해 skip connection을 최적화하여 글로벌 의미 정보와 세밀한 특징의 결합을 향상시킵니다.

- **Technical Details**: 제안된 모델은 Cityscapes 데이터셋을 기반으로 실험을 수행하며, FCN, SegNet, DeepLabv3+, PSPNet과 같은 전통적인 모델들과 비교합니다. 개선된 모델은 mean Intersection over Union (mIoU)와 pixel accuracy (PA)에서 각각 76.5%와 95.3%의 높은 성능을 달성합니다. 이는 복잡한 장면과 흐릿한 목표 경계를 처리하는 데 있어 이 방법의 우수성을 검증합니다.

- **Performance Highlights**: 실험 결과, 개선된 Unet 모델은 특히 복잡한 배경, 장거리 의존성, 다중 스케일 목표를 효과적으로 처리할 수 있는 능력을 보여줍니다. 또한, 이 모델은 자율 주행, 원격 감지 이미지 분석, 의료 이미지 처리와 같은 다양한 응용 분야에서 널리 활용될 수 있는 잠재력을 지니고 있습니다.



### DeblurDiff: Real-World Image Deblurring with Generative Diffusion Models (https://arxiv.org/abs/2502.03810)
- **What's New**: 본 논문은 Latent Kernel Prediction Network (LKPN)를 제안하여 이미지 디블러링(image deblurring)을 개선합니다. 기존의 디퓨전 모델(Diffusion models)을 활용하되, 성능을 극대화하기 위해 LKPN을 조건부 생성의 단계에 통합하여 처리합니다. 이는 특히 강한 흐림(blurriness) 효과를 화이트밸런스할 수 있도록 설계되었습니다.

- **Technical Details**: LKPN은 공간적으로 불변(local)인 커널 대신, 각 잠재적(latent) 픽셀에 적합한 커널을 예측하여 이들을 적응형 컨볼루션(element-wise adaptive convolution, EAC)으로 처리합니다. 이 방법은 입력 이미지의 구조적 정보를 보존하며, 각 단계에서의 반복적인 개선을 통해 명확한 구조와 세부 정보를 복원하는 데 유리합니다.

- **Performance Highlights**: 제안된 LKPN 기반의 방법은 기존의 첨단 디블러링 기법들보다 우수한 성능을 보여주며, 벤치마크와 실제 이미지 환경 모두에서 효과성을 입증하였습니다. 반복적인 개선 과정을 통해 점진적으로 높은 디테일을 복원하고, 최종 이미지의 품질을 크게 향상시켰습니다.



### Gaze-Assisted Human-Centric Domain Adaptation for Cardiac Ultrasound Image Segmentation (https://arxiv.org/abs/2502.03781)
- **What's New**: 이번 연구에서는 심장 초음파 이미지 분할을 위한 새로운 도메인 적응 방법인 Gaze-assisted Human-Centric Domain Adaptation (GAHCDA)을 제안합니다. GAHCDA는 의사의 시선 경로 정보를 활용해 도메인 간 적응을 지원합니다. 이 방법은 기존의 도메인 적응 방법에 비해 시각적 인지를 기반으로 하여 모델이 더 적절하게 타겟 도메인에 맞춰지도록 돕습니다. 실험 결과, GAHCDA는 GAN 기반 방법 및 자가 학습 방법보다 더 효과적으로 심장 초음파 이미지를 분할하는 것으로 나타났습니다.

- **Technical Details**: GAHCDA는 두 가지 주요 모듈로 구성되어 있습니다: Gaze Augment Alignment (GAA)와 Gaze Balance Loss (GBL)입니다. GAA 모듈은 교차 주의(cross-attention) 메커니즘을 사용하여 인간의 시각적 인지를 통합함으로써, 소스 도메인과 타겟 도메인 간의 공통된 특징을 추출합니다. GBL은 시선 히트맵을 활용하여 분할 결과의 구조적 유사성을 높이고, 과도한 또는 부족한 분할 문제를 해결하며, 의사가 주목하는 영역에 더 집중할 수 있도록 합니다.

- **Performance Highlights**: GAHCDA는 심장 초음파 이미지의 도메인 간 간극을 효과적으로 줄이고, 기존의 GAN 기반 방법이나 자가 학습 방법들에 비해 더욱 개선된 성능을 보여줍니다. 특히 GAA와 GBL 모듈을 통해 인지적 안내가 이루어짐으로써, 타겟 도메인에서의 분할 효과성이 크게 향상되었습니다. 이 연구는 임상 응용 가능성이 높으며 향후 의료영상 처리에 기여할 수 있는 잠재력을 내포하고 있습니다.



### Multi-Label Test-Time Adaptation with Bound Entropy Minimization (https://arxiv.org/abs/2502.03777)
Comments:
          Accepted for publication at ICLR 2025; 17 pages; 3 figures

- **What's New**: 본 논문에서는 다중 라벨 시나리오에서의 테스트 시 적응(Multi-Label Test-Time Adaptation, ML–TTA)을 탐구하며, 높은 확률의 여러 라벨의 신뢰도를 동시에 향상시키기 위한 새로운 최적화 목표인 Bound Entropy Minimization (BEM)을 제안합니다. 기존의 TTA 기법들은 주로 가장 높은 확률을 가진 클래스에만 초점을 맞춰 다른 긍정 라벨들의 적응을 저해하는 문제를 해결하고자 합니다. 이를 통해 BEM은 다중 라벨 특성에 맞춰 적절한 라벨 신뢰성을 보장하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: BEM은 두 가지 라벨 집합을 활용하여, 각각 약한 라벨 집합(weak label set)과 강한 라벨 집합(strong label set)을 구성하여 이들을 단일 라벨로 바인딩합니다. 그런 다음, 이들 약한 및 강한 라벨 집합에 대해 각각의 뷰(view)와 설명(caption)에 대해 인스턴스 레벨의 프롬프트를 학습합니다. 이러한 방법은 다양한 데이터셋에서 공유되는 시각-언어(Vision-Language) 공간 덕분에 가능합니다.

- **Performance Highlights**: ML–TTA와 BEM을 통합한 프레임워크는 MSCOCO, VOC, NUSWIDE 데이터셋을 포함하는 다양한 모델 아키텍처와 라벨 시나리오에서 최신 SOTA 방법들보다 우수한 성능을 나타냅니다. 이는 다중 라벨 인스턴스 적응을 가능하게 하여 CLIP의 적응성을 더욱 강화합니다.



### A Retrospective Systematic Study on Hierarchical Sparse Query Transformer-assisted Ultrasound Screening for Early Hepatocellular Carcinoma (https://arxiv.org/abs/2502.03772)
- **What's New**: 이 연구에서는 인공지능(AI)의 최신 발전을 활용하여 간세포암(HCC) 조기 진단의 정확도를 높이기 위한 혁신적인 모델인 계층 스파스 쿼리 변환기(Hierarchical Sparse Query Transformer, HSQformer)를 제안합니다. 이 모델은 합성곱 신경망(Convolutional Neural Networks, CNNs)과 비전 변환기(Vision Transformers, ViTs)의 장점을 통합하여 초음파 검사의 정확성을 강화합니다. 기존의 컴퓨터 보조 진단 시스템의 한계를 극복하고, 효율성과 사용 편의성을 높인 모듈형 아키텍처를 채택했습니다.

- **Technical Details**: HSQformer는 CVN과 ViT의 두 가지 주요 기능 추출기를 기반으로 하여 설계되었습니다. 이 모델은 교차 주의(Cross-attention), 자기 주의(Self-attention), 전문가 혼합(Mixture of experts, MoE) 모듈을 스태킹하여 AI의 진단 지원 가능성을 탐구합니다. 이를 통해 많게는 저수준 세부 정보와 패턴도 효과적으로 캡처하여 정밀한 진단이 가능하도록 합니다.

- **Performance Highlights**: HSQformer는 단일 센터, 다중 센터, 고위험 환자 테스트와 같은 세 가지 임상 시나리오에서 성능 테스트를 수행했으며, 기존의 최첨단 모델인 ConvNext와 SwinTransformer를 꾸준히 초월하는 결과를 보였습니다. 특히 HSQformer는 고급 방사선의사와 진단 정확도가 같았고, 저급 방사선의사들보다 월등한 성과를 기록하여 AI 도구의 임상적 잠재력을 명확히 보여주었습니다.



### RAMOTS: A Real-Time System for Aerial Multi-Object Tracking based on Deep Learning and Big Data Technology (https://arxiv.org/abs/2502.03760)
- **What's New**: 이 논문은 UAV(무인 항공기) 기반의 비디오에서 실시간 Multi-Object Tracking(MOT) 프레임워크를 제안합니다. Apache Kafka와 Apache Spark를 결합하여 효율적이고 신뢰할 수 있는 비디오 스트림 처리를 가능하게 했습니다. YOLOv8/YOLOv10 및 BYTETRACK/BoTSORT와 같은 최첨단 딥러닝 모델을 활용하여 정확한 물체 탐지와 추적을 수행합니다.

- **Technical Details**: 제안된 프레임워크는 Apache Kafka와 Apache Spark의 통합으로 구성돼 기본적인 물체 탐지 및 추적 알고리즘과 결합되어 있습니다. Kafka는 비디오 스트림을 처리하기 위한 메시지 브로커 역할을 하며, Spark는 데이터 처리를 위한 클러스터 컴퓨팅 프레임워크로 사용됩니다. 이 시스템은 모든 UAV로부터 대량의 비디오 데이터를 효율적으로 수신하고 분배하며, 병렬 처리를 통해 실시간으로 물체를 탐지하고 추적합니다.

- **Performance Highlights**: 제안한 시스템은 Visdrone2019-MOT 테스트 세트에서 HOTA 48.14와 MOTA 43.51을 달성했습니다. 단일 GPU에서 실시간 처리 속도 28 FPS를 유지하며, UAV 애플리케이션의 MOT 문제 해결에 딥러닝과 빅데이터 기술의 효과를 보여줍니다. 이러한 성과는 보안, 교통 모니터링, 재난 구조 작전 등 여러 중요한 응용 분야에서의 실시간 상황 인식을 가능하게 합니다.



### Improving Adversarial Robustness via Phase and Amplitude-aware Prompting (https://arxiv.org/abs/2502.03758)
- **What's New**: 본 논문에서는 Phase and Amplitude-aware Prompting (PAP) 방어 메커니즘을 제안합니다. 이 방법은 각 클래스에 대해 위상 및 진폭 프롬프트를 구성하여 모델의 예측을 안정화합니다. 특히, 모델의 견고한 성능을 위해 훈련 중 프롬프트에 대한 가중치를 조정하는 방식이 특징입니다.

- **Technical Details**: 저자들은 적대적 설정에서classification 작업을 수행합니다. Discrete Fourier Transform (DFT)과 그 역변환인 Inverse Discrete Fourier Transform (IDFT)을 사용하여 이미지의 위상 및 진폭 스펙트럼을 추출합니다. 이들은 각각 특징적인 구조와 질감을 나타내며, 적대적 예시와의 비교실험을 통해 이들의 가중치를 조정하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, PAP 방법이 최신 기술 대비 우수한 성능을 나타내며, 자연 이미지를 평가하는 것뿐만 아니라 적대적 훈련된 모델에 대해서도 효과적임을 입증했습니다. 여러 공격 방식에 대한 robust한 성능을 발휘하며, 프롬프트 선택에서의 효율성을 통해 더 나은 전이 가능성을 달성하였습니다.



### Brain Tumor Identification using Improved YOLOv8 (https://arxiv.org/abs/2502.03746)
- **What's New**: 이번 연구에서는 MRI 이미지 내의 뇌종양을 정확하게 탐지하기 위해 수정된 YOLOv8 모델을 제안합니다. 이 모델은 탐지 헤드에서 Non-Maximum Suppression(NMS) 알고리즘을 Real-Time Detection Transformer(RT-DETR)으로 대체하여, 손으로 설계된 불필요한 경계 상자를 제거합니다. 또한, 일반적인 컨볼루션 블록을 ghost convolution으로 교체하여 연산 및 메모리 비용을 줄이며 빠른 추론을 가능하게 합니다.

- **Technical Details**: 본 연구에서 제안하는 모델은 YOLOv8의 백본에 vision transformer 블록을 도입하였으며, 이를 통해 문맥 인식을 기반으로 한 특징을 추출합니다. RT-DETR은 핸드 디자이닝 구성 요소를 제거하여 현대적인 딥러닝 모델의 경량화를 이루었습니다. ghost convolution은 고속 처리를 가능하게 하며, 자원 제약 환경에서도 높은 정확성을 유지합니다.

- **Performance Highlights**: 제안된 모델은 공개된 뇌종양 데이터셋을 사용하여 훈련되었으며, 원본 YOLOv8 모델 및 다양한 객체 탐지기(Faster R-CNN, Mask R-CNN, YOLO 계열, SSD, RetinaNet 등)보다 우수한 성능을 나타냈습니다. 특히, 제안된 모델은 0.91 mAP(mean Average Precision)@0.5를 달성하여 d극적인 성과를 보였습니다.



### Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And Mor (https://arxiv.org/abs/2502.03738)
- **What's New**: 이 논문에서는 Vision Transformer (ViT)에서의 패치화(patchification) 기반 압축 인코딩으로 인한 정보 손실을 철저히 분석합니다. 특히 패치 크기 감소가 모델의 예측 성능에 어떻게 영향을 미치는지를 실험을 통해 관찰하였으며, 1x1 패치 크기, 즉 픽셀 토큰화의 중요성을 강조합니다. 다양한 시각 작업과 다양한 입력 규모에서 유용하게 적용 가능한 결과를 도출하였으며, 모델의 테스트 정확도 또한 84.6%에 도달하였습니다.

- **Technical Details**: 기존 ViT 아키텍처의 패치화 기술은 이미지의 공간적 크기를 줄여 토큰 시퀀스를 단축시키고 계산 비용을 절감할 수 있어 널리 사용되고 있습니다. 그러나 이 연구에서는 패치 크기가 16x16에서 1x1로 감소할수록 테스트 손실이 지속적으로 감소하는 비율의 증가가 관찰되었음을 보였습니다. 이러한 경향은 다양한 비전 작업과 아키텍처에서도 일관되게 나타났으며, 모델에서 패치 크기가 중요한 조정 요소가 될 수 있음을 시사합니다.

- **Performance Highlights**: 이번 연구의 실험 결과, Deit-Base 모델에서 패치 크기를 8x8로 줄이면 ImageNet-1k 분류 벤치마크에서 정확도가 81.8%에서 83.5%로 향상되었습니다. 또한 패치화 과정을 제거하고 50,176개의 토큰으로 구성된 초장기 시각 시퀀스를 생성하여 테스트 정확도가 82.6%에서 84.6%로 향상되었습니다. 이러한 결과는 디코더 헤드가 필수적이지 않다는 점을 보여주며, 속성 세분화 작업에서 성능 저하 없이 결과를 얻게 하였습니다.



### DICE: Distilling Classifier-Free Guidance into Text Embeddings (https://arxiv.org/abs/2502.03726)
- **What's New**: 본 논문에서는 DIstilling CFG by enhancing text Embeddings (DICE)라는 새로운 접근 방식을 소개합니다. DICE는 기존의 Classifier-Free Guidance (CFG)에 의존하지 않고, 텍스트 임베딩을 개선하여 고품질 이미지를 생성하는 방법을 제시합니다. 이로써 DICE는 높은 품질의 정렬된 이미지 생성을 가능하게 하면서, 계산 자원 소모를 줄이고 기존의 이론적 배경을 유지할 수 있습니다.

- **Technical Details**: DICE는 텍스트 임베딩을 무작위 샘플링 과정에서 최적화하여 텍스트와 이미지 간의 정밀한 정렬을 가능하게 합니다. 본 방법은 경량형 향상기를 사용해 텍스트 임베딩을 CFG 기반 감독 아래에서 개선하여, DICE에서 생성된 임베딩은 다양한 기본 모델에서도 활용 가능한 특성을 갖고 있습니다. 이를 통해 DICE는 기존의 텍스트-이미지 모델들과 비교하여 별도의 CFG 모델 평가 없이도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 다양한 Stable Diffusion v1.5 변형, SDXL, PixArt-α를 포함한 실험 결과, DICE는 기존의 CFG 기반 방법들과 동일한 품질의 이미지를 생성하면서도 샘플링 속도를 크게 개선할 수 있음을 보여주었습니다. 또한, DICE는 이미지 품질 향상을 위해 부정적인 프롬프트를 통한 이미지 편집을 지원합니다. 이러한 특성을 통해 DICE는 다양한 상황에서도 효과적으로 활용될 수 있음을 입증하였습니다.



### MD-BERT: Action Recognition in Dark Videos via Dynamic Multi-Stream Fusion and Temporal Modeling (https://arxiv.org/abs/2502.03724)
- **What's New**: 이 연구는 저조도 비디오에서의 동작 인식을 위한 새로운 접근법인 MD-BERT를 소개합니다. MD-BERT는 감마 보정(gamma correction)과 히스토그램 평활화(histogram equalization) 기법을 포함한 다중 스트림(multi-stream) 아키텍처를 채택하여 저조도 환경에서의 도전 과제를 해결합니다. 또한, Dynamic Feature Fusion(DFF) 모듈을 통해 다양한 요소를 통합하여 비디오 프레임 간의 복잡한 상호작용을 효과적으로 포착합니다.

- **Technical Details**: MD-BERT는 Raw dark frames, gamma-enhanced frames, histogram-equalized frames의 세 가지 보조 입력 스트림을 통해 다양한 시각적 특징을 효과적으로 처리합니다. DFF 모듈은 지역적(local) 및 전역적(global) 맥락 정보를 조화롭게 통합하여 고유한 시각적 정보를 강조할 수 있도록 설계되었습니다. BERT 기반의 아키텍처를 통해 긴 시간의 종속성을 포착하며, 포지셔널 인코딩(positional encoding)과 멀티헤드 어텐션(multi-head attention)을 사용하여 과제 인식을 향상시킵니다.

- **Performance Highlights**: ARID V1.0 및 ARID V1.5 데이터셋에서 MD-BERT는 기존의 방법들보다 우수한 성능을 보여주며, 저조도 환경에서도 최신 성능 기준을 확립합니다. 각 입력 스트림의 개별 기여도를 강조하는 Ablation 연구도 실시하여 제안된 DFF 및 BERT 모듈의 효과성을 입증했습니다. 이 연구는 저조도 비디오에서 행동 인식의 새로운 수준을 제시하며, 비디오 인식 기술의 발전에 기여할 것으로 기대됩니다.



### Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignmen (https://arxiv.org/abs/2502.03714)
- **What's New**: 본 연구는 다수의 사전 훈련된 깊은 신경망(DNN)에서 공유되는 해석 가능한 개념을 발견하고 정렬하기 위한 'Universal Sparse Autoencoders(USAEs)' 프레임워크를 제안합니다. 기존의 개념 기반 해석 가능성 방법들은 단일 모델에 초점을 맞춘 반면, USAEs는 여러 모델의 내부 활성화를 동시에 재구성하고 해석할 수 있는 보편적인 개념 공간을 공동 학습합니다. 이를 통해 다양한 작업, 아키텍처 및 데이터 세트를 걸쳐 일반화되는 개념의 공통 요인을 포착하게 됩니다.

- **Technical Details**: USAEs는 단일의 과잉완전한 희소 오토인코더(SAE)를 훈련하여 어떤 모델의 활성화를 입력받고 다른 모델의 활성화에 근사화하는 구조로 설계되었습니다. 전통적인 방법과는 달리 USAE는 end-to-end 방식으로 개념 정렬을 적용하여 여러 모델 간의 효율적인 상호 작용을 가능하게 합니다. 이 방식은 DNN의 내부 표현을 이해하는 데 유용하며, 다수의 비전 모델에 적용하여 저수준 특징부터 고수준 구조까지 다양한 보편 개념을 발견했습니다.

- **Performance Highlights**: USAEs는 세 가지 다양한 비전 모델에 적용되어 흥미로운 발견을 제시하였습니다. 첫째, 낮은 추상화 수준부터 높은 추상화 수준까지 광범위한 보편 개념을 발견하였습니다. 둘째, 개념의 보편성에 중요한 상관관계를 관찰하였으며, DinoV2는 다른 모델에 비해 독특한 특징을 갖고 있다는 정량적 및 정성적 증거를 제공하였습니다. 또한, 보편적 훈련은 모델 특화형 SAE 훈련에서는 발견되지 않는 공유 표현을 허용합니다.



### Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Fr (https://arxiv.org/abs/2502.03687)
- **What's New**: 이번 연구에서는 2D 의료 영상 분류를 위한 클래스 조건부 확산 모델(class conditional diffusion models)의 가능성을 처음으로 탐구했습니다. 새로운 다수 투표 기반 방법을 개발하여 의료 확산 분류기의 성능을 향상시키는 것을 목표로 하고 있습니다. 특히, 기존의 판별 모델(discriminative classifiers)과 비교해도 늦은 훈련 없이 뛰어난 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 확산 모델(difussion models)을 사용하여 의료 영상 분류 작업을 위한 새로운 방법론을 제시합니다. 확산 모델은 데이터를 점진적으로 노이즈로 변환하는 고정된 전방 과정과 이 과정을 통해 학습된 역방향 모델(backward model)로 구성됩니다. 새로운 다수 투표 알고리즘, 반사실적 설명(counterfactual explainability), 불확실성 정량화(uncertainty quantification) 기능을 포함하여 성능을 향상시키는 방법을 소개합니다.

- **Performance Highlights**: CheXpert와 ISIC 흑색종 데이터셋에서의 실험 결과, 확산 모델에서 파생된 분류기가 기존의 의료 영상 판별 모델과 경쟁할 만한 성능을 발휘했습니다. 예를 들어, Stable Diffusion 모델은 ISIC와 CheXpert 각각 100% 및 95%의 분류 정확도를 달성했으며, 신뢰도가 높아짐에 따라 모델의 정확도도 크게 향상되었습니다. 이러한 결과는 모델이 제공하는 불확실성 분석이 의료 분야의 신뢰성과 안전성을 높일 수 있음을 보여줍니다.



### An Empirical Study of Methods for Small Object Detection from Satellite Imagery (https://arxiv.org/abs/2502.03674)
- **What's New**: 이 논문은 원거리 감지 이미지에서 소형 객체를 탐지하기 위한 방법론을 리뷰하고, 성능과 기술적 과제를 분석하기 위한 실증적 평가는 네 가지 최신 기법을 집중적으로 다룹니다. 특히, 도시 위성 이미지에서의 자동차 탐지와 농업용 토지의 위성 이미지에서의 벌통 탐지를 사례로 활용하고 있습니다. 기존 문헌을 바탕으로 소형 객체 탐지에 적합한 여러 최상위 방법을 식별하였으며, xView 및 SkySat와 같은 고해상도 위성 이미지 데이터셋을 실험에 사용했습니다.

- **Technical Details**: 최근 몇 년 동안 다양한 학습 기반(object detection) 방법들이 발전하여 YOLO, SSD, R-CNN 계열 등 여러 기술들이 등장했습니다. 위성 이미지 분석에서의 소형 객체 탐지 과제는 기존 방법들이 직면한 주요 도전 과제를 드러내며, 특히 낮은 해상도에서의 맥락 정보 부족 문제와 불균형한 포지티브 및 네거티브 샘플의 부재가 분석되었습니다. 이 연구에서는 Convolution-based 방법, 특히 YOLO와 SSD 등의 성능을 더 향상시키기 위해 다중 해상도 특징 추출 방법인 Feature Pyramid Networks (FPNs)를 통하여 해결책을 모색합니다.

- **Performance Highlights**: 이 논문에서 수행한 실험은 소형 객체 탐지의 성능을 평가하였고, 다양한 방법이 서로 다른 환경에서 어떻게 작동하는지를 분석했습니다. YOLO 모델은 반복적으로 개선되어 소형 객체 탐지에서의 정확성과 계산 시간을 향상시켰으며, SSD 모델은 빠르고 간단한 방법으로 여전히 소형 객체 탐지에서 유의미한 역할을 하고 있음을 보여주었습니다. 연구 결과, 특히 Dense Objects와 같은 복잡한 상황에서 소형 객체 탐지의 정확성을 높이기 위해 다단계 접근 방식인 Cascade R-CNN이 강력한 성능을 발휘한다는 점을 강조했습니다.



### A Study in Dataset Distillation for Image Super-Resolution (https://arxiv.org/abs/2502.03656)
- **What's New**: 이 논문은 대규모 데이터셋을 소형화된 합성 샘플로 압축하는 개념인 데이터셋 증류(Dataset Distillation)에 대해 탐구하고 있습니다. 특히, 이미지 분류에 주로 연구가 집중된 점을 넘어 이미지 초해상도(Super-Resolution, SR) 분야로의 응용을 확대하였습니다. 실험을 통해 전체 데이터셋과 유사한 SR 성능을 유지하면서 데이터셋 크기를 91.12% 줄일 수 있음을 보여줍니다.

- **Technical Details**: 고해상도(HR) 이미지 재구성을 위해 저해상도(LR) 이미지에서 SR 모델을 훈련시키는 데 필요한 데이터셋 증류 기법을 사용하고 있습니다. SR 품질을 유지하면서 훈련 데이터의 양을 줄이는 데 중점을 두며, 독특하게도 픽셀 공간과 잠재 공간을 비교 분석합니다. 최적화 전략 및 초기화 방법에 대해서도 심도 있는 분석을 수행하여 메모리 효율성과 계산 비용을 최적화하고자 합니다.

- **Performance Highlights**: 본 연구는 합성 데이터셋을 통해 데이터 크기를 대폭 줄이면서도 SR 성능에서 경쟁력 있는 결과를 달성하는 방법론을 제시합니다. 또한, 추가적인 통찰력을 제공함으로써 향후 메모리 효율적인 SR 모델 교육에 대한 기초를 마련하고 있습니다. 결과적으로, 데이터셋 증류와 SR 간의 간극을 메우는 데 중요한 연구 방향을 설정하고 있습니다.



### All-in-One Image Compression and Restoration (https://arxiv.org/abs/2502.03649)
Comments:
          Accepted to WACV 2025 (oral)

- **What's New**: 이번 연구에서는 다양한 손상이 있는 이미지를 압축하면서 동시에 복원을 수행할 수 있는 통합 프레임워크를 제안합니다. 기존의 이미지 압축 기법들은 일반적으로 '깨끗한' 이미지만을 위한 것이었으나, 우리의 방법은 손상이 있는 이미지에서도 우수한 압축 성능을 보여줍니다. 두 가지 주요 정보 집합 기법을 적용하여 이미지의 진정한 내용을 복원하면서도 손상을 효과적으로 제거하는 접근 방식을 제공합니다.

- **Technical Details**: 제안된 방법은 인코더와 디코더, 그리고 공간 엔트로피 모델을 포함하는 통합 네트워크로 구성됩니다. 여기에서 채널 그룹 주의 기법(C-GA)과 공간적으로 분리된 주의 기법(S-DA)을 사용하여 이미지의 내용과 손상을 구분합니다. 이러한 두 가지 주의 메커니즘은 하이브리드 주의 변환기 블록(HATB)에 통합되어 다양한 스케일에서 학습하는 방식으로 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다양한 손상 이미지를 효과적으로 처리할 수 있으며, 깨끗한 데이터에 대해서도 rate-distortion (RD) 성능을 저하시키지 않고 강력한 일반화 능력을 보여줍니다. 또한, 기존의 연속형 솔루션에 비해 높은 계산 효율성을 유지하며, 다양한 실제 시나리오에서도 잘 작동함을 입증합니다.



### Towards Physical Understanding in Video Generation: A 3D Point Regularization Approach (https://arxiv.org/abs/2502.03639)
Comments:
          Project Page: \url{this https URL}

- **What's New**: 새로운 비디오 생성 프레임워크를 제시하며, 이는 3차원(3D) 기하학(geometry)과 동적 인지(dynamics)를 통합합니다. 우리는 2D 비디오에 3D 포인트 궤적을 추가하고 이를 픽셀(픽셀) 공간에서 정렬하여, 3D 인식 비디오 데이터셋인 PointVid를 생성했습니다. 이 데이터셋은 잠재적 확산 모델(latent diffusion model)을 미세 조정하여 3D 카르테시안 좌표에서 2D 객체를 추적하는 능력을 부여합니다.

- **Technical Details**: 우리는 3D 포인트 클라우드(3D point cloud)를 활용하여 비디오 확산 모델을 증강하고 규제하는 방법을 제안합니다. 이 모델은 2D 비디오를 외부 차원 데이터로 풍부하게 만들어 기존 RGB 비디오 모델에서는 결핍된 공간 정보를 보완합니다. PointVid 데이터셋을 통해 객체의 움직임을 정밀하게 추적하여 비디오 내 3D 형상 및 동작을 통합할 수 있습니다.

- **Performance Highlights**: 실험을 통해 복잡한 상호작용이 있는 시나리오에서 더 나은 품질의 비디오를 생성함을 입증하였으며, 특히 형상과 동작의 매끄러운 변화를 달성했습니다. 3D 궤적에 의한 지침을 통해, 우리의 모델은 기존 작업들에 비해 시각적으로 그럴듯한 비디오를 생성할 수 있는 순조로운 전환을 보여줍니다.



### REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations (https://arxiv.org/abs/2502.03629)
- **What's New**: 기존의 이미지 편집 모델들이 실제 사용자 요구를 충족시키지 못하는 문제를 다룬 REALEDIT(Real Edit)라는 새로운 대규모 데이터셋을 소개합니다. 이 데이터셋은 Reddit에서 수집된 진짜 사용자 요청과 인간에 의해 편집된 이미지를 포함하고 있으며, 9300개의 평가 예제를 포함하고 있어 다양한 실제 요구를 테스트할 수 있습니다. REALEDIT는 인간의 편향을 감소시키고 다양한 사용자 요구를 반영하는 구조로 설계되었습니다. 이 연구의 결과는 기존 모델이 이러한 작업에서 부족한 점이 있음을 강조합니다.

- **Technical Details**: REALEDIT 데이터셋은 사람에 의해 편집된 이미지와 그에 대한 요청을 기반으로 한 48K의 훈련 예제와 9300개의 테스트 예제를 포함하고 있습니다. 이를 위해 데이터 수집 파이프라인을 구성하였으며, 두 개의 주요 서브레딧인 r/PhotoshopRequest와 r/estoration에서 받은 요청을 기반으로 하여 데이터를 구성하였습니다. 편집된 이미지들은 원본 이미지와 편집 지침을 포함하여 사용자의 실제 요청을 반영하는 형태로 수집되었습니다. 이 데이터셋은 진짜 편집 요구 사항을 더 효과적으로 반영하며, 기존의 합성 데이터세트의 한계를 극복하고자 합니다.

- **Performance Highlights**: REALEDIT 모델은 기존의 최고 성능 모델보다 165포인트 높은 Elo 스코어를 기록하며 주목받았습니다. 또한 VIEScore와 같은 자동화된 메트릭에서도 92%의 향상을 보여줍니다. 모델은 Reddit에서 새로운 요청에 대해 긍정적인 피드백을 받았으며, 편집 외에도 진짜 편집된 이미지의 탐지 성능이 향상될 가능성도 확인되었습니다. 이 연구는 이미지 편집 작업 외에도 다양한 AI 기반 응용 분야에 대한 데이터셋의 유용성을 강조합니다.



### The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering (https://arxiv.org/abs/2502.03628)
- **What's New**: 본 논문은 Large Vision-Language Models (LVLMs)에서 발생하는 환각 현상(hallucination)에 대해 연구하였습니다. LVLMs는 시각적 및 텍스트 정보를 효과적으로 처리할 수 있지만, 종종 의미는 일관되나 시각적으로 지지되지 않는 내용을 생성을 야기합니다. 이를 해결하기 위해, 논문에서는 새로운 VISTA(VIsual Information Steering with Token-logit Augmentation) 프레임워크를 제안하였으며, 이는 환각을 줄이는 동시에 진정한 정보를 증진시키는 방법을 제공합니다.

- **Technical Details**: 연구에서 LVLMs의 토큰 로짓(token logits) 랭킹을 조사하여 세 가지 주요 패턴을 발견하였습니다. 첫 번째는 시각 정보의 점진적 손실로, 이는 진정한 토큰의 우선 순위가 감소하고 환각 토큰의 우선 순위가 상승하는 현상이 발생합니다. 두 번째 패턴은 의미 있는 토큰이 모델의 중간 계층에서 최고 활성도를 나타내며, 세 번째는 시각적으로 진정한 토큰이 최종 단계에서 결정되지 않을 수 있지만 여전히 높은 랭킹을 유지한다는 것입니다. 이를 바탕으로 VSTA는 시각 정보 강화 및 기능적 토큰 사용을 조합하여 작동합니다.

- **Performance Highlights**: VISTA는 실험을 통해 기존 방법에 비해 환각 현상을 약 40% 감소시키는 효과를 보여주었습니다. 다양한 모델 아키텍처(LLaVA, Shikra, MiniGPT-4, InstructBLIP)에서 높은 성능을 나타내며, 추가적인 훈련이나 모델 수정이 필요 없다는 장점이 있습니다. 이 프레임워크는 다양한 디코딩 전략에 쉽게 적용 가능하며, 열린 생성(open-ended generation) 및 시각적 질문 응답(visual question answering) 등 여러 평가 프로토콜에서 우수한 결과를 제공합니다.



### DynVFX: Augmenting Real Videos with Dynamic Conten (https://arxiv.org/abs/2502.03621)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 실제 비디오에 새로운 동적 콘텐츠를 증강하는 방법을 제시합니다. 사용자가 제공하는 텍스트 지시 사항을 바탕으로, 기존 장면과 자연스럽게 상호작용하는 동적 객체 또는 복잡한 장면 효과를 합성하는 것이 목표입니다. 이 방법은 원래 장면과의 조화를 유지하면서 카메라 움직임, 가리기, 다른 동적 객체와의 상호작용을 고려하여 새로운 콘텐츠를 원활히 통합합니다.

- **Technical Details**: 이 방법은 사전 훈련된 텍스트-비디오 및 비전 언어 모델을 활용하여 훈련 없이 새로운 콘텐츠를 합성합니다. 특히, 증강된 장면을 시각화하고 사용자 지침을 상세한 프롬프트로 변환하기 위해 비전-언어 모델을 사용합니다. 핵심적인 기술로는 Anchor Extended Attention을 통해 원본 비디오에서 추출된 키/값 세트를 추가 문맥으로 포함시켜 정밀한 위치 조정이 가능합니다.

- **Performance Highlights**: 이 방법은 다양한 실제 비디오에 대해 여러 종류의 편집을 적용하여 그 효과성을 입증합니다. 기존의 VFX 자산이나 마스크를 요구하지 않고도 동적 콘텐츠를 자동으로 통합할 수 있다는 점에서 혁신적입니다. 추가적으로, 자동화된 VLM-기반 평가 지표를 제안하여 원래 콘텐츠 보존 및 새로운 콘텐츠 조화를 평가합니다.



### Solar Panel Mapping via Oriented Object Detection (https://arxiv.org/abs/2502.03592)
- **What's New**: 이번 연구에서는 기존의 태양광 패널 감지 방식을 개선하기 위해 회전 객체 감지(rotated object detection) 아키텍처를 적용한 끝에서 끝까지(End-to-End) 딥러닝 프레임워크를 제안합니다. 이 방법은 대규모 항공 이미지를 통해 개별 태양광 패널의 위치를 신속하게 식별할 수 있도록 설계되었습니다. 이 연구는 현재까지의 태양광 패널 탐지를 회전 객체 감지 문제로 모델링한 최초의 사례로, 이를 통해 각 지역의 태양광 패널을 효율적으로 매핑할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 패널의 방향에 관계없이 개별 태양광 패널을 정확히 로컬라이즈하는 것을 목표로 하고 있습니다. 이를 위해 태양광 패널의 진실 그라운드(truth annotations)는 회전된 경계 상자(rotated bounding boxes)로 표현되어, 각각의 상자는 중심 좌표(x,y)와 너비(w), 높이(h), 회전 각도(θ)를 포함하는 5차원 튜플로 구성됩니다. 이 프레임워크는 Faster R-CNN 아키텍처를 기반으로 하며, 회전된 경계 상자 제안(RRPN) 및 회전된 RoI(Region of Interest) 풀링 모듈을 활용하여 패널 인스턴스를 정확히 로컬라이즈합니다.

- **Performance Highlights**: 제안된 프레임워크는 미국 전역에서 수집된 다양한 태양광 발전소 데이터를 기반으로 평가되었으며, 평균 정밀도(mAP) 점수는 83.3%에 도달했습니다. 또한, 태양광 패널의 시각적 특징이 자연 이미지와 비교하여 크게 변하지 않기 때문에, 표본 샘플링과 데이터 증강을 통해 모델 성능을 향상시킬 수 있었습니다. 이러한 결과는 향후 태양광 패널의 감지 및 맵핑 시스템의 자동화를 촉진할 것으로 기대됩니다.



### Clinically-Inspired Hierarchical Multi-Label Classification of Chest X-rays with a Penalty-Based Loss Function (https://arxiv.org/abs/2502.03591)
Comments:
          9 pages with 3 figures, for associated implementation see this https URL

- **What's New**: 이번 연구에서는 임상 해석 가능성을 향상시키면서도 단일 모델, 단일 실행 학습 파이프라인을 유지하는 다중 레이블 흉부 X선(X-ray, CXR) 이미지 분류에 대한 새로운 접근 방식을 제안합니다. CheXpert 데이터셋과 VisualCheXbert에서 파생된 레이블을 활용하여 진단 간의 임상적으로 유의미한 관계를 포착하기 위해 계층적 레이블 그룹화를 통합하였습니다. 이를 통해 고안한 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 통해 레이블 의존성을 강화하였습니다.

- **Technical Details**: 연구진은 어린 레이블이 상위 레이블에 대한 긍정적인 예측이 없이 긍정적으로 예측될 때 패널티를 적용하는 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 제안하였습니다. 패널티 전략으로는 고정 패널티 접근 방식과 데이터 기반 패널티 방법을 탐색하였으며, 데이터 기반 방법은 상위-하위 레이블 간의 의존 가능성에 따라 패널티를 조절합니다. 모델 성능 향상을 목표로, 학습 데이터 установ 일부 조정과 함께 계층적 그룹 재구성의 효과를 종합적으로 분석하였습니다.

- **Performance Highlights**: 제안된 프레임워크는 CheXpert 데이터셋에서 0.9034의 분류 성능을 달성하였으며, 이는 계층적 구조와 커스텀 HBCE 손실 함수의 효능을 입증합니다. 데이터 기반 패널티는 예측 정확도를 향상시킬 가능성을 보였으며, 시각적 설명과 불확실성 추정이 모델 해석 가능성과 투명성을 더욱 높였습니다. 모든 코드, 모델 구성 및 실험 세부 사항은 공공 Git 저장소에 공개되어 연구의 투명성과 재현성을 촉진하고 있습니다.



### CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally (https://arxiv.org/abs/2502.03566)
- **What's New**: 이 논문에서는 CLIP의 compositionality(구성 가능성) 문제를 다룹니다. 이전 연구들이 CLIP이 bag-of-words(BoW) 모델처럼 작동한다고 주장했으며, 그 결과로 객체와 속성을 올바르게 연결하지 못하는 경향이 있음을 보여줍니다. 이 연구에서 제안된 LABCLIP은 속성-객체 바인딩 정보의 문제를 해결하기 위해 선형 변환을 적용하여 CLIP의 성능을 개선합니다.

- **Technical Details**: LABCLIP은 원본 텍스트 임베딩에 선형 변환을 적용하여 코사인 유사도(cosine similarity)를 계산하기 전에 속성 정보를 추출합니다. 이를 통해 복잡한 이미지와 텍스트 쌍에서 속성과 객체를 정확하게 연결합니다. 연구진은 CLIP의 인코더를 동결한 상태에서 합성 부정 샘플을 사용하여 변환을 훈련했으며, 다양한 벤치마크에서 개선된 성능을 입증했습니다.

- **Performance Highlights**: 이 연구에서는 ARO, SugarCrepe, COCO와 같은 여러 벤치마크에서 LABCLIP의 성능을 검증했습니다. CLIP이 객체에 속성을 정확하게 바인딩할 수 있는 능력이 크게 향상되었으며, 이를 통해 Compositional understanding(구성 이해력)의 발전을 이끌었습니다. 이 결과는 CLIP 기반 VLMs(비전-언어 모델)의 발전에 중요한 기여를 할 것으로 기대됩니다.



### Efficient Global Neural Architecture Search (https://arxiv.org/abs/2502.03553)
Comments:
          CAIP2023

- **What's New**: 본 논문에서는 Neural Architecture Search (NAS)의 새로운 접근 방식을 제안하여, 전체 아키텍처 대신 매크로-마이크로 검색 공간을 탐색함으로써 보다 자동화된 설계를 실현합니다. 새로운 구조 인식을 통한 근사화 및 가변 교육 방식이 도입되어 서로 다른 네트워크 간의 공정한 비교를 가능하게 합니다. 이를 통해 기존의 연구보다 2배에서 4배 더 빠른 성능을 자랑하며, EMNIST와 KMNIST에서 새로운 최첨단 결과를 달성합니다.

- **Technical Details**: 논문은 NAS의 효율성을 높이기 위해 모듈형 검색을 사용하고, 각 네트워크의 상대 순위를 결정하기 위해 아키텍처 인식 근사법을 도입합니다. 이는 개별 네트워크에 대한 훈련 프로토콜을 다르게 설정함으로써 이루어집니다. 또한, 검색 전략을 매크로와 마이크로 아키텍처 발견으로 분리하여 네트워크의 복잡성과 데이터셋의 난이도 사이의 적절한 균형을 찾습니다.

- **Performance Highlights**: 제안된 프레임워크는 CIFAR-10, CIFAR-100 및 FashionMNIST 데이터셋에서 경쟁력 있는 아키텍처를 발견하고, 기존의 수작업으로 설계된 네트워크와 비교하여 차별화된 성능을 나타냅니다. 특히, 최신 얼굴 인식 애플리케이션에서도 뛰어난 성능을 발휘하며, Tinyface 데이터셋을 사용한 1:1 얼굴 검증 및 상위-n 식별 작업에서 Adaface보다 나은 결과를 보여줍니다.



### Kronecker Mask and Interpretive Prompts are Language-Action Video Learners (https://arxiv.org/abs/2502.03549)
- **What's New**: 본 논문에서는 CLIP를 비디오 도메인에 효과적으로 적용하기 위한 새로운 접근법인 CLAVER(Contrastive Language-Action Video Learner)를 제안합니다. 기존 연구들이 텍스트 또는 비주얼 브랜치 하나에만 집중했던 반면, CLAVER는 두 브랜치를 모두 활용하여 동작 인식의 성능을 개선합니다. 이것은 정적 시각 객체와 구체적 명사의 정렬에서 동적 행동과 추상적 동사의 정렬로 CLIP의 초점을 전환합니다.

- **Technical Details**: CLAVER는 시간 모델링을 위한 새로운 Kronecker mask attention을 도입합니다. 이 방법은 각 토큰에 대한 시간 수용 필드를 확장하고, 시공간 이질성 유도 편향을 효과적으로 적용하여 시공간 동질화 문제를 완화합니다. 또한, 대규모 언어 모델을 활용하여 다양한 의미론적으로 풍부한 행동의 해석 프롬프트를 생성하여 텍스트 브랜치에 변화를 줍니다.

- **Performance Highlights**: CLAVER는 Kinetics-400 및 Kinetics-600 데이터셋에서 뛰어난 성능을 보였으며, HMDB-51 및 UCF-101에서는 제로샷 및 몇 샷 시나리오에서 경쟁력을 갖췄습니다. 본 연구는 비디오 행동 인식에 있어 CLIP의 적합성을 높이기 위한 효과적인 방법론을 제시합니다.



### Mapping and Localization Using LiDAR Fiducial Markers (https://arxiv.org/abs/2502.03510)
Comments:
          PhD thesis

- **What's New**: 이 논문에서는 LiDAR 기반의 fiducial marker인 LFM(LiDAR fiducial markers)의 새로운 프레임워크를 제안하여 기존의 카메라 기반 시각 fiducial marker(VFM)보다 더 나은 활용 사례를 제공합니다. 특히, 3D 포인트 클라우드 등록 및 3D 맵 통합, 증강 현실(AR)과 같은 다양한 실제 응용 프로그램에 응용될 수 있는 방법을 개발하였습니다. 이 시스템은 LiDAR에서 직접 3D fiducials를 탐지할 수 있는 인텐시티 이미지 기반의 마커 감지 시스템을 도입하였습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 요소로 구성됩니다. 첫째, 인텐시티 이미지 기반의 LiDAR fiducial marker인 IFM 시스템을 개발하여 3D 개체의 포즈 추정을 가능하게 합니다. 둘째, 개선된 알고리즘을 통해 3D 맵에서 마커를 탐지하고 최대 사후 확률(MAP) 문제를 해결하여 포인트 클라우드 및 마커 포즈를 최적화합니다. 마지막으로, 새로운 LFM 기반의 맵핑 및 로컬라이제이션 방법을 제안하여 비정렬 및 저중첩 포인트 클라우드를 등록하고 처리합니다.

- **Performance Highlights**: 다양한 LiDAR 모델을 사용한 실험 결과, 제안된 프레임워크가 기존 방법보다 더 효과적이라는 것을 입증하였습니다. 이 시스템은 AR, 3D 자산 수집, GPS가 없는 로컬라이제이션 및 3D 맵 통합과 같은 로봇 및 컴퓨터 비전 애플리케이션에 대해 효율적이고 저비용, 신뢰할 수 있는 도구로 작용합니다. 추가적으로, Livox-3DMatch 데이터셋도 도입되어 다중 시점 포인트 클라우드 등록 방법의 성능 향상을 이루었습니다.



### Learning Real-World Action-Video Dynamics with Heterogeneous Masked Autoregression (https://arxiv.org/abs/2502.04296)
Comments:
          Website: this https URL

- **What's New**: 본 논문에서는 Heterogeneous Masked Autoregression (HMA)라는 새로운 방법론을 제안합니다. 이 방법론은 동작-비디오 동역학(action-video dynamics)을 모델링하여 고품질 데이터와 로봇 학습의 확대를 위한 평가를 생성합니다. HMA는 서로 다른 로봇 구현체와 도메인, 작업에서의 관찰 및 동작 시퀀스에 대한 이질적 전처리를 활용하여, 실시간 대처가 가능한 효율성을 유지하면서 다양한 환경을 처리할 수 있습니다.

- **Technical Details**: HMA는 마스크 오토리그레션(masked autoregression)을 활용하여 비디오 예측을 위한 양자화된(discrete) 또는 부드러운(soft) 토큰을 생성합니다. 이 접근 방식은 상태-오브젝트(state-action) 모델링을 통해 비디오와 동작 시퀀스를 동시에 생성할 수 있도록 하며, 특히 다양한 로봇 환경에서 동작의 이질성을 처리하는 데 중점을 둡니다. HMA는 3백만 개 이상의 동작 라벨이 포함된 비디오로 사전 훈련되었으며, 40개의 다양한 로봇 구현체에서 동영상을 생성할 수 있습니다.

- **Performance Highlights**: HMA는 이전 로봇 비디오 생성 모델들보다 시각적 충실도와 제어 가능성에서 우수한 성능을 보이며, 15배 더 빠른 실시간 추론 속도를 기록했습니다. 이 모델은 실제 로봇 애플리케이션에서 안정적으로 100프레임 이상의 궤적을 생성할 수 있어 고충실도의 로봇 시뮬레이터 역할을 수행합니다. 또한, 이 모델은 정책 평가 및 합성 데이터 생성에 사용되어 정책 성과를 개선하는 데 기여할 수 있습니다.



### Safeguarding connected autonomous vehicle communication: Protocols, intra- and inter-vehicular attacks and defenses (https://arxiv.org/abs/2502.04201)
- **What's New**: 이 논문은 자율주행차(Connected Autonomous Vehicles, CAV)의 통신 보안에 대한 최신 연구를 체계적으로 분석합니다. CAV의 보안을 강화하기 위한 새로운 공격 분류 시스템과 최적의 보안 프로토콜 제안을 포함하고 있으며, 실제 CAV 응용 프로그램에서 이러한 프로토콜이 어떻게 통합될 수 있는지를 보여주는 사용 사례도 소개합니다. 또한, 기존의 보안 프레임워크와 프로토콜의 효율성을 평가하고, CAV 생태계 내 공격 벡터의 포괄적인 분류법을 제공합니다.

- **Technical Details**: CAV 통신은 차량 간(Vehicle-to-Vehicle, V2V) 및 차량 내부(Intra-vehicle) 통신을 포함하여 다양한 보안 위협에 직면해 있습니다. 공격 벡터는 해커가 접근하는 방법을 의미하며, 가장 일반적인 벡터로는 서버, 키없는 출입 시스템, 모바일 앱, OBD 포트 등이 있습니다. 현재 CAV의 보안 요구사항은 전통적인 보안 솔루션으로는 처리할 수 없는 독특한 특성을 가지고 있으며, 이에 따라 CAV 전용의 새로운 보안 메커니즘이 필요합니다.

- **Performance Highlights**: 논문은 CAV의 통신 보안을 위한 다양한 기술적 방안을 제안하며, 이와 관련하여 최근 연구 동향을 체계적으로 정리합니다. 특히, 기계 학습 기반의 공격 탐지 솔루션과 같은 최신 기술과 암호화 프로토콜 같은 전통적인 접근법 모두를 다루어 다양한 관점을 제공합니다. 이를 통해 CAV 보안의 발전을 돕고, 안전한 자율주행차의 도입을 추진하는 데 있어 필수적인 통찰을 제공합니다.



### Expanding Training Data for Endoscopic Phenotyping of Eosinophilic Esophagitis (https://arxiv.org/abs/2502.04199)
- **What's New**: 이 논문에서는 Eosinophilic Esophagitis (EoE)의 심층 학습 기반 형태 분류 성능을 향상시키기 위해 온라인 플랫폼, 공개 데이터셋 및 전자 교과서에서 수집한 다양한 이미지를 사용하여 훈련 데이터를 증강했습니다. 기존 데이터셋의 이미지 수가 435에서 7050으로 증가했습니다. 이로 인해 AI 모델이 적은 데이터로도 효과적으로 학습하고 EoE의 다양한 표현을 구분할 수 있습니다.

- **Technical Details**: 본 연구에서는 Data-efficient Image Transformer(DeiT)를 활용하여 이미지 분류를 수행하고, 그래디언트 어텐션 롤아웃을 통해 모델의 결정 과정에서 입력의 어떤 부분이 가장 중요한지 시각화했습니다. EREFS(System) 시스템을 사용하여 EoE 환자의 내시경 결과를 문서화하고, 5가지 주요 특징인 Edema, Rings, Exudates, Furrows, Strictures로 분류했습니다. 실험에는 학습, 검증 및 테스트 데이터셋을 사용하여 채택된 하이퍼파라미터와 최적의 성능을 위해 augmentation 기술을 적용했습니다.

- **Performance Highlights**: 결과적으로, 확장된 데이터셋과 모델 개선을 통해 EoE의 진단 정확도와 강건성이 향상되었습니다. 이 연구는 AI 기반 기술이 환자의 진단 및 치료 결과에 긍정적인 영향을 미칠 수 있음을 보여줍니다. 또한, 다양한 상부 위장관 질환의 표현을 학습하여 모델의 일반화 능력을 강화하여 임상 환경에서 다양한 환자 집단에 잘 적응할 수 있도록 하였습니다.



### Generative Adversarial Networks Bridging Art and Machine Intelligenc (https://arxiv.org/abs/2502.04116)
- **What's New**: 이 논문은 Generative Adversarial Networks (GANs)의 기본 원리와 역사적 발전을 상세히 소개하며, 전통적인 생성 모델과 대조합니다. GAN의 기본 개념을 Python 예제를 통해 설명하고, 확률 이론, 통계 및 게임 이론을 포함한 수학적 기초를 다룹니다. 이후 Conditional GANs, DCGANs, InfoGAN, LAPGAN 등 여러 클래식 변형과 Wasserstein GANs, Gradient Penalty가 있는 GAN, Least Squares GAN 및 Spectral Normalization 기법과 같은 고급 훈련 방법론에 대해 설명합니다.

- **Technical Details**: GAN은 두 개의 신경망, 즉 Generator와 Discriminator로 구성되어 서로 대립하는 구조입니다. Generator는 실제 데이터를 모방한 가짜 데이터를 생성하려 하고, Discriminator는 주어진 데이터가 진짜인지 가짜인지를 판단합니다. 두 네트워크는 동시에 훈련되어 Generator가 Discriminator를 속일 수 있을 정도로 발전하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문에서는 GAN의 효율성을 강조하며, GAN이 생성하는 데이터의 품질이 Variational Autoencoders (VAEs)보다 우수하다는 점을 밝혔습니다. GAN은 다양한 데이터 타입을 생성하는 데 유연성을 제공하며, Discriminator의 피드백 루프가 Generator의 성능 향상에 기여합니다. GAN의 역사적 발전 과정을 살펴보면, 이미지 생성의 품질과 다양성에서 꽤나 중요한 이정표를 보여줍니다.



### DEALing with Image Reconstruction: Deep Attentive Least Squares (https://arxiv.org/abs/2502.04079)
- **What's New**: 본 논문은 데이터 기반 이미지 재구성을 위한 새로운 방법을 제안합니다. 기존의 복잡한 깊은 네트워크 아키텍처 대신, 이는 고전적인 Tikhonov 정규화를 기반으로 한 접근법입니다. 이 방법은 중간 재구성을 반복적으로 개선하며, 두 가지 주요 구성 요소(learned filters 및 attention mechanism)를 포함하고 있습니다.

- **Technical Details**: 이 방법은 Quadratic 문제를 해결하여 이미지의 특징을 추출하는 learned filters와 필터 응답의 패널티를 조정하는 attention mechanism을 사용합니다. 특히, 필터의 응답에 대한 손실을 지역적으로 조정함으로써 성능을 최적화합니다. 따라서, 이 연구는 전통적인 정규화 기법과 딥러닝을 연결하고 있습니다.

- **Performance Highlights**: 제안된 방법은 최신 plug-and-play 및 learned regularizer 접근법과 동일한 수준의 성능을 달성합니다. 그러나, 이 방법은 해석 가능성과 강인함, 그리고 수렴성을 제공하여 기존 방법의 한계를 극복합니다. 이로써, 재구성 문제에 대한 보다 원칙적이고 해석 가능한 접근을 제공하고 있습니다.



### A Self-supervised Multimodal Deep Learning Approach to Differentiate Post-radiotherapy Progression from Pseudoprogression in Glioblastoma (https://arxiv.org/abs/2502.03999)
- **What's New**: 본 연구는 방사선 치료(RT) 이후의 가짜 진행(pseudoprogression, PsP)과 실제 진행(true progression, TP)을 구별하기 위한 다중 모달 딥러닝 접근법을 제안합니다. 이 방법은 정형적인 해부학적 MR 영상, 임상 매개변수 및 RT 치료 계획 정보를 통합하여 예측 정확성을 향상시키는 것을 목표로 합니다. 또한, 자가 지도(self-supervised) 비전 트랜스포머(Vision Transformer, ViT)를 사용하여 다중 시퀀스 MR 뇌 볼륨을 인코딩합니다.

- **Technical Details**: 연구에서는 BraTS2021, UPenn-GBM 및 UCSF-PDGM와 같은 공개 글리오마 MRI 데이터셋을 활용하여 비레이블 데이터에서 자가 지도 학습을 진행합니다. 이로 인해 FLAIR 및 T1 대조 후 시퀀스에서 임상적으로 중요한 표현을 생성하고, 임상 데이터 및 RT 치료 계획 정보를 통해 교차 모달(attention) 방식을 적용하여 분류 정확성을 높입니다. 제안된 방법의 성능은 두 개의 서로 다른 센터에서 수집된 데이터셋을 사용하여 검증되었습니다.

- **Performance Highlights**: 제안된 방법은 75.3%의 AUC를 달성하였으며, 기존 데이터 기반 접근법을 초월하는 성능을 보였습니다. 또한, 다수의 임상적으로 사용 가능한 해부학적 MRI 시퀀스와 임상 데이터, RT 치료 계획 정보를 기반으로 하여 실제에서의 실행 가능성을 높였습니다. 이 연구는 PsP와 TP 구분에 대한 데이터 가용성 문제를 해결하고, GBM 환자에 대한 치료 결정 및 최적화된 치료 계획 수립에 기여할 것으로 기대됩니다.



### UniForm: A Unified Diffusion Transformer for Audio-Video Generation (https://arxiv.org/abs/2502.03897)
- **What's New**: 이번 논문에서는 공통의 잠재 공간에서 오디오와 비디오를 동시에 생성하는 UniForm이라는 통합 확산 변환기(Unified Diffusion Transformer)를 소개합니다. 기존의 오디오-비디오 생성 방식은 두 개의 독립적인 모듈에 의존하여 자연스러운 상관 관계를 충분히 활용하지 못했습니다. UniForm은 시각 및 청각 정보를 결합하여 고품질의 오디오-비디오 쌍을 생성하도록 학습합니다.

- **Technical Details**: UniForm은 통합된 잠재 공간에서 오디오와 비디오의 동시 생성을 가능하게 하는 단일 확산 프레임워크를 채택하고 있습니다. 이를 통해 모델은 세 가지 생성 작업: 텍스트-오디블 비디오 생성(T2AV), 오디오-비디오 생성(A2V), 및 비디오-오디오 생성(V2A)을 지원합니다. 각 작업에 대해 텍스트 프롬프트를 추가하여 세밀한 제어 기능을 더해 발휘할 수 있는 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, UniForm은 최신 단일 작업 기준들과 비교하여 동등한 성능을 달성했습니다. 특히, задачи는 특정 작업 데이터셋에 대한 미세 조정 없이 이루어졌으며, 다중 작업 시스템으로 훈련된 성능이 눈에 띕니다. 이 실험들은 오디오와 비디오 간의 정합성과 일관성을 획기적으로 향상시키는 UniForm의 강점을 증명합니다.



### Synthetic Poisoning Attacks: The Impact of Poisoned MRI Image on U-Net Brain Tumor Segmentation (https://arxiv.org/abs/2502.03825)
- **What's New**: 이 연구에서는 U-Net 모델을 사용하여 뇌 종양 분할 작업에서 합성 MRI 데이터의 영향을 평가합니다. GAN 기반 모델을 통해 생성된 T1 대비 강조 MRI 이미지를 여러 비율로 훈련 데이터에 통합하여 성능 변화에 대한 실험을 실시했습니다. 이 연구는 합성 데이터가 일으킬 수 있는 데이터 오염의 위험성을 강조하며, 의료 이미지 분석에서 신뢰할 수 있는 시스템 개발에 기여합니다.

- **Technical Details**: 합성 데이터를 데이터 오염의 형태로 간주하고, U-Net 모델이 훈련 과정에서 합성 데이터 비율이 증가할 때 분할 성능에서 저하가 발생하는지를 조사합니다. Dice 계수, Jaccard 지수, 정확도 및 민감도를 사용하여 성능 저하의 정도를 정량화합니다. Gaussian 기반 모델을 사용하여 CT-MRI 쌍 데이터셋에서 T1-대비 강조 MRI 이미지를 생성하고, 합성 데이터 비율을 16.67%에서 83.33%까지 변화시켜 훈련합니다.

- **Performance Highlights**: 실험 결과, 합성 데이터 비율이 증가함에 따라 모델의 성능이 유의미하게 저하됨을 보였습니다. 예를 들어, Dice 계수는 0.8937에서 0.7474로 감소하며, 정확도와 민감도 또한 비슷한 경향을 보였습니다. 이러한 결과들은 합성 데이터 통합 시 품질 관리를 엄격히 해야 함을 강조합니다.



### UltraBones100k: An Ultrasound Image Dataset with CT-Derived Labels for Lower Extremity Long Bone Surface Segmentation (https://arxiv.org/abs/2502.03783)
Comments:
          13 pages, 4 figures

- **What's New**: 이 연구는 초음파 영상에서 뼈 표면을 세분화하는 새로운 방법론을 제안합니다. 기존의 데이터셋이 전문가의 수작업 레이블링에 의존하고 있어 한계가 있었던 반면, 저자들은 자동으로 생성된 뼈 레이블을 포함한 고품질의 초음파 이미지 데이터셋(주요 예: UltraBones100k)을 구축했습니다. 이는 저비용의 데이터 수집 방법을 제공하여 더 큰 데이터셋을 가능하게 하였습니다.

- **Technical Details**: 제안된 방법론은 추적된 CT 뼈 모델과 초음파 영상을 정밀하게 겹쳐서 이를 바탕으로 자동으로 레이블을 생성하는 방식입니다. 이 연구에서는 14명의 인체 시체의 하체에서 10만 장의 초음파 이미지를 수집하여 뼈 레이블을 부여하였으며, 정밀한 레이블링을 통해 저강도 뼈 영역의 식별 문제를 해결하려고 하였습니다. 이를 통해 높은 정확도와 완전성을 가진 분할 모델을 훈련할 수 있었습니다.

- **Performance Highlights**: UltraBones100k로 훈련된 모델은 모든 측정 지표에서 전문가의 수작업 레이블링을 초과하는 성능을 보여주었습니다. 특히 낮은 강도 영역에서 0.5 mm 거리 임계값에서 320%의 완전성 향상을 기록하여, 저자들은 이 모델이 초음파 데이터 분할의 새로운 기준을 제시한다고 주장합니다. 또한, Wilcoxon signed-rank 테스트 결과, 제안된 방법이 뼈 레이블링의 품질을显著 증가시켰음을 입증하였습니다.



### Variational Control for Guidance in Diffusion Models (https://arxiv.org/abs/2502.03686)
Comments:
          8 pages in main text. Total of 20 pages

- **What's New**: 이 연구에서 제안하는 Diffusion Trajectory Matching (DTM)은 기존의 방법들의 한계를 극복하기 위해 Variational Inference (변분 추론) 및 제어 관점에서 Diffusion 모델을 재조명합니다. 기존의 Classifier Guidance (분류기 안내) 또는 Classifier-Free Guidance (비분류기 안내) 기법들은 추가 모델 학습을 요구하거나 표준화된 가정에 기반하여 샘플 품질을 해치는 경우가 많았습니다. DTM은 이러한 기법들을 통합하여 추가 학습 없이도 고성능을 발휘하도록 설계되었습니다.

- **Technical Details**: DTM 프레임워크는 Guided Diffusion Dynamics (유도 확산 동역학)를 마르코프 체인으로 모델링 하여 통제 신호를 Variational Parameters (변분 매개변수)로 정의합니다. 이로써 생성된 샘플들이 비조건부 샘플 매니폴드에 가깝게 유지되도록 최적화를 적용하며, 원하는 단말 조건을 만족하는 과정을 보장합니다. DTM은 또한 Non-linear Diffusion Trajectory Matching (NDTM)으로 구체화되며, 기존의 최첨단 Diffusion 모델 샘플러와 잘 통합됩니다.

- **Performance Highlights**: NDTM은 ImageNet-256 및 FFHQ-256과 같은 데이터셋에서 슈퍼 해상도 및 인페인팅과 같은 도전적인 문제들에 대해 이전의 최신 기법들을 초월하는 성능을 보여주었습니다. 예를 들어, 이미지 노이즈 제거 문제에서 DTM을 통해 FID 점수 34.31을 달성하며, 기존의 최적 사전 훈련 방법의 FID 78.07를 크게 개선했습니다. 연구팀은 향후 코드도 공개할 계획이라고 밝혔습니다.



### Advancing Weight and Channel Sparsification with Enhanced Saliency (https://arxiv.org/abs/2502.03658)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문은 IEE(Iterative Exploitation and Exploration)라는 혁신적인 접근 방식을 소개하여 비구조적 및 구조적 희소성(sparsity)에 대해 중요도 기준을 향상시키는 방법을 제안합니다. 이 방법은 모델을 활성 구조와 탐험 공간으로 나누어 활성 구조를 최적화하고 탐험 공간에서 매개변수를 다시 평가하여 통일된 중요도 기준에 따라 재통합합니다. 실험을 통해 기존의 단순한 중요도 기준을 개선하여 최첨단 성능과 훈련 비용 절감을 달성할 수 있음을 보여줍니다.

- **Technical Details**: IEE 모델은 두 가지 단계로 구성됩니다: 활용(Exploitation)과 탐험(Exploration) 단계입니다. 활용 단계에서는 현재의 활성 구조를 최적이라고 가정하고 이를 수렴하기 위해 훈련합니다. 탐험 단계에서는 이 최적성을 의심하며 중요도 점수를 바탕으로 덜 중요한 매개변수를 제거하고, 탐험 공간의 모든 매개변수를 잠시 "재활성화"하여 몇 번의 반복 훈련을 실시하여 이 매개변수의 성능 잠재력을 미리 살펴보는 과정이 포함됩니다.

- **Performance Highlights**: ResNet50 모델을 사용한 ImageNet 데이터셋의 실험 결과, IEE 방법은 90% ERK 희소성에서 기존 기법보다 Top-1 정확도를 1.3% 향상시키는 성과를 보였습니다. 특히 HALP와 비교하여 훈련 비용을 70% 이상 절감하고 더 빠르고 정확한 프루닝된 모델을 얻었습니다. 이는 단순히 기존의 Magnitude 및 Taylor 기준을 개선하여 최첨단 결과를 달성할 수 있음을 보여줍니다.



### Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics (https://arxiv.org/abs/2502.03654)
Comments:
          8 pages, excluding references and appendix

- **What's New**: 이번 논문에서는 Gompertz Linear Unit (GoLU)라는 새로운 self-gated activation function을 도입합니다. GoLU는 Gompertz 함수의 비대칭성을 활용하여 기존의 활성화 함수들보다 효과적으로 잠재 공간의 분산을 줄이는 동시에 강력한 gradient flow를 유지합니다. 여러 과제를 통한 실험 결과, GoLU는 최신 활성화 함수에 비해 우수한 성능을 보여주며, 이는 현재의 활성화 함수들에 대한 견고한 대안으로 자리잡고 있습니다.

- **Technical Details**: GoLU는 Gompertz 함수를 게이팅 메커니즘으로 사용하는 self-gated 활성화 함수입니다. 이 함수는 exponentials를 사용하여 무한히 미분 가능하며, ReLU와 그 변형들과는 달리 매끄럽고 비 모노토닉한 특성을 지닙니다. Gompertz 함수의 비대칭성은 Gumbel 분포의 근본적인 비대칭성에서 기인하여 출력의 세기가 다른 gated activation function들에 비해 압축된 효과를 내게 합니다.

- **Performance Highlights**: 다양한 과제를 대상으로 한 실험 결과, GoLU는 기존 self-gated 활성화 함수들보다 더 효과적으로 잠재 표현의 분산을 줄입니다. 이는 모델의 활성화 출력에서 노이즈를 줄여주어, 필수적인 특징을 보존하면서도 과적합(overfitting)을 방지하는 데 도움이 됩니다. GoLU는 각기 다른 데이터 세트에서 탁월한 성능을 보여주며, 이는 고차원 데이터 처리에 대한 향상된 능력을 나타냅니다.



### MetaFE-DE: Learning Meta Feature Embedding for Depth Estimation from Monocular Endoscopic Images (https://arxiv.org/abs/2502.03493)
- **What's New**: 이 논문에서는 'meta feature embedding (MetaFE)'라는 새로운 개념을 도입하여 단일(endoscopic) 이미지에서 깊이 추정을 수행하는 혁신적인 접근 방식을 제안하고 있습니다. 기존의 연구에서는 RGB 이미지에서 직접 깊이를 추정하는 방법이 주를 이루었지만, 이 방법은 해석 가능성과 정확성이 부족한 경우가 많았습니다. MetaFE를 사용하여 RGB 이미지와 깊이 이미지 간의 내재적 상관관계를 탐구하고, 두 단계의 자기 지도 학습(self-supervised learning) 방식을 통해 깊이 이미지를 향상시키는 방법을 제안합니다.

- **Technical Details**: MetaFE-DE는 두 단계로 구성되어 있으며, 첫 번째 단계에서는 diffusion 모델을 사용하여 시공간 정보의 정렬을 통해 MetaFE를 생성하는 'temporal representation learner'를 제안합니다. 두 번째 단계에서는 밝기 보정을 활용하여 자기 지도 단일(depth) 추정을 수행하여 MetaFE를 깊이 이미지로 디코드하는 방법이 포함됩니다. 이 과정에서 원시 픽셀에서 학습된 특징이 정확한 깊이 이미지로 직결될 수 있다는 기존 연구와는 다른 접근 방식을 취하고 있습니다.

- **Performance Highlights**: 다양한 endoscopic 데이터 세트에서 광범위한 평가를 수행한 결과, 제안된 접근 방식이 깊이 추정에서 최고 수준의 성능을 달성했음이 입증되었습니다. 특히, 이 연구에서는 서로 다른 시각적 작업(예: RGB 또는 깊이 이미지로 디코딩된 작업) 간에 공통적인 특징들이 존재하며, 이를 통해 정확한 깊이 추정이 가능하다는 점을 강조하고 있습니다. 공개적인 소스 코드 제공을 통해 후속 연구의 기초를 마련할 계획입니다.



### Can Domain Experts Rely on AI Appropriately? A Case Study on AI-Assisted Prostate Cancer MRI Diagnosis (https://arxiv.org/abs/2502.03482)
- **What's New**: 이번 연구는 방사선 전문의와의 심도 있는 협력을 통해 MRI 이미지를 기반으로 한 전립선암 진단에서 AI 지원 도구의 실제 통합 효과를 조사합니다. 두 가지 실험을 진행하여 AI 지원과 성과 피드백이 도메인 전문가의 의사결정에 미치는 영향을 분석하였습니다. 특히, 두 개의 상이한 작업 흐름을 설계하여 실제 임상 환경에서 AI 툴이 어떻게 사용될 수 있는지를 모델링하였습니다.

- **Technical Details**: 연구는 8명의 방사선 전문의(N=8)를 포함한 사전 등록된 인간 실험을 수행하였으며, 주요 초점은 전립선암 진단을 위한 AI 지원입니다. 최신 AI 모델(Isensee et al., 2021)을 훈련시켜 전립선암 탐지에 필요한 진단 예측과 양성 사례에 대한 병변 주석 맵을 제공하였습니다. 실험은 두 가지 별도의 작업 흐름으로 구성되었고, 첫 번째 연구에서는 독립적인 진단 후 AI 예측을 확인하는 방식으로 진행되었습니다.

- **Performance Highlights**: 연구 결과, 인간-AI 팀은 인간 단독보다 지속적으로 더 높은 성능을 보였으나, AI 단독 성능에는 미치지 못하는 경향이 있었습니다. 성과 피드백을 제공했음에도 불구하고, 인간-AI 팀의 성능 향상은 제한적이었으며, AI 결정을 사전에 보여주는 방식이 방사선 전문의로 하여금 AI를 더욱 신뢰하도록 유도하는 것으로 나타났습니다. 흥미로운 점은 인간-AI 팀의 다수결 결정이 AI 단독보다 뛰어난Complementary performance를 달성했다는 점으로, 이는 인간-AI 협업의 유망한 방향성을 제시합니다.



New uploads on arXiv(cs.AI)

### Strong Equivalence in Answer Set Programming with Constraints (https://arxiv.org/abs/2502.04302)
Comments:
          30 pages

- **What's New**: 이 논문에서는 제약이 포함된 확장된 응답 집합 프로그래밍(Answer Set Programming; ASP)의 강한 동등성(strong equivalence) 개념을 조사합니다. 특히 이 연구는 서로 다른 제약 유형의 복합적 상호 작용을 고려할 때, 이 동등성을 어떻게 정의할 수 있는지를 다룹니다. 한편, 제약을 다루는 여러 clingo 기반 솔버의 언어에서 Here-and-There 논리로의 번역도 제시합니다.

- **Technical Details**: 여기서 제안된 강한 동등성은 다양한 제약이 결합된 컨텍스트에서 두 규칙 세트가 같은 의미를 가지도록 보장하는 강력한 동등성 개념을 따른다. 이 연구는 제약 응답 집합 프로그래밍에서 사용하는 Here-and-There 논리와 제약의 특별한 형태인 HTc를 통해 강한 동등성을 분석합니다. 그 과정에서, 이 모델은 제약 만족 문제(constraint satisfaction problems; CSPs)에 대한 논리적 기초를 제공합니다.

- **Performance Highlights**: 본 논문은 강한 동등성이 제약 응답 집합 프로그래밍에서 어떻게 이루어지는지를 제시하는데, 이는 논리 프로그램의 최적화 과정에서 중요한 기여를 합니다. 예를 들어, 간단한 ASP 프로그램에서의 규칙 조합을 통하여 각 규칙의 중복성을 분석합니다. 또한, 강한 동등성을 결정하는 계산 복잡성에 대한 논의도 제시하여 향후 연구에 중요한 기반을 마련합니다.



### Free Energy Risk Metrics for Systemically Safe AI: Gatekeeping Multi-Agent Study (https://arxiv.org/abs/2502.04249)
Comments:
          9 pages, 1 figure

- **What's New**: 이번 연구에서는 에이전트 기반 및 다중 에이전트 시스템에서 위험을 측정하기 위한 기초로 Free Energy Principle을 조사합니다. 이를 바탕으로 다양한 맥락과 필요에 유연하게 대응할 수 있는 Cumulative Risk Exposure 메트릭을 도입합니다. 기존의 데이터 의존적 이론들과는 대조적으로, 이해관계자들은 시스템 결과에 대한 자신의 선호만 지정하면 되며, 이는 위험 관리와 완화에 대한 간단하고 투명한 의사결정 규칙을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 세계 모델(world model)과 선호 모델(preference model)의 불확실성을 자연스럽게 고려할 수 있으며, 이는 인식적(epistemic) 및 가치적(axiological) 겸손, 간결성(parsimonious), 미래 안전성을 보장합니다. 연구진은 자율주행 차량 환경에서 다중 에이전트에 의해 운영되는 차량 정책이 게이트키퍼(gatekeeper)에 의해 중재되는 상황을 시뮬레이션하여 이 새로운 접근 방식을 증명합니다. 이 구조에서는 각 차량이 지역 안전성을 평가하고, 필요 시 차량 정책에 개입합니다.

- **Performance Highlights**: 연구 결과, 자율주행 차량 집단에 게이트키퍼를 도입했을 때, 낮은 침투율에서도 시스템 안전성을 크게 향상시키는 긍정적인 외부 효과(externality)를 생성할 수 있음을 보여줍니다. 이러한 접근 방식은 다양한 상황에서 위험을 효과적으로 관리할 수 있는 가능성을 제시하며, 안전한 AI 시스템의 설계에 기여할 것으로 기대됩니다.



### Strategic Learning with Local Explanations as Feedback (https://arxiv.org/abs/2502.04058)
- **What's New**: 이 논문에서는 의사결정자가 제시하는 모델에 전략적으로 반응할 수 있는 에이전트들이 얽힌 알고리즘 결정 문제를 탐구합니다. 특히, 완전한 모델 공개가 항상 가능하거나 바람직하지 않을 때, 의사결정자가 어떻게 해석 가능한 설명을 통해 유틸리티를 극대화하면서 에이전트의 복지를 해치지 않는지에 대한 질문을 다룹니다. 이를 위해 기존의 로컬 및 글로벌 설명 방법들을 검토하고, 에이전트를 잘못된 행동으로 유도하지 않기 위한 필요 조건을 설정합니다.

- **Technical Details**: 본 연구는 자동차 보험 가격 책정 시나리오를 예시로 하여, 의사결정자는 고객의 프로필을 기반으로 미래의 사고 비용을 예측하고 이를 통해 보험료를 결정합니다. 에이전트는 자신의 프로필을 전략적으로 조정하여 보험료를 변화시키려 하며, 이에 따라 의사결정자는 로컬 설명을 제공하여 투명성을 확보하려고 합니다. 최적화 문제 해결을 위해 예측 모델과 행동 추천 기반 설명 정책을 공동으로 최적화하는 간단한 알고리즘을 제안합니다.

- **Performance Highlights**: 우리의 실증 결과는 제안된 접근 방식이 알고리즘적 의사결정에서 안전하고 효과적인 부분적 모델 공개를 위한 세련된 전략임을 입증합니다. 특히, 행동 추천 (AR)-기반 설명이 에이전트의 반응을 적절하게 유도하고 결과적으로 유틸리티를 저해하지 않도록 하는데 효과적임을 보여줍니다. 이러한 결과는 설명의 구조적 설계와 관련하여 의사결정자와 에이전트 커뮤니케이션의 효율성을 극대화하는 데 도움을 줄 수 있습니다.



### Fine, I'll Merge It Myself: A Multi-Fidelity Framework for Automated Model Merging (https://arxiv.org/abs/2502.04030)
- **What's New**: 본 논문에서는 모델 병합(Model Merging)을 통해 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력을 향상시키는 자동화된 프레임워크를 제안합니다. 이 프레임워크는 수많은 모델 조합을 탐색할 수 있는 방식으로, 기존의 수동 접근 방식보다 비용을 줄이고 효율성을 높여줍니다. 특히, 우리는 세분화된 병합 전략을 가능하게 하는 레이어별 융합(Layer-wise Fusion, LFS)과 깊이별 통합(Depth-wise Integration, DIS)이라는 두 가지 새로운 검색 공간을 소개하고 있습니다.

- **Technical Details**: 논문에서 제안하는 자동화된 모델 병합 프레임워크는 Multi-Fidelity Optimization (MFO) 기법을 활용하여 병합 과정을 최적화합니다. 이 프레임워크는 제안된 두 가지 검색 공간에서 단일 및 다목적 최적화를 지원하며, 필요한 계산 비용을 줄이며 세부적인 탐색을 가능하게 합니다. 구체적으로, 레이어별 융합은 여러 모델의 이를 병합하며 깊이별 통합은 최적의 계층 관계를 발견하는 데 초점을 맞추고 있습니다. 이러한 방식을 통해 계산 비용을 절감하면서도 모델의 성능을 개선할 수 있습니다.

- **Performance Highlights**: 우리는 여러 벤치마크를 통해 제안된 프레임워크의 효율성을 평가했으며, 결과적으로 다목적 시나리오에서 평균 6.86% 개선을, 도전적인 GSM8K 작업에서는 4.24% 개선을 달성했습니다. 이와 함께, 다양한 추론 벤치마크에서 일관된 효과성을 보였습니다. 우리의 방식은 제한된 계산 자원 내에서 실행되며, 예를 들어 500회의 검색 단계 이내에서 효율적인 병합이 가능합니다.



### Enhancing Online Learning Efficiency Through Heterogeneous Resource Integration with a Multi-Agent RAG System (https://arxiv.org/abs/2502.03948)
- **What's New**: 이번 포스터 논문은 학습 효율성을 높이기 위한 Multi-Agent Retrieval-Augmented Generation (RAG) 시스템을 도입합니다. 이 시스템은 YouTube 튜토리얼, GitHub 저장소, 문서 웹사이트와 같은 다양한 온라인 리소스를 통합하여 정보 검색 및 합성 과정을 자동화합니다. 초기 사용자 연구에서 시스템의 사용성과 유용성이 높게 평가되어, 지식 습득의 효율성을 크게 향상시킬 가능성을 보여주었습니다.

- **Technical Details**: Multi-Agent RAG 시스템은 GPT-4o를 기반으로 하며, 중앙의 Manager Agent가 YouTube Video, GitHub Repository, Documentation Website 및 Generic Search Engine과 같은 네 가지 특화된 에이전트 간의 작업을 조정합니다. 각 에이전트는 독립적으로 운영되며, 관련 정보 검색을 위해 API와 도구를 활용합니다. 시스템의 모듈형 디자인은 확장성과 적응성을 보장하여, 필요에 맞춰 추가 에이전트나 도구를 쉽게 통합할 수 있습니다.

- **Performance Highlights**: 예비 평가 결과, 사용자 설문조사에 참여한 15명의 대학원생들은 Perceived Usefulness (PU) 점수가 평균 75로 중간 높은 유용성을 나타냈지만, 각기 다른 사용자들의 요구에 따라 유용성 인식에 차이가 있음을 보여주었습니다. 반면, Perceived Ease of Use (PEU) 점수는 평균 91.11로 강력한 사용성을 나타내어 대부분 사용자들이 시스템의 유용성을 높이 평가했습니다. 이처럼, 시스템의 높은 사용성과 더불어 유용성을 개선하기 위한 특정적 노력의 필요성이 제기되었습니다.



### Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 (https://arxiv.org/abs/2502.03544)
Comments:
          28 pages, 16 figures

- **What's New**: AlphaGeometry2(AG2)는 2024년 Trinh 외 연구진에 의해 소개된 AlphaGeometry(AG1)의 크게 향상된 버전입니다. AG2는 자동 기하학 문제 해결에서 평균 금메달리스트를 초월하는 성능을 발휘하였고, International Math Olympiad(IMO) 2000-2024 기하학 문제에 대한 해결률을 66%에서 88%로 끌어올렸습니다. 새로운 Gemini 아키텍처를 활용하여 언어 모델링을 향상시키고, 여러 검색 트리를 결합한 지식 공유 메커니즘을 도입했습니다.

- **Technical Details**: 이 논문에서 AG2는 더 강력하고 다양한 데이터셋으로 훈련된 Gemini 기반의 언어 모델을 활용합니다. 빠르고 강력한 기호 엔진과 축소된 규칙 집합을 활용한 최적화 등 여러 방식으로 성능을 개선하였으며, locus 정리와 같은 기하학적 개념과 선형 방정식 등 폭넓은 영역과의 통합도 이루어졌습니다. 혁신적인 검색 알고리즘을 통해 보조 구성 전략을 탐색하고, 검색 프로세스를 가속화하기 위해 지식 공유 메커니즘을 활용하고 있습니다.

- **Performance Highlights**: AG2는 2000-2024 IMO 기하학 문제에 대해 84%라는 인상적인 해결률을 달성하여, AI의 수학적 추론 능력이 크게 발전했음을 시사합니다. AG1에 비해 해결률이 54%에서 84%로 증가하며, 또래 금메달리스트들의 성과를 넘었습니다. 마지막으로 AG2는 자연어 입력으로부터 기하학 문제를 직접 해결하는 완전 자동 시스템 개발을 향해 나아가고 있습니다.



### YINYANG-ALIGN: Benchmarking Contradictory Objectives and Proposing Multi-Objective Optimization based DPO for Text-to-Image Alignmen (https://arxiv.org/abs/2502.03512)
- **What's New**: YinYangAlign은 T2I(텍스트-이미지) 시스템의 정렬 정확성을 체계적으로 정량화하는 고급 벤치마킹 프레임워크로, 다양한 설계 목표 간의 모순을 다룬다. 이 연구는 기존 LLM(대형 언어 모델)의 성공적인 정렬 기술을 T2I 시스템에 적용하여 이미지 생성의 신뢰도와 집합성을 향상시키고자 한다. 특히, YinYangAlign은 인간의 프롬프트, 정렬된 응답 및 비정렬된 AI 생성 산출물을 포함한 자세한 공리 데이터세트를 내놓음으로써, 정렬에 대한 포괄적인 평가를 가능하게 한다.

- **Technical Details**: T2I 모델의 정렬은 사용자 의도를 충실히 반영하고 윤리적 및 미학적 기준을 준수하는 데 필수적이다. 양자 연구에서 주로 고립된 목표에 집중했던 기존의 접근법과 달리, YinYangAlign은 서로 모순되는 여섯 가지 목표를 소개하여 전반적인 조화로운 평가를 가능하게 한다. 이 프레임워크는 예술적 자유와 프롬프트에 대한 충실성을 포함한 주요 갈등을 다루며, 각 목표는 실제 응용에서의 적합성을 고려하여 선택되었다.

- **Performance Highlights**: YinYangAlign의 데이터 세트는 고도의 자동화와 인간 검증을 결합한 하이브리드 주석 프로세스를 통해 설계되었다. 전체 50,000개의 이미지를 평가한 결과, 40,000개의 고품질 이미지로 최종 포맷이 정립되었으며, 주석 프로세스에서 0.83의 카파 점수를 기록하여 높은 일관성과 신뢰성을 입증했다. 또한, 대응 목표 간의 최적화를 위해 도입된 CAO(모순 정렬 최적화) 메커니즘은 각 공리에 대한 손실 설계를 통해 경쟁 목표를 명확하게 모델링하며, 다양한 정렬 패러다임을 위한 적응적인 최적화를 가능하게 한다.



### Examining Two Hop Reasoning Through Information Content Scaling (https://arxiv.org/abs/2502.03490)
- **What's New**: 이 연구는 트랜스포머(transformer)가 두 단계 질문(latent two-hop question)에 대한 답변을 학습하는 능력의 일관성 부족을 탐구하고 있습니다. 전작들에서는 두 단계 QA 데이터셋에 대한 트랜스포머의 크기와 학습 능력이 어떻게 변화하는지를 분석하여, 두 단계 QA에 필요한 정보의 중복성을 강조하고 있습니다. 이를 통해, 적절한 데이터셋 매개변수에서 트랜스포머가 여러 레이어에서 사실 정보를 중복 저장해야 하는 점을 지적하고 있습니다.

- **Technical Details**: 연구는 'native information content'라는 개념을 도입하여, 트랜스포머의 아키텍처에 맞춰 압축 가능성을 연구합니다. 이를 통해 두 단계 질문 응답에서의 정보 용량 규모가 모델 크기와 함께 어떻게 변화하는지를 탐구하며, 기존의 압축 이론과 연관짓고 있습니다. 또한 서로 다른 질문 유형을 처리하는 데 필요한 정보 용량의 증가율에 대한 예측도 제공합니다.

- **Performance Highlights**: 연구 결과, 두 단계 질문 처리에서 트랜스포머의 기능에 있어 효과적인 정보 압축이 필수적이라는 것을 보여줍니다. 일반적인 기능 조합보다 기억(Memorization)을 통해 답변을 더 잘 수행하도록 모델을 트랩(trap)하는 방법을 제안합니다. 이러한 발견은 트랜스포머의 일반화 성능(generalization performance) 향상에 기여할 수 있음을 강조합니다.



### WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs (https://arxiv.org/abs/2502.04326)
- **What's New**: 이 논문에서는 시각, 청각, 텍스트 입력을 동시에 평가하는 첫 번째 멀티모달 비디오 이해 벤치마크인 WorldSense를 소개합니다. 기존 벤치마크와 달리, WorldSense는 오디오와 비디오의 긴밀한 연결을 강조하고 다양하고 양질의 비디오 및 다중 선택 QA 쌍을 통해 종합적인 평가를 가능하게 합니다. 이를 통해 실제 세계 맥락에서의 멀티모달 이해 능력을 평가할 수 있는 플랫폼을 제공합니다.

- **Technical Details**: WorldSense는 1662개의 오디오-비디오 동기화 비디오와 3172개의 다중 선택 질문으로 구성되어 있습니다. 이 자료들은 8개의 주요 도메인과 67개의 세분화된 하위 범주로 체계적으로 분류되어 있으며, 질문은 음성 및 비주얼 정보의 동시 처리를 요구하여 모델의 멀티모달 처리 능력을 rigorously 평가합니다. QA 쌍은 80명의 전문가에 의해 수차례 검증된 고품질 주석을 통해 보장됩니다.

- **Performance Highlights**: 전반적인 실험 결과, 기존 모델들이 실제 세계의 맥락을 이해하는 데 어려움을 겪고 있음을 보여줍니다. 오픈소스 비디오-오디오 모델은 약 25%의 정확성을 보이는 반면, Gemini 1.5 Pro와 같은 상용 모델은 48%의 정확성을 달성하지만 단일 모드에서 성능이 15% 감소합니다. 이러한 결과는 WorldSense의 모달리티 간 강한 결합을 강조하며, 실제 세계 이해 능력에서 상당한 격차가 있음을 드러냅니다.



### Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions (https://arxiv.org/abs/2502.04322)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 안전성 문제를 다루며, 기존의 연구들이 주로 기술적인 지식을 요구하는 공격 방법에 집중하고 있다는 점을 지적합니다. 연구자들은 jailbreak된 응답이 일반 사용자에게 해로운 행동을 유도하는 데 진정으로 유용한지와 간단한 상호작용에서 안전성 취약점이 존재하는지를 탐구합니다. 이를 통해 저자들은 HarmScore라는 새로운 메트릭을 제안하며, 다중 단계, 다국어 상호작용에서 해로운 행동을 유도하기 위한 새로운 프레임워크인 Speak Easy를 소개합니다.

- **Technical Details**: Speak Easy는 일반 사용자가 해로운 내용을 찾기 위해 사용할 수 있는 두 가지 유형의 인간-LM 상호작용인 다중 단계 추론(multi-step reasoning)과 다국어 질문(multilingual querying)을 모사합니다. 연구자들은 이 프레임워크를 통해 사용자가 해로운 쿼리를 여러 개의 무해한 하위 쿼리로 나누어 LLM의 안전 장치를 쉽게 우회할 수 있도록 하는 방법을 제안합니다. 이 논문은 GPT-4o, Qwen2-72B-Instruct 등 여러 안전 정렬된 LLM을 대상으로 Speak Easy의 효과를 체계적으로 평가하며, HarmScore는 인간 평가와 잘 일치함을 보여줍니다.

- **Performance Highlights**: Speak Easy는 여러 기준을 통해 GPT-4o의 공격 성공률(Attack Success Rate, ASR)을 평균 0.319 증가시키고, HarmScore를 0.426 증가시키는 결과를 도출했습니다. 또 다른 분석 연구를 통해 더 많은 분해 단계와 언어를 사용하는 것이 응답의 해로움을 증가시킨다는 것을 입증했습니다. 특이하게도, Speak Easy는 기존의 jailbreak 방법들에도 통합되어 성능 향상을 가져오는 것으로 나타났습니다.



### ChamaleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters (https://arxiv.org/abs/2502.04315)
- **What's New**: 최근 큰 발전을 이루고 있는 대규모 언어 모델(LLMs)의 연구는 다양한 작업에서 놀라운 성능을 보여주고 있습니다. 이 논문에서는 ChamaleonLLM이라는 새로운 프레임워크를 소개하며, 이는 추론(inference) 시 모델이 동적으로 적응할 수 있도록 지원합니다. 전통적인 방법들과는 달리, ChamaleonLLM은 배치(batch) 기반 클러스터링(clustering)과 저차원(low-rank) 업데이트를 실시간으로 생성하는 방식을 활용하여 성능을 극대화합니다.

- **Technical Details**: ChamaleonLLM의 핵심은 배치 통계치를 기반으로 저차원(modification) 업데이트를 동적으로 생성하는 것입니다. 입력은 의미적 및 구문적 유사성에 따라 클러스터로 그룹화되어, 동질적인 입력들로 이루어진 미니 배치(batch)가 생성됩니다. 또한, 하이퍼 네트워크(hyper-network)를 통해 모델의 디코더(decoder) 가중치를 실시간으로 적응시키는 방식을 채택하여, 추론 과정에서 입력 데이터의 세부 사항을 보다 유연하게 반영합니다.

- **Performance Highlights**: 실험 결과, ChamaleonLLM은 기존의 LoRA 방식에 비해 향상된 성능을 보여 주었으며, 동적인 데이터 상황에서도 유연하게 대처할 수 있는 잠재력이 있습니다. 이 접근법은 메모리 및 계산 요구 사항을 줄이면서도, 고성능의 언어 모델 추론을 가능하게 하여 다양한 작업에 적응할 수 있는 잠재력을 지니고 있습니다. ChamaleonLLM은 오픈 소스로 제공되어 실험의 재현성을 보장하며, 연구자들이 이 프레임워크의 이점을 쉽게 활용할 수 있도록 하고 있습니다.



### Great Models Think Alike and this Undermines AI Oversigh (https://arxiv.org/abs/2502.04313)
Comments:
          60 pages, 20 figures

- **What's New**: 본 논문은 AI Oversight의 맥락에서, 언어 모델(LM)의 유사성이 이들 평가 및 감독에서 어떻게 작용하는지를 탐구합니다. 이를 위해, 모델의 실수(overlap in model mistakes)를 기반으로 한 확률적 유사성 측정 지표인 CAPA를 제안합니다. 연구 결과, 유사한 모델들이 상호작용할 때 더 나은 평가를 수행하게 된다라는 사실을 발견하였습니다.

- **Technical Details**: CAPA는 모델의 정확도를 고려하여 유사성을 측정하기 위해 고안된 메트릭입니다. 논문의 기술적 세부사항에서는 CAPA의 수학적 유도와 다수 모델 설정에의 확장 방법이 포함되어 있습니다. 이 메트릭은 전통적인 정합성은 물론, Scatter π와 Fleiss κ 같은 다른 메트릭들과 비교하여 새로운 계산 방식을 도입하고 있습니다.

- **Performance Highlights**: 모델의 능력이 증가함에 따라, 유사한 오류를 내는 경향이 발견되었습니다. 이는 AI Oversight의 위험 요소를 부각시키며, 모델 유사성을 모니터링할 필요성을 강조합니다. 결론적으로, 모델 간의 유사성을 보도하고 수정하는 것이 AI 감독의 새로운 패러다임에서 필수적임을 언급하고 있습니다.



### HOG-Diff: Higher-Order Guided Diffusion for Graph Generation (https://arxiv.org/abs/2502.04308)
- **What's New**: 본 논문에서는 Higher-order Guided Diffusion (HOG-Diff)라는 새로운 그래프 생성 모델을 제안합니다. 이 모델은 생성 큐럴릭럼(curiculum)을 따라 점진적으로 현실적인 그래프를 생성하며, 복잡한 위상 구조를 포착하는 능력을 갖추고 있습니다. HOG-Diff는 높은 차원 정보를 기반으로 하여 전통적인 확산(diffusion) 모델보다 더 강력한 이론적 보장을 보여줍니다.

- **Technical Details**: HOG-Diff는 고차원 위상 정보에 의해 안내되는 그래프 생성 커리큘럼을 도입하여 복잡한 그래프 특성을 포착합니다. 이 모델은 그래프 생성 작업을 더 관리하기 쉬운 서브 작업으로 나누어, 먼저 핵심 구조를 포착하는 높은 차원의 그래프 스켈레톤을 생성한 후, 이를 세부 사항으로 정제하여 완전한 그래프를 생성합니다. 또한, HOG-Diff는 확산 다리(diffusion bridge)와 스펙트럴 확산(spectral diffusion)을 통합하여 효율적인 생성과 그래프 생성 원칙 준수를 보장합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, HOG-Diff는 다른 최신 모델들과 비교했을 때 지속적으로 뛰어난 성능을 보였습니다. 모델의 유용성은 상위 구조 정보를 활용함으로써 그래프 생성의 기능적 중요성을 강조합니다. 이 결과는 HOG-Diff가 기존의 그래프 생성 모델보다 더 나은 성능을 발휘함을 나타냅니다.



### DexterityGen: Foundation Controller for Unprecedented Dexterity (https://arxiv.org/abs/2502.04307)
Comments:
          Project: this https URL

- **What's New**: 본 논문에서는 DexterityGen(DexGen)라는 새로운 훈련 프레임워크를 소개합니다. DexGen은 RL(강화학습)을 사용하여 대규모의 섬세한 운동 원리(motion primitives)를 사전 훈련하며, 이를 통해 로봇이 안전한 행동을 생성하도록 합니다. 실시간 응용에서는 사람의 원격 조작(teleoperation)을 사용하여 DexGen이 의미 있는 조작 기술을 실행하도록 유도합니다.

- **Technical Details**: DexGen 컨트롤러는 로봇 상태를 고려하여 안정적이고 효과적인 행동을 생성하는 조건부 행동 모델을 기반으로 합니다. 이 모델은 시뮬레이션 데이터셋을 통해 사전 훈련되어, 고수준의 명령을 저수준의 구체적인 행동으로 변환하는 데 도움을 줍니다. 이 연구에서는 물체와의 상호작용을 기반으로 한 다양한 조작 기술의 시뮬레이션과 실제 환경에서의 문제 해결을 평가합니다.

- **Performance Highlights**: DexGen은 다양한 작업에서 안정성을 10-100배 향상시키는 데 성공했습니다. 특히, 로봇이 주어진 주사기나 스크류드라이버와 같은 도구를 사용하고 물체를 재배치하는 등의 복잡한 작업을 수행하는 데 있어 탁월한 능력을 보여주었습니다. 이러한 결과는 저수준의 조작 명령을 효과적으로 처리할 수 있는 능력을 입증합니다.



### Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants (https://arxiv.org/abs/2502.04290)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 논문에서는 유망하지 않은 평가를 최소화하기 위해 최적 영역에 전략적으로 집중하는 새로운 글로벌 최적화 알고리즘인 Every Call is Precious (ECP)를 소개합니다. ECP는 Lipschitz 상수를 추정할 필요가 없으므로 추가 함수 평가를 피할 수 있게 됩니다. 또한, 무한한 평가 예산에 대해 노 레그렛(no-regret) 성능을 보장하고, 유한한 예산에서는 미니맥스 최적의 레그렛 (regret) 경계를 달성합니다.

- **Technical Details**: ECP는 작은 수용 영역을 시작으로 점진적으로 최적화 문제를 풀어나가며, εtsubscript𝜀𝑡의 증가되는 값과 이전 반복에서 관찰된 점들을 활용합니다. 이러한 방식은 비용이 많이 드는 목적 함수에서 최적의 점을 평가하는 데 필요한 효율성을 크게 향상시킵니다. 또한, 이 알고리즘은 제한된 평가 예산에서도 모든 평가된 점이 가능한 최대 값의 후보가 되도록 보장합니다.

- **Performance Highlights**: ECP는 30개의 비볼록 다차원 최적화 문제에서 10개의 벤치마크 알고리즘과 비교해 우수한 성과를 보입니다. 이 알고리즘은 Lipschitz, Bayesian, Bandits, 진화적 방법들을 포함한 여러 최신 방법들을 초월하는 성능을 기록했습니다. 또한, 하이퍼 파라미터 선택에 대한 강건성이 입증되었습니다.



### How does a Multilingual LM Handle Multiple Languages? (https://arxiv.org/abs/2502.04269)
Comments:
          10 pages, 8 figures

- **What's New**: 이번 연구는 BLOOM 1.7B와 같은 다언어 모델들의 언어 처리 능력을 면밀히 평가하며, 저자원 언어에 대한 모델의 효과성을 중점적으로 다룹니다. 이 논문에서는 고자원 언어와 저자원 언어 간의 의미적 유사성을 연구하며, 모델의 내적 문법 및 의미 인코딩을 검토합니다. 특히, 이 연구는 다언어 NLP 모델의 개선을 목표로 하여 이를 통해 언어 기술의 포괄성을 제고하려고 합니다.

- **Technical Details**: 이 연구에서는 다언어 단어 임베딩 분석과 예측 작업을 통해 다언어 모델의 성능과 특성을 분석합니다. Google Cloud Translation API를 활용한 고품질 번역을 통해 5,000개의 영어 단어로 구성된 코퍼스를 생성하고, 이를 사용하여 cosine similarity를 통해 의미적 유사성을 평가합니다. 또한, 모델의 내부 동작을 이해하기 위해 숨겨진 상태 분석을 수행하며, 각 레이어의 역할과 과제별 기능을 평가합니다.

- **Performance Highlights**: BLOOM 모델의 성능은 인도유럽어 계열 언어인 힌디어와 타밀어가 초기 레이어에서 0.92-0.95로 높은 유사성을 보인 반면, 아랍어는 초기 유사성이 0.50으로 낮았음을 보여줍니다. 심층 레이어에서는 모든 언어가 유사성 점수가 감소하였으며, 특히 아랍어는 상대적으로 낮은 유사성 점수를 유지했습니다. 이러한 결과는 모델이 저자원 언어 및 특정 언어 과제를 처리하는데 있어 문제를 드러냅니다.



### Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances (https://arxiv.org/abs/2502.04268)
Comments:
          11 pages, 5 figures, 10 tables

- **What's New**: 오리엔티드 객체 탐지(ood)는 자율주행, 항공 이미지, 소매 장면 등에서 필수적인 작업으로 자리잡고 있다. 최근 RBox 라벨링의 비용이 높아지는 현실 속에서, 본 연구는 포인트 주도의 OOD를 통해 효율적으로 객체 감지를 수행할 수 있는 Point2RBox-v2를 소개한다. 이 방법은 객체 간의 공간적 레이아웃을 활용하여 새로운 시너지를 창출하는 데 초점을 맞추고 있다.

- **Technical Details**: Point2RBox-v2의 핵심 원리는 3가지 손실 함수에 기반한다: 첫째, Gaussian overlap loss로 객체를 2D Gaussian 분포로 보고 겹침을 최소화하여 각 인스턴스의 상한을 학습한다. 둘째, Voronoi watershed loss는 Voronoi 분할을 통해 각 인스턴스의 하한을 학습한다. 셋째, consistency loss를 통해 입력 이미지와 증강된 뷰 사이의 크기 및 회전 변화를 학습한다.

- **Performance Highlights**: Point2RBox-v2는 DOTA, HRSC, FAIR1M 벤치마크에서 각각 62.61%, 86.15%, 34.71%의 성능을 기록하며, 밀집된 장면에서 기존 기술보다 우수한 결과를 보인다. 제안된 알고리즘은 경량화되었고, 효율적인 객체 탐지를 제공할 것으로 기대된다. 제공된 코드 또한 연구자들이 사용할 수 있게 공개된다.



### Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion (https://arxiv.org/abs/2502.04263)
Comments:
          Accepted for publication at ICLR 2025

- **What's New**: 이 논문에서는 CLIP와 같은 다중 모달 (multi-modal) Vision-Language Models (VLMs)에서 텍스트와 이미지 인코더를 개별적으로 사용하는 관행이 비효율적임을 보여줍니다. 특히, 이미지 간 유사성 (intra-image similarity) 및 텍스트 간 유사성 (intra-text similarity) 문제를 'intra-modal misalignment'라는 개념으로 정의하여, 이러한 비효율성이 발생하는 원인을 설명합니다.

- **Technical Details**: 저자들은 'modality inversion'을 활용하여 입력 모달리티에서 보완 모달리티로의 표현 변환을 수행합니다. 이를 위해 Optimization-based Textual Inversion (OTI) 및 Optimization-based Visual Inversion (OVI) 기법을 도입하여 고정된 인코더를 사용하여 특징을 변환합니다. 실험을 통해 15개 이상의 데이터셋에서 intra-modal 작업에 대한 inter-modal 방식 접근의 성과 향상을 입증합니다.

- **Performance Highlights**: 실험 결과, inter-modal 접근 방식이 intra-modal 기초 성능을 초과하는 것으로 나타났습니다. CLIP의 인코더 간 intra-modal 유사성을 이용하는 기존의 방법이 성능 저하를 일으킨다는 점을 강조하며, VLM 사전 훈련 과정에서 intra-modal 손실 항을 포함하거나 텍스트와 이미지 임베딩 공간 간의 갭을 줄이면 intra-modal misalignment 문제를 완화할 수 있음을 시사합니다.



### TriNER: A Series of Named Entity Recognition Models For Hindi, Bengali & Marath (https://arxiv.org/abs/2502.04245)
- **What's New**: 이번 연구에서는 인도의 세 가지 주요 언어인 힌디어(Hindi), 벵골어(Bengali), 마라티어(Marathi)에 대한 다국어(NER) 모델을 개발하였습니다. 이러한 언어의 복잡성과 다양성 때문에 발생하는 문제를 해결하기 위해, 단일 모델을 통해 다양한 엔티티 그룹을 통합적으로 식별하는 방법을 제시합니다.

- **Technical Details**: 커스텀 트랜스포머 모델을 훈련하고 몇 가지 사전 훈련된(pretrained) 모델을 미세 조정(fine-tune)하여, 총 6개의 엔티티 그룹에 대해 F1 점수 92.11을 달성했습니다. 이 모델은 자연어 처리의 핵심 작업 중 하나인 개체 인식(NER)의 성능을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: 이 논문에 제시된 모델은 서로 다른 언어 간의 엔티티 그룹 및 태그 이름의 불일치를 크게 줄이는 데 기여할 것으로 기대됩니다. 다국어 NER 모델의 도입은 인도의 다양한 언어 환경에서 더욱 효과적으로 활용될 수 있는 가능성을 제공합니다.



### A Theoretical Framework for Data Efficient Multi-Source Transfer Learning Based on Cram\'er-Rao Bound (https://arxiv.org/abs/2502.04242)
- **What's New**: 본 논문에서는 다중 출처 전이 학습(multi-source transfer learning)에서 데이터 부족 문제를 해결하기 위한 새로운 이론적 프레임워크를 제안합니다. 기존의 연구들이 모든 가능한 샘플을 사용하여 훈련하는 반면, 본 연구는 각 출처 과제로부터 필요한 최적의 샘플 양을 도출하는 접근 방식을 취하고 있습니다.

- **Technical Details**: 제안된 방법론은 교차 엔트로피 손실(cross-entropy loss)에 맞춰 일반화 오차 측정(generalization error measure)을 도입하고, Cramér-Rao Bound를 기반으로 최적의 전이 양을 최소화하여 결정합니다. OTQMS라는 아키텍처 비종속(data-efficient) 알고리즘을 통해 이론적 결과를 구현하여 깊은 다중 출처 전이 학습 모델을 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 다양한 아키텍처와 두 가지 실제 벤치마크 데이터셋을 대상으로 한 분석을 통해 제안된 알고리즘이 최신 기법들(state-of-the-art approaches)보다 정확도와 데이터 효율성 모두에서显著하게 성과를 보였습니다.



### XAttnMark: Learning Robust Audio Watermarking with Cross-Attention (https://arxiv.org/abs/2502.04230)
Comments:
          24 pages, 10 figures

- **What's New**: 이번 논문에서는 Cross-Attention Robust Audio Watermark (XAttnMark)를 통해 오디오 워터마킹의 새로운 접근 방식을 제안합니다. 이는 생성기(generator)와 탐지기(detector) 간의 부분 매개변수 공유(partial parameter sharing)와 효율적인 메시지 검색을 위한 교차 주의(cross-attention) 메커니즘을 적용하여, 강력한 검출과 정확한 귀속(attribution)을 동시에 달성할 수 있도록 설계되었습니다. 또한 새로운 심리음향(psychoacoustic)에 기반한 시간-주파수 마스킹 손실(timely-frequency masking loss)을 도입하여 워터마크의 인지 불가능성을 높였습니다.

- **Technical Details**: XAttnMark는 생성기와 탐지기 간의 매개변수 공유 및 교차 주의 모듈을 통해 메시지 검색의 효율성을 크게 향상시킵니다. 특히, 워터마크 인지 불가능성을 보장하기 위해 비대칭 2D 커널을 사용하여 마스킹 에너지를 계산하고, 그것을 기반으로하는 손실 함수를 설계하였습니다. 이러한 접근법은 다양한 오디오 편집 변환에도 저항력이 뛰어나며, 출처 확인과 검출의 성능을 동시에 최적화할 수 있는 방향으로 나아갑니다.

- **Performance Highlights**: 우리의 방법은 여러 가지 오디오 변형에 대해서도 뛰어난 성능을 보이며, 현재까지의 최고의 검출 및 귀속 성능을 기록했습니다. 특히 강력한 편집 강도가 가해지더라도 성공적으로 워터마크 검출이 가능하다는 점에서 XAttnMark는 기존의 어떠한 방법론보다 두드러진 성과를 보입니다. 이러한 결과는 단순한 품질 유지에 그치지 않고, 실제적이고 다양한 도전 상황에서도 방어력을 인정받았습니다.



### Dark Distillation: Backdooring Distilled Datasets without Accessing Raw Data (https://arxiv.org/abs/2502.04229)
- **What's New**: 이 논문에서는 데이터셋 증류(Dataset Distillation, DD) 과정에서의 보안 위협을 새롭게 다루고 있습니다. 기존 연구들은 데이터셋 소유자가 데이터에 백도어를 삽입한다고 가정하였으나, 본 연구는 제3자가 데이터 분배 과정에서 데이터셋에 백도어를 삽입하여 악의적으로 배포할 수 있는 가능성을 논의합니다. 이러한 접근은.dd 데이터셋을 여전히 취약하게 만들며, 악의적인 데이터셋을 특정 조건 하에서 불과 1분 안에 생성할 수 있다는 점을 강조합니다.

- **Technical Details**: 이 연구는 제3자의 데이터셋 공격 모델을 설정하여, 원시 데이터 접근 권한 없이 백도어를 삽입하는 구성 요소를 제안합니다. 이 과정에서는 증류된 데이터셋에서 각 클래스에 대한 개념적 원형을 재구성하고, 이 원형에 백도어를 주입하여 원래의 최적화 경로를 보존하는 하이브리드 손실 함수(hybrid loss)를 설계합니다. 이로써 실제 이미지에서 백도어가 신뢰성 있게 활성화될 수 있도록 하면서도, 정상 작업에서의 성능 저하를 최소화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 방법이 다양한 데이터셋과 DD 방법, 네트워크 구조 및 훈련 전략에 걸쳐 일반화 가능하다는 점이 입증되었습니다. 뿐만 아니라, 우리의 공격 방법은 효율적이며, 특정 상황에서 악의적인 증류 데이터셋을 불과 1분 안에 합성할 수 있는 능력을 가지고 있습니다. 이러한 결과는 데이터셋 증류가 본래 보안적인 방법으로 간주되고 있다는 기존 믿음에 도전하며, DD의 심각한 보안 취약점을 드러냅니다.



### NLP-Based .NET CLR Event Logs Analyzer (https://arxiv.org/abs/2502.04219)
- **What's New**: 이번 논문에서는 .NET CLR 이벤트 로그 분석을 위한 새로운 도구를 제안합니다. 이 도구는 자연어 처리(Natural Language Processing, NLP) 접근 방식에서 영감을 받은 혁신적인 방법론을 기반으로 합니다. 연구의 초점은 소프트웨어 시스템의 모니터링과 최적화 필요성을 충족하기 위한 이벤트 로그 분석입니다. BERT 기반 아키텍처와 이벤트 로그에 맞게 커스터마이즈된 토크나이제이션 프로세스를 활용하여 실험을 진행하였습니다.

- **Technical Details**: 우리는 범위가 넓고 레이블이 없는 대량의 데이터셋을 다루기 위해 비지도 학습을 활용하였습니다. 특히,.transformer 기반 NLP 방법을 사용하여 이벤트 트레이스의 패턴과 이상치를 식별합니다. 주요 알고리즘으로 BPE (Byte Pair Encoding)를 선택했으며, 이 알고리즘은 가장 일반적인 토큰 쌍을 병합하여 사전을 저장합니다. 최종적으로, 수집된 이벤트 로그는 고유한 유니코드 문자를 사용하여 표현됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 이벤트 시퀀스를 압축하고 반복 패턴을 감지하며 이상치를 식별하는 데 효과적임을 보여줍니다. 특히, 이상치 감지에서 높은 정확도를 기록했습니다. 이는 NLP 기법이 소프트웨어 시스템의 신뢰성과 안정성을 개선하는 데 기여할 수 있음을 입증합니다. 이러한 접근 방식은 학술적 실험뿐만 아니라 산업에서 발생하는 작업 해결에도 유용합니다.



### Algorithmic causal structure emerging through compression (https://arxiv.org/abs/2502.04210)
- **What's New**: 본 연구에서는 인과관계(causality), 대칭성(symmetry), 그리고 압축(compression) 간의 관계를 탐구합니다. 기존의 학습(learning)과 압축 간의 연결 고리를 발전시키고 일반화하여 인과 모델(causal models)이 식별 불가능한 상황을 설정합니다. 새로운 프레임워크(framework)를 제안하여, 다수의 환경에서 데이터를 압축할 때 인과관계가 나타나는 결과로 출현한다고 주장합니다.

- **Technical Details**: 본 논문에서는 전통적인 인과 식별(possible assumptions for causal identifiability)이 적용되지 않을 때 사용할 수 있는 알고리즘적 인과관계(algorithmic causality)의 정의를 제시합니다. 이는 Kolmogorov 복잡성(Kolmogorov complexity)의 상한을 최소화하는 과정에서 나타나는 알고리즘적 인과 및 대칭 구조(symmetric structures)에 대한 이해를 확장합니다. 개입(target intervention) 대상으로부터의 지식 없이도 이러한 구조가 어떻게 발생하는지를 보여줍니다.

- **Performance Highlights**: 이 연구는 큰 언어 모델(large language models)과 같은 머신러닝 모델에서 인과관계가 명시적으로 식별되지 않을 때, 인과가 어떻게 출현할 수 있는지에 대한 새로운 관점을 제공할 수 있다고 가설합니다. 이러한 통찰력은 머신러닝의 다양한 분야에 적용될 수 있는 가능성을 지니고 있습니다.



### The Best Instruction-Tuning Data are Those That F (https://arxiv.org/abs/2502.04194)
- **What's New**: 이번 논문에서는 GRAPE라는 새로운 SFT(Supervised Fine-Tuning) 프레임워크를 제안합니다. 이는 타겟 모델의 사전 훈련 분포에 가장 가까운 응답을 선택하여, 데이터 수집 과정에서 발생할 수 있는 성능 저하 문제를 해결하는 데 중점을 둡니다. GRAPE는 다양한 LLM(Language Model)에서 응답을 수집하고, 그 중 타겟 모델에 대해 높은 확률을 보이는 응답을 선택하여 데이터의 질을 향상시킵니다.

- **Technical Details**: GRAPE는 여러 모델에서 수집된 응답 중 타겟 모델과 가장 유사한 응답을 선택하여 SFT를 진행합니다. 이 과정은 타겟 모델의 확률을 이용해 이루어지며, 기존의 일률적인 응답 대신 모델에 적합한 응답을 사용합니다. 특히, GRAPE의 접근 방식은 데이터 분포의 이동에 따른 문제를 최소화하여 성능 향상을 도모합니다.

- **Performance Highlights**: GRAPE는 LLaMA3.1-8B, Mistral-7B 및 Qwen2.5-7B와 같은 일반적으로 사용되는 LLM에서 테스트되었으며, 기존 베이스라인보다 최대 17.3%의 성능 향상을 기록했습니다. 또한, GRAPE 선택 데이터를 사용하여 Tulu3 및 Olmo2에 대한 후속 데이터에서도 강력한 성능 개선을 보여주었습니다. GRAPE는 데이터의 양을 줄이고 학습 에폭수를 절반으로 줄여도 높은 성능을 유지함으로써 SFT 과정에서 높은 효율을 입증했습니다.



### Archetypal Analysis for Binary Data (https://arxiv.org/abs/2502.04172)
Comments:
          5 pages, Accepted at ICASSP 2025

- **What's New**: 이번 연구에서는 이진 데이터(binary data)에 적합한 아키타이플 분석(archetypal analysis, AA)을 위한 두 가지 새로운 최적화 프레임워크를 제안하고 있습니다. 기존의 AA 방법들이 연속 데이터에 기반하여 개발된 반면, 우리가 제안한 방법은 이진 데이터에 특화되어 있습니다. 특히, Bernoulli 분포에 기반한 두 가지 프레임워크를 통해 아키타입을 정의하고, 관측 데이터를 효율적으로 재구성할 수 있는 방법을 제시합니다.

- **Technical Details**: 첫 번째 프레임워크는 아키타입을 정의하기 위해 적Active set algorithm을 활용하여 희소성(sparsity)을 이용한 접근 방식을 채택하였습니다. 두 번째 프레임워크는 주어진 관측 데이터를 위해 Sequential Minimal Optimization (SMO) 기법을 사용하여 저 차원(low-dimensional) 행렬을 효율적으로 업데이트하는 방법을 다룹니다. 이 두 방법 모두 Bernoulli 분포에 최적화되었으며, 다른 데이터 분포에도 쉽게 확장 가능하다는 장점을 가지고 있습니다.

- **Performance Highlights**: 제안된 최적화 절차는 기존에 사용되던 곱셈 기반 이진 AA 방법보다 우수한 성능을 보임을 확인하였습니다. 연구에서는 합성 데이터(synthetic data) 및 실제 이진 데이터를 통해 두 새로운 접근 방식의 성능 우위를 입증하였습니다. 이로 인해, 아키타이플 분석의 적용 범위가 확대되고 다양한 데이터 환경에서 효율적인 최적화 프레임워크 활용이 가능하다는 것을 보여줍니다.



### UltraIF: Advancing Instruction Following from the Wild (https://arxiv.org/abs/2502.04153)
- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)의 복잡한 지시사항을 따를 수 있는 능력을 향상시키기 위한 새로운 접근법인 UltraIF를 제안합니다. Open-source(오픈 소스) 데이터를 활용하여 LLM을 설계함으로써, 공개된 데이터와 대기업에 의해 훈련된 모델 간의 격차를 해소하고자 합니다.

- **Technical Details**: UltraIF는 실제 사용자 요청을 간단한 쿼리와 제약조건, 그리고 그 제약조건에 대한 평가 질문으로 분해합니다. 이후 UltraComposer라는 모델을 훈련시켜 제약조건과 관련된 프롬프트를 평가 질문과 함께 구성합니다. 이를 통해 복잡한 지시사항을 합성하고, 평가 질문을 이용해 응답을 필터링할 수 있습니다.

- **Performance Highlights**: 실험 결과, UltraIF는 LLaMA-3.1-8B-Base 모델을 성공적으로 조정하여, 5가지 지시사항 처리 기준에서 인스트럭트 버전과 비교해 동등한 성능을 발휘했습니다. 또한, UltraIF는 LLaMA-3.1-8B-Instruct 모델을 self-alignment를 통해 더욱 개선할 수 있음을 보여주었습니다. 이 접근법은 다양한 사용 사례에 대한 가능성을 제시합니다.



### Synthetic Datasets for Machine Learning on Spatio-Temporal Graphs using PDEs (https://arxiv.org/abs/2502.04140)
Comments:
          Currently under review

- **What's New**: 본 논문은 PDE(Partial Differential Equations)를 기반으로 한 최초의 시공간(spatio-temporal) 데이터셋을 제공하며, 이는 그래프 형태로 설계되어 있습니다. 연구자들은 이 데이터셋을 사용하여 특정 응용 프로그램에 맞춘 맞춤형 데이터셋 및 벤치마크를 생성할 수 있습니다. 특히, 이 연구는 실제 전염병 데이터에 대한 모델 성능을 향상시키는 데 기여할 수 있는 사전 학습(pre-training) 방법도 제안합니다.

- **Technical Details**: 이 연구에서는 유한 요소 방법(Finite Element Method, FEM)을 이용하여 다양한 종류의 재난을 모델링하는 세 가지 PDE를 해결했습니다. 생성된 데이터셋은 각기 다른 전염병 역학, 대기 입자 및 쓰나미 파동과 관련된 시나리오를 다루고 있습니다. 이들 PDE는 불규칙한 도메인에서 해결되며, 데이터는 불규칙하게 분포된 관련 지점을 기반으로 제공합니다.

- **Performance Highlights**: 데이터셋을 기반으로 한 벤치마크 결과, 비슷한 문제를 해결하기 위해 기존 기계 학습 모델들이 효과적으로 평가되었습니다. 전염병 데이터셋에 대한 첫 번째 사전 학습 결과는 최대 45%의 성능 향상을 보여 주며, 이는 이 연구의 중요성을 강조합니다. 제공된 코드와 데이터셋 덕분에 연구자들은 개인의 요구에 맞게 전이 학습(transfer learning)을 쉽게 수행할 수 있습니다.



### Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis (https://arxiv.org/abs/2502.04128)
- **What's New**: 최근 언어 모델(GPT 시리즈 및 o1 모델)의 발전은 훈련 및 추론 시 컴퓨팅 리소스를 확장하는 것이 효과적임을 보여주고 있습니다. 이 논문은 확장된 훈련 및 추론 성능을 갖춘 단일 레이어 벡터 양자화(VQ) 코덱과 Transformer 아키텍처를 활용한 TTS 시스템 LLaSA를 제안합니다. LLaSA는 TTS의 자연스러운 음성과 복잡한 프러소디 패턴 생성을 개선하며, 기존의 다단계 TTS 시스템의 복잡성을 줄입니다.

- **Technical Details**: LLaSA는 LLaMA 모델을 기반으로 하여 음성 토큰을 포함하는 확장된 어휘를 사용합니다. 훈련 시, 자연스러운 음성과 프러소디 정확성을 높이기 위해 훈련 시간을 늘리는 것이 중요하다고 설명하고 있습니다. 또한, LLaSA는 음성 이해 모델을 확인자로 사용하여 추론 시 생성 출력을 특정 확인자 바이어스에 맞추어 더 감정적으로 표현할 수 있도록 하는 최적화 기법을 적용합니다.

- **Performance Highlights**: 리브리Speech(LibriSpeech) 테스트 세트 실험에서 LLaSA는 최첨단 성능을 달성하였으며, 감정 표현, 음색 일관성 및 콘텐츠 정확도를 크게 향상시켰습니다. 이 연구에서 제안된 모델은 오픈소스가 되어, TTS 커뮤니티의 혁신을 촉진할 것으로 기대됩니다. 실험 결과, 훈련 및 추론 단계의 확장이 TTS 성능을 크게 향상시키는 것을 입증했습니다.



### Ancient Greek Technology: An Immersive Learning Use Case Described Using a Co-Intelligent Custom ChatGPT Assistan (https://arxiv.org/abs/2502.04110)
Comments:
          5 pages, presented at the 2024 IEEE 3rd International Conference on Intelligent Reality (ICIR 2024), 6th of December, 2024

- **What's New**: 이 논문은 Immersive Learning Case Sheet (ILCS)를 활용하여 몰입형 학습 사례의 설명을 표준화하는 방법을 제안합니다. 연구팀 members는 ILCS와 사례 내용에 대한 친숙도가 달랐기 때문에, 일관된 용어 사용과 프로세스 정렬을 돕기 위해 맞춤형 ChatGPT assistant를 개발했습니다. 이 연구는 몰입형 학습 문헌에 대한 새로운 기여로서 구조화된 사례 보고서의 중요성을 보여줍니다.

- **Technical Details**: ILCS는 몰입형 학습 사례 설명의 일관성을 높이기 위한 방법론적 도구입니다. 이 연구에서는 VRChat을 통해 고대 그리스 기술에 대한 몰입형 학습 경우를 적용했으며, 팀원들은 ILCS 사용에 대한 이해도가 달랐습니다. 최종 ILCS 개발 과정에서 ChatGPT assistant의 도움으로 용어의 일관성과 품질이 향상되었다고 합니다.

- **Performance Highlights**: 연구 결과에 따르면 ILCS는 사례에 대한 구조화된 반성과 해석을 지원합니다. AI 기반 도구의 사용은 연구 관행의 협력과 표준화를 향상시킬 수 있는 잠재력을 보여주었습니다. 그러나 해석 작업에 대한 AI 의존성과 팀 내 전문성 수준의 차이를 관리하는 데에 어려움이 있음을 언급하며, 몰입형 학습 연구 프로세스의 표준화에 있어 AI의 실제 적용에 대한 통찰력을 제공합니다.



### VTutor: An Open-Source SDK for Generative AI-Powered Animated Pedagogical Agents with Multi-Media Outpu (https://arxiv.org/abs/2502.04103)
- **What's New**: 이 논문에서는 VTutor라는 새로운 오픈소스 소프트웨어 개발 키트(SDK)를 소개합니다. VTutor는 Generative AI와 첨단 애니메이션 기술을 결합하여 인간-AI 멀티미디어 상호작용을 위한 매력적이고 적응 가능한 애니메이티드 교육 에이전트(APAs)를 생성합니다. 이 SDK는 LLMs를 사용하여 개인화된 피드백을 실시간으로 제공하고, WebGL 렌더링을 통해 웹 통합을 매끄럽게 하는 기능을 갖추고 있습니다.

- **Technical Details**: VTutor는 텍스트, 음성, 애니메이션을 통합하여 상호작용의 질을 향상시키는 다중미디어 출력 프레임워크를 제공합니다. 사용자는 OpenAI, Azure, Google Cloud Platform 등의 TTS 서비스를 이용해 음성을 생성하고, 실시간 오디오 스트림을 제공하여 더 정교하고 반응적인 상호작용이 가능합니다. VTutor의 LipSync 구성 요소는 uLipSync라는 오픈소스 도구를 활용하여 음성 입력에 따라 입모양 동기화를 실시간으로 수행합니다.

- **Performance Highlights**: VTutor는 다양한 2D 및 3D 캐릭터 모델을 지원하여 사용자 맞춤형 APAs 디자인에 유연성을 제공합니다. 이를 통해 학습자의 상호작용 경험을 개선하며, 자연스러운 음성과 애니메이션 표현을 통해 감정적이며 신뢰성 있는 AI 원칙을 교육 환경에 적용합니다. VTutor는 차세대 APAs의 새로운 기준을 설정하며, 의미 있고 몰입감 있는 인간-AI 상호작용 경험을 촉진하는 접근성 있고 확장 가능한 솔루션을 제공합니다.



### Efficient Few-Shot Continual Learning in Vision-Language Models (https://arxiv.org/abs/2502.04098)
- **What's New**: 이 논문에서는 LoRSU(Low-Rank Adaptation with Structured Updates)라는 새로운 방법을 제안합니다. 이 방법은 VLMs(Vision-Language Models) 내에서 이미지 인코더를 선택적으로 업데이트함으로써 효율적이고 강력한 성능 개선을 목표로 합니다. LoRSU는 구조적 및 지역화된 매개변수 업데이트를 도입하여 기존의 이미지 인코더가 자주 발생하는 오류를 효과적으로 수정하고, 모델의 일반적인 강건성을 유지합니다.

- **Technical Details**: LoRSU 방법은 사전 훈련된 이미지 인코더(CLIP 변형)를 사용하여 고정된 상태에서 VLM을 비주얼 어시스턴트로 배포할 때의 한계를 극복하고자 합니다. 이 방법은 기존 모델의 대부분 오버헤드를 줄이면서 모든 매개변수에 대한 일반적인 업데이트를 수행하여 파라미터 선택성을 극대화 합니다. LoRSU는 매개변수 선택을 위해 이론적 통찰력을 활용하여 중요 매개변수만 업데이트하여 자원 효율성을 달성합니다.

- **Performance Highlights**: 실험 결과, LoRSU는 VQA(Visual Question Answering) 작업에서 25배 이상의 계산 효율성을 보이며 성능을 희생하지 않고 이미지 인코더를 업데이트하는 데 있어 우수한 성능을 발휘하고 있습니다. 또한 이 방법은 특정 도메인으로의 전환에서도 기존 지식을 보존하며, 파라미터 수가 적은 이미지 인코더의 업데이트가 VLM의 성능 개선에 필수적임을 보여줍니다.



### LLMs to Support a Domain Specific Knowledge Assistan (https://arxiv.org/abs/2502.04095)
- **What's New**: 이 연구는 지속 가능성 보고를 위한 도메인 특화 지식 어시스턴트 개발에 대한 새로운 접근법을 제시합니다. 국제 재무 보고 기준(IFRS)를 기반으로 한 최초의 고품질 합성 질문-답변(QA) 데이터셋을 생성하여 기업의 IFRS 보고 지원을 위한 기초를 마련하였습니다. 이는 1,063개의 다양한 QA 쌍으로 구성되어 지속 가능성 보고에서의 여러 사용자 쿼리를 충족합니다.

- **Technical Details**: 이 프로젝트에서는 두 가지 아키텍처를 사용하여 지속 가능성 보고 영역에서 질문-답변 시스템을 설계하였습니다. RAG(검색 증강 생성) 파이프라인과 완전 LLM 기반 파이프라인이 포함되며, 두 아키텍처 모두 QA 데이터셋에 대해 실험하고 미세 조정을 거쳐 개발되었습니다. 최종 파이프라인은 도메인 특화 데이터를 기반으로 미세 조정된 LLM과 복잡한 쿼리를 처리하기 위한 산업 분류 기능을 포함하고 있습니다.

- **Performance Highlights**: RAG 아키텍처는 단일 산업 질문에서 85.32%, 교차 산업 질문에서 72.15%의 정확도를 달성하며 기준 방법론 대비 각각 4.67 및 19.21 퍼센트 포인트 향상되었습니다. 또한 LLM 기반 파이프라인은 단일 산업 질문에서 93.45%, 교차 산업 질문에서 80.30%의 정확도를 기록, 기준 대비 각각 12.80 및 27.36 퍼센트 포인트 개선을 보였습니다.



### Automatic quantification of breast cancer biomarkers from multiple 18F-FDG PET image segmentation (https://arxiv.org/abs/2502.04083)
Comments:
          Submit soon to EJNMMI Research

- **What's New**: 본 연구는 유방암 환자에서 선행항암요법(neoadjuvant chemotherapy, NAC) 후 18F-FDG PET 이미지를 이용하여 종양 분할을 자동으로 수행하는 시스템을 개발했습니다. 연구팀은 243개의 초기 18F-FDG PET 스캔(PET_Bl)과 180개의 추적 18F-FDG PET 스캔(PET_Fu)을 분석하여, 자동으로 종양 영역을 세분화하고 핵심 바이오마커를 추출했습니다. 이는 유방암의 치료 반응을 평가하고 진단의 정확성을 높이기 위한 중요한 첫걸음으로 평가됩니다.

- **Technical Details**: 연구에서는 nnUNet 딥러닝 모델을 사용하여 PET_Bl 스캔에서 종양을 정확히 분할했습니다. PET_Fu 스캔의 경우, 모델을 15개의 후속 검사로 미세 조정하여 NAC 후의 종양 진화를 평가할 수 있도록 하였습니다. 이 과정에서 최대 표준 섭취량(maximum standardized uptake value, SUVmax), 대사 종양 부피(metabolic tumor volume, MTV), 총 병변 당화(total lesion glycolysis, TLG) 등의 바이오마커를 계산하여 종양의 변화를 파악했습니다.

- **Performance Highlights**: nnUNet 모델은 PET_Bl 스캔에서 0.89의 Dice similarity coefficient (DSC)와 3.52 mm의 Hausdorff distance (HD)를 기록하며 우수한 성능을 보였습니다. PET_Fu 스캔에서 미세 조정 후에는 DSC 0.78과 HD 4.95 mm를 달성했습니다. 전반적으로 SUVmax, MTV 및 TLG의 중요한 평균 감소가 관찰되어, 수집된 바이오마커가 유방암의 진행 상황을 평가하는 데 유의미하다는 것을 입증했습니다.



### Predicting Large Language Model Capabilities on Closed-Book QA Tasks Using Only Information Available Prior to Training (https://arxiv.org/abs/2502.04066)
- **What's New**: 이 논문에서는 OpenAI의 GPT-4 기술 보고서를 기반으로 특정 작업에 대한 모델의 성능을 훈련 전에 예측할 수 있다는 새로운 접근 방식을 제시합니다. 이 접근 방식은 리소스 할당을 최적화하고 데이터가 목표 작업과 일치하도록 보장하는 데 중요한 역할을 합니다. 특히, 약 1.5조 토큰의 데이터를 사용하여 세 가지 대형 언어 모델(1.6B, 7B, 13B)을 사전 훈련하며, 사전 훈련 데이터와 지식 보유, 작업별 지식 보유를 예측하는 데 집중합니다.

- **Technical Details**: 이 연구는 560,000달러와 520,000 GPU 시간을 투입하여 세 가지 대형 언어 모델을 사전 훈련했습니다. 사전 훈련 데이터 분석을 위해 지식 트리플(knowledge triples)을 사용하고, 모델의 지식 보유를 평가하는 데 SMI 지표(Size-dependent Mutual Information)를 도입합니다. 이 지표는 사전 훈련 데이터, 모델 크기, 작업별 지식 보유 간의 관계를 정량화하며, 여러 크기의 모델 기반으로 ACC와 SMI 간의 강한 선형 상관관계를 발견했습니다.

- **Performance Highlights**: 실험 결과 SMI 지표와 다양한 크기의 모델의 CBQA 작업에서의 정확도 간에 강한 선형 상관관계가 있음을 확인했습니다. Coefficient of Determination(R²) 값은 0.84를 초과하여 모형이 작업 특정 지식을 얼마나 잘 보유하고 있는지를 효과적으로 예측하는 데 매우 유용함을 시사합니다. 또한, 연구진은 1.6B 규모의 모델에 대한 사전 훈련 데이터와 가중치를 공개하여 후속 연구에 기여하고 있습니다.



### Probe-Free Low-Rank Activation Intervention (https://arxiv.org/abs/2502.04043)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문에서는 FLORAIN이라는 프로브가 필요 없는 활성화 개입 방법을 제안합니다. 이 방법은 특정 활성화 레이어의 모든 어텐션 헤드에 대해 적용되며, 이를 통해 불필요한 분류기 훈련의 필요성을 제거했습니다. FLORAIN은 비선형 저차원 매핑으로 매개변수화되어 있으며, 이는 수정된 활성화와 바람직한 콘텐츠의 매니폴드에서의 투영 간의 거리를 최소화하도록 훈련됩니다.

- **Technical Details**: FLORAIN은 활성화 벡터를 변형하는 방식으로, 기존의 활성화 개입 방법과는 다르게 다층 네트워크에 걸쳐 여러 헤드를 수정하는 대신 한 층에서 효율적으로 작동합니다. 이 방법의 핵심은 바람직한 답변의 영역을 묘사하고 관련된 저차원 변환 매핑을 고려하는 것입니다. 또한, Mahalanobis 거리 투영 아래에서의 분석적 형태로 목표 함수를 제공합니다.

- **Performance Highlights**: 여러 기본 모델을 기준으로 한 실험 결과, FLORAIN은 진실성과 품질을 개선하는 데 있어 여러 기준 방법들보다 일관되게 우수한 성능을 보였습니다. FLORAIN은 빠른 계산을 가능하게 하며, 병렬 처리 조건을 제공하여 후속 처리 시간 감소에 기여합니다.



### Leveraging Reasoning with Guidelines to Elicit and Utilize Knowledge for Enhancing Safety Alignmen (https://arxiv.org/abs/2502.04040)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문에서는 대규모 언어 모델의 안전성을 확보하기 위한 기존의 Refusal Training (RT) 방법의 한계를 분석하고, OOD(Out-Of-Distribution) 공격에 대한 일반화 성능을 향상시키기 위해 새로운 접근법을 제안합니다. 연구 결과, RT는 안전과 관련된 잠재적 지식을 일관되게 이끌어내지 못하는 문제를 드러냈고, 이를 해결하기 위해 Safety Reasoning with Guidelines (SRG)라는 방법을 소개합니다. SRG는 각 쿼리에 대해 단계별 이유 추론을 수행하도록 모델을 훈련시킵니다.

- **Technical Details**: 논문에서는 RT가 OOD 상황에서 일반화 능력이 부족하다고 지적하며, 훈련에서 직접적인 거부에 의존할 경우 모델이 피상적인 단축 경로에 의존하게 되고 탄력적인 표현 매핑을 학습하지 못한다고 설명합니다. 새로운 방법인 SRG는 3가지 필수 구성 요소를 포함하여, 지침에 따라 쿼리에 대한 단계적 추론을 수행하도록 모델을 훈련시키며, 이는 안전 관련 지식 활용을 효과적으로 증진시킵니다. 이 방법은 처리 과정에서 지침의 맥락 조화를 내재화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, SRG 방법이 기존의 RT 방식에 비해 OOD 공격에 대한 일반화 성능을 유의미하게 개선시키는 것으로 나타났습니다. BoN(Best-of-N) 평가 방식을 적용한 결과, N이 증가함에 따라 OOD 공격에 대한 실패 비율(ASR)이 감소하는 현상이 관찰되었습니다. 이 연구는 모델이 안전 관련 잠재 지식을 충분히 보유하고 있지만, 기존 방법이 이를 일관되게 이끌어내지 못했음을 입증하고 있습니다.



### Generalize Drug Response Prediction by Latent Independent Projection for Asymmetric Constrained Domain Generalization (https://arxiv.org/abs/2502.04034)
- **What's New**: 이 연구는 drug response 예측에서의 한계를 극복하기 위해 novel domain generalization framework인 panCancerDR을 제안합니다. 연구진은 각각의 암 유형을 별도의 source domain으로 간주하고, cell lines를 사용하여 domain-specific samples로 정의했습니다. 또한, 새로운 latent independence projection (LIP) 모듈을 도입하여 encoder가 informative하면서도 중복되지 않은 feature를 추출하도록 유도합니다.

- **Technical Details**: panCancerDR은 adversarial domain generalization 접근 방식을 통해 다양한 source domain에서 공통적인 task-relevant feature를 포착합니다. 이 모델은 drug-sensitive samples를 하나의 컴팩트 클러스터로, resistant samples를 여러 개의 클러스터로 분산시키도록 하는 asymmetric adaptive clustering constraint를 적용합니다. 이러한 구조는 각각의 cancer type에서 유래한 독특한 신호를 사용하는 예측 모델로 기능합니다.

- **Performance Highlights**: empirical 실험 결과 panCancerDR은 unseen cancer type에 대해 높은 예측 성능을 보여주며, 단일 세포 단위나 환자 수준에서도 더 나은 성능을 발휘합니다. 이 모델은 in vitro cell line 데이터만을 사용하여 학습되었음에도 불구하고 현재 state-of-the-art 방법들과 비교했을 때 동일하거나 더 뛰어난 결과를 달성했습니다. 이러한 성과는 실제 임상 적용 가능성을 강조합니다.



### Automating a Complete Software Test Process Using LLMs: An Automotive Case Study (https://arxiv.org/abs/2502.04008)
Comments:
          Accepted by International Conference on Software Engineering (ICSE) 2025

- **What's New**: 본 논문은 차량의 내부 시스템과 외부 애플리케이션 간의 상호작용을 검증하는 Vehicle API 테스트의 자동화 시스템을 제안합니다. 특히, 이 시스템은 다양한 문서와 시스템 사양 간의 불일치와 모호성을 해결하도록 설계되었습니다. 실험을 통해 100개가 넘는 API에서 차량 API 테스트를 자동화할 수 있음을 입증하였고, 이는 LLMs(대형 언어 모델)가 인간의 판단이 필요한 반복 작업을 능률적으로 처리할 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 SPAPI라는 웹 서버를 사례로 들어, REST APIs를 통한 차량 상태의 읽기 및 쓰기를 수행하는 과정을 설명합니다. SPAPI의 테스트는 2-3명의 전담 엔지니어 팀에 의해 수행되며, 새로운 API가 출시될 때마다 API 사양 검토, 차량 상태 이해, 테스트 입력 생성 등의 단계가 포함됩니다. 테스트 과정을 수동에서 자동으로 전환하기 위해, LLM은 명확히 정의된 작업을 수행하여 테스트 플로우의 안정성을 보장합니다.

- **Performance Highlights**: SPAPI 테스트의 자동화를 통해 API 제공 속도가 증가하고, 엔지니어들은 단순 반복 작업에서 벗어나 창의적인 작업에 전념할 수 있게 됩니다. 테스트 프로세스의 전자동화가 실현되면, 품질 향상과 더불어 팀 간 협업의 효율성도 높아지며, 기술 부채를 해소하는 데 중요한 역할을 하게 됩니다. 이 연구는 LLM을 활용한 전체 테스트 프로세스의 자동화를 통해 업계에서의 적용 가능성을 보여줍니다.



### Online Learning of Counter Categories and Ratings in PvP Games (https://arxiv.org/abs/2502.03998)
- **What's New**: 이번 논문에서는 전통적인 Elo 등급 시스템의 원칙을 확장하여 실시간 대응 범주 학습(Elo Residual Counter Category learning, Elo-RCC) 알고리즘을 제안합니다. 이 방법은 매 경기 후 실시간으로 등급과 대응 관계를 동적으로 조정하여 스칼라 등급의 설명 가능성을 유지하면서 비가역성을 해결합니다. 기존의 신경망 기반 방법들과는 달리, Elo-RCC는 빠른 업데이트와 균형 있는 매칭을 지원합니다.

- **Technical Details**: Elo-RCC는 기대-maximization (EM) 알고리즘을 활용하여 각 플레이어의 최상의 대응을 기반으로 능력의 확률 분포를 동적으로 학습합니다. 이를 통해 대응 관계와 평가 점수를 실시간으로 조정함으로써 비가역성을 처리하면서 스칼라 등급의 단순성과 해석 가능성을 결합합니다. 이 시스템은 기존의 Neural Rating Table과 Neural Counter Table의 한계를 극복하고, 제한된 전략 복잡성을 갖는 게임에서 성능을 검증했습니다.

- **Performance Highlights**: Elo-RCC는 Lin et al.(2024b)의 공개 데이터 세트를 사용하여 검증된 결과, NCT와 동등한 성능을 달성하면서 실시간 업데이트가 가능하다는 점에서의 실용성을 보여주었습니다. 특히, Elo-RCC는 동적 환경에서 대응 관계를 효과적으로 처리할 수 있는 능력을 강조하며, 온라인 업데이트 방법 중 가장 뛰어난 성능을 보였습니다.



### Ontology-Guided, Hybrid Prompt Learning for Generalization in Knowledge Graph Question Answering (https://arxiv.org/abs/2502.03992)
Comments:
          Accepted By ICSC 2025

- **What's New**: OntoSCPrompt는 여러 Knowledge Graph(KG) 간의 일반화 능력을 향상시키기 위한 혁신적인 두 단계 구조의 KGQA 시스템입니다. 첫 번째 단계에서는 특정 KG에 의존하지 않는 SPARQL 쿼리 구조를 생성하고, 두 번째 단계에서는 KG 특유의 정보를 채워 넣습니다. 또한, 온톨로지 기반의 하이브리드 프롬프트 학습 전략을 통해 KG에 대한 이해도를 증대시킵니다.

- **Technical Details**: KGQA 시스템은 질문을 SPARQL 쿼리 구조로 변환하는 두 가지 단계로 구성됩니다. 첫 번째 단계는 일반적인 SPARQL 구조를 예측하는 것이고, 두 번째 단계는 특정 KG에 대한 개념, 관계 및 엔티티로 이 구조를 채우는 것입니다. 이 과정에서, 다양한 KG의 이질성을 처리하기 위해 개념 및 복잡한 SPARQL 절에 대한 새로운 자리 표시자를 추가하였습니다.

- **Performance Highlights**: OntoSCPrompt는 CWQ, WebQSP 및 LC-QuAD 1.0과 같은 다양한 KGQA 데이터 세트에서 SOTA 접근 방식과 동등한 성과를 냈습니다. 실험 결과는 리트레이닝 없이도 효율적으로 작동하며, DBLP-QuAD 및 CoyPu KG와 같은 보지 못한 도메인 특정 KG에 대해서도 잘 일반화된다는 것을 보여줍니다.



### PGB: One-Shot Pruning for BERT via Weight Grouping and Permutation (https://arxiv.org/abs/2502.03984)
- **What's New**: 이 논문에서는 BERT의 비효율적인 구조를 개선하기 위해 새로운 반구조화 일회성 가지치기 방법인 'Permutation and Grouping for BERT' (PGB)를 제안합니다. PGB는 중요 그룹을 식별하고, 불필요한 가중치를 제거하여 높은 압축 효율성과 희소성을 유지하면서 정확성을 보존합니다. 이 방법은 기존의 반복 가지치기 및 지식 증류(knowledge distillation) 기법보다 간단하고 계산 비용이 적습니다.

- **Technical Details**: PGB 접근 방식은 BERT의 다중 헤드 어텐션(multi-head attention)과 피드포워드 레이어(feed-forward layers)에서 작동하여 개별 가중치의 중요 그룹을 구성하고, 중요하지 않은 모든 가중치를 구조적으로 가지치기합니다. 중요한 그룹이 형성되지 않은 경우 해당 레이어를 통째로 삭제하여 모델을 더욱 압축할 수 있습니다. 논문에서는 GLUE와 SQuAD 벤치마크에서 PGB를 BERT_BASE에 적용한 실험 결과를 통해 기법의 효과성을 입증하였습니다.

- **Performance Highlights**: PGB는 기존의 고급 구조적 가지치기 기법에 비해 계산 비용과 정확도 유지 측면에서 우수한 성능을 나타냈습니다. 실험 결과 PGB는 BERT_BASE 모델을 사용하여 효율성과 정확성 모두에서 최신 기술(SOTA) 가지치기 방법을 능가함을 보여주었습니다. 이를 통해 PGB는 자연어 처리(NLP) 작업에서 더 작은 모델을 보다 효과적으로 사용할 수 있는 가능성을 제시합니다.



### Towards Unified Music Emotion Recognition across Dimensional and Categorical Models (https://arxiv.org/abs/2502.03979)
- **What's New**: 본 논문에서는 Music Emotion Recognition (MER) 분야에서 발생하는 이질적인 감정 레이블 문제를 해결하기 위해 카테고리(label)와 차원(dimensional) 레이블을 결합한 통합 멀티태스크 학습 프레임워크를 제안합니다. 이를 통해 다양한 데이터셋에서의 학습이 가능해지며, 음악적 특징과 MERT embeddings를 결합한 효과적인 입력 표현을 사용하여 상당한 성능 향상을 보여줍니다. 또한, 개별 데이터셋에서 학습된 teacher 모델의 지식을 전이(knowledge distillation)하여 모델의 일반화 능력을 높였습니다.

- **Technical Details**: 제안된 프레임워크는 1) 대규모 자기 지도 학습(self-supervised training) 기반의 MERT embeddings, 2) 코드 진행의 화성적 표현, 3) 음악 키 등 세 가지 유형의 특징을 결합하여 음악 입력을 적절히 표현합니다. 또한, 지식 전이 방법인 knowledge distillation을 통해 이질적인 레이블 타입을 통합하여 학습을 unify하고, 데이터 증강(data augmentation) 전략을 포함시켜 다양한 오디오 입력에 대한 견고함을 확보합니다.

- **Performance Highlights**: 실험 결과, MERT embeddings와 고수준의 음악적 특징(예: 코드 진행 및 키)의 통합이 성능을 크게 향상시키며, 다양한 데이터셋에서 멀티태스크 학습을 통해 단일 데이터셋 성능 또한 개선됨을 보여주었습니다. 특히 MTG-Jamendo 데이터셋에서는 최신 모델들을 초월하는 성과를 기록하며, PR-AUC 0.1543 및 ROC-AUC 0.7810의 최고치를 기록했습니다. 이러한 결과는 MER에서 제안된 접근법의 효과성을 강조합니다.



### MultiFloodSynth: Multi-Annotated Flood Synthetic Dataset Generation (https://arxiv.org/abs/2502.03966)
Comments:
          6 pages, 6 figures. Accepted as Oral Presentation to AAAI 2025 Workshop on Good-Data

- **What's New**: 이 논문에서는 홍수 위험 감지 시스템을 위한 합성 데이터 생성 프레임워크인 MultiFloodSynth를 소개합니다. 이 프레임워크는 다양한 실제 속성을 가상 세계로 옮겨와 홍수 상황을 시뮬레이션하며, 기존 데이터 수집의 한계를 극복하기 위해 최신 generative models를 활용합니다. MultiFloodSynth는 5단계의 다양한 주석을 포함한 풍부한 합성 데이터셋을 제공하여, 실제 데이터셋과의 유사성을 유지하면서도 효율적인 모델 학습 환경을 조성합니다.

- **Technical Details**: 프레임워크는 3D 엔진을 이용하여 가상 도시 홍수 상황을 합성합니다. 이 과정에서 레이아웃(layout), 조명(lighting), 홍수 높이(flood-level)와 같은 다양한 속성을 고려하며, 여러 종류의 주석(annotation) 정보(예: normal map, 세분화(segmentation) 맵, 2D/3D 바운딩 박스)를 생성합니다. MultiFloodSynth는 사용자가 원하는 구조를 설정하고 조정할 수 있는 유연성을 제공하여 최종 가상 장면을 구성할 수 있게 합니다.

- **Performance Highlights**: MultiFloodSynth는 총 70,117장의 이미지를 생성하였으며, 그 중 14,593장이 홍수 장면에 해당합니다. 실제 데이터와 비교했을 때, YOLOv10 모델을 사용한 홍수 감지 성능이 높아졌으며, 향상된 정확성을 보여줍니다. 또한 다양한 주석 형태를 활용하여 다양한 컴퓨터 비전 작업에 대한 응용 가능성을 높였습니다.



### Quantum Circuit Design using a Progressive Widening Monte Carlo Tree Search (https://arxiv.org/abs/2502.03962)
- **What's New**: 이 논문에서는 Variational Quantum Algorithms (VQAs)에서 중요한 문제를 해결하기 위해 Gradient-free Monte Carlo Tree Search (MCTS) 기법을 제안합니다. 기존의 VQAs의 주요 과제인 문제에 맞춤화된 양자 회로 설계를 자동화하는 접근법으로, 새로운 샘플링 기법과 점진적 확장을 통해 동적으로 공간을 탐색할 수 있습니다. 또한, 제안하는 방법이 다양한 응용 분야에서 강력성을 보여줍니다.

- **Technical Details**: PWMCTS는 고전적인 MCTS의 약점을 극복하면서, 문제 구조에 대한 사전 지식 없이도 양자 회로를 설계할 수 있도록 고안되었습니다. 이 접근법은 PQCs를 위한 구조화된 무한 이산 공간으로 정의된 샘플링 기법을 포함하고 있으며, 이전 연구와 비교해 더 얕은 양자 회로로 동일하거나 우수한 결과를 보여줍니다. 이를 통해 CNOT 게이트 수가 최대 세 배 적은 회로를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, PWMCTS는 무Stru_cted 양자 회로를 설계하는 능력에서 무작위 양자 회로를 근사할 수 있으며, 강화된 수치적 자원의 사용을 달성하였다. 이는 기존 MCTS 연구와 비교할 때 양자 회로 평가 회수를 10에서 100배 줄이면서 동등하거나 더 나은 결과를 이루었다는 것을 의미합니다. 전체적으로 PWMCTS는 성능과 자원 효율성을 올릴 수 있는 혁신적인 방법을 제시합니다.



### Improving the Perturbation-Based Explanation of Deepfake Detectors Through the Use of Adversarially-Generated Samples (https://arxiv.org/abs/2502.03957)
Comments:
          Accepted for publication, AI4MFDD Workshop @ IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025), Tucson, AZ, USA, Feb. 2025. This is the authors' "accepted version"

- **What's New**: 이번 논문에서는 감지된 딥페이크로 분류된 입력 이미지의 적대적으로 생성된 샘플을 사용하여, 다양한 입력 특성의 중요성을 추론하고 시각적 설명을 생성하는 새로운 접근 방식을 제안합니다. 이 과정은 Natural Evolution Strategies(NES)를 기반으로 하며, 초기 딥페이크 탐지기의 결정을 반전시키는 샘플을 생성하는 데 초점을 맞추고 있습니다. 이렇게 생성된 샘플을 통해 기존의 설명 방법을 개선하고 보다 신뢰할 수 있는 설명을 제공합니다.

- **Technical Details**: 제안된 접근법은 적대적으로 생성된 샘플을 사용하여 perturbation masks를 형성하며, 이는 특정 입력 특성의 중요성을 추론하는 데 도움을 줍니다. 네 가지 SOTA 설명 방법인 RISE, SHAP, LIME, SOBOL을 수정하여 이 새로운 perturbation 방식을 통합하고 데이터베이스로는 FaceForensics++를 활용하여 성능을 평가합니다. 실험 과정에서 적은 양의 노이즈만을 추가하여 시각적으로 유사한 샘플을 생성하며, OOD(out-of-distribution) 문제를 회피합니다.

- **Performance Highlights**: 수량적 및 정성적 평가를 통해 제안된 perturbation 접근 방식이 기존 설명 방법의 성능에 긍정적인 기여를 했음을 입증하였습니다. 특히, 수정된 설명 방법이 조작된 이미지 영역을 더 정확하게 구별하는 능력을 갖추고 있음을 보여줍니다. 이를 통해, 딥페이크 탐지기에서의 결정에 대한 보다 유용하고 의미 있는 시각적 설명을 제공합니다.



### MAQInstruct: Instruction-based Unified Event Relation Extraction (https://arxiv.org/abs/2502.03954)
Comments:
          Accepted by WWW 2025 short

- **What's New**: 이 논문에서는 MAQInstruct라는 개선된 Instruction-based Event Relation Extraction 프레임워크를 제안합니다. 기존의 방법들이 다중 클래스 분류, MASK 예측, 또는 프로토타입 매칭에 기반해 있었던 것과 달리, MAQInstruct는 주어진 event-relation 지침을 사용하여 이벤트를 선택하는 방식으로 작업을 전환합니다. 이러한 접근은 추론에 필요한 샘플 수를 대폭 줄이면서 성능을 향상시키는 데 기여합니다.

- **Technical Details**: MAQInstruct 프레임워크는 bipartite matching loss를 포함하여 기존의 instruction-based 방법의 생성 순서에 대한 의존성을 줄이는 데 초점을 맞추고 있습니다. 흥미롭게도, 이 프레임워크는 순서의 변화가 이벤트 관계에 미치는 영향을 최소화하여 여러 대규모 언어 모델(LLMs)에서의 성능을 향상시킵니다. 구체적으로, 모델은 이벤트 관계 유형이 event mentions의 수보다 현저히 적기 때문에 훈련 및 추론 샘플을 k×n으로 감소시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 MAQInstruct는 다양한 LLM에서 이벤트 관계 추출 작업을 획기적으로 개선하고 있습니다. 특히, 이 방법은 대규모 언어 모델의 능력을 활용하여 이전의 분류 기반 방법을 초월하는 성능을 보여주었습니다. 또한, bipartite matching loss를 통해 사건 관계 추출 작업을 수행할 때 적절성을 높이고 올바른 답변을 더 잘 생성하도록 돕습니다.



### DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation (https://arxiv.org/abs/2502.03930)
Comments:
          16 pages, 8 figures

- **What's New**: 이번 연구에서는 Diffusion Transformer Autoregressive Modeling (DiTAR)이라는 패치 기반의 오토회귀(autoregressive) 프레임워크를 제안하며, 언어 모델(language model)과 확산 변환기(diffusion transformer)를 결합합니다. 이를 통해 연속적인 매개변수 생성을 위한 효율성을 크게 향상시키고, 계산 요구 사항을 줄이는 효과를 보여줍니다. 특히, DiTAR는 패치 생성을 위한 분할 정복 전략을 활용하여, 언어 모델이 집계된 패치 임베딩을 처리하고, 확산 변환기가 다음 패치를 생성할 수 있도록 합니다.

- **Technical Details**: DiTAR는 연속 토큰의 예측을 위해 causal attention과 bidirectional attention의 강점을 결합한 오토회귀 모델입니다. 이를 통해 DiTAR는 패치를 여러 개로 나누고, 언어 모델이 패치 간 예측을 담당하며, 확산 변환기가 패치 내 예측을 수행하며 효율성을 높이고 있습니다. 새로운 온도(temperature) 정의를 통해 역확산 ODE에서 노이즈의 도입 시점을 정하고, 이를 활용한 샘플링 기법을 제안하여 연속값 언어 모델의 탐색과 활용을 조율합니다.

- **Performance Highlights**: DiTAR는 제로샷 텍스트 투 스피치(zero-shot text-to-speech) 작업에서 뛰어난 성능을 발휘하며, 특히 강인성(robustness), 화자 유사성(speaker similarity), 자연스러움(naturalness) 분야에서 최첨단(SOTA) 성과를 기록했습니다. 이러한 성과는 기존의 모델들이 요구하는 계산량보다 현저히 낮은 비용으로 이루어졌습니다. 특히, 여러 단계를 거치는 복잡한 파이프라인 대신, DiTAR는 언어 모델이 최종 특성을 직접 예측하는 단순화된 방식을 적용하여 우수한 결과를 도출했습니다.



### Adaptation of Task Goal States from Prior Knowledg (https://arxiv.org/abs/2502.03918)
- **What's New**: 이번 논문은 로봇이 관찰된 작업을 기반으로 목표 상태를 자유롭게 설정할 수 있는 프레임워크를 제안합니다. 이는 로봇이 동일한 작업 설명에서 관찰된 것과 달리 더 쉬운 목표를 설정할 수 있게 해줍니다. 고정된 상태가 아닌 값의 범위로 작업 목표 상태를 표현함으로써 로봇이 현실 세계를 더 잘 이해할 수 있도록 합니다.

- **Technical Details**: 작업 목표 상태는 에이전트와 객체 속성의 값 범위로 정의됩니다. 이를 위해 환경 상태 모델과 변형을 정의하고, 단일 작업 시연으로부터 변형을 생성하는 상호작용 방법을 제안합니다. 이러한 구조적 환경 정의를 통해 작업 목표 상태로 변환하기 위한 기술(Skills)을 결정하는 데 중요해집니다.

- **Performance Highlights**: 기존의 기하학적 작업 모델을 확장하여 로봇이 더 다양한 상황에서 유연하게 작업을 수행할 수 있도록 지원하며, 특히 가정 환경과 같은 비조립 작업에서의 여러 가능성을 모델링합니다. 논문에서 제안하는 프레임워크는 로봇이 특정 목표 상태로 환경을 가져오는 데 필요한 단계들을 구성하는 데 유용할 것으로 기대됩니다.



### Experiments with Large Language Models on Retrieval-Augmented Generation for Closed-Source Simulation Softwar (https://arxiv.org/abs/2502.03916)
Comments:
          11 pages, 6 tables

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 접근법을 통해 클로즈드 소스 시뮬레이션 소프트웨어에 LLM(대형 언어 모델)의 응용 가능성을 탐구합니다. RAG 시스템을 통해 LLM의 응답 작성 시 사용자 요청에 따라 관련 문서를 제공하여, 지식 집약적인 작업에 필요한 정보를 지원할 수 있습니다. 이는 기존의 오픈소스 데이터에 교육받지 않은 클로즈드 소스 환경에서도 LLM의 능력을 확장할 가능성을 보여줍니다.

- **Technical Details**: 스무디드 입자 유체역학(SPH)과 같은 시뮬레이션 방법론을 사용하여 LLM의 지식 수준을 테스트합니다. 연구는 Pasimodo라는 클로즈드 소스 소프트웨어에 대한 LLM의 이해도를 평가하며, 관련 내부 지식을 제공한 후 RAG 시스템을 통해 모델을 생성하는 방법을 다룹니다. 이 과정에서 RAG는 LLM이 처리할 수 있는 정보의 품질을 향상시키고, 정보 격차를 줄이는 데 도움이 되는 것으로 나타났습니다.

- **Performance Highlights**: 제시된 초기 실험 결과는 RAG 시스템이 클로즈드 소스 시뮬레이션 소프트웨어에 효과적으로 정보를 접근할 수 있도록 돕는 가능성을 보여줍니다. 다수의 실험 예제를 통해 LLM의 응답에서 나타난 정보 격차와 데이터 불완전성을 발견하였으며, 이러한 결과는 향후 추가 연구의 필요성을 강조합니다. 기능적으로, RAG 접근법은 LLM의 환각(hallucination) 위험을 감소시키는 잠재력을 보입니다.



### UniForm: A Unified Diffusion Transformer for Audio-Video Generation (https://arxiv.org/abs/2502.03897)
- **What's New**: 이번 논문에서는 공통의 잠재 공간에서 오디오와 비디오를 동시에 생성하는 UniForm이라는 통합 확산 변환기(Unified Diffusion Transformer)를 소개합니다. 기존의 오디오-비디오 생성 방식은 두 개의 독립적인 모듈에 의존하여 자연스러운 상관 관계를 충분히 활용하지 못했습니다. UniForm은 시각 및 청각 정보를 결합하여 고품질의 오디오-비디오 쌍을 생성하도록 학습합니다.

- **Technical Details**: UniForm은 통합된 잠재 공간에서 오디오와 비디오의 동시 생성을 가능하게 하는 단일 확산 프레임워크를 채택하고 있습니다. 이를 통해 모델은 세 가지 생성 작업: 텍스트-오디블 비디오 생성(T2AV), 오디오-비디오 생성(A2V), 및 비디오-오디오 생성(V2A)을 지원합니다. 각 작업에 대해 텍스트 프롬프트를 추가하여 세밀한 제어 기능을 더해 발휘할 수 있는 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, UniForm은 최신 단일 작업 기준들과 비교하여 동등한 성능을 달성했습니다. 특히, задачи는 특정 작업 데이터셋에 대한 미세 조정 없이 이루어졌으며, 다중 작업 시스템으로 훈련된 성능이 눈에 띕니다. 이 실험들은 오디오와 비디오 간의 정합성과 일관성을 획기적으로 향상시키는 UniForm의 강점을 증명합니다.



### Rank Also Matters: Hierarchical Configuration for Mixture of Adapter Experts in LLM Fine-Tuning (https://arxiv.org/abs/2502.03884)
- **What's New**: 본 논문에서는 Hierarchical scheme for expert allocation and rank configuration (HILO)를 제안하여, adapter experts의 수와 rank를 동적으로 조절하며 모델 레이어의 복잡성에 맞게 조정하는 방법을 다룹니다. 기존 연구들이 adapter experts의 수에만 초점을 맞춘 데 반해, HILO는 rank 설정의 중요성을 강조하고, trainable parameter를 줄이면서도 모델 정확도를 향상시키는 새로운 접근법을 제공합니다.

- **Technical Details**: HILO는 adapter experts 구성의 효율성을 높이는 데 중점을 두며, 각 레이어에서 adapter experts의 수와 rank를 할당하여 구성합니다. 이 방식은 각 레이어의 representational complexity에 따라 동적으로 조정되며, 이를 통해 기존 방법들보다 높아진 정확도를 달성할 수 있습니다. LoRA와 Mixture of Experts(MoE)를 결합한 기존 접근 방식의 한계를 극복하기 위해, HILO는 더 많은 구성 요소를 고려합니다.

- **Performance Highlights**: 다양한 벤치마크 작업에 대한 실험 결과, HILO는 기존 방법들보다 높은 정확도를 기록하면서도 더 적은 수의 trainable parameters를 도입합니다. 이로 인해 LLMs의 fine-tuning 및 inference 과정에서 효율적인 솔루션을 제공합니다. HILO의 도입으로 인해 연구자들은 앞으로 좀 더 최적화된 PEFT 기법을 기대할 수 있습니다.



### Pursuing Better Decision Boundaries for Long-Tailed Object Detection via Category Information Amoun (https://arxiv.org/abs/2502.03852)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: 이 논문에서는 객체 탐지(object detection)에서 카테고리 정보량(category information amount) 개념을 도입하고 이를 측정하는 방법을 제안합니다. 이 연구는 모델이 인스턴스 수가 비교적 균형 잡힌 데이터셋에서도 카테고리 편향(category bias)을 나타냄을 보여 주며, 단순히 인스턴스 수만으로는 이러한 현상을 설명할 수 없음을 시사합니다. 따라서 카테고리 정보량이 각 카테고리의 학습 난이도를 더 잘 반영한다고 주장합니다.

- **Technical Details**: 제안된 정보량 기반 각도 여유(Information Amount-Guided Angular Margin, IGAM) 손실 함수는 카테고리의 정보량에 따라 각 카테고리의 결정 공간을 동적으로 조정하여 장기간의 데이터셋에서 카테고리 편향을 줄이는 것을 목표로 합니다. IGAM Loss는 저비용의 엔드 투 엔드 훈련 전략을 통해 동적으로 정보량을 업데이트할 수 있습니다. 실험 결과는 LVIS v1.0 및 COCO-LT와 같은 장기간 테스트 데이터셋에서 IGAM 방법이 우수한 성과를 나타내고 있음을 보여줍니다.

- **Performance Highlights**: IGAM Loss를 적용한 결과, 기존의 많은 방법들을 초월하여 장기 데이터셋에서 성능을 개선했습니다. 특히, 희소 카테고리에 대한 모델의 정확성을 크게 향상시켰습니다. 상대적으로 균형 잡힌 Pascal VOC 데이터셋에서도 우리의 방법은 도전적인 카테고리에서 다른 접근법보다 현저히 우수한 성과를 보였습니다.



### Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis (https://arxiv.org/abs/2502.03843)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 정렬을 위한 고품질의 대규모 지침이 얼마나 중요한지를 강조합니다. 기존의 자연어 이해(NLU) 지침 구축 작업은 정보 추출(IE) 중심으로 진행되어, 기계 독해, 질문 응답, 텍스트 분류와 같은 과제를 간과했습니다. 이로 인해 NLU 데이터의 다양성이 떨어지고 훈련된 LLM의 일반화 능력이 감소하는 문제가 발생했습니다. 이를 해결하기 위해, 'Hum'이라는 고품질의 합성 지침 데이터셋을 제안합니다.

- **Technical Details**: Hum은 정보 추출(IE), 기계 독해, 텍스트 분류, 그리고 지침 일반화 과제를 포함하여 다양한 NLU 작업을 위한 지침을 수집하도록 설계되었습니다. 이 데이터셋은 인간-LLMs 협업 메커니즘을 통해 지침 다양성을 더욱 풍부하게 만드는 방식으로 지침을 합성합니다. 연구는 5개의 NLU 작업과 28개의 LLM 일반 능력 평가 데이터셋에서 수행되어, Hum의 효과를 입증했습니다.

- **Performance Highlights**: 실험 결과, Hum은 여섯 개의 LLM의 NLU 능력을 평균 3.1% 향상시키는데 기여했습니다. 다른 일반 능력에는 유의미한 감소가 없었으며, 이는 Hum의 효과적인 지침 차원에서 각 작업의 다양성을 크게 증가시켰음을 보여줍니다. 이 연구는 NLU 작업의 품질 향상을 위한 중요한 진전을 의미합니다.



### A comprehensive survey of contemporary Arabic sentiment analysis: Methods, Challenges, and Future Directions (https://arxiv.org/abs/2502.03827)
Comments:
          Paper accepted to NAACL 2025

- **What's New**: 이 논문은 아랍어 감정 분석(Arabic Sentiment Analysis, ASA)에 대한 최신 연구 동향을 체계적으로 정리하며, 기존 문헌의 한계와 과제를 규명합니다. 특히 딥 러닝(Deep Learning)을 활용한 방법론에 중점을 두고 아랍어 감정 분석의 연구 격차를 일반 감정 분석과 비교하여 강조하고 있습니다. 또한, 향후 연구를 위한 세부적인 방향성을 제시합니다.

- **Technical Details**: 논문에서는 아랍어 감정 분석의 발전 과정을 전통적인 렉시콘 기반 방법(Lexicon-based methods)에서 딥 러닝 기반 방법(Deep Learning-based methods)으로 설명합니다. 또한, 감정 점수(Sentiment Scores)와 단어의 의미를 제공하는 패턴을 학습할 수 있는 기계 학습 방법(Machine Learning Methods)의 소개가 이루어지며, 다양한 피처 엔지니어링(Feature Engineering) 방법이 언급됩니다. 이 섹션에서는 감정 분석 모델의 적용에 있어 아랍어 렉시콘이 중요한 역할을 할 수 있음을 설명합니다.

- **Performance Highlights**: 아랍어 감정 분석 모델의 성능 향상에는 다양한 요소들이 기여합니다. 아랍어 렉시콘을 활용한 데이터 전처리와 감정 가중치 조정이 중요하게 다루어졌으며, 이는 특히 저자원 환경에서 모델 성능을 향상시키는 데 효과적입니다. 실제 사례로 딥 러닝 모델에 렉시콘을 통합하여 성능을 개선한 연구가 소개되며, 복잡한 모델의 해석 가능성(Interpretability) 또한 향상되는 결과가 나타났습니다.



### Syntriever: How to Train Your Retriever with Synthetic Data from LLMs (https://arxiv.org/abs/2502.03824)
Comments:
          the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted

- **What's New**: 이 논문에서는 Syntriever라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 최신 블랙박스 LLMs의 합성 데이터를 활용하여 리트리버 모델을 훈련 및 미세조정할 수 있도록 설계되었습니다. Syntriever는 두 단계로 구성되어 있으며, 첫 번째 단계에서는 LLM 생성 합성 데이터를 사용하여 리트리버를 훈련하고, 두 번째 단계에서는 LLM의 선호도와 리트리버를 정렬합니다.

- **Technical Details**: Syntriever는 LLM의 지식을 효과적으로 추출하는 데 필요한 두 가지 주요 단계를 포함합니다. 첫 번째 단계인 distillation 단계에서는 chain-of-thoughts를 이용해 합성 쿼리와 관련된 문서를 생성하고 LLM이 자체 검증을 통해 환각 가능성을 최소화합니다. 두 번째 단계인 alignment 단계에서는 리트리버를 LLM의 선호도에 맞추기 위해 partial Plackett-Luce ranking 방법을 사용하여 훈련합니다.

- **Performance Highlights**: Syntriever는 다양한 도메인의 벤치마크 데이터셋에서 모든 성능 기준을 크게 초과하는 결과를 보였습니다. 특히, nDCG@10 기준으로 이전의 최고 성능보다 최대 18.6% 향상이 있었습니다. 이 프레임워크는 다양한 기본 리트리버 및 LLM와 결합이 가능하여 검색 정확도 증가를 가져옵니다.



### Large Language Models for Multi-Robot Systems: A Survey (https://arxiv.org/abs/2502.03814)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)과 다중 로봇 시스템(MRS)의 통합에 대해 처음으로 포괄적으로 탐구합니다. 이 연구는 LLMs가 MRS에서 고수준 작업 할당, 중수준 동작 계획, 저수준 행동 생성 및 인간 개입 등 다양한 분야에서의 응용 프로그램을 체계적으로 분류합니다. 또한, 현대 로봇의 지능을 향상시킬 수 있는 기술적 기회를 강조하며, LLM의 고급 문제 이해 및 해결 능력을 활용한 새로운 가능성을 제시합니다.

- **Technical Details**: MRS는 여러 대의 자율 로봇이 협력하여 특정 임무를 수행하는 시스템으로, 집단 지성을 활용해 높은 효율성과 신뢰성을 유지합니다. 본 섹션에서는 MRS와 LLM의 기본 개념을 설명하고, LLM이 MRS에서 어떻게 응용될 수 있는지에 대한 배경 지식을 제공합니다. 또한 중앙 집중형 제어 및 분산 제어라는 두 가지 주요 제어 방식이 MRS 내 상호작용 및 작업 분배를 관리하는 데 사용됩니다.

- **Performance Highlights**: MRS는 환경 모니터링, 창고 자동화, 탐사 등 다양한 분야에서 응용 가능성이 큽니다. LLMs를 MRS에 통합함으로써 다중 로봇 간의 통신 및 유연성을 증진시킬 수 있으며, 이는 실제 환경에서의 복잡한 명령 수행에 도움을 줄 것입니다. 이러한 기술들은 LLMs의 높은 일반화 능력을 활용하여 새로운 시나리오에서도 쉽게 적응할 수 있도록 만들어, MRS의 전체 효율성을 증가시킵니다.



### Understanding and Supporting Formal Email Exchange by Answering AI-Generated Questions (https://arxiv.org/abs/2502.03804)
- **What's New**: 이 논문은 LLM 기반 질문 및 답변(QA) 접근 방식이 이메일 회신 프로세스를 개선할 수 있음을 제안합니다. 사용자가 들어오는 이메일에서 생성된 간단한 질문에 응답하여 이메일에 회신할 수 있도록 도와주는 시스템 ResQ를 개발했습니다. 이 시스템은 기존의 프롬프트 기반 접근 방식에 비해 효율성과 작업 부담을 줄이는 데 효과적임을 입증했습니다.

- **Technical Details**: 연구에서는 LLM을 활용하여 이메일의 내용을 분석하고 사용자가 대답할 수 있는 질문을 생성합니다. 사용자의 질문에 대한 답변을 기반으로 이메일 초안이 작성됩니다. QA 기반 접근 방식은 사용자에게 필요한 요구를 효과적으로 명확히 하는 데 도움이 되는 구조화된 질문에 대한 답변을 요구하는 이전 연구에 기초하고 있습니다.

- **Performance Highlights**: 실험 결과, ResQ는 효율적인 이메일 회신을 촉진하고 작업 부담을 줄이며 이메일 품질을 유지하는 데 기여했습니다. 또한, 사용자는 시스템이 더 나은 커뮤니케이션 품질과 양을 제공한다고 인식함으로써 심리적 거리감이 줄어든다고 보고했습니다. 두 가지 연구 결과는 QA 기반 접근 방식이 사용자와 수신자 간의 관계를 포함한 이메일 작성 과정에 긍정적인 영향을 미친다는 것을 보여줍니다.



### SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning (https://arxiv.org/abs/2502.03801)
- **What's New**: 본 논문은 federated learning (FL)의 데이터 프라이버시를 유지하면서 협력적 모델 훈련을 가능하게 하지만, 클라이언트 측 데이터 오염 공격(client-side data poisoning attacks, DPAs)과 모델 오염 공격(model poisoning attacks, MPAs)에 취약하다는 점을 강조합니다. 다양한 방어 전략들이 제안되었으나, 이들의 평가가 제한적인 공격 방법으로 이루어져 효과성에 대한 우려가 있었습니다. 본 연구는 DPAs와 MPAs에 대한 방어 수단을 통합적으로 분석하고, 이들 두 영역 간의 차이를 명확히 하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 poisoning 공격과 방어 전략에 대한 체계적인 분류(taxonomy)를 제시하며, 각 기법의 설계와 강점, 한계를 설명합니다. 또한 다양한 FL 알고리즘과 데이터 이질성(data heterogeneity)에 대한 통합 비교 평가를 수행하여 개별 및 상호 효과성(mutual effectiveness)을 검증하고 향후 연구를 위한 주요 통찰(insights)을 도출했습니다. 이를 통해 FLPoison이라는 통합 벤치마크를 만들어 15개의 대표적인 오염 공격과 17개의 방어 전략을 평가할 수 있도록 하여, 향후 연구에 기여하고자 합니다.

- **Performance Highlights**: 제안된 연구는 방어 전략이 DPAs와 MPAs에 대해 상호 작용할 수 있는 방안을 제시하며, 각 방어의 성능을 객관적으로 비교함으로써 방어 메커니즘의 발전 방향에 대한 귀중한 통찰을 제공합니다. 또한, FLPoison은 높은 모듈성(modularity)과 확장성(scalability)을 지원하여 다양한 공격 시나리오를 평가하는 데 효과적인 자료를 제공합니다. 코드 또한 공개되어 있어 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers (https://arxiv.org/abs/2502.03793)
- **What's New**: 이 연구에서는 전통적인 태스크 전용 분류 헤드에 의존하지 않고 현대적인 encoder-only 모델인 ModernBERT-Large-Instruct를 도입합니다. 이 모델은 마스킹된 언어 모델링(MLM) 헤드를 활용하여 제너레이티브(classification를 위한 생성적) 기능을 발휘합니다. 특히, 이 모델은 기존의 LLM(large language models)보다 60% 더 적은 매개변수로 MMLU에서 Llama3-1B의 93% 성능을 달성했습니다.

- **Technical Details**: 이 연구에서 사용된 ModernBERT-Large-Instruct 모델은 단순한 훈련 루프와 추론 메커니즘을 채택하여 복잡한 전처리나 엔지니어링된 프롬프트 없이도 강력한 제로샷(Zero-shot) 성능을 보여줍니다. 대량의 현대화된 데이터 믹스를 통해 훈련된 이 모델은 다양한 NLU(자연어 이해) 작업에서 파인튜닝(fine-tuned) 과정 후에도 강력한 성능을 나타낸다고 밝혔습니다. 기존의 encoder-only 모델들에 비해 낮은 오버헤드를 지니면서도 효과적입니다.

- **Performance Highlights**: 제로샷 및 파인튜닝 설정 모두에서 ModernBERT-Large-Instruct는 이전 접근 방식과 경쟁력을 보여줍니다. 특히 뉴스 주제 감지, 텍스트 일관성, 포럼 게시물 주제 식별에서 기존의 분류 헤드 메소드와 어깨를 나란히 하거나 성능을 능가하는 결과를 보였습니다. 이러한 결과는 현대 architecture에 기반한 모델의 중요성과 다각적 데이터 믹스의 가치를 강조합니다.



### ExpProof : Operationalizing Explanations for Confidential Models with ZKPs (https://arxiv.org/abs/2502.03773)
- **What's New**: 이 논문에서는 기계 학습 모델에 대한 설명 가능성을 적대적인 시나리오에서 실용적으로 구현하기 위한 Zero-Knowledge Proofs (ZKPs)를 사용한 새로운 시스템인 ExpProof을 제안합니다. 이 시스템은 설명이 올바르게 계산되었음을 보장하며, 모델의 기밀성을 유지합니다. ExpProof은 설명 관련 알고리즘 LIME의 개선된 버전을 탐구하고, 신경망 및 랜덤 포레스트에서의 성능을 평가합니다.

- **Technical Details**: ExpProof의 핵심 구성 요소는 암호학적 커밋먼트와 Zero-Knowledge Proofs입니다. 커밋먼트 방법은 모델의 가중치와 설명 매개변수를 고정하여 기밀성을 유지하면서도 투명성을 제공합니다. ZKP를 통해 설명이 사전에 정의된 알고리즘을 사용하여 올바르게 계산되었음을 제안자(은행)가 증명할 수 있으며, 검증자는 추가 정보 없이도 이를 확인할 수 있습니다.

- **Performance Highlights**: ExpProof의 실험 결과는 계산적으로 실행 가능하며, 최대 증명 생성 시간은 1.5분, 검증 시간은 0.12초, 증명 크기는 13KB로 나타났습니다. 이는 신경망 및 LIME을 사용한 상용 기계 학습 모델에서의 성능을 보여줍니다. 이 시스템은 기밀성을 유지하면서도 설명 가능성 측면에서 높은 신뢰성을 제공합니다.



### A Retrospective Systematic Study on Hierarchical Sparse Query Transformer-assisted Ultrasound Screening for Early Hepatocellular Carcinoma (https://arxiv.org/abs/2502.03772)
- **What's New**: 이 연구에서는 인공지능(AI)의 최신 발전을 활용하여 간세포암(HCC) 조기 진단의 정확도를 높이기 위한 혁신적인 모델인 계층 스파스 쿼리 변환기(Hierarchical Sparse Query Transformer, HSQformer)를 제안합니다. 이 모델은 합성곱 신경망(Convolutional Neural Networks, CNNs)과 비전 변환기(Vision Transformers, ViTs)의 장점을 통합하여 초음파 검사의 정확성을 강화합니다. 기존의 컴퓨터 보조 진단 시스템의 한계를 극복하고, 효율성과 사용 편의성을 높인 모듈형 아키텍처를 채택했습니다.

- **Technical Details**: HSQformer는 CVN과 ViT의 두 가지 주요 기능 추출기를 기반으로 하여 설계되었습니다. 이 모델은 교차 주의(Cross-attention), 자기 주의(Self-attention), 전문가 혼합(Mixture of experts, MoE) 모듈을 스태킹하여 AI의 진단 지원 가능성을 탐구합니다. 이를 통해 많게는 저수준 세부 정보와 패턴도 효과적으로 캡처하여 정밀한 진단이 가능하도록 합니다.

- **Performance Highlights**: HSQformer는 단일 센터, 다중 센터, 고위험 환자 테스트와 같은 세 가지 임상 시나리오에서 성능 테스트를 수행했으며, 기존의 최첨단 모델인 ConvNext와 SwinTransformer를 꾸준히 초월하는 결과를 보였습니다. 특히 HSQformer는 고급 방사선의사와 진단 정확도가 같았고, 저급 방사선의사들보다 월등한 성과를 기록하여 AI 도구의 임상적 잠재력을 명확히 보여주었습니다.



### PRISM: A Robust Framework for Skill-based Meta-Reinforcement Learning with Noisy Demonstrations (https://arxiv.org/abs/2502.03752)
Comments:
          8 pages main, 19 pages appendix with reference. Submitted to ICML 2025

- **What's New**: 이 논문은 Skill-Based Meta-Reinforcement Learning (Meta-RL)에 대한 새로운 접근법인 Prioritized Refinement for Skill-Based Meta-RL (PRISM)을 제안합니다. PRISM은 noisy offline data 근처에서 탐색을 통해 온라인 궤적을 생성하고 이를 offline 데이터와 결합함으로써 효과적인 스킬 학습을 보장합니다. 이 방법은 노이즈의 영향을 해결하여 스킬 학습의 안정성을 확보하고 긴 수명의 과제를 해결하는 데 우수한 성능을 발휘합니다.

- **Technical Details**: PRISM은 exploration policy를 활용하여 noisy offline data 근처에서 유용한 궤적을 발견, 이를 통해 고품질 데이터를 추출하여 task-relevant skills를 학습합니다. 두 가지 주요 기여를 포함하는 이 프레임워크는 (1) 우선 순위가 매겨진 스킬 정제 프레임워크로, online과 offline 데이터셋에서 스킬을 통합된 방식으로 학습할 수 있도록 합니다. (2) 최대 반환 재라벨링 기법을 통해 noisy offline 궤적을 평가하며, 이를 통해 데이터 품질을 보장합니다.

- **Performance Highlights**: PRISM은 noisy 환경에서도 데이터로부터 학습의 안정성을 확보하여 효과적인 스킬 학습을 달성합니다. 특히, Maze2D 환경에서 noisy offline 궤적에서 스킬을 성공적으로 정제하여 보이지 않는 과제를 해결하는 데 성공했습니다. 이러한 접근 방식은 스킬 기반 Meta-RL의 강건성과 일반화 가능성을 크게 향상시켜 실제 noisy 시나리오에서도 신뢰할 수 있는 성과를 입증합니다.



### Principal Curvatures Estimation with Applications to Single Cell Data (https://arxiv.org/abs/2502.03750)
Comments:
          To be published in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)

- **What's New**: 이 논문에서는 단일 세포 전사체 시퀀싱(single-cell transcriptomic sequencing, scRNAseq) 데이터를 분석하기 위한 데이터 기반 방법인 Adaptive Local PCA (AdaL-PCA)를 제안합니다. AdaL-PCA는 데이터 매니폴드(data manifold)에서의 본질 곡률(intrinsic curvature) 추정을 통해 세포의 차별화(cell differentiation) 과정을 밝혀낼 수 있는 가능성을 확인합니다. 기존 방법에 비해 데이터 밀도 변화에 대한 적응성을 강화하여 다양한 매니폴드에서 안정성을 보장합니다.

- **Technical Details**: 이 방법은 Locally PCA를 바탕으로 하여, 각 포인트 주위의 이웃을 선택하고 이들을 중심으로 하는 데이터 행렬을 형성합니다. 여기서 첫 두 개의 고유벡터는 국소적인 접선 공간(tangent space)의 기준으로 선택되며, 세 번째 고유벡터는 표면의 법선 벡터(normal vector) 역할을 합니다. AdaL-PCA는 설명된 분산 비율을 사용하여 적절한 매개변수를 동적으로 조정하여 곡률(curvature)을 추정합니다.

- **Performance Highlights**: AdaL-PCA는 샘플링 된 표면(surfaces)에서 최첨단 성능을 입증하였으며, PHATE (Potential of Heat-diffusion for Affinity-based Trustworthiness Embeddings) 내 장을 결합하여 단일 세포 RNA 시퀀스 데이터에서 핵심 변화를 식별할 수 있습니다. 이를 통해 세포의 차별화 방향성을 제시하는 주 성질(curvature)을 추정할 수 있었습니다.



### Multiple Invertible and Partial-Equivariant Function for Latent Vector Transformation to Enhance Disentanglement in VAEs (https://arxiv.org/abs/2502.03740)
Comments:
          24 pages, 21 figures

- **What's New**: 본 논문에서는 Variational AutoEncoder(VAE)에서 훈련된 정보를 이해하고 재사용하기 위한 핵심 문제인 분리 학습(disentanglement learning)을 다룹니다. 특히, inductive bias를 주입하기 위한 새로운 방법인 Multiple Invertible and partial-equivariant transformation(MIPE-transformation)을 제안하며, 이 방법은 입력에서 잠재 공간으로의 변환에서 일정 부분의 equivariance를 유지하면서도 잠재 벡터 변환의 invertibility를 보장합니다.

- **Technical Details**: MIPE-transformation의 두 가지 주요 구성 요소는 Invertible and partial-equivariant transformation(IPE-transformation)과 Exponential Family conversion(EF-conversion)입니다. IPE-transformation은 잠재 벡터 간의 변환을 대칭 행렬 지수 함수로 제한하여 부분-equivariant하게 구현하며, EF-conversion은 불확실한 잠재 벡터 분포를 학습 가능한 형태로 전환합니다. 또한, 여러 IPE-transformation과 EF-conversion을 통합하는 아키텍처를 제안합니다.

- **Performance Highlights**: MIPE-transformation은 3D Cars, 3D Shapes, dSprites와 같은 다양한 데이터세트에서 기존 VAE의 분리 성능을 현저히 향상시켰습니다. 귀무가설을 통해 IPE-transformation과 EF-conversion의 결합이 분리된 표현 학습에 미친 긍정적인 영향을 실증적으로 분석하였습니다. 본 연구는 분리 학습을 개선하기 위한 효과적인 방법론을 제시하며, 다양한 최신 VAE에 폭넓게 적용될 수 있는 가능성을 보여줍니다.



### Action-Free Reasoning for Policy Generalization (https://arxiv.org/abs/2502.03729)
Comments:
          13 pages, 10 figures

- **What's New**: 이 연구는 로봇 정책 훈련을 위한 새로운 접근법인 Reasoning through Action-free Data (RAD)를 제안합니다. RAD는 로봇 데모 데이터와 비디오에서 추출한 행동 없는 데이터에서 언어 기반 추론을 활용하여 일반화 가능한 로봇 정책을 학습합니다. 기존의 방법론과 달리 RAD는 사람들이 작업을 수행할 때의 높은 수준의 논리적 추론 정보를 사용하여 로봇 행동 예측을 지원합니다.

- **Technical Details**: RAD는 로봇 데모 데이터와 언어적 추론만 있는 행동 없는 데이터의 혼합을 사용하여 대규모 트랜스포머 모델을 훈련합니다. 로봇 데이터는 모델에게 언어적 추론을 저수준 행동으로 매핑하는 방법을 학습시키고, 행동 없는 데이터는 추론 능력을 향상시킵니다. 또한, 사전 훈련된 비전-언어 모델을 활용하여 추론 체인을 주석을 달아 정보에 대한 풍부한 이해를 제공합니다.

- **Performance Highlights**: 실험 결과, RAD는 행동 없는 데이터에서만 관찰된 작업에서 20% 더 나은 성능을 보여주며, 큰 양의 행동 없는 추론 데이터는 로봇의 일반화 능력을 크게 향상시키는 것으로 나타났습니다. 이러한 결과는 행동 없는 데이터에서의 추론 주도의 학습이 일반화 가능한 로봇 제어에 대한 큰 잠재력을 지니고 있음을 강조합니다.



### MD-BERT: Action Recognition in Dark Videos via Dynamic Multi-Stream Fusion and Temporal Modeling (https://arxiv.org/abs/2502.03724)
- **What's New**: 이 연구는 저조도 비디오에서의 동작 인식을 위한 새로운 접근법인 MD-BERT를 소개합니다. MD-BERT는 감마 보정(gamma correction)과 히스토그램 평활화(histogram equalization) 기법을 포함한 다중 스트림(multi-stream) 아키텍처를 채택하여 저조도 환경에서의 도전 과제를 해결합니다. 또한, Dynamic Feature Fusion(DFF) 모듈을 통해 다양한 요소를 통합하여 비디오 프레임 간의 복잡한 상호작용을 효과적으로 포착합니다.

- **Technical Details**: MD-BERT는 Raw dark frames, gamma-enhanced frames, histogram-equalized frames의 세 가지 보조 입력 스트림을 통해 다양한 시각적 특징을 효과적으로 처리합니다. DFF 모듈은 지역적(local) 및 전역적(global) 맥락 정보를 조화롭게 통합하여 고유한 시각적 정보를 강조할 수 있도록 설계되었습니다. BERT 기반의 아키텍처를 통해 긴 시간의 종속성을 포착하며, 포지셔널 인코딩(positional encoding)과 멀티헤드 어텐션(multi-head attention)을 사용하여 과제 인식을 향상시킵니다.

- **Performance Highlights**: ARID V1.0 및 ARID V1.5 데이터셋에서 MD-BERT는 기존의 방법들보다 우수한 성능을 보여주며, 저조도 환경에서도 최신 성능 기준을 확립합니다. 각 입력 스트림의 개별 기여도를 강조하는 Ablation 연구도 실시하여 제안된 DFF 및 BERT 모듈의 효과성을 입증했습니다. 이 연구는 저조도 비디오에서 행동 인식의 새로운 수준을 제시하며, 비디오 인식 기술의 발전에 기여할 것으로 기대됩니다.



### Efficiently Generating Expressive Quadruped Behaviors via Language-Guided Preference Learning (https://arxiv.org/abs/2502.03717)
Comments:
          8 pages 5 figures

- **What's New**: 이 논문은 언어 기반의 선호 학습(Language-Guided Preference Learning, LGPL)이라는 새로운 접근 방식을 소개합니다. 이 방법은 사전 학습된 대규모 언어 모델(LLM)을 사용하여 초기 행동 샘플을 생성하고, 사용자 피드백을 기반으로 이러한 샘플을 세분화하여 로봇 행동을 개선합니다. LGPL은 사용자 간의 상호작용을 고려하여 로봇의 행동을 빠르게 조정할 수 있는 방법을 제시하여, 샘플 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 연구는 쿼드롭드(quadruped) 로봇의 행동 조정을 위한 문제를 다룹니다. LGPL은 LLM을 활용하여 보상 함수의 후보 파라미터를 생성하며, 이 후보들을 인간의 피드백을 통해 평가하고 조정합니다. 이 과정에서 LLM은 사용자의 선호를 보다 정교하게 이해할 수 있게 해 주며, 이는 다수의 쿼리를 통해 로봇의 행동을 보다 효과적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: LGPL은 단 4개의 쿼리로 정확하고 표현력이 풍부한 행동을 빠르게 학습할 수 있음을 보여줍니다. 실험 결과, LGPL의 L2 손실은 선호 학습과 기존 LLM 파라미터화 방식보다 각각 53%와 62% 낮았으며, 사용자는 LGPL의 생성 행동을 76%의 비율로 선호했습니다. 이는 LGPL이 다양한 사용자 필요에 맞춰 로봇 행동을 보다 효율적으로 적응시킬 수 있음을 나타냅니다.



### Boosting Knowledge Graph-based Recommendations through Confidence-Aware Augmentation with Large Language Models (https://arxiv.org/abs/2502.03715)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)을 활용하여 추천 시스템을 위한 지식 그래프(KG)의 품질과 관련성을 향상시키기 위한 새로운 프레임워크인 CKG-LLMA(Confidence-aware KG-based Recommendation Framework with LLM Augmentation)를 제안합니다. CKG-LLMA는 KG를 고품질 정보로 풍부하게 하는 LLM 기반 서브그래프 증강기, 노이즈 트리플을 필터링하는 신뢰 기반 메시지 전파 메커니즘, 사용자-아이템 상호작용과 KG 데이터를 통합하는 이중 뷰 대조 학습 방법을 포함합니다.

- **Technical Details**: CKG-LLMA의 구조는 LLM의 도움을 받아 KG 기반 추천 시스템 내의 대조 학습, 메시지 전파 및 서브그래프 증강을 결합하여 설계되었습니다. 이 프레임워크는 비록 기존 KG에서 발생하는 노이즈와 구식 관계들을 해결하고, 사용자 및 아이템 표현을 강화하기 위해 두 개의 뷰를 사용하는 대조 학습 방법을 채택합니다. 또한, LLM을 통해 사용자 행동에 대한 설명을 생성하는 과정을 통해 추천 시스템의 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, CKG-LLMA는 여러 공개 데이터 세트에서 다양한 기초 모델보다 우수한 성능을 보였습니다. 우리의 모델은 신뢰할 수 있고 설명 가능하며, 사용자에게 유익한 추천 결과를 생성하는 능력을 보여주었습니다. 특히, LLM의 도입을 통해 KG 기반 추천 시스템의 전반적인 성능을 크게 향상시켰습니다.



### MultiQ&A: An Analysis in Measuring Robustness via Automated Crowdsourcing of Question Perturbations and Answers (https://arxiv.org/abs/2502.03711)
Comments:
          AAAI 2025 Workshop on Preventing and Detecting LLM Misinformation (PDLM) (Oral)

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 답변에서 발생할 수 있는 hallucination 문제를 해결하기 위한 시스템인 MultiQ&A를 제안합니다. MultiQ&A는 LLM이 생성한 답변의 일관성과 견고성을 평가하는 체계적인 접근 방식으로, 다양한 질문의 변형을 만들고 이에 대한 답변을 crowdsourcing하고 있습니다. 실험을 통해 1.9백만 개의 질문 변형과 2.3백만 개의 답변을 검토하였으며, gpt-3.5-turbo 모델이 변수에 대해 비교적 일관성을 유지함을 보여줍니다.

- **Technical Details**: MultiQ&A는 세 가지 구성 요소로 이루어진 강력한 다단계 파이프라인을 갖고 있습니다: Query Rewriter, Answer Generator, Aggregator. Query Rewriter는 원래의 쿼리(q0)를 다양한 의미적으로 일관된 변형으로 변환하며, Answer Generator는 이러한 변수들에 대해 독립적인 gpt-3.5-turbo 에이전트를 사용하여 다수의 답변을 생성합니다. Aggregator는 개별 답변을 통합하여 최종 결과를 도출하는 역할을 수행합니다.

- **Performance Highlights**: MultiQ&A의 실험 결과, 1.9 백만 개의 질문과 2.3 백만 개의 답변을 분석하여 실제 시나리오를 모방한 결과, gpt-3.5-turbo 모델이 의미적으로 안정적이면서도 다양한 표현을 생성함을 추가적으로 확인하였습니다. MultiQ&A는 LLMs의 변동성을 강조하며, 각 질문에 대한 모델의 변화를 보여주는 것을 목표로 한다는 점에서 큰 의의가 있습니다. 따라서, 이 시스템은 기관에서 LLM를 채택하기 위한 신뢰성 있는 프레임워크로 작용할 가능성을 제공합니다.



### Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers (https://arxiv.org/abs/2502.03708)
- **What's New**: 이 논문에서는 Large Language Model (LLM)의 내부 활성화를 통해 의미론적 개념을 감지하고, 원하는 출력으로 유도하는 일반적인 방법을 제안합니다. 특히, 비선형 특성 학습 방법을 사용하여 각 레이어에서 개념을 예측하는 데 중요한 선형 방향을 식별하고, 레이어 간의 특징을 집계하여 강력한 개념 탐지기와 유도 메커니즘을 구축합니다. 이를 통해 환각, 유해성, 독성 및 진실성 결여를 탐지하는 데 최신 결과를 달성하였습니다.

- **Technical Details**: 이 연구에서는 Recursive Feature Machines (RFMs)를 통한 비선형 방법을 사용하여 레이어별 내부 활성화에서 특정 개념을 감지하고 이를 유도하는 프레임워크를 설명합니다. LLM의 활성화로부터 개념 벡터를 집계함으로써 감지 및 유도 프로세스를 개선하며, 데이터 효율성을 높이는 동시에 적은 수의 라벨이 있는 훈련 요청으로도 높은 성능을 보여줍니다. 본 방법은 표준 LLM 추론 파이프라인에 통합할 수 있어 별도의 개념별 튜닝 모델이 필요하지 않습니다.

- **Performance Highlights**: 본 접근 방식은 기존의 탐지 방법보다 향상된 성능을 보여줍니다. 일곱 개의 벤치마크에서 잇따른 실험을 통해 우리의 탐지 방법이 환각과 유해성 개념을 포함하여 다양한 개념을 감지하는 데 효과적임을 입증하였습니다. 또한, 상대적으로 적은 자원을 가진 공개 언어 모델에서도 GPT-4o와 같은 최신 LLM을 초내기 위한 성능 향상도 목격되었습니다.



### LLM Alignment as Retriever Optimization: An Information Retrieval Perspectiv (https://arxiv.org/abs/2502.03699)
Comments:
          26 pages

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 정렬(alignment)을 향상시키기 위한 새로운 직접 최적화 방법인 LarPO(LLM Alignment as Retriever Preference Optimization)를 소개합니다. 기존의 RL(강화 학습) 기반 접근법의 복잡성을 해결하기 위해, 정보 검색(IR) 원리를 활용하여 LLM과 IR 방법론을 연결하는 시스템적인 프레임워크를 제안합니다. 이를 통해 LLM의 생성 모델과 보상 모델을 IR의 검색기-재정렬기(retriever-reranker) 패러다임에 매핑합니다.

- **Technical Details**: LarPO는 LLM의 선호도를 직접 최적화하는 새로운 방법으로, 정보 검색의 세 가지 핵심 원리인 검색기 최적화 목적, 하드 네거티브 마이닝(hard negative mining), 후보 목록 구축(candidate list construction)을 활용하여 정렬 품질을 개선합니다. LLM은 검색기의 역할을 하고, 보상 모델은 재정렬기의 역할을 수행함으로써 효과적인 정렬 성능을 구현합니다. 이러한 접근은 LLM의 생성 및 IR 검색기에서 일반적으로 사용되는 이중 인코더 아키텍처를 활용하여 성과를 도출합니다.

- **Performance Highlights**: 실험 결과, LarPO 방법은 AlpacaEval2에서 38.9%와 MixEval-Hard에서 13.7%의 평균적인 성능 향상을 보여주었습니다. 이러한 결과는 LarPO의 효과를 증명하며, 정보 검색 기법이 LLM 정렬의 품질을 높일 수 있는 새로운 가능성을 제시합니다. 본 연구는 LLM 정렬과 정보 검색 간의 중요한 연결고리를 확립함으로써, 이 분야의 발전을 위한 실용적인 방법을 제공합니다.



### A Comparison of DeepSeek and Other LLMs (https://arxiv.org/abs/2502.03688)
Comments:
          21 pages, 5 figures, 6 tables

- **What's New**: 최근 DeepSeek(딥시크)는 AI 커뮤니티 내외에서 많은 주목을 받고 있으며, 본 논문에서는 DeepSeek과 다른 대형 언어 모델(LLM) 간의 비교를 다룬다. 본 연구는 두 가지 설정에서 작업을 수행하며, 첫 번째는 저자 분류(authorship classification), 두 번째는 인용 분류(citation classification)다. 각 실험에서 DeepSeek은 Claude, Gemini, GPT 및 Llama와 같은 4개의 인기 LLM과 비교된다.

- **Technical Details**: 이 논문에서는 저자 분류와 인용 분류를 통해 고유한 데이터 세트를 활용하여 LLM의 예측 정확도를 분석한다. 저자 분류는 문서가 인간에 의해 작성되었는지 AI에 의해 생성되었는지를 판단하는 작업이다. 인용 분류에서는 작은 텍스트 조각과 그에 대한 인용 유형을 매칭하는 정밀한 분류를 수행한다.

- **Performance Highlights**: DeepSeek은 대부분의 경우 Gemini, GPT 및 Llama에 비해 분류 정확도에서 우위를 보였으나, Claude에는 미치지 못했다. DeepSeek은 상대적으로 느리지만 사용 비용이 낮고, Claude보다 상당히 비쌌다. 출력 유사성 면에서 DeepSeek은 Gemini 및 Claude와 가장 유사한 결과를 보였다.



### Variational Control for Guidance in Diffusion Models (https://arxiv.org/abs/2502.03686)
Comments:
          8 pages in main text. Total of 20 pages

- **What's New**: 이 연구에서 제안하는 Diffusion Trajectory Matching (DTM)은 기존의 방법들의 한계를 극복하기 위해 Variational Inference (변분 추론) 및 제어 관점에서 Diffusion 모델을 재조명합니다. 기존의 Classifier Guidance (분류기 안내) 또는 Classifier-Free Guidance (비분류기 안내) 기법들은 추가 모델 학습을 요구하거나 표준화된 가정에 기반하여 샘플 품질을 해치는 경우가 많았습니다. DTM은 이러한 기법들을 통합하여 추가 학습 없이도 고성능을 발휘하도록 설계되었습니다.

- **Technical Details**: DTM 프레임워크는 Guided Diffusion Dynamics (유도 확산 동역학)를 마르코프 체인으로 모델링 하여 통제 신호를 Variational Parameters (변분 매개변수)로 정의합니다. 이로써 생성된 샘플들이 비조건부 샘플 매니폴드에 가깝게 유지되도록 최적화를 적용하며, 원하는 단말 조건을 만족하는 과정을 보장합니다. DTM은 또한 Non-linear Diffusion Trajectory Matching (NDTM)으로 구체화되며, 기존의 최첨단 Diffusion 모델 샘플러와 잘 통합됩니다.

- **Performance Highlights**: NDTM은 ImageNet-256 및 FFHQ-256과 같은 데이터셋에서 슈퍼 해상도 및 인페인팅과 같은 도전적인 문제들에 대해 이전의 최신 기법들을 초월하는 성능을 보여주었습니다. 예를 들어, 이미지 노이즈 제거 문제에서 DTM을 통해 FID 점수 34.31을 달성하며, 기존의 최적 사전 훈련 방법의 FID 78.07를 크게 개선했습니다. 연구팀은 향후 코드도 공개할 계획이라고 밝혔습니다.



### Reflection-Window Decoding: Text Generation with Selective Refinemen (https://arxiv.org/abs/2502.03678)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 텍스트 생성에서 자기 회귀 해독의 단점을 이론적으로 규명하고, 생성된 내용을 정제하기 위한 슬라이딩 리플렉션 윈도우와 일시정지 기준을 포함한 프레임워크를 제안합니다. 이는 정제(refinement)와 생성을 교환적으로 수행할 수 있는 방법을 통해 효율성과 최적성을 동시에 충족시킬 수 있습니다. 이 접근 방식은 기존의 해독 방식보다 상당한 개선을 보여줍니다.

- **Technical Details**: 연구는 텍스트 생성에서 자기 회귀 방식의 단점을 강조합니다. 자기 회귀 방식은 이전에 생성된 내용을 수정하는 자연스러운 메커니즘이 부족하여 최적의 응답을 보장하지 못합니다. 본 논문에서는 슬라이딩 리플렉션 윈도우와 일시정지 기준을 도입하여 다수의 토큰을 병렬로 예측할 수 있도록 하고, 이로 인해 해독 과정 중 독립적으로 정제와 생성을 오갈 수 있는 구조를 제시합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 접근 방식이 기존 해독 방법보다 성능이 개선되었음을 보여줍니다. 이 방법은 성능 및 효율성 면에서 빔 탐색(beam search)과 유사하거나 더 나은 결과를 제공합니다. 따라서 새로운 접근은 텍스트 생성을 위한 효율적이고 최적화된 해법을 제공하는 데 기여합니다.



### An Empirical Study of Methods for Small Object Detection from Satellite Imagery (https://arxiv.org/abs/2502.03674)
- **What's New**: 이 논문은 원거리 감지 이미지에서 소형 객체를 탐지하기 위한 방법론을 리뷰하고, 성능과 기술적 과제를 분석하기 위한 실증적 평가는 네 가지 최신 기법을 집중적으로 다룹니다. 특히, 도시 위성 이미지에서의 자동차 탐지와 농업용 토지의 위성 이미지에서의 벌통 탐지를 사례로 활용하고 있습니다. 기존 문헌을 바탕으로 소형 객체 탐지에 적합한 여러 최상위 방법을 식별하였으며, xView 및 SkySat와 같은 고해상도 위성 이미지 데이터셋을 실험에 사용했습니다.

- **Technical Details**: 최근 몇 년 동안 다양한 학습 기반(object detection) 방법들이 발전하여 YOLO, SSD, R-CNN 계열 등 여러 기술들이 등장했습니다. 위성 이미지 분석에서의 소형 객체 탐지 과제는 기존 방법들이 직면한 주요 도전 과제를 드러내며, 특히 낮은 해상도에서의 맥락 정보 부족 문제와 불균형한 포지티브 및 네거티브 샘플의 부재가 분석되었습니다. 이 연구에서는 Convolution-based 방법, 특히 YOLO와 SSD 등의 성능을 더 향상시키기 위해 다중 해상도 특징 추출 방법인 Feature Pyramid Networks (FPNs)를 통하여 해결책을 모색합니다.

- **Performance Highlights**: 이 논문에서 수행한 실험은 소형 객체 탐지의 성능을 평가하였고, 다양한 방법이 서로 다른 환경에서 어떻게 작동하는지를 분석했습니다. YOLO 모델은 반복적으로 개선되어 소형 객체 탐지에서의 정확성과 계산 시간을 향상시켰으며, SSD 모델은 빠르고 간단한 방법으로 여전히 소형 객체 탐지에서 유의미한 역할을 하고 있음을 보여주었습니다. 연구 결과, 특히 Dense Objects와 같은 복잡한 상황에서 소형 객체 탐지의 정확성을 높이기 위해 다단계 접근 방식인 Cascade R-CNN이 강력한 성능을 발휘한다는 점을 강조했습니다.



### Advancing Reasoning in Large Language Models: Promising Methods and Approaches (https://arxiv.org/abs/2502.03671)
Comments:
          9 Pages, 1 Figure, IEEE Format

- **What's New**: 이번 논문은 최근의 연구가 진행되고 있는 Large Language Models (LLMs)의 추론 능력 향상 방법들을 종합적으로 리뷰합니다. LLMs는 자연어 처리(NLP) 분야에서 큰 성공을 거두었지만, 심층적인 추론 능력에서 여전히 제한이 있습니다. 저자들은 특히 Chain-of-Thought, Self-Consistency, Tree-of-Thought와 같은 다양한 Prompting Strategies, retrieval-augmented models와 neuro-symbolic integration 같은 Architectural Innovations, 그리고 specialized datasets를 활용한 Learning Paradigms를 통해 LLMs의 추론 능력 강화를 시도하고 있습니다.

- **Technical Details**: LLMs의 추론 능력은 논리적 추론, 수학적 문제 해결 및 상식적 추론 등 다양한 인지 프로세스 유형으로 나눌 수 있습니다. 이 논문에서는 Deductive, Inductive, Abductive 및 Commonsense Reasoning을 소개하며, 전통적인 AI 접근 방식과 LLMs의 차이를 설명합니다. 또한, Chain-of-Thought(CoT) 추론을 포함한 다양한 Prompting Techniques의 중요성을 강조하며, Self-Consistency와 같은 기법을 통해 LLMs의 응답 품질을 개선할 수 있는 방법을 조명합니다.

- **Performance Highlights**: DeepSeek-R1과 같은 최신 LLM은 특히 수학 및 코딩 분야에서 뛰어난 추론 능력을 보여주고 있습니다. 이 모델은 사람과 유사한 분석적 사고를 시뮬레이션하여 복잡한 문제 해결 및 논리적 추론에서 성과를 나타냅니다. CoT와 Self-Consistency 기법은 LLM이 다양한 문제에 대해 더 정확한 추론을 할 수 있도록 돕는 데 기여하고 있으며, 이러한 방법들을 활용한 연구가 지속될 것으로 기대됩니다.



### Unrealized Expectations: Comparing AI Methods vs Classical Algorithms for Maximum Independent S (https://arxiv.org/abs/2502.03669)
Comments:
          24 pages, 7 figures, 8 tables

- **What's New**: 이번 논문에서는 AI(인공지능) 기반의 방법과 전통적인 CPU 기반 방법을 비교하여 Maximum Independent Set (MIS) 문제에 대한 성능을 분석합니다. 특히, GPU를 기반으로 한 AI 방법이 KaMIS라는 상태-of-the-art 전통 솔버보다 뛰어나지 않음을 발견했습니다. 더 나아가, AI 방법이 단순히 임의의 휴리스틱(degree-based greedy)과 유사한 성능을 보이며, 추후 처리(post-processing)를 통해서도 CPU 기반 솔버보다 성능이 저조한 것으로 나타났습니다.

- **Technical Details**: 결정적 문제를 해결하기 위한 CO(combinatorial optimization) 방법론으로 AI 기반 알고리즘이 주목받고 있습니다. 하지만 본 연구에서는 NON-Backtracking AI 방법이 최적의 해를 찾기보다는 가장 단순한 degree-based greedy 접근과 유사한 결론에 도달하게 됨을 발견했습니다. 실험은 다양한 그래프 유형에서 AI 알고리즘이 KaMIS와 비교하여 저조한 성능을 보이는 경향을 분석하는 데 주력하였습니다.

- **Performance Highlights**: KaMIS는 희소한 랜덤 그래프에서 거의 모든 실험에서 AI 기반 알고리즘과 비교하여 강력한 성능을 보였습니다. 입력 그래프 크기가 커지거나 조밀해질수록 KaMIS의 성능 우위가 더욱 뚜렷해졌습니다. 이 결과는 Coja-Oghlan & Efthymiou(2015)에서 제안된 다항식 시간 알고리즘에 대한 상한 추정과 모순되는 것으로 볼 수 있습니다.



### Energy & Force Regression on DFT Trajectories is Not Enough for Universal Machine Learning Interatomic Potentials (https://arxiv.org/abs/2502.03660)
- **What's New**: 이번 연구는 Universal Machine Learning Interactomic Potentials (MLIPs)를 통해 물질 발견을 가속화하는 방법론을 제안합니다. 기존의 MLIP 연구는 밀도 범함수 이론(DFT)에 과도하게 의존하여 신뢰성과 정확성을 보장하는 데 한계를 보여주고 있습니다. 이 논문에서는 더 정확한 시뮬레이션 방법을 이용해 MLIP 교육 데이터를 생성하고, MLIP 메트롤로지 도구를 통해 내재적인 작동 방식을 이해하며, 계산적으로 효율적인 MLIP를 개발할 필요성을 강조합니다.

- **Technical Details**: MLIPs는 양자적 정확성으로 다양한 물질의 잠재 에너지를 모델링하기 위해 훈련됩니다. 이 연구는 Coupled Cluster Theory와 같은 보다 정확한 시뮬레이션 방법을 도입하여 실제 물질 응용의 복잡성을 반영하는 데이터 세트를 생성해야 한다고 주장합니다. 또한, MLIP 평가 방법을 통해 실험적으로 측정된 특성을 모델링하는 MD 시뮬레이션에서 MLIP의 능력과 한계를 효과적으로 이해할 수 있도록 해야 합니다.

- **Performance Highlights**: 효율적인 MLIP 추론 워크플로우는 MD 시뮬레이션에서 전통적인 방법을 능가할 수 있는 잠재력을 unlock(잠금 해제)합니다. 이를 통해 MLIPs는 더 크고 화학적으로 복잡한 시스템을 모델링할 수 있게 됩니다. 최종적으로, 이러한 방향성을 통해 MLIPs는 복잡한 물질을 정확하게 모델링하여 실물 스케일에 적용할 수 있는 대안으로 자리잡을 수 있음을 보여줍니다.



### A Study in Dataset Distillation for Image Super-Resolution (https://arxiv.org/abs/2502.03656)
- **What's New**: 이 논문은 대규모 데이터셋을 소형화된 합성 샘플로 압축하는 개념인 데이터셋 증류(Dataset Distillation)에 대해 탐구하고 있습니다. 특히, 이미지 분류에 주로 연구가 집중된 점을 넘어 이미지 초해상도(Super-Resolution, SR) 분야로의 응용을 확대하였습니다. 실험을 통해 전체 데이터셋과 유사한 SR 성능을 유지하면서 데이터셋 크기를 91.12% 줄일 수 있음을 보여줍니다.

- **Technical Details**: 고해상도(HR) 이미지 재구성을 위해 저해상도(LR) 이미지에서 SR 모델을 훈련시키는 데 필요한 데이터셋 증류 기법을 사용하고 있습니다. SR 품질을 유지하면서 훈련 데이터의 양을 줄이는 데 중점을 두며, 독특하게도 픽셀 공간과 잠재 공간을 비교 분석합니다. 최적화 전략 및 초기화 방법에 대해서도 심도 있는 분석을 수행하여 메모리 효율성과 계산 비용을 최적화하고자 합니다.

- **Performance Highlights**: 본 연구는 합성 데이터셋을 통해 데이터 크기를 대폭 줄이면서도 SR 성능에서 경쟁력 있는 결과를 달성하는 방법론을 제시합니다. 또한, 추가적인 통찰력을 제공함으로써 향후 메모리 효율적인 SR 모델 교육에 대한 기초를 마련하고 있습니다. 결과적으로, 데이터셋 증류와 SR 간의 간극을 메우는 데 중요한 연구 방향을 설정하고 있습니다.



### Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics (https://arxiv.org/abs/2502.03654)
Comments:
          8 pages, excluding references and appendix

- **What's New**: 이번 논문에서는 Gompertz Linear Unit (GoLU)라는 새로운 self-gated activation function을 도입합니다. GoLU는 Gompertz 함수의 비대칭성을 활용하여 기존의 활성화 함수들보다 효과적으로 잠재 공간의 분산을 줄이는 동시에 강력한 gradient flow를 유지합니다. 여러 과제를 통한 실험 결과, GoLU는 최신 활성화 함수에 비해 우수한 성능을 보여주며, 이는 현재의 활성화 함수들에 대한 견고한 대안으로 자리잡고 있습니다.

- **Technical Details**: GoLU는 Gompertz 함수를 게이팅 메커니즘으로 사용하는 self-gated 활성화 함수입니다. 이 함수는 exponentials를 사용하여 무한히 미분 가능하며, ReLU와 그 변형들과는 달리 매끄럽고 비 모노토닉한 특성을 지닙니다. Gompertz 함수의 비대칭성은 Gumbel 분포의 근본적인 비대칭성에서 기인하여 출력의 세기가 다른 gated activation function들에 비해 압축된 효과를 내게 합니다.

- **Performance Highlights**: 다양한 과제를 대상으로 한 실험 결과, GoLU는 기존 self-gated 활성화 함수들보다 더 효과적으로 잠재 표현의 분산을 줄입니다. 이는 모델의 활성화 출력에서 노이즈를 줄여주어, 필수적인 특징을 보존하면서도 과적합(overfitting)을 방지하는 데 도움이 됩니다. GoLU는 각기 다른 데이터 세트에서 탁월한 성능을 보여주며, 이는 고차원 데이터 처리에 대한 향상된 능력을 나타냅니다.



### REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations (https://arxiv.org/abs/2502.03629)
- **What's New**: 기존의 이미지 편집 모델들이 실제 사용자 요구를 충족시키지 못하는 문제를 다룬 REALEDIT(Real Edit)라는 새로운 대규모 데이터셋을 소개합니다. 이 데이터셋은 Reddit에서 수집된 진짜 사용자 요청과 인간에 의해 편집된 이미지를 포함하고 있으며, 9300개의 평가 예제를 포함하고 있어 다양한 실제 요구를 테스트할 수 있습니다. REALEDIT는 인간의 편향을 감소시키고 다양한 사용자 요구를 반영하는 구조로 설계되었습니다. 이 연구의 결과는 기존 모델이 이러한 작업에서 부족한 점이 있음을 강조합니다.

- **Technical Details**: REALEDIT 데이터셋은 사람에 의해 편집된 이미지와 그에 대한 요청을 기반으로 한 48K의 훈련 예제와 9300개의 테스트 예제를 포함하고 있습니다. 이를 위해 데이터 수집 파이프라인을 구성하였으며, 두 개의 주요 서브레딧인 r/PhotoshopRequest와 r/estoration에서 받은 요청을 기반으로 하여 데이터를 구성하였습니다. 편집된 이미지들은 원본 이미지와 편집 지침을 포함하여 사용자의 실제 요청을 반영하는 형태로 수집되었습니다. 이 데이터셋은 진짜 편집 요구 사항을 더 효과적으로 반영하며, 기존의 합성 데이터세트의 한계를 극복하고자 합니다.

- **Performance Highlights**: REALEDIT 모델은 기존의 최고 성능 모델보다 165포인트 높은 Elo 스코어를 기록하며 주목받았습니다. 또한 VIEScore와 같은 자동화된 메트릭에서도 92%의 향상을 보여줍니다. 모델은 Reddit에서 새로운 요청에 대해 긍정적인 피드백을 받았으며, 편집 외에도 진짜 편집된 이미지의 탐지 성능이 향상될 가능성도 확인되었습니다. 이 연구는 이미지 편집 작업 외에도 다양한 AI 기반 응용 분야에 대한 데이터셋의 유용성을 강조합니다.



### The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering (https://arxiv.org/abs/2502.03628)
- **What's New**: 본 논문은 Large Vision-Language Models (LVLMs)에서 발생하는 환각 현상(hallucination)에 대해 연구하였습니다. LVLMs는 시각적 및 텍스트 정보를 효과적으로 처리할 수 있지만, 종종 의미는 일관되나 시각적으로 지지되지 않는 내용을 생성을 야기합니다. 이를 해결하기 위해, 논문에서는 새로운 VISTA(VIsual Information Steering with Token-logit Augmentation) 프레임워크를 제안하였으며, 이는 환각을 줄이는 동시에 진정한 정보를 증진시키는 방법을 제공합니다.

- **Technical Details**: 연구에서 LVLMs의 토큰 로짓(token logits) 랭킹을 조사하여 세 가지 주요 패턴을 발견하였습니다. 첫 번째는 시각 정보의 점진적 손실로, 이는 진정한 토큰의 우선 순위가 감소하고 환각 토큰의 우선 순위가 상승하는 현상이 발생합니다. 두 번째 패턴은 의미 있는 토큰이 모델의 중간 계층에서 최고 활성도를 나타내며, 세 번째는 시각적으로 진정한 토큰이 최종 단계에서 결정되지 않을 수 있지만 여전히 높은 랭킹을 유지한다는 것입니다. 이를 바탕으로 VSTA는 시각 정보 강화 및 기능적 토큰 사용을 조합하여 작동합니다.

- **Performance Highlights**: VISTA는 실험을 통해 기존 방법에 비해 환각 현상을 약 40% 감소시키는 효과를 보여주었습니다. 다양한 모델 아키텍처(LLaVA, Shikra, MiniGPT-4, InstructBLIP)에서 높은 성능을 나타내며, 추가적인 훈련이나 모델 수정이 필요 없다는 장점이 있습니다. 이 프레임워크는 다양한 디코딩 전략에 쉽게 적용 가능하며, 열린 생성(open-ended generation) 및 시각적 질문 응답(visual question answering) 등 여러 평가 프로토콜에서 우수한 결과를 제공합니다.



### AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails (https://arxiv.org/abs/2502.03622)
Comments:
          7 pages, 3 figures, 2 tables, accepted in 4th IEEE International Conference on AI in Cybersecurity (ICAIC)

- **What's New**: AdaPhish는 위협적인 피싱 공격에 대해 자동으로 분석하고 익명화할 수 있는 AI 기반 플랫폼입니다. 이 시스템은 전통적인 phish bowl 접근 방식의 한계를 극복하여, 실시간으로 새로운 피싱 전술에 적응할 수 있습니다. 또한, 감지된 피싱 공격에 대한 자동 보고서와 경고를 제공하여, 사용자에게 즉각적인 대응을 가능하게 합니다.

- **Technical Details**: AdaPhish는 대형 언어 모델(LLMs)과 벡터 데이터베이스를 사용하여 피싱 이메일을 자동으로 익명화하고 분석합니다. 이 시스템은 피싱 트렌드를 장기적으로 추적할 수 있으며, 데이터 처리의 효율성을 크게 향상시킵니다. 이를 통해 사용자는 수작업으로 진행되는 전통적인 방법에 비해 더 나은 보안 교육을 받을 수 있습니다.

- **Performance Highlights**: 자동화된 보고서와 실시간 알람 기능 덕분에 AdaPhish는 대규모 협업 감지를 지원합니다. 이 플랫폼은 사용자가 피싱 공격을 보다 효과적으로 인식하도록 도와주며, 사이버 보안 교육에 기여할 수 있는 확장 가능한 솔루션을 제공합니다. AdaPhish는 이미 다양한 조직에서 효과적인 피싱 탐지 방법으로 자리잡고 있습니다.



### A Novel Zero-Touch, Zero-Trust, AI/ML Enablement Framework for IoT Network Security (https://arxiv.org/abs/2502.03614)
- **What's New**: 이 논문에서는 IoT 생태계를 보호하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Zero Trust, Zero Touch 및 AI/ML 기반의 DDoS 공격 탐지 및 완화 기법을 통합하여 현대 IoT 생태계에서 발생할 수 있는 다양한 보안 위협을 효과적으로 대응할 수 있도록 합니다. 특히, 5G/6G에 최적화된 구조를 통해 지속 가능한 IoT 보안을 실현할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 Zero-Trust 원칙에 기반하여 모든 IoT 트래픽을 인증하고 검증하는 과정을 포함합니다. Zero-Touch 프로비저닝 기능은 IoT 장비의 안전한 온보딩 과정을 자동화하며, AI/ML 알고리즘을 활용하여 실시간으로 이상 징후를 탐지하고 대응합니다. 이 프레임워크는 XGBoost, Random Forest 등 5가지 기계 학습 모델을 비교 분석하여 DDoS 공격 탐지의 효과성을 확보합니다.

- **Performance Highlights**: 비교 분석 결과, 앙상블 기반 접근 방식이 다양한 DDoS 벡터를 탐지하고 완화하는 데 가장 높은 성능을 보였습니다. 이러한 결과는 5G 및 6G 환경에서 IoT 생태계의 보안을 강화하는 데 중요한 기초 자료를 제공합니다. 제안하는 프레임워크는 IoT 보안의 리스크를 낮추고 더욱 강력한 보호를 수행할 수 있도록 설계되었습니다.



### (GG) MoE vs. MLP on Tabular Data (https://arxiv.org/abs/2502.03608)
- **What's New**: 최근 연구에서는 신경망 아키텍처를 표 형식 데이터에 적응시키기 위한 노력들이 증가하고 있습니다. 그러나, 이러한 모델들이 많은 파라미터와 긴 훈련 및 추론 시간을 요구함에도 불구하고, 일반적인 다층 퍼셉트론(MLP) 신경망을 지속적으로 초월하는 경우는 드뭅니다. 이 논문에서는 Gumbel-Softmax 게이팅 함수를 사용하는 Mixture-of-Experts(MoE) 모델인 GG MoE를 소개하며, 이는 38개 데이터셋에서 표준 MoE 및 MLP 모델보다 높은 성능을 보이는 것을 입증합니다.

- **Technical Details**: 논문에서는 세 가지 모델, 즉 MLP, MoE, GG MoE의 성능을 비교합니다. MoE는 K개의 독립 모델인 전문가(expert)와 입력을 전문가들에 대한 확률 분포로 매핑하는 게이팅 함수(gating function) 두 가지 주요 구성 요소로 이루어져 있습니다. GG MoE는 MoE의 게이팅 네트워크에 Gumbel-Softmax 활성화 함수를 도입한 것으로, MoE와 GG MoE는 MLP보다 적은 파라미터를 사용하며, 이 점에서 성능 저하 없이 더 효율적인 아키텍처의 가능성을 보여줍니다.

- **Performance Highlights**: GG MoE는 38개의 데이터셋에 걸쳐 평균적인 성능에서 가장 높은 결과를 기록했습니다. MoE와 GG MoE는 모두 MLP에 비해 상당히 적은 파라미터 수를 활용하고 있으며, 이는 그들의 확장성과 앙상블 방법에 대한 잠재력을 의미합니다. 효율적인 파라미터 사용과 높은 성능을 결합한 GG MoE는 표 형식 데이터 예측에 있어 유망한 대안으로 자리매김할 수 있습니다.



### Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models (https://arxiv.org/abs/2502.03607)
- **What's New**: 최근 확산 모델(diffusion models)의 발전은 로봇 공학에서 다양한 경로를 생성하는 데 큰 잠재력을 보이고 있습니다. 그럼에도 불구하고, 이러한 모델을 다중 로봇 이동 계획(Multi-Robot Motion Planning, MRMP)에 적용하는 것은 충돌 회피 및 운동학적 적합성 등의 중요한 제약을 적용하기 어렵기 때문에 도전적입니다. 본 논문에서는 새로운 접근 방식인 Simultaneous MRMP Diffusion(SMD)를 제안하여 제약 최적화를 확산 샘플링 프로세스에 통합하여 충돌이 없는 경로를 생성합니다.

- **Technical Details**: SMD는 제약이 있는 확산 프로세스 내에서 다중 로봇 경로 생성을 공식화하여 충돌이 없는 운동 계획을 보장합니다. 이 접근 방식은 강화된 Lagrangian 방법을 통해 확산 기반 경로 생성이 충돌 회피 및 운동학적 제약을 만족하도록 만듭니다. 또한, MRMP 평가를 위한 첫 번째 벤치마크를 소개하며, 다양한 로봇 밀도와 장애물 복잡성을 가진 여러 시나리오에서 경로 계획 알고리즘을 평가할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, SMD는 기존의 전통적인 및 학습 기반 이동 계획 방법보다 일관되게 우수한 성능을 보여주며, 복잡한 다중 로봇 환경에서 더 높은 성공률과 효율성을 달성했습니다. SMD는 특히 밀도가 높은 장애물과 비구조적 환경에서도 안정적인 성능을 유지하여, MRMP의 잠재력을 크게 확장합니다.



### Clinically-Inspired Hierarchical Multi-Label Classification of Chest X-rays with a Penalty-Based Loss Function (https://arxiv.org/abs/2502.03591)
Comments:
          9 pages with 3 figures, for associated implementation see this https URL

- **What's New**: 이번 연구에서는 임상 해석 가능성을 향상시키면서도 단일 모델, 단일 실행 학습 파이프라인을 유지하는 다중 레이블 흉부 X선(X-ray, CXR) 이미지 분류에 대한 새로운 접근 방식을 제안합니다. CheXpert 데이터셋과 VisualCheXbert에서 파생된 레이블을 활용하여 진단 간의 임상적으로 유의미한 관계를 포착하기 위해 계층적 레이블 그룹화를 통합하였습니다. 이를 통해 고안한 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 통해 레이블 의존성을 강화하였습니다.

- **Technical Details**: 연구진은 어린 레이블이 상위 레이블에 대한 긍정적인 예측이 없이 긍정적으로 예측될 때 패널티를 적용하는 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 제안하였습니다. 패널티 전략으로는 고정 패널티 접근 방식과 데이터 기반 패널티 방법을 탐색하였으며, 데이터 기반 방법은 상위-하위 레이블 간의 의존 가능성에 따라 패널티를 조절합니다. 모델 성능 향상을 목표로, 학습 데이터 установ 일부 조정과 함께 계층적 그룹 재구성의 효과를 종합적으로 분석하였습니다.

- **Performance Highlights**: 제안된 프레임워크는 CheXpert 데이터셋에서 0.9034의 분류 성능을 달성하였으며, 이는 계층적 구조와 커스텀 HBCE 손실 함수의 효능을 입증합니다. 데이터 기반 패널티는 예측 정확도를 향상시킬 가능성을 보였으며, 시각적 설명과 불확실성 추정이 모델 해석 가능성과 투명성을 더욱 높였습니다. 모든 코드, 모델 구성 및 실험 세부 사항은 공공 Git 저장소에 공개되어 연구의 투명성과 재현성을 촉진하고 있습니다.



### A Multi-Task Learning Approach to Linear Multivariate Forecasting (https://arxiv.org/abs/2502.03571)
- **What's New**: 이번 연구에서는 다변량 시간 시계열 데이터의 정확한 예측을 향상시키기 위해 다중 작업 학습(multi-task learning) 관점에서 접근합니다. 시간 시계열 예측을 다중 작업 문제로 정의하고, 유사한 변수를 그룹화하여 각 그룹이 별도의 작업을 형성하도록 제안합니다. 이를 통해 모델의 동작을 개선하고, 예측과 관련된 다양한 문제를 효과적으로 해결할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구진은 선형 모델의 그래디언트를 분석하여 유사한 변수 그룹을 구성하고, 서로 다른 변수를 균형 있게 처리하는 방법을 제안합니다. 이 방법론은 피어슨 상관 계수를 사용하여 선형 관계를 기반으로 변수를 그룹화하고, 다중 머리 선형 모델(MTLinear)을 구성합니다. 각 그룹은 자체적인 예측 과제를 해결하며, 그래디언트 조정 방식으로 우세한 변수가 전체 예측에 미치는 영향을 조절합니다.

- **Performance Highlights**: MTLinear 모델은 기존의 최첨단 모델들과 비교하여 우수한 성능을 보였습니다. 여러 어려운 벤치마크에서 평가를 진행하여 다변량 예측 작업에서 경쟁력 있는 결과를 나타냈습니다. 이러한 접근법은 다변량 시간 시계열 예측 문제의 해결을 위한 강력한 독립 기법으로 자리매김할 수 있습니다.



### Code Simulation as a Proxy for High-order Tasks in Large Language Models (https://arxiv.org/abs/2502.03568)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.09074

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 사고력과 문제 해결 능력을 평가하기 위해 자연주의적(naturalistic) 및 합성(synthetic) 추론 과제를 수집한다. 연구 결과, 합성 데이터는 자연주의적 데이터 수집보다 훨씬 용이하게 대량으로 구할 수 있는 좋은 대안임을 보여준다. 이 연구는 LLM이 코드 시뮬레이션을 통해 추론 과제를 처리하는 능력을 테스트하는 실험적 파이프라인을 개발하였으며, 이 과정에서 합성 데이터의 활용 가능성을 강조한다.

- **Technical Details**: 연구에서는 문제가 코드 또는 자연어로 표현된 과제 쌍(x, p)으로 지정되며, 이들은 동일한 질문을 표현한다. Python 3을 프로그래밍 언어로 선택하여 LLM의 코드 실행 정확성을 측정하였고, 평가 기준으로는 모델의 대답과 코드 실행 결과 간의 일치를 비교하고 이를 정확도로 표현하였다. 이 방식은 LLM의 추론 능력을 보다 정교하게 분석할 수 있도록 도와준다.

- **Performance Highlights**: 결과적으로, 가장 강력한 LLM은 비교적 강력한 실행 능력을 보였으나, 메모리화(memorisation)와 패턴 인식에 크게 의존함에 따라 취약한 실행 과정을 나타냈다. 다섯 가지 비트리비얼(non-trivial) 자연주의적 작업과 그에 상응하는 코딩 과제를 연계하여 GPT-4, GPT-4o, Llama3.1-405B의 성능 간의 상관 관계를 보였다. 또한 새로운 Chain of Thought 확장을 도입하여 메모리화 문제를 완화하는 방법이 제시되었다.



### Proportional Selection in Networks (https://arxiv.org/abs/2502.03545)
- **What's New**: 이번 연구에서는 네트워크에서 $k$개의 대표 노드를 선택하는 문제를 다루고 있습니다. 연구자는 가장 영향력 있는 노드를 식별하고, 선택이 네트워크의 다양성을 비례적으로 반영하도록 하는 두 가지 목표를 설정하였습니다. 이러한 목표를 달성하기 위해 두 가지 접근 방식을 제안하고, 이론적으로 분석하며 실험을 통해 그 효과를 입증합니다.

- **Technical Details**: 제안된 방법은 영향력(influence)과 다양성(diversity)을 모두 반영하는 노드 선택 알고리즘을 포함합니다. 이 알고리즘은 노드 간의 상관관계(correlation)를 고려하여 최적의 노드를 선정하며, 이론적 분석을 통해 그 정확성과 강건함을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 네트워크 내에서 영향력 있는 노드를 효과적으로 식별하고, 다양한 데이터 세트에 대해 높은 성능을 발휘했습니다. 이 연구는 네트워크 분석(network analysis) 및 대표 샘플링(representative sampling) 분야에서 중요한 기여를 합니다.



### Path Planning for Masked Diffusion Model Sampling (https://arxiv.org/abs/2502.03540)
- **What's New**: 이 논문에서는 마스킹 확산 모델(Masked Diffusion Models, MDMs) 추론 과정에서 토큰이 비밀 해제(unmasking)되는 순서가 생성 품질에 미치는 영향을 조사합니다. 새로운 플래너(planner)를 도입하여 각 단계에서 비밀 해제를 선택하는 방식을 제안합니다. 이를 통해 다양한 대안 비밀 해제 전략이 생성 성능을 개선할 수 있음을 밝혀냅니다.

- **Technical Details**: 연구진은 확장된 증거 하한(evidence lower bound, ELBO)을 도출하였으며, 비밀 해제를 계획하는 프레임워크인 Path Planning (P2)을 제안합니다. P2는 사전 훈련된 BERT 또는 디노이저(denoiser) 자체를 이용하여 비밀 해제 결정을 안내하는 방식으로 설계되었습니다. 이 접근법은 기존의 모든 MDM 샘플링 전략을 일반화(generalize)하여 다양한 작업에서 효과적으로 적용될 수 있도록 합니다.

- **Performance Highlights**: P2 방법론은 언어 생성(in-context learning), 코드 생성, 스토리 채우기(story infilling), 수학적 추론, 역 curse correction 등 다양한 도메인에서 현저한 성능 개선을 보여줍니다. 또한 단백질과 RNA 서열 생성을 포함한 생물학적 서열 생성 분야에서도 뚜렷한 효과를 나타냅니다. 이러한 결과는 새로운 비밀 해제 전략이 생성적 모델의 품질을 극대화할 수 있다는 것을 뒷받침합니다.



### An Empirical Exploration of ChatGPT's Ability to Support Problem Formulation Tasks for Mission Engineering and a Documentation of its Performance Variability (https://arxiv.org/abs/2502.03511)
Comments:
          10 pages, 3 figures, submitted to Conference on Systems Engineering Research (CSER)

- **What's New**: 이 논문은 시스템 공학(Systems Engineering, SE)과 미션 엔지니어링(Mission Engineering, ME) 내에서 생성적 인공지능(generative AI)의 활용 가능성을 탐구합니다. 특히, 메가 시스템 관점에서 문제를 formulat(e)하는 데 있어서 AI의 역할에 주목하고 있습니다. 주목할 만한 것은 NASA 우주 임무 설계 과제를 사례로 들어, ChatGPT-3.5의 이해관계자(stakeholder) 식별 능력을 평가하고 있다는 점입니다.

- **Technical Details**: 본 연구에서는 Large Language Models (LLMs)가 ME 문제 formulat(e) 작업을 지원하는 품질과 일관성을 분석합니다. 구체적으로는 이해관계자 식별 작업에서의 성능을 여러 차례 병행하여 시험하고, 출력물의 품질과 변동성을 질적으로 평가했습니다. 결과적으로, LLM은 사람 중심의 이해관계자를 잘 식별하지만 외부 시스템이나 환경 요인 식별에서는 낮은 성과를 보였습니다.

- **Performance Highlights**: 연구 결과, LLM의 출력물은 일관성이 부족하고, 문제 formulat(e)에는 적합하지 않은 솔루션 특정(output) 경향이 있음을 확인했습니다. 그럼에도 불구하고, ChatGPT는 일부 전문가의 업무 부담을 줄일 수 있는 가능성을 보여주었습니다. 그러나 다양한 병행 시도에서 출력물 간의 큰 변동성을 관찰하여, LLM 사용 시 주의를 기울여야 함을 강조했습니다.



### Elucidation of the Concept of Consciousness from the Theory of Non-Human Communication Agents (https://arxiv.org/abs/2502.03508)
Comments:
          Version febrero 2025, originalmente escrito en diciembre de 2024

- **What's New**: 이번 논문은 비인간 커뮤니케이션 에이전트(ANHC)의 관계적 및 포스트 현상학적 이론을 통해 의식(consciousness) 개념을 설명하는 데 초점을 맞추고 있습니다. 특히, 톰 메칭거(Thomas Metzinger)의 자아 모델 이론(Self Model Theory), 캐서린 헤일즈(Katherine Hayles)의 비의식적(cognitive) 인지 과정 개념, 그리고 레노르와 마누엘 블룸(Lenore and Manuel Blum)의 계산(computation) 이론적 관점을 통해 의식의 복잡한 체계적 조직이 어떻게 나타나는지를 탐구합니다.

- **Technical Details**: 이 논문은 비인간 인지 에이전트(non-human cognitive agents)와의 상호작용을 기초로 하여, 사회기술적 시스템(sociotechnical systems)의 설명 가능성(explainability)이 현대 철학과 과학의 인간 중심적(common sense) 관념에 도전하는 방식을 제시합니다. 이를 통해 의식이 복잡한 계산 시스템에서 어떻게 나타나는지, 그리고 이 조직이 비인간 에이전트의 설계 및 규제와 상호작용에 어떻게 영향을 미치는지를 다루고 있습니다.

- **Performance Highlights**: 마지막으로, 이 논문은 의식의 자율성(autonomy), 자유(freedom), 상호 책임(mutual responsibility)과 같은 개념들에 대한 비판적 통합을 통해 새로운 이해 프레임워크(framework)를 설계하는 데 기여하고자 합니다. 이러한 프레임워크는 상호 연결된 세계에서 에이전시의 포괄적(relational) 이해를 가능하게 해줍니다.



### Enhancing Free-hand 3D Photoacoustic and Ultrasound Reconstruction using Deep Learning (https://arxiv.org/abs/2502.03505)
- **What's New**: 이 연구에서는 MoGLo-Net이라는 모션 기반 학습 네트워크를 소개하여 핸드헬드 포토아큐스틱(photoacoustic, PA) 및 초음파(ultrasound, US) 이미징의 3D 재구성을 향상시킵니다. MoGLo-Net은 자기 주의(self-attention) 메커니즘을 혁신적으로 적용하여 연속적인 초음파 이미지 내의 중요한 영역을 효과적으로 이용하여 정밀한 모션 추정을 가능케 합니다. 또한, 새로운 손실 함수(custom loss function)를 개발하여 모션 매개변수에 대한 학습의 신뢰성을 높입니다.

- **Technical Details**: MoGLo-Net은 2D 이미지를 3D 구조로 재구성하기 위해 ResNet 기반 인코더와 장기 단기 메모리(Long Short-Term Memory, LSTM) 블록을 포함합니다. 이 구조는 인접한 프레임 간의 인코딩된 피처 맵의 상관관계를 직접적으로 접근할 수 있는 특별한 블록으로 이루어져 있으며, 주요 지역을 강조하는 글로벌-로컬 주의 모듈을 갖추고 있습니다. 이 기술을 통해 복잡한 3D 구조를 더욱 정확히 시각화할 수 있습니다.

- **Performance Highlights**: 실험 결과, MoGLo-Net은 정량적 및 정성적 성능 지표 모두에서 현재의 최첨단 방법을 초능가했습니다. 3D 재구성 기술은 범위 제한을 넘어 도플러 초음파 및 포토아큐스틱 이미징을 포함하도록 확장되어 혈관 구조의 3D 시각화를 가능하게 하였습니다. 이 연구의 소스 코드는 공개되어 있으며, 이를 통해 다양한 환경에서의 모델 성능을 평가할 수 있습니다.



### Immersion for AI: Immersive Learning with Artificial Intelligenc (https://arxiv.org/abs/2502.03504)
Comments:
          16 pages. To be published in the Proceedings of the 11th Annual International Conference of the Immersive Learning Research Network (iLRN2025)

- **What's New**: 이 연구는 인공지능(AI)의 관점에서 몰입(Immersion)이 의미하는 바에 대해 성찰합니다. 몰입 학습 이론을 적용하여 AI가 인지 생태계(cognitive ecology)에 참여할 수 있는 방법을 탐구하며, AI를 도구가 아닌 참여자로 간주합니다.

- **Technical Details**: 연구는 몰입의 세 가지 개념적 차원인 시스템(System), 내러티브(Narrative), 에이전시(Agency)를 통해 AI의 역할을 재해석합니다. AI가 외부 디지털 서비스에 둘러싸이고, 데이터의 기원, 변화 및 구조적 발전에 대한 내러티브(narrative)를 해석하여, 인간-AI 협업을 형성하는 운영적 및 전술적 결정(dynamic decisions)을 내릴 수 있는 학습 환경을 설계하는 데 있어 실용적인 함의를 제시합니다.

- **Performance Highlights**: 이 연구는 몰입 학습 이론이 고정된 모델을 넘어서 진화할 수 있는 AI 개발에 기여할 수 있는 방안을 제안합니다. 궁극적으로 인공지능이 몰입적인 학습자이자 진화하는 인간-AI 인지 생태계의 참여자로서 이해될 수 있는 기초를 마련합니다.



### Two in context learning tasks with complex functions (https://arxiv.org/abs/2502.03503)
- **What's New**: 본 논문에서는 작은 transformer 모델이 임의의 다항 함수와 연속 함수들을 근사할 수 있음을 증명합니다. 특히, attention layers만으로 구성된 모델들조차도 이러한 함수들을 정확하게 모델링할 수 있는 능력을 보여줍니다. 기존의 작업과 달리 본 연구는 미리 보지 못한 다항 함수 클래스와 복소 함수의 영점들까지 근사할 수 있다는 점에서 중요합니다.

- **Technical Details**: 본 연구는 ICL(In-Context Learning)이 정의된 시간에 모델이 특정 작업을 학습하는 방식을 다루고 있습니다. 특히, [a,b]의 구간에서 임의의 연속 함수를 ICL1 및 ICL2를 통해 학습할 수 있는지를 평가하였으며, 다양한 학습 설정에서 30개 이상의 모델을 훈련시켰습니다. 논문에서는 attention을 정의하는 매트릭스의 한계로 인해 다항 함수의 클래스 형태를 학습하는 데 어려움이 있음을 수학적으로 설명하고 있습니다.

- **Performance Highlights**: 작은 transformer 모델들은 GPT4와 같은 기존의 대형 언어 모델들(Large Language Models)보다 훨씬 우수한 성능을 보였습니다. 이는 적절한 훈련 데이터와 방법이 제공될 경우 복잡한 추론 능력을 발휘할 수 있음을 시사합니다. 그러나 모델들은 훈련 분포 이외의 값에 대한 일반화 능력이 부족하다는 중요한 제한점도 겪고 있습니다.



### DC-VSR: Spatially and Temporally Consistent Video Super-Resolution with Video Diffusion Prior (https://arxiv.org/abs/2502.03502)
Comments:
          Equal contributions from first two authors

- **What's New**: 이번 논문에서는 DC-VSR(Diffusion-based Consistent Video Super-Resolution)라는 혁신적인 비디오 초해상도(VSR) 접근법을 소개합니다. 기존의 비디오 확산(prior) 모델을 활용하여, 공간적(spatial) 및 시간적(temporal) 일관성을 유지하면서도 사실적인 텍스처를 가진 초해상도 비디오를 생성하는 것이 특징입니다. DC-VSR은 Spatial Attention Propagation(SAP)과 Temporal Attention Propagation(TAP)을 이용하여 정보를 효과적으로 전파합니다.

- **Technical Details**: DC-VSR은 입력된 저해상도(LR) 비디오를 여러 개의 시공간(spatio-temporal) 타일로 분해하여 개별적으로 처리합니다. SAP는 비디오 프레임의 전체 영역을 나타내는 서브샘플링된 피처 맵을 도입하여 공간적 일관성을 확보합니다. 반면, TAP는 시간적으로 인접한 타일 간에 정보를 전파하여 시간적 일관성을 높입니다.

- **Performance Highlights**: 실험 결과, DC-VSR은 다양한 불확실한 열화(degradation) 요소를 포함하는 실제 VSR 작업에서 공간적 및 시간적으로 일관된 고품질의 결과를 달성하였습니다. 기존 VSR 접근법보다 월등한 성능을 보여주며, 특히 실제 텍스처 복원에서 뛰어난 결과를 입증하였습니다.



### Efficient Image Restoration via Latent Consistency Flow Matching (https://arxiv.org/abs/2502.03500)
Comments:
          21 pages, 11 figures

- **What's New**: 최근 생성적 이미지 복원(generative image restoration) 기술의 발전으로 놀라운 성과를 거두고 있지만, 이러한 방법들이 고용량과 높은 컴퓨팅 요구사항 때문에 엣지 디바이스(edge devices)에 적용하기 어려운 한계가 있었습니다. 본 연구에서는 ELIR(Efficient Latent Image Restoration) 방법을 소개하며, 이 방법은 잠재 공간(latent space)에서 동작해 마이크로 최솟값 제곱 오차(minimum mean square error) 추정기를 기반으로 잠재 표현을 예측하고 이를 통해 고화질 이미지를 복원합니다. 이러한 접근은 기존의 확산(diffusion) 및 흐름 기반(flow-based) 방법보다 4배 이상 빠르고, 모델 크기도 4배 이상 줄여 자원 제약이 있는 엣지 디바이스에 적합합니다.

- **Technical Details**: ELIR는 두 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 Latent MMSE 추정기를 도입하여 저하된 이미지의 잠재 표현을 기반으로 조건부 기대값을 계산합니다. 두 번째 단계에서는 Latent Consistency Flow Matching(LCFM)을 통해 잠재 흐름 일치를 실행, 이는 계산 복잡성을 줄이고 NFE(neural function evaluations)의 수를 감소시킵니다. 이 방법은 고해상도 이미지 처리에 의해 발생하는 계산 비용을 줄이기 위한 완벽한 해결책을 제공합니다.

- **Performance Highlights**: 실험을 통해 ELIR의 효율성을 검증하였으며, 블라인드 얼굴 복원, 이미지 초해상도, 이미지 노이즈 제거, 인페인팅 및 칼라화 작업에서 우수한 성능을 보였습니다. ELIR는 메모리 크기를 4배에서 45배까지 줄이고, 처리 속도를 4배에서 270배 향상시키면서도 왜곡(distoion)이나 지각 품질(perceptual quality)을 희생하지 않고 기존의 최첨단 방법들과 경쟁력 있는 성능을 유지합니다.



### Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning (https://arxiv.org/abs/2502.03499)
- **What's New**: 이 논문에서는 기존의 Genomic Foundation Models (GFMs)의 한계를 극복하기 위해 Omni-DNA라는 새로운 다중 작업 교차 모델을 제안합니다. 일반적인 모델과는 달리 Omni-DNA는 생물학적 패턴을 공유하고 여러 하위 작업을 동시에 해결할 수 있는 능력을 가지고 있습니다. 이를 통해 GFMs이 모델 사이에 공유할 수 있는 정보의 활용을 극대화할 수 있습니다.

- **Technical Details**: Omni-DNA는 20 million에서 1 billion까지의 파라미터를 갖는 다양한 크기의 모델로서, DNA 시퀀스에 대한 사전 학습(pretraining)과 다중 작업(finetuning)을 통해 훈련됩니다. 훈련 과정에서 기존의 모델 아키텍처와 토크나이징 전략의 수정과 비교를 통해 최적의 구성요소를 찾고, 이를 통해 유연한 출력 공간을 제공합니다. 이 모델은 DNA2Text 및 DNA2Image와 같은 다양한 하위 작업에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: Omni-DNA는 Nucleotide Transformer 및 GB benchmarks에서 26개의 작업 중 18개에서 최고 성능(SOTA)을 달성했습니다. 또한, 다중 작업을 통해 10개의 아세틸화 및 메틸화 작업을 동시에 처리할 수 있으며, 각각의 작업에 대해 개별적으로 훈련된 모델을 초월하는 성과를 보여줍니다. 최종적으로 DNA 시퀀스를 텍스트 기능 설명이나 이미지로 변환하는 두 가지 복잡한 유전체 작업도 설계하여 Omni-DNA의 교차-모달(지식 간 전환) 기능을 강조합니다.



### Teaching Language Models to Critique via Reinforcement Learning (https://arxiv.org/abs/2502.03492)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 출력 품질을 비판하고 개선하기 위해, 인간의 개입 없이도 피드백을 생성하도록 비평가 모델을 훈련하는 프레임워크인 CTRL (Critic Training via Reinforcement Learning)을 제안합니다. CTRL 프레임워크를 통해 코드 생성 과정에서 비평가가 생성자 모델의 성능을 크게 향상시킬 수 있음을 발견했습니다. 특히, 비평가 모델은 더 강력한 생성 모델과 결합하여 놀라운 성능 향상을 이끌어낼 수 있는 능력을 보여주었습니다.

- **Technical Details**: CTRL 프레임워크는 critic 모델을 task-performing 모델에서 분리하여, 비평이 반복되는 과정에서 최적의 솔루션 생성을 유도합니다. 이 과정에서, Group Relative Policy Optimization (GRPO)을 통해 두 단계의 훈련 파이프라인을 구현합니다. 이 방법은 주어진 코드 생성 문제에서 비평가가 직접적인 피드백을 제공하고, 이를 통해 더 나은 솔루션으로 나아갈 수 있도록 돕습니다.

- **Performance Highlights**: CTRL을 사용한 훈련은 CodeContests, LiveCodeBench, MBPP+, JudgeBench 등 여러 벤치마크에서 자가 비평 방법이나 기존의 강력한 비평 모델보다 눈에 띄게 우수한 성과를 나타냈습니다. 이러한 결과는 상대적으로 약한 비평가 모델이 강력한 생성 모델을 효과적으로 안내할 수 있는 가능성을 보여줍니다. 추가적으로, CTRL은 iterative critique-revision을 통해 테스트 시간 성능을 개선할 수 있으며, CodeContests 벤치마크에서 106.1%의 상대적 향상을 이루어냈습니다.



### Artificial Intelligence and Legal Analysis: Implications for Legal Education and the Profession (https://arxiv.org/abs/2502.03487)
- **What's New**: 이번 연구는 법률 (legal) 및 비법률 (non-legal) 대형 언어 모델 (Large Language Models, LLMs)이 법적 분석을 수행할 수 있는 능력을 평가한 결과를 보고합니다. 연구는 LLM이 법적 추론 과제를 수행하는 데 있어 Issue-Rule-Application-Conclusion (IRAC) 프레임워크를 사용하여 진행되었습니다. 연구 결과, LLM이 기본적인 IRAC 분석을 수행할 수 있지만, 응답이 간략하여 세부 정보가 부족하고, 대답에 대한 확신이 없으며, 잘못된 자신감과 환각 (hallucinations) 증상이 나타나는 한계를 보였습니다.

- **Technical Details**: 이 연구는 법률 문제 분석과 유추적 추론 (analogical reasoning) 관련 과제에서 LLM의 성능을 비교하였습니다. LLM들은 짧고 불충분한 대답으로 인해 법적인 사고를 수행하는데 어려움을 겪고 있으며, 이는 법률 교육 및 실무에 시사점을 제공합니다. 법률 전문가처럼 사고할 수 있는 능력을 저해하는 특성 (traits)을 탐색하며, LLM의 효용을 평가한 것입니다.

- **Performance Highlights**: 연구는 LLM이 법적 분석의 필수 요소인 IRAC 방법론을 사용할 수 있는 가능성을 보여주지만, 과도한 의존이 논리 (logic), 추론 (reasoning), 비판적 사고 (critical thinking) 능력을 상실하게 할 위험도 강조합니다. 미래의 법률 전문가들은 이러한 기술적인 한계를 인지하고 비판적 사고 능력을 키울 필요가 있음을 논의합니다.



### Can Domain Experts Rely on AI Appropriately? A Case Study on AI-Assisted Prostate Cancer MRI Diagnosis (https://arxiv.org/abs/2502.03482)
- **What's New**: 이번 연구는 방사선 전문의와의 심도 있는 협력을 통해 MRI 이미지를 기반으로 한 전립선암 진단에서 AI 지원 도구의 실제 통합 효과를 조사합니다. 두 가지 실험을 진행하여 AI 지원과 성과 피드백이 도메인 전문가의 의사결정에 미치는 영향을 분석하였습니다. 특히, 두 개의 상이한 작업 흐름을 설계하여 실제 임상 환경에서 AI 툴이 어떻게 사용될 수 있는지를 모델링하였습니다.

- **Technical Details**: 연구는 8명의 방사선 전문의(N=8)를 포함한 사전 등록된 인간 실험을 수행하였으며, 주요 초점은 전립선암 진단을 위한 AI 지원입니다. 최신 AI 모델(Isensee et al., 2021)을 훈련시켜 전립선암 탐지에 필요한 진단 예측과 양성 사례에 대한 병변 주석 맵을 제공하였습니다. 실험은 두 가지 별도의 작업 흐름으로 구성되었고, 첫 번째 연구에서는 독립적인 진단 후 AI 예측을 확인하는 방식으로 진행되었습니다.

- **Performance Highlights**: 연구 결과, 인간-AI 팀은 인간 단독보다 지속적으로 더 높은 성능을 보였으나, AI 단독 성능에는 미치지 못하는 경향이 있었습니다. 성과 피드백을 제공했음에도 불구하고, 인간-AI 팀의 성능 향상은 제한적이었으며, AI 결정을 사전에 보여주는 방식이 방사선 전문의로 하여금 AI를 더욱 신뢰하도록 유도하는 것으로 나타났습니다. 흥미로운 점은 인간-AI 팀의 다수결 결정이 AI 단독보다 뛰어난Complementary performance를 달성했다는 점으로, 이는 인간-AI 협업의 유망한 방향성을 제시합니다.



### A Capability Approach to AI Ethics (https://arxiv.org/abs/2502.03469)
- **What's New**: 이번 연구는 AI 윤리를 능력 접근법(capability approach)을 통해 개념화하고 구현하려는 시도를 보여줍니다. 능력 접근법을 통해 AI 윤리를 정의함으로써 두 가지 주요 이점을 제시하고자 합니다.

- **Technical Details**: 첫 번째는 AI 도구의 윤리적 차원을 명확히 하는 데 도움이 된다는 점입니다. 두 번째는 AI 도구 설계 내에서 윤리적 고려 사항을 구현하는 데 대한 지침을 제공한다는 점입니다.

- **Performance Highlights**: 의료 분야의 AI 도구와 관련하여, 윤리를 기반으로 한 AI 감사는 능력 기반 접근법을 통해 큰 이점을 얻을 수 있음을 보여줍니다.



### Where AI Assurance Might Go Wrong: Initial lessons from engineering of critical systems (https://arxiv.org/abs/2502.03467)
Comments:
          Presented at UK AI Safety Institute (AISI) Conference on Frontier AI Safety Frameworks (FAISC 24), Berkeley CA, November 2024

- **What's New**: 이 논문에서는 전통적인 비판 시스템의 안전 공학을 AI 안전 프레임워크의 개발 및 구현과 어떻게 연결할 수 있는지에 대한 분석을 제시합니다. 특히, 비행기 비행 제어와 같은 시스템에서의 안전 공학 수칙을 통해 AI 시스템이의 안전성과 위험 분석을 수행하는 방법을 다룹니다. 또한 Assurance 2.0이라는 새로운 이론 기반의 보증 방법을 제안하여, 의사 결정을 지원하는 사전 조건을 충족할 수 있도록 합니다.

- **Technical Details**: 논문에서는 시스템 엔지니어링, 안전 및 위험 분석, 의사 결정 분석 및 지원을 통해 AI 시스템의 안전 공학을 논의합니다. 네 가지 주요 질문, 즉 '시스템이란 무엇인가?', '어떤 성능을 가져야 하는가?', '시스템 개발에 대한 중대성의 영향은 무엇인가?', '얼마나 신뢰해야 하는가?'에 대해 심도 있는 논의를 진행합니다. 또한, 전통적인 시스템 공학 프로세스를 통해 안전 요구 사항과 위험 분석의 중요성을 강조하고, 이 과정에서 생길 수 있는 반복적 문제를 해결하기 위한 체계적인 접근 방식을 제안합니다.

- **Performance Highlights**: 이 연구는 전통적인 비판 시스템에서의 데이터와 경험을 기반으로 AI 시스템 안전 공학의 발전을 위한 중요한 인사이트를 제공합니다. 전반적으로, 품질 보증과 재난 예방을 위한 더 강력한 프레임워크가 필요하다는 결론에 도달하며, 이를 위해 여러 산업에서의 최선의 관행을 도입하는 방법을 제시합니다. 또한, AI 시스템이 위험 수준을 초과하는 경우 안전 프레임워크가 어떻게 변화해야 하는지를 논의하며, 이는 AI 시스템의 미래 개발에 대한 기초 자료로 활용될 수 있습니다.



New uploads on arXiv(cs.LG)

### Value-Based Deep RL Scales Predictably (https://arxiv.org/abs/2502.04327)
- **What's New**: 이번 논문에서는 머신러닝의 성공을 위해 데이터와 계산(compute)의 스케일링(scaling)이 얼마나 중요한지를 다룹니다. 특히, 우리는 작은 규모의 실험으로부터 얻은 성능 예측이 큰 규모의 실험에서 뗄 수 없다는 점을 강조합니다. 가치 기반 오프 폴리시 RL(Off-Policy Reinforcement Learning) 방법들이 예측 가능하다는 것을 보여줍니다.

- **Technical Details**: 우리는 주어진 성능 수준에 도달하기 위한 데이터와 계산 요구사항이 Pareto frontier에 위치함을 확인하고, 이를 업데이트 대 데이터 비율(UPD)을 통해 제어됩니다. 이 프론티어를 추정하여 더 많은 계산을 제공할 때 필요한 데이터 요구량 및 더 많은 데이터를 제공할 때 필요한 계산 요구량을 예측할 수 있습니다. 또한 주어진 성능에 따라 총 자원 예산을 데이터와 계산에 최적 allocation하여 하이퍼파라미터를 결정하는 방법을 제시합니다.

- **Performance Highlights**: SAC, BRO 및 PQL 알고리즘을 DeepMind Control, OpenAI Gym 및 IsaacGym에서 검증하여 데이터, 계산, 예산 또는 성능의 더 높은 수준으로 외삽(extrapolation)합니다. 이 방법론을 통해 RL 특유의 오버피팅(overfitting) 및 플라스틱성 손실(plasticity loss)의 영향을 관리 가능한 예측 가능한 관계가 형성됩니다.



### The Uniformly Rotated Mondrian Kern (https://arxiv.org/abs/2502.04323)
Comments:
          22 pages, 4 figures, postprint for 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025

- **What's New**: 이번 연구에서는 Mondrian 프로세스를 활용하여 회전 불변(이상적이지 않음) 커널을 근사하는 새로운 무작위 특징 맵을 제안합니다. 균일하게 무작위로 회전된 Mondrian 프로세스를 사용하여 생성된 이 커널은 커널 머신의 계산 비용을 줄이면서 더 향상된 성능을 제공합니다. 이 연구는 비선형 파라미터에 대한 최적 조정 방법으로 활용될 수 있는 새로운 기하학적 결과도 제공합니다.

- **Technical Details**: 연구진은 균일하게 회전된 Mondrian 커널에 대한 닫힌 형 표현을 도출하였으며, 이 커널이 극한 상태로 수렴하는 균일한 수렴 속도를 제시하였습니다. 새로운 이론적 기여에는 확률 기하학에서 정적인 무작위 타일링 이론 기법을 활용하여 일반 다각형의 세포 표현도 포함되어 있습니다. 이 연구는 Mondrian 프로세스의 특성을 극대화하여 기능적 근사를 생성할 수 있는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과는 균일하게 회전된 Mondrian 커널이 기존 Mondrian 커널보다 우수한 성능을 발휘함을 보여주었습니다. 특히, 이 커널은 비선형 데이터셋에서도 뛰어난 성능을 보였으며, 계산 비용의 증가 없이도 효율성을 개선했습니다. 이는 무작위_partitioning 기술이 포함된 다양한 머신러닝 작업에서 긍정적인 결과를 나타내는 것을 의미합니다.



### Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions (https://arxiv.org/abs/2502.04322)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 안전성 문제를 다루며, 기존의 연구들이 주로 기술적인 지식을 요구하는 공격 방법에 집중하고 있다는 점을 지적합니다. 연구자들은 jailbreak된 응답이 일반 사용자에게 해로운 행동을 유도하는 데 진정으로 유용한지와 간단한 상호작용에서 안전성 취약점이 존재하는지를 탐구합니다. 이를 통해 저자들은 HarmScore라는 새로운 메트릭을 제안하며, 다중 단계, 다국어 상호작용에서 해로운 행동을 유도하기 위한 새로운 프레임워크인 Speak Easy를 소개합니다.

- **Technical Details**: Speak Easy는 일반 사용자가 해로운 내용을 찾기 위해 사용할 수 있는 두 가지 유형의 인간-LM 상호작용인 다중 단계 추론(multi-step reasoning)과 다국어 질문(multilingual querying)을 모사합니다. 연구자들은 이 프레임워크를 통해 사용자가 해로운 쿼리를 여러 개의 무해한 하위 쿼리로 나누어 LLM의 안전 장치를 쉽게 우회할 수 있도록 하는 방법을 제안합니다. 이 논문은 GPT-4o, Qwen2-72B-Instruct 등 여러 안전 정렬된 LLM을 대상으로 Speak Easy의 효과를 체계적으로 평가하며, HarmScore는 인간 평가와 잘 일치함을 보여줍니다.

- **Performance Highlights**: Speak Easy는 여러 기준을 통해 GPT-4o의 공격 성공률(Attack Success Rate, ASR)을 평균 0.319 증가시키고, HarmScore를 0.426 증가시키는 결과를 도출했습니다. 또 다른 분석 연구를 통해 더 많은 분해 단계와 언어를 사용하는 것이 응답의 해로움을 증가시킨다는 것을 입증했습니다. 특이하게도, Speak Easy는 기존의 jailbreak 방법들에도 통합되어 성능 향상을 가져오는 것으로 나타났습니다.



### Great Models Think Alike and this Undermines AI Oversigh (https://arxiv.org/abs/2502.04313)
Comments:
          60 pages, 20 figures

- **What's New**: 본 논문은 AI Oversight의 맥락에서, 언어 모델(LM)의 유사성이 이들 평가 및 감독에서 어떻게 작용하는지를 탐구합니다. 이를 위해, 모델의 실수(overlap in model mistakes)를 기반으로 한 확률적 유사성 측정 지표인 CAPA를 제안합니다. 연구 결과, 유사한 모델들이 상호작용할 때 더 나은 평가를 수행하게 된다라는 사실을 발견하였습니다.

- **Technical Details**: CAPA는 모델의 정확도를 고려하여 유사성을 측정하기 위해 고안된 메트릭입니다. 논문의 기술적 세부사항에서는 CAPA의 수학적 유도와 다수 모델 설정에의 확장 방법이 포함되어 있습니다. 이 메트릭은 전통적인 정합성은 물론, Scatter π와 Fleiss κ 같은 다른 메트릭들과 비교하여 새로운 계산 방식을 도입하고 있습니다.

- **Performance Highlights**: 모델의 능력이 증가함에 따라, 유사한 오류를 내는 경향이 발견되었습니다. 이는 AI Oversight의 위험 요소를 부각시키며, 모델 유사성을 모니터링할 필요성을 강조합니다. 결론적으로, 모델 간의 유사성을 보도하고 수정하는 것이 AI 감독의 새로운 패러다임에서 필수적임을 언급하고 있습니다.



### Consistency of augmentation graph and network approximability in contrastive learning (https://arxiv.org/abs/2502.04312)
- **What's New**: 본 연구에서는 대조 학습(Contrastive Learning)의 이론적 기반을 보강하기 위해 데이터 증강(graph Laplacian) 그래프의 점화 및 스펙트럼 일관성을 분석했습니다. 데이터 생성 및 그래프 연결성에 대한 특정 조건 하에서 증강 데이터셋의 크기가 증가함에 따라, 증강 그래프 라플라시안이 자연 데이터 다항식 위에서 가중치가 부여된 Laplace-Beltrami 연산자로 수렴함을 보여주었습니다. 이러한 일관성 결과는 그래프 라플라시안 스펙트럼이 다항식 기하학을 효과적으로 포착함을 보장하며, 신경망의 근사 가능성을 확립하는 견고한 프레임워크를 제공합니다.

- **Technical Details**: 연구에서는 그래프 연결성 파라미터와 데이터 생성 파라미터를 바탕으로 증강 그래프 라플라시안의 수렴 메커니즘을 설명합니다. 각 증강 데이터 포인트 쌍은 평균 공유 유사성에 따라 가중치가 결정된 엣지로 연결되어, 데이터가 물리적 구조를 유지하며 확장될 수 있도록 합니다. 분석을 통해, L2 거리에서 그래프 라플라시안 고유 벡터가 관련된 Laplace-Beltrami 고유 함수로 수렴함을 증명하고, 이러한 성질이 회복 가능성(recoverability) 및 실현 가능성(realizability) 가정을 완화하는 데 기여함을 보여줍니다.

- **Performance Highlights**: 이론적 분석을 통해 증강 그래프 라플라시안과 Laplace-Beltrami 연산자 사이의 스펙트럼 근접성을 명확히 지정하였으며, 이를 통해 신경망의 복잡도를 정량화하였습니다. 연구 결과는 기존의 대조 학습 프레임워크의 한계를 명확히 밝히고, 네트워크가 최적 스펙트럼 대조 손실 솔루션을 복제하는 데 필요한 용량을 정확히 추정할 수 있게 해줍니다. 이로써 데이터 증강의 유용성을 높이고, 딥러닝에서의 기하학적 이해를 확장하는 노력을 기울였습니다.



### Finding Pegasus: Enhancing Unsupervised Anomaly Detection in High-Dimensional Data using a Manifold-Based Approach (https://arxiv.org/abs/2502.04310)
Comments:
          21 pages, 14 figures

- **What's New**: 이 논문에서는 비지도 학습(Unsupervised Learning)에 기반한 이상 탐지(Anomaly Detection, AD) 방법을 고차원 데이터(High-Dimensional Data)에서 효율적으로 적용하는 방법을 제시합니다. 특히, 비지도 이상 탐지가 고차원 데이터에서 종종 어려움을 겪는 문제를 해결하기 위해, 우리는 차원 축소(Dimensionality Reduction, DR) 작업의 관점에서 AD 문제를 재정의합니다. 또한, AD 방법들을 '온 매니폴드'(on manifold)와 '오프 매니폴드'(off manifold)로 분류하는 새로운 프레임워크를 개발했습니다.

- **Technical Details**: AD의 각 방법은 '매니폴드 가설'(Manifold Hypothesis)을 바탕으로 하며, 실제 고차원 데이터의 대부분은 낮은 차원의 매니폴드에 가깝게 존재한다는 원리를 따릅니다. 이 매니폴드는 고차원 공간의 일부로, 데이터 포인트들이 로컬적으로 유클리드 공간과 유사하게 나타납니다. 우리는 이 매니폴드를 통해 고차원 데이터의 물리적 성질을 잘 포착할 수 있는 방식으로 차원 축소를 수행합니다.

- **Performance Highlights**: MNIST 데이터에서의 실험 결과, 기존의 가장 효과적인 AD 방법(예: Isolation Forest)과 단순히 조합했을 경우보다 우리 방법의 AD 성능이 16% 향상되었습니다. 이 결과는 고차원 데이터에 대한 응용 가능성이 높음을 시사합니다. 이러한 성과는 고차원 데이터에서 AD의 정확도(Recall)와 정밀도(Precision)를 증대시키는 방법을 제공하며, 향후 실질적인 데이터에 대한 적용 가능성을 엿볼 수 있게 합니다.



### Targeted Learning for Data Fairness (https://arxiv.org/abs/2502.04309)
- **What's New**: 이번 논문은 알고리즘의 불공정성을 식별하고 제거하는 데 집중해왔던 기존 연구와 달리, 데이터 생성 과정(data generating process)의 공정성(data fairness)을 평가하는 방법론을 제안합니다. 특히, 특히 'garbage in, garbage out' 현상에서 생기는 알고리즘의 편향을 극복하기 위해 데이터 자체의 공정성을 분석하는 것에 중점을 둡니다. 이는 알고리즘의 불공정성이 아닌, 데이터의 불공정성을 직접적으로 평가하는 접근법입니다.

- **Technical Details**: 저자들은 비모수 추론(nonparametric inference)을 위한 유연한 프레임워크인 targeted learning을 사용하여 데이터 공정성을 추론합니다. 논문에서는 demographic parity, equal opportunity, 그리고 conditional mutual information 등 다양한 공정성 측정을 위한 추정량을 도출합니다. 이 과정에서 추정량의 이중 로버스트성(double robustness)을 활용하여 보다 정확한 추정이 가능하다는 점도 강조됩니다.

- **Performance Highlights**: 제안된 방법의 유효성을 검증하기 위해 여러 시뮬레이션을 진행하였고, 실제 데이터에도 적용해 그 성능을 평가하였습니다. 이와 같은 접근은 기존의 모델 중심 접근법과 비교하여 데이터 수준 질문에 더 적합한 결과를 도출할 수 있음을 보여줍니다. 이 논문은 데이터 공정성 분석 및 추론의 새로운 방향을 제시하며, 앞으로의 연구에도 기여할 수 있을 것으로 기대됩니다.



### HOG-Diff: Higher-Order Guided Diffusion for Graph Generation (https://arxiv.org/abs/2502.04308)
- **What's New**: 본 논문에서는 Higher-order Guided Diffusion (HOG-Diff)라는 새로운 그래프 생성 모델을 제안합니다. 이 모델은 생성 큐럴릭럼(curiculum)을 따라 점진적으로 현실적인 그래프를 생성하며, 복잡한 위상 구조를 포착하는 능력을 갖추고 있습니다. HOG-Diff는 높은 차원 정보를 기반으로 하여 전통적인 확산(diffusion) 모델보다 더 강력한 이론적 보장을 보여줍니다.

- **Technical Details**: HOG-Diff는 고차원 위상 정보에 의해 안내되는 그래프 생성 커리큘럼을 도입하여 복잡한 그래프 특성을 포착합니다. 이 모델은 그래프 생성 작업을 더 관리하기 쉬운 서브 작업으로 나누어, 먼저 핵심 구조를 포착하는 높은 차원의 그래프 스켈레톤을 생성한 후, 이를 세부 사항으로 정제하여 완전한 그래프를 생성합니다. 또한, HOG-Diff는 확산 다리(diffusion bridge)와 스펙트럴 확산(spectral diffusion)을 통합하여 효율적인 생성과 그래프 생성 원칙 준수를 보장합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, HOG-Diff는 다른 최신 모델들과 비교했을 때 지속적으로 뛰어난 성능을 보였습니다. 모델의 유용성은 상위 구조 정보를 활용함으로써 그래프 생성의 기능적 중요성을 강조합니다. 이 결과는 HOG-Diff가 기존의 그래프 생성 모델보다 더 나은 성능을 발휘함을 나타냅니다.



### Statistical guarantees for continuous-time policy evaluation: blessing of ellipticity and new tradeoffs (https://arxiv.org/abs/2502.04297)
- **What's New**: 이번 연구에서는 단일 이산 관측된 에르고딕(ergodic) 경로를 이용하여 연속 시간 마르코프 확산 프로세스의 가치 함수(value function) 추정 문제를 다룹니다. 이 연구는 최소 제곱 시차 (least-squares temporal-difference, LSTD) 방법에 대한 비점근적(statistical guarantees) 보장을 제공합니다. 특히, 추정기는 길이 $T$의 경로를 사용할 때 $O(1 / \sqrt{T})$의 수렴 속도를 달성함을 보여줍니다.

- **Technical Details**: 연구의 핵심은 확산 프로세스의 고유한 타원성(ellipticity)이 효과적인 수평선(horizon)이 무한대로 발산하더라도 강력한 성능을 보장한다는 점입니다. 이 논문에서는 통계적 오류(statistical error)의 마르코프(Markovian) 구성 요소가 근사 오류(approximation error)로 제어될 수 있음을 입증하며, 마르팅게일(martingale) 구성 요소는 사용하는 기저 함수(basis functions)의 수에 비례하여 느리게 성장하는 것으로 나타났습니다.

- **Performance Highlights**: 오류의 두 출처를 신중히 조정함으로써, 본 분석에서는 근사(approximation) 오류와 통계(statistical) 오류 간의 새로운 균형(trade-offs)을 밝혀냈습니다. 연구 결과, 확산의 혼합 시간(mixing time)과 기저 함수의 수가 대체로 선형적으로 스케일(scalable)할 때에만 좋은 성능을 유지할 수 있음을 강조합니다. 이러한 발견들은 연속 시간 마르코프 확산 프로세스에 대한 가치 함수 추정 방법의 발전에 기여할 것으로 기대됩니다.



### Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants (https://arxiv.org/abs/2502.04290)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 논문에서는 유망하지 않은 평가를 최소화하기 위해 최적 영역에 전략적으로 집중하는 새로운 글로벌 최적화 알고리즘인 Every Call is Precious (ECP)를 소개합니다. ECP는 Lipschitz 상수를 추정할 필요가 없으므로 추가 함수 평가를 피할 수 있게 됩니다. 또한, 무한한 평가 예산에 대해 노 레그렛(no-regret) 성능을 보장하고, 유한한 예산에서는 미니맥스 최적의 레그렛 (regret) 경계를 달성합니다.

- **Technical Details**: ECP는 작은 수용 영역을 시작으로 점진적으로 최적화 문제를 풀어나가며, εtsubscript𝜀𝑡의 증가되는 값과 이전 반복에서 관찰된 점들을 활용합니다. 이러한 방식은 비용이 많이 드는 목적 함수에서 최적의 점을 평가하는 데 필요한 효율성을 크게 향상시킵니다. 또한, 이 알고리즘은 제한된 평가 예산에서도 모든 평가된 점이 가능한 최대 값의 후보가 되도록 보장합니다.

- **Performance Highlights**: ECP는 30개의 비볼록 다차원 최적화 문제에서 10개의 벤치마크 알고리즘과 비교해 우수한 성과를 보입니다. 이 알고리즘은 Lipschitz, Bayesian, Bandits, 진화적 방법들을 포함한 여러 최신 방법들을 초월하는 성능을 기록했습니다. 또한, 하이퍼 파라미터 선택에 대한 강건성이 입증되었습니다.



### Leveraging Geolocation in Clinical Records to Improve Alzheimer's Disease Diagnosis Using DMV Framework (https://arxiv.org/abs/2502.04288)
- **What's New**: 이번 연구에서는 Alzheimer의 조기 발견을 위해 DMV 프레임워크를 제안하며, 여기에는 Llama3-70B 및 GPT-4o 모델을 사용하여 임상 노트를 분석하고 AD 발병 위험 점수를 예측합니다. 이 작업을 회귀 문제로 설정하여 임상 노트의 언어적 특징과 특정 AD 위험 관련 질문을 답하는 목표 변수를 모델링합니다.

- **Technical Details**: 연구는 다양한 특징 세트를 활용하여 지리적 위치 데이터(geolocation data)를 포함하고, 이를 통해 AD와 관련된 추가적인 환경적 맥락을 포착합니다. 이 프레임워크는 Llama3-70B 및 GPT-4o로의 다양한 임베딩 모델을 활용하여 예측 모델을 구축합니다.

- **Performance Highlights**: 결과는 지리적 위치 정보를 통합함으로써 이전 모델보다 조기 AD 위험 점수 예측 오류를 각각 28.57%(Llama3-70B) 및 33.47%(GPT-4o) 감소시킨다는 것을 보여줍니다. 이러한 통합 접근이 AD 위험 평가의 예측 정확성을 높일 수 있음을 시사하며, 임상 환경에서 조기 진단과 개입을 지원할 수 있습니다.



### DECAF: Learning to be Fair in Multi-agent Resource Allocation (https://arxiv.org/abs/2502.04281)
- **What's New**: 이번 논문에서는 자원 할당 문제를 부각시키며, 중앙 집중식의 자원 할당에서 공정성과 효율성을 학습하는 새로운 방법을 제시합니다. 이에 대한 이론적 기반을 마련하기 위해, 분산 평가(Distributed Evaluation) 및 중앙 집중식 할당(Centralized Allocation) 문제로 이러한 문제를 공식화하였습니다. 연구의 초점은 다중 에이전트 시스템에서 장기적인 공정성을 달성하는 것입니다.

- **Technical Details**: 이 연구에서는 Double Deep Q-Learning 기반의 세 가지 방법론을 제안합니다. 첫 번째 방법은 공정성과 유용성을 동시에 고려하는 공동 가중치 최적화(joint weighted optimization)입니다. 두 번째로는 각각 유용성과 공정성을 학습하는 두 개의 Q-추정기(Q-estimator)를 사용하는 분리 최적화(split optimization)이며, 세 번째 방법은 온라인 정책 변동(online policy perturbation)을 통해 기존의 블랙박스 유용성 함수를 공정한 해결책으로 이끄는 것입니다.

- **Performance Highlights**: 제안된 방법들은 다양한 공정성 기능을 평가할 때에도 기존의 공정한 다중 에이전트 강화 학습(fair MARL) 접근법보다 우수한 성능을 보였습니다. 이 메서드는 자원 할당 분야에서 유연한 온라인 유용성과 공정성 간의 균형을 허용하며, 실시간으로 자원을 할당함에 있어서 공정한 해결책을 도출할 수 있는 가능성을 보여줍니다.



### Orthogonal Representation Learning for Estimating Causal Quantities (https://arxiv.org/abs/2502.04274)
- **What's New**: 이 논문에서는 관찰 데이터에서 인과량을 추정하기 위한 새로운 학습 방법인 OR-학습자(OR-learners)를 제안합니다. 기존의 representation learning 방법들은 이론적인 특성이 부족하여, Neyman-orthogonal learners처럼 double robustness 및 quasi-oracle 효율성과 같은 장점을 제공하지 못했습니다. OR-학습자는 이러한 이론적 특성을 유지하면서도 학습된 representation 기반의 인과량을 일관성 있게 추정할 수 있는 방법을 제공합니다.

- **Technical Details**: OR-학습자는 여러 실험을 통해 특정 정규조건을 만족할 때 기존의 representation learning 방법에 비해 성능을 향상시킵니다. Neman-orthogonal learner와 representation learning 방법을 통합한 포괄적인 프레임워크를 제공합니다. 이러한 방법론은 consistent estimation과 double robustness를 보장하며, quasi-oracle efficiency를 지니고 있어 매우 매력적입니다.

- **Performance Highlights**: 논문의 결과에 따르면, OR-학습자는 여러 실험에서 최신 기술(state-of-the-art) 성과를 달성하였으며, 기존 방법 대비 더 나은 성능을 입증했습니다. 이 연구는 인과량 추정을 위한 representation learning 방법의 새로운 가능성을 제시하며, 여러 실험을 통해 그 우수성을 확인했습니다.



### PILAF: Optimal Human Preference Sampling for Reward Modeling (https://arxiv.org/abs/2502.04270)
- **What's New**: 이 논문은 인간 피드백을 기반으로 하는 강화 학습(RLHF)에서의 정책 최적화의 문제를 해결하기 위한 새로운 접근법인 PILAF를 제안합니다. 기존의 RLHF는 보상 모델이 인간의 가치를 완벽하게 반영하지 못하는 문제를 갖고 있으며, PILAF는 이에 대한 해결책으로 제시됩니다. PILAF는 특히 추가 데이터를 통해 정책을 강화할 수 있는 방법론으로, 실질적인 응용 가능성이 높습니다.

- **Technical Details**: PILAF는 정책 모델과 참조 모델을 보간하여 응답을 생성하는 혁신적인 샘플링 방법입니다. 이 방법은 이론적으로 정당화되며, MLE 손실의 기울기가 최적의 oracle 목표와 일치하도록하여 정렬과 효율성을 향상시킵니다. T-PILAF는 정책을 최적화하기 위한 보상 모델링과 가치 최적화를 일치시켜, 훈련의 안정성과 정보 성능을 높입니다.

- **Performance Highlights**: PILAF는 DPO(Direct Preference Optimization) 환경에서 기존 기준보다 뛰어난 성능을 보였습니다. 모든 설정에서 PILAF는 더 높은 보상을 얻고, 참조 모델과의 KL 발산을 줄이는 결과를 보여주었으며, 반복 DPO에서는 자원 및 계산 비용을 40% 이상 줄였습니다. 이는 PILAF의 실제적이고 강력한 효율성을 입증하는 결과입니다.



### Efficient Randomized Experiments Using Foundation Models (https://arxiv.org/abs/2502.04262)
- **What's New**: 이 논문에서는 여러 foundation models의 예측 결과를 실험 데이터와 결합하여 유효한 통계 추론(valid statistical inference)을 유지하면서 난수 실험(randomized experiments)의 효율성을 향상시키는 새로운 접근 방식을 제안합니다. 특히, Hybrid Augmented Inverse Probability Weighting (H-Aipw)라는 새로운 추정량을 통해 기존의 실험 데이터에만 의존하는 표준 추정량보다 최대 20%까지 샘플 크기를 줄일 수 있는 정밀도 향상을 보여줍니다. 이는 의료와 같은 안전이 중요한 분야에서도 유효한 통계 추정을 보장할 수 있는 가능성을 제시합니다.

- **Technical Details**: H-Aipw 추정량은 여러 개의 잠재적으로 편향된 foundation models의 예측을 통합하여 실험 데이터만 사용하는 기존의 용적 회귀(outcome regression) 대신 사용되는 방법입니다. 이 방법은 기본적으로 통계적 가정(minimal assumptions)이 추가되지 않고도 일관성을 유지하고 점근적으로 정규 분포(asymptotically normal)로 동작하며, 대칭 분산(asymptotic variance)이 기존의 추정량보다 크지 않다는 것을 증명합니다. 특히, 이 추정량은 실험에서 측정된 치료 효과를 추정하는 데 필요한 조건을 충족하며, 이에 관한 여러 실험에서 실증적인 결과를 제공합니다.

- **Performance Highlights**: 여러 난수 실험에서 H-Aipw는 기존의 추정량에 비해 상당한 정밀도 향상(equivalent to a reduction of up to 20% in the sample size)을 보여주었습니다. 이러한 결과는 H-Aipw가 약간의 편향이 있는 모델 예측으로부터도 여전히 유효한 통계 추론을 제공함을 시사합니다. 따라서, H-Aipw는 비용이 많이 드는 환자 모집(paticipant recruitment)과 후속 조치(follow-up)를 제거하면서도 견고한 의사결정 지원을 위한 강력한 도구가 될 수 있습니다.



### Realistic Image-to-Image Machine Unlearning via Decoupling and Knowledge Retention (https://arxiv.org/abs/2502.04260)
- **What's New**: 이번 연구에서는 머신 언러닝(Machine Unlearning) 메커니즘을 개선하기 위해 I2I(이미지-투-이미지) 생성 모델에 대한 새로운 프레임워크를 제안하고 있습니다. 기존 연구들이 Gaussian noise를 forget samples에 대한 출력으로 이용하는 방식에서 나아가, 본 연구에서는 forget samples를 out-of-distribution(OOD) 데이터로 다루어야 한다고 주장하고 있습니다. 이를 통해, 머신 언러닝의 신뢰성을 높이고 사용자 데이터를 보다 효과적으로 보호할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 제안된 두 단계의 언러닝(two-step unlearning) 방법은 우선 gradient ascent(GA) 기법을 사용하여 forget samples에 대한 모델 업데이트를 분리합니다. 이후, 모델이 retain samples에서의 성능을 유지할 수 있도록 추가적으로 fine-tuning을 실시합니다. 이 과정은 언러닝이 실질적으로 이루어지도록 보장하며, 연구 결과 OOD 효과를 통해 forget samples의 영향이 제거된 것을 확인할 수 있었습니다.

- **Performance Highlights**: 제안된 방법은 ImageNet-1K와 Places365와 같은 대규모 데이터셋에서 실험을 통해 유의미한 성능 개선을 보여주었습니다. AutoEncoder, VQ-GAN 및 diffusion model을 포함한 다양한 I2I 생성 모델에 대하여 간단한 AutoEncoder를 사용하여 성능을 비교함으로써, 본 언러닝 프레임워크의 효과성을 입증하고 있습니다. 이러한 평가 결과들은 제안된 접근법의 우수성을 확인하는 데 기여하고 있습니다.



### Adapting to Evolving Adversaries with Regularized Continual Robust Training (https://arxiv.org/abs/2502.04248)
- **What's New**: 이 논문은 지속적인 적대적 훈련(Continual Robust Training, CRT) 접근 방식에 대해 설명하며, 새로운 공격에 대응하기 위해 모델을 스스로 조정하도록 하는 방법을 제안합니다. 또한, 과거의 공격에 대한 강건성을 유지하면서 새로운 공격에 대한 강건성을 향상시키는 방법을 이론적으로 입증하였습니다. 이 연구는 다양한 공격 유형에 대한 모델의 강건성을 동시에 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 지속적인 적대적 강건성(Continual Adaptive Robustness, CAR)의 문제를 정의하고, 공격 간 강건성 격차가 로그 공간(logit space) 내의 입력 간 변동 거리와 관련이 있다는 사실을 확인했습니다. 이를 통해, 로그 거리(logit distance)에 대한 정규화를 통해 강건성을 유지할 수 있다는 점을 밝혔습니다. 연구는 CIFAR-10, CIFAR-100, 그리고 ImageNette 데이터세트를 이용해 이론적 결과를 실증적으로 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안된 정규화 방법(Adversarial ℓ2 Regularization, ALR)을 통해 CRT에서 5.48%의 유니언 정확도 향상이 나타났으며, 이는 다양한 공격 조합에 대해 유의미한 성과를 보입니다. ALR을 활용한 각 실험에서 robust accuracy의 개선이 확인되었고, 초기 훈련 시에 더 효율적인 대안을 찾기 위한 연구도 진행되었습니다. 이 연구는 진화하는 공격에 대한 외부 배포에 강건한 모델의 기초 작업이라 할 수 있습니다.



### A Theoretical Framework for Data Efficient Multi-Source Transfer Learning Based on Cram\'er-Rao Bound (https://arxiv.org/abs/2502.04242)
- **What's New**: 본 논문에서는 다중 출처 전이 학습(multi-source transfer learning)에서 데이터 부족 문제를 해결하기 위한 새로운 이론적 프레임워크를 제안합니다. 기존의 연구들이 모든 가능한 샘플을 사용하여 훈련하는 반면, 본 연구는 각 출처 과제로부터 필요한 최적의 샘플 양을 도출하는 접근 방식을 취하고 있습니다.

- **Technical Details**: 제안된 방법론은 교차 엔트로피 손실(cross-entropy loss)에 맞춰 일반화 오차 측정(generalization error measure)을 도입하고, Cramér-Rao Bound를 기반으로 최적의 전이 양을 최소화하여 결정합니다. OTQMS라는 아키텍처 비종속(data-efficient) 알고리즘을 통해 이론적 결과를 구현하여 깊은 다중 출처 전이 학습 모델을 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 다양한 아키텍처와 두 가지 실제 벤치마크 데이터셋을 대상으로 한 분석을 통해 제안된 알고리즘이 최신 기법들(state-of-the-art approaches)보다 정확도와 데이터 효율성 모두에서显著하게 성과를 보였습니다.



### Graph machine learning for flight delay prediction due to holding manouver (https://arxiv.org/abs/2502.04233)
- **What's New**: 본 연구는 항공기 지연 예측을 위해 그래프 기계 학습(Graph Machine Learning) 기법을 적용한 혁신적인 접근 방식을 제안합니다. 전통적인 기계 학습 모델이 공간-시간 관계를 간과하는 반면, 본 연구에서는 방향성 다중 그래프에서 가장자리 특성 예측(edge feature prediction)으로 문제를 모델링합니다. CatBoost와 Graph Attention Networks (GAT)를 사용하여 항공 교통 네트워크 내의 복잡한 상호 의존성을 포착하여 지연 예측의 정확성을 향상시키고자 합니다.

- **Technical Details**: 연구에서는 두 가지 주요 접근 방식을 사용합니다. 먼저, CatBoost 모델을 사용하여 네트워크 내에서 비행의 중요성을 포착하는 그래프 특성을 활용합니다. 두 번째로 Graph Attention Networks (GAT)를 사용하여 관계형 데이터의 예측에서의 성능을 비교합니다. 이러한 기법들은 지연 예측에 있어 그래프 기반 기계 학습 방법의 유용성을 보여줍니다.

- **Performance Highlights**: CatBoost 모델이 불균형 데이터 세트에서 GAT보다 우수한 성능을 보여주며, 보유 이벤트를 효과적으로 예측할 수 있음을 입증합니다. 또한 이 연구는 사용자가 실시간 지연 예측을 시뮬레이션할 수 있는 웹 기반 도구의 잠재적 운영 영향을 논의합니다. 이러한 연구 결과는 연료 효율성을 향상하고 지연을 줄이며 승객 경험을 개선할 가능성을 시사합니다.



### Algorithmic causal structure emerging through compression (https://arxiv.org/abs/2502.04210)
- **What's New**: 본 연구에서는 인과관계(causality), 대칭성(symmetry), 그리고 압축(compression) 간의 관계를 탐구합니다. 기존의 학습(learning)과 압축 간의 연결 고리를 발전시키고 일반화하여 인과 모델(causal models)이 식별 불가능한 상황을 설정합니다. 새로운 프레임워크(framework)를 제안하여, 다수의 환경에서 데이터를 압축할 때 인과관계가 나타나는 결과로 출현한다고 주장합니다.

- **Technical Details**: 본 논문에서는 전통적인 인과 식별(possible assumptions for causal identifiability)이 적용되지 않을 때 사용할 수 있는 알고리즘적 인과관계(algorithmic causality)의 정의를 제시합니다. 이는 Kolmogorov 복잡성(Kolmogorov complexity)의 상한을 최소화하는 과정에서 나타나는 알고리즘적 인과 및 대칭 구조(symmetric structures)에 대한 이해를 확장합니다. 개입(target intervention) 대상으로부터의 지식 없이도 이러한 구조가 어떻게 발생하는지를 보여줍니다.

- **Performance Highlights**: 이 연구는 큰 언어 모델(large language models)과 같은 머신러닝 모델에서 인과관계가 명시적으로 식별되지 않을 때, 인과가 어떻게 출현할 수 있는지에 대한 새로운 관점을 제공할 수 있다고 가설합니다. 이러한 통찰력은 머신러닝의 다양한 분야에 적용될 수 있는 가능성을 지니고 있습니다.



### Ensuring Reliability via Hyperparameter Selection: Review and Advances (https://arxiv.org/abs/2502.04206)
- **What's New**: 이번 논문에서는 하이퍼파라미터 선택을 다중 가설 테스트 문제로 구성하는 새로운 접근 방식을 소개합니다. 이 접근법은 선택된 하이퍼파라미터로부터 얻어진 인구 위험 측정값에 대한 통계적 보장을 제공할 수 있는 방법을 제시합니다. 특히 Learn-Then-Test (LTT) 프레임워크를 통해 이러한 방법을 형식화하고 있으며, 여러 엔지니어링 관련 시나리오를 고려한 확장도 탐구하고 있습니다.

- **Technical Details**: LTT 기반 하이퍼파라미터 선택 방법의 목표는 하이퍼파라미터 벡터 λ를 선택하여 위험 측정 R(λ)을 제어하는 것입니다. 이러한 방법은 데이터 Z의 진짜, 알려지지 않은 분포 PZ를 기반으로 하는 리스크 측정을 일반적인 함수로 정의합니다. LTT 방법은 후보 하이퍼파라미터 집합과 보유된 검증 데이터 집합을 입력으로 삼아 각 하이퍼파라미터의 신뢰성을 테스트해 나갑니다.

- **Performance Highlights**: LTT 방법은 다중 가설 테스트(MHT)를 통해 하이퍼파라미터의 신뢰성을 평가하여 신뢰할 수 없는 구성 요소를 제외합니다. 테스트의 결과로 얻어지는 p-값이나 e-값은 각 하이퍼파라미터에 대한 증거의 척도를 제공합니다. MHT의 결과로 얻어진 하이퍼파라미터 집합은 신뢰도 보장이 담보된 상태로 추천됩니다.



### "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidenc (https://arxiv.org/abs/2502.04204)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)에 대한 jailbreak 공격 방어를 위한 새로운 접근 방식을 제시합니다. 특히, adversarial suffix jailbreak 공격에 초점을 맞추고 있으며, 길이가 Θ(M)인 악성 접미사가 있는 공격에 대한 방어가 Θ(√M)인 접미사로 훈련된 LLM으로 충분하다는 것을 보여줍니다. 이는 짧은 길이의 adversarial training(AT)을 통해 긴 길이의 공격에 효과적으로 대응할 수 있다는 중요한 발견입니다.

- **Technical Details**: 이 논문은 linear transformer의 adversarial in-context learning을 분석하며, 적절한 이론적 프레임워크를 통해 robust generalization bound를 증명합니다. 이 경계는 학습 및 테스트 세트에서의 adversarial 샘플 수에 따라 다르며, 연구 세트에서의 길이 비율이 중요한 역할을 함을 보입니다. 이러한 이론적 분석은 실제 세계의 LLM에 대한 새로운 in-context adversarial attack을 도입하여 더욱 실질적인 시뮬레이션을 가능하게 합니다.

- **Performance Highlights**: 실험적으로, 여러 오픈 소스 LLM에 대해 GCG 공격을 사용한 AT를 시행하고 다양한 접미사 길이에 대한 robustness를 평가했습니다. 결과는 AT의 adversarial 접미사 길이에 대한 공격 성공률(ASR)과의 긍정적인 상관관계를 보여주며, 길이가 20인 AT로도 접미사 길이가 120인 공격의 성공률을 30% 이상 감소시킬 수 있다는 점에서 효율성을 입증합니다.



### Multi-agent Architecture Search via Agentic Supern (https://arxiv.org/abs/2502.04180)
- **What's New**: 본 논문에서는 Large Language Model (LLM) 기반의 다중 에이전트 시스템을 발전시키기 위한 새로운 접근 방식인 MaAS를 소개합니다. MaAS는 기존의 고정된 설계 방식에서 벗어나, 다양한 쿼리에 맞춰 동적으로 리소스를 할당하는 에이전틱 슈퍼넷(agentic supernet)을 최적화합니다. 이러한 방법은 에이전트 간의 협업과 상호 작용을 통해 에이전트의 인지 경계를 확장하도록 돕습니다.

- **Technical Details**: MaAS는 쿼리에 따라 의존적인 에이전트 시스템을 슈퍼넷에서 샘플링하여 고품질 솔루션을 제공하고 맞춤형 리소스 할당을 가능하게 합니다. 이는 LLM 호출, 도구 호출, 토큰 비용 등을 포함한 복잡한 리소스 관리 기능을 갖추고 있습니다. 이 접근법은 기계 학습(ML) 및 자동화 시스템(AI systems)의 설계를 효율적으로 구현할 수 있도록 도와줍니다.

- **Performance Highlights**: MaAS는 기존의 수작업이나 자동화된 다중 에이전트 시스템보다 6%에서 45%까지 적은 추론 비용을 요구하며, 성능은 0.54%에서 11.82%까지 향상되었습니다. 또한, 다양한 데이터셋과 LLM 백본 간의 전이 가능성이 우수하여 적용 범위가 넓습니다. 이러한 성능은 MaAS가 다중 에이전트 시스템 설계의 혁신적 대안이 될 수 있음을 보여줍니다.



### MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation (https://arxiv.org/abs/2502.04176)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 Multimodal Retrieval-Augmented Multimodal Generation (MRAMG)이라는 새로운 태스크를 도입하여, 텍스트와 이미지를 결합한 답변 생성을 목표로 하고 있습니다. 기존의 Retrieval-Augmented Generation (RAG) 방법들이 텍스트 기반의 출력에 집중했던 반면, MRAMG는 다양한 모드의 정보를 최대한 활용하는 데 중점을 두고 있습니다. 논문에서는 또한 MRAMG 성능을 평가하기 위한 MRAMG-Bench라는 새로운 벤치마크를 제안합니다.

- **Technical Details**: MRAMG-Bench는 총 4,346개의 문서와 14,190개의 이미지, 4,800개의 질문-답변(QA) 쌍으로 구성된 데이터셋입니다. 이 데이터셋은 웹 데이터, 학술 논문, 라이프스타일의 세 가지 카테고리에서 수집되었습니다. 새로운 벤치마크는 LLMs와 MLLMs의 성능을 rigorously(엄격히) 평가할 수 있는 통계적 및 LLM 기반 홀드를 벤치마크에 통합하고 있습니다.

- **Performance Highlights**: 아울러, 본 연구는 텍스트와 이미지 모두를 생성할 수 있는 효율적인 멀티모달 답변 생성 프레임워크를 제안하여 기존의 LLM 기반 접근법에 대한 한계를 극복하고 있습니다. MRAMG-Bench를 통해 11개의 고급 생성 모델의 성능을 포괄적으로 평가한 결과, 각 모델의 능력과 제한 사항에 대한 귀중한 통찰을 제공합니다.



### Archetypal Analysis for Binary Data (https://arxiv.org/abs/2502.04172)
Comments:
          5 pages, Accepted at ICASSP 2025

- **What's New**: 이번 연구에서는 이진 데이터(binary data)에 적합한 아키타이플 분석(archetypal analysis, AA)을 위한 두 가지 새로운 최적화 프레임워크를 제안하고 있습니다. 기존의 AA 방법들이 연속 데이터에 기반하여 개발된 반면, 우리가 제안한 방법은 이진 데이터에 특화되어 있습니다. 특히, Bernoulli 분포에 기반한 두 가지 프레임워크를 통해 아키타입을 정의하고, 관측 데이터를 효율적으로 재구성할 수 있는 방법을 제시합니다.

- **Technical Details**: 첫 번째 프레임워크는 아키타입을 정의하기 위해 적Active set algorithm을 활용하여 희소성(sparsity)을 이용한 접근 방식을 채택하였습니다. 두 번째 프레임워크는 주어진 관측 데이터를 위해 Sequential Minimal Optimization (SMO) 기법을 사용하여 저 차원(low-dimensional) 행렬을 효율적으로 업데이트하는 방법을 다룹니다. 이 두 방법 모두 Bernoulli 분포에 최적화되었으며, 다른 데이터 분포에도 쉽게 확장 가능하다는 장점을 가지고 있습니다.

- **Performance Highlights**: 제안된 최적화 절차는 기존에 사용되던 곱셈 기반 이진 AA 방법보다 우수한 성능을 보임을 확인하였습니다. 연구에서는 합성 데이터(synthetic data) 및 실제 이진 데이터를 통해 두 새로운 접근 방식의 성능 우위를 입증하였습니다. 이로 인해, 아키타이플 분석의 적용 범위가 확대되고 다양한 데이터 환경에서 효율적인 최적화 프레임워크 활용이 가능하다는 것을 보여줍니다.



### Making Sense of Touch: Unsupervised Shapelet Learning in Bag-of-words Sens (https://arxiv.org/abs/2502.04167)
- **What's New**: 이 논문은 NN-STNE라는 신경망 모델을 소개하는데, 이는 t-distributed stochastic neighbor embedding (t-SNE)을 히든 레이어로 사용하여 긴 시간 시계열 데이터를 shapelet의 멤버십 확률로 매핑합니다. Gaussian kernel 기반의 평균 제곱 오차(mean square error)는 지역 데이터 구조를 유지하며, K-means는 비볼록 최적화 문제를 해결하기 위해 shapelet 후보를 초기화하는 데 사용됩니다. 기존 방법과는 달리, t-SNE를 활용하여 저차원 공간의 crowding 문제를 해결하고 L1-norm 정규화를 적용하여 shapelet의 길이를 최적화합니다.

- **Technical Details**: 이 논문에서는 시계열 데이터에서 shapelet을 추출하기 위해 비지도 학습 기법을 적용합니다. 선언된 방법론에서는 시간 시퀀스의 유사성을 측정하고, 유사한 시간 시퀀스를 shapelet 후보와 연결시킵니다. 후보 shapelet의 확률 분포를 기반으로 유사도를 업데이트함으로써 optimal한 shapelet을 학습하게 됩니다. 이러한 기법은 UCR 데이터셋 및 전기 부품 조작 작업에서 높은 군집 정확도를 나타냅니다.

- **Performance Highlights**: 이 연구는 기존의 최첨단 특징 학습 방법들에 비해 군집 정확도가 향상된 결과를 보여줍니다. NN-STNE 모델을 통해 얻은 특징은 UCR 오픈 시계열 데이터셋에서 경쟁력 있는 성능을 달성하여, 로봇공학 분야에서의 시계열 데이터 해석의 가능성을 높입니다. 학습된 shapelets는 시간 시퀀스의 차별적인 특징을 효과적으로 식별할 수 있도록 돕습니다.



### Efficient Distributed Optimization under Heavy-Tailed Nois (https://arxiv.org/abs/2502.04164)
- **What's New**: TailOPT는 경량화된 학습을 위한 새로운 프레임워크로, 주로 heavy-tailed stochastic gradient noise 문제를 해결하기 위해 설계되었습니다. 기존의 nested optimization 접근 방식의 단점을 보완하며, 특히 attention-based 모델에서 효과적인 훈련을 가능하게 합니다. 이 논문에서는 local updates와 함께 무한대의 gradient variance를 가져도 수렴성을 보장하는 이론적 기반을 제공합니다.

- **Technical Details**: TailOPT는 adaptive optimization과 clipping 기술을 활용하여 heavy-tailed noise 문제를 해결합니다. 이 프레임워크의 한 변형인 $Bi^2Clip$은 내부 및 외부 최적자 각각에서 coordinate-wise clipping을 수행하여 메모리 및 통신 효율성을 높입니다. 추가적인 gradient 통계 정보를 저장하거나 전송할 필요 없이 adaptive-like 성능을 달성합니다.

- **Performance Highlights**: Empirical results에서 TailOPT와 $Bi^2Clip$은 여러 언어 작업 및 모델에서 뛰어난 성능을 보여주었습니다. 이들은 기존의 최신 기술들(state-of-the-art methods)을 능가하는 결과를 나타내며, 향후 연구에서도 유망한 방향성을 제시합니다.



### Behavioral Entropy-Guided Dataset Generation for Offline Reinforcement Learning (https://arxiv.org/abs/2502.04141)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이번 연구에서는 Behavorial Entropy (BE)를 사용하여 복잡하고 연속적인 고차원 상태 공간에서 다양한 데이터셋을 생성하는 체계적인 탐색 목표를 제안합니다. 특히, BE는 기존의 Shannon Entropy (SE) 및 Rényi Entropy (RE)보다 더 나은 성능을 보일 것으로 기대되며, 데이터 세트의 다양성과 품질을 높이는 데 중점을 두었습니다. 또한, BE는 연속적인 설정에 대한 새로운 이론적 보장과 실제적인 보상 함수가 개발되었습니다.

- **Technical Details**: BE(Behavioral Entropy)는 인지 및 지각 편향을 포함하는 정밀한 확률 기반 일반화된 엔트로피로, 이 연구에서는 BE를 연속적인 상황에서 효과적으로 적용할 수 있는 방법론을 개발했습니다. k-최근접 이웃(k-nearest neighbor) 추정기를 도출하고, 이를 통해 권장되는 탐색 목표와 보상 함수가 기존의 RL 방법과 결합되어 BE 최대화 정책을 훈련하는 데 활용됩니다. 이러한 방법론은 복잡한 상태 공간에서의 데이터셋 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, BE로 생성된 데이터셋을 기반으로 한 오프라인 RL 알고리즘이 SE, RE, Random Network Distillation (RND), State Marginal Matching (SMM) 데이터셋보다 우수한 성능을 보였습니다. 80%의 작업에서 BE 데이터셋이 RE 데이터셋보다 성능이 뛰어난 것으로 나타났으며, 이러한 경향은 다양한 다운스트림 작업에 걸쳐 관찰되었습니다. BE를 통해 생성된 데이터셋은 데이터 및 샘플 효율성을 높이는 데 기여했습니다.



### Synthetic Datasets for Machine Learning on Spatio-Temporal Graphs using PDEs (https://arxiv.org/abs/2502.04140)
Comments:
          Currently under review

- **What's New**: 본 논문은 PDE(Partial Differential Equations)를 기반으로 한 최초의 시공간(spatio-temporal) 데이터셋을 제공하며, 이는 그래프 형태로 설계되어 있습니다. 연구자들은 이 데이터셋을 사용하여 특정 응용 프로그램에 맞춘 맞춤형 데이터셋 및 벤치마크를 생성할 수 있습니다. 특히, 이 연구는 실제 전염병 데이터에 대한 모델 성능을 향상시키는 데 기여할 수 있는 사전 학습(pre-training) 방법도 제안합니다.

- **Technical Details**: 이 연구에서는 유한 요소 방법(Finite Element Method, FEM)을 이용하여 다양한 종류의 재난을 모델링하는 세 가지 PDE를 해결했습니다. 생성된 데이터셋은 각기 다른 전염병 역학, 대기 입자 및 쓰나미 파동과 관련된 시나리오를 다루고 있습니다. 이들 PDE는 불규칙한 도메인에서 해결되며, 데이터는 불규칙하게 분포된 관련 지점을 기반으로 제공합니다.

- **Performance Highlights**: 데이터셋을 기반으로 한 벤치마크 결과, 비슷한 문제를 해결하기 위해 기존 기계 학습 모델들이 효과적으로 평가되었습니다. 전염병 데이터셋에 대한 첫 번째 사전 학습 결과는 최대 45%의 성능 향상을 보여 주며, 이는 이 연구의 중요성을 강조합니다. 제공된 코드와 데이터셋 덕분에 연구자들은 개인의 요구에 맞게 전이 학습(transfer learning)을 쉽게 수행할 수 있습니다.



### Transfer Learning for Covert Speech Classification Using EEG Hilbert Envelope and Temporal Fine Structur (https://arxiv.org/abs/2502.04132)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문은 구술(오버트) 발화 데이터를 통해 학습된 분류기를 은밀한(코버트) 발화 분류에 활용하는 참조 학습(transfer learning) 접근 방식을 제안합니다. 이 방법은 참가자가 단어를 생각하며 반복하는 과정에서 발생하는 정신적 피로와 학습의 어려움을 줄이는 데 초점을 맞춥니다. 사용된 전기뇌파(EEG) 특성은 Hilbert envelope와 temporal fine structure에서 유래하였으며, bi-directional long-short-term memory(BiLSTM) 모델을 통해 분류 작업을 수행하여, 발표된 발화에 대해 86.44%의 정확도, 은밀한 발화에 대해 79.82%의 정확도를 달성했습니다.

- **Technical Details**: 이 연구는 15명의 참가자로부터 수집된 EEG 데이터를 기반으로 하여 진행되었습니다. 참가자들은 가상 로봇을 조작하기 위해 구술 및 은밀한 발화 둘 다를 사용했습니다. 실험은 오버트와 코버트를 번갈아 진행하며, 각 세션에서는 5개 동작 단어를 랜덤하게 반복하여 총 800회의 반복이 이루어졌습니다. EEG 데이터는 64채널 무선 시스템을 사용해 수집되었으며, 사용된 주요 특징들은 EEG 신호의 진폭 변화와 위상 정보를 포착하는 Hilbert envelope과 temporal fine structure입니다.

- **Performance Highlights**: 제안된 방법은 참가자가 은밀하게 발화하는 동안 수행한 EEG 데이터의 분류 정확도를 높이는 데 성공했습니다. 구술 발화 데이터로 학습한 분류기가 은밀한 발화 데이터에 적용될 때 비약적인 성능 향상을 보여주었습니다. 이 연구는 특히 장애인을 위한 의사소통 도구로서의 BCI 시스템의 가능성을 높이면서, 기존의 복잡한 학습 과정을 줄이기 위한 효과적인 접근 방식을 제공하는 데 기여합니다.



### On the importance of structural identifiability for machine learning with partially observed dynamical systems (https://arxiv.org/abs/2502.04131)
Comments:
          15 pages, 18 figures

- **What's New**: 이 논문에서는 구조적 식별 가능성(Structural Identifiability) 분석을 활용한 새로운 시간 시계열 분류 프레임워크인 구조적 식별 가능성 매핑(Structural Identifiability Mapping, SIM)을 제안합니다. 이 접근법은 관측된 시간 시계열을 비식별 가능한 동역학적 시스템의 식별 가능한 매개변수 조합으로 표현하여, 기존의 분류 방법을 사용할 수 있도록 합니다. 이를 통해 제한된 데이터 샘플을 활용하여도 시간 시계열의 정확한 분류가 가능함을 입증했습니다.

- **Technical Details**: 이 연구는 주로 비식별 가능한 매개변수를 가진 일반 미분 방정식(Ordinary Differential Equation, ODE) 모델에 초점을 맞춥니다. SIM 프레임워크에서는 개별 시간 시계열을 원래의 매개변수 공간이 아닌 구조적으로 식별 가능한 매개변수 조합의 공간에서 표현합니다. 이를 통해 기계 학습 커뮤니티에서 상대적으로 간과되었던 구조적 식별 가능성 분석의 중요성을 강조하고 있습니다.

- **Performance Highlights**: 실험을 통해 제안한 SIM 방법이 생물 의학 및 계산 생물학 툴에서 일반적으로 발견되는 동역학 모델의 정확한 분류를 가능하게 함을 보여주었습니다. 특히, 훈련 샘플 수가 제한된 상황에서도 시간 시계열 관측치를 정확하게 분류할 수 있는 능력이 향상되었다는 결과를 얻었습니다. 이는 구조적 식별 가능성을 고려함으로써 기계 학습의 성과를 극대화할 수 있음을 시사합니다.



### Optimizing Perturbations for Improved Training of Machine Learning Models (https://arxiv.org/abs/2502.04121)
- **What's New**: 이 연구에서는 머신러닝 모델의 훈련 프로토콜 최적화를 위해 perturbation을 사용하여 훈련 효율성을 높이는 새로운 방법론을 제안합니다. 특히, perturbation을 첫 번째 도달 과정(first-passage process)으로 모델링하고, 이를 통해 모델 훈련에서의 다양한 perturbation의 효과를 예측할 수 있는 가능성을 보여줍니다. ResNet-18 모델로 CIFAR-10 분류기를 훈련하면서 최적의 perturbation 및 주파수를 식별하는 방법을 증명했습니다.

- **Technical Details**: 연구에서 다루는 NN 훈련은 G(θ, t)라는 프로파게이터를 통해 유동 확률을 묘사하며, 이는 상태 θ에서 시간 t에 있을 확률을 나타냅니다. Perturbation에서는 FPT(first-passage time)가 도입되며, 이 시간은 모델이 특정 테스트 정확도에 도달하는 최초의 순간으로 정의됩니다. 연구팀은 다양한 perturbation 프로토콜(S&P, warm restarts, stochastic resetting)을 평가하고 이들로 인해 훈련 시간의 가속화를 도출해내는 방식을 제시했습니다.

- **Performance Highlights**: 이 연구는 기존의 튜닝 방법론에 비해 DY 포지셔닝(perturbation positioning)을 통해 최대 4배 더 빠른 수렴을 달성할 수 있음을 보여줍니다. 제안된 접근법은 기존의 훈련 프로세스 대비 개선된 일반화 성능을 입증했으며, 머신러닝 훈련 프로토콜의 새로운 통계적 메커니즘적 접근법을 제공합니다. 이는 모델 훈련에 있어 실험을 줄이는 새로운 분석 방법론을 제시함으로써 훈련 효율성을 향상시킬 수 있는 가능성이 있음을 시사합니다.



### Generative Adversarial Networks Bridging Art and Machine Intelligenc (https://arxiv.org/abs/2502.04116)
- **What's New**: 이 논문은 Generative Adversarial Networks (GANs)의 기본 원리와 역사적 발전을 상세히 소개하며, 전통적인 생성 모델과 대조합니다. GAN의 기본 개념을 Python 예제를 통해 설명하고, 확률 이론, 통계 및 게임 이론을 포함한 수학적 기초를 다룹니다. 이후 Conditional GANs, DCGANs, InfoGAN, LAPGAN 등 여러 클래식 변형과 Wasserstein GANs, Gradient Penalty가 있는 GAN, Least Squares GAN 및 Spectral Normalization 기법과 같은 고급 훈련 방법론에 대해 설명합니다.

- **Technical Details**: GAN은 두 개의 신경망, 즉 Generator와 Discriminator로 구성되어 서로 대립하는 구조입니다. Generator는 실제 데이터를 모방한 가짜 데이터를 생성하려 하고, Discriminator는 주어진 데이터가 진짜인지 가짜인지를 판단합니다. 두 네트워크는 동시에 훈련되어 Generator가 Discriminator를 속일 수 있을 정도로 발전하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문에서는 GAN의 효율성을 강조하며, GAN이 생성하는 데이터의 품질이 Variational Autoencoders (VAEs)보다 우수하다는 점을 밝혔습니다. GAN은 다양한 데이터 타입을 생성하는 데 유연성을 제공하며, Discriminator의 피드백 루프가 Generator의 성능 향상에 기여합니다. GAN의 역사적 발전 과정을 살펴보면, 이미지 생성의 품질과 다양성에서 꽤나 중요한 이정표를 보여줍니다.



### Smart IoT Security: Lightweight Machine Learning Techniques for Multi-Class Attack Detection in IoT Networks (https://arxiv.org/abs/2502.04057)
Comments:
          Accepted in an international conference

- **What's New**: 이 연구는 IoT 기기의 사이버 공격을 탐지하기 위해 강력한 머신러닝 프레임워크 기반의 경량 앙상블 접근 방식을 제안하고 있습니다. 특히, 34가지 공격 유형이 포함된 CICIoT 2023 데이터셋을 사용하여 IoT 애플리케이션을 보호하기 위한 최상의 알고리즘을 평가합니다. 연구 결과, 결정 트리(Decision Tree) 모델이 99.56%의 정확도를 보이며 IoT 보안 강화에 큰 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 머신러닝 방식 중 경량 분류 기반 앙상블 기법을 통해 IoT 네트워크의 다양한 공격 유형을 탐지하는 방법을 제시합니다. 특히, 하이퍼파라미터 튜닝과 특성 추출을 통해 기존 및 새로운 공격 유형을 정확히 인식할 수 있는 IDS를 개발하고 있습니다. 이러한 접근 방식은 IoT 보안의 예측 및 반응 효과성을 증대시키는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, 결정 트리 모델이 99.56%의 테스트 정확도를 기록하며 효과적인 탐지 솔루션으로 자리 잡았습니다. 랜덤 포레스트(Random Forest) 모델은 98.22%의 정확도를 나타내며, 머신러닝 방법이 고차원 데이터 상황에서 매우 효과적임을 시사합니다. 이러한 성과는 IoT 장치의 보안을 강화하는 데 머신러닝 분류기를 활용할 수 있는 가능성을 강조합니다.



### TQ-DiT: Efficient Time-Aware Quantization for Diffusion Transformers (https://arxiv.org/abs/2502.04056)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 Diffusion Transformers (DiTs)의 계산 효율성을 개선하기 위해 모델 양자화 방법을 도입하였습니다. 다중 영역 양자화(MRQ)와 시간 그룹화 양자화(TGQ) 기법을 활용하여 DiT 블록 내의 비대칭 분포 및 시간적 변동에 따른 양자화 오류를 줄이고자 하였습니다. 이러한 방식은 지속 가능한 인공지능에 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: Diffusion 모델은 Gaussian 노이즈를 점진적으로 데이터에 추가하여 고품질 이미지를 생성하는 과정으로 이루어져 있습니다. 본 연구에서는 고정된 양자화 매개변수를 사용하는 대신, TQ-DiT라는 새로운 양자화 알고리즘을 통해 시간적 변화에 대응할 수 있는 방법을 제안했습니다. 이를 통해 8비트 정밀도에서 원래 모델과 유사한 성능을 달성하였고, 6비트 정밀도에서는 다른 기준 모델들을 초과하는 결과를 얻었습니다.

- **Performance Highlights**: 제안된 TQ-DiT 알고리즘은 GPU 자원 소모를 줄이며, 적은 양의 보정 데이터셋을 가지고도 더 높은 품질의 이미지를 생성할 수 있는 능력을 보여줍니다. 연구 결과, FID 점수에서는 W8A8에서 0.29 증가하는 성과를 보였으며, W6A6에서는 다른 베이스라인보다 뛰어난 성능을 달성했습니다. 이러한 성과는 저비트 양자화에 대한 적합성을 확인시켜 주며, 실시간 생성 모델의 효율성을 높이는 데 기여할 수 있음을 나타냅니다.



### Evaluating Inter-Column Logical Relationships in Synthetic Tabular Data Generation (https://arxiv.org/abs/2502.04055)
- **What's New**: 이 연구는 합성 테이블 데이터의 평가에서 기존 방법들이 논리적 관계와 의존성을 유지하는 데 실패하고 있음을 강조합니다. 새로운 세 가지 평가 지표인 Hierarchical Consistency Score (HCS), Multivariate Dependency Index (MDI), 및 Distributional Similarity Index (DSI)가 제안되었습니다. 이러한 지표들은 합성 데이터가 논리적 관계를 얼마나 잘 보존하는지를 평가하기 위해 고안되었습니다.

- **Technical Details**: 합성 테이블 데이터 생성의 복잡성을 해결하기 위해, 본 연구는 데이터 간의 다양한 분포와 상호 의존성을 효과적으로 모델링하는 방법을 모색합니다. 기존의 GAN 기반 방법과 최근의 확산 모델(diffusion models)의 한계를 분석하고, 특성과 타겟 변수가 포함된 joint distribution P(X,Y)를 모델링하기 위해 여러 데이터를 생성하는 방법들을 제안합니다. 새로운 평가 지표들은 기존 방법들과의 성능 비교를 통해 그 유효성을 검증합니다.

- **Performance Highlights**: 실험 결과, 제안된 평가 지표는 기존 방법들이 합성 데이터를 생성할 때 유지하지 못하는 논리적 일관성과 의존성의 기준을 충족시킨다는 것을 보여줍니다. 실세계 산업 데이터셋에서의 결과는 HCS, MDI, DSI 지표가 합성 데이터의 질을 더욱 정밀하게 평가하는 데 기여함을 입증하며, 이는 향후 합성 데이터 생성 방법 개선에 중요한 기초 자료로 활용될 것으로 기대됩니다.



### Precision Agriculture Revolution: Integrating Digital Twins and Advanced Crop Recommendation for Optimal Yield (https://arxiv.org/abs/2502.04054)
- **What's New**: 이 논문은 디지털 트윈 구조를 통해 Agriculture 4.0 기술을 활용하여 농업 생산 방식을 혁신할 수 있는 방안을 제시합니다. Weather APIs, GPS modules 및 NPK soil sensors와 같은 데이터 소스를 통합하여 작물 성장 예측을 정확하게 수행하고 수확 시기를 효과적으로 예측할 수 있는 모델을 구축하는 것이 주된 목표입니다. 이를 통해 농업의 효율성을 향상시키고, 물과 농약의 관리 방식을 개선할 수 있습니다.

- **Technical Details**: 데이터 융합(Data Fusion)을 통해 NPK 센서와 GPS 모듈, Weather APIs로부터 얻은 다양한 데이터를 통합하여 작물 추천 모델(Crop Recommendation Models)을 개선합니다. 디지털 트윈 프레임워크(Digital Twin Framework)를 통해 농업 환경을 가상으로 재현하고, 다양한 작물 개발 시나리오를 시뮬레이션하여 자원의 효율적 관리(Resource Management)를 실현합니다. 고급 환경 분석을 통한 예측 정밀도(Predictive Precision)를 높여 최적의 작물 생산과 수확 시점을 예측할 수 있습니다.

- **Performance Highlights**: 강화된 의사결정(Optimized Decision-making)으로 현재 기후 패턴과 토양 성질을 기반으로 작물을 선택해 최적의 결정이 가능해집니다. 자원의 효율성(Resource Efficiency)을 극대화하여 비용을 절감하고 생태계 지속 가능성을 높입니다. 또한, 작물 성장 단계에 대한 정확한 예측(Accurate Forecasting)을 통해 더 나은 계획과 수확 추정을 가능하게 합니다.



### Decision Trees That Remember: Gradient-Based Learning of Recurrent Decision Trees with Memory (https://arxiv.org/abs/2502.04052)
- **What's New**: 본 논문에서는 ReMeDe Trees라는 새로운 결정 트리(Decision Tree) 알고리즘을 소개합니다. 이는 내부 메모리 메커니즘을 통해 처음으로 결정 트리에 순환성(recurrence)을 통합하여, 시퀀스 데이터를 효과적으로 처리할 수 있도록 설계되었습니다. 이 방법론은 이전의 Gradient-Based Decision Trees를 기반으로 하여, 경량화된 메모리를 활용한 연속적 학습을 가능하게 합니다.

- **Technical Details**: ReMeDe Trees는 결정 트리의 내부 노드를 수정하여 과거의 경험에 기반한 분할 결정을 가능하게 합니다. 이 모델은 경량화된 메모리 상태를 통해 정보를 효율적으로 압축하고, 출력에 따라 메모리를 업데이트하는 혁신적인 절차를 제안합니다. 또한, 경량화된 메모리를 기반으로 모든 학습 과정을 그래디언트 하강법(Gradient Descent)으로 최적화하여, 복잡한 시퀀스 데이터 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: 초기 실험에서는 ReMeDe Trees가 고정 크기의 메모리 창 한계를 극복하고, 내부 상태를 통해 정보를 압축하는 데 성공적인 결과를 보였습니다. 이러한 결과는 ReMeDe Trees가 장기 의존성을 갖는 시계열 데이터 작업에 매우 유망한 접근 방식이 될 수 있음을 시사합니다. 결국, 이 연구는 순차적 데이터 처리에서 결정 트리의 해석 가능성과 구조적 장점을 결합하는 한편, 순환 모델의 장점을 극대화하는 기초 자료를 제공합니다.



### Comparing privacy notions for protection against reconstruction attacks in machine learning (https://arxiv.org/abs/2502.04045)
- **What's New**: 이 논문은 머신러닝 및 연합학습(FL)에서 발생하는 재구성 공격(reconstruction attacks)에 대한 방어 수단으로, 상이한 프라이버시 보장 개념을 가진 메커니즘의 비교를 위한 기초적 프레임워크를 제시합니다. 특히, 차별적 프라이버시(differential privacy, DP)와 메트릭 프라이버시(metric privacy)의 포함된 차이를 명확히 하여, 같은 파라미터를 기반으로 비교할 수 있는 방법을 마련했습니다. 이와 같은 비교는 머신러닝 커뮤니티에서 중요한 문제로 인해, 통합 프레임워크를 통해 접근하는 것이 필요합니다.

- **Technical Details**: 본 연구는 Rényi 차별적 프라이버시(Rényi differential privacy, RDP) 프레임워크를 기반으로 하여, DP 및 메트릭 프라이버시 메커니즘의 차별성을 분석합니다. 연구팀은 메트릭 프라이버시 메커니즘인 VMF( von Mises Fisher)와 DP 메커니즘 간의 파라미터 변환을 통해 공동 비교 메트릭을 개발했습니다. 또한, 이를 통해 몬테카를로 시뮬레이션이나 특정 조건 하에서의 실험적 비교를 초월하여 유의미한 일반화를 이끌어낼 수 있었습니다.

- **Performance Highlights**: 본 연구를 통해 Gaussian 및 VMF 메커니즘은 RDP 프레임워크 하에서 동등한 것으로 간주되지만, 실제로는 정확도와 재구성 공격에 대한 보호 측면에서 상당한 차이를 보인다는 것이 확인되었습니다. 특히, Bayes’ capacity를 통해 재구성 공격에 대한 방어 능력을 평가할 수 있으며, 이는 (ε,δ)-DP보다 온전한 메트릭으로 작용합니다. 이 논문이 제안하는 접근법은 머신러닝 및 심층 학습 애플리케이션 내에서 프라이버시 위험 방지에 있어 새로운 기틀을 마련합니다.



### Probe-Free Low-Rank Activation Intervention (https://arxiv.org/abs/2502.04043)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문에서는 FLORAIN이라는 프로브가 필요 없는 활성화 개입 방법을 제안합니다. 이 방법은 특정 활성화 레이어의 모든 어텐션 헤드에 대해 적용되며, 이를 통해 불필요한 분류기 훈련의 필요성을 제거했습니다. FLORAIN은 비선형 저차원 매핑으로 매개변수화되어 있으며, 이는 수정된 활성화와 바람직한 콘텐츠의 매니폴드에서의 투영 간의 거리를 최소화하도록 훈련됩니다.

- **Technical Details**: FLORAIN은 활성화 벡터를 변형하는 방식으로, 기존의 활성화 개입 방법과는 다르게 다층 네트워크에 걸쳐 여러 헤드를 수정하는 대신 한 층에서 효율적으로 작동합니다. 이 방법의 핵심은 바람직한 답변의 영역을 묘사하고 관련된 저차원 변환 매핑을 고려하는 것입니다. 또한, Mahalanobis 거리 투영 아래에서의 분석적 형태로 목표 함수를 제공합니다.

- **Performance Highlights**: 여러 기본 모델을 기준으로 한 실험 결과, FLORAIN은 진실성과 품질을 개선하는 데 있어 여러 기준 방법들보다 일관되게 우수한 성능을 보였습니다. FLORAIN은 빠른 계산을 가능하게 하며, 병렬 처리 조건을 제공하여 후속 처리 시간 감소에 기여합니다.



### Leveraging Reasoning with Guidelines to Elicit and Utilize Knowledge for Enhancing Safety Alignmen (https://arxiv.org/abs/2502.04040)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문에서는 대규모 언어 모델의 안전성을 확보하기 위한 기존의 Refusal Training (RT) 방법의 한계를 분석하고, OOD(Out-Of-Distribution) 공격에 대한 일반화 성능을 향상시키기 위해 새로운 접근법을 제안합니다. 연구 결과, RT는 안전과 관련된 잠재적 지식을 일관되게 이끌어내지 못하는 문제를 드러냈고, 이를 해결하기 위해 Safety Reasoning with Guidelines (SRG)라는 방법을 소개합니다. SRG는 각 쿼리에 대해 단계별 이유 추론을 수행하도록 모델을 훈련시킵니다.

- **Technical Details**: 논문에서는 RT가 OOD 상황에서 일반화 능력이 부족하다고 지적하며, 훈련에서 직접적인 거부에 의존할 경우 모델이 피상적인 단축 경로에 의존하게 되고 탄력적인 표현 매핑을 학습하지 못한다고 설명합니다. 새로운 방법인 SRG는 3가지 필수 구성 요소를 포함하여, 지침에 따라 쿼리에 대한 단계적 추론을 수행하도록 모델을 훈련시키며, 이는 안전 관련 지식 활용을 효과적으로 증진시킵니다. 이 방법은 처리 과정에서 지침의 맥락 조화를 내재화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, SRG 방법이 기존의 RT 방식에 비해 OOD 공격에 대한 일반화 성능을 유의미하게 개선시키는 것으로 나타났습니다. BoN(Best-of-N) 평가 방식을 적용한 결과, N이 증가함에 따라 OOD 공격에 대한 실패 비율(ASR)이 감소하는 현상이 관찰되었습니다. 이 연구는 모델이 안전 관련 잠재 지식을 충분히 보유하고 있지만, 기존 방법이 이를 일관되게 이끌어내지 못했음을 입증하고 있습니다.



### Generalize Drug Response Prediction by Latent Independent Projection for Asymmetric Constrained Domain Generalization (https://arxiv.org/abs/2502.04034)
- **What's New**: 이 연구는 drug response 예측에서의 한계를 극복하기 위해 novel domain generalization framework인 panCancerDR을 제안합니다. 연구진은 각각의 암 유형을 별도의 source domain으로 간주하고, cell lines를 사용하여 domain-specific samples로 정의했습니다. 또한, 새로운 latent independence projection (LIP) 모듈을 도입하여 encoder가 informative하면서도 중복되지 않은 feature를 추출하도록 유도합니다.

- **Technical Details**: panCancerDR은 adversarial domain generalization 접근 방식을 통해 다양한 source domain에서 공통적인 task-relevant feature를 포착합니다. 이 모델은 drug-sensitive samples를 하나의 컴팩트 클러스터로, resistant samples를 여러 개의 클러스터로 분산시키도록 하는 asymmetric adaptive clustering constraint를 적용합니다. 이러한 구조는 각각의 cancer type에서 유래한 독특한 신호를 사용하는 예측 모델로 기능합니다.

- **Performance Highlights**: empirical 실험 결과 panCancerDR은 unseen cancer type에 대해 높은 예측 성능을 보여주며, 단일 세포 단위나 환자 수준에서도 더 나은 성능을 발휘합니다. 이 모델은 in vitro cell line 데이터만을 사용하여 학습되었음에도 불구하고 현재 state-of-the-art 방법들과 비교했을 때 동일하거나 더 뛰어난 결과를 달성했습니다. 이러한 성과는 실제 임상 적용 가능성을 강조합니다.



### Deep Meta Coordination Graphs for Multi-agent Reinforcement Learning (https://arxiv.org/abs/2502.04028)
- **What's New**: 이 논문은 다중 에이전트 강화 학습(MARL)에서 협동 정책을 학습하기 위한 Deep Meta Coordination Graphs (DMCG)라는 새로운 모델을 제안합니다. 기존의 접근방식들은 에이전트 간의 쌍방향 관계에만 의존했지만, DMCG는 고차 및 간접 관계를 포함하여 복잡한 상호작용을 포착합니다. 이러한 새로운 그래프 구조는 여러 유형의 상호작용과 임의 길이의 다중 홉 연결을 지원하여 에이전트 간의 상호작용을 효과적으로 모델링할 수 있도록 디자인되었습니다.

- **Technical Details**: DMCG는 메타 협동 그래프(MCG)를 기반으로 하며, 동적인 엣지 유형을 도입하여 시간에 따라 그래프 구조가 진화할 수 있습니다. 이를 통해 DMCG는 고차 관계와 간접 상호작용을 simultaneously 포착할 수 있습니다. 신경망 모듈인 그래프 컨볼루션 네트워크(Graph Convolutional Network)를 활용하여 협동 정책을 학습하는 데 강력한 표현력을 제공합니다.

- **Performance Highlights**: DMCG는 여러 협동 문제에서 다른 최신 방법들과 비교하여 샘플 비효율성 문제를 해결하고 보다 나은 성능을 보여줍니다. 특히, StarCraft Multi-Agent Challenge (SMACv2)에서 비효율적인 조정 문제를 해결하는 데 효과적임을 실험을 통해 입증하였습니다. DMCG는 복잡한 환경에서 협동 성과를 향상시키는 데 기여하며, 다양한 다중 에이전트 상호작용 유형을 모델링할 수 있는 능력을 강조합니다.



### Variational Quantum Optimization with Continuous Bandits (https://arxiv.org/abs/2502.04021)
Comments:
          8 pages, 3 Figures + 7-page appendix

- **What's New**: 이번 논문에서는 Continuous Bandits를 통해 Variational Quantum Algorithms (VQA)에 대한 새로운 접근 방식을 소개합니다. 기존의 방법들이 겪는 barren plateau (BP) 문제를 해결하기 위해 bandit 방법을 사용하여, 파라미터 최적화를 위한 기법을 제안합니다. 이 연구는 Continuous 환경에서의 pure exploration을 다루며, 고유한 정보 이론적 하한을 도출했습니다.

- **Technical Details**: VQA는 양자 회로의 파라미터가 고전적 알고리즘에 의해 최적화되는 하이브리드 양자-고전적 알고리즘입니다. 이 연구에서는 Lipschitz smoothness와 같은 수학적 성질을 기반으로 하여 기계 학습의 문맥에서 연속적인 bandit 문제를 다룹니다. 특히, continuous setting에서 각 arm의 최적값을 식별하는 확률론적 접근 방식을 설정하여 알고리즘의 효율성을 증가시킵니다.

- **Performance Highlights**: 연구 결과는 제안된 Continuous Bandit 알고리즘이 PQC와 QAOA와 같은 양자 회로에서 이전의 gradient 기반 방법보다 현저히 우수한 성능을 보인다는 것을 입증합니다. 저자들은 샘플 복잡도가 기존 알고리즘보다 적으며, 고유한 상황에 최적화된 방법이 더 많은 arm을 다룰 때 실용적인 이점을 가지고 있음을 보여주었습니다. 이는 이러한 접근 방식이 향후 VQA 설계에 더 나은 기회를 제공할 수 있음을 말해줍니다.



### PINT: Physics-Informed Neural Time Series Models with Applications to Long-term Inference on WeatherBench 2m-Temperature Data (https://arxiv.org/abs/2502.04018)
- **What's New**: 이번 논문에서는 PINT (Physics-Informed Neural Time Series Models)를 소개합니다. 이 프레임워크는 신경망 시계열 모델에 물리적 제약을 통합하여 복잡한 동역학을 포착하는 능력을 개선합니다. PINT는 ERA5 WeatherBench 데이터셋을 활용하여 2m-온도 데이터의 장기 예측에 중점을 두고, 주기적 동역학을 RNN, LSTM 및 GRU 아키텍처에 적용합니다.

- **Technical Details**: PINT는 단순 조화 진동자 방정식(Simple Harmonic Oscillator Equation)을 물리적 prior로 통합하여 주기적 동역학을 모델링합니다. 이 방정식의 분석적 해(solution)는 신경망의 해석 가능성을 강조하며, 데이터 기반 모델에서 물리적 원칙을 결합하는 효과를 정량적으로 평가할 수 있습니다. PINT는 기존의 전통적인 시계열 모델과 달리 첫 90일의 관측 데이터만을 사용하여 두 해에 걸친 예측을 반복적으로 수행합니다.

- **Performance Highlights**: 실험 결과, PINT는 WeatherBench 데이터셋에서 주기적 트렌드를 포착하고 물리적 원칙과의 정합성을 드러내었습니다. 또한, 선형 회귀 모델을 기반으로 한 벤치마크 테스트를 통해 물리적 제약을 내재화한 모델의 효과를 수치적으로 평가했습니다. PINT는 복잡한 수치 예측에 의존하는 기존 모델들과 차별화되며, 해석 가능하고 강력한 장기 예측 결과를 제공합니다.



### Near-optimal Regret Using Policy Optimization in Online MDPs with Aggregate Bandit Feedback (https://arxiv.org/abs/2502.04004)
- **What's New**: 이 논문은 적대적으로 변경되는 손실과 집합 밴딧 피드백(aggregate bandit feedback)을 가진 온라인 유한 수명 마르코프 의사결정프로세스(MDP)에 대한 처음으로 정책 최적화(Policy Optimization) 알고리즘을 제시합니다. 알려진 동역학(known dynamics)과 미지의 동역학(unknown dynamics) 경우에 대해 다양한 성과를 달성하며, 특히 알려진 동역학에서는 최적의 후회를 성립시킵니다.

- **Technical Details**: 우리가 제안하는 알고리즘은 특정한 정책 최적화 프레임워크를 기반으로 하며, 이는 실제 알고리즘인 NGP(Next Generation Policy), TRPO(Trust Region Policy Optimization), PPO(Proximal Policy Optimization) 등과 강한 연관성이 있습니다. 저자들은 U-값(U-values)이라는 새로운 개념을 도입하여 집합 밴딧 피드백 맥락에서 손실을 효과적으로 구성하고 이를 기반으로 후회(bound)를 최소화합니다. 이 접근법은 일반적인 Q-함수(Q-function) 추정과는 다른 방식으로 유도되어, 집합 밴딧 피드백의 특성에 적합하게 설계되었습니다.

- **Performance Highlights**: 알려진 동역학의 경우 후회 상한이 Θ(H²√SAK)으로 첫 번째로 달성되었으며, 이는 다른 연구들과 비교할 때 탁월한 성과입니다. 미지의 동역학에서도 O(H³S√AK)의 후회 상한을 달성하여, 이전 연구들보다 크게 향상된 성과를 보여줍니다. 본 논문의 결과는 집합 밴딧 피드백을 위한 온라인 MDP의 연구에 새로운 기준을 제시하며, 이전의 최적 후회와의 일치를 보여줍니다.



### Online Learning of Counter Categories and Ratings in PvP Games (https://arxiv.org/abs/2502.03998)
- **What's New**: 이번 논문에서는 전통적인 Elo 등급 시스템의 원칙을 확장하여 실시간 대응 범주 학습(Elo Residual Counter Category learning, Elo-RCC) 알고리즘을 제안합니다. 이 방법은 매 경기 후 실시간으로 등급과 대응 관계를 동적으로 조정하여 스칼라 등급의 설명 가능성을 유지하면서 비가역성을 해결합니다. 기존의 신경망 기반 방법들과는 달리, Elo-RCC는 빠른 업데이트와 균형 있는 매칭을 지원합니다.

- **Technical Details**: Elo-RCC는 기대-maximization (EM) 알고리즘을 활용하여 각 플레이어의 최상의 대응을 기반으로 능력의 확률 분포를 동적으로 학습합니다. 이를 통해 대응 관계와 평가 점수를 실시간으로 조정함으로써 비가역성을 처리하면서 스칼라 등급의 단순성과 해석 가능성을 결합합니다. 이 시스템은 기존의 Neural Rating Table과 Neural Counter Table의 한계를 극복하고, 제한된 전략 복잡성을 갖는 게임에서 성능을 검증했습니다.

- **Performance Highlights**: Elo-RCC는 Lin et al.(2024b)의 공개 데이터 세트를 사용하여 검증된 결과, NCT와 동등한 성능을 달성하면서 실시간 업데이트가 가능하다는 점에서의 실용성을 보여주었습니다. 특히, Elo-RCC는 동적 환경에서 대응 관계를 효과적으로 처리할 수 있는 능력을 강조하며, 온라인 업데이트 방법 중 가장 뛰어난 성능을 보였습니다.



### Tight Bounds on Jensen's Gap: Novel Approach with Applications in Generative Modeling (https://arxiv.org/abs/2502.03988)
- **What's New**: 이 논문에서는 Jensen의 간극(Jensen's gap)을 위한 새로운 일반적인 하한 및 상한을 제시합니다. 연구팀은 Struski et al. (2023)의 최근 연구 결과에 영감을 받아 다양한 함수 및 분포에 대한 새로운 수학적 기법을 도입합니다. 특히, 로그 함수와 로그-정규 분포에 대한 자세한 분석을 통해 실제 데이터 세트에서 생성 모델의 로그 가능성을 정확하게 추정하는 방법을 탐구합니다.

- **Technical Details**: Jensen의 불평등(Jensen's inequality)의 개념은 여러 과학 분야에 중요한 영향을 미쳤습니다. 본 연구에서는 함수 f와 확률 변수 X에 대한 Jensen의 간극을 평가하기 위한 새로운 수학적 경계를 제시하며, 실험적으로 우리의 접근법이 기존 방법보다 더 효과적일 수 있음을 보여줍니다. 이 연구는 Variational Inference (변분 추론) 및 Generative Models (생성 모델)에 관련된 문제를 다루고 있습니다.

- **Performance Highlights**: 우리는 로그 가능성(log-likelihood)을 추정하는 기존의 최첨단 솔루션들과 비교하여, 우리의 접근법이 더 우수하다는 분석적 및 실험적 근거를 제시합니다. 특별히 실제 데이터에서 수행된 실험은 특정 기준이 충족될 경우 우리의 방법이 효과적으로 로그 가능성을 추정할 수 있음을 보여줍니다. 이러한 결과는 머신러닝 커뮤니티 내에서 매우 중요한 의미를 갖습니다.



### Temporal Distribution Shift in Real-World Pharmaceutical Data: Implications for Uncertainty Quantification in QSAR Models (https://arxiv.org/abs/2502.03982)
- **What's New**: 이 연구는 현실적인 분포 변화(context of realistic distribution shifts)라는 새로운 관점에서 QSAR 모델의 불확실성 추정(uncertainty estimation) 방법을 평가합니다. 기존의 불확실성 추정 방법들이 I.I.D.(independent and identically distributed) 설정에서 성능이 좋았던 반면, 실제 데이터에서는 이러한 가정이 깨질 수 있어 성능이 저하되는 문제점을 다룹니다. 실세계 제약 데이터(set of realistic pharmaceutical dataset)를 기반으로 불확실성 추정 방식이 시간에 따라 어떻게 변화하고 영향을 받는지를 분석합니다. 이 연구는 불확실성 추정 방법의 신뢰성을 높이기 위해 실제 데이터에 적합한 방법론의 필요성을 강조합니다.

- **Technical Details**: QSAR 모델은 화합물의 생물학적 활동을 예측하는 데 널리 사용되며, 이러한 모델에서의 불확실성 정량화는 ML 모델의 신뢰성을 높이는 강력한 도구입니다. 연구에서는 Bayesian 접근법, 앙상블 기반 방법 및 후처리 확률 보정(post hoc probability calibration) 방법이 포함됩니다. 또한, 과거 연구에서 사용된 불확실성 추정 방법들이 시간에 따른 데이터 변화(distribution shifts)에 얼마나 민감한지를 분석하고 비교합니다. 이 연구에서는 각각의 불확실성 추정 방법이 주어진 실제 제약 데이터에서 어떻게 작동하는지를 평가합니다.

- **Performance Highlights**: 연구 결과, 레이블(label) 및 특성(descriptor) 공간에서 시간이 지남에 따라 눈에 띄는 변화가 발생하며, 이러한 변화의 크기와 실험의 특성이 명확하게 연관되어 있음을 알게 되었습니다. 또한, 상당한 분포 변화가 QSAR 모델에 사용되는 인기 있는 불확실성 추정 방법의 성능을 저하시킨다는 것이 밝혀졌습니다. 이 결과는 실제 데이터에 의해 도입된 분포 변화 하에서도 신뢰할 수 있는 불확실성 정량화 방법을 찾는 것이 여전히 도전적임을 시사합니다. 연구진은 이러한 통찰을 바탕으로 향후 더 나은 불확실성 추정 방법의 개발 방향을 제시합니다.



### Innovative Framework for Early Estimation of Mental Disorder Scores to Enable Timely Interventions (https://arxiv.org/abs/2502.03965)
- **What's New**: 이번 논문에서는 우울증(depression)과 외상 후 스트레스 장애(PTSD)를 자동으로 분류하는 고급 다중 모달(deep multimodal) 딥러닝 시스템을 제안합니다. 이 시스템은 임상 인터뷰 데이터셋의 텍스트(text)와 오디오(audio) 데이터를 활용하여 조기 탐지 및 정확한 진단의 중요성을 강조합니다.

- **Technical Details**: 제안된 방법은 LSTM(롱 쇼트텀 메모리)와 BiLSTM(양방향 롱 쇼트텀 메모리)의 아키텍처를 결합하여 텍스트와 오디오 모달리티의 특징(feature)을 통합합니다. 텍스트 특징은 언어의 의미 및 문법 구성 요소에 초점을 맞추고, 오디오 특징은 음성의 리듬, 톤, 그리고 음조(pitch)와 같은 음성의 특성을 포착합니다.

- **Performance Highlights**: 제안된 방법은 테스트 데이터셋을 사용하여 우울증에 대해 92%의 분류 정확도와 PTSD에 대해 93%의 분류 정확도를 기록했습니다. 기존의 단일 모달(unimodal) 접근 방식을 초월하여 정확성과 강건성을 입증하며, 정신 건강 상태와 관련된 미세한 패턴을 식별하는 데 있어 향상된 능력을 보여줍니다.



### AL-PINN: Active Learning-Driven Physics-Informed Neural Networks for Efficient Sample Selection in Solving Partial Differential Equations (https://arxiv.org/abs/2502.03963)
- **What's New**: 본 연구에서는 **Active Learning 기반의 Physics-Informed Neural Networks (AL-PINN)** 프레임워크를 제안했습니다. 이 방법은 기존의 PINNs의 효율성과 정확성을 향상시킬 수 있는 잠재력을 가지고 있습니다. AL-PINN은 **Uncertainty Quantification (UQ)**와 결합된 Active Learning을 활용하여 훈련 샘플의 선택을 동적으로 최적화합니다.

- **Technical Details**: AL-PINN은 **Monte Carlo Dropout**을 통해 모델 예측의 불확실성을 추정하고, 높은 불확실성이 있는 영역을 추가 훈련을 위해 선택합니다. 이 과정에서 AL 알고리즘은 훈련 데이터를 증가시키고 PINN 모델을 재훈련하는 반복적인 단계를 포함합니다. 이러한 방식은 데이터가 제한된 실제 응용 프로그램에서도 효율적인 학습을 가능케 합니다.

- **Performance Highlights**: AL-PINN은 기존의 방식에 비해 적은 훈련 샘플로 비슷하거나 더 높은 정확성을 달성했습니다. 특히, **WeatherBench** 데이터셋을 사용한 실제 기후 예측에서 AL-PINN의 뛰어난 성능을 확인했습니다. 데이터 수집 비용이 높은 과학적 또는 공학적 응용 분야에서 AL-PINN의 실행이 유망하다는 결과를 얻었습니다.



### Non-convex composite federated learning with heterogeneous data (https://arxiv.org/abs/2502.03958)
- **What's New**: 이번 논문에서는 비볼록(composite non-convex) 연합 학습(federated learning)에서 근접 연산자 근사(proximal operator evaluation)와 서버 및 클라이언트 간의 통신을 분리하는 혁신적인 알고리즘을 제안합니다. 각 클라이언트는 로컬 업데이트(local updates)를 사용해 서버와의 통신 빈도를 줄이고, 통신 라운드마다 단일 m차원 벡터(d-dimensional vector)만을 전송하여 클라이언트 이동(client drift) 문제를 해결합니다.

- **Technical Details**: 제안된 알고리즘은 비볼록 및 비매끄러운(non-smooth) 문제의 특성을 고려하여 일반적인 비볼록성(non-convexity) 및 근접 Polyak-Lojasiewicz 불평등(proximal Polyak-Lojasiewicz inequality) 하에 제한 잔여 오차(bounded residual error)에 대한 부분 선형(sublinear) 및 선형(linear) 수렴성을 확보합니다. 이 과정에서 클라이언트의 로컬 데이터 의존성을 명시적으로 동원하여 문제를 정의하며, 클라이언트 간 데이터 분포의 차이로 인한 클라이언트 드리프트 현상(class drift)으로 어려움이 발생합니다.

- **Performance Highlights**: Numerical experiments를 통해 일반적인 데이터셋에 대해 제안된 알고리즘이 최첨단(state-of-the-art) 방법들보다 우수한 성능을 보인다는 것을 입증합니다. 또한, 비볼록 및 비매끄러운 손실 함수에 대한 연합 학습에서 알고리즘의 실용적인 성능 저하를 극복할 수 있는 중요한 진전을 제공합니다.



### Bridging the inference gap in Mutimodal Variational Autoencoders (https://arxiv.org/abs/2502.03952)
- **What's New**: 이 논문에서는 여러 다양한 데이터 모달리티를 통합하여 다루기 위한 새로운 비공식적인 접근법인 Multimodal Variational Autoencoder (VAE) 기반의 모델을 제안합니다. 기존의 Mixure-of-Experts 방식의 한계를 극복하고자, 혼합 집합 없이 결합 분포(joint distribution)와 조건부 분포(conditional distribution)를 학습할 수 있는 다단계 훈련 과정을 도입했습니다. 이를 통해 주어진 데이터의 특징을 보다 정확히 반영할 수 있습니다.

- **Technical Details**: 제안된 모델은 변분 추론(variational inference)을 사용하여 결합 분포를 먼저 모델링하고, 이후 Normalizing Flows를 통해 조건부 분포를 모델링함으로써 진짜 posterior를 더 잘 근사합니다. 이를 위해 공통 정보(shared information)를 추출하고 활용하여 생성된 샘플의 조건적 일관성(conditional coherence)을 개선하려고 합니다. 수학적으로는 M개의 모달리티 X를 통해 여러 가지 데이터 샘플을 관찰하고, 결합 분포와 조건부 분포를 파라메트릭 방식으로 근사합니다.

- **Performance Highlights**: 모델을 평가한 여러 벤치마크 데이터세트에서는 기존 방법들보다 뛰어난 성능을 보였습니다. 특히, 생성 데이터의 품질이 개선되었으며, 다양한 데이터 모달리티 간의 상호작용을 수월하게 이해할 수 있도록 해줍니다. 이 연구는 멀티모달 머신러닝 분야에서 중요한 기여를 할 것으로 기대됩니다.



### CleanSurvival: Automated data preprocessing for time-to-event models using reinforcement learning (https://arxiv.org/abs/2502.03946)
- **What's New**: 이 논문은 'CleanSurvival'이라는 새로운 데이터 전처리 프레임워크를 소개합니다. 이 프레임워크는 생존 분석(survival analysis)에 최적화된 자동화된 데이터 전처리를 제공하여 데이터의 품질을 향상시키는 데 중점을 두고 있습니다. 이는 Q-learning을 활용하여 결측치(imputation), 이상치 탐지(outlier detection) 및 특징 추출(feature extraction) 기술의 조합을 최적화합니다.

- **Technical Details**: CleanSurvival은 강화 학습(reinforcement learning) 기술을 활용하여 데이터를 전처리하는 과정의 다양한 결정을 최적화합니다. 이 프레임워크는 연속형(continuous) 및 범주형(categorical) 변수를 모두 처리할 수 있으며, Cox, 랜덤 포레스트(random forest), 신경망(neural network) 등 여러 시간-사건 모델(time-to-event model)에 적용될 수 있습니다. 오픈 소스 Python 패키지로 제공되며, 여러 생존 분석 데이터셋에서 효과성을 입증하였습니다.

- **Performance Highlights**: 실험 결과는 Q-learning 기반의 데이터 전처리가 기존의 표준 접근방식보다 더 나은 예측 성능을 나타낸다는 것을 보여줍니다. 데이터 전처리를 통해 최대 10배 빠르게 모델을 찾을 수 있으며, 결측치 및 데이터의 노이즈 정도에 따라 효과적인 성과를 보여주고 있습니다. 이 프레임워크는 생존 분석 데이터를 처리하는 데 특히 효과적이며, 향후 연구 방향도 제시하고 있습니다.



### Multimodal Data-Driven Classification of Mental Disorders: A Comprehensive Approach to Diagnosing Depression, Anxiety, and Schizophrenia (https://arxiv.org/abs/2502.03943)
- **What's New**: 본 연구는 전기생리학적 데이터인 EEG와 연령, 성별, 교육 수준, IQ와 같은 사회인구학적 특성을 통합한 다중 모달 데이터 통합의 가능성을 조사합니다. 이는 정신 질환, 특히 조현병, 우울증, 불안증을 진단하는 데 도움을 줄 수 있는 방법을 제시합니다. 이러한 과정은 Apache Spark와 CNN(Convolutional Neural Network) 기술을 활용하여 대규모 데이터를 효과적으로 분석할 수 있는 분류 파이프라인을 개발하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 뇌의 활동과 연결 패턴을 평가하기 위해 EEG의 파라미터인 전력 스펙트럼 밀도(PSD)와 일치(coherence)를 분석합니다. 비교 분석을 통해 일치 특성의 중요성이 부각되며, 이를 통해 분류 정확도와 강 robustness가 크게 향상됨을 보여줍니다. 이런 다중 모달 데이터의 통합은 효율적인 진단 도구 개발을 위한 포괄적인 접근의 중요성을 강조합니다.

- **Performance Highlights**: 연구 결과는 정신 건강 진단의 정밀성, 사용성, 이해도를 높이는 데 기여할 수 있는 데이터 기반 접근법의 가능성을 보여줍니다. 본 연구의 발견은 정신 질환 치료에 있어 창의적인 데이터 주도 접근법을 제시하며, 다채로운 데이터 소스를 활용한 사례의 실용성을 강조합니다.



### Unravelling Causal Genetic Biomarkers of Alzheimer's Disease via Neuron to Gene-token Backtracking in Neural Architecture: A Groundbreaking Reverse-Gene-Finder Approach (https://arxiv.org/abs/2502.03938)
- **What's New**: 본 연구에서는 알츠하이머병(Alzheimer's Disease, AD)의 주요 유전적 기여자를 이해하기 위한 새로운 기술인 Reverse-Gene-Finder를 소개합니다. 이 기술은 신경망 아키텍처에서 뉴런과 유전자 간의 역추적(backtracking) 방법으로, AD 발병을 유발하는 새로운 유전적 바이오마커를 밝히고자 합니다. 특히, 유전자와 뉴런 간의 상관관계를 정교하게 분석하여 AD의 원인 유전자를 식별하는 데 중점을 둡니다.

- **Technical Details**: Reverse-Gene-Finder에는 세 가지 주요 혁신이 포함되어 있습니다. 첫째, 가장 원인 유전자(Most Causal Genes, MCGs)는 가장 원인 뉴런(Most Causal Neurons, MCNs)을 활성화하는 확률이 가장 높다는 관찰을 활용합니다. 둘째, 각 유전을 독립적이고 고유한 개체로 입력층에 표현할 수 있는 유전자 토큰 표기법(gene token representation)을 사용합니다. 마지막으로, 기존 신경망 아키텍처와는 달리, MCNs에서 입력층까지 역추적하는 방법을 개발하여 가장 원인 토큰(Most Causal Tokens, MCTs)과 해당 MCGs를 식별합니다.

- **Performance Highlights**: Reverse-Gene-Finder는 높은 해석 가능성(interpretability)과 범용성(generality)을 가지고 있어, 다른 질병 시나리오에서도 활용될 가능성이 높습니다. 이러한 혁신적인 접근 방식은 AD의 발병 기전에 대한 이해를 돕고, 향후 치료 전략 개발에 기여할 것으로 기대됩니다.



### Quantifying Correlations of Machine Learning Models (https://arxiv.org/abs/2502.03937)
- **What's New**: 이 논문은 다중 머신 러닝 모델 간의 오류 상관관계가 안전 문제를 유발할 수 있음을 탐구한다. 특히, 비슷한 알고리즘이나 데이터셋을 공유하는 모델들이 동시에 잘못된 예측을 할 가능성이 크다는 점을 강조한다. 연구팀은 실제 데이터를 바탕으로 이러한 상관관계를 정량화하고, 모델의 공통된 특성이 시스템적 위험을 증가시킬 수 있음을 발견하였다.

- **Technical Details**: 논문은 머신 러닝 모델 오류의 상관관계를 수학적으로 정의하는 새로운 프레임워크를 제안한다. 이 프레임워크는 모델 성능에 영향을 미치는 랜덤 변수와 불확실성의 원천을 파악하여 서로 다른 모델의 오류가 어떻게 연결되는지를 분석한다. 세 가지 시나리오를 통해 오류 상관관계가 어떻게 발생하는지를 설명하며, 각 시나리오의 구체적인 조건을 제시한다.

- **Performance Highlights**: 모델 오류의 상관관계를 정량적으로 분석한 결과, 안전-critical한 환경에서 모델 간의 오류가 동시에 발생할 가능성이 높다는 것을 확인하였다. 이러한 발견은 조직들이 여러 모델을 배포할 때 위험을 더 잘 평가하도록 돕고, 향후 مقا절 효율적인 리스크 관리 전략 개발에 기여할 것으로 기대된다. 또한, 이는 AI 기반 응용프로그램의 신뢰성과 내구성을 강화하기 위한 연구의 기초를 마련하는 데 중요한 역할을 한다.



### HEP-JEPA: A foundation model for collider physics using joint embedding predictive architectur (https://arxiv.org/abs/2502.03933)
Comments:
          11 pages, 3 figures, 8 tables. Project website: this https URL

- **What's New**: 이 논문에서는 고에너지 입자 충돌기인 대형 하드론 충돌기(LHC)에서 사용될 변환기 기반의 기초 모델을 제시합니다. 모델은 Joint Embedding Predictive Architecture(JEPA)에 영감을 받아 자기 지도 학습(self-supervised learning) 전략을 사용하여 제트(jet)를 분류하도록 훈련됩니다. 우리는 다양한 알려진 입자의 1억 개의 제트로 구성된 JetClass 데이터셋을 사용하여 데이터 중심(data-centric) 접근 방식으로 모델을 사전 훈련합니다.

- **Technical Details**: HEP-JEPA는 파트너너들을 제트 구성 요소의 일부를 마스킹(masking)하여 입력으로 사용합니다. 이 구조는 입력 공간 입력 재구성(input reconstruction)의 필요를 제거하고, 보다 효율적인 계산을 가능하게 하여 유의미한 패턴을 학습하는 데 초점을 맞춥니다. 또한, 각 제트에 대한 입자 집합을 지리적 패치로 분할하고, 이를 고유한 패치 토큰으로 프라이밍하여 예측을 수행합니다.

- **Performance Highlights**: 모델링 성능은 상위 태깅(top tagging) 및 라이트 쿼크 제트와 글루온 제트를 구분하는 두 가지 추가 다운스트림 작업에서 평가되었습니다. HEP-JEPA는 참조 데이터셋에 대해 높은 성능을 보였으며, 특화된 메트릭(metrics) 및 기준과 비교하여 뛰어난 결과를 보여 주었습니다. 이는 고에너지 물리학 분야에서 데이터 기반 발견의 혁신을 가져올 수 있는 잠재력을 가지고 있습니다.



### Rank Also Matters: Hierarchical Configuration for Mixture of Adapter Experts in LLM Fine-Tuning (https://arxiv.org/abs/2502.03884)
- **What's New**: 본 논문에서는 Hierarchical scheme for expert allocation and rank configuration (HILO)를 제안하여, adapter experts의 수와 rank를 동적으로 조절하며 모델 레이어의 복잡성에 맞게 조정하는 방법을 다룹니다. 기존 연구들이 adapter experts의 수에만 초점을 맞춘 데 반해, HILO는 rank 설정의 중요성을 강조하고, trainable parameter를 줄이면서도 모델 정확도를 향상시키는 새로운 접근법을 제공합니다.

- **Technical Details**: HILO는 adapter experts 구성의 효율성을 높이는 데 중점을 두며, 각 레이어에서 adapter experts의 수와 rank를 할당하여 구성합니다. 이 방식은 각 레이어의 representational complexity에 따라 동적으로 조정되며, 이를 통해 기존 방법들보다 높아진 정확도를 달성할 수 있습니다. LoRA와 Mixture of Experts(MoE)를 결합한 기존 접근 방식의 한계를 극복하기 위해, HILO는 더 많은 구성 요소를 고려합니다.

- **Performance Highlights**: 다양한 벤치마크 작업에 대한 실험 결과, HILO는 기존 방법들보다 높은 정확도를 기록하면서도 더 적은 수의 trainable parameters를 도입합니다. 이로 인해 LLMs의 fine-tuning 및 inference 과정에서 효율적인 솔루션을 제공합니다. HILO의 도입으로 인해 연구자들은 앞으로 좀 더 최적화된 PEFT 기법을 기대할 수 있습니다.



### Position: Untrained Machine Learning for Anomaly Detection (https://arxiv.org/abs/2502.03876)
Comments:
          6 pages,0 figure

- **What's New**: 이 논문은 3D 포인트 클라우드 데이터 기반의 무훈련(anomaly detection) 이상 탐지 문제를 다루고 있습니다. 특히, 단일 샘플을 사용하여 이상을 탐지하는 방법을 개발하는 것을 목표로 하고 있으며, 이는 실제 제조 산업의 개인화된 제조 분야에서 중요성을 더해주고 있습니다. 기존의 비지도 학습 방법과의 차별성을 강조하며, 제조 표면과 이상에 대한 사전 지식을 활용하는 접근 방식을 소개합니다.

- **Technical Details**: 무훈련 이상 탐지 방법은 레이블이 없는 데이터나 훈련 데이터에 의존하지 않고 단일 샘플로부터 이상을 탐지합니다. 이러한 방법은 표면의 기하학적 특성(local geometry)이나 통계적 방법을 활용하여 이상을 탐지하는데, 이는 대량의 이상 없는 훈련 샘플의 수요를 줄여줍니다. 하지만 모델 구축 시 사전 지식의 통합이 주요 도전 과제가 됩니다.

- **Performance Highlights**: 현재까지의 연구들은 포인트 클라우드 데이터에서 이상 탐지를 위한 다양한 접근 방식을 제안하고 있습니다. 훈련 기반 방법들은 매우 효과적이나, 새로운 이상에 대한 적응력이 떨어지는 문제를 가지고 있습니다. 반면 무훈련 방법들은 다양한 데이터와 이상 유형에 적응할 수 있어 공장에서 실시간으로 이상을 탐지하는 데 유리합니다. 그러나 여전히 고차원 데이터를 효과적으로 표현하고, 하나의 샘플로부터 의미 있는 특징을 학습하는 데에 어려움을 겪고 있습니다.



### Mirror Descent Actor Critic via Bounded Advantage Learning (https://arxiv.org/abs/2502.03854)
- **What's New**: 이 논문은 Mirror Descent Actor Critic (MDAC)이라는 새로운 actor-critic 스타일의 알고리즘을 제안합니다. MDAC는 연속적인 액션 도메인에 대해 Mirror Descent Value Iteration (MDVI)의 적용에서 성능을 크게 향상시키는데, 이는 비봉합된 단순 구현과 비교할 때 critic의 손실 함수에서 actor의 log-density 항을 경계 지음으로써 이루어집니다. 또한, 이 연구는 Advantage Learning (AL)과의 관계를 통해 MDAC의 이론적 근거를 정립하고, 이 경계 전략이 언제 유효한지를 논의합니다.

- **Technical Details**: 논문에서는 Markov Decision Process (MDP)와 알고리즘의 기본 구성 요소를 설명합니다. MDAC의 핵심은 critic의 손실 함수에서 actor의 log-policy 항을 경계 짓는 것입니다. 이러한 경계 지점은 현재 상태-액션 쌍뿐만 아니라 후속 쌍에서도 이루어지며, TD 타겟 신호에 있어 정당화를 제공합니다. MDVI와 그 변형들은 거울 하강법 (mirror descent) 기반 강화 학습에 해당하며, 이는 다양한 이론 도구로 기존 알고리즘을 분석할 수 있게 해줍니다.

- **Performance Highlights**: MDAC는 강력한 비정규화 및 엔트로피만으로 정규화된 방법들보다 더 나은 성능을 보입니다. 실험 결과, 주어진 경계 함수의 적절한 선택이 MDAC의 성능을 향상시키며, 이는 기존 방법들과의 비교를 통해 명확히 입증됩니다. 주목할 만한 점은, 경계 전략이 경량을 줄이는 데 효과적임을 보여주며, Munchausen RL 알고리즘의 성공적인 경험적 성과를 뒷받침합니다.



### Graph Neural Network-Driven Hierarchical Mining for Complex Imbalanced Data (https://arxiv.org/abs/2502.03803)
- **What's New**: 이번 연구에서는 고차원 비대칭 데이터(high-dimensional imbalanced data)를 처리하기 위한 계층적 마이닝 프레임워크(hierarchical mining framework)를 제안합니다. 이 방법은 깊이 그래프 모델(depth graph model)을 활용하여 전통적인 접근법의 성능 한계를 극복합니다. 데이터셋의 구조화된 그래프 표현을 구축하고 그래프 신경망(GNN) 임베딩을 통합하여 샘플 간의 전역 상호 의존성을 효과적으로 포착합니다.

- **Technical Details**: 연구에서는 소수 클래스(minority class) 특성 패턴(characterization) 및 추출(extraction)을 개선하기 위한 계층적 전략(hierarchical strategy)을 사용합니다. 이를 통해 비대칭 데이터 마이닝에서의 정확성 및 강인성을 향상시킵니다. 실험을 통해 제안된 접근 방법이 기존 방법 대비 패턴 발견(patterndiscovery count), 평균 지원(average support), 소수 클래스 커버리지(minority class coverage) 등 핵심 성능 지표에서 상당한 개선을 보였음을 입증했습니다.

- **Performance Highlights**: 주목할 만한 점은 이 방법이 소수 클래스 특성 추출(minority-class feature extraction) 및 패턴 상관 분석(pattern correlation analysis)에서 우수한 성능을 나타낸다는 것입니다. 깊이 그래프 모델과 계층적 마이닝 전략의 결합이 비대칭 데이터 분석의 효율성과 정확성을 크게 향상시킬 수 있음을 강조하고 있습니다. 이는 고차원 복잡 데이터 처리에 대한 새로운 컴퓨팅 프레임워크를 제공하며, 동적으로 변화하는 비대칭 데이터 및 다중 모달 데이터 응용 프로그램에 대한 확장을 위한 기초를 마련합니다.



### MXMap: A Multivariate Cross Mapping Framework for Causal Discovery in Dynamical Systems (https://arxiv.org/abs/2502.03802)
Comments:
          Accepted by CLeaR 2025; Main manuscript 18 pages, appendix 24 pages, 30 tables

- **What's New**: 본 논문에서는 Partial Cross Mapping (PCM)의 확장을 통해 다변량 환경에서의 인과 관계 추론의 효과를 개선한 multiPCM을 소개합니다. 기존 PCM은 단일 변수의 지연 임베딩에만 국한되어 있었으나, multiPCM은 다변량 임베딩을 활용하여 더 효과적인 인과 관계 식별을 가능하게 합니다. 또한, MXMap이라는 새로운 인과 발견 프레임워크를 제안하여, 복잡한 동적 시스템에서의 인과 구조를 설정하고 정제하는 두 단계의 프로세스를 구현합니다.

- **Technical Details**: MXMap은 첫 번째 단계에서 bivariate CCM을 사용하여 초기 인과 그래프를 생성하고, 두 번째 단계에서는 multiPCM을 적용하여 간접 인과 연결을 제거하며 그래프를 정제합니다. 이를 통해 간단한 비순환상의 인과 그래프뿐만 아니라 순환 구조도 처리 가능합니다. 다변량 지연 임베딩을 활용하여 인과 관계를 더욱 신뢰성 있게 추론할 수 있는 기반을 마련하고 있습니다.

- **Performance Highlights**: 연구에서는 시뮬레이션 데이터 및 ERA5 재분석 날씨 데이터셋을 통한 실험을 통해 MXMap의 효과성을 입증했습니다. 여러 기초 모델들과 비교하였을 때, MXMap은 정확성 및 인과 그래프의 정제 능력에서 뛰어난 성능을 보여주었습니다. 다양한 비선형 동적 시스템에서 multiPCM과 MXMap의 유효성을 종합적으로 평가함으로써 연구의 기여점을 명확히 하였습니다.



### Network-Wide Traffic Flow Estimation Across Multiple Cities with Global Open Multi-Source Data: A Large-Scale Case Study in Europe and North America (https://arxiv.org/abs/2502.03798)
- **What's New**: 이번 연구에서는 Global Open Multi-Source (GOMS) 데이터를 활용하여 네트워크 전반의 교통 흐름 추정(Network-wide Traffic Flow Estimation, NTFE)에서 정확도와 일반성을 동시에 개선하는 방안을 제안합니다. GOMS 데이터는 도로 지형과 인구 밀도 등 지리적 및 인구 통계학적 정보를 포함하고 있어, 여러 도시에서 일관성 있게 수집할 수 있는 이점이 있습니다. 더 나아가, 본 연구에서는 GOMS 데이터를 전통적인 표 형식 대신 지도 이미지 형태로 사용하여 공간적 관계를 더 잘 캡처합니다.

- **Technical Details**: 이 연구에서는 주의(attention) 기반의 그래프 신경망(Graph Neural Network, GNN)을 개발하여 GOMS 지도에서 정보를 추출하고 이를 기존의 센서 데이터와 통합합니다. GOMS 이미지는 도로 형태, 건물 모양, 인구 밀도와 같은 다채로운 정보를 포함하고 있으며, 이를 통해 고급 심층 학습(deep learning) 프레임워크를 통해 교통 흐름을 효과적으로 모델링합니다. 이 방법은 15개 도시를 대상으로 한 대규모 사례 연구를 통해 검증되었습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 다수의 도시에서 일관되고 만족스러운 추정 정확도를 보여주어, NTFE 방법론의 정확도와 일반성 간의 균형을 효과적으로 해결할 수 있음을 입증하였습니다. 이는 공공기관이 각 도시의 상황에 맞춰 로컬화된 NTFE 방법을 수립해야 하는 부담을 덜어주고, 데이터의 효과적인 사용을 가능하게 합니다. 또한, GOMS 지도 이미지를 활용한 접근법은 교통 흐름 패턴을 더 잘 포착할 수 있게 해줄 것으로 기대됩니다.



### Distribution learning via neural differential equations: minimal energy regularization and approximation theory (https://arxiv.org/abs/2502.03795)
- **What's New**: 이 논문에서는 Neural Ordinary Differential Equations (ODEs)를 이용하여 변환 맵이 유도하는 변위를 선형 보간하는 시간 종속 ODE 속도장을 발견했습니다. 이는 복잡한 확률 분포를 근사하는 데 유용하며, 수학적 정제성과 안정성을 제공합니다. 기존의 방법들과는 달리, 이러한 속도장들이 특정 최소 에너지 정규화를 포함하는 훈련 목표의 최적해를 제시하는 신경망 표현을 통해 근사 가능합니다.

- **Technical Details**: 우리는 다양한 변환 맵 T에 대해 ODE 속도장이 근사하는 보간 경로를 구축할 수 있음을 확인하였습니다. 특히, 삼각형(Knothe--Rosenblatt) 맵의 경우, 이러한 경로는 원천 밀도와 목표 밀도의 C^k 노름과 다항식적으로 연관되어 있습니다. 최적화된 신경망의 크기가 주어진 정밀도(
ε)에 따라 제한된 크기를 가짐을 증명하며, 이는 효율적인 변환을 촉진합니다.

- **Performance Highlights**: 이 연구 결과는 Wasserstein 또는 Kullback-Leibler 거리에서 목표 분포와의 근사를 달성할 수 있음을 보여줍니다. 학습된 ODE 속도장의 구조에 대한 명확한 특성과 함께, 특정 정규화된 훈련 목표에 대한 최소화자의 구조를 정량화할 수 있었습니다. 이로 인해, 고급 신경망을 사용하여 최적의 분포 근사를 이룰 수 있는 가능성이 열렸습니다.



### Iterate to Accelerate: A Unified Framework for Iterative Reasoning and Feedback Convergenc (https://arxiv.org/abs/2502.03787)
- **What's New**: 본 논문은 Bregman divergence 및 비유클리드 기하학을 활용한 반복 추론을 위한 통합 프레임워크를 제시합니다. 이는 고전적인 미러 하강(mirror descent) 및 동적 프로그래밍(dynamic programming)을 통합할 뿐만 아니라 현대의 대형 언어 모델에서의 사고 과정(chain-of-thought reasoning)을 설명합니다. 이 연구에서는 지각의 필요성을 강조하며 피드백 메커니즘의 중요성을 이론적으로 증명했습니다.

- **Technical Details**: 우리는 Bregman divergence의 개념을 이용해 비유클리드 설정에서 반복 추론 프로세스를 일반화하는 방법론을 개발했습니다. 이론적으로, 제안된 반복 업데이트 방식은 O(1/t²) 수렴률을 달성할 수 있으며, 이는 참고한 기존의 성능 향상 기법들과 함께 최신 신경 계산(neural computation) 및 최적화(optimization) 기법에 적용될 수 있습니다. 여기서 전제 조건으로는 강완전성(strong convexity) 및 매끄러움(smoothness)이 포함됩니다.

- **Performance Highlights**: 반복적 구조(피드백 아키텍처)는 복잡한 고정점 함수(fixed-point functions)를 효율적으로 근사하는 데 필수적이라는 결과를 보였습니다. 또한, 제안된 방법론은 비율이 1/𝑡²인 수렴 속도를 제공함으로써 iterated processing의 필요성을 더하는데 기여합니다. 이는 현재의 추론 시스템에서 관찰되는 행위를 정당화하는 중요한 이론적 근거를 제공합니다.



### StarMAP: Global Neighbor Embedding for Faithful Data Visualization (https://arxiv.org/abs/2502.03776)
- **What's New**: 이 논문은 Star-attracted Manifold Approximation and Projection (StarMAP)이라는 새로운 이웃 임베딩(neighbor embedding) 방법을 제안하며, 이는 PCA(주성분 분석)의 장점을 활용하여 고차원 데이터를 효과적으로 시각화하는 것을 목표로 하고 있습니다. 기존 방법들이 글로벌 구조를 간과하는 문제를 해결하기 위해 StarMAP는 '스타 매력(star attraction)'이라는 개념을 도입하였습니다. 이 접근법은 글로벌 구조의 충실한 보존을 가능하게 하면서도 이웃 임베딩의 해석 가능성과 계산 효율성을 유지합니다.

- **Technical Details**: StarMAP는 PCA 임베딩을 기반으로 하여 고차원 데이터의 앵커 포인트(고정된 기준점)를 계산한 후, 이 앵커와 데이터를 저차원 공간에 동시에 임베딩하는 방법을 사용합니다. 이 과정에서 앵커 포인트는 '스타'로 참조되며, 최적화 중에 고정됩니다. StarMAP은 이 앵커 포인트에 대한 매력을 적용하여 글로벌 구조를 효과적으로 보존하게 하며, 이전 방법들이 도달하기 힘들었던 목표입니다.

- **Performance Highlights**: 실험 결과, StarMAP는 toy 데이터셋, 단일 세포 RNA 시퀀싱 데이터 및 심층 표현(deep representation) 시각화 작업에서 기존 방법들과 비교하여 효과적임을 보여주었습니다. 예를 들어, Mammoth와 MNIST 데이터셋의 시각화에서 StarMAP는 글로벌 형태와 클러스터 구조를 성공적으로 보존하였으며, 이는 기존의 PCA나 UMAP과는 다른 뛰어난 성능을 제공합니다. StarMAP는 단순하지만 고차원 데이터의 충실한 시각화를 구현하는 매우 효과적인 기술임을 입증했습니다.



### ExpProof : Operationalizing Explanations for Confidential Models with ZKPs (https://arxiv.org/abs/2502.03773)
- **What's New**: 이 논문에서는 기계 학습 모델에 대한 설명 가능성을 적대적인 시나리오에서 실용적으로 구현하기 위한 Zero-Knowledge Proofs (ZKPs)를 사용한 새로운 시스템인 ExpProof을 제안합니다. 이 시스템은 설명이 올바르게 계산되었음을 보장하며, 모델의 기밀성을 유지합니다. ExpProof은 설명 관련 알고리즘 LIME의 개선된 버전을 탐구하고, 신경망 및 랜덤 포레스트에서의 성능을 평가합니다.

- **Technical Details**: ExpProof의 핵심 구성 요소는 암호학적 커밋먼트와 Zero-Knowledge Proofs입니다. 커밋먼트 방법은 모델의 가중치와 설명 매개변수를 고정하여 기밀성을 유지하면서도 투명성을 제공합니다. ZKP를 통해 설명이 사전에 정의된 알고리즘을 사용하여 올바르게 계산되었음을 제안자(은행)가 증명할 수 있으며, 검증자는 추가 정보 없이도 이를 확인할 수 있습니다.

- **Performance Highlights**: ExpProof의 실험 결과는 계산적으로 실행 가능하며, 최대 증명 생성 시간은 1.5분, 검증 시간은 0.12초, 증명 크기는 13KB로 나타났습니다. 이는 신경망 및 LIME을 사용한 상용 기계 학습 모델에서의 성능을 보여줍니다. 이 시스템은 기밀성을 유지하면서도 설명 가능성 측면에서 높은 신뢰성을 제공합니다.



### Adaptive Semantic Prompt Caching with VectorQ (https://arxiv.org/abs/2502.03771)
- **What's New**: 이번 연구는 기존의 정적 유사도 임계값(static threshold) 접근 방식이 다양한 프롬프트(prompt)를 효과적으로 분류하는 데 부족하다는 것을 보여줍니다. 사용자는 이제 VectorQ라는 새로운 프레임워크를 통해 임베딩(embedding) 별로 특정한 임계값 지역(threshold regions)을 학습하여, 각 임베딩의 복잡성과 불확실성에 적응할 수 있습니다. 이는 LLM(generated response) 응답 재사용의 정확도를 높이고, 비용을 절감할 수 있는 기회를 제공합니다.

- **Technical Details**: VectorQ는 후보(candidate)와 가장 가까운 이웃(nearest neighbor) 및 해당 이웃의 캐시된 LLM 응답, 그리고 이와 관련된 임계값 지역을 바탕으로 작동합니다. 시스템은 후보와 가장 가까운 이웃 간의 유사성을 분석하여 캐시 응답 재사용 여부를 결정합니다. 이 프레임워크는 베이지안 샘플링(Bayesian sampling) 기반의 접근 방식을 사용하여, 가장 불확실한 지역을 우선하여 재평가하고, 잘못된 캐시 적중을 최소화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: VectorQ는 다양한 데이터 세트에 대한 평가를 통해 기존의 정적의 임계값보다 12배 더 높은 캐시 적중(rate)과 최대 92%의 오류율 감소를 기록했습니다. 이는 LLM 응답 생성을 위한 비용을 대폭 절감할 수 있음을 의미합니다. 결과적으로 VectorQ는 최신 시스템에 비해 모든 정적 임계값에서 일관되게 우수한 성능을 보여주며, 향후 메모리 관리와 프롬프트 캐싱에 대한 새로운 방향성을 제시합니다.



### Learning Reward Machines from Partially Observed Optimal Policies (https://arxiv.org/abs/2502.03762)
- **What's New**: 이번 연구에서는 Inverse Reinforcement Learning (IRL)의 과제를 해결하기 위해, 최적 정책에서 보상 함수를 추론하는 문제를 다룬다. 보상은 Markov Decision Process (MDP) 상태에 연관된 원자 명제(atomic propositions)와 상관되는 보상 기계(reward machine)로 표현된다고 가정한다. 이 논문은 최소한의 정보로 진정한 보상 기계를 알아내는 것을 목표로 한다.

- **Technical Details**: 본 논문에서는 prefix tree policy라는 개념을 도입하여, 각 MDP의 상태와 가능한 원자 명제의 유한한 시퀀스에 분포된 행동을 연관짓는다. 보상 기계의 동치 클래스(equivalence class)를 특성화하고 이 prefix tree policy에서 추출된 정보를 사용하여 보상 기계를 해결하는 SAT 기반 알고리즘을 제안한다. 알고리즘은 prefix tree policy가 충분한 깊이까지 알려져 있을 때, 정확한 보상 기계를 복구하는 것을 입증한다.

- **Performance Highlights**: 여러 예제를 통해 알고리즘의 효과성을 입증했다. 이 접근법은 부분 관측(partial observability) 문제와 부분 도달성(partial reachability) 문제를 해결하며, 보상이나 기계의 상태를 관찰할 필요 없이 최적 정책의 데이터를 직접 활용하는 방식으로 새로운 통찰을 제공한다. 이는 향후 복잡한 문제 해결과 자율 결정 시스템 발전에 기여할 것으로 기대된다.



### Regularization via f-Divergence: An Application to Multi-Oxide Spectroscopic Analysis (https://arxiv.org/abs/2502.03755)
- **What's New**: 본 논문에서는 convolutional neural networks (CNNs)를 활용하여 행성 표면의 화학 조성을 특성화하는 새로운 접근법을 제시합니다. 특히, 화성(Martian) 환경에서 수집한 분광 데이터(spectroscopic data)를 기반으로 암석 샘플의 다산화물 무게(multi-oxide weights)를 예측하는 과제를 다룹니다.

- **Technical Details**: 이 문제는 다중 타겟 회귀(multi-target regression) 문제로 설정되며, f-divergence에 기반한 새로운 정규화 방법을 제안합니다. f-divergence 정규화는 예측값과 잡음(target) 간의 분포 불일치를 제한하며, 과적합(overfitting)을 줄이고, 예측 및 목표 분포 사이의 다이버전스(divergence)가 클 경우 패널티를 부여하는 보조 손실 함수(auxiliary loss function) 역할을 수행합니다.

- **Performance Highlights**: Mars-like 환경에서 수집된 스펙트라를 활용한 실험 결과, 제안된 f-divergence 정규화 방법이 전통적인 정규화 방법인 L1, L2 및 dropout보다 우수한 성능을 보였습니다. 또한, f-divergence 정규화와 기존 정규화를 결합했을 때 성능이 더욱 향상되어 독립적으로 사용된 정규화 방법들보다 높은 성능을 달성했습니다.



### PRISM: A Robust Framework for Skill-based Meta-Reinforcement Learning with Noisy Demonstrations (https://arxiv.org/abs/2502.03752)
Comments:
          8 pages main, 19 pages appendix with reference. Submitted to ICML 2025

- **What's New**: 이 논문은 Skill-Based Meta-Reinforcement Learning (Meta-RL)에 대한 새로운 접근법인 Prioritized Refinement for Skill-Based Meta-RL (PRISM)을 제안합니다. PRISM은 noisy offline data 근처에서 탐색을 통해 온라인 궤적을 생성하고 이를 offline 데이터와 결합함으로써 효과적인 스킬 학습을 보장합니다. 이 방법은 노이즈의 영향을 해결하여 스킬 학습의 안정성을 확보하고 긴 수명의 과제를 해결하는 데 우수한 성능을 발휘합니다.

- **Technical Details**: PRISM은 exploration policy를 활용하여 noisy offline data 근처에서 유용한 궤적을 발견, 이를 통해 고품질 데이터를 추출하여 task-relevant skills를 학습합니다. 두 가지 주요 기여를 포함하는 이 프레임워크는 (1) 우선 순위가 매겨진 스킬 정제 프레임워크로, online과 offline 데이터셋에서 스킬을 통합된 방식으로 학습할 수 있도록 합니다. (2) 최대 반환 재라벨링 기법을 통해 noisy offline 궤적을 평가하며, 이를 통해 데이터 품질을 보장합니다.

- **Performance Highlights**: PRISM은 noisy 환경에서도 데이터로부터 학습의 안정성을 확보하여 효과적인 스킬 학습을 달성합니다. 특히, Maze2D 환경에서 noisy offline 궤적에서 스킬을 성공적으로 정제하여 보이지 않는 과제를 해결하는 데 성공했습니다. 이러한 접근 방식은 스킬 기반 Meta-RL의 강건성과 일반화 가능성을 크게 향상시켜 실제 noisy 시나리오에서도 신뢰할 수 있는 성과를 입증합니다.



### Principal Curvatures Estimation with Applications to Single Cell Data (https://arxiv.org/abs/2502.03750)
Comments:
          To be published in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)

- **What's New**: 이 논문에서는 단일 세포 전사체 시퀀싱(single-cell transcriptomic sequencing, scRNAseq) 데이터를 분석하기 위한 데이터 기반 방법인 Adaptive Local PCA (AdaL-PCA)를 제안합니다. AdaL-PCA는 데이터 매니폴드(data manifold)에서의 본질 곡률(intrinsic curvature) 추정을 통해 세포의 차별화(cell differentiation) 과정을 밝혀낼 수 있는 가능성을 확인합니다. 기존 방법에 비해 데이터 밀도 변화에 대한 적응성을 강화하여 다양한 매니폴드에서 안정성을 보장합니다.

- **Technical Details**: 이 방법은 Locally PCA를 바탕으로 하여, 각 포인트 주위의 이웃을 선택하고 이들을 중심으로 하는 데이터 행렬을 형성합니다. 여기서 첫 두 개의 고유벡터는 국소적인 접선 공간(tangent space)의 기준으로 선택되며, 세 번째 고유벡터는 표면의 법선 벡터(normal vector) 역할을 합니다. AdaL-PCA는 설명된 분산 비율을 사용하여 적절한 매개변수를 동적으로 조정하여 곡률(curvature)을 추정합니다.

- **Performance Highlights**: AdaL-PCA는 샘플링 된 표면(surfaces)에서 최첨단 성능을 입증하였으며, PHATE (Potential of Heat-diffusion for Affinity-based Trustworthiness Embeddings) 내 장을 결합하여 단일 세포 RNA 시퀀스 데이터에서 핵심 변화를 식별할 수 있습니다. 이를 통해 세포의 차별화 방향성을 제시하는 주 성질(curvature)을 추정할 수 있었습니다.



### PINS: Proximal Iterations with Sparse Newton and Sinkhorn for Optimal Transpor (https://arxiv.org/abs/2502.03749)
Comments:
          12 pages, 5 figures

- **What's New**: 이 연구에서는 대규모 최적 수송 문제를 효율적으로 해결하기 위한 PINS(근접 반복 프로세스 및 희소 뉴턴 및 싱크혼 방법)를 제안합니다. PINS는 정밀도와 효율성을 크게 향상시키는 새로운 접근법으로, 불안정성 문제를 해결하고 일관된 수렴 속도를 보장합니다. 특히, PINS는 최적화 문제를 두 단계로 나누어 해결하여 대규모 데이터 세트에서도 뛰어난 성능을 발휘할 수 있습니다.

- **Technical Details**: PINS는 두 단계의 알고리즘을 통하여 최적화 문제를 해결합니다. 첫 번째 단계에서는 싱크혼 알고리즘(Sinkhorn algorithm)을 사용하여 엔트로피 정규화 된 최적 수송 문제에 대한 근사 해를 효율적으로 계산합니다. 두 번째 단계에서는 뉴턴 방법(Newton's method)을 활용하여 해를 정교화하고, 반복적인 희소화 기법을 적용하여 계산 효율성을 극대화합니다.

- **Performance Highlights**: PINS는 세 가지 주요 이점을 가지고 있습니다. 첫째, 고도의 정확도로 최적 솔루션을 계산하여 엔트로피 정규화 방법의 근사 한계를 극복합니다. 둘째, 뉴턴 단계에서의 반복 희소화는 전체 효율성을 크게 향상시켜 누적적인 계산 비용 절감 효과를 발생시킵니다. 셋째, PINS는 하이퍼파라미터 선택에 대한 강력한 로버스트성(robustness)을 보여 줌으로써, 기존 방법들이 겪는 민감성을 줄입니다.



### Multiple Invertible and Partial-Equivariant Function for Latent Vector Transformation to Enhance Disentanglement in VAEs (https://arxiv.org/abs/2502.03740)
Comments:
          24 pages, 21 figures

- **What's New**: 본 논문에서는 Variational AutoEncoder(VAE)에서 훈련된 정보를 이해하고 재사용하기 위한 핵심 문제인 분리 학습(disentanglement learning)을 다룹니다. 특히, inductive bias를 주입하기 위한 새로운 방법인 Multiple Invertible and partial-equivariant transformation(MIPE-transformation)을 제안하며, 이 방법은 입력에서 잠재 공간으로의 변환에서 일정 부분의 equivariance를 유지하면서도 잠재 벡터 변환의 invertibility를 보장합니다.

- **Technical Details**: MIPE-transformation의 두 가지 주요 구성 요소는 Invertible and partial-equivariant transformation(IPE-transformation)과 Exponential Family conversion(EF-conversion)입니다. IPE-transformation은 잠재 벡터 간의 변환을 대칭 행렬 지수 함수로 제한하여 부분-equivariant하게 구현하며, EF-conversion은 불확실한 잠재 벡터 분포를 학습 가능한 형태로 전환합니다. 또한, 여러 IPE-transformation과 EF-conversion을 통합하는 아키텍처를 제안합니다.

- **Performance Highlights**: MIPE-transformation은 3D Cars, 3D Shapes, dSprites와 같은 다양한 데이터세트에서 기존 VAE의 분리 성능을 현저히 향상시켰습니다. 귀무가설을 통해 IPE-transformation과 EF-conversion의 결합이 분리된 표현 학습에 미친 긍정적인 영향을 실증적으로 분석하였습니다. 본 연구는 분리 학습을 개선하기 위한 효과적인 방법론을 제시하며, 다양한 최신 VAE에 폭넓게 적용될 수 있는 가능성을 보여줍니다.



### Mitigating the Participation Bias by Balancing Extreme Ratings (https://arxiv.org/abs/2502.03737)
Comments:
          In Proceedings of the ACM Web Conference 2025,15 pages

- **What's New**: 이 논문은 참여 편향(participation bias) 하에서의 강건한(rugged) 평점 집계(rating aggregation) 문제를 다룹니다. 전통적인 평균 방법이 특정 평점은 누락되는 경우로 인해 왜곡될 위험이 있다는 점을 지적합니다. 이 연구에서 제안한 두 가지 집계기(aggregator)는 알려진(sample size가 알려진 경우) 경우와 알려지지 않은 경우에 대처합니다.

- **Technical Details**: 첫 번째 집계기인 Balanced Extremes Aggregator는 극단적인 평점을 균형적으로 조합하여 숨겨진 평점을 추정합니다. 두 번째 집계기인 Polarizing-Averaging Aggregator는 표본 크기가 무한히 커질 때 최적의 성능을 보입니다. 이 방법론은 각 평점의 보고 확률이 독립적이라고 가정하며, 참여 확률에 따라 평점이 비워질 수 있음을 고려합니다.

- **Performance Highlights**: 제안된 집계기들은 수치 결과에서 참여 편향을 완화하는 데 있어 단순 평균(simple averaging) 및 스펙트럴 방법(spectral method)보다 우수하다는 것을 입증했습니다. 또한 실세계 데이터셋에서 이들 집계기의 효과성이 추가로 검증되었습니다.



### Optimal Control of Fluid Restless Multi-armed Bandits: A Machine Learning Approach (https://arxiv.org/abs/2502.03725)
- **What's New**: 이 논문에서는 상태 방정식이 아핀(affine) 또는 이차(quadratic)인 유체 레스틀레스 다중 암 대안(FRMAB) 문제를 최적 제어하기 위해 기계 학습 접근 방식을 제안합니다. FRMAB 문제의 기본 속성을 파악하여 효율적인 기계 학습 기반 알고리즘을 설계하였고, 이 알고리즘을 통해 다양한 초기 상태에서 여러 인스턴스를 해결하여 포괄적인 학습 세트를 생성합니다. 최적 분류 결정 트리(Optimal Classification Trees, OCT-H)를 사용하여 상태 피드백 정책을 학습하고, 이를 기계 유지관리, 전염병 제어 및 어업 관리 문제에 적용하여 성능을 테스트합니다.

- **Technical Details**: FRMAB 모델은 유한 시간 구간 T<∞를 가지며, 각 프로젝트는 상태 변수로 표현됩니다. 각 프로젝트의 상태 진화는 1차 자율 Ordinary Differential Equation(ODE)을 따른다고 명시되어 있습니다. 연구는 보통 결정론적 근사(deterministic approximation)를 활용하여 스토캐스틱(stochastic) 문제의 복잡성을 경감하는 방법을 제시하며, 이를 통해 최적 제어 문제를 해결하기 위한 유용한 구조를 제공합니다. OCT-H 알고리즘은 의사결정 트리 기법을 활용한 기계 학습 접근법으로, 연속적으로 변하는 최적 제어 문제에 대한 최적 정책 학습에 적합합니다.

- **Performance Highlights**: 제안하는 방법을 통해 높은 품질의 상태 피드백 정책을 생성하며, 유체 문제에 대한 직접 수치 알고리즘 대비 최대 2600만 배의 속도 향상을 달성했습니다. 이는 다양한 초기 상태에서 반복적으로 최적 제어 문제를 해결해야 하는 실제 응용에서 중요한 성능 개선을 보여줍니다. 또한, 유체 대안에서 얻어진 혁신적인 접근 방식은 기존의 스토캐스틱 모델을 효과적으로 처리하는 것을 가능하게 합니다.



### On the Expressive Power of Subgraph Graph Neural Networks for Graphs with Bounded Cycles (https://arxiv.org/abs/2502.03703)
- **What's New**: 이 논문은 그래프 신경망(GNNs)의 한계 진단과 그 한계를 극복하기 위한 연구를 다룹니다. 특히, k-hop 서브그래프 GNN을 통해 인접 노드로부터의 정보를 집계하고 서브그래프 구조를 통합하는 방법을 제안합니다. 이에 따라, 사이클(파사드) 길이가 2k+1을 초과하지 않는 그래프에 대해 모든 permutation-invariant/equivariant 지속 함수(continuous function)를 근사할 수 있음을 입증하였습니다.

- **Technical Details**: 해당 연구에서는 메시지 전달 메커니즘(message-passing mechanism)에 기초하여 노드 표현을 개선하는 방법을 제시합니다. 노드의 특징 벡터는 이웃 노드로부터 정보를 집계하여 반복적으로 업데이트 됩니다. 각 반복 또는 계층에서 업데이트는 두 단계의 통합(local transformation)과 집계(aggregation)로 구성되어 있으며, 여기서 학습 가능한 함수가 적용됩니다.

- **Performance Highlights**: 수행된 수치 실험은 정보 집계 거리(information aggregation distance)와 사이클 크기(cycle size) 간의 관계를 검증하며, k-hop GNN의 성능을 입증합니다. 이 연구는 그래프 데이터에 대한 GNN의 설명력을 향상시키는데 기여하고, 다양한 분야에서의 응용 가능성을 시사합니다.



### How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies (https://arxiv.org/abs/2502.03698)
- **What's New**: 이번 연구는 Learning from Demonstration (LfD) 알고리즘의 적대적 공격에 대한 포괄적인 연구를 진행한 최초의 연구로, 최신 행동 클로닝 알고리즘의 취약성을 중점적으로 분석합니다. LfD 알고리즘은 로봇 조작 작업에서 강력한 성능을 보였지만, 적대적 공격에 대한 취약성은 크게 간과되어왔습니다. 우리는 이러한 알고리즘이 다중 모달 액션 분포 및 외란에 대한 안정적인 처리를 해야 하는 고유한 도전 과제를 가지고 있음을 밝힙니다.

- **Technical Details**: 본 연구에서는 Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), VQ-Behavior Transformer (VQ-BET)와 같은 여러 LfD 알고리즘의 취약성을 조사하였습니다. 특히 IBC와 DP는 명시적이지 않은 정책으로, 공격에 대한 접근 방식이 더욱 세밀해져야 함을 강조합니다. 우리는 이 알고리즘에 대한 새로운 샘플링 기반 공격 방법을 제안하며, 이를 통해 보다 효율적인 공격을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 대부분의 LfD 알고리즘은 적대적 방해에 대해 높은 취약성을 보였습니다. 특히, Randomized Smoothing 방어 기법의 효용이 제한적임을 발견하였으며, 이는 복잡한 제어 작업에서 효과적이지 않음을 의미합니다. 마지막으로, Diffusion Policy는 실험한 정책 중 가장 강력한 내성을 보였으며, 이는 다단계 예측 과정에서 기인할 수 있음을 제시합니다.



### DocMIA: Document-Level Membership Inference Attacks against DocVQA Models (https://arxiv.org/abs/2502.03692)
Comments:
          ICLR 2025

- **What's New**: 이 논문에서는 Document Visual Question Answering (DocVQA) 모델을 겨냥한 신종 멤버십 추론 공격(Membership Inference Attack, MIA) 두 가지를 소개합니다. 이는 모델 아키텍처와 매개변수에 대한 완전한 접근 권한을 가진 화이트박스(white-box) 설정과, 모델의 출력만을 이용하는 블랙박스(black-box) 설정을 위해 설계되었습니다. 공격자는 추가 데이터에 접근할 수 없는 상황을 가정하였으며, 이는 더 현실적인 시나리오임에도 불구하고 더 많은 도전 과제를 제공합니다.

- **Technical Details**: 제안된 공격 방법인 Document-level Membership Inference Attack (DocMIA)은 동일 문서가 훈련 세트에 여러 번 등장하는 상황에서 발생하는 문제를 해결합니다. 자동 회귀(auto-regressive) 출력에서 logit 기반의 전통적인 지표를 추출하는 것이 어려운 점을 극복하기 위해, DocMIA를 위한 구별 가능한(feature discriminative) 메트릭을 생성하는 새로운 방법론을 제안합니다. 이 연구는 기존 모델에 대한 보편적인 공격 기술 외에도, 화이트박스 및 블랙박스 환경에서 추가 데이터 없이 수행 가능합니다.

- **Performance Highlights**: 세 가지 멀티모달 DocVQA 모델과 두 개의 데이터셋에서 비교 평가를 수행한 결과, 본 연구의 공격 방법은 기존의 최첨단 멤버십 추론 공격을 능가하는 성능을 보여준 바 있습니다. 특히, 세 가지 접근 방식(가장 간단한 레이어 미세 조정(Fine-tuning), LoRA 방식을 적용한 미세 조정, 이미지 그래디언트 사용)을 통해 DocMIA의 효과를 입증했습니다. 이와 같은 혁신적인 접근법은 DocVQA 모델의 개인 정보 보호 위험을 강조하면서, 해당 분야의 기존 연구와 차별화되는 점을 부각시킵니다.



### Variational Control for Guidance in Diffusion Models (https://arxiv.org/abs/2502.03686)
Comments:
          8 pages in main text. Total of 20 pages

- **What's New**: 이 연구에서 제안하는 Diffusion Trajectory Matching (DTM)은 기존의 방법들의 한계를 극복하기 위해 Variational Inference (변분 추론) 및 제어 관점에서 Diffusion 모델을 재조명합니다. 기존의 Classifier Guidance (분류기 안내) 또는 Classifier-Free Guidance (비분류기 안내) 기법들은 추가 모델 학습을 요구하거나 표준화된 가정에 기반하여 샘플 품질을 해치는 경우가 많았습니다. DTM은 이러한 기법들을 통합하여 추가 학습 없이도 고성능을 발휘하도록 설계되었습니다.

- **Technical Details**: DTM 프레임워크는 Guided Diffusion Dynamics (유도 확산 동역학)를 마르코프 체인으로 모델링 하여 통제 신호를 Variational Parameters (변분 매개변수)로 정의합니다. 이로써 생성된 샘플들이 비조건부 샘플 매니폴드에 가깝게 유지되도록 최적화를 적용하며, 원하는 단말 조건을 만족하는 과정을 보장합니다. DTM은 또한 Non-linear Diffusion Trajectory Matching (NDTM)으로 구체화되며, 기존의 최첨단 Diffusion 모델 샘플러와 잘 통합됩니다.

- **Performance Highlights**: NDTM은 ImageNet-256 및 FFHQ-256과 같은 데이터셋에서 슈퍼 해상도 및 인페인팅과 같은 도전적인 문제들에 대해 이전의 최신 기법들을 초월하는 성능을 보여주었습니다. 예를 들어, 이미지 노이즈 제거 문제에서 DTM을 통해 FID 점수 34.31을 달성하며, 기존의 최적 사전 훈련 방법의 FID 78.07를 크게 개선했습니다. 연구팀은 향후 코드도 공개할 계획이라고 밝혔습니다.



### Chaos into Order: Neural Framework for Expected Value Estimation of Stochastic Partial Differential Equations (https://arxiv.org/abs/2502.03670)
- **What's New**: 이 논문에서는 확률적 부분 미분 방정식(Stochastic Partial Differential Equations, SPDEs)의 새로운 신경망 프레임워크를 소개합니다. 이 접근 방식은 기존의 이산화(discretization)를 제거함으로써, 임의의 시공간(spatio-temporal) 지점에서 기대값(expected values)을 직접 추정할 수 있게 해줍니다. 전통적인 기법에서의 계산 비효율성을 극복하면서 고차원 문제에 대한 해결 가능성을 제시합니다.

- **Technical Details**: 논문에서는 Loss Enforced Conditions (LEC)와 Model Enforced Conditions (MEC)의 두 가지 신경망 아키텍처를 비교합니다. LEC는 손실 함수(loss function)에 물리적 제약 조건을 통합하고, MEC는 이러한 제약을 네트워크 구조에 직접 내장합니다. 이를 통해 저자는 스토캐스틱 열 방정식(stochastic heat equation), Burgers 방정식, Kardar-Parisi-Zhang(KPZ) 방정식의 다양한 차원에서 성능을 비교합니다.

- **Performance Highlights**: LEC는 잔여 최소화(residual minimization)와 일반화(generalization)에서 우수한 성능을 보이는 반면, MEC는 초기 조건(initial conditions)을 정확하게 지키고 경계 조건(boundary condition) 강제화에서 높은 정확성을 자랑합니다. 신경망 기반의 SPDE 솔버의 잠재력을 강조하며, 특히 기존 기법이 실패했던 고차원 문제에서 그 가능성을 보여줍니다.



### Unrealized Expectations: Comparing AI Methods vs Classical Algorithms for Maximum Independent S (https://arxiv.org/abs/2502.03669)
Comments:
          24 pages, 7 figures, 8 tables

- **What's New**: 이번 논문에서는 AI(인공지능) 기반의 방법과 전통적인 CPU 기반 방법을 비교하여 Maximum Independent Set (MIS) 문제에 대한 성능을 분석합니다. 특히, GPU를 기반으로 한 AI 방법이 KaMIS라는 상태-of-the-art 전통 솔버보다 뛰어나지 않음을 발견했습니다. 더 나아가, AI 방법이 단순히 임의의 휴리스틱(degree-based greedy)과 유사한 성능을 보이며, 추후 처리(post-processing)를 통해서도 CPU 기반 솔버보다 성능이 저조한 것으로 나타났습니다.

- **Technical Details**: 결정적 문제를 해결하기 위한 CO(combinatorial optimization) 방법론으로 AI 기반 알고리즘이 주목받고 있습니다. 하지만 본 연구에서는 NON-Backtracking AI 방법이 최적의 해를 찾기보다는 가장 단순한 degree-based greedy 접근과 유사한 결론에 도달하게 됨을 발견했습니다. 실험은 다양한 그래프 유형에서 AI 알고리즘이 KaMIS와 비교하여 저조한 성능을 보이는 경향을 분석하는 데 주력하였습니다.

- **Performance Highlights**: KaMIS는 희소한 랜덤 그래프에서 거의 모든 실험에서 AI 기반 알고리즘과 비교하여 강력한 성능을 보였습니다. 입력 그래프 크기가 커지거나 조밀해질수록 KaMIS의 성능 우위가 더욱 뚜렷해졌습니다. 이 결과는 Coja-Oghlan & Efthymiou(2015)에서 제안된 다항식 시간 알고리즘에 대한 상한 추정과 모순되는 것으로 볼 수 있습니다.



### Privacy-Preserving Generative Models: A Comprehensive Survey (https://arxiv.org/abs/2502.03668)
- **What's New**: 이번 연구는 GANs와 VAEs에 대한 프라이버시 및 유틸리티 관점을 체계적으로 분류한 첫 번째 설문조사입니다. 100개의 연구 논문을 분석하여 프라이버시를 보존하는 생성 모델에 대해 심도 있는 논의를 제시하고 있습니다. 이는 새로운 연구자들이 이 분야에서 공통적인 과제와 미래 연구 방향에 대한 통찰을 제공받을 수 있도록 돕습니다.

- **Technical Details**: 이 논문은 프라이버시 보장 생성 모델의 최신 동향을 포괄적으로 리뷰하며, GANs와 VAEs를 중심으로 합니다. 또한 차별적 프라이버시(differential privacy)와 그 주요 특성(합성(composition) 및 사후 처리(post-processing))을 설명하며, 생성 모델의 보안성과 프라이버시 접근 방식을 분류합니다. 주요 기여 중 하나는 프라이버시 및 유틸리티 메트릭스에 대한 새로운 세분화를 제공하는 것입니다.

- **Performance Highlights**: 이 연구는 GAN과 VAE의 성능을 강조하며 이들 모델이 합성 데이터 생성과 관련하여 뛰어난 결과를 나타내고 있음을 보여줍니다. 추가적으로, 기존 문헌과 비교하여 현재까지의 프라이버시 및 유틸리티 메트릭스를 종합적으로 정리하여 향후 연구 방향과 도전과제를 제시하고 있습니다. 다양한 프라이버시 공격 유형을 다루고 공격자의 모델에 대한 선입견에 따른 민감한 정보 유출 가능성도 탐구합니다.



### Advancing Weight and Channel Sparsification with Enhanced Saliency (https://arxiv.org/abs/2502.03658)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문은 IEE(Iterative Exploitation and Exploration)라는 혁신적인 접근 방식을 소개하여 비구조적 및 구조적 희소성(sparsity)에 대해 중요도 기준을 향상시키는 방법을 제안합니다. 이 방법은 모델을 활성 구조와 탐험 공간으로 나누어 활성 구조를 최적화하고 탐험 공간에서 매개변수를 다시 평가하여 통일된 중요도 기준에 따라 재통합합니다. 실험을 통해 기존의 단순한 중요도 기준을 개선하여 최첨단 성능과 훈련 비용 절감을 달성할 수 있음을 보여줍니다.

- **Technical Details**: IEE 모델은 두 가지 단계로 구성됩니다: 활용(Exploitation)과 탐험(Exploration) 단계입니다. 활용 단계에서는 현재의 활성 구조를 최적이라고 가정하고 이를 수렴하기 위해 훈련합니다. 탐험 단계에서는 이 최적성을 의심하며 중요도 점수를 바탕으로 덜 중요한 매개변수를 제거하고, 탐험 공간의 모든 매개변수를 잠시 "재활성화"하여 몇 번의 반복 훈련을 실시하여 이 매개변수의 성능 잠재력을 미리 살펴보는 과정이 포함됩니다.

- **Performance Highlights**: ResNet50 모델을 사용한 ImageNet 데이터셋의 실험 결과, IEE 방법은 90% ERK 희소성에서 기존 기법보다 Top-1 정확도를 1.3% 향상시키는 성과를 보였습니다. 특히 HALP와 비교하여 훈련 비용을 70% 이상 절감하고 더 빠르고 정확한 프루닝된 모델을 얻었습니다. 이는 단순히 기존의 Magnitude 및 Taylor 기준을 개선하여 최첨단 결과를 달성할 수 있음을 보여줍니다.



### Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics (https://arxiv.org/abs/2502.03654)
Comments:
          8 pages, excluding references and appendix

- **What's New**: 이번 논문에서는 Gompertz Linear Unit (GoLU)라는 새로운 self-gated activation function을 도입합니다. GoLU는 Gompertz 함수의 비대칭성을 활용하여 기존의 활성화 함수들보다 효과적으로 잠재 공간의 분산을 줄이는 동시에 강력한 gradient flow를 유지합니다. 여러 과제를 통한 실험 결과, GoLU는 최신 활성화 함수에 비해 우수한 성능을 보여주며, 이는 현재의 활성화 함수들에 대한 견고한 대안으로 자리잡고 있습니다.

- **Technical Details**: GoLU는 Gompertz 함수를 게이팅 메커니즘으로 사용하는 self-gated 활성화 함수입니다. 이 함수는 exponentials를 사용하여 무한히 미분 가능하며, ReLU와 그 변형들과는 달리 매끄럽고 비 모노토닉한 특성을 지닙니다. Gompertz 함수의 비대칭성은 Gumbel 분포의 근본적인 비대칭성에서 기인하여 출력의 세기가 다른 gated activation function들에 비해 압축된 효과를 내게 합니다.

- **Performance Highlights**: 다양한 과제를 대상으로 한 실험 결과, GoLU는 기존 self-gated 활성화 함수들보다 더 효과적으로 잠재 표현의 분산을 줄입니다. 이는 모델의 활성화 출력에서 노이즈를 줄여주어, 필수적인 특징을 보존하면서도 과적합(overfitting)을 방지하는 데 도움이 됩니다. GoLU는 각기 다른 데이터 세트에서 탁월한 성능을 보여주며, 이는 고차원 데이터 처리에 대한 향상된 능력을 나타냅니다.



### The Cost of Shuffling in Private Gradient Based Optimization (https://arxiv.org/abs/2502.03652)
Comments:
          54 pages, 6 figures

- **What's New**: 본 연구에서는 차별적 개인 정보 보호(Differentially Private, DP)를 고려한 볼록 경험적 위험 최소화(convex empirical risk minimization, ERM)에 대해 다룹니다. 기존의 DP-SGD 알고리즘은 이론적으로 잘 확립되어 있으나, 실제 구현에서는 데이터의 시퀀셜하게 순회하는 섞인(Shuffled) 그래디언트 방법에 의존하고 있습니다. 이러한 방법들은 개인 정보 보호와 정확성 간의 이론적 교환이 잘 이해되지 않아 이론과 실제 간의 간극이 발생하고 있습니다.

- **Technical Details**: 연구에서는 반복을 통한 개인 정보 보호 증폭(Privacy Amplification by Iteration, PABI)과 Stein의 보조정리가 적용되어 DP-ShuffleG의 최초 경험적 초과 위험 경계(excess risk bound)를 제공합니다. 연구 결과에 따르면 데이터 섞기 기술이 DP-SGD에 비해 DP-ShuffleG의 경험적 초과 위험을 악화시키는 것으로 나타났습니다. 이에 대한 해결책으로 공개 데이터 샘플을 활용한 하이브리드 접근 방식인 Interleaved-ShuffleG를 제안합니다.

- **Performance Highlights**: Interleaved-ShuffleG는 개인적 샘플과 공개 샘플을 번갈아 사용하여 경험적 초과 위험을 효과적으로 줄여줍니다. 새로운 최적화 프레임워크는 대체 목표(surrogate objectives), 적응형 잡음 주입(adaptive noise injection), 비슷하지 않은 측정 기준(dissimilarity metric)을 도입하였습니다. 다양한 데이터셋과 작업에 대한 실험을 통해, Interleaved-ShuffleG의 성능이 여러 기준선에 비해 우수함을 입증하였습니다.



### Efficient Optimal PAC Learning (https://arxiv.org/abs/2502.03620)
- **What's New**: 최근 Hanneke [2016b]와 Larsen [2023]의 연구에 의해 이진 분류 설정에서 최적의 PAC 학습자가 개발되었습니다. 이 학습자들은 각각 똑똑한 결정론적 서브샘플링(sampling) 방식과 Breiman [1996]의 고전적인 배깅(bagging) 휴리스틱을 활용하여 도출되었습니다. 논문에서는 이러한 최적 PAC 학습자들이 경험적 위험 최소화(empirical risk minimization) 알고리즘과 연결된 계산 비용을 다르게 평가할 수 있는 대안적인 관점을 제공하고자 합니다.

- **Technical Details**: PAC 모델(Probably Approximately Correct)은 1984년 Valiant에 의해 소개되었으며, 기계 학습 이론의 핵심 개념으로 자리 잡고 있습니다. 이 모델은 학습 알고리즘이 레이블이 있는 학습 예제 집합으로부터 학습하여 보지 못한 예제의 레이블을 높은 정확도로 예측할 수 있어야 한다는 아이디어를 기반으로 합니다. 특히 이 논문은 PAC 학습의 실현 가능(realizable) 설정을 고려하며, 진정한 개념(true concept)이 유한 VC 차원(finite VC-dimension)을 가진 가설 클래스(hypothesis class) 내에 포함되어 있다고 가정합니다.

- **Performance Highlights**: 이 연구에서는 경험적 위험 최소화와 연결된 계산 비용의 무게 중심이 다른 최적 PAC 학습자가 존재함을 보여줍니다. 이 최적 학습자는 경험적 위험 최소화 알고리즘에 의해 초래된 계산 비용을 줄일 수 있는 새로운 무역(offering a different tradeoff)을 제시합니다. 이러한 접근은 최적의 학습자가 제공하는 샘플 복잡도(sample complexity)와 관련하여 새로운 통찰을 제공합니다.



### Swarm Characteristic Classification using Robust Neural Networks with Optimized Controllable Inputs (https://arxiv.org/abs/2502.03619)
- **What's New**: 이 논문에서는 자율 항공기, 무장선 등의 공격을 예측하고 대응하기 위해 선행된 연구에서 획득한 데이터셋을 확장하여 보다 견고한 신경망(NN) 모델을 개발하였다. 향후 military engagements에서의 불확실성을 고려한 데이터의 다양화를 통해 NN 분류 정확도를 개선하였다. 또한, 공격자 반응을 유도하여 분류의 정확도를 극대화할 수 있는 새로운 최적화 프레임워크를 제안하였다.

- **Technical Details**: 연구는 이전 연구의 시나리오를 바탕으로 하며, 특정 전술을 사용하는 적의 경과를 예측하기 위해 다변량 시간 연속 데이터(multivariate time series data)를 이용한 감독 신경망 시간 연속 분류(NN TSC) 기법을 사용하였다. 검증된 변수인 defender의 수와 초음파 사물의 움직임, 측정 잡음과 같은 불확실성 요소를 통해 훈련된 NN의 성능을 평가하였다. 또한 적의 전술을 예측하기 위한 다중 클래스 출력(multi-class outputs)으로 NN의 성능을 개선하였다.

- **Performance Highlights**: 훈련된 NN은 각기 다른 운영 조건 하에서 우수한 성능을 보였으며, 데이터셋의 변형을 통해 NN의 분류 정확도를 극대화할 수 있었다. Robust NN은 자원과 궤도 제약에 대한 적합성을 확보하면서도 효율적인 분류를 가능하게 하여 실제 방어 시나리오에서의 활용도를 높였다. 이 연구에서 제안된 최적화된 방어자 동작 프레임워크는 주어진 분류 신뢰 수준에 도달하기 위해 필요한 최소 방어자 수를 파악하는 데 기여할 수 있다.



### The Logical Implication Steering Method for Conditional Interventions on Transformer Generation (https://arxiv.org/abs/2502.03618)
- **What's New**: 이번 논문에서는 사전 훈련된 트랜스포머 모델에 논리적 의미 모델 조정(Logical Implication Model Steering, LIMS) 방법을 도입하여 개념 벡터를 활용해 모델의 생성 행동을 투명하게 조정할 수 있는 방법을 제안합니다. LIMS는 신경-기호(neuro-symbolic) 논리를 통합하여 모델의 사고 능력을 강화하는 새로운 접근을 제공합니다. 특히, 사용자는 특정 개념의 존재에 따라 모델이 어떻게 반응하는지를 쉽게 프로그래밍할 수 있습니다.

- **Technical Details**: LIMS는 주어진 입력에서 개념 P의 존재를 감지하면, 해당하는 생성 행동 Q로 연결되는 회로를 모델 내부에 추가하는 방식으로 작동합니다. 구체적으로, 이는 'P(x)면 Q(x)'의 형태로 정의되며, 이는 사용자가 요청한 특정 행동을 제어합니다. 이 방법은 모델의 분류 정확도와 제어된 행동 조정 간의 해석 가능성과 투명성을 높입니다. 또한, LIMS는 저자원 환경에서도 메모리와 계산 자원을 크게 요구하지 않으며, 기계적 추론을 통해 새로운 개념 벡터를 생성하는 데 유용합니다.

- **Performance Highlights**: LIMS는 데이터 포인트가 극히 적은 상황에서도 효율적으로 기능을 발휘하며, 특히 100개 훈련 데이터 포인트로도 유의미한 결과를 도출했습니다. 예를 들어, 허위 정보를 감지하고 불확실한 질문에 대한 반응을 줄이는 데 효과적이며, 수학 문제 해결 시 자동 사고 과정을 통한 추론을 개선합니다. 또한, LIMS 변형인 m-LIMS는 기존 파라미터와의 병합을 통해 손쉽게 배포할 수 있으며, 해석 가능성이 약간 감소하는 대신 동일한 성능을 유지합니다.



### A Novel Zero-Touch, Zero-Trust, AI/ML Enablement Framework for IoT Network Security (https://arxiv.org/abs/2502.03614)
- **What's New**: 이 논문에서는 IoT 생태계를 보호하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Zero Trust, Zero Touch 및 AI/ML 기반의 DDoS 공격 탐지 및 완화 기법을 통합하여 현대 IoT 생태계에서 발생할 수 있는 다양한 보안 위협을 효과적으로 대응할 수 있도록 합니다. 특히, 5G/6G에 최적화된 구조를 통해 지속 가능한 IoT 보안을 실현할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 Zero-Trust 원칙에 기반하여 모든 IoT 트래픽을 인증하고 검증하는 과정을 포함합니다. Zero-Touch 프로비저닝 기능은 IoT 장비의 안전한 온보딩 과정을 자동화하며, AI/ML 알고리즘을 활용하여 실시간으로 이상 징후를 탐지하고 대응합니다. 이 프레임워크는 XGBoost, Random Forest 등 5가지 기계 학습 모델을 비교 분석하여 DDoS 공격 탐지의 효과성을 확보합니다.

- **Performance Highlights**: 비교 분석 결과, 앙상블 기반 접근 방식이 다양한 DDoS 벡터를 탐지하고 완화하는 데 가장 높은 성능을 보였습니다. 이러한 결과는 5G 및 6G 환경에서 IoT 생태계의 보안을 강화하는 데 중요한 기초 자료를 제공합니다. 제안하는 프레임워크는 IoT 보안의 리스크를 낮추고 더욱 강력한 보호를 수행할 수 있도록 설계되었습니다.



### (GG) MoE vs. MLP on Tabular Data (https://arxiv.org/abs/2502.03608)
- **What's New**: 최근 연구에서는 신경망 아키텍처를 표 형식 데이터에 적응시키기 위한 노력들이 증가하고 있습니다. 그러나, 이러한 모델들이 많은 파라미터와 긴 훈련 및 추론 시간을 요구함에도 불구하고, 일반적인 다층 퍼셉트론(MLP) 신경망을 지속적으로 초월하는 경우는 드뭅니다. 이 논문에서는 Gumbel-Softmax 게이팅 함수를 사용하는 Mixture-of-Experts(MoE) 모델인 GG MoE를 소개하며, 이는 38개 데이터셋에서 표준 MoE 및 MLP 모델보다 높은 성능을 보이는 것을 입증합니다.

- **Technical Details**: 논문에서는 세 가지 모델, 즉 MLP, MoE, GG MoE의 성능을 비교합니다. MoE는 K개의 독립 모델인 전문가(expert)와 입력을 전문가들에 대한 확률 분포로 매핑하는 게이팅 함수(gating function) 두 가지 주요 구성 요소로 이루어져 있습니다. GG MoE는 MoE의 게이팅 네트워크에 Gumbel-Softmax 활성화 함수를 도입한 것으로, MoE와 GG MoE는 MLP보다 적은 파라미터를 사용하며, 이 점에서 성능 저하 없이 더 효율적인 아키텍처의 가능성을 보여줍니다.

- **Performance Highlights**: GG MoE는 38개의 데이터셋에 걸쳐 평균적인 성능에서 가장 높은 결과를 기록했습니다. MoE와 GG MoE는 모두 MLP에 비해 상당히 적은 파라미터 수를 활용하고 있으며, 이는 그들의 확장성과 앙상블 방법에 대한 잠재력을 의미합니다. 효율적인 파라미터 사용과 높은 성능을 결합한 GG MoE는 표 형식 데이터 예측에 있어 유망한 대안으로 자리매김할 수 있습니다.



### Bilevel ZOFO: Bridging Parameter-Efficient and Zeroth-Order Techniques for Efficient LLM Fine-Tuning and Meta-Training (https://arxiv.org/abs/2502.03604)
- **What's New**: 이 논문에서는 Bilevel ZOFO(Zeroth-Order-First-Order) 방법을 제안하여 대형 사전 훈련된 언어 모델(LLM)을 효율적으로 미세 조정하는 새로운 프레임워크를 탐구합니다. ZO 방법이 PEFT(Parameter-Efficient Fine-Tuning)와 결합되어 하드 프롬프트의 민감성을 완화하며, 두 가지 최적화 수준을 통해 상호 성능을 향상시킵니다. 이를 통해 기존의 PEFT 및 ZO 방법보다 더 나은 성능을 달성하면서 메모리 효율성을 유지하는 것을 목표로 합니다.

- **Technical Details**: Bilevel ZOFO 방법은 두 개의 최적화 루프를 활용하여 사전 훈련된 LLM의 하드 프롬프트에 대한 민감성을 줄이고, 전체 모델의 그래디언트를 계산하지 않고도 효율적인 메모리 사용을 유지합니다. 이 방법은 여기에서 PEFT 파라미터를 최적화하는 과정과 기본 모델을 조정하는 과정을 중첩하여 실행됩니다. 이로 인해 고비용의 전통적인 방법의 자원 요구 사항을 해결할 수 있습니다.

- **Performance Highlights**: 실험 결과, Bilevel ZOFO는 단일 작업 설정에서 기존 PEFT 및 ZO 방법을 초월하는 성능을 달성했습니다. 또한 다중 작업 학습에서도 효과적이며, 더욱 컴퓨터 자원을 절약하면서 성능을 유지하거나 향상시키는 가능성을 보여줍니다. 이러한 경량의 접근 방식을 통해 새로운 과제에 적합한 메타 트레이닝 프로세스가 가능해지는 장점이 있습니다.



### Stein Discrepancy for Unsupervised Domain Adaptation (https://arxiv.org/abs/2502.03587)
Comments:
          24 pages, 9 figures

- **What's New**: 이번 논문에서는 비지도 도메인 적응(unsupervised domain adaptation, UDA)에서 스틴 불일치(Stein discrepancy)를 활용한 새로운 방법을 제안합니다. 이 방법은 소스 도메인과 타겟 도메인 간의 불일치를 측정하여, 특히 타겟 데이터가 부족한 경우에도 효과적으로 성능을 개선할 수 있습니다. 저자들은 또한 비핵심형(non-kernelized) 및 핵심형(kernelized) 스틴 불일치 두 가지 버전을 포함하여 다양한 타겟 분포 추정 방식에 대해 설명합니다.

- **Technical Details**: 이 연구는 스틴 불일치를 기반으로 한 새로운 UDA 방법론을 제시하며, 이는 제한된 샘플이 있는 시나리오에서도 잘 작동합니다. 스틴 불일치는 특정 함수 클래스에서 적합한 함수의 점수 함수(score function)에 기초하여 비대칭적으로 두 개의 분포 간의 거리를 측정합니다. 연구진은 또한 타겟 도메인과 소스 도메인의 분포 간의 스틴 불일치를 통한 일반화 오차의 상한을 이론적으로 도출합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 도메인 불일치 측정 방식을 사용한 방법들보다 우수한 성능을 보임을 알 수 있었습니다. 특히 데이터가 부족한 상황에서 더욱 두드러진 성과를 냈습니다. 또한 이 접근법은 다양한 UDA 프레임워크와 통합될 수 있는 유연성도 지니고 있습니다.



### Clone-Resistant Weights in Metric Spaces: A Framework for Handling Redundancy Bias (https://arxiv.org/abs/2502.03576)
Comments:
          v1

- **What's New**: 이번 연구에서는 메트릭 공간에서의 요소 집합에 대한 새로운 이론적 프레임워크를 제안합니다. 요소의 분포가 임의적이며, 심지어 적대적일 수 있는 상황에서 이러한 요소를 어떻게 가중치를 부여할 수 있을지에 대한 문제를 다룹니다. 'Clone-proof representation functions'라는 개념을 통해 비슷한 객체들이 가중치를 공유함으로써 다중성으로 인한 편향을 피하는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 일반적인 메트릭 공간을 고려하여 최대 불확실성 원칙(maximum uncertainty principle)을 확장합니다. 이 프레임워크는 대칭성(symmetry), 연속성(continuity), 클론 방지(clone-proofness)와 같은 몇 가지 공리들을 포함하고 있습니다. 또한, 유클리드 공간(Euclidean spaces)의 중요한 경우에서 이러한 공리를 만족하는 representation functions의 존재에 대해서도 다룹니다.

- **Performance Highlights**: 이 연구는 다양한 맥락에서의 데이터 포인트, 작업 집합 또는 투표 조언 애플리케이션에서 개인의 정치적 의견을 다룰 때의 견고한 도메인 적응을 위한 해결책을 제공합니다. 특히, 제안하는 방법은 요소의 중요성을 분배하여 유사 객체들 간의 편향을 방지하는 복잡한 문제를 해결하는 데 기여할 것입니다.



### A Multi-Task Learning Approach to Linear Multivariate Forecasting (https://arxiv.org/abs/2502.03571)
- **What's New**: 이번 연구에서는 다변량 시간 시계열 데이터의 정확한 예측을 향상시키기 위해 다중 작업 학습(multi-task learning) 관점에서 접근합니다. 시간 시계열 예측을 다중 작업 문제로 정의하고, 유사한 변수를 그룹화하여 각 그룹이 별도의 작업을 형성하도록 제안합니다. 이를 통해 모델의 동작을 개선하고, 예측과 관련된 다양한 문제를 효과적으로 해결할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구진은 선형 모델의 그래디언트를 분석하여 유사한 변수 그룹을 구성하고, 서로 다른 변수를 균형 있게 처리하는 방법을 제안합니다. 이 방법론은 피어슨 상관 계수를 사용하여 선형 관계를 기반으로 변수를 그룹화하고, 다중 머리 선형 모델(MTLinear)을 구성합니다. 각 그룹은 자체적인 예측 과제를 해결하며, 그래디언트 조정 방식으로 우세한 변수가 전체 예측에 미치는 영향을 조절합니다.

- **Performance Highlights**: MTLinear 모델은 기존의 최첨단 모델들과 비교하여 우수한 성능을 보였습니다. 여러 어려운 벤치마크에서 평가를 진행하여 다변량 예측 작업에서 경쟁력 있는 결과를 나타냈습니다. 이러한 접근법은 다변량 시간 시계열 예측 문제의 해결을 위한 강력한 독립 기법으로 자리매김할 수 있습니다.



### Controllable Sequence Editing for Counterfactual Generation (https://arxiv.org/abs/2502.03569)
- **What's New**: 이 연구에서는 CLEF라는 새로운 모델을 제안하여 '안전한' 대안 경로를 생성하는 데 필요한 정밀한 조건 설정 및 타이밍을 통해 시퀀스 편집의 정확성을 높였습니다. 기존 접근 방식의 한계를 극복하고, 치료 및 약물 개입과 같은 다양한 상황에서 효과적으로 적용될 수 있습니다. CLEF는 특정 시간 후에만 영향을 미치는 국소 교정이 가능하여 복잡한 생물학적 데이터에 대한 의사 결정에 실질적인 기여를 할 것으로 기대됩니다.

- **Technical Details**: CLEF 모델은 시퀀스에서 어떻게 개입이 이루어져야 하는지를 표현하는 시간 개념을 학습하여, 특정 변수만을 조작하고 변화가 발생하는 시간을 정확하게 조정할 수 있습니다. 이는 의료 및 생물학적 데이터 처리 시 매우 중요한 역할을 하며, 구조적 및 시간적 제약을 준수하여 시퀀스 편집을 수행합니다. 실험 결과 CLEF는 즉각적(Immediate) 및 지연된(Delayed) 시퀀스 편집에서 기존의 최첨단 모델보다 크게 향상된 성능을 보였습니다.

- **Performance Highlights**: CLEF는 즉각적인 시퀀스 편집에서 최대 36.01% MAE 개선을 이루었으며, 지연된 시퀀스 편집에서도 기반 모델보다 최대 65.71% MAE 개선 효과를 보였습니다. 또한 정밀한 수정이 가능하여, 제1형 당뇨병 환자를 위한 건강한 대안 경로를 생성하는 데 성공했습니다. CLEF는 나아가 다양한 사전 훈련된 시퀀스 인코더를 사용하여 적응 및 성능 향상을 이끌어낼 수 있는 유연성을 가지고 있습니다.



### Code Simulation as a Proxy for High-order Tasks in Large Language Models (https://arxiv.org/abs/2502.03568)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.09074

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 사고력과 문제 해결 능력을 평가하기 위해 자연주의적(naturalistic) 및 합성(synthetic) 추론 과제를 수집한다. 연구 결과, 합성 데이터는 자연주의적 데이터 수집보다 훨씬 용이하게 대량으로 구할 수 있는 좋은 대안임을 보여준다. 이 연구는 LLM이 코드 시뮬레이션을 통해 추론 과제를 처리하는 능력을 테스트하는 실험적 파이프라인을 개발하였으며, 이 과정에서 합성 데이터의 활용 가능성을 강조한다.

- **Technical Details**: 연구에서는 문제가 코드 또는 자연어로 표현된 과제 쌍(x, p)으로 지정되며, 이들은 동일한 질문을 표현한다. Python 3을 프로그래밍 언어로 선택하여 LLM의 코드 실행 정확성을 측정하였고, 평가 기준으로는 모델의 대답과 코드 실행 결과 간의 일치를 비교하고 이를 정확도로 표현하였다. 이 방식은 LLM의 추론 능력을 보다 정교하게 분석할 수 있도록 도와준다.

- **Performance Highlights**: 결과적으로, 가장 강력한 LLM은 비교적 강력한 실행 능력을 보였으나, 메모리화(memorisation)와 패턴 인식에 크게 의존함에 따라 취약한 실행 과정을 나타냈다. 다섯 가지 비트리비얼(non-trivial) 자연주의적 작업과 그에 상응하는 코딩 과제를 연계하여 GPT-4, GPT-4o, Llama3.1-405B의 성능 간의 상관 관계를 보였다. 또한 새로운 Chain of Thought 확장을 도입하여 메모리화 문제를 완화하는 방법이 제시되었다.



### TD-M(PC)$^2$: Improving Temporal Difference MPC Through Policy Constrain (https://arxiv.org/abs/2502.03550)
- **What's New**: 모델 기반 강화 학습(model-based reinforcement learning) 알고리즘들은 높은 데이터 효율성과 지속적인 제어에서의 뛰어난 성능으로 주목받고 있습니다. 기존의 SAC 스타일 정책 반복(policy iteration) 방식은 데이터 생성 과정에서 발생하는 지속적인 가치 과대평가(permanent value overestimation) 문제를 드러냈습니다. 이를 해결하기 위해, OOD(Out-Of-Distribution) 쿼리를 줄이는 정책 정규화(policing regularization) 항을 제안하여 개선된 가치 학습을 도모하고 있습니다.

- **Technical Details**: 제안된 TD-M(PC)2 알고리즘은 온라인 계획을 통해 수집된 데이터를 더 잘 활용하도록 설계되었습니다. 이 방법은 기존의 TD-MPC2 프레임워크에 최소한의 수정을 통해 통합되며, 추가적인 컴퓨팅 요구 없이 가치 및 정책 사전(state value and policy prior)을 효과적으로 획득합니다. 벤치마크 테스트와 61 DOF( Degrees of Freedom) 휴머노이드 작업에서 다른 방법들이 개선된 성능을 보이며, 복잡한 환경에서도 뛰어난 결과를 나타내고 있습니다.

- **Performance Highlights**: 실험 결과 제안된 TD-M(PC)2 알고리즘은 TD-MPC2와 비교하여 많은 대폭 향상을 보여줍니다. 특히 61 DOF 휴머노이드 작업에서는 성능의 향상이 두드러지게 나타났습니다. 이 연구는 정책 개선과 계획자를 결합함으로써 발생하는 구조적 결함을 해결하는 데 필요성을 느끼고 있으며, 지속적인 제어 문제에서의 통계적 안정성을 더욱 강화할 수 있습니다.



### Path Planning for Masked Diffusion Model Sampling (https://arxiv.org/abs/2502.03540)
- **What's New**: 이 논문에서는 마스킹 확산 모델(Masked Diffusion Models, MDMs) 추론 과정에서 토큰이 비밀 해제(unmasking)되는 순서가 생성 품질에 미치는 영향을 조사합니다. 새로운 플래너(planner)를 도입하여 각 단계에서 비밀 해제를 선택하는 방식을 제안합니다. 이를 통해 다양한 대안 비밀 해제 전략이 생성 성능을 개선할 수 있음을 밝혀냅니다.

- **Technical Details**: 연구진은 확장된 증거 하한(evidence lower bound, ELBO)을 도출하였으며, 비밀 해제를 계획하는 프레임워크인 Path Planning (P2)을 제안합니다. P2는 사전 훈련된 BERT 또는 디노이저(denoiser) 자체를 이용하여 비밀 해제 결정을 안내하는 방식으로 설계되었습니다. 이 접근법은 기존의 모든 MDM 샘플링 전략을 일반화(generalize)하여 다양한 작업에서 효과적으로 적용될 수 있도록 합니다.

- **Performance Highlights**: P2 방법론은 언어 생성(in-context learning), 코드 생성, 스토리 채우기(story infilling), 수학적 추론, 역 curse correction 등 다양한 도메인에서 현저한 성능 개선을 보여줍니다. 또한 단백질과 RNA 서열 생성을 포함한 생물학적 서열 생성 분야에서도 뚜렷한 효과를 나타냅니다. 이러한 결과는 새로운 비밀 해제 전략이 생성적 모델의 품질을 극대화할 수 있다는 것을 뒷받침합니다.



### Teaching Language Models to Critique via Reinforcement Learning (https://arxiv.org/abs/2502.03492)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 출력 품질을 비판하고 개선하기 위해, 인간의 개입 없이도 피드백을 생성하도록 비평가 모델을 훈련하는 프레임워크인 CTRL (Critic Training via Reinforcement Learning)을 제안합니다. CTRL 프레임워크를 통해 코드 생성 과정에서 비평가가 생성자 모델의 성능을 크게 향상시킬 수 있음을 발견했습니다. 특히, 비평가 모델은 더 강력한 생성 모델과 결합하여 놀라운 성능 향상을 이끌어낼 수 있는 능력을 보여주었습니다.

- **Technical Details**: CTRL 프레임워크는 critic 모델을 task-performing 모델에서 분리하여, 비평이 반복되는 과정에서 최적의 솔루션 생성을 유도합니다. 이 과정에서, Group Relative Policy Optimization (GRPO)을 통해 두 단계의 훈련 파이프라인을 구현합니다. 이 방법은 주어진 코드 생성 문제에서 비평가가 직접적인 피드백을 제공하고, 이를 통해 더 나은 솔루션으로 나아갈 수 있도록 돕습니다.

- **Performance Highlights**: CTRL을 사용한 훈련은 CodeContests, LiveCodeBench, MBPP+, JudgeBench 등 여러 벤치마크에서 자가 비평 방법이나 기존의 강력한 비평 모델보다 눈에 띄게 우수한 성과를 나타냈습니다. 이러한 결과는 상대적으로 약한 비평가 모델이 강력한 생성 모델을 효과적으로 안내할 수 있는 가능성을 보여줍니다. 추가적으로, CTRL은 iterative critique-revision을 통해 테스트 시간 성능을 개선할 수 있으며, CodeContests 벤치마크에서 106.1%의 상대적 향상을 이루어냈습니다.



### ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features (https://arxiv.org/abs/2502.04320)
- **What's New**: 이 논문에서는 multi-modal diffusion transformers(DiT)가 가지는 데이터 표현이 해석 가능성을 증대시키는 독특한 속성을 가지고 있는지를 탐구합니다. 이를 위해 저자들은 ConceptAttention이라는 새로운 방법을 소개하며, 이 방법은 DiT의 attention layers에서 생성된 saliency maps를 활용하여 이미지 내 텍스트 개념을 정확하게 위치시키는 역할을 합니다. 기존의 cross-attention 메커니즘에 비해 선형 프로젝션을 통해 더 선명한 saliency maps를 생성할 수 있다는 주요 발견을 제공합니다.

- **Technical Details**: ConceptAttention은 DiT의 attention layers의 파라미터를 재사용하여 매우 맥락화된 concept embeddings를 생성하는 방식을 사용합니다. 특히 각 텍스트 개념은 그에 해당하는 시각적 요소(예: '용', '태양')와 연결될 수 있습니다. 이 방법은 추가 학습을 필요로 하지 않으며, 주어진 이미지와 concept embeddings 간의 선형 프로젝션을 수행하여 고품질의 saliency maps를 생성합니다.

- **Performance Highlights**: ConceptAttention은 제로샷(Zero-shot) 이미지 분할 작업에서 최고 성능을 달성하며, ImageNet-Segmentation 데이터셋과 Pascal VOC의 단일 클래스 서브셋에서 총 11가지의 다른 제로샷 해석성 방법을 초월하는 성과를 입증합니다. 이러한 성과는 DiT 모델의 표현이 분할과 같은 비전 과제로의 이전 가능성을 잘 보여줍니다. 저자들은 ConceptAttention을 통해 연구자들이 텍스트-이미지 생성 모델의 복잡한 역학 관계를 해석하고 탐구할 수 있도록 하는 기회를 제공합니다.



### ChamaleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters (https://arxiv.org/abs/2502.04315)
- **What's New**: 최근 큰 발전을 이루고 있는 대규모 언어 모델(LLMs)의 연구는 다양한 작업에서 놀라운 성능을 보여주고 있습니다. 이 논문에서는 ChamaleonLLM이라는 새로운 프레임워크를 소개하며, 이는 추론(inference) 시 모델이 동적으로 적응할 수 있도록 지원합니다. 전통적인 방법들과는 달리, ChamaleonLLM은 배치(batch) 기반 클러스터링(clustering)과 저차원(low-rank) 업데이트를 실시간으로 생성하는 방식을 활용하여 성능을 극대화합니다.

- **Technical Details**: ChamaleonLLM의 핵심은 배치 통계치를 기반으로 저차원(modification) 업데이트를 동적으로 생성하는 것입니다. 입력은 의미적 및 구문적 유사성에 따라 클러스터로 그룹화되어, 동질적인 입력들로 이루어진 미니 배치(batch)가 생성됩니다. 또한, 하이퍼 네트워크(hyper-network)를 통해 모델의 디코더(decoder) 가중치를 실시간으로 적응시키는 방식을 채택하여, 추론 과정에서 입력 데이터의 세부 사항을 보다 유연하게 반영합니다.

- **Performance Highlights**: 실험 결과, ChamaleonLLM은 기존의 LoRA 방식에 비해 향상된 성능을 보여 주었으며, 동적인 데이터 상황에서도 유연하게 대처할 수 있는 잠재력이 있습니다. 이 접근법은 메모리 및 계산 요구 사항을 줄이면서도, 고성능의 언어 모델 추론을 가능하게 하여 다양한 작업에 적응할 수 있는 잠재력을 지니고 있습니다. ChamaleonLLM은 오픈 소스로 제공되어 실험의 재현성을 보장하며, 연구자들이 이 프레임워크의 이점을 쉽게 활용할 수 있도록 하고 있습니다.



### DexterityGen: Foundation Controller for Unprecedented Dexterity (https://arxiv.org/abs/2502.04307)
Comments:
          Project: this https URL

- **What's New**: 본 논문에서는 DexterityGen(DexGen)라는 새로운 훈련 프레임워크를 소개합니다. DexGen은 RL(강화학습)을 사용하여 대규모의 섬세한 운동 원리(motion primitives)를 사전 훈련하며, 이를 통해 로봇이 안전한 행동을 생성하도록 합니다. 실시간 응용에서는 사람의 원격 조작(teleoperation)을 사용하여 DexGen이 의미 있는 조작 기술을 실행하도록 유도합니다.

- **Technical Details**: DexGen 컨트롤러는 로봇 상태를 고려하여 안정적이고 효과적인 행동을 생성하는 조건부 행동 모델을 기반으로 합니다. 이 모델은 시뮬레이션 데이터셋을 통해 사전 훈련되어, 고수준의 명령을 저수준의 구체적인 행동으로 변환하는 데 도움을 줍니다. 이 연구에서는 물체와의 상호작용을 기반으로 한 다양한 조작 기술의 시뮬레이션과 실제 환경에서의 문제 해결을 평가합니다.

- **Performance Highlights**: DexGen은 다양한 작업에서 안정성을 10-100배 향상시키는 데 성공했습니다. 특히, 로봇이 주어진 주사기나 스크류드라이버와 같은 도구를 사용하고 물체를 재배치하는 등의 복잡한 작업을 수행하는 데 있어 탁월한 능력을 보여주었습니다. 이러한 결과는 저수준의 조작 명령을 효과적으로 처리할 수 있는 능력을 입증합니다.



### Learning Real-World Action-Video Dynamics with Heterogeneous Masked Autoregression (https://arxiv.org/abs/2502.04296)
Comments:
          Website: this https URL

- **What's New**: 본 논문에서는 Heterogeneous Masked Autoregression (HMA)라는 새로운 방법론을 제안합니다. 이 방법론은 동작-비디오 동역학(action-video dynamics)을 모델링하여 고품질 데이터와 로봇 학습의 확대를 위한 평가를 생성합니다. HMA는 서로 다른 로봇 구현체와 도메인, 작업에서의 관찰 및 동작 시퀀스에 대한 이질적 전처리를 활용하여, 실시간 대처가 가능한 효율성을 유지하면서 다양한 환경을 처리할 수 있습니다.

- **Technical Details**: HMA는 마스크 오토리그레션(masked autoregression)을 활용하여 비디오 예측을 위한 양자화된(discrete) 또는 부드러운(soft) 토큰을 생성합니다. 이 접근 방식은 상태-오브젝트(state-action) 모델링을 통해 비디오와 동작 시퀀스를 동시에 생성할 수 있도록 하며, 특히 다양한 로봇 환경에서 동작의 이질성을 처리하는 데 중점을 둡니다. HMA는 3백만 개 이상의 동작 라벨이 포함된 비디오로 사전 훈련되었으며, 40개의 다양한 로봇 구현체에서 동영상을 생성할 수 있습니다.

- **Performance Highlights**: HMA는 이전 로봇 비디오 생성 모델들보다 시각적 충실도와 제어 가능성에서 우수한 성능을 보이며, 15배 더 빠른 실시간 추론 속도를 기록했습니다. 이 모델은 실제 로봇 애플리케이션에서 안정적으로 100프레임 이상의 궤적을 생성할 수 있어 고충실도의 로봇 시뮬레이터 역할을 수행합니다. 또한, 이 모델은 정책 평가 및 합성 데이터 생성에 사용되어 정책 성과를 개선하는 데 기여할 수 있습니다.



### Prediction-Powered E-Values (https://arxiv.org/abs/2502.04294)
- **What's New**: 이 논문은 prediction-powered inference (예측 기반 추론) 개념을 e-values (e-값)에 적용하여 통계적 추론의 가능성을 크게 확대하는 방법을 제안합니다. 이전의 방법들은 주로 Z-추정 문제에 한정되었지만, 이 연구는 이러한 한계를 넘어 다양한 추론 작업을 수행할 수 있도록 합니다. e-values는 p-values (p-값)에 대한 매력적인 대안으로, 적은 가정 하에서도 강력한 절차를 제공하는 장점을 가지고 있습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 임의의 복잡성을 가진 예측 모델을 활용하여 '비싼' 데이터를 '저렴한' 데이터로부터 예측함으로써 발생하는 데이터를 보완합니다. e-values를 통해 실행 가능한 모든 추론 절차는 우리의 예측 기반 방법으로 변환할 수 있으며, 이 과정에서 예측 모델을 지속적으로 업데이트하여 더 나은 데이터 효율성을 생성할 수 있습니다. 이러한 이점 덕분에 예측 기반 e-values는 시퀀셜 (sequential) 추론에 유용합니다.

- **Performance Highlights**: 총 네 가지 사례 연구를 통해 제안된 방법의 강력함을 입증합니다. 여기에는 당뇨병 유병률 추정, 위험 모니터링을 위한 가설 테스트, 변화 점 탐지 및 원인 발견 절차가 포함됩니다. 특히, 데이터 수집 비용을 크게 줄이면서도 성능이 현저히 개선됨을 보여 주며, 이는 활용 가능성을 높이고 있는 장점으로 작용합니다.



### Retro-Rank-In: A Ranking-Based Approach for Inorganic Materials Synthesis Planning (https://arxiv.org/abs/2502.04289)
- **What's New**: 이 논문은 Retro-Rank-In이라고 불리는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 표적 및 전구체 물질의 임베딩을 통해 공통의 잠재 공간에서 역합성을 재정의하고, 이종 화합물의 이진 그래프에서 쌍별 순위 모델을 학습합니다. 이는 실험실에서 새로운 전구체를 찾는 과정을 크게 개선할 수 있는 가능성을 지닙니다.

- **Technical Details**: Retro-Rank-In는 구성 수준의 트랜스포머 기반 재료 인코더와 표적 물질과 전구체 후보 간의 화학적 적합성을 평가하는 순위 생성기로 구성됩니다. 이 시스템은 전구체가 훈련 중에 본 적이 없는 경우에도 선택이 가능하도록 유연성을 제공합니다. 또, 공통 임베딩 공간을 통해 전구체와 표적 물질을 통합하여 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: Retro-Rank-In는 기존 메서드와 비교하여 새로운 전구체 추천의 가능성을 보여주며, 훈련 데이터에서 본 적이 없는 조합을 예측할 수 있는 능력이 있습니다. 본 논문에서 다룬 실험들은 Retro-Rank-In이 분포 외 일반화 및 후보 집합 순위 작성에서 새로운 최첨단 성능을 달성했음을 보여줍니다. 이는 무기 물질 합성을 가속화하는 강력한 도구로 자리 잡을 것입니다.



### Gaussian Process Regression for Inverse Problems in Linear PDEs (https://arxiv.org/abs/2502.04276)
- **What's New**: 이 논문은 선형 부분 미분 방정식(PDE)에 의해 지배되는 역 문제를 해결하기 위한 계산 효율적인 알고리즘을 소개합니다. 이 알고리즘은 고급 교환 대수 및 대수 분석에 근거하여 정해진 이전을 기반으로 한 가우시안 과정(Gaussian processes)을 사용하여 선형 PDE의 해를 모델링합니다. Macaulay2 컴퓨터 대수 소프트웨어를 활용하여 이 이전을 알고리즘적으로 구현할 수 있습니다.

- **Technical Details**: 본 연구는 알제브라 분석에서 Ehrenpreis–Palamodov 정리를 활용하여 선형 PDE의 해를 모델링하는 적합한 GP 이전을 구축합니다. 이 방법은 주어진 PDE의 정확한 해를 생성할 수 있는 이전을 제공합니다. 특히 2차원 파동 방정식의 경우, 이 방법을 통해 파동의 전파를 재구성하고 알 수 없는 파동 속도를 학습하는 데 효과적임을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 노이즈가 포함된 데이터에 대해 추가적인 복잡성 없이도 정확한 결과를 달성할 수 있는 능력을 제공합니다. 계산 효율성과 예측 정확성 사이의 균형을 잘 이루어내어 자원이 제한된 환경에서도 강력한 선택이 될 수 있습니다.



### Electrical Impedance Tomography for Anisotropic Media: a Machine Learning Approach to Classify Inclusions (https://arxiv.org/abs/2502.04273)
Comments:
          27 pages, 17 figures

- **What's New**: 본 논문은 Electrical Impedance Tomography (EIT)에서 도체 내부의 하나 또는 여러 개의 포함물(inclusion)을 식별하는 문제를 다룹니다. 이 방법은 Electrostatic 측정 결과를 Dirichlet-to-Neumann (D-N) 행렬을 통해 분석하며, 인공지능 기법인 인공 신경망(ANN)과 Support Vector Machines (SVM)을 활용하여 포함물의 크기와 개수, 그리고 이방성(anisotropy)의 존재를 탐지합니다.

- **Technical Details**: EIT의 역 전도 문제(inverse conductivity problem)는 전도율(conductivity) σ를 D-N 경계 맵을 통해 결정하는 것입니다. 이 논문에서는 16개의 전극을 사용하는 실험 설정을 통해, 충분한 데이터 수집만으로도 두 번의 측정으로 포함물의 크기를 정확하게 예측할 수 있음을 보여줍니다. 또한, 이 모델은 이방성에 대한 정보를 포함한 전극 설정을 사용하여 성능을 향상시켜 윤곽을 잡습니다.

- **Performance Highlights**: 모델을 통해 90% 이상의 높은 정확도로 이방성 포함물의 탐지가 가능하다는 결과를 제시하였습니다. 실험 데이터 및 시뮬레이션을 결합하여 진단의 신뢰성을 높였으며, 병행하는 연구 분야로는 의료 영상(medical imaging) 및 비파괴 시험(non-destructive testing) 등이 있습니다. 이러한 결과는 기계 학습 접근 방식을 EIT의 전통적인 분석 방법과 통합하는 데 있어 큰 잠재력을 보여줍니다.



### Variational decision diagrams for quantum-inspired machine learning applications (https://arxiv.org/abs/2502.04271)
Comments:
          8 pages, 3 figures, presented at Quantum Information in Spain (ICE-9)

- **What's New**: 이 논문은 Quantum Machine Learning (QML)에서 Decision Diagrams (DDs)의 새로운 활용 가능성을 제안합니다. Variational Decision Diagrams (VDDs)라는 새로운 그래프 구조를 도입하여, DD의 구조적 이점과 변분 방법의 적응성을 결합하여 양자 상태를 효율적으로 표현합니다. 본 연구는 VDD의 훈련 가능성을 조사하면서, 특정 양자 해밀토니안에 대한 바닥 상태 추정을 위해 VDD를 적용합니다.

- **Technical Details**: VDD는 Binary Directed Acyclic Multigraph (BDAMG) 형태로 구성되며, 루트 노드와 말단 노드의 구조를 정의합니다. 각 노드는 qubit 인덱스를 나타내며, 간선은 확률 진폭 정보를 보유합니다. 이 구조는 주어진 양자 상태에 대한 확률 진폭을 효과적으로 계산할 수 있게 해줍니다. 특히, VDD는 노드와 매개변수화된 간선을 통해 양자 상태의 완전하고 암묵적인 정규화된 표현을 제공합니다.

- **Performance Highlights**: 논문에서는 VDD를 사용하여 'Accordion ansatz'와 같은 특정 설정에서 barren plateau 현상이 나타나지 않음을 보여줍니다. 이는 qubit 수에 따른 경량화된 그래디언트 분산의 비지수적 스케일링을 보여줍니다. 실험을 통해 VDD가 여러 해밀토니안을 위한 바닥 상태 추정에서 효과적임을 입증하였습니다.



### Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion (https://arxiv.org/abs/2502.04263)
Comments:
          Accepted for publication at ICLR 2025

- **What's New**: 이 논문에서는 CLIP와 같은 다중 모달 (multi-modal) Vision-Language Models (VLMs)에서 텍스트와 이미지 인코더를 개별적으로 사용하는 관행이 비효율적임을 보여줍니다. 특히, 이미지 간 유사성 (intra-image similarity) 및 텍스트 간 유사성 (intra-text similarity) 문제를 'intra-modal misalignment'라는 개념으로 정의하여, 이러한 비효율성이 발생하는 원인을 설명합니다.

- **Technical Details**: 저자들은 'modality inversion'을 활용하여 입력 모달리티에서 보완 모달리티로의 표현 변환을 수행합니다. 이를 위해 Optimization-based Textual Inversion (OTI) 및 Optimization-based Visual Inversion (OVI) 기법을 도입하여 고정된 인코더를 사용하여 특징을 변환합니다. 실험을 통해 15개 이상의 데이터셋에서 intra-modal 작업에 대한 inter-modal 방식 접근의 성과 향상을 입증합니다.

- **Performance Highlights**: 실험 결과, inter-modal 접근 방식이 intra-modal 기초 성능을 초과하는 것으로 나타났습니다. CLIP의 인코더 간 intra-modal 유사성을 이용하는 기존의 방법이 성능 저하를 일으킨다는 점을 강조하며, VLM 사전 훈련 과정에서 intra-modal 손실 항을 포함하거나 텍스트와 이미지 임베딩 공간 간의 갭을 줄이면 intra-modal misalignment 문제를 완화할 수 있음을 시사합니다.



### Combining Language and App UI Analysis for the Automated Assessment of Bug Reproduction Steps (https://arxiv.org/abs/2502.04251)
Comments:
          12 pages, to appear in the Proceedings of the 33rd IEEE/ACM International Conference on Program Comprehension (ICPC'25)

- **What's New**: 이 논문에서는 버그 보고서에서 오류 재현 단계를 자동으로 식별하고 추출하는 새로운 기법인 AstroBR을 제안합니다. AstroBR은 LLM(GPT-4)의 언어 이해 능력을 활용하여 잘못된 S2R(재현 단계)을 개선하고 결측된 단계를 생성하여 양질의 피드백을 제공합니다. 또한, 기존 기법보다 25.2% 더 나은 성능을 보이며 결측된 S2R도 71.4% 더 정확하게 제안합니다.

- **Technical Details**: AstroBR은 애플리케이션 실행 모델을 정적 분석 대신 동적 분석을 통해 구성하며, 각 S2R에 대해 앱 상호작용을 추적하여 최적의 경로를 식별합니다. 이 과정에서, S2R의 기술적 세부 정보(예: 사용자 행동 및 GUI 구성 요소)를 추출하고 이를 기반으로 품질 보고서를 생성합니다. 논문에서는 S2R 설명의 품질을 평가하기 위해 Chaparro et al.이 제안한 품질 모델을 채택하고, 이를 위한 여러 가지 프롬프트 템플릿을 검토했습니다.

- **Performance Highlights**: AstroBR은 21개의 버그 보고서와 73개의 S2R 문장을 포함하는 테스트 데이터셋에서 기존의 최첨단 기법인 Euler와 비교하여 더 우수한 결과를 기록했습니다. 특히, AstroBR은 품질 주석에서 25.2% 높은 F1 점수를 달성하고, 결측된 S2R 식별에서는 71.4% 더 좋은 F1 점수를 보였습니다. 이러한 성과는 AstroBR의 효과적인 품질 좌표 매핑 방식과 LLM의 활용을 통해 이루어졌습니다.



### Free Energy Risk Metrics for Systemically Safe AI: Gatekeeping Multi-Agent Study (https://arxiv.org/abs/2502.04249)
Comments:
          9 pages, 1 figure

- **What's New**: 이번 연구에서는 에이전트 기반 및 다중 에이전트 시스템에서 위험을 측정하기 위한 기초로 Free Energy Principle을 조사합니다. 이를 바탕으로 다양한 맥락과 필요에 유연하게 대응할 수 있는 Cumulative Risk Exposure 메트릭을 도입합니다. 기존의 데이터 의존적 이론들과는 대조적으로, 이해관계자들은 시스템 결과에 대한 자신의 선호만 지정하면 되며, 이는 위험 관리와 완화에 대한 간단하고 투명한 의사결정 규칙을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 세계 모델(world model)과 선호 모델(preference model)의 불확실성을 자연스럽게 고려할 수 있으며, 이는 인식적(epistemic) 및 가치적(axiological) 겸손, 간결성(parsimonious), 미래 안전성을 보장합니다. 연구진은 자율주행 차량 환경에서 다중 에이전트에 의해 운영되는 차량 정책이 게이트키퍼(gatekeeper)에 의해 중재되는 상황을 시뮬레이션하여 이 새로운 접근 방식을 증명합니다. 이 구조에서는 각 차량이 지역 안전성을 평가하고, 필요 시 차량 정책에 개입합니다.

- **Performance Highlights**: 연구 결과, 자율주행 차량 집단에 게이트키퍼를 도입했을 때, 낮은 침투율에서도 시스템 안전성을 크게 향상시키는 긍정적인 외부 효과(externality)를 생성할 수 있음을 보여줍니다. 이러한 접근 방식은 다양한 상황에서 위험을 효과적으로 관리할 수 있는 가능성을 제시하며, 안전한 AI 시스템의 설계에 기여할 것으로 기대됩니다.



### Student-t processes as infinite-width limits of posterior Bayesian neural networks (https://arxiv.org/abs/2502.04247)
- **What's New**: 본 논문에서는 Bayesian Neural Networks (BNN)와 관련된 연구의 중요한 한계를 극복하는 새로운 접근법을 제안합니다. 기존의 연구들은 고정된 분산 가정을 바탕으로 하였으나, 우리는 Inverse-Gamma prior를 도입하여 BNN의 포스터리어 행동을 보다 유연하게 모델링합니다. 이로 인해 BNN은 불확실성(uncertainty) 모델링에서 클래식한 Gaussian process뿐만 아니라 Student-t process로 수렴하게 됩니다.

- **Technical Details**: 우리는 BNN의 매개변수가 Gaussian prior를 따르며, 마지막 숨겨진 층과 Gaussian likelihood의 분산이 Inverse-Gamma prior를 따르는 경우에 대해 분석합니다. 우리의 증명은 Wasserstein metric을 활용하여 Student-t process 근사값의 수렴 속도를 제어합니다. 더욱이, 이론적 기반을 확장하여 계층적 Gaussian-Inverse-Gamma 모델을 채택합니다.

- **Performance Highlights**: 이 연구 결과는 BNN의 불확실성 모델링을 향상시키며, Bayesian deep learning의 실제 응용에 있어 유용한 통찰력을 제공합니다. 특히, prior 분포를 조정함으로써 포스터리어 BNN의 수렴 특성을 확장하는 새로운 연구 방향을 열어줍니다. 또한, 논문에서는 실험 결과를 제공하여 이론적 주장의 유효성을 검증합니다.



### TriNER: A Series of Named Entity Recognition Models For Hindi, Bengali & Marath (https://arxiv.org/abs/2502.04245)
- **What's New**: 이번 연구에서는 인도의 세 가지 주요 언어인 힌디어(Hindi), 벵골어(Bengali), 마라티어(Marathi)에 대한 다국어(NER) 모델을 개발하였습니다. 이러한 언어의 복잡성과 다양성 때문에 발생하는 문제를 해결하기 위해, 단일 모델을 통해 다양한 엔티티 그룹을 통합적으로 식별하는 방법을 제시합니다.

- **Technical Details**: 커스텀 트랜스포머 모델을 훈련하고 몇 가지 사전 훈련된(pretrained) 모델을 미세 조정(fine-tune)하여, 총 6개의 엔티티 그룹에 대해 F1 점수 92.11을 달성했습니다. 이 모델은 자연어 처리의 핵심 작업 중 하나인 개체 인식(NER)의 성능을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: 이 논문에 제시된 모델은 서로 다른 언어 간의 엔티티 그룹 및 태그 이름의 불일치를 크게 줄이는 데 기여할 것으로 기대됩니다. 다국어 NER 모델의 도입은 인도의 다양한 언어 환경에서 더욱 효과적으로 활용될 수 있는 가능성을 제공합니다.



### A Classification System Approach in Predicting Chinese Censorship (https://arxiv.org/abs/2502.04234)
- **What's New**: 이번 논문은 중국 인터넷에서 소셜 미디어인 Weibo 게시물이 검열될지 여부를 예측하기 위해 분류기(classifier)를 사용하는 연구입니다. 연구는 랜덤 샘플링을 통해 정제된 중국어 구문 데이터셋을 구축하고, 이 데이터셋에서 바이너리 검열 마킹을 이용해 4개의 로지스틱 회귀 모델을 분류하는 데 활용되었습니다. 또한, 사전 훈련된 트랜스포머(transformer) 모델을 사용해 유사한 분류 작업을 수행하여 Fined-Tuned BERT 모델이 다른 방법들보다 우수한 성능을 보였음을 밝혔습니다.

- **Technical Details**: 중국어 언어의 특성상, 의미 있는 구문을 토큰화하기 위해 Jieba라는 검증된 NLP 라이브러리를 사용하였습니다. 논문에서는 TF-IDF 접근법에서 영감을 받아 4가지 정보 검색기(information retriever)를 훈련 데이터에 적용하였으며, 이를 통해 각각의 특성 벡터(feature vector)를 추출하였습니다. 로지스틱 회귀 모델에서 가장 좋은 성능을 나타낸 Fined-Tuned BERT 모델의 활용이 강조되었고, 이를 통해 NLP 기법을 통해 검열 라벨 시스템의 가능성을 탐구하고 있습니다.

- **Performance Highlights**: 평가 지표로는 매크로 F1 점수와 ROC-AUC를 사용하였으며, Fined-Tuned BERT 모델이 다른 분류 방법보다 우수한 성능을 기록했습니다. 또한, 2012년 Weibo의 검열 키워드 리스트를 활용하여 모델링을 진행하였고, 모델링 결과 검열되지 않은 데이터에서의 불균형도 고려하여 3%인 검열 데이터를 포함한 전체 데이터의 효과적으로 분석했습니다. 이 연구는 Weibo에서의 검열 이해를 더 깊이 있게 할 것으로 기대됩니다.



### XAttnMark: Learning Robust Audio Watermarking with Cross-Attention (https://arxiv.org/abs/2502.04230)
Comments:
          24 pages, 10 figures

- **What's New**: 이번 논문에서는 Cross-Attention Robust Audio Watermark (XAttnMark)를 통해 오디오 워터마킹의 새로운 접근 방식을 제안합니다. 이는 생성기(generator)와 탐지기(detector) 간의 부분 매개변수 공유(partial parameter sharing)와 효율적인 메시지 검색을 위한 교차 주의(cross-attention) 메커니즘을 적용하여, 강력한 검출과 정확한 귀속(attribution)을 동시에 달성할 수 있도록 설계되었습니다. 또한 새로운 심리음향(psychoacoustic)에 기반한 시간-주파수 마스킹 손실(timely-frequency masking loss)을 도입하여 워터마크의 인지 불가능성을 높였습니다.

- **Technical Details**: XAttnMark는 생성기와 탐지기 간의 매개변수 공유 및 교차 주의 모듈을 통해 메시지 검색의 효율성을 크게 향상시킵니다. 특히, 워터마크 인지 불가능성을 보장하기 위해 비대칭 2D 커널을 사용하여 마스킹 에너지를 계산하고, 그것을 기반으로하는 손실 함수를 설계하였습니다. 이러한 접근법은 다양한 오디오 편집 변환에도 저항력이 뛰어나며, 출처 확인과 검출의 성능을 동시에 최적화할 수 있는 방향으로 나아갑니다.

- **Performance Highlights**: 우리의 방법은 여러 가지 오디오 변형에 대해서도 뛰어난 성능을 보이며, 현재까지의 최고의 검출 및 귀속 성능을 기록했습니다. 특히 강력한 편집 강도가 가해지더라도 성공적으로 워터마크 검출이 가능하다는 점에서 XAttnMark는 기존의 어떠한 방법론보다 두드러진 성과를 보입니다. 이러한 결과는 단순한 품질 유지에 그치지 않고, 실제적이고 다양한 도전 상황에서도 방어력을 인정받았습니다.



### Keep It Light! Simplifying Image Clustering Via Text-Free Adapters (https://arxiv.org/abs/2502.04226)
- **What's New**: 이 논문에서는 장애물인 대량의 텍스트 모드와 복잡한 학습 과정을 사용하지 않고도 경쟁력 있는 성능을 달성할 수 있는 심층 클러스터링(deep clustering) 방법론인 SCP(Simple Clustering via Pre-trained models)를 제안합니다. 이 방법은 텍스트 최적화 없이 미리 훈련된 비전 모델의 기능 표현을 활용하여 소규모 클러스터 헤드만 훈련합니다. 이를 통해 실세계 응용 프로그램에 더 쉽게 적용할 수 있는 간단한 클러스터링 파이프라인을 구현합니다.

- **Technical Details**: SCP는 주로 CIFAR-10, CIFAR-20, CIFAR-100, STL-10, ImageNet-10, ImageNet-Dogs와 같은 벤치마크 데이터셋을 이용하여 성능을 실험합니다. SCP의 성능은 기존의 복잡한 방법들과 비교할 때 경쟁력이 있으며, 이 방식은 전통적인 클러스터링 기술보다 훨씬 효율적입니다. 그 이론적 근거로는 이미지와 텍스트 정보가 결합되지 않아도 효율적인 클러스터링이 이루어질 수 있다는 점을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과, SCP는 기존 최첨단 성능(SOTA)과 비슷한 수준의 경쟁력을 보이며, 특히 비전 모델의 대표성만으로도 성능을 발휘할 수 있다는 것을 강조합니다. 이 파이프라인은 표준 L4 GPU에서 실행이 가능하여 널리 사용할 수 있는 가능성을 나타내며, 이러한 접근 방식은 클러스터링 성능 향상에 기여할 것으로 기대됩니다.



### The Best Instruction-Tuning Data are Those That F (https://arxiv.org/abs/2502.04194)
- **What's New**: 이번 논문에서는 GRAPE라는 새로운 SFT(Supervised Fine-Tuning) 프레임워크를 제안합니다. 이는 타겟 모델의 사전 훈련 분포에 가장 가까운 응답을 선택하여, 데이터 수집 과정에서 발생할 수 있는 성능 저하 문제를 해결하는 데 중점을 둡니다. GRAPE는 다양한 LLM(Language Model)에서 응답을 수집하고, 그 중 타겟 모델에 대해 높은 확률을 보이는 응답을 선택하여 데이터의 질을 향상시킵니다.

- **Technical Details**: GRAPE는 여러 모델에서 수집된 응답 중 타겟 모델과 가장 유사한 응답을 선택하여 SFT를 진행합니다. 이 과정은 타겟 모델의 확률을 이용해 이루어지며, 기존의 일률적인 응답 대신 모델에 적합한 응답을 사용합니다. 특히, GRAPE의 접근 방식은 데이터 분포의 이동에 따른 문제를 최소화하여 성능 향상을 도모합니다.

- **Performance Highlights**: GRAPE는 LLaMA3.1-8B, Mistral-7B 및 Qwen2.5-7B와 같은 일반적으로 사용되는 LLM에서 테스트되었으며, 기존 베이스라인보다 최대 17.3%의 성능 향상을 기록했습니다. 또한, GRAPE 선택 데이터를 사용하여 Tulu3 및 Olmo2에 대한 후속 데이터에서도 강력한 성능 개선을 보여주었습니다. GRAPE는 데이터의 양을 줄이고 학습 에폭수를 절반으로 줄여도 높은 성능을 유지함으로써 SFT 과정에서 높은 효율을 입증했습니다.



### Multi-task Online Learning for Probabilistic Load Forecasting (https://arxiv.org/abs/2502.04163)
Comments:
          2024 IEEE Sustainable Power and Energy Conference

- **What's New**: 이번 논문에서는 다양한 소비 패턴을 가진 여러 개체의 전력 수요를 예측하기 위해 온라인 및 확률적(load forecasting) 기술을 적용한 다중 작업 학습(multi-task learning) 기법을 제안합니다. 기존의 기술들이 전력 수요의 불확실성을 효과적으로 평가하지 못하거나 소비 패턴의 동적 변화를 반영하지 못하는 한계를 보완하는 방식으로 발전하였습니다. 이 기법은 여러 개체의 동적 유사성을 활용하여 전력 수요에 대한 정확한 확률적 예측을 제공합니다.

- **Technical Details**: 제안된 방법은 벡터 값 히든 마르코프 모델(Vector-valued Hidden Markov Models, HMMs)을 기반으로 하여 다중 작업 전력 수요 예측을 위한 온라인 학습 기법을 개발하는 방식으로 구성됩니다. 이 기술은 여러 개체에 대한 최근의 HMM 매개변수를 사용하여 확률적 예측을 도출하며, 시간이 지남에 따라 변하는 소비 패턴 및 개체 간 의존성을 동적으로 적합하도록 설계되었습니다. 이로써 다양한 소비 시나리오에서 현재 다중 작업 학습 접근법의 효과성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 다양한 전력 소비 시나리오에서 기존의 다중 작업 기술들과 비교하여 유의미한 개선을 보임을 입증합니다. 다중 개체의 전력 수요 예측 정확성이 향상되었으며, 이를 통해 전력 관리의 효율성을 높일 수 있는 잠재력을 지닌 것으로 평가됩니다. 본 연구에서 개발된 기법은 실시간 데이터 학습을 통해 전통적인 수요 예측의 한계를 극복하고 있습니다.



### A Pseudo Markov-Chain Model and Time-Elapsed Measures of Mobility from Collective Data (https://arxiv.org/abs/2502.04162)
Comments:
          27 pages, 11 figures

- **What's New**: 이 논문에서는 시간 경과에 따른 흐름을 이해하기 위한 pseudo Markov-chain 모델을 개발했습니다. 데이터는 시간과 공간으로 집계된 집합적 이동 데이터를 기반으로 하며, 이를 통해 개인 이동 데이터에 대한 기존의 이동성 척도와 유사한 측정을 개발합니다. NetMob 2024 데이터 챌린지를 통해 이 모델을 적용하여 도시 내 통계 및 통근 패턴과 일치하는 흥미로운 결과를 도출했습니다.

- **Technical Details**: 이 모델은 개별 장치의 통화 데이터(CDR), GPS 데이터 및 모바일 애플리케이션의 맥락적 사용 데이터를 수집하여 만들어졌습니다. 집합적 이동 데이터에서 각 시간 간격에 대한 기본 집계 정보는 지리적 Cell에서 탐지된 장치 수와 Cell 쌍 간의 이동 횟수로 구성됩니다. 이를 통해 우리는 개별 데이터보다 높은 수준의 익명성을 유지하면서도 이동성을 간략하게 설명할 수 있는 척도를 도출하려고 합니다.

- **Performance Highlights**: 모델을 사용한 분석에서 발견된 결과는 기존의 통계 및 도시 내 통근 패턴과 일치하며, 환경 변화와 지속 가능 발전 맥락에서 인간 이동성을 개선적으로 이해하는 데 기여할 것으로 기대됩니다. 이 모델은 또한 비교 연구, 계획 및 예측과 같은 다양한 응용 분야에 활용될 수 있는 가능성을 보입니다.



### A data-driven two-microphone method for in-situ sound absorption measurements (https://arxiv.org/abs/2502.04143)
Comments:
          41 pages, 8 figures

- **What's New**: 본 연구는 무한 다공 슬랩의 음향 흡수 계수를 추정하기 위해 신경망(neural network)과 두 개의 마이크로폰을 이용한 측정 방식(data-driven approach)을 제안합니다. 1D 컨볼루셔널 네트워크는 두 마이크로폰에서 측정된 음압(sound pressure) 간의 복소수 전이 함수(complex-valued transfer function)를 기반으로 음향 흡수 계수를 예측합니다. 이 방법은 Delany-Bazley-Miki 모델을 이용해 생성된 수치 데이터로 학습되고 검증되어 다양한 수치 샘플에 대해 정확한 예측을 보여줍니다.

- **Technical Details**: 제안된 방법은 포화 재료에 대한 각도 의존적인 음향 흡수 계수를 측정하는 것을 목표로 하며, 두 마이크로폰 측정을 통해 얻은 복소 가치 전이 함수의 스펙트럼 데이터를 사용합니다. 1D 잔여 신경망(residual neural network)이 100Hz에서 2000Hz 간의 주파수 범위에서 음향 흡수 계수를 예측하도록 훈련 되고 실험적 테스트가 진행됩니다. 또한, 이 연구는 포기재 샘플의 다양한 크기, 입사각 및 위치에 대해 실험적으로 검증되었습니다.

- **Performance Highlights**: 실험 결과 신경망은 포화 재료의 음향 흡수 성능을 신뢰성 있게 예측할 수 있는 가능성을 보여줍니다. 네트워크에 의해 예측된 음향 흡수 계수는 이론적으로 얻은 값 및 임피던스 튜브(impedance tube)에서 얻은 값과 잘 비교되었습니다. 이 연구에서 제안한 방법은 설치 후 및 실제 운전 조건에서 음향 재료의 음향 흡수 계수를 추정하는데 유망한 전망을 제공합니다.



### DEALing with Image Reconstruction: Deep Attentive Least Squares (https://arxiv.org/abs/2502.04079)
- **What's New**: 본 논문은 데이터 기반 이미지 재구성을 위한 새로운 방법을 제안합니다. 기존의 복잡한 깊은 네트워크 아키텍처 대신, 이는 고전적인 Tikhonov 정규화를 기반으로 한 접근법입니다. 이 방법은 중간 재구성을 반복적으로 개선하며, 두 가지 주요 구성 요소(learned filters 및 attention mechanism)를 포함하고 있습니다.

- **Technical Details**: 이 방법은 Quadratic 문제를 해결하여 이미지의 특징을 추출하는 learned filters와 필터 응답의 패널티를 조정하는 attention mechanism을 사용합니다. 특히, 필터의 응답에 대한 손실을 지역적으로 조정함으로써 성능을 최적화합니다. 따라서, 이 연구는 전통적인 정규화 기법과 딥러닝을 연결하고 있습니다.

- **Performance Highlights**: 제안된 방법은 최신 plug-and-play 및 learned regularizer 접근법과 동일한 수준의 성능을 달성합니다. 그러나, 이 방법은 해석 가능성과 강인함, 그리고 수렴성을 제공하여 기존 방법의 한계를 극복합니다. 이로써, 재구성 문제에 대한 보다 원칙적이고 해석 가능한 접근을 제공하고 있습니다.



### AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inferenc (https://arxiv.org/abs/2502.04077)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 인해 Key-Value(KV) 캐시 압축을 통한 효율적인 추론이 주목받고 있습니다. 하지만 기존 방법들은 주로 휴리스틱 순위를 사용하여 중요 KV 토큰을 파악하는 데 한계가 있어 LLM 성능 저하를 초래합니다. 이에 본 연구에서는 AttentionPredictor라는 학습 기반의 중요 토큰 식별 접근법을 제안하고 있습니다.

- **Technical Details**: AttentionPredictor는 경량의 합성곱 모델을 활용하여 시공간 패턴(spatiotemporal patterns)을 포착하고 다음 토큰의 attention score를 예측합니다. 이 방법은 기존의 휴리스틱 방식으로는 캡처하기 어려운 동적인 시간 패턴을 학습 기반으로 해결하는 데 초점을 맞추고 있습니다. 알고리즘의 메모리 소비가 거의 없으면서도 정확한 예측을 제공하는 점이 주요 특징입니다.

- **Performance Highlights**: 제안된 방법은 LongBench 데이터셋에서 16배의 KV 캐시 압축을 달성하면서 기존 최고 성능보다 41% 향상된 결과를 보여주었습니다. 또한 32K 컨텍스트에서의 지연 시간을 1.4배 단축시킬 수 있었습니다. 이러한 성과는 AttentionPredictor의 정확한 중요한 토큰 식별 덕분으로, LLM 성능을 유지하며 효율적인 캐시 관리가 가능하다는 것을 시사합니다.



### Exploring Imbalanced Annotations for Effective In-Context Learning (https://arxiv.org/abs/2502.04037)
- **What's New**: 본 연구는 불균형(class imbalance) 클래스 분포가 주석 데이터셋의 In-context learning (ICL) 수행에 미치는 영향을 처음으로 연구했습니다. 실험을 통해 기존의 재조정(rebalance) 방법이 ICL의 불균형 문제를 완화하지 못하며, 오히려 문제를 악화시킬 수 있음을 발견하였습니다. 이러한 배경에서 저자들은 클래스 가중치(class weighting)를 이용하여 원래의 점수 함수(original scoring functions)를 수정하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 저자들은 주석 데이터셋과 테스트 데이터셋 간의 분포 차이를 두 가지 구성 요소인 클래스별 가중치(class-wise weights)와 조건적 바이어스(conditional bias)로 나누어 설명합니다. 이들은 균형 잡힌 검증 데이터셋에서의 실험적 오류를 최소화하여 조건적 바이어스를 추정하고, 이를 통해 원래의 점수 함수를 수정하여 ICL의 성능을 개선할 수 있습니다. 저자들은 효과적 수(number)와 같은 클래스별 가중치를 사용하고, 조건적 바이어스를 추정하는 방식을 채택하였습니다.

- **Performance Highlights**: 저자들은 Amazon, AgNews, Yelp 등 7개의 다양한 데이터셋에서 extensive한 평가를 통해 제안한 방법의 효과성을 입증하였습니다. 예를 들어, 100 비율의 불균형 데이터셋에서 ICL의 테스트 정확도가 37.83%에서 43.29%로 증가하여 무려 5.46%의 개선을 보였습니다. 이러한 성능 향상은 생성(generation) 작업에서도 ICL의 성능을 개선하는 데 유효함을 확인하였습니다.



### Fine, I'll Merge It Myself: A Multi-Fidelity Framework for Automated Model Merging (https://arxiv.org/abs/2502.04030)
- **What's New**: 본 논문에서는 모델 병합(Model Merging)을 통해 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력을 향상시키는 자동화된 프레임워크를 제안합니다. 이 프레임워크는 수많은 모델 조합을 탐색할 수 있는 방식으로, 기존의 수동 접근 방식보다 비용을 줄이고 효율성을 높여줍니다. 특히, 우리는 세분화된 병합 전략을 가능하게 하는 레이어별 융합(Layer-wise Fusion, LFS)과 깊이별 통합(Depth-wise Integration, DIS)이라는 두 가지 새로운 검색 공간을 소개하고 있습니다.

- **Technical Details**: 논문에서 제안하는 자동화된 모델 병합 프레임워크는 Multi-Fidelity Optimization (MFO) 기법을 활용하여 병합 과정을 최적화합니다. 이 프레임워크는 제안된 두 가지 검색 공간에서 단일 및 다목적 최적화를 지원하며, 필요한 계산 비용을 줄이며 세부적인 탐색을 가능하게 합니다. 구체적으로, 레이어별 융합은 여러 모델의 이를 병합하며 깊이별 통합은 최적의 계층 관계를 발견하는 데 초점을 맞추고 있습니다. 이러한 방식을 통해 계산 비용을 절감하면서도 모델의 성능을 개선할 수 있습니다.

- **Performance Highlights**: 우리는 여러 벤치마크를 통해 제안된 프레임워크의 효율성을 평가했으며, 결과적으로 다목적 시나리오에서 평균 6.86% 개선을, 도전적인 GSM8K 작업에서는 4.24% 개선을 달성했습니다. 이와 함께, 다양한 추론 벤치마크에서 일관된 효과성을 보였습니다. 우리의 방식은 제한된 계산 자원 내에서 실행되며, 예를 들어 500회의 검색 단계 이내에서 효율적인 병합이 가능합니다.



### MultiFloodSynth: Multi-Annotated Flood Synthetic Dataset Generation (https://arxiv.org/abs/2502.03966)
Comments:
          6 pages, 6 figures. Accepted as Oral Presentation to AAAI 2025 Workshop on Good-Data

- **What's New**: 이 논문에서는 홍수 위험 감지 시스템을 위한 합성 데이터 생성 프레임워크인 MultiFloodSynth를 소개합니다. 이 프레임워크는 다양한 실제 속성을 가상 세계로 옮겨와 홍수 상황을 시뮬레이션하며, 기존 데이터 수집의 한계를 극복하기 위해 최신 generative models를 활용합니다. MultiFloodSynth는 5단계의 다양한 주석을 포함한 풍부한 합성 데이터셋을 제공하여, 실제 데이터셋과의 유사성을 유지하면서도 효율적인 모델 학습 환경을 조성합니다.

- **Technical Details**: 프레임워크는 3D 엔진을 이용하여 가상 도시 홍수 상황을 합성합니다. 이 과정에서 레이아웃(layout), 조명(lighting), 홍수 높이(flood-level)와 같은 다양한 속성을 고려하며, 여러 종류의 주석(annotation) 정보(예: normal map, 세분화(segmentation) 맵, 2D/3D 바운딩 박스)를 생성합니다. MultiFloodSynth는 사용자가 원하는 구조를 설정하고 조정할 수 있는 유연성을 제공하여 최종 가상 장면을 구성할 수 있게 합니다.

- **Performance Highlights**: MultiFloodSynth는 총 70,117장의 이미지를 생성하였으며, 그 중 14,593장이 홍수 장면에 해당합니다. 실제 데이터와 비교했을 때, YOLOv10 모델을 사용한 홍수 감지 성능이 높아졌으며, 향상된 정확성을 보여줍니다. 또한 다양한 주석 형태를 활용하여 다양한 컴퓨터 비전 작업에 대한 응용 가능성을 높였습니다.



### Fairness Aware Reinforcement Learning via Proximal Policy Optimization (https://arxiv.org/abs/2502.03953)
- **What's New**: 이번 논문은 민감한 특성(예: 인종, 성별, 사회경제적 지위)을 고려하여 다중 에이전트 시스템(Multi-Agent Systems, MAS)에서 보상의 공정한 배분을 다루고 있습니다. Proximal Policy Optimization(PPO)에 페널티 항을 추가하여 공정성을 향상시키는 방법을 제안하며, 이는 과거 성과의 불균형을 최소화하고 향후 결정을 공정하게 만드는 두 가지 구성 요소를 통합합니다. 실험에서는 Allelopathic Harvest 게임에서 fair-PPO가 기존 PPO보다 공정성 메트릭에서 우수한 정책을 생성한다는 것을 보여주었습니다.

- **Technical Details**: 제안된 fair-PPO 알고리즘은 PPO의 목표 함수를 수정하여 공정성 메트릭에서 파생된 페널티 항을 포함하고 있습니다. 첫 번째 성분은 민감한 특성을 가진 에이전트 간의 보상 불균형을 최소화하며, 두 번째 성분은 각 에이전트의 가치 함수에 따라 예상 보상의 불균형을 최소화합니다. 이를 통해 에이전트의 정책이 보상 극대화와 공정을 동시에 고려하도록 유도합니다.

- **Performance Highlights**: fair-PPO는 모든 공정성 메트릭에서 기존 PPO보다 더 공정한 정책을 생성하였습니다. 이 알고리즘을 통해 공정성을 확보하는 대가로 보상은 감소하지만, 민감한 특성을 가진 에이전트와 그렇지 않은 에이전트가 비슷한 비율의 보상을 포기하는 것으로 나타났습니다. 또한, 과거 및 미래 기대 보상에 대한 페널티 구성 요소가 에이전트의 행동을 효과적으로 변화시키고 공정성을 개선한다는 것을 실험을 통해 입증하였습니다.



### DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation (https://arxiv.org/abs/2502.03930)
Comments:
          16 pages, 8 figures

- **What's New**: 이번 연구에서는 Diffusion Transformer Autoregressive Modeling (DiTAR)이라는 패치 기반의 오토회귀(autoregressive) 프레임워크를 제안하며, 언어 모델(language model)과 확산 변환기(diffusion transformer)를 결합합니다. 이를 통해 연속적인 매개변수 생성을 위한 효율성을 크게 향상시키고, 계산 요구 사항을 줄이는 효과를 보여줍니다. 특히, DiTAR는 패치 생성을 위한 분할 정복 전략을 활용하여, 언어 모델이 집계된 패치 임베딩을 처리하고, 확산 변환기가 다음 패치를 생성할 수 있도록 합니다.

- **Technical Details**: DiTAR는 연속 토큰의 예측을 위해 causal attention과 bidirectional attention의 강점을 결합한 오토회귀 모델입니다. 이를 통해 DiTAR는 패치를 여러 개로 나누고, 언어 모델이 패치 간 예측을 담당하며, 확산 변환기가 패치 내 예측을 수행하며 효율성을 높이고 있습니다. 새로운 온도(temperature) 정의를 통해 역확산 ODE에서 노이즈의 도입 시점을 정하고, 이를 활용한 샘플링 기법을 제안하여 연속값 언어 모델의 탐색과 활용을 조율합니다.

- **Performance Highlights**: DiTAR는 제로샷 텍스트 투 스피치(zero-shot text-to-speech) 작업에서 뛰어난 성능을 발휘하며, 특히 강인성(robustness), 화자 유사성(speaker similarity), 자연스러움(naturalness) 분야에서 최첨단(SOTA) 성과를 기록했습니다. 이러한 성과는 기존의 모델들이 요구하는 계산량보다 현저히 낮은 비용으로 이루어졌습니다. 특히, 여러 단계를 거치는 복잡한 파이프라인 대신, DiTAR는 언어 모델이 최종 특성을 직접 예측하는 단순화된 방식을 적용하여 우수한 결과를 도출했습니다.



### Blackwell's Approachability with Approximation Algorithms (https://arxiv.org/abs/2502.03919)
- **What's New**: 본 연구는 Blackwell의 접근 가능성 문제를 재조명하며, 플레이어와 적대자가 반복적으로 수행하는 벡터값 게임에 대해 다룹니다. 기존의 접근 가능성 문제에서 플레이어가 최적화를 통한 접근을 시도하는 대신, 제약이 있는 문제에서 근사 알고리즘을 통한 접근성을 보장합니다. 또한, 모노톤 선호를 가진 플레이어를 위한 접근 알고리즘을 제시합니다.

- **Technical Details**: 논문에서는 플레이어의 행동 집합이 NP-Hard 최적화 문제로 인해 최적화가 어려운 경우를 주로 다룹니다. 주요한 기여로는, 플레이어와 적대자의 행동 셋이 근사 알고리즘을 통해 접근 가능할 때, 특정한 스케일링을 통해 접근 가능성을 보장할 수 있는 알고리즘을 개발했습니다. 이 알고리즘은 플레이어의 평균 손실을 일정한 비율에 근접하게 만드는 것을 목표로 합니다.

- **Performance Highlights**: 연구에 따르면, 플레이어의 행동 집합이 α𝒳 > 1 비율을 가진 근사 오라클을 통해 접근 가능할 경우, 주어진 목표 집합이 접근 가능하다면, 플레이어는 효율적으로 접근할 수 있는 방법을 개발하였습니다. 결과적으로, 이 방법을 통해 플레이어의 평균 손실이 무한히 증가함에 따라 도달할 수 있는 손실 벡터가 존재함을 증명했습니다.



### Technical Report: Generating the WEB-IDS23 Datas (https://arxiv.org/abs/2502.03909)
- **What's New**: 이 논문에서는 다양한 공격 유형을 포함한 최신의 데이터셋을 생성하기 위한 모듈형 트래픽 생성기를 개발했다고 소개합니다. 기존의 네트워크 침입 탐지 시스템(NIDS) 연구에 사용되는 데이터셋은 공격에 대한 레이블이 충분하지 않으며, 작은 샘플 사이즈로 인해 오버피팅(overfitting) 문제를 발생시킵니다. 저자들은 1200만 개 이상의 샘플과 정밀한 레이블을 포함하는 데이터셋을 생성함으로써 이러한 한계를 극복하기 위한 노력을 했습니다.

- **Technical Details**: 제안된 트래픽 생성기는 Python 기반으로 개발되었으며, HTTP(S), FTP, SMTP, SSH 등 다양한 프로토콜을 지원합니다. 이 생성기는 정상 사용자 행동을 모사하는 benign 모드와 13가지의 다양한 공격을 실행할 수 있는 attack 모드를 지원합니다. 두 가지 모드에 따라 무작위로 행동을 트리거하여 정상 트래픽을 생성하는 동시에, 공격 유형에 대한 실제 시뮬레이션을 제공합니다.

- **Performance Highlights**: 최종적으로, 이 데이터셋은 12,059,749개의 샘플로 구성되며, 82개의 흐름 수준 특징과 21개의 세분화된 레이블을 포함합니다. 성능 측정을 위해 제안된 데이터셋은 오픈 소스 저장소에 배포될 예정이며, 이는 향후 NIDS 연구 및 개발에 매우 유용한 자원이 될 것입니다. 또한 웹 공격 유형이 다른 데이터셋에서 과소 대표되는 문제를 해결하기 위해, 여러 웹 공격이 포함되어 실제 세계의 트래픽 패턴을 반영합니다.



### InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers (https://arxiv.org/abs/2502.03885)
- **What's New**: 이 논문에서는 InfinitePOD라는 혁신적인 트랜시버 중심의 HBD 아키텍처를 제안합니다. 이는 Optical Circuit Switching(OCS)을 통해 연결성과 동적 스위칭을 통합하여, 유연한 포인트-다-포인트 연결을 제공합니다. 이 디자인은 데이터 센터 전반에 걸쳐 확장 가능성을 제공하면서, 비용 폭등 없이 단일 노드에서의 결함을 격리할 수 있는 이점을 갖추고 있습니다.

- **Technical Details**: InfinitePOD는 SiPh 기반의 저비용 OCS 트랜시버(OCSTrx)와 재구성 가능한 K-Hop 링 토폴로지를 활용하여 설계되었습니다. 이 구조는 전통적으로 구현된 단일 포인트 연결 설계를 넘어, 고르게 배치된 노드들 간의 동적 재구성을 통해 연결을 최적화하고 대역폭 활용을 극대화합니다. 이 아키텍처는 GPU 간의 통신을 효율적으로 최적화하도록 설계되었습니다.

- **Performance Highlights**: Evaluation 결과, InfinitePOD는 NVL-72의 31% 비용으로 운영되며, GPU 낭비 비율은 거의 제로에 가까워 NDV-72 및 TPUv4에 비해 한 차원 가까이 낮습니다. 또한, 노드 결함 비율이 7% 이하일 때 거의 제로에 가까운 교차 ToR 트래픽을 생성하며, NVIDIA DGX 대비 Model FLOPs Utilization을 3.37배 향상시킵니다.



### Knowing When to Stop Matters: A Unified Algorithm for Online Conversion under Horizon Uncertainty (https://arxiv.org/abs/2502.03817)
Comments:
          36 pages, 6 figures

- **What's New**: 이 논문은 동적으로 변화하는 가격 하에서 가변 자원(예: 에너지)을 순차적으로 거래하여 최대 이익을 추구하는 온라인 변환 문제(online conversion problem)를 다룹니다. 이 연구는 거래 기간이 알려지거나 중간에 공개되거나 완전히 불확실한 상황에서 거래 결정을 관리하는 데 중점을 둡니다. 제안된 통합 알고리즘은 이러한 다양한 거래 기간 모델에 대해 최적의 경쟁 보장을 달성하며, 거래별 최대 허용량 제한(box constraints)을 고려합니다.

- **Technical Details**: 저자들은 다양한 기간 불확실성 모델에 대해 OC 문제를 다루는 통합 알고리즘을 제안합니다. 알고리즘은 (i) OC-Known(알려진 기간), (ii) OC-Notice(통지된 기간), (iii) OC-Unknown(완전히 알려지지 않은 기간) 및 (iv) OC-Prediction(예측된 기간)의 네 가지 불확실성 모델을 포함합니다. 이 알고리즘은 경쟁 분석 하에서 강력한 보장을 제공하며, 특정한 예측 오류 상황에서도 강력한 성능을 유지합니다.

- **Performance Highlights**: 제안된 학습 보강 알고리즘(learning-augmented algorithm)은 예측이 정확할 때 최적에 가까운 결과를 얻는 동시에, 예측이 신뢰할 수 없을 때도 안정적인 성과를 유지합니다. 특히, 이 알고리즘은 예측이 부족하거나 정확할 때에도 예측 정보 없이 작동하는 알고리즘보다 뛰어난 성능을 보여줍니다. 본 연구는 기간 불확실성 및 상자 제약을 효과적으로 다루어 OC 알고리즘 설계에 중요한 진전을 이룬 점에서 의의가 있습니다.



### Should Code Models Learn Pedagogically? A Preliminary Evaluation of Curriculum Learning for Real-World Software Engineering Tasks (https://arxiv.org/abs/2502.03806)
Comments:
          Accepted by the 22nd International Conference on Mining Software Repositories (MSR 25)

- **What's New**: 이 연구에서는 Curriculum Learning (CL)과 관련된 기존의 난이도 측정을 활용한 코드 모델 학습 방식을 분석합니다. 특히, 코드 길이와 사이클로마틱 복잡성을 기준으로 다양한 난이도 수준을 설정하여 CL의 효과를 조사합니다. 이전 연구와 달리, 연속적인 난이도 변화에 따른 코드 이해 및 생성 능력 향상을 목표로 합니다.

- **Technical Details**: 연구에서 사용된 두 가지 주요 SE 작업은 코드 클론 탐지(code clone detection)와 코드 요약(code summarization)입니다. CL 훈련 스케줄은 훈련 데이터를 난이도에 따라 쉽게, 보통, 어렵고, 매우 어렵게 나누어 구성하였습니다. CodeT5 모델을 사용하고 CodeXGLUE 벤치마크 데이터로 실험하여 각 난이도 수준이 모델의 성능에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 실험 결과, CL이 기존의 랜덤 스케줄보다 코드 클론 탐지 및 코드 요약 작업에서 성능을 개선하지 못했음을 확인했습니다. 흥미롭게도, 모델 성능은 훈련 데이터의 25% 만으로도 성숙기에 도달했으며, 추가 학습 데이터는 미미한 성능 향상만을 가져왔습니다. 이러한 결과는 모델의 표현 용량 제한이나 작업의 자연스러운 난이도 때문일 수 있음을 시사합니다.



### SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning (https://arxiv.org/abs/2502.03801)
- **What's New**: 본 논문은 federated learning (FL)의 데이터 프라이버시를 유지하면서 협력적 모델 훈련을 가능하게 하지만, 클라이언트 측 데이터 오염 공격(client-side data poisoning attacks, DPAs)과 모델 오염 공격(model poisoning attacks, MPAs)에 취약하다는 점을 강조합니다. 다양한 방어 전략들이 제안되었으나, 이들의 평가가 제한적인 공격 방법으로 이루어져 효과성에 대한 우려가 있었습니다. 본 연구는 DPAs와 MPAs에 대한 방어 수단을 통합적으로 분석하고, 이들 두 영역 간의 차이를 명확히 하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 poisoning 공격과 방어 전략에 대한 체계적인 분류(taxonomy)를 제시하며, 각 기법의 설계와 강점, 한계를 설명합니다. 또한 다양한 FL 알고리즘과 데이터 이질성(data heterogeneity)에 대한 통합 비교 평가를 수행하여 개별 및 상호 효과성(mutual effectiveness)을 검증하고 향후 연구를 위한 주요 통찰(insights)을 도출했습니다. 이를 통해 FLPoison이라는 통합 벤치마크를 만들어 15개의 대표적인 오염 공격과 17개의 방어 전략을 평가할 수 있도록 하여, 향후 연구에 기여하고자 합니다.

- **Performance Highlights**: 제안된 연구는 방어 전략이 DPAs와 MPAs에 대해 상호 작용할 수 있는 방안을 제시하며, 각 방어의 성능을 객관적으로 비교함으로써 방어 메커니즘의 발전 방향에 대한 귀중한 통찰을 제공합니다. 또한, FLPoison은 높은 모듈성(modularity)과 확장성(scalability)을 지원하여 다양한 공격 시나리오를 평가하는 데 효과적인 자료를 제공합니다. 코드 또한 공개되어 있어 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### Guiding Two-Layer Neural Network Lipschitzness via Gradient Descent Learning Rate Constraints (https://arxiv.org/abs/2502.03792)
Comments:
          26 pages, 8 figures

- **What's New**: 이번 연구에서는 Lipschitz 활성화 함수를 가진 두 층 신경망의 학습 속도(learning rate)에 점진적인 감쇠를 적용함으로써, 높은 수준의 Lipschitz 정규성을 보장할 수 있음을 보여줍니다. 이는 일반적인 경량 감소(gradient descent) 방법을 사용하면서도 경험적 위험(empirical risk)의 수렴 속도에 영향을 미치지 않는다는 점에서 의미가 있습니다. 이를 통해 통계적 행동이 과도한 매개변수(overparameterization)에 독립적이라는 점도 확인되었습니다.

- **Technical Details**: 본 논문에서는 평균 제곱 오차(mean squared error, MSE) 손실 함수를 통해 두 층 신경망을 훈련하며, 이를 통해 Huber 손실 함수에서의 위험을 추정하고 검증합니다. 특히, 학습 속도를 점차 줄이는 방식이 최적 수렴을 보장하며, Lipschitz 정규성을 띠는 두 층 MLP를 교육하는 데 필요한 조건을 제시합니다. 이는 최적의 수렴 속도를 보장하기 위해 일반적인 경량 감소 방법과 호환 가능하다는 점을 강조합니다.

- **Performance Highlights**: 이론적 결과는 간단한 수치 실험을 통해 검증되었으며, 상수 단계 크기로 훈련된 네트워크가 점진적으로 감쇠하는 학습 속도로 훈련된 네트워크와 유사한 학습 및 정규성 속성을 보임을 확인했습니다. 이러한 발견은 네트워크가 이미 높은 정규 학습 모델임을 시사하며, 이로 인해 두 층 신경망이 갖는 일반화 경계가 널리 영향을 미칠 수 있음을 보여줍니다.



### Brain Tumor Identification using Improved YOLOv8 (https://arxiv.org/abs/2502.03746)
- **What's New**: 이번 연구에서는 MRI 이미지 내의 뇌종양을 정확하게 탐지하기 위해 수정된 YOLOv8 모델을 제안합니다. 이 모델은 탐지 헤드에서 Non-Maximum Suppression(NMS) 알고리즘을 Real-Time Detection Transformer(RT-DETR)으로 대체하여, 손으로 설계된 불필요한 경계 상자를 제거합니다. 또한, 일반적인 컨볼루션 블록을 ghost convolution으로 교체하여 연산 및 메모리 비용을 줄이며 빠른 추론을 가능하게 합니다.

- **Technical Details**: 본 연구에서 제안하는 모델은 YOLOv8의 백본에 vision transformer 블록을 도입하였으며, 이를 통해 문맥 인식을 기반으로 한 특징을 추출합니다. RT-DETR은 핸드 디자이닝 구성 요소를 제거하여 현대적인 딥러닝 모델의 경량화를 이루었습니다. ghost convolution은 고속 처리를 가능하게 하며, 자원 제약 환경에서도 높은 정확성을 유지합니다.

- **Performance Highlights**: 제안된 모델은 공개된 뇌종양 데이터셋을 사용하여 훈련되었으며, 원본 YOLOv8 모델 및 다양한 객체 탐지기(Faster R-CNN, Mask R-CNN, YOLO 계열, SSD, RetinaNet 등)보다 우수한 성능을 나타냈습니다. 특히, 제안된 모델은 0.91 mAP(mean Average Precision)@0.5를 달성하여 d극적인 성과를 보였습니다.



### MD-BERT: Action Recognition in Dark Videos via Dynamic Multi-Stream Fusion and Temporal Modeling (https://arxiv.org/abs/2502.03724)
- **What's New**: 이 연구는 저조도 비디오에서의 동작 인식을 위한 새로운 접근법인 MD-BERT를 소개합니다. MD-BERT는 감마 보정(gamma correction)과 히스토그램 평활화(histogram equalization) 기법을 포함한 다중 스트림(multi-stream) 아키텍처를 채택하여 저조도 환경에서의 도전 과제를 해결합니다. 또한, Dynamic Feature Fusion(DFF) 모듈을 통해 다양한 요소를 통합하여 비디오 프레임 간의 복잡한 상호작용을 효과적으로 포착합니다.

- **Technical Details**: MD-BERT는 Raw dark frames, gamma-enhanced frames, histogram-equalized frames의 세 가지 보조 입력 스트림을 통해 다양한 시각적 특징을 효과적으로 처리합니다. DFF 모듈은 지역적(local) 및 전역적(global) 맥락 정보를 조화롭게 통합하여 고유한 시각적 정보를 강조할 수 있도록 설계되었습니다. BERT 기반의 아키텍처를 통해 긴 시간의 종속성을 포착하며, 포지셔널 인코딩(positional encoding)과 멀티헤드 어텐션(multi-head attention)을 사용하여 과제 인식을 향상시킵니다.

- **Performance Highlights**: ARID V1.0 및 ARID V1.5 데이터셋에서 MD-BERT는 기존의 방법들보다 우수한 성능을 보여주며, 저조도 환경에서도 최신 성능 기준을 확립합니다. 각 입력 스트림의 개별 기여도를 강조하는 Ablation 연구도 실시하여 제안된 DFF 및 BERT 모듈의 효과성을 입증했습니다. 이 연구는 저조도 비디오에서 행동 인식의 새로운 수준을 제시하며, 비디오 인식 기술의 발전에 기여할 것으로 기대됩니다.



### Detecting Backdoor Attacks via Similarity in Semantic Communication Systems (https://arxiv.org/abs/2502.03721)
- **What's New**: 이 논문은 Generative AI (GAI)를 활용하여 의미적 정보를 전송하는 Semantic communication 시스템의 보안 문제, 특히 Backdoor 공격에 대한 새로운 방어 메커니즘을 제안합니다. 기존 방어 메커니즘이 모델 구조를 변경하거나 데이터 형식 제약을 가하는 것과 달리, 제안된 방법은 의미적 유사성을 이용하여 Backdoor 공격을 탐지하며, 이러한 방식을 통해 악의적 샘플을 효과적으로 식별합니다.

- **Technical Details**: 이 시스템은 훈련 단계와 추론 단계로 나뉘며, 훼손된 샘플을 드러내기 위해 의미적 유사성을 바탕으로 한 임계값 기반 탐지 프레임워크를 사용합니다. 훈련 중에 입력 샘플과 클린 데이터 간의 의미적 편차를 분석하고, 이를 통해 세팅된 임계값 아래의 샘플은 훼손된 것으로 플래그가 지정됩니다. 기존 모델 구조를 변경하지 않고도 이러한 탐지 방식을 통해 깨끗한 샘플의 처리가 가능하다는 것이 핵심입니다.

- **Performance Highlights**: 실험 결과, 제안된 방어 메커니즘은 다수의 훼손 비율에 걸쳐 높은 탐지 정확도와 재현율을 달성했습니다. 이는 의미적 유사성 탐지 기법이 Backdoor 공격에 대해 효과적으로 작동하며, 모델의 무결성이나 클린 샘플에 대한 정보 전송의 정확성을 손상시키지 않고도 탐지가 가능하다는 것을 보여줍니다.



### Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignmen (https://arxiv.org/abs/2502.03714)
- **What's New**: 본 연구는 다수의 사전 훈련된 깊은 신경망(DNN)에서 공유되는 해석 가능한 개념을 발견하고 정렬하기 위한 'Universal Sparse Autoencoders(USAEs)' 프레임워크를 제안합니다. 기존의 개념 기반 해석 가능성 방법들은 단일 모델에 초점을 맞춘 반면, USAEs는 여러 모델의 내부 활성화를 동시에 재구성하고 해석할 수 있는 보편적인 개념 공간을 공동 학습합니다. 이를 통해 다양한 작업, 아키텍처 및 데이터 세트를 걸쳐 일반화되는 개념의 공통 요인을 포착하게 됩니다.

- **Technical Details**: USAEs는 단일의 과잉완전한 희소 오토인코더(SAE)를 훈련하여 어떤 모델의 활성화를 입력받고 다른 모델의 활성화에 근사화하는 구조로 설계되었습니다. 전통적인 방법과는 달리 USAE는 end-to-end 방식으로 개념 정렬을 적용하여 여러 모델 간의 효율적인 상호 작용을 가능하게 합니다. 이 방식은 DNN의 내부 표현을 이해하는 데 유용하며, 다수의 비전 모델에 적용하여 저수준 특징부터 고수준 구조까지 다양한 보편 개념을 발견했습니다.

- **Performance Highlights**: USAEs는 세 가지 다양한 비전 모델에 적용되어 흥미로운 발견을 제시하였습니다. 첫째, 낮은 추상화 수준부터 높은 추상화 수준까지 광범위한 보편 개념을 발견하였습니다. 둘째, 개념의 보편성에 중요한 상관관계를 관찰하였으며, DinoV2는 다른 모델에 비해 독특한 특징을 갖고 있다는 정량적 및 정성적 증거를 제공하였습니다. 또한, 보편적 훈련은 모델 특화형 SAE 훈련에서는 발견되지 않는 공유 표현을 허용합니다.



### MultiQ&A: An Analysis in Measuring Robustness via Automated Crowdsourcing of Question Perturbations and Answers (https://arxiv.org/abs/2502.03711)
Comments:
          AAAI 2025 Workshop on Preventing and Detecting LLM Misinformation (PDLM) (Oral)

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 답변에서 발생할 수 있는 hallucination 문제를 해결하기 위한 시스템인 MultiQ&A를 제안합니다. MultiQ&A는 LLM이 생성한 답변의 일관성과 견고성을 평가하는 체계적인 접근 방식으로, 다양한 질문의 변형을 만들고 이에 대한 답변을 crowdsourcing하고 있습니다. 실험을 통해 1.9백만 개의 질문 변형과 2.3백만 개의 답변을 검토하였으며, gpt-3.5-turbo 모델이 변수에 대해 비교적 일관성을 유지함을 보여줍니다.

- **Technical Details**: MultiQ&A는 세 가지 구성 요소로 이루어진 강력한 다단계 파이프라인을 갖고 있습니다: Query Rewriter, Answer Generator, Aggregator. Query Rewriter는 원래의 쿼리(q0)를 다양한 의미적으로 일관된 변형으로 변환하며, Answer Generator는 이러한 변수들에 대해 독립적인 gpt-3.5-turbo 에이전트를 사용하여 다수의 답변을 생성합니다. Aggregator는 개별 답변을 통합하여 최종 결과를 도출하는 역할을 수행합니다.

- **Performance Highlights**: MultiQ&A의 실험 결과, 1.9 백만 개의 질문과 2.3 백만 개의 답변을 분석하여 실제 시나리오를 모방한 결과, gpt-3.5-turbo 모델이 의미적으로 안정적이면서도 다양한 표현을 생성함을 추가적으로 확인하였습니다. MultiQ&A는 LLMs의 변동성을 강조하며, 각 질문에 대한 모델의 변화를 보여주는 것을 목표로 한다는 점에서 큰 의의가 있습니다. 따라서, 이 시스템은 기관에서 LLM를 채택하기 위한 신뢰성 있는 프레임워크로 작용할 가능성을 제공합니다.



### First-ish Order Methods: Hessian-aware Scalings of Gradient Descen (https://arxiv.org/abs/2502.03701)
- **What's New**: 이 논문에서는 경량 기계 학습 문제 최적화를 위한 새로운 접근법을 제안합니다. 기존의 경량 방법들이 가진 학습 속도 (learning rate) 선택의 어려움을 해결하기 위해 Hessian 정보를 활용한 비율 조정 방법을 도입했습니다. 이 방법은 함수의 곡률을 고려하여 비선형 환경에서도 안정적인 스텝 크기 보장을 제공합니다.

- **Technical Details**: 본 연구에서는 Hessian 정보를 사용하여 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량 경량

- **Performance Highlights**: 이 접근법은 비선형 상황에서도 글로벌 수렴을 달성하며, 실험적으로도 다양한 기계 학습 작업에서 효과성을 입증했습니다. 특히, 제안된 방법은 스텝 크기를 조정할 때 기존의 첫 번째 및 두 번째 순서 방법의 좋은 특성을 합친 듯한 성능을 나타냅니다. 이는 컴퓨팅 overhead를 줄이고 더 빠른 수렴 속도를 제공함으로써 머신 러닝 문제 최적화에 있어 새로운 방향성을 제시합니다.



### Cascaded Learned Bloom Filter for Optimal Model-Filter Size Balance and Fast Rejection (https://arxiv.org/abs/2502.03696)
- **What's New**: 이 논문에서는 Cascaded Learned Bloom Filter (CLBF)라는 새로운 구조의 학습된 블룸 필터를 제안합니다. CLBF는 머신 러닝 모델과 블룸 필터의 크기 간의 최적의 균형을 유지하고, 거부 시간(reject time)을 최소화하는 데 초점을 맞추고 있습니다. 또한, 이 방법은 기존의 학습된 블룸 필터보다 메모리 사용량을 최대 24% 줄이고, 거부 시간을 14배까지 감소시키는 실험 결과를 보여줍니다.

- **Technical Details**: CLBF는 동적 프로그래밍(dynamic programming)을 기반으로 하여 최적의 하이퍼파라미터 설정에 따라 자동으로 구성 조정을 수행합니다. CLBF는 더 큰 머신 러닝 모델을 학습한 후, 적절하게 크기를 줄이는 방법을 제공합니다. 또한, 중간 블룸 필터와 머신 러닝 모델의 잠정적인 출력에 기반한 분기를 통해 신속한 거부가 가능하도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, CLBF는 블룸 필터의 최신 기법인 Partitioned Learned Bloom Filter (PLBF)와 비교했을 때 메모리 사용을 24% 줄이고, 거부 시간을 14배 단축하였습니다. 이러한 성과는 CLBF가 실질적으로 메모리 효율성과 빠른 응답 시간을 동시에 달성할 수 있음을 보여줍니다.



### Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Fr (https://arxiv.org/abs/2502.03687)
- **What's New**: 이번 연구에서는 2D 의료 영상 분류를 위한 클래스 조건부 확산 모델(class conditional diffusion models)의 가능성을 처음으로 탐구했습니다. 새로운 다수 투표 기반 방법을 개발하여 의료 확산 분류기의 성능을 향상시키는 것을 목표로 하고 있습니다. 특히, 기존의 판별 모델(discriminative classifiers)과 비교해도 늦은 훈련 없이 뛰어난 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 확산 모델(difussion models)을 사용하여 의료 영상 분류 작업을 위한 새로운 방법론을 제시합니다. 확산 모델은 데이터를 점진적으로 노이즈로 변환하는 고정된 전방 과정과 이 과정을 통해 학습된 역방향 모델(backward model)로 구성됩니다. 새로운 다수 투표 알고리즘, 반사실적 설명(counterfactual explainability), 불확실성 정량화(uncertainty quantification) 기능을 포함하여 성능을 향상시키는 방법을 소개합니다.

- **Performance Highlights**: CheXpert와 ISIC 흑색종 데이터셋에서의 실험 결과, 확산 모델에서 파생된 분류기가 기존의 의료 영상 판별 모델과 경쟁할 만한 성능을 발휘했습니다. 예를 들어, Stable Diffusion 모델은 ISIC와 CheXpert 각각 100% 및 95%의 분류 정확도를 달성했으며, 신뢰도가 높아짐에 따라 모델의 정확도도 크게 향상되었습니다. 이러한 결과는 모델이 제공하는 불확실성 분석이 의료 분야의 신뢰성과 안전성을 높일 수 있음을 보여줍니다.



### Controlled LLM Decoding via Discrete Auto-regressive Biasing (https://arxiv.org/abs/2502.03685)
- **What's New**: 이번 논문에서는 기존의 에너지 기반 디코딩 방식을 개선하여, 텍스트 생성의 제어 가능성을 높이는 새로운 접근 방식을 제시합니다. 기존의 기법들은 매개 변수를 조정하더라도 유창성(fluency)과 제약(constraint) 만족 간의 균형이 부족한 문제를 가지고 있었습니다. 이에 따라, 우리는 연속 공간에서 샘플링하는 것이 아니라, 텍스트 토큰의 자연스러운 이산(discrete) 공간에서 작동하는 Discrete Auto-regressive Biasing이라는 새로운 디코딩 알고리즘을 제안합니다.

- **Technical Details**: 우리가 제안한 방법은 생성된 시퀀스와 보조 바이어스 시퀀스에 대한 결합 분포를 정의하여 제어된 텍스트 생성을 위한 새로운 수식을 도입합니다. 이를 통해, 기울기(gradient)를 활용하면서 완전히 이산 텍스트 도메인에서 작동하는 Langevin-within-Gibbs 샘플링 알고리즘을 제시합니다. 이 접근 방식은 제약 만족도를 효과적으로 향상시키면서도, 유창성을 유지하거나 심지어 개선하는 데 기여합니다.

- **Performance Highlights**: 우리의 제어된 디코딩 방법은 감정 제어(sentiment control), 언어 해독(language detoxification), 키워드 유도 생성(keyword-guided generation) 등의 작업에서 두드러진 장점을 나타냅니다. 제안된 방법은 이전의 기술에 비해 계산 비용이 낮으면서도 성능을 유지하거나 향상시키는 것을 보여줍니다. 이러한 결과는 대규모 언어 모델의 출력에 대해 사용자 정의 제약을 효과적으로 적용할 수 있는 가능성을 제시합니다.



### Reflection-Window Decoding: Text Generation with Selective Refinemen (https://arxiv.org/abs/2502.03678)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 텍스트 생성에서 자기 회귀 해독의 단점을 이론적으로 규명하고, 생성된 내용을 정제하기 위한 슬라이딩 리플렉션 윈도우와 일시정지 기준을 포함한 프레임워크를 제안합니다. 이는 정제(refinement)와 생성을 교환적으로 수행할 수 있는 방법을 통해 효율성과 최적성을 동시에 충족시킬 수 있습니다. 이 접근 방식은 기존의 해독 방식보다 상당한 개선을 보여줍니다.

- **Technical Details**: 연구는 텍스트 생성에서 자기 회귀 방식의 단점을 강조합니다. 자기 회귀 방식은 이전에 생성된 내용을 수정하는 자연스러운 메커니즘이 부족하여 최적의 응답을 보장하지 못합니다. 본 논문에서는 슬라이딩 리플렉션 윈도우와 일시정지 기준을 도입하여 다수의 토큰을 병렬로 예측할 수 있도록 하고, 이로 인해 해독 과정 중 독립적으로 정제와 생성을 오갈 수 있는 구조를 제시합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 접근 방식이 기존 해독 방법보다 성능이 개선되었음을 보여줍니다. 이 방법은 성능 및 효율성 면에서 빔 탐색(beam search)과 유사하거나 더 나은 결과를 제공합니다. 따라서 새로운 접근은 텍스트 생성을 위한 효율적이고 최적화된 해법을 제공하는 데 기여합니다.



### Physically consistent predictive reduced-order modeling by enhancing Operator Inference with state constraints (https://arxiv.org/abs/2502.03672)
Comments:
          27 pages, 10 figures

- **What's New**: 본 연구에서는 char combustion과 같은 복합 다물리학 시스템의 수치 시뮬레이션에서 발생하는 물리적 제약사항을 포함하는 새로운 접근법을 제시합니다. 이 방법은 Operator Inference(연산자 추정)의 기법을 보강하여 높은 차원의 비선형 편미분 방정식으로 지배되는 시스템의 저차원 표현을 학습할 수 있게 합니다. 이후 제안된 모델은 일반적인 Operator Inference 및 다른 안정성 향상 방법들과 비교됩니다.

- **Technical Details**: 연구에서는 저차원 모델 예측 과정에서 특성 제약(state constraints)을 통합하여 모델의 안정성을 개선합니다. 또한, 최적의 정규화 하이퍼파라미터를 선택하는 새로운 방법을 제안하며, 이 성과는 특정 성능 지표에 기반합니다. 이를 통해 시스템의 복잡성을 줄이는 동시에 안정성과 정확성을 향상시킬 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 char combustion에 대한 모델 학습에서 안정성과 정확성이 다른 방법보다 우수한 상태 예측을 제공합니다. 특히, 이 기법은 훈련 영역을 넘어 200% 이상 외삽할 수 있으며, 계산적으로 효율적이고 물리적으로 일관성을 유지합니다.



### Contrastive Learning for Cold Start Recommendation with Adaptive Feature Fusion (https://arxiv.org/abs/2502.03664)
- **What's New**: 이 논문에서는 대조 학습(contrastive learning)을 통합한 콜드 스타트 추천 모델을 제안합니다. 콜드 스타트(Cold Start) 시나리오에서 사용자 및 항목 상호작용 데이터가 부족하여 추천 시스템의 성능 저하 문제를 해결하는 것을 목표로 합니다. 이 모델은 적응형 특성 선택 모듈을 통해 주요 특성의 가중치를 동적으로 조정하며 추천 성능을 향상시킵니다.

- **Technical Details**: 모델은 사용자 속성(user attributes), 항목 메타 정보(item meta-information), 그리고 맥락적 특징(contextual features)을 효과적으로 통합하기 위해 다중 모달(feature fusion) 메커니즘을 결합합니다. 대조 학습 메커니즘을 도입하여 양성 및 음성 샘플 쌍을 구성함으로써 특성 표현의 강인성과 일반화 능력을 향상시킵니다. 실험은 MovieLens-1M 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 제안된 모델은 HR, NDCG, MRR, Recall 등에서 Matrix Factorization, LightGBM, DeepFM, AutoRec와 같은 주류 추천 방법보다 현저히 우수한 성능을 보였으며, 특히 콜드 스타트 시나리오에서 두드러진 효과를 나타냈습니다. Ablation 실험을 통해 각 모듈의 모델 성능 향상에 대한 중요성을 검증하였고, 학습률 민감도 분석에서는 적절한 학습률이 모델의 최적화 효과에 필수적이라는 것을 보여주었습니다.



### Energy & Force Regression on DFT Trajectories is Not Enough for Universal Machine Learning Interatomic Potentials (https://arxiv.org/abs/2502.03660)
- **What's New**: 이번 연구는 Universal Machine Learning Interactomic Potentials (MLIPs)를 통해 물질 발견을 가속화하는 방법론을 제안합니다. 기존의 MLIP 연구는 밀도 범함수 이론(DFT)에 과도하게 의존하여 신뢰성과 정확성을 보장하는 데 한계를 보여주고 있습니다. 이 논문에서는 더 정확한 시뮬레이션 방법을 이용해 MLIP 교육 데이터를 생성하고, MLIP 메트롤로지 도구를 통해 내재적인 작동 방식을 이해하며, 계산적으로 효율적인 MLIP를 개발할 필요성을 강조합니다.

- **Technical Details**: MLIPs는 양자적 정확성으로 다양한 물질의 잠재 에너지를 모델링하기 위해 훈련됩니다. 이 연구는 Coupled Cluster Theory와 같은 보다 정확한 시뮬레이션 방법을 도입하여 실제 물질 응용의 복잡성을 반영하는 데이터 세트를 생성해야 한다고 주장합니다. 또한, MLIP 평가 방법을 통해 실험적으로 측정된 특성을 모델링하는 MD 시뮬레이션에서 MLIP의 능력과 한계를 효과적으로 이해할 수 있도록 해야 합니다.

- **Performance Highlights**: 효율적인 MLIP 추론 워크플로우는 MD 시뮬레이션에서 전통적인 방법을 능가할 수 있는 잠재력을 unlock(잠금 해제)합니다. 이를 통해 MLIPs는 더 크고 화학적으로 복잡한 시스템을 모델링할 수 있게 됩니다. 최종적으로, 이러한 방향성을 통해 MLIPs는 복잡한 물질을 정확하게 모델링하여 실물 스케일에 적용할 수 있는 대안으로 자리잡을 수 있음을 보여줍니다.



### A Study in Dataset Distillation for Image Super-Resolution (https://arxiv.org/abs/2502.03656)
- **What's New**: 이 논문은 대규모 데이터셋을 소형화된 합성 샘플로 압축하는 개념인 데이터셋 증류(Dataset Distillation)에 대해 탐구하고 있습니다. 특히, 이미지 분류에 주로 연구가 집중된 점을 넘어 이미지 초해상도(Super-Resolution, SR) 분야로의 응용을 확대하였습니다. 실험을 통해 전체 데이터셋과 유사한 SR 성능을 유지하면서 데이터셋 크기를 91.12% 줄일 수 있음을 보여줍니다.

- **Technical Details**: 고해상도(HR) 이미지 재구성을 위해 저해상도(LR) 이미지에서 SR 모델을 훈련시키는 데 필요한 데이터셋 증류 기법을 사용하고 있습니다. SR 품질을 유지하면서 훈련 데이터의 양을 줄이는 데 중점을 두며, 독특하게도 픽셀 공간과 잠재 공간을 비교 분석합니다. 최적화 전략 및 초기화 방법에 대해서도 심도 있는 분석을 수행하여 메모리 효율성과 계산 비용을 최적화하고자 합니다.

- **Performance Highlights**: 본 연구는 합성 데이터셋을 통해 데이터 크기를 대폭 줄이면서도 SR 성능에서 경쟁력 있는 결과를 달성하는 방법론을 제시합니다. 또한, 추가적인 통찰력을 제공함으로써 향후 메모리 효율적인 SR 모델 교육에 대한 기초를 마련하고 있습니다. 결과적으로, 데이터셋 증류와 SR 간의 간극을 메우는 데 중요한 연구 방향을 설정하고 있습니다.



### Rule-based Evolving Fuzzy System for Time Series Forecasting: New Perspectives Based on Type-2 Fuzzy Sets Measures Approach (https://arxiv.org/abs/2502.03650)
- **What's New**: 이 논문에서는 새로운 진화형 퍼지 시스템(evolving fuzzy system), 즉 ePL-KRLS-FSM+을 제안합니다. 이 모델은 진화형 퍼지 모델링 방법을 개선하여 participatory learning(PL), kernel recursive least squares(KRLS), type-2 퍼지 로직 및 데이터 변환을 결합합니다. 이를 통해 데이터의 불확실성을 더욱 잘 처리할 수 있는 type-2 퍼지 집합을 생성하고 측정할 수 있게 됩니다.

- **Technical Details**: ePL-KRLS-FSM+는 type-2 퍼지 집합을 생성하기 위해 이차 멤버십 함수(secondary membership function)를 도입하며, 새로운 방법으로 예측 모델의 호환성 척도(compatibility measure)를 계산하고, 퍼지 집합 설계를 위한 새로운 방법을 제공합니다. 이 모델은 다양한 데이터 세트를 비교할 수 있는 새로운 가능성을 추가하며, 퍼지 집합을 위한 변수를 변경하는 기능도 포함되어 있습니다.

- **Performance Highlights**: 제안된 모델의 예측 성능은 Mackey-Glass 지연 미분 방정식과 대만 자본화 가중 주가 지수(TAIEX) 데이터셋을 통해 평가되었습니다. 성능 비교 결과, ePL-KRLS-FSM+ 모델은 type-1 모델 및 기타 예측 방법들에 비해 낮은 오류 지표와 적은 최종 규칙 수를 기록하며, 경쟁력을 갖춘 것으로 나타났습니다.



### Looking for the Inner Music: Probing LLMs' Understanding of Literary Sty (https://arxiv.org/abs/2502.03647)
- **What's New**: 최근 연구에 따르면, 언어 모델이 전통적인 스타일 분석에서 생각했던 것보다 훨씬 동안의 짧은 문학 구절의 저자를 식별하는 데 훈련될 수 있다는 것을 보여줍니다. 우리는 이러한 결과를 저자 식별로 재현하고 새로운 데이터셋을 통해 소설 장르 분석으로 확장합니다. 흥미롭게도, LLM들은 저자와 장르를 구별할 수 있지만 서로 다른 방식으로 작동하며 이는 메모리 기반 접근과 특성 학습 방식의 차이를 보여줍니다.

- **Technical Details**: 우리는 훈련된 LLM이 저자 스타일을 정의하는 특징을 찾기 위해 세 가지 방법을 사용합니다. 여기에는 입력 텍스트에 대한 직접적인 구문적 제거(syntactic ablations) 및 모델의 내부 구조를 분석하는 방법이 포함됩니다. 결과적으로, 저자 스타일은 장르 레벨 스타일보다 정의하기가 더 쉽고, 작은 구문적 결정 및 문맥적 단어 사용에 더 큰 영향을 받는 것으로 나타났습니다.

- **Performance Highlights**: 모델들은 매우 짧은 텍스트(20~50 단어)에서 저자와 장르를 인식하는 데 있어 무작위 정확도를 초과하는 성과를 보였습니다. 가장 큰 LLM인 Llama-3와 Flan-T5는 27명의 저자 및 5개의 장르에 대한 텍스트를 각각 50% 이상의 정확도로 분류하여 가장 높은 성능을 기록했습니다. 이러한 결과는 문학적 신호가 이 규모에서도 존재함을 확인시켜 주며, 저자 스타일과 장르 스타일을 구별하는 데 사용되는 특징이 다름을 보여줍니다.



### SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models (https://arxiv.org/abs/2502.03638)
- **What's New**: 본 연구에서는 SymmCD라는 새로운 확산 기반 생성 모델을 제안하여 결정의 대칭성을 생성 과정에 명확히 포함시켜, 실제 결정의 대칭을 반영하는 다양한 새로운 결정 재료를 생성하는 방법을 소개합니다. 기존 모델들이 대칭 정보의 복제를 통해 생성된 결정만을 제공한 반면, SymmCD는 비대칭 단위(asymmetric unit)를 분해하여 대칭 변환(symmetric transformations)을 학습하는 방식으로 전환됩니다.

- **Technical Details**: SymmCD는 결정 재료를 비대칭 단위와 그에 적용할 대칭 변환으로 나누어 확산 과정을 통해 이들의 결합 분포를 학습합니다. 이 과정에서 대칭 변환은 새로운 해석 가능한 표현으로 모델링되어 다양한 결정 대칭 그룹에 걸쳐 일반화할 수 있는 능력을 제공합니다. 연구에서는 새로운 모델을 기존 데이터셋인 Materials Project의 하위 집합에 적용하여, 현실적인 대칭과 예측된 특성을 갖춘 다양한 결정들을 생성하는데 성공하였습니다.

- **Performance Highlights**: SymmCD는 기존 방법보다 우수한 성능을 보여주며, 특히 대칭 구조를 생성하는데 필요한 계산 효율성이 개선되었습니다. 연구 결과는 SymmCD가 안정적인 구조를 생성하는 데에서 이전 모델들과 동등한 성능을 발휘하며, 다양한 대칭성과 구조 다양성을 잘 나타낸다는 것을 강조합니다. 이러한 성과는 새롭게 설계된 대칭 정보 표현 덕분에 가능하였습니다.



### REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations (https://arxiv.org/abs/2502.03629)
- **What's New**: 기존의 이미지 편집 모델들이 실제 사용자 요구를 충족시키지 못하는 문제를 다룬 REALEDIT(Real Edit)라는 새로운 대규모 데이터셋을 소개합니다. 이 데이터셋은 Reddit에서 수집된 진짜 사용자 요청과 인간에 의해 편집된 이미지를 포함하고 있으며, 9300개의 평가 예제를 포함하고 있어 다양한 실제 요구를 테스트할 수 있습니다. REALEDIT는 인간의 편향을 감소시키고 다양한 사용자 요구를 반영하는 구조로 설계되었습니다. 이 연구의 결과는 기존 모델이 이러한 작업에서 부족한 점이 있음을 강조합니다.

- **Technical Details**: REALEDIT 데이터셋은 사람에 의해 편집된 이미지와 그에 대한 요청을 기반으로 한 48K의 훈련 예제와 9300개의 테스트 예제를 포함하고 있습니다. 이를 위해 데이터 수집 파이프라인을 구성하였으며, 두 개의 주요 서브레딧인 r/PhotoshopRequest와 r/estoration에서 받은 요청을 기반으로 하여 데이터를 구성하였습니다. 편집된 이미지들은 원본 이미지와 편집 지침을 포함하여 사용자의 실제 요청을 반영하는 형태로 수집되었습니다. 이 데이터셋은 진짜 편집 요구 사항을 더 효과적으로 반영하며, 기존의 합성 데이터세트의 한계를 극복하고자 합니다.

- **Performance Highlights**: REALEDIT 모델은 기존의 최고 성능 모델보다 165포인트 높은 Elo 스코어를 기록하며 주목받았습니다. 또한 VIEScore와 같은 자동화된 메트릭에서도 92%의 향상을 보여줍니다. 모델은 Reddit에서 새로운 요청에 대해 긍정적인 피드백을 받았으며, 편집 외에도 진짜 편집된 이미지의 탐지 성능이 향상될 가능성도 확인되었습니다. 이 연구는 이미지 편집 작업 외에도 다양한 AI 기반 응용 분야에 대한 데이터셋의 유용성을 강조합니다.



### The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering (https://arxiv.org/abs/2502.03628)
- **What's New**: 본 논문은 Large Vision-Language Models (LVLMs)에서 발생하는 환각 현상(hallucination)에 대해 연구하였습니다. LVLMs는 시각적 및 텍스트 정보를 효과적으로 처리할 수 있지만, 종종 의미는 일관되나 시각적으로 지지되지 않는 내용을 생성을 야기합니다. 이를 해결하기 위해, 논문에서는 새로운 VISTA(VIsual Information Steering with Token-logit Augmentation) 프레임워크를 제안하였으며, 이는 환각을 줄이는 동시에 진정한 정보를 증진시키는 방법을 제공합니다.

- **Technical Details**: 연구에서 LVLMs의 토큰 로짓(token logits) 랭킹을 조사하여 세 가지 주요 패턴을 발견하였습니다. 첫 번째는 시각 정보의 점진적 손실로, 이는 진정한 토큰의 우선 순위가 감소하고 환각 토큰의 우선 순위가 상승하는 현상이 발생합니다. 두 번째 패턴은 의미 있는 토큰이 모델의 중간 계층에서 최고 활성도를 나타내며, 세 번째는 시각적으로 진정한 토큰이 최종 단계에서 결정되지 않을 수 있지만 여전히 높은 랭킹을 유지한다는 것입니다. 이를 바탕으로 VSTA는 시각 정보 강화 및 기능적 토큰 사용을 조합하여 작동합니다.

- **Performance Highlights**: VISTA는 실험을 통해 기존 방법에 비해 환각 현상을 약 40% 감소시키는 효과를 보여주었습니다. 다양한 모델 아키텍처(LLaVA, Shikra, MiniGPT-4, InstructBLIP)에서 높은 성능을 나타내며, 추가적인 훈련이나 모델 수정이 필요 없다는 장점이 있습니다. 이 프레임워크는 다양한 디코딩 전략에 쉽게 적용 가능하며, 열린 생성(open-ended generation) 및 시각적 질문 응답(visual question answering) 등 여러 평가 프로토콜에서 우수한 결과를 제공합니다.



### Multivariate Conformal Prediction using Optimal Transpor (https://arxiv.org/abs/2502.03609)
- **What's New**: 이번 연구는 다변량 공간에서의 conformal prediction (CP) 방법을 확장하여 머신러닝 모델의 불확실성을 정량화하는 새로운 접근법, OTCP(Optimal Transport Conformal Prediction)를 제안합니다. 기존의 CP 접근법은 수치 기준(score functions)으로 단일 변량 분석에 중점을 두었지만, OTCP는 최적 수송(optimal transport) 이론을 활용하여 다변량 환경에서의 예측 세트를 구성합니다. 이를 통해 기존의 CP의 적용 범위를 넓히고, 분포에 구애받지 않는 커버리지 보장을 유지합니다.

- **Technical Details**: OTCP는 다변량 예측의 공동 행동(joint behavior)을 포착하는 분포 자유의 불확실성 세트를 생성하기 위해 최적 수송 이론을 사용합니다. 이 방법은 differentiable transport map 추정기를 활용하여 다변량 점수 함수(multivariate score functions)의 정의와 계산을 가능하게 합니다. 또한 Sinkhorn 문제를 통해 계산된 엔트로픽 맵(entropic map)을 사용하여 접근 방식을 구현하고, 커버리지 보장을 유지하면서도 계산적으로 효율적으로 설계되었습니다.

- **Performance Highlights**: OTCP의 효과는 최근 발표된 다변량 회귀 문제의 벤치마크 데이터셋에서 상당한 향상을 보여주며, 기존의 CP 접근법과의 성능 비교를 통해 그 유용성이 입증되었습니다. 이 연구는 OTCP 방법이 통계적 추정 및 계산의 균형에서 발생하는 무역-off(trade-offs)를 해결하는 데 어떤 기여를 하는지에 대한 논의를 포함하고 있습니다. 또한 유사한 접근법을 제안한 Thurin et al.(2025)와의 차별점을 강조합니다.



### Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models (https://arxiv.org/abs/2502.03607)
- **What's New**: 최근 확산 모델(diffusion models)의 발전은 로봇 공학에서 다양한 경로를 생성하는 데 큰 잠재력을 보이고 있습니다. 그럼에도 불구하고, 이러한 모델을 다중 로봇 이동 계획(Multi-Robot Motion Planning, MRMP)에 적용하는 것은 충돌 회피 및 운동학적 적합성 등의 중요한 제약을 적용하기 어렵기 때문에 도전적입니다. 본 논문에서는 새로운 접근 방식인 Simultaneous MRMP Diffusion(SMD)를 제안하여 제약 최적화를 확산 샘플링 프로세스에 통합하여 충돌이 없는 경로를 생성합니다.

- **Technical Details**: SMD는 제약이 있는 확산 프로세스 내에서 다중 로봇 경로 생성을 공식화하여 충돌이 없는 운동 계획을 보장합니다. 이 접근 방식은 강화된 Lagrangian 방법을 통해 확산 기반 경로 생성이 충돌 회피 및 운동학적 제약을 만족하도록 만듭니다. 또한, MRMP 평가를 위한 첫 번째 벤치마크를 소개하며, 다양한 로봇 밀도와 장애물 복잡성을 가진 여러 시나리오에서 경로 계획 알고리즘을 평가할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, SMD는 기존의 전통적인 및 학습 기반 이동 계획 방법보다 일관되게 우수한 성능을 보여주며, 복잡한 다중 로봇 환경에서 더 높은 성공률과 효율성을 달성했습니다. SMD는 특히 밀도가 높은 장애물과 비구조적 환경에서도 안정적인 성능을 유지하여, MRMP의 잠재력을 크게 확장합니다.



### Clinically-Inspired Hierarchical Multi-Label Classification of Chest X-rays with a Penalty-Based Loss Function (https://arxiv.org/abs/2502.03591)
Comments:
          9 pages with 3 figures, for associated implementation see this https URL

- **What's New**: 이번 연구에서는 임상 해석 가능성을 향상시키면서도 단일 모델, 단일 실행 학습 파이프라인을 유지하는 다중 레이블 흉부 X선(X-ray, CXR) 이미지 분류에 대한 새로운 접근 방식을 제안합니다. CheXpert 데이터셋과 VisualCheXbert에서 파생된 레이블을 활용하여 진단 간의 임상적으로 유의미한 관계를 포착하기 위해 계층적 레이블 그룹화를 통합하였습니다. 이를 통해 고안한 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 통해 레이블 의존성을 강화하였습니다.

- **Technical Details**: 연구진은 어린 레이블이 상위 레이블에 대한 긍정적인 예측이 없이 긍정적으로 예측될 때 패널티를 적용하는 계층적 이진 교차 엔트로피(HBCE) 손실 함수를 제안하였습니다. 패널티 전략으로는 고정 패널티 접근 방식과 데이터 기반 패널티 방법을 탐색하였으며, 데이터 기반 방법은 상위-하위 레이블 간의 의존 가능성에 따라 패널티를 조절합니다. 모델 성능 향상을 목표로, 학습 데이터 установ 일부 조정과 함께 계층적 그룹 재구성의 효과를 종합적으로 분석하였습니다.

- **Performance Highlights**: 제안된 프레임워크는 CheXpert 데이터셋에서 0.9034의 분류 성능을 달성하였으며, 이는 계층적 구조와 커스텀 HBCE 손실 함수의 효능을 입증합니다. 데이터 기반 패널티는 예측 정확도를 향상시킬 가능성을 보였으며, 시각적 설명과 불확실성 추정이 모델 해석 가능성과 투명성을 더욱 높였습니다. 모든 코드, 모델 구성 및 실험 세부 사항은 공공 Git 저장소에 공개되어 연구의 투명성과 재현성을 촉진하고 있습니다.



### HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inferenc (https://arxiv.org/abs/2502.03589)
- **What's New**: 본 연구에서는 분리된 대규모 언어 모델(LLM)의 추론을 위한 새로운 접근 방식인 HACK(Homomorphic Acceleration via Compression of the KV cache)를 제안합니다. HACK은 KV 데이터의 비정형화를 제거하고, 양자화된 KV 데이터에서 직접 계산을 수행함으로써 계산 비용을 줄입니다. 이 방식은 기존 KV 양자화 방법들의 성능 저하 문제를 해결하기 위한 것입니다.

- **Technical Details**: HACK는 양자화된 행렬에서 KV 관련 행렬 곱셈을 수행함으로써 KV 전송 지연 및 메모리 요구 사항을 줄입니다. 이 방법은 KV 데이터로 인한 계산 오버헤드를 줄일 수 있으며, 효율적인 메모리 사용을 보장합니다. HACK는 FlashAttention-2와 vLLM에 통합되어 실험을 진행하였고, 기록된 성능 향상을 보여줍니다.

- **Performance Highlights**: HACK는 분리된 LLM 추론의 Job Completion Time(JCT)를 최대 70.9%까지 줄였으며, 최첨단 KV 양자화 방법에 비해서도 최대 52.3% 향상된 성능을 보입니다. 실험 결과는 다양한 모델, 데이터셋, GPU 구성에서 HACK의 효과를 입증하며, HACK의 코드가 오픈 소스 형태로 공개되었습니다.



### CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally (https://arxiv.org/abs/2502.03566)
- **What's New**: 이 논문에서는 CLIP의 compositionality(구성 가능성) 문제를 다룹니다. 이전 연구들이 CLIP이 bag-of-words(BoW) 모델처럼 작동한다고 주장했으며, 그 결과로 객체와 속성을 올바르게 연결하지 못하는 경향이 있음을 보여줍니다. 이 연구에서 제안된 LABCLIP은 속성-객체 바인딩 정보의 문제를 해결하기 위해 선형 변환을 적용하여 CLIP의 성능을 개선합니다.

- **Technical Details**: LABCLIP은 원본 텍스트 임베딩에 선형 변환을 적용하여 코사인 유사도(cosine similarity)를 계산하기 전에 속성 정보를 추출합니다. 이를 통해 복잡한 이미지와 텍스트 쌍에서 속성과 객체를 정확하게 연결합니다. 연구진은 CLIP의 인코더를 동결한 상태에서 합성 부정 샘플을 사용하여 변환을 훈련했으며, 다양한 벤치마크에서 개선된 성능을 입증했습니다.

- **Performance Highlights**: 이 연구에서는 ARO, SugarCrepe, COCO와 같은 여러 벤치마크에서 LABCLIP의 성능을 검증했습니다. CLIP이 객체에 속성을 정확하게 바인딩할 수 있는 능력이 크게 향상되었으며, 이를 통해 Compositional understanding(구성 이해력)의 발전을 이끌었습니다. 이 결과는 CLIP 기반 VLMs(비전-언어 모델)의 발전에 중요한 기여를 할 것으로 기대됩니다.



### Online Learning Algorithms in Hilbert Spaces with $\beta-$ and $\phi-$Mixing Sequences (https://arxiv.org/abs/2502.03551)
- **What's New**: 이 논문은 혼합 프로세스를 기반으로 하는 복원 커널 힐버트 공간(RKHS) 내의 온라인 알고리즘을 다룹니다. 기존의 독립적이고 동일하게 분포된(i.i.d.) 샘플 대신, 특정한 의존 구조를 가진 샘플에서 학습하는 방법을 제안합니다. 이를 통해 마르코프 체인을 활용하여 의존성이 있는 샘플에서도 최적 수렴 속도를 도출하고, i.i.d. 케이스의 특수 예로서의 의미를 확장합니다.

- **Technical Details**: 저자들은 확률적 경량화(stochastic gradient) 알고리즘을 마르코프 체인에 적응시켜 새로운 학습 프레임워크를 발전시킵니다. 이로 인해 마르코프 체인의 혼합 시간(mixing time)은 관련성이 없어지며, 의존 구조로 인한 추가적인 요인을 고려한 에러 추정 상한을 설정합니다. 이 또한 베타-혼합(b-mixing) 및 파이-혼합(ϕ-mixing) 계수를 사용하여 의존성을 정량화합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 RKHS의 온라인 학습 알고리즘에 대한 초기 에러 상한 및 샘플 에러의 확률적 상한을 제시함으로써, 최적 수렴 속도에 가까운 결과를 도출합니다. 이 논문에서 제안하는 방법은 r이 0에 가까워질수록 의존성이 약해지고, 에러 상한이 i.i.d. 사례의 경계에 수렴함을 보여줍니다. 전체적으로 의존에 기반한 새로운 이론을 통해 비-i.i.d. 샘플의 학습 요구를 충족시키는 기초를 마련합니다.



### Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 (https://arxiv.org/abs/2502.03544)
Comments:
          28 pages, 16 figures

- **What's New**: AlphaGeometry2(AG2)는 2024년 Trinh 외 연구진에 의해 소개된 AlphaGeometry(AG1)의 크게 향상된 버전입니다. AG2는 자동 기하학 문제 해결에서 평균 금메달리스트를 초월하는 성능을 발휘하였고, International Math Olympiad(IMO) 2000-2024 기하학 문제에 대한 해결률을 66%에서 88%로 끌어올렸습니다. 새로운 Gemini 아키텍처를 활용하여 언어 모델링을 향상시키고, 여러 검색 트리를 결합한 지식 공유 메커니즘을 도입했습니다.

- **Technical Details**: 이 논문에서 AG2는 더 강력하고 다양한 데이터셋으로 훈련된 Gemini 기반의 언어 모델을 활용합니다. 빠르고 강력한 기호 엔진과 축소된 규칙 집합을 활용한 최적화 등 여러 방식으로 성능을 개선하였으며, locus 정리와 같은 기하학적 개념과 선형 방정식 등 폭넓은 영역과의 통합도 이루어졌습니다. 혁신적인 검색 알고리즘을 통해 보조 구성 전략을 탐색하고, 검색 프로세스를 가속화하기 위해 지식 공유 메커니즘을 활용하고 있습니다.

- **Performance Highlights**: AG2는 2000-2024 IMO 기하학 문제에 대해 84%라는 인상적인 해결률을 달성하여, AI의 수학적 추론 능력이 크게 발전했음을 시사합니다. AG1에 비해 해결률이 54%에서 84%로 증가하며, 또래 금메달리스트들의 성과를 넘었습니다. 마지막으로 AG2는 자연어 입력으로부터 기하학 문제를 직접 해결하는 완전 자동 시스템 개발을 향해 나아가고 있습니다.



### Optimistic {\epsilon}-Greedy Exploration for Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.03506)
- **What's New**: 이 논문에서는 협력적 다중 에이전트 강화 학습의 중앙 집중 훈련과 분산 실행(CTDE) 패러다임에서, 기존의 단조 값 분해 방법의 표현 한계를 극복하기 위해 낙관적 $	extepsilon$-탐색 방법을 제안하고 있습니다. 이 접근 방식은 탐색 중 최적 행동의 샘플링을 증가시켜 가치 추정의 오류를 바로잡는 데 초점을 맞추고 있습니다. 실험 결과는 제안한 방법이 알고리즘의 성능을 크게 향상시키고, 비최적해를 피하는 데 효과적임을 보여줍니다.

- **Technical Details**: CTDE는 훈련 중 모든 에이전트의 공동 행동을 기반으로 한 전역 가치 추정을 가능하게 하여, 독립적인 결정을 내릴 수 있도록 지원합니다. 그러나 전통적인 값 분해 방법은 과도한 단조성 제약 때문에 가치 추정의 정확성에 제한을 받습니다. 궤적은 최적 행동을 포함하더라도 비최적 행동의 영향을 받아 최적 행동의 가치를 과소 추정할 수 있습니다. 이 논문은 낙관적 추정 네트워크를 소개하여 각 에이전트가 최적 행동을 식별하도록 합니다.

- **Performance Highlights**: 제안된 낙관적 $	extepsilon$-탐색 전략은 QMIX 프레임워크 내에서 통합되어 다양한 환경에서 테스트되었습니다. 실험 결과는 제안된 방법이 기존 알고리즘보다 더 나은 성능을 보여주었으며, 가치 추정의 과소 평가 문제를 효과적으로 해결하는 데 기여함을 나타냅니다. 이 연구는 다중 에이전트 환경에서 최적 행동 샘플링의 중요성을 강조하며, 이를 통해 정책 훈련을 개선할 수 있음을 입증하고 있습니다.



### Enhancing Free-hand 3D Photoacoustic and Ultrasound Reconstruction using Deep Learning (https://arxiv.org/abs/2502.03505)
- **What's New**: 이 연구에서는 MoGLo-Net이라는 모션 기반 학습 네트워크를 소개하여 핸드헬드 포토아큐스틱(photoacoustic, PA) 및 초음파(ultrasound, US) 이미징의 3D 재구성을 향상시킵니다. MoGLo-Net은 자기 주의(self-attention) 메커니즘을 혁신적으로 적용하여 연속적인 초음파 이미지 내의 중요한 영역을 효과적으로 이용하여 정밀한 모션 추정을 가능케 합니다. 또한, 새로운 손실 함수(custom loss function)를 개발하여 모션 매개변수에 대한 학습의 신뢰성을 높입니다.

- **Technical Details**: MoGLo-Net은 2D 이미지를 3D 구조로 재구성하기 위해 ResNet 기반 인코더와 장기 단기 메모리(Long Short-Term Memory, LSTM) 블록을 포함합니다. 이 구조는 인접한 프레임 간의 인코딩된 피처 맵의 상관관계를 직접적으로 접근할 수 있는 특별한 블록으로 이루어져 있으며, 주요 지역을 강조하는 글로벌-로컬 주의 모듈을 갖추고 있습니다. 이 기술을 통해 복잡한 3D 구조를 더욱 정확히 시각화할 수 있습니다.

- **Performance Highlights**: 실험 결과, MoGLo-Net은 정량적 및 정성적 성능 지표 모두에서 현재의 최첨단 방법을 초능가했습니다. 3D 재구성 기술은 범위 제한을 넘어 도플러 초음파 및 포토아큐스틱 이미징을 포함하도록 확장되어 혈관 구조의 3D 시각화를 가능하게 하였습니다. 이 연구의 소스 코드는 공개되어 있으며, 이를 통해 다양한 환경에서의 모델 성능을 평가할 수 있습니다.



### Two in context learning tasks with complex functions (https://arxiv.org/abs/2502.03503)
- **What's New**: 본 논문에서는 작은 transformer 모델이 임의의 다항 함수와 연속 함수들을 근사할 수 있음을 증명합니다. 특히, attention layers만으로 구성된 모델들조차도 이러한 함수들을 정확하게 모델링할 수 있는 능력을 보여줍니다. 기존의 작업과 달리 본 연구는 미리 보지 못한 다항 함수 클래스와 복소 함수의 영점들까지 근사할 수 있다는 점에서 중요합니다.

- **Technical Details**: 본 연구는 ICL(In-Context Learning)이 정의된 시간에 모델이 특정 작업을 학습하는 방식을 다루고 있습니다. 특히, [a,b]의 구간에서 임의의 연속 함수를 ICL1 및 ICL2를 통해 학습할 수 있는지를 평가하였으며, 다양한 학습 설정에서 30개 이상의 모델을 훈련시켰습니다. 논문에서는 attention을 정의하는 매트릭스의 한계로 인해 다항 함수의 클래스 형태를 학습하는 데 어려움이 있음을 수학적으로 설명하고 있습니다.

- **Performance Highlights**: 작은 transformer 모델들은 GPT4와 같은 기존의 대형 언어 모델들(Large Language Models)보다 훨씬 우수한 성능을 보였습니다. 이는 적절한 훈련 데이터와 방법이 제공될 경우 복잡한 추론 능력을 발휘할 수 있음을 시사합니다. 그러나 모델들은 훈련 분포 이외의 값에 대한 일반화 능력이 부족하다는 중요한 제한점도 겪고 있습니다.



### Proxy Prompt: Endowing SAM and SAM 2 with Auto-Interactive-Prompt for Medical Segmentation (https://arxiv.org/abs/2502.03501)
- **What's New**: 이 논문에서는 SAM과 SAM2의 임상 적용을 촉진하기 위해 자동 프롬프트 생성 및 인간-모델 상호작용 강화를 목표로 합니다. Proxy Prompt(프록시 프롬프트, PP)를 제안하며, 이는 사전 주석이 달린 마스크를 활용하여 비대상 데이터로부터 자동 생성됩니다. 이 방법은 프롬프트 생성을 통해 다양한 임상 요구를 충족시키는 것을 목표로 합니다.

- **Technical Details**: PP는 비대상 이미지/비디오의 사전 주석 마스크를 활용하여 자동으로 생성되며, 세 가지 단계의 맥락 선택 전략을 통해 비대상 데이터에서 가장 대표적인 정보를 선택합니다. Contextual Selective Module(CSM)은 이 과정에서 이뤄지며, 다양한 이미지 및 비디오 데이터를 유기적으로 연결할 수 있습니다. 또한, Contextual Colorization Module(CCM)을 통해 사용자가 정의한 객체의 특징을 강조해 인간-모델 상호작용을 강화합니다.

- **Performance Highlights**: 광범위한 평가를 통해 이 방법은 네 개의 공개 데이터셋에서 최첨단 성능을 달성했습니다. 기존의 전량 훈련된 모델에 비해 유사한 결과를 달성하였으며, 단 16개의 이미지 마스크로도 훈련이 가능함을 보여줍니다. 이는 향후 의료 작업에 적합한 대안으로 작용할 가능성이 높습니다.



### Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning (https://arxiv.org/abs/2502.03499)
- **What's New**: 이 논문에서는 기존의 Genomic Foundation Models (GFMs)의 한계를 극복하기 위해 Omni-DNA라는 새로운 다중 작업 교차 모델을 제안합니다. 일반적인 모델과는 달리 Omni-DNA는 생물학적 패턴을 공유하고 여러 하위 작업을 동시에 해결할 수 있는 능력을 가지고 있습니다. 이를 통해 GFMs이 모델 사이에 공유할 수 있는 정보의 활용을 극대화할 수 있습니다.

- **Technical Details**: Omni-DNA는 20 million에서 1 billion까지의 파라미터를 갖는 다양한 크기의 모델로서, DNA 시퀀스에 대한 사전 학습(pretraining)과 다중 작업(finetuning)을 통해 훈련됩니다. 훈련 과정에서 기존의 모델 아키텍처와 토크나이징 전략의 수정과 비교를 통해 최적의 구성요소를 찾고, 이를 통해 유연한 출력 공간을 제공합니다. 이 모델은 DNA2Text 및 DNA2Image와 같은 다양한 하위 작업에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: Omni-DNA는 Nucleotide Transformer 및 GB benchmarks에서 26개의 작업 중 18개에서 최고 성능(SOTA)을 달성했습니다. 또한, 다중 작업을 통해 10개의 아세틸화 및 메틸화 작업을 동시에 처리할 수 있으며, 각각의 작업에 대해 개별적으로 훈련된 모델을 초월하는 성과를 보여줍니다. 최종적으로 DNA 시퀀스를 텍스트 기능 설명이나 이미지로 변환하는 두 가지 복잡한 유전체 작업도 설계하여 Omni-DNA의 교차-모달(지식 간 전환) 기능을 강조합니다.



### Examining Two Hop Reasoning Through Information Content Scaling (https://arxiv.org/abs/2502.03490)
- **What's New**: 이 연구는 트랜스포머(transformer)가 두 단계 질문(latent two-hop question)에 대한 답변을 학습하는 능력의 일관성 부족을 탐구하고 있습니다. 전작들에서는 두 단계 QA 데이터셋에 대한 트랜스포머의 크기와 학습 능력이 어떻게 변화하는지를 분석하여, 두 단계 QA에 필요한 정보의 중복성을 강조하고 있습니다. 이를 통해, 적절한 데이터셋 매개변수에서 트랜스포머가 여러 레이어에서 사실 정보를 중복 저장해야 하는 점을 지적하고 있습니다.

- **Technical Details**: 연구는 'native information content'라는 개념을 도입하여, 트랜스포머의 아키텍처에 맞춰 압축 가능성을 연구합니다. 이를 통해 두 단계 질문 응답에서의 정보 용량 규모가 모델 크기와 함께 어떻게 변화하는지를 탐구하며, 기존의 압축 이론과 연관짓고 있습니다. 또한 서로 다른 질문 유형을 처리하는 데 필요한 정보 용량의 증가율에 대한 예측도 제공합니다.

- **Performance Highlights**: 연구 결과, 두 단계 질문 처리에서 트랜스포머의 기능에 있어 효과적인 정보 압축이 필수적이라는 것을 보여줍니다. 일반적인 기능 조합보다 기억(Memorization)을 통해 답변을 더 잘 수행하도록 모델을 트랩(trap)하는 방법을 제안합니다. 이러한 발견은 트랜스포머의 일반화 성능(generalization performance) 향상에 기여할 수 있음을 강조합니다.



### Dementia Classification Using Acoustic Speech and Feature Selection (https://arxiv.org/abs/2502.03484)
- **What's New**: 이 연구는 ADReSS 챌린지 데이터를 활용하여 알츠하이머 환자와 건강한 대조군을 분류하는 기계 학습 모델을 개발했습니다. 기존 연구와 달리 오디오 녹음을 활성 음성 세그먼트로 나누지 않고 전체 녹음에서 음향적 특징을 추출하여 분석했습니다. 이를 통해 조기 진단의 효율성을 높이는 방안을 제시하며, 신속하고 비용 효율적인 진단 가능성을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 Ridge Linear Regression, Extreme Minimal Learning Machine, 그리고 Linear Support Vector Machine 모델을 사용하여 특성 중요도 점수를 계산하였습니다. Ridge 모델은 Leave-One-Subject-Out 교차검증에서 87.8%의 분류 정확도를 달성하였으며, EMLM 모델은 85.3%와 79.2%의 정확도로 외부 데이터셋에서도 효과적이었습니다. 연구팀은 전체 녹음에서 추출한 음향적 특징을 기반으로 기계 학습 모델을 훈련시켜 빠르고 정확한 진단을 구현했습니다.

- **Performance Highlights**: 연구 결과, 이 방법을 통해 분류 정확도가 70%를 초과하는 성과를 거두었습니다. 특히, LOSO 교차검증에서 80% 이상의 높은 정확도를 기록하며 기존 연구들과 비교했을 때 최고의 성과 중 하나로 평가되고 있습니다. 전반적으로, 본 연구는 동일한 데이터셋과 음향적 특징 추출 방식을 사용한 다른 연구들과 비교하여 우수한 결과를 보여주었습니다.



### Can Domain Experts Rely on AI Appropriately? A Case Study on AI-Assisted Prostate Cancer MRI Diagnosis (https://arxiv.org/abs/2502.03482)
- **What's New**: 이번 연구는 방사선 전문의와의 심도 있는 협력을 통해 MRI 이미지를 기반으로 한 전립선암 진단에서 AI 지원 도구의 실제 통합 효과를 조사합니다. 두 가지 실험을 진행하여 AI 지원과 성과 피드백이 도메인 전문가의 의사결정에 미치는 영향을 분석하였습니다. 특히, 두 개의 상이한 작업 흐름을 설계하여 실제 임상 환경에서 AI 툴이 어떻게 사용될 수 있는지를 모델링하였습니다.

- **Technical Details**: 연구는 8명의 방사선 전문의(N=8)를 포함한 사전 등록된 인간 실험을 수행하였으며, 주요 초점은 전립선암 진단을 위한 AI 지원입니다. 최신 AI 모델(Isensee et al., 2021)을 훈련시켜 전립선암 탐지에 필요한 진단 예측과 양성 사례에 대한 병변 주석 맵을 제공하였습니다. 실험은 두 가지 별도의 작업 흐름으로 구성되었고, 첫 번째 연구에서는 독립적인 진단 후 AI 예측을 확인하는 방식으로 진행되었습니다.

- **Performance Highlights**: 연구 결과, 인간-AI 팀은 인간 단독보다 지속적으로 더 높은 성능을 보였으나, AI 단독 성능에는 미치지 못하는 경향이 있었습니다. 성과 피드백을 제공했음에도 불구하고, 인간-AI 팀의 성능 향상은 제한적이었으며, AI 결정을 사전에 보여주는 방식이 방사선 전문의로 하여금 AI를 더욱 신뢰하도록 유도하는 것으로 나타났습니다. 흥미로운 점은 인간-AI 팀의 다수결 결정이 AI 단독보다 뛰어난Complementary performance를 달성했다는 점으로, 이는 인간-AI 협업의 유망한 방향성을 제시합니다.



### Foundation for unbiased cross-validation of spatio-temporal models for species distribution modeling (https://arxiv.org/abs/2502.03480)
- **What's New**: 이 논문은 종 분포 모델링(Species Distribution Modeling, SDM)에서 공간 자율 상관(spatial autocorrelation, SAC)의 영향을 줄이기 위한 교차 검증(cross-validation, CV) 전략을 평가합니다. LAST FOLD와 RETRAIN 두 가지 훈련 방식을 비교하면서 각각의 강점과 한계를 밝히고, 새로운 공간-시간 교차 검증 방법을 소개합니다. 이 연구는 SDM 데이터의 공간적 및 시간적 구조에 맞춰 CV 접근 방식을 조정할 필요성을 강조하며, 예측 모델의 신뢰성을 높이기 위한 기초를 제공합니다.

- **Technical Details**: 연구 방법에서는 SDM을 구축하기 위해 데이터 수집, 전처리, 모델 선택, 공간 교차 검증, 하이퍼파라미터 튜닝 및 성능 평가의 체계적인 파이프라인을 따릅니다. 1994년부터 2018년까지의 환경 데이터와 종 데이터를 수집하고, 공간 및 환경 차단 방법과 새로운 공간-시간 차단 방법을 포함한 다양한 교차 검증 전략으로 모델을 훈련하고 검증하였습니다. 모델 성능은 Receiver Operating Characteristic Area Under the Curve (ROC AUC)와 평균 절대 오차(Mean Absolute Error, MAE)를 통해 평가되었습니다.

- **Performance Highlights**: LAST FOLD 방식이 일관되게 낮은 오류율과 강한 상관 관계를 나타냈으며, 공간 차단에서 최적의 거리(SP 422)와 ENV가 가장 우수한 성과를 보였습니다. 스페이크 및 피어슨 상관 계수는 각각 0.485 및 0.548에 도달했지만, ENV는 주요 환경 변화가 포함된 장기 예측에는 부적합할 수 있습니다. 이 연구는 다양한 기계 학습 모델에 대한 하이퍼파라미터 최적화의 중요성을 강조하며, 목적에 맞는 SDM을 생성하는 데 기여할 수 있는 프레임워크를 제공합니다.



### Leveraging Reviewer Experience in Code Review Comment Generation (https://arxiv.org/abs/2409.10959)
- **What's New**: 이 연구에서는 기존 코드 리뷰 프로세스에서 발생하는 인간 리뷰어의 피로를 줄이기 위해 경험 기반의 훈련 방법을 제안합니다. 특히, 경험-주의 손실 함수(experience-aware loss functions, ELF)를 통해 리뷰어의 경험을 코드 리뷰 품질의 신호로 활용함으로써, 경험이 풍부한 리뷰어의 주석이 모델의 행동에 더 큰 영향을 미치도록 합니다. 이를 통해 자동화된 코드 리뷰 모델에 전통적인 소프트웨어 엔지니어링 개념을 통합하는 방법을 제시했습니다.

- **Technical Details**: 이 연구는 자동화된 코드 리뷰를 위해 세 가지 주요 업무인 코드 변경 품질 추정, 코드 리뷰 주석 생성, 그리고 코드 수정 작업을 다룹니다. 특히, 코드 리뷰 주석 생성(task)에 중점을 두고 있으며, 이는 자연어 처리와 딥러닝 언어 모델을 이용하여 코드 변경의 다양한 문제를 식별하는 됩니다. ELF는 리뷰어의 과거 경험을 모델의 손실 함수에 가중치로 반영하여, 경험이 많은 리뷰어가 생성하는 코드 리뷰의 질을 향상시키는 방법입니다.

- **Performance Highlights**: ELF는 기존의 SOTA(Sate-of-the-Art) 모델인 CodeReviewer보다 높은 품질의 리뷰를 생성하는 것으로 평가받았습니다. ELF는 정확도 면에서 BLEU-4 기준으로 +5% 개선, 유용성 측면에서는 +56% 향상을 보였으며, 기능적 결함을 탐지하는 데 있어 +129%의 증가를 나타냈습니다. 실험 결과, ELF는 코드 리뷰의 질을 높이고, 경험이 있는 리뷰어의 관점을 모델에 반영하여 더 효과적인 코드 리뷰 프로세스를 실현할 수 있음을 보여줍니다.



### Robust Reward Alignment via Hypothesis Space Batch Cutting (https://arxiv.org/abs/2502.02921)
Comments:
          17 pages, including appendix

- **What's New**: 이 논문에서는 기존의 보상 설계 방법을 개선하기 위한 강력하고 효율적인 보상 정렬 방법을 제안합니다. 새로운 방법은 'hypothesis space batched cutting'이라는 기하학적으로 해석 가능한 관점을 기반으로 하여, 배치된 인간 선호에 따라 보상 가설 공간을 iteratively 정제합니다. 이 과정에서 잘못된 인간 선호에 대한 저항력을 확보하여 잘못된 데이터에 대해 강력한 성능을 보장합니다.

- **Technical Details**: 보상-파라메트릭 마르코프 결정 과정 (MDP)을 기반으로 하여, 상태 공간(S)과 행동 공간(A) 내에서 보상 함수(r𝜽)를 파라미터화하고 있습니다. 제안된 방법은 인간의 선호에 따라 가설 공간에 비선형 컷을 적용시키며, 각 배치 내에서 쟁점이 되는 선호를 그룹하여 적절한 컷을 결정하는 투표 기능을 도입합니다. 이러한 과정은 인간 쿼리의 복잡성을 제한하며, 잘못된 선호의 영향을 관리하기 위한 보수적인 컷팅 방법을 포함합니다.

- **Performance Highlights**: 다양한 작업에서 모델 예측 제어 설정으로 평가한 결과, 제안된 방법이 오류 없는 환경에서는 최첨단 방법과 유사하거나 우수한 성능을 달성했습니다. 특히, 높은 비율의 잘못된 인간 선호를 처리할 때, 기존의 방법을 상당히 초월하는 성능을 보여주었습니다. 이는 제안된 framework가 실제 응용에서 매우 유용함을 입증합니다.




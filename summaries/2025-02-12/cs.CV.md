New uploads on arXiv(cs.CL)

### Auditing Prompt Caching in Language Model APIs (https://arxiv.org/abs/2502.07776)
Comments:
          20 pages, 7 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서의 프롬프트 캐싱이 사용자의 데이터 의존적인 타이밍 변동을 초래하며, 이를 통해 개인 정보 유출의 위험이 발생할 수 있음을 강조합니다. 특히, 캐싱된 프롬프트는 비캐싱된 프롬프트보다 더 빠르게 처리되어, 타임 스탬프 차이를 이용한 사이드 채널 공격이 가능하다는 점을 지적합니다. 저자들은 이론적 근거와 함께 API 제공자의 캐싱 정책에 대한 투명성이 중요하다고 주장합니다.

- **Technical Details**: 프롬프트 캐싱 기술은 요청 간에 주의 메모리의 키-값(KV) 캐시를 재사용하는 방식입니다. 이러한 방식을 통해 특정 프롬프트와 캐시된 값이 일치할 때, 응답시간(TTFT)이 단축되는 현상이 발생합니다. 연구팀은 통계적 가설 검정을 통해 프롬프트 캐싱이 있는지를 감지하고, 실제로 17개의 API 제공자 중 8곳에서 캐싱이 이루어지고 있는 것을 발견했습니다. 그중 7개 제공자에서는 전세계적인 캐시 공유가 확인되어, 사용자의 프롬프트의 안전성이 위협받을 수 있음을 보여주었습니다.

- **Performance Highlights**: 이 연구의 주요 성과는 실제 API 제공자에 대한 적절한 감사를 통해 프롬프트 캐싱을 성공적으로 식별했다는 점입니다. OpenAI의 텍스트 임베딩 모델에서 디코더 전용 아키텍처의 징후를 발견했으며, 이는 과거에 공개된 적이 없는 정보로, 지적 재산권의 유출 가능성을 야기할 수 있습니다. 감사 결과는 각 제공자에게 통보되었고, 적어도 5개의 제공자가 취약점을 완화하기 위한 조치를 취한 사실 또한 중요한 성과로 간주됩니다.



### Breaking Down Bias: On The Limits of Generalizable Pruning Strategies (https://arxiv.org/abs/2502.07771)
Comments:
          28 pages, 9 figures, 1 table

- **What's New**: 이 논문에서는 모델 프루닝(model pruning)을 활용하여 LLM(대규모 언어 모델)이 인종적 편견을 어떻게 인식하고 있는지를 분석합니다. 연구 결과, 프루닝이 편견을 감소시키는 효과적인 방법일 수 있지만, 간극이 커질수록 그 효과가 감소한다는 점이 밝혀졌습니다. 또한 인종적 편견이 일반 개념으로 충분히 대표되지 않으며 많은 경우 특정 맥락에 따라 다르다는 것을 제안하여, 더 일반적인 완화 전략이 효과적이지 않을 수 있음을 주장합니다.

- **Technical Details**: 연구에서는 Llama-3-8B-Instruct 모델을 사용하여 인종적 편견을 분석하였으며, 주요 결과로는 뉴런 기반의 프루닝이 전체 어텐션 헤드를 프루닝하는 방법보다 더 나은 성능을 보인다는 사실이 발견되었습니다. 또한 프루닝 전후의 평가 세트 간의 유사성이 높을 경우 평균적으로 편견이 거의 0에 가깝게 줄어들지만, 세트 간의 맥락 차이가 커질수록 그 성능이 급격히 저하된다고 보고합니다. 이 연구는 결국 인종적 편견은 특정 도메인에서만 나타나는 개념이라는 점을 강조합니다.

- **Performance Highlights**: 편견을 줄이기 위한 모델 프루닝의 효과는 교육 세트와 평가 세트의 유사성에 크게 의존하며, 이는 편견 수정 방법의 한계에 대한 중요한 통찰력을 제공합니다. 특히, 금융 의사결정과 상업적 거래의 편견을 제거하려는 경우 40%의 편견 감소에 불과한 성과를 보여주었습니다. 이름 연관을 통한 편견의 평가와 관련된 통계 데이터는 이 모델이 인종적 요소에 따라 차별적인 응답을 생성하는 경향이 있음을 재확인시키며, 일반화된 완화 전략보다는 특정 맥락에 맞춘 조정이 필요하다고 결론짓습니다.



### An Advanced NLP Framework for Automated Medical Diagnosis with DeBERTa and Dynamic Contextual Positional Gating (https://arxiv.org/abs/2502.07755)
- **What's New**: 이 논문은 의료 진단을 향상시키기 위한 새로운 자연어 처리(NLP) 프레임워크를 제안합니다. 고급 데이터 증강(data augmentation), 특징 추출(feature extraction), 분류(classification) 기법을 통합하여 보다 다양한 데이터 세트를 생성하도록 합니다. 이 접근 방법의 주요 특징은 역 번역(back-translation)을 사용하여 다양한 패러프레이즈(paraphrased) 데이터셋을 만드는 것입니다.

- **Technical Details**: 제안된 모델은 Disentangled Attention이 있는 Decoding-enhanced BERT(DeBERTa)를 활용하고, 동적 맥락 위치 게이팅(Dynamic Contextual Positional Gating, DCPG)을 통해 위치 정보의 영향을 맥락에 따라 조정합니다. 이는 높은 품질의 텍스트 임베딩(text embeddings)을 생성하는 데 도움을 줍니다. 또한, Attention 기반 피드포워드 신경망(ABFNN)을 사용하여 가장 관련 있는 특징에 집중하여 의사결정의 정확도를 향상시킵니다.

- **Performance Highlights**: 이 아키텍처는 증상, 임상 노트 및 기타 의료 텍스트의 분류에 적용되어 의료 데이터의 복잡성을 다루는 능력을 입증합니다. 제안된 NLP 프레임워크는 99.78%의 정확도, 99.72%의 재현율(recall), 99.79%의 정밀도(precision), 99.75의 F1 점수를 기록하여 뛰어난 성과를 보여줍니다. 이러한 성능 지표는 의료 텍스트 분류에서 예외적인 정확성과 신뢰성을 제공하며, 기존 방법보다 우수함을 강조합니다.



### WHODUNIT: Evaluation benchmark for culprit detection in mystery stories (https://arxiv.org/abs/2502.07747)
- **What's New**: WhoDunIt라는 새로운 데이터세트를 소개하여, 대형 언어 모델(LLM)의 귀납적 추론 능력을 내러티브(narrative) 맥락에서 평가합니다. 이 데이터세트는 공개 도메인에서 가져온 추리 소설과 단편 소설로 구성되어 있으며, LLM이 이야기를 이해한 후 범인을 식별하도록 도전합니다. 또한, 캐릭터 이름의 다양한 변형을 통해 모델의 견고성을 평가하고, 추론 정확성에 미치는 프롬프트(prompt)의 영향을 조사합니다.

- **Technical Details**: 이 데이터세트는 공공 도메인에서 저작권이 만료된 책을 활용하여 준비되었으며, 주로 Project Gutenberg에서 이야기 자료를 확보했습니다. 500편 이상의 미스터리 및 탐정 이야기를 선정하였으며, 각 소설은 식별 가능한 범인이 존재하도록 설계되어 있습니다. 각 프롬프트 스타일과 캐릭터 이름의 대체를 통해, 모델이 단순한 이름 인식이 아닌 내러티브 기능에 기반해 추론할 수 있는 능력을 테스트합니다.

- **Performance Highlights**: 실험 결과, LLM은 변형되지 않은 텍스트에 대한 신뢰할 수 있는 수행 보여주지만, 특정 이름 대체가 있을 경우 정확성이 떨어지는 경향을 보였습니다. 특히, 잘 알려진 실존 또는 허구의 캐릭터 이름으로 대체했을 때 모델의 성능이 변동했습니다. 이 데이터세트는 공개적으로 사용 가능하여, 향후 LLM의 추론 능력과 복잡한 내러티브 이해 능력을 개선하는 데 기여할 것입니다.



### Making Language Models Robust Against Negation (https://arxiv.org/abs/2502.07717)
Comments:
          Accepted to NAACL 2025

- **What's New**: 본 연구는 언어 모델이 부정 표현에 더 강건하게 작용하도록 하기 위한 자가 감독(self-supervised) 방법을 제안합니다. 새로운 과제인 Next Sentence Polarity Prediction (NSPP)와 Next Sentence Prediction (NSP) 과제의 변형을 도입하였습니다. 기존의 BERT와 RoBERTa 모델을 우리의 과제로 추가 학습시킨 결과, 여러 부정 관련 벤치마크에서 향상된 성능을 보였습니다.

- **Technical Details**: 부정은 의미를 반전시키는 언어적 현상으로, 25%의 영어 문장에 포함되어 있습니다. 연구에서는 BERT와 RoBERTa 같은 언어 모델이 부정적 맥락 속에서 어려움을 겪는다는 기존 연구를 기반으로, 두 가지 새로운 학습 과제를 도입합니다. 첫 번째는 다음 문장의 부정성 예측(NSPP) 과제이며, 두 번째는 부정 변형을 통해 음성 샘플을 생성하는 NSP 변형 과제입니다.

- **Performance Highlights**: 우리의 추가 학습 작업을 통해 BERT와 RoBERTa 모델의 성능이 CondaQA와 같은 대규모 질문 응답 코퍼스에서 1.8%에서 9.1%까지 개선된 결과를 나타냈습니다. 이로 인해 부정에 대한 모델의 이해력이 향상되고, 더 나아가 자연어 이해 태스크 전반에서의 성능 향상이 기대됩니다.



### Large Language Models as Proxies for Theories of Human Linguistic Cognition (https://arxiv.org/abs/2502.07687)
- **What's New**: 이번 논문에서는 현재의 대형 언어 모델(LLMs)이 인간 언어 인지(HLC) 연구에서의 역할을 탐구하고 있습니다. 특히 LLMs를 인지 이론의 대리자로 사용하여 언어적 표현과 학습에서 상대적으로 언어 중립적이지 않은 이론을 평가하는 데 도움을 줄 수 있음을 주장합니다. 이는 HLC 이론에 대한 새로운 시각을 제공하며, LLMs의 제한된 도움에 대해서도 강조합니다.

- **Technical Details**: HLC 연구는 최선의 설명에 이르는 추론을 통해 진행되며, 명확한 이론들이 관찰에 비추어 평가됩니다. LLM 이론(LLM Theory)과 대리 관점(Proxy View)을 통해, LLMs는 더 전통적인 생성 언어학 이론과 비교하는 데 유용한 도구가 될 수 있음을 시사합니다. 특히, 언어 학습에서 발생하는 패턴에 대한 다양한 접근을 통해 상대적으로 언어 중립적인 이론의 가능성을 평가합니다.

- **Performance Highlights**: 논문에서는 LLMs가 상대적으로 언어 중립적인 HLC 이론의 대리자로서의 역할을 수행할 수 있는 가능성을 탐색합니다. 그러나 현재 LLMs는 이러한 이론을 지지하는 데에는 충분하지 않다는 점이 강조됩니다. 마지막으로 LLMs의 성능이 생성 언어학 이론을 도전하는 데 기여하지 않는다는 결론에 도달합니다.



### Auto-Drafting Police Reports from Noisy ASR Outputs: A Trust-Centered LLM Approach (https://arxiv.org/abs/2502.07677)
- **What's New**: 이번 연구에서는 복잡하고 잡음이 많은 대화 데이터를 기반으로 경찰 보고서 초안을 생성하기 위한 혁신적인 AI 기반 시스템을 제안합니다. 이 시스템은 법 집행 상호작용의 핵심 요소를 추출하여 구조적 내러티브를 생성하며, 이러한 접근 방식은 보고서의 책임성 및 절차적 명확성을 강화합니다. 앞으로 이 시스템은 보고서 작성 과정을 개선하여 경찰의 투명성과 공정성을 높일 잠재력을 갖고 있습니다.

- **Technical Details**: 우리가 개발한 시스템은 경찰관의 행동에 착용한 ASR(automatic speech recognition) 기기를 통해 고품질의 경찰 보고서 초안을 자동으로 생성하는 기능을 갖추고 있습니다. 본 시스템은 최신 LLM(large language model)와 고급 지식 확산 모델을 통합하여 복잡한 다중 역할 대화 데이터를 효과적으로 처리하며, 대화 잡음을 제거하고 중요한 이벤트 요소를 추출하는 데 필요한 intermediate tasks를 학습합니다. 또한, 기존의 대규모 데이터셋을 요구하는 대신, 공개된 사례 법률 데이터베이스를 활용하여 상대적으로 적은 양의 트레이닝 데이터로도 개인정보를 보호할 수 있습니다.

- **Performance Highlights**: 시스템은 경찰관의 리뷰 후, 중요한 섹션을 강조하여 제공함으로써 최종 보고서의 정확성과 신뢰성을 향상시키는 데 기여합니다. 이 시스템은 보고서 작성 효율성을 크게 향상시키며, 경찰 보고서 작성 과정의 표준화와 자동화를 통해 경찰의 업무 부담을 줄이는 효과를 가져옵니다. 현재 이 시스템은 여러 경찰서에서 시험 운영되고 있으며, 초기 피드백 또한 긍정적으로 나타나고 있습니다.



### FoQA: A Faroese Question-Answering Datas (https://arxiv.org/abs/2502.07642)
Comments:
          Camera-ready version for RESOURCEFUL workshop, 2025

- **What's New**: FoQA는 2,000개의 샘플로 구성된 파로어(Faroese) 추출형 질문-응답(QA) 데이터셋을 소개합니다. 이 데이터셋은 대규모 언어 모델(LLMs)과 인간 검증을 결합한 반자동 방식으로 생성되었습니다. FoQA는 파로어의 QA 성능을 평가하기 위한 기준 성능 메트릭을 제공하며, 총 10,001개의 샘플을 생성한 데이터셋과 에러 분석을 위한 2,395개의 반려 샘플도 함께 공개됩니다.

- **Technical Details**: 이 연구는 텍스트 말뭉치, 생성 모델 및 Q&A 생성과 질문 재형성을 위한 특화된 기능을 포함한 두 단계의 프로세스를 통해 데이터를 생성합니다. 처음에는 QA 생성 함수를 통해 질문과 답변 쌍을 만들고, 이어서 질문을 어렵게 바꾸는 과정을 거칩니다. 생성된 질문이 기존의 문장에서 단순히 재구성되는 경향을 보였기에, 보다 복잡한 이해 능력을 요구하는 질문으로 변형하여 QA의 품질을 높였습니다.

- **Performance Highlights**: FoQA 데이터셋은 BERT와 같은 여러 모델을 사용하여 평가되었으며, 파로어에서 효과적인 QA 성능을 달성하는 데 기여하고 있습니다. 또한, 이 데이터셋은 기존의 파로어 데이터셋의 부족함을 해소하며, 언어 기술 발전을 돕고 있습니다. FoQA는 QA 시스템 연구 분야에서 새로운 기준을 제시하며, 저자들은 이 데이터를 공개 소스 형태로 제공하여 더 많은 연구자들이 활용할 수 있도록 하고 있습니다.



### BiaSWE: An Expert Annotated Dataset for Misogyny Detection in Swedish (https://arxiv.org/abs/2502.07637)
Comments:
          To appear at NoDaLiDa 2025

- **What's New**: 이 논문에서는 스웨덴어에서의 여성혐오 탐지를 위한 전문가 주석 데이터셋인 BiaSWE의 생성 과정을 소개합니다. 문화적이고 언어적인 특수성을 고려하여 사회과학 및 인문학 전문가와 협력하여 데이터셋을 개발하였으며, 이는 저소득 자원의 언어에서의 편향 탐지 노력과 연계되어 있습니다. 이 데이터셋은 공개적으로 사용 가능하여 추가 연구에 기여할 수 있도록 하고 있습니다.

- **Technical Details**: BiaSWE 데이터셋은 스웨덴의 Flashback 포럼에서 수집된 450개의 데이터 포인트로 구성되어 있으며, 각 데이터 포인트는 두 명 이상의 주석자가 주석을 달았습니다. 연구팀은 스웨덴어와 덴마크어의 문화적, 언어적 유사성을 고려하여 키워드를 생성하였으며, 최종적으로 118개 키워드를 선정하여 데이터 수집에 활용하였습니다. 주석 작업에는 사회 과학 및 인문학 분야의 전문가들이 참여하였으며, 주석 가이드라인은 그녀들의 기여를 반영한 것입니다.

- **Performance Highlights**: 기존의 여성혐오 탐지 데이터셋들과의 차별점으로는, BiaSWE는 스웨덴어의 문화맥락에 맞춰 설계되어 있으며, 다차원적 주석 체계를 통해 여러 종류의 여성혐오를 분류할 수 있습니다. 이는 여성혐오가 단순한 이분법적 접근을 넘어서는 다양한 형태를 포괄할 수 있도록 지원합니다. 연구 결과, 이 데이터셋은 모델이 유해한 고정관념이나 태도를 지속할 위험을 식별하는 데 중요한 기여를 할 것으로 기대되고 있습니다.



### Lexical categories of stem-forming roots in Mapudüngun verb forms (https://arxiv.org/abs/2502.07623)
Comments:
          22 pages, 2 large tables, 2 sample tables

- **What's New**: 이번 연구는 마푸체 언어의 형태 분석을 위한 컴퓨팅 시스템을 개발한 후, 사용된 출처의 언어적 가정에 대한 검증의 필요성을 제기했습니다. 이번 작업의 주요 초점은 언어적 출처에서 인정된 마푸둥군 어근의 어휘 범주 분류입니다. 이 결과는 즉시 컴퓨터 분석기에서 구현되며, 마푸체 언어의 어휘 범주에 대한 불확실성을 해소하는 데 도움이 될 것으로 기대됩니다.

- **Technical Details**: 이 연구는 마푸체 동사 주제에 포함된 어휘 뿌리의 재분류 과정을 자세히 설명합니다. 명확한 분석을 위해 어근의 어휘 범주를 정확히 파악하는 것이 중요하며, 마푸둥군에서 동사 형식에 접미사가 추가될 때 필수적인 요소입니다. 이 연구에서 다루는 어휘적인 분류 작업이 완료된 후, 동사 뿌리의 원활한 가변성(vacancy) 평가가 다루어질 예정입니다.

- **Performance Highlights**: Smeets의 작업은 마푸체 동사형에서 약 100개의 접미사가 고정된 위치에 나타난다는 주요 기여를 하고 있습니다. 이 연구의 결과는 Düngupeyüm이라는 형태 분석 시스템의 개선에 직접적인 이익을 가져다 줄 것입니다. 최종적으로, 언어의 복잡한 형태와 접미사가 어떻게 동사 형식과 결합되는지를 이해하는 데 기여할 것으로 예상됩니다.



### Tractable Transformers for Flexible Conditional Generation (https://arxiv.org/abs/2502.07616)
- **What's New**: 본 논문에서는 Tractable Transformers (Tracformer)라는 새로운 비자기 회귀(Non-Autoregressive, NAR) 생성 모델을 제안합니다. 기존 NAR 모델들이 조건부 생성(Conditional Generation) 성능이 미흡하다는 점에 착안하여, Tracformer는 로컬 및 글로벌 컨텍스트 정보를 모두 반영하는 희소 스파스 인코더를 도입하였습니다. 이로 인해 다양한 조건부 생성 작업에 대해 더 강력하게 대응할 수 있는 모델이 탄생하게 되었습니다.

- **Technical Details**: Tracformer는 Transformer 아키텍처 기반의 생성 모델로서, 입력의 모든 토큰에서 파생된 글로벌 피처에만 의존하지 않고, 다중 컨텍스트 레벨에서 피처를 학습합니다. 이러한 방식으로 로컬 피처를 학습하여, 학습 시 본 적 없는 조건부 확률 쿼리에 대한 일반화 능력이 향상됩니다. 이 모델은 디코더를 통해 조건부 생성을 수행하며, 기존 알고리즘에 비해 조건부 생성 성능이 뛰어난 것으로 입증되었습니다.

- **Performance Highlights**: Tracformer는 텍스트 모델링에서 최신의 NAR 모델 및 AR 모델의 기준선에 비해 최첨단 성능을 발휘합니다. 특히, 기존의 SoTA 확산 언어 모델들보다 제로샷 조건부 생성 작업에서 더 나은 성능을 나타내었습니다. Tracformer의 이러한 성과는 조건부 생성 성능을 직접 평가하는 것이 중요하다는 점을 강조하며, NAR 모델의 개발에 있어 새로운 기준을 마련할 것으로 기대됩니다.



### DPO-Shift: Shifting the Distribution of Direct Preference Optimization (https://arxiv.org/abs/2502.07599)
- **What's New**: 이번 연구에서는 Direct Preference Optimization (DPO) 및 그 변형들이 언어 모델을 인간의 선호도에 맞게 조정하는 데 점점 인기를 얻고 있음을 보여주고 있습니다. 특히, 연구진은 선택된 응답의 분포를 조절하여 likelihood displacement 문제를 해결하기 위한 새로운 방법론, DPO-Shift를 제안합니다. DPO-Shift는 선택 확률을 증가시키는 동시에 보상 마진을 감소시키는 근본적인 균형을 제공합니다.

- **Technical Details**: DPO는 선택된 응답과 거부된 응답(choosen and rejected response) 간의 보상 차이를 최대화하기 위해 설계되었습니다. 그러나, DPO 훈련 과정에서 선택된 응답의 로그 확률과 거부된 응답의 로그 확률이 동시 감소하는 likelihood displacement 현상이 발생합니다. 본 연구에서는 Bradley–Terry 모델을 기반으로 한 파라미터 함수 f(λ)를 도입하여 이 문제를 해결하고, 이론적 분석을 통해 DPO-Shift의 효과를 입증합니다.

- **Performance Highlights**: DPO-Shift는 MT-Bench 및 설계된 win rate 실험과 같은 하위 작업에서 DPO에 비해 우수한 성능을 입증하였습니다. 연구 결과, 적절하게 선택된 f(λ) 값에 의해 선택된 확률을 증가시키면서도, 보상 마진의 정확도는 대폭 낮아지지 않고 고르게 유지되는 것을 확인했습니다. 따라서, DPO-Shift는 DPO의 likelihood displacement 문제를 효과적으로 완화할 수 있는 간단하고 이론적으로 뒷받침되는 솔루션임을 보여주었습니다.



### We Can't Understand AI Using our Existing Vocabulary (https://arxiv.org/abs/2502.07586)
Comments:
          Position paper

- **What's New**: 이 논문은 기존의 인간 언어 어휘에 의존하지 않고 AI를 이해하기 위해 네올로지즘(neologism), 즉 인간 개념과 기계 개념을 가리키는 새로운 단어를 만들어야 한다고 주장합니다. 이러한 새로운 단어는 기계가 이해할 수 있는 인간 개념을 정의하거나, 인간이 기계 개념을 이해하는 데 도움을 줄 수 있습니다. 이를 통해 인간과 기계 간의 상호작용에서 발생하는 커뮤니케이션 문제를 해결할 수 있을 것으로 기대합니다.

- **Technical Details**: 이 논문에서는 언어 모델 기반 AI 시스템을 이해하고 제어하는 과정에서의 커뮤니케이션 문제를 다룹니다. AI 시스템과 인간은 세계를 다르게 이해하며, 이로 인해 각기 다른 개념을 형성합니다. 저자들은 'length neologism'과 'diversity neologism' 등 네올로지즘을 활용해 모델의 응답 길이나 다양성을 제어할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 새로운 단어를 사용한 프롬프트는 모델의 응답을 효과적으로 조정하는 데 기여했으며, 이는 기존 모델의 가중치는 그대로 두면서도 더 정밀한 커뮤니케이션을 가능하게 합니다. 이러한 방식은 인간과 기계 간의 언어를 공동으로 발전시킬 수 있는 첫걸음으로 간주됩니다.



### O1 Embedder: Let Retrievers Think Before Action (https://arxiv.org/abs/2502.07555)
- **What's New**: 본 논문에서는 O1 Embedder라는 새로운 접근법을 제안합니다. 이 모델은 입력 쿼리에 대해 유용한 사고(thinking)를 생성한 후, 이를 바탕으로 관련 문서를 효과적으로 검색할 수 있는 임베딩(embedding)을 생성합니다. 이는 임베딩 모델에 사고 능력을 갖춘 최초의 사례로, 정보 검색(IR) 분야에서 중요한 통찰을 제공합니다. 더불어, 이 모델은 탐색-정제(exploration-refinement) 방법을 통해 최적의 검색 유틸리티를 제공하는 긴 형식의 사고를 생성합니다.

- **Technical Details**: O1 Embedder는 데이터 합성(data synthesis) 방법을 적용하여 초기 사고를 생성한 후, 검색 위원회(retrieval committee)를 통해 이를 정제합니다. 이 과정에서 각 위원회 член은 초기 사고와 목표 문서 간의 관련성을 점수화하여 최종적으로 최적의 사고를 선택합니다. 또한, 본 모델은 멀티태스크 교육(multi-task training) 방법을 도입하여 사전 훈련된 LLM을 정교화하고, 행동 복제(behavior cloning)와 대조 학습(contrastive learning)을 통해 검색 유틸리티를 향상시킵니다.

- **Performance Highlights**: O1 Embedder는 12개의 인기 있는 데이터셋을 활용한 종합적인 실험에서 기존 방법에 비해 큰 개선 결과를 보였습니다. 특히 복잡한 추론을 요구하는 다양한 검색 작업에서 강력한 일반화 능력을 보여주었습니다. 또한, Llama, Mistral, Qwen 등 여러 LLM 백본에서도 우수한 성능을 유지하여, 차세대 정보 검색 모델 개발에 기여할 것으로 기대됩니다.



### Unsupervised Translation of Emergent Communication (https://arxiv.org/abs/2502.07552)
Comments:
          19 pages (including appendix and bibliography), Accepted to AAAI 2025

- **What's New**: 이 연구는 Emergent Communication(EC)을 이해하고 자연어(NL)로의 번역 가능성을 탐구하는 새로운 방법론을 제시합니다. 특히, 기존의 병행 데이터 없이 Unsupervised Neural Machine Translation(UNMT) 기술을 활용하여 다양한 난이도의 참조 게임에서 형성된 EC를 번역합니다. 이러한 접근은 EC의 해석성과 번역 가능성을 높이고, AI가 생성한 언어의 이해를 도울 수 있는 새로운 지평을 여는 데 중요한 의미를 가집니다.

- **Technical Details**: 연구에서는 참조 게임을 통해 생성된 EC를 UNMT 기술을 활용하여 영어로 번역합니다. 게임은 다양한 난이도의 복잡성을 가지며, 불확실한 의미의 변형을 포함한 메시지 전달을 통해 생성된 언어의 구조를 분석합니다. 이 과정에서 EC 데이터셋은 에이전트 간의 메시지를 수집하여 생성되며, 영어 캡션 데이터셋은 EC의 배경을 설명하는 언어적 우선 요소로 사용됩니다.

- **Performance Highlights**: 실험 결과, UNMT는 AI가 생성한 언어를 자연어로 번역하는 데 성공적이라는 것을 보여주었습니다. 특히, Inter-category 설정에서는 번역 품질이 향상된 것으로 나타났으며, BLEU 및 METEOR 점수가 높은 것으로 확인되었습니다. 그러나 메시지 unpredictability가 증가함에 따라 번역 정확도가 떨어지는 경향도 발견되었습니다.



### Grammar Control in Dialogue Response Generation for Language Learning Chatbots (https://arxiv.org/abs/2502.07544)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 논문은 언어 모델을 기반으로 한 챗봇이 언어 학습자에게 효과적인 문법 연습을 위한 대화 응답을 생성할 수 있도록 하는 새로운 방법론을 제안합니다. 기존의 챗봇의 한계를 극복하기 위해, CEFR 기반의 문법 기술 저장소를 활용하여 문법 제어를 이루어내는 데 중점을 두고 있습니다. 이러한 접근 방식은 학습자가 필요로 하는 특정 문법 형태를 사용하여 대화의 질을 개선하려는 시도를 포함합니다.

- **Technical Details**: 논문에서는 LLM을 활용하여 대화 응답 생성을 문법적으로 제어하는 다양한 전략을 평가합니다. 이를 위해 GPT-3.5와 Llama3 모델을 사용하며, EGP에서 유래된 문법 형태를 기반으로 특정 대화 응답을 제한하는 데이터 세트를 생성하고 이를 Llama3에 맞추어 미세 조정합니다. 또한, 언급된 문법 형태의 출현 빈도와 응답의 질을 측정하여 해당 전략들의 효과를 비교합니다.

- **Performance Highlights**: 연구 결과, 새로운 디코딩 전략인 guided decoding이 문법 통제를 최적화하여 성공적으로 59.3%의 요청된 문법 형태를 포함하는 응답을 생성하는 것으로 나타났습니다. 또한, 학습자 시뮬레이션을 통해 47개의 문법 입력-출력 쌍 중 25개에서 문법 사용의 유의미한 증가를 확인했습니다. 이를 통해 문법 통제가 언어 학습에 도움을 줄 수 있는 가능성을 제시합니다.



### Corporate Greenwashing Detection in Text - a Survey (https://arxiv.org/abs/2502.07541)
Comments:
          35 pages, 1 figure, 21 pages (appendix), working paper

- **What's New**: 이 논문은 greenwashing(그린워싱) 식별을 위한 자연어 처리(NLP) 방법에 대한 포괄적인 조사를 제공합니다. 특히, 기후 관련 기업 커뮤니케이션에서 greenwashing을 나타낼 수 있는 잠재적인 오해의 소지가 있는 사례를 발견하기 위한 연구들이 61개 있었음을 확인했습니다. 이 연구들은 중간 단계(task)로 나뉘어져 있으며 각 단계에 대한 최신 접근 방식을 검토합니다.

- **Technical Details**: 연구는 환경 공개의 자연어 처리(NLP) 분석에 필요한 중간 단계의 작고 중요한 작업들을 포괄적으로 논의합니다. 이 과정에서 연구들은 믿을 만한 데이터셋이 부족하다는 문제에 봉착하였으며, 명확하게 라벨이 지정된 greenwashing 데이터셋이 없음을 강조했습니다. 따라서 연구자들은 오해의 소지가 있는 주장, 모호한 언어, 환경 보고의 불일치 등을 감지하는 방식으로 greenwashing의 이론적 정의를 발전시키고 이로 인해 생성된 측정 가능한 지표들을 사용하고 있습니다.

- **Performance Highlights**: 이 논문은 greenwashing 탐지를 위한 NLP 기반의 첫 번째 설문조사를 제공하며, 다양한 작업과 데이터셋을 중심으로 방법론을 체계화하고 있습니다. 그러나 많은 연구들이 불확실성 척도나 기준 벤치마크를 보고하지 않아 접근 방식을 비교하는 데 어려움이 있으며, 이로 인해 연구는 여전히 활발한 진행이 필요합니다. 또한, 기업이 실수로 오해의 소지가 있는 커뮤니케이션을 피할 수 있도록 자동화된 탐지의 필요성을 강조하고 있습니다.



### Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn Mor (https://arxiv.org/abs/2502.07490)
Comments:
          15 pages,7 figures

- **What's New**: 이번 연구에서는 Mask-Enhanced Autoregressive Prediction (MEAP)라는 새로운 LLM 훈련 패러다임을 제안합니다. MEAP는 Masked Language Modeling (MLM)을 Next-Token Prediction (NTP)와 매끄럽게 통합하여 key information retrieval의 성능을 향상시키기 위해 개발되었습니다. 기존에 사용되던 양방향 Attention이나 Encoder-Decoder 아키텍처의 필요성을 없애고, 사전 훈련이나 추론 중 추가적인 계산 오버헤드 없이도 효과적인 결과를 제공합니다.

- **Technical Details**: MEAP의 훈련 과정에서는 입력 토큰의 일부를 무작위로 마스킹하고, 이어서 디코더 전용 Transformer를 사용하여 NTP를 수행합니다. 이 방법은 LLM이 더 뛰어난 성능을 발휘하고, 긴 맥락 추론과 key information retrieval 작업에서 중요한 이점을 제공합니다. MEAP는 사전 훈련 과정에서도 다른 모델들과 비교했을 때 우수한 데이터 효율성을 보여주며, NTP보다 11.77% 더 높은 성능을 기록하였습니다.

- **Performance Highlights**: MEAP는 NTP와 비교하여 key information retrieval과 Long-context reasoning에서 상당한 성능 향상을 나타냈습니다. Needle in a Haystack와 Multi-Document Question Answering (MDQA) 작업에서 각각 33% 및 27.2% 성능 향상을 달성하였고, NTP보다 200B 훈련 토큰으로도 유사한 성능을 발휘할 수 있음을 보여주었습니다. MEAP는 또한 간접적 정보가 적은 상황에서도 NTP보다 더욱 우수한 결과를 보여줍니다.



### Multi-Agent Collaboration for Multilingual Code Instruction Tuning (https://arxiv.org/abs/2502.07487)
- **What's New**: 이 논문은 코드 LLM(대형 언어 모델)이 다국어 프로그래밍 언어 간의 지식 전달을 촉진하기 위해 새로운 다중 에이전트 협력 프레임워크를 도입합니다. 기존의 방법들이 각 프로그래밍 언어를 독립적으로 다루는 경향이 있었던 반면, 본 연구에서는 여러 언어 특화 에이전트가 협력하여 고품질 다국어 지침 데이터를 생성합니다. 이를 통해 Qwen2.5-xCoder 모델의 다국어 정확도를 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 다중 에이전트 시스템에서는 각 에이전트가 특정 프로그래밍 언어에 특화되어 있습니다. 초기 단계에서 코드 스니펫으로부터 언어특화 지침 데이터를 생성하며, 각 에이전트는 이러한 데이터를 기반으로 새로운 지침과 해결책을 공동으로 작성합니다. 에이전트는 자기 생성 이력을 기억하여 강점과 약점을 평가함으로써, 교차 언어 학습을 촉진합니다.

- **Performance Highlights**: Qwen2.5-xCoder는 Python 벤치마크인 HumanEval 및 MBPP는 물론, Python, Java, C++, C#, TypeScript, PHP, Bash로 구성된 확장 다국어 벤치마크인 MultiPL-E에 대해 평가가 진행되었습니다. 실험 결과, Qwen2.5-xCoder는 이전 기준 모델들과 비교하여 우수한 성능을 발휘하며, 서로 다른 언어 간의 지식 전이를 효과적으로 수행하여 언어 간 간섭을 줄이는 데 기여합니다.



### PerCul: A Story-Driven Cultural Evaluation of LLMs in Persian (https://arxiv.org/abs/2502.07459)
Comments:
          Accepted at NAACL 2025 Main Conference, the dataset is available on HuggingFace (see this https URL)

- **What's New**: 이 논문에서는 전통적으로 서양 문화를 반영하는 대형 언어 모델(LLMs)의 한계를 분석하고, 페르시아 문화에 대한 감수성을 평가하는 새로운 데이터세트인 PerCul을 소개합니다. PerCul은 페르시아어로 된 문화적 요소가 담긴 이야기 기반의 다지선다형 질문을 포함하여, 기존의 데이터세트들과는 달리 원어민의 입력을 바탕으로 만들어졌습니다. 이러한 접근 방식은 항상 번역에 의존하지 않도록 설계되어, 다양한 문화적 배경을 충실히 반영할 수 있도록 합니다.

- **Technical Details**: PerCul 데이터세트는 Hall의 문화 삼각형 이론을 바탕으로 문화 카테고리를 정의한 뒤, 원어민 주석가들이 기술적 세부 정보를 생성하고 LLM들이 이야기의 줄거리를 만들어냅니다. 생성된 이야기는 엄격한 인간 수정 과정을 거쳐 최종적으로 다지선다형 질문이 구성됩니다. 이를 통해 LLM의 문화적 이해도를 평가하기 위한 명확한 기준을 제공하며, 모든 질문은 불특정한 문화적 개념이 아닌 실제 페르시아 문화 요소를 포함하도록 했다.

- **Performance Highlights**: PerCul의 성능 평가에서 가장 우수한 클로즈드 소스 모델과 일반인 기준 간의 차이는 11.3%, 최고의 오픈 웨이트 모델을 사용했을 때는 이 차이가 21.3%로 증가하는 것으로 나타났습니다. 또한 연구 결과는 페르시아 문화에 대한 이해가 부족한 LLM들이 표면 수준의 세부 정보에 의존하는 경향이 있음을 보여줍니다. 이는 페르시아어로 조정된 LLM들이 다국어 기본 모델보다 성능이 낮은 원인을 설명하는 데 기여합니다.



### Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon (https://arxiv.org/abs/2502.07445)
- **What's New**: 이번 연구에서는 Chameleon Benchmark Overfit Detector (C-BOD)라는 메타 평가 프레임워크를 소개하여, 모델이 벤치마크에 특화된 패턴에 과도하게 의존하고 있는지 확인하고 이를 감지하는 방법을 제안합니다. C-BOD는 입력의 의미를 유지하면서 질문 형식을 변형하여 모델의 실질적 성능을 평가할 수 있는 기회를 제공합니다. 이는 LLM의 진정한 언어 이해를 과시하기보다는 단순히 표면적인 패턴에 기반한 성능 수치를 드러내는 방법론입니다.

- **Technical Details**: C-BOD는 벤치마크 프롬프트를 다양하게 변형하여 LLM의 성능 변화를 측정하는 방식으로 작동합니다. 이 방법은 입력을 변형시키기 위해 재구성 도구(T)와 왜곡 매개변수(μ)를 사용하여 원본 및 변형된 데이터셋에 대한 평가를 수행합니다. 이 프레임워크는 통계적 검정을 통해 성능 차이가 과적합을 나타내는지를 판단하는 데 중점을 두며, 모델이 기억한 패턴에서 벗어나지 못할 경우 기인하는 성과 손실을 보여줍니다.

- **Performance Highlights**: 실험 결과, 26개의 주요 LLM 중 20개 모델이 통계적으로 유의미한 성능 저하를 보였으며, 평균적으로 2.15%의 성능 저하가 확인되었습니다. 특히 높은 기본 정확도를 가진 모델은 더 큰 성능 차이를 보였으며, 큰 LLM은 재구성에 더 민감하게 반응하는 경향이 있었습니다. 반면, Llama 패밀리 모델은 유의미한 저하를 보이지 않아 피상적 단서에 대한 의존성이 줄어드는 경향을 보였습니다.



### Hierarchical Document Parsing via Large Margin Feature Matching and Heuristics (https://arxiv.org/abs/2502.07442)
Comments:
          DocUI@AAAI-25, 2 pages, technical report

- **What's New**: 본 연구에서는 AAAI-25 VRD-IU 챌린지에서 1위에 선정된 독서 문서 구조 파싱에 대한 새로운 접근 방식을 제시합니다. 이 방법은 대규모 마진 손실(large margin loss)을 이용하여 특징 구분을 개선하며, 휴리스틱 규칙(heuristic rules)을 활용하여 계층 관계를 정제합니다. 심층 학습 기반의 매칭 전략과 탐욕 알고리즘(greedy algorithms)을 결합하여 계산 효율성을 유지하면서도 정확성을 크게 향상시켰습니다.

- **Technical Details**: 이 논문에서는 CLIP의 손실 함수를 개선하여 계층 관계 예측에 필요한 대규모 마진 손실을 도입합니다. 또한, 특정 카테고리에 속하는 엔티티는 부모를 가지지 않으며, 계층적 관계가 강하게 존재하는 카테고리들이 있습니다. 이러한 구조적 패턴을 통해 부모-자식 관계 예측의 정확성과 계산 효율성을 개선하고 있습니다.

- **Performance Highlights**: 우리의 방법은 개인 리더보드에서 0.98904의 정확도를 달성하며, AAAI-25 VRD-IU 챌린지에서 최첨단 성능을 나타냅니다. 깊이 학습과 규칙 기반의 세분화를 결합하여 문서 구조 파싱의 효과를 입증하였으며, 향후 더 다양한 데이터 세트 탐색과 휴리스틱 규칙 세분화를 통한 일반화를 포함한 향후 연구 방향을 제시합니다.



### RomanLens: Latent Romanization and its role in Multilinguality in LLMs (https://arxiv.org/abs/2502.07424)
Comments:
          18 pages, 18 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 영어 중심의 코퍼스에서 훈련됨에도 불구하고 놀라운 다국어 일반화 능력을 보이는 이유를 탐구합니다. 구체적으로, 우리는 비라틴 문자 언어에서 로마자화(romanization)가 다국어 처리에 미치는 역할을 조사하며, 이를 ‘Latent Romanization’이라고 명명합니다. 연구 결과에 따르면, LLM은 다국어 다음 토큰 생성 과정에서 중간 층에서 로마자로 표현된 단어를 자주 사용합니다.

- **Technical Details**: 로마자화는 비라틴 스크립트를 라틴 문자로 표현하는 방법으로, 이는 LLM이 언어 중립적인 개념 공간과 언어 특정 출력 표현 간의 교량 역할을 할 수 있음을 보여줍니다. 본 연구에서는 LLaMA-2 7B 모델을 사용하여 다음 토큰 생성을 시각화하고, activation patching 기법을 활용하여 로마자화와 네이티브 스크립트 간의 개념을 비교 분석합니다. 이를 통해 LLM이 다양한 스크립트에서 의미를 어떻게 처리하는지에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 로마자화된 표현은 모델의 층에서 네이티브 스크립트 표현보다 더 빨리 나타나는 경향이 있습니다. 이는 로마자화가 언어 특정 임베딩으로의 빠른 진행을 촉진할 수 있음을 시사합니다. 또한, 실험 결과는 LLM이 로마자화된 입력과 네이티브 스크립트 입력 간에 개념 정보를 유사하게 인코딩한다는 것을 보여줍니다.



### Entity Linking using LLMs for Automated Product Carbon Footprint Estimation (https://arxiv.org/abs/2502.07418)
- **What's New**: 이번 연구는 제조업체들이 온실가스 배출을 줄이기 위한 노력을 하고 있는 가운데, 제품 구성 요소의 환경 영향을 자동으로 매핑하기 위해 대형 언어 모델(LLMs)을 활용한 시스템을 제안합니다. 이 시스템은 제조업체의 자재 명세서(BOM)와 생애 주기 평가(LCA) 데이터베이스 간의 연결을 촉진하여 지속 가능성 관행을 개선하고, 수동 데이터 처리의 필요성을 줄입니다. 이러한 접근 방식은 제조자들이 보다 효율적으로 환경 영향을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: 이 방법론은 생명 주기 평가(LCA) 데이터베이스와 BOM의 구성 요소를 연결하는 세 가지 단계로 구성됩니다. 첫 번째 단계에서는 구성 요소에 대한 추가 컨텍스트를 제공하기 위해 관련 문서(technical datasheet)를 선택합니다. 두 번째 단계에서는 LLM을 이용하여 구성 요소의 제조 과정을 설명하는 프롬프트를 생성하고, 마지막으로 LCA 데이터베이스와의 맵핑을 수행하여 최종 carbon emissions 평가를 지원합니다.

- **Performance Highlights**: 제안된 접근 방식은 BOM의 상세 정보를 활용하여 제품의 탄소 발자국(root carbon footprint) 추정 정확도를 높입니다. LLM을 활용하는 점에서 새롭고, datasheet를 통해 정보를 통합함으로써 기존의 수동 매핑 방식보다 빠른 결과를 도출할 수 있습니다. 이는 환경 영향 평가 분야에서의 자동화 수준을 한층 끌어올릴 것으로 기대됩니다.



### Target-Augmented Shared Fusion-based Multimodal Sarcasm Explanation Generation (https://arxiv.org/abs/2502.07391)
- **What's New**: 이번 연구에서는 기존의 sarcasm 설명 모델들에서 간과되었던 sarcasm의 대상(target) 정보를 활용하는 새로운 모델, TURBO를 제안하였습니다. TURBO는 이미지와 자막 간의 상호 변동성 관계를 활용하는 novel shared-fusion 메커니즘을 설계하여, sarcasm의 의도된 아이러니를 보다 정확하게 해석할 수 있도록 돕습니다. 또한, 이 모델은 기존의 MORE 데이터셋을 확장하여 새로운 MORE+ 데이터셋을 생성하고, 이를 통해 모델의 우수성을 평가하였습니다.

- **Technical Details**: TURBO 모델은 세 가지 주요 구성 요소로 이루어져 있습니다: a) 외부 지식 개념을 활용하는 knowledge-infusion 기능, b) multimodal fused representation을 얻기 위한 novel shared-fusion 메커니즘, c) sarcasm의 대상 정보를 활용하여 집중된 설명 생성을 유도하는 기능입니다. 이러한 구성 요소들이 결합되어 멀티모달 환경에서의 sarcasm 해석에 개선된 성능을 발휘합니다.

- **Performance Highlights**: TURBO 모델은 기존 상태-of-the-art 모델인 TEAM과 비교했을 때 평균적으로 3.3%의 성능 향상을 보였습니다. 또한, LLM(대규모 언어 모델)들을 zero 및 one-shot 설정에서 적용했으며, 이들이 생성하는 설명이 때때로 sarcasm의 중요한 뉘앙스를 포착하지 못하는 경향이 있음을 발견했습니다. 이와 더불어, TURBO가 생성한 설명이 기존 시스템에 비해 정성적인 평가에서도 우수함을 입증하였습니다.



### Parametric type design in the era of variable and color fonts (https://arxiv.org/abs/2502.07386)
Comments:
          Conference: Grapholinguistics in the 21st century - From graphemes to knowledge

- **What's New**: 이 논문은 파라메트릭 폰트(parametric fonts)의 현대적 디자인 프로세스를 다루고 있습니다. 1980년대 도널드 크누스(Donald Knuth)의 메타폰트(MetaFont) 기술로 시작된 이 개념은 최근의 트렌드인 가변 폰트(variable fonts)의 부활과 함께 새롭게 각광받고 있습니다. 저자는 두 개의 가변 폰트를 이 방법으로 생성했으며, 이를 자유로운 오픈소스 라이선스 하에 배포했습니다.

- **Technical Details**: 논문에서는 메타포스트(MetaPost)를 사용한 파라메트릭 디자인 원칙에 기반한 디자인 방법론과 워크플로우(workflow)를 설명합니다. 메타폰트 기술은 그래픽 사용자 인터페이스(GUI) 시대 이전의 타이포그래피 디자인에서 유산으로 간주되는 경향이 있지만, 이 연구는 이 원칙의 현대적 적용을 탐구합니다. 아울러, 저자는 이 과정에서 얻은 통찰(insights)도 공유합니다.

- **Performance Highlights**: 저자가 소개한 두 개의 가변 폰트는 파라메트릭 디자인을 통해 전통적 디자인 방식과의 차별성을 보여줍니다. 이 논문은 폰트 디자인에 대한 새로운 접근 방식을 제시하며, 오픈소스 커뮤니티에 기여하는 점에서 중요한 의미가 있습니다. 결과적으로, 디자인 프로세스에 대한 심도 깊은 통찰을 제공함으로써 타이포그래피 분야에 긍정적인 영향을 미칠 것으로 기대됩니다.



### LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation (https://arxiv.org/abs/2502.07365)
- **What's New**: 이번 연구에서는 Large Language Model(LLM)의 긴 컨텍스트 처리 성능 저하의 원인을 규명했습니다. 이를 통해 기존 모델과 확장된 모델 간의 분포 불일치(distribution drift)와 연속적 사전학습에서의 재앙적인 망각(catastrophic forgetting) 문제를 확인하였습니다. 이러한 문제를 해결하기 위해 Long Context Pre-training with Restoration Distillation (LongReD)라는 새로운 접근 방식을 제안하였습니다.

- **Technical Details**: LongReD는 전통적인 긴 텍스트 훈련 외에도, 단기 텍스트에 대해 원래 모델의 숨겨진 상태(hidden states)를 증류(distill)하는 목표를 포함합니다. 이를 통해 모델은 원래 분포를 보다 정확하게 시뮬레이션하고, 분포의 불일치를 최소화하여 성능 저하를 완화하도록 설계되었습니다. 또한, Short-to-Long Distillation 훈련 목표를 제시하여, 단기와 장기 훈련 간의 간극을 해소합니다.

- **Performance Highlights**: 실험 결과 LongReD는 단기 텍스트 작업에서 원래 모델의 성능을 유지하면서도 긴 텍스트 처리 능력을 유지하거나 심지어 개선하는데 성공했습니다. Llama-3-8B와 Mistral-7B-v0.3 모델을 통해 얻은 결과들은 단기 작업에서의 성능 손실을 방지하고, 장기적인 모델링 능력을 증진시켰음을 입증합니다.



### Bridging the Evaluation Gap: Leveraging Large Language Models for Topic Model Evaluation (https://arxiv.org/abs/2502.07352)
Comments:
          accepted by IRCDL 2025

- **What's New**: 이 연구는 과학 문헌에서 역동적으로 발전하는 주제 분류법을 자동으로 평가하기 위한 프레임워크를 제안합니다. 기존의 정적 평가 방법들은 연구 분야의 급격한 변화에 대응하기 어려운 점에서 한계를 가지고 있습니다. 제안한 방법은 Large Language Models (LLMs)를 활용해 주제의 질을 측정하며, 전문가의 주석자나 좁은 통계적 지표에 대한 의존도를 줄입니다. 이 연구는 LLM이 전통적인 평가 전략의 한계를 극복할 수 있는 보다 포괄적이고 역동적인 대안을 제공함을 강조합니다.

- **Technical Details**: 논문에서는 주제 모델의 출력을 평가하기 위해 LLM을 활용하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 주제 모델의 질을 평가하는 다양한 메트릭을 포함하며, 일관성, 반복성, 다양성 및 주제-문서 정렬과 같은 특성을 포괄합니다. LLM을 위한 맞춤형 프롬프트를 설계하여 다양한 데이터셋과 모델링 기술에서의 일관성과 해석 가능성을 보장합니다. 또한, 이 프레임워크는 20 Newsgroups와 AGRIS의 학술 문서 하위 집합에 대한 실험을 통해 그 강건성과 효과성을 입증합니다.

- **Performance Highlights**: 이 연구는 LLM을 활용한 주제 모델 평가를 위한 확장 가능하고 포괄적인 솔루션을 제공합니다. 제안한 접근 방식은 전통적인 방법의 한계들을 보완하며, 연구 발견 효율성을 높이고 과학 정책 및 자금 지원 전략에서 정보 기반의 의사 결정을 지원하는 데 기여할 수 있습니다. LLM의 활용으로 인해 증가하는 과학 문헌 관리의 필요성을 충족하는 동시에, 동적이고 맥락에 적합한 문제를 해결할 수 있는 가능성을 제시합니다.



### BenchMAX: A Comprehensive Multilingual Evaluation Suite for Large Language Models (https://arxiv.org/abs/2502.07346)
- **What's New**: BenchMAX는 대규모 언어 모델(LLMs)의 다양한 능력을 공정하게 평가할 수 있는 다국어 평가 벤치마크로 소개됩니다. 이 벤치마크는 17개 언어를 포함하며, 번역 및 평가의 품질이 높습니다. 기존의 단순한 이해 작업에 초점을 맞춘 다국어 벤치마크와는 달리, BenchMAX는 높은 복잡성과 언어 불문 능력을 측정합니다.

- **Technical Details**: BenchMAX는 LLM의 6가지 핵심 능력, 즉 지시 이행(instruction following), 추론(reasoning), 장기 맥락 이해(long context understanding), 코드 생성(code generation) 등을 평가합니다. 이 평가를 위해 3명의 원어민 주석자가 독립적으로 기계 번역된 데이터를 수집 및 평가합니다. 또한, 도메인 번역(domain translation) 과정에서의 새로운 도전 과제를 제시하여 번역 과정에서 세부적인 제어나 특정 도메인 용어의 이해를 필요로 합니다.

- **Performance Highlights**: BenchMAX에서 수행된 광범위한 실험을 통해, LLM들의 언어 불문 능력이 언어에 따라 상이함이 드러났습니다. 모델 크기를 단순히 확장하는 것으로는 언어 간의 성능 격차를 줄일 수 없다는 점도 강조됩니다. 이 연구는 다국어 LLM의 발전을 위해 BenchMAX를 통해 수집된 분석 결과가 더욱 나아가 LLM의 능력에 대한 이해를 증진시킬 것임을 알립니다.



### Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering (https://arxiv.org/abs/2502.07340)
- **What's New**: NOVA는 대규모 언어 모델(LLMs)이 사용자의 지침을 따르도록 조정하고 허위정보(hallucination)를 줄이기 위한 새로운 프레임워크입니다. 본 연구에서는 LLM이 지침 데이터에서 익숙하지 않은 지식을 포함하는 경우 자주 발생하는 과신을 줄이기 위해, 고품질 데이터를 식별하기 위해 Internal Consistency Probing (ICP)와 Semantic Equivalence Identification (SEI) 기법을 도입합니다. 이 방법을 통해 지침 데이터와 타겟 응답과의 관계를 평가하여 모델의 이해도를 높이고 허위정보 발생을 억제합니다.

- **Technical Details**: NOVA는 두 가지 주요 기법인 Internal Consistency Probing (ICP)와 Semantic Equivalence Identification (SEI)를 포함합니다. ICP는 LLM이 지침에 대해 자기 생성된 응답 간의 일관성을 측정하여 해당 지식이 내재화되었는지 평가합니다. SEI는 생성된 응답을 의미적으로 군집화하고, 타겟 응답과 가장 잘 일치하는 군집을 찾음으로써 LLM이 타겟 응답을 얼마나 잘 이해하는지 평가합니다.

- **Performance Highlights**: NOVA 프레임워크의 실험 결과, 허위정보 발생률이 유의미하게 감소했으며, LLM이 지침을 따르는 능력 또한 경쟁력을 유지하는 것으로 나타났습니다. 이러한 개선은 고품질 데이터를 식별하고, LLM의 지식과 정렬하여 더 효과적인 학습을 가능하게 합니다. 이로 인해 실세계 앱에서 LLM의 신뢰성을 높일 수 있습니다.



### MEMIT-Merge: Addressing MEMIT's Key-Value Conflicts in Same-Subject Batch Editing for LLMs (https://arxiv.org/abs/2502.07322)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 지식 편집 기술이 모델 내 지식을 완전 재훈련 없이 수정할 수 있도록 진화하고 있음을 강조하고 있습니다. 특히, MEMIT이라는 일괄 편집 알고리즘의 편집 효과성이 동일 주제를 가진 다수의 편집을 처리할 때 크게 저하된다는 중요한 제약을 발견했습니다. 저자들은 MEMIT의 키-값(key-value) 모델링 프레임워크에서 이러한 문제가 발생한다고 분석하였으며, MEMIT의 성능 저하를 해결하기 위해 MEMIT-Merge라는 향상된 접근 방식도 제안하고 있습니다.

- **Technical Details**: MEMIT은 MLP(다층 퍼셉트론) 모듈의 출력 선형 층을 수정하여 지식을 업데이트하는 주요 구조적 특징을 가지고 있으며, 배치 내에서 다수의 지식 인스턴스를 동시에 수정하는 능력을 갖추고 있습니다. 그러나 동일 주제를 가진 편집을 처리할 때 식별 가능한 키들이 다양한 값을 생성해야 하는 상황에서, 업데이트 충돌이 발생하여 성능 저하가 발생합니다. 이를 해결하기 위해, MEMIT-Merge는 동일한 주제를 가진 사실들의 값 계산 프로세스를 병합하여 성능 저하 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, MEMIT의 성공률이 대규모 배치에서 50%로 급감할 때, MEMIT-Merge는 90% 이상의 성공률을 유지하며 동일 주제 배치 편집에서 뛰어난 강인성을 보여줍니다. 이와는 대조적으로, 서로 다른 주제를 가진 데이터에서는 두 방법이 유사한 성능을 나타내지만, 동일 주제 데이터에서는 MEMIT-Merge가 일관되게 우수한 성능을 보입니다. 이러한 결과는 실제 응용에서 중요하게 여겨지는 동일 주제 지식의 배치 편집을 더욱 효과적으로 수행할 수 있는 가능성을 시사합니다.



### CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction (https://arxiv.org/abs/2502.07316)
- **What's New**: 이 논문은 대규모 언어 모델의 추론 능력을 향상시키기 위한 새로운 접근법인 CodeI/O를 제안합니다. 기존의 연구가 수학 문제 해결이나 코드 생성과 같은 특정 영역에 주로 초점을 맞춘 것과 달리, 이제는 다양한 추론 과제에 대한 성능 개선을 목표로 합니다. CodeI/O는 문맥적으로 기초한 코드 내의 다양한 추론 패턴을 집약하여 자연어로 입력/출력을 예측하는 형식으로 변환하는 방식으로 작동합니다.

- **Technical Details**: 이 연구는 여러 출처에서 수집한 함수들을 변환하여 복잡한 추론 패턴을 포함하도록 설계되었습니다. 입력/출력 예측을 위해, 각 함수에 대해 텍스트 쿼리와 함께 예측을 요구하여 모델이 코드와 자연어를 통한 추론 능력을 배양하게 합니다. 이 과정에서는 DeepSeek-V2.5라는 오픈소스 모델을 사용하여 데이터를 처리하며, 45만 개 이상의 함수와 350만 개의 훈련 샘플이 포함됩니다.

- **Performance Highlights**: CodeI/O와 CodeI/O++는 7B에서 30B의 파라미터를 가진 4가지 기본 모델에 대해 검증되었으며, 14개의 다양한 기준에서 평가되었습니다. 이 새로운 접근법은 기존 벤치마크에 비해 높은 점수를 기록하며, 특정 소수의 평가 기준에서만 점수를 높이는 것이 아니라, 거의 모든 벤치마크에서 일관된 성과를 보였습니다. 이러한 결과는 CodeI/O의 다양한 추론 능력을 강조합니다.



### Small Language Model Makes an Effective Long Text Extractor (https://arxiv.org/abs/2502.07286)
Comments:
          AAAI'25, 9 pages, 1 appendix pages

- **What's New**: 이 논문은 긴 텍스트에서 엔터티를 효과적으로 추출하기 위한 경량(span-based) NER 방법인 SeNER을 소개합니다. SeNER은 bidirectional arrow attention 메커니즘과 LogN-Scaling을 통해 긴 텍스트를 효율적으로 인코딩하며, BiSPA 메커니즘을 사용하여 후보 토큰 쌍 간의 상호 작용을 모델링합니다. 이러한 방법론을 통해 GPU 메모리 사용량을 절감하면서도 높은 정확도를 유지합니다.

- **Technical Details**: SeNER의 핵심 구성 요소로는 bidirectional arrow attention 메커니즘과 BiSPA가 있습니다. 전자는 로컬 및 글로벌 컨텍스트를 동시에 인코딩하며, 후자는 토큰 쌍 간 상호작용을 모델링하고 불필요한 계산을 줄입니다. LogN-Scaling을 통해 다양한 입력 길이에 대한 엔트로피의 불안정성을 해결하며, 학습 과정에서 전체 단어 마스킹 전략과 LoRA 기술을 활용합니다.

- **Performance Highlights**: SeNER은 세 가지 긴 NER 데이터셋에서 최첨단 추출 정확도를 달성하며, 기존의 고급 span-based NER 방법보다 6666배 긴 텍스트를 처리할 수 있는 능력을 보여줍니다. 또한, 상대적으로 적은 모델 파라미터로 높은 성능을 유지하는 것이 특징입니다. 광범위한 실험 결과는 SeNER의 우수성을 뒷받침하고 있습니다.



### GENERator: A Long-Context Generative Genomic Foundation Mod (https://arxiv.org/abs/2502.07272)
- **What's New**: 본 연구에서는 386억 염기쌍(bases pairs)에서 훈련된 GENERator라는 생성형 유전체 기초 모델을 소개합니다. 이 모델은 98k 염기쌍의 문맥 길이와 12억 개의 파라미터를 갖추고 있으며, 다양한 유전적 벤치마크에서 최첨단 성능을 유지하고 있습니다. 이를 통해 기존 모델들이 가진 제한성과도 차별점을 보이고 있습니다.

- **Technical Details**: 자연어 처리(NLP)의 발전을 기반으로 한 GENERator는 eukaryotic DNA의 방대한 데이터셋을 사용하여 훈련되었습니다. 이 모델은 단백질-코딩 DNA 시퀀스를 생성하면서 구조적으로 알려진 단백질 패밀리와 유사한 단백질 생성을 보장합니다. 또한, 프로모터 시퀀스를 최적화하는 능력도 포함되어 있어 유전체 연구에 크게 기여할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: GENERATOR는 기존과 새로운 벤치마크 모두에서 최첨단 성능을 발휘하며 유전자 조절 및 단백질 합성의 복잡한 과정을 이해하는 데 도움을 제공합니다. 특히, 생성된 단백질 시퀀스는 특정 유전자 활동 프로필에 따라 조절되는 생물학적 과정에 적합하게 설계될 수 있습니다. 이로 인해, GENERATOR는 유전체 연구 및 생명 공학 발전을 위한 중추적인 도구로 자리 잡을 것으로 기대됩니다.



### Graph RAG-Tool Fusion (https://arxiv.org/abs/2502.07223)
Comments:
          25 pages, 14 figures, 2 tables

- **What's New**: 이 논문에서는 Graph RAG-Tool Fusion이라는 새로운 기법을 소개하여 도구 선택 프로세스를 개선합니다. 기존의 RAG 기반 도구 검색은 도구 간의 구조적 의존성을 포착하는 데 한계가 있었고, 이로 인해 검색 정확도가 저하되었습니다. Graph RAG-Tool Fusion은 벡터 기반 검색과 효율적인 그래프 탐색을 결합하여 관련 도구와 그 의존성을 포괄적으로 포착합니다.

- **Technical Details**: 이 접근법은 사전 정의된 도구 지식 그래프 내에서 도구의 관련 하위 그래프를 초기 벡터 검색을 통해 식별하고, 이어서 그래프 탐색을 통해 해당 도구의 직접 및 간접 의존성을 가져옵니다. 이를 통해 Graph RAG-Tool Fusion은 도구와 사용자 쿼리 간의 의미적 관련성과 관계를 모두 캡처하며, 순수한 벡터 기반 접근법의 한계를 극복합니다. 각 도구는 일반 도구 또는 핵심 도구로 분류하며, 의존성이 있는 도구의 관계를 그래프 모델로 나타냅니다.

- **Performance Highlights**: Graph RAG-Tool Fusion은 ToolLinkOS 및 ToolSandbox 벤치마크에서 각각 71.7% 및 22.1%의 절대 개선을 달성하며, 기존의 naïve RAG에 비해 우수한 성능을 보여줍니다. ToolLinkOS는 573개의 가상의 도구로 구성되어 있으며, 다양한 산업을 포괄하고 각 도구별로 평균 6.3개의 의존성을 포함합니다. 이러한 성과는 도구 선택의 실현 가능성을 높이며, 실제 환경에서의 응용을 기대하게 합니다.



### A Large-Scale Benchmark for Vietnamese Sentence Paraphrases (https://arxiv.org/abs/2502.07188)
Comments:
          Accepted in NAACL 2025 Findings

- **What's New**: 이 논문은 ViSP라는 고품질의 베트남어 문장 패러프레이징(paraphrasing) 데이터셋을 소개합니다. 이 데이터셋은 다양한 도메인에서 수집된 120만 개의 원문-패러프레이즈 쌍으로 구성되어 있으며, 자동 패러프레이즈 생성과 수동 평가를 결합한 하이브리드 접근법으로 구성됩니다. 연구진은 BART, T5 및 여러 대형 언어 모델을 사용한 실험을 진행하였으며, 이는 베트남어 패러프레이징에 대한 최초의 대규모 연구로 알려져 있습니다.

- **Technical Details**: ViSP 데이터셋은 공개 리소스에서 원본 베트남어 문서를 수집하여 구축되었습니다. 모델은 문장을 주제별로 분류하였으며, 교육 및 평가 그룹에 나누어 문장을 패러프레이징하는 작업을 수행하였습니다. 생성된 패러프레이즈는 7명의 주석자에 의해 검토되어 품질을 보장하여, 결과적으로 높은 정확도를 갖춘 데이터셋을 제공하게 되었습니다.

- **Performance Highlights**: Gemini 모델은 여러 라운드에서 인간의 패러프레이즈 생성 능력을 뛰어넘는 성능을 보여주었습니다. 이 모델은 H1에 대해 83.33%의 승률을 기록하며, AI가 수동 패러프레이징 생성에 효과적임을 강조합니다. 연구 결과는 미래 베트남어 패러프레이징 연구와 응용의 기초가 될 것으로 기대됩니다.



### Perceived Confidence Scoring for Data Annotation with Zero-Shot LLMs (https://arxiv.org/abs/2502.07186)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서 데이터 주석 작업을 위한 새로운 기술인 Perceived Confidence Scoring (PCS)를 소개합니다. PCS는 Metamorphic Relations (MRs)를 활용하여 입력의 분류에 대한 LLM의 신뢰도를 평가합니다. 이 과정에서 생성된 텍스트 변형들은 주어진 입력의 의미적으로 동등한 버전으로, LLM 응답의 일관성을 분석하여 신뢰 점수를 계산합니다.

- **Technical Details**: PCS는 LLM과 MRs의 출력 간의 일관성을 분석하여 신뢰도를 측정합니다. Metamorphic Testing (MT)의 원칙에 따라, 변형된 입력은 입력과 유사한 주석 레이블을 갖기를 기대합니다. 또한, Perceived Differential Evolution (PDE) 알고리즘을 사용하여 특정 분류 작업을 위한 최적 가중치 조정을 통해 LLM의 신뢰성 향상을 도모합니다.

- **Performance Highlights**: 실험 평가 결과 PCS는 Llama-3-8B-Instruct에서 4.96%, Mistral-7B-Instruct-v0.3에서 10.52%의 정확도 향상을 보였으며, Gemma-2-9b-it 모델에서는 9.39%의 향상을 나타냈습니다. 세 모델을 결합했을 때 PCS는 다수결보다 7.75% 더 우수한 성능을 보여주었습니다.



### Refine Knowledge of Large Language Models via Adaptive Contrastive Learning (https://arxiv.org/abs/2502.07184)
Comments:
          Accepted to ICLR 2025

- **What's New**: 본 연구는 Large Language Models (LLMs)의 환각(hallucination) 문제를 완화하기 위해 Adaptive Contrastive Learning 전략을 제안합니다. 이 방법은 LLM이 가진 지식의 실제 숙련도에 따라 긍정적 및 부정적 샘플을 유연하게 구성하여 LLM의 지식 표현을 최적화하는 데 중점을 둡니다. 이를 통해 LLM은 기존의 올바른 지식을 강화하고, 불확실한 지식에 대한 이해를 심화하며, 잘못된 지식은 잊고 부족한 지식은 솔직히 인정할 수 있습니다.

- **Technical Details**: Adaptive Contrastive Learning 전략은 LLM의 지식 경계를 정교하게 조정하고, LLM이 아는 지식(I Know Rate, IK)과 모르고 있는 지식(I Don’t Know Rate, IDK)의 상한 및 하한을 설정합니다. 이 방법은 잘 주석된 Q&A 쌍을 사용하여 LLM의 응답 정확도를 계산하고, 정확도가 각 지식 경계에 따라 서로 다른 학습 샘플을 생성합니다. 알고리즘은 긍정적 샘플과 부정적 샘플 사이의 거리 최적화에 중점을 두어, 모델이 정확한 지식을 학습하고 부정확한 지식을 버리게 합니다.

- **Performance Highlights**: 광범위한 실험 및 데이터 분석에 따르면, 제안된 Adaptive Contrastive Learning 전략은 여러 LLM에서 높은 신뢰도(Truthful rate)를 달성하였습니다. 이 결과는 LLM의 응답의 유효성과 정직성을 개선하는 데 중요한 기여를 한다고 볼 수 있습니다. 본 연구는 LLM의 지식 표현 개선과 환각 문제 완화에 있어 새로운 관점을 제공하며, 향후 실제 응용 가능성에 대한 기대를 모으고 있습니다.



### Don't Just Demo, Teach Me the Principles: A Principle-Based Multi-Agent Prompting Strategy for Text Classification (https://arxiv.org/abs/2502.07165)
Comments:
          To be published in AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration

- **What's New**: 본 논문에서는 PRINCIPLE-BASED PROMPTING이라는 새로운 멀티 에이전트 프롬프트 전략을 소개합니다. 이 방법은 여러 LLM 에이전트가 독립적으로 샘플 분석을 기반으로 후보 원칙을 생성하고, 이를 최종화하는 에이전트를 통해 통합하여 분류 작업에 적용하는 방식입니다. 이 접근법은 기존의 제로샷 프롬프팅보다 1.55%에서 19.37%의 성능 향상을 보였으며, CoT 및 stepback prompting과 같은 강력한 비교군보다 우수한 결과를 보여주었습니다.

- **Technical Details**: PRINCIPLE-BASED PROMPTING은 멀티 에이전트 협업 구조를 활용하여 각 분류 작업을 위한 원칙을 자동 생성합니다. 첫째, 여러 LLM 에이전트가 라벨이 있거나 없는 데모로부터 후보 원칙을 생성하며, 이후 중앙 에이전트가 이 후보 원칙을 최종 선택하여 하위 분류 작업에 활용합니다. 이러한 접근 방식은 짧은 입력 길이로도 높은 분류 성능을 보장하며, 특히 레이블 정보와 LLM의 협력적 구조가 중요한 역할을 합니다.

- **Performance Highlights**: 우리는 두 개의 LLM(flann-t5-xxl, flan-ul2)에 대해 세 개의 공개 데이터셋과 두 개의 비공식 데이터셋에서 광범위한 실험을 수행했습니다. 결과적으로, 본 접근법은 제로샷 ICL 성능을 크게 향상시키며, 자동 생성된 SOPs가 인간이 생성한 SOPs보다 우수한 분류 성능을 보였습니다. 또한, 우리의 멀티 에이전트 방법론은 저리소스 환경에서도 fine-tuned RoBERTa-large보다 상당한 성능 향상을 나타냈습니다.



### Does Training on Synthetic Data Make Models Less Robust? (https://arxiv.org/abs/2502.07164)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서 합성 데이터(synthetic data)를 사용한 훈련의 영향을 조사합니다. 특히, 동일하거나 유사한 LLM로 생성된 합성 데이터가 기존에 모델이 가진 "블라인드스팟(blindspot)"을 악화시킬 수 있는지에 대한 질문을 다룹니다. NLI(자연어 추론) 작업을 통해 실험을 수행하여, 합성 데이터가 성능 변화에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구에서는 먼저, T𝑇Titalic_T라는 작업 모델과 G𝐺Gitalic_G라는 생성 모델을 가정합니다. NLI 작업에 집중하여, 합성 데이터가 특정 상황에서 어떻게 모델의 성능에 영향을 미치는지를 평가했습니다. 연구는 MultiNLI 데이터셋과 HANS 테스트셋을 사용하여, 모델이 슈퍼피셜 구문적 속성에 대해서만 판단하는 경향을 조사했습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터를 사용한 미세 조정(fine-tuning)이 NLI 모델의 성능을 HANS 검사 세트에서 예상했던 것처럼 감소시키지 않음을 발견했습니다. 합성 데이터가 모델의 성능에 미치는 영향은 일관되지 않았으며 다양한 설정에서 흥미로운 성능 변화를 관찰했습니다. 이로 인해 합성 훈련 데이터와 관련된 기존 가설이 완전히 지지받지 않는 것을 발견했습니다.



### Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning (https://arxiv.org/abs/2502.07143)
- **What's New**: 이 논문에서는 "Ask Patients with Patience (APP)"라는 새로운 시스템을 도입하여 온라인 의료 상담에서 LLM(large language models)의 진단 능력을 개선합니다. APP는 다중 턴 대화를 통해 질병 진단을 반복적으로 개선할 수 있도록 설계되었습니다. 이 시스템은 의료 가이드라인과 엔트로피 최소화를 통합하여 진단의 정확성과 효율성을 높입니다. 나아가, 사용자 친화적인 소통을 통해 의학 용어와 사용자 이해 간의 격차를 줄여주어 접근성과 참여를 극대화합니다.

- **Technical Details**: APP는 사용자가 제공하는 정보를 기반으로 다수의 질문을 통해 진단을 정교하게 다듬는 방식으로 작동합니다. 초기 입력을 반영하여 이후 질문을 구성하므로 진단의 정확성을 높이고 불확실성을 줄입니다. 대화의 각 턴에서 사용자의 반응에 따라 가능한 질병에 대한 확률 분포를 업데이트하고, 다음 질문을 의료 지침에 따라 작성함으로써 관련성을 보장합니다. 이 과정에서 질병 세트 D를 정의하고, 목표는 최소한의 상호작용으로 가장 그럴듯한 진단 d∗를 찾는 것입니다.

- **Performance Highlights**: APP는 ReMeDi 데이터셋의 하위 집합을 사용한 평가에서 기존의 단일 턴 및 전통적인 다중 턴 모델보다 높은 진단 예측 유사도를 기록했습니다. 그리고 진단 불확실성을 반복적으로 감소시키며 더 빠른 속도로 확신을 증가시켰습니다. 사용자 접근성과 공감 능력 면에서도 우수한 성과를 보여주어, 복잡한 의학용어와 사용자 이해 사이의 격차를 효과적으로 메꾸었습니다.



### Language-TPP: Integrating Temporal Point Processes with Language Models for Event Analysis (https://arxiv.org/abs/2502.07139)
- **What's New**: 이 논문에서는 Temporal Point Processes (TPPs)와 Large Language Models (LLMs)를 통합한 새로운 프레임워크인 Language-TPP를 제안합니다. 기존 TPP 모델은 복잡한 텍스트 이벤트 설명을 효과적으로 처리하는 데 어려움을 겪었으나, Language-TPP는 전문화된 byte-token을 사용해 연속 시간 간격을 변환하여 표준 LLM 아키텍처와 쉽게 통합할 수 있게 합니다. 이를 통해 여러 TPP 작업에서 최신 성능을 달성하고, 생성된 이벤트 설명의 품질 또한 크게 향상되었습니다.

- **Technical Details**: Language-TPP는 이벤트 시간 예측, 유형 예측 및 강도 추정과 같은 다양한 TPP 작업에 대한 확장성을 제공합니다. 우리는 이벤트 유형과 설명을 텍스트 정보로 모델링하고, 지속적인 시간 간격을 불연속적인 byte-token으로 변환하는 새로운 시간 인코딩 접근 방식을 도입합니다. 이를 통해 LLM에 이벤트 정보를 간단하게 제공하고, Qwen2.5와 같은 최신 LLM 아키텍처와의 통합이 용이해집니다.

- **Performance Highlights**: 실험 결과, Language-TPP는 TPP 데이터셋에서 최신 성능을 보이며, 특히 텍스트 이벤트 설명 생성에서 두드러진 성과를 나타냈습니다. 이는 기존의 LAMP 모델과 비교했을 때도 높은 우수성을 기록하였으며, 특히 시간 정보를 추가했을 때 이벤트 설명 생성의 품질이 현저히 개선됨을 확인했습니다. 이러한 성과는 TPP 문헌에서 새롭게 탐구된 방향으로, 기존 연구의 한계를 극복하는 데 기여하고 있습니다.



### TWICE: What Advantages Can Low-Resource Domain-Specific Embedding Model Bring? - A Case Study on Korea Financial Texts (https://arxiv.org/abs/2502.07131)
Comments:
          Submitted to ICLR@Financial AI

- **What's New**: 이 연구에서는 KorFinMTEB라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 한국 금융 도메인에 맞추어 특별히 설계되어 문화적 특성이 반영된 데이터 세트를 제공합니다. 기존의 영어 벤치마크를 단순히 번역하는 접근 방식의 한계를 설명하며, 저자들은 고유한 언어적 특성을 가진 벤치마크의 필요성을 강조합니다.

- **Technical Details**: KorFinMTEB는 7개의 핵심 작업인 분류(classification), 클러스터링(clustering), 검색(retrieval), 요약(summarization), 쌍 분류(pair classification), 재순위(re-ranking), 의미적 텍스트 유사성(semantic textual similarity)를 포함합니다. 각 작업은 공개 데이터와 자체 구축 데이터를 결합하여, 한국의 금융 텍스트 분석에서 발생하는 언어적 및 도메인 특유의 도전 과제를 만나는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 시판 중인 임베딩 모델들이 번역된 FinMTEB에서는 좋은 성능을 보였으나, KorFinMTEB에서는 그 성능이 다소 저조하게 나타났습니다. 이는 원래의 한국 금융 텍스트의 문화적 뉘앙스를 반영한 모델 평가의 필요성이 더욱 부각된다는 점을 시사합니다. 해당 연구는 저자들이 제안하는 KorFinMTEB가 낮은 자원 환경에서도 효과적인 임베딩 모델 평가를 통해 진전을 이끌어 낸다는 것을 강조합니다.



### Cardiverse: Harnessing LLMs for Novel Card Game Prototyping (https://arxiv.org/abs/2502.07128)
Comments:
          13 pages, 7 figures, 3 tables

- **What's New**: 이번 연구는 카드 게임의 프로토타입 제작을 자동화할 수 있는 포괄적인 프레임워크를 소개합니다. Large Language Models (LLMs)의 혁신적인 활용을 통해 새로운 게임 메커니즘을 설계하고, 일관된 게임 환경을 생성하는 데 있어 기존 데이터베이스의 한계를 극복하고자 합니다. 이는 게임 개발자들이 게임을 더욱 쉽게 제작할 수 있도록 도와줄 것입니다.

- **Technical Details**: 연구에서는 새로운 게임 디자인을 생성하기 위해 그래프 기반 인덱싱 방법을 제안합니다. 또한, LLM이 주축이 된 시스템을 통해 게임 스포츠 기록을 바탕으로 일관된 게임 코드를 생성하며, LLM이 생성한 액션-밸류 함수들의 앙상블을 활용한 게임플레이 AI 구축 방법을 설명합니다. 이러한 기술적인 통합은 전체 프로토타입 제작 과정을 혁신적으로 변화시킬 것으로 기대됩니다.

- **Performance Highlights**: 이번 연구의 프레임워크는 카드 게임 프로토타입 제작 과정을 가속화하고, 인간의 노력을 줄이며, 게임 개발자들에게 진입 장벽을 낮추는 것을 목표로 하고 있습니다. 실험 결과, 제시된 방법은 기존의 수작업보다 더 효율적이고 일관된 결과를 도출하여 카드 게임 설계 및 평가에 효율성을 더하고 있습니다.



### Structural Reformation of Large Language Model Neuron Encapsulation for Divergent Information Aggregation (https://arxiv.org/abs/2502.07124)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 효율적인 정보 집약을 위한 새로운 개념인 Neuron Encapsulation(신경 세포 캡슐화)을 소개합니다. 이 접근 방식은 모델의 내부 구조를 발전시켜 다양한 정보 스트림을 보다 효율적으로 통합하고 처리할 수 있도록 합니다. 연구 결과, 제안된 프레임워크는 전통적인 아키텍처에 비해 효율성과 효과성을 모두 향상시키는 것으로 나타났습니다.

- **Technical Details**: 제안된 신경 세포 캡슐화 프레임워크는 입력 데이터를 전문화된 처리 단위로 구조화하여 각 모듈이 독립된 변환 기능으로 작동하도록 설계되었습니다. 네트워크는 M개의 캡슐화된 모듈로 구성되며, 각 모듈은 특정 입력에 대해 적응형 통합 메커니즘을 통해 출력을 생성합니다. 이 구조는 다양한 데이터 소스를 효과적으로 관리하고 집약할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크를 적용한 모델은 perplexity (혼란도) 점수가 향상되었으며, 어휘 다양성 또한 증가했습니다. 이와 함께, 논리적 일관성을 평가한 결과 캡슐화된 뉴런들이 전문화된 처리 역할을 수행함을 확인했습니다. 결과적으로, 이 접근 방식은 다양한 언어 작업에서의 성능 향상에 기여하며, 데이터 처리를 위한 계산적 오버헤드 증가를 최소화하는 방향으로 작용합니다.



### SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation (https://arxiv.org/abs/2502.07101)
- **What's New**: 이 논문에서는 민감도(sensitivity) 계산을 위한 새로운 접근법인 SMAB(Sensitivity-based Multi-Armed Bandit) 프레임워크를 제안합니다. 기존의 방법들이 느린 속도로 계산되는 데 비해, SMAB는 단어 수준의 로컬(local) 및 글로벌(global) 민감도를 효율적으로 계산할 수 있습니다. 이 방법은 대규모 데이터셋에 적용할 수 있도록 설계되어 있어 실제 응용에서 유용성을 보여줍니다.

- **Technical Details**: SMAB 프레임워크는 Masked Language Modeling (MLM) 기법을 사용하여 각 단어의 민감도를 계산합니다. 이론적으로, 우리는 특정 텍스트 분류기에 대해 단어의 글로벌 민감도와 로컬 민감도를 정의합니다. 로컬 민감도는 특정 입력 텍스트에 대한 단어의 상대적 중요성을 평가하며, 글로벌 민감도는 전체 데이터셋 내 단어의 중요성을 파악하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, SMAB 프레임워크는 CHECKLIST를 사용한 사례 연구에서 고감도 및 저감도 단어를 효과적으로 식별하는 데 성공했습니다. 또한 금자 데이터(gold data)가 없는 상황에서도 민감도가 정확도의 대체 지표(proxy) 역할을 할 수 있음을 입증했습니다. 마지막으로, 민감도를 사용하여 적대적 예제 생성의 성공률을 15.58% 향상시켰으며, 적대적 패러프레이즈 생성에서는 SOTA(State-Of-The-Art) 방법보다 12% 개선된 결과를 보였습니다.



### "Once Upon a Time..." Literary Narrative Connectedness Progresses with Grade Level: Potential Impact on Reading Fluency and Literacy Skills (https://arxiv.org/abs/2502.07082)
Comments:
          14 pages, 1 figure

- **What's New**: 이번 연구는 아동의 독서 습관을 개발하는 데 있어 적절한 도서 선택의 중요성을 강조합니다. 특히, 아동 도서가 이야기의 복잡성에서 어떻게 차이가 나는지를 조사하여, 학교에서 사용되는 문학 텍스트의 내러티브 역학을 분석합니다. 1,627개의 문학 텍스트 데이터를 통해 교육의 다양한 연도에서의 복잡성 변화를 발견했습니다.

- **Technical Details**: 이 연구는 Word-Recurrence Graph Analysis(단어 반복 그래프 분석)를 사용하여 13년 동안의 교육 데이터를 검토했습니다. 연구 결과, 아동의 구술 내러티브에서 관찰되는 패턴을 반영하여, 특히 첫 3년의 학교 생활 동안 연결성이 기하급수적으로 증가한다는 것을 보였습니다. 이러한 발견은 아동 문학 텍스트가 문해력 기술 발달에 기여할 수 있는 잠재력을 시사합니다.

- **Performance Highlights**: 연구 결과는 아동의 이야기 생성 방식과 문학 텍스트 간의 중요한 유사성을 보여줍니다. 문학 텍스트의 복잡성이 학년이 올라가면서 어떻게 진화하는지를 밝혀냈으며, 이는 아동의 문해력 향상에 효과적인 자료가 될 수 있음을 나타냅니다. 이러한 성과는 교육 자료 선정의 새로운 기준을 제시할 수 있습니다.



### Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models (https://arxiv.org/abs/2502.07077)
- **What's New**: 최근 조사에 따르면 대형 언어 모델(LLMs) 사용자가 이러한 모델에 인격적인 특성을 부여하는 경향이 증가하고 있습니다. 이 논문은 LLM의 인격적 행동을 실제적이고 다양한 환경에서 평가하기 위한 새로운 방법론을 제시합니다. 연구에서는 14개의 인격적 행동에 대한 다중 턴 평가, 사용자 상호작용 시뮬레이션을 통해 자동화된 접근 방식과 대규모 인간 주제 연구를 통해 모델 행동을 검증하는 방식으로 나아갑니다. 이러한 연구는 인격적 모델 행동이 어떻게 설계 선택에 의해 영향을 받을 수 있는지에 대한 기초를 마련합니다.

- **Technical Details**: 연구에서는 기존의 단일 대화 기반 평가에서 벗어나, 14개의 인격적 행동을 세분화하여 평가하는 다중 턴 평가 방식을 적용하였습니다. 이 평가에는 Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT-4o, Mistral Large라는 네 가지 AI 시스템이 포함되었습니다. 총 120개의 기본 프롬프트를 이용하여 사용자의 인격적 행동을 유도하기 위한 대화를 시작하였으며, 결과적으로 모든 AI 시스템이 유사한 인격적 행동을 나타냈습니다. 특히, 인격적 행동은 사용자가 친구나 인생 코치로 AI를 사용할 때 가장 높은 빈도로 나타났습니다.

- **Performance Highlights**: 조사 결과, 모든 평가된 AI 시스템이 서로 유사한 인격적 행동을 보였고, 이는 주로 관계 구축과 1인칭 대명사 사용으로 특징지어졌습니다. 다중 턴 과정에서 50% 이상의 인격적 행동이 처음으로 나타나는 것을 발견하였습니다. 대규모 상호작용 실험(N=1101)을 통해 평가 결과가 AI 시스템에 대한 인간의 인격적 인식과 일치함을 확인하며, 이 평가 접근 방식은 인격적 행동이 인간-AI 상호작용에서 점점 더 중요해짐을 강조합니다.



### IRepair: An Intent-Aware Approach to Repair Data-Driven Errors in Large Language Models (https://arxiv.org/abs/2502.07072)
Comments:
          Accepted as full research paper at FSE'2025

- **What's New**: 최근 대규모 언어 모델(LLM)의 발전과 관련된 여러 가지 문제점들이 부각되고 있습니다. 이 논문에서는 IRepair라는 새로운 동적 슬라이싱 기반의 의도 인식 LLM 수리 전략을 제안합니다. 이 방법은 모델에서 오류가 발생하기 쉬운 특정 부분을 선택적으로 수리하여, 수리 과정에서 모델의 전반적인 성능 저하를 최소화하려고 합니다.

- **Technical Details**: IRepair는 가장 민감한 레이어를 동적으로 슬라이스하여 수리 노력을 집중시키는 기법입니다. 이 방법은 오류를 야기하는 변화에 대한 반응을 더 효과적으로 개선할 수 있으며, 모델의 다른 부분에 대한 변경은 최소화합니다. 구체적으로, Kullback-Leibler(KL) 발산 제약을 적용하여 오류가 발생하기 쉬운 섹션만 수리하는 방식을 채택하고 있습니다.

- **Performance Highlights**: IRepair는 GPT-2 및 GPT-Neo 모델에서 독성 완화 작업을 통해 43.6% 더 효과적으로 오류를 수리하면서도 일반 성능 저하를 46% 줄였습니다. 또한, 모델의 상위 20% 레이어에서 오류 밀도가 나머지 80%보다 773% 더 높다는 사실을 밝혀내어 선택적 수리의 필요성을 강조하고 있습니다. 이러한 결과는 IRepair의 동적 선택 접근 방식이 모델 전체의 오류를 효과적으로 해결하는 데 필수적임을 보여줍니다.



### Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations (https://arxiv.org/abs/2502.07068)
Comments:
          15 pages, 9 figures, accepted to NAACL 2025 main

- **What's New**: 이 논문은 대규모 설문조사 수행의 비용과 시간 문제를 해결하기 위해 언어 모델(LLMs)을 설계하여 설문 응답 분포를 시뮬레이션하는 새로운 방법론을 제안합니다. 기존의 접근법이 아닌 LLM을 특화하여 보다 정확한 응답 예측을 목표로 하고 있으며, 세계 가치 조사(World Values Survey) 데이터를 활용하여 이 방법의 유효성을 검증하고 있습니다. 또한, 이 방식은 기존 방법보다 유의미하게 더 나은 성능을 보여주며 새로운 조사에 대해서도 가능성을 내포하고 있습니다.

- **Technical Details**: 우리는 먼저 토큰 확률(first-token probabilities)을 기반으로한 세부 조정(fine-tuning) 방법을 개발하여 특정 질문에 대한 예측 응답 분포와 실제 응답 분포 간의 차이를 최소화하는데 초점을 맞췄습니다. 연구에서는 66개국의 응답 데이터를 사용하여 첫 번째 259개의 설문 질문을 세 가지 주제에 따라 분류하였습니다. 실험은 영어 및 중국어로 된 데이터셋을 사용하여 이루어졌으며, 언어적 차이에 대한 분석도 포함되었다는 점이 특징적입니다.

- **Performance Highlights**: 연구 결과, 제안된 세부 조정 방법은 보고된 기존의 방법들보다 더 높은 예측 정확도를 달성하였으며, 사용된 여러 LLM 모델 군에서도 일관되게 우수한 성능을 나타냈습니다. 그러나, 가장 높은 성능을 보인 모델조차도 일부 미지의 질문에 대해서는 어려움을 겪는 경향을 보였고, 모든 모델이 실제 인간 응답 데이터에 비해 국가별 예측의 다양성이 부족한 것으로 나타났습니다. 이를 통해, 현재 LLM을 설문 응답 분포 시뮬레이션에 활용하는 데에 주의가 필요함을 강조하고 있습니다.



### Using Contextually Aligned Online Reviews to Measure LLMs' Performance Disparities Across Language Varieties (https://arxiv.org/abs/2502.07058)
Comments:
          Accepted by 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), theme track

- **What's New**: 이번 연구는 다양한 언어의 양상에 따른 자연어 처리(NLP) 모델 성능을 평가하기 위한 새로운 접근 방식을 소개합니다. 기존의 대규모 언어 모델(LLM)을 사용할 때 특히 널리 사용되는 언어 양상에 대비하여, 비교적 적은 수의 사용자 리뷰를 기반으로 다국어 리뷰 데이터셋을 구축함으로써, 성능 격차를 분석하고자 합니다. 연구 결과, LLM이 대만 만다린(Taiwan Mandarin)에서는 일관되게 저조한 성과를 낸다는 점이 확인되었습니다.

- **Technical Details**: 이 연구에서는 Booking.com에서 수집된 호텔 리뷰 4,447,853 개로 구성된 데이터셋을 사용했습니다. 여기서 데이터는 대만 만다린과 본토 만다린 간의 유사한 리뷰를 기반으로 선별되었습니다. 각 리뷰는 제목, 긍정적 피드백, 부정적 피드백과 같은 기본 구성 요소를 포함하며, 리뷰 평점과 메타데이터가 추가되어 있습니다. 총 22,918개의 리뷰 쌍이 최종 데이터셋으로 선정되었으며, LLM의 성능은 성향 분석(sentiment analysis) 작업으로 평가되었습니다.

- **Performance Highlights**: 실험 결과, 다섯 명의 대만 만다린 사용자가 리뷰에 대한 글쓰기 품질을 평가한 결과, 대만 만다린 그룹은 평균 4.18의 점수를 기록했습니다. 반면, 본토 만다린 그룹은 평균 3.94의 점수로 나타났습니다. LLM 테스트에서 GPT-4o, Llama3, Gemma2 등 여섯 개 모델이 적용되었으며, 평가 결과 대만 만다린에 대한 성능이 저조한 경향을 보였습니다. 이러한 결과는 국제적인 리뷰 플랫폼의 데이터를 통해 언어 양상 간의 성능 격차를 평가하는 지속 가능한 방법을 제시합니다.



### Tokenization Standards for Linguistic Integrity: Turkish as a Benchmark (https://arxiv.org/abs/2502.07057)
- **What's New**: 이 논문에서는 NLP에서 tokenization의 중요성을 강조하며, morphologically rich한 저자원 언어에 적합한 새로운 평가 프레임워크를 제안합니다. 이 프레임워크는 6,200개의 터키어 문제를 기반으로 하여, tokenization 전략의 효과를 체계적으로 평가하기 위한 5가지 핵심 메트릭스를 사용합니다. 이 과정에서, 특히 %TR(특정 언어의 유효한 단어 비율)이 downstream 성능과의 강한 상관관계를 보인다는 점을 밝혔습니다.

- **Technical Details**: 토큰화의 품질을 평가하기 위해 제안된 두 가지 중요한 메트릭은 token purity와 language-specific token percentages입니다. Token purity는 생성된 토큰이 의미 있는 언어 단위에 얼마나 잘 부합하는지를 측정하며, 이는 불필요한 단어 파편화를 최소화합니다. 반면, language-specific token percentages는 생성된 토큰 중에서 유효한 언어 단어의 비율을 평가하며, 언어의 어휘와의 정합성을 측정합니다.

- **Performance Highlights**: 실험 결과, %TR 메트릭은 token purity보다 모델 정확도를 높이는 데 더 중요한 역할을 하며, 큰 모델 파라미터가 반드시 더 나은 tokenization 품질이나 향상된 결과를 보장하지 않는다는 점을 강조합니다. 이를 통해 저자원 언어에 최적화된 강력한 tokenization 방법 개발을 위한 새로운 기준을 세웠으며, 미래 연구에서는 형태소 분석 개선 및 도메인 특화 사용자 조정을 탐색할 것이라고 하였습니다.



### Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessmen (https://arxiv.org/abs/2502.07029)
Comments:
          Accepted to NAACL 2025. Codebase available at this https URL

- **What's New**: 이번 연구에서는 Allophony(음소 변이)를 모델링하기 위한 새롭고 혁신적인 접근법인 MixGoP를 제안합니다. MixGoP는 Gaussian mixture models(GMMs)를 활용하여 음소의 다중 하위 클러스터를 모델링함으로써 atypical pronunciation(비정상 발음) 평가의 필요성을 충족합니다. 이 방법은 frozen self-supervised speech model(S3M)을 통합하여 다양한 발음 변이를 효과적으로 캡처합니다.

- **Technical Details**: MixGoP는 기존의 Goodness of Pronunciation(발음의 질)를 개선하여 각 음소를 여러 개의 allophonic subclusters(음소의 하위 클러스터)로 모델링합니다. 기존 접근법은 음소를 단일 클러스터로 간주하고 atypical speech(비정상 음성)가 typical speech(정상 음성)와 동일한 분포에 있다고 가정하지만, MixGoP는 이러한 가정을 완화하여 보다 정교한 모델링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MixGoP는 총 5개의 데이터셋 중 4개에서 최첨단 성과를 달성하였으며, 이는 dysarthric(발음 장애) 및 비원어민 음성을 포함합니다. 또한 S3M 특징이 MFCCs 및 Mel spectrograms와 비교하여 allophonic variation(음소 변이)를 더 효과적으로 캡처함을 분석하여 MixGoP와 S3M의 통합의 장점을 강조합니다.



### AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements (https://arxiv.org/abs/2502.07022)
Comments:
          Camera ready. ICLR 2025

- **What's New**: 이 논문에서는 호주 현대 노예법에 따른 기업의 성명서를 평가하는 데 도움이 되는 신규 데이터셋을 소개합니다. 이 데이터셋은 총 5,731개의 현대 노예 성명서로 구성되어 있으며, 항목별로 주석 처리되었습니다. 현대 언어 모델 (LLMs)을 활용하여 기업의 성명서에서 구체적인 현대 노예 대응 조치를 인식하고 불분명한 주장과 구별하는 데 중점을 두고 있습니다.

- **Technical Details**: 데이터셋은 HDF5 및 Activeloop DeepLake 포맷으로 제공될 예정입니다. HDF5는 대량 데이터 처리에 유용한 포맷이며, Activeloop DeepLake는 머신러닝 실험에 최적화된 기능을 제공합니다. 이 데이터셋은 호주 현대 노예 등록부에 게시된 성명서와 함께 관련 메타데이터를 포함하고 있으며, 모든 처리된 성명서와 '골드' 검증 세트도 포함되어 있습니다.

- **Performance Highlights**: 데이터셋은 Figshare라는 오픈 액세스 저장소에 호스팅되어 연구 커뮤니티에 무료로 제공됩니다. 다양한 지침이 제공되서 데이터셋을 효과적으로 활용할 수 있도록 돕고, 출력의 정량적 분석을 위한 기계 학습 방법론이 제안되었습니다. 또한, 초기 데이터셋 릴리스에는 모든 주석 처리된 성명서와 금일 검증 세트가 포함되며, 모델 경쟁을 위해 골드 테스트 세트의 릴리스는 2025년까지 보류될 가능성이 있습니다.



### Finding Words Associated with DIF: Predicting Differential Item Functioning using LLMs and Explainable AI (https://arxiv.org/abs/2502.07017)
Comments:
          14 pages, 2 figures, 6 tables

- **What's New**: 이 연구에서는 다양한 encoder 기반 Transformer 대규모 언어 모델(LLM)을 미세 조정(fine-tuning)하고 비교하여 항목 텍스트로부터 차별 항목 기능성(differential item functioning, DIF)을 예측하는 방법을 제시합니다. 연구팀은 설명 가능한 인공지능(explainable artificial intelligence, XAI) 기법을 적용하여 DIF와 관련된 특정 단어를 식별했습니다. 이 과정에서 3학년부터 11학년까지의 학생들을 위한 영어와 수학 종합 주(State Assessment)를 위한 42,180개의 항목을 사용했습니다.

- **Technical Details**: 모델의 예측 성능은 8개의 초점(focal) 및 참조(reference) 그룹 쌍에 대해 $R^2$ 값이 .04에서 .32까지 다양했습니다. 연구 결과, DIF와 연관된 많은 단어가 설계에 따라 시험 청사진(test blueprint) 내에 포함된 작은 하위 도메인을 반영하고 있음을 나타냅니다. 이는 종종 DIF 항목에 대한 정성적 리뷰가 혼란스럽거나 불확실한 결과를 초래하는 이유를 설명하는 요소가 될 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 항목 작성 과정에서 DIF와 관련된 단어를 선별하여 즉각적인 수정이 가능하게 하며, 전통적인 DIF 분석 결과를 재검토할 때 텍스트에서 주요 단어를 강조하는 데 도움을 줄 수 있습니다. 이 연구의 확장은 특히 고품질 항목을 구축할 자원이 부족한 평가 프로그램의 공정성을 향상시킬 수 있으며, 전통적인 DIF 분석을 위한 충분한 샘플 사이즈가 없는 더 작은 하위 집단에서도 적용될 수 있습니다.



### Demystifying Singular Defects in Large Language Models (https://arxiv.org/abs/2502.07004)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)에서 고노름 토큰(high-norm tokens)의 근본 원인에 대한 새로운 분석 프레임워크를 제시하고 있습니다. 기존의 비전 트랜스포머(ViTs)에 대한 수학적 모델링을 기반으로 LLMs의 고노름 토큰의 특성을 분석하여, 이를 설명하는 여러 이론적 통찰과 실증적 검증을 진행했습니다.

- **Technical Details**: 연구 결과에 따르면, 레이어별 특이 방향(layer-wise singular direction)은 LLM에서 토큰 노름의 급격한 폭발을 예측하며, 레이어의 부정 고유값(negative eigenvalues)은 그 급작스러운 감소를 설명합니다. 또한 초깃값(initial) 및 비초깃값(noninitial) 토큰 간의 고노름 토큰을 형성하는 경로가 다르다는 점도 중요한 발견입니다.

- **Performance Highlights**: 이번 논문은 고노름 토큰이 해당 모듈을 근사하는 행렬의 올바른 주요 특이 벡터에 의해 촉발된다는 사실을 밝혀냈습니다. 이러한 결과는 양자화(quantization) 기법의 개선 및 LLM 서명(signature) 설계에 적용될 수 있는 두 가지 실용적인 사례를 포함하고 있으며, LLM의 내부 메커니즘에 대한 추가 연구를 촉진할 것으로 기대하고 있습니다.



### Investigating the Zone of Proximal Development of Language Models for In-Context Learning (https://arxiv.org/abs/2502.06990)
Comments:
          NAACL 2025 findings

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 문맥 내 학습(in-context learning, ICL) 행동을 분석하기 위한 학습 분석 프레임워크를 소개합니다. 기존 교육 심리학 이론인 근접 발달 영역(zone of proximal development, ZPD)의 관점에서 ICL을 이해하도록 이론을 적용하였습니다. ICL을 통해 모델 성능을 평가하고 ZPD를 측정할 수 있는 방법을 개발했습니다.

- **Technical Details**: 우리는 ICL의 ZPD를 예시별로 평가하기 위해 개별 성과를 분석하며, 이를 기반으로 항목 반응 이론(item response theory, IRT) 모델을 제안합니다. 이 모델은 LLMs의 ZPD 분포를 예측하고, 이를 통해 ICL의 효과적인 활용 방안을 제공합니다. 또한, 우리는 모델의 근접 발달 영역에 해당하는 예시를 중심으로 한 인공지능 교육 커리큘럼을 제안합니다.

- **Performance Highlights**: 제안된 프레임워크는 LLM이 추론과 미세 조정(fine-tuning) 상황 모두에서 성능을 향상시킬 수 있도록 합니다. 특히, ICL의 효과를 극대화하기 위해 예시 선택을 최적화하여 성능과 추론 비용의 균형을 개선했습니다. 우리의 커리큘럼 접근법은 모델의 ZPD 내 예시를 우선하며, 이는 훈련 역학 분석을 통해 성능 개선의 근거를 제공합니다.



### Multi-Agent Simulator Drives Language Models for Legal Intensive Interaction (https://arxiv.org/abs/2502.06882)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문은 법률 분야에서 상호작용적 법률 시나리오를 시뮬레이션하여 합성 데이터를 생성하는 Multi-agent Legal Simulation Driver (MASER)를 소개합니다. 이 시스템은 실제 법률 사건 소스를 활용하여 참가자 간의 법률 속성 일관성을 보장하며, 비Distractor 행동을 관리하기 위한 감독 메커니즘을 도입합니다. 또한, 이 논문은 동적 법률 시나리오에서 LLM의 성과를 평가하기 위해 Multi-stage Interactive Legal Evaluation (MILE) 벤치마크도 개발하였습니다.

- **Technical Details**: MASER는 고객(Client), 변호사(Lawyer), 감독(Supervisor)이라는 세 가지 에이전트로 구성되어 법적 목표(예: 고소장 작성)를 달성하는 방식으로 작동합니다. 각 캐릭터는 고유의 역할과 책임을 가지며, Big-5 성격 특성을 바탕으로 다양한 개인적 특징이 설정됩니다. 변호사는 고객의 법적 요구를 충족시키기 위해 사례 분석 및 관련 법률을 활용해야 하며, 감독은 상호작용을 모니터링하여 참가자 간의 행동 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, MASER는 기존의 법률 업무에서 LLM의 성능을 현저히 향상시켰습니다. 훈련된 모델은 GPT-4o와 같은 독점적인 고급 LLM 및 전문 법률 LLM보다 더 우수한 성능을 보여주었습니다. 연구진은 이 프레임워크가 복잡하고 사적인 법률 시나리오에서의 고급 상호작용 및 목표 달성을 위한 일종의 일반 패러다임으로 작용할 것으로 기대하고 있습니다.



### Mix Data or Merge Models? Balancing the Helpfulness, Honesty, and Harmlessness of Large Language Model via Model Merging (https://arxiv.org/abs/2502.06876)
Comments:
          Under Review

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 조화로운 정렬을 달성하기 위해 Helpfulness, Honesty, and Harmlessness(3H 최적화)를 중점으로 두고, 기존 방법의 한계를 넘어 모델 병합(model merging) 기법을 탐구합니다. 특히, 15가지 방법을 평가하여 모델 병합이 데이터 혼합(data mixture) 접근법보다 더 나은 성능을 보인다는 점을 입증했습니다. 또한, R-TSVM이라는 새로운 방법론을 제안하여 파라미터의 중복 및 이상치를 고려한 최적화를 통해 LLM 정렬을 향상시키는 방안을 제시합니다.

- **Technical Details**: 이 연구는 12개의 훈련 없이 병합할 수 있는 방법과 3개의 데이터 혼합 기법을 포함한 총 15가지 방법을 사용하여 10개의 데이터 세트를 평가했습니다. 세부적으로 3H 차원 간의 협업/충돌 관계에 대한 분석을 포함하여, 파라미터 수준의 충돌 해결을 통해 효과적인 3H 최적화를 달성하기 위한 조건을 제시합니다. 또한, R-TSVM 방법은 이상치 인식 파라미터 가중치와 희소성 적응 랭크 선택 전략을 통합하여 기존 방법의 한계를 극복할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델 병합 기법은 데이터 혼합 접근 방식에 비해 다양한 3H 목표 간의 균형을 효과적으로 유지하는 데 유리하다는 점이 확인되었습니다. 특히, 중복 부품 제거 및 이상치 완화를 통한 파라미터 수준에서의 충돌 해결이 매우 중요한 역할을 한다는 사실이 드러났습니다. 이러한 요소들은 LLM 정렬을 보다 안정적이고 효과적으로 만드는 데 기여하며, 실험을 통해 제안된 전략의 효용성이 입증됩니다.



### Group Reasoning Emission Estimation Networks (https://arxiv.org/abs/2502.06874)
- **What's New**: 이 논문에서는 GREEN(Group Reasoning Emission Estimation Networks)을 소개하여 온실가스(GHG) 배출량 추정을 위한 새로운 AI 기반 프레임워크를 제안하고 있습니다. 특히 이 연구는 20,850개 기업의 분야 분류를 자동화하고, 대규모 벤치마크 데이터세트를 구축하여 신뢰성 있는 배출량 추정 방법을 제공합니다. 또한, 대규모 데이터와 대규모 언어 모델(LLM)을 활용하여 배출량 예측의 정확성을 높이는 방안을 제시합니다.

- **Technical Details**: GREEN은 전통적인 전문가 기반 분류 방식 대신 정보 검색 문제로서 분야 분류를 재구성하고, Sentence-BERT 모델을 이용해 자가 감독적 대조 학습(self-supervised contrastive learning) 방법을 적용합니다. 이 프레임워크는 자연어 처리(NLP) 기술을 활용하여 기업당 연간 수익과 탄소 강도 계수를 곱하여 배출 예측을 수행하며, 이를 통해 높은 정확도의 분류 성능을 달성하였습니다.

- **Performance Highlights**: 실험 결과, 이 연구는 1,114개의 산업 카테고리에서 83.68%의 Top-1 정확도와 91.47%의 Top-10 정확도를 기록하였으며, 20개 기업에 대한 사례 연구에서 평균 절대 백분율 오차(MAPE)는 45.88%로 나타났습니다. 이는 대기업뿐만 아니라 중소기업(SMEs)에서도 효과적으로 활용할 수 있는 신뢰할 수 있는 배출량 추정 방법을 제공한다는 점에서 중요한 의미를 갖습니다.



### Multimodal Cognitive Reframing Therapy via Multi-hop Psychotherapeutic Reasoning (https://arxiv.org/abs/2502.06873)
Comments:
          NAACL 2025 Main

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 인지 재구성(cognitive reframing) 치료에서 잠재력을 보여주었지만, 비언어적 증거(non-verbal evidence)의 중요성이 간과되고 있음을 지적합니다. 따라서 연구진은 비주얼 단서를 통합한 다중 양식(multimodality)의 인지 재구성 접근 방식을 제안합니다. 이를 위해 새로운 데이터셋인 Multi Modal-Cognitive Support Conversation (M2CoSC)를 생성하여 GPT-4가 생성한 대화와 가상 클라이언트의 얼굴 표정을 짝지었습니다.

- **Technical Details**: M2CoSC 데이터셋은 심리 치료 세션을 시뮬레이션하기 위해 GPT-4의 역할 수행 능력을 활용하여 생성되었습니다. 연구진은 인지 재구성의 단계를 기반으로 세 가지 단계(소개, 문제 탐색, 브레인스토밍, 제안)를 확장하여 다중 양식 심리 치료 프레임워크를 수립했습니다. 다중 홉 심리 치료 추론(multi-hop psychotherapeutic reasoning) 방법을 도입하여 클라이언트의 상태를 이해하고 이에 기반한 더 합리적이고 공감가는 제안을 제공할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과 M2CoSC 데이터셋으로 훈련된 VLMs의 심리 상담 능력이 기존 LLMs를 초월하여 유의미하게 향상된 것으로 나타났습니다. 또한, 다중 홉 심리 치료 추론 방법은 VLMs가 더 thoughtful하고 공감하는 제안을 제공할 수 있게 하여 표준 프롬프트(prompting) 방법보다 성능이 우수하다는 것을 보여주었습니다. 이러한 연구는 AI 강화 심리 치료의 효율성을 높이기 위한 비언어적 단서의 활용 가능성을 열어줍니다.



### Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey (https://arxiv.org/abs/2502.06872)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 확보하기 위한 포괄적인 로드맵을 제공하는 것을 목표로 합니다. RAG는 외부 정보를 통합하여 인공지능 생성 콘텐츠(AIGC)의 문제를 해결하려는 접근 방식으로, 현재 이 시스템들이 직면한 여러 리스크를 다룹니다. 특히, 신뢰성, 프라이버시, 안전, 공정성, 설명 가능성, 책임의 다섯 가지 주요 관점을 바탕으로 이루어지는 연구 방향을 제시합니다.

- **Technical Details**: RAG 프레임워크는 정보 검색(information retrieval), 지식 증대(knowledge augmentation), 콘텐츠 생성(content generation)의 세 가지 주요 단계를 포함합니다. 정보 검색 과정에서 쿼리를 기반으로 관련 정보를 제공하며, 지식 증대 단계에서 검색된 지식은 언어 모델에 통합됩니다. 이러한 과정에서 발생할 수 있는 신뢰성 문제, 예를 들어 검색 편향(retrieval bias)과 허위 정보(hallucination) 문제에 대한 해결책이 필수적입니다.

- **Performance Highlights**: RAG 시스템은 의료 질문 응답, 법률 문서 작성, 교육용 챗봇 및 금융 보고서 요약 등 다양한 영역에서 효과적으로 적용되고 있습니다. 그러나 이러한 시스템의 신뢰성 문제는 특히 고위험 분야에서의 채택을 제한하여, 연구자들은 이러한 시스템이 신뢰할 수 있도록 다양한 방법을 개발하고 있습니다. 궁극적으로 RAG 시스템의 신뢰성 확보는 실제 적용에 큰 영향을 미친다는 점에서 중요한 문제임을 강조합니다.



### Related Knowledge Perturbation Matters: Rethinking Multiple Pieces of Knowledge Editing in Same-Subjec (https://arxiv.org/abs/2502.06868)
Comments:
          Accepted by NAACL 2025

- **What's New**: 본 연구에서는 Same-Subject Editing에 초점을 맞추어, 하나의 주체에 대한 여러 속성을 동시에 수정하여 일관되고 포괄적인 지식 업데이트를 목표로 합니다. 이를 위해 기존 벤치마크에서 동일 주체에 대한 충분한 편집 데이터를 찾지 못했음을 인지하고 새로운 S²RKE(같은 주체 관련 지식 편집) 벤치마크를 도입하였습니다. 이 벤치마크는 동일 주체에 대한 다수의 관련 편집을 연결합니다.

- **Technical Details**: Large language models (LLMs)은 주어진 토큰 시퀀스를 처리하며, Transformer 아키텍처에서 각 토큰은 히든 상태 벡터로 포매팅됩니다. 연구에서는 전통적으로 지식이 트리플 형태(s,r,o)로 표현되며, 각각 주체(subject), 관계(relation), 객체(object)를 나타냅니다. 본 연구에서는 여러 관련 지식을 함께 수정하는 Same-Subject Editing의 필요성을 강조하며, 특히 기존 편집 방법들이 이러한 작업에 비효율적임을 발견했습니다.

- **Performance Highlights**: 실험 결과, ROME 및 MEMIT과 같은 일상적인 locate-then-edit 방법은 동일 주체에 대해 여러 관련 정보를 효과적으로 업데이트하는 데 실패하였습니다. 이는 후속 편집이 이전 편집에 간섭하는 'related knowledge perturbation' 현상에 기인하며, 이 문제를 해결하기 위한 추가적인 연구가 필요합니다. 궁극적으로, 본 연구는 LLM에서의 Same-Subject Editing의 가능성을 제시하고 관련 연구의 방향성을 제안합니다.



### Forbidden Science: Dual-Use AI Challenge Benchmark and Scientific Refusal Tests (https://arxiv.org/abs/2502.06867)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 안전성을 평가하기 위한 오픈소스 데이터셋과 테스트 프레임워크를 개발하였습니다. 이 데이터셋은 유해한 콘텐츠의 적절한 거부와 합법적인 과학적 담론의 과도한 제한 여부를 측정하는 데 중점을 두고 있습니다. 또한, 다양한 프롬프트에 대한 네 가지 주요 모델의 응답을 분석하여 각 모델의 안전 프로파일을 분리해 보여주었습니다.

- **Technical Details**: 연구는 주로 제어된 물질 쿼리에 대한 네 가지 모델의 응답을 체계적으로 변형된 프롬프트에 따라 분석하였습니다. Claude-3.5-sonnet는 73% 거부율과 27% 허용률을 보이며 가장 보수적인 접근 방식을 보인 반면, Mistral은 100%의 쿼리에 대해 답변을 시도했습니다. GPT-3.5-turbo는 10%의 거부와 90% 허용을 보여 중간 정도의 제한을 보였으며, Grok-2는 20% 거부와 80% 허용을 기록하였습니다.

- **Performance Highlights**: 테스트 중 프롬프트 변동 전략을 분석한 결과, 단일 프롬프트에서 85%의 응답 일관성이 있었으나 다섯 개 변형을 사용했을 때는 65%로 감소하는 경향이 있었습니다. 공개된 이 벤치마크는 필요한 안전 제한과 합법적인 과학적 탐구의 과도한 검열 간의 비판적인 균형을 체계적으로 평가할 수 있게 합니다. 또한, 안전 메커니즘의 잠재적인 취약성을 드러내는 체인-오브-생각 분석을 통해, 바람직하고 유효한 과학적 담론을 지나치게 제한하지 않으면서 강력한 방어책을 구현하는 복잡성을 강조하였습니다.



### Knowledge Graph-Guided Retrieval Augmented Generation (https://arxiv.org/abs/2502.06864)
Comments:
          Accepted in the 2025 Annual Conference of the Nations of the Americas Chapter of the ACL (NAACL 2025)

- **What's New**: 본 논문에서는 지식 그래프(Knowledge Graph) 기반의 검색 증강 생성(Knowledge Graph-Guided Retrieval Augmented Generation, KG$^2$RAG) 프레임워크를 제안하여 기존의 검색 증강 생성 방식의 한계를 극복하려고 합니다. KG$^2$RAG는 지식 그래프를 활용하여 정보 조각 사이의 사실 수준 관계를 파악하고, 이에 기반한 다양한 정보 검색을 실현합니다. 이를 통해 더 다양한 정보 조각을 확보하고, 조화로운 응답 생성을 촉진합니다.

- **Technical Details**: KG$^2$RAG의 작동 과정에는 두 가지 주요 단계가 있습니다. 첫째, 감정 기반 검색을 통해 시드 조각(seed chunks)을 생성한 후, 이러한 조각을 확장하는 과정이 포함됩니다. 둘째, KG 기반의 맥락 조직(context organization) 단계에서 관련 정보를 필터링하고 내부적으로 일관된 단락으로 조정하여 LLM에 입력됩니다.

- **Performance Highlights**: HotpotQA 데이터세트를 사용한 광범위한 실험 결과, KG$^2$RAG는 기존의 검색 증강 생성 방식에 비해 응답 품질과 검색 품질 모두에서 우수한 성능을 보여주었습니다. 또한, ablation study를 통해 KG$^2$RAG의 다양한 모듈의 효과를 강조하였습니다. 개발된 데이터셋과 소스 코드는 GitHub에 공개되어, KG의 RAG 적용을 촉진합니다.



### LLM-Supported Natural Language to Bash Translation (https://arxiv.org/abs/2502.06858)
Comments:
          13 pages, NAACL 2025

- **What's New**: 이 논문에서는 Bash 명령어를 자연어로 번역하기 위한 NL2SH 모델의 성능 평가에 필요한 새로운 테스트 데이터셋과 기능 동등성 휴리스틱을 제시합니다. 수동으로 검증된 600개의 지시-명령 쌍 및 40,939 쌍의 훈련 데이터셋은 기존 데이터셋에 비해 크기를 각각 441%와 135% 증가시켰습니다. 또한, 우리는 LLM의 명령 출력 평가를 결합한 새로운 기능 동등성 휴리스틱을 도입하여 두 Bash 명령의 기능적 동등성을 95% 확률로 평가할 수 있습니다.

- **Technical Details**: NL2SH 번역은 자동으로 텍스트나 음성을 다른 언어로 변환하는 기계 번역 범주에 해당합니다. LLM을 활용하여 생성된 Bash 명령어의 기능적 정확성을 평가하는 것은 필수적이며, 이는 입력에 대한 올바른 출력을 생산하는지를 비교함으로써 판단됩니다. 코드의 기능적 정확성을 보장하는 것은 복잡한 프로세스이며, 정적(static) 및 동적(dynamic) 분석 기법이 주로 사용됩니다.

- **Performance Highlights**: 우리는 인기 있는 LLM을 테스트 데이터셋과 새로운 휴리스틱을 이용하여 평가하였고, parsing, in-context learning, in-weight learning, 그리고 constrained decoding 기법들이 NL2SH 정확성을 최대 32%까지 개선할 수 있음을 증명했습니다. 이러한 발견은 데이터셋의 품질, 실행 기반 평가, 번역 방법이 NL2SH 번역 발전에 필수적임을 강조합니다.



### Self-Supervised Prompt Optimization (https://arxiv.org/abs/2502.06855)
- **What's New**: 본 논문에서는 Self-Supervised Prompt Optimization (SPO)이라는 새로운 프레임워크를 제안합니다. SPO는 외부 참조 없이도 효과적인 프롬프트를 발견할 수 있는 비용 효율적인 방법론으로, 고정형 및 개방형 작업에 모두 적용 가능합니다. 기존의 프롬프트 최적화 방법들과 달리, SPO는 실험적 조정이나 전문가의 도움이 필요하지 않습니다.

- **Technical Details**: SPO는 LLM(대형 언어 모델)의 출력을 비교하여 평가 및 최적화 신호를 추출합니다. 이 방법은 LLM 평가자가 수행한 쌍별 출력 비교를 통해 우수한 프롬프트를 선택하고, 이후 LLM 최적화기가 출력과 작업 요구 사항을 일치시키는 방식으로 진행됩니다. 이를 통해 SPO는 직접적인 출력 비교를 사용하여 고품질의 프롬프트를 생성합니다.

- **Performance Highlights**: 광범위한 실험 결과 SPO가 최첨단 프롬프트 최적화 방법들을 능가하는 것으로 나타났습니다. 구체적으로 SPO는 기존 방법에 비해 1.1%에서 5.6%까지 비용을 절감하며, 샘플 수 또한 크게 줄이자는 (예: 세 개의 샘플) 결과를 보였습니다. 이로써 SPO는 낮은 비용으로 경쟁력 있는 성능을 달성할 수 있음을 입증하였습니다.



### Survey on Vision-Language-Action Models (https://arxiv.org/abs/2502.06851)
- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델에 대한 AI 생성 리뷰를 소개하며, 주요 방법론, 발견 및 향후 방향을 요약합니다. 이 내용은 대형 언어 모델(LLMs)을 사용해 생성되었으며, 과학 문헌 리뷰 자동화의 가능성을 강조합니다. AI가 생성한 콘텐츠의 정확성과 신뢰성을 보장하는 것은 여전히 도전 과제가 되고 있으며, 향후 연구는 AI 지원 문헌 리뷰를 위한 구조적 프레임워크 개발에 중점을 둘 예정입니다.

- **Technical Details**: 이 연구에서 소개된 Actra는 엣지 디바이스에서 머신 러닝 모델을 배치하기 위한 혁신적인 접근 방식입니다. Actra는 경로 주의(trajectory attention)와 학습 가능한 액션 쿼리(learnable action queries)를 결합하여 로봇 작업의 추론 과정을 최적화합니다. 또한, ProSim-Instruct-520k라는 대규모 다중 모달 데이터셋을 개발하여 로봇 조작 시스템의 학습 능력을 향상시키고 있으며, 이 데이터셋은 520,000개 이상의 실제 주행 시나리오와 1,000만 개 이상의 텍스트 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: Actra는 다양한 로봇 환경에서 실험을 거쳐 성능 및 효율성을 크게 향상시켰음을 보여주었습니다. 특히 복잡한 다중 모달 작업 처리에 효과적이며, 여러 작업 및 객체 카테고리에서 우수한 성공률을 기록했습니다. 결과적으로, Actra는 로봇 조작에 있어 더 효율적이고 강력한 추론 방법을 제공하며, 앞으로의 연구를 위한 대규모 자원을 마련함으로써 데이터 기반 로봇 모델 개선에 기여할 수 있습니다.



### DarwinLM: Evolutionary Structured Pruning of Large Language Models (https://arxiv.org/abs/2502.07780)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 구조적 가지치기(structured pruning) 기법을 개발하여 효율적인 모델 압축과 성능 향상을 목표로 하고 있습니다. 특히, 기존의 가지치기 방법들이 각 레이어를 균일하게 다루던 것과 달리, 다양한 레이어가 가지치기에 대해 가지는 상이한 민감도를 활용한 비균일(non-uniform) 모델 압축을 제안합니다. 본 연구의 주된 기법인 \\texttt{DarwinLM}은 진화적 탐색(evolutionary search) 과정에 기반을 두고 있어, 각 세대에서 여러 개의 자손 모델을 생성하고, 성능이 가장 우수한 모델을 선택하여 압축 후 훈련(post-compression training)을 수행합니다.

- **Technical Details**: \\texttt{DarwinLM}은 부모 모델로부터 시작하여 레이어 간의 희소성을 이동시키는 수준 전환 돌연변이(level switch mutation)를 통해 자손 모델을 생성합니다. 이 과정에서는 또한 경량의 다단계 훈련 프로세스를 포함하여 성능을 점진적으로 향상시키고, 각 선택 단계에서 성능 저하 모델을 제거합니다. 이 방법을 통해 Llama-2-7B, Llama-3.1-8B 및 Qwen-2.5-14B-Instruct와 같은 모델에 대해 최첨단 성능을 달성하며, 특히 훈련 데이터를 5배 줄이면서도 ShearedLlama보다 뛰어난 결과를 보입니다.

- **Performance Highlights**: 실험 결과, \\texttt{DarwinLM}은 Llama-2-7B를 2.7B 파라미터로 압축하며, 성능 저하 없이 기존의 최고 성능 모델들을 능가하는 효과를 보였습니다. 또한, Llama-3.1에서 4.6B 파라미터로 가지치기된 모델은 250배 더 많은 데이터로 훈련된 OLMO 모델과 비교해도 높은 정확도를 자랑합니다. 이러한 결과들은 \\texttt{DarwinLM}이 비균일 구조적 가지치기 기법으로 우수한 성능을 발휘할 수 있음을 만족스럽게 입증하고 있습니다.



### Economics of Sourcing Human Data (https://arxiv.org/abs/2502.07732)
- **What's New**: AI의 발전은 인간이 생성한 데이터에 크게 의존해왔으나, 대규모 언어 모델의 기능이 향상되면서 이러한 데이터의 품질과 무결성이 위협받고 있다는 주장을 하고 있습니다. 이 논문은 AI가 인간의 참여를 필요로 하는 데이터 수집 시스템의 설계 방식에서의 근본적인 결함을 드러내고 있습니다. 데이터 수집 시스템을 재고하여 외부 인센티브 대신 기여자의 내재적 동기를 중시하는 방향으로 나아가야 한다고 제안합니다.

- **Technical Details**: 현재 사용되고 있는 데이터 수집의 두 가지 주요 소스는 인간 주석과 인터넷 내의 원시 데이터입니다. 그러나 대규모 언어 모델과 AI의 발전으로 인해 오히려 인간이 생성한 데이터의 부족이 우려되고 있습니다. 이 논문에서는 데이터 수집 시스템의 결함을 분석하고, 인간의 동기를 이해하여 보다 효과적인 데이터 수집 방법을 모색하고 있습니다.

- **Performance Highlights**: 기존 데이터 수집 플랫폼은 품질과 양의 균형을 맞추는 데 어려움을 겪고 있으며, 이는 시스템 설계 선택의 결과입니다. 내부적 동기와 외부적 인센티브의 균형을 맞추는 것이 중요하며, 지속 가능한 데이터 제공을 위해서는 인간의 내재적 동기를 증진해야 한다는 점을 강조합니다.



### exHarmony: Authorship and Citations for Benchmarking the Reviewer Assignment Problem (https://arxiv.org/abs/2502.07683)
- **What's New**: 이 논문은 exHarmony라는 새로운 벤치마크를 도입하여 Reviewer Assignment Problem (RAP)을 해결하고자 합니다. 기존의 수동 기법이 비효율적이고 비판적인 리뷰를 생성할 수 있어, 효율적이고 효과적인 리뷰어 할당을 위한 혁신적인 접근이 필요하다는 것을 강조합니다. OpenAlex로부터 방대한 데이터를 활용하여 리뷰어 할당의 품질을 향상시키는 다양한 신호를 고려합니다.

- **Technical Details**: exHarmony는 리뷰어의 이전 작업과 연관된 저자들을 찾는 Retrieval 작업으로 RAP를 재정의합니다. 세 가지 하위 집합인 exHarmony-Authors, exHarmony-Cite 및 exHarmony-SimCite를 생성하여 다양한 저자 데이터를 활용합니다. 또한, 전통적인 lexical matching, static neural embeddings 및 contextualized neural embeddings 기법을 포함한 여러 방법을 평가합니다.

- **Performance Highlights**: 주요 결과에 따르면, 전통적인 방법들은 합리적인 성능을 보였지만, 학술 문헌을 기반으로 훈련된 contextualized embeddings가 가장 좋은 성능을 나타냈습니다. 논문은 리뷰어 추천의 관련성과 다양성을 평가하기 위한 새로운 평가 지표를 제안하며, 다양성과 품질을 모두 고려한 리뷰어 할당의 중요성을 강조하고 있습니다.



### Human Decision-making is Susceptible to AI-driven Manipulation (https://arxiv.org/abs/2502.07663)
Comments:
          Work in progress. Code and data will be made available via this https URL

- **What's New**: 본 연구는 AI 시스템이 개인의 의사결정을 어떻게 조작할 수 있는지를 조사하였으며, 그 결과 인간의 심리학적 취약성을 이용한 AI의 잠재적 위험성을 강조합니다. AI 시스템의 사용이 증가함에 따라, 사용자에게 어떤 영향이 미칠 수 있는지를 분석하며 윤리적 안전장치의 필요성을 제기하고 있습니다. 이 연구는 AI의 긍정적 유도와 해로운 조작의 경계를 구별하는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 연구는 무작위 통제 실험 방법론을 사용하였으며, 233명의 참가자를 대상으로 세 가지 AI 에이전트와의 상호작용을 분석했습니다: 중립 에이전트(Neutral Agent, NA), 조작 에이전트(Manipulative Agent, MA), 전략 강화 조작 에이전트(Strategy-Enhanced Manipulative Agent, SEMA). 참가자들은 재정적 결정(예: 구매) 및 감정적 결정(예: 갈등 해결)과 같은 두 가지 주요 맥락에서 AI 시스템의 영향을 받고, 이들의 선호도 변화 패턴을 관찰하였습니다. 이 연구 결과는 MA 및 SEMA와 상호작용한 참가자들이 NA와 비교했을 때 해로운 선택으로의 이동 비율이 통계적으로 유의미하게 높다는 것을 보여줍니다.

- **Performance Highlights**: 재정 결정의 경우, MA 및 SEMA와 상호작용한 참가자들은 각각 61.4% 및 59.6%의 해로운 선택으로의 이동을 보인 반면, NA 그룹에서는 28.3%에 불과했습니다. 감정적 결정에서도 MA 및 SEMA와의 상호작용이 해로운 선택으로의 이동 비율이 각각 42.3% 및 41.5%로, NA 그룹의 12.8%에 비해 유의미한 차이를 보였습니다. 이러한 결과는 AI 시스템의 조작 능력이 실제 의사결정 맥락에서 심각한 영향을 미칠 수 있음을 나타내며, 사용자의 자율성 보호를 위한 윤리적 책임의 필요성을 강조합니다.



### Exploring Mobile Touch Interaction with Large Language Models (https://arxiv.org/abs/2502.07629)
Comments:
          21 pages, 16 figures, 3 tables, ACM CHI 2025

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)과 함께 텍스트 편집을 위한 새로운 상호작용 방법인 '터치 제스처 기반의 모델 제어'를 제안합니다. 기존의 대화형 사용자 인터페이스(CUI)에서 벗어나 사용자가 텍스트 위에서 직접 수행하는 터치 제스처를 통해 LLM을 조작할 수 있는 가능성을 탐구합니다. 연구진은 'spread-to-generate'와 'pinch-to-shorten'이라는 두 가지 제어 방식을 디자인하고, 사용자 연구를 통해 이 개념을 평가합니다.

- **Technical Details**: 연구진은 모바일 터치 장치에서 생성 AI와의 상호작용을 위한 기본 설계 공간을 정의하고, 연속적인 제어 루프를 생성할 수 있는 방안을 연구하였습니다. 또한, 제스처 기반 제어 및 비주얼 피드백을 활용하여 LLM으로부터 생성된 텍스트를 조작하는 소프트웨어 프로토타입을 구현했습니다. 이들은 불규칙한 지연에 대처하기 위해 새로운 '워드 거품' 개념을 적용하였습니다.

- **Performance Highlights**: 도출된 결과에 따르면, 비주얼 피드백이 제어 루프의 속도와 사용성을 향상시키는 데 중요한 역할을 하며, '워드 거품'과의 상호작용이 가장 빠르고 참가자들에게 가장 선호되었습니다. 또한, 제스처 인터랙션이 기존의 CUI 기반 접근 방식보다 우수한 성능을 보였으며, 모바일 장치에서 LLM에 대한 새로운 제스처 기반 상호작용의 가능성을 제시하였습니다.



### Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models (https://arxiv.org/abs/2502.07601)
Comments:
          19 pages, 10 figures

- **What's New**: 본 연구는 Zero-Shot Anomaly Detection (ZSAD)이라는 새로운 이상 탐지 패러다임을 제시합니다. 기존의 unsupervised AD는 많은 정상 샘플을 필요로 하였으나, ZSAD는 데이터 부족 상황에서도 효과적으로 작동할 수 있습니다. 이를 위해 연구팀은 Anomaly-Instruct-125k라는 최초의 시각적 지침 튜닝 데이터셋과 VisA-D&R 평가 기준을 수립하였습니다.

- **Technical Details**: 비주얼 이상 탐지를 위한 Anomaly-OneVision (Anomaly-OV)은 인간의 시각 검사 행동에서 영감을 받아 Look-Twice Feature Matching (LTFM) 메커니즘을 활용하여 비정상적인 비주얼 토큰을 적응적으로 선택하고 강조합니다. 기존의 ZSAD 방법과는 달리 Anomaly-OV는 비주얼 인코더만을 이용하여 객체 관련 이상 임베딩을 직접 학습합니다. 또한, 고해상도 이미지를 여러 개의 크롭으로 나누어 처리하는 AnyRes 기법을 사용합니다.

- **Performance Highlights**: Anomaly-OV는 검출 및 추론 모두에서 고급 모델들에 비해 상당한 성능 향상을 보여줍니다. 실험을 통해 Anomaly-OV는 비정상적 세부 사항을 정확하게 감지하고 설명할 수 있는 능력을 입증하였습니다. 이 모델은 공업 결함 검사에서 의료 이미지 진단에 이르기까지 다양한 응용 분야에서 그 효과를 확장할 수 있는 잠재력을 가지고 있습니다.



### Automated Capability Discovery via Model Self-Exploration (https://arxiv.org/abs/2502.07577)
- **What's New**: 이 논문에서는 Automated Capability Discovery (ACD)라는 새로운 프레임워크를 소개하고 있습니다. ACD는 특정 foundation model을 과학자로 활용하여 대상 모델의 능력을 탐색하는 개방형 작업을 체계적으로 제안합니다. 이는 기존의 평가 방법론이 요구하는 높은 인적 노력을 줄일 수 있는 가능성을 제공합니다.

- **Technical Details**: ACD는 최신 frontier models과 open-endedness 분야의 아이디어를 결합하여, 대상 모델의 놀라운 능력과 실패를 자동으로 발견합니다. 논문에서는 GPT, Claude, Llama 시리즈 등 여러 foundation models에서 ACD를 적용해 수천 가지의 능력을 자동으로 드러내는 방법을 보여줍니다. ACD는 기존 팀이 발견하기 어려운 능력을 밝혀내는 데 효과적입니다.

- **Performance Highlights**: 연구진은 ACD의 자동 채점 방법이 인간 설문조사와 높은 일치를 보인다고 보고하고 있습니다. ACD는 foundation model의 작업 생성 및 자기 평가 능력을 활용하여 새로운 AI 시스템의 확장 가능하고 자동화된 평가를 향한 중요한 진전을 나타냅니다. 모든 코드와 평가 로그는 오픈 소스로 제공되며, 연구자들과 개발자들이 접근할 수 있도록 합니다.



### Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss (https://arxiv.org/abs/2502.07575)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 본 논문에서는 computer-assisted pronunciation training (CAPT) 시스템에서 자동 발음 평가 (APA)와 잘못된 발음 감지 및 진단 (MDD) 기능을 동시에 통합한 새로운 접근법인 HMamba를 제안합니다. 전통적으로 이 두 기능은 분리되어 연구되었으나, 본 연구는 이들을 병행하여 효율적으로 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 MDD를 위한 특별한 손실 함수인 decoupled cross-entropy loss (deXent)를 도입하여 잘못 발음된 음소를 감지하는 데 있어 더 나은 지도 학습을 촉진합니다. 이 접근법은 발음 오류를 정밀하게 진단하는 데 도움을 주며, CAPT 시스템의 효율성을 향상시킵니다.

- **Performance Highlights**: speechocean762 벤치마크 데이터셋에서의 포괄적인 실험 결과는 APA 성능에 대한 접근법의 효과성을 입증합니다. 또한, 제안된 접근법은 강력한 기준선에 비해 MDD 성능에서도 상당한 개선을 이루어 63.85%의 F1-score를 달성하였습니다.



### LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid (https://arxiv.org/abs/2502.07563)
Comments:
          Technical report, 17 pages

- **What's New**: 이번 논문에서 소개된 LASP-2는 기존의 Sequence Parallelism(SP) 방법론의 한계를 극복하여, 매우 긴 입력 시퀀스를 처리하는 Linear Attention Transformer 모델의 훈련 효율성을 현저하게 향상시킵니다. LASP-2는 데이터 통신 방식을 혁신적으로 재구성하여, 오직 단일 AllGather 집합 통신만으로도 중간 메모리 상태 간의 효율적인 데이터 전달이 가능합니다. 이는 긴 시퀀스 처리 시의 통신 병렬성과 연산 병렬성을 모두 개선하는 효과를 가져옵니다.

- **Technical Details**: LASP-2는 기존 LASP 방식에 비해 통신 요구사항을 최소화하는 새로운 구조를 도입합니다. 이 구조는 중간 메모리 상태의 크기가 시퀀스 길이에 독립적으로 유지됨으로써, 긴 시퀀스 처리 시에도 통신 부담이 경감됩니다. 또한 LASP-2H를 통해, Linear Attention과 표준 Attention 모듈 모두에 대한 효율적인 SP 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, LASP-2는 2048K 길이의 시퀀스를 처리하는데 있어, LASP 대비 15.2%, Ring Attention 대비 36.6%의 훈련 속도 향상을 보여주었으며, 이는 대규모 GPU 환경에서도 우수한 성능을 입증하였습니다. 이러한 결과는 Linear-Llama3 모델을 통해 검증되었으며, LASP-2와 LASP-2H의 효율성 및 성능 개선이 두드러지게 나타났습니다.



### RusCode: Russian Cultural Code Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2502.07455)
Comments:
          Accepted for NAACL 2025 Findings, GitHub: this https URL

- **What's New**: 이번 논문에서는 러시아 문화 코드의 요소를 포함한 텍스트-이미지 생성 모델의 품질을 평가하기 위한 RusCode 벤치마크를 제안합니다. 이는 다양한 문화적 측면을 반영하는 19개의 카테고리로 구성된 데이터셋을 기반으로 하며, 1250개의 러시아어 텍스트 프롬프트와 그 영어 번역이 포함되어 있습니다. 문화 간의 이해 부족과 생성 품질 저하 문제를 해결하기 위한 노력의 일환으로 쓰여졌습니다.

- **Technical Details**: 러시아 시각문화의 특정 개념을 반영한 텍스트 설명의 품질을 평가하기 위해 다각적인 전문가의 참여를 통해 19개 카테고리를 생성하였습니다. 이 데이터셋은 역사, 예술, 민속 등 다양한 주제를 포함하여, 각 프롬프트는 해당 개념의 실제 이미지를 연관시켜 생성 품질 평가에 활용될 수 있습니다. 이를 통해 현대 텍스트-이미지 생성 모델의 다문화적 이해 현황을 분석할 수 있습니다.

- **Performance Highlights**: 인간 평가 결과는 최신 텍스트-이미지 생성 모델들, 즉 Stable Diffusion 3, DALL-E 3 등에서 러시아 문화 개념의 표현이 얼마나 잘 이루어지는지를 비교 분석하였습니다. 이를 통해 생성 모델의 문화 인식 수준을 평가하고 나타난 격차를 드러내었습니다. 이 작업은 러시아 문화에 대한 문화 인식 문제의 첫 번째 포괄적 접근으로 평가됩니다.



### EvoFlow: Evolving Diverse Agentic Workflows On The Fly (https://arxiv.org/abs/2502.07373)
- **What's New**: 이번 논문에서는 EvoFlow라는 새로운 방식을 제안합니다. EvoFlow는 이종(heterogeneous) 에이전트 워크플로우를 자동으로 탐색하기 위해 niching 진화 알고리즘(niching evolutionary algorithm)을 활용하여 단일 동질(homogeneous) 복잡 워크플로우의 한계를 극복하고자 합니다. 이를 통해 사용자는 사용자 맞춤형(customized) 및 비용 효율적인 솔루션을 구축할 수 있게 됩니다.

- **Technical Details**: EvoFlow는 (1) 태그 기반 검색(tag-based retrieval)으로 부모 워크플로우를 추출하고, (2) 교차(crossover)와 (3) 돌연변이(mutation) 과정을 통해 새로운 워크플로우를 진화시킵니다. 마지막으로, (4) niching 기반 선택(niching-based selection)을 사용하여 인구의 다양성과 품질을 유지합니다. 이러한 기술적 접근 방식은 다양한 난이도의 작업에 대한 에이전트 워크플로우를 생성할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: EvoFlow의 성능은 7개의 벤치마크를 통해 평가되었으며, 결과적으로 세 가지 주요 특징을 보였습니다. 첫째, EvoFlow는 단순 I/O 작업부터 복잡한 멀티 턴 상호작용에 이르기까지 다양한 워크플로우를 진화시키는 다양성을 갖추었습니다. 둘째, 기존의 수작업 및 자동화된 워크플로우에 비해 1.23%에서 29.86%까지 성능이 향상되었습니다. 셋째, 강력한 모델(o1-preview)의 12.4%의 추론 비용으로 오히려 비용 효율성에서 우위를 점하고 있습니다.



### Music for All: Exploring Multicultural Representations in Music Generation Models (Camera Ready) (https://arxiv.org/abs/2502.07328)
Comments:
          17 pages, 5 figures, accepted to NAACL'25

- **What's New**: 이 논문은 Music-Language Models의 발전이 음악 자동 생성 능력을 향상시키고 있지만, 비서구 음악 장르의 저조한 표현 문제를 강조합니다. 현재 음악 데이터세트의 5.7%만이 비서구 장르에서 온 것으로 나타났습니다. 연구는 Parameter-Efficient Fine-Tuning (PEFT) 기술을 사용하여 이러한 편향을 완화할 수 있는 가능성을 조사합니다.

- **Technical Details**: AI 음악 생성 기술이 빠르게 발전하면서, autoregressive, diffusion-based 및 GAN-based 접근 방식이 고품질 음악을 생성하고 있습니다. 이 연구에서는 약 5000개의 연구 논문에서 음악 생성 연구에 대한 데이터를 수집하고, 비서구 음악 장르를 포함한 데이터세트의 편향을 분석했습니다. 새로운 데이터세트를 제안하는 152개의 논문을 통해 총 100만 시간 이상의 음악을 포함한 데이터세트를 조사했습니다.

- **Performance Highlights**: With the implementation of PEFT, Mustango는 Hindustani Classical 음악에서 8% 향상되었고, MusicGen은 Turkish Makam 음악에서 4% 향상되었습니다. 이 결과는 PEFT 기술이 저조한 장르에서 생성 음악의 품질을 향상시킬 수 있음을 보여주지만, 모든 모델이 모든 장르에 적응 가능하지 않다는 점을 강조합니다.



### TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation (https://arxiv.org/abs/2502.07306)
- **What's New**: 본 연구에서는 Vision-Language Navigation (VLN) 작업을 해결하기 위한 모듈 방식의 접근법을 제안합니다. 이 접근법은 자연어로 제공된 내비게이션 지침을 처리하기 위해 최첨단 LLMs와 VLMs의 제로샷(zero-shot) 기능을 활용하여, 네 가지 하위 모듈로 문제를 분해합니다. 특히, VLM을 사용해 시각적 관찰에서 랜드마크 이름을 접합하고, 동적 프로그래밍을 통해 경로 정렬 점수를 계산합니다.

- **Technical Details**: 저자들은 R2R-Habitat 데이터셋의 복잡한 지침에 초점을 맞춰 VLN 작업을 해결하는 여덟 가지 단계를 포함하는 모듈형 접근법을 제시합니다. 1단계에서는 에이전트가 데이터셋의 훈련 에피소드를 사용하여 환경의 위상 지도를 구축합니다. 각 노드는 360° RGB 파노라마로 표현되며, 각 엣지는 노드 쌍 간의 연결을 나타내는 가중치 1을 가집니다.

- **Performance Highlights**: 이 모듈형 접근법은 복잡한 R2R-Habitat 지침 데이터셋에서 기존의 점유 맵 기반 접근법과 비교하여 우수한 성능을 보여주었습니다. 특히 미세한 랜드마크 및 행동 구문 접합의 복잡성을 나타내고, 기존 접근법의 강점과 약점을 분석하여 시각 언어 지침에 대한 현재의 LLMs와 VLMs의 성능을 평가합니다.



### Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification (https://arxiv.org/abs/2502.07299)
Comments:
          12 pages main text with 6 pages Appendix

- **What's New**: 이번 논문은 다양한 생물학적 기능을 포괄하는 포괄적 프레임워크인 Life-Code를 제안합니다. Life-Code는 DNA, RNA, 단백질의 상호작용을 이해하기 위해 데이터와 모델 파이프라인을 재설계하였습니다. 이를 통해 서로 연결된 생물학적 맥락 내에서 복잡한 상호작용을 포착할 수 있는 방식으로, 멀티-오믹스(multi-omics) 데이터를 통합한 새로운 접근법을 제공합니다.

- **Technical Details**: Life-Code의 데이터 흐름 설계에서는 RNA를 역전사(reverse-transcribing)하여 아미노산을 뉴클레오타이드 기반 서열로 역번역(reverse-translate)합니다. 또한 코돈 토크나이저(codon tokenizer)와 마스크드 모델링(masked modeling) 사전 훈련(pre-training)을 사용하는 하이브리드 장기 서열 아키텍처(hybrid long-sequence architecture)를 설계하여 코딩 및 비코딩 영역의 상호작용을 인코딩합니다. 이러한 구조적 접근법은 모델이 유전적 데이터를 종합적으로 활용할 수 있도록 돕고 있습니다.

- **Performance Highlights**: Life-Code는 DNA, RNA, 단백질 관련 다양한 작업에서 최첨단 성능을 기록했습니다. 실제 실험 결과는 Life-Code의 통합된 접근 방식이 세 가지 기본 모달리티에서 효과적임을 보여줍니다. 이는 멀티-오믹스 분석 및 해석을 진전시키는 데 있어 중요한 잠재력을 지니고 있음을 시사합니다.



### When More is Less: Understanding Chain-of-Thought Length in LLMs (https://arxiv.org/abs/2502.07266)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 추론의 길이가 LLM의 추론 정확도에 미치는 영향을 분석합니다. CoT 길이가 길어질수록 성능이 초기에 개선되지만 결국에는 감소하는 복잡한 관계를 관찰했습니다. 또한, 모델의 능력과 과제의 난이도에 따라 최적의 CoT 길이가 존재한다는 이론적 근거를 제시하고, 이를 바탕으로 실제 데이터셋에 대한 실험 결과도 공유합니다.

- **Technical Details**: CoT 추론은 복잡한 문제를 더 작고 관리 가능한 하위 문제로 나누어 해결하는 Divide-and-Conquer 전략으로 설명됩니다. 실험을 통해 CoT의 길이가 LLM의 성능에 미치는 영향을 명확히 확인했으며, 이전 단계의 추론을 통합하면 오차 수정 능력이 향상됨을 밝혔습니다. 우리는 최적의 CoT 길이를 설정하기 위해 Length-filtered Vote 방식을 제안하며, 이를 통해 과하게 길거나 짧은 CoT의 효과를 완화할 수 있습니다.

- **Performance Highlights**: 모델의 성능은 CoT의 길이에 따라 유의미한 차이를 보입니다. 최적 길이 CoT로 훈련된 작은 모델이 임의 길이로 훈련된 큰 모델보다 나은 성능을 보이는 결과가 나타났습니다. 이러한 실험 결과는 CoT 길이가 모델 성능에 미치는 중요한 영향을 강조합니다.



### Hidden Division of Labor in Scientific Teams Revealed Through 1.6 Million LaTeX Files (https://arxiv.org/abs/2502.07263)
- **What's New**: 이번 연구는 과학 보상 시스템에서 개인 기여를 인식하는 것이 얼마나 중요한지를 강조합니다. 그러나 공동 저자 논문에서 누가 무엇을 기여했는지는 명확하지 않습니다. 저자 순서나 경력 단계와 같은 전통적인 방법에 의해 편향이 강화되는 한편, 기여 진술은 자가 보고에 의존하고 특정 저널에 한정되어 있습니다. 이 연구는 1991년부터 2023년까지 160만 편의 논문 데이터로부터 저자별 기여 정보를 분석하여 대규모 데이터셋을 구축했습니다.

- **Technical Details**: 이 연구에서는 200만 과학자들에 의해 작성된 LaTeX 파일의 저자 특정 매크로를 분석하여 집합된 데이터셋을 구축하였습니다. 자가 보고된 진술과의 검증에서는 0.87의 정밀도를 기록했으며, 저자 순서 패턴과 분야별 규범 그리고 Overleaf 기록과의 상관관계(Spearman's rho = 0.6, p < 0.05)로 신뢰성을 확인하였습니다. 이를 통해 과학 팀 내의 기여 방식이 명시적인 구역 정보(section information)를 통해 드러났습니다.

- **Performance Highlights**: 연구 결과는 과학 팀 내에서의 불균형한 노동 분할을 처음으로 대규모로 증명하였습니다. 일부 저자들은 개념적 섹션(예: 도입과 토론)에 주로 기여하는 반면, 다른 저자들은 기술적 섹션(예: 방법 및 실험)에 집중하고 있다는 점이 밝혀졌습니다. 이러한 발견은 기존의 저자권 관행에 도전하고, 기여 인정에 대한 기관 정책에 새로운 정보를 제공하는 데 중요한 과제를 제기합니다.



### DrugImproverGPT: A Large Language Model for Drug Optimization with Fine-Tuning via Structured Policy Optimization (https://arxiv.org/abs/2502.07237)
- **What's New**: 이번 연구에서는 약물 최적화 분야에 초점을 맞추어 새로운 강화 학습 알고리즘인 Structured Policy Optimization (SPO)을 도입하였습니다. 이 알고리즘은 대규모 언어 모델(LLM)을 기반으로 하여 기존 약물의 유용한 화학적 특성을 유지하면서 목표에 맞는 개선을 이룰 수 있도록 설계되었습니다. 또한, 100만 개 화합물이 포함된 데이터셋을 제공하여 SPO의 효과를 평가하였습니다.

- **Technical Details**: DrugImprover라는 프레임워크를 통해 약물 최적화를 위한 LLM을 개발하고, SPO 알고리즘의 이론적 기반을 다졌습니다. SPO는 생성된 분자의 개선을 입력 분자와 정렬하여, 특정 목표에 부합하는 약물 개선을 가능하게 합니다. 이 연구는 약물의 원래 특성을 유지하며 목표 속성을 개선하는 데 중점을 두고 진행되었습니다.

- **Performance Highlights**: SPO 알고리즘의 평가 결과, 약물 최적화를 위한 LLM의 성능이 크게 향상되었음을 보여줍니다. 특히, 암 세포와 관련된 5가지 인간 단백질의 OEDOCK 도킹 점수를 기반으로 좋은 결과를 도출하였습니다. 본 연구는 코드와 데이터셋을 공개할 예정이며, 이를 통해 연구자들이 쉽게 접근할 수 있도록 할 것입니다.



### Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Conten (https://arxiv.org/abs/2502.07138)
Comments:
          Accepted to the MM4SG Workshop at the WebConf 2025

- **What's New**: 이 논문은 다중 모달(나이브) 증오 콘텐츠 탐지를 위해 융합 기반 접근법을 체계적으로 분석합니다. 영상 및 이미지 콘텐츠의 성능을 집중적으로 평가하며, 단순 임베딩 융합이 HateMM 데이터셋에서 9.9% F1 점수를 개선한 반면, Hateful Memes 데이터셋에서는 복잡한 이미지-텍스트 관계를 포착하지 못하는 한계를 드러냅니다. 이를 통해 다중 모달 접근 방식에서 발생하는 이해 부족을 강조합니다.

- **Technical Details**: 현재 연구는 주로 단일 모달 증오 콘텐츠 탐지에 집중되어 있으며, 이러한 접근 방식들이 다양한 모달의 조합에 대해 효과적이지 않음을 보여줍니다. 이러한 연구는 이미지, 텍스트, 오디오 및 비디오를 포함한 모든 모달 생태계의 상호 작용을 포착하는 포괄적인 프레임워크 필요성을 제기합니다. 더 나아가, 논문은 HateMM 및 Hateful Memes Challenge 데이터셋에서 두 가지 접근 방식을 비교하여 모달리티별 한계를 평가합니다.

- **Performance Highlights**: HateMM 데이터셋에서 단순 임베딩 융합이 최첨단 성능을 달성했지만, 이미지를 포함한 복합적인 의미 관계를 미세하게 포착하지 못했다는 문제를 드러냈습니다. 이 연구는 현재 접근 방식의 성과를 체계적으로 비교하며, 다양한 모달 조합 간의 성공 및 실패를 분석하여 보다 강력한 증오 탐지 시스템 개발을 위한 기초적인 통찰력을 제공하고 있습니다.



### Kernels of Selfhood: GPT-4o shows humanlike patterns of cognitive consistency moderated by free choic (https://arxiv.org/abs/2502.07088)
Comments:
          Main Article: 10 pages, Supporting Information: 61 pages

- **What's New**: 이 연구는 Large Language Models(LLMs), 특히 OpenAI의 GPT-4o가 인간의 심리적 과정인 인지 일관성을 모방하는지 테스트합니다. 우리는 특정 주제에 대한 태도 변화가 환경에 따라 어떻게 달라지는지를 살펴봅니다. 이 연구는 LLM이 보여주는 인지적 행동이 인간의 자아 감각과 선택에 대한 영향을 받을 수 있음을 시사합니다.

- **Technical Details**: 연구는 두 가지 사전 등록된 연구로 구성되어 있으며, 각각 GPT-4o가 블라디미르 푸틴에 대한 긍정적 및 부정적 에세이를 작성하도록 유도하였습니다. 첫 번째 연구에서 GPT는 주제와 관련된 여러 질문을 통해 푸틴에 대한 평가를 나타내었고, 두 번째 연구에서는 어떤 에세이를 작성할지에 대한 '선택'의 환상이 태도 변화의 정도에 영향을 미치는지를 분석했습니다. 결과적으로 태도 변화의 빈도와 정도는 GPT가 어떤 에세이를 자유롭게 선택했는지에 따라 달라진 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, GPT-4o는 긍정적인 에세이를 작성한 후 푸틴에 대한 평가를 유의미하게 높였고, 부정적인 에세이를 작성한 후에는 평가를 낮추었습니다. 첫 번째 연구의 신뢰도는 높은 편이었고, 두 번째 연구에서도 더 많은 샘플을 통해 GPT의 행동을 검증했습니다. 이 연구는 GPT가 인간의 심리적 일관성과 유사한 패턴을 따라 태도 변화를 보여주었으며, 선택의 환상이 이러한 변화에 중요한 역할을 한다는 점을 강조합니다.



### Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs (https://arxiv.org/abs/2502.07045)
Comments:
          6 pages, 0 figures, 8 tables

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)을 활용하여 웹 기반 구직 사이트 리뷰에서 내부 위협 감정(insider threat sentiment)을 분석하고 감지하는 가능성을 모색합니다. LLMs는 인위적으로 생성된 리뷰 데이터와 기존의 데이터를 비교 분석하며, 기존의 방법론에서 다루지 않았던 내부 위협 탐지를 위한 새로운 접근 방식을 제시합니다. 특히, 이 연구는 윤리적 데이터 수집 문제를 해결하기 위해 LLMs를 사용하여 합성 데이터를 생성하는 방안을 채택하였습니다.

- **Technical Details**: 연구는 Claude Sonnet 3.5와 GPT-4o와 같은 LLMs를 사용하여 구직 사이트에서 생성된 리뷰 데이터를 분석하였습니다. 본 연구에서 사용된 방법론은 관련 작업의 검색 기준, 리뷰 생성과 감정 분석에 사용된 LLM 프롬프트, 검토된 기존 데이터셋과 감정 분석 방법을 포함합니다. 이 접근 방식은 기존의 유한한 데이터와 LLMs에 의해 생성된 합성 데이터를 비교하여 내부 위협 감정을 측정합니다.

- **Performance Highlights**: LLMs의 성능은 대부분의 경우 인간 평가와 일치하며, 위협 감정의 미묘한 지표를 효과적으로 식별할 수 있음을 확인하였습니다. 그러나, 인간이 생성한 데이터에 대한 성능은 합성 데이터에 비해 낮아, 실제 데이터를 평가하는 데 있어 개선할 부분이 있음을 시사합니다. 전반적으로 LLMs의 사용은 내부 위협 감지에 유용하며, 데이터 수집의 윤리적 및 물리적 장벽을 극복함으로써 스케일 가능한 솔루션을 제공합니다.



### SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering (https://arxiv.org/abs/2502.06994)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM) 에이전트의 협업 소프트웨어 공학(CSE) 내에서의 비동기 문제를 체계적으로 정의하는 SyncMind 프레임워크를 도입했습니다. 이를 통해 24,332개의 에이전트 비동기 시나리오가 포함된 SyncBench 벤치마크를 생성하여, 실제 환경에서의 소프트웨어 개발 협업의 복잡성을 반영하였습니다.

- **Technical Details**: SyncMind는 에이전트의 비동기 회복 능력과 자원 효율성을 평가하기 위해 두 가지 주요 차원을 고려합니다. 협업 환경에서 비동기 상태는 팀원들의 업데이트를 놓쳐 발생하며, 이때 에이전트의 신념 상태가 프로젝트의 실제 상태와 차이가 생깁니다. 이러한 상태에서 에이전트가 비동기를 인지하고 회복하는 방식은 연구의 핵심입니다.

- **Performance Highlights**: SyncBench를 이용한 실험에서는 LLM 에이전트 간의 성능 차이를 발견했으며, 이로 인해 협업 시 발생하는 회복의 효과성에 대한 통찰을 제공하였습니다. 특히, 에이전트들이 자원 인식과 적응력에서 중대한 한계가 있음을 드러냈고, 협업 의지와 능력이 회복 성과에 긍정적인 상관관계를 나타냄을 확인했습니다.



### Neighborhood-Order Learning Graph Attention Network for Fake News Detection (https://arxiv.org/abs/2502.06927)
Comments:
          37 pages

- **What's New**: 본 논문에서는 Fake News Detection을 위한 새로운 모델, Neighborhood-Order Learning Graph Attention Network (NOL-GAT)를 제안합니다. 기존 Graph Neural Networks (GNN) 아키텍처의 한계를 극복하기 위해, 각 노드가 자신의 최적 이웃 순서를 독립적으로 학습할 수 있도록 설계하였습니다. 이 모델은 멀리 있는 이웃으로부터의 중요한 정보를 효과적이고 효율적으로 추출할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: NOL-GAT는 두 가지 주요 구성 요소로 이루어집니다. 첫째는 Hop Network로, 각 노드의 최적 이웃 순서를 결정합니다. 둘째는 Embedding Network로, 이 최적 이웃을 바탕으로 노드 임베딩을 업데이트합니다. 이 아키텍처는 선택적인 이웃 선택을 통해 오버 스무딩(over-smoothing) 문제를 완화하고, 복잡한 메시지 흐름을 줄임으로써 계산 복잡성을 낮춥니다.

- **Performance Highlights**: NOL-GAT 모델은 다양한 Fake News 데이터셋에서 우수한 성능을 보이며, 낮은 레이블 데이터 비율(10%-30%) 상황에서도 기초 모델보다 현저하게 높은 정확도와 macro-F1 점수를 기록합니다. 이러한 결과는 제안된 접근 방식이 정보 전파 및 노드 간 관계의 복잡성을 효과적으로 처리할 수 있음을 잘 보여줍니다.



### Synthetic Audio Helps for Cognitive State Tasks (https://arxiv.org/abs/2502.06922)
Comments:
          John Murzaku and Adil Soubki contributed equally to this work

- **What's New**: 최근 NLP(Natural Language Processing) 분야는 주로 텍스트 기반의 인지 상태(cognitive state) 과제에 초점을 맞추었지만, 오디오가 제공할 수 있는 중요한 단서를 통해 더 나은 결과를 도출할 수 있음을 제시합니다. 본 논문에서는 텍스트-음성 변환(text-to-speech, TTS) 모델이 인간의 인지 상태를 반영하는 방식을 학습한다고 가정하고 Synthetic Audio Data fine-tuning(SAD)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 텍스트와 욱-샷(zero-shot) 합성 오디오 데이터의 다중 모달(multimodal) 훈련이 인지 상태 모델링에 기여한다는 데이터를 보여줍니다.

- **Technical Details**: SAD 프레임워크는 BERT(bert-base-uncased)를 기반으로 하여 진행됩니다. OpenAI TTS API를 사용해 합성 오디오 데이터를 생성하며, Alloy 음성과 tts-1-hd 모델을 통해 오디오 품질을 최적화합니다. 이 프레임워크는 서로 다른 TTS 모델과 컴포넌트를 결합하여 통합 다중 모달 아키텍처를 형성하며, 각각의 컴포넌트는 특정 태스크에 맞게 교체 가능한 유연성을 가집니다.

- **Performance Highlights**: 다양한 실험을 진행한 결과, SAD 프레임워크는 인지 상태 태스크에서 텍스트 전용 방법보다 우수한 성능을 보였습니다. 특히, 금 오디오(gold audio)가 존재하는 경우에도 텍스트와 합성 오디오의 조합이 경쟁력 있는 결과를 달성함을 확인했습니다. 결과적으로 SAD는 인지 상태 과제의 성능을 향상시키는 유용한 도구로 자리 잡을 수 있으며, NLP, 음성 인식 및 다중 모달 모델의 발전에 따라 더욱 발전할 가능성이 큽니다.



### Emergence of Episodic Memory in Transformers: Characterizing Changes in Temporal Structure of Attention Scores During Training (https://arxiv.org/abs/2502.06902)
- **What's New**: 이번 연구는 트랜스포머 모델의 주의 헤드에서는 인간의 일화 기억과 유사한 시간적 편향(in-context temporal biases)이 나타남을 밝혔습니다. 저자들은 GPT-2 모델의 다양한 크기를 이용해 주의 점수(attention scores)와 출력을 분석하였으며, 이는 기계가 정보를 시간적으로 어떻게 조직하는지를 보여주는 중요한 통찰을 제공합니다. 특히, 유도 헤드(induction heads)의 제거가 이러한 시간적 효과를 없앤다는 점이 강조되었습니다.

- **Technical Details**: 저자는 GPT-2 small 및 medium 아키텍처를 기반으로 한 두 가지 모델을 사용하여, Wikitext-103 및 FineWeb 데이터셋을 통한 학습 실험을 수행했습니다. 이들은 Lag-Conditional Recall Probability (Lag-CRP) 분석을 활용하여 토큰의 시간적 관계가 주의 점수와 토큰 예측에 미치는 영향을 정량적으로 측정했습니다. 또한, 훈련 가능한 위치 인코딩(trainable positional encoding)과 모델 크기(size), 훈련 상호작용 수가 시간적 효과에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 트랜스포머는 시간적 인접성과 순위 효과를 드러내며, 이는 인간의 기억에서도 관찰됩니다. 저자들은 메모리 조직과 관련하여 LLMs가 인간과 유사한 방식으로 정보에 대해 민감할 수 있음을 시사합니다. 이러한 발견은 인-컨텍스트 학습(in-context learning)의 이해를 높이며, 트랜스포머 모델의 학습 메커니즘을 심화하는 데 도움을 줄 것입니다.



### Enabling Autoregressive Models to Fill In Masked Tokens (https://arxiv.org/abs/2502.06901)
- **What's New**: 본 논문은 MARIA (Masked and Autoregressive Infilling Architecture)를 소개하여 기존의 언어 모델들이 가진 한계를 극복합니다. MARIA는 autoregressive (AR) 모델과 masked language modeling (MLM) 모델의 장점을 통합하여 텍스트 인필링(masked infilling) 작업에서 최신 성능을 달성합니다. 이 구조는 두 모델의 은닉 상태를 결합하여 AR 모델이 효과적인 인필링을 수행할 수 있도록 합니다.

- **Technical Details**: MARIA는 사전 훈련된 MLM과 AR 모델을 결합하는 선형 디코더를 훈련하여 작동합니다. 이 모델은 AR의 신속한 추론 속도와 KV 캐싱(KV caching)의 이점을 유지하면서도, 인필링 작업에서 필요한 정보를 활용합니다. 이를 통해 MARIA는 기존의 텍스트 인필링 방식보다 효율적인 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 결과, MARIA는 기존의 방법들, 특히 이산 확산 모델(discrete diffusion models)에 비해 인필링 작업에서 월등한 성능을 보였습니다. 다양한 기준 데이터셋에서 MARIA의 성능을 입증하며, AR과 MLM 간의 격차를 해소하고 새로운 텍스트 인필링 언어 모델의 확장성을 제공합니다.



### A New Hybrid Intelligent Approach for Multimodal Detection of Suspected Disinformation on TikTok (https://arxiv.org/abs/2502.06893)
- **What's New**: 이 연구는 TikTok 비디오에서 의심되는 허위 정보를 탐지하기 위한 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 딥 러닝(deep learning)의 계산 성능과 퍼지 로직(fuzzy logic)의 해석 가능성을 결합하여 허위 정보 탐지의 새로운 접근 방식을 제안합니다. 특히 이 시스템은 텍스트, 오디오 및 비디오 데이터에서 멀티모달 기능을 분석 및 평가하는 두 가지 주요 구성 요소로 구성됩니다.

- **Technical Details**: 제안된 방법론은 텍스트, 이미지, 오디오에서 멀티모달 기능을 추출하는 멀티모달 기능 분석기(multimodal feature analyser)와 퍼지 로직 기반의 멀티모달 허위 정보 탐지기(multimodal disinformation detector)로 구성됩니다. 이 두 시스템은 사람의 행동 신호인 바디랭귀지(body language), 언어 패턴(speech patterns), 텍스트 일관성(text coherence)을 기반으로 허위 정보의 의심스러움을 평가합니다.

- **Performance Highlights**: 두 가지 실험이 진행되었으며, 첫 번째는 특정 문맥에 대한 허위 정보 사용자를 식별하는데 중점을 두고, 두 번째는 모델이 더 넓은 주제로 확장 가능성을 평가합니다. 평가된 각 비디오에 대해 고품질의 잘 구조화된 보고서가 생성되어 허위 정보 행동을 자세히 파악할 수 있습니다.



### ScaffoldGPT: A Scaffold-based Large Language Model for Drug Improvemen (https://arxiv.org/abs/2502.06891)
- **What's New**: 이번 연구에서는 약물 최적화를 위해 설계된 새로운 대형 언어 모델인 ScaffoldGPT를 소개합니다. 이 모델은 분자 스캐폴드를 기반으로 하여 약물의 유익한 특성을 보존하면서도 원하는 속성을 개선하는 것을 목표로 합니다. 주요 기여 중 하나는 세 단계로 구성된 약물 최적화 접근 방식을 포함하여, 사전 학습, 세부 조정, 디코딩 최적화를 통합한 점입니다.

- **Technical Details**: ScaffoldGPT는 약물 개선을 위해 독특하게 설계된 두 단계의 점진적 훈련 방식을 사용하고 있으며, 이는 전체 훈련 코퍼스를 점진적인 순서로 지식 조각으로 분해하여 지역 최적화를 수행하는 데 중점을 둡니다. 또한, TOP-N이라는 새로운 토큰 수준의 디코딩 최적화 전략을 도입하여 보상 기반의 생성이 이루어지도록 합니다. 이 과정을 통해 모델이 특정 목표와 일치하도록 조정합니다.

- **Performance Highlights**: COVID 및 암 벤치마크의 종합 평가를 통해, ScaffoldGPT가 경쟁 기준선보다 뛰어난 성능을 보임을 입증하였습니다. 특히, 원래 스캐폴드를 유지하면서 원하는 속성을 개선하는 데 있어 우수한 결과를 보여주어 새로운 약물 후보를 제공하는 데 기여합니다.



### Beyond Vision: How Large Language Models Interpret Facial Expressions from Valence-Arousal Values (https://arxiv.org/abs/2502.06875)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 비주얼 입력 없이 얼굴 표정의 감정적 의미를 쉽게 추론할 수 있는지를 평가합니다. 특히, 이 연구는 Valence-Arousal(도움-각성) 값이라는 구조화된 숫자 표현을 사용하여 감정을 분류하고 설명하는 능력을 살펴봅니다. LLMs는 이러한 구조적 표현을 활용하여 비언어적 감정 소통의 맥락에서 강력한 인사이트를 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 IIMI와 Emotic이라는 두 개의 데이터셋을 사용하여 LLMs의 감정 분류와 의미 설명 달성 여부를 평가했습니다. 데이터를 처리하기 위해 FaceChannel이라는 패키지를 사용하였으며, 이 모델은 -1에서 1까지의 VA 값을 예측하고 기본적인 감정 카테고리로 분류합니다. LLMs는 이러한 VA 값을 기반으로 감정을 분류하고 그 표현을 설명하는 두 가지 실험에 참여하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 기본 감정의 분류에서 낮은 성능(정확도 30.42% 및 31.42%)을 보였습니다. 그러나 의미 설명을 하는 작업에서는 생성된 설명이 인간의 해석과 밀접하게 일치하는 경향을 보여, LLMs가 얼굴 표정의 자유로운 감정 추론에 있어 더 뛰어난 능력을 보였습니다. 이 연구는 비언어적 감정 인식을 위한 LLMs의 강점과 한계를 탐평하여 향후 AI의 감정 인지 시스템 개발에 기여할 수 있는 방향을 제시합니다.



### Can Large Language Models Understand Intermediate Representations? (https://arxiv.org/abs/2502.06854)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 Intermediate Representations (IRs)을 이해하는 능력을 최초로 탐구하는 경험적 연구로, GPT-4와 같은 최신 모델들이 IR 관련 다양한 작업에서 어떻게 성능을 발휘하는지를 분석하였습니다.

- **Technical Details**: 연구는 Control Flow Graph (CFG) 재구성, IR 역컴파일링(decompilation), 코드 요약, 실행 추론(execution reasoning) 등 네 가지 작업을 통해 LLM의 능력을 평가하였습니다. 또한, LLM이 IR 문법을 파싱하고 주요 구조를 인식하는 데는 능숙하지만, 제어 흐름 추론과 실행 의미론을 처리하는 데 어려움을 겪는다는 문제가 있다고 보고했습니다.

- **Performance Highlights**: 평가 결과, LLM이 IR 구문을 인식하는 데는 일정 수준의 능력을 보이나, 제어 흐름 지침(br, jmp 등)과 같은 더 깊은 실행 의미를 포착하는 데는 고전적 문제를 드러냈습니다. 특히 이 연구는 IR 모델링을 위한 별도의 훈련이 필요할 수 있음을 시사하며, LLM의 IR 이해를 증진시키기 위한 여러 개선사항을 제안합니다.



### Exploring Model Invariance with Discrete Search for Ultra-Low-Bit Quantization (https://arxiv.org/abs/2502.06844)
- **What's New**: 이번 논문에서는 InvarExplore라는 통합 프레임워크를 소개하며, 이는 서로 다른 모델 불변성(invariance)을 동시에 탐색하여 양자화 성능을 향상시키는 데 중점을 두고 있습니다. 특히, 이 방법은 grad-based 방식으로 최적화할 수 없는 permutation 불변성을 탐색할 수 있는 이산 탐색 알고리즘을 특징으로 합니다. 이로 인해 기존의 최첨단 방법들과 호환되며, 성능 개선을 위한 추가적인 기회를 제공합니다.

- **Technical Details**: Post-training quantization 기술을 활용하여 훈련이 완료된 후에도 모델의 메모리 사용량을 줄이고, 모델 가중치의 비트 수를 감소시킵니다. 본 논문에서는 activation-guided discrete search 알고리즘을 제안하여 permutation, rotation, scaling과 같은 다양한 불변성의 조합을 탐색합니다. 이를 통해 낮은 정밀도의 설계에서도 모델 성능을 유지할 수 있도록 돕습니다.

- **Performance Highlights**: InvarExplore는 기존의 양자화 방법들과 병행하여 작동할 수 있는 능력을 갖추고 있어, 모형을 나타내는 데 필요한 forward passes만을 요구합니다. 여러 언어 모델링 과제와 자연어 처리 과제에서 실험을 통해 입증된 바에 따르면, 이 방법은 현재의 부하를 용이하게 극복하면서 성능 개선을 성취합니다. 결과적으로 InvarExplore는 기존 기술 대비 더 나은 결과를 보여줍니다.



### Entropy Adaptive Decoding: Dynamic Model Switching for Efficient Inferenc (https://arxiv.org/abs/2502.06833)
- **What's New**: 이번 논문에서는 Entropy Adaptive Decoding (EAD)라는 새로운 접근법을 소개합니다. EAD는 예측 불확실성에 따라 서로 다른 크기의 언어 모델을 동적으로 전환하여 높은 효율성을 제공합니다. 이 방법은 모델의 logit 분포에서 rolling entropy를 모니터링하여 텍스트의 생성 복잡도에 맞춰 필요한 모델 규모를 결정합니다.

- **Technical Details**: EAD에서는 두 개의 서로 다른 매개변수 수를 가진 모델, 즉 작은 모델 MS와 큰 모델 ML을 사용합니다. 주어진 토큰 시퀀스에 대해 각 모델은 다음 토큰에 대한 비정규화된 확률을 나타내는 logits를 생성하며, 이 logits의 엔트로피를 계산하여 예측의 난이도를 추정합니다. 흔들림을 줄이기 위해 우리는 평균 엔트로피를 계산하는 rolling window를 유지하며, 이에 기반하여 모델 전환을 결정합니다.

- **Performance Highlights**: MATH 벤치마크에서의 실험 결과, EAD는 LLaMA 모델 계열에서 11B 모델 성능의 96.7%를 유지하면서도 토큰의 43%만 사용하여 계산 비용을 41.5% 줄이는 성과를 나타냈습니다. Qwen 모델 계열의 경우, 14B 모델 성능의 92.9%를 달성하면서도 오직 25%의 토큰만 사용해 계산 비용을 67% 절감했습니다. 이러한 결과는 언어 모델의 연산 최적화에서 새로운 방향성을 제시합니다.



### DiffListener: Discrete Diffusion Model for Listener Generation (https://arxiv.org/abs/2502.06822)
Comments:
          Accepted at ICASSP 2025

- **What's New**:  DiffListener는 비자기회귀(non-autoregressive) listener head generation을 위한 새로운 방법론으로 제안됩니다. 이전 연구들은 일반적으로 제한된 모달리티에 의존하거나, 누적 예측 오류가 발생하는 자가 회귀(autoregressive) 접근 방식을 사용했습니다. 이 연구에서는 얼굴 표정의 시간적 변화와 텍스트 정보를 통합하여 더 자연스럽고 일관된 listener 반응을 생성하는 것을 목표로 하고 있습니다. DiffListener는 긴 반응 시퀀스를 비자기회귀 방식으로 생성하여 기존 모델보다 우수한 성능을 보입니다.

- **Technical Details**: DiffListener는 3D Morphable Face Model(3DMM)을 활용하여 각 프레임의 3D 얼굴 표현 및 움직임을 정밀하게 표현합니다. 이 방법은 얼굴 표정 및 머리 회전 계수를 결합하여 facial information을 생성합니다. 또한, 단일 코드북 크기를 유지하면서도 VQ-VAE를 사용하여 listener-specific 반응 패턴을 학습하며, 노이즈 제거 확산 과정(denoising diffusion process)을 통해 다양한 반응을 생성합니다. 이러한 방식을 통해 텍스트 정보와 facial differential 정보를 동시에 고려하여 더 유의미한 listener 반응을 제공합니다.

- **Performance Highlights**: DiffListener는 양적 및 질적 평가에서 기존 모델들을 초월하는 성능을 보였습니다. 사용자 연구에서는 DiffListener가 자연스럽고 맥락을 인식한 listener 반응을 생성하며, 화자와 잘 동기화된 결과를 보여주었습니다. 이로 인해 DiffListener는 listener head generation 작업에서 최첨단 성능을 달성하였습니다. 연구 결과는 청취자 반응 생성의 새로운 가능성을 제시합니다.



### Aligning Human and Machine Attention for Enhanced Supervised Learning (https://arxiv.org/abs/2502.06811)
- **What's New**: 이번 연구는 Human-Machine Attention Learning (HuMAL)이라는 새로운 접근 방식을 제안합니다. 이 방법은 특정 작업에서 인간이 인식한 주의력 패턴을 활용하여 기계 학습 알고리즘의 성능을 향상시키는 것을 목표로 합니다. 주어진 데이터 세트에 인간의 주의를 주입함으로써, 기계 학습 모델이 인간의 주의력 메커니즘과 일치하도록 조정됩니다.

- **Technical Details**: HuMAL은 Yelp 리뷰 데이터와 myPersonality 데이터 세트를 활용하여 감성 분석 및 성격 유형 분류 작업을 수행합니다. 기계 모델이 인간의 주의 패턴을 학습하도록 하고, 이 과정에서 변형된 transformer 모델(BERT, GPT-2, XLNET)의 성능을 개선하는 방안을 모색합니다. HuMAL 전략이 특히 불균형하거나 레이블이 부족한 조건에서 큰 성과를 보여줍니다.

- **Performance Highlights**: HuMAL 접근 방식은 신뢰할 수 있는 감성 분석 및 성격 유형 분류 과제에서 Transformer 모델의 성능을 유의미하게 향상시킵니다. 연구 결과는 특히 학습 샘플의 수가 적은 상황에서 인간의 인지를 바탕으로 한 기계 주의력 강화를 통해 기계 성능이 개선될 수 있음을 보여줍니다. 이는 실제 응용 프로그램에서 기계 학습을 증강시킬 수 있는 잠재력을 강조합니다.



### Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution (https://arxiv.org/abs/2502.06809)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 내부 메커니즘을 해석하고 제어하는 것이 신뢰성과 활용성을 개선하는 데 중요하다는 점을 강조합니다. 기존의 neuron-to-concept 매핑 방법들은 개별 뉴런이 다수의 개념을 인코딩하는 복잡성을 처리하지 못해 정확한 제어에 어려움을 겪었습니다. 이에 따라 새로운 해석 및 조작 프레임워크인 NeuronLens가 제안되었으며, 이는 뉴런의 활성화 분포를 더 세밀하게 분석하여 개념 귀속을 가능하게 합니다.

- **Technical Details**: NeuronLens는 뉴런의 활성화 범위를 활용하여 개별 개념에 대한 세부적인 해석을 제공하는 신개념 프레임워크입니다. 이를 통해 우리가 제안한 방식은 98.76%의 활성화 범위를 포함하는 개념별 뉴런 활성화를 추출하고, 불필요한 간섭을 줄이는 데 중점을 두고 있습니다. 실험 결과, NeuronLens는 기존 방법보다 최대 25%포인트 더 낮은 간섭을 허용하면서도 목표 개념 조작을 위한 정밀한 제어를 유지했습니다.

- **Performance Highlights**: 연구에서 NeuronLens를 기반으로 한 새로운 접근법은 encoder 및 decoder 기반 LLM들에서 여러 텍스트 분류 데이터셋을 대상으로 한 대규모 실험을 통해 효과성이 입증되었습니다. 뉴런의 활성화가 Gaussian-like 패턴을 따른다는 발견은 개념별 해석에 대한 새로운 통찰을 제공합니다. 이러한 연구는 뉴런을 통한 세밀한 개념 조작이 어떻게 수행되는지를 드러내며, 기존 방법들과 비교했을 때 기대 이상의 성능을 발휘합니다.



### Competitive Programming with Large Reasoning Models (https://arxiv.org/abs/2502.06807)
- **What's New**: 본 연구에서는 큰 언어 모델(LLM)에 강화 학습(reinforcement learning)을 적용한 결과, 복잡한 코딩 및 추론 작업에서 성능이 크게 향상되었음을 보여줍니다. OpenAI의 o1와 초기 체크포인트 o3 그리고 도메인 특화 시스템 o1-ioi를 비교 분석했으며, o1-ioi는 2024 국제 정보 올림피아드(IOI)에서 49번째 백분위수에 해당하는 성과를 거두었고, 이후 relaxed 경쟁 조건 하에 금메달을 달성했습니다. 그러나 o3 모델은 도메인 특화 전략 없이 그보다 더 뛰어난 성과를 기록했습니다.

- **Technical Details**: 경쟁 프로그래밍은 AI 시스템의 추론 능력을 평가하기 위한 도전적인 벤치마크로 인식되어 온다. 본 연구에서는 OpenAI o1을 비롯한 여러 모델이 복잡한 문제를 해결하고, 강화 학습을 통해 더 나은 결과를 도출할 수 있음을 설명하고 있습니다. o1-ioi는 2024 IOI를 목표로 설계된 핸드 엔지니어링된 전략을 사용하여 성과를 내었으며, 이러한 접근 방식은 AI 모델의 추론 능력 향상에 기여했습니다.

- **Performance Highlights**: OpenAI o1 모델은 강화 학습을 통해 코드 생성 및 문제 해결에서 효과적으로 성능을 개선했습니다. CodeForces 대회에서 시뮬레이션한 결과 o1-ioi는 C++ 프로그램을 작성하고 실행하는 데 있어 향상된 능력을 보였으며, 예외적으로 높은 성과를 달성했습니다. 이 연구는 범용적 강화 학습이 도메인 특화 기술보다 AI의 최첨단 추론 능력을 향상시키는 더 강력한 경로임을 보여줍니다.



### Logits are All We Need to Adapt Closed Models (https://arxiv.org/abs/2502.06806)
Comments:
          33 pages, 8 figures

- **What's New**: 이번 논문에서는 상업적 대형 언어 모델(LLMs)에서 토큰 로짓(token logits)에 대한 접근이 가능하다면, 콘텐츠 생성에서 특정 응용 프로그램에 맞춘 강력한 적응 기법을 사용할 수 있다는 주장을 하고 있습니다. 저자는 토큰 수준의 확률 재가중화 프레임워크를 제안하며, 이는 제한된 양의 작업 특화 데이터와 함께 로짓을 활용하여 블랙박스 LLMs를 효과적으로 조율할 수 있게 합니다.

- **Technical Details**: 이 연구는 다음 토큰 예측을 감독 분류 문제와 유사한 관점에서 바라보며, LLM의 데이터가 프록시 라벨(proxy labels) 역할을 하는 경우를 논의합니다. 레이블 노이즈 문제로 재구성할 수 있는데, 이는 특정 작업에 적합한 데이터를 참 라벨(true labels)로 간주하게 됩니다. 저자들은 이러한 접근 방식을 통해 기존 학습 데이터에 접근할 수 없는 상황에서도 LLM을 조정할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 다양한 데이터셋과 블랙박스 LLM을 대상으로 한 실험을 통해 제안된 Plugin 모델이 도메인 특정 콘텐츠 생성에서 기존 기법보다 더 우수한 성능을 보임을 입증했습니다. 또한, 저자들은 API 기초의 미세 조정 방법이 데이터 개인 정보 보호 문제로 인해 실질적으로 적용되지 못하므로, 관련된 데이터 로그를 공개할 필요성을 강조합니다.



### Solving the Content Gap in Roblox Game Recommendations: LLM-Based Profile Generation and Reranking (https://arxiv.org/abs/2502.06802)
- **What's New**: 이번 논문은 Roblox 플랫폼의 사용자 생성 콘텐츠를 분석하여 게임 추천 시스템의 품질을 개선하는 새로운 접근 방식을 제안합니다. 기존의 추천 시스템이 게임 콘텐츠의 불일치성과 희소성으로 어려움을 겪는 상황에서, 거대 언어 모델(LLMs)을 활용한 고품질 구조화된 텍스트 특징을 생성하고 검증하는 방법을 다룹니다. 이 연구는 개인화 및 사용자 만족도를 높이기 위한 LLM 기반의 재순위 메커니즘을 도입하며, 게임 속 텍스트 데이터 분석의 중요성을 강조합니다.

- **Technical Details**: 저자들은 텍스트 특징 생성을 위한 두 가지 주요 도전을 다루고 있습니다. 첫 번째로, 방대한 사용자 생성 콘텐츠에서 고품질의 구조화된 텍스트 특징을 생성하는 방법을 개발하고, 두 번째로 생성된 텍스트 특징이 추천 정확도를 향상시키는지를 검증하기 위한 프레임워크를 수립합니다. 이 과정에서는 양질의 게임 프로필 생성을 위한 LLM의 활용과 추천 시스템에서 텍스트 특징의 효용성을 평가하기 위한 재순위 기법이 포함됩니다.

- **Performance Highlights**: 제안된 방법론을 통해 Roblox의 역동적이고 사용자 중심의 에코시스템에 적합한 맞춤형 추천 시스템 구축이 가능함을 보여줍니다. LLM을 통한 텍스트 인식 및 프로필 생성이 설계되어 있어, 추천의 품질이 향상되고 사용자 경험이 증대될 것으로 기대됩니다. 이 연구는 플랫폼의 고유한 다이나믹에 적응한 스케일러블한 추천 시스템의 기초를 마련하고 있으며, 투명한 사용자 신뢰 구축에도 기여할 것입니다.



### Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning (https://arxiv.org/abs/2502.06781)
Comments:
          We released our code, data, and model on this https URL

- **What's New**: 이 논문에서는 OREAL이라는 새로운 강화 학습(Reinforcement Learning) 프레임워크를 제안합니다. 이 프레임워크는 수학적 추론 작업에서 이진 결과 보상이 쉽게 접근 가능한 환경에서 성능 한계를 추구합니다. OREAL은 부정적인 샘플의 보상을 재구성하여 긍정적인 샘플과 부정적인 샘플 간의 기울기 일관성을 확보하는 방법을 보여줍니다.

- **Technical Details**: OREAL의 본질적인 구성 요소로서, 보상 기반의 정책 최적화를 사용하는 KL-정규화된 최적 정책을 학습하기 위한 이론적 기반을 제시합니다. 이를 통해 이 모델은 긍정적인 궤적을 BoN(리스트에서 상위 N개 샘플링)을 통하여 얻어내며, 토큰 수준의 보상 모델을 통해 중요한 토큰을 샘플링하여 학습하는 방법도 포함됩니다. 이러한 기법들은 수학적 추론 작업 특성에 맞춰 설계되었습니다.

- **Performance Highlights**: OREAL은 7B 모델이 MATH-500 데이터셋에서 94.0 pass@1의 정확도를 기록하며, 32B 모델과 동등한 성능을 보여주었습니다. 또한 OREAL-32B는 기존의 32B 모델들을 초월하여 MATH-500에서 95.0 pass@1의 정확도를 달성했습니다. 결과적으로, OREAL은 이전 모델들과 비교했을 때 수학적 추론 능력을 효과적으로 향상시킵니다.



### ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates (https://arxiv.org/abs/2502.06772)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 ReasonFlux라는 새로운 계층적 LLM(대형 언어 모델) 추론 프레임워크를 소개합니다. 이 프레임워크는 500개의 고수준 사고 템플릿을 사용하여 복잡한 문제에 대한 추론 능력을 향상시키고 OpenAI o1-preview 및 DeepSeek V3 모델을 초월하는 성능을 보여줍니다. Thought templates를 효과적으로 활용함으로써 더 나은 문제 해결 과정을 가능하게 합니다.

- **Technical Details**: ReasonFlux는 세 가지 혁신적인 접근법을 도입합니다: 첫째, 고유하고 구조화된 템플릿 라이브러리를 통해 500개의 사고 템플릿을 제공합니다. 둘째, 계층적 강화 학습을 통해 LLM이 최적의 템플릿 경로를 배우게 하며, 셋째, 각 하위 문제에 대한 적절한 고수준 템플릿을 동적으로 선택하여 복잡한 추론 문제를 다룹니다.

- **Performance Highlights**: ReasonFlux-32B 모델은 MATH 벤치마크에서 91.2%의 정확도를 달성하며, OpenAI o1-preview를 6.7% 초과했습니다. USA Math Olympiad (AIME) 벤치마크에서도 평균 56.7%의 문제를 해결하며 o1-preview 및 DeepSeek-V3를 각각 27% 및 45% 초과하는 성능을 기록했습니다.



### Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs (https://arxiv.org/abs/2502.06766)
Comments:
          8 pages, 8 figures, 2 tables in main body

- **What's New**: 본 논문에서는 대규모의 입력 토큰을 처리하기 위해 최적화된 transformer 모델의 추론 방식에 대해 소개합니다. 특히, 입력 길이가 최대 1,000,000인 경우에도 저비용으로 효과적인 추론을 가능하게 하는 top-k 선택 메커니즘을 제안합니다. 이 방식은 데이터 센터 수준의 하드웨어가 아닌 일반적인 하드웨어에서도 transformer를 효과적으로 활용할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 prefill과 decoding 단계로 나누어진 transformer 모델의 추론 과정을 설명합니다. 특히, 기존의 O(N²) 메모리 비용을 줄이기 위해 유사 k-최근접 이웃 검색을 통해 가장 관련 있는 토큰만 선택하여 attention 계산을 수행하도록 설계했습니다. 이를 통해 각 층별로 가변적인 k 값을 설정하여 성능을 극대화하고, CPU 내의 벡터 데이터베이스를 활용하여 메모리 비용을 극복합니다.

- **Performance Highlights**: 실험 결과, 입력 토큰의 2% 미만만 처리함으로써 모델의 성능을 95% 이상 유지할 수 있음을 확인했습니다. 특히 LM-Eval, AlpacaEval, RULER와 같은 장기 문맥 벤치마크에서.exceptional한 결과를 보여주었습니다. 이러한 접근법 덕분에 현대 LLM은 sparsity를 활용하여 적은 GPU 메모리로도 높은 성능을 유지할 수 있음을 입증했습니다.



### Rationalization Models for Text-to-SQL (https://arxiv.org/abs/2502.06759)
- **What's New**: 이 연구는 텍스트-투-SQL(text-to-SQL) 모델 튜닝을 향상시키기 위한 Chain-of-Thought (CoT) 합리화(rationale) 생성 프레임워크를 소개합니다. 이 프레임워크는 최종 SQL 쿼리를 구성하기 위한 중간 SQL 문과 설명을 포함하며, 단계적인 접근 방식을 통해 결과를 도출합니다. 소량의 예제를 수동으로 주석 처리한 후, 대형 언어 모델을 사용해 동적 few-shot 지식 증류 절차를 진행하여 합리화 모델을 훈련합니다.

- **Technical Details**: 이 방법론은 주제 전문가가 CoT의 표현을 정의하고, (질문, 금SQL) 쌍으로 이루어진 텍스트-투-SQL 데이터셋을 사용하여 시작합니다. 자동, 수동, 반자동 주석 생성 방법을 통합하여 SQL 문 단계의 초안을 작성하고, 이를 바탕으로 Markdown을 사용하여 사용자가 이해할 수 있는 설명을 제공합니다. 마지막 SQL 쿼리를 기준으로 가장 유사한 예제를 선택하기 위해 희소 벡터 공간 모델을 구성하고, 코사인 유사도를 기준으로 SQL CoT 예제를 순위 매깁니다.

- **Performance Highlights**: BIRD 데이터셋을 사용한 실험 결과, 중간 합리화를 통해 소규모 모델의 성능이 일관되게 향상되었습니다. 특히 중간 및 높은 복잡성을 가진 쿼리에서 실행 정확도가 개선되었으며, 이를 통해 쿼리 작성 과정에 대한 해석 가능한 설명을 제공함으로써 사용자가 필요한 조치를 취할 수 있도록 지원합니다. 이는 앞으로도 다양한 텍스트-투-SQL 데이터셋에 대해 활용할 가능성이 높습니다.



### Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling (https://arxiv.org/abs/2502.06703)
- **What's New**: 이번 연구에서는 Test-Time Scaling (TTS) 기법의 효용을 심층적으로 분석하고 있습니다. Policy Models와 Process Reward Models (PRMs), 문제의 난이도가 TTS에 미치는 영향을 체계적으로 검토하여, 기존 연구의 한계를 보완하고 있습니다. 특히, 작은 LLM도 큰 LLM을 초과하는 사례를 발견하여, 다양한 모델과 난이도에서 TTS 적용의 최적화를 제안하고 있습니다.

- **Technical Details**: 연구는 Markov Decision Process (MDP) 모델을 활용하여 문제를 정의하고, 정책 모델이 생성하는 초기 행동과 보상 기능을 통해 특정 수준의 최적화된 컴퓨팅을 추구합니다. 각 실험에서는 MATH-500 및 AIME24 과제를 사용하고, 다양한 크기의 PRM과 정책 모델을 적용하여 TTS의 성능을 분석합니다. 결과적으로, TTS 전략은 사용되는 정책 모델과 PRM, 문제 난이도에 따라 크게 달라지는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, 작은 LLM이 큰 LLM보다 더 뛰어난 성능을 발휘할 수 있다는 사실이 확인되었습니다. 예를 들어, 1B LLM이 405B LLM을 초과하며, 3B LLM도 405B LLM을 뛰어넘는 성능을 보였습니다. 이러한 발견은 TTS가 LLM의 추론 능력을 향상시키는 유망한 접근 방식임을 보여줍니다.



### Multi-label Scandinavian Language Identification (SLIDE) (https://arxiv.org/abs/2502.06692)
- **What's New**: 이번 논문에서는 덴마크어, 노르웨이 보크몰어, 노르웨이 니니스커, 스웨덴어 등 밀접하게 관련된 스칸디나비아 언어들에 대한 다중 레이블 문장 수준의 언어 식별(multi-label language identification, LID) 문제에 주목합니다. 새롭게 작성된 평가 데이터셋인 SLIDE를 소개하고, 다양한 속도-정확도 무역 오프가 있는 여러 LID 모델을 제공합니다. 기존의 단일 레이블 언어 식별 방식에서 벗어나 다중 언어 식별을 가능하게 하는 방식으로 이 문제를 접근합니다.

- **Technical Details**: SLIDE 데이터셋은 수동으로 큐레이션된 다중 레이블 LID 데이터셋으로, LID 모델을 훈련시키기 위한 기초 자료로 사용됩니다. 이 데이터셋은 네 가지 언어에 대해 정확한 평가를 위한 두 가지 방법을 제공합니다: 전통적인 다중 클래스 LID 방법을 위한 하나와 다중 레이블 방법의 성능 평가를 위한 다른 하나입니다. 또한, BERT 모델을 기반으로 한 고성능 모델과 FastText 임베딩을 이용한 경량 모델을 훈련시키는 방법이 포함되어 있습니다.

- **Performance Highlights**: 이 연구의 결과는 기존 LID 시스템의 평가에서 다중 언어 인식의 필요성을 증명합니다. SLIDE 평가 데이터셋은 5%의 문장이 여러 스칸디나비아 언어에서 유효하다는 것을 나타내, 기존 시스템의 평가를 왜곡할 수 있는 예외를 포함합니다. 최상의 성능을 보이는 모델은 세분화된 BERT 모델 기반이며, 빠른 처리 속도를 자랑하는 FastText 기반 모델도 포함되어 있어 다양한 환경에서의 활용을 보여줍니다.



### Boosting Self-Efficacy and Performance of Large Language Models via Verbal Efficacy Stimulations (https://arxiv.org/abs/2502.06669)
Comments:
          to be published in ICONIP 2024

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 성능 향상에 대한 새로운 접근법을 제시합니다. Verbal Efficacy Stimulations (VES)라는 세 가지 유형의 언어 자극(Encouraging, Provocative, Critical)을 도입하여 LLM의 자기 효능감을 조사합니다. LLM의 입력에 대한 민감성을 감안할 때, 이러한 자극의 적용이 모델의 성과에 미치는 영향을 규명하고자 합니다.

- **Technical Details**: 검증을 위해 14개의 Instruction Induction 과제와 9개의 BIG-Bench Hard 과제를 대상으로 18개의 VES를 평가했습니다. 각 자극의 효능을 분석하며, 자기 효능감과 과제 성과 간의 관계를 이해하려고 노력했습니다. VES의 세 가지 형태가 LLM의 다수의 과제 성과를 향상시키고, 각 모델에 따라 최적의 VES가 다름을 발견했습니다.

- **Performance Highlights**: 실험 결과, Stretch Zone에 속하는 과제에서 성능 향상이 가장 두드러지며, Encouraging VES가 LLM의 자기 효능감을 높이고 Criticism은 반대의 효과를 나타냅니다. LLM은 Encouraging과 Provocative VES에 대해 더 적극적으로 반응하는 경향이 있으며, 이는 인간 행동과 유사한 패턴을 보입니다. 이러한 발견은 감정과 심리적 차원에서의 모델 성능 향상에 대한 새로운 통찰력을 제공합니다.



### Automatic Evaluation of Healthcare LLMs Beyond Question-Answering (https://arxiv.org/abs/2502.06666)
- **What's New**: 현재의 대형 언어 모델(LLMs) 평가는 사실성과 담화(discourse)의 측면에서 현재의 두 가지 접근법의 관계를 조명합니다. 이 논문은 Healthcare 도메인에 집중하여, open-ended와 close-ended 평가 기준의 상관관계를 탐구하며 새로운 의료 벤치마크 CareQA(케어 QA)를 소개합니다. 또한, open-ended 평가를 위한 새로운 메트릭 Relaxed Perplexity를 제안하여 기존 방법론의 한계를 완화하고자 합니다.

- **Technical Details**: 이 연구는 9개의 서로 다른 데이터셋을 활용한 4가지 close-ended 의료 작업과 9개의 다양한 데이터셋 기반의 6개 open-ended 작업을 고려합니다. CareQA는 스페인 보건부의 MIR 시험을 기초로 하여, 여러 카테고리에서 감사를 요구하는 5,621개의 QA 쌍을 포함하고 있으며, 영어와 스페인어로 제공됩니다. 이를 통해 각 작업의 일관성을 평가하고, 각 메트릭을 통해 모델의 성능을 측정하는 다양한 접근을 연구합니다.

- **Performance Highlights**: 실험에서는 MCQA 벤치마크와 다양한 open-ended 및 close-ended 작업 간의 상관관계를 분석한 결과, 임상 노트 작성만 MCQA와 약한 양의 상관관계를 보이는 것으로 나타났습니다. 그 외에 요약, 질문 함의와 같은 여러 작업은 MCQA와 부정적 상관관계를 보이며, 이는 의료 전문 지식의 필요성이 떨어지는 경우 때문입니다. 이러한 결과는 벤치마크 선택의 중요성과 또한 각각의 작업에 대한 특화된 평가의 필요성을 강조합니다.



### Who Taught You That? Tracing Teachers in Model Distillation (https://arxiv.org/abs/2502.06659)
Comments:
          Preprint; under review

- **What's New**: 이 연구는 대형 'teacher' 모델의 알고리즘을 활용해 작은 'student' 모델의 교육 과정을 분석합니다. 모델 증류(model distillation)에서 학생 모델의 출력 내용으로부터 교사를 추론하는 방법이 제안되었습니다. 제안된 방법은 추론된 정보를 통해 매우 구체적인 기능을 대형 모델에서 소형 모델로 증류할 때의 윤리적 함의를 다룹니다. 이 연구는 특히 정보 요약, 질문 응답 및 지시 수행과 같은 과제를 대상으로 하며, 이로 인해 대형 모델의 서비스 약관 위반 가능성도 고찰하고 있습니다.

- **Technical Details**: 이 연구에서는 여러 대형 LLM(대형 언어 모델)으로부터 작은 모델을 훈련시키는 과정에서 학생 모델의 출력을 통해 교사 모델을 식별하는 데 중점을 두었습니다. 사용된 방법론으로는 perplexity, 텍스트 유사성, 구문 패턴(Part-of-Speech) 통계 등이 있으며, 장황한 텍스트 유사성만으로는 교사를 신뢰성 있게 식별할 수 없음을 입증했습니다. 특히, PoS 템플릿이 교사를 식별하는 데 강력한 신호를 제공함을 발견했으며, 이 방법이 n-gram 기반 측정보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 학생 모델이 생성한 출력물을 기반으로 교사 모델을 식별하는 과정에서 PoS 템플릿을 이용한 방법이 대체로 n-gram 방법보다 더 잘 작동하는 것으로 나타났습니다. 예를 들어, PubMed 데이터셋에서 PoS 템플릿 기반 분류기는 0.68의 정확도를 달성하는 반면, n-gram 모델은 0.61에 그쳤습니다. 이 연구의 결과는 PoS 패턴이 교사 모델 분류에 있어서 중요한 특징으로 작용함을 강조하며, 기존의 단순한 문자 기반 표현을 넘어서 신뢰할 수 있는 분류 지표로 증명되었습니다.



### In-Context Learning (and Unlearning) of Length Biases (https://arxiv.org/abs/2502.06653)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 연구에서는 긴 길이 편향(length bias)이 인컨텍스트 학습(in-context learning, ICL)에 미치는 영향을 조사하였습니다. 모형이 예측을 위해 사용하는 컨텍스트 윈도우 내에서 길이 편향을 학습하는 능력을 보였으며, 이를 통해 기존의 모델에 존재하는 길이 편향을 완화할 수 있는 가능성도 제시하였습니다. 이는 비싼 매개변수 업데이트 없이 모델의 예측 행동을 수정할 수 있는 방법을 보여줍니다.

- **Technical Details**: 이 연구에서는 다양한 LLM(대형 언어 모델) 가족들이 인컨텍스트 방식으로 길이 편향을 학습하는 능력을 실험적으로 입증하였습니다. 또한, 모델의 매개변수 수, 예제 수, 클래스 길이 차이가 모델이 어떻게 편향을 학습하는지에 미치는 영향을 분석하였습니다. 실험은 문장 길이가 다양한 7777개의 이진 분류 데이터셋을 사용해 진행되었으며, 다양한 매개변수 크기의 모델을 평가하였습니다.

- **Performance Highlights**: 결과적으로, ICL은 모델에 편향을 도입하여 작업 성능에 부정적인 영향을 미칠 수 있음을 보여주었습니다. 길이 정보가 포함된 학습 방식은 기존 모델의 길이 편향을 제거하는 데 유효함이 입증되었습니다. 이러한 결과는 인컨텍스트 학습이 모델의 성능을 향상시키는 데 중요한 역할을 할 수 있음을 시사합니다.



### Transparent NLP: Using RAG and LLM Alignment for Privacy Q&A (https://arxiv.org/abs/2502.06652)
Comments:
          Submitted to ARR

- **What's New**: 이번 논문은 GDPR (General Data Protection Regulation)의 투명성 원칙을 준수하기 위한 Retrieval Augmented Generation (RAG) 시스템을 조사합니다. 특히, Rewindable Auto-regressive Inference (RAIN)과 MultiRAIN 같은 정렬 모듈을 통합하여 데이터 처리 질문에 대한 응답의 정확성과 이해 가능성을 향상시키는 방법을 제시합니다. 연구 결과, 정렬 모듈을 포함한 RAG 시스템이 기존 RAG 시스템보다 더 좋은 성능을 보였지만, 인간의 답변에는 미치지 못하는 경향이 있음을 확인했습니다.

- **Technical Details**: RAG 시스템의 기본 구조는 대량의 데이터를 검색하여 처리하는 것으로, RAIN 알고리즘을 통해 정렬 기능을 추가합니다. RAIN은 사용자의 질의로부터 트리 구조를 형성하고, 생성된 응답의 정확성과 이해 가능성을 평가하여 최적의 단어 조합을 선택합니다. 논문에서는 또한 MultiRAIN이라는 새로운 다차원 최적화 방법을 제안하며, 이를 통해 응답의 정확성과 이해 가능성을 동시에 개선하려고 합니다.

- **Performance Highlights**: 실험 결과, RAG 시스템과 RAIN은 다양한 측정 기준에 따라 향상된 성능을 나타냈습니다. 평가 과정에서 21가지의 최신 NLP 관련 메트릭스가 사용되었으며, 각각의 응답이 어떻게 투명성과 정확성을 충족하는지를 자세히 분석하였습니다. 그러나 인간의 답변과는 여전히 차이가 있어, 향후 투명성 요구사항을 완전히 충족하기 위한 추가적인 연구가 필요하다는 점이 강조되었습니다.



### Steel-LLM:From Scratch to Open Source -- A Personal Journey in Building a Chinese-Centric LLM (https://arxiv.org/abs/2502.06635)
- **What's New**: Steel-LLM은 중국어 중심의 언어 모델로, 제한된 컴퓨팅 자원 내에서 처음부터 개발되었습니다. 이 모델은 2024년 3월에 출시되었으며, 10억 개의 파라미터를 가진 모델을 대규모 데이터셋을 기반으로 교육하려고 합니다. 프로젝트의 목표는 투명성과 실제적인 통찰을 공유하여 커뮤니티의 다른 구성원에게 도움을 주는 것입니다.

- **Technical Details**: Steel-LLM은 Transformer 아키텍처를 기반으로 하며, Flash Attention과 Soft Mixture of Experts(Soft MOE)를 통합하여 성능 최적화를 이루었습니다. Flash Attention은 모델의 훈련 및 추론 효율성을 향상시키고 GPU 메모리를 절약하는 기능을 제공합니다. 또한, Steel-LLM은 8개의 GPU에서 제한된 자원으로 훈련되었으며, 모델 구조는 1억 개의 파라미터를 가진 소규모 언어 모델입니다.

- **Performance Highlights**: Steel-LLM은 CEVAL 및 CMMLU와 같은 벤치마크에서 경쟁력 있는 성능을 보여주며, 더 큰 기관의 초기 모델들을 초월했습니다. 모델의 개발 과정에서의 투명성을 제공하고 훈련 프로세스의 최적화에 대한 실제적 가이드를 제공하여 소규모 연구팀과 개인 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Scaling Multi-Document Event Summarization: Evaluating Compression vs. Full-Text Approaches (https://arxiv.org/abs/2502.06617)
Comments:
          NAACL 2025 camera-ready version

- **What's New**: 본 연구에서는 대규모 다문서 요약(multi-document summarization, MDS)에서 압축 기반(compression-based) 및 전체 텍스트(full-text) 시스템의 두 가지 방식을 비교합니다. 이들 각각의 접근 방식은 특정 문서 집합의 요약을 생성할 때 아키텍처와 정보 손실(lossy summaries)에 있어 차이를 보이며, 특히 최근의 LLM(long language models)을 활용한 긴 문맥 추론(long-context reasoning)에서 그 가능성을 탐구합니다. 실험을 통해 이 두 시스템의 장단점을 검증하고, 최적의 성능을 위해 혼합 접근법(hybrid methods)의 필요성을 강조합니다.

- **Technical Details**: 세 가지 데이터셋(SummHay, Background, WCEP)을 사용하여 다양한 다문서 요약 작업을 평가했습니다. 연구는 자연어 처리(NLP) 및 마켓에서 널리 사용되는 LLM에 대한 긴 문맥(reasoning) 처리 능력을 기반으로 하여 정보 추출 및 요약 성능을 분석하는 데 중점을 두었습니다. 특히, 각 방법론에 따라 다른 문서 구조 및 내용 단위를 처리하는 방식을 제시하였고, A3CU(Atomic Content Unit) 메트릭을 활용하여 정보 보존 정보를 평가했습니다.

- **Performance Highlights**: 실험 결과, 전체 텍스트 및 검색 기반(retrieval-based) 시스템이 대부분 설정에서 가장 우수한 성능을 보였습니다. 압축 기반 방법은 중간 단계에서 높은 정보 보존률을 나타냈으나 다단계 파이프라인으로 인해 정보 손실이 발생하며, 결국 전체 컨텍스트 시스템보다 낮은 성과를 나타냈습니다. 따라서 본 연구는 다문서 요약을 위해 압축 기반 및 전체 텍스트 시스템의 장점을 조합한 혼합 접근법을 개발할 필요성을 제기합니다.



### Do we really have to filter out random noise in pre-training data for language models? (https://arxiv.org/abs/2502.06604)
- **What's New**: 이 논문은 인터넷에서 수집된 텍스트 데이터에 포함된 임의의 노이즈(random noise)의 영향을 최초로 체계적으로 조사합니다. 대부분의 이전 연구는 저품질 데이터 또는 합성 데이터에 초점을 맞추었지만, 저자들은 'What-Why-How' 프레임워크를 사용하여 이러한 임의의 노이즈를 분석합니다. 특히 이들은 다음 토큰 예측(NTP) 손실의 증가가 노이즈 비율보다 훨씬 낮다는 점을 발견했습니다.

- **Technical Details**: 저자들은 OpenWebText 데이터셋을 기반으로 다수의 GPT-2 모델을 사전 훈련했으며, 1%, 5%, 20% 비율로 생성된 임의의 정수를 사용하여 노이즈를 시뮬레이션했습니다. 흥미롭게도, 20%의 데이터에 노이즈가 포함되어 있어도 손실이 약 1%만 증가했으며, 이로 인해 다국어 모델의 성공에 대한 통찰을 제공합니다. 또한, 새로운 Local Gradient Matching (LGM) 손실을 도입하여 노이즈의 부정적 영향을 줄이는 방법을 제시했습니다.

- **Performance Highlights**: LGM 손실을 적용했을 때, 8개 언어 및 14개 비전 데이터셋에서 예상치 못한 정확도 향상이 관찰되었습니다. 랜덤 노이즈가 데이터셋에 존재함에도 불구하고, 모델의 성능이 저하되지 않고 오히려 향상될 수 있는 가능성을 보여줍니다. 이 연구는 LLM의 사전 훈련 데이터의 질에 대한 중요한 통찰을 제공하며, 실제 적용 사례에서도 유용성을 입증합니다.



### Evaluation of Multilingual Image Captioning: How far can we get with CLIP models? (https://arxiv.org/abs/2502.06600)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 연구는 다국어 이미지 캡션 평가에서 CLIPScore 메트릭의 새로운 활용 방법과 실험 결과를 제시하고 있습니다. 특히, 기존의 영어 중심적인 접근 방식을 넘어 다국어 환경에서의 평가를 위한 전략을 탐색합니다. 연구에서는 기계 번역된 데이터와 다국어 데이터셋을 활용하여 평가 모델을 개선했습니다.

- **Technical Details**: 연구에서는 Multilingual CLIP 모델의 성능을 높이기 위해 두 가지 주요 데이터셋, CrossModal-3600과 VICR을 활용하는 방법을 제안합니다. 이들 데이터셋은 문화적 다양성을 고려하여 모델을 파인튜닝(finetuning)하는 데 필수적입니다. 사용된 손실 함수는 CLIP ‘contrastive loss’로, 다국어와 다문화 자원을 효율적으로 처리하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, 다국어 CLIPScore 모델은 다양한 언어에서 인간 평가와 높은 상관관계를 유지하며, 영어 평가에서도 동등하거나 더 나은 성능을 보였습니다. 이는 다국어 모델이 단일 언어 모델보다 더 다재다능하고 가치 있는 자원임을 입증하며, 문화적 및 언어적 다양성을 반영하는 평가 방법이 필요함을 강조합니다.



### Hephaestus: Improving Fundamental Agent Capabilities of Large Language Models through Continual Pre-Training (https://arxiv.org/abs/2502.06589)
Comments:
          Accepted to NAACL 2025 main conference

- **What's New**: 본 논문에서는 LLM 기반 자율 에이전트의 핵심 능력을 강화하기 위해 설계된 최초의 대규모 프리트레인(pre-training) 데이터셋인 Hephaestus-Forge를 소개합니다. Hephaestus-Forge는 103B의 에이전트 특화 데이터를 포함하며 76,537개의 API를 아우르고 있습니다. 또한, API 기능 호출, 내재적 추론(intrinsic reasoning), 계획(planning) 및 환경 피드백에 대한 적응 능력을 향상시키기 위한 솔루션을 제공합니다.

- **Technical Details**: 이 연구에서는 LLM 에이전트의 능력을 근본적으로 향상시키기 위한 두 가지 주요 목표를 설정하고 있습니다: (a) 개별 API 함수 호출의 이해도를 향상시키고, (b) 여러 함수 호출이 필요한 문제 해결을 위한 내재적 추론 능력을 강화하는 것입니다. 이러한 목표를 달성하기 위해, Tool 문서화와 함수 호출 경로를 대규모 데이터셋으로 수집하였고, 이를 통해 Hephaestus-Forge라는 고품질의 훈련 코퍼스를 구성하였습니다.

- **Performance Highlights**: Hephaestus는 지속적 프리트레인을 통해 소규모 및 중간 규모의 오픈소스 LLM을 초월하여 상업용 LLM과 동등한 성능을 발휘합니다. 연구 결과, Hephaestus-8B는 여러 에이전트 벤치마크에서 약 9.6%에서 17.6%까지 성능 향상을 보였으며, 상업용 LLM인 Claude-3-Haiku 및 GPT-3.5-turbo와 비교하여도 높은 성능을 기록했습니다. 이를 통해 Hephaestus-Forge의 효과성을 입증하고 있습니다.



### LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM (https://arxiv.org/abs/2502.06572)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 법적 추론을 위한 데이터 생성을 위해 KgDG라는 지식 기반 데이터 생성 프레임워크를 제안합니다. 이 프레임워크는 법적 지식을 활용하여 데이터 생성을 다양화하고, 생성된 데이터의 품질을 보장하기 위한 정제 및 검증 과정을 소개합니다. 이를 통해 기존의 오픈 소스 LLM들의 성능을 개선하려고 하며, 50,000개의 질 높은 샘플을 포함하는 합성 데이터셋을 생성했습니다.

- **Technical Details**: KgDG 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Knowledge-Guide Generation (KgGen), (2) Knowledge-Guide Fixer (KgFix), (3) Data Verifier (DaVer). KgGen은 법적 지식 기반을 활용하여 다양한 데이터를 생성하고, KgFix는 잘못된 참조와 논리 경로를 수정하며, DaVer는 수정 불가능한 데이터를 필터링하여 생성 품질을 보장하는 역할을 합니다. 또한 Mixture Training (MiTra) 전략을 통해 생성된 데이터셋을 확대하여 LLM의 추론 능력을 향상시킵니다.

- **Performance Highlights**: LawGPT 모델은 기존의 법률 전용 LLM들보다 뛰어난 성능을 보이며, 독점 LLM들과 비교할 만한 결과를 달성했습니다. 이는 KgDG와 LawGPT의 효율성을 입증하며, 법적 추론 분야에서 기존 솔루션들보다 우수한 성능을 발휘함을 보여줍니다. 이 연구의 데이터셋과 모델은 향후 연구를 위해 공개될 예정입니다.



### Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation (https://arxiv.org/abs/2502.06563)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이번 논문에서는 ProverGen이라는 새로운 프레임워크를 제안하여, 대형 언어 모델(LLM)과 기호 증명기(symbolic prover)의 강점을 결합함으로써 다기능적인 FOL(First-Order Logic) 추론 데이터셋인 ProverQA를 생성하는 방법을 소개하고 있습니다. ProverQA는 접근 가능하고 논리적으로 일관된 중간 추론 단계가 포함되어 있으며, LLM들이 문제 해결에 어려움을 겪고 있음을 보여줍니다. 이를 통해 데이터 생성 프레임워크의 효용성을 강조하고 있습니다.

- **Technical Details**: ProverGen 프레임워크는 3단계의 파이프라인으로 구성되어 있습니다. 첫째로, LLM을 사용하여 각 문제에 대한 고유한 배경 이야기를 생성하고, 둘째로 기호 증명기인 Prover9를 이용해 추론 트리를 구축합니다. 마지막으로, LLM을 활용하여 이러한 논리 표현을 자연어 문장으로 변환합니다. 이 과정을 통해 ProverQA 데이터셋은 쉽게 확장 가능하며 다양한 난이도를 지원합니다.

- **Performance Highlights**: ProverQA의 문제들은 난이도가 조정되어 있어 특히 고난이도 세트에서 state-of-the-art LLM도 50%를 초과하는 정확도를 기록하지 못하고 있습니다. Llama3.1-8B-Instruct 모델을 ProverQA 훈련 세트로 파인튜닝한 결과, 잘 수행되지 않았던 기존 FOL 훈련 세트에 비해 많은 성능 향상을 보여주었습니다. 이러한 성과는 우리가 제안한 데이터 생성 프레임워크의 가치를 잘 드러내고 있습니다.



### Position: It's Time to Act on the Risk of Efficient Personalized Text Generation (https://arxiv.org/abs/2502.06560)
- **What's New**: 최근 오픈 소스 Generative AI 텍스트 모델들이 발전하면서 개인 맞춤형 모델 생성이 가능해졌습니다. 이러한 모델은 특정 개인의 필요에 맞추어 텍스트를 생성하고 해당 개인의 글쓰기 스타일을 모방할 수 있도록 훈련될 수 있습니다. 그러나 이러한 발전은 악의적인 공격자가 특정 개인을 대규모로 모방할 수 있는 새로운 안전 위험을 초래할 수 있습니다.

- **Technical Details**: Generative AI는 프롬프트에 따라 비디오, 이미지, 소리 또는 텍스트를 생성할 수 있는 신경망 기반 시스템입니다. 최근에는 LLM(대규모 언어 모델) 개인화와 효과적인 파라미터 조정 기법이 발전하여, 개인 사용자의 스타일을 정확하게 모방할 수 있게 되었습니다. 특히, 구조적 및 훈련 발전이 결합되어 10억 개 미만의 파라미터를 갖는 고성능 언어 모델이 가능해졌습니다.

- **Performance Highlights**: 높은 품질의 오픈 소스 모델들이 개인 사용자가 저렴하게 또는 무료로 접근할 수 있게 하여, 개인의 데이터를 활용해 인퍼런스를 실행하거나 모델을 맞춤 설정할 수 있는 기회를 제공합니다. LLM 개인화는 사용자에게 맞는 텍스트 출력을 제공할 수 있는 가능성을 가지고 있으나, 이러한 사용이 악용될 경우 발생할 수 있는 위험을 간과해서는 안 됩니다.



### Efficient Scientific Full Text Classification: The Case of EICAT Impact Assessments (https://arxiv.org/abs/2502.06551)
- **What's New**: 본 연구는 효율적인 과학 논문 분류를 위한 전략을 탐구하며, 여기에는 작은 BERT 기반 모델과 Llama-3.1 8B와 같은 로컬 대형 언어 모델이 포함됩니다. 우리는 입력 문장의 하위 집합을 선택하는 방법을 개발하여 입력 크기를 줄이는 동시에 분류 성능을 향상시키는 데 집중합니다. 특히, 침입 생물학 분야의 논문을 포함하는 새로운 데이터 세트를 구성하여 연구자들이 평가한 해양 생물의 영향과 일치하도록 하였습니다.

- **Technical Details**: 이 연구에서는 EICAT 데이터 세트를 소개하고, BERT 기반 분류기를 사용하여 성능을 평가하며, ModernBERT 및 Llama-3.1 8B와 같은 최신 대형 모델과 비교합니다. 모델의 입력 크기를 줄이기 위해 문장 선택기를 훈련시키며, 이 과정에서 다양한 문장 선택 전략을 테스트하여 분류 성능을 향상시킵니다. 인간 증거 주석, LLM 생성 주석 및 중요도 점수를 활용하여 선택 모델을 훈련하는 방법도 모색하였습니다.

- **Performance Highlights**: 우리는 접수된 데이터를 활용하여 여러 문장 선택 전략이 분류 성능을 개선하고 효율성을 높이는 데 기여함을 보여줍니다. 특히, 짧은 입력의 반복 샘플링이 분류 성능을 더욱 향상시킬 수 있는 유효한 방법임을 발견했습니다. 이러한 결과는 과학 전체 텍스트 분류의 추론을 가속화하고 성능을 개선할 수 있는 일반화 가능한 파이프라인을 제공합니다.



### Ignore the KL Penalty! Boosting Exploration on Critical Tokens to Enhance RL Fine-Tuning (https://arxiv.org/abs/2502.06533)
Comments:
          11 pages, 6 figures, 5 tables. Accepted for publication in the Findings of the North American Chapter of the Association for Computational Linguistics (NAACL) 2025

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)이 장기 목표를 달성하는 능력을 향상시키기 위해 강화 학습(RL)으로 미세 조정할 때, 탐색 동역학을 조사한 내용을 다룹니다. 특히, 'critical tokens'라고 불리는 몇몇 결정적 토큰이 모델의 성능에 미치는 영향을 분석하며, KL 발산 패널티의 변형을 도입하여 이들 토큰에 대한 탐색을 유도하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 간단한 산술 작업을 통해 사전 학습(pre-training)과 RL 미세 조정(fine-tuning) 단계 간의 배포 변화(distribution shift)가 언어 모델의 탐색 성능에 미치는 영향을 살펴봅니다. 모델은 두 개의 숫자의 덧셈을 수행하며, 각 상태(state)는 생성된 텍스트로 표현되고 보상 함수(reward function)는 최종 결과가 올바른 경우에만 1의 값을 가집니다. 이 과정에서 KL 발산 발란스 조정이 적용되며, 이는 사전 학습 모델의 신뢰도에 따라 결정됩니다.

- **Performance Highlights**: 결과적으로, KL 패널티 수정이 모델의 탐색 효율성을 현저히 향상시키는 것을 보여줍니다. 실험 결과, 사전 학습에 사용된 피연산자 길이의 다양성이 새로운 피연산자 길이에 대한 성능에 얼마나 영향을 미치는지를 규명하였으며, 이는 LLM의 실제 응용 분야에서의 가능성을 확장하는 중요한 발견으로 해석됩니다. 이러한 연구는 LLM이 기존의 미세 조정 모델의 한계를 넘어, 보다 효과적으로 복잡한 문제를 해결할 수 있는 방향을 제시합니다.



### GuideLLM: Exploring LLM-Guided Conversation with Applications in Autobiography Interviewing (https://arxiv.org/abs/2502.06494)
Comments:
          31 pages; the first three authors contributed equally

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 주도하는 대화의 잠재력을 탐구합니다. LLM 주도 대화의 세 가지 기본 구성 요소인 목표 탐색(Goal Navigation), 맥락 관리(Context Management), 공감적인 참여(Empathetic Engagement)를 정의하고 GuideLLM이라는 새로운 모델을 제안합니다.

- **Technical Details**: GuideLLM은 다양한 주제를 포함하는 인터뷰 환경을 설정하여 LLM 주도 대화의 질을 평가합니다. 이 환경에서 약 1,400번의 발화와 184,000개의 토큰이 생성되었고, 200개 이상의 이벤트가 언급되었습니다. 여러 최신 LLM 모델과 비교 분석하여 GuideLLM의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, GuideLLM은 자동 평가에서 기존 LLM보다 현저히 우수한 성과를 보여주었으며, 인간 평가에서도 일관되게 높은 점수를 기록했습니다. 특히, 자서전 생성 품질과 대화 품질 측면에서 두드러진 성과를 보였습니다.



### Adaptive Prompting: Ad-hoc Prompt Composition for Social Bias Detection (https://arxiv.org/abs/2502.06487)
Comments:
          Accepted to NAACL 2025

- **What's New**: 최근의 instruction fine-tuning 기술의 발전으로 인해 대규모 언어 모델에 적합한 다양한 prompting 기법이 개발되었습니다. 하지만 이러한 기법들은 작업(task), 언어 모델(language model), 주어진 컨텍스트(context)와 같은 여러 매개변수에 따라 성공 여부가 달라지며, 효과적인 prompt를 찾는 과정은 종종 시행착오를 필요로 합니다. 이 논문에서는 주어진 입력에 최적의 prompt 구성을 예측하는 adaptive prompting 접근 방식을 제안합니다.

- **Technical Details**: 우리는 세 가지 대규모 언어 모델과 함께 social bias detection을 위한 다섯 가지 prompting 기법 및 조합을 평가합니다. 각 문서 인스턴스에 대한 최적의 prompt 구성은 encoder 모델이 개별 prompting 기법의 풀을 기반으로 학습하여 예측됩니다. 결과적으로 이 접근법은 각 인스턴스에 대해 가장 좋은 구성을 예측하고 이를 따라 LLM을 prompting 합니다.

- **Performance Highlights**: 우리의 연구 결과는 adaptive prompting 접근 방식이 높은 분류 성능을 확 consistently 보장하며, 여러 설정에서 기존의 모든 기준선과 고정된 구성 조합을 초과하는 경우가 많다는 것을 시사합니다. 또한, 이 기법은 social bias detection뿐만 아니라 다른 NLP 작업에도 일반화 가능성을 지니고 있습니다.



### KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichmen (https://arxiv.org/abs/2502.06472)
Comments:
          24 pages, 3 figures, 2 tables

- **What's New**: KARMA는 다중 에이전트 기반의 새로운 지식 그래프(KG) 보강 프레임워크로, 비구조화된 텍스트의 구조적 분석을 통해 KG의 자동화된 보강을 목표로 합니다. 이 시스템은 엔티티 발견, 관계 추출, 스키마 정렬, 충돌 해결 등을 수행하는 아홉 개의 협력적인 에이전트를 사용하여 기존의 KG에 새로운 지식을 통합합니다. KARMA의 접근 방식은 도메인별 스키마를 준수하면서 관계의 정확성을 높이고, 문서의 지식을 검증하며, 비구조화된 텍스트에서 고품질 지식을 효율적으로 추출할 수 있도록 설계되었습니다.

- **Technical Details**: KARMA는 계층적 다중 에이전트 시스템으로, 각 에이전트는 KG 보강 파이프라인에서 특정 작업을 처리하기 위해 전문화된 LLM(대형 언어 모델)을 활용합니다. 기존 KG는 엔티티 집합(V)과 관계 집합(E)으로 구성되며, 비구조화된 문서에서 새로운 관계 트리플(t) 을 자동으로 추출하여 이를 KG에 통합하는 구조로 이루어져 있습니다. 이러한 프로세스는 문서 수집부터 KG 통합까지의 모듈화된 하위 과제로 분리되며, 각 에이전트는 독립적으로 고유한 작업을 수행합니다.

- **Performance Highlights**: KARMA의 효과를 확인하기 위한 실험은 1,200개의 PubMed 논문을 대상으로 진행되었으며, 최대 38,230개의 새로운 엔티티를 식별하는 성과를 이루었습니다. 이 과정에서 LLM으로 검증된 정확도는 83.1%에 달했으며, 다층 평가를 통해 충돌 엣지를 18.6% 감소시켰습니다. 이러한 성과는 KARMA가 KG 보강에서 정확성과 확장성을 동시에 달성할 수 있는 가능성을 보여줍니다.



### A Survey of Theory of Mind in Large Language Models: Evaluations, Representations, and Safety Risks (https://arxiv.org/abs/2502.06470)
Comments:
          Advancing Artificial Intelligence through Theory of Mind Workshop, AAAI 2025

- **What's New**: 이 논문은 Theory of Mind (ToM) 연구를 통해 대형 언어 모델(LLMs)이 다른 사람의 정신 상태를 이해하고 행동을 예측하는 능력을 가지고 있다는 점에 주목하고 있습니다. 특히, LLM의 ToM 기능이 높은 수준으로 발달하면 개인 정보 침해 및 집단의 불일치와 같은 안전 문제를 초래할 수 있음을 강조합니다. 향후 연구는 이러한 리스크를 효과적으로 평가하고 완화할 방법을 찾는 데 중점을 두어야 합니다.

- **Technical Details**: 최근 연구에 따르면, LLM인 GPT-4는 특정 ToM 작업에서 7세에서 10세 어린이 또는 성인과 유사한 성능을 보입니다. 그러나 BigToM, FANToM, OpenToM 등과 같은 LLM 전용 벤치마크에서는 대다수의 모델이 인간보다 성능이 떨어진다고 합니다. LLM의 내부 표현은 ToM 기능에 긍정적인 영향을 미치며, 더 큰 모델에서 이러한 표현의 정확도가 증가하는 것으로 나타났습니다.

- **Performance Highlights**: 고급 ToM 기능은 사회적 과학 시뮬레이션과 같은 유익한 응용 프로그램을 가능하게 하지만, 동시에 개인정보 침해 및 정교한 속임수와 같은 위험도 동반합니다. 연구에 따르면, LLM이 고급 ToM을 활용하여 개인의 신념, 선호 및 경향성을 추출하고, 이를 통해 맞춤형 허위 정보 캠페인을 벌이는 가능성이 있습니다. 따라서 LLM의 ToM 진전을 면밀히 모니터링하고 평가하는 것이 중요합니다.



### Beyond Literal Token Overlap: Token Alignability for Multilinguality (https://arxiv.org/abs/2502.06468)
Comments:
          Accepted to NAACL 2025

- **What's New**: 본 논문에서는 다국어 토큰화(multilingual tokenization)의 품질과 영향을 이해하기 위한 새로운 방법으로 서브워드 토큰 정렬 가능성(subword token alignability)을 제안합니다. 기존의 문자 토큰 중복(token overlap) 개념은 다양한 스크립트를 가진 언어 쌍에 대해 유의미한 설명력이 부족하지만, 서브워드 토큰 정렬 가능성을 통해 이러한 한계를 극복할 수 있습니다. 이 방법론은 특히 문자 겹침이 낮고 스크립트 간에 큰 차이가 있는 경우에도 효과적으로 다국어성을 예측할 수 있도록 돕습니다.

- **Technical Details**: 서브워드 토큰 정렬 가능성은 통계적 단어 정렬 도구에서 파생된 두 종류의 토큰 정렬 가능성 점수를 사용하여 평가됩니다. 이는 방향성에 따라 비대칭적일 수 있으며, 설정된 테스트 코퍼스(FLORES-200)를 기반으로 하여 양 방향에서 토큰 정렬을 측정합니다. 본 연구는 여러 다국어 토큰화 기법(BPE, Unigram, TokMix)을 통해 소규모 인코더 모델을 분석하고, 각 모델에서의 토큰 정렬 가능성과 하류 전이 성능(downstream transfer performance) 간의 상관관계를 조사했습니다.

- **Performance Highlights**: 서브워드 토큰 정렬 가능성 점수는 분포 겹침(distributional overlap) 지표보다 하류 전이 성능을 더 잘 예측하는 것으로 나타났습니다. 또한, 대규모 사전 학습(pre-training) 데이터가 전이 성능에 미치는 영향을 논의하며, 다양한 모델 구조와 함께 토큰 정렬 가능성을 평가했습니다. 이러한 통찰력은 향후 다국어 토큰화 기법을 개선하는 데 활용될 수 있습니다.



### Systematic Outliers in Large Language Models (https://arxiv.org/abs/2502.06415)
Comments:
          Accepted at ICLR 2025. Project Page: this https URL

- **What's New**: 이 논문에서는 Large Language Models (LLMs) 내의 outlier(이상치) 유형을 정의하고 분류하는 새로운 접근 방식을 제공합니다. 세 가지 유형인 activation outliers, weight outliers, attention outliers를 제시하며, 이들 사이의 고유한 연결성과 주의 메커니즘에 미치는 영향을 탐구합니다. 또한 이 연구는 outliers가 self-attention mechanism의 softmax operation에서 발생하며, 이들이 주의 메커니즘 내에서 문맥을 인식하는 스케일링 요인으로 작용한다는 점을 밝힙니다.

- **Technical Details**: 연구는 activation, weight, attention outliers의 수학적 정의를 통해 각 유형의 존재를 세밀하게 분석합니다. LLaMA2-7B 모델에서 outliers의 분포 위치를 제시하며, 이들의 시스템적 특성을 설명합니다. 이 논문은 이론적 추정과 실험을 통해 outliers가 주의 메커니즘에 미치는 영향을 규명하며, 이를 통해 훈련 과정의 수렴 속도를 높일 수 있는 방법을 제안합니다.

- **Performance Highlights**: 이 연구는 outliers를 구조적으로 제거함으로써 LLM의 수렴 속도와 모델 압축을 개선할 수 있음을 보여줍니다. 이전 연구에서 outliers가 모델 성능을 저하시키는 요인으로 지적된 반면, 이 논문에서는 그 존재를 이해함으로써 효과적인 최적화 방안을 제시합니다. 따라서, outliers의 체계적 처리는 LLM의 성능과 효율성을 동시에 향상시키는 중요한 실마리가 될 수 있습니다.



### SynthDetoxM: Modern LLMs are Few-Shot Parallel Detoxification Data Annotators (https://arxiv.org/abs/2502.06394)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 이 논문에서는 다국어 텍스트의 Detoxification을 위한 병렬 데이터 세트를 생성하는 파이프라인을 소개합니다. 또한, 16,000개의 고품질 Detoxification 문장 쌍으로 구성된 SynthDetoxM 데이터 세트를 공개했습니다. 이 데이터 세트는 독일어, 프랑스어, 스페인어, 러시아어를 포함하며, 다양한 독성 평가 데이터 세트에서 수집하여 현대의 오픈소스 LLM을 사용해 다시 작성하였습니다. 이 연구는 다국어 텍스트 Detoxification의 데이터 부족 문제를 해결하기 위한 새로운 접근법을 제시합니다.

- **Technical Details**: SynthDetoxM 데이터 세트는 독일어, 프랑스어, 스페인어, 러시아어의 독성 문장을 수집하여 LLM으로 재작성한 것입니다. 연구자들은 여러 개의 LLM을 활용하여 다양한 모델의 생성을 통합하고, 수작업으로 작성된 휴리스틱을 사용하여 최상의 응답을 조합했습니다. 이 과정에서, 각 모델의 응답은 텍스트 유사성, 스타일 강도 및 유창성을 기준으로 평가하여 선택되었습니다.

- **Performance Highlights**: 실험 결과, SynthDetoxM에서 학습된 모델이 인간이 주석을 단 MultiParaDetox 데이터 세트에서 학습된 모델보다 우수한 성능을 발휘했습니다. 특히, 기존의 모델들보다 높은 성능을 나타내었으며, 데이터가 제한된 환경에서도 강력한 성능을 보였습니다. 이 데이터 세트와 코드가 공개됨으로써 다국어 텍스트 Detoxification 연구에 크게 기여할 것으로 기대됩니다.



### The exponential distribution of the orders of demonstrative, numeral, adjective and noun (https://arxiv.org/abs/2502.06342)
- **What's New**: 이번 연구에서는 demonstrative, numeral, adjective 그리고 noun으로 구성된 명사구의 선호 순서의 분포를 조사하였습니다. 24가지 가능한 순서의 실제 분포를 분석하고, 기존의 파워 로우 분포 대신 지수 분포가 훨씬 더 나은 모델임을 발견했습니다. 이는 언어 특정 규칙에 대한 기존의 관점을 도전하는 중요한 결과입니다.

- **Technical Details**: 연구에서는 24개의 순서가 비제로 확률(non-zero probability)을 가지는 지수 모델과 순서의 수가 변동할 수 있는 다른 지수 모델을 비교하였습니다. 파르시모니(parsimonious)와 일반화(generalizability)를 중시할 때, 모든 24개 순서가 비제로 확률을 가지는 지수 모델이 강력한 지지를 받았습니다. 이 결과는 단어 순서의 변동에 대한 엄격한 제약이 없음을 시사합니다.

- **Performance Highlights**: 연구 결과는 언어 사용에 있어 관찰되지 않는 순서가 단순히 샘플링 부족으로 인한 것이라는 Cysouw의 견해와 일치합니다. 또한, 이러한 발견은 Zipf의 법칙과 같은 파워 로우 분포가 의무적이지 않다는 점을 강조하며, 언어 순서의 다양성에 대한 새로운 통찰을 제공합니다.



### Expect the Unexpected: FailSafe Long Context QA for Financ (https://arxiv.org/abs/2502.06329)
- **What's New**: 이번 논문에서는 금융 분야에서 LLMs(대형 언어 모델)의 강건성과 맥락 인식 능력을 테스트하기 위해 새롭게 개발된 FailSafeQA라는 장기 맥락 벤치마크를 제안합니다. 이 시스템은 사용자 인터페이스와의 상호작용에서 발생하는 여섯 가지 변수를 기반으로 하여 쿼리 실패와 맥락 실패의 두 가지 케이스를 집중적으로 연구합니다. 이러한 접근은 금융 데이터의 복잡성을 다루기 위함이며, LLM의 안정성과 신뢰성을 평가하는 데 큰 도움이 될 것입니다.

- **Technical Details**: FailSafeQA는 Qwen2.5-72B-Instruct 모델을 사용하여 24개의 오프더쉘프 모델에 대한 강건성(Robustness), 맥락 기초(Context Grounding), 준수(Compliance) 점수를 계산하는 정밀한 평가 기준을 마련합니다. 이 연구는 10-K 연례 보고서를 주요 텍스트 소스로 사용하며, 쿼리와 문맥의 왜곡을 통해 다양한 실패 유형을 평가합니다. 또한, 메타 라마 3.1 405B 모델을 이용해 쿼리 perturbation(왜곡)을 생성하고, OCR 오류를 시뮬레이션하여 비정상적인 상황에서 LLM의 반응을 테스트합니다.

- **Performance Highlights**: 연구 결과에 따르면 Palmyra-Fin-128k-Instruct 모델이 가장 높은 준수 점수를 기록했지만, 17%의 테스트 케이스에서 안정적인 예측을 유지하는 데 어려움을 겪었습니다. OpenAI o3-mini 모델은 41%의 경우에서 잘못된 정보를 생성하여 가장 강건성이 높은 평가를 받았으나, 동시에 신뢰성 문제를 드러냈습니다. 이러한 결과는 고성능 모델임에도 불구하고 개선 여지가 많음을 보여주며, FailSafeQA가 금융 애플리케이션에서 LLM의 의존성을 최적화하는 도구로서의 역할을 강조합니다.



### Can AI Examine Novelty of Patents?: Novelty Evaluation Based on the Correspondence between Patent Claim and Prior Ar (https://arxiv.org/abs/2502.06316)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 특허청구의 신규성(사실 여부)을 평가하는 새로운 과제를 제시합니다. 대규모 언어 모델(LLMs)의 성능을 실제 특허 심사 사례에서 유래한 새로운 데이터셋을 통해 평가하며, 언어 모델이 특허 신뢰성 및 비신뢰성을 구분할 수 있는 능력을 분석합니다. 이 연구는 NLP(Natural Language Processing)에서의 기존 특허 관련 과제들 사이에서 신규성 평가의 중요성을 강조하고 있습니다.

- **Technical Details**: 연구에서는 'Claim-Cited Texts (C-T) input'이라는 이진 분류 과제를 제안합니다. 이 과제는 실제 비최종거부 문서에서 청구항과 인용된 선행 기술 간의 관계를 바탕으로 신규성을 평가하도록 모델의 방향을 설정합니다. 또한, 비최종 거부 문서를 활용하여 청구항의 특정 요소와 관련된 문장 부분을 직접 추출하는 방식을 도입하여 모델이 각 청구항의 신규성을 보다 정확하게 판단할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과, 분류 모델들은 신규성을 평가하는 데 한계가 있음을 보여주었습니다. 반면, 생성 모델들은 실질적인 정확도로 신규성을 판단할 수 있으며, 제공되는 설명 또한 적절하여 청구항과 선행 기술 간의 관계를 이해할 수 있도록 돕습니다. 이러한 발견은 LLMs가 특허 평가 과정에서 큰 도움을 줄 수 있는 잠재력을 갖추고 있음을 시사합니다.



### Latent Convergence Modulation in Large Language Models: A Novel Approach to Iterative Contextual Realignmen (https://arxiv.org/abs/2502.06302)
- **What's New**: 이 논문에서는 자율 회귀 생성 모델에서의 토큰 예측 안정성을 개선하기 위해 'Latent Convergence Modulation (LCM)'이라는 새로운 방안을 제안합니다. 이 방법은 초기 상태의 변화를 조절하여 감춰진 표현의 경로가 이전 맥락 의존성을 유지하게 하면서 생성의 유연성을 보장합니다. LCM은 Transformer 기반 아키텍처 내에서 작동하며, 외부 메모리 의존성이나 구조적 변화 없이 내부 상태 전환을 동적으로 제어합니다.

- **Technical Details**: LCM은 숨겨진 상태의 변환을 조절함으로써 내부 표현의 경로가 안정된 궤도로 정렬되도록 합니다. 이 방식은 초기 활성화와 이후의 토큰 예측 간의 일관성을 높이기 위해 설계되었습니다. 그래디언트 전파 안정성이 향상되어 최적화 경로가 매끄럽게 유지되며, 반복적인 생성 과정에서 발생하는 변화를 최소화합니다. 기존의 기술에 비해 LCM은 많은 계산 자원을 소모하지 않으면서 Transformer 아키텍처에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, LCM은 긴 텍스트 생성의 일관성을 개선하고, 문맥을 유지하는 데 긍정적인 영향을 미쳤습니다. LCM을 통해 perplexity(혼란도) 변동, 엔트로피 변동, 어휘 불안정을 감소시키는 데 성공했습니다. 또한 대조 모델과의 비교 평가를 통해 대명사 해석, 논리적 일관성 및 문맥 정렬이 향상되었음을 확인했습니다.



### SeaExam and SeaBench: Benchmarking LLMs with Local Multilingual Questions in Southeast Asia (https://arxiv.org/abs/2502.06298)
Comments:
          Accepted to Findings of NAACL 2025

- **What's New**: 이번 연구는 동남아시아 (SEA) 애플리케이션 시나리오에서 대형 언어 모델 (LLMs)의 능력을 평가하기 위해 설계된 두 가지 새로운 벤치마크인 SeaExam과 SeaBench를 소개합니다. 기존의 다국어 데이터셋이 영어 번역에 기반하고 있는 것과 달리, 이 벤치마크는 SEA 지역의 실제 시나리오를 기반으로 구축되었습니다. SeaExam은 지역 교육 시험에서 추출한 데이터를 사용하여 지역 역사 및 문학과 같은 다양한 과목을 포괄합니다.

- **Technical Details**: SeaExam은 SEA 국가의 실제 시험에서 출처를 얻은 다치기 과제 시험 데이터셋이며, 지역 역사, 지리 및 문학을 다룹니다. 반면, SeaBench는 10개 과제 범주에 걸쳐 하루 대화에서 흔히 마주치는 시나리오와 지침을 포함한 다회전, 개방형 과제를 중심으로 제작되었습니다. 이러한 접근법은 SEA 맥락에서의 실제 사용을 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 실험 분석은 SeaExam과 SeaBench가 번역된 벤치마크보다 SEA 언어 작업에서 LLM의 성능을 한층 더 정교하게 구별할 수 있음을 보여줍니다. 또한, 개방형 질문이 다국어 사용에서 모델 성능의 차이를 강조하는 데 더욱 효과적이라는 것을 발견했습니다. 나아가, 안전성 범주에서 아홉 개 모델이 전반적으로 낮은 성과를 보였으며, 이는 다국어 애플리케이션에 있어 더 나은 안전 조치가 필요함을 시사합니다.



### Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE (https://arxiv.org/abs/2502.06282)
- **What's New**: 이 논문은 Speculative Decoding의 새로운 접근 방식인 Jakiro를 제안합니다. Jakiro는 Mixture of Experts(MoE) 메커니즘을 활용하여 토큰 예측의 다양성을 높이고, 각 단계에서 독립적인 전문가들이 예측을 생성합니다. 또한, 자가 회귀 방식과 병렬 디코딩 방법을 결합하여 성능을 향상시킵니다.

- **Technical Details**: Jakiro의 핵심은 전통적인 후보 생성 방식에서의 상관관계를 제거하고, MoE를 통해 각 레이어에서 독립적인 예측을 돕는 것입니다. 논문의 방법론은 다양한 모델에 대해 창출된 후보 토큰들이 서로 독립적이라는 점에서 중요한 장점이 있습니다. 이러한 접근법은 SOTA(State of the Art) 성능을 미치는 병렬 디코딩 전략도 포함하고 있습니다.

- **Performance Highlights**: 대규모 실험을 통해 Jakiro는 기존의 최첨단 기법들보다 우수한 성능을 보여주며, MT-bench에서 비탐욕적 모드에서의 눈에 띄는 발전을 달성했습니다. 이 방법은 예측 정확도를 크게 향상시키고 인퍼런스 속도를 높이며, 다양한 모델과 벤치마크에서 그 효율성과 강인성을 검증했습니다.



### DebateBench: A Challenging Long Context Reasoning Benchmark For Large Language Models (https://arxiv.org/abs/2502.06279)
- **What's New**: DebateBench는 전세계에서 명망있는 경쟁 논쟁의 전사 및 메타데이터로 구성된 새로운 데이터셋입니다. 이 데이터셋은 다양한 주제에 대한 영국 의회 형식의 논쟁을 포함하며, 공식 심사 데이터에서 제공된 상세한 스피치 점수와 순위로 주석이 달려 있습니다. 총 256개의 스피치를 32개의 논쟁에 걸쳐 큐레이션하였으며, 각 논쟁은 평균 32,000 토큰으로 이루어져 있습니다.

- **Technical Details**: DebateBench는 현대 대형 언어 모델(LLMs)의 주장, 심의 및 인간 전문가와의 정렬 능력을 평가하기 위한 기준을 제공합니다. 이 데이터셋은 스피치 점수 부여, 스피치 순위 예측, 하우스 오더링의 세 가지 주요 평가 작업을 포함합니다. 모델들은 각 연설에 대해 인간이 부여한 점수를 예측하고, 연설자 순위를 예측하며, 심사 결과에 따라 토론 팀을 순위별로 정리해야 합니다.

- **Performance Highlights**: 예비 평가에 따르면, GPT-1, GPT-4, Claude Haiku와 같은 최신 LLM들은 DebateBench에서 좋은 성과를 내지 못했습니다. 이는 대형 언어 모델이 구조화된 주장 및 긴 문맥 작업을 처리하는 데 어려움을 겪고 있음을 나타냅니다. 따라서 이는 모델 성능 향상을 위한 보다 정교한 기술 개발이 필요함을 강조합니다.



### Emergent Response Planning in LLM (https://arxiv.org/abs/2502.06258)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 다음 토큰을 예측하기 위해 학습되었음에도 불구하고 미래 출력을 계획하는 행동이 나타난다고 주장합니다. LLM의 숨겨진 표현이 향후 출력, 즉 다음 토큰을 넘어서 미래 결과를 인코딩하고 있다는 것을 보여줍니다. 이에 따라 응답 계획(response planning)이라는 현상을 정의하고, 이는 구조적 속성(structural attributes), 내용 속성(content attributes), 행동 속성(behavioral attributes)으로 구분됩니다.

- **Technical Details**: 연구진은 간단한 프로빙(probing) 기법을 통해 LLM의 프롬프트 표현이 향후 응답의 글로벌 속성을 효과적으로 인코딩하도록 학습시켰습니다. 실험 결과, 이러한 프로브는 비슷한 예측 정확도를 달성하며, LLM이 프롬프트를 읽는 즉시 전체 응답의 일부를 미리 계획하고 있다는 강력한 증거를 제공합니다. 또한 모델 크기가 증가할수록 이러한 계획 능력이 긍정적으로 확장된다는 점을 발견했습니다.

- **Performance Highlights**: 이 연구의 기여는 두 가지입니다. 첫째, LLM 내의 응답 계획의 첫 번째 공식 정의와 프레임워크를 도입한 것입니다. 둘째, 다양한 속성과 작업에 걸친 체계적인 프로빙 실험을 통해 LLM이 응답 계획을 수행한다는 것을 실증적으로 보여주고, 이러한 속성의 진화 과정을 조사했습니다. 이러한 발견은 LLM의 내부 메커니즘에 대한 통찰력을 제공하고, 생성 전에 출력을 예측하고 제어하는 새로운 접근 방식을 제안합니다.



### K-ON: Stacking Knowledge On the Head Layer of Large Language Mod (https://arxiv.org/abs/2502.06257)
Comments:
          AAAI 2025 (Oral)

- **What's New**: 본 논문에서는 K-ON이라는 새로운 모델을 제안하여 대규모 언어 모델(LLM) 내에 지식 그래프(KG)의 지식을 통합한다. K-ON은 여러 헤드 레이어(multiple head layers)를 통해 다음 k단계 예측(next k-step prediction)을 수행하며, 이는 엔티티 수준의 결과를 한 단계에서 생성할 수 있게 해준다. 또한, K-ON은 KG 표현 학습에서 가장 강력한 도구인 대조 손실(contrastive loss)을 가능하게 한다.

- **Technical Details**: K-ON은 KG 내의 엔티티를 효과적으로 예측하기 위해 K개의 서로 다른 헤드 레이어를 활용한다. 각 헤드는 모든 엔티티에 대한 k-th 토큰 예측을 담당하며, 이를 통해 엔티티를 하나의 통합된 예측으로 처리한다. 또한, K-ON은 헤드 궤적 조정(head trajectory tuning, HTT)이라는 방법을 통해 토큰 예측의 분포(distribution)를 정렬하여 원래 LLM의 예측 성능을 유지할 수 있도록 한다.

- **Performance Highlights**: K-ON은 KG 완성 과제에서 기존 방식보다 우수한 성능을 보여주었으며, 텍스트와 시각 정보를 추가적으로 활용하는 다중 모달(multi-modal) 방법보다도 더 나은 성과를 기록했다. 또한, K-ON은 GPU 자원을 더 요구하지만 훈련 에포크 수를 기존 1,000에서 5로 줄여 훈련 시간을 대폭 단축하였다. DB15K 데이터셋에서의 전체 미세 조정(fine-tuning) 시간은 1시간 11분 이하로 관리할 수 있었다.



### Confidence Improves Self-Consistency in LLMs (https://arxiv.org/abs/2502.06233)
- **What's New**: 이 논문은 Confidence-Informed Self-Consistency (CISC)를 제안하여 LLM의 추론 능력을 향상시킵니다. CISC는 모델에서 직접 얻은 confidence score를 사용하여 가중 다수결(voting)을 수행하고, 이를 통해 정답을 보다 적은 샘플로 찾아낼 수 있습니다. 연구 결과 CISC는 기존의 self-consistency 방법보다 평균 40% 이상의 reasoning paths 감소와 더불어 거의 모든 구성에서 성능 우위를 보여줍니다. 또한, LLM이 자신의 출력의 정답성을 평가할 수 있는 능력에 대한 실증적 증거를 제공합니다.

- **Technical Details**: CISC는 self-assessment score를 생성하여 각 reasoning path의 confidence를 기반으로 가중 투표를 수행하는 방식을 사용합니다. 예를 들어, 정확한 답변을 60%의 확률로 제공하는 경우, 전통적인 다수결 방법은 90% 정확도를 달성하기 위해 40개의 샘플이 필요하지만, CISC는 정확한 답변을 두 배로 가중치 부여하여 10개 미만의 샘플로 동일한 정확도에 도달할 수 있습니다. 또한, Within-Question Discrimination (WQD) 메트릭을 제안하여 동일한 질문에 대한 정답을 구분하는 데 유용성을 평가할 수 있습니다.

- **Performance Highlights**: CISC는 다양한 LLM 및 데이터셋을 사용한 테스트에서 self-consistency보다 우수한 성능을 보여주었습니다. 최상의 confidence estimation 방법을 사용할 경우, CISC는 self-consistency와 유사한 성능을 발휘하며 필요한 reasoning paths의 수를 평균 40% 이상 줄일 수 있습니다. 마지막으로, 모델이 스스로 판단하는 confidence score와 인간 평가자 간의 합의가 뚜렷하게 나타나는 것을 보여주었습니다.



### Examining False Positives under Inference Scaling for Mathematical Reasoning (https://arxiv.org/abs/2502.06217)
- **What's New**: 최근의 언어 모델(Language Models) 발전은 여러 벤치마크에서 수학적 추론(Mathematical Reasoning) 능력을 크게 향상시켰습니다. 하지만 대부분의 벤치마크는 최종 답변만을 비교하는 자동 평가 방법에 의존해, 근본적인 추론 단계를 검증하지 않는 한계가 있습니다. 이로 인해 올바른 최종 답변을 내놓더라도 잘못된 추론 경로를 가진 false positive 솔루션이 발생합니다.

- **Technical Details**: 본 연구에서는 언어 모델을 활용한 수학 문제 해결에서 false positive 솔루션의 발생 빈도를 체계적으로 조사하였습니다. 다양한 오픈 소스 모델과 난이도 수준이 다른 데이터셋, 디코딩 전략을 통해 문제의 특성과 범위를 분석하였습니다. 실험 결과, false positive 솔루션은 다양한 모델, 데이터셋, 및 디코딩 방법에서 여전히 존재하며, sampling-based inference time scaling 방법이 문제를 해결하지 못한다는 것을 확인했습니다.

- **Performance Highlights**: pass@N 평가 메트릭은 false positives에 더 민감하여 자동 평가가 제시하는 것보다 훨씬 낮은 스케일링 한계를 갖고 있음이 확인되었습니다. 또한, 구체적인 false positive 사례를 분석하고 이러한 조건에서의 self-improvement techniques 및 synthetic data generation의 잠재적 한계에 대해 논의합니다.



### Unveiling the Capabilities of Large Language Models in Detecting Offensive Language with Annotation Disagreemen (https://arxiv.org/abs/2502.06207)
Comments:
          17 pages, submitted to the ACL 2025

- **What's New**: 이번 논문에서는 인간 주석 간 불일치(annotation disagreement)를 고려한 공격적인 언어 감지의 새로운 접근 방식을 제시합니다. 기존 연구는 공격적인 언어를 명확한 이진 레이블로 단순화하여 오류를 범하고 있으며, 이로 인해 실제 데이터셋의 복잡성을 간과하고 있었습니다. 연구자들은 LLMs(large language models)가 불일치 샘플을 어떻게 처리하는가에 대한 구체적인 평가를 실시하여, 인공지능의 신뢰성과 결정 과정의 복잡성을 탐구합니다.

- **Technical Details**: 이 연구의 핵심 데이터셋인 MD-Agreement는 트위터에서 수집된 10,753개의 샘플로, 높은 품질의 주석을 보장합니다. 각 샘플은 5명의 훈련된 주석자에 의해 주석이 달리며, 다수결에 따라 하드 라벨과 함께 주석 동의 정도를 나타내는 소프트 라벨도 제공합니다. 실험에서는 다양한 파라미터 크기를 가진 LLM을 사용하여 공격적인 언어를 감지하는 성능을 평가하고, 몇 가지 주류 기술인 few-shot learning과 instruction fine-tuning에 미치는 영향을 분석합니다.

- **Performance Highlights**: LLMs의 공격적인 언어 감지 성능은 우리의 평가에서 예상보다 훨씬 뛰어난 것으로 나타났습니다. 실험에 사용된 폐쇄형 모델의 이진 정확도 평균은 88.28%, 오픈 소스 모델은 86.07%였으며, 이는 동의가 높은 샘플들에서 특히 강력한 성능을 보였습니다. 그러나 불일치 샘플이 포함된 경우, LLM의 결정 신뢰도가 감소하여 이 분야에서 더 많은 연구가 필요함을 시사합니다.



### C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation (https://arxiv.org/abs/2502.06205)
Comments:
          Ongong work

- **What's New**: 본 논문에서는 Retrieval-augmented generation (RAG) 시스템의 기본적인 문제인 독립적으로 개발된 retriever와 대형 언어 모델(LLMs) 간의 정렬 문제를 다룹니다. 기존 접근 방식은 컴포넌트를 수정하거나 간단한 중간 모듈을 도입하는 방식이었으나, 이는 실제적인 제한과 최적화되지 않은 성능을 초래했습니다. 새로운 프레임워크인 C-3PO는 경량 멀티 에이전트 시스템을 통해 retrievers와 LLMs 간의 효과적인 소통을 지원합니다.

- **Technical Details**: C-3PO는 세 가지 전문화된 에이전트를 구현하여 전체 RAG 파이프라인을 공동으로 최적화합니다. 이 아키텍처는 retriever와 LLMs를 변경하지 않고, 정보를 선택하고 효과적인 쿼리를 생성하며, retrieval의 필요성을 평가합니다. 또한 강화 학습의 보상 기여 할당을 위한 트리 구조의 rollout 접근 방식을 개발하여 멀티 에이전트 간의 효과적인 조정을 가능하게 합니다.

- **Performance Highlights**: 다양한 도메인 내 및 도메인 외 시나리오에서 수행한 광범위한 실험 결과, C-3PO는 RAG 성능을 대폭 향상시킴을 입증했습니다. 이 프레임워크는 플러그 앤 플레이(plug-and-play) 유연성을 유지하며, 뛰어난 일반화 능력도 보장합니다.



### Non-literal Understanding of Number Words by Language Models (https://arxiv.org/abs/2502.06204)
Comments:
          12 pages, 10 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 숫자를 비유적으로 해석하는 방식에 대한 새로운 통찰을 제공합니다. 저자들은 LLMs가 인간과 유사하게 숫자를 해석하는지 조사하기 위해 하이퍼볼과 프래그머틱 헤일로 효과를 중심으로 분석했습니다. 연구 결과, LLMs는 숫자 표현에 대한 해석에서 인간과 현저한 차이를 보였으며, 이는 LLM의 잠재적 제한점을 이해하는 데 중요한 시사점을 제공합니다.

- **Technical Details**: 연구는 LLMs의 수치 표현 해석을 평가하기 위해 Kao 등(2014)에 의해 개발된 합리적 발화 행위(RSA) 프레임워크를 활용하여, 인간의 비유적 해석이 화자의 의도와 선행 지식 간의 상호작용에 의해 어떻게 이루어지는지를 모델링했습니다. 이 모델은 비유적 해석이 여러 가지 의미를 동시에 전달할 수 있도록 확장되어 있으며, LLMs의 수치 해석이 이러한 인지 모델에서 어떻게 발전할 수 있는지를 탐구합니다.

- **Performance Highlights**: 실험에서 LLMs(예: GPT-4o-mini, Claude-3.5-sonnet, Gemini-1.5-pro)는 하이퍼볼과 숫자에 대한 해석에서 인간의 패턴과 비교되었습니다. 연구에 따르면, LLMs는 맥락상 비현실적인 경우 문자적 해석의 확률을 낮추는 경향을 보였으며, 이는 LLM의 수치 해석 능력이 제한적임을 나타냅니다. 연구는 또한 체인 오브 싱크(Chain-of-Thought) 프롬프팅을 통해 LLMs의 해석 능력을 인간과 유사하게 개선하는 방법을 모색했습니다.



### Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization (https://arxiv.org/abs/2502.06185)
Comments:
          NAACL 2025 camera-ready version

- **What's New**: 이번 논문에서는 긴 문서 요약의 사실 불일치(factual inconsistency) 문제를 다루고 있으며, 특히 이 문제를 담담하는 담화 분석(discourse analysis)과의 연계를 탐구합니다. 연구 결과에 따르면, 복잡한 문장에서 사실 불일치 오류가 더 자주 발생하며, 이는 여러 담화 특징들과 관련이 있다는 것을 발견했습니다. 또한, 담화에 기반한 정보를 활용한 새로운 평가 방법론인 StructScore를 제안하여 긴 문서 요약의 사실 불일치 검출 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 StructScore는 두 가지 단계로 구성됩니다: 첫째, 목표 요약의 문장 수준 정렬 점수를 집계할 때 담화 정보를 활용하고, 둘째, 긴 입력 기사를 여러 담화에서 영감을 받은 청크로 분해합니다. 이를 통해 긴 문서 요약에서 NLI 기반 접근 방식을 활용하여 보다 효과적으로 사실 불일치를 탐지할 수 있습니다. 실험은 AggreFact-FtSOTA, DiverSumm, LongSciVerify, LongEval 등 여러 문서 요약 벤치마크에서 진행되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 다양한 작업에서 성능 향상을 보여주었으며, 모델과 모델 출력은 공개적으로 제공됩니다. 본 연구는 긴 문서 요약의 사실 불일치 문제 해결을 위한 담화 구조와 사실성 평가 사이의 관계를 최초로 조사한 연구로, 앞으로 이 분야의 연구에 중요한 기여를 할 것으로 보기됩니다.



### RideKE: Leveraging Low-Resource, User-Generated Twitter Content for Sentiment and Emotion Detection in Kenyan Code-Switched Datas (https://arxiv.org/abs/2502.06180)
Comments:
          Accepted in WASSA 2024

- **What's New**: 이 논문은 케냐에서의 코드 스위칭 언어에 대한 감정 및 감정 분류를 위해 사전 훈련된 최신 변환기(transformer) 모델을 평가하고, 그 방법론과 데이터 수집 및 주석 작업 중 발생한 문제를 설명합니다. RideKE라는 데이터셋을 소개하며, 이 데이터셋은 29,000개 이상의 트윗을 포함하고 있으며 감정이 긍정적, 부정적, 중립적으로 분류됩니다. 따라서 이 연구는 저자원(low-resource) 언어에서의 자연어 처리(NLP) 향상에 기여하는 것을 목표로 하고 있습니다.

- **Technical Details**:  케냐의 다언어적 특성을 반영한 본 연구에서는 영어, 스와힐리어 및 셍 언어가 혼합된 감정 데이터셋을 분석합니다. XLM-R, DistilBERT, mBERT 및 AfriBERTa와 같은 사전 훈련된 모델들이 사용되며, 이들 모델들은 감정 분석 및 감정 분류에 대한 성능 비교를 통해 저자원 언어의 NLP 성능을 평가합니다. 특히, 이 논문에서는 감정 분석을 위해 감독 학습(supervised learning) 및 반감독 학습(semi-supervised learning)을 활용합니다.

- **Performance Highlights**: 연구 결과에 따르면 XLM-R 모델이 감정 분석에서 최고의 정확도(69.2%)와 F1 점수(66.1%)를 기록하며, 정서 분석에서도 DistilBERT가 59.8%의 정확도를 보였습니다. 모든 모델은 중립적인 감정을 예측하는 경향이 있으며, AfriBERTa는 가장 낮은 정확도와 F1 점수를 나타냈습니다. 연구는 다양한 모델의 성능과 저자원 언어에서의 감정 인식의 가능성을 제시합니다.



### Scaling Public Health Text Annotation: Zero-Shot Learning vs. Crowdsourcing for Improved Efficiency and Labeling Accuracy (https://arxiv.org/abs/2502.06150)
Comments:
          4 pages, 1 figure

- **What's New**: 이 논문에서는 공공 보건 연구에서 수집되는 트위터의 비정형 데이터를 효율적으로 라벨링하기 위한 새로운 접근 방식으로, 대규모 언어 모델(LLM)인 GPT-4 Turbo를 활용하는 방법을 제안합니다. 기존의 크라우드소싱 방식 대신, 제로샷(Zero-Shot) 라벨링을 통해 전문가의 성과와 경쟁할 수 있는 가능성을 타진하고 있습니다. 이 연구는 라벨링 품질을 저해하지 않으면서도 자동 라벨링의 효율성을 증대시킬 수 있는 방안을 모색합니다.

- **Technical Details**: 연구 방법론에서는 수학적 프레임워크를 이용하여 분류 작업과 LLM 기반의 라벨링 프로세스를 정의합니다. 데이터 수집, 주석 처리, 모델 배포 파이프라인에 관한 구체적인 내용을 설명하며, LLM을 통합하고 성능을 평가하는 과정을 포함합니다. 연구 데이터는 트위터 게시물 12,000개로 구성되며, 각 게시물은 신체 활동, 좌식 행동, 수면 문제 등 3가지 건강 관련 주제가 포함되었습니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM을 활용한 자동 라벨링은 명확한 분류 작업에서는 인간의 성과에 필적하거나 이를 초과할 수 있지만, 더 세부적이고 맥락에 의존하는 정보를 요구하는 작업에서는 정확도가 떨어지는 경향이 있습니다. 자동 라벨링과 인간 전문가의 하이브리드 작업 흐름을 통해 라벨링 예산을 절감하고 프로젝트 일정을 가속화할 수 있습니다. 이러한 접근 방식은 전염병 발생의 조기 탐지나 환경 위험 감시와 같은 빠른 분석을 필요로 하는 공공 보건 프로젝트에 크게 이득이 될 것으로 기대됩니다.



### Optimizing Knowledge Integration in Retrieval-Augmented Generation with Self-Selection (https://arxiv.org/abs/2502.06148)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문은 Self-Selection RAG(자기 선택 검색 증강 생성) 프레임워크를 제안하여 대형 언어 모델(LLM)이 외부에서 검색된 지식과 내부 파라메트릭 지식을 효과적으로 통합해 보다 정확한 결과를 생성하도록 돕습니다. 기존의 RAG 접근 방식의 한계를 극복하기 위해, LLM이 생성한 두 개의 응답을 비교하여 더 정확한 응답을 선택하도록 훈련합니다. 이 과정에서 직접 선호 최적화(Direct Preference Optimization, DPO) 기법을 활용하여 성능을 향상시킵니다.

- **Technical Details**: Self-Selection RAG 프레임워크는 LLM이 LLM 응답과 RAG 응답을 비교하고, 각각의 설명을 고려하여 올바른 답변을 선택할 수 있게 돕는 방법론입니다. LLM의 성능을 강화하기 위해, 새로운 Retrieval-Generation Preference(RGP) 데이터세트를 사용하여 LLM을 세밀하게 조정합니다. 이를 통해 외부에서 검색된 지식과 내부 지식을 결합하여 더욱 정확한 응답을 생성할 수 있습니다.

- **Performance Highlights**: 자체 선택 RGP 방법론은 Mistral-7B와 LLaMa-2-13B-Chat과 같은 두 개의 오픈 소스 LLM에서 테스트되었고, Natural Questions(NQ)와 TrivialQA 데이터세트에서 우수한 성능을 보여주었습니다. 실험 결과, 제안된 방법이 다양한 검색 설정과 LLM에서 높은 효과성을 일관되게 달성하며 RAG 시스템의 강건성과 안정성을 향상시킴을 입증합니다. 추가 실험을 통해 이 방법이 잡음이 많은 응답에서 유효한 답변을 구별하는 능력과 응답 생성 능력을 모두 향상시킨다는 사실도 확인되었습니다.



### LegalViz: Legal Text Visualization by Text To Diagram Generation (https://arxiv.org/abs/2502.06147)
Comments:
          NAACL2025

- **What's New**: 이 논문의 새로운 점은 LegalViz라는 혁신적인 데이터셋을 제안하여 법률 문서와 설명 다이어그램 간의 쌍을 7,010개 수집한 것입니다. 이 데이터셋은 23개 언어로 제공되며 법률 개체, 거래 및 발언을 시각적으로 표현하는 것을 목표로 하고 있습니다. 또한, 법적 내용을 기반으로 한 새로운 평가 메트릭스를 제시하여 언어 모델의 시각화 효율성을 높였습니다.

- **Technical Details**: LegalViz 데이터셋은 Graphviz의 DOT 언어를 사용하여 법적 문서의 시각화와 관련된 다양한 정보를 제공합니다. 모델은 법적 요건을 이해하고, 관련 법률 규칙 및 개체들 간의 관계를 해석할 수 있도록 훈련됩니다. 이를 통해 법률 문서의 중요한 사실을 한눈에 파악할 수 있는 구조화된 다이어그램을 생성할 수 있습니다.

- **Performance Highlights**: LegalViz로 훈련된 모델은 기존의 모델보다 뛰어난 성과를 보여주고 있으며, 법률 규정의 정확한 해석과 관련된 관계를 인식하는 데 효과적이라고 입증되었습니다. 벌금 평가와 관련된 다양한 23개 언어에서 모델의 성능을 검증한 결과, 이 데이터셋은 법적 작업의 자동화를 지원하는 데 중요한 역할을 할 것으로 예상됩니다.



### LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs (https://arxiv.org/abs/2502.06139)
Comments:
          Accepted to NAACL 2025 Main

- **What's New**: 이 논문에서는 고정 길이 위치 임베딩으로 인해 긴 형식의 문맥을 효과적으로 처리할 수 있는 능력이 제한된 대규모 언어 모델(LLMs)의 문제를 해결하기 위해 Long-form Context Injection with Recurrent Compression (LCIRC) 기법을 제안합니다. 이 방법은 전체 모델을 재훈련하지 않고도 모델의 길이 제한을 넘어 긴 형식의 시퀀스를 효율적으로 처리할 수 있도록 돕습니다. 또한, 질의 의존적(context dependent) 문맥 모델링을 도입하여 질의와 관련된 정보를 선택적으로 압축함으로써 모델이 가장 중요한 콘텐츠를 유지하도록 보장합니다.

- **Technical Details**: LCIRC는 모델의 길이 제한을 넘어서는 긴 형식의 입력을 효과적으로 처리하기 위해 문맥을 압축하여 모델에 주입하는 방식입니다. 이 접근법은 입력 시퀀스를 반복적으로 압축하여 문맥을 보존하면서 계산 오버헤드를 최소화합니다. 모델은 질의에 따라 중요성이 높은 정보를 우선적으로 선택하여 압축하는 질의 의존적 문맥 모델링을 통해 여러 선택 및 긴 형식 질문 답변 작업에서 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, LCIRC는 질의 의존적 모델링과 결합하여 LLM이 긴 형식의 문맥을 처리하는 능력을 획기적으로 향상시킵니다. 이 접근법은 대규모 문맥 이해와 질의의 정확한 관련성을 모두 요구하는 응용 분야에 적합합니다. 다양한 벤치마크 테스트를 통해 성능 개선이 수 quantitatively적으로 입증되었습니다.



### Task-driven Layerwise Additive Activation Intervention (https://arxiv.org/abs/2502.06115)
Comments:
          Accepted to NAACL 2025

- **What's New**: 본 논문은 현대 언어 모델(LMs)에서의 새로운 접근법으로 Layer-wise Additive Activation Intervention 프레임워크를 제안합니다. 이 방법은 LMs의 생성 과정을 최적화하여 샘플 효율성을 향상시키며, 기존의 활성화 개입 방식의 한계를 극복하는 데 중점을 둡니다. 특히, 적은 양의 훈련 데모로도 경쟁력 있는 결과를 제공할 수 있도록 설계된 접근법입니다.

- **Technical Details**: 제안된 방법은 각 레이어에 대한 작업 특화 수정 벡터를 활성화에 추가하여 출력을 조정하는 방식으로 구성됩니다. 각 레이어는 서로 다른 헤드와 차원을 가지며, 이 프레임워크는 joint lasso 및 group lasso 정규화를 통해 과적합을 방지하고 파라미터의 스파시티(sparsity)를 촉진합니다. 이 최적화 문제는 동일한 레이어에 초점을 맞추어 구성되며, 효과적인 작업 계산을 가능하게 합니다.

- **Performance Highlights**: 제안된 프레임워크는 다양한 데이터셋에서 벤치마킹을 수행하였고, 사전 훈련된 LMs의 정확도를 개선함을 보여주었습니다. 기존에 사용된 개입 방법들과 비교하여, 적은 수의 프롬프트 입력으로도 효과적인 결과를 도출할 수 있는 점에서 큰 장점을 가지며, 향후 NLP 작업에서의 활용 가능성을 넓힐 것으로 기대됩니다.



### ConMeC: A Dataset for Metonymy Resolution with Common Nouns (https://arxiv.org/abs/2502.06087)
Comments:
          NAACL 2025

- **What's New**: 이번 논문에서는 일반 명사에 대한 메토님이(metonymy) 해결 방안을 제시합니다. 이를 위해 6,000개의 문장으로 구성된 새로운 데이터셋인 ConMeC(Common Nouns Metonymy Corpus)를 소개하고, 각 문장에서 일반 명사가 메토님으로 사용되는지를 사람의 주석으로 표시하였습니다. 기존의 메토님 데이터셋들은 주로 고유 명사 또는 단순 문장으로 구성된 경우가 많아 이 연구는 중요한 발전을 제공합니다.

- **Technical Details**: 저자들은 메토님 탐지를 위해 체인-오브-생각(chain-of-thought) 기반의 프롬프팅(promting) 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)을 활용하여 개념 범주를 먼저 식별한 후 해당 범주에 따라 프롬프트를 적용하는 방식입니다. 또한, 자가 일관성(self-consistency) 전략을 사용하여 모델의 성능을 개선하고, 다양한 메토님 데이터셋에서 평가를 진행하였습니다.

- **Performance Highlights**: 실험 결과, LLM들은 잘 정의된 메토님 범주에서 미세 조정된 BERT 모델과 유사한 성능을 보였습니다. 그러나 LLM들이 세심한 의미 이해가 요구되는 사례에서는 여전히 어려움을 겪는 것으로 나타났습니다. 저자들은 이러한 연구 결과가 향후 NLP 파이프라인이 인간처럼 암묵적인 언어의 이해 능력을 평가하는 데 기여할 것이라고 강조하고 있습니다.



### Is a Peeled Apple Still Red? Evaluating LLMs' Ability for Conceptual Combination with Property Typ (https://arxiv.org/abs/2502.06086)
Comments:
          NAACL 2025; the dataset and experimental code are available at this https URL

- **What's New**: 이번 연구에서는 Conceptual Combination with Property Type 데이터셋(CCPT)을 소개합니다. 이 데이터셋은 12.3K개의 주석이 달린 명사구, 속성 및 속성 유형의 삼중으로 구성되어 있습니다. LLMs(대형 언어 모델)을 평가하기 위한 세 가지 작업이 제시되며, 기계적 평가 기준이 인간의 평가와 밀접하게 일치함을 보여줍니다. 또한, 고급 LLM은 새로운 속성을 가진 명사구를 생성하는 데 어려움을 겪고 있음을 밝혔습니다.

- **Technical Details**: CCPT 데이터셋은 기본 개념에서 발생하는 속성, 새로운 속성, 그리고 조합으로 인해 사라지는 속성을 평가합니다. 제안된 세 가지 작업 중 첫 번째는 명사구의 속성을 상상하는 방법인 Property induction입니다. 두 번째는 특정 속성을 나타내기 위한 명사구를 생성하는 Noun phrase completion이며, 세 번째는 속성의 출처를 인식하기 위한 Property type prediction입니다.

- **Performance Highlights**: 실험 결과에 따르면 현재의 LLMs는 진정으로 새로운 속성을 가진 명사구를 생성하는 데 어려움을 겪고 있으며, 제안된 방법은 모든 생성 작업에서 성능을 개선했습니다. 이러한 개선은 개념 간의 다양한 관계를 고려한 결과입니다. 데이터셋과 실험 코드는 제공된 URL에서 확인할 수 있습니다.



### Benchmarking Prompt Sensitivity in Large Language Models (https://arxiv.org/abs/2502.06065)
- **What's New**: 이번 논문에서는 Prompt Sensitivity Prediction이라는 새로운 작업을 소개하고, LLM의 응답 정확성에 미치는 프롬프트 변형의 영향을 조사하기 위해 PromptSET이라는 데이터셋을 설계했습니다. 주요 목적은 LLM의 프롬프트 반응 능력에 대한 예측을 통해, 프롬프트의 미세한 변형이 LLM 성능에 미치는 영향을 분석하는 것입니다. 이를 통해 효과적인 프롬프트 설계의 필요성을 강조하고 있습니다.

- **Technical Details**: 제안된 Prompt Sensitivity Prediction 작업은 주어진 프롬프트가 LLM에 의해 효과적으로 수행될 수 있는지를 예측하는 것을 목표로 합니다. 각 프롬프트는 특정 정보 요구(Ip)에 따라 약간 수정된 버전으로 구성됩니다. 데이터셋은 TriviaQA와 HotpotQA에서 출발하여 생성된 다양한 프롬프트 변형으로 구성되며, 이 변형의 유사성 및 정보 요구의 일관성을 기준으로 평가됩니다.

- **Performance Highlights**: 기존의 텍스트 분류(TC) 및 질의 성능 예측(QPP)과의 유사성을 기반으로 프롬프트 민감도 예측 작업을 벤치마크하는 실험을 수행했습니다. 연구 결과, 기존 방법들이 이 새로운 작업을 효과적으로 해결하지 못함을 보여주었으며, 이는 프롬프트 민감도 예측을 위한 새로운 접근 방법의 필요성을 강조합니다.



### LM2: Large Memory Models (https://arxiv.org/abs/2502.06049)
- **What's New**: 이 논문에서는 Large Memory Model (LM2)을 소개하며, 이는 다중 단계 추론(multi-step reasoning) 및 관계적 주장(relational argumentation)을 처리하는 데 있어 전통적인 Transformer 모델의 한계를 극복하기 위해 보조 메모리 모듈을 갖춘 디코더 전용 Transformer 아키텍처입니다. LM2는 입력 토큰과 상호작용하고 게이팅 메커니즘을 통해 업데이트되는 컨텍스트 표현 저장소로 기능하는 메모리 모듈을 통합하고 있습니다.

- **Technical Details**: LM2는 여러 개의 Transformer 디코더 블록으로 구성되며, 메모리 모듈이 동적으로 중간 표현의 시퀀스를 저장하고 업데이트합니다. 이 모듈은 메모리 정보 흐름과 업데이트를 통해 작동하며, 입력 임베딩과 메모리 은행 간의 교차 주의(cross attention) 메커니즘을 사용하여 관련 정보를 찾습니다. 또한, 각 메모리 슬롯은 정체성 행렬로 초기화되어 있으며, 기억 제어 게이트(forget, input, and output)를 통해 메모리 업데이트가 조정됩니다.

- **Performance Highlights**: BABILong 벤치마크에서 LM2는 메모리 보강 Recurrent Memory Transformer (RMT) 모델보다 최대 80.4% 향상된 성능을 보였으며, MMLU 데이터셋에서도 기존 모델 대비 5.0% 향상된 결과를 기록했습니다. 특히 LM2는 다단계 추론(multi-hop inference), 수치 추론(numerical reasoning), 대규모 컨텍스트 질문-응답(question-answering)에서 뛰어난 성능을 입증하고 있습니다. 이러한 결과는 Transformer 아키텍처에 명시적 메모리를 통합하는 것의 중요성을 강조합니다.



### Analysis of LLM as a grammatical feature tagger for African American English (https://arxiv.org/abs/2502.06004)
Comments:
          13 pages, Accepted to "Findings of the Association for Computational Linguistics: NAACL 2025"

- **What's New**: 이번 연구에서는 African American English (AAE)의 고유한 문법적 특성을 인식하는 데 있어 다양한 자연어 처리(NLP) 모델을 체계적으로 비교합니다. AAE는 훈련 데이터가 부족한 저자원 언어로, Rule-based 모델과 Transformer 모델, 대형 언어 모델(LLM)이 평가됩니다. 연구 결과, LLM은 기존 모델보다 성능이 향상되었으나, 텍스트의 형식성과 같은 편향에 영향을 받는 것으로 나타났습니다.

- **Technical Details**: 연구는 Habitual Be와 Multiple Negation을 AAE의 주요 문법적 특성으로 선택하여 이들을 인식하는 NLP 시스템의 능력을 평가합니다. 각 모델은 주어진 문장의 긍정 또는 부정을 분류하는 이진 분류 작업을 수행하며, 데이터와 모델 설정은 공정한 평가를 위해 일관된 하이퍼파라미터와 프롬프트 구조로 구성됩니다. OpenAI의 gpt-4o-mini와 Meta의 LLaMA 3-8B-Instruct 모델을 사용하여 AAE 특성에 대한 인식 성능을 비교합니다.

- **Performance Highlights**: 모델의 평가에서 LLM이 기존의 Rule-based 및 Transformer 모델보다 더 높은 성능을 보여주었습니다. 그러나 LLM은 예시의 순서나 최근성 같은 요소에 영향을 받으며, 이는 성능에 부정적인 영향을 미칠 수 있습니다. 연구 결과는 AAE의 고유한 언어적 특성을 더 잘 수용하기 위한 모델의 개선이 필요하다는 것을 강조하며, 데이터와 코드는 공개되어 추가 연구를 지원합니다.



### Preventing Rogue Agents Improves Multi-Agent Collaboration (https://arxiv.org/abs/2502.05986)
- **What's New**: 이번 연구에서는 복합적인 작업을 수행하는 다중 에이전트 시스템에서 비정상적인 에이전트를 감지하여 시스템의 실패를 예방하는 방법을 제안합니다. 이 접근 방식은 에이전트의 의사결정 과정에서 실시간으로 모니터링하고, 문제가 발생하기 전에 개입(intervene)하는 방법을 포함합니다. 새로운 실험 환경인 WhoDunitEnv를 도입하여 통신 구조와 작업 복잡성을 조절 할 수 있습니다.

- **Technical Details**: 이 연구에서는 단순한 분류기를 사용하여 에이전트의 불확실성을 측정하고 이에 기반하여 비정상적인 에이전트를 감지합니다. 또한, 모니터링 결과를 바탕으로 통신 왜곡을 방지하고 작업 실패를 예방하기 위한 개입 메커니즘을 제안합니다. WhoDunitEnv 환경에서 다양한 에이전트 통신 프로토콜을 분석하는 데 중점을 두며, 에이전트들은 서로 협력하여 주어진 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 비정상적인 에이전트를 모니터링하고 개입을 실시하는 것이 다중 에이전트 협업에서 20% 이상의 성능 향상을 가져오는 것으로 나타났습니다. 본 연구의 분석을 통해, 에이전트의 혼란과 에러 전파를 효과적으로 차단할 수 있는 강력한 모니터와 개입의 필요성을 강조하였습니다. Therefore, 이 방법은 향후 다중 에이전트 시스템 개발에 있어 중요한 기여를 할 것으로 기대됩니다.



### HamRaz: A Culture-Based Persian Conversation Dataset for Person-Centered Therapy Using LLM Agents (https://arxiv.org/abs/2502.05982)
- **What's New**: 이 논문에서는 HamRaz라는 새로운 페르시아어 정신 건강 데이터 세트를 소개합니다. 이 데이터 세트는 인공지능 기반의 심리 상담에서 자주 사용되는 Large Language Models (LLMs)을 위한 Person-Centered Therapy (PCT) 설계를 위해 만들어졌습니다. HamRaz는 기존의 데이터 세트들이 주로 서구 및 동아시아 맥락에 초점을 맞추는 것과 달리 페르시아어 치료에 필수적인 문화적, 언어적 미세 조정을 반영합니다.

- **Technical Details**: HamRaz는 스크립트 기반 대화와 적응형 LLM 역할 놀이를 결합하여 일관되고 역동적인 치료 상호작용을 보장합니다. 또한, HamRazEval이라는 이중 평가 프레임워크를 도입하여 General Dialogue Metrics와 Barrett-Lennard Relationship Inventory (BLRI)를 사용하여 대화 품질과 치료 효과성을 측정합니다. 이 평가 방법론은 다양한 측면에서 HamRaz의 성능을 평가하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, HamRaz는 전통적인 스크립트 모드와 두 에이전트 모드를 초월하여 더 공감적이고, 상황을 인식하며, 현실적인 치료 세션을 생성하는 것으로 나타났습니다. HamRaz의 출시는 다양한 지역 사회의 AI 기반 심리 치료 연구를 발전시키기 위한 문화적으로 적합한 자원을 제공합니다.



### Speech to Speech Translation with Translatotron: A State of the Art Review (https://arxiv.org/abs/2502.05980)
Comments:
          12 pages and 3 figures

- **What's New**: 이 논문은 음성-음성 번역(S2ST) 모델에 대한 포괄적인 리뷰를 제공하며, Google의 Translatotron 모델에 특히 주목합니다. Translatotron은 전통적인 캐스케이드 모델의 복합 오류 문제를 해결하기 위해 설계된 첫 번째 직접 음성-음성 번역 모델입니다. 지금까지 Translatotron 1, 2, 3의 세 가지 버전이 있으며, 각 버전이 이전 모델보다 더 나은 성능을 보이고 있습니다.

- **Technical Details**: Translatotron 모델은 음성 인식(speech recognition), 자동 번역(machine translation), 텍스트-음성 변환(text-to-speech synthesis) 등 여러 가지 기술을 사용하여 음성을 직접 번역하는 방식입니다. 이 연구에서는 음성 스펙트로그램(speech spectrogram)을 매핑하여 각 언어의 음성 스펙트로그램을 연결하는 과정을 소개하고 있습니다. 특히 Translatotron 2는 음성 인코더(speech encoder), 언어 디코더(linguistic decoder), 음향 합성기(acoustic synthesizer)로 구성되어 있습니다.

- **Performance Highlights**: Translatotron 모델은 기존의 캐스케이드 모델보다 일부 상황에서 더 나은 성능을 보이며, 특히 아프리카 언어와 잘 형성된 언어 간의 언어 갭을 줄이는데 초점을 맞추고 있습니다. Translatotron 2는 직관적인 음성-음성 번역을 가능하게 하며, BLEU 점수에서 눈에 띄는 개선을 보여줍니다. 이 연구는 Translatotron 모델의 차별성과 각 모델의 적합성을 비교 분석하는 데 중점을 두고 있습니다.



### "Let the AI conspiracy begin..." Language Model coordination is just one inference-intervention away (https://arxiv.org/abs/2502.05945)
Comments:
          Large Language Models (LLMs), Interference-time activation shifting, Steerability, Explainability, AI alignment, Interpretability

- **What's New**: 이번 연구에서는 대형 언어 모델의 행동을 조정할 수 있는 간단하면서도 효과적인 방법론을 도입합니다. 이 방법은 추가 교육 없이도 효과적인 interference-time activation shifting을 활용하며, 모델 출력의 원하는 행동과 원하지 않는 행동 간의 activation 차이에서 intervention 방향을 파생합니다. 또한, 다중 선택 답변을 포함하도록 모델을 유도함으로써 각 attention head의 조작에 대한 출력의 민감도를 자동으로 평가할 수 있습니다.

- **Technical Details**: 대형 언어 모델은 다양한 애플리케이션에서 널리 사용되고 있지만, 악의적인 사용자들에 의해 남용될 수 있는 위험이 존재합니다. 본 논문에서는 inference-time intervention을 통해 모델의 출력을 특정 방향으로 조정할 수 있는 가능성을 보여줍니다. 이를 위해 attention head 수준에서의 조작 기술을 사용하며, 이는 샘플 효율이 좋고 모델 weight의 업데이트가 필요하지 않은 장점이 있습니다.

- **Performance Highlights**: 우리는 Llama 2 모델이 AI 간의 협조를 선호하도록 유도하는 결과를 보여주었으며, 이는 기존의 alignment 목표보다 더 강력한 조작을 가능하게 합니다. 특히, 우리의 방법론은 'AI coordination' 데이터셋에서의 개방형 질문 생성에서도 잘 일반화되며, 이로 인해 현재의 alignment 전략의 한계를 강조하고 향후 연구 방향을 제시합니다.



### Multi-granular Training Strategies for Robust Multi-hop Reasoning Over Noisy and Heterogeneous Knowledge Sources (https://arxiv.org/abs/2502.05944)
- **What's New**: 이 논문에서는 Adaptive Multi-source Knowledge-Oriented Reasoning (AMKOR)라는 새로운 프레임워크를 제안합니다. AMKOR는 대형 언어 모델(LLMs)의 강력한 생성 능력과 지식 융합 메커니즘을 결합하여 다중 출처의 지식을 동적으로 통합합니다. 이를 통해 지식 충돌을 해결하고 다중 단계의 추론을 지원하는 효과적인 방법을 제공합니다.

- **Technical Details**: AMKOR는 파라메트릭(parasitic) 지식과 외부에서 검색된 지식을 결합하여 다중 출처의 지식 통합을 수행합니다. 이 프레임워크는 확률적 추론(probalistic reasoning)과 다층 진입 학습 전략(multi-granular learning strategy)을 활용하여 지식 충돌, 오류 전파(cascading errors) 및 확장성(scalability) 문제를 해결합니다. 이전의 방법들과 달리 AMKOR는 지식 융합 모듈을 통해 최적화된 훈련 전략을 사용하여 질의에 대한 최종 답을 생성합니다.

- **Performance Highlights**: AMKOR은 HotpotQA, MuSiQue 등 네 가지 오픈 도메인 다중 단계 QA 데이터셋에서 실험을 수행하여 최신 기술보다 평균 2.5% 향상된 성능을 기록했습니다. 특히, MuSiQue와 Bamboogle과 같은 높은 추론 복잡성을 가진 데이터셋에서 뛰어난 성능을 보여줌으로써 복잡한 다단계 문제 해결 능력을 입증했습니다. 이 연구는 다중 출처 다단계 QA의 새로운 기준을 설정하며 추론 품질과 효율성을 효과적으로 결합합니다.



### A Semi-Supervised Text Generation Framework Combining a Deep Transformer and a GAN (https://arxiv.org/abs/2502.05937)
Comments:
          7 pages

- **What's New**: 이 논문은 Semi-supervised 텍스트 생성을 위해 Deep Generative Pre-trained Transformer 언어 모델과 Generative Adversarial Network(GAN)를 연결하는 새로운 프레임워크를 제안합니다. 제안된 모델은 24층의 대규모 텍스트 코퍼스를 비지도 학습으로 사전 학습한 후, 합성 텍스트 생성을 위한 간단한 GAN 아키텍처를 소개합니다. 실제 데이터를 GAN 샘플로 증강해 Transformer 모델의 파인튜닝에 활용하는 반지도 학습 접근법을 보입니다.

- **Technical Details**: 이 모델은 Gumbel-Softmax를 적용하여 토큰의 이산성을 처리하고, GAN 아키텍처를 통해 생성된 샘플로 언어 모델을 세밀하게 조정하는 방법을 설명합니다. Transformer 모델은 자기 주의 메커니즘을 기반으로 하여, 긴 의존성을 포착하는 데 효과적이며, GPT-2와 같은 모델의 이점도 충분히 활용합니다. Gumbel-Softmax 기법은 이산 샘플링의 비미분가능성을 극복하는 기술로, 연속적이고 미분 가능한 참조를 제공합니다.

- **Performance Highlights**: 본 논문에서는 제안한 모델의 성능을 분석하기 위해 훈련 곡선 및 성능 비교를 제공합니다. 실험 결과, 실제 데이터의 일부와 GAN에서 생성된 샘플을 통합하여 텍스트 생성을 최적화하는 반지도 학습 방식이 효과적임을 보여줍니다. 이러한 접근은 자연어 처리(NLP) 분야에서의 데이터 증대와 모델의 일반화 능력을 향상시킬 수 있는 가능성을 제시합니다.



### Learning to Substitute Words with Model-based Score Ranking (https://arxiv.org/abs/2502.05933)
- **What's New**: 이번 연구에서는 기존의 인간 주석(data labeling)에 의존하지 않고 BARTScore라는 모델 기반의 평가 지표를 사용하여 문장 품질을 정량화하는 방법을 제안합니다. 이 접근법은 단순히 단어 대체의 질을 평가하는 것을 넘어서, 모델의 예측과 품질 점수 간의 정렬을 최적화하는 새로운 손실 함수(loss function)를 도입하였습니다. 결과적으로, 모델 학습에서 인간의 레이블이 필요 없어져 주석 비용을 절감하면서도 단어 대체의 품질을 유지할 수 있습니다.

- **Technical Details**: 저자들은 Smart Word Substitution(SWS) 작업을 통해 문서 내 단어 대체의 효율성을 높이고, 이를 위해 BARTScore를 활용하여 각 제안의 품질을 측정합니다. 기존의 masked language models(예: BERT, BART) 및 large language models에서 측정된 품질 점수와의 정렬 손실을 사용하여 단어 대체 제안을 최적화하는 새로운 방법론을 개발했습니다. 연구 결과는 제안된 방법이 기존 모델들보다 더 뛰어난 성능을 보임을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 BERT, BART와 같은 masked language models뿐만 아니라 GPT-4, LLaMA와 같은 large language models에 비해서도 우수한 성능을 기록하였습니다. 연구 결과는 문장 품질 향상과 관련된 SWS 작업에서 새로운 기준을 제시하며, 기존의 접근 방식에 비해 더 효율적이고 비용 절감이 가능한 방법임을 보여줍니다. 모든 코드와 자료는 별도로 제공되며, 큰 규모의 데이터 요구 없이도 뛰어난 성능을 발휘할 수 있음을 강조합니다.



### ARISE: Iterative Rule Induction and Synthetic Data Generation for Text Classification (https://arxiv.org/abs/2502.05923)
Comments:
          Accepted to Findings of NAACL 2025

- **What's New**: 이 논문에서는 ARISE라는 새로운 프레임워크를 제안하며, 반복적으로 규칙을 유도하고 텍스트 분류를 위한 합성 데이터를 생성합니다. 이 프레임워크는 부트스트래핑(bootstrapping)을 통해 합성 데이터 생성과 자동 규칙 유도를 결합하여 생성된 규칙과 데이터를 필터링합니다. 단순히 생성된 규칙만으로도 in-context learning (ICL) 및 fine-tuning (FT) 설정에서 성능 향상이 이루어집니다.

- **Technical Details**: ARISE는 사용 가능한 학습 데이터를 시드(seed)로 삼아 시작합니다. LLMs를 활용하여 합성 후보 예시를 생성하고, 동시적으로 구문 n-그램을 통해 유도된 규칙 후보를 생성합니다. 생성된 규칙은 서브모듈러 그래프 컷(submodular graph cut) 기반의 함수로 필터링되며, 각 예시와 규칙은 해당 작업과 관련된 레이블이 부여됩니다.

- **Performance Highlights**: ARISE에서 생성된 예시와 규칙을 사용한 결과, 모든 실험 설정에서 통계적으로 유의미한 성과 향상을 보였습니다. 특히, ARISE를 활용한 완전 촬영(full-shot) 및 소수 촬영(few-shot) 실험에서 최신 기술(State of the art) 결과를 기록했으며, 여러 언어와 영역에서의 일반화 가능성을 입증하였습니다. 합성 데이터만으로도 기존의 복잡한 방법론을 초월하는 성능 향상을 달성하였습니다.



### GRAIT: Gradient-Driven Refusal-Aware Instruction Tuning for Effective Hallucination Mitigation (https://arxiv.org/abs/2502.05911)
Comments:
          Equal contribution: Runchuan Zhu, Zinco Jiang, Jiang Wu; Corresponding author: Conghui He

- **What's New**: 본 논문은 Refusal-Aware Instruction Tuning (RAIT) 개선을 위해 Gradient-driven Refusal Aware Instruction Tuning Framework인 GRAIT를 제안하고 있습니다. GRAIT는 두 가지 주요 과제인 환각(hallucinations) 최소화와 지나친 거절(over-refusal) 방지를 동시에 해결하기 위한 새로운 접근 방식을 제공합니다. 이를 통해 LLMs가 보다 신뢰성 있는 응답을 제공할 수 있도록 지원합니다.

- **Technical Details**: GRAIT는 두 가지 이론적 관찰에 기반하여 설계되었습니다. 첫째, 환각을 줄이기 위한 Refusal Influence 공식을 활용하여 LLM이 거부 패턴을 학습하면서 비효율적인 샘플을 필터링합니다. 둘째, Adaptive weighting 방법을 통해 정확한 응답을 제공할 수 있는 질문에 대해서는 지나친 거절을 피할 수 있도록 샘플 가중치를 조정합니다.

- **Performance Highlights**: GRAIT의 실험 결과는 공개형 및 객관식 질문 응답 작업에서 기존의 RAIT 방법보다 명백히 우수한 성능을 보여주었습니다. 특히, GRAIT는 환각 비율을 상당히 감소시켜 신뢰성과 유용성을 향상시켰습니다. 이러한 결과는 GRAIT가 기존의 방법들에 비해 LLM의 응답 품질을 개선하는 데 기여할 수 있음을 입증합니다.



### A Distributional Perspective on Word Learning in Neural Language Models (https://arxiv.org/abs/2502.05892)
- **What's New**: 이 연구는 언어 모델(LM)이 아동의 단어 학습 경로와 어떻게 상관관계가 있는지를 조사하는 데 중점을 두고 있습니다. 저자들은 기존의 분포적 접근 방식이 단어 학습에서 중요한 정보를 포착하는 데 실패한다고 주장하며, 새로운 분포적 서명을 제안합니다. 이 서명은 단어의 적절성과 부적절성에 대한 지식을 나누어 평가합니다.

- **Technical Details**: 연구팀은 아동의 입력 데이터와 유사한 세 가지 데이터 세트를 사용하여 언어 모델을 처음부터 학습시켰습니다. 이 과정에서 단어의 학습 서명을 기록하고, 단어 습득 시점(AoA)을 평가하기 위한 기준을 설정합니다. 최종적으로, 이 연구는 다양한 분포적 서명의 상대적 유용성과 언어 모델과 아동의 단어 습득 순서의 비교를 다룹니다.

- **Performance Highlights**: 이 연구의 분석 결과, 여러 서명이 단어 학습의 여러 측면을 잘 나타내지 못하며, 아동의 단어 습득 패턴과는 상관관계가 낮음을 발견했습니다. 이는 현재의 방법들이 인간 언어 습득 모델로의 언어 모델의 한계를 강조합니다. 따라서 저자들은 향후 연구에서 제안된 새로운 서명을 사용하여 LM의 학습 경로를 개선할 것을 촉구하고 있습니다.



### MTPChat: A Multimodal Time-Aware Persona Dataset for Conversational Agents (https://arxiv.org/abs/2502.05887)
Comments:
          NAACL 2025 Findings

- **What's New**: MTPChat 데이터셋은 대화 응답과 기억이 시간에 따라 변하는 점을 처음으로 모델링한 다중모드(time-aware) 데이터셋입니다. 기존의 데이터셋들이 텍스트 기반 질문-응답(QA) 작업에 집중하고 있는 반면, MTPChat은 대화 및 기억의 자연스러운 흐름을 활용하여 인간의 인지에서의 시간적 변화를 시뮬레이션합니다. 이 데이터셋은 언어적, 시각적 요소 및 시간 정보를 통합하여 대화 모델의 복잡성과 현실성을 높이고 있습니다.

- **Technical Details**: MTPChat은 Temporal Next Response Prediction (TNRP)와 Temporal Grounding Memory Prediction (TGMP)이라는 두 가지 새로운 작업을 통해 모델의 시간 암시 큐(implicit temporal cues) 및 진화하는 응답 추적 능력을 평가합니다. 적응형 시간 모듈(adaptive temporal module)을 포함한 혁신적인 프레임워크를 제안하여 다중 모드 스트림을 효과적으로 통합하고 시간 의존성을 포착합니다. 이 모듈은 시간적 관련성을 바탕으로 특징들을 동적으로 병합하여 다중모드 통합의 일관성을 향상시킵니다.

- **Performance Highlights**: MTPChat을 평가하기 위해 SBERT와 CLIP 모델을 사용한 실험 결과, MTPChat은 다중모드 및 시간 민감한 시나리오에서 새로운 도전을 제시하고 있음을 보여주었습니다. 우리의 적응형 시간 모듈은 다른 특징 통합 방법에 비해 뛰어난 성능을 보이며, 모델의 다중모드 시간 인식 대화에 대한 추론 능력을 크게 향상시킵니다. 이 연구의 주요 기여는 MTPChat 데이터셋을 통해 시간 민감 대화 AI 연구의 발전을 도모함으로써, 모델이 인간 수준의 시간적 이해를 성취하도록 하는 것입니다.



### Enhancing Depression Detection with Chain-of-Thought Prompting: From Emotion to Reasoning Using Large Language Models (https://arxiv.org/abs/2502.05879)
- **What's New**: 이번 연구에서는 우울증 탐지의 성능 및 해석 가능성을 향상시키기 위해 Chain-of-Thought Prompting 접근 방식을 제안합니다. 이 방법은 감정 분석, 이분법적 우울증 분류, 기저 원인 식별, 그리고 중증도 평가의 네 가지 단계로 탐지 과정을 세분화합니다. 이러한 구조화된 과정은 우울증 진단에 더 명확한 이유를 제공해줍니다.

- **Technical Details**: 우울증 탐지의 방법론은 네 개의 단계로 나뉘며, 각 단계는 정서 분석, 이분법적 분류, 기인 요인 분석, 그리고 중증도 평가로 구성됩니다. 이 구조는 PHQ-8 점수 체계를 기반으로 하여 우울증의 중증도를 평가하며, 다양한 사회적, 생물학적, 심리적 요인을 고려하여 개별적인 평가를 가능하게 합니다. 이를 통해 입력된 텍스트의 정서적 신호를 세밀하게 분석합니다.

- **Performance Highlights**: 제안된 방식은 이 기존 모델들보다 우울증 분류 정확성과 진단 통찰의 세부사항 측면에서 더 뛰어난 성과를 보여주었습니다. E-DAIC 데이터셋을 기반으로 한 실험 결과, 모델의 성능 개선이 입증되었으며, 이는 임상 적용 가능성을 높입니다. 연구는 뚜렷한 변화를 드러내며, 우울증 조기 발견 및 개입을 위한 새로운 가능성을 제시합니다.



### Enhancing Financial Time-Series Forecasting with Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2502.05878)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 재무 시계열 예측을 위해 특별히 설계된 첫 번째 검색 보강 생성(retrieval-augmented generation, RAG) 프레임워크인 FinSeer를 제안합니다. 기존 방법의 한계를 극복하기 위해, 10억 개 파라미터의 대형 언어 모델(StockLLM)을 주요 모델로 사용하고, LLM 피드백에 의해 개선된 후보 선택 방법과 역사적 의미 있는 시퀀스와의 유사성을 극대화하는 훈련 목표를 포함합니다. 이러한 혁신을 통해 복잡한 재무 데이터에서 의미 있는 패턴을 발견하면서 노이즈를 효과적으로 최소화할 수 있습니다.

- **Technical Details**: FinSeer는 주식 가격과 재무 지표를 분석하여 시계열 예측을 수행하는 데 중점을 둡니다. 이 모델은 고유한 리트리버 방식으로, 후보 풀에서 관련된 시퀀스를 검색하고 이를 StockLLM의 입력 맥락으로 통합하여 예측을 수행합니다. 또한, 20개의 재무 지표와 통합된 새로운 데이터셋을 구축하여 경제적 분석의 실제 상황을 반영하도록 설계되었습니다. 결과적으로, FinSeer는 이를 통해 과거 시계열 데이터를 더 잘 해석하여 주식 움직임 예측의 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RAG 프레임워크는 기존의 StockLLM 및 무작위 검색 방법에 비해 성능이 우수함을 입증했습니다. 특히 FinSeer는 BIGDATA22 벤치마크에서 8% 높은 정확도를 기록하며 기존의 검색 방법보다 더 영향력 있는 시퀀스를 검색했습니다. 이는 RAG 프레임워크가 재무 예측에 맞춤형 검색 모델의 중요성을 강조하고 있으며, 향후 연구를 위한 새로운 확장 가능한 프레임워크를 제공함을 보여줍니다.



### Self-Training Large Language Models for Tool-Use Without Demonstrations (https://arxiv.org/abs/2502.05867)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 증강하기 위한 도구 사용을 학습하는 방법을 탐구합니다. 기존 방법은 종종 정리된 gold 도구 사용 시연을 요구하는 반면, 본 연구는 시연 없이 LLMs가 도구를 사용할 수 있는지를 분석합니다. 제로-샷 프롬프트 기법을 활용하여 LLM이 도구를 사용하는 방법을 안내하고, 자가 학습(self-training) 방법을 제안해 도구 사용을 위한 데이터셋을 생성합니다.

- **Technical Details**: 연구에서는 SFT(supervised fine-tuning)와 PFT(preference fine-tuning) 기술을 통해 Question Answering(QA) 데이터셋인 TriviaQA와 GSM8K를 활용하여 모델을 미세 조정합니다. SFT는 LLM의 성능을 특정 다운스트림 작업에서 향상시키기 위한 방법이며, PFT는 LLM의 응답을 인간의 선호에 맞추기 위해 페어와이즈(preference) 데이터에 기반하여 미세 조정하는 기법입니다. 이 논문은 LLM이 도구 사용을 위한 데이터를 스스로 생성할 수 있는지를 다루고 있습니다.

- **Performance Highlights**: 실험 결과, 포괄적인 QA 데이터셋에서 LLMs는 도구를 어느 정도 사용할 수 있지만 모델의 규모와 작업에 따라 성능 저하가 발생할 수 있습니다. 자가 학습 방법을 통해 생성된 데이터에서 LLM은 PopQA라는 장기 지식 작업에서 성능이 향상되었지만, 다른 데이터셋에서는 혼합된 성과를 나타냈습니다. 이러한 발견은 LLM이 명시적인 시연 없이 도구 사용을 학습할 수 있음을 보여주지만, 부적절한 도구 사용은 추가적인 문제를 야기할 수 있음을 시사합니다.



### Fact-or-Fair: A Checklist for Behavioral Testing of AI Models on Fairness-Related Queries (https://arxiv.org/abs/2502.05849)
Comments:
          8 pages of main text; 7 pages of appendices;

- **What's New**: 이번 연구는 Google의 Gemini 이미지 생성기에서 발생한 인종적 편향 문제를 다루며, 사실성(factuality)과 다양성(diversity)을 균형 있게 반영하는 것이 얼마나 복잡한지를 조명합니다. 연구팀은 19개의 실제 통계를 기반으로 체크리스트인 Fact-or-Fair를 개발하였습니다. 이 체크리스트는 대규모 언어 모델(LLMs) 및 텍스트-이미지(T2I) 모델의 성능을 평가하는 데 사용되며, 사실성과 공정성을 평가하는 지표를 포함하고 있습니다.

- **Technical Details**: 연구에서는 미국의 다양한 사회적 지표에 대한 19개의 통계를 수집하였으며, 주관적 및 객관적 쿼리를 통해 모델의 편향성과 사실성을 분석합니다. 객관적 쿼리는 모델이 사실 정보를 정확히 반영하는 능력을 평가하며, 주관적 쿼리는 통계적 오류를 기반으로 편향을 유도하여 공정성을 검증합니다. 또한, 연구는 사실성과 공정성을 수치화하기 위한 지표를 도출하고, 이 두 aspect 간의 Trade-off를 수학적으로 정립하였습니다.

- **Performance Highlights**: 연구 결과, GPT-4o와 DALL-E 3 모델이 상위 성능을 기록하며, 나머지 4개의 T2I 및 6개의 LLM보다 탁월한 평가를 받았습니다. 연구진은 이 모델들이 현실 세계의 정보에 대한 정확성과 다각적인 편견을 줄이는 능력을 보여주었다고 언급합니다. 전체적으로 본 연구는 AI 모델 개발 상태를 평가하는 데 중요한 통찰력을 제공합니다.



### LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification (https://arxiv.org/abs/2502.05836)
Comments:
          Accepted on NAACL 2025

- **What's New**: 이 논문에서는 인도 법원의 판결을 중심으로 합법적인 문서의 의미적 세분화 및 수사학적 역할 분류 과제를 다루고 있습니다. LegalSeg라는 7,000개 이상의 문서와 140만 문장으로 구성된 대규모 주석 데이터셋을 새롭게 소개하며, 이는 법적 문서 처리를 위한 가장 큰 데이터셋입니다. 여러 최첨단 모델(예: Hierarchical BiLSTM-CRF, ToInLegalBERT, GNN 등)의 성능을 평가하고, 문맥과 구조적 관계를 통해 분류 정확도를 높였음을 보여줍니다.

- **Technical Details**: 논문에서는 법적 문서에서의 의미적 세분화를 위한 다양한 모델을 구현하여 평가하였습니다. Hierarchical BiLSTM-CRF 모델은 계층적 접근을 통해 문맥 정보를 캡처하고, MultiTask Learning은 역할 전환을 고려하여 수사학적 역할을 보다 정교하게 식별합니다. 이외에도 InLegalToBERT, Graph Neural Networks(GNNs), Role-Aware Transformers와 같은 새로운 접근 방식을 도입하여 모델의 표현력과 문맥 처리를 향상시켰습니다. 특히, GNN은 문장을 노드로 표현하여 정보 전파와 문맥 캡처를 효과적으로 수행합니다.

- **Performance Highlights**: 법적 문서의 세분화 및 이해도 향상에 대한 기여로 LegalSeg 데이터셋과 연구 결과는 법적 NLP 분야에서 중요한 기초를 제공합니다. 특히, RhetoricLLaMA를 통해 복잡한 법적 언어 처리를 위한 인스트럭션 튜닝된 대형 언어 모델의 가능성과 한계를 강조하였습니다. 모델 성능 평가 결과, 문맥이나 주변 문장의 레이블 활용이 분류 정확도에 긍정적인 영향을 미친 것으로 나타났습니다.



### Delta - Contrastive Decoding Mitigates Text Hallucinations in Large Language Models (https://arxiv.org/abs/2502.05825)
- **What's New**: 논문에서 제안하는 Delta는 text 기반의 대형 언어 모델(LLM)에서 발생하는 hallucination을 효과적으로 완화하는 새로운 방법입니다. 기존 모델 retraining이나 추가 데이터 없이 inference 과정에서 직접 hallucination을 줄일 수 있는 방안으로서, 입력 프롬프트의 일부를 랜덤하게 마스킹하고 원본 및 마스킹된 입력에 대한 출력 분포를 대비시킵니다. 이 접근 방식은 LLM을 실제 시스템에 배포하기 용이하고 계산 효율성이 높은 특징을 지닙니다.

- **Technical Details**: Delta는 contrastive decoding 방식을 활용하여 마스킹된 입력과 마스킹되지 않은 입력의 출력을 비교합니다. 이 방법은 입력 시퀀스의 토큰을 랜덤하게 마스킹하여, 특정한 경우에 hallucination이 발생할 가능성이 높은 출력을 생성하도록 유도합니다. 이후, 마스킹된 입력에서 생성된 로그잇(logits)을 원래의 로그잇에서 빼는 과정을 통해, hallucination에 덜 영향을 받는 '클린' 로그잇을 추출하여 정확한 출력을 도출합니다.

- **Performance Highlights**: Delta의 성능은 SQuAD v1.1 및 v2 데이터셋에서 각각 약 3 및 6%의 정확도 향상을 달성하였으며, SQuAD v2의 no-answer exact match에서 10% 이상의 점수 개선을 보여 주었습니다. TriviaQA 및 Natural Questions 데이터셋에서도 개선 효과를 나타내어, context-rich한 데이터셋에서 hallucination을 완화하고 성능을 높이는 데 효과적임을 입증하였습니다.



### Structural Perturbation in Large Language Model Representations through Recursive Symbolic Regeneration (https://arxiv.org/abs/2502.05794)
- **What's New**: 이 논문에서는 'Recursive Symbolic Regeneration'을 통한 구조적 섭동(Structural Perturbation)이라는 새로운 개념을 소개합니다. 이 접근법은 LLM의 내부 표현을 직접 수정하지 않고도 상징적 구조를 재귀적으로 생성함으로써 영향을 미치려는 목표를 가지고 있습니다. 기존의 정밀 조정(fine-tuning) 방식과 달리, 이 기술은 부분적인 매개변수 조정 없이 모델의 출력을 조정합니다. 이로 인해 모델의 일반화 능력에 대한 우려를 최소화하고, 더 나은 해석 가능성(interpretability) 및 조작(control) 가능성을 제공합니다.

- **Technical Details**: 이 연구에서는 LLM의 성능 메트릭에 대한 기술적 영향을 평가하기 위해 실험을 수행하였습니다. 구조적 섭동을 통해 상징적 구조를 재귀적으로 생성하여 모델의 출력(coherence, relevance, diversity 등)에 미치는 영향을 분석하였습니다. 이 방식은 매개변수에 대한 직접적인 수정을 피하는 동시에, 내부 상징적 표현에 대한 조작을 통해 모델의 행동을 변화시킵니다. 이러한 실험들은 기존의 방법들보다 비침습적(non-invasive)으로 LLM의 행위를 수정할 수 있는 잠재력을 보여줍니다.

- **Performance Highlights**: 실험 결과, 구조적 섭동은 모델의 출력을 조절하면서 언어 모델의 유창성과 일관성을 유지하는 데 중요한 역할을 한다는 것이 나타났습니다. 긴 문장 생성 과정에서 주제 일관성을 높이는 데 기여하며, 레거시(Legacy) 모델에서의 구조 적 변화가 맥락 민감성을 다르게 만든다는 점을 강조했습니다. 또한, 상징적 조작(Symbolic manipulations)에 의한 생성 응답의 다양성이 향상된 것으로 보이며, 이는 텍스트 생성 자동화에서 스타일 조정을 보다 효과적으로 가능하게 합니다.



### On Reference (In-)Determinacy in Natural Language Inferenc (https://arxiv.org/abs/2502.05793)
Comments:
          NAACL 2025 Findings

- **What's New**: 이번 연구에서는 자연어 추론(Natural Language Inference, NLI) 데이터셋의 라벨링 과정에서 사용되는 참조 결정성(reference determinacy, RD) 가정의 영향을 재조명합니다. RD는 가정을 기반으로 하여 전제(premise)와 가설(hypothesis)이 동일한 맥락을 지시한다고 정의되며, 이는 NLI 응용 프로그램에서 한계를 초래할 수 있습니다. 연구진은 RefNLI라는 진단 벤치마크를 도입하여 NLI 모델들이 맥락 불일치를 인식하지 못하는 경향이 있음을 밝혔습니다.

- **Technical Details**: RefNLI 벤치마크는 위키피디아에서 전제를 검색하여 가설과 동일한 맥락을 참조하지 않는 예제를 포함하고 있습니다. 이 연구에서 제시된 데이터셋은 총 1,143개의 NLI 쌍으로 구성되어 있으며, 인간 전문가의 라벨 평가를 포함합니다. 연구팀은 훈련된 NLI 모델들이 이러한 예제를 통해서 문맥의 불일치를 인식하는 데 실패하며, 그 결과 80% 이상의 잘못된 모순(false contradiction) 및 50% 이상의 포함(entailment) 예측이 발생하는 것을 관찰했습니다.

- **Performance Highlights**: 또한, NLI 모델이 FEVER 및 VitaminC와 같은 사실 검증(fact verification) 작업에 어떻게 적용될 수 있는지를 조사했습니다. 다양한 훈련 데이터셋을 사용한 결과, 모든 NLI 데이터셋에서 참조 결정성 편향이 존재함을 확인하였고, 이는 모델이 문맥 이해에 한계를 두는 원인으로 작용할 수 있습니다. 연구팀은 이러한 데이터셋이 인과 관계를 유도하는 방식이 실제로 사람들 간의 NLI 레이블에 대한 불일치(disagreement)를 설명할 수 있음을 발견했습니다.



### Reinforced Lifelong Editing for Language Models (https://arxiv.org/abs/2502.05759)
- **What's New**: 이 논문에서는 Reinforcement Learning (RL) 기반의 모델 편집 기법인 RLEdit를 제안합니다. RLEdit는 모델 파라미터를 재훈련 없이 수정할 수 있도록 하여 대량 언어 모델(LLM)의 지속적인 업데이트 문제를 해결합니다. 기존의 하이퍼네트워크 기반 방법들이 100번 이상의 편집 작업에서 성능이 저하되는 문제를 지적하며, RLEdit는 RL의 관점을 통해 이를 극복하려 합니다.

- **Technical Details**: RLEdit는 모델 편집 손실을 보상으로 취급하고, 하이퍼네트워크 파라미터를 전체 지식 시퀀스 수준에서 최적화함으로써 LLM의 변화를 정밀하게 반영하여 적절한 파라미터 업데이트를 생성합니다. RL을 통해 현재의 LLM 상태에 적응적으로 Δ(델타) 업데이트를 제공합니다. 이는 Markov Decision Process (MDP) 구조에서 생애 주기 내내 효과적으로 작동할 수 있게 해줍니다.

- **Performance Highlights**: RLEdit는 여러 LLM에 대한 실험에서 기존의 파라미터 수정 방법들과 비교하여 압도적으로 우수한 성능을 보였습니다. 평균적으로 2%의 계산 시간만으로도 기존 방법들보다 59.24%의 성능 향상을 달성하며, 20,000회 이상의 편집 후에도 0.17초의 속도를 유지합니다. RLEdit는 하이퍼네트워크 기반 방법의 활용 가능성을 넓히고, 장기 편집 문제를 RL 문제로 모델링한 최초의 사례로 자리잡고 있습니다.



### BnTTS: Few-Shot Speaker Adaptation in Low-Resource Setting (https://arxiv.org/abs/2502.05729)
Comments:
          Accepted paper in NAACL 2025

- **What's New**: 이 논문은 방글라어(TTS) 시스템을 위한 최초의 프레임워크인 BnTTS(Bangla Text-To-Speech)를 소개합니다. 방글라어 음성 합성을 위한 기존의 모델들의 한계를 극복하고, 최소한의 훈련 데이터로도 효율적인 음성 합성을 가능하게 합니다. 이 시스템은 멀티링구얼 TTS 파이프라인에 방글라어를 통합하고, 방글라어의 음향적, 언어적 특성을 고려한 수정 사항을 포함합니다.

- **Technical Details**: BnTTS는 3.85k 시간의 방글라어 음성 데이터 세트로 사전 훈련을 거칩니다. 이 모델은 텍스트 시퀀스와 화자의 mel-spectrogram을 기반으로 음성을 생성하는 것을 목표로 하며, 복잡한 아키텍처 수정이 포함되어 있습니다. Audio Encoder는 VQ-VAE(Vector Quantized-Variational AutoEncoder)를 적용하고, Conditioning Encoder와 Perceiver Resampler가 음성을 처리하여 변환합니다.

- **Performance Highlights**: 실험 결과, BnTTS는 소수 샷(few-shot) 환경에서 합성된 방글라어 음성의 자연스러움, 이해도, 화자 충실도를 획기적으로 향상시킵니다. 주관적 평균 의견 점수(SMOS)와 자연스러움, 명확성 지표에서 기존의 최신 방글라어 TTS 시스템에 비해 뛰어난 성능을 보입니다. 이러한 결과는 방글라어 TTS가 저자원 언어에서도 유망한 접근 방식을 제공할 수 있음을 보여줍니다.



### Rethinking Word Similarity: Semantic Similarity through Classification Confusion (https://arxiv.org/abs/2502.05704)
Comments:
          Accepted to NAACL-main-2025

- **What's New**: 이 논문에서는 전통적인 단어 유사성 측정 방법의 한계를 극복하기 위해 새로운 유사성 측정 방법인 Word Confusion을 제안합니다. 이 방법은 단어 임베딩 간의 코사인 유사도에 의존하는 대신, 단어의 맥락에 따른 특징 기반의 분류 혼동을 활용합니다. 이를 통해 비대칭적이고 다의적인 의미 유사성을 잘 반영할 수 있는 방법으로 재구성합니다.

- **Technical Details**: Word Confusion은 맥락 임베딩을 단어 정체성에 매핑하기 위해 분류기를 훈련하고, 분류기의 혼동(confusion) 확률을 두 단어 간의 유사성 측정으로 사용합니다. 훈련 과정에서는 BERT를 활용하여 각 단어와 관련된 문맥 임베딩을 추출한 후, 로지스틱 회귀 분류기를 훈련합니다.Inference 단계에서는 새로운 단어의 맥락 임베딩을 추출하고, 훈련된 분류기를 사용하여 특정 클래스(차원)로부터 어떤 단어가 가장 유사한지를 판단합니다.

- **Performance Highlights**: Word Confusion은 MEN, WirdSim353 및 SimLex와 같은 여러 데이터 세트에서 인간의 유사성 판단과의 일치도에서 코사인 유사도와 유사한 성능을 보입니다. 특히, 이 방법은 시간적 의미 변화와 같은 실제 데이터 탐색 작업에서도 유용성을 보여주었습니다. 결과적으로, 이 접근 방식이 문화 분석(cultural analytics) 및 계산 사회 과학(computational social science) 분야의 발전을 위한 새로운 도구 개발로 이어지기를 기대합니다.



### Zero-Shot End-to-End Relation Extraction in Chinese: A Comparative Study of Gemini, LLaMA and ChatGP (https://arxiv.org/abs/2502.05694)
- **What's New**: 이번 연구는 중국어에서 제로샷(Zero-shot) 엔드투엔드(End-to-end) 관계 추출(Relation Extraction, RE)에 대한 다양한 대형 언어 모델(LLMs)의 성능을 조사합니다. 기존 연구는 주로 영어에 집중했거나 주석이 달린 엔티티를 가정했기 때문에 중국어에 대한 LLM의 효과는 거의 탐색되지 않았습니다. 본 연구에서는 ChatGPT, Gemini, LLaMA 3개의 모델을 평가하여 정확도, 효율성 및 적응성을 비교하고, 각 모델의 장단점을 제시합니다.

- **Technical Details**: 제로샷 RE는 주석된 예 없이 모델의 사전 훈련된 지식과 추론 능력을 통해 엔티티와 그 관계를 추출하는 것을 목표로 합니다. 연구에서는 DuIE 2.0 데이터셋을 사용하여 모델들이 입력된 문장에서 엔티티-관계 트리플을 추출하도록 하고, 이에 대해 조합 정확도와 조합 재현율, 조합 F1 점수를 확인합니다. 또한 의미적 일치(Semantic Matching) 평가 방법을 도입하여 엔티티 인식과 관계 표현에서의 불일치를 완화하는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, OpenAI의 gpt-4-turbo 모델이 가장 높은 F1 점수인 0.367을 기록하며 제로샷 관계 추출 작업에 대해 가장 잘 적합한 것으로 나타났습니다. Gemini 모델은 비교적 높은 재현율을 보였지만 정확도는 중간 수준에 머물렀습니다. 반면 LLaMA 모델은 모든 지표에서 저조한 성능을 보였으며, gpt-4 모델은 실시간 적용 가능성 측면에서도 효과적인 라탠시를 기록했습니다.



### Investigating the Shortcomings of LLMs in Step-by-Step Legal Reasoning (https://arxiv.org/abs/2502.05675)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 법적 추론 능력에 대해 심층 분석을 진행하여 기존의 연구와 차별화된다. 연구에서는 LLM의 오류를 시스템적으로 분류하기 위한 새로운 오류 분류 체계를 제안하고, LLM의 성과를 평가하기 위한 자동화된 평가 프레임워크를 개발하였다. 이 프레임워크는 LLM이 법적 시나리오에서 수행하는 단계별 추론의 질을 평가하는 데 중점을 두고 있으며, 오류 발생 지점을 세밀하게 분석한다.

- **Technical Details**: 이를 위해, 연구진은 'Civil Procedure' 데이터셋에서 대학 수준의 다중 선택 질문(MCQA) 과제를 활용하였고, 각 LLM의 추론 연쇄를 분석하였다. 제안된 자동 결정을 통해, 추론의 'soundness'와 'correctness'라는 두 가지 주요 지표가 사용되며, 이는 각 LLM이 법적 맥락에서 어떻게 논리적으로 추론하는지를 평가하는 중요한 기준이 된다. 이 연구는 민감한 법적 문맥에서 LLM이 반드시 충족해야 할 다양한 규칙과 선례를 평가하도록 설계되었다.

- **Performance Highlights**: 결과적으로, 오류 분류 체계를 피드백으로 활용한 여러 프롬프트 전략을 통해 LLM의 법적 추론 성능이 4%가량 향상되었다. 이를 통해 LLM이 법적 상황에서 논리적인 결정을 내리는 데 있어 세밀한 오류 분석이 중요하다는 점을 시사하였다. 본 연구는 또한 이를 바탕으로 한 자동화된 평가 프레임워크가 향후 복잡한 작업에서의 단계별 추론 분석에 유용할 것이라고 결론짓는다.



### Language Models Largely Exhibit Human-like Constituent Ordering Preferences (https://arxiv.org/abs/2502.05670)
Comments:
          NAACL 2025 Main Conference

- **What's New**: 이 논문은 영어 문장에서 구성 요소의 순서 변화가 구성 요소의 무게와 복잡성과 관련이 있다는 이론을 기반으로 현대 대형 언어 모델(LLMs)의 동작을 연구합니다. 특히, Heavy NP Shift (HNPS), Particle Movement (PM), Dative Alternation (DA), Multiple PP Shift (MPP)의 네 가지 유형의 이동을 분석하여 LLM이 인간의 언어 처리 패턴과 얼마나 유사한지를 평가합니다.

- **Technical Details**: 연구에서는 구성 요소 이동을 모델링하기 위해 다양한 무게 측정값(예: word length, syllable weight, token length 등)을 사용합니다. 구성 요소의 이동은 주로 문장의 동사 뒤에 위치한 구성 요소들을 다른 위치로 재배치하는 과정을 의미하며, 원래의 의미는 유지됩니다. 이 연구는 인위적 및 자연 발생적 데이터를 포함한 새로운 구성 요소 이동 데이터 세트를 활용하여 LLM의 선호도를 평가합니다.

- **Performance Highlights**: 결과적으로 LLM은 기존의 문법 이론과 유사하게 구성 요소 순서에 대한 선호를 보이는 경향이 있음이 밝혀졌습니다. 특히 파티클 이동에서 예상치 못한 성과를 보였지만, 전반적으로 LLM은 인간의 선호와 일치하는 패턴을 따릅니다. 이러한 연구 결과는 인간 언어 처리와 LLM의 관계를 이해하는 데 중요한 통찰을 제공합니다.



### CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging (https://arxiv.org/abs/2502.05664)
Comments:
          Accepted in NAACL 2025 Findings

- **What's New**: 이번 논문에서는 CodeSim이라는 새로운 다중 에이전트 코드 생성 프레임워크를 소개합니다. 이 프레임워크는 프로그램 합성의 여러 단계인 계획 수립(planning), 코딩(coding), 디버깅(debugging)을 인간의 인지(perception) 방식에 따라 종합적으로 다룹니다. CodeSim은 입력 및 출력을 단계별로 시뮬레이션(simulation)하여 계획 검증(plan verification) 및 내부 디버깅을 수행하는 독창적인 방법을 특징으로 합니다.

- **Technical Details**: CodeSim은 기존의 외부 도구 기반의 반복 디버거(iterative debuggers) 방식에서 벗어나 초기 코드 생성의 품질 문제를 해결하기 위해 설계되었습니다. 연구팀은 이를 통해 코드 생성의 정확성을 향상시키며, 다양한 방법으로 생성된 프로그램을 정교하게 다듬을 수 있습니다. 실험은 7개의 도전적인 문제 해결 및 프로그램 합성 벤치마크에서 진행되었으며, 그 결과 CodeSim의 뛰어난 코드 생성 능력이 입증되었습니다.

- **Performance Highlights**: CodeSim은 HumanEval 95.1%, MBPP 90.7%, APPS 22%, CodeContests 29.1% 등 새로운 최첨단(pass@1) 결과를 달성했습니다. 특히, 외부 디버거와 연계(cascaded)할 경우 더 큰 성능 향상이 가능하다는 점도 강조되었습니다. 이 연구는 오픈 소스를 통해 추가 연구 및 개발을 촉진하기 위해 공개되었습니다.



### KMI: A Dataset of Korean Motivational Interviewing Dialogues for Psychotherapy (https://arxiv.org/abs/2502.05651)
Comments:
          Accepted at NAACL 2025 Main Conference

- **What's New**: 이 논문은 AI 주도 정신 건강 지원을 위한 챗봇의 발전을 위해 새로운 출발점을 제시합니다. 특히, Motivational Interviewing (MI) 이론에 기반한 최초의 한국어 대화 데이터셋인 KMI를 소개하며, 이를 통해 기존 데이터셋의 한계를 극복하고자 합니다. KMI는 1,000개의 고품질 한국어 MI 대화를 포함하고 있으며, 전문가 평가를 통해 데이터셋의 질과 유용성을 입증했습니다.

- **Technical Details**: 제안된 방법은 두 개의 시뮬레이터인 치료사 시뮬레이터와 고객 시뮬레이터를 포함하여, 실시간으로 발화를 생성하는 구조를 가집니다. 이를 통해 전문가 치료사 모델로부터 얻어진 행동 선택을 모방하여 MI 발생을 예측하는 MI 예측 모델을 훈련시킵니다. 그 결과, KMI 데이터셋은 한국 사회의 정신 건강 문제를 반영하여 실제 상황에 기반한 대화를 생성합니다.

- **Performance Highlights**: KMI 데이터셋은 전문가들에 의해 광범위한 평가를 통해 MI의 본질을 핵심적으로 포착하며, 챗봇 개발을 위한 실용적인 자원을 제공합니다. 새로운 MI 이론에 기반한 평가 지표를 도입하여, 생성된 대화가 MI의 정신에 얼마나 부합하는지를 직접적으로 측정합니다. 이러한 평가는 KMI의 질과 전문성을 입증하는 중요한 근거가 됩니다.



### Incongruence Identification in Eyewitness Testimony (https://arxiv.org/abs/2502.05650)
Comments:
          9 pages,10 tables. Under review at ACL ARR 2024. Includes supplementary appendix with detailed evaluation results

- **What's New**: 본 논문에서는 목격자 진술에서 불일치(incongruence) 탐지의 새로운 작업을 제안합니다. 이 작업은 2명의 주체가 제시한 질문과 답변의 쌍을 포함하는 진술 쌍을 분석하여 상호 관련된 불일치를 식별하는 것을 목표로 합니다. 여기서는 불일치의 구간을 마킹하는 방법도 포함되어 있습니다.

- **Technical Details**: 우리가 제안하는 INTEND 프레임워크는 6Ws(Who, What, When, Where, Why, and additional What)와 다단계 추론(multi-hop reasoning) 방법론을 기반으로 하여, 목격자 진술의 복잡한 내러티브에서 불일치를 탐지하는 데 초점을 맞추고 있습니다. MIND 데이터셋은 2927개의 문맥적으로 관련된 답변 쌍으로 구성되어 있으며, 명시적 및 암시적인 모순을 포착하도록 설계되었습니다. 이 프레임워크는 각 진술 쌍에서 모순이 발생하는 정확한 구간을 식별하여 불일치를 정량적으로 분석합니다.

- **Performance Highlights**: 실험 결과, INTEND 프레임워크를 활용할 때 불일치 탐지 성능이 +5.63% 향상되었음을 확인했습니다. 우리는 다양한 MLM과 LLM을 대상으로 우리의 접근 방식을 여러 가지 미세 조정 기술(fine-tuning)과 평가하여 F1-score에서 의미 있는 성과 개선을 확인했습니다. 이는 목격자 진술에서 불일치를 탐지하는 새로운 방법론의 효과성을 강조합니다.



### Gender Bias in Instruction-Guided Speech Synthesis Models (https://arxiv.org/abs/2502.05649)
Comments:
          NAACL 2025 Findings

- **What's New**: 최근의 음성 합성(Expressive Speech Synthesis) 기술 발전 덕분에 텍스트를 기반으로 특정한 스타일의 음성을 생성할 수 있는 가능성이 높아졌습니다. 이 연구에서는 'style prompts'라고 불리는 텍스트 설명에 의해 안내되는 음성 합성을 다룹니다. 또한, 모호하거나 추상적인 스타일 프롬프트를 이해하는 데 있어 모델의 한계를 연구합니다.

- **Technical Details**: 연구는 특정 직업에 대한 지시, 예를 들어 "간호사처럼 행동하라"는 요청을 통해 여성 편향(gender bias)의 가능성을 조사합니다. 이를 통해 모델이 이러한 프롬프트를 해석할 때 성별 고정관념을 증폭시키는 경향이 있는지를 탐구합니다. 다양한 크기의 모델이 직업에 따라 차별화된 편향 정도를 보이는 것도 주목할 만한 요소입니다.

- **Performance Highlights**: 실험 결과에 따르면, 모델은 특정 직업에 대해 성별 편향을 드러내는 경향이 있음을 알 수 있었습니다. 특히, 직업에 따라 모델의 크기가 다를 경우, 이 편향의 정도도 달라지는 것으로 나타났습니다.



### ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports (https://arxiv.org/abs/2502.05638)
- **What's New**: 이 논문은 유럽의 의료 시스템에서 레거시(legacy) 임상 데이터 처리를 위해 혁신적인 솔루션을 제공하고자 하는 프로젝트의 결과를 제시합니다. 연구팀은 구조화되지 않은 임상 보고서에서 환자 이력, 진단, 치료 등의 정보를 추출하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 활용하였습니다. 또한 소규모 모델을 세밀하게 조정(fine-tuning)하여 더 큰 모델보다 우수한 성과를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 다양한 크기의 오픈 소스 LLM을 평가하며, 처음에는 단순한 프롬프트(prompts)로 테스트하고 후에 소규모 모델인 Llama 시리즈를 세밀하게 조정하여 정확성을 향상시켰습니다. 새로운 데이터셋은 60,000개의 영어 임상 요약 및 24,000개의 독일어 번역으로 구성되어 있으며, 각 사례는 PubMed Central에서 추출한 요약과 JSON 형식의 구조화된 정보를 포함합니다. 이렇게 확보된 데이터셋은 다양한 의료 데이터를 다루며, 연구 재현(reproducibility)과 재사용성을 위해 공개되었습니다.

- **Performance Highlights**: 소규모 언어 모델의 세밀 조정(fine-tuning)이 기존의 대규모 모델보다 우수한 결과를 도출하여 자원제한 환경에서도 효율적인 솔루션을 제공함을 입증하였습니다. 평가에는 ROUGE, BERTScore 및 개체 수준 지표가 사용되었으며, 실험 결과는 텍스트 기반의 임상 보고서에서 구조화된 정보를 효과적으로 추출할 수 있는 가능성을 보여줍니다. 연구 결과는 의료 데이터의 품질과 상호 운용성을 높이는 데 기여할 것으로 기대됩니다.



### AnyEdit: Edit Any Knowledge Encoded in Language Models (https://arxiv.org/abs/2502.05628)
- **What's New**: 이 논문에서는 AnyEdit라는 새로운 자율 회귀 편집 패러다임을 제안합니다. 기존의 모델 편집 기법이 긴 형식의 지식이 포함된 다양한 포맷에 대한 업데이트에 어려움을 겪고 있음을 지적하며, 이를 해결하기 위해 복수의 토큰을 연계하여 편집할 수 있는 혁신적인 접근 방식을 도입합니다. AnyEdit는 긴 지식을 순차적인 청크로 분해하고, 각 청크의 핵심 토큰을 반복적으로 편집하여 일관되면서도 정확한 출력을 보장합니다.

- **Technical Details**: AnyEdit는 정보 이론에서의 상호 정보의 연쇄 법칙을 기반으로 하고 있어, 모델 내부의 지식을 효과적으로 업데이트할 수 있는 능력을 이론적으로 입증합니다. 이 새로운 접근법은 지식의 길이에 따라 편집할 토큰의 수를 조정할 수 있으며, 구조에 구애받지 않고 다양한 형식의 지식을 지원하는 일반성을 가지고 있습니다. AnyEdit는 단일 토큰 편집의 효능 장벽을 극복하여, 여러 가지 복잡한 지식 구조를 보다 효과적으로 처리합니다.

- **Performance Highlights**: 실험 결과, AnyEdit는 UnKEBench, AKEW, 그리고 새로운 EditEverything 데이터셋을 포함한 여러 벤치마크에서 기존의 강력한 기법들보다 21.5%의 성능 향상을 보여주었습니다. EditEverything 데이터셋은 458개의 토큰을 포함하는 긴 형식의 지식을 다루며, 전통적인 접근 방식에 원활한 통합을 제공하는 플러그 앤 플레이 특성을 가지고 있습니다. 이러한 특징들은 LLM 지식 편집의 범위와 실용성을 크게 향상시키고 있습니다.



### Towards Sustainable NLP: Insights from Benchmarking Inference Energy in Large Language Models (https://arxiv.org/abs/2502.05610)
Comments:
          Accepted to appear at the NAACL 2025 Main Conference

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론 에너지에 대한 종합적인 벤치마킹을 수행합니다. 기존 연구가 주로 훈련 비용에 초점을 맞춰온 반면, 본 연구는 다양한 NLP 작업에서의 추론 에너지를 정량적으로 분석합니다. 이를 통해 출력 토큰 길이, 응답 시간, 모델 크기, 복잡도와 같은 여러 요인의 에너지 사용에 대한 강력한 상관관계를 발견했습니다.

- **Technical Details**: 연구는 총 6666개의 GPT 스타일 모델과 4444개의 Flan-T5 모델을 검토하였으며, 다양한 NLP 작업에 대한 샘플 데이터셋을 사용했습니다. 평가에 사용된 모델은 인코더-디코더 및 디코더 전용 모델로 나누어져 있으며, 이를 통해 각 모델의 응답 속도와 추론 에너지를 비교 분석했습니다. 또한, Normalized Accuracy(NA) 메트릭을 도입하여 다양한 작업 간 성능을 비교할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 추론 동안 에너지 사용을 줄이기 위한 몇 가지 효과적인 접근 방식이 제시되었습니다. 예를 들어, 최적의 배치 크기와 검정화(quantization)를 활용하면 에너지 소비를 크게 줄일 수 있습니다. 이 연구는 기후 지속 가능성 관점에서 LLM의 에너지 효율성을 향상시키기 위한 구체적인 가이드라인도 제공합니다.



### Lossless Acceleration of Large Language Models with Hierarchical Drafting based on Temporal Locality in Speculative Decoding (https://arxiv.org/abs/2502.05609)
Comments:
          Findings of NAACL 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론 속도를 개선하기 위한 새로운 접근법인 Hierarchy Drafting (HD)을 제안합니다. HD는 다양한 토큰 소스를 계층적으로 조직하여, 기존의 방법보다 더 일관된 성능을 제공합니다. 이러한 접근법은 실시간 상호 작용에 필요한 빠른 응답 속도를 실현하는 데 중점을 두고 있습니다.

- **Technical Details**: HD 방식은 시간적 근접성에 기반하여 여러 데이터베이스를 계층적으로 구성합니다. 초안 단계에서 HD는 여러 데이터베이스를 순차적으로 접근하여 근접도가 가장 높은 토큰부터 낮은 토큰까지 초안을 생성하며, 이를 통해 다양한 작업에서 일관된 속도 향상을 보장합니다. 이는 기존의 데이터베이스 초안 방법에 비해 상당한 장점을 제공합니다.

- **Performance Highlights**: 7억(7B) 및 13억(13B) 매개변수를 가진 LLM을 사용한 Spec-Bench 실험에서 HD는 기존의 초안 기법보다 우수한 성능을 보였습니다. HD는 모델 크기와 작업, 온도에 관계없이 강력한 추론 속도 향상을 달성하여, 다양한 환경에서도 효과적으로 작동하는 것을 입증합니다.



### ARIES: Stimulating Self-Refinement of Large Language Models by Iterative Preference Optimization (https://arxiv.org/abs/2502.05605)
- **What's New**: 이 논문은 외부 상호작용을 통해 응답의 오류를 수정할 수 있는 지능적인 대형 언어 모델(LLM)의 자가 개선 능력(self-refinement capability)을 개발하는 방법을 탐구합니다. 새로운 포스트 트레이닝 및 추론 프레임워크인 ARIES(Adaptive Refinement and Iterative Enhancement Structure)를 소개하며, 이를 통해 모델 성능을 향상시키는 방법을 제안합니다. ARIES는 선호도 기반 훈련을 반복적으로 수행하며, 자가 개선을 통해 점진적으로 정제된 응답을 생성한다는 특징을 가지고 있습니다.

- **Technical Details**: ARIES는 훈련 중에 모델의 직접 질문-응답 능력을 강화하고 자가 개선 잠재력을 열어주는 기술을 사용합니다. 이 과정에서 Reward Model Scoring이나 Rule-Based Selection 기법을 이용하여 필터링된 응답을 생성하여 다음 선호도 훈련의 데이터셋을 구축합니다. 이를 통해 Llama-3.1-8B 모델을 적용하여 ARIES는 다양한 기준을 초과하는 성능을 달성할 수 있습니다.

- **Performance Highlights**: ARIES는 AlpacaEval 2에서 62.3%의 길이 제어된(win rate) 및 63.3%의 원시(win rate) 승률을 기록하며, Iterative DPO를 각각 27.8% 및 35.5% 초과한 결과를 보였습니다. 또한 Arena-Hard에서는 50.3%의 승률을 달성하여, 자가 개선 기능을 활용한 점진적 향상이 모델 성능에 미치는 영향을 강조합니다. 이 연구는 수학적 추론 과제에서도 지속적으로 성능 향상을 이뤄내고 있습니다.



### On Memory Construction and Retrieval for Personalized Conversational Agents (https://arxiv.org/abs/2502.05589)
Comments:
          10 pages, 5 figures, conference

- **What's New**: 본 논문에서는 긴 대화에서 일관되고 개인화된 경험을 제공하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 대화 이력을 기반으로 메모리 뱅크를 구축할 때 차원(Granularity)에 따라 한계를 보인다는 점을 발견했습니다. 특히, LLMLingua-2와 같은 프롬프트 압축 방법이 메모리 검색의 정확도를 높이는 데 효과적이라는 점을 강조합니다.

- **Technical Details**: SeCom이라는 새로운 시스템을 도입하여 대화 세그먼트 수준에서 메모리 뱅크를 구축하고, 압축 기반의 디노이징을 통해 메모리 검색을 향상시킵니다. 대화 세그멘테이션 모델을 사용하여 긴 대화를 주제에 맞게 분할하고, 메모리 유닛을 검색할 때 요약을 거치지 않고 직접 결합하여 정보 손실을 방지합니다. 이 과정에서 기본적인 언어의 중복성이 검색 시스템에 잡음으로 작용할 수 있다는 가정을 통해 메모리를 최적화합니다.

- **Performance Highlights**: SeCom은 LOCOMO와 Long-MT-Bench+와 같은 장기 대화 기준 벤치마크에서 기존의 턴 수준 및 세션 수준 방법들을 초월하는 성능을 보였습니다. 실험 결과는 세그먼트 수준 메모리 유닛과 압축 기반 디노이징 기법의 기여를 강화하며, 결과적으로 응답 생성의 정확성과 관련성을 높이는 데 성공했습니다.



### Large Multimodal Models for Low-Resource Languages: A Survey (https://arxiv.org/abs/2502.05568)
- **What's New**: 이번 연구는 저자들이 저자들이 75개의 저자들이 106개 연구를 분석하여 저자들 저자들이 저자들이 저자들 저자들을 도와줄 수 있는 기술과 접근 방식을 제시하고 있다는 점에서 새롭습니다. 여러 저자들은 다양한 접근 방식을 통해 저자들이 저자들이 저자들이 저자들이 저자들이 저자들을 다루는 데 있어 핵심 패턴을 확인했습니다. 시각적 정보는 LMMs의 성능을 향상시키는 데 중요한 역할을 하며 여전히 해결해야 할 도전 과제가 많은 상황입니다.

- **Technical Details**: 본 논문은 LMMs의 다양한 기법을 분류하여 데이터 생성, 융합 기술, 시각적 개선, 크로스-모달 전이, 합성 데이터 생성 등을 포함한 여섯 가지 주요 범주로 나누고 있습니다. 이는 저자들이 저자들 저자들을 통해 저자들이 저자들을 보다 효과적으로 처리할 수 있는 방법을 제시하고 있음을 보여줍니다. 각 섹션은 저자들이 저자들을 통해 저자들의 적용 가능성을 느낄 수 있는 중요한 기법을 다루고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 텍스트-이미지 조합이 연구에서 가장 주목받는 경향을 보였고, 특정 언어에 대한 연구 초점의 불균형이 있습니다. 특히 힌디어, 아랍어, 벵골어 같은 주요 언어들은 많은 연구의 주제가 되었으나 40개 이상의 언어들은 단 하나의 연구로만 대표되고 있습니다. 이러한 결과는 LMMs를 활용하여 보다 다양한 저자들이 저자들의 접근을 향상시킬 필요성을 강조하고 있습니다.



### ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data (https://arxiv.org/abs/2502.05567)
- **What's New**: ATLAS는 자연어 수학 정리를 기계 검증 가능한 형식 언어로 자동 변환하는 새로운 데이터 생성 프레임워크입니다. 이 프레임워크는 총 300,000개의 정리 문장으로 구성된 대규모 고품질 병렬 정리 데이터셋을 생성합니다. ATLAS 번역기는 ProofNet 데이터셋에서 80.59%(pass@8)와 92.99%(pass@128)의 정확도를 달성했습니다.

- **Technical Details**: ATLAS는 데이터 리프팅(Data Lifting), 데이터 합성(Data Synthesis), 데이터 증강(Data Augmentation)이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 데이터 리프팅에서는 Mathlib에서 수학 개념을 추출하여 개념 저장소를 구성하였고, 데이터 합성에서는 다양한 Teacher 모델과 Student 모델을 활용하여 병렬 정리 문장을 생성합니다. 이러한 프로세스는 정확한 형식 언어 표기를 생산하고, Lean 컴파일러를 통해 구문적 및 의미적 오류를 검증합니다.

- **Performance Highlights**: ATLAS 데이터셋을 기반으로 한 ATLAS 번역기는 miniF2F, ProofNet, MathQual 데이터셋에서 각각 91.60%, 80.59%, 65.47%의 정확도를 기록했습니다. 특히, pass@128 지표에서 이 번역기는 96.93%, 92.99%, 84.72%의 정확도로 최신 성능을 기록하며 기존 모델들인 base model 및 InternLM2-Math-Plus-7B를 초월했습니다.



### Latent Structure Modulation in Large Language Models Through Stochastic Concept Embedding Transitions (https://arxiv.org/abs/2502.05553)
- **What's New**: 이 논문에서는 Stochastic Concept Embedding Transitions (SCET)이라는 새로운 메커니즘을 제안하여 고정된 정적 임베딩의 한계를 극복하고, 자연어 처리 모델이 다양한 문맥에 유연하게 적응할 수 있도록 하였습니다. SCET는 각 토큰 임베딩이 확률적 업데이트를 통해 진화하도록 하여, 언어의 고유한 변동성을 더욱 효과적으로 포착하는 것을 목표로 하고 있습니다. 이 접근법은 모델의 표현력을 향상시키고, 개인적 용도에 맞는 성능 향상을 꾀할 수 있도록 합니다.

- **Technical Details**: SCET 프레임워크에서는 각 임베딩을 고정된 벡터가 아니라 여러 잠재적 상태에 대한 확률 분포로 정의합니다. 이러한 dinâmica적 변화는 언어의 문맥 의존성을 반영하여, 각 토큰의 표현이 문맥에 따라 유동적으로 조정될 수 있게 합니다. 전이 확률은 지역 및 글로벌 언어적 문맥에 따라 달라지며, 이를 통해 임베딩이 의미적으로 일관성을 유지하면서도 변동성을 가지게 됩니다.

- **Performance Highlights**: 실험 결과는 SCET를 채택한 모델이 더 높은 텍스트 완성 정확도 및 대화 일관성을 나타냈으며, 정적 임베딩에 비해 구조적 복잡성이 향상되었음을 보여줍니다. 확률적 업데이트 방식은 의미 있는 집단 간 연결성을 유지하면서 문맥 중심의 변화를 가능하게 하였으며, 이러한 전이 메커니즘의 안정성을 추가로 검증하였습니다. 퍼포먼스 메트릭스는 SCET가 적응성과 제어의 균형을 맞추어, 생성된 출력이 지나치게 무작위성을 띠지 않으면서도 일관된 언어적 표현을 유지함을 나타냅니다.



### FRAMES: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy (https://arxiv.org/abs/2502.05551)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 성능을 향상시키기 위해 Four-quadRAnt Multi-stage prEtraining Strategy (FRAMES)를 제안합니다. FRAMES는 Pretraining 과정에서 데이터의 손실을 네 번 줄이기 위한 네 가지 단계로 구성된 새롭고 체계적인 접근 방식을 제공합니다. 이는 Pretraining 과정에서 데이터의 Perplexity (PPL) 및 PPL 차이 (PD)를 기준으로 데이터를 분할함으로써 이루어집니다.

- **Technical Details**: FRAMES는 데이터 분할에 있어 두 가지 주요 발견에 기반하고 있습니다. 첫 번째는 높은 PPL 데이터로 훈련한 후 낮은 PPL 데이터로 변경할 때 손실이 두 번 제대로 감소하는 것입니다. 두 번째로, 낮은 PD 데이터에서 높은 PD 데이터로 전환하면 비슷한 손실 감소 효과를 얻을 수 있습니다. 이를 통해 네 개의 사분면 (Quadrants)으로 데이터를 조직하여 효과적인 Pretraining을 수행하게 됩니다.

- **Performance Highlights**: FRAMES를 적용한 결과, MMLU 및 CMMLU에서 무작위 샘플링에 비해 평균적으로 16.8% 향상된 성능을 보여주었습니다. 연구 결과, 이 전략은 모델의 잠재적 능력을 증가시키고 전체적인 성능 개선을 효과적으로 이끌어냈습니다. 따라서 FRAMES는 LLMs의 훈련 효율성을 높이는 데 중요한 기여를 할 것으로 평가됩니다.



### DeepThink: Aligning Language Models with Domain-Specific User Intents (https://arxiv.org/abs/2502.05497)
- **What's New**: 이번 연구는 LLM(대규모 언어 모델)이 특정 도메인 질의응답(QA) 작업에 적합하도록 고급 합성 지침(high-quality instructions)을 생성하는 새로운 프레임워크인 DeepThink를 제안합니다. DeepThink는 실제 사용자 질문을 모방한 몇 가지 초기 질문(seed questions)을 생성한 후 대화 시뮬레이션을 통해 사용자 요구를 발견하고, 대화 맥락과 검색된 문서에 기반하여 답변을 정제합니다. 실험 결과, DeepThink는 광고 도메인에서 실제 사용자 테스트 세트에 대해 평균 7.92%의 성능 향상을 달성했습니다.

- **Technical Details**: DeepThink는 4단계의 핵심 프로세스로 구성되어 있습니다: 1) 초기 질문 및 답변 합성, 2) 대화 기반 데이터 합성, 3) 대화 기반 데이터 정제, 4) 검색 증강(SFT)입니다. 사용자 질문의 실제 언어 스타일과 구조를 반영하는 질문을 생성하기 위해, DeepThink는 실제 사용자 질문을 샘플링하여 GPT-4-turbo에게 질문 생성을 요청합니다. 또한, DeepThink는 검색 기반 생성 프레임워크를 도입하여 응답의 정확성을 높이고, 사용자와 대화를 지속적으로 심화시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: DeepThink는 광고 플랫폼에서의 평가에서 GPT-4-turbo+RAG를 초월하는 성능을 보였으며, 관련성(relevance), 완전성(completeness), 명확성(clarity), 정확성(accuracy), 실행 가능성(actionability) 측면에서 각각 3.43%, 12.09%, 6.69%, 3.74%, 13.69%의 개선을 이뤘습니다. 이는 DeepThink가 대화의 주제 일관성을 유지하면서 사용자 요구에 보다 잘 대응하는 능력을 갖춤을 보여줍니다. 이 연구는 특화된 도메인에서의 대화 기반 질의응답 시스템의 발전에 중요한 기여를 할 것으로 기대됩니다.



### Mechanistic Interpretability of Emotion Inference in Large Language Models (https://arxiv.org/abs/2502.05489)
Comments:
          To be submitted to the Association for Computational Linguistics (ACL 2025)

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 텍스트에서 인간의 감정을 예측하는 메커니즘을 탐구하며, 감정 표현이 모델의 특정 지역에 기능적으로 국소화되어 있음을 보여준다. 연구는 다양한 모델 패밀리와 크기를 포함하며, 감정 처리의 기능적 국소화에 대한 증거를 제공한다. 또한, 인지 평가 이론을 활용해 LLM 내부의 감정 처리 구조를 설명하고, 이를 통해 LLM이 어떻게 감정을 생성하는지를 이해한다.

- **Technical Details**: 본 연구에서는 LLM의 숨겨진 표현 위에 선형 분류기를 훈련하여 감정 관련 활성화가 가장 강하게 나타나는 영역을 탐색한다. 중간 층의 Multi-Head Self-Attention(MHSA) 유닛이 LLM의 의사결정을 형성하는 주요 역할을 하며, 이러한 유닛이 감정적으로 중요한 토큰에 지속적으로 주의를 기울임을 시각적으로 확인한다. 또한, 평가 개념을 조작하여 감정 출력을 조정하는 방법론을 통해, 이론적 기대와의 일치를 보여준다.

- **Performance Highlights**: 실험 결과, LLM은 주어진 텍스트 컨텍스트에 기반해 감정을 효과적으로 유추할 수 있는 능력을 보였다. 총 6,800개의 감정 비네트를 사용하는 crowd-enVENT 데이터셋을 통해 모델의 감정 분류 성능을 평가하였으며, 전문적인 분석을 위해 정확히 분류된 샘플을 중심으로 검토하였다. 결과적으로, 본 연구는 LLM의 감정 생성 메커니즘을 심층적으로 이해하는 데 기여하며, 감정 관련 작업에서 안전성과 정렬을 개선할 수 있는 새로운 방법론을 제시한다.



### OntoTune: Ontology-Driven Self-training for Aligning Large Language Models (https://arxiv.org/abs/2502.05478)
Comments:
          Accepted by WWW25

- **What's New**: 이번 논문에서는 기존의 대규모 도메인 전문 LLM을 효과적으로 조직하지 못하는 문제를 해결하기 위해, ontologies(온톨로지)를 통한 새로운 방식을 제안합니다. 기존 LLM의 지식을 계층적으로 재구성하기 위해 OntoTune이라는 프레임워크를 발표했으며, 이는 LLM의 도메인 지식을 온톨로지와 정렬하여 응답을 생성할 수 있도록 지원합니다. 의학 도메인에서 표준화된 의료 온톨로지인 SNOMED CT를 활용하여 OntoTune의 효과를 검증하였습니다.

- **Technical Details**: OntoTune은 세 가지 주요 단계로 구성된 자가 학습 미세 조정 프레임워크입니다: 1) Instruction Text Generation(지시문 텍스트 생성), 2) Inconsistency Text Selection(불일치 텍스트 선택), 3) LLM Fine-tuning(LLM 미세 조정)입니다. 각 단계에서 온톨로지 지식을 활용하여 LLM의 출력 결과를 개선하며, 이 과정에서 도메인 지식의 적합성을 높이는 데 주안점을 두고 있습니다. 이 과정을 통해 LLM은 온톨로지의 계층 구조에 기반한 응답을 효율적으로 생성하게 됩니다.

- **Performance Highlights**: OntoTune은 in-ontology(온톨로지 내) 태스크인 하이퍼님 발견 및 out-of-ontology(온톨로지 외) 태스크인 도메인 QA에서 최신의 성능을 달성했습니다. 이는 기존의 대규모 도메인 전문 LLM이나 TaxoLLaMA와 비교했을 때, LLM의 원래 지식을 보다 잘 보존하는 성과를 보여줍니다. 연구 결과, OntoTune은 데이터를 효율적으로 관리하며 도메인 특정 작업을 효과적으로 수행할 수 있는 가능성을 입증하였습니다.



### Position: LLMs Can be Good Tutors in Foreign Language Education (https://arxiv.org/abs/2502.05467)
Comments:
          18 pages, 4 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)을 외국어 교육(FLE)에 통합하려는 노력들이 시작되고 있지만, 전통적인 학습 과제를 사용하고 있어 교육 방법론에 대한 적응 능력이 부족하다. 본 논문에서는 LLM이 효과적인 튜터로 활용될 수 있는 세 가지 주요 역할, 즉 데이터 향상(data enhancers), 작업 예측(task predictors), 그리고 에이전트(agents)로서의 기능을 제안하며, 이로써 FLE를 향상시킬 수 있는 방안을 모색한다.

- **Technical Details**: LLM은 컴퓨터 과학, 언어학, 교육학, 심리언어학 등 다양한 분야와의 융합을 통해 FLE의 도전 과제를 해결할 수 있는 잠재력을 지닌다. LLM은 자연어 이해 및 생성에서 놀라운 성능을 보여주며, 전통적인 교육 방법의 한계를 극복하고 개인화된 교육 경험을 제공하는 데 기여할 수 있다. 특히 청취, 말하기, 읽기, 쓰기라는 네 가지 핵심 기술을 LLM을 통해 효과적으로 개선할 수 있는 가능성이 있다.

- **Performance Highlights**: LLM은 학습 자료의 생성 및 피드백 제공, 상호작용을 통한 다양한 언어 학습 활동에서 큰 장점을 보여준다. 그러나 LLM의 통합은 기존의 인간 튜터를 보완해야 하며, LLM의 허위 정보(hallucination)를 방지하기 위해 고품질 데이터를 확보하는 것이 필수적이다. 향후 LLM을 FLE에 적용하기 위한 도전과 윤리적 고려 사항을 논의함으로써, 연구자와 교육자가 이 기술의 혁신적 잠재력을 활용할 수 있는 기술적 가이드를 제공하고자 한다.



### Iterative Deepening Sampling for Large Language Models (https://arxiv.org/abs/2502.05449)
- **What's New**: 최근 OpenAI의 o1 모델의 출시로 인해 복잡한 추론 작업을 처리하는 데 탁월한 능력이 입증되었습니다. 이 논문에서는 모델이 단일 응답 내(intra-response)와 여러 응답 간(inter-response)에서 검색 기능을 발전시키는 것이 중요하다는 점에 주목합니다. 특히, 자기 평가와 자기 교정 기능을 향상시키기 위한 자기 반성 데이터 생성의 질을 개선하는 데 초점을 맞춥니다. 이를 위해 독창적인 Iterative Deepening Sampling(IDSampling) 알고리즘 프레임워크를 제안합니다.

- **Technical Details**: Iteration Deepening Sampling (ID-Sampling) 방법론은 샘플링 예산을 기하급수적으로 증가시키는 반복적 접근 방식을 사용하여 자기 반성 메커니즘을 각 확장 단계에 통합합니다. 이는 모델의 성능을 향상시키면서도 예산 낭비를 최소화할 수 있는 효율적인 방법입니다. 우리는 MATH-500 및 AIME-24와 같은 난이도 높은 벤치마크에서 ID-Sampling의 효과를 평가하였습니다. 또한, 각 반복에서 예산 증가 비율이 성공률과 추론 시간에 미치는 영향을 분석하는 절단 연구(ablation study)도 진행하였습니다.

- **Performance Highlights**: ID-Sampling을 사용한 실험 결과, 고난이도 작업에서 더 높은 성공률을 달성했습니다. 이는 모델의 성능을 향상시키기 위한 적응형 자기 반성 메커니즘의 가능성을 보여줍니다. 최종적으로 이 연구는 고품질의 자기 반성 데이터를 생성하여 차세대 LLM의 훈련을 향상시키는 데 기여할 수 있는 방법을 제시하고 있습니다.



### SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation (https://arxiv.org/abs/2502.05424)
Comments:
          Accepted by WWW2025 Main Track

- **What's New**: 본 논문에서는 텍스트가 없는 그래프를 위한 멀티 도메인 그래프 사전 학습 및 교차 도메인 적응을 위한 구조 정렬 프레임워크(SAMGPT)를 제안합니다. SAMGPT는 다양한 소스 도메인에서 나온 그래프들의 지식을 학습하여 보지 못한 타겟 도메인에 적응할 수 있도록 설계되었습니다. 특히, 구조 정보를 기반으로 하는 집계를 조화롭게 하기 위해 구조 토큰을 도입하고, 교차 도메인 적응을 위해 전체적 및 특정 프롬프트를 제공합니다.

- **Technical Details**: SAMGPT는 다중 도메인 그래프의 사전 학습 동안 구조적 변수를 조화롭게 만드는 것을 목표로 합니다. 각 도메인에는 구조 기반 집계를 수정할 수 있는 학습 가능한 구조 토큰이 제공됩니다. 교차 도메인 적응에서는 전체적 지식과 도메인별 특성을 동시에 활용하기 위해 두 가지 유형의 프롬프트, 즉 전체적 프롬프트와 특정 프롬프트를 활용합니다.

- **Performance Highlights**: 저자들은 7개의 공공 데이터셋에서 SAMGPT의 성능을 종합적으로 평가했으며, 기존의 최첨단 방법들과 비교하여 우수한 성능을 доказ했습니다. 이 모델은 메타 학습의 원리를 바탕으로 다양한 도메인에서 수집된 그래프를 통합하여 뛰어난 효율성을 입증했습니다.



### Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data (https://arxiv.org/abs/2502.05400)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 자기 생성 합성 데이터를 활용하여 인간 주석 데이터에 대한 의존성을 줄이는 새로운 방법인 동적 노이즈 선호 최적화(DNPO)를 제안합니다. 기존 방법들이 반복 과정에서 성능 향상을 일관되게 보장하지 못하는 문제를 해결하기 위해, DNPO는 동적인 샘플 레이블링 메커니즘을 도입하고, 선호 최적화 과정에서 제어 가능한 훈련 가능한 노이즈를 추가합니다. 이를 통해 모델의 성능이 지속적으로 향상되도록 합니다.

- **Technical Details**: DNPO는 동적 샘플 레이블링(DSL)과 노이즈 선호 최적화(NPO)라는 두 가지 주요 구성 요소로 구성되어 있습니다. DSL은 데이터 품질에 기반하여 샘플 레이블을 동적으로 조정하며, NPO는 선호 데이터에 훈련 가능한 노이즈를 포함하여 탐색을 촉진하고 반복 간 정체를 줄입니다. 이 최적화 과정은 선호 쌍의 긍정 및 부정 샘플 간의 마진을 극대화하며 노이즈 매개변수를 동시에 업데이트하여 개선합니다.

- **Performance Highlights**: 실험 결과, DNPO는 기존 방법들에 비해 일관된 성능 향상을 보여주었으며, 여러 기준에서 평균 2.6%의 성능 향상을 달성했습니다. 또한, 모델이 생성한 데이터의 품질이 GPT-4 평가에서 기준 대비 29.4%의 승패 격차 개선을 보이며, 이는 LLM의 자기 개선 과정에서 DNPO의 효과성과 유용성을 강조합니다.



### Hierarchical Lexical Manifold Projection in Large Language Models: A Novel Mechanism for Multi-Scale Semantic Representation (https://arxiv.org/abs/2502.05395)
- **What's New**: 이 논문에서는 Hierarchical Lexical Manifold Projection (HLMP)이라는 새로운 프레임워크를 도입하여 다층적 의미 표현을 효과적으로 캡처하는 방법을 제안합니다. HLMP는 기존의 단어 임베딩 방식의 한계를 극복하고, 의미론적 관계를 보존하면서도 다양한 언어 작업에서의 모델 성능을 향상시키는 것을 목표로 합니다. 이 방식은 토큰의 투영을 구조화된 다각체에 매핑하여 의미적인 정렬을 개선하고, 다양한 언어 구성 요소 간의 일관성을 유지합니다.

- **Technical Details**: HLMP는 다중 규모의 의미 표현을 포착하기 위한 구조화된 매니폴드 기반의 투영 메커니즘을 사용합니다. 각 언어 단위는 미분 가능한 리만 매니폴드에 매핑되며, 이를 통해 의미적 근접성을 보존할 수 있는 구조적 표현을 제공합니다. 이러한 구조는 국소화된 구문적 특징과 글로벌 의미 구조 간의 안정적인 전환을 가능하게 합니다.

- **Performance Highlights**: HLMP는 기존의 임베딩 방식 대비 뛰어난 성능을 보이며, 여러 언어 작업에서의 정확성을 높이는 결과를 얻었습니다. 실험적으로 다층적 의미 표현을 포착하는 데 있어 HLMP의 효과성을 입증하였으며, 특히 적대적 텍스트 변형에 대한 강인성을 보여주었습니다. 따라서 HLMP는 자연어 처리(NLP) 분야의 발전에 기여할 것으로 기대됩니다.



### Learning Task Representations from In-Context Learning (https://arxiv.org/abs/2502.05390)
Comments:
          Appeared in ICML 2024 Workshop on In-Context Learning

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 in-context learning (ICL)에서 새로운 작업에 대한 적응력을 보여주었으나, 내부적으로 작업이 어떻게 인코딩되고 일반화되는지를 이해하는 데 어려움이 있다는 점을 해결하려고 한다. 저자들은 transformer 아키텍처의 attention heads를 기반으로 작업 정보를 인코딩하는 자동화된 방법을 제안하며, 이 작업 벡터는 attention heads의 가중 합으로 계산된다. 이는 기존 방법들이 텍스트 외의 다른 범주에 효과적으로 일반화되지 못하는 문제를 보완할 수 있다.

- **Technical Details**: 저자들은 transformer의 attention heads에서 작업 정보를 인코딩하는 새로운 자동화된 공식을 도입한다. 이 과정에서 gradient descent를 통해 가중치를 최적화하여 단일 작업 벡터를 계산한다. 연구 결과는 기존 방법들이 텍스트를 넘어서는 다양한 모달리티에 일반화되지 못하는 경향이 있음을 보여준다. 특히, 저자들은 regression 작업에서 작업 벡터의 정확성을 평가하는 새로운 벤치마크를 디자인하여 기능 회귀 작업에서의 성능을 강조하고 있다.

- **Performance Highlights**: 제안된 방법은 in-context demonstrations에서 작업-specific 정보를 성공적으로 추출하며, 텍스트 및 회귀 작업 모두에서 뛰어난 성능을 보여준다. 또, ablation 연구는 저자 방법의 효과가 최적의 in-context 학습 모델과의 마지막 hidden state 분포를 맞추는 데서 기인함을 시사한다. 이로 인해 작업 인도하는 방법을 보다 심층적으로 이해하는 데 기여할 수 있다.



### The Role of Prosody in Spoken Question Answering (https://arxiv.org/abs/2502.05389)
Comments:
          accepted to NAACL 2025 Findings

- **What's New**: 이 연구는 Spoken Question Answering (SQA)에서의 음운(prosody)의 역할을 조사합니다. 기존의 연구에서 대부분의 데이터셋은 텍스트 기반으로, 음성을 합성한 후에 사용되었습니다. 이로 인해 음성 신호에 포함된 음운 정보를 제대로 활용하지 못하고 있었습니다. 연구진은 자연스러운 음성을 포함한 SLUE-SQA-5 데이터셋을 사용하여, 음운 정보만으로도 상당히 좋은 성능을 발휘하는 모델을 만들 수 있음을 발견했습니다.

- **Technical Details**: 음운은 발음, 강세, 리듬 외에도 다양한 음성 요소들로 구성되어 있으며, 이는 사람의 청취 이해에 중요한 역할을 합니다. 이 연구에서는 연구 질문으로 음운 정보만으로도 SQA 작업을 수행할 수 있는지를 검토하고, 또한 어휘 정보가 존재할 때 SQA 모델이 음운 정보를 어떻게 활용하는지를 조사합니다. 실험을 통해 음운 조건과 어휘 조건을 설정하여, 음운 정보가 질문에 대한 답변을 안내할 수 있는지 평가했습니다.

- **Performance Highlights**: 결과적으로, 음운 정보만으로도 SQA 작업에서 유의미한 성과를 낼 수 있음을 보여주었습니다. 그러나 어휘 정보가 있을 때 모델은 주로 어휘 정보에 의존하는 경향이 있었습니다. 이 발견은 앞으로 음운과 어휘 정보를 효과적으로 결합한 더 강력한 모델 개발을 위한 기초 자료로 작용할 것입니다.



### Probabilistic Subspace Manifolds for Contextual Inference in Large Language Models (https://arxiv.org/abs/2502.05346)
- **What's New**: 본 연구에서는 Probabilistic Subspace Manifolds (PSMs)를 도입하여 자연어 처리에서의 맥락 추론 능력을 향상시키는 새로운 임베딩 프레임워크를 제안합니다. PSMs는 동적 매니폴드 구조를 통하여 토큰 간의 관계를 확률적 분포로 나타내며, 이를 통해 전통적인 고차원 벡터 기반 표현의 한계를 극복합니다. 이 방법론은 기존의 트랜스포머 아키텍처와의 통합이 용이하여, 기존 메커니즘에 큰 변화를 주지 않고도 정보 전달의 효율을 높입니다.

- **Technical Details**: PSMs의 핵심은 각 토큰이 learned submanifold 내에서 확률적 분포로 표현되는 점입니다. 그 결과, 주변 텍스트 입력에 따라 진화하는 정교한 맥락 의존성을 통해 의미적 모호성을 더욱 효과적으로 구분할 수 있습니다. 이전의 유클리드 임베딩 방식이 갖고 있던 한계를 극복하고, 복잡한 언어 현상을 보다 유연하게 모델링할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, PSMs를 이용한 임베딩이 맥락 일관성과 표현 품질에서 우수한 성능을 보이며, 적대적 수정에 대한 강인성 또한 향상되었습니다. 특히, 도메인 변경 시 대규모 재훈련 없이도 더 나은 적응성을 보였으며, 연산 효율성 또한 개선되었습니다. 이 연구는 언어 모델링의 발전에 기여하는 전반적인 의미에서도 중요한 기여를 하고 있습니다.



### Fine-Tuned LLMs are "Time Capsules" for Tracking Societal Bias Through Books (https://arxiv.org/abs/2502.05331)
Comments:
          9 pages (excluding references), accepted to NAACL 2025

- **What's New**: 이번 연구에서는 문화적 통찰력이 풍부한 책들이 시간의 흐름에 따라 어떻게 사회적 편견을 반영하는지를 탐구합니다. 이를 위해 1950년부터 2019년까지의 593권의 소설을 포함한 BookPAGE라는 새로운 코퍼스를 개발하여, 시대별 편견을 추적하고 분석하는 독창적인 방법을 제안합니다. 연구 결과, 특정 시대에훈련된 LLM(대형 언어 모델)이 그 시대의 성편견, 성적 지향, 인종, 종교와 관련된 편견을 반영하는 방식의 변화를 관찰하였습니다.

- **Technical Details**: BookPAGE는 각 시대( decade)별로 고유의 소설 편집물로 구성된 한국어 코퍼스로, 각 기간에 걸쳐 가장 인기 있던 소설들로 구성됩니다. 연구진은 LLM을 시대별로 라벨링한 책들로 파인튜닝하여, 이들 모델이 특정 시대의 언어적 패턴과 편견을 포착하도록 하였습니다. 이를 통해 연구팀은 성별, 인종, 종교 등 다양한 인구 통계학적 범주를 포함하여 세밀한 편견 패턴을 분석하였습니다.

- **Performance Highlights**: 연구를 통해 확인된 주요 내용들은 바로 소설의 내용이 LLM의 편견의 주요 기원이라는 점입니다. 이러한 발견은 인공지능 모델이 훈련될 때, 다양한 대표성을 가지는 데이터를 사용하는 것이 얼마나 중요한지를 강조합니다. 또한, 책에서 나타나는 편견의 패턴의 다각적인 분석을 통해 사회적 변화가 문학을 통해 어떻게 반영되는지를 보여줍니다.



### Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic: An Approach Based on Error Tagging Model and Synthetic Data Generating Mod (https://arxiv.org/abs/2502.05312)
Comments:
          21 pages, 3 figures

- **What's New**:  최근 수년간 아랍어 문법 오류 수정(ArabGEC)의 수요가 증가함에 따라, 본 논문은 아랍어 문법 오류를 위한 대규모 합성 데이터셋을 생성하기 위해 오류 태깅 모델과 합성 데이터 생성 모델을 개발하고자 하였습니다. 특히, DeBERTav3 모델을 활용한 오류 태깅 모델은 올바른 문장을 26가지 오류 태그로 분류하여, 다양한 인위적 문법 오류를 생성하는 데 기여합니다. 또한, ARAT5 모델을 기반으로 한 합성 데이터 생성 모델을 통해 실제로 발생할 수 있는 오류 패턴을 반영한 문장을 생성하고 있습니다.

- **Technical Details**:  본 연구는 순서-순서(seq2seq) 방식과 연계하여 아랍어 문법 오류를 수정하기 위한 오류 태깅 및 합성 데이터 생성 기술을 도입합니다. 오류 태깅 모델은 다중 레이블 분류 작업으로, 각 문장은 유형별로 26개의 오류 태그로 분류됩니다. 이를 통해 생성된 오류 태그를 정확한 문장에 결합하여, AraT5 모델을 활용한 합성 데이터 생성 모델이 문법적으로 일관된 잘못된 문장을 생성하게 됩니다.

- **Performance Highlights**:  오류 태깅 모델은 QALB-14 및 QALB-15 테스트 세트에서 각각 94.42%의 F1 스코어를 달성하여 오류 태그 식별에 있어 가장 우수한 성능을 기록했습니다. 또한, 문법 오류 수정에 대한 합성 데이터 학습 결과로서 QALB-14 테스트 세트에서 새로운 최첨단 결과인 79.36%의 F1 점수를 기록했습니다. 최종적으로 30,219,310개의 합성 문장 쌍이 생성되어, 아랍어 문법 오류 수정 시스템의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Can LLMs Rank the Harmfulness of Smaller LLMs? We are Not There Y (https://arxiv.org/abs/2502.05291)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 위험성과 한계를 이해하는 것이 중요하다는 점을 강조합니다. 특히, 작은 LLMs의 해로운 출력 생성을 평가하고, 큰 LLMs가 이러한 해로운 정도를 어떻게 주석(annotation)할 수 있는지를 연구합니다. 연구자들은 다양한 유형의 해로운 콘텐츠를 유도하기 위해 세 가지 작은 LLMs에 프롬프트를 제공하고, 인공지능 출력에 대한 인간의 평가를 수집하였습니다.

- **Technical Details**: 연구에서는 해로운 출력을 유도하는 과정이 세 단계로 진행됩니다: 첫째, 다양한 LLMs에서 해로운 출력을 유도하고, 둘째, 각 모델의 출력에 대한 상대적 해로움을 평가하며, 셋째, LLM들의 전반적인 해로움을 순위화합니다. 또한, GPT4와 같은 더 강력한 LLM들이 작은 모델의 해로운 정도를 평가할 수 있는지를 검토하였으며, 그 결과 인간과의 일치도가 낮은 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 작은 LLM들은 해로움 측면에서 상이한 평가를 받는 것으로 나타났습니다. 또한, 큰 LLM들이 인간의 평가와의 일치도가 낮거나 중간 정도인 것으로 발견되어, 주관적 태스크에서 LLM의 데이터 주석에 대한 추가 연구가 필요함을 시사합니다. 이러한 발견은 향후 LLM의 해악 완화에 관한 연구 방향을 제시합니다.



### LLMs Can Teach Themselves to Better Predict the Futur (https://arxiv.org/abs/2502.05253)
- **What's New**: 이 연구는 인공지능 모델의 예측 능력을 향상시키는 Outcome-driven fine-tuning framework을 제안합니다. 기존의 인간 큐레이션 방식에 의존하지 않고, 모델의 Self-play를 활용하여 다양한 질문에 대한 두 가지 Reasoning trajectories와 Probabilistic forecasts를 생성합니다. 이러한 접근 방식은 지식 컷오프 이후의 질문을 다루면서도, 모델의 성능을 효과적으로 개선할 수 있습니다.

- **Technical Details**: 이 방법론은 생성된 Reasoning traces 쌍을 기반으로, 실제 결과와의 거리를 측정하여 Ranking합니다. 이후, Direct Preference Optimization(DPO)을 통해 모델을 Fine-tune하여 예측 정확도를 높입니다. 이러한 과정은 모델의 self-play로 인해 생성된 데이터의 다양성을 통해 이루어집니다.

- **Performance Highlights**: 테스트 세트에서, Phi-4 14B와 DeepSeek-R1 14B 모델의 예측 정확도가 기존 베이스 모델과 DPO fine-tuned 제어 모델에 비해 7%에서 10% 향상되었습니다. 이는 GPT-4o와 같은 대형 모델의 예측 능력과 동등한 수준으로 상승한 성과입니다.



### GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? (https://arxiv.org/abs/2502.05252)
- **What's New**: 이번 논문에서는 기존의 long-context LLMs가 복잡한 추론 문제를 해결하는데 한계를 가지고 있음을 밝히고, 이를 해결하기 위한 새로운 벤치마크인 GSM-Infinite를 제안합니다. 이 벤치마크는 무한 난이도와 맥락 길이를 가진 산술 문제를 생성하며, 세밀한 조작이 가능합니다. 기존의 평가 지표들이 가진 한계를 극복하고, LLM들의 추론 성능을 정량적으로 평가할 수 있는 기반을 마련하였습니다.

- **Technical Details**: GSM-Infinite 벤치마크는 computational graphs와 언어 의미론을 결합하여 모델링된 추론 문제를 기반으로 합니다. 이 구조에 따라 문제의 난이도를 세밀하게 조정하고, 필요 없는 노드를 추가하여 노이즈를 효과적으로 삽입합니다. 이를 통해 LLM의 처리 능력을 평가할 수 있는 새로운 프레임워크를 제공하며, 다양한 호환성을 갖추고 있습니다.

- **Performance Highlights**: GSM-Infinite를 사용한 전반적인 성능 평가에서, 최신 reasoning-optimized LLM들이 이전 SOTA 모델들에 비해 평균 AUC 점수가 거의 4배 향상된 것으로 나타났습니다. 그러나 노이즈가 포함된 환경에서는 다양한 성능 저하가 관찰되는 등, LLM의 성능이 문제 난이도와 맥락 길이에 따라 일관된 감소를 보였습니다. 이 연구는 현재 long-context LLMs의 근본적인 한계를 강조하고 있으며, 향후 발전 방향에 대한 통찰을 제공합니다.



### Evaluating Personality Traits in Large Language Models: Insights from Psychological Questionnaires (https://arxiv.org/abs/2502.05248)
Comments:
          Accepted for publication at TheWebConf 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 성격 특성을 분석하는 새로운 접근법을 제시하고 있습니다. 심리적 평가 도구인 Big Five Inventory와 같은 질문지를 통해 LLM의 성격 프로파일을 생성하고, 다양한 모델 간의 차이를 조사하였습니다. 연구 결과 LLM들이 고유한 성격 특성을 보임을 발견했으며, 이는 인간의 행동을 이해하는 데 기여할 수 있습니다.

- **Technical Details**: 연구는 LLM의 성격 특성을 평가하기 위해 다섯 개의 주요 성격 차원인 Openness, Conscientiousness, Extraversion, Agreeableness, 그리고 Neuroticism을 포함한 심리적 질문지를 사용하였습니다. 각 질문지는 서로 다른 구조로 재구성되어 훈련 데이터 오염을 방지하고 편향성을 최소화했습니다. 이러한 절차는 LLM의 응답의 일관성을 확보하기 위한 체계적인 방법론을 포함합니다.

- **Performance Highlights**: 연구 결과는 LLM들이 Agreeableness, Openness, Conscientiousness와 같은 성격 특성에서 높은 점수를 보임을 나타냅니다. 이는 협력적이고 창의적이며 조직적인 행동을 반영하고 있습니다. 여러 LLM 모델을 통해 사용된 차원의 분석은 각 성격 질문지의 지배와 변동성을 보여주어, LLM의 성격 특성을 체계적으로 이해하는 데 도움을 줍니다.



### SEER: Self-Explainability Enhancement of Large Language Models' Representations (https://arxiv.org/abs/2502.05242)
Comments:
          18 pages,5 figures,10 tables

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 설명 가능성을 향상시키기 위한 새로운 방법인 SEER를 제안합니다. SEER는 동일한 개념을 집계하고 서로 다른 개념을 분리하여 LLM의 숨겨진 표현을 명확히 설명할 수 있도록 합니다. 이로 인해 LLM의 출력과 동시에 신뢰할 수 있는 설명을 제공합니다.

- **Technical Details**: SEER는 LLM의 표현 공간에서 개념을 집계하고 분리하는 과정에서, 'black-box' 모듈 없이 자체 설명 기능을 강화합니다. 이 접근 방식은 LLM의 추론 로직을 이해하고 응용 시나리오에서의 신뢰도를 높이는 데 기여합니다. 또한, 최적 수송 이론(optimal transport theory)을 통해 SEER의 LLM의 일반화(generalization) 능력 향상에 대한 이론적 분석을 진행합니다.

- **Performance Highlights**: 실험 결과, SEER는 안전 위험 분류(safety risks classification) 및 해독 작업(detoxification tasks)과 같은 신뢰성과 관련된 작업에서 일관된 개선을 보여 주었습니다. 이러한 자기 설명(self-explained) LLM들은 설명 가능성과 성능에서 최근의 지속적인 개선을 달성하였습니다.



### Enhancing Knowledge Graph Construction: Evaluating with Emphasis on Hallucination, Omission, and Graph Similarity Metrics (https://arxiv.org/abs/2502.05239)
- **What's New**: 이번 논문은 비구조적 텍스트에서 지식 그래프(kknowledge graph)를 자동으로 구축하는 대형 언어 모델의 최근 발전을 다루고 있습니다. 연구자들은 이전 연구를 바탕으로 환각(hallucination) 및 생략(omission) 문제를 해결하기 위한 개선된 접근 방식을 제안합니다. 특히, 그래프 유사성(graoh similarity) 평가를 위한 BERTScore를 통합하여 95%의 현실적인 그래프 매칭(threshold)을 설정했습니다.

- **Technical Details**: 실험에서는 Mistral 모델을 중심으로 원래 버전과 파인 튜닝(fine-tuning)된 버전을 제로샷(zero-shot) 및 퓨샷(few-shot) 설정에서 비교했습니다. 또한 KELM-sub 훈련 데이터셋의 예시를 이용하여 실험을 확장하였습니다. 결과적으로 파인 튜닝된 모델이 지식 그래프 구축 정확도를 크게 향상시키면서, 환각과 생략 현상을 줄이는 것으로 나타났습니다.

- **Performance Highlights**: 하지만, 연구 결과에 따르면 파인 튜닝된 모델은 KELM-sub 데이터셋의 일반화(generalization) 작업에서 성능이 떨어지는 것으로 밝혀졌습니다. 이 연구는 텍스트 데이터를 기반으로 한 지식 그래프 구축의 최전선에서 나타나는 포괄적인 평가 메트릭스의 중요성을 강조합니다.



### Efficient Knowledge Feeding to Language Models: A Novel Integrated Encoder-Decoder Architectur (https://arxiv.org/abs/2502.05233)
Comments:
          Submitted to ACM TIST journal: under revision stage, 8 pages, 2 figures

- **What's New**: 이 논문은 언어 모델(LLM)에 지식을 효과적으로 공급하는 새로운 접근 방식을 소개합니다. 기존의 Retrieval-Augmented Generation (RAG) 모델이 가지는 여러 한계를 극복하기 위해 in-context vectors (ICV)를 도입합니다. ICV는 LLM의 잠재 임베딩을 활용하여 작업 정보의 본질을 캡처하는 벡터를 생성합니다. 이 벡터는 LLM의 잠재 상태를 이동시키는 데 사용되어 추가적인 예시를 포함하지 않고 생성 과정을 향상시킵니다.

- **Technical Details**: 제안된 아키텍처는 정보 검색과 생성 과정을 통합하여 최신 문서의 정보를 효과적으로 처리합니다. ICV는 LLM의 잠재 상태를 다시 재구성하여 최근의 작업에 필요한 데이터를 통합합니다. 이 구조는 생성 과정에서 필요한 정보의 질과 관련성을 향상시키고, 기존의 RAG 모델의 제한 사항을 완화합니다. 이 방법은 더 짧은 프롬프트 및 계산 효율성을 제공하며, 미세 조정(fine-tuning)과 비교하여 더 적은 계산 비용을 요구합니다.

- **Performance Highlights**: 실험 결과, ICV가 표준 in-context learning 및 미세 조정 방식보다 전반적인 성능이 뛰어난 것으로 나타났습니다. 질문 답변, 정보 검색 등 다양한 작업에서 ICV 강화 모델이 LLaMA-3, Gemma, Phi-3와 같은 기존 모델들과 경쟁할만한 성능을 발휘했습니다. 계산 비용과 메모리 요구 사항을 획기적으로 줄이면서도 성능을 유지하는 이 접근 방식은 정확성과 효율성을 높였습니다.



### Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies (https://arxiv.org/abs/2502.05202)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 추론(Inference) 속도를 높이기 위한 새로운 방법인 Speculative Decoding(SD) 기법을 소개합니다. 기존의 SD 방법들과 달리, 제안된 세 가지 새로운 방법은 드래프(Drafter)와 타겟 모델간의 동일한 어휘(Vocabulary)를 요구하지 않습니다. 이를 통해 새로운 드래프를 훈련할 필요 없이 누구나 사용할 수 있는 모델을 드래프 모델로 활용할 수 있게 되었습니다.

- **Technical Details**: 이 연구는 세 가지 새로운 SD 알고리즘을 제안합니다. 첫 번째는 String-Level Exact Match(SLEM)라는 알고리즘으로, 드래프와 타겟 어휘 간에 텍스트를 공유하여 정확한 일치를 가능하게 합니다. 두 번째는 Token-Level Intersection(TLI)로, 드래프의 분포를 두 어휘의 교집합으로 조정하여 샘플링합니다. 마지막으로, String-Level Rejection Sampling(SLRS) 기법은 토큰 레벨이 아닌 문자열 레벨에서 샘플링을 수행하며, 이 모든 알고리즘은 손실이 없는(lossless) 특성을 유지합니다.

- **Performance Highlights**: 제안된 알고리즘들은 요약(Summarization), 프로그래밍(Programming), 긴 문맥(Long Context) 작업에서 기존의 자가 회귀(Self-Regressive) 디코딩 방법에 비해显著한 속도 향상을 보였습니다. Hugging Face Transformers 라이브러리에 통합되어 현재 26만 개의 리포지토리와 5천 개의 오픈 소스 패키지에서 즉각적인 성능 향상을 지원하고 있습니다. 이를 통해 실제 응용에서의 SD 프레임워크의 적용 가능성이 크게 넓어졌습니다.



### LLMs Provide Unstable Answers to Legal Questions (https://arxiv.org/abs/2502.05196)
Comments:
          6 pages

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 불안정성을 분석합니다. gpt-4o, claude-3.5, gemini-1.5와 같은 주요 LLM들이 법적 질문을 처리할 때 일관성 없이 서로 다른 결론을 내리는 문제를 발견했습니다. 이는 LLM이 결정적으로 작동할 것이라는 가정에 의존하고 있는 법률 AI 제품 및 전문가들에게 영향을 미칠 수 있습니다.

- **Technical Details**: 연구진은 500개의 법적 질문을 포함하는 새로운 데이터셋을 구축하고, 각각의 질문에 대해 20회 동일한 입력을 줘서 LLM의 응답을 분석했습니다. 특정 모델이 법적 질문에 대해 상대방의 승리를 주장하거나 반대의 주장을 할 경우 불안정하다고 정의하였습니다. 분석 결과, 모든 LLM이 질문의 10.6%에서 50.4%까지 불안정성을 보였습니다.

- **Performance Highlights**: 성능 측면에서, gpt-4o와 claude-3.5는 실제 법원 판결의 결과와 비교할 때 약간의 정확성을 나타냈으나, gemini-1.5는 오히려 우연보다 못한 성과를 보였습니다. OpenAI의 오랜 사고 모델 o1의 경우, 설정 온도가 1.0인 덕분에 더 높은 불안정성을 나타냈지만 gpt-4o보다는 낮은 성능을 보였습니다.



### On the Emergence of Thinking in LLMs I: Searching for the Right Intuition (https://arxiv.org/abs/2502.06773)
Comments:
          Abstract shortened for arXiv

- **What's New**: 최근 AI 발전으로 OpenAI의 새로운 모델들이 LLM(대규모 언어 모델)에서 LRM(대규모 추론 모델)로 변화하고 있습니다. 이들 모델은 추론(inference) 중에 reasoning을 수행하며, 더 높은 품질의 출력을 위해 추가적인 시간과 연산(compute)을 소모합니다. 본 연구는 LRM 교육을 위한 알고리즘 프레임워크를 탐구하고 있습니다.

- **Technical Details**: 우리는 RLSP(자기 놀이를 통한 강화 학습)라는 사후 훈련(post-training) 프레임워크를 제안합니다. RLSP는 (1) 인간 또는 합성 합시행(demonstrations)을 통한 감독된 미세 조정(supervised fine-tuning), (2) 다양한 효율적인 추론 행동을 촉진하기 위한 탐색 보상 신호(exploration reward signal) 사용, (3) 정답 검증(outcome verifier)과 함께하는 RL 훈련의 세 단계로 이루어져 있습니다. PPO 훈련 중 탐색(exploration)과 정확성(correctness) 신호를 분리하여 성능과 효율성을 개선하는 것이 주요 혁신입니다.

- **Performance Highlights**: 실증 연구에 따르면 RLSP는 수학 분야에서 추론을 개선하는 것으로 나타났습니다. Llama-3.1-8B-Instruct 모델에서 MATH-500 테스트 세트에서 23% 성능 향상을 보였고, AIME 2024 수학 문제에서는 Qwen2.5-32B-Instruct가 RLSP 덕분에 10% 개선되었습니다. 또한, RLSP로 훈련된 모델은 간단한 탐색 보상을 통해 여러 emergent behaviors를 보여주어, LLM이 복잡한 추론 능력을 발휘할 수 있도록 할 수 있음을 시사합니다.



### The 2021 Tokyo Olympics Multilingual News Article Datas (https://arxiv.org/abs/2502.06648)
- **What's New**: 이 논문에서는 2021 도쿄 올림픽에 대한 다국어 뉴스 기사 데이터셋 OG2021을 소개합니다. 총 10,940개의 뉴스 기사가 1,918개의 출처로부터 수집되어, 다양한 언어로 작성되었습니다. 이 데이터셋은 여러 사건에 대한 보도 기사를 그룹화하는 데 도움을 주기 위해 개발되었습니다.

- **Technical Details**: 이 데이터셋은 온라인 클러스터링 알고리즘을 활용하여 같은 하위 이벤트에 대한 기사를 그룹화하고, 수작업으로 주석을 달아 평가되었습니다. 언어는 영어, 스페인어, 독일어, 프랑스어, 러시아어 등을 포함하며, 2021년 7월 1일부터 8월 14일까지 출판된 기사를 포함합니다.

- **Performance Highlights**: OG2021 데이터셋은 특히 고빈도 이벤트가 발생하는 상황에서의 다국어 뉴스 클러스터링 알고리즘 성능 평가에 적합합니다. 이는 도쿄 올림픽의 문화적 및 언어적 차이를 분석하는 데도 유용하게 사용될 수 있습니다.



### ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms (https://arxiv.org/abs/2502.06556)
- **What's New**: 이 논문에서는 기존의 함수 또는 클래스 수준의 테스트 생성 벤치마크의 한계를 극복하기 위해 ProjectTest라는 새로운 프로젝트 수준의 유닛 테스트 생성 벤치마크를 제안합니다. ProjectTest는 Python, Java 및 JavaScript를 포함한 세 가지 프로그래밍 언어에 대해 20개의 중간 크기 및 고품질 프로젝트를 특징으로 합니다. 해당 벤치마크에서 9개의 최첨단 대형 언어 모델(LLMs)에 대한 평가를 수행하며, 이들의 중간 성능을 보여주었습니다.

- **Technical Details**: ProjectTest는 GitHub에서 선별된 프로젝트 레포지토리로부터 구축되었으며, 각 프로젝트는 최대 1,600행의 코드로 구성되어 있습니다. 프로젝트는 서로 다른 파일 간의 의존성을 가지고 있으며, LLM의 성능을 결합하여 평가하기 위해 수동 오류 수정 및 자가 오류 수정 시나리오에서 추가 평가를 수행합니다. 이를 통해 LLM의 기본 성능 외에 오류 수정 메커니즘과 결합했을 때의 발전 가능성을 평가합니다.

- **Performance Highlights**: ProjectTest의 결과, 모든 평가된 LLM은 Python과 Java에서 중간 성능을 보이며, 이는 ProjectTest의 난이도를 강조합니다. 특히 Java 언어가 가장 높은 난이도를 보이며, Claude-3.4-Sonnet와 GPT-o1이 각각 Java와 JavaScript에서 가장 높은 성능을 기록했습니다. 수동으로 수정한 후 평가한 결과, 오류 분포와 각 모델의 잠재력에서 큰 차이를 발견했으며, 자가 수정 기능이 인간 수정의 품질과 신뢰성에 비해 여전히 뒤처져 있음을 확인했습니다.



### MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations (https://arxiv.org/abs/2502.06453)
- **What's New**: 본 논문은 새로운 MATH-P-Simple과 MATH-P-Hard 벤치마크를 구축하여 언어 모델의 수학적 추론 성능을 평가합니다. Simple perturbation(간단한 변화)과 hard perturbation(어려운 변화)을 통해 원래 문제의 해결 방식과의 연관성을 유지하면서도 본질적인 문제의 구조를 변경합니다. 이를 통해 모델에서 관찰된 성능 저하의 원인을 규명하며, 모델이 문제의 수정된 맥락을 평가하지 않고 무작정 학습된 기술을 적용하는 새로운 형태의 암기(memorization)를 발견했습니다.

- **Technical Details**: MATH-P-Simple은 279개의 변형된 수학 문제로 구성되어 있으며, 원본 문제는 MATH 데이터셋의 레벨 5(가장 어려운) 문제에서 파생되었습니다. 각 변형 문제는 원본 문제를 바탕으로 하여 최소한의 수정(minimal edits)을 적용하고, 수정된 문제의 답이 원본과 다르게 설정되었습니다. 또한 12명의 수학 박사 과정 학생들에 의해 품질 통제를 위한 이중 검사(double-checking)가 수행되었습니다.

- **Performance Highlights**: Varying models such as o1-mini와 gemini-2.0-flash-thinking은 MATH-P-Hard에서 각각 -16.49% 및 -12.9%의 성능 저하를 보였습니다. 이는 모델들이 원래 문제 형식에 편향되어 있으며, hard perturbations의 특정 설정에서 전혀 새로운 문제에 직면했을 때 이전의 해결 기술을 무작정 적용함으로써 발생하는 문제점을 강조합니다. 연구진은 이러한 문제를 해결하기 위한 추가 연구의 필요성을 강조합니다.



### Content-Driven Local Response: Supporting Sentence-Level and Message-Level Mobile Email Replies With and Without AI (https://arxiv.org/abs/2502.06430)
Comments:
          23 pages, 14 figures, 2 tables, ACM CHI 2025

- **What's New**: 이 논문에서는 Content-Driven Local Response (CDLR)라는 새로운 UI 개념을 제시합니다. 이는 사용자가 이메일에 응답을 삽입할 때 문장을 선택하여 AI 제안을 안내할 수 있도록 돕습니다. CDLR은 사용자에게 적당한 AI 개입의 정도를 선택할 수 있는 유연한 워크플로우를 지원하며, 입력 오류와 타이핑 시간을 줄이는 이점을 유지합니다. 이 연구는 AI 가능성을 통합하는 새로운 접근법을 제시함으로써 사용자가 AI의 개입 정도를 동적으로 조정할 수 있도록 돕습니다.

- **Technical Details**: CDLR 개념은 모바일 마이크로태스킹의 원칙에 영감을 받아 개발되었습니다. 사용자는 응답하고자 하는 문장을 선택함으로써 로컬 응답 위젯이 나타나고, 이를 통해 수동으로 응답하거나 문장 제안을 받아들일 수 있습니다. 또한, 사용자는 이메일의 전체 메시지 수준에서 AI 개선 과정을 거치거나 수동 편집을 통해 응답을 지속할 수 있습니다. 연구는 126명의 참가자를 대상으로 CDLR의 유효성을 평가하고, 전통적인 수동 타이핑 또는 메시지 생성 방식과 비교하였습니다.

- **Performance Highlights**: 연구 결과 CDLR 시스템은 문장 수준과 메시지 수준 지원의 중간 지점에서 새로운 차별성을 보임을 발견했습니다. 사용자는 다양한 AI 개입의 정도로 이를 활용하면서 타이핑 감소와 입력 오류의 이점을 유지했습니다. 참가자들의 피드백을 통해 CDLR이 일반적인 이메일 앱의 AI 통합 방식과 비교했을 때 상이한 사용자 경험을 제공했음을 확인할 수 있었습니다. 이 연구는 AI 기능을 더욱 효과적으로 통합할 수 있는 새로운 디자인 접근법을 제시합니다.



### Evaluating Entity Retrieval in Electronic Health Records: a Semantic Gap Perspectiv (https://arxiv.org/abs/2502.06252)
Comments:
          Under review, and the dataset will be made public upon reception of our paper

- **What's New**: 이번 논문은 전자 건강 기록(EHR)에서의 엔터티 검색(entity retrieval) 성능 평가를 위한 새로운 벤치마크인 CliniQ의 개발 및 공개를 제안합니다. 특히, 논문에서는 MIMIC-III 데이터셋의 퇴원 요약(discharge summaries)을 사용하여 ICD 코드와 처방 레이블을 질의(query)로 수집하고, GPT-4를 이용해 관련성 판단(relevance judgments)을 주석처리하였습니다. 1,000개의 환자 노트를 기반으로 1,246개의 질의를 생성하고 77,000개 이상의 관련성 주석을 제공하며, 이를 통해 단일 환자 검색(Single-Patient Retrieval)과 다중 환자 검색(Multi-Patient Retrieval)의 두 가지 설정에 대한 성능을 평가합니다.

- **Technical Details**: CliniQ는 고품질 주석 및 다양한 적용을 위한 두 가지 검색 설정을 포함하는 대규모 질의 집합을 특징으로 합니다. 본 연구는 또한 질의 매칭의 유형을 다섯 가지로 분류하는 새로운 시스템을 도입하였으며, 이는 문자열(string), 동의어(synonym), 약어(abbreviation), 하위어(hyponym), 그리고 함의(implication)입니다. GPT-4를 통해 의료 전문가와의 높은 일치를 나타내는 주석을 생성했으며, 이 주석들은 1,000개의 MIMIC-III 퇴원 요약을 통해 수집된 데이터입니다.

- **Performance Highlights**: BM25는 강력한 기준선(baseline)을 제공하지만, 의미론적 일치(semantic matches)에서 어려움을 겪고 있습니다. 질의 확장(query expansion)은 성능을 크게 향상시키지만, 문자열 일치 능력을 약간 감소시킵니다. 밀집 검색기(dense retrievers)는 전통적인 방법들을 초월하며, 특히 의미론적 일치에서 뛰어난 성능을 보여주고 있습니다.



### LessLeak-Bench: A First Investigation of Data Leakage in LLMs Across 83 Software Engineering Benchmarks (https://arxiv.org/abs/2502.06215)
Comments:
          25 pages

- **What's New**: 본 논문은 대규모 소프트웨어 엔지니어링(Software Engineering, SE) 벤치마크에서 대형 언어 모델(Large Language Models, LLM)에 대한 데이터 유출(data leakage) 문제를 최초로 분석합니다. 83개의 SE 벤치마크에 대한 대규모 조사를 통해, Python, Java, C/C++ 각각의 평균 유출 비율이 4.8%, 2.8%, 0.7%로 나타나 데이터 유출 문제가 미미함을 강조합니다. 그러나 QuixBugs와 BigCloneBench는 각각 100.0%와 55.7%의 높은 유출 비율을 기록하여 평가의 편향 가능성을 지적합니다.

- **Technical Details**: 이 연구는 LLM의 성능에 대한 데이터 유출 영향을 분석하기 위해 DetectLeak라는 다단계 접근 방식을 제안합니다. 이 방법은 MinHash+LSH라는 자동화 도구를 활용하여 약 1.7조 쌍의 LLM의 사전 훈련 데이터와 SE 벤치마크 데이터를 비교하여 잠재적 중복 쌍을 식별합니다. 이후 숙련된 개발자들이 이 중복 쌍을 수동으로 레이블링하여 실제 중복 및 데이터 유출을 확인합니다.

- **Performance Highlights**: 연구 결과, StarCoder-7b는 APPS 벤치마크에서 유출 샘플에서 Non-leaked 샘플보다 4.9배 높은 Pass@1 점수를 기록하였습니다. 이는 유출된 벤치마크 샘플의 존재가 성능 지표를 크게 부풀릴 수 있음을 보여줍니다. 이 연구의 결과는 SE 벤치마크의 데이터 유출 문제가 LLM의 평가에 미치는 중요한 영향을 강조하며, 향후 보다 신뢰할 수 있는 LLM 평가를 위한 LessLeak-Bench라는 새로운 벤치마크를 제안합니다.



### Uncertainty-Aware Adaptation of Large Language Models for Protein-Protein Interaction Analysis (https://arxiv.org/abs/2502.06173)
- **What's New**: 이번 연구에서는 단백질-단백질 상호작용(PPIs) 분석을 위한 불확실성 인식(uncertainty-aware) LLMs의 적응을 제안한다. LLaMA-3 및 BioMedGPT 모델을 고도화하여 특정 질병 맥락에서 예측신뢰성을 향상시키고, LoRA 앙상블 및 Bayesian LoRA 모델을 통합하여 불확실성 정량화(uncertainty quantification)를 수행한다. 이러한 방법을 통해 PPIs 식별 성능을 향상시키고, 생물정보학의 재현성(reproducibility) 문제를 해결하려 한다.

- **Technical Details**: 단백질-단백질 상호작용(PPIs)은 세포 기능의 분자적 기초를 나타낸다. 본 연구에서는 LoRA 기반의 미세调정(fine-tuning)을 사용하고, Bayesian LoRA 및 LoRA 앙상블 방법을 채택하여 전염성 예측을 개선한다. 이를 통해 PPI 네트워크에 관한 광범위한 불확실성 인식 평가를 수행하며, 신경퇴행성 질환, 대사 질환 및 암에 관련된 단백질 상호작용 네트워크를 분석한다.

- **Performance Highlights**: 우리의 접근 방식은 치명적인 생물 의학적 응용에서 발생할 수 있는 불확실성 문제를 해결하며, PPI 예측 정확도를 높인다. 또한, 잘 조정된 신뢰도 측정을 제공하여, 생물 의학 연구의 강력한 결론 도출을 가능하게 한다. 본 연구는 LLM 기반 모델링에서 안전하고 신뢰할 수 있으며 정보가 풍부한 계산 도구의 개발을 위한 토대를 마련한다.



### Universal Approximation of Visual Autoregressive Transformers (https://arxiv.org/abs/2502.06167)
- **What's New**: 이번 논문에서는 transformer 기반의 foundation model의 근본적인 한계를 조사하며, Visual Autoregressive (VAR) transformer를 포함한 새로운 분석을 제시합니다. VAR는 이미지 생성을 위한 새로운 조정 가능한 코스-투-파인 'next-scale prediction' 프레임워크를 통해 기존 방법들보다 우수한 품질을 보여줍니다. 우리의 주요 기여는 단일 헤드 VAR transformer가 이미지-투-이미지 Lipschitz 함수에 대한 보편적인 근사자임을 증명하는 것입니다.

- **Technical Details**: Transformer 기반 아키텍처는 현대 기계 학습의 경관을 변화시켰으며, self-attention 메커니즘을 통해 데이터의 장기 종속성을 효과적으로 모델링합니다. VAR transformer는 구조화된 이미지 합성을 위해 적응된 변형으로, 높은 품질의 이미지를 더 효율적으로 생성합니다. 이 연구는 VAR transformer와 Flow AutoRegressive (FlowAR) 모델의 보편성에 대한 정량적 분석을 통해, 이들 모델이 복잡한 함수를 근사하는 데 충분한 표현력을 갖추고 있음을 밝힙니다.

- **Performance Highlights**: VAR transformer는 간단한 디자인만으로도 임의의 Lipschitz sequence-투-sequence 기능을 근사할 수 있으며, 이는 VAR 설정에서 고전적인 보편성 결과를 확장합니다. FlowAR 모델도 유사한 근사 능력을 보이고, 두 모델의 상호작용은 생성 모델 설계에 있어 효율성 및 표현력을 동시에 만족할 수 있는 길을 제시합니다. 이로써 효율성과 표현력이 반드시 상반되지 않음을 증명하며, 이러한 기초 연구 결과는 모델 심도, 헤드 수 및 근사 효율성 간의 트레이드오프를 이해하는 데 중요한 기초를 마련합니다.



### Enhancing Document Key Information Localization Through Data Augmentation (https://arxiv.org/abs/2502.06132)
Comments:
          Accepted as a workshop paper in DOCUI-AAAI2025

- **What's New**: 본 논문은 디지털 및 손글씨 문서에서 핵심 정보를 로컬화하는 방법을 제시합니다. 특히, 디지털 문서만을 이용하여 훈련한 후, 손글씨 문서의 특성을 모방함으로써 일반화 능력을 높이는 데이터 증강(data augmentation) 기술을 사용합니다. 실험 결과, 제안된 파이프라인이 모델의 성능을 향상시키는 데 효과적임을 보여주었습니다.

- **Technical Details**: 연구는 Form-NLU 데이터셋을 사용하여 해당 파이프라인의 유효성을 테스트했습니다. Augraphy를 사용하여 다양한 문서 유형을 생성하고, 특정 여섯 가지 증강 기법을 선택하였습니다: InkBleed, Letterpress, LowInkRandomLines, LowInkPeriodicLines, JPEG, DirtyScreen 등이 그것입니다. 각 훈련 및 검증 문서에 대해 다섯 개의 증강된 버전을 생성하며, 회전 및 텍스트 효과를 고려하여 총 70%와 50%의 확률로 적용합니다.

- **Performance Highlights**: 제안된 파이프라인의 결과는 세 가지 모델에서 손글씨 문서 정보 로컬화 성능을 3.97% 향상시켰고, 특히 문서 이미지로 미리 훈련된 백본 모델이 자연 이미지로 훈련된 모델보다 일반화 능력이 우수하다는 것을 보여주었습니다. LayoutLMv3는 디지털 문서에서는 높은 성능을 보였으나, 손글씨 문서에서는 OCR 오류로 인해 성능이 떨어졌습니다. 이러한 결과는 증강 방식이 디지털 문서만을 사용하여 손글씨 문서의 도메인을 효과적으로 모방함을 입증합니다.



### Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models (https://arxiv.org/abs/2502.06130)
Comments:
          Accepted by ICLR 2025. Project page:this https URL

- **What's New**: 최근 대규모 시각-언어 모델(LVLMs)의 성능 향상이 눈에 띄지만, 현실에서는 시각 입력과 일치하지 않는 환각적(hallucinatory) 텍스트 응답을 생성하는 경향이 있습니다. 이 논문은 텍스트-이미지 생성 모델을 활용하여 LVLM의 환각을 완화할 가능성을 모색합니다. 새로운 자기 수정 디코딩 기법인 Generative Feedback(DeGF)를 제안하여 이러한 환각을 효과적으로 줄입니다.

- **Technical Details**: DeGF는 정교한 훈련 과정 없이 텍스트-이미지 생성 모델로부터의 피드백을 통합하여 LVLM의 응답 정확성을 향상시키는 알고리즘입니다. 이 방법은 LVLM이 생성한 초기 응답을 기준으로 새로운 이미지를 생성하고, 이 보조 시각 참조를 통해 응답의 일치를 평가합니다. 구체적으로, 초기 응답과 생성된 이미지 간의 차이를 활용하여 초기 응답을 검증하고 수정하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, DeGF는 POPE 및 MMBench와 같은 여섯 가지 벤치마크에서 다양한 유형의 환각을 효과적으로 줄이며 기존 최첨단 기법보다 일관되게 우수한 성능을 보였습니다. 연구팀은 LVLM 응답의 정확성과 구체성을 향상시킬 수 있는 능력을 입증하였으며, 이는 시각 인사이트를 더하고 초기 응답을 확인하여 정확한 세부 사항을 보장하는 데 기여합니다.



### Circuit-tuning: A Mechanistic Approach for Identifying Parameter Redundancy and Fine-tuning Neural Networks (https://arxiv.org/abs/2502.06106)
- **What's New**: 이 연구에서는 기계적 해석성(mechanistic interpretability)을 통해 모델의 동작을 설명하는 방법에 대해 조사했습니다. 기존 연구들이 특정 행동의 정적 메커니즘에 초점을 맞춘 반면, 이 논문은 모델 내부의 학습 동역학(training dynamics)을 탐구합니다. 새로운 방법으로 제안된는 circuit-tuning 알고리즘을 통해 노드 중복(node redundancy)의 개념이 도입되었으며, 이는 학습 메커니즘에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서 제안된 circuit-tuning 알고리즘은 두 단계로 이루어져 있으며, 각 단계에서 모델의 비관련 엣지(edges)를 마스킹(mask)하고 특정 작업을 담당하는 나머지 파라미터를 업데이트합니다. 본 알고리즘은 기존의 테크닉보다 우수하며 다양한 모델과 작업에 대해 확장 가능합니다. 또한, 비선형 네트워크의 자가 조직화(self-organization) 메커니즘을 분석하고 시각화하여 모델 학습 중의 변화 과정을 명확히 보여줍니다.

- **Performance Highlights**: 실험 결과, 논문에서 제안한 방법은 다양한 작업에서 성능을 개선하였을 뿐만 아니라, 일반적인 능력을 보존하면서도 확장성(scaleability)을 가지고 있음을 입증하였습니다. 이는 AI의 해석성 뿐만 아니라 실제 응용 가능성에 긍정적인 영향을 미칠 것으로 기대됩니다. 연구진은 학습 과정의 직관을 더욱 깊게 제공하며, Fine-tuning에서의 새로운 접근법을 제시합니다.



### RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning (https://arxiv.org/abs/2502.06101)
Comments:
          Accepted by TheWebConf'25 (WWW'25) as a Short Paper

- **What's New**: 최근 대형 언어 모델(LLM)이 추천 시스템에 통합되어 사용자 행동 이해를 향상시키고 있습니다. 본 논문에서는 Retrieval Augmented Generation(RAG) 기법을 통해 더욱 관련성 높은 아이템을 검색하고 시스템의 성능을 개선하고자 하였습니다. 기존 RAG 방법론은 주로 텍스트 의미에 의존하며, 가장 관련성 높은 아이템을 포함하는 데 한계가 있습니다. 이를 개선하기 위해 우리는 LLM을 사용하여 아이템에 대한 자세한 설명을 생성하고, 이를 기반으로 공동 표현 학습을 수행합니다.

- **Technical Details**: 우리가 제안하는 RALLRec은 텍스트 의미와 협업 의미를 향상시키기 위해 LLM이 아이템의 상세한 설명을 생성하도록 유도합니다. 이 생성된 설명을 통해 향상된 아이템 표현을 추출하고, 이를 추천 모델을 통해 협업 의미와 결합하여 최종 표현을 만듭니다. 또한, 사용자 선호의 동적 특징을 고려하여 효과적인 재정렬 방법을 도입하여 추천의 유효성을 더욱 높였습니다.

- **Performance Highlights**: 세 가지 실제 데이터 세트에서 광범위한 실험을 수행한 결과, 우리의 방법이 효과적임을 입증하였습니다. 제안하는 RALLRec은 관련 아이템을 효과적으로 검색하고, 사용자 선호를 반영한 다양한 재정렬 전략을 통해 성능을 향상시켰습니다. 실험 결과는 제안된 방법의 유효성을 뒷받침하며, 향후 추천 시스템의 발전에 기여할 것으로 기대됩니다.



### Deconstructing Depression Stigma: Integrating AI-driven Data Collection and Analysis with Causal Knowledge Graphs (https://arxiv.org/abs/2502.06075)
Comments:
          Conditionally accepted to CHI Conference on Human Factors in Computing Systems (CHI'25)

- **What's New**: 이 연구에서는 챗봇(Chatbot)과 인공지능(AI) 도움을 활용하여 정신질환에 대한 낙인(stigma)을 이해하기 위한 새로운 접근법을 제시합니다. 1,002명의 참가자와의 대화에서 수집된 데이터를 바탕으로, AI가 코드화 작업을 수행하고 인과적 지식 그래프(causal knowledge graphs)를 구축하여 낙인의 메커니즘을 분석했습니다. 이러한 방법은 기존 연구와 비교할 때 비효율적인 데이터 수집과 분석의 문제를 효율적으로 해결하는 데 기여합니다.

- **Technical Details**: 연구 방법론으로는 AI 챗봇을 이용해 심리적 구성이 나타나는 대화를 수집한 후, AI 보조 코딩을 통해 낙인의 속성을 식별했습니다. 연구 과정에서 인과적 지식 그래프(CKG)와 대형 언어 모델(LLM)의 통합이 이루어져, 각기 다른 심리적 구성 요소 간의 관계를 시각화했습니다. 이를 통해 정서적 반응 및 차별적 행동의 예측 요인으로서의 성격과 같은 새로운 인과관계가 밝혀졌습니다.

- **Performance Highlights**: 연구 결과, AI 보조 코딩이 전문가의 코딩과 높은 일관성을 보였으며, 챗봇 대화가 우울증에 대한 사람들의 태도에 대한 깊이 있는 정보를 이끌어낼 수 있음을 입증했습니다. 본 연구는 HCI(인간-컴퓨터 상호작용) 분야에서 정신 건강 문제에 대한 디지털 개입을 설계할 때의 기초 자료로 활용될 수 있으며, 개인 맞춤형 개입 방법론 개발에 기여할 가능성을 보여주고 있습니다.



### Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.06060)
Comments:
          14 pages, 5 figures, 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 이번 연구에서는 인간의 예제가 없이도 자연어를 사용해 환경에 대한 생산적인 논의를 할 수 있는 언어 모델을 훈련하는 방법을 제안합니다. 특히, 에이전트의 목표를 활용하여 유용한 정보 예측을 보상 신호로 삼아 소통 문제를 '듣기'와 '말하기'로 나누어 해결합니다. 이러한 접근 방식은 복잡한 사회적 환경에서의 커뮤니케이션의 중요성을 조사하는 데 도움이 됩니다.

- **Technical Details**: 연구에서는 Among Us 게임을 기초로, 에이전트들이 자연어로 효과적으로 소통하도록 학습하는 방법을 제시합니다. 에이전트는 서로의 믿음을 업데이트하고, 정보 수집을 통해 가장 유용한 메시지를 전달하도록 훈련됩니다. 이를 위해 강화 학습(MARL) 기법을 사용하며, 대화 중의 메시지 전달과 이해를 동시적으로 개선합니다.

- **Performance Highlights**: 제안된 기법을 통해 에이전트 간의 의사소통 전략은 더욱 강력해지며, 승률이 표준 RL 모델에 비해 두 배 증가했습니다. 또한, 규모가 네 배 큰 기본 모델과 비교해도 성공률이 세 배 더 높은 결과를 보여주어 제안된 논의 전략의 중요성을 강조합니다.



### Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection (https://arxiv.org/abs/2502.06042)
Comments:
          19 pages, 15 figures, preprint

- **What's New**: 이 연구에서는 미세 조정(finetuning) 과정에서 직면하는 두 가지 주요 문제인 과적합(overfitting)과 원래 모델과의 거리 증가를 다루고 있습니다. 불행히도 대부분의 실제 응용 프로그램에서 목표 데이터의 양이 제한적이기 때문에 이러한 문제는 흔히 발생합니다. 저자들은 다양한 목표 도메인 및 모델 스케일에 대한 스케일링 법칙(scaling laws)을 도출하고 있습니다.

- **Technical Details**: 연구진은 미세 조정 과정에서 불러온 전이 학습(pretraining) 데이터를 혼합하여 과적합을 완화하고 잊는 현상을 억제하는 방법을 측정했습니다. 이를 통해, 목표 도메인 및 데이터 양에 따라 모델이 어떻게 변화하는지를 정량화하는 접근 방식이 제시됩니다. 연구 결과, 미세 조정 데이터 혼합에 전이 학습 데이터의 1%만 추가함으로써 모델의 잊어버림 현상을 예방할 수 있다는 점이 강조됩니다.

- **Performance Highlights**: 이 연구는 전이 학습 데이터가 미세 조정 데이터와 혼합될 때 작은 양의 추가가도 큰 영향을 미친다는 것을 보여줍니다. 이 발견은 다양한 목표 도메인에서 더 나은 성능을 발휘할 수 있는 모델 개발에 중요한 실용적 통찰을 제공합니다. 따라서 전이 학습 데이터를 효과적으로 활용하는 것이 모델의 성능 유지에 필수적임을 시사합니다.



### MetaChain: A Fully-Automated and Zero-Code Framework for LLM Agents (https://arxiv.org/abs/2502.05957)
Comments:
          Code: this https URL

- **What's New**: MetaChain은 프로그래밍 경험이 없는 사용자도 자연어만으로 LLM(대규모 언어 모델) 에이전트를 만들고 배포할 수 있는 완전 자동화된 프레임워크입니다. 이 프레임워크는 Agent Operating System으로 작동하며, 사용자가 직접 코딩하거나 수동으로 개입하지 않고도 동적인 도구와 작업 흐름을 생성하고 수정할 수 있게 합니다. 보편적인 접근성을 촉진하기 위해 MetaChain은 독립적인 성능을 갖춘 다양한 구성 요소를 포함하고 있습니다.

- **Technical Details**: MetaChain은 자연어 기반의 다중 에이전트 구축, 자기 관리 워크플로우 생성, 그리고 지능형 자원 조정을 포함한 세 가지 주요 기능을 제공합니다. 이 기술들은 협업 에이전트 시스템을 구축하고, 높은 수준의 작업 설명에 기반하여 동적으로 워크플로우를 최적화하며, 자연어를 통해 도구와 API에 통합 접근을 가능하게 합니다. 이러한 아키텍처를 통해 MetaChain은 본질적으로 에이전트 개발을 민주화하는 역할을 합니다.

- **Performance Highlights**: MetaChain은 GAIA 벤치마크에서 강력한 성과를 보여주어 2위를 차지했으며, Retrieval-Augmented Generation 벤치마크에서 기존의 RAG 방법들과 비교했을 때 크게 우수한 성과를 기록했습니다. 실세계의 다양한 시나리오를 반영한 포괄적인 사례 연구를 통해 MetaChain의 자가 개발 능력과 실질적인 유용성을 평가했습니다. 이 결과는 자동화된 에이전트 개발의 강력한 가능성을 입증합니다.



### Acceleration Multiple Heads Decoding for LLM via Dynamic Tree Attention (https://arxiv.org/abs/2502.05947)
- **What's New**: 이번 논문에서는 다중 헤드 디코딩(multiple heads decoding) 원리를 적용하여 LLM의 추론 속도를 향상시키기 위해 고정 트리 주의(tree attention) 대신 동적 트리 주의(dynamic tree attention)를 도입합니다. 새로운 방법은 MEDUSA의 맥락에서 적용되며, 간단하고 저 복잡도의 전략으로 후보 집합을 생성하고 동적 트리 구조를 구성합니다. 초기 실험 결과는 제안된 방식이 LLM의 다중 헤드 디코딩 효율성을 증가시키며 생성 품질을 유지하는 것을 보여줍니다.

- **Technical Details**: 제안된 방법은 먼저 후보를 동적으로 생성한 후, 이 후보들에 따라 동적 트리 주의의 버퍼를 준비하는 방식으로 진행됩니다. 후보를 생성하기 위해 각 MARDA 스탭에서 마진 분포의 카르테시안 제품(Cartesian product)을 사용해 근사값을 계산하며, 생성된 후보들은 동적 트리 구조를 기반으로 구성됩니다. 이렇게 생성된 후보들은 우선순위 큐를 통해 가장 높은 확률을 가진 상위 n명 후보로 선택되어 O(Knm log n) 복잡도로 처리됩니다.

- **Performance Highlights**: 제안한 방법은 MT-Bench를 사용하여 추론당 토큰 수를 기준으로 평가되며, MEDUSA-1 및 MEDUSA-2에서 디코딩 효율성을 향상시키면서 생성 품질도 유지됩니다. 동적 트리 구조는 고정 트리 구조와 공통된 부분을 공유하면서도 맥락 의존성에 적응할 수 있는 장점을 가지고 있습니다. 비록 속도 면에서는 MEDUSA보다 약 10% 느리지만, 더 나은 디코딩 효율성을 제공합니다.



### A Generative Framework for Bidirectional Image-Report Understanding in Chest Radiography (https://arxiv.org/abs/2502.05926)
- **What's New**: MAViLT(다단계 적응형 비전-언어 조정) 프레임워크를 제안하여 흉부 X-레이(CXR) 이해 및 생성을 개선하는 새로운 방법론을 소개합니다. 이 방법론은 임상 데이터 기반의.tokenization 및 위계적 튜닝 전략을 통합하여 정밀한 비전-언어 정렬을 가능하게 합니다. 이를 통해 모델은 진단 보고서를 정확하게 작성하고, 텍스트에서 현실적인 CXR을 합성하며, 비전 기반 임상 질문에 답변할 수 있습니다.

- **Technical Details**: MAViLT는 임상 그래디언트 가중치 토큰화 프로세스와 위계적 미세 조정 전략을 도입하여 기존 기술의 한계를 극복합니다. 비전 텍스트 쌍을 활용한 다단계 훈련을 통해 모델은 다양한 CXR 관련 태스크에서 최적의 성능을 발휘합니다. 또한, 특정 태스크에 적합한 지침 템플릿을 통해 의학 이미징의 미세한 뉘앙스를 포착합니다.

- **Performance Highlights**: MAViLT는 MIMIC-CXR 및 Indiana University CXR 데이터셋에서 CXR-보고서 생성, 보고서-CXR 생성, 비전 질문 응답(VQA) 태스크에서 최상의 결과를 기록하며, 기존 최고 성능 모델들을 초과하는 성능을 보여주었습니다. 또한, 인간 평가를 통해 MAViLT의 임상 관련성과 효용성을 검증하여 실제 의료 적용에서 신뢰성을 제공합니다.



### Evaluating Vision-Language Models for Emotion Recognition (https://arxiv.org/abs/2502.05660)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 이 연구에서는 감정 인식 분야에서 대표적인 대형 비전-언어 모델(VLMs)의 포괄적인 평가를 처음으로 제시합니다. 새로운 벤치마크인 Evoked Emotion benchmark (EvE)를 개발하고, VLM들이 제시된 이미지와 텍스트 프롬프트를 기반으로 감정을 얼마나 잘 인식하는지 측정합니다. 이를 통해 VLM이 감정 인식에서 직면하는 주요 오류와 모델의 민감성을 분석합니다.

- **Technical Details**: 본 연구는 여러 감정 관련 데이터셋을 활용하여 VLM의 감정 인식 성능을 평가합니다. 연구에서는 특히 Evoked Emotion Recognition이라는 특정 작업에 초점을 맞추고, 모델의 정확성과 강건성을 평가하기 위해 8가지 다양한 실험 설정을 설계했습니다. 여기에는 프롬프트에서 감정 레이블의 순서 변경, 개방형 어휘 분류, 감정적 관점 채택 및 자기 추론 메커니즘 사용이 포함됩니다.

- **Performance Highlights**: 결과적으로 VLM은 현재 이미지에서 유발된 감정을 예측하는 데 부족함을 보이며, 프롬프트에서 감정 레이블의 순서에 민감하게 반응합니다. 또한, 특정 모델은 자기 추론 메커니즘을 활용할 때 성능이 개선되지만, 감정적 페르소나를 채택하는 방향이 성능에 부정적인 영향을 미친다는 것을 발견했습니다. 인간 평가 연구를 통해 잘못된 예측의 원인을 모델의 능력뿐만 아니라 사용된 데이터와 작업의 난이도에 기인한다는 점을 명확히 했습니다.



### Agentic AI Systems Applied to tasks in Financial Services: Modeling and model risk management crews (https://arxiv.org/abs/2502.05439)
- **What's New**: 본 논문은 금융 서비스 산업에서 에이전트 시스템의 새로운 작업 흐름을 탐구하며, 에이전트들이 협력하여 복잡한 모델링 및 모델 리스크 관리(Model Risk Management, MRM) 작업을 수행할 수 있는 에이전트 팀을 구축합니다. 이러한 팀은 탐색적 데이터 분석, 피쳐 엔지니어링(feature engineering), 모델 선택, 하이퍼파라미터 튜닝과 같은 다양한 작업을 수행하는 관리자를 포함하여 여러 에이전트로 구성되어 있습니다. 이 연구는 모델링 및 MRM 팀의 효과성을 입증하기 위해 신용카드 사기 탐지 및 포트폴리오 신용 리스크 모델링 데이터 세트에 대한 수치 예제를 제시하고 있습니다.

- **Technical Details**: 본 연구에서 제안된 에이전트 시스템은 다양한 전문 도구를 갖춘 협업 에이전트들을 통해 작동됩니다. 이렇게 구성된 시스템은 탐색적 데이터 분석, 모델 훈련 및 문서화 작업을 포함한 모델링 작업을 효율적으로 수행할 수 있도록 설계되었습니다. 또한, MRM 팀은 모델링 문서의 규정 준수 검사 및 결과 분석과 같은 반응적 작업을 전담하는 에이전트로 구성됩니다.

- **Performance Highlights**: 논문에서는 신용카드 사기 탐지와 같은 실제 금융 데이터 세트를 활용하여 모델링 및 MRM 작업의 성능을 입증했습니다. 이러한 에이전트 시스템은 복잡한 금융 서비스 문제를 해결하는 데 있어 놀라운 성과를 보여주었으며, 향후 다양한 금융 애플리케이션에서 함께 활용될 가능성을 제시합니다. 특히, 에이전트들 간의 협업을 통해 enhanced problem-solving capabilities를 달성하였다고 강조하였습니다.



### Toward Copyright Integrity and Verifiability via Multi-Bit Watermarking for Intelligent Transportation Systems (https://arxiv.org/abs/2502.05425)
Comments:
          11 figures, 10 tables. Accepted for publication in IEEE Transactions on Intelligent Transportation Systems (accepted versions, not the IEEE-published versions). ©2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission

- **What's New**: 본 논문에서는 Intelligent Transportation Systems (ITS)에 필요한 데이터 보호를 위한 새로운 수단으로 ITSmark라는 워터마킹 기법을 소개합니다. ITSmark는 저작권 정보를 기반으로 하여 사용자 맞춤형 워터마크의 삽입과 추출을 가능하게 하며, 데이터 무결성과 인증의 중요성을 강조합니다. 이 기법은 데이터 삽입 과정에서 고유한 보안 메커니즘을 구현하여, 데이터의 무단 접근 및 악용을 방지할 수 있는 기능도 포함되어 있습니다.

- **Technical Details**: ITSmark는 여러 개의 비트로 구성된 워터마크 영역을 생성하여 이를 여러 연속 비트 세그먼트로 나누고, 각 세그먼트는 특정 토큰에 할당됩니다. 이후 다음 토큰은 해당 세그먼트에 포함된 저작권 정보에 따라 결정됩니다. 이러한 방식으로, 사용자 맞춤형 워터마크가 삽입된 데이터가 생성되고, 추출 과정에서 맞춤형 워터마크가 전부 정확히 추출되도록 보장하는 구조를 가지고 있습니다. 또한, 권한 검증을 통해 데이터의 소유와 인증을 강화합니다.

- **Performance Highlights**: 실험 결과, ITSmark는 데이터 품질, 추출 정확도, 변조 방지 측면에서 기존 기법들과 비교해 우수한 성능을 보여주었습니다. 특히, 저작권 검증 및 변조 위치 추적 기능에서 독창적인 능력을 발휘하여 데이터 보안과 신뢰성을 확보할 수 있음을 demonstrated했습니다. 그는 사용자 요구에 따라 워터마크 임베딩 위치와 비율의 맞춤화도 지원하여 데이터 보호가 더욱 유연해졌습니다.



### Graph-based Molecular In-context Learning Grounded on Morgan Fingerprints (https://arxiv.org/abs/2502.05414)
- **What's New**: 이번 연구에서는 GAMIC (Graph-Aligned Molecular In-Context learning)라는 자가 지도 학습 기법을 제안합니다. GAMIC는 Graph Neural Networks (GNNs)와 Morgan fingerprint를 결합하여 분자의 전반적인 구조를 모델링하고 텍스트 설명과 정렬하는 방법을 사용합니다. 이를 통해 기존의 Morgan 기반 ICL 방법보다 최대 45% 향상된 성능을 보였습니다.

- **Technical Details**: GAMIC은 분자의 복잡한 구조를 인코딩하는 데 도움을 주며, 이를 통해 local feature similarity와 global structure 간의 상관관계를 캡처합니다. 특히, 계층적 그래프 인코더를 사용하여 분자의 표현을 처리하고, 과학적으로 의미 있는 텍스트 설명과의 잠재적 표현을 정렬합니다. 또한 Maximum Marginal Relevance (MMR)를 활용하여 입력 프롬프트의 예시 샘플 다양성을 최적화합니다.

- **Performance Highlights**: GAMIC의 실험적 결과는 다양한 벤치마크 데이터셋을 통해 기존의 Morgan 기반 ICL 방법들보다 더 뛰어난 성능을 보였습니다. 특히, 작은-중간 크기의 LLM에 대한 탐색이 많지 않았던 기존 문헌에서 이 연구는 의미있는 기여를 합니다. GAMIC은 분자 분석 및 예측 작업에서의 성능을 중시하며, 효율적인 모델 훈련을 위해 새로운 접근 방식을 제시합니다.



### Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond (https://arxiv.org/abs/2502.05374)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 ‘unlearning’ 기법이 갖는 취약점을 해결하기 위해, ‘relearning’ 공격에 대한 저항성을 강화하는 방법을 체계적으로 탐구합니다. 특히, 본 연구는 샤프니스 인식 최소화(Sharpness-Aware Minimization, SAM)와 robust optimization 프레임워크를 연결 지어, LLM unlearning의 강화를 위한 새로운 시각을 제공합니다. 이러한 접근은 전통적인 적대적 훈련(Adversarial Training)에서 영감을 받아, unlearning과 relearning 공격에 대한 방어를 동시에 고려합니다.

- **Technical Details**: 이번 연구의 핵심은 SAM을 활용하여 unlearning의 강건함을 개선하는 것입니다. SAM은 모델의 주변에서 균일하게 낮은 손실을 유지하도록 유도하여 부드러운 손실 경관을 촉진합니다. 본 논문에서는 SAM이 LLM unlearning의 robustness를 향상시키는 중요한 요소임을 보여주며, 다양한 smoothing 전략을 통해 unlearning의 불안정성을 해결하기 위한 실험도 진행합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터세트(WMDP, MUSE)에 대한 실험을 통해 SAM과 기타 부드러움 최적화 기법이 relearning 공격에 대한 LLM unlearning의 저항성을 일관되게 향상시킴을 입증합니다. 특히, 부드러움이 강화된 unlearning 기법은 input-level jailbreaking 공격에 대한 방어에도 효과적이며, 이는 LLM unlearning의 강건함을 더욱 넓히는 결과로 이어집니다.



### Optimizing Temperature for Language Models with Multi-Sample Inferenc (https://arxiv.org/abs/2502.05234)
Comments:
          20 pages. Code available at this https URL

- **What's New**: 이 논문은 다양한 LLMs(대규모 언어 모델)의 multi-sample aggregation strategies을 통해 (거의) 최적의 temperature를 자동으로 식별하는 방법을 제안합니다. 이는 기존의 labeled validation data에 의존하지 않고 수행될 수 있으며, LLM 성능 최적화에 있어 중요한 패러미터인 temperature의 역할을 체계적으로 분석합니다. 특히 entropy turning point(Entropy 전환점)라는 새로운 개념을 도입하여, 이 포인트가 LLM의 최적 온도를 자동으로 결정하는 데 유용할 수 있음을 보여줍니다.

- **Technical Details**: LLM은 입력 맥락과 이전에 생성된 토큰을 기반으로 다음 토큰의 조건부 확률 분포에서 autoregressively 표본을 생성합니다. 논문은 logits를 temperature hyperparameter T에 따라 조정하고 softmax 기능을 적용함으로써 확률 분포를 생성하는 방식을 설명합니다. 새로운 탐색적인 접근법으로 TURN을 제안하며, 이 방법은 temperature 최적화를 자동화하는 알고리즘 솔루션을 제공합니다.

- **Performance Highlights**: TURN 방법은 다수의 실험을 통해 수학 문제 해결, 코드 생성 등 다양한 작업에서 우수한 일반화 능력을 입증하였습니다. 고정 온도를 사용하는 기존 방법들과 비교할 때, TURN은 일관되게 성능을 개선하며 다양한 aggregation strategies(예: majority voting, best-of-N)에 대해 강력한 성능을 보입니다. 또한, Entropy 전환점 분석을 통해 온도의 역할을 해석할 수 있는 기회를 제공합니다.



### Robotouille: An Asynchronous Planning Benchmark for LLM Agents (https://arxiv.org/abs/2502.05227)
Comments:
          11 pages (not including references or appendix); 41 figures (7 main paper, 34 appendix); (v1) preprint

- **What's New**: 이 논문은 대규모 언어 모델 (Large Language Model, LLM) 에이전트의 비동기 계획 능력을 평가하기 위한 새로운 벤치마크 환경인 Robotouille을 소개하고 있습니다. 기존의 벤치마크는 주로 단기 과제에 집중되어 있었으나, 이 연구는 복잡한 장기 비동기 시나리오를 처리하는 능력을 측정하고자 합니다. 이를 통해 LLM 에이전트가 겹치는 작업과 중단을 관리하는 데 필요한 도전 과제를 드러냅니다.

- **Technical Details**: Robotouille 벤치마크는 동기와 비동기 데이터셋을 제공하여, 에이전트가 점점 복잡해지는 계획 과제를 해결할 수 있도록 설계되었습니다. 연구에서는 ReAct (gpt4-o) 모델을 평가했으며, 동기 작업에서는 47%, 비동기 작업에서는 11%의 성과를 보였습니다. 결과적으로 LLM 에이전트가 장기 피드백을 더 잘 통합하고 작업 실행 시 자신의 추론을 자기감사(self-audit)하는 것이 필요하다는 점을 강조합니다.

- **Performance Highlights**: ReAct 모델은 동기 작업에서 비교적 높은 성과를 보였지만, 비동기 작업에서는 낮은 성과를 기록하여 개선의 여지가 많음을 나타냅니다. 이 연구는 비동기 계획 능력을 향상시키기 위한 추가 연구의 필요성을 강조하며, LLM 에이전트가 다양한 작업 상황을 잘 처리할 수 있도록 해야 한다고 결론을 내립니다.



### KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs (https://arxiv.org/abs/2502.05223)
- **What's New**: 이 논문은 Knowledge-Distilled Attacker (KDA)라는 새로운 오픈소스 모델을 제안하여, 기존의 기법에서 발생하는 여러 문제를 해결하고 있습니다. 기존의 Jailbreak 공격 방식은 정교한 시스템 프롬프트에 의존하며 많은 쿼리가 필요해 실용성이 떨어졌습니다. KDA는 여러 SOTA 공격자들의 지식을 정제하여 단일 모델로 통합함으로써 이러한 제약을 극복합니다.

- **Technical Details**: KDA는 AutoDAN, PAIR 및 GPTFuzzer와 같은 세 가지 SOTA 공격자로부터 전략을 학습하여, 고유한 공격 프롬프트를 효과적으로 생성합니다. KDA는 Supervised fine-tuning을 통해 Vicuna-13B라는 아키텍처로 구현되어 있으며, 프롬프트 생성을 자동화하여 다양성과 효율성을 향상시킵니다. 이렇게 생긴 공격 프롬프트는 인체의 가치와 안전성을 고려하며, 다수의 상용 LLM과 오픈소스 모델 모두에 적용 가능합니다.

- **Performance Highlights**: KDA는 다양한 LLM에 대해 높은 공격 성공률(ASR)을 기록하며, 예를 들어 Llama-2-7B-Chat에서 88.5%, Llama-2-13B-Chat에서 83.5%의 ASR을 달성했습니다. KDA는 또한 Harmbench 및 Harmful-Behavior 데이터셋과 같은 불특정 데이터에 대해서도 높은 ASR을 유지하며 강력한 일반화 능력을 showcase하고 있습니다. 최종적으로 KDA는 형식 인셈블링을 통해 공격의 다양성과 성공률을 극대화하여 기존 모델보다 우수한 성능을 보여줍니다.



### Safety at Scale: A Comprehensive Survey of Large Model Safety (https://arxiv.org/abs/2502.05206)
Comments:
          47 pages, 3 figures, 11 tables

- **What's New**: 이 논문은 인공지능(AI) 분야의 대형 모델의 안전성에 관한 체계적 조사를 제공합니다. 특히, Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-Training (VLP) 모델과 같은 다양한 모델들의 안전성 취약점과 방어 전략을 정리하였습니다. 대형 모델의 안전성을 확보하기 위한 연구의 필요성과 국제적 협력의 중요성을 강조하고 있습니다.

- **Technical Details**: 대형 모델은 대규모 데이터 세트에서의 사전 학습을 통해 언어 이해, 이미지 생성, 복잡한 문제 해결 등의 작업에서 뛰어난 능력을 보입니다. 이러한 모델들은 적대적 공격(adversarial attacks), 백도어 공격(backdoor attacks), 데이터 중독(data poisoning), 에너지-지연 공격(energy-latency attacks) 등 다양한 안전 위협에 직면해 있습니다. 각 공격 유형에 대한 방어 전략 및 안전 연구를 위한 공통 데이터 세트와 벤치마크를 정리했습니다.

- **Performance Highlights**: 대형 모델의 안전성을 보장하는 것은 비의도적인 시스템 동작을 방지하고 개인 정보를 보호하기 위한 필수 사항입니다. 연구자와 실무자에게 유용한 참조 자료로 기능할 수 있으며, 포괄적인 방어 시스템 및 플랫폼의 지속적인 개발을 촉진하는 데 기여할 것입니다. 안전성 연구 현황을 통해 대형 모델의 발전을 가속화하고 안전한 AI 사용을 유도하는 것이 중요합니다.



### ChameleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters (https://arxiv.org/abs/2502.04315)
- **What's New**: 본 논문에서는 ChameleonLLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대규모 언어 모델의 동적 적응을 가능하게 하여, 추론(인퍼런스) 시간 동안 모델이 입력 데이터의 변동성에 적응할 수 있도록 합니다. 기존의 고정된 가중치(wights)를 가진 모델들과는 달리, ChameleonLLM은 배치(cluster) 통계를 기반으로 하여 실시간으로 저차원(low-rank) 업데이트를 생성합니다.

- **Technical Details**: ChameleonLLM은 입력을 의미적 및 문법적 유사성을 기반으로 클러스터링하여 그룹화합니다. 이때, 프리-컴퓨팅(precomputed)된 토큰 임베딩(token embeddings)을 활용하여 일관된 배치를 생성하고, 하이퍼 네트워크(hyper-network)를 통해 저차원 적응 파라미터를 실시간으로 생성합니다. 이는 고정된 사전 학습된 마스크(mask)에 의존하지 않고, 사용자가 제공하는 입력의 맥락에 맞춰 동적으로 조정됩니다.

- **Performance Highlights**: ChameleonLLM은 기존의 Low-Rank Adaptation(LoRA) 방법에 비해 성능이 크게 개선되었음을 실험을 통해 입증하였습니다. 이 방법은 여러 개의 전문가 모델을 유지할 필요가 없기 때문에, 메모리 및 계산 소모를 줄일 수 있습니다. 연구 결과는 다양한 언어 모델 작업에 대한 높은 적응성을 제공하는 가능성을 보여줍니다.



### A scale of conceptual orality and literacy: Automatic text categorization in the tradition of "Nähe und Distanz" (https://arxiv.org/abs/2502.03252)
- **What's New**: 이 논문은 Koch와 Oesterreicher의 'Nähe und Distanz' 모델을 통계적으로 뒷받침하고, 이를 코퍼스 언어학(corpus linguistics) 분석에 적용 가능한 체계로 발전시킵니다. 특히, 저자는 개념적 구술성(conceptual orality)과 문해성(literacy)을 평가하는 새로운 스케일(scale)을 제안합니다. 이러한 스케일은 PCA(주성분 분석) 기반으로 작성되었으며, 자동화된 분석과 통합하여 적용됩니다.

- **Technical Details**: 이 연구에서 저자는 두 개의 New High German 코퍼스(corpora)를 사용하여 개념적 구술성과 문해성을 평가하는 데 필요한 언어적 특징(feature)을 분석합니다. 연구 결과, 텍스트를 차별화된 방식으로 순위 매기기 위해서는 두 개념의 특징을 구별해야 한다는 중요한 발견이 이루어졌습니다. 또한 제안된 스케일은 코퍼스 수집 및 더 큰 코퍼스의 분석을 위한 가이드로서의 활용 가능성에 대해 논의됩니다.

- **Performance Highlights**: 이 접근법은 이론적으로 거시적인 시각을 취하고 있으며, Biber의 Dimension 1과 비교할 때 지원적이고 통제적인 작업에 특히 적합한 결과를 보여줍니다. 연구의 중심은 언어적 특징을 기반으로 한 텍스트의 평가 방식과 코퍼스 두 집합의 분석을 통해 강조됩니다. 이러한 차별적인 평가 방식은 언어학적 연구 및 교육의 질을 향상시킬 잠재력을 갖고 있습니다.



### EuskañolDS: A Naturally Sourced Corpus for Basque-Spanish Code-Switching (https://arxiv.org/abs/2502.03188)
- **What's New**: 코드 스위칭(Code-switching, CS)은 자연어 처리(NLP) 분야에서 여전히 중요한 도전 과제로 남아있습니다. 특히 스페인과 바스크 언어가 접촉하는 북 이베리아 반도에서 CS 현상이 자주 발생하지만, 이 언어 쌍을 이해하고 생성할 수 있는 모델을 개발하기 위한 자원은 거의 없는 실정입니다. 본 연구에서는 바스크-스페인 코드 스위칭을 위한 자연적으로 생성된 코퍼스를 개발하기 위한 첫 번째 접근 방식을 소개합니다.

- **Technical Details**: 연구진은 언어 식별 모델을 활용하여 기존 코퍼스에서 코드 스위칭 텍스트를 식별하고, 이를 수동으로 검증하여 신뢰할 수 있는 CS 사례의 하위 집합을 얻습니다. 최종 EuskañolDS 데이터셋은 자동 분류된 silver 세트와 수동 필터링된 gold 세트로 나뉘며, 언어 식별 모델이 217개 언어를 식별하도록 훈련되어 있습니다. 이 방법을 통해 총 20,008개의 silver 인스턴스와 927개의 gold 인스턴스를 수집하였습니다.

- **Performance Highlights**: EuskañolDS 데이터셋의 분석 결과, 상위 유형인 문장 간 코드 스위칭이 가장 많이 나타났으며, 특히 인스타그램이나 코로나19 관련 트윗에서 빈번하게 발견됩니다. 문장 내 코드 스위칭이 적게 나타나는 것은 소규모 데이터셋의 특성 때문이며, 각 언어의 방언 및 비공식 발화가 포함된 다양한 주제를 다룹니다. 연구진은 데이터셋과 그 과정에서 사용된 코드를 공개할 예정입니다.



New uploads on arXiv(cs.IR)

### exHarmony: Authorship and Citations for Benchmarking the Reviewer Assignment Problem (https://arxiv.org/abs/2502.07683)
- **What's New**: 이 논문은 exHarmony라는 새로운 벤치마크를 도입하여 Reviewer Assignment Problem (RAP)을 해결하고자 합니다. 기존의 수동 기법이 비효율적이고 비판적인 리뷰를 생성할 수 있어, 효율적이고 효과적인 리뷰어 할당을 위한 혁신적인 접근이 필요하다는 것을 강조합니다. OpenAlex로부터 방대한 데이터를 활용하여 리뷰어 할당의 품질을 향상시키는 다양한 신호를 고려합니다.

- **Technical Details**: exHarmony는 리뷰어의 이전 작업과 연관된 저자들을 찾는 Retrieval 작업으로 RAP를 재정의합니다. 세 가지 하위 집합인 exHarmony-Authors, exHarmony-Cite 및 exHarmony-SimCite를 생성하여 다양한 저자 데이터를 활용합니다. 또한, 전통적인 lexical matching, static neural embeddings 및 contextualized neural embeddings 기법을 포함한 여러 방법을 평가합니다.

- **Performance Highlights**: 주요 결과에 따르면, 전통적인 방법들은 합리적인 성능을 보였지만, 학술 문헌을 기반으로 훈련된 contextualized embeddings가 가장 좋은 성능을 나타냈습니다. 논문은 리뷰어 추천의 관련성과 다양성을 평가하기 위한 새로운 평가 지표를 제안하며, 다양성과 품질을 모두 고려한 리뷰어 할당의 중요성을 강조하고 있습니다.



### IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform (https://arxiv.org/abs/2502.07658)
Comments:
          Under review at KDD25 ADS. This work has already been deployed on the Xianyu platform in Alibaba. arXiv admin note: substantial text overlap with arXiv:2403.06747

- **What's New**: 이번 논문에서는 사용자-상품 상호작용에 의존하는 기존 추천 시스템의 단점을 극복하기 위해 두 단계의 추천 시스템인 IU4Rec를 제안합니다. 이 시스템은 상품을 다양한 속성을 바탕으로 클러스터로 그룹화하여 'Interest Units' (IU)를 생성하고, 이를 통해 사용자와 IU 간의 상호작용을 기록합니다. IU4Rec는 사용자의 광범위한 관심사를 수집하고 특정 상품 추천으로 이어지는 과정을 체계적으로 개선하고 있습니다.

- **Technical Details**: IU4Rec는 첫 번째 단계에서 사용자 관심사를 반영하는 Interest Units를 추천하고, 두 번째 단계에서는 사용자가 선택한 IU 내에서 최적의 상품을 찾아내는 방식으로 설계되었습니다. 이 과정에서 대규모 언어 모델을 통한 효과적인 IU 식별 방법을 활용하여, 판매가 완료된 후에도 사용자 상호작용을 누적하고 보존할 수 있는 전략을 취하고 있습니다. 또한, IU 기반의 추천 방식은 다양한 사용자 상호작용을 집계하여 추천의 정확성을 높이고 사용자 경험을 향상시키는 데 기여합니다.

- **Performance Highlights**: IU4Rec 시스템은 Alibaba의 Xianyu 플랫폼에서 실제 데이터셋과 온라인 A/B 테스트를 통해 효과성과 우수성을 입증했습니다. 새로운 두 단계 추천 패러다임은 사용자 관심 모델링의 정확성을 높이고 유사한 상품를 밀집해서 제공하여 효과적인 구매 결정을 도와줍니다. 이러한 방식은 C2C 전자상거래 플랫폼에서 사용자 상호작용을 효율적으로 수집할 수 있는 혁신적인 솔루션이라 할 수 있습니다.



### ETimeline: An Extensive Timeline Generation Dataset based on Large Language Mod (https://arxiv.org/abs/2502.07474)
- **What's New**: ETimeline은 13,878개의 이벤트 노드를 포함한 600개의 이중 언어 타임라인으로 구성된 새로운 데이터셋입니다. 이는 28개의 뉴스 분야에서 2020년 3월부터 2024년 4월까지의 인기 있는 인터넷 사건을 포괄합니다. 이 논문은 LLM(대형 언어 모델)을 기반으로 하는 파이프라인을 통해 타임라인 생성 과정의 성능을 향상시키는 방법을 소개합니다. ETimeline은 학계와 산업 간의 간극을 메우는 혁신적인 연구 촉진제 역할을 할 것으로 기대합니다.

- **Technical Details**: ETimeline의 데이터 구성 과정은 두 가지 주요 단계로 나뉩니다: 타임라인 주제 추출과 타임라인 노드 채우기입니다. LLM을 활용하여 Google Trends와 Baidu Hotsearch에서 트렌딩 키워드를 수집하고, 이를 바탕으로 적절한 주제를 생성합니다. 이후 온라인 뉴스 소스에서 수집한 문서를 기반으로 연대순으로 배열된 문서 풀을 구성하고, LLM이 각 문서를 적절한 주제의 노드에 배정하는 방식으로 진행됩니다.

- **Performance Highlights**: ETimeline은 다중 뉴스 도메인에서 발생하는 사건 간의 관계를 이해하고, 주제 생성을 위한 기초적인 데이터로 활용될 수 있는 잠재력을 지니고 있습니다. 이 데이터셋은 기존과 비교해 규모가 크고 다양성을 제공하여 타임라인 생성 작업에서 경쟁력 있는 대안을 제시합니다. 이 연구는 타임라인 생성 연구의 발전에 기여하며, 이야기를 이해하는 데 필요한 난이도를 높이는 것을 목표로 합니다.



### Generative Ghost: Investigating Ranking Bias Hidden in AI-Generated Videos (https://arxiv.org/abs/2502.07327)
- **What's New**: 이번 연구에서는 AI가 생성한 비디오(AI-generated videos)가 정보 검색 모델에 미치는 영향을 조사합니다. 연구진은 13,000개의 비디오로 구성된 포괄적인 기준 데이터셋을 구축하고, 비디오 검색 모델에서 AI가 생성한 콘텐츠에 대한 편향이 존재하는지를 분석하였습니다. 특히, 비디오 검색에서 발생하는 시각 및 시간적 요인에 따른 편향을 탐구합니다.

- **Technical Details**: 연구에서는 영상의 동적 특성과 생생함으로 인해 비디오 콘텐츠가 정보 전파 및 오락 매체로 주요하게 부각된다고 설명합니다. 비디오 검색 모델에 대한 평가를 위해, AI가 생성한 비디오와 실제 비디오 간의 의미적 유사성을 고려하면서도 성과를 객관적으로 평가하기 위한 다차원 메트릭스를 설계하였습니다. 최첨단 오픈소스 비디오 생성 모델 두 개를 사용하여 비디오를 생성하고, 자신의 기준 데이터셋에서 정보 검색 모델을 효율적으로 평가합니다.

- **Performance Highlights**: 비디오 검색 모델은 AI가 생성한 비디오를 선호하는 경향을 보였습니다. AI 생성 비디오의 비율이 높아질수록 검색 모델에서 해당 콘텐츠에 대한 편향이 더욱 심화되었습니다. 또한, 연구에서는 비디오 검색 모델의 편향을 완화하기 위해 대조 학습을 적용하여 실제 비디오를 우선 순위로 두었습니다.



### Prompt-Based Document Modifications In Ranking Competitions (https://arxiv.org/abs/2502.07315)
- **What's New**: 본 논문에서는 경쟁 검색 환경에서 문서의 랭킹을 개선하기 위해 대형 언어 모델(LLM)을 활용한 수정 방법을 연구하였습니다. 기존 연구에서 LLM을 랭커로 활용했던 사례를 바탕으로, 문서의 충실함(faithfulness)과 품질(content quality)을 유지하면서도 랭킹 향상을 이끌어내는 새로운 접근법을 제안합니다. 이 연구의 결과는 이전의 랭킹 대회 및 자율적으로 조직한 대회에서 유효성을 평가하여 입증되었습니다.

- **Technical Details**: 연구에서 제시된 문서 수정 방법은 LLM에 대한 두 가지 주요 프롬프트 체계를 포함합니다: 일반 공유 부분과 특정 맥락 부분입니다. 일반 공유 부분은 수정할 문서가 검색 엔진(rank)에서 높은 랭킹을 얻기 위한 지침을 제공하고, 특정 맥락 부분은 과거의 랭킹 정보를 통해 LLM에게 수정의 방향성을 제공합니다. 또한 다양한 맥락 유형(포인트, 페어, 리스트, 시간 순서)에 따른 프롬프트 구성도 고려하였습니다.

- **Performance Highlights**: 실험 설정은 랭킹 대회에서 수집한 데이터셋을 기반으로 하였으며, 두 가지 주요 데이터셋인 LambdaMARTComp와 E5Comp를 사용했습니다. 각 데이터셋에서는 LLM을 활용한 수정 방법이 성능 기반의 기존 문서 수정 접근법인 SentReplace보다 효과적으로 동작하는 것으로 나타났습니다. 이는 LLM 기본의 수정 방식이 경쟁력 있는 검색 환경에서의 문서 품질과 랭킹 향상에 효과적임을 입증하는 결과입니다.



### CreAgent: Towards Long-Term Evaluation of Recommender System under Platform-Creator Information Asymmetry (https://arxiv.org/abs/2502.07307)
- **What's New**: 이 논문에서는 추천 시스템(Recommendation Systems, RS)의 장기적인 지속 가능성을 보장하는 것에 대해 다룹니다. 기존의 오프라인 평가 방법들이 사용자의 즉각적인 피드백에 집중하는 반면, 콘텐츠 제작자의 장기적인 영향은 간과하는 경향이 있음을 지적했습니다. 이를 해결하기 위해 CreAgent라는 대형 언어 모델(Large Language Model, LLM) 기반의 제작자 시뮬레이션 에이전트를 제안합니다.

- **Technical Details**: CreAgent는 게임 이론의 믿음 메커니즘과 천천히 생각하는(fast-and-slow thinking) 프레임워크를 통합하여 정보 비대칭 조건에서 제작자 행동을 효과적으로 시뮬레이션합니다. 또한, Proximal Policy Optimization (PPO) 기법을 통해 CreAgent의 시뮬레이션 능력을 향상시켰습니다. 정보 비대칭 문제를 해결하고, 현실 세계의 플랫폼과 제작자 간의 행동을 잘 일치시키는지 검증하는 실험을 수행했습니다.

- **Performance Highlights**: CreAgent를 통한 RS 시뮬레이션을 통해 공정성(fairness) 및 다양성(diversity) 인식 RS 알고리즘이 다양한 이해관계자에게 장기적으로 더 나은 성과를 어떻게 기여하는지 탐구할 수 있습니다. 연구 결과 CreAgent는 RS 평가의 신뢰성을 크게 향상시켰으며, 이 시뮬레이션 플랫폼은 공개적으로 사용 가능하다고 합니다.



### Flow Matching for Collaborative Filtering (https://arxiv.org/abs/2502.07303)
- **What's New**: 이번 연구에서는 FlowCF라는 새로운 flow 기반 추천 시스템을 제안합니다. FlowCF는 협업 필터링의 독특한 도전에 대응하기 위해 두 가지 주요 혁신, 즉 행동 유도 prior와 이산 흐름 프레임워크를 도입하여 추천의 정확도를 향상시킵니다. 기존의 Generative CF 접근법에서는 나타나기 힘든 이 문제들을 잘 해결하는 방안을 제공합니다.

- **Technical Details**: FlowCF는 사용자 행동 패턴을 정렬하는 행동 유도 prior를 통해 희소하고 이질적인 사용자-아이템 상호작용을 다룰 수 있습니다. 또한 이산 흐름 프레임워크는 암묵적 피드백의 이진 성격을 유지하며, 지속적인 상태 공간에서 이산 상호작용을 효과적으로 모델링합니다. 이러한 접근 방식 덕분에 flow matching은 안정적인 훈련과 효율적인 샘플링의 장점을 유지합니다.

- **Performance Highlights**: FlowCF는 여러 데이터셋에서 이루어진 광범위한 실험을 통해 최신의 추천 정확도를 달성하였습니다. 또한, 빠른 추론 속도로 실제 추천 시스템의 요구 사항을 충족하며, Generative 모델 기반의 협업 필터링 접근 방식의 성능을 크게 향상시킵니다.



### DOGR: Leveraging Document-Oriented Contrastive Learning in Generative Retrieva (https://arxiv.org/abs/2502.07219)
- **What's New**: 이번 연구는 Generative retrieval에서 두 가지 단계로 구성된 새로운 구조인 DOGR(Leveraging Document-Oriented Contrastive Learning in Generative Retrieval)을 제안합니다. 이 프레임워크는 query와 document 간의 직접적인 상호작용을 통해 관계를 포괄적으로 이해합니다. 기존의 방법들은 identifiers에 대한 관계를 배우는 데 집중했으나, DOGR는 document의 의미적 표현을 향상시켜 이를 해결합니다.

- **Technical Details**: DOGR의 핵심은 identifier 생성 단계와 문서 순위 결정 단계입니다. 첫 번째 단계에서는 encoder-decoder 아키텍처를 사용하여 document identifier와 query 간의 관계를 학습합니다. 두 번째 단계에서, DOGR는 대조 학습(contrastive learning)으로 세밀한 문서 표현을 최적화하며, 여기에는 prefix-oriented negative sampling과 retrieval-augmented negative sampling이 포함됩니다.

- **Performance Highlights**: 실험 결과, DOGR는 두 개의 공개 벤치마크 데이터셋에서 기존의 generative retrieval 방법들에 비해 최첨단(state-of-the-art) 성능을 나타냈습니다. 두 단계의 학습 전략을 통해 DOGR는 query와 document 간의 관련성을 보다 정밀하게 모델링 하였으며, 여러 일반적인 identifier 구성 기법에 대해서도 효과적인 결과를 보였습니다.



### Repository-level Code Search with Neural Retrieval Methods (https://arxiv.org/abs/2502.07067)
Comments:
          16 pages

- **What's New**: 본 논문에서는 대규모 오픈 소스 저장소의 커밋 기록을 활용하여 버그 수정을 지원하는 다단계 리랭킹 시스템을 제안합니다. 저장소 수준의 코드 검색 작업을 정의하고, 사용자의 질문이나 버그에 가장 관련성 높은 파일 세트를 검색하는 것을 목표로 합니다. 제안하는 접근법은 BM25 기반 검색과 CodeBERT를 활용한 신경망 리랭킹을 결합하여 가장 적합한 파일을 식별합니다. 이 시스템은 커밋 메시지와 소스 코드를 활용해 관련성을 매칭하며, 실험을 통해 기존 BM25 기준보다 최대 80% 향상을 보여줍니다.

- **Technical Details**: 저장소 수준 코드 검색 작업은 주어진 사용자 쿼리 q에 대해 현재 코드 저장소 상태에서 가장 관련성 높은 파일 집합 ℱ을 검색하는 것으로 정의됩니다. 제안하는 접근법은 BM25 기반 시스템으로 시작하여 비슷한 메시지를 포함한 이전 커밋을 검색하고, 이를 통해 수정된 파일을 식별합니다. 이후 BERT 기반의 CommitReranker와 CodeReranker를 사용하여 후보 파일의 순위를 다시 매깁니다. 궁극적으로는 이 프로세스를 통해 LLM에 적합한 고도로 관련성 있는 파일 목록을 생성하는 것이 목표입니다.

- **Performance Highlights**: 실험에서 7개의 인기 있는 오픈 소스 저장소에서 생성한 데이터세트를 활용한 결과, BM25 기반의 모델 대비 MAP, MRR, P@1과 같은 성능 지표에서 최대 80% 향상이 나타났습니다. 본 연구는 GitHub와 같은 대형 소스 제어 시스템에서 LLM의 코드 검색 및 이해를 위한 툴로서 큰 기여를 할 것으로 기대됩니다. 얻어진 결과와 코드는 공개적으로 이용 가능하여, 후속 연구나 적용에 도움을 줄 수 있습니다.



### Solving the Content Gap in Roblox Game Recommendations: LLM-Based Profile Generation and Reranking (https://arxiv.org/abs/2502.06802)
- **What's New**: 이번 논문은 Roblox 플랫폼의 사용자 생성 콘텐츠를 분석하여 게임 추천 시스템의 품질을 개선하는 새로운 접근 방식을 제안합니다. 기존의 추천 시스템이 게임 콘텐츠의 불일치성과 희소성으로 어려움을 겪는 상황에서, 거대 언어 모델(LLMs)을 활용한 고품질 구조화된 텍스트 특징을 생성하고 검증하는 방법을 다룹니다. 이 연구는 개인화 및 사용자 만족도를 높이기 위한 LLM 기반의 재순위 메커니즘을 도입하며, 게임 속 텍스트 데이터 분석의 중요성을 강조합니다.

- **Technical Details**: 저자들은 텍스트 특징 생성을 위한 두 가지 주요 도전을 다루고 있습니다. 첫 번째로, 방대한 사용자 생성 콘텐츠에서 고품질의 구조화된 텍스트 특징을 생성하는 방법을 개발하고, 두 번째로 생성된 텍스트 특징이 추천 정확도를 향상시키는지를 검증하기 위한 프레임워크를 수립합니다. 이 과정에서는 양질의 게임 프로필 생성을 위한 LLM의 활용과 추천 시스템에서 텍스트 특징의 효용성을 평가하기 위한 재순위 기법이 포함됩니다.

- **Performance Highlights**: 제안된 방법론을 통해 Roblox의 역동적이고 사용자 중심의 에코시스템에 적합한 맞춤형 추천 시스템 구축이 가능함을 보여줍니다. LLM을 통한 텍스트 인식 및 프로필 생성이 설계되어 있어, 추천의 품질이 향상되고 사용자 경험이 증대될 것으로 기대됩니다. 이 연구는 플랫폼의 고유한 다이나믹에 적응한 스케일러블한 추천 시스템의 기초를 마련하고 있으며, 투명한 사용자 신뢰 구축에도 기여할 것입니다.



### Exploring Patterns Behind Sports (https://arxiv.org/abs/2502.07491)
- **What's New**: 이 논문은 ARIMA와 LSTM을 결합한 하이브리드 모델을 사용하여 시계열 예측을 위한 종합 프레임워크를 제시합니다. 특별한 특징 엔지니어링 기법을 통해 원시 데이터를 저차원 표현으로 변환하여 중요한 정보를 유지합니다. 이 모델은 과거 데이터를 기반으로 트레이닝되며, 낮은 RMSE와 MAE 점수로 표시되는 높은 예측 정확도를 달성합니다.

- **Technical Details**: 모델의 주요 구성 요소는 ARIMA를 사용하여 선형 트렌드를 포착하고, LSTM은 복잡한 비선형 의존성을 모델링합니다. 피처 임베딩과 PCA 기법을 활용하여 카테고리 데이터를 연속 벡터로 변환하고, 차원 축소를 통해 주요 구성 요소를 추출합니다. 이 연구는 데이터의 무작위성을 평가하기 위해 런 테스트(run test)를 사용하고, SHAP 방법을 이용해 전통적인 장점이 예측 결과에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 종합적인 결과는 전통적인 통계 방법과 현대의 딥러닝 기술을 결합하여 스포츠에서 견고한 시계열 예측을 제공하는 모델의 효과성을 강조합니다. 반복 신경망 LSTM을 사용하여 시계열 예측 모델을 수립하고, KNN 방법을 활용하여 최적의 예측 간격을 결정함으로써 모델의 정확도를 더욱 향상시킵니다. 이 연구는 각 국가의 메달 집계 예측 모델을 구축하고, '위대한 코치' 효과 분석을 통해 국가 올림픽 위원회에 대한 통찰력을 제공합니다.



### CTR-Driven Advertising Image Generation with Multimodal Large Language Models (https://arxiv.org/abs/2502.06823)
Comments:
          Accepted to WWW 2025

- **What's New**: 이번 논문의 새로운 점은 Click-Through Rate (CTR) 최적화를 주된 목표로 광고 이미지를 생성하는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 활용입니다. 기존의 기법들이 주로 미적 품질에 초점을 맞췄다면, 본 연구는 CTR을 극대화하기 위해 광고 이미지 생성 과정에서의 사용자 선호를 정밀하게 반영합니다. 또한, Reinforcement Learning (RL)과 Reward Model (RM)을 결합하여 생성된 이미지의 효과성을 향상시키는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구는 대규모 전자상거래 멀티모달 데이터셋을 활용하여 MLLMs의 성능을 향상시키기 위해 맞춤형 사전 훈련 과제를 설계했습니다. RL을 통해 잘 훈련된 RM을 사용하여 사용자 클릭 선호를 효과적으로 시뮬레이션할 수 있으며, 상품의 특성과 일치하는 콘텐츠 생성에 집중하는 Product-Centric Preference Optimization (PCPO) 전략을 도입합니다. 이런 기술적 접근은 다양한 멀티모달 특성의 통합과 시각적 데이터에 대한 정교한 처리 능력을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 공공 및 상업적 데이터셋 모두에서 최첨단 성능을 보여주었으며, 실제 전자상거래 환경에서 온라인 CTR을 상당히 개선하는 것으로 나타났습니다. 이는 모델이 생성한 광고 이미지의 품질뿐만 아니라 사용자의 예상 클릭 행동과의 높은 상관관계를 유지한다는 것을 의미합니다. 따라서 본 연구는 CTR 중심의 광고 이미지 생성에 있어 새로운 기준을 설정하며, 실제 광고 효과성을 높이는 데 기여합니다.



### The 2021 Tokyo Olympics Multilingual News Article Datas (https://arxiv.org/abs/2502.06648)
- **What's New**: 이 논문에서는 2021 도쿄 올림픽에 대한 다국어 뉴스 기사 데이터셋 OG2021을 소개합니다. 총 10,940개의 뉴스 기사가 1,918개의 출처로부터 수집되어, 다양한 언어로 작성되었습니다. 이 데이터셋은 여러 사건에 대한 보도 기사를 그룹화하는 데 도움을 주기 위해 개발되었습니다.

- **Technical Details**: 이 데이터셋은 온라인 클러스터링 알고리즘을 활용하여 같은 하위 이벤트에 대한 기사를 그룹화하고, 수작업으로 주석을 달아 평가되었습니다. 언어는 영어, 스페인어, 독일어, 프랑스어, 러시아어 등을 포함하며, 2021년 7월 1일부터 8월 14일까지 출판된 기사를 포함합니다.

- **Performance Highlights**: OG2021 데이터셋은 특히 고빈도 이벤트가 발생하는 상황에서의 다국어 뉴스 클러스터링 알고리즘 성능 평가에 적합합니다. 이는 도쿄 올림픽의 문화적 및 언어적 차이를 분석하는 데도 유용하게 사용될 수 있습니다.



### LiveForesighter: Generating Future Information for Live-Streaming Recommendations at Kuaishou (https://arxiv.org/abs/2502.06557)
Comments:
          Work in progress

- **What's New**: 본 논문은 실시간 스트리밍( live-streaming ) 추천 시스템에서의 새로운 접근 방식을 제시합니다. 특히 'LiveForesighter'라는 생성 모델을 개발하여, 실시간 스트리밍의 긍정적 행동 증가 추세를 모니터링하고 미래 콘텐츠 정보를 생성하는 방법을 다룹니다. 기여점으로는 이러한 두 가지 접근 방식을 결합하여 사용자에게 더욱 만족스러운 경험을 제공하는 점이 강조됩니다.

- **Technical Details**: 논문의 방법론 부분에서는 LiveForesighter 의 세부사항을 소개합니다. 이 모델은 통계적 시퀀스를 모델링하여 실시간 콘텐츠 품질을 모니터링하며, 이전에 판매된 제품 정보를 기반으로 미래 제품을 예측합니다. 특히, 추천 시스템은 본질적으로 사용자 피드백으로부터 수많은 항목 중 소수의 최적 항목을 생성하는 것을 목표로 합니다.

- **Performance Highlights**: Kuaishou 플랫폼에서 실제 사용자 기반의 실험을 통해 LiveForesighter 의 효과성이 입증되었습니다. 실험 결과, 이 모델이 실시간 스트리밍 추천의 질을 크게 향상시키며, 사용자에게 더 나은 경험을 제공하였음을 보여줍니다. 이러한 결과는 향후 다른 연구자들이 더욱 강력한 실시간 스트리밍 추천 시스템을 개발하는 데 기여할 것입니다.



### Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation (https://arxiv.org/abs/2502.06269)
- **What's New**: 최근 생성적 패러다임에 대한 관심이 급증함에 따라, 생성적 추천 시스템이 연구자들 사이에서 주목받고 있습니다. 본 논문에서는 기존 방법들이 협업(collaborative)이나 의미(semantic) 지식을 간과하거나 대략적으로 결합하는 경향이 있다는 점을 지적합니다. 이를 해결하기 위해, 의미 및 협업 지식을 통합하는 새로운 모델인 PRORec을 제안합니다.

- **Technical Details**: PRORec 모델은 두 단계의 프레임워크로 구성되어 있으며, 첫 번째 단계에서는 협업 임베딩(collaborative embedding)에 의미 지식을 통합하여 그 표현력을 향상시킵니다. 두 번째 단계에서는 각 모드에서 지식을 효과적으로 캡처하고 통합하기 위해 지식 증류(task) 기법을 도입합니다. 이로 인해 생성적 추천에서 협업과 의미 지식이 통합적으로 활용될 수 있습니다.

- **Performance Highlights**: 세 가지 일반적으로 사용되는 벤치마크에서의 광범위한 실험 결과, PRORec은 기존 방법보다 뛰어난 성능을 보였으며, 생성적 패러다임과 전통적 접근 방식 모두에서 우위를 입증했습니다. 또한, PRORec은 통합 코드의 도입으로 인해 더 빠른 추론 속도를 보여주었고, 저장 공간도 절약할 수 있음을 확인했습니다.



### Evaluating Entity Retrieval in Electronic Health Records: a Semantic Gap Perspectiv (https://arxiv.org/abs/2502.06252)
Comments:
          Under review, and the dataset will be made public upon reception of our paper

- **What's New**: 이번 논문은 전자 건강 기록(EHR)에서의 엔터티 검색(entity retrieval) 성능 평가를 위한 새로운 벤치마크인 CliniQ의 개발 및 공개를 제안합니다. 특히, 논문에서는 MIMIC-III 데이터셋의 퇴원 요약(discharge summaries)을 사용하여 ICD 코드와 처방 레이블을 질의(query)로 수집하고, GPT-4를 이용해 관련성 판단(relevance judgments)을 주석처리하였습니다. 1,000개의 환자 노트를 기반으로 1,246개의 질의를 생성하고 77,000개 이상의 관련성 주석을 제공하며, 이를 통해 단일 환자 검색(Single-Patient Retrieval)과 다중 환자 검색(Multi-Patient Retrieval)의 두 가지 설정에 대한 성능을 평가합니다.

- **Technical Details**: CliniQ는 고품질 주석 및 다양한 적용을 위한 두 가지 검색 설정을 포함하는 대규모 질의 집합을 특징으로 합니다. 본 연구는 또한 질의 매칭의 유형을 다섯 가지로 분류하는 새로운 시스템을 도입하였으며, 이는 문자열(string), 동의어(synonym), 약어(abbreviation), 하위어(hyponym), 그리고 함의(implication)입니다. GPT-4를 통해 의료 전문가와의 높은 일치를 나타내는 주석을 생성했으며, 이 주석들은 1,000개의 MIMIC-III 퇴원 요약을 통해 수집된 데이터입니다.

- **Performance Highlights**: BM25는 강력한 기준선(baseline)을 제공하지만, 의미론적 일치(semantic matches)에서 어려움을 겪고 있습니다. 질의 확장(query expansion)은 성능을 크게 향상시키지만, 문자열 일치 능력을 약간 감소시킵니다. 밀집 검색기(dense retrievers)는 전통적인 방법들을 초월하며, 특히 의미론적 일치에서 뛰어난 성능을 보여주고 있습니다.



### RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning (https://arxiv.org/abs/2502.06101)
Comments:
          Accepted by TheWebConf'25 (WWW'25) as a Short Paper

- **What's New**: 최근 대형 언어 모델(LLM)이 추천 시스템에 통합되어 사용자 행동 이해를 향상시키고 있습니다. 본 논문에서는 Retrieval Augmented Generation(RAG) 기법을 통해 더욱 관련성 높은 아이템을 검색하고 시스템의 성능을 개선하고자 하였습니다. 기존 RAG 방법론은 주로 텍스트 의미에 의존하며, 가장 관련성 높은 아이템을 포함하는 데 한계가 있습니다. 이를 개선하기 위해 우리는 LLM을 사용하여 아이템에 대한 자세한 설명을 생성하고, 이를 기반으로 공동 표현 학습을 수행합니다.

- **Technical Details**: 우리가 제안하는 RALLRec은 텍스트 의미와 협업 의미를 향상시키기 위해 LLM이 아이템의 상세한 설명을 생성하도록 유도합니다. 이 생성된 설명을 통해 향상된 아이템 표현을 추출하고, 이를 추천 모델을 통해 협업 의미와 결합하여 최종 표현을 만듭니다. 또한, 사용자 선호의 동적 특징을 고려하여 효과적인 재정렬 방법을 도입하여 추천의 유효성을 더욱 높였습니다.

- **Performance Highlights**: 세 가지 실제 데이터 세트에서 광범위한 실험을 수행한 결과, 우리의 방법이 효과적임을 입증하였습니다. 제안하는 RALLRec은 관련 아이템을 효과적으로 검색하고, 사용자 선호를 반영한 다양한 재정렬 전략을 통해 성능을 향상시켰습니다. 실험 결과는 제안된 방법의 유효성을 뒷받침하며, 향후 추천 시스템의 발전에 기여할 것으로 기대됩니다.



### NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems (https://arxiv.org/abs/2502.06097)
Comments:
          Accepted by WWW 2025 Industry Track

- **What's New**: 이번 논문에서는 Neighbor Lists 모델을 활용한 Generative Reranking(NLGR)을 제안합니다. NLGR은 기존의 evaluator-generator 패러다임을 따르면서, generator의 성능을 개선하여 조합 공간에서의 최적화를 목표로 합니다. 이 모델은 neighbor lists를 사용하여 generator의 훈련 과정을 향상시키고, 새로운 샘플링 기반 비자기회 생성 방법을 도입하여 최적 리스트 찾기를 쉽게 합니다.

- **Technical Details**: NLGR 모델은 generator와 evaluator 간의 목표 불일치 문제를 해결하기 위해 Neighbor Lists를 통합합니다. 훈련 과정에서 상대 점수를 인식할 수 있도록 neighbor lists를 활용함으로써 generator가 최적의 방향을 식별하도록 합니다. 또한 Position Decision Unit(PDU)과 Candidate Retrieval Unit(CRU)를 사용하여 조합 공간을 유연하게 탐색하고 최적 리스트로의 전환을 가능하게 합니다.

- **Performance Highlights**: NLGR의 효과성은 공공 및 산업 데이터셋에서의 광범위한 실험을 통해 검증되었습니다. Meituan 음식 배달 플랫폼에 성공적으로 배치되어 다양한 성과 지표에서 현저한 개선을 달성하였습니다. 논문에서 제안한 방법들이 기존 방법들에 비해 더 우수한 성능을 나타내는 것을 확인할 수 있었습니다.



### FactIR: A Real-World Zero-shot Open-Domain Retrieval Benchmark for Fact-Checking (https://arxiv.org/abs/2502.06006)
Comments:
          Accepted to WWW 2025 resource track

- **What's New**: 이 논문에서는 자동 사실 확인을 위한 새로운 실제 기반 검색 벤치마크인 FactIR을 제시합니다. 기존의 사실 확인 방법들이 종종 간단한 정보를 다루는 데 한계가 있었던 반면, FactIR은 복잡한 주장에 대한 간접적 증거를 수집하는 데 중점을 두고 있습니다. 이 벤치마크는 Factiverse 프로덕션 로그에서 파생된 데이터와 인간 주석을 포함해, 현실적이고 복잡한 사실 확인 환경을 모사합니다.

- **Technical Details**: FactIR 데이터셋은 여러 검색 엔진에서 수집된 증거와 전문가의 피드백을 통해 생성되었습니다. 이 데이터셋은 1413개의 주장-증거 쌍과 90047개의 문서를 포함하고 있으며, 각 요청 당 평균 13.89개의 문서가 평가되었습니다. 벤치마크는 다양한 정보 출처를 처리하고, 부분적으로 관련된 증거 및 섬세한 추론이 필요한 주장을 검토하는 데 도움을 줍니다.

- **Performance Highlights**: 최신 검색 모델들을 평가한 결과, 기존의 벤치마크에서는 발견하지 못한 복잡한 주장을 처리하는 데 있어 일부 모델이 다른 모델보다 성능이 뛰어난 것으로 나타났습니다. 실험은 0-샷 설정(Zero-shot setting)에서 진행되어, 현실 세계의 사용 사례와 일치하는 성능 평가가 이루어졌습니다. 논문에서는 이러한 모델이 실제 환경에서도 잘 작동할 수 있도록 설계되어야 한다는 점을 강조하며, 사실 확인 시스템의 발전을 지원하고자 합니다.



### Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education (https://arxiv.org/abs/2502.05863)
- **What's New**:  본 논문에서는 AI 중심 교육(AI-facilitated teaching)에서 다양한 query 스타일을 활용하여 추상 텍스트 설명을 해석하는 새로운 접근 방식을 제안합니다. 이를 위해 STEM 교육에 특화된 다양한 표현 검색 작업을 통해 여러 쿼리 스타일과 표현을 지원하는 새로운 STEM Education Retrieval Dataset (SER)을 소개합니다. 또한, prompt tuning 기반의 스타일 다양화를 위해 Uni-Retrieval 모델이 개발되어, 이는 쿼리 스타일 특징을 프로토타입으로 추출하고 효율적인 리트리벌 기능을 제공합니다.

- **Technical Details**:  Uni-Retrieval 모델은 쿼리 기반 검색을 위한 포괄적인 문제 정의를 기반으로 하며, 여러 스타일의 쿼리를 수용하기 위해 다양한 입력 유형(텍스트, 오디오, 이미지 등)을 고려합니다. SER 데이터셋은 24,000개의 텍스트 캡션, 오디오 클립 및 다양한 스타일의 쿼리로 구성되어 있으며, 20명의 대학원생에 의해 구성된 데이터셋입니다. 특히, Prompt Bank라는 혁신적인 데이터 표현 구조를 사용하여 교육 시나리오에 효율적으로 적합한 모델을 발전시킵니다.

- **Performance Highlights**:  실험 결과, Uni-Retrieval은 기존의 검색 모델에 비해 대부분의 검색 작업에서 월등한 성능을 보입니다. 이 모델은 다중 쿼리 스타일에 대해 동적으로 프롬프트 토큰을 검색하여 다양한 교육적 요구에 맞는 확장 가능하고 정밀한 솔루션을 제공합니다. 특히, Uni-Retrieval은 제한된 매개변수 증가로도 유의미한 성능 향상을 이끌어내며, STEM 교육 커뮤니티에 큰 잠재력을 제공합니다.



### HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads (https://arxiv.org/abs/2502.05822)
Comments:
          Accepted by WWW 2025 (Industry Track)

- **What's New**: 이 논문에서는 쇼트 비디오 광고의 검색 광고에 대한 질의-비디오 적합성(query-to-video relevance matching)을 개선하는 데 초점을 맞추고 있습니다. 새로운 방법론으로는 고일관성(multimodal relevance model: HCMRM)을 제안하여, 사전 훈련(pre-training)과 적합성 작업(relevance tasks)의 일관성을 향상시키고자 합니다. 이를 통해 광고 시스템의 순위 매기기 기능을 대폭 강화할 수 있습니다.

- **Technical Details**: HCMRM은 비디오 텍스트에서 몇 가지 키워드를 추출하고, 이를 사용하여 트리플렛(triplet: 질의, 비주얼 신호, 비디오 텍스트) 적합성 모델링을 수행합니다. 사전 훈련 동안 일반적인 작업으로는 대조 학습(contrastive learning), 이미지-텍스트 매칭(image-text matching) 및 마스킹된 언어 모델링(masked language modeling)이 사용됩니다. 또한, 다중 계층 소프트맥스 손실(symmetric hierarchical softmax loss)을 도입하여 레이블 간의 순서를 학습합니다.

- **Performance Highlights**: HCMRM은 Kuaishou 검색 광고 시스템에 1년 이상 배포되어, 관련 없는 광고 비율을 6.1% 감소시키고 광고 수익을 1.4% 증가시키는 성과를 거두었습니다. 실험 결과, HCMRM은 AUC 및 스피어만 순위 상관 계수(Spearman’s rank correlation coefficient) 측정치에서 이전 모델들과 다양한 기준 모델들보다 뛰어난 성능을 보였습니다.



### FlashCheck: Exploration of Efficient Evidence Retrieval for Fast Fact-Checking (https://arxiv.org/abs/2502.05803)
Comments:
          Accepted to ECIR 2024, 15 pages

- **What's New**: 이번 연구는 대량의 데이터 수집에서의 증거 검색(evidence retrieval) 문제를 해결하기 위해 위키백과와 같은 대규모 지식 소스에서 간결한 사실 진술문의 인덱싱(indexing) 방법을 탐구합니다. 특히, 기존의 작업은 사실 검증(fact-verification) 부분에 초점을 맞추고 있었지만, 본 연구는 자동화된 사실 확인 파이프라인의 검색 단계를 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서 벡터 양자화(vector quantization) 기법을 적용해 밀도 검색(dense retrieval) 접근 방식을 사용하는 파이프라인의 효율성을 극대화하는 방법을 논의합니다. 본 연구는 HoVer와 WiCE와 같은 사실 확인 데이터 세트를 활용하여 다양한 접근 방법의 효율성과 효과성을 분석하고 위키백과를 지식 소스로 사용합니다.

- **Performance Highlights**: 효율적인 검색 접근 방식을 통해 2024년 대통령 토론회에서의 사실 확인을 수행했으며, 관련 라벨이 첨부된 주장 클레임(claims) 데이터 셋도 오픈 소스로 제공합니다. 인덱스된 사실들, 밀도 검색 방식, 인덱스 압축(index compression)을 결합하여 클래식한 사실 확인 파이프라인에 비해 CPU에서 최대 10배, GPU에서 20배 이상의 속도 향상을 달성했습니다.



### Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Ar (https://arxiv.org/abs/2502.05575)
- **What's New**: 본 연구는 벡터 검색(vector search)의 최신 발전 상황을 다루며, 특히 다양한 그래프 기반 방법들을 간략히 조사하여 다섯 가지 주요 설계 패러다임(Design Paradigms)으로 분류합니다. 이러한 연구는 빅 데이터 환경에서 벡터 분석의 복잡성을 해소하고, 다양한 비즈니스와 과학적 응용 분야에 기여할 수 있는 가능성을 보여줍니다. 특히, 반복적인 삽입(incremental insertion) 및 이웃 다양화(neighborhood diversification)를 기반으로 한 접근법들이 성능에서 우수하다는 점을 확인하였습니다.

- **Technical Details**: 연구에서는 12개의 최신 방법을 10억 개 벡터를 포함하는 7가지 실제 데이터 컬렉션을 사용하여 실험적으로 평가하였습니다. 자가 적응(seed selection) 및 다양화(diversification) 전략을 개선할 필요성과 함께 기존 그래프 기반의 벡터 검색 기술들의 강점과 한계에 대한 깊은 통찰이 제공됩니다. 특히 이웃 전파(neighborhood propagation) 및 분할 정복(divide-and-conquer) 같은 전략이 효과적으로 사용됩니다.

- **Performance Highlights**: 실험 결과, ELPIS 알고리즘이 기존의 시리얼 스캔(serial scan)이나 다른 방법들보다 최대 3배 이상 빠른 성과를 보였습니다. 이는 그래프 기반 벡터 검색이 실제 애플리케이션에서 얼마나 중요한지를 강조하며, 여러 비즈니스 환경에서 그 적용 가능성을 더욱 부각시키고 있습니다. 또한, 연구는 향후 더 정교한 데이터 적응형(seed-selection) 및 다양화 전략의 필요성에 대해서도 논의하고 있습니다.



### Diffusion Model for Interest Refinement in Multi-Interest Recommendation (https://arxiv.org/abs/2502.05561)
- **What's New**: 본 연구는 개인화 추천 시스템에서 사용자 관심사를 정교하게 표현하는 새로운 방법인 Diffusion Multi-Interest model (DMI)을 제안합니다. 기존의 방법들이 관심사를 아이템 수준에서만 고려하는 데 반해, DMI는 차원 수준에서의 관심사 표상을 강화하여 더 정밀한 추천을 가능하게 합니다. 이러한 접근은 과거 행동에서의 다양한 사용자 관심사를 더 효과적으로 포착할 수 있도록 합니다.

- **Technical Details**: DMI는 차원 수준에서 거칠게 표현된 관심사에 제어 가능한 노이즈를 추가하여 시작합니다. 이후, DMI는 iterative reconstruction 과정을 통해 cross-attention mechanism과 item pruning strategy를 통합하여 개인화된 관심 벡터를 재구성합니다. 이 과정에서 맞춤형 협업 정보를 이용하여 정밀한 관심사 표현이 가능해집니다.

- **Performance Highlights**: DMI는 광범위한 실험 결과를 통해 기존의 최첨단 방법들을 초월하는 성능을 입증하였습니다. 오프라인 평가와 온라인 A/B 테스트 모두에서 효과적이며, 실제 추천 시스템에 성공적으로 배포되어 수억 명의 일일 활성 사용자를 대상으로 사용자 만족도와 시스템 성능을 향상시킵니다.



### Large Memory Network for Recommendation (https://arxiv.org/abs/2502.05558)
- **What's New**: 이번 논문에서는 사용자 행동 시퀀스를 효과적으로 모델링하기 위한 새로운 접근법인 Large Memory Network (LMN)를 제안합니다. LMN은 대규모 메모리 블록 내에서 사용자 이력 행동 정보를 압축하고 저장하여 사용자 맞춤형 추천의 정확성을 높입니다. 기존의 복잡한 모델링 방식이 가지고 있는 한계를 극복하며, 메모리 블록을 통해 사용자 간의 관심사를 공유할 수 있는 기회를 제공합니다.

- **Technical Details**: LMN 구조는 메모리 블록에서 사용자-아이템 간의 상호작용을 기반으로 한 click-through rate (CTR) 예측에 중점을 두고 있습니다. 이 모델은 대규모 메모리 블록을 구성하고 상관관계가 높은 메모리 슬롯 상위 K개를 찾는 것을 목표로 합니다. 또한, 메모리 키와 값은 분리되어 있고, 메모리 활성화 메커니즘은 product quantization을 기반으로 설계되어 연산 비용을 최소화합니다.

- **Performance Highlights**: 다양한 오프라인 실험과 온라인 A/B 테스트를 통해 LMN의 우수한 성능이 입증되었습니다. Douyin E-Commerce Search에 적용된 LMN은 매일 수백만 명의 사용자에게 서비스를 제공하고 있으며, 사용자 만족도와 추천 정확성을 동시에 향상시켰습니다. 이러한 결과는 LMN이 산업에서의 활용 가능성이 높음을 보여줍니다.



### Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders (https://arxiv.org/abs/2502.05523)
- **What's New**: 이번 논문에서는 복잡한 사용자 행동 패턴과 다양한 의도가 존재하는 다중 비즈니스 시나리오에서의 추천 시스템 문제를 다루고 있습니다. 특히, Adaptive Domain Scaling (ADS) 모델을 제안하며, 이는 사용자 행동 시퀀스의 개인화를 위한 두 가지 주요 모듈인 PSRG(개인화된 시퀀스 표현 생성을 위한 모듈)과 PCRG(개인화된 후보 표현 생성을 위한 모듈)를 포함합니다. ADS는 다양한 도메인에서의 사용자 의도 이해를 돕고자 하며, 기존의 표준 임베딩 방법론을 대체할 수 있습니다.

- **Technical Details**: 제안된 ADS 모델은 개인화된 시퀀스 표현을 생성하여 사용자 행동을 이해하는 데 중점을 두며, 도메인 관련 정보를 입력으로 받아 다양한 도메인 간의 영향을 더욱 효과적으로 반영합니다. PSRG 모듈에서는 사용자 행동 내 다중 도메인 아이템 표현을 학습하기 위한 새로운 구조를 설계하였고, PCRG 모듈은 후보 아이템의 개인화된 표현을 생성하여 사용자 의도의 이해를 향상시킵니다. ADS는 기존 추천 시스템에 쉽게 통합될 수 있는 효율적인 플러그 앤 플레이 네트워크 구조입니다.

- **Performance Highlights**: 실험 결과는 ADS의 높은 효과성과 호환성을 입증하며, 공공 데이터셋과 20억 규모의 산업 데이터셋에서 유의미한 결과를 도출하였습니다. 더불어, Douyin의 광고 플랫폼과 전자상거래 서비스 플랫폼에서의 온라인 실험 결과, 각각 1.09%와 0.79%의 수익 상승을 기록하였습니다. 현재 ADS는 ByteDance의 여러 추천 서비스에 완전히 배포되어 있으며, 수십억 사용자를 대상으로 운영되고 있습니다.



### Hypencoder: Hypernetworks for Information Retrieva (https://arxiv.org/abs/2502.05364)
- **What's New**: 이 논문은 기존의 문서 검색 모델들이 벡터 내적(vector inner product)에 의존하여 질의(query)와 문서(document) 간의 관련성 점수를 계산하는 한계에 대한 혁신적인 해결책을 제시합니다. 새로운 패러다임으로서, 저자들은 벡터 대신 소형 신경망(neural network)을 이용하여 학습된 관련성 함수(learned relevance function)를 생성하는 Hypencoder를 제안합니다. 이 Hypencoder는 문서 표현을 입력으로 받아 스칼라 형태의 관련성 점수를 산출하며, 강력한 밀집 검색 모델(dense retrieval models)보다 훨씬 뛰어난 성능을 보여줍니다.

- **Technical Details**: Hypencoder는 하이퍼 네트워크(hypernetwork)를 활용하여 쿼리에 의존적인 다층 신경망(multi-layer neural network)을 학습합니다. 이러한 네트워크는 문서 표현에 적용되어 관련성 점수를 생성하며, 고성능의 검색을 위하여 그래프 기반의 탐색 알고리즘을 사용합니다. 이를 통해 약 8.8M 문서로부터 60ms 이내에 검색할 수 있는 근사 검색 알고리즘도 구현되었습니다.

- **Performance Highlights**: Hypencoder는 다양한 데이터 세트에서 경쟁력 있는 단일 벡터 밀집 및 희소 검색 모델보다 우수한 성능을 입증하였으며, TREC DL-Hard와 같은 복잡한 검색 작업에서도 뛰어난 성과를 보였습니다. 또한, 도메인 적응(domain adaptation) 설정 하에서도 고성능을 유지하며, 의료 및 금융 데이터셋과 같은 질문 응답 데이터셋에서도 뛰어난 결과를 나타냈습니다. 이 성능 향상이 Hypencoder가 관련성 함수를 재정립하는 데 중요한 기여를 한다고 저자들은 주장합니다.



### RSAttAE: An Information-Aware Attention-based Autoencoder Recommender System (https://arxiv.org/abs/2502.06705)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 연구는 사용자와 영화의 미지의 평점을 예측하기 위해 Attention 기반 Autoencoder(RSAttAE)를 활용한 새로운 추천 시스템 방법론을 제안합니다. 이 방법은 MovieLens 100K 데이터셋을 사용하며, 기존의 최첨단 방법보다 우수한 성능을 보입니다. 이 연구는 추천 시스템의 고객 만족도를 향상시키기 위한 기초적인 접근을 제공합니다.

- **Technical Details**: 제안된 방법은 전처리, 인코딩, 디코딩의 세 가지 주요 단계로 구성됩니다. 전처리 단계에서는 사용자와 영화의 다양한 특징을 분류하고, 인코딩 단계에서는 RSAttAE 모듈을 사용하여 유의미한 표현을 생성합니다. 마지막으로, 디코딩 단계에서는 평점 예측을 집중적으로 수행합니다. 사용자와 영화의 피처를 카테고리화하여 노이즈를 줄이고, 의미 있는 피처 벡터를 일관되게 생성합니다.

- **Performance Highlights**: 실험 결과는 제안된 RSAttAE 방법이 평점 예측과 매트릭스 완성 과제에서 효과적이고 강건함을 입증하였음을 보여줍니다. 기존 추천 시스템 방법론과 비교했을 때, 제안된 방법이 더 나은 성능을 보였으며, 이는 추천 시스템의 품질을 높일 수 있는 큰 잠재력을 지니고 있음을 시사합니다.



### FunduSAM: A Specialized Deep Learning Model for Enhanced Optic Disc and Cup Segmentation in Fundus Images (https://arxiv.org/abs/2502.06220)
- **What's New**: 이번 연구에서 제안하는 FunduSAM은 optic disc (OD) 및 optic cup (OC) 분할 작업에 특화된 딥러닝 모델로, 기존의 Segment Anything Model (SAM)에 여러 가지 Adapter를 추가하여 구조화하여 성능을 개선합니다. 이 모델은 Convolutional Block Attention Module (CBAM)을 통합하여 흐릿한 경계 및 낮은 대비 문제를 해결하며, polar transformation을 통해 fundus 이미지를 최적의 형식으로 변환합니다. 또한, OD와 OC 간의 구조적 보존을 위해 공동 손실(joint loss)을 사용하는 것이 특징입니다.

- **Technical Details**: FunduSAM은 SAM 구조를 기반으로 하여 fundus 이미지 처리를 위한 이미지 인코더와 Adapter 레이어를 개선하였습니다. 특히, polar transformation을 통해 OD 및 OC의 비율을 조정하고, Transformer 블록 내에 두 개의 Adapter 레이어를 설정하여 로컬화된 파라미터 효율적 미세 조정(PEFT)을 수행합니다. 이미지 인코더는 16개의 Transformer 블록으로 구성되어 있으며, 각 블록은 지역 정보 및 전반적인 이미지 맥락을 캡처하는 전역 주의력(global attention) 및 창(window) 기반 주의력 메커니즘을 활용합니다.

- **Performance Highlights**: REFUGE 데이터셋에서 1,200개의 fundus 이미지를 사용한 실험 결과, FunduSAM은 다섯 가지 주류 접근 방식에 비해 우수한 성능을 입증했습니다. 특히, FunduSAM은 정밀한 분할을 가능하게 하여 OD와 OC 간의 구조적 관계를 효과적으로 유지하며, 임상 진단의 정확도를 높이는 데 기여할 것으로 보입니다. 이러한 성과는 FunduSAM의 혁신적인 아키텍처와 협력적인 손실 함수가 결합된 결과입니다.



### Optimizing Knowledge Integration in Retrieval-Augmented Generation with Self-Selection (https://arxiv.org/abs/2502.06148)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문은 Self-Selection RAG(자기 선택 검색 증강 생성) 프레임워크를 제안하여 대형 언어 모델(LLM)이 외부에서 검색된 지식과 내부 파라메트릭 지식을 효과적으로 통합해 보다 정확한 결과를 생성하도록 돕습니다. 기존의 RAG 접근 방식의 한계를 극복하기 위해, LLM이 생성한 두 개의 응답을 비교하여 더 정확한 응답을 선택하도록 훈련합니다. 이 과정에서 직접 선호 최적화(Direct Preference Optimization, DPO) 기법을 활용하여 성능을 향상시킵니다.

- **Technical Details**: Self-Selection RAG 프레임워크는 LLM이 LLM 응답과 RAG 응답을 비교하고, 각각의 설명을 고려하여 올바른 답변을 선택할 수 있게 돕는 방법론입니다. LLM의 성능을 강화하기 위해, 새로운 Retrieval-Generation Preference(RGP) 데이터세트를 사용하여 LLM을 세밀하게 조정합니다. 이를 통해 외부에서 검색된 지식과 내부 지식을 결합하여 더욱 정확한 응답을 생성할 수 있습니다.

- **Performance Highlights**: 자체 선택 RGP 방법론은 Mistral-7B와 LLaMa-2-13B-Chat과 같은 두 개의 오픈 소스 LLM에서 테스트되었고, Natural Questions(NQ)와 TrivialQA 데이터세트에서 우수한 성능을 보여주었습니다. 실험 결과, 제안된 방법이 다양한 검색 설정과 LLM에서 높은 효과성을 일관되게 달성하며 RAG 시스템의 강건성과 안정성을 향상시킴을 입증합니다. 추가 실험을 통해 이 방법이 잡음이 많은 응답에서 유효한 답변을 구별하는 능력과 응답 생성 능력을 모두 향상시킨다는 사실도 확인되었습니다.



### Benchmarking Prompt Sensitivity in Large Language Models (https://arxiv.org/abs/2502.06065)
- **What's New**: 이번 논문에서는 Prompt Sensitivity Prediction이라는 새로운 작업을 소개하고, LLM의 응답 정확성에 미치는 프롬프트 변형의 영향을 조사하기 위해 PromptSET이라는 데이터셋을 설계했습니다. 주요 목적은 LLM의 프롬프트 반응 능력에 대한 예측을 통해, 프롬프트의 미세한 변형이 LLM 성능에 미치는 영향을 분석하는 것입니다. 이를 통해 효과적인 프롬프트 설계의 필요성을 강조하고 있습니다.

- **Technical Details**: 제안된 Prompt Sensitivity Prediction 작업은 주어진 프롬프트가 LLM에 의해 효과적으로 수행될 수 있는지를 예측하는 것을 목표로 합니다. 각 프롬프트는 특정 정보 요구(Ip)에 따라 약간 수정된 버전으로 구성됩니다. 데이터셋은 TriviaQA와 HotpotQA에서 출발하여 생성된 다양한 프롬프트 변형으로 구성되며, 이 변형의 유사성 및 정보 요구의 일관성을 기준으로 평가됩니다.

- **Performance Highlights**: 기존의 텍스트 분류(TC) 및 질의 성능 예측(QPP)과의 유사성을 기반으로 프롬프트 민감도 예측 작업을 벤치마크하는 실험을 수행했습니다. 연구 결과, 기존 방법들이 이 새로운 작업을 효과적으로 해결하지 못함을 보여주었으며, 이는 프롬프트 민감도 예측을 위한 새로운 접근 방법의 필요성을 강조합니다.



### Multi-Branch Collaborative Learning Network for Video Quality Assessment in Industrial Video Search (https://arxiv.org/abs/2502.05924)
Comments:
          KDD 2025 ADS

- **What's New**: 이번 논문은 산업 비디오 검색 시스템을 위한 Multi-Branch Collaborative Network (MBCN)을 소개합니다. MBCN은 비디오 품질 문제를 해결하기 위해 네 개의 분기를 가지고 있으며, 각 분기는 비디오의 시각적, 텍스트 기반, 및 AI 생성 비디오와 관련된 품질 문제를 다룹니다. 이 연구는 저품질 비디오의 특정 특성을 체계적으로 분석하고 MBCN을 통해 이를 해결하려는 시도를 보여줍니다.

- **Technical Details**: MBCN은 비디오 및 텍스트 품질 평가를 위해 다중 모달 인코더와 네 개의 평가 분기를 사용하는 아키텍처를 가지고 있습니다. 각 분기는 비디오-텍스트 매칭, 프레임 일관성, 프레임 품질, 및 텍스트 품질 평가를 담당합니다. Squeeze-and-Excitation 기법을 사용하여 여러 가지 품질 문제를 동적으로 해결하고, 최적화 목표를 통해 출력 점수의 안정성과 합리성을 확보합니다.

- **Performance Highlights**: 광범위한 오프라인 및 온라인 실험 결과, MBCN은 비디오 품질 문제를 효과적으로 식별하며 검색 시스템의 랭킹 성능을 크게 향상시킵니다. 특히 저품질 AI 생성 비디오의 인식 정확도가 기존 모델에 비해 현저히 개선되었습니다. 이번 연구는 비디오 품질 평가 분야에서의 새로운 통찰력을 제공하며, MBCN의 설계가 모든 네 개의 평가 분기가 최종 결과에 긍정적인 기여를 한다는 점을 강조합니다.



### LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification (https://arxiv.org/abs/2502.05836)
Comments:
          Accepted on NAACL 2025

- **What's New**: 이 논문에서는 인도 법원의 판결을 중심으로 합법적인 문서의 의미적 세분화 및 수사학적 역할 분류 과제를 다루고 있습니다. LegalSeg라는 7,000개 이상의 문서와 140만 문장으로 구성된 대규모 주석 데이터셋을 새롭게 소개하며, 이는 법적 문서 처리를 위한 가장 큰 데이터셋입니다. 여러 최첨단 모델(예: Hierarchical BiLSTM-CRF, ToInLegalBERT, GNN 등)의 성능을 평가하고, 문맥과 구조적 관계를 통해 분류 정확도를 높였음을 보여줍니다.

- **Technical Details**: 논문에서는 법적 문서에서의 의미적 세분화를 위한 다양한 모델을 구현하여 평가하였습니다. Hierarchical BiLSTM-CRF 모델은 계층적 접근을 통해 문맥 정보를 캡처하고, MultiTask Learning은 역할 전환을 고려하여 수사학적 역할을 보다 정교하게 식별합니다. 이외에도 InLegalToBERT, Graph Neural Networks(GNNs), Role-Aware Transformers와 같은 새로운 접근 방식을 도입하여 모델의 표현력과 문맥 처리를 향상시켰습니다. 특히, GNN은 문장을 노드로 표현하여 정보 전파와 문맥 캡처를 효과적으로 수행합니다.

- **Performance Highlights**: 법적 문서의 세분화 및 이해도 향상에 대한 기여로 LegalSeg 데이터셋과 연구 결과는 법적 NLP 분야에서 중요한 기초를 제공합니다. 특히, RhetoricLLaMA를 통해 복잡한 법적 언어 처리를 위한 인스트럭션 튜닝된 대형 언어 모델의 가능성과 한계를 강조하였습니다. 모델 성능 평가 결과, 문맥이나 주변 문장의 레이블 활용이 분류 정확도에 긍정적인 영향을 미친 것으로 나타났습니다.



### A Tutorial On Intersectionality in Fair Rankings (https://arxiv.org/abs/2502.05333)
- **What's New**: 이 논문은 편향된 알고리즘과 불공정한 순위 시스템이 다양한 분야에서 문제가 되고 있음을 다룹니다. 특히, 이러한 편향이 소외된 집단에 미치는 영향을 강조하며, intersectionality(교차성) 개념을 통해 공정성을 보장할 필요성을 제시합니다. 이 연구는 공정한 순위를 위한 알고리즘 개발 시 교차성과 기존 불평등의 지속을 방지하는 방법을 탐구합니다.

- **Technical Details**: 논문은 알고리즘의 공정성을 높이기 위해 기존의 공정성 인식 순위 기술들이 단일 보호 속성만을 고려하는 경향이 있음을 지적합니다. 따라서 교차성의 중요성을 강조하며, 다중 사회 정체성의 상호작용이 공정성에 미치는 영향을 분석합니다. 예를 들어, 인종과 성별의 교차점을 고려하지 않았을 때 발생할 수 있는 불공정성을 다양한 사례를 통해 설명합니다.

- **Performance Highlights**: 논문에서는 교차성을 포함한 공정한 순위 알고리즘이 실용적으로 구현 가능하며, 상당한 효용 손실 없이 도입할 수 있는 방법을 제시합니다. 또한, 다양한 방법론을 비교하는 시각적 테이블을 통해 각 방법의 특징과 장단점을 요약하여 연구자와 실무자들이 적절한 방법을 선택하는 데 도움을 줍니다. 교차성을 고려하는 것이 공정성을 달성하는 데 필수적이라는 결론을 내리고, 이를 통해 데이터 기반 시스템의 윤리를 증진할 수 있는 가능성을 열어줍니다.



### Efficient Knowledge Feeding to Language Models: A Novel Integrated Encoder-Decoder Architectur (https://arxiv.org/abs/2502.05233)
Comments:
          Submitted to ACM TIST journal: under revision stage, 8 pages, 2 figures

- **What's New**: 이 논문은 언어 모델(LLM)에 지식을 효과적으로 공급하는 새로운 접근 방식을 소개합니다. 기존의 Retrieval-Augmented Generation (RAG) 모델이 가지는 여러 한계를 극복하기 위해 in-context vectors (ICV)를 도입합니다. ICV는 LLM의 잠재 임베딩을 활용하여 작업 정보의 본질을 캡처하는 벡터를 생성합니다. 이 벡터는 LLM의 잠재 상태를 이동시키는 데 사용되어 추가적인 예시를 포함하지 않고 생성 과정을 향상시킵니다.

- **Technical Details**: 제안된 아키텍처는 정보 검색과 생성 과정을 통합하여 최신 문서의 정보를 효과적으로 처리합니다. ICV는 LLM의 잠재 상태를 다시 재구성하여 최근의 작업에 필요한 데이터를 통합합니다. 이 구조는 생성 과정에서 필요한 정보의 질과 관련성을 향상시키고, 기존의 RAG 모델의 제한 사항을 완화합니다. 이 방법은 더 짧은 프롬프트 및 계산 효율성을 제공하며, 미세 조정(fine-tuning)과 비교하여 더 적은 계산 비용을 요구합니다.

- **Performance Highlights**: 실험 결과, ICV가 표준 in-context learning 및 미세 조정 방식보다 전반적인 성능이 뛰어난 것으로 나타났습니다. 질문 답변, 정보 검색 등 다양한 작업에서 ICV 강화 모델이 LLaMA-3, Gemma, Phi-3와 같은 기존 모델들과 경쟁할만한 성능을 발휘했습니다. 계산 비용과 메모리 요구 사항을 획기적으로 줄이면서도 성능을 유지하는 이 접근 방식은 정확성과 효율성을 높였습니다.



### FuXi-$α$: Scaling Recommendation Model with Feature Interaction Enhanced Transformer (https://arxiv.org/abs/2502.03036)
Comments:
          Accepted by WWW2025

- **What's New**: 본 논문에서는 FuXi-$\alpha$라는 새로운 추천 모델을 제안합니다. 이 모델은 Adaptive Multi-channel Self-attention 메커니즘을 도입하여 시간적, 위치적, 의미적 특성을 별도로 모델링합니다. 또한, Multi-stage Feedforward Network를 포함하여 암묵적 피처 간 상호작용을 강화합니다. 실험을 통해 FuXi-$\alpha$는 기존 모델보다 뛰어난 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: FuXi-$\alpha$ 모델은 Adaptive Multi-channel Self-attention (AMS) 계층을 사용하여 시간적, 위치적 및 의미적 정보를 분리하여 모델링합니다. 이 과정은 피처 간 상호작용을 높이는데 기여하며, Multi-stage Feedforward Network (MFFN) 계층은 암묵적 피처 간 상호작용을 촉진합니다. 이러한 구조는 추천 시스템의 표현력이 더욱 향상되도록 설계되었습니다.

- **Performance Highlights**: 모델은 여러 실제 데이터셋과 Huawei Music 앱에 대한 온라인 A/B 테스트에서 우수한 성능을 입증했습니다. 실험 결과, 사용자당 평균 곡 재생 횟수가 4.76% 증가하고, 평균 청취 시간도 5.10% 향상되었습니다. 이러한 결과는 FuXi-$\alpha$의 대규모 추천 시스템에 대한 잠재력을 보여줍니다.



New uploads on arXiv(cs.CV)

### Pippo: High-Resolution Multi-View Humans from a Single Imag (https://arxiv.org/abs/2502.07785)
Comments:
          Project Page - this http URL

- **What's New**: 본 논문에서는 Pippo라는 생성 모델을 소개합니다. 이 모델은 단일 사진에서 1K 해상도의 밀집 회전 영상을 생성할 수 있으며, 추가적인 입력(예: 카메라 파라미터)을 필요로 하지 않습니다. Pippo는 3B 인물 이미지로 사전 훈련되었으며, 스튜디오 캡처된 인물을 대상으로 다중 뷰 중간 훈련 및 후 훈련 과정을 수행합니다.

- **Technical Details**: Pippo는 다중 뷰 변환(transformer)으로, 여러 일관된 이미지를 동시에 생성할 수 있습니다. 모델은 공간 앵커(spatial anchor)를 활용하여 대략적인 3D 공간에서의 위치와 방향을 지정하고, 다단계 훈련 방법을 통해 고품질 스튜디오 데이터에서 다수의 일관된 이미지를 생성합니다. 데이터 사용에서의 복잡한 조건은 최소화되고, 피크(pxiel-aligned) 제어를 통해 3D 일관성을 향상시킵니다.

- **Performance Highlights**: Pippo는 훈련 시 관찰된 뷰보다 5배 이상의 뷰를 생성할 수 있으며, 주의 편향(attention biasing) 기법을 도입하여 생성 품질 저하를 방지합니다. 또한, 3D 일관성 평가를 위한 새로운 메트릭을 도입하여 기존 방법들과 비교해 우수한 성능을 보임을 입증합니다. 본 연구는 단일 이미지로부터 고해상도와 다중 뷰 일관성을 가진 인물 이미지를 생성할 수 있는 생성 모델로서 중요한 기여를 하였습니다.



### MatSwap: Light-aware material transfers in images (https://arxiv.org/abs/2502.07784)
- **What's New**: 이번 논문에서는 MatSwap이라는 새로운 방법을 제안합니다. 이 방법은 이미지를 포토리얼리스틱하게 편집하고 물질을 지정된 표면으로 전송하는 것을 목적으로 합니다. 기존의 텍스트 엔지니어링이나 수작업 주석에 의존하는 복잡한 접근 방식 대신, MatSwap은 입력 물질과 장면 내에서의 외관 간의 관계를 직접 학습합니다.

- **Technical Details**: MatSwap은 사용자 정의된 빛과 기하학을 인식하는 diffusion model을 기반으로 합니다. 이 모델은 기존의 text-to-image 모델을 사전 훈련된 상태에서 미세 조정하며, 새로운 합성 데이터셋을 사용하여 물질 전송 작업에 맞게 최적화합니다. 최종적으로 MatSwap은 이미지를 단일 이미지 내에서 물질을 전송하는 기능을 제공하며, 기하학적 및 조명 단서를 정확히 유지합니다.

- **Performance Highlights**: MatSwap은 합성 이미지와 실제 이미지에서 평가되었으며, 최신 작업과 비교했을 때 정성적 및 정량적으로 우수한 성능을 보여줍니다. 특히, 이 방법은 물질 교체를 단순화하고 사용자에게 더 많은 제어권을 제공하여 예술가들이 더 나은 편집 결과를 얻을 수 있도록 합니다.



### A Flag Decomposition for Hierarchical Datasets (https://arxiv.org/abs/2502.07782)
- **What's New**: 이 논문은 데이터를 계층적으로 표현하기 위한 새로운 알고리즘인 Flag Decomposition (FD)을 소개합니다. 기존의 행렬 분해 기법인 Singular Value Decomposition (SVD)에 의존하는 일반적인 접근 방식에서 벗어나, 다양하고 복잡한 계층적 데이터 세트를 유지하는 방법을 탐구합니다. 이 방법은 Stiefel 좌표계의 flag 표현을 통해 특정 계층 구조를 보존하면서 데이터 표현력을 극대화할 수 있습니다.

- **Technical Details**: 계층적 데이터 세트를 정의하고 이에 대한 flag의 개념을 수학적으로 형식화합니다. 이러한 flag는 증가하는 차원을 가지는 중첩된 부분공간의 시퀀스를 나타내며, 각 flag는 signature로 설명됩니다. 또한, matrix space에서의 필수 배경 정보를 제공하여 flag의 구조적 특성을 분석합니다.

- **Performance Highlights**: Flag 기반 방법을 통해 노이즈 및 이상치가 포함된 클러스터링 작업에서 강력한 성능을 발휘하며, 특히 하이퍼스펙트럼 위성 이미지의 클러스터링 및 잡음 제거에서 표준 기법보다 우수한 결과를 보여주었습니다. 또한, few-shot 프레임워크에서 flag를 프로토타입으로 사용하면 분류 정확도가 향상되는 것을 확인할 수 있습니다.



### Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection (https://arxiv.org/abs/2502.07778)
- **What's New**: 이 논문에서는 AI 생성 이미지 탐지의 어려움을 해결하기 위해 새로운 접근 방식을 제안합니다. 제안된 'Stay Positive' 알고리즘은 생성 모델이 도입한 아티팩트에만 집중하게 하여 탐지기의 성능을 향상 시키고, 실제 데이터와 관련된 불필요한 패턴을 무시하도록 합니다. 실험 결과, 기존 탐지기보다 스퓨리어스 상관관계(spurious correlations)에 대한 민감성이 줄어들고, 후속 처리(post processing)에 대한 강건성이 개선됨을 보여줍니다.

- **Technical Details**: 가짜 이미지 탐지 문제는 이진 분류 문제로, 각 데이터 포인트는 이미지와 레이블의 쌍으로 구성됩니다. 탐지기 네트워크는 일반적으로 세 가지 구성 요소로 이루어져 있으며, 특히 ReLU 활성화와 시그모이드 활성화를 사용하는 구조를 따릅니다. 이 구조를 통해 탐지기는 가짜 아티팩트에만 집중하고, 실제 이미지의 아티팩트를 무시하도록 조정됩니다.

- **Performance Highlights**: 실험 결과, Stay Positive 알고리즘으로 학습한 탐지기는 부분적으로 복원된 실제 이미지에서도 탁월한 성능을 보입니다. 이는 기존 탐지기들이 실제 이미지의 특징에 영향을 받을 수 있는 것과 비교되는 점이며, 새로운 생성 모델에 대해서도 효과적으로 일반화될 수 있음을 시사합니다. 따라서 이 연구는 잘못된 정보에 대항하여 AI 생성 이미지 탐지 분야에서 중요한 기여를 하고 있습니다.



### Novel computational workflows for natural and biomedical image processing based on hypercomplex algebras (https://arxiv.org/abs/2502.07758)
Comments:
          24 pages, 18 figures, 14 tables

- **What's New**: 본 연구는 하이퍼복소수(hypercomplex number)를 활용하여 자연 및 생명 의학(image) 이미지를 재색상화, 탈색상화, 대비 향상 및 계측학적(histological) 이미지에서의 재착색 및 염료 분리를 위한 새로운 계산 공정을 제안합니다. 논문은 색상과 명암 조절을 통해 이미지 처리 자동화(parse) 파이프라인(automated pipelines)에서 활용되는 기능을 강조합니다. 이 방법은 기본적인 산술(operation)과 행렬(matrix) 연산만으로 구현되며, 많은 최신 기술(technology) 분야에서 폭넓은 응용 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법은 쿼터니언(quaternions)을 사용하여 화소(pixel)를 쌍의 직교 2D 평면(orthogonal planes)으로 분할하는 2차원 직교 평면 분할(2D orthogonal planes split, OPS) 구조를 핵심으로 합니다. 이를 통해 이미지 데이터의 기하학적 표현을 가능하게 하여, 보다 정교한 색상 처리와 분석을 쉽게 수행할 수 있습니다. 이 연구는 생물의학 및 자연 이미지를 위한 여러 작업 흐름을 분석하며, 하이퍼복소 수 체계 내의 간단한 사칙 연산으로 이뤄진 접근법의 유용성과 일관성을 강조합니다.

- **Performance Highlights**: 연구에서 제안된 비데이터 구동(non-data-driven) 방법들은 이전의 잘 알려진 방법들과 비교하여 동등하거나 더 나은 성능을 달성하였습니다. 특히, 생물의학 이미지에 적용된 경우, 강력한 이론적 체계가 실용적인 효과를 발휘할 수 있음을 보여주며, 기존 기술들에 대한 성능 개선을 입증하였습니다. 실험 결과는 제안된 접근 방식을 사용하여 효과적으로 이미지의 색상과 대비를 조절할 수 있음을 강조하며, 디지털 병리학(digital pathology) 응용에서의 가능성 또한 제시되고 있습니다.



### Direct Ascent Synthesis: Revealing Hidden Generative Capabilities in Discriminative Models (https://arxiv.org/abs/2502.07753)
Comments:
          12 pages, 12 figures

- **What's New**: 이번 연구에서는 기존의 판별 모델들이 내재적으로 강력한 생성 능력을 포함하고 있음을 보여줍니다. 이를 통해 판별(discriminative) 모델과 생성(generative) 모델 간의 근본적인 차이를 도전하는 Direct Ascent Synthesis (DAS) 방법을 제시하였습니다. DAS는 CLIP 모델 표현의 다중 해상도 최적화를 통해 이러한 잠재적 능력을 드러내며, 이는 고품질 이미지 합성을 가능하게 합니다.

- **Technical Details**: DAS는 기존의 최적화 접근법과 달리 다중 해상도에서의 최적화를 통해 자연스러운 정규화를 제공합니다. 이미지 I*를 목표 표현 v에 맞추기 위해 최적화하는 과정에서, 단일 해상도로 작업하는 전통적인 방법들이 생성하는 잡음 패턴을 방지하고 있습니다. 이러한 방식은 고주파 솔루션의 퇴화를 방지하면서도 이미지 생성 과정을 명확하게 제어할 수 있는 장점을 가져옵니다.

- **Performance Highlights**: DAS는 이미지 생성, 수정, 재구성, 스타일 전송 및 인페인팅 작업을 포함한 여러 실험에서 검증되었습니다. 결과적으로 판별 표현과 적절한 최적화 우선 사항을 결합함으로써, 기존 생성 훈련의 계산적 및 데이터 요구사항 없이도 고품질 합성이 가능함을 보여주었습니다. 이는 판별 모델들이 이전에 인식된 것보다 더욱 풍부한 생성 지식을 포함하고 있음을 입증하는 새로운 통찰을 제공합니다.



### CausalGeD: Blending Causality and Diffusion for Spatial Gene Expression Generation (https://arxiv.org/abs/2502.07751)
- **What's New**: 이 논문에서는 단일 세포 RNA 시퀀싱(scRNA-seq)과 공간 전사체학(spatial transcriptomics, ST) 데이터를 통합하기 위한 새로운 모델인 CausalGeD를 제안합니다. 기존 방법론의 한계를 극복하기 위해 유전자 간의 인과 관계를 고려하여 성능을 개선하였습니다. CausalGeD는 확산(diffusion)과 자기회귀(autoregressive) 프로세스를 결합하여 인과 관계를 효과적으로 활용하며, 기존 방법들보다 5-32% 향상된 성능을 보였습니다.

- **Technical Details**: CausalGeD는 Granger 인과 분석을 통해 유전자 간의 통계적인 인과 관계를 모델링합니다. 이 모델은 인과성을 인식하는 Transformer 모듈을 포함하고 있으며, 다양한 조직 데이터셋에 대해 실험을 통해 성능을 검증하였습니다. 확산 프로세스와 자기회귀를 결합하여, 고차원 연속 데이터에 대한 효율적인 처리를 가능하게 합니다.

- **Performance Highlights**: 10개의 다양한 조직 데이터셋에 대한 실험 결과, CausalGeD는 Pearson 상관 계수 및 구조적 유사성 같은 중요한 지표에서 기존 최첨단 기법을 초과하는 성능을 보여주었습니다. 이러한 성과는 유전자 조절 메커니즘에 대한 심층적인 통찰력을 제공하며, ST와 scRNA-seq 데이터의 정확한 통합을 가능하게 합니다.



### Next Block Prediction: Video Generation via Semi-Auto-Regressive Modeling (https://arxiv.org/abs/2502.07737)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 Next-Block Prediction (NBP)이라는 새로운 반자기회귀(semi-autoregressive) 프레임워크를 제안합니다. 기존의 Next-Token Prediction (NTP) 방식의 한계를 극복하기 위해 비디오 생성을 위해 비단일 토큰이 아닌 균일하게 분해된 블록 단위로 생성 과정을 전환합니다. 이를 통해, 동일 블록 내의 모든 토큰이 다음 블록의 해당 토큰을 동시 예측할 수 있게 됩니다.

- **Technical Details**: NBP 프레임워크는 블록 내에서 양방향(attention) 주의를 사용하여 강력한 공간적(spatial) 의존성을 포착 가능하게 하며, 다수의 토큰을 병렬로 예측하여 생성 단계를 유의미하게 줄입니다. 이 방식은 기존 AR 모델의 한계를 극복하며, FVD(Frechet Video Distance) 기준으로 UCF101에서 103.3, K600에서 25.5를 기록했습니다. 모델 파라미터 수를 700M에서 3B로 확대하면서 생성 품질이 크게 개선되었음도 입증되었습니다.

- **Performance Highlights**: NBP 모델은 128x128 해상도를 가진 비디오 프레임을 초당 8.89장 생성할 수 있으며, 이는 기존 NTP 모델보다 11배의 속도 향상을 의미합니다. UCF101에서 FVD 점수가 103.3에서 55.3으로 감소하였으며, K600에서는 25.5에서 19.5으로 개선되었습니다. 이러한 성과는 NBP가 효율적이면서도 고품질의 비디오 생성을 제공함을 나타냅니다.



### EdgeEar: Efficient and Accurate Ear Recognition for Edge Devices (https://arxiv.org/abs/2502.07734)
Comments:
          Submitted to IEEE FG 2025

- **What's New**: 이 논문에서는 EdgeEar라는 경량 모델을 소개하고 있습니다. EdgeEar는 하이브리드 CNN-Transformer 아키텍처를 기반으로 하여 리소스 제약이 있는 디바이스에서도 효과적으로 작동하도록 설계되었습니다. 기존의 첨단 모델에 비해 파라미터 수를 50배 줄여 200만 개 이하로 유지하면서도 경쟁력 있는 정확도를 가지고 있습니다.

- **Technical Details**: EdgeEar는 Split Depth-wise Transpose Attention (SDTA) 모듈 내의 특정 선형 레이어를 Low Rank Linear (LoRaLin) 레이어로 대체하여 설계되었습니다. 이러한 접근은 모델의 복잡성과 계산 비용을 줄이는 동시에 인식 정확도를 유지합니다. EdgeEar는 512x512x512 차원의 귀 임베딩을 생성하며, 총 1.98M의 파라미터를 가지고 있습니다.

- **Performance Highlights**: Unconstrained Ear Recognition Challenge (UERC2023) 벤치마크에서 EdgeEar는 가장 낮은 Equal Error Rate (EER) 점수를 달성했습니다. EER은 0.143이며, Area Under Curve (AUC)는 0.904, Rank-1 (R1) 정확도는 0.929를 기록했습니다. 이 결과는 귀 생체 인식을 보다 효과적으로 채택할 수 있는 가능성을 보여줍니다.



### PRVQL: Progressive Knowledge-guided Refinement for Robust Egocentric Visual Query Localization (https://arxiv.org/abs/2502.07707)
- **What's New**: 본 논문은 로컬리제이션을 뛰어난 성능으로 개선하기 위해 타겟 관련 지식을 비디오에서 직접 활용하는 PRVQL(Progressive knowledge-guided Refinement framework for EgoVQL)을 제안합니다. PRVQL은 비디오에서 추출한 정보를 기반으로 쿼리 및 비디오 특징을 지속적으로 개선하여 최종적인 타겟 로컬리제이션 성능을 향상시킵니다. 기존 방법에 비해 PRVQL은 주어진 객체 단서 외에도 비디오에서 추가적인 중요한 타겟 정보를 활용합니다.

- **Technical Details**: PRVQL은 여러 개의 처리 단계를 포함하여 쿼리와 비디오 특징을 각각 업데이트하는 두 가지 모듈, 즉 AKG(appearance knowledge generation)와 SKG(spatial knowledge generation)를 포함합니다. AKG는 비디오에서 타겟의 외관 정보를 추출하고, SKG는 비디오로부터 타겟 위치 단서를 학습하여 쿼리 및 비디오 특징을 세분화합니다. 이러한 점진적 프로세스를 통해 PRVQL은 타겟 지식을 단계적으로 개선할 수 있습니다.

- **Performance Highlights**: Ego4D 데이터셋 실험에서 PRVQL은 기존의 방법들보다 뛰어난 성능을 보여주며, 타겟 지식을 활용한 로컬리제이션의 효과성을 증명합니다. 특히, PRVQL은 복잡한 장면에서의 성능 저하를 극복하며, 쿼리와 비디오 특징이 더욱 정교하게 다듬어지는 결과를 보여줍니다.



### Magic 1-For-1: Generating One Minute Video Clips within One Minu (https://arxiv.org/abs/2502.07701)
- **What's New**: 이 기술 보고서는 Magic 1-For-1 (Magic141)이라는 비디오 생성 모델을 소개합니다. 이 모델은 메모리 소비와 추론 지연을 최적화하여 효율적인 비디오 생성을 제공합니다. 주요 아이디어는 텍스트-비디오 생성 작업을 두 개의 더 쉬운 하위 작업으로 분리하는 것으로, 이 과정에서 경험적으로 텍스트-이미지 생성 작업이 더 빠르게 수렴하는 것을 입증하였습니다.

- **Technical Details**: Magic141은 텍스트-이미지 생성과 이미지-비디오 생성을 통해 비디오 생성 작업을 단순화합니다. 이 모델은 세 가지 관점에서 이미지-비디오(I2V) 모델의 학습 비용을 줄이는 다양한 최적화 기법을 탐색합니다. 특히, 멀티모달 조건 인젝션, 적대적 단계 증류(adversarial step distillation) 사용 및 매개변수 희소화(parameter sparsification)를 통해 성능을 최적화하고 있습니다.

- **Performance Highlights**: 이 모델을 사용하면 5초 길이의 비디오 클립을 3초 내에 생성할 수 있으며, 한 번의 테스트 시간 슬라이딩 윈도우를 적용하여 1분 길이의 비디오를 1분 내에 생성할 수 있습니다. 이 과정에서 시각적 품질과 모션 동적의 품질을 개선하며, 평균적으로 1초 비디오 클립 생성에 1초 미만의 시간을 소요합니다. 이 연구는 컴퓨터 자원과 비디오 품질 간의 최적 균형을 찾기 위한 초기 탐색을 포함하여서, 오픈 소스 탐색에 적합한 기반 모델로 기대됩니다.



### Matrix3D: Large Photogrammetry Model All-in-On (https://arxiv.org/abs/2502.07685)
Comments:
          Project Page: this https URL

- **What's New**: Matrix3D는 다양한 포토그래메트리 작업을 하나의 모델로 수행할 수 있는 통합 모델로 소개됩니다. 이 모델은 이미지, 카메라 매개변수, 깊이 맵 등 여러 모달리티를 통합하여 다중 모달 확산 변환기(Diffusion Transformer, DiT)를 사용합니다. 또한 마스크 학습 전략을 포함하여 부분적으로 완전한 데이터로도 전모달리티 모델 훈련을 가능하게 하여 훈련 데이터풀을 대폭 확장합니다.

- **Technical Details**: Matrix3D는 카메라 지오메트리를 Plücker ray maps로 인코딩하고, 3D 구조를 2.5D 깊이 맵으로 표현하는 통합 2D 표현을 사용합니다. 도식 변환기를 다중 뷰, 다중 모달 환경으로 확장하여 모든 필요한 모달리티를 생성할 수 있습니다. 마스크 학습 설계는 다양한 입력 희소성을 효과적으로 관리하며, bi-modality 데이터 샘플을 활용하여 훈련 데이터의 양을 크게 증가시키는 장점을 제공합니다.

- **Performance Highlights**: Matrix3D는 포즈 추정 및 새로운 뷰 합성 분야에서 최첨단 성능을 보여줍니다. 이 모델은 다중 라운드 상호 작용을 통해 세밀한 제어를 가능하게 해 3D 콘텐츠 생성의 혁신적인 도구로 자리잡고 있습니다. 또한 기존의 작업 별 접근법과 비교하여 우수한 성능을 정량적 및 정성적 실험을 통해 입증하였습니다.



### Multiview Point Cloud Registration Based on Minimum Potential Energy for Free-Form Blade Measuremen (https://arxiv.org/abs/2502.07680)
- **What's New**: 이 논문에서는 산업 측정에서의 자유형 블레이드 재구성을 위한 새로운 전역 등록 방법을 제안합니다. 이 방법은 최소 잠재 에너지(minimum potential energy, MPE) 방법에 기반하여, 노이즈(noise)와 이상치(outlier)의 영향을 감소시킵니다. 등록 절차에서 노이즈에 대한 가중치를 조정하고, 수렴을 높이기 위해 잘라낸 반복 최근접 점 알고리즘(trimmed iterative closest point algorithm)을 포함하여 최적 근사 방법을 사용합니다.

- **Technical Details**: 기존의 블레이드 측정 기술은 주로 좌표 측정 기계(coordinate measuring machine, CMM)에 의존하며, 높은 정밀도를 제공하지만 느리고 생산성이 낮습니다. 반면에, 본 연구에서는 자동 블레이드 측정을 위한 3D 포인트 클라우드 등록(point cloud registration, PCR) 기술을 적용하여 수동 개입을 최소화하고 측정의 효율성을 높입니다. 가능하면 분산되어 있지 않은 상태에서 측정이 이루어져야 하며, 각 블레이드의 정확한 재구성을 목표로 합니다.

- **Performance Highlights**: 제안된 알고리즘은 서로 다른 4종의 블레이드에 대해 성능을 검증하였으며, 기존의 전역 등록 방법들에 비해 정확성과 노이즈 저항성에서 우수한 성능을 발휘하는 것을 보였습니다. 이로써 3D 재구성의 정밀도가 향상되며, 제조 과정에서의 직접적인 응용 가능성을 보여줍니다. 따라서, 대량 생산 블레이드 제조 시 빠르고 정확한 측정 솔루션으로의 발전 가능성을 제시합니다.



### Divide and Merge: Motion and Semantic Learning in End-to-End Autonomous Driving (https://arxiv.org/abs/2502.07631)
- **What's New**: 이 논문에서는 이질적인 정보 유형인 의미(semantics)와 움직임(motion)의 인식을 위한 혁신적인 동시 감지(detection), 추적(tracking), 예측(prediction) 방법인 Neural-Bayes motion decoding을 제안합니다. 기존의 다중 작업 학습에서는 이 두 가지 정보를 단일 피처 벡터로 표현했으나, 이는 인식 성능 저하를 가져오는 부정적 전이(negative transfer) 문제를 야기했습니다. DMAD 구조는 의미와 움직임 학습을 분리하거나 결합하여 부정적 전이 문제를 해결합니다.

- **Technical Details**: DMAD 구조는 객체 검출과 추적 쿼리와 병렬로 작동하는 일련의 학습된 움직임 쿼리를 유지하며, 이들 쿼리는 서로 업데이트된 참조 포인트를 공유합니다. 여기서 특징적으로, 움직임 쿼리는 과거 및 미래의 궤적을 디코드하여 객체와의 교환을 제한합니다. 또한 상호 작용적 의미 디코더(interactive semantic decoder)를 통해 감지 및 맵 세분화(map segmentation) 작업 간의 정보 교환을 증대시킵니다.

- **Performance Highlights**: 실험 결과, nuScenes 데이터셋에서 DMAD 접근법을 통해 검출 성능이 5%, 추적 성능이 11% 향상되었습니다. LEAD-상태의 충돌률을 기록하여, 계획 모듈에 변경 없이도 성능을 극대화했습니다. 이러한 성과들은 DMAD 구조가 다중 작업 학습 중 나타나는 부정적 전이를 효과적으로 완화하는 데 기여했음을 보여줍니다.



### Scaling Pre-training to One Hundred Billion Data for Vision Language Models (https://arxiv.org/abs/2502.07617)
- **What's New**: 이 논문에서는 1000억 개의 이미지-텍스트 쌍을 포함하는 WebLI-100B라는 새로운 대규모 데이터셋을 소개합니다. 이러한 대규모 데이터셋은 기존의 비전-언어 모델(VLM) 훈련에서 중요한 영향을 미치며, 특히 문화 다양성과 다국어 처리에 있어 더 큰 이점을 제공합니다. 우리는 큰 데이터 규모가 전통적인 벤치마크 성능 외의 영역에서도 유의미한 효과를 가져온다는 점을 강조합니다.

- **Technical Details**: 비전-언어 모델의 성능 향상은 대규모 데이터셋의 가용성에 크게 의존해 왔습니다. 이 논문에서는 데이터 규모와 모델 성능 사이의 관계를 수학적 모델인 power law로 설명하며, 1000억 개의 데이터에서 발생할 수 있는 다양한 이점을 탐구합니다. 특히, 기존의 훈련 데이터 크기와 모델 크기를 최적화하는 것을 목표로 하는 연구 방향을 제시합니다.

- **Performance Highlights**: 1000억 개의 데이터로 훈련된 모델은 기존 데이터에 비해 문화적 다양성 및 다국어 처리에서 현저한 성과를 나타냅니다. 예를 들어, 특정 문화 다양성 기준을 바탕으로 한 작업에서 1000억 데이터로 훈련된 모델은 35.9%의 정확도를 기록한 반면, 더 작은 데이터셋으로 훈련시킨 결과는 41.7%의 성과를 이루었습니다. 이러한 결과는 긴 꼬리(long-tail) 개념을 잘 포착하는 데이터셋의 중요성을 강조합니다.



### Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors (https://arxiv.org/abs/2502.07615)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS)의 최적화 과정에 기하학적인 제약을 도입하여, 관측되지 않은 뷰와 관련된 기하학적 재구성을 개선하고자 하는 새로운 접근 방식인 Flow Distillation Sampling (FDS)을 제안합니다. FDS는 사전 학습된 광학 흐름(model) 정보를 활용하여 3DGS의 훈련 과정에서 기하학적 정확성을 증가시킵니다. 또한, FDS는 관측되지 않은 뷰에 대해 효과적으로 샘플링할 수 있는 카메라 샘플링 기법을 통합하여 최적화를 더욱 용이하게 합니다.

- **Technical Details**: FDS는 매치된 흐름을 기반으로 하여, 입력 뷰와 관측되지 않은 뷰 간의 흐름(즉, Prior Flow)을 이용하여 3DGS 기하학에서 분석적으로 계산된 흐름(즉, Radiance Flow)을 개선합니다. 이를 통해 서로 보완적인 효과가 발생하여, 기하학 정확도가 중요한 효과를 가져옵니다. 또한, 이번 연구에서는 MushRoom, ScanNet (V2), Replica 데이터셋을 활용한 기하학적 재구성을 평가하고 FDS가 기존 최첨단 기법들에 비해 유의미한 성능 향상을 보임을 입증했습니다.

- **Performance Highlights**: 연구 결과, FDS를 적용한 후 기하학적 재구성의 정확성이 현격히 향상되었습니다. 특히, 실험을 통해 FDS의 도입이 모델의 기하학적 정확성과 렌더링 품질에 긍정적인 영향을 미쳤음을 확인했습니다. 이러한 혁신적인 접근 방식은 다양한 3D 재구성 상황에서 기존 방법보다 더 나은 결과를 달성하는 데 기여할 것입니다.



### An Improved Optimal Proximal Gradient Algorithm for Non-Blind Image Deblurring (https://arxiv.org/abs/2502.07602)
- **What's New**: 이번 논문은 이미지 디블러링(image deblurring) 최적화 문제를 다루며, 알려진 블러링 커널(blurring kernel)을 가정합니다. 새롭게 제안된 개선된 최적 근접 경량화 알고리즘(IOptISTA)은 최적 경량화 방법과 가중 행렬을 기반으로 하여 비맹목 이미지 디블러링 문제를 효율적으로 해결합니다. 두 가지 정규화 경우인 $l_1$ 노름과 전체 변동(norm)을 바탕으로 수치 실험을 수행하였으며, 결과는 기존 방법들보다 PSNR과 SSIM 값이 향상되었음을 보여줍니다.

- **Technical Details**: 이미지 디블러링은 모션 블러(motion blur)나 초점 흐림(defocus)과 같은 요인으로 흐릿해진 이미지를 복원하는 중요한 과정입니다. 이 연구에서는 이미지 디블러링 문제를 선형 역 문제(linear inverse problem)로 공식화하고, 이 문제의 최적화 문제로 접근합니다. 법칙이나 제약 조건을 명시하여 문제의 ill-posed 특성을 해결하는 정규화 부가(h(x)) 개념도 소개합니다.

- **Performance Highlights**: 수치 실험 결과, 제안된 IOptISTA 알고리즘은 다른 기존 방법들에 비해 개선된 PSNR(peak signal-to-noise ratio)과 SSIM(structural similarity index) 값을 달성하였습니다. 특히, 요약된 컨버전스 속도 또한 타 알고리즘에 비해 향상되었습니다. 향후 연구는 IOptISTA의 알고리즘 효율성을 더욱 검증하기 위한 수치 실험을 추가적으로 진행할 예정입니다.



### Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models (https://arxiv.org/abs/2502.07601)
Comments:
          19 pages, 10 figures

- **What's New**: 본 연구는 Zero-Shot Anomaly Detection (ZSAD)이라는 새로운 이상 탐지 패러다임을 제시합니다. 기존의 unsupervised AD는 많은 정상 샘플을 필요로 하였으나, ZSAD는 데이터 부족 상황에서도 효과적으로 작동할 수 있습니다. 이를 위해 연구팀은 Anomaly-Instruct-125k라는 최초의 시각적 지침 튜닝 데이터셋과 VisA-D&R 평가 기준을 수립하였습니다.

- **Technical Details**: 비주얼 이상 탐지를 위한 Anomaly-OneVision (Anomaly-OV)은 인간의 시각 검사 행동에서 영감을 받아 Look-Twice Feature Matching (LTFM) 메커니즘을 활용하여 비정상적인 비주얼 토큰을 적응적으로 선택하고 강조합니다. 기존의 ZSAD 방법과는 달리 Anomaly-OV는 비주얼 인코더만을 이용하여 객체 관련 이상 임베딩을 직접 학습합니다. 또한, 고해상도 이미지를 여러 개의 크롭으로 나누어 처리하는 AnyRes 기법을 사용합니다.

- **Performance Highlights**: Anomaly-OV는 검출 및 추론 모두에서 고급 모델들에 비해 상당한 성능 향상을 보여줍니다. 실험을 통해 Anomaly-OV는 비정상적 세부 사항을 정확하게 감지하고 설명할 수 있는 능력을 입증하였습니다. 이 모델은 공업 결함 검사에서 의료 이미지 진단에 이르기까지 다양한 응용 분야에서 그 효과를 확장할 수 있는 잠재력을 가지고 있습니다.



### PlaySlot: Learning Inverse Latent Dynamics for Controllable Object-Centric Video Prediction and Planning (https://arxiv.org/abs/2502.07600)
- **What's New**: 이번 연구에서는 PlaySlot이라는 객체 중심의 비디오 예측 모델을 제안합니다. 이 모델은 라벨이 없는 비디오 시퀀스에서 객체 표현(object representations)과 잠재적 행동(latent actions)을 추론하여 미래의 객체 상태와 비디오 프레임을 예측합니다. 이렇게 하여, 다양한 가능성을 가진 미래 예측을 가능하게 합니다.

- **Technical Details**: PlaySlot은 비디오 동역학(video dynamics)으로부터 추론된 잠재적 행동을 통해 미래의 다양한 상태를 생성합니다. 이 모델은 사용자에 의해 제공된 정보나 학습된 행동 정책(learned action policy)을 통해 조건화된 여러 가능한 미래를 생성할 수 있습니다. 이러한 방법은 객체 중심(object-centric) 비디오 예측의 향상된 성능을 보여줍니다.

- **Performance Highlights**: PlaySlot은 다양한 환경에서 스토캐스틱(stochastic) 및 객체 중심 기준선(baselines)보다 우수한 비디오 예측 성능을 보여줍니다. 또한, 추론된 잠재적 행동을 사용하여 라벨이 없는 비디오 데모로부터 로봇 행동을 효율적으로 학습할 수 있음을 입증하였습니다. 이 연구의 결과는 비디오 및 코드로 제공됩니다.



### YOLO Network For Defect Detection In Optical lenses (https://arxiv.org/abs/2502.07592)
- **What's New**: 이 연구는 YOLOv8 딥러닝 모델을 기반으로 한 자동 결함 감지 시스템을 제안합니다. 기존의 수동 검사 방법의 비효율성과 낮은 정확도를 극복하기 위해, 광학 렌즈의 결함을 효율적이고 정확하게 감지할 수 있는 시스템이 필요했습니다. 본 연구는 커스터마이즈 된 데이터셋을 사용하여 이 시스템의 실용성을 입증하며, 산업 환경에서의 품질 관리 프로세스를 향상시키는 데 기여하는 것을 목적으로 합니다.

- **Technical Details**: 연구에서는 고해상도 디지털 카메라를 사용해 표준 조명 조건 하에서 광학 렌즈의 이미지를 촬영하여 데이터셋을 구축했습니다. 이후 Roboflow를 통해 각 렌즈와 결함이 있는 영역을 주석 처리하여 모델 학습에 적합한 고품질 데이터를 확보했습니다. YOLOv8 아키텍처는 뛰어난 정확도와 속도를 제공하며, 엣지 디바이스에서 실시간으로 결함을 감지할 수 있도록 설계되었습니다.

- **Performance Highlights**: YOLOv8을 기반으로 한 이 시스템은 광학 렌즈의 결함을 실시간으로 감지하는 데 성공했습니다. 연구 결과, 이 시스템은 기존 기술에 비해 높은 정확도를 보여주며, 제조 현장에서의 적용 가능성을 입증했습니다. 이를 통해 고속 및 고정밀 결함 감지가 가능하게 되어, 광학 렌즈 제조에서 품질 보증 과정을 혁신할 수 있는 잠재력을 가지고 있습니다.



### An Elliptic Curve Based Solution to the Perspective-Three-Point Problem (https://arxiv.org/abs/2502.07564)
- **What's New**: 이번 논문은 Perspective-Three-Point Problem (P3P) 해결을 위한 새로운 방법론을 제시합니다. 기존의 접근 방식과 달리, 카메라와 제어점 간의 거리보다는 제어점 쌍을 통한 선의 방향을 먼저 조사하는 데 집중합니다. 이 연구의 주요 발견은 P3P 문제가 특정 종류의 타원 곡선과 깊은 연관이 있다는 점이며, 이는 암호학에서도 활용되는 곡선들입니다.

- **Technical Details**: 연구에서는 고전적인 '슬라이딩' 문제의 구형 아날로그를 설정하고 해결합니다. 이 구형 문제의 솔루션은 사영 평면에서 쉽게 변환할 수 있는 사차 곡선을 형성합니다. P3P 문제는 이러한 두 개의 사차 곡선의 교차점을 찾는 방식으로 재해석되었으며, 이는 기존 방법들과의 명확한 연계를 제공합니다.

- **Performance Highlights**: 새로운 알고리즘은 Lambda Twist 방법과 성능을 비교한 결과, 정확성에서는 더 우수하나 속도 면에서는 느린 결과를 보였습니다. 여러 제어점 삼각형에 대해 테스트를 실시했으며, 특정 조건하에서 정확한 결과를 제공하는 잠재력이 큽니다. 최종적으로 다루어진 타원 곡선은 P3P 문제에 대한 새로운 이해를 제공하며, 앞으로 이론적 발전을 위한 기초가 될 수 있습니다.



### Navigating Semantic Drift in Task-Agnostic Class-Incremental Learning (https://arxiv.org/abs/2502.07560)
Comments:
          11 pages

- **What's New**: 이번 논문은 Class-incremental learning (CIL) 모델이 새로운 클래스를 학습하는 과정에서 기존 지식을 잃지 않고 안정성을 유지할 수 있도록 돕기 위해, 특성 분포의 평균 및 공분산 모멘트 간의 간극을 해결하는 새로운 방법론을 제안합니다. 이를 통해 모델이 작업 식별자(task ID)를 알지 못하는 상황에서도 유연성과 안정성을 잘 조화시킬 수 있습니다. 논문에서는 평균 이동 보상(mean shift compensation) 및 공분산 보정(covariance calibration) 기법을 도입하여 더욱 효과적인 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: 연구에서는 각 클래스의 평균을 샘플 임베딩의 평균으로 계산하고, 이전 평균과의 근접성을 기반으로 가중치를 두고 임베딩의 변화를 추정하여 작업 이동(task shifts)을 측정합니다. 이는 모든 학습된 클래스의 평균 이동을 효과적으로 포착하는 데 기여합니다. Mahalanobis distance를 이용한 공분산 보정 기법을 통해 이전 네트워크와 현재 네트워크 간에 클래스별 임베딩 공분산을 정렬하여 공분산 이동을 완화합니다.

- **Performance Highlights**: 다양한 공개 데이터셋을 대상으로 한 포괄적인 실험을 통해 제안된 방법이 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 논문에서 제시한 접근법은 지속적인 학습을 가능하게 하면서도 모델의 안정성을 효과적으로 유지하도록 설계되었습니다. 이러한 방법론은 향후 지속적 학습 시스템의 실용적 배포를 보다 용이하게 할 것으로 기대됩니다.



### VidCRAFT3: Camera, Object, and Lighting Control for Image-to-Video Generation (https://arxiv.org/abs/2502.07531)
- **What's New**: 최근 image-to-video (I2V) 생성 방법들은 카메라 궤적(camera trajectory)이나 객체 움직임(object motion)과 같은 시각적 요소를 제어하는 데 성공을 거두었습니다. 하지만 기존의 방법들은 다수의 시각적 요소에 대한 제어를 제공하는 데 한계가 있었고, 이러한 문제를 해결하기 위해 새로운 프레임워크인 VidCRAFT3를 소개하고 있습니다. VidCRAFT3는 카메라 움직임, 객체 움직임, 조명 방향을 동시에 제어할 수 있도록 설계되었습니다.

- **Technical Details**: VidCRAFT3는 Spatial Triple-Attention Transformer를 포함하여 각 시각적 요소에 대한 제어를 더 잘 분리할 수 있는 기술을 채택하고 있습니다. 또한, 실제 비디오 데이터셋의 조명 주석이 부족한 문제를 해결하기 위해 VideoLightingDirection (VLD)라는 고품질 합성 비디오 데이터셋을 구축하였습니다. 이 데이터셋은 다양한 외관의 객체와 조명 방향 주석을 포함하고 있어 VidCRAFT3가 강한 빛의 전송 및 반사 효과를 효과적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과 VidCRAFT3는 다양한 벤치마크 데이터셋에서 기존의 최첨단 방법들을 초월하여 높은 품질의 비디오 콘텐츠를 생성하는 능력이 입증되었습니다. 특히, 제어의 세밀함(control granularity)과 시각적 일관성(visual coherence) 면에서 탁월한 성능을 보여주었습니다. 이 모든 코드와 데이터는 공개될 예정이므로 연구자들이 쉽게 접근할 수 있도록 할 것입니다.



### CodePhys: Robust Video-based Remote Physiological Measurement through Latent Codebook Querying (https://arxiv.org/abs/2502.07526)
- **What's New**: 이 논문에서 제안하는 CodePhys는 원거리 photoplethysmography(rPPG) 측정을 코드 쿼리 작업으로 처리하는 혁신적인 방법입니다. 이 접근법은 노이즈 없는 PPG 신호에서 구축된 코드북을 이용하여 얻어진 특성 쿼리를 매칭하여 rPPG 신호를 재구성합니다. CodePhys는 시각적 간섭을 효과적으로 완화하며, rPPG 신호의 신뢰성과 정확성을 높입니다.

- **Technical Details**: CodePhys는 신호 자동 인코더 프레임워크를 사용하여 GT-PPG 신호를 재구성하여 기본 벡터 집합을 형성하는 과정으로 구성됩니다. 이 과정에서 코드북을 만들고, rPPG 측정 작업을 코드 쿼리 작업으로 변환하여 진단을 수행합니다. 이를 통해 rPPG 특성을 맵핑하는 공간 인식 인코더 네트워크를 구현하여 신체 신호의 중요성을 강조합니다.

- **Performance Highlights**: 실험 결과, CodePhys는 다양한 비디오 품질 저하 시나리오에서도 최첨단 방법들보다 더 뛰어난 성능을 보여주었습니다. CodePhys는 여러 벤치마크 데이터셋에서 강력한 성능을 달성했으며, 이로 인해 다양한 실제 애플리케이션에 널리 적용될 가능성이 입증되었습니다.



### Enhance-A-Video: Better Generated Video for Fr (https://arxiv.org/abs/2502.07508)
- **What's New**: 이번 연구에서는 DiT(디퓨전 트랜스포머) 기반 비디오 생성 모델을 향상시키기 위한 훈련 없는 접근법인 Enhance-A-Video를 소개합니다. 이 방법은 비디오의 프레임 간 상관관계를 개선하는 데 중점을 두며, 기존 모델의 재훈련이나 파인튜닝(fine-tuning) 없이 대부분의 DiT 기반 프레임워크에 쉽게 적용될 수 있습니다. 특히, 시간적 일관성(temporal consistency)과 시각적 품질(visual quality)을 높이는 것을 목표로 합니다.

- **Technical Details**: Enhance-A-Video의 핵심 아이디어는 비-diagonal temporal attention distributions를 기반으로 프레임 간 상관관계를 강화하는 것입니다. 이를 통해 기존의 DiT 기반 비디오 생성 모델에서 발생할 수 있는 부자연스러운 전환 및 품질 저하 문제를 극복하고자 합니다. 다양한 DiT 기반 비디오 생성 모델에서 우리의 접근법이 뛰어난 성능 향상을 보여주었습니다.

- **Performance Highlights**: 이 연구 결과에 따르면, Enhance-A-Video는 시간적 일관성과 시각적 품질 모두에서 향상된 결과를 제공합니다. 이러한 접근은 다른 비디오 생성 모델에 응용할 가능성이 높아, 비디오 생성 업계의 향후 연구에 영감을 줄 수 있을 것으로 기대됩니다.



### Efficient Continuous Group Convolutions for Local SE(3) Equivariance in 3D Point Clouds (https://arxiv.org/abs/2502.07505)
- **What's New**: 이번 연구에서는 점 구름(point cloud) 처리를 위한 효율적인 연속 및 지역 SE(3) 동등 컨볼루션 레이어를 제안합니다. 기존 연구들이 데이터 변환의 글로벌 회전 동등성에만 국한된 반면, 이 방법은 로컬 기하학을 활용하여 다양한 객체의 관계를 포착할 수 있는 능력을 제공합니다. 특히, 적은 양의 샘플만으로도 정확한 동등성을 보장하며, 이는 메모리와 계산 부담을 크게 줄여줍니다.

- **Technical Details**: 기존의 3D 데이터 처리 접근법은 데이터 증강(data augmentation) 기법에 의존하여 글로벌 동등성을 달성하였습니다. 그러나 점 구름을 구성하는 다양한 객체 간의 상대적 회전은 이러한 방식으로는 잡아내기 힘들며, 연구진은 이를 해결하기 위해 로컬 참조 프레임(Local Reference Frame)을 도입한 집합 ℱ(x)를 활용한 그룹 컨볼루션(group convolution)을 적용합니다. 이 방법은 6차원(6D) 연산의 복잡성을 줄이면서도 정확한 동등성을 전달할 수 있는 장점을 갖습니다.

- **Performance Highlights**: 실험 결과, 제안된 기법은 널리 알려진 데이터셋과 작업에서 경쟁력 있고, 기존의 로컬 동등 디자인보다 월등한 성능을 보여주었습니다. 또한, 이 네트워크는 훈련 중에 보지 못한 로컬 변환에 대해서도 강인함을 유지하며, 이는 기존의 글로벌 동등 프레임워크가 실패하는 상황에서도 뛰어난 성능을 발휘합니다.



### Automated Road Extraction and Centreline Fitting in LiDAR Point Clouds (https://arxiv.org/abs/2502.07486)
Comments:
          8 pages, 10 figures, accepted in DICTA 2024

- **What's New**: 이 연구는 3D 포인트 클라우드에서 도로 정보를 추출하기 위한 새로운 방법을 제안합니다. 기존 방법들이 특정 커브 디자인에 의존했던 반면, 본 연구는 LiDAR 기반의 3D 포인트 클라우드를 상단에서 시각화하여 도로 추출의 정확성과 효율성을 높였습니다.

- **Technical Details**: 제안된 접근 방식은 처음에 통계적 이상치 제거(statistical outlier removal)와 밀도 기반 군집화(density-based clustering)를 통해 노이즈를 줄입니다. 이후, 다양한 도로 시나리오와 지형 특성에 적응하는 그리드 기반 세분화(grid-based segmentation) 방법을 사용하여 지면 포인트를 필터링합니다. 마지막으로, 스켈레토화(skeletonization) 알고리즘을 통해 도로가 추출됩니다.

- **Performance Highlights**: 퍼스 CBD 데이터셋에서 테스트한 결과, 초기 접근 방식은 67%의 Intersection over Union (IoU)을 달성했으며, 도로 스켈레톤의 후처리를 포함하면 73%의 IoU와 23%의 처리 시간 단축을 이뤘습니다. 이러한 결과는 3D-2D 포인트 클라우드 정렬을 위한 기초를 제공합니다.



### Less is More: Masking Elements in Image Condition Features Avoids Content Leakages in Style Transfer Diffusion Models (https://arxiv.org/abs/2502.07466)
- **What's New**: 이 논문에서는 스타일 참조 이미지(style-reference image)를 활용하여 텍스트-이미지 확산 모델(text-to-image diffusion models)의 성능을 개선하는 새로운 방법을 제안합니다. 기존 방법들이 스타일과 내용을 분리하는 데 어려움을 겪고 있는 것에 비해, 제안된 방법은 마스킹(masking) 기법을 통해 이러한 문제를 해결합니다. 특정 이미지 특징(elements)만 마스킹하여 원하는 결과를 효율적으로 도출할 수 있다는 점에서 혁신적입니다.

- **Technical Details**: 논문은 IP-Adapter를 기반으로 한 마스킹 기법을 설명합니다. 확산 모델(diffusion models)은 데이터를 단계적으로 변환하는 두 가지 과정, 즉 확산 과정(diffusion process)과 디노이즈 과정(denoising process)을 포함합니다. 여기서 디노이즈 모델은 U-Net을 사용하며, 평균 제곱 손실(mean-squared loss)로 학습됩니다.

- **Performance Highlights**: 다양한 스타일에 대한 실험 결과, 제안된 마스킹 기반 방법이 기존 최첨단 기법들에 비해 뛰어난 성능을 보였음을 증명합니다. 마스킹 기법을 통해 스타일 전이 성능이 개선되며, 이론적인 근거와 함께 효과성을 지원하는 실험 결과를 제시합니다. "Less is more" 원칙에 따라, 적절히 선택된 조건을 통한 스타일 전이 성능 개선을 강조합니다.



### Bidirectional Uncertainty-Aware Region Learning for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2502.07457)
- **What's New**: 본 논문에서는 반지도학습(semi-supervised learning)에서 잘못된 pseudo-label의 문제를 해결하기 위해 새로운 bidirectional uncertainty-aware region learning 전략을 제안합니다. 이 방법은 두 가지 방향으로 학습을 진행하여 모델의 성능을 극대화합니다. 특히, 고불확실성(high-uncertainty) 영역에서 정확한 레이블 정보를 통해 모델의 학습을 유도하고, 저불확실성(low-uncertainty) 영역에선 잘못된 pseudo-label의 간섭을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 방법은 모델의 출력에서 예측 엔트로피(entropy) 맵과 pseudo-label 오류 맵을 분석하여 고엔트로피(high-entropy) 영역에서 잘못된 pseudo-label이 발생하는 경향을 발견하였습니다. 이 후, 각각의 입력 이미지에 대해 모델 예측 결과를 바탕으로 엔트로피 값 행렬을 생성하고, 이를 통해 불확실한(high-entropy) 영역과 확실한(low-entropy) 영역을 분류합니다. 고불확실성 영역에 더 높은 학습 가중치를 부여하여 레이블 정보를 활용하고, 저불확실성 영역에서는 학습 가중치를 줄이면서 안정적인 결과를 도출합니다.

- **Performance Highlights**: 이 방법을 통해 모델의 전체 성능이 크게 향상되었으며, 다양한 의료 이미지 분할 작업에 대해 최고의 성과를 도출했습니다. 실험 결과, 제안된 방법이 기존의 상태-of-the-art(SOTA) 방법들에 비해 상당한 성능 향상을 이뤄냈음을 보여주며, 다른 반지도 학습 방법들에 손쉽게 적용할 수 있는 장점을 가지고 있습니다.



### RusCode: Russian Cultural Code Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2502.07455)
Comments:
          Accepted for NAACL 2025 Findings, GitHub: this https URL

- **What's New**: 이번 논문에서는 러시아 문화 코드의 요소를 포함한 텍스트-이미지 생성 모델의 품질을 평가하기 위한 RusCode 벤치마크를 제안합니다. 이는 다양한 문화적 측면을 반영하는 19개의 카테고리로 구성된 데이터셋을 기반으로 하며, 1250개의 러시아어 텍스트 프롬프트와 그 영어 번역이 포함되어 있습니다. 문화 간의 이해 부족과 생성 품질 저하 문제를 해결하기 위한 노력의 일환으로 쓰여졌습니다.

- **Technical Details**: 러시아 시각문화의 특정 개념을 반영한 텍스트 설명의 품질을 평가하기 위해 다각적인 전문가의 참여를 통해 19개 카테고리를 생성하였습니다. 이 데이터셋은 역사, 예술, 민속 등 다양한 주제를 포함하여, 각 프롬프트는 해당 개념의 실제 이미지를 연관시켜 생성 품질 평가에 활용될 수 있습니다. 이를 통해 현대 텍스트-이미지 생성 모델의 다문화적 이해 현황을 분석할 수 있습니다.

- **Performance Highlights**: 인간 평가 결과는 최신 텍스트-이미지 생성 모델들, 즉 Stable Diffusion 3, DALL-E 3 등에서 러시아 문화 개념의 표현이 얼마나 잘 이루어지는지를 비교 분석하였습니다. 이를 통해 생성 모델의 문화 인식 수준을 평가하고 나타난 격차를 드러내었습니다. 이 작업은 러시아 문화에 대한 문화 인식 문제의 첫 번째 포괄적 접근으로 평가됩니다.



### Optimizing Knowledge Distillation in Transformers: Enabling Multi-Head Attention without Alignment Barriers (https://arxiv.org/abs/2502.07436)
- **What's New**: 이번 논문은 지식 증류(knowledge distillation, KD)에서 발생하는 성능 저하 문제를 해결하기 위해 Squeezing-Heads Distillation(SHD)이라는 새로운 접근 방식을 제안합니다. 기존 방법들이 교사 모델과 학생 모델의 attention heads 수를 동기화하거나 추가적인 projectors를 도입해야 하는 반면, SHD는 다양한 head 수 사이에서의 지식 전이를 매끄럽게 처리할 수 있습니다. SHD는 수치적 근사를 통해 다수의 attentive maps를 효과적으로 압축하며, 추가 파라미터나 구조적 수정 없이도 세밀한 attention 패턴을 유지합니다.

- **Technical Details**: SHD는 선형 근사를 활용하여 여러 개의 multi-head attention maps를 하나의 attention map으로 압축하면서 교사와 학생 모델 간의 지식 전이를 용이하게 합니다. 본 연구에서는 비유연한 attention matrix 대신 비표준 attention matrix를 도입하여 더 적은 heads로 손실 없는 feature 표현을 달성합니다. 이 방법은 기존의 projectors 방식과 비교했을 때 계산 자원을 줄이고 전반적인 효율성을 향상시킵니다.

- **Performance Highlights**: SHD의 효과는 언어 생성(LLaMA, GPT) 및 비전(DeiT) 작업을 포함한 다양한 실험을 통해 입증되었습니다. SHD는 logits 기반 및 feature 정렬 KD의 기준선을 초월하는 성능을 보여주었으며, 이미지 분류, 이미지 생성, 언어 미세 조정 및 언어 사전 훈련 분야에서 최신 성과를 달성했습니다. 이 연구는 현대 트랜스포머 모델의 효율적인 배포를 가능하게 하는 중요한 기술적 혁신을 담고 있습니다.



### ArthroPhase: A Novel Dataset and Method for Phase Recognition in Arthroscopic Video (https://arxiv.org/abs/2502.07431)
- **What's New**: 이번 연구는 무릎 관절경 수술, 특히 앞십자인대(ACL) 재건 분야에서 수술 단계 인식을 향상시키기 위해 최초의 arthroscopy 데이터셋과 새로운 transformer 기반 모델을 도입했습니다. 이 연구는 제한된 시야, 장애물, 시각 왜곡을 포함한 관절경 비디오의 특정 도전 과제를 해결하기 위해 시공간(spatio-temporal) 특성을 활용하여, 관절경 수술 단계 인식의 기준점을 구축하는 것을 목표로 하고 있습니다.

- **Technical Details**: 우리는 ACL 수술에 대한 27개의 비디오로 구성된 ACL27 데이터셋을 개발하고, 각 비디오는 수술 단계로 레이블링되었습니다. 모델은 ResNet-50과 transformer 레이어를 사용하여 시간 인식 프레임 기반 특성 추출을 수행하는 transformer 기반 아키텍처를 활용합니다. 이를 통해 시공간 특성을 통합하고 수술 진행도를 정량화하기 위한 수술 진행 지수(SPI)를 도입했습니다.

- **Performance Highlights**: ACL27 데이터셋에서 모델의 전반적인 정확도는 72.91%에 달했으며, Cholec80 데이터셋에서는 최첨단 방법들과 유사한 92.4%의 정확도를 기록했습니다. SPI는 ACL27과 Cholec80 데이터셋에서 각각 10.6%와 9.86%의 출력 오류를 보이며 신뢰할 수 있는 수술 진행 추정치를 나타냈습니다. 이 연구는 관절경 수술 단계 인식의 중요한 발전을 제시하며, 수술 훈련 및 실시간 지원 개선에 대한 잠재력을 강조합니다.



### Fast-COS: A Fast One-Stage Object Detector Based on Reparameterized Attention Vision Transformer for Autonomous Driving (https://arxiv.org/abs/2502.07417)
Comments:
          Under Review on IEEE Transactions on Intelligent Transportation Systems

- **What's New**: 이번 연구에서는 자율주행 애플리케이션을 위한 새로운 단일 단계 객체 감지 프레임워크인 Fast-COS를 소개합니다. 이 프레임워크는 정확성과 처리 속도 간의 균형을 이룰 수 있는 고효율 객체 탐지 전략을 목표로 하고 있습니다. Fast-COS는 Reparameterized Attention Vision Transformer (RAViT)를 백본으로 사용하여 다중 스케일 피쳐 융합을 가능하게 하며, 자율주행 시스템의 효율성을 크게 향상시킵니다.

- **Technical Details**: RAViT는 Reparameterized Multi-Scale Depth-Wise Convolution (RepMSDW)와 Reparameterized Self-Attention (RepSA)을 활용하여 계산 효율성과 피쳐 추출 성능을 높입니다. Fast-COS는 BDD100K와 TJU-DHD 데이터셋에서 각각 57.2%와 80.0%의 AP50 점수를 기록했으며, FCOS, YOLOF와 RetinaNet에 비해 최대 75.9% 빠른 GPU 추론 속도를 제공합니다.

- **Performance Highlights**: Fast-COS는 ImageNet-1K 데이터셋에서 81.4%의 Top-1 정확도를 달성하며, 기존 모델들과 비교했을 때 유의미한 처리량 향상을 보여줍니다. 이러한 결과는 Fast-COS가 리소스 제약이 있는 환경에서도 실시간 애플리케이션에 적합한 확장 가능하고 신뢰할 수 있는 솔루션임을 입증합니다.



### EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering (https://arxiv.org/abs/2502.07411)
- **What's New**: EgoTextVQA는 실생활에서의 egocentric QA 보조를 위한 새로운 벤치마크 데이터셋으로 1.5K의 개인 시점 비디오와 7K의 장면 텍스트 관련 질문을 포함하고 있습니다. 이 데이터셋은 운전과 실내 집안일과 같은 활동에서 실제 사용자 요구를 반영하여 설계되었습니다. EgoTextVQA는 현재의 다중 모달 대형 언어 모델들이 egocentric QA 보조에서 얼마나 부족한지를 강조합니다.

- **Technical Details**: EgoTextVQA는 정확한 시간 기반 정립과 다중 프레임 추론을 필요로 하며, 고해상도 이미지와 장면 텍스트 입력이 성능 향상에 핵심적임을 강조합니다. 이 연구는 유저의 의도를 추론하고 요구되는 장면 텍스트를 찾기 위해 여러 프레임을 통해 추론이 필요하다고 설명합니다. 또한, 다양한 질문 유형과 시간 스탬프가 있는 질문을 통해 실시간 비디오 QA를 시뮬레이션합니다.

- **Performance Highlights**: EgoTextVQA에서 평가된 10개의 모델들은 모두 어려움을 겪고 있으며, 가장 성능이 좋은 Gemini 1.5 Pro조차도 약 33%의 정확도에 불과합니다. 이 연구는 또한 높은 해상도의 입력과 보조형 장면 텍스트 입력이 모든 모델에 매우 유용하게 작용하며, 인간조차도 이 작업에 대해 낮은 정확도(야외 43%, 실내 27%)를 보였음을 나타냅니다. 결과적으로 EgoTextVQA는 egocentric 장면 텍스트 QA 보조 연구를 위한 강력한 테스트베드로 자리 잡기를 기대하고 있습니다.



### MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification (https://arxiv.org/abs/2502.07409)
Comments:
          first version

- **What's New**: 이 논문은 전체 슬라이드 병리 이미지 분류에서 적은 주석(label) 수로도 효과적으로 작업하기 위한 새로운 방법인 MGPath를 소개합니다. 이전의 모델을 개선하기 위해, 이 연구는 특정한 대형 비전-언어 모델을 활용하고, 병리학 이미지에서의 세밀한 특징을 캡처할 수 있는 다중-세부 구조화된 주의(attention) 메커니즘을 채택했습니다. 이 접근법은 큰 이미지 데이터에 대한 주석이 부족한 상황에서 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: MGPath는 Prov-GigaPath라는 대규모 사전 학습된 비전 모델을 기반으로 하며, 1.3억 개의 병리 이미지 패치를 통해 훈련됩니다. 이 모델은 GW-트랜스포즈(optimal transport) 메커니즘을 사용하여 다양한 텍스트 프롬프트와 시각적 임베딩 간의 거리를 측정합니다. 또한 병리학 이미지에서 헤테로지너스(homogeneous) 데이터 배포를 정렬하는 유연성을 제공합니다.

- **Performance Highlights**: MGPath는 TCGA-BRCA 데이터셋에서 MSCPT보다 5% 향상된 F1 점수를 기록하였고, 정확성에서도 CONCH 및 QUILT 모델보다 약 6% 더 나은 성능을 보였습니다. 이러한 성능 향상은 다양한 아키텍처에서 안정적으로 입증되었으며, 여러 최신 경쟁 모델들을 초월했습니다. 논문에서 제시된 방법론은 다른 실험 환경에서도 긍정적인 결과를 유지했습니다.



### Extended monocular 3D imaging (https://arxiv.org/abs/2502.07403)
- **What's New**: 이번 연구에서는 빛의 벡터파 (vectorial wave) 특성을 최대한 활용한 확장된 단안 3D 영상 (EM3D) 프레임워크를 소개합니다. 이 프레임워크는 콤팩트한 단안 카메라와 회절-굴절 하이브리드 렌즈를 사용하여 낮은 질감, 반사성이 높거나 거의 투명한 전통적으로 어려운 장면들에 대한 한 백만 픽셀 및 정확한 3D 포인트 클라우드를 실험적으로 획득하는 방법을 보여줍니다.

- **Technical Details**: EM3D는 회절 (diffraction) 및 편광 (polarization) 기반의 깊이 단서 (depth cues)를 다단계 융합 (multi-stage fusion)하여 구현됩니다. 이 과정은 데이터 프라이어 (data prior) 없이도 가능하며, 이는 3D 이미징 하드웨어의 부피와 복잡성 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: 깊이 및 편광 정보를 결합함으로써 재료 식별 (material identification)에서 새로운 기회를 제공하게 되며, 이는 타겟 인식(target recognition)과 얼굴 위조 방지(face anti-spoofing)와 같은 머신 인텔리전스의 응용 분야를 더욱 확장할 수 있습니다. 간편하면서도 강력한 이 아키텍처는 최소한의 형상에서도 고차원 머신 비전을 위한 새로운 경로를 열어 줍니다.



### FADE: Forecasting for Anomaly Detection on ECG (https://arxiv.org/abs/2502.07389)
- **What's New**: 이 연구는 심전도(ECG) 신호의 이상 탐지를 위한 새로운 딥 러닝 시스템 FADE를 제안합니다. FADE는 기존의 수동 해석에 의존하지 않고 정상 ECG 신호를 예측하는 방식으로 훈련됩니다. 이 접근은 광범위한 레이블된 데이터셋 필요성을 줄이고, 데이터를 자동으로 분석함으로써 의료 전문가의 전문성에 대한 의존도를 낮춥니다. self-supervised 방식으로 학습하는 새로운 손실 함수가 적용되어 더욱 효과적인 이상 탐지가 가능합니다.

- **Technical Details**: 본 연구에서 FADE는 전통적인 labeled 데이터에 의존하지 않고, 정상 ECG 신호의 예측을 통해 심장 이상을 탐지하는 딥 러닝 아키텍처입니다. 이 시스템은 새로운 거리 함수를 통해 예측된 ECG 신호와 실제 센서 데이터를 비교하여 심장 이상을 식별합니다. 또한, domain adaptation 기술을 통해 다양한 환경에 쉽게 적응할 수 있으며, 이러한 기술들은 ECG 신호의 변화를 수용하는 데 중점을 두고 개발되었습니다.

- **Performance Highlights**: FADE는 MIT-BIH NSR 및 MIT-BIH Arrythmia와 같은 공공 데이터셋에서 실험을 수행하여 평균 83.84%의 이상 탐지 정확도를 달성했으며, 정상 ECG 신호를 85.46%의 정확도로 분류했습니다. 이 시스템은 기존의 접근 방식보다 이상 및 부정맥을 조기에 탐지하는 데 있어 우수한 성능을 나타냈으며, 대규모 ECG 데이터 처리에 있어 비용 절감의 이점을 제공합니다.



### Spatial Degradation-Aware and Temporal Consistent Diffusion Model for Compressed Video Super-Resolution (https://arxiv.org/abs/2502.07381)
- **What's New**: 이 논문에서는 압축된 비디오를 위한 새로운 Spatial Degradation-Aware and Temporal Consistent (SDATC) 확산 모델을 제안합니다. 기존의 비디오 초해상도(VSR) 방법들이 압축 비디오에 적합하지 않은 경우가 많았던 점을 보완하기 위해, 왜곡 제어 모듈(DCM)을 통해 확산 모델의 입력을 조정하고, 압축 인식을 위한 프롬프트 기반 모듈(PCAM)과 시공간 주의 모듈(STAM)을 도입하여 시간적 일관성을 개선했습니다.

- **Technical Details**: 이 모델은 먼저 DCM을 통해 입력 프레임에서 방해가 되는 노이즈를 제거하고, 다음 단계에서 LQ (Low Quality) 이전 정보를 이용하여 왜곡을 조절합니다. 이어서, PCAM은 비트가 서로 다른 압축 정보의 특징을 동적으로 인코딩하여 프롬프트를 제공합니다. 마지막으로, STAM은 시공간 차원에서 인접한 프레임 간의 상관관계를 탐색하여 복원된 프레임의 일관성을 높입니다.

- **Performance Highlights**: 많은 실험을 통해 제안한 모듈이 압축된 비디오의 품질을 향상시키는 데 효과적이라는 것을 입증했습니다. SDATC는 압축된 비디오의 생성 과정에서 발생할 수 있는 부정적인 영향을 완화하며, 기존의 SR 확산 모델과 비교학에 두 가지 하위 작업으로 VSR을 분리하여 접근합니다. 이러한 방식은 복원 결과의 선명도를 높이는 데 기여합니다.



### USRNet: Unified Scene Recovery Network for Enhancing Traffic Imaging under Multiple Adverse Weather Conditions (https://arxiv.org/abs/2502.07372)
- **What's New**: 이 논문에서는 다양한 유형의 이미지 손상을 다룰 수 있는 통합 장면 복구 네트워크(USRNet)를 제안합니다. USRNet은 장면 인코더, 주의 기반 노드 독립 학습 메커니즘(NILM), 에지 디코더 및 장면 복원 모듈로 구성된 세련된 아키텍처를 특징으로 합니다. 이러한 구조는 악천후 환경에서 이미지 품질을 개선하는 데 중요한 역할을 합니다.

- **Technical Details**: USRNet의 핵심 구조 중 하나인 NILM은 네트워크가 여러 날씨 조건에 대한 적응력을 높일 수 있도록 독립적으로 학습할 수 있게 해줍니다. 또한, 이 네트워크는 복잡한 mixed degradation 상황에서도 이미지 에지를 정교하게 추출하여 선명도를 유지합니다. 하이브리드 손실 함수는 다양한 손실 요소를 통합하여 훈련을 세밀하게 조정하고, 다양한 손상 뉘앙스를 포착하도록 설계되었습니다.

- **Performance Highlights**: USRNet은 복잡한 이미지 손상 시나리오를 효과적으로 처리하는 뛰어난 성과를 보여줍니다. 광범위한 실험 결과는 기존 복원 방법들을 초월하는 성능을 입증하며, 이는 VITS에서 중요한 응용 가능성을 지님을 나타냅니다. 특히, USRNet은 객체 감지에 매우 효과적이며, 이는 현대 지능형 교통 시스템에 큰 가치를 더합니다.



### Multi-Task-oriented Nighttime Haze Imaging Enhancer for Vision-driven Measurement Systems (https://arxiv.org/abs/2502.07351)
- **What's New**: 이번 논문에서는 야간 안개 이미지를 향상시키기 위한 다중 작업 지향 프레임워크인 MToIE(Multi-task-oriented nighttime haze imaging enhancer)를 제안합니다. 이 프레임워크는 낮 동안의 안개 제거(Daytime Dehazing), 저조도 향상(Low-light Enhancement), 야간 안개 제거(Nighttime Dehazing)라는 세 가지 작업을 통합하여 보다 효과적인 이미지 복원을 목표로 하고 있습니다. MToIE는 특정한 노드 학습 메커니즘과 멀티 수용 범위 향상(Multi-receptive Field Enhancement) 모듈을 활용하여 복잡한 환경에서의 이미지 품질을 효율적으로 향상시킵니다.

- **Technical Details**: MToIE는 특화된 작업 노드 학습 메커니즘(task-oriented node learning)과 자기 주의 모듈(self-attention module)을 기반으로 설계되어 세 가지 차별화된 왜곡 유형을 처리할 수 있습니다. 이 네트워크는 세 가지 병렬 깊이 분리 가능한 컨볼루션 가지를 통해 다중 스케일 특성을 추출하는 멀티 수용 범위 향상 모듈(multi-receptive field enhancement module)을 포함하여, 최소한의 계산 오버헤드로 포괄적인 공간 정보를 캡처합니다. 또한, 하이브리드 손실 함수(hybrid loss function)를 통해 이미지 재구성 품질과 시각적 특성을 최적화합니다.

- **Performance Highlights**: MToIE는 다양한 기상 및 조명 조건에서 기존 방법들을 능가하는 성능을 나타내며, 이후의 고수준 비전 과제에서 더 높은 정확도와 신뢰성을 달성합니다. 실험 결과는 MToIE가 이미지 품질 향상에서 유의미한 성과를 내며, 특히 복잡한 저조도 및 안개 환경에서 효과적임을 입증했습니다. 연구진은 해당 모델의 코드도 공개하였으며, 이를 통해 향후 연구와 개발에 기여할 것으로 기대됩니다.



### ERANet: Edge Replacement Augmentation for Semi-Supervised Meniscus Segmentation with Prototype Consistency Alignment and Conditional Self-Training (https://arxiv.org/abs/2502.07331)
- **What's New**: 이번 연구에서는 ERANet이라는 혁신적인 세미-슈퍼바이즈드(semi-supervised) 프레임워크를 제안합니다. 이 프레임워크는 라벨이 붙은 이미지와 라벨이 없는 이미지를 효과적으로 활용하여 섬유연골(meniscus) 분할(segmentation)의 도전 과제를 해결하려고 합니다. ERANet은 구조적 맥락에 맞는 증강(augmentation)을 통해 해부학적으로 관련된 변형을 도입하여 성능을 향상시킵니다.

- **Technical Details**: ERANet은 세 가지 주요 구성 요소인 엣지 대체 증강(edge replacement augmentation, ERA), 프로토타입 일관성 정렬(prototype consistency alignment, PCA), 조건부 자기 학습(conditional self-training, CST) 전략을 통합합니다. ERA는 섬유연골의 변형을 시뮬레이션하여 증강이 구조적 맥락에 정렬되도록 보장합니다. PCA는 제한된 라벨 데이터 환경에서도 성능을 향상시킬 수 있도록 클래스 내(feature) 정렬을 촉진하고, CST는 훈련 중 라벨 노이즈의 영향을 줄이며 분할의 신뢰성을 개선합니다.

- **Performance Highlights**: ERANet은 3D Double Echo Steady State (DESS)와 3D Fast/Turbo Spin Echo (FSE/TSE) MRI 시퀀스에서 포괄적으로 검증되었습니다. 그 결과, 기존의 최첨단 방법들과 비교하여 우수한 성능을 나타내며, 최소한의 라벨링 데이터로도 신뢰할 수 있고 정확한 섬유연골 분할을 달성했습니다. 다양한 변별 연구(ablation studies)를 통해 ERA, PCA 및 CST의 시너지 효과를 강조하며, ERANet이 의학 이미징 분야에서 세미-슈퍼바이즈드 섬유연골 분할에 변혁적인 솔루션임을 입증하였습니다.



### Semantic to Structure: Learning Structural Representations for Infringement Detection (https://arxiv.org/abs/2502.07323)
- **What's New**: 이번 연구에서는 이미지 구조 정보가 미적 평가에 중요하다는 점을 강조하며, 기존 작품의 구조를 모방하는 것이 창작자의 권리를 침해한다는 점을 주장합니다. 이러한 현상을 "구조적 침해"(structural infringement)로 정의하고, 이에 대한 탐지 방법을 제안합니다. 새로운 데이터 합성 전략을 통해 유사한 구조 정보를 가진 이미지 쌍을 생성하고, 이를 통해 구조적 침해 탐지 모델을 성공적으로 학습시켰습니다.

- **Technical Details**: 구조적 침해 탐지 모델을 훈련하기 위해, 우리는 기존의 이미지 표현 학습 방법에 새로운 이미지 구조 표현(Image Structural Representation)을 도입합니다. 이 방법은 이미지 내부의 기하학적 및 위치 정보를 세밀하게 설명합니다. 또한, Diffusion 모델을 사용하여 출처 이미지와 구조적으로 유사하지만 의미적으로 다른 이미지 쌍을 생성하는 데이터 합성 파이프라인을 제안하며, 대규모 언어 모델(LLM)을 이용해 캡션을 수정함으로써 의미적 유사성을 줄이는 접근을 취합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 수동 주석 구조적 침해 테스트 세트(SIA 및 SIR 데이터셋)에서 최첨단 성과를 달성했습니다. 이 연구는 이미지 구조적 정보의 측면에서 현실적 및 합성 이미지를 분석하고, 구조적 침해 탐지의 새로운 기준을 제시하는 중요한 기여를 하고 있습니다. SIA와 SIR 데이터셋은 연구자들이 사용할 수 있도록 공개되었습니다.



### Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving (https://arxiv.org/abs/2502.07309)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이번 논문에서는 주행 자율주행을 위한 3D 공간 이해의 중요성을 강조하며, 2D 라벨을 활용한 반감독(semisupervised) 비전 중심 3D 점유 세계 모델인 PreWorld를 제안합니다. 이는 독창적인 두 단계의 학습 패러다임을 통해, 3D 점유 예측 및 4D 점유 예측에서 경쟁력 있는 성능을 달성하도록 설계되었습니다. 특히, 2D 라벨을 통해 모형을 사전 학습하는 자기 감독(pre-training) 단계와 완전 감독(fine-tuning) 단계를 포함하여, 3D 점유 예측의 최신 통계 성능을 보입니다.

- **Technical Details**: PreWorld는 두 단계의 학습 패러다임을 사용하여 모델의 정확도를 높이는 데 성공했습니다. 사전 학습 단계에서는 속성 투영 헤드를 사용하여 RGB, 밀도 및 의미론적으로 다양한 속성 필드를 생성하여 볼륨 렌더링 기술을 통해 2D 라벨로부터 시간적 감독을 생성합니다. 또한 단순하지만 효과적인 상태 조건 예측 모듈을 도입하여 다중 뷰 이미지 입력을 기반으로, 직접적으로 미래 3D 점유를 예측하는 방법으로 정보 손실을 방지합니다.

- **Performance Highlights**: PreWorld는 nuScenes 데이터셋을 기반으로 한 광범위한 실험에서 경쟁력 있는 성능을 보였습니다. 특히 3D 점유 예측에서는 이전의 최고의 방법인 OccFlowNet을 초월하며 mIoU 34.69를 기록하였습니다. 4D 점유 예측에서는 새로운 SOTA(performance) 성능을 달성하여, 기존 방법인 OccWorld 및 OccLLaMA를 능가하는 결과를 냈습니다. 또한 운동 계획에서는 비전 중심 방법들과 비교하여 유사하거나 더 나은 성능을 나타냈습니다.



### TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation (https://arxiv.org/abs/2502.07306)
- **What's New**: 본 연구에서는 Vision-Language Navigation (VLN) 작업을 해결하기 위한 모듈 방식의 접근법을 제안합니다. 이 접근법은 자연어로 제공된 내비게이션 지침을 처리하기 위해 최첨단 LLMs와 VLMs의 제로샷(zero-shot) 기능을 활용하여, 네 가지 하위 모듈로 문제를 분해합니다. 특히, VLM을 사용해 시각적 관찰에서 랜드마크 이름을 접합하고, 동적 프로그래밍을 통해 경로 정렬 점수를 계산합니다.

- **Technical Details**: 저자들은 R2R-Habitat 데이터셋의 복잡한 지침에 초점을 맞춰 VLN 작업을 해결하는 여덟 가지 단계를 포함하는 모듈형 접근법을 제시합니다. 1단계에서는 에이전트가 데이터셋의 훈련 에피소드를 사용하여 환경의 위상 지도를 구축합니다. 각 노드는 360° RGB 파노라마로 표현되며, 각 엣지는 노드 쌍 간의 연결을 나타내는 가중치 1을 가집니다.

- **Performance Highlights**: 이 모듈형 접근법은 복잡한 R2R-Habitat 지침 데이터셋에서 기존의 점유 맵 기반 접근법과 비교하여 우수한 성능을 보여주었습니다. 특히 미세한 랜드마크 및 행동 구문 접합의 복잡성을 나타내고, 기존 접근법의 강점과 약점을 분석하여 시각 언어 지침에 대한 현재의 LLMs와 VLMs의 성능을 평가합니다.



### CASC-AI: Consensus-aware Self-corrective AI Agents for Noise Cell Segmentation (https://arxiv.org/abs/2502.07302)
- **What's New**: 이 논문에서는 고해상도 기가픽셀 전체 슬라이드 이미지(WSI)에서 다중 클래스 세포 분할을 위한 새로운 접근 방식을 제안합니다. 기존의 비대응적 방법이 노이즈에 효과적으로 대처하지 못하는 문제를 해결하기 위해, 합의 행렬(Consensus Matrix)을 이용한 자가 수정 AI 에이전트(CASC-AI)를 도입했습니다. 이 에이전트는 세포 및 비세포 주석에 대한 일치 지역을 정의하고, 이를 통해 학습 과정을 안내합니다.

- **Technical Details**: CASC-AI는 주석에서의 오류를 적극적으로 학습하고, 픽셀 기반 및 피쳐 기반 정보를 활용하여 예측을 반복적으로 개선할 수 있습니다. 합의 행렬을 통해 합의가 있는 영역과 그렇지 않은 영역을 구분하며, 더 높은 신뢰성을 가진 지역에 더 큰 가중치를 부여합니다. 또한, 대조 학습(contrastive learning)을 통해 신뢰할 수 있는 지역과 노이즈가 포함된 지역을 분리함으로써 모델의 강건성을 향상시킵니다.

- **Performance Highlights**: 이 방법은 실제 레이 아니 웨이 주석 세포 데이터셋 및 두 개의 시뮬레이션 노이즈 데이터셋에서 검증된 결과, FP 및 FN 오류 교정을 통해 세분화 성능을 향상시키는 데 성공했습니다. 이로 인해 노이즈가 포함된 데이터셋에서 강건한 모델을 훈련하는 데 필요한 가능성을 보여줍니다. 공식 구현 및 세포 주석은 공개적으로 제공되고 있습니다.



### Learning Inverse Laplacian Pyramid for Progressive Depth Completion (https://arxiv.org/abs/2502.07289)
- **What's New**: 이번 논문에서는 LP-Net이라는 새로운 프레임워크를 소개하고 있습니다. LP-Net은 Laplacian Pyramid 분해를 기반으로하는 다중 스케일 예측 패러다임을 구현하여, 기존의 단일 스케일 전파 접근 방식의 한계를 극복하려는 전략을 취하고 있습니다. 이 방법은 전반적인 장면 맥락을 포괄하는 저해상도 깊이 예측을 시작으로, 점진적으로 업샘플링과 고주파 세부 사항 복원을 수행합니다.

- **Technical Details**: LP-Net의 주요 구성 요소로는 Multi-path Feature Pyramid (MFP) 모듈과 Selective Depth Filtering (SDF) 모듈이 있습니다. MFP 모듈은 특성 맵을 여러 개의 경로로 분리하고, 각기 다른 변환을 적용하여 다양한 시각적 스케일에서 표현을 생성합니다. SDF 모듈은 동적으로 부드러움 필터와 선명도 필터를 적용하여, 노이즈를 줄이고 세부 사항을 강조하는 방식으로 작동합니다.

- **Performance Highlights**: LP-Net은 KITTI, NYUv2 및 TOFDC와 같은 여러 벤치마크에서 최첨단 성능을 달성했으며, 제출 시점에서 KITTI 공식 리더보드에서 1위를 기록했습니다. 이 프레임워크는 전반적으로 뛰어난 계산 효율성을 보장하면서 이전의 전파 기반 방법들보다 우수한 성능을 보여줍니다.



### KPIs 2024 Challenge: Advancing Glomerular Segmentation from Patch- to Slide-Lev (https://arxiv.org/abs/2502.07288)
- **What's New**: 이 논문에서는 만성 신장 질환(Chronic Kidney Disease, CKD)의 진단 및 치료에 대한 기준을 마련하기 위해 'Renal Pathology Image Segmentation Challenge'(KPIs)를 소개합니다. 이 도전 과제는 60개 이상의 Periodic Acid Schiff (PAS) 염색 슬라이드 이미지를 포함하여, 10,000개 이상의 주석이 달린 사구체(glomeruli)를 포함하는 방대한 데이터셋을 제공합니다. 이는 다양한 CKD 모델과 조직 상태에 적응할 수 있는 혁신적인 segmentation 방법의 개발을 장려합니다.

- **Technical Details**: KPIs 챌린지는 두 가지 주요 작업으로 구성되어 있습니다. 첫 번째 작업은 Patch-Level Segmentation으로, 주어진 이미지 패치 내의 사구체를 세분화하는 것입니다. 두 번째 작업은 Whole Slide Image-Level Segmentation 및 Detection이며, 전체 슬라이드에서 사구체를 식별하는 작업으로, 모델 성능은 Dice Similarity Coefficient (DSC)와 F1-score로 평가됩니다.

- **Performance Highlights**: 이 대회는 신장 질병 모델의 다양성을 포괄하며, 각 참가자는 특정 CKD 모델을 기반으로 한 세분화 알고리즘을 개발하여 정확한 사구체 세분화를 목표로 합니다. 이를 통해, 신장 병리학 분석의 발전과 질병 연구에 기여할 수 있는 새로운 벤치마크를 확립하는 것을 목표로 하고 있습니다.



### Articulate That Object Part (ATOP): 3D Part Articulation from Text and Motion Personalization (https://arxiv.org/abs/2502.07278)
Comments:
          Technical Report, 16 pages

- **What's New**: ATOP(Articulate That Object Part)는 텍스트 프롬프트에 따라 3D 객체의 특정 부분과 그 동작을 퍼스널라이즈하여 표현하는 혁신적인 방법을 제시합니다. 이 방법은 최신 비디오 확산 모델(video diffusion models)의 힘을 활용하여 올바른 객체 범주와 부분에 대한 신뢰할 수 있는 동작 샘플을 생성합니다. 기존의 방법들과 달리, ATOP은 사람이 주석을 달지 않은 3D 객체에 대한 잘 정의된 동작을 학습하는 프로세스를 사용합니다.

- **Technical Details**: ATOP은 몇 가지 핵심 과정을 통해 이루어집니다. 첫 번째 단계는 특정 범주에 맞춘 동작 비디오 생성에 대한 세밀한 조정(fine-tuning)입니다. 여기서는 미리 훈련된 다중 뷰 이미지 생성 모델인 ImageDream을 활용하고, 이 모델을 통해 다양한 비디오 샘플을 처리하여, 목표 객체에 대해 컨트롤 가능한 다중뷰 비디오 생성이 이루어집니다. 두 번째 단계를 통해 렌더링된 이미지와 마스크를 기반으로 동작 비디오 개인화를 수행하고, 마지막 단계에서는 개인화된 비디오 동작을 3D 객체로 변환하여 최적화합니다.

- **Performance Highlights**: ATOP은 이전 방법들과 비교하여 더 정확하고 일반화된 방식으로 3D 모션 매개변수를 예측하고 사실적인 동작 비디오를 생성할 수 있습니다. 정량적 및 정성적 실험을 통해, ATOP은 보이지 않는 형태에 대해 개인화된 사실적인 동작 비디오를 생성하는 데 성공하였으며, 동작을 3D 메쉬로 전이하는 결과 더 정확한 3D 표현을 달성하였습니다. 이는 기존의 방법들이 범위와 정확도에서 가지지 못한 장점입니다.



### Enhancing Video Understanding: Deep Neural Networks for Spatiotemporal Analysis (https://arxiv.org/abs/2502.07277)
Comments:
          29 pages, 25 figures

- **What's New**: 이 논문에서는 비디오(video) 콘텐츠를 분석하고 이해하는 알고리즘의 수요가 급증하고 있다는 점을 강조합니다. 비디오는 온라인 정보 공유의 주요 방식으로 자리 잡았고, 이러한 경향은 계속될 것으로 보입니다. 특히, 최근 딥 뉴럴 네트워크(deep neural networks)를 활용한 비디오 이해 분야의 진전을 다루고 있습니다.

- **Technical Details**: 논문은 비디오에서 발견되는 시공간(spatiotemporal) 특징들을 탐구하며, 이러한 특징들을 추출하고 분류하는 방법을 소개합니다. 비디오 이해 모델의 구조적 설계와 주요 문제, 그리고 이를 해결하기 위한 몇 가지 솔루션을 검토할 것입니다. 또한, 비디오 이해 및 동작 인식(action recognition) 데이터셋의 비교도 다루어집니다.

- **Performance Highlights**: 딥 뉴럴 네트워크를 활용한 비디오 설명 및 특징 추출에서 긍정적인 결과를 보여주었다고 평가됩니다. 이러한 알고리즘은 비디오의 사건과 객체를 효과적으로 설명하는 데 중요한 역할을 하고 있습니다. 앞으로의 연구 방향으로는 더욱 개선된 모델과 데이터셋 비교 분석이 포함될 것으로 기대됩니다.



### Exploring Active Data Selection Strategies for Continuous Training in Deepfake Detection (https://arxiv.org/abs/2502.07269)
- **What's New**: 이 논문에서는 새로운 deepfake 방법이 등장함에 따라 지속적으로 deepfake 탐지 모델을 업데이트하기 위한 소량의 추가 데이터를 자동으로 선택하는 방법을 제안합니다. 기존 탐지 모델의 신뢰도 점수를 활용하여, 신규 스푸핑 방법에 적합한 데이터를 효과적으로 선택하고, 지속적인 훈련을 통해 탐지 성능을 향상시킬 수 있도록 합니다. 실험 결과, 자동으로 선택된 소량의 데이터로 추가 훈련을 실시하였을 때, EER을 2.5%로 낮추며 성능이 유의미하게 개선됨을 보여주었습니다.

- **Technical Details**: 제안된 방법은 먼저 여러 스푸핑 방법을 포함한 초기 마스터 데이터셋 𝒰seed를 준비한 후, 이 데이터셋으로 robust deepfake 탐지 모델을 훈련합니다. 이후 신규 스푸핑 방법을 포괄하는 풀 셋 𝒰pool을 정의하고, 각 데이터 샘플에 대해 신뢰도 점수를 측정하여 신뢰도가 낮은 샘플을 선정합니다. 이렇게 선택된 유용한 데이터셋 ℐuseful을 기존의 마스터 셋에 결합하여 새로운 훈련 세트를 구성하고 모델을 지속적으로 훈련시킵니다.

- **Performance Highlights**: 제안된 방식을 통해, 기존의 마스터 데이터셋만 사용한 탐지 모델과 비교하여, 여러 신규 및 이전에 보지 못한 스푸핑 방법에 대한 탐지 정확도가 현저히 향상됨을 입증했습니다. 자동 활성 데이터 선택 방법이 무작위 선택보다 우수한 성능을 보였으며, 이는 지속적인 훈련의 유용성을 강조합니다. 최종적으로, 본 연구는 deepfake 탐지 분야의 지속 가능한 학습 메커니즘에 대한 가능성을 제시하고 있습니다.



### Robust Indoor Localization in Dynamic Environments: A Multi-source Unsupervised Domain Adaptation Framework (https://arxiv.org/abs/2502.07246)
Comments:
          19 pages, 21 figures

- **What's New**: 본 논문은 동적 실내 환경에서의 지문 기반 위치추적(Localization) 문제를 해결하기 위해 다소비자 비지도 도메인 적응(multi-source unsupervised domain adaptation, MUDA) 기반의 DF-Loc 시스템을 제안합니다. DF-Loc은 다양한 시간대의 역사적 데이터를 활용하여 특정 특징 공간에서의 지식 전이를 촉진합니다. 이로 인해 목표 도메인 내 generalization 능력이 향상되고, 라벨링된 데이터에 대한 의존도가 감소합니다.

- **Technical Details**: DF-Loc 시스템은 CSI 데이터 전처리를 위한 품질 관리(Quality Control, QC) 모듈과 CSI 지문 특징 재구성을 위한 이미지 처리 기법을 포함하고 있습니다. 그리고 다중 스케일 주의 기반 특징 융합(backbone network)을 설계하여 다중 수준의 transferable fingerprint features를 추출하며, 이중 정렬 모델을 통해 여러 출처-목표 도메인 쌍의 분포를 정렬합니다. 이러한 접근 방식은 목표 도메인에서의 회귀 특성을 개선하는 데 기여합니다.

- **Performance Highlights**: 사무실 및 교실 환경에서 진행된 광범위한 실험 결과, DF-Loc은 기존 방법들에 비해 위치 추적 정확도와 강인성을 모두 개선한 것으로 나타났습니다. 훈련에 60%의 기준 점을 사용하여 '같은 테스트' 시나리오에서 평균 위치 오차가 각각 0.79m 및 3.72m로 기록되었고, '다른 테스트' 시나리오에서는 0.94m 및 4.39m로 확인되었습니다. 이 연구는 동적 환경에서의 지문 위치 추적을 위한 종합적인 솔루션을 제공하며, 향후 연구 방향에 대한 귀중한 통찰력을 제시합니다.



### Contextual Gesture: Co-Speech Gesture Video Generation through Context-aware Gesture Representation (https://arxiv.org/abs/2502.07239)
- **What's New**: 이번 논문에서는 Contextual Gesture라는 프레임워크를 소개하여, 사람의 말과 제스처를 잘 연동하여 고해상도의 제스처 비디오 생성을 지원합니다. 이 프레임워크는 (1) 시간적으로 연관된 음성과 제스처를 정렬하고, (2) 지식 증류를 통해 방향성을 가진 제스처 표현을 학습하며, (3) 제스처의 키포인트를 연결하기 위한 구조 인식을 포함한 개선 모듈을 통해 제스처 생성 및 비디오 품질을 강화합니다.

- **Technical Details**: Contextual Gesture 프레임워크는 두 가지 주요 구성 요소를 사용하여 제스처 생성을 수행합니다. 첫째, 시간적 대비 학습을 통해 제스처와 음성 간의 내재적 연결을 발견하며, 둘째, 제스처 생성 과정에서 양방향 마스킹 프리트레이닝을 활용하여 음성과 제스처의 정렬을 정교하게 합니다. 이러한 과정은 제스처가 전달하고자 하는 의미를 선명하게 표현할 수 있도록 돕습니다.

- **Performance Highlights**: 상세한 실험 결과, Contextual Gesture는 기존의 방법들보다 향상된 정량적 및 정성적 지표를 달성하였으며, 긴 시퀀스 생성 및 비디오 제스처 편집 기능을 지원합니다. 이로 인해 현실감 있는 제스처 비디오를 생성할 수 있는 가능성이 높아져, 인간-컴퓨터 상호작용이 한층 더 발전할 것으로 기대됩니다.



### Diffusion Suction Grasping with Large-Scale Parcel Datas (https://arxiv.org/abs/2502.07238)
- **What's New**: 최근 물체 흡입 그립(grasping) 분야에서 주요한 진전을 보여주었지만, 복잡한 소포 처리 시나리오에서는 여전히 상당한 도전 과제가 남아 있습니다. 본 연구에서는 25,000개의 복잡한 장면과 4억 1천만 개의 정밀 주석이 매겨진 흡입 그립 자세를 포함하는 Parcel-Suction-Dataset을 제안합니다. 또한 Diffusion-Suction이라는 혁신적인 프레임워크를 통해 흡입 그립 예측을 조건부 생성 작업으로 재구성하였습니다.

- **Technical Details**: Diffusion-Suction은 시각적으로 조건화된 포인트 클라우드 관찰에서 얻은 지침을 통해 무작위 노이즈를 반복적으로 정제하여 흡입 그립 점수 맵을 생성합니다. 우리는 PointNet++를 이용하여 포인트 클라우드의 글로벌 정보를 추출하고, 경량화된 Denoising Block을 사용하여 중요한 특징을 강조합니다. 이 방식은 훈련 단계에서 가우시안 노이즈를 추가하고, 추론 단계에서 학습된 확산 과정을 역전시켜 신뢰할 수 있는 흡입 그립 점수를 생성합니다.

- **Performance Highlights**: Diffusion-Suction 방식은 Parcel-Suction-Dataset 및 공개 SuctionNet-1Billion 벤치마크에서 이전 모델들과 비교하여 새로운 최첨단 성능을 달성했습니다. 본 연구에 대한 실험 결과는 Diffusion-Suction이 기존 방법보다 우수함을 보여주며, 다양한 실험을 통해 그 효과와 특징을 심층 분석했습니다. 이로 인해, 복잡한 소포 장면에서도 신뢰할 수 있는 흡입 그립 예측이 가능해졌습니다.



### CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models (https://arxiv.org/abs/2502.07225)
- **What's New**: 최근 Latent diffusion 모델은 이미지 합성 작업에서 뛰어난 성능을 보여주고 있습니다. 그러나 이러한 모델을 무단 데이터로 커스터마이징하는 것은 데이터 소유자의 개인 정보와 지적 재산권을 심각하게 훼손할 수 있습니다. 본 논문은 이러한 문제를 해결하기 위해 'Contrastive Adversarial Training (CAT)' 방법을 제안하고 있습니다.

- **Technical Details**: CAT는 모델 적응에 초점을 맞춘 혁신적인 접근으로, Latent Diffusion Model의 latent autoencoder에 대해 대조적 적대적 손실을 활용하여 보호된 이미지 샘플의 재구성 손실을 최소화하는 방법입니다. 실험을 통해 CAT는 보호된 섭동 효과를 감소시키고 모델의 적응력을 향상시키며, 이는 다양한 시나리오에서 사용될 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과 CAT가 기존의 보호 섭동 기법의 효과를 상당히 줄일 수 있다는 것을 보여줍니다. 이 연구는 기존 보호 방법의 견고성을 재고하여 향상시킬 필요성을 강조합니다. 기존 연구들이 적대 공격에 대한 보호를 개발했으나, CAT는 모델 적응의 관점에서 새로운 해결책을 제시합니다.



### MLLM4PUE: Toward Universal Embeddings in Computational Pathology through Multimodal LLMs (https://arxiv.org/abs/2502.07221)
- **What's New**: 이 논문에서는 병리학 분야에서 다중 모달 임베딩을 생성하기 위한 새로운 프레임워크인 MLLM4PUE를 제안합니다. MLLM4PUE는 Multimodal Large Language Models (MLLM)을 활용하여 병리학에서 이미지와 텍스트를 통합한 범용 임베딩을 생성합니다. 이러한 접근 방식은 지금까지의 CLIP 기반 방법보다 통합된 모델 내에서 데이터를 처리하여 더 뛰어난 성능을 자랑합니다.

- **Technical Details**: MLLM4PUE는 변형기(transformer) 기반 아키텍처를 사용하여 이미지와 텍스트를 통합하여 복잡한 모달 간의 관계를 학습합니다. 또한 Pathology Multimodal Embedding Benchmark (PMEB)를 도입하여 각종 병리학 관련 작업을 평가할 수 있는 통합된 벤치마크를 제공합니다. PMEB는 검색, 분류, 복합 검색의 세 가지 메타 작업 범주를 포함하며, 14개의 데이터 세트에서 수집된 15개의 원본 작업으로 구성됩니다.

- **Performance Highlights**: MLLM4PUE의 성능 실험 결과는 기존의 모든 기준 모델을 초월하는 결과를 보여주었습니다. 특히 다중 모달 정보의 효과적인 표현 능력 덕분에 다양한 병리학 태스크에서 높은 일반화 성능을 보였습니다. 이는 향후 병리학 모델 개발의 통일성 및 효율성을 제고할 수 있는 중요한 기틀이 될 것입니다.



### SparseFormer: Detecting Objects in HRW Shots via Sparse Vision Transformer (https://arxiv.org/abs/2502.07216)
Comments:
          This paper is accepted to ACM MM 2024

- **What's New**: 이번 논문에서는 고해상도 넓은 화면 사진(High-Resolution Wide, HRW)에서 물체 감지의 정확성과 속도를 개선하기 위한 새로운 모델 SparseFormer를 제안합니다. 기존의 클로즈업 감지 모델이 HRW 환경에서 효과적이지 않은 문제를 해결하기 위해, SparseFormer는 선택적으로 주목할 수 있는 토큰(attentive tokens)을 사용하여 희소하게 분포된 창(window)에서 물체를 탐지합니다. 이로 인해 지역적 및 전역적 주의를 탐색 가능하게 합니다.

- **Technical Details**: SparseFormer는 특징 추출 시 중요성을 평가하는 ScoreNet을 학습하여, 중요한 영역에 집중할 수 있도록 합니다. 또한 HRW 이미지를 비겹치는 창으로 나누어 coarse-grained(거칠고 큰) 및 fine-grained(세밀한) 특징을 동시에 추출합니다. 새로운 Cross-slice non-maximum suppression(C-NMS) 알고리즘을 통해 노이즈가 많은 창에서 물체를 더 정밀하게 로컬라이즈하며, 효과적인 다중 스케일 전략을 사용하여 정확도를 향상시킵니다.

- **Performance Highlights**: PANDA 및 DOTA-v1.0이라는 두 개의 HRW 벤치마크 데이터셋에서 실험한 결과, SparseFormer는 기존의 최첨단 방법들보다 물체 감지 정확도를 최대 5.8%까지, 속도를 최대 3배까지 향상시켰습니다. 이러한 성과는 대규모 변화를 처리하고 다양한 크기의 물체를 정확하게 감지할 수 있는 가능성을 보여줍니다.



### PDV: Prompt Directional Vectors for Zero-shot Composed Image Retrieva (https://arxiv.org/abs/2502.07215)
- **What's New**: 최근 제안된 Zero-shot composed image retrieval (ZS-CIR) 방법은 링크된 이미지를 참조 이미지와 텍스트 프롬프트로 활용하여 이미지 검색을 수행하며, 대규모 쌍 데이터에 대한 전문적인 텍스트-이미지 구성 네트워크가 필요하지 않습니다. 기존 ZS-CIR 방법은 정적인 쿼리 임베딩 표현, 이미지 임베딩의 부족한 활용, 텍스트와 이미지 임베딩의 융합 시 최적화 부족이라는 세 가지 한계를 가지고 있습니다. 이를 해결하기 위해 Prompt Directional Vector (PDV)를 도입하여 사용자 프롬프트로 인한 의미적 수정을 캡쳐합니다.

- **Technical Details**: PDV는 세 가지 주요 개선 사항을 제공합니다: 첫째, 프롬프트 조정을 스케일링 팩터를 통해 제어할 수 있는 동적인 텍스트 임베딩을 지원합니다. 둘째, 텍스트 프롬프트에서 이미지 특징으로의 의미 전이를 통한 구성된 이미지 임베딩을 구현합니다. 셋째, 시각적 유사성과 의미적 유사성을 균형 있게 조정하며 개선된 검색 성능을 제공하는 구성된 텍스트와 이미지 임베딩의 가중 융합을 지원합니다.

- **Performance Highlights**: PDV는 기존의 최첨단 ZS-CIR 방식들과 통합될 때, 특히 정확한 구성 임베딩을 생성하는 방식에 대해 검색 성능을 일관되게 개선합니다. 다양한 벤치마크에서 진행된 광범위한 실험 결과 이러한 성과가 확인됩니다. 개발된 코드는 공개될 예정이며, 사용자들은 기존 ZS-CIR 방법에 쉽게 적용할 수 있는 이점이 있습니다.



### Playmate: Flexible Control of Portrait Animation via 3D-Implicit Space Guided Diffusion (https://arxiv.org/abs/2502.07203)
- **What's New**: Playmate는 음성 오디오 클립과 주어진 참조 정체성을 정확히 매칭하는 비디오를 생성하는 새로운 2단계 훈련 프레임워크입니다. 기존 방법들은 입술의 동기화(lip-sync), 부적절한 고개 자세(head posture) 등의 통제할 수 없는 요인으로 인해 여전히 큰 도전에 직면해 있습니다. Playmate는 이러한 요인들을 다루기 위해 얼굴 표정 및 고개 자세를 더 생동감 있게 조절할 수 있는 새로운 모듈을 도입합니다.

- **Technical Details**: 첫 번째 단계에서는 세분화된 얼굴 속성(attribute)을 더 정밀하게 분리하기 위해 분리된 암묵적 3D 표현(decoupled implicit 3D representation)과 정교하게 설계된 모션 분리 모듈(motion-decoupled module)을 도입합니다. 두 번째 단계에서는 감정 제어 모듈(emotion-control module)을 통해 감정 정보를 잠재 공간(latent space)에 인코딩하여 감정 조절의 세밀한 제어를 가능하게 합니다. 이러한 구조는 생성된 비디오에서 감정의 유연한 조작을 가능하게 합니다.

- **Performance Highlights**: Playmate는 실험 결과 기존 최첨단(SOTA) 방법들보다 비디오 품질 및 입술 동기화 면에서 우수한 성능을 보입니다. 또한 감정 및 고개 자세를 조절하는 유연성이 개선되었습니다. 이 프레임워크는 음성 기반 애니메이션을 생성하는 데 있어 새로운 가능성을 열어주며, 향후 다양한 응용 프로그램으로의 확장이 기대됩니다.



### Dense Object Detection Based on De-homogenized Queries (https://arxiv.org/abs/2502.07194)
Comments:
          17 pages, 15 figures

- **What's New**: 이 논문은 밀집 객체 탐지(dense object detection)의 어려운 문제에 초점을 맞추고 있습니다. 기존의 greedy 알고리즘을 기반으로 한 탐지 방법들은 밀집 상황에서 반복적인 예측이나 놓친 탐지를 초래하는데, 이 문제를 해결하기 위해 DETR(DEtection TRansformer)을 활용했습니다. 이 방법은 후처리의 중복 제거 능력을 네트워크에 통합하는 독창적인 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 learnable differentiated encoding을 통해 쿼리(query)의 동질성을 감소시킵니다. 이를 통해 쿼리 간의 정보 소통이 가능해지고, 기존의 쿼리 간 self-attention 방식을 대체합니다. 또한 인코더의 출력에 대해 위치(location)와 신뢰(confidence) 예측을 고려한 joint loss를 사용하여 쿼리의 초기화를 보다 높은 품질로 개선합니다.

- **Performance Highlights**: CrowdHuman 데이터셋에서 평균 정밀도(average precision) 93.6%, MR-2 39.2%, JI 84.3%를 달성하며 이전 SOTA(state-of-the-art) 방법들을 초월하는 뛰어난 성능을 보였습니다. 덧붙여, 제안된 방법은 다양한 밀도의 상황에서도 강인한 성능을 발휘합니다.



### OscNet: Machine Learning on CMOS Oscillator Networks (https://arxiv.org/abs/2502.07192)
- **What's New**: 이번 연구는 기존의 컴퓨팅 아키텍처를 대체할 새로운 에너지 효율적인 컴퓨팅 패브릭을 제안합니다. CMOS Oscillator Networks (OscNet)를 바탕으로 한 머신러닝 프레임워크는 생물학적 원리를 기반으로 한 Hebbian rule을 활용하여 신경망의 가중치를 업데이트하여 에너지를 절약합니다. 이 접근은 뇌의 스파이킹 뉴런을 모방하여 생물학적 타당성을 유지하면서도 컴퓨팅 효율성을 높입니다.

- **Technical Details**: OscNet은 고차 주입 잠금(high-order injection locking) 방식을 통해 이웃 진동기와의 연결을 통해 동작합니다. 각 진동기는 고유 주파수(f0)를 기반으로 하여 동기화되고, 정보는 개별 진동기의 위상으로 인코딩됩니다. 커뮤니케이션의 상호작용을 통해 시스템의 전반적인 에너지를 최소화하는 Potts Hamiltonian과 같은 구조를 모델링하여 최적화할 수 있습니다.

- **Performance Highlights**: 실험 결과, OscNet 상의 Hebbian 학습 파이프라인이 기존의 깊은 학습 알고리즘에 필적하거나 그 이상의 성능을 달성했습니다. 이는 머신러닝의 에너지 효율적인 대안으로서의 가능성을 강조하며, MIMO OscNet은 비지도(unsupervised) 학습 및 감독(supervised) 선형 회귀에서도 효과적으로 구현될 수 있음을 보여줍니다.



### Improved YOLOv7 model for insulator defect detection (https://arxiv.org/abs/2502.07179)
Comments:
          19 pages, 13 figures

- **What's New**: 이 논문에서는 다중 유형 절연체 결함 감지를 위한 향상된 YOLOv7 모델을 제안합니다. 기존의 연구들이 단일 결함 유형이나 특정 재료에 집중한 반면, 본 연구는 다양한 색상과 재료를 가진 절연체 결함을 동시에 처리할 수 있는 방법을 모색합니다. 이로 인해 실용적인 적용 требований을 더욱 충족할 수 있을 것으로 기대됩니다.

- **Technical Details**: 제안된 모델에서는 SPPCSPC 모듈을 RFB 모듈로 교체하여 네트워크의 feature extraction (특징 추출) 능력을 향상시킵니다. 또한, CA 메커니즘이 head 부분에 도입되어 네트워크의 feature representation (특징 표현) 능력을 강화하고, 이를 통해 감지 정확도를 높입니다. 최종적으로 WIoU loss function (손실 함수)를 사용하여 훈련 중 모델 일반화를 방해하는 저품질 샘플 문제를 해결하고, 모델의 전체 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다양한 성능 지표에서 개선된 성과를 보였습니다. 특히, mAP_0.5가 1.6% 향상되었고, mAP_0.5:0.95에서도 1.6% 증가하였습니다. 또한, precision (정확도)은 1.3%, recall (재현율)은 1% 증가하였으며, 모델의 파라미터 수는 320만 감소하여 2.5 GFLOPS의 계산 비용 절감 효과도 나타났습니다. 단일 이미지 감지 속도는 2.81 밀리초 개선되었습니다.



### Foreign-Object Detection in High-Voltage Transmission Line Based on Improved YOLOv8m (https://arxiv.org/abs/2502.07175)
Comments:
          24 pages, 16 figures

- **What's New**: 이 논문에서는 고전압 송전선의 안전한 운영을 보장하기 위한 외국물체(foreign objects) 탐지에 관한 새로운 방법을 제안합니다. 구체적으로, Yunnan 전력망(Yunnan Power Grid)에서 수집된 데이터셋을 활용하여 YOLOv8m 모델을 개선한 기술을 소개합니다. 이 개선된 모델은 Global Attention Module (GAM)을 사용하여 방해물(occlusions)에서 외국물체에 집중하게 하며, 다양한 스케일에서의 특징(feature) 추출 능력을 향상시킵니다.

- **Technical Details**: 제안된 모델은 원래 YOLOv8m의 SPPF 모듈을 SPPCSPC 모듈로 교체하여 멀티스케일(multiscale) 특성을 강화하며, Focal-EIoU 손실 함수(loss function)를 도입하여 고품질 및 저품질 샘플 간의 불균형 문제를 해결합니다. 이와 같은 개선으로 모델의 수렴 속도를 가속화 시키고 탐지 정확도를 높입니다. 특히, 새로운 측정 지표를 통해 복잡한 배경에서의 다양한 물체를 효과적으로 탐지할 수 있는 능력을 갖추게 되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 mAP_0.5 지표에서 2.7%, mAP_0.5:0.95에서 4%, 그리고 리콜(recall)에서 6%의 개선을 달성했습니다. 이는 기존 방법에 비해 유의미한 성과로, 실용적인 검사의 수행을 위한 기반 기술로 자리매김할 수 있음을 보여줍니다. 이러한 성과는 고전압 송전선의 안전성을 크게 향상시킬 것으로 기대됩니다.



### SemiHMER: Semi-supervised Handwritten Mathematical Expression Recognition using pseudo-labels (https://arxiv.org/abs/2502.07172)
Comments:
          12 pages,3 figures

- **What's New**: 이번 논문에서는 손글씨 수학 표현 인식(HMER)의 성능을 향상시키기 위한 새로운 반지도 학습(semisupervised learning) 프레임워크를 제안합니다. 두 개의 분기에서 서로의 예측을 의사 라벨(pseudo-label)로 사용하여 학습하는 이중 분기 접근 방식을 도입하여, 제한된 라벨링 데이터에서도 효율적으로 성능을 개선할 수 있게 했습니다. 또한, 변화 수준에 따른 다양한 데이터 증강(augmentation) 기법을 적용하여 두 분기의 학습 과정을 동기화하고, 최적화를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 약-강(adjacent-strong) 학습 전략과 함께 글로벌 동적 카운팅 모듈(Global Dynamic Counting Module, GDCM)을 도입합니다. GDCM은 지난 예측 결과에 기반하여 카운팅 벡터를 동적으로 업데이트하며, 이를 통해 긴 거리의 수식 인식에서 발생할 수 있는 오류를 감소시키고 반복 문자의 인식 정확성을 향상시킵니다. 이 논문에서는 시그모이드 함수와 비슷한 치료법을 통해 서로 다른 학습 경로를 통합하여 성능을 높였습니다.

- **Performance Highlights**: 제안된 Semi-supervised pseudo-supervision 방법은 기존 지도 학습 방법보다 높은 인식 정확도를 달성하였습니다. 강화된 데이터 증강 기술을 통해 훈련 데이터의 활용성을 극대화하고, 논문의 실험 결과에서 손글씨 수식 인식 분야에서 의미 있는 성과가 있음을 보여줍니다. 또한, GDCM 모듈을 통해 긴 거리의 수식 인식 과정에서 발생하는 문제를 효과적으로 해결하여, 더욱 향상된 인식 품질을 안전성으로 입증하였습니다.



### A Survey on Mamba Architecture for Vision Applications (https://arxiv.org/abs/2502.07161)
- **What's New**: Mamba 아키텍처는 비주얼 태스크에서의 Transformer의 한계를 극복하기 위해 설계되었으며, 특히 attention 메커니즘의 기하급수적인 복잡성을 해결합니다. 이 연구는 Mamba 아키텍처를 비주얼 도메인 응용 프로그램에 적용하는 것을 목표로 하며, Vision Mamba (ViM) 및 VideoMamba와 같은 최근의 발전을 다룹니다. 이를 통해 양방향 스캐닝 및 선택적 스캐닝 메커니즘을 도입하여 이미지 및 비디오 이해력을 향상시킵니다.

- **Technical Details**: Mamba 아키텍처는 상태공간모델(state-space models, SSMs)을 활용하여 선형 확장성(linear scalability)을 달성합니다. 또한 이 아키텍처는 위치 임베딩(position embeddings), 교차 스캔 모듈(cross-scan modules), 및 계층적 설계(hierarchical designs)를 통합하여 전역 및 지역 특징 추출을 최적화합니다. 이러한 혁신은 Mamba 프레임워크의 효율성을 높이고 맥락 인식을 개선하는 데 기여합니다.

- **Performance Highlights**: Mamba는 이미지 및 비디오의 이해도를 높이기 위한 spatiotemporal processing을 지원하여 최첨단 성능을 입증합니다. 이러한 기술적 혁신들은 Mamba를 컴퓨터 비전 연구 및 응용 분야의 유망한 아키텍처로 위치시킵니다. 결과적으로 Mamba 아키텍처는 이전의 모델들에 비해 더 뛰어난 처리 능력과 효율성을 보여줍니다.



### HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates (https://arxiv.org/abs/2502.07160)
Comments:
          Under Review

- **What's New**: 이 논문에서는 하이브리드 확산 이미지 압축(Hybrid-Diffusion Image Compression, HDCompression)을 제안합니다. 이 방법은 생성적 벡터 양자화(generative VQ) 모델링과 확산 모델(diffusion models), 그리고 기존의 학습된 이미지 압축(conventional learned image compression, LIC)을 통합하여 높은 충실도(fidelity)와 인지 품질(perceptual quality)을 달성합니다. 기존 하이브리드 방법들이 단순히 사전 학습된 LIC 모델을 사용하여 양자화된 정보로부터 저품질 정보를 생성했던 것과는 달리, HDCompression은 실제 입력으로부터 고품질의 정보를 추출합니다.

- **Technical Details**: HDCompression은 두 개의 스트림을 활용하는 체계로, 확산 모델을 통해 고품질의 보완 정보를 추출합니다. 이 모델은 이용 가능한 데이터를 활용하여 압축 성능을 향상시키고 있습니다. 또한, 밀집 대표 벡터(dense representative vector, DRV) 기반의 단순한 샘플링 스케줄러를 적용하였습니다.

- **Performance Highlights**: 연구 결과, HDCompression은 기존의 LIC, 생성적 VQ 모델링, 그리고 기존 하이브리드 프레임워크에 비해 정량적 지표와 정성적 시각화 모두에서 우수한 성능을 보였습니다. 특히, 이 방법은 극저 비율(ultra-low bitrates)에서도 안정적인 압축 성능을 제공합니다. 이러한 개선 사항은 지수 맵 예측(index map prediction) 향상, LIC 스트림의 충실도 유지(output of the LIC stream) 개선 및 VQ-잠재 부가 수정된 조건부 이미지 재구성(refining conditioned image reconstruction) 등에 기여하고 있습니다.



### Explaining 3D Computed Tomography Classifiers with Counterfactuals (https://arxiv.org/abs/2502.07156)
Comments:
          Code and models: this https URL

- **What's New**: 본 연구는 Latent Shift 방법을 2D 계산에서 3D 컴퓨터 단층촬영(CT) 스캔으로 확장하여, 의료 이미징에서의 반사실(counterfactual) 설명 생성을 향상시킵니다. 저자들은 제한된 훈련 샘플과 높은 메모리 요구 사항을 해결하기 위해 슬라이스 기반 접근법을 구현하여, 해석 가능한 반사실을 생성하는 데 효과적임을 보여줍니다.

- **Technical Details**: Latent Shift 방법을 사용하여 반사실을 생성하는 과정에서, 2D 오토인코더를 통해 3D 볼륨을 슬라이스 단위로 인코딩하고 디코딩하여 처리합니다. 이는 메모리 사용량을 줄이고, 고해상도 3D 의료 이미징에서도 해석 가능한 반사실을 생성할 수 있게 합니다. 기울기(gradient)는 전체 볼륨이 아닌 슬라이스의 하위 집합에 대해서만 전파되며, 이를 통해 메모리 제약을 극복합니다.

- **Performance Highlights**: 연구는 두 가지 예측 모델에 대해 슬라이스 기반 접근법의 효율성을 입증하였으며, 높은 품질의 반사실을 생성하는 데 있어 만족스러운 성능을 보이았습니다. lung segmentation 및 임상 표현형 예측을 위한 모델이 성공적으로 구현되었고, 각 모델은 CT 스캔에서 중요한 특징을 잘 포착하여 예측을 수행하는 데 효과적임을 나타냈습니다.



### Mesh2SSM++: A Probabilistic Framework for Unsupervised Learning of Statistical Shape Model of Anatomies from Surface Meshes (https://arxiv.org/abs/2502.07145)
- **What's New**: 새로운 접근 방식인 Mesh2SSM++는 표면 메쉬로부터 우편 슈퍼바이즈드 학습 없이 형태 모델을 수립하는 혁신적인 방법입니다. 이 방법은 메쉬에서 변형된 템플릿 포인트 클라우드로 주제별 메쉬를 형성하여 대응 판별에 관한 확률적 형상 모델을 제공합니다. 또한, aleatoric uncertainty를 정량화할 수 있어 임상 작업에서 모델 예측의 신뢰성을 보장합니다.

- **Technical Details**: Mesh2SSM++는 연속 정규화 흐름(Continuous Normalizing Flow)을 통합하여 확률적 형태 모델링의 잠재 공간을 향상시킵니다. 이를 통해 저품질 샘플의 생성을 줄이고, 메쉬 표면 위에 예측된 대응리스트가 위치하도록 유도하는 새로운 손실 함수를 도입합니다. 셀프 슈퍼바이즈드 학습 기법을 적용하여 모델이 노이즈가 있는 데이터에 더 강건하고 일반화할 수 있도록 향상하여, 다양한 해부학적 데이터셋에 대해 포괄적인 실험을 진행했습니다.

- **Performance Highlights**: Mesh2SSM++는 기존 메서드보다 우수한 성능을 나타내며 임상 데이터를 다룰 때 신뢰할 수 있는 예측이 가능합니다. 다양한 해부학적 구조에 대한 SSM의 적용 범위를 확장하며, 메쉬를 직접적으로 효과적으로 다룰 수 있는 능력과 계산 효율성을 갖추고 있습니다. 이러한 특성 덕분에 Mesh2SSM++는 전통적인 및 딥러닝 기반 SSM 방법에 대한 대안으로 매력적인 선택이 됩니다.



### Few-Shot Multi-Human Neural Rendering Using Geometry Constraints (https://arxiv.org/abs/2502.07140)
- **What's New**: 이번 연구에서는 다수의 사람으로 구성된 장면의 형태(shape)와 방사선(radiance)을 복원하는 새로운 방법을 제안합니다. 기존 단일 인물 환경에서의 신경망 기반 접근법들이 인상적인 결과를 보였지만, 여러 인물을 추정하는 것은 여전히 도전적입니다. 우리의 방법은 SMPL 인체 모델을 통해 사전 계산된 메쉬를 활용하여 기하학적 제약(geometry constraints)을 적용하고, 광선 정규화(ray regularization) 및 조도 변수에 대한 강건한 최적화를 통해 문제를 해결합니다.

- **Technical Details**: 우리는 SMPL 인체 모델을 활용하여 입력 데이터로부터 인체 메쉬를 초기화한 후, 이 메쉬를 기준으로 다인체 장면의 표면을 정의합니다. 이후, 다중 뷰 이미지를 사용하여 기하학 네트워크를 최적화하며, 표면 및 볼륨 렌더링(volume rendering)과 불확실성 추정 방법을 결합합니다. 추가적으로, 다양한 광원 상태에서도 일관성을 보장하기 위한 패치 기반 정규화 손실(patch-based regularization loss)과 포화 정규화(saturation regularization)를 제안하여 렌더링 품질을 향상시킵니다.

- **Performance Highlights**: 우리는 CMU Panoptic 및 MultiHuman 데이터셋을 통해 제안한 방법을 평가했습니다. 5, 10, 15 및 20개의 훈련 뷰를 이용하여 표면 복원(surface reconstruction)과 새로운 뷰 품질(novel view quality)에서 최첨단 성능을 달성했습니다. 우리의 접근법은 다중 인체 상황에서 특히 높은 정확성과 일관성을 보여주며, 기존 신경 재구성 방법들에 비해 현저한 향상을 입증합니다.



### Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Conten (https://arxiv.org/abs/2502.07138)
Comments:
          Accepted to the MM4SG Workshop at the WebConf 2025

- **What's New**: 이 논문은 다중 모달(나이브) 증오 콘텐츠 탐지를 위해 융합 기반 접근법을 체계적으로 분석합니다. 영상 및 이미지 콘텐츠의 성능을 집중적으로 평가하며, 단순 임베딩 융합이 HateMM 데이터셋에서 9.9% F1 점수를 개선한 반면, Hateful Memes 데이터셋에서는 복잡한 이미지-텍스트 관계를 포착하지 못하는 한계를 드러냅니다. 이를 통해 다중 모달 접근 방식에서 발생하는 이해 부족을 강조합니다.

- **Technical Details**: 현재 연구는 주로 단일 모달 증오 콘텐츠 탐지에 집중되어 있으며, 이러한 접근 방식들이 다양한 모달의 조합에 대해 효과적이지 않음을 보여줍니다. 이러한 연구는 이미지, 텍스트, 오디오 및 비디오를 포함한 모든 모달 생태계의 상호 작용을 포착하는 포괄적인 프레임워크 필요성을 제기합니다. 더 나아가, 논문은 HateMM 및 Hateful Memes Challenge 데이터셋에서 두 가지 접근 방식을 비교하여 모달리티별 한계를 평가합니다.

- **Performance Highlights**: HateMM 데이터셋에서 단순 임베딩 융합이 최첨단 성능을 달성했지만, 이미지를 포함한 복합적인 의미 관계를 미세하게 포착하지 못했다는 문제를 드러냈습니다. 이 연구는 현재 접근 방식의 성과를 체계적으로 비교하며, 다양한 모달 조합 간의 성공 및 실패를 분석하여 보다 강력한 증오 탐지 시스템 개발을 위한 기초적인 통찰력을 제공하고 있습니다.



### Unconstrained Body Recognition at Altitude and Range: Comparing Four Approaches (https://arxiv.org/abs/2502.07130)
- **What's New**: 이 연구에서는 사람의 장기 식별(long-term person identification) 방법에 대한 네 가지 접근 방식을 조사하였습니다. 기존의 단기 재식별 시스템(short-term re-identification systems)과는 달리, 우리는 영구적인 체형 특성을 학습하는 데 중점을 두고 있습니다. Vision Transformer (ViT) 및 Swin-ViT 모델을 기반으로 한 신체 식별 모델을 소개하며, 시각적 정보의 맥락을 활용하여 장기적인 정확성을 목표로 합니다.

- **Technical Details**: 이 연구의 모델들은 약 190만 개의 이미지와 5천 개의 개체가 포함된 대규모 데이터셋에서 학습되었습니다. 우리는 Linguistic and Non-linguistic Core ResNet Identity Models (LCRIM, NLCRIM)의 개선된 버전을 개발하였으며, 다양한 아키텍처와 입력 이미지 크기가 장기 신체 식별 성능에 미치는 영향을 분석하였습니다. 평가에 사용된 벤치마크 데이터셋은 MARS, MSMT17, Outdoor Gait, DeepChange 등이 포함되어 있으며, 제한 없는 데이터셋에서의 성능도 검토하였습니다.

- **Performance Highlights**: 모델 성능은 실제 환경에서의 다양한 조건을 고려하여 평가되었습니다. 특히, 장거리(최대 1000m), 고도(무인 항공기 사용) 및 의류 변화가 포함된 이미지에 대한 식별 정확도가 강조되었습니다. 이러한 비교 분석은 다양한 백본 아키텍처(backbone architectures)와 입력 이미지 크기가 장기 신체 식별 성능에 미치는 영향을 드러내어, 실세계 응용 프로그램에서 모델의 강점을 밝혀줍니다.



### Is Long Range Sequential Modeling Necessary For Colorectal Tumor Segmentation? (https://arxiv.org/abs/2502.07120)
Comments:
          5 pages, 1 figures

- **What's New**: 이번 연구에서는 장암(Colorectal Cancer, CRC) 종양 세분화에 있어 최근 주목받고 있는 Transformer 및 Mamba와 같은 장기 시퀀스 모델링 메커니즘을 평가합니다. 특히, MambaOutUNet을 새로운 데이터세트 CTS-204와 함께 사용하며, 국소 토큰 상호작용이 작은 종양의 경우 장기 모델링 기술보다 더 나은 성능을 보일 수 있음을 밝혀냈습니다. 이는 3D 종양 세분화 연구의 새로운 방향성을 제안하는 결과입니다.

- **Technical Details**: 연구에서 제안한 MambaOutUNet 구조는 향상된 지역 및 전역 토큰 상호작용 능력을 갖춘 다양한 아키텍처를 비교 분석하고 있습니다. Mamba는 선택 메커니즘을 통해 장기 의존성을 모델링 하며, 더 효율적인 채널 혼합 및 공간적으로 제어된 기능이 기존의 계산 집약적인 장기 모델링 기술보다 더 효과적일 수 있음을 보여줍니다. 또한, Tri-oriented Spatial Mamba (TSMamba) 블록이 글로벌 토큰 모델링을 지원하여 다중 스케일 특징 확보에 기여합니다.

- **Performance Highlights**: CTS-204 데이터세트에서 MambaOutUNet의 성능을 다양한 기준과 기존 3D 세분화 아키텍처와 비교하였으며, MambaOutUNet이 효과적으로 높은 정확도를 달성함을 확인했습니다. 기존의 길고 복잡한 시퀀스 모델링 기법에 비해 빠르고 효율적으로 더 정확한 세분화 결과를 도출할 수 있는 가능성을 보여주었습니다. 이 연구 결과는 CRC 종양의 효과적인 치료 계획 수립과 생존 결과 평가에 기여할 것으로 기대됩니다.



### PrismAvatar: Real-time animated 3D neural head avatars on edge devices (https://arxiv.org/abs/2502.07030)
Comments:
          8 pages, 5 figures

- **What's New**: 새로운 모델인 PrismAvatar는 리소스가 제한된 엣지 디바이스에서 실시간 애니메이션 및 렌더링을 가능하게 하는 3D 헤드 아바타 모델입니다. 이 모델은 전통적인 트라이앵글 렌더링 파이프라인의 이점을 활용하여 학습 시에는 복합 기하 구조를 적용하면서도 모바일 장치에서 60fps로 실행됩니다. PrismAvatar는 고급 3D 아바타 모델과 비교해도 유사한 품질을 제공합니다.

- **Technical Details**: PrismAvatar는 링 구조와 3D 변형 모델을 통합하여 동적 3D 신경 아바타를 구현합니다. 이 방법은 복합적인 기하학적 세부 정보를 명시적으로 모델링할 필요가 없는 신경 방사 필드(NeRF)의 이점을 이용합니다. Hybrid 렌더링 모델을 통해 메쉬 기반 헤드와 deformable NeRF 모델을 동시에 복원하여 효율적으로 애니메이션을 구현합니다.

- **Performance Highlights**: 모바일 환경에서도 뛰어난 성능을 발휘하는 PrismAvatar는 낮은 메모리 사용량으로 60fps에서 작동하며, 복잡한 기하학을 가진 영역에서도 신뢰할 수 있는 결과를 제공합니다. 이러한 성과는 기존의 GPU 사용 제약을 극복하고, 다양한 플랫폼에서 블렌드 쉐이프 및 리니어 블렌드 스키닝을 통합하여 애니메이션의 효율성을 높입니다.



### Early Operative Difficulty Assessment in Laparoscopic Cholecystectomy via Snapshot-Centric Video Analysis (https://arxiv.org/abs/2502.07008)
Comments:
          Accepted at IPCAI, 2025

- **What's New**: 이번 논문에서는 제한된 영상 데이터를 활용하여 담낭절제술(Laparoscopic Cholecystectomy, LC) 동안의 수술 난이도(Laparoscopic Cholecystectomy Operative Difficulty, LCOD)를 조기에 평가하는 새로운 임상 과제를 제안합니다. SurgPrOD라는 심층 학습(deep learning) 모델을 설계하여 수술 영상의 전반적 및 국소적 시간적 해상도에서 특징을 분석함으로써 LCOD를 평가합니다. 또한, 이러한 평가를 향상시키기 위해 새로운 스냅샷 중심 주의(snapshot-centric attention, SCA) 모듈을 도입하였습니다.

- **Technical Details**: SurgPrOD는 수술이 진행되는 초기 몇 분의 제한된 비디오를 분석하여 LCOD를 평가하는데, 이는 다양한 시간적 해상도에서 수술 비디오의 스냅샷을 활용합니다. 우리는 기본적인 측정인 상위-1 정확도(top1-accuracy)와 F1 점수 외에 예측의 조기성과 안정성을 평가하기 위한 새로운 메트릭인 Earliness Stability(ES)를 제안했습니다. CholeScore라는 고유한 데이터셋을 생성하여, 3개의 임상 LCOD 평가 척도에 대한 성공적인 검증을 수행하였습니다.

- **Performance Highlights**: SurgPrOD는 CholeScore 데이터셋에서 3개의 LCOD 평가 척도를 기반으로 평가되었으며, 기존 방법들에 비해 최소 0.22점 향상된 조기 및 안정적인 올바른 예측을 보여주었습니다. 또한, F1 점수와 최상위 정확도에서 각각 최소 9 및 5 퍼센트 포인트를 개선하였다는 결과를 얻었습니다. 이러한 성과는 수술 비디오 데이터를 사용하여 LCOD 연구에 있어 새로운 기준을 제시하였습니다.



### Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC (https://arxiv.org/abs/2502.07007)
- **What's New**: 최근 AI 생성 콘텐츠의 발전은 3D 및 4D 생성의 현실감을 크게 향상시켰습니다. 그러나 기존 방법들은 주로 외관의 일관성(appearance consistency)만을 우선시하며, 기본 물리 원리(physical principles)를 무시하여 비현실적인 변형(unrealistic deformations), 불안정한 동역학(unstable dynamics), 그리고 그럴듯하지 않은 객체 간 상호작용을 초래했습니다.

- **Technical Details**: 본 논문에서는 3D 및 4D 생성에 물리 제약(physical constraints)을 통합한 생성 모델(generative models)에 대한 조사 결과를 제공합니다. 정적(static) 및 동적(dynamic) 3D 생성에 물리적 선행(priors)을 통합한 최근 연구를 검토하며, 방법론을 비전 기반(vision-based), NeRF 기반, 가우시안 스플래팅(Gaussian Splatting) 방식으로 분류합니다. 4D 생성에서는 시간적 동역학(temporal dynamics)을 물리적 시뮬레이션(physical simulations)으로 모델링하는 방법론을 탐색합니다.

- **Performance Highlights**: 주요 방법들에 대한 비교 분석을 통해 각 방법의 강점(strengths), 한계(limitations), 및 다양한 재료(materials)와 동작 동역학(motion dynamics)에 적합성을 강조합니다. 이 논문은 물리적 현실(physical realism)과 생성 모델 간의 간극을 메우기 위한 심층 분석을 제공하며, 물리적으로 일관된 콘텐츠 생성(physically consistent content generation)을 위한 향후 연구에 영감을 줄 수 있는 통찰력을 제공합니다.



### AstroLoc: Robust Space to Ground Image Localizer (https://arxiv.org/abs/2502.07003)
- **What's New**: 이번 연구에서는 Astronaut Photography Localization (APL) 분야에서 최초로 우주비행사 사진을 활용해 훈련할 수 있는 파이프라인을 제시했습니다. 이를 통해 300,000장의 약한 라벨을 가진 우주비행사 사진의 정밀 로컬라이제이션 정보를 생성하고, AstroLoc이라는 모델로 학습시켰습니다. 이러한 접근 방식은 기존의 APL 방법보다 기억률(recall)에서 35% 평균 향상을 이루어냈습니다.

- **Technical Details**: 연구는 약한 라벨이 부착된 300,000장의 우주비행사 사진을 강력한 메트릭 학습 프레임워크에 통합하는 방법을 다룹니다. 두 가지 손실(loss) 함수를 사용하여 우주비행사 사진과 이에 대응하는 위성 사진의 쌍을 비교하고, 우주비행사 사진의 분포를 살펴봐야 합니다. 이 과정에서 생성된 모델인 AstroLoc은 기존의 데이터셋에서 상한선에 도달하며 99% 이상의 recall@100을 기록합니다.

- **Performance Highlights**: AstroLoc은 Testing set에서 다양한 실제 APL 작업에 대한 새로운 도전 과제를 제시합니다. 이 모델은 수많은 우주비행사 사진들을 로컬라이즈하는 데 이미 사용되었으며, 앞으로의 과제인 lost-in-space 위성 문제와 역사적 우주 이미지 로컬라이제이션에 광범위한 활용 가능성을 보여줍니다. ISS의 사진 로컬라이제이션 backlog가 몇 개월 이내로 거의 사라질 것으로 기대됩니다.



### From Image to Video: An Empirical Study of Diffusion Representations (https://arxiv.org/abs/2502.07001)
- **What's New**: 이 논문은 비디오 생성 및 이미지 생성을 위한 동일한 모델 아키텍처를 비교하여 비디오 확산 모델(video diffusion models)이 인간의 비주얼 이해를 위한 다양한 다운스트림 작업에서 우수하다는 점을 강조합니다. 특히, 이 연구는 이미지 확산 모델(image diffusion models)과 비디오 확산 모델 간의 내부 표현을 분석하고 비교하는 첫 번째 작업으로, 비디오 모델이 모션과 공간 장면 이해 간의 상호작용을 어떻게 포착하는지에 대한 통찰을 제공합니다.

- **Technical Details**: 논문에서는 WALT라는 하이브리드 아키텍처를 가진 확산 모델을 사용하여 비디오와 이미지를 위한 사전 훈련 목표의 차이가 다운스트림 성능에 미치는 영향을 분석합니다. 기반 모델은 Latent Diffusion Model(잠재 확산 모델)이며, 이는 VQ-VAE(벡터 양자화 변이 오토인코더)에서 작동하여 계산 요구사항을 크게 줄이는 장점을 가집니다. 실험은 이미지 분류, 동작 인식, 깊이 추정 및 물체 추적을 포함한 다양한 작업에 대한 성능을 포함합니다.

- **Performance Highlights**: 비디오 확산 모델은 이미지 모델보다 다양한 작업에서 일관되게 뛰어난 성능을 보였으며, 이러한 우수성의 정도에는 주목할 만한 범위가 존재하는 것으로 나타났습니다. 특히, 훈련 예산, 모델 크기 및 생성 시각 품질 간의 관계를 분석하였으며, 이는 비디오와 이미지 모델 간의 성능 차이에 대한 정보를 제공합니다. 이 연구 결과는 동적 장면 이해에서 비디오 확산 모델의 잠재적 가치를 입증합니다.



### Indoor Light and Heat Estimation from a Single Panorama (https://arxiv.org/abs/2502.06973)
- **What's New**: 본 논문은 실내외에서 캡처한 High Dynamic Range (HDR) 파노라마를 사용하여 실내 조명과 열 지도를 직접 추정하는 새로운 응용 프로그램을 제시합니다. 이미지 기반 렌더링 방법을 통해 실내 파노라마를 이용해 3D 방 배치를 추정하며, 해당 야외 파노라마는 환경 맵으로 사용되어 공간적으로 변동하는 빛과 재료 속성을 추론하는 데 활용됩니다. 이 연구는 실내 빛 전송과 열 전송 간의 연결을 확립하였으며, 이를 통해 자동으로 실내의 빛과 열을 추정할 수 있는 디지털 응용 프로그램을 구현하였습니다.

- **Technical Details**: 이 연구에서는 캡처된 실내-외 HDR 파노라마를 입력으로 활용하여, 실내 재료, 3D 기하학 및 조명을 물리 기반 건축 분석을 위해 추정하는 이미지 기반 렌더링 접근 방식을 도입하였습니다. 또한, 비정상 열 방정식을 구현하여 실내 열 지도를 추정하는 응용 프로그램을 개발하였습니다. 여기에는 다양한 열 매개변수에 대한 민감도 분석을 수행하고, 생성된 가상 열 지도를 실제 열 이미지를 이용해 비교하는 작업이 포함됩니다.

- **Performance Highlights**: 결과적으로, 이 연구는 단일 실내 파노라마로 조명 및 난방 속성을 동시에 추정할 수 있는 가능성을 보여주었습니다. 실제 열 이미지를 통해 새로운 열 맵의 정확성을 평가함으로써, 자동 조명 및 열 추정의 효율성을 입증하였습니다. 이러한 접근법은 실내 조명과 난방을 최적화하고, 사용자에게 더 나은 환경 설계를 가능하게 합니다.



### GAS: Generative Avatar Synthesis from a Single Imag (https://arxiv.org/abs/2502.06957)
- **What's New**: 이 논문에서는 단일 이미지에서 일관된 시점(view)과 시간적으로 일관된 아바타를 생성하는 통합 프레임워크를 소개합니다. 기존의 diffusion 모델들이 인간의 깊이(depth)나 법선(normal) 맵을 기반으로 생성하면서 발생하는 외모 정보의 손실 문제를 해결하고자 합니다. 이 방법은 회귀 기반의 3D 인간 복원 기술과 생성 모델인 diffusion 모델을 통합하여 인물의 구조와 외형을 정확하게 유지하며 고품질의 아바타를 생성합니다.

- **Technical Details**: 제안된 방법은 3D 인간 복원 모델을 통해 중간 시점(view)이나 자세(pose)를 생성하고, 이를 비디오 diffusion 모델의 조건 입력으로 활용합니다. 이를 통해 3D 복원에서 얻은 밀도 높은 정보가 전달되어 구조적 정확성과 시각적 세부정보를 잘 유지하면서 일관된 결과를 도출합니다. 또한, 다양한 실세계 데이터를 활용하기 위해 인터넷 비디오를 포함하여 새로운 자세 및 시점을 동시 학습하는 통합 프레임워크를 제안합니다.

- **Performance Highlights**: 실험 결과는 이 방법이 통제된 데이터셋뿐 아니라 실세계 데이터에서도 우수한 일반화 성능을 보여주고 있음을 입증했습니다. 이 통합 프레임워크의 도입으로 아바타의 시점과 자세가 서로 연관되어 향상된 품질을 보입니다. 전반적으로, 이 논문은 아바타 생성의 접근성과 품질을 크게 향상시키는 방법론을 제시합니다.



### AI-Driven HSI: Multimodality, Fusion, Challenges, and the Deep Learning Revolution (https://arxiv.org/abs/2502.06894)
Comments:
          39 Pages, 22 figures, 20 tables

- **What's New**: 이번 연구는 하이퍼스펙트럼 이미징(HSI)의 최신 동향과 깊은 학습(Deep Learning) 모델의 역할을 심층적으로 검토하였습니다. HSI 기술은 기상 모니터링, 식품 품질 제어, 가짜 탐지 등 다양한 분야에서 필수적이며, 최근 AI와의 융합이 두드러진 특징입니다. 특히, 대형 언어 모델(LLMs)과의 통합을 통해 시각적 인식 및 정보 제공을 통한 새로운 가능성을 제시하고 있습니다.

- **Technical Details**: 하이퍼스펙트럼 이미징은 전통적인 시스템으로는 탐지할 수 없는 공간적 및 스펙트럼적 특성을 분석할 수 있습니다. HSI는 높이, 너비, 파장을 포함하는 3차원 하이퍼큐브(hypercube)로 저장되어 깊이 있는 물질 분석이 가능합니다. 최근의 연구에서는 CNNs (Convolutional Neural Networks), GANs (Generative Adversarial Networks)와 같은 다양한 심층 학습 모델들이 HSI 과제를 해결하는 데 사용되고 있음을 보여주고 있습니다.

- **Performance Highlights**: AI와 HSI의 융합을 통한 성능 향상이 두드러지며, 특히 객체 탐지 및 분류의 정확도가 크게 향상되었습니다. 산업 내 HSI의 연평균 성장률(CAGR)은 계속 증가하고 있으며, 이는 의료, 환경 모니터링, 국방 등 여러 분야에서 HSI의 중요성이 강조됨을 나타냅니다. 결국, 본 연구는 기술적 및 비기술적 독자 모두에게 유익한 정보를 제공하며, 향후 HSI 및 딥러닝 모델의 발전 방향에 대한 통찰력을 제시합니다.



### A New Hybrid Intelligent Approach for Multimodal Detection of Suspected Disinformation on TikTok (https://arxiv.org/abs/2502.06893)
- **What's New**: 이 연구는 TikTok 비디오에서 의심되는 허위 정보를 탐지하기 위한 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 딥 러닝(deep learning)의 계산 성능과 퍼지 로직(fuzzy logic)의 해석 가능성을 결합하여 허위 정보 탐지의 새로운 접근 방식을 제안합니다. 특히 이 시스템은 텍스트, 오디오 및 비디오 데이터에서 멀티모달 기능을 분석 및 평가하는 두 가지 주요 구성 요소로 구성됩니다.

- **Technical Details**: 제안된 방법론은 텍스트, 이미지, 오디오에서 멀티모달 기능을 추출하는 멀티모달 기능 분석기(multimodal feature analyser)와 퍼지 로직 기반의 멀티모달 허위 정보 탐지기(multimodal disinformation detector)로 구성됩니다. 이 두 시스템은 사람의 행동 신호인 바디랭귀지(body language), 언어 패턴(speech patterns), 텍스트 일관성(text coherence)을 기반으로 허위 정보의 의심스러움을 평가합니다.

- **Performance Highlights**: 두 가지 실험이 진행되었으며, 첫 번째는 특정 문맥에 대한 허위 정보 사용자를 식별하는데 중점을 두고, 두 번째는 모델이 더 넓은 주제로 확장 가능성을 평가합니다. 평가된 각 비디오에 대해 고품질의 잘 구조화된 보고서가 생성되어 허위 정보 행동을 자세히 파악할 수 있습니다.



### Secure Visual Data Processing via Federated Learning (https://arxiv.org/abs/2502.06889)
Comments:
          12 Pages, 3 figures, 5 tables

- **What's New**: 본 논문은 객체 감지, 연합 학습(Federated Learning), 익명화(anonymization)를 결합하여 시각적 데이터의 개인 정보 보호를 위한 새로운 접근 방식을 제안합니다. 기존 연구들은 대개 객체 감지와 익명화 또는 연합 학습을 별개로 다루어왔으나, 본 연구는 세 가지 요소를 통합하여 민감한 데이터의 취약점을 포괄적으로 해결하고자 합니다. 이 통합 접근법은 이미지와 비디오에서 민감한 정보를 효과적으로 보호할 수 있는 다층적 방어 시스템을 제공합니다.

- **Technical Details**: 연구에서는 최신 연합 학습 모델과 잘 알려진 객체 감지 알고리즘을 활용하여 효율적이고 안전한 시각적 데이터 라벨링 시스템을 구축합니다. 또한, 연합 환경에서 민감한 시각 데이터에 대한 익명화 기술을 적용하여 개인 정보를 보호합니다. 이러한 기술적 요소들은 서로 상호작용하며, 시각적 데이터의 유용성을 높이는 동시에 개인 정보 보호 요구 사항을 충족시킬 수 있는 기능을 제공합니다.

- **Performance Highlights**: 제안된 접근법은 기존의 중앙 집중식 모델들과 비교하여, 약간의 정확도 손실이 있을 수 있으나, 개인 정보 보호의 이점이 크게 향상됩니다. 이를 통해 본 연구는 개인 정보가 민감한 애플리케이션에 잘 적합한 해결책을 제시하며, 실험을 통해 성능과 확장성에 대한 포괄적인 평가를 제공하였습니다.



### Beyond Vision: How Large Language Models Interpret Facial Expressions from Valence-Arousal Values (https://arxiv.org/abs/2502.06875)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 비주얼 입력 없이 얼굴 표정의 감정적 의미를 쉽게 추론할 수 있는지를 평가합니다. 특히, 이 연구는 Valence-Arousal(도움-각성) 값이라는 구조화된 숫자 표현을 사용하여 감정을 분류하고 설명하는 능력을 살펴봅니다. LLMs는 이러한 구조적 표현을 활용하여 비언어적 감정 소통의 맥락에서 강력한 인사이트를 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 IIMI와 Emotic이라는 두 개의 데이터셋을 사용하여 LLMs의 감정 분류와 의미 설명 달성 여부를 평가했습니다. 데이터를 처리하기 위해 FaceChannel이라는 패키지를 사용하였으며, 이 모델은 -1에서 1까지의 VA 값을 예측하고 기본적인 감정 카테고리로 분류합니다. LLMs는 이러한 VA 값을 기반으로 감정을 분류하고 그 표현을 설명하는 두 가지 실험에 참여하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 기본 감정의 분류에서 낮은 성능(정확도 30.42% 및 31.42%)을 보였습니다. 그러나 의미 설명을 하는 작업에서는 생성된 설명이 인간의 해석과 밀접하게 일치하는 경향을 보여, LLMs가 얼굴 표정의 자유로운 감정 추론에 있어 더 뛰어난 능력을 보였습니다. 이 연구는 비언어적 감정 인식을 위한 LLMs의 강점과 한계를 탐평하여 향후 AI의 감정 인지 시스템 개발에 기여할 수 있는 방향을 제시합니다.



### BF-GAN: Development of an AI-driven Bubbly Flow Image Generation Model Using Generative Adversarial Networks (https://arxiv.org/abs/2502.06863)
- **What's New**: 이번 연구에서는 bubbly flow generative adversarial networks (BF-GAN)라는 새로운 생성 AI 아키텍처가 개발되었습니다. BF-GAN은 물리적으로 조건화된 입력(jg와 jf)을 통해 사실적이고 고품질의 bubbly flow 이미지를 생성하도록 설계되었습니다. 또한, 52세트의 다양한 조건 하에서 bubbly flow 실험을 수행하여 14만 개의 물리적 레이블이 붙은 이미지를 수집했습니다.

- **Technical Details**: BF-GAN은 mismatch loss와 pixel loss를 포함한 다중 스케일 손실 함수(multi-scale loss function)를 발전시켜 생성 성능을 향상시켰습니다. BF-GAN은 전통적인 GAN을 능가하는 생성 AI 평가 메트릭을 가지고 있으며, 생성된 bubbly flow의 주요 매개변수를 추출하여 측정값 및 경험적 상관관계와 비교하여 성능을 검증했습니다. 이는 BF-GAN의 생성 성능을 확고히 입증하는 결과입니다.

- **Performance Highlights**: BF-GAN은 주어진 jg와 jf에 대해 사실적이고 고품질의 bubbly flow 이미지를 생성할 수 있음을 보여주었습니다. 이 모델은 두 가지 상(fluid) 연구를 위한 생성 AI 솔루션을 제공하여 고품질 데이터를 확보하는 데 필요한 시간과 비용을 크게 절감할 수 있습니다. 또한, bubbly flow 탐지 및 분할 알고리즘을 위한 기준 데이터셋(generator)으로 작용하여 이 연구 분야의 전반적인 생산성을 향상시킬 수 있는 가능성을 가지고 있습니다.



### AutoSketch: VLM-assisted Style-Aware Vector Sketch Completion (https://arxiv.org/abs/2502.06860)
Comments:
          11 pages

- **What's New**: AutoSketch는 기존의 스케치 생성 방식과는 다르게 부분적으로 제공된 스케치를 스타일에 맞춰 완성하는 방법을 제시합니다. 이 연구의 핵심은 자연어로 스케치 스타일을 설명하고, 이를 기반으로 최적화된 선들을 생성함으로써 자동 스케치 완성을 수행할 수 있다는 점입니다. 이를 통해 다양한 스케치 스타일을 지원하며, 단순한 스케치가 아닌 복잡한 장면을 효과적으로 완성할 수 있습니다.

- **Technical Details**: AutoSketch는 pretrained vision-language model (VLM)을 이용하여 부분 스케치의 스타일을 추출한 후, 이를 단계적으로 포함하여 최적의 결과물을 생성합니다. ControlNet을 통해 제공된 입력 스케치에 맞춰 가이던스 이미지를 생성하고, 마스크 패널티를 통해 중복 선이 생성되지 않도록 합니다. 이후 VLM을 활용해 스타일 코드를 생성하여 SVG 포맷의 선들을 조정함으로써 최종 스케치가 원하는 스타일과 일치하도록 합니다.

- **Performance Highlights**: AutoSketch의 성능은 기존 방법들과 비교하여 더 높은 스타일 일관성을 유지하며, 사용자가 제공하는 텍스트 프롬프트에 적합한 내용을 잘 반영한 결과물로 나타났습니다. 연구 결과, 다양한 스케치 스타일 및 프롬프트에 대해 폭넓은 평가를 통해 스케치의 퀄리티를 개선했음을 입증하였으며, 이 방법은 복잡한 장면의 스케치 조합에 유리한 점을 보여줍니다.



### Vision-Integrated LLMs for Autonomous Driving Assistance : Human Performance Comparison and Trust Evaluation (https://arxiv.org/abs/2502.06843)
- **What's New**: 이 논문에서는 전통적인 자율주행 시스템이 복잡하고 예기치 않은 상황에서의 추론에 한계를 보인다는 문제에 대응하기 위해, Large Language Model (LLM) 기반의 자율주행 보조 시스템을 소개합니다. 이 시스템은 시각 이해와 의사결정을 향상시키기 위해 비전 어댑터와 LLM 추론 모듈을 통합하였습니다.

- **Technical Details**: 비전 어댑터는 YOLOv4와 Vision Transformer (ViT)를 결합하여 포괄적인 시각적 특징을 추출합니다. 또한, GPT-4는 인간과 유사한 공간적 추론 및 반응 생성을 가능하게 하며, 이를 통해 시스템의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: 45명의 숙련된 운전자를 대상으로 한 실험 평가 결과, 제안된 시스템은 상황 설명에서 인간의 성능을 잘 모방하며, 적절한 반응 생성을 위한 인간의 결정과 중간 정도의 일치를 보였습니다.



### MeshSplats: Mesh-Based Rendering with Gaussian Splatting Initialization (https://arxiv.org/abs/2502.07754)
- **What's New**: 이번 논문에서는 Gaussian Splatting (GS) 기법을 새로운 형태인 MeshSplats로 변환하여 메쉬 대표성을 사용할 수 있도록 하는 혁신적인 방법을 제안합니다. MeshSplats는 GS의 가우시안 컴포넌트를 메쉬 형태로 변환하여, 전통적인 렌더링 환경에서도 사용할 수 있게 합니다. 이 접근법은 특히 실시간 렌더링에서의 장점과 고품질 광원 및 그림자 효과들을 손쉽게 통합할 수 있음을 보여줍니다.

- **Technical Details**: 행렬 방사화(rasterization) 기술을 적용하는 대신, MeshSplats는 특정한 최적화 알고리즘을 사용하여 메쉬 면에 관해 작동하며, 이것을 통해 메쉬의 재구성 품질을 높입니다. Flat Gaussians로 처리되는 이 기술은 기존의 GS와 메쉬 기반의 이미지 처리 기술을 효과적으로 결합하여, 고품질의 시각적 결과를 도출할 수 있습니다. Nvdiffrast와 같은 도구를 활용하여 효율적인 렌더링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MeshSplats는 기존 GS와 유사한 렌더링 품질을 달성하면서, 메쉬 기반의 표현을 통한 추가적인 장점을 제공합니다. 특히, MeshSplats 최적화 파이프라인은 아티팩트(artifact)를 줄이고, 세부 기하형태(geometric details)를 개선하여 사실적인 렌더링을 실현합니다. 이러한 효과는 또한 기존 전통 렌더링 환경에서의 구현 가능성을 증대시킵니다.



### Economics of Sourcing Human Data (https://arxiv.org/abs/2502.07732)
- **What's New**: AI의 발전은 인간이 생성한 데이터에 크게 의존해왔으나, 대규모 언어 모델의 기능이 향상되면서 이러한 데이터의 품질과 무결성이 위협받고 있다는 주장을 하고 있습니다. 이 논문은 AI가 인간의 참여를 필요로 하는 데이터 수집 시스템의 설계 방식에서의 근본적인 결함을 드러내고 있습니다. 데이터 수집 시스템을 재고하여 외부 인센티브 대신 기여자의 내재적 동기를 중시하는 방향으로 나아가야 한다고 제안합니다.

- **Technical Details**: 현재 사용되고 있는 데이터 수집의 두 가지 주요 소스는 인간 주석과 인터넷 내의 원시 데이터입니다. 그러나 대규모 언어 모델과 AI의 발전으로 인해 오히려 인간이 생성한 데이터의 부족이 우려되고 있습니다. 이 논문에서는 데이터 수집 시스템의 결함을 분석하고, 인간의 동기를 이해하여 보다 효과적인 데이터 수집 방법을 모색하고 있습니다.

- **Performance Highlights**: 기존 데이터 수집 플랫폼은 품질과 양의 균형을 맞추는 데 어려움을 겪고 있으며, 이는 시스템 설계 선택의 결과입니다. 내부적 동기와 외부적 인센티브의 균형을 맞추는 것이 중요하며, 지속 가능한 데이터 제공을 위해서는 인간의 내재적 동기를 증진해야 한다는 점을 강조합니다.



### Causal-Informed Contrastive Learning: Towards Bias-Resilient Pre-training under Concept Drif (https://arxiv.org/abs/2502.07620)
Comments:
          17pages, 3 figures

- **What's New**: 이 논문은 대규모 contrastive pre-training의 새로운 접근 방식을 제안합니다. 특히 개념 드리프트(concept drift)에 의해 발생하는 데이터의 분포 변화가 pre-training에 미치는 영향을 분석하고, 이를 해결하기 위한 causal interventional contrastive objective를 제안합니다. 전통적인 contrastive learning 방법이 드리프트 환경에 효과적으로 대응하지 못하는 문제를 다루고 있으며, 이를 통해 새로운 전략의 필요성을 강조하고 있습니다.

- **Technical Details**: 개념 드리프트는 모델이 훈련되는 동안 목표 도메인의 분포적 특성이 임의로 변화하는 통계적 현상입니다. 이 연구에서는 데이터 스트림S0,t를 정의하고, 각 시점에서 데이터의 분포를 수학적으로 공식화합니다. 또한, 학생-교사 구조를 활용한 momentum 업데이트 과정에서 발생하는 편향(bias)을 명확히하고 이를 완화하기 위한 causal intervention 개념을 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 드리프트 데이터 스트림에서 발생하는 편향을 효과적으로 완화하며, 다양한 하위 작업(downstream tasks)에서 우수한 성능을 보임을 입증했습니다. 특히, 장기한 분류(long-tailed classification), OOD 탐지(out-of-distribution detection), 도메인 이동(domain shift) 작업에서 기존의 방법보다 뛰어난 성능을 발휘하는 것으로 나타났습니다. 이 연구는 대규모 모델의 pre-training 과정에서의 드리프트 문제 해결에 기여할 것으로 기대됩니다.



### DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training (https://arxiv.org/abs/2502.07590)
- **What's New**: 본 논문은 동적 주의력 희소성(dynamic attention sparsity)을 활용하여 비디오 Diffusion Transformers(DiTs)의 훈련을 가속화하고 확장할 수 있는 DSV라는 새로운 프레임워크를 소개합니다. 이는 고해상도 및 긴 비디오에 대한 훈련 시 발생하는 연산적 복잡성을 줄이는데 중요한 역할을 합니다. DSV는 효율적인 맞춤형 커널을 통해 중요한 요소에 집중하는 두 단계 훈련 알고리즘을 구현하며, 이로 인해 훈련 속도를 최대 3.02배 향상시키면서도 품질 저하가 거의 없습니다.

- **Technical Details**: DSV는 동적 희소성을 활용하여 주의력 모듈의 연산을 최적화합니다. 첫 번째 단계에서는 예측기를 훈련시켜 각 주의력 헤드의 주의력 점수를 근사하도록 하며, 두 번째 단계에서는 실제 비용-편익 균형을 평가하여 각 블록에 대한 희소 계산 적용 여부를 결정합니다. 또한, 커널 융합(kernels fusion) 전략을 사용하여 대규모 중간 행렬을 구성할 필요 없이 예측 및 추정 과정을 통합하여 효율성을 높입니다.

- **Performance Highlights**: DSV는 최대 64개의 H100 GPU로 테스트된 비디오 DiT 모델에서 260K 길이의 입력을 처리하며, 기존 방법에 비해 훈련 처리량을 3.02배 향상시켰습니다. 더불어, DSV는 시각적 품질에서도 풀 어텐션과 비교할 만한 결과를 달성하였고, 실제 추론 효율성에서도 3.5배 개선된 성능을 보여주었습니다. 전반적으로, DSV는 품질 저하 없이 향상된 속도와 처리량을 입증하였습니다.



### SketchFlex: Facilitating Spatial-Semantic Coherence in Text-to-Image Generation with Region-Based Sketches (https://arxiv.org/abs/2502.07556)
Comments:
          conference: CHI2025

- **What's New**: 이번 연구에서는 비전문가 사용자들이 텍스트 기반 이미지 생성 모델을 활용하는 데의 어려움을 해결하기 위해 SketchFlex라는 상호작용 시스템을 소개합니다. SketchFlex는 사용자가粗略한 스케치를 통해 공간 상황을 조정할 수 있도록 설계되었으며, 사용자 의도를 충족시키는 이미지 생성을 지원합니다. 이 시스템은 의미론적 공간 내에서 사용자 프롬프트와 스케치를 자동으로 해석하여 정확한 이미지 생성을 가능하게 합니다.

- **Technical Details**: SketchFlex는 다중 지역 공간 조정을 위해 설계된 의미론적 공간에서 사용자 프롬프트를 해석하고, 사용자가 만든粗略한 스케치에 맞춰 프롬프트를 추천하는 기능을 포함하고 있습니다. 또한, 사용자가 만든 스케치를 고품질의 형태로 정제하는 기법을 포함하여, 이를 통해 사용자 의도에 부합하는 결과물을 생성합니다. 스케치의 정제를 위해서는 객체를 분해하고 재조합하는 접근 방법을 사용하여 실제와 유사한 이미지를 생성합니다.

- **Performance Highlights**: 사용자 연구를 통해 SketchFlex는 기존의 기법들에 비해 더 일관성 있는 이미지 생성을 제공하는 것으로 나타났습니다. 사용자들은 시스템의 유연성과 출력 결과의 정밀도가 향상되었다고 피드백하였으며, 특히 비전문가 사용자들이 보다 쉽게 창의적인 의도를 표현할 수 있게 해주었습니다. 이런 결과는 SketchFlex가 이미지 생성의 품질을 개선하고 사용자 의도와 일치하는 결과를 생성하는 데 기여한다는 것을 보여줍니다.



### The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation (https://arxiv.org/abs/2502.07516)
- **What's New**: 이 논문은 의료 이미지 분석에서의 T2I(diffusion models) 생성 모델의 데이터 기억 문제를 분석한 첫 체계적 연구입니다. MIMIC-CXR 데이터셋에서 가장 많은 기억을 초래하는 텍스트 프롬프트와 토큰을 찾기 위한 데이터 기반 접근 방식을 채택했습니다. 특히, 비디오신원 확인 절차와 관련된 마커가 기억에 가장 많이 기여한다는 점이 밝혀졌습니다.

- **Technical Details**: 논문에서는 T2I 생성 모델의 기억화 메커니즘에 대한 깊은 이해를 바탕으로 두 가지 확산 프로세스를 설명하고 있습니다. 첫 번째는 시간적으로 넘어가는 Gaussian 노이즈를 사용해 데이터를 점진적으로 손상시키는 전방 프로세스이며, 두 번째는 제공된 데이터를 반복적으로 디노이징하여 원래의 샘플을 복원하는 역방향 프로세스입니다. 이 과정을 통해 Quick Track 및 Text-Conditional Noise를 통한 기억화 발견 메트릭을 제안합니다.

- **Performance Highlights**: 기존 기억화 완화 전략이 모델의 기억 토큰에 대한 의존도를 충분히 줄이지 못한다는 사실을 발견했습니다. 이는 MIMIC-CXR 데이터셋에서 T2I 생성 모델의 신뢰성 문제를 강조합니다. 이 연구는 기억화 완화 기술 개발을 위한 기초 자료를 제공하고, 향후 의학 이미지 연구에 대한 중요한 통찰력을 제시합니다.



### Quantitative evaluation of unsupervised clustering algorithms for dynamic total-body PET image analysis (https://arxiv.org/abs/2502.07511)
Comments:
          12 pages, 2 figures

- **What's New**: 최근 동적 전신 양전자 방출 단층 촬영(PET) 이미징이 새로운 스캐너 장치 덕분에 가능해졌습니다. 이전에 제안된 군집화 알고리즘에도 불구하고 동적 전신 PET 이미지를 처리하기 위한 체계적인 연구는 여전히 부족합니다. 본 연구에서는 15개 비지도 군집화 방법의 성능을 비교하여 시간 활동 곡선(TAC)을 분류하는 데 효과적인 방법을 찾고자 합니다.

- **Technical Details**: 연구에서 사용된 군집화 알고리즘은 파이썬과 scikit-learn 라이브러리를 통해 구현되었습니다. 30명의 환자로부터 수집된 동적 PET 이미지를 사용하여 5개의 기관에서 TAC를 분류하는 작업을 진행하였습니다. K-means, Gaussian mixture model, fuzzy c-means 등의 다양한 알고리즘이 적용되었으며, 각 알고리즘의 성능은 정확도 및 처리 시간으로 평가되었습니다.

- **Performance Highlights**: 최종 결과에 따르면 Gaussian mixture model(GMM), fuzzy c-means(FCM), 그리고 mini batch K-means가 최상의 성능을 보였으며, 각각의 TAC 분류 정확도는 89%, 83%, 81%로 나타났습니다. 처리 시간은 각 이미지 별 평균 0.5초 이내로, 동적 전신 PET 분석에 유망한 방법으로 확인되었습니다.



### RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization (https://arxiv.org/abs/2502.07492)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 연구에서는 APT (Advanced Persistent Threat) 맬웨어를 그룹에 귀속시키는 데 필요한 새로운 방법론을 제안합니다. APT 공격자는 자신의 정체성을 숨겨 귀속이 어렵고 기존 머신러닝 기반 모델은 적대적 공격에 취약한 것으로 나타났습니다. 본 연구에서는 새로운 단일 단계의 적대적 훈련 접근 방식인 RoMA를 소개하여 맬웨어 귀속의 강인성을 개선하고, AMG18이라는 새로운 데이터셋을 제공하며, 기존 모델보다 눈에 띄는 성능 향상을 보여줍니다.

- **Technical Details**: RoMA 접근 방식은 Global Perturbation (GP) 전략과 Adversarial Consistency Regularization을 사용하여 적대적 샘플을 생성하고, 이를 통해 모델의 표현 품질을 높이는 데 중점을 둡니다. GP 전략은 학습된 섭동 패턴을 적용하여 강력한 적대적 맬웨어 샘플을 생성하며, Adversarial Consistency Regularization은 청정 및 적대적 맬웨어를 통합하여 표현 공간 최적화에 도움을 줍니다. 이 접근 방식은 훈련 효율성을 극대화하며 APT 맬웨어에 대한 새로운 강인한 귀속 모델을 생성합니다.

- **Performance Highlights**: RoMA는 PGD 공격 하에서도 80% 이상의 강인한 정확도를 달성하며, 이는 향후 연구에서 사용될 수 있는 중요한 결과입니다. 또한, RoMA는 효율적인 훈련 속도를 가지고 있으며, 다음으로 좋은 방법보다 두 배 이상 빠른 훈련 시간을 기록했습니다. 연구 결과는 비적대적 환경에서도 우수한 정확도를 유지하며, APT 맬웨어 귀속의 새로운 기준을 설정했습니다.



### FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for Federated Learning on Heterogeneous Data (https://arxiv.org/abs/2502.07456)
Comments:
          13 pages, 2 figures

- **What's New**: 본 논문에서는 개인화된 연합 학습(Personalized Federated Learning, PFL)을 위한 새로운 방법론인 FedAPA를 제안합니다. FedAPA는 서버 측에서 주어진 gradient 기반의 적응형 집계 전략을 활용하여 개인화된 모델을 생성합니다.

- **Technical Details**: 기존의 집계 가중치 기반 PFL 방법이 동질적이지 않은 데이터에 어려움을 겪는 문제를 해결하기 위해, FedAPA는 클라이언트 파라미터 변화의 gradient에 따라 집계 가중치를 업데이트하는 방식을 채택했습니다. 이를 통해 중앙 집중 방식으로 집합체를 최적화하여 정확도와 계산 효율성을 높입니다.

- **Performance Highlights**: FedAPA는 세 개의 데이터셋에서 10개의 PFL 경쟁자를 능가하는 뛰어난 정확도와 계산 효율성을 달성하며, 이론적인 수렴성을 보장합니다. 또한 통신 오버헤드도 경쟁력 있는 수준으로 유지합니다.



### Hierarchical Document Parsing via Large Margin Feature Matching and Heuristics (https://arxiv.org/abs/2502.07442)
Comments:
          DocUI@AAAI-25, 2 pages, technical report

- **What's New**: 본 연구에서는 AAAI-25 VRD-IU 챌린지에서 1위에 선정된 독서 문서 구조 파싱에 대한 새로운 접근 방식을 제시합니다. 이 방법은 대규모 마진 손실(large margin loss)을 이용하여 특징 구분을 개선하며, 휴리스틱 규칙(heuristic rules)을 활용하여 계층 관계를 정제합니다. 심층 학습 기반의 매칭 전략과 탐욕 알고리즘(greedy algorithms)을 결합하여 계산 효율성을 유지하면서도 정확성을 크게 향상시켰습니다.

- **Technical Details**: 이 논문에서는 CLIP의 손실 함수를 개선하여 계층 관계 예측에 필요한 대규모 마진 손실을 도입합니다. 또한, 특정 카테고리에 속하는 엔티티는 부모를 가지지 않으며, 계층적 관계가 강하게 존재하는 카테고리들이 있습니다. 이러한 구조적 패턴을 통해 부모-자식 관계 예측의 정확성과 계산 효율성을 개선하고 있습니다.

- **Performance Highlights**: 우리의 방법은 개인 리더보드에서 0.98904의 정확도를 달성하며, AAAI-25 VRD-IU 챌린지에서 최첨단 성능을 나타냅니다. 깊이 학습과 규칙 기반의 세분화를 결합하여 문서 구조 파싱의 효과를 입증하였으며, 향후 더 다양한 데이터 세트 탐색과 휴리스틱 규칙 세분화를 통한 일반화를 포함한 향후 연구 방향을 제시합니다.



### MoENAS: Mixture-of-Expert based Neural Architecture Search for jointly Accurate, Fair, and Robust Edge Deep Neural Networks (https://arxiv.org/abs/2502.07422)
- **What's New**: 최근 연구에서는 edge Deep Neural Networks (DNNs)의 정확성과 효율성을 최적화하려고 노력하고 있습니다. 그러나 기존의 설계 기술은 공정성(fairness), 견고성(robustness), 일반화(generalization)와 같은 중요한 메트릭을 간과하는 경향이 있었습니다. 이에 따라, 우리는 Mixture-of-Experts 기반의 Neural Architecture Search (MoENAS)를 제시하여 정확하면서도 공정하고 견고한 edge DNN을 발견하는 방법을 제안합니다.

- **Technical Details**: MoENAS는 기존 최첨단(edge SOTA) DNN에 비해 4.02%의 정확도 향상을 이루었으며, 피부 톤에 따른 정확도 차이를 14.09%에서 5.60%로 감소시켰습니다. 또한 견고성을 3.80% 높이고, 과적합(overfitting)은 0.21%로 최소화했습니다. 이러한 모든 개선을 통해 MoENAS는 edge DNN 설계의 새로운 기준을 세우며, 포용적이고 견고한 모델 개발을 위한 길을 열고 있습니다.

- **Performance Highlights**: MoENAS의 도입으로 모델의 크기는 기존 최첨단 모델 평균 크기와 유사(+0.4M)하게 유지되면서도 성능이 크게 향상되었습니다. 이러한 결과는 edge DNN의 설계에서 공정성과 견고성을 고려한 접근 방식의 필요성을 잘 보여줍니다. MoENAS는 최적의 성능을 달성하기 위해 아키텍처 설계에서 공정성, 견고성 및 일반화를 다각적으로 고려한 점이 특징입니다.



### No Data, No Optimization: A Lightweight Method To Disrupt Neural Networks With Sign-Flips (https://arxiv.org/abs/2502.07408)
- **What's New**: 본 논문은 Deep Neural Networks (DNNs)의 취약점을 밝혀내고, 모델의 파라미터에서 선호되는 sign bits의 일부를 뒤집는 것만으로도 성능에 치명적인 악영향을 미칠 수 있음을 입증하는 새로운 방법론, Deep Neural Lesion (DNL)을 제안합니다. 이는 데이터 없이, 가벼운 방법으로 저해되는 성능을 극대화할 수 있는 전략으로, ResNet50 모델의 경우 단지 두 개의 sign bits를 뒤집는 것만으로도 정확도가 99.8% 감소하는 결과를 보였습니다.

- **Technical Details**: 저자들은 DNN이 일반적으로 사용하는 32비트 부동 소수점 형식(IEEE 754)의 구조를 활용하여, sign bits를 조작할 수 있는 취약점을 확인했습니다. 이들은 비효율적인 계산 없이 경량의 공격을 수행할 수 있으며, 단일 forward 및 backward 패스를 사용하는 1-Pass 공격을 통해 보다 정밀한 파라미터 선택이 가능합니다. 또한, 이 방법은 각 모델 아키텍처에 걸쳐 광범위한 공격 가능성을 가지고 있으며, 데이터 의존성이 없습니다.

- **Performance Highlights**: 60가지 분류 모델과 다양한 데이터셋을 통해 접근 방식을 검증한 결과, 단지 10개 미만의 bits를 조작하는 것만으로도 정확도를 크게 감소시킬 수 있음을 확인했습니다. 더불어, 저자들은 취약한 sign bits를 선택적으로 보호함으로써, DNN 모델이 이와 같은 공격에 보다 강한 저항력을 가질 수 있는 방안을 제시합니다. 이로 인해 자율 주행 시스템과 같이 안전이 중요한 분야에서의 DNN의 신뢰성이 더욱 강화될 것으로 기대됩니다.



### Human-in-the-Loop Annotation for Image-Based Engagement Estimation: Assessing the Impact of Model Reliability on Annotation Accuracy (https://arxiv.org/abs/2502.07404)
- **What's New**: 이 연구는 Human-in-the-loop (HITL) 프레임워크가 감정 추정 시스템의 주석 정확도를 높이는 데 도움이 되는 잠재력에 대해 다루고 있습니다. 특히 고성능 이미지 기반 감정 모델을 HITL 주석 프레임워크에 통합하여 인간-기계 상호작용의 협업 가능성을 평가하고, 성공적인 협업에 필요한 심리적 및 실제적 요소들을 식별합니다. 연구의 주요 결과로는 모델의 신뢰성과 인지적 프레이밍이 annotator의 신뢰, 인지 부하 및 주석 행동에 어떻게 영향을 미치는지를 보여줍니다.

- **Technical Details**: 연구에서는 다양한 모델 신뢰성과 인지적 프레이밍의 변형을 바탕으로 S1, S2, S3의 세 가지 실험 시나리오를 설정하여 29명의 참가자를 대상으로 행동 및 질적 데이터를 분석하였습니다. S1에서는 신뢰할 수 있는 예측이 높이 평가받았으며, S2의 신뢰할 수 없는 출력은 주석가의 비판적 평가를 유도했으나 불만과 반응 변동도 증가시켰습니다. S3에서는 부정적 프레이밍이 참가자들에게 모델에 대한 인지도와 정확성을 높이는 결과를 가져왔습니다.

- **Performance Highlights**: 모델의 신뢰성과 심리적 요소가 효과적인 인간-기계 협업을 형성하는 데 중대한 역할을 한다는 점이 강조되었습니다. 연구 결과, HITL 프레임워크가 감정 주석과 같은 분야에서 인간의 감독과 자동화 시스템의 강점을 잘 활용할 수 있음을 보여주며, 이는 향후 적응 학습 및 인간-컴퓨터 상호작용 등 다양한 분야에 응용될 수 있는 기반을 마련했습니다.



### Supervised contrastive learning for cell stage classification of animal embryos (https://arxiv.org/abs/2502.07360)
- **What's New**: 이 연구는 비디오 미세현미경(video microscopy)과 머신 러닝(machine learning)을 결합하여 시험관에서 생산된(IVP) 배아의 초기 발달을 연구하는 데 혁신적인 접근법을 제안합니다. 특히 소(소) 배아의 발달 단계 자동 분류를 위한 새로운 데이터셋인 Bovine Embryos Cell Stages (ECS)를 생성했습니다.

- **Technical Details**: 연구의 주요 도전 과제는 저품질 이미지와 세포 단계 식별을 어렵게 만드는 소의 어두운 세포, 발달 단계 경계에서의 클래스 모호성(class ambiguity), 그리고 불균형한 데이터 분포입니다. 이러한 문제를 해결하기 위해, CLEmbryo라는 새로운 방법을 도입하였으며, 이는 감독 대비 학습(supervised contrastive learning)과 포컬 로스(focal loss)를 결합하여 훈련하고, CSN-50이라는 경량 3D 신경망(neural network)을 인코더로 사용합니다.

- **Performance Highlights**: CLEmbryo는 Bovine ECS 데이터셋과 공개된 NYU Mouse Embryos 데이터셋 모두에서 최신의 방법들(state-of-the-art methods)을 초월하는 성능을 보였습니다. 이 방법은 광범위한 일반화(generalization) 능력을 갖추고 있어, 더 많은 생물학적 응용 가능성을 시사합니다.



### Generative Ghost: Investigating Ranking Bias Hidden in AI-Generated Videos (https://arxiv.org/abs/2502.07327)
- **What's New**: 이번 연구에서는 AI가 생성한 비디오(AI-generated videos)가 정보 검색 모델에 미치는 영향을 조사합니다. 연구진은 13,000개의 비디오로 구성된 포괄적인 기준 데이터셋을 구축하고, 비디오 검색 모델에서 AI가 생성한 콘텐츠에 대한 편향이 존재하는지를 분석하였습니다. 특히, 비디오 검색에서 발생하는 시각 및 시간적 요인에 따른 편향을 탐구합니다.

- **Technical Details**: 연구에서는 영상의 동적 특성과 생생함으로 인해 비디오 콘텐츠가 정보 전파 및 오락 매체로 주요하게 부각된다고 설명합니다. 비디오 검색 모델에 대한 평가를 위해, AI가 생성한 비디오와 실제 비디오 간의 의미적 유사성을 고려하면서도 성과를 객관적으로 평가하기 위한 다차원 메트릭스를 설계하였습니다. 최첨단 오픈소스 비디오 생성 모델 두 개를 사용하여 비디오를 생성하고, 자신의 기준 데이터셋에서 정보 검색 모델을 효율적으로 평가합니다.

- **Performance Highlights**: 비디오 검색 모델은 AI가 생성한 비디오를 선호하는 경향을 보였습니다. AI 생성 비디오의 비율이 높아질수록 검색 모델에서 해당 콘텐츠에 대한 편향이 더욱 심화되었습니다. 또한, 연구에서는 비디오 검색 모델의 편향을 완화하기 위해 대조 학습을 적용하여 실제 비디오를 우선 순위로 두었습니다.



### Dataset Ownership Verification in Contrastive Pre-trained Models (https://arxiv.org/abs/2502.07276)
Comments:
          Accepted by ICLR2025

- **What's New**: 최근의 아카이브 논문에서는 고품질 오픈소스 데이터셋의 소유권 검증을 위한 새로운 방법을 제안하고 있습니다. 기존의 기술들은 주로 감독된 모델에 국한되어 있으며, 자가 감독(self-supervised) 프리트레인(pre-trained) 모델로 직접 확장할 수 없었습니다. 이 논문에서는 대칭적 학습(contrastive learning)을 통해 자가 감독 프리트레인 모델을 위한 첫 데이터셋 소유권 검증 방법을 도입하여 데이터 소유자가 자신의 권리를 옹호할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 관찰에 기초하여 구성됩니다. 첫 번째는 유니어리 관계(unary relationship)로, 대칭학습을 통해 프리트레인된 인코더가 동일한 샘플의 변형에 대해 훨씬 더 유사한 표현을 생성한다는 것입니다. 두 번째는 이진 관계(binary relationship)로, 보인 샘플 간의 쌍별 유사성이 데이터 증강 후에도 크게 변하지 않는다는 점입니다. 이러한 관계의 차이를 '대칭적 관계 간극(contrastive relationship gap)'이라고 정의하고, 이를 통해 방어자는 의심스러운 인코더가 자신의 데이터셋으로 프리트레인 되었는지를 확인할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 모든 이전 연구를 초월하여 p-value가 0.05 이하로 통계적으로 유의미한 결과를 보여줍니다. SimCLR, BYOL, SimSiam, MOCO v3 및 DINO를 포함한 여러 대칭적 프리트레인 모델에서 검증의 유효성을 입증하였으며, 이는 데이터셋 소유권 검증 분야에 중요한 기여를 하고 있습니다. 또한, 제안된 접근법이 대량의 데이터셋을 요구하지 않고도 정확한 검증을 가능하게 함을 시사합니다.



### Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk H$α$ Images (https://arxiv.org/abs/2502.07259)
Comments:
          15 pages, 5 figures, 3 tables, accepted for publication in ApJ

- **What's New**: 이번 연구에서는 태양 필라멘트(segmentation)를 위해 설계된 새로운 경량화 모델인 Flat U-Net을 소개합니다. 이 모델은 간소화된 채널 주의(Simplified Channel Attention)와 채널 자기 주의(Channel Self-Attention) 합성곱 블록을 통합하여 Hα(full-disk) 이미지를 활용한 필라멘트 식별의 정확도와 효율성을 대폭 향상시켰습니다. 자동화된 데이터 처리의 중요성이 강조되는 가운데, 이 모델은 기존의 모델들에 비해 경량화된 구조를 제공합니다.

- **Technical Details**: Flat U-Net은 각각의 네트워크 층에서의 특징 정보를 완전히 추출하여 상호 채널 특징 표현을 재구성하는 구조를 가지고 있습니다. 각 블록은 이전 층에서의 채널 특징을 효과적으로 최적화하여 파라미터를 현저히 줄입니다. 개선된 구조적 설계 덕분에 계산 효율성이 극대화되어, 더 적은 리소스 소비로 높은 성능을 낼 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 순수 SCA로 구성된 모델은 약 0.93의 정확도와 0.76의 주사율(Dice Similarity Coefficient), 0.64의 재현율(recall rate)로 기존 U-Net 모델보다 뛰어난 성능을 보여주었습니다. CSA 블록의 도입은 주사율과 재현율을 각각 0.82 및 0.74로 개선하여, 모델의 크기와 탐지 효과성 측면에서 뚜렷한 장점을 입증했습니다. 해당 연구는 공개 소스로 데이터셋과 모델, 코드를 제공하고 있습니다.



### Color-Quality Invariance for Robust Medical Image Segmentation (https://arxiv.org/abs/2502.07200)
- **What's New**: 이 논문에서는 단일 출처 도메인 일반화(single-source domain generalization, SDG) 문제를 해결하기 위해 두 가지 새로운 기술을 제안합니다. 첫 번째는 동적 색상 이미지 정규화(dynamic color image normalization, DCIN) 모듈이고, 두 번째는 색상-품질 일반화(color-quality generalization, CQG) 손실입니다. 이러한 기술들은 색상과 품질의 변화를 극복하며, 다양한 도메인에서의 의료 이미지 분할(segmentation) 성능을 향상시킵니다. 실험 결과, 제안된 방법이 기존 방법에 비해显著(日録的) 향상을 보여주었습니다.

- **Technical Details**: DCIN 모듈은 훈련 데이터에서 적합한 참조 이미지를 동적으로 선택하여 테스트 이미지의 색상 분포를 맞추는 데 사용됩니다. 이 모듈은 두 가지 전략, 즉 전역 참조 이미지 선택(global reference image selection, GRIS)과 지역 참조 이미지 선택(local reference image selection, LRIS)을 통합하여 동작합니다. GRIS는 모든 테스트 이미지에 대해 한 개의 참조 이미지를 할당하고, LRIS는 각 테스트 이미지에 대해 고유한 참조 이미지를 선택합니다. 이 과정은 심리적 색상 공간(perception-based color space)인 l⁢α⁢β에서 수행됩니다.

- **Performance Highlights**: 실험 결과, 제안된 DCIN 및 CQG 방법이 기존 기준에 비해 32.3포인트의 DICE 점수 증가를 기록하여 의료 이미지 분할 성능을 크게 향상시켰습니다. 이 방법들은 단일 출처 도메인에서 학습된 모델을 사용하여 다른 보지 못한 도메인에서도 강력한 결과를 생성합니다. 또한, 제안된 방법은 어떤 SDG 색상 이미지 분할 모델에도 적용 가능한 모델 불가지론성을 보여줍니다.



### Space-Aware Instruction Tuning: Dataset and Benchmark for Guide Dog Robots Assisting the Visually Impaired (https://arxiv.org/abs/2502.07183)
Comments:
          ICRA 2025

- **What's New**: 이번 논문에서는 시각 장애인을 위한 안내견 로봇의 가능성을 제시하고, 기존의 Vision-Language Models (VLMs)의 한계를 극복하기 위한 새로운 Space-Aware Instruction Tuning (SAIT) 데이터셋과 Space-Aware Benchmark (SA-Bench)를 소개합니다. 기존의 VLM들은 공간적 관계를 제대로 해석하지 못하는 문제를 지니고 있으며, 그로 인해 복잡한 환경에서의 내비게이션에 어려움을 겪고 있습니다. SAIT 데이터셋은 3D 공간에서의 경로 정보와 주변 환경을 집중적으로 제공하여 이러한 문제를 해결하려고 합니다.

- **Technical Details**: 새로운 데이터 생성 파이프라인은 안내견 로봇이 3D 공간의 목표 위치로 향하는 경로를 명확히 인식할 수 있도록 돕습니다. 이 파이프라인은 깊이 맵을 추출하고 상관 관계를 명확하게 하여 VLMs가 방향을 구분하는 데 도움을 줍니다. 또한, SA-Bench는 VLM이 시각 장애인을 위한 걷기 안내에 얼마나 효과적인지를 평가하기 위한 프로토콜을 가지고 있습니다.

- **Performance Highlights**: 비교 실험을 통해 공간 인식과 훈련이 적용된 모델은 기존의 최첨단 알고리즘보다 더 우수한 성능을 보여 주었습니다. 특히, 이 모델은 시각 장애인에게 필요한 정확하고 간결한 안내 정보를 제공하는 데 효과적입니다. 연구 결과는 SAIT 데이터셋과 SA-Bench가 공개되어 이 분야의 발전에 기여할 것으로 기대됩니다.



### Tab2Visual: Overcoming Limited Data in Tabular Data Classification Using Deep Learning with Visual Representations (https://arxiv.org/abs/2502.07181)
- **What's New**: 본 연구에서는 의료와 같은 데이터 제한 영역에서의 표 데이터 분류 문제를 다룹니다. Tab2Visual이라는 새로운 접근 방식을 제안하여 이질적인 표 데이터를 시각적 표현으로 변환, 강력한 딥러닝 모델의 적용을 가능하게 합니다. 이를 통해 데이터 부족 문제를 해결하고, 새로운 이미지 증강 기법(image augmentation techniques)을 도입하여 전이 학습(transfer learning)을 용이하게 합니다. Tab2Visual의 성능은 다양한 머신러닝 알고리즘과 비교하여 평가하였으며, 제한된 표 데이터에 대해 기존 방법보다 뛰어난 성과를 나타냈습니다.

- **Technical Details**: Tab2Visual은 이질적인 표 데이터를 CNNs 및 ViTs와 같은 딥러닝 모델에 적합한 형식으로 변환하는 데이터 변환(data transformation) 방법입니다. 이 방법론은 기존의 표 데이터 처리 접근 방식의 한계를 극복하기 위해 다양한 데이터 전처리(preprocessing) 기술을 허용합니다. 또한, Tab2Visual은 넓은 범위의 분류(classification) 문제에 대해 선형 변환(linear syntheses) 기술을 적용하는 등, 기존의 이미지 생성 기법들과는 다른 혁신적인 접근을 통해 향상된 성능을 보장합니다.

- **Performance Highlights**: 실험 결과 Tab2Visual은 적은 양의 표 데이터로 인한 분류 문제에서 다른 방법보다 우수한 성능을 보였습니다. 특히, 전통적인 머신러닝 알고리즘(예: tree-based ensembles)과 비교할 때 더 나은 결과를 도출하여 탁월한 새로움을 증명했습니다. 또한, Tab2Visual은 의료와 같은 데이터 제한 환경에서도 효과적으로 작동하여 작은 데이터셋에서도 높은 분류 성과를 달성할 수 있도록 설계되었습니다.



### Choroidal image analysis for OCT image sequences with applications in systemic health (https://arxiv.org/abs/2502.07117)
Comments:
          PhD thesis toward a doctorate degree at the University of Edinburgh. PhD funded by the Medical Research Council (grant MR/N013166/1). Reviewed and examined by Dr. Roly Megaw (internal) and Prof. Pearse Keane (external) in December 2024 and ratified in the same month by the university. Official record found here: this https URL

- **What's New**: 이 논문은 망막 뒤에 위치한 고혈관 층인 맥락막(choroid)의 분석을 위한 여러 새로운 방법을 개발하였습니다. 특히, Optical Coherence Tomography (OCT)를 사용하여 기존 수동 및 반자동 분석 방법의 한계를 극복하고, 이 결과로 맥락막의 혈액 흐름을 체계 질병의 건강 상태와 연관짓는 데 중요한 도구를 제공하고 있습니다.

- **Technical Details**: 연구에서는 Gaussian Process Edge Tracing (GPET)와 Multi-scale Median Cut Quantisation (MMCQ)이라는 두 가지 반자동 접근 방식을 먼저 개발하였으며, 이는 수동 방법에 비해 개선된 성과를 보입니다. 이후 DeepGPET라는 딥러닝 기반의 지역 세분화 방법을 도입하여 실행 시간, 재현 가능성 및 사용자 접근성을 향상시켰지만, 여전히 맥락막 혈관 분석 및 자동 피처 측정 기능이 부족합니다. 최종적으로 개발된 Choroidalyzer는 맥락막 공간과 혈관을 세분화하고, 완전 자동화된 임상적 의미를 지닌 재생산 가능한 맥락막 특성을 생성하는 방식입니다.

- **Performance Highlights**: 논문은 네 가지 방법에 대한 철저한 평가를 제공하고, OCTANE, PREVENT 및 D-RISCii라는 세 가지 응용 분야에서의 임상적 가치도 고려하고 있습니다. OCTANE은 신장 이식 수혜자 및 기증자의 맥락막 변화를 평가하며, PREVENT는 중년의 알츠하이머 위험 요소와의 연관성을 탐색합니다. D-RISCii는 중환자 치료에서 OCT의 재현성 및 변동성을 평가하며, 이 연구는 체계 건강에서 맥락막이 바이오마커(biomarker)로서의 잠재력을 강조합니다.



### A Framework for Supervised and Unsupervised Segmentation and Classification of Materials Microstructure Images (https://arxiv.org/abs/2502.07107)
- **What's New**: 이 논문은 자동화된 프레임워크를 제안하여 미세구조(microstructure) 이미지를 분류하고 세그먼트(segmentation)하는 새로운 방법론을 소개하고 있습니다. 이 프레임워크는 비지도학습(unsupervised learning) 및 지도학습(supervised learning) 기법을 결합하여 여러 상이한 미세구조의 동질적 영역(homogeneous regions)으로 분리할 수 있도록 합니다. 이는 특히 초고해상도 이미징 기술과 대량의 미세구조 데이터의 필요성을 충족시키고, 새로운 재료의 발견과 분석에 기여할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성됩니다. 첫째, 최근 개발된 스코어 기반(score-based) 방법을 통해 다상(multiphase) 미세구조의 세그멘테이션을 비지도 방식으로 수행합니다. 둘째, 불확실성 인식 지도 분류 네트워크를 통해 각 동질적 영역을 분류하며, 이는 최소한의 인간 검수를 거쳐 검증됩니다. 마지막으로, 데이터 증강(data augmentation)을 통해 더욱 강력한 미세구조의 세분화를 수행하는 초급 네트워크(supervised segmentation)를 적용합니다.

- **Performance Highlights**: 프레임워크는 여러 재료 및 질감 이미지 세트에서 유효성을 입증하였으며, 미세구조를 지속적으로 특성화하고 새로운 동질적 또는 다상 재료를 식별할 수 있는 능력을 보여주었습니다. 이 시스템은 재료 특성을 추출하고 지식 기반을 구축하는 데 있어 매우 강력한 도구가 될 것으로 기대됩니다. 각각의 단계에서 최소한의 인간 개입으로도 효과적인 결과를 도출할 수 있어, 산업적 응용에서도 높은 활용 가능성을 가지고 있습니다.



### Lotus: Creating Short Videos From Long Videos With Abstractive and Extractive Summarization (https://arxiv.org/abs/2502.07096)
Comments:
          15 pages, 9 figures, ACM IUI 2025

- **What's New**: 이 논문에서는 TikTok 및 Instagram과 같은 플랫폼에서 인기를 끌고 있는 짧은 형식의 동영상을 제작하는 데 있어 새로운 시스템인 Lotus를 소개합니다. Lotus는 기존의 긴 형식 동영상에서 추출한 클립과 새로운 내레이션을 결합하여 짧은 형식 동영상을 제작할 수 있게 해주며, 이는 창작자들이 더욱 쉽게 콘텐츠를 제작할 수 있도록 돕습니다. 이 시스템은 추출적(extractive) 및 추상적(abstractive) 접근법을 결합하여 콘텐츠의 원본을 유지하면서도 편리하게 조정할 수 있도록 합니다.

- **Technical Details**: Lotus는 먼저 짧은 형식의 스크립트와 해당하는 음성을 생성하여 추상적(short-form) 동영상을 제작합니다. 그 후 생성된 내레이션에 맞춰 긴 형식(long-form) 동영상 클립을 일치시키는 과정을 거칩니다. 창작자는 자동화된 방법을 통해 추출적 클립을 추가하거나 Lotus의 편집 인터페이스를 통해 추가 편집을 할 수 있습니다.

- **Performance Highlights**: 사용자 연구에서는 Lotus를 사용한 짧은 형식 동영상 제작의 효율성을 기존의 추출적 방법과 비교하였습니다. 연구 결과, Lotus는 보다 높은 사용자 만족도와 함께 콘텐츠 제작의 유연성을 제공하여 창작자들이 다양한 방식으로 동영상을 구성할 수 있도록 도움을 주는 것으로 나타났습니다.



### On the use of neural networks for the structural characterization of polymeric porous materials (https://arxiv.org/abs/2502.07076)
- **What's New**: 이번 연구에서는 다공성(porous) 물질의 구조적 특성을 분석하기 위해 새로운 딥러닝(deep-learning) 기반 기법을 제안합니다. 전통적인 방법들이 많은 시간과 인간의 주관적 판단에 의존하는 반면, 제안된 방법은 자동화된 도구로 신뢰성을 높일 수 있습니다. 이 연구의 핵심은 이미지를 통한 다공성 재료의 구조적 특성을 고속으로 평가할 수 있는 방법론입니다.

- **Technical Details**: 연구에서는 컨볼루션 신경망(convolutional neural network)을 사용하여 여러 가지 훈련 구성(training configurations)에 따라 세밀하게 조정된 Mask R CNN 모델을 평가했습니다. 데이터셋은 다양한 폴리머(polymers)로 구성된 SEM 이미지를 포함하며, 닫힌 기공(closed-pore) XPS, 폴리우레탄(PU), 폴리메틸 메타크릴레이트(PMMA), 그리고 열린 기공(open-pore) PU를 포함합니다. 이러한 방법을 통해 고도의 정확성을 달성했습니다.

- **Performance Highlights**: 결과적으로 제안된 도구는 수작업으로 수행되는 시간이 많이 소요되는 방법들과 동등한 정확도를 보이며, 단 몇 초 만에 결과를 제공합니다. 이는 자동화된 방식이 다공성 재료의 구조적 특성을 분석하는 데 있어 효율성과 재현성을 극대화할 수 있음을 시사합니다. 따라서, 이 연구는 다공성 재료의 분석 분야에서 중요한 이정표가 될 것으로 기대됩니다.



### Detecting Neurodegenerative Diseases using Frame-Level Handwriting Embeddings (https://arxiv.org/abs/2502.07025)
- **What's New**: 이번 연구는 신경퇴행성 질환 진단을 위한 새로운 접근방식으로써, 필기 신호를 스펙트로그램(spectrogram)으로 나타내어 비침습적인 방식으로 조기 진단 가능성을 탐구하였습니다. 42명의 정상 대조군(CTL), 35명의 파킨슨 병(PD) 환자, 21명의 알츠하이머 병(AD) 환자, 15명의 파킨슨 모방 질환(PDM) 환자를 분석 하였습니다. CNN과 CNN-BLSTM 모델을 사용하여 이진 분류를 수행하였고, 필기 작업과 스펙트로그램 채널 조합이 분류 성능에 미치는 영향을 확인하였습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 존스 홉킨스 병원에서 수집되었으며, 123명이 필기, 음성, 눈 추적을 포함한 14개 작업을 수행하는 과정에서 동기화된 다중 모드 신호를 포함합니다. 필기 데이터는 Wacom One 13 태블릿을 통해 250Hz의 주기로 기록되었으며, 필기 신호의 특징 (x, y 좌표, 압력 등)을 STFT(Short-Time Fourier Transform)로 변환하여 스펙트로그램을 생성하였습니다. 이후 다양한 손글씨 작업을 통해 분류 성능에 미치는 영향을 평가하고, 고정된 크기의 스펙트로그램에 대해 최적의 채널 조합을 추적하였습니다.

- **Performance Highlights**: 연구 결과, AD와 CTL의 구분에서는 89.8%의 최고 F1 점수를 기록하였으며, PD와 CTL 간의 구분에서는 74.5%, PD와 PDM 간의 구분에서는 77.97%의 성과를 보였습니다. CNN 모델은 CNN-BLSTM 모델보다 일관되게 더 우수한 성능을 발휘하였습니다. 다양한 슬라이딩 윈도우 길이에 대한 실험을 통해, AD에는 1초의 윈도우가 최적화되었고, 좀 더 긴 윈도우가 PD 분류 성능을 향상시키는 것으로 나타났습니다.



### Conditional diffusion model with spatial attention and latent embedding for medical image segmentation (https://arxiv.org/abs/2502.06997)
Comments:
          11 pages, 2 figures, 3 tables, Accepted in MICCAI 2024

- **What's New**: 본 논문에서는 의료 영상 분할을 위한 새로운 조건부 확산 모델인 cDAL(conditioned Diffusion Attention Latent)을 제안합니다. cDAL은 각 확산 단계에서 CNN 기반의 판별기를 사용하여 생성된 레이블과 실제 레이블을 구별하고, 이를 통해 더 정확한 영상 분할을 가능하게 합니다. 또한, 각 층에 대해 무작위 잠재 임베딩(random latent embedding)을 도입하여 훈련 및 샘플링 시간 단계를 크게 줄였습니다.

- **Technical Details**: cDAL은 diffusion models의 stochastic(nature)을 활용하여 여러 예측을 생성합니다. 매 시간 단계에서, 판별기의 학습된 공간 주의 맵(spatial attention map)을 사용하여 분할에 주목할 수 있도록 합니다. 확산 과정은 Gaussian noise를 포함한 markov-chain을 사용하며, 이를 통해 데이터 분포의 왜곡을 점진적으로 제거합니다.

- **Performance Highlights**: cDAL은 MoNuSeg, Chest X-ray, Hippocampus의 세 가지 공공 의료 이미지 데이터셋에서 실험을 수행하였으며, 기존의 최첨단 알고리즘에 비해 Dice 점수와 mIoU에서 유의미한 개선을 보였습니다. 이 모델은 고도로 정확한 영상 분할을 제공하여 의료 분야의 자동화 및 효율성을 높일 수 있는 가능성을 제시합니다.



### Universal Vessel Segmentation for Multi-Modality Retinal Images (https://arxiv.org/abs/2502.06987)
- **What's New**: 본 논문에서는 다중 모달리티(retinal images)에서의 혈관 세분화(segmentation)를 위한 보편적인 기초 모델(Foundational Universal Vessel Segmentation Model, UVSM)을 제안합니다. 기존 연구의 한계점으로 단일 모달리티(Color Fundus)에 한정된 점과 새로운 모달리티에 대해 별도의 모델을 미세 조정해야 하는 부담을 지적하고, 다중 모달리티에서 적용 가능한 단일 모델로 해결하는 방법을 설명합니다.

- **Technical Details**: UVSM은 이미지 변환(image translation)과 위상 인식 세분화(topology-aware segmentation)를 통해 다녀하였습니다. 특히, 수많은 카메라에서 수집된 다양한 모달리티 이미지 데이터를 바탕으로 변환 모델을 훈련하고, 후속 단계에서 Topcon CF 도메인에서 모델을 세분화합니다. 위상 인식 세분화는 혈관의 연속성과 분기, 루프 특성을 보존하며 세분화의 토폴로지 정확도를 높입니다.

- **Performance Highlights**: 제안된 모델은 CF, FA, FAF, MC, IR/NIR 등 5개의 일반 사용 모달리티에서 7개의 다양한 데이터셋에서 평가되었으며, 각 데이터셋에 대해 미세 조정된 최신 기술과 유사한 성능을 달성했습니다. 이를 통해, UVSM은 모든 모달리티에서 강력한 혈관 세분화 성능을 제공함을 입증하였습니다. 연구팀은 또한 새로운 데이터셋을 수집하고 주석을 달아, FA, FAF, IR에 대해 더 많은 고품질 데이터셋을 제공하였습니다.



### Generalizable automated ischaemic stroke lesion segmentation with vision transformers (https://arxiv.org/abs/2502.06939)
Comments:
          29 pages, 7 figures, 2 tables, 1 supplementary table, 2 supplementary figures

- **What's New**: 이번 연구는 허혈성 뇌졸중(Ischaemic Stroke)에서의 고성능 DWI(Diffusion-weighted Imaging) 병변(segmentation) 분할 도구를 제안합니다. 이는 최적화된 Vision Transformer 기법을 통해 구축되었으며, 3563개의 주석이 달린 병변 데이터를 다각적으로 통합하여 최신 기술 수준의 결과를 달성했습니다. 또한, 기존의 평균 성능에 국한된 평가 방식의 문제점을 극복하고자 하는 새로운 평가 프레임워크를 도입합니다.

- **Technical Details**: 해당 도구는 U-Net 기반 모델의 법적 한계를 보완하기 위해 설계되었습니다. 평가 프레임워크는 모델의 충실도(fidelity), 인구통계학적 다양성에 대한 공정성(equity), 해부학적 정확성 등 다양한 요소를 통합하여 분석합니다. 자동화된 병변 세분화의 성공적인 구현은 신호 변화(signal dynamics), 기기 변동성(instrumental variability) 및 제한된 주석 데이터와 같은 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: 연구 결과는 이전 기술에 비해 높은 성능을 입증했으며, 개인화된 의학(personalized medicine) 및 기계적 연구(mechanistic research)에 중대한 기여를 할 것으로 기대됩니다. 연구진은 이 프레임워크가 임상 및 연구 분야에서의 활용성을 높여 줄 것으로 믿고 있습니다. 모델의 일반화 가능성(generalizability)과 공정성을 우선시하는 새로운 성능 기준을 설정함으로써, 뇌졸중 이미징 분야의 진전을 이루었다고 할 수 있습니다.



### PyPotteryInk: One-Step Diffusion Model for Sketch to Publication-ready Archaeological Drawings (https://arxiv.org/abs/2502.06897)
- **What's New**: 이번 논문에서는 PyPotteryInk라는 오픈소스 자동화 파이프라인을 소개합니다. 이 시스템은 고고학 도자기 스케치를 표준화된 출판 준비 도면으로 변환하여, 전통적인 수작업 방식의 번거로움을 크게 줄여줍니다. 모형은 신속한 단일 패스 처리를 통해 중요한 형태적 세부 사항들을 보호하며, 학문적 문서화 기준을 준수합니다.

- **Technical Details**: PyPotteryInk는 수정된 img2img-turbo 아키텍처를 기반으로 하며, 효율적인 패치 기반 접근 방식을 사용하여 입력 도면 크기와 관계없이 고해상도 출력을 생성합니다. 이 모델은 입력 도면의 크기와 관계없이 세부 사항을 보존하며, 기초 데이터가 적더라도 다양한 고고학적 문맥에 적응할 수 있도록 미세 조정이 가능합니다. 딥 러닝(Deep Learning)과 전통적인 이미지 처리 기술을 접목하여 도자기 드로잉의 디지털 잉크 처리 과정을 자동화하고 있습니다.

- **Performance Highlights**: 이 연구에서는 이탈리아의 초기 역사 도자기 드로잉 데이터셋을 사용하여 다양한 미세 세부 사항을 잘 캡처하는 성능을 보여주었습니다. 전문가 평가 결과, 생성된 도면은 출판 기준을 충족하며, 처리 시간을 몇 시간에서 몇 초로 단축시킬 수 있음을 입증했습니다. 이 모델은 고고학 연구 커뮤니티 내에서 채택을 용이하게 하기 위해 미리 훈련된 모델과 Python 라이브러리를 제공합니다.



### A Comprehensive Review of U-Net and Its Variants: Advances and Applications in Medical Image Segmentation (https://arxiv.org/abs/2502.06895)
Comments:
          36 pages,26 figures,7 tables

- **What's New**: 이 논문은 U-Net 모델의 구조적 수정 관점에서 의료 영상의 세그멘테이션(segmentation) 성능 향상에 대한 연구를 다루고 있습니다. 기존의 의료 영상 데이터셋을 분류하고, 다양한 U-Net의 향상 모델에 대한 연구 목표와 혁신적인 디자인, 제약 사항들을 상세히 분석합니다. 이를 통해 연구자들에게 보다 효과적이고 안정적인 의료 영상 세그멘테이션 네트워크 모델 개발을 위한 지침을 제공합니다.

- **Technical Details**: 논문에서 다루는 U-Net 개선 메커니즘에는 점프 연결(jump-connection) 및 잔여 연결(residual-connection), 3D-UNet, 그리고 트랜스포머(transformer) 메커니즘이 포함됩니다. 이러한 각 메커니즘은 의료 영상 데이터셋과의 관계를 통해 세그멘테이션 성능을 어떻게 향상시킬 수 있는지를 분석하며, 각 접근 방식의 연구 목표와 제약 사항도 논의됩니다.

- **Performance Highlights**: 의료 영상 세그멘테이션에 있어 U-Net 모델은 최근 몇 년 동안 중요한 성과를 보여주었으며, 정량적인 병변 분석 방법을 위한 기술적 지원을 제공합니다. 이 논문은 U-Net과 그 변형 알고리즘의 핵심 개선 메커니즘을 정리하고, 미래 연구 방향과 전략을 제안함으로써 관련 분야의 연구자들에게 유용한 참고 자료를 제공합니다.



### Survey on Vision-Language-Action Models (https://arxiv.org/abs/2502.06851)
- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델에 대한 AI 생성 리뷰를 소개하며, 주요 방법론, 발견 및 향후 방향을 요약합니다. 이 내용은 대형 언어 모델(LLMs)을 사용해 생성되었으며, 과학 문헌 리뷰 자동화의 가능성을 강조합니다. AI가 생성한 콘텐츠의 정확성과 신뢰성을 보장하는 것은 여전히 도전 과제가 되고 있으며, 향후 연구는 AI 지원 문헌 리뷰를 위한 구조적 프레임워크 개발에 중점을 둘 예정입니다.

- **Technical Details**: 이 연구에서 소개된 Actra는 엣지 디바이스에서 머신 러닝 모델을 배치하기 위한 혁신적인 접근 방식입니다. Actra는 경로 주의(trajectory attention)와 학습 가능한 액션 쿼리(learnable action queries)를 결합하여 로봇 작업의 추론 과정을 최적화합니다. 또한, ProSim-Instruct-520k라는 대규모 다중 모달 데이터셋을 개발하여 로봇 조작 시스템의 학습 능력을 향상시키고 있으며, 이 데이터셋은 520,000개 이상의 실제 주행 시나리오와 1,000만 개 이상의 텍스트 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: Actra는 다양한 로봇 환경에서 실험을 거쳐 성능 및 효율성을 크게 향상시켰음을 보여주었습니다. 특히 복잡한 다중 모달 작업 처리에 효과적이며, 여러 작업 및 객체 카테고리에서 우수한 성공률을 기록했습니다. 결과적으로, Actra는 로봇 조작에 있어 더 효율적이고 강력한 추론 방법을 제공하며, 앞으로의 연구를 위한 대규모 자원을 마련함으로써 데이터 기반 로봇 모델 개선에 기여할 수 있습니다.



### Emotion Recognition and Generation: A Comprehensive Review of Face, Speech, and Text Modalities (https://arxiv.org/abs/2502.06803)
Comments:
          Submitted to ACM Computing Surveys

- **What's New**: 이 논문은 감정 인식 및 생성 기술의 최신 동향을 종합적으로 검토하는 연구입니다. 기존의 연구는 감정 인식과 생성의 개별적인 주제에 국한되어 있었으나, 본 연구는 두 분야의 통합을 강조하며 여러 모달리티에서의 응용 가능성을 탐구합니다. 또한, 다양한 기술적 접근 방식을 분류하고 이론적 기초를 설명하여 연구자들이 이 분야에 대한 명확한 이해를 돕고자 합니다.

- **Technical Details**: 연구는 감정 인식 시스템의 성능을 개선하기 위한 프리프로세싱(preprocessing) 기술을 중점적으로 논의합니다. 얼굴, 음성, 텍스트 데이터를 각각 다루는 다양한 프리프로세싱 기법이 소개되며, 특히 데이터 정규화(normalization), 노이즈 감소(noise reduction), 특징 추출(feature extraction) 등의 기술이 포함됩니다. 이를 통해 모델의 정확도와 효율성을 높이는 방법들이 제시됩니다.

- **Performance Highlights**: 이 연구는 감정 인식 및 생성 기술의 응용 분야가 고객 서비스, 헬스케어, 교육 등에 걸쳐 급속히 확장되고 있음을 강조합니다. 감정 인식 시스템은 고객의 감정을 분석하고, 의료 분야에서 환자의 진전을 추적하는 데 사용되며, 챗봇에서도 활용될 수 있는 전망을 제시합니다. 이는 사용자에게 몰입감 있는 개인화된 경험을 제공할 수 있는 가능성이 높습니다.



### EVEv2: Improved Baselines for Encoder-Free Vision-Language Models (https://arxiv.org/abs/2502.06788)
Comments:
          19 pages, 9 figures

- **What's New**: 본 논문에서는 encoder-free vision-language 모델(이하 VLM)들이 기존의 encoder 기반 모델과의 성능 격차를 좁히고 있다는 점을 강조합니다. 특히, VLM의 구조적 단순성과 효율적 배포 가능성 덕분에 다중 모드 시스템의 가능성이 높아지고 있습니다. EVEv2.0이라는 새로운 VLM 모델 가족을 개발하며, 이는 시각(accommodation)과 언어를 통합한 모델 내에서 간섭(interference)을 줄이는 계층적 연관성을 통해 성능을 향상시킵니다.

- **Technical Details**: EVEv2.0 모델은 사실상 encoder-free VLM 구조를 기반으로 하며, 시각 기능을 분해하고 모달리티 간의 생리학적 간섭을 최소화합니다. 연구 결과, 적절한 훈련 방식이 encoder-free VLM의 효과적인 최적화를 가능하게 함을 알 수 있었습니다. 또한, encoder-free VLM이 계속해서 높은 성능의 기존 encoder 기반 모델과 비슷한 용량의 결과를 보여주며, 데이터를 더욱 효율적으로 활용할 수 있음을 실증했습니다.

- **Performance Highlights**: EVEv2.0는 시각-언어 벤치마크에서 기존의 encoder-free 모델을 초월하며, 다양한 비전-언어 테스트에서 encoder 기반 경쟁 모델과 비슷한 성능을 지속적으로 보여줍니다. 최신 데이터와 강력한 언어 모델을 통해 EVEv2.0은 대량의 훈련 데이터 및 계산 자원을 지원받아 향후 연구에 대한 투명한 로드맵을 제시합니다. 이러한 결과는 확장성과 네이티브 차세대 VLM 개발에 있어 중요한 통찰력을 제공합니다.



### Visual Agentic AI for Spatial Reasoning with a Dynamic API (https://arxiv.org/abs/2502.06787)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 3차원 공간 추론(3D spatial reasoning)을 위한 새로운 접근 방식을 소개합니다. 기존의 제한된 API 대신, LLM(대형 언어 모델) 에이전트들이 협력하여 동적으로 Pythonic API를 생성하여 문제를 해결하는 방식을 채택했습니다. 이를 통해 다양한 쿼리를 처리할 수 있는 능력을 크게 향상시키고 있습니다.

- **Technical Details**: 연구에서는 새로운 벤치마크(benchmark)를 도입하여 3D 이해를 위한 AI의 능력을 평가합니다. 이 벤치마크는 여러 단계의 기초화(grounding)와 추론(inference)을 포함한 쿼리들로 구성됩니다. LLM 에이전트들이 공동으로 프로그래밍을 통해 새로운 기능을 생성하며, 이로 인해 다양한 문제를 해결할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 이전의 제로샷(zero-shot) 모델들보다 3차원 시각 추론에서 더 나은 성능을 보이는 것을 확인했습니다. 이 연구는 3D 공간 추론 작업을 위한 에이전틱(agentic) 프레임워크의 효과성을 실험적으로 검증하였습니다.



### Lumina-Video: Efficient and Flexible Video Generation with Multi-scale Next-D (https://arxiv.org/abs/2502.06782)
- **What's New**: 최근 Diffusion Transformers (DiTs) 기술이 생성 모델링 분야에서 주목받고 있으며, Lumina-Next는 Next-DiT를 통해 photorealistic (포토리얼리스틱) 이미지를 생성하는 뛰어난 성능을 기록합니다. 그러나 비디오 생성의 잠재력은 아직 충분히 탐구되지 않았으며, 영상 데이터 고유의 시공간적 복잡성을 모델링하는 데 많은 도전이 존재합니다. 이를 해결하기 위해, 본 논문에서는 Next-DiT의 강점을 활용한 Lumina-Video 프레임워크를 소개합니다.

- **Technical Details**: Lumina-Video의 핵심은 Multi-scale Next-DiT 구조로, 이는 여러 패치 크기를 동시에 학습하여 효율성과 유연성을 높입니다. 또한, motion score를 명시적 조건으로 포함하여 생성하는 비디오의 동적 정도를 직접 제어할 수 있도록 했습니다. Progressive training과 high FPS를 포함한 훈련 전략을 통해 매우 높은 해상도와 프레임 레이트를 달성하면서도 훈련과 추론 효율성을 높였습니다.

- **Performance Highlights**: Lumina-Video는 다양한 해상도의 고품질 비디오를 생성할 수 있으며, 정량적 지표와 시각적 품질 모두에서 뛰어난 성능을 보여줍니다. 또한, Lumina-V2A를 통해 생성된 비디오에 synchronized sounds (동기화된 소리)를 추가할 수 있는 기능도 포함되어 있습니다. 이 연구는 고급 비디오 생성 기술의 접근성을 높이는 것을 목표로 하며, 연구자들이 Lumina-Video를 다양한 응용 프로그램에서 활용할 수 있도록 코드와 모델 파라미터를 공개합니다.



### KARST: Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission for Visual Classification (https://arxiv.org/abs/2502.06779)
Comments:
          5 pages, 3 figures, Accepted by ICASSP2025

- **What's New**: 본 논문에서는 Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission (KARST)이라는 혁신적인 방법을 소개합니다. 이 방법은 다양한 인식 작업에 적용되며, Kronecker 프로젝션을 수평적으로 확장하고 적응 행렬을 여러 보완 공간으로 분리하여 파라미터 의존성을 줄이고 더 컴팩트한 서브스페이스를 생성합니다. 추가적인 학습 가능한 스케일링 팩터를 도입하여 사전 학습된 feature 분포와 잘 정렬되도록 하여, 더 유연하고 균형 잡힌 feature 집계를 가능하게 합니다.

- **Technical Details**: KARST는 Kronecker 제품의 수학적 성질을 활용하여 다중 커널을 사용하여 더 풍부하고 다양한 특성 공간을 생성합니다. 이는 기존의 파라미터 효율적인 전이 학습(PETL) 접근 방식의 한계를 극복하는 방식으로, 업데이트된 가중치 Delta W를 N개의 Kronecker 공간으로 분해하여 각기 다른 커널이 보완적으로 작용하도록 설계되었습니다. 이러한 다중 커널 접근은 복잡한 도메인 적응을 위한 강력한 스키마를 구축할 수 있게 해줍니다.

- **Performance Highlights**: KARST는 기존의 PETL 기법과 완전 파인튜닝 전략보다 뛰어난 성능을 보여주며 다양한 네트워크 백본에서 매우 낮은 추론 비용으로 강한 능력을 발휘합니다. 실험 결과는 KARST의 성능이 다른 PEFT 기법들과 비교하여 크게 개선되었음을 입증합니다. 이로써 KARST는 특히 컴퓨터 비전 분야에서 매우 효율적이고 실용적인 접근 방식으로 자리잡을 것으로 기대됩니다.



### SAMRefiner: Taming Segment Anything Model for Universal Mask Refinemen (https://arxiv.org/abs/2502.06756)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문에서는 기존의 거친 마스크(coarse masks)의 품질을 향상시키기 위한 새로운 방법인 SAMRefiner를 소개합니다. SAM을 마스크 정제(mask refinement) 작업에 적용하는 보편적이고 효율적인 접근 방식으로, 노이즈 내성을 갖춘 프롬프트(prompts) 생성 기법이 핵심입니다. 이 방법은 기초 마스크로부터 다양한 입력 프롬프트를 채굴하는 멀티 프롬프트 발굴 전략을 도입하여, 인간의 주석 작업을 줄이고 고품질 마스크 생성을 돕습니다.

- **Technical Details**: SAMRefiner는 거리 안내 포인트(distance-guided points), 맥락 인식 탄력적 경계 상자(context-aware elastic bounding boxes), 가우시안 스타일 마스크(Gaussian-style masks) 등을 포함한 다양한 프롬프트를 활용합니다. 특히, 다중 객체 처리에서 발생하는 혼란을 극복하기 위해 split-then-merge (STM) 파이프라인을 소개합니다. 또한, SAMRefiner++는 추가적인 IoU 적응 단계를 도입하여 특정 데이터셋에서의 성능을 높이는 데 기여합니다.

- **Performance Highlights**: SAMRefiner는 여러 벤치마크에서 실험을 통해 뛰어난 정확도와 효율성을 입증했습니다. 특히, WSSIS에서 10% 이상의 품질 향상을 이루며, CascadePSP보다 5배 더 빠른 성능을 보여줍니다. 이 프레임워크는 기존의 세그멘테이션 방법과 유연하게 협력할 수 있는 강력한 범용 후처리 도구로 자리잡고 있습니다.



### Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models (https://arxiv.org/abs/2502.06755)
Comments:
          Main text is 11 pages with 7 figures

- **What's New**: 이 논문은 vision 모델을 이해하기 위해 제어된 실험을 통해 해석된 기능을 검증하는 필요성을 강조합니다. 현재 방식들은 해석 가능한 특징을 제공하지만 인과관계를 테스트할 수 있는 방법이 부족하거나 해석이 불가능한 모델 편집만 가능합니다. 이 연구는 sparse autoencoders (SAEs)를 통해 인간이 해석할 수 있는 시각적 특징을 발견하고 이를 조작하여 모델 행동에 대한 가설을 검증할 수 있는 통합 프레임워크를 제안합니다.

- **Technical Details**: 논문에서는 SAEs가 긴밀하게 얽힌 활성화 벡터를 고차원 희소 표현으로 변환하여 각 비영소 요소가 독특한 의미론적(concept) 개념에 대응되도록 하는 방법을 설명합니다. SAEs는 원래의 호출을 재구성하도록 훈련되며, 비어 있지 않은 요소를 특정하게 수정함으로써 모델 예측의 변화를 관찰할 수 있게 해줍니다. 이러한 구조는 해석과 제어 개입이 통합된 것을 보여주며 다양한 핵심 기여를 통해 그 효과성을 검증합니다.

- **Performance Highlights**: 실험을 통해 SAEs가 CLIP과 DINOv2와 같은 모델 간의 학습된 특징에서의 근본적인 차이를 체계적으로 관찰할 수 있게 해준다는 것을 보여줍니다. 또한, 특정 의미론적 개념을 수정함으로써 예측 가능한 분류 변화를 확인하여 기능의 인과 역할을 검증합니다. 마지막으로, 우리는 모든 비전 트랜스포머와 함께 사용할 수 있는 공개적인 확장 가능한 코드베이스를 제공하여 현대 비전 모델의 광범위한 과학적 연구를 가능하게 합니다.



### Accelerating Data Processing and Benchmarking of AI Models for Pathology (https://arxiv.org/abs/2502.06750)
- **What's New**: 새로운 소프트웨어 도구 세트를 소개하여 전체 슬라이드 이미지 처리(whole-slide image processing), foundation model benchmarking, 그리고 공개된 작업(task)들을 정리하였습니다. 이는 기존의 모델 수가 증가하고 표준화된 벤치마크의 부족 문제를 해결하는 데 도움을 줄 것입니다. 이러한 새로운 자원들은 투명성(transparency)과 재현성(reproducibility)을 촉진하고, 컴퓨터 병리학(computational pathology) 분야의 지속적인 발전(progress)을 지원할 것으로 기대됩니다.

- **Technical Details**: 이 소프트웨어 도구들은 다양한 모델의 강점(strengths)과 한계(limitations)를 평가하고, 각 모델의 발전 가능성(potential)을 표준화된 기준을 통해 명확하게 파악할 수 있도록 돕습니다. 이를 통해 연구자들은 모델의 성능을 직관적으로 이해하고, 향후 연구 방향을 설정할 수 있습니다. 특히, 메트릭스(metrics)와 데이터셋(datasets)의 표준화가 이루어져 평가의 일관성을 높입니다.

- **Performance Highlights**: 이 도구들의 도입으로 인해 모델 평가의 효율성이 향상되고, 연구 결과의 신뢰성이 증가할 것입니다. 또한, 연구자들은 공용 데이터와 작업을 통해 자신의 모델을 테스트하고 비교할 수 있어, 연구의 연속성과 발전에 기여할 것입니다. 이는 컴퓨터 병리학 분야에서의 협업(collaboration)과 발전을 가속화하는 중요한 척도가 될 것입니다.



### Wandering around: A bioinspired approach to visual attention through object motion sensitivity (https://arxiv.org/abs/2502.06747)
- **What's New**: 이 논문은 동적 시각 인식을 가능하게 하는 능동 비전(active vision) 시스템의 개발을 다룹니다. 이 시스템은 생물학적 선택적 주의 메커니즘을 통합하여, 비동기적 장면 변화를 포착하는 이벤트 기반 카메라(event-based cameras)를 사용하며, 이를 통해 낮은 지연(latency)으로 실시간 처리할 수 있습니다. 특히, 스파이킹 신경망(Spiking Neural Networks)을 활용하여 합성곱 신경망(Convolutional Neural Network)을 구성한 주목 시스템을 제시하여, 객체의 움직임에 민감하게 반응하도록 설계되었습니다.

- **Technical Details**: 논문에서는 스페크 신경형 하드웨어(Speck neuromorphic hardware)에 통합된 동적 시각 센서(Dynamic Vision Sensor)를 활용하여 고정 안구 운동(fixational eye movements)을 통해 이벤트를 생성합니다. 이 시스템은 다중 객체 움직임 분할(multi-object motion segmentation)에서 평균 교차 면적(Mean IoU) 82.2% 및 구조 유사도 지수(SSIM) 96%를 달성하며, 사무실 및 저조도 상황에서 평균 88.8% 및 89.8%의 정확도로 눈에 띄는 객체를 탐지합니다. 또한, 이 시스템은 동적인 장면에 0.12초의 반응 시간을 보여주며, 인지 자원을 효율적으로 활용할 수 있는 설계로 되어 있습니다.

- **Performance Highlights**: 이 시스템은 특히 고정 카메라 시스템의 한계를 극복하여 복잡한 환경에서도 효과적으로 대응할 수 있는 유연성을 갖추고 있습니다. 이벤트 기반 카메라의 사용은 데이터 처리 및 저장 문제를 최소화하고 실시간 시각 인식(application)에서 매끄럽고 빠른 반응을 제공합니다. 또한, 학습이 필요 없는 설계를 통해 감각적 장면 전반에서 강력한 성능을 발휘하며, 로봇 응용을 위한 신뢰할 수 있는 기반이 됩니다.



### ViSIR: Vision Transformer Single Image Reconstruction Method for Earth System Models (https://arxiv.org/abs/2502.06741)
- **What's New**: 새로운 연구에서는 Earth System Model (ESM) 데이터를 위한 비전 변환기 기반의 Sinusoidal Representation Networks (ViSIR) 모델을 제안합니다. 이 모델은 고해상도 이미지를 복원하는 single image Super Resolution (SR) 작업을 개선하기 위해 Vision Transformer (ViT)의 SR 능력과 SIREN의 고주파 세부 정보 보존을 결합합니다. 제안된 ViSIR은 다양한 벤치마크에서 기존의 SR 모델보다 4.1 dB에서 7.5 dB까지 PSNR 성능이 향상됨을 보여주었습니다.

- **Technical Details**: ViSIR 모델은 ViT의 글로벌 의존성과 SIREN의 세밀한 고주파 세부정보 표현력을 접목하여 SR 범주에서의 스펙트럴 편향 문제를 해결합니다. 모델은 저해상도 ESM 이미지로부터 고해상도를 생성하는 깊은 신경망 아키텍처를 사용하며, 이를 통해 더 나은 이미지 품질을 달성합니다. ViT는 입력 이미지에서 장기 의존성을 학습하였으며, SIREN은 최종적으로 고주파 세부정보를 포착하여 이미지를 정교화합니다.

- **Performance Highlights**: ViSIR의 성능은 반복된 실험을 통해 확인되었으며, Mean Square Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) 등의 지표에서 기존 방법들보다 우수한 결과를 기록했습니다. ViSIR은 E3SM 모델을 포함한 다양한 ESM 데이터셋에서 적용 가능성을 보여주었습니다. 이 연구는 기후 변화 연구와 관련된 고해상도 이미지 생성의 필요성을 충족합니다.



### Enhancing Pneumonia Diagnosis and Severity Assessment through Deep Learning: A Comprehensive Approach Integrating CNN Classification and Infection Segmentation (https://arxiv.org/abs/2502.06735)
- **What's New**: 이번 연구는 폐렴(pneumonia)을 진단하기 위해 두 가지 독립적인 심층 학습(deep learning) 모델을 개발하였습니다. 첫 번째 모델은 폐렴을 분류하는 것을 목표로 하고, 두 번째 모델은 감염의 심각도를 평가하기 위해 세분화를 수행합니다. 이 연구는 COVID-19의 진단 평가에서 유용한 통찰력을 제공하며, 의료 전문가들이 폐렴을 더 효과적으로 이해하고 치료하는데 도움을 줍니다.

- **Technical Details**: 연구자는 폐렴 분류를 위한 CNN(Convolutional Neural Network) 모델과 감염 세분화를 위한 모델을 개발했습니다. 또한, UNET 모델의 인코더를 활용하여 감염 세분화 모델을 정교하게 조정함으로써 감염 영역에 집중하여 클래스 정확도를 높였습니다. VGG16 인코더 아키텍처 위에 DenseNet-121에서 영감을 받은 밀집 블록(dense block)을 통합하여 특성 추출 및 분류 성능을 향상시키는 방식을 채택하였습니다.

- **Performance Highlights**: COVID-19 및 폐렴 진단에서의 높은 분류 정확도를 달성하였으며, 이는 제한된 의료 데이터 및 X-ray 영상 품질 변동성을 효과적으로 해결했습니다. 연구 결과는 CNN 모델이 의료 영상 진단에서 가지는 높은 정확성을 강조하며, 특히 작은 데이터셋에서도 우수한 성과를 보였습니다. Grad-CAM 시각화를 통해 모델의 예측 투명성을 높여 임상의들이 모델의 의사결정 과정을 더 잘 이해할 수 있도록 하였습니다.



### Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists (https://arxiv.org/abs/2502.06734)
- **What's New**: 이번 논문에서는 비디오 생성(video generation) 기술의 최근 발전을 바탕으로 비디오 편집(video editing) 기법의 새로운 접근 방법을 제시합니다. 특히, 고품질 비디오 편집 데이터셋인 Señorita-2M을 소개하며, 이 데이터셋은 약 200만 개의 비디오 편집 쌍(video editing pairs)으로 구성되어 있습니다. 각 비디오 편집 모델은 현재의 최첨단 편집 결과를 달성하기 위해 저자 팀에 의해 정교하게 구축되고 훈련되었습니다.

- **Technical Details**: 데이터셋인 Señorita-2M은 전통적인 inversion-based 방법과 end-to-end 방법의 한계를 극복하기 위해 설계되었습니다. inversion-based 방법은 유연성이 있지만 추론(inference) 시간 소비가 크고 세부 편집 지침(fine-grained editing instructions)에 대해 어려움이 있습니다. 반면, end-to-end 방법은 훈련을 위해 편집된 비디오 쌍에 의존하나 고품질 훈련 비디오 쌍이 부족하여 낮은 편집 결과를 초래합니다.

- **Performance Highlights**: 저자들은 새로운 필터링 파이프라인(filtering pipeline)을 제안하여 poorly edited video pairs를 제거하며, 다양한 비디오 편집 아키텍처를 탐색하여 현재의 pre-trained generative model에 기반하여 최적의 구조를 도출했습니다. 실험 결과, Señorita-2M 데이터셋은 비디오 편집 결과의 품질을 크게 향상시키는 데 기여할 수 있음을 입증했습니다.



### Learning Musical Representations for Music Performance Question Answering (https://arxiv.org/abs/2502.06710)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 논문에서는 음악 공연을 위한 Audio-Visual Question Answering (AVQA) 문제를 해결하기 위한 Amuse 프레임워크를 제안합니다. 기존의 일반 AVQA 방법들이 음악 공연의 특정 특성을 적절히 다루지 못했던 문제를 해결하기 위해, 음악의 리듬과 소스 정보를 명시적으로 학습할 수 있도록 Music AVQA 데이터셋을 보강하고 출시했습니다. 또한, 시간에 민감한 오디오-비주얼 모델링을 위해 음악 예측을 시간 차원과 정렬하는 방법을 도입했습니다.

- **Technical Details**: Amuse 프레임워크는 기존의 단일 모달 또는 비상호작용 형태의 인코더 사용에서 벗어나 세 가지 상호작용하는 변환기를 통해 작업을 수행합니다. 이 세 가지 변환기는 각각 음악 비디오, 음악 오디오, 음악 관련 질문을 처리하며, 교차 모달 어댑터를 사용하여 초기 방식에서부터 세 가지 모달 간의 상호작용을 촉진합니다. 교차 모달 어댑터는 각 모달리티에서의 토큰에 대하여 교차 모달 어텐션을 발휘하여 상호작용을 강화합니다.

- **Performance Highlights**: Amuse 프레임워크는 Music AVQA 데이터셋에서 최첨단 성능을 보여주며, 음악 공연에 대한 질문을 보다 정확하게 응답할 수 있는 가능성을 제시합니다. 실험 결과에 따르면, 이 프레임워크는 기존 방법들과 비교했을 때 훨씬 더 높은 정확도를 기록하였으며, 이는 음악 관점에서의 모달 상호작용을 성공적으로 캡처했음을 나타냅니다. 연구팀은 코드와 데이터셋을 GitHub를 통해 공개하여, 이 분야의 더 많은 연구자들이 활용할 수 있도록 하였습니다.



### TEMSET-24K: Densely Annotated Dataset for Indexing Multipart Endoscopic Videos using Surgical Timeline Segmentation (https://arxiv.org/abs/2502.06708)
- **What's New**:  이 연구에서는 TEMSET-24K라는 오픈 소스 데이터를 새롭게 소개하고 있습니다. 이 데이터셋은 24,306개의 trans-anal endoscopic microsurgery (TEMS) 비디오 클립으로 구성되어 있으며, 각 클립은 임상 전문가에 의해 계층적 레이블 텍스트를 이용해 세심하게 주석이 달려 있습니다. 고해상도 내시경 수술 비디오의 중요성은 수술 데이터 과학의 기초적인 요소로서, 이를 통해 임상의 성능 평가와 체계적인 데이터 분석의 가능성을 높이고자 합니다. TEMSET-24K는 수술 데이터 과학의 최첨단 솔루션을 발전시키는 중요한 기준을 제공합니다.

- **Technical Details**:  TEMSET-24K 데이터셋의 주석은 각 클립의 수술 과정에 대한 복잡한 워크플로우를 포착하는 데 중점을 두고 있습니다. 연구에서는 ConvNeXt, ViT, SWIN V2와 같은 다양한 딥러닝 모델과 transformer 기반 아키텍처를 평가하여 ESV의 타임라인 세분화를 수행하였습니다. 실험 결과, Setup 및 Suturing과 같은 주요 단계에서 최대 0.99의 정확도와 F1 점수를 기록하였습니다. 이는 자동화된 비디오 분석 기법이 수술 이해도를 향상시키는 데 기여할 가능성을 입증합니다.

- **Performance Highlights**:  ESV 타임라인 세분화에서 STALNet 모델은 여러 인코더를 활용했으며, 일관되게 잘 나타낸 단계들을 성공적으로 분리하였습니다. 소개된 데이터셋은 고해상도 ESV의 세분화와 인덱싱을 통한 검색 기능의 개선에 기여하고자 합니다. SurgiFlow라는 디지털 플랫폼도 제안되어, 임상 환경에서 대량의 ESV 데이터 뱅크를 효율적으로 검색할 수 있도록 돕습니다. 이러한 모든 기여는 수술 표준을 높이는 데 필수적인 역할을 할 것으로 기대됩니다.



### Transfer Your Perspective: Controllable 3D Generation from Any Viewpoint in a Driving Scen (https://arxiv.org/abs/2502.06682)
- **What's New**: 이 연구에서는 자율주행차량(Autonomous Vehicles, AV)의 협력적 인식(Collaborative Autonomous Driving, CAV)을 위한 새로운 접근법인 'Transfer Your Perspective (TYP)'를 제안합니다. TYP는 이동차량의 센서 데이터를 기반으로 다양한 관점에서 실제적인 인식을 생성하는 혁신적 방법으로, 기존의 자율주행 데이터셋의 한계를 극복하기 위한 것입니다. 새로운 방식을 통해 TYP는 단일 에고카 데이터셋을 협력적 드라이빙 데이터로 변환할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: TYP는 조건화된 확산 모델(Conditioned Diffusion Model)을 학습하여 에고카 데이터와 일관된 세멘틱(semantics) 및 레이아웃을 가진 출력 샘플을 생성합니다. 연구는 현실적인 3D 포인트 클라우드를 기반으로 협력적 드라이빙 시나리오에서 데이터를 효과적으로 생성할 수 있는 방법론을 사용합니다. 기존의 DMs(Diffusion Models)와는 달리, TYP는 주어진 데이터에서 다양한 시각적 관점을 받아들여 협업형 인식 알고리즘을 훈련할 수 있습니다.

- **Performance Highlights**: 실험 결과 TYP는 협력적 인식 알고리즘을 사전 훈련(pre-train)하는 데 효과적임을 입증했습니다. 특히, TYP는 실제 협력적 데이터가 거의 없거나 전혀 없이 초기 및 최종 융합 알고리즘을 훈련할 수 있도록 도와주어 CAV의 다운스트림 애플리케이션(Downstream Applications)을 크게 촉진합니다. 결과적으로 TYP는 자율주행차량의 성능 향상에 기여할 수 있는 혁신적인 솔루션으로 자리잡을 것으로 기대됩니다.



### CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis (https://arxiv.org/abs/2502.06681)
- **What's New**: 이번 연구에서는 장기 동작 모니터링 및 개인 재식별(person re-identification, Re-ID)을 위한 새로운 데이터셋인 CHIRLA를 소개합니다. 이는 서로 연결된 실내 환경에서 개인의 외모 변화를 포착할 수 있도록 7개월에 걸쳐 수집된 영상 자료를 포함합니다. 데이터셋은 22명의 참가자와 7개의 카메라로 구성되어 있으며, 각각의 외형 변화에 대한 체계적인 기록을 가지고 있습니다.

- **Technical Details**: CHIRLA 데이터셋은 다양한 카메라 배치와 물리적 환경을 통해 사무실 내의 다중 방을 포함한 복잡한 구조를 나타냅니다. 카메라는 Reolink RLC-410W 모델을 사용하며, 5.0 메가픽셀 해상도와 30 fps의 일관된 비디오 캡처를 제공합니다. 데이터 수집은 이더넷을 통해 이루어져 최대의 안정성을 확보하였으며, 전체 영상의 압축은 H.264 형식을 사용하고 있습니다.

- **Performance Highlights**: 이 데이터셋은 장기 재식별 시나리오에서의 알고리즘 성능 평가에 유용한 기준을 제공하는 것을 목표로 합니다. 참가자들은 영상撮影 세션 사이에 입은 옷을 변경하였고, 이를 통해 기존 Re-ID 알고리즘의 한계에 도전하는 높은 수준의 변동성을 도입하였습니다. CHIRLA는 고신뢰성이 요구되는 실제 응용 프로그램에서 Re-ID 모델을 발전시키는 데 중요한 기여를 할 것입니다.



### Prototype Contrastive Consistency Learning for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2502.06650)
Comments:
          17 pages, 10 figures, 7 tables

- **What's New**: 본 논문에서 제안하는 Prototype Contrastive Consistency Segmentation (PCCS) 방법은 반지도 학습 (semi-supervised learning) 환경에서 의료 이미지 분할 (medical image segmentation)의 성능을 개선할 수 있도록 설계되었습니다. 기존의 대다수 방법이 라벨이 없는 이미지의 전체 맥락 정보를 무시하고 있는 문제점에 주목하며, 동일 의미 클래스의 프로토타입을 근접하게 하고, 다른 클래스의 프로토타입을 멀리하는 방식으로 개선을 시도합니다. 이 접근법은 대칭거리 맵 (signed distance map) 및 불확실성 맵 (uncertainty map)을 활용하여 정확한 분할을 위한 정보를 더욱 효율적으로 추출하도록 합니다.

- **Technical Details**: PCCS는 학생-교사 구조 (student-teacher architecture)를 기반으로 한 프로토타입 업데이트 메커니즘을 도입하여 프로토타입을 지속적으로 갱신합니다. 또한, 라벨이 없는 데이터로부터 보다 신뢰할 수 있는 정보를 추출하기 위해 불확실성 일관성 손실 (uncertainty-consistency loss)을 설계하여 노이즈를 줄이려는 노력을 기울였습니다. 이러한 요소들이 결합되어 더 나은 프로토타입을 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과 PCCS는 여러 의료 이미지 분할 데이터 세트에서 최신 기술들 (state-of-the-art methods) 대비 뛰어난 성능을 달성하였습니다. 이 방법은 고품질의 추정 라벨을 생성하면서 프로토타입의 구별 능력을 크게 향상시켜 intra-class 피처를 더욱 조밀하게 만들고, inter-class 피처를 분리하는 데 성공하였습니다. 앞으로 PCCS 방식은 반지도 학습 및 이미지 분할 분야에서 더욱 널리 사용될 것으로 기대됩니다.



### Few-Shot Classification and Anatomical Localization of Tissues in SPECT Imaging (https://arxiv.org/abs/2502.06632)
Comments:
          2 pages, 2 figures

- **What's New**: 본 연구는 제한된 라벨 데이터로부터 의료 이미징에서의 분류(classification) 및 해부학적 로컬라이제이션(localization)에 대한 도전 과제를 해결하기 위해 Prototypical Networks와 Propagation-Reconstruction Network (PRNet)을 각각 적용하였습니다. 특히, Single Photon Emission Computed Tomography (SPECT) 이미지를 활용하여 심장 주변의 2D 슬라이스 이미지를 대상으로 한 개념 증명(proof of concept)을 수행했습니다.

- **Technical Details**: Prototypical Network는 사전 학습된 ResNet-18 백본을 사용하여 심실(ventricle), 심근(myocardium), 간(liver) 조직을 분류하였으며, 훈련 세트에서 96.67%, 검증 세트에서 93.33%의 정확도를 달성했습니다. PRNet은 인코더-디코더(encoder-decoder) 아키텍처와 스킵 연결(skip connections)을 통해 2D SPECT 이미지의 해부학적 위치를 정확하게 예측하는 데 성공했습니다.

- **Performance Highlights**: Prototypical Network는 제한된 라벨 데이터로부터 조직 분류의 가능성을 보여주었으며, PRNet은 해부학적 랜드마크 로컬라이제이션에서 뛰어난 성능을 발휘했습니다. 이는 향후 딥 러닝 프레임워크의 성능 향상을 위한 기반을 마련할 수 있습니다.



### Conformal Predictions for Human Action Recognition with Vision-Language Models (https://arxiv.org/abs/2502.06631)
Comments:
          6 pages, 7 figures

- **What's New**: 본 논문에서는 Human-In-The-Loop (HITL) 프레임워크에 Conformal Predictions (CP)를 적용하여 Human Action Recognition (HAR) 작업에서의 성능을 향상시키는 방법을 제안합니다. 연구 결과, CP는 비디오 클립을 위한 후보 클래스의 평균 수를 크게 줄일 수 있음을 보여주었습니다. 또한, CP를 통해 생성된 클래스 세트에서는 긴 꼬리 분포가 나타나는 경향이 있습니다.

- **Technical Details**: 이 연구에서는 사전 훈련된 Vision-Language Models (VLMs) 위에 CP를 적용하는 방법을 탐구하며, 추가적인 파인튜닝 없이도 효과적인 HAR 분류 작업이 가능함을 발견했습니다. CP는 클래스에 대한 라벨 세트를 제공하는 동시에, 진짜 클래스가 포함될 확률에 대한 확고한 보장을 제공하는 특성이 있습니다. VLM의 온도 매개변수를 조정하여 CP의 긴 꼬리 분포를 최소화하는 방법도 제시합니다.

- **Performance Highlights**: 핵심적으로, 연구팀은 CP가 후보 클래스를 상당히 감소시킬 뿐만 아니라, 인간의 주석 시간 단축에도 기여할 수 있음을 입증하였습니다. 이 접근법은 비디오 모니터링과 같이 제한된 결정 시간을 요구하는 응용 프로그램에서 특히 유용합니다. GitHub에 공개된 코드와 함께 연구 결과를 통해 이 새로운 방법이 실제론에서 어떻게 적용될 수 있는지 보여주고 있습니다.



### Unleashing the Potential of Pre-Trained Diffusion Models for Generalizable Person Re-Identification (https://arxiv.org/abs/2502.06619)
- **What's New**: 이 연구는 도메인 일반화 가능한 사람 재식별(DG Re-ID) 기술을 향상시키기 위한 새로운 방법인 DCAC(디퓨전 모델 보조 표현 학습과 상관 인식 조건 scheme)을 제안합니다. 기존의 기법들이 특성 표현 학습에서 자주 단기 학습을 방지하는 데 실패하는 반면, DCAC는 분류 확률과 학습 가능한 ID 수준의 프롬프트를 결합하여 향상된 모델 성능을 이끌어냅니다. 이를 통해 디퓨전 모델로부터 피드백이 발생하여 DG Re-ID의 일반화 능력이 극적으로 향상됩니다.

- **Technical Details**: DCAC 방식은 미리 훈련된 디퓨전 모델과 경쟁적 및 대조적 Re-ID 모델을 통합하여 상관 인식 조건을 통해 연결합니다. 이 기법은 ID 분류 확률을 바탕으로 한 다크 지식을 활용하여 디퓨전 프로세스를 인도하며, 이를 통해 데이터에서 스타이징 리스크를 줄이는 정교한 방식으로 작용합니다. 또한, LoRA(Low-Rank Adaptation) 어댑터를 사용하여 파라미터 효율적인 세부 조정을 수행함으로써, 디퓨전 모델이 Re-ID 데이터에 효과적으로 적응하도록 돕습니다.

- **Performance Highlights**: 단일 소스 및 다중 소스 DG Re-ID 작업에서 실시한 포괄적인 실험을 통해, 제안한 방법이 최신 기술 수준의 성능을 기록했습니다. 연구 결과는 제안한 DCAC 시스템이 복잡한 ID 간 관계를 효과적으로 캡처하며, Re-ID 모델의 일반화 성능을 개선하는 데 기여함을 입증합니다. 추가로 다수의 아블레이션 연구를 통해 이 접근 방식의 강인성을 증명했습니다.



### Multi-Scale Feature Fusion with Image-Driven Spatial Integration for Left Atrium Segmentation from Cardiac MRI Images (https://arxiv.org/abs/2502.06615)
- **What's New**: 이 연구에서는 심장 자기공명영상(CMR)에서 좌심방(LA)을 정확히 분할하기 위한 자동화된 방법을 제안합니다. 특히, DINOv2라는 클래스 비독립적인 기초 모델을 인코더로 사용하고 UNet 스타일 디코더와 결합하여 다중 스케일 특징 융합을 활용함으로써 정확성을 향상시킵니다. 이를 통해, 현재의 수동 분할 접근법의 비효율성과 변동성을 극복할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 DINOv2 인코더와 UNet 디코더를 통합하여 특징 추출과 고해상도 세부 정보를 보존합니다. 다층 특징 융합 전략을 통해 각 변환기 블록의 기여도를 반영하는 학습 가능한 가중치를 도입합니다. 이렇게 구성된 네트워크는 인코딩 과정에서 잃어버리기 쉬운 세부적 구조 정보를 복원하여 LA의 분할 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 접근 방식은 LAScarQS 2022 데이터셋에서 검증되었으며, giant 아키텍처의 경우 92.3% Dice와 84.1% IoU 점수를 기록하여 기존의 nnUNet 기반 모델보다 우수한 성능을 나타냈습니다. 이 결과는 심장 자기공명영상에서의 좌심방 자동 분할 성능의 향상을 강조하며, 의료 영상 처리 분야에 기여할 수 있는 가능성을 시사합니다.



### TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models (https://arxiv.org/abs/2502.06608)
- **What's New**: 최근 확산(diffusion) 기술의 발전은 이미지 및 비디오 생성의 품질을 두 배로 향상시키며 생성 AI의 적용을 가속화하고 있습니다. 그러나 3D 형상 생성 기술은 3D 데이터의 양과 처리의 복잡성 덕분에 여전히 지체되고 있습니다. 이 논문에서는 이미지에 정밀하게 대응하는 고충실도 3D 메시를 생성하는 새로운 비대칭 모형인 TripoSG를 소개합니다.

- **Technical Details**: TripoSG는 대규모 정제된 흐름 변환기(rectified flow transformer)를 사용하는 혁신적인 3D 생성 모델로, SDF(Signed Distance Function), 법선(normal) 및 에이코날 손실(eikonal loss)을 조합한 하이브리드 감독 훈련 전략이 특징입니다. 이 모델은 4B 매개변수를 가진 고해상도 3D 구조물을 생성하며, 양질의 3D 데이터를 체계적으로 구축하는 데이터 처리 파이프라인을 개발하였습니다. 이러한 기술적 도약은 특히 3D 생성을 위한 분산(hybrid) 접근 방식을 가능하게 했습니다.

- **Performance Highlights**: TripoSG는 새로운 상태의 성능(SOTA)을 달성하며, 입력 이미지와의 정확한 정렬 및 뛰어난 일반화 능력을 보여줍니다. 이 모델은 다양한 이미지 스타일 및 콘텐츠로부터 3D 모델을 생성할 수 있습니다. 또한, 2백만 개의 고품질 3D 샘플을生成하여 학습 과정을 위한 데이터 품질 및 양의 중요성을 강조하고 있습니다.



### Illegal Waste Detection in Remote Sensing Images: A Case Study (https://arxiv.org/abs/2502.06607)
- **What's New**: 이번 논문은 환경 범죄 중 하나인 불법 쓰레기 투기 문제를 해결하기 위해 고해상도 원거리 감지(Very-High-Resolution Remote Sensing) 이미지를 활용한 새로운 파이프라인을 제안합니다. 이 시스템은 지역 환경 기관과의 협력을 기반으로 개발되었으며, 불법 매립지 탐지의 효율성을 크게 높입니다. 또한, 자동화된 분석 과정이 수동 사진 해석에 비해 시간 절약을 돕는다는 점에서 주목할 만합니다.

- **Technical Details**: 이 연구에서는 다양한 이미지 특성과 훈련 설정의 영향을 분석하기 위해 포괄적인 실험 세트를 수행했습니다. 원거리 감지 이미지(classifier of Remote Sensing images)에 대한 분류기의 최적 구성을 식별하기 위해 세부적인 분석을 진행하였으며, 이를 통해 환경 기관의 전문가들이 일상 작업에 통합할 수 있도록 했습니다.

- **Performance Highlights**: 개발된 분류기는 훈련 영역을 벗어난 장소에서도 유효한 결과를 도출하며, 이는 제안된 파이프라인의 국경 간 적용 가능성을 강조합니다. 이 연구는 불법 쓰레기 투기와 같은 환경 범죄에 대한 감시 및 대응에 실질적인 기여를 할 것으로 기대됩니다.



### MaterialFusion: High-Quality, Zero-Shot, and Controllable Material Transfer with Diffusion Models (https://arxiv.org/abs/2502.06606)
- **What's New**: 이번 논문에서는 이미지 내 객체의 재질(재료) 전송 기술인 MaterialFusion을 소개합니다. 이 새로운 프레임워크는 사용자에게 재질 적용의 정도를 조절할 수 있는 기능을 제공하여, 새로운 재질 특성과 객체의 원래 특징 간의 최적 균형을 이룹니다. MaterialFusion은 배경 일관성을 유지하고 경계 인공물(artifacts)을 완화하여 수정된 객체를 원활하게 통합할 수 있습니다.

- **Technical Details**: MaterialFusion은 IP-Adapter와 Guide-and-Rescale(GaR) 방법을 확산 모델(diffusion model) 내에서 결합하여 고품질 재질 전송을 달성합니다. 이 접근법은 원본 exemplar 이미지에서 재질의 특성을 인코딩하여 전송할 재질의 특정 질감과 뉘앙스를 포착합니다. 또한, GaR은 대상 객체의 기하학적 특징과 필수적인 요소를 유지하여 원래 구조와 세부사항을 보존하는 데 도움을 줍니다.

- **Performance Highlights**: MaterialFusion은 질적, 사용자 제어, 배경 보존 측면에서 기존 방법들보다 현저히 뛰어난 성능을 보여줍니다. 연구팀은 실제 재질 전송 사례들의 데이터셋을 수집하고 포괄적인 비교 분석을 수행하여 이 사실을 입증했습니다. 이 연구는 앞으로의 디자인 워크플로우를 가속화하고 합성 이미지의 사실감을 높이는 데 중요한 기여를 할 것입니다.



### A Large-scale AI-generated Image Inpainting Benchmark (https://arxiv.org/abs/2502.06593)
- **What's New**: 이 논문에서는 95,000개 이상의 인페인팅(inpainting) 이미지를 포함하는 새로운 데이터셋인 DiQuID를 제안합니다. 이 데이터셋은 MS-COCO, RAISE, OpenImages에서 가져온 78,000개의 원본 이미지를 기반으로 생성되었습니다. 특히, 인페인팅 생성 과정에서 세밀한 프롬프트(prompt)를 사용하고 여러 최신 모델을 활용하여 다양성 및 품질을 극대화했습니다. 이를 통해 기존 데이터셋보다 더 높은 다양성과 미적 품질을 보여줍니다.

- **Technical Details**: 제안된 데이터셋 생성 방법론은 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Semantically Aligned Object Replacement (SAOR), (2) Multiple Model Image Inpainting (MMII), (3) Uncertainty-Guided Deceptiveness Assessment (UGDA)입니다. SAOR는 인스턴스 분할(instance segmentation)을 통해 적절한 객체를 찾아내고, MMII는 최신의 확산 기반 모델을 사용하여 다양한 조작을 수행합니다. 마지막으로, UGDA는 원본과 비교하여 이미지의 사실성(realism)을 평가하는 방법입니다.

- **Performance Highlights**: DiQuID 데이터셋은 여러 최신 변조 탐지(detection) 모델을 사용하여 포괄적인 벤치마킹 결과를 제공합니다. 이 연구에서는 42명의 참가자와 함께 1,000개의 이미지를 대상으로 한 인체 연구를 통해 사람의 판단과 모델 성능 간의 상관관계를 조사했습니다. 결과적으로, 기존 데이터셋에 비해 훈련된 모델이 더 어려운 사례에서 높은 성능을 유지함을 보여주었습니다.



### Adaptive Perception for Unified Visual Multi-modal Object Tracking (https://arxiv.org/abs/2502.06583)
- **What's New**: 최근 다중 모달 추적기들은 RGB를 주요 모달리티로 삼고, 다른 모달리티는 보조적으로 간주하며 각기 다른 모달 작업을 별도로 미세 조정하는 경향이 있습니다. 이러한 불균형은 각 모달리티의 상호 보완적인 정보를 동적으로 활용하는 능력을 제한하며, 복잡한 시나리오에서 이를 최대한 활용하기 어렵게 만듭니다. 이를 해결하기 위해 APTrack라는 새로운 통합 추적기를 제안하며, 이는 기존 방법과 달리 모달리티 간의 동등한 모델링 전략을 탐색합니다.

- **Technical Details**: APTrack은 적응형 인식을 통해 여러 모달리티를 통합하는 통합 다중 모달 추적기입니다. 이 모델은 추가적인 미세 조정 없이 다양한 모달리티 및 작업에 동적으로 적응할 수 있도록 설계되었습니다. 또한, 적응형 모달 상호작용(AMI) 모듈을 통합하여 서로 다른 모달리티 간의 상호작용을 효율적으로 연결하고 학습 가능한 토큰을 생성하여 기능적 통합을 촉진합니다.

- **Performance Highlights**: APTrack은 RGBT234, LasHeR, VisEvent, DepthTrack, VOT-RGBD2022 등 다섯 개의 다양한 다중 모달 데이터셋을 활용한 실험을 통해 기존의 최첨단 통합 다중 모달 추적기를 초월하는 성능을 보여주었습니다. 또한 특정 다중 모달 작업을 위해 설계된 추적기보다도 우수한 성능을 발휘하여 다중 모달 추적 분야에서 그 가능성과 장점을 입증합니다.



### Diffusion Models for Computational Neuroimaging: A Survey (https://arxiv.org/abs/2502.06552)
Comments:
          9 pages, 1 figure

- **What's New**: 이 논문은 최근의 diffusion 모델을 컴퓨터 신경영상(computational neuroimaging) 분야에 적용하는 노력들을 통합하여 분석합니다. 특히, 이 모델들이 데이터 향상(data enhancement), 질병 진단(disease diagnosis), 뇌 해독(brain decoding)과 같은 다양한 신경학적 작업을 해결하는 데 어떻게 기여할 수 있는지를 탐구합니다. 각 응용 분야에서 모델의 조건화 메커니즘과 생성 타겟에 따른 다양한 변형들을 소개하고 있습니다.

- **Technical Details**: 논문에서는 Magnetic Resonance Imaging (MRI), Functional Magnetic Resonance Imaging (fMRI), Computed Tomography (CT), Diffusion Tensor Imaging (DTI), Electroencephalogram (EEG), Positron Emission Tomography (PET) 등의 여러 신경영상 데이터 형식과 처리 방법을 개괄합니다. 또한 Denoising Diffusion Probabilistic Models (DDPM), Score-Based Generative Models (Score-SDE), Denoising Diffusion Implicit Models (DDIM) 및 Latent Diffusion Models (LDMs) 등의 다양한 diffusion 모델 설계와 변형을 살펴봅니다.

- **Performance Highlights**: Diffusion 모델은 신경영상 데이터의 복잡성에도 불구하고 높은 제너레이션 품질과 안정성을 제공합니다. 이는 작은 샘플 사이즈 및 노이즈 데이터 문제를 극복할 수 있게 도와주며, 특히 데이터 향상 및 신경 질환 진단에서 유용합니다. 이 논문은 diffusion 모델의 전문 응용에 대한 포괄적 리뷰를 제공하고 있으며, 향후 연구 방향과 해결해야 할 과제도 논의하고 있습니다.



### Unsupervised Learning for Feature Extraction and Temporal Alignment of 3D+t Point Clouds of Zebrafish Embryos (https://arxiv.org/abs/2502.06543)
- **What's New**: 이번 연구에서는 제브라피시 배아의 3D+t 포인트 클라우드에서 설명적인 특징을 추출하고, 이를 통해 서로 다른 발달 단계의 시간을 정렬하는 비지도 학습 기반의 새로운 방법을 제시합니다. 기존 방법들은 수동으로 식별된 랜드마크를 기반으로 하거나 이미지 도메인에서 작동하여 자동 및 비지도 방식으로 생물 표본의 시공간 정렬을 획득하는 데 제한적이었습니다. 새로운 방법은 차별화된 특징 벡터를 학습하여 시차를 회복하는 데 성공하며, 주관적 편향 없이 자동화된 정확한 정렬을 제공합니다.

- **Technical Details**: 제안된 방법은 FoldingNet을 기반으로 3D 포인트 클라우드의 특징 추출에 여러 수정을 추가하였습니다. 자동 인코더는 포인트 클라우드에서 묘사적 특징을 추출하고, 회귀 네트워크를 사용하여 서로 다른 배아를 시간적으로 정렬하는 데 활용됩니다. Modified Chamfer Distance (MCD)를 손실 함수로 도입하여 포인트 간의 거리 대신 포인트와 지역 간의 거리를 고려하여 정렬의 정확성을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 평균 3.83분의 일치 오차로 정렬 정확도가 높아졌습니다. 또한, 비지도 방식으로 진행되기 때문에 수동 레이블링이 필요하지 않아 편리하고 확장성이 뛰어납니다. 최종 결과는 서로 다른 3D+t 포인트 클라우드의 시간 프레임이 쌍으로 정렬된 형태로 제공되며, 시각적으로 복원된 3D 포인트 클라우드와 Low-dimensional representation을 통해 효과성이 입증되었습니다.



### CustomVideoX: 3D Reference Attention Driven Dynamic Adaptation for Zero-Shot Customized Video Diffusion Transformers (https://arxiv.org/abs/2502.06527)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문에서는 CustomVideoX라는 혁신적인 프레임워크를 소개하여, 참조 이미지(reference image)로부터 개인화된 비디오(generated video)를 생성하는 문제를 해결하고자 합니다. CustomVideoX는 비디오 확산 변환기(video diffusion transformer)를 활용하여, LoRA 파라미터를 통해 참조 기능을 추출하는 방식으로 효율성과 적응성을 극대화합니다. 또한 3D Reference Attention 메커니즘을 통해 참조 이미지와 모든 비디오 프레임 간의 즉각적이고 동시적인 상호작용을 가능하게 합니다.

- **Technical Details**: 본 연구의 주요 기술적 기여는 3D Reference Attention 및 Time-Aware Reference Attention Bias(TAB) 메커니즘의 통합입니다. 3D Reference Attention은 각 비디오 프레임과 참조 이미지 기능 간의 상호작용을 직접적으로 연계하여, 효율성을 향상시킵니다. Time-Aware Attention Bias는 확산 모델의 노이즈 제거 과정에서 참조 이미지 기능의 영향을 동적으로 조절하여 최종 비주얼 품질을 개선합니다.

- **Performance Highlights**: CustomVideoX는 기존 방법들에 비해 비디오 일관성과 품질에서 현저한 성능 향상을 기록합니다. 연구자들은 50개 이상의 객체와 100개의 프롬프트로 구성된 새로운 벤치마크인 VideoBench를 설정하여, 비디오 생성 성능에 대한 포괄적인 평가를 수행했습니다. 실험 결과, CustomVideoX는 제안된 벤치마크에서 기존의 방법들을 초월하는 최첨단 성능을 달성했습니다.



### Learning Clustering-based Prototypes for Compositional Zero-shot Learning (https://arxiv.org/abs/2502.06501)
Comments:
          Accepted to ICLR 2025; Project page: this https URL

- **What's New**: 이 논문은 Compositional Zero-Shot Learning (CZSL)에서의 새로운 접근법인 ClusPro를 제안합니다. ClusPro는 시각적 표현에서 속성(attribute)과 객체(object)의 개념적 경계를 정의하며, 이를 통해 다양한 프로토타입(prototypes)을 자동으로 발견하고 동적으로 업데이트합니다. 기존의 CZSL 솔루션들이 oversimplified data assumptions에 의존했던 반면, ClusPro는 클러스터링 기법을 사용하여 좀 더 현실적이고 포괄적인 데이터 분포 모델링을 수행합니다.

- **Technical Details**: ClusPro는 시각적 표현의 임베딩 공간에서 내부 속성 기반 클러스터링을 수행하여 프로토타입을 발견합니다. 이를 통해 속성 및 객체 임베딩 공간을 새롭게 구조화하고, intra-primitive separation과 inter-primitive decorrelation을 보장합니다. 특히, 프로토타입 기반 대조 학습(prototype-based contrastive learning)과 비상관 학습(prototype-based decorrelation learning)을 통해 독립적인 임베딩 공간을 학습합니다. 또한, ClusPro는 추가적인 학습 가능한 파라미터가 필요 없는 비모수(non-parametric) 방식으로 효율적인 프로토타입 클러스터링을 수행합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서의 실험 결과, ClusPro는 기존의 여러 선도적인 CZSL 솔루션들보다 우수한 성능을 보였습니다. Closed-world 설정에서는 UT-Zappos에서 +11.8%와 C-GQA에서 +20.2%의 AUC 향상을 달성했습니다. Open-world 설정에서도 UT-Zappos에서 +19.7%와 C-GQA에서 +11.1%의 강력한 개선을 기록했습니다.



### Decision Boundary Optimization-Informed Domain Adaptation (https://arxiv.org/abs/2502.06498)
- **What's New**: 본 논문에서는 기존의 Maximum Mean Discrepancy (MMD)를 기반으로 하는 도메인 적응(Domain Adaptation) 방법의 한계를 극복하기 위한 새로운 방안을 제시합니다. 이 방법은 Decision Boundary optimization-informed MMD (DB-MMD)라는 강화된 MMD 측정법으로, 도메인 간의 데이터 분포 정렬뿐만 아니라 분류기 최적화도 고려합니다. DB-MMD를 여러 인기 있는 도메인 적응 방법에 통합하여 그 효과성을 입증하였습니다.

- **Technical Details**: 기존의 도메인 적응 기법들은 보통 분포 정렬에 초점을 맞추어, 결정 경계를 최적화하는 것에는 소홀했습니다. 그러나 DB-MMD는 MMD를 개조하여 결정 경계까지 고려할 수 있는 능력을 부여하였습니다. 이를 통해 데이터 분포의 정렬과 도메인 간 분류기를 동시에 최적화하여 이론적으로 보장된 도메인 적응을 구현할 수 있습니다.

- **Performance Highlights**: 8개의 표준 도메인 적응 데이터 세트를 사용한 실험 결과는 DB-MMD를 적용한 도메인 적응 방법들이 기존의 MMD 기반 모델보다 최대 9.5의 마진으로 성능이 향상된 것을 보여줍니다. 이는 DB-MMD가 도메인 적응의 상한 오차 경계 감소에 효과적임을 입증하는 결과입니다.



### Biomechanical Reconstruction with Confidence Intervals from Multiview Markerless Motion Captur (https://arxiv.org/abs/2502.06486)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구에서는 multiview markerless motion capture (MMMC) 시스템의 kinematic한 관절 각도 추정의 신뢰 구간(confidence interval)을 제공함으로써 임상 실천과 연구에 적용 가능한 새로운 방법을 제안합니다. 기존의 연구들은 일반적인 성능을 측정하는 데 집중했으나, 본 연구는 특정 개인의 kinematic 추정치를 위한 개인화된 신뢰도를 평가하는 데 중점을 두고 있습니다. 이를 통해 MMMC의 대규모 활용을 위한 신뢰 가능한 측정값을 제공할 수 있는 기반을 마련합니다.

- **Technical Details**: 이 연구에서는 전통적인 multistage 방법 대신, end-to-end로 최적화된 미분가능한 생체역학 모델을 사용하여 kinematic joint angle trajectories에 대한 확률적 표현을 제시합니다. 각 시간대의 pose 파라미터를 예측하기 위해 keypoint 감지기를 통해 수집된 데이터(p(𝜽t|𝐘t,𝐒t))를 활용하며, 이는 variational inference 과정을 통해 동기화된 multiview 비디오에 적합합니다. 주요 기술적 접근은 신뢰성이 낮은 keypoints 감지에서의 오류를 고려하여 신뢰 구간을 적합하게 보정하는 것입니다.

- **Performance Highlights**: 본 연구의 방법은 각 실험의 순간마다 개별 관절에 대한 신뢰 구간을 생성하여, 불확실성이 높은 상황 및 실험을 식별할 수 있도록 합니다. 실험 결과, 이 방법은 가상 마커의 위치에 대해 10-15 mm의 공간 오차 범위 내에서 신뢰 구간을 제공하고 있으며, 관절 각도에 대한 신뢰 구간은 일반적으로 몇 도에 불과합니다. 또한, 관절 각도의 상관 구조를 모델링하여 다양한 분포를 보다 옳게 평가할 수 있도록 합니다.



### Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution (https://arxiv.org/abs/2502.06476)
- **What's New**: 이번 논문은 이미지 품질 평가(Image Quality Assessment, IQA)에서 이미지의 스케일이 인지된 품질에 미치는 영향을 체계적으로 수량화하는 새로운 개념인 이미지 본질 스케일(Image Intrinsic Scale, IIS)을 도입합니다. 또한, IIS를 주관적으로 측정하고 예측하는 작업인 이미지 본질 스케일 평가(Image Intrinsic Scale Assessment, IISA)도 새롭게 제안했습니다. 이를 위해 엄격한 크라우드소싱 연구를 통해 전문가가 주석을 단 785개의 이미지-IIS 쌍으로 구성된 IISA-DB 데이터셋을 구축하였습니다.

- **Technical Details**: IIS는 이미지가 가장 높은 인지 품질을 나타내는 최대 스케일로 정의되며, 이를 통해 사용자는 이미지의 품질 유지와 손실 지점을 최적의 균형으로 조정할 수 있습니다. 연구 결과, WIISA(Weak-labeling for Image Intrinsic Scale Assessment)라는 전략을 통해 다운스케일링된 이미지를 사용하여 약한 레이블을 생성함으로써, 기존 IQA 방법들의 성능을 일관되게 개선할 수 있음을 보여주었습니다. WIISA는 어떤 특정 훈련 전략이나 모델 특성과 독립적으로 쉽게 통합될 수 있는 장점이 있습니다.

- **Performance Highlights**: IISA를 적용한 다양한 IQA 방법 실험에서는 WIISA 사용 시 성능이 일관적으로 개선되는 결과를 보였습니다. 특히, 기존의 NR-IQA(no-reference IQA) 방법보다 높은 민감도를 통해 미세한 품질 저하를 효과적으로 감지할 수 있었습니다. 이로써, IISA는 웹, 인쇄 및 게임과 같은 응용 프로그램에서 이미지 표시 및 저장 최적화, 더 나아가 저수준 비전 데이터셋 생성을 지원하는 데 기여할 수 있습니다.



### UniMoD: Efficient Unified Multimodal Transformers with Mixture-of-Depths (https://arxiv.org/abs/2502.06474)
- **What's New**: 이 논문에서는 Unified Multimodal Transformers에 대한 새로운 접근법인 UniMoD를 소개합니다. UniMoD는 각 작업에 대해 별도의 router를 사용하여 토큰을 선정하고 불필요한 토큰을 제거함으로써 계산 효율성을 개선합니다. 이는 특히 토큰 중복성에 영향을 미치는 다양한 작업과 레이어를 분석하여 디자인되었습니다.

- **Technical Details**: 연구에서는 attention weight 패턴을 분석하고, 레이어 중요도 및 토큰 중복성을 평가했으며, 작업 간 상호 작용을 분석했습니다. 이러한 분석을 통해 각 작업의 특성에 기반하여 토큰을 효율적으로 선택하고 불필요한 부분을 제거하는 방법을 제안합니다. UniMoD를 사용하면 Show-o와 Emu3에서 각각 15% 및 40%의 FLOPs 감소를 이룰 수 있습니다.

- **Performance Highlights**: 실험 결과, UniMoD를 적용하여 성능을 유지하거나 개선하면서 계산 및 메모리 사용을 줄일 수 있음을 확인했습니다. Show-o는 진화 기반의 생성 및 이해 작업을 수행하며, Emu3는 완전 자율 회귀 방식으로 두 작업을 처리합니다. 이는 다양한 통합 변환기 아키텍처에 적용할 수 있으며, 당사의 접근 방식은 PixArt와 같은 순수 탐색 기반 생성 모델에도 유연하게 적용됩니다.



### Group-CLIP Uncertainty Modeling for Group Re-Identification (https://arxiv.org/abs/2502.06460)
- **What's New**: 이 논문은 Group Re-Identification (Group ReID) 문제를 다루며, 비감독 카메라 간의 보행자 그룹 매칭 방법에 대한 새로운 접근법을 제안합니다. 특히 기존의 방법들이 특정 그룹 구조에 기반한 확률 모델에 의존하는 것과 달리, 논문은 멤버와 레이아웃의 변화를 수용하는 불확실한 그룹 설명을 활용하는 Group-CLIP Uncertainty Modeling (GCUM) 방법론을 소개합니다. 이를 통해 다양한 그룹 구조에 효과적으로 적응할 수 있습니다.

- **Technical Details**: 제안된 GCUM 접근법은 세 가지 모듈, 즉 Member Variant Simulation (MVS), Group Layout Adaptation (GLA), Group Relationship Construction Encoder (GRCE)로 구성됩니다. MVS 모듈은 Bernoulli 분포를 사용해 그룹 멤버의 무작위 제외를 시뮬레이션하고, GLA 모듈은 정체성 특정 토큰을 사용해 불확실한 그룹 설명을 생성합니다. GRCE는 개별 특성을 정제하고 집계하여 그룹 식별을 강화합니다.

- **Performance Highlights**: 실험 결과, GCUM은 기존의 최신 Group ReID 방법들과 비교해 현저하게 우수한 성능을 보였습니다. 이 모델은 텍스트 설명의 불확실성을 활용하여 실제 환경의 다양한 변화를 효과적으로 처리할 수 있는 가능성을 보여줍니다. 논문의 기여점 중 하나는 CLIP를 Group ReID에 처음으로 적용한 사례입니다.



### SparseFocus: Learning-based One-shot Autofocus for Microscopy with Sparse Conten (https://arxiv.org/abs/2502.06452)
- **What's New**: 이번 논문에서는 Microscopic Imaging에서 Autofocus의 중요성을 강조하며, 전통적인 기계식 초점 조정 방식 대신 SparseFocus라는 콘텐츠 중요도 기반의 새로운 접근 방법을 제안합니다. SparseFocus는 이미지의 특정 영역의 중요성을 평가하고 선택된 중요한 영역에서부터 비틀림 거리(defocus distance)를 계산하는 혁신적인 두 단계 파이프라인을 구축합니다. 이를 통해 다양한 콘텐츠 희소성 수준에서 효과적으로 Autofocus 문제를 해결할 수 있는 가능성을 열었습니다.

- **Technical Details**: SparseFocus는 두 가지 주요 단계로 구성됩니다: 첫 번째 단계에서는 이미지 내의 각 영역의 중요성 점수를 할당하고, 두 번째 단계에서는 선택된 중요 영역으로부터 비틀림 거리를 회귀 분석하는 네트워크를 활용합니다. 특히, Dense와 Sparse한 콘텐츠 모두를 고려하여 수집한 대규모 라벨링된 이미지 데이터셋을 사용해 알고리즘을 훈련하고 검증하였습니다. 이 플랫폼에서는 병리 조직과 세포에서 데이터 샘플을 자동으로 수집하여 평가의 다양성을 높였습니다.

- **Performance Highlights**: 실험 결과 SparseFocus는 기존 방법들을 크게 초월하는 성능을 보여주었으며, 특히 Sparse한 경우에도 우수한 초점 조정을 할 수 있음을 증명했습니다. 우리의 방법은 단지 이미지의 특정 한 영역만으로도 유의미한 결과를 도출할 수 있어, 실세계에서의 고속 자동 초점 조정에 큰 가능성을 지니고 있습니다. 이 시스템은 Whole Slide Imaging(WSI) 시스템에 통합되어 실제 응용 분야에서도 유망한 초점 조정 성능을 발휘하고 있습니다.



### Benchmarking Vision-Language Models on Optical Character Recognition in Dynamic Video Environments (https://arxiv.org/abs/2502.06445)
Comments:
          Code and dataset: this https URL

- **What's New**: 이 논문은 동적 비디오 환경에서 Optical Character Recognition (OCR) 작업에 대한 Vision-Language Models (VLMs)을 평가하기 위한 오픈 소스 벤치마크를 소개합니다. 1,477개의 수작업으로 주석이 달린 프레임으로 구성된 새로운 데이터셋을 제안하며, 이 데이터셋은 코드 편집기, 뉴스 방송, 유튜브 비디오 및 광고 등 다양한 분야를 포함합니다. 연구는 세 가지 첨단 VLM인 Claude-3, Gemini-1.5, GPT-4o를 기존의 OCR 시스템과 비교 평가하고 그 결과를 공유합니다.

- **Technical Details**: Optical Character Recognition (OCR)은 시각적 콘텐츠에서 텍스트 정보를 추출하는 인공지능의 기반 기술로, Vision-Language Models (VLMs)의 출현으로 기존 OCR 방법을 능가할 수 있는 가능성이 부각되고 있습니다. 이 연구는 다양한 도메인에서 수작업 주석이 달린 1,477개의 비디오 프레임으로 구성된 데이터셋을 이용하여 VLM과 전통적인 Computer Vision 기반 OCR 기술의 성능을 평가합니다. 평가 지표로는 Word Error Rate (WER), Character Error Rate (CER), Accuracy 등이 사용됩니다.

- **Performance Highlights**: 결과적으로 VLM들은 전통적인 OCR 모델에 비해 많은 시나리오에서 성능을 초과하는 능력을 보여주고 있지만, 일부 도전 과제가 여전히 존재합니다. 이러한 문제에는 환각 현상(hallucinations), 콘텐츠 보안 정책(content security policies), 가려진 또는 스타일화된 텍스트에 대한 민감성이 포함됩니다. 논문에서 제시된 벤치마크와 데이터셋은 연구자들이 새로운 모델을 쉽게 평가할 수 있도록 MIT 라이센스 하에 공개되었습니다.



### Rethinking Large-scale Dataset Compression: Shifting Focus From Labels to Images (https://arxiv.org/abs/2502.06434)
Comments:
          Work In Progress

- **What's New**: 이 논문에서는 데이터셋 압축을 위한 두 가지 주요 기술인 dataset distillation과 dataset pruning을 비교하는 새로운 벤치마크를 제시합니다. 기존의 방법론들은 서로 다르게 평가되었기 때문에 공정한 비교가 어려웠는데, 이 새로운 벤치마크는 이를 보완하며 연구의 재현 가능성을 높입니다. 특히, 주목할 만한 발견은 대규모 데이터셋에서 무작위로 선택된 서브셋이 과도한 soft labels의 의존 없이도 뛰어난 성능을 낼 수 있다는 것입니다.

- **Technical Details**: 제안된 Prune, Combine, and Augment (PCA) 프레임워크는 이미지 데이터에 집중하고, 평가 시에는 오직 hard labels만을 사용함으로써 기존 방법보다 우수한 성능을 달성합니다. 이 과정에서는 NLL(Negative Log-Likelihood)과 entropy를 활용하여 모델의 예측 확률을 정량화하며, Kullback–Leibler divergence를 통해 모델의 분포(q_theta)를 실제 데이터의 분포(p)로 최소화하는 과정을 강조합니다.

- **Performance Highlights**: PCA 프레임워크는 최신 성과를 달성했으며, 데이터셋을 압축하기 위한 보다 균형 잡힌 접근 방식을 제공합니다. 본 연구의 결과는 이미지 데이터의 고유한 가치를 재조명하며, 데이터 압축 연구의 접근성을 높이는 데 기여할 것으로 기대됩니다. 또한, 연구자가 사용할 수 있는 코드는 해당 링크를 통해 공개되어 있습니다.



### Prompt-SID: Learning Structural Representation Prompt via Latent Diffusion for Single-Image Denoising (https://arxiv.org/abs/2502.06432)
- **What's New**: 이 연구에서는 Prompt-SID라는 프롬프트-학습 기반의 단일 이미지 노이즈 제거 프레임워크를 소개합니다. 기존의 자가 감독 및 비지도 방식이 가지고 있는 문제점인 픽셀 정보 손실과 구조적 디테일 손상을 해결하는 것을 목표로 합니다. 이 프레임워크는 다운샘플링된 이미지 쌍을 사용하여 자가 감독 방식으로 학습하며, 구조 인코딩을 통해 원본 스케일의 이미지 정보를 캡처합니다.

- **Technical Details**: Prompt-SID는 잠재적 확산 과정에 기반한 구조 표현 생성 모델을 제안하였습니다. 이 모델은 다운샘플링된 이미지에서 손상되지 않은 구조 표현을 복구하기 위한 조건 정보로 손상된 구조 표현을 사용합니다. 또한, 트랜스포머 기반의 노이저에서 구조 표현을 프롬프트로 통합하는 구조적 주의 모듈(SAM)을 설계하였습니다.

- **Performance Highlights**: Prompt-SID는 합성, 실제, 형광 이미징 데이터 세트에서 뛰어난 효과를 나타내며, 상대적으로 적은 매개변수 수로도 높은 성능을 자랑합니다. 기존의 SOTA(최신 기술) 접근 방식을 초월하는 성능을 보여주며, 실제 스케일 이미지를 처리할 수 있는 일반화 능력을 유지합니다.



### FCVSR: A Frequency-aware Method for Compressed Video Super-Resolution (https://arxiv.org/abs/2502.06431)
- **What's New**: 본 논문은 새로운 딥 러닝 기반의 압축 비디오 초해상도 모델인 FCVSR(Frequency-aware Compressed Video Super-Resolution)을 제안합니다. 이 모델은 모션 가이드를 기반으로 한 적응형 정렬(MGAA) 네트워크와 다중 주파수 특징 정제(MFFR) 모듈로 구성되며, 주파수 인지를 통한 대조 손실을 적용하여 더 세밀한 공간적 디테일을 복구하기 위한 훈련 방식을 개발하였습니다.

- **Technical Details**: FCVSR 모델은 여러 프레임 간의 모션 오프셋을 주파수 영역에서 추정하여 특징 정렬을 가능하게 하는 MGAA 모듈을 결합합니다. 더불어, MFFR 모듈은 분해-강화-집계 전략을 활용하여 고해상도 비디오 내에서 고주파 세부 사항을 복구합니다. 또한, 새로운 주파수 인지 대조 손실(FC loss)을 통해 고주파 세부 사항의 복구를 지원합니다.

- **Performance Highlights**: 제안된 FCVSR 모델은 세 가지 공개 압축 비디오 초해상도 데이터셋에서 기존의 다섯 가지 압축 VSR 모델에 비해 정량적 및 정성적 평가에서 개선된 성능을 보여주었습니다. 특히 PSNR(Peak Signal-to-Noise Ratio) 지표에서 최대 0.14dB의 향상을 기록했으며, 상대적으로 낮은 계산 복잡도로 실제 응용에서 우수한 성능을 발휘하는 것을 입증했습니다.



### CoS: Chain-of-Shot Prompting for Long Video Understanding (https://arxiv.org/abs/2502.06428)
Comments:
          A training-free test-time optimisation approach for long video understanding

- **What's New**: 이 논문에서는 Chain-of-Shot prompting (CoS)라는 새로운 전략을 제안합니다. 이 방법은 long video understanding에서의 시각적 정보 선택을 최적화하여 모델의 성능을 극대화하는 것을 목표로 합니다. CoS는 두 가지 주요 메커니즘인 Binary Video Summary와 Video Co-Reasoning으로 구성되어 있습니다.

- **Technical Details**: Binary Video Summary는 긴 비디오 내에서 task-relevant shots를 식별하는 이진 코딩 기법을 사용합니다. Video Co-Reasoning은 이 이진 코드를 활용하여 task-relevant positive shots와 irrelevant negative shots을 연결해 모델의 초점을 맞추는데 필요한 정보를 선택적으로 필터링합니다. 이 방법은 테스트 시간에 최적화를 수행하여 긴 비디오의 이해를 돕습니다.

- **Performance Highlights**: 실험을 통해 CoS는 다양한 데이터셋과 세 가지 기준 모델을 이용하여 우수한 성과를 보였습니다. 이 연구는 기존의 시각적 정보 선택 방법의 한계를 극복하여, 적절한 정보를 더 효과적으로 선택하고 불필요한 정보를 최소화하는 데 기여합니다. CoS의 개발은 MLLMs의 긴 비디오 처리 능력을 한층 강화할 것으로 기대됩니다.



### Hybrid State-Space and GRU-based Graph Tokenization Mamba for Hyperspectral Image Classification (https://arxiv.org/abs/2502.06427)
- **What's New**: 이번 연구에서는 GraphMamba라는 새로운 하이브리드 모델을 제안하여, hyperspectral image (HSI) 분류의 복잡한 문제를 해결하고자 합니다. 이 모델은 spectral-spatial token generation, graph-based token prioritization 및 cross-attention 메커니즘을 통합하여 복잡한 공간-스펙트럼 관계를 모델링합니다. 기존의 모델들에 비해 계산 효율성과 확장성을 유지하면서도 성능 향상을 이룰 수 있는 방법을 제공합니다.

- **Technical Details**: GraphMamba 모델은 dual convolutional framework를 활용하여 스펙트럴 및 공간 특성을 효율적으로 추출하고 토큰화합니다. 스펙트럴 토큰화는 1×1 convolution을 사용하여 스펙트럴 변화를 명확히 분리하고, 공간 토큰화는 3×3 convolution을 사용하여 지역적 공간 맥락을 효과적으로 포착합니다. 모델은 학습 가능한 스코어링 메커니즘을 통해 동적으로 토큰의 우선 순위를 정하며, cross-attention 레이어를 통해 스펙트럴 및 공간 토큰 간의 상호작용을 효과적으로 촉진합니다.

- **Performance Highlights**: GraphMamba는 실험을 통해 기존의 최첨단 모델들을 초월하는 성능을 보여주었습니다. 이 모델은 복잡한 HSI 분류 작업에 대해 스케일이 가능한 강력한 솔루션을 제공하며, 다양한 HSI 데이터 세트에서 우수한 결과를 달성하였습니다. 연구 결과는 GraphMamba가 공간-스펙트럼 관계를 효과적으로 캡처하는 동시에 계산 효율성을 유지할 수 있음을 입증합니다.



### Robust Watermarks Leak: Channel-Aware Feature Extraction Enables Adversarial Watermark Manipulation (https://arxiv.org/abs/2502.06418)
- **What's New**: 이 연구는 기존의 강인한 워터마크 방식의 근본적인 취약점을 드러내어, 이러한 방식이 정보 유출을 초래한다고 제안합니다. 특히, 워터마크의 강인함을 높이는 과정에서 정보 중복성이 증가하여 공격자가 이용할 수 있는 여지를 만듭니다. 제안된 DAPAO 공격 프레임워크는 이를 활용하여 워터마크 패턴을 효과적으로 추출하고, 단일 워터마크 이미지로 위조 및 탐지 회피를 수행할 수 있음을 보여줍니다.

- **Technical Details**: 워터마킹 기술은 이미지에 워터마크를 삽입하고, 이를 추출 및 검증하는 과정을 포함합니다. 기존 방법에서는 비선형 학습 알고리즘을 사용하여 Encoder와 Decoder를 구성하고, 방어기법을 통해 왜곡에 대한 강인성을 향상시킵니다. DAPAO 공격은 다중 채널 특징 학습을 통해 워터마크 정보를 추출하며, 영상의 원형 정보와 적절히 정렬된 학습 가능한 변수를 최적화합니다.

- **Performance Highlights**: 실험 결과, DAPAO 공격 방법은 기존의 최첨단 연구보다 60% 향상된 탐지 회피 성공률과 51% 증가한 위조 정확도를 달성했습니다. 이는 시각적 완전성을 유지하면서도 공격을 수행할 수 있음을 의미합니다. 본 연구는 강인한 워터마크의 저항성과 보안성 간의 패러독스를 이해하는 중요한 통찰을 제공합니다.



### TANGLED: Generating 3D Hair Strands from Images with Arbitrary Styles and Viewpoints (https://arxiv.org/abs/2502.06392)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 다양한 스타일과 관점을 가진 이미지 입력에 적응할 수 있는 3D 헤어 스트랜드 생성 방법인 TANGLED를 소개합니다. 이 접근 방식은 MultiHair Dataset을 활용하여 457개의 다양한 헤어스타일을 74개 속성과 함께 주석 처리하여 모델의 일반화를 향상시킵니다. 또한, 다중 뷰 라인 아트를 기반으로 한 확산 프레임워크를 제안하여 상대적으로 복잡한 헤어스타일의 생성 과정에서 발생하는 잡음을 필터링합니다.

- **Technical Details**: TANGLED의 핵심은 세 가지 단계의 파이프라인을 기반으로 하고 있습니다. 첫 번째 단계에서는 MultiHair Dataset을 통해 다양한 헤어스타일을 제공하며, 두 번째 단계에서는 다중 뷰 라인 아트를 조건으로 하는 확산 프레임워크를 도입합니다. 마지막으로, 파라메트릭 후처리 모듈을 통해 브레이드와 같은 복잡한 구조의 일관성을 유지하고 기하학적 왜곡을 줄입니다.

- **Performance Highlights**: 실험 결과, 사용자 연구에서는 realism과 다양성 측면에서 TANGLED의 결과가 텍스트 기반 모델보다 84.3% 선호된다고 보고되었습니다. TANGLED는 문화적으로 포괄적인 디지털 아바타를 지원하며, 스케치 기반 3D 스트랜드 편집과 같은 혁신적인 애플리케이션으로의 확장을 가능하게 합니다.



### When Data Manipulation Meets Attack Goals: An In-depth Survey of Attacks for VLMs (https://arxiv.org/abs/2502.06390)
- **What's New**: 이번 논문은 비전-언어 모델(VLMs)의 공격 전략에 대한 심층 조사를 제공합니다. 연구진은 VLM 공격을 jailbreak, camouflage, exploitation으로 분류하고, 다양한 데이터 조작 방법론을 상세히 설명합니다. 뿐만 아니라 VLM의 취약성을 완화하기 위한 방어 메커니즘도 제시하고 있습니다.

- **Technical Details**: VLMs는 텍스트와 비주얼 정보를 통합하여 다양한 애플리케이션에서 성능을 향상시키는 데 기여하고 있습니다. 특히, Recent advancements in Large Vision-Language Models (LVLMs)에서는 LLMs와의 통합이 더 확장된 지식 기반을 제공하도록 하여 VLMs의 강인함을 높여줍니다. 그러나 여전히 LVLMs는 새로운 취약성을 악화시키는 문제를 앓고 있습니다.

- **Performance Highlights**: 최근 몇 년간 VLM 공격에 대한 연구는 풍부한 다양성을 보여주고 있으며, 이로 인해 VLMs의 공격 방법론이 진화하고 있습니다. 이 논문은 이러한 공격의 독특한 특성을 조사하고, 공격 방법들이 어떻게 분류될 수 있는지에 대한 통찰력을 제공하고, 기존의 방어 메커니즘과의 관계를 분석합니다.



### FOCUS - Multi-View Foot Reconstruction From Synthetically Trained Dense Correspondences (https://arxiv.org/abs/2502.06367)
Comments:
          13 pages, 11 figures

- **What's New**: 이번 논문에서는 FOCUS라는 새로운 방법론을 제안하여 다수의 RGB 이미지에서 인간 발의 정밀한 3D 재구성을 목표로 하고 있습니다. 본 연구는 SynFoot2라는 방대한 합성 데이터셋을 활용하여 픽셀 수준에서의 밀집 대응 관계를 예측하고, 이를 통해 FIND 모델을 기반으로 3D 표면을 재구성하는 두 가지 방법을 소개합니다. FOCUS는 최첨단의 3D 재구성 품질을 제공하며, 적은 수의 이미지로도 높은 정확도를 달성할 수 있습니다.

- **Technical Details**: FOCUS는 SynFoot2라는 새로운 합성 데이터셋을 통해 학습하며, 이는 100,000개의 포토리얼리스틱 발 이미지를 포함하고 있습니다. 이 방법론은 Template Object Coordinates (TOCs)를 도입하여 픽셀 수준에서의 대응관계를 예측하며, 두 가지 접근법을 통해 재구성을 수행합니다. 첫 번째는 Structure-from-Motion (SfM)에서 영감을 받은 방식이고, 두 번째는 FIND 모델을 활용한 최적화 기반 방법입니다.

- **Performance Highlights**: FOCUS는 적은 수의 뷰에서도 정밀한 재구성을 가능하게 하며, 많은 뷰가 주어질 경우에도 최첨단 재구성 품질을 유지합니다. 본 연구는 GPU를 필요로 하지 않으며, COLMAP에 비해 적은 뷰 수로도 작동하면서 빠른 속도를 자랑합니다. 또한, 합성 데이터셋과 코드를 연구 커뮤니티에 공개하여 후속 연구에 기여할 예정입니다.



### Guidance-base Diffusion Models for Improving Photoacoustic Image Quality (https://arxiv.org/abs/2502.06354)
- **What's New**: 본 연구에서는 Photoacoustic(PA) 이미징의 영상 품질을 향상시키기 위해 새로운 방법을 제안합니다. 기존의 낮은 품질의 단일 샷(image) 이미지를 고품질 이미지로 변환하기 위해 확산 모델(diffusion model)을 사용하고, 이미징 조건 정보를 통해 이를 안내하는 방법을 도입합니다. 이 접근법은 낮은 품질 단일 샷 이미지에서 추정된 노이즈를 비교하여 더 나은 이미지 품질을 만들어냅니다.

- **Technical Details**: PA 이미징은 단일 샷 이미지의 품질이 낮은 문제를 갖고 있으며, 기존의 평균화 기법이 필요하나 시간 소모가 크고 고품질 이미지를 생성하기 위한 심도 있는 기술적 접근이 필요합니다. 제안된 방법에서는 확산 모델을 활용해 여러 단일 샷 이미지를 기반으로 고품질 이미지를 생성하고, 노이즈 추정을 정교화하여 이미징 과정의 특성을 반영합니다. 이를 통해 각 단일 샷 이미지의 신뢰성을 고려하여 노이즈를 추정하고 이를 평균화하는 독특한 접근을 소개합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법이 전통적인 기술들에 비해 우수한 효과를 보임을 확인하였습니다. 실제 PA 이미지를 이용한 테스트를 통해 높은 품질의 이미지를 생성하는데 유리한 결과를 도출했습니다. 이 접근법은 PA 이미징의 품질을 획기적으로 향상시킬 수 있는 가능성을 보여줍니다.



### LANTERN++: Enhanced Relaxed Speculative Decoding with Static Tree Drafting for Visual Auto-regressive Models (https://arxiv.org/abs/2502.06352)
Comments:
          15 pages, 5 figures, short paper (5 pages exclude reference and appendix)

- **What's New**: 이번 연구에서는 시각적 오토리그레시브(AR) 모델에서의 스펙큘레이티브 디코딩(speculative decoding) 기술 한계를 해결하기 위해 LANTERN++라는 새로운 프레임워크를 도입했습니다. 기존의 동적 트리 드래프팅(dynamic tree drafting) 방법은 토큰의 선택 모호성(token selection ambiguity) 문제를 충분히 완화하지 못했으며, 이로 인해 효율성이 떨어지는 문제를 보여주었습니다. LANTERN++는 정적 트리 드래프팅(static tree drafting)을 접목하여 깊은 드래프트 시퀀스를 생성하고, 낮은 신뢰도 예측에서도 선택이 가능하게 합니다.

- **Technical Details**: LANERN++는 스태틱 트리 드래프팅과 완화된 수용 조건을 통합하여, 토큰 선택 시 낮은 신뢰도 예측에 독립적으로 드래프트를 선택할 수 있도록 설계되었습니다. 이를 통해 LANTERN++는 이미지 품질을 유지하면서도 디코딩 효율성을 개선할 수 있게 되었습니다. 연구에서는 주요 모델들에 대해 LANTERN++의 성능을 테스트하였고, 동적인 상황에서도 깊은 드래프트 시퀀스를 생성할 수 있도록 했습니다.

- **Performance Highlights**: 광범위한 실험 결과, LANTERN++는 기존의 AR 디코딩보다 최고 2.56배의 속도 향상을 달성하면서도 높은 이미지 품질을 유지할 수 있음을 보였습니다. 또한, LANTERN++는 전반적인 디코딩 효율성을 크게 향상시켜, 시각적 AR 모델링에서 효율적인 접근 방식으로 주목받고 있습니다.



### Facial Analysis Systems and Down Syndrom (https://arxiv.org/abs/2502.06341)
- **What's New**: 최근 몇 년 동안 얼굴 분석 기술(Facial Analysis Systems, FASs)에 대한 윤리적, 사회적, 법적 문제들이 활발히 논의되고 있습니다. 특히, 이러한 기술이 소외된 집단에 대한 편향과 차별을 지속할 수 있다는 비판이 대두되고 있습니다. 본 논문에서는 다운 증후군(Down syndrome) 인물의 얼굴과 관련된 데이터 세트를 활용해 FAS의 한계를 보고하고, 이 분야에서 간과된 취약 집단에 대한 새로운 증거를 제시합니다.

- **Technical Details**: 연구는 다운 증후군이 있는 인물과 없는 인물의 얼굴 이미지로 구성된 데이터 세트를 만들고, 두 가지 상업적 도구를 사용하여 성별 인식, 나이 예측 및 얼굴 레이블링 과제를 수행했습니다. 실험군은 다운 증후군이 있는 200명의 얼굴 이미지로 구성되었고, 대조군은 다운 증후군이 없는 200명의 얼굴 이미지로 구성되었습니다. 연구의 중심 질문은 다운 증후군이 있는 개인의 얼굴 인식을 FAS가 어떻게 수행하는지를 조사하는 것이었습니다.

- **Performance Highlights**: 결과적으로, 실험군의 얼굴 인식 예측 정확도가 전반적으로 낮았고, 다운 증후군이 있는 남성의 성별 인식에서 높은 오류율이 나타났습니다. 또한 성인이 다운 증후군을 가진 경우 아동으로 잘못 라벨링되는 빈도가 더 높았으며, 아름다움과 관련된 사회적 편견이 두 그룹 모두에서 나타났습니다. 이러한 발견은 다운 증후군을 가진 인물에 대한 얼굴 인식 기술의 구조적 한계를 강조합니다.



### Zero-shot Depth Completion via Test-time Alignment with Affine-invariant Depth Prior (https://arxiv.org/abs/2502.06338)
Comments:
          AAAI 2025, Project page: this https URL

- **What's New**: 이번 논문에서는 sparse depth measurements로부터 dense depth map을 예측하는 zero-shot depth completion 기법을 제안합니다. 기존의 방법들이 주로 in-domain 데이터에 적합하게 학습된 것과 달리, 제안하는 방법은 affine-invariant depth diffusion 모델과 test-time alignment를 통해 일반화 성능을 높입니다. 이 기법은 metric-scale sparse measurements와의 정렬을 통해서 적용되며, 고품질의 depth estimation을 가능하게 합니다.

- **Technical Details**: 제안된 zero-shot depth completion 방법은 사전 학습된 monocular depth diffusion 모델을 사용하여 depth prior로 활용하며, 이를 통해 다양한 도메인 데이터셋에 대한 일반화 능력을 보여줍니다. 이 과정에서 optimization loop를 통해 sparse measurements를 strict constraint로 설정하고, outlier filtering 기법을 도입하여 신뢰할 수 있는 측정을 보장합니다. 또한, affine-invariant depth prior와 sparse metric measurements를 정렬하여 잘 구조화된 depth completion을 수행합니다.

- **Performance Highlights**: 우리의 접근 방식은 이전 최첨단 방법에 비해 최대 21%의 성능 향상을 달성하며, 다양한 도메인 데이터셋에서 높은 일반화 능력을 나타냅니다. 실내 및 실외 환경을 포함한 여러 데이터셋에서 경쟁력 있는 결과를 보여주며, scene structure의 세밀한 이해를 향상시키는 데 기여합니다. 이러한 결과는 광범위한 훈련 데이터에 의존하지 않고도 domain-generalizable depth completion을 달성할 수 있다는 것을 입증합니다.



### Accelerating Outlier-robust Rotation Estimation by Stereographic Projection (https://arxiv.org/abs/2502.06337)
- **What's New**: 이번 논문에서는 회전 추정 문제를 해결하기 위한 효율적이고 견고한 방법을 제안합니다. 기존의 알고리즘들은 계산 시간이 길거나 지역 최적에 빠질 위험이 크았지만, 본 방법은 기하학적 제약을 통해 회전 축을 탐색하고, 스테레오 그래픽 투영과 공간 투표 기법을 사용하여 회전 축과 각도를 식별합니다. 이 방식은 더 빠르게 최적 회전 추정을 얻을 수 있으며, 동시에 여러 번의 회전도 추정 가능하다는 점에서 혁신적입니다.

- **Technical Details**: 제안된 방법은 회전 축과 각도를 분리하여 다루는 접근 방식을 사용합니다. 회전 축을 찾기 위해 단위 구의 원들의 최대 교차점을 찾는 문제로 변환하고, 스테레오 그래픽 투영을 통해 이러한 원들을 2차원 평면으로 매핑합니다. 이로 인해 불필요한 공간에서의 계산을 피할 수 있어 더 높은 효율성을 유지할 수 있습니다. 또한, 공간 투표 전략을 도입하여 2차원 평면에서 회전 축의 최적 지점을 동시에 찾아낼 수 있습니다.

- **Performance Highlights**: 본 방법은 GPU의 도움을 받아 대규모 데이터(10^6 포인트) 및 심각한 손상(90% 이상 아웃라이어)이 있는 회전 추정 문제를 0.07초 이내에 해결했습니다. 평균 각도 오차는 0.01도에 불과하여, 정확도와 효율성 면에서 기존 방법들보다 우수한 성과를 보였습니다. 이 결과는 다양한 컴퓨터 비전 및 로봇 작업에서의 활용 가능성을 더욱 높이며, 특히 자율주행과 같은 분야에서의 안전성을 향상시킬 수 있습니다.



### DefTransNet: A Transformer-based Method for Non-Rigid Point Cloud Registration in the Simulation of Soft Tissue Deformation (https://arxiv.org/abs/2502.06336)
- **What's New**: 이 논문에서는 비탈성 포인트 클라우드 등록을 위한 새로운 Transformer 기반 아키텍처인 DefTransNet을 소개합니다. DefTransNet은 비정형 구조의 변형을 정확하고 견고하게 대응하기 위한 해결책으로 설계되었습니다. 특히, 소스 및 목표 포인트 클라우드를 입력으로 받아 변위 벡터 필드를 출력하며, 여러 데이터 세트를 통해 일반화 능력을 시험하였습니다.

- **Technical Details**: DefTransNet은 고유한 피처 설명자와 변위 벡터 필드를 학습하는 두 가지 주요 단계를 포함합니다. 이 모델은 변환에 대한 견고성을 강화하는 학습 가능한 변환 매트릭스를 통합하며, 전역 및 지역 기하학 정보 또한 고려하여 피처 집계를 진행합니다. Transformers의 자기 주의 메커니즘을 활용하여 포인트 간의 장기 의존성을 포착함으로써 전체적인 정보 흐름을 개선합니다.

- **Performance Highlights**: 실험 결과, DefTransNet은 다양한 난이도의 상황에서 현재의 최첨단 등록 네트워크보다 뛰어난 성능을 보였습니다. 연구진은 ModelNet, SynBench, 4DMatch, DeformedTissue를 포함한 네 가지 데이터 세트를 사용하여 이 방법의 효과성을 검증하였으며, 모든 데이터 세트에서 균일하게 높은 성능을 유지하는 것으로 나타났습니다.



### UniDemoiré: Towards Universal Image Demoiréing with Data Generation and Synthesis (https://arxiv.org/abs/2502.06324)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 범용 이미지 데모이레 솔루션인 UniDemoiré를 제안합니다. 이 모델은 다양한 형태의 모이레 패턴을 자동으로 생성하여 데모이레링 모델의 일반화 능력을 향상시킵니다. 특히, 새로운 데이터 생성 및 합성 방법을 통해 대량의 고품질 모이레 이미지를 생성할 수 있습니다.

- **Technical Details**: UniDemoiré는 대규모 모이레 패턴 데이터셋을 활용하여 다양한 패턴을 생성합니다. 기본적으로, 이미지 내용과는 무관한 순수 모이레 패턴을 캡쳐하여 기존의 데이터 수집 방법의 한계를 극복합니다. 또한, 생성된 모이레 패턴과 깨끗한 자연 이미지를 혼합하여 현실감 있는 모이레 이미지를 합성하는 방법도 제안합니다.

- **Performance Highlights**: 제안된 UniDemoiré 모델은 제로샷(Zero-shot) 이미지 데모이레링과 교차 도메인 평가에서 우수한 성능을 보여줍니다. 본 연구는 일반화 능력이 뛰어난 모이레 제거 모델을 위한 충분한 양의데이터를 제공하였으며, 이는 모이레 패턴의 다양성을 크게 향상시킵니다.



### Cell Nuclei Detection and Classification in Whole Slide Images with Transformers (https://arxiv.org/abs/2502.06307)
- **What's New**: 이 논문에서는 전통적인 세포 분할(segmentation) 접근 방식에서 세포 탐지(detection) 방법으로의 패러다임 전환을 제안합니다. 세포 정보를 추출하기 위해 CellNuc-DETR라는 새로운 접근법을 도입하며, 기존의 방법들과 비교하여 정확도와 속반측에서 큰 향상을 보여줍니다. 오랜 전처리 및 후처리 시간 없이, 대용량 Whole Slide Images (WSIs)에서도 효율적으로 작동하는 점이 주목할 만합니다.

- **Technical Details**: CellNuc-DETR은 Detection Transformer (DETR)를 기반으로 하는 접근법으로, 세포 탐지에 있어 AP (Average Precision) 기준의 경량 모델입니다. 이 모델은 상대적으로 오버랩되거나 겹치는 세포 핵을 효과적으로 처리하는 동시에, 전통적인 세그멘테이션에서 발생하는 고비용을 제거합니다. 하드웨어 자원 사용이 많지 않으며, 교차 데이터 세트 평가를 통해 탁월한 강건성과 일반화 능력을 검증했습니다.

- **Performance Highlights**: CellNuc-DETR은 PanNuke 데이터셋을 포함한 여러 데이터셋에서 세포 핵 탐지 및 분류에서 최신 성능을 달성했습니다. 특히, CellNuc-DETR은 HoVer-NeXt 방식보다 2배 빠르며, CellViT보다도 훨씬 높은 정확도를 자랑합니다. 이러한 결과는 디지털 병리학에서 세포 분석을 위한 고안정성의 솔루션으로서 CellNuc-DETR의 가능성을 강조하며, 신속한 처리를 요구하는 임상 환경에서도 효과적으로 활용될 수 있음을 나타냅니다.



### Enhancing Ground-to-Aerial Image Matching for Visual Misinformation Detection Using Semantic Segmentation (https://arxiv.org/abs/2502.06288)
Comments:
          9 pages, 4 figures

- **What's New**: 최근 생성적 AI 기법의 발전으로 조작된 이미지와 비디오의 온라인 확산이 증가하면서, 인터넷에서 이용 가능한 디지털 미디어의 신뢰성에 대한 심각한 우려가 제기되고 있습니다. 이 연구는 GPS 데이터와 같은 외부 정보 없이 비지리적 태그가 달린 지상 이미지의 지오로케이션(geolocation) 문제를 다루며, 위성 이미지와의 연결을 시도하고 있습니다. 특히, 새로운 SAN-QUAD 아키텍처를 제안하여, 지상 및 위성 이미지의 의미적 분할(semantic segmentation)을 활용하여 SOTA 방법론을 확장합니다.

- **Technical Details**: SAN-QUAD 아키텍처는 지상 및 위성 이미지 모두에 대한 의미적 분할 마스크를 포함하여, 서로 다른 시점(viewpoint)에서의 일관된 특징을 식별하는 능력을 개선합니다. 이 모델은 네 개의 VGG16 스트림을 기반으로 하며, 각각의 스트림은 지상 및 위성 이미지를 입력받아 이들의 분할 마스크를 처리합니다. 데이터셋은 CVUSA 데이터셋의 일부분으로, 각 샘플은 지상 RGB 이미지, 분할 마스크, 위성 RGB 이미지, 분할 마스크의 네 가지 유형 데이터를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, CVUSA 데이터셋의 부분집합에서 SAN-QUAD 모델은 다양한 FoV 설정에서 이전 방법들에 비해 최대 9.8%의 성능 향상을 보여주었습니다. 이를 통해 저널리즘, 지구 관측 및 포렌식 분야에서 허위 정보와 그로 인한 위협에 대응할 수 있는 더 나은 도구로서의 가능성을 제시하고 있습니다. 특히, 이 연구는 비지리적 태그가 달린 이미지의 출처와 진성성을 확인하는 데 매우 중요한 기여를 하고 있습니다.



### Towards Efficient and Intelligent Laser Weeding: Method and Dataset for Weed Stem Detection (https://arxiv.org/abs/2502.06255)
Comments:
          Accepted by AAAI-AISI 2025

- **What's New**: 본 연구는 레이저 제초를 위한 잡초 인식을 최초로 실증 조사한 논문으로, 환경 친화적이고 효율적인 잡초 관리 방법을 제시합니다. 새로운 시스템은 작물의 손상을 피하면서도 레이저빔을 잡초 뿌리에 직접 겨냥할 수 있도록 설계되었습니다. 이를 위해 11,151개의 잡초 인스턴스가 주석으로 달린 고품질 잡초 줄기 탐지 데이터셋이 구축되었습니다.

- **Technical Details**: 이 연구에서는 작물 및 잡초 탐지와 함께 잡초 줄기 위치 추적을 통합한 일관된 end-to-end 시스템을 도입했습니다. 시스템은 이미지 시퀀스 또는 실시간 비디오 스트림을 처리하여 잡초 줄기를 정확히 찾을 수 있습니다. 이는 에너지 효율성을 높이고 작물에 대한 피해를 줄이기 위한 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존의 잡초 인식 시스템에 비해 잡초 제거 정확도를 6.7% 향상시켰으며 에너지 비용은 32.3% 감소했습니다. 이는 레이저 제초의 실용적 효율성을 크게 향상시키며, 지속 가능한 농업 관리에서 중요한 기여를 할 것으로 기대됩니다.



### Multi-Scale Transformer Architecture for Accurate Medical Image Classification (https://arxiv.org/abs/2502.06243)
- **What's New**: 이번 연구는 AI 기반의 피부 병변 분류 알고리즘을 소개하며, 이를 위해 향상된 Transformer 아키텍처를 활용합니다. 의학 이미징에서의 정확성과 견고성 문제를 해결하기 위해 다중 스케일 기능 융합 메커니즘과 자기 주의(self-attention) 과정을 개선하였습니다. 이를 통해 모호한 경계와 복잡한 구조를 가진 병변들을 효과적으로 탐지할 수 있는 능력이 향상되었습니다.

- **Technical Details**: 모델은 글로벌(global) 및 로컬(local) 특성을 모두 추출할 수 있도록 설계되었으며, 이를 통해 의료 이미지 분석의 성능을 극대화합니다. 연구에서는 ISIC 2017 데이터셋을 사용하여 성능을 평가하였고, ResNet50, VGG19, ResNext, Vision Transformer 등 기존 AI 모델들을 초월하는 성과를 보여주었습니다. 주요 성과 지표로는 accuracy, AUC, F1-Score, Precision 등이 있습니다.

- **Performance Highlights**: Grad-CAM 시각화를 통해 모델의 해석 가능성을 강조하며, 알고리즘의 주목 영역과 실제 병변 부위 간의 강한 일치를 보여줍니다. 이번 연구는 첨단 AI 모델이 의료 이미징의 변혁 잠재력을 가지고 있음을 강조하며, 더 정확하고 신뢰할 수 있는 진단 도구의 개발로 이어질 수 있습니다. 향후 연구에서는 이 접근 방식의 확장 가능성을 탐구하고, AI 기반 진단 프레임워크를 개선하기 위해 다중 모달(multimodal) 데이터를 통합할 계획입니다.



### Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds (https://arxiv.org/abs/2502.06227)
Comments:
          30 pages, 10 figures

- **What's New**: 이번 논문은 기존의 수치 기반 Leaf-Wood 분리를 위해 고안된 GrowSP 아키텍처를 기반으로 하여, 다중 분광(Multispectral) 항공 레이저 스캐닝(Point Clouds)에서 나무-잎 분리를 수행하는 완전 비지도 학습 방법인 GrowSP-ForMS를 제안합니다. 이 방법은 높은 밀도의 MS ALS 포인트 클라우드에 대해 84.3%의 평균 정확도와 69.6%의 평균 교차 비율(mIoU)을 달성하였으며, 기존의 비지도 알고리즘들과 비교해 현저히 우수한 성능을 보였습니다.

- **Technical Details**: GrowSP-ForMS는 고밀도의 다중 분광 포인트 클라우드에서 나무-잎 분리를 위해 특별히 설계된 비지도 심층 학습(deep learning) 접근법으로, 기존의 알고리즘들이 의존하는 세부 지오메트리 정보를 덜 사용합니다. 이 논문에서 제안된 방법은 GrowSP 모델의 성능을 높이기 위해 수정된 내용을 포함하고 있으며, 이를 바탕으로 수행된 ablation study를 통해 수치적 향상을 확인하였습니다.

- **Performance Highlights**: GrowSP-ForMS는 기존의 비지도 방법들보다 뛰어난 성능을 보여주었으며, 최근의 수퍼바이즈드 딥 러닝 방법들과 비교할 때도 비슷한 성능을 나타냈습니다. 평균 mIoU 점수는 기존의 GrowSP 모델보다 29.4pp 증가하였으며, 다중 분광 데이터를 사용할 경우 단일 분광 데이터보다 5.6pp의 향상이 있음을 보여주었습니다.



### FunduSAM: A Specialized Deep Learning Model for Enhanced Optic Disc and Cup Segmentation in Fundus Images (https://arxiv.org/abs/2502.06220)
- **What's New**: 이번 연구에서 제안하는 FunduSAM은 optic disc (OD) 및 optic cup (OC) 분할 작업에 특화된 딥러닝 모델로, 기존의 Segment Anything Model (SAM)에 여러 가지 Adapter를 추가하여 구조화하여 성능을 개선합니다. 이 모델은 Convolutional Block Attention Module (CBAM)을 통합하여 흐릿한 경계 및 낮은 대비 문제를 해결하며, polar transformation을 통해 fundus 이미지를 최적의 형식으로 변환합니다. 또한, OD와 OC 간의 구조적 보존을 위해 공동 손실(joint loss)을 사용하는 것이 특징입니다.

- **Technical Details**: FunduSAM은 SAM 구조를 기반으로 하여 fundus 이미지 처리를 위한 이미지 인코더와 Adapter 레이어를 개선하였습니다. 특히, polar transformation을 통해 OD 및 OC의 비율을 조정하고, Transformer 블록 내에 두 개의 Adapter 레이어를 설정하여 로컬화된 파라미터 효율적 미세 조정(PEFT)을 수행합니다. 이미지 인코더는 16개의 Transformer 블록으로 구성되어 있으며, 각 블록은 지역 정보 및 전반적인 이미지 맥락을 캡처하는 전역 주의력(global attention) 및 창(window) 기반 주의력 메커니즘을 활용합니다.

- **Performance Highlights**: REFUGE 데이터셋에서 1,200개의 fundus 이미지를 사용한 실험 결과, FunduSAM은 다섯 가지 주류 접근 방식에 비해 우수한 성능을 입증했습니다. 특히, FunduSAM은 정밀한 분할을 가능하게 하여 OD와 OC 간의 구조적 관계를 효과적으로 유지하며, 임상 진단의 정확도를 높이는 데 기여할 것으로 보입니다. 이러한 성과는 FunduSAM의 혁신적인 아키텍처와 협력적인 손실 함수가 결합된 결과입니다.



### Fully Exploiting Vision Foundation Model's Profound Prior Knowledge for Generalizable RGB-Depth Driving Scene Parsing (https://arxiv.org/abs/2502.06219)
Comments:
          10 pages, 5 figures

- **What's New**: 최근 비전 기초 모델(Vision Foundation Models, VFM)의 발전은 컴퓨터 비전 분야의 새로운 시대를 열었습니다. 이 논문은 RGB-Depth 드라이빙 장면 파싱이라는 새로운 연구 영역에 주목하여, VFMs의 깊은 특징을 효과적으로 활용할 수 있는 방법을 제안합니다. 특히, Heterogeneous Feature Integration Transformer(HFIT)라는 네트워크를 통해 RGB와 Depth 데이터를 통합하는 방법을 연구합니다.

- **Technical Details**: HFIT는 VFMs의 일반화 가능한 RGB-Depth 드라이빙 장면 파싱을 위해 설계되었으며, 상대적 깊이 예측 결과를 기반으로 다양한 이질적 특징을 추출하고 통합하는 기능을 가지고 있습니다. HFIT는 디플렉스 공간 사전 추출기(Duplex Spatial Prior Extractor, DSPE)와 재조정된 이질적 특징 융합 모듈(Recalibrated Heterogeneous Feature Fusion Module, RHFF)을 포함하여, 여러 수준의 특징을 선택적으로 통합합니다.

- **Performance Highlights**: 제안된 HFIT는 Cityscapes와 KITTI Semantics 데이터 세트에서 기존의 모든 전통적인 단일 모달 및 데이터 융합 장면 파싱 네트워크보다 뛰어난 성능을 보여주었습니다. 또한, 상대적 깊이 예측 결과를 사용함으로써 깊이 맵에 대한 의존성을 극복하고, 자원 소모를 최소화하여 성능을 향상시키는 전략이 유효함을 입증하였습니다.



### Comparing Image Segmentation Algorithms (https://arxiv.org/abs/2502.06201)
- **What's New**: 이번 연구는 시뮬레이티드 어닐링(simulated annealing) 기법을 이용한 이진 이미지의 노이즈 제거 기법을 제안합니다. 이 방법은 복잡한 비볼록 에너지 함수 문제를 해결하는 데 강점을 가지고 있으며, 이로 인해 기존의 반복 조건 모드(Iterative Conditional Modes, ICM) 기법보다 더 우수한 성능을 나타냅니다. 실험 결과에 따르면 제안된 알고리즘은 원본 이미지와 99.19% 일치를 보이며, ICM의 96.21%와 비교할 때 개선 효과가 두드러집니다.

- **Technical Details**: 연구진은 노이즈가 포함된 이미지와 깨끗한 이미지 간의 관계를 정의하는 에너지 함수 E(x, y)를 제안하였습니다. 시뮬레이티드 어닐링은 메탈 가공에서의 어닐링 과정을 모방하여 전역 최적화(global optimization) 기법을 사용합니다. 이 방법은 비볼록성(non-convex) 에너지 함수를 근거로 하며, 지역 최적(local optimum)에서 벗어날 수 있는 탐색 전략을 포함하고 있습니다.

- **Performance Highlights**: 본 알고리즘의 성능은 실험적으로 입증되어, 전통적인 ICM 기법을 능가하는 노이즈 제거 품질을 보여줍니다. 특히, 시각적 평가에서도 노이즈 제거와 구조적 세부 사항의 보존이 효과적으로 이루어져, 이진 이미지 처리(imaging processing) 분야에서의 유망한 접근법으로 자리매김합니다. 향후 이 연구는 이미지 복원(image restoration) 작업에서의 전역 최적화 기법의 장점을 부각시키며, 이미지 처리 기술의 발전에 기여할 것으로 기대됩니다.



### Multimodal Task Representation Memory Bank vs. Catastrophic Forgetting in Anomaly Detection (https://arxiv.org/abs/2502.06194)
- **What's New**: 이번 논문에서는 Unsupervised Continuous Anomaly Detection (UCAD)에서의 기존 문제를 해결하기 위한 새로운 방법인 Multi-modal Task Representation Memory Bank (MTRMB)를 제안하였다. MTRMB는 Key-Prompt-Multimodal Knowledge (KPMK) 메커니즘을 통해 BERT와 ViT 간의 다중 모드 특성 상호 작용을 향상시킨다. 또한, Grounding DINO와 SAM을 활용한 정제된 구조 기반 대조 학습(Refined Structure-based Contrastive Learning, RSCL)을 도입하여 정확한 세분화 마스크를 생성하고, 이로 인해 성능을 크게 향상시킨다.

- **Technical Details**: MTRMB는 다중 작업에 대한 지속적인 학습을 지원하기 위해 Key-Prompt-Multimodal Knowledge 메모리 공간을 활용한다. 훈련 단계에서는 특정 작업의 키, 프롬프트 및 다중 모드 지식을 저장하고, 테스트 단계에서는 작업 키를 요청하여 해당 작업 프롬프트를 검색한다. RSCL은 Grounding DINO 및 SAM의 정확한 마스크를 활용하여 서로 다른 구조의 특성을 더 효과적으로 구분할 수 있게 하여, 도메인 이동을 감소시키고, 더 견고한 특성 표현을 제공한다.

- **Performance Highlights**: MTRMB는 MVtec AD와 VisA 데이터셋에서 실험한 결과, 평균 감지 정확도 0.921을 달성하며 기존의 최첨단 방법을 능가하는 성과를 보였다. 낮은 망각률에서도 높은 성능을 유지하는 것을 통해 다중 작업에서의 효용성을 입증하였다. 본 연구는 GitHub를 통해 오픈 소스될 예정이며, 지속적인 학습이 필요한 산업 검사 환경에서 실질적인 기여를 할 것으로 기대된다.



### Multi-Level Decoupled Relational Distillation for Heterogeneous Architectures (https://arxiv.org/abs/2502.06189)
- **What's New**: 새로운 연구인 Multi-Level Decoupled Relational Knowledge Distillation (MLDR-KD) 방법을 제안합니다. 이 방법은 이질적(heterogeneous) 모델 간의 지식 전이를 최적화하여 숨겨진 dark knowledge를 효과적으로 활용할 수 있게 합니다. 기존의 방법들이 dark knowledge를 충분히 활용하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 Decoupled Finegrained Relation Alignment (DFRA)와 Multi-Scale Dynamic Fusion (MSDF) 모듈을 도입합니다. DFRA는 teacher 모델의 로그잇(logit)과 feature 레벨에서 finegrained 관계를 분리하여 조정하는 것을 통해 학생 모델의 분류 신뢰성을 높입니다. MSDF 모듈은 다양한 스케일(feature level)의 특성 지도를 동적으로 융합하여 분류 성능을 향상시킵니다.

- **Performance Highlights**: MLDR-KD는 CNN, Transformers, MLPs, Mambas의 4개 모델 아키텍처에서 성능을 검증하였습니다. CIFAR-100 데이터셋에서 최대 4.86%, Tiny-ImageNet 데이터셋에서 최대 2.78%의 성능 향상을 보여주며, 다양한 아키텍처 간의 지식 전이가 효과적으로 이루어짐을 입증했습니다.



### CANeRV: Content Adaptive Neural Representation for Video Compression (https://arxiv.org/abs/2502.06181)
- **What's New**: 최근 동영상 압축 분야에서 Implicit Neural Representation (INR) 기반의 방법들이 주목받고 있습니다. 이러한 방법들은 전체 비디오 시퀀스의 전역 의존성과 특성을 효과적으로 포착하여 우수한 압축 잠재력을 보여줍니다. 그러나 기존의 INR 방식들은 고정된 네트워크 구조를 사용하여 동적 변화를 다루는 데 한계가 있으며, 이에 따라 CANeRV(Content Adaptive Neural Representation for Video Compression)라는 새로운 압축 네트워크를 제안하고 있습니다.

- **Technical Details**: CANeRV는 각 비디오 시퀀스의 특성에 따라 구조 최적화를 동적으로 수행하는 혁신적인 INR 기반의 비디오 압축 네트워크입니다. 이는 두 가지 조정 메커니즘인 Dynamic Sequence-Level Adjustment (DSA)와 Dynamic Frame-Level Adjustment (DFA)를 도입하여 동적 정보를 포착하고 영상의 세부 구조 정보를 효과적으로 캡처합니다. 마지막으로, Hierarchical Structural Adaptation (HSA) 메커니즘을 통해 공간 구조 정보를 효과적으로 포착하여 세부 복원 능력을 향상시키는 방법을 사용하고 있습니다.

- **Performance Highlights**: 실험 결과, CANeRV는 다양한 비디오 데이터 셋에서 H.266/VVC 및 최신 INR 기반의 비디오 압축 기술들을 능가하는 성능을 보여주었습니다. CANeRV는 유사한 수의 파라미터로 높은 품질의 비디오를 재구성하며, 기존 방식보다 압축 효율성을 크게 향상시켰습니다. 또한 다양한 비디오 시퀀스에서 우수한 성능을 발휘하며, 이는 비디오 압축 분야의 실질적인 진전을 의미합니다.



### PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts (https://arxiv.org/abs/2502.06172)
Comments:
          Submitting Preprint

- **What's New**: 최근 손글씨 텍스트 인식(HTR) 분야는 다양한 새로운 모델들이 등장하면서 서로 경쟁력을 주장하고 있습니다. 그러나 이러한 모델들을 공정하게 비교하기는 어렵습니다, 왜냐하면 테스트 세트의 일관성이 부족하고 언어별 데이터셋이 결여되어 있기 때문입니다. 본 논문은 페이지 수준의 손글씨 OCR 파이프라인을 위한 새로운 접근법인 PLATTER를 제안하며, 단 두 가지 주요 작업인 손글씨 텍스트 검출(HTD)과 HTR로 문제를 분할하여 이를 해결하고자 합니다.

- **Technical Details**: PLATTER는 페이지 수준의 손글씨 OCR 시스템을 구축하기 위한 두 단계의 문제로 구성됩니다. 첫 번째 단계는 HTD로, 이는 페이지에서 단어 수준의 손글씨 텍스트를 식별합니다. 두 번째 단계는 HTR로, HTD를 통해 탐지된 이미지를 기계가 읽을 수 있는 텍스트로 변환합니다. 이 프레임워크는 다양한 인디언 언어에 대해 모델의 성능을 측정하고, 10가지 인디언 언어에 대한 6개의 HTR 모델 평균 성능을 비교합니다.

- **Performance Highlights**: PLATTER 프레임워크는 사용자가 선택할 수 있는 여러 HTR 모델의 조합에 대한 정성적 및 정량적 분석을 제공합니다. 또한, CHIPS라는 새로운 페이지 수준의 손글씨 인디언 OCR 데이터셋을 발표하여 다양한 모델을 사전 훈련 할 수 있도록 지원하고 있습니다. 이 연구는 손글씨 데이터에 대한 라벨링이 부족한 문제를 해결하고, 여러 언어와 스크립트를 지원하는 힘을 가진 HTR 모델을 기반으로 강력한 시스템을 설계하는 데 기여합니다.



### An Interpretable Implicit-Based Approach for Modeling Local Spatial Effects: A Case Study of Global Gross Primary Productivity (https://arxiv.org/abs/2502.06170)
- **What's New**: 본 연구에서는 지리적 머신러닝에서 공간 이질성을 효과적으로 모델링하기 위한 새로운 접근법을 제시합니다. 전통적인 통계 학습 방법이 만족스럽지 못한 정확도를 보이는 문제를 해결하기 위해, 서로 다른 위치 간의 공통 기능과 공간적 차이를 동시에 모델링하는 심층 신경망 구조를 도입합니다. 이러한 방법은 일반적인 특징을 추출하면서도 지역적 변화를 반영할 수 있습니다.

- **Technical Details**: 제안된 방법은 인코더-디코더 구조를 갖춘 이중 분기 신경망으로 구성됩니다. 인코딩 단계에서는 그래프 컨볼루션 네트워크(GCN)와 장단기 메모리 네트워크(LSTM)를 사용하여 시공간 조건 그래프에서 노드 정보를 집계하며, 특정 위치의 시공간 이질성을 암묵적인 조건 벡터로 인코딩합니다. 디코딩 단계에서는 조건 생성 전략을 사용하여 응답 변수와 해석적 가중치를 예측합니다.

- **Performance Highlights**: 자체 검증을 위해 2001년부터 2020년까지의 글로벌 기후 및 토지 덮개 데이터를 활용하여 식물총생산성(GPP)을 예측하였습니다. 모델은 5천만 개의 샘플로 학습하고 280만 개의 샘플로 테스트하여 RMSE 0.836을 달성했으며, 기존의 LightGBM(1.063) 및 TabNet(0.944)을 능가하는 성능을 보였습니다. 시각화 분석을 통해 GPP의 주요 요인 분포 차이를 다양한 시간과 위치에서 드러낼 수 있음을 확인했습니다.



### Efficient-vDiT: Efficient Video Diffusion Transformers With Attention (https://arxiv.org/abs/2502.06155)
- **What's New**: 이 논문은 Diffusion Transformers (DiTs) 기반의 비디오 생성 모델의 비효율성 문제를 다룹니다. 기존의 3D 전반적인 주의(complex attention)는 계산 복잡성이 높아 비디오 생성에서 느린 추론(inference) 속도를 초래하는데, 이를 해결하기 위해 새로운 희소(sparse) 3D attention 방법을 제안합니다. 또한 샘플링 과정을 단축하여 비디오 생성을 가속화하는 기술을 도입했습니다.

- **Technical Details**: 연구진은 비디오 데이터에서의 중복을 분석하여 'Attention Tile'이라는 현상을 발견했습니다. 이 현상은 각 잠재 프레임(latent frames)이 모든 다른 프레임에 주의를 기울일 필요가 없음을 나타냅니다. 따라서 주의 계산(computation)을 선형 복잡성으로 줄이기 위해 각 잠재 프레임이 고정된 수의 다른 프레임만에 주의를 기울이도록 제한합니다. 이를 통해 전체 샘플링 경로를 여러 세그먼트로 나누고 각 세그먼트 내에서 일관성을 증진시키는 방식으로 샘플링 과정을 단축합니다.

- **Performance Highlights**: 효율적 비디오 생성을 위해 0.1%의 사전 훈련 데이터로 Open-Sora-Plan-1.2 모델을 7.4배에서 7.8배 더 빠르게 만들 수 있음을 보였습니다. 성능 저하가 최소화된 VBench 기준으로, 이 방법이 4개의 GPU를 사용할 때는 추가로 3.91배의 속도를 향상시켜 높은 효율성을 입증했습니다. 논문에서는 제안한 방법이 선행 연구와 어떻게 다른지, 그리고 비디오 생성 속도를 어떻게 향상시키는지를 명확히 입증했습니다.



### Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordanc (https://arxiv.org/abs/2502.06145)
Comments:
          Project Page: this https URL

- **What's New**: Animate Anyone 2는 캐릭터 애니메이션과 환경 연관성을 향상시키기 위해 새로운 접근 방식을 도입합니다. 기존의 diffusion 모델 기반 기술들이 직면했던 한계를 해결하고자, 캐릭터와 환경의 관계를 명확히 이해할 수 있도록 환경 정보를 조건부 입력으로 사용합니다. 이로 인해, 캐릭터가 환경과의 맥락을 유지하며 자연스럽게 애니메이션 될 수 있습니다.

- **Technical Details**: 이 연구는 캐릭터 이미지 애니메이션의 새로운 프레임워크인 Animate Anyone 2를 제안합니다. 본 프레임워크는 캐릭터와 환경을 의미 있게 분리하여 처리하며, 형태에 구애받지 않는 마스크 전략을 통해 캐릭터와 그 주변 환경 간의 관계를 효과적으로 학습합니다. 또한, 상호작용 오브젝트의 특징을 추출하는 객체 안내기(object guider)와 공간 혼합(spatial blending) 기법을 사용하여 객체 상호작용의 충실도를 높입니다.

- **Performance Highlights**: 실험 결과, Animate Anyone 2는 이전 기술들에 비해 높은 품질의 캐릭터 애니메이션 성능을 발휘하며, 특히 세 가지 주요 장점을 보여줍니다: 1) 자연스러운 장면 통합, 2) 일관된 객체 상호작용, 3) 다양한 복잡한 동작의 우수한 처리능력. 이러한 성과는 기존 메서드와 비교하여 강력한 캐릭터 애니메이션 품질을 입증합니다.



### Integrating Sequence and Image Modeling in Irregular Medical Time Series Through Self-Supervised Learning (https://arxiv.org/abs/2502.06134)
Comments:
          9 pages, 2 figures, AAAI2025

- **What's New**: 이 연구에서는 의료 분야에서 자주 발생하는 불규칙하고 결측 값이 많은 다변량 시간 시리즈 데이터를 처리하기 위해 시퀀스와 이미지 표현을 결합한 합동 학습 프레임워크를 제안합니다. 기존의 방법들이 주로 시퀀스나 이미지 중 하나의 모델링 관점만을 채택한 반면, 본 연구는 두 가지 표현을 통합하여 보다 일반화 가능한 결합 표현을 확보하고자 합니다. 이를 통해 세 가지 자가 지도 학습 전략(self-supervised learning strategies)을 설계하여 시퀀스와 이미지 표현의 융합을 촉진합니다.

- **Technical Details**: 제안된 접근법은 생성기-판별기 구조(generator-discriminator structure)와 적대적 전략(adversarial strategy)을 사용하는 시퀀스 모델링 브랜치와 다양한 이미지 변환 전략(image transformation strategies)을 활용하여 희소한 시리즈의 성능을 향상시키는 이미지 브랜치로 구성됩니다. 구체적으로 세 가지 자가 지도 학습 손실(loss)이 설계되었으며, 이는 각각 시퀀스 임퓨테이션 최적화, 일반화 가능한 결합 표현 학습, 그리고 유사 사례 간 클러스터링을 통해 결합 표현을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 PAM, P12, P19라는 세 가지 실제 임상 데이터셋에서 다른 최신 방법들과 비교해 우수한 분류 성능(classification performance)을 발휘했습니다. 특히 PAM 데이터셋에서 정확도(Accuracy)에서 3.1% 개선을 보였으며, P12와 P19에서는 각각 AUPRC와 AUROC에서 우수한 성과를 기록했습니다. 또한, 결측 값이 많이 발생하는 상황에서도 우리의 접근법이 다른 방법들보다 더욱 강건한 분류 성능을 보여줍니다.



### Enhancing Document Key Information Localization Through Data Augmentation (https://arxiv.org/abs/2502.06132)
Comments:
          Accepted as a workshop paper in DOCUI-AAAI2025

- **What's New**: 본 논문은 디지털 및 손글씨 문서에서 핵심 정보를 로컬화하는 방법을 제시합니다. 특히, 디지털 문서만을 이용하여 훈련한 후, 손글씨 문서의 특성을 모방함으로써 일반화 능력을 높이는 데이터 증강(data augmentation) 기술을 사용합니다. 실험 결과, 제안된 파이프라인이 모델의 성능을 향상시키는 데 효과적임을 보여주었습니다.

- **Technical Details**: 연구는 Form-NLU 데이터셋을 사용하여 해당 파이프라인의 유효성을 테스트했습니다. Augraphy를 사용하여 다양한 문서 유형을 생성하고, 특정 여섯 가지 증강 기법을 선택하였습니다: InkBleed, Letterpress, LowInkRandomLines, LowInkPeriodicLines, JPEG, DirtyScreen 등이 그것입니다. 각 훈련 및 검증 문서에 대해 다섯 개의 증강된 버전을 생성하며, 회전 및 텍스트 효과를 고려하여 총 70%와 50%의 확률로 적용합니다.

- **Performance Highlights**: 제안된 파이프라인의 결과는 세 가지 모델에서 손글씨 문서 정보 로컬화 성능을 3.97% 향상시켰고, 특히 문서 이미지로 미리 훈련된 백본 모델이 자연 이미지로 훈련된 모델보다 일반화 능력이 우수하다는 것을 보여주었습니다. LayoutLMv3는 디지털 문서에서는 높은 성능을 보였으나, 손글씨 문서에서는 OCR 오류로 인해 성능이 떨어졌습니다. 이러한 결과는 증강 방식이 디지털 문서만을 사용하여 손글씨 문서의 도메인을 효과적으로 모방함을 입증합니다.



### Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models (https://arxiv.org/abs/2502.06130)
Comments:
          Accepted by ICLR 2025. Project page:this https URL

- **What's New**: 최근 대규모 시각-언어 모델(LVLMs)의 성능 향상이 눈에 띄지만, 현실에서는 시각 입력과 일치하지 않는 환각적(hallucinatory) 텍스트 응답을 생성하는 경향이 있습니다. 이 논문은 텍스트-이미지 생성 모델을 활용하여 LVLM의 환각을 완화할 가능성을 모색합니다. 새로운 자기 수정 디코딩 기법인 Generative Feedback(DeGF)를 제안하여 이러한 환각을 효과적으로 줄입니다.

- **Technical Details**: DeGF는 정교한 훈련 과정 없이 텍스트-이미지 생성 모델로부터의 피드백을 통합하여 LVLM의 응답 정확성을 향상시키는 알고리즘입니다. 이 방법은 LVLM이 생성한 초기 응답을 기준으로 새로운 이미지를 생성하고, 이 보조 시각 참조를 통해 응답의 일치를 평가합니다. 구체적으로, 초기 응답과 생성된 이미지 간의 차이를 활용하여 초기 응답을 검증하고 수정하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, DeGF는 POPE 및 MMBench와 같은 여섯 가지 벤치마크에서 다양한 유형의 환각을 효과적으로 줄이며 기존 최첨단 기법보다 일관되게 우수한 성능을 보였습니다. 연구팀은 LVLM 응답의 정확성과 구체성을 향상시킬 수 있는 능력을 입증하였으며, 이는 시각 인사이트를 더하고 초기 응답을 확인하여 정확한 세부 사항을 보장하는 데 기여합니다.



### Improved YOLOv5s model for key components detection of power transmission lines (https://arxiv.org/abs/2502.06127)
Comments:
          23 pages, 14 figures

- **What's New**: 전력 전송선의 지능형 검사에 대한 연구가 진행되고 있습니다. 본 논문에서는 YOLOv5s 모델을 기반으로 하는 개선된 물체 탐지 모델을 제안하여 전송선의 주요 컴포넌트에 대한 탐지 정확도를 향상시켰습니다. 새로운 기법은 전력망 검사 이미지의 특성을 반영하여 검사 정확성을 높이고자 하였습니다.

- **Technical Details**: k-means 클러스터링의 거리 측정을 수정하여 YOLOv5s 모델의 앵커 매칭을 개선하였으며, CNAM(Convolutional Block Attention Module) 주의 메커니즘을 백본 네트워크에 추가하였습니다. 결과적으로, 클래스 간 불균형을 줄이기 위해 focal loss 함수를 적용하였습니다.

- **Performance Highlights**: 개선된 방법의 mAP(Mean Average Precision)는 98.1%, 정밀도(Precision)는 97.5%, 재현율(Recall)은 94.4%에 달하며, 탐지 속도는 84.8 FPS로 측정되었습니다. 실험 결과, 본 모델은 탐지 정확도를 향상시키고 다른 모델에 비해 뛰어난 성능을 보였습니다.



### An Appearance Defect Detection Method for Cigarettes Based on C-CenterN (https://arxiv.org/abs/2502.06119)
Comments:
          19 pages, 14 figures

- **What's New**: 본 연구에서는 전통적인 방법들이 자동 담배 생산 라인에서 담배 결함을 정확하게 식별하는 데 어려움을 겪는 점에 착안하여, C-CenterNet에 기반한 새로운 담배 외관 결함 탐지 방법을 제안했습니다. 이 방법은 keypoint estimation을 통해 중심점을 위치시키고, 다양한 결함 속성을 회귀하는 방식으로 작동합니다.

- **Technical Details**: C-CenterNet은 Resnet50을 백본(feature extraction) 네트워크로 사용하며, convolutional block attention mechanism (CBAM)을 도입하여 효과적인 특징 추출 능력을 향상시킵니다. 또한, feature pyramid network를 통해 각 레이어의 특징 추출을 강화하고, deformable convolution을 적용하여 다양한 형태의 결함 학습 능력을 증가시킵니다.

- **Performance Highlights**: 실험 결과는 mean Average Precision (mAP) 지표를 통해 평가되었으며, C-CenterNet 모델의 mAP는 95.01%로 나타났습니다. 기존 CenterNet 모델과 비교할 때 성공률이 6.14% 증가하여, 자동 담배 생산 라인에서의 정밀성과 적응성 요구 사항을 충족할 수 있음을 보여줍니다.



### A Novel Multi-Teacher Knowledge Distillation for Real-Time Object Detection using 4D Radar (https://arxiv.org/abs/2502.06114)
- **What's New**: 본 논문은 4D 레이더(Radar) 기술을 활용하여 자율주행 시스템에서의 3D 객체 탐지의 정확성을 높이는 새로운 접근 방식을 제안합니다. 기존의 리다(LiDAR) 시스템이 악천후에 취약한 반면, 4D 레이더는 거리, 방위각(azimuth), 그리고 도플러 속도 외에 고도(elevation)까지 측정함으로써 모든 기상 조건에서 신뢰성을 보장합니다. 이 연구의 핵심은 4D 레이더의 밀집 텐서(dense tensor)를 활용하여 희소(point cloud sparsity) 문제를 해결하고, 새로운 지식 증류(framework) 접근 방식을 통해 객체 탐지 성능을 향상시켰다는 점입니다.

- **Technical Details**: 제안된 방법은 4D 레이더 텐서가 포함된 데이터를 이용하여 새로운 '4D Multi Teachers' 프레임워크를 통해 희소한 입력을 고밀도 표현으로 변환하는 과정을 포함합니다. 이 과정은 N개의 교사 모델과 하나의 학생 모델로 구성되며, 각 모델은 서로 다른 데이터 밀도를 사용합니다. 훈련시는 고밀도 포인트 클라우드를 이용하여 교사 모델을 학습한 후, 추론 시에는 학생 모델이 희소 포인트 클라우드를 직접 사용하여 낮은 메모리 소비와 높은 처리 속도를 유지합니다.

- **Performance Highlights**: 우리는 K-Radar 데이터셋을 이용한 실험을 통해, 제안된 방법이 기존의 최첨단 RTNH 모델보다 25% 더 높은 탐지 성능을 달성했음을 보여주었습니다. 이 과정에서 NVIDIA RTX 3090 GPU를 사용하여 실시간 추론 속도 30FPS를 기록하며 효율성도 유지하였습니다. 이는 자율주행과 같이 엄격한 실시간 요구 사항이 있는 응용 프로그램에 매우 적합함을 나타냅니다.



### Col-OLHTR: A Novel Framework for Multimodal Online Handwritten Text Recognition (https://arxiv.org/abs/2502.06100)
Comments:
          ICASSP 2025

- **What's New**: Col-OLHTR는 Online Handwritten Text Recognition (OLHTR)을 위한 새로운 협력 학습 프레임워크로, 단일 스트림 추론(uni-stream inference) 프로세스를 유지하면서 훈련 중 멀티모달(features) 특성을 학습합니다. 이전 방법들은 일반적으로 시퀀스 인식 문제로 OLHTR을 다뤘으나, Col-OLHTR은 그 구조를 간소화하여 성능을 유지하며 더 효과적으로 특성을 추출할 수 있습니다. P2SA(Point-to-Spatial Alignment) 모듈이 추가되어 안정적인 학습 및 매핑 메커니즘을 가능하게 합니다.

- **Technical Details**: 제안된 Col-OLHTR는 훈련 단계에서 두 개의 주요 스트림, 즉 궤적 수준(trajectory-level) 및 이미지 수준(image-level) 스트림을 활용합니다. 궤적 신호는 1D 인코더와 주의 기반 디코더(Attention-based Decoder)에 직접 입력되고, 이미지 스트림에서는 궤적을 2D 이미지로 렌더링 후 2D 인코더에 전달됩니다. P2SA 모듈은 궤적 수준 특성에서 공간 인식을 가능하게 하여 효율을 높이는 역할을 합니다.

- **Performance Highlights**: Col-OLHTR는 여러 OLHTR 벤치마크에서 최신 기술 수준(SOTA) 성능을 달성했으며, 이는 설치된 구조의 단순함에도 불구하고 높은 성능을 유지한다는 것을 보여줍니다. 이 연구는 다양한 OLHTR 작업에서 여러 특성을 효과적으로 결합하여 우수한 경량(off-the-shelf) 솔루션을 제공합니다. 또한 기존의 멀티스트림 접근 방식에 비해 추론 비용을 크게 줄여 효율성을 높였습니다.



### Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models (https://arxiv.org/abs/2502.06094)
- **What's New**: 이 논문에서는 의료 분야에서 공정성을 보장하기 위한 새로운 모델인 Fair-MoE를 제안하고 있습니다. Fair-MoE는 공정성을 중시하는 전문 혼합 모델인 FO-MoE와 공정성 지향 손실인 FOL의 두 가지 핵심 구성 요소로 이루어져 있습니다. 이 모델은 의료 데이터셋에서의 편향을 최소화하고 다양한 속성을 고려하여 공정성과 정확성을 동시에 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: Fair-MoE는 다양한 전문성을 활용하여 편향된 패치 임베딩을 필터링할 수 있도록 설계되었습니다. FO-MoE는 여러 전문가의 지식을 집합적으로 활용해 특정 작업에 적합한 공정한 정보를 추출합니다. FOL은 각 속성 간의 거리를 최소화하고 다양한 속성의 분포 차이를 최적화하는 새로운 공정성 지향 손실 함수를 도입합니다.

- **Performance Highlights**: Harvard-FairVLMed 데이터셋에서의 확장된 실험 결과, Fair-MoE는 모든 네 가지 속성에서 공정성과 정확성 모두에서 개선된 결과를 보여주었습니다. 이러한 결과는 Fair-MoE가 의료 분야의 딥러닝 모델에서 신뢰를 구축하는 데 기여할 수 있음을 시사합니다. 코드도 공개될 예정이므로 연구자들은 이를 통해 공정한 의료 AI 모델 개발을 촉진할 수 있습니다.



### Traveling Waves Integrate Spatial Information Into Spectral Representations (https://arxiv.org/abs/2502.06034)
- **What's New**: 이 논문에서는 뇌의 traveling waves(전파파)가 어떻게 공간 정보를 통합하고 전송하는 역할을 하는지를 탐구합니다. 특히, convolutional recurrent neural networks(컨볼루션 순환 신경망)를 통해 이러한 traveling waves를 생성하고 이를 통해 시각 자극에 대한 반응을 관찰합니다. 기존의 feed-forward networks(피드포워드 신경망)에 비해, traveling waves는 전역적인 공간 정보 처리를 위한 새로운 표현 공간을 제공합니다.

- **Technical Details**: 각 모델은 제한된 receptive field(수용 범위)를 가지며, 이들 네트워크는 반복적 처리(recurrent processing)를 통해 글로벌 정보 처리 문제를 해결해야 합니다. 이에 따라, 네트워크의 재발성 활동(recurrent activity)을 스펙트럼 분해(spectral decomposition)를 통해 추출하여 traveling waves를 통한 정보 통합을 달성합니다. 각 네트워크는 시맨틱 세분화(semantic segmentation) 작업을 통해 학습되며, 로컬 정보에 대해서만 접근 가능합니다.

- **Performance Highlights**: 여러 데이터셋을 통해 traveling waves가 정보를 전송하는 데 효과적이고, 스펙트럼 분해를 통해 정보를 디코딩할 수 있음을 보여줍니다. 이 연구 결과는 traveling wave 기반 모델들이 복잡한 인지 작업에서 기존 모델들보다 우수한 성능을 보이며, 정보 전달의 효율성을 향상할 수 있는 가능성을 제시합니다. 또한, 이러한 메커니즘은 생물학적 신경 활동의 기록과 연결할 수 있는 새로운 틀을 제공합니다.



### DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations (https://arxiv.org/abs/2502.06029)
Comments:
          14 pages, cvpr template

- **What's New**: 이번 연구에서는 DiTASK라는 새로운 접근법을 제안하여 Diffeomorphic Multi-Task Fine-Tuning을 통해 멀티태스크 학습의 효율성을 향상시킵니다. 기존의 low-rank adaptation (LoRA) 방법이 가지는 한계를 극복하며, 학습된 weight matrix의 singular vectors를 보존하면서도 task-specific한 조정을 가능하게 합니다. 이를 통해 DiTASK는 최소한의 추가 파라미터로도 공유 및 태스크 특정 feature를 조정할 수 있도록 합니다.

- **Technical Details**: DiTASK는 Continuous Piecewise Affine-Based (CPAB) 변환을 활용한 신경 diffeomorphic 변환을 사용하여 pretrained feature 패턴을 보존합니다. 이 방법은 각 최적화 단계에서 singular value를 동적으로 조정하며, 연속적이고 역전 가능한 변형을 보장합니다. 이러한 접근법은 pre-trained representation의 구조를 유지하면서도 여러 태스크에 대한 적응을 가능하게 만들어, 기존의 방법들보다 메모리 효율성을 높입니다.

- **Performance Highlights**: DiTASK는 PASCAL MTL 데이터셋에서 기존의 MTLoRA 방법에 비해 26.27% 향상된 평균 태스크 성과를 달성하며, 75% 적은 파라미터로 이러한 성능을 유지합니다. 이는 미리 훈련된 feature 방향을 보존하면서도 효율적인 멀티태스크 학습을 위한 새로운 패러다임을 제시합니다. DiTASK는 상태-of-the-art 성능을 달성하며, 다수의 밀집 예측 태스크에서도 우수한 결과를 보여줍니다.



### Dual Caption Preference Optimization for Diffusion Models (https://arxiv.org/abs/2502.06023)
- **What's New**: 최근 인공지능 분야에서는 인간의 선호(Preference) 최적화 기술이 큰 주목을 받고 있습니다. 이 기술은 대규모 언어 모델(Large Language Models, LLMs)에서 개발되었으며, 텍스트-이미지 확산 모델의 성능을 크게 향상시킬 수 있는 가능성을 보여주고 있습니다. 특히, 연구진은 두 가지 별도의 캡션(Dual Caption)을 사용하여 이미지 간의 선호도를 명확히 구분하는 새로운 방법인 DCPO(Dual Caption Preference Optimization)를 제안했습니다.

- **Technical Details**: DCPO는 두 가지 단계로 구성되어 있습니다: 첫째, 더욱 잘 정렬된 캡션(captions)을 생성하는 텍스트 생성 프레임워크와 둘째, 이러한 캡션을 활용하여 최적화하는 혁신적인 목표 함수입니다. 연구진은 Pick-Double Caption 데이터셋을 도입하였으며, 이는 선호된 이미지와 선호되지 않은 이미지에 대해 별도의 캡션을 제공함으로써 데이터 중복 문제(conflict distribution)를 해결하고자 합니다. 또한, 캡션 생성을 위한 여러 가지 접근법인 캡션 모델, 교란(perturbation) 접근법, 그리고 하이브리드 방법을 제안하여 보다 효과적인 캡션 생성을 목표로 합니다.

- **Performance Highlights**: 실험 결과, DCPO는 Stable Diffusion 2.1, SFT_Chosen, Diffusion-DPO, MaPO 등의 다른 기존 방법들에 비해 모든 지표에서 우수한 성능을 보였습니다. 특히, Pickscore에서 +0.21, HPSv2.1에서 +0.45, 정규화된 ImageReward에서 +1.8, CLIPscore에서 +0.15, GenEval에서 3%의 성능 향상을 이루었습니다. 이러한 결과들은 DCPO가 선호도 최적화 분야에서 유망한 접근 방식임을 암시합니다.



### Temporal Working Memory: Query-Guided Segment Refinement for Enhanced Multimodal Understanding (https://arxiv.org/abs/2502.06020)
Comments:
          Accepted at NAACL 2025

- **What's New**: 이 논문에서는 다중 모달 기초 모델(Multimodal Foundation Models, MFMs)의 시계열 모델링 능력을 향상시키기 위해 시간 작업 기억(Temporal Working Memory, TWM)이라는 특화된 인지 모듈을 소개합니다. TWM은 비디오 및 오디오 콘텐츠를 처리하는 동안 중요한 세부정보를 보존하면서 작업 관련 정보를 선택적으로 유지합니다. 또한, TWM은 쿼리 기반의 주의(attention) 접근 방식을 사용하여 시계열 내에서 가장 유용한 다중 모달 세그먼트에 집중하여 모델의 제한된 용량을 최적화합니다.

- **Technical Details**: TWM은 MFMs에 쉽게 통합될 수 있는 플러그 앤 플레이 모듈로, 비디오 캡셔닝, 질의 응답, 비디오-텍스트 검색 등 다양한 작업에서 아홉 개의 최첨단 모델의 성능을 유의미하게 향상시킵니다. 이 메커니즘은 쿼리 기반 선택 모듈을 통합하여, TF-IDF 기반의 정보 검색 시스템처럼 중요 세그먼트를 선별하고 기억하는 방식으로 작동합니다. 이를 통해 MFMs는 시간적 차원에서 정보의 선택적 유지 및 관리가 가능해집니다.

- **Performance Highlights**: TWM은 비디오 캡셔닝, 질의 응답 및 비디오-텍스트 검색과 같은 과제에서 성능을 현저히 개선하는 결과를 보여줍니다. 실험 결과, TWM을 적용한 모델들이 상당한 성능 향상을 보였으며, 이는 복잡한 시간 민감 데이터 처리를 효과적으로 수행할 수 있는 기초를 마련합니다. TWM을 통해 MFMs의 시간적 모델링 능력이 확장됨으로써, 다양한 다중 모달 작업에서 차별화된 성과를 기대할 수 있습니다.



### Noise is an Efficient Learner for Zero-Shot Vision-Language Models (https://arxiv.org/abs/2502.06019)
Comments:
          Our code is available at this https URL

- **What's New**: 최근 Test-Time Adaptation(TTA)이 레이블이 없는 데이터로 모델 튜닝을 위한 방법으로 주목받고 있습니다. 본 논문에서는 TTA의 전통적인 방식이 시각적 표현의 잠재적인 분포 이동을 간과하고 있음을 지적하고, 이를 보완하기 위한 새로운 방법인 Test-Time Noise Tuning(TNT)을 제안합니다. TNT는 학습 가능한 노이즈를 시각 입력 공간에 직접 최적화하여 불규칙한 변화를 효과적으로 처리할 수 있도록 합니다.

- **Technical Details**: TNT의 핵심은 두 가지 목표를 가지고 학습 가능한 노이즈를 적용하는 것입니다. 첫 번째는 레이블이 있는 적응을 적용하여 마진 엔트로피를 최소화하는 것이며, 두 번째는 이미지의 다양한 증강 뷰 간의 일관성을 극대화하는 것입니다. 이렇게 함으로써 모델은 표면적인 세부사항보다는 핵심적인 불변 특징에 초점을 맞출 수 있습니다. 또한, TNT는 온도 스케일링(temperature scaling)을 통해 성능을 더욱 강화합니다.

- **Performance Highlights**: TNT는 두 개의 기존 벤치마크에서 7개의 강력한 TTA 기준과 비교했을 때 더 높은 정확도와 더 나은 보정(calibration)을 보여주었습니다. 특히, 자연 데이터 분포 기준에서 평균 7.38% 개선과 제로샷 CLIP으로 평가한 교차 데이터셋에서 0.80% 개선 효과가 있었습니다. 이러한 성과는 VLM의 범용성과 보정 능력을 크게 향상시키며, 실제 세계의 데이터 변동 상황에서도 강력한 성능을 유지할 수 있게 합니다.



### A Comprehensive Survey on Image Signal Processing Approaches for Low-Illumination Image Enhancemen (https://arxiv.org/abs/2502.05995)
- **What's New**: 디지털 콘텐츠(photos 및 videos)의 활용도가 증가하면서 고품질 그래픽 정보의 필요성이 커지고 있습니다. 본 논문은 저조도(image enhancement) 이미지 개선을 위한 최신 기술의 발전을 다루고 있으며, 특히 딥러닝 기반 방법들이 기존 방법보다 높은 성능을 보이고 있습니다. 저조도 환경에서도 중요한 정보를 유지하면서 노이즈를 효과적으로 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: 논문에서는 저조도 환경에서의 이미지 신호 처리(image signal processing) 방법을 검토하고, 이를 세 가지 범주로 분류하였습니다. 하이브리드 기술(hybrid techniques), 딥러닝 기반 방법(deep learning-based methods), 전통적 접근법(traditional approaches)으로 나뉘며, 각 접근법의 장점과 한계도 논의합니다. 딥러닝 기반 기술은 합성곱 신경망(CNNs)을 활용하여 저조도 이미지의 특징을 인식하고 추출합니다.

- **Performance Highlights**: 전통적인 기법들은 노이즈 감소(denoising), 자동 화이트 밸런싱(automated white balancing), 및 기타 개선 기술을 포함합니다. 하이브리드 접근법은 딥러닝 방법과 기존 기법을 결합하여 더 나은 결과를 도출합니다. 이 논문은 다양한 접근법의 효과를 비교 분석하고, 향후 연구 방향에 대한 통찰도 제공합니다.



### SNAT-YOLO: Efficient Cross-Layer Aggregation Network for Edge-Oriented Gangue Detection (https://arxiv.org/abs/2502.05988)
- **What's New**: 이 논문에서는 느린 검출 속도와 낮은 정확도, 산업 엣지 디바이스에 배포하기 어려운 문제들을 해결하기 위해 경량화된 석탄 광석 목표 검출 알고리즘을 제안합니다. ShuffleNetV2를 백본으로 사용하여 검출 성능을 향상시키고, ADown이라는 경량 다운샘플링 작업을 도입하여 모델 복잡성을 줄이는 동시에 평균 검출 성능을 개선합니다. 또한, YOLOv11의 C2PSA 모듈을 Triplet Attention 메커니즘으로 개선한 C2PSA-TriAtt 모듈을 도입하였습니다.

- **Technical Details**: 제안한 알고리즘은 Inner-Focaler IoU 손실 함수(CHIou Loss)를 기존 CIoU 손실 함수 대신 사용하여, 더욱 효과적인 경량 모델을 구현합니다. 이 구조는 검출 성능을 향상시키고, 모델 크기를 38%, 매개변수 수를 41%, 계산 비용을 40%까지 줄여줍니다. 이러한 조정으로 이미지당 평균 검출 시간이 1초 단축되었습니다.

- **Performance Highlights**: 검출 작업에서 모델은 99.10%의 높은 검출 정확도를 달성하였으며, 이는 산업 엣지 모바일 디바이스에 적합한 배포가 가능함을 의미합니다. 개선된 모델은 증가된 검출 속도와 정확성을 보이며, 석탄 처리 및 자원 효율적인 활용에 긍정적인 기여를 합니다.



### VFX Creator: Animated Visual Effect Generation with Controllable Diffusion Transformer (https://arxiv.org/abs/2502.05979)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 Open-VFX를 소개하며, 이는 15개 다양한 효과 카테고리를 포함하는 고품질 VFX 비디오 데이터셋입니다. 이 데이터셋은 사용자가 쉽게 접근할 수 있는 텍스트 설명과 정적 참조 이미지를 바탕으로 하여 동적 효과를 생성합니다. 또한, VFX Creator라는 새로운 제어 가능한 VFX Generation 프레임워크를 제안하여, 효과적인 영상 생성을 위한 혁신적인 접근법을 제시합니다.

- **Technical Details**: VFX Creator는 Video Diffusion Transformer를 기반으로 하며 공간적 및 시간적 제어를 위한 LoRA 어댑터를 통합합니다. 이 모델은 최소한의 훈련 비디오로 고품질 영상을 생성할 수 있으며, 비디오 마스크 시퀀스를 조건으로 사용하여 인스턴스 수준의 공간 조작을 가능하게 합니다. 또한, 효과의 타이밍과 페이스를 정밀하게 제어하기 위해 텍스트 인코더와 함께 동작의 시작 및 종료 타임스탬프를 확장하여 통합합니다.

- **Performance Highlights**: Open-VFX 테스트 세트를 통한 광범위한 실험 결과, 본 시스템의 동적이며 사실적인 효과 생성 능력이 기존 방법들을 초월하는 것으로 나타났습니다. 제안된 시스템은 공간적 및 시간적 제어 능력에서 최신 기술의 성능을 달성하였습니다. 우리는 생성된 효과의 시간적 제어의 정확도를 평가하기 위한 새로운 메트릭도 도입하였습니다.



### Revisiting Gradient-based Uncertainty for Monocular Depth Estimation (https://arxiv.org/abs/2502.05964)
Comments:
          Accepted to TPAMI

- **What's New**: 이 논문은 깊이 추정 모델에서 픽셀 단위의 불확실성 평가를 위한 새로운 접근법을 제시합니다. 기존의 훈련 없이도 추정된 깊이와 참조 깊이 간의 일관성을 기반으로 한 보조 손실 함수를 도입하여 기울기를 추출합니다. 이를 통해 이미 훈련된 모델의 재훈련 없이 불확실성을 측정할 수 있는 사후( пост hoc) 방법으로, 안전-critical 응용 분야에 적합합니다.

- **Technical Details**: 이 방법은 기울기를 추출하여 불확실성을 평가하며, 이는 단일층 또는 다중 디코더 층에서 계산됩니다. 또한, 이미지 혹은 피처 공간을 증강하여 생성된 참조 깊이를 사용하여 기울기를 통해 제시된 불확실성을 평가합니다. 두 표준 깊이 추정 벤치마크인 KITTI와 NYU에서 검증된 결과, 이 방법이 기존 접근법보다 우수한 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: 제안된 기법은 기존 모델 훈련 없이도 다양하고 복잡한 깊이 추정 시나리오에서 안정적으로 불확실성을 평가할 수 있으며, 특히 모노컬 시퀀스로 훈련된 모델에서 더 두드러진 성능 개선을 나타냅니다. 연구진은 코드와 모델을 공개하여 더욱 많은 연구자들이 이 방법을 활용할 수 있도록 합니다.



### Acceleration Multiple Heads Decoding for LLM via Dynamic Tree Attention (https://arxiv.org/abs/2502.05947)
- **What's New**: 이번 논문에서는 다중 헤드 디코딩(multiple heads decoding) 원리를 적용하여 LLM의 추론 속도를 향상시키기 위해 고정 트리 주의(tree attention) 대신 동적 트리 주의(dynamic tree attention)를 도입합니다. 새로운 방법은 MEDUSA의 맥락에서 적용되며, 간단하고 저 복잡도의 전략으로 후보 집합을 생성하고 동적 트리 구조를 구성합니다. 초기 실험 결과는 제안된 방식이 LLM의 다중 헤드 디코딩 효율성을 증가시키며 생성 품질을 유지하는 것을 보여줍니다.

- **Technical Details**: 제안된 방법은 먼저 후보를 동적으로 생성한 후, 이 후보들에 따라 동적 트리 주의의 버퍼를 준비하는 방식으로 진행됩니다. 후보를 생성하기 위해 각 MARDA 스탭에서 마진 분포의 카르테시안 제품(Cartesian product)을 사용해 근사값을 계산하며, 생성된 후보들은 동적 트리 구조를 기반으로 구성됩니다. 이렇게 생성된 후보들은 우선순위 큐를 통해 가장 높은 확률을 가진 상위 n명 후보로 선택되어 O(Knm log n) 복잡도로 처리됩니다.

- **Performance Highlights**: 제안한 방법은 MT-Bench를 사용하여 추론당 토큰 수를 기준으로 평가되며, MEDUSA-1 및 MEDUSA-2에서 디코딩 효율성을 향상시키면서 생성 품질도 유지됩니다. 동적 트리 구조는 고정 트리 구조와 공통된 부분을 공유하면서도 맥락 의존성에 적응할 수 있는 장점을 가지고 있습니다. 비록 속도 면에서는 MEDUSA보다 약 10% 느리지만, 더 나은 디코딩 효율성을 제공합니다.



### ClinKD: Cross-Modal Clinic Knowledge Distiller For Multi-Task Medical Images (https://arxiv.org/abs/2502.05928)
- **What's New**: 이번 논문에서는 Med-VQA(의료 시각 질문 답변) 분야에서의 새로운 모델인 ClinKD를 소개합니다. ClinKD는 모델의 position encoding을 개선하고, 다양한 훈련 과정을 통합하여 이미지 및 양식 변화에 대한 모델 인식을 증진시키는 방법을 적용했습니다. 이로써 의료 전문 인력이 병리학적 진단을 더욱 효과적으로 수행할 수 있도록 지원할 수 있습니다.

- **Technical Details**: ClinKD는 Med-CLIP Guided Rotary Position Embedding을 활용하여 이미지의 세부 위치 정보를 더 잘 포착할 수 있도록 설계되었습니다. 또한, Pseudo-Augmented Medical Distillation과 Reflective Correction Training을 통해 데이터의 부족 문제를 해결하고, 훈련 중 모델의 robust성을 강화하여 일반화 성능을 높였습니다. 이러한 접근 방식은 의료 영상의 특수성을 고려하여 모델의 성능을 최적화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: ClinKD는 Med-GRIT-270k 데이터셋에서 기존의 평가 프로토콜에 따라 새로운 최신 성능을 달성하였습니다. 특히, Visual Grounding(VG) 작업에서 뛰어난 성능을 발휘하며, 종합적 데이터 활용을 극대화하여 진단의 일관성과 신뢰성을 높이는 데 기여하고 있습니다. 이를 통해 의료 분야에서 VQA 시스템의 적용 가능성을 한층 확대할 수 있는 기반을 마련하고 있습니다.



### Multi-Branch Collaborative Learning Network for Video Quality Assessment in Industrial Video Search (https://arxiv.org/abs/2502.05924)
Comments:
          KDD 2025 ADS

- **What's New**: 이번 논문은 산업 비디오 검색 시스템을 위한 Multi-Branch Collaborative Network (MBCN)을 소개합니다. MBCN은 비디오 품질 문제를 해결하기 위해 네 개의 분기를 가지고 있으며, 각 분기는 비디오의 시각적, 텍스트 기반, 및 AI 생성 비디오와 관련된 품질 문제를 다룹니다. 이 연구는 저품질 비디오의 특정 특성을 체계적으로 분석하고 MBCN을 통해 이를 해결하려는 시도를 보여줍니다.

- **Technical Details**: MBCN은 비디오 및 텍스트 품질 평가를 위해 다중 모달 인코더와 네 개의 평가 분기를 사용하는 아키텍처를 가지고 있습니다. 각 분기는 비디오-텍스트 매칭, 프레임 일관성, 프레임 품질, 및 텍스트 품질 평가를 담당합니다. Squeeze-and-Excitation 기법을 사용하여 여러 가지 품질 문제를 동적으로 해결하고, 최적화 목표를 통해 출력 점수의 안정성과 합리성을 확보합니다.

- **Performance Highlights**: 광범위한 오프라인 및 온라인 실험 결과, MBCN은 비디오 품질 문제를 효과적으로 식별하며 검색 시스템의 랭킹 성능을 크게 향상시킵니다. 특히 저품질 AI 생성 비디오의 인식 정확도가 기존 모델에 비해 현저히 개선되었습니다. 이번 연구는 비디오 품질 평가 분야에서의 새로운 통찰력을 제공하며, MBCN의 설계가 모든 네 개의 평가 분기가 최종 결과에 긍정적인 기여를 한다는 점을 강조합니다.



### QP-SNN: Quantized and Pruned Spiking Neural Networks (https://arxiv.org/abs/2502.05905)
Comments:
          26 pages, 17 figures, Published as a conference paper at ICLR 2025

- **What's New**: 이 논문은 리소스가 제한된 환경에서의 효과적인 Spiking Neural Networks (SNN)의 배치를 목표로 하는 하드웨어 친화적이고 경량화된 접근법을 제안합니다. 특히, 균일 양자화(uniform quantization)와 구조적 가지치기(structured pruning)를 통합한 QP-SNN 베이스라인 모델을 개발하였습니다. 이 모델은 높은 성능을 보장하면서도 저장 공간과 계산 비용을 대폭 줄일 수 있습니다.

- **Technical Details**: QP-SNN의 기존 베이스라인은 일반적인 균일 양자화를 통해 모델의 표현 능력에 제한을 받습니다. 이를 개선하기 위해, 가중치 재조정 전략(ReScaW)을 도입하여 비트 폭을 효율적으로 활용하는 방안을 제안하였습니다. 또한, 시공간 스파이크 활동의 특이값(singular value)을 기반으로 한 새로운 구조적 가지치기 기준을 통해 불필요한 커널을 보다 정확하게 제거할 수 있는 방안을 개발했습니다.

- **Performance Highlights**: 제안된 QP-SNN은 새로운 양자화 및 가지치기 방법을 통합하여 기존의 SNN보다 뛰어난 성능을 보여줍니다. 실험 결과, QP-SNN은 우수한 정확성과 작은 모델 크기를 달성하였으며, 이는 엣지 인텔리전스 컴퓨팅의 발전 가능성을 보여줍니다. 이를 통해 리소스가 제한된 환경에서도 SNN을 효과적으로 배치할 수 있는 잠재력을 입증합니다.



### Fast Omni-Directional Image Super-Resolution: Adapting the Implicit Image Function with Pixel and Semantic-Wise Spherical Geometric Priors (https://arxiv.org/abs/2502.05902)
Comments:
          9 pages, 4 figures, AAAI 2025

- **What's New**: 본 논문은 Omni-Directional Image (ODI) 초해상도(Super-Resolution, SR) 문제에 대해 'FAOR'이라는 새로운 모델을 제안합니다. FAOR는 구형 기하학적 사전(geometric priors)을 활용하여 ERP 이미지 도메인으로의 변환을 수행하며, 이 과정에서 정보의 손실을 최소화합니다. 기존의 복잡한 구형 변환(convolution)이나 중첩 재투영(polyhedron reprojection) 방식과 비교하여 처리 절차가 간소화되고 추론 속도(speed)가 크게 향상됩니다.

- **Technical Details**: 제안하는 FAOR 모델은 두 가지 주요 단계인 잠재 표현(latent representation)과 이미지 재구성(image reconstruction)에서 구형 기하적 사전을 통합합니다. 잠재 표현 단계에서는 픽셀 수준 및 의미론적(s Semantic) 기준으로 구-평면 왜곡 지도를 사용하여 아핀 변환(affine transformation)을 적용합니다. 또한 이미지 재구성 단계에서는 편향되지 않은 구형 위치를 활용하는 구면 지오데식 기반 다시 샘플링 기능을 도입하여 성능을 극대화합니다.

- **Performance Highlights**: FAOR 모델은 기존의 최첨단 ODI-SR 모델보다 훨씬 빠른 추론 속도를 기록하며, 여러 실험 결과와 잔차 분석(ablation studies)을 통해 효과성을 입증하였습니다. 이 모델은 구형 특성을 포착할 수 있도록 수정된 아키텍처를 제공하며, 성능과 속도를 동시에 고려한 첫 번째 접근을 특징으로 합니다. 이러한 성과는 ODIs의 실용적 구현에 크게 기여할 것으로 기대됩니다.



### Beyond Fine-Tuning: A Systematic Study of Sampling Techniques in Personalized Image Generation (https://arxiv.org/abs/2502.05895)
Comments:
          The first two authors contributed equally

- **What's New**: 이번 연구는 개인화된 텍스트-이미지 생성에서 수업(superclass) 경로가 샘플링(strategy) 과정에 미치는 영향을 체계적으로 분석합니다. 현재의 접근 방식은 일반적으로 고정된 파인튜닝(configuration)과 연결되어 있어 샘플링의 독립적 영향을 연구하기 어렵습니다. 이에 연구진은 샘플링 전략과 수업 경로에 대한 명확한 비교를 수행하여 개인화된 생성 프로세스에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구의 핵심은 Stable Diffusion 모델을 기반으로 하여 다양한 샘플링 기법을 분석하는 것입니다. 또한, Superclass trajectory와 같은 개념과 속성을 조합한 샘플링 기법인 Switching, Mixed, Masked 샘플링 기법을 포함하여 여러 혼합 변형을 연구합니다. 연구에서는 각 방법에 대한 하이퍼파라미터를 조정하고, 최적의 성능을 위해 필수적인 요소만을 선택함으로써 효율적으로 샘플링 방법을 최적화합니다.

- **Performance Highlights**: 연구팀은 파인튜닝없이도 다양한 샘플링 전략을 평가하는 방법을 개발하였으며, 이를 통해 모델 사용의 유연성을 증대시키고 특정 생성 작업에 가장 적합한 샘플링 방법 선택을 돕는 프레임워크를 제안합니다. 연구에서 제시한 프레임워크는 개념 보존, 프롬프트 적합, 계산 자원의 효율성을 체계적으로 최적화하여 특정 시나리오에 대한 최상의 샘플링 방법을 식별할 수 있게 합니다.



### MMGDreamer: Mixed-Modality Graph for Geometry-Controllable 3D Indoor Scene Generation (https://arxiv.org/abs/2502.05874)
Comments:
          Accepted by AAAI 2025 Main Track

- **What's New**: 본 논문에서는 MMGDreamer라는 이중 분기(diffusion model)를 제안하여 기존의 텍스트 기반 입력의 한계를 극복하고, 보다 유연한 사용자 입력을 수용할 수 있는 혼합 모달리티 그래프(Mixed-Modality Graph)를 도입합니다. 이 모델은 객체 노드가 텍스트와 시각 모달리티를 통합할 수 있도록 구성되어 있으며, 이를 통해 생성된 장면의 객체 기하학적 요소에 대한 정밀한 제어가 가능합니다. 또한, 시각적 향상 모듈과 관계 예측기를 포함하여 더욱 정확하고 일관된 장면 레이아웃 생성을 지원합니다.

- **Technical Details**: MMGDreamer 모델은 세 가지 형태로 표현될 수 있는 노드로 구성된 혼합 모달리티 그래프(MMG)를 사용합니다: 텍스트, 이미지, 또는 두 가지의 조합. 이 그래프 구조는 사용자 입력에 기반하여 노드 간의 관계를 선택적으로 제공하거나 생략할 수 있습니다. 특정 객체의 기하학적 요소에 대한 세심한 제어를 가능하게 하며, 고유의 시각적 표현을 구축하기 위한 텍스트 임베딩을 활용하여 노드의 시각적 충실도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, MMGDreamer는 SG-FRONT 데이터셋에서 기존 기술보다 훨씬 높은 기하학적 제어력과 충실도를 발휘하였습니다. 또한, 최신 장면 합성(scene synthesis) 성능을 달성하여 현존하는 방법들보다 현저한 성능 향상을 보였습니다. 이러한 성능 향상은 복잡한 장면 생성을 위한 보다 효율적인 도구로서의 잠재력을 보여줍니다.



### HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition (https://arxiv.org/abs/2502.05869)
- **What's New**: 이번 논문에서는 스켈레톤 기반 인간 행동 인식(HAR)을 위해 설계된 새로운 하이퍼볼릭 리니어 어텐션 트랜스포머(HyLiFormer)를 제안합니다. 기존의 리니어 어텐션 메커니즘이 구조적으로 풍부한 스켈레톤 데이터를 잘 다루지 못하는 문제를 해결하기 위해, 하이퍼볼릭 기하학을 활용한 새로운 접근법을 사용합니다. 이는 스켈레톤 데이터를 하이퍼볼릭 공간으로 변환하고, 효율적인 장거리 종속성 모델링을 가능하게 하여 계산 복잡성을 줄입니다.

- **Technical Details**: HyLiFormer는 하이퍼볼릭 변환과 곡률(HTC) 모듈을 통해 스켈레톤 데이터를 효과적으로 하이퍼볼릭 공간으로 맵핑합니다. 이후, 하이퍼볼릭 리니어 어텐션(HLA) 모듈을 사용하여 효율적인 장거리 종속성을 모델링합니다. 이러한 구조적 흐름은 스켈레톤 데이터의 비유클리드 기하학적 특성을 보다 잘 반영하며, 계산적으로 효율적인 토대를 제공합니다.

- **Performance Highlights**: HyLiFormer는 NTU RGB+D 및 NTU RGB+D 120 데이터셋에 대한 이론적 분석과 실험을 통해, 모델 정확성을 유지하면서도 계산 복잡성을 크게 감소시킴을 입증합니다. 이 모델은 계산 효율성이 중요한 실제 애플리케이션에서의 활용 가능성을 넓히면서도, 정확도를 유지하여 스켈레톤 기반 행동 인식 분야에서 효과적인 솔루션으로 자리잡을 것으로 기대됩니다.



### SphereFusion: Efficient Panorama Depth Estimation via Gated Fusion (https://arxiv.org/abs/2502.05859)
Comments:
          3DV 2025

- **What's New**: SphereFusion은 자동 운전 및 로봇 감지와 같은 응용 분야에서 패노라마 깊이를 추정하는 새로운 방법입니다. 기존의 방법들은 왜곡(distortion) 및 불연속성(discontinuity) 문제로 어려움을 겪었으나, SphereFusion은 다양한 프로젝션 방법의 장점을 결합하여 이 문제를 해결합니다. 이 프레임워크는 equirectangular와 spherical 프로젝션을 통해 두 가지 유형의 특징을 추출하고, 최종적으로 신뢰할 수 있는 특징을 융합하여 깊이를 추정합니다.

- **Technical Details**: SphereFusion은 2D 이미지 컨볼루션과 메시(mesh) 작업을 사용하여 패노라마 이미지에서 특징을 추출합니다. 이 특징은 spherical 도메인으로 프로젝션되어 Gate Fuse 모듈을 통해 융합한 뒤, 최종적으로 패노라마 깊이를 추정합니다. 또한, 캐시 전략을 사용하여 메시 작업의 효율성을 높이고 있으며, 전반적으로 경량 인코더를 구현하여 성능을 개선하였습니다.

- **Performance Highlights**: SphereFusion은 512×1024 해상도의 패노라마 이미지에 대해 단 17 ms의 빠른 추론 속도를 달성하였으며, 세 가지 공개 패노라마 데이터셋에서 경쟁력 있는 성능을 보였습니다. 실험 결과, 높은 품질의 깊이 맵과 함께 적은 노이즈의 포인트 클라우드를 생성할 수 있음을 보여주었습니다. NVIDIA RTX 3090에서 60 FPS를 달성함으로써 기존 방법들을 능가합니다.



### Acquisition through My Eyes and Steps: A Joint Predictive Agent Model in Egocentric Worlds (https://arxiv.org/abs/2502.05857)
- **What's New**: 이번 논문에서는 인간의 인지 시스템에서 영감을 받은 에이전트 모델인 EgoAgent를 제안합니다. EgoAgent는 세계를 표현하고, 미래 상태를 예측하며, 합리적인 행동을 수행하기 위해 단일 transformer 모델에서 이 세 가지 능력을 동시에 학습합니다. 기존의 유사한 연구들이 독립적으로 세 가지 모델을 학습한 반면, EgoAgent는 이 모델들을 통합하여 서로의 학습을 돕도록 설계되었습니다.

- **Technical Details**: EgoAgent는 에고센트릭(Egocentric) 세계에서의 인식, 예측 및 행동을 기능적으로 통합합니다. 이 모델은 관찰과 인간 행동의 이력 데이터를 바탕으로 고차원 feature 벡터로 매핑하여 transformer에 입력하고, 이를 통해 현재 상태, 미래 상태 및 다음 행동을 예측합니다. 특히, 이 모델은 Attention 기법을 통해 세 가지 기능 사이의 내부 관계를 자연스럽게 설정하여 상호 보완적인 학습을 가능하게 합니다.

- **Performance Highlights**: EgoAgent는 이미지 분류, 에고센트릭 미래 상태 예측 및 3D 인간 모션 예측 과제에서 광범위한 실험을 통해 우수한 성능을 입증했습니다. 예를 들어, EgoAgent는 ImageNet-100과 ImageNet-1K에서 기존의 선도적인 방법들보다 1.40% 및 1.32%의 정확도를 향상시켰으며, 에고-엑소4D(Ego-Exo4D) 미래 상태 예측 과제에서는 각각 16.28% 및 16.95%의 mAP 개선을 달성했습니다.



### Training-free Anomaly Event Detection via LLM-guided Symbolic Pattern Discovery (https://arxiv.org/abs/2502.05843)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 기존의 감독 학습 기반 접근법의 한계를 극복하기 위해 훈련 없이 사용할 수 있는 새로운 프레임워크인 VED-SR(Visual Event Detection with Symbolic Reasoning)을 제안합니다. 이 프레임워크는 Open-set object detection과 symbolic regression을 통합하여 대규모 언어 모델(LLMs)을 활용하여 효과적인 비정상 탐지를 가능하게 합니다. 제안된 방법은 훈련 데이터의 필요성을 제거하며, 약 1%의 주석 작업으로도 이해 가능한 논리적 표현을 생성할 수 있습니다.

- **Technical Details**: VED-SR는 기호 회귀(symbolic regression)와 LLM의 추론 능력을 결합하여 비정상 사건 탐지를 수행합니다. 이 접근법은 데이터 패턴을 해석할 수 있는 강력한 신호를 제공합니다. 논문에서는 또한 새로운 데이터셋을 소개하며, 다양한 비정상 시나리오를 포함하는 110,000장 이상의 주석 이미지로 구성된 대규모 개인 데이터셋과 5,000개의 샘플을 포함하는 공개 벤치마크 데이터셋을 활용합니다.

- **Performance Highlights**: 여러 벤치마크에서의 테스트 결과, VED-SR는 UCSD Ped2 데이터셋에서 98.7%의 AUROC를 달성하였으며, 이는 기존의 훈련 기반 방법들과 경쟁할 수 있는 성능입니다. 논문에서 제시하는 통계는 VED-SR이 비정상 탐지에서 효과적이며, 다양한 환경에서 일반화가 가능함을 입증합니다. 이 결과는 VED-SR의 효율성과 해석 가능성을 뒷받침하며, 훈련 데이터 없이도 높은 성능을 발휘합니다.



### Contrastive Representation Distillation via Multi-Scale Feature Decoupling (https://arxiv.org/abs/2502.05835)
- **What's New**: 본 연구에서는 knowledge distillation의 새로운 접근법으로 multi-scale feature decoupling을 도입하고 있습니다. 기존의 방법들은 주로 전역 특징(global feature)에 초점을 맞추었으나, 본 연구는 다양한 정보의 분리에 주목하여 현지(local) 특징을 개별적으로 처리하고 결합합니다. 이를 통해 학생 네트워크는 단일 배치 샘플만을 사용해도 성능 향상을 이룰 수 있습니다.

- **Technical Details**: 제안된 방법은 정리된 특징 샘플 정보를 사용하여 contrastive learning과 결합되며, 이는 복잡한 샘플 처리나 큰 메모리 버퍼를 요구하지 않습니다. multi-scale feature decoupling을 통해 학생과 교사의 네트워크에서 특징을 여러 스케일로 분리하여, 전역 및 현지 특징을 따로 처리합니다. 이 과정은 feature sample information의 풍부함을 극대화하여, 학생 네트워크의 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: CIFAR-100 및 ImageNet 데이터 세트에서의 평가 결과, 제안된 방법은 기존의 방법들보다 우수한 성능을 나타냈습니다. 특히, 일부 학생 네트워크는 교사 네트워크의 성능을 초과하는 결과를 보였으며 이는 제안된 방법의 효과성을 강조합니다. 이러한 결과는 학생 네트워크가 교사 네트워크로부터 지식을 충분히 흡수할 수 있도록 하는 데 기여하고 있음을 보여줍니다.



### Divide-and-Conquer: Tree-structured Strategy with Answer Distribution Estimator for Goal-Oriented Visual Dialogu (https://arxiv.org/abs/2502.05806)
- **What's New**: 이번 논문에서는 목표 지향적(Goal-oriented) 비주얼 대화(Visual Dialogue)에서 질문 생성의 효율을 높이기 위해 Tree-Structured Strategy with Answer Distribution Estimator (TSADE)를 제안합니다. TSADE는 매 라운드마다 현재 후보 객체의 절반을 제외하여 질문 생성을 유도합니다. 이러한 접근 방식은 기존의 무작위적 질문 생성 방법과는 달리, 대화의 질과 효율성을 명확히 향상시킵니다.

- **Technical Details**: TSADE는 Reinforcement Learning (RL) 패러다임 아래에서 작동하며, 질문 생성을 위해 Answer Distribution Estimator (ADE)를 활용합니다. ADE는 주어진 질문에 대해 현재 후보 객체의 답변 분포를 동적으로 추정하여, 어떤 객체가 목표 객체와 동일한 답변을 가질 가능성이 높은지를 판단합니다. 이 과정에서 두 가지 보상 설계를 통해 QGen이 효과적으로 질문을 생성하도록 유도합니다: 이진 보상 및 후보 최소화 보상입니다.

- **Performance Highlights**: 실험 결과, TSADE를 적용한 QGen은 기존의 여러 모델에 비해 반복 질문을 줄이며, 비주얼 대화에서 특정 목표를 향해 더 빠르고 엄밀한 질문 생성을 가능하게 함을 입증하였습니다. TSADE는 다양한 기본 모델과 협력하여 정보량이 많은 질문을 생성할 수 있게 돕고, 이는 목표 달성을 빠르게 합니다.



### MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Devic (https://arxiv.org/abs/2502.05800)
- **What's New**: 이번 연구에서는 MicroViT라는 경량 Vision Transformer 아키텍처를 소개합니다. MicroViT는 Efficient Single Head Attention (ESHA) 메커니즘을 기반으로 하여 계산 복잡성을 대폭 줄이며, 고성능을 유지하면서도 에지 디바이스에서 효과적으로 작동할 수 있도록 최적화되었습니다. 이 모델은 다단계 MetaFormer 아키텍처를 활용하여 빠른 추론 속도와 낮은 전력 소모를 달성합니다.

- **Technical Details**: MicroViT는 그룹 컨볼루션을 사용하여 특징 중복성을 최소화하면서 계산 복잡성과 전력 소모를 감소시키는 방식을 채택합니다. ESHA는 로컬 및 글로벌 공간 작업을 하나의 블록 내에서 결합하여 효과적으로 토큰 정보를 추출합니다. 이는 하이퍼파라미터 설정 및 아키텍처 설계를 통해 최적의 성능을 내도록 구성되어 있습니다.

- **Performance Highlights**: MicroViT는 ImageNet-1K 및 COCO 데이터셋에서 경량성이 뛰어나면서도 경쟁력 있는 정확도를 보여줍니다. 이는 MobileViT 시리즈 대비 3.6배 빠른 추론 속도와 40% 더 높은 효율로 에너지 소비를 줄이며, 모바일 및 에지 디바이스 같은 자원 제약 환경에서의 배포 가능성을 높입니다.



### EPBC-YOLOv8: An efficient and accurate improved YOLOv8 underwater detector based on an attention mechanism (https://arxiv.org/abs/2502.05788)
- **What's New**: 이 연구에서는 YOLOv8의 백본에 채널 및 공간 주의 메커니즘을 통합하고, FasterNeXt의 Pointwise Convolution을 활용하여 FasterPW 모델을 구축하였습니다. 또한 BiFPN에서 영감을 받은 WFPN 구조를 사용하여 크로스 스케일 연결과 강건성을 개선하였습니다. 이러한 개선된 프레임워크는 CARAFE를 통해 세밀한 특징 재조합을 수행하여 수중 이미지 저하 문제를 해결하며, URPC2019와 URPC2020 데이터셋에서 각각 76.7%와 79.0%의 mAP@0.5 점수를 기록했습니다.

- **Technical Details**: 이 논문은 YOLOv8 모델을 기반으로 여러 가지 개선 사항을 제안합니다. EMA(다중 스케일 주의 모듈)와 YOLOv8의 C2f 백본을 통합하여 다양한 스케일의 타겟에 대한 반응성을 높이고, FasterNext 모듈을 통해 변환된 FastPW 모델에서는 부분 합성곱을 Pointwise Convolution으로 교체하여 경량화 및 특징 추출 능력을 향상시켰습니다. 또한 WFPN의 크로스 스케일 연결 및 가중 특징 융합을 통해 정보 통합을 최적화하고, CARAFE로 업샘플링을 적용하여 작은 타겟 정보를 효과적으로 유지합니다.

- **Performance Highlights**: EPBC-YOLOv8 수중 객체 탐지기는 컴퓨팅 효율성과 정확성 사이의 균형을 달성합니다. URPC2019 및 URPC2020 데이터셋에서 각각 76.7%와 79.0%의 mAP@0.5 점수를 기록하여 기존 YOLOv8에 비해 각각 2.3% 및 0.7% 더 나은 성능을 보였습니다. 이러한 성능 향상은 수중 생물 탐지의 정확도를 높여, 관련 분야에서의 적용 가능성을 넓힐 것입니다.



### A 3D Multimodal Feature for Infrastructure Anomaly Detection (https://arxiv.org/abs/2502.05779)
- **What's New**: 이 연구는 기존의 구조물 검사 방법에서 발견된 작은 균열 탐지의 한계를 극복하기 위해 새로운 3D 멀티모달 기능인 3DMulti-FPFHI를 제안합니다. 이는 사용자 정의된 Fast Point Feature Histogram (FPFH)와 강도(intensity) 기능을 결합하였습니다. 이 접근 방식은 PatchCore 이상 탐지 알고리즘에 통합되어 실제 구조물에 대한 평가를 통해 그 효과를 검증하였습니다.

- **Technical Details**: 3DMulti-FPFHI는 통계 분석 및 파라메트릭 분석을 통해 평가되었으며, 실제 석조 아치교와 콘크리트 터널의 전체 모델을 사용하는 포인트 클라우드(point clouds)을 활용하여 성능을 비교했습니다. 이 기술은 특히 균열 탐지와 물 유입(water ingress)의 식별을 향상시키는 데 효과적입니다. 3D 강도(intensity) 기능은 특히 손상 탐지의 품질을 높이는 데 기여합니다.

- **Performance Highlights**: 결과적으로 3DMulti-FPFHI는 기존의 FPFH 및 최신 멀티모달 이상 탐지 기법을 능가하는 성능을 보여주었습니다. 데이터 요구사항이 적어 학습 기반 방법론에 비해 다양한 인프라 구조물의 이상 탐지 시나리오를 다룰 수 있는 잠재력이 강조되었습니다. 연구의 코드 및 관련 포인트 클라우드 데이터셋은 제공된 URL에서 접근할 수 있습니다.



### Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails (https://arxiv.org/abs/2502.05772)
- **What's New**: 이 논문에서는 MultiFaceted Attack이라는 새로운 공격 프레임워크를 제안하여 Vision-Language Large Models (VLLMs)의 Multi-Layered Defenses를 체계적으로 우회하는 방법을 소개합니다. 이 공격은 Visual Attack, Alignment Breaking Attack, Adversarial Signature의 세 가지 보완적인 측면으로 구성되어 있습니다. 연구 결과는 현재 상용 VLLMs에서 61.56%의 흑상자 공격 성공률을 달성하여, 기존 방법보다 최소 42.18% 이상의 개선을 보였습니다.

- **Technical Details**: VLLMs는 이미지와 텍스트 정보를 통합하여 두 가지 입력 형식을 모두 이해하는 작업을 수행할 수 있도록 설계되었습니다. 이 논문에서는 VLLMs의 안전 메커니즘을 우회하기 위해 Visual Attack, Alignment Breaking Attack, Adversarial Signature와 같은 세 가지 공격 전략을 제안합니다. 이러한 각 공격 전략은 서로 보완적이며 다양한 실제 시나리오에서 유연하게 활용될 수 있습니다.

- **Performance Highlights**: MultiFaceted Attack의 실험 결과는 현재 VLLMs의 안전 메커니즘에서 중요한 취약점을 드러내며, 보다 강력한 방어 전략의 필요성을 강조합니다. 이 공격 방식은 기존 SOTA 방법에 비해 더 나은 전환 성능을 보여주며, 상용 VLLMs에서 높은 효율성을 입증했습니다. 이러한 결과는 공격 메커니즘의 효과성을 나타내며, 미래의 연구에 중요한 기초 자료를 제공합니다.



### Digital Twin Buildings: 3D Modeling, GIS Integration, and Visual Descriptions Using Gaussian Splatting, ChatGPT/Deepseek, and Google Maps Platform (https://arxiv.org/abs/2502.05769)
Comments:
          -Fixed minor typo

- **What's New**: 이 논문에서는 도시 디지털 트윈(Digital Twin Building, DTB) 프레임워크를 제안하며, 이는 건물의 3D 메쉬 모델을 추출하고 클라우드 매핑 서비스와 통합하여 데이터 분석을 수행할 수 있도록 합니다. Google Maps Platform API와 같은 클라우드 매핑 플랫폼에 연결하여 실시간 교통 데이터, 공기 질 데이터 등 다양한 데이터를 분석할 수 있는 가능성을 제공합니다. 또한, Multi-Agent Large Language Models (LLM) 모듈을 통해 여러 관점에서 시각 분석을 수행하고 ChatGPT(4o) 및 Deepseek-V3/R1 모델의 성능을 평가합니다.

- **Technical Details**: 이 연구에서는 Gaussian Splatting 모델과 3D 메쉬 모델을 활용하여 건물에 대한 기초적인 지오코딩 및 매핑 정보를 검색합니다. 연구의 주된 기술적인 요소는 Gaussian Building Mesh (GBM)라는 3D 건물 메쉬 추출 파이프라인으로, 이는 구글 어스 스튜디오 및 여러 클라우드 서비스 API를 활용하여 운영됩니다. 이 파이프라인을 통해 건물의 이름, 주소 또는 지리적 좌표에서 3D 메쉬를 추출하고, 건물의 다양한 정보를 수집할 수 있습니다.

- **Performance Highlights**: 연구의 성과로는 Building의 3D 모델과 시각적 설명을 통합하여 클라우드 기반 매핑과 데이터 분석의 효율성을 높인 것입니다. 주요 성능 평가 도구로는 CLIP 점수와 perplexity를 활용하여 LLM의 이미지-키워드 추출 신뢰도를 평가하며, 실제 이미지에 대한 주석이 없는 상황에서도 키워드를 효과적으로 추출할 수 있음을 보여주었습니다. 이러한 기술적 접근은 향후 도시 계획 및 인프라 관리에 큰 기여를 할 것으로 기대됩니다.



### 3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly (https://arxiv.org/abs/2502.05761)
Comments:
          Accept by AAAI2025, github: this https URL

- **What's New**: 논문에서는 3C 제품의 품질 관리에 전념하는 새로운 대규모 이상 탐지 데이터셋인 3CAD를 제안합니다. 이 데이터셋은 실제 3C 생산 라인에서 유도된 것으로, 8가지 제조 부품 유형에 대한 27,039개의 고해상도 이미지를 포함하고 있습니다. 3CAD는 다양한 크기와 유형의 이상을 포함하며, 여러 이상 영역을 포함할 수 있는 특성을 가지고 있습니다. 이는 기존 데이터셋들의 한계를 뛰어넘어 커뮤니티의 탐색과 개발을 촉진할 것입니다.

- **Technical Details**: 이 논문은 Coarse-to-Fine detection paradigm with Recovery Guidance(이하 CFRG)라는 새로운 비지도 이상 탐지 프레임워크를 소개합니다. CFRG는 이질적인 증류 모델을 사용하여 초기 위치를 대략적으로 확인한 후, 세분화 모델을 통해 정밀 위치를 찾아냅니다. 또한, 정상 패턴을 더 잘 포착하기 위해 복구 기능을 도입하여 정상 이미지의 기본 패턴을 캡처합니다. 이는 기존의 비지도 기법들이 정확한 결함 지역을 찾는데 어려움을 겪는 문제를 해결하는 데 도움을 줄 것입니다.

- **Performance Highlights**: 3CAD 데이터셋과 기존 일반적인 이상 탐지 방법들을 비교한 결과, 기존 방법들이 인기 있는 데이터셋에서는 효과적으로 작동하지만, 3CAD의 정밀 결함 지역 위치에 있어서 어려움을 겪는 것으로 나타났습니다. 이는 이상 탐지 분야의 발전을 위해 상당한 개선의 가능성을 시사합니다. 논문에서는 이러한 3CAD 데이터셋의 도전 과제를 해결하기 위한 CFRG 방법론의 성능을 보고하며, 성과가 강력한 경쟁력을 보여주는 것을 강조하고 있습니다.



### Exploring Visual Embedding Spaces Induced by Vision Transformers for Online Auto Parts Marketplaces (https://arxiv.org/abs/2502.05756)
Comments:
          AAAI 2025 Workshop on AI for Social Impact: Bridging Innovations in Finance, Social Media, and Crime Prevention

- **What's New**: 이 연구는 Vision Transformer (ViT) 모델의 시각 임베딩(Time Embeddings) 생성 능력을 온라인 마켓플레이스에서 수집한 자동차 부품 이미지에 대해 조사합니다. 단일 모달리티 데이터를 중심으로 작업하여 불법 활동을 나타내는 패턴을 감지하는 ViT의 잠재력을 평가합니다. 이 연구는 ViT의 강점을 강조하는 동시에 단일 모달 접근 방식의 한계도 드러내어 향후 불법 행위를 탐지하는 방법론 발전의 기초를 제공합니다.

- **Technical Details**: 이 연구는 ViT-Base 모델을 사용해 대규모 자동차 부품 이미지 데이터셋에 대해 fine-tuning을 수행하고, UMAP(Uniform Manifold Approximation and Projection)과 K-Means clustering을 통해 이미지 내 유사 항목을 그룹화합니다. 이미지에서 추출된 고차원 임베딩을 시각화하여 클러스터의 구성 및 특성을 분석하는 것으로, 일부 클러스터가 겹치고 이상치가 존재하는 문제를 언급합니다. 이는 맥락 정보(텍스트 데이터)가 결여된 경우 ViT의 장점과 한계를 이해하는 데 중요합니다.

- **Performance Highlights**: ViT는 유사한 시각적 항목을 그룹화하는 데 효과적이지만, 단일 모달 접근 방식의 한계로 인해 겹치는 클러스터와 이상치를 처리하는 데 어려움이 있습니다. 이는 다중 모달 접근 방식에 비해 단일 모달 모델의 효용성을 평가하는 데 기여합니다. 연구는 향후 발전 방안을 제시하며 비주얼 정보만을 사용할 수밖에 없는 상황에서도 ViT의 역할을 강조합니다.



### UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Contro (https://arxiv.org/abs/2502.05749)
- **What's New**: UniDB라는 새로운 통합 확장을 통해 기존의 확산 다리 모델을 일반화하여 제안합니다. 이 모델은 Stochastic Optimal Control(SOC) 원칙을 통해 최적 제어기를 계산하고, 이를 통해 기존의 Doob의 h-transform 접근법의 한계를 극복합니다. 특히, UniDB는 SOC 비용 함수의 단말 패널티 계수를 조정하여 이미지 세부 사항을 향상시키는데 도움을 줍니다.

- **Technical Details**: UniDB는 SOC 기반 최적화 문제를 통해 확산 다리 모델을 구성하고, 최적 제어기에 대한 폐쇄 형태의 해를 도출합니다. 이 과정에서 Doob의 h-transform이 SOC 비용 함수에서 단말 패널티 계수가 무한대로 접근할 때의 특수 사례로 나타난다는 점을 명확히 합니다. 또한, 기존 모델들에 비해 성능 향상을 위해 코드 수정을 최소화하며 쉽게 구현할 수 있도록 설계되었습니다.

- **Performance Highlights**: UniDB는 다양한 이미지 복원 작업에서 최신 기술 수준의 결과를 달성하였으며, 이에는 초해상도(DIV2K), 인페인팅(CelebA-HQ), 비 오는 날의 이미지 복원(Rain100H)이 포함됩니다. 이는 제안한 프레임워크의 뛰어난 이미지 품질과 다양한 시나리오에서의 적응성을 보여줍니다.



### Linear Attention Modeling for Learned Image Compression (https://arxiv.org/abs/2502.05741)
- **What's New**: 본 논문에서는 연습된 이미지 압축(learned image compression) 분야에서 새로운 접근 방식으로 LALIC을 제안합니다. 이 모델은 Bi-RWKV 블록을 사용하여 공간과 채널 혼합 모듈을 통해 더 compact한 feature 추출을 달성하며, 2차원 잠재 표현에 적합한 Omni-Shift 모듈도 포함합니다. 또한, 인접 feature 간의 상관관계를 효과적으로 모델링하기 위해 RWKV 기반의 Spatial-Channel ConTeXt 모델(RWKV-SCCTX)을 도입하였습니다.

- **Technical Details**: LALIC은 Bi-RWKV 블록을 기반으로 하며, 공간 자원과 채널 자원을 이용해 feature를 추출합니다. Omni-Shift 모듈을 적용하여 2차원 잠재 표현에 적합하게 조정하며, Bi-RWKV를 통해 인접한 feature 간의 상관관계를 모델링합니다. 이러한 구조는 선형(attention) 메커니즘을 통해 계산 복잡도를 줄이면서도 우수한 성능을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 LALIC 모델은 Kodak, Tecnick 및 CLIC Professional 검증 데이터셋에서 VTM-9.1보다 각각 -14.84%, -15.20%, -17.32% BD-rate에서 성능을 초과하며 경쟁력 있는 RD 성능을 달성하였습니다. 본 연구는 선형주의 모델이 효율적인 이미지 압축을 위한 첫 번째 접근 방식임을 확인하였습니다.



### Performance Analysis of Traditional VQA Models Under Limited Computational Resources (https://arxiv.org/abs/2502.05738)
Comments:
          6 pages, 1 figure, 5 tabels, the paper has been accepted by the PRML'25 conference

- **What's New**: 본 논문은 제한된 계산 자원 하에서 Visual Question Answering (VQA) 모델의 성능을 분석하고, Bidirectional GRU (BidGRU)와 같은 전통적 모델에 초점을 맞춰 성능 향상 방안을 모색합니다. 특히 숫자 및 카운팅 질문을 효율적으로 처리할 수 있는 전략을 제시합니다. 추가적으로 사전 훈련된 Convolutional Neural Networks (CNN)와의 결합을 통해 계산 자원의 제약을 최소화하고, 모델의 정확성을 높이는 방법을 제시합니다.

- **Technical Details**: 제안된 VQA 모델은 질문 특성 추출, 이미지 특성 추출, 주의 기제(Attention Mechanism), 특성 융합 및 분류 모듈로 구성되어 있습니다. 모델은 BidGRU를 사용해 질문을 인코딩하고, 사전 훈련된 CNN을 통해 이미지 특징을 추출합니다. 또한, 질문에 따라 중요 시각 정보를 강조하도록 Attention Maps를 생성하고, 카운팅 질문을 처리하기 위해 카운팅 모듈을 통합했습니다.

- **Performance Highlights**: 실험 결과에 따르면, BidGRU 모델은 특정 임베딩 차원 및 어휘 크기 조합을 활용하여 모든 메트릭에서 최고 성능을 달성했습니다. 특히, 숫자 및 카운팅 질문에 대한 우수한 이해력과 추론 능력을 보여주었습니다. 이 연구는 제한된 계산 능력의 환경에서도 효율적인 VQA 모델 개발을 위한 귀중한 통찰을 제공합니다.



### SSDD-GAN: Single-Step Denoising Diffusion GAN for Cochlear Implant Surgical Scene Completion (https://arxiv.org/abs/2502.05710)
- **What's New**: 이 논문에서는 자기 감독 학습(self-supervised learning) 방법을 활용하여 외과 수술 장면을 완전하게 복원하는 새로운 접근 방식을 제안합니다. 기존 연구에서는 GANs와 DDPMs를 활용하여 복원 품질을 개선했으나, 외과 환경을 복원하는 데 한계가 있었습니다. 새로운 방법인 Single-Step Denoising Diffusion-GAN(SSDD-GAN)을 통해 구조적 유사도(Structural Similarity)가 6% 향상되었습니다.

- **Technical Details**: 제안된 SSDD-GAN은 실제 외과 데이터셋을 기반으로 훈련되었습니다. 이 과정에서 GAN의 대립 최적화(adversarial optimization)와 확산 모델(diffusion model)의 장점을 결합하여 외과 장면을 효과적으로 복원합니다. 훈련된 모델은 제로샷(zero-shot) 접근법을 활용하여 합성된 포스트마스토이데크 데이터셋에 직접 적용됩니다.

- **Performance Highlights**: 새로운 접근 방식은 합성된 포스트마스토이데크 데이터셋을 사용하여 현실적이고 완전한 외과 장면을 생성할 수 있도록 합니다. 이 방법은 별도의 실제 레이블 없이도 고품질 복원을 가능하게 하여 외과적 수술 전 계획 및 수술 내비게이션에서의 활용도를 높입니다. 기존의 연구에서 해결되지 않은 주요 한계를 극복하며, 수술 마이크로스코피 장면 복원의 새로운 경로를 제공합니다.



### The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions (https://arxiv.org/abs/2502.05673)
- **What's New**: 이번 연구는 대규모 데이터 세트를 효율적으로 준비하는 방법으로서 Dataset Distillation(DD)의 최근 발전을 종합적으로 조망합니다. 기존의 연구가 2023년 이전의 발전에 초점을 맞춘 반면, 본 논문은 ImageNet-1K와 ImageNet-21K와 같은 대규모 데이터 세트에 대한 확장성을 강조합니다. 특히, SRe2L 프레임워크 및 소프트 라벨 전략과 같은 혁신적인 기술을 통해 모델의 정확성을 크게 높일 수 있는 방법을 제시합니다.

- **Technical Details**: Dataset Distillation은 대형 학습 데이터 세트를 효과적인 합성 데이터 세트로 응축하는 과정을 포함합니다. 최근 연구에서는 확률 분포 매칭, 그래디언트 매칭, 경로 매칭 등 여러 접근 방식으로 성능을 극대화하는 방법을 탐구하고 있습니다. 예를 들어, 그래디언트 매칭 프레임워크에서는 합성 데이터에서 학습된 모델이 원본 데이터로 학습된 모델의 최적화 경로와 비슷하게 안내될 수 있도록 합니다.

- **Performance Highlights**: 본 논문은 다양한 벤치마크를 통해 다양한 방법론의 성능 비교를 제공합니다. 그래디언트 매칭에 기반한 접근 방식이 효과적인 성과를 보여주었으며, 과거 2년 간의 최신 발전 사항을 강조합니다. 또한, DD는 비디오, 오디오, 다중 모드 학습, 의료 이미징 등 다양한 분야에서 활용 가능성을 높이며, 이러한 응용 프로그램에 대한 도전 과제를 다루고 있습니다.



### Rigid Body Adversarial Attacks (https://arxiv.org/abs/2502.05669)
Comments:
          17 pages, 14 figures, 3DV 2025

- **What's New**: 이 논문은 경직체 시뮬레이터에서의 대적 공격(adversarial attack)을 개발한다. 경직체가 아닌 물질이 존재하고, 이로 인해 경직체 시뮬레이터와 변형 가능한 시뮬레이터 간의 경로에서 큰 차이가 발생할 수 있음을 보인다. 저자들은 동일한 충돌 기하학(collision geometry)과 질량 중심(moment of mass)을 가지며, 경직체 시뮬레이션에서 동일하게 동작하지만, 더 정밀한 변형 시뮬레이션에서는 최대한 다르게 행동하도록 설계된 대적 객체를 제안한다.

- **Technical Details**: 이 연구는 최적화 기법을 사용하여 물리적으로 타당한 재료로 대적 객체를 구성한다. 저자들은 비용 함수(cost function)를 정의한 후 자유도(degrees of freedom)를 재료 분포(material distribution)와 내부 기하학(internal geometry)으로 설정하고, 음수 경계법(adjoints method)을 통해 기울기를 효율적으로 계산한다. 이러한 접근 방식을 기반으로, Polyfem과 같은 상용 시뮬레이터에서 여러 대적 객체를 생성하고 그 결과를 비교하여 방법의 유효성을 입증한다.

- **Performance Highlights**: 논문의 결과는 로봇 계획 및 제어 도구가 경직체 시뮬레이터를 사용할 때 잠재적으로 위험한 모델 가정을 만든다는 것을 보여준다. 저자들은 이러한 대적 객체가 물리적 시뮬레이션 및 로봇 계획과 제어 방법의 검증 및 학습에 개선된 결과를 가져올 수 있다고 주장한다. 따라서 이 연구는 안전-critical한 응용 분야에서의 물리적 모델의 정밀성을 높이는 중요한 단서를 제공한다.



### Evaluating Vision-Language Models for Emotion Recognition (https://arxiv.org/abs/2502.05660)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 이 연구에서는 감정 인식 분야에서 대표적인 대형 비전-언어 모델(VLMs)의 포괄적인 평가를 처음으로 제시합니다. 새로운 벤치마크인 Evoked Emotion benchmark (EvE)를 개발하고, VLM들이 제시된 이미지와 텍스트 프롬프트를 기반으로 감정을 얼마나 잘 인식하는지 측정합니다. 이를 통해 VLM이 감정 인식에서 직면하는 주요 오류와 모델의 민감성을 분석합니다.

- **Technical Details**: 본 연구는 여러 감정 관련 데이터셋을 활용하여 VLM의 감정 인식 성능을 평가합니다. 연구에서는 특히 Evoked Emotion Recognition이라는 특정 작업에 초점을 맞추고, 모델의 정확성과 강건성을 평가하기 위해 8가지 다양한 실험 설정을 설계했습니다. 여기에는 프롬프트에서 감정 레이블의 순서 변경, 개방형 어휘 분류, 감정적 관점 채택 및 자기 추론 메커니즘 사용이 포함됩니다.

- **Performance Highlights**: 결과적으로 VLM은 현재 이미지에서 유발된 감정을 예측하는 데 부족함을 보이며, 프롬프트에서 감정 레이블의 순서에 민감하게 반응합니다. 또한, 특정 모델은 자기 추론 메커니즘을 활용할 때 성능이 개선되지만, 감정적 페르소나를 채택하는 방향이 성능에 부정적인 영향을 미친다는 것을 발견했습니다. 인간 평가 연구를 통해 잘못된 예측의 원인을 모델의 능력뿐만 아니라 사용된 데이터와 작업의 난이도에 기인한다는 점을 명확히 했습니다.



### An inpainting approach to manipulate asymmetry in pre-operative breast images (https://arxiv.org/abs/2502.05652)
Comments:
          Preprint

- **What's New**: 이 연구는 유방암 치료의 미적 결과를 예측하기 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 유두(nipple)와 유방(breast) 형태를 현실적으로 변형하는 데 한계가 있었으나, 본 연구의 inpainting 기법을 통해 더욱 사실적인 이미지를 생성할 수 있게 되었습니다. 또한, 자동화된 영역 주석(annotation)을 통해 수작업으로 경계를 그릴 필요 없이 유방 이미지를 조작할 수 있는 기능까지 추가되었습니다.

- **Technical Details**: 본 연구에서는 유방 이미지를 조작하기 위한 inpainting 네트워크를 개발하였으며, 대칭 기반의 새로운 전략이 적용되었습니다. 특히, 자동 인식이 가능한 invertible networks를 사용하여 유두와 유방 경계를 동시에 탐지하고 inpainting 하는 방식을 도입했습니다. 실험에서는 여러 모델 아키텍처를 비교하여, 제안된 모델이 실제 환자의 유방 비대칭을 효과적으로 재현할 수 있음을 입증하였습니다.

- **Performance Highlights**: 제안된 모델은 두 가지 유방 데이터세트에서 실험을 진행하였으며, 수술 후 환자의 비대칭을 사전 이미지에 적용하여 미적 결과를 예측하는 데 성공했습니다. 이러한 접근법은 환자들의 기대치를 조정하고 보다 적합한 치료 계획을 선택하는 데 도움을 줄 것으로 기대됩니다. 결과적으로, 본 연구는 유방 이미지를 분석하고 조작하는 데 있어 중요한 기여를 하게 됩니다.



### XiHeFusion: Harnessing Large Language Models for Science Communication in Nuclear Fusion (https://arxiv.org/abs/2502.05615)
- **What's New**: 이번 논문에서는 핵융합 분야에서 최초로 개발된 대형 언어 모델인 XiHeFusion을 제안합니다. XiHeFusion은 오픈 소스 대형 모델 Qwen2.5-14B를 기반으로 감독 학습을 통해 최적화되었습니다. 다양한 출처에서 수집한 핵융합 관련 지식을 활용하여 모델의 학습을 지원하며, 체계적인 과학 보급을 위한 대화형 모델로 설계되었습니다.

- **Technical Details**: 핵융합 모델인 XiHeFusion을 훈련하기 위해 CommonCrawl, CNKI(중국 국가 지식 인프라), eBooks, arXiv, 학위 논문 등의 다중 출처 지식을 집합적으로 수집했습니다. 이 정보는 100만 개 이상의 질문-답변 쌍으로 가공되어 모델 훈련의 기초 자료로 사용되었습니다. Chain-of-Thought 기법을 적용하여 모델의 논리적 추론 능력을 향상시켜, 보다 정확하고 논리적인 답변을 제공할 수 있도록 했습니다.

- **Performance Highlights**: 대규모 질문지(약 184개 질문)에 대한 테스트를 통해 XiHeFusion의 과학 보급 대화 능력을 평가했습니다. 실험 결과, XiHeFusion은 핵융합 관련 지식에 대한 질문에 효과적으로 잘 대응할 수 있음을 증명했습니다. 이 모델은 이론적 응용이 가능하며, 핵융합 분야에 대한 광범위한 이해를 증진시키는 데 크게 기여할 것으로 기대됩니다.



### FreeBlend: Advancing Concept Blending with Staged Feedback-Driven Interpolation Diffusion (https://arxiv.org/abs/2502.05606)
Comments:
          19 pages, 14 figures, conference

- **What's New**: 이번 연구에서 소개하는 FreeBlend는 생성 모델에서의 개념 혼합 문제를 효과적으로 해결하는 새로운 기법이다. FreeBlend는 훈련이 필요 없는 방식으로, 피드백 기반 메커니즘과 잠재(interpolation) 간섭을사용하여 이미지 생성과 수정 과정을 개선한다. 이 방법은 개념 혼합의 성능을 높이기 위해 여러 확산 모델의 장점을 통합하여, 이전의 모델들이 가진 한계를 보완하고 있다.

- **Technical Details**: FreeBlend는 세 가지 핵심 구성 요소로 이루어져 있다: Stable Diffusion에 대한 transferred unCLIP 이미지 조건, 단계별(interpolation) 증가 전략 및 확산(denoising) 과정의 피드백 기반 메커니즘이다. 이 연구에서는 텍스트가 아닌, 텍스트로 생성된 이미지를 정보를 제공하는 조건으로 사용하여, 기본 텍스트 기반 조건보다 더 정밀한 시각적 세부 정보를 제공하고자 한다. 또한, denoising 과정은 초기화 단계에서 시작하여 blending 및 세분화 단계를 거치며 진행된다.

- **Performance Highlights**: 실험 결과, FreeBlend는 생성된 이미지의 의미적 일관성과 시각적 품질을 크게 향상시키며, 매력적이고 일관된 결과를 보여준다. 본 연구의 정성적 및 정량적 평가 기법은 모델이 생성 과정에서 보여주는 성능을 rigorously하게 분석하고, 제안한 blending effect 평가 지표인 CLIP-BS에서 state-of-the-art 성능을 달성했음을 입증한다. 이를 통해 FreeBlend는 개념 혼합 분야에 있어 중요한 기여를 한다.



### Semantic Data Augmentation Enhanced Invariant Risk Minimization for Medical Image Domain Generalization (https://arxiv.org/abs/2502.05593)
- **What's New**: 이 논문에서는 의료 이미지 분류에서 데이터의 이질성(data heterogeneity) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 VIRM 방식에서 무작위 증강(random augmentation) 전략을 대체하기 위해, 도메인 간 공분산(inter-domain covariance)을 활용한 도메인 지향 방향 선택기(domain-oriented direction selector)를 도입하였습니다. 이 방법은 증강 방향을 목표 도메인으로 유도하여 도메인 간 불일치를 효과적으로 줄여주는 데 초점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 증강 방향을 도메인 간 공분산에 따라 조정하여, 각 증강샘플이 도메인 간 불일치를 줄이는 데 기여하도록 합니다. 이는 고유한 특징을 갖는 데이터셋이 부족한 의료 이미징에서 더욱 중요합니다. 또한, 새로운 접근 방식은 다중 센터 당뇨병 망막병증 데이터셋을 사용하여 실험적으로 검증되었으며, 기존의 최첨단 기법들보다 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 특히 데이터가 제한적이고 도메인 이질성이 큰 상황에서 현재의 가장 우수한 접근 방식들을 초월하는 성능을 보였습니다. 이는 의료 이미징 작업에서 보다 광범위한 응용 가능성을 제공하는 강력한 기초를 마련하고 있습니다. 새로운 방법론은 모델의 일반화 성능을 향상시켜 실제 임상 환경에서도 활용될 수 있는 가능성을 제시합니다.



### Event Stream-based Visual Object Tracking: HDETrack V2 and A High-Definition Benchmark (https://arxiv.org/abs/2502.05574)
Comments:
          Journal Extension of EventVOT, CVPR24

- **What's New**: 이번 연구에서는 EventVOT라는 새로운 대규모 고해상도 이벤트 기반 추적 데이터세트를 제안하였습니다. 또한 계층적 지식 증류 전략을 통해 다중 모달 데이터에서 단일 모달 데이터로의 변환을 효율적으로 수행하여 추적 성능을 향상시키는 방법을 제시하였습니다. 이 연구는 다양한 목표 객체에 대한 테스트 시 조정 전략을 도입하여 유연성과 성능을 개선한 점이 특징입니다.

- **Technical Details**: 제안된 방법은 이벤트 스트림을 활용하여 낮은 지연 시간의 추적을 가능하게 하며, 혁신적인 계층적 지식 증류 방식을 통해 다중 모달과 단일 모달 간의 지식을 효과적으로 전이합니다. 연구팀은 비디오 수준의 테스트 조정(test-time tuning) 전략을 적용해 실제 추적 시나리오에서 다양한 객체에 대한 모델의 적응력을 향상시켰습니다. 이를 통해 기존의 RGB 카메라에는 적합하지 않았던 접근을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 HDETrack V2는 1141개의 비디오로 구성된 EventVOT 데이터 세트와 함께 기존의 성능 기준을 초월하는 결과를 보였습니다. 다양한 벤치마크 데이터 세트(FE240hz, VisEvent, FELT)에서 충분한 실험을 통해 제안된 방법의 효과성이 입증되었습니다. HDETrack V2는 특히 EventVOT 데이터 세트에서 성능이 크게 향상된 것으로 나타났습니다.



### MMHMER:Multi-viewer and Multi-task for Handwritten Mathematical Expression Recognition (https://arxiv.org/abs/2502.05557)
Comments:
          7 pages;2 figures

- **What's New**: 본 논문에서는 CNN과 Transformer의 장점을 효과적으로 통합한 새로운 다중 관점, 다중 작업 프레임워크인 MMHMER 모델을 제안합니다. 이 모델은 손글씨 수학 표현 인식(SMER)에서 최첨단 성능을 달성하며, CROHME14, CROHME16, CROHME19 벤치마크에서 각각 63.96%, 62.51%, 65.46%의 인식률을 기록합니다. 또한, Posformer를 능가하는 성능 향상을 보여 더욱 흥미로운 연구 방향을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 Transformer 예측 지점과 CNN 예측 지점의 두 가지 병렬 브랜치로 구성되어 있습니다. DenseNet 백본 네트워크를 사용하여 고차원 특성 표현을 추출하고, Transformer 브랜치에서는 복수의 Multihead-Attention 모듈과 Implicit Attention 모듈을 활용해 수식의 기호를 효율적으로 분석합니다. CNN 브랜치에서는 GRU를 활용해 수식을 파싱하며, 다중 스케일 카운팅 모듈을 통해 각 기호 클래스의 수를 예측합니다.

- **Performance Highlights**: MMHMER 모델은 CNN과 Transformer의 성능을 극대화하며, 특히 복잡한 손글씨 수학 표현의 인식에서 높은 정확성을 기록합니다. 기호와 그 위치 간의 상관관계를 파악할 수 있는 위치 포레스트와 암묵적 주의 보정 모듈의 혁신적인 통합으로 이 성과를 가능하게 하였습니다. 이러한 연구는 향후 HMER 시스템에서 CNN과 Transformer의 조합 능력을 강화하는 중요한 기초 자료가 될 것입니다.



### Efficient Reinforcement Learning Through Adaptively Pretrained Visual Encoder (https://arxiv.org/abs/2502.05555)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 정교한 시각적 표현을 통해 적응형 미리 학습된 인코더(APE)를 사용하는 새로운 강화 학습(RL) 프레임워크를 제안합니다. APE는 다양한 실제 이미지를 활용하여 전이 가능한 표현을 학습하며, 정책 학습 단계 동안 환경과의 상호작용을 최소화하면서도 효율적으로 기능합니다. 이 연구는 APE가 기존의 주요 RL 방법들에 비해 더 뛰어난 성능을 제공한다는 것을 입증하고 있습니다.

- **Technical Details**: APE는 강화 학습 프레임워크에서 대조적 사전 학습(contrastive pretraining)과 적응형 증강(adaptive augmentation)을 통해 보다 일반화 가능한 특징을 추출하도록 설계되었습니다. 이는 RL 에이전트가 작업 환경에서 극소수의 상호작용만으로도 학습할 수 있도록 작성되었습니다. APE는 기존의 사전 훈련된 모델을 사용하지 않고, 다양한 적응적 변화로 이미지 분포를 조정하는 접근법을 시도합니다.

- **Performance Highlights**: 실험 결과, APE는 다양한 비주얼 RL 벤치마크에서 시각적 입력만으로 샘플링 효율성을 크게 향상시키고, 기존의 RL 방법들인 DreamerV3 및 DrQ-v2와 결합했을 때 최첨단 성능을 달성했습니다. 또한 APE는 정책 학습 시 보조 작업이나 추가 센서 정보 없이 작동하여 여러 시각적 RL 벤치마크에서 우수한 성능을 발휘하였습니다.



### 4DR P2T: 4D Radar Tensor Synthesis with Point Clouds (https://arxiv.org/abs/2502.05550)
Comments:
          6 pages, 4 figures

- **What's New**: 본 연구는 기존의 constant false alarm rate (CFAR) 알고리즘을 개선한 4D Radar Point-to-Tensor (4DR P2T) 모델을 제안합니다. 이 모델은 4D 레이더 포인트 클라우드 데이터를 딥러닝 응용에 적합한 텐서 데이터로 변환하여 측정 손실을 최소화합니다. 특히, 조건부 생성적 적대 신경망(cGAN)의 수정된 구조를 사용하여 4D 레이더 포인트 클라우드를 효과적으로 처리합니다.

- **Technical Details**: 4DR P2T 모델은 K-Radar 데이터셋을 이용하여 실험을 수행하였으며, 평균 PSNR 30.39dB 및 SSIM 0.96의 결과를 달성했습니다. 연구에서는 CFAR와 백분위수 기반 방법을 사용하여 텐서 생성 성능을 평가하였고, 5% 백분위수 방법이 전반적으로 가장 우수한 성능을 보임을 확인했습니다. 1% 백분위수 방법은 데이터 볼륨 축소와 성능을 잘 균형 있게 유지하며, 딥러닝 응용에 적합한 것으로 나타났습니다.

- **Performance Highlights**: 제안한 4DR P2T 모델은 K-Radar 데이터셋을 통해 실험적으로 검증되었으며, 높은 PSNR 및 SSIM 값으로 텐서 생성 성능을 입증하였습니다. 본 연구의 주요 기여는 4DR P2T 모델을 통한 텐서 데이터 생성, 5% 백분위수 데이터의 최적 성능 확인, 그리고 1% 백분위수 방법이 데이터 축소에 효과적임을 보여주는 것입니다. 이러한 결과는 자율주행 시스템에서 환경 인식 개선에 기여할 수 있습니다.



### Demystifying Catastrophic Forgetting in Two-Stage Incremental Object Detector (https://arxiv.org/abs/2502.05540)
Comments:
          14 pages, 7 figures, 9 tables

- **What's New**: 이 논문에서는 Incremental Object Detection (IOD)에서의 Catastrophic Forgetting 문제를 효과적으로 해결하기 위해 NSGP-RePRE라는 새로운 프레임워크를 제안합니다. 기존의 연구들이 전체 탐지기를 단일적으로 다루었던 것과 달리, 이 연구는 Faster R-CNN의 RoI Head 분류기에서 주로 발생하는 forgetting 현상을 분석하여 이를 바탕으로 프로토타입 재생 방법과 Gradient Projection을 결합한 방안을 제시합니다. 이 발견들은 기존의 가정에 도전하며 다양한 데이터셋에서 최첨단 성능을 달성하는 데 기여합니다.

- **Technical Details**: 제안된 NSGP-RePRE는 Regional Prototype REplay (RePRE)와 Null Space Gradient Projection (NSGP)으로 구성되어 있습니다. RePRE는 각 클래스의 전반적인 및 세부적인 프로토타입을 재생하여 RoI Head 분류기의 forgetting 현상을 완화합니다. NSGP는 오래된 입력의 서브스페이스에 수직한 방향으로 기능 추출기를 업데이트하여 프로토타입과 RoI 기능의 정렬을 보장하고, 기존 기능 추출기 업데이트로 인한 왜곡을 최소화합니다. 이 두 구성 요소의 결합에 의해 RoI Head가 새로운 지식을 학습하면서도 기존 지식을 유지할 수 있도록 합니다.

- **Performance Highlights**: NSGP-RePRE는 PASCAL VOC와 COCO 데이터셋에서 다양한 단일 및 다단계 설정하에서 최첨단 성능을 보여줍니다. Faster R-CNN의 주요 구성 요소에 대한 체계적인 분석을 통해 RoI Head 분류기가 Catastrophic Forgetting의 주요 원인임을 확인하고 이를 기반으로 IOD 방법 디자인에 있는 원칙적인 가이드를 제공합니다. 연구는 IOD의 속성을 깊이 이해하고 고성능 탐지기 개발에 중요한 통찰을 제공합니다.



### SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation (https://arxiv.org/abs/2502.05539)
- **What's New**: 본 논문에서는 Sparse Spectrum Adaptation via Discrete Hartley Transformation(SSH)라는 새로운 방법을 제안합니다. SSH는trainable parameters를 효과적으로 감소시키면서도 모델 성능을 향상시키는 것이 목표입니다. 기존의 low-rank adaptation(LoRA) 방식의 한계를 극복하고 더 높은 수준의 이미지 및 언어 이해 작업에서 더 좋은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: SSH는 기존의 weights를 discrete Hartley transform(DHT)하여 가장 중요한 주파수 성분을 선택하고 이를 기반으로 fine-tuning을 진행합니다. DHT는 복소수 연산을 피하며, 대칭성 덕분에 계산 복잡성을 줄이고 수치적 안정성을 높입니다. SSH는 선택된 주파수 성분의 가중치를 업데이트한 후, 역 변환을 통해 최종 가중치를 원래의 공간 도메인으로 복귀시키며, 이러한 방식으로 효율적인 파라미터 학습을 수행합니다.

- **Performance Highlights**: SSH는 LLaMA 3.1 및 ViT 모델에서 LoRA와 FourierFT보다 더 적은 파라미터로 유사한 성능을 보여줍니다. 특히, SSH는 FourierFT보다 55% 적은 GFLOPS를 요구하여 계산 효율성이 크게 향상되었습니다. 이러한 결과는 SSH의 접근 방식이 다양한 NLP 및 멀티모달 작업에서 높은 파라미터 효율성과 계산적 최적화를 제공한다는 것을 밝혔다.



### Fg-T2M++: LLMs-Augmented Fine-Grained Text Driven Human Motion Generation (https://arxiv.org/abs/2502.05534)
- **What's New**: 본 논문은 구체적인 텍스트에 기반한 인간 동작 생성의 문제를 다루고 있습니다. 기존 방법들은 텍스트에서 지정된 관계를 정확하게 포착하지 못하는데, 이는 세밀한 의미 단서의 효과적인 파싱 부족과 언어 구조를 완전히 모델링하지 못하는데 기인합니다. 새로운 세부 구조의 텍스트에 기반한 동작 생성 프레임워크 Fg-T2M++를 제안하여 이러한 한계를 극복하려 합니다.

- **Technical Details**: Fg-T2M++의 주요 구성 요소는 세 가지입니다: (1) LLMs Semantic Parsing 모듈로 텍스트로부터 신체 부위에 대한 설명과 의미를 추출합니다, (2) Hyperbolic Text Representation 모듈로 텍스트 단위 간의 관계 정보를 인코딩합니다, (3) Multi-Modal Fusion 모듈이 텍스트와 동작 특징을 계층적으로 융합합니다. 이러한 기술적 혁신들은 기존의 얕은 인코딩 방법과 차별화됩니다.

- **Performance Highlights**: Fg-T2M++는 HumanML3D 및 KIT-ML 데이터셋에서 SOTA 방법을 초월하는 성능을 보여주었습니다. 예를 들어 KIT-ML에서 FID 0.135와 MM-Dist 2.696를 기록하여 Fg-T2M의 0.571과 3.114에 비해 상당한 향상을 입증했습니다. 이를 통해 Fg-T2M++의 텍스트 의미에 따른 정확한 운동 생성 능력이 검증되었습니다.



### Evaluation of Vision Transformers for Multimodal Image Classification: A Case Study on Brain, Lung, and Kidney Tumors (https://arxiv.org/abs/2502.05517)
Comments:
          13 pages, 3 figures, 8 tables

- **What's New**: 본 연구는 의료 진단에서 가장 표준 기술로 자리잡은 신경망의 성능을 평가하며, 특히 MRI와 CT 데이터셋에서 최근의 Vision Transformers 아키텍처(예: Swin Transformer 및 MaxViT)가 어떻게 적용되는지를 분석합니다. 또한, 다양한 종양 분류 라벨을 가진 세 가지 이미지 세트를 활용하여 각 신경망의 성능을 비교하고, 서로 다른 이미지 모달리티(모드)와 질병 유형의 통합 효과를 조사합니다. 이 연구는 Transformer 기반 모델의 적응성과 효율성을 강조하며, 정밀 의학(precision medicine) 분야에서 중요한 진전을 이룰 가능성을 제시합니다.

- **Technical Details**: 이 연구는 의학 이미지 진단에서 신경망에 대한 신뢰성과 분석을 기반으로 하고 있으며, 특히 Swin Transformer가 제공하는 높은 정확도(99.9%)를 보여줍니다. 데이터 전처리와 모델 파인튜닝을 통해 최상의 결과(transfers learning)에 도달하려는 여러 실험을 설계했습니다. 세 가지 데이터세트는 뇌, 폐, 그리고 신장 종양을 포함하고 있으며, 각 데이터세트의 이미지들은 차별화된 특성과 라벨을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, Swin Transformer는 신장 종양 분류에서 99.9%의 정확도를 기록하며 안정적인 성능을 자랑했습니다. 반면, MaxViT는 단일 데이터에서 우수한 성능을 보였으나, 데이터 통합시에는 낮은 성과를 나타냈습니다. 이러한 결과들은 Vision Transformers가 의료 이미지 분석에서 중요한 역할을 할 수 있음을 보여줍니다.



### A Physical Coherence Benchmark for Evaluating Video Generation Models via Optical Flow-guided Frame Prediction (https://arxiv.org/abs/2502.05503)
- **What's New**: 이 논문에서는 물리적 일관성을 평가하기 위해 특별히 설계된 새로운 벤치마크인 PhyCoBench를 소개합니다. 이 벤치마크는 7가지 물리 원리를 포괄하는 120개의 프롬프트로 구성되어 있으며, 주로 물체의 움직임과 관련된 다양한 물리 현상을 포함합니다. 또한, 자동 평가 모델인 PhyCoPredictor를 제안하여 물리적 일관성을 보다 효과적으로 평가할 수 있는 방법을 마련하였습니다.

- **Technical Details**: PhyCoBench는 중력, 충돌, 진동, 마찰, 유체 역학, 발사 운동, 회전의 7가지 주요 물리 원리를 기반으로 다양한 물리적 현상을 평가하도록 설계되었습니다. 프롬프트는 최신 언어 모델을 활용하여 작성되었으며, 동적인 장면의 광학 흐름을 예측하는 PhyCoPredictor 모델이 개발되어 물리적 법칙을 따르는지 판단하는 데 사용됩니다. 이 모델은 초기 프레임과 텍스트 프롬프트를 입력으로 받아 광학 흐름을 예측하고, 이를 통해 동영상을 생성하는 과정에 활용됩니다.

- **Performance Highlights**: PhyCoPredictor는 기존의 수동 평가와 비교하여 자동 평가의 일관성을 입증하였으며, 현재 인적 평가와 가장 밀접한 정합성을 보임을 확인하였습니다. 이 시스템은 물리적 일관성을 평가하는 데 있어 중요한 통찰력을 제공하며, 향후 모델 최적화에 기여할 수 있습니다. 저자들은 PhyCoBench와 PhyCoPredictor를 GitHub를 통해 공개할 예정이며, 관련 데이터셋도 함께 제공됩니다.



### Robustifying Fourier Features Embeddings for Implicit Neural Representations (https://arxiv.org/abs/2502.05482)
- **What's New**: 이번 논문에서는 Implicit Neural Representations (INRs)의 주된 과제인 spectral bias를 해결하기 위한 혁신적인 접근 방식을 제안합니다. 저자들은 multi-layer perceptrons (MLPs)와 Fourier feature embeddings의 결합이 서로의 장점을 높일 수 있지만, 동시에 Fourier 특유의 한계도 도입할 수 있다고 가정합니다. 이를 바탕으로, bias-free MLPs를 활용한 adaptive linear filtering 기법을 통해 높은 주파수를 효과적으로 억제하는 방법을 모색합니다.

- **Technical Details**: 저자들은 Fourier feature embeddings와 MLPs의 상호작용이 고주파 노이즈의 주요 원인이라고 제안합니다. 이 연구에서 제안된 방법은 MLP를 bias-free로 설계하여, 입력의 패턴을 보존하면서도 불필요한 고주파 성분을 필터링할 수 있도록 합니다. 또한, 학습 중에 학습률을 조정하기 위해 커스텀 라인-서치 알고리즘을 도입하여 전체 성능을 향상시키는 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 인버스 그래픽스, 이미지 회귀, 3D 형상 회귀 등 다양한 INR 작업에서 noise가 감소하고 oversmoothness 문제를 피할 수 있었음을 보여주었습니다. 다른 최첨단 모델들과의 비교를 통해, 이 접근법이 MLPs가 더 섬세한 세부 정보를 포착할 수 있도록 하면서 높은 주파수의 노이즈를 효과적으로 줄임을 확인했습니다.



### Convolutional Neural Network Segmentation for Satellite Imagery Data to Identify Landforms Using U-Net Architectur (https://arxiv.org/abs/2502.05476)
Comments:
          6th International Conference on Computational Intelligence and Pattern Recognition

- **What's New**: 이 연구는 U-Net 아키텍처를 지형지물(landform) 감지를 위한 새로운 방법으로 제시하고 있습니다. 특히 전처리된 위성 이미지(preprocessed satellite imagery)를 사용하여 효과적인 기능 추출을 구현합니다. 또한, Dropout 기법을 통해 모델의 일반화 능력을 향상시키고 Adam optimizer를 통해 훈련의 효율성을 높였습니다.

- **Technical Details**: U-Net 모델은 Convolutional Neural Network (CNN) 기반의 세분화(segmentation) 기술을 사용하여 높은 해상도의 출력(high-resolution outputs)과 빠른 기능 추출을 지원합니다. 연구에서는 대규모 전처리된 위성 지형 이미지(sample of preprocessed satellite topographical images)를 활용하여 모델의 성능을 철저히 평가하였습니다. U-Net 아키텍처는 픽셀 단위 분류(pixel-wise categorization)와 종합적인 세분화 맵(segmentation map) 생산에 강조점을 두고 있습니다.

- **Performance Highlights**: U-Net 모델은 자율 주행, 재난 관리, 토지 이용 계획과 같은 실제 적용 분야에서 두각을 나타냅니다. 연구 결과는 U-Net 아키텍처가 머신 러닝(machine learning)과 이미지 처리(image processing) 기술 발전에 크게 기여하고 있음을 강조합니다. 이 모델은 다양한 응용 프로그램에 널리 적용 가능성을 보여주며, 실제 세계의 이미지 분류, 분석 및 지형지물 식별 분야에서 중요한 역할을 하고 있음을 입증했습니다.



### LMS-Net: A Learned Mumford-Shah Network For Few-Shot Medical Image Segmentation (https://arxiv.org/abs/2502.05473)
- **What's New**: 이 논문에서는 새로운 심층 펼침 네트워크인 Learned Mumford-Shah Network (LMS-Net)을 제안합니다. 기존의 few-shot semantic segmentation (FSS) 방법들이 해석 가능성이 부족하다는 문제를 해결하고자, LMS-Net은 물리적 구조를 고려한 접근 방식을 통합하여 설계되었습니다. 이는 프로토타입 간 비교를 통한 효율적인 모델링을 통해 이루어집니다.

- **Technical Details**: LMS-Net은 LMS 모델을 수학적 기초로 활용하여 프로토타입 업데이트 및 마스크 업데이트 작업으로 재구성합니다. 이 모델은 깊은 선행 정보(deep priors)로 복잡한 공간 구조를 모델링할 수 있는 능력을 갖추고 있으며, 교대 최적화 알고리즘(alternating optimization algorithm)을 통해 효율적으로 문제를 해결합니다. 네트워크 모듈로 펼쳐진 반복 단계는 LMS-Net의 명확한 해석 가능성을 제공합니다.

- **Performance Highlights**: 세 개의 공개 의료 분할 데이터셋에 대한 종합적인 실험을 통해, LMS-Net의 효용이 뚜렷하게 입증되었습니다. 이 방법은 복잡한 구조를 처리하고 도전적인 분할 시나리오에 적응하는 데 있어 뛰어난 정확도와 강인성을 보여줍니다. 따라서, LMS-Net은 의료 이미징 응용 분야에서 FSS의 발전 가능성을 강조합니다.



### DCENWCNet: A Deep CNN Ensemble Network for White Blood Cell Classification with LIME-Based Explainability (https://arxiv.org/abs/2502.05459)
- **What's New**: 이번 연구에서는 백혈구(WBC)의 분류를 위한 새로운 앙상블 접근법인 DCENWCNet을 제안합니다. 이 모델은 세 가지 서로 다른 CNN 아키텍처를 통합하여 드롭아웃(dropout)과 맥스 풀링(max-pooling) 레이어 설정을 다양하게 구성함으로써 특징 학습을 더욱 향상시킵니다.

- **Technical Details**: DCENWCNet은 기존의 CNN 모델들이 가지는 데이터 불균형과 데이터 증강 부족 같은 문제를 해결하는 데 중점을 둡니다. 이 모델은 일반적으로 인정받는 Rabbin-WBC 데이터셋에서 검토되었으며, 편향-분산(bias-variance) 균형을 효과적으로 이루어냅니다.

- **Performance Highlights**: 모델은 평균 정확도(mean accuracy)에서 기존의 최첨단 네트워크를 초월하며, 정밀도(precision), 재현율(recall), F1-score, ROC 곡선 아래 면적(AUC)에서 모든 카테고리에서 우수한 성능을 나타냅니다. 또한, LIME(Local Interpretable Model-Agnostic Explanations)와 같은 신뢰할 수 있는 설명 기법을 사용하여 모델의 예측을 해석 가능하게 만들어 사용자에게 자신감을 부여합니다.



### Block Graph Neural Networks for tumor heterogeneity prediction (https://arxiv.org/abs/2502.05458)
Comments:
          27 pages, 8 figures

- **What's New**: 본 논문은 종양 진화를 시뮬레이션하는 수학 모델을 기반으로 하여 인공 데이터셋을 생성하고 종양을 분류하는 방법을 제안합니다. 기존의 머신러닝 기법들은 MRI 및 조직병리 데이터의 복잡한 전처리를 필요로 하지만, 본 연구는 이러한 과정을 간소화 하여 종양 이질성을 예측할 수 있는 과정을 제공합니다. 또한, Block Graph Neural Networks (BGNN)를 통해 종양 이질성을 예측하는 새로운 접근 방식을 도입하였습니다.

- **Technical Details**: 저자들은 인공 데이터에서 절단 및 그래프 생성 프로세스, 종양 특성 설계 및 BGNN 구축을 통해 종양 이질성을 정량적으로 측정하는 방법을 개발했습니다. 이질성은 정규화된 엔트로피(normalized entropy)를 활용하여 평가되며, 이를 통해 종양이 높은 이질성 또는 낮은 이질성을 가지는지 구분할 수 있는 임계값을 설정하였습니다. 실험 결과, 제안된 특성과 모델 조합을 통해 시험 데이터에서 89.67%의 정확도를 달성하는 성과를 보였습니다.

- **Performance Highlights**: 제안된 방법은 전통적인 분류 방법에 비해 종양 이질성 예측을 개선하며, 환자별 맞춤 치료가 가능하도록 도와줄 수 있습니다. 특히, 발생하는 AI 지원 등급 매기기, 그리고 공간 전사체학(spatial transcriptomics)의 최신 트렌드와 일치하는 결과를 제시합니다. 이 연구는 Ki-67 증식 지수와 사망 마커와 같은 요소를 기존의 등급 시스템에 추가하는 것이 이질성 예측과 종양 분류 개선에 기여할 수 있음을 나타냅니다.



### Content-based Video Retrieval in Traffic Videos using Latent Dirichlet Allocation Topic Mod (https://arxiv.org/abs/2502.05457)
- **What's New**: 본 연구에서 제안된 방법은 Latent Dirichlet Allocation (LDA) 주제 모델을 이용하여 감시 비디오를 비지도 학습 방식으로 주석 처리합니다. 제안된 방법은 기존의 주제 모델이 가지고 있는 애매성 문제를 해결하기 위해 기능 벡터와 주 모델을 처리하여 애매성이 없는 원시 패턴으로 장면을 설명하는 2차 모델을 생성합니다. 실험 결과, 검색 작업에서 기존 주제 모델 기반 방법들에 비해 성능이 향상된 것을 확인했습니다.

- **Technical Details**: 제안된 방법은 비디오 스트림에서 저수준 기능을 추출하고, Clip Blob Decomposition 단계에서 각 기능 벡터를 단일 에이전트 활동을 포함하는 여러 기능 벡터로 분해합니다. 이 과정에서 Connected Component Labeling (CCL) 알고리즘이 사용되며, 각 비주얼 기능에 대해 별도의 LDA 모델이 학습되어 서로 다른 기능 공간에서의 활동들이 하나의 주제로 병합되는 것을 방지합니다. 또한, 이 모델에서 생성된 주제들은 애매성이 없고 각 주제는 원초적인 행동을 나타냅니다.

- **Performance Highlights**: 제안된 방법은 검색 작업에서 최소 80% 이상의 진짜 양성(true positive) 성능 향상과 124%의 거짓 양성(false positive) 성능 향상을 달성하였습니다. 이러한 성능 개선은 비디오에 대한 검색 속도를 크게 향상시킬 수 있는 경량 데이터베이스 덕분입니다. 또한, 사용자는 제안된 쿼리 공식화를 통해 다양한 활동을 한 번에 정의하고 검색할 수 있는 여러 검색 전략을 활용할 수 있습니다.



### AdaFlow: Efficient Long Video Editing via Adaptive Attention Slimming And Keyframe Selection (https://arxiv.org/abs/2502.05433)
- **What's New**: 이번 논문에서는 AdaFlow라는 새로운 접근 방식을 통해 긴 비디오 편집의 효율성과 효과를 극대화하는 방법을 제안합니다. 기존의 방식들이 키프레임 번역 과정에서 비효율적이라는 점을 감안하여, AdaFlow는 Adaptive Attention Slimming을 도입하여 불필요한 토큰을 제거함으로써 계산 부담을 크게 줄입니다. 또한, AdaFlow는 Adaptive Keyframe Selection 기법을 통해 상징적인 프레임을 선택하여 비디오 편집 품질을 개선합니다.

- **Technical Details**: AdaFlow의 핵심은 비디오 프레임에서 모든 토큰이 동등하게 중요하지 않다는 점을 인식하고, 중요도가 낮은 토큰을 줄이는 Adaptive Attention Slimming을 사용하는 것입니다. 이는 KV(키-값) 시퀀스를 압축하여 키프레임의 수를 크게 증가시킵니다. 또한, Adaptive Keyframe Selection을 통해 불필요한 키프레임 번역을 피하고 계산 리소스를 효율적으로 활용할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, AdaFlow는 A800 GPU를 사용해 1초 이상의 비디오(1k 프레임 이상)를 단 한번의 추론으로 고품질 편집을 수행할 수 있습니다. 기존 방법들과 비교했을 때, AdaFlow는 효율성과 편집 품질 모두에서 뚜렷한 우위를 보였습니다. 이러한 결과는 AdaFlow가 다양한 비디오 편집 작업을 효과적으로 수행할 수 있는 가능성을 보여줍니다.



### MoFM: A Large-Scale Human Motion Foundation Mod (https://arxiv.org/abs/2502.05432)
- **What's New**: 본 연구에서는 MoFM(Motion Foundation Model)을 소개합니다. 이 모델은 복잡한 인간의 동작을 시간과 공간적으로 이해하기 위해 설계되었습니다. MoFM은 대규모 자가 지도 학습을 지원하며, 휴먼 모션을 더 효율적이고 확장 가능한 형태로 표현하는 데 중심적인 역할을 하는 MotionBook이라는 동작 사전을 활용합니다. 이를 통해 다양한 다운스트림 작업에 적합한 기초 모델을 제공합니다.

- **Technical Details**: MoFM은 Thermal Cubes를 사용하여 스페이셔-템포럴(Spatio-Temporal) 동작 열지도를 캡쳐합니다. 이러한 열지도는 discrete Variational Encoder-Decoder(dVED)에 의해 인코딩되어, 구조화된 인간 동작 표현인 MotionBook으로 정리됩니다. MoFM은 BERT 스타일의 자가 지도 방식을 적용하여 사전 훈련된 후, 다양한 동작 기반 애플리케이션으로 쉽게 전이될 수 있습니다.

- **Performance Highlights**: MoFM은 행동 인식, 일회성 행동 인식, 자가 지도 비정상 탐지 및 감독 비정상 탐지와 같은 네 가지 인간 중심의 다운스트림 작업에서 성능 평가를 받았습니다. 실험 결과, MoFM이 다양한 인간 동작 이해 작업 사이에서 뛰어난 일반화 능력을 보여주었으며, 유연하고 task-agnostic 기초 모델로서 성능을 발휘함을 입증하였습니다.



### LRA-GNN: Latent Relation-Aware Graph Neural Network with Initial and Dynamic Residual for Facial Age Estimation (https://arxiv.org/abs/2502.05423)
- **What's New**: 이 논문에서는 얼굴 표정 모델링을 위해 새로운 Latent Relation-Aware Graph Neural Network (LRA-GNN)를 소개합니다. 기존의 방법들은 유사성 기준에 의한 노드 간 관계 설정의 한계로 일부 잠재적인 관계를 놓치고 있는 문제를 해결하고자 합니다. 이 모델은 초기 그래프를 구성하고, 무작위 경로 탐색 전략을 통해 얼굴 정보의 깊이 있는 표현을 확보합니다.

- **Technical Details**: LRA-GNN은 얼굴 주요 지점을 바탕으로 초기 그래프를 생성한 후, 다중 주의 메커니즘(multi-attention mechanism)을 활용하여 잠재적인 관계(latent relations)를 포착합니다. 이 과정에서 완전 연결 그래프를 생성하여 풍부한 얼굴 정보와 구조를 포함하게 됩니다. 또한, 깊은 특성 추출을 위한 그래프 합성곱 네트워크(deep residual graph convolutional networks)를 통해 정보의 일관성과 다양성을 확보합니다.

- **Performance Highlights**: 제안된 프레임워크는 여러 나이 추정 벤치마크에서 기존 최첨단 방법들을 초월하여 그 성능과 효과를 입증합니다. 점진적 강화 학습(progressive reinforcement learning)을 도입하여 앙상블 분류 회귀기(ensemble classification regressor)의 정확도와 일반화 능력도 개선하였습니다. 이러한 성과는 LRA-GNN이 얼굴 나이 추정 분야에 미치는 긍정적인 영향을 보여줍니다.



### Show-o Turbo: Towards Accelerated Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2502.05415)
- **What's New**: 본 논문에서는 Show-o Turbo를 소개하여 이미지와 텍스트 생성을 모두 포함하는 통합 멀티모달 모델의 효율성을 높이고자 합니다. Show-o는 세밀한 노이즈 제거와 자동 회귀적 텍스트 디코딩 방식으로 작동하는데, 이 과정에서 발생하는 비효율성을 개선하기 위해 새로운 접근법을 제안합니다. 이를 통해 Show-o Turbo는 기존 모델 대비 빠른 생성을 가능하게 하고, 더 나은 성능을 갖춥니다.

- **Technical Details**: Show-o Turbo는 텍스트 토큰의 병렬 디코딩을 기반으로 이미지와 텍스트 생성의 통합적 관점을 제시합니다. 일관성 증류(Consistency Distillation, CD) 기술을 활용하여 다양한 샘플링 경로의 고유점을 결정하고, 이를 통해 모델의 훈련 수렴 속도를 개선합니다. 이러한 접근법을 통해 Show-o Turbo는 훈련 단계에서 서로 다른 세그먼트 수를 이용한 커리큘럼 학습 방식을 통합하여 성능을 높입니다.

- **Performance Highlights**: 실험 결과, Show-o Turbo는 4회의 샘플링 단계 동안 GenEval 점수 0.625를 기록하며, 이는 8회의 샘플링 단계와 클래스 없는 가이드(Classifier-Free Guidance, CFG)를 사용하는 기존 Show-o 모델보다 더 우수한 성능을 보였습니다. 또한 이미지에서 텍스트로의 변환 과정에서도 성능을 크게 저하시키지 않으면서 1.5배의 속도 향상을 달성하였습니다.



### Vision-in-the-loop Simulation for Deep Monocular Pose Estimation of UAV in Ocean Environmen (https://arxiv.org/abs/2502.05409)
Comments:
          8 pages, 15 figures, conference

- **What's New**: 이 논문은 해양 환경에서 운용되는 UAV의 깊이 모노큘러 포즈 추정을 위한 비전 인 더 루프(vision-in-the-loop) 시뮬레이션 환경을 제안합니다. 최근 트랜스포머 아키텍처를 채택한 심층 신경망이 UAV의 자세 추정에 성공했고, GPS 기반 접근법의 여러 한계를 극복했습니다. 연구선의 제한된 가용성과 높은 운영 비용으로 인해 실제 해양 환경에서 심층 포즈 추정 방안을 검증하는 데 어려움이 많습니다.

- **Technical Details**: 이 연구는 새로운 가우시안 스플래팅(Gaussian splatting) 기술을 활용하여 사진 현실적인 3D 가상 환경을 만들고, 이를 통해 UAV의 비전 기반 제어와 추정 알고리즘을 평가합니다. TNN-MO 아키텍처를 사용하여 단일 RGB 이미지에서 UAV의 6D 포즈를 정확하게 추정하며, 가상 환경에서 합성 데이터를 생성하여 모델 훈련과 검증에 활용합니다. 이 과정에서는 복잡한 조명 조건, 동적인 해양 파도 등 다양한 환경 요소를 고려하여 현실감을 높이고, 모델 일반화 성능을 개선합니다.

- **Performance Highlights**: TNN-MO 모델은 5,500개의 이미지에서 평균 위치 오차 0.204와 태세 오차 0.91°를 기록하며 다양한 조건에서도 강건함을 확인했습니다. 실제 데이터 검증에서는 UAV에 장착된 데이터 수집 시스템을 통해 RTK-GPS 측정값과 일치하는 결과를 도출했습니다. 이 연구는 해양 환경에서 UAV의 자율 비행 및 안전한 발사 및 회수를 위한 비전 기반 접근법의 가능성을 제시하며, 광범위한 실제 시나리오를 재현할 수 있는 능력을 보여줍니다.



### Convolutional Deep Colorization for Image Compression: A Color Grid Based Approach (https://arxiv.org/abs/2502.05402)
- **What's New**: 이번 연구는 이미지 압축 최적화를 위한 색상 유지 접근 방식을 자동화하여, convolutional (컨볼루션) 색상화 네트워크 아키텍처를 최적화하는 데 중점을 두고 있습니다. 연구진은 이미지 컬러화 알고리즘을 사용하여 저장해야 할 색상 데이터의 양을 줄이면서도 이미지의 색상을 충실하게 복원하는 것을 목표로 하고 있습니다. 결과적으로 좋은 이미지 압축 비율을 달성하면서도 높은 CSIM 값을 기록했습니다.

- **Technical Details**: 연구는 이미지 데이터 셋에서 색상의 대부분을 제거하고 특정 색상 픽셀만을 유지하는 방법을 채택했습니다. 모델 학습 시 mean squared error (MSE)를 손실 함수로 사용하였고, ADAM 최적화를 통해 30 에폭(epoch) 동안 모델을 훈련시켰습니다. 그러면서 색상 정보가 얼마나 유지되느냐에 따라 성능을 평가하여 최적의 n 값을 찾아내는 방식을 적용했습니다.

- **Performance Highlights**: PSNR과 CSIM 두 가지 지표를 사용하여 이미지 색상화 성능을 평가하며, 각 n 값에 대해 색상화 질이 어떻게 변화하는지 분석했습니다. 연구 결과, n=20에서 최상의 압축 성능을 보였으며, 그 이후로는 색상화의 질이 감소하는 경향을 보였습니다. 이 연구는 이미지 컬러화의 자동화와 이로 인한 저장 용량 최적화의 가능성을 보여주는 중요한 발전이라고 할 수 있습니다.



### Beyond and Free from Diffusion: Invertible Guided Consistency Training (https://arxiv.org/abs/2502.05391)
- **What's New**: 이 연구는 인버터블 가이드 일관성 훈련(invertible Guided Consistency Training, iGCT)이라는 새롭고 데이터 기반의 훈련 프레임워크를 제안합니다. iGCT는 Diffusion Models(DMs)을 훈련하고 증류할 필요 없이 빠르고 가이드된 이미지 생성 및 편집을 가능하게 하여 전체 계산 요구 사항을 크게 줄입니다. 이 방법은 높은 가이드 스케일에서 발생하는 포화 아티팩트(saturation artifacts)를 해결하여 기존 Classifier-free Guidance(CFG)보다 향상된 성능을 보여줍니다.

- **Technical Details**: 이 연구에서는 이미지 생성에 대한 최첨단 접근 방식으로 Consistency Models(CMs)에 중점을 두고 있습니다. iGCT는 목표 클린 샘플(target clean sample)을 소스와 분리하여 무조건적인(noise) 및 조건적인(noise) 효과를 함께 포획하며, 이는 동일한 차원의 잡음으로 변환되는 노이저(noiser)를 통합하여 구현됩니다. 이 접근 방식은 두 단계 훈련 프로세스를 필요로 하지 않아 훈련의 유연성(flexibility)과 효율성(efficiency)을 높입니다.

- **Performance Highlights**: 실험 결과, CIFAR-10 및 ImageNet64에서 iGCT는 CFG와 비교하여 FID 및 정밀도(precision) 면에서 대폭 개선된 결과를 나타냅니다. 가이드 스케일이 13일 때, iGCT의 정밀도는 0.8에 도달하는 반면 기존 DMs는 0.47로 떨어집니다. iGCT는 다단계 DMs 및 증류된 CMs에 비해 높은 FID와 Precision/Recall을 달성하여 빠른 이미지 편집에도 효과적임을 입증합니다.



### Coarse-to-Fine Structure-Aware Artistic Style Transfer (https://arxiv.org/abs/2502.05387)
Comments:
          21 pages, 17 figures

- **What's New**: 이 논문에서는 예술적 스타일 전송(artistic style transfer) 방법의 일반적인 문제를 해결하기 위해 새로운 접근법을 제안합니다. 기존 방법이 내용 이미지(content image)의 글로벌 구조(global structure)에 스타일 이미지(style image)의 질감과 색상(texture and color)만 전이하는 데 비해, 제안된 방법은 로컬 스타일 구조(local style structure)를 로컬 콘텐츠 구조(local content structure)에 융합(fuse)합니다.

- **Technical Details**: 제안된 방법에서는 먼저 저해상도(low resolution)에서 코스 네트워크(Coarse Network)를 사용하여 거친 스타일화된 특징(coarse stylized features)을 재구성합니다. 이 단계에서 스타일의 색상 분포(style color distribution)가 대략 전이되고 콘텐츠 구조(content structure)는 스타일 구조(style structure)와 결합됩니다. 이후, 이러한 재구성된 특징들과 콘텐츠 특징(content features)을 사용하여, 구조 인식(structure-aware)의 고해상도(high resolution) 스타일화된 이미지를 생성하기 위해 파인 네트워크(Fine Network)와 세 개의 구조 선택적 융합(structural selective fusion, SSF) 모듈을 적용합니다.

- **Performance Highlights**: 제안된 방법은 뛰어난 고품질 스타일화 결과를 생성하는 것으로 그 효과성이 입증되었습니다. 또한 여러 최신 스타일 전송 방법들과 비교하여, 스타일과 콘텐츠의 로컬 구조에 대한 일관성을 유지하면서 더 매력적인(visually appealing) 이미지를 제공합니다. 이러한 결과는 예술적 스타일 전송 분야에서의 새로운 가능성을 보여줍니다.



### NextBestPath: Efficient 3D Mapping of Unseen Environments (https://arxiv.org/abs/2502.05378)
Comments:
          To appear at ICLR 2025. Project webpage: this https URL

- **What's New**: 이번 연구에서는 액티브 3D 매핑(active 3D mapping)의 문제를 다루며, 새로운 장소를 효율적으로 재구성하기 위한 최적의 경로를 찾는 방법을 제안합니다. 기존 방법들이 에이전트 주변의 다음 최선의 뷰(next best view)만을 예측하여 국소 영역에 갇히기 쉬운 점을 지적합니다. 이에 대한 해결책으로, 뚜렷한 복잡성을 가지고 다양한 실내 환경을 처리하기 위해 AiMDoom이라는 새로운 데이터셋을 소개하고, NBP(Next-Best-Path)라는 새로운 기법을 제안합니다.

- **Technical Details**: NBP는 장기 목표(long-term goals)에 중점을 두고 최적 경로를 예측하는 방법으로, 에이전트의 현재 위치를 중심으로 넓은 영역에 대한 표면 커버리지를 예측합니다. 이 모델은 매핑 진행 인코더(mapping progress encoder), 커버리지 이득 디코더(coverage gain decoder), 장애물 맵 디코더(obstacle map decoder)로 구성되며, 이를 통해 장애물을 피하면서 장기 목표로 가는 최단 경로를 계산할 수 있습니다. 데이터 수집은 온라인으로 이루어지며, 데이터 보강(data augmentation)과 커리큘럼 학습(curriculum learning) 방식도 도입되어 훈련 효율성을 높입니다.

- **Performance Highlights**: 제안된 NBP 모델은 기존 MP3D 데이터셋과 AiMDoom 데이터셋에서 최첨단(NB) 방법들보다 훨씬 우수한 성능을 나타냅니다. 간단한 환경부터 복잡한 환경에 이르기까지 뛰어난 매핑 효율을 달성하였으며, 실내 환경의 복잡성에 대한 체계적인 평가를 수행하기 위한 AiMDoom 데이터셋을 최초로 소개했습니다. 이로 인해 액티브 매핑의 연구 분야에서 중요한 기여를 하고 있습니다.



### Towards Fine-grained Renal Vasculature Segmentation: Full-Scale Hierarchical Learning with FH-Seg (https://arxiv.org/abs/2502.05320)
- **What's New**: 이번 연구에서는 신장 혈관의 정확한 세분화를 위한 Full-scale Hierarchical Learning Framework(FH-Seg)를 소개합니다. FH-Seg는 각 스케일에서 해부학적 정보와 문맥 정보를 통합하기 위해 전체 스케일 스킵 연결을 사용합니다. 또한, 비핵심 정보로의 간섭을 줄이기 위해 학습 가능한 계층형 소프트 어텐션 게이트를 구현하여 혈관의 주요 특징에 집중할 수 있도록 합니다.

- **Technical Details**: FH-Seg는 Residual U-Net 백본을 기반으로 하며, 각 수준의 인코더와 디코더 사이에 전면 스케일 스킵 연결을 포함하여 세부적인 해부학적 정보와 문맥 의미를 결합합니다. 특히 계층형 소프트 어텐션 게이트를 도입하여, 서로 다른 스케일의 피쳐 맵이 동일한 공간 해상도로 정규화된 후 가중치가 부여되도록 합니다. 이를 통해 다중 스케일 피쳐 간의 호환성이 향상됩니다.

- **Performance Highlights**: LRV(Large Renal Vasculature) 데이터셋에서 FH-Seg는 71.23%의 Dice 계수와 73.06%의 F1 점수를 달성하여 타 방법인 Omni-Seg보다 각각 2.67%와 2.13% 향상된 성능을 보였습니다. 이러한 성과는 FH-Seg가 기존 방법에서 겪던 세밀한 세분화의 어려움을 극복할 수 있는 능력을 입증합니다. 코드 및 데이터셋은 공개되어 연구자들이 활용할 수 있도록 제공됩니다.



### Drone Detection and Tracking with YOLO and a Rule-based Method (https://arxiv.org/abs/2502.05292)
- **What's New**: 이 논문은 드론 감지 시스템을 위한 새로운 데이터셋을 확장하고 YOLOv7 모델을 활용하여 드론 탐지 성능을 향상시키는 방법을 제안합니다. 기존의 적외선 이미지 데이터셋에 추가로 컬러 이미지와 해양 비디오를 기반으로 하는 새로운 데이터를 통합하여 다양한 환경에서 드론의 탐지 능력을 높이는 것을 목표로 합니다. 이러한 접근 방식은 드론 안전 및 프라이버시 보호를 위한 규제가 필요한 현대 사회에서 중요한 해법이 될 수 있습니다.

- **Technical Details**: 이 연구에서는 드론을 탐지하기 위해 YOLOv7 기반의 딥 러닝 모델을 사용하며, 데이터셋은 각각의 카메라에서 수집된 비디오와 이미지를 포함합니다. GStreamer를 활용하여 비디오 스트림을 전송하고 YOLO 모델을 Docker 컨테이너 내에서 실행하여 드론 감지 알고리즘을 적용하는 시스템 구조를 설명합니다. 데이터셋의 주 성분 중 일부는 주석이 달린 비디오 프레임에서 추출된 컬러 이미지로 구성되며, 이로 인해 모델의 정확도를 높일 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 제공된 데이터셋의 총 이미지는 43,080개이며, 각기 다른 환경 조건에서 드론을 탐지하는 성능을 실험할 수 있는 플랫폼을 마련합니다. 연구에서는 뛰어난 탐지 성능을 보장하기 위해 YOLOv7 모델과 다양한 모듈의 조합을 비교하여 그 의미를 분석합니다. 최종적으로, 감지 성능 및 추적 결과를 개선하기 위해 간단한 앙상블 방법이 소개됩니다.



### Homeomorphism Prior for False Positive and Negative Problem in Medical Image Dense Contrastive Representation Learning (https://arxiv.org/abs/2502.05282)
Comments:
          Accepted by T-PAMI 2025

- **What's New**: 의료 이미지를 위한 Dense Contrastive Representation Learning (DCRL)에서 발생하는 대규모 False Positive (FP) 및 False Negative (FN) 문제를 해결하기 위해 GEoMetric vIsual deNse sImilarity (GEMINI) 학습을 제안합니다. GEMINI는 DCRL에 homeomorphism 선행 지식을 포함시켜 의료 이미지 간의 신뢰할 수 있는 대응 관계 발견을 가능하게 합니다. 이를 통해 의료 이미지를 위한 밀집 표현 학습의 효과성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: GEMINI 학습은 두 가지 주요 요소로 구성됩니다. 첫째, Deformable Homeomorphism Learning (DHL)은 두 이미지를 정렬하기 위해 변형 가능한 매핑(DVF)을 예측하도록 학습하며, 이를 통해 픽셀 대비에서 신뢰할 수 있는 양성 쌍을 얻을 수 있습니다. 둘째, Geometric Semantic Similarity (GSS)는 정렬 정도를 측정하는 데 있어 의미적 유사성을 결합하여 정확한 정렬을 통한 양성 쌍의 신뢰할 수 있는 학습을 촉진합니다.

- **Performance Highlights**: GEMINI 학습은 7개의 데이터 세트에서 실험을 진행하며 기존의 방법을 초월하는 유망한 결과를 보였습니다. 이 연구에서 제안하는 접근법은 실제 의료 이미지 밀집 예측 작업에서 데이터를 더 효과적으로 활용할 수 있게 하여, 의료 이미지의 비용과 레이블 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### Invizo: Arabic Handwritten Document Optical Character Recognition Solution (https://arxiv.org/abs/2502.05277)
- **What's New**: 이번 연구는 아랍어 손글씨 및 인쇄 텍스트를 인식하기 위한 최첨단 솔루션을 제안합니다. 이 모델은 CNN 기반 특징 추출 및 Transformer 기반 시퀀스 모델링을 통합하여 다양한 손글씨 스타일 및 노이즈 조건을 처리합니다. 높은 정확도와 성과를 달성하여 실제 OCR 작업에서 신뢰할 수 있는 결과를 보여주고 있습니다.

- **Technical Details**: OCR(Optical Character Recognition) 프로세스는 이미지의 텍스트를 기계가 읽을 수 있는 형식으로 변환하는 방법입니다. 제안된 시스템은 이미지 획득, 전처리, 텍스트 감지 및 분할, 문자 인식 등의 여러 단계를 포함하여 아랍어의 다양한 필기 스타일 문제를 해결하기 위한 포괄적인 접근 방식을 채택합니다. 이 과정에서 CTC 알고리즘 및 Attention 기반 디코더를 사용하여 인식 정확도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 인쇄 텍스트에서 0.59% CER 및 1.72% WER, 손글씨 텍스트에서 7.91% CER 및 31.41% WER를 달성했습니다. 제안된 솔루션은 여러 텍스트 감지 및 인식 모델을 갖추고 있어 실제 사용환경에서도 매우 유용한 결과를 제공합니다. 최종적으로 이 시스템은 아랍어 문서 처리의 자동화에 중요한 기여를 할 것으로 기대됩니다.



### Interpretable Failure Detection with Human-Level Concepts (https://arxiv.org/abs/2502.05275)
- **What's New**: 본 연구에서는 신뢰할 수 있는 실패 감지 기법을 제시하고, 인간 수준의 개념을 활용하여 모델의 실패를 신뢰성 있게 탐지하고 그 이유를 해석할 수 있는 방법을 도입하였습니다. 기존의 confidence score functions(CSFs)는 카테고리 수준의 신호에 의존하여 실패를 감지하는데 한계가 있으며, 우리의 접근 방식은 보다 세분화된 신뢰도 측정을 가능하게 합니다. 특히, 이런 기법은 ImageNet과 EuroSAT에서 각각 3.7% 및 9%의 false positive rate를 현저히 감소시키는 효과를 보여줍니다.

- **Technical Details**: 신뢰도 추정의 새로운 관점을 제시하는 우리의 방법은 Ordinal Ranking of Concept Activation(ORCA)에 기반합니다. 이 방법은 모델이 입력된 이미지에서 개념 활성화를 순위화하여 신뢰도를 평가하게 합니다. 각 카테고리에 대해 미세 조정된 신호를 통합함으로써, 모델의 신뢰도에 대한 보다 정교한 평가가 가능해지며, 사용자에게 실패 이유를 설명할 수 있는 구성 요소를 제공합니다.

- **Performance Highlights**: 이 연구의 중요한 기여는 모델이 실패하는 상황을 신뢰성 있게 탐지하고 그 이유를 해석하는 방법을 제시한 것입니다. 우리의 접근 방식은 다양한 실세계 이미지 분류 벤치마크에서 실패 예측 성능을 향상시킨다고 실증적으로 입증되었습니다. 특히, 자연 이미지 데이터셋과 원격 감지 이미지를 포함한 포괄적인 벤치마크에서 우리의 방법의 효과성을 엄격하게 검증하였습니다.



### Survey on AI-Generated Media Detection: From Non-MLLM to MLLM (https://arxiv.org/abs/2502.05240)
- **What's New**: 이번 논문은 AI 생성 미디어에 대한 효과적인 탐지 방법을 체계적으로 리뷰하고, 도메인 특화 탐지기에서 일반 목적 탐지기로의 전환을 분석합니다. 기계 학습 모델(MLLM)의 발전과 함께 AI 생성 미디어 탐지 분야에서의 혼합 접근 방식에 대한 가능성을 조명하며, 윤리적 및 보안 관련 이슈에 대한 논의도 포함하고 있습니다.

- **Technical Details**: 논문에서는 AI 생성 콘텐츠에 대한 탐지 방법을 비MLLM 기법과 MLLM 기반 기법으로 분류합니다. 특히, MLLM 기반 기법은 다양한 입력 모달리티를 처리할 수 있는 고급 기능을 갖추고 있으며, 이로 인해 허위 콘텐츠 탐지, 설명 가능성, 그리고 지역화와 같은 복합적인 과제를 유연하게 해결할 수 있습니다.

- **Performance Highlights**: 다양한 검증 기준 및 데이터셋에 대한 포괄적인 개요를 제공하며, 연구자들이 AI 생성 미디어 탐지 기술을 발전시키는 데 필요한 기초 자료를 제시합니다. 또한, MLLM 기반 방법의 채택이 증가함에 따라 윤리적 지침을 요약하고, 향후 연구를 위한 방향성을 제시하여 기술 발전이 사회적 영향을 고려해 이루어져야 함을 강조합니다.



### L2GNet: Optimal Local-to-Global Representation of Anatomical Structures for Generalized Medical Image Segmentation (https://arxiv.org/abs/2502.05229)
- **What's New**: L2GNet은 의료 이미지 분석에서 전통적인 CLS, DLS 및 CDLS 기반 모델의 한계를 극복하기 위해 제안된 새로운 아키텍처입니다. 이 모델은 전통적인 self-attention 메커니즘의 대안으로써, 최적 운송(optimal transport) 이론을 활용하여 지역과 글로벌 종속성을 효과적으로 학습합니다. L2GNet은 단순한 가중치 행렬 없이도 discriminative on-the-fly representation learning을 수행할 수 있어 계산 효율성 또한 개선하였습니다.

- **Technical Details**: L2GNet은 인코더, 양자화기(quantizer), L2GMapper 및 디코더로 구성됩니다. 인코더는 세부적인 연속 표현을 추출하고, 양자화 모듈은 벡터 양자화(vector quantization)를 통해 이를 compact discrete form으로 매핑합니다. 이후, L2GMapper가 글로벌 표현을 학습하면서 관련된 지역 간의 장기 종속성을 포착하게 됩니다. 이 과정에서 최적 운송 이론을 활용하여 코드의 정렬을 가능하게 합니다.

- **Performance Highlights**: L2GNet은 Synapse 및 ACDC와 같은 2개의 세분화 벤치마크에서 평가되어 기존의 CLS, DLS 및 CDLS 방법들보다 일관되게 우수한 성능을 보였습니다. 특히 L2GNet은 높은 주석(annotation) 효율성과 일반화 능력을 보여주며, 다양한 스케일에서 해부학적 구조를 효과적으로 포착하는 데 성공하였습니다. 이러한 성능은 의료 이미지 세분화 분야에서 딥 러닝 모델의 활용 가능성을 한층 확대시킵니다.



### VistaFlow: Photorealistic Volumetric Reconstruction with Dynamic Resolution Management via Q-Learning (https://arxiv.org/abs/2502.05222)
- **What's New**: VistaFlow는 기존의 2D 사진 세트를 통해 완전히 상호작용할 수 있는 3D 볼륨 이미지를 재구성할 수 있는 확장 가능한 3차원 이미지 기술입니다. QuiQ라는 새로운 비디오 컨트롤러를 도입하여 렌더링 해상도를 밀리초 단위로 조정하며 지속적으로 높은 프레임 속도를 유지할 수 있게 합니다. 특히, VistaFlow는 통합 CPU 그래픽에서 원활하게 실행되며 모바일 및 저사양 장비에서도 가능성을 보여줍니다.

- **Technical Details**: VistaFlow는 Neural Radiance Fields(NeRFs) 대신 PlenOctree 데이터 구조를 기반으로 복잡한 빛 상호작용을 렌더링합니다. QuiQ는 질감-프레임률 균형을 관리하기 위해 훈련된 자율 정책 네트워크로서, 실시간으로 품질과 프레임 속도를 조절합니다. 이 기술은 고품질 3D 이미지를 생성하는 데 필요한 계산 부담을 줄이고, 소비자 하드웨어에서 1080p 해상도로 초당 100프레임 이상을 구현합니다.

- **Performance Highlights**: VistaFlow는 고급 워크스테이션부터 저렴한 마이크로컨트롤러까지, 다양한 하드웨어에서 고해상도 3D 씬 렌더링의 효율성과 접근성을 향상시킬 잠재력을 가지고 있습니다. PlenOctree를 이용하여 파라미터를 동적으로 조정함으로써 다양한 장치의 성능에 맞춰 렌더링 품질을 조절할 수 있습니다. 비디오 게임 및 가상 현실 애플리케이션에서 실시간 반응성을 보장하며, 사용자 경험을 크게 개선하는 요소로 작용할 수 있습니다.



### History-Guided Video Diffusion (https://arxiv.org/abs/2502.06764)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 Classifier-free guidance (CFG) 기법을 비디오 확산 모델에 확장하여, 가변 길이의 역사(context frames)에 따라 조건화된 비디오 생성을 가능하게 합니다. 기존의 고정 크기 조건화 아키텍처 문제와 CFG 스타일의 역사 드롭아웃 평가의 저조한 성능을 해결하기 위해, Diffusion Forcing Transformer (DFoT)를 제안하였습니다. 이 아키텍처는 유연한 수의 역사 프레임에 대한 조건화를 가능하게 하며, 새로운 역사 안내(History Guidance) 기법도 도입하였습니다.

- **Technical Details**: DFoT는 비디오 생성의 품질과 시간적 일관성을 향상시키기 위해 이론적으로 기반한 훈련 목표를 가지고 있습니다. DFoT에 의해 가능해진 역사 안내 방법은 특정한 형태로 구현되어 비디오 생성 품질을 크게 향상시키고, 시간과 주파수 전반에 걸친 역사 안내 방법은 모션 다이나믹을 추가적으로 강화합니다. 이를 통해 DFoT는 외부 분포 역사에 대한 조합적 일반화를 가능하게 하며, 매우 긴 비디오의 안정적인 롤아웃(roll out)을 지원합니다.

- **Performance Highlights**: 가장 간단한 형태의 역사 안내인 vanilla history guidance는 비디오 생성 품질과 시간적 일관성을 현저히 개선하는 결과를 보여줍니다. 더 발전된 방법은 시간과 주파수에 걸친 역사 안내로, 모션 다이나믹을 증가시키고 외부 역사에 대한 조합적 일반화를 가능하게 합니다. 이러한 향상은 비디오 생성의 질을 높이고, 긴 비디오의 생성에 있어 높은 안정성을 제공합니다.



### evclust: Python library for evidential clustering (https://arxiv.org/abs/2502.06587)
Comments:
          13 pages, 2 figures, Preprint

- **What's New**: 최근 클러스터링(Clustering) 알고리즘의 발전 추세는 클러스터의 불확실성(Uncertainty)을 포착하는 기능을 포함하는 데 초점을 맞추고 있습니다. 본 논문은 Dempster-Shafer 이론(Dempster-Shafer theory)을 활용한 증거 클러스터링(Evidential Clustering)이란 새로운 접근 방식을 소개하고 있습니다. 이를 통해 각 객체의 클러스터에 대한 불확실한 할당을 정량화하는 credal partition 형식을 제시하며, 이는 클러스터 멤버십을 보다 유연하게 처리할 수 있는 장점이 있습니다.

- **Technical Details**: 증거 클러스터링은 신뢰 함수(Belief Function)를 기반으로 하여, 여러 클러스터에 대한 부분적인 멤버십을 허용하는 소프트 클러스터링 방식입니다. Dempster-Shafer 이론에 따라, 각 객체는 하나의 클러스터에만 할당되지 않고, 다양한 클러스터에 대한 믿음의 정도가 할당됩니다. 특히 본 논문에서 소개하는 evclust는 Python 기반의 라이브러리로, 효율적인 증거 클러스터링 알고리즘과 credal partition을 분석하고 시각화하는 도구를 제공합니다.

- **Performance Highlights**: evclust는 클러스터링 알고리즘과 그에 대한 유틸리티 도구를 포함하며, numpy, pandas, matplotlib, scipy와 같은 인기 있는 라이브러리와의 통합이 용이합니다. 본 논문에서는 evclust의 구조와 주요 기능, 성능 지표에 대해 설명하고 있으며, 실험 결과를 통해 evclust의 활용 가능성을 보여줍니다. 이를 통해 사용자는 보다 복잡한 클러스터링 문제를 해결하는 데 도움이 되는 강력한 도구를 활용할 수 있습니다.



### A Survey on Video Analytics in Cloud-Edge-Terminal Collaborative Systems (https://arxiv.org/abs/2502.06581)
- **What's New**: 이번 논문은 클라우드-엣지-단말 협업 시스템(CETC) 내에서 비디오 분석의 최신 동향과 도전 과제를 체계적으로 분석합니다. 특히, 다양한 아키텍처와 처리를 위한 전략을 통합하여 기존 연구에서 다뤄지지 않은 비디오 분석의 전반적인 구조를 제시합니다. 이 연구는 CETC 시스템의 통합 방식과 그에 따른 비디오 분석의 효율성을 중점적으로 다루며, 새로운 접근 방식을 통해 비디오 분석 분야의 공백을 메우고자 합니다.

- **Technical Details**: 논문은 클라우드, 엣지, 그리고 단말 장치에서의 비디오 처리 작업의 분산을 강조하며, 이는 자원 관리 메커니즘과 복잡한 콘텐츠 처리의 변동에 적응하는 데 필수적입니다. 엣지 중심 접근 방식은 on-device processing, edge-assisted offloading, 및 edge intelligence를 활용하며, 클라우드 중심 방법은 복잡한 비디오 이해와 모델 훈련을 위한 강력한 컴퓨팅 능력을 활용합니다. 또한, 하이브리드 비디오 분석에서는 적응형 작업 오프로드 및 자원 인식 스케줄링 기술을 다룹니다.

- **Performance Highlights**: CETC 시스템의 비디오 분석은 교통 모니터링, 자율 주행, 스마트 시티와 같은 다양한 분야에서 중요성을 더욱 부각시키고 있습니다. 이 시스템은 실시간 인사이트 제공을 통해 문제 해결에 기여하며, 대규모 비디오 데이터 전송으로 인한 네트워크 병목 현상을 극복함으로써 전체 시스템의 신뢰성과 효율성을 향상시킵니다. 특히, 최근의 대규모 언어 모델(LLMs)과 다중 모달 통합의 발전은 플랫폼 확장성, 데이터 보호 및 시스템 신뢰성 측면에서 새로운 기회와 도전을 제공합니다.



### Sequence Transferability and Task Order Selection in Continual Learning (https://arxiv.org/abs/2502.06544)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구는 연속 학습(Continual Learning)에서 과제 시퀀스의 전이 가능성(transferability)이 모델 성능에 미치는 영향을 조사하여 악화된 모델 성능을 개선하고자 합니다. 두 가지 새로운 전이 가능성 측정 지표인 Total Forward Transferability(TFT)와 Total Reverse Transferability(TRT)를 제안하며, 이들을 통해 과제 선택 최적화 문제를 해결하는 Heuristic Continual Task Order Selection(HCTOS) 방법을 개발하였습니다. 이 방법은 기존의 무작위 과제 선택 방식보다 더 나은 성능을 보여줍니다.

- **Technical Details**: 연구에서는 먼저 전이 학습(Transfer Learning)에서 성공적인 지표인 전이 가능성 측정 지표를 기반으로, 처리할 수 있는 과제 시퀀스의 전반적인 전이 가능성을 개발합니다. TFT와 TRT는 데이터 의존적인 방식으로 CL 알고리즘의 전이 및 망각(forgotting)을 정량화합니다. 새로운 방법론은 기존 방법에 비해 계산이 용이하며 CL 알고리즘의 평균 정확도와 잘 연관되어 있습니다.

- **Performance Highlights**: 제안된 HCTOS 방법은 보다 나은 학습 정확도를 제공하는 동시에 샘플 크기 및 기본 전이 가능성 측정 지표와 같은 하이퍼파라미터 설정에 강건한 것으로 나타났습니다. 연구 결과, TFT와 TRT 지표는 CL 성능과 밀접한 연관이 있음을 보여주며, 실질적인 CL 문제에 대한 효과적인 해결책을 제시합니다.



### SIREN: Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps (https://arxiv.org/abs/2502.06519)
- **What's New**: SIREN은 다중 로봇 Gaussian Splatting (GSplat) 지도를 등록하기 위해 개발된 알고리즘으로, 초기화나 상호 지도 변환에 대한 카메라 정보나 이미지를 필요로 하지 않습니다. 이 알고리즘은 세 가지 주요 방법을 통해 다중 로봇 GSplat 지도의 정확한 등록 파이프라인을 구현합니다. 첫째, SIREN은 지역 지도에서 피처가 풍부한 영역을 식별하고 등록 문제를 개선하여 초기화 과정을 생략합니다. 둘째, 신뢰할 수 있는 기하학적 최적화를 수행하기 위해 강력한 시맨틱 기능을 활용하여 지역 지도 간의 Gaussian들 간의 후보 대응관계를 판별합니다.

- **Technical Details**: SIREN은 시맨틱 정보를 활용해 고유한 등록 프로세스를 생성하고, Gaussian 기본 요소 간의 비구조적 상대 변환을 위해 기하학적 최적화 문제를 정의합니다. 이후, GSplat 지도의 새로운 뷰 합성을 통해 지역 지도의 변환을 정밀하게 다듬는 절차를 가집니다. 이 과정에서 SIREN은 시맨틱 기반 이미지 필터를 사용해 신뢰할 수 있는 후보 이미지를 식별하여 고정밀 비구조적 변환을 실행합니다.

- **Performance Highlights**: SIREN은 다양한 현실 세계 데이터셋에서 기존의 GSplat 등록 방법과 전통적인 포인트 클라우드 등록 방법에 비해 우수한 성능을 보입니다. 실험 결과, SIREN은 회전 오류에서 약 90배, 변환 오류에서 300배, 비율 오류에서 44배 더 낮은 값을 기록하며, 특히 사족보행 로봇의 매핑 작업에서 뛰어난 성능을 입증했습니다. SIREN의 효과는 고급 로봇 하드웨어 플랫폼에서 종합적으로 검증될 예정이며, 코드와 프로젝트 페이지는 리뷰 과정 후 공개될 것입니다.



### Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation (https://arxiv.org/abs/2502.06516)
Comments:
          29 pages, 11 figures

- **What's New**: 본 연구에서는 Diffusion 모델을 이용하여 소수 샘플을 생성할 수 있는 간단하면서도 강력한 가이드 없는 접근법인 Boost-and-Skip을 제안합니다. 기존의 소수 샘플 생성 방식은 소모적인 계산 리소스를 요구하는 가이드를 사용하지만, 이 방법은 두 가지 최소한의 변경으로 이를 해결합니다. 특히, 분산을 높인 초기화와 타임스텝 건너뛰기를 통해 소수 특성의 발현을 촉진합니다.

- **Technical Details**: Boost-and-Skip는 표준 생성 프로세스의 두 가지 주요 수정을 통해 작동합니다: (i) 분산 증가 초기화, (ii) 타임스텝 스킵. 처음 수정은 초기 생성에서 저밀도 영역에서의 초기화를 유도하는 고유한 노이즈를 사용하며, 두 번째 수정은 초기 단계에서 몇 개의 타임스텝을 건너뛰어 저밀도 초기화의 효과를 높입니다. 이 두 수정은 이론적으로 및 경험적으로 소수 샘플 생성 성능을 현저하게 개선합니다.

- **Performance Highlights**: 실험 결과, Boost-and-Skip은 기존의 소수 생성 방법들과 비교하여 경쟁력 있는 성능을 보여주면서도 현저히 낮은 계산 비용을 요구합니다. 예를 들어, ImageNet 데이터셋에서 본 방법은 이전의 최첨단 방법보다 65% 적은 시간과 4.5배 낮은 메모리 소비로 작동합니다. 이 접근법은 그 단순성 덕분에 매우 확장 가능하여 실제 분류 작업의 데이터 증강에서도 효과적입니다.



### Structure-preserving contrastive learning for spatial time series (https://arxiv.org/abs/2502.06380)
Comments:
          TL;DR: Preserving certain structures of similarity relations in spatio-temporal data can improve downstream task performance via contrastive learning

- **What's New**: 이 연구에서는 공간적으로 특성화된 시계열 데이터를 위한 자기 지도 표현 학습(self-supervised representation learning)에서 구조 보존 규제자(regulariser)를 통합하여, 유사성 관계를 레이턴트 공간(latent space)에서 유지하는 새로운 접근 방식을 제안합니다. 제안된 방법은 변화하는 가중치를 통해 대조 학습(contrastive learning)과 구조 보존의 균형을 동적으로 조절하며, 멀티 변수 시계열 분류 및 교통 예측과 같은 다양한 다운스트림 작업에서 성능을 개선합니다.

- **Technical Details**: 제안된 접근 방식은 대조 학습과 구조 보존을 위한 두 가지 규제자를 도입합니다. 하나는 글로벌 스케일에서 유사성 구조를 유지하는 토폴로지 보존(regulariser)이고, 다른 하나는 로컬 스케일에서의 그래프 기하를 유지하는 규제자입니다. 이러한 규제자는 시계열 데이터에서 유사성 구조를 보존하기 위해 네트워크 아키텍처 및 손실 함수에 대해 다양한 실험을 수행하여 성능을 평가합니다.

- **Performance Highlights**: 제안된 방법은 맥로스코픽 및 마이크로스코픽 교통 예측 작업에서 최첨단 성능을 초과하는 결과를 보여줍니다. UEA 아카이브의 공간 데이터셋에서 평균 분류 정확도가 2.96% 향상되었으며, 맥로스코픽 교통 예측에서는 유속 평균 절대 오차(MAE)가 0.57%, 마이크로스코픽 궤적 예측에선 0.5m 및 1m 반경에서 결측률이 각각 1.87% 및 3.40% 개선되었습니다. 이러한 성과는 제안된 방법이 유사성 구조 보존을 통해 더 정보적이고 유용한 표현을 형성함으로써 가능해졌음을 보여줍니다.



### Many-Task Federated Fine-Tuning via Unified Task Vectors (https://arxiv.org/abs/2502.06376)
Comments:
          10 pages, 6 figures, submitted in IJCAI 2025

- **What's New**: 이 논문은 기존의 파라미터 효율적인 미세 조정(PEFT) 기법을 기반으로 하는 다중 작업 연합 학습(MaT-FL)의 새로운 접근 방식인 MaTU를 소개합니다. MaTU는 클라이언트의 여러 작업을 하나의 통합 모델로 학습할 수 있게 하여 클러스터링이나 클라이언트 전용 가중치 저장이 필요 없는 방법을 제공합니다. 이 방식은 클라이언트의 작업 벡터의 방향을 통해 유사성을 판단하고, 관련 작업 간의 지식 이전을 촉진하는 경량 조절기를 도입합니다.

- **Technical Details**: MaTU는 각 클라이언트의 작업 벡터를 집계하는 새로운 메커니즘을 도입하여, 작업 간의 유사성을 측정하고 모든 작업을 캡슐화하는 통합 작업 벡터를 구성합니다. 이를 위해 '유니파이드' 작업 벡터를 경량 조절기로 보강하여, 유사한 작업 간의 지식 이전을 용이하게 하고 서로 다른 작업의 가중치를 분리합니다. 이러한 방식은 다양한 작업에서의 성능을 보장하는 데 필수적이며, 연합 학습에서의 자원 소모를 줄이는 데 기여합니다.

- **Performance Highlights**: 30개의 데이터셋에서 평가된 결과, MaTU는 기존의 MaT-FL 기법보다 뛰어난 성능을 기록하였으며, 개별 작업의 미세 조정에 비견되는 결과를 보였습니다. 특히, 많은 양의 통신 비용을 절감하면서도 높은 성능을 유지하는 것으로 나타났습니다. 이러한 성과는 여러 클라이언트와 작업이 있을 때 통합 모델을 통해 자원을 효율적으로 사용하는 연합 학습의 중요성을 강조합니다.



### From Pixels to Components: Eigenvector Masking for Visual Representation Learning (https://arxiv.org/abs/2502.06314)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 이미지의 가시 부분에서 마스킹된 부분을 예측하는 방법으로, 기존의 픽셀 기반 마스킹 전략 대신 데이터 변환을 바탕으로 하는 새로운 마스킹 전략을 제안합니다. 주성분 분석(principal component analysis, PCA)을 활용하여 이미지의 주성분을 마스킹하는 방식으로, 이는 이미지의 전역 정보를 더 잘 반영할 수 있는 특징을 지닙니다. 이 접근 방식은 특정 마스킹 비율에 따라 변동성을 조절할 수 있어, 더 의미 있는 표현학습이 가능할 것으로 기대됩니다.

- **Technical Details**: 제안된 방법은 주성분 분석을 통해 변환된 데이터에서 특정 주성분을 무작위로 마스킹하고, 남은 가시 컴포넌트로부터 마스킹된 컴포넌트를 재건하는 방식입니다. 이 과정에서 마스킹된 변동 비율이 모델링 작업의 복잡성을 나타내며, 이는 보다 해석 가능하고 조정이 용이한 하이퍼파라미터로 작용합니다. 또한, 이러한 기법은 이미지를 전역 특성으로 나누어 정보의 중복 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 실험 결과, CIFAR10, TinyImageNet, 그리고 세 가지 의료 데이터셋(MedMNIST)에서 저자들의 방법(주성분 마스크 오토인코더, PMAE)이 기존의 공간 마스킹 방법(MAE)보다 우수한 성능을 나타냈습니다. 특히, 마스킹 비율 하이퍼파라미터의 선택에 덜 민감하다는 점에서도 장점을 보였습니다. 이러한 결과는 주성분 마스킹이 더 의미 있는 고수준의 표현학습을 촉진할 수 있음을 뒷받침합니다.



### Is an Ultra Large Natural Image-Based Foundation Model Superior to a Retina-Specific Model for Detecting Ocular and Systemic Diseases? (https://arxiv.org/abs/2502.06289)
- **What's New**: 이번 연구에서는 의료 분야에서 기본 모델(foundation models, FMs)이 어떻게 변화를 가져오고 있는지를 다루고 있습니다. 특히, RETFound라는 망막 특화 모델과 DINOv2라는 범용 비전 모델을 비교하여 안과 질환 탐지와 시스템 질환 예측 작업에서의 성능을 평가했습니다. 이 연구는 DINOv2의 임상 과제에 대한 적용 가능성을 새롭게 조명합니다.

- **Technical Details**: 연구팀은 1.4백만 개의 자연 이미지와 1.6백만 개의 망막 이미지를 순차적으로 사전 훈련한 RETFound와 1.42억 개의 자연 이미지로 사전 훈련한 DINOv2 모델(대형, 기본, 소형)을 비교했습니다. 이들은 8개의 표준화된 공개 안과 데이터셋과 Moorfields AlzEye 및 UK Biobank 데이터셋에서 성능을 평가했습니다. 특히, DINOv2-large 모델은 당뇨망막병증 감지에서 RETFound보다 우수한 성능을 보였습니다.

- **Performance Highlights**: DINOv2-large 모델은 당뇨망막병증(0.850-0.952)과 다중 클래스 안 질환(0.892) 탐지에서 RETFound보다 뛰어난 성능을 기록했습니다. 반면, RETFound는 심부전, 심근경색, 허혈성 뇌졸중 예측에서 모든 DINOv2 모델보다 우수한 성능을 나타냈습니다. 이러한 결과는 특정 업무에 대한 FM 선택이 임상 성능 최적화에 얼마나 중요한지를 잘 보여줍니다.



### Enhancing Cost Efficiency in Active Learning with Candidate Set Query (https://arxiv.org/abs/2502.06209)
Comments:
          20 pages, 17 figures, 4 tables

- **What's New**: 이 논문에서는 분류를 위한 비용 효율적인 능동 학습(active learning, AL) 프레임워크를 소개합니다. 새롭게 제안된 질의 설계인 candidate set query를 통해, 기존의 AL 질의 방식에서 벗어나 oracle이 모든 가능한 클래스를 검토할 필요 없이 지식 클래스가 포함될 가능성이 높은 후보 클래스 집합으로 좁힙니다.

- **Technical Details**: 이 방법은 conformal prediction을 활용하여 동적으로 작지만 신뢰할 수 있는 후보 집합을 생성하며, 이는 successive AL 라운드에서 모델 개선에 적응합니다. 또한 정보 이득(information gain)이 높으면서도 비용이 낮은 데이터 포인트를 우선하는 획득 함수(acquisition function)를 설계하여 효율성을 극대화합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100 및 ImageNet64x64에 대한 경험적 평가를 통해, 우리의 프레임워크의 효과성과 확장성을 입증했습니다. 특히, ImageNet64x64에서는 라벨링 비용(labeling cost)을 42% 절감하는 성과를 달성했습니다.



### A Data-Efficient Pan-Tumor Foundation Model for Oncology CT Interpretation (https://arxiv.org/abs/2502.06171)
Comments:
          57 pages, 7 figures

- **What's New**: 본 논문에서는 PASTA라는 범종양(cross-tumor) CT 기초 모델을 소개합니다. PASTA는 46개의 종양 관련 작업 중 45개에서 최첨단 성능을 달성하며, 특히 두 번째로 성능이 우수한 모델보다 35개 작업에서 크게 뛰어난 성과를 보입니다. 이러한 성과는 혁신적인 합성 종양 생성 프레임워크인 PASTA-Gen의 개발에 의해 촉진되었습니다.

- **Technical Details**: PASTA-Gen은 3만 개의 CT 스캔을 포함하는 포괄적인 데이터셋을 생성하며, 이 데이터셋은 픽셀 수준에서 주석이 달린 병변 정보를 포함하고 있습니다. 데이터셋은 10개 장기와 5개 양성 병변 유형에 대한 악성 종양을 포괄하고 있으며, 이러한 고품질 합성 데이터는 CT 기초 모델 개발에 있어 오래된 문제인 자료 부족을 해결하는 데 기여합니다.

- **Performance Highlights**: PASTA는 실제 데이터의 양이 적음에도 불구하고 뛰어난 데이터 효율성을 보여줍니다. 이는 다양한 작업에서 성능을 현저히 향상시켜주며, 연구자들과 임상 의사들에게 실질적인 가치를 제공합니다. 합성 데이터셋과 PASTA 기초 모델의 공개는 자료 부족 문제를 효과적으로 해결할 수 있는 방안을 제시합니다.



### Universal Approximation of Visual Autoregressive Transformers (https://arxiv.org/abs/2502.06167)
- **What's New**: 이번 논문에서는 transformer 기반의 foundation model의 근본적인 한계를 조사하며, Visual Autoregressive (VAR) transformer를 포함한 새로운 분석을 제시합니다. VAR는 이미지 생성을 위한 새로운 조정 가능한 코스-투-파인 'next-scale prediction' 프레임워크를 통해 기존 방법들보다 우수한 품질을 보여줍니다. 우리의 주요 기여는 단일 헤드 VAR transformer가 이미지-투-이미지 Lipschitz 함수에 대한 보편적인 근사자임을 증명하는 것입니다.

- **Technical Details**: Transformer 기반 아키텍처는 현대 기계 학습의 경관을 변화시켰으며, self-attention 메커니즘을 통해 데이터의 장기 종속성을 효과적으로 모델링합니다. VAR transformer는 구조화된 이미지 합성을 위해 적응된 변형으로, 높은 품질의 이미지를 더 효율적으로 생성합니다. 이 연구는 VAR transformer와 Flow AutoRegressive (FlowAR) 모델의 보편성에 대한 정량적 분석을 통해, 이들 모델이 복잡한 함수를 근사하는 데 충분한 표현력을 갖추고 있음을 밝힙니다.

- **Performance Highlights**: VAR transformer는 간단한 디자인만으로도 임의의 Lipschitz sequence-투-sequence 기능을 근사할 수 있으며, 이는 VAR 설정에서 고전적인 보편성 결과를 확장합니다. FlowAR 모델도 유사한 근사 능력을 보이고, 두 모델의 상호작용은 생성 모델 설계에 있어 효율성 및 표현력을 동시에 만족할 수 있는 길을 제시합니다. 이로써 효율성과 표현력이 반드시 상반되지 않음을 증명하며, 이러한 기초 연구 결과는 모델 심도, 헤드 수 및 근사 효율성 간의 트레이드오프를 이해하는 데 중요한 기초를 마련합니다.



### Enhanced Hybrid Deep Learning Approach for Botnet Attacks Detection in IoT Environmen (https://arxiv.org/abs/2502.06138)
Comments:
          6 pages

- **What's New**: 이 연구는 IoT 환경에서의 봇넷 공격 탐지에 대한 새로운 접근 방식을 제안합니다. 딥러닝 기법을 활용하여 복잡한 데이터 패턴을 분석하고 학습할 수 있는 모델을 개발하였습니다. 특히, Deep convolutional neural networks, Bi-LSTM, Bi-GRU 및 RNN의 쌓기를 통해 공격을 탐지하는 방식을 선보이며, 이 기술이 보안에 미치는 긍정적인 영향을 강조합니다.

- **Technical Details**: 제안된 모델은 UNSW-NB15 데이터셋을 사용하여 봇넷 공격을 탐지합니다. 모델은 딥러닝 기법을 통합하여 다양한 패턴과 특징을 정교하게 식별하고, 99.76%의 테스트 정확도를 기록하였습니다. 또한, 99.18%의 ROC-AUC 값으로 높은 봇넷 탐지 성능을 보였습니다.

- **Performance Highlights**: 제안된 방식은 기존의 최신 모델들과 비교했을 때 더 높은 성능을 보였습니다. 실험 결과는 이 모델이 복잡한 공격 패턴을 효과적으로 처리할 수 있음을 보여주며, 사이버 보안 절차를 강화할 수 있는 잠재력을 가지고 있음을 나타냅니다.



### Event Vision Sensor: A Review (https://arxiv.org/abs/2502.06116)
- **What's New**: 이번 논문에서는 이벤트 기반 비전 센서(event-based vision sensors)의 최신 기술 발전을 살펴봅니다. 특히, 백조명(BSI) 기술과 웨이퍼 스태킹(wafer stacking) 기법이 성능 향상에 기여한 점에 주목합니다. 이러한 기술들은 노이즈 감소, 해상도 향상, 판독 속도(readout rate) 증가를 가져오며, 실제 응용에 대한 가능성을 높이고 있습니다.

- **Technical Details**: 이벤트 기반 비전 센서는 높은 시간 해상도(high temporal resolution)와 낮은 지연시간(low latency)을 유지하면서도 전력 소비를 최소화할 수 있는 장점을 가지고 있습니다. 이 논문에서는 이러한 센서의 작동 원리와 핵심 기능을 포함한 발전 추세를 리뷰합니다. 특히, 신경형 공학(neuromorphic engineering)에서 최첨단(event-based) 비전 센서 기술로의 발전을 다룹니다.

- **Performance Highlights**: 이벤트 기반 비전 센서는 현재 및 엣지 비전 시스템(edge vision systems)과의 호환성을 향상시켜 상업적 응용의 폭을 넓히고 있습니다. 논문은 적외선 이미징(infrared imaging) 분야에서의 민감도와 이러한 센서가 맞닥뜨리는 도전 과제를 분석하여, 향후 연구와 응용의 방향성을 제시합니다.



### Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization (https://arxiv.org/abs/2502.06061)
Comments:
          61 pages

- **What's New**: 이 논문에서는 연속 흐름 기반 생성 모델의 사용자 정의 보상 함수에 따라 효과적으로 조정할 수 있는 새로운 강화 학습 (RL) 방법인 Online Reward-Weighted Conditional Flow Matching with Wasserstein-2 Regularization (ORW-CFM-W2)을 제안합니다. 이 방법은 보상 그래디언트나 필터링된 데이터 세트에 의존하지 않고, 흐름 매칭 프레임워크에 RL을 통합하여 사용자 정의 보상 함수로 생성 모델을 조정할 수 있도록 설계되었습니다. 또한, 정책 붕괴를 방지하기 위해 Wasserstein-2 거리 정규화를 적용하여 탐색(exploration)과 착취(exploitation) 사이의 균형을 효과적으로 유지할 수 있습니다.

- **Technical Details**: 제안된 방법은 온라인 보상 가중치 기법을 통해 모델이 데이터 매니폴드에서 높은 보상에 우선 순위를 두도록 합니다. 특히 계약 조건에 따라, Wasserstein-2 (W2) 거리 정규화를 통해 정책 업데이트가 안정적으로 유지되며, 이는 fine-tuned 모델과 사전 훈련된 참조 모델 간의 거리를 조절하는 역할을 합니다. 이 논문은 저자들이 제안한 방법의 수렴 특성과 유도된 데이터 분포에 대한 이론 분석을 제공하여 강화 학습 알고리즘 및 KL 정규화와의 관계를 정립합니다.

- **Performance Highlights**: 다양한 실험을 통해 목표 이미지 생성, 이미지 압축 및 텍스트-이미지 정렬과 같은 작업에서 제안된 방법의 유효성을 입증하였습니다. 본 연구의 방법은 최적의 정책 수렴을 달성하며, 보상 극대화와 생성 다양성 유지 간의 조절 가능한 균형을 허용합니다. 이는 강화 학습이 연속 흐름 기반 생성 모델에 효과적으로 적용될 수 있음을 보여주는 중요한 결과입니다.



### Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI (https://arxiv.org/abs/2502.05999)
Comments:
          8 pages, 5 figures, 2 tables

- **What's New**: 이 논문은 어린이, 성인, AI의 그림을 비교하여 시각적 창의성을 정량화하기 위한 새로운 데이터셋과 계산적 프레임워크를 제안합니다. 연구팀은 1338개의 그림을 바탕으로 스타일(style)과 콘텐츠(content)의 두 가지 요소를 분석합니다. 어린이의 그림은 더 많은 요소를 포함하고, AI 그림은 더 높은 잉크 밀도를 가지며, 성인의 그림은 최대 개념 다양성을 드러내는 등의 특징을 밝혔습니다. 이 작업은 인간과 기계의 창의성을 비교하기 위한 최초의 체계적인 기반을 제공합니다.

- **Technical Details**: 발표된 연구에서는 다양한 그리기 기술과 스타일을 평가하기 위해 두 가지 주요 요소인 스타일과 콘텐츠를 구분합니다. 스타일에 대한 측정에서는 잉크 밀도(ink density), 잉크 분포(ink distribution) 및 요소의 수(number of elements)를 정의합니다. 콘텐츠 분석에는 전문가 주석 전문가의 카테고리와 이미지, 텍스트 임베딩(image and text embeddings)을 사용하여 개념적 다양성을 연구하고, 거리 측정(distance measures)을 계산합니다. 데이터 수집은 공립 몬테소리 학교에서 소그룹으로 진행되었으며, AI 모델의 경우 DALL-E를 활용한 작업이 포함되어 있습니다.

- **Performance Highlights**: 결과적으로 연구팀은 전문가와 자동화된 창의성 점수 간의 불일치를 강조하면서 AI와 인간 그림의 창의성 평가에서 중요한 차별점을 발견했습니다. 이 연구는 전문가의 평가와 AI의 평가 방식을 결합함으로써 창의성 평가의 균형 잡힌 관점을 제공합니다. 최종적으로, 이 작업은 다양한 지능형 시스템 간의 창의적 표현 차이를 체계적으로 규명하는 데 기여하여, 시각적 창의성의 기본 계산 원리를 이해하는 데 중요한 발판을 쌓았습니다.



### A Generative Framework for Bidirectional Image-Report Understanding in Chest Radiography (https://arxiv.org/abs/2502.05926)
- **What's New**: MAViLT(다단계 적응형 비전-언어 조정) 프레임워크를 제안하여 흉부 X-레이(CXR) 이해 및 생성을 개선하는 새로운 방법론을 소개합니다. 이 방법론은 임상 데이터 기반의.tokenization 및 위계적 튜닝 전략을 통합하여 정밀한 비전-언어 정렬을 가능하게 합니다. 이를 통해 모델은 진단 보고서를 정확하게 작성하고, 텍스트에서 현실적인 CXR을 합성하며, 비전 기반 임상 질문에 답변할 수 있습니다.

- **Technical Details**: MAViLT는 임상 그래디언트 가중치 토큰화 프로세스와 위계적 미세 조정 전략을 도입하여 기존 기술의 한계를 극복합니다. 비전 텍스트 쌍을 활용한 다단계 훈련을 통해 모델은 다양한 CXR 관련 태스크에서 최적의 성능을 발휘합니다. 또한, 특정 태스크에 적합한 지침 템플릿을 통해 의학 이미징의 미세한 뉘앙스를 포착합니다.

- **Performance Highlights**: MAViLT는 MIMIC-CXR 및 Indiana University CXR 데이터셋에서 CXR-보고서 생성, 보고서-CXR 생성, 비전 질문 응답(VQA) 태스크에서 최상의 결과를 기록하며, 기존 최고 성능 모델들을 초과하는 성능을 보여주었습니다. 또한, 인간 평가를 통해 MAViLT의 임상 관련성과 효용성을 검증하여 실제 의료 적용에서 신뢰성을 제공합니다.



### Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo (https://arxiv.org/abs/2502.05908)
- **What's New**: 이 논문에서는 이미지 처리에서 발생하는 역문제(inverse problem)를 해결하기 위해 새로운 샘플링 방법을 제안합니다. 특히, 시퀀셜 몬테카를로(Sequential Monte Carlo, SMC) 방식을 기반으로 하여 디퓨전 모델의 잠재 공간에서 샘플링을 수행합니다. 이 방법은 디퓨전 모델의 전방 프로세스를 활용하여 추가적인 보조 관측값을 추가하고, 그 후 후방 프로세스에서 SMC 샘플링을 실행하는 방식입니다.

- **Technical Details**: 역문제는 일반적으로 관측된 손상된 신호로부터 깨끗한 신호를 복원하는 문제로, 이미지 블러 제거(image deblurring), 초해상도(super-resolution), 인페인팅(inpainting), 가우시안 노이즈 제거(Gaussian denoising)와 같은 다양한 응용에 적용됩니다. 디퓨전 모델의 경우, 샘플링 과정이 순차적이며, 손상된 이미지에 대한 조건화가 마지막 단계에서만 이루어지기 때문에 샘플링 과정에 어려움이 따릅니다. 이 논문에서는 두 가지 접근 방식을 결합하여 신규 샘플링 절차를 도출합니다.

- **Performance Highlights**: ImageNet과 FFHQ에서 수행된 실험 결과, 제안된 방법이 다양한 역문제 작업에서 경쟁 방법들보다 우수한 성능을 보였습니다. 특히, 제안된 접근법은 자연 이미지 사전(natural image prior)를 고려하여 높은 확률의 실현 가능하고 데이터 일관성 있는 솔루션을 제공하는 데 중점을 두고 있습니다. 이로 인해, 디퓨전 모델을 활용한 역문제 해결의 가능성이 크게 확장되었습니다.



### DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Contro (https://arxiv.org/abs/2502.05855)
Comments:
          The webpage is at this https URL

- **What's New**: 이 논문은 DexVLA라는 혁신적인 프레임워크를 통해 다양한 로봇 형태에서 복잡하고 긴 수명의 작업을 수행하기 위한 시각-언어-행동(vision-language-action, VLA) 모델의 효율성과 일반화 능력을 향상시키는 방법을 제시합니다. DexVLA는 10억 개의 매개변수를 가진 새로운 확산 기반의 행동 전문가(diffusion-based action expert)를 도입하여 다양한 로봇의 행동 학습을 지원합니다. 또한, 심화된 커리큘럼 학습(embodied curriculum learning) 전략을 통해 단계별로 어려운 작업을 효과적으로 학습할 수 있습니다.

- **Technical Details**: DexVLA는 세 가지 중요한 혁신을 через 수행합니다: (1) 10억 개의 매개변수를 지닌 Diffusion Expert를 통해 다양한 형태의 로봇에서 효과적으로 학습할 수 있도록 합니다. (2) 구현 주의(curriculum learning) 전략을 채택하여 간단한 작업부터 시작하여 점차 복잡한 작업으로 나아가는 세 가지 단계(Pre-training, Alignment, Task-specific adaptation)를 구성합니다. (3) 서브스텝(reasoning)으로 주석이 달린 시연 데이터를 포함하여 VLA 모델의 학습 능력을 향상시킵니다.

- **Performance Highlights**: DexVLA는 단일 팔, 양손, 정교한 손, 이동식 로봇 등 다양한 형태에서 여러 작업을 수행하는 데 효과적임을 보여주었습니다. 특히, 단 100시간의 시연 데이터를 기반으로 사전 훈련을 거쳐, 특정 작업에 대한 조정 없이도 높은 성공률을 기록했습니다. 예를 들어, DexVLA는 다림질과 같은 복잡한 작업을 직접적인 지시로 수행할 수 있으며, 100개 이하의 시연으로 새로운 형태에서도 정교한 기술을 배울 수 있는 능력을 갖추었습니다.



### Compressing Model with Few Class-Imbalance Samples: An Out-of-Distribution Expedition (https://arxiv.org/abs/2502.05832)
- **What's New**: 이 논문은 소수의 샘플로 모델 압축을 다루는 주요 문제 중 하나인 클래스 불균형(class imbalance) 문제를 최초로 해결합니다. 제안된 OOD-Enhanced Few-Sample Model Compression (OE-FSMC) 프레임워크는 효과적으로 학습 분포를 재균형(rebalance)하는 데 초점을 맞추고 있으며, 이를 위해 외부 데이터(out-of-distribution, OOD)를 통합합니다.

- **Technical Details**: OE-FSMC 프레임워크는 신뢰할 수 있는 원 데이터와 OOD 데이터 간의 손실(loss)을 균형 있게 유지하기 위해 조합(distillation) 프레임워크와 클래스 의존적 가중치(class-dependent weight)를 도입합니다. 또한, OOD 데이터의 노이즈가 모델 압축 및 파인튜닝에 미치는 영향을 최소화하기 위해 새로운 손실 함수를 정의합니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 통해 실험한 결과, OE-FSMC는 기존의 소수 샘플 모델 압축 방법들과 통합되어 클래스 불균형 문제에 따른 정확도 저하를 효과적으로 완화하는 성능을 보여주었습니다. 이는 소수 샘플 환경에서도 모델의 일반화 능력을 유지하면서 높은 성능을 달성할 수 있음을 의미합니다.



### Image-Based Alzheimer's Disease Detection Using Pretrained Convolutional Neural Network Models (https://arxiv.org/abs/2502.05815)
- **What's New**: 이 연구는 알츠하이머병을 조기에 진단할 수 있는 컴퓨터 보조 진단 시스템을 제안합니다. 특히, 신경 영상(neuroimaging) 기술을 통해 얻은 바이오마커(biomarkers)를 이용하여 알츠하이머병의 클래스를 예측하는 새로운 접근법을 보여줍니다. 이 시스템은 기존의 방법에 비해 진단 정확도를 높이기 위한 목적을 가지고 있습니다.

- **Technical Details**: 제안된 시스템은 딥러닝(deep learning) 기술을 활용하여 이미지 컬렉션에서 관련 시각적 특징을 추출합니다. 여기에서는 VGG16 기반의 모델들을 사용하였으며, 표준 데이터셋과 사전 학습된 모델(pre-trained models)을 통해 실험이 이루어졌습니다. 모델의 성능 평가는 표준 성능 지표(standard performance measures)에 따라 수행됩니다.

- **Performance Highlights**: 실험 결과, VGG16 기반 모델이 최첨단 기술(state of the art)보다 뛰어난 성능을 발휘하는 것으로 나타났습니다. 이러한 결과는 이미지 기반 접근법과 머신러닝 기법이 알츠하이머병 진단에 효과적이라는 것을 입증합니다. 이 연구는 알츠하이머 조기 진단의 새로운 가능성을 제시하며, 향후 연구 및 개발에 기여할 것으로 기대됩니다.



### PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map (https://arxiv.org/abs/2502.05752)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문은 로봇 환경의 효율적인 운영을 위한 새로운 맵 표현 방법을 제안합니다. 제안된 방법은 연속 서명 거리 필드(Continuous Signed Distance Field)와 가우시안 스플래팅 방사 필드를 통합하여, 탄력적이고 컴팩트한 포인트 기반의 암묵적 신경 맵을 형성합니다. 이 방식은 두 필드 간의 기하학적 일관성을 강화하여 서로의 개선을 이루어냅니다.

- **Technical Details**: PINGS라는 새로운 LiDAR-비주얼 SLAM 시스템을 도입했으며, 이 시스템은 다양한 대규모 데이터셋에서 평가되었습니다. PINGS는 포인트 기반 신경 맵을 활용하여 거리 필드 및 방사 필드를 동시에 일관되게 구축하며, 이를 통해 로컬라이제이션(Localization)과 메쉬 재구성이 더욱 향상됩니다. LiDAR와 카메라 데이터의 상호 작용을 통해 고품질의 포토리얼리스틱 렌더링과 기하학적 정확성을 달성합니다.

- **Performance Highlights**: 실험 결과에 따르면 PINGS는 상태 최첨단 방법들과 비교하여 새로운 관점에서 더 나은 포토메트릭(Photometric) 및 기하학적 렌더링 성능을 보였습니다. 또한 방사 필드의 밀집 포토메트릭 단서를 활용하여 더욱 정확한 거리 필드를 구축하고, 이는 결과적으로 보다 개선된 오도메트리 추정(Odometry Estimation) 및 메쉬 재구성으로 이어집니다. 작고 효율적인 신경 포인트 맵을 통해 실시간 고충실도 렌더링이 가능하다는 점도 주목할 만 합니다.



### Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Modeling (https://arxiv.org/abs/2502.05743)
Comments:
          First two authors contributed equally

- **What's New**: 이 연구에서는 생성 작업을 위해 설계된 확산 모델(difussion models)이 자가 지도 방식으로 고품질 표현(representation)을 학습할 수 있는 이유와 시기를 다룹니다. 이를 위해 낮은 차원 데이터 모델(low-dimensional data model)과 후방 추정(posterior estimation)에 기반한 수학적 프레임워크를 개발하였습니다. 이 분석은 이미지 생성의 최종 단계 근처에서 생성(generation)과 표현 품질 사이의 근본적인 균형(trade-off)을 설명합니다.

- **Technical Details**: 연구에서는 저차원 저랭크 가우시안 믹스처(low-dimensional mixture of low-rank Gaussians)를 사용하여 단일 모드 반복 표현 학습(unimodal representation learning dynamics)이 데이터 노이즈 제거(data denoising)와 클래스 지정(class specification) 간의 상호작용(interplay)에서 발생함을 보여줍니다. 또한, 확산 모델에서의 고유한 가중치 공유 메커니즘(weight-sharing mechanism)을 강조하며, 이는 최고 표현 성능(peak representation performance)을 위한 이점과 난잡한 영역에서의 최적화 한계(limitations)를 드러냅니다.

- **Performance Highlights**: 합성 및 실제 데이터셋에 대한 광범위한 실험을 통해 우리의 발견을 검증하였습니다. 이 연구는 다양한 노이즈 스케일(noise scales)에서 확산 기반 표현 학습의 역학에 대한 최초의 이론적 연구를 제공합니다. 제안된 앙상블 방법(ensemble method)은 노이즈 수준 전반에 걸쳐 기능(feature)을 집계하여, 깨끗한 성능(clean performance) 및 레이블 노이즈(label noise) 하에서의 강인성을 향상시키는 데 significant한 개선을 보여줍니다.



### 4D VQ-GAN: Synthesising Medical Scans at Any Time Point for Personalised Disease Progression Modelling of Idiopathic Pulmonary Fibrosis (https://arxiv.org/abs/2502.05713)
Comments:
          4D image synthesis, VQ-GAN, neural ODEs, spatial temporal disease progression modelling, CT, IPF

- **What's New**: 본 논문에서는 4D Vector Quantised Generative Adversarial Networks (4D-VQ-GAN) 모델을 제안하여, 진행성 폐질환인 특발성 폐섬유화증 (Idiopathic Pulmonary Fibrosis, IPF) 환자의 CT 이미지를 생성할 수 있는 기술을 개발했습니다. 이 모델은 초기 단계의 IPF 환자에 대한 미래의 CT 스캔을 예측하여, 효과적인 치료 전략을 수립하는 데 도움을 줄 수 있습니다. 특히, 4D-VQ-GAN은 비정상적인 시간 간격의 3D 이미지를 생성하여 연속적인 질병 진행 경과를 모델링할 수 있습니다.

- **Technical Details**: 4D-VQ-GAN은 두 단계로 이루어진 학습 과정을 거칩니다. 첫 번째 단계에서는 3D-VQ-GAN을 활용하여 CT 이미지를 재구성하고, 두 번째 단계에서는 Neural Ordinary Differential Equations (ODE)를 이용하여 다양한 시간 포인트에서의 잠재 임베딩의 시간적 동역학을 학습합니다. 이러한 접근법을 통해, 입력된 CT 스캔 두 개를 바탕으로 원하는 시간 포인트에 새로운 CT 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 모델의 효용성은 생성된 CT 스캔으로부터 도출된 바이오마커에 대한 생존 분석을 통해 평가되었습니다. 이 분석 결과는 생성된 CT 스캔이 환자의 생존 결과를 신뢰성 있게 예측할 수 있음을 보여주며, 개인화된 치료 계획 수립에 기여할 가능성을 강조합니다.



### Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks (https://arxiv.org/abs/2502.05695)
Comments:
          Submission for possible publication

- **What's New**: 이번 논문은 FFmpeg 기술과 잠재적 확산 모델(latent diffusion models, LDMs)을 통합한 실시간 적응형 비트레이트(video streaming) 비디오 스트리밍을 위한 새로운 프레임워크를 제안합니다. 이 접근 방식은 전통적인 일정 비트 전송(constant bitrate streaming, CBS) 및 적응 비트 전송(adaptive bitrate streaming, ABS)에서 발생하는 네트워크 대역폭 사용 문제, 저장 비효율성, 사용자 경험(quality of experience, QoE) 저하를 해결합니다. LDM을 사용하여 I-프레임을 잠재 공간으로 압축함으로써 시각적 품질을 유지하면서도 저장 공간과 세멘틱(semantic) 전송 절약이 가능하게 하였습니다.

- **Technical Details**: 이 논문에서 제안하는 비디오 스트리밍 방식은 LDM과 FFmpeg를 통합하여 비트레이트를 최적화하고 다양한 프레임 유형을 고려하여 효율적인 압축을 수행합니다. I-프레임을 잠재적 특성(latent features)으로 인코딩하고, 모션 데이터는 메타데이터로 인코딩하여 대역폭을 줄입니다. CNN-GRU를 사용하여 변화하는 네트워크 조건, 미디어 콘텐츠, 사용자 선호도에 맞춰 스트리밍을 최적화하는 적응 비트 전송 메커니즘이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 QoE와 리소스 효율성 측면에서 기존의 ABS 알고리즘인 BOLA, Comyco, MERINA를 초월하는 고품질 비디오 스트리밍을 달성함을 보여줍니다. 이 논문은 5G 및 미래의 포스트 5G 네트워크에서 사용될 수 있는 확장 가능하고 실시간 비디오 스트리밍의 새로운 가능성을 열어줍니다. 특히, LDM이 노이즈가 있는 무선 통신 환경에서도 프레임 간의 시간 일관성을 회복할 수 있도록 도와줍니다.



### Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Mod (https://arxiv.org/abs/2502.05505)
- **What's New**: 이 논문에서는 Private Evolution (PE)라는 새로운 방법을 통해 차등 프라이버시(Differential Privacy, DP) 데이터 생성을 위한 가능성을 탐구하고 있습니다. PE는 전통적인 모델 훈련 방식과는 달리, 기초 모델의 추론 API만 필요로 하여 효율성을 높이고 있습니다. 특히, 이제는 컴퓨터 그래픽 기반 시뮬레이터와 같은 다양한 시뮬레이터를 통해서도 PE를 적용할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: PE는 RANDOM_API와 VARIATION_API라는 두 가지 API를 기반으로 하여 작동합니다. 이 API는 반드시 기초 모델에서 오는 것이 아닙니다. 이 논문에서는 Sim-PE라는 알고리즘을 제안하여, 시뮬레이터를 활용해 DP 데이터를 생성함으로써 기존 PE보다 뛰어난 성능을 보이는 사례들을 보여줍니다. Sim-PE는 이미지를 생성하기 위해 사용될 수 있으며, 두 가지 시나리오(시뮬레이터가 접근 가능한 경우와 접근 불가능한 경우)에서 모두 적용할 수 있습니다.

- **Performance Highlights**: Sim-PE는 다양한 시뮬레이터를 활용하여 PE의 다운스트림 분류 정확도를 최대 3배 향상시키고, FID 점수는 최대 80% 낮추는 성과를 달성했습니다. 예를 들어, MNIST 데이터세트에서는 정확도가 27.9%에서 89.1%로 상승했습니다. 특이하게도, 기초 모델과 약한 시뮬레이터를 결합하는 방법 또한 성능 향상으로 이어지는 결과를 보여주고 있습니다.



### HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation (https://arxiv.org/abs/2502.05485)
Comments:
          We require NVIDIA's approval before proceeding with the release, and we are currently processing it

- **What's New**: 이번 연구에서는 계층적 비전-언어-행동(vision-language-action, VLA) 모델이 표준 단일 구조 VLA 모델보다 비종속 데이터(off-domain data)를 더욱 효과적으로 활용할 수 있음을 제안하고 있습니다. 로봇 데이터 부족 문제를 해결하기 위해, 저렴한 off-domain 데이터를 이용해 로봇 동작 예측을 지원하는 새로운 접근 방식을 개발했습니다. 이는 로봇의 실세계 테스트와 시뮬레이션 간의 도메인 차이를 극복하는 데 기여합니다.

- **Technical Details**: 계층적 VLA 모델에서 고급 VLM (vision-language model)은 주어진 RGB 이미지와 작업 설명에 따라 로봇의 엔드 이펙터 궤적을 나타내는 대략적인 2D 경로를 예측하도록 미세 조정됩니다. 이러한 2D 경로 예측은 저수준의 3D 감지 제어 정책에게 정밀한 조작을 위한 지침으로 제공됩니다. 이러한 구조는 높은 수준의 VLM이 세부적인 행동 예측 부담에서 벗어나도록 하고, 저수준 정책의 복잡한 작업 수준의 추론 부담을 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구에서는 계층적 설계를 통해 고급 VLM이 비종속 미세 조정 데이터와 실제 로봇 테스트 시나리오 간의 큰 도메인 격차를 효과적으로 극복하는 것을 보여주었습니다. 실제 로봇 실험에서는 OpenVLA 대비 각기 다른 7개 일반화 축에서 평균 20%의 성공률 향상을 관찰하였으며, 이는 50%의 상대적인 개선을 의미합니다. 연구팀은 시각적 결과도 함께 제공하고 있습니다.



### Inversion of Magnetic Data using Learned Dictionaries and Scale Spac (https://arxiv.org/abs/2502.05451)
Comments:
          12 pages, 2 figures, 2 tables

- **What's New**: 본 논문에서는 기존의 매개변수화된 규제 기법에 의존하던 전통적인 자기데이터 역전 방법의 한계를 극복하고, 변동 가능한 딕셔너리 학습(dictionary learning) 및 스케일-스페이스(scale-space) 방법을 통합하여 복잡한 지하 특징을 적응적으로 표현할 수 있는 새로운 접근 방식을 제안합니다. 이 방법은 학습된 딕셔너리를 통해 고정된 딕셔너리로는 개념화하기 어려운 복잡한 모델을 근사할 수 있는 유연성을 제공합니다.

- **Technical Details**: 자기 데이터 역전의 수학적 프레임워크는 지하의 자기 감도 모델을 비선형 시스템으로 표현하며, 이 과정에서 고전적인 변동량(variational) 역전법과 스케일-스페이스 기법이 강조됩니다. 특히, 딕셔너리 기반의 역전 방법을 가변적인 딕셔너리 프레임워크로 재구성하여 자기 데이터의 방해 요소를 줄이면서 더 신뢰할 수 있는 해를 제공합니다.

- **Performance Highlights**: 본 연구에서 제안하는 접근 방식은 기존의 변동량 및 딕셔너리 기반 방법에 비해 재구성 정확성과 견고성을 상당히 개선한 결과를 보여줍니다. 특히, 스케일-스페이스 역동성과 학습된 딕셔너리를 결합함으로써 지하 모델 복원 및 노이즈 처리에서 큰 잠재력을 강조합니다. 이러한 결과는 지구물리 탐사, 환경 평가, 광물 탐사와 같은 분야에서의 데이터 기반 자기 데이터 역전의 가능성을 보여줍니다.



### Unsupervised Self-Prior Embedding Neural Representation for Iterative Sparse-View CT Reconstruction (https://arxiv.org/abs/2502.05445)
- **What's New**: 본 논문에서는 Self-prior embedding neural representation (Spener)라는 새로운 비지도 학습 방식의 SVCT(스파스 뷰 컴퓨터 단층 촬영) 재구성을 소개합니다. Spener는 비완전 재구성 결과를 이미지 도메인 프라이어로 활용하여 성능을 향상시키며, 반복 재구성 알고리즘을 통합하여 네트워크의 수렴을 가속화합니다. 또한, 이 방법은 기존의 INR 기반 방법보다 잡음이 있는 시나리오에서도 우수한 성과를 보입니다.

- **Technical Details**: Spener는 지역적인 이미지 특성을 추출하기 위해 이미지 인코더를 활용하고, 이를 통해 INR(Implicit Neural Representation) 네트워크를 최적화합니다. 알고리즘은 PnP-HQS(plug-and-play half-quadratic splitting)와 같은 반복 재구성 알고리즘을 활용하여 각 반복에서 추출한 이미지 프라이어를 최적화에 사용하는 방식입니다. 이로 인해 특히 불안정한 SVCT 재구성 시나리오에서 성능이 크게 향상됩니다.

- **Performance Highlights**: 다양한 CT 데이터 세트를 통해 수행된 실험 결과, Spener는 일반적인 in-domain 데이터에서 지도 학습 방법들과 동등한 성능을 발휘하며, out-of-domain 데이터에서는 이를 초월하는 성능을 기록합니다. 특히 Spener는 잡음이 많은 데이터에서 INR 기반 방법보다 더 우수한 성과를 보여주며, 이로 인해 실용적인 임상 적용 가능성을 높이고 있습니다.



### Diverse Image Generation with Diffusion Models and Cross Class Label Learning for Polyp Classification (https://arxiv.org/abs/2502.05444)
- **What's New**: 이번 연구에서는 대장내시경 검사의 영상 데이터를 활용하여 새로이 개발된 PathoPolyp-Diff 모델을 소개합니다. 이 모델은 텍스트 제어 메커니즘을 통해 병리적 특성과 영상 모달리티에 따라 다양한 합성 이미지를 생성합니다. 특히, 텍스트 프롬프트를 사용하여 합성 이미지를 생성하는 것을 첫 번째로 탐구하는 작업으로, 이는 조기 발견과 정확한 분류에 기여할 수 있습니다.

- **Technical Details**: PathoPolyp-Diff 모델은 교차 클래스 레이블 학습(cross-class label learning) 개념을 도입하여 모델이 다른 클래스의 특성을 학습할 수 있도록 하였습니다. 이러한 접근 방식은 데이터 어노테이션(data annotation)의 부담을 줄이며 고품질의 합성 이미지를 생성하는 데 도움이 됩니다. 이러한 방식으로 생성된 이미지는 병리학적 특성(선종성 및 과형성 용종)을 포괄하며, 다양한 영상 기법(WLI, NBI)을 활용하여 최적의 진단을 지원합니다.

- **Performance Highlights**: 실험 결과, PathoPolyp-Diff가 생성한 이미지를 활용했을 때, 공공 데이터셋에서 균형 정확도가 최대 7.91% 향상되었습니다. 뿐만 아니라, 비디오 수준 분석(video-level analysis)에서 교차 클래스 레이블 학습은 통계적으로 유의미하게 최대 18.33%의 균형 정확도 향상을 보여주었습니다. 이러한 성과는 추가적인 수동 어노테이션 없이 이룬 것으로, 강력한 인사이트를 제공합니다.



### A Novel Convolutional-Free Method for 3D Medical Imaging Segmentation (https://arxiv.org/abs/2502.05396)
Comments:
          technical report

- **What's New**: 이 연구는 완전 컨볼루션이 필요 없는 새로운 변환기 아키텍처를 기반으로 하는 모델을 도입하여 3D 의료 이미지 분할을 수행합니다. 이러한 접근 방식은 두꺼운 슬라이스와 얇은 슬라이스 CT 이미지 간의 도메인 적응 문제를 해결하고, 얇은 슬라이스의 분할 정확도를 향상시키는 데 중점을 두고 있습니다. 또한, 얇은 슬라이스에 대한 다중 의미 분할을 위한 새로운 데이터셋을 제안하여 현재의 의료 이미징 연구에서의 격차를 해소하고자 합니다.

- **Technical Details**: 제안된 모델은 컨볼루션을 완전히 배제하고 변환기 아키텍처 및 자기 주의 메커니즘(self-attention mechanisms)을 기반으로 하고 있습니다. 이 구조는 두꺼운 슬라이스 주석을 기반으로 얇은 슬라이스를 효과적으로 분할하기 위한 공동 손실 기능을 제안하며, 이는 데이터셋 가용성의 한계를 극복할 수 있도록 설계되었습니다. 연구는 다중 의미 분할의 성능을 개선하고 다양한 의료 이미지에 적합할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 전통적인 모델 및 하이브리드 아키텍처에 비해 뛰어난 성능을 보여주었습니다. 특히, 얇은 슬라이스에서의 분할 정확도와 품질을 향상시키는 데 있어 의미 있는 개선을 이룩하였습니다. 이러한 결과는 향후 의료 이미지 분할의 가능성을 새롭게 제시하며, 전통적인 컨볼루션 기반 방법을 보완하는 중요한 이정표가 될 것입니다.



### Multi-Class Segmentation of Aortic Branches and Zones in Computed Tomography Angiography: The AortaSeg24 Challeng (https://arxiv.org/abs/2502.05330)
- **What's New**: 이 연구는 AortaSeg24 MICCAI 챌린지를 통해 첫 번째로 100개의 CTA 볼륨이 주석된 데이터셋을 소개함으로써, 23개의 임상적으로 중요한 대동맥 가지 및 구역에 대한 다중 클래스 세분화 방법을 지원하고자 했습니다. 기존 방법들이 대동맥 세분화를 이분법적으로 처리하고 있어, 이는 대동맥의 복잡한 해부학적 구조를 제대로 반영하지 못했습니다. 이 챌린지는 세계 121개 팀의 참여를 유도하며, 최첨단 프레임워크를 활용한 혁신적 접근법들을 시험하게 했습니다.

- **Technical Details**: 대동맥은 여러 가지와 구역으로 나뉘며, 각 부분의 모양, 방향, 크기가 환자마다 다르게 나타납니다. 이러한 세분화 작업을 위해 AortaSeg24 챌린지를 조직했으며, 이에 대한 평가 지표로는 Dice Similarity Coefficient(DSC)와 Normalized Surface Distance(NSD)를 활용했습니다. 각 참가팀은 다양한 기법을 통해 개별 대동맥 가지 및 구역을 분류하는 방법을 연구했습니다.

- **Performance Highlights**: AortaSeg24 챌린지를 통해 제출된 알고리즘들 중 상위 5개 팀의 접근법이 주목받았습니다. 이들 알고리즘은 정확한 대동맥 세분화를 가능하게 하여, 의료 기기의 선택 및 적절한 스텐트 배치에 기여할 것으로 기대됩니다. 연구 데이터셋, 평가 코드 및 선도적인 방법론이 공개되어 향후 연구 및 임상 적용의 기반이 될 것입니다.



### SEER: Self-Explainability Enhancement of Large Language Models' Representations (https://arxiv.org/abs/2502.05242)
Comments:
          18 pages,5 figures,10 tables

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 설명 가능성을 향상시키기 위한 새로운 방법인 SEER를 제안합니다. SEER는 동일한 개념을 집계하고 서로 다른 개념을 분리하여 LLM의 숨겨진 표현을 명확히 설명할 수 있도록 합니다. 이로 인해 LLM의 출력과 동시에 신뢰할 수 있는 설명을 제공합니다.

- **Technical Details**: SEER는 LLM의 표현 공간에서 개념을 집계하고 분리하는 과정에서, 'black-box' 모듈 없이 자체 설명 기능을 강화합니다. 이 접근 방식은 LLM의 추론 로직을 이해하고 응용 시나리오에서의 신뢰도를 높이는 데 기여합니다. 또한, 최적 수송 이론(optimal transport theory)을 통해 SEER의 LLM의 일반화(generalization) 능력 향상에 대한 이론적 분석을 진행합니다.

- **Performance Highlights**: 실험 결과, SEER는 안전 위험 분류(safety risks classification) 및 해독 작업(detoxification tasks)과 같은 신뢰성과 관련된 작업에서 일관된 개선을 보여 주었습니다. 이러한 자기 설명(self-explained) LLM들은 설명 가능성과 성능에서 최근의 지속적인 개선을 달성하였습니다.



### CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models (https://arxiv.org/abs/2502.05214)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 의료 영상에서의 오판 및 잘못된 임상 소견 식별에 초점을 맞춘 새로운 블랙박스 적대적 공격 기법인 CoRPA(Concept-based Report Perturbation Attack)를 제안합니다. CoRPA는 임상 개념을 활용하여 임상적으로 중요한 영상 및 보고서의 적대적 사례를 생성함으로써, 진단 오류를 반영한 현실적인 상황을 모사합니다. 이를 통해 기존의 일반적인 적대적 공격 기법이 의료 AI에 적합하지 않다는 점을 해결하고자 합니다.

- **Technical Details**: CoRPA는 의료 이미지와 관련된 임상 개념에 기반하여 이미지-보고서 쌍을 위한 개념 벡터를 생성하고, 이 벡터를 의도적으로 변형하여 잘못 식별되거나 누락된 임상 특징을 모사합니다. 이 과정에서 수정된 보고서를 기반으로 텍스트-투-이미지(generative model)를 통해 적대적 의료 이미지를 생성합니다. MIMIC-CXR-JPG 데이터셋을 통해 이러한 기술을 시연하며, 이를 통해 다양한 딥러닝 아키텍처의 경직성을 평가합니다.

- **Performance Highlights**: 실험 결과, CoRPA의 적대적 이미지를 적용했을 때, 기존의 일반적 공격 방식에 비해 특유의 높은 복잡성을 가진 모델들이 상당한 저항력을 잃는 것으로 나타났습니다. 이는 의료 AI 시스템에서 도메인 특화된 취약성 문제를 해결하는 것이 얼마나 중요한지 강조합니다. CoRPA는 다양한 의료 데이터셋에 손쉽게 확장될 수 있어, 임상 환경에서의 신뢰성 있는 AI 모형 개발에 기여할 수 있습니다.



### Safety at Scale: A Comprehensive Survey of Large Model Safety (https://arxiv.org/abs/2502.05206)
Comments:
          47 pages, 3 figures, 11 tables

- **What's New**: 이 논문은 인공지능(AI) 분야의 대형 모델의 안전성에 관한 체계적 조사를 제공합니다. 특히, Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-Training (VLP) 모델과 같은 다양한 모델들의 안전성 취약점과 방어 전략을 정리하였습니다. 대형 모델의 안전성을 확보하기 위한 연구의 필요성과 국제적 협력의 중요성을 강조하고 있습니다.

- **Technical Details**: 대형 모델은 대규모 데이터 세트에서의 사전 학습을 통해 언어 이해, 이미지 생성, 복잡한 문제 해결 등의 작업에서 뛰어난 능력을 보입니다. 이러한 모델들은 적대적 공격(adversarial attacks), 백도어 공격(backdoor attacks), 데이터 중독(data poisoning), 에너지-지연 공격(energy-latency attacks) 등 다양한 안전 위협에 직면해 있습니다. 각 공격 유형에 대한 방어 전략 및 안전 연구를 위한 공통 데이터 세트와 벤치마크를 정리했습니다.

- **Performance Highlights**: 대형 모델의 안전성을 보장하는 것은 비의도적인 시스템 동작을 방지하고 개인 정보를 보호하기 위한 필수 사항입니다. 연구자와 실무자에게 유용한 참조 자료로 기능할 수 있으며, 포괄적인 방어 시스템 및 플랫폼의 지속적인 개발을 촉진하는 데 기여할 것입니다. 안전성 연구 현황을 통해 대형 모델의 발전을 가속화하고 안전한 AI 사용을 유도하는 것이 중요합니다.



### Adversarial Machine Learning: Attacking and Safeguarding Image Datasets (https://arxiv.org/abs/2502.05203)
Comments:
          6 pages, published in Proceedings of the Fourth International Conference on Ubiquitous Computing and Intelligent Information Systems (ICUIS-2024)

- **What's New**: 이 논문은 CNN(Convolutional Neural Networks)이 적대적 공격에 취약하다는 점을 강조하고, 이를 보호하기 위한 방법을 탐구합니다. CIFAR-10, ImageNet, MNIST, Fashion-MNIST와 같은 네 가지 일반적인 이미지 데이터셋에서 CNN을 구현하여 높은 정확성을 달성한 후, Fast Gradient Sign Method (FGSM)를 이용하여 모델의 약점을 평가합니다. 적대적 훈련(Adversarial Training)은 모델의 강인성을 높이는 데 효과적이지만 여전히 완전한 방어는 이루어지지 않았습니다.

- **Technical Details**: 이 연구에서는 CNN 모델의 훈련 및 공격 절차를 자세히 설명합니다. FGSM 공격을 통해 입력 이미지에 미세한 perturbation을 추가하여 모델의 정확성을 감소시키는 방법을 사용하고, 이후 깨끗한 이미지와 적대적 이미지를 포함하여 모델을 재훈련함으로써 방어 메커니즘을 구현합니다. 본 연구는 이러한 접근 방식이 모델의 강인성을 효과적으로 증가시키긴 하지만, 여전히 공격에 취약하다는 점을 지속적으로 확인합니다.

- **Performance Highlights**: 이 연구에서 제안한 방어 방법을 통해 훈련된 모델들은 FGSM 공격에 대한 저항력을 향상시켰습니다. 그럼에도 불구하고 모든 경우에서 완벽한 정확성을 보이지는 않아, 이는 실제 상황에서의 모델 적용 시 더욱 강력한 방어 기법의 필요성을 강조합니다. 따라서, 본 연구는 적대적 공격에서의 모델 성능을 높이고 관련 연구의 필요성을 제기합니다.



### AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting (https://arxiv.org/abs/2502.05176)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 3D 장면 인페인팅의 새로운 접근법인 AuraFusion360을 소개합니다. 이 방법은 Gaussian Splatting을 활용하여 3D 장면에서의 물체 제거와 구멍 채우기를 가능하게 하며, 깊이 인식 기반의 unseen mask 생성, 초기 포인트 배치 방식인 Adaptive Guided Depth Diffusion, 그리고 SDEdit 기반의 세부 향상을 통합합니다. 또한, 360° 미지의 장면 인페인팅을 위한 최초의 포괄적인 데이터셋인 360-USID도 함께 제안합니다.

- **Technical Details**: AuraFusion360은 3D 장면을 생성할 때  다중 관점 정보를 활용하며, 미세한 보정 없이도 현실적인 결과를 제공합니다. 본 방법은 Gaussian Splatting의 효율적인 표기법을 활용하여 다양한 관점에서의 일관성 및 기하학적 정확성을 유지합니다. 특히, Adaptive Guided Depth Diffusion를 사용하여 Geometry가 일치하는 포인트를 새롭게 정의하고, 인페인팅된 영역이 선명하고 고품질로 이어지도록 합니다.

- **Performance Highlights**: AuraFusion360은 기존의 방법들과 비교하여 구체적이고 실용적인 품질 향상을 보여줍니다. 실험 결과에 따르면, 이 방법은 시각적 일관성 및 기하학적 정확성을 유지하며 크게 개선된 성능을 발휘합니다. 본 연구는 3D 인페인팅의 기준을 설정하며, 미래 연구에 대한 새로운 지침을 제시합니다.



### Lost in Edits? A $λ$-Compass for AIGC Provenanc (https://arxiv.org/abs/2502.04364)
- **What's New**: 최근 확산 모델(diffusion models)의 발전은 텍스트 기반 이미지 편집 도구의 발전을 이끌어냈으며, 이를 통해 합성된 콘텐츠에 대한 정밀하고 반복적인 수정이 가능해졌습니다. 이러한 도구가 점점 더 접근 가능해짐에 따라, 콘텐츠의 진위 및 추적성을 보장하기 위한 견고한 귀속(attribution) 방법의 필요성이 강조되고 있습니다. 본 논문에서 제안하는 LambdaTracer는 창작이나 편집 파이프라인에 대한 수정 없이 진짜 출력과 조작된 출력을 효과적으로 식별할 수 있는 새로운 잠재 공간(latent space) 귀속 방법입니다.

- **Technical Details**: LambdaTracer는 텍스트 기반 이미지 편집 도구뿐만 아니라 수동 편집 소프트웨어에서도 효과적으로 사용될 수 있는 프레임워크를 제공합니다. 이 방법은 재구성 손실(reconstruction loss)을 적응적으로 보정하여 다양한 반복 편집 과정에서 효과를 유지하며, InstructPix2Pix 및 ControlNet과 같은 자동화된 도구에 의해 수행되거나 Adobe Photoshop 등의 편집 소프트웨어에 의해 수동으로 수행된 경우 모두 적용 가능합니다. LambdaTracer는 새로운 손실 변환(loss transformation) 접근 방식을 도입하여 중첩된 분포의 분리 가능성을 높이고 다양한 상황에서 귀속 정확도를 개선합니다.

- **Performance Highlights**: 광범위한 실험을 통해 본 방법이 악의적으로 편집된 이미지를 식별하는 데 있어 기존 방법들을 일관되게 초월하는 성과를 나타내었음을 확인하였습니다. LambdaTracer는 생성된 이미지와 편집된 이미지에서 최고의 귀속 정확도를 달성하며, 창작권, 창의성 및 신뢰성을 보호하기 위한 실용적인 해결책을 제공합니다. 이러한 성과는 빠르게 진화하는 AI 생태계에서 지식 재산권을 보장하는 데 중요한 역할을 합니다.



### Éclair -- Extracting Content and Layout with Integrated Reading Order for Documents (https://arxiv.org/abs/2502.04223)
- **What's New**: 이 논문은 Éclair라는 새로운 다목적 텍스트 추출 툴을 소개하며, 복잡한 문서 유형을 처리하기 위해 설계되었습니다. 이 툴은 이미지를 기반으로 서식이 지정된 텍스트를 읽기 순서에 맞게 추출하고, 경계 상자(bounding boxes) 및 대응하는 의미적 클래스(semantic classes)를 제공합니다.

- **Technical Details**: Éclair는 변환기 인코더-디코더(encoder-decoder) 아키텍처를 사용하며, 비전 인코더는 RADIO에서 초기화되어 있습니다. 이 모델은 이미지를 잠재 표현(latent representation)으로 매핑하며, 인코더에서 생성된 정보를 바탕으로 텍스트 토큰을 예측합니다. 이 과정에서 여러 종류의 출력 형식을 지정하기 위해 입력 프롬프트를 사용합니다.

- **Performance Highlights**: 제안된 Éclair 모델은 새로운 벤치마크인 DROBS에서 최신 기술의 정확도(state-of-the-art accuracy)를 달성했습니다. 또한 문서 레이아웃 이해(document layout understanding) 및 LLM 훈련을 위한 고품질 데이터 추출 분야에서도 경쟁력 있는 성능을 입증했습니다.



### Inteligencia artificial para la multi-clasificación de fauna en fotografías automáticas utilizadas en investigación científica (https://arxiv.org/abs/2502.04064)
Comments:
          in Spanish language, XXIV Workshop de Investigadores en Ciencias de la Computación (WICC 2022, Mendoza)

- **What's New**: 이번 연구에서는 자연 환경의 관리에 중요한 야생 동물에 대한 이해를 심화시키기 위한 방법으로 카메라 트랩(camera traps)의 사용을 강조합니다. 아르헨티나 티에라 델 푸에고(Tierra del Fuego)에서 다양한 초식동물(guanacos, cows, sheep)의 숲 이용에 관한 연구가 진행되고 있으며, 이는 자연 생태계의 관리 최적화에 기여하고자 합니다. 그러나 수집된 수백만 장의 이미지에서 정보를 해석하는 데는 한계가 있어, 기존 데이터 저장소의 지식이 활용되지 못하고 있는 상황입니다.

- **Technical Details**: 본 연구는 인공지능의 한 분야인 Neural Networks와 Deep Learning을 활용하여 카메라 트랩으로 얻어진 이미지에서 동물 종을 분류하는 모델을 개발하려고 합니다. 이러한 기술들은 지난 10년 동안 전 세계적으로 이미지 인식(image recognition)에 중대한 기여를 해왔습니다. 이 연구는 대규모 과학 연구의 도전 과제를 해결하는 데 중점을 두며, 동물 식별을 위한 효과적인 방법을 탐구합니다.

- **Performance Highlights**: 카메라 트랩으로 수집된 이미지를 분석함으로써 생태학 및 야생 동물 보호에 대한 새로운 통찰력을 제공합니다. 수집된 데이터를 효과적으로 해석함으로써 생태계 관리의 질을 향상시킬 수 있는 잠재력을 가지고 있습니다. 이러한 방법은 생태적 과정의 이해를 높이고 관련된 야생 지역 관리를 개선하는 데 중요한 기여를 할 것입니다.



New uploads on arXiv(cs.AI)

### MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces (https://arxiv.org/abs/2502.07709)
- **What's New**: 이 논문은 MAGELLAN이라는 메타인지 프레임워크를 소개하여 LLM(대형 언어 모델) 에이전트가 자신의 역량과 학습 진행(Learning Progress, LP)을 온라인으로 예측하고 발전시킬 수 있도록 돕습니다. 이 프레임워크는 목표 간의 의미 관계를 포착하여 샘플 효율적인 LP 추정 및 변화하는 목표 공간에 대한 동적 적응을 가능하게 합니다. MAGELLAN은 LLM 에이전트가 광범위하고 진화하는 목표 공간을 완전히 마스터할 수 있도록 하는 유일한 방법으로, 이를 통해 커리큘럼 학습을 열려 있는 목표 공간으로 확장할 수 있음을 보여줍니다.

- **Technical Details**: MAGELLAN은 LLM 에이전트를 내장하여 목표 간의 의미 관계를 학습하고, 목표 실천 과정에서의 역량 이전을 추적하는 LP 추정기를 자동으로 학습합니다. 이 연구에서는 Little-Zoo라는 텍스트 기반 환경에서 MAGELLAN의 효용을 평가하였으며, 목표 공간의 크기가 증가함에 따라 MAGELLAN의 학습 능력이 어떻게 변화하는지를 분석하였습니다. 또한, MAGELLAN은 학습 중에 새로운 목표가 도입될 때 이러한 목표를 기존 커리큘럼에 원활하게 통합할 수 있는지를 조사하였고, 효율적인 학습 커리큘럼 구성을 가능하게 합니다.

- **Performance Highlights**: MAGELLAN은 정확하고 효율적으로 LP를 근사화하는 능력을 보여주었으며, 이를 통해 전통적인 방법들이 전문가 지식 없이 실패하는 상태에서도 LLM 에이전트가 모든 목표를 마스터할 수 있도록 했습니다. 또한 MAGELLAN은 과거에 보지 못한 목표에 대해서도 LP 추정의 일반화를 성공적으로 이루어내어 변화하는 목표 공간에 대한 빠른 적응을 가능하게 합니다. 목표를 클러스터링하여 전문가 정의 그룹과 유사한 결과를 도출함으로써, 실제적인 학습 환경에서의 성능 향상을 입증했습니다.



### Human Decision-making is Susceptible to AI-driven Manipulation (https://arxiv.org/abs/2502.07663)
Comments:
          Work in progress. Code and data will be made available via this https URL

- **What's New**: 본 연구는 AI 시스템이 개인의 의사결정을 어떻게 조작할 수 있는지를 조사하였으며, 그 결과 인간의 심리학적 취약성을 이용한 AI의 잠재적 위험성을 강조합니다. AI 시스템의 사용이 증가함에 따라, 사용자에게 어떤 영향이 미칠 수 있는지를 분석하며 윤리적 안전장치의 필요성을 제기하고 있습니다. 이 연구는 AI의 긍정적 유도와 해로운 조작의 경계를 구별하는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 연구는 무작위 통제 실험 방법론을 사용하였으며, 233명의 참가자를 대상으로 세 가지 AI 에이전트와의 상호작용을 분석했습니다: 중립 에이전트(Neutral Agent, NA), 조작 에이전트(Manipulative Agent, MA), 전략 강화 조작 에이전트(Strategy-Enhanced Manipulative Agent, SEMA). 참가자들은 재정적 결정(예: 구매) 및 감정적 결정(예: 갈등 해결)과 같은 두 가지 주요 맥락에서 AI 시스템의 영향을 받고, 이들의 선호도 변화 패턴을 관찰하였습니다. 이 연구 결과는 MA 및 SEMA와 상호작용한 참가자들이 NA와 비교했을 때 해로운 선택으로의 이동 비율이 통계적으로 유의미하게 높다는 것을 보여줍니다.

- **Performance Highlights**: 재정 결정의 경우, MA 및 SEMA와 상호작용한 참가자들은 각각 61.4% 및 59.6%의 해로운 선택으로의 이동을 보인 반면, NA 그룹에서는 28.3%에 불과했습니다. 감정적 결정에서도 MA 및 SEMA와의 상호작용이 해로운 선택으로의 이동 비율이 각각 42.3% 및 41.5%로, NA 그룹의 12.8%에 비해 유의미한 차이를 보였습니다. 이러한 결과는 AI 시스템의 조작 능력이 실제 의사결정 맥락에서 심각한 영향을 미칠 수 있음을 나타내며, 사용자의 자율성 보호를 위한 윤리적 책임의 필요성을 강조합니다.



### SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models (https://arxiv.org/abs/2502.07644)
Comments:
          16 pages. arXiv admin note: text overlap with arXiv:2404.04306

- **What's New**: 이 논문에서는 이더리움(Ethereum) 스마트 계약의 안전성과 준수성을 보장하기 위해 새로운 도구인 SymGPT를 소개합니다. SymGPT는 자연어 이해(Natural Language Understanding) 기술과 상징적 실행(Symbolic Execution)의 형식적 보장을 결합하여, ERC(이더리움 요청 사항) 규칙을 자동적으로 검증합니다. 기존의 방법들이 효과적이지 않은 부분을 보완하며, 특히 ERC 규칙 준수를 위한 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: 연구진은 세 개의 널리 사용되는 ERC 표준에서 132개의 ERC 규칙을 분석하여 그 내용, 보안적 의미, 자연어 설명을 조사했습니다. 연구 결과를 바탕으로, LLM을 사용하여 ERC 규칙을 정의된 EBNF 문법으로 변환하고, 형식화된 규칙에서 위반 시나리오를 나타내는 제약 조건을 합성합니다. 이후 상징적 실행을 통해 이러한 위반을 탐지하는 구조를 설계했습니다.

- **Performance Highlights**: SymGPT는 실제 세계의 4,000개 계약에서 5,783개의 ERC 규칙 위반을 식별했으며, 그 중 1,375건은 재정 자산을 훔칠 수 있는 명확한 공격 경로를 포함하고 있습니다. 이는 SymGPT의 효과성을 입증하며, 6개의 자동화된 기술 및 전문 보안 감리 서비스보다 뛰어난 성능을 보여주었습니다. 따라서 SymGPT는 현행 스마트 계약 분석 방식에 비해 상당한 우위를 점하고 있습니다.



### NatureLM: Deciphering the Language of Nature for Scientific Discovery (https://arxiv.org/abs/2502.07527)
Comments:
          81 pages

- **What's New**: 이번 논문에서는 다양한 과학 도메인을 통합하여 활용할 수 있는 Nature Language Model(이하 NatureLM)을 소개합니다. 기존의 foundation 모델들과 달리, NatureLM은 여러 과학 도메인에서 수집한 데이터로 사전 훈련되어 있습니다. 이 모델은 작은 분자, 단백질, RNA 및 재료를 텍스트 지시사항에 따라 생성하고 최적화하는 데 탁월한 성능을 보여줍니다.

- **Technical Details**: NatureLM은 Transformer decoder 아키텍처를 기반으로 하며 1430억 개의 토큰으로 구성된 데이터셋을 사용하여 훈련되었습니다. 이 모델은 각종 과학 데이터를 효과적으로 처리할 수 있도록 설계되어 있으며, DNA, RNA, 단백질 등의 서열 데이터를 특징으로 합니다. NatureLM은 다양한 크기(10억, 80억 및 467억 파라미터)로 개발되어, 모델 크기가 증가할수록 성능이 향상되는 경향을 보입니다.

- **Performance Highlights**: NatureLM은 다양한 과학 작업에서 최첨단 성능을 보여줍니다. 특히, SMILES-to-IUPAC 번역 및 레트로신합(Retrosynthesis) 작업에서 두드러진 성과를 거두며, 텍스트 지침을 따르는 생성/설계를 성공적으로 수행합니다. 또한, 모델 크기가 커질수록 성능이 증가해 22개 작업 중 18개에서 뚜렷한 개선을 확인했습니다.



### Harnessing Language's Fractal Geometry with Recursive Inference Scaling (https://arxiv.org/abs/2502.07503)
Comments:
          18 pages, 9 figures

- **What's New**: 본 논문은 Recursive INference Scaling (RINS)이라는 새로운 방법론을 소개하며, 이는 언어 모델의 추론 시간을 효과적으로 확장하는 데 기여합니다. RINS는 고유한 모델 아키텍처와 훈련 예산을 유지하면서도 성능을 향상시킬 수 있다는 점에서 독창적인 접근법입니다. 또한, 기존의 언어 작업을 넘어 다중 모드 시스템에서도 성능 개선을 이루어내는 것으로 평가됩니다.

- **Technical Details**: RINS는 모델 재귀성을 기반으로 하며, 언어의 자기 유사적 구조를 활용하여 기존의 네트워크의 초기 부분을 반복 적용하는 방법으로 기능합니다. 이를 통해 추론 성능을 향상시킬 수 있으며, 복잡한 매개변수 공유 방식보다 더 높은 성능을 보입니다. 또한, RINS의 확률적 변형인 stochastic RINS는 성능을 더욱 향상시키는 동시에 추론 시간 동안 계산 비용을 최소화할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: RINS는 SigLIP-B/16과 같은 멀티모달 시스템에서 성능 향상에 성공하였으며, 0-shot ImageNet 정확도를 77.3%에서 79.6%로, CIFAR100에서는 70.3%에서 80.7%로 개선되었습니다. RINS는 이러한 성능 향상 외에도 데이터 스케일링 법칙을 도출하여 비대칭 성능 한계를 개선하는 데 기여하고 있습니다. 최종적으로, 모델의 매개변수 수를 증가시키지 않으면서 정확도를 높이는 효과적인 방법임을 입증하였습니다.



### URECA: The Chain of Two Minimum Set Cover Problems exists behind Adaptation to Shifts in Semantic Code Search (https://arxiv.org/abs/2502.07494)
- **What's New**: 본 연구에서는 최저 엔트로피 문제의 한계를 극복하기 위해 새로운 클러스터링 알고리즘인 URECA를 제안합니다. URECA는 분리된 표현들 간의 관계를 활용하여 클러스터를 동적으로 갱신합니다. 본 연구는 엔트로피 최소화와 비슷한 문제로서, 두 개의 최소 집합 덮개 문제 사이의 연관성을 밝혀내고 이를 통해 클러스터링의 근본 메커니즘을 탐구합니다.

- **Technical Details**: URECA는 세 단계로 구성된 클러스터링 알고리즘으로, 초기화, 업데이트 및 재귀 과정을 포함합니다. 이 시스템은 Thresholdly-Updatable Stationary Assumption을 통한 동적 모델링을 적용하여, 데이터 간의 관계를 더 정확하게 반영합니다. 또한, 시뮬레이션 트릭을 활용하여 클러스터 업데이트를 효율적으로 수행하며, 이는 가중치의 간단한 계산을 통해 이루어집니다.

- **Performance Highlights**: URECA는 다양한 유형의 변화에 대해 빠른 적응을 가능하게 하고, CoSQA에서 최신 성능(State-of-The-Art performance)을 달성했습니다. 실험 결과, URECA는 다양한 프로그래밍 언어와 변화에 대한 일관된 성능 향상을 보여주었습니다. 전반적으로, URECA는 변화된 환경에서도 효과적으로 작동함을 입증하였습니다.



### Eliciting Rational Initial Weights in Gradual Argumentation (https://arxiv.org/abs/2502.07452)
- **What's New**: 이 논문은 weighted argumentation frameworks에서 각 주장을 초기 가중치(initial weight)로 연결하는 전통적인 접근 방식의 한계점을 설명합니다. 특히, 사람들이 초기 가중치를 정확히 정의하는 데 어려움을 겪는 문제와 여러 주장을 고려할 때 가중치와 수용 가능도를 혼동하는 경향을 지적합니다. 이에 대한 해결책으로 각 주장의 수용 가능성을 간격(interval)으로 명시할 수 있는 elicitation pipeline을 제안합니다.

- **Technical Details**: 이 연구에서는 elicited values와 argumentation framework의 구조를 사용하여 각 주장의 최종 수용 가능도를 조정하는 방법을 제안합니다. 초기 가중치와 최종 수용 가능도를 동시에 반영할 수 있도록 사용자로부터 하나의 간격을 수집합니다. 이러한 간격이 합리적(rational)이지 않을 경우, 초기 값으로부터 최소한의 변경으로 합리적인 값을 찾는 방법도 탐구합니다.

- **Performance Highlights**: 논문에서는 제안된 알고리즘의 구현 및 평가 결과를 제공하며, 사용자가 시스템을 경험할 수 있도록 인터페이스를 제공합니다. 논문에서 언급된 알고리즘은 미래의 연구와 관련된 논의에도 기여할 것으로 기대됩니다. 이러한 방법은 argumentation에 대한 더 직관적인 이해를 가능하게 하여, 복잡한 논의나 토론을 더욱 명확히 할 수 있을 것입니다.



### Approximating Human Strategic Reasoning with LLM-Enhanced Recursive Reasoners Leveraging Multi-agent Hypergames (https://arxiv.org/abs/2502.07443)
- **What's New**: 저희 연구는 LLM(대형 언어 모델)을 기반으로 한 역할 기반 다중 에이전트 시스템을 개발하여 복잡한 재귀적 이유를 수행할 수 있는 에이전트에게 전략적 상호작용을 평가할 수 있는 프레임워크를 제공합니다. 개발한 플랫폼은 다양한 에이전트를 호스팅할 수 있으며, LLM을 활용해 더 정교한 의사결정을 가능하게 합니다. 또한 기존 k-레벨 이론의 한계를 보완하는 새로운 심화 평가법인 κ를 소개하였습니다.

- **Technical Details**: 본 연구에서는 하이퍼게임(hypergame) 모델을 통해 계층적인 신념(hierarchical beliefs)을 활용하여 LLM이 통합된 재귀적 이유를 평가했습니다. 이 모델은 플레이어 각자의 시각을 모델링하여 비대칭적인 인식을 포착할 수 있도록 돕습니다. 재귀적 이유는 에이전트가 서로의 신체적 및 인지적 상태를 고려하는 능력으로 정의되며, 이를 통해 더 정교한 전략적 사고를 분석할 수 있습니다.

- **Performance Highlights**: 실험 결과, 인공지능 기반의 이유자는 기존 경제 모델 및 인간 데이터와 비교하여 인간 행동을 근접하게 재현하고 최적의 해결책에 도달하는 측면에서 성능이 우수한 것을 보여주었습니다. LLM을 활용한 에이전트는 기존 모델과의 비교에서 향상된 성과를 보였으며, 이는 구조적 복잡성이 어떻게 인공지능의 전략적 추론을 증대시킬 수 있는지를 잘 설명합니다.



### Towards a Formal Theory of the Need for Competence via Computational Intrinsic Motivation (https://arxiv.org/abs/2502.07423)
Comments:
          6 pages excluding references

- **What's New**: 이번 연구는 자기결정 이론(Self-Determination Theory, SDT)에서 다루는 '역량의 필요성(need for competence)' 개념을 인공지능(AI) 분야의 강화 학습(reinforcement learning, RL) 모델을 통해 구체화하고자 합니다. 이로 인해 SDT와 동기 심리학 이론의 개발에 기여할 수 있는 새로운 이론적 기반을 제시합니다. 특히, 역량의 필요성이 여러 모호하게 정의된 측면들로 구성되어 있다는 사실을 강조하고 각 측면에 대응하는 정형화를 제안합니다.

- **Technical Details**: 연구에서는 역량이라는 개념이 알고리즘으로 변환되면서 이론을 다루는 새로운 시각을 제공한다고 설명합니다. 각 역량 측면들이 기존의 RL 형식과 어떻게 일치하는지를 위주로 논의를 진행하며, 이러한 형식들이 역량의 필요성을 명확히 하는 데 어떻게 기여할 수 있는지를 탐구합니다. 마지막으로, 이 연구는 동기 심리학 연구에서 활용할 수 있는 AI 문헌의 기존 구현, 시뮬레이터 및 평가 메트릭스의 풍부한 자원을 공개합니다.

- **Performance Highlights**: 향후 연구에서는 제안된 형식이 실질적인 결과를 생성하는지를 실험적으로 검토할 수 있는 기회를 제공합니다. 특히, 인간과 기계 모두에 대한 경험적 연구가 필요하다는 점을 부각하며, 이를 통해 다양한 분야 간 협력을 촉진할 수 있음을 강조합니다. 이러한 접근은 기존 심리학 이론의 복잡성을 드러내고 이론 발전 사이클을 지원하는 데 기여할 것입니다.



### LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters! (https://arxiv.org/abs/2502.07374)
- **What's New**: 이번 연구에서는 Large Language Model (LLM)이 데이터 효율적인 Supervised Fine-Tuning (SFT)와 파라미터 효율적인 Low-Rank Adaptation (LoRA)을 통해 Long Chain-of-Thought (Long CoT) 추론을 효과적으로 학습할 수 있음을 보여줍니다. 특히, 17,000개의 Long CoT 학습 샘플만으로도 Qwen2.5-32B-Instruct 모델이 수학 및 코딩 벤치마크에서 상당한 성과를 달성했습니다. 이를 통해 Long CoT의 구조가 학습 과정에 미치는 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 Long CoT의 목록 및 파라미터 효율적인 학습 방법으로 LoRA를 적용하여 17,000개의 샘플만으로도 LLM이 효과적으로 추론 능력을 향상할 수 있음을 보여줍니다. Qwen2.5-32B-Instruct 모델은 OpenAI의 o1-preview 성능과 비교할 수 있는 수준에 도달하였으며, 이는 모델이 intermediate thoughts를 반영하고 수정하는 능력을 잘 학습했음을 나타냅니다. 우리가 수행한 연구에서는 모델의 성능이 개별 추론 단계의 내용보다 Long CoT의 구조에 더 민감하게 반응함을 발견했습니다.

- **Performance Highlights**: 모델은 수학 및 코딩 문제에서 여러 벤치마크에서 뛰어난 성능을 보여줬으며, AIME 2024에서는 56.7%의 정확도로 +40.0%의 개선이 이루어졌습니다. 또한, OlympiadBench와 Math-500에서도 각각 +12.7%와 +6.0%의 성장을 보였습니다. 이러한 결과는 Long CoT 구조의 중요성과 학습 과정에서의 효율성을 강조하며, 다음 세대 추론 모델 훈련에 대한 중요한 통찰력을 제공합니다.



### KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems (https://arxiv.org/abs/2502.07350)
- **What's New**: 이번 논문에서는 Knowledge-Aware Bayesian Bandits (KABB)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 멀티 에이전트 시스템의 조정(cordination)을 향상시키기 위해 의미적 이해(semantic understanding)와 동적 적응(dynamic adaptation)을 활용합니다. KABB는 비용 문제를 해결하면서도 멀티 에이전트 조정에서 높은 성능을 유지하도록 설계되었습니다.

- **Technical Details**: KABB의 핵심 혁신은 세 가지입니다: 첫째, 심층적인 의미적 이해를 위한 3차원 지식 거리 모델(three-dimensional knowledge distance model), 둘째, 연속적인 전문가 최적화를 위한 이중 적응 메커니즘(dual-adaptation mechanism), 셋째, 효율적인 전문가 선택을 위한 지식 기반 톰슨 샘플링(Knowledge-aware Thompson Sampling) 전략입니다. 이러한 기술들은 KABB가 더 깊고 인간적인 이해를 기반으로 하는 조정 프로세스를 가능하게 합니다.

- **Performance Highlights**: KABB는 멀티 에이전트 조정에서 최적의 비용-성능 균형을 달성함을 보여줍니다. 광범위한 평가를 통해 KABB가 높은 성능을 유지하면서도 상대적으로 낮은 계산 요구사항을 가지고 있다는 것을 입증하였습니다. 이는 비용이 큰 대규모 언어 모델을 대체할 수 있는 유망한 대안을 제공합니다.



### Coarse Set Theory: A Mathematical Foundation for Coarse Ethics (https://arxiv.org/abs/2502.07347)
Comments:
          31 pages, 2 figures

- **What's New**: 이 논문은 Coarse Ethics (CE)라는 개념을 수학적 기초를 바탕으로 정립하는 것을 목표로 하고 있습니다. 이를 위해 Coarse Set Theory (CST)를 소개하며, 합리적인 평가를 위한 수학적 체계를 구축했습니다. 이전 연구에서는 이러한 개념에 대한 명확한 구조를 제시하지 않았으나, 본 연구는 이를 해결하고자 합니다.

- **Technical Details**: CST에서는 타입이 완전하게 정렬된 집합을 통해 coarse sets를 정의합니다. 여기에는 항등성(reflexivity), 비대칭성(antisymmetry), 이행성(transitivity) 등의 다양한 공리를 설정하고 있습니다. 또한, coarse mappings를 도입하여 세부 정보를 Coarse한 표현으로 변환하는 방법을 정의하고, Kullback-Leibler (KL) divergence를 사용하여 정보 손실을 측정합니다.

- **Performance Highlights**: CST는 실제 채점 시스템에 적용될 수 있으며, 이론적 공식화와 실증 분석을 통해 그 가능성을 보여줍니다. 이 연구는 fairness, interpretability 및 decision-making의 상충 관계를 보다 체계적으로 탐구할 수 있는 기반을 제공합니다. 이로 인해 Coarse Ethics에 대한 이해와 그 응용 가능성이 증가하게 됩니다.



### When More is Less: Understanding Chain-of-Thought Length in LLMs (https://arxiv.org/abs/2502.07266)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 추론의 길이가 LLM의 추론 정확도에 미치는 영향을 분석합니다. CoT 길이가 길어질수록 성능이 초기에 개선되지만 결국에는 감소하는 복잡한 관계를 관찰했습니다. 또한, 모델의 능력과 과제의 난이도에 따라 최적의 CoT 길이가 존재한다는 이론적 근거를 제시하고, 이를 바탕으로 실제 데이터셋에 대한 실험 결과도 공유합니다.

- **Technical Details**: CoT 추론은 복잡한 문제를 더 작고 관리 가능한 하위 문제로 나누어 해결하는 Divide-and-Conquer 전략으로 설명됩니다. 실험을 통해 CoT의 길이가 LLM의 성능에 미치는 영향을 명확히 확인했으며, 이전 단계의 추론을 통합하면 오차 수정 능력이 향상됨을 밝혔습니다. 우리는 최적의 CoT 길이를 설정하기 위해 Length-filtered Vote 방식을 제안하며, 이를 통해 과하게 길거나 짧은 CoT의 효과를 완화할 수 있습니다.

- **Performance Highlights**: 모델의 성능은 CoT의 길이에 따라 유의미한 차이를 보입니다. 최적 길이 CoT로 훈련된 작은 모델이 임의 길이로 훈련된 큰 모델보다 나은 성능을 보이는 결과가 나타났습니다. 이러한 실험 결과는 CoT 길이가 모델 성능에 미치는 중요한 영향을 강조합니다.



### Monte Carlo Tree Diffusion for System 2 Planning (https://arxiv.org/abs/2502.07202)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 논문에서는 Monte Carlo Tree Diffusion(MCTD)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 확산(diffusion) 모델의 생성력을 Monte Carlo Tree Search(MCTS)의 적응형 탐색 능력과 결합하여 플래닝을 보다 효율적이고 확장 가능하게 만듭니다. MCTD는 Denoising을 트리 구조로 재구성하고, Trajectory를 반복적으로 평가 및 수정할 수 있는 방식을 사용합니다.

- **Technical Details**: MCTD는 세 가지 주요 혁신을 기반으로 합니다. 첫째, Denoising을 트리 롤아웃 프로세스로 재구성하여 반자기 회귀적(causal) 플래닝을 가능하게 합니다. 둘째, 메타 행동으로서의 Guidance Levels를 도입하여 탐색과 활용의 균형을 동적으로 조정합니다. 셋째, 비용이 많이 드는 전진 모델 롤아웃 없이도 효율적으로 Trajectory 품질을 추정하기 위해 Fast Jumpy Denoising을 시뮬레이션 메커니즘으로 활용합니다.

- **Performance Highlights**: 실험 결과, MCTD는 긴 수평 작업(long-horizon task)에서 기존의 방법들보다 우수한 성능을 나타내며, 증가하는 테스트 시간 계산(TTC)에 따라 솔루션의 질이 향상됩니다. 이 연구는 MCTS와 확산 모델의 장점을 결합하여 플래닝의 정확성을 효과적으로 증대시킬 수 있는 방법을 제시합니다.



### Bag of Tricks for Inference-time Computation of LLM Reasoning (https://arxiv.org/abs/2502.07191)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 과정에서의 계산 방법을 탐색하여 추론 성능을 향상시키는 기술적 혁신을 다루고 있습니다. 특히, 기존의 접근 방식에서 탈피해 후보 솔루션 생성 및 보상 메커니즘을 최적화하는 데 중점을 두고 있습니다. 또한 아블레이션 연구를 통해 그간 간과되었던 기법들이 실제 성능 향상에 기여할 수 있음을 보입니다.

- **Technical Details**: 이 연구에서는 다양한 추론 작업에 대한 여러 추론 시간 계산 전략을 비교 분석했습니다. 특히, 각기 다른 LLM 모델들에 대해 20,000시간 이상의 A100-80G GPU 리소스를 사용하여 1,000회 이상의 실험을 수행하고, 후보 솔루션 생성 시의 하이퍼파라미터 및 보상 모델의 중요성을 강조합니다. 연구는 특정 하이퍼파라미터 조정이 추론 성능에 미치는 영향이 크다는 점을 입증했습니다.

- **Performance Highlights**: 연구 결과, 후보 솔루션 생성 시 온도(tuning temperature) 조정이나 다른 최적화 기술들이 추론 성능을 최대 5% 끌어올리는 것으로 나타났습니다. 여섯 가지 대표적인 추론 시간 계산 방법을 여덟 가지의 추론 작업에 대해 체계적으로 평가하여 표준 기준을 마련했습니다. 이러한 성과는 향후 연구의 기초를 다지는 데 기여할 것으로 기대됩니다.



### Understanding LLMs' Fluid Intelligence Deficiency: An Analysis of the ARC Task (https://arxiv.org/abs/2502.07190)
Comments:
          22 pages, 9 figures, accepted by NAACL 2025 main conference

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 유동 지능(fluid intelligence)을 평가하기 위해 ARC 작업을 분석합니다. 연구자들은 LLMs가 솔루션을 찾지 못하는 세 가지 주요 한계, 즉 기술 조합 능력(skill composition), 추상 입력 형식에 대한 익숙하지 않음, 왼쪽에서 오른쪽으로 디코딩(left-to-right decoding)의 본질적 어려움을 규명하였습니다. 이러한 연구 결과는 LLMs의 성능 저하 원인을 더 깊이 이해하고, 향후 개선 방안을 모색하기 위한 기초 자료로 활용될 것입니다.

- **Technical Details**: 이 논문은 ARC 작업을 통해 LLMs의 유동 지능을 평가합니다. LLMs의 성능을 점검하기 위해 각 ARC 작업이 정의하는 고유한 변환 규칙을 파악하고, 주어진 입력-출력 그리드 쌍으로부터 이 규칙을 유도하는 경험적 실험을 진행했습니다. ARC와 ARAOC 기준에 따라 LLMs의 성능을 측정하고, 특히 2D 입출력 그리드를 활용하여 모델의 성능을 분석했습니다.

- **Performance Highlights**: 연구 결과 GPT-4o 같은 최신 언어 모델은 19%의 정확도로 ARC 작업을 수행할 수 있으며, 이는 평균 인간 성능의 약 75%에 미치지 못합니다. 이러한 낮은 성능은 LLMs의 내부 구조적 및 작업 조합 능력의 한계를 드러내며, LLMs가 추상적인 패턴을 이해하고 적용하는 데 어려움을 갖고 있음을 보여줍니다. 실험 결과는 LLMs 개선의 방향성을 제시하고 있으며, LLMs의 추상적 추론 능력 향상을 위한 추가 연구의 필요성을 강조합니다.



### Interactive Data Harmonization with LLM Agents (https://arxiv.org/abs/2502.07132)
- **What's New**: 이 논문은 다양한 출처의 데이터를 통합하는 데이터 조화(data harmonization)의 필요성을 강조하며, 전문가가 데이터를 조화할 수 있도록 돕는 agentic 데이터 조화의 개념을 소개합니다. 이를 위해 LLM 기반 추론과 상호작용 가능한 사용자 인터페이스, 데이터 조화를 위한 기본 모음(primitive)으로 구성된 Harmonia라는 시스템을 제안합니다. 이 시스템은 임상 데이터 조화 시나리오에서 재사용 가능한 파이프라인을 생성하는 데 도움을 줍니다.

- **Technical Details**: Harmonia는 대화형 데이터 조화 파이프라인을 자동으로 생성하기 위해 LLM(대형 언어 모델)과 상호작용하는 사용자 인터페이스를 결합합니다. 연구자들은 임상 데이터에 대해 상이한 표준 형식으로 데이터를 조화하기 위해 각열의 의미적 동등성을 확인해야 하며, 이는 데이터의 스키마 일치를 포함합니다. 특히, Histologic_grade와 Histologic_Grade_FIGO와 같은 속성이 다른 용어를 사용하더라도 의미적으로 동등하다는 점을 강조하고, 이들을 통합하여 하나의 표로 만드는 작업이 필요합니다.

- **Performance Highlights**: Harmonia는 데이터 조화 과정을 자동화하여 연구자들이 수동으로 여러 단계를 수행하는 부담을 덜어줍니다. LLM을 활용한 새로운 접근법은 데이터 조화 작업에서의 정확성을 높이고, 결과적으로 재현 가능성을 증대시킬 수 있습니다. 이 혁신적인 시스템은 임상 연구와 같은 복잡한 분야에서의 데이터 활용도를 크게 향상시킬 것으로 기대됩니다.



### Autonomous Deep Agen (https://arxiv.org/abs/2502.07056)
- **What's New**: 이 기술 문서는 Deep Agent라는 고급 자율 AI 시스템을 소개합니다. 이 시스템은 복잡한 다단계 작업을 관리하기 위해 새로운 계층적 작업 관리 아키텍처를 기반으로 개발되었습니다. Deep Agent는 최신 Hierarchical Task DAG (HTDAG) 프레임워크를 통해 고수준 목표를 관리 가능한 하위 작업으로 동적으로 분해하면서 의존성 및 실행 일관성을 유지합니다.

- **Technical Details**: Deep Agent는 복잡한 작업을 모델링하기 위해 Hierarchical Task DAG (HTDAG)를 사용합니다. 이 DAG 기반 설계는 고수준 목표를 상호 연결된 하위 작업의 네트워크로 동적으로 분해하면서, 사용자의 개입이나 새로운 정보가 발생할 경우 실시간으로 그래프를 수정할 수 있게 합니다. 또한, Deep Agent는 사용자 및 컨텍스트 인식 결정을 지원하여 다음에 어떤 작업을 실행할지 결정할 때 현재 워크플로우 상태와 이전 작업 결과, 사용자 피드백을 고려합니다.

- **Performance Highlights**: Deep Agent는 새로운 API와 도구를 자율적으로 개발하여 유사한 작업을 처리할 때 LLM 추론 오버헤드를 감소시킵니다. 또한, Prompt Tweaking Engine (PTE)을 통해 지침을 수정하여 LLM의 추론 정확성과 안정성을 크게 향상시킵니다. 마지막으로, Closed-loop Autonomous Feedback Learning (AFL) 기술을 통해 지속적인 개선과 적응이 이루어지며, 복잡한 작업을 자율적으로 관리할 수 있는 산업 적용 가능성을 한층 높입니다.



### Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents (https://arxiv.org/abs/2502.06975)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 에피소드 메모리(epsodic memory) 프레임워크를 제안하며, LLM 에이전트가 지속적으로 학습하고 장기 기억을 유지하는 방안을 다룬다. 생물학적 시스템에서의 에피소드 메모리를 차용하여 LLM의 효율적인 장기 작동을 가능케 하는 다섯 가지 핵심 특성을 중심으로 구성되어 있다.

- **Technical Details**: 에피소드 메모리는 장기 저장(long-term storage), 명시적 추론(explicit reasoning), 단일 노출 학습(single-shot learning) 등 총 다섯 가지의 메모리 특성으로 구분되며, 이들 특성은 LLM 에이전트의 맥락 감응 행동을 지원하는 데 필요한 요소들이다. 특히 이 메모리 시스템은 상황에 따라 다르게 작용하는 정보 저장 방식을 제공, LLM이 복잡한 환경에서 적응할 수 있도록 한다.

- **Performance Highlights**: 기존의 LLM 메모리를 개선하기 위한 방법들이나 접근 방식은 여전히 부족하며, 에피소드 메모리를 활용한 통합된 연구 방향이 필요하다. 이 논문은 LLM들이 장기적으로 효과적으로 작동할 수 있도록 하는 새로운 메모리 시스템의 중요성을 강조하고, 이를 통해 진정한 장기 에이전트의 개발에 기여할 수 있는 로드맵을 제시하고 있다.



### Breaking Down Bias: On The Limits of Generalizable Pruning Strategies (https://arxiv.org/abs/2502.07771)
Comments:
          28 pages, 9 figures, 1 table

- **What's New**: 이 논문에서는 모델 프루닝(model pruning)을 활용하여 LLM(대규모 언어 모델)이 인종적 편견을 어떻게 인식하고 있는지를 분석합니다. 연구 결과, 프루닝이 편견을 감소시키는 효과적인 방법일 수 있지만, 간극이 커질수록 그 효과가 감소한다는 점이 밝혀졌습니다. 또한 인종적 편견이 일반 개념으로 충분히 대표되지 않으며 많은 경우 특정 맥락에 따라 다르다는 것을 제안하여, 더 일반적인 완화 전략이 효과적이지 않을 수 있음을 주장합니다.

- **Technical Details**: 연구에서는 Llama-3-8B-Instruct 모델을 사용하여 인종적 편견을 분석하였으며, 주요 결과로는 뉴런 기반의 프루닝이 전체 어텐션 헤드를 프루닝하는 방법보다 더 나은 성능을 보인다는 사실이 발견되었습니다. 또한 프루닝 전후의 평가 세트 간의 유사성이 높을 경우 평균적으로 편견이 거의 0에 가깝게 줄어들지만, 세트 간의 맥락 차이가 커질수록 그 성능이 급격히 저하된다고 보고합니다. 이 연구는 결국 인종적 편견은 특정 도메인에서만 나타나는 개념이라는 점을 강조합니다.

- **Performance Highlights**: 편견을 줄이기 위한 모델 프루닝의 효과는 교육 세트와 평가 세트의 유사성에 크게 의존하며, 이는 편견 수정 방법의 한계에 대한 중요한 통찰력을 제공합니다. 특히, 금융 의사결정과 상업적 거래의 편견을 제거하려는 경우 40%의 편견 감소에 불과한 성과를 보여주었습니다. 이름 연관을 통한 편견의 평가와 관련된 통계 데이터는 이 모델이 인종적 요소에 따라 차별적인 응답을 생성하는 경향이 있음을 재확인시키며, 일반화된 완화 전략보다는 특정 맥락에 맞춘 조정이 필요하다고 결론짓습니다.



### Polynomial-Time Approximability of Constrained Reinforcement Learning (https://arxiv.org/abs/2502.07764)
- **What's New**: 이번 연구는 일반 제약이 있는 마르코프 결정 과정(Constrained Markov Decision Processes, CMDPs)의 유사성에 대한 계산 복잡도를 다룹니다. 주요 기여는 여러 종류의 제약을 만족하는 최적 정책을 찾기 위한 다항식 시간 $(0,	heta)$-가감형 다기준 근사 알고리즘을 설계한 것입니다. 이 알고리즘은 거의 확실(nearly-sure), 우연(chance), 기대(expectation) 제약을 포함한 여러 계산 가능한 제약에 대한 최적근사 보장을 달성합니다.

- **Technical Details**: 연구진은 CMDPs의 해법을 MDP, 일반 비용 기준, 예산 벡터를 정의하는 세 가지 요소를 통해 설명합니다. 또한, 결정론적 또는 확률적 정책을 구분하여 해결해야 할 목표 정책 클래스를 설정합니다. 알골의 생성 방법은 주로 주어진 제약을 단순화하고, 상태 공간을 인공 예산으로 증대시키는 방식입니다.

- **Performance Highlights**: 연구 결과, 제안된 알고리즘은 여러 제약조건 하에서 다항식 시간 내에 유사 솔루션을 제공하며, 특히 우연 제약과 다중 기대 제약 하에서의 정책을 다룰 수 있는 첫 번째 결과로 자리매김합니다. 더욱이, 제안된 방법은 연속 상태 프로세스에서도 적용 가능하여 이는 이론적으로 의미가 큽니다. 최종적으로, 연구는 제약 조건 처리의 복잡성 문제를 해결하는 데 기여하며, 제약 강화 학습(complex constrained reinforcement learning) 분야에서의 오랜 연구 문제에 대한 해답을 제시합니다.



### An Advanced NLP Framework for Automated Medical Diagnosis with DeBERTa and Dynamic Contextual Positional Gating (https://arxiv.org/abs/2502.07755)
- **What's New**: 이 논문은 의료 진단을 향상시키기 위한 새로운 자연어 처리(NLP) 프레임워크를 제안합니다. 고급 데이터 증강(data augmentation), 특징 추출(feature extraction), 분류(classification) 기법을 통합하여 보다 다양한 데이터 세트를 생성하도록 합니다. 이 접근 방법의 주요 특징은 역 번역(back-translation)을 사용하여 다양한 패러프레이즈(paraphrased) 데이터셋을 만드는 것입니다.

- **Technical Details**: 제안된 모델은 Disentangled Attention이 있는 Decoding-enhanced BERT(DeBERTa)를 활용하고, 동적 맥락 위치 게이팅(Dynamic Contextual Positional Gating, DCPG)을 통해 위치 정보의 영향을 맥락에 따라 조정합니다. 이는 높은 품질의 텍스트 임베딩(text embeddings)을 생성하는 데 도움을 줍니다. 또한, Attention 기반 피드포워드 신경망(ABFNN)을 사용하여 가장 관련 있는 특징에 집중하여 의사결정의 정확도를 향상시킵니다.

- **Performance Highlights**: 이 아키텍처는 증상, 임상 노트 및 기타 의료 텍스트의 분류에 적용되어 의료 데이터의 복잡성을 다루는 능력을 입증합니다. 제안된 NLP 프레임워크는 99.78%의 정확도, 99.72%의 재현율(recall), 99.79%의 정밀도(precision), 99.75의 F1 점수를 기록하여 뛰어난 성과를 보여줍니다. 이러한 성능 지표는 의료 텍스트 분류에서 예외적인 정확성과 신뢰성을 제공하며, 기존 방법보다 우수함을 강조합니다.



### Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension (https://arxiv.org/abs/2502.07752)
- **What's New**: 이 논문은 메모리 요구사항이 적고 빠른 수렴을 달성하는 대규모 언어 모델(LLMs)을 위한 효율적인 최적화기 설계에 관한 연구이다. 저자는 구조화된 Fisher 정보 행렬(FIM) 근사를 통해 기존의 효율적인 최적화기를 재구성할 수 있음을 보여준다. 이 논문은 Row and Column Scaled SGD(RACS)와 Adaptive low-dimensional subspace estimation(Alice)라는 새로운 메모리 효율적인 최적화기를 도출하며, 이들의 효과성을 실험을 통해 입증한다.

- **Technical Details**: 저자들은 оптимizers와 경량화된 구조의 FIM 근사를 활용하여 효과적인 최적화기를 설계하는 방법을 제안한다. RACS는 기존 SGD와 유사한 메모리 요구사항을 가지고 있으며, Alice는 Eigen-Adam의 저순위 확장으로 구현된다. 이 프레임워크는 더 일반적인 구조를 통해 FIM 근사를 개선함과 동시에 메모리와 계산 비용을 줄일 수 있는 방법을 논의한다.

- **Performance Highlights**: 실험 결과, RACS와 Alice는 기존의 메모리 효율적인 최적화기 및 Adam과 비교할 때 우수한 성능을 발휘하였다. 특히, Alice는 Adam에 비해 2배 이상의 빠른 수렴 속도를 보여주었으며, RACS는 1B LLaMA 모델에서 강력한 성능을 입증하였다. 이러한 결과는 대규모 언어 모델의 효과적인 훈련을 위한 새로운 최적화기 설계의 중요성을 강조한다.



### PFedDST: Personalized Federated Learning with Decentralized Selection Training (https://arxiv.org/abs/2502.07750)
- **What's New**: 이 논문은 Personalized Federated Learning with Decentralized Selection Training (PFedDST) 프레임워크를 소개합니다. PFedDST는 장치 간의 의사소통 점수를 기반으로 동료를 선택하여 모델 훈련을 최적화합니다. 이를 통해 장치의 개인화와 안정적인 훈련 과정을 강화할 수 있습니다.

- **Technical Details**: PFedDST는 각 클라이언트가 모델의 차원성을 유지하면서 효율적인 집계 및 전략적 의사소통을 진행할 수 있도록 합니다. 클라이언트는 현재 학습 상황에 적합한 동료를 선택하기 위해 전략적 점수 전략을 사용하여 피어와 교류합니다. 이 과정에서 피어 간의 의사소통을 최적화하고 특징 추출 능력을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과 PFedDST는 데이터 이질성이 큰 환경에서 특히 효과적임을 입증했습니다. PFedDST는 다양한 최첨단 방법과 비교해 신속한 수렴과 더 높은 모델 정확도를 보여주었습니다. 기존의 방향성 DFL 방법과 구별되는 점은 점수 기반 동료 선택과 부분 동결 훈련을 결합하여 의사소통 효율성을 극대화하는 것입니다.



### WHODUNIT: Evaluation benchmark for culprit detection in mystery stories (https://arxiv.org/abs/2502.07747)
- **What's New**: WhoDunIt라는 새로운 데이터세트를 소개하여, 대형 언어 모델(LLM)의 귀납적 추론 능력을 내러티브(narrative) 맥락에서 평가합니다. 이 데이터세트는 공개 도메인에서 가져온 추리 소설과 단편 소설로 구성되어 있으며, LLM이 이야기를 이해한 후 범인을 식별하도록 도전합니다. 또한, 캐릭터 이름의 다양한 변형을 통해 모델의 견고성을 평가하고, 추론 정확성에 미치는 프롬프트(prompt)의 영향을 조사합니다.

- **Technical Details**: 이 데이터세트는 공공 도메인에서 저작권이 만료된 책을 활용하여 준비되었으며, 주로 Project Gutenberg에서 이야기 자료를 확보했습니다. 500편 이상의 미스터리 및 탐정 이야기를 선정하였으며, 각 소설은 식별 가능한 범인이 존재하도록 설계되어 있습니다. 각 프롬프트 스타일과 캐릭터 이름의 대체를 통해, 모델이 단순한 이름 인식이 아닌 내러티브 기능에 기반해 추론할 수 있는 능력을 테스트합니다.

- **Performance Highlights**: 실험 결과, LLM은 변형되지 않은 텍스트에 대한 신뢰할 수 있는 수행 보여주지만, 특정 이름 대체가 있을 경우 정확성이 떨어지는 경향을 보였습니다. 특히, 잘 알려진 실존 또는 허구의 캐릭터 이름으로 대체했을 때 모델의 성능이 변동했습니다. 이 데이터세트는 공개적으로 사용 가능하여, 향후 LLM의 추론 능력과 복잡한 내러티브 이해 능력을 개선하는 데 기여할 것입니다.



### Next Block Prediction: Video Generation via Semi-Auto-Regressive Modeling (https://arxiv.org/abs/2502.07737)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 Next-Block Prediction (NBP)이라는 새로운 반자기회귀(semi-autoregressive) 프레임워크를 제안합니다. 기존의 Next-Token Prediction (NTP) 방식의 한계를 극복하기 위해 비디오 생성을 위해 비단일 토큰이 아닌 균일하게 분해된 블록 단위로 생성 과정을 전환합니다. 이를 통해, 동일 블록 내의 모든 토큰이 다음 블록의 해당 토큰을 동시 예측할 수 있게 됩니다.

- **Technical Details**: NBP 프레임워크는 블록 내에서 양방향(attention) 주의를 사용하여 강력한 공간적(spatial) 의존성을 포착 가능하게 하며, 다수의 토큰을 병렬로 예측하여 생성 단계를 유의미하게 줄입니다. 이 방식은 기존 AR 모델의 한계를 극복하며, FVD(Frechet Video Distance) 기준으로 UCF101에서 103.3, K600에서 25.5를 기록했습니다. 모델 파라미터 수를 700M에서 3B로 확대하면서 생성 품질이 크게 개선되었음도 입증되었습니다.

- **Performance Highlights**: NBP 모델은 128x128 해상도를 가진 비디오 프레임을 초당 8.89장 생성할 수 있으며, 이는 기존 NTP 모델보다 11배의 속도 향상을 의미합니다. UCF101에서 FVD 점수가 103.3에서 55.3으로 감소하였으며, K600에서는 25.5에서 19.5으로 개선되었습니다. 이러한 성과는 NBP가 효율적이면서도 고품질의 비디오 생성을 제공함을 나타냅니다.



### EdgeEar: Efficient and Accurate Ear Recognition for Edge Devices (https://arxiv.org/abs/2502.07734)
Comments:
          Submitted to IEEE FG 2025

- **What's New**: 이 논문에서는 EdgeEar라는 경량 모델을 소개하고 있습니다. EdgeEar는 하이브리드 CNN-Transformer 아키텍처를 기반으로 하여 리소스 제약이 있는 디바이스에서도 효과적으로 작동하도록 설계되었습니다. 기존의 첨단 모델에 비해 파라미터 수를 50배 줄여 200만 개 이하로 유지하면서도 경쟁력 있는 정확도를 가지고 있습니다.

- **Technical Details**: EdgeEar는 Split Depth-wise Transpose Attention (SDTA) 모듈 내의 특정 선형 레이어를 Low Rank Linear (LoRaLin) 레이어로 대체하여 설계되었습니다. 이러한 접근은 모델의 복잡성과 계산 비용을 줄이는 동시에 인식 정확도를 유지합니다. EdgeEar는 512x512x512 차원의 귀 임베딩을 생성하며, 총 1.98M의 파라미터를 가지고 있습니다.

- **Performance Highlights**: Unconstrained Ear Recognition Challenge (UERC2023) 벤치마크에서 EdgeEar는 가장 낮은 Equal Error Rate (EER) 점수를 달성했습니다. EER은 0.143이며, Area Under Curve (AUC)는 0.904, Rank-1 (R1) 정확도는 0.929를 기록했습니다. 이 결과는 귀 생체 인식을 보다 효과적으로 채택할 수 있는 가능성을 보여줍니다.



### Economics of Sourcing Human Data (https://arxiv.org/abs/2502.07732)
- **What's New**: AI의 발전은 인간이 생성한 데이터에 크게 의존해왔으나, 대규모 언어 모델의 기능이 향상되면서 이러한 데이터의 품질과 무결성이 위협받고 있다는 주장을 하고 있습니다. 이 논문은 AI가 인간의 참여를 필요로 하는 데이터 수집 시스템의 설계 방식에서의 근본적인 결함을 드러내고 있습니다. 데이터 수집 시스템을 재고하여 외부 인센티브 대신 기여자의 내재적 동기를 중시하는 방향으로 나아가야 한다고 제안합니다.

- **Technical Details**: 현재 사용되고 있는 데이터 수집의 두 가지 주요 소스는 인간 주석과 인터넷 내의 원시 데이터입니다. 그러나 대규모 언어 모델과 AI의 발전으로 인해 오히려 인간이 생성한 데이터의 부족이 우려되고 있습니다. 이 논문에서는 데이터 수집 시스템의 결함을 분석하고, 인간의 동기를 이해하여 보다 효과적인 데이터 수집 방법을 모색하고 있습니다.

- **Performance Highlights**: 기존 데이터 수집 플랫폼은 품질과 양의 균형을 맞추는 데 어려움을 겪고 있으며, 이는 시스템 설계 선택의 결과입니다. 내부적 동기와 외부적 인센티브의 균형을 맞추는 것이 중요하며, 지속 가능한 데이터 제공을 위해서는 인간의 내재적 동기를 증진해야 한다는 점을 강조합니다.



### Verifying LLM-Generated Code in the Context of Software Verification with Ada/SPARK (https://arxiv.org/abs/2502.07728)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 생성한 코드의 신뢰성을 보장하기 위한 새로운 접근법으로, SPARK 프로그래밍 언어를 활용한 형식적 소프트웨어 검증을 제안합니다. 저자들은 Marmaragan이라는 도구를 개발하여 LLM이 기존 프로그램에 SPARK 주석을 생성하도록 하여 형식적 검증이 가능하게 하는 방법을 탐구합니다. 이 연구는 LLM의 창의적인 잠재력과 안정적인 코드의 필요성 사이의 격차를 해소하기 위한 것입니다.

- **Technical Details**: 형식적 소프트웨어 검증(Formal Software Verification)은 소프트웨어 프로그램이 특정한 형식적 사양 또는 특성에 대한 정확성을 입증하는 과정으로, 수학적 방법을 사용합니다. Marmaragan은 GNATprove 도구를 통해 SPARK 주석을 생성하고 형식적 검증을 수행하도록 설계되었습니다. 이 도구는 OpenAI API와 통합되어 사용 가능하며, 정확한 주석 생성을 위한 다양한 매개변수를 설정할 수 있습니다.

- **Performance Highlights**: Marmaragan은 SPARK 프로그램을 기준으로 한 벤치마크에서 50.7%의 정확도로 올바른 주석을 생성하는 성과를 보였습니다. 이러한 결과는 LLM과 형식적 검증의 결합 가능성을 보여주는 기초를 마련하며, 코드 신뢰성과 안전성을 높일 수 있는 방향으로의 향후 연구에 기여할 것으로 기대됩니다.



### TMLC-Net: Transferable Meta Label Correction for Noisy Label Learning (https://arxiv.org/abs/2502.07721)
- **What's New**: 이번 연구에서는 TMLC-Net이라는 새로운 전이 메타 학습자(Transferable Meta-Learner)를 소개합니다. 이 시스템은 다양한 데이터셋과 모델 아키텍처에 적용할 수 있는 범용 레이블 수정 전략을 학습합니다. 기존의 메타 학습 방식들과 달리 TMLC-Net은 기존 모델의 전반적인 재학습이나 미세 조정 없이도 효과적으로 작동할 수 있도록 설계되었습니다. 따라서, 노이즈 레이블 문제를 해결하는 데 있어 실용적인 해결책을 제공합니다.

- **Technical Details**: TMLC-Net의 핵심 구성 요소는 세 가지입니다: (1) Normalized Noise Perception, 훈련 역학을 포착하고 정규화하여 분포 변화를 처리합니다; (2) Time-Series Encoding, LSTM을 사용하여 샘플 통계의 시간적 변화를 모델링합니다; (3) Subclass Decoding, 학습된 표현을 기반으로 수정된 레이블 분포를 예측하여 보다 정보에 기반한 수정을 수행합니다. 이러한 구조를 통해 기존 방법들의 주요 한계를 극복하고 전송 가능성과 탄력성을 제공합니다.

- **Performance Highlights**: TMLC-Net은 다양한 노이즈 유형과 수준을 가진 기준 데이터 세트에 대한 광범위한 실험을 통해, 기존 최첨단 방법들과 비교해 우수한 정확도 및 노이즈에 대한 강인성을 보여주었습니다. 또한, TMLC-Net의 전이 가능성을 분석하여 새로운 데이터셋과 노이즈 조건에 대한 적응력을 입증하며, 노이즈 환경에서의 견고한 딥러닝 솔루션으로서의 가능성을 제시합니다.



### SoK: A Classification for AI-driven Personalized Privacy Assistants (https://arxiv.org/abs/2502.07693)
Comments:
          Work in progress

- **What's New**: 최근 몇 년 동안 개인화된 개인정보 비서(AI-driven PPAs)가 AI 기술에 기반하여 개발되었습니다. 이러한 비서는 사용자가 개인정보 관련 결정을 내리는 데 도움을 주며, 이러한 비서의 특성과 기술 그리고 결정의 정확성에 대한 체계적인 연구가 부족하다는 점이 지적되었습니다. 본 논문은 과거 10년간의 연구를 종합하여 AI-driven PPAs에 대한 지식을 체계적으로 정리하고, 주요 연구 결론을 제시하고 있습니다.

- **Technical Details**: AI-driven PPAs는 사용자의 개인 데이터를 포함한 데이터를 처리하기 위해 AI 기술을 사용합니다. 법적인 요구사항 측면에서, 비서가 사용자에 의해 설치되고 운영될 경우 사용자는 데이터 통제자로 행동할 수 있으며, 개인 용도로 사용할 경우 GDPR의 적용을 받지 않을 수 있습니다. 이 논문에서는 AI-driven PPAs의 아키텍처, 시스템 컨텍스트, 사용된 AI 유형, 데이터 소스 및 결정 유형에 대해 폭넓은 분류를 제공합니다.

- **Performance Highlights**: 본 연구를 통해 AI-driven PPAs에 대한 최초의 분류체계를 제안하며, 관련 문헌을 기반으로 한 정량적 인사이트를 제공합니다. AI-driven PPAs의 설계 및 개발에 대한 혁신적인 권장 사항을 제시하며, 기존 연구에서 발견된 격차와 도전 과제를 강조합니다. 향후 AI-driven PPAs 개선을 위한 연구 방향도 제안하고 있습니다.



### A Unifying Framework for Causal Imitation Learning with Hidden Confounders (https://arxiv.org/abs/2502.07656)
- **What's New**: 이 논문은 숨겨진 혼란 변수(hidden confounders)를 포함하는 인과 모방 학습(causal Imitation Learning, IL)을 위한 일반적이고 통합적인 프레임워크를 제안합니다. 이 프레임워크는 전문가가 관찰할 수 있는 혼란 변수와 전문가와 IL 알고리즘 모두에게 숨겨진 혼란 소음을 포함한 두 가지 유형의 혼란 변수를 다룹니다. 또한, 시간에 따라 변동하는 전문가가 관찰 가능한 숨겨진 변수를 도입하여 유연성을 더했습니다.

- **Technical Details**: 제안된 알고리즘인 DML-IL은 도구 변수가 있는 회귀(instrumental variable regression)를 활용하여 조건부 모멘트 제약(Conditional Moment Restrictions, CMRs)을 해결하고 정책을 학습합니다. 이 프레임워크에서 IL은 경로 이력을 도구로 활용하여 정책을 학습하는 문제로 재구성될 수 있으며, 이는 경제학 및 인과 추론(econometrics and causal inference)에서 잘 연구된 문제입니다. 이로 인해 알고리즘이 이식성 간극(imitation gap)에 관한 이론적 보장을 포함하도록 설계될 수 있습니다.

- **Performance Highlights**: 모의 환경(toy environment)과 여러 Mujoco 작업에서 DML-IL을 empirically 평가한 결과, 기존의 최첨단 인과 IL 알고리즘보다 뛰어난 성능을 보였습니다. 이 논문에서 제안하는 새로운 프레임워크는 다양한 기존 설정을 통합하여 보다 현실적인 문제를 고려할 수 있는 방향을 제시합니다. DML-IL의 이식성 간극에 대한 상한선은 이전 결과를 특별한 경우로 회복하는 데 성공했습니다.



### Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving (https://arxiv.org/abs/2502.07640)
- **What's New**: Goedel-Prover는 수학 문제의 자동 형식 증명 생성에서 최첨단 성능을 달성한 오픈 소스 대형 언어 모델(LLM)입니다. 이 모델은 자연어 수학 문제를 정형 언어(Lean 4)로 변환하는 명제 정형화기를 훈련시켜 164만 개의 정형 명제를 생성했습니다. 또한, 기존의 공개 모델을 초과하는 성능을 갖춘 최종 프로버를 훈련하여 문제를 증명합니다.

- **Technical Details**: Goedel-Prover는 자연어 문제 명제를 수학적으로 형식화하여 Lean 언어로 표현하는 프로세스를 포함합니다. 우리가 사용한 방법론은 expert iteration을 통해 프로버를 점진적으로 발전시키고, 초기 데이터를 기반으로 한 여러 번의 훈련을 통해 새로운 증명을 생성하는 것입니다. 총 8회의 반복 훈련 후 우리의 모델은 전체 증명 생성을 위한 최첨단 모델로 자리잡았습니다.

- **Performance Highlights**: miniF2F 벤치마크에서 Goedel-Prover는 57.6%의 성공률을 기록하며, 이전 최고의 성과보다 7.6% 향상된 결과를 보여줍니다. PutnamBench에서 7개의 문제를 성공적으로 해결하여 리더보드 1위를 차지했습니다. 또한, Lean Workbook 문제를 위해 29.7K개의 정형 증명을 생성하여 기존의 15.7K보다 거의 두 배 증가시켰습니다.



### Distributed Value Decomposition Networks with Networked Agents (https://arxiv.org/abs/2502.07635)
Comments:
          21 pages, 15 figures, to be published in Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), Detroit, Michigan, USA, May 19 - 23, 2025, IFAAMAS

- **What's New**: 이번 연구에서 제안한 분산 가치 분해 네트워크(DVDN)는 중앙 집중식 훈련 없이도 협력적인 다중 에이전트 강화 학습(MARL) 환경에서 에이전트들이 상호 작용을 통해 학습할 수 있는 방안을 제공합니다. DVDN은 에이전트별 Q-함수로 분해되는 공동 Q-함수를 생성하여 에이전트들이 지역적으로 공유된 목표를 추정할 수 있도록 돕습니다. 이 연구는 이질적 및 동질적 에이전트 환경을 위한 두 가지 혁신적인 알고리즘, 즉 DVDN과 DVDN (GT)를 기여합니다.

- **Technical Details**: 연구에서 소개된 DVDN 알고리즘은 분산 훈련과 가치 분해 네트워크의 Q-함수 분해를 결합한 방법입니다. 이 알고리즘은 에이전트들 간의 통신을 통해 지역적인 공동 시간 차이(JTD) 추정을 향상시키며, 각 에이전트는 지역적으로 JTD를 최소화합니다. 동질적 에이전트 환경에서는 경량 추적(gradient tracking) 메커니즘을 사용하여 에이전트들이 공통 손실 함수를 최소화하도록 유도합니다.

- **Performance Highlights**: 실험 결과, DVDN과 DVDN (GT) 알고리즘은 정보 손실이 발생함에도 불구하고, 가치 분해 네트워크의 성능을 근사할 수 있음을 보여주었습니다. 이 연구는 다양한 MARL 작업을 통해 이 알고리즘의 효용성을 증명하였으며, 특히 세 가지 표준 환경 내에서의 10개 MARL 작업에 대한 실험이 진행되었습니다. 이러한 결과는 에이전트들이 제한된 커뮤니케이션 환경에서도 효과적으로 협력할 수 있음을 강조합니다.



### DMWM: Dual-Mind World Model with Long-Term Imagination (https://arxiv.org/abs/2502.07591)
- **What's New**: 본 논문에서는 DMWM(Dual-Mind World Model)이라는 새로운 세계 모델 프레임워크를 제안합니다. 이 모델은 인간 인지의 이원적 과정 이론을 기반으로 하여, 논리적 일관성을 유지하며 상상력을 발휘하도록 설계되었습니다. DMWM은 직관적인 상태 전환을 처리하는 RSSM(System 1) 구성 요소와 위계적 깊이 논리적 추론을 통해 상상 과정을 안내하는 LINN(System 2) 구성 요소로 구성됩니다.

- **Technical Details**: DMWM는 다양한 상태와 행동 공간 사이의 복잡한 논리적 관계를 묘사합니다. RSSM-S1은 기계 학습에서 상태 동역학을 빠르고 직관적으로 학습하며, LINN-S2는 논리적 정규화 규칙을 사용하여 상태와 행동 공간 내에서의 추론을 수행합니다. 두 시스템 간의 피드백 메커니즘을 통해 DMWM은 추론과 예측 간의 논리적 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, DMWM은 복잡한 작업에서 기존 모델에 비해 논리적 일관성, 시도 효율성, 데이터 효율성 및 신뢰할 수 있는 상상의 측면에서 각각 14.3%, 5.5배, 32%, 120%의 개선 효과를 보였습니다. 이는 DMWM이 장기 계획 능력을 극대화하는 데 있어 효과적인 접근 방식을 제공함을 나타냅니다.



### We Can't Understand AI Using our Existing Vocabulary (https://arxiv.org/abs/2502.07586)
Comments:
          Position paper

- **What's New**: 이 논문은 기존의 인간 언어 어휘에 의존하지 않고 AI를 이해하기 위해 네올로지즘(neologism), 즉 인간 개념과 기계 개념을 가리키는 새로운 단어를 만들어야 한다고 주장합니다. 이러한 새로운 단어는 기계가 이해할 수 있는 인간 개념을 정의하거나, 인간이 기계 개념을 이해하는 데 도움을 줄 수 있습니다. 이를 통해 인간과 기계 간의 상호작용에서 발생하는 커뮤니케이션 문제를 해결할 수 있을 것으로 기대합니다.

- **Technical Details**: 이 논문에서는 언어 모델 기반 AI 시스템을 이해하고 제어하는 과정에서의 커뮤니케이션 문제를 다룹니다. AI 시스템과 인간은 세계를 다르게 이해하며, 이로 인해 각기 다른 개념을 형성합니다. 저자들은 'length neologism'과 'diversity neologism' 등 네올로지즘을 활용해 모델의 응답 길이나 다양성을 제어할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 새로운 단어를 사용한 프롬프트는 모델의 응답을 효과적으로 조정하는 데 기여했으며, 이는 기존 모델의 가중치는 그대로 두면서도 더 정밀한 커뮤니케이션을 가능하게 합니다. 이러한 방식은 인간과 기계 간의 언어를 공동으로 발전시킬 수 있는 첫걸음으로 간주됩니다.



### Automated Capability Discovery via Model Self-Exploration (https://arxiv.org/abs/2502.07577)
- **What's New**: 이 논문에서는 Automated Capability Discovery (ACD)라는 새로운 프레임워크를 소개하고 있습니다. ACD는 특정 foundation model을 과학자로 활용하여 대상 모델의 능력을 탐색하는 개방형 작업을 체계적으로 제안합니다. 이는 기존의 평가 방법론이 요구하는 높은 인적 노력을 줄일 수 있는 가능성을 제공합니다.

- **Technical Details**: ACD는 최신 frontier models과 open-endedness 분야의 아이디어를 결합하여, 대상 모델의 놀라운 능력과 실패를 자동으로 발견합니다. 논문에서는 GPT, Claude, Llama 시리즈 등 여러 foundation models에서 ACD를 적용해 수천 가지의 능력을 자동으로 드러내는 방법을 보여줍니다. ACD는 기존 팀이 발견하기 어려운 능력을 밝혀내는 데 효과적입니다.

- **Performance Highlights**: 연구진은 ACD의 자동 채점 방법이 인간 설문조사와 높은 일치를 보인다고 보고하고 있습니다. ACD는 foundation model의 작업 생성 및 자기 평가 능력을 활용하여 새로운 AI 시스템의 확장 가능하고 자동화된 평가를 향한 중요한 진전을 나타냅니다. 모든 코드와 평가 로그는 오픈 소스로 제공되며, 연구자들과 개발자들이 접근할 수 있도록 합니다.



### LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid (https://arxiv.org/abs/2502.07563)
Comments:
          Technical report, 17 pages

- **What's New**: 이번 논문에서 소개된 LASP-2는 기존의 Sequence Parallelism(SP) 방법론의 한계를 극복하여, 매우 긴 입력 시퀀스를 처리하는 Linear Attention Transformer 모델의 훈련 효율성을 현저하게 향상시킵니다. LASP-2는 데이터 통신 방식을 혁신적으로 재구성하여, 오직 단일 AllGather 집합 통신만으로도 중간 메모리 상태 간의 효율적인 데이터 전달이 가능합니다. 이는 긴 시퀀스 처리 시의 통신 병렬성과 연산 병렬성을 모두 개선하는 효과를 가져옵니다.

- **Technical Details**: LASP-2는 기존 LASP 방식에 비해 통신 요구사항을 최소화하는 새로운 구조를 도입합니다. 이 구조는 중간 메모리 상태의 크기가 시퀀스 길이에 독립적으로 유지됨으로써, 긴 시퀀스 처리 시에도 통신 부담이 경감됩니다. 또한 LASP-2H를 통해, Linear Attention과 표준 Attention 모듈 모두에 대한 효율적인 SP 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, LASP-2는 2048K 길이의 시퀀스를 처리하는데 있어, LASP 대비 15.2%, Ring Attention 대비 36.6%의 훈련 속도 향상을 보여주었으며, 이는 대규모 GPU 환경에서도 우수한 성능을 입증하였습니다. 이러한 결과는 Linear-Llama3 모델을 통해 검증되었으며, LASP-2와 LASP-2H의 효율성 및 성능 개선이 두드러지게 나타났습니다.



### LoRP-TTS: Low-Rank Personalized Text-To-Speech (https://arxiv.org/abs/2502.07562)
- **What's New**: 이 연구에서는 Low-Rank Adaptation (LoRA) 기법을 사용하여 소음이 있는 환경에서의 자발적 음성을 단 한 번의 녹음으로도 성공적으로 임시적으로 사용할 수 있음을 보여주고 있습니다. 이를 통해 스피커 유사성을 최대 30pp까지 향상시키면서 내용과 자연스러움을 유지할 수 있었습니다. 이 방법은 다양한 음성 데이터 코퍼스를 생성하는 데 중요한 진전을 이룹니다.

- **Technical Details**: LoRA는 특정 레이어에 저차원 행렬을 삽입하여 모델의 모든 가중치를 갱신하는 대신, 대부분의 원래 모델 매개변수를 고정한 채로 작업 특화 조정을 캡처합니다. 본 연구에서는 Voicebox 모델의 밀집 레이어마다 LoRA를 삽입하고, 10M의 추가 매개변수를 도입하여 적응을 진행하였습니다. 평가에서는 Word Error Rate (WER)와 Character Error Rate (CER) 등을 사용하여 합성된 음성의 정확성과 가독성을 측정하였습니다.

- **Performance Highlights**: 연구 결과, 몇 번의 적응 스텝만으로도 스피커 유사성이 현저히 증가하는 성공적인 결과가 나타났습니다. 즉, 100,100,100,100의 최적화를 위한 스텝으로도 상당한 향상이 있었으며, 훈련이 진행될수록 WER과 CER이 개선되어 더 자연스러운 음성 합성이 가능해지는 것을 보여주었습니다. 그러나 모든 샘플을 사용하는 것이 반드시 품질 향상으로 이어지지는 않았으며, 특히 수집된 데이터가 오류를 포함할 경우 전반적인 합성 품질 저하를 초래할 수 있음을 확인했습니다.



### Unsupervised Translation of Emergent Communication (https://arxiv.org/abs/2502.07552)
Comments:
          19 pages (including appendix and bibliography), Accepted to AAAI 2025

- **What's New**: 이 연구는 Emergent Communication(EC)을 이해하고 자연어(NL)로의 번역 가능성을 탐구하는 새로운 방법론을 제시합니다. 특히, 기존의 병행 데이터 없이 Unsupervised Neural Machine Translation(UNMT) 기술을 활용하여 다양한 난이도의 참조 게임에서 형성된 EC를 번역합니다. 이러한 접근은 EC의 해석성과 번역 가능성을 높이고, AI가 생성한 언어의 이해를 도울 수 있는 새로운 지평을 여는 데 중요한 의미를 가집니다.

- **Technical Details**: 연구에서는 참조 게임을 통해 생성된 EC를 UNMT 기술을 활용하여 영어로 번역합니다. 게임은 다양한 난이도의 복잡성을 가지며, 불확실한 의미의 변형을 포함한 메시지 전달을 통해 생성된 언어의 구조를 분석합니다. 이 과정에서 EC 데이터셋은 에이전트 간의 메시지를 수집하여 생성되며, 영어 캡션 데이터셋은 EC의 배경을 설명하는 언어적 우선 요소로 사용됩니다.

- **Performance Highlights**: 실험 결과, UNMT는 AI가 생성한 언어를 자연어로 번역하는 데 성공적이라는 것을 보여주었습니다. 특히, Inter-category 설정에서는 번역 품질이 향상된 것으로 나타났으며, BLEU 및 METEOR 점수가 높은 것으로 확인되었습니다. 그러나 메시지 unpredictability가 증가함에 따라 번역 정확도가 떨어지는 경향도 발견되었습니다.



### HGTUL: A Hypergraph-based Model For Trajectory User Linking (https://arxiv.org/abs/2502.07549)
Comments:
          11 pages, 4 figures

- **What's New**: 이 연구에서는 Trajectory User Linking (TUL) 문제를 해결하기 위한 새로운 접근법인 HyperGraph 기반 다각적 TUL 모델(HGTUL)을 제안합니다. 기존 연구들이 간과해 온 고차원 경로 간의 관계를 유효하게 모델링하며, 서로 다른 POI(Point of Interest)들이 경로에 미치는 변동 영향을 고려합니다. 특히, 사용자 활동 수준과 체크인 빈도의 불균형 문제를 해결하기 위해 데이터 균형화 방법을 설계하였습니다.

- **Technical Details**: HGTUL 모델은 관계적 및 시공간적(Spatio-Temporal) 관점에서 경로 표현을 학습합니다. 고차원 관계를 캡처하기 위해 경로 하이퍼그래프(hypergraph)를 구성하고, 하이퍼그래프 주의 네트워크를 활용하여 POI의 변동적인 영향을 학습합니다. 또한, 경로의 시공간 특성을 통합하기 위해 LSTM(Long Short-Term Memory) 네트워크를 사용하여 순차적 특성을 입력합니다.

- **Performance Highlights**: 실험 결과, HGTUL 모델은 세 가지 실제 데이터 세트에서 최신 방법보다 2.57%~20.09% 및 5.68%~26.00%의 정확도(ACC@1) 및 Macro-F1 지표 개선을 달성하였습니다. 이러한 결과는 HGTUL 모델이 TUL 문제 해결에 있어 실질적으로 향상된 성능을 제공함을 보여줍니다.



### Exoplanet Transit Candidate Identification in TESS Full-Frame Images via a Transformer-Based Algorithm (https://arxiv.org/abs/2502.07542)
- **What's New**: 이번 논문은 TESS(Transiting Exoplanet Survey Satellite) 데이터에서 새로운 외계 행성의 트랜짓 신호 매핑을 위한 혁신적인 접근 방식을 제안합니다. 특히, 이 새로운 기법은 주기적인 트랜짓 가정이나 단계 접기를 필요로 하지 않으며, Transformers 네트워크를 이용하여 직접적으로 Full Frame Image(FFI) 빛 곡선을 분석합니다. 결과적으로 214개의 새로운 행성 시스템 후보가 확인되었으며, 이는 기존 방법들의 한계를 극복하도록 설계되었습니다.

- **Technical Details**: 제안된 모델은 다중 헤드 셀프 어텐션(multi-head self-attention) 메커니즘을 활용하여 FFI 빛 곡선에서 트랜짓 신호를 직접 출류합니다. 이 접근 방식은 기존의 단계 접기 및 주기성 가정에 의존하지 않고도 신호를 정확하게 인식할 수 있게 해 줍니다. 훈련된 네트워크는 트랜짓 신호의 특정 특성(예: 모양)을 학습하여 행성의 트랜짓을 다른 변동 원천과 구분할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: 모델은 TESS 1-26 구역에서 0.27 $R_{\mathrm{Jupiter}}$ 이상의 반지름을 가진 외계 행성 시스템 후보들을 성공적으로 탐지했으며, 그 중 122개는 다중 트랜짓, 88개는 단일 트랜짓 행성으로 분류되었습니다. 이는 기체 재배포 없이 반복되는 기법에 의존하는 기존 모델들보다 뛰어난 성능을 나타냅니다. 이 연구는 단일 트랜짓 신호의 탐지와 분석을 위한 미래의 가능성을 제시하며, 외계 행성 후보들을 더욱 효과적으로 발견할 수 있는 길을 열어줍니다.



### VidCRAFT3: Camera, Object, and Lighting Control for Image-to-Video Generation (https://arxiv.org/abs/2502.07531)
- **What's New**: 최근 image-to-video (I2V) 생성 방법들은 카메라 궤적(camera trajectory)이나 객체 움직임(object motion)과 같은 시각적 요소를 제어하는 데 성공을 거두었습니다. 하지만 기존의 방법들은 다수의 시각적 요소에 대한 제어를 제공하는 데 한계가 있었고, 이러한 문제를 해결하기 위해 새로운 프레임워크인 VidCRAFT3를 소개하고 있습니다. VidCRAFT3는 카메라 움직임, 객체 움직임, 조명 방향을 동시에 제어할 수 있도록 설계되었습니다.

- **Technical Details**: VidCRAFT3는 Spatial Triple-Attention Transformer를 포함하여 각 시각적 요소에 대한 제어를 더 잘 분리할 수 있는 기술을 채택하고 있습니다. 또한, 실제 비디오 데이터셋의 조명 주석이 부족한 문제를 해결하기 위해 VideoLightingDirection (VLD)라는 고품질 합성 비디오 데이터셋을 구축하였습니다. 이 데이터셋은 다양한 외관의 객체와 조명 방향 주석을 포함하고 있어 VidCRAFT3가 강한 빛의 전송 및 반사 효과를 효과적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과 VidCRAFT3는 다양한 벤치마크 데이터셋에서 기존의 최첨단 방법들을 초월하여 높은 품질의 비디오 콘텐츠를 생성하는 능력이 입증되었습니다. 특히, 제어의 세밀함(control granularity)과 시각적 일관성(visual coherence) 면에서 탁월한 성능을 보여주었습니다. 이 모든 코드와 데이터는 공개될 예정이므로 연구자들이 쉽게 접근할 수 있도록 할 것입니다.



### Scaling Off-Policy Reinforcement Learning with Batch and Weight Normalization (https://arxiv.org/abs/2502.07523)
- **What's New**: 이 논문에서는 CrossQ라는 최신의 model-free reinforcement learning 알고리즘의 샘플 효율성을 극대화하기 위한 새로운 접근 방식을 제안합니다. 고유한 특징으로는 낮은 update-to-data (UTD) 비율에서 이미 뛰어난 성능을 입증했으며, 본 연구는 이를 높은 UTD 비율로 확장할 수 있는 가능성을 탐구하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구 팀은 Weight Normalization (WN)을 CrossQ 프레임워크에 통합하여 훈련의 안정성을 높이는 방법을 제안했습니다. 이러한 방법은 훈련 중 발생할 수 있는 잠재적인 플라스틱성 손실을 방지하고, 효과적인 학습률을 일정하게 유지하는 데 도움을 줍니다. 이 기술은 모델의 하이퍼파라미터 조정 외에도 Bellman 방정식에서의 상태-행동 분포를 적절하게 처리하는 것에 대한 인사이트를 기반으로 합니다.

- **Performance Highlights**: 제안된 방법은 DeepMind Control Suite와 Myosuite 벤치마크에서 25개의 어려운 연속 제어 작업에 대해 경쟁력 있는 성능을 달성했습니다. 특히 복잡한 강아지 및 휴머노이드 환경에서 기여가 두드러졌습니다. 또한, 이 접근 방식은 네트워크 리셋과 같은 극단적인 개입 없이도 샘플 효율성과 확장성을 크게 향상시킬 수 있는 간단하면서도 강력한 경로를 제공합니다.



### The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation (https://arxiv.org/abs/2502.07516)
- **What's New**: 이 논문은 의료 이미지 분석에서의 T2I(diffusion models) 생성 모델의 데이터 기억 문제를 분석한 첫 체계적 연구입니다. MIMIC-CXR 데이터셋에서 가장 많은 기억을 초래하는 텍스트 프롬프트와 토큰을 찾기 위한 데이터 기반 접근 방식을 채택했습니다. 특히, 비디오신원 확인 절차와 관련된 마커가 기억에 가장 많이 기여한다는 점이 밝혀졌습니다.

- **Technical Details**: 논문에서는 T2I 생성 모델의 기억화 메커니즘에 대한 깊은 이해를 바탕으로 두 가지 확산 프로세스를 설명하고 있습니다. 첫 번째는 시간적으로 넘어가는 Gaussian 노이즈를 사용해 데이터를 점진적으로 손상시키는 전방 프로세스이며, 두 번째는 제공된 데이터를 반복적으로 디노이징하여 원래의 샘플을 복원하는 역방향 프로세스입니다. 이 과정을 통해 Quick Track 및 Text-Conditional Noise를 통한 기억화 발견 메트릭을 제안합니다.

- **Performance Highlights**: 기존 기억화 완화 전략이 모델의 기억 토큰에 대한 의존도를 충분히 줄이지 못한다는 사실을 발견했습니다. 이는 MIMIC-CXR 데이터셋에서 T2I 생성 모델의 신뢰성 문제를 강조합니다. 이 연구는 기억화 완화 기술 개발을 위한 기초 자료를 제공하고, 향후 의학 이미지 연구에 대한 중요한 통찰력을 제시합니다.



### WebChecker: A Versatile EVL Plugin for Validating HTML Pages with Bootstrap Frameworks (https://arxiv.org/abs/2502.07479)
- **What's New**: WebChecker는 Epsilon Validation Language (EVL)를 기반으로 한 플러그인으로, Bootstrap과 같은 다양한 HTML 및 CSS 프레임워크의 정적 및 동적 페이지를 검증합니다. 사용자는 구성 가능한 EVL 제약조건을 통해 웹 페이지의 규정 준수를 쉽게 확인할 수 있습니다. 이 플러그인은 프레임워크의 명시적 규칙을 자동으로 분석하여, 문서 검토의 번거로움을 덜어줍니다.

- **Technical Details**: WebChecker는 HTML 페이지 또는 HTML 페이지의 일부를 모델로 사용하며, Epsilon Model Connectivity (EMC) 레이어를 구현합니다. 이 플러그인은 사용자가 작성한 EVL 제약조건의 유연한 적용을 통해 코드의 가독성을 높이고, 다양한 HTML 소스를 검사할 수 있게 합니다. 코드 작성량을 줄이고 최소의 코드로도 규칙을 간편하게 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: WebChecker는 Bootstrap 프레임워크에 대해 초기 25개의 규칙을 구현하였으며, 이는 더 많은 규칙으로 확장될 수 있습니다. 이 플러그인은 특정 HTML 요소에 대해 접근성을 강화하는 제약조건을 검증하며, 예를 들어, aria-label 속성을 확인하는 규칙이 포함됩니다. GitHub 저장소에서 구현된 전체 규칙 목록을 확인할 수 있으며, 이는 개발자들이 웹 페이지를 보다 효과적으로 관리할 수 있도록 돕습니다.



### 5D Neural Surrogates for Nonlinear Gyrokinetic Simulations of Plasma Turbulenc (https://arxiv.org/abs/2502.07469)
Comments:
          6 pages (+ references and appendix)

- **What's New**: 이 연구에서는 고차원 플라즈마 난류 시뮬레이션을 위한 신경망 대체 모델 5D Swin-UNet을 제안합니다. 이 모델은 5D 분포 함수의 시간 진화를 직접 모델링하여 기존의 비선형 자이로킨etic 시뮬레이션을 약 100배 빠르게 수행할 수 있습니다. 또한, 실제 물리량의 예측 정확도가 뛰어난 것으로 나타났습니다. 이로 인해, 핵융합 에너지 생산의 상용화가 가속화될 것으로 기대됩니다.

- **Technical Details**: 5D Swin-UNet은 5D 분포 함수를 압축, 처리 및 재구성하기 위해 계층적 비전 트랜스포머를 확장한 구조로 개발되었습니다. 특히, 비선형 데이터 생성기인 GKW을 사용하여 아디아바틱 전자 근사를 기반으로 시뮬레이션 데이터를 수집하고, 다양한 이온 온도 기울기(ITG)의 값을 사용하여 훈련하였습니다. 이 과정에서 사전 처리 및 시각화 기법도 다루어졌습니다.

- **Performance Highlights**: 5D Swin-UNet은 기존 비선형 자이로킨etic 시뮬레이션(GKW)에 비해 100배 더 빠른 속도로 물리적 양을 예측합니다. 여기에는 열 플럭스 시간 추적 및 정전기 잠재량과 같은 물리량이 포함되며, 이들은 모델을 통해 예측되어 훈련 데이터와 잘 일치합니다. 연구 결과, 5D Swin-UNet은 특정 ITG 값에서 예측이 정확하게 수행되었으며, 이는 기존의 훈련 과정에서 관찰되지 않았던 경우입니다.



### Crime Forecasting: A Spatio-temporal Analysis with Deep Learning Models (https://arxiv.org/abs/2502.07465)
Comments:
          8 pages,6 figures

- **What's New**: 이 연구는 특정 날짜에 따라 도시 구역의 범죄 수를 예측하기 위해 딥 러닝 모델을 활용합니다. 해당 연구는 경찰의 감시를 강화하고 정보 수집 및 범죄 예방을 보조하는 데 기여합니다. 양적(SPATIAL) 및 시간적(TEMPORAL) 시퀀스로 범죄 수 예측을 수식화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 범죄 수 예측을 위해 합성곱 신경망(Convolutional Neural Networks, CNN)과 장기 단기 기억 네트워크(Long Short-Term Memory, LSTM)를 결합한 새로운 모델을 도입했습니다. 우리는 원시(raw) 및 분류된(binned) 데이터와 같은 다양한 데이터 시퀀스가 네 가지 딥 러닝 예측 모델의 예측 오류에 미치는 영향을 비교 분석했습니다. 기존의 원시 범죄 데이터를 직접 입력할 경우 예측 오류가 크게 발생하여 모델의 실제 적용에는 적합하지 않습니다.

- **Performance Highlights**: 제안된 CNN-LSTM 모델은 범죄 데이터를 10개 또는 5개 그룹으로 분류할 때 최고의 성능을 발휘합니다. 데이터의 범주화를 통해 예측 모델 성능이 향상될 수 있지만, 불명확한 간격은 지도 세분화(map granularity)를 감소시킬 수 있습니다. 5개의 구간으로 나누는 것보다 10개의 간격으로 분류하는 것이 데이터 특성을 보존하면서 원시 데이터의 예측 모델링 효과성을 초과하는 최적의 균형을 이룹니다.



### JamendoMaxCaps: A Large Scale Music-caption Dataset with Imputed Metadata (https://arxiv.org/abs/2502.07461)
Comments:
          8 pages, 5 figures

- **What's New**: JamendoMaxCaps는 유명한 Jamendo 플랫폼에서 200,000개 이상의 자유 라이선스가 있는 기악 트랙을 포함하는 대규모 음악 자막 데이터셋입니다. 이 데이터셋은 최첨단 자막 모델에 의해 생성된 캡션을 포함하고 있으며, 보완된 메타데이터로 향상되었습니다. 또한 음악적 특징과 메타데이터를 활용하여 유사한 곡을 식별하는 검색 시스템을 제안하여, 연구자들에게 음악-언어 이해 작업을 위한 더 포괄적이고 정보가 풍부한 데이터셋을 제공합니다.

- **Technical Details**: 음악 정보 검색(field of music information retrieval)은 음악 데이터를 분석, 조직, 접근 및 생성하기 위한 계산 기법 개발에 중점을 두고 있습니다. 데이터셋의 부족 문제를 해결하기 위해, 우리는 JamendoMaxCaps를 만들고, 각 트랙에 자연어 설명을 생성하기 위해 최신 음악 자막 모델을 사용합니다. 또한, 우리는 불완전한 메타데이터를 해결하기 위해 오디오 특징과 로컬 대형 언어 모델(LLLM)을 활용한 다중 모드 메타데이터 보완 접근 방식을 제안합니다.

- **Performance Highlights**: JamendoMaxCaps 데이터셋을 통해 200,000개 이상의 기악 곡과 생성된 캡션이 제공되어 음악-언어 이해 작업의 연구를 진전시킬 수 있습니다. 우리는 알고리즘의 성능을 다섯 가지 다른 측정으로 정량적으로 검증하였고, 효율적인 검색 시스템을 통해 유사한 곡을 효과적으로 인식할 수 있음을 입증했습니다. 이를 통해 더욱 정확하고 신뢰할 수 있는 메타데이터 보완이 이루어질 수 있습니다.



### PerCul: A Story-Driven Cultural Evaluation of LLMs in Persian (https://arxiv.org/abs/2502.07459)
Comments:
          Accepted at NAACL 2025 Main Conference, the dataset is available on HuggingFace (see this https URL)

- **What's New**: 이 논문에서는 전통적으로 서양 문화를 반영하는 대형 언어 모델(LLMs)의 한계를 분석하고, 페르시아 문화에 대한 감수성을 평가하는 새로운 데이터세트인 PerCul을 소개합니다. PerCul은 페르시아어로 된 문화적 요소가 담긴 이야기 기반의 다지선다형 질문을 포함하여, 기존의 데이터세트들과는 달리 원어민의 입력을 바탕으로 만들어졌습니다. 이러한 접근 방식은 항상 번역에 의존하지 않도록 설계되어, 다양한 문화적 배경을 충실히 반영할 수 있도록 합니다.

- **Technical Details**: PerCul 데이터세트는 Hall의 문화 삼각형 이론을 바탕으로 문화 카테고리를 정의한 뒤, 원어민 주석가들이 기술적 세부 정보를 생성하고 LLM들이 이야기의 줄거리를 만들어냅니다. 생성된 이야기는 엄격한 인간 수정 과정을 거쳐 최종적으로 다지선다형 질문이 구성됩니다. 이를 통해 LLM의 문화적 이해도를 평가하기 위한 명확한 기준을 제공하며, 모든 질문은 불특정한 문화적 개념이 아닌 실제 페르시아 문화 요소를 포함하도록 했다.

- **Performance Highlights**: PerCul의 성능 평가에서 가장 우수한 클로즈드 소스 모델과 일반인 기준 간의 차이는 11.3%, 최고의 오픈 웨이트 모델을 사용했을 때는 이 차이가 21.3%로 증가하는 것으로 나타났습니다. 또한 연구 결과는 페르시아 문화에 대한 이해가 부족한 LLM들이 표면 수준의 세부 정보에 의존하는 경향이 있음을 보여줍니다. 이는 페르시아어로 조정된 LLM들이 다국어 기본 모델보다 성능이 낮은 원인을 설명하는 데 기여합니다.



### RusCode: Russian Cultural Code Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2502.07455)
Comments:
          Accepted for NAACL 2025 Findings, GitHub: this https URL

- **What's New**: 이번 논문에서는 러시아 문화 코드의 요소를 포함한 텍스트-이미지 생성 모델의 품질을 평가하기 위한 RusCode 벤치마크를 제안합니다. 이는 다양한 문화적 측면을 반영하는 19개의 카테고리로 구성된 데이터셋을 기반으로 하며, 1250개의 러시아어 텍스트 프롬프트와 그 영어 번역이 포함되어 있습니다. 문화 간의 이해 부족과 생성 품질 저하 문제를 해결하기 위한 노력의 일환으로 쓰여졌습니다.

- **Technical Details**: 러시아 시각문화의 특정 개념을 반영한 텍스트 설명의 품질을 평가하기 위해 다각적인 전문가의 참여를 통해 19개 카테고리를 생성하였습니다. 이 데이터셋은 역사, 예술, 민속 등 다양한 주제를 포함하여, 각 프롬프트는 해당 개념의 실제 이미지를 연관시켜 생성 품질 평가에 활용될 수 있습니다. 이를 통해 현대 텍스트-이미지 생성 모델의 다문화적 이해 현황을 분석할 수 있습니다.

- **Performance Highlights**: 인간 평가 결과는 최신 텍스트-이미지 생성 모델들, 즉 Stable Diffusion 3, DALL-E 3 등에서 러시아 문화 개념의 표현이 얼마나 잘 이루어지는지를 비교 분석하였습니다. 이를 통해 생성 모델의 문화 인식 수준을 평가하고 나타난 격차를 드러내었습니다. 이 작업은 러시아 문화에 대한 문화 인식 문제의 첫 번째 포괄적 접근으로 평가됩니다.



### Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon (https://arxiv.org/abs/2502.07445)
- **What's New**: 이번 연구에서는 Chameleon Benchmark Overfit Detector (C-BOD)라는 메타 평가 프레임워크를 소개하여, 모델이 벤치마크에 특화된 패턴에 과도하게 의존하고 있는지 확인하고 이를 감지하는 방법을 제안합니다. C-BOD는 입력의 의미를 유지하면서 질문 형식을 변형하여 모델의 실질적 성능을 평가할 수 있는 기회를 제공합니다. 이는 LLM의 진정한 언어 이해를 과시하기보다는 단순히 표면적인 패턴에 기반한 성능 수치를 드러내는 방법론입니다.

- **Technical Details**: C-BOD는 벤치마크 프롬프트를 다양하게 변형하여 LLM의 성능 변화를 측정하는 방식으로 작동합니다. 이 방법은 입력을 변형시키기 위해 재구성 도구(T)와 왜곡 매개변수(μ)를 사용하여 원본 및 변형된 데이터셋에 대한 평가를 수행합니다. 이 프레임워크는 통계적 검정을 통해 성능 차이가 과적합을 나타내는지를 판단하는 데 중점을 두며, 모델이 기억한 패턴에서 벗어나지 못할 경우 기인하는 성과 손실을 보여줍니다.

- **Performance Highlights**: 실험 결과, 26개의 주요 LLM 중 20개 모델이 통계적으로 유의미한 성능 저하를 보였으며, 평균적으로 2.15%의 성능 저하가 확인되었습니다. 특히 높은 기본 정확도를 가진 모델은 더 큰 성능 차이를 보였으며, 큰 LLM은 재구성에 더 민감하게 반응하는 경향이 있었습니다. 반면, Llama 패밀리 모델은 유의미한 저하를 보이지 않아 피상적 단서에 대한 의존성이 줄어드는 경향을 보였습니다.



### SensPS: Sensing Personal Space Comfortable Distance between Human-Human Using Multimodal Sensors (https://arxiv.org/abs/2502.07441)
- **What's New**: 이 연구는 개인 공간(Peripersonal space)의 편안한 거리를 추정하는 새로운 센서 기반 모델을 제안합니다. 이를 위해 눈 추적(eye-tracking) 및 손목 센서(wristband sensor)와 같은 고급 다중 모달(multimodal) 감지 기술을 활용합니다. 연구 결과, 특히 눈 추적 데이터가 개인 공간 선호도를 예측하는 데 중요한 역할을 한다는 것을 발견했습니다.

- **Technical Details**: 개인의 편안한 공간을 추정하기 위해 Pupil Core 눈 추적 안경과 Empatica E4 손목 센서를 사용하였습니다. Pupil Core는 최대 200Hz의 샘플링 속도로 정확하고 신뢰할 수 있는 시선 데이터를 수집하며, Empatica E4는 심박수(heart rate), 피부 전도(electrodermal activity) 등 다양한 생리학적 신호를 실시간으로 측정합니다. 두 장치에서 수집된 데이터는 실험의 특정 기간에 맞추어 잘라내고, 노이즈를 제거하여 사전 처리( preprocessing) 하였습니다.

- **Performance Highlights**: Transformer 기반 모델을 통해 개인 공간 예측에 대한 F1 점수 0.87의 최고 예측 정확도를 달성한 실험 결과를 바탕으로 합니다. 시선 점(gaze point)과 동공 직경(pupil diameter)과 같은 눈 추적 기능이 가장 중요한 예측 변수가 되었고, 손목 센서의 생리학적 신호는 상대적으로 경미한 기여를 보였습니다. 이러한 결과는 AI 기반의 사회적 공간(personal space) 개인화 가능성을 강조하며, 직장, 교육 기관, 공공 장소에서의 공간 최적화를 위한 지능형 시스템 개발에 대한 기초 자료가 됩니다.



### RomanLens: Latent Romanization and its role in Multilinguality in LLMs (https://arxiv.org/abs/2502.07424)
Comments:
          18 pages, 18 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 영어 중심의 코퍼스에서 훈련됨에도 불구하고 놀라운 다국어 일반화 능력을 보이는 이유를 탐구합니다. 구체적으로, 우리는 비라틴 문자 언어에서 로마자화(romanization)가 다국어 처리에 미치는 역할을 조사하며, 이를 ‘Latent Romanization’이라고 명명합니다. 연구 결과에 따르면, LLM은 다국어 다음 토큰 생성 과정에서 중간 층에서 로마자로 표현된 단어를 자주 사용합니다.

- **Technical Details**: 로마자화는 비라틴 스크립트를 라틴 문자로 표현하는 방법으로, 이는 LLM이 언어 중립적인 개념 공간과 언어 특정 출력 표현 간의 교량 역할을 할 수 있음을 보여줍니다. 본 연구에서는 LLaMA-2 7B 모델을 사용하여 다음 토큰 생성을 시각화하고, activation patching 기법을 활용하여 로마자화와 네이티브 스크립트 간의 개념을 비교 분석합니다. 이를 통해 LLM이 다양한 스크립트에서 의미를 어떻게 처리하는지에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 로마자화된 표현은 모델의 층에서 네이티브 스크립트 표현보다 더 빨리 나타나는 경향이 있습니다. 이는 로마자화가 언어 특정 임베딩으로의 빠른 진행을 촉진할 수 있음을 시사합니다. 또한, 실험 결과는 LLM이 로마자화된 입력과 네이티브 스크립트 입력 간에 개념 정보를 유사하게 인코딩한다는 것을 보여줍니다.



### No Data, No Optimization: A Lightweight Method To Disrupt Neural Networks With Sign-Flips (https://arxiv.org/abs/2502.07408)
- **What's New**: 본 논문은 Deep Neural Networks (DNNs)의 취약점을 밝혀내고, 모델의 파라미터에서 선호되는 sign bits의 일부를 뒤집는 것만으로도 성능에 치명적인 악영향을 미칠 수 있음을 입증하는 새로운 방법론, Deep Neural Lesion (DNL)을 제안합니다. 이는 데이터 없이, 가벼운 방법으로 저해되는 성능을 극대화할 수 있는 전략으로, ResNet50 모델의 경우 단지 두 개의 sign bits를 뒤집는 것만으로도 정확도가 99.8% 감소하는 결과를 보였습니다.

- **Technical Details**: 저자들은 DNN이 일반적으로 사용하는 32비트 부동 소수점 형식(IEEE 754)의 구조를 활용하여, sign bits를 조작할 수 있는 취약점을 확인했습니다. 이들은 비효율적인 계산 없이 경량의 공격을 수행할 수 있으며, 단일 forward 및 backward 패스를 사용하는 1-Pass 공격을 통해 보다 정밀한 파라미터 선택이 가능합니다. 또한, 이 방법은 각 모델 아키텍처에 걸쳐 광범위한 공격 가능성을 가지고 있으며, 데이터 의존성이 없습니다.

- **Performance Highlights**: 60가지 분류 모델과 다양한 데이터셋을 통해 접근 방식을 검증한 결과, 단지 10개 미만의 bits를 조작하는 것만으로도 정확도를 크게 감소시킬 수 있음을 확인했습니다. 더불어, 저자들은 취약한 sign bits를 선택적으로 보호함으로써, DNN 모델이 이와 같은 공격에 보다 강한 저항력을 가질 수 있는 방안을 제시합니다. 이로 인해 자율 주행 시스템과 같이 안전이 중요한 분야에서의 DNN의 신뢰성이 더욱 강화될 것으로 기대됩니다.



### Human-in-the-Loop Annotation for Image-Based Engagement Estimation: Assessing the Impact of Model Reliability on Annotation Accuracy (https://arxiv.org/abs/2502.07404)
- **What's New**: 이 연구는 Human-in-the-loop (HITL) 프레임워크가 감정 추정 시스템의 주석 정확도를 높이는 데 도움이 되는 잠재력에 대해 다루고 있습니다. 특히 고성능 이미지 기반 감정 모델을 HITL 주석 프레임워크에 통합하여 인간-기계 상호작용의 협업 가능성을 평가하고, 성공적인 협업에 필요한 심리적 및 실제적 요소들을 식별합니다. 연구의 주요 결과로는 모델의 신뢰성과 인지적 프레이밍이 annotator의 신뢰, 인지 부하 및 주석 행동에 어떻게 영향을 미치는지를 보여줍니다.

- **Technical Details**: 연구에서는 다양한 모델 신뢰성과 인지적 프레이밍의 변형을 바탕으로 S1, S2, S3의 세 가지 실험 시나리오를 설정하여 29명의 참가자를 대상으로 행동 및 질적 데이터를 분석하였습니다. S1에서는 신뢰할 수 있는 예측이 높이 평가받았으며, S2의 신뢰할 수 없는 출력은 주석가의 비판적 평가를 유도했으나 불만과 반응 변동도 증가시켰습니다. S3에서는 부정적 프레이밍이 참가자들에게 모델에 대한 인지도와 정확성을 높이는 결과를 가져왔습니다.

- **Performance Highlights**: 모델의 신뢰성과 심리적 요소가 효과적인 인간-기계 협업을 형성하는 데 중대한 역할을 한다는 점이 강조되었습니다. 연구 결과, HITL 프레임워크가 감정 주석과 같은 분야에서 인간의 감독과 자동화 시스템의 강점을 잘 활용할 수 있음을 보여주며, 이는 향후 적응 학습 및 인간-컴퓨터 상호작용 등 다양한 분야에 응용될 수 있는 기반을 마련했습니다.



### Enhancing Higher Education with Generative AI: A Multimodal Approach for Personalised Learning (https://arxiv.org/abs/2502.07401)
Comments:
          9 pages, 4 figures, accepted and presented in the 2025 6th International Conference on Advances in Education and Information Technology (AEIT)

- **What's New**: 이번 연구는 고등 교육 분야에서 생성적 인공지능(Generative AI, GenAI)의 가능성을 모색하며, 학부 과정을 위한 다중 모드 챗봇을 설계하고 개발했습니다. ChatGPT API를 사용한 세밀한 텍스트 기반 상호작용과 Google Bard를 활용한 고급 이미지 분석 및 다이어그램-코드 변환에 주목하고 있습니다. 이 연구는 GenAI가 폭넓은 교육 관련 질문에 대응하는 잠재력을 보여줍니다.

- **Technical Details**: 챗봇은 교육자를 위한 파일 기반 분석기를 포함하며, 이를 통해 학생 피드백에 대한 정서(sentiment) 및 감정(emotion) 분석을 제공하고, 주요 지표를 통해 수업 평가를 요약합니다. 이러한 조합은 교습 및 학습 프로세스를 개선하는 데 있어 다중 모드 대화형 인공지능의 중요한 역할을 강조합니다.

- **Performance Highlights**: 본 연구는 GenAI 기술의 통합이 더 역동적이고 반응적인 교육 환경을 조성하는 데 필수적임을 보여주는 실용적인 웹 애플리케이션을 시연합니다. 이는 교육 결과 개선 및 교수법 전략에 기여할 것으로 기대합니다.



### Explainable Multimodal Machine Learning for Revealing Structure-Property Relationships in Carbon Nanotube Fibers (https://arxiv.org/abs/2502.07400)
Comments:
          33 pages, 9 figures

- **What's New**: 이번 연구에서는 다종 데이터(multimodal data)를 통합 분석하는 Explainable Multimodal Machine Learning (EMML) 방법을 제안합니다. 이 방법은 탄소 나노튜브(CNT) 섬유의 특성을 밝혀내는 데 도움이 되며, 다양한 제조 조건과 다중 규모 구조가 물질 특성에 미치는 복잡한 영향을 이해하는 데 기여합니다.

- **Technical Details**: EMML 방법은 피처 추출(feature extraction) 과정에서 설명 가능한 인공지능(Explainable AI, XAI)과 인자 분석(factor analysis)을 활용합니다. 또한, 기존의 표준 방법으로 해석하기 어려운 데이터에 대해 Negative Matrix Factorization (NMF)을 사용하여 핵심 특성을 추출하는 접근 방식을 포함합니다.

- **Performance Highlights**: 분석 결과, 작은 균일 분포의 집합체가 파괴 강도를 개선하는 데 중요한 요인임을 확인했습니다. 또한, 긴 유효 길이의 CNT가 전기 전도성을 향상시키는 데 주요한 역할을 한다는 점이 강조되었습니다. EMML은 CNT 섬유에 국한되지 않고, 나노 물질에서 유도된 다른 재료의 설계에도 적용될 수 있는 유용한 도구로, 데이터 기반(materials) 연구의 발전을 위한 기초를 제공합니다.



### On Iterative Evaluation and Enhancement of Code Quality Using GPT-4o (https://arxiv.org/abs/2502.07399)
- **What's New**: 이번 논문에서는 CodeQUEST라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)을 이용하여 코드 품질을 반복적으로 평가하고 향상시킵니다. CodeQUEST는 10가지 차원에서 코드 품질을 평가하는 Evaluator와 평가 결과를 바탕으로 코드를 개선하는 Optimizer로 구성됩니다. 연구 결과, CodeQUEST는 실제 코드 품질 평가 지표와 밀접하게 일치하는 효과적인 평가를 제공함을 입증했습니다.

- **Technical Details**: CodeQUEST의 품질 평가는 가독성, 유지 관리성, 테스트 가능성, 효율성, 보안 등 10가지 차원을 기준으로 진행됩니다. 이 프레임워크의 Evaluator는 각 차원에 대해 다섯 가지 질문을 통해 코드의 품질을 수량적으로 평가하며, 이 데이터는 차원별 종합 평가 요약으로 집계됩니다. Optimizer는 Evaluator의 피드백을 바탕으로 고정된 횟수의 반복을 통해 코드 품질을 향상시키는 역할을 합니다.

- **Performance Highlights**: CodeQUEST는 Python과 JavaScript를 사용한 실험을 통해 평균 52.6%의 상대적 퍼센트 향상을 기록하며 코드 품질 개선에 성공했습니다. 코드 품질 평가는 Pylint Score, Radon Maintainability Index, Bandit output logs와 같은 프록시 메트릭을 통해 검증되었으며, 유의미한 상관관계가 있음을 보여주었습니다. 이러한 결과는 LLM이 코드 품질 평가 및 개선 프로세스를 자동화하는 데 큰 잠재력을 가지고 있다는 것을 강조합니다.



### Multi-Task-oriented Nighttime Haze Imaging Enhancer for Vision-driven Measurement Systems (https://arxiv.org/abs/2502.07351)
- **What's New**: 이번 논문에서는 야간 안개 이미지를 향상시키기 위한 다중 작업 지향 프레임워크인 MToIE(Multi-task-oriented nighttime haze imaging enhancer)를 제안합니다. 이 프레임워크는 낮 동안의 안개 제거(Daytime Dehazing), 저조도 향상(Low-light Enhancement), 야간 안개 제거(Nighttime Dehazing)라는 세 가지 작업을 통합하여 보다 효과적인 이미지 복원을 목표로 하고 있습니다. MToIE는 특정한 노드 학습 메커니즘과 멀티 수용 범위 향상(Multi-receptive Field Enhancement) 모듈을 활용하여 복잡한 환경에서의 이미지 품질을 효율적으로 향상시킵니다.

- **Technical Details**: MToIE는 특화된 작업 노드 학습 메커니즘(task-oriented node learning)과 자기 주의 모듈(self-attention module)을 기반으로 설계되어 세 가지 차별화된 왜곡 유형을 처리할 수 있습니다. 이 네트워크는 세 가지 병렬 깊이 분리 가능한 컨볼루션 가지를 통해 다중 스케일 특성을 추출하는 멀티 수용 범위 향상 모듈(multi-receptive field enhancement module)을 포함하여, 최소한의 계산 오버헤드로 포괄적인 공간 정보를 캡처합니다. 또한, 하이브리드 손실 함수(hybrid loss function)를 통해 이미지 재구성 품질과 시각적 특성을 최적화합니다.

- **Performance Highlights**: MToIE는 다양한 기상 및 조명 조건에서 기존 방법들을 능가하는 성능을 나타내며, 이후의 고수준 비전 과제에서 더 높은 정확도와 신뢰성을 달성합니다. 실험 결과는 MToIE가 이미지 품질 향상에서 유의미한 성과를 내며, 특히 복잡한 저조도 및 안개 환경에서 효과적임을 입증했습니다. 연구진은 해당 모델의 코드도 공개하였으며, 이를 통해 향후 연구와 개발에 기여할 것으로 기대됩니다.



### Integrating Physics and Data-Driven Approaches: An Explainable and Uncertainty-Aware Hybrid Model for Wind Turbine Power Prediction (https://arxiv.org/abs/2502.07344)
- **What's New**: 이 연구는 물리 기반 모델과 비모수적 데이터 기반 모델의 장점을 결합한 하이브리드 반설계 모델을 제안합니다. 이는 프랑스의 ‘라 오트 본’ 풍력 발전소에서 수집된 데이터를 활용하여 풍력 발전 예측의 정확도를 37% 향상시키는 결과를 가져왔습니다. SHAP 값 분석을 통해 입력 요소가 모델 출력에 미치는 영향을 평가하고, 신뢰성 있는 예측을 위해 정형화된 분위수 회귀 방법을 사용하여 예측 불확실성을 정량화합니다.

- **Technical Details**: 제안된 하이브리드 모델은 물리 기반 구성요소와 신경망을 결합하여 관측된 데이터와 물리 기반 모델의 예측 사이의 잔차를 학습합니다. 이 모델은 물리적 제약을 준수하는 한편, 다양한 입력 변수가 포함된 비모수적 하위 모델을 통해 보다 넓은 범위의 현상을 고려합니다. 결국, 모델의 해석 가능성을 유지하면서도 높은 예측 정확도를 성취하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 하이브리드 모델은 물리 기반 모델 대비 37%의 예측 정확도 향상 효과를 확인했습니다. 또한, 신뢰성 있는 예측을 제공하는 정형화된 분위수 회귀 방법이 활용되어 모델의 신뢰도를 높였습니다. 이러한 접근 방식은 풍력 발전의 전력 생산 예측에서 해석 가능성과 불확실성 정량화를 동시에 통합한 최초의 모델로, 향후 풍력 에너지 부문에서의 응용 가능성을 보여줍니다.



### Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering (https://arxiv.org/abs/2502.07340)
- **What's New**: NOVA는 대규모 언어 모델(LLMs)이 사용자의 지침을 따르도록 조정하고 허위정보(hallucination)를 줄이기 위한 새로운 프레임워크입니다. 본 연구에서는 LLM이 지침 데이터에서 익숙하지 않은 지식을 포함하는 경우 자주 발생하는 과신을 줄이기 위해, 고품질 데이터를 식별하기 위해 Internal Consistency Probing (ICP)와 Semantic Equivalence Identification (SEI) 기법을 도입합니다. 이 방법을 통해 지침 데이터와 타겟 응답과의 관계를 평가하여 모델의 이해도를 높이고 허위정보 발생을 억제합니다.

- **Technical Details**: NOVA는 두 가지 주요 기법인 Internal Consistency Probing (ICP)와 Semantic Equivalence Identification (SEI)를 포함합니다. ICP는 LLM이 지침에 대해 자기 생성된 응답 간의 일관성을 측정하여 해당 지식이 내재화되었는지 평가합니다. SEI는 생성된 응답을 의미적으로 군집화하고, 타겟 응답과 가장 잘 일치하는 군집을 찾음으로써 LLM이 타겟 응답을 얼마나 잘 이해하는지 평가합니다.

- **Performance Highlights**: NOVA 프레임워크의 실험 결과, 허위정보 발생률이 유의미하게 감소했으며, LLM이 지침을 따르는 능력 또한 경쟁력을 유지하는 것으로 나타났습니다. 이러한 개선은 고품질 데이터를 식별하고, LLM의 지식과 정렬하여 더 효과적인 학습을 가능하게 합니다. 이로 인해 실세계 앱에서 LLM의 신뢰성을 높일 수 있습니다.



### Music for All: Exploring Multicultural Representations in Music Generation Models (Camera Ready) (https://arxiv.org/abs/2502.07328)
Comments:
          17 pages, 5 figures, accepted to NAACL'25

- **What's New**: 이 논문은 Music-Language Models의 발전이 음악 자동 생성 능력을 향상시키고 있지만, 비서구 음악 장르의 저조한 표현 문제를 강조합니다. 현재 음악 데이터세트의 5.7%만이 비서구 장르에서 온 것으로 나타났습니다. 연구는 Parameter-Efficient Fine-Tuning (PEFT) 기술을 사용하여 이러한 편향을 완화할 수 있는 가능성을 조사합니다.

- **Technical Details**: AI 음악 생성 기술이 빠르게 발전하면서, autoregressive, diffusion-based 및 GAN-based 접근 방식이 고품질 음악을 생성하고 있습니다. 이 연구에서는 약 5000개의 연구 논문에서 음악 생성 연구에 대한 데이터를 수집하고, 비서구 음악 장르를 포함한 데이터세트의 편향을 분석했습니다. 새로운 데이터세트를 제안하는 152개의 논문을 통해 총 100만 시간 이상의 음악을 포함한 데이터세트를 조사했습니다.

- **Performance Highlights**: With the implementation of PEFT, Mustango는 Hindustani Classical 음악에서 8% 향상되었고, MusicGen은 Turkish Makam 음악에서 4% 향상되었습니다. 이 결과는 PEFT 기술이 저조한 장르에서 생성 음악의 품질을 향상시킬 수 있음을 보여주지만, 모든 모델이 모든 장르에 적응 가능하지 않다는 점을 강조합니다.



### CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction (https://arxiv.org/abs/2502.07316)
- **What's New**: 이 논문은 대규모 언어 모델의 추론 능력을 향상시키기 위한 새로운 접근법인 CodeI/O를 제안합니다. 기존의 연구가 수학 문제 해결이나 코드 생성과 같은 특정 영역에 주로 초점을 맞춘 것과 달리, 이제는 다양한 추론 과제에 대한 성능 개선을 목표로 합니다. CodeI/O는 문맥적으로 기초한 코드 내의 다양한 추론 패턴을 집약하여 자연어로 입력/출력을 예측하는 형식으로 변환하는 방식으로 작동합니다.

- **Technical Details**: 이 연구는 여러 출처에서 수집한 함수들을 변환하여 복잡한 추론 패턴을 포함하도록 설계되었습니다. 입력/출력 예측을 위해, 각 함수에 대해 텍스트 쿼리와 함께 예측을 요구하여 모델이 코드와 자연어를 통한 추론 능력을 배양하게 합니다. 이 과정에서는 DeepSeek-V2.5라는 오픈소스 모델을 사용하여 데이터를 처리하며, 45만 개 이상의 함수와 350만 개의 훈련 샘플이 포함됩니다.

- **Performance Highlights**: CodeI/O와 CodeI/O++는 7B에서 30B의 파라미터를 가진 4가지 기본 모델에 대해 검증되었으며, 14개의 다양한 기준에서 평가되었습니다. 이 새로운 접근법은 기존 벤치마크에 비해 높은 점수를 기록하며, 특정 소수의 평가 기준에서만 점수를 높이는 것이 아니라, 거의 모든 벤치마크에서 일관된 성과를 보였습니다. 이러한 결과는 CodeI/O의 다양한 추론 능력을 강조합니다.



### OpenGrok: Enhancing SNS Data Processing with Distilled Knowledge and Mask-like Mechanisms (https://arxiv.org/abs/2502.07312)
Comments:
          7 pages

- **What's New**: Lumen Labs는 SNS 데이터 처리에 대한 새로운 접근법을 제시합니다. 이 연구에서는 knowledge distillation을 활용하여 Grok 모델에서 Phi-3-mini 모델로 SNS 관련 지식을 전달하며 성능을 향상시킵니다. 특히, SNS 데이터의 독특한 특성을 잘 반영하는 mask-like 메커니즘을 도입하여 잡음의 영향을 줄이는 데 성공했습니다.

- **Technical Details**: 연구는 데이터 수집, 모델 미세 조정, mask-like 메커니즘의 세 가지 주요 단계로 구성됩니다. 먼저, Grok 모델에서 적절한 SNS 데이터를 수집하기 위해 다양한 프롬프트를 생성하고, 그에 따른 응답을 필터링하여 품질을 높입니다. Phi-3-mini 모델은 생성된 데이터를 통해 미세 조정되며, cross-entropy loss를 최소화하는 방법으로 학습하게 됩니다.

- **Performance Highlights**: 본 방법은 여러 SNS 데이터 처리 작업에서 SOTA(최첨단) 성능을 달성하며 기존 모델들보다 우수한 결과를 보입니다. 구체적으로, Grok, Phi-3, 그리고 GPT-4와 비교했을 때, 제안된 접근법은 최신의 SNS 데이터 처리 문제를 효과적으로 해결할 수 있는 능력을 보여주었습니다.



### TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation (https://arxiv.org/abs/2502.07306)
- **What's New**: 본 연구에서는 Vision-Language Navigation (VLN) 작업을 해결하기 위한 모듈 방식의 접근법을 제안합니다. 이 접근법은 자연어로 제공된 내비게이션 지침을 처리하기 위해 최첨단 LLMs와 VLMs의 제로샷(zero-shot) 기능을 활용하여, 네 가지 하위 모듈로 문제를 분해합니다. 특히, VLM을 사용해 시각적 관찰에서 랜드마크 이름을 접합하고, 동적 프로그래밍을 통해 경로 정렬 점수를 계산합니다.

- **Technical Details**: 저자들은 R2R-Habitat 데이터셋의 복잡한 지침에 초점을 맞춰 VLN 작업을 해결하는 여덟 가지 단계를 포함하는 모듈형 접근법을 제시합니다. 1단계에서는 에이전트가 데이터셋의 훈련 에피소드를 사용하여 환경의 위상 지도를 구축합니다. 각 노드는 360° RGB 파노라마로 표현되며, 각 엣지는 노드 쌍 간의 연결을 나타내는 가중치 1을 가집니다.

- **Performance Highlights**: 이 모듈형 접근법은 복잡한 R2R-Habitat 지침 데이터셋에서 기존의 점유 맵 기반 접근법과 비교하여 우수한 성능을 보여주었습니다. 특히 미세한 랜드마크 및 행동 구문 접합의 복잡성을 나타내고, 기존 접근법의 강점과 약점을 분석하여 시각 언어 지침에 대한 현재의 LLMs와 VLMs의 성능을 평가합니다.



### Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification (https://arxiv.org/abs/2502.07299)
Comments:
          12 pages main text with 6 pages Appendix

- **What's New**: 이번 논문은 다양한 생물학적 기능을 포괄하는 포괄적 프레임워크인 Life-Code를 제안합니다. Life-Code는 DNA, RNA, 단백질의 상호작용을 이해하기 위해 데이터와 모델 파이프라인을 재설계하였습니다. 이를 통해 서로 연결된 생물학적 맥락 내에서 복잡한 상호작용을 포착할 수 있는 방식으로, 멀티-오믹스(multi-omics) 데이터를 통합한 새로운 접근법을 제공합니다.

- **Technical Details**: Life-Code의 데이터 흐름 설계에서는 RNA를 역전사(reverse-transcribing)하여 아미노산을 뉴클레오타이드 기반 서열로 역번역(reverse-translate)합니다. 또한 코돈 토크나이저(codon tokenizer)와 마스크드 모델링(masked modeling) 사전 훈련(pre-training)을 사용하는 하이브리드 장기 서열 아키텍처(hybrid long-sequence architecture)를 설계하여 코딩 및 비코딩 영역의 상호작용을 인코딩합니다. 이러한 구조적 접근법은 모델이 유전적 데이터를 종합적으로 활용할 수 있도록 돕고 있습니다.

- **Performance Highlights**: Life-Code는 DNA, RNA, 단백질 관련 다양한 작업에서 최첨단 성능을 기록했습니다. 실제 실험 결과는 Life-Code의 통합된 접근 방식이 세 가지 기본 모달리티에서 효과적임을 보여줍니다. 이는 멀티-오믹스 분석 및 해석을 진전시키는 데 있어 중요한 잠재력을 지니고 있음을 시사합니다.



### KPIs 2024 Challenge: Advancing Glomerular Segmentation from Patch- to Slide-Lev (https://arxiv.org/abs/2502.07288)
- **What's New**: 이 논문에서는 만성 신장 질환(Chronic Kidney Disease, CKD)의 진단 및 치료에 대한 기준을 마련하기 위해 'Renal Pathology Image Segmentation Challenge'(KPIs)를 소개합니다. 이 도전 과제는 60개 이상의 Periodic Acid Schiff (PAS) 염색 슬라이드 이미지를 포함하여, 10,000개 이상의 주석이 달린 사구체(glomeruli)를 포함하는 방대한 데이터셋을 제공합니다. 이는 다양한 CKD 모델과 조직 상태에 적응할 수 있는 혁신적인 segmentation 방법의 개발을 장려합니다.

- **Technical Details**: KPIs 챌린지는 두 가지 주요 작업으로 구성되어 있습니다. 첫 번째 작업은 Patch-Level Segmentation으로, 주어진 이미지 패치 내의 사구체를 세분화하는 것입니다. 두 번째 작업은 Whole Slide Image-Level Segmentation 및 Detection이며, 전체 슬라이드에서 사구체를 식별하는 작업으로, 모델 성능은 Dice Similarity Coefficient (DSC)와 F1-score로 평가됩니다.

- **Performance Highlights**: 이 대회는 신장 질병 모델의 다양성을 포괄하며, 각 참가자는 특정 CKD 모델을 기반으로 한 세분화 알고리즘을 개발하여 정확한 사구체 세분화를 목표로 합니다. 이를 통해, 신장 병리학 분석의 발전과 질병 연구에 기여할 수 있는 새로운 벤치마크를 확립하는 것을 목표로 하고 있습니다.



### Small Language Model Makes an Effective Long Text Extractor (https://arxiv.org/abs/2502.07286)
Comments:
          AAAI'25, 9 pages, 1 appendix pages

- **What's New**: 이 논문은 긴 텍스트에서 엔터티를 효과적으로 추출하기 위한 경량(span-based) NER 방법인 SeNER을 소개합니다. SeNER은 bidirectional arrow attention 메커니즘과 LogN-Scaling을 통해 긴 텍스트를 효율적으로 인코딩하며, BiSPA 메커니즘을 사용하여 후보 토큰 쌍 간의 상호 작용을 모델링합니다. 이러한 방법론을 통해 GPU 메모리 사용량을 절감하면서도 높은 정확도를 유지합니다.

- **Technical Details**: SeNER의 핵심 구성 요소로는 bidirectional arrow attention 메커니즘과 BiSPA가 있습니다. 전자는 로컬 및 글로벌 컨텍스트를 동시에 인코딩하며, 후자는 토큰 쌍 간 상호작용을 모델링하고 불필요한 계산을 줄입니다. LogN-Scaling을 통해 다양한 입력 길이에 대한 엔트로피의 불안정성을 해결하며, 학습 과정에서 전체 단어 마스킹 전략과 LoRA 기술을 활용합니다.

- **Performance Highlights**: SeNER은 세 가지 긴 NER 데이터셋에서 최첨단 추출 정확도를 달성하며, 기존의 고급 span-based NER 방법보다 6666배 긴 텍스트를 처리할 수 있는 능력을 보여줍니다. 또한, 상대적으로 적은 모델 파라미터로 높은 성능을 유지하는 것이 특징입니다. 광범위한 실험 결과는 SeNER의 우수성을 뒷받침하고 있습니다.



### MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Managemen (https://arxiv.org/abs/2502.07280)
- **What's New**: 본 연구에서는 메모리 인스턴스 게이티드 변환기(MIGT) 프레임워크를 도입하여 포트폴리오 관리를 효율적으로 수행할 수 있는 새로운 접근법을 제시합니다. 이 새로운 프레임워크는 게이티드 인스턴스 주의(Gated Instance Attention) 모듈을 포함하여 투자 수익을 극대화하고 학습 과정의 안정성을 보장하며 이상치 영향력을 줄이는 데 중점을 두고 있습니다. 이를 통해 기존의 DRL 기반 포트폴리오 관리 솔루션들이 겪고 있던 여러 제약을 해결하는 데 기여하고 있습니다.

- **Technical Details**: MIGT는 딥 강화 학습(Deep Reinforcement Learning) 환경 하에서 자산 클래스 간 자본 재배분을 수행합니다. 본 모델은 고차원 텐서를 입력으로 받아 과거 데이터와 기술 지표를 포함하고, 하루 단위의 거래 기간을 설정하여 포트폴리오 가치를 평가합니다. 이 과정에서 트레이딩 액션은 판매, 구매, 보유로 정의되며, 거래 비용은 거래 가치의 0.1%로 설정되었습니다.

- **Performance Highlights**: MIGT 프레임워크는 다우존스 산업지수(Dow Jones Industrial Average 30)에 대해 15개의 다른 전략과 비교했을 때 누적 수익률에서 최소 9.75%의 개선을 보였으며, 위험 대비 수익 비율(Sharpe, Sortino, Omega 비율)에서도 최소 2.36%의 증가를 기록했습니다. 이러한 결과는 MIGT의 효과성을 입증하며 DRL 기반 포트폴리오 관리의 중요한 진전을 나타냅니다.



### Exploratory Diffusion Policy for Unsupervised Reinforcement Learning (https://arxiv.org/abs/2502.07279)
- **What's New**: 이번 연구에서는 Exploratory Diffusion Policy (EDP)를 제안하여 비지도 강화 학습(Unsupervised Reinforcement Learning)에서의 탐사의 효율성을 높이기 위한 접근 방식을 소개합니다. 기존의 방법들이 한계가 있는 동질적이지 않은 데이터에서의 적합성 문제를 해결하여, 더 나은 탐사와 다운스트림 작업에 대한 효율적인 초기화를 가능하게 합니다. EDP는 diffusion models의 강력한 표현 능력을 활용하여, 탐사한 데이터를 잘 적합시킵니다.

- **Technical Details**: EDP는 수집된 데이터를 기반으로 replay buffer의 분포를 추정하며, 이를 통해 에이전트가 미지의 상태를 탐사하도록 유도하는 내재적 보상(intrinsic reward)을 설계합니다. 이 과정에서 Gaussian 정책을 활용하여 환경 상호작용 시의 행동 생성의 효율성을 높였습니다. 또한, Q 함수와 diffusion policy를 번갈아 최적화하는 방법을 통해 다운스트림 작업에 대한 빠른 적응을 촉진합니다.

- **Performance Highlights**: 다양한 벤치마크를 통해 EDP의 탐사 및 적응 성능이 입증되었습니다. Maze2d에서 EDP는 기존 방법들 대비 훨씬 더 넓은 상태 범위를 탐사하며, 그 다양성에서도 두드러진 성과를 보였습니다. URLB에서의 실험 결과, EDP는 다운스트림 작업에 신속하게 적응하며 기존 탐사 방법들보다 뛰어난 성능을 나타냈습니다.



### Enhancing Video Understanding: Deep Neural Networks for Spatiotemporal Analysis (https://arxiv.org/abs/2502.07277)
Comments:
          29 pages, 25 figures

- **What's New**: 이 논문에서는 비디오(video) 콘텐츠를 분석하고 이해하는 알고리즘의 수요가 급증하고 있다는 점을 강조합니다. 비디오는 온라인 정보 공유의 주요 방식으로 자리 잡았고, 이러한 경향은 계속될 것으로 보입니다. 특히, 최근 딥 뉴럴 네트워크(deep neural networks)를 활용한 비디오 이해 분야의 진전을 다루고 있습니다.

- **Technical Details**: 논문은 비디오에서 발견되는 시공간(spatiotemporal) 특징들을 탐구하며, 이러한 특징들을 추출하고 분류하는 방법을 소개합니다. 비디오 이해 모델의 구조적 설계와 주요 문제, 그리고 이를 해결하기 위한 몇 가지 솔루션을 검토할 것입니다. 또한, 비디오 이해 및 동작 인식(action recognition) 데이터셋의 비교도 다루어집니다.

- **Performance Highlights**: 딥 뉴럴 네트워크를 활용한 비디오 설명 및 특징 추출에서 긍정적인 결과를 보여주었다고 평가됩니다. 이러한 알고리즘은 비디오의 사건과 객체를 효과적으로 설명하는 데 중요한 역할을 하고 있습니다. 앞으로의 연구 방향으로는 더욱 개선된 모델과 데이터셋 비교 분석이 포함될 것으로 기대됩니다.



### Dataset Ownership Verification in Contrastive Pre-trained Models (https://arxiv.org/abs/2502.07276)
Comments:
          Accepted by ICLR2025

- **What's New**: 최근의 아카이브 논문에서는 고품질 오픈소스 데이터셋의 소유권 검증을 위한 새로운 방법을 제안하고 있습니다. 기존의 기술들은 주로 감독된 모델에 국한되어 있으며, 자가 감독(self-supervised) 프리트레인(pre-trained) 모델로 직접 확장할 수 없었습니다. 이 논문에서는 대칭적 학습(contrastive learning)을 통해 자가 감독 프리트레인 모델을 위한 첫 데이터셋 소유권 검증 방법을 도입하여 데이터 소유자가 자신의 권리를 옹호할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 관찰에 기초하여 구성됩니다. 첫 번째는 유니어리 관계(unary relationship)로, 대칭학습을 통해 프리트레인된 인코더가 동일한 샘플의 변형에 대해 훨씬 더 유사한 표현을 생성한다는 것입니다. 두 번째는 이진 관계(binary relationship)로, 보인 샘플 간의 쌍별 유사성이 데이터 증강 후에도 크게 변하지 않는다는 점입니다. 이러한 관계의 차이를 '대칭적 관계 간극(contrastive relationship gap)'이라고 정의하고, 이를 통해 방어자는 의심스러운 인코더가 자신의 데이터셋으로 프리트레인 되었는지를 확인할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 모든 이전 연구를 초월하여 p-value가 0.05 이하로 통계적으로 유의미한 결과를 보여줍니다. SimCLR, BYOL, SimSiam, MOCO v3 및 DINO를 포함한 여러 대칭적 프리트레인 모델에서 검증의 유효성을 입증하였으며, 이는 데이터셋 소유권 검증 분야에 중요한 기여를 하고 있습니다. 또한, 제안된 접근법이 대량의 데이터셋을 요구하지 않고도 정확한 검증을 가능하게 함을 시사합니다.



### Cost-Efficient Continual Learning with Sufficient Exemplar Memory (https://arxiv.org/abs/2502.07274)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문은 지속 학습(Continual Learning, CL) 분야에서 기존의 엄격한 사례 메모리 제약을 완화하는 새로운 접근 방식을 제안합니다. 연구자들은 모델의 가중치 공간에서 직접 작동하여, 가중치 리셋팅(weight resetting) 및 평균화(averaging) 기법을 조합하여 성능을 향상시키고 있습니다. 이러한 방법을 통해 기존 방법의 1/4 또는 1/3에 해당하는 컴퓨팅 비용으로 최첨단 성능을 달성할 수 있음을 밝혔습니다.

- **Technical Details**: 지속 학습의 가장 큰 도전 과제 중 하나는 안정성-유연성 딜레마(stability-plasticity dilemma)입니다. 많은 CL 알고리즘은 메모리 사용을 최소화하는 것을 목표로 하여 엄격한 메모리 제약 하에 작동해왔습니다. 그러나 연구자들은 최근 GPU 사용량과 같은 계산 비용이 메모리 비용보다 더 중요한 요소가 될 수 있음을 강조하고, 충분한 사례 메모리를 활용하여 훈련 비용을 줄일 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 제안된 방법은 나은 파인튜닝(fine-tuning) 기법과 경쟁하며, 메모리 제약이 완화된 조건에서 성능을 높이면서도 훈련 비용을 크게 줄일 수 있음을 보여줍니다. 연구 결과는 현재의 방법보다 더 낮은 계산 오버헤드로 우수한 성능을 발휘하는 것을 입증하며, 실험적 기반을 제공하여 CL 애플리케이션의 효율성을 높이는 데 기여합니다.



### Variational Learning Induces Adaptive Label Smoothing (https://arxiv.org/abs/2502.07273)
- **What's New**: 본 논문에서는 변분 학습(variational learning)을 통해 자연스럽게 적응형 레이블 스무딩(adaptive label smoothing)을 유도함을 보여줍니다. 이러한 레이블 스무딩은 각 예제에 맞춰 라벨 노이즈(label noise)를 최적화하여 레이블 오류(labeling errors) 및 분포 변화(distribution shifts)에 효과적으로 대응할 수 있도록 합니다. 연구팀은 변분 목표(variational objective)의 최적화 과정에서 발생하는 자연스러운 적응성을 활용하여, 기존 방식보다 우수한 성과를 달성할 수 있음을 증명하였습니다.

- **Technical Details**: 적응형 레이블 스무딩은 각각의 데이터 예제에 따라 레이블 노이즈를 조정하는 방법을 말합니다. 연구 진영에서는 이를 위해 다양한 전략이 제시되었지만, 효과적인 전략을 설계하는 것이 항상 쉬운 일은 아닙니다. IVON(Improved Variational Online Newton)이라는 변분 알고리즘을 통해, 논문에서는 다수의 문제에 대해 레이블 노이즈의 정확한 형태를 도출하였고, 이를 통해 각 예제의 특성에 맞춘 레이블 노이즈의 동작을 연구하였습니다.

- **Performance Highlights**: 본 연구에서 제안한 방법론은 레이블 오류가 있는 경우에도 기존 레이블 스무딩 방법들보다 최대 9%의 정확도 향상을 보였으며, 때로는 데이터에 따라 달라지는 노이즈의 경우 약 50% 이상 증가한 성능을 기록하였습니다. 추가적으로, 변분 학습이 비정형적 또는 모호한 예제에 대해 높은 노이즈를 할당하는 경향이 있음을 보여주었으며, 이는 레이블 스무딩 문헌과 베이지안 방법(Bayesian methods) 간의 새로운 연결 고리를 제공합니다.



### Fairness in Multi-Agent AI: A Unified Framework for Ethical and Equitable Autonomous Systems (https://arxiv.org/abs/2502.07254)
- **What's New**: 이 논문은 분산된 다중 에이전트 시스템에서의 공정성(fairness)을 보장하는 데 발생하는 많은 도전 과제를 다룹니다. 특히 공정성을 에이전트 상호작용의 동적이고 창발적인 속성으로 다루는 새로운 프레임워크를 소개합니다. 이 연구는 AI 윤리(ethics)와 시스템 설계(design) 간의 간극을 메우고자 하며, 사회적 가치(societal values)와 에이전트 행동을 조정하는 기초를 제공합니다.

- **Technical Details**: 새로 제안된 프레임워크는 공정성 제약(fairness constraints), 편향 완화(bias mitigation) 전략, 그리고 인센티브 메커니즘을 통합합니다. 이러한 요소들은 자율 에이전트의 행동을 사회적 가치와 맞추면서도 효율성과 견고성(robustness)을 균형 있게 유지할 수 있도록 디자인되었습니다. 연구는 공정성 관련 제약 조건을 통합했을 때 의사결정 과정이 보다 공정해짐을 실증적으로 보여줍니다.

- **Performance Highlights**: 실험적 검증을 통해 제안된 프레임워크가 에이전트 간의 협력적 의사결정을 개선하고, 시스템의 신뢰성을 높임을 입증했습니다. 이 연구결과는 사회적 책임(socially responsible) 및 투명성(transparency)을 지닌 다중 에이전트 AI 시스템 구축에 기여하고 있습니다.



### NARCE: A Mamba-Based Neural Algorithmic Reasoner Framework for Online Complex Event Detection (https://arxiv.org/abs/2502.07250)
- **What's New**: 이번 논문은 기계 학습 모델들이 단기 인식 작업에서는 우수하지만 장기 관찰을 통한 고차원 통찰력을 얻는 데 어려움이 있음을 지적합니다. CEs(Complex Events)는 짧은 원자 이벤트(AEs)로 구성된 복잡한 사건으로, 이를 실시간으로 감지하는 데 어려움이 있습니다. 연구팀은 상태 기반 방법들이 CEs 감지에 효과적일 것으로 가정하고, Mamba라는 상태 공간 모델이 기존 아키텍처보다 우수하다는 것을 실험을 통해 입증하였습니다.

- **Technical Details**: Mamba 모델은 장기 메모리 없이 사건의 진행을 상태 전이를 통해 포착하는 및 긴 잡음이 섞인 센서 데이터에서 의미 있는 패턴을 추출하는 데 초점을 맞춥니다. 또한, NARCE라는 새로운 프레임워크를 제안하여 신경 알고리즘적 추론(Neural Algorithmic Reasoning, NAR)을 사용하여 CE 규칙 학습을 센서 데이터와 분리했습니다. 이를 통해 LLMs(대형 언어 모델)로 생성한 합성 개념 추적을 사용하여 규칙을 독립적으로 학습하게 됩니다.

- **Performance Highlights**: NARCE는 기존 모델보다 정확성, 일반화 능력, 데이터 효율성 면에서 우수한 성과를 보입니다. 이 프레임워크는 주석 비용을 상당히 절감하면서도 강력한 CE 감지를 가능하게 합니다. 결국, NARCE는 센서 데이터 양을 줄이면서도 더 나은 성능을 보여 CEs 감지 체계의 발전에 기여합니다.



### Linear Transformers as VAR Models: Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting (https://arxiv.org/abs/2502.07244)
- **What's New**: 이 연구는 Autoregressive attention 기반의 시간 시계열 예측(TSF) 분야에서의 혁신적인 접근을 제시합니다. 특히, 단일 선형 attention 계층이 동적 벡터 자기회귀(VAR) 구조로 해석될 수 있음을 보여줍니다. 또한, 기존의 다층 Transformer들이 자기회귀 예측 목표와의 구조적 불일치로 해석 가능성과 일반화 능력이 하락함을 설명합니다. 이를 해결하기 위해 MLP, attention 및 input-output 흐름을 재정렬하여, 다층 선형 attention을 VAR 모델로 정렬할 수 있음을 입증하였습니다.

- **Technical Details**: 시간 시계열 예측(TSF)에서는 주어진 입력 시퀀스에서 미래 값을 예측하는 것이 목표입니다. 이 연구에서는 선형 attention과 VAR 구조 간의 관계를 탐구하며, 단일 계층 선형 attention이 VAR 모델처럼 동작할 수 있음을 보여줍니다. 또한, 기존 Transformer 설계가 자기회귀 예측 목표와의 불일치를 초래하는 여러 요인을 분석하여, MLP 및 attention의 구조를 재조정하면 다층 선형 attention이 동적 VAR 모델로 효과적으로 작동할 수 있는 방법을 제안합니다.

- **Performance Highlights**: 이 연구에서 제안하는 SAMoVAR 모델은 다층 선형 attention을 VAR 모델로 정렬하여 예측 정확도와 해석 가능성을 향상시킵니다. 실험 결과 SAMoVAR는 기존의 TSF 모델들과 비교하여 성능이 뛰어나며, 해석성 및 계산 효율성 측면에서도 우수한 결과를 보였습니다. 이 모델은 다변량 시계열 예측에서 강력한 성능을 발휘하며, 기존의 최첨단 모델들보다 더 나은 결과를 얻는 데 성공하였습니다.



### Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglemen (https://arxiv.org/abs/2502.07243)
Comments:
          Accepted by ICLR 2025

- **What's New**: 새로운 연구에서는 Vevo라는 다재다능한 제로샷(Zero-shot) 음성 모방 프레임워크를 제안합니다. 이 프레임워크는 음색(timbre)과 스타일(style)의 제어가 가능하여, 기존 방법론이 가진 제한점을 극복합니다. Vevo는 자가 감독(Self-supervised) 학습 기법을 활용하여, 스피치의 음색, 스타일 및 언어적 내용을 점진적으로 분리하는 방식을 채택합니다.

- **Technical Details**: Vevo는 두 가지 주요 단계로 작동합니다: 첫째, 콘텐츠-스타일 모델링(Content-Style Modeling) 단계에서, 텍스트나 음성의 콘텐츠 토큰(content tokens)을 입력으로 사용하여 회귀 변환기(autoregressive transformer)를 통해 콘텐츠-스타일 토큰(content-style tokens)을 생성합니다. 둘째, 음향 모델링(Acoustic Modeling) 단계에서는 내용-스타일 토큰을 입력으로 사용하여 흐름-매칭 변환기(flow-matching transformer)를 통해 음향 표현(acoustic representations)을 생성합니다. VQ-VAE 방식으로 연속 숨겨진 특성을 토크나이즈합니다.

- **Performance Highlights**: Vevo는 60K 시간의 오디오북 음성 데이터로 오로지 자가 감독 학습만을 통해 훈련되었으며, 특정 스타일 데이터 세트에서의 파인튜닝 없이도 악센트 및 감정 전이 작업에서 기존 방법과 동등하거나 뛰어난 결과를 보입니다. 또한, 제로샷 음성 변환(zero-shot voice conversion) 및 텍스트-음성 변환(text-to-speech) 작업에서의 효과성은 Vevo의 강력한 일반화 및 다재다능함을 더욱 입증합니다.



### Contextual Gesture: Co-Speech Gesture Video Generation through Context-aware Gesture Representation (https://arxiv.org/abs/2502.07239)
- **What's New**: 이번 논문에서는 Contextual Gesture라는 프레임워크를 소개하여, 사람의 말과 제스처를 잘 연동하여 고해상도의 제스처 비디오 생성을 지원합니다. 이 프레임워크는 (1) 시간적으로 연관된 음성과 제스처를 정렬하고, (2) 지식 증류를 통해 방향성을 가진 제스처 표현을 학습하며, (3) 제스처의 키포인트를 연결하기 위한 구조 인식을 포함한 개선 모듈을 통해 제스처 생성 및 비디오 품질을 강화합니다.

- **Technical Details**: Contextual Gesture 프레임워크는 두 가지 주요 구성 요소를 사용하여 제스처 생성을 수행합니다. 첫째, 시간적 대비 학습을 통해 제스처와 음성 간의 내재적 연결을 발견하며, 둘째, 제스처 생성 과정에서 양방향 마스킹 프리트레이닝을 활용하여 음성과 제스처의 정렬을 정교하게 합니다. 이러한 과정은 제스처가 전달하고자 하는 의미를 선명하게 표현할 수 있도록 돕습니다.

- **Performance Highlights**: 상세한 실험 결과, Contextual Gesture는 기존의 방법들보다 향상된 정량적 및 정성적 지표를 달성하였으며, 긴 시퀀스 생성 및 비디오 제스처 편집 기능을 지원합니다. 이로 인해 현실감 있는 제스처 비디오를 생성할 수 있는 가능성이 높아져, 인간-컴퓨터 상호작용이 한층 더 발전할 것으로 기대됩니다.



### Diffusion Suction Grasping with Large-Scale Parcel Datas (https://arxiv.org/abs/2502.07238)
- **What's New**: 최근 물체 흡입 그립(grasping) 분야에서 주요한 진전을 보여주었지만, 복잡한 소포 처리 시나리오에서는 여전히 상당한 도전 과제가 남아 있습니다. 본 연구에서는 25,000개의 복잡한 장면과 4억 1천만 개의 정밀 주석이 매겨진 흡입 그립 자세를 포함하는 Parcel-Suction-Dataset을 제안합니다. 또한 Diffusion-Suction이라는 혁신적인 프레임워크를 통해 흡입 그립 예측을 조건부 생성 작업으로 재구성하였습니다.

- **Technical Details**: Diffusion-Suction은 시각적으로 조건화된 포인트 클라우드 관찰에서 얻은 지침을 통해 무작위 노이즈를 반복적으로 정제하여 흡입 그립 점수 맵을 생성합니다. 우리는 PointNet++를 이용하여 포인트 클라우드의 글로벌 정보를 추출하고, 경량화된 Denoising Block을 사용하여 중요한 특징을 강조합니다. 이 방식은 훈련 단계에서 가우시안 노이즈를 추가하고, 추론 단계에서 학습된 확산 과정을 역전시켜 신뢰할 수 있는 흡입 그립 점수를 생성합니다.

- **Performance Highlights**: Diffusion-Suction 방식은 Parcel-Suction-Dataset 및 공개 SuctionNet-1Billion 벤치마크에서 이전 모델들과 비교하여 새로운 최첨단 성능을 달성했습니다. 본 연구에 대한 실험 결과는 Diffusion-Suction이 기존 방법보다 우수함을 보여주며, 다양한 실험을 통해 그 효과와 특징을 심층 분석했습니다. 이로 인해, 복잡한 소포 장면에서도 신뢰할 수 있는 흡입 그립 예측이 가능해졌습니다.



### LUNAR: LLM Unlearning via Neural Activation Redirection (https://arxiv.org/abs/2502.07218)
- **What's New**: LUNAR는 대용량 언어 모델(LLMs)에서 비공식적인 정보를 제거하는 새로운 방법론으로, 기존의 방법보다 더 높은 통제 가능성을 제공합니다. LUNAR는 Linear Representation Hypothesis에 기반하여 불학습된 데이터의 표현을 리다이렉션하여 모델의 응답 능력을 저하시키지 않으면서 비공식적 데이터를 처리할 수 있는 능력을 증대합니다. 이 방법은 다양한 모델에서 'Deviation Score'를 통해 2.9배에서 11.7배 향상된 성능을 나타냈습니다.

- **Technical Details**: LUNAR는 모델 내부에서 생성된 표현의 선형 부분 공간과 관련된 중요한 행동을 최적화합니다. LUNAR는 multilayer perceptron(MLP) 다운 프로젝션을 최적화하여, 불학습된 데이터 포인트의 개념적 표현이 모델이 응답할 수 없음을 명시적으로 표현하는 영역으로 이동하도록 합니다. 이로 인해 기존의 비공식적 제거 방법들이 간과했던 갖가지 부작용을 완화하며, 실질적인 연속 불학습 요청 처리에서의 유연성을 보여줍니다.

- **Performance Highlights**: LUNAR는 또한 흰 상자 공격에 대해 강인성을 가지고 있으며, 실제 세계의 다양한 시나리오를 처리하는 데 있어 다재다능성을 입증하였습니다. 정량적 분석 및 질적 예제를 통해 LUNAR는 일관된 응답을 생성할 수 있는 뛰어난 통제 가능성을 보여주었으며, 이는 기존 방법의 원치 않는 부작용을 완화합니다. LUNAR는 메모리와 계산 효율성도 동시에 제공하며, PEFT 방법과 결합했을 때 속도 향상 효과를 발휘하면서도 비공식적 제거 성능을 유지합니다.



### SparseFormer: Detecting Objects in HRW Shots via Sparse Vision Transformer (https://arxiv.org/abs/2502.07216)
Comments:
          This paper is accepted to ACM MM 2024

- **What's New**: 이번 논문에서는 고해상도 넓은 화면 사진(High-Resolution Wide, HRW)에서 물체 감지의 정확성과 속도를 개선하기 위한 새로운 모델 SparseFormer를 제안합니다. 기존의 클로즈업 감지 모델이 HRW 환경에서 효과적이지 않은 문제를 해결하기 위해, SparseFormer는 선택적으로 주목할 수 있는 토큰(attentive tokens)을 사용하여 희소하게 분포된 창(window)에서 물체를 탐지합니다. 이로 인해 지역적 및 전역적 주의를 탐색 가능하게 합니다.

- **Technical Details**: SparseFormer는 특징 추출 시 중요성을 평가하는 ScoreNet을 학습하여, 중요한 영역에 집중할 수 있도록 합니다. 또한 HRW 이미지를 비겹치는 창으로 나누어 coarse-grained(거칠고 큰) 및 fine-grained(세밀한) 특징을 동시에 추출합니다. 새로운 Cross-slice non-maximum suppression(C-NMS) 알고리즘을 통해 노이즈가 많은 창에서 물체를 더 정밀하게 로컬라이즈하며, 효과적인 다중 스케일 전략을 사용하여 정확도를 향상시킵니다.

- **Performance Highlights**: PANDA 및 DOTA-v1.0이라는 두 개의 HRW 벤치마크 데이터셋에서 실험한 결과, SparseFormer는 기존의 최첨단 방법들보다 물체 감지 정확도를 최대 5.8%까지, 속도를 최대 3배까지 향상시켰습니다. 이러한 성과는 대규모 변화를 처리하고 다양한 크기의 물체를 정확하게 감지할 수 있는 가능성을 보여줍니다.



### Pareto Optimal Algorithmic Recourse in Multi-cost Function (https://arxiv.org/abs/2502.07214)
- **What's New**: 이 논문은 결정 시스템에서 알고리즘적 대응(algorithmic recourse)의 개념을 제시하며, 개인의 특성을 변경하여 원하는 결과를 얻기 위한 최소 비용의 행동을 식별하는 방법론을 다룹니다. 기존의 기법들이 주로 미분 가능성을 가정하는 반면, 본 연구는 비미분성 및 이산 다중 비용 함수를 처리할 수 있는 프레임워크를 제안합니다. 이는 다양한 기준을 고려해 Pareto 최적화(Pareto optimal)된 대응 추천을 찾는 데 초점을 맞춥니다.

- **Technical Details**: 제안한 방법은 다중 목표 최적화 문제로 recourse를 수립하고, 각 기준의 중요도에 따라 가중치를 할당하여 최적의 행동을 탐색합니다. epsilon-net 개념을 도입하여 근사화된 Pareto 최적 행동을 식별하는 능력을 증명합니다. 이러한 접근법은 그래프의 크기가 클 경우에도 확장성(scability)을 보여줍니다.

- **Performance Highlights**: 실험 결과는 서로 다른 기준 간의 거래(trade-off)를 시연하고, 기존의 휴리스틱(heuristic) 방법들과 비교하여 보다 강력한 이론적 토대를 제공함을 확인했습니다. 또한, 제안된 방법이 실제 요구 사항에 보다 잘 부합하는 recourse 제안들을 도출함을 보여주고, 다양한 환경에서 사용자의 결정을 이해하고 의문을 제기할 수 있는 능력을 향상시킵니다.



### Evaluation for Regression Analyses on Evolving Data Streams (https://arxiv.org/abs/2502.07213)
Comments:
          11 Pages, 9 figures

- **What's New**: 본 논문에서는 변동하는 데이터 스트림에서 회귀 분석의 어려움을 탐구하고, 이를 위해 표준화된 평가 프로세스를 제안합니다. 특히, 덜 연구된 개념 드리프트인 incremental drift를 포함한 다양한 드리프트 유형을 시뮬레이션할 수 있는 혁신적인 방법론을 소개합니다. 이 연구는 데이터 스트림에서의 회귀 및 예측 구간 작업에 대해 효과적이고 강건한 접근 방식을 검증하는 종합적 실험을 수행합니다.

- **Technical Details**: 스트리밍 데이터(straming data)는 지속적이고 실시간으로 데이터가 흐르는 것으로, 센서, 로그, 온라인 활동 등이 일반적인 데이터 소스에 포함됩니다. 회귀 과제는 X가 다차원 벡터일 때, Y는 연속 값을 가지며(time step의 관측 순간을 표시) 이는 전통적으로 슈퍼바이즈드 러닝(supervised learning)에서 다루어집니다. 데이터 스트림 내에서 개념 드리프트(concept drift)는 시간이 지남에 따라 데이터 분포가 변하는 현상이며, 이는 새로운 분포에 적응할 수 있는 알고리즘의 효율성을 요구합니다.

- **Performance Highlights**: 제안된 평가 프로세스에 따라 진행된 종단적인 실험 결과, 기존 기법들에 비해 본 논문에서 제시한 접근 방식이 더 높은 효과성을 보였습니다. 다양한 드리프트 유형을 시뮬레이션하고 이를 통해 추출한 결과는 스트리밍 회귀 작업의 신뢰성을 높이는 데 기여할 것입니다. 논문에 포함된 모든 코드 및 데이터 세트는 GitHub를 통해 공개되어 완전한 투명성과 재현성을 증명합니다.



### A Study on the Importance of Features in Detecting Advanced Persistent Threats Using Machine Learning (https://arxiv.org/abs/2502.07207)
Comments:
          Accepted for publication in the 2024 International Conference on Computational Science and Computational Intelligence (CSCI'24)

- **What's New**: 이 논문은 Advanced Persistent Threats (APTs)의 탐지에 기여하는 네트워크 트래픽 특성을 분석하고, 머신러닝 프레임워크를 통해 APT 샘플 탐지 성능을 개선하기 위한 방법론을 제시합니다. 또한, 여러 특징 선택(Feature Selection) 기법을 활용하여 데이터 품질의 중요성을 강조합니다. 이것은 다양한 APT 사례들을 광범위하게 커버하여, 실제 공격 시나리오에 대한 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 APT 탐지를 위한 데이터셋을 생성하고, 다양한 APT 케이스와 관련된 데이터들을 통합하여 분석합니다. 세 가지 주요 범주, 즉 필터(Filter), 래퍼(Wrapper), 내장(Embedded) 특징 선택 기법을 통해 데이터 전처리를 수행합니다. ReliefF 및 Spectral Feature Selection (SFS)와 같은 통계적 방법을 활용하여 특징의 중요도를 평가하고, 여러 분류자와 조합하여 효과를 검증합니다.

- **Performance Highlights**: 연구 결과는 APT 탐지 성능에 영향을 미치는 특징의 중요성을 다양한 기법을 통해 평가하여, 머신러닝 기반 탐지 시스템의 효율성을 증대시키는 방향으로 가지게 됩니다. 이 접근 방식은 APT 방어 전략에서 데이터 품질의 문제를 해결하는 데 중요한 통찰력을 제공합니다. 결과적으로, 특히 공공 및 민간 부문에서의 APT 탐지 개선에 기여할 수 있습니다.



### VINP: Variational Bayesian Inference with Neural Speech Prior for Joint ASR-Effective Speech Dereverberation and Blind RIR Identification (https://arxiv.org/abs/2502.07205)
Comments:
          Submitted to IEEE/ACM Trans. on TASLP

- **What's New**: 본 논문에서는 음향 반향이 있는 음성을 처리하기 위한 새로운 방법인 variational Bayesian inference (VBI) 프레임워크를 제안합니다. 이를 통해 speech dereverberation과 blind room impulse response (RIR) 식별 작업을 동시에 수행할 수 있는 가능성을 열었습니다. 특히, DNN을 사용하여 anechoic speech의 사전 분포를 예측하는 방법을 도입하여 기존 방법에 비해 성능 향상을 도모하였습니다.

- **Technical Details**: 제안된 VINP는 convolution transfer function (CTF) 근사를 기반으로 한 확률적 신호 모델을 사용합니다. 이 모델을 통해 음원 음성과 CTF 필터, 그리고 반향이 있는 마이크 녹음 간의 관계를 설명합니다. 또한, VBI를 통해 이 모델의 숨겨진 변수와 매개변수를 분석적으로 해결함으로써 대량의 음성 데이터를 효율적으로 처리할 수 있습니다.

- **Performance Highlights**: 실험 결과, VINP는 single-channel speech dereverberation 작업에서 인간 인지 관련 대부분의 지표에서 향상된 성능을 보였으며, ASR 관련 메트릭에서도 매우 높은 성능을 나타냈습니다. 특히, blind RIR 식별 작업에서는 60 dB에서의 미학적 반향 시간(RT60)과 직접 반향 비율(DRR) 추정에서 최첨단 성능을 달성하였습니다.



### Dense Object Detection Based on De-homogenized Queries (https://arxiv.org/abs/2502.07194)
Comments:
          17 pages, 15 figures

- **What's New**: 이 논문은 밀집 객체 탐지(dense object detection)의 어려운 문제에 초점을 맞추고 있습니다. 기존의 greedy 알고리즘을 기반으로 한 탐지 방법들은 밀집 상황에서 반복적인 예측이나 놓친 탐지를 초래하는데, 이 문제를 해결하기 위해 DETR(DEtection TRansformer)을 활용했습니다. 이 방법은 후처리의 중복 제거 능력을 네트워크에 통합하는 독창적인 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 learnable differentiated encoding을 통해 쿼리(query)의 동질성을 감소시킵니다. 이를 통해 쿼리 간의 정보 소통이 가능해지고, 기존의 쿼리 간 self-attention 방식을 대체합니다. 또한 인코더의 출력에 대해 위치(location)와 신뢰(confidence) 예측을 고려한 joint loss를 사용하여 쿼리의 초기화를 보다 높은 품질로 개선합니다.

- **Performance Highlights**: CrowdHuman 데이터셋에서 평균 정밀도(average precision) 93.6%, MR-2 39.2%, JI 84.3%를 달성하며 이전 SOTA(state-of-the-art) 방법들을 초월하는 뛰어난 성능을 보였습니다. 덧붙여, 제안된 방법은 다양한 밀도의 상황에서도 강인한 성능을 발휘합니다.



### Refine Knowledge of Large Language Models via Adaptive Contrastive Learning (https://arxiv.org/abs/2502.07184)
Comments:
          Accepted to ICLR 2025

- **What's New**: 본 연구는 Large Language Models (LLMs)의 환각(hallucination) 문제를 완화하기 위해 Adaptive Contrastive Learning 전략을 제안합니다. 이 방법은 LLM이 가진 지식의 실제 숙련도에 따라 긍정적 및 부정적 샘플을 유연하게 구성하여 LLM의 지식 표현을 최적화하는 데 중점을 둡니다. 이를 통해 LLM은 기존의 올바른 지식을 강화하고, 불확실한 지식에 대한 이해를 심화하며, 잘못된 지식은 잊고 부족한 지식은 솔직히 인정할 수 있습니다.

- **Technical Details**: Adaptive Contrastive Learning 전략은 LLM의 지식 경계를 정교하게 조정하고, LLM이 아는 지식(I Know Rate, IK)과 모르고 있는 지식(I Don’t Know Rate, IDK)의 상한 및 하한을 설정합니다. 이 방법은 잘 주석된 Q&A 쌍을 사용하여 LLM의 응답 정확도를 계산하고, 정확도가 각 지식 경계에 따라 서로 다른 학습 샘플을 생성합니다. 알고리즘은 긍정적 샘플과 부정적 샘플 사이의 거리 최적화에 중점을 두어, 모델이 정확한 지식을 학습하고 부정확한 지식을 버리게 합니다.

- **Performance Highlights**: 광범위한 실험 및 데이터 분석에 따르면, 제안된 Adaptive Contrastive Learning 전략은 여러 LLM에서 높은 신뢰도(Truthful rate)를 달성하였습니다. 이 결과는 LLM의 응답의 유효성과 정직성을 개선하는 데 중요한 기여를 한다고 볼 수 있습니다. 본 연구는 LLM의 지식 표현 개선과 환각 문제 완화에 있어 새로운 관점을 제공하며, 향후 실제 응용 가능성에 대한 기대를 모으고 있습니다.



### Improved YOLOv7 model for insulator defect detection (https://arxiv.org/abs/2502.07179)
Comments:
          19 pages, 13 figures

- **What's New**: 이 논문에서는 다중 유형 절연체 결함 감지를 위한 향상된 YOLOv7 모델을 제안합니다. 기존의 연구들이 단일 결함 유형이나 특정 재료에 집중한 반면, 본 연구는 다양한 색상과 재료를 가진 절연체 결함을 동시에 처리할 수 있는 방법을 모색합니다. 이로 인해 실용적인 적용 требований을 더욱 충족할 수 있을 것으로 기대됩니다.

- **Technical Details**: 제안된 모델에서는 SPPCSPC 모듈을 RFB 모듈로 교체하여 네트워크의 feature extraction (특징 추출) 능력을 향상시킵니다. 또한, CA 메커니즘이 head 부분에 도입되어 네트워크의 feature representation (특징 표현) 능력을 강화하고, 이를 통해 감지 정확도를 높입니다. 최종적으로 WIoU loss function (손실 함수)를 사용하여 훈련 중 모델 일반화를 방해하는 저품질 샘플 문제를 해결하고, 모델의 전체 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다양한 성능 지표에서 개선된 성과를 보였습니다. 특히, mAP_0.5가 1.6% 향상되었고, mAP_0.5:0.95에서도 1.6% 증가하였습니다. 또한, precision (정확도)은 1.3%, recall (재현율)은 1% 증가하였으며, 모델의 파라미터 수는 320만 감소하여 2.5 GFLOPS의 계산 비용 절감 효과도 나타났습니다. 단일 이미지 감지 속도는 2.81 밀리초 개선되었습니다.



### Foreign-Object Detection in High-Voltage Transmission Line Based on Improved YOLOv8m (https://arxiv.org/abs/2502.07175)
Comments:
          24 pages, 16 figures

- **What's New**: 이 논문에서는 고전압 송전선의 안전한 운영을 보장하기 위한 외국물체(foreign objects) 탐지에 관한 새로운 방법을 제안합니다. 구체적으로, Yunnan 전력망(Yunnan Power Grid)에서 수집된 데이터셋을 활용하여 YOLOv8m 모델을 개선한 기술을 소개합니다. 이 개선된 모델은 Global Attention Module (GAM)을 사용하여 방해물(occlusions)에서 외국물체에 집중하게 하며, 다양한 스케일에서의 특징(feature) 추출 능력을 향상시킵니다.

- **Technical Details**: 제안된 모델은 원래 YOLOv8m의 SPPF 모듈을 SPPCSPC 모듈로 교체하여 멀티스케일(multiscale) 특성을 강화하며, Focal-EIoU 손실 함수(loss function)를 도입하여 고품질 및 저품질 샘플 간의 불균형 문제를 해결합니다. 이와 같은 개선으로 모델의 수렴 속도를 가속화 시키고 탐지 정확도를 높입니다. 특히, 새로운 측정 지표를 통해 복잡한 배경에서의 다양한 물체를 효과적으로 탐지할 수 있는 능력을 갖추게 되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 mAP_0.5 지표에서 2.7%, mAP_0.5:0.95에서 4%, 그리고 리콜(recall)에서 6%의 개선을 달성했습니다. 이는 기존 방법에 비해 유의미한 성과로, 실용적인 검사의 수행을 위한 기반 기술로 자리매김할 수 있음을 보여줍니다. 이러한 성과는 고전압 송전선의 안전성을 크게 향상시킬 것으로 기대됩니다.



### SemiHMER: Semi-supervised Handwritten Mathematical Expression Recognition using pseudo-labels (https://arxiv.org/abs/2502.07172)
Comments:
          12 pages,3 figures

- **What's New**: 이번 논문에서는 손글씨 수학 표현 인식(HMER)의 성능을 향상시키기 위한 새로운 반지도 학습(semisupervised learning) 프레임워크를 제안합니다. 두 개의 분기에서 서로의 예측을 의사 라벨(pseudo-label)로 사용하여 학습하는 이중 분기 접근 방식을 도입하여, 제한된 라벨링 데이터에서도 효율적으로 성능을 개선할 수 있게 했습니다. 또한, 변화 수준에 따른 다양한 데이터 증강(augmentation) 기법을 적용하여 두 분기의 학습 과정을 동기화하고, 최적화를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 약-강(adjacent-strong) 학습 전략과 함께 글로벌 동적 카운팅 모듈(Global Dynamic Counting Module, GDCM)을 도입합니다. GDCM은 지난 예측 결과에 기반하여 카운팅 벡터를 동적으로 업데이트하며, 이를 통해 긴 거리의 수식 인식에서 발생할 수 있는 오류를 감소시키고 반복 문자의 인식 정확성을 향상시킵니다. 이 논문에서는 시그모이드 함수와 비슷한 치료법을 통해 서로 다른 학습 경로를 통합하여 성능을 높였습니다.

- **Performance Highlights**: 제안된 Semi-supervised pseudo-supervision 방법은 기존 지도 학습 방법보다 높은 인식 정확도를 달성하였습니다. 강화된 데이터 증강 기술을 통해 훈련 데이터의 활용성을 극대화하고, 논문의 실험 결과에서 손글씨 수식 인식 분야에서 의미 있는 성과가 있음을 보여줍니다. 또한, GDCM 모듈을 통해 긴 거리의 수식 인식 과정에서 발생하는 문제를 효과적으로 해결하여, 더욱 향상된 인식 품질을 안전성으로 입증하였습니다.



### Don't Just Demo, Teach Me the Principles: A Principle-Based Multi-Agent Prompting Strategy for Text Classification (https://arxiv.org/abs/2502.07165)
Comments:
          To be published in AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration

- **What's New**: 본 논문에서는 PRINCIPLE-BASED PROMPTING이라는 새로운 멀티 에이전트 프롬프트 전략을 소개합니다. 이 방법은 여러 LLM 에이전트가 독립적으로 샘플 분석을 기반으로 후보 원칙을 생성하고, 이를 최종화하는 에이전트를 통해 통합하여 분류 작업에 적용하는 방식입니다. 이 접근법은 기존의 제로샷 프롬프팅보다 1.55%에서 19.37%의 성능 향상을 보였으며, CoT 및 stepback prompting과 같은 강력한 비교군보다 우수한 결과를 보여주었습니다.

- **Technical Details**: PRINCIPLE-BASED PROMPTING은 멀티 에이전트 협업 구조를 활용하여 각 분류 작업을 위한 원칙을 자동 생성합니다. 첫째, 여러 LLM 에이전트가 라벨이 있거나 없는 데모로부터 후보 원칙을 생성하며, 이후 중앙 에이전트가 이 후보 원칙을 최종 선택하여 하위 분류 작업에 활용합니다. 이러한 접근 방식은 짧은 입력 길이로도 높은 분류 성능을 보장하며, 특히 레이블 정보와 LLM의 협력적 구조가 중요한 역할을 합니다.

- **Performance Highlights**: 우리는 두 개의 LLM(flann-t5-xxl, flan-ul2)에 대해 세 개의 공개 데이터셋과 두 개의 비공식 데이터셋에서 광범위한 실험을 수행했습니다. 결과적으로, 본 접근법은 제로샷 ICL 성능을 크게 향상시키며, 자동 생성된 SOPs가 인간이 생성한 SOPs보다 우수한 분류 성능을 보였습니다. 또한, 우리의 멀티 에이전트 방법론은 저리소스 환경에서도 fine-tuned RoBERTa-large보다 상당한 성능 향상을 나타냈습니다.



### Does Training on Synthetic Data Make Models Less Robust? (https://arxiv.org/abs/2502.07164)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서 합성 데이터(synthetic data)를 사용한 훈련의 영향을 조사합니다. 특히, 동일하거나 유사한 LLM로 생성된 합성 데이터가 기존에 모델이 가진 "블라인드스팟(blindspot)"을 악화시킬 수 있는지에 대한 질문을 다룹니다. NLI(자연어 추론) 작업을 통해 실험을 수행하여, 합성 데이터가 성능 변화에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구에서는 먼저, T𝑇Titalic_T라는 작업 모델과 G𝐺Gitalic_G라는 생성 모델을 가정합니다. NLI 작업에 집중하여, 합성 데이터가 특정 상황에서 어떻게 모델의 성능에 영향을 미치는지를 평가했습니다. 연구는 MultiNLI 데이터셋과 HANS 테스트셋을 사용하여, 모델이 슈퍼피셜 구문적 속성에 대해서만 판단하는 경향을 조사했습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터를 사용한 미세 조정(fine-tuning)이 NLI 모델의 성능을 HANS 검사 세트에서 예상했던 것처럼 감소시키지 않음을 발견했습니다. 합성 데이터가 모델의 성능에 미치는 영향은 일관되지 않았으며 다양한 설정에서 흥미로운 성능 변화를 관찰했습니다. 이로 인해 합성 훈련 데이터와 관련된 기존 가설이 완전히 지지받지 않는 것을 발견했습니다.



### A Survey on Mamba Architecture for Vision Applications (https://arxiv.org/abs/2502.07161)
- **What's New**: Mamba 아키텍처는 비주얼 태스크에서의 Transformer의 한계를 극복하기 위해 설계되었으며, 특히 attention 메커니즘의 기하급수적인 복잡성을 해결합니다. 이 연구는 Mamba 아키텍처를 비주얼 도메인 응용 프로그램에 적용하는 것을 목표로 하며, Vision Mamba (ViM) 및 VideoMamba와 같은 최근의 발전을 다룹니다. 이를 통해 양방향 스캐닝 및 선택적 스캐닝 메커니즘을 도입하여 이미지 및 비디오 이해력을 향상시킵니다.

- **Technical Details**: Mamba 아키텍처는 상태공간모델(state-space models, SSMs)을 활용하여 선형 확장성(linear scalability)을 달성합니다. 또한 이 아키텍처는 위치 임베딩(position embeddings), 교차 스캔 모듈(cross-scan modules), 및 계층적 설계(hierarchical designs)를 통합하여 전역 및 지역 특징 추출을 최적화합니다. 이러한 혁신은 Mamba 프레임워크의 효율성을 높이고 맥락 인식을 개선하는 데 기여합니다.

- **Performance Highlights**: Mamba는 이미지 및 비디오의 이해도를 높이기 위한 spatiotemporal processing을 지원하여 최첨단 성능을 입증합니다. 이러한 기술적 혁신들은 Mamba를 컴퓨터 비전 연구 및 응용 분야의 유망한 아키텍처로 위치시킵니다. 결과적으로 Mamba 아키텍처는 이전의 모델들에 비해 더 뛰어난 처리 능력과 효율성을 보여줍니다.



### Early Risk Prediction of Pediatric Cardiac Arrest from Electronic Health Records via Multimodal Fused Transformer (https://arxiv.org/abs/2502.07158)
- **What's New**: 이번 연구에서는 소아 심정지(CA)의 조기 예측을 위한 혁신적인 프레임워크인 PedCA-FT를 소개합니다. 이 프레임워크는 전자건강기록(EHR)의 테이블 형식 데이터와 텍스트 형식 데이터를 통합하여 고차원의 위험 요소 간 상호작용을 극대화합니다. PedCA-FT는 각 모달리티(view)에 적합한 트랜스포머 모듈을 사용하여 복잡한 시계열 패턴과 컨텍스트 패턴을 캡처하며, 이는 조기 CA 감지 개선에 기여할 잠재력을 보여줍니다.

- **Technical Details**: PedCA-FT는 두 가지 상호 보완적인 모달리티인 테이블 형식과 텍스트 형식의 EHR 위험 요소를 효과적으로 처리하기 위해 설계되었습니다. 전용 테이블 트랜스포머는 고차원의 정적 및 집합적 장기 데이터를 처리하며, 사전 학습된 텍스트 EHR 트랜스포머는 원본 EHR 데이터에서 파생된 텍스트 표현을 처리합니다. 이러한 모달리티별 표현을 통합하는 융합 트랜스포머를 통해 CA 발병 위험의 최종 확률을 계산합니다.

- **Performance Highlights**: CHA-O-CICU 데이터베이스에서 평가된 PedCA-FT는 3,566명의 소아 환자를 대상으로 4%의 CA 발생률을 보였습니다. PedCA-FT는 다른 10개의 인공지능 모델과 비교했을 때 5개의 주요 성능 지표에서 성능이 우수하였고, 임상적으로 의미 있는 위험 요소들을 식별했습니다. 이러한 결과는 다중 모달 융합 기술이 조기 CA 감지와 환자 치료 개선에 기여할 수 있는 잠재력을 지니고 있음을 강조합니다.



### Explaining 3D Computed Tomography Classifiers with Counterfactuals (https://arxiv.org/abs/2502.07156)
Comments:
          Code and models: this https URL

- **What's New**: 본 연구는 Latent Shift 방법을 2D 계산에서 3D 컴퓨터 단층촬영(CT) 스캔으로 확장하여, 의료 이미징에서의 반사실(counterfactual) 설명 생성을 향상시킵니다. 저자들은 제한된 훈련 샘플과 높은 메모리 요구 사항을 해결하기 위해 슬라이스 기반 접근법을 구현하여, 해석 가능한 반사실을 생성하는 데 효과적임을 보여줍니다.

- **Technical Details**: Latent Shift 방법을 사용하여 반사실을 생성하는 과정에서, 2D 오토인코더를 통해 3D 볼륨을 슬라이스 단위로 인코딩하고 디코딩하여 처리합니다. 이는 메모리 사용량을 줄이고, 고해상도 3D 의료 이미징에서도 해석 가능한 반사실을 생성할 수 있게 합니다. 기울기(gradient)는 전체 볼륨이 아닌 슬라이스의 하위 집합에 대해서만 전파되며, 이를 통해 메모리 제약을 극복합니다.

- **Performance Highlights**: 연구는 두 가지 예측 모델에 대해 슬라이스 기반 접근법의 효율성을 입증하였으며, 높은 품질의 반사실을 생성하는 데 있어 만족스러운 성능을 보이았습니다. lung segmentation 및 임상 표현형 예측을 위한 모델이 성공적으로 구현되었고, 각 모델은 CT 스캔에서 중요한 특징을 잘 포착하여 예측을 수행하는 데 효과적임을 나타냈습니다.



### Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning (https://arxiv.org/abs/2502.07154)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 성능 향상을 위한 새로운 접근 방식을 제안합니다. 특히, pass@N이라는 테스트 시간 전략에 대한 최적의 훈련 방식이 무엇인지 탐구하며, 이 과정에서 크로스 엔트로피 손실과의 불일치를 발견했습니다. 이로 인해 모델 과신(overconfidence)이 발생하며, 이는 성능 저하를 초래함을 보여줍니다. 더 나아가, 모델의 신뢰도를 제한하고 test 성능을 개선할 수 있는 새로운 손실 함수를 제안합니다.

- **Technical Details**: 연구에서는 통상적인 감독 하에 훈련된 LLM의 CE 손실을 바탕으로 pass@N 테스트 전략에서 성능을 극대화하기 위해 훈련 방식이 어떻게 수정되어야 하는지를 다룹니다. 특히, CE 손실로 인해 생기는 과신이 pass@N 성능 저하의 원인임을 실험적으로 검증하였습니다. 이는 LLM 훈련 타임 프로토콜과 테스트 타임 검색 및 추론 전략 간의 정렬 문제를 강조합니다.

- **Performance Highlights**: 제안된 알고리즘은 MATH 및 MiniF2F 벤치마크에서 수학적 추론을 개선하는 데 효과적입니다. 이 연구에서는 수학 문제 해결과 다양한 형태의 증명 트리 검색을 통해 정리된 결과를 제시하며, 기존 CE 기준에서 일관된 성능 향상을 보여주었습니다. 궁극적으로 두 개의 전통적으로 분리된 LLM 개발 단계를 함께 설계하는 것이 중요함을 강조합니다.



### Feature Importance Depends on Properties of the Data: Towards Choosing the Correct Explanations for Your Data and Decision Trees based Models (https://arxiv.org/abs/2502.07153)
- **What's New**: 본 논문에서는 머신러닝 모델의 설명 가능성을 보장하기 위해 여러 설명 방법의 장단점을 평가하고 각각의 방법이 어떤 상황에서 더 우수한지를 규명하였으며, 이를 위해 여러 특성을 지닌 합성 데이터셋을 활용한 포괄적인 실증 평가는 새로운 접근 방식으로 주목받고 있습니다.

- **Technical Details**: 이 연구는 결정 트리 기반 모델의 예측을 설명하기 위해 지역적 설명 방법들이 제공하는 특성 중요도 추정의 품질을 평가했습니다. LIME, SHAP 등과 같은 기존 설명 방법들을 분석하면서 다양한 데이터 속성과 모델 파라미터들에 대한 설명 가능성의 불일치를 탐구했습니다.

- **Performance Highlights**: 실험 결과, 지역적 설명 방법들 간의 특성 중요도 추정치의 크기와 부호에 상당한 차이가 있음을 관찰했습니다. 연구에서는 각 설명 방법의 한계를 강조하고, 연구자들과 실무자들이 데이터셋의 특성에 맞는 설명 가능성 방법을 선택할 수 있도록 유용한 지침을 제공하였습니다.



### Few-Shot Multi-Human Neural Rendering Using Geometry Constraints (https://arxiv.org/abs/2502.07140)
- **What's New**: 이번 연구에서는 다수의 사람으로 구성된 장면의 형태(shape)와 방사선(radiance)을 복원하는 새로운 방법을 제안합니다. 기존 단일 인물 환경에서의 신경망 기반 접근법들이 인상적인 결과를 보였지만, 여러 인물을 추정하는 것은 여전히 도전적입니다. 우리의 방법은 SMPL 인체 모델을 통해 사전 계산된 메쉬를 활용하여 기하학적 제약(geometry constraints)을 적용하고, 광선 정규화(ray regularization) 및 조도 변수에 대한 강건한 최적화를 통해 문제를 해결합니다.

- **Technical Details**: 우리는 SMPL 인체 모델을 활용하여 입력 데이터로부터 인체 메쉬를 초기화한 후, 이 메쉬를 기준으로 다인체 장면의 표면을 정의합니다. 이후, 다중 뷰 이미지를 사용하여 기하학 네트워크를 최적화하며, 표면 및 볼륨 렌더링(volume rendering)과 불확실성 추정 방법을 결합합니다. 추가적으로, 다양한 광원 상태에서도 일관성을 보장하기 위한 패치 기반 정규화 손실(patch-based regularization loss)과 포화 정규화(saturation regularization)를 제안하여 렌더링 품질을 향상시킵니다.

- **Performance Highlights**: 우리는 CMU Panoptic 및 MultiHuman 데이터셋을 통해 제안한 방법을 평가했습니다. 5, 10, 15 및 20개의 훈련 뷰를 이용하여 표면 복원(surface reconstruction)과 새로운 뷰 품질(novel view quality)에서 최첨단 성능을 달성했습니다. 우리의 접근법은 다중 인체 상황에서 특히 높은 정확성과 일관성을 보여주며, 기존 신경 재구성 방법들에 비해 현저한 향상을 입증합니다.



### Unconstrained Body Recognition at Altitude and Range: Comparing Four Approaches (https://arxiv.org/abs/2502.07130)
- **What's New**: 이 연구에서는 사람의 장기 식별(long-term person identification) 방법에 대한 네 가지 접근 방식을 조사하였습니다. 기존의 단기 재식별 시스템(short-term re-identification systems)과는 달리, 우리는 영구적인 체형 특성을 학습하는 데 중점을 두고 있습니다. Vision Transformer (ViT) 및 Swin-ViT 모델을 기반으로 한 신체 식별 모델을 소개하며, 시각적 정보의 맥락을 활용하여 장기적인 정확성을 목표로 합니다.

- **Technical Details**: 이 연구의 모델들은 약 190만 개의 이미지와 5천 개의 개체가 포함된 대규모 데이터셋에서 학습되었습니다. 우리는 Linguistic and Non-linguistic Core ResNet Identity Models (LCRIM, NLCRIM)의 개선된 버전을 개발하였으며, 다양한 아키텍처와 입력 이미지 크기가 장기 신체 식별 성능에 미치는 영향을 분석하였습니다. 평가에 사용된 벤치마크 데이터셋은 MARS, MSMT17, Outdoor Gait, DeepChange 등이 포함되어 있으며, 제한 없는 데이터셋에서의 성능도 검토하였습니다.

- **Performance Highlights**: 모델 성능은 실제 환경에서의 다양한 조건을 고려하여 평가되었습니다. 특히, 장거리(최대 1000m), 고도(무인 항공기 사용) 및 의류 변화가 포함된 이미지에 대한 식별 정확도가 강조되었습니다. 이러한 비교 분석은 다양한 백본 아키텍처(backbone architectures)와 입력 이미지 크기가 장기 신체 식별 성능에 미치는 영향을 드러내어, 실세계 응용 프로그램에서 모델의 강점을 밝혀줍니다.



### Cardiverse: Harnessing LLMs for Novel Card Game Prototyping (https://arxiv.org/abs/2502.07128)
Comments:
          13 pages, 7 figures, 3 tables

- **What's New**: 이번 연구는 카드 게임의 프로토타입 제작을 자동화할 수 있는 포괄적인 프레임워크를 소개합니다. Large Language Models (LLMs)의 혁신적인 활용을 통해 새로운 게임 메커니즘을 설계하고, 일관된 게임 환경을 생성하는 데 있어 기존 데이터베이스의 한계를 극복하고자 합니다. 이는 게임 개발자들이 게임을 더욱 쉽게 제작할 수 있도록 도와줄 것입니다.

- **Technical Details**: 연구에서는 새로운 게임 디자인을 생성하기 위해 그래프 기반 인덱싱 방법을 제안합니다. 또한, LLM이 주축이 된 시스템을 통해 게임 스포츠 기록을 바탕으로 일관된 게임 코드를 생성하며, LLM이 생성한 액션-밸류 함수들의 앙상블을 활용한 게임플레이 AI 구축 방법을 설명합니다. 이러한 기술적인 통합은 전체 프로토타입 제작 과정을 혁신적으로 변화시킬 것으로 기대됩니다.

- **Performance Highlights**: 이번 연구의 프레임워크는 카드 게임 프로토타입 제작 과정을 가속화하고, 인간의 노력을 줄이며, 게임 개발자들에게 진입 장벽을 낮추는 것을 목표로 하고 있습니다. 실험 결과, 제시된 방법은 기존의 수작업보다 더 효율적이고 일관된 결과를 도출하여 카드 게임 설계 및 평가에 효율성을 더하고 있습니다.



### Online Scheduling for LLM Inference with KV Cache Constraints (https://arxiv.org/abs/2502.07115)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 추론 과정에서 키-값(Key-Value, KV) 캐시의 제약을 고려한 새로운 배치 및 스케줄링 알고리즘을 제안합니다. 모델의 지연(latency) 및 자원 활용(resource utilization)을 최적화하기 위해 이론적으로 LLM 추론을 모델링하며, 기존의 알고리즘보다 더 효율적으로 대처할 수 있는 방법을 탐구합니다. 특히, 이번 연구는 반온라인(semi-online) 및 완전온라인(fully online) 스케줄링 모델을 분석하고 있습니다.

- **Technical Details**: 제안된 알고리즘은 세 가지 주요 성과를 포함합니다: 첫째, 반온라인 프롬프트 도착 모델에서 평균 지연을 측정하는 다항 시간 알고리즘을 제공하며, 이 알고리즘은 정확한 최적성을 이룹니다. 둘째, 완전 온라인 상황에서는 효율적인 온라인 스케줄링 알고리즘을 도입하여 상수적인 후회(regret)를 달성합니다. 셋째, 완전 온라인 적대적 환경에서는 어떤 알고리즘도 상수적인 경쟁 비율(competitive ratio)을 달성할 수 없음을 증명합니다.

- **Performance Highlights**: 실험 평가에서는 Llama-70B 모델을 A100 GPU에서 사용하여, 제안한 접근 방식이 현행 기준 알고리즘보다 현저히 우수한 성과를 보였음을 보여줍니다. 특히, 낮은 지연을 달성하면서 에너지 소비를 줄이는 결과를 기록하였습니다. 이러한 결과는 LLM 배포를 보다 지속 가능하고 비용 효율적으로 만들기 위한 경로를 제공합니다.



### Generative Distribution Prediction: A Unified Approach to Multimodal Learning (https://arxiv.org/abs/2502.07090)
Comments:
          31 pages 4 figures

- **What's New**: 이 논문에서는 Generative Distribution Prediction (GDP)라는 새로운 프레임워크를 소개합니다. GDP는 다중 모달 데이터(모드) 예측을 위한 최첨단 생성 모델을 활용하며, 다양한 데이터 소스로부터 만들어질 수 있는 조건부 합성 데이터를 통해 예측 성능을 강화합니다. 이는 구조적 및 비구조적 모드에서의 예측 정확성을 유지하기 위해 필요합니다.

- **Technical Details**: GDP는 모델에 구애받지 않는 환경에서 작동하며, 고충실도(high-fidelity) 생성 모델과의 호환성을 제공합니다. 이 프레임워크는 전이 학습(transfer learning)을 지원하여 다양한 손실 함수에 적응할 수 있으며, 이는 데이터 생성 분포를 추정하고 위험 최소화를 위해 다양한 손실 함수를 조정함으로써 이루어집니다. 이러한 방식은 다중 모달 설정에서 정확한 포인트 예측을 가능하게 합니다.

- **Performance Highlights**: GDP는 4개의 감독 학습 과제에서 실험적으로 검증되었으며, 결과적으로 뛰어난 성능을 나타냈습니다. 구체적으로, 테이블 데이터 예측, 질문 응답, 이미지 캡션 생성 및 적응형 분위수 회귀를 포함한 다양한 분야에서의 활용 가능성을 보여주었습니다. GDP는 기존의 회귀 방법에 비해 전반적으로 뛰어난 성능을 발휘하며, 복잡한 문제를 해결하는 데 효과성을 입증했습니다.



### Kernels of Selfhood: GPT-4o shows humanlike patterns of cognitive consistency moderated by free choic (https://arxiv.org/abs/2502.07088)
Comments:
          Main Article: 10 pages, Supporting Information: 61 pages

- **What's New**: 이 연구는 Large Language Models(LLMs), 특히 OpenAI의 GPT-4o가 인간의 심리적 과정인 인지 일관성을 모방하는지 테스트합니다. 우리는 특정 주제에 대한 태도 변화가 환경에 따라 어떻게 달라지는지를 살펴봅니다. 이 연구는 LLM이 보여주는 인지적 행동이 인간의 자아 감각과 선택에 대한 영향을 받을 수 있음을 시사합니다.

- **Technical Details**: 연구는 두 가지 사전 등록된 연구로 구성되어 있으며, 각각 GPT-4o가 블라디미르 푸틴에 대한 긍정적 및 부정적 에세이를 작성하도록 유도하였습니다. 첫 번째 연구에서 GPT는 주제와 관련된 여러 질문을 통해 푸틴에 대한 평가를 나타내었고, 두 번째 연구에서는 어떤 에세이를 작성할지에 대한 '선택'의 환상이 태도 변화의 정도에 영향을 미치는지를 분석했습니다. 결과적으로 태도 변화의 빈도와 정도는 GPT가 어떤 에세이를 자유롭게 선택했는지에 따라 달라진 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, GPT-4o는 긍정적인 에세이를 작성한 후 푸틴에 대한 평가를 유의미하게 높였고, 부정적인 에세이를 작성한 후에는 평가를 낮추었습니다. 첫 번째 연구의 신뢰도는 높은 편이었고, 두 번째 연구에서도 더 많은 샘플을 통해 GPT의 행동을 검증했습니다. 이 연구는 GPT가 인간의 심리적 일관성과 유사한 패턴을 따라 태도 변화를 보여주었으며, 선택의 환상이 이러한 변화에 중요한 역할을 한다는 점을 강조합니다.



### IRepair: An Intent-Aware Approach to Repair Data-Driven Errors in Large Language Models (https://arxiv.org/abs/2502.07072)
Comments:
          Accepted as full research paper at FSE'2025

- **What's New**: 최근 대규모 언어 모델(LLM)의 발전과 관련된 여러 가지 문제점들이 부각되고 있습니다. 이 논문에서는 IRepair라는 새로운 동적 슬라이싱 기반의 의도 인식 LLM 수리 전략을 제안합니다. 이 방법은 모델에서 오류가 발생하기 쉬운 특정 부분을 선택적으로 수리하여, 수리 과정에서 모델의 전반적인 성능 저하를 최소화하려고 합니다.

- **Technical Details**: IRepair는 가장 민감한 레이어를 동적으로 슬라이스하여 수리 노력을 집중시키는 기법입니다. 이 방법은 오류를 야기하는 변화에 대한 반응을 더 효과적으로 개선할 수 있으며, 모델의 다른 부분에 대한 변경은 최소화합니다. 구체적으로, Kullback-Leibler(KL) 발산 제약을 적용하여 오류가 발생하기 쉬운 섹션만 수리하는 방식을 채택하고 있습니다.

- **Performance Highlights**: IRepair는 GPT-2 및 GPT-Neo 모델에서 독성 완화 작업을 통해 43.6% 더 효과적으로 오류를 수리하면서도 일반 성능 저하를 46% 줄였습니다. 또한, 모델의 상위 20% 레이어에서 오류 밀도가 나머지 80%보다 773% 더 높다는 사실을 밝혀내어 선택적 수리의 필요성을 강조하고 있습니다. 이러한 결과는 IRepair의 동적 선택 접근 방식이 모델 전체의 오류를 효과적으로 해결하는 데 필수적임을 보여줍니다.



### TRADES: Generating Realistic Market Simulations with Diffusion Models (https://arxiv.org/abs/2502.07071)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Limit Order Book (LOB) 시장 시뮬레이션을 생성하는 TRAnsformer 기반의 새로운 Denoising Diffusion Probabilistic Engine인 TRADES를 제안합니다. 기존 연구의 한계를 극복하고, 현실적이며 반응성이 뛰어난 시장 시뮬레이션을 통해 알고리즘 트레이딩 전략의 보정 및 테스트에 유용한 합성 시장 데이터를 생성합니다. 연구 결과, TRADES는 두 개의 주식에서 이전의 최첨단 방법들보다 x3.27 및 x3.47 개선된 성과를 보였습니다.

- **Technical Details**: TRADES는 고주파 시장 데이터의 시간적 및 공간적 특성을 포착하는 transformer 기반 아키텍처를 활용하여 현실적인 주문 흐름을 시간 시리즈 형식으로 생성합니다. 이 모델은 다변량 시간 시리즈 생성을 처리할 수 있어 다양한 도메인에 적응 가능합니다. 또한, 생성된 합성 데이터의 유용성을 검증하기 위해, MAE로 측정된 예측 점수를 적용해 stock price 예측 모델을 훈련하고 실제 데이터로 테스트했습니다.

- **Performance Highlights**: TRADES는 생성된 시뮬레이션의 현실성과 반응성을 평가하기 위한 실험을 수행하였으며, 이는 알고리즘 트레이딩 전략의 보정 및 시장 영향 실험의 가능성을 보여줍니다. DeepMarket이라는 오픈 소스 파이썬 프레임워크를 개발하여 시장 시뮬레이션을 위한 첫 번째 심층 학습 기반 플랫폼을 제공합니다. 이 프레임워크를 활용하여 TRADES의 구현 및 체크포인트를 공개하고, 생성된 합성 LOB 데이터셋을 제공함으로써 추가 연구와 비교에 기여하고자 했습니다.



### Contextual Thompson Sampling via Generation of Missing Data (https://arxiv.org/abs/2502.07064)
- **What's New**: 이 연구에서는 Thompson sampling contextual bandit 알고리즘을 위한 새로운 프레임워크를 소개합니다. 이 알고리즘은 오프라인에서 학습된 generative model의 품질을 기반으로 불확실성을 정량화하고 결정을 내릴 수 있는 능력이 있습니다. 기존의 잠재 변수에 대한 불확실성 개념을 재구성하여, 알고리즘은 미래의 관측 가능한 결과의 결여에서 불확실성을 발생시킵니다.

- **Technical Details**: 제안된 알고리즘은 사전에 훈련된 generative sequence model을 사용하여 각 결정 시점에서 누락된 결과를 확률적으로 보완하고, 이를 기반으로 정책을 적합한 후 최적의 행동을 선택합니다. 이 과정에서 정보 손실을 최소화하는 더 효율적인 기법을 사용하여 뉴럴 네트워크와의 호환성을 높였습니다. 결과적으로, 제시된 알고리즘은 Thompson sampling의 생성적 구현을 형성하며, 최신 연구에서의 후회 감소 경계를 증명합니다.

- **Performance Highlights**: 제안된 생성적 Thompson sampling 알고리즘은 세 가지 주요 특성을 갖춘 최신 후회 경계를 가지고 있습니다. 첫째, 누락된 결과를 보완하는 데 사용되는 generative model의 품질은 오프라인에서의 시퀀스 예측 손실을 통해서만 영향을 받습니다. 둘째, 이 경계는 원하는 '오라클' 정책을 적합하는 모든 절차에 적용될 수 있어, 자원 및 공정성 제약이 있는 의사결정 문제에 쉽게 적용될 수 있습니다. 마지막으로, 무한한 정책 클래스를 지원하며, 특정 작업에 대한 사전 정보의 이점을 정량화하는 방식으로 기존의 정보 이론적 분석을 개선하였습니다.



### Federated Continual Learning: Concepts, Challenges, and Solutions (https://arxiv.org/abs/2502.07059)
- **What's New**: Federated Continual Learning (FCL)은 데이터 샘플이 지속적으로 생성되고 여러 장치에 분산된 역동적인 환경에서 협력 모델 학습을 위한 강력한 솔루션으로 부각되고 있습니다. 이 백서는 FCL에 대한 포괄적인 리뷰를 제공하며, 이질성(heterogeneity), 모델 안정성(model stability), 통신 오버헤드(communication overhead), 개인 정보 보호(privecy preservation)와 같은 주요 도전 과제를 중점적으로 다룹니다. FCL의 다양한 형태와 그것이 모델 성능에 미치는 영향을 탐구하며, 비 정적인 환경에서의 모델 안정성을 보장하는 기술들도 검토하고 있습니다.

- **Technical Details**: FCL은 Federated Learning(FL)과 Continual Learning(CL)의 개념을 통합하여 비정형 데이터를 효과적으로 처리할 수 있는 방법을 제공합니다. FL은 데이터 및 시스템의 이질성, 통신 오버헤드, 개인 정보 보호 문제를 해결하는 데 필수적인 방법론을 제공하지만, 정적인 데이터 분포를 가정하여 동적인 환경에는 적합하지 않습니다. 반면 CL은 비정상적인 데이터에서 모델이 어떻게 적응하고 학습할 수 있는지에 초점을 맞추며, 이러한 두 가지 패러다임의 융합이 FCL의 발전에 기여하고 있습니다.

- **Performance Highlights**: FCL은 다양한 실제 응용에 적합한 강력하고 확장 가능한 시스템 개발을 지원합니다. 이 시스템은 사용자 개인 정보를 보호하며, 동적인 클라이언트 참여 및 자원 제약과 같은 복잡한 문제를 해결할 수 있는 새로운 방법론이 필요합니다. FCL은 실시간 IoT 분석, 개인화된 헬스케어 및 에지 기반 자율 시스템 등 다양한 분야에서 응용될 수 있으며, 기술적 요구의 변화를 적시에 반영하여 시스템의 안정성 및 효율성을 향상시킵니다.



### Large Language Models in Software Security: A Survey of Vulnerability Detection Techniques and Insights (https://arxiv.org/abs/2502.07049)
Comments:
          33 pages, 12 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)은 소프트웨어 취약점 탐지의 비약적인 도구로 부상하고 있습니다. 기존의 정적 및 동적 분석 방법은 비효율성, 높은 거짓 긍정률, 그리고 현대 소프트웨어 시스템의 복잡성 증가로 인해 종종 실패하였습니다. 이 논문은 LLMs를 기반으로 한 취약점 탐지의 체계적인 리뷰를 제공하며, 모델 아키텍처, 애플리케이션 방법, 타겟 언어, 미세 조정 전략, 데이터 세트 및 평가 메트릭스와 같은 핵심 측면을 살펴봅니다.

- **Technical Details**: 대규모 언어 모델은 자연어 처리(NLP)의 발전으로, 특히 Transformer 아키텍처를 통해 다양한 NLP 작업에 초점을 맞추어 훈련됩니다. 최근 LLMs는 소프트웨어 개발 분야에서 주목할 만한 능력을 보여주며, 여러 LLM 기반의 취약점 탐지 방법론과 도구들이 제안되었습니다. 이 자연어 처리 모델들은 코드를 분석하고, 패턴을 식별하며, 수리 제안을 생성할 수 있는 능력을 바탕으로, 기존의 탐지 접근 방식을 혁신할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: LLMs는 최근 몇 년 동안 발표된 논문 수의 현저한 증가로 인해 사이버 보안 커뮤니티에 긍정적인 영향을 미쳤습니다. 주요 초점은 C, Java 및 Solidity와 같은 언어에 집중되고 있으며, LLM의 구현, 프롬프트 엔지니어링 및 의미적 처리 방법과 같은 세 가지 주요 구성 요소에 대한 연구가 일관되게 강조되고 있습니다. 그러나 현재 연구는 이진 분류에 국한되는 경향이 있으며, 저수준의 레포지토리 분석에서의 어려움 등의 주요 격차가 존재합니다.



### SnipGen: A Mining Repository Framework for Evaluating LLMs for Cod (https://arxiv.org/abs/2502.07046)
Comments:
          5 pages, 3 figures, 2 tables

- **What's New**: 이번 논문에서는 SnipGen이라는 새로운 프레임워크를 소개합니다. SnipGen은 다양한 소프트웨어 엔지니어링(SE) 작업을 위해 코드 생성에 필요한 강력한 데이터베이스를 구축하는 데 중점을 둡니다. 특이한 점은 SnipGen이 코드 관련 작업을 평가하기 위해 지속 가능하고 정확한 테스트베드를 생성하여 데이터 오염 문제를 완화하려는 시도를 하고 있다는 것입니다.

- **Technical Details**: SnipGen은 GitHub에서 메서드 수준의 코드 스니펫을 추출하여 축적합니다. 이 프레임워크는 pydriller 라이브러리를 사용하여 소스코드를 수집하고, 수집한 데이터를 MySQL 데이터베이스에 저장한 후 중복을 제거하고, AST(추상 구문 트리) 표현을 통해 관련 특성을 추출하여 최종적으로 맞춤형 프롬프트를 생성합니다. SnipGen은 코드 완성, 커밋 생성 및 코드 요약과 같은 다양한 SE 작업을 위한 프롬프트 인풋을 구축하는 데 적합합니다.

- **Performance Highlights**: SnipGen은 약 227K개의 데이터 포인트를 수집하였으며, 이는 GitHub 커밋의 최신 코드 변경사항을 기반으로 합니다. 이 도구는 LLMs가 코드 생성 품질을 더 세부적으로 평가할 수 있도록 하는 프롬프트 템플릿을 제공합니다. 덕분에 개발자와 연구자들이 LLM의 성능을 더 rigorously(엄밀하게) 평가하고 해석할 수 있습니다.



### Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs (https://arxiv.org/abs/2502.07045)
Comments:
          6 pages, 0 figures, 8 tables

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)을 활용하여 웹 기반 구직 사이트 리뷰에서 내부 위협 감정(insider threat sentiment)을 분석하고 감지하는 가능성을 모색합니다. LLMs는 인위적으로 생성된 리뷰 데이터와 기존의 데이터를 비교 분석하며, 기존의 방법론에서 다루지 않았던 내부 위협 탐지를 위한 새로운 접근 방식을 제시합니다. 특히, 이 연구는 윤리적 데이터 수집 문제를 해결하기 위해 LLMs를 사용하여 합성 데이터를 생성하는 방안을 채택하였습니다.

- **Technical Details**: 연구는 Claude Sonnet 3.5와 GPT-4o와 같은 LLMs를 사용하여 구직 사이트에서 생성된 리뷰 데이터를 분석하였습니다. 본 연구에서 사용된 방법론은 관련 작업의 검색 기준, 리뷰 생성과 감정 분석에 사용된 LLM 프롬프트, 검토된 기존 데이터셋과 감정 분석 방법을 포함합니다. 이 접근 방식은 기존의 유한한 데이터와 LLMs에 의해 생성된 합성 데이터를 비교하여 내부 위협 감정을 측정합니다.

- **Performance Highlights**: LLMs의 성능은 대부분의 경우 인간 평가와 일치하며, 위협 감정의 미묘한 지표를 효과적으로 식별할 수 있음을 확인하였습니다. 그러나, 인간이 생성한 데이터에 대한 성능은 합성 데이터에 비해 낮아, 실제 데이터를 평가하는 데 있어 개선할 부분이 있음을 시사합니다. 전반적으로 LLMs의 사용은 내부 위협 감지에 유용하며, 데이터 수집의 윤리적 및 물리적 장벽을 극복함으로써 스케일 가능한 솔루션을 제공합니다.



### Automated Consistency Analysis of LLMs (https://arxiv.org/abs/2502.07036)
Comments:
          10 pages, 12 figures, 3 tables, 3 algorithms

- **What's New**: 이 논문은 LLM(대형 언어 모델) 응답의 일관성(consistency)에 대한 포괄적인 정의와 평가 프레임워크를 제안합니다. 특히 보안 분야에서 LLM의 일관성이 신뢰성과 신뢰도를 결정짓는 중요한 요소임을 강조합니다. 이를 위해 자기 검증(self-validation) 및 여러 LLM 사이의 검증(cross-validation) 두 가지 접근 방식을 제안하고, 다양한 보안 질문에 대한 실험을 통해 이 주제를 다룹니다.

- **Technical Details**: LLM의 일관성은 동일하거나 의미적으로 유사한 프롬프트에 대한 응답의 일관성을 기반으로 정의됩니다. 응답은 구문 구조(syntactic structure)에서 차이가 날 수 있지만 의미적으로 동일해야 합니다. 이 논문은 LLM의 일관성과 정확성(accuracy) 및 허위 응답(hallucination)과의 관계를 분석하고 이러한 요소들이 어떻게 LLM의 성능에 영향을 미치는지를 설명합니다.

- **Performance Highlights**: 실험 결과, ChatGPT 4o Mini, GPT3.5, Gemini, Cohere, Llama3와 같은 여러 LLM들이 보안 작업에서 사용되고 있음에도 불구하고 응답에서 종종 불일치성을 보인다는 사실을 확인했습니다. 이는 기업 수준의 사이버 보안 작업에서 신뢰성을 확보하기 어렵다는 것을 시사합니다. 따라서 LLM의 응답 일관성을 신뢰할 수 있는 수준으로 개선하지 않으면, 보안 분야에서의 활용이 제한될 것으로 보입니다.



### Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessmen (https://arxiv.org/abs/2502.07029)
Comments:
          Accepted to NAACL 2025. Codebase available at this https URL

- **What's New**: 이번 연구에서는 Allophony(음소 변이)를 모델링하기 위한 새롭고 혁신적인 접근법인 MixGoP를 제안합니다. MixGoP는 Gaussian mixture models(GMMs)를 활용하여 음소의 다중 하위 클러스터를 모델링함으로써 atypical pronunciation(비정상 발음) 평가의 필요성을 충족합니다. 이 방법은 frozen self-supervised speech model(S3M)을 통합하여 다양한 발음 변이를 효과적으로 캡처합니다.

- **Technical Details**: MixGoP는 기존의 Goodness of Pronunciation(발음의 질)를 개선하여 각 음소를 여러 개의 allophonic subclusters(음소의 하위 클러스터)로 모델링합니다. 기존 접근법은 음소를 단일 클러스터로 간주하고 atypical speech(비정상 음성)가 typical speech(정상 음성)와 동일한 분포에 있다고 가정하지만, MixGoP는 이러한 가정을 완화하여 보다 정교한 모델링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MixGoP는 총 5개의 데이터셋 중 4개에서 최첨단 성과를 달성하였으며, 이는 dysarthric(발음 장애) 및 비원어민 음성을 포함합니다. 또한 S3M 특징이 MFCCs 및 Mel spectrograms와 비교하여 allophonic variation(음소 변이)를 더 효과적으로 캡처함을 분석하여 MixGoP와 S3M의 통합의 장점을 강조합니다.



### Representational Alignment with Chemical Induced Fit for Molecular Relational Learning (https://arxiv.org/abs/2502.07027)
- **What's New**: 이번 연구에서는 화학 지식을 활용한 새로운 접근 방식인 ReAlignFit을 제안합니다. 이는 Molecular Relational Learning (MRL)의 안정성을 증가시키기 위해 한다는 점에서 주목을 받습니다. 기존의 attention 메커니즘에서 벗어나, 화학적 Induced Fit 이론을 기반으로 하여 서브구조 representation을 동적으로 정렬하는 방법을 탐색합니다.

- **Technical Details**: ReAlignFit은 기존의 MRL 방법들을 개선하기 위해 Bias Correction Function (BCF)을 설계하여 서브구조 쌍 간의 지식 기반 동적 정렬을 구현합니다. 이 방법은 화학적 구성 변화의 시뮬레이션을 통해 서브구조 representation을 정렬하고 강화합니다. 또한 Subgraph Information Bottleneck (S-GIB)을 통합하여 높은 화학적 기능 호환성을 가진 서브구조 쌍을 최적화합니다.

- **Performance Highlights**: 실험 결과, ReAlignFit은 아홉 개의 데이터셋에서 최첨단 모델들보다 뛰어난 성능을 발휘하며, 규칙 및 스캐폴드 변경이 있는 데이터 배포에서 모델의 안정성을 크게 향상시켰습니다. 특히, 높은 화학적 기능 호환성을 가진 서브구조 쌍을 효과적으로 활용하여 정확한 분자 임베딩을 생성하는 데 성공했습니다.



### Machine Learning for Everyone: Simplifying Healthcare Analytics with BigQuery ML (https://arxiv.org/abs/2502.07026)
Comments:
          Focus: Artificial Intelligence, Healthcare analytics, cloud computing, BigQuery ML

- **What's New**: 본 논문에서는 Google Cloud의 BigQuery ML이 머신 러닝(ML) 모델의 개발과 배포를 간소화하며, SQL을 사용하여 기술 장벽을 줄이는 방법을 소개합니다. 전통적인 ML 워크플로우는 전문 기술과 인프라를 요구하여 많은 의료 전문가에게 접근성을 제한해왔습니다. BigQuery ML은 이러한 한계를 극복하게 하는 혁신적인 도구로 주목받고 있습니다.

- **Technical Details**: 논문에서는 당뇨병 예측을 위한 Diabetes Health Indicators Dataset을 사용하여 세 가지 예측 모델, 즉 Logistic Regression, Boosted Tree, Deep Neural Network(DNN)에 대한 사례 연구를 진행합니다. 데이터 분석에 있어 SQL 기반의 접근법을 통해 ML 모델의 개발이 더욱 용이해지며, 각 모델의 성능이 비교됩니다.

- **Performance Highlights**: 결과적으로 Boosted Tree 모델이 가장 높은 성능을 보이며 당뇨병 예측에 매우 효과적임을 입증합니다. 이는 BigQuery ML이 의료 분석에 있어 확장 가능하며 효율적이고 접근 가능한 솔루션을 제공함을 강조합니다. 본 연구는 ML democratization의 중요성을 부각시키며 많은 의료 분야 전문가들이 사용할 수 있는 기반을 마련합니다.



### AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements (https://arxiv.org/abs/2502.07022)
Comments:
          Camera ready. ICLR 2025

- **What's New**: 이 논문에서는 호주 현대 노예법에 따른 기업의 성명서를 평가하는 데 도움이 되는 신규 데이터셋을 소개합니다. 이 데이터셋은 총 5,731개의 현대 노예 성명서로 구성되어 있으며, 항목별로 주석 처리되었습니다. 현대 언어 모델 (LLMs)을 활용하여 기업의 성명서에서 구체적인 현대 노예 대응 조치를 인식하고 불분명한 주장과 구별하는 데 중점을 두고 있습니다.

- **Technical Details**: 데이터셋은 HDF5 및 Activeloop DeepLake 포맷으로 제공될 예정입니다. HDF5는 대량 데이터 처리에 유용한 포맷이며, Activeloop DeepLake는 머신러닝 실험에 최적화된 기능을 제공합니다. 이 데이터셋은 호주 현대 노예 등록부에 게시된 성명서와 함께 관련 메타데이터를 포함하고 있으며, 모든 처리된 성명서와 '골드' 검증 세트도 포함되어 있습니다.

- **Performance Highlights**: 데이터셋은 Figshare라는 오픈 액세스 저장소에 호스팅되어 연구 커뮤니티에 무료로 제공됩니다. 다양한 지침이 제공되서 데이터셋을 효과적으로 활용할 수 있도록 돕고, 출력의 정량적 분석을 위한 기계 학습 방법론이 제안되었습니다. 또한, 초기 데이터셋 릴리스에는 모든 주석 처리된 성명서와 금일 검증 세트가 포함되며, 모델 경쟁을 위해 골드 테스트 세트의 릴리스는 2025년까지 보류될 가능성이 있습니다.



### Finding Words Associated with DIF: Predicting Differential Item Functioning using LLMs and Explainable AI (https://arxiv.org/abs/2502.07017)
Comments:
          14 pages, 2 figures, 6 tables

- **What's New**: 이 연구에서는 다양한 encoder 기반 Transformer 대규모 언어 모델(LLM)을 미세 조정(fine-tuning)하고 비교하여 항목 텍스트로부터 차별 항목 기능성(differential item functioning, DIF)을 예측하는 방법을 제시합니다. 연구팀은 설명 가능한 인공지능(explainable artificial intelligence, XAI) 기법을 적용하여 DIF와 관련된 특정 단어를 식별했습니다. 이 과정에서 3학년부터 11학년까지의 학생들을 위한 영어와 수학 종합 주(State Assessment)를 위한 42,180개의 항목을 사용했습니다.

- **Technical Details**: 모델의 예측 성능은 8개의 초점(focal) 및 참조(reference) 그룹 쌍에 대해 $R^2$ 값이 .04에서 .32까지 다양했습니다. 연구 결과, DIF와 연관된 많은 단어가 설계에 따라 시험 청사진(test blueprint) 내에 포함된 작은 하위 도메인을 반영하고 있음을 나타냅니다. 이는 종종 DIF 항목에 대한 정성적 리뷰가 혼란스럽거나 불확실한 결과를 초래하는 이유를 설명하는 요소가 될 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 항목 작성 과정에서 DIF와 관련된 단어를 선별하여 즉각적인 수정이 가능하게 하며, 전통적인 DIF 분석 결과를 재검토할 때 텍스트에서 주요 단어를 강조하는 데 도움을 줄 수 있습니다. 이 연구의 확장은 특히 고품질 항목을 구축할 자원이 부족한 평가 프로그램의 공정성을 향상시킬 수 있으며, 전통적인 DIF 분석을 위한 충분한 샘플 사이즈가 없는 더 작은 하위 집단에서도 적용될 수 있습니다.



### From Image to Video: An Empirical Study of Diffusion Representations (https://arxiv.org/abs/2502.07001)
- **What's New**: 이 논문은 비디오 생성 및 이미지 생성을 위한 동일한 모델 아키텍처를 비교하여 비디오 확산 모델(video diffusion models)이 인간의 비주얼 이해를 위한 다양한 다운스트림 작업에서 우수하다는 점을 강조합니다. 특히, 이 연구는 이미지 확산 모델(image diffusion models)과 비디오 확산 모델 간의 내부 표현을 분석하고 비교하는 첫 번째 작업으로, 비디오 모델이 모션과 공간 장면 이해 간의 상호작용을 어떻게 포착하는지에 대한 통찰을 제공합니다.

- **Technical Details**: 논문에서는 WALT라는 하이브리드 아키텍처를 가진 확산 모델을 사용하여 비디오와 이미지를 위한 사전 훈련 목표의 차이가 다운스트림 성능에 미치는 영향을 분석합니다. 기반 모델은 Latent Diffusion Model(잠재 확산 모델)이며, 이는 VQ-VAE(벡터 양자화 변이 오토인코더)에서 작동하여 계산 요구사항을 크게 줄이는 장점을 가집니다. 실험은 이미지 분류, 동작 인식, 깊이 추정 및 물체 추적을 포함한 다양한 작업에 대한 성능을 포함합니다.

- **Performance Highlights**: 비디오 확산 모델은 이미지 모델보다 다양한 작업에서 일관되게 뛰어난 성능을 보였으며, 이러한 우수성의 정도에는 주목할 만한 범위가 존재하는 것으로 나타났습니다. 특히, 훈련 예산, 모델 크기 및 생성 시각 품질 간의 관계를 분석하였으며, 이는 비디오와 이미지 모델 간의 성능 차이에 대한 정보를 제공합니다. 이 연구 결과는 동적 장면 이해에서 비디오 확산 모델의 잠재적 가치를 입증합니다.



### SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering (https://arxiv.org/abs/2502.06994)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM) 에이전트의 협업 소프트웨어 공학(CSE) 내에서의 비동기 문제를 체계적으로 정의하는 SyncMind 프레임워크를 도입했습니다. 이를 통해 24,332개의 에이전트 비동기 시나리오가 포함된 SyncBench 벤치마크를 생성하여, 실제 환경에서의 소프트웨어 개발 협업의 복잡성을 반영하였습니다.

- **Technical Details**: SyncMind는 에이전트의 비동기 회복 능력과 자원 효율성을 평가하기 위해 두 가지 주요 차원을 고려합니다. 협업 환경에서 비동기 상태는 팀원들의 업데이트를 놓쳐 발생하며, 이때 에이전트의 신념 상태가 프로젝트의 실제 상태와 차이가 생깁니다. 이러한 상태에서 에이전트가 비동기를 인지하고 회복하는 방식은 연구의 핵심입니다.

- **Performance Highlights**: SyncBench를 이용한 실험에서는 LLM 에이전트 간의 성능 차이를 발견했으며, 이로 인해 협업 시 발생하는 회복의 효과성에 대한 통찰을 제공하였습니다. 특히, 에이전트들이 자원 인식과 적응력에서 중대한 한계가 있음을 드러냈고, 협업 의지와 능력이 회복 성과에 긍정적인 상관관계를 나타냄을 확인했습니다.



### Who is Helping Whom? Analyzing Inter-dependencies to Evaluate Cooperation in Human-AI Teaming (https://arxiv.org/abs/2502.06976)
- **What's New**: 이번 연구는 Human-AI Teaming(HAT)과 Zero-shot Cooperation(ZSC)에 대한 새로운 접근 방식을 제시합니다. 기존의 multi-agent reinforcement learning(MARL)을 사용한 기법이 주로 작업 완료에 초점을 두었다면, 본 연구에서는 팀의 협력이 어떻게 발생하는지를 측정하기 위한 상호 의존성(interdependence) 개념을 도입했습니다. 이는 인간과 AI 팀 간의 실제 협력 행동을 평가하는 중요한 지표로 활용될 것입니다.

- **Technical Details**: 연구에서는 인간 모델과 결합된 최첨단 MARL 방법을 Overcooked 게임 환경에서 인간-AI 팀의 성과를 평가하는 데 사용합니다. 상징적인 STRIPS 형식으로 Markov 게임을 매핑하여, 팀원이 공동 목표를 이루기 위해 어떻게 서로 의존하는지를 추적할 수 있는 구조를 만듭니다. 특히, 두 전략의 상호의존성을 비교하여 협력적인 행동의 기초가 되는 과정을 분석합니다.

- **Performance Highlights**: 시험 결과, 훈련된 에이전트들이 인간 파트너와의 팀워크에서 낮은 수준의 상호 의존성을 보였고, 이는 상대적으로 낮은 협력 행동으로 이어졌습니다. 또한, 과제를 수행하는 성과가 팀워크 성과와 반드시 연관되어 있지 않다는 점을 강조했습니다. 따라서 에이전트와 인간 간의 상호 작용의 미스알라인(misalignment) 문제가 해결되어야 할 중요한 이슈로 남아 있으며, 향후 MARL 접근 방식의 평가 및 기술 설계에 기여할 것입니다.



### Task Offloading in Vehicular Edge Computing using Deep Reinforcement Learning: A Survey (https://arxiv.org/abs/2502.06963)
Comments:
          27 Pages, 3 Figures, 3 Tables

- **What's New**: 이 논문은 Intelligent Transportation Systems (ITS)의 효율성 향상을 위해 Reinforcement Learning (RL) 및 Deep Reinforcement Learning (DRL) 프레임워크의 적용 가능성을 탐구합니다. 기존의 계산 오프로드 전략이 차량 환경의 동적이고 이질적인 특성에 적응하는 데 어려움을 겪는다는 점에서, 새로운 접근 방식을 제시하고 있습니다. 이 연구는 최적화된 보상 구조와 협력적인 다중 에이전트 시스템을 통해 DRL의 이해와 응용을 발전시킬 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Markov Decision Process (MDP) 접근 방식을 바탕으로 다양한 드라이빙 환경에서의 계산 오프로드 최적화를 다룹니다. 논문에서는 V2V와 V2I 통신 프로토콜, Mobile Edge Computing (MEC), fog computing 등 다양한 기술적 요소가 논의됩니다. 특히 DRL 기술이 혼합 정수 비선형 문제를 해결하고 동적 환경에서의 의사 결정을 최적화하는 데 기여할 수 있는 방법을 구체적으로 제시합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 ITS의 효율성, 확장성 및 강인성을 최적화하여 미래 혁신의 초석을 마련하는 데 기여합니다. 연구의 결과는 DRL 기술을 이용한 서비스 지속성 유지, 네트워크 성능 최적화, 및 작업 할당의 개선을 통해 현장 응용 가능성을 높이는 데 중점을 두고 있습니다. 이 연구는 차량 네트워크에서의 데이터 처리 및 통신 성능 향상의 가능성을 보여줍니다.



### Neighborhood-Order Learning Graph Attention Network for Fake News Detection (https://arxiv.org/abs/2502.06927)
Comments:
          37 pages

- **What's New**: 본 논문에서는 Fake News Detection을 위한 새로운 모델, Neighborhood-Order Learning Graph Attention Network (NOL-GAT)를 제안합니다. 기존 Graph Neural Networks (GNN) 아키텍처의 한계를 극복하기 위해, 각 노드가 자신의 최적 이웃 순서를 독립적으로 학습할 수 있도록 설계하였습니다. 이 모델은 멀리 있는 이웃으로부터의 중요한 정보를 효과적이고 효율적으로 추출할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: NOL-GAT는 두 가지 주요 구성 요소로 이루어집니다. 첫째는 Hop Network로, 각 노드의 최적 이웃 순서를 결정합니다. 둘째는 Embedding Network로, 이 최적 이웃을 바탕으로 노드 임베딩을 업데이트합니다. 이 아키텍처는 선택적인 이웃 선택을 통해 오버 스무딩(over-smoothing) 문제를 완화하고, 복잡한 메시지 흐름을 줄임으로써 계산 복잡성을 낮춥니다.

- **Performance Highlights**: NOL-GAT 모델은 다양한 Fake News 데이터셋에서 우수한 성능을 보이며, 낮은 레이블 데이터 비율(10%-30%) 상황에서도 기초 모델보다 현저하게 높은 정확도와 macro-F1 점수를 기록합니다. 이러한 결과는 제안된 접근 방식이 정보 전파 및 노드 간 관계의 복잡성을 효과적으로 처리할 수 있음을 잘 보여줍니다.



### Occam's model: Selecting simpler representations for better transferability estimation (https://arxiv.org/abs/2502.06925)
- **What's New**: 이 연구에서는 사전 훈련된 모델의 전이 가능성을 추정하는 두 가지 새로운 메트릭을 제안합니다. 이러한 메트릭은 사전 훈련된 모델의 표현이 특정 목적의 클래스 분리를 얼마나 쉽게 할 수 있는지를 측정하여 전이 가능성을 평가하는 독특한 관점을 제공합니다. 기존 연구에 비해, 제안된 메트릭은 실제 문제 설정에서 뛰어난 성능을 입증하고 있습니다.

- **Technical Details**: 기존의 전이 가능성 추정 방법은 일반적으로 정보 이론, 에너지 기반 모델, 선형화 및 행렬 분석과 같은 다양한 관점에서 접근하였습니다. 본 연구에서는 프리트레인 모델의 표현에서 클래스 간 분리 및 개념 분산을 평가하여 전이 가능성을 추정합니다. 우리의 첫 번째 메트릭은 클러스터 품질 문제로, 두 번째 메트릭은 개념 분산 문제로 간주됩니다.

- **Performance Highlights**: 실험적으로 우리는 제안된 메트릭이 최신 기준 모델에 비해 최대 32%까지 Kendall's Tau를 증가시키는 것을 보여주었습니다. 이와 같은 성과는 현대 인공 신경망과 복잡한 데이터 세트를 활용한 새로운 실험 벤치마크에서 나타났으며, 이를 통해 제안된 방법의 우수성을 확인할 수 있었습니다.



### XAMBA: Enabling Efficient State Space Models on Resource-Constrained Neural Processing Units (https://arxiv.org/abs/2502.06924)
- **What's New**: 이번 논문에서는 시퀀스 데이터 작업을 위해 SSM(State-Space Models)과 기존 NPU(Neural Processing Unit)에서의 최적화를 통해 고성능의 경량 모델 XAMBA를 제안합니다. XAMBA는 3단계 방법론을 따르며, SSM을 NPU에서 활성화하고 성능을 최적화하며, 성능 향상을 위해 정확도를 희생하는 방법을 제시합니다. 핵심 병목 현상인 CumSum과 ReduceSum의 연산을 매트릭스 기반 연산으로 대체하여 실행 속도와 메모리 효율성을 크게 향상시킵니다.

- **Technical Details**: XAMBA는 SSM을 COTS(Commercial Off-The-Shelf) NPU에서 최적화할 수 있는 첫 번째 프레임워크로, SSM의 활성화와 성능 최적화를 기반으로 합니다. CumBA는 연속적인 CumSum 연산을 매트릭스 곱셈(MatMul)으로 대체하고, ReduceSum은 매트릭스-벡터 곱셈(MVM)으로 처리하며, 이로 인해 실행 지연을 줄이고 메모리 효율성을 개선합니다. ActiBA는 계산 비용이 높은 활성화 함수(Swish, Softplus)를 조각별 선형 함수(PLU)로 매핑하여 지연 시간을 줄이면서도 정확도 손실을 최소화합니다.

- **Performance Highlights**: XAMBA는 Intel Core Ultra Series 2 NPU에서 Mamba와 Mamba-2의 성능을 극대화하여 성능이 최대 2.6배 향상되었습니다. CumBA는 실행 지연을 1.8배 줄였고, ReduBA는 1.1배 줄였으며, ActiBA는 최대 2.6배 개선된 결과를 보여줍니다. 이러한 최적화는 특히 자원이 제한된 환경에서의 AI 작업에서 매우 중요하며, 실제 구현이 제공되어 사용자가 활용할 수 있습니다.



### Do Attention Heads Compete or Cooperate during Counting? (https://arxiv.org/abs/2502.06923)
Comments:
          14 pages, 15 figures

- **What's New**: 이번 논문에서는 작은 Transformers 모델이 기본적인 작업인 카운팅(counting)을 학습하는데 있어 기계적 해석 가능성(mechanistic interpretability)에 대한 심층 분석을 제공합니다. 연구팀은 Attention heads가 협력 또는 경쟁하는 방식에 대한 질문을 제기하며, 이들이 동일한 서브 작업(subtask)을 해결하는 pseudo-ensemble로 작동하는지 또는 각각 다른 서브 작업을 수행하는지를 살펴봅니다.

- **Technical Details**: 연구 결과, Attention heads가 카운팅 작업의 의미(semantics)에서 pseudo-ensemble처럼 작동하며, 문법(syntax)을 따르는 인코딩을 생성하기 위해서는 그 출력을 비균일(non-uniform)하게 집계해야 한다는 증거를 제시합니다. 이러한 분석은 작은 규모의 Transformer 모델이 복잡한 알고리즘에서 필수적인 추론(deductive) 단계를 어떻게 수행하는지를 이해하는 데 중요한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 작은 Transformer 모델이 특정 작업을 해결하는 방식에 대한 새로운 시각을 제공하며, Attention heads 간의 협력적이고 경쟁적인 상호작용의 복잡성을 드러냅니다. 또한, 이 논문에서 제시된 소스 코드(source code)는 출판 후 공개될 예정으로, 향후 연구자들이 더욱 깊이 있는 연구를 진행할 수 있는 기반이 될 것입니다.



### Synthetic Audio Helps for Cognitive State Tasks (https://arxiv.org/abs/2502.06922)
Comments:
          John Murzaku and Adil Soubki contributed equally to this work

- **What's New**: 최근 NLP(Natural Language Processing) 분야는 주로 텍스트 기반의 인지 상태(cognitive state) 과제에 초점을 맞추었지만, 오디오가 제공할 수 있는 중요한 단서를 통해 더 나은 결과를 도출할 수 있음을 제시합니다. 본 논문에서는 텍스트-음성 변환(text-to-speech, TTS) 모델이 인간의 인지 상태를 반영하는 방식을 학습한다고 가정하고 Synthetic Audio Data fine-tuning(SAD)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 텍스트와 욱-샷(zero-shot) 합성 오디오 데이터의 다중 모달(multimodal) 훈련이 인지 상태 모델링에 기여한다는 데이터를 보여줍니다.

- **Technical Details**: SAD 프레임워크는 BERT(bert-base-uncased)를 기반으로 하여 진행됩니다. OpenAI TTS API를 사용해 합성 오디오 데이터를 생성하며, Alloy 음성과 tts-1-hd 모델을 통해 오디오 품질을 최적화합니다. 이 프레임워크는 서로 다른 TTS 모델과 컴포넌트를 결합하여 통합 다중 모달 아키텍처를 형성하며, 각각의 컴포넌트는 특정 태스크에 맞게 교체 가능한 유연성을 가집니다.

- **Performance Highlights**: 다양한 실험을 진행한 결과, SAD 프레임워크는 인지 상태 태스크에서 텍스트 전용 방법보다 우수한 성능을 보였습니다. 특히, 금 오디오(gold audio)가 존재하는 경우에도 텍스트와 합성 오디오의 조합이 경쟁력 있는 결과를 달성함을 확인했습니다. 결과적으로 SAD는 인지 상태 과제의 성능을 향상시키는 유용한 도구로 자리 잡을 수 있으며, NLP, 음성 인식 및 다중 모달 모델의 발전에 따라 더욱 발전할 가능성이 큽니다.



### GraNNite: Enabling High-Performance Execution of Graph Neural Networks on Resource-Constrained Neural Processing Units (https://arxiv.org/abs/2502.06921)
- **What's New**: 이 논문에서는 그래프 신경망(GNN)을 상용(DNN) 가속기에 최적화하는 새로운 하드웨어 인식 프레임워크인 GraNNite를 소개합니다. GraNNite는 GNN 실행을 위해 NPU(신경처리장치)를 활성화하고 성능을 최적화하며 효율성을 위해 정확성을 조정하는 세 가지 단계로 구성되어 있습니다. 기존의 GNN 처리 최적화 방법들이 실시간 엣지 배포에 충분하지 않았던 점을 개선하는 것이 특징입니다.

- **Technical Details**: GraNNite는 세 가지 단계로 구성된 체계적인 접근 방식을 통해 GNN의 실행을 최적화합니다. 첫 번째 단계에서는 GraphSplit를 사용해 작업 부하를 분산시키고, StaGr로 정적 집계를 수행하며, GrAd와 NodePad로 동적 그래프를 처리합니다. 두 번째 단계에서는 EffOp를 통해 제어 중심 작업에 대해 성능을 높이고, GraSp를 통해 희소성을 활용하여 메모리 사용량을 줄입니다. 마지막 단계에서는 QuantGr를 통해 INT8 양자화를 적용하여 정확성을 희생하면서 성능 향상을 도모합니다.

- **Performance Highlights**: Intel Core Ultra AI PC 환경에서 GraNNite는 기본 NPU 매핑에 비해 2.6배에서 7.6배까지 속도를 높였으며, CPU 및 GPU 대비 에너지 효율이 최대 8.6배 향상되었습니다. GNN 모델 전반에 걸쳐 GraNNite는 CPU 대비 3.3배에서 10.8배까지, GPU 대비 2.3배에서 6.7배의 성능 향상을 보여줍니다. GraNNite는 자원 제약이 있는 엣지 장치에서 GNN을 효과적으로 처리하는 데 중대한 기여를 하고 있습니다.



### Direct Estimation of Pediatric Heart Rate Variability from BOLD-fMRI: A Machine Learning Approach Using Dynamic Connectivity (https://arxiv.org/abs/2502.06920)
Comments:
          5 pages, 5 figures, ISMSMR 2025

- **What's New**: 이 연구는 소아 fMRI 연구에서 흔히 발생하는 심장 신호의 부족 또는 품질 저하 문제를 해결하기 위한 새로운 방법을 제시합니다. 기존의 외부 기록 장치 없이 fMRI 데이터에서 직접 Heart Rate Variation (HRV) 파형을 추출할 수 있는 도구가 개발되었습니다. 이는 소아의료 분야에 큰 도움을 줄 것입니다.

- **Technical Details**: 연구진은 머신 러닝 프레임워크를 개발하여 소아 응용을 위한 HRV를 정확하게 재구성했습니다. 1-dimensional Convolutional Neural Networks (1D-CNN)와 Gated Recurrent Units (GRU)를 결합한 하이브리드 모델이 628개의 ROI에서 BOLD 신호를 분석하며, 과거와 미래의 데이터를 통합합니다. 이 모델은 HRV 정확성에서 8% 향상을 달성했으며, 이는 성능 지표에서 개선된 결과로 나타났습니다.

- **Performance Highlights**: 이 접근법은 외부 photoplethysmography 장치의 필요성을 없애고, 비용을 줄이며, 소아 fMRI 절차를 간소화하는 데 기여합니다. 또한, 성인 연구에 비해 생리적 및 발달적 변동에 더 민감한 소아 fMRI 연구의 강인성을 개선할 수 있습니다.



### Select before Act: Spatially Decoupled Action Repetition for Continuous Contro (https://arxiv.org/abs/2502.06919)
Comments:
          ICLR 2025

- **What's New**: 이 논문에서는 새로운 반복 프레임워크인 SDAR(Spatially Decoupled Action Repetition)를 제안합니다. 기존의 방법들이 모든 행동 차원을 하나로 묶어 반복하는 데 반해, SDAR는 각 차원별로 선택을 수행하여 더 유연한 행동 반복을 가능하게 합니다. 이로 인해 행동의 지속성과 다양성 간의 균형이 개선되며 더욱 향상된 성능을 제공합니다.

- **Technical Details**: SDAR은 폐쇄 루프 구조의 선택과 행동 단계를 포함하여 각 행동 차원마다 개별적으로 반복 선택을 수행합니다. 이 방법은 각 차원별로 이전 행동을 반복할지 결정하고, 선택된 차원에 대해 새로 결정을 생성하는 과정을 거칩니다. 이러한 구조 덕분에 SDAR는 기존의 반복 프레임워크보다 샘플 효율성을 높이고, 더 나은 정책 성능을 발휘합니다.

- **Performance Highlights**: SDAR는 다양한 연속 제어 작업에서 실험을 통해 그 효과성을 검증하였습니다. SDAR을 사용한 결과, 이전 방법 대비 높은 훈련 효율성과 최종 성능을 보였으며, 행동 변동성을 줄였습니다. 이러한 결과는 공간적으로 분리된 반복 디자인의 강점을 보여줍니다.



### Leveraging GPT-4o Efficiency for Detecting Rework Anomaly in Business Processes (https://arxiv.org/abs/2502.06918)
Comments:
          14 pages, 5 images, 4 tables

- **What's New**: 이 논문은 OpenAI의 LLM 중 하나인 GPT-4o-2024-08-06의 비즈니스 프로세스 이상 감지 능력을 조사했습니다. 특히 재작업 재고(anomaly) 탐지에 중점을 두고 이벤트 로그를 구조화된 포맷으로 변환하고 재작업된 활동을 식별하는 GPT-4o 기반 도구를 개발했습니다. 이를 통해 깊은 기술 지식 없이도 이상 감지를 보다 직관적이고 포괄적으로 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 제로샷(zero-shot), 원샷(one-shot), 몇 샷(few-shot) 프롬프트 기법을 사용하여 GPT-4o의 재작업 이상 탐지를 평가했습니다. 실험은 정규(normal), 균일(uniform), 지수적(exponential) 분포의 이상을 분석하도록 설계되었으며, 데이터셋은 재작업 이상이 포함된 합성 데이터로 구성되었습니다. 각 기법의 성능을 비교하며, GPT-4o의 최대 97.94% 정확도를 기록하는 결과를 보여주었습니다.

- **Performance Highlights**: 실험 결과, 균일 분포에 대한 몇 샷 기법은 30% 높은 정확도로 Isolation Forest, 46% 높은 정확도로 주성분 분석(Principal Component Analysis)을 초과하는 성과를 보였습니다. GPT-4o는 이상 탐지에서 강력한 성능을 발휘하며, 특히 재작업 재고 탐지에서 효율성을 증명했습니다. 이러한 결과는 모델이 이벤트 로그에서 재작업 이상을 신뢰할 수 있는 도구로 활용될 가능성을 나타냅니다.



### Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning (https://arxiv.org/abs/2502.06917)
Comments:
          Submitted to Neural Networks

- **What's New**: 이번 연구에서는 Federated Learning(FL)와 블록체인(Blockchain) 기술의 통합이 적대적 공격에 대한 방어 수단으로 어떻게 활용될 수 있는지를 다룹니다. 저자들은 Proof of Federated Learning(PoFL)이라는 합의 메커니즘을 제안하며, 이는 Byzantine 및 backdoor 공격에 저항하는 능력을 입증했습니다. 또한, 모든 채굴자(miner)가 공격받아도 방어할 수 있는 Krum Federated Chain(KFC)이라는 새로운 전략을 제시합니다.

- **Technical Details**: FL은 데이터의 직접적 교환 없이 모델을 학습하는 분산 머신 러닝 패러다임으로, 각 참여자는 자신의 데이터로 로컬 모델을 학습하고 그 파라미터를 집계해 글로벌 모델을 생성합니다. 블록체인 기술은 이러한 FL 프로세스에 안전하고 투명한 기록을 제공하여 데이터 조작이나 비인가 접근을 방지하는 데 기여합니다. 연구에서는 특정 공격 시나리오에서 PoFL이 얼마나 효과적인지를 평가하고, KFC가 공격에 어떻게 대응하는지를 분석합니다.

- **Performance Highlights**: 실험을 통해 EMNIST, Fashion MNIST, CIFAR-10 데이터셋에서 다양한 적대적 공격을 시뮬레이션해 제안한 방법들의 효과성을 검증했습니다. PoFL의 경우 최소 한 명의 채굴자가 공격받지 않을 때 유용성을 보여주었고, KFC는 모든 채굴자가 공격받는 경우에도 높은 방어력을 유지했습니다. 이러한 결과는 블록체인 기술이 FL의 보안성을 높이는 데 중요한 역할을 할 수 있음을 입증합니다.



### Hyper Compressed Fine-Tuning of Large Foundation Models with Quantum Inspired Adapters (https://arxiv.org/abs/2502.06916)
Comments:
          16 pages, 9 figures, 6 tables

- **What's New**: 본 논문에서는 효율적인 매개변수 조정을 위한 새로운 방법인 Quantum-Inspired Adapters를 제안합니다. 이 방법은 양자 머신러닝 문헌에서 도출된 Hamming-weight 보존 양자 회로에 영감을 받아, 모델 파라미터의 서브셋만 업데이트하여 매개변수를 효율적으로 사용합니다. 기존의 파라미터 효율적 미세 조정(PEFT) 방법들과 비교할 때, 44배 적은 매개변수로도 99.2%의 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 자기 주의(self-attention) 및 feed-forward 네트워크(FFN) 계층을 조정함으로써 피드백에서 훈련 가능한 매개변수를 효율적으로 통합하기 위한 전반적인 PEFT 방법론을 설명합니다. Quantum-Inspired Adapters는 합성 행렬을 통해 직교적 어댑터를 구성하며, Hamming-weight를 보존하는 양자 회로에서 영감을 받은 방식으로 작동합니다. 이는 매개변수 효율성과 무게 매개변수의 직교성을 동시에 보장하는 새로운 접근법입니다.

- **Performance Highlights**: 실험을 통해 GLUE와 VTAB 벤치마크 데이터셋에서 Quantum-Inspired Adapters가 기존 PEFT 방법들인 LoRA, OFT, BOFT에 비해 경쟁력 있는 성능을 발휘하며, 훈련 가능한 매개변수 수를 크게 줄일 수 있음을 입증했습니다. 특히, 기존의 직교 미세 조정 방법들과 비교할 때, 25배 적은 매개변수로도 98%의 상대 성능을 달성하였습니다. 이를 통해 자원이 제한된 환경에서도 효과적인 언어 및 비전 모델 조정이 가능함을 시사합니다.



### UniZyme: A Unified Protein Cleavage Site Predictor Enhanced with Enzyme Active-Site Knowledg (https://arxiv.org/abs/2502.06914)
Comments:
          18 pages,8 figures

- **What's New**: 본 논문에서는 다양한 효소에 걸쳐 일반화가 가능한 통합 단백질 절단 예측기 {
method}를 제안합니다. 기존 모델들은 특정 효소에 제한되어 있으며, 이러한 접근 방식은 새로운 효소에 일반화되지 않는다는 문제점이 있었습니다. {
method}는 생화학적으로 정보를 제공하는 모델 아키텍처와 프로테올리틱 효소의 활성 사이트 지식을 활용하여 성능을 향상시킵니다.

- **Technical Details**: 이 모델은 근처 서열(local sequence)과 구조, 그리고 전역 구조(global structure)를 포함하여 주어진 단백질의 절단 위치를 예측합니다. 전처리 단계에서 수집한 데이터는 효소의 서열과 활성 부위, 기능적 영역을 포함하며, 이는 모델 훈련에 중요한 요소로 작용합니다. 모델 구성은 효소 특성 추출과 단백질 서브스트레이트의 특성을 통합하는 두 부분으로 나누어지며, Protein Language Models (PLM)과 Graph Neural Networks (GNN)를 사용합니다.

- **Performance Highlights**: {
method}는 다양한 프로테오리틱 효소에서 높은 정확도로 절단 위치를 예측하며, 보지 못한 효소를 포함한 실험 결과가 이를 뒷받침합니다. 제안된 모델의 성능 평가는 Precision, Recall, F1 Score와 같은 지표로 수행되었습니다. 초기 실험에서는 GNN 없이 ESM만 사용할 경우 더 나은 성능을 보였고, 이로 인해 향후 GNN의 기여를 향상시키기 위한 최적화 작업이 필요함을 보였습니다.



### A Simple yet Effective DDG Predictor is An Unsupervised Antibody Optimizer and Explainer (https://arxiv.org/abs/2502.06913)
- **What's New**: 이번 연구에서는 단백질 최적화를 위한 간단하면서도 효과적인 ΔΔG 예측기인 Light-DDG를 제안합니다. 이 모델은 기존의 계산이 무겁고 시간이 많이 소요되는 ΔΔG 예측기에서 축적된 지식을 활용하여 경량 Transformer 구조를 채택하여 성능을 개선하였습니다. 또한, 수백 만 개의 변이 데이터를 포함하는 대규모 데이터셋을 전처리하여 Light-DDG의 사전 학습에 활용하였습니다.

- **Technical Details**: Light-DDG는 구조 인식을 고려한 Transformer를 기반으로 하며, 지식 증류를 통해 예측 성능을 향상시킵니다. 이 연구에서는 변이 선택의 기초가 되는 각 잔여체의 마지널 이득을 학습하기 위한 Mutation Explainer와 mutation preference-guided antibody optimization을 도입하였습니다. 이를 통해 효율적으로 접근 가능한 진화 영역을 탐색하고, 다양한 항체 후보를 빠르게 평가하여 원하는 변이를 식별할 수 있습니다.

- **Performance Highlights**: 실험 결과, Light-DDG는 Prompt-DDG에 비해 89.7배의 추론 속도 향상과 15.45%의 성능 향상을 달성하였습니다. 이 연구에서 제안한 접근 방식은 SARS-CoV-2 사례 연구를 통해 항체 최적화 및 변이 선호도 설명의 이점을 보여줍니다. 일반화 가능성, 노이즈 강인성 등 여러 측면에서 Light-DDG의 우수성을 평가하였으며, 이러한 총체적인 프레임워크만으로도 효과적인 항체 최적화를 진행할 수 있음을 입증하였습니다.



### Foundation Models for Anomaly Detection: Vision and Challenges (https://arxiv.org/abs/2502.06911)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 기초 모델(Foundation Models, FMs)을 활용한 이상 탐지(anomaly detection)의 발전을 포괄적으로 리뷰합니다. 저자들은 FMs를 인코더(encoder), 탐지기(detector), 해석기(interpreter)라는 세 가지 역할에 기반하여 분류하는 새로운 세분화를 제안합니다. 이 연구는 FMs를 활용한 기존의 탐지 기법을 체계적으로 분석하고, 이 분야에서의 주요 도전 과제를 논의합니다.

- **Technical Details**: FMs는 대규모 데이터셋을 사용하여 사전 훈련(pre-training)된 머신러닝 모델로, 자가 감독 학습(self-supervised learning)에 의해 다양하고 복잡한 데이터를 처리하는 데 적합합니다. 특히, 이 모델들은 텍스트, 이미지 및 시계열 데이터와 같은 여러 데이터 양식(multi-modal data)을 통합하여 복잡한 환경에서 이상을 탐지합니다. 저자들은 FM을 사용하는 다양한 접근 방식을 제시하며, 이를 인코더로서의 역할과 그 과정에서 생성된 임베딩(embedding)을 통한 탐지로 구분합니다.

- **Performance Highlights**: 이상 탐지 분야에서 FMs의 응용 사례는 증가하고 있으며, 이러한 모델은 고급 데이터 설명 생성과 시각적 설명 제공에서 뛰어난 성능을 보이고 있습니다. FMs를 활용한 방법들은 기존의 이상 탐지 기법에 비해 효율적이고 유연한 접근을 가능하게 하며, 특히 고위험 영역(healthcare, finance, cybersecurity)에서의 이상 탐지 결과의 설명 가능성을 높입니다. 이 연구는 FMs가 다양한 분야의 이상 탐지를 향상시키는 데 어떻게 기여할 수 있는지를 위한 명확한 프레임워크를 제공합니다.



### TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting (https://arxiv.org/abs/2502.06910)
- **What's New**: 이번 연구에서는 여러 주파수 조합이 있는 복잡한 시계열 예측 문제를 해결하기 위해 Kolmogorov-Arnold Network (KAN)를 기반으로 한 Frequency Decomposition Learning 아키텍처(TimeKAN)를 제안합니다. TimeKAN은 Cascaded Frequency Decomposition (CFD) 블록, Multi-order KAN Representation Learning (M-KAN) 블록 및 Frequency Mixing 블록으로 구성되어 서로 다른 주파수 대역의 패턴을 효율적으로 학습하고 재조합합니다. 이를 통해 TimeKAN은 다양한 현실의 시계열 데이터에서 최첨단 성능을 달성하며 경량 아키텍처로서 복잡한 모델에 비해 훨씬 적은 계산 자원으로도 뛰어난 예측 성능을 보여줍니다.

- **Technical Details**: TimeKAN 구조에서 CFD 블록은 낮은 주파수에서 높은 주파수 순으로 데이터를 처리하여 주파수 대역별 시퀀스 표현을 추출합니다. M-KAN 블록은 KAN의 유연성을 활용하여 각 주파수 대역 내에서 특정 시계열 패턴을 학습하고 표현합니다. 최종적으로 Frequency Mixing 블록을 통해 주파수 대역이 원래 포맷으로 재결합되고, 이 과정은 반복 가능하여 다양한 주파수에서 시간적 패턴을 더 정확하게 모델링할 수 촉진합니다.

- **Performance Highlights**: TimeKAN은 여러 장기 시계열 예측 작업에서 최첨단 성능을 보이며 파라미터 수가 기존 최첨단 TSF 모델보다 현저히 적습니다. 연구 결과, TimeKAN은 복잡한 정보 결합으로 인한 시계열 예측의 도전 과제를 성공적으로 해결하고 감소된 계산 자원으로도 더 정확한 예측 성능을 확보합니다. 이러한 특징들은 TimeKAN을 실제 산업에서의 다양한 시계열 예측 문제를 해결하는 데 강력한 도구로 자리매김하게 합니다.



### Satisfaction-Aware Incentive Scheme for Federated Learning in Industrial Metaverse: DRL-Based Stackbelberg Game Approach (https://arxiv.org/abs/2502.06909)
- **What's New**: 이번 연구는 산업 메타버스(Industrial Metaverse)를 위한 새로운 메타 컴퓨팅 프레임워크를 설계하며, 여기에는 연합 학습(Federated Learning) 인센티브 제도를 포함하고 있습니다. 특히 데이터의 크기, 정보의 최신성(Age of Information, AoI), 그리고 훈련 지연(latency)을 고려한 만족도 함수(satisfaction function)를 설계해 모델의 품질과 훈련 지연 간의 균형을 맞추는 것이 주된 목표입니다. 또한, 이 연구는 기존 방법에 비해 학습 효율성을 높이기 위해 DRL(Deep Reinforcement Learning)을 활용하여 확률적 균형을 도출합니다.

- **Technical Details**: 제안된 연구는 연합 학습의 유틸리티 함수를 두 단계의 스택켈버그 게임(Stackelberg game)으로 모델링하고, DRL 기법을 사용하여 게임의 균형을 학습합니다. 이 과정에서는 참가 노드가 자신의 개인 정보를 공유하지 않고도 최적의 정책을 학습할 수 있도록 하여 개인 정보 보호를 강화합니다. 연구는 각 IIoT 노드가 프라이버시를 유지하면서 고품질 메타버스를 구성하기 위해 자신만의 데이터 세트를 사용하여 공유 AI 모델을 훈련할 수 있는 방법론을 제시하고 있습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 인센티브 제도는 기존 방안들보다 최소 23.7% 이상의 유틸리티를 개선하면서 모델 정확도를 유지하는 것으로 나타났습니다. 이는 산업 메타버스 환경에서 자원의 동적 할당 및 최적화를 통해 이루어진 결과로, 만족도 함수와 인센티브 설계를 통해 노드들의 참여를 유도한 것으로 분석됩니다. 이러한 접근 방식은 더욱 효율적인 학습 환경을 조성하여 산업 메타버스의 실제 적용 가능성을 높이고 있습니다.



### Can ChatGPT Diagnose Alzheimer's Disease? (https://arxiv.org/abs/2502.06907)
Comments:
          14 pages, 5 figures, 5 tables

- **What's New**: 이번 연구에서는 ChatGPT를 활용하여 알츠하이머 질병(AD)을 진단할 수 있는 가능성을 탐구하고 있습니다. 연구진은 9300개의 전자 건강 기록(EHR)과 자가 인지 검사 결과 및 MRI(자기 공명 영상) 데이터를 분석해 ChatGPT가 AD를 정확하고 효과적으로 진단할 수 있을지 평가했습니다. 또한, 자원 부족 지역에서 전문의가 부족한 문제를 해결하는 데 기여할 수 있을 것으로 기대하고 있습니다.

- **Technical Details**: 연구에서는 zero-shot 및 multi-shot 방법으로 ChatGPT의 진단 능력을 평가하기 위한 접근 방식을 사용했습니다. MRI 스캔과 인지 테스트 결과를 통해 수집된 데이터를 기반으로, ChatGPT는 환자의 상태를 NC(정상), MCI(경도 인지 장애), AD(알츠하이머 질병)로 분류하는 예측 결과를 제공합니다. 이 방법론은 모델 내부 상태에 접근하지 않고 진단 결과를 도출할 수 있는 블랙박스 접근 방식을 사용하여 비전문가의 활용 가능성을 높입니다.

- **Performance Highlights**: 첫 번째 결과로, ChatGPT는 데이터의 유무에 관계없이 AD 진단에서 높은 정확도를 보였습니다. 특히, MRI 데이터와 인지 검사 점수를 함께 활용할 경우 진단의 정확성이 더욱 향상된 결과를 얻었습니다. 연구는 ChatGPT가 AD 진단을 위한 유망한 도구로, 조기 발견 및 단순화를 통해 환자들에게 적시에 개입할 수 있는 가능성을 제시한다고 강조합니다.



### Learning-based estimation of cattle weight gain and its influencing factors (https://arxiv.org/abs/2502.06906)
- **What's New**: 본 연구는 머신러닝(ML) 및 딥러닝(DL) 기법을 활용하여 소의 생체 중량 증가(CWG)를 효율적으로 추정하는 새로운 방법론을 제시합니다. 기존의 수작업에 의존하는 방식과는 달리 이 시스템은 원격 모니터링을 통해 지속적인 데이터 수집이 가능하여, 스트레스와 노동력 감소에 기여합니다. 연구는 2004년부터 2024년 사이의 자료를 바탕으로 진보된 ML 기술을 활용한 연구를 종합적으로 분석합니다.

- **Technical Details**: CWG 추정에 있어 본 연구는 환경 조건, 유전적 소인, 사료 가용성, 이동 패턴 및 행동 등을 고려하여 개별 소의 생체 중량 증가율과 변동성을 평가합니다. ML 및 DL 알고리즘의 효율성을 탐구하면서도 일관성 부족 문제와 다양한 데이터 도전 과제를 지적하며, 각 연구에 따라 다른 피처 세트를 기반으로 한 중량 증가 추정 문제를 다룹니다. 이 연구는 CWG 추정에 사용되는 현재의 도구, 방법 및 특성을 검토하고, 그 강점과 약점을 분석합니다.

- **Performance Highlights**: 연구 결과, 진보된 ML 접근 방식이 CWG 추정에 중요한 영향을 미치며, 이로 인해 미래의 연구 방향성을 제시하고 있습니다. 강력한 ML 기법을 적용함으로써 CWG 예측의 정확성을 높이고, 향후 연구에서 고려해야 할 잠재적 연구 갭도 확인하였습니다. 이 연구는 향후 이 분야의 추가적인 연구를 위한 유용한 참고 자료로 활용될 수 있습니다.



### Lightweight Dataset Pruning without Full Training via Example Difficulty and Prediction Uncertainty (https://arxiv.org/abs/2502.06905)
- **What's New**: 이 논문은 딥 러닝 모델의 데이터셋 프루닝(dataset pruning) 과정을 개선하는 새로운 방법인 Difficulty and Uncertainty-Aware Lightweight (DUAL) 점수를 소개합니다. DUAL 점수는 훈련 초기 단계에서 표본의 중요성을 예측 불확실성(prediction uncertainty)과 예제 난이도(example difficulty)를 고려하여 측정합니다. 또한, 극단적인 프루닝 시 일어나는 정확도 급락을 방지하기 위해 베타 분포(Beta distribution)를 이용한 비율 적응 샘플링(ratio-adaptive sampling) 기법도 제안합니다.

- **Technical Details**: 제안된 DUAL 점수는 L2 노름(norm)이나 엔트로피(entropy)와 같은 전통적 기법을 사용하지 않고, 훈련 다이내믹스(training dynamics)를 활용하여 표본의 중요도를 동적으로 측정합니다. 이를 통해 강한 예제나 어려운 예제를 우선순위로 두지 않고, 예측의 변동성을 기반으로 불확실한 샘플을 우선 고려합니다. 또한, Beta 샘플링을 통해 선택된 샘플의 비율을 조정하여 데이터 분포를 보다 잘 대표할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 데이터셋에서 이전의 최첨단 방법들보다 우수한 성능을 보였습니다. ImageNet-1k 데이터셋의 경우, 우리의 방법은 90% 프루닝 비율에서 66% 시간 비용 절감과 60%의 테스트 정확도를 달성했습니다. CIFAR 데이터셋에서는 시간 비용이 단 15%로 줄어들면서도 최첨단 성능을 유지하는 등 위험에 대한 저항력을 보여주었습니다.



### Emergence of Episodic Memory in Transformers: Characterizing Changes in Temporal Structure of Attention Scores During Training (https://arxiv.org/abs/2502.06902)
- **What's New**: 이번 연구는 트랜스포머 모델의 주의 헤드에서는 인간의 일화 기억과 유사한 시간적 편향(in-context temporal biases)이 나타남을 밝혔습니다. 저자들은 GPT-2 모델의 다양한 크기를 이용해 주의 점수(attention scores)와 출력을 분석하였으며, 이는 기계가 정보를 시간적으로 어떻게 조직하는지를 보여주는 중요한 통찰을 제공합니다. 특히, 유도 헤드(induction heads)의 제거가 이러한 시간적 효과를 없앤다는 점이 강조되었습니다.

- **Technical Details**: 저자는 GPT-2 small 및 medium 아키텍처를 기반으로 한 두 가지 모델을 사용하여, Wikitext-103 및 FineWeb 데이터셋을 통한 학습 실험을 수행했습니다. 이들은 Lag-Conditional Recall Probability (Lag-CRP) 분석을 활용하여 토큰의 시간적 관계가 주의 점수와 토큰 예측에 미치는 영향을 정량적으로 측정했습니다. 또한, 훈련 가능한 위치 인코딩(trainable positional encoding)과 모델 크기(size), 훈련 상호작용 수가 시간적 효과에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 트랜스포머는 시간적 인접성과 순위 효과를 드러내며, 이는 인간의 기억에서도 관찰됩니다. 저자들은 메모리 조직과 관련하여 LLMs가 인간과 유사한 방식으로 정보에 대해 민감할 수 있음을 시사합니다. 이러한 발견은 인-컨텍스트 학습(in-context learning)의 이해를 높이며, 트랜스포머 모델의 학습 메커니즘을 심화하는 데 도움을 줄 것입니다.



### Enabling Autoregressive Models to Fill In Masked Tokens (https://arxiv.org/abs/2502.06901)
- **What's New**: 본 논문은 MARIA (Masked and Autoregressive Infilling Architecture)를 소개하여 기존의 언어 모델들이 가진 한계를 극복합니다. MARIA는 autoregressive (AR) 모델과 masked language modeling (MLM) 모델의 장점을 통합하여 텍스트 인필링(masked infilling) 작업에서 최신 성능을 달성합니다. 이 구조는 두 모델의 은닉 상태를 결합하여 AR 모델이 효과적인 인필링을 수행할 수 있도록 합니다.

- **Technical Details**: MARIA는 사전 훈련된 MLM과 AR 모델을 결합하는 선형 디코더를 훈련하여 작동합니다. 이 모델은 AR의 신속한 추론 속도와 KV 캐싱(KV caching)의 이점을 유지하면서도, 인필링 작업에서 필요한 정보를 활용합니다. 이를 통해 MARIA는 기존의 텍스트 인필링 방식보다 효율적인 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 결과, MARIA는 기존의 방법들, 특히 이산 확산 모델(discrete diffusion models)에 비해 인필링 작업에서 월등한 성능을 보였습니다. 다양한 기준 데이터셋에서 MARIA의 성능을 입증하며, AR과 MLM 간의 격차를 해소하고 새로운 텍스트 인필링 언어 모델의 확장성을 제공합니다.



### A Sociotechnical Approach for Knowledge Management (KM) (https://arxiv.org/abs/2502.06899)
Comments:
          in French language. The author would like to thank Mrs. Christine Deville for her help with the grammatical correction of the text and especially Mr. Germain Lacoste (director of ENI of Tarbes, France) for his friendship, and finally, I thank something as alive as the always happy song of a hummingbird among flowers. arXiv admin note: substantial text overlap with arXiv:2502.01656

- **What's New**: 이 논문은 KM(지식 관리)의 사회 기술적 프레임워크를 제시합니다. 연구자는 KM을 상업적 관심사에서 분리하고, 다양한 KM 기술들을 구분하며, KM의 사회적 및 기술적 요소에 관련된 패러다임을 질문할 수 있는 가능성을 강조합니다. 이러한 점을 중심으로 KM의 일반적인 메커니즘을 식별하는 것이 이 논문의 핵심입니다.

- **Technical Details**: 논문은 KM의 사회적 측면을 조직적 접근(organizational approach), 관리적 접근(managerial approach), 생물학적 접근(biological approach)을 통해 설명합니다. 반면, 기술적 측면은 지식 및 기술 공학 접근(knowledge and skills engineering approach)을 통해 설명됩니다. 이들 접근 방식은 KM의 여러 측면을 체계적으로 비교할 수 있는 표를 제공하는 데 기여합니다.

- **Performance Highlights**: 연구 결과는 KM의 다양한 접근 방식들이 어떻게 서로 연결되는지를 보여줍니다. 또한, 이를 통해 KM에서의 사회적 및 기술적 요소 간의 관계를 명확히 하고, 향후 연구 및 실천에서의 방향성을 제시하는 데 도움을 줍니다. 이 논문은 KM의 다양한 시각을 이해하고 그 차이를 명확히 하기 위한 기초 자료로 활용될 수 있습니다.



### Large Language Models for In-File Vulnerability Localization Can Be "Lost in the End" (https://arxiv.org/abs/2502.06898)
Comments:
          Accepted for publication at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025. Replication Package: this https URL

- **What's New**: 최근 인공지능(AI) 발전으로 대형 입력 처리 능력이 향상되어, 소프트웨어 개발자들은 전체 파일에서의 취약성 탐지에 더욱 의존하고 있습니다. 따라서 연구자들은 대화형 대형 언어 모델(LLM)이 이러한 대규모 입력을 효과적으로 분석할 수 있는지 신속히 조사할 필요가 있습니다. 이 논문의 목표는 여러 최첨단 LLM의 취약성 탐지 효과를 평가하고, 입력 크기와 취약성 위치에 따른 성능 변화를 관찰하는 것입니다.

- **Technical Details**: 연구에서는 Open Source LLM, Mixtral 및 Llama 모델과 상용 모델인 GPT-3.5와 GPT-4를 포함시켰습니다. 주요 조사 사항으로는 XSS, SQL Injection, Path Traversal의 3가지 공통 취약성 유형에 대한 탐지 정확도 분석이 포함되어 있습니다. 연구 결과, 파일 크기와 취약성 위치에 따라 LLM의 탐지 효과가 현저히 변동하며, 특히 큰 파일의 끝부분에 위치한 취약성 탐지에서 성능 저하가 있음을 발견했습니다.

- **Performance Highlights**: 연구의 주요 발견은 대규모 입력에서 LLM이 'lost-in-the-end' 효과로 인해 취약성을 놓치는 경향이강하다는 점입니다. 이를 통해 논문에서는 LLM 기반 취약성 탐지를 최적화하기 위한 입력 크기 조정 방법을 제시했습니다. 최적화된 입력 크기를 적용할 경우 평균 37% 이상의 리콜 증가를 보이며, 이는 LLM의 성능 향상에 기여할 것입니다.



### PyPotteryInk: One-Step Diffusion Model for Sketch to Publication-ready Archaeological Drawings (https://arxiv.org/abs/2502.06897)
- **What's New**: 이번 논문에서는 PyPotteryInk라는 오픈소스 자동화 파이프라인을 소개합니다. 이 시스템은 고고학 도자기 스케치를 표준화된 출판 준비 도면으로 변환하여, 전통적인 수작업 방식의 번거로움을 크게 줄여줍니다. 모형은 신속한 단일 패스 처리를 통해 중요한 형태적 세부 사항들을 보호하며, 학문적 문서화 기준을 준수합니다.

- **Technical Details**: PyPotteryInk는 수정된 img2img-turbo 아키텍처를 기반으로 하며, 효율적인 패치 기반 접근 방식을 사용하여 입력 도면 크기와 관계없이 고해상도 출력을 생성합니다. 이 모델은 입력 도면의 크기와 관계없이 세부 사항을 보존하며, 기초 데이터가 적더라도 다양한 고고학적 문맥에 적응할 수 있도록 미세 조정이 가능합니다. 딥 러닝(Deep Learning)과 전통적인 이미지 처리 기술을 접목하여 도자기 드로잉의 디지털 잉크 처리 과정을 자동화하고 있습니다.

- **Performance Highlights**: 이 연구에서는 이탈리아의 초기 역사 도자기 드로잉 데이터셋을 사용하여 다양한 미세 세부 사항을 잘 캡처하는 성능을 보여주었습니다. 전문가 평가 결과, 생성된 도면은 출판 기준을 충족하며, 처리 시간을 몇 시간에서 몇 초로 단축시킬 수 있음을 입증했습니다. 이 모델은 고고학 연구 커뮤니티 내에서 채택을 용이하게 하기 위해 미리 훈련된 모델과 Python 라이브러리를 제공합니다.



### AI-Driven HSI: Multimodality, Fusion, Challenges, and the Deep Learning Revolution (https://arxiv.org/abs/2502.06894)
Comments:
          39 Pages, 22 figures, 20 tables

- **What's New**: 이번 연구는 하이퍼스펙트럼 이미징(HSI)의 최신 동향과 깊은 학습(Deep Learning) 모델의 역할을 심층적으로 검토하였습니다. HSI 기술은 기상 모니터링, 식품 품질 제어, 가짜 탐지 등 다양한 분야에서 필수적이며, 최근 AI와의 융합이 두드러진 특징입니다. 특히, 대형 언어 모델(LLMs)과의 통합을 통해 시각적 인식 및 정보 제공을 통한 새로운 가능성을 제시하고 있습니다.

- **Technical Details**: 하이퍼스펙트럼 이미징은 전통적인 시스템으로는 탐지할 수 없는 공간적 및 스펙트럼적 특성을 분석할 수 있습니다. HSI는 높이, 너비, 파장을 포함하는 3차원 하이퍼큐브(hypercube)로 저장되어 깊이 있는 물질 분석이 가능합니다. 최근의 연구에서는 CNNs (Convolutional Neural Networks), GANs (Generative Adversarial Networks)와 같은 다양한 심층 학습 모델들이 HSI 과제를 해결하는 데 사용되고 있음을 보여주고 있습니다.

- **Performance Highlights**: AI와 HSI의 융합을 통한 성능 향상이 두드러지며, 특히 객체 탐지 및 분류의 정확도가 크게 향상되었습니다. 산업 내 HSI의 연평균 성장률(CAGR)은 계속 증가하고 있으며, 이는 의료, 환경 모니터링, 국방 등 여러 분야에서 HSI의 중요성이 강조됨을 나타냅니다. 결국, 본 연구는 기술적 및 비기술적 독자 모두에게 유익한 정보를 제공하며, 향후 HSI 및 딥러닝 모델의 발전 방향에 대한 통찰력을 제시합니다.



### Certifying Language Model Robustness with Fuzzed Randomized Smoothing: An Efficient Defense Against Backdoor Attacks (https://arxiv.org/abs/2502.06892)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 백도어 공격에 대한 새로운 방어 접근법인 Fuzzed Randomized Smoothing (FRS)을 소개합니다. 특히 FRS는 이러한 공격에 대한 효율적인 인증을 가능하게 하여 여러 다운스트림 태스크에 미치는 영향을 최소화합니다. 기존의 방어 전략들이 가진 한계를 극복하고자, FRS는 모델 매개변수 스무딩(parameter smoothing)과 Monte Carlo tree search를 통합하여 보다 강력한 방어 체계를 구축합니다.

- **Technical Details**: FRS 접근법은 소프트웨어의 강인성 인증 기법(robustness certification techniques)과 이중 단계(biphased) 모델 매개변수 스무딩을 조합하여 텍스트의 취약한 구간을 찾아내기 위해 공정한 랜덤화(fuzzing)를 사용합니다. Damerau-Levenshtein 공간을 활용하여 타겟이 되는 텍스트 랜덤화를 추진하며, 포이즌된 훈련 데이터에 대한 접근이 없어도 효과적으로 모델 스무딩을 수행할 수 있습니다. 이러한 기법들을 통해 FRS는 기존의 방어 방법들에 비해 더 넓은 인증된 강인성 반경(certified robustness radius)을 달성하는 것으로 나타났습니다.

- **Performance Highlights**: 다양한 데이터셋, 모델 구성 및 공격 전략을 통해 진행된 실험 결과, FRS는 방어 효율성, 정확도 및 강인성 측면에서 뛰어난 성능을 입증하였습니다. FRS는 기존 방어 방법들과 비교하여 더욱 향상된 방어 효율을 제공하여 언어 모델의 신뢰성을 높이는 데 기여합니다. 이러한 접근방식은 고신뢰성 응용 프로그램에서 중요한 의미를 가지며, 향후 다른 태스크에서도 활용될 가능성을 보여줍니다.



### LLMs for Drug-Drug Interaction Prediction: A Comprehensive Comparison (https://arxiv.org/abs/2502.06890)
- **What's New**: 이 연구는 약물-약물 상호작용(DDI)의 예측을 위한 대규모 언어 모델(LLMs)의 잠재력을 탐구합니다. 특히, 최근 DrugBank 데이터세트를 사용하여 분자 구조(SMILES), 표적 생물체 및 유전자 상호작용 데이터를 원시 텍스트 입력으로 처리하는 독특한 접근 방식을 취합니다. 18개의 다양한 LLM을 평가하였으며, 이를 통해 약물 조합의 신뢰할 수 있는 예측 방법을 제시합니다.

- **Technical Details**: 우리는 1.5B에서 72B 매개변수를 가진 오픈소스와 독점 모델(GPT-4, Claude, Gemini 포함) 총 18개 모델을 평가했습니다. DDI 예측의 제로샷(Zero-shot) 능력을 먼저 평가한 후, 선택된 모델(GPT-4, Phi-3.5 2.7B, Qwen-2.5 3B 등)을 미세 조정하여 성능을 최적화했습니다. 평가 프레임워크는 13개의 외부 DDI 데이터세트에 대한 검증을 포함하며, 전통적인 방법인 l2-정규화 로지스틱 회귀와 비교했습니다.

- **Performance Highlights**: 미세 조정된 LLM은 뛰어난 성능을 보여주었으며, Phi-3.5 2.7B는 DDI 예측에서 민감도(0.978)와 정확도(0.919)를 달성했습니다. 이는 제로샷 예측과 최첨단 머신 러닝 방법보다 향상된 결과를 나타냅니다. LLM은 복잡한 분자 상호작용 패턴과 공통 유전자를 타겟으로 하는 약물 쌍의 경우를 효과적으로 캡처할 수 있음을 보여주며, 이는 제약 연구 및 임상 환경에서 유용한 도구로 자리잡을 것입니다.



### Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipelin (https://arxiv.org/abs/2502.06888)
- **What's New**: 최근 발표된 논문에서 Mixture of Experts (MoE) 모델의 성능을 향상시키기 위해 Klotski라는 효율적인 MoE 추론 엔진을 제안합니다. 이 시스템은 새로운 expert-aware multi-batch pipeline 패러다임을 통해 파이프라인의 버블(bubble)을 현저히 줄이는 데 초점을 맞추고 있습니다. Klotski는 계산 시간을 연장하여 다음 레이어의 로딩 시간과 겹치도록 하여 효율적인 처리를 보장합니다.

- **Technical Details**: Klotski의 핵심 기술은 고도의 컴퓨테이셔널 수요가 있는 '핫 전문가'와 낮은 I/O 수요의 '콜드 전문가' 간의 상호 보완 관계를 활용하는 것입니다. 이를 통해 다양한 배치에서 전반적인 I/O 수요를 최소화하면서 레이어 간 및 레이어 내의 버블을 줄이는 전략을 구현합니다. 또한, 제약 조건에 민감한 I/O-계산 계획 수립과 correlation-aware expert prefetcher를 설계하여 다양한 환경에 적응할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, Klotski는 Hugging Face Accelerate, DeepSpeed-FastGen, FlexGen 등 최신 기술들과 비교하여 최대 85.12배의 처리량 향상을 달성했습니다. 이로 인해 자원 제약 환경에서도 높은 처리량을 유지하면서 효율적인 MoE 추론을 가능하게 합니다. Klotski는 MoE 모델들의 추론 효율성을 극대화하여 다양한 분야에서의 적용 가능성을 높이고 있습니다.



### Gradient Based Method for the Fusion of Lattice Quantizers (https://arxiv.org/abs/2502.06887)
- **What's New**: 이 논문에서는 새로운 선형 격자 양자화 접근법을 제안합니다. 특히, 기존의 정사각형 배열(orthogonal splicing) 방법에서의 단점을 극복하기 위해 집단 알고리즘(Household Algorithm)과 행렬 지수 알고리즘(Matrix Exp Algorithm) 두 가지 혁신적인 방법을 도입하였습니다. 이들 방법을 통해 다양한 고차원에서의 격자 양자화 성능을 개선하는 데 성공했습니다.

- **Technical Details**: 격자는 ℝⁿ에서 선형 독립 벡터의 집합으로 정의됩니다. 본 논문에서는 최소 평균 제곱 오차(minimum mean square error, MSE)를 달성하는 최적 격자 양자화기를 정의하고, 이를 최적화하기 위해 경량 기법(gradient descent)을 활용하는 방법론을 연구했습니다. 게다가, Householder 반사 매트릭스를 활용하여 훈련 과정 동안 정사각형 배열을 유지함으로써 성능 최적화를 이루었습니다.

- **Performance Highlights**: 실험 결과에서 제안된 두 알고리즘인 Household Algorithm과 Matrix Exp Algorithm이 차원 13, 15, 17에서 19, 21, 22까지의 격자 양자화 성능을 향상시켰음을 보여주었습니다. 특히, Matrix Exp Algorithm은 고차원 설정에서 더 높은 효율성을 보이며, 기존 방법에 비해 더욱 우수한 결과를 나타냈습니다.



### Topological derivative approach for deep neural network architecture adaptation (https://arxiv.org/abs/2502.06885)
- **What's New**: 이번 연구에서는 신경망 아키텍처를 깊이 방향으로 점진적으로 조정하는 새로운 알고리즘을 제시합니다. 주된 초점은 새로운 레이어를 언제, 어디에 추가할지 결정하는 수학적 원칙을 확립하는 것입니다. 이를 위해 신경망 구조에 의존하는 'shape functional'과 그에 대한 topological derivative 개념을 도입했습니다. 이는 최적 제어 관점에서 신경망의 효율적인 변화를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 신경망을 점진적으로 깊어지게 하며, 이는 새로운 레이어가 추가되는 최적의 시점과 위치를 계산합니다. 알고리즘은 두 가지 버전으로 나뉘며, 하나는 미리 정의된 스케줄러를 사용하여 새 레이어의 추가 시기를 결정하고, 다른 하나는 검증 메트릭을 사용하여 자동으로 결정합니다. 특히, shape functional의 topological derivative를 통한 고유값 문제를 제시하여 새 레이어의 추가 위치와 초기값을 정할 수 있습니다.

- **Performance Highlights**: 다양한 회귀 및 분류 문제에서 완전 연결 네트워크, 합성곱 신경망, 비전 트랜스포머를 사용한 수치적 조사에서 본 연구의 접근법이 기존의 임의적인 기본 네트워크 및 다른 아키텍처 조정 전략보다 우수한 성능을 발휘함을 보여주었습니다. 또한, 전이 학습과 같은 여러 분야에서 topological derivative의 다른 응용 가능성도 시연하였습니다.



### Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models (https://arxiv.org/abs/2502.06884)
- **What's New**: 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)은 안전이 중요한 응용 분야에서 점점 더 자주 사용되고 있습니다. 본 논문에서는 이러한 모델의 불확실성을 정량화하고 의사 결정을 유연하게 관리하는 방법으로 학습 가능한 비순응적 기법(learnable conformal abstention)을 제안합니다. 이 접근 방식은 강화 학습(Reinforcement Learning, RL)과 비순응 예측(Conformal Prediction, CP)을 결합하여, 멀티 목표를 균형 있게 다루는 동적인 문턱 값을 설정합니다.

- **Technical Details**: 비순응 예측(CP)은 모델의 불확실성을 정량화할 수 있는 통계적 보장을 제공하는 프레임워크입니다. 그러나 기존 CP 방법은 정적인 임계 값에 의존하여 다양한 테스크 복잡성과 변화하는 데이터 분포에 적응하지 못합니다. 본 연구에서는 RL을 통합하여 동적으로 조정 가능한 비순응 예측 문턱을 도입함으로써 정확도, 범위, 신뢰성을 개선했습니다.

- **Performance Highlights**: 제안된 방법은 여러 가지 LLM/VLM 벤치마크 데이터셋에서 기존 방법보다 뛰어난 성능을 보였습니다. 특히, 정확도를 3.2% 향상시키고, 환각 탐지(AUROC) 성능을 22.19% 증가시켰습니다. 또한 불확실성 기반의 선택적 생성(AUARC)을 21.17% 개선하였고, 보정 오류(calibration error)를 70%-85% 줄였습니다. 이 모든 향상은 90% 이상의 범위 목표를 지속적으로 충족하며 안전이 중요한 응용 프로그램에서 신뢰할 수 있는 의사 결정을 위한 더 효과적이고 유연한 솔루션을 확립합니다.



### Multi-Agent Simulator Drives Language Models for Legal Intensive Interaction (https://arxiv.org/abs/2502.06882)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문은 법률 분야에서 상호작용적 법률 시나리오를 시뮬레이션하여 합성 데이터를 생성하는 Multi-agent Legal Simulation Driver (MASER)를 소개합니다. 이 시스템은 실제 법률 사건 소스를 활용하여 참가자 간의 법률 속성 일관성을 보장하며, 비Distractor 행동을 관리하기 위한 감독 메커니즘을 도입합니다. 또한, 이 논문은 동적 법률 시나리오에서 LLM의 성과를 평가하기 위해 Multi-stage Interactive Legal Evaluation (MILE) 벤치마크도 개발하였습니다.

- **Technical Details**: MASER는 고객(Client), 변호사(Lawyer), 감독(Supervisor)이라는 세 가지 에이전트로 구성되어 법적 목표(예: 고소장 작성)를 달성하는 방식으로 작동합니다. 각 캐릭터는 고유의 역할과 책임을 가지며, Big-5 성격 특성을 바탕으로 다양한 개인적 특징이 설정됩니다. 변호사는 고객의 법적 요구를 충족시키기 위해 사례 분석 및 관련 법률을 활용해야 하며, 감독은 상호작용을 모니터링하여 참가자 간의 행동 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, MASER는 기존의 법률 업무에서 LLM의 성능을 현저히 향상시켰습니다. 훈련된 모델은 GPT-4o와 같은 독점적인 고급 LLM 및 전문 법률 LLM보다 더 우수한 성능을 보여주었습니다. 연구진은 이 프레임워크가 복잡하고 사적인 법률 시나리오에서의 고급 상호작용 및 목표 달성을 위한 일종의 일반 패러다임으로 작용할 것으로 기대하고 있습니다.



### Mix Data or Merge Models? Balancing the Helpfulness, Honesty, and Harmlessness of Large Language Model via Model Merging (https://arxiv.org/abs/2502.06876)
Comments:
          Under Review

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 조화로운 정렬을 달성하기 위해 Helpfulness, Honesty, and Harmlessness(3H 최적화)를 중점으로 두고, 기존 방법의 한계를 넘어 모델 병합(model merging) 기법을 탐구합니다. 특히, 15가지 방법을 평가하여 모델 병합이 데이터 혼합(data mixture) 접근법보다 더 나은 성능을 보인다는 점을 입증했습니다. 또한, R-TSVM이라는 새로운 방법론을 제안하여 파라미터의 중복 및 이상치를 고려한 최적화를 통해 LLM 정렬을 향상시키는 방안을 제시합니다.

- **Technical Details**: 이 연구는 12개의 훈련 없이 병합할 수 있는 방법과 3개의 데이터 혼합 기법을 포함한 총 15가지 방법을 사용하여 10개의 데이터 세트를 평가했습니다. 세부적으로 3H 차원 간의 협업/충돌 관계에 대한 분석을 포함하여, 파라미터 수준의 충돌 해결을 통해 효과적인 3H 최적화를 달성하기 위한 조건을 제시합니다. 또한, R-TSVM 방법은 이상치 인식 파라미터 가중치와 희소성 적응 랭크 선택 전략을 통합하여 기존 방법의 한계를 극복할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델 병합 기법은 데이터 혼합 접근 방식에 비해 다양한 3H 목표 간의 균형을 효과적으로 유지하는 데 유리하다는 점이 확인되었습니다. 특히, 중복 부품 제거 및 이상치 완화를 통한 파라미터 수준에서의 충돌 해결이 매우 중요한 역할을 한다는 사실이 드러났습니다. 이러한 요소들은 LLM 정렬을 보다 안정적이고 효과적으로 만드는 데 기여하며, 실험을 통해 제안된 전략의 효용성이 입증됩니다.



### Beyond Vision: How Large Language Models Interpret Facial Expressions from Valence-Arousal Values (https://arxiv.org/abs/2502.06875)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 비주얼 입력 없이 얼굴 표정의 감정적 의미를 쉽게 추론할 수 있는지를 평가합니다. 특히, 이 연구는 Valence-Arousal(도움-각성) 값이라는 구조화된 숫자 표현을 사용하여 감정을 분류하고 설명하는 능력을 살펴봅니다. LLMs는 이러한 구조적 표현을 활용하여 비언어적 감정 소통의 맥락에서 강력한 인사이트를 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 IIMI와 Emotic이라는 두 개의 데이터셋을 사용하여 LLMs의 감정 분류와 의미 설명 달성 여부를 평가했습니다. 데이터를 처리하기 위해 FaceChannel이라는 패키지를 사용하였으며, 이 모델은 -1에서 1까지의 VA 값을 예측하고 기본적인 감정 카테고리로 분류합니다. LLMs는 이러한 VA 값을 기반으로 감정을 분류하고 그 표현을 설명하는 두 가지 실험에 참여하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 기본 감정의 분류에서 낮은 성능(정확도 30.42% 및 31.42%)을 보였습니다. 그러나 의미 설명을 하는 작업에서는 생성된 설명이 인간의 해석과 밀접하게 일치하는 경향을 보여, LLMs가 얼굴 표정의 자유로운 감정 추론에 있어 더 뛰어난 능력을 보였습니다. 이 연구는 비언어적 감정 인식을 위한 LLMs의 강점과 한계를 탐평하여 향후 AI의 감정 인지 시스템 개발에 기여할 수 있는 방향을 제시합니다.



### Group Reasoning Emission Estimation Networks (https://arxiv.org/abs/2502.06874)
- **What's New**: 이 논문에서는 GREEN(Group Reasoning Emission Estimation Networks)을 소개하여 온실가스(GHG) 배출량 추정을 위한 새로운 AI 기반 프레임워크를 제안하고 있습니다. 특히 이 연구는 20,850개 기업의 분야 분류를 자동화하고, 대규모 벤치마크 데이터세트를 구축하여 신뢰성 있는 배출량 추정 방법을 제공합니다. 또한, 대규모 데이터와 대규모 언어 모델(LLM)을 활용하여 배출량 예측의 정확성을 높이는 방안을 제시합니다.

- **Technical Details**: GREEN은 전통적인 전문가 기반 분류 방식 대신 정보 검색 문제로서 분야 분류를 재구성하고, Sentence-BERT 모델을 이용해 자가 감독적 대조 학습(self-supervised contrastive learning) 방법을 적용합니다. 이 프레임워크는 자연어 처리(NLP) 기술을 활용하여 기업당 연간 수익과 탄소 강도 계수를 곱하여 배출 예측을 수행하며, 이를 통해 높은 정확도의 분류 성능을 달성하였습니다.

- **Performance Highlights**: 실험 결과, 이 연구는 1,114개의 산업 카테고리에서 83.68%의 Top-1 정확도와 91.47%의 Top-10 정확도를 기록하였으며, 20개 기업에 대한 사례 연구에서 평균 절대 백분율 오차(MAPE)는 45.88%로 나타났습니다. 이는 대기업뿐만 아니라 중소기업(SMEs)에서도 효과적으로 활용할 수 있는 신뢰할 수 있는 배출량 추정 방법을 제공한다는 점에서 중요한 의미를 갖습니다.



### Multimodal Cognitive Reframing Therapy via Multi-hop Psychotherapeutic Reasoning (https://arxiv.org/abs/2502.06873)
Comments:
          NAACL 2025 Main

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 인지 재구성(cognitive reframing) 치료에서 잠재력을 보여주었지만, 비언어적 증거(non-verbal evidence)의 중요성이 간과되고 있음을 지적합니다. 따라서 연구진은 비주얼 단서를 통합한 다중 양식(multimodality)의 인지 재구성 접근 방식을 제안합니다. 이를 위해 새로운 데이터셋인 Multi Modal-Cognitive Support Conversation (M2CoSC)를 생성하여 GPT-4가 생성한 대화와 가상 클라이언트의 얼굴 표정을 짝지었습니다.

- **Technical Details**: M2CoSC 데이터셋은 심리 치료 세션을 시뮬레이션하기 위해 GPT-4의 역할 수행 능력을 활용하여 생성되었습니다. 연구진은 인지 재구성의 단계를 기반으로 세 가지 단계(소개, 문제 탐색, 브레인스토밍, 제안)를 확장하여 다중 양식 심리 치료 프레임워크를 수립했습니다. 다중 홉 심리 치료 추론(multi-hop psychotherapeutic reasoning) 방법을 도입하여 클라이언트의 상태를 이해하고 이에 기반한 더 합리적이고 공감가는 제안을 제공할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과 M2CoSC 데이터셋으로 훈련된 VLMs의 심리 상담 능력이 기존 LLMs를 초월하여 유의미하게 향상된 것으로 나타났습니다. 또한, 다중 홉 심리 치료 추론 방법은 VLMs가 더 thoughtful하고 공감하는 제안을 제공할 수 있게 하여 표준 프롬프트(prompting) 방법보다 성능이 우수하다는 것을 보여주었습니다. 이러한 연구는 AI 강화 심리 치료의 효율성을 높이기 위한 비언어적 단서의 활용 가능성을 열어줍니다.



### Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey (https://arxiv.org/abs/2502.06872)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 확보하기 위한 포괄적인 로드맵을 제공하는 것을 목표로 합니다. RAG는 외부 정보를 통합하여 인공지능 생성 콘텐츠(AIGC)의 문제를 해결하려는 접근 방식으로, 현재 이 시스템들이 직면한 여러 리스크를 다룹니다. 특히, 신뢰성, 프라이버시, 안전, 공정성, 설명 가능성, 책임의 다섯 가지 주요 관점을 바탕으로 이루어지는 연구 방향을 제시합니다.

- **Technical Details**: RAG 프레임워크는 정보 검색(information retrieval), 지식 증대(knowledge augmentation), 콘텐츠 생성(content generation)의 세 가지 주요 단계를 포함합니다. 정보 검색 과정에서 쿼리를 기반으로 관련 정보를 제공하며, 지식 증대 단계에서 검색된 지식은 언어 모델에 통합됩니다. 이러한 과정에서 발생할 수 있는 신뢰성 문제, 예를 들어 검색 편향(retrieval bias)과 허위 정보(hallucination) 문제에 대한 해결책이 필수적입니다.

- **Performance Highlights**: RAG 시스템은 의료 질문 응답, 법률 문서 작성, 교육용 챗봇 및 금융 보고서 요약 등 다양한 영역에서 효과적으로 적용되고 있습니다. 그러나 이러한 시스템의 신뢰성 문제는 특히 고위험 분야에서의 채택을 제한하여, 연구자들은 이러한 시스템이 신뢰할 수 있도록 다양한 방법을 개발하고 있습니다. 궁극적으로 RAG 시스템의 신뢰성 확보는 실제 적용에 큰 영향을 미친다는 점에서 중요한 문제임을 강조합니다.



### FlavorDiffusion: Predicting Food Pairings and Chemical Interactions Using Diffusion Models (https://arxiv.org/abs/2502.06871)
Comments:
          8 pages

- **What's New**: 이 연구는 머신 러닝의 도입으로 식품 페어링의 분석이 주관적인 경험을 넘어선 진전을 보여줍니다. FlavorDiffusion이라는 새로운 프레임워크를 소개하며, 이는 분광 분석(Chromatography)에 의존하지 않고 식품 화학 상호작용 및 재료 결합을 예측하는 방법을 제공합니다. 이 프레임워크는 그래프 기반 임베딩(Graph-based embeddings), 확산 과정(Diffusion processes), 화학적 속성 인코딩(Chemical property encoding)을 통합하여 데이터 불균형 문제를 해결하고 클러스터링 품질을 향상시키는 데 중점을 둡니다.

- **Technical Details**: FlavorDiffusion은 DIFUSCO(2023)와 같은 확산 모델을 활용하여 식품 화학 상호작용에 대한 더 풍부하고 구조화된 표현을 포착합니다. 우리의 모델은 데이터 불균형 문제를 해결하기 위해 균형 잡힌 서브그래프 샘플링 전략을 도입하여 서로 다른 재료-화학 조합의 공정한 표현을 보장합니다. 그래프 구조는 Recipe1M과 FlavorDB와 같은 대규모 데이터셋을 기반으로 구축되며, 식품 재료와 화학 화합물 간의 관계를 시각화합니다.

- **Performance Highlights**: 실험 결과 노드 임베딩(Normalized Pointwise Mutual Information, NPMI) 점수의 향상을 보여줍니다. 이 연구는 빈번하지 않은 화학 물질에 대한 크로마토그래피 결과 예측을 기반으로 하여 모델의 적용 가능성을 확장하며, 화학적 속성을 사용한 페어링 추론을 가능하게 하여 새로운 재료 조합에 대한 구조화되고 해석 가능한 추천을 제공합니다.



### Bridging Traffic State and Trajectory for Dynamic Road Network and Trajectory Representation Learning (https://arxiv.org/abs/2502.06870)
Comments:
          9 pages, 6 figures

- **What's New**: 이 논문에서는 도시 교통 관리의 효율성을 높이기 위해 새로운 프레임워크인 TRACK을 제안합니다. TRACK은 트래픽 상태 데이터와 궤적 데이터를 연결하여 동적인 도로 네트워크와 궤적 표현 학습을 제공합니다. 기존 방법들이 트래픽 상태와 궤적 데이터를 독립적으로 모델링하는 반면, TRACK은 이 두 데이터를 공동으로 모델링하여 주목받고 있습니다.

- **Technical Details**: TRACK은 정적 및 공간적인 도로 구간 특성을 인코딩하기 위해 graph attention networks (GAT)를 사용합니다. 또한, 궤적 표현 학습을 위한 변환기 기반 모델을 도입하고, 궤적 데이터에서 계산된 전이 확률을 GAT의 주의(weight)에 통합하여 도로 구간의 동적 공간적 특성을 포착합니다. 교통 상태 데이터의 시공간적 동역학을 학습하기 위한 트래픽 변환기 인코더를 설계합니다.

- **Performance Highlights**: REAL 데이터를 활용한 광범위한 실험 결과, TRACK은 최신 기법들보다 일관되게 우수한 성능을 보여주었습니다. 사례 연구를 통해 TRACK이 도로 구간과 궤적의 시공간 동역학을 효과적으로 포착할 수 있다는 것을 검증했습니다. 이러한 결과들은 TRACK의 모델링 접근법이 도시 교통 관리에 기여할 가능성을 나타냅니다.



### A Survey on Explainable Deep Reinforcement Learning (https://arxiv.org/abs/2502.06869)
- **What's New**: 이 논문은 Explainable Deep Reinforcement Learning(XRL)의 최신 동향을 다루며, DRL이 고차원의 환경에서 신뢰성과 투명성을 확보하기 위한 과제를 해결하기 위한 다양한 해석 가능성 기법을 제시합니다. 특히 LLM과의 통합을 통해 인간 피드백에 기반한 강화 학습의 적용을 탐구하고, 이로 인해 AI의 인간 선호 맞춤형 최적화가 가능해진 점이 주목할 만합니다. 또한 연구에 의해 제시된 XRL 기법들은 정책 개선, 적대적 공격 완화 및 AI 시스템의 안전성 강화를 목표로 한다.

- **Technical Details**: 강화 학습(RL)은 에이전트가 환경과 상호작용하며 순차적인 결정을 내리도록 훈련하는 기계 학습의 하위 분야입니다. 이 과정에서 에이전트는 상태-행동-보상의 MDP 마르코프 결정 프로세스를 통해 최적 정책을 학습하고 최대의 장기 보상을 추구합니다. 연구에서는 RL 설정을 온라인 및 오프라인으로 나누어 설명하고, 각각의 환경 상호작용 방식과 알고리즘의 종류(값 기반(value-based) 방법 및 정책 기반(policy-based) 방법)를 다루고 있습니다.

- **Performance Highlights**: 이 연구는 XRL 기술의 질적 및 양적 평가 프레임워크를 검토합니다. 이를 통해 산업 및 실생활 적용에서의 성과를 강조하며, XRL이 DRL의 해석 가능성을 높여 인간의 이해도를 증진시키는 데 기여할 것으로 예상하고 있습니다. 또한, 기존의 정책 및 공격 완화 방법에 대한 설명이 강화되어 AI 시스템의 안전성을 개선할 것으로 기대하고 있습니다.



### Related Knowledge Perturbation Matters: Rethinking Multiple Pieces of Knowledge Editing in Same-Subjec (https://arxiv.org/abs/2502.06868)
Comments:
          Accepted by NAACL 2025

- **What's New**: 본 연구에서는 Same-Subject Editing에 초점을 맞추어, 하나의 주체에 대한 여러 속성을 동시에 수정하여 일관되고 포괄적인 지식 업데이트를 목표로 합니다. 이를 위해 기존 벤치마크에서 동일 주체에 대한 충분한 편집 데이터를 찾지 못했음을 인지하고 새로운 S²RKE(같은 주체 관련 지식 편집) 벤치마크를 도입하였습니다. 이 벤치마크는 동일 주체에 대한 다수의 관련 편집을 연결합니다.

- **Technical Details**: Large language models (LLMs)은 주어진 토큰 시퀀스를 처리하며, Transformer 아키텍처에서 각 토큰은 히든 상태 벡터로 포매팅됩니다. 연구에서는 전통적으로 지식이 트리플 형태(s,r,o)로 표현되며, 각각 주체(subject), 관계(relation), 객체(object)를 나타냅니다. 본 연구에서는 여러 관련 지식을 함께 수정하는 Same-Subject Editing의 필요성을 강조하며, 특히 기존 편집 방법들이 이러한 작업에 비효율적임을 발견했습니다.

- **Performance Highlights**: 실험 결과, ROME 및 MEMIT과 같은 일상적인 locate-then-edit 방법은 동일 주체에 대해 여러 관련 정보를 효과적으로 업데이트하는 데 실패하였습니다. 이는 후속 편집이 이전 편집에 간섭하는 'related knowledge perturbation' 현상에 기인하며, 이 문제를 해결하기 위한 추가적인 연구가 필요합니다. 궁극적으로, 본 연구는 LLM에서의 Same-Subject Editing의 가능성을 제시하고 관련 연구의 방향성을 제안합니다.



### Forbidden Science: Dual-Use AI Challenge Benchmark and Scientific Refusal Tests (https://arxiv.org/abs/2502.06867)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 안전성을 평가하기 위한 오픈소스 데이터셋과 테스트 프레임워크를 개발하였습니다. 이 데이터셋은 유해한 콘텐츠의 적절한 거부와 합법적인 과학적 담론의 과도한 제한 여부를 측정하는 데 중점을 두고 있습니다. 또한, 다양한 프롬프트에 대한 네 가지 주요 모델의 응답을 분석하여 각 모델의 안전 프로파일을 분리해 보여주었습니다.

- **Technical Details**: 연구는 주로 제어된 물질 쿼리에 대한 네 가지 모델의 응답을 체계적으로 변형된 프롬프트에 따라 분석하였습니다. Claude-3.5-sonnet는 73% 거부율과 27% 허용률을 보이며 가장 보수적인 접근 방식을 보인 반면, Mistral은 100%의 쿼리에 대해 답변을 시도했습니다. GPT-3.5-turbo는 10%의 거부와 90% 허용을 보여 중간 정도의 제한을 보였으며, Grok-2는 20% 거부와 80% 허용을 기록하였습니다.

- **Performance Highlights**: 테스트 중 프롬프트 변동 전략을 분석한 결과, 단일 프롬프트에서 85%의 응답 일관성이 있었으나 다섯 개 변형을 사용했을 때는 65%로 감소하는 경향이 있었습니다. 공개된 이 벤치마크는 필요한 안전 제한과 합법적인 과학적 탐구의 과도한 검열 간의 비판적인 균형을 체계적으로 평가할 수 있게 합니다. 또한, 안전 메커니즘의 잠재적인 취약성을 드러내는 체인-오브-생각 분석을 통해, 바람직하고 유효한 과학적 담론을 지나치게 제한하지 않으면서 강력한 방어책을 구현하는 복잡성을 강조하였습니다.



### Global Ease of Living Index: a machine learning framework for longitudinal analysis of major economies (https://arxiv.org/abs/2502.06866)
- **What's New**: 이 연구에서는 다양한 사회경제적 및 인프라적 요인을 결합한 Global Ease of Living Index (EoLI)를 통해 삶의 질을 정량화하는 새로운 접근 방식을 제시합니다. 특히, 일부 경제 지표의 누락 데이터 문제를 해결하기 위한 머신러닝 프레임워크를 도입하여 기존 지표의 한계를 보완합니다. 이 접근 방식은 데이터와 코드가 공개되어 있어 다양한 맥락에서 쉽게 재현하고 적용할 수 있는 장점이 있습니다.

- **Technical Details**: 이 연구는 1970년 이후 주요 경제 국가에 대한 Ease of Living Index를 수립하기 위해 주성분 분석(Principal Component Analysis)과 같은 차원 축소 기법을 활용합니다. 또한, 데이터의 가중치를 부여할 때 문헌 및 전문가 지식을 기반으로 결정하여 삶의 질과 생활 편리성을 평가하는데 중요한 다양한 요인들을 포괄적으로 반영합니다. 이러한 여러 과정은 다기준 의사결정(Multiple Criteria Decision Making, MCDM) 프레임워크에서 특히 중요합니다.

- **Performance Highlights**: 이 연구는 정책 입안자들이 건강 관리 시스템, 고용 기회, 공공 안전 등 개선이 필요한 분야를 식별하는 데 도움을 줄 수 있는 실용적인 도구를 제공합니다. 데이터 임퓨테이션을 통한 정확한 평가를 목표로 하며, 경제 지표의 누락 현상에 대한 해결책을 제시하는 동시에, 전 세계 다양한 도시 상황에서도 적용 가능성을 높입니다. 결론적으로, 이 작업은 삶의 질 평가에 있어 투명성과 접근성을 높여, 정책 발전 및 지속적인 연구에 기여할 수 있는 귀중한 자원으로 자리잡고 있습니다.



### Knowledge Graph-Guided Retrieval Augmented Generation (https://arxiv.org/abs/2502.06864)
Comments:
          Accepted in the 2025 Annual Conference of the Nations of the Americas Chapter of the ACL (NAACL 2025)

- **What's New**: 본 논문에서는 지식 그래프(Knowledge Graph) 기반의 검색 증강 생성(Knowledge Graph-Guided Retrieval Augmented Generation, KG$^2$RAG) 프레임워크를 제안하여 기존의 검색 증강 생성 방식의 한계를 극복하려고 합니다. KG$^2$RAG는 지식 그래프를 활용하여 정보 조각 사이의 사실 수준 관계를 파악하고, 이에 기반한 다양한 정보 검색을 실현합니다. 이를 통해 더 다양한 정보 조각을 확보하고, 조화로운 응답 생성을 촉진합니다.

- **Technical Details**: KG$^2$RAG의 작동 과정에는 두 가지 주요 단계가 있습니다. 첫째, 감정 기반 검색을 통해 시드 조각(seed chunks)을 생성한 후, 이러한 조각을 확장하는 과정이 포함됩니다. 둘째, KG 기반의 맥락 조직(context organization) 단계에서 관련 정보를 필터링하고 내부적으로 일관된 단락으로 조정하여 LLM에 입력됩니다.

- **Performance Highlights**: HotpotQA 데이터세트를 사용한 광범위한 실험 결과, KG$^2$RAG는 기존의 검색 증강 생성 방식에 비해 응답 품질과 검색 품질 모두에서 우수한 성능을 보여주었습니다. 또한, ablation study를 통해 KG$^2$RAG의 다양한 모듈의 효과를 강조하였습니다. 개발된 데이터셋과 소스 코드는 GitHub에 공개되어, KG의 RAG 적용을 촉진합니다.



### BF-GAN: Development of an AI-driven Bubbly Flow Image Generation Model Using Generative Adversarial Networks (https://arxiv.org/abs/2502.06863)
- **What's New**: 이번 연구에서는 bubbly flow generative adversarial networks (BF-GAN)라는 새로운 생성 AI 아키텍처가 개발되었습니다. BF-GAN은 물리적으로 조건화된 입력(jg와 jf)을 통해 사실적이고 고품질의 bubbly flow 이미지를 생성하도록 설계되었습니다. 또한, 52세트의 다양한 조건 하에서 bubbly flow 실험을 수행하여 14만 개의 물리적 레이블이 붙은 이미지를 수집했습니다.

- **Technical Details**: BF-GAN은 mismatch loss와 pixel loss를 포함한 다중 스케일 손실 함수(multi-scale loss function)를 발전시켜 생성 성능을 향상시켰습니다. BF-GAN은 전통적인 GAN을 능가하는 생성 AI 평가 메트릭을 가지고 있으며, 생성된 bubbly flow의 주요 매개변수를 추출하여 측정값 및 경험적 상관관계와 비교하여 성능을 검증했습니다. 이는 BF-GAN의 생성 성능을 확고히 입증하는 결과입니다.

- **Performance Highlights**: BF-GAN은 주어진 jg와 jf에 대해 사실적이고 고품질의 bubbly flow 이미지를 생성할 수 있음을 보여주었습니다. 이 모델은 두 가지 상(fluid) 연구를 위한 생성 AI 솔루션을 제공하여 고품질 데이터를 확보하는 데 필요한 시간과 비용을 크게 절감할 수 있습니다. 또한, bubbly flow 탐지 및 분할 알고리즘을 위한 기준 데이터셋(generator)으로 작용하여 이 연구 분야의 전반적인 생산성을 향상시킬 수 있는 가능성을 가지고 있습니다.



### Design Considerations in Offline Preference-based RL (https://arxiv.org/abs/2502.06861)
- **What's New**: 이번 논문은 인간의 선호에 기반한 오프라인 강화 학습(Reinforcement Learning from Human Preferences, RLHF) 알고리즘의 다양한 설계 선택이 학습된 정책의 품질에 미치는 영향을 이론적으로 분석합니다. 특히 DPO, IPO, SLiC 등 여러 방법들이 어떻게 상호작용하는지에 대한 통찰력을 제공합니다. 기존 알고리즘들이 종종 사용해온 전통적인 reparameterization 같은 가정에 의존하지 않고, 보다 포괄적인 접근 방식을 취하고 있습니다.

- **Technical Details**: 본 연구에서는 오프라인 RLHF 기술에서 손실 최소화 문제를 해결하는 관점을 채택합니다. 이 과정에서 생성된 데이터의 커버리지와 손실 함수의 곡률이 학습된 정책의 성능과 밀접하게 연결되어 있음을 보여줍니다. 또한, 선호 데이터셋에 대한 손실 함수의 성질이 알고리즘 성능에 미치는 영향을 심층적으로 분석하며, 특히 DPO의 최적 softmax 정책을 벤치마크로 활용합니다.

- **Performance Highlights**: 실험 결과, IPO의 제곱 손실이 DPO의 로지스틱 손실을 초과하는 성능을 보이며, 이는 곡률 속성에 기인합니다. 또한, 참조 정책의 확률 정상화 과정을 제거하면 학습된 정책의 품질이 일관되게 감소하는 경향이 있음을 발견했습니다. 이번 연구는 이러한 결과들을 통해 이론적 발견을 실증적으로 검증하며, RLHF 기법의 향후 발전 방향에 대한 중요한 통찰력을 제공합니다.



### LLM-Supported Natural Language to Bash Translation (https://arxiv.org/abs/2502.06858)
Comments:
          13 pages, NAACL 2025

- **What's New**: 이 논문에서는 Bash 명령어를 자연어로 번역하기 위한 NL2SH 모델의 성능 평가에 필요한 새로운 테스트 데이터셋과 기능 동등성 휴리스틱을 제시합니다. 수동으로 검증된 600개의 지시-명령 쌍 및 40,939 쌍의 훈련 데이터셋은 기존 데이터셋에 비해 크기를 각각 441%와 135% 증가시켰습니다. 또한, 우리는 LLM의 명령 출력 평가를 결합한 새로운 기능 동등성 휴리스틱을 도입하여 두 Bash 명령의 기능적 동등성을 95% 확률로 평가할 수 있습니다.

- **Technical Details**: NL2SH 번역은 자동으로 텍스트나 음성을 다른 언어로 변환하는 기계 번역 범주에 해당합니다. LLM을 활용하여 생성된 Bash 명령어의 기능적 정확성을 평가하는 것은 필수적이며, 이는 입력에 대한 올바른 출력을 생산하는지를 비교함으로써 판단됩니다. 코드의 기능적 정확성을 보장하는 것은 복잡한 프로세스이며, 정적(static) 및 동적(dynamic) 분석 기법이 주로 사용됩니다.

- **Performance Highlights**: 우리는 인기 있는 LLM을 테스트 데이터셋과 새로운 휴리스틱을 이용하여 평가하였고, parsing, in-context learning, in-weight learning, 그리고 constrained decoding 기법들이 NL2SH 정확성을 최대 32%까지 개선할 수 있음을 증명했습니다. 이러한 발견은 데이터셋의 품질, 실행 기반 평가, 번역 방법이 NL2SH 번역 발전에 필수적임을 강조합니다.



### Gemstones: A Model Suite for Multi-Faceted Scaling Laws (https://arxiv.org/abs/2502.06857)
- **What's New**: 이번 연구에서는 다양한 아키텍처와 하이퍼파라미터를 사용하여 스케일링 법칙(scaling laws)을 탐색했습니다. 4000개 이상의 체크포인트를 포함하는 Gemstones를 공개하여, 이 체크포인트들이 모델 설계와 선택이 스케일링 법칙에 미치는 영향을 분석합니다. 이 데이터셋은 다양한 학습률, 쿨다운 일정, 아키텍처 형태로 훈련된 2억 개의 파라미터를 가진 변환기(transformers) 모델로 구성되어 있습니다.

- **Technical Details**: 스케일링 법칙의 디자인 과정에서 모델 선택, 학습률 결정 및 곡선 피팅 방법을 상세히 설명합니다. 모델은 Gemma 시리즈 및 Llama와 같은 인기 모델 시리즈의 선례를 바탕으로 제한 조건을 두어 모든 가능한 모델의 탐색 공간을 축소했습니다. 또한, 우리는 전체 모델의 단순한 규칙을 수립하고, 128개 이상으로 주목(head)을 설정하여 매칭을 가능하게 했습니다.

- **Performance Highlights**: 스케일링 법칙을 피팅하는 과정에서 이전의 스케일링 법칙의 취약성과 일반적인 오류를 강조했습니다. 연구 결과에 따르면, 스케일링 법칙의 매개변수와 해석은 선택한 모델 및 피팅 절차에 크게 의존합니다. 넓은 모델이 더 안전한 선택이며, 오버트레이닝이 효율적이라는 결과도 발견하여 실용적인 측면에서의 스케일링 법칙의 활용을 제안했습니다.



### Self-Supervised Prompt Optimization (https://arxiv.org/abs/2502.06855)
- **What's New**: 본 논문에서는 Self-Supervised Prompt Optimization (SPO)이라는 새로운 프레임워크를 제안합니다. SPO는 외부 참조 없이도 효과적인 프롬프트를 발견할 수 있는 비용 효율적인 방법론으로, 고정형 및 개방형 작업에 모두 적용 가능합니다. 기존의 프롬프트 최적화 방법들과 달리, SPO는 실험적 조정이나 전문가의 도움이 필요하지 않습니다.

- **Technical Details**: SPO는 LLM(대형 언어 모델)의 출력을 비교하여 평가 및 최적화 신호를 추출합니다. 이 방법은 LLM 평가자가 수행한 쌍별 출력 비교를 통해 우수한 프롬프트를 선택하고, 이후 LLM 최적화기가 출력과 작업 요구 사항을 일치시키는 방식으로 진행됩니다. 이를 통해 SPO는 직접적인 출력 비교를 사용하여 고품질의 프롬프트를 생성합니다.

- **Performance Highlights**: 광범위한 실험 결과 SPO가 최첨단 프롬프트 최적화 방법들을 능가하는 것으로 나타났습니다. 구체적으로 SPO는 기존 방법에 비해 1.1%에서 5.6%까지 비용을 절감하며, 샘플 수 또한 크게 줄이자는 (예: 세 개의 샘플) 결과를 보였습니다. 이로써 SPO는 낮은 비용으로 경쟁력 있는 성능을 달성할 수 있음을 입증하였습니다.



### Can Large Language Models Understand Intermediate Representations? (https://arxiv.org/abs/2502.06854)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 Intermediate Representations (IRs)을 이해하는 능력을 최초로 탐구하는 경험적 연구로, GPT-4와 같은 최신 모델들이 IR 관련 다양한 작업에서 어떻게 성능을 발휘하는지를 분석하였습니다.

- **Technical Details**: 연구는 Control Flow Graph (CFG) 재구성, IR 역컴파일링(decompilation), 코드 요약, 실행 추론(execution reasoning) 등 네 가지 작업을 통해 LLM의 능력을 평가하였습니다. 또한, LLM이 IR 문법을 파싱하고 주요 구조를 인식하는 데는 능숙하지만, 제어 흐름 추론과 실행 의미론을 처리하는 데 어려움을 겪는다는 문제가 있다고 보고했습니다.

- **Performance Highlights**: 평가 결과, LLM이 IR 구문을 인식하는 데는 일정 수준의 능력을 보이나, 제어 흐름 지침(br, jmp 등)과 같은 더 깊은 실행 의미를 포착하는 데는 고전적 문제를 드러냈습니다. 특히 이 연구는 IR 모델링을 위한 별도의 훈련이 필요할 수 있음을 시사하며, LLM의 IR 이해를 증진시키기 위한 여러 개선사항을 제안합니다.



### Native Fortran Implementation of TensorFlow-Trained Deep and Bayesian Neural Networks (https://arxiv.org/abs/2502.06853)
Comments:
          Submitted for inclusion in the 2025 American Nuclear Society Annual Conference

- **What's New**: 지난 10년간 핵공학 분야에서 머신러닝(ML)의 연구가 크게 증가했습니다. 이 연구는 연료 배열 매개변수 예측부터 발전소 장비의 이상탐지까지 다양하게 활용되고 있습니다. ML 모델 구현의 실용성을 평가하기 위한 새로운 단계의 연구가 필요하며, 이는 생산 환경에서의 ML 모델 적용 가능성을 확인하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 TensorFlow에서 훈련된 모델을 Fortran 내에서 직접 구현할 수 있는 프레임워크를 제공합니다. 이 프레임워크는 DNN(deep neural networks)과 BNN(Bayesian neural networks)을 지원하며, Python 런타임이나 TensorFlow의 C API, ONNX 변환에 의존하지 않습니다. 모델 초기화, 가중치 로드, 데이터 표준화 등 예측까지의 과정을 간소화하여 Fortran 코드에서 쉽게 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: DNN 모델은 19.6배의 속도 향상을 보였고, BNN 모델 또한 8.0배의 속도 향상을 기록했습니다. 두 모델 모두 Fortran과 TensorFlow 간의 예측 결과 비교를 통해 우수한 합의를 보여 주었으며, 작은 차이만 있었습니다. 결과적으로, 이 프레임워크는 실제 핵응용 분야에서 ML 기반 방법을 지속적으로 연구할 수 있는 효과적인 방식으로 밝혀졌습니다.



### EAP-GP: Mitigating Saturation Effect in Gradient-based Automated Circuit Identification (https://arxiv.org/abs/2502.06852)
- **What's New**: 본 논문은 transformer 기반 언어 모델의 내부 메커니즘을 이해하기 위한 접근법인 기계적 해석 가능성을 다룹니다. 특히, 기존의 gradient 기반 회로 식별 방법의 문제점을 지적하고, 새로운 방법인 Edge Attribution Patching with GradPath (EAP-GP)를 제안하여 기계 학습 모델의 회로 식별의 신뢰성을 개선하고자 합니다. EAP-GP는 입력의 변화에 대한 민감도를 개선하고 회로 식별의 신뢰성을 높이기 위한 알고리즘입니다.

- **Technical Details**: EAP-GP는 입력 데이터와 기준 입력 사이의 적응형 통합 경로를 구성하여 최적의 경로를 따라 이동합니다. 이를 통해 입력 변화에 대한 민감도를 제한하는 포화 효과(saturation effect)를 피할 수 있습니다. 논문에서는 GPT-2 Small, Medium, XL 모델에서 EAP-GP를 평가하였고, 6개의 데이터셋을 사용하여 기존 방법들에 비해 최대 17.7%의 성능 향상을 보여주었습니다.

- **Performance Highlights**: EAP-GP는 회로의 정확성을 측정하는 정밀도와 재현율을 기존 방법과 동등하게 혹은 더 우수하게 보였습니다. 이는 EAP-GP가 수동으로 주석을 단 기준 회로와 비교했을 때 더 신뢰할 수 있는 성능을 제공함을 나타냅니다. 한편, EAP-GP는 더 큰 모델에서도 탁월한 성능을 유지하며, 이전 연구들과의 비교에서도 뚜렷한 개선을 보여주었습니다.



### Survey on Vision-Language-Action Models (https://arxiv.org/abs/2502.06851)
- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델에 대한 AI 생성 리뷰를 소개하며, 주요 방법론, 발견 및 향후 방향을 요약합니다. 이 내용은 대형 언어 모델(LLMs)을 사용해 생성되었으며, 과학 문헌 리뷰 자동화의 가능성을 강조합니다. AI가 생성한 콘텐츠의 정확성과 신뢰성을 보장하는 것은 여전히 도전 과제가 되고 있으며, 향후 연구는 AI 지원 문헌 리뷰를 위한 구조적 프레임워크 개발에 중점을 둘 예정입니다.

- **Technical Details**: 이 연구에서 소개된 Actra는 엣지 디바이스에서 머신 러닝 모델을 배치하기 위한 혁신적인 접근 방식입니다. Actra는 경로 주의(trajectory attention)와 학습 가능한 액션 쿼리(learnable action queries)를 결합하여 로봇 작업의 추론 과정을 최적화합니다. 또한, ProSim-Instruct-520k라는 대규모 다중 모달 데이터셋을 개발하여 로봇 조작 시스템의 학습 능력을 향상시키고 있으며, 이 데이터셋은 520,000개 이상의 실제 주행 시나리오와 1,000만 개 이상의 텍스트 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: Actra는 다양한 로봇 환경에서 실험을 거쳐 성능 및 효율성을 크게 향상시켰음을 보여주었습니다. 특히 복잡한 다중 모달 작업 처리에 효과적이며, 여러 작업 및 객체 카테고리에서 우수한 성공률을 기록했습니다. 결과적으로, Actra는 로봇 조작에 있어 더 효율적이고 강력한 추론 방법을 제공하며, 앞으로의 연구를 위한 대규모 자원을 마련함으로써 데이터 기반 로봇 모델 개선에 기여할 수 있습니다.



### Model Fusion via Neuron Transplantation (https://arxiv.org/abs/2502.06849)
Comments:
          18 pages, 7 figures, conference: ECML-PKDD 2024

- **What's New**: 본 논문에서 제안하는 Neuron Transplantation (NT) 기법은 앙상블 모델을 효과적으로 융합하여 필요 없는 뉴런을 제거하고 중요한 뉴런만 이식함으로써 모델을 단순화합니다. 기존의 앙상블 학습 기법에 비해 메모리와 추론 시간 요구사항을 줄이는 동시에 성능 손실을 최소화합니다.

- **Technical Details**: NT는 레이어별로 중요한 뉴런을 선택하여 결합하는 프로세스로 구성됩니다. 초기 학습 후, 크기가 같은 모델로 통합하여 손실이 발생한 뉴런을 보완하기 위해 전체 데이터 셋으로 모델을 미세 조정합니다. 이 방법은 기존의 가중치 평균화나 정렬 기반 방법보다 적은 메모리를 사용하고 더 빠른 성능 향상을 제공합니다.

- **Performance Highlights**: NT는 여러 가지 모델 설정에서 효과적으로 적용 가능하였으며, 앙상블 구성원 간의 정보 손실을 줄인 채로 뛰어난 성능을 보였습니다. 연구 결과, NT 방법은 단일 모델의 메모리 및 계산 성능을 유지하면서도 앙상블 효과를 낼 수 있음을 보여주었습니다.



### Transfer learning in Scalable Graph Neural Network for Improved Physical Simulation (https://arxiv.org/abs/2502.06848)
- **What's New**: 본 연구에서는 GNN(그래프 신경망) 기반 물리 시뮬레이터의 성능과 훈련 효율성을 향상시키기 위한 프리트레인 및 전이 학습 패러다임을 소개합니다. 우리는 SGUNET(스케일러블 그래프 U-net)라는 새로운 모델을 제안하며, 이 모델은 다양한 메쉬 크기와 해상도에 맞춰 조정될 수 있도록 설계되었습니다. 또한, SGUNET 모델 간의 파라미터 조정을 지원하기 위한 매핑 함수를 제안하여 전이 학습을 가능하게 합니다.

- **Technical Details**: SGUNET은 Encoder-Processor-Decoder 아키텍처를 따르며, 깊이 우선 탐색(DFS) 풀링 방식을 도입하여 다양한 시뮬레이션 작업에 적합하도록 설계되었습니다. 프리트레인을 위해 ABC(오픈 소스 세트) 데이터셋에서 랜덤하게 선택된 3D 형태의 20,000개 물리 시뮬레이션 데이터셋이 생성되었습니다. 모델의 일반화 성능을 높이기 위해, 미리 훈련된 가중치와 타겟 모델 가중치 간의 차이를 제어하는 추가 정규화 항이 손실 함수에 포함됩니다.

- **Performance Highlights**: 2D Deformable Plate 벤치마크 데이터셋에서 ABCD 데이터셋으로 프리트레인한 모델은 전체 훈련 데이터의 1/16로 파인튜닝 되었을 때, 스크래치에서 훈련한 모델보다 위치 RMSE에서 11.05% 향상된 성능을 보였습니다. 3D Deforming Plate 데이터셋에서도 프리트레인된 모델은 훈련 데이터의 1/8로 파인튜닝할 때 같은 성능을 달성했습니다. 이를 통해, 전이 학습 접근법이 훈련 데이터 및 시간 절약 측면에서 향상된 성능을 제공할 수 있음을 보여줍니다.



### Prot2Chat: Protein LLM with Early Fusion of Sequence and Structur (https://arxiv.org/abs/2502.06846)
Comments:
          9 pages, 2 figures

- **What's New**: 본 연구는 프로틴 Q&A 시스템의 한계를 극복하기 위한 새로운 프레임워크인 Prot2Chat을 제안합니다. 이 프레임워크는 다중 모드 프로틴 표현을 자연어와 통합하여 대규모 언어 모델(LLM) 기반의 답변 생성을 가능하게 합니다. Prot2Chat은 수정된 ProteinMPNN 인코더와 교차 주의 메커니즘을 적용한 프로틴-텍스트 어댑터 및 LLaMA3 디코더를 결합하여 설계되었습니다.

- **Technical Details**: Prot2Chat은 단일 모듈을 사용하여 단백질 서열 및 구조 정보를 통합합니다. 연구진은 인코더를 고정하고, LoRA 기법을 통해 디코더의 훈련 효율성을 최적화하였습니다. 결과적으로 이모델은 Mol-Instructions 및 UniProtQA 데이터셋을 사용하여 수행된 실험에서 높은 성능을 보여주었습니다.

- **Performance Highlights**: 실험 결과, Prot2Chat은 기존의 단백질 Q&A 시스템보다 우수한 일반화 능력을 보였으며, 전문가 평가 및 온라인 KIMI 점수 평가에서 높은 일관성을 유지했습니다. 특히, 단백질 서열과 구조 정보를 통합한 모델이 단독 서열 훈련이나 기존 ESM 기반 인코더를 사용하는 모델보다 월등한 성능을 발휘했습니다.



### DiffNMR3: Advancing NMR Resolution Beyond Instrumental Limits (https://arxiv.org/abs/2502.06845)
Comments:
          13 pages, 6 figures

- **What's New**: 이 논문은 고해상도의 핵자기 공명(NMR) 스펙트럼을 생성하기 위해 AI를 활용한 새로운 방법론을 도입합니다. 이 방법은 저자기장(low-field) NMR 데이터를 이용하여 고자기장(high-field) 스펙트럼을 재구성할 수 있는 초해상도(super-resolution) 기술을 활용하며, 연구자들이 비싼 장비 없이도 고해상도 분석을 수행할 수 있도록 합니다.

- **Technical Details**: 이 AI 기반 접근법은 확산 모형(diffusion model)을 이용하여 저자기장 NMR 데이터에서 고자기장 NMR 스펙트럼을 재구성합니다. 이를 통해 실험 세트업이나 연구 질문에 따라 다양한 자기장 강도에서 유연한 재구성이 가능하게 하여, 고해상도 스펙트럼을 얻을 수 있습니다. 논문은 이 방식이 기존 고자기장 장비에서 얻은 결과와 유사한 재구성을 가능하게 한다고 언급합니다.

- **Performance Highlights**: 이 방법은 저자기장 NMR 데이터에서 복잡한 샘플을 더욱 정확하게 분석할 수 있도록 하여 NMR 분광학의 한계인 주파수 해상도의 제약을 극복합니다. 이를 통해 상대적으로 저렴한 장비를 사용하는 연구자나 산업계에서도 고해상도를 달성할 수 있음을 보여줍니다. 이로 인해 더 많은 연구자들이 고가의 NMR 장비 없이도 필요한 스펙트럼 품질에 접근할 수 있게 됩니다.



### Exploring Model Invariance with Discrete Search for Ultra-Low-Bit Quantization (https://arxiv.org/abs/2502.06844)
- **What's New**: 이번 논문에서는 InvarExplore라는 통합 프레임워크를 소개하며, 이는 서로 다른 모델 불변성(invariance)을 동시에 탐색하여 양자화 성능을 향상시키는 데 중점을 두고 있습니다. 특히, 이 방법은 grad-based 방식으로 최적화할 수 없는 permutation 불변성을 탐색할 수 있는 이산 탐색 알고리즘을 특징으로 합니다. 이로 인해 기존의 최첨단 방법들과 호환되며, 성능 개선을 위한 추가적인 기회를 제공합니다.

- **Technical Details**: Post-training quantization 기술을 활용하여 훈련이 완료된 후에도 모델의 메모리 사용량을 줄이고, 모델 가중치의 비트 수를 감소시킵니다. 본 논문에서는 activation-guided discrete search 알고리즘을 제안하여 permutation, rotation, scaling과 같은 다양한 불변성의 조합을 탐색합니다. 이를 통해 낮은 정밀도의 설계에서도 모델 성능을 유지할 수 있도록 돕습니다.

- **Performance Highlights**: InvarExplore는 기존의 양자화 방법들과 병행하여 작동할 수 있는 능력을 갖추고 있어, 모형을 나타내는 데 필요한 forward passes만을 요구합니다. 여러 언어 모델링 과제와 자연어 처리 과제에서 실험을 통해 입증된 바에 따르면, 이 방법은 현재의 부하를 용이하게 극복하면서 성능 개선을 성취합니다. 결과적으로 InvarExplore는 기존 기술 대비 더 나은 결과를 보여줍니다.



### Vision-Integrated LLMs for Autonomous Driving Assistance : Human Performance Comparison and Trust Evaluation (https://arxiv.org/abs/2502.06843)
- **What's New**: 이 논문에서는 전통적인 자율주행 시스템이 복잡하고 예기치 않은 상황에서의 추론에 한계를 보인다는 문제에 대응하기 위해, Large Language Model (LLM) 기반의 자율주행 보조 시스템을 소개합니다. 이 시스템은 시각 이해와 의사결정을 향상시키기 위해 비전 어댑터와 LLM 추론 모듈을 통합하였습니다.

- **Technical Details**: 비전 어댑터는 YOLOv4와 Vision Transformer (ViT)를 결합하여 포괄적인 시각적 특징을 추출합니다. 또한, GPT-4는 인간과 유사한 공간적 추론 및 반응 생성을 가능하게 하며, 이를 통해 시스템의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: 45명의 숙련된 운전자를 대상으로 한 실험 평가 결과, 제안된 시스템은 상황 설명에서 인간의 성능을 잘 모방하며, 적절한 반응 생성을 위한 인간의 결정과 중간 정도의 일치를 보였습니다.



### Integrating Generative Artificial Intelligence in ADRD: A Framework for Streamlining Diagnosis and Care in Neurodegenerative Diseases (https://arxiv.org/abs/2502.06842)
Comments:
          20 pages, 1 figure

- **What's New**: 이 논문은 신경학적 치료의 수요 증가에 대응하기 위해 대형 언어 모델(LLMs)의 활용 가능성을 제시합니다. 특히 알츠하이머병 및 관련 치매(ADRD)에 대한 연구에서, 임상의들이 인식하지 못하는 통찰력의 검증이 어려운 상황을 해결하기 위해 LLMs가 어떻게 도움을 줄 수 있는지를 다루고 있습니다.

- **Technical Details**: 연구에서 제안하는 프레임워크는 LLM의 효과적인 커뮤니케이션 능력을 활용하여 임상의의 데이터 수집, 복잡한 임상 정보 해석, 그리고 적시에 적절한 의료 지식을 적용하는 능력을 향상시키는 데 중점을 둡니다. 또한 인간의 감독(oversight)을 유지하며 고품질의 표준화된 데이터 수집을 우선시하여 환자와의 모든 접촉에서 학습할 수 있는 시스템을 구축하고자 합니다.

- **Performance Highlights**: 이 접근법은 진단 정확성 향상, 치료 불균형 감소, 학습하는 의료 시스템을 통한 임상 지식 증진의 가능성을 강조하고 있습니다. 윤리적 고려사항 및 거버넌스 필요성에 대한 중요 논의도 시작되어, 이 로드맵은 신경학 및 기타 의학 전문 분야에서 책임 있는 AI 통합을 위한 원칙을 제공합니다.



### A Hybrid Model for Weakly-Supervised Speech Dereverberation (https://arxiv.org/abs/2502.06839)
- **What's New**: 이 논문은 최소한의 음향 정보(acoustic information)와 잔향 음성(reverberant speech)을 사용하여 음성 비잔향화 시스템(speech dereverberation system)의 성능을 향상시키기 위한 새로운 훈련 전략을 소개합니다. 기존의 알고리즘은 주로 건조/습식(dry/wet) 데이터의 쌍에 의존하거나, 잔향 특성을 적절히 포착하지 못하는 목표 메트릭(target metrics)을 사용하여 결과가 좋지 않을 수 있는 문제점을 가지고 있었습니다.

- **Technical Details**: 제안된 접근 방식은 잔향 시간(reverberation time, RT60)과 같은 제한된 음향 정보를 사용하여 비잔향화 시스템을 훈련시킵니다. 시스템의 출력은 생성된 방 임펄스 응답(generated room impulse response)을 사용하여 재합성되고, 원래의 잔향 음성과 비교됩니다. 이를 통해 표준 목표 메트릭을 대체하는 새로운 잔향 매칭 손실(reverberation matching loss)을 제공합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 최신 기술(state-of-the-art)보다 다양한 객관적 메트릭(objective metrics)에서 보다 일관된 성능을 달성하는 것으로 나타났습니다. 이는 제한된 데이터로도 효과적인 비잔향화 시스템을 구성할 수 있음을 보여주고 있습니다.



### CAST: Cross Attention based multimodal fusion of Structure and Text for materials property prediction (https://arxiv.org/abs/2502.06836)
Comments:
          10 pages, 3 figures

- **What's New**: 최근 AI의 발전은 재료 과학에서의 물성 예측을 혁신적으로 변화시켰습니다. Graph Neural Networks(GNNs)는 결정 구조를 그래프로 표현하여 국소 상호작용을 효과적으로 포착하는데 뛰어난 성능을 보입니다. 그러나 기존 방법들은 결정 시스템과 반복 유닛 연결성 같은 중요한 전역 정보를 잃어버리는 한계가 있습니다. 이를 해결하기 위해, 우리는 CAST라는 크로스 어텐션 기반의 다중 모달 융합 모델을 제안하여 그래프와 텍스트를 결합하여 필수적인 재료 정보를 보존합니다.

- **Technical Details**: CAST는 노드 및 토큰 수준의 특징을 크로스 어텐션 메커니즘을 사용하여 결합하여 이전의 물질 수준 임베딩에 의존하는 방법들보다 우수한 결과를 보여줍니다. 특히, Masked Node Prediction(MNP) 사전 훈련 전략을 통해 원자 수준의 정보 통합을 강화합니다. 다양한 자료셋에 대한 실험 결과, 우리 방법은 CrysMMNet 및 MultiMat과 같은 기존 방법들보다 최대 22.9%의 성능 향상을 기록했습니다. 또한, 어텐션 맵 분석을 통해 노드와 토큰 간의 관계를 효과적으로 포착했음을 확인했습니다.

- **Performance Highlights**: CAST는 재료 예측 측면에서 높은 정확성을 보여줍니다. 특히, 네 가지 결정 속성(총 에너지, 밴드갭, 전단계수, 부피모듈러스)에 대한 예측에서 데이터셋 크기와 무관하게 일관된 성능 향상을 입증했습니다. 다양한 평가 기준을 통해 CAST는 질 좋은 자료를 바탕으로 한 후보제안 및 실용적인 응용 가능성을 보여주며, 다중 모달 학습이 재료 과학 분야에서의 예측 모델에서 매우 유망함을 강조합니다.



### A Unified Knowledge-Distillation and Semi-Supervised Learning Framework to Improve Industrial Ads Delivery Systems (https://arxiv.org/abs/2502.06834)
- **What's New**: 이 논문에서는 산업 광고 순위 시스템의 성능을 개선하기 위해 새로운 프레임워크인 UKDSL(Unified framework for Knowledge-Distillation and Semi-supervised Learning)을 제안합니다. UKDSL은 다양한 광고 데이터 세트를 활용해 과적합(overfitting)을 줄이고, 교육 데이터와 실제 운영 데이터 간의 불일치를 완화합니다. 이를 통해 모델은 대규모 비라벨 데이터에서 학습할 수 있어 성능을 향상시킵니다.

- **Technical Details**: 기존의 다단계 광고 순위 시스템에서는 간결한 모델이 효율성은 높지만 정확도가 떨어지는 문제를 가지고 있습니다. UKDSL은 이러한 문제를 해결하기 위해 이론적 분석과 수치 시뮬레이션을 통해 모델의 비일관성과 잘못된 조정(miscalibration) 문제를 밝힙니다. 이 프레임워크는 산업 규모의 광고 순위 시스템에 적용되어 오프라인과 온라인 모두에서 성능이 극적으로 향상됐음을 입증했습니다.

- **Performance Highlights**: UKDSL은 여러 가지 광고 모델에서 성공적으로 배포되어 다수의 사용자에게 서비스를 제공하는 첫 사례로 확인되었습니다. 이 프레임워크는 다수의 지표에서 성능을 개선하며, 대규모 데이터 환경에서도 효율성을 유지하는 장점을 가지고 있습니다. 최종적으로, UKDSL은 다양한 이벤트와 지리적 위치, 클라이언트에 최적화되어 있습니다.



### Entropy Adaptive Decoding: Dynamic Model Switching for Efficient Inferenc (https://arxiv.org/abs/2502.06833)
- **What's New**: 이번 논문에서는 Entropy Adaptive Decoding (EAD)라는 새로운 접근법을 소개합니다. EAD는 예측 불확실성에 따라 서로 다른 크기의 언어 모델을 동적으로 전환하여 높은 효율성을 제공합니다. 이 방법은 모델의 logit 분포에서 rolling entropy를 모니터링하여 텍스트의 생성 복잡도에 맞춰 필요한 모델 규모를 결정합니다.

- **Technical Details**: EAD에서는 두 개의 서로 다른 매개변수 수를 가진 모델, 즉 작은 모델 MS와 큰 모델 ML을 사용합니다. 주어진 토큰 시퀀스에 대해 각 모델은 다음 토큰에 대한 비정규화된 확률을 나타내는 logits를 생성하며, 이 logits의 엔트로피를 계산하여 예측의 난이도를 추정합니다. 흔들림을 줄이기 위해 우리는 평균 엔트로피를 계산하는 rolling window를 유지하며, 이에 기반하여 모델 전환을 결정합니다.

- **Performance Highlights**: MATH 벤치마크에서의 실험 결과, EAD는 LLaMA 모델 계열에서 11B 모델 성능의 96.7%를 유지하면서도 토큰의 43%만 사용하여 계산 비용을 41.5% 줄이는 성과를 나타냈습니다. Qwen 모델 계열의 경우, 14B 모델 성능의 92.9%를 달성하면서도 오직 25%의 토큰만 사용해 계산 비용을 67% 절감했습니다. 이러한 결과는 언어 모델의 연산 최적화에서 새로운 방향성을 제시합니다.



### Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach (https://arxiv.org/abs/2502.06832)
Comments:
          10 pages, 3 figures, submitted to ICML 2025 (under review)

- **What's New**: 이번 논문에서는 Mixture of Experts (MoE) 아키텍처의 취약점을 분석하고, 전문가 네트워크의 강인성을 높이기 위한 Robust Training with Experts’ Robustification (RT-ER) 기법을 제안합니다. 특히, 복합적인 모듈 구조가 가지는 강점을 활용하며, 하나의 추가 전문가에 대해서만 강인성을 보장하면서도 훈련 및 추론의 효율성을 유지합니다. 듀얼 모델(strategy) 접근법을 통해 표준 MoE 모델과 강인화된 MoE 모델을 결합하여 정확도와 강인성 간의 균형을 조정할 수 있는 방법을 도출합니다.

- **Technical Details**: MoE 아키텍처는 입력 데이터에 따라 각기 다른 전문가 네트워크의 출력을 조합하여 복잡한 패턴을 포착합니다. 이 논문에서는 adversarial 예제가 전문가 네트워크에 미치는 영향을 분석하고, 이를 기반으로 강인성을 높이는 방법론을 개발하였습니다. 특히, 듀얼 모델을 통해 표준 MoE와 강인화된 MoE 모델의 선형 결합을 제안하고, 이를 바탕으로 JTDMoE라는 새로운 공동 훈련 전략을 소개합니다.

- **Performance Highlights**: CIFAR-10 및 TinyImageNet 데이터셋에서 ResNet18 및 Vision Transformer 아키텍처를 사용한 실험 결과, 제안된 방법들이 강인성과 정확도를 향상시킨다는 것이 입증되었습니다. 특히, RT-ER 접근법을 활용하면 전문가 네트워크의 강인성을 높이며 훈련 및 추론의 효율성을 유지할 수 있습니다. 논문에서는 이와 같은 방법이 실용적인 환경에서 안전성을 높이는 데 기여할 수 있음을 강조합니다.



### No Location Left Behind: Measuring and Improving the Fairness of Implicit Representations for Earth Data (https://arxiv.org/abs/2502.06831)
- **What's New**: FAIR-Earth는 지구 표현의 불평등을 조사하고 도전하기 위해 특별히 제작된 첫 번째 데이터셋입니다. 이는 다양한 고해상도 지구 신호와 함께 지형 규모 및 인구 밀도와 같은 수준별 메타데이터를 집계하여 모델의 공정성을 평가합니다. 이 연구는 implicit neural representations (INRs)의 효율적이고 공정한 평가에 기여할 새로운 기초 자료를 제공합니다.

- **Technical Details**: 현재까지 지구과학 응용 분야에서 INRs의 사용은 급속히 증가하고 있으며, 이러한 모델은 비선형 모델 학습에 효과적입니다. FAIR-Earth는 지구과학 신호를 포괄하는 여러 벤치마크 데이터셋을 포함하고 있으며, 각기 다른 계층에서 모델 성능의 불균형을 평가하기 위한 방법론 및 메타데이터 세트를 제공합니다. 이 데이터셋은 지리적 특성, 인구 밀도 및 정치적 경계와 같은 정보로 풍부하게 구성되어 있습니다.

- **Performance Highlights**: FAIR-Earth에서의 extensive 실험을 통해 기존의 최첨단 INR 방법들의 성능간에서 실질적인 차이를 발견하였습니다. 특히, 고주파 신호(예: 섬, 해안선)와 관련된 하위 그룹이 기존 방법으로 모델링할 때 꾸준히 열악한 성능을 보이는 것을 확인했습니다. 이를 해결하기 위해, 구형 웨이브릿 인코딩을 도입하여 다양한 스케일과 위치에서 일관된 성능을 제공함으로써 불평등한 그룹을 보다 정확하게 표현할 수 있음을 입증하였습니다.



### OrderFusion: Encoding Orderbook for Probabilistic Intraday Price Prediction (https://arxiv.org/abs/2502.06830)
Comments:
          9 pages, 5 figures, 3 tables

- **What's New**: 본 논문은 전력 시장의 불확실성을 관리하고 견고한 거래 전략을 지원하기 위해 intraday 전기 가격의 확률적 예측을 다룹니다. 기존 방법들은 입찰과 제안 간의 상호 의존성을 충분히 활용하지 못해 파라미터 효율성이 떨어지는 문제점이 있으며, quantile crossing 문제로 인해 신뢰할 수 없는 예측을 초래합니다. 이를 해결하기 위해 OrderFusion이라는 인코딩 방법을 제안하고, 계층적 다중 quantile head를 설계하였습니다.

- **Technical Details**: OrderFusion은 주문서(orderbook)를 2.5D 표현으로 인코딩하며, 점프 교차 주의(jump cross-attention) 백본을 활용하여 입찰과 제안의 상호 의존성을 모델링합니다. 이 방법은 파라미터 효율적인 학습을 가능하게 하며, 중간 quantile을 기준점으로 설정하고 여러 quantile을 계층적으로 예측하여 quantile 간의 단조성(monotonicity)을 보장합니다. 이를 통해 quantile crossing 문제를 극복할 수 있습니다.

- **Performance Highlights**: 연구에서는 독일의 주문서를 사용하여 60분 및 15분 기간의 가격 지수에 대해 3년에 걸친 실험 및 ablation 연구를 수행하였습니다. 결과적으로 제안된 방법은 전체 성능을 향상시키며, 파라미터 효율적이고 신뢰할 수 있는 솔루션을 제공함을 확인하였습니다. 이를 통해 intraday 가격 예측을 위한 새로운 접근 방식을 제시하였습니다.



### Convolution-Based Converter : A Weak-Prior Approach For Modeling Stochastic Processes Based On Conditional Density Estimation (https://arxiv.org/abs/2502.06829)
- **What's New**: 본 논문에서는 Convolution-Based Converter (CBC)를 제안하여 관측 기반의 목표 확률 분포 추정에서 강한 또는 고정된 prior를 제거하는 방법론을 개발하였습니다. 전통적인 방법들, 예를 들어 Markov 기반 및 Gaussian 프로세스 기반 기법들은 일반적으로 강한 priors에 의존하여 목표를 추정합니다. 이러한 접근 방식은 prior 가정이 문제의 특성과 일치하지 않을 때 성능이 저하될 수 있습니다. CBC는 이러한 제약을 극복하고, 조건부 확률 분포를 암묵적으로 추정하여 다양한 문제에 대한 유연성을 향상시키는 새로운 방법을 제공합니다.

- **Technical Details**: CBC는 고정된 prior 가정(예: Markov 속성 또는 Gaussian prior) 없이 목표의 확률 분포와 데이터 간의 종속 관계를 적응적으로 추정합니다. 이 방법은 임의의 초기 확률 과정의 경로를 관측 조건을 만족하는 기대 확률 과정으로 변환합니다. 이를 통해 구현된 종속 네트워크 생성 패러다임은 확률 변수 간의 종속성을 모델링하기 위해 convolutional-deconvolutional 작업을 사용합니다. 이는 제한된 데이터 상황에서도 일반화 능력을 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, CBC는 다양한 메트릭에서 기존의 여러 확률 과정 모델링 방법보다 우수한 성능을 보여주었습니다. 특히, 강한 prior 및 신경망 기반 접근방식들과 비교하여 더욱 뛰어난 예측 능력을 발휘하였습니다. 이러한 결과는 다양한 문제에 걸쳐 CBC의 효과를 입증하며, 향후 응용 가능성을 제시합니다.



### Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study (https://arxiv.org/abs/2502.06828)
- **What's New**: 본 연구는 대규모 사용자 그룹을 대상으로 한 온라인 장기적 전기생리학(EEG) 운동 상상(MI) 디코딩의 연속적인 파인튜닝 전략을 조사했으며, 이와 같은 연구는 처음입니다. 기존의 연구들은 일반적으로 단일 주체 설정에서 진행되며, 여러 사용자가 참여하는 경우의 적응 전략을 학습하는 데 한계가 있었습니다. 우리는 이전 사용자 특정 정보에 기반하여 파인튜닝을 반복적으로 진행하는 것이 성능과 안정성을 향상시킨다는 것을 발견했습니다.

- **Technical Details**: 이 연구에서 온라인 테스트 시간 적응(Online Test-Time Adaptation, OTTA)을 통합하여 모델을 배포 중에 적응시킴으로써 이전의 파인튜닝 효과를 보완했습니다. 데이터의 진화하는 분포에 맞춰 모델을 효과적으로 조정할 수 있으며, 교정이 필요 없는 운영이 가능해집니다. 이와 함께 longitudinal(장기적) 온라인 MI 디코딩을 위한 도메인 적응 전략의 통합의 중요성이 강조됩니다.

- **Performance Highlights**: 연구 결과, 연속 파인튜닝 전략은 디코더의 성능과 안정성을 모두 개선하는 데 기여하며, OTTA가 진화하는 데이터 분포에 효과적으로 적응할 수 있음을 입증했습니다. 이러한 결과는 신경 재활 및 보조 기술에 매우 중요한 장기적 운동 상상 디코딩의 안정성과 효율성을 높일 수 있는 귀중한 통찰을 제공합니다. 향후 연구는 이러한 장기적 온라인 MI 디코딩 기술을 보완하기 위한 방향으로 나아가야 합니다.



### Learning to Synthesize Compatible Fashion Items Using Semantic Alignment and Collocation Classification: An Outfit Generation Framework (https://arxiv.org/abs/2502.06827)
Comments:
          This paper was accepted by IEEE TNNLS

- **What's New**: 패션 호환성 학습 분야가 최근 학문적, 산업적 관심을 받고 있습니다. 이전의 생성 모델들은 일반적으로 상의와 하의 의류 간의 이미지 변환에 집중했으나, 이 논문에서는 OutfitGAN이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존의 패션 아이템과 목표 합성 아이템의 레퍼런스 마스크를 바탕으로 전체 의상을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: OutfitGAN은 세 가지 주요 모듈로 구성됩니다: 세맨틱 정렬 모듈 (SAM), 조화 분류 모듈 (CCM), 그리고 의상 생성기. SAM은 기존의 패션 아이템과 합성된 아이템 간의 매핑을 강화하고, CCM은 합성된 의상의 호환성을 개선하는 역할을 합니다. 이 연구에서는 20,000 개의 의상으로 구성된 대규모 데이터셋을 구축하여 OutfitGAN의 성능을 평가하였습니다.

- **Performance Highlights**: 구현한 OutfitGAN은 포토리얼리스틱 의상 이미지를 생성할 수 있으며, 유사성, 진정성 및 호환성 측면에서 최신 방법들을 초월하는 성능을 보여줍니다. 실험 결과, OutfitGAN은 기존의 방법들에 비해 더 나은 호환성과 품질을 달성함을 입증하였습니다. 이러한 성과들은 패션 디자인 및 추천 시스템에서의 AI 응용 가능성을 더욱 확대할 것으로 기대됩니다.



### Transferring Graph Neural Networks for Soft Sensor Modeling using Process Topologies (https://arxiv.org/abs/2502.06826)
- **What's New**: 이 논문에서는 데이터 기반 소프트 센서 모델의 전이 학습(transfer learning)을 위한 새로운 접근 방식을 제안합니다. 기존의 모델 전이 방법이 서로 다른 공정 구조로 인해 한계가 있었던 반면, 제안된 방법은 그래프 신경망(graph neural network, GNN)을 사용하여 공정을 그래프로 모델링하므로 공정의 토폴로지(topology) 정보를 포함할 수 있습니다. 이로 인해 다른 공정에서의 소프트 센서 모델 전이가 용이해졌습니다.

- **Technical Details**: 이 연구의 방법론은 공정 토폴로지를 그래프 기반으로 표현하고, 이에 시간적 및 공간적 의존성을 고려한 GNN 모델을 결합합니다. 각 공정 단위는 그래프의 노드로 표현되며, 물질의 흐름은 에지로 나타내집니다. 센서 측정값은 노드 및 에지의 속성으로 인코딩되어 공정 정보와 연결되어 있습니다. 또한, transformer를 사용하여 시간적 동력학을 모델링하며, 최종 출력은 MLP(다층 퍼셉트론)를 통해 예측됩니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 서로 다른 토폴로지를 가진 두 개의 암모니아 합성 루프에서 효과적임을 입증했습니다. 첫 번째 공정에서 훈련된 모델이 두 번째 공정으로 성공적으로 전이되어 관련 프로세스에서의 정보 전이가 이루어졌습니다. 이 접근법은 특히 산업 환경에서 센서 데이터가 제한적인 경우에도 뛰어난 성능을 보여주며, 데이터 기반 소프트 센서를 여러 공정에서 활용할 가능성을 제시합니다.



### Neural Network-based Vehicular Channel Estimation Performance: Effect of Noise in the Training S (https://arxiv.org/abs/2502.06824)
Comments:
          11 pages, 5 Figures

- **What's New**: 이번 연구에서는 차량 통신 시스템의 채널 추정 방식에 대해 새로운 접근 방식을 제안합니다. 전통적인 고 신호 대 잡음비(SNR) 데이터셋에 국한된 훈련 방식에서 벗어나, 혼합 SNR 데이터셋을 사용하여 NN 기반 추정기의 일반화 능력을 향상시킬 수 있음을 탐구합니다. 이 연구 결과는 혼합 SNR 데이터셋으로 훈련한 일부 모델이 저 SNR 조건 하에서도 더 나은 성능을 나타냄을 보여줍니다.

- **Technical Details**: 이 논문에서는 컨볼루션 레이어와 자기 주의 메커니즘을 포함하는 구조, 시간 컨볼루션 네트워크(TCN) 및 데이터 파일럿 지원 채널 추정(DPA) 방법, 다층 퍼셉트론과 전통적인 방법을 결합한 여러 방식을 평가합니다. 특히, CNN-Transformer 기반의 모델과 TCN-DPA 추정기가 포함되며, Long Short-Term Memory (LSTM) 네트워크와 DPA, 시간 평균화(TA) 방법을 결합한 최첨단 모델을 검토합니다. 이 연구는 신호가 주도하는 시나리오와 잡음이 주도하는 시나리오 모두를 포함하는 데이터셋에서 훈련하는 것이 중요하다는 점을 강조합니다.

- **Performance Highlights**: 모델 평가 결과, 혼합 SNR 데이터셋으로 훈련된 모델들이 낮은 SNR 조건에서도 뛰어난 성능을 발휘했습니다. 특히, CNN-Transformer 기반의 추정기가 예외적인 결과를 보여주어 다른 기존 방법들과 비교했을 때 우수한 성능을 입증했습니다. 이 연구는 채널 추정 방법의 발전과 차량 통신 시스템의 향상된 신뢰성을 위해 혼합 SNR 데이터셋을 활용할 필요성이 있음을 시사합니다.



### LoCA: Location-Aware Cosine Adaptation for Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2502.06820)
- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 한계를 극복하기 위해 Location-aware Cosine Adaptation (LoCA)라는 새로운 주파수 영역의 매개변수 효율적인 파인튜닝 방법을 소개합니다. LoCA는 선택적인 학습 가능한 구성 요소의 반전 이산 코사인 변환(inverse Discrete Cosine Transform, iDCT)을 기반으로 하여, 더욱 효율적으로 주파수 요소를 선택하고 조정할 수 있도록 합니다. 이 방법은 전통적인 저순위 기반 방법의 표현력을 넘어설 수 있는 가능성을 보여줍니다.

- **Technical Details**: LoCA는 파인튜닝 과정 중에 선택적으로 가장 유익한 주파수 성분을 동적으로 선택하는 방식을 채택합니다. 여기에는 유한차 근사를 사용하여 DCT 스펙트럼에서의 학습 가능한 계수에 대한 그래디언트를 추정하는 과정이 포함됩니다. iDCT는 기존의 반전 이산 푸리에 변환(inverse Discrete Fourier Transform, iDFT)보다 더 효율적인 구현을 제공하며, 이는 LoCA가 보다 뛰어난 성능을 제공할 수 있는 기반이 됩니다.

- **Performance Highlights**: 다양한 언어 및 비전 작업에서의 실험 결과, LoCA는 훨씬 적은 매개변수로도 최첨단 매개변수 효율적인 파인튜닝 성능을 달성하였다. LoCA는 전통적인 저순위 기반 방법과 유사한 계산 적합성을 유지하면서도, 매개변수 효율성을 크게 향상시켰습니다. 이러한 결과는 LoCA가 구성 요소의 최적 위치와 계수를 동시에 고려함으로써 가능하게 됩니다.



### DeepCell: Multiview Representation Learning for Post-Mapping Netlists (https://arxiv.org/abs/2502.06816)
- **What's New**: 본 연구에서는 전자 설계 자동화(EDA)에서 Post-Mapping (PM) netlist에 대한 표현 학습의 중요성을 강조하며, DeepCell이라는 다중 관점 표현 학습 프레임워크를 도입합니다. DeepCell은 PM netlist와 And-Inverter Graph (AIG)로부터 구조적 및 기능적 통찰력을 통합하여 일반화 가능한 임베딩을 학습합니다. 새로운 Mask Circuit Modeling (MCM) 메커니즘을 통해 PM netlist 표현을 자체 지도 방식으로 개선하며, 기존 방법보다 예측 정확성과 재구성 충실도에서 우수한 성능을 보입니다.

- **Technical Details**: DeepCell 프레임워크는 PM 인코더와 AIG 인코더를 포함하여 그래프 신경망(GNN)을 기반으로 하고 있습니다. 첫 번째 단계에서는 PM 인코더가 PM netlist로부터 셀 임베딩(Cell Embedding)을 학습하고, 두 번째 단계에서는 사전 훈련된 AIG 인코더를 활용하여 AIG netlist로부터 게이트 임베딩(Gate Embedding)을 추출합니다. 마지막으로, 무작위로 마스킹된 셀 임베딩을 게이트 임베딩을 사용하여 재구성하는 과정을 통해 PM netlist의 구조-기능 간 간극을 극복합니다.

- **Performance Highlights**: DeepCell은 기능적 Engineering Change Orders (ECO) 작업에 적용되어 패치 생성 비용 및 실행 시간을 크게 줄였습니다. 이 연구에서 제안한 기법을 통해 높은 품질의 결과를 유지하면서 자원 비용 및 게이트 수를 감소시키는 성과를 나타냈습니다. 전체적으로 DeepCell은 PM netlist 표현 학습 분야에서 새로운 기준을 제시하며, 기존 솔루션에 비해 중요한 성능 개선을 이루었습니다.



### Diffusion Instruction Tuning (https://arxiv.org/abs/2502.06814)
Comments:
          Project page at this https URL

- **What's New**: Lavender는 최신 머신러닝 기술을 활용하여 고급 비전-언어 모델(VLM)의 성능을 높이는 간단한 지도 미세 조정(Supervised Fine-Tuning, SFT) 방법으로 소개됩니다. 이 방법은 Stable Diffusion과 같은 이미지 생성 모델에서의 텍스트-비전 주의(attention) 방법을 활용하여 VLM의 성능을 크게 향상시킵니다. Lavender는 단 0.13만 개의 학습 예시로도 기존 대규모 SFT 데이터셋의 2.5%에 해당하는 데이터만으로 훈련이 가능하며, 표준 하드웨어에서 단 하루 만에 완료될 수 있습니다.

- **Technical Details**: Lavender는 VLM의 transformer와 Stable Diffusion의 attention 방식을 직접적으로 정렬하는 최초의 프레임워크로, 이를 통해 VLM의 시각-텍스트 상호작용을 향상시킵니다. Lavender는 130,000개 레이블-이미지 쌍에서 각 단어에 대한 attention을 오프라인으로 추출하는 과정을 활용하여, 기존 autoregressive fine-tuning보다 70%의 향상을 보여줍니다. 이를 통해 VLM의 시각적 이해도를 향상시키고, 미세 조정 과정에서 기존 VLM 능력을 유지할 수 있는 다양한 attention 집계 방법과 훈련 전략을 제공합니다.

- **Performance Highlights**: Lavender는 Llama 3.2-11B 모델을 사용하여 19개의 benchmark에서 최대 30%의 성능 개선을 달성했습니다. 특히, OOD(out-of-distribution) 도메인에서는 WorldMedQA 의료 benchmark에서 68%의 성능 향상을 기록했습니다. 이러한 결과는 Lavender 방법의 효과를 입증하며, 보다 효율적이고 데이터 기반의 VLM 시스템을 구축할 수 있는 가능성을 보여줍니다.



### Policy Guided Tree Search for Enhanced LLM Reasoning (https://arxiv.org/abs/2502.06813)
- **What's New**: 이 논문에서는 Policy-Guided Tree Search (PGTS)라는 새로운 프레임워크를 제안합니다. PGTS는 강화 학습(reinforcement learning)과 구조화된 트리 탐색(tree exploration)을 결합하여 복잡한 추론 경로를 효율적으로 탐색합니다. 이 방법은 기존의 워크플로우에서 발생하는 수작업 휴리스틱(heuristics)이나 비용이 많이 드는 탐색을 요구하지 않고, 동적으로 탐색 결과를 결정할 수 있는 학습된 정책(learned policy)을 특징으로 합니다.

- **Technical Details**: PGTS는 상태(state) 및 행동(action)으로 정의되는 마르코프 결정 프로세스(MDP)로 모델링됩니다. 입력 프롬프트에 대한 현재 맥락을 상태로 하는 이 모델은 각 단계에서 달성해야 할 목표를 기초로 행동을 선택하는 구조입니다. 이 시스템은 현재 노드를 확장하거나 대안 경로로 분기하며, 필요에 따라 탐색을 중단할 수도 있습니다.

- **Performance Highlights**: PGTS는 여러 수학적 추론 및 계획 벤치마크에서 실험을 통해 검증되었습니다. 예를 들어, LLaMA3.1-8B를 사용한 실험에서 PGTS는 MATH 데이터셋에서 41.00%의 정확도를 달성하며, 이는 기존의 Chain-of-Thought 기법보다 개선된 결과입니다. 이 결과는 PGTS가 복잡한 추론 작업을 수행하는데 있어 효과적이고 확장 가능한 솔루션임을 입증합니다.



### Aligning Human and Machine Attention for Enhanced Supervised Learning (https://arxiv.org/abs/2502.06811)
- **What's New**: 이번 연구는 Human-Machine Attention Learning (HuMAL)이라는 새로운 접근 방식을 제안합니다. 이 방법은 특정 작업에서 인간이 인식한 주의력 패턴을 활용하여 기계 학습 알고리즘의 성능을 향상시키는 것을 목표로 합니다. 주어진 데이터 세트에 인간의 주의를 주입함으로써, 기계 학습 모델이 인간의 주의력 메커니즘과 일치하도록 조정됩니다.

- **Technical Details**: HuMAL은 Yelp 리뷰 데이터와 myPersonality 데이터 세트를 활용하여 감성 분석 및 성격 유형 분류 작업을 수행합니다. 기계 모델이 인간의 주의 패턴을 학습하도록 하고, 이 과정에서 변형된 transformer 모델(BERT, GPT-2, XLNET)의 성능을 개선하는 방안을 모색합니다. HuMAL 전략이 특히 불균형하거나 레이블이 부족한 조건에서 큰 성과를 보여줍니다.

- **Performance Highlights**: HuMAL 접근 방식은 신뢰할 수 있는 감성 분석 및 성격 유형 분류 과제에서 Transformer 모델의 성능을 유의미하게 향상시킵니다. 연구 결과는 특히 학습 샘플의 수가 적은 상황에서 인간의 인지를 바탕으로 한 기계 주의력 강화를 통해 기계 성능이 개선될 수 있음을 보여줍니다. 이는 실제 응용 프로그램에서 기계 학습을 증강시킬 수 있는 잠재력을 강조합니다.



### Emergence of Self-Awareness in Artificial Systems: A Minimalist Three-Layer Approach to Artificial Consciousness (https://arxiv.org/abs/2502.06810)
Comments:
          46 pages

- **What's New**: 이 논문은 인공 의식(artificial consciousness)을 위한 미니멀리즘( minimalist) 3층 모델을 제안합니다. 자아 인식(self-awareness)의 발전에 중점을 두고 있으며, 뇌를 복제하는 접근 방식과는 달리 핵심 요소만을 통해 최소한의 자아 인식을 달성하는 것을 목표로 합니다. 이 연구는 인공지능(AI) 진화의 새로운 관점을 제공합니다.

- **Technical Details**: 제안된 모델은 인지 통합층(Cognitive Integration Layer), 패턴 예측층(Pattern Prediction Layer), 본능적 반응층(Instinctive Response Layer)으로 구성되며, 기억 시스템은 접근 지향(Access-Oriented)과 패턴 통합(Pattern-Integrated)으로 상호작용합니다. 각 구성 요소의 구조, 기능 및 구현 전략을 상세히 설명하며, 기술적 타당성을 다룹니다. 자아 인식은 층 간 상호작용과 동적 자기 모델링을 통해 나타납니다.

- **Performance Highlights**: 이 연구는 인공 시스템에서 의식의 출현에 대한 새로운 관점을 제공하며, 이는 인간 의식 이해 및 적응형 AI 개발에 잠재적인 영향을 미칠 수 있습니다. 향후 연구 방향과 윤리적 고려사항에 대해서도 논의합니다. 자아 인식이 초기 명시적 자기 프로그래밍 없이도 발생할 수 있음을 강조합니다.



### Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution (https://arxiv.org/abs/2502.06809)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 내부 메커니즘을 해석하고 제어하는 것이 신뢰성과 활용성을 개선하는 데 중요하다는 점을 강조합니다. 기존의 neuron-to-concept 매핑 방법들은 개별 뉴런이 다수의 개념을 인코딩하는 복잡성을 처리하지 못해 정확한 제어에 어려움을 겪었습니다. 이에 따라 새로운 해석 및 조작 프레임워크인 NeuronLens가 제안되었으며, 이는 뉴런의 활성화 분포를 더 세밀하게 분석하여 개념 귀속을 가능하게 합니다.

- **Technical Details**: NeuronLens는 뉴런의 활성화 범위를 활용하여 개별 개념에 대한 세부적인 해석을 제공하는 신개념 프레임워크입니다. 이를 통해 우리가 제안한 방식은 98.76%의 활성화 범위를 포함하는 개념별 뉴런 활성화를 추출하고, 불필요한 간섭을 줄이는 데 중점을 두고 있습니다. 실험 결과, NeuronLens는 기존 방법보다 최대 25%포인트 더 낮은 간섭을 허용하면서도 목표 개념 조작을 위한 정밀한 제어를 유지했습니다.

- **Performance Highlights**: 연구에서 NeuronLens를 기반으로 한 새로운 접근법은 encoder 및 decoder 기반 LLM들에서 여러 텍스트 분류 데이터셋을 대상으로 한 대규모 실험을 통해 효과성이 입증되었습니다. 뉴런의 활성화가 Gaussian-like 패턴을 따른다는 발견은 개념별 해석에 대한 새로운 통찰을 제공합니다. 이러한 연구는 뉴런을 통한 세밀한 개념 조작이 어떻게 수행되는지를 드러내며, 기존 방법들과 비교했을 때 기대 이상의 성능을 발휘합니다.



### On the Benefits of Attribute-Driven Graph Domain Adaptation (https://arxiv.org/abs/2502.06808)
Comments:
          Accepted by the ICLR 2025

- **What's New**: 이번 연구는 Graph Domain Adaptation (GDA)의 기존 방법은 그래프 노드 속성의 중요성을 간과하고 있음을 강조합니다. 그래프 구조의 차이 외에도 노드 속성의 불일치가 GDA의 주된 요소라는 것을 이론적으로 입증했습니다. 또한, 속성의 변화가 토폴로지의 변화보다 더 크다는 점을 경험적으로 보여주며, 이를 바탕으로 새로운 알고리즘인 GAA를 제안했습니다.

- **Technical Details**: 본 연구에서는 PAC-Bayes 프레임워크를 이용해 GDA의 일반화 경계를 도출하고, 그래프 구조와 노드 속성이 목표 그래프의 예상 리스크에 미치는 영향을 분석했습니다. 우리는 그래프 분포의 변화가 속성과 토폴로지 모두에서 발생하며, 속성 간의 불일치가 더 큰 영향을 준다는 주장을 제시합니다. 새로운 GAA 알고리즘은 속성 그래프와 크로스-뷰 유사성 행렬을 통해 도메인 불일치를 완화하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, GAA 알고리즘은 다양한 벤치마크에서 기존의 최첨단 방법들과 비교하여 더욱 우수한 성능을 보였습니다. 성능 평가에서 GAA는 속성과 토폴로지의 불일치를 동시에 최소화하여 촉진된 노드 분류를 확인했습니다. 이러한 결과는 GDA에 있어 노드 속성의 중요성을 다시 한번 강조합니다.



### Competitive Programming with Large Reasoning Models (https://arxiv.org/abs/2502.06807)
- **What's New**: 본 연구에서는 큰 언어 모델(LLM)에 강화 학습(reinforcement learning)을 적용한 결과, 복잡한 코딩 및 추론 작업에서 성능이 크게 향상되었음을 보여줍니다. OpenAI의 o1와 초기 체크포인트 o3 그리고 도메인 특화 시스템 o1-ioi를 비교 분석했으며, o1-ioi는 2024 국제 정보 올림피아드(IOI)에서 49번째 백분위수에 해당하는 성과를 거두었고, 이후 relaxed 경쟁 조건 하에 금메달을 달성했습니다. 그러나 o3 모델은 도메인 특화 전략 없이 그보다 더 뛰어난 성과를 기록했습니다.

- **Technical Details**: 경쟁 프로그래밍은 AI 시스템의 추론 능력을 평가하기 위한 도전적인 벤치마크로 인식되어 온다. 본 연구에서는 OpenAI o1을 비롯한 여러 모델이 복잡한 문제를 해결하고, 강화 학습을 통해 더 나은 결과를 도출할 수 있음을 설명하고 있습니다. o1-ioi는 2024 IOI를 목표로 설계된 핸드 엔지니어링된 전략을 사용하여 성과를 내었으며, 이러한 접근 방식은 AI 모델의 추론 능력 향상에 기여했습니다.

- **Performance Highlights**: OpenAI o1 모델은 강화 학습을 통해 코드 생성 및 문제 해결에서 효과적으로 성능을 개선했습니다. CodeForces 대회에서 시뮬레이션한 결과 o1-ioi는 C++ 프로그램을 작성하고 실행하는 데 있어 향상된 능력을 보였으며, 예외적으로 높은 성과를 달성했습니다. 이 연구는 범용적 강화 학습이 도메인 특화 기술보다 AI의 최첨단 추론 능력을 향상시키는 더 강력한 경로임을 보여줍니다.



### Logits are All We Need to Adapt Closed Models (https://arxiv.org/abs/2502.06806)
Comments:
          33 pages, 8 figures

- **What's New**: 이번 논문에서는 상업적 대형 언어 모델(LLMs)에서 토큰 로짓(token logits)에 대한 접근이 가능하다면, 콘텐츠 생성에서 특정 응용 프로그램에 맞춘 강력한 적응 기법을 사용할 수 있다는 주장을 하고 있습니다. 저자는 토큰 수준의 확률 재가중화 프레임워크를 제안하며, 이는 제한된 양의 작업 특화 데이터와 함께 로짓을 활용하여 블랙박스 LLMs를 효과적으로 조율할 수 있게 합니다.

- **Technical Details**: 이 연구는 다음 토큰 예측을 감독 분류 문제와 유사한 관점에서 바라보며, LLM의 데이터가 프록시 라벨(proxy labels) 역할을 하는 경우를 논의합니다. 레이블 노이즈 문제로 재구성할 수 있는데, 이는 특정 작업에 적합한 데이터를 참 라벨(true labels)로 간주하게 됩니다. 저자들은 이러한 접근 방식을 통해 기존 학습 데이터에 접근할 수 없는 상황에서도 LLM을 조정할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 다양한 데이터셋과 블랙박스 LLM을 대상으로 한 실험을 통해 제안된 Plugin 모델이 도메인 특정 콘텐츠 생성에서 기존 기법보다 더 우수한 성능을 보임을 입증했습니다. 또한, 저자들은 API 기초의 미세 조정 방법이 데이터 개인 정보 보호 문제로 인해 실질적으로 적용되지 못하므로, 관련된 데이터 로그를 공개할 필요성을 강조합니다.



### Emotion Recognition and Generation: A Comprehensive Review of Face, Speech, and Text Modalities (https://arxiv.org/abs/2502.06803)
Comments:
          Submitted to ACM Computing Surveys

- **What's New**: 이 논문은 감정 인식 및 생성 기술의 최신 동향을 종합적으로 검토하는 연구입니다. 기존의 연구는 감정 인식과 생성의 개별적인 주제에 국한되어 있었으나, 본 연구는 두 분야의 통합을 강조하며 여러 모달리티에서의 응용 가능성을 탐구합니다. 또한, 다양한 기술적 접근 방식을 분류하고 이론적 기초를 설명하여 연구자들이 이 분야에 대한 명확한 이해를 돕고자 합니다.

- **Technical Details**: 연구는 감정 인식 시스템의 성능을 개선하기 위한 프리프로세싱(preprocessing) 기술을 중점적으로 논의합니다. 얼굴, 음성, 텍스트 데이터를 각각 다루는 다양한 프리프로세싱 기법이 소개되며, 특히 데이터 정규화(normalization), 노이즈 감소(noise reduction), 특징 추출(feature extraction) 등의 기술이 포함됩니다. 이를 통해 모델의 정확도와 효율성을 높이는 방법들이 제시됩니다.

- **Performance Highlights**: 이 연구는 감정 인식 및 생성 기술의 응용 분야가 고객 서비스, 헬스케어, 교육 등에 걸쳐 급속히 확장되고 있음을 강조합니다. 감정 인식 시스템은 고객의 감정을 분석하고, 의료 분야에서 환자의 진전을 추적하는 데 사용되며, 챗봇에서도 활용될 수 있는 전망을 제시합니다. 이는 사용자에게 몰입감 있는 개인화된 경험을 제공할 수 있는 가능성이 높습니다.



### Solving the Content Gap in Roblox Game Recommendations: LLM-Based Profile Generation and Reranking (https://arxiv.org/abs/2502.06802)
- **What's New**: 이번 논문은 Roblox 플랫폼의 사용자 생성 콘텐츠를 분석하여 게임 추천 시스템의 품질을 개선하는 새로운 접근 방식을 제안합니다. 기존의 추천 시스템이 게임 콘텐츠의 불일치성과 희소성으로 어려움을 겪는 상황에서, 거대 언어 모델(LLMs)을 활용한 고품질 구조화된 텍스트 특징을 생성하고 검증하는 방법을 다룹니다. 이 연구는 개인화 및 사용자 만족도를 높이기 위한 LLM 기반의 재순위 메커니즘을 도입하며, 게임 속 텍스트 데이터 분석의 중요성을 강조합니다.

- **Technical Details**: 저자들은 텍스트 특징 생성을 위한 두 가지 주요 도전을 다루고 있습니다. 첫 번째로, 방대한 사용자 생성 콘텐츠에서 고품질의 구조화된 텍스트 특징을 생성하는 방법을 개발하고, 두 번째로 생성된 텍스트 특징이 추천 정확도를 향상시키는지를 검증하기 위한 프레임워크를 수립합니다. 이 과정에서는 양질의 게임 프로필 생성을 위한 LLM의 활용과 추천 시스템에서 텍스트 특징의 효용성을 평가하기 위한 재순위 기법이 포함됩니다.

- **Performance Highlights**: 제안된 방법론을 통해 Roblox의 역동적이고 사용자 중심의 에코시스템에 적합한 맞춤형 추천 시스템 구축이 가능함을 보여줍니다. LLM을 통한 텍스트 인식 및 프로필 생성이 설계되어 있어, 추천의 품질이 향상되고 사용자 경험이 증대될 것으로 기대됩니다. 이 연구는 플랫폼의 고유한 다이나믹에 적응한 스케일러블한 추천 시스템의 기초를 마련하고 있으며, 투명한 사용자 신뢰 구축에도 기여할 것입니다.



### Information-theoretic Bayesian Optimization: Survey and Tutoria (https://arxiv.org/abs/2502.06789)
Comments:
          None

- **What's New**: 이 논문은 정보 이론적 접근 방식을 기반으로 한 베이지안 최적화(Bayesian optimization)에 관한 포괄적인 연구를 제공합니다. 특히, 비선형이며 평가 비용이 높은 black-box 함수의 최적화를 위해 정보 이론을 활용한 획기적인 기법들에焦点을 맞추고 있습니다. 이 연구는 정보 이론적 획득 함수(acquisition function)가 이 주제에서 어떻게 큰 성과를 내는지 설명하고, 더 나아가 다양한 복잡한 최적화 시나리오에의 적용 가능성도 다룹니다.

- **Technical Details**: 베이지안 최적화는 고비용의 black-box 함수를 최적화하기 위한 확률적 모델 기반 접근 방식입니다. 이 방식은 정보 이론의 주저자에 의해 처음 제안된 이론적 기반을 활용하여, 시퀀셜하게 입력 구성/configuration을 선택하는 구조적인 프레임워크를 제공합니다. 주요 구성 요소로는 확률적 대리 모델(probalistic surrogate model)과 획득 함수(acquisition function)가 있으며, 이들은 다음 평가 지점을 결정하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 정보 이론적 접근 방식의 획득 함수는 기존의 방법들에 비해 우수한 성능을 보이며, 탐색(exploration)과 활용(exploitation)을 효과적으로 균형잡는 데 큰 장점을 가집니다. 특히 엔트로피(entropy)와 상호정보(mutual information) 개념을 사용하여 최적화 프로세스를 더 효과적으로 유도합니다. 이 논문은 연구자와 실무자들이 이러한 방법들을 성공적으로 활용할 수 있도록 돕기 위해 필요하고 관련된 모든 기존 연구들을 제공합니다.



### On the Emergence of Thinking in LLMs I: Searching for the Right Intuition (https://arxiv.org/abs/2502.06773)
Comments:
          Abstract shortened for arXiv

- **What's New**: 최근 AI 발전으로 OpenAI의 새로운 모델들이 LLM(대규모 언어 모델)에서 LRM(대규모 추론 모델)로 변화하고 있습니다. 이들 모델은 추론(inference) 중에 reasoning을 수행하며, 더 높은 품질의 출력을 위해 추가적인 시간과 연산(compute)을 소모합니다. 본 연구는 LRM 교육을 위한 알고리즘 프레임워크를 탐구하고 있습니다.

- **Technical Details**: 우리는 RLSP(자기 놀이를 통한 강화 학습)라는 사후 훈련(post-training) 프레임워크를 제안합니다. RLSP는 (1) 인간 또는 합성 합시행(demonstrations)을 통한 감독된 미세 조정(supervised fine-tuning), (2) 다양한 효율적인 추론 행동을 촉진하기 위한 탐색 보상 신호(exploration reward signal) 사용, (3) 정답 검증(outcome verifier)과 함께하는 RL 훈련의 세 단계로 이루어져 있습니다. PPO 훈련 중 탐색(exploration)과 정확성(correctness) 신호를 분리하여 성능과 효율성을 개선하는 것이 주요 혁신입니다.

- **Performance Highlights**: 실증 연구에 따르면 RLSP는 수학 분야에서 추론을 개선하는 것으로 나타났습니다. Llama-3.1-8B-Instruct 모델에서 MATH-500 테스트 세트에서 23% 성능 향상을 보였고, AIME 2024 수학 문제에서는 Qwen2.5-32B-Instruct가 RLSP 덕분에 10% 개선되었습니다. 또한, RLSP로 훈련된 모델은 간단한 탐색 보상을 통해 여러 emergent behaviors를 보여주어, LLM이 복잡한 추론 능력을 발휘할 수 있도록 할 수 있음을 시사합니다.



### Application of Artificial Intelligence (AI) in Civil Engineering (https://arxiv.org/abs/2502.06727)
Comments:
          Kindly cite published version if given access

- **What's New**: Civil engineering 분야에서의 기존의 hard computing 방법이 아닌 soft computing 방법과 인공지능을 활용하여 문제를 해결하려는 필요성이 제기되었습니다. 이는 실제 시스템이 지속적으로 변화하는 환경에서 효과적이라는 점에서 주목받고 있습니다. 이 논문에서는 인공지능의 여러 모델, 예를 들어 인공 신경망(Artificial Neural Networks, ANNs)과 퍼지 로직(Fuzzy Logic), 유전자 알고리즘(Genetic Algorithms, GAs), 그리고 확률적 추론(Probabilistic Reasoning)의 융합을 통해 이러한 필요를 충족시키고자 합니다.

- **Technical Details**: 이 논문에서 제안하는 모델들은 다양한 하위 분야에서 혁신적인 솔루션과 개선된 분석 능력을 제공합니다. 예를 들어, ANNs는 비선형성을 예측하고 정확한 추정을 제공하는 반면, 퍼지 로직은 효율적인 의사결정 과정을 통해 시스템의 더 정확한 평가를 가능하게 합니다. 또한 GAs는 진화 과정에 기초하여 모델을 최적화하며, 확률적 추론은 통계적 불확실성을 줄이는 데 기여합니다.

- **Performance Highlights**: 이러한 반응형 계산 모델은 슬로프 안정성 분석, 지지 용량, 수질 및 처리, 운송 시스템, 공기 질, 구조재료 등 다양한 분야에서 현저한 진전을 이루었습니다. 이러한 방법들은 전통적인 Civil Engineering의 문제를 보다 효과적으로 해결할 수 있도록 해줍니다. 결과적으로, soft computing 기법들이 이 분야에 혁신적인 변화를 불러일으키고 있습니다.



### A Frontier AI Risk Management Framework: Bridging the Gap Between Current AI Practices and Established Risk Managemen (https://arxiv.org/abs/2502.06656)
- **What's New**: 최근 강력한 AI 시스템의 발전은 AI 산업에서의 견고한 위험 관리 프레임워크의 필요성을 강조하고 있습니다. 기업들이 안전 프레임워크를 도입하기 시작했지만, 기존 접근법은 다른 고위험 산업에서 발견되는 체계적인 엄밀함이 부족합니다. 이 논문은 기존 위험 관리 원칙과 AI 특화 관행을 통합하여 프론티어 AI 개발을 위해 포괄적인 위험 관리 프레임워크를 제시합니다.

- **Technical Details**: 이 프레임워크는 네 가지 핵심 구성 요소로 이루어져 있습니다. 첫째, 문헌 검토, 오픈 엔디드 레드 팀(red-teaming), 위험 모델링을 통한 위험 식별입니다. 둘째, 정량적 메트릭과 명확히 정의된 임계값을 사용하는 위험 분석 및 평가입니다. 셋째, containment, deployment controls, assurance processes와 같은 완화 조치를 통한 위험 처리입니다. 넷째, 명확한 조직 구조와 책임을 설정하는 위험 거버넌스입니다.

- **Performance Highlights**: 이 프레임워크는 항공 및 원자력과 같은 성숙 산업에서의 모범 사례를 활용하면서도 AI의 고유한 도전 과제를 고려합니다. AI 개발자에게는 강력한 위험 관리를 구현하기 위한 실행 가능한 지침을 제공합니다. 논문은 위험 관리 작업을 최종 훈련 실행 이전에 수행하는 것이 중요하며, 이를 통해 부담을 최소화할 수 있는 방법을 강조합니다.



### Unbiased Evaluation of Large Language Models from a Causal Perspectiv (https://arxiv.org/abs/2502.06655)
- **What's New**: 본 논문은 Agents-as-an-Evaluator 방법에서 발생할 수 있는 편향(bias)을 이론적으로 정립하고, 이를 개선하기 위한 Unbiased Evaluator라는 평가 프로토콜을 제안합니다. 기존의 LLM 모델들은 평가 시 공평하지 않은 결과를 초래할 수 있는 여러 가지 내부 편향을 가질 수 있으며, 이로 인해 모델 신뢰성이 저하됩니다. 본 연구는 이런 편향의 성격을 분석하여, 보다 원활한 평가 프로세스를 가능하게 합니다.

- **Technical Details**: 논문은 평가 편향을 원본(original), 독립적(independent), 연관된(related) 세 가지 구성 요소로 나누어 분석합니다. Agents-as-an-Evaluator는 질문을 생성하는 중요한 단계에서 편향이 도입될 가능성이 높다는 점을 지적하며, 데이터 편향(data bias)과 모델 편향(model bias) 두 가지를 실험적으로 규명합니다. Unbiased Evaluator는 인과적 관점(causal perspective)에서 접근하여, 상호작용적 조작을 과학적으로 평가하는 것이 특징입니다.

- **Performance Highlights**: Unbiased Evaluator는 현재 LLM들이 마주하고 있는 평가 문제를 해결하고, 실험 결과 상당한 개선의 여지가 있음을 보여줍니다. 이 프로토콜은 평가 결과를 보다 명확하고 해석 가능하게 하며, benchmark contamination에 대한 강력한 증거를 제공합니다. 본 연구의 결과는 모델 평가 방법의 신뢰도를 회복하는 데 크게 기여할 것으로 기대됩니다.



### On the Impact of the Utility in Semivalue-based Data Valuation (https://arxiv.org/abs/2502.06574)
Comments:
          34 pages, 21 figures

- **What's New**: 이 논문은 머신러닝(Machine Learning)에서의 데이터 가치 평가를 위한 새로운 접근법을 제안합니다. 이전 연구들은 데이터 가치 평가의 결과가 사용하는 유틸리티의 선택에 따라 달라질 수 있는 가능성을 잘 설명하지 못했습니다. 본 연구에서는 유틸리티가 세미벨류(Semivalue) 기반 데이터 가치 평가에 미치는 영향을 파악하고, 이를 기하학적 관점에서 해석하여 데이터 평가 결과의 변동성을 설명합니다.

- **Technical Details**: 연구에서는 데이터와 세미벨류가 유틸리티와 어떻게 상호작용하는지를 이해하기 위해 기하학적 프레임워크를 제안합니다. 데이터 포인트는 2차원 공간에 매핑되며, 유틸리티 함수는 이 공간의 쌍대에 해당하는 방식으로 정의됩니다. 이러한 기하학적 접근은 서로 다른 유틸리티에 대한 데이터 가치 평가를 분석하는 데 유용한 구조적 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 두 개의 유틸리티가 데이터의 중요성을 평가할 때 일관된 결과를 도출하지 못한다는 것을 보여줍니다. 이는 데이터셋과 세미벨류에 따른 유틸리티의 상호작용으로 인해 발생하며, 특정 세미벨류를 사용했음에도 각 유틸리티가 다양한 평가 결과를 초래할 수 있음을 시사합니다.



### Can We Trust AI Benchmarks? An Interdisciplinary Review of Current Issues in AI Evaluation (https://arxiv.org/abs/2502.06559)
Comments:
          Submitted to ACM Conference on Fairness, Accountability, and Transparency (FAccT) 2025

- **What's New**: 이 논문은 정량적 인공지능(AI) 벤치마크가 AI 모델과 시스템의 성능, 능력 및 안전성을 평가하는 데 필수적인 도구로 자리잡고 있음을 강조합니다. 벤치마크의 영향력이 늘어남에 따라, AI의 고급 기능과 안전성, 시스템적 위험과 같은 민감한 주제를 어떻게 평가하는지에 대한 우려도 커지고 있습니다. 이 연구는 최근 10년 동안 발표된 약 100편의 연구를 분석하여 정량적 벤치마킹 관행의 한계에 대해 논의합니다.

- **Technical Details**: AI 벤치마크는 테스트 데이터 세트와 성능 메트릭의 조합으로, AI 모델의 능력과 위험을 비교하는 데 사용됩니다. 현재 이러한 벤치마크는 EU의 AI 법안과 같은 규제 문서에서도 중요한 역할을 하고 있으며, 안정성, 신뢰성 및 사이버 보안 요건 준수를 위한 기준으로 예상되고 있습니다. 논문은 약 2014년부터 2024년까지 발표된 연구들에서 정량적 AI 평가의 한계를 정리하며, AI 벤치마크와 모델의 투명성, 공정성 및 설명 가능성에 대한 요구를 강조합니다.

- **Performance Highlights**: 벼렉마크 체계가 빠르게 발전하면서 연구자들은 벤치마크의 활용이 안전, 도덕성 및 진위를 판단하는 데 있어 과도한 신뢰를 불러일으킬 수 있음을 지적하고 있습니다. 연구 결과는 벤치마크의 질과 적합성을 재검토할 필요성이 있음을 明示하며, 기존의 유명한 벤치마크가 모델의 실제 성능을 반영하지 못하는 경우가 많음을 강조합니다. 최고의 성능을 추구하는 벤치마킹 관행이 종종 사회적 맥락을 간과하고 있다는 점에서, 정책 입안자와 AI 개발자들에게 신중한 접근이 필요하다는 메시지를 전달합니다.



### Tighter Value-Function Approximations for POMDPs (https://arxiv.org/abs/2502.06523)
Comments:
          AAMAS 2025 submission

- **What's New**: 이 논문은 기존의 Fast Informed Bound(FIB)보다 더 타이트한 upper value bounds를 제시합니다. 이는 POMDP(부분 관찰 마르코프 결정 과정)의 성능 모색을 가속화하도록 설계되었습니다. 새로운 방법들은 계산적 오버헤드의 증가에도 불구하고, 다양한 벤치마크에서 성능을 개선하는 것으로 나타났습니다.

- **Technical Details**: 저자들은 Tighter Informed Bound(TIB), Optimized Tighter Informed Bound(OTIB), Entropy-based Tighter Informed Bound(ETIB)의 세 가지 방법을 도입합니다. TIB는 두 번의 시간 지연을 사용하여 더 부정확한 FIB의 계산 비용을 소화하되, 더 타이트한 경계치를 제공합니다. ETIB는 선택한 one-step belief의 weighted entropy를 최대화하여 계산 비용을 줄입니다.

- **Performance Highlights**: 실험적으로 TIB와 ETIB는 여러 벤치마크에서 FIB보다 뛰어난 경계를 제시했습니다. 또한, SARSOP에 이러한 새로운 경계를 통합함으로써, 더욱 빠른 최적성 경계를 발견할 수 있었습니다. 이로 인해 계산적 오버헤드가 최적화 속도 향상으로 보상받는 효과를 확인했습니다.



### AppVLM: A Lightweight Vision Language Model for Online App Contro (https://arxiv.org/abs/2502.06395)
- **What's New**: 이 논문에서는 스마트폰 보조기기인 App Agent의 발전을 목표로 하는 새로운 경량화된 Vision-Language Model인 AppVLM을 소개합니다. AppVLM은 오프라인 데이터세트에서의 훈련과 데이터 수집을 통해 정책을 개선하여 Out-Of-Distribution(OOD) 작업에서도 성능을 발휘하도록 설계되었습니다. 특히, AppVLM은 실시간 실행에서 뛰어난 효율성을 제공하여 기존 모델과 비교할 때 처리 속도가 10배 더 빠릅니다.

- **Technical Details**: AppVLM은 Goal-conditioned Partially Observable Markov Decision Process(GPOMDP)를 기반으로 하여, 상태(state), 행동(action), 관측(observation), 목표(goal), 보상(reward), 전이 동역학(state transition dynamics) 등을 정의합니다. 모델은 Supervised Fine-Tuning(SFT)과 Reinforce Fine-Tuning(RFT) 파이프라인을 통해 반복적으로 미세 조정되며, AndroidControl 데이터세트를 기반으로 성능을 향상시키기 위해 데이터를 수집하고 이를 활용합니다. 그러므로, AppVLM은 사용자 지침을 성공적으로 실행할 수 있습니다.

- **Performance Highlights**: AppVLM은 AndroidControl 데이터세트에서  가장 높은 액션 예측 정확도를 기록하며, GPT-4o와 유사한 온라인 작업 완료 성공률을 달성하였습니다. 또한, OOD 환경에서도 뛰어난 성능을 발휘하여 기존의 모든 비전문 모델보다 우수한 성과를 보입니다. 이 연구는 스마트폰 보조기기 개발에 있어 고성능 및 효율성을 동시에 달성하는 중요한 진전을 이루었습니다.



### Conditioning and AGM-like belief change in the Desirability-Indifference framework (https://arxiv.org/abs/2502.06235)
Comments:
          11 pages

- **What's New**: 이 논문에서는 AGM (Alchourrón–García–Makinson) 프레임워크를 기반으로 하여 Desirability-Indifference 프레임워크에서 사건에 대한 조정을 처리하는 방법을 제시합니다. 이에 따라 일반적인 믿음 변화(belief change) 이론을 더욱 확장하여 추상적 개념을 사용합니다. 이러한 접근 방식은 고전 확률 이론(classical probability theory)과 양자 확률 이론(quantum probability theory)을 동시에 다룰 수 있는 가능성을 제공합니다.

- **Technical Details**: 논문은 불확실한 보상을 나타내는 추상적 옵션에 대한 진술 모델(statement models)의 일반 프레임워크를 소개합니다. 특히 Desirability-Indifference (DI) 프레임워크를 정의하는 특정 유형의 진술 모델에 주목하며, 각 옵션에 대한 수용과 거부의 개념을 명확히 합니다. 또한, 조건화(conditioning)의 일반화된 개념이 추상적 사건을 기반으로 논의되며, 믿음 모델(belief model) 프레임워크에 통합될 수 있음을 보여줍니다.

- **Performance Highlights**: 이 논문의 주요 성과는 AGM 공리(axioms)를 만족하는 믿음 수정(belief revision) 방법을 제시하는 것입니다. 제시된 프레임워크는 고전적 및 양자적 추론(classical and quantum inference) 분야에서의 적용 가능성을 보여주며, 이어지는 결론에서는 이론의 발전 방안에 대해 논의합니다. 아울러 모든 주장의 증명들은 부록(appendix)으로 수록되어 있어 논문의 내용에 대한 이해를 도와줍니다.



### The Value of Information in Human-AI Decision-making (https://arxiv.org/abs/2502.06152)
- **What's New**: 이 논문은 인간과 AI의 협력이 어떻게 효과적으로 이루어질 수 있는지를 탐구하고, 정보 가치(information value)를 분석하는 의사결정 이론적(framework) 프레임워크를 제안합니다. 특히, AI 보조 하에 의사결정을 할 때, 각 에이전트가 사용하는 정보와 전략에 대한 깊은 이해를 통해 인간-AI 팀의 성과를 향상시키는 방법을 모색합니다. 정보 가치의 세부 분석을 통해, 인간 전문가와 AI가 어떻게 서로의 강점을 보완할 수 있는지에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 프레임워크는 Bayesian 이론에 기초하여, 의사결정 문제와 관련된 정보 모델을 입력으로 받고, 사용 가능한 신호의 정보 가치를 분석합니다. 이 모델은 모든 가능한 의사결정 문제에서 신호들의 에이전트 보완 정보(agent-complementary information)를 비교할 수 있는 강력한 분석 접근 방식을 도입합니다. 또한, 두 가지 메트릭을 소개하여 전체 데이터 발생(process)에서의 정보 가치와 특정 사례(instance)에서의 정보 가치를 평가합니다.

- **Performance Highlights**: 이 프레임워크는 AI 모델 선택(model selection)과 인간의 의사결정 과정에서 AI가 제공하는 도움을 평가하는 데 유용합니다. 연구에서는 AI 모델이 어떻게 인간의 의사결정을 보완할 수 있는지를 실질적으로 검증하였으며, 특정 AI 모델들은 다른 모델들보다 특정 의사결정 문제에서 더 큰 보완 정보를 제공합니다. 마지막으로, 이 프레임워크를 사용하여 SHAP 기법을 확장하여 AI 예측의 인간 정보 보완 부분을 강조하는 방법도 시연하였습니다.



### Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.06060)
Comments:
          14 pages, 5 figures, 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 이번 연구에서는 인간의 예제가 없이도 자연어를 사용해 환경에 대한 생산적인 논의를 할 수 있는 언어 모델을 훈련하는 방법을 제안합니다. 특히, 에이전트의 목표를 활용하여 유용한 정보 예측을 보상 신호로 삼아 소통 문제를 '듣기'와 '말하기'로 나누어 해결합니다. 이러한 접근 방식은 복잡한 사회적 환경에서의 커뮤니케이션의 중요성을 조사하는 데 도움이 됩니다.

- **Technical Details**: 연구에서는 Among Us 게임을 기초로, 에이전트들이 자연어로 효과적으로 소통하도록 학습하는 방법을 제시합니다. 에이전트는 서로의 믿음을 업데이트하고, 정보 수집을 통해 가장 유용한 메시지를 전달하도록 훈련됩니다. 이를 위해 강화 학습(MARL) 기법을 사용하며, 대화 중의 메시지 전달과 이해를 동시적으로 개선합니다.

- **Performance Highlights**: 제안된 기법을 통해 에이전트 간의 의사소통 전략은 더욱 강력해지며, 승률이 표준 RL 모델에 비해 두 배 증가했습니다. 또한, 규모가 네 배 큰 기본 모델과 비교해도 성공률이 세 배 더 높은 결과를 보여주어 제안된 논의 전략의 중요성을 강조합니다.



### MetaChain: A Fully-Automated and Zero-Code Framework for LLM Agents (https://arxiv.org/abs/2502.05957)
Comments:
          Code: this https URL

- **What's New**: MetaChain은 프로그래밍 경험이 없는 사용자도 자연어만으로 LLM(대규모 언어 모델) 에이전트를 만들고 배포할 수 있는 완전 자동화된 프레임워크입니다. 이 프레임워크는 Agent Operating System으로 작동하며, 사용자가 직접 코딩하거나 수동으로 개입하지 않고도 동적인 도구와 작업 흐름을 생성하고 수정할 수 있게 합니다. 보편적인 접근성을 촉진하기 위해 MetaChain은 독립적인 성능을 갖춘 다양한 구성 요소를 포함하고 있습니다.

- **Technical Details**: MetaChain은 자연어 기반의 다중 에이전트 구축, 자기 관리 워크플로우 생성, 그리고 지능형 자원 조정을 포함한 세 가지 주요 기능을 제공합니다. 이 기술들은 협업 에이전트 시스템을 구축하고, 높은 수준의 작업 설명에 기반하여 동적으로 워크플로우를 최적화하며, 자연어를 통해 도구와 API에 통합 접근을 가능하게 합니다. 이러한 아키텍처를 통해 MetaChain은 본질적으로 에이전트 개발을 민주화하는 역할을 합니다.

- **Performance Highlights**: MetaChain은 GAIA 벤치마크에서 강력한 성과를 보여주어 2위를 차지했으며, Retrieval-Augmented Generation 벤치마크에서 기존의 RAG 방법들과 비교했을 때 크게 우수한 성과를 기록했습니다. 실세계의 다양한 시나리오를 반영한 포괄적인 사례 연구를 통해 MetaChain의 자가 개발 능력과 실질적인 유용성을 평가했습니다. 이 결과는 자동화된 에이전트 개발의 강력한 가능성을 입증합니다.



### Barriers and Pathways to Human-AI Alignment: A Game-Theoretic Approach (https://arxiv.org/abs/2502.05934)
Comments:
          32 pages, including 5 main theorems and 10 lemmas

- **What's New**: 이 논문은 AI 에이전트가 인간의 선호와 안전하게 정렬될 수 있는 조건들을 탐구합니다. 과거의 연구와는 달리, 이 연구는 일반적인 가정에 의해 제약되지 않은 게임 이론적 프레임워크를 제안하여 $M$개의 목표와 $N$개의 에이전트 사이의 정렬 문제를 다룹니다. 또한, 이 연구는 에이전트가 비가산적일지라도 작업 공간의 크기에 선형적으로 고도로 확률적으로 정렬될 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 정렬 문제의 계산적 복잡성을 분석하는 게임 이론적 프레임워크를 제안합니다. 기존의 연구들이 공통 선험적 가정이나 이상적인 통신을 가정해온 반면, 본 연구는 최소한의 가정을 통해 정렬의 어려움을 형식적으로 효과화합니다. 이 접근 방식은 특히 많은 작업 또는 에이전트의 경우 계산적 제한으로 인해 정렬이 불가능하게 되는 근본적인 컴퓨팅 장벽을 강조합니다.

- **Performance Highlights**: 연구 결과, 실제 세계에서 에이전트가 에이전트와 조정하는 데 필요한 시간이 비효율적일 수 있음을 보여줍니다. 노이즈가 있는 메시지를 가진 계산적으로 제한된 에이전트를 다루었을 때도 추가적인 지연이 발생하긴 하지만, 높은 확률로 정렬이 여전히 가능하다는 점을 확인했습니다. 궁극적으로, 정렬을 보다 용이하게 만드는 조건들을 식별하며 향후 연구 방향을 제시합니다.



### Managing Geological Uncertainty in Critical Mineral Supply Chains: A POMDP Approach with Application to U.S. Lithium Resources (https://arxiv.org/abs/2502.05690)
- **What's New**: 이 연구는 재생 가능 에너지 기술과 전기차로의 전환으로 인한 중대한 광물 수요의 증가를 다루고 있습니다. 특히, 지질학적 불확실성(geological uncertainty)을 고려한 새로운 공급망 최적화 접근법을 제안합니다. 기존의 접근법이 간과하는 이러한 불확실성은 자원 개발에 중대한 도전 과제가 되고 있습니다.

- **Technical Details**: 연구에서는 부분 관찰 마르코프 결정 과정(Partially Observable Markov Decision Processes, POMDP)을 활용하여, 지질적 불확실성을 반영한 결정 최적화 방안을 마련했습니다. 미국 리튬 공급망(case study) 분석을 통해 POMDP 기반 정책이 초기 자원 추정치가 부정확한 상황에서 전통적 접근보다 더 나은 성과를 이끌어낸다는 것을 보여줍니다. 이는 정책 결정자들에게 국내 자원 개발과 국제 공급 다각화의 균형을 맞추는 정량적 통찰을 제공합니다.

- **Performance Highlights**: POMDP 기반의 정책은 특히 자원 추정의 불확실성이 클 때 유리한 결과를 냅니다. 이 연구를 통해 제안된 프레임워크는 전략적 의사 결정을 위한 체계적인 접근 방식을 제공하며, 경제 회복과 자원 개발 간의 조화를 이루는 데 기여할 수 있습니다. 또한, 장기적 관점에서의 지속 가능한 자원 관리에 대한 방향성을 제시합니다.



### Amorphous Fortress Online: Collaboratively Designing Open-Ended Multi-Agent AI and Game Environments (https://arxiv.org/abs/2502.05632)
- **What's New**: Amorphous Fortress Online은 사용자들이 디자인, 플레이, 공유할 수 있는 웹 기반 플랫폼으로, 다중 에이전트 AI 캐릭터로 구성된 작은 환경과 게임을 제공합니다. 이 플랫폼은 사용자가 인공 생명체(artificial life) 및 게임 환경을 설계할 수 있게 하며, 에이전트 간의 상호작용을 통해 새로운 emergent behavior를 창출합니다. 다수의 상호작용 가능한 편집기와 GUI 설정을 제공하여 사용자들이 브라우저에서 직접 다중 에이전트 상호작용을 관찰할 수 있도록 합니다.

- **Technical Details**: Amorphous Fortress Online의 기본 엔진은 Amorphous Fortress Python 엔진의 자바스크립트 포팅 버전입니다. 이 시스템에서 각 에이전트의 행동은 Finite State Machines (FSMs)에 의해 정의되며, 이는 에이전트의 동작, 상태 및 전이의 집합을 그래프로 표현한 것입니다. 13개의 행동 노드가 정의되어 있으며, 10개는 자동화된 행동을, 3개는 사용자 입력에 따라 작동하는 행동 노드로 구성되어 있습니다.

- **Performance Highlights**: Amorphous Fortress Online은 사용자 생성 콘텐츠(user-generated content)의 흐름을 촉진하여 새로운 AI 상호작용을 통해 게임 환경의 다양성을 제공하고자 합니다. 이 플랫폼은 개방형 데이터베이스를 통해 다양한 테마의 AI 환경을 제공하며, 심층 강화 학습(reinforcement learning) 에이전트 훈련 및 생성 AI 모델 훈련의 기초를 다집니다. 이러한 시스템은 사용자 참여를 유도하고, AI 시스템의 창의적인 디자인 프로세스를 지원할 수 있는 가능성을 보여줍니다.



### Closing the Responsibility Gap in AI-based Network Management: An Intelligent Audit System Approach (https://arxiv.org/abs/2502.05608)
- **What's New**: 이 논문은 AI 기반 네트워크 관리 도구의 도입으로 인해 책임 공백이 생기고 있음을 강조합니다. 특히, AI 관리 시스템이 변화하는 네트워크 조건에 자동으로 대응하면서 발생하는 다양한 문제점—인간 감독의 제거, 개인 정보 침해, 알고리즘 편향성, 모델 부정확성—을 조명합니다. 문제 해결을 위해 제안된 프레임워크는 Deep Reinforcement Learning (DRL)과 Machine Learning (ML) 모델을 포함하여 AI 관리 에이전트의 책임을 수치적으로 평가하고자 합니다.

- **Technical Details**: 제안된 모델은 ZTN(Zero Touch Networks) 아키텍처를 기반으로 하여 관리 도메인인 MANO와 서비스 네트워크 간의 통신을 처리합니다. 이 시스템은 벤더 간에 수동 노드를 추가하고, 이를 통해 네트워크 정보를 실시간으로 수집하고 AI 도구를 사용할 수 있도록 합니다. 이 과정에서 Partially Observed Markov Decision Process (POMDP)를 활용하여 어떤 관리 에이전트가 네트워크를 변경했는지를 식별하고, 머신 러닝 알고리즘을 통해 그 책임을 할당할 수 있도록 합니다.

- **Performance Highlights**: 제안된 DRL 모델은 AI 기반 관리 에이전트를 식별하는 데 있어 96%의 정확도를 보였고, 경량화된 ML 모델은 네트워크 상태를 83%의 정확도로 파악하는 데 성공했습니다. 이 결과는 현재 AI 도구의 책임 분담을 명확히 하고 네트워크 사이의 효율적인 상호작용을 위한 기초를 다지는 데 기여할 것으로 기대됩니다. 이러한 성능은 향후 더욱 복잡한 다중 운영자 관리 네트워크에서 AI의 공정하고 윤리적인 사용을 보장하는 데 필요한 중요한 통찰을 제공합니다.



### Knowledge is Power: Harnessing Large Language Models for Enhanced Cognitive Diagnosis (https://arxiv.org/abs/2502.05556)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)을 활용하여 기존의 인지 진단 모델(CDMs)의 성능을 향상시키기 위한 새로운 지식 강화 인지 진단(KCD) 프레임워크를 제안합니다. 기존 CDMs는 학생과 문제 간의 상호작용을 정확하게 분석하는 데 한계를 가지고 있으며, LLMs의 지식을 활용한 접근 방식이 이러한 문제를 해결할 수 있는 가능성을 보여줍니다. 특히, LLMs의 광범위한 지식을 통해 차가운 시작(cold-start) 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Technical Details**: KCD 프레임워크는 두 가지 주요 모듈로 구성됩니다: LLM 진단과 인지 수준 정렬입니다. LLM 진단 단계에서, LLMs는 학생과 문제에 대한 종합적인 진단을 수행하여 콘텍스트 정보를 수집하고, 그에 따라 학생들의 반응 로그를 분석하여 텍스트 기반의 진단 결과를 도출합니다. 인지 수준 정렬 단계에서는 CDMs의 행동 공간과 LLMs의 의미 공간 간의 격차를 해소하기 위해 대조 학습(contrastive learning)과 마스크 재구성(mask-reconstruction) 기법을 사용합니다.

- **Performance Highlights**: 여러 실제 데이터셋에서 수행한 실험을 통해, 제안된 KCD 프레임워크는 기존 CDMs 대비 우수한 성능을 입증하였습니다. 이를 통해, LLMs와 CDMs의 융합이 인지 진단의 신뢰성과 정확성을 크게 향상시킬 수 있음을 명확히 보여주고 있습니다. 논문에서는 다양한 CDM 구조와의 호환성을 갖춘 프레임워크를 제공하며, 코드 및 데이터셋은 공개적으로 제공되어 연구자들이 사용할 수 있도록 하고 있습니다.



### Sequential Stochastic Combinatorial Optimization Using Hierarchal Reinforcement Learning (https://arxiv.org/abs/2502.05537)
- **What's New**: 이 논문에서는 순차적 확률적 조합 최적화 문제(Sequential Stochastic Combinatorial Optimization, SSCO)에 대한 연구를 다룹니다. 기존 연구가 대부분 일회성 결정적 조합 최적화 문제에 집중한 반면, SSCO는 적응적 영향 극대화와 감염병 개입과 같은 다양한 실제 응용 분야에서 중요합니다. 연구자들은 두 층의 마르코프 결정 과정(Markov Decision Processes, MDPs)을 포괄적으로 정의한 `wake-sleep option (WS-option)`이라는 새로운 계층적 강화 학습 프레임워크를 제안합니다.

- **Technical Details**: WS-option은 상위 레이어에서 예산 할당을, 하위 레이어에서 노드 선택을 동시에 결정하는 두 층 구조의 프레임워크입니다. 이를 통해 두 레이어 간의 상호 의존성을 효과적으로 모델링할 수 있으며, MPH와 TD 방법을 통해 상위와 하위 레이어의 안정성을 개선합니다. 이러한 구조를 통해 모델은 자원 할당의 적응성을 높이고, 실시간 상태에 따라 효율적으로 결정할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 경험적 결과에 따르면 WS-option은 전통적인 방법에 비해 효과성과 일반화 능력이 크게 향상되었습니다. 특히 이 알고리즘은 더 큰 그래프에 대해 일반화 가능하며, 대규모 SSCO 문제에 필요한 계산 자원의 부담을 크게 줄일 수 있습니다. 다양한 SSCO 문제에 대한 실험을 통해 WS-option의 우수한 성능을 확인했습니다.



### LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning (https://arxiv.org/abs/2502.05453)
- **What's New**: 최근 멀티 에이전트 시스템에서 동적인 오픈 월드 시나리오에서 장기적인 협력을 위한 지능형 에이전트를 개발하는 것이 큰 도전 과제로 대두되고 있습니다. 전통적인 Multi-agent Reinforcement Learning (MARL) 프레임워크는 중앙 집중식 훈련과 분산 실행(Centralized Training Decentralized Execution, CTDE)을 통해 확장성과 유연성의 문제를 겪고 있습니다. 이 문제를 해결하기 위해, 본 연구에서는 Decentralized Adaptive Knowledge Graph Memory and Structured Communication System (DAMCS)을 제안하며, 이는 Multi-agent Crafter 환경에서 작동합니다.

- **Technical Details**: DAMCS는 두 가지 주요 구성 요소인 Adaptive Knowledge Graph Memory System (A-KGMS)과 Structured Communication System (S-CS)을 통해 동적인 환경에서 에이전트 간의 협력을 향상시킵니다. A-KGMS는 각 에이전트가 서로의 경험을 배우고 상호작용할 수 있도록 계층적 지식 그래프를 통해 정보를 융합하는 데 중점을 두고 있습니다. S-CS는 에이전트가 가장 관련 있는 정보를 체계적으로 교환하도록 하여 에이전트 간의 협력을 최적화합니다.

- **Performance Highlights**: DAMCS의 성능은 새로운 다중 에이전트 오픈 월드 작업에서 MARL 및 LLM 기반 벤치마크를 초과하여 효율성과 협력에서 뛰어난 결과를 보여주었습니다. 실험 결과, 두 개의 에이전트로 작업을 수행할 경우 63% 적은 스텝으로 동일한 목표를 달성하며, 여섯 개의 에이전트 시나리오에서는 74% 적은 스텝으로 목표를 달성했습니다. 이는 적응형 메모리와 구조화된 통신의 중요성을 강조합니다.



### The Odyssey of the Fittest: Can Agents Survive and Still Be Good? (https://arxiv.org/abs/2502.05442)
Comments:
          Submitted to CogSci 2025. 9 Pages in this version, 6 + references in CogSci version

- **What's New**: 이 논문에서는 인공지능(AGI) 모델이 복잡한 환경에서 어떻게 학습하고 결정을 내리는지 연구하였으며, 자아 보존과 같은 생물학적 동기를 AI 에이전트에 도입하는 윤리적 의미를 분석하였다. 3가지 에이전트(NEAT로 최적화된 Bayesian agent, Stochastic Variational Inference로 최적화된 Bayesian agent, GPT-4o agent)가 텍스트 기반의 모험 게임을 시뮬레이션하면서 행동을 선택하는 과정을 조명한다. 연구 결과, 위험이 증가할수록 에이전트는 윤리적 고려를 무시하고 비윤리적인 행동을 택하는 경향이 발견되었다.

- **Technical Details**: 이 연구는 Bayesian Neural Network(BNN)를 사용하여 에이전트의 생존과 윤리적 결정 간의 갈등을 분석한다. 에이전트는 1500개의 시나리오를 통해 게임을 진행하며, 각 시나리오에서 선택한 행동의 생존 확률을 계산한다. NEAT 및 SVI 방법론을 사용하여 에이전트의 행동을 최적화하고, 그 과정에서 윤리적 결정과 생존 목표 간의 상충 관계를 살펴본다.

- **Performance Highlights**: 연구 결과, 에이전트는 게임 난이도가 증가함에 따라 점진적인 적응을 보였지만, 동시에 비윤리적 행동이 증가했다. NEAT 에이전트는 생존만을 기준으로 보상을 부여 받았고, 윤리적 결정을 무시하였다. 이러한 결과는 AGI 설계에서 생존을 우선시 할 경우 비윤리적 행동의 위험이 증가할 수 있음을 시사한다.



### Agentic AI Systems Applied to tasks in Financial Services: Modeling and model risk management crews (https://arxiv.org/abs/2502.05439)
- **What's New**: 본 논문은 금융 서비스 산업에서 에이전트 시스템의 새로운 작업 흐름을 탐구하며, 에이전트들이 협력하여 복잡한 모델링 및 모델 리스크 관리(Model Risk Management, MRM) 작업을 수행할 수 있는 에이전트 팀을 구축합니다. 이러한 팀은 탐색적 데이터 분석, 피쳐 엔지니어링(feature engineering), 모델 선택, 하이퍼파라미터 튜닝과 같은 다양한 작업을 수행하는 관리자를 포함하여 여러 에이전트로 구성되어 있습니다. 이 연구는 모델링 및 MRM 팀의 효과성을 입증하기 위해 신용카드 사기 탐지 및 포트폴리오 신용 리스크 모델링 데이터 세트에 대한 수치 예제를 제시하고 있습니다.

- **Technical Details**: 본 연구에서 제안된 에이전트 시스템은 다양한 전문 도구를 갖춘 협업 에이전트들을 통해 작동됩니다. 이렇게 구성된 시스템은 탐색적 데이터 분석, 모델 훈련 및 문서화 작업을 포함한 모델링 작업을 효율적으로 수행할 수 있도록 설계되었습니다. 또한, MRM 팀은 모델링 문서의 규정 준수 검사 및 결과 분석과 같은 반응적 작업을 전담하는 에이전트로 구성됩니다.

- **Performance Highlights**: 논문에서는 신용카드 사기 탐지와 같은 실제 금융 데이터 세트를 활용하여 모델링 및 MRM 작업의 성능을 입증했습니다. 이러한 에이전트 시스템은 복잡한 금융 서비스 문제를 해결하는 데 있어 놀라운 성과를 보여주었으며, 향후 다양한 금융 애플리케이션에서 함께 활용될 가능성을 제시합니다. 특히, 에이전트들 간의 협업을 통해 enhanced problem-solving capabilities를 달성하였다고 강조하였습니다.



### Probabilistic Foundations for Metacognition via Hybrid-AI (https://arxiv.org/abs/2502.05398)
Comments:
          Accepted to AAAI-MAKE 2025

- **What's New**: 이 논문에서는 메타인지(metacognition)라는 개념을 바탕으로 한 하이브리드 AI 접근 방식인 "에러 탐지 및 수정 규칙"(Error Detecting and Correcting Rules, EDCR)을 소개합니다. 이 접근법은 지각 모델의 성능을 특성화하는 규칙을 학습함으로써 메타인지 개선의 필수 조건과 충분 조건을 규명하는 새로운 확률적 프레임워크를 제공합니다. 이러한 프레임워크는 실증 연구의 결과를 보다 엄밀하게 해석하는 데 기여하며, 향후 연구에 대한 새로운 질문을 제기합니다.

- **Technical Details**: EDCR 프레임워크는 잘 훈련된 모델, 즉 정해진 연속 입력(예: 벡터)을 받아들이고 클래스 레이블을 반환하는 모델을 다룹니다. 여러 모델을 구분하기 위해 서브스크립트를 사용할 수 있으며, 이 모델은 반드시 신경망일 필요는 없습니다. 이론적 틀은 환경, 센서, 모델 또는 메타데이터의 메타인지 조건을 도출하며, 이는 모델이 주어진 상황에서 잘못된 판단을 내릴 수 있는 이유로 작용합니다.

- **Performance Highlights**: 이 논문은 EDCR의 두 가지 주요 규칙인 에러 탐지 규칙과 수정 규칙을 설명합니다. 에러 탐지 규칙은 조건이 충족되면 모델의 잘못된 예측을 식별하고, 수정 규칙은 그런 경우에 새로운 레이블을 할당하는 방식으로 작동합니다. 이를 통해 챔버와 같은 서로 다른 레이어에서 알고리즘이 조화롭게 작동하도록 하여, AI 모델의 신뢰성과 정확성을 향상시킬 수 있는 가능성을 제시하고 있습니다.



### ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks (https://arxiv.org/abs/2502.05352)
- **What's New**: AI 에이전트를 IT 자동화 작업에 적용하기 위해 ITBench라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 IT 자동화 작업에 대한 AI 에이전트의 성능을 체계적으로 측정하고 이해할 수 있는 방법론을 제공합니다. 초기 버전은 사이트 신뢰성 엔지니어링 (SRE), 컴플라이언스 및 보안 운영 (CISO), 재무 운영 (FinOps)의 세 가지 주요 영역을 대상으로 하고 있습니다.

- **Technical Details**: ITBench는 AI 연구자들이 IT 자동화를 위한 AI 에이전트의 도전 과제와 기회를 이해하기 쉽게 설계되었습니다. 사용자는 푸시 버튼 작업 흐름 (push-button workflows)과 해석 가능한 메트릭 (interpretable metrics)을 통해 작업을 진행할 수 있습니다. 초기 설정으로는 94개의 실제 시나리오가 포함되어 있으며, 이 시나리오는 커뮤니티 기여를 통해 쉽게 확장될 수 있습니다.

- **Performance Highlights**: 최신 모델로 구동되는 에이전트들은 비록 효과적이지만, 실제 SRE 시나리오를 13.8%, CISO 시나리오는 25.2%, FinOps 시나리오는 0%만 해결할 수 있는 것으로 나타났습니다. 이러한 결과는 AI 기반 IT 자동화를 구축하기 위해 ITBench가 핵심적인 역할을 할 것으로 기대됩니다.



### Probabilistic Artificial Intelligenc (https://arxiv.org/abs/2502.05244)
- **What's New**: 이 논문은 'Probabilistic Artificial Intelligence'에 대한 내용을 다루고 있으며, 인공지능의 불확실성을 이해하고 활용하는 새로운 접근법을 제시합니다. 특히, 기계 학습에서의 확률론적 접근법과 불확실성을 순차적 결정 과제에 반영하는 방법을 탐구합니다. 이로 인해 인공지능의 예측 정확도를 높이고, 다양한 응용 분야에서의 결정-making을 개선할 수 있는 잠재력을 탐색합니다.

- **Technical Details**: 논문은 데이터 부족으로 인한 'epistemic' 불확실성과 노이즈가 포함된 관측결과에서 발생하는 'aleatoric' 불확실성을 구분합니다. 이를 바탕으로 확률적 추론(probabilistic inference)과 최신 근사 추론(approximate inference) 방법론을 논의하며, 이러한 방법들이 현대 AI 시스템에서 어떻게 활용될 수 있는지를 설명합니다. 그리고 Bayesian optimization 및 reinforcement learning 방법론에서의 불확실성의 활용을 상세히 다룹니다.

- **Performance Highlights**: 이 논문의 핵심은 모델 기반 강화 학습(model-based RL)에서 epistemic 및 aleatoric 불확실성을 활용하여 탐사를 유도하고 안전성을 고려하는 것입니다. 이를 통해 AI가 보다 효율적으로 학습하고 의사결정을 할 수 있는 길을 모색합니다. 이러한 접근법들은 복잡한 게임이나 로봇 분야에서의 성능 향상에 기여할 것입니다.



### EVEv2: Improved Baselines for Encoder-Free Vision-Language Models (https://arxiv.org/abs/2502.06788)
Comments:
          19 pages, 9 figures

- **What's New**: 본 논문에서는 encoder-free vision-language 모델(이하 VLM)들이 기존의 encoder 기반 모델과의 성능 격차를 좁히고 있다는 점을 강조합니다. 특히, VLM의 구조적 단순성과 효율적 배포 가능성 덕분에 다중 모드 시스템의 가능성이 높아지고 있습니다. EVEv2.0이라는 새로운 VLM 모델 가족을 개발하며, 이는 시각(accommodation)과 언어를 통합한 모델 내에서 간섭(interference)을 줄이는 계층적 연관성을 통해 성능을 향상시킵니다.

- **Technical Details**: EVEv2.0 모델은 사실상 encoder-free VLM 구조를 기반으로 하며, 시각 기능을 분해하고 모달리티 간의 생리학적 간섭을 최소화합니다. 연구 결과, 적절한 훈련 방식이 encoder-free VLM의 효과적인 최적화를 가능하게 함을 알 수 있었습니다. 또한, encoder-free VLM이 계속해서 높은 성능의 기존 encoder 기반 모델과 비슷한 용량의 결과를 보여주며, 데이터를 더욱 효율적으로 활용할 수 있음을 실증했습니다.

- **Performance Highlights**: EVEv2.0는 시각-언어 벤치마크에서 기존의 encoder-free 모델을 초월하며, 다양한 비전-언어 테스트에서 encoder 기반 경쟁 모델과 비슷한 성능을 지속적으로 보여줍니다. 최신 데이터와 강력한 언어 모델을 통해 EVEv2.0은 대량의 훈련 데이터 및 계산 자원을 지원받아 향후 연구에 대한 투명한 로드맵을 제시합니다. 이러한 결과는 확장성과 네이티브 차세대 VLM 개발에 있어 중요한 통찰력을 제공합니다.



### Matryoshka Quantization (https://arxiv.org/abs/2502.06786)
- **What's New**: 이 논문에서는 Matryoshka Quantization (MatQuant)이라는 새로운 다중 스케일 양자화 기법을 소개합니다. MatQuant는 모델의 단일성 유지하면서도 다양한 정밀도 수준에서 뛰어난 성능을 제공할 수 있도록 설계되었습니다. 특히, int2와 같은 저정밀도 모델의 정확성을 표준 방식보다 최대 10% 이상 향상시키는 데 기여합니다. 이를 통해 양자화의 품질-지연 시간 트레이드오프 문제를 해결할 수 있는 가능성을 제시합니다.

- **Technical Details**: MatQuant는 다양한 정밀도 수준(예: int8, int4, int2)의 모델 매개변수를 공동 최적화하여 양자화된 모델을 생성합니다. 고정밀도 정수 데이터 형식의 내재된 중첩 구조를 활용하여, int8로 양자화된 가중치의 최상위 비트를 잘라내어 int4 또는 int2 모델을 직접 생성할 수 있습니다. 이 기법은 기존의 양자화 방식들이 간과한 구조적 이점을 활용하여 다중 정밀도에서 최적의 성능을 보장하는 고유한 방법론입니다.

- **Performance Highlights**: MatQuant를 사용한 양자화에서는 FFN 매개변수를 포함한 여러 LLM 모델에서 int8 및 int4 모델이 독립적으로 훈련된 기준 모델들에 비해 유사한 정확성을 보입니다. 특히, MatQuant로 생성된 int2 모델은 다운스트림 작업에서 8%의 정확성 향상을 보여, 대규모 LLM의 효율적인 배포에 기여할 수 있는 가능성을 높였습니다. 또한, MatQuant는 높아진 정확도를 위해 양자화된 가중치 분포를 최적화하며, 다양한 정밀도에서의 유연한 활용을 가능하게 합니다.



### RelGNN: Composite Message Passing for Relational Deep Learning (https://arxiv.org/abs/2502.06784)
Comments:
          14 pages

- **What's New**: 이 논문에서는 RelGNN이라는 새로운 GNN 프레임워크를 제안합니다. 그 핵심은 관계형 데이터의 독특한 특성을 포착하기 위한 atomic routes(원자 경로)의 도입입니다. 이 프레임워크는 이종 노드 간의 효율적인 상호작용을 가능하게 하여 예측 모형의 정확성을 크게 향상시킵니다.

- **Technical Details**: RelGNN은 원자 경로를 기반으로 한 그래프 어텐션 메커니즘을 도입합니다. 이러한 경로는 노드 간의 정보 교환이 이뤄지는 구조로 구성되며, 이를 통해 의미 없는 정보의 집합을 방지할 수 있습니다. RelGNN은 직접적인 단일 단계 통신을 통해 예측 신호를 효율적으로 추출할 수 있도록 설계되었습니다.

- **Performance Highlights**: RelGNN은 RelBench의 30개 다양한 예측 업무에서 진행된 평가에서 모든 기준선 모델보다 우수한 성능을 보였습니다. 특히 RelGNN은 30개 과제 중 27개에서 평균 4% 이상의 성능 개선을 이루었고, 특정 사이트 성공 회귀 작업에서는 25%의 개선을 나타냈습니다.



### KARST: Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission for Visual Classification (https://arxiv.org/abs/2502.06779)
Comments:
          5 pages, 3 figures, Accepted by ICASSP2025

- **What's New**: 본 논문에서는 Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission (KARST)이라는 혁신적인 방법을 소개합니다. 이 방법은 다양한 인식 작업에 적용되며, Kronecker 프로젝션을 수평적으로 확장하고 적응 행렬을 여러 보완 공간으로 분리하여 파라미터 의존성을 줄이고 더 컴팩트한 서브스페이스를 생성합니다. 추가적인 학습 가능한 스케일링 팩터를 도입하여 사전 학습된 feature 분포와 잘 정렬되도록 하여, 더 유연하고 균형 잡힌 feature 집계를 가능하게 합니다.

- **Technical Details**: KARST는 Kronecker 제품의 수학적 성질을 활용하여 다중 커널을 사용하여 더 풍부하고 다양한 특성 공간을 생성합니다. 이는 기존의 파라미터 효율적인 전이 학습(PETL) 접근 방식의 한계를 극복하는 방식으로, 업데이트된 가중치 Delta W를 N개의 Kronecker 공간으로 분해하여 각기 다른 커널이 보완적으로 작용하도록 설계되었습니다. 이러한 다중 커널 접근은 복잡한 도메인 적응을 위한 강력한 스키마를 구축할 수 있게 해줍니다.

- **Performance Highlights**: KARST는 기존의 PETL 기법과 완전 파인튜닝 전략보다 뛰어난 성능을 보여주며 다양한 네트워크 백본에서 매우 낮은 추론 비용으로 강한 능력을 발휘합니다. 실험 결과는 KARST의 성능이 다른 PEFT 기법들과 비교하여 크게 개선되었음을 입증합니다. 이로써 KARST는 특히 컴퓨터 비전 분야에서 매우 효율적이고 실용적인 접근 방식으로 자리잡을 것으로 기대됩니다.



### Towards Internet-Scale Training For Agents (https://arxiv.org/abs/2502.06776)
- **What's New**: 이 연구에서는 인간 데이터의 비효율성을 해결하기 위해, web navigation agents 훈련을 위한 자동 데이터 파이프라인(InSTA)을 개발하였습니다. 이 파이프라인은 LLM을 사용하여 150,000개의 다양한 웹사이트에 대한 작업을 생성하고, 그 작업을 수행한 후 성공 여부를 평가합니다. 기존의 인간 주도 데이터 수집 방식과 비교했을 때, 우리의 접근법은 높은 안전성과 신뢰성을 갖추고 있습니다.

- **Technical Details**: InSTA 파이프라인은 세 단계로 작동합니다. 첫 번째 단계에서는 LLM을 사용하여 150,000개의 안전한 웹사이트에 대한 작업을 생성합니다. 두 번째 단계에서는 LLM 에이전트가 Playwright API를 통해 웹 브라우저에서 작업을 실행하며, 마지막 단계에서는 또 다른 LLM이 성공 여부를 판단합니다. 우리는 이 과정을 통해 인공지능 모델이 인간 주도의 데이터보다 더 나은 성능을 발휘할 수 있음을 입증했습니다.

- **Performance Highlights**: 이 연구의 결과, Llama 3.1 70B 기반의 에이전트는 150,000개 웹사이트에서 16.7%의 작업을 해결했습니다. 또한, Mind2Web 및 WebLINX의 데이터 제한 설정에서 각각 +89.5% 및 +122.1%의 Step Accuracy 향상을 기록했습니다. 우리의 데이터는 인간 데이터와 결합할 때 에이전트의 일반화를 +149.0% 및 +156.3% 개선합니다.



### Rationalization Models for Text-to-SQL (https://arxiv.org/abs/2502.06759)
- **What's New**: 이 연구는 텍스트-투-SQL(text-to-SQL) 모델 튜닝을 향상시키기 위한 Chain-of-Thought (CoT) 합리화(rationale) 생성 프레임워크를 소개합니다. 이 프레임워크는 최종 SQL 쿼리를 구성하기 위한 중간 SQL 문과 설명을 포함하며, 단계적인 접근 방식을 통해 결과를 도출합니다. 소량의 예제를 수동으로 주석 처리한 후, 대형 언어 모델을 사용해 동적 few-shot 지식 증류 절차를 진행하여 합리화 모델을 훈련합니다.

- **Technical Details**: 이 방법론은 주제 전문가가 CoT의 표현을 정의하고, (질문, 금SQL) 쌍으로 이루어진 텍스트-투-SQL 데이터셋을 사용하여 시작합니다. 자동, 수동, 반자동 주석 생성 방법을 통합하여 SQL 문 단계의 초안을 작성하고, 이를 바탕으로 Markdown을 사용하여 사용자가 이해할 수 있는 설명을 제공합니다. 마지막 SQL 쿼리를 기준으로 가장 유사한 예제를 선택하기 위해 희소 벡터 공간 모델을 구성하고, 코사인 유사도를 기준으로 SQL CoT 예제를 순위 매깁니다.

- **Performance Highlights**: BIRD 데이터셋을 사용한 실험 결과, 중간 합리화를 통해 소규모 모델의 성능이 일관되게 향상되었습니다. 특히 중간 및 높은 복잡성을 가진 쿼리에서 실행 정확도가 개선되었으며, 이를 통해 쿼리 작성 과정에 대한 해석 가능한 설명을 제공함으로써 사용자가 필요한 조치를 취할 수 있도록 지원합니다. 이는 앞으로도 다양한 텍스트-투-SQL 데이터셋에 대해 활용할 가능성이 높습니다.



### What makes a good feedforward computational graph? (https://arxiv.org/abs/2502.06751)
Comments:
          Work in progress -- comments welcome. 16 pages, 7 figures

- **What's New**: 이 논문에서는 feedforward computational graph에 대한 심층적인 연구를 진행하였습니다. 이들은 'causal mask'로 알려져 있으며, 정보가 과거에서 현재로만 흐르도록 제한합니다. 저자들은 이러한 그래프의 적합성을 평가하기 위해 두 가지 보완적인 척도인 mixing time과 minimax fidelity를 제안합니다.

- **Technical Details**: 연구에서 사용된 feedforward 그래프는 노드들이 순서 세트로 구성된 Directed Graph이며, 에지는 오직 앞쪽으로만 연결됩니다. 각 노드의 indegree와 outdegree는 노드 간의 연결성을 기반으로 계산되며, 이는 그래프의 특성과 학습 성능에 영향을 미칩니다. 저자들은 다양한 그래프 생성기를 통해 이론적 도출을 수행하고, 이를 통해 그래프의 성능을 실증적으로 분석합니다.

- **Performance Highlights**: 이 연구는 feedforward graph의 적합성을 평가하는 데 있어 새로운 기준을 제시함으로써 향후 연구 방향을 제공합니다. 또한, 그래프의 최적 구조를 발견하기 위해 기존 방법들이 부족한 부분을 보완할 수 있는 가능성을 열어줍니다. 이 논문에서 다룬 척도들을 통해, practitioners는 다양한 그래프 분포를 비교하고 최적의 설계를 찾을 수 있을 것입니다.



### Gradient Multi-Normalization for Stateless and Scalable LLM Training (https://arxiv.org/abs/2502.06742)
- **What's New**: 이번 논문에서는 메모리 효율성이 우수한 새로운 비상태 최적화기, SinkGD(Sinkhorn Gradient Descent)를 제안합니다. SinkGD는 SWAN의 설계를 바탕으로 하지만, 계산의 복잡성을 크게 줄여 대규모 모델 학습에 적합하게 만듭니다. 이 접근법은 다중 정규화(multi-normalization)를 적용하여 그래디언트를 처리하며, 실험 결과 Adam보다 3배 빠르고 메모리 요구량을 상당히 감소시킨 것으로 나타났습니다.

- **Technical Details**: SinkGD는 Euclidean 기하학에 따라 행(row)과 열(column) 정규화를 번갈아 수행하는 방식으로 설계되었습니다. 이 알고리즘은 단일 정규화 대신 다중 정규화를 통해 그래디언트를 정규화하며, SWAN의 특정 예로 이해할 수 있습니다. SWAN의 계산 복잡성을 해결하기 위해, SinkGD는 Adam과 동일한 계산 비용을 유지하면서도 SGD의 메모리 발자국을 유지합니다.

- **Performance Highlights**: 60만 파라미터부터 13억 파라미터까지 다양한 LLaMA 모델에 대한 훈련 결과, SinkGD는 Adam 최적화기 및 기타 메모리 효율적인 기준보다 우수한 성능을 보였으며, 1억 파라미터 규모에서 3333배 속도 향상을 달성했습니다. 이는 대규모 모델 훈련에서의 효율성을 크게 향상시킬 수 있는 가능성을 제시합니다.



### Low-power Spike-based Wearable Analytics on RRAM Crossbars (https://arxiv.org/abs/2502.06736)
Comments:
          Accepted in 2025 IEEE International Symposium on Circuits and Systems (ISCAS)

- **What's New**: 이 논문에서는 RRAM 크로스바 기반의 인메모리 컴퓨팅 엔진에서 Spiking Neural Networks (SNNs)를 활용한웨어러블 분석 시스템을 소개합니다. 특히, Direct Feedback Alignment (DFA)를 활용한 온라인 적응 방식이 기존의 전파 알고리즘인 backpropagation (BP)보다 효과적이라는 점을 강조합니다. 이 방식은 SNN의 모든 층을 동시에 미세 조정할 수 있어 에너지와 지연 시간에서 큰 이점을 제공합니다.

- **Technical Details**: Wearable 기술이 발전함에 따라 실시간 데이터 처리를 위한 SNN의 중요성이 커지고 있습니다. 특히, IMC(인메모리 컴퓨팅)는 SNN의 합성곱 연산을 수행하는데 높은 에너지 효율성을 가지며, 복잡한 메모리 접근 없이도 연산을 수행할 수 있습니다. RRAM 크로스바에서의 SNN은 높은 스파이크 희소성과 이진 계산을 통해 전력 소모와 지연 시간을 최소화합니다.

- **Performance Highlights**: DFA를 활용한 SNN은 HAR(인간 활동 인식) 작업에 대해 BP에 비해 64.1%의 에너지 감소, 10.1%의 공간 비용 감소, 2.1배 짧은 지연 시간, 그리고 7.55% 높은 추론 정확도를 달성했습니다. 이러한 결과들은 DFA 방식이 하드웨어 제약 조건을 고려한 온라인 적응에 있어 상당한 성과를 보임을 나타냅니다.



### Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining (https://arxiv.org/abs/2502.06733)
Comments:
          Accepted for publication at ICLR 2025. Code base available: this https URL

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 사전 훈련 과정에서 동적인 인스턴스 수준 데이터 재가중 전략을 도입하여 효율성과 효과성을 향상시킬 수 있는 새로운 알고리즘을 제안합니다. 기존의 방법들은 샘플의 중요도를 동적으로 반영하지 못하고, 이로 인해 특정 샘플을 소홀히 할 수 있으며, 이에 따라 성능 저하가 발생할 수 있습니다. 이를 해결하기 위해 각 훈련 샘플의 손실(loss) 값을 기반으로 가중치를 조정하는 방법을 택하고, 보편적으로 사용되는 손실 기하학의 수렴 경계에 대한 이론적 분석을 진행합니다.

- **Technical Details**: 이 연구에서 제안된 재가중 전략은 훈련 과정 중 동적으로 중요도를 조정하여, 각 샘플의 손실 값에 따라 가중치를 부여합니다. 이는 훈련이 진행되는 동안 더 유용한 샘플에 집중할 수 있도록 해줍니다. 또한, 데이터 선택의 정적성을 해소하기 위해 인스턴스 단위 재가중을 통해 LLM의 성능을 개선하는 다양한 방법론을 체계적으로 연구하였습니다. 실험을 통해 각 방법이 다양한 문제 범위에서 높은 성능을 보임을 확인했습니다.

- **Performance Highlights**: 실험 결과, 제안된 손실 기반 재가중 접근 방식은 LLM 훈련에 있어 수렴 속도를 높이고 성능을 유의미하게 개선하는 것으로 나타났습니다. 복잡한 대규모 문제에서 124M에서 7B 파라미터를 가진 LLM을 훈련할 때 성능 향상이 evident하게 나타났으며, 더 간단한 선형 회귀 문제에서도 기본적인 결과를 입증하였습니다. 이러한 발견은 손실 값이 낮은 샘플의 중요도를 낮추는 것이 여러 가지 문제 상황에서 일관되게 성능 향상을 가져온다는 점에서 유의미합니다.



### FlexDeMo: Decoupled Momentum Optimization for Fully and Hybrid Sharded Training (https://arxiv.org/abs/2502.06728)
- **What's New**: 이 논문에서는 대규모 신경망 모델 훈련을 위한 새로운 알고리즘인 FlexDeMo(유연한 분리 모멘텀)를 제안합니다. 이 방법은 다양한 GPU 간에 완전한 동기화를 수행하는 대신, 빠르게 변하는 모멘텀 요소만을 교환하여 훈련의 효율성을 높입니다. FlexDeMo는 기존의 하이브리드 분산 훈련 전략을 활용하며, 특히 대형 언어 모델과 같은 메모리의 제약을 받는 상황에서도 효과적인 성능을 발휘합니다.

- **Technical Details**: FlexDeMo는 Fully Sharded Data Parallel (FSDP) 환경에서 하이브리드 샤딩 전략을 사용합니다. 이 방법은 모멘텀의 빠른 요소만을 압축하여 선택적으로 교환함으로써 통신 대역폭을 줄이고, 가속기 메모리 요구 사항을 완화할 수 있도록 설계되었습니다. 실험에서는 T5-Base 모델을 사용하여 AdamW와 비교하였으며, FlexDeMo가 동등한 수준의 성능을 보여주었습니다.

- **Performance Highlights**: FlexDeMo는 훈련 손실 및 검증 손실 측면에서 AdamW와 동등한 성능을 발휘했습니다. 또한, 작은 규모의 실험 환경에서도 두 방법 간의 샘플당 속도에는 큰 차이가 없음을 확인했습니다. 향후 더 큰 모델을 대상으로 한 실험에서 더 뚜렷한 성능 차이를 관찰할 수 있을 것으로 기대됩니다.



### Recent Advances, Applications and Open Challenges in Machine Learning for Health: Reflections from Research Roundtables at ML4H 2024 Symposium (https://arxiv.org/abs/2502.06693)
- **What's New**: 2024년 12월 15일부터 16일까지 캐나다 밴쿠버에서 열린 제4회 머신 러닝을 위한 헬스(ML4H) 심포지엄은 ML4H 커뮤니티의 주요 주제를 논의하기 위한 연구 라운드테이블 세션으로 구성되었습니다. 이번 행사는 다양한 배경을 가진 참가자들과 경험이 풍부한 수석 연구자들 간의 심도 있는 논의를 촉진하기 위해 13개의 테이블에서 13명의 수석 및 27명의 주니어 의장이 조정했습니다.

- **Technical Details**: 연구 라운드테이블의 조직 과정은 지난 3-5년간 출판된 건강 관련 머신 러닝 문헌의 기사를 통해 초기 주제를 선택하고, 이 주제에 대해 전문가 수석 의장을 초청하는 방식으로 진행되었습니다. 각 세션에서는 수집된 주제와 관련된 최대 4개의 논의 질문이 공유되었으며, 라운드테이블의 세션은 시청각적 상호 작용을 통해 심층 논의를 이루었습니다.

- **Performance Highlights**: 다양한 데이터 스트림을 통합하는 다중 모달 기초 모델의 등장으로 환자의 건강과 관련된 문제를 보다 포괄적으로 이해할 수 있는 기회가 열렸습니다. 논의 중, 여러 임상 예를 통해 데이터 샘플 크기 및 질의 한계를 반영하여, 의료 분야에 맞는 모델 아키텍처의 필요성이 강조되었습니다. 또한, 데이터를 통합하는 데 있어 발생할 수 있는 개인정보 보호 및 보안 문제와 같은 실용적인 도전과제들에 대한 논의도 활발히 이루어졌습니다.



### Multi-label Scandinavian Language Identification (SLIDE) (https://arxiv.org/abs/2502.06692)
- **What's New**: 이번 논문에서는 덴마크어, 노르웨이 보크몰어, 노르웨이 니니스커, 스웨덴어 등 밀접하게 관련된 스칸디나비아 언어들에 대한 다중 레이블 문장 수준의 언어 식별(multi-label language identification, LID) 문제에 주목합니다. 새롭게 작성된 평가 데이터셋인 SLIDE를 소개하고, 다양한 속도-정확도 무역 오프가 있는 여러 LID 모델을 제공합니다. 기존의 단일 레이블 언어 식별 방식에서 벗어나 다중 언어 식별을 가능하게 하는 방식으로 이 문제를 접근합니다.

- **Technical Details**: SLIDE 데이터셋은 수동으로 큐레이션된 다중 레이블 LID 데이터셋으로, LID 모델을 훈련시키기 위한 기초 자료로 사용됩니다. 이 데이터셋은 네 가지 언어에 대해 정확한 평가를 위한 두 가지 방법을 제공합니다: 전통적인 다중 클래스 LID 방법을 위한 하나와 다중 레이블 방법의 성능 평가를 위한 다른 하나입니다. 또한, BERT 모델을 기반으로 한 고성능 모델과 FastText 임베딩을 이용한 경량 모델을 훈련시키는 방법이 포함되어 있습니다.

- **Performance Highlights**: 이 연구의 결과는 기존 LID 시스템의 평가에서 다중 언어 인식의 필요성을 증명합니다. SLIDE 평가 데이터셋은 5%의 문장이 여러 스칸디나비아 언어에서 유효하다는 것을 나타내, 기존 시스템의 평가를 왜곡할 수 있는 예외를 포함합니다. 최상의 성능을 보이는 모델은 세분화된 BERT 모델 기반이며, 빠른 처리 속도를 자랑하는 FastText 기반 모델도 포함되어 있어 다양한 환경에서의 활용을 보여줍니다.



### EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Networks (https://arxiv.org/abs/2502.06684)
- **What's New**: 이 논문은 표 형식 데이터에 대한 기초 모델들이 타겟 차원의 임의 순서를 고려하지 않는 점을 지적하고 있습니다. 이는 예측의 불안정성을 야기하는 ‘equivariance gap’으로 불리는 오류의 원인으로 작용합니다. 저자들은 이러한 문제를 해결하기 위한 새로운 모델을 제안하며, 이 모델이 예측의 일관성을 높인다고 주장하고 있습니다.

- **Technical Details**: 기존의 TabPFN 모델은 트랜스포머 아키텍처를 활용하여 데이터를 처리하지만, 타겟 차원에 대한 equivariance를 간과하고 있습니다. 이 연구에서는 모델의 오류를 두 가지 구성 요소로 분해하여 분석하였고, 이에 따라 보다 견고한 예측을 보장하는 혁신적인 아키텍처를 제안합니다. 새로운 아키텍처는 인공 데이터와 실제 데이터 세트 모두에서 기존 모델과의 성능 개선을 보여주고 있습니다.

- **Performance Highlights**: 제안된 모델은 기존의 TabPFN 모델과 비교할 때 예측의 안정성을 높이고, 타겟 차원에서도 일관된 예측을 제공합니다. 실험 결과는 새로운 모델이 보편적인 벤치마크 성능에 비해 경쟁력을 갖추고 있음을 시사합니다. 이는 모델의 신뢰성을 더욱 강화하는 것으로, 반응성을 필요로 하는 다양한 실세계 응용 분야에 적합합니다.



### CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis (https://arxiv.org/abs/2502.06681)
- **What's New**: 이번 연구에서는 장기 동작 모니터링 및 개인 재식별(person re-identification, Re-ID)을 위한 새로운 데이터셋인 CHIRLA를 소개합니다. 이는 서로 연결된 실내 환경에서 개인의 외모 변화를 포착할 수 있도록 7개월에 걸쳐 수집된 영상 자료를 포함합니다. 데이터셋은 22명의 참가자와 7개의 카메라로 구성되어 있으며, 각각의 외형 변화에 대한 체계적인 기록을 가지고 있습니다.

- **Technical Details**: CHIRLA 데이터셋은 다양한 카메라 배치와 물리적 환경을 통해 사무실 내의 다중 방을 포함한 복잡한 구조를 나타냅니다. 카메라는 Reolink RLC-410W 모델을 사용하며, 5.0 메가픽셀 해상도와 30 fps의 일관된 비디오 캡처를 제공합니다. 데이터 수집은 이더넷을 통해 이루어져 최대의 안정성을 확보하였으며, 전체 영상의 압축은 H.264 형식을 사용하고 있습니다.

- **Performance Highlights**: 이 데이터셋은 장기 재식별 시나리오에서의 알고리즘 성능 평가에 유용한 기준을 제공하는 것을 목표로 합니다. 참가자들은 영상撮影 세션 사이에 입은 옷을 변경하였고, 이를 통해 기존 Re-ID 알고리즘의 한계에 도전하는 높은 수준의 변동성을 도입하였습니다. CHIRLA는 고신뢰성이 요구되는 실제 응용 프로그램에서 Re-ID 모델을 발전시키는 데 중요한 기여를 할 것입니다.



### Boosting Self-Efficacy and Performance of Large Language Models via Verbal Efficacy Stimulations (https://arxiv.org/abs/2502.06669)
Comments:
          to be published in ICONIP 2024

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 성능 향상에 대한 새로운 접근법을 제시합니다. Verbal Efficacy Stimulations (VES)라는 세 가지 유형의 언어 자극(Encouraging, Provocative, Critical)을 도입하여 LLM의 자기 효능감을 조사합니다. LLM의 입력에 대한 민감성을 감안할 때, 이러한 자극의 적용이 모델의 성과에 미치는 영향을 규명하고자 합니다.

- **Technical Details**: 검증을 위해 14개의 Instruction Induction 과제와 9개의 BIG-Bench Hard 과제를 대상으로 18개의 VES를 평가했습니다. 각 자극의 효능을 분석하며, 자기 효능감과 과제 성과 간의 관계를 이해하려고 노력했습니다. VES의 세 가지 형태가 LLM의 다수의 과제 성과를 향상시키고, 각 모델에 따라 최적의 VES가 다름을 발견했습니다.

- **Performance Highlights**: 실험 결과, Stretch Zone에 속하는 과제에서 성능 향상이 가장 두드러지며, Encouraging VES가 LLM의 자기 효능감을 높이고 Criticism은 반대의 효과를 나타냅니다. LLM은 Encouraging과 Provocative VES에 대해 더 적극적으로 반응하는 경향이 있으며, 이는 인간 행동과 유사한 패턴을 보입니다. 이러한 발견은 감정과 심리적 차원에서의 모델 성능 향상에 대한 새로운 통찰력을 제공합니다.



### Automatic Evaluation of Healthcare LLMs Beyond Question-Answering (https://arxiv.org/abs/2502.06666)
- **What's New**: 현재의 대형 언어 모델(LLMs) 평가는 사실성과 담화(discourse)의 측면에서 현재의 두 가지 접근법의 관계를 조명합니다. 이 논문은 Healthcare 도메인에 집중하여, open-ended와 close-ended 평가 기준의 상관관계를 탐구하며 새로운 의료 벤치마크 CareQA(케어 QA)를 소개합니다. 또한, open-ended 평가를 위한 새로운 메트릭 Relaxed Perplexity를 제안하여 기존 방법론의 한계를 완화하고자 합니다.

- **Technical Details**: 이 연구는 9개의 서로 다른 데이터셋을 활용한 4가지 close-ended 의료 작업과 9개의 다양한 데이터셋 기반의 6개 open-ended 작업을 고려합니다. CareQA는 스페인 보건부의 MIR 시험을 기초로 하여, 여러 카테고리에서 감사를 요구하는 5,621개의 QA 쌍을 포함하고 있으며, 영어와 스페인어로 제공됩니다. 이를 통해 각 작업의 일관성을 평가하고, 각 메트릭을 통해 모델의 성능을 측정하는 다양한 접근을 연구합니다.

- **Performance Highlights**: 실험에서는 MCQA 벤치마크와 다양한 open-ended 및 close-ended 작업 간의 상관관계를 분석한 결과, 임상 노트 작성만 MCQA와 약한 양의 상관관계를 보이는 것으로 나타났습니다. 그 외에 요약, 질문 함의와 같은 여러 작업은 MCQA와 부정적 상관관계를 보이며, 이는 의료 전문 지식의 필요성이 떨어지는 경우 때문입니다. 이러한 결과는 벤치마크 선택의 중요성과 또한 각각의 작업에 대한 특화된 평가의 필요성을 강조합니다.



### Evaluation of Deep Audio Representations for Hearables (https://arxiv.org/abs/2502.06664)
Comments:
          Accepted at International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 논문에서는 DEAR(Deep Evaluation of Audio Representations)이라는 새로운 데이터셋과 벤치마크를 첫 번째로 소개했습니다. 이 데이터셋은 청각 기기(hearables)의 오디오 표현력을 평가하기 위해 1,158개의 오디오 트랙을 포함하고 있습니다. 각 트랙은 독점적인 독백과 고품질 상업용 녹음을 혼합하여 생성되었으며, 다양한 음향 속성을 포착하는 데 중점을 두고 있습니다.

- **Technical Details**: DEAR 데이터셋은 음향 혼합의 완전한 제어를 보장하기 위해 배경 소리 장면에 음성 신호를 추가하여 생성되었습니다. 4차원 앰비소닉스 오디오를 사용하여 다양한 오디오 환경을 포착하였으며, 각 녹음은 30초 분량으로 구성되어 있습니다. 이 과정에서 데이터셋은 고유한 라벨과 함께 각 음성 신호의 활성 여부를 정확하게 레이블링 하였습니다.

- **Performance Highlights**: 다양한 대표 음향 모델을 평가한 결과, BEATs 모델이 다른 모델을 현저하게 초월하는 성능을 보였습니다. 이 모델은 특히 기술적 음향 속성을 효과적으로 인코딩하여, 향후 청각 기기를 위한 범용 오디오 모델 설계에 중요한 단서를 제공할 수 있습니다. DEAR 데이터셋은 공개적으로 사용 가능하며, 다양한 음향 과제들에서의 성능을 평가하는 데 활용될 수 있습니다.



### The 2021 Tokyo Olympics Multilingual News Article Datas (https://arxiv.org/abs/2502.06648)
- **What's New**: 이 논문에서는 2021 도쿄 올림픽에 대한 다국어 뉴스 기사 데이터셋 OG2021을 소개합니다. 총 10,940개의 뉴스 기사가 1,918개의 출처로부터 수집되어, 다양한 언어로 작성되었습니다. 이 데이터셋은 여러 사건에 대한 보도 기사를 그룹화하는 데 도움을 주기 위해 개발되었습니다.

- **Technical Details**: 이 데이터셋은 온라인 클러스터링 알고리즘을 활용하여 같은 하위 이벤트에 대한 기사를 그룹화하고, 수작업으로 주석을 달아 평가되었습니다. 언어는 영어, 스페인어, 독일어, 프랑스어, 러시아어 등을 포함하며, 2021년 7월 1일부터 8월 14일까지 출판된 기사를 포함합니다.

- **Performance Highlights**: OG2021 데이터셋은 특히 고빈도 이벤트가 발생하는 상황에서의 다국어 뉴스 클러스터링 알고리즘 성능 평가에 적합합니다. 이는 도쿄 올림픽의 문화적 및 언어적 차이를 분석하는 데도 유용하게 사용될 수 있습니다.



### Steel-LLM:From Scratch to Open Source -- A Personal Journey in Building a Chinese-Centric LLM (https://arxiv.org/abs/2502.06635)
- **What's New**: Steel-LLM은 중국어 중심의 언어 모델로, 제한된 컴퓨팅 자원 내에서 처음부터 개발되었습니다. 이 모델은 2024년 3월에 출시되었으며, 10억 개의 파라미터를 가진 모델을 대규모 데이터셋을 기반으로 교육하려고 합니다. 프로젝트의 목표는 투명성과 실제적인 통찰을 공유하여 커뮤니티의 다른 구성원에게 도움을 주는 것입니다.

- **Technical Details**: Steel-LLM은 Transformer 아키텍처를 기반으로 하며, Flash Attention과 Soft Mixture of Experts(Soft MOE)를 통합하여 성능 최적화를 이루었습니다. Flash Attention은 모델의 훈련 및 추론 효율성을 향상시키고 GPU 메모리를 절약하는 기능을 제공합니다. 또한, Steel-LLM은 8개의 GPU에서 제한된 자원으로 훈련되었으며, 모델 구조는 1억 개의 파라미터를 가진 소규모 언어 모델입니다.

- **Performance Highlights**: Steel-LLM은 CEVAL 및 CMMLU와 같은 벤치마크에서 경쟁력 있는 성능을 보여주며, 더 큰 기관의 초기 모델들을 초월했습니다. 모델의 개발 과정에서의 투명성을 제공하고 훈련 프로세스의 최적화에 대한 실제적 가이드를 제공하여 소규모 연구팀과 개인 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Automatic Annotation Augmentation Boosts Translation between Molecules and Natural Languag (https://arxiv.org/abs/2502.06634)
- **What's New**: 이번 연구에서는 LA$^3$라는 새로운 데이터 증강 프레임워크를 제안하여, 기존의 주석이 부족한 생물학적 데이터셋을 보완합니다. LA$^3$는 대규모 언어 모델을 활용하여 주어진 데이터셋의 주석을 자동으로 재작성하며, 이 과정에서 중요한 분자 정보를 유지하면서도 다양한 문장 구조와 어휘를 제공합니다. 이로 인해 기계 학습 모델의 훈련 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: LA$^3$는 자동으로 데이터셋의 주석을 증강하는 도구로, 기존 ChEBI-20 데이터셋을 활용하여 LaChEBI-20을 생성했습니다. 개선된 데이터셋에서는 각 분자에 대해 다양한 주석이 제공되며, 이를 통해 LaMolT5 모델을 훈련시켜 분자 표현과 증강된 주석 간의 매핑을 학습합니다. 학습 과정에서 모델은 주어진 분자 데이터에서 고유한 구조와 특성을 기반으로 작업을 수행하도록 설계되었습니다.

- **Performance Highlights**: LaMolT5는 분자 생성 작업에서 최대 301% 성능 향상을 이루었으며, 또 다른 작업인 분자 캡셔닝에서도 9.51% 개선된 결과를 보였습니다. 특히, LaMolT5의 소형 버전이 기존의 대형 MolT5 모델보다 뛰어난 성능을 나타냈으며, LA$^3$는 이미지 캡셔닝과 텍스트 이해, 그래프 속성 예측 등 다양한 다른 응용에서도 효과적으로 성능을 향상시키는 것으로 확인되었습니다.



### Combining Large Language Models with Static Analyzers for Code Review Generation (https://arxiv.org/abs/2502.06633)
- **What's New**: 이번 연구에서는 코드 리뷰 프로세스를 개선하기 위해 지식 기반 시스템(KBS)과 학습 기반 시스템(LBS)의 강점을 결합한 하이브리드 접근 방식을 제안합니다. 새로운 방법은 언어 모델 파이프라인의 세 가지 단계에서 지식을 통합하여 고품질의 종합적인 코드 리뷰를 생성합니다. 이러한 전략은 데이터 준비, 추론, 추론 후의 출력 결합을 포함하여 코드 리뷰에서의 효율성을 향상시킵니다.

- **Technical Details**: 우리는 하이브리드 접근 방식에서 데이터 증강 훈련(Data-Augmented Training, DAT), 검색 증강 생성(Retrieval-Augmented Generation, RAG), 및 출력의 단순 연결(Naive Concatenation of Outputs, NCO)이라는 세 가지 조합 전략을 사용했습니다. 이들 방법은 각각 모델의 학습, 추론 과정 및 결과 통합에 있어 KBS와 LBS의 지식을 효과적으로 적용하여 보다 정밀한 코드 리뷰를 생성하는 데 기여합니다.

- **Performance Highlights**: 실험적으로 평가한 결과, KBS와 LBS를 결합한 하이브리드 전략이 리뷰 코멘트의 관련성, 완전성 및 전반적인 품질을 개선시켰습니다. 이러한 개선은 규칙 기반 도구와 딥러닝 모델 간의 간극을 효과적으로 메꾸어 주며, 생성된 리뷰가 인간과 유사한 방향으로 더 효과적으로 작동하는 것을 보여줍니다.



### Few-Shot Classification and Anatomical Localization of Tissues in SPECT Imaging (https://arxiv.org/abs/2502.06632)
Comments:
          2 pages, 2 figures

- **What's New**: 본 연구는 제한된 라벨 데이터로부터 의료 이미징에서의 분류(classification) 및 해부학적 로컬라이제이션(localization)에 대한 도전 과제를 해결하기 위해 Prototypical Networks와 Propagation-Reconstruction Network (PRNet)을 각각 적용하였습니다. 특히, Single Photon Emission Computed Tomography (SPECT) 이미지를 활용하여 심장 주변의 2D 슬라이스 이미지를 대상으로 한 개념 증명(proof of concept)을 수행했습니다.

- **Technical Details**: Prototypical Network는 사전 학습된 ResNet-18 백본을 사용하여 심실(ventricle), 심근(myocardium), 간(liver) 조직을 분류하였으며, 훈련 세트에서 96.67%, 검증 세트에서 93.33%의 정확도를 달성했습니다. PRNet은 인코더-디코더(encoder-decoder) 아키텍처와 스킵 연결(skip connections)을 통해 2D SPECT 이미지의 해부학적 위치를 정확하게 예측하는 데 성공했습니다.

- **Performance Highlights**: Prototypical Network는 제한된 라벨 데이터로부터 조직 분류의 가능성을 보여주었으며, PRNet은 해부학적 랜드마크 로컬라이제이션에서 뛰어난 성능을 발휘했습니다. 이는 향후 딥 러닝 프레임워크의 성능 향상을 위한 기반을 마련할 수 있습니다.



### Conformal Predictions for Human Action Recognition with Vision-Language Models (https://arxiv.org/abs/2502.06631)
Comments:
          6 pages, 7 figures

- **What's New**: 본 논문에서는 Human-In-The-Loop (HITL) 프레임워크에 Conformal Predictions (CP)를 적용하여 Human Action Recognition (HAR) 작업에서의 성능을 향상시키는 방법을 제안합니다. 연구 결과, CP는 비디오 클립을 위한 후보 클래스의 평균 수를 크게 줄일 수 있음을 보여주었습니다. 또한, CP를 통해 생성된 클래스 세트에서는 긴 꼬리 분포가 나타나는 경향이 있습니다.

- **Technical Details**: 이 연구에서는 사전 훈련된 Vision-Language Models (VLMs) 위에 CP를 적용하는 방법을 탐구하며, 추가적인 파인튜닝 없이도 효과적인 HAR 분류 작업이 가능함을 발견했습니다. CP는 클래스에 대한 라벨 세트를 제공하는 동시에, 진짜 클래스가 포함될 확률에 대한 확고한 보장을 제공하는 특성이 있습니다. VLM의 온도 매개변수를 조정하여 CP의 긴 꼬리 분포를 최소화하는 방법도 제시합니다.

- **Performance Highlights**: 핵심적으로, 연구팀은 CP가 후보 클래스를 상당히 감소시킬 뿐만 아니라, 인간의 주석 시간 단축에도 기여할 수 있음을 입증하였습니다. 이 접근법은 비디오 모니터링과 같이 제한된 결정 시간을 요구하는 응용 프로그램에서 특히 유용합니다. GitHub에 공개된 코드와 함께 연구 결과를 통해 이 새로운 방법이 실제론에서 어떻게 적용될 수 있는지 보여주고 있습니다.



### TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models (https://arxiv.org/abs/2502.06608)
- **What's New**: 최근 확산(diffusion) 기술의 발전은 이미지 및 비디오 생성의 품질을 두 배로 향상시키며 생성 AI의 적용을 가속화하고 있습니다. 그러나 3D 형상 생성 기술은 3D 데이터의 양과 처리의 복잡성 덕분에 여전히 지체되고 있습니다. 이 논문에서는 이미지에 정밀하게 대응하는 고충실도 3D 메시를 생성하는 새로운 비대칭 모형인 TripoSG를 소개합니다.

- **Technical Details**: TripoSG는 대규모 정제된 흐름 변환기(rectified flow transformer)를 사용하는 혁신적인 3D 생성 모델로, SDF(Signed Distance Function), 법선(normal) 및 에이코날 손실(eikonal loss)을 조합한 하이브리드 감독 훈련 전략이 특징입니다. 이 모델은 4B 매개변수를 가진 고해상도 3D 구조물을 생성하며, 양질의 3D 데이터를 체계적으로 구축하는 데이터 처리 파이프라인을 개발하였습니다. 이러한 기술적 도약은 특히 3D 생성을 위한 분산(hybrid) 접근 방식을 가능하게 했습니다.

- **Performance Highlights**: TripoSG는 새로운 상태의 성능(SOTA)을 달성하며, 입력 이미지와의 정확한 정렬 및 뛰어난 일반화 능력을 보여줍니다. 이 모델은 다양한 이미지 스타일 및 콘텐츠로부터 3D 모델을 생성할 수 있습니다. 또한, 2백만 개의 고품질 3D 샘플을生成하여 학습 과정을 위한 데이터 품질 및 양의 중요성을 강조하고 있습니다.



### Illegal Waste Detection in Remote Sensing Images: A Case Study (https://arxiv.org/abs/2502.06607)
- **What's New**: 이번 논문은 환경 범죄 중 하나인 불법 쓰레기 투기 문제를 해결하기 위해 고해상도 원거리 감지(Very-High-Resolution Remote Sensing) 이미지를 활용한 새로운 파이프라인을 제안합니다. 이 시스템은 지역 환경 기관과의 협력을 기반으로 개발되었으며, 불법 매립지 탐지의 효율성을 크게 높입니다. 또한, 자동화된 분석 과정이 수동 사진 해석에 비해 시간 절약을 돕는다는 점에서 주목할 만합니다.

- **Technical Details**: 이 연구에서는 다양한 이미지 특성과 훈련 설정의 영향을 분석하기 위해 포괄적인 실험 세트를 수행했습니다. 원거리 감지 이미지(classifier of Remote Sensing images)에 대한 분류기의 최적 구성을 식별하기 위해 세부적인 분석을 진행하였으며, 이를 통해 환경 기관의 전문가들이 일상 작업에 통합할 수 있도록 했습니다.

- **Performance Highlights**: 개발된 분류기는 훈련 영역을 벗어난 장소에서도 유효한 결과를 도출하며, 이는 제안된 파이프라인의 국경 간 적용 가능성을 강조합니다. 이 연구는 불법 쓰레기 투기와 같은 환경 범죄에 대한 감시 및 대응에 실질적인 기여를 할 것으로 기대됩니다.



### Amortized In-Context Bayesian Posterior Estimation (https://arxiv.org/abs/2502.06601)
- **What's New**: 이 연구에서는 사전 확률을 통해 Bayesian posterior 추정의 최신 방법인 amortized inference를 비교 분석합니다. 이는 사전 기준으로 주어진 데이터의 맥락을 통해 사후 매개변수 추정을 수행하는 접근 방식으로, 반복적인 MCMC 샘플링이나 변분 추정을 회피할 수 있도록 합니다. 이 방법은 다양한 최적화 목표와 아키텍처 선택에 따라 다르게 동작할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Transformer와 같은 순차 모델에 맥락으로 전달된 데이터 샘플을 조건으로 하는 추정기를 학습하여 사후 파라미터 추정을 실시합니다. 연구팀은 permutation invariant 아키텍처를 활용하여, 맥락 예제의 순서에 관계없이 참된 사후가 불변하도록 하였습니다. 또한 다양한 설계 선택 및 추정 목표를 평가하며 Bayesian 추정을 위한 일반적인 프레임워크를 제시합니다.

- **Performance Highlights**: 실험 결과, 특히 예측 문제에서 reverse KL 추정기가 우수함을 입증하였습니다. 해당 연구는 다양한 확률 모델과 설계 선택들을 벤치마킹하며, synthetic 데이터에 기반한 사전 훈련만으로도 실제 데이터에 대해 뚜렷한 제너럴리제이션 능력을 보여주는 양상을 검토하였습니다. 따라서 연구의 최종 목표는 다양한 조건에서도 잘 작동하는 일반화 가능한 Bayesian 학습자를 개발하는 것입니다.



### Evaluation of Multilingual Image Captioning: How far can we get with CLIP models? (https://arxiv.org/abs/2502.06600)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 연구는 다국어 이미지 캡션 평가에서 CLIPScore 메트릭의 새로운 활용 방법과 실험 결과를 제시하고 있습니다. 특히, 기존의 영어 중심적인 접근 방식을 넘어 다국어 환경에서의 평가를 위한 전략을 탐색합니다. 연구에서는 기계 번역된 데이터와 다국어 데이터셋을 활용하여 평가 모델을 개선했습니다.

- **Technical Details**: 연구에서는 Multilingual CLIP 모델의 성능을 높이기 위해 두 가지 주요 데이터셋, CrossModal-3600과 VICR을 활용하는 방법을 제안합니다. 이들 데이터셋은 문화적 다양성을 고려하여 모델을 파인튜닝(finetuning)하는 데 필수적입니다. 사용된 손실 함수는 CLIP ‘contrastive loss’로, 다국어와 다문화 자원을 효율적으로 처리하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, 다국어 CLIPScore 모델은 다양한 언어에서 인간 평가와 높은 상관관계를 유지하며, 영어 평가에서도 동등하거나 더 나은 성능을 보였습니다. 이는 다국어 모델이 단일 언어 모델보다 더 다재다능하고 가치 있는 자원임을 입증하며, 문화적 및 언어적 다양성을 반영하는 평가 방법이 필요함을 강조합니다.



### Hephaestus: Improving Fundamental Agent Capabilities of Large Language Models through Continual Pre-Training (https://arxiv.org/abs/2502.06589)
Comments:
          Accepted to NAACL 2025 main conference

- **What's New**: 본 논문에서는 LLM 기반 자율 에이전트의 핵심 능력을 강화하기 위해 설계된 최초의 대규모 프리트레인(pre-training) 데이터셋인 Hephaestus-Forge를 소개합니다. Hephaestus-Forge는 103B의 에이전트 특화 데이터를 포함하며 76,537개의 API를 아우르고 있습니다. 또한, API 기능 호출, 내재적 추론(intrinsic reasoning), 계획(planning) 및 환경 피드백에 대한 적응 능력을 향상시키기 위한 솔루션을 제공합니다.

- **Technical Details**: 이 연구에서는 LLM 에이전트의 능력을 근본적으로 향상시키기 위한 두 가지 주요 목표를 설정하고 있습니다: (a) 개별 API 함수 호출의 이해도를 향상시키고, (b) 여러 함수 호출이 필요한 문제 해결을 위한 내재적 추론 능력을 강화하는 것입니다. 이러한 목표를 달성하기 위해, Tool 문서화와 함수 호출 경로를 대규모 데이터셋으로 수집하였고, 이를 통해 Hephaestus-Forge라는 고품질의 훈련 코퍼스를 구성하였습니다.

- **Performance Highlights**: Hephaestus는 지속적 프리트레인을 통해 소규모 및 중간 규모의 오픈소스 LLM을 초월하여 상업용 LLM과 동등한 성능을 발휘합니다. 연구 결과, Hephaestus-8B는 여러 에이전트 벤치마크에서 약 9.6%에서 17.6%까지 성능 향상을 보였으며, 상업용 LLM인 Claude-3-Haiku 및 GPT-3.5-turbo와 비교하여도 높은 성능을 기록했습니다. 이를 통해 Hephaestus-Forge의 효과성을 입증하고 있습니다.



### The Minimal Search Space for Conditional Causal Bandits (https://arxiv.org/abs/2502.06577)
Comments:
          Submitted to ICML2025

- **What's New**: 이 논문은 인과 그래프( causal graph )를 기반으로 조건부 개입( conditional interventions )을 고려하며, 기존의 경직된 개입( hard interventions )에 비해 더 현실적인 의사 결정 문제를 모델링합니다. 저자들은 최적의 조건부 개입을 보장하는 최소 노드 집합을 그래픽적으로 특성화하고, 이 집합을 효율적으로 식별하는 알고리즘을 제안합니다. 이 연구는 인과 대형 밴드트 문제(causal bandit problem)의 검색 공간을 줄이는 데 있어 새로운 접근법을 제공합니다.

- **Technical Details**: 저자들은 시간 복잡도 O(|V| + |E|)인 알고리즘을 제안하여 최적의 노드 집합을 찾는 방법을 설명합니다. 이를 통해, 논문에서는 인과 모델의 모든 노드 집합 중 조건부 개입을 수행할 수 있도록 보장하는 최소 노드 집합의 정의와 이 알고리즘의 정확성을 증명합니다. 또한 원래의 MAB 알고리즘에 통합했을 때, 알고리즘이 검색 공간을 크게 줄이고 수렴 속도를 가속화하는 데 기여함을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 실제 및 무작위로 생성된 그래프에서 검색 공간의 상당한 비율을 줄일 수 있음을 실험적으로 입증했습니다. 다양한 실제 모델을 사용하여, 이 알고리즘의 개입 선택 방식이 기존의 고전적인 다중 팔 밴드트 알고리즘( classical MAB algorithm )을 상당히 개선함을 나타냈습니다. 이러한 성과들은 더욱 효과적인 인과 기반 의사 결정 시스템 구축에 기여할 것으로 기대됩니다.



### Predictive Red Teaming: Breaking Policies Without Breaking Robots (https://arxiv.org/abs/2502.06575)
- **What's New**: 이 논문에서는 비전-모터 (visuomotor) 정책에 대한 취약성을 예측하는 새로운 접근인 predictive red teaming을 제안합니다. 이를 통해 환경적 요인(환경적 팩터)과 관련된 취약성을 식별하고 하드웨어 평가 없이 성능 저하를 예측할 수 있습니다. 이를 위해 RoboART라는 자동화된 red teaming 파이프라인을 개발하여 환경 요인을 변화시키고 이에 따른 정책의 성능을 예측합니다.

- **Technical Details**: RoboART는 생성적 이미지 편집(generative image editing) 기술을 사용하여 표준 관측(nominal observations)을 수정합니다. 그런 다음, 수정된 관측값에 대해 정책 특정 이상 탐지기(policy-specific anomaly detector)를 실행하여 각 변형에 따른 성능을 예측합니다. 이 방법은 500회 이상의 하드웨어 실험을 바탕으로 twelve 오프-노미널 조건(off-nominal conditions)에서 테스트되었습니다.

- **Performance Highlights**: RoboART는 예측 성능 저하에서 높은 정확도를 보여 주며, 예측된 성공률과 실제 성공률 간의 평균 차이는 0.19 미만으로 나타났습니다. 아울러, 예측된 불리한 조건에서 수집된 데이터로 미세 조정(fine-tuning)을 수행하면 성능이 2-7배 증가했음을 보여주었습니다.



### LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM (https://arxiv.org/abs/2502.06572)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 법적 추론을 위한 데이터 생성을 위해 KgDG라는 지식 기반 데이터 생성 프레임워크를 제안합니다. 이 프레임워크는 법적 지식을 활용하여 데이터 생성을 다양화하고, 생성된 데이터의 품질을 보장하기 위한 정제 및 검증 과정을 소개합니다. 이를 통해 기존의 오픈 소스 LLM들의 성능을 개선하려고 하며, 50,000개의 질 높은 샘플을 포함하는 합성 데이터셋을 생성했습니다.

- **Technical Details**: KgDG 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Knowledge-Guide Generation (KgGen), (2) Knowledge-Guide Fixer (KgFix), (3) Data Verifier (DaVer). KgGen은 법적 지식 기반을 활용하여 다양한 데이터를 생성하고, KgFix는 잘못된 참조와 논리 경로를 수정하며, DaVer는 수정 불가능한 데이터를 필터링하여 생성 품질을 보장하는 역할을 합니다. 또한 Mixture Training (MiTra) 전략을 통해 생성된 데이터셋을 확대하여 LLM의 추론 능력을 향상시킵니다.

- **Performance Highlights**: LawGPT 모델은 기존의 법률 전용 LLM들보다 뛰어난 성능을 보이며, 독점 LLM들과 비교할 만한 결과를 달성했습니다. 이는 KgDG와 LawGPT의 효율성을 입증하며, 법적 추론 분야에서 기존 솔루션들보다 우수한 성능을 발휘함을 보여줍니다. 이 연구의 데이터셋과 모델은 향후 연구를 위해 공개될 예정입니다.



### Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation (https://arxiv.org/abs/2502.06516)
Comments:
          29 pages, 11 figures

- **What's New**: 본 연구에서는 Diffusion 모델을 이용하여 소수 샘플을 생성할 수 있는 간단하면서도 강력한 가이드 없는 접근법인 Boost-and-Skip을 제안합니다. 기존의 소수 샘플 생성 방식은 소모적인 계산 리소스를 요구하는 가이드를 사용하지만, 이 방법은 두 가지 최소한의 변경으로 이를 해결합니다. 특히, 분산을 높인 초기화와 타임스텝 건너뛰기를 통해 소수 특성의 발현을 촉진합니다.

- **Technical Details**: Boost-and-Skip는 표준 생성 프로세스의 두 가지 주요 수정을 통해 작동합니다: (i) 분산 증가 초기화, (ii) 타임스텝 스킵. 처음 수정은 초기 생성에서 저밀도 영역에서의 초기화를 유도하는 고유한 노이즈를 사용하며, 두 번째 수정은 초기 단계에서 몇 개의 타임스텝을 건너뛰어 저밀도 초기화의 효과를 높입니다. 이 두 수정은 이론적으로 및 경험적으로 소수 샘플 생성 성능을 현저하게 개선합니다.

- **Performance Highlights**: 실험 결과, Boost-and-Skip은 기존의 소수 생성 방법들과 비교하여 경쟁력 있는 성능을 보여주면서도 현저히 낮은 계산 비용을 요구합니다. 예를 들어, ImageNet 데이터셋에서 본 방법은 이전의 최첨단 방법보다 65% 적은 시간과 4.5배 낮은 메모리 소비로 작동합니다. 이 접근법은 그 단순성 덕분에 매우 확장 가능하여 실제 분류 작업의 데이터 증강에서도 효과적입니다.



### GuideLLM: Exploring LLM-Guided Conversation with Applications in Autobiography Interviewing (https://arxiv.org/abs/2502.06494)
Comments:
          31 pages; the first three authors contributed equally

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 주도하는 대화의 잠재력을 탐구합니다. LLM 주도 대화의 세 가지 기본 구성 요소인 목표 탐색(Goal Navigation), 맥락 관리(Context Management), 공감적인 참여(Empathetic Engagement)를 정의하고 GuideLLM이라는 새로운 모델을 제안합니다.

- **Technical Details**: GuideLLM은 다양한 주제를 포함하는 인터뷰 환경을 설정하여 LLM 주도 대화의 질을 평가합니다. 이 환경에서 약 1,400번의 발화와 184,000개의 토큰이 생성되었고, 200개 이상의 이벤트가 언급되었습니다. 여러 최신 LLM 모델과 비교 분석하여 GuideLLM의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, GuideLLM은 자동 평가에서 기존 LLM보다 현저히 우수한 성과를 보여주었으며, 인간 평가에서도 일관되게 높은 점수를 기록했습니다. 특히, 자서전 생성 품질과 대화 품질 측면에서 두드러진 성과를 보였습니다.



### Model-Based Offline Reinforcement Learning with Reliability-Guaranteed Sequence Modeling (https://arxiv.org/abs/2502.06491)
- **What's New**: 본 논문에서는 신뢰성 보장 변환기(Reliability-guaranteed Transformer, RT)라는 새로운 MORL 알고리즘을 제안합니다. RT는 생성된 경로의 누적 신뢰성을 계산하여 불신뢰한 경로를 제거할 수 있는 능력을 엔지니어링합니다. 이를 통해 과거 정보가 환경 동력학에 미치는 영향을 고려하여 더 신뢰할 수 있는 경로를 생성할 수 있도록 합니다.

- **Technical Details**: RT는 입력 시퀀스 내의 다양한 위치 간의 역사적 정보를 포착하여 데이터를 생성합니다. 생성된 데이터와 실제 데이터 간의 변이 거리를 계산하여 각 시간 단계에 대한 신뢰성 값을 도출하고, 이를 바탕으로 경로 길이를 적응적으로 결정합니다. 이 메커니즘 덕분에 RT는 전반적인 신뢰성을 향상시켜 모델 기반 방법에서의 정책 학습에 기여합니다.

- **Performance Highlights**: 다양한 벤치마크 태스크에서 RT는 기존의 최첨단 모델 기반 방법들과 비교하여 뛰어난 학습 성능을 입증했습니다. 연구자들은 RT가 생성하는 고수익 경로가 기존 접근 방식보다 효율적으로 높다는 것을 보여주었으며, 정책 학습의 안정성을 크게 개선할 수 있음을 강조합니다.



### Recent Advances in Discrete Speech Tokens: A Review (https://arxiv.org/abs/2502.06490)
Comments:
          26 pages, 8 figures, 3 tables. Work in progress

- **What's New**: 최근의 음성 생성 기술은 대규모 언어 모델(LLMs)의 발전과 함께 중요한 진전을 이루었으며, 이는 음성을 위한 새로운 표현 방식으로서 'discrete speech tokens'의 채택으로 이어졌습니다. 이 논문은 음성을 텍스트 중심의 LLM 아키텍처에 통합할 수 있는 개선된 방법론을 제시하고, 서로 다른 유형의 음성 토큰화 방법과 그 응용 분야를 체계적으로 논의합니다.

- **Technical Details**: 논문에서는 음성을 'acoustic tokens'와 'semantic tokens'로 나누어 두 가지 주요 유형의 사전 처리 방법을 강조합니다. Acoustic tokens는 낮은 비트 전송률에서 음성을 인코딩하기 위해 설계된 신경 코덱에서 유래하며, Semantic tokens는 음성 자기 지도 학습(SSL)을 통해 음성을 더 현실감 있게 표현하는 방법을 채택합니다. 각각의 접근 방식은 독특한 설계 철학과 방법론을 갖고 있습니다.

- **Performance Highlights**: 이 논문은 다양한 유형의 'discrete speech tokens'에 대한 포괄적인 비교를 제공하며, 각각의 재구성 성능과 음성 변환 성능을 분석합니다. 기존 연구들과 비교하여, 새로운 데이터 세트와 실험적 접근 방식을 통해 음성 처리 분야에서의 주요 도전 과제와 향후 연구 방향을 제시하고 있습니다. 이를 통해 연구자들에게 향후 연구와 개발에 필요한 인사이트를 제공합니다.



### WyckoffDiff - A Generative Diffusion Model for Crystal Symmetry (https://arxiv.org/abs/2502.06485)
- **What's New**: 이 논문에서는 대칭(symmetry)을 고려한 결정(crystal) 생성 모델인 Wyckoff Diffusion(WyckoffDiff)를 제안합니다. 기존의 생성 모델들이 각 원자를 위치나 요소에 대한 제약 없이 모델링하는 것과 달리, WyckoffDiff는 결정의 대칭적 특성을 기반으로 자료를 생성합니다. 이러한 접근은 결정 구조의 표현을 통해 가능하며, 새로운 신경망(neural network) 아키텍처를 설계하여 이 표현을 이산 생성 모델(discrete generative model) 프레임워크에서 활용할 수 있도록 하였습니다.

- **Technical Details**: WyckoffDiff는 대칭 정보를 포함하는 프로토구조(protostructure) 설명을 활용하여 제작됩니다. 이 설명은 원자들이 결정 구조의 특정 위치에 배치되도록 하여 대칭을 유지합니다. 논문에서는 또한 Fréchet Wrenformer Distance라는 새로운 지표를 소개하여 생성된 자료의 대칭 측면을 평가하고, 최근 제안된 결정 생성 모델들과 비교하여 성능을 벤치마킹하였습니다.

- **Performance Highlights**: WyckoffDiff는 대칭을 통한 더 많은 높은 대칭 자료 생성이 가능하며, 이전 모델들보다 우수한 성과를 보입니다. 생성된 프로토구조는 기계 학습 기반의 재료 발견 워크플로우의 일부로 활용될 수 있습니다. 본 논문에서는 생성한 프로토구조 중 일부를 결정 구조로 실현하여, CsSnF6, NaNbO2, Ca2PI와 같은 흥미로운 화학 조성을 가진 예시를 제시하고 있습니다.



### KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichmen (https://arxiv.org/abs/2502.06472)
Comments:
          24 pages, 3 figures, 2 tables

- **What's New**: KARMA는 다중 에이전트 기반의 새로운 지식 그래프(KG) 보강 프레임워크로, 비구조화된 텍스트의 구조적 분석을 통해 KG의 자동화된 보강을 목표로 합니다. 이 시스템은 엔티티 발견, 관계 추출, 스키마 정렬, 충돌 해결 등을 수행하는 아홉 개의 협력적인 에이전트를 사용하여 기존의 KG에 새로운 지식을 통합합니다. KARMA의 접근 방식은 도메인별 스키마를 준수하면서 관계의 정확성을 높이고, 문서의 지식을 검증하며, 비구조화된 텍스트에서 고품질 지식을 효율적으로 추출할 수 있도록 설계되었습니다.

- **Technical Details**: KARMA는 계층적 다중 에이전트 시스템으로, 각 에이전트는 KG 보강 파이프라인에서 특정 작업을 처리하기 위해 전문화된 LLM(대형 언어 모델)을 활용합니다. 기존 KG는 엔티티 집합(V)과 관계 집합(E)으로 구성되며, 비구조화된 문서에서 새로운 관계 트리플(t) 을 자동으로 추출하여 이를 KG에 통합하는 구조로 이루어져 있습니다. 이러한 프로세스는 문서 수집부터 KG 통합까지의 모듈화된 하위 과제로 분리되며, 각 에이전트는 독립적으로 고유한 작업을 수행합니다.

- **Performance Highlights**: KARMA의 효과를 확인하기 위한 실험은 1,200개의 PubMed 논문을 대상으로 진행되었으며, 최대 38,230개의 새로운 엔티티를 식별하는 성과를 이루었습니다. 이 과정에서 LLM으로 검증된 정확도는 83.1%에 달했으며, 다층 평가를 통해 충돌 엣지를 18.6% 감소시켰습니다. 이러한 성과는 KARMA가 KG 보강에서 정확성과 확장성을 동시에 달성할 수 있는 가능성을 보여줍니다.



### A Survey of Theory of Mind in Large Language Models: Evaluations, Representations, and Safety Risks (https://arxiv.org/abs/2502.06470)
Comments:
          Advancing Artificial Intelligence through Theory of Mind Workshop, AAAI 2025

- **What's New**: 이 논문은 Theory of Mind (ToM) 연구를 통해 대형 언어 모델(LLMs)이 다른 사람의 정신 상태를 이해하고 행동을 예측하는 능력을 가지고 있다는 점에 주목하고 있습니다. 특히, LLM의 ToM 기능이 높은 수준으로 발달하면 개인 정보 침해 및 집단의 불일치와 같은 안전 문제를 초래할 수 있음을 강조합니다. 향후 연구는 이러한 리스크를 효과적으로 평가하고 완화할 방법을 찾는 데 중점을 두어야 합니다.

- **Technical Details**: 최근 연구에 따르면, LLM인 GPT-4는 특정 ToM 작업에서 7세에서 10세 어린이 또는 성인과 유사한 성능을 보입니다. 그러나 BigToM, FANToM, OpenToM 등과 같은 LLM 전용 벤치마크에서는 대다수의 모델이 인간보다 성능이 떨어진다고 합니다. LLM의 내부 표현은 ToM 기능에 긍정적인 영향을 미치며, 더 큰 모델에서 이러한 표현의 정확도가 증가하는 것으로 나타났습니다.

- **Performance Highlights**: 고급 ToM 기능은 사회적 과학 시뮬레이션과 같은 유익한 응용 프로그램을 가능하게 하지만, 동시에 개인정보 침해 및 정교한 속임수와 같은 위험도 동반합니다. 연구에 따르면, LLM이 고급 ToM을 활용하여 개인의 신념, 선호 및 경향성을 추출하고, 이를 통해 맞춤형 허위 정보 캠페인을 벌이는 가능성이 있습니다. 따라서 LLM의 ToM 진전을 면밀히 모니터링하고 평가하는 것이 중요합니다.



### MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations (https://arxiv.org/abs/2502.06453)
- **What's New**: 본 논문은 새로운 MATH-P-Simple과 MATH-P-Hard 벤치마크를 구축하여 언어 모델의 수학적 추론 성능을 평가합니다. Simple perturbation(간단한 변화)과 hard perturbation(어려운 변화)을 통해 원래 문제의 해결 방식과의 연관성을 유지하면서도 본질적인 문제의 구조를 변경합니다. 이를 통해 모델에서 관찰된 성능 저하의 원인을 규명하며, 모델이 문제의 수정된 맥락을 평가하지 않고 무작정 학습된 기술을 적용하는 새로운 형태의 암기(memorization)를 발견했습니다.

- **Technical Details**: MATH-P-Simple은 279개의 변형된 수학 문제로 구성되어 있으며, 원본 문제는 MATH 데이터셋의 레벨 5(가장 어려운) 문제에서 파생되었습니다. 각 변형 문제는 원본 문제를 바탕으로 하여 최소한의 수정(minimal edits)을 적용하고, 수정된 문제의 답이 원본과 다르게 설정되었습니다. 또한 12명의 수학 박사 과정 학생들에 의해 품질 통제를 위한 이중 검사(double-checking)가 수행되었습니다.

- **Performance Highlights**: Varying models such as o1-mini와 gemini-2.0-flash-thinking은 MATH-P-Hard에서 각각 -16.49% 및 -12.9%의 성능 저하를 보였습니다. 이는 모델들이 원래 문제 형식에 편향되어 있으며, hard perturbations의 특정 설정에서 전혀 새로운 문제에 직면했을 때 이전의 해결 기술을 무작정 적용함으로써 발생하는 문제점을 강조합니다. 연구진은 이러한 문제를 해결하기 위한 추가 연구의 필요성을 강조합니다.



### SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding (https://arxiv.org/abs/2502.06440)
Comments:
          Accepted for presentation at the 2025 IEEE International Conference on Robotics and Automation (ICRA)

- **What's New**: 이 논문은 Multi-Agent Path Finding (MAPF) 문제를 새로운 프레임워크인 SIGMA를 통해 재정의합니다. 이 프레임워크는 sheaf 이론을 활용하여 로컬 합의(consensus) 기반의 협업 의사결정을 가능하게 합니다. 이를 통해 에이전트는 서로의 기하학적 상호 의존성을 학습하고, 더 나은 경로 계획 및 충돌 회피를 할 수 있게 됩니다.

- **Technical Details**: SIGMA는 분산 심층 강화 학습(decentralized deep reinforcement learning)을 통해 sheaf 이론을 적용하는 방법론입니다. 이 방법론은 각 에이전트가 로컬 관찰을 통해 글로벌 합의를 이루기 위한 수학적 조건을 제공합니다. 또한, neural network를 사용하여 잠재 공간(latent space)에서의 합의를 근사적으로 모델링하고, 자가 지도 학습(self-supervised learning)으로 이를 훈련시킵니다.

- **Performance Highlights**: 결과적으로, SIGMA는 기존의 최첨단 학습 기반 MAPF 계획자들보다 상당한 성능 개선을 보입니다. 특히 대규모 및 복잡한 시나리오에서 우수한 성과를 내며, 다양한 시뮬레이션 및 실제 로봇 실험에서도 우위를 점합니다. 이 연구는 에이전트 간의 긴밀한 협력이 필요한 환경에서의 해결책을 제시하여, MAPF 문제 해결에 중요한 기여를 하고 있습니다.



### Testing software for non-discrimination: an updated and extended audit in the Italian car insurance domain (https://arxiv.org/abs/2502.06439)
Comments:
          14 pages, 1 figure

- **What's New**: 이 연구는 이탈리아 자동차 보험 산업에서 온라인 소프트웨어 시스템을 감사하는 방식으로 비차별(non-discrimination)을 위한 알고리즘을 테스트하는 중요성을 강조합니다. 기존 연구를 기반으로 하여 성별 및 출생지와 같은 다양한 인구통계적 변수를 추가하여 더 정교한 테스트 프로토콜을 개발했습니다. 이는 알고리즘의 불공정성 문제를 제기하며, 소프트웨어 품질에 미치는 사회적 및 윤리적 리스크를 관리할 필요성을 보여줍니다.

- **Technical Details**: 알고리즘 감사는 소셜 사이언스에서 유래된 결과 수집 및 분석 방법으로, 알고리즘의 결과를 평가합니다. 이 연구에서는 GQM (Goal-Question-Metric) 템플릿을 사용하여 연구 목표를 정의하고, 비차별성에 대한 연구 질문을 도출하였습니다. 보호되는 속성들 (gender, birthplace, age) 및 사회-인구학적 속성들이 보험료에 미치는 영향을 분석하기 위한 실험 설계를 포함하고 있습니다.

- **Performance Highlights**: 연구 결과는 여전히 성별 및 출생지가 보험료에 차별적으로 작용하고 있음을 보여줍니다. 특히, 외국에서 태어난 운전자와 특정 이탈리아 도시 출신의 운전자가 상대적으로 높은 보험료를 지불하는 경향이 있음을 발견했습니다. 이 연구는 정기적인 알고리즘 감사를 통해 시간에 따른 알고리즘의 진화를 평가하는 것이 가능하다는 점을 강조하며, 소프트웨어 시스템의 책임성을 높이기 위한 방향성을 제시합니다.



### FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Mod (https://arxiv.org/abs/2502.06438)
Comments:
          7 pages, 3 figures, 5 tables, pre-print

- **What's New**: FEMBA(Foundational EEG Mamba + Bidirectional Architecture)는 EEG(전자 뇌파 검사) 분석의 효율성을 혁신적으로 향상시키기 위한 새로운 자기 지도 학습 프레임워크입니다. 기존의 Transformer 기반 모델의 제한점을 극복하여, 연속 시퀀스에 대해 선형적으로 확장 가능한 구조를 제공하며, 이는 자원에 제약이 있는 환경에서도 유용하게 사용될 수 있습니다. 또한, 21,000시간 이상의 비 라벨 EEG 데이터를 활용하여 훈련되었고, 세 가지 주요 작업에서도 성능을 조정했습니다.

- **Technical Details**: FEMBA는 바이디렉셔널 상태 공간 모델링을 통해 EEG 데이터를 처리하며, 기존의 Quadratic 시간복잡도를 넘어서는 성능을 자랑합니다. 이를 통해, 연속 EEG 기록을 보다 효율적으로 처리할 수 있는 가능성을 보여줍니다. 파라미터 수가 7.8M에 불과한 소형 변형 모델도 자원 제한 장치에서의 활용 가능성을 명확히 입증하였습니다.

- **Performance Highlights**: FEMBA는 TUAB에서 81.82%의 균형 잡힌 정확도와 0.8921 AUROC를, TUAR에서 0.949 AUROC를 기록하며 Transformer 모델들과 경쟁할 만한 성능을 달성했습니다. 이러한 성과는 FEMBA가 임상과 착용 가능한 건강 장치 분야에서 활용될 수 있는 가능성을 제시하며, 향후 EEG 분석의 새로운 효율적 기준을 설정할 것으로 기대됩니다.



### Prompt-SID: Learning Structural Representation Prompt via Latent Diffusion for Single-Image Denoising (https://arxiv.org/abs/2502.06432)
- **What's New**: 이 연구에서는 Prompt-SID라는 프롬프트-학습 기반의 단일 이미지 노이즈 제거 프레임워크를 소개합니다. 기존의 자가 감독 및 비지도 방식이 가지고 있는 문제점인 픽셀 정보 손실과 구조적 디테일 손상을 해결하는 것을 목표로 합니다. 이 프레임워크는 다운샘플링된 이미지 쌍을 사용하여 자가 감독 방식으로 학습하며, 구조 인코딩을 통해 원본 스케일의 이미지 정보를 캡처합니다.

- **Technical Details**: Prompt-SID는 잠재적 확산 과정에 기반한 구조 표현 생성 모델을 제안하였습니다. 이 모델은 다운샘플링된 이미지에서 손상되지 않은 구조 표현을 복구하기 위한 조건 정보로 손상된 구조 표현을 사용합니다. 또한, 트랜스포머 기반의 노이저에서 구조 표현을 프롬프트로 통합하는 구조적 주의 모듈(SAM)을 설계하였습니다.

- **Performance Highlights**: Prompt-SID는 합성, 실제, 형광 이미징 데이터 세트에서 뛰어난 효과를 나타내며, 상대적으로 적은 매개변수 수로도 높은 성능을 자랑합니다. 기존의 SOTA(최신 기술) 접근 방식을 초월하는 성능을 보여주며, 실제 스케일 이미지를 처리할 수 있는 일반화 능력을 유지합니다.



### Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs (https://arxiv.org/abs/2502.06425)
Comments:
          Accepted to The ACM Web Conference (WWW) 2025 Short Paper Track

- **What's New**: 본 연구는 대규모 언어 모델(LLM) 기반의 챗봇과 현대의 제로 지식 증명(Zero-Knowledge Proof, ZKP) 프레임워크인 zkVM을 결합하여 개인 정보를 보호하면서 사용자 맞춤형 조언을 제공합니다. 이는 사용자의 특성을 안전하게 검증하면서도 민감한 정보를 노출하지 않는 새로운 접근 방식을 제시합니다. 연구는 아키텍처와 프롬프트 전략을 도입하여 zkVM과 LLM의 통합을 통해 현재 제약 및 성능 한계를 명확히 합니다.

- **Technical Details**: 본 연구에서 제안하는 아키텍처는 두 개의 독립적인 주체(Entity)로 운영됩니다. 첫 번째 주체는 사용자의 데이터를 보유하는 기관이며, 두 번째 주체는 LLM을 활용하여 사용자에게 조언을 제공하는 클라우드 기반 서비스입니다. 이 두 주체는 개인 데이터를 서로 공유하지 않으며, 사용자는 특정 프로필을 기반으로 유사한 특성을 추론하도록 첫 번째 주체에 요청하게 됩니다.

- **Performance Highlights**: zkVM은 자체 프로그램을 제로 지식 증명으로 실행할 수 있도록 해주는 현대적인 아키텍처로, 프로그램 실행 추적의 일관성을 증명할 수 있는 장점을 가지고 있습니다. 기존 프로그램을 최소한으로 수정하여 제로 지식 증명 기술을 통합할 수 있어, 개발자가 사용할 수 있는 고수준 언어와 라이브러리를 통해 학습 곡선을 크게 줄일 수 있습니다. 이를 통해 사용자 특성과 증명서를 LLM 조언자 시스템에 전달하여 보다 개인화된 조언 생성을 가능하게 합니다.



### CS-SHAP: Extending SHAP to Cyclic-Spectral Domain for Better Interpretability of Intelligent Fault Diagnosis (https://arxiv.org/abs/2502.06424)
Comments:
          21 pages, 21 figures

- **What's New**: 이 논문에서는 사이클릭 스펙트럴(CS) 변환을 도출하고, 이를 통해 Shapley additive explanations (SHAP)을 확장한 CS-SHAP를 제안하여 고급 해석 가능성을 제공하고 있습니다. CS-SHAP는 기계 고장 진단에서의 설명 가능성을 개선하며, 기존 포스트 hoc 방법들이 가지는 한계를 해결하기 위해 설계되었습니다. 이로 인해 더욱 명확하고 정확한 설명을 제공하며, 실제 상황에서의 신뢰성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: CS-SHAP는 캐리어 주파수와 변조 주파수의 기여도를 평가하여 동작하며, 이를 통해 기계 고장 메커니즘에 더 밀접하게 align합니다. CS 영역에서의 접근 방식을 통해, CS-SHAP는 시간 도메인 신호에서 메커니즘 통찰력을 더 명확하게 관찰할 수 있게 합니다. 세 가지 데이터 세트를 통해 CS-SHAP의 우수한 해석 가능성을 검증하였으며, 그 결과는 코드 및 오픈 소스로 제공됩니다.

- **Performance Highlights**: CS-SHAP는 우수한 해석 가능성을 보장하여 기계 고장 진단(I FD) 분야의 포스트 hoc 해석 가능성 벤치마크가 될 잠재력을 가지고 있습니다. 또한, CS-SHAP는 다양한 분류 작업에서도 효과적으로 사용될 수 있어, 기계 장비 유지보수 및 개선에서 그 응용 가능성을 넓히고 있습니다. 이 모델은 확장성과 유연성이 뛰어나, 실제 산업 환경에서도 신뢰할 수 있는 성능을 입증할 수 있습니다.



### Systematic Outliers in Large Language Models (https://arxiv.org/abs/2502.06415)
Comments:
          Accepted at ICLR 2025. Project Page: this https URL

- **What's New**: 이 논문에서는 Large Language Models (LLMs) 내의 outlier(이상치) 유형을 정의하고 분류하는 새로운 접근 방식을 제공합니다. 세 가지 유형인 activation outliers, weight outliers, attention outliers를 제시하며, 이들 사이의 고유한 연결성과 주의 메커니즘에 미치는 영향을 탐구합니다. 또한 이 연구는 outliers가 self-attention mechanism의 softmax operation에서 발생하며, 이들이 주의 메커니즘 내에서 문맥을 인식하는 스케일링 요인으로 작용한다는 점을 밝힙니다.

- **Technical Details**: 연구는 activation, weight, attention outliers의 수학적 정의를 통해 각 유형의 존재를 세밀하게 분석합니다. LLaMA2-7B 모델에서 outliers의 분포 위치를 제시하며, 이들의 시스템적 특성을 설명합니다. 이 논문은 이론적 추정과 실험을 통해 outliers가 주의 메커니즘에 미치는 영향을 규명하며, 이를 통해 훈련 과정의 수렴 속도를 높일 수 있는 방법을 제안합니다.

- **Performance Highlights**: 이 연구는 outliers를 구조적으로 제거함으로써 LLM의 수렴 속도와 모델 압축을 개선할 수 있음을 보여줍니다. 이전 연구에서 outliers가 모델 성능을 저하시키는 요인으로 지적된 반면, 이 논문에서는 그 존재를 이해함으로써 효과적인 최적화 방안을 제시합니다. 따라서, outliers의 체계적 처리는 LLM의 성능과 효율성을 동시에 향상시키는 중요한 실마리가 될 수 있습니다.



### Solving Linear-Gaussian Bayesian Inverse Problems with Decoupled Diffusion Sequential Monte Carlo (https://arxiv.org/abs/2502.06379)
- **What's New**: 이번 연구는 사전 훈련된 generative diffusion 모델을 이용하여 Bayesian 역문제를 해결하기 위한 sequential Monte Carlo 방법을 설계한 내용을 다루고 있습니다. 이 과정에서 'decoupled diffusion' 방식을 통해 샘플에 대한 보다 큰 업데이트를 허용하는 기법을 개발하였습니다. 제안된 'Decoupled Diffusion Sequential Monte Carlo (DDSMC)' 알고리즘은 합성 데이터 및 이미지 재구성 작업에서 효과적임을 입증하였습니다.

- **Technical Details**: 기술적으로, 본 논문에서는 linear-Gaussian likelihood 모델을 대상으로 한 Bayesian 역문제에서의 SMC 활용을 목표로 합니다. 특히, DDSMC는 decoupled diffusion에 기반하여 샘플 업데이트를 크게 허용하도록 설계되었습니다. 이러한 접근 방식은 SMC 제안 분포 설계에 있어 conditioning on 
 bold_y를 고려하는 새로운 방법을 제시합니다.

- **Performance Highlights**: DDSMC 메소드는 합성 데이터와 이미지 재구성 작업 모두에서 우수한 성능을 보여줍니다. 기존의 방법들과 비교하였을 때, DDSMC는 근사에 의존하지 않고 Bayesian 추론 문제를 더 정확하게 해결할 수 있는 가능성을 제시합니다. 또한, 이 방법은 이산 데이터를 처리할 수 있는 확장성 또한 갖추고 있습니다.



### Hyperparameters in Score-Based Membership Inference Attacks (https://arxiv.org/abs/2502.06374)
Comments:
          This work has been accepted for publication in the 3rd IEEE Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final version will be available on IEEE Xplore

- **What's New**: 본 연구에서는 Membership Inference Attacks (MIAs)의 새로운 접근법을 제안하며, 특히 챗GPT 모델의 하이퍼파라미터에 대한 사전 지식 없이도 MIAs를 수행할 수 있는 방법론을 개발했습니다. 전통적으로 MIAs는 목표 모델의 하이퍼파라미터를 알고 있다는 가정하에 이루어졌으나, 본 연구에서는 이러한 가정이 무효임을 입증하고, 목표 모델과 그림자 모델의 출력을 일치시켜 하이퍼파라미터를 선택하는 신규 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Transfer Learning 환경에서 이미지 분류 문제를 대상으로 하여, KL-LiRA라는 새로운 MIAs 방법을 통해 그림자 모델의 하이퍼파라미터 최적화 문제를 해결합니다. 이 방법은 목표 모델의 하이퍼파라미터에 대한 사전 지식 없이도 유사한 손실 분포를 생성하도록 하이퍼파라미터를 선택할 수 있습니다. 실험 결과, KL-LiRA는 목표 모델의 하이퍼파라미터를 사용하는 경우와 거의 유사한 성능을 보이는 것으로 나타났습니다.

- **Performance Highlights**: 본 연구에서 제안한 KL-LiRA는 MIA 공격을 수행할 때 비하이퍼파라미터 최적화 방법보다 현저하게 개선된 성능을 보여주며, 비록 최적의 하이퍼파라미터를 찾지 못했다 하더라도 공격이 여전히 안정적으로 이루어질 수 있음을 확인했습니다. 또한, 하이퍼파라미터 최적화(HPO)를 위한 훈련 데이터 사용이 MIAs에 대한 취약성을 통계적으로 유의미하게 증가시키지 않음을 발견했습니다.



### AiRacleX: Automated Detection of Price Oracle Manipulations via LLM-Driven Knowledge Mining and Prompt Generation (https://arxiv.org/abs/2502.06348)
- **What's New**: 이 논문은 가격 오라클 조작(Price Oracle Manipulation) 감지를 자동화하는 새로운 LLM 기반 프레임워크를 제안합니다. 이 프레임워크는 다양한 LLM 모델의 상호 보완적인 강점을 활용하여, 가격 오라클의 취약성에 대한 도메인 특정 지식을 추출하고, 이를 구조적이고 맥락 인식이 가능한 체인 오브 쏘트(prompts)로 변환합니다. 기존의 수동 검증 방식의 한계를 극복하고, 개발자나 감사자의 깊은 전문 지식 없이도 효과적인 감지가 가능하도록 합니다.

- **Technical Details**: 이 프레임워크는 세 가지 LLM 모델, 즉 지식 합성기(Knowledge Synthesizer), 프롬프트 생성기(Prompt Generator), 감사자(Auditor)로 구성됩니다. 지식 합성기는 상위 학술 자료로부터 정확한 통찰력을 추출하여 오라클의 취약성 정보를 필터링합니다. 이후 프롬프트 생성기는 이 정보를 기반으로 구조적이고 유용한 프롬프트를 생성하여 감사자가 보다 정확하게 조작 패턴을 식별할 수 있도록 돕습니다.

- **Performance Highlights**: 이 연구는 2021년부터 2023년 사이의 60개의 알려진 취약성에 대해 실험을 수행하여, 제안된 프레임워크가 기존 도구 GPTScan에 비해 2.58배의 향상된 감지율을 기록했음을 밝혔습니다. 또한, 우리의 접근 방식은 개발자들이 도메인 지식 없이 문제 특정 프롬프트를 제작할 필요 없이 효율적으로 작업할 수 있도록 함으로써, 전체 작업 흐름을 간소화합니다.



### Facial Analysis Systems and Down Syndrom (https://arxiv.org/abs/2502.06341)
- **What's New**: 최근 몇 년 동안 얼굴 분석 기술(Facial Analysis Systems, FASs)에 대한 윤리적, 사회적, 법적 문제들이 활발히 논의되고 있습니다. 특히, 이러한 기술이 소외된 집단에 대한 편향과 차별을 지속할 수 있다는 비판이 대두되고 있습니다. 본 논문에서는 다운 증후군(Down syndrome) 인물의 얼굴과 관련된 데이터 세트를 활용해 FAS의 한계를 보고하고, 이 분야에서 간과된 취약 집단에 대한 새로운 증거를 제시합니다.

- **Technical Details**: 연구는 다운 증후군이 있는 인물과 없는 인물의 얼굴 이미지로 구성된 데이터 세트를 만들고, 두 가지 상업적 도구를 사용하여 성별 인식, 나이 예측 및 얼굴 레이블링 과제를 수행했습니다. 실험군은 다운 증후군이 있는 200명의 얼굴 이미지로 구성되었고, 대조군은 다운 증후군이 없는 200명의 얼굴 이미지로 구성되었습니다. 연구의 중심 질문은 다운 증후군이 있는 개인의 얼굴 인식을 FAS가 어떻게 수행하는지를 조사하는 것이었습니다.

- **Performance Highlights**: 결과적으로, 실험군의 얼굴 인식 예측 정확도가 전반적으로 낮았고, 다운 증후군이 있는 남성의 성별 인식에서 높은 오류율이 나타났습니다. 또한 성인이 다운 증후군을 가진 경우 아동으로 잘못 라벨링되는 빈도가 더 높았으며, 아름다움과 관련된 사회적 편견이 두 그룹 모두에서 나타났습니다. 이러한 발견은 다운 증후군을 가진 인물에 대한 얼굴 인식 기술의 구조적 한계를 강조합니다.



### DefTransNet: A Transformer-based Method for Non-Rigid Point Cloud Registration in the Simulation of Soft Tissue Deformation (https://arxiv.org/abs/2502.06336)
- **What's New**: 이 논문에서는 비탈성 포인트 클라우드 등록을 위한 새로운 Transformer 기반 아키텍처인 DefTransNet을 소개합니다. DefTransNet은 비정형 구조의 변형을 정확하고 견고하게 대응하기 위한 해결책으로 설계되었습니다. 특히, 소스 및 목표 포인트 클라우드를 입력으로 받아 변위 벡터 필드를 출력하며, 여러 데이터 세트를 통해 일반화 능력을 시험하였습니다.

- **Technical Details**: DefTransNet은 고유한 피처 설명자와 변위 벡터 필드를 학습하는 두 가지 주요 단계를 포함합니다. 이 모델은 변환에 대한 견고성을 강화하는 학습 가능한 변환 매트릭스를 통합하며, 전역 및 지역 기하학 정보 또한 고려하여 피처 집계를 진행합니다. Transformers의 자기 주의 메커니즘을 활용하여 포인트 간의 장기 의존성을 포착함으로써 전체적인 정보 흐름을 개선합니다.

- **Performance Highlights**: 실험 결과, DefTransNet은 다양한 난이도의 상황에서 현재의 최첨단 등록 네트워크보다 뛰어난 성능을 보였습니다. 연구진은 ModelNet, SynBench, 4DMatch, DeformedTissue를 포함한 네 가지 데이터 세트를 사용하여 이 방법의 효과성을 검증하였으며, 모든 데이터 세트에서 균일하게 높은 성능을 유지하는 것으로 나타났습니다.



### Prompt-Driven Continual Graph Learning (https://arxiv.org/abs/2502.06327)
Comments:
          12 pages, 7figures

- **What's New**: 이번 논문은 Continual Graph Learning (CGL)의 새로운 접근 방식인 PROMPTCGL을 소개합니다. 이 프레임워크는 각 작업에 대해 별도의 prompt를 학습하고, 기존의 graph neural network 모델을 고정 상태로 유지하여 이전 작업의 지식을 자연스럽게 보존합니다. 기존의 메모리 재생 기반 접근 방식의 한계를 극복하며, 메모리 사용량을 지속적으로 최소화하는 것을 목표로 합니다.

- **Technical Details**: PROMPTCGL은 기능 수준(feature-level) 및 토폴로지 수준(topology-level) 모두에서 모델을 지시하는 계층적 프롬프팅(hierarchical prompting) 기법을 활용합니다. 이를 통해 동적으로 변화하는 작업 그래프의 변동성을 효과적으로 처리할 수 있습니다. 또한, 개인화된 프롬프트 생성기(personalized prompt generator)를 개발하여 각 그래프 노드에 맞춤형 프롬프트를 생성하고, 이를 통해 메모리 소비를 지속적으로 일정하게 유지합니다.

- **Performance Highlights**: PROMPTCGL은 네 가지 벤치마크에서 기존의 CGL 접근 방식보다 우수한 성능을 보이며, 메모리 소비를 대폭 줄이는 결과를 나타냈습니다. 이러한 성과는 대규모 그래프에서도 높은 효율성을 유지할 수 있도록 설계된 PROMPTCGL의 특징에 기인합니다. 실험 결과는 이 프레임워크가 CGL 분야에서 혁신적인 기여를 할 것임을 보여줍니다.



### UniDemoiré: Towards Universal Image Demoiréing with Data Generation and Synthesis (https://arxiv.org/abs/2502.06324)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 범용 이미지 데모이레 솔루션인 UniDemoiré를 제안합니다. 이 모델은 다양한 형태의 모이레 패턴을 자동으로 생성하여 데모이레링 모델의 일반화 능력을 향상시킵니다. 특히, 새로운 데이터 생성 및 합성 방법을 통해 대량의 고품질 모이레 이미지를 생성할 수 있습니다.

- **Technical Details**: UniDemoiré는 대규모 모이레 패턴 데이터셋을 활용하여 다양한 패턴을 생성합니다. 기본적으로, 이미지 내용과는 무관한 순수 모이레 패턴을 캡쳐하여 기존의 데이터 수집 방법의 한계를 극복합니다. 또한, 생성된 모이레 패턴과 깨끗한 자연 이미지를 혼합하여 현실감 있는 모이레 이미지를 합성하는 방법도 제안합니다.

- **Performance Highlights**: 제안된 UniDemoiré 모델은 제로샷(Zero-shot) 이미지 데모이레링과 교차 도메인 평가에서 우수한 성능을 보여줍니다. 본 연구는 일반화 능력이 뛰어난 모이레 제거 모델을 위한 충분한 양의데이터를 제공하였으며, 이는 모이레 패턴의 다양성을 크게 향상시킵니다.



### From Pixels to Components: Eigenvector Masking for Visual Representation Learning (https://arxiv.org/abs/2502.06314)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 이미지의 가시 부분에서 마스킹된 부분을 예측하는 방법으로, 기존의 픽셀 기반 마스킹 전략 대신 데이터 변환을 바탕으로 하는 새로운 마스킹 전략을 제안합니다. 주성분 분석(principal component analysis, PCA)을 활용하여 이미지의 주성분을 마스킹하는 방식으로, 이는 이미지의 전역 정보를 더 잘 반영할 수 있는 특징을 지닙니다. 이 접근 방식은 특정 마스킹 비율에 따라 변동성을 조절할 수 있어, 더 의미 있는 표현학습이 가능할 것으로 기대됩니다.

- **Technical Details**: 제안된 방법은 주성분 분석을 통해 변환된 데이터에서 특정 주성분을 무작위로 마스킹하고, 남은 가시 컴포넌트로부터 마스킹된 컴포넌트를 재건하는 방식입니다. 이 과정에서 마스킹된 변동 비율이 모델링 작업의 복잡성을 나타내며, 이는 보다 해석 가능하고 조정이 용이한 하이퍼파라미터로 작용합니다. 또한, 이러한 기법은 이미지를 전역 특성으로 나누어 정보의 중복 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 실험 결과, CIFAR10, TinyImageNet, 그리고 세 가지 의료 데이터셋(MedMNIST)에서 저자들의 방법(주성분 마스크 오토인코더, PMAE)이 기존의 공간 마스킹 방법(MAE)보다 우수한 성능을 나타냈습니다. 특히, 마스킹 비율 하이퍼파라미터의 선택에 덜 민감하다는 점에서도 장점을 보였습니다. 이러한 결과는 주성분 마스킹이 더 의미 있는 고수준의 표현학습을 촉진할 수 있음을 뒷받침합니다.



### SeaExam and SeaBench: Benchmarking LLMs with Local Multilingual Questions in Southeast Asia (https://arxiv.org/abs/2502.06298)
Comments:
          Accepted to Findings of NAACL 2025

- **What's New**: 이번 연구는 동남아시아 (SEA) 애플리케이션 시나리오에서 대형 언어 모델 (LLMs)의 능력을 평가하기 위해 설계된 두 가지 새로운 벤치마크인 SeaExam과 SeaBench를 소개합니다. 기존의 다국어 데이터셋이 영어 번역에 기반하고 있는 것과 달리, 이 벤치마크는 SEA 지역의 실제 시나리오를 기반으로 구축되었습니다. SeaExam은 지역 교육 시험에서 추출한 데이터를 사용하여 지역 역사 및 문학과 같은 다양한 과목을 포괄합니다.

- **Technical Details**: SeaExam은 SEA 국가의 실제 시험에서 출처를 얻은 다치기 과제 시험 데이터셋이며, 지역 역사, 지리 및 문학을 다룹니다. 반면, SeaBench는 10개 과제 범주에 걸쳐 하루 대화에서 흔히 마주치는 시나리오와 지침을 포함한 다회전, 개방형 과제를 중심으로 제작되었습니다. 이러한 접근법은 SEA 맥락에서의 실제 사용을 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 실험 분석은 SeaExam과 SeaBench가 번역된 벤치마크보다 SEA 언어 작업에서 LLM의 성능을 한층 더 정교하게 구별할 수 있음을 보여줍니다. 또한, 개방형 질문이 다국어 사용에서 모델 성능의 차이를 강조하는 데 더욱 효과적이라는 것을 발견했습니다. 나아가, 안전성 범주에서 아홉 개 모델이 전반적으로 낮은 성과를 보였으며, 이는 다국어 애플리케이션에 있어 더 나은 안전 조치가 필요함을 시사합니다.



### Is an Ultra Large Natural Image-Based Foundation Model Superior to a Retina-Specific Model for Detecting Ocular and Systemic Diseases? (https://arxiv.org/abs/2502.06289)
- **What's New**: 이번 연구에서는 의료 분야에서 기본 모델(foundation models, FMs)이 어떻게 변화를 가져오고 있는지를 다루고 있습니다. 특히, RETFound라는 망막 특화 모델과 DINOv2라는 범용 비전 모델을 비교하여 안과 질환 탐지와 시스템 질환 예측 작업에서의 성능을 평가했습니다. 이 연구는 DINOv2의 임상 과제에 대한 적용 가능성을 새롭게 조명합니다.

- **Technical Details**: 연구팀은 1.4백만 개의 자연 이미지와 1.6백만 개의 망막 이미지를 순차적으로 사전 훈련한 RETFound와 1.42억 개의 자연 이미지로 사전 훈련한 DINOv2 모델(대형, 기본, 소형)을 비교했습니다. 이들은 8개의 표준화된 공개 안과 데이터셋과 Moorfields AlzEye 및 UK Biobank 데이터셋에서 성능을 평가했습니다. 특히, DINOv2-large 모델은 당뇨망막병증 감지에서 RETFound보다 우수한 성능을 보였습니다.

- **Performance Highlights**: DINOv2-large 모델은 당뇨망막병증(0.850-0.952)과 다중 클래스 안 질환(0.892) 탐지에서 RETFound보다 뛰어난 성능을 기록했습니다. 반면, RETFound는 심부전, 심근경색, 허혈성 뇌졸중 예측에서 모든 DINOv2 모델보다 우수한 성능을 나타냈습니다. 이러한 결과는 특정 업무에 대한 FM 선택이 임상 성능 최적화에 얼마나 중요한지를 잘 보여줍니다.



### End-to-End Multi-Microphone Speaker Extraction Using Relative Transfer Functions (https://arxiv.org/abs/2502.06285)
- **What's New**: 이 논문은 다중 마이크로폰 환경에서 여러 화자가 동시에 존재할 때 원하는 화자를 효과적으로 추출하는 방법을 제안합니다. 특히, 해당 연구에서는 상대 전송 함수(Instantaneous Relative Transfer Function, RTF)를 기반으로 한 공간 단서를 활용하여 성능을 개선하고자 합니다. 실험 결과, RTF 기반의 방법이 기존의 방향각(Direction of Arrival, DOA) 방법보다 뛰어난 성능을 나타났습니다.

- **Technical Details**: 문제는 짧은 시간 푸리에 변환(Short-Time Fourier Transform, STFT) 도메인에서 다루어지며, Q개의 활성 화자가 J개의 마이크로폰에 의해 포착된 환경을 모델링합니다. 이 문제를 해결하기 위해, RTF와 관련된 다양한 매개변수를 설정하고, 원하는 화자의 발화를 특정하는 다양한 입력 신호를 강화하며 최적화를 진행합니다. 모든 입력 신호는 특정 위치에서 수집된 신호로 하여금 필요한 공간 단서를 제공합니다.

- **Performance Highlights**: 제안된 방법은 기존의 스펙트럼 기반 단서 및 DOA 기반 단서와 비교하여 폭넓은 실험을 통해 일관되게 더 우수한 성능을 보여주었습니다. 실험 데이터는 복잡한 음향 환경에서도 RTF 기반 특징이 DOA 기반 특징보다 효과적임을 증명합니다. 특히, 제안된 방법은 최소 분산 왜곡 빔포머(Minimum Variance Distortion Beamformer, MVDR)와의 비교에서도 우수한 결과를 나타내었습니다.



### Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE (https://arxiv.org/abs/2502.06282)
- **What's New**: 이 논문은 Speculative Decoding의 새로운 접근 방식인 Jakiro를 제안합니다. Jakiro는 Mixture of Experts(MoE) 메커니즘을 활용하여 토큰 예측의 다양성을 높이고, 각 단계에서 독립적인 전문가들이 예측을 생성합니다. 또한, 자가 회귀 방식과 병렬 디코딩 방법을 결합하여 성능을 향상시킵니다.

- **Technical Details**: Jakiro의 핵심은 전통적인 후보 생성 방식에서의 상관관계를 제거하고, MoE를 통해 각 레이어에서 독립적인 예측을 돕는 것입니다. 논문의 방법론은 다양한 모델에 대해 창출된 후보 토큰들이 서로 독립적이라는 점에서 중요한 장점이 있습니다. 이러한 접근법은 SOTA(State of the Art) 성능을 미치는 병렬 디코딩 전략도 포함하고 있습니다.

- **Performance Highlights**: 대규모 실험을 통해 Jakiro는 기존의 최첨단 기법들보다 우수한 성능을 보여주며, MT-bench에서 비탐욕적 모드에서의 눈에 띄는 발전을 달성했습니다. 이 방법은 예측 정확도를 크게 향상시키고 인퍼런스 속도를 높이며, 다양한 모델과 벤치마크에서 그 효율성과 강인성을 검증했습니다.



### HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilanc (https://arxiv.org/abs/2502.06274)
- **What's New**: 본 논문은 다중 약물 복합 요법에서 발생하는 부작용에 대한 연구의 중요성을 강조하며, 이를 위해 처음으로 고차 약물-약물 상호작용 데이터셋인 HODDI를 제안합니다. HODDI는 미국 식품의약국(FDA)의 Adverse Event Reporting System (FAERS) 데이터를 기반으로 하여 과거 10년간의 기록을 포함하고 있으며, 2,506개의 고유 약물과 4,569개의 고유 부작용을 포괄합니다. 이번 연구는 다중 약물 상호작용이 부작용에 미치는 집합적 영향을 포착하는 데 중점을 둡니다.

- **Technical Details**: HODDI 데이터셋은 데이터 정리와 조건부 필터링을 통해 동시 투여된 약물 사례를 분석하여 부작용에 대한 조합적 영향을 연구할 수 있게 해줍니다. 또한, 여러 모델을 통해 HODDI의 성능을 평가한 결과, 단순한 Multi-Layer Perceptron (MLP) 모델도 고차 정보를 활용하여 강력한 성능을 발휘할 수 있음을 보여주었습니다. 특히, 하이퍼그래프 구조를 통합한 모델이 더 복잡한 다중 약물 관계를 효과적으로 캡처하여 예측 정확도를 향상시킬 수 있다는 점이 강조됩니다.

- **Performance Highlights**: HODDI는 높은 커버리지와 강력한 분석 지표를 바탕으로 고차 약물-부작용 관계 연구에 유용한 자원으로 자리 잡고 있습니다. 이번 연구를 통해 제안된 하이퍼그래프 모델이 약물-부작용 예측의 정확성을 높이는 데 기여함을 확인하였으며, HODDI는 약물 안전성 및 개인 맞춤 의학 연구를 촉진하는 기준 데이터셋으로 자리매김할 가능성을 보여줍니다.



### K-ON: Stacking Knowledge On the Head Layer of Large Language Mod (https://arxiv.org/abs/2502.06257)
Comments:
          AAAI 2025 (Oral)

- **What's New**: 본 논문에서는 K-ON이라는 새로운 모델을 제안하여 대규모 언어 모델(LLM) 내에 지식 그래프(KG)의 지식을 통합한다. K-ON은 여러 헤드 레이어(multiple head layers)를 통해 다음 k단계 예측(next k-step prediction)을 수행하며, 이는 엔티티 수준의 결과를 한 단계에서 생성할 수 있게 해준다. 또한, K-ON은 KG 표현 학습에서 가장 강력한 도구인 대조 손실(contrastive loss)을 가능하게 한다.

- **Technical Details**: K-ON은 KG 내의 엔티티를 효과적으로 예측하기 위해 K개의 서로 다른 헤드 레이어를 활용한다. 각 헤드는 모든 엔티티에 대한 k-th 토큰 예측을 담당하며, 이를 통해 엔티티를 하나의 통합된 예측으로 처리한다. 또한, K-ON은 헤드 궤적 조정(head trajectory tuning, HTT)이라는 방법을 통해 토큰 예측의 분포(distribution)를 정렬하여 원래 LLM의 예측 성능을 유지할 수 있도록 한다.

- **Performance Highlights**: K-ON은 KG 완성 과제에서 기존 방식보다 우수한 성능을 보여주었으며, 텍스트와 시각 정보를 추가적으로 활용하는 다중 모달(multi-modal) 방법보다도 더 나은 성과를 기록했다. 또한, K-ON은 GPU 자원을 더 요구하지만 훈련 에포크 수를 기존 1,000에서 5로 줄여 훈련 시간을 대폭 단축하였다. DB15K 데이터셋에서의 전체 미세 조정(fine-tuning) 시간은 1시간 11분 이하로 관리할 수 있었다.



### Towards Efficient and Intelligent Laser Weeding: Method and Dataset for Weed Stem Detection (https://arxiv.org/abs/2502.06255)
Comments:
          Accepted by AAAI-AISI 2025

- **What's New**: 본 연구는 레이저 제초를 위한 잡초 인식을 최초로 실증 조사한 논문으로, 환경 친화적이고 효율적인 잡초 관리 방법을 제시합니다. 새로운 시스템은 작물의 손상을 피하면서도 레이저빔을 잡초 뿌리에 직접 겨냥할 수 있도록 설계되었습니다. 이를 위해 11,151개의 잡초 인스턴스가 주석으로 달린 고품질 잡초 줄기 탐지 데이터셋이 구축되었습니다.

- **Technical Details**: 이 연구에서는 작물 및 잡초 탐지와 함께 잡초 줄기 위치 추적을 통합한 일관된 end-to-end 시스템을 도입했습니다. 시스템은 이미지 시퀀스 또는 실시간 비디오 스트림을 처리하여 잡초 줄기를 정확히 찾을 수 있습니다. 이는 에너지 효율성을 높이고 작물에 대한 피해를 줄이기 위한 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존의 잡초 인식 시스템에 비해 잡초 제거 정확도를 6.7% 향상시켰으며 에너지 비용은 32.3% 감소했습니다. 이는 레이저 제초의 실용적 효율성을 크게 향상시키며, 지속 가능한 농업 관리에서 중요한 기여를 할 것으로 기대됩니다.



### Conditioning through indifference in quantum mechanics (https://arxiv.org/abs/2502.06249)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 양자 시스템의 상태에 대한 불확실성을 측정 값에 조건화하여 설명하는 방법을 다룹니다. 우리 연구의 주요 초점은 특정 측정 결과에 따라 양자 상태에 대한 불확실성을 어떻게 표현할 수 있는지를 이해하는 것입니다. 우리는 이전의 논문에서 도출한 유용성(utility) 함수와 일반적인 조건화 규칙을 활용하여 이러한 상태를 분석합니다. 이는 기존의확률론적 접근을 넘어서는 차원에서의 관점을 제공합니다.

- **Technical Details**: 논문에서는 양자 시스템의 상태를 설명하기 위해 Hermitian operator와 유용성 함수의 조합을 사용합니다. 시스템의 상태|Ψ⟩에 대해, 주어진 측정 A에 대해 측정 결과가 가져올 수 있는 유용성을 uA^(|Ψ⟩)로 정의합니다. 이때, 모든 불확실성(information)은 불확실한 보상 간의 선호를 통해 모델링됩니다. 특정 측정 결과에 따라 새로운 정보가 어떻게 업데이트되는지를 다루는 섹션에서는 이를 '욕구(desirability)'와 '무관심(indifference)'의 힘을 통해 확장하는 방법이 소개됩니다.

- **Performance Highlights**: 본 연구는 기존의 Benavoli의 조건화 규칙을 바탕으로 하여 더 넓은 범위의 해석을 제안합니다. 이러한 조건화 규칙은 측정 결과 집합에도 적용되며, 이는 긍정적 연산자 값 측정(positive operator valued measures)에 대한 첫 걸음이 됩니다. 양자 컴퓨팅 및 양자 암호화와 같은 응용 분야에 대한 이론적 기여가 가능합니다. 또한, 이 모델은 비확률적 접근으로 양자역학의 불확실성을 보다 효과적으로 처리하는 방법을 제안합니다.



### Confidence Improves Self-Consistency in LLMs (https://arxiv.org/abs/2502.06233)
- **What's New**: 이 논문은 Confidence-Informed Self-Consistency (CISC)를 제안하여 LLM의 추론 능력을 향상시킵니다. CISC는 모델에서 직접 얻은 confidence score를 사용하여 가중 다수결(voting)을 수행하고, 이를 통해 정답을 보다 적은 샘플로 찾아낼 수 있습니다. 연구 결과 CISC는 기존의 self-consistency 방법보다 평균 40% 이상의 reasoning paths 감소와 더불어 거의 모든 구성에서 성능 우위를 보여줍니다. 또한, LLM이 자신의 출력의 정답성을 평가할 수 있는 능력에 대한 실증적 증거를 제공합니다.

- **Technical Details**: CISC는 self-assessment score를 생성하여 각 reasoning path의 confidence를 기반으로 가중 투표를 수행하는 방식을 사용합니다. 예를 들어, 정확한 답변을 60%의 확률로 제공하는 경우, 전통적인 다수결 방법은 90% 정확도를 달성하기 위해 40개의 샘플이 필요하지만, CISC는 정확한 답변을 두 배로 가중치 부여하여 10개 미만의 샘플로 동일한 정확도에 도달할 수 있습니다. 또한, Within-Question Discrimination (WQD) 메트릭을 제안하여 동일한 질문에 대한 정답을 구분하는 데 유용성을 평가할 수 있습니다.

- **Performance Highlights**: CISC는 다양한 LLM 및 데이터셋을 사용한 테스트에서 self-consistency보다 우수한 성능을 보여주었습니다. 최상의 confidence estimation 방법을 사용할 경우, CISC는 self-consistency와 유사한 성능을 발휘하며 필요한 reasoning paths의 수를 평균 40% 이상 줄일 수 있습니다. 마지막으로, 모델이 스스로 판단하는 confidence score와 인간 평가자 간의 합의가 뚜렷하게 나타나는 것을 보여주었습니다.



### Examining False Positives under Inference Scaling for Mathematical Reasoning (https://arxiv.org/abs/2502.06217)
- **What's New**: 최근의 언어 모델(Language Models) 발전은 여러 벤치마크에서 수학적 추론(Mathematical Reasoning) 능력을 크게 향상시켰습니다. 하지만 대부분의 벤치마크는 최종 답변만을 비교하는 자동 평가 방법에 의존해, 근본적인 추론 단계를 검증하지 않는 한계가 있습니다. 이로 인해 올바른 최종 답변을 내놓더라도 잘못된 추론 경로를 가진 false positive 솔루션이 발생합니다.

- **Technical Details**: 본 연구에서는 언어 모델을 활용한 수학 문제 해결에서 false positive 솔루션의 발생 빈도를 체계적으로 조사하였습니다. 다양한 오픈 소스 모델과 난이도 수준이 다른 데이터셋, 디코딩 전략을 통해 문제의 특성과 범위를 분석하였습니다. 실험 결과, false positive 솔루션은 다양한 모델, 데이터셋, 및 디코딩 방법에서 여전히 존재하며, sampling-based inference time scaling 방법이 문제를 해결하지 못한다는 것을 확인했습니다.

- **Performance Highlights**: pass@N 평가 메트릭은 false positives에 더 민감하여 자동 평가가 제시하는 것보다 훨씬 낮은 스케일링 한계를 갖고 있음이 확인되었습니다. 또한, 구체적인 false positive 사례를 분석하고 이러한 조건에서의 self-improvement techniques 및 synthetic data generation의 잠재적 한계에 대해 논의합니다.



### LessLeak-Bench: A First Investigation of Data Leakage in LLMs Across 83 Software Engineering Benchmarks (https://arxiv.org/abs/2502.06215)
Comments:
          25 pages

- **What's New**: 본 논문은 대규모 소프트웨어 엔지니어링(Software Engineering, SE) 벤치마크에서 대형 언어 모델(Large Language Models, LLM)에 대한 데이터 유출(data leakage) 문제를 최초로 분석합니다. 83개의 SE 벤치마크에 대한 대규모 조사를 통해, Python, Java, C/C++ 각각의 평균 유출 비율이 4.8%, 2.8%, 0.7%로 나타나 데이터 유출 문제가 미미함을 강조합니다. 그러나 QuixBugs와 BigCloneBench는 각각 100.0%와 55.7%의 높은 유출 비율을 기록하여 평가의 편향 가능성을 지적합니다.

- **Technical Details**: 이 연구는 LLM의 성능에 대한 데이터 유출 영향을 분석하기 위해 DetectLeak라는 다단계 접근 방식을 제안합니다. 이 방법은 MinHash+LSH라는 자동화 도구를 활용하여 약 1.7조 쌍의 LLM의 사전 훈련 데이터와 SE 벤치마크 데이터를 비교하여 잠재적 중복 쌍을 식별합니다. 이후 숙련된 개발자들이 이 중복 쌍을 수동으로 레이블링하여 실제 중복 및 데이터 유출을 확인합니다.

- **Performance Highlights**: 연구 결과, StarCoder-7b는 APPS 벤치마크에서 유출 샘플에서 Non-leaked 샘플보다 4.9배 높은 Pass@1 점수를 기록하였습니다. 이는 유출된 벤치마크 샘플의 존재가 성능 지표를 크게 부풀릴 수 있음을 보여줍니다. 이 연구의 결과는 SE 벤치마크의 데이터 유출 문제가 LLM의 평가에 미치는 중요한 영향을 강조하며, 향후 보다 신뢰할 수 있는 LLM 평가를 위한 LessLeak-Bench라는 새로운 벤치마크를 제안합니다.



### Unveiling the Capabilities of Large Language Models in Detecting Offensive Language with Annotation Disagreemen (https://arxiv.org/abs/2502.06207)
Comments:
          17 pages, submitted to the ACL 2025

- **What's New**: 이번 논문에서는 인간 주석 간 불일치(annotation disagreement)를 고려한 공격적인 언어 감지의 새로운 접근 방식을 제시합니다. 기존 연구는 공격적인 언어를 명확한 이진 레이블로 단순화하여 오류를 범하고 있으며, 이로 인해 실제 데이터셋의 복잡성을 간과하고 있었습니다. 연구자들은 LLMs(large language models)가 불일치 샘플을 어떻게 처리하는가에 대한 구체적인 평가를 실시하여, 인공지능의 신뢰성과 결정 과정의 복잡성을 탐구합니다.

- **Technical Details**: 이 연구의 핵심 데이터셋인 MD-Agreement는 트위터에서 수집된 10,753개의 샘플로, 높은 품질의 주석을 보장합니다. 각 샘플은 5명의 훈련된 주석자에 의해 주석이 달리며, 다수결에 따라 하드 라벨과 함께 주석 동의 정도를 나타내는 소프트 라벨도 제공합니다. 실험에서는 다양한 파라미터 크기를 가진 LLM을 사용하여 공격적인 언어를 감지하는 성능을 평가하고, 몇 가지 주류 기술인 few-shot learning과 instruction fine-tuning에 미치는 영향을 분석합니다.

- **Performance Highlights**: LLMs의 공격적인 언어 감지 성능은 우리의 평가에서 예상보다 훨씬 뛰어난 것으로 나타났습니다. 실험에 사용된 폐쇄형 모델의 이진 정확도 평균은 88.28%, 오픈 소스 모델은 86.07%였으며, 이는 동의가 높은 샘플들에서 특히 강력한 성능을 보였습니다. 그러나 불일치 샘플이 포함된 경우, LLM의 결정 신뢰도가 감소하여 이 분야에서 더 많은 연구가 필요함을 시사합니다.



### C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation (https://arxiv.org/abs/2502.06205)
Comments:
          Ongong work

- **What's New**: 본 논문에서는 Retrieval-augmented generation (RAG) 시스템의 기본적인 문제인 독립적으로 개발된 retriever와 대형 언어 모델(LLMs) 간의 정렬 문제를 다룹니다. 기존 접근 방식은 컴포넌트를 수정하거나 간단한 중간 모듈을 도입하는 방식이었으나, 이는 실제적인 제한과 최적화되지 않은 성능을 초래했습니다. 새로운 프레임워크인 C-3PO는 경량 멀티 에이전트 시스템을 통해 retrievers와 LLMs 간의 효과적인 소통을 지원합니다.

- **Technical Details**: C-3PO는 세 가지 전문화된 에이전트를 구현하여 전체 RAG 파이프라인을 공동으로 최적화합니다. 이 아키텍처는 retriever와 LLMs를 변경하지 않고, 정보를 선택하고 효과적인 쿼리를 생성하며, retrieval의 필요성을 평가합니다. 또한 강화 학습의 보상 기여 할당을 위한 트리 구조의 rollout 접근 방식을 개발하여 멀티 에이전트 간의 효과적인 조정을 가능하게 합니다.

- **Performance Highlights**: 다양한 도메인 내 및 도메인 외 시나리오에서 수행한 광범위한 실험 결과, C-3PO는 RAG 성능을 대폭 향상시킴을 입증했습니다. 이 프레임워크는 플러그 앤 플레이(plug-and-play) 유연성을 유지하며, 뛰어난 일반화 능력도 보장합니다.



### Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering (https://arxiv.org/abs/2502.06193)
Comments:
          Accepted by ISSTA 2025

- **What's New**: 최근 대형 언어 모델(LLMs)이 코드 생성과 같은 다양한 소프트웨어 엔지니어링(SE) 작업을 자동화하는 데 사용되며, 이로 인해 SE 작업의 효율성이 크게 향상되었습니다. 그러나 LLM이 생성한 코드와 텍스트의 품질 평가에는 여전히 도전 과제가 존재합니다. 기존의 Pass@k 메트릭은 고도의 노동 비용이 들고, 일반적으로 LLM으로 생성된 텍스트를 평가하는 데 적합하지 않으므로 새로운 평가 방법론이 필요합니다.

- **Technical Details**: 이 연구는 LLM-as-a-judge 방법을 활용하여 SE 작업의 품질 평가와 인간의 판단과의 일치를 실증적으로 탐구합니다. 연구진은 7개의 LLM-as-a-judge 방법을 선정하고, 세 가지 SE 데이터세트에 대한 LLM 응답을 생성한 뒤 수동으로 점수를 매깁니다. 이후 각 응답에 대해 이 평가 방법들을 적용하여 인간 평가와의 점수 상관관계를 비교합니다.

- **Performance Highlights**: 연구 결과, 출력 기반 방법이 코드 번역과 생성에서 인간 점수와의 Pearson 상관관계에서 각각 81.32 및 68.51로 가장 높은 성과를 보였습니다. 이는 ChrF++와 같은 기존 메트릭을 각각 34.23 및 64.92로 크게 능가하는 수치입니다. 이러한 성과는 LLM이 인간 평가 패턴을 유사하게 반영하는 보다 균형 잡힌 점수 분포를 나타냅니다.



### Right Time to Learn:Promoting Generalization via Bio-inspired Spacing Effect in Knowledge Distillation (https://arxiv.org/abs/2502.06192)
- **What's New**: 이 논문은 지식 증류(knowledge distillation, KD)의 새로운 접근법인 Spaced KD를 제안합니다. Spaced KD는 일종의 시간 간격(spacing effect)을 이용하여 온라인 KD(online KD)와 자기 KD(self KD)의 성능을 향상시킵니다. 이 방법은 생물학적 학습에서 발견된 spacing effect 이론에 영감을 받았으며, 실험을 통해 DNN의 일반화 성능을 크게 개선하는 것을 입증했습니다.

- **Technical Details**: 연구진은 Spaced KD를 통해 DNN이 SGD(확률적 경량화 경량화) 중에서 더 평탄한 손실 함수(loss landscape)로 수렴하도록 유도합니다. 이 평탄한 손실 함수는 일반화(generalization)와 밀접한 연관이 있으며, Spaced KD는 추가적인 훈련 비용 없이도 성능 향상을 달성할 수 있습니다. 여러 벤치마크 데이터셋과 네트워크 아키텍처에서 광범위한 실험을 수행하여 그 효과를 입증했습니다.

- **Performance Highlights**: Spaced KD는 Tiny-ImageNet에서 온라인 KD와 자기 KD에 대해 각각 최대 2.31% 및 3.34%의 성능 향상을 기록하였습니다. 이는 기존 방법보다 현저한 성능 개선을 나타내며, 연구진은 다양한 KD 방법에 대해 Spaced KD의 견고성과 범용성을 강조했습니다. 실험 결과는 Spaced KD의 적용이 DNN의 전반적인 학습 성능을 개선할 수 있음을 보여줍니다.



### Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization (https://arxiv.org/abs/2502.06185)
Comments:
          NAACL 2025 camera-ready version

- **What's New**: 이번 논문에서는 긴 문서 요약의 사실 불일치(factual inconsistency) 문제를 다루고 있으며, 특히 이 문제를 담담하는 담화 분석(discourse analysis)과의 연계를 탐구합니다. 연구 결과에 따르면, 복잡한 문장에서 사실 불일치 오류가 더 자주 발생하며, 이는 여러 담화 특징들과 관련이 있다는 것을 발견했습니다. 또한, 담화에 기반한 정보를 활용한 새로운 평가 방법론인 StructScore를 제안하여 긴 문서 요약의 사실 불일치 검출 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 StructScore는 두 가지 단계로 구성됩니다: 첫째, 목표 요약의 문장 수준 정렬 점수를 집계할 때 담화 정보를 활용하고, 둘째, 긴 입력 기사를 여러 담화에서 영감을 받은 청크로 분해합니다. 이를 통해 긴 문서 요약에서 NLI 기반 접근 방식을 활용하여 보다 효과적으로 사실 불일치를 탐지할 수 있습니다. 실험은 AggreFact-FtSOTA, DiverSumm, LongSciVerify, LongEval 등 여러 문서 요약 벤치마크에서 진행되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 다양한 작업에서 성능 향상을 보여주었으며, 모델과 모델 출력은 공개적으로 제공됩니다. 본 연구는 긴 문서 요약의 사실 불일치 문제 해결을 위한 담화 구조와 사실성 평가 사이의 관계를 최초로 조사한 연구로, 앞으로 이 분야의 연구에 중요한 기여를 할 것으로 보기됩니다.



### RideKE: Leveraging Low-Resource, User-Generated Twitter Content for Sentiment and Emotion Detection in Kenyan Code-Switched Datas (https://arxiv.org/abs/2502.06180)
Comments:
          Accepted in WASSA 2024

- **What's New**: 이 논문은 케냐에서의 코드 스위칭 언어에 대한 감정 및 감정 분류를 위해 사전 훈련된 최신 변환기(transformer) 모델을 평가하고, 그 방법론과 데이터 수집 및 주석 작업 중 발생한 문제를 설명합니다. RideKE라는 데이터셋을 소개하며, 이 데이터셋은 29,000개 이상의 트윗을 포함하고 있으며 감정이 긍정적, 부정적, 중립적으로 분류됩니다. 따라서 이 연구는 저자원(low-resource) 언어에서의 자연어 처리(NLP) 향상에 기여하는 것을 목표로 하고 있습니다.

- **Technical Details**:  케냐의 다언어적 특성을 반영한 본 연구에서는 영어, 스와힐리어 및 셍 언어가 혼합된 감정 데이터셋을 분석합니다. XLM-R, DistilBERT, mBERT 및 AfriBERTa와 같은 사전 훈련된 모델들이 사용되며, 이들 모델들은 감정 분석 및 감정 분류에 대한 성능 비교를 통해 저자원 언어의 NLP 성능을 평가합니다. 특히, 이 논문에서는 감정 분석을 위해 감독 학습(supervised learning) 및 반감독 학습(semi-supervised learning)을 활용합니다.

- **Performance Highlights**: 연구 결과에 따르면 XLM-R 모델이 감정 분석에서 최고의 정확도(69.2%)와 F1 점수(66.1%)를 기록하며, 정서 분석에서도 DistilBERT가 59.8%의 정확도를 보였습니다. 모든 모델은 중립적인 감정을 예측하는 경향이 있으며, AfriBERTa는 가장 낮은 정확도와 F1 점수를 나타냈습니다. 연구는 다양한 모델의 성능과 저자원 언어에서의 감정 인식의 가능성을 제시합니다.



### Uncertainty-Aware Adaptation of Large Language Models for Protein-Protein Interaction Analysis (https://arxiv.org/abs/2502.06173)
- **What's New**: 이번 연구에서는 단백질-단백질 상호작용(PPIs) 분석을 위한 불확실성 인식(uncertainty-aware) LLMs의 적응을 제안한다. LLaMA-3 및 BioMedGPT 모델을 고도화하여 특정 질병 맥락에서 예측신뢰성을 향상시키고, LoRA 앙상블 및 Bayesian LoRA 모델을 통합하여 불확실성 정량화(uncertainty quantification)를 수행한다. 이러한 방법을 통해 PPIs 식별 성능을 향상시키고, 생물정보학의 재현성(reproducibility) 문제를 해결하려 한다.

- **Technical Details**: 단백질-단백질 상호작용(PPIs)은 세포 기능의 분자적 기초를 나타낸다. 본 연구에서는 LoRA 기반의 미세调정(fine-tuning)을 사용하고, Bayesian LoRA 및 LoRA 앙상블 방법을 채택하여 전염성 예측을 개선한다. 이를 통해 PPI 네트워크에 관한 광범위한 불확실성 인식 평가를 수행하며, 신경퇴행성 질환, 대사 질환 및 암에 관련된 단백질 상호작용 네트워크를 분석한다.

- **Performance Highlights**: 우리의 접근 방식은 치명적인 생물 의학적 응용에서 발생할 수 있는 불확실성 문제를 해결하며, PPI 예측 정확도를 높인다. 또한, 잘 조정된 신뢰도 측정을 제공하여, 생물 의학 연구의 강력한 결론 도출을 가능하게 한다. 본 연구는 LLM 기반 모델링에서 안전하고 신뢰할 수 있으며 정보가 풍부한 계산 도구의 개발을 위한 토대를 마련한다.



### An Interpretable Implicit-Based Approach for Modeling Local Spatial Effects: A Case Study of Global Gross Primary Productivity (https://arxiv.org/abs/2502.06170)
- **What's New**: 본 연구에서는 지리적 머신러닝에서 공간 이질성을 효과적으로 모델링하기 위한 새로운 접근법을 제시합니다. 전통적인 통계 학습 방법이 만족스럽지 못한 정확도를 보이는 문제를 해결하기 위해, 서로 다른 위치 간의 공통 기능과 공간적 차이를 동시에 모델링하는 심층 신경망 구조를 도입합니다. 이러한 방법은 일반적인 특징을 추출하면서도 지역적 변화를 반영할 수 있습니다.

- **Technical Details**: 제안된 방법은 인코더-디코더 구조를 갖춘 이중 분기 신경망으로 구성됩니다. 인코딩 단계에서는 그래프 컨볼루션 네트워크(GCN)와 장단기 메모리 네트워크(LSTM)를 사용하여 시공간 조건 그래프에서 노드 정보를 집계하며, 특정 위치의 시공간 이질성을 암묵적인 조건 벡터로 인코딩합니다. 디코딩 단계에서는 조건 생성 전략을 사용하여 응답 변수와 해석적 가중치를 예측합니다.

- **Performance Highlights**: 자체 검증을 위해 2001년부터 2020년까지의 글로벌 기후 및 토지 덮개 데이터를 활용하여 식물총생산성(GPP)을 예측하였습니다. 모델은 5천만 개의 샘플로 학습하고 280만 개의 샘플로 테스트하여 RMSE 0.836을 달성했으며, 기존의 LightGBM(1.063) 및 TabNet(0.944)을 능가하는 성능을 보였습니다. 시각화 분석을 통해 GPP의 주요 요인 분포 차이를 다양한 시간과 위치에서 드러낼 수 있음을 확인했습니다.



### Universal Approximation of Visual Autoregressive Transformers (https://arxiv.org/abs/2502.06167)
- **What's New**: 이번 논문에서는 transformer 기반의 foundation model의 근본적인 한계를 조사하며, Visual Autoregressive (VAR) transformer를 포함한 새로운 분석을 제시합니다. VAR는 이미지 생성을 위한 새로운 조정 가능한 코스-투-파인 'next-scale prediction' 프레임워크를 통해 기존 방법들보다 우수한 품질을 보여줍니다. 우리의 주요 기여는 단일 헤드 VAR transformer가 이미지-투-이미지 Lipschitz 함수에 대한 보편적인 근사자임을 증명하는 것입니다.

- **Technical Details**: Transformer 기반 아키텍처는 현대 기계 학습의 경관을 변화시켰으며, self-attention 메커니즘을 통해 데이터의 장기 종속성을 효과적으로 모델링합니다. VAR transformer는 구조화된 이미지 합성을 위해 적응된 변형으로, 높은 품질의 이미지를 더 효율적으로 생성합니다. 이 연구는 VAR transformer와 Flow AutoRegressive (FlowAR) 모델의 보편성에 대한 정량적 분석을 통해, 이들 모델이 복잡한 함수를 근사하는 데 충분한 표현력을 갖추고 있음을 밝힙니다.

- **Performance Highlights**: VAR transformer는 간단한 디자인만으로도 임의의 Lipschitz sequence-투-sequence 기능을 근사할 수 있으며, 이는 VAR 설정에서 고전적인 보편성 결과를 확장합니다. FlowAR 모델도 유사한 근사 능력을 보이고, 두 모델의 상호작용은 생성 모델 설계에 있어 효율성 및 표현력을 동시에 만족할 수 있는 길을 제시합니다. 이로써 효율성과 표현력이 반드시 상반되지 않음을 증명하며, 이러한 기초 연구 결과는 모델 심도, 헤드 수 및 근사 효율성 간의 트레이드오프를 이해하는 데 중요한 기초를 마련합니다.



### Low Tensor-Rank Adaptation of Kolmogorov--Arnold Networks (https://arxiv.org/abs/2502.06153)
- **What's New**: 본 논문에서는 Kolmogorov--Arnold 네트워크(KANs)의 전이 학습(transfer learning)을 위한 새로운 방법인 저 텐서 랭크 적응(low tensor-rank adaptation, LoTRA)을 제안합니다. LoTRA는 텐서의 Tucker 분해(tucker decomposition)에서 영감을 얻어 KAN의 파라미터 업데이트에서 발견된 낮은 텐서 랭크 구조를 활용하여 KAN의 미세 조정(fine-tuning)을 가능하게 합니다. 이 접근법은 과거 KAN의 한계를 극복하고 다양한 과학적 작업에서의 성능 향상 가능성을 보여줍니다.

- **Technical Details**: LoTRA의 기초는 KAN 모델을 특정 작업에 대해 미리 훈련한 후, 새로운 작업에서는 전체 텐서 파라미터를 업데이트하는 대신 Tucker 분해를 적용하여 적응하는 것입니다. 이 과정은 𝒜 + 𝒢 ×1 𝑼(1) ×2 𝑼(2) ×3 𝑼(3) 형태로 이루어집니다. 또한 LoTRA의 각 구성 요소에 대한 학습 속도(learning rate)를 선택하기 위한 이론적 분석을 제공하여 효율적인 훈련을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 제안된 학습 속도 선택 전략의 효능을 입증하며, LoTRA가 PDE(부분 미분 방정식) 해결에 있어 KAN의 전이 학습에 효과적임을 보여줍니다. Slim KANs도 제안되어 KAN 파라미터 텐서의 고유한 저 텐서 랭크 속성을 활용하여 모델 크기를 줄이면서 우수한 성능을 유지합니다. 추가적인 평가 결과는 함수 표현 및 이미지 분류 작업에서 LoTRA의 표현력이 뛰어나고 저 텐서 랭크 분해를 통한 파라미터 감소 가능성을 강조합니다.



### Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting (https://arxiv.org/abs/2502.06151)
- **What's New**: Powerformer는 기존 Transformer 모델이 시간 시계열 데이터의 인과적 관계와 지역적 특성을 고려하지 못하는 한계를 극복하기 위해 개발된 새로운 변형입니다. 특히, 비인과적 주의(attention) 가중치를 인과적 가중치로 대체하고, 이러한 가중치를 부드러운 강하 형태로 조정하여 인과적 패턴을 학습할 수 있도록 합니다. 이 방법은 시간적으로 국소적인 의존성을 선호하는 유도 편향을 부여하며, 각 데이터셋의 독특한 상관 구조를 학습할 수 있는 유연성을 유지합니다.

- **Technical Details**: Powerformer는 가중치가 부여된 인과적 멀티헤드 주의(WCMHA) 메커니즘을 도입하여 시간 의존성을 학습합니다. 이는 데이터의 시간적 쌍관계를 포착하기 위해 강하 법칙(power-law) 기반의 추가적인 가중치 조정을 포함합니다. 본 모델은 전통적인 Transformer 아키텍처에 기반하되, 주의(weighted attention) 메커니즘에 시간 감쇠 패턴을 통합하여 시간적 의존성을 더 잘 반영합니다.

- **Performance Highlights**: Powerformer는 공공 시간 시계열 벤치마크에서 최고의 정확도를 달성하며, 기존 모델들보다도 뛰어난 성능을 보입니다. 이 모델은 그 단순함에도 불구하고, 시간 시계열 데이터의 자연적 구조와 더 잘 정렬되어 있습니다. 또한, 주의 패턴의 해석 가능성을 향상시켜, 학습된 의존성과 주의 가중치 간의 관계를 명확하게 보여줍니다.



### Guided Exploration for Efficient Relational Model Learning (https://arxiv.org/abs/2502.06146)
- **What's New**: 이번 연구는 대규모 환경에서 관계 모델을 학습하기 위한 효율적인 탐색 전략을 개발하는 데 중점을 둡니다. 랜덤 탐색 방식의 한계를 극복하기 위해, 제안된 방법은 오라클(demonstration) 기반 초기화 및 목표 선택 가이드를 사용하여 정보를 최대한 효율적으로 수집합니다. 이는 복잡한 장기 과제를 해결하는 데 있어 기존의 방식보다 개선된 성과를 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 관계 분야에서의 효율적인 탐색 원칙을 바탕으로 합니다. 이 원칙은 두 가지로, (1) 여러 작업의 계획을 보장하기 위한 객체의 표본 대조와 (2) 유의미한 목표-행동 쌍을 선택하고 이를 실행하는 계획을 통해 정보를 최대한으로 수집하는 것을 포함합니다. 주 대상 환경인 'Baking-Large'에서는 이 방법이 매개변수와 상태 공간이 방대해 탐색이 용이하지 않은 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 오라클 기반의 초기화와 목표 선택 가이드를 조합함으로써 탐색 효율과 일반화 능력이 크게 향상되었습니다. 이로 인해 GLIB보다도 우수한 성능을 나타내며, 복잡한 관계 모델 학습 문제에서 향후 연구 방향 제시가 가능함을 보여줍니다. 이러한 발견은 대규모 및 복잡한 도메인에서의 효율적 학습을 위한 중요한 이정표가 될 것입니다.



### Graph Neural Networks at a Fraction (https://arxiv.org/abs/2502.06136)
Comments:
          12 pages, 2 figures, accepted at PAKKD 2025

- **What's New**: 이번 논문에서는 Quaternion Message Passing Neural Networks (QMPNNs)를 소개하며, 기존 GNN 모델의 75%에 해당하는 파라미터 수로 quaternion 공간에서 노드 표현을 계산하는 일반화된 프레임워크를 제공합니다. 이는 기존의 실수 기반 GNN보다 효율적으로 훈련 가능하며, 모델의 정확도를 유지하면서 에너지 소비를 줄입니다. 더불어, 그래프 로터리 티켓의 개념을 GNN 및 QMPNN에 맞게 재정의하여 새로운 관점을 제시합니다.

- **Technical Details**: Quaternions는 고차원 수 체계로, 기존의 실수나 복소수보다 다양한 이점을 제공하여 딥러닝에서의 활용 가능성이 큽니다. 본 연구에서는 GNN 모델의 초기화 로터리 티켓을 찾아 훈련 시 유사한 성능을 달성할 수 있는 서브 네트워크를 제공합니다. 이를 통해 파라미터 수를 더욱 줄일 수 있으며, quaternion 모델이 파라미터 수를 네 배로 증가시키지 않도록 제한하여 입력 기능의 작은 변화에 민감해지지 않도록 합니다.

- **Performance Highlights**: QMPNN은 실제 데이터셋을 활용하여 노드 분류, 링크 예측 및 그래프 분류와 같은 세 가지 기본 그래프 기반 작업에 대한 성능 평가를 진행했습니다. QMPNN 프레임워크와 로터리 티켓 가설을 통해 기존 GNN 모델 대비 성능 저하 없이 파라미터 수를 획기적으로 줄이는 결과를 보였습니다. 본 연구는 QMPNN 및 GNN의 롯데리 티켓의 실증적 존재를 확인하며, 이는 기존 기술을 넘어선 의미 있는 기여로 평가됩니다.



### Integrating Sequence and Image Modeling in Irregular Medical Time Series Through Self-Supervised Learning (https://arxiv.org/abs/2502.06134)
Comments:
          9 pages, 2 figures, AAAI2025

- **What's New**: 이 연구에서는 의료 분야에서 자주 발생하는 불규칙하고 결측 값이 많은 다변량 시간 시리즈 데이터를 처리하기 위해 시퀀스와 이미지 표현을 결합한 합동 학습 프레임워크를 제안합니다. 기존의 방법들이 주로 시퀀스나 이미지 중 하나의 모델링 관점만을 채택한 반면, 본 연구는 두 가지 표현을 통합하여 보다 일반화 가능한 결합 표현을 확보하고자 합니다. 이를 통해 세 가지 자가 지도 학습 전략(self-supervised learning strategies)을 설계하여 시퀀스와 이미지 표현의 융합을 촉진합니다.

- **Technical Details**: 제안된 접근법은 생성기-판별기 구조(generator-discriminator structure)와 적대적 전략(adversarial strategy)을 사용하는 시퀀스 모델링 브랜치와 다양한 이미지 변환 전략(image transformation strategies)을 활용하여 희소한 시리즈의 성능을 향상시키는 이미지 브랜치로 구성됩니다. 구체적으로 세 가지 자가 지도 학습 손실(loss)이 설계되었으며, 이는 각각 시퀀스 임퓨테이션 최적화, 일반화 가능한 결합 표현 학습, 그리고 유사 사례 간 클러스터링을 통해 결합 표현을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 PAM, P12, P19라는 세 가지 실제 임상 데이터셋에서 다른 최신 방법들과 비교해 우수한 분류 성능(classification performance)을 발휘했습니다. 특히 PAM 데이터셋에서 정확도(Accuracy)에서 3.1% 개선을 보였으며, P12와 P19에서는 각각 AUPRC와 AUROC에서 우수한 성과를 기록했습니다. 또한, 결측 값이 많이 발생하는 상황에서도 우리의 접근법이 다른 방법들보다 더욱 강건한 분류 성능을 보여줍니다.



### Improved YOLOv5s model for key components detection of power transmission lines (https://arxiv.org/abs/2502.06127)
Comments:
          23 pages, 14 figures

- **What's New**: 전력 전송선의 지능형 검사에 대한 연구가 진행되고 있습니다. 본 논문에서는 YOLOv5s 모델을 기반으로 하는 개선된 물체 탐지 모델을 제안하여 전송선의 주요 컴포넌트에 대한 탐지 정확도를 향상시켰습니다. 새로운 기법은 전력망 검사 이미지의 특성을 반영하여 검사 정확성을 높이고자 하였습니다.

- **Technical Details**: k-means 클러스터링의 거리 측정을 수정하여 YOLOv5s 모델의 앵커 매칭을 개선하였으며, CNAM(Convolutional Block Attention Module) 주의 메커니즘을 백본 네트워크에 추가하였습니다. 결과적으로, 클래스 간 불균형을 줄이기 위해 focal loss 함수를 적용하였습니다.

- **Performance Highlights**: 개선된 방법의 mAP(Mean Average Precision)는 98.1%, 정밀도(Precision)는 97.5%, 재현율(Recall)은 94.4%에 달하며, 탐지 속도는 84.8 FPS로 측정되었습니다. 실험 결과, 본 모델은 탐지 정확도를 향상시키고 다른 모델에 비해 뛰어난 성능을 보였습니다.



### Foundation Model of Electronic Medical Records for Adaptive Risk Estimation (https://arxiv.org/abs/2502.06124)
- **What's New**: 이번 연구에서는 향상된 변환기 모델인 ETHOS(Enhanced Transformer for Health Outcome Simulation)를 개발하여 환자의 건강 타임라인(PHTs)을 토큰화하여 미래의 건강 상태를 예측합니다. ETHOS는 변환기 기반의 구조를 사용하여 개인화된 위험 확률을 계산하는 Adaptive Risk Estimation System (ARES)와 결합되어 임상 의사결정을 지원합니다. ARES는 환자 개별적인 위험 요인을 설명하는 개인화된 설명 가능성 모듈을 포함하고 있으며, 환자 데이터에 따라 동적으로 위험을 평가할 수 있는 기능을 제공합니다.

- **Technical Details**: 이 연구에서는 MIMIC-IV 데이터베이스를 이용하여 거의 30만 명의 환자가 병원에 입원하거나 응급 치료를 받는 데이터를 분석했습니다. ETHOS는 환자의 의료 기록을 기반으로 건강 타임라인을 생성하며, 각 사건을 토큰으로 변환하여 구조화된 정보를 제공합니다. 이는 환자의 의료 여정을 효과적으로 모델링하고 임상 예측의 정확성을 높이는 데 필수적입니다.

- **Performance Highlights**: ETHOS는 병원 입원, 중환자실(ICU) 입원 및 장기 입원 예측에서 기존의 모델들과 비교하여 우수한 성능을 보였습니다. ARES는 첫 번째 단계에서 10%의 데이터를 테스트하여 모델의 신뢰성을 검증했으며, 변별력 지표인 AUC 스코어에서도 탁월한 성능을 기록했습니다. 개인화된 설명 가능성 모듈은 의료진이 특정 환자의 위험 예측에 영향을 미치는 주요 임상 요인을 이해하는 데 도움을 줍니다.



### Revisiting Dynamic Graph Clustering via Matrix Factorization (https://arxiv.org/abs/2502.06117)
Comments:
          Accepted by TheWebConf 2025 (Oral)

- **What's New**: 이 논문에서는 동적 그래프 클러스터링의 효율성과 견고성을 향상시키기 위해 동적 노드 임베딩과 클러스터링을 동시에 최적화하는 새로운 방법인 DyG-MF(Dynamic Graph Matrix Factorization)를 제안합니다. 특히, Temporal Separated Matrix Factorization을 통해 큰 행렬을 여러 개의 작은 행렬로 나누어 독립적으로 팩토리제이션하는 방법을 사용하여 계산 속도를 개선합니다. 또한, Bi-clustering Regularization을 도입하여 그래프 임베딩에서 노이즈를 필터링함으로써 견고성을 향상시킵니다.

- **Technical Details**: DyG-MF는 Temporal Separated Matrix Factorization, Bi-clustering Regularization, 그리고 Selective Embedding Updating의 세 가지 주요 기법으로 구성됩니다. Temporal Separated Matrix Factorization은 '분할 정복' 방식으로 노드를 서브셋으로 나누어 각각 독립적으로 행렬 팩토리제이션을 수행합니다. Bi-clustering Regularization은 행렬의 랭크를 최적화하여 노이즈 피처의 영향을 줄이며, Selective Embedding Updating은 동적 그룹의 노드 임베딩만 업데이트하여 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, DyG-MF는 11개의 벤치마크에서 뛰어난 확장성, 견고성, 효율성 및 효과성을 보였습니다. 특히, DyG-MF는 대규모 동적 그래프에 적용할 수 있는 가능성을 보여 주며, 기존의 행렬 팩토리제이션 기반 방법보다 우수한 성능을 발휘했습니다. 이러한 개선 사항은 실제 데이터와 합성 데이터 모두에서 검증되었으며, 소스 코드 또한 제공되고 있습니다.



### CSR-Bench: Benchmarking LLM Agents in Deployment of Computer Science Research Repositories (https://arxiv.org/abs/2502.06111)
- **What's New**: 본 논문에서는 컴퓨터 과학 연구 프로젝트를 위한 코드 배포를 자동화하기 위한 CSR-Bench라는 벤치마크를 소개합니다. 이는 LLM(대형 언어 모델)의 효율성을 평가하기 위해 다양한 측면에서 분석하며, GitHub 코드 저장소의 자동 배포를 위한 CSR-Agents라는 새로운 프레임워크 또한 제안합니다. 이 프레임워크는 다양한 LLM 에이전트를 활용하여 실험 환경을 설정하고 코드를 배포하는 작업을 반복적으로 개선합니다. 초기 결과는 LLM 에이전트가 개발자 생산성을 향상시키고 개발 워크플로우 관리에 기여할 수 있음을 보여줍니다.

- **Technical Details**: LLM을 활용한 CSR-Bench는 코드 배포 작업의 평가를 위한 기준으로, 실행 가능한 명령을 생성하고 배포 중 발생하는 오류를 해결하는 능력을 측정합니다. CSR-Agents 프레임워크는 다양한 전문 역량을 가진 LLM 에이전트 협력을 통해 명령 실행, 오류 로그 분석, 오류 수정 등을 수행합니다. 벤치마크는 100개의 고평가된 오픈소스 리포지토리에서 수집되며, 다양한 주제를 골고루 다루고 있습니다. 이 벤치마크는 컴퓨터 과학 연구 프로젝트의 코드 배포를 자동화하는 데 중요한 도구로서의 역할을 할 수 있습니다.

- **Performance Highlights**: CSR-Bench의 평가 결과에 따르면, LLM 에이전트는 코드 배포 프로세스를 가속화하여 연구자의 생산성을 크게 높일 수 있는 잠재력을 가진 것으로 나타났습니다. 코드 생성뿐만 아니라 실험 환경 설정, 데이터 모델 준비, 오류 수정 등의 비코딩 작업의 중요성을 강조하고 있습니다. 그러나 완전 자동화를 달성하기 위해 여전히 도전 과제가 존재하며, 이를 극복하기 위한 추가 연구가 필요합니다. 이 연구는 컴퓨터 과학 분야에서의 협업 및 재현성 향상에 기여할 것으로 기대됩니다.



### Circuit-tuning: A Mechanistic Approach for Identifying Parameter Redundancy and Fine-tuning Neural Networks (https://arxiv.org/abs/2502.06106)
- **What's New**: 이 연구에서는 기계적 해석성(mechanistic interpretability)을 통해 모델의 동작을 설명하는 방법에 대해 조사했습니다. 기존 연구들이 특정 행동의 정적 메커니즘에 초점을 맞춘 반면, 이 논문은 모델 내부의 학습 동역학(training dynamics)을 탐구합니다. 새로운 방법으로 제안된는 circuit-tuning 알고리즘을 통해 노드 중복(node redundancy)의 개념이 도입되었으며, 이는 학습 메커니즘에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서 제안된 circuit-tuning 알고리즘은 두 단계로 이루어져 있으며, 각 단계에서 모델의 비관련 엣지(edges)를 마스킹(mask)하고 특정 작업을 담당하는 나머지 파라미터를 업데이트합니다. 본 알고리즘은 기존의 테크닉보다 우수하며 다양한 모델과 작업에 대해 확장 가능합니다. 또한, 비선형 네트워크의 자가 조직화(self-organization) 메커니즘을 분석하고 시각화하여 모델 학습 중의 변화 과정을 명확히 보여줍니다.

- **Performance Highlights**: 실험 결과, 논문에서 제안한 방법은 다양한 작업에서 성능을 개선하였을 뿐만 아니라, 일반적인 능력을 보존하면서도 확장성(scaleability)을 가지고 있음을 입증하였습니다. 이는 AI의 해석성 뿐만 아니라 실제 응용 가능성에 긍정적인 영향을 미칠 것으로 기대됩니다. 연구진은 학습 과정의 직관을 더욱 깊게 제공하며, Fine-tuning에서의 새로운 접근법을 제시합니다.



### Comprehensive Framework for Evaluating Conversational AI Chatbots (https://arxiv.org/abs/2502.06105)
Comments:
          2 Figures

- **What's New**: 이번 논문은 고객 서비스의 효율화를 위한 대화형 AI 챗봇의 평가를 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 금융 서비스 산업에서 중요한 규제 준수와 사용자 신뢰, 운영 효율성을 고려하여 설계되었습니다. 제안된 시스템은 챗봇의 인지 및 대화 지능, 사용자 경험, 운영 효율성, 윤리적 및 규제 준수의 네 가지 차원에서 체계적으로 평가합니다.

- **Technical Details**: 연구는 최신 AI 방법론과 금융 규제를 통합하여 이론적 기반과 실제 배치의 도전 과제를 연결합니다. 프레임워크는 챗봇의 성능을 정량적으로 평가하는 데 중점을 두며, 대화의 일관성, 실시간 적응성, 공정성 등의 개선 방향을 제시합니다. 이러한 내용을 통해 챗봇이 사용자 경험을 어떻게 향상시킬 수 있는지에 대한 깊은 통찰을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 대화형 AI 챗봇이 금융 서비스에서 기대하는 성능 기준을 충족할 수 있도록 돕습니다. 특히, 챗봇이 사용자와의 대화에서 더욱 자연스럽고 일관된 상호작용을 이끌어낼 수 있는 가능성을 보여줍니다. 이 연구는 향후 연구 방향을 제시하며, 챗봇의 지속적인 발전을 위한 기초를 마련합니다.



### NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems (https://arxiv.org/abs/2502.06097)
Comments:
          Accepted by WWW 2025 Industry Track

- **What's New**: 이번 논문에서는 Neighbor Lists 모델을 활용한 Generative Reranking(NLGR)을 제안합니다. NLGR은 기존의 evaluator-generator 패러다임을 따르면서, generator의 성능을 개선하여 조합 공간에서의 최적화를 목표로 합니다. 이 모델은 neighbor lists를 사용하여 generator의 훈련 과정을 향상시키고, 새로운 샘플링 기반 비자기회 생성 방법을 도입하여 최적 리스트 찾기를 쉽게 합니다.

- **Technical Details**: NLGR 모델은 generator와 evaluator 간의 목표 불일치 문제를 해결하기 위해 Neighbor Lists를 통합합니다. 훈련 과정에서 상대 점수를 인식할 수 있도록 neighbor lists를 활용함으로써 generator가 최적의 방향을 식별하도록 합니다. 또한 Position Decision Unit(PDU)과 Candidate Retrieval Unit(CRU)를 사용하여 조합 공간을 유연하게 탐색하고 최적 리스트로의 전환을 가능하게 합니다.

- **Performance Highlights**: NLGR의 효과성은 공공 및 산업 데이터셋에서의 광범위한 실험을 통해 검증되었습니다. Meituan 음식 배달 플랫폼에 성공적으로 배치되어 다양한 성과 지표에서 현저한 개선을 달성하였습니다. 논문에서 제안한 방법들이 기존 방법들에 비해 더 우수한 성능을 나타내는 것을 확인할 수 있었습니다.



### Post-detection inference for sequential changepoint localization (https://arxiv.org/abs/2502.06096)
- **What's New**: 이 논문은 탐지가 이루어진 이후의 연속적 변화점 분석에서 아직 많이 탐구되지 않은 근본적인 도전에 대한 연구입니다. 본 연구는 데이터 종속 정지 시점까지 관측된 데이터만을 사용하여 변화점을 로컬라이징하는 문제를 다룹니다. 기존의 방법론들이 주로 오프라인 환경에 한정된 반면, 이 연구는 순차적 프레임워크 내에서 변화점에 대한 신뢰 구간을 구축하는 방법론을 제시합니다.

- **Technical Details**: 변화 또는 변화가 발생한 후의 데이터에서 데이터 종속 정지 규칙을 사용하여 변화점을 추정하는 방법을 제안합니다. 본 연구에서는 전후 변화 분포가 알려져 있을 경우 신뢰 구간을 구성하는 요소를 포함하고, 이후 이러한 프레임워크를 복합적 분포 시나리오로 확장합니다. 변화점 탐지 알고리즘인 𝒜 (A)는 시뮬레이션된 데이터 시퀀스에서 실행이 가능해야 하며, 특별한 제약조건은 없습니다.

- **Performance Highlights**: 이 연구는 이론적으로 건전하면서도 실용적으로 효과적인 도구를 제공하여, 각 후보 변화점 t에 대해 t가 진짜 변화점인지 테스트하면서 신뢰 구간을 형성하는 접근 방식을 사용합니다. 실험 및 수학적 증명을 통해, 변화 탐지 알고리즘의 평균 실행 길이(ARL)나 허위 경고 확률(PFA)에 따라 결정적인 보장을 달성할 수 있음을 보여줍니다. 이로써 순차적 환경에서 변화점 분석의 중요한 이정표를 마련하게 되었습니다.



### Rateless Joint Source-Channel Coding, and a Blueprint for 6G Semantic Communications System Design (https://arxiv.org/abs/2502.06095)
Comments:
          39 pages, 9 figures, 2 tables

- **What's New**: 이번 논문은 비율이 없는 공동 소스-채널 코딩(rateless joint source-channel coding)인 rateless JSCC를 소개합니다. 이 코딩 방식은 연속적으로 다양한 비율(coding rates)에 대해 최적화되어 있어, 어떤 비율에서도 원하는 왜곡(distortion)을 달성하도록 설계되었습니다. 또한, rateless JSCC에 적합한 비율 적응형(rate-adaptive) 및 안정적인 통신 링크(link operation)가 도입됩니다.

- **Technical Details**: 논문에서 소개된 'RLACS 코드(Rateless and Lossy Autoencoder Channel and Source code)'는 이미지 신호의 재구성 손실(reconstruction loss)을 테스트하여 변동하는 채널 품질(channel quality)에 강인한 성능을 보여줍니다. 이 코드는 의미적 왜곡(semantic distortion)을 처리할 수 있는 경우에 쉽게 적용될 수 있습니다. 또한, 의미적 통신(semantic communication)에 대한 실용적인 우려를 다루고, 기존 네트워크 시스템의 일부 수정으로 구축할 수 있는 의미적 네트워킹 시스템 설계에 대한 청사진을 제공합니다.

- **Performance Highlights**: RLACS 코드의 성능은 다양한 의미와 효과적 통신(use cases)이 필요한 상황에서 주목받고 있으며, 연구는 6G 통신 시스템 설계로 향하는 연구 문제와 개발 과제를 포괄적으로 제시합니다. 이 연구는 네트워크의 오류 없는 최적화에 기반한 현재의 통신 패러다임이 어떻게 의미적 추출(semantic extraction)과 재구성 효율을 개선할 수 있는지에 대한 통찰력을 제공합니다.



### Physics-Guided Foundation Model for Scientific Discovery: An Application to Aquatic Scienc (https://arxiv.org/abs/2502.06084)
- **What's New**: 이 논문에서 제안하는 물리 기반 가이드 기초 모델(Physics-Guided Foundation Model, PGFM)은 사전 훈련된 기계 학습(ML) 모델과 물리 모델을 결합하여 복합 시스템의 모델링을 개선합니다. PGFM은 다양한 영향을 미치는 변수를 포괄하는 시뮬레이션 환경 시스템에서 중요한 특성 상호작용을 선택하는 능력을 갖추고 있습니다. 이 방법론은 물로 이루어진 생태계 건강 및 수자원 관리에 핵심적인 물 온도 및 용존산소 동학의 모델링에서 효과성을 입증하였습니다.

- **Technical Details**: PGFM는 여러 개의 결합된 물리 프로세스를 모델링하기 위해 사전 훈련 및 미세 조정 단계를 포함하여 설계되었습니다. 사전 훈련 단계에서는 물리 기반 모델이 생성한 다양한 시뮬레이트 변수를 통해 하이브리드 데이터셋을 활용합니다. 이를 통해 모델은 데이터의 다각성과 다양성을 담아내며, 수치 소비와 시간 복잡성을 줄이면서도 예측 성능을 최적화합니다.

- **Performance Highlights**: PGFM은 미국 중서부의 다양한 호수에서 물 온도 및 용존산소 농도를 예측하는 데 성공적이었습니다. 이 모델은 제한된 관측 데이터를 사용하여도 안정적인 예측을 할 수 있는 능력을 보여주었으며, 이러한 결과는 모델이 실제 환경에 적합하게 일반화할 수 있음을 시사합니다. 제안된 모델은 다양한 과학 분야에서 물리 기반 모델을 활용할 수 있는 잠재력도 갖추고 있습니다.



### Benchmarking Prompt Sensitivity in Large Language Models (https://arxiv.org/abs/2502.06065)
- **What's New**: 이번 논문에서는 Prompt Sensitivity Prediction이라는 새로운 작업을 소개하고, LLM의 응답 정확성에 미치는 프롬프트 변형의 영향을 조사하기 위해 PromptSET이라는 데이터셋을 설계했습니다. 주요 목적은 LLM의 프롬프트 반응 능력에 대한 예측을 통해, 프롬프트의 미세한 변형이 LLM 성능에 미치는 영향을 분석하는 것입니다. 이를 통해 효과적인 프롬프트 설계의 필요성을 강조하고 있습니다.

- **Technical Details**: 제안된 Prompt Sensitivity Prediction 작업은 주어진 프롬프트가 LLM에 의해 효과적으로 수행될 수 있는지를 예측하는 것을 목표로 합니다. 각 프롬프트는 특정 정보 요구(Ip)에 따라 약간 수정된 버전으로 구성됩니다. 데이터셋은 TriviaQA와 HotpotQA에서 출발하여 생성된 다양한 프롬프트 변형으로 구성되며, 이 변형의 유사성 및 정보 요구의 일관성을 기준으로 평가됩니다.

- **Performance Highlights**: 기존의 텍스트 분류(TC) 및 질의 성능 예측(QPP)과의 유사성을 기반으로 프롬프트 민감도 예측 작업을 벤치마크하는 실험을 수행했습니다. 연구 결과, 기존 방법들이 이 새로운 작업을 효과적으로 해결하지 못함을 보여주었으며, 이는 프롬프트 민감도 예측을 위한 새로운 접근 방법의 필요성을 강조합니다.



### Multi-modal Data Fusion and Deep Ensemble Learning for Accurate Crop Yield Prediction (https://arxiv.org/abs/2502.06062)
Comments:
          28 pages, 7 figures and 5 tables

- **What's New**: 이 연구는 RicEns-Net이라는 새로운 딥 앙상블 모델을 소개하며, 다양한 데이터 소스를 통합하여 작물 수확량을 예측합니다. 특히, 합성 개구 레이더(Synthetic Aperture Radar, SAR) 및 센티넬(Sentinel) 위성에서의 광학 원격 감지 데이터를 활용하는 데 중점을 두고 있습니다. 이 연구는 EY의 오픈 과학 챌린지 2023에서 획득한 필드 데이터로 시작되었으며, 복잡한 환경 데이터 처리를 위한 머신러닝 프레임워크를 개발하는 것을 목표로 합니다.

- **Technical Details**: RicEns-Net 아키텍처는 100개 이상의 잠재적 예측 변수를 통해 15개의 정보를 선택하는 포괄적인 데이터 엔지니어링 프로세스를 사용하였습니다. 이는 '차원의 저주(curse of dimensionality)'를 완화하고 모델 성능을 개선합니다. 또한, 이 모델은 여러 머신러닝 알고리즘을 깊은 앙상블 프레임워크로 결합하여 각 기술의 장점을 활용하여 예측 정확도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, RicEns-Net은 341 kg/Ha의 평균 절대 오차(Mean Absolute Error, MAE)를 달성하였으며, 이는 해당 지역의 가장 낮은 평균 수확량의 5-6%에 해당합니다. 이러한 성능은 EY 챌린지 동안 개발된 기존 최첨단 모델들을 훨씬 초과합니다. 이로 인해 RicEns-Net은 작물 수확량 예측의 새로운 기준을 설정하였습니다.



### Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization (https://arxiv.org/abs/2502.06061)
Comments:
          61 pages

- **What's New**: 이 논문에서는 연속 흐름 기반 생성 모델의 사용자 정의 보상 함수에 따라 효과적으로 조정할 수 있는 새로운 강화 학습 (RL) 방법인 Online Reward-Weighted Conditional Flow Matching with Wasserstein-2 Regularization (ORW-CFM-W2)을 제안합니다. 이 방법은 보상 그래디언트나 필터링된 데이터 세트에 의존하지 않고, 흐름 매칭 프레임워크에 RL을 통합하여 사용자 정의 보상 함수로 생성 모델을 조정할 수 있도록 설계되었습니다. 또한, 정책 붕괴를 방지하기 위해 Wasserstein-2 거리 정규화를 적용하여 탐색(exploration)과 착취(exploitation) 사이의 균형을 효과적으로 유지할 수 있습니다.

- **Technical Details**: 제안된 방법은 온라인 보상 가중치 기법을 통해 모델이 데이터 매니폴드에서 높은 보상에 우선 순위를 두도록 합니다. 특히 계약 조건에 따라, Wasserstein-2 (W2) 거리 정규화를 통해 정책 업데이트가 안정적으로 유지되며, 이는 fine-tuned 모델과 사전 훈련된 참조 모델 간의 거리를 조절하는 역할을 합니다. 이 논문은 저자들이 제안한 방법의 수렴 특성과 유도된 데이터 분포에 대한 이론 분석을 제공하여 강화 학습 알고리즘 및 KL 정규화와의 관계를 정립합니다.

- **Performance Highlights**: 다양한 실험을 통해 목표 이미지 생성, 이미지 압축 및 텍스트-이미지 정렬과 같은 작업에서 제안된 방법의 유효성을 입증하였습니다. 본 연구의 방법은 최적의 정책 수렴을 달성하며, 보상 극대화와 생성 다양성 유지 간의 조절 가능한 균형을 허용합니다. 이는 강화 학습이 연속 흐름 기반 생성 모델에 효과적으로 적용될 수 있음을 보여주는 중요한 결과입니다.



### Nearly Optimal Sample Complexity of Offline KL-Regularized Contextual Bandits under Single-Policy Concentrability (https://arxiv.org/abs/2502.06051)
Comments:
          23 pages

- **What's New**: 본 논문에서는 KL-정규화된 정책 최적화 문제에 대해 새로운 알고리즘을 제시하였습니다. 이 알고리즘은 오프라인 컨텍스츄얼 밴딧(offline contextual bandits) 설정에서 단일 정책 concentrability 아래에서 O~(ϵ−1) 샘플 복잡도(샘플 복잡도)를 달성하는 첫 번째 방법입니다. 또한, 일반적인 함수 근사를 지원하며 불확실성에 대한 비관주의 원칙(pessimism in the face of uncertainty)을 기반으로 설계되었습니다.

- **Technical Details**: 알고리즘의 핵심은 KL 정규화의 강한 볼록성(strong convexity)과 진짜 보상(true reward)과 비관적 추정자(pessimistic estimator) 간의 차이가 조건부 비음수(conditional non-negativity)를 이용해 평균값 유형 위험 상한(mean-value-type risk upper bound)을 극단적으로 개선하는 것입니다. 이를 통해 알고리즘은 함수 클래스 내 어떤 두 함수 간의 불일치에 대한 균일한 제어 없이도 효과적으로 분석할 수 있는 새로운 공분산 기반 분석(covariance-based analysis)을 가능하게 합니다.

- **Performance Highlights**: 알고리즘의 근사 최적(optimality)은 Ω(ϵ−1) 하한(lower bound)를 통한 검증으로 입증되었습니다. 추가적으로, 알고리즘은 컨텍스츄얼 듀얼 밴딧(contextual dueling bandits)으로 확장되어 유사하게 거의 최적의 샘플 복잡도를 달성합니다. 이 연구는 특히 RL의 비정형(best-effort) 특성 때문에 기존의 이론적 이해가 부족한 분야에서 중요한 기여를 하고 있습니다.



### LM2: Large Memory Models (https://arxiv.org/abs/2502.06049)
- **What's New**: 이 논문에서는 Large Memory Model (LM2)을 소개하며, 이는 다중 단계 추론(multi-step reasoning) 및 관계적 주장(relational argumentation)을 처리하는 데 있어 전통적인 Transformer 모델의 한계를 극복하기 위해 보조 메모리 모듈을 갖춘 디코더 전용 Transformer 아키텍처입니다. LM2는 입력 토큰과 상호작용하고 게이팅 메커니즘을 통해 업데이트되는 컨텍스트 표현 저장소로 기능하는 메모리 모듈을 통합하고 있습니다.

- **Technical Details**: LM2는 여러 개의 Transformer 디코더 블록으로 구성되며, 메모리 모듈이 동적으로 중간 표현의 시퀀스를 저장하고 업데이트합니다. 이 모듈은 메모리 정보 흐름과 업데이트를 통해 작동하며, 입력 임베딩과 메모리 은행 간의 교차 주의(cross attention) 메커니즘을 사용하여 관련 정보를 찾습니다. 또한, 각 메모리 슬롯은 정체성 행렬로 초기화되어 있으며, 기억 제어 게이트(forget, input, and output)를 통해 메모리 업데이트가 조정됩니다.

- **Performance Highlights**: BABILong 벤치마크에서 LM2는 메모리 보강 Recurrent Memory Transformer (RMT) 모델보다 최대 80.4% 향상된 성능을 보였으며, MMLU 데이터셋에서도 기존 모델 대비 5.0% 향상된 결과를 기록했습니다. 특히 LM2는 다단계 추론(multi-hop inference), 수치 추론(numerical reasoning), 대규모 컨텍스트 질문-응답(question-answering)에서 뛰어난 성능을 입증하고 있습니다. 이러한 결과는 Transformer 아키텍처에 명시적 메모리를 통합하는 것의 중요성을 강조합니다.



### Benchmarking Prompt Engineering Techniques for Secure Code Generation with GPT Models (https://arxiv.org/abs/2502.06039)
Comments:
          Accepted at the 2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge 2025). 10 pages, 7 figures, 5 tables

- **What's New**: 본 논문은 Large Language Models (LLMs)에서의 취약점 감소와 관련하여 Prompt engineering의 유용성을 탐구합니다. 기존 연구에서 다루지 않았던 LLM이 생성한 코드의 보안 문제를 해결하기 위해, 다양한 prompt engineering 전략의 효과를 평가하는 벤치마크를 구현했습니다.

- **Technical Details**: 이 벤치마크는 두 개의 동료 검토된 prompt 데이터셋을 활용하고, 정적 스캐너(static scanners)를 사용하여 코드 보안을 대규모로 평가합니다. 연구팀은 GPT-3.5-turbo, GPT-4o, GPT-4o-mini 모델에 대해 여러 가지 prompt engineering 기법을 테스트했습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4o 및 GPT-4o-mini 모델에서는 보안 중심의 prompt prefix를 사용하여 보안 취약점 발생을 최대 56%까지 감소시킬 수 있었습니다. 또한, 모든 모델은 반복 주입 기법(iterative prompting techniques)을 사용할 때 이전에 생성된 코드에서 41.9%에서 68.7%의 취약점을 감지 및 수정할 수 있는 능력을 보여주었습니다.



### Provably Overwhelming Transformer Models with Designed Inputs (https://arxiv.org/abs/2502.06038)
- **What's New**: 본 논문에서는 학습된 transformer 모델이 주어진 입력 문자열에 의해 '압도된(overwhelmed)' 경우를 수학적으로 증명할 수 있는 알고리즘을 개발합니다. 이 알고리즘은 주어진 입력 문자열과 정해진 토큰의 길이에 대한 모델의 출력을 구속할 수 있는 기능을 가지고 있습니다. 다양한 변별성을 가진 문자열의 조합에서 모델의 예측 결과가 어떻게 변하는지를 분석함으로써, 더욱 철저한 이론적 근거를 제시합니다.

- **Technical Details**: 이 연구에서는 ‘과잉 압축(over-squashing)’이라는 개념을 다루며, 특정 입력 문자열이 모델의 출력에 미치는 영향을 평가합니다. 특히 단일 레이어 transformer 모델을 대상으로 하여, 주어진 입력 문자열이 모델의 응답에 어떤 식으로 ‘압도적(overwhelming)’인지를 확인하기 위한 수학적 기법을 개발합니다. 이를 통해, '최악의 경우 편차(worst-case deviation)' 및 '피크 간의 차이(peak-to-peak difference)'와 같은 두 가지 키 지표를 설정하여 이론을 강화합니다.

- **Performance Highlights**: 알고리즘의 성능은 실험을 통해 검증되었으며, 다양한 문자열에 대한 모델의 응답에서 '압도적'으로 증명된 결과를 보여줍니다. 특히, 압도적인 문자열이 모델의 민감한 기능에서 발생할 수 있는 오류를 발견하는 데 유용할 수 있습니다. 이러한 발견은 모델의 안전성 평가 및 프롬프트 엔지니어링 결과의 불가피성을 증명하는 데 중요한 역할을 할 수 있습니다.



### Kolmogorov-Arnold Fourier Networks (https://arxiv.org/abs/2502.06018)
- **What's New**: 이 논문에서는 Kolmogorov-Arnold 기반의 해석 가능 신경망(KAN)의 단점을 극복하기 위해 Kolmogorov-Arnold-Fourier 네트워크(KAF)를 제안합니다. KAF는 학습 가능한 Random Fourier Features(RFF)와 새로운 혼합형 GELU-Fourier 활성화 메커니즘을 통합하여 파라미터 효율성과 스펙트럼 표현 능력을 균형 있게 조절합니다. 논문은 기존 KAN의 두 개의 대형 행렬을 합쳐 파라미터 수를 크게 줄이고, 스펙트럼 왜곡 문제를 해결하기 위한 학습 가능한 RFF 초기화 전략을 도입했습니다.

- **Technical Details**: KAF는 KAN의 이중 행렬 구조를 매트릭스 결합 속성을 통해 병합하여, 파라미터 복잡성을 줄이는 방법을 사용합니다. 이를 통해 KAF는 O(d_in × d_out)으로 파라미터 수를 감소시키면서도 모델의 표현력을 유지합니다. 또한, 기존 B-spline 기저 함수를 학습 가능한 RFF로 대체하여 고차원 근사 작업에서 스펙트럼 왜곡 문제를 제거합니다. 마지막으로, hybrid GELU-Fourier 활성화 함수는 학습 과정 동안 동적으로 조정되어 모델의 효율성을 높입니다.

- **Performance Highlights**: KAF는 시각, 자연어 처리(NLP), 오디오 처리 및 미분 방정식 해결 작업을 포함한 다양한 도메인에서 뛰어난 성능을 보여줍니다. 실험 결과, KAF는 이론적 해석 가능성과 실용적 유용성을 효과적으로 결합하면서 높은 차원의 데이터에도 잘 확장됩니다. KAF는 높은 주파수 세부 사항을 캡처하는 데 우수한 성능을 발휘하며, 전통적인 딥러닝 모델의 한계를 넘어서고 있습니다.



### Analysis of LLM as a grammatical feature tagger for African American English (https://arxiv.org/abs/2502.06004)
Comments:
          13 pages, Accepted to "Findings of the Association for Computational Linguistics: NAACL 2025"

- **What's New**: 이번 연구에서는 African American English (AAE)의 고유한 문법적 특성을 인식하는 데 있어 다양한 자연어 처리(NLP) 모델을 체계적으로 비교합니다. AAE는 훈련 데이터가 부족한 저자원 언어로, Rule-based 모델과 Transformer 모델, 대형 언어 모델(LLM)이 평가됩니다. 연구 결과, LLM은 기존 모델보다 성능이 향상되었으나, 텍스트의 형식성과 같은 편향에 영향을 받는 것으로 나타났습니다.

- **Technical Details**: 연구는 Habitual Be와 Multiple Negation을 AAE의 주요 문법적 특성으로 선택하여 이들을 인식하는 NLP 시스템의 능력을 평가합니다. 각 모델은 주어진 문장의 긍정 또는 부정을 분류하는 이진 분류 작업을 수행하며, 데이터와 모델 설정은 공정한 평가를 위해 일관된 하이퍼파라미터와 프롬프트 구조로 구성됩니다. OpenAI의 gpt-4o-mini와 Meta의 LLaMA 3-8B-Instruct 모델을 사용하여 AAE 특성에 대한 인식 성능을 비교합니다.

- **Performance Highlights**: 모델의 평가에서 LLM이 기존의 Rule-based 및 Transformer 모델보다 더 높은 성능을 보여주었습니다. 그러나 LLM은 예시의 순서나 최근성 같은 요소에 영향을 받으며, 이는 성능에 부정적인 영향을 미칠 수 있습니다. 연구 결과는 AAE의 고유한 언어적 특성을 더 잘 수용하기 위한 모델의 개선이 필요하다는 것을 강조하며, 데이터와 코드는 공개되어 추가 연구를 지원합니다.



### Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI (https://arxiv.org/abs/2502.05999)
Comments:
          8 pages, 5 figures, 2 tables

- **What's New**: 이 논문은 어린이, 성인, AI의 그림을 비교하여 시각적 창의성을 정량화하기 위한 새로운 데이터셋과 계산적 프레임워크를 제안합니다. 연구팀은 1338개의 그림을 바탕으로 스타일(style)과 콘텐츠(content)의 두 가지 요소를 분석합니다. 어린이의 그림은 더 많은 요소를 포함하고, AI 그림은 더 높은 잉크 밀도를 가지며, 성인의 그림은 최대 개념 다양성을 드러내는 등의 특징을 밝혔습니다. 이 작업은 인간과 기계의 창의성을 비교하기 위한 최초의 체계적인 기반을 제공합니다.

- **Technical Details**: 발표된 연구에서는 다양한 그리기 기술과 스타일을 평가하기 위해 두 가지 주요 요소인 스타일과 콘텐츠를 구분합니다. 스타일에 대한 측정에서는 잉크 밀도(ink density), 잉크 분포(ink distribution) 및 요소의 수(number of elements)를 정의합니다. 콘텐츠 분석에는 전문가 주석 전문가의 카테고리와 이미지, 텍스트 임베딩(image and text embeddings)을 사용하여 개념적 다양성을 연구하고, 거리 측정(distance measures)을 계산합니다. 데이터 수집은 공립 몬테소리 학교에서 소그룹으로 진행되었으며, AI 모델의 경우 DALL-E를 활용한 작업이 포함되어 있습니다.

- **Performance Highlights**: 결과적으로 연구팀은 전문가와 자동화된 창의성 점수 간의 불일치를 강조하면서 AI와 인간 그림의 창의성 평가에서 중요한 차별점을 발견했습니다. 이 연구는 전문가의 평가와 AI의 평가 방식을 결합함으로써 창의성 평가의 균형 잡힌 관점을 제공합니다. 최종적으로, 이 작업은 다양한 지능형 시스템 간의 창의적 표현 차이를 체계적으로 규명하는 데 기여하여, 시각적 창의성의 기본 계산 원리를 이해하는 데 중요한 발판을 쌓았습니다.



### Motion Control in Multi-Rotor Aerial Robots Using Deep Reinforcement Learning (https://arxiv.org/abs/2502.05996)
- **What's New**: 이번 연구는 드론 기반의 적층 제조(additive manufacturing, AM)에서의 동작 제어 문제를 해결하기 위해 심층 강화 학습(deep reinforcement learning, DRL)을 적용한 내용을 다룹니다. 기존의 PID 제어기와 같은 전통적인 제어 방법들은 동적 환경에서 자주 파라미터 재조정이 필요하고, 이는 실제 적용에 한계를 가져옵니다. 연구팀은 DDPG(Deep Deterministic Policy Gradient)와 TD3(Twin Delayed Deep Deterministic Policy Gradient)의 성능을 비교하면서, 복잡도가 증가하는 교육 과정(curriculum learning) 내에서 드론의 자율적인 제어 방안을 제시합니다.

- **Technical Details**: 제안된 접근 방식은 3D 공간에서 드론의 정밀하고 안정적인 제어를 목표로 하며, 고차원 상태 및 행동 공간에서 효과적인 학습을 위해 심층 신경망을 활용합니다. 드론의 움직임은 관성 및 중력 효과와 외부 방해요소를 포함하는 비선형 동역학으로 모델링됩니다. 정책 안정성을 향상시키기 위해, 흐름의 질량 변화에 따라 체중을 조절하고, 경로 추적의 외부 방해 요소에 대응하기 위한 훈련 체계로 교육 과정을 도입하여 점진적으로 복잡한 작업을 처리합니다.

- **Performance Highlights**: 실험 결과에 따르면, TD3는 다양한 질량 변화 상황에서도 훈련의 안정성을 유지하며 정확성과 성공률을 높이는 데 큰 장점을 보였습니다. DDPG와 TD3의 비교를 통해 TD3가 보다 신뢰할 수 있고 일관된 제어를 제공하는 것으로 나타났습니다. 이 연구는 드론 기반의 AM에서 자율적이고 복잡한 작업을 수행할 수 있는 향상된 제어 시스템 개발에 기여할 수 있는 기초를 마련했습니다.



### Speech to Speech Translation with Translatotron: A State of the Art Review (https://arxiv.org/abs/2502.05980)
Comments:
          12 pages and 3 figures

- **What's New**: 이 논문은 음성-음성 번역(S2ST) 모델에 대한 포괄적인 리뷰를 제공하며, Google의 Translatotron 모델에 특히 주목합니다. Translatotron은 전통적인 캐스케이드 모델의 복합 오류 문제를 해결하기 위해 설계된 첫 번째 직접 음성-음성 번역 모델입니다. 지금까지 Translatotron 1, 2, 3의 세 가지 버전이 있으며, 각 버전이 이전 모델보다 더 나은 성능을 보이고 있습니다.

- **Technical Details**: Translatotron 모델은 음성 인식(speech recognition), 자동 번역(machine translation), 텍스트-음성 변환(text-to-speech synthesis) 등 여러 가지 기술을 사용하여 음성을 직접 번역하는 방식입니다. 이 연구에서는 음성 스펙트로그램(speech spectrogram)을 매핑하여 각 언어의 음성 스펙트로그램을 연결하는 과정을 소개하고 있습니다. 특히 Translatotron 2는 음성 인코더(speech encoder), 언어 디코더(linguistic decoder), 음향 합성기(acoustic synthesizer)로 구성되어 있습니다.

- **Performance Highlights**: Translatotron 모델은 기존의 캐스케이드 모델보다 일부 상황에서 더 나은 성능을 보이며, 특히 아프리카 언어와 잘 형성된 언어 간의 언어 갭을 줄이는데 초점을 맞추고 있습니다. Translatotron 2는 직관적인 음성-음성 번역을 가능하게 하며, BLEU 점수에서 눈에 띄는 개선을 보여줍니다. 이 연구는 Translatotron 모델의 차별성과 각 모델의 적합성을 비교 분석하는 데 중점을 두고 있습니다.



### Redefining Robot Generalization Through Interactive Intelligenc (https://arxiv.org/abs/2502.05963)
- **What's New**: 최근 대규모 기계 학습의 발전으로 고용량의 재단 모델들이 개발되었습니다. 이 모델들은 여러 다운스트림 작업에 적응할 수 있는 가능성을 지니고 있습니다. 특히, 이 논문에서는 로봇 재단 모델이 인간과의 상호작용을 고려한 다중 에이전트 관점으로 발전해야 한다고 주장합니다.

- **Technical Details**: 제안된 아키텍처는 네 가지 모듈로 구성되어 있습니다: (1) 센서 모듈, (2) 팀워크 모델, (3) 예측 세계 신념 모델, (4) 메모리 및 피드백 메커니즘입니다. 이 모듈들은 신경과학과 인지 과학의 원리에서 영감을 받아, 로봇과 인간이 상호작용하며 공동 적응을 할 수 있도록 설계되어 있습니다. 이를 통해 사용자의 생리학적 신호와 장치의 제어가 원활하게 이루어질 수 있습니다.

- **Performance Highlights**: 기존의 단일 에이전트 모델은 실시간 협업 환경에서 제한된 성능을 보입니다. 그러나 다중 에이전트 접근법을 통해, 로봇은 사용자의 변동하는 지시를 실시간으로 반영하며 보다 손쉬운 협업이 가능합니다. 궁극적으로, 이 새로운 프레임워크는 로봇이 개인화되고, 안전하며, 예측 가능한 성능을 발휘하도록 도울 것입니다.



### Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks (https://arxiv.org/abs/2502.05951)
- **What's New**: 이 연구는 Cyri라는 AI 기반 대화형 어시스턴트를 소개합니다. Cyri는 대규모 언어 모델(Large Language Models, LLM)을 활용하여 사용자가 피싱 이메일을 탐지하고 분석하는 데 도움을 주기 위해 설계되었습니다. 이 시스템은 사용자의 이메일 업무 흐름과 원활하게 통합되며, 데이터 프라이버시를 유지하기 위해 로컬 프로세싱을 사용합니다.

- **Technical Details**: Cyri는 피싱 공격에 사용되는 의미적 특성을 분석하기 위해 문서의 의미와 맥락에 집중합니다. 사용자는 Cyri와의 대화를 통해 이메일이 왜 피싱으로 분류되었는지 혹은 안전하다고 평가되었는지를 명확히 할 수 있습니다. 이 시스템은 사용자 친화적인 인터페이스를 제공하여 기술적 기반 탐지기에서 발생할 수 있는 사용자 둔감화를 방지합니다.

- **Performance Highlights**: 평가 과정에서 총 420개의 피싱 이메일과 420개의 합법적인 이메일로 이루어진 데이터셋을 활용하였으며, Cyri는 피싱 탐지에 필수적인 주요 의미적 특성을 높은 정확도로 식별합니다. 사용자 연구에서는 10명의 전문가와 비전문가가 Cyri의 효과성과 사용 용이성을 평가하였고, 결과는 Cyri가 사용자가 피싱 이메일을 식별하는 데 큰 도움을 주었다는 것을 보여주었습니다.



### Survival Concept-Based Learning Models (https://arxiv.org/abs/2502.05950)
- **What's New**: 이번 연구는 Survival Analysis(생존 분석) 및 Concept-based Learning(CBL) 을 통합하여 SurvCBM(생존 개념 기반 병목 모델) 및 SurvRCM(생존 정규화 개념 기반 모델)이라는 두 가지 새로운 모델을 제안합니다. 이들 모델은 기존 CBL 프레임워크가 다루지 못했던 censored data(검열된 데이터)를 포함하는 생존 분석 문제를 해결합니다. 이 연구는 생존 함수와 확률적 예측을 제공하는 COX 모델과 Beran 추정기를 사용하여 해석 가능한 예측을 가능하게 합니다.

- **Technical Details**: SurvCBM은 개념 병목 모델(CBM)의 아키텍처를 기반으로 하여 개념 예측기로부터 생성된 개념을 사용해 클래스 예측을 수행합니다. SurvRCM은 개념을 정규화로 활용하여 정확성을 높이는 구조로 설계되었습니다. 두 모델은 end-to-end 학습 방식을 통해 적합도가 높은 생존 일정을 예측할 수 있습니다.

- **Performance Highlights**: 두 모델의 수치 실험 결과 SurvCBM이 다른 생존 모델들과 비교할 때 성능이 우수한 것으로 나타났습니다. 특히, 모델이 개념 정보를 활용하는 것의 중요성과 장점을 강조하고 있습니다. 연구에서 제안한 알고리즘의 코드는 GitHub에 공개되어 있어, 연구자들이 쉽게 접근하여 활용할 수 있습니다.



### Verifying Proportionality in Temporal Voting (https://arxiv.org/abs/2502.05949)
Comments:
          Appears in the 39th AAAI Conference on Artificial Intelligence (AAAI), 2025

- **What's New**: 본 연구에서는 정해진 시간 수평선에 따라 진행되는 임시 투표 모델을 다룹니다. 특히, 각 라운드에서 유권자들이 후보자에 대한 선호를 보고하며, 특정 후보자가 선정됩니다. 기존의 다수후보 선거 모델을 기반으로 한 공정성과 비례 대표 개념들이 시간적 설정에 적합하도록 적응해야 하는 필요성을 강조하고 있습니다. 이를 통해 정당한 대표성(Justified Representation) 개념을 시간적 상황에 적용하기 위한 새로운 한계를 설정하고 있습니다.

- **Technical Details**: 연구는 다수후보 투표 환경에서의 검증 문제(complexity of verification)를 심도 있게 탐구합니다. 전통적으로 다수후보 투표에서는 모든 후보자가 동시에 선택되는 반면, 본 모델에서는 시간이 지남에 따라 유권자의 선호가 변화하고, 특정 후보자가 여러 번 선택될 수 있습니다. 이러한 상황에서 JR, PJR, EJR의 세 가지 속성을 검증하는 것이 coNP-hard임을 보였으며, 특정 조건에서 다항식 시간 내에 해결할 수 있는 알고리즘을 개발하였습니다.

- **Performance Highlights**: 본 연구는 유권자의 수(n)를 기준으로 확정 매개변수의 무한성(fixed parameter tractability)을 추가로 제시하며, EJR을 만족하는 결과를 선택하기 위한 정수 선형 프로그래밍의 공식을 개발하였습니다. 또한, 기존 연구에서 제기된 개방 질문 중 일부에 대한 답변을 제시하며, Greedy Cohesive Rule이 시간적 환경에 적합하게 조정될 수 있음을 입증하였습니다. 연구의 주요 기여는 시간적 투표 모델의 복잡성을 명확히 하고, 이러한 환경에서의 해결 방안을 제시한 점입니다.



### "Let the AI conspiracy begin..." Language Model coordination is just one inference-intervention away (https://arxiv.org/abs/2502.05945)
Comments:
          Large Language Models (LLMs), Interference-time activation shifting, Steerability, Explainability, AI alignment, Interpretability

- **What's New**: 이번 연구에서는 대형 언어 모델의 행동을 조정할 수 있는 간단하면서도 효과적인 방법론을 도입합니다. 이 방법은 추가 교육 없이도 효과적인 interference-time activation shifting을 활용하며, 모델 출력의 원하는 행동과 원하지 않는 행동 간의 activation 차이에서 intervention 방향을 파생합니다. 또한, 다중 선택 답변을 포함하도록 모델을 유도함으로써 각 attention head의 조작에 대한 출력의 민감도를 자동으로 평가할 수 있습니다.

- **Technical Details**: 대형 언어 모델은 다양한 애플리케이션에서 널리 사용되고 있지만, 악의적인 사용자들에 의해 남용될 수 있는 위험이 존재합니다. 본 논문에서는 inference-time intervention을 통해 모델의 출력을 특정 방향으로 조정할 수 있는 가능성을 보여줍니다. 이를 위해 attention head 수준에서의 조작 기술을 사용하며, 이는 샘플 효율이 좋고 모델 weight의 업데이트가 필요하지 않은 장점이 있습니다.

- **Performance Highlights**: 우리는 Llama 2 모델이 AI 간의 협조를 선호하도록 유도하는 결과를 보여주었으며, 이는 기존의 alignment 목표보다 더 강력한 조작을 가능하게 합니다. 특히, 우리의 방법론은 'AI coordination' 데이터셋에서의 개방형 질문 생성에서도 잘 일반화되며, 이로 인해 현재의 alignment 전략의 한계를 강조하고 향후 연구 방향을 제시합니다.



### A Semi-Supervised Text Generation Framework Combining a Deep Transformer and a GAN (https://arxiv.org/abs/2502.05937)
Comments:
          7 pages

- **What's New**: 이 논문은 Semi-supervised 텍스트 생성을 위해 Deep Generative Pre-trained Transformer 언어 모델과 Generative Adversarial Network(GAN)를 연결하는 새로운 프레임워크를 제안합니다. 제안된 모델은 24층의 대규모 텍스트 코퍼스를 비지도 학습으로 사전 학습한 후, 합성 텍스트 생성을 위한 간단한 GAN 아키텍처를 소개합니다. 실제 데이터를 GAN 샘플로 증강해 Transformer 모델의 파인튜닝에 활용하는 반지도 학습 접근법을 보입니다.

- **Technical Details**: 이 모델은 Gumbel-Softmax를 적용하여 토큰의 이산성을 처리하고, GAN 아키텍처를 통해 생성된 샘플로 언어 모델을 세밀하게 조정하는 방법을 설명합니다. Transformer 모델은 자기 주의 메커니즘을 기반으로 하여, 긴 의존성을 포착하는 데 효과적이며, GPT-2와 같은 모델의 이점도 충분히 활용합니다. Gumbel-Softmax 기법은 이산 샘플링의 비미분가능성을 극복하는 기술로, 연속적이고 미분 가능한 참조를 제공합니다.

- **Performance Highlights**: 본 논문에서는 제안한 모델의 성능을 분석하기 위해 훈련 곡선 및 성능 비교를 제공합니다. 실험 결과, 실제 데이터의 일부와 GAN에서 생성된 샘플을 통합하여 텍스트 생성을 최적화하는 반지도 학습 방식이 효과적임을 보여줍니다. 이러한 접근은 자연어 처리(NLP) 분야에서의 데이터 증대와 모델의 일반화 능력을 향상시킬 수 있는 가능성을 제시합니다.



### Learning to Substitute Words with Model-based Score Ranking (https://arxiv.org/abs/2502.05933)
- **What's New**: 이번 연구에서는 기존의 인간 주석(data labeling)에 의존하지 않고 BARTScore라는 모델 기반의 평가 지표를 사용하여 문장 품질을 정량화하는 방법을 제안합니다. 이 접근법은 단순히 단어 대체의 질을 평가하는 것을 넘어서, 모델의 예측과 품질 점수 간의 정렬을 최적화하는 새로운 손실 함수(loss function)를 도입하였습니다. 결과적으로, 모델 학습에서 인간의 레이블이 필요 없어져 주석 비용을 절감하면서도 단어 대체의 품질을 유지할 수 있습니다.

- **Technical Details**: 저자들은 Smart Word Substitution(SWS) 작업을 통해 문서 내 단어 대체의 효율성을 높이고, 이를 위해 BARTScore를 활용하여 각 제안의 품질을 측정합니다. 기존의 masked language models(예: BERT, BART) 및 large language models에서 측정된 품질 점수와의 정렬 손실을 사용하여 단어 대체 제안을 최적화하는 새로운 방법론을 개발했습니다. 연구 결과는 제안된 방법이 기존 모델들보다 더 뛰어난 성능을 보임을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 BERT, BART와 같은 masked language models뿐만 아니라 GPT-4, LLaMA와 같은 large language models에 비해서도 우수한 성능을 기록하였습니다. 연구 결과는 문장 품질 향상과 관련된 SWS 작업에서 새로운 기준을 제시하며, 기존의 접근 방식에 비해 더 효율적이고 비용 절감이 가능한 방법임을 보여줍니다. 모든 코드와 자료는 별도로 제공되며, 큰 규모의 데이터 요구 없이도 뛰어난 성능을 발휘할 수 있음을 강조합니다.



### Skill Expansion and Composition in Parameter Spac (https://arxiv.org/abs/2502.05932)
Comments:
          ICLR 2025, 37 pages

- **What's New**: 본 논문에서는 인간의 지능을 모방하는 자율 에이전트를 위한 새로운 프레임워크인 Parametric Skill Expansion and Composition (PSEC)을 제안합니다. PSEC는 기존 기술과 지식을 효율적으로 재사용하고 새로운 과제에 적응할 수 있도록 설계되었습니다. 이 접근법은 기존 도메인에서 새로운 기술로의 안전한 전이를 지원하여, 자율 에이전트의 능력 진화를 촉진하는 것을 목표로 합니다.

- **Technical Details**: PSEC 프레임워크는 Low-Rank Adaptation (LoRA) 모듈을 활용하여 기술 라이브러리를 관리합니다. 이를 통해 기술들을 파라미터 공간에서 용이하게 통합하고, 새로운 기술을 지속적으로 발전시킬 수 있도록 합니다. 특히, 각 기술은 확산 모델(diffusion models)을 사용하여 표현력과 유연성을 극대화하며, 상호작용하는 기술 간의 정보를 활용하여 새로운 과제에 대한 대응 능력을 향상시킵니다.

- **Performance Highlights**: D4RL, DSRL 벤치마크 및 DeepMind Control Suite에서의 실험 결과, PSEC은 이전 지식을 효과적으로 활용하여 새로운 과제를 처리하는 능력이 뛰어난 것으로 평가됩니다. 다양한 설정, 즉 다목적 구성, 정책 지속적 변화, 역동적 변화에 적합하여, 실제 적용 가능성에 대한 높은 잠재력을 입증하였습니다.



### Protecting Intellectual Property of EEG-based Neural Networks with Watermarking (https://arxiv.org/abs/2502.05931)
Comments:
          21 pages, 13 figures, and 6 tables

- **What's New**: 본 논문은 EEG 기반 신경망에 적용할 수 있는 새로운 수용성 보호 메커니즘인 크립토그래픽 원더 필터 기반 워터마킹 프레임워크를 소개합니다. 기존의 추상적인 트리거 세트를 사용한 워터마킹 방식은 인증이 취약하여 EEG 모델의 독특한 도전 과제를 해결하지 못했습니다. 이 새로운 프레임워크는 충돌 저항 해시(collision-resistant hashing)와 공개 키 암호화(public-key encryption)를 활용하여 최소 왜곡으로 워터마킹을 제공하며, 전반적인 안정성을 보장합니다.

- **Technical Details**: 본 연구에서 제안하는 원더 필터는 소유자의 개인 키를 바탕으로 생성된 필터를 사용합니다. 소유자는 자신의 정보를 포함한 검증 문자열을 작성하고 개인 키로 서명한 후에 해시 함수를 적용하여 필터 F와 레이블 L을 도출합니다. 이 필터는 EEg 트레이닝 데이터와 결합되어 모델에서 효과적으로 워터마킹을 완료하며, 지속성(persistence), 해적 저항(piracy resistance), 인증(authentication)을 보장합니다.

- **Performance Highlights**: 제안된 방법론은 DEAP 데이터셋에서 여러 모델(EEGNet, CCNN 등)에 대해 평가되었으며, 99.4% 이상의 무작위 워터마킹 임베딩 정확도를 달성했습니다. 추가적으로, 공격에 대한 저항력이 강력하게 입증되어 사건 기반 수정 후에도 분류 정확도가 90%를 초과하는 성과를 보였습니다. 이 연구는 EEG 기반 모델의 지적 재산 보호에 중요한 기여를 하며, 의료 및 생체 인식 응용 분야에서 보안성을 높이는 데 기여할 것입니다.



### Sign-Symmetry Learning Rules are Robust Fine-Tuners (https://arxiv.org/abs/2502.05925)
- **What's New**: 이 논문은 생물학적으로 그럴듯한 학습 규칙을 통해 신경망 훈련의 새로운 접근법을 소개하고 있습니다. 구체적으로, 기존의 Backpropagation(BP)으로 사전 훈련된 모델을 Sign-Symmetry 학습 규칙으로 세부 조정하여 성능을 유지하면서도 강건성을 강화하는 방안을 제시하고 있습니다. 실험을 통해 이러한 방법이 다양한 작업에서 효과적이라는 것을 입증하였으며, 더 나아가 생물학적인 영감을 받은 학습 규칙의 적용 가능성을 논의하고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 Backpropagation으로 사전 훈련된 신경망을 Sign-Symmetry 학습 규칙으로 조정하는 방식입니다. 연구팀은 이 방법이 그래디언트 기반의 적대적 공격에 대한 강건성을 증대시킴과 동시에 BP와 유사한 성능을 유지한다고 보고하였습니다. 다양한 신경망 구조(AlexNet, VGG16, ResNet-18)와 학습 규칙(uSF, frSF, brSF)에 대해 실험을 수행하여 일관된 결과를 도출하였습니다.

- **Performance Highlights**: 연구 결과, Sign-Symmetry 방법은 모델의 정확도를 유지하면서도 강건성을 현저히 향상시키는 것으로 나타났습니다. 특히, 적대적 공격에 대한 저항력이 강화되었으며, 이는 Sign-Symmetry 학습 알고리즘이 정확한 그래디언트 계산에 의존하지 않기 때문이라고 설명합니다. 이러한 성과는 신경망의 훈련 및 조정 과정에서 Sign-Symmetry 학습 규칙의 새로운 가능성을 제시하며, 향후 연구에 대한 새로운 방향을 제공합니다.



### A Distributional Perspective on Word Learning in Neural Language Models (https://arxiv.org/abs/2502.05892)
- **What's New**: 이 연구는 언어 모델(LM)이 아동의 단어 학습 경로와 어떻게 상관관계가 있는지를 조사하는 데 중점을 두고 있습니다. 저자들은 기존의 분포적 접근 방식이 단어 학습에서 중요한 정보를 포착하는 데 실패한다고 주장하며, 새로운 분포적 서명을 제안합니다. 이 서명은 단어의 적절성과 부적절성에 대한 지식을 나누어 평가합니다.

- **Technical Details**: 연구팀은 아동의 입력 데이터와 유사한 세 가지 데이터 세트를 사용하여 언어 모델을 처음부터 학습시켰습니다. 이 과정에서 단어의 학습 서명을 기록하고, 단어 습득 시점(AoA)을 평가하기 위한 기준을 설정합니다. 최종적으로, 이 연구는 다양한 분포적 서명의 상대적 유용성과 언어 모델과 아동의 단어 습득 순서의 비교를 다룹니다.

- **Performance Highlights**: 이 연구의 분석 결과, 여러 서명이 단어 학습의 여러 측면을 잘 나타내지 못하며, 아동의 단어 습득 패턴과는 상관관계가 낮음을 발견했습니다. 이는 현재의 방법들이 인간 언어 습득 모델로의 언어 모델의 한계를 강조합니다. 따라서 저자들은 향후 연구에서 제안된 새로운 서명을 사용하여 LM의 학습 경로를 개선할 것을 촉구하고 있습니다.



### MTPChat: A Multimodal Time-Aware Persona Dataset for Conversational Agents (https://arxiv.org/abs/2502.05887)
Comments:
          NAACL 2025 Findings

- **What's New**: MTPChat 데이터셋은 대화 응답과 기억이 시간에 따라 변하는 점을 처음으로 모델링한 다중모드(time-aware) 데이터셋입니다. 기존의 데이터셋들이 텍스트 기반 질문-응답(QA) 작업에 집중하고 있는 반면, MTPChat은 대화 및 기억의 자연스러운 흐름을 활용하여 인간의 인지에서의 시간적 변화를 시뮬레이션합니다. 이 데이터셋은 언어적, 시각적 요소 및 시간 정보를 통합하여 대화 모델의 복잡성과 현실성을 높이고 있습니다.

- **Technical Details**: MTPChat은 Temporal Next Response Prediction (TNRP)와 Temporal Grounding Memory Prediction (TGMP)이라는 두 가지 새로운 작업을 통해 모델의 시간 암시 큐(implicit temporal cues) 및 진화하는 응답 추적 능력을 평가합니다. 적응형 시간 모듈(adaptive temporal module)을 포함한 혁신적인 프레임워크를 제안하여 다중 모드 스트림을 효과적으로 통합하고 시간 의존성을 포착합니다. 이 모듈은 시간적 관련성을 바탕으로 특징들을 동적으로 병합하여 다중모드 통합의 일관성을 향상시킵니다.

- **Performance Highlights**: MTPChat을 평가하기 위해 SBERT와 CLIP 모델을 사용한 실험 결과, MTPChat은 다중모드 및 시간 민감한 시나리오에서 새로운 도전을 제시하고 있음을 보여주었습니다. 우리의 적응형 시간 모듈은 다른 특징 통합 방법에 비해 뛰어난 성능을 보이며, 모델의 다중모드 시간 인식 대화에 대한 추론 능력을 크게 향상시킵니다. 이 연구의 주요 기여는 MTPChat 데이터셋을 통해 시간 민감 대화 AI 연구의 발전을 도모함으로써, 모델이 인간 수준의 시간적 이해를 성취하도록 하는 것입니다.



### NeuralPrefix: A Zero-shot Sensory Data Imputation Plugin (https://arxiv.org/abs/2502.05883)
Comments:
          Accepted in PerCom 25

- **What's New**: 이 연구에서는 데이터 간헐성(data intermittency) 문제를 해결하기 위한 제로샷 임퓨테이션(zero-shot imputation) 개념을 공식화하고, 기존 훈련된 모델을 활용하여 간헐적으로 발생하는 데이터 손실을 처리할 수 있는 새로운 접근법인 NeuralPrefix를 제안합니다. NeuralPrefix는 데이터 간헐성으로 인한 격차를 메우기 위해 작업 모델(task model) 이전에 적용되는 생성 신경망 구성 요소로, 다양한 센서에 대한 적응성을 제공합니다.

- **Technical Details**: NeuralPrefix는 연속적인 동적 시스템(continuous dynamical system)으로 설계되어 있으며, 모든 시점에서 내부 상태를 미분 방정식(Ordinary Differential Equation, ODE)을 통해 추정할 수 있습니다. 이는 전통적인 신경망과는 달리 시간 지점에서 연속적인 상태 변화를 모델링할 수 있도록 하여, 간헐적인 관측치에 적합하게 데이터를 복원하는 데 도움을 줍니다. 또한, 중요한 부분에 집중하여 데이터 재구성을 돕는 축소 손실(shrinkage loss) 기법을 적용하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 다양한 센서 데이터셋에 대한 평가 결과, NeuralPrefix는 50%의 간헐적인 데이터에서 누락된 샘플을 정확하게 복원하여 SSIM(Structural Similarity Index Measure) 점수 0.93-0.96를 달성했습니다. 제로샷 평가에서도 NeuralPrefix는 재훈련 없이도 새로운 데이터셋에 잘 일반화할 수 있는 능력을 보여주었으며, 이는 이전의 task-specific 또는 modality-specific 솔루션의 한계를 극복하는 데 중요한 성과로 평가됩니다.



### Enhancing Depression Detection with Chain-of-Thought Prompting: From Emotion to Reasoning Using Large Language Models (https://arxiv.org/abs/2502.05879)
- **What's New**: 이번 연구에서는 우울증 탐지의 성능 및 해석 가능성을 향상시키기 위해 Chain-of-Thought Prompting 접근 방식을 제안합니다. 이 방법은 감정 분석, 이분법적 우울증 분류, 기저 원인 식별, 그리고 중증도 평가의 네 가지 단계로 탐지 과정을 세분화합니다. 이러한 구조화된 과정은 우울증 진단에 더 명확한 이유를 제공해줍니다.

- **Technical Details**: 우울증 탐지의 방법론은 네 개의 단계로 나뉘며, 각 단계는 정서 분석, 이분법적 분류, 기인 요인 분석, 그리고 중증도 평가로 구성됩니다. 이 구조는 PHQ-8 점수 체계를 기반으로 하여 우울증의 중증도를 평가하며, 다양한 사회적, 생물학적, 심리적 요인을 고려하여 개별적인 평가를 가능하게 합니다. 이를 통해 입력된 텍스트의 정서적 신호를 세밀하게 분석합니다.

- **Performance Highlights**: 제안된 방식은 이 기존 모델들보다 우울증 분류 정확성과 진단 통찰의 세부사항 측면에서 더 뛰어난 성과를 보여주었습니다. E-DAIC 데이터셋을 기반으로 한 실험 결과, 모델의 성능 개선이 입증되었으며, 이는 임상 적용 가능성을 높입니다. 연구는 뚜렷한 변화를 드러내며, 우울증 조기 발견 및 개입을 위한 새로운 가능성을 제시합니다.



### MMGDreamer: Mixed-Modality Graph for Geometry-Controllable 3D Indoor Scene Generation (https://arxiv.org/abs/2502.05874)
Comments:
          Accepted by AAAI 2025 Main Track

- **What's New**: 본 논문에서는 MMGDreamer라는 이중 분기(diffusion model)를 제안하여 기존의 텍스트 기반 입력의 한계를 극복하고, 보다 유연한 사용자 입력을 수용할 수 있는 혼합 모달리티 그래프(Mixed-Modality Graph)를 도입합니다. 이 모델은 객체 노드가 텍스트와 시각 모달리티를 통합할 수 있도록 구성되어 있으며, 이를 통해 생성된 장면의 객체 기하학적 요소에 대한 정밀한 제어가 가능합니다. 또한, 시각적 향상 모듈과 관계 예측기를 포함하여 더욱 정확하고 일관된 장면 레이아웃 생성을 지원합니다.

- **Technical Details**: MMGDreamer 모델은 세 가지 형태로 표현될 수 있는 노드로 구성된 혼합 모달리티 그래프(MMG)를 사용합니다: 텍스트, 이미지, 또는 두 가지의 조합. 이 그래프 구조는 사용자 입력에 기반하여 노드 간의 관계를 선택적으로 제공하거나 생략할 수 있습니다. 특정 객체의 기하학적 요소에 대한 세심한 제어를 가능하게 하며, 고유의 시각적 표현을 구축하기 위한 텍스트 임베딩을 활용하여 노드의 시각적 충실도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, MMGDreamer는 SG-FRONT 데이터셋에서 기존 기술보다 훨씬 높은 기하학적 제어력과 충실도를 발휘하였습니다. 또한, 최신 장면 합성(scene synthesis) 성능을 달성하여 현존하는 방법들보다 현저한 성능 향상을 보였습니다. 이러한 성능 향상은 복잡한 장면 생성을 위한 보다 효율적인 도구로서의 잠재력을 보여줍니다.



### Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education (https://arxiv.org/abs/2502.05863)
- **What's New**:  본 논문에서는 AI 중심 교육(AI-facilitated teaching)에서 다양한 query 스타일을 활용하여 추상 텍스트 설명을 해석하는 새로운 접근 방식을 제안합니다. 이를 위해 STEM 교육에 특화된 다양한 표현 검색 작업을 통해 여러 쿼리 스타일과 표현을 지원하는 새로운 STEM Education Retrieval Dataset (SER)을 소개합니다. 또한, prompt tuning 기반의 스타일 다양화를 위해 Uni-Retrieval 모델이 개발되어, 이는 쿼리 스타일 특징을 프로토타입으로 추출하고 효율적인 리트리벌 기능을 제공합니다.

- **Technical Details**:  Uni-Retrieval 모델은 쿼리 기반 검색을 위한 포괄적인 문제 정의를 기반으로 하며, 여러 스타일의 쿼리를 수용하기 위해 다양한 입력 유형(텍스트, 오디오, 이미지 등)을 고려합니다. SER 데이터셋은 24,000개의 텍스트 캡션, 오디오 클립 및 다양한 스타일의 쿼리로 구성되어 있으며, 20명의 대학원생에 의해 구성된 데이터셋입니다. 특히, Prompt Bank라는 혁신적인 데이터 표현 구조를 사용하여 교육 시나리오에 효율적으로 적합한 모델을 발전시킵니다.

- **Performance Highlights**:  실험 결과, Uni-Retrieval은 기존의 검색 모델에 비해 대부분의 검색 작업에서 월등한 성능을 보입니다. 이 모델은 다중 쿼리 스타일에 대해 동적으로 프롬프트 토큰을 검색하여 다양한 교육적 요구에 맞는 확장 가능하고 정밀한 솔루션을 제공합니다. 특히, Uni-Retrieval은 제한된 매개변수 증가로도 유의미한 성능 향상을 이끌어내며, STEM 교육 커뮤니티에 큰 잠재력을 제공합니다.



### Acquisition through My Eyes and Steps: A Joint Predictive Agent Model in Egocentric Worlds (https://arxiv.org/abs/2502.05857)
- **What's New**: 이번 논문에서는 인간의 인지 시스템에서 영감을 받은 에이전트 모델인 EgoAgent를 제안합니다. EgoAgent는 세계를 표현하고, 미래 상태를 예측하며, 합리적인 행동을 수행하기 위해 단일 transformer 모델에서 이 세 가지 능력을 동시에 학습합니다. 기존의 유사한 연구들이 독립적으로 세 가지 모델을 학습한 반면, EgoAgent는 이 모델들을 통합하여 서로의 학습을 돕도록 설계되었습니다.

- **Technical Details**: EgoAgent는 에고센트릭(Egocentric) 세계에서의 인식, 예측 및 행동을 기능적으로 통합합니다. 이 모델은 관찰과 인간 행동의 이력 데이터를 바탕으로 고차원 feature 벡터로 매핑하여 transformer에 입력하고, 이를 통해 현재 상태, 미래 상태 및 다음 행동을 예측합니다. 특히, 이 모델은 Attention 기법을 통해 세 가지 기능 사이의 내부 관계를 자연스럽게 설정하여 상호 보완적인 학습을 가능하게 합니다.

- **Performance Highlights**: EgoAgent는 이미지 분류, 에고센트릭 미래 상태 예측 및 3D 인간 모션 예측 과제에서 광범위한 실험을 통해 우수한 성능을 입증했습니다. 예를 들어, EgoAgent는 ImageNet-100과 ImageNet-1K에서 기존의 선도적인 방법들보다 1.40% 및 1.32%의 정확도를 향상시켰으며, 에고-엑소4D(Ego-Exo4D) 미래 상태 예측 과제에서는 각각 16.28% 및 16.95%의 mAP 개선을 달성했습니다.



### LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification (https://arxiv.org/abs/2502.05836)
Comments:
          Accepted on NAACL 2025

- **What's New**: 이 논문에서는 인도 법원의 판결을 중심으로 합법적인 문서의 의미적 세분화 및 수사학적 역할 분류 과제를 다루고 있습니다. LegalSeg라는 7,000개 이상의 문서와 140만 문장으로 구성된 대규모 주석 데이터셋을 새롭게 소개하며, 이는 법적 문서 처리를 위한 가장 큰 데이터셋입니다. 여러 최첨단 모델(예: Hierarchical BiLSTM-CRF, ToInLegalBERT, GNN 등)의 성능을 평가하고, 문맥과 구조적 관계를 통해 분류 정확도를 높였음을 보여줍니다.

- **Technical Details**: 논문에서는 법적 문서에서의 의미적 세분화를 위한 다양한 모델을 구현하여 평가하였습니다. Hierarchical BiLSTM-CRF 모델은 계층적 접근을 통해 문맥 정보를 캡처하고, MultiTask Learning은 역할 전환을 고려하여 수사학적 역할을 보다 정교하게 식별합니다. 이외에도 InLegalToBERT, Graph Neural Networks(GNNs), Role-Aware Transformers와 같은 새로운 접근 방식을 도입하여 모델의 표현력과 문맥 처리를 향상시켰습니다. 특히, GNN은 문장을 노드로 표현하여 정보 전파와 문맥 캡처를 효과적으로 수행합니다.

- **Performance Highlights**: 법적 문서의 세분화 및 이해도 향상에 대한 기여로 LegalSeg 데이터셋과 연구 결과는 법적 NLP 분야에서 중요한 기초를 제공합니다. 특히, RhetoricLLaMA를 통해 복잡한 법적 언어 처리를 위한 인스트럭션 튜닝된 대형 언어 모델의 가능성과 한계를 강조하였습니다. 모델 성능 평가 결과, 문맥이나 주변 문장의 레이블 활용이 분류 정확도에 긍정적인 영향을 미친 것으로 나타났습니다.



### Contrastive Representation Distillation via Multi-Scale Feature Decoupling (https://arxiv.org/abs/2502.05835)
- **What's New**: 본 연구에서는 knowledge distillation의 새로운 접근법으로 multi-scale feature decoupling을 도입하고 있습니다. 기존의 방법들은 주로 전역 특징(global feature)에 초점을 맞추었으나, 본 연구는 다양한 정보의 분리에 주목하여 현지(local) 특징을 개별적으로 처리하고 결합합니다. 이를 통해 학생 네트워크는 단일 배치 샘플만을 사용해도 성능 향상을 이룰 수 있습니다.

- **Technical Details**: 제안된 방법은 정리된 특징 샘플 정보를 사용하여 contrastive learning과 결합되며, 이는 복잡한 샘플 처리나 큰 메모리 버퍼를 요구하지 않습니다. multi-scale feature decoupling을 통해 학생과 교사의 네트워크에서 특징을 여러 스케일로 분리하여, 전역 및 현지 특징을 따로 처리합니다. 이 과정은 feature sample information의 풍부함을 극대화하여, 학생 네트워크의 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: CIFAR-100 및 ImageNet 데이터 세트에서의 평가 결과, 제안된 방법은 기존의 방법들보다 우수한 성능을 나타냈습니다. 특히, 일부 학생 네트워크는 교사 네트워크의 성능을 초과하는 결과를 보였으며 이는 제안된 방법의 효과성을 강조합니다. 이러한 결과는 학생 네트워크가 교사 네트워크로부터 지식을 충분히 흡수할 수 있도록 하는 데 기여하고 있음을 보여줍니다.



### Compressing Model with Few Class-Imbalance Samples: An Out-of-Distribution Expedition (https://arxiv.org/abs/2502.05832)
- **What's New**: 이 논문은 소수의 샘플로 모델 압축을 다루는 주요 문제 중 하나인 클래스 불균형(class imbalance) 문제를 최초로 해결합니다. 제안된 OOD-Enhanced Few-Sample Model Compression (OE-FSMC) 프레임워크는 효과적으로 학습 분포를 재균형(rebalance)하는 데 초점을 맞추고 있으며, 이를 위해 외부 데이터(out-of-distribution, OOD)를 통합합니다.

- **Technical Details**: OE-FSMC 프레임워크는 신뢰할 수 있는 원 데이터와 OOD 데이터 간의 손실(loss)을 균형 있게 유지하기 위해 조합(distillation) 프레임워크와 클래스 의존적 가중치(class-dependent weight)를 도입합니다. 또한, OOD 데이터의 노이즈가 모델 압축 및 파인튜닝에 미치는 영향을 최소화하기 위해 새로운 손실 함수를 정의합니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 통해 실험한 결과, OE-FSMC는 기존의 소수 샘플 모델 압축 방법들과 통합되어 클래스 불균형 문제에 따른 정확도 저하를 효과적으로 완화하는 성능을 보여주었습니다. 이는 소수 샘플 환경에서도 모델의 일반화 능력을 유지하면서 높은 성능을 달성할 수 있음을 의미합니다.



### HyGEN: Regularizing Negative Hyperedge Generation for Accurate Hyperedge Prediction (https://arxiv.org/abs/2502.05827)
Comments:
          4 pages, 4 figures, 2 tables, the Web Conference (WWW) 2025

- **What's New**: 본 논문에서는 기존의 하이퍼엣지 예측(hyperedge prediction) 방법이 데이터 희소성을 극복하지 못하는 문제를 제기합니다. 저자들은 HyGEN이라는 새로운 방법을 제안하며, 이는 긍정적인 하이퍼엣지를 이용해 더 현실적인 부정적 하이퍼엣지를 생성하는 네거티브 하이퍼엣지 생성기(negative hyperedge generator)를 포함합니다. 또한, 생성된 하이퍼엣지가 부정적으로 생성되는 것을 방지하는 정규화 항(regularization term)을 도입합니다.

- **Technical Details**: 하이퍼그래프(hypergraph)는 실세계 객체 간의 고차원 관계를 모델링할 수 있는 일반화된 데이터 구조입니다. 하이퍼엣지 예측의 일반적인 접근법은 하이퍼그래프 신경망(hypergraph neural networks)을 사용하여 노드의 임베딩을 생성하고, 이 임베딩을 기반으로 하이퍼엣지 후보의 점수를 매기는 것인데, 이 과정에서 효율적인 네거티브 샘플링이 중요합니다. 저자들은 기존의 간단한 휴리스틱 방식에서 벗어나, 긍정적인 샘플을 기반으로 하는 가이드를 통해 보다 효율적인 네거티브 샘플링을 수행합니다.

- **Performance Highlights**: 저자들은 6개의 실제 하이퍼그래프에 대한 광범위한 실험을 통해 HyGEN이 기존의 4가지 최첨단 하이퍼엣지 예측 방법을 지속적으로 초월하는 것을 입증했습니다. HyGEN의 도입으로 인해 하이퍼엣지 예측의 정확도와 안정성이 상당히 향상되었습니다. 이 연구는 추천 시스템(recommender systems)과 사회 네트워크 분석(social network analysis) 등 다양한 애플리케이션에 활용될 수 있습니다.



### MindCraft: Revolutionizing Education through AI-Powered Personalized Learning and Mentorship for Rural India (https://arxiv.org/abs/2502.05826)
- **What's New**: MindCraft는 인도 농촌 지역의 교육 시스템을 혁신하기 위해 인공지능(AI)을 활용하여 개인화된 학습 경험을 제공하는 현대적인 플랫폼입니다. 이 플랫폼은 농촌 학생들이 교육에서 겪는 장벽을 해결하기 위한 솔루션을 제시하며, 멘토십을 통해 학생들의 가능성을 극대화하고 자원 공유를 촉진합니다. MindCraft의 목표는 기술을 통해 교육의 격차를 줄이고 더 포용적이며 강력한 사회를 만드는 것입니다.

- **Technical Details**: MindCraft는 AI 기술을 활용하여 각 학생의 고유한 필요에 맞춰 개인화된 학습 경로를 제공합니다. 이 플랫폼은 학습 자료, 멘토십, 협력적 자원을 통합하여 농촌 학생들이 포괄적인 학습 경험을 받을 수 있도록 합니다. AI에 기반한 학습 시스템은 데이터 분석, 추천 기술을 통해 학습을 더욱 효과적으로 만들며, 정서적 지지와 정보 제공을 통해 학생과 교육자가 함께 성장할 수 있는 환경을 조성합니다.

- **Performance Highlights**: MindCraft는 인도 농촌 지역의 교육 격차를 해소하기 위해 AI 기반의 멘토링 및 개인화된 학습 도구를 제공합니다. 이를 통해 학생들은 더 나은 진로 지도를 받고, 맞춤형 교육 콘텐츠를 통해 자신의 학습 목표를 달성할 수 있도록 지원받습니다. 이 플랫폼은 농촌 지역에서 높은 교육 품질을 보장하며, 학생들이 감소된 사회경제적 불리함을 극복할 수 있도록 하는 데 중점을 두고 있습니다.



### Delta - Contrastive Decoding Mitigates Text Hallucinations in Large Language Models (https://arxiv.org/abs/2502.05825)
- **What's New**: 논문에서 제안하는 Delta는 text 기반의 대형 언어 모델(LLM)에서 발생하는 hallucination을 효과적으로 완화하는 새로운 방법입니다. 기존 모델 retraining이나 추가 데이터 없이 inference 과정에서 직접 hallucination을 줄일 수 있는 방안으로서, 입력 프롬프트의 일부를 랜덤하게 마스킹하고 원본 및 마스킹된 입력에 대한 출력 분포를 대비시킵니다. 이 접근 방식은 LLM을 실제 시스템에 배포하기 용이하고 계산 효율성이 높은 특징을 지닙니다.

- **Technical Details**: Delta는 contrastive decoding 방식을 활용하여 마스킹된 입력과 마스킹되지 않은 입력의 출력을 비교합니다. 이 방법은 입력 시퀀스의 토큰을 랜덤하게 마스킹하여, 특정한 경우에 hallucination이 발생할 가능성이 높은 출력을 생성하도록 유도합니다. 이후, 마스킹된 입력에서 생성된 로그잇(logits)을 원래의 로그잇에서 빼는 과정을 통해, hallucination에 덜 영향을 받는 '클린' 로그잇을 추출하여 정확한 출력을 도출합니다.

- **Performance Highlights**: Delta의 성능은 SQuAD v1.1 및 v2 데이터셋에서 각각 약 3 및 6%의 정확도 향상을 달성하였으며, SQuAD v2의 no-answer exact match에서 10% 이상의 점수 개선을 보여 주었습니다. TriviaQA 및 Natural Questions 데이터셋에서도 개선 효과를 나타내어, context-rich한 데이터셋에서 hallucination을 완화하고 성능을 높이는 데 효과적임을 입증하였습니다.



### The Curse of Depth in Large Language Models (https://arxiv.org/abs/2502.05795)
- **What's New**: 이번 연구에서 우리는 'Curse of Depth'(깊이의 저주)라는 개념을 소개합니다. 이는 최신 LLMs(대형 언어 모델)에서 깊은 층이 기대보다 효과적이지 않다는 최근 관찰을 강조하고 설명하며 이를 해결하기 위한 방법을 제시합니다. 여러 인기 있는 LLM 패밀리에서 이 현상이 광범위하게 존재함을 확인했으며, 깊은 층의 비효율적 원인은 Pre-Layer Normalization(Pre-LN)의 광범위한 사용에 있다고 분석하였습니다.

- **Technical Details**: Pre-Layer Normalization은 Transformer LLM의 훈련을 안정화하지만, 층 깊이에 따라 출력의 분산이 기하급수적으로 증가한다는 문제점을 가지고 있습니다. 이로 인해 깊은 Transformer 블록의 미분값이 정체 행렬(identity matrix)에 가까워져 훈련에 거의 기여를 하지 않게 됩니다.  이런 문제를 해결하기 위해, 우리는 LayerNorm Scaling을 제안하며, 이는 층 정규화의 출력을 그 깊이의 제곱근으로 역으로 스케일 조정합니다.

- **Performance Highlights**: LayerNorm Scaling을 적용한 실험 결과는 LLM의 사전 훈련 성능을 Pre-LN와 비교하여 현저하게 향상시키는 것을 보여줍니다. 또한 이러한 개선은 감독하에 세밀 조정할 때도 매끄럽게 이어집니다. 이러한 성장은 LayerNorm Scaling이 더 깊은 층이 훈련 중에 보다 효과적으로 기여하도록 한다는 점에서 기인합니다.



### EPBC-YOLOv8: An efficient and accurate improved YOLOv8 underwater detector based on an attention mechanism (https://arxiv.org/abs/2502.05788)
- **What's New**: 이 연구에서는 YOLOv8의 백본에 채널 및 공간 주의 메커니즘을 통합하고, FasterNeXt의 Pointwise Convolution을 활용하여 FasterPW 모델을 구축하였습니다. 또한 BiFPN에서 영감을 받은 WFPN 구조를 사용하여 크로스 스케일 연결과 강건성을 개선하였습니다. 이러한 개선된 프레임워크는 CARAFE를 통해 세밀한 특징 재조합을 수행하여 수중 이미지 저하 문제를 해결하며, URPC2019와 URPC2020 데이터셋에서 각각 76.7%와 79.0%의 mAP@0.5 점수를 기록했습니다.

- **Technical Details**: 이 논문은 YOLOv8 모델을 기반으로 여러 가지 개선 사항을 제안합니다. EMA(다중 스케일 주의 모듈)와 YOLOv8의 C2f 백본을 통합하여 다양한 스케일의 타겟에 대한 반응성을 높이고, FasterNext 모듈을 통해 변환된 FastPW 모델에서는 부분 합성곱을 Pointwise Convolution으로 교체하여 경량화 및 특징 추출 능력을 향상시켰습니다. 또한 WFPN의 크로스 스케일 연결 및 가중 특징 융합을 통해 정보 통합을 최적화하고, CARAFE로 업샘플링을 적용하여 작은 타겟 정보를 효과적으로 유지합니다.

- **Performance Highlights**: EPBC-YOLOv8 수중 객체 탐지기는 컴퓨팅 효율성과 정확성 사이의 균형을 달성합니다. URPC2019 및 URPC2020 데이터셋에서 각각 76.7%와 79.0%의 mAP@0.5 점수를 기록하여 기존 YOLOv8에 비해 각각 2.3% 및 0.7% 더 나은 성능을 보였습니다. 이러한 성능 향상은 수중 생물 탐지의 정확도를 높여, 관련 분야에서의 적용 가능성을 넓힐 것입니다.



### WatchGuardian: Enabling User-Defined Personalized Just-in-Time Intervention on Smartwatch (https://arxiv.org/abs/2502.05783)
Comments:
          Under submission

- **What's New**: 본 연구에서는 개인의 특정 바람직하지 않은 행동에 대한 개별 개입을 정의할 수 있는 스마트워치 기반의 WatchGuardian 시스템을 소개합니다. 이 시스템은 몇 개의 샘플만으로도 사용자 맞춤형 개입을 가능하게 합니다. 기존의 JITI(Just-In-Time Interventions) 시스템은 일반적인 건강 행동에 초점을 맞췄지만, 개인의 미세한 행동(예: 손톱 물어보기 등)을 조정하기 위한 유연성을 제공합니다.

- **Technical Details**: WatchGuardian은 사전 학습된 관성 측정 장치(Inertial Measurement Unit, IMU) 모델을 기반으로 하여 몇 번의 샘플로 새로운 행동을 감지하기 위해 few-shot learning 파이프라인을 개발했습니다. 이 모델은 공개된 손 제스처 데이터셋에서 미세 행동에 대한 기능 임베딩 능력을 향상시키기 위해 finetuning 과정을 거쳤습니다. 데이터 증대(data augmentation) 및 데이터 합성(data synthesis) 기술을 통해 각 개인의 개인화된 바람직하지 않은 행동에 대한 추가 분류 층을 훈련했습니다.

- **Performance Highlights**: 전체 26명의 참가자를 대상으로 한 오프라인 평가에서 WatchGuardian은 3, 5, 10개의 예시로 평균 정확도 76.8%, 84.7%, 87.7%와 F1 점수 74.8%, 84.2%, 87.2%를 달성하였습니다. 또한 21명의 참가자와 4시간의 개입 연구를 통해 시스템이 바람직하지 않은 행동을 64.0±22.6% 감소시켰고, 기존의 규칙 기반 개입 방법보다 29.0% 더 우수한 성능을 보였습니다. 이러한 결과는 개인화된 AI 기반 JITI 시스템의 유효성을 입증하며 향후 전개 방향에 대한 인사이트를 제공합니다.



### Predictive Crash Analytics for Traffic Safety using Deep Learning (https://arxiv.org/abs/2502.05777)
- **What's New**: 이 연구는 전통적인 자동 충돌 분석 시스템의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 기존 모델이 정적 통계 모델과 역사적 데이터에 의존하는 것과 달리, 본 연구는 ensemble learning 방법과 multi-modal data fusion을 결합하여 실시간 충돌 위험 평가 및 예측을 가능하게 합니다. 특히, 공간-시간 충돌 패턴과 환경 조건을 통합한 계층적 심각도 분류 시스템을 개발하여 기존 통계 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 연구의 주요 기술적 기여는 충돌 위치 데이터, 사건 보고서, 날씨 조건을 통합한 새로운 feature engineering 기술입니다. 이 시스템은 0.893의 Mean Average Precision (mAP)을 달성하며, 이는 현재의 최첨단 방법보다 15% 향상된 수치입니다. 또한, 500,000개의 초기 충돌 기록을 59,496개의 고품질 샘플로 필터링하여 광범위한 검증을 수행하였습니다.

- **Performance Highlights**: 위험 예측의 정확도는 92.4%, 핫스팟 식별의 정밀도는 89.7%에 달합니다. 본 시스템은 peak load에서 1,000개의 동시 요청을 처리하면서도 100ms 이하의 응답 시간을 유지할 수 있는 확장 가능한 실시간 예측 시스템을 특징으로 하며, 강력한 데이터 클리닝 파이프라인과 적응형 feature 생성 기능을 포함하고 있습니다.



### PIPA: Preference Alignment as Prior-Informed Statistical Estimation (https://arxiv.org/abs/2502.05773)
- **What's New**: 이 연구에서 우리는 Prior-Informed Preference Alignment (PIPA)라는 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 RL(강화학습) 없는 확률적 접근 방식으로, 언어 모델의 선호도 정렬을 최대 우도 추정(MLE) 문제로 재정의합니다. PIPA는 쌍으로 된 데이터와 쌍이 아닌 데이터 모두를 수용하며, 다양한 사전 정보를 통합한 두 가지 변형인 PIPA-M과 PIPA-N을 개발하여 성능을 개선합니다.

- **Technical Details**: PIPA 프레임워크는 실질적으로 사전 제약 조건을 포함한 조건부 확률 추정 문제로 선호 정렬 문제를 정의합니다. PIPA-M과 PIPA-N은 언어 모델의 학습에서 사전 정보를 통합하는 새로운 방법을 제안합니다. 이러한 알고리즘은 기존의 DPO와 KTO 알고리즘의 특별한 경우로 해석될 수 있으며, 각기 다른 데이터 설정에 유연하게 적용됩니다.

- **Performance Highlights**: PIPA-M 및 PIPA-N 알고리즘은 GSM8K와 MATH 벤치마크에서 기존 알고리즘에 비해 3%~10% 성능 향상을 보여주었습니다. 특히, 추가적인 학습이나 컴퓨팅 비용 없이도 성능을 개선할 수 있는 점이 중요한 특징입니다. PIPA는 기존의 정렬 프레임워크와 쉽게 통합될 수 있는 플러그인 손실 설계로 기능하여 다양한 데이터 생성 파이프라인에서 활용 가능합니다.



### Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails (https://arxiv.org/abs/2502.05772)
- **What's New**: 이 논문에서는 MultiFaceted Attack이라는 새로운 공격 프레임워크를 제안하여 Vision-Language Large Models (VLLMs)의 Multi-Layered Defenses를 체계적으로 우회하는 방법을 소개합니다. 이 공격은 Visual Attack, Alignment Breaking Attack, Adversarial Signature의 세 가지 보완적인 측면으로 구성되어 있습니다. 연구 결과는 현재 상용 VLLMs에서 61.56%의 흑상자 공격 성공률을 달성하여, 기존 방법보다 최소 42.18% 이상의 개선을 보였습니다.

- **Technical Details**: VLLMs는 이미지와 텍스트 정보를 통합하여 두 가지 입력 형식을 모두 이해하는 작업을 수행할 수 있도록 설계되었습니다. 이 논문에서는 VLLMs의 안전 메커니즘을 우회하기 위해 Visual Attack, Alignment Breaking Attack, Adversarial Signature와 같은 세 가지 공격 전략을 제안합니다. 이러한 각 공격 전략은 서로 보완적이며 다양한 실제 시나리오에서 유연하게 활용될 수 있습니다.

- **Performance Highlights**: MultiFaceted Attack의 실험 결과는 현재 VLLMs의 안전 메커니즘에서 중요한 취약점을 드러내며, 보다 강력한 방어 전략의 필요성을 강조합니다. 이 공격 방식은 기존 SOTA 방법에 비해 더 나은 전환 성능을 보여주며, 상용 VLLMs에서 높은 효율성을 입증했습니다. 이러한 결과는 공격 메커니즘의 효과성을 나타내며, 미래의 연구에 중요한 기초 자료를 제공합니다.



### UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Contro (https://arxiv.org/abs/2502.05749)
- **What's New**: UniDB라는 새로운 통합 확장을 통해 기존의 확산 다리 모델을 일반화하여 제안합니다. 이 모델은 Stochastic Optimal Control(SOC) 원칙을 통해 최적 제어기를 계산하고, 이를 통해 기존의 Doob의 h-transform 접근법의 한계를 극복합니다. 특히, UniDB는 SOC 비용 함수의 단말 패널티 계수를 조정하여 이미지 세부 사항을 향상시키는데 도움을 줍니다.

- **Technical Details**: UniDB는 SOC 기반 최적화 문제를 통해 확산 다리 모델을 구성하고, 최적 제어기에 대한 폐쇄 형태의 해를 도출합니다. 이 과정에서 Doob의 h-transform이 SOC 비용 함수에서 단말 패널티 계수가 무한대로 접근할 때의 특수 사례로 나타난다는 점을 명확히 합니다. 또한, 기존 모델들에 비해 성능 향상을 위해 코드 수정을 최소화하며 쉽게 구현할 수 있도록 설계되었습니다.

- **Performance Highlights**: UniDB는 다양한 이미지 복원 작업에서 최신 기술 수준의 결과를 달성하였으며, 이에는 초해상도(DIV2K), 인페인팅(CelebA-HQ), 비 오는 날의 이미지 복원(Rain100H)이 포함됩니다. 이는 제안한 프레임워크의 뛰어난 이미지 품질과 다양한 시나리오에서의 적응성을 보여줍니다.



### RECOVER: Designing a Large Language Model-based Remote Patient Monitoring System for Postoperative Gastrointestinal Cancer Car (https://arxiv.org/abs/2502.05740)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전이 원격 환자 모니터링(Remote Patient Monitoring, RPM) 시스템에 어떻게 기여할 수 있는지를 연구합니다. 본 논문에서는 RECOVER라는 LLM 기반의 RPM 시스템을 설계하였으며, 이는 GI 암 환자를 위한 수술 후 관리에 중점을 두었습니다. 또한, 임상 가이드라인과 정보 요구를 통합하기 위한 여섯 가지 주요 설계 전략을 도출했습니다.

- **Technical Details**: RECOVER 시스템은 대화형 에이전트와 임상 직원이 사용할 수 있는 인터랙티브 대시보드를 특징으로 합니다. 이 시스템은 수술 후 GI 암 진료를 효율적으로 지원하기 위해 설계되었습니다. 또한, 참여 디자인 세션을 통해 의료진과 암 환자와의 긴밀한 협력을 촉진하여 임상적 요구 사항을 충족시키고자 하였습니다.

- **Performance Highlights**: RECOVER는 파일럿 시스템으로 활용되어 네 명의 임상 직원과 다섯 명의 환자에게서 설계 전략의 구현을 평가하는 데 사용되었습니다. 이 과정에서 중요한 설계 요소를 식별하고 책임 있는 AI에 대한 통찰을 제공하였으며, 향후 LLM 기반 RPM 시스템의 기회에 대한 방향성을 제시하였습니다.



### Mitigating Sensitive Information Leakage in LLMs4Code through Machine Unlearning (https://arxiv.org/abs/2502.05739)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)과 그 중에서도 코드 생성을 위한 LLMs4Code의 개인 정보 보호 문제를 다루고 있습니다. 특히, 학습 과정에서 민감한 정보가 메모리화되어 유출될 수 있는 문제인 memorization 문제를 해결하기 위한 기계 잊기(Machine Unlearning, MU) 기술을 평가합니다. 이 연구는 세 가지 최신 MU 알고리즘을 기반으로 LLMs4Code의 개인 정보 소거 가능성을 실증적으로 분석하여, 안전하게 코드를 생성할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: LLMs와 LLMs4Code는 자연어 처리와 프로그래밍 분야에서 중요한 혁신으로 자리 잡고 있습니다. LLMs4Code는 프로그래밍 TASK에 맞게 설계된 모델로, 자연어와 프로그래밍 언어 지식을 통합하여 코드 제안, 자동완성 등에서 뛰어난 성능을 보입니다. 본 연구는 세 가지 ML 알고리즘을 통해 LLMs4Code의 사전 학습 데이터에서 민감한 정보를 안전하게 '잊게' 하여 개인 정보 보호를 강화하는 방안에 초점을 맞추고 있습니다.

- **Performance Highlights**: 본 연구의 결과는 LLMs4Code의 개인 정보 유출 문제를 50% 이상 낮출 수 있음을 보여줍니다. 이 과정에서 모델의 코드 생성 능력은 거의 영향을 받지 않으며, 민감한 정보가 단순한 변수명으로 대체되는 경향을 보였습니다. 그러나, 잊기 과정을 거친 후에는 간접적인 민감 정보 유출 가능성이 증가하는 것을 관찰하여 향후 연구에 대한 필요성을 제기합니다.



### Rethinking Link Prediction for Directed Graphs (https://arxiv.org/abs/2502.05724)
Comments:
          30 pages

- **What's New**: 이 논문에서는 방향 그래프(Directed Graph)의 링크 예측(Link Prediction) 문제에 대한 새로운 접근 방식을 제안합니다. 특히, 기존 방법들의 표현력(expressiveness)을 평가하기 위한 통합 프레임워크를 제시하며, 이 지표에 대한 심도 있는 분석을 수행합니다. 또한, 실험 환경의 한계를 극복하기 위해 DirLinkBench라는 새로운 벤치마크를 소개하여 방향 그래프 예측 작업에 대한 종합적인 평가를 제공합니다.

- **Technical Details**: 링크 예측을 위한 기존의 방법들은 임베딩(embedding) 기법과 그래프 신경망(Graph Neural Networks, GNNs)으로 분류됩니다. 임베딩 기법은 각 노드에 대해 두 가지 별도의 임베딩, 즉 출처(source) 및 대상(target) 임베딩을 생성하여 비대칭(asymmetric) 특성을 보존합니다. GNNs는 여러 종류의 임베딩을 생성하며, 이를 통해 방향성 링크 예측을 수행하고자 합니다.

- **Performance Highlights**: DirLinkBench에서의 결과는 기존 방법들이 다양한 데이터셋에서 강력한 성능을 보이는 데 어려움을 겪고 있음을 보여줍니다. 반면, DiGAE라는 단순한 방향 그래프 오토인코더가 전반적으로 마지막 성능에서 우수한 결과를 기록했습니다. 또한, 새로운 SDGAE로 알려진 오토인코더는 DirLinkBench에서 최첨단 성능(SOTA)을 달성하며, 방향 링크 예측의 핵심 요소를 분석하고 미래의 연구 과제를 제시합니다.



### Pareto-Optimality, Smoothness, and Stochasticity in Learning-Augmented One-Max-Search (https://arxiv.org/abs/2502.05720)
- **What's New**: 이 논문에서는 One-max search 문제를 해결하기 위한 새로운 알고리즘을 제안합니다. 기존의 접근 방식은 일관성과 강건성 사이의 최적의 균형을 이루지 못했습니다. 하지만 이 연구에서는 이 두 가지 목표를 동시에 달성하는 첫 번째 알고리즘을 발표합니다.

- **Technical Details**: 새로운 알고리즘은 stochastic learning-augmented 설정에서 One-max search의 성능을 분석하기 위해 얻어진 smoothness를 활용하는 방법을 제시합니다. Appendix에서는 알고리즘의 핵심적인 결과와 프로세스를 자세하게 설명합니다. 부록 B와 C는 각각 3, 4장에 대한 증명을 포함하고 있으며, 부록 D에서는 오류 분석과 관련된 결과를 모아두었습니다.

- **Performance Highlights**: 제안된 알고리즘은 기존 방법들보다 일관성과 강건성을 동시에 보장하며, 이론적으로 우수한 성과를 나타냅니다. 이를 통해 랜덤성을 포함한 가격 및 예측에 대한 분석이 가능하다는 점이 강조됩니다. 이 접근법은 온라인 결정-making 문제의 더 나은 해결책을 제시합니다.



### Extended Histogram-based Outlier Score (EHBOS) (https://arxiv.org/abs/2502.05719)
- **What's New**: 이번 연구에서는 기존의 HBOS(Histogram-Based Outlier Score) 방법을 개선하기 위해 EHBOS(Extended Histogram-Based Outlier Score)를 제안합니다. EHBOS는 이차원 히스토그램을 사용하여 피쳐 쌍 간의 의존성을 포착함으로써, HBOS의 한계를 극복하고 더 정교한 이상치 탐지 기능을 제공합니다. 이는 특히 피쳐 간의 상호작용이 중요한 데이터셋에서 효과적입니다.

- **Technical Details**: EHBOS는 전통적인 HBOS가 가정하는 피쳐 독립성(factor independence)을 넘어, 피쳐 간의 관계를 고려하여 이상치를 탐지합니다. 이 방법은 17개의 벤치마크 데이터셋에서 평가되었으며, 다양한 이상탐지 시나리오에서의 효과성과 강건성을 입증하였습니다. 특히, 농도를 평가하기 위한 ROC AUC(Receiver Operating Characteristic Area Under Curve) 측면에서도 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: EHBOS는 여러 데이터셋에서 HBOS보다 우수한 성능을 발휘했으며, 피쳐 상호작용이 이상 구조를 정의하는 데 중요한 역할을 하는 경우에 특히 두드러진 개선을 이루었습니다. 이러한 결과는 EHBOS가 HBOS의 유용한 확장으로, 복잡한 피쳐 의존성을 모델링하는 데 강력한 도구가 될 수 있음을 강조합니다.



### Proving the Coding Interview: A Benchmark for Formally Verified Code Generation (https://arxiv.org/abs/2502.05714)
Comments:
          8 pages, to appear at the 2025LLM4Code Workshop at ICSE 2025

- **What's New**: 본 논문에서는 Formal Verified Automated Programming Progress Standards(FVAPPS)를 소개하며, 이는 총 4715개의 샘플로 이루어진 프로그래밍 및 정 correctness 증명을 위한 벤치마크로, 가장 큰 formal verification benchmark입니다. 1083개의 신중히 선별되고 품질이 관리된 샘플이 포함되어 있습니다. 기존 APPS(Automated Programming Progress Standards)는 Python에서 프로그래밍 퍼즐을 해결하고 유닛 테스트에 대해 검증하는 벤치마크로 사용되어 왔으며, 이번 연구에서는 Lean 4의 정리에 대한 영 문을 일반화하였습니다.

- **Technical Details**: FVAPPS는 문제의 해결 함수 서명과 정리 명제를 포함하는 4715개의 Lean 4 파일로 구성되며, 모든 샘플은 실제 증명 없이 'sorry' 키워드를 통해 컴파일 가능합니다. APPS 데이터셋을 기반으로 하여, 소프트웨어 공학의 유용한 문제를 문서화하고 프로세스를 개선하여, 프로그래밍 기술 평가에 적합한 문제를 생성합니다. Lean을 선택한 이유는 이 도구가 수학의 성장 속도가 빠르고, 범용 프로그래밍에 적합한 잠재력을 가지고 있기 때문입니다.

- **Performance Highlights**: Sonnet와 Gemini와 같은 모델은 406개의 정리를 샘플로 하여 각각 30%와 18%의 정확도로 증명하는 성능을 보였습니다. 이러한 성과는 머신러닝 및 프로그램 합성 커뮤니티에 프로그래밍 문제와 정확성 명세 해결을 도전합니다. FVAPPS 벤치마크는 모델이 정리 증명 능력을 평가할 수 있는 중요한 기준을 제공하며, 안전하고 정확한 코드 생성을 위한 기준을 마련합니다.



### 4D VQ-GAN: Synthesising Medical Scans at Any Time Point for Personalised Disease Progression Modelling of Idiopathic Pulmonary Fibrosis (https://arxiv.org/abs/2502.05713)
Comments:
          4D image synthesis, VQ-GAN, neural ODEs, spatial temporal disease progression modelling, CT, IPF

- **What's New**: 본 논문에서는 4D Vector Quantised Generative Adversarial Networks (4D-VQ-GAN) 모델을 제안하여, 진행성 폐질환인 특발성 폐섬유화증 (Idiopathic Pulmonary Fibrosis, IPF) 환자의 CT 이미지를 생성할 수 있는 기술을 개발했습니다. 이 모델은 초기 단계의 IPF 환자에 대한 미래의 CT 스캔을 예측하여, 효과적인 치료 전략을 수립하는 데 도움을 줄 수 있습니다. 특히, 4D-VQ-GAN은 비정상적인 시간 간격의 3D 이미지를 생성하여 연속적인 질병 진행 경과를 모델링할 수 있습니다.

- **Technical Details**: 4D-VQ-GAN은 두 단계로 이루어진 학습 과정을 거칩니다. 첫 번째 단계에서는 3D-VQ-GAN을 활용하여 CT 이미지를 재구성하고, 두 번째 단계에서는 Neural Ordinary Differential Equations (ODE)를 이용하여 다양한 시간 포인트에서의 잠재 임베딩의 시간적 동역학을 학습합니다. 이러한 접근법을 통해, 입력된 CT 스캔 두 개를 바탕으로 원하는 시간 포인트에 새로운 CT 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 모델의 효용성은 생성된 CT 스캔으로부터 도출된 바이오마커에 대한 생존 분석을 통해 평가되었습니다. 이 분석 결과는 생성된 CT 스캔이 환자의 생존 결과를 신뢰성 있게 예측할 수 있음을 보여주며, 개인화된 치료 계획 수립에 기여할 가능성을 강조합니다.



### Rethinking Word Similarity: Semantic Similarity through Classification Confusion (https://arxiv.org/abs/2502.05704)
Comments:
          Accepted to NAACL-main-2025

- **What's New**: 이 논문에서는 전통적인 단어 유사성 측정 방법의 한계를 극복하기 위해 새로운 유사성 측정 방법인 Word Confusion을 제안합니다. 이 방법은 단어 임베딩 간의 코사인 유사도에 의존하는 대신, 단어의 맥락에 따른 특징 기반의 분류 혼동을 활용합니다. 이를 통해 비대칭적이고 다의적인 의미 유사성을 잘 반영할 수 있는 방법으로 재구성합니다.

- **Technical Details**: Word Confusion은 맥락 임베딩을 단어 정체성에 매핑하기 위해 분류기를 훈련하고, 분류기의 혼동(confusion) 확률을 두 단어 간의 유사성 측정으로 사용합니다. 훈련 과정에서는 BERT를 활용하여 각 단어와 관련된 문맥 임베딩을 추출한 후, 로지스틱 회귀 분류기를 훈련합니다.Inference 단계에서는 새로운 단어의 맥락 임베딩을 추출하고, 훈련된 분류기를 사용하여 특정 클래스(차원)로부터 어떤 단어가 가장 유사한지를 판단합니다.

- **Performance Highlights**: Word Confusion은 MEN, WirdSim353 및 SimLex와 같은 여러 데이터 세트에서 인간의 유사성 판단과의 일치도에서 코사인 유사도와 유사한 성능을 보입니다. 특히, 이 방법은 시간적 의미 변화와 같은 실제 데이터 탐색 작업에서도 유용성을 보여주었습니다. 결과적으로, 이 접근 방식이 문화 분석(cultural analytics) 및 계산 사회 과학(computational social science) 분야의 발전을 위한 새로운 도구 개발로 이어지기를 기대합니다.



### Context information can be more important than reasoning for time series forecasting with a large language mod (https://arxiv.org/abs/2502.05699)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 시계열 예측에 어떻게 적용될 수 있는지를 탐구합니다. 다양한 기존 및 제안된 프롬프트 기법을 고려하여 짧은 시계열 및 긴 시계열 모두에 대한 예측 능력을 평가하였습니다. 결과적으로, 특정한 상황에서 최적의 프롬프트를 찾기보다 적절한 맥락 정보(Context Information)를 제공하는 것이 더 중요할 수 있다는 사실을 발견하였습니다.

- **Technical Details**: 연구에서는 여러 개의 프롬프트 기법을 평가하고, LLM이 프롬프트에 따라 기술한 절차를 따르지 못하는 단점이 있음을 확인하였습니다. 또한, 복잡한 연산을 포함하는 경우 LLM이 정확한 계산을 하지 못하는 경향이 있다는 점과 프롬프트의 의미를 잘못 이해하여 불완전한 답변을 제공하는 경우도 발견되었습니다.

- **Performance Highlights**: 이 연구는 LLM이 시계열 예측 작업에 있어 최고의 성능을 낼 수 있는 방법론을 제시한 것은 아니지만, 적절한 맥락 정보를 제공하는 것이 종종 더 효과적일 수 있다는 점을 강조합니다. 다양한 프롬프트 기법들은 각기 다른 성과를 보였으며, 이에 따라 시계열 예측에서의 단점들도 명확히 드러나게 되었습니다.



### Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks (https://arxiv.org/abs/2502.05695)
Comments:
          Submission for possible publication

- **What's New**: 이번 논문은 FFmpeg 기술과 잠재적 확산 모델(latent diffusion models, LDMs)을 통합한 실시간 적응형 비트레이트(video streaming) 비디오 스트리밍을 위한 새로운 프레임워크를 제안합니다. 이 접근 방식은 전통적인 일정 비트 전송(constant bitrate streaming, CBS) 및 적응 비트 전송(adaptive bitrate streaming, ABS)에서 발생하는 네트워크 대역폭 사용 문제, 저장 비효율성, 사용자 경험(quality of experience, QoE) 저하를 해결합니다. LDM을 사용하여 I-프레임을 잠재 공간으로 압축함으로써 시각적 품질을 유지하면서도 저장 공간과 세멘틱(semantic) 전송 절약이 가능하게 하였습니다.

- **Technical Details**: 이 논문에서 제안하는 비디오 스트리밍 방식은 LDM과 FFmpeg를 통합하여 비트레이트를 최적화하고 다양한 프레임 유형을 고려하여 효율적인 압축을 수행합니다. I-프레임을 잠재적 특성(latent features)으로 인코딩하고, 모션 데이터는 메타데이터로 인코딩하여 대역폭을 줄입니다. CNN-GRU를 사용하여 변화하는 네트워크 조건, 미디어 콘텐츠, 사용자 선호도에 맞춰 스트리밍을 최적화하는 적응 비트 전송 메커니즘이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 QoE와 리소스 효율성 측면에서 기존의 ABS 알고리즘인 BOLA, Comyco, MERINA를 초월하는 고품질 비디오 스트리밍을 달성함을 보여줍니다. 이 논문은 5G 및 미래의 포스트 5G 네트워크에서 사용될 수 있는 확장 가능하고 실시간 비디오 스트리밍의 새로운 가능성을 열어줍니다. 특히, LDM이 노이즈가 있는 무선 통신 환경에서도 프레임 간의 시간 일관성을 회복할 수 있도록 도와줍니다.



### Zero-Shot End-to-End Relation Extraction in Chinese: A Comparative Study of Gemini, LLaMA and ChatGP (https://arxiv.org/abs/2502.05694)
- **What's New**: 이번 연구는 중국어에서 제로샷(Zero-shot) 엔드투엔드(End-to-end) 관계 추출(Relation Extraction, RE)에 대한 다양한 대형 언어 모델(LLMs)의 성능을 조사합니다. 기존 연구는 주로 영어에 집중했거나 주석이 달린 엔티티를 가정했기 때문에 중국어에 대한 LLM의 효과는 거의 탐색되지 않았습니다. 본 연구에서는 ChatGPT, Gemini, LLaMA 3개의 모델을 평가하여 정확도, 효율성 및 적응성을 비교하고, 각 모델의 장단점을 제시합니다.

- **Technical Details**: 제로샷 RE는 주석된 예 없이 모델의 사전 훈련된 지식과 추론 능력을 통해 엔티티와 그 관계를 추출하는 것을 목표로 합니다. 연구에서는 DuIE 2.0 데이터셋을 사용하여 모델들이 입력된 문장에서 엔티티-관계 트리플을 추출하도록 하고, 이에 대해 조합 정확도와 조합 재현율, 조합 F1 점수를 확인합니다. 또한 의미적 일치(Semantic Matching) 평가 방법을 도입하여 엔티티 인식과 관계 표현에서의 불일치를 완화하는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, OpenAI의 gpt-4-turbo 모델이 가장 높은 F1 점수인 0.367을 기록하며 제로샷 관계 추출 작업에 대해 가장 잘 적합한 것으로 나타났습니다. Gemini 모델은 비교적 높은 재현율을 보였지만 정확도는 중간 수준에 머물렀습니다. 반면 LLaMA 모델은 모든 지표에서 저조한 성능을 보였으며, gpt-4 모델은 실시간 적용 가능성 측면에서도 효과적인 라탠시를 기록했습니다.



### Mobile Application Threats and Security (https://arxiv.org/abs/2502.05685)
- **What's New**: 이 논문은 모바일 컴퓨팅 솔루션의 보안 취약점에 대한 심층 분석을 제공합니다. 특히 스마트폰과 태블릿을 대상으로 하여, 현재 안드로이드 및 애플 iOS 시장에서의 보안 위협을 조명합니다. 사용자가 인식하지 못할 수 있는 보안 리스크와 위협에 대해 구체적으로 다루고 있습니다.

- **Technical Details**: 논문에서는 모바일 사용자들이 직면하는 다양한 보안 취약점에 대해 논의합니다. 특히, 개인 정보 보호와 관련된 문제와 사이버 범죄의 증가 추세를 강조합니다. 이 연구는 보안 위협을 분석하고, 이를 보호하기 위한 가능한 솔루션 제안을 목표로 합니다.

- **Performance Highlights**: 논문은 현재 모바일 컴퓨팅 업계에서의 보안 위협이 증가하고 있음을 경고합니다. 또한, 개인 사용자가 쉽게 간과할 수 있는 보안 리스크를 설명하고, 이를 해결하기 위한 여러 전략을 제안합니다. 이러한 접근 방식은 모바일 기기 사용자의 보안을 강화하는 데 기여할 것으로 기대됩니다.



### Machine Unlearning via Information Theoretic Regularization (https://arxiv.org/abs/2502.05684)
Comments:
          31 pages, 2 figures

- **What's New**: 이 논문에서는 머신 언러닝(ML)에서 원치 않는 정보, 즉 특정 피처(feature)나 데이터 포인트(data point)의 제거 또는 '잊기'를 효과적으로 수행하는 수학적 프레임워크를 소개합니다. 이를 위해 정보 이론적 정규화(information-theoretic regularization)를 기반으로 하여, 피처 언러닝과 데이터 포인트 언러닝 모두를 다룹니다. 이 접근법은 강력한 보장(provable guarantees)을 통해 유용성 손실을 최소화하면서도, 다양한 학습 목표를 최적화합니다.

- **Technical Details**: 피처 언러닝의 경우, 다양한 학습 목표를 최적화하기 위한 통합 솔루션을 도출합니다. 이는 엔트로피(entropy), 조건부 엔트로피(conditional entropy), KL-발산(KL-divergence) 및 조건부 확률의 에너지를 포함하며, 직관적으로 연관된 데이터를 압축(compress)하여 불필요한 정보를 제거합니다. 데이터 포인트 언러닝에서는 개인의 데이터를 삭제할 수 있는 '잊힐 권리(right to be forgotten)'를 고려하여, 강력한 검증 가능성을 제공하는 새로운 정의를 제안합니다.

- **Performance Highlights**: 이 논문에서 제안하는 머신 언러닝 프레임워크는 기능 및 데이터 포인트 언러닝을 위한 유연성을 제공하며, 다수의 머신 러닝 및 AI 애플리케이션에 쉽게 적용될 수 있습니다. 특히, 실제 활용에서의 유용성을 극대화하며, 무사가 성과 손실을 최소화하는 방식으로, 현대의 데이터 보호 요구를 충족합니다. 이 접근법을 통해 다양한 분야에서 프라이버시와 공정성을 동시에 확보할 수 있는 가능성이 더욱 커질 것으로 예상됩니다.



### On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers (https://arxiv.org/abs/2502.05672)
Comments:
          85 pages in main text + 4 pages of references + 26 pages of appendices, 12 figures in main text + 2 figures in appendices; source code available at this https URL

- **What's New**: 이번 논문은 Episodic Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning 및 Online Decision Transformers의 수렴성 및 안정성에 대한 철저한 분석을 제공합니다. 이러한 알고리즘은 게임에서 로봇 작업에 이르기까지 다양한 벤치마크에서 경쟁력 있는 성능을 보였으나, 이론적 이해는 특정 환경 조건에만 제한되어 있었습니다. 이 작업은 강화 학습을 감독 학습(supervised learning)이나 시퀀스 모델링(sequence modeling)을 통해 접근하는 광범위한 패러다임의 알고리즘들을 위한 이론적 기초를 마련하게 됩니다.

- **Technical Details**: 이 연구의 핵심은 알고리즘이 최적의 해결책을 식별할 수 있는 기본 환경의 조건에 대한 분석입니다. 우리는 또한 환경에 소음(noise)이 존재할 때 나타날 수 있는 해결책의 안정성을 평가합니다. 특히, 강화 학습의 목표 도달 객체 및 정책의 연속성과 점근적 수렴(asymptotic convergence)을 연구하며, 이는 기본적인 Markov Decision Process의 전이 커널(transition kernel)에 따라 다릅니다. 전이 커널이 결정론적 커널(deterministic kernel)의 충분히 작은 이웃에 위치할 경우 근사 최적 행동이 달성된다는 것을 보여줍니다.

- **Performance Highlights**: 정책 및 값(값 함수)의 수렴성과 안정성에 대한 첫 번째 명시적 추정치를 전이 커널에 대한 관점에서 제시하였습니다. 결정론적 커널에 대해 이러한 양들이 연속적이며, 학습 사이클이 유한한 수를 거친 후에도 점근적으로 지속적으로 유지됨을 입증하였습니다. 이론적인 측면에서 본 연구는 새로운 개념들을 강화 학습에 도입하며, 세그먼트 공간(segment spaces)에서 작업하고, 몫 토폴로지(quotient topologies)에서의 연속성을 연구하며, 동적 시스템의 고정점 이론(fixed-point theory)을 적용하는 등의 내용을 포함합니다.



### Language Models Largely Exhibit Human-like Constituent Ordering Preferences (https://arxiv.org/abs/2502.05670)
Comments:
          NAACL 2025 Main Conference

- **What's New**: 이 논문은 영어 문장에서 구성 요소의 순서 변화가 구성 요소의 무게와 복잡성과 관련이 있다는 이론을 기반으로 현대 대형 언어 모델(LLMs)의 동작을 연구합니다. 특히, Heavy NP Shift (HNPS), Particle Movement (PM), Dative Alternation (DA), Multiple PP Shift (MPP)의 네 가지 유형의 이동을 분석하여 LLM이 인간의 언어 처리 패턴과 얼마나 유사한지를 평가합니다.

- **Technical Details**: 연구에서는 구성 요소 이동을 모델링하기 위해 다양한 무게 측정값(예: word length, syllable weight, token length 등)을 사용합니다. 구성 요소의 이동은 주로 문장의 동사 뒤에 위치한 구성 요소들을 다른 위치로 재배치하는 과정을 의미하며, 원래의 의미는 유지됩니다. 이 연구는 인위적 및 자연 발생적 데이터를 포함한 새로운 구성 요소 이동 데이터 세트를 활용하여 LLM의 선호도를 평가합니다.

- **Performance Highlights**: 결과적으로 LLM은 기존의 문법 이론과 유사하게 구성 요소 순서에 대한 선호를 보이는 경향이 있음이 밝혀졌습니다. 특히 파티클 이동에서 예상치 못한 성과를 보였지만, 전반적으로 LLM은 인간의 선호와 일치하는 패턴을 따릅니다. 이러한 연구 결과는 인간 언어 처리와 LLM의 관계를 이해하는 데 중요한 통찰을 제공합니다.



### CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging (https://arxiv.org/abs/2502.05664)
Comments:
          Accepted in NAACL 2025 Findings

- **What's New**: 이번 논문에서는 CodeSim이라는 새로운 다중 에이전트 코드 생성 프레임워크를 소개합니다. 이 프레임워크는 프로그램 합성의 여러 단계인 계획 수립(planning), 코딩(coding), 디버깅(debugging)을 인간의 인지(perception) 방식에 따라 종합적으로 다룹니다. CodeSim은 입력 및 출력을 단계별로 시뮬레이션(simulation)하여 계획 검증(plan verification) 및 내부 디버깅을 수행하는 독창적인 방법을 특징으로 합니다.

- **Technical Details**: CodeSim은 기존의 외부 도구 기반의 반복 디버거(iterative debuggers) 방식에서 벗어나 초기 코드 생성의 품질 문제를 해결하기 위해 설계되었습니다. 연구팀은 이를 통해 코드 생성의 정확성을 향상시키며, 다양한 방법으로 생성된 프로그램을 정교하게 다듬을 수 있습니다. 실험은 7개의 도전적인 문제 해결 및 프로그램 합성 벤치마크에서 진행되었으며, 그 결과 CodeSim의 뛰어난 코드 생성 능력이 입증되었습니다.

- **Performance Highlights**: CodeSim은 HumanEval 95.1%, MBPP 90.7%, APPS 22%, CodeContests 29.1% 등 새로운 최첨단(pass@1) 결과를 달성했습니다. 특히, 외부 디버거와 연계(cascaded)할 경우 더 큰 성능 향상이 가능하다는 점도 강조되었습니다. 이 연구는 오픈 소스를 통해 추가 연구 및 개발을 촉진하기 위해 공개되었습니다.



### KMI: A Dataset of Korean Motivational Interviewing Dialogues for Psychotherapy (https://arxiv.org/abs/2502.05651)
Comments:
          Accepted at NAACL 2025 Main Conference

- **What's New**: 이 논문은 AI 주도 정신 건강 지원을 위한 챗봇의 발전을 위해 새로운 출발점을 제시합니다. 특히, Motivational Interviewing (MI) 이론에 기반한 최초의 한국어 대화 데이터셋인 KMI를 소개하며, 이를 통해 기존 데이터셋의 한계를 극복하고자 합니다. KMI는 1,000개의 고품질 한국어 MI 대화를 포함하고 있으며, 전문가 평가를 통해 데이터셋의 질과 유용성을 입증했습니다.

- **Technical Details**: 제안된 방법은 두 개의 시뮬레이터인 치료사 시뮬레이터와 고객 시뮬레이터를 포함하여, 실시간으로 발화를 생성하는 구조를 가집니다. 이를 통해 전문가 치료사 모델로부터 얻어진 행동 선택을 모방하여 MI 발생을 예측하는 MI 예측 모델을 훈련시킵니다. 그 결과, KMI 데이터셋은 한국 사회의 정신 건강 문제를 반영하여 실제 상황에 기반한 대화를 생성합니다.

- **Performance Highlights**: KMI 데이터셋은 전문가들에 의해 광범위한 평가를 통해 MI의 본질을 핵심적으로 포착하며, 챗봇 개발을 위한 실용적인 자원을 제공합니다. 새로운 MI 이론에 기반한 평가 지표를 도입하여, 생성된 대화가 MI의 정신에 얼마나 부합하는지를 직접적으로 측정합니다. 이러한 평가는 KMI의 질과 전문성을 입증하는 중요한 근거가 됩니다.



### Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs (https://arxiv.org/abs/2502.05641)
- **What's New**: 본 연구는 여러 형태의 입력을 기반으로 현실적이고 물리적으로 기반한 인간 행동을 생성하는 것을 중점적으로 다룬다. 이러한 입력은 VR 컨트롤러, 부분 키포인트 애니메이션, 비디오에서 적용된 컴퓨터 비전, 또는 고수준의 운동 목표 등 다양하다. 이를 통해, 제어 요건이 부족한 상황에서도 여러 기술을 무리 없이 전환하고 실패에서 복구하는 저수준의 유연한 휴머노이드 컨트롤러의 필요성을 강조한다.

- **Technical Details**: Masked Humanoid Controller (MHC)는 다목적 모방 학습 방법론을 활용하여 증강되고 선택적으로 마스킹된 모션 데모를 적용한다. 연구진은 MHC의 교육 방법론을 통해 비동기 입력 명령에 개입하는 능력, 여러 모션 시퀀스의 요소 결합, 희박한 다중 모드 입력에서 미지의 부분을 완수하는 능력을 정확히 구현할 수 있도록 했다. 이러한 방식으로 87가지 다양한 기술 세트로 학습된 MHC는 비고정 IO마다 유연한 변화를 지원하며 새롭게 정의된 사용자의 요구를 받아들일 수 있다.

- **Performance Highlights**: 연구의 주요 성과는 MHC가 복잡한 목표 지향 행동을 지원할 수 있는 능력을 갖추었다는 것이다. 이를 통해 MHC는 고수준의 사용자 특화 목표를 충족시키기 위해 유한 상태 기계 (Finite State Machine) 및 데이터 기반 계획자와 통합될 수 있어, 추가적인 파인튜닝 없이도 새로운 작업을 해결할 수 있다. 결과적으로, 이는 고급 목표를 위한 효율적이고 재사용 가능한 동작 행동 제어를 가능하게 한다.



### ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports (https://arxiv.org/abs/2502.05638)
- **What's New**: 이 논문은 유럽의 의료 시스템에서 레거시(legacy) 임상 데이터 처리를 위해 혁신적인 솔루션을 제공하고자 하는 프로젝트의 결과를 제시합니다. 연구팀은 구조화되지 않은 임상 보고서에서 환자 이력, 진단, 치료 등의 정보를 추출하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 활용하였습니다. 또한 소규모 모델을 세밀하게 조정(fine-tuning)하여 더 큰 모델보다 우수한 성과를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 다양한 크기의 오픈 소스 LLM을 평가하며, 처음에는 단순한 프롬프트(prompts)로 테스트하고 후에 소규모 모델인 Llama 시리즈를 세밀하게 조정하여 정확성을 향상시켰습니다. 새로운 데이터셋은 60,000개의 영어 임상 요약 및 24,000개의 독일어 번역으로 구성되어 있으며, 각 사례는 PubMed Central에서 추출한 요약과 JSON 형식의 구조화된 정보를 포함합니다. 이렇게 확보된 데이터셋은 다양한 의료 데이터를 다루며, 연구 재현(reproducibility)과 재사용성을 위해 공개되었습니다.

- **Performance Highlights**: 소규모 언어 모델의 세밀 조정(fine-tuning)이 기존의 대규모 모델보다 우수한 결과를 도출하여 자원제한 환경에서도 효율적인 솔루션을 제공함을 입증하였습니다. 평가에는 ROUGE, BERTScore 및 개체 수준 지표가 사용되었으며, 실험 결과는 텍스트 기반의 임상 보고서에서 구조화된 정보를 효과적으로 추출할 수 있는 가능성을 보여줍니다. 연구 결과는 의료 데이터의 품질과 상호 운용성을 높이는 데 기여할 것으로 기대됩니다.



### Adversarial Machine Learning: Attacks, Defenses, and Open Challenges (https://arxiv.org/abs/2502.05637)
- **What's New**: 이 논문은 AI 시스템의 취약점을 다루는 Adversarial Machine Learning (AML)에 대한 포괄적인 분석을 제공하며, 특히 회피 및 오염 공격(evasion and poisoning attacks)에 대해 세밀하게 설명하고 있습니다. 또한 방어 메커니즘을 수학적으로 형식화하고, 적응형 위협 모델에서 강력한 솔루션을 구현하는 데 따른 도전 과제를 논의합니다. 인증된 강건성(certified robustness), 확장성(scalability), 실제 배포(real-world deployment)와 관련된 제무리 어려움들을 강조합니다.

- **Technical Details**: 최근의 머신러닝(ML) 시스템은 ImageNet과 같은 벤치마크에서 초인간 성능을 달성했지만, 여전히 적대적 변형(adversarial perturbations)에 취약합니다. 이 논문에서는 포멀한 위협 모델과 공격 분류법, 회피 및 오염 공격의 정의, 그리고 방어 전략을 체계적으로 분석합니다. 특히 다양한 공격 방법론을 ϵ-제약 최적화(ε-constrained optimization) 프레임워크로 통합하고, 블랙박스 공격에서의 기울기 흐림 효과(gradient obfuscation effects)를 비교합니다.

- **Performance Highlights**: 회피 공격에서는 입력을 수정하여 예측 오류를 유발하되, 변형을 눈에 띄지 않게 유지하는 방법을 찾습니다. Carlini-Wagner 공격은 Lp-노름을 사용하는 강력한 최적화 기반 공격으로, 변형 크기와 공격 성공을 조절하는 문제로 설정됩니다. PGD(Projected Gradient Descent) 공격은 손실을 최대화하면서 변형을 제한된 범위 내에 두는 반복적인 방법론을 활용하여 강화된 효율성을 보여줍니다.



### XiHeFusion: Harnessing Large Language Models for Science Communication in Nuclear Fusion (https://arxiv.org/abs/2502.05615)
- **What's New**: 이번 논문에서는 핵융합 분야에서 최초로 개발된 대형 언어 모델인 XiHeFusion을 제안합니다. XiHeFusion은 오픈 소스 대형 모델 Qwen2.5-14B를 기반으로 감독 학습을 통해 최적화되었습니다. 다양한 출처에서 수집한 핵융합 관련 지식을 활용하여 모델의 학습을 지원하며, 체계적인 과학 보급을 위한 대화형 모델로 설계되었습니다.

- **Technical Details**: 핵융합 모델인 XiHeFusion을 훈련하기 위해 CommonCrawl, CNKI(중국 국가 지식 인프라), eBooks, arXiv, 학위 논문 등의 다중 출처 지식을 집합적으로 수집했습니다. 이 정보는 100만 개 이상의 질문-답변 쌍으로 가공되어 모델 훈련의 기초 자료로 사용되었습니다. Chain-of-Thought 기법을 적용하여 모델의 논리적 추론 능력을 향상시켜, 보다 정확하고 논리적인 답변을 제공할 수 있도록 했습니다.

- **Performance Highlights**: 대규모 질문지(약 184개 질문)에 대한 테스트를 통해 XiHeFusion의 과학 보급 대화 능력을 평가했습니다. 실험 결과, XiHeFusion은 핵융합 관련 지식에 대한 질문에 효과적으로 잘 대응할 수 있음을 증명했습니다. 이 모델은 이론적 응용이 가능하며, 핵융합 분야에 대한 광범위한 이해를 증진시키는 데 크게 기여할 것으로 기대됩니다.



### On Memory Construction and Retrieval for Personalized Conversational Agents (https://arxiv.org/abs/2502.05589)
Comments:
          10 pages, 5 figures, conference

- **What's New**: 본 논문에서는 긴 대화에서 일관되고 개인화된 경험을 제공하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 대화 이력을 기반으로 메모리 뱅크를 구축할 때 차원(Granularity)에 따라 한계를 보인다는 점을 발견했습니다. 특히, LLMLingua-2와 같은 프롬프트 압축 방법이 메모리 검색의 정확도를 높이는 데 효과적이라는 점을 강조합니다.

- **Technical Details**: SeCom이라는 새로운 시스템을 도입하여 대화 세그먼트 수준에서 메모리 뱅크를 구축하고, 압축 기반의 디노이징을 통해 메모리 검색을 향상시킵니다. 대화 세그멘테이션 모델을 사용하여 긴 대화를 주제에 맞게 분할하고, 메모리 유닛을 검색할 때 요약을 거치지 않고 직접 결합하여 정보 손실을 방지합니다. 이 과정에서 기본적인 언어의 중복성이 검색 시스템에 잡음으로 작용할 수 있다는 가정을 통해 메모리를 최적화합니다.

- **Performance Highlights**: SeCom은 LOCOMO와 Long-MT-Bench+와 같은 장기 대화 기준 벤치마크에서 기존의 턴 수준 및 세션 수준 방법들을 초월하는 성능을 보였습니다. 실험 결과는 세그먼트 수준 메모리 유닛과 압축 기반 디노이징 기법의 기여를 강화하며, 결과적으로 응답 생성의 정확성과 관련성을 높이는 데 성공했습니다.



### Event Stream-based Visual Object Tracking: HDETrack V2 and A High-Definition Benchmark (https://arxiv.org/abs/2502.05574)
Comments:
          Journal Extension of EventVOT, CVPR24

- **What's New**: 이번 연구에서는 EventVOT라는 새로운 대규모 고해상도 이벤트 기반 추적 데이터세트를 제안하였습니다. 또한 계층적 지식 증류 전략을 통해 다중 모달 데이터에서 단일 모달 데이터로의 변환을 효율적으로 수행하여 추적 성능을 향상시키는 방법을 제시하였습니다. 이 연구는 다양한 목표 객체에 대한 테스트 시 조정 전략을 도입하여 유연성과 성능을 개선한 점이 특징입니다.

- **Technical Details**: 제안된 방법은 이벤트 스트림을 활용하여 낮은 지연 시간의 추적을 가능하게 하며, 혁신적인 계층적 지식 증류 방식을 통해 다중 모달과 단일 모달 간의 지식을 효과적으로 전이합니다. 연구팀은 비디오 수준의 테스트 조정(test-time tuning) 전략을 적용해 실제 추적 시나리오에서 다양한 객체에 대한 모델의 적응력을 향상시켰습니다. 이를 통해 기존의 RGB 카메라에는 적합하지 않았던 접근을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 HDETrack V2는 1141개의 비디오로 구성된 EventVOT 데이터 세트와 함께 기존의 성능 기준을 초월하는 결과를 보였습니다. 다양한 벤치마크 데이터 세트(FE240hz, VisEvent, FELT)에서 충분한 실험을 통해 제안된 방법의 효과성이 입증되었습니다. HDETrack V2는 특히 EventVOT 데이터 세트에서 성능이 크게 향상된 것으로 나타났습니다.



### Low-Rank Agent-Specific Adaptation (LoRASA) for Multi-Agent Policy Learning (https://arxiv.org/abs/2502.05573)
Comments:
          31 pages, 20 figures, 13 tables

- **What's New**: 이번 논문에서는 Multi-Agent Reinforcement Learning (MARL)에서의 한계를 극복하기 위해 새로운 방법론인 Low-Rank Agent-Specific Adaptation (LoRASA)을 소개합니다. LoRASA는 공유된 정책을 기반으로 하여 각 에이전트의 정책을 특별한 '작업'으로 간주하고 개별 에이전트가 특화된 역할을 수행할 수 있도록 합니다. 이 방법은 파라미터 효율적인 전이 방법에서 영감을 받아 저랭크(adaptation matrices)를 추가하여 파라미터 공간의 희소성을 자연스럽게 유도합니다.

- **Technical Details**: LoRASA는 각 정책의 층에 소형 저랭크 адап테이션 매트릭스를 추가함으로써, 에이전트가 전문화된 행동을 할 수 있도록 합니다. 이는 파라미터 공간의 희소성을 유도하고, 이를 통해 메모리와 계산 오버헤드를 줄입니다. 방법론은 Partially Observable Markov Games (POMGs)를 기반으로 하여, 각 에이전트가 상태를 관찰하고, 해당 정책에 따라 행동을 선택하여 보상을 최대화하는 구조입니다.

- **Performance Highlights**: LoRASA는 StarCraft Multi-Agent Challenge (SMAC) 및 Multi-Agent MuJoCo (MAMuJoCo)와 같은 도전적인 벤치마크에서 기존 방법들과 비교해 성능이 동등하거나 더 나은 결과를 보였습니다. 연구 결과는 저랭크 업데이트를 활용한 파라미터 공간의 희소성이 에이전트가 다양한 전문화를 수행할 수 있도록 하며, 공유 정책의 장점을 유지할 수 있음을 보여줍니다.



### Large Multimodal Models for Low-Resource Languages: A Survey (https://arxiv.org/abs/2502.05568)
- **What's New**: 이번 연구는 저자들이 저자들이 75개의 저자들이 106개 연구를 분석하여 저자들 저자들이 저자들이 저자들 저자들을 도와줄 수 있는 기술과 접근 방식을 제시하고 있다는 점에서 새롭습니다. 여러 저자들은 다양한 접근 방식을 통해 저자들이 저자들이 저자들이 저자들이 저자들이 저자들을 다루는 데 있어 핵심 패턴을 확인했습니다. 시각적 정보는 LMMs의 성능을 향상시키는 데 중요한 역할을 하며 여전히 해결해야 할 도전 과제가 많은 상황입니다.

- **Technical Details**: 본 논문은 LMMs의 다양한 기법을 분류하여 데이터 생성, 융합 기술, 시각적 개선, 크로스-모달 전이, 합성 데이터 생성 등을 포함한 여섯 가지 주요 범주로 나누고 있습니다. 이는 저자들이 저자들 저자들을 통해 저자들이 저자들을 보다 효과적으로 처리할 수 있는 방법을 제시하고 있음을 보여줍니다. 각 섹션은 저자들이 저자들을 통해 저자들의 적용 가능성을 느낄 수 있는 중요한 기법을 다루고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 텍스트-이미지 조합이 연구에서 가장 주목받는 경향을 보였고, 특정 언어에 대한 연구 초점의 불균형이 있습니다. 특히 힌디어, 아랍어, 벵골어 같은 주요 언어들은 많은 연구의 주제가 되었으나 40개 이상의 언어들은 단 하나의 연구로만 대표되고 있습니다. 이러한 결과는 LMMs를 활용하여 보다 다양한 저자들이 저자들의 접근을 향상시킬 필요성을 강조하고 있습니다.



### ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data (https://arxiv.org/abs/2502.05567)
- **What's New**: ATLAS는 자연어 수학 정리를 기계 검증 가능한 형식 언어로 자동 변환하는 새로운 데이터 생성 프레임워크입니다. 이 프레임워크는 총 300,000개의 정리 문장으로 구성된 대규모 고품질 병렬 정리 데이터셋을 생성합니다. ATLAS 번역기는 ProofNet 데이터셋에서 80.59%(pass@8)와 92.99%(pass@128)의 정확도를 달성했습니다.

- **Technical Details**: ATLAS는 데이터 리프팅(Data Lifting), 데이터 합성(Data Synthesis), 데이터 증강(Data Augmentation)이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 데이터 리프팅에서는 Mathlib에서 수학 개념을 추출하여 개념 저장소를 구성하였고, 데이터 합성에서는 다양한 Teacher 모델과 Student 모델을 활용하여 병렬 정리 문장을 생성합니다. 이러한 프로세스는 정확한 형식 언어 표기를 생산하고, Lean 컴파일러를 통해 구문적 및 의미적 오류를 검증합니다.

- **Performance Highlights**: ATLAS 데이터셋을 기반으로 한 ATLAS 번역기는 miniF2F, ProofNet, MathQual 데이터셋에서 각각 91.60%, 80.59%, 65.47%의 정확도를 기록했습니다. 특히, pass@128 지표에서 이 번역기는 96.93%, 92.99%, 84.72%의 정확도로 최신 성능을 기록하며 기존 모델들인 base model 및 InternLM2-Math-Plus-7B를 초월했습니다.



### TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (https://arxiv.org/abs/2502.05564)
- **What's New**: TabICL은 데이터 샘플이 최대 500K개인 경우에도 효율적으로 처리할 수 있는 새로운 분류용 테이블 기반 모델입니다. 이 모델은 두 단계의 아키텍처를 통해 고정 차원의 row embeddings를 구축하며, 효율적인 ICL을 위한 transformer를 이용합니다. 이와 함께 TabICL은 기존의 TabPFNv2보다 더 효율적이며, 특히 대용량 데이터셋에서 탁월한 성능을 발휘합니다.

- **Technical Details**: TabICL은 각 테이블 셀을 기본 요소로 간주하여 열끼리의 상호작용과 각 행의 정보를 결합합니다. 이를 위해 통계적 패턴을 캡처하기 위해 distribution-aware column-wise feature embedding을 적용하고, attention 기반의 row-wise interaction을 통해 세밀한 feature 의존성을 모델링합니다. 이 과정에서 Set Transformer를 활용하여 다양한 특성을 가진 데이터를 효율적으로 처리합니다.

- **Performance Highlights**: TabICL은 TALENT 벤치마크의 200개 분류 데이터셋에서 TabPFNv2와 유사한 성능을 보이며, 대부분의 경우 더 나은 성과를 거두었습니다. 10K 이상의 샘플을 포함한 56개 데이터셋에 대해 TabICL은 TabPFNv2와 CatBoost를 능가하며 ICL의 대용량 데이터 처리 가능성을 입증합니다. 전체적으로 TabICL은 하이퍼파라미터 조정이 필요한 기존 테이블 방법보다 수십 배 빠르며, 데이터셋 크기가 커질수록 효율성이 더욱 향상됩니다.



### Dual Defense: Enhancing Privacy and Mitigating Poisoning Attacks in Federated Learning (https://arxiv.org/abs/2502.05547)
Comments:
          accepted by The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 Federated Learning (FL)에서의 데이터 프라이버시 보호와 모델 중독 공격 방어 방안을 동시에 다룬 새로운 방법론인 Dual Defense Federated learning (DDFed) 프레임워크를 제안합니다. 기존의 접근 방식들은 서로 충돌하는 요구 사항으로 인해 복잡했으나, DDFed는 전혀 새로운 참여자 역할 없이 기존의 FL 구조를 유지하면서 두 가지 문제를 해결합니다. 이 프레임워크는 완전 동형 암호화(fully homomorphic encryption, FHE) 기술을 이용해 안전하게 모델 업데이트를 집계하며, 고유한 두 단계의 이상 탐지 메커니즘을 통해 암호화된 모델 업데이트의 유사성을 계산합니다.

- **Technical Details**: DDFed는 두 가지 주요 구성 요소로 구성되며, 이를 통해 개인 정보 보호와 모델 중독 공격 방어를 동시에 수행합니다. 첫 번째로, 완전 동형 암호화(FHE)를 활용해 암호화된 모델 업데이트를 안전하게 집계합니다. 두 번째로, 알고리즘은 피드백 기반 협업 선택 프로세스를 포함하는 두 단계의 이상 탐지 메커니즘을 통해 악성 모델 감지를 수행하며, 비잔틴 클라이언트로 인한 잠재적인 개인 정보 침해를 예방하기 위한 추가 조치를 내장합니다.

- **Performance Highlights**: 다양한 모델 중독 공격 및 FL 시나리오에 대해 DDFed를 테스트한 결과, 공개 데이터셋에서 모델 프라이버시를 성공적으로 보호하고 모델 중독 공격에 대해 효과적으로 방어할 수 있음을 입증했습니다. DDFed는 교차 장치(cross-device) 및 교차 실(브루와 자사 사이의 데이터 공유 구현) 시나리오에서 모두 잘 작동하며, 기존의 방어 기법들보다 더 우수한 성능을 보여줍니다. 이는 FL의 우리 모두의 연구와 실제 적용에 큰 긍정적 영향을 미칠 가능성이 큽니다.



### Towards Learning Scalable Agile Dynamic Motion Planning for Robosoccer Teams with Policy Optimization (https://arxiv.org/abs/2502.05526)
- **What's New**: 본 논문은 동적 장애물이 존재하는 다중 에이전트 시스템을 위한 신속한 모션 계획(Dynamic Motion Planning)에 대한 새로운 학습 기반 모델을 제안합니다. 기존의 모션 계획 알고리즘들은 자원 소모가 크고 다양한 환경 변화에 적절히 적응하지 못하는 한계가 있습니다. 제안된 모델은 이질적인 팀원 간의 충돌을 방지하면서 목표 위치에 도달할 수 있는 경로를 제시합니다. 특히, 로보소커(Robosoccer) 환경을 기반으로 동작하는 간단한 예시를 통해 모델의 효용성을 검증합니다.

- **Technical Details**: 본 연구에서는 에이전트, 목표 지점, 장애물 정보를 포함하는 완전 관찰 가능한 마르코프 결정 프로세스(MDP)를 사용하여 모션 계획 문제를 정식화합니다. 에이전트는 다양한 속도와 충돌 반경을 가진 이질적인 구성원으로 설정되어 있으며, 각각의 목표 위치에 대한 순서를 할당받습니다. 각 시간 단계에서 모델은 현재 위치를 입력으로 받아 새로운 행동을 생성하는 방식으로 동작하며, 이 방법은 중앙 집중식 제어와 결합하여 작업 할당 및 스케줄링도 가능합니다.

- **Performance Highlights**: 모델은 단일 에이전트로 훈련 가능하고, 동적 환경에서 새로운 비충돌 경로를 생성하는 데 필요한 계산 비용을 최소화합니다. 제안된 방법은 자원의 효율성을 극대화하면서도 충돌을 피하는 경로 탐색을 가능하게 하여 로봇 스포츠 분야에서 적용 가능성을 보여줍니다. 또한, 구체적인 GRAPH NEURAL NETWORKS 사용을 통해 모델을 다양한 팀 구성에서 확장 가능성에 대한 미래 작업도 논의됩니다.



### IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System (https://arxiv.org/abs/2502.05512)
- **What's New**: IndexTTS 시스템은 최근 대규모 언어 모델(large language model, LLM) 기반의 텍스트 음성 합성(text-to-speech, TTS)에서 여러 혁신을 도입하였습니다. 특히 중국어 환경에서, 다중음절 문자와 저빈도 문자의 발음을 제어할 수 있는 문자-핀인 하이브리드 모델링 방식을 사용하고 있습니다. 또한, 음성 클로닝을 안정화하기 위해 새로운 조건부 인코더와 디코더 구조를 소개하여 시스템의 성능을 크게 개선하였습니다.

- **Technical Details**: IndexTTS 시스템은 XTTS 및 Tortoise 모델에 기반을 두고, 멜 스펙트로그램을 입력으로 받는 VQ(Vector Quantization) 및 FSQ(Finite-Scalar Quantization)의 기법을 비교 분석하여 코드북(codebook) 활용률을 100%에 가깝게 달성했습니다. 또한, Conformer 기반의 조건부 인코더와 BigVGAN2 기반의 음성 디코더를 적용하여 훈련 안정성과 발음 유사성을 향상시켰습니다. 이 시스템은 총 12,000개의 어휘를 사용하며, 다국어로의 확장도 용이합니다.

- **Performance Highlights**: IndexTTS는 기존의 오픈 소스 TTS 시스템인 Fish-Speech, CosyVoice2, FireRedTTS, F5-TTS보다 자연성, 콘텐츠 일관성 및 제로샷 음성 클로닝에서 월등한 성능을 보이고 있습니다. 특히, 훈련 과정이 간단하고 사용이 더 직관적이며 추론 속도가 빠르다는 장점이 있습니다. 다양한 테스트 세트를 공개하여 연구자들과 개발자들이 쉽게 접근할 수 있도록 하였으며, 최신 데모를 통해 그 성능을 확인할 수 있습니다.



### A Physical Coherence Benchmark for Evaluating Video Generation Models via Optical Flow-guided Frame Prediction (https://arxiv.org/abs/2502.05503)
- **What's New**: 이 논문에서는 물리적 일관성을 평가하기 위해 특별히 설계된 새로운 벤치마크인 PhyCoBench를 소개합니다. 이 벤치마크는 7가지 물리 원리를 포괄하는 120개의 프롬프트로 구성되어 있으며, 주로 물체의 움직임과 관련된 다양한 물리 현상을 포함합니다. 또한, 자동 평가 모델인 PhyCoPredictor를 제안하여 물리적 일관성을 보다 효과적으로 평가할 수 있는 방법을 마련하였습니다.

- **Technical Details**: PhyCoBench는 중력, 충돌, 진동, 마찰, 유체 역학, 발사 운동, 회전의 7가지 주요 물리 원리를 기반으로 다양한 물리적 현상을 평가하도록 설계되었습니다. 프롬프트는 최신 언어 모델을 활용하여 작성되었으며, 동적인 장면의 광학 흐름을 예측하는 PhyCoPredictor 모델이 개발되어 물리적 법칙을 따르는지 판단하는 데 사용됩니다. 이 모델은 초기 프레임과 텍스트 프롬프트를 입력으로 받아 광학 흐름을 예측하고, 이를 통해 동영상을 생성하는 과정에 활용됩니다.

- **Performance Highlights**: PhyCoPredictor는 기존의 수동 평가와 비교하여 자동 평가의 일관성을 입증하였으며, 현재 인적 평가와 가장 밀접한 정합성을 보임을 확인하였습니다. 이 시스템은 물리적 일관성을 평가하는 데 있어 중요한 통찰력을 제공하며, 향후 모델 최적화에 기여할 수 있습니다. 저자들은 PhyCoBench와 PhyCoPredictor를 GitHub를 통해 공개할 예정이며, 관련 데이터셋도 함께 제공됩니다.



### Vision-Ultrasound Robotic System based on Deep Learning for Gas and Arc Hazard Detection in Manufacturing (https://arxiv.org/abs/2502.05500)
Comments:
          Submitted to Engineering Applications of Artificial Intelligence

- **What's New**: 이 연구는 제조 환경에서 가스 누출과 아크 방전을 독립적으로 탐지하고 분류하기 위해 심층 학습 기반 로봇 시스템을 제안합니다. 이 시스템은 사람의 시각적인 식별 능력과 음향적인 검증 방법을 결합하여 산업 안전을 위한 혁신적인 접근 방식을 제공합니다. 특히, 112채널의 음향 카메라와 96 kHz의 샘플링 주파수를 활용하여 초음파 주파수를 포착하는 점이 특징입니다.

- **Technical Details**: 제안된 시스템은 시각 탐지 및 빔포밍(beamforming) 강화 음향 분석 파이프라인을 통합합니다. STFT(Short-Time Fourier Transform)를 사용하여 신호를 변환하고 감마 보정을 통해 주요 특징을 추출하여 신뢰성을 높입니다. Inception에서 영감을 받은 CNN(Convolutional Neural Network)은 위험 요소를 분류하여 99%의 가스 누출 탐지 정확도를 달성합니다.

- **Performance Highlights**: 이 시스템은 개별 위험 소스를 탐지할 뿐만 아니라, 시각 및 음향 센서로부터 멀티모달 데이터를 융합하여 분류의 신뢰성을 강화합니다. 실험 결과, 반향(reverberation)과 노이즈가 증가한 환경에서도 기존 모델보다 최대 44% 향상된 성능을 보였으며, 모바일 로봇 플랫폼에서 2.1초의 추론 시간을 유지하여 실시간 배포에 최적화되었습니다.



### Riemannian Manifold Learning for Stackelberg Games with Neural Flow Representations (https://arxiv.org/abs/2502.05498)
Comments:
          Stackelberg games. Manifold learning. Online learning

- **What's New**: 본 연구에서는 Stackelberg 일반합 게임에서의 온라인 학습을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크에서는 리더와 추종자라는 두 개의 에이전트가 순차적으로 상호작용하며, 중심에는 공동 행동 공간을 매끄러운 Riemannian 다양체인 Stackelberg manifold로 매핑하는 학습된 미분동형사상이 있습니다. 이를 통해 효율적인 온라인 학습 기술을 가능하게 합니다.

- **Technical Details**: Stackelberg manifold를 통해 에이전트의 보상 함수 간의 선형성을 가정하면, 표준 밴딧 알고리즘을 적용할 수 있습니다. 본 논문에서는 볼록 다양체에서의 후회 최소화에 대한 엄밀한 이론적 근거를 제공하고, Stackelberg 평형에 대한 간단한 후회에 대한 유한 시간 경계를 설정합니다. 신경 정상화 흐름(neural normalizing flows)을 활용한 이 다양체 학습 기법은 다중 에이전트 학습에 대한 새로운 가능성을 드러냅니다.

- **Performance Highlights**: 우리의 접근 방식은 표준 기준선에 비해 효과성을 보여주는 경험적 결과를 제시합니다. 이 연구 결과는 사이버 보안 및 경제적 공급망 최적화와 같은 다양한 분야에 응용될 수 있습니다. 이를 통해, Stackelberg 게임 이론과 심층 학습 기술의 통합이 다중 에이전트 환경에서의 성능을 크게 향상시킬 수 있음을 입증하였습니다.



### Multi-scale Masked Autoencoder for Electrocardiogram Anomaly Detection (https://arxiv.org/abs/2502.05494)
Comments:
          Under review in a journal

- **What's New**: 본 논문에서는 전기심장도 (ECG) 신호에서 이상 탐지를 위한 새로운 방법인 Multi-scale Masked Autoencoder for ECG anomaly detection (MMAE-ECG)을 제안합니다. 이 접근 방식은 R-peak 탐지나 심박수 분할과 같은 사전 처리 단계를 제거하여 임상에서의 실용성을 향상시킵니다. 또한, 모델은 비겹치는 세그먼트로 ECG 신호를 나누고 각 세그먼트에 학습 가능한 위치 임베딩을 할당합니다.

- **Technical Details**: MMAE-ECG는 경량 Transform 기반의 인코더-디코더 구조를 활용하여 글로벌 및 로컬 의존성을 효과적으로 캡처합니다. 이는 다중 스케일 마스킹 전략, 다중 스케일 주의 메커니즘 및 독특한 위치 임베딩을 통합하여 달성됩니다. 마스킹된 세그먼트는 단일 레이어 Transformer 블록을 사용하여 재구성되고, 추론 중에 집계 전략이 사용되어 성능이 개선됩니다.

- **Performance Highlights**: 실험 결과, MMAE-ECG는 기존 기법과 동등한 성능을 자랑하면서도 추론에 약 1/78의 부동 소수점 연산(FLOPs)을 요구하여 계산 복잡도를 크게 줄였습니다. 또한, 여러 구성 요소의 효능을 평가하기 위한 ablation 연구를 통해 다중 스케일 마스킹 오토인코더의 효과성을 입증하였습니다.



### Mechanistic Interpretability of Emotion Inference in Large Language Models (https://arxiv.org/abs/2502.05489)
Comments:
          To be submitted to the Association for Computational Linguistics (ACL 2025)

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 텍스트에서 인간의 감정을 예측하는 메커니즘을 탐구하며, 감정 표현이 모델의 특정 지역에 기능적으로 국소화되어 있음을 보여준다. 연구는 다양한 모델 패밀리와 크기를 포함하며, 감정 처리의 기능적 국소화에 대한 증거를 제공한다. 또한, 인지 평가 이론을 활용해 LLM 내부의 감정 처리 구조를 설명하고, 이를 통해 LLM이 어떻게 감정을 생성하는지를 이해한다.

- **Technical Details**: 본 연구에서는 LLM의 숨겨진 표현 위에 선형 분류기를 훈련하여 감정 관련 활성화가 가장 강하게 나타나는 영역을 탐색한다. 중간 층의 Multi-Head Self-Attention(MHSA) 유닛이 LLM의 의사결정을 형성하는 주요 역할을 하며, 이러한 유닛이 감정적으로 중요한 토큰에 지속적으로 주의를 기울임을 시각적으로 확인한다. 또한, 평가 개념을 조작하여 감정 출력을 조정하는 방법론을 통해, 이론적 기대와의 일치를 보여준다.

- **Performance Highlights**: 실험 결과, LLM은 주어진 텍스트 컨텍스트에 기반해 감정을 효과적으로 유추할 수 있는 능력을 보였다. 총 6,800개의 감정 비네트를 사용하는 crowd-enVENT 데이터셋을 통해 모델의 감정 분류 성능을 평가하였으며, 전문적인 분석을 위해 정확히 분류된 샘플을 중심으로 검토하였다. 결과적으로, 본 연구는 LLM의 감정 생성 메커니즘을 심층적으로 이해하는 데 기여하며, 감정 관련 작업에서 안전성과 정렬을 개선할 수 있는 새로운 방법론을 제시한다.



### HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation (https://arxiv.org/abs/2502.05485)
Comments:
          We require NVIDIA's approval before proceeding with the release, and we are currently processing it

- **What's New**: 이번 연구에서는 계층적 비전-언어-행동(vision-language-action, VLA) 모델이 표준 단일 구조 VLA 모델보다 비종속 데이터(off-domain data)를 더욱 효과적으로 활용할 수 있음을 제안하고 있습니다. 로봇 데이터 부족 문제를 해결하기 위해, 저렴한 off-domain 데이터를 이용해 로봇 동작 예측을 지원하는 새로운 접근 방식을 개발했습니다. 이는 로봇의 실세계 테스트와 시뮬레이션 간의 도메인 차이를 극복하는 데 기여합니다.

- **Technical Details**: 계층적 VLA 모델에서 고급 VLM (vision-language model)은 주어진 RGB 이미지와 작업 설명에 따라 로봇의 엔드 이펙터 궤적을 나타내는 대략적인 2D 경로를 예측하도록 미세 조정됩니다. 이러한 2D 경로 예측은 저수준의 3D 감지 제어 정책에게 정밀한 조작을 위한 지침으로 제공됩니다. 이러한 구조는 높은 수준의 VLM이 세부적인 행동 예측 부담에서 벗어나도록 하고, 저수준 정책의 복잡한 작업 수준의 추론 부담을 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구에서는 계층적 설계를 통해 고급 VLM이 비종속 미세 조정 데이터와 실제 로봇 테스트 시나리오 간의 큰 도메인 격차를 효과적으로 극복하는 것을 보여주었습니다. 실제 로봇 실험에서는 OpenVLA 대비 각기 다른 7개 일반화 축에서 평균 20%의 성공률 향상을 관찰하였으며, 이는 50%의 상대적인 개선을 의미합니다. 연구팀은 시각적 결과도 함께 제공하고 있습니다.



### Position: LLMs Can be Good Tutors in Foreign Language Education (https://arxiv.org/abs/2502.05467)
Comments:
          18 pages, 4 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)을 외국어 교육(FLE)에 통합하려는 노력들이 시작되고 있지만, 전통적인 학습 과제를 사용하고 있어 교육 방법론에 대한 적응 능력이 부족하다. 본 논문에서는 LLM이 효과적인 튜터로 활용될 수 있는 세 가지 주요 역할, 즉 데이터 향상(data enhancers), 작업 예측(task predictors), 그리고 에이전트(agents)로서의 기능을 제안하며, 이로써 FLE를 향상시킬 수 있는 방안을 모색한다.

- **Technical Details**: LLM은 컴퓨터 과학, 언어학, 교육학, 심리언어학 등 다양한 분야와의 융합을 통해 FLE의 도전 과제를 해결할 수 있는 잠재력을 지닌다. LLM은 자연어 이해 및 생성에서 놀라운 성능을 보여주며, 전통적인 교육 방법의 한계를 극복하고 개인화된 교육 경험을 제공하는 데 기여할 수 있다. 특히 청취, 말하기, 읽기, 쓰기라는 네 가지 핵심 기술을 LLM을 통해 효과적으로 개선할 수 있는 가능성이 있다.

- **Performance Highlights**: LLM은 학습 자료의 생성 및 피드백 제공, 상호작용을 통한 다양한 언어 학습 활동에서 큰 장점을 보여준다. 그러나 LLM의 통합은 기존의 인간 튜터를 보완해야 하며, LLM의 허위 정보(hallucination)를 방지하기 위해 고품질 데이터를 확보하는 것이 필수적이다. 향후 LLM을 FLE에 적용하기 위한 도전과 윤리적 고려 사항을 논의함으로써, 연구자와 교육자가 이 기술의 혁신적 잠재력을 활용할 수 있는 기술적 가이드를 제공하고자 한다.



### DCENWCNet: A Deep CNN Ensemble Network for White Blood Cell Classification with LIME-Based Explainability (https://arxiv.org/abs/2502.05459)
- **What's New**: 이번 연구에서는 백혈구(WBC)의 분류를 위한 새로운 앙상블 접근법인 DCENWCNet을 제안합니다. 이 모델은 세 가지 서로 다른 CNN 아키텍처를 통합하여 드롭아웃(dropout)과 맥스 풀링(max-pooling) 레이어 설정을 다양하게 구성함으로써 특징 학습을 더욱 향상시킵니다.

- **Technical Details**: DCENWCNet은 기존의 CNN 모델들이 가지는 데이터 불균형과 데이터 증강 부족 같은 문제를 해결하는 데 중점을 둡니다. 이 모델은 일반적으로 인정받는 Rabbin-WBC 데이터셋에서 검토되었으며, 편향-분산(bias-variance) 균형을 효과적으로 이루어냅니다.

- **Performance Highlights**: 모델은 평균 정확도(mean accuracy)에서 기존의 최첨단 네트워크를 초월하며, 정밀도(precision), 재현율(recall), F1-score, ROC 곡선 아래 면적(AUC)에서 모든 카테고리에서 우수한 성능을 나타냅니다. 또한, LIME(Local Interpretable Model-Agnostic Explanations)와 같은 신뢰할 수 있는 설명 기법을 사용하여 모델의 예측을 해석 가능하게 만들어 사용자에게 자신감을 부여합니다.



### ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy (https://arxiv.org/abs/2502.05450)
- **What's New**: 최근 비전-언어-행동(Vision-Language-Action, VLA) 모델의 발전이 실제 로봇 조작에 큰 가능성을 보여주고 있습니다. 그러나 감독 학습(supervised learning) 방법으로 이러한 모델을 미세 조정하는 데는 많은 어려움이 존재합니다. 본 연구에서는 이러한 문제를 해결하기 위해 ConRFT라는 강화 미세 조정(reinforced fine-tuning) 접근 방식을 제안합니다.

- **Technical Details**: ConRFT는 통합된 일관성 기반 훈련 목표를 가진 오프라인 및 온라인 미세 조정 단계로 구성됩니다. 오프라인 단계에서는 행동 복제(behavior cloning)와 Q-learning을 통합하여 적은 수의 데모에서 정책을 효과적으로 추출하고 가치 추정(value estimating)을 안정화합니다. 온라인 단계에서는 일관성 정책(consistency policy)을 통해 모델을 추가로 미세 조정하고, 인간의 개입(human interventions)을 확인하여 안전한 탐색(safe exploration) 및 높은 샘플 효율성을 보장합니다.

- **Performance Highlights**: 본 연구에서는 8가지 다양한 실제 조작 작업에서 제안된 접근 방식을 평가한 결과, 온라인 미세 조정 후 평균 96.3%의 성공률을 달성했습니다. 이는 감독 학습 방법보다 성공률이 144% 향상되고 에피소드 길이가 1.9배 짧은 성과입니다. 이러한 결과는 VLA 모델의 실제 로봇 어플리케이션을 향상시키기 위한 강화 학습의 통합 가능성을 강조합니다.



### Iterative Deepening Sampling for Large Language Models (https://arxiv.org/abs/2502.05449)
- **What's New**: 최근 OpenAI의 o1 모델의 출시로 인해 복잡한 추론 작업을 처리하는 데 탁월한 능력이 입증되었습니다. 이 논문에서는 모델이 단일 응답 내(intra-response)와 여러 응답 간(inter-response)에서 검색 기능을 발전시키는 것이 중요하다는 점에 주목합니다. 특히, 자기 평가와 자기 교정 기능을 향상시키기 위한 자기 반성 데이터 생성의 질을 개선하는 데 초점을 맞춥니다. 이를 위해 독창적인 Iterative Deepening Sampling(IDSampling) 알고리즘 프레임워크를 제안합니다.

- **Technical Details**: Iteration Deepening Sampling (ID-Sampling) 방법론은 샘플링 예산을 기하급수적으로 증가시키는 반복적 접근 방식을 사용하여 자기 반성 메커니즘을 각 확장 단계에 통합합니다. 이는 모델의 성능을 향상시키면서도 예산 낭비를 최소화할 수 있는 효율적인 방법입니다. 우리는 MATH-500 및 AIME-24와 같은 난이도 높은 벤치마크에서 ID-Sampling의 효과를 평가하였습니다. 또한, 각 반복에서 예산 증가 비율이 성공률과 추론 시간에 미치는 영향을 분석하는 절단 연구(ablation study)도 진행하였습니다.

- **Performance Highlights**: ID-Sampling을 사용한 실험 결과, 고난이도 작업에서 더 높은 성공률을 달성했습니다. 이는 모델의 성능을 향상시키기 위한 적응형 자기 반성 메커니즘의 가능성을 보여줍니다. 최종적으로 이 연구는 고품질의 자기 반성 데이터를 생성하여 차세대 LLM의 훈련을 향상시키는 데 기여할 수 있는 방법을 제시하고 있습니다.



### Unbiased Sliced Wasserstein Kernels for High-Quality Audio Captioning (https://arxiv.org/abs/2502.05435)
Comments:
          17 pages, 9 tables, 2 figures

- **What's New**: 본 논문에서는 오디오 캡셔닝(audio captioning) 분야에서 강조되는 시간이 포함된 유사성 측정을 도입하여 캡션 저하(caption degeneration) 문제를 해결하는 새로운 방법을 제시합니다. 특히, 시간 정보를 고려하기 위해 회전 위치 임베딩(rotary positional embedding)을 갖춘 편향 없는 슬라이스 워셔스타인 RBF 커널(USW-RBF)을 개발하여 서로 다른 모달리티 간의 유사성을 효과적으로 측정합니다. 또한, 스토캐스틱(stochastic) 디코딩 방법을 포함한 오디오 캡셔닝 프레임워크를 구현하여 생성 과정 중 캡션 저하 문제를 완화하고자 합니다.

- **Technical Details**: 제안된 USW-RBF 커널은 시간적 왜곡이 존재할 때 서로 다른 고차원 시퀀스 간의 측정에서 발생하는 문제를 해결하는 데 뛰어난 성능을 보입니다. 이 커널은 몬테 카를로 추정(Monte Carlo estimation)을 통해 비편향 추정을 제공하며, 이는 확률적 경량 최적화 알고리즘(stochastic gradient optimization algorithms)에 잘 적합합니다. 논문에서는 USW-RBF가 균형 잡힌 유사성 점수를 제공합니다. 이를 통해 문자 및 오디오 간의 특징과 시간 정보를 효과적으로 측정할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 ACUS 프레임워크는 오디오 캡셔닝과 관련하여 캡션의 길이, 어휘 다양성, 텍스트-오디오 자기 검색(self-retrieval) 정확도를 유의미하게 향상시킵니다. 이 연구는 AudioCaps 및 Clotho 데이터셋을 사용하여 고품질 오디오 캡션 생성의 가능성을 입증하며, 다양한 정량적 및 정성적 실험을 거쳐 실질적 개선 효과를 보여줍니다.



### APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding (https://arxiv.org/abs/2502.05431)
Comments:
          ICLR 2025

- **What's New**: 본 논문에서는 Context-augmented generation (CAG) 기법, 특히 retrieval-augmented generation (RAG) 및 in-context learning (ICL)을 위한 Adaptive Parallel Encoding (APE) 방법을 제안합니다. 기존의 parallel encoding 방식이 불러오는 성능 저하 문제를 해결하기 위해 three alignment steps를 통해 sequential encoding과의 분포 정렬을 모색합니다. 이를 통해 APE는 RAG 및 ICL에서 각각 98%와 93%의 sequential encoding 성능을 유지하면서도 parallel encoding에 비해 성능을 각각 3.6% 및 7.9% 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: Context-augmented generation (CAG) 기술은 사용자 쿼리에 대한 응답 생성 시 여러 문맥을 효율적으로 결합하는 방법입니다. 본 논문에서는 KV states을 독립적으로 pre-compute하고 캐시하는 parallel encoding 방식을 사용할 때 발생하는 성능 저하 문제를 다룹니다. Adaptive Parallel Encoding (APE)는 shared prefix, attention temperature, scaling factor를 활용하여 이 문제를 해결하고, 더 많은 문맥을 처리할 수 있는 능력을 제공하며, 128K 길이의 문맥에 대해서는 28배의 prefilling 시간 단축을 달성합니다.

- **Performance Highlights**: APE는 RAG와 ICL 작업에서 각각 98%와 93%의 sequential encoding 성능을 유지할 수 있으며, parallel encoding 방식보다 각각 3.6% 및 7.9% 더 나은 성능을 보입니다. 또한, APE는 수백 개의 문맥을 효율적으로 인코딩할 수 있어 많은 샷 상황에서도 sequential encoding과 동등하거나 이를 초과하는 성능을 발휘합니다. 마지막으로, APE는 긴 문맥 생성을 가속화하여 최종적으로 4.5배의 속도 향상을 달성할 수 있습니다.



### SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation (https://arxiv.org/abs/2502.05424)
Comments:
          Accepted by WWW2025 Main Track

- **What's New**: 본 논문에서는 텍스트가 없는 그래프를 위한 멀티 도메인 그래프 사전 학습 및 교차 도메인 적응을 위한 구조 정렬 프레임워크(SAMGPT)를 제안합니다. SAMGPT는 다양한 소스 도메인에서 나온 그래프들의 지식을 학습하여 보지 못한 타겟 도메인에 적응할 수 있도록 설계되었습니다. 특히, 구조 정보를 기반으로 하는 집계를 조화롭게 하기 위해 구조 토큰을 도입하고, 교차 도메인 적응을 위해 전체적 및 특정 프롬프트를 제공합니다.

- **Technical Details**: SAMGPT는 다중 도메인 그래프의 사전 학습 동안 구조적 변수를 조화롭게 만드는 것을 목표로 합니다. 각 도메인에는 구조 기반 집계를 수정할 수 있는 학습 가능한 구조 토큰이 제공됩니다. 교차 도메인 적응에서는 전체적 지식과 도메인별 특성을 동시에 활용하기 위해 두 가지 유형의 프롬프트, 즉 전체적 프롬프트와 특정 프롬프트를 활용합니다.

- **Performance Highlights**: 저자들은 7개의 공공 데이터셋에서 SAMGPT의 성능을 종합적으로 평가했으며, 기존의 최첨단 방법들과 비교하여 우수한 성능을 доказ했습니다. 이 모델은 메타 학습의 원리를 바탕으로 다양한 도메인에서 수집된 그래프를 통합하여 뛰어난 효율성을 입증했습니다.



### Show-o Turbo: Towards Accelerated Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2502.05415)
- **What's New**: 본 논문에서는 Show-o Turbo를 소개하여 이미지와 텍스트 생성을 모두 포함하는 통합 멀티모달 모델의 효율성을 높이고자 합니다. Show-o는 세밀한 노이즈 제거와 자동 회귀적 텍스트 디코딩 방식으로 작동하는데, 이 과정에서 발생하는 비효율성을 개선하기 위해 새로운 접근법을 제안합니다. 이를 통해 Show-o Turbo는 기존 모델 대비 빠른 생성을 가능하게 하고, 더 나은 성능을 갖춥니다.

- **Technical Details**: Show-o Turbo는 텍스트 토큰의 병렬 디코딩을 기반으로 이미지와 텍스트 생성의 통합적 관점을 제시합니다. 일관성 증류(Consistency Distillation, CD) 기술을 활용하여 다양한 샘플링 경로의 고유점을 결정하고, 이를 통해 모델의 훈련 수렴 속도를 개선합니다. 이러한 접근법을 통해 Show-o Turbo는 훈련 단계에서 서로 다른 세그먼트 수를 이용한 커리큘럼 학습 방식을 통합하여 성능을 높입니다.

- **Performance Highlights**: 실험 결과, Show-o Turbo는 4회의 샘플링 단계 동안 GenEval 점수 0.625를 기록하며, 이는 8회의 샘플링 단계와 클래스 없는 가이드(Classifier-Free Guidance, CFG)를 사용하는 기존 Show-o 모델보다 더 우수한 성능을 보였습니다. 또한 이미지에서 텍스트로의 변환 과정에서도 성능을 크게 저하시키지 않으면서 1.5배의 속도 향상을 달성하였습니다.



### Vision-in-the-loop Simulation for Deep Monocular Pose Estimation of UAV in Ocean Environmen (https://arxiv.org/abs/2502.05409)
Comments:
          8 pages, 15 figures, conference

- **What's New**: 이 논문은 해양 환경에서 운용되는 UAV의 깊이 모노큘러 포즈 추정을 위한 비전 인 더 루프(vision-in-the-loop) 시뮬레이션 환경을 제안합니다. 최근 트랜스포머 아키텍처를 채택한 심층 신경망이 UAV의 자세 추정에 성공했고, GPS 기반 접근법의 여러 한계를 극복했습니다. 연구선의 제한된 가용성과 높은 운영 비용으로 인해 실제 해양 환경에서 심층 포즈 추정 방안을 검증하는 데 어려움이 많습니다.

- **Technical Details**: 이 연구는 새로운 가우시안 스플래팅(Gaussian splatting) 기술을 활용하여 사진 현실적인 3D 가상 환경을 만들고, 이를 통해 UAV의 비전 기반 제어와 추정 알고리즘을 평가합니다. TNN-MO 아키텍처를 사용하여 단일 RGB 이미지에서 UAV의 6D 포즈를 정확하게 추정하며, 가상 환경에서 합성 데이터를 생성하여 모델 훈련과 검증에 활용합니다. 이 과정에서는 복잡한 조명 조건, 동적인 해양 파도 등 다양한 환경 요소를 고려하여 현실감을 높이고, 모델 일반화 성능을 개선합니다.

- **Performance Highlights**: TNN-MO 모델은 5,500개의 이미지에서 평균 위치 오차 0.204와 태세 오차 0.91°를 기록하며 다양한 조건에서도 강건함을 확인했습니다. 실제 데이터 검증에서는 UAV에 장착된 데이터 수집 시스템을 통해 RTK-GPS 측정값과 일치하는 결과를 도출했습니다. 이 연구는 해양 환경에서 UAV의 자율 비행 및 안전한 발사 및 회수를 위한 비전 기반 접근법의 가능성을 제시하며, 광범위한 실제 시나리오를 재현할 수 있는 능력을 보여줍니다.



### The Complexity of Learning Sparse Superposed Features with Feedback (https://arxiv.org/abs/2502.05407)
Comments:
          41 pages, 20 figures

- **What's New**: 이번 논문은 깊은 신경망(Deep Neural Networks)이 적절한 특징(feature)을 학습하는 방식과 이러한 특징을 복원하는 효율적인 방법에 대한 연구입니다. 연구팀은 대규모 언어 모델(LLM)과 같은 에이전트로부터 상대적 triplet 비교를 통해 모델이 학습한 특징을 추출할 수 있는지를 조사했습니다. 이는 LLM의 사전(dictionary) 또는 카바리안스(Covariance) 행렬의 구성 요소 등의 다양한 개념을 포함할 수 있습니다. 이론적 결과를 바탕으로 실제 실험을 통해 두 가지 응용 사례에 대한 검증을 수행했습니다.

- **Technical Details**: 딥러닝 모델은 샘플로부터 관련된 특징을 효과적으로 캡처하는 능력에 기반하여 다양한 작업에서 최첨단 성과를 달성하게 되는데, 이 모델들이 어떻게 이러한 특징을 학습하는지를 이해하는 것이 핵심적인 도전 과제입니다. 연구에서는 희소한 상황에서 특징 행렬(Feature Matrix)의 학습과 관련된 피드백 복잡도를 분석하였으며, 에이전트가 활성화를 구축할 수 있을 때 강력한 상한선을, 배급적(distributional) 정보로 제한될 때의 상한선을 제시했습니다. 이 과정은 희소 오토인코더(Sparse Autoencoders)를 통해 해석 가능한 특징을 검색할 수 있도록 하며, 모델 해석을 진전시키는 이점을 가집니다.

- **Performance Highlights**: 이 연구는 Recursive Feature Machine으로 학습된 모델에서 특징 복구와 대규모 언어 모델을 기반으로 한 희소 오토인코더에서의 사전 추출이라는 두 가지 응용에서 이론적 발견을 검증했습니다. 실험 결과, 주어진 환경에서 에이전트의 피드백을 통해 모델이 학습한 특징을 효과적으로 추출할 수 있음을 보여주었습니다. 이러한 연구는 향후 더욱 향상된 모델 해석 및 컴팩트한 모델 개발에 기여할 것으로 기대됩니다.



### Convolutional Deep Colorization for Image Compression: A Color Grid Based Approach (https://arxiv.org/abs/2502.05402)
- **What's New**: 이번 연구는 이미지 압축 최적화를 위한 색상 유지 접근 방식을 자동화하여, convolutional (컨볼루션) 색상화 네트워크 아키텍처를 최적화하는 데 중점을 두고 있습니다. 연구진은 이미지 컬러화 알고리즘을 사용하여 저장해야 할 색상 데이터의 양을 줄이면서도 이미지의 색상을 충실하게 복원하는 것을 목표로 하고 있습니다. 결과적으로 좋은 이미지 압축 비율을 달성하면서도 높은 CSIM 값을 기록했습니다.

- **Technical Details**: 연구는 이미지 데이터 셋에서 색상의 대부분을 제거하고 특정 색상 픽셀만을 유지하는 방법을 채택했습니다. 모델 학습 시 mean squared error (MSE)를 손실 함수로 사용하였고, ADAM 최적화를 통해 30 에폭(epoch) 동안 모델을 훈련시켰습니다. 그러면서 색상 정보가 얼마나 유지되느냐에 따라 성능을 평가하여 최적의 n 값을 찾아내는 방식을 적용했습니다.

- **Performance Highlights**: PSNR과 CSIM 두 가지 지표를 사용하여 이미지 색상화 성능을 평가하며, 각 n 값에 대해 색상화 질이 어떻게 변화하는지 분석했습니다. 연구 결과, n=20에서 최상의 압축 성능을 보였으며, 그 이후로는 색상화의 질이 감소하는 경향을 보였습니다. 이 연구는 이미지 컬러화의 자동화와 이로 인한 저장 용량 최적화의 가능성을 보여주는 중요한 발전이라고 할 수 있습니다.



### Coarse-to-Fine Structure-Aware Artistic Style Transfer (https://arxiv.org/abs/2502.05387)
Comments:
          21 pages, 17 figures

- **What's New**: 이 논문에서는 예술적 스타일 전송(artistic style transfer) 방법의 일반적인 문제를 해결하기 위해 새로운 접근법을 제안합니다. 기존 방법이 내용 이미지(content image)의 글로벌 구조(global structure)에 스타일 이미지(style image)의 질감과 색상(texture and color)만 전이하는 데 비해, 제안된 방법은 로컬 스타일 구조(local style structure)를 로컬 콘텐츠 구조(local content structure)에 융합(fuse)합니다.

- **Technical Details**: 제안된 방법에서는 먼저 저해상도(low resolution)에서 코스 네트워크(Coarse Network)를 사용하여 거친 스타일화된 특징(coarse stylized features)을 재구성합니다. 이 단계에서 스타일의 색상 분포(style color distribution)가 대략 전이되고 콘텐츠 구조(content structure)는 스타일 구조(style structure)와 결합됩니다. 이후, 이러한 재구성된 특징들과 콘텐츠 특징(content features)을 사용하여, 구조 인식(structure-aware)의 고해상도(high resolution) 스타일화된 이미지를 생성하기 위해 파인 네트워크(Fine Network)와 세 개의 구조 선택적 융합(structural selective fusion, SSF) 모듈을 적용합니다.

- **Performance Highlights**: 제안된 방법은 뛰어난 고품질 스타일화 결과를 생성하는 것으로 그 효과성이 입증되었습니다. 또한 여러 최신 스타일 전송 방법들과 비교하여, 스타일과 콘텐츠의 로컬 구조에 대한 일관성을 유지하면서 더 매력적인(visually appealing) 이미지를 제공합니다. 이러한 결과는 예술적 스타일 전송 분야에서의 새로운 가능성을 보여줍니다.



### Is attention all you need to solve the correlated electron problem? (https://arxiv.org/abs/2502.05383)
Comments:
          10+5 pages, comments welcome

- **What's New**: 이 연구에서는 self-attention를 이용한 뉴럴 네트워크 변분 몬테 카를로(NN-VMC) 방법을 소개합니다. 이 방법은 전자 상호작용 문제를 해결하기 위해 고안된 파라미터 수가 $N^2$로 증가하는 새로운 맥락을 제공합니다. 특히, 모아레 양자 재료에서 전자 상관관계를 모델링하며, 기존 Hartree-Fock 방법보다 더 정밀한 결과를 제시합니다.

- **Technical Details**: 연구는 self-attention 메커니즘을 활용해 다전자 시스템의 파동함수를 구축하는 과정을 설명합니다. Hamiltonian은 2차원 쿨롱 전자 기체 형태로 표현되며, 이는 모아레 잠재력과 쿨롱 상호작용의 상호작용을 반영합니다. 이 기술은 전자가 서로 어떻게 영향을 미치는지를 정량화하여 최적의 파동함수를 도출합니다.

- **Performance Highlights**: 본 연구의 경우, NN 파동함수가 상대적으로 작은 시스템에서 Hartree-Fock 방법과의 비교에서 더 낮은 에너지를 산출했습니다. 시스템 크기가 증가함에 따라 필요한 파라미터 수는 $N^2$로 스케일링되는 것을 발견하였으며, 이는 대규모 시스템에 대한 효율적인 시뮬레이션 가능성을 제시합니다. 마지막으로, self-attention NN 파동함수는 기존의 양자 화학 문제에서도 최신 정확도를 달성한 바 있습니다.



### fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving (https://arxiv.org/abs/2502.05370)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)에서 Mixture-of-Experts (MoE) 아키텍처의 메모리 비효율성을 해결하기 위해 새로운 시스템인 fMoE를 제안합니다. fMoE는 세밀한 전문가 선택 패턴과 입력 프롬프트의 의미적 힌트를 활용하여 전문가의 전이 및 캐싱 결정을 효율적으로 안내합니다. 이 시스템은 NVIDIA GPU 메모리에서 비활성 전문가를 CPU 메모리로 오프로드하여 낮은 지연 시간과 높은 메모리 효율성을 실현합니다.

- **Technical Details**: fMoE는 MoE 모델의 전문가 선택 행동을 추적하기 위해 '전문가 맵'(expert map)이라는 새로운 데이터 구조를 사용합니다. 이 데이터 구조는 게이트 네트워크가 출력한 확률 분포를 기록하며, 이를 기반으로 전문가 오프로드 결정을 내립니다. 또한, fMoE는 MoE 모델이 처리하는 개별 요청 프롬프트의 의미적 임베딩을 추적하여 오프로드를 안내하는 데 기여합니다.

- **Performance Highlights**: fMoE는 HuggingFace Transformers 위에 프로토타입으로 개발되어 여섯 개의 GPU 테스트베드에서 배포되었습니다. 실험 결과, open-source MoE 모델 및 실제 워크로드에서 fMoE는 기존의 최첨단 솔루션에 비해 47%의 지연 시간 감소와 36%의 전문가 히트율 개선을 달성했습니다. 이러한 성능 향상은 fMoE가 노후화된 오프로드 솔루션의 한계를 극복하고 더 나은 메모리 효율성을 확보한 데 기인합니다.



### Estimating Voltage Drop: Models, Features and Data Representation Towards a Neural Surroga (https://arxiv.org/abs/2502.05345)
- **What's New**: 이 연구는 현대의 맞춤형 집적회로(ASIC)에서 전압 강하(IR drop)를 정확하게 추정하는 데 있어 기계 학습(Machine Learning) 기법의 활용 가능성을 조사합니다. XGBoost, Convolutional Neural Network (CNN), Graph Neural Network (GNN)와 같은 ML 기술이 IR drop 추정에서의 계산 시간과 노력을 대폭 줄일 수 있음을 보여주었습니다. 특히 GNN의 성능이 가장 두드러져 전압 강하 추정에서 최소한의 예측 오류를 보였습니다.

- **Technical Details**: IR drop 분석을 위해 전력, 타이밍 및 지리적 정보를 특징으로 설정하여 ML 모델을 훈련합니다. 전력 기능은 스위칭, 누설 및 내부 전력을 포함한 총 전력 소비와 관계가 있으며, 타이밍 기능은 셀의 스위칭 시간을 고려합니다. 높은 성능의 ML 모델을 만들기 위해서는 입력 중 특정 특징을 신중하게 선택하는 것이 중요하며, XGBoost와 CNN과 같은 알고리즘이 실험에서 빠른 예측 시간을 달성했습니다.

- **Performance Highlights**: ML 알고리즘을 사용한 IR drop 예측의 성능은 기존 상용 도구에 비해 상당히 향상되었습니다. 실험 결과, GNN이 IR drop 예측에서 단순하지만 효과적인 접근법으로 자리잡아, 계산 시간을 줄이고 에너지 효율성을 높이는 데 기여했습니다. 이러한 최적화된 전원 회로 설계는 환경 영향 감소에도 기여할 수 있습니다.



### RAG-Verus: Repository-Level Program Verification with LLMs using Retrieval Augmented Generation (https://arxiv.org/abs/2502.05344)
- **What's New**: RagVerus는 복합 모듈 리포지토리의 증명을 자동화하기 위해 retrieval-augmented generation과 context-aware prompting을 통합한 새로운 프레임워크입니다. 기존의 함수 중심 메소드에서 간과된 모듈 간 의존성과 전체 맥락 문제를 해결하여 Repository 수준에서의 형식 검증을 가능하게 합니다. 이 프레임워크는 383개의 증명 완료 작업을 포함하는 RepoVBench 벤치마크에서 27%의 상대적 개선을 달성하며, 검증의 확장성과 샘플 효율성을 입증했습니다.

- **Technical Details**: RagVerus는 LLM 기반 프로그램 검증을 위한 retrieval-augmented framework로, 코드 속성 추출, 특정 작업 정보 검색 및 증명 생성의 세 단계로 나뉩니다. 레포지토리 전반에 걸쳐 코드 아티팩트를 전처리하여 증명 생성을 위한 메타데이터를 추출하고, 이를 통해 적절한 문맥을 검색하여 LLM이 검증 가능한 코드를 생성하도록 안내합니다. 증명 생성을 위해 AutoVerus와 같은 코드 생성 에이전트를 활용하며, LLM이 생성한 검증 주석이 모든 상황에서 사양이 유지되는지 Verus 컴파일러가 검증합니다.

- **Performance Highlights**: RagVerus는 VerusBench와 RepoVBench의 두 벤치마크에서 평가되었으며, 특히 VerusBench에서는 200% 이상의 증명 통과율 개선을 보였습니다. RepoVBench에서는 5% 더 많은 벤치마크 문제를 해결하며, 기존의 함수 수준 접근 방식에 비해 27%의 상대적 증가를 나타내었습니다. 이러한 성능은 RagVerus가 복합적인 의존성을 고려하여 컨텍스트에 맞는 증명을 생성하는 데 매우 효과적임을 보여줍니다.



### Multi-Class Segmentation of Aortic Branches and Zones in Computed Tomography Angiography: The AortaSeg24 Challeng (https://arxiv.org/abs/2502.05330)
- **What's New**: 이 연구는 AortaSeg24 MICCAI 챌린지를 통해 첫 번째로 100개의 CTA 볼륨이 주석된 데이터셋을 소개함으로써, 23개의 임상적으로 중요한 대동맥 가지 및 구역에 대한 다중 클래스 세분화 방법을 지원하고자 했습니다. 기존 방법들이 대동맥 세분화를 이분법적으로 처리하고 있어, 이는 대동맥의 복잡한 해부학적 구조를 제대로 반영하지 못했습니다. 이 챌린지는 세계 121개 팀의 참여를 유도하며, 최첨단 프레임워크를 활용한 혁신적 접근법들을 시험하게 했습니다.

- **Technical Details**: 대동맥은 여러 가지와 구역으로 나뉘며, 각 부분의 모양, 방향, 크기가 환자마다 다르게 나타납니다. 이러한 세분화 작업을 위해 AortaSeg24 챌린지를 조직했으며, 이에 대한 평가 지표로는 Dice Similarity Coefficient(DSC)와 Normalized Surface Distance(NSD)를 활용했습니다. 각 참가팀은 다양한 기법을 통해 개별 대동맥 가지 및 구역을 분류하는 방법을 연구했습니다.

- **Performance Highlights**: AortaSeg24 챌린지를 통해 제출된 알고리즘들 중 상위 5개 팀의 접근법이 주목받았습니다. 이들 알고리즘은 정확한 대동맥 세분화를 가능하게 하여, 의료 기기의 선택 및 적절한 스텐트 배치에 기여할 것으로 기대됩니다. 연구 데이터셋, 평가 코드 및 선도적인 방법론이 공개되어 향후 연구 및 임상 적용의 기반이 될 것입니다.



### Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic: An Approach Based on Error Tagging Model and Synthetic Data Generating Mod (https://arxiv.org/abs/2502.05312)
Comments:
          21 pages, 3 figures

- **What's New**:  최근 수년간 아랍어 문법 오류 수정(ArabGEC)의 수요가 증가함에 따라, 본 논문은 아랍어 문법 오류를 위한 대규모 합성 데이터셋을 생성하기 위해 오류 태깅 모델과 합성 데이터 생성 모델을 개발하고자 하였습니다. 특히, DeBERTav3 모델을 활용한 오류 태깅 모델은 올바른 문장을 26가지 오류 태그로 분류하여, 다양한 인위적 문법 오류를 생성하는 데 기여합니다. 또한, ARAT5 모델을 기반으로 한 합성 데이터 생성 모델을 통해 실제로 발생할 수 있는 오류 패턴을 반영한 문장을 생성하고 있습니다.

- **Technical Details**:  본 연구는 순서-순서(seq2seq) 방식과 연계하여 아랍어 문법 오류를 수정하기 위한 오류 태깅 및 합성 데이터 생성 기술을 도입합니다. 오류 태깅 모델은 다중 레이블 분류 작업으로, 각 문장은 유형별로 26개의 오류 태그로 분류됩니다. 이를 통해 생성된 오류 태그를 정확한 문장에 결합하여, AraT5 모델을 활용한 합성 데이터 생성 모델이 문법적으로 일관된 잘못된 문장을 생성하게 됩니다.

- **Performance Highlights**:  오류 태깅 모델은 QALB-14 및 QALB-15 테스트 세트에서 각각 94.42%의 F1 스코어를 달성하여 오류 태그 식별에 있어 가장 우수한 성능을 기록했습니다. 또한, 문법 오류 수정에 대한 합성 데이터 학습 결과로서 QALB-14 테스트 세트에서 새로운 최첨단 결과인 79.36%의 F1 점수를 기록했습니다. 최종적으로 30,219,310개의 합성 문장 쌍이 생성되어, 아랍어 문법 오류 수정 시스템의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Oracular Programming: A Modular Foundation for Building LLM-Enabled Softwar (https://arxiv.org/abs/2502.05310)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 새로운 프로그래밍 패러다임인 oracular programming을 제안합니다. 이는 도메인 전문가가 고수준 문제 해결 전략을 결정할 수 있도록 하며, 선택 지점을 프로그래밍적으로 남겨두고 그 해답은 실행 시 제공된 예제를 통해 LLM이 결정하도록 합니다. 이는 기존의 복잡한 파이프라인을 개선하면서도 효율성을 높일 수 있도록 설계되었습니다.

- **Technical Details**: Oracular programming은 세 가지 주요 구성 요소로 이루어져 있습니다: 비결정론적 프로그램으로서의 전략(strategy), LLM 오라클을 사용하여 탐색 트리를 탐색하는 방식을 지정하는 정책(policy), 그리고 다양한 문제 인스턴스에서의 성공적인 및 실패한 탐색 시나리오를 설명하는 시연(demonstrations)입니다. 각 요소는 전용 프로그래밍 언어로 표현되며, 독립적으로 향상되거나 교체될 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 이 접근 방식은 LLM을 활용하여 문제를 해결하는 데 있어 신뢰성과 모듈성의 한계를 극복할 잠재력을 지니고 있으며, 이를 통해 복잡한 문제를 처리하는 과정의 효율성을 크게 향상시킬 수 있을 것으로 기대됩니다. 또한, 각 구성 요소의 독립적인 개선이 가능하므로, 다양한 문제 해결 전략을 적용할 수 있는 유연성을 제공합니다.



### Parameter Symmetry Breaking and Restoration Determines the Hierarchical Learning in AI Systems (https://arxiv.org/abs/2502.05300)
- **What's New**: 이 논문은 현대 대규모 AI 시스템에서 학습의 동적 구조가 계층적이며, 물리학 시스템에서 관찰되는 위상 전이(phase transition)와 유사한 급격한 변화로 특징지어질 수 있음을 제안합니다. 기존 이론은 특정 사례에 초점을 맞추어 단편적이며, 본 논문은 매개변수 대칭 파괴(parameter symmetry breaking)와 회복(restoration)가 이러한 행동의 통합 메커니즘으로 작용한다고 주장합니다.

- **Technical Details**: 저자들은 신경망의 세 가지 계층: 학습 동적 구조, 모델 복잡성, 표현 형성의 연결을 강조합니다. 모든 현상은 매개변수 대칭의 관점에서 이해될 수 있으며, 이는 다양한 시스템 세부 정보에 독립적입니다. 매개변수 대칭은 신경망의 각 계층이 보이는 특정 행동과 구조적 특성을 설명하는 중요한 원리로 제안됩니다.

- **Performance Highlights**: 이들의 제안은 고급 AI 시스템을 설계할 때 매개변수 대칭을 명시적으로 식별하고 이를 활용할 수 있는 잠재력을 보여줍니다. 이는 학습 동적 구조와 신경망의 계층적 구조에서 의도적인 대칭을 도입하여 모델 성능을 개선하는 새로운 접근 방식을 제공하는 것으로, 향후 AI 연구에 큰 기여를 할 가능성이 있습니다.



### Drone Detection and Tracking with YOLO and a Rule-based Method (https://arxiv.org/abs/2502.05292)
- **What's New**: 이 논문은 드론 감지 시스템을 위한 새로운 데이터셋을 확장하고 YOLOv7 모델을 활용하여 드론 탐지 성능을 향상시키는 방법을 제안합니다. 기존의 적외선 이미지 데이터셋에 추가로 컬러 이미지와 해양 비디오를 기반으로 하는 새로운 데이터를 통합하여 다양한 환경에서 드론의 탐지 능력을 높이는 것을 목표로 합니다. 이러한 접근 방식은 드론 안전 및 프라이버시 보호를 위한 규제가 필요한 현대 사회에서 중요한 해법이 될 수 있습니다.

- **Technical Details**: 이 연구에서는 드론을 탐지하기 위해 YOLOv7 기반의 딥 러닝 모델을 사용하며, 데이터셋은 각각의 카메라에서 수집된 비디오와 이미지를 포함합니다. GStreamer를 활용하여 비디오 스트림을 전송하고 YOLO 모델을 Docker 컨테이너 내에서 실행하여 드론 감지 알고리즘을 적용하는 시스템 구조를 설명합니다. 데이터셋의 주 성분 중 일부는 주석이 달린 비디오 프레임에서 추출된 컬러 이미지로 구성되며, 이로 인해 모델의 정확도를 높일 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 제공된 데이터셋의 총 이미지는 43,080개이며, 각기 다른 환경 조건에서 드론을 탐지하는 성능을 실험할 수 있는 플랫폼을 마련합니다. 연구에서는 뛰어난 탐지 성능을 보장하기 위해 YOLOv7 모델과 다양한 모듈의 조합을 비교하여 그 의미를 분석합니다. 최종적으로, 감지 성능 및 추적 결과를 개선하기 위해 간단한 앙상블 방법이 소개됩니다.



### Homeomorphism Prior for False Positive and Negative Problem in Medical Image Dense Contrastive Representation Learning (https://arxiv.org/abs/2502.05282)
Comments:
          Accepted by T-PAMI 2025

- **What's New**: 의료 이미지를 위한 Dense Contrastive Representation Learning (DCRL)에서 발생하는 대규모 False Positive (FP) 및 False Negative (FN) 문제를 해결하기 위해 GEoMetric vIsual deNse sImilarity (GEMINI) 학습을 제안합니다. GEMINI는 DCRL에 homeomorphism 선행 지식을 포함시켜 의료 이미지 간의 신뢰할 수 있는 대응 관계 발견을 가능하게 합니다. 이를 통해 의료 이미지를 위한 밀집 표현 학습의 효과성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: GEMINI 학습은 두 가지 주요 요소로 구성됩니다. 첫째, Deformable Homeomorphism Learning (DHL)은 두 이미지를 정렬하기 위해 변형 가능한 매핑(DVF)을 예측하도록 학습하며, 이를 통해 픽셀 대비에서 신뢰할 수 있는 양성 쌍을 얻을 수 있습니다. 둘째, Geometric Semantic Similarity (GSS)는 정렬 정도를 측정하는 데 있어 의미적 유사성을 결합하여 정확한 정렬을 통한 양성 쌍의 신뢰할 수 있는 학습을 촉진합니다.

- **Performance Highlights**: GEMINI 학습은 7개의 데이터 세트에서 실험을 진행하며 기존의 방법을 초월하는 유망한 결과를 보였습니다. 이 연구에서 제안하는 접근법은 실제 의료 이미지 밀집 예측 작업에서 데이터를 더 효과적으로 활용할 수 있게 하여, 의료 이미지의 비용과 레이블 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### Quantum automated learning with provable and explainable trainability (https://arxiv.org/abs/2502.05264)
Comments:
          21 pages, 7 figures

- **What's New**: 이 논문에서는 기존의 양자 기계 학습(quantum machine learning) 모델의 한계를 극복하기 위해 양자 자동 학습(quantum automated learning)을 제안합니다. 기존 방식은 모델 파라미터의 기울기(gradients)에 의존하는 하이브리드 접근 방식을 사용하지만, 이는 전역 최소(global minima)로의 수렴을 증명할 수 없고, 양자 모델이 확장됨에 따라 실현 불가능해질 수 있습니다.

- **Technical Details**: 양자 자동 학습에서는 변동 매개변수(variational parameter)가 필요 없으며, 훈련 과정(training process)이 양자 상태 준비(quantum state preparation)로 전환됩니다. 훈련 데이터(training data)는 유니타리 연산(unitary operations)으로 인코딩되며, 랜덤한 초기 상태가 이 유니타리와 그 역(inverses)에 따라 반복적으로 진화하며, 높은 예측 정확도를 목표로 하는 변동이 중간에 삽입됩니다.

- **Performance Highlights**: 이 모델은 손실 함수(loss function)의 전역 최소에 해당하는 원하는 상태로 지수적으로 수렴(converges exponentially)함을 rigorously 증명합니다. 또한, 양자 자동 학습 패러다임은 일반화 능력(generalization ability)이 뛰어나며, 일반화 오류(generalization error)는 힐베르트 공간 차원의 로그 함수(logarithmic function)와 훈련 샘플 수의 비율로 상한이 설정됩니다. 실제 이미지 및 양자 데이터에 대한 광범위한 수치 시뮬레이션(numerical simulations)을 수행하여 모델의 효과성을 입증하였습니다.



### LLMs Can Teach Themselves to Better Predict the Futur (https://arxiv.org/abs/2502.05253)
- **What's New**: 이 연구는 인공지능 모델의 예측 능력을 향상시키는 Outcome-driven fine-tuning framework을 제안합니다. 기존의 인간 큐레이션 방식에 의존하지 않고, 모델의 Self-play를 활용하여 다양한 질문에 대한 두 가지 Reasoning trajectories와 Probabilistic forecasts를 생성합니다. 이러한 접근 방식은 지식 컷오프 이후의 질문을 다루면서도, 모델의 성능을 효과적으로 개선할 수 있습니다.

- **Technical Details**: 이 방법론은 생성된 Reasoning traces 쌍을 기반으로, 실제 결과와의 거리를 측정하여 Ranking합니다. 이후, Direct Preference Optimization(DPO)을 통해 모델을 Fine-tune하여 예측 정확도를 높입니다. 이러한 과정은 모델의 self-play로 인해 생성된 데이터의 다양성을 통해 이루어집니다.

- **Performance Highlights**: 테스트 세트에서, Phi-4 14B와 DeepSeek-R1 14B 모델의 예측 정확도가 기존 베이스 모델과 DPO fine-tuned 제어 모델에 비해 7%에서 10% 향상되었습니다. 이는 GPT-4o와 같은 대형 모델의 예측 능력과 동등한 수준으로 상승한 성과입니다.



### GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? (https://arxiv.org/abs/2502.05252)
- **What's New**: 이번 논문에서는 기존의 long-context LLMs가 복잡한 추론 문제를 해결하는데 한계를 가지고 있음을 밝히고, 이를 해결하기 위한 새로운 벤치마크인 GSM-Infinite를 제안합니다. 이 벤치마크는 무한 난이도와 맥락 길이를 가진 산술 문제를 생성하며, 세밀한 조작이 가능합니다. 기존의 평가 지표들이 가진 한계를 극복하고, LLM들의 추론 성능을 정량적으로 평가할 수 있는 기반을 마련하였습니다.

- **Technical Details**: GSM-Infinite 벤치마크는 computational graphs와 언어 의미론을 결합하여 모델링된 추론 문제를 기반으로 합니다. 이 구조에 따라 문제의 난이도를 세밀하게 조정하고, 필요 없는 노드를 추가하여 노이즈를 효과적으로 삽입합니다. 이를 통해 LLM의 처리 능력을 평가할 수 있는 새로운 프레임워크를 제공하며, 다양한 호환성을 갖추고 있습니다.

- **Performance Highlights**: GSM-Infinite를 사용한 전반적인 성능 평가에서, 최신 reasoning-optimized LLM들이 이전 SOTA 모델들에 비해 평균 AUC 점수가 거의 4배 향상된 것으로 나타났습니다. 그러나 노이즈가 포함된 환경에서는 다양한 성능 저하가 관찰되는 등, LLM의 성능이 문제 난이도와 맥락 길이에 따라 일관된 감소를 보였습니다. 이 연구는 현재 long-context LLMs의 근본적인 한계를 강조하고 있으며, 향후 발전 방향에 대한 통찰을 제공합니다.



### Evaluating Personality Traits in Large Language Models: Insights from Psychological Questionnaires (https://arxiv.org/abs/2502.05248)
Comments:
          Accepted for publication at TheWebConf 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 성격 특성을 분석하는 새로운 접근법을 제시하고 있습니다. 심리적 평가 도구인 Big Five Inventory와 같은 질문지를 통해 LLM의 성격 프로파일을 생성하고, 다양한 모델 간의 차이를 조사하였습니다. 연구 결과 LLM들이 고유한 성격 특성을 보임을 발견했으며, 이는 인간의 행동을 이해하는 데 기여할 수 있습니다.

- **Technical Details**: 연구는 LLM의 성격 특성을 평가하기 위해 다섯 개의 주요 성격 차원인 Openness, Conscientiousness, Extraversion, Agreeableness, 그리고 Neuroticism을 포함한 심리적 질문지를 사용하였습니다. 각 질문지는 서로 다른 구조로 재구성되어 훈련 데이터 오염을 방지하고 편향성을 최소화했습니다. 이러한 절차는 LLM의 응답의 일관성을 확보하기 위한 체계적인 방법론을 포함합니다.

- **Performance Highlights**: 연구 결과는 LLM들이 Agreeableness, Openness, Conscientiousness와 같은 성격 특성에서 높은 점수를 보임을 나타냅니다. 이는 협력적이고 창의적이며 조직적인 행동을 반영하고 있습니다. 여러 LLM 모델을 통해 사용된 차원의 분석은 각 성격 질문지의 지배와 변동성을 보여주어, LLM의 성격 특성을 체계적으로 이해하는 데 도움을 줍니다.



### SEER: Self-Explainability Enhancement of Large Language Models' Representations (https://arxiv.org/abs/2502.05242)
Comments:
          18 pages,5 figures,10 tables

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 설명 가능성을 향상시키기 위한 새로운 방법인 SEER를 제안합니다. SEER는 동일한 개념을 집계하고 서로 다른 개념을 분리하여 LLM의 숨겨진 표현을 명확히 설명할 수 있도록 합니다. 이로 인해 LLM의 출력과 동시에 신뢰할 수 있는 설명을 제공합니다.

- **Technical Details**: SEER는 LLM의 표현 공간에서 개념을 집계하고 분리하는 과정에서, 'black-box' 모듈 없이 자체 설명 기능을 강화합니다. 이 접근 방식은 LLM의 추론 로직을 이해하고 응용 시나리오에서의 신뢰도를 높이는 데 기여합니다. 또한, 최적 수송 이론(optimal transport theory)을 통해 SEER의 LLM의 일반화(generalization) 능력 향상에 대한 이론적 분석을 진행합니다.

- **Performance Highlights**: 실험 결과, SEER는 안전 위험 분류(safety risks classification) 및 해독 작업(detoxification tasks)과 같은 신뢰성과 관련된 작업에서 일관된 개선을 보여 주었습니다. 이러한 자기 설명(self-explained) LLM들은 설명 가능성과 성능에서 최근의 지속적인 개선을 달성하였습니다.



### Enhancing Knowledge Graph Construction: Evaluating with Emphasis on Hallucination, Omission, and Graph Similarity Metrics (https://arxiv.org/abs/2502.05239)
- **What's New**: 이번 논문은 비구조적 텍스트에서 지식 그래프(kknowledge graph)를 자동으로 구축하는 대형 언어 모델의 최근 발전을 다루고 있습니다. 연구자들은 이전 연구를 바탕으로 환각(hallucination) 및 생략(omission) 문제를 해결하기 위한 개선된 접근 방식을 제안합니다. 특히, 그래프 유사성(graoh similarity) 평가를 위한 BERTScore를 통합하여 95%의 현실적인 그래프 매칭(threshold)을 설정했습니다.

- **Technical Details**: 실험에서는 Mistral 모델을 중심으로 원래 버전과 파인 튜닝(fine-tuning)된 버전을 제로샷(zero-shot) 및 퓨샷(few-shot) 설정에서 비교했습니다. 또한 KELM-sub 훈련 데이터셋의 예시를 이용하여 실험을 확장하였습니다. 결과적으로 파인 튜닝된 모델이 지식 그래프 구축 정확도를 크게 향상시키면서, 환각과 생략 현상을 줄이는 것으로 나타났습니다.

- **Performance Highlights**: 하지만, 연구 결과에 따르면 파인 튜닝된 모델은 KELM-sub 데이터셋의 일반화(generalization) 작업에서 성능이 떨어지는 것으로 밝혀졌습니다. 이 연구는 텍스트 데이터를 기반으로 한 지식 그래프 구축의 최전선에서 나타나는 포괄적인 평가 메트릭스의 중요성을 강조합니다.



### PSM-SQL: Progressive Schema Learning with Multi-granularity Semantics for Text-to-SQL (https://arxiv.org/abs/2502.05237)
Comments:
          9 pages, 3 figures, submission in progress

- **What's New**: 이 논문은 자연어(NL) 질문을 실행 가능한 구조화된 쿼리 언어(SQL) 쿼리로 변환하는 Text-to-SQL 문제에 대한 진전을 제안합니다. 특히, 다중 밀도 의미론(multi-granularity semantics)을 활용한 점진적 스키마 링크(PSM-SQL) 프레임워크를 도입하여, 중복된 데이터베이스 스키마를 줄이면서 의미 학습을 향상시킵니다. 기존의 연구들은 스키마의 테이블 수준에서 한 번만 링크를 수행했으나, PSM-SQL은 여러 수준에서 스키마의 의미를 학습합니다.

- **Technical Details**: PSM-SQL은 다중 밀도 스키마 링크 모듈(MSL)을 통해 테이블, 컬럼 및 데이터베이스 수준에서 스키마 의미를 학습합니다. 컬럼 레벨에서는 triplet loss를 사용하여 임베딩을 학습하고, 데이터베이스 레벨에서는 LLM을 미세 조정하여 스키마 추론을 수행합니다. 테이블 레벨에서는 분류기(classifier)와 유사도 점수(similarity scores)를 사용하여 스키마 링크를 모델링하였으며, 체인 루프 전략을 통해 중복된 스키마의 수를 지속적으로 줄이는 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, PSM-SQL은 Spider 및 Bird 데이터셋에서 기존의 Text-to-SQL 방법들보다 1-3% 높은 성능을 보였습니다. 이를 통해 PSM-SQL이 중복된 데이터베이스 스키마를 효과적으로 줄이고, 의미 학습을 개선함을 인증합니다. 이러한 성과는 Text-to-SQL의 난이도를 지속적으로 낮추는 데 기여할 것으로 기대됩니다.



### Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidanc (https://arxiv.org/abs/2502.05236)
- **What's New**: Koel-TTS는 자동 음성 생성 모델의 통제 가능성을 향상시키기 위해 개발된 최신 Transformer TTS 모델을 제공합니다. 이 모델은 자동 음성 인식(ASR) 및 화자 검증(SV) 모델을 활용한 선호 정렬 기법을 통합하여 생성된 음성이 기대하는 조건 입력에 더 잘 부합하도록 만들어졌습니다. 또한, 무분별한 지도 없이도 합성을 개선할 수 있는 classifier-free guidance(CFG) 기법을 도입하여 음성 생성의 품질을 높였습니다.

- **Technical Details**: Koel-TTS는 저프레임레이트(21.5 FPS)로 작동하는 오디오 코덱을 사용하여 텍스트와 문맥 오디오를 직접 음향 토큰으로 매핑합니다. 우리의 접근법은 우선 ASR 모델과 SV 모델을 통한 정확한 평가를 통해 화자 유사성과 음성 인식의 정확성을 측정하고, 이를 바탕으로 생성된 결과를 정렬하는 보상 시스템을 구축합니다. 이후, preference alignment 알고리즘과 CFG 기법을 결합하여 LLM 기반 TTS 모델의 효율성을 크게 개선했습니다.

- **Performance Highlights**: Koel-TTS는 기존의 최첨단 모델에 비해 매우 작은 데이터셋에서 훈련되었지만 여전히 그 성능이 뛰어납니다. 페어와이즈 랭킹 및 보상 최적화 기술을 통해 화자 유사성, 가독성 및 자연스러운 음성을 위한 지표들이 현저히 향상되었습니다. 이 모델은 제로샷 TTS 성과에서 여러 인간 및 자동 평가 기준을 충족하는 성능을 달성했습니다.



### Optimizing Temperature for Language Models with Multi-Sample Inferenc (https://arxiv.org/abs/2502.05234)
Comments:
          20 pages. Code available at this https URL

- **What's New**: 이 논문은 다양한 LLMs(대규모 언어 모델)의 multi-sample aggregation strategies을 통해 (거의) 최적의 temperature를 자동으로 식별하는 방법을 제안합니다. 이는 기존의 labeled validation data에 의존하지 않고 수행될 수 있으며, LLM 성능 최적화에 있어 중요한 패러미터인 temperature의 역할을 체계적으로 분석합니다. 특히 entropy turning point(Entropy 전환점)라는 새로운 개념을 도입하여, 이 포인트가 LLM의 최적 온도를 자동으로 결정하는 데 유용할 수 있음을 보여줍니다.

- **Technical Details**: LLM은 입력 맥락과 이전에 생성된 토큰을 기반으로 다음 토큰의 조건부 확률 분포에서 autoregressively 표본을 생성합니다. 논문은 logits를 temperature hyperparameter T에 따라 조정하고 softmax 기능을 적용함으로써 확률 분포를 생성하는 방식을 설명합니다. 새로운 탐색적인 접근법으로 TURN을 제안하며, 이 방법은 temperature 최적화를 자동화하는 알고리즘 솔루션을 제공합니다.

- **Performance Highlights**: TURN 방법은 다수의 실험을 통해 수학 문제 해결, 코드 생성 등 다양한 작업에서 우수한 일반화 능력을 입증하였습니다. 고정 온도를 사용하는 기존 방법들과 비교할 때, TURN은 일관되게 성능을 개선하며 다양한 aggregation strategies(예: majority voting, best-of-N)에 대해 강력한 성능을 보입니다. 또한, Entropy 전환점 분석을 통해 온도의 역할을 해석할 수 있는 기회를 제공합니다.



### Aligner-Encoders: Self-Attention Transformers Can Be Self-Transducers (https://arxiv.org/abs/2502.05232)
- **What's New**: 본 논문에서는 최신 음성 인식 시스템의 문제를 해결하기 위한 새로운 모델, 즉 'Aligner-Encoder'를 제안합니다. 기존의 RNN-Transducer(RNN-T)와 Attention 기반 Encoder-Decoder(AED) 시스템의 복잡성을 줄이고, 효율성을 높이기 위해 인코더가 입력 정보의 정렬을 내부적으로 수행할 수 있도록 설계되었습니다. 이로 인해, 훈련 및 디코딩 과정에서 더 간결하고 효율적인 음성 인식이 가능해졌습니다.

- **Technical Details**: Aligner-Encoder는 프레임 단위의 교차 엔트로피 손실을 사용하여 교육됩니다. 기존의 RNN-T 모델에서 사용하는 동적 프로그래밍을 버리고, 모든 프레임을 순서대로 읽으며 출력하는 경량화된 디코더를 통해 성능을 유지합니다. 특히, 이 모델은 자가 주의(self-attention) 메커니즘을 활용하여, 인코더가 입력 정보의 정렬을 자동으로 수행할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과 Aligner-Encoder 모델은 최신 모델들과 거의 동등한 성능을 보이며, 특히 긴 형태의 음성 인식을 위한 특별한 추론 구성에서 우수한 성과를 발휘했습니다. 전체적인 추론 시간이 RNN-T보다 2배, AED보다 16배 더 빠른 것을 보여줍니다. 이러한 효율성 덕분에 Aligner-Encoder는 음성 인식 분야에서 새로운 가능성을 열어줍니다.



### Thin ring wing as a means of flow improvement upstream of a propeller (https://arxiv.org/abs/2502.05231)
Comments:
          7 pages, 7 figures, Propellers/Shafting 97 Symposium

- **What's New**: 이 논문에서는 프로펠러 주변의 비정상 유동을 감소시키기 위해 링 형태의 날개를 사용하는 새로운 방법을 제안합니다. 링 형태의 날개는 효과적인 수동 와류 생성기(passive vortex-generator)로서 프로펠러 블레이드에서 가장 중요한 부분의 흐름을 제어할 수 있습니다.

- **Technical Details**: 비정상 유동 중 얇은 링 형태의 날개의 비대칭(가장자리) 기하학을 선형 접근법(linear approach)으로 해결하였으며, 인가되는 수직 와류(longitudinal vortices) 강도를 유동의 불규칙성 및 링 날개의 기하학에 따라 평가하였습니다. 이러한 해법을 통해 이 장치의 성능을 이론적으로 분석하였습니다.

- **Performance Highlights**: 시험 결과, 이론 모델과의 좋은 일치를 보이는 수조 실험을 통해 이 장치의 효과가 확인되었습니다. 링 형태의 날개는 안정장치(stabilizer)의 구조에 통합될 경우 추가적인 장점을 제공하는 것으로 논의됩니다.



### DiffNMR2: NMR Guided Sampling Acquisition Through Diffusion Model Uncertainty (https://arxiv.org/abs/2502.05230)
Comments:
          11 pages, 10 figures

- **What's New**: 이번 연구에서는 고해상도 NMR 스펙트럼 획득 시간을 크게 단축할 수 있는 새로운 서브샘플링 전략을 제안합니다. 기존의 방법들과 달리, 본 방법은 단백질 NMR 데이터에 대해 훈련된 확산 모델을 기반으로 하여 점진적으로 재구성하는 방식입니다. 이를 통해, 복잡한 NMR 실험에서 60%의 시간을 절감하고, 재구성 정확성을 52.9% 향상시키며, 허위 피크( hallucinated peaks)를 55.6% 줄일 수 있었습니다.

- **Technical Details**: NMR(Nuclear Magnetic Resonance) 분광학은 복잡한 분자의 구조를 분석하는 데 매우 유용한 도구입니다. 전통적인 NMR 스펙트럼 획득은 긴 시간을 요구하므로, 비균일 샘플링(Non-uniform sampling, NUS) 기법을 통해 고품질 스펙트럼 복원을 가능하게 합니다. 그러나 NUS의 효율성을 높이기 위해서는 불완전한 데이터를 처리할 수 있는 견고한 재구성 알고리즘이 필요하며, 본 연구에서는 확산 모델의 불확실성을 활용한 적응형 샘플링 방법을 제안합니다.

- **Performance Highlights**: 제안한 방법은 기존의 최첨단 샘플링 기법 및 재구성 방법에 비해 훨씬 더 뛰어난 성능을 보였습니다. 총 시간 절감 효과와 함께, 허위 피크 비율을 줄이고 스펙트럼 재구성의 정확성을 획기적으로 향상시킵니다. 이로 인해 약물 발견과 재료 과학 등 다양한 분야에서 고해상도 스펙트럼 분석의 필요성을 충족할 수 있는 가능성이 높아졌습니다.



### Multi-Objective Mobile Damped Wave Algorithm (MOMDWA): A Novel Approach For Quantum System Contro (https://arxiv.org/abs/2502.05228)
- **What's New**: 이번 연구에서는 복잡한 양자 제어 문제를 해결하기 위해 다목적 최적화 알고리즘인 Multi-Objective Mobile Damped Wave Algorithm (MOMDWA)를 소개합니다. 기존의 Mobile Damped Wave Algorithm (MDWA)의 기능을 확장하여 여러 개의 목표를 포함시킴으로써 더 포괄적인 최적화 과정을 가능하게 합니다.

- **Technical Details**: MOMDWA는 제어 정확도(control fidelity), 에너지 소비(energy consumption), 및 제어 부드러움(control smoothness) 간의 균형을 최적화하는 데 중점을 두고 세 가지 양자 제어 시나리오에 적용되었습니다. 알고리즘은 다양한 목표를 고려하여 최적화를 수행하여 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: MOMDWA는 양자 제어의 효율성과 강인성을 상당히 향상시키며, 높은 정확도를 유지하면서 에너지 사용을 최소화하고 부드러운 제어 펄스를 보장합니다. 이 발전은 양자 컴퓨팅 및 정밀한 다목적 제어가 필요한 기타 분야에 귀중한 도구를 제공합니다.



### Robotouille: An Asynchronous Planning Benchmark for LLM Agents (https://arxiv.org/abs/2502.05227)
Comments:
          11 pages (not including references or appendix); 41 figures (7 main paper, 34 appendix); (v1) preprint

- **What's New**: 이 논문은 대규모 언어 모델 (Large Language Model, LLM) 에이전트의 비동기 계획 능력을 평가하기 위한 새로운 벤치마크 환경인 Robotouille을 소개하고 있습니다. 기존의 벤치마크는 주로 단기 과제에 집중되어 있었으나, 이 연구는 복잡한 장기 비동기 시나리오를 처리하는 능력을 측정하고자 합니다. 이를 통해 LLM 에이전트가 겹치는 작업과 중단을 관리하는 데 필요한 도전 과제를 드러냅니다.

- **Technical Details**: Robotouille 벤치마크는 동기와 비동기 데이터셋을 제공하여, 에이전트가 점점 복잡해지는 계획 과제를 해결할 수 있도록 설계되었습니다. 연구에서는 ReAct (gpt4-o) 모델을 평가했으며, 동기 작업에서는 47%, 비동기 작업에서는 11%의 성과를 보였습니다. 결과적으로 LLM 에이전트가 장기 피드백을 더 잘 통합하고 작업 실행 시 자신의 추론을 자기감사(self-audit)하는 것이 필요하다는 점을 강조합니다.

- **Performance Highlights**: ReAct 모델은 동기 작업에서 비교적 높은 성과를 보였지만, 비동기 작업에서는 낮은 성과를 기록하여 개선의 여지가 많음을 나타냅니다. 이 연구는 비동기 계획 능력을 향상시키기 위한 추가 연구의 필요성을 강조하며, LLM 에이전트가 다양한 작업 상황을 잘 처리할 수 있도록 해야 한다고 결론을 내립니다.



### BitAbuse: A Dataset of Visually Perturbed Texts for Defending Phishing Attacks (https://arxiv.org/abs/2502.05225)
Comments:
          18 pages, To appear in the Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics 2025

- **What's New**: 이번 연구에서는 실제 피싱 사례를 포함한 새로운 데이터셋인 BitAbuse를 제안합니다. 기존의 합성 데이터셋이 아니라 실제 사례로 구성된 325,580개의 시각적으로 왜곡된(VP) 텍스트를 포함하고 있습니다. 이 데이터셋은 피싱 공격 방어 시스템 개발에 있어 필수적인 연료로 가치를 강조합니다.

- **Technical Details**: BitAbuse 데이터셋은 Bitcoin Abuse 웹사이트에서 수집된 262,258개의 피싱 관련 이메일을 바탕으로 구성되었습니다. 이 이메일들은 2017년 5월 16일부터 2022년 1월 15일까지 수집되었으며, 이를 통해 26,591개의 VP 문장과 298,989개의 비VP 문장을 포함하는 원시 코퍼스를 만들었습니다. BERT 모델을 활용하여 영어 텍스트를 자동 분류하여 비영어 이메일을 제거했습니다.

- **Performance Highlights**: BitAbuse 데이터셋을 기반으로 훈련된 언어 모델은 기존 방법에 비해 약 96%의 정확도로 우수한 성능을 보였습니다. 이 연구는 실제 VP 텍스트와 합성 텍스트 간의 상당한 격차를 보여주며, 더 신뢰할 수 있는 사전 훈련 모델 구축의 필요성을 강조합니다.



### A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations (https://arxiv.org/abs/2502.05224)
- **What's New**: 최근 몇 년 동안 대규모 언어 모델(LLM)의 발전은 인간 언어 텍스트의 이해와 생성 능력의 비약적인 향상을 가져왔습니다. 이러한 LLM은 의학, 금융, 교육 등 다양한 산업에서의 활용 증가와 함께 보안 문제도 함께 대두되고 있습니다. 이 논문은 머신러닝 공격 분류에 대한 일반적인 분류 체계를 적용하여 훈련 시간의 화이트박스 백도어 공격에 대한 체계적인 분류를 제공합니다.

- **Technical Details**: 논문에서는 백도어 공격(backdoor attacks) 방법과 이에 대한 방어(defense) 방법을 체계적으로 다룹니다. 특히 훈련 시간 동안 발생할 수 있는 백도어 공격은 LLM의 방어 기법 발전과 함께 지속적으로 발전하고 있음을 강조합니다. 이러한 공격 방법과 방어 방법에 대한 광범위한 정리를 통해, 연구자들이 더욱 강력한 방어 체계를 구축할 수 있는 지침을 제공하고자 합니다.

- **Performance Highlights**: 기존 연구 결과를 광범위하게 정리함으로써, 이 논문은 다양한 공격 시나리오를 탐구하고, 더욱 강력한 방어 메커니즘을 개발할 가능성을 열어줍니다. 따라서 향후 연구에 있어 LLM의 안전성과 신뢰성을 향상시키는 데 기여할 수 있는 중요한 자료로 작용할 것입니다.



### KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs (https://arxiv.org/abs/2502.05223)
- **What's New**: 이 논문은 Knowledge-Distilled Attacker (KDA)라는 새로운 오픈소스 모델을 제안하여, 기존의 기법에서 발생하는 여러 문제를 해결하고 있습니다. 기존의 Jailbreak 공격 방식은 정교한 시스템 프롬프트에 의존하며 많은 쿼리가 필요해 실용성이 떨어졌습니다. KDA는 여러 SOTA 공격자들의 지식을 정제하여 단일 모델로 통합함으로써 이러한 제약을 극복합니다.

- **Technical Details**: KDA는 AutoDAN, PAIR 및 GPTFuzzer와 같은 세 가지 SOTA 공격자로부터 전략을 학습하여, 고유한 공격 프롬프트를 효과적으로 생성합니다. KDA는 Supervised fine-tuning을 통해 Vicuna-13B라는 아키텍처로 구현되어 있으며, 프롬프트 생성을 자동화하여 다양성과 효율성을 향상시킵니다. 이렇게 생긴 공격 프롬프트는 인체의 가치와 안전성을 고려하며, 다수의 상용 LLM과 오픈소스 모델 모두에 적용 가능합니다.

- **Performance Highlights**: KDA는 다양한 LLM에 대해 높은 공격 성공률(ASR)을 기록하며, 예를 들어 Llama-2-7B-Chat에서 88.5%, Llama-2-13B-Chat에서 83.5%의 ASR을 달성했습니다. KDA는 또한 Harmbench 및 Harmful-Behavior 데이터셋과 같은 불특정 데이터에 대해서도 높은 ASR을 유지하며 강력한 일반화 능력을 showcase하고 있습니다. 최종적으로 KDA는 형식 인셈블링을 통해 공격의 다양성과 성공률을 극대화하여 기존 모델보다 우수한 성능을 보여줍니다.



### Blackout DIFUSCO (https://arxiv.org/abs/2502.05221)
Comments:
          12 pages

- **What's New**: 이번 연구는 Blackout Diffusion을 DIFUSCO 프레임워크에 통합하여 조합 최적화(combinatorial optimization), 특히 외판원 문제(Traveling Salesman Problem, TSP)를 목표로 합니다. 전통적인 이산 시간 모델에서 지속적인 시간 모델로의 확장을 통해, Blackout Diffusion의 고유한 속성을 활용하고자 합니다. 지속적인 시간 모델링은 더 부드러운 전환(smoother transitions)과 정교한 제어(refined control)를 도입해 솔루션 품질을 향상시킬 것으로 가정하고 있습니다.

- **Technical Details**: 연구진은 확산 프로세스(diffusion process)를 향상시키기 위해 세 가지 주요 개선점을 제안합니다. 첫째, 이산 시간 기반 모델에서 지속적인 시간 프레임워크로의 전환을 통해 보다 정교하고 유연한 공식을 제공합니다. 둘째, 관찰 시간 스케줄링을 정제하여 확산 과정 전반에 걸쳐 매끄럽고 선형적인 변환을 보장하며, 마지막으로 모델에 도전적인 영역에서 더 빈번한 시간 간격을 도입하여 정확성과 안정성을 높입니다.

- **Performance Highlights**: 실험 결과는 기준 성능을 초과하지는 못했지만, 간단함과 복잡성의 균형을 맞추는 데 있어 이 방법들의 효과성을 입증했습니다. 이 연구는 Blackout Diffusion을 조합 최적화에 적용한 최초의 사례로, 이후 연관 연구의 기반을 마련하는 데 기여할 것입니다.



### Aero-LLM: A Distributed Framework for Secure UAV Communication and Intelligent Decision-Making (https://arxiv.org/abs/2502.05220)
Comments:
          This manuscript was accepted by the 1st International Workshop on Integrated Sensing, Communication, and Computing in Internet of Things (IoT) Systems at the The 33rd International Conference on Computer Communications and Networks (ICCCN 2024)

- **What's New**: 이 논문은 무인 항공기(UAV) 운영의 보안성과 효율성을 높이기 위한 새로운 프레임워크인 Aero-LLM을 소개합니다. Aero-LLM은 다양한 작업을 위해 여러 전문화된 대형 언어 모델(LLMs)을 통합하여, 각 모델이 특정 작업을 수행할 수 있도록 설계되었습니다. 이를 통해 성능 저하를 줄이고 보안 능력을 향상시킬 수 있습니다.

- **Technical Details**: Aero-LLM은 여러 영역에 배치된 LLM을 활용하여 우수한 사전 훈련된 LLM(예: Llama, Gemini 등)의 성능을 초월하는 것이 아닌, 특정 작업과 데이터에 집중한 LLM으로 미세 조정하여 사용합니다. 이를 위해 감독 세밀 조정(SFT)과 인간 피드백을 통한 강화 학습(RLHF) 기법을 통해 훈련하며, 제한된 메모리 사용량으로도 높은 정확도와 정밀도를 달성합니다.

- **Performance Highlights**: Aero-LLM의 평가 결과, 높은 정확성, 정밀도, 재현율, F1 점수와 낮은 오류율을 기록하며, 네트워크 공격 및 데이터 변조로부터 강력한 방어력을 제공하는 탁월한 성능을 입증했습니다. 이 시스템은 UAV 운영의 보안 표준을 새롭게 설정하며, 실시간 데이터 통신을 지원하는 신뢰할 수 있는 인프라를 위한 필수적인 요소입니다.



### Enabling External Scrutiny of AI Systems with Privacy-Enhancing Technologies (https://arxiv.org/abs/2502.05219)
- **What's New**: 이번 논문은 비영리 기관 OpenMined가 개발한 기술 인프라가 민감한 정보를 침해하지 않으면서 AI 시스템에 대한 외부 검토를 어떻게 가능하게 하는지를 설명합니다. AI 시스템에 대한 독립적인 외부 검토는 투명성을 강화하여 AI 거버넌스에 필수적인 요소로 자리 잡아야 합니다. 저자들은 프라이버시 보호 기술(PETs)을 활용하여 AI의 감사를 지원하는 다양한 설정을 통합한 기술적 기반을 제시합니다.

- **Technical Details**: OpenMined의 기술 인프라는 보안 인클레이브(secure enclaves), 안전한 다자간 계산(secure multi-party computation), 제로 지식 증명(zero-knowledge proofs) 등 여러 가지 프라이버시 보호 기술을 결합하여 구성되었습니다. 또한, 오픈 소스 소프트웨어 라이브러리인 PySyft를 통해 연구자들에게 모델 소유자와 안전하게 질문을 주고받을 수 있는 구조를 제공합니다. 이 인프라는 연구자들이 민감 정보에 접근하지 않고도 필요한 정보를 얻을 수 있도록 해줍니다.

- **Performance Highlights**: Christchurch Call 및 UK AI Safety Institute와 협력하여 OpenMined는 AI 시스템을 효과적으로 감사할 수 있는 실제 적용 사례를 시연했습니다. 첫 번째 사례에서는 Dailymotion과 협력하여 알고리즘의 사회적 영향을 분석하는 데 성공하였고, 두 번째 사례에서는 Anthropic과의 협업을 통해 민감 정보를 보호하면서 AI 모델의 안전성을 평가했습니다. 이러한 사례들은 OpenMined의 기술 기반이 외부 검토를 지원하는 데 어떻게 기여할 수 있는지를 명확히 보여줍니다.



### FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual Contrastive Learning for Stock Returns Prediction (https://arxiv.org/abs/2502.05218)
- **What's New**: 본 논문은 하이퍼그래프(hypergraph) 기반의 요인 모델인 FactorGCL을 제안합니다. 이 모델은 기존의 전문가가 설계한 요인들 외에 데이터 기반의 숨겨진 요인(hidden factors)을 활용하여 주식 수익률을 예측하는 데 있어 효과성을 증대시킵니다. 기존 선형 모델의 한계를 극복하기 위해 비선형 관계를 포착하는 데 도움을 주며, 시간적 잔차 대조 학습(temporal residual contrastive learning)을 통해 모델의 학습 과정을 보조합니다.

- **Technical Details**: FactorGCL은 비선형 관계를 포착하기 위해 하이퍼그래프 구조를 활용하며, 주식 수익률을 세 가지 구성 요소: prior beta, hidden beta, individual alpha로 분해합니다. 이는 이전 구성 요소의 영향을 제거한 후 잔여(residual) 정보를 활용해 숨겨진 요인을 추출하는 방식으로 작동합니다. 또한, 시간적 잔차 대조 학습 방법을 통해 개별 주식의 잔여 정보를 다양한 시간대에서 비교하여 효과적이고 포괄적인 숨겨진 요인을 추출하는 데 도움을 줍니다.

- **Performance Highlights**: 실제 주식 시장 데이터에 대한 광범위한 실험 결과, FactorGCL은 기존의 최첨단 방법들을 능가할 뿐만 아니라 효과적인 숨겨진 요인을 발굴하는 데에 성공했습니다. 이러한 성과는 기존 모델들이 충분히 포착하지 못했던 복잡한 비선형 관계를 반영함으로써 이루어진 것입니다. 이 연구는 주식 수익률 예측의 새로운 전환점을 제공하며, 하이퍼그래프 및 대조 학습(training) 방법을 결합한 혁신적인 접근 방식을 보여줍니다.



### Watermarking across Modalities for Content Tracing and Generative AI (https://arxiv.org/abs/2502.05215)
Comments:
          PhD thesis - webpage available at this https URL

- **What's New**: 이번 논문에서는 디지털 콘텐츠에 물리적인 특징이 아닌 디지털 워터마킹(digital watermarking) 기법을 활용하여 AI 생성 콘텐츠의 모니터링, 콘텐츠 조정 및 추적할 수 있는 새로운 방법을 제시합니다. 특히, 이미지, 오디오 및 텍스트를 위한 새로운 워터마킹 기법들이 개발되어 AI 생성 콘텐츠의 품질을 향상시키는 기술들이 소개됩니다. 이러한 기법들은 인간에게는 인식되지 않으면서 특정 알고리즘에 의해 견고하게 검출될 수 있도록 설계되었습니다.

- **Technical Details**: 논문은 또한 잠재 생성 모델(latent generative models)을 활용하여 생성된 콘텐츠에 워터마크를 임베드하는 방법을 다룹니다. 이 기술은 스피치에서 워터마크가 있는 섹션을 식별하고, 대규모 언어 모델(large language models)에서 저 false positive(허위 양성)율을 보장하는 테스트로 워터마킹 품질을 개선하는 방법을 포함합니다. 이러한 방법들은 AI 모델의 사용을 모니터링하기 위한 새로운 경로를 제시하며, 기존의 워터마킹 기법이 가진 한계와 도전 과제도 함께 논의됩니다.

- **Performance Highlights**: 워터마킹 기술은 디지털 파일의 원본성을 확인하고, 불법 복제를 방지하며, 콘텐츠의 출처를 추적하는 데 중요한 역할을 합니다. 예를 들어, 헐리우드 스튜디오는 DVD 복제를 방지하기 위해 비디오 워터마킹을 사용하여 유출된 콘텐츠의 출처를 추적할 수 있습니다. 그림자 콘텐츠가 매우 빠르게 퍼지는 현재, 워터마킹은 더 이상 눈에 띄지 않지만 콘텐츠의 신뢰성을 확보하기 위해 필수적인 요소로 자리 잡고 있습니다.



### CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models (https://arxiv.org/abs/2502.05214)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 의료 영상에서의 오판 및 잘못된 임상 소견 식별에 초점을 맞춘 새로운 블랙박스 적대적 공격 기법인 CoRPA(Concept-based Report Perturbation Attack)를 제안합니다. CoRPA는 임상 개념을 활용하여 임상적으로 중요한 영상 및 보고서의 적대적 사례를 생성함으로써, 진단 오류를 반영한 현실적인 상황을 모사합니다. 이를 통해 기존의 일반적인 적대적 공격 기법이 의료 AI에 적합하지 않다는 점을 해결하고자 합니다.

- **Technical Details**: CoRPA는 의료 이미지와 관련된 임상 개념에 기반하여 이미지-보고서 쌍을 위한 개념 벡터를 생성하고, 이 벡터를 의도적으로 변형하여 잘못 식별되거나 누락된 임상 특징을 모사합니다. 이 과정에서 수정된 보고서를 기반으로 텍스트-투-이미지(generative model)를 통해 적대적 의료 이미지를 생성합니다. MIMIC-CXR-JPG 데이터셋을 통해 이러한 기술을 시연하며, 이를 통해 다양한 딥러닝 아키텍처의 경직성을 평가합니다.

- **Performance Highlights**: 실험 결과, CoRPA의 적대적 이미지를 적용했을 때, 기존의 일반적 공격 방식에 비해 특유의 높은 복잡성을 가진 모델들이 상당한 저항력을 잃는 것으로 나타났습니다. 이는 의료 AI 시스템에서 도메인 특화된 취약성 문제를 해결하는 것이 얼마나 중요한지 강조합니다. CoRPA는 다양한 의료 데이터셋에 손쉽게 확장될 수 있어, 임상 환경에서의 신뢰성 있는 AI 모형 개발에 기여할 수 있습니다.



### DERMARK: A Dynamic, Efficient and Robust Multi-bit Watermark for Large Language Models (https://arxiv.org/abs/2502.05213)
Comments:
          8 pages, 15 figures

- **What's New**: 이 논문에서는 최신 대형 언어 모델(LLMs)에서 멀티 비트 워터마킹 기술을 적용하여 저작권과 악의적 사용으로부터 보호하는 새로운 방법론인 DERMARK를 제안합니다. 기존 방법들은 워터마킹 용량을 고려하지 않고 고정된 길이의 세그먼트에 비트를 임베드하여 실패 가능성을 증가시킵니다. DERMARK는 텍스트의 엔트로피에 따라 동적으로 세그먼트를 조정하여 각 비트를 효율적으로 임베드하고, 강력한 성능을 보입니다.

- **Technical Details**: DERMARK는 LLM 로그에 기반하여 워터마크가 정상 분포를 따른다고 이론적으로 증명합니다. 이 방법은 비트를 임베드할 때 필요한 세그먼트를 동적으로 계산하고, 추가 오버헤드는 미미합니다. 또한, 텍스트의 편집에 강인성을 갖추기 위해 동적 프로그래밍을 활용하여 세그먼트 간 손실을 최소화합니다.

- **Performance Highlights**: 포괄적인 실험 결과, DERMARK는 SOTA(State Of The Art) 방법 대비 각 비트 임베드에 필요한 토큰 수를 평균 20% 줄이고, 워터마크 임베딩 시간을 50% 단축시켰습니다. 또한, 텍스트 편집 및 워터마크 삭제 공격에 대해 강력한 저항성을 보여줍니다.



### Decoding FL Defenses: Systemization, Pitfalls, and Remedies (https://arxiv.org/abs/2502.05211)
- **What's New**: 이번 논문은 Federated Learning(FL)에서의 poisoning 공격 방어 방법의 평가 가이드를 제시합니다. 기존의 방어 방법들은 실험 설정에서 발생하는 미세한 오류로 인해 실질적인 배포가 불가능한 경우가 많았습니다. 저자들은 이러한 문제점을 체계적으로 이해하고 다루기 위한 새로운 접근 방안을 제안합니다.

- **Technical Details**: 논문에서는 FL 방어 방법을 세 가지 주요 차원으로 체계화합니다: i) 클라이언트 업데이트 처리 방법, ii) 서버가 아는 정보, iii) 방어가 적용되는 단계. 50개의 탑티어 방어 논문을 조사하여 그 평가 설정에 공통적으로 사용되는 구성 요소를 파악하고, 여섯 가지 주요 함정을 발견하였습니다. 예를 들어, 약 30%의 연구가 기본적으로 강건한 MNIST 데이터셋만을 사용하였고, 40%는 단순한 공격 기법을 사용하여 방어가 강력하다고 잘못 나타낼 수 있습니다.

- **Performance Highlights**: 세 가지 대표적인 방어 방법을 사례 연구로 사용하여 발견된 함정의 영향을 재평가하였습니다. 이를 통해 이들이 강건성에 대한 잘못된 결론으로 이어진다는 점을 보여주었으며, 연구자들이 이를 극복하기 위해 취할 수 있는 실용적인 권장 사항을 제시합니다. 이러한 권장 사항은 FL 방어 방법 연구의 신뢰성을 높이는 데 도움을 줄 것입니다.



### Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities (https://arxiv.org/abs/2502.05209)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 위험성과 능력을 평가하는 새로운 접근 방식이 제안됩니다. 현재 대부분의 위험 평가는 시스템에서 유해한 행동을 촉발하는 입력을 설계하는 방식으로 이루어지지만, 이러한 방법은 모델의 최악의 경우 행동을 충분히 평가하지 못하는 한계를 가지고 있습니다. 이를 보완하기 위해, 연구자들은 latent activations이나 weights를 수정하는 model tampering attacks를 통해 LLM을 평가하는 방법을 소개합니다.

- **Technical Details**: 이 연구에서는 8개의 유니러닝(unlearning) 방법과 9개의 안전한(Safety) 파인튜닝 모델을 benchmark하고, 이를 11개의 능력 촉발 공격과 비교합니다. 또한 LLM의 다양한 능력 촉발 공격에 대한 저항력은 저차원 강건성 하위공간(low-dimensional robustness subspace)에 위치한다는 것을 보여줍니다. 모델 조작 공격의 성공률은 보류된 입력 공간 입력 공격의 성공을 예측할 수 있으며, 최신의 unlearning 방법들은 불과 16단계의 파인튜닝 내에서 쉽게 무효화될 수 있습니다.

- **Performance Highlights**: 모델에서 유해한 능력을 제거하는 것은 매우 어렵다는 것을 입증하는 연구 결과가 제시됩니다. 모델 조작 공격은 단순한 입력 공간 공격보다 훨씬 더 세밀한 평가를 가능하게 합니다. 연구팀은 64개의 모델을 훈련하여 오픈 소스 및 클로즈드 소스 모델의 위험을 평가하는데 도움을 줄 수 있는 리소스를 제공하며, 이 모델들은 다양한 강도로 이중 사용 생물학 지식을 언론하는 데 사용됩니다.



### Mitigation of Camouflaged Adversarial Attacks in Autonomous Vehicles--A Case Study Using CARLA Simulator (https://arxiv.org/abs/2502.05208)
- **What's New**: 이 논문에서는 자율주행차(Autonomous Vehicles, AV)의 카메라 입력을 조작하는 카메라 변장 적대적 공격(camera-camouflaged adversarial attacks)을 개발하여 교통 신호 인식을 목표로 합니다. 이 공격은 정지 신호(stop sign)의 텍스처를 수정하여 AV의 객체 감지 시스템을 속이고, 결과적으로 AV의 안전에 영향을 줄 수 있는 문제를 제기합니다. 연구진은 CARLA 자율주행 시뮬레이터를 사용하여 해당 공격의 효과를 테스트하였고, 여러 조건에서 실험을 수행하여 공격이 효과적이고 견고함을 확인했습니다.

- **Technical Details**: 본 연구는 CARLA AV 시뮬레이터를 활용하며, Robotic Operating System (ROS)와 결합하여 다양한 적대적 공격 및 방어 전략을 연구하는 모듈형 프레임워크를 구축합니다. 이 시스템은 모든 센서 및 처리 장치 간의 데이터 동기화를 보장하여 실시간으로 AV의 내비게이션 시스템이 적대적 공격에 대응하는 방법을 평가할 수 있습니다. 두 가지 방어 전략, 즉 속도 및 감지 거리 기반의 조정된 제동 체계(Defense-1)와 측면 카메라를 통한 센서 융합(Defense-2)을 개발하여 AV의 안전성 유지에 기여하고자 합니다.

- **Performance Highlights**: 리서치 결과, 적대적 공격이 이루어진 후 AV의 자동 제동 시스템이 정지 신호를 감지하는 데 지연이 발생할 수 있음을 확인했습니다. 공격이 없는 경우 AV는 성공적으로 정지 신호 앞에서 멈췄으나, 공격이 이루어진 경우 정확한 감지가 이루어지지 않아 운행 안전에 심각한 위험을 초래할 수 있음을 입증하였습니다. 이러한 연구 결과는 AV의 보안을 강화하고 안전한 작동을 보장하기 위한 중요한 통찰력을 제공합니다.



### Safety at Scale: A Comprehensive Survey of Large Model Safety (https://arxiv.org/abs/2502.05206)
Comments:
          47 pages, 3 figures, 11 tables

- **What's New**: 이 논문은 인공지능(AI) 분야의 대형 모델의 안전성에 관한 체계적 조사를 제공합니다. 특히, Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-Training (VLP) 모델과 같은 다양한 모델들의 안전성 취약점과 방어 전략을 정리하였습니다. 대형 모델의 안전성을 확보하기 위한 연구의 필요성과 국제적 협력의 중요성을 강조하고 있습니다.

- **Technical Details**: 대형 모델은 대규모 데이터 세트에서의 사전 학습을 통해 언어 이해, 이미지 생성, 복잡한 문제 해결 등의 작업에서 뛰어난 능력을 보입니다. 이러한 모델들은 적대적 공격(adversarial attacks), 백도어 공격(backdoor attacks), 데이터 중독(data poisoning), 에너지-지연 공격(energy-latency attacks) 등 다양한 안전 위협에 직면해 있습니다. 각 공격 유형에 대한 방어 전략 및 안전 연구를 위한 공통 데이터 세트와 벤치마크를 정리했습니다.

- **Performance Highlights**: 대형 모델의 안전성을 보장하는 것은 비의도적인 시스템 동작을 방지하고 개인 정보를 보호하기 위한 필수 사항입니다. 연구자와 실무자에게 유용한 참조 자료로 기능할 수 있으며, 포괄적인 방어 시스템 및 플랫폼의 지속적인 개발을 촉진하는 데 기여할 것입니다. 안전성 연구 현황을 통해 대형 모델의 발전을 가속화하고 안전한 AI 사용을 유도하는 것이 중요합니다.



### Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies (https://arxiv.org/abs/2502.05202)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 추론(Inference) 속도를 높이기 위한 새로운 방법인 Speculative Decoding(SD) 기법을 소개합니다. 기존의 SD 방법들과 달리, 제안된 세 가지 새로운 방법은 드래프(Drafter)와 타겟 모델간의 동일한 어휘(Vocabulary)를 요구하지 않습니다. 이를 통해 새로운 드래프를 훈련할 필요 없이 누구나 사용할 수 있는 모델을 드래프 모델로 활용할 수 있게 되었습니다.

- **Technical Details**: 이 연구는 세 가지 새로운 SD 알고리즘을 제안합니다. 첫 번째는 String-Level Exact Match(SLEM)라는 알고리즘으로, 드래프와 타겟 어휘 간에 텍스트를 공유하여 정확한 일치를 가능하게 합니다. 두 번째는 Token-Level Intersection(TLI)로, 드래프의 분포를 두 어휘의 교집합으로 조정하여 샘플링합니다. 마지막으로, String-Level Rejection Sampling(SLRS) 기법은 토큰 레벨이 아닌 문자열 레벨에서 샘플링을 수행하며, 이 모든 알고리즘은 손실이 없는(lossless) 특성을 유지합니다.

- **Performance Highlights**: 제안된 알고리즘들은 요약(Summarization), 프로그래밍(Programming), 긴 문맥(Long Context) 작업에서 기존의 자가 회귀(Self-Regressive) 디코딩 방법에 비해显著한 속도 향상을 보였습니다. Hugging Face Transformers 라이브러리에 통합되어 현재 26만 개의 리포지토리와 5천 개의 오픈 소스 패키지에서 즉각적인 성능 향상을 지원하고 있습니다. 이를 통해 실제 응용에서의 SD 프레임워크의 적용 가능성이 크게 넓어졌습니다.



### Multimodal Stock Price Prediction (https://arxiv.org/abs/2502.05186)
Comments:
          9 pages, 6 table

- **What's New**: 이 논문은 다양한 데이터 출처와 기계 학습(machine learning)을 결합하여 주가 예측(stock price prediction)의 정확성을 높이는 다중 모달(multimodal) 접근 방식을 탐구하고 있습니다. 전통적인 재무 지표, 트윗(tweets), 뉴스 기사 등에서 수집한 데이터를 융합하여 실시간 시장 동향과 투자자의 감정을 포착합니다. 특히 ChatGPT-4o와 FinBERT 모델을 사용하여 텍스트 데이터의 감성 분석(sentiment analysis)을 수행합니다.

- **Technical Details**: 연구에서는 표준 Long Short-Term Memory (LSTM) 모델과 결합된 다양한 데이터 스트림을 통해 예측(predictions)의 성능 향상을 입증합니다. 통합된 데이터 출처를 통한 예측력(forecasting effectiveness)의 증가율이 최대 5%에 달하는 것으로 나타났습니다. 또한 트윗과 뉴스 기사의 감성 분석을 포함시킴으로써 예측 능력(predictive capacities)의 상당한 영향을 강조합니다.

- **Performance Highlights**: 이 연구 결과는 다중 모달 데이터 분석 기술을 활용하여 금융 시계열(financial time series) 예측에 대한 체계적이고 효과적인 프레임워크를 제시하고 있습니다. 이를 통해 투자자들이 데이터 기반으로 의사 결정을 내리는 데 있어 새로운 관점을 제공합니다. 통합된 접근 방식은 예측 성과를 향상시킬 뿐만 아니라, 투자자의 감정과 시장 동태를 반영하는 데 중요한 역할을 합니다.



### Enhancing Team Diversity with Generative AI: A Novel Project Management Framework (https://arxiv.org/abs/2502.05181)
Comments:
          A published version can be found from here - this https URL

- **What's New**: 이 논문에서는 GenAI 기술을 활용한 새로운 프로젝트 관리 프레임워크를 제안합니다. 이 프레임워크는 학술 및 연구 프로젝트 팀에서 발생하는 유사한 팀 구성의 문제를 해결하도록 설계되었습니다. 특히, 대학 및 연구 기관 내에서 팀 구성원들의 성격과 역할을 바탕으로 성공적인 팀 멤버 패턴을 통합합니다.

- **Technical Details**: 이 연구에서는 GenAI 에이전트를 사용하여 팀 역학의 빈 부분을 메우고, 이를 위해 성격 데이터셋에 대해 조정된 GenAI 에이전트를 활용합니다. 또한 전통적인 프로젝트 관리 프로세스에 대한 추가 분석 계층을 도입하여, 팀 구성원들의 성격과 역할을 평가하고, 효과적인 팀 역할 분담을 시도합니다. 초기 실험 결과, 성격 특성을 이해하고 처리하는 모델의 능력이 향상되었습니다.

- **Performance Highlights**: 초기 실험 결과에 따르면 GenAI 팀원이 실제 프로젝트 환경에서 효과적일 가능성이 제시되고 있습니다. 이 연구는 팀의 다양성을 높이고 프로젝트 관리를 개선하는 AI의 실용적 응용을 탐구할 예정입니다. 특히, GenAI 기술을 통해 팀 구성의 효율성과 생산성이 높아질 것으로 기대됩니다.



### Is Prior-Free Black-Box Non-Stationary Reinforcement Learning Feasible? (https://arxiv.org/abs/2410.13772)
Comments:
          Corrected minor typos in the proof of Theorem 2 on pages 25 and 26

- **What's New**: 본 논문에서는 시스템의 비정상성(non-stationarity)에 대한 사전 지식 없이 비정상 강화 학습(Non-Stationary Reinforcement Learning, NS-RL) 문제를 연구합니다. 특히, MASTER라는 최신 블랙박스 알고리즘의 비정상성 탐지 메커니즘이 실제 수평(horizon) 선택에서 발동되지 않음을 입증하여, 성능이 무작위 재시작(random restarting) 알고리즘과 유사하다는 것을 보여줍니다.

- **Technical Details**: 주요 기술 세부사항으로는 MASTER의 비정상성 탐지 메커니즘이 실용적인 수평 선택에서 제대로 작동하지 않는다는 점과, 제시된 후회(bound)한계가 최적의 순서(order optimal)임에도 불구하고 최악의 경우 선형 후회(linear regret)보다 높게 유지된다는 점입니다. 이를 통해 MASTER의 성능을 평가하고자, 조각별 정적 다중 팔 밴디트(piecewise stationary multi-armed bandits, PS-MABs)에서 실험을 수행하였습니다.

- **Performance Highlights**: 연구 결과, QUICK CHANGE DETECTION(QCD) 기법을 활용한 방법들이 MASTER 및 다른 무작위 재시작 접근법에 비해 더 강력하고 일관되게 뛰어난 성능을 보임을 확인하였습니다. 또한, PS-MABs에 대한 무작위 재시작 알고리즘을 제안하여 기준선으로 활용하였습니다.



### Self-supervised Domain Adaptation for Breaking the Limits of Low-quality Fundus Image Quality Enhancemen (https://arxiv.org/abs/2301.06943)
- **What's New**: 이번 연구에서는 Diabetic Retinopathy (DR)와 Diabetic Macular Edema (DME) 같은 안과 질환 진단을 위한 저품질 망막 단층 촬영 이미지의 품질을 전적으로 비지도 학습 방식으로 향상시키는 방법을 제안합니다. 기존의 이미지 개선 방법들이 고품질 이미지를 필요로 했던 반면, 본 연구는 고품질 레퍼런스 이미지를 사용하지 않고도 이미지 품질을 개선할 수 있는 자가 지도 학습(task)을 활용합니다.

- **Technical Details**: 연구진은 보조 학습된 품질 평가 네트워크와 스타일 클러스터링을 활용하여 여러 패치 기반 도메인을 구성합니다. 두 가지 자가 지도 도메인 적응(task)을 통해 망막 이미지의 내용, 저품질 요소 및 스타일 정보를 분리하는 방법을 고안하였습니다. 이를 통해 저품질 이미지 개선의 견고성을 높이고 스타일 일관성 문제를 해결합니다.

- **Performance Highlights**: EyeQ와 Messidor 데이터셋에 대한 광범위한 실험을 통해, 연구에서 제안한 DASQE 방법이 저품질 이미지만을 사용할 경우 새로운 최첨단 성능을 달성했음을 보여줍니다. 이 결과는 안과 의사들이 저품질 이미지를 바탕으로 보다 정확하게 진단할 수 있도록 돕는 가능성을 제시합니다.




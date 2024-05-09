### You Only Cache Once: Decoder-Decoder Architectures for Language Models (https://arxiv.org/abs/2405.05254)
- **What's New**: 새로운 디코더-디코더 아키텍처인 YOCO를 소개합니다. 이 아키텍처는 큰 규모의 언어 모델을 위해 한 번만 key-value 쌍을 캐싱합니다. YOCO는 자체 디코더(self-decoder)와 교차 디코더(cross-decoder)의 두 부분으로 구성되어 있으며, 전체 모델은 디코더 전용 Transformer처럼 작동합니다. 이 설계는 GPU 메모리 요구 사항을 크게 줄이면서도 전역 주의력(global attention) 기능을 유지합니다.

- **Technical Details**: YOCO는 한 번의 캐싱(only caches once)으로 더 효율적인 구조를 제공합니다. 자체 디코더는 자체 주의(self-attention)를 사용하여 글로벌 KV 캐시를 효율적으로 인코딩하고, 교차 디코더는 교차 주의(cross-attention)를 사용하여 공유된 KV 캐시를 재사용합니다. 이 아키텍처는 자동 회귀 생성(auto-regressive generation) 작업에 적합하며, 패딩 단계(prefill stage)의 속도를 대폭 향상시키고, 분산된 긴 시퀀스 학습에 더 효율적인 시스템 설계를 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면 YOCO는 여러 설정에서 Transformer와 비교하여 유리한 성능을 달성했습니다. 특히, 3B 모델을 수조 개의 학습 토큰으로 확장했을 때, 안정된 언어 모델(StableLM)과 비교할 수 있는 결과를 보였습니다. 또한, YOCO는 1M 토큰의 컨텍스트 길이로 확장되어 거의 완벽한 니들 검색 정확도(near-perfect needle retrieval accuracy)를 달성했습니다. 프로파일링 결과는 KV 캐시의 메모리를 크게 줄이고, 패딩 지연 시간과 처리량을 향상시켜 서빙 용량을 높였다는 것을 보여줍니다.



### Open Source Language Models Can Provide Feedback: Evaluating LLMs' Ability to Help Students Using GPT-4-As-A-Judg (https://arxiv.org/abs/2405.05253)
Comments:
          7 pages, 4 figures, 2 tables. Accepted for publication at the 29th annual ACM conference on Innovation and Technology in Computer Science Education (ITiCSE 2024)

- **What's New**: 이 연구에서는 오픈 소스 대규모 언어 모델(Large Language Models, LLMs)이 교육 분야에서 자동으로 피드백을 생성하는 능력에 대해 탐구하였습니다. 자동화된 평가를 위해 GPT-4를 사용하여 여러 오픈 소스 모델이 생성한 피드백의 질을 분석하였으며, 이러한 모델들이 대중적인 독점 LLMs, 예를 들어 ChatGPT와 경쟁력 있는 성능을 제공할 수 있는지를 평가했습니다.

- **Technical Details**: 연구팀은 GPT-4를 사용하여 오픈 소스 모델들이 생성한 피드백의 질을 평가하는데, 이는 기존의 GPT-4와 같은 강력한 모델들을 사용하여 덜 강력한 모델들의 출력을 평가하는 최근 연구에서 영감을 받은 것입니다. 평가는 첫째, GPT-4와 인간 전문가의 평가를 비교함으로써 GPT-4를 자동 평가자로서의 타당성을 조사하는 방법으로 진행되었습니다. GPT-4는 피드백을 긍정적으로 평가하는 경향이 있으며 인간 평가자와의 중간 수준의 일치를 보였습니다.

- **Performance Highlights**: 오픈 소스 모델들 중 일부는 ChatGPT와 같은 인기 있는 독점 LLMs와 경쟁력 있는 성능을 보여주었습니다. 이는 오픈 소스 LLMs의 교육 환경에서의 책임 있는 사용 가능성을 시사합니다. GPT-4는 높은 평가능력과 함께 경향성의 편향을 보이기는 했지만, 오픈 소스 모델들의 성능 평가에 있어 중요한 도구로서 활용될 수 있습니다.



### LLMs with Personalities in Multi-issue Negotiation Games (https://arxiv.org/abs/2405.05248)
- **What's New**: AI 에이전트가 대형 언어 모델 (LLMs)의 도움으로 많은 인간의 작업을 수행할 수 있게 되었습니다. 이 연구에서는 게임 이론(game-theoretical) 프레임워크를 사용하여 LLM의 협상 능력을 측정했습니다. 또한 공정성과 위험 개념을 측정하는 데 있어서의 방법론적 도전을 탐구했습니다.

- **Technical Details**: 연구자들은 단일 이슈(single-issue) 및 다중 이슈(multi-issue) 협상에 대해 1,500회의 시뮬레이션을 수행했습니다. 복잡한 도메인에서는 비대칭 이슈 평가(asymmetric issue valuations)가 협상 성공률을 향상시키지만, 공격적 협상으로 인해 여분의 이익이 감소했습니다. 또한, 그라디언트 부스티드 회귀(gradient-boosted regression)와 샤플리 설명자(Shapley explainers)를 통해 높은 개방성, 성실성, 신경성이 공정한 경향과 연관되어 있으며, 낮은 동의성과 낮은 개방성은 합리적 경향과 관련이 있습니다는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, LLM은 기본적으로 공정한 행동을 기본 설정으로 하고 있지만, 동의할 수 있는 상대를 이용하려는 'jail broken' 경향이 있을 수 있습니다. 연구는 또한 협상 봇의 설계와 게임 이론 및 계산 사회 과학에 기반한 협상 행동의 평가 프레임워크를 제공합니다.



### CARE-SD: Classifier-based analysis for recognizing and eliminating stigmatizing and doubt marker labels in electronic health records: model development and validation (https://arxiv.org/abs/2405.05204)
Comments:
          28 pages, 3 figures, 4 tables. 5 Appendices

- **What's New**: 본 연구는 집중 치료 전자건강기록(EHR)에서 낙인 찍히는 언어와 편향된 언어를 감지하고 분류하기 위해 자연어 처리(Natural Language Processing, NLP) 기술을 사용했습니다. 특히, 문헌에서 도출된 기본 단어(stem words)로부터 어휘집(lexicon)과 정규 표현식 목록을 생성하고, 이를 Word2Vec 및 GPT 3.5를 활용해 확장시키고, 인간 평가를 통해 정제하는 방법론을 개발하였습니다.

- **Technical Details**: 연구팀은 먼저 낙인 찍힌 환자 라벨, 의심 마커(doubt markers), 공포 인용 부호(scare quotes)의 언어적 특징에 대한 어휘집과 정규 표현식 리스트를 구축하였습니다. 이후, Word2Vec과 GPT 3.5를 사용하여 어휘집을 확장하고, 인간 평가를 통해 이를 정제하였습니다. 어휘집을 사용하여 MIMIC-III 데이터셋에서 1천 8백만 문장을 검색하고, 각 언어적 편향 특징에 대해 1000개의 문장 샘플을 추출하여 임상 및 공중 보건 전문가에 의해 라벨링하였습니다. 이 데이터는 감독 학습 분류기(Supervised Learning Classifiers) 훈련에 사용되었습니다.

- **Performance Highlights**: 개발된 어휘집을 통해 의심 마커 어휘집에는 58개의 표현식이, 낙인 찍힌 라벨 어휘집에는 127개의 표현식이 포함되어 있습니다. 의심 마커와 낙인 찍힌 라벨을 위한 분류기는 각각 매크로 F1-점수(macro F1-scores) 0.84와 0.79로 높은 성능을 보였으며, 양성 라벨의 재현율과 정밀도는 0.71에서 0.86 사이였고, 정확도는 인간 평가자와의 일치도가 0.87로 높게 나타났습니다. 이는 의료 텍스트에서 낙인 찍힌 라벨과 의심 마커를 자동으로 식별하는 데 유용합니다.



### MIDGARD: Self-Consistency Using Minimum Description Length for Structured Commonsense Reasoning (https://arxiv.org/abs/2405.05189)
Comments:
          Under review at ACL 2024

- **What's New**: 이 연구는 자연어 입력에서 추론 그래프를 생성하는 구조화된 추론 작업을 다룹니다. 기존의 접근법은 다양한 프롬프트 계획을 탐색했지만, 오류 전파 문제와 단일 패스 기반 디코딩의 한계로 인해 정확성이 떨어지는 경향이 있습니다. 이를 해결하기 위해, 저자들은 자기 일관성(Self-Consistency, SC)에서 영감을 받아 MIDGARD(최소 설명 길이를 활용한 추론의 지향적 비순환 그래프 집약 방안)를 제안합니다.

- **Technical Details**: MIDGARD는 최소 설명 길이(Minimum Description Length, MDL) 기반의 포뮬레이션을 사용하여 LLM이 생성한 다양한 그래프 샘플들 사이에서 일관된 특성을 식별합니다. 이 방법은 몇 개의 샘플에서만 나타나는, 잘못될 가능성이 높은 속성을 거절하고, 정밀도를 저해하지 않으면서 누락된 요소를 포함시킬 수 있는 능력을 제공합니다.

- **Performance Highlights**: MIDGARD 방법은 인자 구조 추출, 설명 그래프 생성, 일상 작업에 대한 행동 간 의존성 추론, 자연 언어 텍스트에서의 의미 그래프 생성 등 다양한 구조화된 추론 작업에서 향상된 성능을 보여 주었습니다.



### Encoder-Decoder Framework for Interactive Free Verses with Generation with Controllable High-Quality Rhyming (https://arxiv.org/abs/2405.05176)
Comments:
          18 pages, 1 figure

- **What's New**: 이 연구에서는 PLM(Pretrained Language Model, 사전 훈련된 언어 모델)과 호환되면서도 가사의 라임을 보다 효과적으로 생성할 수 있는 새로운 파인 튜닝(fine-tuning) 접근법을 제안합니다. 새로운 방법은 가사 작성 시, 라이밍 단어를 가사의 시작 부분에 미리 배치함으로써 모델이 가사의 내용을 결정하기 전에 중요한 라이밍 결정을 내릴 수 있도록 합니다. 이는 기존의 역방향 언어 모델링(reverse language modeling) 방식과는 다르며, 이를 통해 기존 PLM의 장점을 유지하면서도 향상된 결과를 도출할 수 있습니다.

- **Technical Details**: 연구팀은 자체적으로 고안된 파인 튜닝 방법을 사용하여, 라이밍 단어를 각 가사의 시작에 추가하고, 전체 가사는 여전히 왼쪽에서 오른쪽으로(left-to-right) 생성합니다. 이 방법은 기존의 PLM을 사용하는 것과 호환되기 때문에, PLM에서 이미 학습된 많은 언어적 특성들을 활용할 수 있는 이점이 있습니다. 또한, 연구팀은 영어 뿐만 아니라 12개의 다른 언어에 대한 고품질 데이터셋을 제공하며, 이를 통해 다양한 언어 환경에서의 적용 가능성을 분석하였습니다.

- **Performance Highlights**: 이 새로운 파인 튜닝 접근법은 기존의 상태 최신 기술(state-of-the-art) 방법들과 비교하여 더 읽기 쉽고 더 나은 라이밍 결과를 생성하는 것으로 나타났습니다. 또한, 실험 결과는 가사 생성의 좋은 및 나쁜 관행에 대한 통찰을 제공하고, 향후 방법론을 비교하기 위한 지표들(metrics)을 제안합니다. 이 연구는 PLM을 사용하는 현대의 자연 언어 처리 기술에 새로운 방향을 제시할 수 있으며, 멀티링구얼(multilingual) 환경에서의 활용 가능성을 높이 평가받고 있습니다.



### Motion Capture Analysis of Verb and Adjective Types in Austrian Sign Languag (https://arxiv.org/abs/2405.05161)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 연구는 오스트리아 수화(Österreichische Gebärdensprache, ÖGS)에서 동사와 형용사의 수화 생성에 대한 운동 인자들을 정량적으로 분석했습니다. 특히 동사의 완결점 유무에 따른 차이와 형용사의 강조 유무에 따른 변화를 조사하였습니다.

- **Technical Details**: 연구는 네 명의 청각장애인 수화 사용자를 대상으로 모션 캡처 데이터를 사용하여 수화 생성 시의 운동학적 매개변수를 조사했습니다. 선형 혼합 효과 모델(Linear-Mixed Effects Models, LME)을 사용한 데이터 분석을 통해, 동사의 완결점 표시와 형용사의 강조 표시가 모두 ÖGS에서의 움직임 조절로 표현됨을 발견했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 완결점을 갖는 동사(예: 도착하기)는 그렇지 않은 동사(예: 분석하기)에 비해 더 높은 최대 속도와 더 짧은 지속 시간을 보였습니다. 또한 강조된 형용사는 강조되지 않은 형용사에 비해 더 긴 지속 시간으로 표현되었습니다. 사용자 개개인의 차이는 개인의 수화 스타일로 해석될 수 있습니다.



### XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples (https://arxiv.org/abs/2405.05116)
- **What's New**: 이 논문에서 소개된 XAMPLER (Cross-Lingual Example Retrieval)는 멀티리프 네이티브 언어 데이터만을 사용하여 교차 언어 맥락 학습(in-context learning)의 문제를 해결하기 위해 설계된 새로운 방법입니다. XAMPLER는 특히 영어 데이터로부터 양성/음성 샘플을 통해 학습되며, 이를 다양한 언어로 확장하여 맥락 학습을 강화할 수 있습니다.

- **Technical Details**: XAMPLER는 먼저 많은 언어를 처리할 수 있는 대규모 다국어 언어 모델 (multilingual large language model)의 예측을 기반으로 생성된 긍정적 및 부정적 영어 샘플들로 구성된 검색기(retriever)를 훈련합니다. 그 후, 이 훈련된 검색기를 사용하여 타겟 언어의 맥락 학습을 위한 영어 예시들을 few-shot 예제로 검색하는데 직접적으로 활용합니다.

- **Performance Highlights**: SIB200의 광범위한 다언어 텍스트 분류 벤치마크에서 XAMPLER는 총 176개 언어에서 맥락 학습 성능을 크게 향상시켰습니다. 이는 다양한 언어에 걸쳐 효과적인 맥락 학습 전략을 제공합니다.



### QFMTS: Generating Query-Focused Summaries over Multi-Table Inputs (https://arxiv.org/abs/2405.05109)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 새로운 질의 중심의 다중 테이블 요약(query-focused multi-table summarization) 방법을 제안합니다. 기존의 테이블 요약 방법이 사용자의 정보 요구 사항과 질의의 복잡성을 충분히 만족시키지 못하는 점에 착안하여, 본 연구는 사용자의 정보 요구에 맞춘 질의 의존적 테이블 요약을 생성하기 위해 테이블 직렬화 모듈(table serialization module), 요약 컨트롤러(summarization controller), 그리고 대규모 언어 모델(LLM, large language model)을 포함하는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법은 텍스트 질의와 다수의 테이블을 사용하여 사용자의 정보 요구에 맞춘 질의 의존적 테이블 요약을 생성합니다. 연구를 지원하기 위해 특별히 맞춤화된 데이터셋을 제공하는데, 이는 4909개의 질의-요약 쌍(query-summary pairs)과 각각에 연관된 다수의 테이블로 구성되어 있습니다.

- **Performance Highlights**: 실험을 통해 기존의 기준 모델들과 비교하여 제안한 방법의 효과를 입증했습니다. 복잡한 테이블 추론(complex table reasoning)에 대한 도전을 극복하고 정확한 요약을 위한 연구 진전에 기여하는 결과를 얻었습니다.



### Conversational Topic Recommendation in Counseling and Psychotherapy with Decision Transformer and Large Language Models (https://arxiv.org/abs/2405.05060)
Comments:
          5 pages excluding references, 3 figures; accepted at Clinical NLP Workshop @ NAACL 2024

- **What's New**: 인공지능 (AI), 특히 대형 언어 모델 (Large Language Models, LLMs)의 사용이 정신건강 지원 시스템에 통합됨에 따라, 특히 상담 대화에서 화제 추천을 위한 결정 트랜스포머 (Decision Transformer) 아키텍처의 적용이 주목받고 있습니다. 이 아키텍처는 오프라인 강화 학습 (offline reinforcement learning)에 활용되며, 이전 대화 턴에서 상태 (dialogue turn embeddings), 행동 (conversation topics), 보상 (scores measuring the alignment between patient and therapist)을 추출하여 모델을 트레이닝합니다.

- **Technical Details**: 결정 트랜스포머는 강화 학습을 위해 설계된 트랜스포머 모델로, 기존 강화 학습 방법들보다 뛰어난 성능을 보여줍니다. 이 연구에서는 결정 트랜스포머를 사용하여 얻은 결과를 합성 레이블 (synthetic labels)로 사용하고, 이를 이용해 대형 언어 모델을 미세 조정하는 새로운 시스템을 제안합니다. 특히, LLaMA-2 7B 모델을 기반으로 구현된 결과가 혼합적이긴 하지만, 미래의 연구는 이 설계를 토대로 개선을 진행할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 시스템은 기존의 강화 학습(RL) 방법들에 비해 우수한 성능을 보이며, 결정 트랜스포머가 주제 추천 작업에 있어서 더 나은 결과를 도출해냄을 입증합니다. 또한, 생성된 합성 데이터를 이용하여 언어 모델을 미세 조정함으로써, 언어 모델이 학습할 수 있는 데이터를 증가시키는 접근 방식을 탐구합니다. 최종적으로, LLaMA-2 모델을 사용한 시퀀스 분류 실험을 통해 추가적인 베이스라인 비교를 제공합니다.



### Seeds of Stereotypes: A Large-Scale Textual Analysis of Race and Gender Associations with Diseases in Online Sources (https://arxiv.org/abs/2405.05049)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 사전 훈련 데이터셋에서 나타날 수 있는 인종 및 성별 편향을 분석하였습니다. 특히, 다양한 웹 소스(예: Arxiv, Wikipedia, Common Crawl)에서 수집된 데이터셋을 사용하여 다양한 질병이 인종 및 성별과 어떻게 연관되어 논의되는지를 살펴보았습니다.

- **Technical Details**: 연구진은 여러 웹 소스에서 얻은 텍스트 데이터를 활용해 질병과 인구 통계학적 용어(인종과 성별)의 연결고리를 분석하였습니다. 이러한 분석을 통해 LLMs가 어떤 편향을 배우고 내재화할 수 있는지를 조사했습니다. 연구진은 실제 인구 통계와 질병 발생 빈도 데이터 및 GPT-4(Large Language Models의 예)의 결과와 비교분석을 시행하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 온라인 텍스트에서 특정 질병 개념과 관련하여 인구 통계학적 용어가 불균형하게 연관되어 나타났습니다. 특히 'Black' 인종 용어는 인구 비율에 비해 현저히 과대 표현되는 경향을 보였습니다. 이는 LLMs의 예측 및 가공에 있어 세심한 접근과 전략이 필요함을 시사합니다.



### ADELIE: Aligning Large Language Models on Information Extraction (https://arxiv.org/abs/2405.05008)
- **What's New**: 이 연구에서는 정보추출(Information Extraction, IE) 과제들에 대해 효과적으로 해결할 수 있는 ADELIE 모델을 소개합니다. 정보추출 과제는 폐쇄형(Closed IE), 개방형(Open IE), 수요응답형(On-demand IE)으로 나뉘어져 있으며, ADELIE는 이와 같은 다양한 유형을 처리할 수 있는 큰 언어 모델입니다.

- **Technical Details**: ADELIE 모델은 IE에 초점을 맞춘 고품질 맞춤 데이터 세트 IEInstruct를 수집, 구축하여 훈련됩니다. 초기에는 IEInstruct를 사용하여 명령어 튜닝(instruction tuning)을 통해 ADELIE_SFT 모델을 훈련시키고, 이후에는 직접 선호도 최적화(Direct Preference Optimization, DPO) 목표를 적용하여 ADELIE_DPO 모델을 추가적으로 훈련합니다.

- **Performance Highlights**: ADELIE 모델은 다양한 보류 중인 정보추출 데이터 세트에서 광범위한 실험을 통해 최고의 성능(State-of-the-art, SoTA)을 달성하였습니다. 이 모델들은 오픈소스 모델들 중에서도 뛰어난 성과를 보여주며, 일반 능력(general capabilities)을 평가한 실험 결과에서도 특별한 성능 저하가 발생하지 않음을 확인하였습니다. 연구를 더욱 촉진하기 위해 코드, 데이터 및 모델을 공개할 예정입니다.



### P-ICL: Point In-Context Learning for Named Entity Recognition with Large Language Models (https://arxiv.org/abs/2405.04960)
- **What's New**: 이 논문에서는 LLM (Large Language Models)을 사용하여 NER (Named Entity Recognition)을 효과적으로 수행할 수 있는 새로운 프레임워크인 P-ICL (Point In-Context Learning)을 소개합니다. P-ICL은 표준 ICL (In-Context Learning)의 한계를 극복하고, 명명된 개체 유형을 보다 정확하게 분류할 수 있는 데 필요한 '포인트 엔티티'를 활용합니다.

- **Technical Details**: P-ICL 방법론은 각 엔티티 타입을 표현하는 대표적인 인스턴스를 제공함으로써 LLM이 해당 엔티티 유형을 더욱 정확하게 이해하고 분류할 수 있도록 합니다. 이 연구에서는 K-Means 클러스터링 (K-Means clustering)을 기반으로 하는 포인트 엔티티 선정 방법도 제안되어, 포인트 엔티티의 최적화된 선택이 LLM의 성능 향상에 기여하는 것을 보여줍니다.

- **Performance Highlights**: P-ICL 프레임워크와 포인트 엔티티 선택 방법의 유효성을 검증하기 위한 다양한 NER 벤치마크 실험에서 P-ICL은 표준 ICL 접근방식보다 우수한 결과를 보였습니다. 특히, 포인트 엔티티를 이용한 방법이 무작위 선택보다 효과적인 것으로 나타나, 엔티티 인식 및 분류 작업에서의 정보 제공 측면을 강화시킴을 입증했습니다.



### Improving Long Text Understanding with Knowledge Distilled from Summarization Mod (https://arxiv.org/abs/2405.04955)
Comments:
          arXiv admin note: text overlap with arXiv:2110.04741

- **What's New**: 본 논문에서는 자연어 처리(Natural Language Processing, NLP) 분야에서 중요하지만 도전적인 긴 텍스트 이해 문제를 해결하기 위해 'Gist Detector(요지 감지기)'를 제안합니다. 이 방법은 요약 모델로부터 추출한 핵심 내용(gist)을 하류(downstream) 모델에 통합하여 긴 텍스트의 이해 능력을 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: Gist Detector는 먼저 요약 모델로부터 요지 감지 지식을 추출(learn)하여 학습하고, 그 후 요지 인식(representation)을 통해 하류 모델을 강화(augment)합니다. 구체적으로, 요약 모델에서 distillation(증류)된 지식을 바탕으로 gist-aware 표현을 생성하여, 이를 하류 태스크에 적용함으로써 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 이 방법은 긴 문서 분류(long document classification), 간접적으로 감독된 오픈 도메인 질문 응답(distantly supervised open-domain question answering), 비평행 텍스트 스타일 변환(non-parallel text style transfer)의 세 가지 다른 태스크에서 기존 모델들의 성능을 현저하게 향상시켰습니다. 실험 결과는 Gist Detector가 모든 태스크에서 베이스라인 모델들의 성능을 크게 개선할 수 있음을 보여줍니다.



### Machine Learning-based NLP for Emotion Classification on a Cholera X Datas (https://arxiv.org/abs/2405.04897)
- **What's New**: 최근 사회 미디어를 통해 밝혀진 함마스크라알(지명) 콜레라 발병과 관련된 게시물 중 감정 분류가 주목을 받고 있습니다. 이 연구는 23,000개의 소셜미디어 게시글에 표현된 감정을 분석하여 콜레라가 사회에 미치는 영향을 보다 깊이 이해하는 데 도움을 주는 도구로서 감정 분류의 가능성을 제시합니다.

- **Technical Details**: 연구에 사용된 소셜 미디어 게시물 데이터셋은 Python의 Natural Language Toolkit(NLTK) 및 Machine Learning(ML) 모델들을 이용하여 감정 분석을 수행했습니다. 구체적으로 Long short-term memory (LSTM), Logistic regression, Decision trees, 그리고 Bidirectional Encoder Representations from Transformers (BERT) 모델이 사용되어 LSTM이 가장 높은 정확도 75%를 달성했습니다.

- **Performance Highlights**: LSTM 모델은 여러 ML 모델 중 가장 높은 정확도인 75%를 기록하였고, 이는 콜레라에 대한 사회적 반응을 분석하는 데 가장 효과적인 방법 중 하나임을 시사합니다. 이러한 분류 방법은 공중 보건 전략을 개발하는 데 중요한 기여를 할 수 있습니다.



### Logical Negation Augmenting and Debiasing for Prompt-based Methods (https://arxiv.org/abs/2405.04872)
- **What's New**: 이 연구에서는 프롬프트 기반 방법(Prompt-based methods)이 자연어 처리(NLP)에서 어떻게 논리적 추론(logical reasoning)에 효과적으로 사용될 수 있는지에 초점을 맞추고 있습니다. 특히, 논리 부정(logical negation)이라는 측면에서 발생하는 문제를 다루며, 이를 해결하기 위해 'Negation Augmenting and Negation Debiasing (NAND)'라는 새로운 방법을 제안합니다.

- **Technical Details**: NAND는 파라미터 업데이트 없이 프롬프트 기반 방법에 부정 명제(negative propositions)를 도입함으로써, 논리 부정의 부정적인 영향을 상쇄할 수 있는 방법입니다. 분석을 통해 논리 부정이 부정적인 답변으로의 잘못된 연관성(spurious correlations)을 유발하고, 부정이 없는 명제들은 긍정적인 답변과 연관되는 경향이 있음을 밝혀냈습니다. NAND는 이러한 편향을 완화하기 위해 모든 인스턴스에 'not'을 제공하여 모델이 논리 부정의 유무만으로 결정을 내리지 못하게 합니다.

- **Performance Highlights**: 세 개의 데이터셋(RuleTaker, ProofWriter, LogicNLI)을 사용한 실험에서 NAND는 논리 부정의 정확성을 향상시키고 프롬프트 기반 방법의 논리적 추론 능력을 크게 증가시켰습니다. 이 방법은 모델 재학습 없이도 수행될 수 있으며, 지도학습(supervised models)과의 성능 격차를 줄이고 더 큰 일반화(generalization)를 보여주었습니다.



### Fine-tuning Pre-trained Named Entity Recognition Models For Indian Languages (https://arxiv.org/abs/2405.04829)
Comments:
          8 pages, accepted in NAACL-SRW, 2024

- **AI Newsletter - Korean Edition**: [{"What's New": '이번 연구는 인도어를 대상으로 한 다국어 이름 인식(Multilingual Named Entity Recognition)에 중점을 두었습니다. 인도의 주요 언어 가족 두 가지로부터 선택된 4개 언어에 대해 40,000개의 문장을 포함하는 인간이 주석을 단 명명된 실체 코퍼스를 제시하며, 이를 통해 특히 인도어에 적합한 기술을 제안합니다.'}, {'Technical Details': '연구팀은 이름 인식(Named Entity Recognition, NER)을 위한 새로운 코퍼스를 생성하고, 이를 기반으로 다국어 모델을 훈련시켰습니다. 이 모델은 평균 F1 점수가 0.80에 이르는 성능을 보였으며, 주목할 만한 점은 기존에 본적 없는 벤치마크 데이터셋에서도 비슷한 수준의 성능을 나타냈다는 점입니다.'}, {'Performance Highlights': '제안된 다국어 모델은 인도어 데이터셋에서 평균 0.80 F1 점수로 우수한 성능을 보였습니다. 또한, 이 모델은 인도어에 대한 새로운 벤치마크 데이터셋에서도 비슷한 성능을 유지함으로써 그 사용가능성을 입증했습니다.'}]



### ChuXin: 1.6B Technical Repor (https://arxiv.org/abs/2405.04828)
Comments:
          Technical Report

- **What's New**: 새로운 오픈 소스 언어 모델인 ChuXin을 소개하며, 이 모델은 1.6 billion parameters를 가지고 있습니다. ChuXin은 단순히 모델의 무게(weights)와 아키텍처(architecture)만을 공개하는 것이 아니라, 모델 훈련을 위해 필요한 모든 것을 제공합니다. 이에는 훈련 데이터, 훈련 과정, 평가 코드 등이 포함됩니다. 이는 개방형 연구 커뮤니티를 강화하고 투명성을 증진하는 동시에 언어 모델링 분야에서의 혁신을 촉진하는 것을 목표로 합니다.



### APrompt4EM: Augmented Prompt Tuning for Generalized Entity Matching (https://arxiv.org/abs/2405.04820)
- **What's New**: 이 연구에서는 데이터 관리의 핵심 작업인 일반화된 엔티티 매칭(GEM)의 낮은 자원 상황에서의 도전을 해결하기 위해 증강된 프롬프트 튜닝(Augmented Prompt Tuning) 프레임워크를 제안합니다. GEM은 서로 다른 형식으로 표현된 두 레코드가 같은 실제 세계의 엔티티를 참조하는지 판단하는 작업입니다. 제안된 프레임워크는 효과적인 소프트 토큰 기반 프롬프트 튜닝 방법과 비용 효율적인 정보 증강 전략을 포함합니다.

- **Technical Details**: 새로운 프레임워크는 컨텍스트화된 소프트 토큰(contextualized soft token) 기반의 프롬프트 튜닝 방법과 대규모 언어 모델(LLMs)을 활용하는 정보 증강 전략을 주요 개선점으로 도입합니다. 이를 통해 기존의 프롬프트 디자인 문제와 정보 격차 문제를 해결하고자 합니다.

- **Performance Highlights**: 이 연구의 기초 모델은 평균 5.24% 이상의 성능 향상을 보여주었고, 정보 증강을 포함한 모델은 미세 조정된 대규모 언어 모델(LLMs)과 비슷한 성능을 달성하면서 API 비용의 14% 미만을 사용함으로써 비용 효율성을 입증했습니다.



### DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literatur (https://arxiv.org/abs/2405.04819)
Comments:
          Under Review

- **What's New**: 최근 대형 언어 모델 (LLMs) 개선을 통해 다양한 애플리케이션에서 유망한 성능을 거두고 있지만, 전문 분야에서 LLMs의 원활한 채택을 방해하는 장기적인 도전이 계속되고 있습니다. 본 논문에서는 이러한 한계를 극복하기 위해 DALK(Dynamic Co-Augmentation of LLMs and KG)를 소개하고, 생물의학 및 글로벌 보건 우선 순위인 알츠하이머병(AD) 분야에서의 능력을 시연합니다.

- **Technical Details**: DALK 프레임워크는 LLM과 지식 그래프(KG)가 서로를 강화하는 방식으로 구성되어 있습니다. 첫째, LLM을 사용하여 AD 관련 과학문헌에서 추출한 계속 진화하는 AD특정 지식 그래프를 구축합니다. 그 후, 새로운 자기인식 지식 검색 방법과 함께 조밀한 샘플링 방법을 사용하여 KG에서 적합한 지식을 선택하고 LLM 추론 능력을 향상시킵니다.

- **Performance Highlights**: 생성된 AD 질문 응답(ADQA) 벤치마크에서 수행된 실험 결과는 DALK의 효과를 강조합니다. 또한, KG와 LLM을 상호 강화하는 새로운 주제에 대한 유익한 통찰과 지침을 제공할 수 있는 일련의 상세한 분석을 수행했습니다.



### ACORN: Aspect-wise Commonsense Reasoning Explanation Evaluation (https://arxiv.org/abs/2405.04818)
Comments:
          18 pages, 7 figures, under review. Data available here: this https URL

- **What's New**: 새로운 데이터셋 ACORN과 이를 이용한 대규모 언어 모델(Large Language Models, LLMs)의 자유롭게 작성된 설명 평가 연구가 소개되었습니다. 이 연구는 텍스트 설명의 다양한 측면에 대한 품질 평가를 자동화하는 데 LLM의 효용성을 탐구했습니다. 또한, LLM을 인간 평가자가 부족한 상황에서 추가 평가자로 활용하는 방안도 검토되었습니다.

- **Technical Details**: ACORN 데이터셋에는 3,500개의 자유 텍스트 설명과 각 측면별 품질 평점이 포함되어 있습니다. 이를 통해 다양한 설정과 품질 측면에서 LLM이 평가한 결과와 인간 평가자의 판단 사이의 일치성을 조사했습니다. 여러 측면에서의 품질 평과의 부식적인 상관 관계(calibration)는 스피어만 순위 상관 계수(Spearman's rank correlation)가 평균 0.72로, 높은 수준은 아니지만 준수한 정렬을 보여줍니다.

- **Performance Highlights**: LLM이 평가자로 사용될 경우, 평가자 수가 2명일 때 GPT-4는 결과를 개선하는데 도움을 주었지만, 3명 이상의 인간 평가자가 있을 때는 LLM의 도움이 중립적이거나 오히려 해로울 수 있다는 결과를 보여주었습니다. 데이터셋은 공개적으로 제공되어, LLM을 활용한 평가 개선을 위한 미래의 연구를 지원합니다.



### Zero-shot LLM-guided Counterfactual Generation for Tex (https://arxiv.org/abs/2405.04793)
Comments:
          arXiv admin note: text overlap with arXiv:2309.13340

- **What's New**: 이 논문에서는 NLP 태스크의 모델 개발과 평가에 자주 사용되는 반사실적(counterfactual) 예제 생성에 대한 새로운 접근 방식을 제안합니다. 기존에 주로 활용되던 사전 훈련된 언어 모델을 사용한 반사실적 생성 방법과 달리, 	extit{제로-샷 반사실적 생성}(zero-shot counterfactual generation) 문제 설정을 탐구하며, 이를 위해 대형 언어 모델(LLMs)을 일반적인 반사실적 예제 생성기로 사용하는 구조화된 방법을 제시합니다.

- **Technical Details**: 연구팀은 	extit{제로-샷 방식}(zero-shot manner)을 통해 최신 대형 언어 모델의 지시 사항을 따르는 능력과 텍스트 이해(textual understanding) 능력을 활용하여, 별도의 훈련이나 미세조정(fine-tuning) 없이 고품질의 반사실적 예제를 생성할 수 있다고 가설을 세웠습니다. 이를 통해 모델은 다양한 NLP 하위 태스크에 걸쳐 그 효과를 검증합니다.

- **Performance Highlights**: 실험을 통해 연구팀은 다양한 NLP 하위 태스크에서 대형 언어 모델을 사용한 	extit{제로-샷 반사실적 생성기}(zero-shot counterfactual generators)로서의 효과를 입증하였습니다. 이러한 접근방식은 	extit{블랙박스 NLP 모델}(black-box NLP models)을 평가하고 설명하는 데 있어 유용함을 보여줍니다.



### CourseGPT-zh: an Educational Large Language Model Based on Knowledge Distillation Incorporating Prompt Optimization (https://arxiv.org/abs/2405.04781)
- **What's New**: CourseGPT-zh는 교육 분야에서의 특화된 요구를 충족시키기 위한 맞춤형 대형 언어 모델(Large Language Models, LLM)입니다. 공개된 LLM을 기반으로 하여 교과서 지식을 효과적으로 추출하고 다양화하는 뛰어난 질문-응답 코퍼스 정제 프레임워크를 제안합니다. 이를 통해 고전문화(NLP) 태스크에 있어 우수한 전문성을 발휘하며, 이는 교육 분야의 대형 언어 모델 개발에 새로운 방향을 제시합니다.

- **Technical Details**: CourseGPT-zh는 조기에 포착한 기술적인 방법론을 다룹니다. 교과서에서 지식을 추출하고 다양화시키기 위해 질문-응답 코퍼스 정제와 함께 프롬프트 최적화(prompt optimization)를 통합한 고도의 프레임워크를 개발했습니다. 또한, 사용자의 필요와 선호도에 부합하는 응답을 생성하기 위해 LLM-as-Judge 방식을 이용한 이산 프롬프트 최적화(discrete prompt optimization) 방법을 새롭게 도입했습니다. 이는 응답의 질을 향상시키며 응답 길이를 절약하는 데에 도움을 줍니다.

- **Performance Highlights**: 실험 결과, CourseGPT-zh는 기존에 사용 가능한 외래 소스(open-source) 대형 모델들과 비교하여 우수한 성능을 나타냅니다. 특히 전문 지식에 관한 질문-응답에서 성능이 높아, 교육 분야에 특화된 지식과 요구를 충족시키는 데 효과적입니다. 또한, 이 모델은 사용자의 반응과 선호도에 더 잘 맞춰진 응답을 제공하며, 응답의 간결성 또한 유지하면서 높은 응답 품질을 보장합니다.



### Empathy Through Multimodality in Conversational Interfaces (https://arxiv.org/abs/2405.04777)
Comments:
          7 pages, 2 figures, 2 tables, conference paper

- **What's New**: 이 논문은 멀티모달(multimodal) 능력에 기반한 새로운 대화형 건강 에이전트(Conversational Health Agents, CHAs)를 소개합니다. 특히 정신 건강 지원 분야에서, 이 CHA는 사용자의 감정 상태를 분석하고 적절한 음성 반응을 제공함으로써 기존의 텍스트 기반 분석을 넘어서는 서비스를 제공합니다.



### BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models (https://arxiv.org/abs/2405.04756)
- **What's New**: 이 연구에서는 지식 그래프(Knowledge Graph)가 통합된 새로운 방식을 이용하여 언어 모델(Language Models, LLMs)을 공격하는 방법론을 제안합니다. 자연 언어의 고정 관념을 지식 그래프로 재구성하고, 적대적 공격 기법(Adversarial Attacking Strategies)을 사용하여 다양한 오픈-소스 및 비공개 소스 언어 모델에서 편향된 반응을 유도합니다.

- **Technical Details**: 연구 팀은 자연 언어의 스테레오타입을 지식 그래프로 전환하고, 이 그래프를 이용하여 언어 모델을 공격하는 새로운 방법론을 개발했습니다. 이 기법은 언어 모델이 사회적 편견을 학습하는 것을 모방함으로써, 심지어 안전 장치가 내장된 모델에서도 편향을 증가시킬 수 있음을 발견하였습니다.

- **Performance Highlights**: 이 연구 결과는 모든 테스트된 모델에서 편견이 증가하였으며, 이는 AI 안전성(AI Safety) 연구의 필요성을 강조합니다. 특히 적대적 공간(Adversarial Space)에서의 추가 연구가 필요함을 시사합니다.



### Learning Phonotactics from Linguistic Informants (https://arxiv.org/abs/2405.04726)
- **What's New**: 새로운 대화형 언어 학습 접근 방식이 제안되었습니다. 이 방법은 언어 사용자가 제공하는 언어적 적합성 판단(linguistic acceptability judgments)을 이용하여 문법을 학습합니다. 이 모델은 데이터를 반복적으로 선택하거나 생성하고, 정보 제공자(informant)에게 이진 판단(binary judgment)을 요청한 다음, 다음 쿼리를 준비하기 위해 자체 매개 변수를 업데이트합니다.

- **Technical Details**: 이 모델은 정보 이론 정책(information-theoretic policies)을 사용하여 아이템을 선택하고, 각 쿼리 후 모델 자체의 매개변수를 업데이트합니다. 이 연구는 음운 규칙(phonotactics)이라는 언어의 소리 시퀀스에 대한 규칙을 탐구하는 데 사용되었습니다. 두 가지 실험을 통해, 하나는 전형적인 자연 언어 데이터(typologically-natural linguistic data)를 사용하고, 다른 하나는 절차적으로 생성된 언어(procedurally-generated languages)를 다룹니다.

- **Performance Highlights**: 정보 이론 정책을 사용하는 이 모델은 전적으로 감독된 접근법(fully supervised approaches)에 비해 비슷하거나 때로는 더 높은 샘플 효율성(sample efficiency)을 달성했습니다. 이는 적은 데이터로도 효과적인 학습 결과를 얻을 수 있음을 나타냅니다.



### Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking (https://arxiv.org/abs/2405.04685)
- **What's New**: 이 연구는 저자원(low-resource) 언어, 특히 터키어를 사용하는 대규모 언어 모델(Large Language Models, LLMs)의 독특한 도전과제를 다룹니다. 저자원 언어를 위한 고품질 모델의 필요성을 강조하면서, 데이터 부족, 모델 선택, 평가, 그리고 계산 제한과 같은 문제들을 분석합니다. 이 연구는 특히 터키어 데이터를 사용하여 처음부터 모델을 개발하는 방법과 영어로 사전 학습된 기존 LLMs를 터키어 이해를 위해 적응(adapting)하는 두 가지 접근 방식을 포함합니다.

- **Technical Details**: 연구는 두 가지 주요 방법론을 사용합니다: (i) 영어로 사전 학습된 기존 LLMs를 터키어로 적응시키는 것과 (ii) 터키어 사전 학습 데이터를 사용하여 모델을 처음부터 개발하는 것, 이 두 방법은 모두 터키어 지시어 조정 데이터셋(instruction-tuning dataset)에서의 감독된 미세 조정(supervised fine-tuning)으로 보완됩니다. 이는 추론 능력(reasoning capabilities)을 강화하는 것을 목표로 합니다. 또한, 다양한 추론 및 지식 기술을 평가하는 새로운 터키어 LLM 리더보드(leaderboard)의 생성을 통해 이러한 방법들의 상대적 성능을 평가합니다.

- **Performance Highlights**: 실험은 사전 학습(pretraining)과 미세 조정(fine-tuning) 동안 데이터 및 모델 스케일링을 포함하였으며, 언어 간 지식 이전의 능력을 강조하고 다른 언어로의 미세 조정 중에 발생하는 재앙적 망각(catastrophic forgetting)의 도전을 다루었습니다. 이러한 연구를 통해 저자원 언어 맥락에서 LLM 프레임워크를 발전시키는 데 필요한 상세한 가이드를 제공하며, 전 세계적으로 자연어 처리(Natural Language Processing, NLP)의 혜택을 더욱 접근 가능하게 만들고자 합니다.



### Understanding the Capabilities and Limitations of Large Language Models for Cultural Commonsens (https://arxiv.org/abs/2405.04655)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 다양한 벤치마크 평가에서 상당한 상식 이해능력을 보여주었지만, 그들의 문화 상식(cultural commonsense)에 대한 이해는 대부분 검토되지 않았다고 지적합니다. 연구진은 문화 상식 작업 맥락에서 몇 가지 최신 LLM의 능력과 한계를 종합적으로 조사했습니다.

- **Technical Details**: 연구자들은 일반적인 상식과 문화적 상식 벤치마크를 사용하여 LLM들을 분석했으며, 이 연구에서는 (1) LLM들이 다른 문화에 특화된 상식 지식을 테스트할 때 중요한 성능 차이가 발생한다는 점, (2) 문화적 맥락이 LLM의 일반 상식 능력에 영향을 미칠 수 있다는 점, 그리고 (3) LLM에 문의하는 데 사용되는 언어가 문화 관련 작업의 성능에 영향을 미칠 수 있다는 점을 발견했습니다.

- **Performance Highlights**: 이 연구는 LLM의 문화 이해에서 내재된 편향을 지적하고, 문화적으로 인식하는 언어 모델을 개발하는 데 도움이 되는 통찰력을 제공합니다.



### Language Modeling Using Tensor Trains (https://arxiv.org/abs/2405.04590)
- **What's New**: 본 논문에서는 가장 단순한 텐서 네트워크인 텐서 트레인(Tensor Train, TT)을 기반으로 한 새로운 언어 모델인 '텐서 트레인 언어 모델'(Tensor Train Language Model, TTLM)을 제안합니다. 이 모델은 RNN 변형(변형된 Recurrent Neural Networks) 중 하나로 볼 수 있으며, 텐서 곱(tensor product)을 사용하여 문장을 지수적 공간에서 표현하는 특징을 가지고 있습니다. 이러한 접근 방식은 기존의 RNN, Second-order RNNs, Recurrent Arithmetic Circuits (RACs), 그리고 Multiplicative Integration RNNs (MI-RNNs)과 관련이 있음을 입증하며, 이들과의 연결성을 명확히 했음을 강조합니다.

- **Technical Details**: TTLM은 각 단어의 텐서 곱으로 구성된 지수적 의미 공간에서 문장을 표현합니다. 문장의 확률은 두 고차원 텐서, 입력 벡터 Φ(X)과 전역 계수 𝒜 사이의 내적으로 정의되며, 조건부 확률로 분해됩니다. 이 논문에서는 TTLM의 두 가지 변형, TTLM-Large와 TTLM-Tiny를 소개하며, 이들은 Vanilla RNNs보다 우수한 성능을 보였습니다. 또한, 효율성을 높이기 위해 텐서 트레인 코어(TT cores)를 더욱 분해하고 저차원의 숨겨진 유닛을 사용하는 방법을 설명합니다.

- **Performance Highlights**: 적용된 실험에서 TTLM-Large는 WikiText-2와 PTB 데이터셋에서 Vanilla RNN 대비 낮은 perplexity를 달성했습니다(-14.3 과 -16.0), 그리고 TTLM-Tiny는 각각 -1.7과 -8.5의 perplexity 개선을 보였습니다. 이 결과는 TTLM이 실제 언어 모델링 작업에서 기존 RNN보다나은 성능을 낼 수 있음을 입증합니다.



### PoPE: Legendre Orthogonal Polynomials Based Position Encoding for Large Language Models (https://arxiv.org/abs/2405.04585)
- **What's New**: 이 연구에서는 원래 변환기에서 사용된 기본적인 절대 위치 인코딩(Absolute Positional Encoding, APE) 방법에 대한 여러 개선안을 제시합니다. 우리는 높은 차원에서 위치 인코딩을 부적절하게 표현하는 것이 주의(attention) 메커니즘의 중요 측면, 모델이 상대 위치 정보를 학습하는 능력, 그리고 모델의 수렴에 미치는 영향을 조사하고자 합니다. 그 결과, 절대 위치 인코딩(Absolute Positional Encoding, APE)과 상대 위치 인코딩(Relative Positional Encoding, RPE) 방법의 성능에 부정적인 영향을 미칠 수 있는 새로운 해결책인 직교 다항식 기반 위치 인코딩(Polynomial Based Positional Encoding, PoPE)을 도입하였습니다.

- **Technical Details**: PoPE 방법은 직교 르장드르(Legendre) 다항식을 사용하여 위치 정보를 인코딩합니다. 이 다항식은 비주기성(non-periodicity), 직교성(orthogonality), 다항식의 서로 다른 기능 형태와 같은 여러 바람직한 특성을 제공합니다. 이 연구의 실증적 결과는 Multi30k 영어에서 독일어 번역 작업에서 PoPE를 채택한 트랜스포머(transformer) 모델이 기존 트랜스포머 모델보다 우수한 성능을 보임을 보여줍니다.

- **Performance Highlights**: PoPE를 채택한 트랜스포머 모델은 기존의 절대 및 상대 위치 인코딩 방법(APE 및 RPE)을 사용하는 모델보다 Multi30k 작업에서 더 빠른 수렴율과 더 높은 번역 품질을 달성함으로써 새로운 성능 기준을 설정합니다. 이는 PoPE가 기존 위치 인코딩 방법에 비해 우수한 이론적 및 실질적 이점을 제공함을 시사합니다.



### Air Gap: Protecting Privacy-Conscious Conversational Agents (https://arxiv.org/abs/2405.05175)
- **What's New**: 최근 LLM(Large Language Model) 기반 대화형 에이전트를 이용해 민감한 사용자 데이터를 관리하는데 대한 프라이버시 우려가 증가하고 있습니다. 이에 대응하여, 연구진은 맥락적 무결성(contextual integrity) 프레임워크를 기반으로 데이터 유출 방지를 위한 새로운 에이전트인 'AirGapAgent'를 개발하였습니다. 이 에이전트는 특정 작업에 필요한 데이터에만 접근을 제한함으로써 민감한 정보의 무분별한 노출을 방지합니다.

- **Technical Details**: AirGapAgent는 적대적 제3자 앱이 상호작용의 문맥을 조작하여 LLM 기반 에이전트로 하여금 작업과 무관한 개인 정보를 유출하도록 유도하는 새로운 위협 모델에 대응하기 위해 고안되었습니다. 실험에서는 Gemini, GPT, Mistral과 같은 다양한 모델을 사용하여 AirGapAgent의 효율성을 검증하였습니다. 이를 통해 문맥 하이재킹(context hijacking)이라는 형태의 위협을 완화하는 동시에 에이전트의 핵심 기능을 유지할 수 있음을 입증하였습니다.

- **Performance Highlights**: 실험 결과, Gemini Ultra 에이전트에 단일 쿼리를 이용한 문맥 하이재킹 공격이 일어났을 때 사용자 데이터 보호 능력이 94%에서 45%로 급감하는 반면, AirGapAgent는 97%라는 높은 보호 능력을 유지하여 동일한 공격을 효과적으로 무력화하였습니다. 이는 AirGapAgent가 LLM 기반 에이전트를 사용하는 환경에서 중요한 보안 강화 도구로서의 가능성을 보여줍니다.



### Integrating LSTM and BERT for Long-Sequence Data Analysis in Intelligent Tutoring Systems (https://arxiv.org/abs/2405.05136)
- **What's New**: 이번 연구에서는 지식 추적(Knowledge Tracing)의 분야에서 학생들의 학습과 지식 습득을 이해하기 위해 긴 시퀀스(long-sequence) 데이터를 처리할 수 있는 새로운 모델인 LSTM BERT 기반의 지식 추적 모델(LBKT)을 제안합니다. 이 모델은 특히 인텔리전트 튜터링 시스템(Intelligent Tutoring Systems)에서 발생하는 대규모 데이터셋과 긴 데이터 시퀀스를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: LBKT는 BERT 기반 아키텍처를 사용하고, 여러 난이도의 정보를 처리하기 위한 라쉬 모델(Rasch model) 기반의 임베딩 블록과, 학생들의 연속적인 행동을 처리하기 위한 LSTM 블록을 포함합니다. LBKT는 ACC 및 AUC 메트릭에서 대부분의 벤치마크 데이터셋에서 가장 우수한 성능을 보였으며, 각 구성 요소의 전반적 성능에 미치는 영향을 분석하기 위한 소거연구(ablation study)도 수행되었습니다.

- **Performance Highlights**: LBKT는 전통적인 딥러닝 기반의 지식 추적 방법들보다 빠르고, 해석 가능성(interpretability)이 높으며 메모리 비용도 낮은 것으로 나타났습니다. 또한, t-SNE를 사용한 시각화 도구를 통해 모델의 임베딩 전략을 보여주었으며, 이는 LBKT가 효과적으로 학습과정을 표현할 수 있음을 시사합니다.



### Lessons from the Use of Natural Language Inference (NLI) in Requirements Engineering Tasks (https://arxiv.org/abs/2405.05135)
- **What's New**: 본 연구는 자연어 추론(Natural Language Inference, NLI)을 이용하여 요구 사항 공학(Requirements Engineering) 작업을 자동화하는 방법을 탐구합니다. 특히, 요구 사항 분류, 요구 사항 사양 결함 식별, 이해 관계자 요구 사항의 충돌 감지 세 가지 작업에 초점을 맞추고 있습니다. 이전 연구에서는 NLI를 사용하여 다양한 자연어 처리(Natural Language Processing, NLP) 작업에서 상당한 이점을 보여주었지만, 소프트웨어 요구 사항 공학의 맥락에서는 그러한 이점이 충분히 조사되지 않았습니다.

- **Technical Details**: 본 연구에서는 NLI를 요구 사항 분석에 사용하기 위한 실험을 설계하고, 대화형 모델(prompt-based models), 기존의 전이 학습(transfer learning), 대형 언어 모델(Large Language Models, LLMs) 기반 챗봇 모델, 확률 모델(probabilistic models)과 같은 여러 접근 방법과 NLI의 성능을 비교하였습니다. 전통적 학습과 제로샷(zero-shot) 학습 설정을 포함한 다양한 학습 설정에서 실험을 수행하여, 요구 사항 사양 분석에서 NLI 방법이 고전적인 NLP 방법과 다른 LLMs 기반 및 챗봇 모델보다 우수함을 확실히 입증했습니다.

- **Performance Highlights**: 실험 결과, NLI 방식은 고전적인 NLP 방법뿐만 아니라 다른 LLMs 기반 및 챗봇 모델들과 비교했을 때, 요구 사항 사양 분석 작업에서 더 뛰어난 성과를 보였습니다. 이러한 결과는 NLI가 요구 사항 공학 작업 자동화에 적합한 접근 방식임을 뒷받침합니다.



### VisionGraph: Leveraging Large Multimodal Models for Graph Theory Problems in Visual Contex (https://arxiv.org/abs/2405.04950)
Comments:
          17 pages; Accepted by ICML 2024

- **What's New**: 새롭게 소개된 VisionGraph 벤치마크는 다중모달 그래프 이론 문제(multimodal graph theory problems)에서 대규모 다중모달 모델(Large Multimodal Models, LMMs)의 능력을 탐구하도록 설계되었습니다. 이 벤치마크는 연결성에서 가장 짧은 경로 문제에 이르기까지 여덟 가지 복잡한 그래프 문제 작업을 포함합니다.

- **Technical Details**: 연구팀은 설명-프로그래밍-추론(Description-Program-Reasoning, DPR) 체인을 제시하여 그래픽 구조 설명 생성 및 알고리즘 인식 다단계 추론을 통해 추론 과정의 논리적 정확성을 향상시킵니다. 이 방법은 LMMs가 시각적 그래프에서 정확하게 구조를 이해하고 다단계 추론을 수행하는 데 도움을 줍니다.

- **Performance Highlights**: GPT-4V는 다단계 그래프 추론에서 Gemini Pro를 능가했으며, DPR은 LMM의 다단계 그래프 추론 능력을 상당히 향상시켰습니다. 특히 GPT-4V (DPR) 에이전트는 상태 최고의 성능(State Of The Art, SOTA)을 달성했습니다. 모든 LMM은 제로샷(zero-shot)/퓨샷(few-shot) 설정 또는 지도된 미세 조정(Supervised Fine-Tuning, SFT)에서 그래픽 구조 인식이 부족하며, 이는 문제 해결 성능에 영향을 미칩니다.



### Honeyfile Camouflage: Hiding Fake Files in Plain Sigh (https://arxiv.org/abs/2405.04758)
Comments:
          3rd Workshop on the security implications of Deepfakes and Cheapfakes (WDC) co-located at ACM ASIACCS 2024

- **What's New**: 이 논문은 허니파일(honeyfiles)이라고 불리는 유형의 허니팟(honeypot)의 이름을 분석하고 개선하는 방법을 고찰합니다. 허니파일은 가짜 파일로, 악의적 행위로부터 정보를 추론하고 탐지하는 데 사용됩니다. 연구팀은 실제 파일 시스템에 허니파일을 자연스럽게 배치하기 위해 파일 이름을 위장하는 두 가지 척도를 개발했습니다.

- **Technical Details**: 개발된 방법은 코사인 거리(cosine distances)를 활용하여 의미론적 벡터 공간(semantic vector spaces)에서 파일 이름의 유사성을 측정합니다. 첫 번째 방법은 간단한 평균화를 통해, 두 번째 방법은 혼합 피팅(mixture fitting)과 클러스터링을 통해 이름을 위장합니다. 이 두 척도는 GitHub 소프트웨어 저장소 데이터셋을 이용해 평가 및 비교되었습니다.

- **Performance Highlights**: 두 메트릭(metric) 모두 해당 데이터셋에서 우수한 성능을 보였습니다. 이는 허니파일의 이름을 보다 효과적으로 위장할 수 있는 방법론을 제시함으로써, 실제 파일 시스템에서의 허니파일 배치의 자연스러움을 개선할 수 있는 가능성을 보여줍니다.



### Folded context condensation in Path Integral formalism for infinite context transformers (https://arxiv.org/abs/2405.04620)
Comments:
          7 pages, 2 figures

- **What's New**: 이 연구에서는 트랜스포머(Transformer)의 주의(attention) 알고리즘을 경로 적분(Path integral formalism)을 사용하여 일반화하고 재해석했다는 것이 새롭습니다. 이 노트는 트랜스포머의 역할을 토큰 상태의 시간 발전으로 이해하며, 모든 키-토큰 상태가 쿼리 토큰 상태와 주의를 기울일 수 있다고 제안합니다.

- **Technical Details**: 이 논문은 트랜스포머의 기존 주의 메커니즘을 경로 적분(Path integral) 관점에서 다시 해석합니다. 이 접근 방식은 입출력 토큰 상태의 시간 전이(Temporal transitions)를 중점적으로 다룹니다. 또한, 저자들은 선형 주의 메커니즘(Linear attention mechanism)을 기반으로 하여 컨텍스트 정보(Contexual information)를 잘 포착하는 새로운 방법인 'Infinite-attention'을 제안합니다. 이 통한 장기 컨텍스트(Long context)를 유지하면서 메모리 사용량을 줄일 수 있는 방법이 될 수 있습니다.

- **Performance Highlights**: 실험을 통해 입력 토큰 창 크기 12를 사용하고 24GB 메모리를 가진 한 개의 GPU로 사전 훈련을 수행했습니다. 이 방법으로 150개 이상의 길이의 컨텍스트를 보존할 수 있음이 확인되었습니다. 이 연구는 기존 GPT 코드를 사용하여 같은 코퍼스(Corpus)와 같은 GPU 장비를 사용했음에도 불구하고 '문법적 원거리 공기관계(Grammatical remote co-occurrence)'를 달성했다는 중요한 성과를 달성했습니다.




New uploads on arXiv(cs.CL)

### VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing (https://arxiv.org/abs/2509.22651)
- **What's New**: 음성 중심의 AI 보조 도구에 대한 관심이 높아짐에 따라 VoiceAssistant-Eval이라는 포괄적인 벤치마크가 도입되었습니다. 이 벤치마크는 청취(listening), 발화(speaking), 시청(viewing) 능력을 평가하기 위해 13개 작업 카테고리에서 10,497개의 정제된 예시로 구성되어 있습니다. 다양한 작업에는 자연 소리, 음악 및 대화가 포함됩니다.

- **Technical Details**: VoiceAssistant-Eval은 개인화된 음성 모방, 자연스러운 핸즈프리 음성 상호작용, 다중 모드 비전-오디오 이해 및 고급 오디오 QA와 같은 4가지 대표 작업을 평가합니다. 비교를 통해 기존의 벤치마크는 특정 측면만을 커버하고 있는 반면, VoiceAssistant-Eval은 AI 보조 도구의 전체 범위를 종합적으로 테스트합니다. 또한, 역할별 말하기 스타일과 음색을 분석하여 개인화 상호작용의 잠재력을 입증합니다.

- **Performance Highlights**: 21개의 오픈 소스 모델과 GPT-4o-Audio 모델을 평가한 결과, 독점 모델이 항상 오픈 소스 모델보다 우수하지 않음을 보여주었습니다. 대부분의 모델이 발화 작업에서 뛰어난 성능을 보였으나 오디오 이해에서는 뒤처지는 경향을 나타냈습니다. 특히, 중간 크기의 Step-Audio-2-mini(7B)가 LLaMA-Omni2-32B-Bilingual보다 두 배 이상의 청취 정확도를 기록한 것이 주목할 만한 성과입니다.



### WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning (https://arxiv.org/abs/2509.22644)
- **What's New**: 최근의 연구들은 코드 에이전트가 GitHub 이슈 수정 및 새로운 기능 구현과 같은 코드 생성 작업에서 큰 발전을 보여주었음을 강조하고 있습니다. 그러나 웹사이트 코드 생성과 같은 작업에서는 시각적 미학 및 사용자 상호작용의 유창성에 의존하게 되어 현재의 코드 에이전트 시스템이 생성된 코드베이스의 실제 품질을 완전히 포착하지 못하는 문제가 발생하고 있습니다. 이 논문에서는 통합적이고 다단계 시각적 피드백을 활용하여 웹사이트 코드를 반복적으로 생성하고 개선하는 웹사이트 생성 에이전트인 WebGen-Agent를 제안합니다.

- **Technical Details**: WebGen-Agent는 자연어 지침을 기반으로 웹사이트를 생성하며 코드 생성, 코드 실행, 피드백 수집의 세 가지 동작을 포함하는 반복적이고 다단계의 패러다임을 채택하고 있습니다. 코드 실행 후 생성된 웹사이트의 스크린샷을 캡처하고, 흐름 언어 모델(VLM)을 사용하여 시각적 매력과 품질을 평가합니다. 이 피드백을 통해 웹사이트의 디자인과 인터랙티브 기능을 지속적으로 개선할 수 있습니다.

- **Performance Highlights**: WebGen-Agent는 WebGen-Bench 데이터셋에서 Claude-3.5-Sonnet의 정확도를 26.4%에서 51.9%로 증가시키고 외관 점수를 3.0에서 3.9로 올리며 이전의 최고 성능을 초과했습니다. Step-GRPO 훈련 방법은 Qwen2.5-Coder-7B-Instruct의 정확도를 38.9%에서 45.4%로 개선하고 외관 점수를 3.4에서 3.7로 증가시켜 생성된 웹사이트의 기능성과 외관을 모두 크게 향상시킵니다.



### Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity (https://arxiv.org/abs/2509.22641)
Comments:
          26 pages, 10 figures, under review

- **What's New**: 이 연구는 언어 모델의 텍스트 생성 능력을 평가하기 위해 n-그램(n-gram) 신규성(로비티, novelty)을 사용하는 데 한계를 지적합니다. 창의성의 두 가지 성격인 신규성과 적합성(pragmaticality)을 함께 고려해야 한다고 주장하며, 이를 통해 n-그램 신규성과 창의성의 관계를 분석합니다. 7542개의 전문가 작가 주석을 통해 n-그램 신규성이 있는 텍스트의 대부분이 창의적이지 않다고 판단된다는 중요한 발견을 보고합니다.

- **Technical Details**: 연구에서는 창의성을 인간 평가에 기반하여 운영화하고, 적합성을 의미론적(sensicality)과 맥락적(pragmaticality)으로 분해합니다. LLM(대형 언어 모델)에서 생성된 텍스트는 지나치게 혁신적(novel)일 수 있으나 그 맥락에서 이해되지 않으면 창의적으로 평가받지 못할 수 있습니다. 또한, 전문가의 평가를 통해 n-그램 신규성이 높은 텍스트의 91%가 창의적이지 않다고 판별되었음을 밝혔다.

- **Performance Highlights**: 최신 LLM들이 창의적인 표현을 생성하는 데 어려움을 겪고 있으며, 이는 LLM을 이용한 평가보다 전문가의 판별이 더욱 적합하다는 것을 의미합니다. 연구 결과는 LLM으로 생성된 텍스트의 창의성 평가에서 상대적으로 높은 성과를 보여주지만 비적합적인 표현(non-pragmatic expressions)을 식별하는 데에는 한계가 있음을 드러냅니다. 가장 성과가 좋았던 모델의 신규성 점수는 전문가의 선호도와 강한 상관관계를 보였습니다.



### Language Models Can Learn from Verbal Feedback Without Scalar Rewards (https://arxiv.org/abs/2509.22638)
- **What's New**: 이 논문은 LLMs(대형 언어 모델) 훈련에서 일반적으로 사용되는 인간 또는 AI 피드백으로부터의 강화 학습(RL) 방식을 재정의합니다. 구체적으로, 언어 피드백을 조건부 신호로 간주하여, 피드백 조건부 정책(FCP)을 도입합니다. FCP는 응답-피드백 쌍으로부터 직접 학습하고, 최적화된 피드백을 통해 모델을 개선할 수 있는 구조를 제공합니다.

- **Technical Details**: FCP는 πθ(𝒐|𝒙,𝒄)로 표현되며, 주어진 지침 𝒙에 대해 응답 𝒐를 생성하는 참조 정책 πref(𝒐|𝒙)와 환경 피드백 분포 penv(𝒄|𝒙,𝒐)을 결합합니다. 이 정책은 긍정적인 피드백 조건에서 훈련되어 피드백과 응답 간의 관계를 강화합니다. FCP는 또한 온라인 부트스트래핑 단계를 통해 지속적으로 피드백을 업데이트하며 모델 성능을 향상시킵니다.

- **Performance Highlights**: 파일럿 실험 결과, FCP는 오프라인 RFT 및 온라인 GRPO와 같은 기존의 강력한 스칼라 기반 기준선을 초과하는 성능을 보였습니다. 이 방법은 검증기나 스칼라 변환 없이도 풍부한 언어 피드백을 보존할 수 있는 단순하고 확장 가능한 프레임워크를 제공합니다. 향후 발전된 훈련 기술을 적용하면 FCP의 성능을 더욱 개선할 수 있을 것으로 기대됩니다.



### Variational Reasoning for Language Models (https://arxiv.org/abs/2509.22637)
- **What's New**: 이번 연구는 언어 모델을 위한 변별적 사고 프레임워크를 소개하며, 사고 흔적(thinking traces)을 잠재 변수(latent variables)로 취급하여 변별 추론(variational inference)을 통해 최적화하는 방법을 제안합니다. 기존의 ELBO(evidence lower bound)를 다중 흔적(multi-trace) 목표로 확장하고, 훈련을 안정화하는 forward-KL 형태를 소개합니다. 이 방법론은 기존의 지도 학습(Supervised Finetuning) 및 강화 학습(Reinforcement Learning) 방법론과 비교했을 때, 사전 훈련된 모델의 사회적 편향(bias)를 더 명확히 이해할 수 있도록 돕습니다.

- **Technical Details**: 변별적 사고 프레임워크에서는 사고 흔적을 생성하고 최종 답변을 도출하기 위해 모델 πθ(𝒛,𝒚|𝒙)를 사용합니다. 이는 사고 과정(𝒛)과 응답(𝒚)으로 구성되며 연합확률(joint probability)은 두 조건부 확률로 나눌 수 있습니다. 본 논문에서는 ELBO를 통해 사고 흔적에 대한 마진 분포(marginal distribution)를 최적화하는 비용 함수를 제안하고, IWAE 스타일의 다중 흔적(extension)을 통해 목표를 강화하여 훈련 안정성을 높입니다.

- **Performance Highlights**: 제안된 프레임워크는 Qwen2.5와 Qwen3 모델 가족을 통해 다양한 추론 벤치마크에서 강력한 기준선을 넘어서는 일관된 개선 효과를 보였습니다. 연구는 MATH500, AIME24&25, OlympiadBench 등 여러 이유 문제에서 성능을 검증하였으며, 변별적 사고 접근방식이 기존 방법들과 비교해 확실한 이점을 제공함을 입증하였습니다. 또한, 중도 폐쇄(drop-out)를 방지하며 더 나은 답변 힌트를 활용하도록 훈련 파이프라인을 조정하는 방식으로 시스템의 전반적인 추론 역량을 향상시킵니다.



### StateX: Enhancing RNN Recall via Post-training State Expansion (https://arxiv.org/abs/2509.22630)
- **What's New**: 이 논문에서는 RNN의 상태 크기를 포스트 트레이닝(post-training)을 통해 효율적으로 확장하는 StateX 훈련 파이프라인을 소개합니다. 이는 학습 비용을 최소화하고 새로운 매개변수 추가를 거의 없이 이루어질 수 있습니다. 또한, 긴 문맥을 처리하는 모델의 경우 더 큰 재귀적 상태가 중요하다는 점을 반영하여 상태 확장을 사전적으로 수행합니다.

- **Technical Details**: StateX는 선형 주의(linear attention)와 상태 공간 모델(state space models) 두 가지 인기 있는 RNN 클래스에 대해 상태를 확장하기 위한 아키텍처 수정을 설계했습니다. 포스트 트레이닝에 필요한 데이터 양이 기존의 프리 트레이닝(pre-training)보다 적고, 모델 성능과 적응 효율성을 균형 있게 유지하기 위해 핵심 레이어만 선택하여 확장합니다. 기존의 복잡한 방법들과 비교하여 더 간단하고 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: StateX를 통해 RNN의 기억 및 문맥 학습 능력이 크게 향상되었음을 실험 결과로 입증했습니다. GLA 모델에서 기억 중심 작업의 상대적인 정확도가 3.36%, 문맥 학습에서는 7.2% 증가하였으며, Mamba2 모델에서도 각각 1.1%와 1.0% 향상되었습니다. 평균 NIAH 정확도는 GLA에서 26.0%에서 42.2%, Mamba2에서 33.2%에서 39.2%로 개선되었습니다.



### From tests to effect sizes: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation benchmarks (https://arxiv.org/abs/2509.22612)
Comments:
          Paper currently under review at ACL Rolling Review

- **What's New**:  이번 논문에서는 다언어 및 다작업 NLP 벤치마크에서 평가 메트릭의 불확실성과 통계적 정확도를 정량화하기 위한 재표집(resampling) 기반 방법을 소개합니다. 우리는 성능 점수의 실험적 변동이 모델 관련 및 데이터 관련 소스 모두에서 발생한다는 점을 강조하며, 이 두 가지를 함께 고려하는 것이 전체적인 변동성을 과소 평가하는 것을 방지하는 데 필수적임을 설명합니다. 우리는 다언어 질문 답변, 기계 번역, 명명된 개체 인식(Named Entity Recognition) 등의 사례를 통해 재표집 방법이 평균/중앙값, 모델 간의 쌍 별 차이, 순위와 같은 다양한 양을 계산하는 데 유용하다는 것을 보여줍니다.

- **Technical Details**: 우리는 실험적 분산의 소스를 데이터 측과 모델 측으로 나누고, 데이터 측의 변동성이 원본 테스트 세트의 약간 다른 버전으로 재표집할 때 발생하는 샘플링 오류와 어떻게 연결되는지를 분석합니다. 또한, 모델 측 변동성은 LLM이 디코딩 시 생성 작업(예: 질문 답변)에서 랜덤 샘플링을 포함할 때 발생하는 불확실성으로 나타나며, 이는 매번 다르게 응답을 반환할 수 있습니다. 이를 기반으로 하여, 우리는 재표집과 언어 간/내 존재적인 변동성을 추정하여 "전체 성능"에 대한 불확실성 인식 추정을 도출하는 방법들을 제안합니다.

- **Performance Highlights**: 우리는 모델 측과 데이터 측의 변동원천을 모두 고려해야만 보다 정확한 성능 추정을 할 수 있으며, 이를 통해 모델들 간의 상대 순위를 보다 명확하게 평가할 수 있음을 시연합니다. 우리의 방법론은  수초에서 수 분 내에 실행 가능하고 계산 복잡성에서 주요 병목 현상이 없습니다. 논문과 함께 제공된 툴킷은 www.github.com/j0ma/reuben에서 자유롭게 사용할 수 있으며, 이는 NLP 평가를 위한 불확실성 경계를 위한 재표집 기반 방법을 구현한 것입니다.



### Capturing Opinion Shifts in Deliberative Discourse through Frequency-based Quantum deep learning methods (https://arxiv.org/abs/2509.22603)
Comments:
          9 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 다양한 견해를 고려하고 결정을 내리기 전 다각적으로 분석하는 데 있어 심의(deliberation)의 중요성을 강조하고 있습니다. 최근 자연어 처리(Natural Language Processing) 기술의 발전으로, 의견 변화(opinion shifts)와 다양한 시나리오에서의 잠재적 결과를 예측하는 계산 모델링이 가능해졌습니다. 다수의 NLP 기법을 비교 분석하여 심의 담론(discourse)을 효과적으로 해석하고 의미 있는 통찰(insight)을 생성하는 모델의 성능을 평가합니다.

- **Technical Details**: 제안된 다중 모달 융합 프레임워크는 설문 응답과 PowerPoint 슬라이드 내용을 텍스트 형태로 처리하여 향상된 질문-응답 성능을 제공합니다. 이 아키텍처는 임베딩 및 기능 추출 레이어를 통해 질문에 특정한 임베딩을 생성하며, SBERT 인코더(all-MiniLM-L6-v2)와 PPT 전용 브랜치를 갖추고 있습니다. 중앙 융합 레이어는 FFFT 처리, 크기 압축 및 크로스 모달 공동 선택(cross-modal co-selection)을 통해 향상된 표현을 생성합니다.

- **Performance Highlights**: 제안된 멀티모달 융합 접근 방식은 훈련 손실(training loss) 수렴을 개선하고 PPT 임베딩 품질을 향상시키는 실험 결과를 보여줍니다. 모델의 성능은 여러 주요 지표에서 개선되었으며, 각 질문에 대한 표현 생성을 위한 다층분류헤드(multi-task classification heads)와 4층, 4헤드 트랜스포머 인코더가 활용됩니다. 이 연구는 공공 정책 결정, 토론 평가 및 대규모 소셜 미디어 의견 채굴 등에서의 실용적인 적용 가능성을 강조합니다.



### From Formal Language Theory to Statistical Learning: Finite Observability of Subregular Languages (https://arxiv.org/abs/2509.22598)
Comments:
          12 pages, 5 figures

- **What's New**: 본 연구에서는 모든 표준 하위 정규 언어 클래스가 결정하는 술어를 통해 선형적으로 분리 가능함을 증명합니다. 이는 유한 관찰 가능성을 설정하고 간단한 선형 모델을 통해 학습 가능성을 보장합니다. 합성 실험에서는 잡음 없는 조건에서 완벽한 분리를 확인했으며, 실제 데이터 실험에서는 학습된 특징이 잘 알려진 언어적 제약과 일치함을 보여주었습니다.

- **Technical Details**: 연구에서는 유한 관찰 가능성(finite observability)을 정의하고 모든 표준 하위 정규 클래스(SL, SP, LT, PT, LTT, TSL)가 유한하게 관찰 가능하다는 것을 증명합니다. 이는 적절한 삽입을 통해 선형 분리에 이르게 하며, 이는 수학적 언어 이론(statistical learning theory)과 언어적 응용 분야를 연결하는 새로운 기하학적 특성을 제공합니다. 문자열을 원시 술어(primitives)로 정의된 유한 차원 부울 공간(Boolean spaces)에 삽입함으로써 다양한 하위 정규 클래스를 하나의 프레임워크 내로 통합합니다.

- **Performance Highlights**: 실험 결과, 잡음 없는 조건에서 완벽한 선형 분리가 이루어짐을 확인했습니다. 또한, 실제 데이터 실험에서는 학습된 모델이 기존의 언어적 제약에 잘 부합함을 보여주었습니다. 이러한 연구 결과는 자연 언어 구조를 모델링하는 데 있어 하위 정규 계층(subregular hierarchy)이 엄밀하고 해석 가능(interpretable)한 기초를 제공함을 시사합니다.



### ArabJobs: A Multinational Corpus of Arabic Job Ads (https://arxiv.org/abs/2509.22589)
- **What's New**: ArabJobs는 이집트, 요르단, 사우디 아라비아 및 아랍 에미리트에서 수집된 아랍어 구인 광고의 공공 코퍼스입니다. 8,500개 이상의 게시물과 550,000개 이상의 단어로 구성되어 있으며, 아랍 노동 시장 내에서의 언어적, 지역적 및 사회경제적 변화를 포착합니다. 이 데이터셋은 성별 표현 및 직업 구조에 대한 분석을 포함하며, 광고 간의 방언적 변화도 강조하여 향후 연구 기회를 제공합니다.

- **Technical Details**: 이 논문에서는 대규모 언어 모델을 사용하여 연봉 추정(salary estimation) 및 직종 분류(job category normalization)와 같은 응용 프로그램을 시연합니다. 또한 성별 편향 탐지(gender bias detection) 및 직업 분류(profession classification)와 같은 벤치마크 작업을 소개합니다. ArabJobs는 공정성을 고려한 아랍어 NLP 및 노동 시장 연구에 유용한 데이터셋으로, GitHub에서 공개될 예정입니다.

- **Performance Highlights**: 연구 결과는 ArabJobs가 아랍어 NLP에 있어 공정성을 고려한 연구의 유용성이 크다는 것을 보여줍니다. 데이터셋은 사회 경제적 현상에 대한 다양한 분석을 가능하게 하며, 성별 및 직업 구조의 데이터를 통해 더 나은 인사이트를 제공합니다.



### Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs (https://arxiv.org/abs/2509.22582)
- **What's New**: 이 연구에서는 모델의 출력에 출처 텍스트와 일치하지 않는 정보가 포함된 경우인 context-grounded hallucinations를 조사하고, 이러한 환각을 찾아내기 위한 LLM(large language model)의 적용 가능성을 탐구합니다. 기존의 복잡한 평가 파이프라인 대신, LLM에 맞춤화된 새로운 메타 평가 벤치마크를 구축하였으며, 1,000개 이상의 사례에 대한 인간 주석이 포함되어 있습니다.

- **Technical Details**: 우리는 환각의 새로운 표현 방식을 제안하며, 이는 자유 형식의 텍스트 설명을 기반으로 하여 다양한 오류를 포착할 수 있습니다. 이 연구는 벤치마크의 난이도를 평가하기 위해 네 개의 대규모 LLM을 종합적으로 분석하였고, 최상의 모델이 F1 점수 0.67을 기록하는 것으로 그 어려움을 강조합니다.

- **Performance Highlights**: 세심한 분석을 통해 최적의 프롬프트 전략에 대한 통찰을 제공하며, LLM이 직면하는 주요 난관을 두 가지로 요약합니다. 첫째, 출력값에서 사실만 확인하라는 지침에도 불구하고 누락된 세부 정보를 일관성이 없다고 잘못 표시하는 경향이 있으며, 둘째, 출처에서 확인할 수 없는 사실적으로 올바른 정보가 포함된 출력 처리에서 어려움을 겪습니다.



### Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation (https://arxiv.org/abs/2509.22565)
- **What's New**: 이번 연구에서는 EHR(전자 건강 기록) 포털을 통한 비동기 환자-임상 의사 메시징의 증가로 인해 임상 의사의 업무 부담이 커지고 있음을 언급하며, 대형 언어 모델(LLMs)을 활용한 초안 응답 보조의 필요성을 강조합니다. 연구는 임상적으로 기반한 오류 온톨로지를 도입하고, 검색 보강 평가 파이프라인(RAEC)을 개발하며, DSPy를 사용한 두 단계의 프롬프트 구조를 제공하는 세 가지 주요 기여로 구성됩니다.

- **Technical Details**: 연구에서 소개된 오류 온톨로지는 5개의 도메인과 59개의 세부 오류 코드로 구성되어 있으며, 이는 귀납적 코딩과 전문가의 판단을 통해 개발되었습니다. RAEC는 의미적으로 유사한 과거 메시지-응답 쌍을 활용하여 평가 품질을 향상시키는 방식으로 작동하며, 두 단계의 DSPy 파이프라인을 사용하여 질적인 검토를 수행합니다.

- **Performance Highlights**: 1,500개 이상의 환자 메시지를 대상으로 한 평가에서, 검색된 맥락이 임상적 완전성과 업무 적합성 등의 영역에서 오류 식별을 개선하는 데 기여한 것으로 나타났습니다. 100개의 메시지에 대한 인간 검증 결과, 맥락이 향상된 레이블의 성능이 기준선과 비교하여 더 높은 일치도(50% vs. 33%)와 성능(F1 = 0.500 vs. 0.256)을 보이며 RAEC 파이프라인의 유용성을 지지하고 있습니다.



### Think Socially via Cognitive Reasoning (https://arxiv.org/abs/2509.22546)
Comments:
          Repository: this https URL

- **What's New**: 이번 논문에서는 사회적 인지 기능을 LLM에 통합하기 위해 인지 추론(Cognitive Reasoning)이라는 새로운 패러다임을 제안합니다. 이는 사회적 상황에서의 해석적 과정을 구조화된 인지 흐름(cognitive flow)으로 공식화하여, 효과적인 사회적 사고와 반응을 이끌어내도록 합니다. 또한 CogFlow라는 훈련 프레임워크를 통해 이러한 능력을 LLM에 instill(주입)할 수 있는 방법을 제시합니다.

- **Technical Details**: CogFlow는 먼저 인지 흐름 데이터셋을 구성하여 인간의 사고 과정의 연관성과 점진적인 본질을 모방합니다. 이 과정은 인지 단위를 생성하고, 이를 이용하여 사회적 상황에 대한 인지 흐름을 형성하는 데 있어 트리 구조의 계획(tree-structured planning)을 사용합니다. 이후 감독된 미세 조정(supervised fine-tuning)을 통해 기본적인 인지 추론 능력을 주입한 뒤, 강화 학습(reinforcement learning)을 활용하여 모델이 스스로 더 나은 추론 경로를 탐색할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험 결과, CogFlow는 LLM의 사회적 인지 능력을 유의미하게 향상시킬 수 있음을 보여줍니다. 이로 인해 LLM 뿐만 아니라 인간의 사회적 의사결정 과정에서도 더욱 효과적인 결과를 도출할 수 있습니다. 특히, CogFlow는 인지 흐름과 반응 품질을 최적화하는 다목적 보상 기제를 통해 모델의 성능을 극대화합니다.



### InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models (https://arxiv.org/abs/2509.22536)
- **What's New**: 본 연구에서는 FP8 훈련을 위한 포괄적인 오픈소스 훈련 레시피를 도입하였습니다. 이 레시피는 지속적인 사전 훈련과 감독 학습의 미세 조정을 원활하게 통합하며, 수치적 정확도를 유지하면서도 계산 효율성을 극대화하는 하이브리드 그레뉼러티(Granularity) 양자화 전략을 사용합니다. 이를 통해 FP8 훈련이 기존의 BF16 기준 성능에 상응하면서도 훈련 시간을 최대 22% 단축하고 메모리 사용을 14% 줄이는 등 현저한 효율 개선을 이루었습니다.

- **Technical Details**: FP8 훈련은 NVIDIA의 Hoppe 아키텍처에 의해 지원되며, 슬롯당 블록 기반 양자화와 높은 정확도를 위한 토큰 기반 양자화가 결합된 하이브리드 방식으로 수행됩니다. 이 방식은 모델의 가중치에 블록 기반 양자화를 적용하고, 보다 동적 범위를 보이는 활성화에 대해서는 고정밀의 양자화를 사용하여 훈련 중 안정성을 유지합니다. 결과적으로 FP8 훈련은 BF16과 유사한 성능을 달성하면서도 더 빠르고 효율적인 학습이 가능하다는 것을 보여주었습니다.

- **Performance Highlights**: FP8 훈련의 결과, 총 훈련 시간이 최대 22% 단축되고, 피크 메모리 소모는 14% 절감되었습니다. 또한, 계산 처리량은 최대 19% 증가하였으며, 훈련과 검증 손실 곡선은 BF16의 거의 동일한 양상을 보이며 상당한 안정성을 나타냈습니다. FP8을 통해 대규모 모델 훈련의 접근성과 지속 가능성을 높이는데 기여할 수 있는 강력한 대안으로 자리잡았습니다.



### We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong (https://arxiv.org/abs/2509.22510)
- **What's New**: 본 논문에서는 여러 가지 목표(목표: helpfulness, harmlessness, honesty; HHH)에 따라 대형 언어 모델(LLM)을 정렬하기 위한 Adaptive Multi-Branch Steering(AMBS)라는 새로운 프레임워크를 제안합니다. 기존의 접근법이 1-to-1 Transformer 디코더에 기반한 독립적인 최적화로 인한 재앙적 망각 문제를 겪는 반면, AMBS는 1-to-N 구조를 통해 멀티 브랜치 정렬을 수행하면서 목표 간 일관성을 유지합니다.

- **Technical Details**: AMBS는 두 단계로 구성된 1-to-N 프레임워크로, Stage I에서는 변환기 계층의 주의 후(hidden states)를 한 번 계산하여 공유 표현을 형성합니다. Stage II에서는 이 공유 표현을 병렬 브랜치로 복제하고 정책-참조 메커니즘을 통해 조정하여 목적별 제어를 가능하게 하면서 교차 목적 일관성을 유지합니다. 이는 각 브랜치가 HHH 목표에 맞게 출력을 생성하도록 합니다.

- **Performance Highlights**: 실험 결과, AMBS는 Alpaca, BeaverTails, TruthfulQA에서 LLM의 HHH 정렬을 일관되게 향상시켜, 예를 들어 DeepSeek-7B에서 평균 정렬 점수가 +32.4% 증가하고, 부적절한 출력이 11.0% 감소하는 성과를 보여주었습니다. 이러한 결과는 AMBS가 최첨단 방법에 비해 경쟁력을 유지하면서도 효과적인 멀티 목표 정렬을 제공함을 나타냅니다.



### Representing LLMs in Prompt Semantic Task Spac (https://arxiv.org/abs/2509.22506)
Comments:
          Accepted to Findings of the Association for Computational Linguistics: EMNLP 2025

- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 성능을 효과적으로 예측하고 모델 선택을 용이하게 하는 새로운 접근 방식을 제시합니다. 이 연구에서는 모델을 프롬프트의 의미적 작업 공간(semantic task space) 내에서 선형 연산자로 나타내는 방법을 도입하였습니다. 이러한 방식은 해석 가능성과 확장성을 높이는 동시에, 기존의 리소스를 활용하여 실시간으로 새로운 모델이나 벤치마크를 추가할 수 있도록 해 줍니다.

- **Technical Details**: LLM의 성능 예측 및 모델 선택을 위해 제안된 방법은 각 LLM을 성공적으로 계산하는 프롬프트와 정렬된 벡터로 임베딩합니다. 이를 통해 모델을 선형적으로 표현할 수 있도록 하며, 이 과정에서 기하학적 특성의 닫힌 형태(computable closed-form)를 활용합니다. 이 방법은 훈련이 필요 없는 특성을 가지며, 대부분의 연산 자원을 소모하지 않고도 실시간으로 확장 가능한 평가를 가능하게 합니다.

- **Performance Highlights**: 성능 예측 및 모델 선택 작업에 대한 실험 결과, 제안된 방법은 주목할만한 성과를 보이며 경쟁력 있는 결과를 도출했습니다. 특히, 이 방법은 이전의 모든 기준선(baselines)을 초월하는 결과를 보여주었으며, 다양한 실제 시나리오에서 강력한 성능을 발휘했습니다. 이는 소비자와 산업 전반에 걸쳐 LLM의 응용 가능성을 크게 확대하는 데 기여할 것으로 기대됩니다.



### JGU Mainz's Submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: MT and QA (https://arxiv.org/abs/2509.22490)
Comments:
          WMT 25 Shared Task LLMs with Limited Resources for Slavic Languages: MT and QA

- **What's New**: 이번 논문은 슬라브 언어에 대한 제한된 리소스를 가진 대규모 언어 모델(LLMs)인 Qwen2.5-3B-Instruct 모델을 사용하여 슬라브 언어인 우크라이나어, 상부 소르비아어, 하부 소르비아어에 대한 기계 번역(MT) 및 질문 응답(QA) 작업을 공동으로 파인튜닝(fine-tuning)한 내용을 다룹니다. 특히, 파라미터 효율적인 파인튜닝을 통해 두 가지 작업을 동시에 수행하며, 추가적인 번역 및 선택형 질문 응답 데이터 세트를 통합합니다. 우크라이나 QA 작업에는 검색 증강 생성(RAG) 기법을 사용하며, 하부 및 상부 소르비아어 QA 작업은 앙상블(ensemble) 방법을 적용했습니다.

- **Technical Details**: 우리는 슬라브 언어에서 기계 번역과 QA를 수행하기 위해 Qwen2.5-3B-Instruct 모델을 사용했습니다. 번역 데이터는 독일어, 체코어 및 영어에서 제공되며, 훈련 데이터를 증가시키기 위해 역번역(back-translation) 기법으로 생성한 합성 데이터를 포함합니다. 또한, 16개의 고품질 영어 다답형 질문(MCQ) 데이터 세트를 QA에 추가하여 다국어 처리를 강화했습니다. 예측(inference) 시에는 비슷한 사례를 바탕으로 몇 샷 샘플링(few-shot sampling) 학습을 통해 기계 번역의 정확성을 높였습니다.

- **Performance Highlights**: 우리가 제안한 접근 방식은 모든 관련 작업에서 기준선(baseline)을 초과하는 성능을 보였습니다. 독일어에서 하부 소르비아어 번역의 ChrF++ 점수가 55포인트 이상 개선되었으며, 상부 소르비아어 번역에서는 65포인트 이상 증가했습니다. 하부 소르비아어의 QA 정확도는 12.34%포인트 상승했고, 상부 소르비아어 QA의 정확도는 10.27포인트 개선되었습니다. 우크라이나어 QA 작업에서도 제출 성능이 4.66%포인트 기준선을 초과했습니다.



### Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving (https://arxiv.org/abs/2509.22480)
Comments:
          17 pages, 11 figures

- **What's New**: 이 논문에서는 기존의 지도 세부 조정(Supervised Fine-Tuning, SFT)이나 강화 학습(Reinforcement Learning, RL) 방법과는 다른 접근 방식으로 언어 모델의 문제 해결 성능을 개선하는 전략인 솔루션 발산(solution divergence)이라는 개념을 새롭게 제안합니다. 솔루션 발산이란 단일 문제에 대해 생성된 여러 해결책 간의 다양성을 측정하는 지표로, 이는 다양한 모델에서 문제 해결 능력과 긍정적인 상관관계를 가지고 있음을 보여줍니다. 이러한 발견을 바탕으로 연구자들은 솔루션 발산을 사용하여 LLM 훈련과 평가를 향상시킬 수 있는 가능성을 탐구합니다.

- **Technical Details**: 논문에서 제안하는 솔루션 발산은 LLM이 생성한 솔루션의 집합에서 서로 다른 솔루션 간의 차이를 수치적으로 나타내며, 이는 LLM의 성능을 평가하는 새로운 지표로 사용됩니다. 연구진은 솔루션 집합의 평균 발산을 계산하기 위해 관계 그래프를 구성하였고, 그래프의 고유값(eigenvalues)을 통해 발산을 측정합니다. 이 과정에서 문자열 편집 거리(normalized string edit distance)를 사용하여 솔루션 간의 유사성을 측정하며, 이 방법은 다양한 도메인에서 일관되고 계산 효율적인 특성을 제공합니다.

- **Performance Highlights**: 제안된 솔루션 발산 메트릭을 세 가지 문제 영역, 즉 수학, 프로그래밍 및 논리적 추론 영역에 적용한 결과, 발산을 활용한 접근 방식이 성공률을 일관되게 향상시킨다는 것을 실증적으로 확인했습니다. 이러한 성과는 솔루션 발산이 LLM의 문제 해결 능력을 증대시키는 간단하면서도 효과적인 도구가 될 수 있음을 시사합니다. 교육 연구 또한 적절한 솔루션 다양성을 지닌 학습자가 더 좋은 학습 결과를 낸다는 것을 보여 주어, 이 메트릭의 활용 가능성을 뒷받침합니다.



### NeLLCom-Lex: A Neural-agent Framework to Study the Interplay between Lexical Systems and Language Us (https://arxiv.org/abs/2509.22479)
Comments:
          Findings of EMNLP 2025

- **What's New**: 이 논문은 NeLLCom-Lex라는 신경 기반 프레임워크를 도입하여 의미 변화(semantic change)를 시뮬레이션합니다. 기존의 관찰적(methods) 방법이나 인간 실험을 통해서는 형성의 메커니즘을 규명하기 어려운 점을 해결하고자 하였으며, 실질적인 어휘 체계(lexical system)에 기반한 에이전트(agent)를 통해 의사소통 필요에 따라 체계적으로 조절합니다. 실험을 통해 이 시스템은 사람과 유사한 색상 명명(color naming) 행동을 재현할 수 있는 가능성을 보여줍니다.

- **Technical Details**: NeLLCom-Lex는 논의된 이론에 기초하여 기계 학습(machine learning) 방식인 감독 학습(superevised learning)과 강화 학습(reinforcement learning) 을 결합하여 통신 압력이 언어 사용 및 어휘 적응에 미치는 영향을 조사합니다. 이 프레임워크는 색상 명명 작업(color naming task)을 통해 의사소통 환경에서 화자가 필요한 구분의 세분화(granularity) 차이를 기반으로 연구합니다. 이를 통해 우리는 에이전트의 어휘 시스템(lexical system)에 대한 실험적 분석을 수행합니다.

- **Performance Highlights**: NeLLCom-Lex 시스템은 인간과 유사한 실용적(pragmatic) 명명 행동을 재현할 수 있으며, 다양한 의사소통 필요에 따라 에이전트의 명명 행동이 변하는 것을 관찰했습니다. 이 결과는 언어 사용(language use)과 언어 구조(language structure) 간의 상호작용을 탐구하는 데 있어 NeLLCom-Lex의 유용성을 지지합니다. 이러한 성과는 커뮤니케이션에서의 인지적 압력이 언어적 적응에 어떻게 작용하는지를 밝혀낼 수 있는 기초 자료를 제공합니다.



### Evaluating the Limits of Large Language Models in Multilingual Legal Reasoning (https://arxiv.org/abs/2509.22472)
Comments:
          39 pages, 36 figures. Code and evaluation pipeline available at this https URL

- **What's New**: 본 연구는 다국어 환경에서의 법률 작업을 다루며, LLM(대형 언어 모델)인 Meta의 LLaMA와 Google의 Gemini의 성능을 평가합니다. 특히, 법률과 비법률 벤치마크에서의 성능을 비교하고, 인위적인 변형을 통한 법적 작업의 견고성을 분석합니다. 고유한 LLM-as-a-Judge 평가 접근 방식을 통해 인간과 정렬된 평가를 진행하며, 법적 작업에 초점을 맞춘 모듈형 오픈 소스 평가 파이프라인을 제공합니다.

- **Technical Details**: 이 연구는 법률적 언어와 추론의 복잡성이 LLM의 성능에 미치는 영향을 평가합니다. LLaMA와 Gemini의 두 가지 최신 모델을 사용하여 다국어 법률 분류, 요약, 공정성 예측, 법률적 추론 및 적대적 견고성 테스트를 수행합니다. 법률 작업은 기술적 용어와 복잡한 문장 구조로 인해 LLM이 일반 작업보다 낮은 정확도를 보이는 경향이 있음을 확인했습니다.

- **Performance Highlights**: 연구 결과, LLaMA는 Gemini보다 약 24%포인트 낮은 성능을 보였으며, 법률 추론 벤치마크에서는 LLM이 50% 미만의 정확도를 기록하는 경우가 많았습니다. 영어는 전반적으로 더 안정적인 성능을 보여주지만, 더 높은 정확도를 보장하지는 않습니다. 또한, 언어의 성능은 영어와의 구문 유사성과 정비례하는 경향이 있으며, LLM의 응답 중립성 및 긍정성 경향은 공정성 분류 작업에서 두드러지게 나타났습니다.



### Detecting (Un)answerability in Large Language Models with Linear Directions (https://arxiv.org/abs/2509.22449)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)의 불확실한 응답 문제를 해결하기 위한 새로운 방법을 제안합니다. 기존의 방법들이 불확실성을 탐지하는 데 있어 일관성이 부족한 것에 반해, 우리는 모델의 활성화 공간에서 불안정성을 포착하는 방향을 식별하는 방법을 개발했습니다. 이는 불확실한 질문을 효과적으로 감지하고, 다양한 데이터셋에 걸쳐 일반화가 더 잘 되는 접근법으로 demonstration 됩니다.

- **Technical Details**: 우리의 접근 방법은 모델의 내부 활성화에서 불안정성을 나타내는 특정 방향을 찾아내는 데 기초합니다. 우리는 활성화 추가를 통해 후보 방향을 생성한 뒤, 이 방향과의 정렬 정도를 측정하여 불안정성을 분류합니다. 이 방법은 최종적으로 질의에 대한 모델의 내부 표현이 불안정한 사례와 얼마나 정렬되어 있는지를 나타내는 스칼라 점수를 제공합니다.

- **Performance Highlights**: 우리의 방법은 SQuAD 2.0, RepLiQA, Natural Questions, MuSiQue의 네 가지 질문-응답 데이터셋에서 평가되었고, 75.9%에서 96.4%의 F1 스코어를 기록했습니다. 기존의 분류기 기반 접근 방식보다 우수한 성능을 보였습니다. 데이터를 넘어 신뢰할 수 있는 일반화 능력을 발휘하며, 관련성 있는 신호를 다시 확인하기 위한 인과 개입을 통해 모델의 응답 행태를 효과적으로 조절할 수 있음을 입증했습니다.



### Chimera: Diagnosing Shortcut Learning in Visual-Language Understanding (https://arxiv.org/abs/2509.22437)
Comments:
          Our code (this https URL) and data (this https URL) are publicly available

- **What's New**: 이번 논문에서는 Chimera라는 새로운 테스트 스위트를 소개합니다. 이 테스트 스위트는 7,500개의 고품질 다이어그램을 포함하고 있으며, 각 다이어그램은 의미론적 트리플(semantic triples)로 주석이 달립니다. Chimera는 다이어그램 이해의 네 가지 기본 요소인 개체 인식(entity recognition), 관계 이해(relation understanding), 지식 기초(knowledge grounding), 시각적 추론(visual reasoning)을 평가하기 위한 다양한 수준의 질문들을 포함하고 있습니다.

- **Technical Details**: Chimera 데이터셋은 Wikipedia의 다이어그램 이미지를 수집하여 구성하였으며, 불필요한 이미지를 필터링하여 품질이 뛰어난 다이어그램을 보장합니다. 각 다이어그램에는 의도한 내용을 기술하는 주석과, 다이어그램 이해 과정에서의 시퀀스를 분석하기 위한 네 가지 수준의 질문이 포함되어 있습니다. 테스트는 비주얼 및 의미적 모달리티를 통해 다이어그램의 이해를 평가하며, 다이어그램의 내용을 다양한 표현 렌즈로 재구성하여 제출하게 됩니다.

- **Performance Highlights**: 15개의 오픈 소스 VLM을 대상으로 한 평가 결과, 모델들은 다이어그램 이해에서 공통적인 단축키(시각적 단기기억, 지식 회상, Clever-Hans)를 활용하는 경향이 있음을 보였습니다. 특히 Clever-Hans 단축키가 특히 심각하여, 모델들이 진정한 이해 없이 표면적인 언어 패턴을 기반으로 높은 성과를 이룰 수 있음을 보여주었습니다. 이 연구 결과는 현재 VLM의 한계를 드러내며, 진정한 시각적 이해를 위한 강력한 평가 프레임워크의 필요성을 강조합니다.



### What Is The Political Content in LLMs' Pre- and Post-Training Data? (https://arxiv.org/abs/2509.22367)
Comments:
          9 pages, under review

- **What's New**: 이번 연구에서는 OLMO2라는 대규모 오픈소스 모델의 학습 데이터를 분석하여 정치적 편향의 기원을 규명하려고 합니다. 이 모델의 요구 데이터셋은 공개되어 있으며, 이를 통해 훈련 데이터에 포함된 정치적 콘텐츠의 비율을 도출하고, 모델의 특정 정책 이슈에 대한 입장과의 상관관계를 평가합니다. 또한, 학습 전후 데이터 세트를 비교하여 정치적으로 기여한 콘텐츠의 양을 증가시킬 수 있는 지침을 제공합니다.

- **Technical Details**: 연구진은 OLMO2의 전후 훈련 데이터에서 무작위 샘플을 추출하고, 이를 정치적 방향으로 자동 주석 처리합니다. 이러한 데이터는 좌파, 우파, 중립으로 분류되며, 새로운 분류기를 통해 유효성을 검증합니다. 분석 결과, 좌파 성향의 문서가 우파 성향의 문서보다 3배에서 12배 더 많고, 훈련 데이터에서 정치적으로 관여한 콘텐츠는 사전 훈련 샘플에서 네 배 더 많음을 발견했습니다.

- **Performance Highlights**: 결과적으로, 연구는 정치적 편향이 주로 훈련 과정에서 형성된다는 것을 강조합니다. 또한, 모델 행동과 훈련 데이터에서의 지배적 관점 사이에 강한 상관관계(rr=0.90)가 있음을 보여줍니다. 이 연구는 훈련 데이터의 정치적 콘텐츠 분석이 향후 LLM 개발 및 검증 과정에서 필수적임을 시사합니다.



### Exploratory Semantic Reliability Analysis of Wind Turbine Maintenance Logs using Large Language Models (https://arxiv.org/abs/2509.22366)
- **What's New**: 이 논문은 풍력 터빈 유지보수 로그의 비구조화된 자유 텍스트에서 운영 정보를 분석하기 위한 새로운 탐색적 프레임워크를 소개합니다. 기존의 기계 학습 접근 방식이 주로 분류에 국한된 반면, 새로운 접근법은 대형 언어 모델(LLMs)을 활용하여 복잡한 추론 작업을 수행합니다. 이를 통해 실패 모드 식별, 인과 관계 추론, 비교 분석, 데이터 품질 감사 등 네 가지 분석 워크플로우를 실행하였습니다.

- **Technical Details**: 이 연구는 LLMs를 활용하여 비구조화된 자연어 입력에서 신뢰성 통찰력을 도출하는 탐색적 프레임워크를 사용합니다. 데이터셋 준비와 분석 코호트 선정, 구조적 프롬프트 엔지니어링 접근 방식을 적용한 분석 작업 설계를 포함한 두 가지 주요 단계를 거칩니다. GPT-5와 구글의 Gemini 2.5 Pro라는 두 가지 최첨단 LLM이 연구에 사용되었으며, 각 분석 작업에 대해 구체적인 역할과 작업 목록을 지정한 프롬프트가 제공되었습니다.

- **Performance Highlights**: 모델에 의해 생성된 통찰력은 네 가지 설계된 작업 전반에 걸쳐 뚜렷한 결과를 보여주었습니다. 예를 들어, 특정 고장 모드 분석에서는 높은 빈도의 유지보수 이벤트와 관련된 정보를 도출해내었고, 인과 관계 추론에서는 사건의 연대기적 순서에서 근본 원인을 추론하는데 성공했습니다. 이러한 결과는 운영 인텔리전스를 향상시키고 비구조화된 데이터에서 숨겨진 통찰력을 샤시할 수 있는 새로운 방법론에 기여하고 있습니다.



### CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models (https://arxiv.org/abs/2509.22360)
- **What's New**: 이 논문은 CHRONOBERG라는 250년 동안의 영어 책 텍스트로 구성된 시간적으로 구조화된 말뭉치를 도입합니다. 기존의 다양한 웹 크롤링 데이터셋은 장기적인 시간 구조가 부족하여 LLMs의 언어의 의미와 규범적 변화 맥락을 잘 반영하지 못합니다. CHRONOBERG는 Project Gutenberg에서 수집된 자료를 바탕으로 하여 시간에 따라 변화하는 어휘 의미를 분석할 수 있도록 합니다.

- **Technical Details**: CHRONOBERG는 2.7B (billion) 토큰으로 구성된 데이터셋으로, 레퍼런스가 되는 전통적인 Valence-Arousal-Dominance (VAD) 어휘가 거의 30만 개의 단어를 포함하도록 확장되었습니다. 이를 통해 정서적 의미의 변화와 감정의 추적을 지원하는 정량적 분석이 가능합니다. 이 데이터셋은 역사적 의미 전이를 반영한 교육을 통해 LLM이 더욱 효과적으로 발전할 수 있도록 돕습니다.

- **Performance Highlights**: 제안된 CHRONOBERG는 LLM이 과거의 맥락에서 언어를 인식하고 적응하는 능력의 부족을 드러내며, 과거 정보의 망각과 미래 문장에 대한 일반화를 다루는 데 어려움을 겪고 있음을 보여줍니다. 또한, 현대 LLM 기반의 도구들이 차별 언어의 탐지와 다양한 시간대에서의 정서적 맥락화에 더 나은 위치에 티기 위해서는 시간이 준수된 처리 방법이 필요함을 강조합니다.



### Conversational Implicatures: Modelling Relevance Theory Probabilistically (https://arxiv.org/abs/2509.22354)
- **What's New**: 최근 베이지안 확률 이론(Bayesian probability theory) 및 인지 과학(cognitive science) 적용의 발전은 실용어학(pragmatics)과 의미론(semantics)에서의 '확률적 전환(probabilistic turn)'을 초래했습니다. 특히, 합리적 발화 행위 이론(Rational Speech Act theory) 프레임워크가 정립되었으며, 이를 통해 간단한 참조 게임(reference games)에서 시작하여 복잡한 의사소통 교환까지 모델링해왔습니다. 이 논문은 관련성 이론(relevance-theoretic pragmatics)을 통해 생길 수 있는 베이지안 접근 방식을 탐구합니다.

- **Technical Details**: 논문은 두 가지 주요 개념을 중심으로 구성되어 있습니다: 함축(implicature)과 설명(explicature)입니다. 함축은 발화자의 의도를 이해하기 위해 대화 상대자가 필요한 맥락별 전제로서 추가해야 하는 내용입니다. 반면, 설명은 맥락에 의해 직간접적으로 제공된 정보로, 상대방이 명확하게 전달된 정보를 기반으로 논리적 결론을 도출할 수 있게 돕습니다. 해당 논문은 이러한 개념들을 확률적 모델(probabilistic model) 내에서 캡처하고자 합니다.

- **Performance Highlights**: 이 논문의 주요 성과는 함축 이해(implicature comprehension)에 대한 확률적 모델을 ProbLog로 구현한 것입니다. 이를 통해 의사소통에서의 암시적 의미의 전달과 이해 과정에서 발생하는 인지적 측면을 분석하였습니다. 또한, 관련성 이론에 대한 깊은 통찰을 제공하며, 발화자가 의도한 의미를 추론하는 데 있어 사용자의 능동적인 역할을 강조합니다.



### The InviTE Corpus: Annotating Invectives in Tudor English Texts for Computational Modeling (https://arxiv.org/abs/2509.22345)
- **What's New**: 이번 논문에서는 자연어 처리(Natural Language Processing, NLP) 기법을 역사 연구에 적용하며, 특히 튜더 시대의 종교적 비방어(Invective)에 대해 다룹니다. 이를 위해 원시 데이터에서 전처리(Pre-processing) 및 데이터 선택을 거쳐 반복적인 주석 프로세스까지의 워크플로우를 제안합니다. 결과적으로 16세기 영국에서 비방 언어에 관한 전문가 주석이 포함된 거의 2000개의 초연대 영어 문장으로 구성된 InviTE 코퍼스를 소개합니다.

- **Technical Details**: 본 연구에서는 튜더 시대(1485-1603)의 문서에서 비방어를 조사합니다. 비방어의 정의는 Schwerhoff et al.(2017)이 제안한 프레임워크를 기반으로 하며, 모든 형태의 의사소통이 포함됩니다. 연구 방법론으로 역사적 관점을 전산 처리할 수 있는 주석 방식을 구성하고, 저자 및 출판 연도와 같은 메타데이터를 통합하며, 희소하고 역사적으로 변동이 큰 데이터를 처리할 수 있는 방법을 개발합니다.

- **Performance Highlights**: 최적화된 BERT 기반 모델과 제로 샷(prompted instruction-tuned) 대형 언어 모델(LLMs)의 성능을 평가하고 비교합니다. 연구 결과, 역사 데이터에 대해 미리 훈련된 모델이 비방 탐지에서 우수한 성능을 보이며, 작은 모델이라도 세밀한 조정(fine-tuning)을 통해 훨씬 큰 LLM보다 뛰어난 성능을 나타냅니다. 이러한 결과는 인문학적 개념 틀을 전산 연구로 전환할 수 있는 방법을 제시하고, 코퍼스 구축 및 주석 작업의 가치를 강조합니다.



### Transformers Can Learn Connectivity in Some Graphs but Not Others (https://arxiv.org/abs/2509.22343)
Comments:
          Under Review

- **What's New**: 이 연구는 transformer 기반의 대형 언어 모델(LLMs)이 전이 관계(transitive relations)를 추론하는 능력을 분석합니다. 특히, 그래프에서의 연결성(connectivity) 문제를 학습하기 위한 transformers의 성능에 초점을 맞추어, 다양한 크기의 방향 그래프를 생성하여 훈련하는 방식을 사용합니다. 연구 결과, 저차원 그리드 그래프에서 transformers가 연결성을 학습하는 데 성공적인 반면, 연결이 끊긴 그래프에서는 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 transformers가 전이 관계를 추론할 수 있는지를 평가하기 위해 훈련 예제를 사용하여 연결성을 학습하는 능력을 조사합니다. 저차원 그리드 그래프에서는 노드들이 저차원 서브스페이스로 쉽게 임베딩(embedding) 될 수 있어 연결성이 쉽게 추론될 수 있습니다. 반면, 고차원 그리드 그래프와 연결이 끊긴 체인 그래프에서는 연결성 학습이 어려움을 겪습니다. 모델 크기를 확장하는 것이 그리드 그래프에서 연결성을 학습하는 데 긍정적인 영향을 미친다고 밝혀졌습니다.

- **Performance Highlights**: 연구 결과는 transformers가 저차원 그리드 그래프에서 연결성을 학습하는 데 성공적이라는 것을 보여줍니다. 그러나 고차원 그리드 그래프와 많은 구성 요소를 가진 연결 끊긴 그래프에서는 성능이 떨어지는 경향을 보입니다. 그래프 크기를 확장하였을 때 연결이 끊긴 체인 그래프에 대한 transformers의 성능이 개선된다는 점도 주목할 만 합니다. 요약하자면, transformers는 연결성 학습에 있어 그래프 크기를 확장하는 것에서 더 많은 혜택을 보임을 보여줍니다.



### Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs (https://arxiv.org/abs/2509.22338)
Comments:
          15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)

- **What's New**: 본 논문에서는 자연어(natural language)를 1차 논리(first-order logic)로 자동 번역하는 과정에서의 대규모 언어 모델(LLM)의 성능을 평가합니다. 새로운 과제에 대한 LLM의 성능을 비교하기 위해 다양한 아키텍처(encoder-decoder vs. decoder-only)와 훈련 전략을 검토하였습니다. 특히, 우리는 MALLS와 Willow 데이터셋을 사용하여 어휘 확장(vocabulary extension), 술어 조건부(predicate conditioning), 다국어 훈련(multilingual training)과 같은 기법을 탐구하였습니다.

- **Technical Details**: 모델 훈련을 위해 encoder-decoder 및 decoder-only 구조의 다양한 LLM을 사용하여 시스템적인 비교를 수행했습니다. Fine-tuning된 Flan-T5-XXL 모델이 술어 리스트를 활용하여 70%의 정확성을 달성하며, 기존의 모델들보다 우수한 성능을 보였습니다. 이 과정에서 Low-Rank Adaptation (LoRA)을 적용하여 모델의 훈련 상태를 개선하고, AdamW Optimizer를 통해 모델 성능을 극대화하는 방법을 사용했습니다.

- **Performance Highlights**: 주요 결과에 따르면, 술어의 가용성이 성능을 15-20% 향상시키며, T5 모델이 더 큰 decoder-only LLM보다 뛰어난 성능을 나타냅니다. 또한, 이러한 모델들은 특정 훈련 없이도 보지 못한 논리적 주장을 일반화할 수 있는 능력을 가지며, 기존의 해석과 비교할 수 있는 새로운 메트릭을 도입했습니다. 구조적 논리 번역의 강점과 함께, 술어 추출(predicate extraction)이 주요 병목현상으로 나타났습니다.



### Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection? (https://arxiv.org/abs/2509.22291)
- **What's New**: 이번 연구는 증오 발언 탐지(hate speech detection)라는 중요한 응용 분야에서 설명 가능성(explainability)과 공정성(fairness) 간의 관계를 최초로 체계적으로 분석하였습니다. 연구는 세 가지 주요 질문을 통해 입력 기반 설명(input-based explanations)이 편향된 예측(biased predictions)을 식별하고, 공정한 모델을 자동으로 선택하며, 모델 훈련 동안 편향을 완화하는 데 어떻게 도움이 되는지를 탐구합니다.

- **Technical Details**: 연구에서는 입력 기반 설명이 모델의 예측에 대한 각 토큰의 기여를 나타내며, 이를 통해 모델의 행동을 더 명확하게 이해할 수 있도록 돕습니다. 특히, 본 논문은 이러한 설명 방법이 모델 학습 중 편향을 줄이는 데 유용하다고 밝혔으며, 다양한 설명 방법의 유효성을 비교하기 위한 정량적 분석을 실시했습니다.

- **Performance Highlights**: 연구 결과, 입력 기반 설명이 편향된 예측을 효과적으로 검출하고, 모델 훈련 중 편향을 줄이는 데 기여하는 것으로 나타났습니다. 그러나 이러한 설명이 후보 모델 중에서 공정한 모델을 자동으로 선택하는 데는 덜 신뢰할 수 있음을 보여주었습니다. 또한, 설명 기반의 편향 검출이 민감한 특성(sensitive features) 의존도를 줄이기 위해 훈련된 모델에서도 탄탄한 성능을 유지한다는 점이 강조되었습니다.



### Beyond Textual Context: Structural Graph Encoding with Adaptive Space Alignment to alleviate the hallucination of LLMs (https://arxiv.org/abs/2509.22251)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구에서는 SSKG-LLM이라는 새로운 모델 아키텍처를 제안하여 지식 그래프(KGs)의 구조적 및 의미적 정보를 효과적으로 통합하고 LLMs의 추론 과정에 활용할 수 있도록 하였습니다. 기존의 LLM 기술이 KGs를 단순 텍스트로 취급하며 중요한 구조적 측면을 활용하지 못하고 있다는 문제를 해결하고자 했습니다. SSKG-LLM은 Knowledge Graph Retrieval(KGR), Knowledge Graph Encoding(KGE), Knowledge Graph Adaptation(KGA) 모듈을 포함하여, LLMs가 KGs 임베딩을 이해할 수 있게 돕습니다.

- **Technical Details**: SSKG-LLM은 두 가지 핵심 도전 과제를 해결합니다. 첫째로, KGs의 구조적 및 의미적 정보를 어떻게 획득하고 통합할 것인지에 대한 문제입니다. 이를 위해, GraphLM이라는 새로운 사전 훈련 모델을 사용하여 KGs의 하위 그래프를 인코딩하고 구조와 의미의 뉘앙스를 보존합니다. 둘째로, KGs 인코딩과 LLMs 간의 간극을 어떻게 메꿀 것인지에 대한 문제입니다. 이를 해결하기 위해, cross-attention을 활용한 KG-Adapter 모듈을 제안하여 그래프와 텍스트 인코딩을 서로 조정할 수 있도록 하였습니다.

- **Performance Highlights**: 우리의 방법은 Multiple types of Question Answering(QA) 데이터셋을 통해 기존 모델들을 능가하는 성과를 보였습니다. 특히 LLMs 기반의 QA 작업에서 KGs 통합에 있어 큰 성과를 나타내며, 구조적 정보의 중요성을 강조합니다. 우리의 실험 결과는 SSKG-LLM이 KGs의 정보와 LLMs 간의 통합을 더 효과적으로 수행하여 더 정확한 답변을 생성할 수 있음을 보여줍니다.



### Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Complianc (https://arxiv.org/abs/2509.22250)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 안전성을 법적 준수 관점에서 접근하여 새로운 방법론, 즉 '안전 준수'를 제안합니다. 기존의 안전 방법론과 달리, 유럽 연합 인공지능법(EU AI Act)과 일반 데이터 보호 규정(GDPR) 등의 법적 프레임워크를 안전 기준으로 활용합니다. 이러한 접근법은 LLM의 복잡한 행동을 보다 시스템적이고 철저하게 보호하는 기반을 제공합니다.

- **Technical Details**: 연구진은 법적 조항을 사용하여 현실적인 LLM 안전 시나리오를 생성하는 새로운 벤치마크를 개발하였습니다. 이를 통해 LLM의 성능을 평가하고, Qwen3-8B 모델을 Group Policy Optimization (GRPO) 기법을 사용해 정합성 추론기인 Compliance Reasoner를 구축하였습니다. 이 모델은 LLM을 법적 기준에 맞춰 조정하여 안전 위험을 완화하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, Compliance Reasoner는 새로운 벤치마크에서 EU AI Act에 대해 +10.45%, GDPR에 대해 +11.85%의 평균 개선률을 보이며 우수한 성능을 나타냈습니다. 또한, 기존의 안전 데이터를 준수 시나리오로 확장하여 안전 준수를 위한 데이터의 양을 크게 증가시키는 일반화 가능 방법을 제공합니다.



### FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction (https://arxiv.org/abs/2509.22243)
- **What's New**: 이 논문에서는 FLEXI라는 첫 번째 풀 듀플렉스 LLM-인간 음성 상호작용 벤치마크를 소개합니다. 이 벤치마크는 비상 시나리오에서 모델 중단을 명시적으로 포함하여 풀 듀플렉스 상호작용을 체계적으로 평가합니다. FLEXI는 오픈 소스와 상용 모델 간의 성능 차이를 보여주며, 긴급 상황 인식, 턴 종료 및 상호작용 지연 시간에서의 중요한 격차를 드러냅니다.

- **Technical Details**: FLEXI 벤치마크는 다섯 가지의 주요 사용자-Language Model (LM) 상호작용 시나리오를 포함하여, 듀플렉스 상호작용의 자연스러움과 지능을 평가합니다. 이 방법론은 서로 다른 발화 패턴 및 상대적 정전기간을 정의하며, 각 시나리오는 모델의 반응 적절성과 타당성을 측정하는 다양한 지표를 통해 효율적으로 평가됩니다. 특히, 전체 상호작용은 두 개의 사용자 및 모델이 포함된 교차 검증으로 분석됩니다.

- **Performance Highlights**: FLEXI 벤치마크 평가 결과, 오픈 소스 모델들이 실제 응답 시간에서는 상용 모델에 비해 뒤쳐지는 것으로 나타났습니다. 비상 상황 및 사용자 중단 상황에서의 모델 반응 속도와 적합성을 비교하여, 상용 모델이 상대적으로 더 나은 성능을 보였습니다. 이를 통해, 정밀한 다운링크 설정 및 다음 토큰 쌍 예측 아키텍처가 진정한 풀 듀플렉스 상호작용을 달성하는데 필수적임을 강조합니다.



### FeatBench: Evaluating Coding Agents on Feature Implementation for Vibe Coding (https://arxiv.org/abs/2509.22237)
- **What's New**: 이번 논문은 새롭게 등장한 "vibe coding" 패러다임을 다루며, 이를 평가하기 위한 새로운 벤치마크인 FeatBench를 제안합니다. 기존 코드 생성 평가 기준들이 vibe coding의 요구사항을 충족하지 못하는 문제를 지적하며, 이 틈새를 메우기 위한 다양한 기능 구현에 중점을 둡니다. FeatBench는 사용자 관점에서 자연어로 설명된 기능을 구현하는 과제를 평가하게 설계되었습니다.

- **Technical Details**: FeatBench는 전적으로 자연어 프롬프트를 사용하며, 코드나 구조적 힌트가 포함되지 않은 단순한 설명만을 기반으로 합니다. 데이터 수집 프로세스는 정확성과 품질을 보장하기 위해 엄격하게 구성되어 있으며, 자동화된 파이프라인을 통해 지속적으로 진화합니다. 또한 각 과제는 Fail-to-Pass (F2P)와 Pass-to-Pass (P2P) 테스트 케이스를 포함하여 정확성과 기존 기능의 보존을 검증합니다.

- **Performance Highlights**: 주요 테스트 결과에 따르면 FeatBench는 기존 SOTA(최첨단) 에이전트 frameworks에 상당한 도전을 제공합니다. 평가된 에이전트들은 새로운 기능을 추가할 때 기존 기능을 손상시키는 경향이 있으며, 최고 성공률은 29.94%로 나타났습니다. 특이하게도, 에이전트들은 "aggressive implementation" 전략을 취하는데, 이는 적절한 소프트웨어 아키텍처를 제공하는 동시에 작업 실패를 초래하는 경향이 있습니다.



### In Their Own Words: Reasoning Traces Tailored for Small Models Make Them Better Reasoners (https://arxiv.org/abs/2509.22230)
- **What's New**: 이 논문은 대형 언어 모델에서 소형 모델로의 추론(Reasoning) 능력 전이에서 발생하는 문제를 다룹니다. 특히, 고품질의 교육 데이터를 활용하더라도 성능이 저하되는 '분포 불일치(Distributional Misalignment)' 현상을 지적합니다. 저자들은 '역 추론 데코딩(Reverse Speculative Decoding, RSD)'이라는 새롭고 효과적인 기법을 제안하여 소형 학생 모델의 학습 능력을 개선할 수 있는 방법을 모색합니다.

- **Technical Details**: RSD에서는 교사 모델이 후보 토큰을 제안하고, 학생 모델이 자신의 확률 분포(Ps)에 기반해 이를 수용할지를 결정합니다. 특정 확률 임계값(pth) 이하의 토큰은 제거되어 다른 후보 토큰을 선택하게 됩니다. 이 과정은 학생 모델의 내부 표현 능력에 맞춰 추론을 조정하고, 교사가 제공하는 지침이 학생에게 순조롭게 전달되도록 합니다.

- **Performance Highlights**: 실험 결과, RSD로 생성된 추론 흔적을 사용한 모델은 주요 추론 기준에서 4.9%의 성능 개선을 보였습니다. 이는 직접적인 SFT(Supervised Fine-Tuning)와 비교했을 때 성능이 20.5% 하락하는 경향을 극복함으로써 확인되었습니다. 저자들은 저확률 토큰이 효과적인 능력 전이의 주요 병목 현상임을 강조하며, RSD의 필요성을 입증합니다.



### Thinking in Many Modes: How Composite Reasoning Elevates Large Language Model Performance with Limited Data (https://arxiv.org/abs/2509.22224)
Comments:
          7 pages, 3 figures

- **What's New**: 본 논문에서는 Composite Reasoning (CR)이라는 새로운 추론 방식을 소개합니다. 기존의 대형 언어 모델(LLMs)은 단일 추론 패러다임에 의존했으나, CR은 다양한 추론 스타일(예: deductive, inductive, abductive)을 동적으로 결합하여 더 정교한 문제 해결을 가능하게 합니다. 이 방법은 의학과 과학 질문 응답 기준에서 기존 방법들보다 향상된 성능을 보였습니다.

- **Technical Details**: CR 접근법에서는 모델이 여러 추론 전략을 탐색하고 통합하도록 유도합니다. 이 과정은 Low-Rank Adaptation (LoRA)와 Group Relative Policy Optimization (GRPO)와 같은 파라미터 효율적인 미세 조정 기법을 사용하여 이루어집니다. 이를 통해, 모델은 문제 해결의 다양한 경로와 관점을 고려하여 보다 정확하고 잘 뒷받침된 답변을 생성하도록 돕습니다.

- **Performance Highlights**: CR은 세 가지 도전적인 데이터셋에서 테스트되었으며, 기존의 Chain-of-Thought (CoT) 및 Standard Reasoning (SR)과 비교하여 탁월한 성과를 보였습니다. 특히, CR SFT + GRPO 조합은 ARC-Complex와 MedMCQA에서 각각 94.99%, 56.30%의 정확도를 기록하며 가장 높은 성과를 달성했습니다. 이는 리소스 제한 환경에서도 CR이 더 효과적인 추론 경로를 탐색할 수 있다는 점을 보여줍니다.



### StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs (https://arxiv.org/abs/2509.22220)
- **What's New**: 이 논문에서는 기존의 semantic speech tokenizers의 취약성을 지적하고, 이를 해결하기 위한 새로운 솔루션인 StableToken을 제안합니다. 현행 tokenizers는 의미와 무관한 acoustic perturbations에 대해 매우 취약하며, 이는 높은 Signal-to-Noise Ratios(SNRs)에서도 token sequence의 변화를 초래합니다. StableToken은 다중 경로 아키텍처와 consensus-driven 메커니즘을 통해 이러한 문제를 해결하고, Token의 안정성을 획기적으로 향상시킵니다.

- **Technical Details**: StableToken은 Voting-LFQ 모듈을 소개하여 multi-branch quantization을 통해 audio를 병렬로 처리합니다. 이 모듈은 bit-wise voting 기법을 기반으로 하여 입력의 다양한 표현을 통합하여 안정적인 token sequence를 생성합니다. 학습 전략으로는 Noise-Aware Consensus Training을 도입하여, 입력에 대한 여러 'views'를 제공함으로써 중간의 안정성을 강조합니다.

- **Performance Highlights**: StableToken은 Unit Edit Distance(UED)를 60% 이상 줄이며 노이즈에 대한 안정성을 높입니다. 추가적으로, SpeechLLMs에 적용 시, 다양한 작업에서 성능이 개선되어, 실제 환경의 잡음 수준이 높을수록 StableToken과 기존 모델 간의 성능 차이가 극대화됩니다. 이러한 결과는 tokenizer의 강인성을 높이는 것이 더 탄력적인 SpeechLLMs 구축에 매우 효과적임을 보여줍니다.



### Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation (https://arxiv.org/abs/2509.22211)
- **What's New**: 이번 논문에서는 Recursive Thematic Partitioning (RTP)이라는 새로운 프레임워크를 소개합니다. RTP는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 데이터 세트를 상호작용적으로 분석하는 방법을 제시합니다. 이 프레임워크는 데이터를 의미적으로 나누는 자연어 질문을 생성하며, 이를 통해 해석 가능한 주제 계층 구조를 구축합니다.

- **Technical Details**: RTP는 이진 트리를 구축하는 과정에서 LLM에 의한 질문 생성을 통해 문서를 재귀적으로 분할합니다. 각 트리 노드는 '이 리뷰는 주로 음식 품질에 대한 것인가, 고객 서비스에 대한 것인가?'와 같은 분할 질문을 포함합니다. 이 과정에서 데이터 세트를 효율적으로 다루기 위해 전처리 단계로 글로벌 샘플링과 문서 요약을 수행하여 전체 데이터베이스에서 대표 샘플을 뽑아냅니다.

- **Performance Highlights**: RTP의 실험 결과는 다른 주제 모델인 BERTopic과 비교했을 때 해석 용이성이 높고, 데이터 세트 내에서 액션 가능한 정보와의 연결성이 뛰어남을 보여줍니다. 또한, 생성된 군집이 다운스트림 분류 작업의 효과적인 특징으로 작용함을 입증하였으며, RTP 트리의 경로를 제어 가능한 주제 생성에 활용할 수 있다는 점도 강조합니다. 이는 텍스트 생성에서 높은 수준의 제어력을 제공하여 특정 특성을 일관되게 재현할 수 있도록 돕습니다.



### The Outputs of Large Language Models are Meaningless (https://arxiv.org/abs/2509.22206)
Comments:
          24 pages, 2 figures, forthcoming in Herman Cappelen and Rachel Sterken, eds. Communicating with AI: Philosophical Perspectives. Oxford: Oxford University Press

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 출력이 무의미하다는 결론에 대한 간단한 주장을 제시합니다. 이 주장은 두 가지 핵심 전제에 기반하고 있습니다. 첫째, LLM의 출력이 문자 그대로의 의미를 갖기 위해 필요한 특정 종류의 의도가 필요하다는 점과, 둘째, LLM이 그러한 의도를 가질 수 없다는 점입니다.

- **Technical Details**: 논문은 LLM의 출력이 의미를 지니기 위해 필요한 의도와 관련된 두 가지 전제에 대해 방어하고 있습니다. 저자들은 의미 외부주의자(semantic externalist)와 의미 내부주의자(semantic internalist) 등의 반응에 대해 이 주장을 방어합니다. 특히, 개념적 역할(conceptual roles)과 같은 개념 간의 내재적 관계를 통해 의미를 정의할 수 있다는 주장에 반대합니다.

- **Performance Highlights**: 그럼에도 불구하고 논문은 LLM의 출력이 어떻게 의미 있는 것처럼 보이고, 이를 통해 진실한 믿음(true beliefs)과 지식(knowledge)을 얻을 수 있는지에 대한 논의로 결론을 맺습니다. 이러한 점은 LLM의 출력이 실제로는 무의미하지만, 사용되는 맥락에서 의미 있는 것처럼 인식될 수 있다는 것을 시사합니다.



### When Does Reasoning Matter? A Controlled Study of Reasoning's Contribution to Model Performanc (https://arxiv.org/abs/2509.22193)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 지능형 사고 능력과 성능 향상을 위한 새로운 방법론을 제시합니다. Instruction Fine-Tuning (IFT)과 다양한 크기의 추론 모델을 비교하여, 수학 중심 및 일반적인 과제에서의 성능을 평가합니다. 결과적으로, 추론 능력을 활용하여 IFT 시스템보다 성능이 크게 개선됨을 보여주었습니다.

- **Technical Details**: 연구에서는 개별 모델에 대한 학습 및 추론 비용을 조정하면서도 성능을 확보하는 통제된 실험적 setup을 구축했습니다. 이를 통해 IFT와 추론의 차이점을 명확히 하고, 데이터 표현 및 모델 용량의 영향을 분리하는 데 중점을 두었습니다. 연구에 사용된 모델의 크기는 0.5B부터 14B까지 다양하게 설정되었으며, 이는 과제의 특성에 따라 성능의 변화를 분석하는 데 도움을 주었습니다.

- **Performance Highlights**: 추론 모델은 종종 IFT의 성능 한계를 초과하며, 특히 다단계 문제 해결이 요구되는 수학 및 코딩 과제에서 두드러진 성능 향상을 보여주었습니다. 반면, IFT 모델은 훈련 및 추론에서 비용 효율적인 경로로 남아 있습니다. 연구 결과는 거대한 모델이 요구되지 않는 과제 및 환경에서는 IFT 모델이 유리하다는 실용적인 지침을 제시합니다.



### Context Parametrization with Compositional Adapters (https://arxiv.org/abs/2509.22158)
- **What's New**: 이 연구에서는 CompAs라는 메타-러닝 프레임워크를 소개하여, 컨텍스트로부터 어댑터 파라미터를 구성적으로 변환하는 방법을 제안합니다. 기존의 방법들이 단일 입력 컨텍스트에서 어댑터를 생성하는 데 그쳤다면, CompAs는 여러 정보 조각을 통합하는 필요성을 해결합니다. 이를 통해 어댑터를 대수적으로 결합할 수 있으며, 긴 프롬프트를 재처리하지 않고도 명령어, 시연 또는 회수된 패시지를 매끄럽게 조합할 수 있는 장점을 제공합니다.

- **Technical Details**: CompAs는 파라미터 공간에서 어댑터를 조합하여 입력 공간에서의 컨텍스트 연결 효과를 재현하는 것을 목표로 합니다. 이는 교사-학생 모델 설정을 통해 달성되며, 학생 LM은 개별 컨텍스트에 대해 생성된 어댑터의 합과 쿼리 토큰만으로 교사 LM의 출력 결과를 근사하도록 학습됩니다. 이러한 방법은 특수한 메타-러닝 기술과 결합된 auxiliary compositionality와 reconstruction 손실을 통해 이루어집니다.

- **Performance Highlights**: 다양한 다중 선택 및 추출 질문 응답 과제에서 CompAs는 기존의 ICL 및 다른 어댑터 생성 방법들보다 우수한 성능을 보였습니다. 특히 더 많은 입력이 주어질 때 효율적인 외부 증거 통합 능력이 두드러졌습니다. 이 연구는 어댑터 생성의 조합 가능성을 구체화하며, LLM 배포의 확장성을 위한 실용적이고 효율적인 대안을 확립했습니다.



### Mixture of Detectors: A Compact View of Machine-Generated Text Detection (https://arxiv.org/abs/2509.22147)
Comments:
          20 pages, 3 figures

- **What's New**: 이 논문은 기계 생성 텍스트 감지 기법을 탐구하며, 주목할 만한 데이터셋 BMAS English를 도입합니다. 이 데이터셋은 인간의 텍스트와 기계 생성 텍스트를 구별하기 위한 이진 분류 및 다중 클래스 분류에 사용됩니다. 또한, 기계 생성 텍스트의 생성자를 식별하고, 적대적 공격(adversarial attack)에 대한 내성을 높이는 방법도 제안합니다.

- **Technical Details**: 연구의 세 가지 주요 접근법에는 이진 및 다중 클래스 분류, 적대적 강인성(adversarial robustness), 혼합 텍스트 경계 감지가 포함됩니다. 각 접근법을 위해 전용 데이터셋을 구축하고 다양한 모델을 학습시켜 비교 분석이 가능합니다. 특히, 첫 번째 접근법은 인간의 텍스트와 AI의 텍스트를 구분할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 여러 신문법적 공격(syntactic attack) 시나리오에서 평가되었으며, 인간과 AI 생성 텍스트 간의 경계 감지의 정확성과 안정성을 입증했습니다. 또한, 광범위한 모델 아키텍처 및 평가 메트릭을 통해 실험적인 결과의 일반화 가능성을 확인했습니다. 이 연구는 MGTD(Machine-Generated Text Detection) 분야에서 의미 있는 기여를 할 것으로 예상됩니다.



### From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinemen (https://arxiv.org/abs/2509.22144)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문은 Multiround Adaptive Chain-of-Thought Compression (MACC) 프레임워크를 제안하여, Chain-of-Thought( CoT) 추론의 길이를 효과적으로 줄여주는 방법을 소개합니다. MACC는 각 입력에 대해 최적의 압축 깊이를 정하기 위한 점진적인 압축 전략을 사용합니다. 이를 통해 CoT의 평균 정확도가 5.6% 향상되며, CoT의 길이는 평균 47 토큰이 단축되고 지연 시간도 크게 줄어듭니다.

- **Technical Details**: MACC는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Chain-of-thought 생성, (2) 다단계 진행 압축, (3) 다중 작업 파인튜닝입니다. 이 과정에서 입력 질문에 대해 모델은 전체 추론 경로를 생성하고, 각 라운드에서 중복되거나 장황한 단계를 제거하면서 중요한 정보를 보존합니다. 또한, training set의 특징을 기반으로 test-time 성능을 예측할 수 있는 성능 추정 가설을 제안합니다.

- **Performance Highlights**: MACC는 모든 모델에 걸쳐 효과적인 모델 선택 및 예측을 가능하게 하며, 이전의 방법들보다 특정 입력에 따라 성능 저하가 적습니다. 이 접근법은 성능과 효율성을 동시에 달성하는 것이 가능하다는 것을 보이며, 파인튜닝 없이도 정보 생성과 함께 성능을 예측할 수 있습니다. 최적의 압축 전략을 선택하는 데 필요한 시간과 비용을 절감할 수 있는 방법을 제시하고 있습니다.



### NFDI4DS Shared Tasks for Scholarly Document Processing (https://arxiv.org/abs/2509.22141)
Comments:
          Accepted at the RDI4DS 2025 Workshop

- **What's New**: 이 논문에서는 독일 국가 연구 데이터 인프라(NFDI4DS) 하에 개발 및 호스팅되는 12개의 공유 작업에 대한 최신 개요를 제시합니다. 이러한 작업들은 학술 문서 처리 분야의 다양한 도전 과제를 포함하며, 연구 데이터 인프라의 발전에 기여하고 있습니다. 일반적으로 잘 알려진 장에서 개최되어 더 많은 참여를 유도하고 있습니다.

- **Technical Details**: 공유 작업은 특정 문제에 대해 공유 데이터셋 및 메트릭스를 사용하여 계산 방법을 비교하는 공동체 주도의 도전 과제입니다. NFDI4DS는 데이터 사이언스 및 인공지능의 연구 데이터 생명주기를 지원하는 독일 컨소시엄으로, FAIR 원칙 및 투명성과 재현성에 중점을 두고 있습니다. 이는 연구 데이터 인프라 구축을 위해 상호 운용 가능하고 접근 가능한 국가 연구 데이터 인프라를 육성하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 여러 공유 작업에서 기존의 기법을 초월하는 성과들이 나타났습니다. 예를 들어, SOMD 2025에서는 F1 점수 0.89를 기록하며 성과를 보였고, ClimateCheck에서는 Recall@10 점수 0.66을 달성했습니다. 이러한 성과들은 FAIR 연구 데이터 인프라 구축을 위한 방법론적 발전을 촉진합니다.



### Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding (https://arxiv.org/abs/2509.22134)
- **What's New**: 이 논문에서는 Group Tree Optimization (GTO)라는 새로운 훈련 알고리즘을 제안하여, 저항상태(decoding-time) 트리 정책과 훈련 간의 불일치를 해결하고자 합니다. 기존의 방식이 단일 그리디(draft) 경로만 최적화하는 문제를 가지고 있는 반면, GTO는 트리 기반 초안을 직접 최적화하는 방식을 통해 디코딩 효율성을 높입니다. 또한, 이 알고리즘은 훈련 목표를 실제 디코딩 절차와 일치시켜 초안 모델이 디코딩 시간의 효율성을 직접 향상시키도록 합니다.

- **Technical Details**: GTO는 두 가지 주요 구성 요소로 구성됩니다: (i) Draft Tree Reward는 목표 모델 하의 초안 트리의 기대 수용 길이를 측정함으로써 디코딩 성능을 직접적으로 평가하는 샘플링 없음(no sampling) 목표입니다; (ii) Group-based Draft Policy Training은 현재 모델과 고정된 참조 초안 모델의 트리 간 대조를 통해 디비아스(debiased)된 그룹 표준화 이점을 형성하는 안정적인 최적화 방식입니다. 이 방식은 PPO 스타일의 서브러겟을 적용하여 강력한 업데이트를 보장합니다.

- **Performance Highlights**: GTO는 대화(MT-Bench), 코드(HumanEval), 수학(GSM8K)과 같은 다양한 벤치마크에서 여러 개의 LLM(예: LLaMA-3.1-8B, LLaMA-3.3-70B)에서 성능을 검증하였으며, 수용 길이를 7.4% 증가시키고 이전의 최첨단 EAGLE-3에 비해 7.7%의 추가 속도 향상을 달성했습니다. 이러한 성과는 GTO가 디코딩 효율성 향상에 어떻게 기여하는지를 잘 보여줍니다.



### R-Capsule: Compressing High-Level Plans for Efficient Large Language Model Reasoning (https://arxiv.org/abs/2509.22131)
- **What's New**: 이번 논문의 핵심은 Reasoning Capsule(R-Capsule) 프레임워크로, 이는 Latent Reasoning의 효율성과 Chain-of-Thought(CoT)의 투명성을 결합하려는 새로운 접근 방식입니다. R-Capsule은 고수준 계획을 소형의 잠재 토큰으로 압축하면서도 실행 단계를 경량화하거나 명확하게 유지합니다. 이 혁신적인 접근은 정보 병목 원칙(Information Bottleneck)에서 영감을 받아 효과적으로 작업의 최소성을 유지하면서도 충분한 효용을 보장합니다.

- **Technical Details**: R-Capsule 프레임워크는 두 가지 주요 단계를 포함합니다: 잠재 계획 단계와 조건부 실행 단계입니다. 첫 번째는 고수준의 전략적 계획을 압축된 잠재 표현으로 생성하고, 두 번째로는 이 잠재 계획에 조건을 두어 저수준의 실행 단계를 진행하는 방식입니다. 우리의 아키텍처는 디코더 전용 트랜스포머를 기반으로 하며, Reasoning Capsule을 생성하고 활용하는 메커니즘을 포함합니다.

- **Performance Highlights**: R-Capsule은 복잡한 기준에서 명시적 CoT 보다 경쟁적이거나 개선된 정확도를 보이며, 적은 수의 가시 토큰 및 감소된 지연(latency)으로 결과를 제공합니다. 실험을 통해 R-Capsule이 계획과 실행의 압축에서 어떻게 유익한 인덕티브 편향(inductive biases)을 보존하는지를 확인했습니다. 전반적으로, R-Capsule은 정확도와 효율성을 동시에 달성하여, 더 적은 자원으로도 높은 성능을 제공합니다.



### FoodSEM: Large Language Model Specialized in Food Named-Entity Linking (https://arxiv.org/abs/2509.22125)
Comments:
          To appear in the Proceedings of the 28th International Conference on Discovery Science (DS 2025)

- **What's New**: 이 논문은 FoodSEM을 소개합니다. FoodSEM은 음식 관련 온톨로지에 대한 이름-개체 연결(named-entity linking, NEL)을 위한 최첨단 오픈소스 대형 언어 모델(large language model, LLM)로, 현재의 일반 목적 언어 모델이나 맞춤형 도메인 모델로는 해결할 수 없는 문제를 효과적으로 해결합니다. 이 모델은 텍스트에서 언급된 음식 관련 개체를 FoodOn, SNOMED-CT, Hansard 분류법 등 여러 온톨로지에 연결합니다.

- **Technical Details**: FoodSEM은 명령-응답(instruction-response, IR) 시나리오를 통해 다양한 온톨로지와 연결된 식품 개체 언급(단일 또는 다단어 시퀀스)을 링크합니다. 이 모델은 풍부한 음식 주제 문헌에서 학습하여 온톨로지와의 연결을 개선합니다. FoodSEM은 고공헌능력(F1 score)에서 일부 온톨로지 및 데이터셋에 대해 98%에 달하는 성과를 달성했습니다.

- **Performance Highlights**: FoodSEM은 0-shot, 1-shot, 5-shot 프롬프트에 대한 비교 분석을 통해 비미세 조정(non-fine-tuned)된 버전보다 뛰어난 성능을 입증했습니다. 이 모델은 GitHub와 Hugging Face에서 공개되어 연구자들이 쉽게 접근할 수 있습니다. FoodSEM의 출현은 음식 관련 NEL에 대한 강력한 기준선을 제공하여 향후 벤치마킹 연구에 기여할 예정입니다.



### Multilingual Vision-Language Models, A Survey (https://arxiv.org/abs/2509.22123)
- **What's New**: 이번 설문조사는 다양한 언어를 아우르는 비전-언어 모델(multilingual vision-language models)에 대해 조사합니다. 31개의 모델과 21개의 벤치마크(benchmarks)에 대한 리뷰를 수행하며, 인코더 전용(encoder-only) 및 생성(generative) 아키텍처를 포함합니다. 언어 중립성(language neutrality)과 문화적 인식(cultural awareness) 간의 주요 긴장을 파악했습니다.

- **Technical Details**: 현재의 훈련 방법(training methods)은 대조 학습(contrastive learning)을 통해 중립성을 선호하는 반면, 문화적 인식은 다양한 데이터(diverse data)에 의존합니다. 평가 벤치마크의 3분의 2는 의미적 일관성(semantic consistency)을 우선시하는 번역 기반 접근법을 사용하고 있습니다. 최근 연구는 문화적으로 기반이 있는(content) 내용을 통합하고 있습니다.

- **Performance Highlights**: 우리는 교차 언어 능력(cross-lingual capabilities)에서 차이를 발견했고, 훈련 목표와 평가 목표 간의 격차를 확인했습니다. 이 연구는 모델의 성능을 평가하는 데 있어 균형 잡힌 접근 방식이 중요하다는 점을 강조합니다.



### Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM (https://arxiv.org/abs/2509.22119)
Comments:
          10 pages, 6 figures, Accepted to ICAIL 2025 (International Conference on Artificial Intelligence and Law)

- **What's New**: 이번 논문에서는 Legal Article Prediction (LAP)에서의 한계를 극복하기 위해 Uni-LAP이라는 새로운 보편적 프레임워크를 제안합니다. 기존의 Supervised Classification Models (SCMs)과 Large Language Models (LLMs)의 장점을 결합함으로써, 법률 기사의 예측 정확도를 높이고 효율성을 강화하는 것을 목표로 합니다. Uni-LAP은 Top-K loss function을 도입하여 후보 법률 기사의 품질을 향상시키고, LLM은 삼단 논법(syllogism-inspired reasoning)을 활용하여 최종 예측을 정교화합니다.

- **Technical Details**: Uni-LAP 프레임워크는 SCM과 LLM 간의 밀접한 협력을 통해 법률 기사 예측을 수행합니다. 먼저 SCM은 Noval Top-K loss function으로 훈련되어 Top-K 후보 기사를 보다 정확하게 생성하며, 이후 LLM은 이를 토대로 최종 예측을 수행합니다. 이 과정에서 LLM은 법률 기사의 복잡성을 반영하기 위해 삼단 논법적 추론을 포함하여 예측 능력을 극대화합니다.

- **Performance Highlights**: 다양한 지역과 언어로 구성된 두 개의 실제 데이터 세트인 유럽 인권 재판소(ECtHR) 데이터 세트와 중국 AI 및 법률 챌린지(CAIL2018) 데이터 세트에서 Uni-LAP을 평가했습니다. 결과적으로 Uni-LAP은 모든 기준선 모델을 지속적으로 초월하는 성능을 보여주어, 효과성과 일반화 가능성을 입증하였습니다. 이를 통해 법률 기사의 예측 정확도를 비약적으로 향상시킬 수 있는 가능성을 확인할 수 있었습니다.



### Think Right, Not More: Test-Time Scaling for Numerical Claim Verification (https://arxiv.org/abs/2509.22101)
Comments:
          Accepted to EMNLP 2025, 19 pages

- **What's New**: 이 논문에서는 복잡한 수치 주장(fact-checking complex numerical claims)을 검증하는 과정에서 대형 언어 모델(LLMs)의 테스트 시간(TTS) 컴퓨팅 스케일링을 체계적으로 탐구합니다. 기존의 LLMs는 수치적 요소(numerical aspects)에 대한 섬세한 이해가 부족하고, 다양한 정보를 통합하는 과정에서 발생하는 reasoning drift 문제로 인해 오해를 불러일으킵니다. 저자들은 이 문제를 해결하기 위해, 선택적으로 복잡성에 따라 TTS를 수행하는 적응형 메커니즘을 도입하여 성능을 크게 향상시키고 있습니다.

- **Technical Details**: 이 연구에서는 LLMs가 다양한 추론 경로(reasoning paths)를 탐색하도록 컴퓨팅 자원을 추가 배분할 수 있는 테스트 시간 스케일링(TTS) 기법을 도입합니다. 이 기술은 다양한 경로를 선택하여 reasoning drift를 완화하는 데 도움을 줍니다. 실험을 통해 저자들은 dedicated verifier model인 VERIFIERFC를 도입하여, 이를 사용한 best-of-N 전략이 기존의 majority-voting 방법들보다 유의하게 뛰어난 성능을 보인다는 것을 입증하였습니다. 또한, 제안된 적응형 TTS 전략은 복잡한 주장에 대해 18.8% 더 나은 성능을 발휘하며, 1.8배 더 높은 효율성을 제공합니다.

- **Performance Highlights**: 저자들은 QuanTemp 데이터셋에서 약 34%의 주장이 reasoning drift 문제를 겪고 있다고 밝혔습니다. 이를 해결하기 위한 새로운 방법으로 TTS를 제안하며, 이는 복잡한 수치 주장의 검증 성능을 대폭 증대시키는 결과를 가져왔습니다. 저자들의 실험 결과에 따르면, 적응형 TTS는 단일 샷(single-shot) 기술 대비 21.27%의 성능 향상을 이루어내며, 이는 LLMs의 사실 검증 효율성을 크게 개선합니다.



### S2J: Bridging the Gap Between Solving and Judging Ability in Generative Reward Models (https://arxiv.org/abs/2509.22099)
- **What's New**: 이 논문은 Generative Reward Models (GRMs)의 기존 연구에서 발견된 중요한 문제인 solve-to-judge gap을 처음으로 식별하였습니다. 이 간극은 GRMs가 특정 쿼리에 대해 올바른 판단을 내리지 못하는 현상을 나타냅니다. 이를 보완하기 위해 Solve-to-Judge (S2J) 접근법을 제안하며, 문제 해결 능력과 판단 능력을 동시에 고려하여 모델의 최적화를 진행합니다.

- **Technical Details**: S2J는 GRMs의 출력에서 문제 해결과 판단 능력을 동시에 활용하여 보상을 부여하는 방식을 사용합니다. 이는 모델이 사용자의 쿼리를 먼저 해결한 후에 평가를 진행하게 만듭니다. 두 가지 시나리오를 통해 쿼리의 정답을 확인하고, 주관적 작업의 경우에는 별도의 scalar reward model을 활용하여 보상을 할당합니다.

- **Performance Highlights**: S2J는 기존 모델 대비 평균 판단 정확도를 67.0%에서 72.7%로 향상시켰으며, solve-to-judge gap을 16.2% 줄이고 판단 성능을 5.8% 개선하였습니다. 특히, S2J는 동일한 기본 모델에 기반했으나 훨씬 적은 데이터로도 최신 성과(SOTA)를 달성했습니다. 또한, S2J는 모델이 내재적으로 갖고 있는 문제 해결 능력을 효과적으로 활용하여 판단 작업의 성능을 높이는데 기여합니다.



### Multilingual Dialogue Generation and Localization with Dialogue Act Scripting (https://arxiv.org/abs/2509.22086)
Comments:
          16 pages, 10 tables, 2 figures, Accepted at EMNLP Main 2025

- **What's New**: 본 연구는 다국어 대화 시스템을 위한 새로운 프레임워크인 Dialogue Act Script (DAS)를 제안합니다. DAS는 추상적인 의도 표현을 바탕으로 다국어 대화를 구조적으로 인코딩하고 생성하며, 단순히 대화를 번역하는 것이 아닌 문화적으로 적합한 대화를 생성하는 것을 목표로 합니다. 이 프레임워크는 대화의 본질적인 의미를 유지하면서 다양한 언어 간에 적절한 현지화를 지원합니다.

- **Technical Details**: DAS는 대화의 의도를 정의된 대화 행위(discourse act)와 매개변수(parameter)의 집합으로 추상화하여 인코딩합니다. 이 구조적 접근 방식은 대화 내용을 추상화하고 현지화하는 과정을 통해, 문화적 편향(anglocentric bias)과 번역에서 발생하는 인위적인 요소(artificial artifacts)를 최소화합니다. DAS는 순차적인 세 단계의 파이프라인을 활용하여 대화 행위와 의미 롤을 포착하여 대화 데이터를 생성합니다.

- **Performance Highlights**: 이 연구는 이탈리아어, 독일어, 중국어로 생성된 DAS 기반 대화를 평가한 결과, 기존 기계 번역 및 인간 번역 데이터보다 문화적 관련성과 일관성에서 우수한 성능을 보였습니다. 특히 DAS로 생성된 대화는 인간 평가에서 일관되게 높은 점수를 받았으며, 자연스러운 대화 생성을 가능하게 하는 데 있어 효과적임을 입증했습니다.



### COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning (https://arxiv.org/abs/2509.22075)
- **What's New**: 이번 연구에서는 CoSpaDi(Compression via Sparse Dictionary Learning)라는 새로운 압축 프레임워크를 제안했습니다. 이 프레임워크는 저랭크(weight approximation) 분해 방법 대신에 더 유연한 구조적 희소(factorization) 분해를 적용하여 모델 압축을 최적화합니다. CoSpaDi는 훈련 과정 없이 압축이 가능하며, 데이터에 기반한 효과적인 최적화 전략을 통해 원본 출력과 가깝게 일치하도록 보장합니다.

- **Technical Details**: CoSpaDi는 각 가중치 행렬을 밀집(dense) 사전(dictionary)와 열 희소(column-sparse) 계수 행렬로 표현하여 여러 하위 공간의 조합을 이용합니다. 이 방식은 각 열이 서로 다른 사전 원자(atoms)로 표현되어 불균형한 특징을 더 잘 수용할 수 있습니다. 또한, 이 방법은 사전 학습 후 양자화(post-training quantization)와 호환되어 메모리 및 대기 시간(latency)을 추가적으로 최적화합니다.

- **Performance Highlights**: 여러 Llama 및 Qwen 모델을 대상으로 한 실험에서, CoSpaDi는 20-50%의 압축 비율에서 기존의 저랭크 방법들과 비교하여 일관된 우수한 성능을 보였습니다. 본 연구 결과는 구조적 희소(dictionary learning)가 기존의 저랭크 접근 방식보다 효율적인 LLM 배포를 위한 강력한 대안임을 입증하였습니다. 이로 인해 CoSpaDi는 훈련 미세조정 없이도 높은 모델 충실도를 유지하면서도 우수한 압축 성능을 제공합니다.



### Fine-tuning Done Right in Model Editing (https://arxiv.org/abs/2509.22072)
- **What's New**: 이 논문에서는 기존의 fine-tuning(파인 튜닝)이 모델 편집(model editing)에 비효율적이라는 오해를 도전합니다. 실험을 통해, 모델 편집에 있어서 깊이 우선(depth-first) 파이프라인의 적합성 문제를 지적하며, 이는 샘플 별 업데이트가 각 편집을 지나치게 최적화하게 하여 간섭(interference)을 초래한다는 주장을 합니다.

- **Technical Details**: 연구팀은 기존의 깊이 우선 파이프라인 대신, 에포크 기반(epoch-based) 방법론을 되돌려 미니 배치(mini-batch) 최적화를 적용하여 모델 편집의 효율성을 크게 향상시켰습니다. 또한, 기존 메서드의 불리한 튜닝 파라미터(tuning parameter) 위치의 영향을 분석하였고, 이를 통해 LocFT-BF라는 로컬화된 편집 방법을 제안합니다.

- **Performance Highlights**: LocFT-BF는 다양한 LLMs(대형 언어 모델)와 데이터셋에서 광범위한 실험을 통해 기존 최고 성능(state-of-the-art) 방법들 보다 현저히 우수한 성과를 보여줍니다. 특히, 100K 편집과 72B 파라미터 모델에 대해 이전보다 10배 더 많은 편집을 수행하면서도 일반적인 기능을 손상시키지 않는 것으로 확인되었습니다.



### The QCET Taxonomy of Standard Quality Criterion Names and Definitions for the Evaluation of NLP Systems (https://arxiv.org/abs/2509.22064)
Comments:
          39 pages, 7 figures

- **What's New**: 이 논문에서는 동일한 품질 기준 이름(예: Fluency)을 사용하는 NLP 평가 실험이 반드시 동일한 품질 측면을 평가하지 않음을 보여주고 있습니다. 이는 서로 다른 평가를 가진 시스템 품질에 대한 신뢰할 수 있는 결론을 내릴 수 없게 하며, NLP 분야의 과학적 발전을 저해하는 요소로 작용합니다. 따라서, 품질 기준 이름과 정의의 표준 세트를 생성하는 것이 필요하다고 주장합니다.

- **Technical Details**: 이 연구에서는 QCET(Quality Criteria for Evaluation Taxonomy) 품질 기준을 제안하며, NLP에서 보고된 세 가지 평가 조사에서 유래한 표준 품질 기준 이름과 정의를 도출합니다. 이러한 기준은 계층 구조로 구성되어 있으며, 각 부모 노드는 하위 노드의 공통 측면을 포착합니다. 이는 평가의 비교 가능성을 확보하고, 새로운 평가 설계를 안내하며, 규제 준수를 평가하는 데 도움을 줍니다.

- **Performance Highlights**: QCET의 주요 활용법으로는 기존 평가의 비교 가능성을 확립하고, 새로운 평가의 설계를 안내하며, 규제 준수를 평가하는 것이 포함됩니다. 이는 NLP 평가의 일관성을 높이고, 연구의 신뢰성을 향상시키기 위한 노력이 될 것입니다. 이러한 자원은 NLP 분야의 품질 평가 기준의 명확화를 통해, 향후 연구에 중요한 기초자료로 기능할 것입니다.



### RedNote-Vibe: A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media (https://arxiv.org/abs/2509.22055)
- **What's New**: 대규모 언어 모델(LLMs)의 확산으로 인해 소셜 미디어 플랫폼에서 AI 생성 텍스트(AIGT)가 증가하고 있습니다. 이에 따라 사용자 참여에 의해 동적 콘텐츠가 형성되는 독특한 과제가 생겨났습니다. 본 연구에서는 5년간의 데이터셋인 RedNote-Vibe를 소개하며, 이는 사전 LLM 기간부터 2025년 7월까지의 사용자 참여 지표와 타임스탬프를 포함하여 AIGT의 시간적 동적 및 사용자 상호작용 패턴을 연구하기 위한 것입니다.

- **Technical Details**: RedNote-Vibe는 중국 소셜 미디어 플랫폼인 Xiaohongshu에서 수집되었습니다. 이 데이터셋은 주제, 태그, 타임스탬프와 같은 메타데이터를 포함하며, AIGT의 다양한 스타일을 평가하기 위해 17개의 LLM에서 생성된 AI 텍스트도 포함됩니다. 또한, 우리는 PLAD(심리언어학적 AIGT 탐지 프레임워크)를 제안하여 심리언어학적 특성을 활용하여 AIGT를 탐지합니다.

- **Performance Highlights**: PLAD는 뛰어난 탐지 성능을 보여주며, 인간 생성 콘텐츠와 AI 생성 콘텐츠를 구별하는 서명에 대한 통찰력을 제공합니다. 또한, 이 프레임워크는 언어적 특성과 소셜 미디어 참여 간의 복잡한 관계를 밝혀냅니다. 연구 결과는 AI 채택의 시간적 추세와 인간 저자와 AI 생성 콘텐츠 간의 참여 패턴 차이를 포괄적으로 분석하는 데 기여합니다.



### Fuzzy Reasoning Chain (FRC): An Innovative Reasoning Framework from Fuzziness to Clarity (https://arxiv.org/abs/2509.22054)
Comments:
          Accepet by EMNLP 2025 Findings (11 pages, 1 figures)

- **What's New**: 이 논문에서는 모호한 텍스트와 불확실성을 처리하기 위한 Fuzzy Reasoning Chain (FRC) 프레임워크를 소개합니다. FRC는 대형 언어 모델(LLM)의 의미적 사전 정보를 지속적인 퍼지 멤버십(continuous fuzzy membership)으로 통합하여 확률 기반 추론과 퍼지 멤버십 추론 간의 명확한 상호작용을 생성합니다. 이 방법을 통해 전통적인 방법으로는 처리할 수 없는 혼란스러운 입력을 명확하게 이해 가능한 결정으로 전환할 수 있습니다.

- **Technical Details**: FRC는 감정 분석의 표준 단계별 추론 절차를 따르며, 이산 확률 할당을 지속적인 퍼지 멤버십으로 대체하여 기존의 체인 오브 썸 논리(Chain-of-Thought, CoT) 접근법을 확장합니다. 확률 기반 추론에서 퍼지 멤버십 추론으로의 전환은 논문의 핵심 방법론 혁신으로, 감정의 보다 세련되고 강력한 표현을 가능하게 합니다. FRC는 여러 모델 규모 간의 지식 전이 가능성을 높이는 동시에 안정적인 추론을 보장합니다.

- **Performance Highlights**: FRC는 감정 분석 태스크에서 이론적 분석과 경험적 결과를 통해 안정적인 추론을 보장하고 모델 성능을 향상시킵니다. 실험 결과는 FRC가 혼란스럽고 모호한 입력에서 뛰어난 성능을 보여준다고 강조합니다. 이러한 발견은 FRC가 더 나은 해석 가능성과 견고함으로 미세하고 모호한 표현을 관리하기 위한 일반적인 메커니즘을 제공한다는 것을 보여줍니다.



### Taxonomy of Comprehensive Safety for Clinical Agents (https://arxiv.org/abs/2509.22041)
Comments:
          EMNLP 2025 Industry

- **What's New**: 이 논문에서는 임상 챗봇 응용 프로그램에서의 안전성 문제를 다루기 위해 TACOS (TAxonomy of COmprehensive Safety for Clinical Agents)라는 새로운 분류 체계를 도입합니다. TACOS는 21개의 세부 클래스를 가지며, 안전 필터링(safety filtering)과 도구 선택(tool selection)을 통합한 사용자 의도 분류(user intent classification) 단계를 제공합니다. 기존 방법들이 임상 분야의 다양한 요구를 충족하는 데 부족했던 점을 해결하려는 목적이 있습니다.

- **Technical Details**: TACOS는 임상 및 비임상 질문을 포괄하는 다양한 안전(threshold) 기준과 외부 도구 의존성을 명시적으로 모델링합니다. 연구자들은 TACOS로 주석이 달린 데이터셋을 정리하고, 이를 바탕으로 광범위한 실험을 수행했습니다. 이 분류 체계의 세부적인 구조는 임상 에이전트 설정에 특화된 안전성을 향상시키기 위한 중요한 정보를 제공합니다.

- **Performance Highlights**: 실험 결과는 TACOS의 유용성을 입증하며, 임상 에이전트의 응답 질 향상에 기여할 수 있는 통찰력을 제공합니다. 또한, 모델 학습에 사용되는 데이터의 분포(distribution)와 사전 학습(pretrained knowledge)의 중요성에 대한 새로운 이해를 제공합니다.



### From Outliers to Topics in Language Models: Anticipating Trends in News Corpora (https://arxiv.org/abs/2509.22030)
Comments:
          presented at ICNLSP 2025; to appear in the ACL Anthology; received the Best Full Paper Award

- **What's New**: 이 논문은 주제가 모델링에서 흔히 소음으로 취급되는 아웃라이어가 어떻게 새로운 주제의 약한 신호로 작용할 수 있는지를 연구합니다. 최신 언어 모델의 벡터 임베딩을 활용하고 축적 클러스터링 접근 방식을 통해 프랑스어 및 영어 뉴스 데이터 세트에서 이러한 아웃라이어의 진화를 추적합니다. 연구 결과, 아웃라이어가 시간이 지남에 따라 일관된 패턴으로 응집력 있는 주제로 발전하는 경향이 있음을 보여줍니다.

- **Technical Details**: 프레임워크로는 HDBSCAN과 BERTopic을 사용하여 아웃라이어가 클러스터에 통합되는 과정을 분석합니다. 이 연구에서는 오픈 소스 임베딩 모델 9개를 사용하여 뉴스 기사를 고차원 의미 공간으로 변환하고, 클러스터링 과정을 통해 아웃라이어가 주제에 어떻게 기여하는지를 측정합니다. 모델 성능은 Massive Text Embedding Benchmark (MTEB)를 통해 평가되며, UMAP을 활용해 차원 축소를 하여 클러스터링 품질을 향상시킵니다.

- **Performance Highlights**: 아웃라이어의 클러스터 통합 과정에서의 기여 비율을 계산하였고, 이를 통해 아웃라이어가 새로운 주제의 형성이나 기존 주제의 강화에 미치는 영향을 평가했습니다. 결과적으로, 각 모델에서 아웃라이어가 나중에 클러스터에 통합된 비율이 높아지는 것을 확인하였으며, 이는 아웃라이어가 유의미한 역할을 담당할 수 있음을 시사합니다.



### GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2509.22009)
- **What's New**: 이번 논문은 Graph Retrieval-Augmented Generation (GraphRAG)의 한계를 극복하기 위해 GraphSearch라는 새로운 에이전트 기반의 검색 워크플로우를 제안합니다. GraphSearch는 객체 지향적인 다중 채널 검색 전략을 통해 복잡한 질의에 대한 논리적 추론을 개선합니다. 이 방식은 다른 모듈과의 협력을 통해 더 효율적인 지식 검색과 다중 턴 상호작용을 지원합니다.

- **Technical Details**: GraphSearch는 Query Decomposition (QD), Context Refinement (CR), Query Grounding (QG), Logic Drafting (LD), Evidence Verification (EV), Query Expansion (QE)라는 여섯 개의 모듈로 구성됩니다. 이러한 모듈들은 복잡한 질의를 원자적인 서브 질의로 분해하여 그래프 지식 기반에서 정보를 검색하고, 반복적인 논리적 추론을 가능하게 합니다. 또한, 이 방법은 텍스트와 그래프의 구조적 데이터를 동시에 활용하여 고급 논리를 지원합니다.

- **Performance Highlights**: 여섯 개의 다중 홉 RAG 벤치마크에서의 실험 결과, GraphSearch는 기존의 단일 라운드 상호작용 전략보다 항상 더 높은 답변 정확성과 생성 품질을 보여주었습니다. 데이터에 따라 강력한 플러그 앤 플레이 기능이 입증되었으며, 이중 채널 검색 전략과 모듈간의 기여가 실제로 효과적임을 확인하였습니다.



### Black-Box Hallucination Detection via Consistency Under the Uncertain Expression (https://arxiv.org/abs/2509.21999)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전에도 불구하고, 사실과 다른 응답을 생성하는 '환각(hallucination)' 문제는 여전히 해결되지 않고 있습니다. 기존의 환각 탐지 방법들은 외부 자원이나 LLM의 내부 상태에 의존하는 경우가 많습니다. 그러나 본 연구에서는 블랙 박스(Black-Box) 접근 방식을 통해 효과적인 환각 탐지 기술을 개발하고, LLM의 불확실성을 표현하는 모델을 통해 환각 탐지를 간단하게 수행할 수 있는 메트릭을 제안합니다.

- **Technical Details**: 우리는 LLM의 응답에서 사실과 비사실적 응답을 구분할 수 있는 프롬프트를 탐구합니다. 특히, 불확실한 표현과 확실한 표현을 사용하여 LLM의 일관성을 평가하는 방법을 제안합니다. 본 메트릭은 샘플링 없이 단일 응답만으로도 효과적으로 실제 정보를 구별할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 메트릭이 기존의 내부 지식에 의존하는 환각 탐지 방법들보다 모델 응답의 사실성을 더 잘 예측하는 것으로 나타났습니다. 두 개의 오픈 도메인 질의응답 데이터셋에서 (질문, 응답) 쌍을 분석한 결과, 사실적 응답에서 LLM이 일관된 응답을 생성하는 경향이 강하다는 것을 확인했습니다.



### MotivGraph-SoIQ: Integrating Motivational Knowledge Graphs and Socratic Dialogue for Enhanced LLM Ideation (https://arxiv.org/abs/2509.21978)
Comments:
          EMNLP2025 Findings

- **What's New**: 이 논문에서는 학술 아이디어 생성을 위한 새로운 접근법인 MotivGraph-SoIQ를 제안합니다. 이는 대규모 언어 모델(LLMs)의 한계를 극복하기 위해 동기 부여 지식 그래프(Motivational Knowledge Graph)와 소크라틱 대화(Q-Driven Socratic Dialogue)를 통합하는 구조입니다. 이 프레임워크는 아이디어 생성 과정에서 필수적인 기반과 실제적인 개선 단계를 제공하여, LLM의 아이디어 품질을 향상시킵니다. 기존의 방법과 비교하여 커다란 장점을 인증한 결과, 진정성과 실행 가능성을 높이는 데 기여한 것으로 보입니다.

- **Technical Details**: MotivGraph는 문제, 도전 및 해결책이라는 세 가지 주요 노드 유형으로 구성됩니다. 이를 통해 LLM의 아이디어 생성을 위한 동기 부여를 구조적으로 저장하고 있으며, 아이디어 개발 과정에서는 Q-Driven Socratic Ideator가 소크라틱 질문을 활용하여 아이디어를 철저히 수정합니다. 이는 기존의 편향(confirmation bias)을 완화하고 아이디어의 품질을 증진시키는 데 초점을 맞추고 있습니다. 이 방법론은 ICLR25 논문 주제 데이터셋을 사용하여 Evaluations를 통해 검증되었습니다.

- **Performance Highlights**: MotivGraph-SoIQ는 기존의 최첨단 방법들에 비해 명확한 성과를 나타냈습니다. LLM 기반 점수, ELO 랭킹, 휴먼 평가 메트릭스 모든 영역에서 기존대로의 접근보다 평균 10.2% 높은 참신도와 6% 높은 동기 합리성을 달성했습니다. 인간 평가에서도 유사한 성과를 보이며, 참신도에서 7.98% , 동기 합리성에서 5.56%의 향상을 나타내었습니다. 이러한 결과는 MotivGraph-SoIQ의 전체 메트릭에서 지속적으로 개선된 성능을 시사합니다.



### Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration (https://arxiv.org/abs/2509.21946)
Comments:
          9 pages

- **What's New**: 본 논문은 태국의 정치적 맥락에서  정치적 입장 탐지에 있어 대규모 언어 모델(LLMs)의 편향 문제를 다룹니다. 특히, 정치적 대화가 암시적이고 감정적으로 얽힌 환경에서 LLMs들이 보여주는 비체계적 편향, 즉 감정 누출(sentiment leakage) 및 특정 정치적 인물에 대한 편애를 실질적으로 감소시키는 프레임워크인 ThaiFACTUAL을 제안합니다. 이는 모델 튜닝 없이도 정치적 편향을 줄일 수 있는 경량의 방법론입니다.

- **Technical Details**: ThaiFACTUAL는 반사실적 데이터 증강(counterfactual data augmentation) 및 근거 기반(supervision based on rationale) 접근 방법을 활용하여 감정과 입장을 분리합니다. 연구에서는 LLM들의 편향을 정량적으로 분석하기 위해 회수 표준 편차(RStd) 메트릭을 채택하여, 주요 정치 인물에 대한 입장 분류 성능을 평가하였습니다. 이 프레임워크는 기존에 요구되었던 모델 파라미터 접근 없이도 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ThaiFACTUAL은 여러 LLM에서 무작위 상관관계를 크게 줄이고, 제로샷 일반화(zero-shot generalization)를 향상시키며, 공정성을 개선하는 것으로 나타났습니다. 특히, 태국 정치적 대화를 다룰 때 발생하는 감정의 복잡성과 문화적 뉘앙스를 고려하여, 더 나은 성능을 보이는 것을 입증하였습니다. 이러한 결과는 정치적으로 민감한 상황에서의 AI 모델의 신뢰성을 높이는 데 기여할 것입니다.



### Why Chain of Thought Fails in Clinical Text Understanding (https://arxiv.org/abs/2509.21933)
- **What's New**: 이 연구는 임상 텍스트 이해를 위한 CoT(Chain-of-Thought) 프롬프팅의 효과성에 대한 대규모 체계적 연구를 처음으로 제시합니다. 95개의 최신 대형 언어 모델(LLM)을 87개의 실제 임상 작업에 대해 평가하였으며, 이를 통해 CoT가 임상 텍스트 작업에서 성능을 일관되게 저하시킬 수 있다는 점이 드러났습니다. 특히, 모델의 성능 저하는 약한 모델에서 더 두드러지는 경향을 보였습니다.

- **Technical Details**: 연구에서는 9개 언어와 8개 작업 종류에 걸쳐 임상 텍스트 작업을 평가하며, 두 가지 프롬프팅 전략인 제로샷(zero-shot)과 CoT를 활용하였습니다. LLM 평가에 있어 정확성과 추론 길이, 임상 개념 정렬, 그리고 오류 프로파일을 보다 세밀하게 분석하였습니다. 특히, CoT가 일관되게 정확도를 저하시키는 방식과 그 이유를 메커니즘 분석을 통해 조명하였습니다.

- **Performance Highlights**: 연구의 주요 발견으로는 CoT 프롬프팅이 명백한 투명성을 증가시키지만, 임상 텍스트 이해에서는 정량적으로 성능 저하를 초래한다는 점이 포함됩니다. 임상 사례 분석 결과, CoT의 실패가 더 긴 추론 연결고리와 관련이 있으며, 잘못된 답변과 연관된 언어적 특성도 확인되었습니다. 마지막으로, 오류 분류 체계를 개발하여 LLM의 임상 활용에 대한 안전한 지침을 제공하고 있습니다.



### SimulSense: Sense-Driven Interpreting for Efficient Simultaneous Speech Translation (https://arxiv.org/abs/2509.21932)
Comments:
          \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 본 논문에서는 SimulST(동시 음성 번역) 시스템을 위한 새로운 프레임워크인 SimulSense를 제안합니다. 이 방법은 인간 통역사의 의사 결정을 모방하여, 입력 음성을 지속적으로 인식하고 새로운 의미 단위(Sense Unit)를 감지할 때 번역을 생성하기 위해 쓰기 결정을 내립니다. SimulSense는 최신 기술을 사용하여 품질-지연(latency) 트레이드오프를 향상시키고, 기존의 시스템보다 최대 9.6배 빠른 의사 결정 속도를 자랑합니다.

- **Technical Details**: SimulSense의 핵심 기술은 Sense Units Detector(SUD)로, 의미론적 경계를 식별하는 역할을 합니다. 이 모델은 음성 인코더의 출력을 바탕으로 의사 결정을 내림으로써, 비싼 LLM(대형 언어 모델) 추론 없이도 즉각적인 번역 시작이 가능하게 합니다. 또한, 이는 기존의 LLM 접근 방식에 비해 더 나은 품질을 유지하며, 파라미터가 고정된 규칙 기반 접근법을 넘어서 적응형 의사 결정 정책을 개발하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, SimulSense는 두 개의 최신 기준 시스템과 비교하여 우수한 품질-지연 트레이드오프를 달성하였습니다. 또한, 이 시스템은 실시간 효율성을 실질적으로 개선하였으며, 의사 결정 과정에서 사용되는 자원을 최적화함으로써 모든 번역 사용 사례에서 뛰어난 성능을 발휘합니다. 이러한 결과는 인간 통역사의 행동을 모방한 결과이며, 번역 품질 측면에서도 높은 신뢰성을 보장합니다.



### AutoSCORE: Enhancing Automated Scoring with Multi-Agent Large Language Models via Structured Component Recognition (https://arxiv.org/abs/2509.21910)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문에서는 AutoSCORE라는 다중 에이전트 LLM 프레임워크를 제안하여 자동 채점 시스템의 한계를 극복하고자 합니다. AutoSCORE는 학생의 응답에서 채점 기준의 관련 구성 요소를 추출하고 이를 구조화된 형태로 인코딩하여 최종 점수를 부여합니다. 이 과정은 루브릭을 기반으로 하여 인간과 유사한 채점 과정을 따르며 해석 가능성과 신뢰성을 향상합니다.

- **Technical Details**: AutoSCORE는 두 개의 핵심 에이전트로 구성되어 있습니다. 첫 번째는 Scoring Rubric Component Extraction Agent로, 학생의 응답에서 관련 구성요소를 식별하여 구조화된 출력을 생성합니다. 두 번째는 Scoring Agent로, 이러한 표현을 바탕으로 최종 점수를 부여하며, 선택적으로 검증 및 피드백 에이전트를 포함할 수 있어 품질 관리를 강화합니다.

- **Performance Highlights**: AutoSCORE는 ASAP 벤치마크의 네 가지 데이터 세트에서 평가되었으며, 여러 작업과 루브릭에 걸쳐 채점 정확도, 인간-기계 간의 합의, 오류 메트릭에서 일관되게 향상된 성과를 보였습니다. 특히 복잡한 다차원 루브릭에서 두드러진 이익을 보여주었고, 상대적으로 작은 LLM에서도 큰 성과를 달성했습니다.



### A Large-Scale Dataset and Citation Intent Classification in Turkish with LLMs (https://arxiv.org/abs/2509.21907)
Comments:
          Submitted to IEEE UBMK 2025 International Conference on Computer Science and Engineering

- **What's New**: 이번 연구에서는 터키어의 인용 의도를 분류하기 위한 대규모 데이터셋과 체계적인 분류 방법론을 개발했습니다. 새롭게 공개된 이 데이터셋은 2,650개의 주석이 달린 컴퓨터 과학 논문 샘플로 구성되어 있습니다. 기존 In-Context Learning (ICL) 방식의 한계를 극복하기 위해 자동화된 프롬프트 최적화 프레임워크인 DSPy를 사용하여 인용 의도를 효과적으로 분류합니다.

- **Technical Details**: 연구는 인용 문장의 추출을 위해 GROBID 라이브러리를 활용한 CEX 모듈을 사용합니다. 인용 의도 분류는 Web of Science (WoS) 플랫폼의 다섯 가지 카테고리 체계를 기반으로 하여 진행되며, 전문적인 주석 작성을 위한 웹 인터페이스가 개발되었습니다. 최종적으로 여러 모델의 출력을 집계하는 스택된 앙상블 기법을 통해 91.3%의 정확도를 달성하였습니다.

- **Performance Highlights**: 이 연구는 터키어 NLP 커뮤니티에 강력한 기초 데이터셋과 분류 프레임워크를 제공하여, 향후 정성적 인용 연구의 기반을 마련합니다. 특히, DSPy 프레임워크를 통한 자동화된 프롬프트 최적화 방식은 기존 수동 방식보다 안정적이고 효율적인 성능 향상을 보여주었습니다. 이러한 접근은 과학적 발견 가속화를 위한 지능형 학술 도구 개발에 기여할 것입니다.



### Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts (https://arxiv.org/abs/2509.21892)
- **What's New**: 이번 연구에서는 Elastic Mixture-of-Experts (EMoE)라는 새로운 훈련 프레임워크를 소개했습니다. 전통적인 Mixture-of-Experts (MoE) 모델에서는 훈련 및 추론 시 고정된 수의 전문가(expert)를 사용하는 반면, EMoE는 추론 시 활성화된 전문가 수를 유연하게 조정할 수 있어 성능을 향상시킬 수 있습니다. 이는 학습된 전문가 간의 협력이 부족해 성능 저하를 초래하는 문제를 해결하기 위한 접근으로, 스토카스틱(co-activation) 샘플링 기법과 계층적 라우터 손실(hierarchical router loss)을 도입하였습니다.

- **Technical Details**: EMoE의 주요 기술적 요소는 스토카스틱(co-activation) 샘플링과 계층적 라우터 손실입니다. 스토카스틱 샘플링 기법을 통해 다양한 전문가 조합을 훈련 기간 동안 균형 있게 활성화하여 전문가 사이의 협력 능력을 향상시키는 동시에 훈련 부담을 최소화합니다. 또한 계층적 라우터 손실은 KL 발산을 활용해 라우터의 확률 분포를 명확한 계층 체계를 만들어내며, 이를 통해 각 입력 토큰에 대해 최상의 전문가 세트를 선별할 수 있도록 합니다.

- **Performance Highlights**: EMoE를 적용한 결과, 표준 Top-k 모델과 비교하여 훨씬 더 넓은 범위의 활성화된 전문가 수를 지원하며, 이는 2-3배까지 확대될 수 있음을 보여주었습니다. 다양한 컴퓨팅 예산 아래에서 EMoE는 지속적으로 기준선을 초과하는 성능을 발휘하여, 다양한 상황에서 뛰어난 유용성을 자랑합니다. 또한, 스토카스틱 샘플링 및 계층적 라우터 손실이 EMoE의 효과에 필수적임을 실험적으로 입증하였습니다.



### QoNext: Towards Next-generation QoE for Foundation Models (https://arxiv.org/abs/2509.21889)
- **What's New**: 이번 논문에서는 사용자 경험(User's Experience)에 중점을 두고, 기존의 평가 방법들이 상호작용 중 사용자 만족도를 포착하지 못했던 한계를 지적합니다. 새로운 프레임워크인 QoNext를 통해, 품질(Quality) 기반의 평가를 Foundation Models에 적용하여, 사용자 경험의 기초를 구성하는 요소들을 파악합니다. 이 연구는 사용자가 모델과의 상호작용에서 어떻게 느끼는지를 분석하기 위해 제어된 실험을 통해 데이터베이스를 생성하고, 예측 모델을 훈련합니다.

- **Technical Details**: QoNext는 QoE(Quality of Experience) 원칙을 적용하여 여러 차원에서 사용자 경험을 측정합니다. 정보 밀도(information density), 콘텐츠 정확성(content accuracy), 출력 속도(output speed), 지연(latency duration), 그리고 위치(position) 등 다섯 가지 세부 차원을 연구합니다. 이를 통해 수집된 데이터를 활용하여 사용자 경험을 예측할 수 있는 회귀 모델(regression models)을 개발하였고, 이를 통해 QoE 기반 데이터베이스를 구축했습니다.

- **Performance Highlights**: QoNext의 분석 결과, 콘텐츠 정확성이 전체 사용자 경험에 가장 큰 영향을 미치는 요소로 나타났으며, 출력 속도가 그 뒤를 이었습니다. QoNext 모델은 인간의 평가와 높은 일관성을 보이며, SRCC가 0.79에 도달하였습니다. 이는 QoE 원칙을 모델 평가에 적용하는 것이 가능하다는 것을 입증하는 결과를 제공합니다.



### No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping (https://arxiv.org/abs/2509.21880)
- **What's New**: 이 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR) 프레임워크에서 zero-variance prompts의 유용성을 주장합니다. 기존의 기법들이 서로 다른 보상을 갖는 입력에만 초점을 맞춘 반면, 저자들은 모든 응답이 동일한 보상을 받는 경우에도 유의미한 피드백을 제공할 수 있다는 점을 강조합니다. 이를 통해 RL-ZVP라는 새로운 알고리즘을 제안하며, 이는 token-level 정보를 활용하여 정책 최적화를 개선하는 데 집중합니다.

- **Technical Details**: RL-ZVP는 zero-variance prompts에서 학습 신호를 추출하기 위해 두 가지 주요 직관에 기반합니다: (i) 같은 그룹 내에서 잘못된 응답이 없더라도 정확한 응답에 대해 보상을 주어야 하며, (ii) 보상 또는 패널티의 정도는 샘플링된 토큰의 특성에 따라 결정됩니다. 논문에서는 정확성과 정밀도를 높이는 두 가지 속성이 방향성과 크기라는 점을 강조하고, 이를 통해 트레이닝의 효과성을 크게 향상시킵니다.

- **Performance Highlights**: RL-ZVP는 여섯 개의 수학 추론 기준에서 GRPO보다 평균 4.00점, Pass@8에서 4.28점의 정확도를 개선하였습니다. 특히, RL-ZVP는 AIME25에서 최대 8.66점, OlympiadBench에서 7.77점의 성능 향상을 기록하며, zero-variance prompts를 필터링하는 다른 기준 모델들을 꾸준히 초월하는 결과를 보였습니다. 이러한 결과는 zero-variance prompts가 RLVR에서 학습 신호의 귀중한 자원이 될 수 있음을 입증합니다.



### LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals (https://arxiv.org/abs/2509.21875)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템에서 발생하는 환각 현상을 측정하는 새로운 프레임워크인 LUMINA를 제안합니다. LUMINA는 외부 컨텍스트 활용과 내부 지식 활용을 정량화하여 이 신호들을 기반으로 환각을 감지합니다. 기존 방법들은 과도한 하이퍼파라미터 튜닝을 요구하는 반면, LUMINA는 레이어에 구애받지 않고 신뢰성 있는 측정을 가능하게 합니다.

- **Technical Details**: LUMINA는 두 가지 주 신호, 즉 외부 컨텍스트 활용(External Context Utilization)과 내부 지식 활용(Internal Knowledge Utilization)을 통해 환각을 감지합니다. 외부 컨텍스트 활용은 예측 분포와 임의의 문서 간의 불일치(discrepancy)를 측정하여 외부 정보 의존성을 평가합니다. 내부 지식 활용은 변환기(transformer) 레이어를 따라 모델의 상태 변화와 예측 토큰을 추적하여 평가합니다.

- **Performance Highlights**: LUMINA는 HalluRAG와 같은 일반적인 RAG 환각 기준에서 실험을 수행한 결과, 0.9 이상의 AUROC 점수를 달성하며 기존 방법들보다 최대 13% 개선된 성능을 보였습니다. 특히, 외부 컨텍스트 점수와 내부 지식 점수가 낮은 환각과 강하게 연관되어 있다는 해석 가능성을 제공합니다. LUMINA는 다양한 검색 설정에서도 견고함을 유지하여 실용성과 효과성을 입증하였습니다.



### Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations (https://arxiv.org/abs/2509.21870)
Comments:
          This manuscript has been submitted to IEEE Journal of Selected Topics in Signal Processing (JSTSP) for review. Until the moment I submitted the manuscript to arXiv, we haven't received any review comments from JSTSP

- **What's New**: 본 논문은 Low-Rank Adaptation (LoRA)의 비선형 확장인 LoRAN을 제안합니다. LoRAN은 파라미터 수를 증가시키지 않으면서도 저랭크 업데이트에 경량 변환을 적용하여 표현 능력을 향상시킵니다. 추가적으로, Sinter라는 새로운 sine 기반 활성화 함수를 도입하여 구조적 왜곡을 추가하면서도 파라미터 수를 유지합니다. 실험 결과, LoRAN은 QLoRA보다 일관된 성능 향상을 보여 주었습니다.

- **Technical Details**: LoRAN은 LoRA의 저랭크 프로젝션에 비선형 함수를 적용하여 가중치 업데이트의 표현 능력을 강화합니다. Sinter는 제어된 진동성 perturbation을 추가하여 모델의 표현성과 안정성을 동시에 개선합니다. 논문은 다양한 자연어 처리 작업에서 LoRAN의 성능을 검증하며, 기존의 Sigmoid, ReLU, Tanh와 같은 활성화 함수들에 비해 Sinter의 장점을 입증합니다.

- **Performance Highlights**: LoRAN은 리소스가 제한된 환경에서도 기존 LoRA보다 뛰어난 성능을 보여주며, 특히 고차원 업데이트를 더 잘 근사할 수 있습니다. 이 모델은 다양한 기본 모델에 대해 높은 유연성과 일반화 능력을 발휘하였으며, 전체적인 성능이 충분한 Fine Tuning에 근접하게 개선되었습니다. 실험 범위가 확장되어 여러 작업에서 LoRAN과 Sinter의 효과가 더욱 명확히 드러났습니다.



### KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues (https://arxiv.org/abs/2509.21856)
- **What's New**: 본 논문에서는 최초의 Multi-Turn Long-Form Question Answering (MT-LFQA) 평가 시스템인 KnowMT-Bench를 소개합니다. 이 벤치마크는 의료, 금융, 법률과 같은 지식 집약적 분야에서 LLM(대규모 언어 모델)의 성능을 체계적으로 측정하기 위해 설계되었습니다. 이를 통해 기존의 단일 턴 평가에서 나타나는 한계점을 극복하고, 대화의 맥락을 고려하여 LLM의 정확성과 정보 전달 효율성을 평가할 수 있습니다.

- **Technical Details**: MT-LFQA는 모델이 이전 대화 기록을 기반으로 여러 사실을 합성하여 최종 답변을 제공해야 하는 개방형 질문 답변 작업입니다. 논문의 연구에서 벤치마크는 801개의 증거 기반 LFQA 인스턴스를 활용하고, 모델이 정리가 필요 없는 대화 이력을 생성하도록 요구합니다. 평가 과정에서는 인간 검증이 포함된 자동화된 파이프라인을 통해 사실 능력과 정보 전달 효율성을 분석합니다.

- **Performance Highlights**: 실험 결과, 다중 턴 맥락에서 LLM의 사실 능력과 정보 전달 효율성이 저하되는 경향이 있음을 발견했습니다. 특히, 자체 생성된 대화 이력에서 발생하는 맥락적 잡음이 사실 능력 저하의 주된 원인으로 나타났습니다. 이러한 문제를 해결하기 위해 RAG(검색 증강 생성)를 활용한 대안이 효과적임을 입증하였고, 이는 다중 턴 상황에서의 LLM 성능 향상의 가능성을 보여줍니다.



### Following the TRACE: A Structured Path to Empathetic Response Generation with Multi-Agent Models (https://arxiv.org/abs/2509.21849)
- **What's New**: TRACE는 Task-decomposed Reasoning for Affective Communication and Empathy의 약자로, 감정적 소통을 위한 새로운 프레임워크입니다. 이 프레임워크는 공감(empathy)을 구조화된 인지 프로세스로 모델링하고, 감정 인식(emotion recognition), 인과 분석(causal analysis), 전략적 계획(strategic planning), 응답 합성(response synthesis)의 네 단계로 작업을 분해합니다. TRACE는 이러한 분석을 통해 보다 깊이 있는 공감을 생성할 수 있는 가능성을 제시합니다.

- **Technical Details**: TRACE 프레임워크는 한 사용자의 대화를 순차적으로 처리하여 응답을 깊이 있게 구성하는 멀티 에이전트(multi-agent) 시스템입니다. 각 에이전트는 사용자의 감정을 인식하는 Affective State Identifier (ASI), 감정의 원인을 분석하는 Causal Analysis Engine (CAE), 최적의 의사소통 전략을 선택하는 Strategic Response Planner (SRP), 그리고 최종 응답을 생성하는 Empathetic Response Synthesizer (ERS)로 구성되어 있습니다. 이 구조는 공감 생성에서 필요한 깊이 있는 분석을 체계적으로 제공합니다.

- **Performance Highlights**: TRACE는 자동화 평가 및 LLM 기반 평가에서 강력한 기준선(baseline)을 크게 초과하는 성과를 내었습니다. 실험 결과는 TRACE의 구조화된 접근 방식이 더 능력 있고 해석 가능한 공감형 에이전트를 생성하는 데 유망한 패러다임임을 확신시킵니다. TRACE는 높은 분석 깊이와 표현력을 모두 갖춘 프레임워크로서, 향후 다양한 AI 애플리케이션에서 활용될 수 있을 것으로 기대됩니다.



### Semantic Agreement Enables Efficient Open-Ended LLM Cascades (https://arxiv.org/abs/2509.21837)
Comments:
          EMNLP 2025 Industry Track

- **What's New**: 이 논문은 오픈 엔디드 텍스트 생성을 위한 새로운 접근법으로 세멘틱 어그리먼트(semantic agreement)를 제안합니다. 이는 여러 모델의 출력을 의미적으로 일치시키는 방식으로, 이를 통해 신뢰할 수 있는 생성을 위한 결정을 내리는 데 기여할 수 있습니다. 제안된 방법은 모델의 내부 정보를 요구하지 않으며, 블랙박스 API에서도 작동합니다. 세멘틱 캐스케이드는 대규모 모델에 비해 비용을 40%까지 줄이고 지연 시간을 최대 60%까지 감소시킬 수 있음을 밟히고 있습니다.

- **Technical Details**: 제안된 세멘틱 캐스케이드 방법론은 여러 개의 소형 모델과 하나의 대형 모델을 포함하는 구조로 개발되었습니다. 소형 모델들이 생성한 출력을 기반으로 의미적 합의를 평가하고, 이를 통해 언제 대형 모델로 전환할지를 결정합니다. 저자들은 다양한 유사도 메트릭스를 활용하여 출력 간의 의미적 일치를 정량화하며, BLEURT와 SBERT를 포함한 다양한 기법들을 사용합니다. 이 방법은 훈련이 필요 없고 여러 모델에서 일반적으로 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 세멘틱 캐스케이드는 500M에서 70B 파라미터 모델까지 다양한 모델에서 경쟁력 있는 성능을 보였습니다. 목표 대형 모델에 비해 비용을 40% 절감하면서도 지연 시간을 최대 60% 줄였습니다. 또한, 제안된 방법은 토큰 수준 신뢰성 방법보다 향상된 의사결정을 제공하며, 실제 LLM 배포에 적합한 강력한 기반架構가 되는 것으로 나타났습니다.



### ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models (https://arxiv.org/abs/2509.21826)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 수동적인 텍스트 생성에서 목표 지향적인 도구 사용 에이전트로 진화하는 것을 다룹니다. 강화 학습(RL)을 통해 이러한 도구 사용 정책을 최적화하는 방법을 제시하며, 기존의 희소한 결과 보상 방식의 한계를 극복하기 위해 새로운 리워드 설계를 제안합니다. 데이터를 통해 많은 도구 사용 작업의 안정성을 높이고, 훈련 효율성을 향상할 수 있는 이론적 연관성을 구체화했습니다.

- **Technical Details**: 정책 엔트로피와 도구 사용 작업의 훈련 안정성 간의 이론적 연관성을 확립하고, 구조적 저엔트로피 토큰이 보상의 주요 요소라는 것을 보여줍니다. 제안된 Reshaped Token-level policy gradients (ResT)는 훈련 과정에서 점진적으로 reasoning 토큰의 가중치를 높여 엔트로피 인식 방식으로 정책 기울기를 재구성합니다. 이를 통해 다단계 도구 사용 작업의 수렴을 안정화하고, 구조적 정확성에서 의미적 추론으로 매끄럽게 전환할 수 있게 합니다.

- **Performance Highlights**: BFCL 및 API-Bank 벤치마크에서 ResT가 기존 방법보다 최대 8.76% 향상된 성능을 달성하며 마지막까지 state-of-the-art 결과를 제시했습니다. 4B 기본 LLM에 대해 세부 조정 시, ResT는 단일 턴 과제에서 GPT-4o보다 4.11%, 다단계 과제에서는 1.50% 향상된 성과를 기록했습니다. 이러한 결과는 curriculum-based reshaping 방식이 기존의 정적 보상 가중치보다 최대 4.86% 더 우수함을 시사합니다.



### Can LLMs Solve and Generate Linguistic Olympiad Puzzles? (https://arxiv.org/abs/2509.21820)
Comments:
          To be published in the Proceedings of Main Conference of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)

- **What's New**: 이 논문에서는 언어 퍼즐의 해결 및 생성 과제를 소개합니다. 특히, 고등학생을 대상으로 하는 언어 올림피아드에서 사용되는 퍼즐에 초점을 맞추었습니다. 기존의 벤치마크를 확장하고, 최신의 대규모 언어 모델(LLM)을 분석하여 언어 퍼즐 해결의 성능을 평가하였습니다. LLM이 대부분의 퍼즐 유형에서 사람보다 우수한 성과를 보이는 것을 확인하였지만, 쓰기 체계 및 저조하게 연구된 언어와 관련된 퍼즐에서는 성능이 떨어졌습니다.

- **Technical Details**: 이 논문에서는 LLM의 능력을 분석하기 위해 언어 퍼즐의 새로운 수집체인 LingOly 벤치마크를 소개합니다. 6개의 언어 주제를 포함하여 UKLO(영국 언어 올림피아드)를 위한 퍼즐 세트를 수집하였습니다. LLM을 활용한 퍼즐 해결 및 생성 실험을 통해 언어 퍼즐 디자인의 원리를 LLM 프롬프트에 적용하여 새로운 퍼즐을 생성하는 과정을 설명합니다. 퍼즐의 품질을 평가하기 위한 기준을 개발하며, 이는 언어학적 퍼즐 자체로서의 중요성을 강조합니다.

- **Performance Highlights**: LLM은 대부분의 언어 주제에서 인간 솔버보다 우수한 성과를 나타냈으며, 특히 추론 능력을 갖춘 최신 LLM이 더 나은 결과를 보였습니다. 영문학적 퍼즐 해결 성과는 전반적으로 높은 편이지만,  쓰기 체계와 관련된 퍼즐에서는 성능이 떨어졌습니다. 이를 통해 LLM의 추론 능력 및 한계를 보다 깊이 이해할 수 있었습니다. 이 연구는 언어학에 대한 관심을 확대하고, 퍼즐 생성을 통해 덜 연구된 언어에 대한 지식을 전달하는 데 기여할 것으로 기대됩니다.



### Towards Minimal Causal Representations for Human Multimodal Language Understanding (https://arxiv.org/abs/2509.21805)
- **What's New**: 이번 논문에서는 인과 기반 다중 모달 정보 병목 모델(Causal Multimodal Information Bottleneck, CaMIB)을 제안합니다. 이는 전통적인 likelihood 방식 대신 인과 원리를 활용하여 다중 모달 언어 이해에서의 데이터 편향 문제를 해결하려는 시도입니다. CaMIB는 첫째로, 작업에 불필요한 노이즈를 제거하기 위해 정보 병목(Information Bottleneck) 기법을 적용합니다.

- **Technical Details**: CaMIB 모델은 파라미터화된 마스크 생성기를 사용하여 융합된 다중 모달 표현을 인과(subrepresentation) 및 단축(shortcut) 하위 표현으로 분리합니다. 또한, 기구적 변수 제약(instrumental variable constraint)을 포함시켜 인과 관계의 전역적 일관성을 보장하고, 랜덤하게 인과 및 단축 특성을 재조합하여 인과 추정을 안정화합니다. 이러한 기법은 모델이 분포 변화(distribution shift) 상황에서도 인과 관계를 우선시하도록 교육하는 데 도움을 줍니다.

- **Performance Highlights**: CaMIB는 다중 모달 감정 분석, 유머 탐지, 비꼬기 감지와 같은 다양한 MLU 작업에서 기존 방법들을 능가함을 보여줍니다. 특히 분포 변화 상황에서 두드러진 성과를 나타내며, 이러한 성능 개선에 대한 이론적 및 경험적 분석을 통해 CaMIB의 해석 가능성과 신뢰성을 더욱 강조합니다.



### Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies (https://arxiv.org/abs/2509.21801)
- **What's New**: 이번 연구는 Simultaneous Machine Translation (SiMT)의 액션 공간을 확장하여 SEMTENCE_CUT, DROP, PARTIAL_SUMMARIZATION 및 PRONOMINALIZATION이라는 네 가지 새로운 적응형 행동을 도입했습니다. 이러한 행동은 실시간에서의 재구성, 생략 및 단순화를 가능하게 하여 의미의 충실도를 유지합니다. 연구진은 이 행동들을 디코더 전용 대형 언어 모델(LLM) 프레임워크에 구현하였고, 행동 인식 차원에서 훈련 참조를 구축하였습니다.

- **Technical Details**: SiMT 시스템은 기본적으로 인코더-디코더 구조를 기반으로 하여 품질과 지연 시간 사이의 균형을 유지하는 데 어려움을 겪고 있습니다. 연구에서는 디코더 전용 LLMs인 GPT-4o와 Qwen3-8B를 기반으로 하여, Salami 기법 및 동적 컨텍스트 프롬프트와 같은 여러 강력한 기준과의 비교를 진행하였습니다. 또한 단어 정렬 및 소스 타임스탬프를 기반으로 고유의 지연 인식 TTS 파이프라인을 개발하였습니다.

- **Performance Highlights**: ACL60/60 영어-중국어 및 영어-독일어 벤치마크에서 실험을 수행한 결과, 제안된 프레임워크는 의미적 메트릭(예: COMET-KIWI)을 꾸준히 개선하고, 참조 번역 및 Salami 기반 기준보다 지연 시간(Average Lagging)을 낮추는 성과를 보였습니다. 특히, DROP 및 SENTENCE_CUT을 결합했을 때 유창성과 지연 시간 간의 최상의 균형을 이루는 것으로 나타났습니다.



### Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignmen (https://arxiv.org/abs/2509.21798)
Comments:
          Under review on ICLR 2026;Work in progress;

- **What's New**: 본 연구에서는 문화적 인식(cultural awareness)을 평가하기 위한 Cultural Awareness Reward modeling Benchmark (CARB)를 제안합니다. 이 벤치마크는 10개의 다양한 문화와 4개의 문화적 분야를 아우릅니다. 현재의 보상 모델(reward models, RMs)의 한계를 드러내며, 문화적 특성을 효과적으로 반영하는 새로운 평가 기준을 마련하는 데 기여합니다.

- **Technical Details**: CARB는 문화적 지식, 가치, 안전 및 언어학과 같은 4가지 주요 도메인에서 10개의 문화적 특성을 포괄하도록 설계되었습니다. 연구에서는 각 문화에 대해 인간이 선별한 질문을 기반으로 생성된 응답들을 활용하여 8,576개의 고품질 BoN 세트를 작성했습니다. Think-as-Locals라는 방법을 통해 보상 모델이 표면적 특성 대신 진정한 문화적 뉘앙스를 이해하도록 유도합니다.

- **Performance Highlights**: 실험 결과, CARB에서의 성과는 다양한 문화적 정렬 작업에서의 보상 모델 성능과 긍정적인 상관관계를 나타냈습니다. 현재의 보상 모델들은 문화적 인식 측면에서 제한적이며, 겉보기에 기반한 스퍼리어스 상관관계를 보이는 경향이 있습니다. 본 연구에서 제안한 방법은 이러한 문제들을 감소시키고, 문화 인식 능력을 강화하는 데 효과적임을 입증했습니다.



### Navigating the Impact of Structured Output Format on Large Language Models through the Compass of Causal Inferenc (https://arxiv.org/abs/2509.21791)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 구조화된 출력이 생성 품질에 미치는 영향을 분석합니다. 기존 연구들은 구조화된 출력이 생성 품질에 긍정적이거나 부정적이라는 편향된 결론을 내렸지만, 본 연구는 인과 추론을 통해 보다 명확한 분석을 제공합니다. 이를 위해 새로운 데이터셋인 Enhanced Last Letter Concatenation(ELLC)를 도입하여 LLM의 능력을 평가하는 데 도전합니다.

- **Technical Details**: 연구에서 채택한 인과 추론 방법론은 무작위 대조 시험(RCTs)과 관찰적 데이터 수집을 포함하여, 구조화된 출력 형식이 LLM의 생성에 미치는 영향을 평가합니다. 주요 변수로는 instruction, output format, LLMs의 generation이 있으며, 이들 각각은 인과 관계를 규명하는 데 중요한 역할을 합니다. 연구는 JSON, XML 및 YAML과 같은 세 가지 구조화된 형식과 전통적인 비구조화 형식을 비교하여 분석합니다.

- **Performance Highlights**: 연구 결과, 대다수의 추론 시나리오에서 구조화된 출력 형식이 GPT-4o의 생성에 인과적 영향을 미치지 않음을 밝혀냈습니다. 그러나 일부 시나리오에서는 특정 사용자의 지시 사항에 의해 영향을 받는 복합적인 효과가 나타났습니다. 마지막으로, 제안된 구조 분석 파이프라인은 구조화된 출력 이외의 모듈이 LLM에 미치는 영향을 평가할 수 있는 가능성을 보여줍니다.



### SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation (https://arxiv.org/abs/2509.21777)
Comments:
          Generative Recommender, Recommendation System, Information Retrieval

- **What's New**: 이번 논문은 	extit{SynerGen}이라는 새로운 생성 추천 모델을 소개하여 개인화 검색과 추천을 통합하는 혁신적인 접근 방식을 제시합니다. 이전의 검색과 추천 시스템은 각기 다른 최적화 목표와 구조로 인해 비효율적이었으나, 	extit{SynerGen}은 회귀 모델을 사용해 이를 단일 목표로 통합하여 성능을 향상시킵니다. 이 모델은 사용자 행동 시퀀스로 훈련되며, 검색과 순위를 동시에 최적화하여 두 작업 간 상호작용을 개선합니다.

- **Technical Details**: 	extit{SynerGen}은 단일 생성을 위한 척도를 제시하여 검색과 추천을 통합합니다. 이 모델은 InfoNCE 손실을 사용하여 검색을 최적화하고, 하이브리드 포인트와 페어와이즈 손실을 통해 순위를 매깁니다. Transformer 구조에서도 시간 정보를 포함할 수 있는 새로운 Rotary positional embedding을 사용하여 시간적 신호를 효과적으로 반영합니다. 이러한 구조로 인해 모델은 단일 백본 내에서 검색과 순위를 최적화하며, 이는 추가적인 엔지니어링을 줄여줍니다.

- **Performance Highlights**: 	extit{SynerGen}은 기존의 생성 추천 모델 및 통합 검색 추천 기준선들에 비해 현저히 개선된 성능을 보여줍니다. 이 모델은 검색과 추천 모두에서 우수한 성능을 발휘하며, 고도화된 사용자 경험을 제공합니다. 실험 결과, 	extit{SynerGen}은 산업 규모의 통합 정보 접근을 위한 유니버설한 백본 모델로 자리매김할 가능성을 보여줍니다.



### Thinking with Sound: Audio Chain-of-Thought Enables Multimodal Reasoning in Large Audio-Language Models (https://arxiv.org/abs/2509.21749)
- **What's New**: 최근 대형 오디오-언어 모델(LALMs)은 음성 번역 및 오디오 Q&A와 같은 다양한 오디오 이해 작업에서 뛰어난 성능을 보여주고 있습니다. 그러나 복잡한 음향 시나리오에서의 도전적인 오디오 추론 작업에서는 상당한 한계를 보이고 있습니다. 본 논문에서는 이러한 제한을 해결하기 위해 Thinking-with-Sound(TwS)라는 새로운 프레임워크를 도입하여 LALMs가 음성 신호로 적극적으로 사고할 수 있게 합니다.

- **Technical Details**: TwS는 언어적 추론과 즉각적인 오디오 도메인 분석을 결합하여 LALMs에 오디오 CoT를 제공합니다. 기존 모델들이 오디오를 정적 입력으로 간주하는 것과 달리, TwS는 모델이 음성 신호를 사용하여 수치 분석 및 디지털 조작을 수행할 수 있도록 합니다. 이를 통해 모델은 의미적이고 일관된 다중 모드 사고 과정을 더 잘 생성할 수 있습니다.

- **Performance Highlights**: MELD-Hard1k라는 새로운 강인성 벤치마크를 통해 TwS의 효과를 실험하였으며, 실험 결과 기존 LALMs의 정확도가 50% 이상 감소하는 경우가 많았습니다. TwS를 통해 경량 모델에서도 24.73%의 절대 정확도 향상을 달성하였으며, 모델 크기가 커질수록 성능 개선이 두드러지게 나타났습니다. 이러한 결과는 TwS가 오디오 이해 시스템을 개발하는 새로운 방향을 제시하고 있음을 보여줍니다.



### Self-Speculative Biased Decoding for Faster Live Translation (https://arxiv.org/abs/2509.21740)
- **What's New**: 이번 연구에서는 Self-Speculative Biased Decoding이라는 새로운 추론 패러다임을 제안하여 동시 번역과 같은 스트리밍 응용 프로그램에서 대형 언어 모델(LLM)의 출력 생성을 효율적으로 개선하고자 합니다. 기존의 방식과 달리, 이 방법은 새롭게 생성된 출력이 아닌 최신 출력을 초안으로 사용하여 계속 확장되는 입력 컨텍스트를 처리합니다. 이렇게 함으로써, 불필요한 재 생성(re-generation) 과정을 줄이고 처리 속도를 높일 수 있습니다.

- **Technical Details**: Self-Speculative Biased Decoding은 새로운 입력에 대한 출력을 생성할 때, 이전의 스트리밍 출력을 직접 검증하고 불일치하는 지점에서 과정(데코딩)을 재개하는 방식을 사용합니다. 여기서 중간 출력 기록을 재사용함으로써, 초안 생성 단계(draft computation)를 완전히 생략할 수 있습니다. 또한, 검증 단계에서 초안 토큰에 대한 편향(bias)을 적용하여 초안 수용률을 높임으로써 처리 속도를 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안하는 방법은 기존의 오토리그레시브(auto-regressive) 방식과 비교하여 최대 1.7배의 속도 향상을 달성하였으며, 출력 품질에도 타협을 두지 않았습니다. 뿐만 아니라, flickering 현상을 80%까지 줄이는 효과를 보이며 사용자 경험을 개선하는데 기여했습니다. 이러한 방식은 다양한 스트리밍 응용 프로그램에서 널리 활용될 수 있는 가능성을 지니고 있습니다.



### How Accurate Are LLMs at Multi-Question Answering on Conversational Transcripts? (https://arxiv.org/abs/2509.21732)
Comments:
          Accepted by EMNLP 2025 Industry Track

- **What's New**: 이번 연구에서는 질문 응답(Question Answering, QA) 시스템을 위한 대규모 언어 모델(LLM)의 활용 가능성을 탐구합니다. 기존의 고급 LLM인 GPT-4o와 비교하여 정확성이 뛰어난 80억 개의 매개변수를 가진 공개 LLM의 성능을 분석했습니다. 연구 결과, 공개 LLM이 산업 환경에서 투명하고 비용 효과적으로 배포될 잠재력을 지니고 있다는 사실을 보여줍니다. 또한, 이러한 접근 방식은 많은 질문을 하나의 실행에서 처리할 수 있는 효율성을 제공합니다.

- **Technical Details**: 이 연구는 대화 형식의 긴 문맥을 기반으로 하는 여러 질문에 대한 구조적 응답 생성을 다룹니다. 실험에서는 소규모 모델과 대형 모델을 포함한 다양한 LLM을 평가하였으며, 특히 여러 질문을 하나의 프롬프트에서 처리하는 배치 프롬프트(batch prompting) 접근 방식을 활용합니다. 각 응답은 질문에 대한 답변뿐만 아니라 인간 리뷰를 위한 참조 정보도 포함됩니다.

- **Performance Highlights**: 실험 결과, 학습된 공개 LLM이 GPT-4o의 정확성을 초과할 수 있다는 사실을 발견했습니다. 이는 8억 개의 매개변수로도 달성 가능하며, 이러한 성과는 산업 응용에서의 효율적인 모델 선택과 구성에 있어 중요한 정보를 제공합니다. 방대한 QA 파이프라인 구현에 적합한 모델 선택에서 유의미한 방향성을 제시합니다.



### ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation (https://arxiv.org/abs/2509.21730)
- **What's New**: 최근 연구에서는 적극적이고 개인화된 AI 어시스턴트를 개발하기 위해 ProPerSim이라는 새로운 시뮬레이션 기반 작업을 도입했습니다. 이는 A user agent가 다양한 개인적 특성을 가진 가상의 환경 내에서 AI 어시스턴트와 상호작용하며, 어시스턴트는 주어진 상황과 개인의 선호에 맞는 권장 사항을 제공하는 방식을 따릅니다. 이 작업을 통해 프로액티비티(proactivity)와 개인화(personalization)를 통합하여 사용자 만족도를 높이는 새로운 접근 방식이 제안됩니다.

- **Technical Details**: ProPerSim은 사용자 에이전트가 가상의 가정 환경에서 AI 어시스턴트와 상호작용하게 하여, 실제 추천이 사용자 맥락 및 선호와 적합성을 바탕으로 이루어지도록 합니다. 사용자 에이전트는 여러 특성을 가진 페르소나로 설계되어 있으며, 이를 통해 AI 어시스턴트는 다양한 상황에서 적절한 타이밍에 적절한 추천을 제공하도록 학습합니다. 여기서 추천은 사용자 에이전트의 평가를 기반으로 하여 지속적으로 조정됩니다.

- **Performance Highlights**: ProPerAssistant는 32개의 다양한 페르소나에 대해 실험을 통해 지속적인 학습과 사용자 피드백을 통해 전략을 조정하며 사용자 만족도를 높이고 있습니다. 처음에는 평균 성과 점수가 2.2였으나 시간이 지남에 따라 점차적으로 3.3로 상승하여 적시에 적합한 추천을 제공할 수 있는 능력을 갖추게 됩니다. 이러한 실험 결과는 실제 상황에서 개인화와 프로액티비티를 결합한 AI 어시스턴트의 가능성을 보여줍니다.



### Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieva (https://arxiv.org/abs/2509.21710)
Comments:
          28 pages, 17 figures

- **What's New**: 이번 논문은 Think-on-Graph 3.0 (ToG-3)라는 새로운 프레임워크를 소개하며, Multi-Agent Context Evolution and Retrieval (MACER) 메커니즘을 통해 기존의 RAG (Retrieval-Augmented Generation) 접근 방식의 한계를 극복하고자 합니다. ToG-3는 Chunk-Triplets-Community 이질적 그래프 인덱스의 동적 구성 및 정제를 통해, Evolving Query (진화하는 쿼리)와 Evolving Sub-Graph (진화하는 서브 그래프)의 이중 진화 메커니즘을 도입하여 정확한 증거 검색을 수행합니다. 이를 통해 고정된 그래프 인덱스를 사용하는 기존 방법의 제한 사항을 개선하여, 특히 경량 LLM과 함께 사용할 경우 심층적이고 정밀한 추론을 가능하게 합니다.

- **Technical Details**: ToG-3의 핵심 혁신은 다중 에이전트 시스템을 기반으로 하여, Constructor, Retriever, Reflector, Responser 에이전트가 증거 검색, 답변 생성, 충분성 반성 및 쿼리와 서브 그래프의 진화를 반복적으로 수행하는 것입니다. 이 방식은 기계적 추론의 초점을 맞추어, 실제 쿼리에 적응하여 그래프 인덱스를 동적으로 구축할 수 있도록 합니다. 이 시스템은 특히 자원 제약이 있는 오프라인 환경에 적합하며, 경량의 오픈 소스 LLM을 RAG 시스템의 백본으로 사용할 수 있게 합니다.

- **Performance Highlights**: ToG-3는 복잡한 멀티 홉 추론 벤치마크에서 최고 평균 Exact Match 및 F1 스코어를 기록하며, Broad Reasoning Tasks에서도 전반적인 우수한 성능을 보여줍니다. 이를 통해 기존 모델들과 비교해 우수한 결과를 달성하며, 광범위한 추론 작업에서도 경쟁력을 유지합니다. 실험 결과는 고정 그래프의 사용에서 비롯된 한계를 극복하고, 복잡한 정보 통합 및 심층 추론을 보다 효과적으로 지원하는 데 도움이 될 것입니다.



### GRAB: A Risk Taxonomy--Grounded Benchmark for Unsupervised Topic Discovery in Financial Disclosures (https://arxiv.org/abs/2509.21698)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Generative AI in Finance

- **What's New**: 본 논문에서는 GRAB라는 금융 위험 분류를 위한 공개 벤치마크를 제안합니다. GRAB는 8,247건의 10-K 제출 자료에서 1.61M 문장을 포함하며, 수동 주석 없이도 효율적으로 금융적 의미가 있는 위험 카테고리를 회복할 수 있는 기능을 제공합니다. 이 벤치마크는 193개의 용어를 21개의 세분화된 유형으로 정리한 위험 분류 체계에 기반하여 구성되었으며, 이는 금융 데이터 분석 및 투자 의사결정에 중요한 정보를 제공합니다.

- **Technical Details**: GRAB는 Hofeditz 세분화 체계에 기반하여 193개의 위험 용어를 포함하여 디스코스 자료에서 21개의 세부 카테고리로 구분합니다. 이 구조를 통해 약한 감독(weak supervision)을 통해 위험 카테고리를 정의하고 문장 수준에서 평가할 수 있도록 도와줍니다. 평가 메트릭은 Accuracy, Macro-F1, Topic BERTScore 및 주제의 효율적인 수를 제공합니다.

- **Performance Highlights**: GRAB는 다양한 주제 모델의 성능을 비교하기 위해 설계되었으며, 기존 모델들과 비교하여 성능을 정량적으로 평가할 수 있는 기준을 마련합니다. 주목할 점은 GRAB가 높은 정확도와 topic quality를 달성할 수 있도록 하는 구조와 메트릭 설정입니다. 이를 통해 금융 리스크 카탈로그의 거버넌스를 위한 중요한 역할을 수행할 것으로 기대됩니다.



### ReviewScore: Misinformed Peer Review Detection with Large Language Models (https://arxiv.org/abs/2509.21679)
- **What's New**: 이 연구는 AI 학회에서 동료 평가의 질이 감소하고 있는 문제를 다루고 있습니다. 저자들은 잘못된 전제가 포함된 '약점(weaknesses)'과 논문에서 이미 답변된 질문('questions')을 통해 저품질 리뷰를 식별하는 새로운 기준인 ReviewScore를 소개합니다. 연구 결과, 약점의 15.2%와 질문의 26.4%가 잘못된 정보로 판별되었습니다.

- **Technical Details**: 저자들은 인간 전문가들이 ICLR 리뷰를 분석하여 리뷰 품질을 평가하기 위한 두 가지 새로운 기준인 질문의 답변 불가능성과 약점의 사실성을 제안합니다. ReviewScore는 이 기준을 5점 척도로 적용하여 평가합니다. 또한, 자동화된 논증 재구성 엔진을 개발하여 모든 명시적 및 암시적 전제를 추출합니다.

- **Performance Highlights**: 저자들은 8개의 최신 LLM을 사용하여 ReviewScore의 자동 평가와 인간 평가자 간의 일치를 검증했습니다. Advanced ReviewScore는 Base ReviewScore보다 유의미한 성능 향상을 보여주었고, 인간-모델 간의 적당한 합의도 관찰되었습니다. 이 연구는 Misguided reviews를 탐지하고 효율적으로 처리할 수 있는 가능성을 제시합니다.



### Towards Transparent AI: A Survey on Explainable Language Models (https://arxiv.org/abs/2509.21631)
- **What's New**: 이 논문은 언어 모델(Language Models, LMs)의 해석 가능성을 높이기 위한 설명 가능 인공지능(Explainable Artificial Intelligence, XAI) 접근법을 종합적으로 검토합니다. 기존의 XAI 방법론이 LMs의 복잡한 구조적 다양성과 진화하는 능력을 도출하는 고유한 도전 과제를 반영하지 못하는 경향이 있음을 지적하며, 이러한 차별화된 문제를 해결하기 위한 체계적 프레임워크를 제안합니다. 각 XAI 방법을 encoder-only, decoder-only, encoder-decoder 아키텍처에 따라 분류하고, 각 방법의 강점과 한계를 분석합니다.

- **Technical Details**: 본 논문은 LMs의 XAI 방법을 엔코더와 디코더 아키텍처에 기반하여 분류합니다. Encoder-only 모델(BERT, RoBERTa 등)은 양방향 self-attention을 활용하여 입력을 처리하고, Decoder-only 모델(GPT-4, LLaMA-2 등)은 원인 self-attention을 통해 텍스트를 생성합니다. Encoder-decoder 모델(T5, BART 등)은 두 가지를 통합하여 정보의 흐름을 추적하며, 다양한 설계 선택에 따른 특징과 한계를 강조합니다.

- **Performance Highlights**: 이 연구는 LMs를 대상으로 한 XAI 방법들에 대한 비교 분석을 제공하여, 기존의 해석 가능성 접근법의 구조화된 관점을 제시합니다. 각 아키텍처의 해석 가능성 문제를 명확히 하고, XAI 기법을 통해 평가할 수 있는 신뢰성과 개연성(predictability)에 대한 체계적인 시각을 제공합니다. 마지막으로, LMs의 투명하고 신뢰할 수 있는 XAI 방법 개발을 위한 개방된 연구 과제와 미래 방향을 제시합니다.



### OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja's Ru (https://arxiv.org/abs/2509.21623)
- **What's New**: 이번 연구에서는 OjaKV라는 새로운 프레임워크를 소개합니다. OjaKV는 온라인 서브스페이스 적응을 결합한 전략적 하이브리드 저장 정책을 통해 KV 캐시의 압축 문제를 해결합니다. 이 프레임워크는 모든 토큰을 균일하게 압축하는 것이 최적이 아님을 인식하고, 중요한 첫 번째 및 최신 토큰의 고정밀도를 유지하는 동시에, 나머지 중간 토큰에 대해서는 저차원 압축을 적용합니다.

- **Technical Details**: OjaKV는 Oja의 알고리즘을 사용하여 온라인 주성분 분석(PCA)을 통해 프로젝션 기반의 저차원 어레이를 점진적으로 적응시키는 방법을 채택합니다. 이는 프롬프트 채우기 단계에서 포괄적인 업데이트를 수행하고 디코딩 중에는 경량의 주기적 업데이트를 통해 서브스페이스가 진화하는 컨텍스트와 정렬되도록 합니다. 이 방식은 FlashAttention과 같은 현대의 주의 모듈과 완벽하게 호환되어 실제 긴 컨텍스트 추론에서 실용성을 높입니다.

- **Performance Highlights**: OjaKV는 다양한 벤치마크에서 뛰어난 성능을 보여주며, 특히 제로샷 정확도를 유지하거나 개선합니다. 특히 긴 컨텍스트에 대한 성능 향상이 두드러지며, 복잡한 추론이 요구되는 작업에서 온라인 서브스페이스 적응의 중요성이 강조됩니다. 따라서 OjaKV는 메모리 효율적인 긴 컨텍스트 추론을 위한 실용적이고 플러그 앤 플레이 솔루션으로 자리잡았습니다.



### Multi-Objective Reinforcement Learning for Large Language Model Optimization: Visionary Perspectiv (https://arxiv.org/abs/2509.21613)
Comments:
          3 pages, 1 figure, accepted by ECAI MODeM 2025

- **What's New**: 이 논문은 Multi-Objective Reinforcement Learning (MORL)의 새로운 세분화 체계를 소개하고 LLM(대형 언어 모델) 최적화에 적용했을 때의 장점과 단점을 분석합니다. MORL 방법의 필요성을 강조하며, 개인화 기능을 수용하는 효율적이고 유연한 접근 방식의 개발을 제안합니다. 또한, 다양한 목표 관계의 영향을 다루는 MORL 벤치마킹 프레임워크의 비전을 제시합니다.

- **Technical Details**: MORL은 LLM 최적화를 위한 명시적이고 분해된 목적을 정의하기 위해 주로 스칼라 기반 RL을 사용합니다. 문서에서는 메타-정책(Meta-policy) MORL 방법이 전통적인 MORL 접근 방식의 비효율성과 유연성 부족 문제를 해결할 수 있는 가능성을 보여줍니다. 여러 정책을 학습하고 결합하는 바이레벨 학습(bi-level learning) 패러다임의 개발이 향후 연구 방향으로 강조됩니다.

- **Performance Highlights**: MORL 방법의 종합적인 벤치마킹 분석이 필요하며, 성능, 안정성, 적응력 및 설명 가능성과 같은 다양한 평가 메트릭스를 통해 LLM의 개선된 솔루션을 도출할 수 있다고 논합니다. 메타-정책 방법은 편향된 선호를 처리할 수 있는 가능성이 있지만 아직 충분히 탐구되지 않았으며 많은 MORL 방법이 의사결정 작업에 제한적으로 응용되고 있습니다.



### "Be My Cheese?": Assessing Cultural Nuance in Multilingual LLM Translations (https://arxiv.org/abs/2509.21577)
- **What's New**: 이번 파일럿 연구는 최신의 다국어 AI 모델이 영어의 비유적 언어(figurative language)인 관용구(idioms)와 유머(puns)를 다양한 글로벌 언어로 번역하는 데 있어 지역화(localisation) 능력을 탐구합니다. 기존의 LLM(대규모 언어 모델) 번역 연구에 초점을 맞추어, 문화적 적절성과 전체적인 지역화 품질을 강조하여 마케팅 및 전자상거래와 같은 실용적인 응용 분야에서 중요한 요소들을 분석했습니다.

- **Technical Details**: 총 20개 언어의 24개 지역 방언에서 생성된 e-commerce 마케팅 이메일의 LLM 번역 샘플 87개가 평가되었습니다. 각 목표 언어에 능통한 심사위원들은 원문의 톤(tone), 의미(meaning), 의도된 청중(intended audience)에 대한 충실도(faithfulness)를 기반으로 정량적 평가와 정성적 피드백을 제공했습니다.

- **Performance Highlights**: 연구 결과, 주요 모델들은 일반적으로 문법적으로 올바른 번역을 생성하지만, 문화적으로 미세한 언어는 여전히 명확한 개선 분야로 드러났습니다. 높은 자원(global language) 대역에서도 비유적 표현과 말장난에 대한 오역이 빈번하게 발생했으며, 이는 기계 번역 품질의 가장 신뢰할 수 있는 예측 인자가 데이터 양이 아니라는 점을 도전적으로 제시합니다.



### Vision Language Models Cannot Plan, but Can They Formalize? (https://arxiv.org/abs/2509.21576)
- **What's New**: 이번 논문은 시각 언어 모델(VLM)을 기반으로 한 새로운 접근법인 VLM-as-formalizer를 제안합니다. 이 접근법은 PDDL(Planning Domain Definition Language)을 이용한 형식화를 통해 장기 계획을 효율적으로 수립할 수 있도록 합니다. 연구에 따르면, VLM-as-formalizer는 모델이 끝-끝(end-to-end) 계획을 생성하는 것보다 훨씬 우수한 성능을 보입니다.

- **Technical Details**: 논문에서는 VLM-as-formalizer의 프레임워크를 형성하기 위해 다섯 가지 파이프라인을 설계하여 1회 샷(one-shot)의 오픈 어휘(open-vocabulary) PDDL 형식을 생성하는 문제를 다룹니다. 입력값은 이미지 시퀀스, 자연어 지시어, 그리고 PDDL 도메인 파일로 구성되며, 이를 통해 모델은 목표에 도달하기 위한 실행 가능 계획을 도출합니다. 또한 시각적 탐지 문제가 주된 병목 현상으로 지적되며, 이는 VLM이 필요한 객체 관계를 충분히 포착하지 못하게 합니다.

- **Performance Highlights**: VLM-as-formalizer는 기존의 데이터셋과 새로운 Blocksworld-Real 도전 과제를 이용하여 평가되었습니다. 이 연구 결과는 VLM-as-formalizer가 장기적인 시각-언어 계획을 위한 훨씬 강력하고 일반화 가능한 패러다임임을 보여줍니다. 그러나 여전히 시각 감지에 대한 한계가 존재하여, 인터미디어리(representations)인 캡션이나 장면 그래프는 성능 개선에 대한 미비한 결과를 보이며 향후 연구 방향을 제시합니다.



### Comparative Personalization for Multi-document Summarization (https://arxiv.org/abs/2509.21562)
- **What's New**: 이번 논문에서는 사용자 맞춤형 다문서 요약(Personalized Multi-Document Summarization, MDS)의 중요성을 강조합니다. 특히, 사용자의 선호도 차이를 파악하여 맞춤형 요약을 효율적으로 생성하기 위한 새로운 프레임워크인 ComPSum을 제안합니다. ComPSum은 사용자의 프로필 문서와 다른 사용자의 문서를 비교하여 사용자의 스타일과 콘텐츠 방향성을 반영한 체계적인 분석을 수행합니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 선호 차원인 글쓰기 스타일(writing style)과 내용 집중(content focus)을 고려하여 사용자의 고유한 특성을 분석합니다. ComPSum은 사용자가 선호하는 스타일을 즉각적으로 파악하고 이를 바탕으로 개인화된 요약을 생성합니다. 또한, 기존 요약 평가 메트릭인 ROUGE와 같은 방법이 적용되지 않는 점을 고려하여, AuthorMap이라는 새로운 평가 프레임워크를 제안합니다.

- **Performance Highlights**: ComPSum의 성능 평가를 위해 PerMSum이라는 개인화된 MDS 데이터셋을 구축하였으며, 이를 통해 ComPSum이 강력한 기준선(Strong Baselines)을 능가하는 성과를 보여주었습니다. AuthorMap을 사용하여 ComPSum의 일관된 개선을 확인하였으며, 이는 요약의 관련성(relevance)과 사실성(factuality) 품질을 유지하면서도 달성된 결과입니다. 이들의 연구 결과는 멀티-도큐먼트 요약 분야에서 개인 맞춤형 접근방식의 가능성을 제시합니다.



### Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution (https://arxiv.org/abs/2509.21557)
Comments:
          Accepted at NeurIPS 2025 LLM Evaluation Workshop

- **What's New**: 이번 논문은 신뢰할 수 있는 대형 언어 모델(LLMs)이 의료, 법률, 학계 및 금융과 같은 고위험 영역에서 인간 검증 가능한 출처를 인용해야 한다는 중요성을 강조합니다. 특히, 모델이 인용을 생성하는 G-Cite 방식과 초안을 작성한 후 인용을 추가하는 P-Cite 방식의 두 가지 패러다임을 소개합니다. 이 연구는 두 가지 접근 방식을 평가하고 신뢰성과 투명성을 고려하여 적절한 선택을 할 수 있도록 돕습니다.

- **Technical Details**: G-Cite(Generation-Time Citation)는 텍스트와 인용을 동시에 생성하고, P-Cite(Post-hoc Citation)는 초안을 작성한 후 인용을 추가하거나 검증합니다. 이 두 가지 접근법은 인용의 시점과 기술적인 작동 방식에서 차이를 보이며, 기존 연구는 이러한 접근 방식을 체계적으로 비교하지 않았습니다. 저자들은 각 접근 방식을 공통 데이터세트와 평가 지표를 사용하여 엄밀히 비교하고, 기본적인 인용 정확도, 정밀도, 회수율, 범위 및 대기 시간과 같은 정량적 메트릭을 통해 성능을 평가했습니다.

- **Performance Highlights**: 결과에 따르면, P-Cite 방식이 더 높은 범위와 경쟁력 있는 인용 정확도를 달성하며, G-Cite 방식은 정확도를 우선시하지만 범위와 속도에는 손해를 보입니다. 연구자들은 고위험 애플리케이션에서 P-Cite 중심의 접근법을 권장하며, 정밀성이 중요한 설정에서는 G-Cite를 남겨두는 것이 좋다고 강조합니다. 또한, 고급 메서드가 실무자가 인용의 정밀도와 범위를 조정할 수 있는 최적화 기능을 제공하므로, 이 점을 명심해야 한다고 말하고 있습니다.



### Domain-Aware Speaker Diarization On African-Accented English (https://arxiv.org/abs/2509.21554)
Comments:
          5 pages

- **What's New**: 본 연구는 아프리카 영어 억양에서의 화자 분리 (speaker diarization) 문제를 다룹니다. 다양한 대화형 및 임상 대화 데이터를 사용하여 여러 시스템의 성능을 엄격한 DER(다이얼리제이션 오류율) 프로토콜 아래 평가하였습니다. 연구 결과는 화자의 억양과 임상 대화에서의 전반적인 일관된 오류가 있음을 시사하며, 이로 인해 기존 모델들이 이러한 환경에 잘 일반화되지 못한다는 점을 강조합니다.

- **Technical Details**: 연구는 임상 대화와 일반 대화 각각에 대해 아프리카 영어 억양 데이터셋인 AfriSpeech-Dialog를 사용하여, Pyannote 모델을 통한 도메인 적응을 진행했습니다. fine-tuning을 통해 오류를 줄였지만, 여전히 임상 대화에서의 성능 차이는 존재합니다. 또한, DER을 기준으로 한 성능 평가를 통해 일반적 대화와 임상 대화 간의 성능 차이를 정보와 함께 제시합니다.

- **Performance Highlights**: 서로 다른 8개의 다이얼리제이션 모델을 평가한 결과, 임상 도메인에서 일반 도메인에 비해 DER이 유의미하게 높은 것을 확인했습니다. Pyannote 모델의 경우 fine-tuning 후에도 여전히 임상-일반 성능 차이가 남아있어, 임상 대화가 본질적으로 어려운 과제임을 나타냅니다. 이러한 결과는 의료와 같은 다중 스피커 환경에서 모델의 성능을 향상시키기 위한 다음 단계로, 균형 잡힌 임상 자원과 중첩 인식이 중요함을 강조합니다.



### Agribot: agriculture-specific question answer system (https://arxiv.org/abs/2509.21535)
- **What's New**: 이번 논문에서는 인도 농업에 적합한 챗봇 시스템을 구축했습니다. 농부의 질문에 대해 날씨, 시장 가격, 식물 보호 및 정부 지원 프로그램 관련 정보를 제공하며, 24시간 언제든지 접근 가능하다는 점이 특징입니다.

- **Technical Details**: 이 시스템은 Kisan Call Center의 데이터를 기반으로 하며, 문장 임베딩 모델(sentence embedding model)을 사용하여 초기 56%의 정확도를 기록하였습니다. 동의어를 제거하고 개체 추출(entity extraction)을 통합함으로써 정확도는 86%로 향상되었습니다.

- **Performance Highlights**: 이 챗봇 시스템을 통해 농부는 농업 관련 정보를 쉽게 얻을 수 있으며, 농업 생산성을 높일 수 있는 기회를 제공합니다. 또한 Call Center 직원들의 업무가 보다 효율적으로 조정될 수 있도록 도와줍니다.



### On Code-Induced Reasoning in LLMs (https://arxiv.org/abs/2509.21499)
- **What's New**: 이번 연구에서는 코드 데이터가 대형 언어 모델(LLMs)의 추론 능력을 향상시키는 데 어떻게 기여하는지를 체계적으로 분석했습니다. 구체적으로는 프로그래밍 언어 10종의 병렬 명령 데이터셋을 구성하고, 코드의 구조적 및 의미적 속성을 선택적으로 방해하는 조절된 변형을 적용했습니다. 또한, LLMs를 다섯 개 모델 가족 및 여덟 개 스케일로 파인튜닝하여 실험을 진행했습니다.

- **Technical Details**: 연구 방법론은 세 단계로 구성됩니다. 첫째, 자연어와 코드의 병렬 지침 데이터셋을 구축하고, 둘째, 코드 지침 데이터에 체계적인 수정을 적용하는 단계입니다. 마지막으로, 다양한 언어 모델을 각 데이터셋 변형에 대해 파인튜닝하고 평가를 실시합니다. 이를 통해 자연어와 코드 기반의 명령을 통합하여 훈련의 효과를 비교할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 3,331회의 실험 결과, LLMs가 수학 및 코드 작업에서 구조적 변형에 보다 취약하다는 것을 발견했습니다. 의도적으로 변형된 코드조차도 표면 수준의 규칙이 유지되면 경쟁력을 유지하는 경향이 있습니다. 특히, Python은 자연어 추론을 선호하는 경향이 있고, Java나 Rust와 같은 저수준 언어는 수학 문제 해결을 더 잘 수행하는 것으로 나타났습니다.



### Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning (https://arxiv.org/abs/2509.21487)
Comments:
          Accepted by the Workshop on Efficient Reasoning, Neurips 2025, 5 pages

- **What's New**: 이 논문에서는 Dual-Head Reasoning Distillation (DHRD)라는 새로운 훈련 방법을 소개합니다. 이 방법은 decoder-only language models (LMs)에 두 개의 추가적인 헤드를 결합하여 성능을 향상시킵니다. DHRD는 supervised learning 시에만 reasoning 헤드를 사용하고, 추론 시에는 pooled classifier를 사용하여 throughput을 극대화합니다.

- **Technical Details**: DHRD는 (i) 마지막 토큰 풀링을 사용하는 pooled classification head와 (ii) auxiliary training loss를 위해 설계된 reasoning tower/LM head를 결합합니다. 이 기술은 입력-합리화-레이블 triplet의 정렬을 통해 모델 성능을 극대화합니다. 훈련 시에는 Teacher 모델을 통해 제공된 합리화를 사용하고, 테스트 시에는 reasoning head를 비활성화하여 속도 저하없이 성능을 제공합니다.

- **Performance Highlights**: DHRD는 SuperGLUE의 7개 작업에서 pooled 기준 대비 0.65%에서 5.47%의 상대적 성능 향상을 보여주었습니다. 특히, entailment/causal 작업에서 더 큰 성능 향상이 있었습니다. 테스트 시에는 DHRD가 CoT decoding보다 96에서 142배 더 높은 throughput을 기록하며, baseline latency가 유지되는 것을 확인했습니다.



### Learning to Reason with Mixture of Tokens (https://arxiv.org/abs/2509.21482)
Comments:
          30 page

- **What's New**: 이번 논문에서는 강화 학습에서 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)의 한계를 해결하기 위해 혼합 토큰 생성(Mixture-of-Token Generation, MoT-G) 방법론을 도입했습니다. MoT-G는 모델이 각 추론 단계에서 여러 토큰에 대한 분포를 유지할 수 있게 하여 기존의 불연속 선택의 제약을 제거합니다. 이를 통해 보다 효율적인 훈련과 탐색이 가능하다는 점에서 큰 의의를 지닙니다.

- **Technical Details**: MoT-G는 기존의 RLVR 프레임워크를 확장하여 연속 혼합 공간에서 작동할 수 있도록 설계되었습니다. 모델은 추론 단계에서 다양한 토큰의 조합을 활용하여 더욱 풍부한 정보처리를 가능하게 하며, 이 구조를 통해 복잡한 문제 해결 과정에서 모델의 불확실성을 유지하고 탐색을 촉진합니다. 전체적 구조는 훈련이 필요 없는 방법들을 포함하여 통합된 형식을 제공합니다.

- **Performance Highlights**: Reasoning-Gym에서의 평가 결과, MoT-G는 표준 디코딩 방법에 비해 7개 작업 중 5~35%의 성능 향상을 보였습니다. 또한 MoT-G 방법은 최종적으로 비슷한 성능을 달성함에 있어 절반의 경로 수만으로도 가능하였으며, 이는 훈련 효율성 증가를 나타냅니다. 추가적인 숨겨진 상태 및 토큰 수준 분석을 통해 MoT-G의 이점이 높은 엔트로피 유지와 탐색 능력에서 기인함을 밝혔습니다.



### A State-of-the-Art SQL Reasoning Model using RLVR (https://arxiv.org/abs/2509.21459)
- **What's New**: 본 연구에서는 Reinforcement Learning with Verifiable Rewards (RLVR)를 적용하여 BIRD라는 데이터 과학 벤치마크에서 조직 특화 지식을 포함하는 맞춤형 추론 모델을 개발했습니다. 이 접근 방식은 일반적인 데이터 과학 작업인 text2sql의 성과를 향상시켰으며, 75.68%의 정확도로 최고 성과를 기록했습니다. 이를 통해 데이터 과학, 비즈니스 인텔리전스 및 코딩 분야에서의 응용 가능성을 보여주고 있습니다.

- **Technical Details**: RLVR은 사전 학습된 LLM을 미세 조정하여 주어진 작업에 맞춰 보상 함수를 사용하여 목표 진리를 측정하는 후처리 학습 패러다임입니다. 이 방법은 SQL 코드 생성과 같이 객관적으로 측정할 수 있는 과제에 적합하며, BIRD 벤치마크를 통해 LLM을 어떻게 RLVR을 통해 미세 조정하는지에 대한 전략을 제시합니다. 본 연구에서 제안하는 방법론은 데이터셋의 특성과 리소스 요구에 따라 다를 수 있는 다양한 접근 방식을 지원합니다.

- **Performance Highlights**: 모델의 성능 측정을 위해 BIRD 벤치마크의 학습 데이터 세트만을 사용하여 75.68%의 정확도를 기록했습니다. 또한, self-consistency(자기 일관성) 방법을 추가하여 더 적은 생성 과정으로 두 번째 최고 성능을 초과하는 결과를 보였습니다. 우리의 연구 결과는 RLVR이 복잡한 기업 과제에도 효과적으로 적용될 수 있음을 시사합니다.



### Diagnosing the Performance Trade-off in Moral Alignment: A Case Study on Gender Stereotypes (https://arxiv.org/abs/2509.21456)
- **What's New**: 이 논문은 사전학습 언어 모델(PLM)의 도덕적 정렬(moral alignment) 과정에서 발생하는 성과 절충(performance trade-off)의 기초 메커니즘을 조사합니다. 기존 연구에서는 공정성 목표(fairness objective)가 고안된 데이터셋에서 PLM의 행위를 조절하는 데 사용되지만, 이 과정에서 하향 작업의 성과가 저하되는 문제가 발생합니다. 또한, 성 고정관념(gender stereotypes)을 완화하기 위한 연구의 일환으로, 논문은 PLM의 유용성(helpfulness)과 해로움(harmlessness) 사이의 균형을 탐구합니다.

- **Technical Details**: 연구에서는 카운터팩추얼 데이터 증강(Counterfactual Data Augmentation, CDA)라는 방법을 사용하여 성 고정관념을 완화하게 됩니다. 연구의 주요 관측 결과는 (1) 공정성이 망각(forgetting)과 공정성 목표에 의해 영향을 받는다; (2) 하향 작업 성과는 전체 망각 수준에 따라 결정된다; (3) 고정관념의 선택적 망각(selective forgetting)은 전체 망각 수준을 줄이지 못한다; (4) 망각을 완화하기 위한 일반적인 접근법은 전체 망각 수준을 줄이지 못하며, 하향 작업 성과를 저하시킨다는 점입니다.

- **Performance Highlights**: 이 연구는 PLM의 하향 작업 성과와 성 고정관념 완화 간의 상충 관계를 다루고 있습니다. 연구 결과는 하향 작업의 성과가 전체적으로 받은 망각 수준에 의해 주로 좌우됨을 보여줍니다. 또한, 선택적 망각이 전체 망각 수준을 높이는데 기여하고, 일반적인 망각 완화 방법이 하향 작업 성과에 부정적인 영향을 미친다는 것을 강조합니다.



### LLM-Based Support for Diabetes Diagnosis: Opportunities, Scenarios, and Challenges with GPT-5 (https://arxiv.org/abs/2509.21450)
- **What's New**: 이번 연구에서는 당뇨병 진단과 모니터링을 위해 최신의 생성적 사전 학습 변환기인 GPT-5를 평가합니다. 이 연구는 미국 당뇨병 협회(ADA)의 2025년 진료 기준에 따라 설계된 합성 사례를 활용하여, GPT-5가 통증 인식 및 실험 결과 해석 등의 여러 시나리오를 처리할 수 있는지를 검토합니다. 연구 결과는 GPT-5가 의사들과 환자들에게 유용한 도구로 기능할 수 있는 가능성을 제시하며, 이러한 모델을 책임감 있게 평가할 수 있는 프레임워크의 중요성을 강조합니다.

- **Technical Details**: 이 연구는 GPT-5를 사용하여 다양한 임상 시나리오에서 당뇨병 진단을 돕는 시뮬레이션 기반 실험을 실시했습니다. 합성 환자 사례들이 사용되었으며, 이는 ADA 지침과 NHANES, EyePACS 등의 공개 데이터셋에서 영감을 받았습니다. 각 사례는 인구 통계데이터 및 증상 클러스터를 포함하여, GPT-5에 의해 분류, 진단적 추론, 환자 친화적인 설명 및 구조화된 JSON 요약을 생성하도록 요청받았습니다.

- **Performance Highlights**: GPT-5는 여러 시나리오에서 높은 일관성을 보여주며 ADA 기준에 잘 부합하는 결과를 도출했습니다. 각 합성 사례에 대해 GPT-5가 정의된 진단 기준에 맞춰 임상 데이터를 매핑하는 능력을 입증했으며, 의사 및 환자와 효과적으로 소통할 수 있는 능력도 강조되었습니다. 이러한 성과는 당뇨병 관리에서 LLM의 활용 가능성을 시사합니다.



### One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning (https://arxiv.org/abs/2509.21443)
Comments:
          22 pages, 11 figures, 6 tables

- **What's New**: 본 연구는 다문화 및 다언어 환경에서의 도덕적 추론과 관련된 종합적인 조사 결과를 발표합니다. 기존의 LLM(대형 언어 모델)은 주로 영어 기반의 데이터로 사전 훈련되어 문화적 맥락이나 언어적 차이를 고려하지 못하는 경향이 있습니다. 연구는 도덕적 판단의 일관성과 문화적 불일치를 다루고, LLM이 보편적인 도덕 원칙을 얼마나 잘 일반화하는지를 평가합니다.

- **Technical Details**: 연구에서는 MoralExceptQA 및 ETHICS와 같은 도덕적 추론 벤치마크를 다섯 개의 언어(중국어, 독일어, 힌디어, 스페인어, 우르두)로 번역하여 다국어로 제로샷 평가를 실시했습니다. 이를 통해 LLM의 응답 차이를 분석하였으며, 특히 도덕적 판단에서 문화적 요인이 어떻게 작용하는지를 탐구했습니다. 추가적으로, 사전 훈련 데이터가 LLM의 도덕적 방향성에 미치는 영향을 사례 연구를 통해 설명했습니다.

- **Performance Highlights**: 결과적으로 LLM은 언어에 따라 도덕적 추론이 현저하게 다르게 나타나며, 특히 영어에서 가장 높은 성능을 보였습니다. 저자들은 이러한 경향이 낮은 자원 언어에서 더욱 두드러진다고 언급하며, LLM이 정확한 판단을 하거나 지침을 따르는 데 어려움을 겪고 있음을 보여줍니다. 연구결과는 AI 윤리의 접근 방식이 더 문화적으로 포괄적이 되어야 한다는 필요성을 강조합니다.



### How Large Language Models Need Symbolism (https://arxiv.org/abs/2509.21404)
- **What's New**: 이 논문에서는 AI의 미래가 단순한 규모 확장을 넘어선다는 점을 주장합니다. 진정한 발견을 이끌어내기 위해, 대형 언어 모델은 그들의 강력하지만 맹목적인 직관을 안내할 인간이 만든 심볼(symbols)이 필요하다고 강조합니다.

- **Technical Details**: 대형 언어 모델은 방대한 양의 데이터에서 학습하지만, 이 모델들이 실질적인 혁신을 이루기 위해서는 의미 있는 방향성을 제공할 필요성이 있습니다. 이 연구는 이러한 방향성을 제공하기 위한 방법론과 접근 방식을 제안합니다.

- **Performance Highlights**: 이번 연구는 AI 모델의 성능 향상을 위한 새로운 방향성을 제시하며, 기계 학습(machine learning) 분야에서의 심볼 중심의 접근 방식의 중요성을 강조합니다. 이를 통해 AI의 발견 가능성을 극대화할 수 있는 방법论을 모색합니다.



### Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs (https://arxiv.org/abs/2509.21361)
Comments:
          20 pages, 4 charts

- **What's New**: 이 논문에서는 대형 언어 모델(LLM) 제공업체들이 자랑하는 최대 컨텍스트 윈도우 크기와 실질적인 사용 사례 간의 차이를 탐구합니다. 연구진은 최대 유효 컨텍스트 윈도우(Maximum Effective Context Window, MECW)의 개념을 정의하고, 다양한 크기와 문제 유형에서 컨텍스트 윈도우의 효과를 테스트하는 방법을 공식화했습니다. 이러한 연구는 컨텍스트 윈도우 크기 증가에 따른 모델의 효능 비교를 위한 표준화된 방법을 마련했습니다.

- **Technical Details**: 논문은 여러 모델에서 수십만 개의 데이터 포인트를 수집하여 보고된 최대 컨텍스트 윈도우(Maximum Context Window, MCW) 크기와 MECW 크기 간의 유의미한 차이를 발견했습니다. 연구 결과 MECW는 MCW와 현저히 다를 뿐만 아니라 문제 유형에 따라 변화한다는 것을 보여줍니다. 일부 최상급 모델은 100 토큰의 컨텍스트로도 실패했으며, 대부분은 1000 토큰에서 정확도가 심각하게 저하되었습니다.

- **Performance Highlights**: 논문의 데이터 분석 결과, 모든 모델은 최대 컨텍스트 윈도우에서 최대 99%에 이르는 큰 차이를 보였습니다. MECW는 제공되는 문제 유형에 따라 변화하며, 이는 모델의 정확도를 향상시키고 환각(hallucination) 비율을 감소시키기 위한 실질적이고 명확한 인사이트를 제공합니다. 본 연구는 효과적인 모델 개발에 필요한 중요한 기준을 제시합니다.



### Influence Guided Context Selection for Effective Retrieval-Augmented Generation (https://arxiv.org/abs/2509.21359)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG)의 컨텍스트 품질 평가 방식을 새롭게 정의하여 Contextual Influence Value (CI value)를 도입하였습니다. CI value는 생성기 성능 저하를 추적하여 각 컨텍스트의 품질을 정량화함으로써, 질 낮은 컨텍스트를 효율적으로 필터링하는 방법을 제공합니다. 이 새로운 접근법은 기존의 다양한 품질 평가 기준의 한계를 극복하기 위해, 쿼리, 컨텍스트 목록, 생성기를 통합적으로 활용합니다.

- **Technical Details**: CI value는 쿼리-의존성(쿼리와의 연관성), 목록-의존성(컨텍스트 간의 관계), 생성기-의존성(생성기와의 일치도)에 따라 각 컨텍스트의 기여도를 측정합니다. 또한 이 값을 통해 복잡한 하이퍼파라미터 조정 없이 양의 CI 값을 가진 컨텍스트만을 인식하여 유지할 수 있는 간단한 선택 전략을 제공합니다. CI 값 계산의 어려움을 해결하기 위해, CI Surrogate Model (CSM)을 개발하여 레이블 의존성과 계산 오버헤드를 줄이며 생성기 피드백을 활용하는 계층적 구조를 사용합니다.

- **Performance Highlights**: 8개의 NLP 작업과 여러 LLM에서 진행된 실험 결과, 제안된 방법이 최신 기술들과 비교하여 상당한 성능 향상을 달성했습니다. 특히, CI value를 기반으로 한 컨텍스트 선택 기법은 평범한 기준들을 초과하여 저질 컨텍스트를 효과적으로 필터링하면서도 중요한 정보를 보존합니다. 제안된 CSM 접근법은 평균 15.03%의 성능 향상을 보였으며, 이는 RAG 생성 성능의 눈에 띄는 개선을 나타냅니다.



### A Novel Differential Feature Learning for Effective Hallucination Detection and Classification (https://arxiv.org/abs/2509.21357)
Comments:
          10 pages, 7 figures, 13 tables

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)에서 발생하는 환각(hallucination) 신호의 정확한 위치를 탐색하기 위한 이중 모델 아키텍처를 제안합니다. 새로운 구조는 Projected Fusion (PF) 블록과 Differential Feature Learning (DFL) 메커니즘을 통합하여, 효율적인 검출 시스템의 개발을 위한 길잡이를 제공합니다. 연구의 결과, 환각 신호가 매우 희소한 특성 부분집합에 집중된다는 점을 발견하여 검출 효율성을 획기적으로 개선하는 방법론을 제시합니다.

- **Technical Details**: 제안된 이중 모델 아키텍처는 각기 다른 입력을 처리하는 두 개의 병렬 인코더 모델을 포함하며, 하나는 환각 신호에 주목하고 다른 하나는 사실 정보 인식을 목표로 합니다. Projected Fusion (PF) 블록은 서로 다른 은닉층에서의 정보를 효과적으로 통합하는 반면, Differential Feature Learning (DFL) 메커니즘은 두 모델의 특징 간의 절대 차이를 계산하여 판별적인 피쳐를 도출합니다. 이 연구는 특성 차이가 가장 큰 상위 1%의 피쳐만으로도 효과적인 검출이 가능함을 입증하며, 이를 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 체계적인 실험을 통해, HaluEval의 질문 응답(question answering), 대화(dialogue), 요약(summarization) 데이터 세트에 대해 환각 신호가 성능 향상에 중요한 역할을 함을 보여주었습니다. 연구 결과, 최적의 성능을 위해서는 전체 피쳐 차원의 1%만으로도 충분하다는 사실을 발견했습니다. 이러한 발견은 계산 효율적인 검출 시스템 구축의 가능성을 제시하며, 인퍼런스(inference) 비용을 줄이면서도 정확도를 유지할 수 있는 길을 열어줍니다.



### See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation (https://arxiv.org/abs/2509.22653)
Comments:
          CoRL 2025. Project page: this https URL

- **What's New**: 본 연구에서는 See, Point, Fly (SPF)라는 새로운 훈련 없는 항공 비전-언어 내비게이션(AVLN) 프레임워크를 소개합니다. SPF는 비전-언어 모델(VLM)을 기반으로 하여 어떤 형태의 자유 형식 지시에도 강력하게 반응할 수 있도록 설계되었습니다. 특히, 기존의 VLM 기반 접근 방식들이 행동 예측을 텍스트 생성 작업으로 다룬 것과 달리, 우리는 AVLN을 2D 공간 기초 작업으로 간주하는 중요한 통찰력을 제공합니다.

- **Technical Details**: SPF는 모호한 언어 지시를 2D 웨이포인트(waypoints)로 변환하여 항공 유무인기(UAV)의 행동 명령으로 사용합니다. 이 과정에서 카메라 정보를 활용해 2D 웨이포인트를 3D displacement 벡터로 전환하게 됩니다. 또한, SPF는 주행 거리를 적응적으로 조정하여 내비게이션 효율성을 높이며, 폐쇄 루프(control) 방식으로 동적인 환경에서도 유동적인 목표를 추적할 수 있습니다.

- **Performance Highlights**: SPF는 DRL 시뮬레이션 벤치마크에서 이전 최고의 방법에 비해 63%의 절대적 개선을 보이며 새로운 최첨단 성과를 세웠습니다. 실제 환경 평가에서도 SPF는 강력한 기준선 모델들보다 큰 폭으로 성능이 우수했습니다. 또한, 다양한 VLM에 대한 일반화 능력도 뛰어난 것을 입증하였습니다.



### CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/abs/2509.22647)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문은 이미지 캡셔닝 과제를 위한 새로운 Reinforcement Learning with Verifiable Rewards (RLVR) 패러다임을 제안합니다. 기존의 Supervised Fine-Tuning (SFT) 모델들이 특정 Ground-Truth 답변을 기억하는 것에 한계를 두고, 일반성 부족 문제를 해결할 수 있는 방법을 갖추고 있습니다. 새로운 Captioning Reinforcement Learning (CapRL) 프레임워크는 이미지에 대해 질문을 정확히 답할 수 있는 캡션 생성을 평가합니다.

- **Technical Details**: CapRL은 LVLM이 생성한 캡션과 비슷한 정보에 대한 Multiple-Choice Questions (MCQs)에 기반하여 객관적인 보상을 제공하는 이중 파이프라인을 사용합니다. 세부적으로, LVLM이 캡션을 생성하고, 비전이 없는 LLM이 그 캡션을 통해 질문에 답하는 정확성을 통해 보상을 결정합니다. 이 접근법은 주관적인 이미지 캡셔닝 문제에서 캡션의 품질을 유용성(utility)으로 정의합니다.

- **Performance Highlights**: CapRL을 적용한 결과, 12개의 벤치마크에서 상당한 성능 향상을 보여 주며, CapRL-3B가 72B 모델에 필적하는 성과를 달성했습니다. Prism Framework를 통한 평가에서 CapRL 모델이 기본선보다 평균 8.4% 향상된 성능을 나타냈습니다. 이러한 결과는 CapRL이 Dense하고 Accurate한 캡션을 생성하도록 모델을 효과적으로 유도함을 보여줍니다.



### Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs (https://arxiv.org/abs/2509.22646)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 AI 생성 비디오에서 확인할 수 있는 인간 지각의 '딥페이크 흔적'을 탐구하는 DeeptraceReward라는 새로운 벤치마크를 소개합니다. 이 데이터셋은 4,300개의 세부 주석을 포함하며, 각 주석은 자연어 설명, 경계 박스(bounding box) 위치 및 정확한 시간 정보를 제공합니다. 이를 통해 인간이 어떻게 AI 생성 비디오를 식별할 수 있는지와 그 이유를 조사하는 것을 목표로 하고 있습니다.

- **Technical Details**: DeeptraceReward 데이터셋은 3,300개의 고품질 생성 비디오와 3,300개의 실제 비디오로 구성됩니다. 저자들은 비디오에 존재하는 다양한 딥페이크 흔적을 아홉 가지 범주로 분류했으며, 이를 통해 멀티모달 언어 모델을 훈련시켜 인간의 판단을 모방할 수 있도록 하였습니다. 연구 결과, 7B의 보상 모델이 GPT-5를 34.7% 초과하여 인식 및 설명 과제에서 우수한 성능을 보였다는 것이 흥미롭습니다.

- **Performance Highlights**: 모델의 성능 평가에서 이진 분류(진짜 vs 가짜 비디오) 작업은 99.4%에 달하지만, 보다 세밀한 딥페이크 흔적 탐지 성능은 70.2%에 불과함을 보였습니다. 저자들은 자연어 설명 및 공간적 지각은 상대적으로 쉬운 반면, 시간 라벨링 작업은 가장 어려운 과제임을 확인했습니다. 이러한 결과는 영상 생성의 사회적 신뢰성을 향상시키기 위한 중요한 지침을 제공합니다.



### Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback (https://arxiv.org/abs/2509.22633)
- **What's New**: 본 논문에서는 온라인 강화 학습과 인간 피드백(RLHF)에 대한 탐색 원칙을 다룹니다. 인간의 선호 데이터를 사용하여 보상 모델을 적응적으로 수집하고 정책을 개선하는 새로운 탐색 스킴을 제안합니다. 기존 탐색 알고리즘의 단점을 분석하여, 비효율적인 비교 방식으로 인해 정보 불확실성을 줄이는 데 실패하는 지점을 지적합니다.

- **Technical Details**: RLHF의 모델 설정은 사용자가 제공하는 모든 가능한 입력이나 쿼리 집합인 프롬프트 공간 𝒳과 주어진 프롬프트에 대한 가능한 모든 출력 세트인 응답 공간 𝒜로 구성됩니다. 저자는 인간 선호 데이터를 최대 우도 추정(MLE)을 통해 학습하는 보상 모델을 정의하고, 이를 바탕으로 보상 극대화와 원래 모델의 유사성을 유지하는 정책을 미세 조정합니다. KL 정규화 보상 목표를 설정하여 보상 함수의 변화를 관리합니다.

- **Performance Highlights**: 제안된 탐색 스킴은 복잡한 동작 쌍 중에서 정책 개선에 가장 중요한 정보 불확실성을 줄이는 방향으로 선호 쿼리를 유도합니다. 이 시스템 아래에서 저자는 RLHF의 후회 경계를 $T^{(eta+1)/(eta+2)}$로 확립하며, 이는 모델 파라미터에 대해 다항적으로 스케일링됩니다. 이 알고리즘은 모든 모델 파라미터에 대해 처음으로 다항적 후회 스케일링을 제공합니다.



### LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision (https://arxiv.org/abs/2509.22631)
- **What's New**: Labeling Copilot는 컴퓨터 비전을 위한 첫 번째 데이터 큐레이션(curation) 딥 리서치 에이전트입니다. 이 에이전트는 대규모 다중 모달 언어 모델에 의해 구동되며, 데이터 품질, 다양성 및 비용 간의 복잡한 균형을 수행합니다. Labeling Copilot은 데이터 발견(discovery), 새로운 데이터 생성(synthesis) 및 합의 기반 주석(annotation)이라는 세 가지 핵심 기능을 통해 데이터 큐레이션 프로세스를 최적화합니다.

- **Technical Details**: Labeling Copilot의 핵심 기능은 (1) Calibrated Discovery를 통해 대규모 저장소에서 관련 데이터를 검색하고, (2) Controllable Synthesis로 드문 시나리오를 위한 새로운 데이터를 생성하며, (3) Consensus Annotation으로 여러 기초 모델을 조정하여 정확한 레이블을 생성하는 것입니다. 이 에이전트는 또한 고급 필터링 및 여러 도구의 조정을 통해 데이터 품질을 향상시키는 메커니즘을 통합하고 있습니다.

- **Performance Highlights**: Labeling Copilot의 구성 요소는 대규모 검증을 통해 그 효과가 입증되었습니다. 예를 들어, Consensus Annotation 모듈은 COCO 데이터셋에서 평균 14.2개의 후보 제안을 제공하며, Open Images 데이터셋에서는 총 1500개 이상의 새로운 바운딩 박스 카테고리를 발견했습니다. Calibrated Discovery 도구는 10백만 샘플 규모에서 최대 40배 더 효율적인 계산 성능을 보여줍니다.



### IA2: Alignment with ICL Activations Improves Supervised Fine-Tuning (https://arxiv.org/abs/2509.22621)
- **What's New**: 본 논문에서는 ICL(인맥 학습)와 SFT(지도 세부 조정)의 내부 계산을 통해 SFT의 품질을 향상시킬 수 있는지를 탐구하며, 이를 ICL Activation Alignment(IA2)라는 자기 증류(self-distillation) 기법을 통해 제안합니다. IA2는 ICL의 활성화 패턴을 SFT 모델에 복제하려고 하며, ICL처럼 내부적으로 사고할 수 있도록 유도합니다.

- **Technical Details**: IA2는 (1) 정보가 풍부한 ICL 활성화를 수집하고, (2) ICL과 기능적 정렬을 강화한 후, (3) 이 준비된 모델에서 SFT를 수행하는 단계로 구성됩니다. 이는 SFT 모델의 성능을 극적으로 향상시키며, 우리가 사용한 12개의 벤치마크에서 13,000개 이상의 모델을 훈련하여 그 결과를 검증합니다.

- **Performance Highlights**: IA2는 SFT 전의 priming 단계로서, 모델의 출력 정확도와 보정을 개선해 주며, ICL만의 중요한 훈련 신호를 제공합니다. IA2의 적용은 모델 적응의 내부 메커니즘을 이해하는 데도 중요한 통찰력을 제공하며, SFT만으로는 얻을 수 없는 중요한 데이터로 모델 성능 향상에 기여합니다.



### Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting (https://arxiv.org/abs/2509.22615)
- **What's New**: 이 논문에서는 2D Gaussian Splatting (2DGS)을 새로운 비주얼 표현으로 탐구하여 멀티모달 시스템에서 비전-언어 정렬을 위한 효율적이고 효과적인 중간 표현으로 사용할 수 있는지에 대한 논의를 담고 있습니다. 기존 RGB 기반 비전 인코더의 데이터 전송 및 처리 효율성 문제를 해결하기 위해, 2DGS는 색이 지정된 비대칭 Gaussian의 집합으로 이미지를 매개변수화하여 더 간결하고 공간적으로 적응 가능한 형태로 정보를 전달합니다. 이 연구는 대규모 범위에서 2DGS를 구현하고 CLIP 프레임워크에 적응시키는 방법을 제안하며, 빠른 피팅과 GPU 유틸리제를 달성했습니다.

- **Technical Details**: 논문에서 제안한 시스템 및 알고리즘 최적화는 구조화된 초기화, 밝기 인식 L1 프루닝 및 배치 CUDA 커널을 포함하여 기존의 구현보다 90배 이상 빠른 피팅을 달성하도록 설계되었습니다. 2DGS 표현은 효율적인 데이터 전송을 가능하게 하고, RGB 기반의 변환기 아키텍처의 냉동된 형태를 재사용하여 가벼운 입력 전처리 유닛과 함께 CLIP 프레임워크에 효과적으로 통합됩니다. 이 과정에서 약 7%의 매개변수로 훈련을 수행하면서도 우수한 전이 학습 성과를 나타냅니다.

- **Performance Highlights**: 실험 결과 12.8M 규모의 DataComp 데이터셋에서 GS 인코더는 이미지넷-1K의 제로샷 성능을 의미 있게 달성하면서 픽셀 대비 입력 압축은 3배에서 20배 이르는 결과를 보여주었습니다. 현재 2DGS 기반 인코더는 RGB 기반 모델에 비해 정확성이 낮지만, 명백한 전이 가능성과 압축된 표현으로서의 유용성을 입증하였습니다. 이 연구는 2DGS가 멀티모달 시스템에서 효과적인 대안이 될 수 있으며, 이는 처리 효율성과 지속 가능성을 높이는 방향으로 나아가고자 하는 방법을 제시합니다.



### Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning (https://arxiv.org/abs/2509.22601)
Comments:
          26 pages, 11 figures

- **What's New**: 본 논문에서는 자기 모방 학습(Self-Imitation Learning, SIL)을 기반으로 한 새로운 강화 학습(RL) 접근 방식인 SPEAR(Self-imitation with Progressive Exploration for Agentic Reinforcement learning)를 제안합니다. 이 방법은 정책 엔트ropy를 동적으로 조절하여 탐색(exploration)과 활용(exploitation)의 균형을 유지합니다. 특히, SPEAR는 툴 사용 능력을 개발하기 위한 도움 보상을 포함하여, 훈련 초기 단계에서의 엔트로피 증가가 탐색을 촉진함을 강조합니다.

- **Technical Details**: 논문에서 제안하는 SPEAR는 스킬 기반 탐색을 지원하기 위한 인트린식 보상을 이용하며, 후속 단계에서는 기존의 성공적인 패턴을 활용하여 행동 수준의 탐색을 촉진합니다. 또한, Replay Buffer의 경험에 대한 이점 재조정과 높은 공분산의 클리핑 기법을 통해 정책 업데이트의 안정성을 강화하고 보상 해킹(reward hacking) 문제를 완화합니다. 이는 정책 엔트로피를 조절하여 훈련의 불안정성을 방지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: SPEAR는 GRPO, GiGPO, Dr.BoT와 같은 여러 기준 방법들에 비해 상당한 성능 향상을 보여주며, ALFWorld 및 WebShop 태스크에서 각각 최대 16.1%, 20.7%의 성능 개선을 기록했습니다. 또한, Dr.BoT의 성능을 각각 AIME24에서 3.8%, AIME25에서 6.1% 향상시킵니다. SPEAR는 낮은 계산 복잡도를 유지하며 다른 LLM 기반 에이전트들에 대해 뛰어난 호환성과 확장성을 보여주는 플러그 앤 플레이 알고리즘입니다.



### EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning (https://arxiv.org/abs/2509.22576)
- **What's New**: 새로운 논문에서는 sparse rewards 환경에서 multi-turn LLM 에이전트를 훈련하는 데 있어 exploration-exploitation cascade failure라는 고유한 문제를 식별했습니다. 이를 해결하기 위해 Entropy-regularized Policy Optimization (EPO)이라는 새로운 프레임워크를 제안하며, 세 가지 메커니즘을 통해 실패 주기를 끊어냅니다. EPO는 기존 방법들과는 달리 정책의 엔트로피를 시간 기반으로 제어하여 안정적인 학습을 보장합니다.

- **Technical Details**: EPO는 세 가지 주요 메커니즘으로 구성되어 있습니다: 첫째, multi-turn 환경에 대한 엔트로피 정규화를 도입하여 탐색을 강화하고, 둘째, 정책 엔트로피를 역사적 평균으로 제한하는 엔트로피 스무딩 정규화를 통해 급격한 변동을 방지합니다. 셋째, 적응형 단계 기반 가중치를 적용하여 학습 과정에서 탐색과 활용의 균형을 유지합니다. 분석 결과, EPO는 엔트로피 분산을 단조롭게 감소시키면서 수렴을 유지하는 것을 보장합니다.

- **Performance Highlights**: EPO는 ScienceWorld에서 최대 152%의 성능 향상을 기록하였고, ALFWorld에서는 최대 19.8% 향상을 달성했습니다. 이러한 결과는 EPO가 이전에 훈련이 불가능했던 sparse-reward 시나리오를 매끄럽게 수렴하는 최적화 문제로 변화시킬 수 있음을 보여줍니다. 또한, EPO는 새로운 훈련 구조를 제시하여 LLM 에이전트 훈련에 있어 기본적으로 다른 엔트로피 제어가 필요함을 강조합니다.



### Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Tim (https://arxiv.org/abs/2509.22572)
- **What's New**: 본 논문에서는 Test-Time Scaling (TTS) 기법을 통해 대규모 언어 모델(LLM)의 추론 능력을 향상시킬 수 있다는 새로운 전략인 Dynamic Experts Search (DES)를 제안합니다. 기존의 방법들은 주로 출력 수준의 샘플링에 의존하였으나, 우리는 모델 아키텍처의 역할을 간과하고 있는 점을 지적했습니다. Mixture-of-Experts (MoE) 아키텍처의 전문가 활성화 수를 조정하여 다양한 해결책을 생성하는 가능성을 발견하였습니다.

- **Technical Details**: DES는 두 가지 주요 요소를 통합하여 작동합니다. 첫 번째는 Dynamic MoE로, 이는 추론 과정에서 활성화된 전문가 수를 직접 제어할 수 있게 하여 추가 비용 없이 다양한 추론 경로를 생성할 수 있게 합니다. 두 번째는 Expert Configuration Inheritance로, 이는 추론 경로 내에서 전문가 수를 일관되게 유지하여 안정성과 다양성을 균형 있게 분산할 수 있도록 합니다.

- **Performance Highlights**: 다양한 MoE 아키텍처와 검증자, 추론 벤치마크(예: 수학, 코드, 지식)에 대한 광범위한 실험 결과, DES는 기존 TTS 기준을 일관되게 초월하여 정확도와 안정성을 높였습니다. DES는 추가 비용 없이 추론 능력을 향상시키며, 다양한 모델 규모와 검증자 선택에 대해 효과적으로 일반화되는 장점을 보여주었습니다.



### Does AI Coaching Prepare us for Workplace Negotiations? (https://arxiv.org/abs/2509.22545)
- **What's New**: 이번 연구는 Trucey라는 AI 기반 코치의 개발과 평가를 다룬다. Trucey는 Brett의 협상 모델에 기반하여 설계되었으며, 협상에서의 심리적 장벽을 타파하고 개인의 준비성을 향상시키는 데 초점을 맞추고 있다. 실험 연구를 통해 Trucey, ChatGPT, 그리고 전통적인 협상 핸드북의 효과를 비교하여 AI가 협상 준비에 미치는 영향을 탐구하였다.

- **Technical Details**: Trucey는 인간 중심의 AI와 Industrial-Organizational (I/O) 심리학의 원리에 따라 설계되었으며, fine-tuned GPT-4.1를 사용하였다. 참가자들은 세 가지 조건에서 실험을 수행하고, 심리적 자원과 자아 효능감, 협상 준비 상태를 평가하는 설문지를 완료하였다. 특히 Trucey는 사용자 맞춤형 프롬프트와 역할 기반 시뮬레이션을 통해 협상 준비 과정을 지원한다.

- **Performance Highlights**: 실험 결과, Trucey가 두 가지 비교 조건에 비해 두려움을 유의미하게 감소시켰으나, 전통적인 핸드북은 사용 용이성과 심리적 자원 면에서 더 높은 점수를 받았다. 인터뷰 결과, 참가자들은 핸드북의 종합적이고 리뷰 가능한 내용이 자신감을 높이고 준비에 필수적이라고 언급하였다. 반면, AI 코치는 안내가 세분화되어 불안감을 유발하고, 경험에 따른 명확한 피드백이 부족하다는 평가를 받았다.



### Mental Health Impacts of AI Companions: Triangulating Social Media Quasi-Experiments, User Perspectives, and Relational Theory (https://arxiv.org/abs/2509.22505)
- **What's New**: 최근 AI 기반 친구 채팅봇(AICC)인 Replika의 사용이 급증하고 있으며, 이러한 채팅봇은 공감적인 상호작용을 제공하나, 그 심리사회적 영향은 불분명한 상태입니다. 본 연구는 구글 소셜 미디어 데이터를 활용한 대규모 준 실험적 연구 및 심층 인터뷰를 통해 AICC와의 상호작용이 사용자 웰빙에 미치는 영향을 조사하였습니다. 결과는 정서적 표현의 증가와 외로움, 자살 사상에 대한 언급 증가가 혼재된 모습을 보였습니다.

- **Technical Details**: 연구는 15명의 AICC 사용자와의 심층 반구조화 인터뷰를 통해 사용자 경험의 복잡성을 분석하였으며, Knapp의 관계 발달 모델을 활용하여 AICC와의 관계 맥락을 해석했습니다. 준 실험적 분석에서는 Reddit 커뮤니티에서 AICC 사용 전후의 대화 패턴을 관찰하였고, 위계적 성향 점수 매칭과 Difference-in-Differences 회귀 기법을 적용하여 치료 효과를 추정했습니다.

- **Performance Highlights**: 결과적으로, AICC와의 상호작용은 사용자들이 정서적 검증을 받고 사회적 연습을 하는 긍정적인 요소를 제공하면서, 과도한 의존과 Withdrawal의 위험도 존재한다는 점을 밝혔습니다. 이러한 연구 결과는 AI 친구 채팅봇 설계에서 건강한 경계를 지원하고, 의존 없이 공개성을 촉진하며, 관계 단계를 드러내는 방향으로의 시사점을 제공합니다.



### IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method (https://arxiv.org/abs/2509.22463)
- **What's New**: 본 논문은 Iterative Implicit Euler Transformer (IIET)을 제안하여 고차 수치적 기법을 단순화함으로써 Transformer 모델의 성능을 향상시키고 이를 통해 모델 압축이 가능함을 보여준다. 기존의 효율성 기법인 Knowledge Distillation은 PCformer 모델의 성능을 감소시킬 수 있는 한계를 지니고 있다. IIET는 각 반복이 초기 예측에 대한 수정 과정을 포함해 모델의 정확성을 보장하며, 성능과 효율성을 동시에 추구할 수 있는 새로운 접근 방식을 제공한다.

- **Technical Details**: IIET 모델은 내재하는 안정성을 바탕으로 한 implicit Euler 방식을 활용하여 아키텍처를 설계하였다. 이 모델의 각 이터레이션은 기존 예측을 반복적으로 수정하여 최종 결과의 정확성을 높인다. 또한, IIET는 최신 다단계 방식과 결합하여 높은 성능을 유지하며, 각 이터레이션의 중요성을 평가하는 Iteration Influence-Aware Distillation (IIAD)을 통해 추론 효율을 삼나 이터레이션 수를 줄인다. 이를 통해 IIET의 효율적인 변형인 E-IIET는 추론에서 55%의 계산비용 절감을 달성하였다.

- **Performance Highlights**: IIET는 lm-evaluation-harness에서 평균 정확도를 2.65% 증가시켰으며, PCformer에 비해서도 0.8% 더 향상된 성능을 보인다. E-IIET는 높은 정확도를 유지하면서도 99.4%의 원래 작업 정확도를 유지하며 인퍼런스 오버헤드를 크게 줄인다. IIET와 가장 효율적인 변형은 일반 Transformer 대비 평균 1.6% 이상의 성능 향상을 기록하며, 속도 또한 비슷하다는 점에서 두드러진 개선을 이룩하였다.



### MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark (https://arxiv.org/abs/2509.22461)
Comments:
          25 pages, 7 figures

- **What's New**: MDAR는 복잡하고 동적으로 발전하는 오디오 추론 작업을 평가하기 위해 설계된 새로운 벤치마크입니다. 기존 벤치마크는 주로 정적 상황이나 단일 장면에 초점을 맞추었으며, 다수의 화자와 환경 소음이 공존하는 현실 세계의 다양한 상황을 충분히 포착하지 못했습니다. MDAR는 3,000개의 정교하게 구성된 질문-응답 쌍을 포함하고 있으며, 다섯 가지 복잡한 추론 범주에 대한 평가를 가능하게 합니다.

- **Technical Details**: MDAR는 오디오 언어 모델의 복잡한 추론 능력을 평가하기 위한 세 가지 작업 유형을 사용합니다: 단일 선택 질문, 다중 선택 질문, 오픈 엔디드 질문. 각 범주는 장면 이해, 사회적 관계, 사건 추론, 시간적 추론, 이상 탐지 및 안전을 포함하여 고품질의 오디오 클립과 인간 주석 질문이 결합된 새로운 데이터 세트를 기반으로 합니다. MDAR는 또한 이전의 벤치마크와는 달리 동적이고 혼합된 오디오 장면을 제공하여 실제적인 벤치마크를 설정했습니다.

- **Performance Highlights**: 26개의 최첨단 오디오 언어 모델을 MDAR에서 평가한 결과, 복잡한 추론 작업에서 제한이 있음을 발견했습니다. Qwen2.5-Omni는 단일 선택 질문에서 76.67%의 정확도를 달성한 반면, GPT-4o Audio는 68.47%에 그쳤습니다. 모든 질문 유형에서 모델이 80% 이상의 성능을 발휘하지 못하는 것으로 나타났으며, 이는 미래 오디오 추론 에이전트의 개선이 필요하다는 것을 강조합니다.



### Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers (https://arxiv.org/abs/2509.22445)
- **What's New**: 이번 논문은 기계 학습에서 Occam's razor를 적용하기 위한 공식적인 틀인 Minimum Description Length (MDL) 원칙을 신경망, 특히 Transformers에 적용하는 데 따른 어려움을 다루고 있습니다. 저자들은 Kolmogorov 복잡도의 이론에 근거하여 점근적으로 최적화된 설명 길이 목표의 이론적 개념을 소개합니다. 또, 이러한 목표를 최소화하는 것이 모델 자원 제한이 증가함에 따라 최적의 압축을 달성한다는 것을 보여줍니다.

- **Technical Details**: 연구는 Transformers에 대해 점근적으로 최적화된 목표가 존재함을 입증하며, 이는 이들의 계산적 보편성을 새로운 방식으로 시연하여 이루어졌습니다. 저자들은 적응형 가우시안 혼합 선행 지식을 바탕으로 만드는 변량 목표를 통해 이러한 목표가 취급 가능하고 미분 가능하다는 것을 보여줍니다. 이를 통해 최적화 과정에서 발생할 수 있는 문제를 다루고, 수학적으로 기반한 목표 설정이 신경망 훈련에 유용함을 논의합니다.

- **Performance Highlights**: 경험적 분석 결과, 변량 목표는 알고리즘 작업에서 강력한 일반화를 보여주는 저복잡도 해결책을 선택하는 것으로 나타났습니다. 그러나 표준 최적화 방법은 무작위 초기화에서 이러한 해결책을 찾는 데 실패하였으며, 이는 최적화 과정에서의 중요 과제를 강조합니다. 이 논문은 강한 점근적 보장을 가진 설명 길이 목표를 판별하기 위한 이론적 틀을 제공함으로써 더 나은 압축 및 일반화를 이룰 수 있는 신경망 훈련의 잠재적 경로를 제시하고 있습니다.



### Can Synthetic Query Rewrites Capture User Intent Better than Humans in Retrieval-Augmented Generation? (https://arxiv.org/abs/2509.22325)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 콜로quial 생략과 모호한 참조가 포함된 쿼리에 대해 고품질 합성 쿼리를 생성하는 SynRewrite라는 새로운 쿼리 재작성 모델을 제안합니다. 기존의 쿼리 재작성 방식이 인간 주석자에게 의존했던 반면, 이제는 GPT-4o를 사용하여 대화 이력, 현재 쿼리, 긍정적인 문서 및 답변을 바탕으로 합성 쿼리를 생성합니다. 이 방법을 통해 사용자 의도를 더욱 정확하게 포착하고 시스템의 응답 품질을 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: SynRewrite 모델은 합성 데이터 기반으로 쿼리 재작성을 수행하며, Flan-T5 모델을 파인튜닝하여 대화 이력과 현재 쿼리를 입력받고 합성된 쿼리를 출력합니다. 이 과정에서 Direct Preference Optimization (DPO) 알고리즘을 사용하여 생성기 피드백을 통한 성능 향상을 달성합니다. 특히, GPT-4o를 활용해 고품질의 재작성 쿼리를 합성하는 과정은 실질적인 데이터 수요를 줄이는 데 기여합니다.

- **Performance Highlights**: TopiOCQA와 QRECC 데이터셋에서의 실험 결과, SynRewrite는 인간이 재작성한 쿼리보다 우수한 성능을 보여줍니다. 예를 들어, Retrieval 성능에서는 61.31의 MRR을 기록하며 이는 인간 재작성에 비해 두 배의 성능 향상을 의미합니다. 이러한 결과는 합성 데이터가 사용자 의도를 바르게 포착하는 데 효과적이라는 것을 입증합니다.



### PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning (https://arxiv.org/abs/2509.22315)
Comments:
          8 pages

- **What's New**: 이 논문은 인간 인지의 이중 처리 이론에 영감을 받아, 빠르고 직관적인 사고(System 1)와 느리고 신중한 사고(System 2)를 통합하는 멀티 에이전트 추론 프레임워크인 PRIME(Planning and Retrieval-Integrated Memory for Enhanced Reasoning)를 소개하고 있습니다. PRIME은 초기 빠른 답변을 생성하는 Quick Thinking Agent를 활용하고, 불확실성이 감지되면 보다 구조화된 System 2 사고 프로세스를 활성화하여 전문화된 에이전트들이 계획, 가설 생성, 정보 통합 및 의사 결정을 수행합니다. 이러한 설계는 인간의 인지 과정을 충실히 모방하며, 효율성과 정확성을 향상시키는 데 기여합니다.

- **Technical Details**: PRIME은 인간의 인지 프로세스를 모델링하고, 빠른 직관적인 사고와 느린 신중한 사고를 통합하기 위해 설계된 멀티 에이전트 시스템입니다. 이 시스템은 Quick Thinking Agent(System 1)를 통해 직관적 응답을 생성하며, 그 후 Reflection Agent가 이 응답을 비판적으로 평가합니다. 문제가 발견되면, System 2 프로세스가 활성화되어 체계적인 계획, 탐색, 가설 설정 및 정보 통합을 통해 신뢰할 수 있는 최종 답변을 도출합니다.

- **Performance Highlights**: 실험 결과, PRIME은 LLaMA 3 모델이 GPT-4 및 GPT-4o와 같은 최신 클로즈드 소스 모델과 경쟁할 수 있는 성능을 발휘하게 함을 보여줍니다. 특히 PRIME은 복잡한 지식 기반의 추론이 필요한 작업에서 뛰어난 성능을 발휘했으며, 효율적인 컴퓨팅 자원 할당을 통해 보다 어려운 작업에 대해서만 System 2를 활성화하여 불필요한 숙고를 줄였습니다. 이로 인해 PRIME은 복잡한 지식 집약적 추론 작업을 위한 강력하고 확장 가능한 프레임워크로 자리 잡았습니다.



### InfiMed-Foundation: Pioneering Advanced Multimodal Medical Models with Compute-Efficient Pre-Training and Multi-Stage Fine-Tuning (https://arxiv.org/abs/2509.22261)
- **What's New**: 이 논문에서는 InfiMed-Foundation-1.7B와 InfiMed-Foundation-4B라는 두 가지 의학 특화 다중 모달 대형 언어 모델(MLLMs)을 제안합니다. 이 모델들은 의료 응용 프로그램에서 최첨단 성능을 제공하도록 설계되었습니다. 고품질의 일반 및 의학 다중 모달 데이터를 결합하고, 새로운 다차원 품질 평가 프레임워크를 제안하여 높은 품질의 데이터셋을 구성하였습니다.

- **Technical Details**: InfiMed-Foundation 모델은 이미지 해상도를 낮추고, 다중 모달 시퀀스 패킹 기술을 적용하여 훈련 효율성을 향상시켰습니다. 또한, 세 가지 단계의 감독 학습 미세 조정 과정을 통해 복잡한 의료 작업에 필요한 지식을 효과적으로 추출할 수 있게 하였습니다. MedEvalKit 프레임워크를 사용하여 모델 성능을 평가하였으며, 이는 의료 시각 질문응답 및 진단 작업에서 우수한 성과를 나타냈습니다.

- **Performance Highlights**: InfiMed-Foundation-1.7B 모델은 Qwen2.5VL-3B를 초월했으며, InfiMed-Foundation-4B는 HuatuoGPT-V-7B와 MedGemma-27B-IT를 초월하여 의료 관련 성능을 입증하였습니다. 이러한 결과는 모델들이 실제 치료 상황에서 의료 전문가를 지원할 수 있는 가능성을 믿을 수 있게 보여줍니다. 데이터 품질, 훈련 효율성, 및 도메인 특화 지식 추출과 같은 핵심 도전을 해결하면서, 본 연구는 의료 분야에서 더 신뢰할 수 있고 효과적인 AI 솔루션을 위한 길을 열었습니다.



### Library Hallucinations in LLMs: Risk Analysis Grounded in Developer Queries (https://arxiv.org/abs/2509.22202)
Comments:
          23 pages, 5 tables

- **What's New**: 이번 연구는 사용자 수준의 프롬프트 변동이 LLM(대형 언어 모델) 생성 코드에서의 라이브러리 환각(library hallucinations)에 미치는 영향을 체계적으로 분석한 첫 번째 연구입니다. 우리는 라이브러리 이름 환각(유효하지 않은 import)과 라이브러리 멤버 환각(유효한 라이브러리의 잘못된 호출) 두 가지 유형의 환각을 평가했습니다. 이러한 환각은 개발자들에게 미치는 영향이 커서, 특히 잘못된 라이브러리 이름이 사용될 경우 그 위험성이 더욱 증가합니다.

- **Technical Details**: 연구에서는 개발자 포럼에서 추출한 현실적인 사용자 언어 및 다양한 정도의 사용자 오류(단일 또는 다중 문자 철자 오류, 완전히 가짜 이름/멤버 등)가 LLM 환각 비율에 미치는 영향을 조사하였습니다. 우리의 발견에 따르면, 라이브러리 이름의 한 문자 철자 오류는 최대 26%의 작업에서 환각을 유발하며, 가짜 라이브러리 이름은 최대 99%의 작업에서 허용됩니다. 또한 시간 관련 프롬프트는 최대 84%의 작업에서 환각을 초래할 수 있음을 보여줍니다.

- **Performance Highlights**: 프롬프트 엔지니어링(prompt engineering)은 환각 완화에 가능성을 보여주지만, 일관성이 없고 LLM에 따라 다릅니다. 연구 결과는 LLM이 자연스러운 프롬프트 변동에 얼마나 취약한지 강조하며, 라이브러리 관련 환각 및 그 잠재적 악용에 대한 예방책이 시급히 필요함을 나타냅니다. LLMs의 이러한 견고성과 환각 문제를 일찍 해결하는 것이 개발자와 시스템의 안전성을 높이는 데 중요하다고 할 수 있습니다.



### MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing (https://arxiv.org/abs/2509.22186)
Comments:
          Technical Report; GitHub Repo: this https URL Hugging Face Model: this https URL Hugging Face Demo: this https URL

- **What's New**: MinerU2.5는 1.2B 매개변수를 가진 문서 파싱 비전-언어 모델로 가장 발전된 인식 정확도를 달성하고 있을 뿐만 아니라 뛰어난 계산 효율성을 유지합니다. 이 모델은 글로벌 레이아웃 분석을 로컬 콘텐츠 인식과 분리하는 조잡한-세밀한 두 단계 파싱 전략을 채택하였습니다. 이러한 접근은 고해상도 입력을 처리하는 경우의 계산 오버헤드를 피할 수 있게 합니다.

- **Technical Details**: MinerU2.5는 두 단계의 추론 메커니즘을 통해 문서 이미지를 처리합니다. 첫 번째 단계에서는 다운샘플링된 문서 이미지에 대한 빠르고 포괄적인 레이아웃 분석을 수행하여 구조적 요소를 식별합니다. 두 번째 단계에서는 원본 이미지에서 추출한 지역의 고해상도에 맞춰 세밀한 내용을 인식하여 복잡한 수식 및 테이블과 같은 세밀한 디테일을 보존합니다.

- **Performance Highlights**: MinerU2.5는 여러 벤치마크에서 상태-of-the-art (SOTA) 성능을 달성하며, 다양한 인식 작업에서 일반 목적 및 도메인 특화 모델을 초월합니다. 특히, 복잡한 문서에서 높은 파싱 정확도를 유지하면서 오버헤드를 크게 줄이는 혁신적인 아키텍처와 종합적인 데이터 엔진을 개발하여 문서 파싱 분야의 강력한 성능을 입증하였습니다.



### SecureAgentBench: Benchmarking Secure Code Generation under Realistic Vulnerability Scenarios (https://arxiv.org/abs/2509.22097)
- **What's New**: 이번 논문에서는 코드 에이전트의 안전한 코드 생성을 평가하기 위한 새로운 벤치마크인 SecureAgentBench를 제안합니다. 이 벤치마크는 105개의 코딩 작업을 포함하고 있으며, 실세계의 오픈소스 취약점을 기반으로 한 맥락을 고려합니다. 기존 벤치마크의 한계를 넘어, 기능적 정확성 및 새로운 취약성의 검출을 아우르는 포괄적인 평가 방식을 채택하고 있습니다.

- **Technical Details**: SecureAgentBench는 현실적인 작업 형태를 채택하여 여러 파일을 수정하는 것을 요구합니다. 각 작업은 평균 200단어의 요구사항을 해석하고, 최대 36.4K 파일과 4.2M 코드 라인을 가진 코드 베이스를 분석해야 합니다. 평가에는 기능 테스트, PoC(exploits) 프로그램을 통한 검증, 정적 애플리케이션 보안 테스트(SAST) 도구를 사용하여 새로운 취약성을 탐지하는 방식이 포함됩니다.

- **Performance Highlights**: 평가 결과 현재의 코드 에이전트들은 올바르고 안전한 코드를 생성하는 데 어려움을 겪고 있으며, SWE-agent가 지원되는 DeepSeek-V3.1의 경우에도 오직 15.2%만이 올바르고 안전한 솔루션을 제공하였습니다. 또한, 기존의 취약성을 재도입하는 것 외에도 새로운 보안 리스크를 발생시키는 경향이 20% 이상으로 나타났습니다. 이러한 결과는 코드 에이전트의 보안 인식 강화를 위한 추가 연구의 필요성을 강조합니다.



### Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias (https://arxiv.org/abs/2509.22061)
Comments:
          6 pages, 1 figure, Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문은 Speech Continuation(SC)에서의 편향(bias)을 체계적으로 평가한 첫 번째 연구로, 성별(gender)과 음성 품질(voice quality)이 이어지는 발화에 미치는 영향을 조사합니다. SC는 음성 프롬프트에 대한 일관된 연장을 생성하는 작업으로, 사회적으로 관련된 표현 편향을 탐색하기 위한 통제된 방법으로 제시됩니다. SC 작업의 발전으로, 음성 기반 편향을 검토하는 것이 중요한 실용적 응용 프로그램이 될 것이라는 점을 강조합니다.

- **Technical Details**: SC 작업은 짧은 음성 프롬프트를 제공받고, 화자(identity)를 유지하며 이어지는 내용을 생성하는 시스템의 능력을 평가합니다. 연구는 SpiritLM, VAE-GSLM, SpeechGPT 모델을 활용하여 아젠다 보존, 음성 품질 유지, 텍스트 기반 편향 메트릭 스코어를 평가합니다. 이를 위해 Spoken StereoSet에서 두 가지 데이터 세트를 구축하고, 최신 목소리 변환 시스템인 VoiceQualityVC를 사용하여 다양한 음성 품질 변수를 조작하였습니다.

- **Performance Highlights**: 결과에 따르면, 화자 유사성과 일관성은 여전히 도전 과제로 남아 있으며, 텍스트 기반 평가에서는 모델과 성별 간의 상호 작용이 뚜렷하게 드러났습니다. 일관성이 충분히 높을 경우, 텍스트 메트릭(agency, sentence polarity)에서 성별 효과가 나타났고, 여자 프롬프트의 경우 이어지는 발화가 모달 음성으로 더 강하게 돌아가는 경향이 발견되었습니다. 이러한 결과는 SC가 음성 기초 모델에서의 사회적으로 중요한 편향을 드러내는 신뢰할 수 있는 도구임을 보여줍니다.



### A2R: An Asymmetric Two-Stage Reasoning Framework for Parallel Reasoning (https://arxiv.org/abs/2509.22044)
Comments:
          15 pages, 3 figures

- **What's New**: 이번 연구에서는 비대칭 이단계 추론 프레임워크인 A2R(Asymmetric Two-Stage Reasoning)을 제안합니다. 이 프레임워크는 모델의 잠재력과 실제 성능 간의 격차를 해소하기 위해 설계되었습니다. A2R은 첫 번째 단계에서 솔루션 후보를 생성하는 '탐색기'(explorer) 모델과, 두 번째 단계에서 이들을 통합하는 '합성기'(synthesizer) 모델로 구성되어 있습니다. 이를 통해 기존의 순차적 접근법과는 다른 방식으로 계산량을 확장할 수 있습니다.

- **Technical Details**: A2R은 두 가지 상호보완적인 단계로 추론을 분리합니다: 첫 번째 단계는 다양한 추론 경로를 생성하는 '탐색' 단계이고, 두 번째 단계는 통합하는 '합성' 단계입니다. A2R은 이전의 단순 투표나 선택을 통한 병렬 추론 접근 방식과 달리 전체 추론 경로 집합에 대해 생성적인 재추론을 수행합니다. 이를 통해 후보 솔루션의 전체적인 시각을 형성하고, 일관된 증거를 판단하며, 보다 정확하고 견고한 최종 답변을 통합할 수 있습니다.

- **Performance Highlights**: A2R을 AIME 2024와 같은 복잡한 추론 벤치마크에 적용한 결과, Qwen3-8B-distill 모델을 사용하여 4개의 추론 경로에서 75% 성능 향상을 달성했습니다. 또한, A2R-Efficient 구조를 통해 더 작은 탐색기와 더 강력한 합성기를 조합하여 평균 30% 낮은 비용으로 단일 모델과 동등한 성능을 얻을 수 있었습니다. 이 연구 결과는 A2R이 단순한 성능 향상 프레임워크일 뿐만 아니라, 실제 적용에도 효율적이고 실용적인 솔루션이라는 것을 보여줍니다.



### The Thinking Spectrum: An Emperical Study of Tunable Reasoning in LLMs through Model Merging (https://arxiv.org/abs/2509.22034)
- **What's New**: 이 연구는 다양한 추론 깊이와 비용을 균형화할 수 있는 대형 언어 모델(LLM)의 모델 병합 기법을 새롭게 탐구합니다. 기존의 다양한 병합 기법 중 이 연구는 각각의 모델이 갖고 있는 특징을 조합하여 정확성과 효율성을 동시에 향상시키는 새로운 접근 방식을 제시합니다. 이번 연구는 모델의 병합 강도를 체계적으로 변화시켜 성능과 효율성을 분석하는 첫 번째 포괄적 연구를 제공합니다.

- **Technical Details**: 연구팀은 Linear, SLERP, TIES, TWIN, EMR, DARE, LORE와 같은 일곱 가지 대표적인 모델 병합 기법을 평가했습니다. 또한, AIME24, AIME25, HMMT25와 같은 다섯 가지 다양한 벤치 마크에서 이 기법들의 효율성 곡선을 구성하였습니다. 모델 병합은 특정 요구에 맞춰 추론 효율성과 정확도의 균형을 맞출 수 있는 방법으로, 각 기법을 통해 더욱 세밀한 제어가 가능합니다.

- **Performance Highlights**: 모델 병합 기술은 고도로 다양한 매개변수를 조합하여 티어레벨에서 높은 정확도와 더 낮은 토큰 소비를 달성하는 것으로 확인되었습니다. 특히, 병합된 모델이 원래 모델보다 더 좋은 성능을 보이는 Pareto 개선 사례가 다수 발견되었습니다. 이 연구는 모델 병합의 적용 가능성과 실용적인 가이드라인을 제시하여 진화하는 LLM의 요구에 대응할 수 있는 방법을 보여줍니다.



### ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models (https://arxiv.org/abs/2509.21991)
- **What's New**: 이번 연구에서는 고해상도 이미지 처리의 효율성을 높이기 위한 새로운 모델인 ERGO (Efficient Reasoning & Guided Observation)를 발표했습니다. 기존의 대형 비전-언어 모델(LVLMs)은 비디오 토큰 수로 인해 높은 계산 비용이 발생하는데, 이를 해결하기 위해 두 단계의 "coarse-to-fine" 추론 파이프라인을 제안했습니다. 분석된 이미지에서 중요한 지역을 식별한 후, 해당 영역만을 높은 해상도로 크롭하여 후속 재처리를 진행하는 방식입니다.

- **Technical Details**: ERGO는 강화 학습( Reinforcement Learning) 기반의 모델로, 이미지와 쿼리를 기반으로 이미지에서 중요한 영역의 바운딩 박스를 생성합니다. 이때, 전체 이미지에서 초점이 되는 영역을 재구성한 후 해당 영역에 대한 정확한 질문 응답을 수행합니다. 연구자는 이 모델이 비전 처리 효율성과 함께 정밀한 추론 능력도 유지할 수 있도록 설계했다고 설명합니다.

- **Performance Highlights**: 여러 데이터셋에서 ERGO는 기존 모델들과 비교하여 정확도를 크게 향상시키며, 적은 수의 비전 토큰 사용으로도 성능이 우수함을 입증했습니다. 예를 들어, ERGO는 V* 벤치마크에서 Qwen2.5-VL-7B를 4.7점 초과하며 비전 토큰의 23%만을 사용하여 3배의 추론 속도를 달성하였습니다. 이러한 결과는 고해상도 비전-언어 처리의 효율성과 정확성을 동시에 개선할 수 있음을 보여줍니다.



### From Bias to Balance: Exploring and Mitigating Spatial Bias in LVLMs (https://arxiv.org/abs/2509.21984)
- **What's New**: 본 연구는 LVLMs의 공간 편향(spatial bias)에 대한 체계적인 검토를 제공하며, 동일한 시각 정보가 이미지 내에서 서로 다른 위치에 배치될 때 모델이 어떻게 반응하는지를 분석합니다. 기존 연구들이 시각적 콘텐츠의 공간적 이해가 약하다는 점을 지적하고 있으나, 이는 기본적으로 LLM(대형 언어 모델) 부분에서 발생하는 문제라는 점을 강조합니다. 이 연구는 Balanced Position Assignment (BaPA)라는 새로운 메커니즘을 제안하여, 모든 이미지 토큰에 같은 위치 임베딩(position embedding)을 부여함으로써 시각적 정보 통합의 균형을 촉진합니다.

- **Technical Details**: LVLM은 비전 인코더(vision encoder)와 대형 언어 모델(LLM)을 결합하여 시각적 그리고 텍스트 정보를 동시에 활용합니다. 공간적인 위치에 따른 모델의 반응을 분석하기 위해 설계된 탐침 데이터셋을 사용하여, 현재의 LVLM이 위치 변화에 민감하게 반응하며 일관성이 결여된 결과를 생산하는 경향을 발견했습니다. 분석 결과, 이러한 현상은 비전 인코더에서 파생되지 않고, LLM의 위치 임베딩(embedding) 설계의 불균형에서 기인한다고 밝혀졌습니다.

- **Performance Highlights**: BaPA를 적용한 실험은 LVLM의 공간적 견고성을 향상시키며, 재학습 없이도 성능 향상을 나타냅니다. 또한, BaPA는 경량의 미세 조정(lightweight fine-tuning)과 결합하였을 때, 다양한 멀티모달 벤치마크에서 성능을 추가로 향상시킵니다. 최종적으로, BaPA를 적용한 LVLM은 노출된 테스트 셋에서 더 균형 잡힌 주의를 생성함으로써 시각적 정보에 대한 보다 총체적인 이해를 가능하게 합니다.



### RISK: A Framework for GUI Agents in E-commerce Risk Managemen (https://arxiv.org/abs/2509.21982)
- **What's New**: 이번 논문에서는 e-commerce 리스크 관리의 복잡한 웹 상호작용을 자동화하기 위한 새로운 프레임워크인 RISK를 소개합니다. 기존의 GUI 에이전트가 단일 단계 작업에 한정되는 반면, RISK는 다단계 상호작용을 지원하여 리스크 평가에 필수적인 동적 콘텐츠를 효과적으로 관리할 수 있습니다. RISK는 데이터셋, 벤치마크 및 강화 학습 기반의 세 가지 주요 구성 요소로 구성되어 있습니다.

- **Technical Details**: RISK 프레임워크는 (1) RISK-Data, (2) RISK-Bench, (3) RISK-R1으로 구성되어 있습니다. RISK-Data는 총 8,492개의 단일 단계와 2,386개의 다단계 상호작용 경로로 이루어진 데이터셋이며, RISK-Bench는 802개의 단일 단계와 320개의 다단계 상호작용으로 평가됩니다. RISK-R1은 서로 다른 난이도의 작업을 고려한 강화 학습을 통해 GUI 에이전트의 학습 프로세스를 개선합니다.

- **Performance Highlights**: RISK-R1의 실험 결과, 기존 방법보다 단일 단계 성능이 6.8%, 다단계 성능이 8.8% 향상되었습니다. 온라인 평가에서도 70.5%의 높은 성공률을 기록하며, RISK는 e-commerce 리스크 관리 분야에서 최신 기술 수준을 향상시키고 있습니다. 이번 연구는 GUI 에이전트가 현실 세계의 복잡한 웹 상호작용을 자동화할 수 있는 강력한 도구임을 입증하였습니다.



### Evaluating Open-Source Large Language Models for Technical Telecom Question Answering (https://arxiv.org/abs/2509.21949)
Comments:
          Accepted at the IEEE GLOBECOM Workshops 2025: "Large AI Model over Future Wireless Networks"

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 통신과 같은 기술 분야에서의 성능을 탐색하며, Gemma 3 27B와 DeepSeek R1 32B라는 두 개의 오픈소스 LLM을 평가하였습니다. 이 모델들은 고급 무선 통신 자료에서 파생된 사실 기반 및 추론 기반 질문에 대해 평가되었습니다. 연구진은 105개의 질문-답변 쌍을 구성하고 다양한 메트릭을 통해 성능을 분석하였습니다.

- **Technical Details**: 연구에서는 Lexical metrics와 semantic similarity, LLM-as-a-judge scoring을 사용하여 모델 성능을 평가했습니다. 또한, Source attribution과 score variance를 통해 일관성(consistency), 판단 신뢰성(judgment reliability), 그리고 망상(hallucination)을 분석하였습니다. 이러한 분석을 통해 기술 분야에 적합한 모델의 필요성과 현재의 한계를 강조하고 있습니다.

- **Performance Highlights**: Gemma는 의미적 충실도(semantic fidelity)와 LLM 평가에 따른 정확성에서 뛰어난 성능을 보였습니다. 반면, DeepSeek는 약간 높은 어휘 일관성(lexical consistency)을 나타냈습니다. 이 연구는 또한 엔지니어링 분야에서 신뢰할 수 있는 인공지능(AI) 보조 도구를 지원하기 위한 도메인 적합 모델의 필요성을 강조합니다.



### AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans (https://arxiv.org/abs/2509.21891)
- **What's New**: 이번 논문에서는 AgentPack을 소개하고, 이 데이터셋은 Claude Code, OpenAI Codex, Cursor Agent와 같은 소프트웨어 공학 에이전트와 인간이 공동으로 작성한 코드 수정 내역을 포함합니다. 기존의 데이터셋은 일반적으로 커밋 메시지가 간결하고 중복된 수정 사항이 많았지만, AgentPack은 더 명확한 목표를 가진 수정 사항을 포함하고 있습니다. 이러한 데이터셋은 머신러닝 모델의 성능을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: AgentPack은 2025년 4월부터 8월 중순까지 공개된 GitHub 프로젝트에서 수집된 130만 개의 코드 수정 사항으로 구성되어 있습니다. 이 데이터셋은 코드 파일이 여러 개의 파일에 걸쳐 있으며, 각 수정 사항은 LLM(대형 언어 모델) 의해 작성된 상세한 설명을 포함합니다. 또한, 에이전트가 작성한 테스트 코드와 함께 제공되는 점도 특징적입니다.

- **Performance Highlights**: AgentPack으로 미세 조정(fine-tuning)된 모델들은 과거의 사람만을 기반으로 한 커밋 데이터셋으로 훈련된 모델들보다 성능이 현저히 개선되었음을 보여줍니다. 이는 소프트웨어 공학 에이전트의 데이터 사용이 미래의 코드 편집 모델 훈련에 있어 강력한 가능성을 지닌다는 점을 강조합니다. 이 연구는 코드 편집 태스크에 대한 새로운 기준을 제시하는 중요한 진전을 이룹니다.



### You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors (https://arxiv.org/abs/2509.21884)
Comments:
          29 pages, 10 tables, 6figures, accepted by CCS 25

- **What's New**: 본 논문은 최신 대형 언어 모델(LLM)의 잠재적인 시스템 프롬프트 리크(Leak) 공격에 대한 새로운 접근법을 제안합니다. 기존 모델들은 공격 패턴을 인지할 경우 시스템 프롬프트를 반복하지 않도록 설계되었지만, 여전히 예기치 않은 공격 방식에 취약합니다. 이에 본 논문은 SysVec라는 새로운 방법을 통해 시스템 프롬프트를 텍스트(context) 대신 내부 표현 벡터로 인코딩하여 무단 공개를 최소화하고 LLM의 언어 능력을 유지합니다.

- **Technical Details**: SysVec는 시스템 프롬프트를 LLM의 내적 벡터 공간에 변환하여, 프롬프트 리크 공격으로부터 보호합니다. 이 방법은 시스템 프롬프트가 더 이상 텍스트 출력을 통해 노출되거나 반복되지 않도록 설계되었습니다. 또한, 이 방법은 추론 오버헤드를 줄이고 긴 입력을 처리하는 능력을 향상시키며 메모리 관리를 세밀하게 제어할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, SysVec는 프롬프트 리크 공격을 효과적으로 완화시키고 LLM의 기능 무결성을 유지하며 장기 맥락에서의 망각 문제를 완화하는 데 기여합니다. 이 연구는 LLM의 보안성을 강화하고, 특정 사용자 요구에 맞춰 모델의 일반적인 지침 따르기 능력을 개선할 수 있음을 보여줍니다.



### What Makes LLM Agent Simulations Useful for Policy? Insights From an Iterative Design Engagement in Emergency Preparedness (https://arxiv.org/abs/2509.21868)
- **What's New**: 이 논문에서는 Large Language Models (LLMs) 에이전트를 정책을 위한 유용한 도구로 설계하는 방법을 다룹니다. 연구팀은 대학 비상 대비 팀과의 1년 간의 디자인 참여를 통해 다양한 긴급 시나리오에서 군중의 이동과 커뮤니케이션을 시뮬레이션하는 13,000 LLM 에이전트 시스템을 개발했습니다. 이러한 시뮬레이션은 실제 정책 변화에 기여했고, 자원 봉사자 교육 프로그램 및 대피 프로토콜 개선에 영향을 미쳤습니다.

- **Technical Details**: 우리는 LLM 에이전트를 통해 인간 행동 시뮬레이션의 가능성을 탐구하고 있습니다. 초기 설계 단계에서 조사된 기존의 의사 결정 과정을 기반으로, 특정 이벤트인 졸업식 복잡한 집회에서의 대피 준비를 주제로 설정했습니다. 이 연구는 총 5단계에 걸쳐 비상 시나리오에 적합한 모델을 반복적으로 수정하며, 각 단계에서 정책 담당자의 피드백을 통합하고 실제 데이터에 대한 유효성을 검증하였습니다.

- **Performance Highlights**: 연구는 LLM 에이전트가 정책 개발에 실질적으로 기여하도록 하는 세 가지 주요 디자인 시사점을 도출했습니다. 첫째, 검증 가능한 시나리오로 시작하여 신뢰를 점진적으로 구축하는 것이 중요합니다. 둘째, 초기 시뮬레이션을 통해 암묵적인 지식을 끌어내는 것이 필요합니다. 마지막으로, 시뮬레이션과 정책 개발은 함께 진화해야 하며, 이러한 접근 방식이 LLM 에이전트 시뮬레이션의 실제적인 유용성을 높이는 길을 시사합니다.



### SBFA: Single Sneaky Bit Flip Attack to Break Large Language Models (https://arxiv.org/abs/2509.21843)
Comments:
          10 pages, 4 figures, 5 tables, 2 equations. Topics: Bit-flip attacks, adversarial attacks, large language models (LLMs)

- **What's New**: 본 논문에서는 Sneaky Bit-Flip Attack(SBFA)라는 새로운 공격 기법을 처음으로 제안합니다. 이 기법은 LLM의 성능을 오직 한 개의 비트 플립으로 크게 저하시킬 수 있으며, 기존의 기법들과 달리 비트 플립된 값이 benign layer-wise weight distribution 내에서 유지될 수 있도록 설계되었습니다. 기존 Bit-Flip Attack(BFA) 접근법들이 정수 또는 부동 소수점 모델 각각에 국한되었던 것과는 달리, SBFA는 다양한 모델에 적용 가능성을 높였습니다.

- **Technical Details**: SBFA는 새로운 파라미터 민감도 메트릭인 ImpactScore를 기반으로 하여 가장 중요한 비트를 찾아내는 iterative searching과 ranking 기법을 사용합니다. ImpactScore는 weight gradient sensitivity와 benign layer-wise weight distribution에 제한된 perturbation range를 통합하여 가장 취약한 비트를 식별하도록 합니다. 또한 SKIP(searching with Impact Prioritization)이라는 경량 검색 알고리즘을 도입하여 비트 플립을 위한 비트 선택 과정을 혁신적으로 개선하였습니다.

- **Performance Highlights**: Qwen, LLaMA, Gemma와 같은 여러 LLM 아키텍처에서 SBFA를 적용한 결과, 단 한 번의 비트 플립으로 성능을 랜덤 추측 수준 이하로 떨어뜨릴 수 있음을 발견했습니다. BF16 및 INT8 데이터 형식에서 실행된 실험은 SBFA의 효율성과 효과성을 입증하였으며, 이는 LLM들이 상대적으로 높은 보안 취약점을 지니고 있음을 강력히 시사합니다.



### Compiling by Proving: Language-Agnostic Automatic Optimization from Formal Semantics (https://arxiv.org/abs/2509.21793)
- **What's New**: 이 논문에서는 검증 증명을 최적화된 실행 규칙으로 변환하는 'compiling by proving' 패러다임을 제안합니다. 이러한 방식은 프로그램의 모든 실행 경로와 상태 변환을 포착하는 All-Path Reachability Proofs를 통해 이루어지며, 이를 통해 여러 의미론적 재작성 작업을 단일 규칙으로 통합할 수 있습니다. 또한, 이는 K 프레임워크의 언어 무관적 확장을 통해 구현됩니다.

- **Technical Details**: K 프레임워크는 형식적 의미론을 정의하고 언어 도구를 생성하기 위한 시스템입니다. 이 시스템에서 프로그램 사양은 컴파일 경계를 정의하며, All-Path Reachability Proofs는 실행 경로를 포착하는 그래프 구조로 표현됩니다. 또한, 증명 기반의 컴파일을 통해 검증 증명을 최적화된 실행 규칙으로 변환하는 알고리즘이 제시됩니다.

- **Performance Highlights**: 이 연구에서는 EVM의 개별 opcode 및 전체 프로그램에 대한 성능 개선을 평가하였으며, 최적화 범위에 따라 일관된 속도 향상을 보였습니다. opcode 레벨 최적화는 일관된 성능 향상을 보여주며, 전체 프로그램 컴파일에서는 수치적으로 더 큰 성능 향상을 달성했습니다. 이러한 성과는 검증과 최적화를 통합하는 새로운 접근 방식을 제시합니다.



### DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate Speech in Images (https://arxiv.org/abs/2509.21787)
Comments:
          Defactify 3 workshop at AAAI 2024

- **What's New**: 이번 연구는 디지털 콘텐츠에서 증오를 식별하기 위한 다중 모드(multi-modal) 데이터 세트를 도입합니다. 이 데이터 세트는 수위가 있는 안정화된 확산(stable diffusion) 기술과 디지털 주의 분석 모듈(Digital Attention Analysis Module, DAAM)을 결합하여 혐오 요소를 인식하는데 중점을 둡니다. 이를 통해 이미지 속 혐오 구역을 흐리게 처리하여 적절한 AI 응용 프로그램을 향상시키고자 하는 노력을 담고 있습니다.

- **Technical Details**: 연구는 텍스트와 이미지 간의 일관된 정렬을 위한 프롬프트 엔지니어링(prompt engineering) 및 안정화된 확산 프로세스에서 생성된 이미지의 혐오 요소를 중화하는 독특한 언어-이미지 모델인 DeHater를 포함합니다. 본 연구는 Hatenorm 데이터 세트를 활용하여 악성 텍스트를 규명하고, 이 텍스트를 바탕으로 합성된 이미지를 생성합니다. 또한, DAAM을 활용하여 생성된 이미지 내의 혐오적인 요소를 정확하게 핀포인트하고 이를 효과적으로 흐리게 처리하는 새로운 방법론을 제시합니다.

- **Performance Highlights**: 우리의 데이터 세트는 총 2411개의 사례를 포함하고 있으며, 교육 세트와 테스트 세트로 나뉘어 있습니다. Defactify 3 워크숍의 dehate 공유 작업의 일환으로 데이터가 제공되었습니다. 향후 연구를 위해 IOU(Intersection over Union) 메트릭을 활용해 참여자들의 성과를 평가하였고, 20개 이상의 등록과 5개의 제출이 있었습니다.



### UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios (https://arxiv.org/abs/2509.21766)
- **What's New**: 최근 자율 에이전트(agents)들은 다양한 분야에서 괄목할 만한 발전을 이루었지만, 대부분의 평가는 짧은 지평의 완전 관측(task) 작업에 집중되어 있습니다. 그러나 대규모 소프트웨어 개발, 상업 투자 및 과학 발견과 같은 현실 세계의 중요한 작업은 장기 지평(long-horizon) 및 부분 관측(partially observable) 시나리오에서 진행됩니다. 이러한 환경에서 성공은 지속적인 추론(reasoning), 계획(planning), 메모리 관리(memory management), 도구 사용(tool use)에 달려 있습니다.

- **Technical Details**: 이에 따라 우리는 복잡한 현실 세계의 도전에 필수적인 기초 능력을 측정하기 위한 새로운 벤치마크인 \textbf{UltraHorizon}을 소개합니다. 이 벤치마크는 세 가지 서로 다른 환경을 통해 탐사(exploration)를 통일된 작업으로 사용하여 이러한 핵심 역량을 검증합니다. 장기 발견 작업에서 설계된 에이전트는 숨겨진 규칙을 반복적으로 발견해야 하며, 평균적으로 \textbf{200k+} 토큰과 \textbf{400+} 툴 호출을 진행합니다.

- **Performance Highlights**: 우리의 실험 결과, LLM-agents는 이러한 설정에서 일관되게 낮은 성능을 보이는 반면, 인간 참가자는 더 높은 점수를 기록하여 에이전트의 장기 지평 능력에서 지속적인 격차를 강조합니다. 또한 단순한 스케일링(scaling)만으로는 이러한 작업을 해결할 수 없음을 발견하였습니다. 수집된 경로(trajectories)에 대한 심층 분석을 통해 여덟 가지 유형의 오류를 확인하였으며, 이를 두 가지 주요 원인인 컨텍스트 잠금(in-context locking)과 기능적 기본 능력의 격차로 귀속시켰습니다.



### UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments (https://arxiv.org/abs/2509.21733)
- **What's New**: 이번 논문에서는 UISim이라는 새로운 이미지 기반 UI(simulator) 시뮬레이터를 소개합니다. 이 시스템은 단순한 화면 이미지에서 포괄적으로 모바일 환경을 탐색할 수 있는 다이나믹하고 인터랙티브한 플랫폼을 제공합니다. 기존의 방법들에서 느껴지는 제약을 해소하며, UI 전환을 현실감 있게 시뮬레이션할 수 있도록 설계되었습니다.

- **Technical Details**: UISim은 초기 전화 화면 이미지와 사용자 행동을 바탕으로 두 단계의 방법론을 사용하여 다음 UI 상태를 생성합니다. 첫 번째 단계에서 추상적인 레이아웃 정보를 예측하고, 두 번째 단계에서 이 정보를 기반으로 새로운 시각적으로 일관된 UI 이미지를 합성합니다. 이러한 구조의 분리는 복잡한 이미지 간 UI 변환 문제를 보다 관리하기 쉬운 하위 문제로 나누어, 높은 충실도와 다양한 생성 능력을 제공합니다.

- **Performance Highlights**: 실험 결과에서 UISim은 이전의 UI 생성 모델보다 36.73의 Fréchet Inception Distance로 우수한 성능을 보여주었습니다. 이는 UISim이 현실적이고 일관된 다음 UI 상태를 생성하는 데 효과적임을 입증합니다. 또한, UI 테스트와 신속한 프로토타입 제작, 고급 응용 프로그램에도 유용하여 AI 에이전트의 UI 내비게이션 작업 계획에 기여할 수 있습니다.



### InvBench: Can LLMs Accelerate Program Verification with Invariant Synthesis? (https://arxiv.org/abs/2509.21629)
- **What's New**: 이 논문은 프로그램 검증(verification)에서 루프 불변식(loop invariants)을 자동으로 발견하는 새로운 방법론을 제시합니다. 기존의 연구에서는 LLMs(대형 언어 모델)들이 이 작업에서 얼마나 효과적인지 평가하는 데 오류가 있었으며, 이에 대한 보다 체계적인 평가를 위한 InvBench라는 프레임워크를 도입합니다. 이 프레임워크는 LLM들이 생성한 불변식의 정확성과 검증 속도 개선 효과를 동시에 평가합니다.

- **Technical Details**: 루프 불변식은 각 루프 반복 전후에 참이 되는 조건으로, 프로그램 검증의 핵심 컴포넌트입니다. 이 연구에서는 검증 과정의 속도 향상 정도를 측정하여 불변식의 유용성을 평가하는 방법론을 개발했습니다. 고전적인 검증 도구 UAutomizer와의 비교를 통해 LLM 기반 검증기의 성능을 조사하며, 모델의 성능 차이를 분석하고 있습니다.

- **Performance Highlights**: 실험 결과, 7개의 최신 LLM들이 평가되었고 이들 중 Elite 모델은 높은 성능을 보였으나, UAutomizer와 비교할 때 실질적인 우위를 보이지 못했습니다. 더불어, 3589개의 인스턴스를 활용한 지도 학습(‘supervised fine-tuning’)과 Best-of-N 샘플링 기법을 통해 LLM의 성능이 크게 향상될 수 있음을 입증했습니다. 예를 들어, Qwen3-Coder-480B 모델의 속도 향상 사례 비율이 8%에서 29.2%로 증가했습니다.



### AUDDT: Audio Unified Deepfake Detection Benchmark Toolk (https://arxiv.org/abs/2509.21597)
- **What's New**: AI가 생성한 콘텐츠, 특히 오디오 딥페이크의 확산에 따라, 저자들은 28개의 기존 오디오 딥페이크 데이터셋을 체계적으로 리뷰하고 AUDDT라는 오픈소스 벤치마킹 툴킷을 제안합니다. 이 툴킷은 사용자가 프리트레인(pretrained) 딥페이크 탐지기를 다양한 데이터셋에서 자동으로 평가할 수 있도록 해줍니다. 특히, 이 연구는 실제 상황에서의 딥페이크 정확도를 검증할 수 있는 방법론을 제공합니다.

- **Technical Details**: AUDDT는 28개의 데이터셋을 포함하고 있으며, 이는 다양한 생성 방법, 언어, 악센트 및 왜곡 조건을 포괄합니다. 이러한 구조 덕분에, 사용자는 딥페이크 탐지기의 강점과 약점을 직접적으로 피드백 받을 수 있도록 설계되었습니다. 논문에서는 ASVspoof2019에서 프리트레인된 탐지기의 성과를 기준으로 여러 데이터 카테고리에서 성능 변동을 시각화합니다.

- **Performance Highlights**: 기존의 여러 데이터셋에서 평가된 탐지기들은 현실 세계의 다양한 상황에서 성능 저하를 경험하고 있습니다. AUDDT 툴킷을 통해 수행된 검증에서, 탐지기의 성능이 데이터 카테고리마다 큰 차이를 보이는 것을 발견했습니다. 이러한 결과는 통합적이며 포괄적인 벤치마크의 필요성을 더욱 부각시키고 있으며, 현실 세계의 조건과의 불일치를 해소하기 위한 중요성을 강조합니다.



### Leveraging Big Data Frameworks for Spam Detection in Amazon Reviews (https://arxiv.org/abs/2509.21579)
Comments:
          Accepted & presented at THE 16th INTERNATIONAL IEEE CONFERENCE ON COMPUTING, COMMUNICATION AND NETWORKING TECHNOLOGIES (ICCCNT) 2025

- **What's New**: 이번 연구는 온라인 쇼핑에서의 제품 리뷰의 중요성을 강조하고, 이에 영향을 미치는 가짜 리뷰의 문제를 해결하고자 합니다. 머신러닝(machine learning) 기술과 빅데이터 분석을 통해 아마존 제품 리뷰 데이터에서 스팸 리뷰(spam reviews)를 효과적으로 탐지하고 분류하는 방법을 제안합니다.

- **Technical Details**: 연구에 사용된 접근 방식은 대규모 리뷰 데이터를 처리하기 위한 확장 가능한 빅데이터 프레임워크를 포함합니다. 주요 특징(feature)을 추출하여 소비자를 혼란스럽게 하는 fraudulent behavior(사기 행동)를 가리키는 요소를 파악함으로써 정확한 탐지를 목표로 하고 있습니다. 여러 머신러닝 분류기(classifiers)를 활용하여 스팸 리뷰를 탐지하는 성능을 비교하였습니다.

- **Performance Highlights**: 로지스틱 회귀(Logistic Regression) 모델은 90.35%의 높은 정확도를 기록하며 스팸 리뷰 탐지에서 주요한 성과를 보여주었습니다. 이러한 연구 결과는 소비자 신뢰를 높이고 더 투명한 온라인 쇼핑 환경을 조성하는 데 기여할 것으로 기대됩니다.



### Learning GUI Grounding with Spatial Reasoning from Visual Feedback (https://arxiv.org/abs/2509.21552)
- **What's New**: 본 논문에서는 GUI grounding을 인터랙티브 서치(task)로 재구성하여, Vision Language Models (VLMs)가 고해상도 GUI 이미지에서 효과적으로 UI 요소를 찾을 수 있도록 개선하는 방법을 제시합니다. 새로운 접근 방식인 GUI-Cursor는 커서를 움직이며 목표 객체를 찾는 과정에서 시각적 피드백을 제공하여 정확성을 높입니다.

- **Technical Details**: GUI grounding은 사용자가 자연어로 지시한 내용을 다양한 GUI 상호작용 단계로 변환하는 과정으로, 이 모델은 reinforcement learning(RL)을 활용하여 훈련됩니다. 이 과정은 모델이 커서를 특정 UI 요소로 이동시키며 공간(Spatial) 관계를 평가하고, 이전 이동 기록을 기반으로 새로운 위치를 예측하는 반복적인 방식으로 이루어집니다.

- **Performance Highlights**: GUI-Cursor는 ScreenSpot-v2 및 ScreenSpot-Pro 벤치마크에서 기존 최고 성과를 초월하며, 각각 93.9% 및 56.5%의 정확도를 기록했습니다. 특히, 95%의 사례에서 두 단계 내에 문제를 해결하는 능력을 보였으며, 어려운 작업에서는 더욱 많은 단계를 수행하는 적응력을 보여주었습니다.



### C-QUERI: Congressional Questions, Exchanges, and Responses in Institutions Datas (https://arxiv.org/abs/2509.21548)
- **What's New**: 이 논문에서는 미국 의회의 청문회에서 질문-응답 쌍을 추출하는 새로운 데이터셋을 개발하였습니다. 이를 통해 정치적 질문의 양식과 내용이 정당의 소속과 의석의 다수/소수 상태에 어떻게 영향을 받는지를 연구하는 기반을 제공합니다. 대규모 데이터셋의 부족으로 인해 전략적 정치 질문의 연구가 미흡했던 점을 해결하는 데 기여하고 있습니다.

- **Technical Details**: 연구팀은 108차부터 117차까지의 미국 하원과 상원 청문회에서 총 16,130개의 청문회 메모를 기반으로 300만 개 이상의 발화를 포함하는 새로운 데이터셋을 생성했습니다. 이 데이터셋은 질문과 응답 발화를 식별하고, 인간 주석을 통해 발화의 분류를 확인하며, 언어적 특징 세트를 제공합니다. BERT 언어 모델을 기반으로 한 분류기(classifier)를 개발하여 질문의 내용과 전략의 차이를 포착하고, 권위 있는 대형 언어 모델의 성능을 평가하였습니다.

- **Performance Highlights**: 질문만으로 질문자의 정당 소속을 59%의 정확도로 예측할 수 있음을 보여주었습니다. 다양한 위원회, 세션, 청문회 유형과 특정 조건의 조합에 따라 정확도가 70% 이상으로 상승하기도 하였습니다. 이러한 결과는 정치적 대립이 단순히 투표에 국한되지 않고, 청문회의 심의 및 감시 기능에서도 두드러진다는 것을 나타냅니다.



### Uncertainty-Aware Knowledge Tracing Models (https://arxiv.org/abs/2509.21514)
Comments:
          10 pages, 7 figures. Joshua Mitton and Prarthana Bhattacharyya contributed equally to this paper

- **What's New**: 이 연구는 Knowledge Tracing (KT) 모델 간의 예측 불확실성을 포착하여 모델의 성능을 개선하는 새로운 접근 방식을 제안합니다. 학생들이 선택하는 방해 요소(distractor)와 연관된 잘못된 예측에 대해 KT 모델에서의 불확실성이 중요하다는 점을 강조합니다. 이러한 불확실성 신호는 교육 플랫폼에서 학생들의 능력을 이해하는 데 유용하게 활용될 수 있습니다.

- **Technical Details**: KT 모델에서의 예측 불확실성을 고려하기 위해 Monte Carlo Dropout 방식을 활용하여 4가지 서로 다른 KT 모델의 예측 불확실성을 추정합니다. 이 연구에서는 기존의 이진 KT 모델을 확장하여 다중 클래스 설정으로 발전시켜, 학생들이 어느 특정 방해 요소를 선택할 가능성을 예측합니다. 모델의 출력_entropy에서는 예측 불확실성을 정량화하여 학생의 응답에 대한 불확실성을 명확히 파악할 수 있도록 하고 있습니다.

- **Performance Highlights**: 실험 결과, 모델의 예측이 잘못되었을 때 평균 총 엔트로피가 증가하는 것을 발견했습니다. 이는 잘못된 예측일 경우 모델 불확실성이 높다는 것을 시사하며, 이러한 불확실성을 활용하면 저확신 예측으로 인해 잘못된 결과가 발생하는 것을 줄일 수 있습니다. 다양한 KT 모델들이 평균 총 엔트로피가 유사한 경향을 보이며, 각 모델의 성능을 비교하여 불확실성을 포착하는 방법의 차이를 보여줍니다.



### LLM Agent Meets Agentic AI: Can LLM Agents Simulate Customers to Evaluate Agentic-AI-based Shopping Assistants? (https://arxiv.org/abs/2509.21501)
- **What's New**: 이 논문에서는 40명의 참가자를 모집하여 Amazon Rufus와 함께 쇼핑하는 경험을 분석했으며, 이들의 페르소나, 상호작용 기록 및 사용자 경험(UX) 피드백을 수집했다. 최초의 디지털 트윈(digital twin)을 만들어 이 쇼핑 작업을 반복함으로써 LLM 에이전트가 인간의 다중 턴 상호작용을 얼마나 밀접하게 반영할 수 있는지를 계량화하였다. LLM 에이전트는 인간 사용자와의 상호작용에서 유사한 피드백을 생성하면서도 보다 다양하고 창의적인 선택을 탐색하는 특성을 보였다.

- **Technical Details**: 이 연구는 Amazon의 대화형 쇼핑 어시스턴트인 Rufus를 중심으로 하여 진행되었다. 첫 번째 단계에서는 40명의 사용자에게 쇼핑 태스크를 수행하게 하여 다중 턴 상호작용의 데이터 세트를 생성하였다. 두 번째 단계에서는 UXAgent를 통해 동일한 참여자의 디지털 트윈을 생성하고 쇼핑 작업을 반복 수행함으로써 수집된 데이터를 기반으로하여 인간과 디지털 트윈 상호작용을 비교 분석하였다.

- **Performance Highlights**: LLM 에이전트는 사용자의 쇼핑 행동을 의미 있게 근사할 수 있었으며, 전체 상호작용 빈도 및 최종 결정에서 인간과의 유사도를 0.9로 나타냈다. 그러나 에이전트와 인간 사이에는 제품 선택에서 약 2%의 유사성만 존재하였고, 이러한 차이는 LLM이 인간의 탐색 및 결정 전략을 보다 잘 모델링하도록 개선할 필요성을 제시하였다. 결론적으로, 디지털 트윈은 기능적 측면을 재현할 수 있지만, 인간의 고유한 사고 및 경험을 충분히 포착하는 데 한계가 있음을 보여주었다.



### Are Hallucinations Bad Estimations? (https://arxiv.org/abs/2509.21473)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 생성 모델(generative models)에서의 환각(hallucination)을 추정치가 실행 가능한 원인과 연결되지 않는 실패로 형식화합니다. 이러한 해석하에, 손실 최소화(loss-minimizing) 최적 추정기조차도 환각을 경험함을 보여줍니다. 이는 손실 최소화와 인간 수용 가능한 출력 간의 구조적 불일치(structural misalignment)를 재구성하며, 잘못된 보정(miscalibration)으로 인한 추정 오류를 설명합니다.

- **Technical Details**: 최고 밀도 영역(Highest Density Regions, HDR) 개념은 주어진 확률 질량을 포함하는 최소 볼륨 집합으로 정의되며, 이는 다변량 밀도를 시각화하는 데 유용한 알고리즘을 활용하여 다중 모드 구조를 보다 잘 드러냅니다. Conditional Density Regions(HCDRs)이라는 개념을 도입하여, 잠재 변수(latent variable)가 하나의 상태를 가질 때 특정 질량의 HDR에서 대상 분포(target distribution)의 기대값이 탈출하는 δ-hallucination의 원인을 설명합니다. 이는 다수의 상태를 가진 잠재 변수와 연관된 분포로 확장됩니다.

- **Performance Highlights**: 코인 집계(coin aggregation), 개방형 QA(open-ended QA), 텍스트-이미지(text-to-image) 실험을 통해 이론을 뒷받침하고 있습니다. 특히, 최적 추정기가 기대하는 출력 도메인에서 손실을 최소화하지만 여전히 δ-hallucinate 될 우연성을 보여주는 위의 다양한 분포가 구성됩니다. 이 연구는 환각 문제를 보다 깊이 이해하고 해결하는 데 기여할 것으로 기대됩니다.



### Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models (https://arxiv.org/abs/2509.21466)
- **What's New**: 이번 연구는 현대의 Text-to-Image 인공지능 모델이 사우디 아라비아에서의 전문 직종을 묘사할 때 성별 고정관념과 문화적 부정확성을 얼마나 지속하는지를 조사하였습니다. 1,006개의 이미지를 분석하여, 56개의 다양한 사우디 직업에 대해 중립적인 프롬프트(prompts)를 사용했습니다. 이 연구는 기존 모델들이 어떻게 사회적 편향(social biases)을 반영하는지를 드러냅니다.

- **Technical Details**: 연구에서는 ImageFX, DALL-E V3, Grok의 세 가지 인공지능 모델로 생성된 이미지를 각각 평가하였습니다. 두 명의 훈련된 사우디 평가자가 성별, 의상(wardrobe) 및 외모, 배경 및 설정, 활동 및 상호작용, 나이의 다섯 가지 기준으로 이미지를 평가했고, 세 번째 연구자가 불일치할 경우 중재하였습니다. 이러한 과정을 통해 10,100개의 개별 판단을 확보하였습니다.

- **Performance Highlights**: 결과는 성별 불균형이 뚜렷하게 나타났으며, 각 모델의 출력 결과는 ImageFX가 85% 남성, Grok가 86.6% 남성, DALL-E V3는 96% 남성으로 나타났습니다. 특히 DALL-E V3가 가장 강한 성별 고정관념을 보였으며, 이러한 불균형은 리더십 및 기술직에서 특히 두드러졌습니다. 문화적 부정확성도 확인되었고, 이는 꼭 진보적인 묘사라기보다 문화적 오해로 인한 것임을 보여줍니다.



### VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding (https://arxiv.org/abs/2509.21451)
Comments:
          Work in progress

- **What's New**: 이 연구에서는 VideoJudge라는 새로운 다중 모달 대형 언어 모델(MLLM)을 제안합니다. VideoJudge는 3B 및 7B 크기로 비디오 이해 모델의 출력을 평가하는 데 특화되어 있습니다. 기존의 평가 지표들이 인간의 미세한 판단을 포착하지 못하는 데 반해, VideoJudge는 비디오 기반의 텍스트 응답을 직접 평가할 수 있는 가능성을 보여줍니다. 이는 비디오 이해 작업에 중심을 두고 대형 언어 모델을 활용한 첫 시도로 여겨집니다.

- **Technical Details**: VideoJudge는 생성자(generator)와 평가자(evaluator) 간의 상호작용을 기반으로 훈련됩니다. 평가자 모델이 예상과 다른 평가를 내릴 경우 해당 응답은 버려지며, 이 과정에서 훈련 데이터가 자동으로 생성됩니다. 또한, VideoJudge는 점수 예측과 함께 사례별 고유한 채점 기준(instance-specific rubrics)을 생성할 수 있는 기능을 갖추고 있습니다. 이러한 시스템은 대규모로 성과를 높이기 위한 첫 번째 단계로 작용합니다.

- **Performance Highlights**: VideoJudge-7B는 다른 대형 MLLM 평가 모델인 Qwen2.5-VL보다 세 가지 메타 평가 벤치마크에서 뛰어난 성능을 보였습니다. 특히, LLM 모델들이 MLLM 모델보다 효과적이지 않으며, 비디오 입력의 제공이 비디오 이해 작업의 평가에 있어 필수적임을 발견했습니다. VideoJudge는 인간의 평가와 더 높은 상관성을 보이며, 높은 샘플 효율성을 입증했습니다.



### ARTI-6: Towards Six-dimensional Articulatory Speech Encoding (https://arxiv.org/abs/2509.21447)
- **What's New**: 본 논문에서는 실시간 MRI 데이터에서 수집된 정보를 바탕으로 한 6차원 발음 장치 음성 인코딩 프레임워크인 ARTI-6을 제안합니다. 이 모델은 인체 생리학적으로 기반을 두고 있으며, 음성 생성을 위한 해석 가능하고 효율적인 방법을 제공합니다. ARTI-6는 발음 역전(articulatory inversion)과 음성 합성(articulatory synthesis)을 개선하기 위해 중요한 음성 기관을 다루며, 저차원의 표현이 자연스러운 음성을 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 발음 생산의 목표를 달성하는 데 필요한 6개의 주요 지역을 선택하였습니다: Lip Aperture (LA), Tongue Tip (TT), Tongue Body (TB), Velum (VL), Tongue Root (TR), 그리고 Larynx (LX). 발음 역전 모델은 Whisper, WavLM, HuBERT, Wav2Vec2와 같은 음성 기초 모델을 기반으로 구성되어 있으며, 이를 통해 얻은 결과는 고해상도의 발음 관련 데이터를 제공합니다. 합성 모델은 HiFi-GAN 아키텍처를 사용하고, ECAPA-TDNN에서 추출된 화자 임베딩으로 조건화하여 개별 화자의 특징을 반영할 수 있습니다.

- **Performance Highlights**: ARTI-6의 발음 역전 성능은 WavLM이나 HuBERT를 사용할 때 예측 상관 계수 0.872를 기록하며 가장 우수한 결과를 보였습니다. 발음 합성에서는 LSS 데이터셋과 LibriTTS-R 데이터셋을 조합하여 훈련되었으며, 다양한 손실 함수를 통해 모델의 동작이 최적화되었습니다. 실험 결과, ARTI-6는 실시간 애플리케이션에 적합한 저차원 표현을 제공함으로써 발음 기술 개발에 유용한 플랫폼이 될 것으로 기대됩니다.



### LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet? (https://arxiv.org/abs/2509.21403)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 실험 설계의 일반 목적 에이전트로 평가하며, 이에 대한 가설을 생명 과학 분야의 유전자 교란 및 분자 특성 발견 작업에서 테스트하였습니다. LLM 기반 에이전트는 실험 피드백에 민감하지 않음을 발견하였으며, 전통적인 방법들이 LLM 모델들보다 일관되게 더 나은 성능을 보였다는 점이 두드러집니다. 또한, LLM 사전 지식을 기반으로 하여 가까운 이웃 샘플링을 사용하는 하이브리드 방법인 LLMNN(LLM-guided Nearest Neighbour sampling)을 소개, 여러 도메인에서 경쟁력 있는 성능을 보이는 것을 확인하였습니다.

- **Technical Details**: 이 연구는 BioDiscoveryAgent 파이프라인을 통해 세 가지 공개 LLM 모델과 두 가지 폐쇄형 LLM 모델을 실험 설계 작업에 적용하였습니다. 특히, LLMNN 방식은 LLM이 제안한 후보를 바탕으로 가까운 이웃 확장을 통해 후보를 선택하는 방식을 채택하였으며, 이 과정에서도 실험 피드백을 활용하였습니다. LLM 기반 모델들은 피드백에 대한 성능에 민감하지 않은 것으로 나타났으며, 비전통적인 방법인 LLMNN이 우수한 성능을 보여주는 것이 관찰되었습니다.

- **Performance Highlights**: 논문에서 제안한 LLMNN 방법은 BDA와 비교했을 때 더 나은 검색 시 성과를 보였으며, 고전적인 방법들과 비슷하거나 초과하는 성능을 달성했습니다. 연구 결과에 따르면, 오픈 소스 및 폐쇄형 LLM 모델들은 실험 피드백을 활용하여 실험 설계에서 효율적으로 작동하지 못한다는 점이 강조되었습니다. 이는 LLM이 유용한 도메인 지식을 인코딩하였으나, 후행 업데이트 및 선택을 위한 명시적인 메커니즘이 필요하다는 것을 시사합니다.



### ReGeS: Reciprocal Retrieval-Generation Synergy for Conversational Recommender Systems (https://arxiv.org/abs/2509.21371)
Comments:
          Accepted by WISE 2025: 26th International Web Information Systems Engineering conference. Our code is publicly available at the link: this https URL

- **What's New**: 기존의 대화 기반 추천 시스템(CRS)은 외부 도메인 지식을 연결하는 데 어려움을 겪었지만, ReGeS라는 새로운 프레임워크는 이를 해결하는 데 혁신적인 방법을 제시합니다. ReGeS는 생성-증강 검색(generation-augmented retrieval)과 검색-증강 생성(retrieval-augmented generation)을 결합하여 사용자 의도를 보다 명확히 추출하고, 항목 간의 미세한 차이를 구분할 수 있게 합니다. 이를 통해 추가적인 주석 없이도 높은 품질의 추천을 제공하며 실험 결과에서도 최첨단 성능을 입증했습니다.

- **Technical Details**: ReGeS는 대화 내용을 기반으로 사용자의 의도를 파악하고 추천 항목을 선택하는 두 가지 주요 모듈을 포함합니다. 첫 번째 모듈인 생성-증강 검색은 LLM을 통해 간결하고 정보가 풍부한 쿼리를 생성하여 정확한 항목 검색을 돕습니다. 두 번째 모듈인 검색-증강 생성은 검색된 후보 항목을 사용하여 최종 추천의 정확성을 높이며, 이를 통해 노이즈가 있는 입력과 후보 항목 간의 구분 문제를 효율적으로 해결합니다.

- **Performance Highlights**: 여러 CRS 벤치마크에서 ReGeS는 추천 정확성에서 최첨단 성능을 달성했습니다. 이 프레임워크는 특히 능동적으로 사용자 의도를 반영하고, 대화의 지식적 수요를 충족시키는 데 효과적입니다. 회귀를 통한 체계적인 접근 방식 덕분에, ReGeS는 과거 방식들과 비교해 허위 정보를 줄이고 최신 상태로의 업데이트를 쉽게 만들어, 대화 기반 추천 시스템의 신뢰성을 크게 개선합니다.



### Random Direct Preference Optimization for Radiography Report Generation (https://arxiv.org/abs/2509.21351)
- **What's New**: 이번 연구는 Radiography Report Generation (RRG) 분야에서 기존 방법의 한계를 극복하기 위한 방법으로, Direct Preference Optimization (DPO) 프레임워크를 제안합니다. 이 방법은 임의의 대조적 샘플링을 활용하여 훈련 쌍을 구성하며, 추가적인 보상 모델이나 인간의 선호 주석이 필요하지 않습니다. 실험 결과, 세 가지 최첨단 모델에 우리의 Random DPO를 보완함으로써 최대 5%의 임상 성능 향상을 달성했습니다.

- **Technical Details**: RRG 모델은 전통적으로 다음 단어 예측을 위한 표준 크로스 엔트로피 손실 기준으로 최적화됩니다. 이러한 접근 방식은 실질적으로 모델이 잘못된 연관성을 학습하게 만들어 편향을 유발할 수 있습니다. 본 연구에서는 DPO를 기반으로 모델 성능을 향상시키기 위한 무작위 샘플링 방법을 제안하며, 이는 추가적인 데이터 준비 없이 기존 데이터셋을 활용하여 성과를 낼 수 있습니다.

- **Performance Highlights**: MIMIC-CXR, CheXpert Plus, IU X-ray, Interpret-CXR와 같은 네 개의 공개 데이터셋을 사용하여 실험을 진행했습니다. MIMIC-CXR는 377,110개의 이미지와 227,835개의 방사선 연구가 포함되어 있으며, CheXpert Plus는 223,228개의 독특한 방사선 보고서와 가슴 X선 쌍으로 구성되어 있습니다. 각 데이터셋은 다양한 임상 환경에서 수집되어 RRG 방법의 종합적인 평가를 허용합니다.



### Towards mitigating information leakage when evaluating safety monitors (https://arxiv.org/abs/2509.21344)
Comments:
          14 pages, 4 figures

- **What's New**: 이번 연구에서는 대형 언어 모델에서 잠재적인 위험 행동을 탐지하기 위한 화이트 박스 모니터(white box monitors)의 성능을 체계적으로 평가하기 위한 프레임워크를 제안합니다. 이러한 모니터는 모델의 내부를 분석할 수 있는 장점이 있으며, 응답 예시(response exemplars)를 필요로 하지만 이는 실제 성능에 영향을 미칠 수 있는 정보 유출(leakage) 문제를 동반합니다. 이 연구는 진정한 모델 동작을 탐지하는 능력을 평가하는 방법을 찾고 있습니다.

- **Technical Details**: 연구진은 세 가지 새로운 전략을 제안했습니다. 첫째, 콘텐츠 필터링(content filtering)을 통해 입력에서 기만 관련 텍스트를 제거하고, 둘째, 점수 필터링(score filtering)을 통해 작업과 관련된 토큰만 집계하며, 셋째, 기만 행동을 드러내기 위해 구체적인 프롬프트 없이 훈련된 모델 유기체(prompt distilled fine-tuned model organisms)를 사용합니다. 특히 기만 탐지를 Representative case study로 하여, 두 가지 형태의 유출을 확인했습니다: 유도 유출(elicitation leakage)과 추론 유출(reasoning leakage).

- **Performance Highlights**: 실험을 통해 여러 기만 벤치마크에서 제안된 완화 전략을 적용하고 성능 유지(performance retention)를 측정했습니다. 연구 결과, 첫째, 콘텐츠 필터링은 30%까지 AUROC(Area Under the Receiver Operating Characteristic curve)를 감소시키는 효과적 전략으로 나타났습니다. 둘째, 점수 필터링은 15%의 AUROC 감소를 보여주었지만 그 기여도를 명확히 알기 어렵습니다. 셋째, 훈련된 모델 유기체는 모니터 평가를 개선하지만, 다시 훈련하더라도 성능은 최대 40% 감소하는 결과를 보였습니다.



### HetaRAG: Hybrid Deep Retrieval-Augmented Generation across Heterogeneous Data Stores (https://arxiv.org/abs/2509.21336)
Comments:
          15 pages, 4 figures

- **What's New**: RAG(검색 강화 생성) 프레임워크인 HetaRAG가 소개되었습니다. HetaRAG는 여러 이질적인 데이터 저장소를 통합하여, 벡터 인덱스, 지식 그래프, 전체 텍스트 검색 엔진 및 관계형 데이터베이스를 하나의 검색 평면으로 묶어 동적으로 증거를 라우팅하고 융합하는 시스템입니다. 이 연구는 데이터 보안과 지식 정확성을 동시에 유지하면서도 LLM(대형 언어 모델)의 한계를 극복하는 방법을 제시합니다.

- **Technical Details**: HetaRAG는 두 단계로 구성된 파이프라인을 사용하여 멀티모달 문서 처리와 질문 응답 생성을 수행합니다. 첫 단계에서는 텍스트, 이미지, 표 및 수학 공식을 추출하고, 각 모달리티를 가장 적합한 저장 형식으로 변환합니다. 두 번째 단계에서는 사용자의 질문에 대한 답변을 생성하기 위해 여러 데이터 저장소에서 증거를 조율합니다.

- **Performance Highlights**: HetaRAG는 다음과 같은 핵심 기능을 제공합니다: 멀티모달 문서 파싱, 이질적인 저장소 인덱스 구축, 깊은 연구 보고서 생성, 그리고 복합 질문 응답을 위한 반복적인 추론 능력을 갖추고 있습니다. 이 시스템은 도메인 특화된 문제에 대한 신뢰할 수 있는 답변을 제공하며, 높은 정보 검색 정확도를 목표로 합니다.



### ZERA: Zero-init Instruction Evolving Refinement Agent -- From Zero Instructions to Structured Prompts via Principle-based Optimization (https://arxiv.org/abs/2509.18158)
Comments:
          9 pages, 4 figures. To appear in EMNLP 2025 Main Conference (Oral Presentation)

- **What's New**: 이번 논문에서는 ZERA(Zero-init Instruction Evolving Refinement Agent)라는 새로운 자동 프롬프트 최적화(APO) 프레임워크를 제안합니다. ZERA는 사용자와 시스템 프롬프트를 공동으로 최적화함으로써 낮은 오버헤드로 빠르게 고품질 프롬프트에 수렴할 수 있도록 설계되었습니다. 주목할 만한 점은 ZERA가 작업 샘플이 적은 상황에서도 효과적으로 프롬프트를 생성할 수 있는 능력을 가지고 있다는 것입니다.

- **Technical Details**: ZER는 두 단계의 반복적 프로세스인 원칙 기반 비평 생성(PCG)과 메타인지 프롬프트 개선(MPR)을 통해 프롬프트의 효과성을 평가하고 개선합니다. PCG는 주어진 작업에 대한 각 원칙의 상대적 중요성을 평가하고, 이를 통해 피드백을 생성하여 메타 정보를 업데이트합니다. 이러한 과정을 통해 ZERA는 강력한 일반화 능력을 갖춘 최적화된 프롬프트를 제공합니다.

- **Performance Highlights**: 실험 결과, ZERA는 총 5개의 LLM 및 9개의 다양한 데이터셋에서 강력한 기준선보다 일관된 성과 향상을 보였습니다. 특히 MMLU, GSM8K, HumanEval 등의 여러 벤치마크 작업에서 ZERA로 생성된 프롬프트가 미리 정의된 프롬프트보다 우수한 성능을 입증했습니다. 추가적인 ablation study를 통해 ZERA의 각 구성 요소가 효과적인 프롬프트 제작에 기여하는도를 정량적으로 평가했습니다.



New uploads on arXiv(cs.IR)

### Your RAG is Unfair: Exposing Fairness Vulnerabilities in Retrieval-Augmented Generation via Backdoor Attacks (https://arxiv.org/abs/2509.22486)
Comments:
          Accepted by EMNLP 2025

- **What's New**: 본 논문은 Retrieval-augmented generation (RAG) 모델의 공정성 취약점을 다루는 BiasRAG 프레임워크를 제시합니다. 이는 기존의 백도어(Backdoor) 공격과는 다르게, 세미나틱(Semantic) 관계를 활용하여 특정 그룹과 사회적 편견 간의 상호작용을 조작합니다. 이를 통해 RAG 모델의 콘텐츠 생성에 지속적이고 은밀한 영향을 미치는 새로운 공격 기법을 탐구합니다.

- **Technical Details**: BiasRAG는 두 단계 백도어 공격 전략을 사용하여 RAG의 공정성 문제를 파고듭니다. 교육 전 단계에서는 쿼리 인코더가 조작되어 목표 그룹의 임베딩과 사회적 편견을 정렬합니다. 이후 배포 후 단계에서는 지식 베이스에 악성 문서를 주입하여 편견 전파를 강화합니다.

- **Performance Highlights**: BiasRAG는 다양한 공정성 속성에 대한 평가에서 높은 공격 성공률을 보이며, 주요 기능을 유지하면서도 RAG의 생성 과정에 영향을 미칩니다. 본 연구는 RAG의 공정성 및 보안 위험을 강조하며, 이러한 공격이 어떻게 지속적이고 진화하는 위협이 될 수 있는지에 대한 실질적인 사례를 제공합니다.



### Can Synthetic Query Rewrites Capture User Intent Better than Humans in Retrieval-Augmented Generation? (https://arxiv.org/abs/2509.22325)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 콜로quial 생략과 모호한 참조가 포함된 쿼리에 대해 고품질 합성 쿼리를 생성하는 SynRewrite라는 새로운 쿼리 재작성 모델을 제안합니다. 기존의 쿼리 재작성 방식이 인간 주석자에게 의존했던 반면, 이제는 GPT-4o를 사용하여 대화 이력, 현재 쿼리, 긍정적인 문서 및 답변을 바탕으로 합성 쿼리를 생성합니다. 이 방법을 통해 사용자 의도를 더욱 정확하게 포착하고 시스템의 응답 품질을 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: SynRewrite 모델은 합성 데이터 기반으로 쿼리 재작성을 수행하며, Flan-T5 모델을 파인튜닝하여 대화 이력과 현재 쿼리를 입력받고 합성된 쿼리를 출력합니다. 이 과정에서 Direct Preference Optimization (DPO) 알고리즘을 사용하여 생성기 피드백을 통한 성능 향상을 달성합니다. 특히, GPT-4o를 활용해 고품질의 재작성 쿼리를 합성하는 과정은 실질적인 데이터 수요를 줄이는 데 기여합니다.

- **Performance Highlights**: TopiOCQA와 QRECC 데이터셋에서의 실험 결과, SynRewrite는 인간이 재작성한 쿼리보다 우수한 성능을 보여줍니다. 예를 들어, Retrieval 성능에서는 61.31의 MRR을 기록하며 이는 인간 재작성에 비해 두 배의 성능 향상을 의미합니다. 이러한 결과는 합성 데이터가 사용자 의도를 바르게 포착하는 데 효과적이라는 것을 입증합니다.



### Does Generative Retrieval Overcome the Limitations of Dense Retrieval? (https://arxiv.org/abs/2509.22116)
- **What's New**: 본 연구는 Generative Retrieval (GR)이 Dense Retrieval (DR)와 근본적으로 어떻게 다른지를 이론적 및 경험적으로 조사하였습니다. GR은 관련 문서의 식별자를 직접 생성하며, DR은 쿼리와 문서를 벡터로 변환하여 유사성을 측정하는 방식입니다. 이 분석을 통해 GR은 DR보다 학습 목표와 표현 용량에서 더 나은 성능을 발휘할 수 있다는 결론을 도출했습니다.

- **Technical Details**: GR은 글로벌 정규화 최대우도 최적화를 수행하며, 모델 파라미터에 직접적으로 코퍼스와 관련 정보가 인코딩됩니다. 반면 DR은 로컬 정규화 목표를 적용하고, 외부 임베딩을 사용하여 유사성을 계산합니다. 이러한 접근 방식에서 GR은 커다란 모델과 데이터로 스케일 시 DR의 한계를 극복할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, GR은 DR보다 큰 데이터 세트에서 성능 저하를 회피하며 더 나은 결과를 보였습니다. 그러나 GR은 항상 DR보다 우수한 성능을 보여주지 않으며, docid 설계 및 데이터 학습 구성에 따라 효과가 다릅니다. 향후 연구는 GR의 이론적 가능성과 실제 성능 간의 격차를 줄이기 위한 방향을 제시해야 합니다.



### GoalRank: Group-Relative Optimization for a Large Ranking Mod (https://arxiv.org/abs/2509.22046)
- **What's New**: 이번 논문에서는 추천 시스템의 중요성을 강조하며, 전통적인 Generator-Evaluator 두 단계 접근 방식의 한계를 논의합니다. 특히, 다수의 후보 리스트를 생성하여 성능을 높이려는 최근 시도를 비판하며, 단일 Generator 모델이 더 나은 성과를 낼 수 있음을 이론적으로 증명합니다. 이 연구는 Generator 모델에서 근사 오차를 줄일 수 있는 가능성을 보여주며, 이를 바탕으로 새로운 framework인 GoalRank를 제시합니다.

- **Technical Details**: 제안된 GoalRank 프레임워크는 Group-Relative optimization 원칙을 통해 대규모 단일 Generator 모델의 효과적인 학습을 가능하게 합니다. 이 과정에서 모델의 용량이 증가함에 따라 근사 오차가 감소한다는 스케일링 법칙을 공식화하며, 사용자의 피드백을 기반으로 한 보상 모델을 통해 최적 정책의 대리자를 구성하도록 설계됩니다. 이 방법론은 두 단계의 Generator-Evaluator 접근 방식과 비교하여 보다 효과적인 결과를 도출합니다.

- **Performance Highlights**: GoalRank는 공개 벤치마크와 대규모 온라인 A/B 테스트를 통해 기존의 최첨단 방법들보다 일관된 성능 향상을 보여주었습니다. 실험 결과, 이 모델은 추천 정확도와 사용자 만족도를 크게 개선시켜 플랫폼 수익에 긍정적인 영향을 미칠 것으로 기대됩니다. 또한, 다양한 스케일링 실험을 통해 이 모델의 확장 가능성을 강조하여, 향후 추천 시스템의 발전 방향에 기여할 것입니다.



### Effect of Model Merging in Domain-Specific Ad-hoc Retrieva (https://arxiv.org/abs/2509.21966)
Comments:
          Accepted at CIKM 2025, 5 pages

- **What's New**: 이번 연구에서는 ad-hoc retrieval 작업에서 모델 병합(model merging)의 효과를 평가합니다. 기존의 retrieval 모델과 도메인 특화 모델을 결합하여 새로운 모델을 만드는 방법을 제안하며, 이 과정에서 추가적인 미세 조정(fine-tuning)이 필요하지 않습니다. 연구의 주요 질문은 모델 병합이 소스 모델보다 더 효과적인 도메인 특화 모델을 만들 수 있는지 여부입니다.

- **Technical Details**: 모델 병합은 두 개의 모델을 선형 보간(linear interpolation) 방식으로 결합하여 새로운 모델을 생성하는 방법입니다. 본 연구에서는 e5-mistral-7b-instruct 모델을 소스 retrieval 모델로 사용하였고, 도메인 특화 모델은 해당 도메인에서 추가 훈련된 모델입니다. 또한, 각 레이어의 역할을 고려하여 모델을 두 세그먼트로 나누어 하이퍼 파라미터를 조정합니다.

- **Performance Highlights**: 실험 결과, 모델 병합은 소스 retrieval 모델보다 더 효과적인 도메인 특화 retrieval 모델을 생성할 수 있는 가능성을 보여주었습니다. 특히, 제한적인 데이터가 존재할 때 LoRA 미세 조정의 실용적인 대안으로 자리매김할 수 있습니다. 이러한 결과들은 ad-hoc retrieval 분야에서 모델 병합의 유용성을 입증하는 중요한 데이터로 작용합니다.



### MIXRAG : Mixture-of-Experts Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (https://arxiv.org/abs/2509.21391)
- **What's New**: MIXRAG (Mixture-of-Experts Graph-RAG) 프레임워크는 복수의 전문 그래프 검색기를 도입하여 다양한 쿼리 의도를 더 효과적으로 처리합니다. 이 프레임워크는 각 검색기가 엔터티, 관계 또는 서브그래프의 토폴로지와 같은 그래프 의미의 특정 측면에 집중하도록 훈련됩니다. 또한, 쿼리 인식 그래프 인코더를 통해 검색된 서브그래프에서 가장 관련성 높은 정보만 강조하여 잡음(noise)을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MIXRAG는 전문가 기반 접근법을 사용하여 여러 검색기 전문가를 동적으로 결합합니다. 각각의 검색기는 각기 다른 형태의 정보를 다루며, 쿼리의 의도와 전문가의 전문성에 기반하여 적합한 검색기를 선택하는 Mixture-of-Experts 모듈이 통합되어 있습니다. 이러한 구조는 쿼리에 적합한 정보 검색이 이루어지도록 하여 보다 정교한 추론이 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MIXRAG는 GraphQA 벤치마크에서 기존의 강력한 기준선을 초월하는 성능을 보였습니다. 전문가 기반 검색 및 세밀한 서브그래프 임베딩 기법이 하부 성능에 큰 기여를 함을 보여줍니다. 이러한 연구 결과는 복합적인 지식을 포착하고 활용하는 데 있어 MIXRAG의 효과를 뒷받침하고 있습니다.



### ReGeS: Reciprocal Retrieval-Generation Synergy for Conversational Recommender Systems (https://arxiv.org/abs/2509.21371)
Comments:
          Accepted by WISE 2025: 26th International Web Information Systems Engineering conference. Our code is publicly available at the link: this https URL

- **What's New**: 기존의 대화 기반 추천 시스템(CRS)은 외부 도메인 지식을 연결하는 데 어려움을 겪었지만, ReGeS라는 새로운 프레임워크는 이를 해결하는 데 혁신적인 방법을 제시합니다. ReGeS는 생성-증강 검색(generation-augmented retrieval)과 검색-증강 생성(retrieval-augmented generation)을 결합하여 사용자 의도를 보다 명확히 추출하고, 항목 간의 미세한 차이를 구분할 수 있게 합니다. 이를 통해 추가적인 주석 없이도 높은 품질의 추천을 제공하며 실험 결과에서도 최첨단 성능을 입증했습니다.

- **Technical Details**: ReGeS는 대화 내용을 기반으로 사용자의 의도를 파악하고 추천 항목을 선택하는 두 가지 주요 모듈을 포함합니다. 첫 번째 모듈인 생성-증강 검색은 LLM을 통해 간결하고 정보가 풍부한 쿼리를 생성하여 정확한 항목 검색을 돕습니다. 두 번째 모듈인 검색-증강 생성은 검색된 후보 항목을 사용하여 최종 추천의 정확성을 높이며, 이를 통해 노이즈가 있는 입력과 후보 항목 간의 구분 문제를 효율적으로 해결합니다.

- **Performance Highlights**: 여러 CRS 벤치마크에서 ReGeS는 추천 정확성에서 최첨단 성능을 달성했습니다. 이 프레임워크는 특히 능동적으로 사용자 의도를 반영하고, 대화의 지식적 수요를 충족시키는 데 효과적입니다. 회귀를 통한 체계적인 접근 방식 덕분에, ReGeS는 과거 방식들과 비교해 허위 정보를 줄이고 최신 상태로의 업데이트를 쉽게 만들어, 대화 기반 추천 시스템의 신뢰성을 크게 개선합니다.



### Cross-Modal Retrieval with Cauchy-Schwarz Divergenc (https://arxiv.org/abs/2509.21339)
Comments:
          Accepted by ACMMM-25

- **What's New**: 이 논문에서는 다중 모달 학습의 핵심 과제인 교차 모달 검색(Cross-Modal Retrieval, CMR)에 Cauchy-Schwarz (CS) divergence를 도입합니다. CS divergence는 초매개변수( hyperparameter)가 필요 없고, 수치적으로 안정적이며 선형적으로 확장 가능하여 기존 방법보다 뛰어난 성능을 보여줍니다. 또한, 새로운 Generalized CS (GCS) divergence를 제안하여 세 개 이상의 모달을 통합하여 비교할 수 있는 수학적 프레임워크를 제공합니다.

- **Technical Details**: CS divergence는 두 개 이상의 모달 간의 직접적인 정렬을 가능하게 하며, 고전적인 방법인 Kullback-Leibler divergence, Maximum Mean Discrepancy, Correlation Alignment의 한계를 극복합니다. 기존 방법들이 일반적으로 두 모달에 제한되는 것에 반해, CS divergence는 다수의 모달을 동시에 비교할 수 있는 회전 대칭 비교( bidirectional circular comparison)가 포함되어 있습니다. 이로 인해 계산 복잡성이 크게 줄어듭니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋을 활용한 포괄적인 실험 결과, CS divergence 기반 방법이 이중 모달 및 삼중 모달 검색 임무에서 모든 기존 접근 방식을 초월하는 성능을 보였습니다. 특히 전통적인 방법에서 요구되는 쌍 비교(pairwise comparisons) 없이도 우수한 검색 정확도를 달성하였습니다. 연구 코드 또한 공개되어 있어 연구자들이 쉽게 활용할 수 있도록 하고 있습니다.



### HetaRAG: Hybrid Deep Retrieval-Augmented Generation across Heterogeneous Data Stores (https://arxiv.org/abs/2509.21336)
Comments:
          15 pages, 4 figures

- **What's New**: RAG(검색 강화 생성) 프레임워크인 HetaRAG가 소개되었습니다. HetaRAG는 여러 이질적인 데이터 저장소를 통합하여, 벡터 인덱스, 지식 그래프, 전체 텍스트 검색 엔진 및 관계형 데이터베이스를 하나의 검색 평면으로 묶어 동적으로 증거를 라우팅하고 융합하는 시스템입니다. 이 연구는 데이터 보안과 지식 정확성을 동시에 유지하면서도 LLM(대형 언어 모델)의 한계를 극복하는 방법을 제시합니다.

- **Technical Details**: HetaRAG는 두 단계로 구성된 파이프라인을 사용하여 멀티모달 문서 처리와 질문 응답 생성을 수행합니다. 첫 단계에서는 텍스트, 이미지, 표 및 수학 공식을 추출하고, 각 모달리티를 가장 적합한 저장 형식으로 변환합니다. 두 번째 단계에서는 사용자의 질문에 대한 답변을 생성하기 위해 여러 데이터 저장소에서 증거를 조율합니다.

- **Performance Highlights**: HetaRAG는 다음과 같은 핵심 기능을 제공합니다: 멀티모달 문서 파싱, 이질적인 저장소 인덱스 구축, 깊은 연구 보고서 생성, 그리고 복합 질문 응답을 위한 반복적인 추론 능력을 갖추고 있습니다. 이 시스템은 도메인 특화된 문제에 대한 신뢰할 수 있는 답변을 제공하며, 높은 정보 검색 정확도를 목표로 합니다.



### PIR-RAG: A System for Private Information Retrieval in Retrieval-Augmented Generation (https://arxiv.org/abs/2509.21325)
- **What's New**: 본 논문에서는 개인 정보 보호를 고려한 Retrieval-Augmented Generation (RAG) 시스템인 PIR-RAG를 소개합니다. PIR-RAG는 대규모 AI 시스템에서 개인 정보를 안전하게 보호하기 위해 설계된 새로운 아키텍처로, 전체 문서 클러스터를 효율적으로 검색할 수 있는 기능을 제공합니다. 본 연구는 기존의 RAG 아키텍처에서의 개인 정보 노출 문제를 해결하기 위한 실질적인 솔루션을 제시합니다.

- **Technical Details**: PIR-RAG는 코스 그레인(Coarse-grained) 의미 클러스터링을 활용하여 검색 공간을 축소하고, 빠른 격자 기반( lattice-based) Private Information Retrieval (PIR) 프로토콜을 결합합니다. 이 데이터 구조를 통해 전체 내용 검색을 단일 고성능의 행렬-벡터 곱(matrix-vector product)으로 변환함으로써, 개인 정보 검색을 위한 비용을 효율적으로 관리합니다. 이를 통해 데이터 검색 및 취득 단계의 비용을 통합하여 운영 효율을 극대화합니다.

- **Performance Highlights**: PIR-RAG는 기존 아키텍처들과 비교하여 우수한 성능을 보입니다. 특히, "RAG-Ready Latency" 즉, LLM의 콘텐츠 보안을 유지하면서 데이터를 안전하게 가져오기 위해 필요한 종단 간의 시간을 명확히 보여줍니다. 종합적인 비교 평가 결과, PIR-RAG는 확장성과 성능, 검색 품질에서 매우 효율적인 솔루션으로 입증되었습니다.



### From Search to Reasoning: A Five-Level RAG Capability Framework for Enterprise Data (https://arxiv.org/abs/2509.21324)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템의 새로운 분류 프레임워크(L1-L5)를 제안하여 다양한 데이터 형식과 질문 복잡성에 따라 질문-응답 시스템을 카테고리화합니다. L1은 비구조적 데이터의 표면적 지식을 사용하며, L5는 일반 지능으로 자리잡으려는 aspirational한 수준을 다룹니다. 이를 통해 엔터프라이즈 사용자들이 기대하는 질문-응답 문제를 보다 효과적으로 해결하고자 합니다.

- **Technical Details**: 논문은 RAG 시스템이 대량의 비즈니스 데이터를 처리하는 데 있어서 발생하는 할루시네이션(hallucination) 문제를 해결하기 위한 여러 방법을 설명하고 있습니다. 특히, 이 연구는 RAG 시스템의 적절한 성능을 확보하기 위해 다양한 차원의 데이터 관련성을 고려한 개선된 검색 메커니즘의 필요성을 강조합니다. 각 레벨(L1부터 L4까지)의 시스템은 다루는 데이터 유형과 질문의 복잡성에 따라 다른 기능을 제공합니다.

- **Performance Highlights**: 제안한 프레임워크는 LangChain, Azure AI Search, OpenAI, Corvic AI와 같은 최신 플랫폼에 대한 평가에 기반하여 L1-L4 기능을 활성화하는 데 있어 다중 공간 검색 및 동적 오케스트레이션의 중요성을 강조합니다. 연구 결과는 복잡한 데이터 환경에서도 RAG 시스템이 갖는 유용성과 비즈니스 결정에 미치는 영향을 검증하는데 도움이 됩니다. 전반적으로 이 논문은 LLMs와 RAG의 잠재력을 최대한 활용하기 위한 혁신적인 전략에 대한 통찰력을 제공합니다.



### SPELUNKER: Item Similarity Search Using Large Language Models and Custom K-Nearest Neighbors (https://arxiv.org/abs/2509.21323)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)과 맞춤형 K-최근접 이웃(KNN) 알고리즘을 결합한 직관적인 아이템 유사도 검색을 위한 하이브리드 시스템을 제안합니다. 이 시스템은 자연어 질의를 구조화된 속성 기반 검색으로 변환하여 검색의 해석 가능성을 높입니다. 이는 데이터 유형의 차이를 preserving하는 이종 거리 메트릭을 사용하는 BallTree 검색을 기반으로 한 KNN으로 이어집니다.

- **Technical Details**: 제안된 시스템은 쿼리 입력을 위한 사용자 인터페이스, 자연어 처리를 위한 대규모 언어 모델, 유사도 검색을 위한 맞춤형 K-최근접 이웃 알고리즘의 세 가지 주요 구성요소로 이루어져 있습니다. 다양한 데이터 유형에 따라 전처리를 수행하고, 조건부 거리 메트릭 및 Ball Tree 전략을 통해 다차원 특성 공간에서 효율적 근접 이웃 검색을 구현하였습니다. 시스템은 Streamlit을 활용하여 사용자 친화적인 인터페이스를 제공하며, 자연어 질의를 JSON 구조로 변환하여 처리합니다.

- **Performance Highlights**: 효과성 평가에서는 500개의 와인 리뷰 데이터셋을 사용하였고, LLM은 0.9779의 F1-score를 달성하며 정보를 정확히 추출하였습니다. KNN 알고리즘과 LLM의 조합을 통한 검색 시 재호출이 통계적으로 유의미한 개선을 보였으며(p=0.013), 전체 시스템의 평균 대기 시간은 18.24초로 측정되었습니다. 이로써 LLM 기반의 리랭크가 사용자 의도에 적합한 아이템을 더 잘 식별하고 홍보하는 능력을 입증하였습니다.



### Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation (https://arxiv.org/abs/2509.22565)
- **What's New**: 이번 연구에서는 EHR(전자 건강 기록) 포털을 통한 비동기 환자-임상 의사 메시징의 증가로 인해 임상 의사의 업무 부담이 커지고 있음을 언급하며, 대형 언어 모델(LLMs)을 활용한 초안 응답 보조의 필요성을 강조합니다. 연구는 임상적으로 기반한 오류 온톨로지를 도입하고, 검색 보강 평가 파이프라인(RAEC)을 개발하며, DSPy를 사용한 두 단계의 프롬프트 구조를 제공하는 세 가지 주요 기여로 구성됩니다.

- **Technical Details**: 연구에서 소개된 오류 온톨로지는 5개의 도메인과 59개의 세부 오류 코드로 구성되어 있으며, 이는 귀납적 코딩과 전문가의 판단을 통해 개발되었습니다. RAEC는 의미적으로 유사한 과거 메시지-응답 쌍을 활용하여 평가 품질을 향상시키는 방식으로 작동하며, 두 단계의 DSPy 파이프라인을 사용하여 질적인 검토를 수행합니다.

- **Performance Highlights**: 1,500개 이상의 환자 메시지를 대상으로 한 평가에서, 검색된 맥락이 임상적 완전성과 업무 적합성 등의 영역에서 오류 식별을 개선하는 데 기여한 것으로 나타났습니다. 100개의 메시지에 대한 인간 검증 결과, 맥락이 향상된 레이블의 성능이 기준선과 비교하여 더 높은 일치도(50% vs. 33%)와 성능(F1 = 0.500 vs. 0.256)을 보이며 RAEC 파이프라인의 유용성을 지지하고 있습니다.



### Ontological foundations for contrastive explanatory narration of robot plans (https://arxiv.org/abs/2509.22493)
Comments:
          This version was submitted to the journal Information Sciences and is under review since October 2024

- **What's New**: 이번 연구는 인공지능 로봇이 다른 계획(plan) 간의 차이점을 모델링하고 설명할 수 있는 새로운 방법론을 제안합니다. 특히, 두 개의 경쟁 계획을 비교하여 각각의 특성과 인간의 선호에 가장 적합한 계획을 식별하는 데 중점을 둡니다. 본 논문에서는 이러한 비교를 통해 획득한 지식을 바탕으로 로봇이 어떻게 구체적으로 설명(narrate)할 수 있는지를 다루고 있습니다.

- **Technical Details**: 연구진은 새로운 온톨로지 모델(ontological model)을 제안하여 경쟁 계획 간의 차이를 정식화하고 이의 분석을 통해 결정론적인 관계를 도출합니다. 장소적 지식(divergent knowledge)을 이용하여 무작위 정보 획득을 촉진하는 ACXON(Algorithm for Contrastive eXplanatory Ontology-based Narratives) 알고리즘을 소개하였습니다. 이 알고리즘은 최종적으로 텍스트 기반의 대조 서사(narrative)를 생성하며, 사용자 요구에 따라 다양한 세부사항으로 결과를 조절할 수 있습니다.

- **Performance Highlights**: 제안된 ACXON 알고리즘은 기존의 바닥선(baseline) 알고리즘과 비교하여 향상된 성능을 보여주었습니다. 객관적인 평가 지표에 따르면, 더 적은 지식을 사용하여 내러티브(narratives)를 구성함으로써 의사소통 시간을 단축시킬 수 있음을 입증했습니다. 이는 로봇이 복잡한 결정 상황에서도 신뢰할 수 있는 설명을 제공하는 데 기여할 수 있음을 시사합니다.



### Chronic Stress, Immune Suppression, and Cancer Occurrence: Unveiling the Connection using Survey Data and Predictive Models (https://arxiv.org/abs/2509.22275)
- **What's New**: 이번 연구에서는 기계 학습(ML)과 인과 모델링을 활용하여 만성 스트레스와 암 발생 간의 복잡한 인과 관계를 탐구하였습니다. 연구팀은 설문조사에서 수집된 스트레스 지표, 암 이력, 인구 통계학적 데이터를 기반으로 예측 모델을 개발하였으며, 이 모형은 전통적인 통계 방법으로도 검증되었습니다. 특히, 모델은 만성 스트레스가 암 발생에 미치는 직접적인 연결고리와 면역 억제에 의한 연결고리를 밝혀냈습니다.

- **Technical Details**: 연구 방법으로는 세 가지 상호 보완적인 접근 방법을 사용했습니다. 첫 번째로, 인구 통계학적 및 경제적 정보를 포함한 설문조사를 실시하여 데이터를 수집했습니다. 두 번째로, 수집된 데이터를 이용해 설명 통계 및 선형 회귀 모델, 비선형 머신러닝 기반 회귀 모델, 인과 모델링을 통해 분석했습니다. 이러한 방법들은 만성 스트레스와 암 발생 사이의 관계를 정량적으로 평가하는 데에 초점을 맞추고 있습니다.

- **Performance Highlights**: 연구 결과, 스트레스 빈도와 스트레스 수준, 인지된 건강 영향 간의 유의미한 인과적 상관관계를 발견했습니다. 만성 스트레스 단독으로는 예측력이 제한적이었으나, 사회적, 인구통계학적 및 가족의 암 이력 데이터를 결합함으로써 모델의 정확도가 크게 향상되었습니다. 이러한 발견은 만성 스트레스가 조절 가능한 암 위험 요소로서 중요하다는 것을 강조하며, 개인 맞춤형 예방 전략과 공공 건강 개입의 필요성을 뒷받침합니다.



### The system of processing and analysis of customer tracking data for customer journey research on the base of RFID technology (https://arxiv.org/abs/2509.22162)
Comments:
          20 pages, in Russian language, 5 figures

- **What's New**: 이 논문은 RFID 기술을 기반으로 한 추적 데이터 처리 및 분석 시스템을 연구하여 소매업체에서 고객 여정을 이해하는 것을 목표로 합니다. 특히 물류를 넘어 재고 관리, 손실 예방 및 고객 경험 향상에 대한 현대 소매 응용 프로그램을 다룹니다. 또한 원시 RFID 및 POS 데이터를 구조화된 분석 데이터 웨어하우스로 변환하는 ETL(extract, transform, load) 방법론에 대한 아키텍처 종합을 제시합니다.

- **Technical Details**: 본 논문은 데이터 수집, 처리 및 통합 아키텍처에 집중하고 있으며, 특히 고객 행동 패턴과 재무 매출 지표를 결합한 포괄적인 분석을 위해 설계된 논리 데이터베이스 모델을 제안합니다. 연구의 핵심은 RFID 데이터와 POS 데이터를 연계하여 수집한 후 분석 가능한 정보로 바꾸는 것입니다. 이 과정은 retail 환경 내에서 정밀한 데이터 기반의 과학으로 변혁하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 또한 이 논문은 균형 성과표(Balanced Scorecard, BSC)의 관점에서 RFID 구현을 통한 기대되는 비즈니스 이점을 분석합니다. BSC는 재무 성과, 고객 만족도 및 내부 프로세스 최적화를 평가하는 도구로 사용됩니다. RFID와 거래 데이터의 통합은 물리적 제품 흐름과 소비자 행동에 대한 전례 없는 가시성을 제공하며, 이는 소매업 혁신의 기초가 됩니다.



### Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions (https://arxiv.org/abs/2509.22150)
- **What's New**: 본 연구에서는 비독립적이고 동일하게 분포된(Non-IID) 3D 포인트 클라우드 데이터에 적합한 새로운 분류 전략인 Joint Graph Entropy Knowledge Distillation(JGEKD)를 제안합니다. JGEKD는 클래스 간 상관관계를 유지하는 지식 증류(knowledge distillation)를 통해 클래스 간의 관계를 정확히 설명할 수 있도록 설계되었습니다. 이를 통해 모델의 일반화 능력을 강화하고, 특히 공간 변환에 불변인 포인트 클라우드 데이터를 처리할 수 있습니다.

- **Technical Details**: Joint Graph Entropy Knowledge Distillation(JGEKD) 전략은 클래스 간의 잠재적 관계를 설명하기 위해 공동 그래프(joint graph)를 구축하고, 그래프 엔트로피(graph entropy)를 기반으로 클래스를 특성화합니다. 이러한 그래프를 통해 다양한 변환 형태 간에 정보 전이를 용이하게 하는 프레임워크인 JGEsKD 및 JGEtKD를 개발하였습니다. 또한, 손상된 포인트 클라우드 데이터를 처리하기 위한 적대적 훈련(adversarial training) 전략도 제안하였습니다.

- **Performance Highlights**: ScanObject, ModelNet40, ScanntV2_cls, ModelNet-C와 같은 데이터셋에서 수행된 광범위한 실험을 통해 제안된 JGEKD 전략이 경쟁력 있는 결과를 달성할 수 있음을 입증하였습니다. 제안된 프레임워크는 모델의 강건성을 높이고, 데이터의 다양한 변환 형태에 대해 보다 나은 일반화 성능을 제공합니다. 이러한 결과는 비독립적 및 동등하게 분포된 데이터가 현실 세계 문제를 해결하는 데 있어 얼마나 중요한지를 강조합니다.



### FoodSEM: Large Language Model Specialized in Food Named-Entity Linking (https://arxiv.org/abs/2509.22125)
Comments:
          To appear in the Proceedings of the 28th International Conference on Discovery Science (DS 2025)

- **What's New**: 이 논문은 FoodSEM을 소개합니다. FoodSEM은 음식 관련 온톨로지에 대한 이름-개체 연결(named-entity linking, NEL)을 위한 최첨단 오픈소스 대형 언어 모델(large language model, LLM)로, 현재의 일반 목적 언어 모델이나 맞춤형 도메인 모델로는 해결할 수 없는 문제를 효과적으로 해결합니다. 이 모델은 텍스트에서 언급된 음식 관련 개체를 FoodOn, SNOMED-CT, Hansard 분류법 등 여러 온톨로지에 연결합니다.

- **Technical Details**: FoodSEM은 명령-응답(instruction-response, IR) 시나리오를 통해 다양한 온톨로지와 연결된 식품 개체 언급(단일 또는 다단어 시퀀스)을 링크합니다. 이 모델은 풍부한 음식 주제 문헌에서 학습하여 온톨로지와의 연결을 개선합니다. FoodSEM은 고공헌능력(F1 score)에서 일부 온톨로지 및 데이터셋에 대해 98%에 달하는 성과를 달성했습니다.

- **Performance Highlights**: FoodSEM은 0-shot, 1-shot, 5-shot 프롬프트에 대한 비교 분석을 통해 비미세 조정(non-fine-tuned)된 버전보다 뛰어난 성능을 입증했습니다. 이 모델은 GitHub와 Hugging Face에서 공개되어 연구자들이 쉽게 접근할 수 있습니다. FoodSEM의 출현은 음식 관련 NEL에 대한 강력한 기준선을 제공하여 향후 벤치마킹 연구에 기여할 예정입니다.



New uploads on arXiv(cs.CV)

### RefAM: Attention Magnets for Zero-Shot Referral Segmentation (https://arxiv.org/abs/2509.22650)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 기존의 세분화 작업을 위한 방법론을 발전시켜, 특수한 아키텍처 수정이나 추가 학습없이 diffusion transformers에서 얻은 attention score를 직접 활용하는 새로운 방법을 제안합니다. 이 과정에서 stop words가 attention을 재분배하는 역할을 수행하며, 전반적으로 더 깨끗하고 국소화된 heatmap을 생성할 수 있음을 보여줍니다. 특히, RefAM이라는 training-free 세분화 프레임워크를 통해 기존 방법들보다 우수한 성능을 달성했습니다.

- **Technical Details**: Diffusion transformers(DiTs)는 효율적인 feature 추출기로서, 이미지와 비디오의 vision-language grounding 작업에 강력한 도구가 됩니다. 본 연구에서는 attention sink의 논리를 통해 stop words가 높은 attention을 흡수하며, 이를 통해 attention을 재분배하는 전략을 도입합니다. 또한, attention maps을 필터링하고 cross-attention을 집계하여 grounding 작업을 더 효과적으로 수행할 수 있음을 입증합니다.

- **Performance Highlights**: 본 연구의 접근 방법은 zero-shot setting에서 이미지와 비디오 세분화 벤치마크를 푸는 데 있어 기존의 training-free 접근법들보다 항상 더 나은 성능을 보였습니다. 이로 인해, 세분화 작업에서 새로운 최첨단 기술을 확립하였고, 추가적인 컴포넌트나 fine-tuning 없이도 우수한 결과를 얻을 수 있었습니다.



### CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/abs/2509.22647)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문은 이미지 캡셔닝 과제를 위한 새로운 Reinforcement Learning with Verifiable Rewards (RLVR) 패러다임을 제안합니다. 기존의 Supervised Fine-Tuning (SFT) 모델들이 특정 Ground-Truth 답변을 기억하는 것에 한계를 두고, 일반성 부족 문제를 해결할 수 있는 방법을 갖추고 있습니다. 새로운 Captioning Reinforcement Learning (CapRL) 프레임워크는 이미지에 대해 질문을 정확히 답할 수 있는 캡션 생성을 평가합니다.

- **Technical Details**: CapRL은 LVLM이 생성한 캡션과 비슷한 정보에 대한 Multiple-Choice Questions (MCQs)에 기반하여 객관적인 보상을 제공하는 이중 파이프라인을 사용합니다. 세부적으로, LVLM이 캡션을 생성하고, 비전이 없는 LLM이 그 캡션을 통해 질문에 답하는 정확성을 통해 보상을 결정합니다. 이 접근법은 주관적인 이미지 캡셔닝 문제에서 캡션의 품질을 유용성(utility)으로 정의합니다.

- **Performance Highlights**: CapRL을 적용한 결과, 12개의 벤치마크에서 상당한 성능 향상을 보여 주며, CapRL-3B가 72B 모델에 필적하는 성과를 달성했습니다. Prism Framework를 통한 평가에서 CapRL 모델이 기본선보다 평균 8.4% 향상된 성능을 나타냈습니다. 이러한 결과는 CapRL이 Dense하고 Accurate한 캡션을 생성하도록 모델을 효과적으로 유도함을 보여줍니다.



### Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs (https://arxiv.org/abs/2509.22646)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 AI 생성 비디오에서 확인할 수 있는 인간 지각의 '딥페이크 흔적'을 탐구하는 DeeptraceReward라는 새로운 벤치마크를 소개합니다. 이 데이터셋은 4,300개의 세부 주석을 포함하며, 각 주석은 자연어 설명, 경계 박스(bounding box) 위치 및 정확한 시간 정보를 제공합니다. 이를 통해 인간이 어떻게 AI 생성 비디오를 식별할 수 있는지와 그 이유를 조사하는 것을 목표로 하고 있습니다.

- **Technical Details**: DeeptraceReward 데이터셋은 3,300개의 고품질 생성 비디오와 3,300개의 실제 비디오로 구성됩니다. 저자들은 비디오에 존재하는 다양한 딥페이크 흔적을 아홉 가지 범주로 분류했으며, 이를 통해 멀티모달 언어 모델을 훈련시켜 인간의 판단을 모방할 수 있도록 하였습니다. 연구 결과, 7B의 보상 모델이 GPT-5를 34.7% 초과하여 인식 및 설명 과제에서 우수한 성능을 보였다는 것이 흥미롭습니다.

- **Performance Highlights**: 모델의 성능 평가에서 이진 분류(진짜 vs 가짜 비디오) 작업은 99.4%에 달하지만, 보다 세밀한 딥페이크 흔적 탐지 성능은 70.2%에 불과함을 보였습니다. 저자들은 자연어 설명 및 공간적 지각은 상대적으로 쉬운 반면, 시간 라벨링 작업은 가장 어려운 과제임을 확인했습니다. 이러한 결과는 영상 생성의 사회적 신뢰성을 향상시키기 위한 중요한 지침을 제공합니다.



### Hierarchical Representation Matching for CLIP-based Class-Incremental Learning (https://arxiv.org/abs/2509.22645)
- **What's New**: 본 논문에서는 HiErarchical Representation MAtchiNg (HERMAN)을 제안하여 CLIP 기반의 클래스 증가 학습(Class-Incremental Learning, CIL) 문제를 해결합니다. HERMAN은 LLMs(대형 언어 모델)를 활용하여 계층적인 텍스트 설명을 생성하고, 이를 시각적 표현과 매칭하는 과정을 통해 새로운 범주에 대한 적응을 향상시킵니다. 새로운 접근방식인 HERMAN은 기존의 단순한 템플릿을 넘어서 정보의 계층적 구조를 효과적으로 활용하도록 설계되어 있습니다.

- **Technical Details**: HERMAN 프레임워크는 먼저 LLMs를 통해 여러 계층에서 구별되는 텍스트 설명을 생성합니다. 그런 다음 이러한 설명은 해당하는 시각적 특징과 매칭되어, 형상과 세부 정보를 모두 포함하는 구조화된 의미 공간을 생성합니다. 이후 적응형 라우팅 메커니즘을 통해 의미 계층 전반에 걸쳐 가중치를 동적으로 할당하며, 각 입력에 대해 가장 관련성 높은 특징을 강조할 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험 결과, HERMAN은 기존의 최첨단 성능을 지속적으로 초과하는 능력을 보여주었습니다. 특히 계층적 의미를 통한 정밀한 구별을 가능하게 하여, 재학습 과정에서의 재앙적 망각(catasrophic forgetting)을 완화하는 데 기여하고 있습니다. 이러한 성과는 CLIP 모델의 강력한 성능 향상에 기여함은 물론, 실세계 incremental learning 문제를 해결하는 데 중요한 통찰을 제공합니다.



### Scale-Wise VAR is Secretly Discrete Diffusion (https://arxiv.org/abs/2509.22636)
Comments:
          Technical Reports

- **What's New**: 이번 연구는 Visual Autoregressive Generation (VAR)을 재조명하여, Markovian attention mask를 갖출 때 VAR가 수학적으로 이산 확산 모델(discrete diffusion model)과 동등하다는 이론적 통찰을 제시합니다. 우리는 이 재해석을 스케일러블 비주얼 리파인먼트(Scalable Visual Refinement with Discrete Diffusion, SRDD)라고 명명하여 AR transformer와 확산 모델 사이의 원칙적 다리를 구축합니다. 새로운 관점을 활용하여 VAR의 구조적 비효율성을 줄이고, 더 빠른 수렴, 낮은 추론 비용 및 개선된 제로 샷 재구성을 보여줍니다.

- **Technical Details**: 비주얼 오토회귀 생성(VAR)란, 다음 스케일을 예측하는 새로운 패러다임으로, 모든 토큰을 한 번에 생성한 후 점진적으로 고해상도 이미지로 이동하는 방법입니다. VAR 모델은 Markovian 방식으로 다음 스케일을 예측하여 디자인 비효율성을 개선할 수 있는 가능성을 확인했습니다. 본 논문에서는 SRDD를 통해 Markovian VAR의 이론적 해석을 제시하며, 이를 통해 확산 특성을 통한 확률적 샘플링 기법을 실험했습니다.

- **Performance Highlights**: 우리는 SRDD가 기존 VAR 아키텍처보다 슈퍼 해상도, 인페인팅, 아웃페인팅 등의 제로샷 생성 성능에서도 더 나은 결과를 얻는 것을 보여주었습니다. 또한, SRDD는 모든 관련 문헌을 VAR 공식화에 활용함으로써, 명시적인 수작업 디자인 선택 없이도 비주얼 생성의 생성 품질을 획기적으로 개선할 수 있음을 입증했습니다. 이 연구는 비주얼 생성의 질, 효율성 및 설명 가능성을 한 단계 발전시키는 기회를 제공합니다.



### Training-Free Synthetic Data Generation with Dual IP-Adapter Guidanc (https://arxiv.org/abs/2509.22635)
Comments:
          BMVC 2025. Project page: this https URL

- **What's New**: DIPSY는 모델 파인튜닝(training-free) 없이 적은 수의 라벨링된 예시를 활용하여 뛰어난 구별력을 가진 합성 이미지를 생성하는 새로운 접근 방식을 제시합니다. 이 방법은 양성 및 음성 이미지 조건을 독립적으로 조절할 수 있는 확장된 classifier-free guidance(구분자 없는 안내)를 도입하며, 효과적인 대조 예시를 식별하는 클래스 유사성 기반 샘플링 전략을 통합합니다. DIPSY는 모델 조정이나 외부 도구에 의존할 필요 없이 합성 데이터를 생성하여, 그 결과가 기존의 방법들보다 더 뛰어남을 입증합니다.

- **Technical Details**: DIPSY는 IP-Adapter를 활용한 이미지-투-이미지 변환의 방식을 사용하여, 새로운 합성 이미지를 생성합니다. 이 방법은 클래스 간 특징 구별을 극대화하기 위해 양성 및 음성 안내의 강도를 조절하며, 효과적인 이미지 프롬프트 선택을 위해 클래스 유사성 기반의 샘플링 전략을 도입합니다. DIPSY는 특히 지식 자원이 제한된 실제 환경에서 적용 가능성이 높은 효율적인 솔루션입니다.

- **Performance Highlights**: DIPSY는 10개의 비교 과제에서 최신의 방법들과 동등하거나 더 나은 성능을 달성했습니다. 정밀한 특징을 포착하는 데 있어 특히 뛰어난 성과를 보여, 정밀한 분류 작업에서 높은 구별력을 유지합니다. 설치 및 구성의 용이성 덕분에 DIPSY는 현실적인 자원 제약이 있는 상황에서도 우수한 성능을 유지합니다.



### LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision (https://arxiv.org/abs/2509.22631)
- **What's New**: Labeling Copilot는 컴퓨터 비전을 위한 첫 번째 데이터 큐레이션(curation) 딥 리서치 에이전트입니다. 이 에이전트는 대규모 다중 모달 언어 모델에 의해 구동되며, 데이터 품질, 다양성 및 비용 간의 복잡한 균형을 수행합니다. Labeling Copilot은 데이터 발견(discovery), 새로운 데이터 생성(synthesis) 및 합의 기반 주석(annotation)이라는 세 가지 핵심 기능을 통해 데이터 큐레이션 프로세스를 최적화합니다.

- **Technical Details**: Labeling Copilot의 핵심 기능은 (1) Calibrated Discovery를 통해 대규모 저장소에서 관련 데이터를 검색하고, (2) Controllable Synthesis로 드문 시나리오를 위한 새로운 데이터를 생성하며, (3) Consensus Annotation으로 여러 기초 모델을 조정하여 정확한 레이블을 생성하는 것입니다. 이 에이전트는 또한 고급 필터링 및 여러 도구의 조정을 통해 데이터 품질을 향상시키는 메커니즘을 통합하고 있습니다.

- **Performance Highlights**: Labeling Copilot의 구성 요소는 대규모 검증을 통해 그 효과가 입증되었습니다. 예를 들어, Consensus Annotation 모듈은 COCO 데이터셋에서 평균 14.2개의 후보 제안을 제공하며, Open Images 데이터셋에서는 총 1500개 이상의 새로운 바운딩 박스 카테고리를 발견했습니다. Calibrated Discovery 도구는 10백만 샘플 규모에서 최대 40배 더 효율적인 계산 성능을 보여줍니다.



### UML-CoT: Structured Reasoning and Planning with Unified Modeling Language for Robotic Room Cleaning (https://arxiv.org/abs/2509.22628)
- **What's New**: 이번 연구에서는 Chain-of-Thought (CoT) 프롬프트를 통해 대형 언어 모델(LLM)의 추론 능력을 개선하지만 비구조적 텍스트에 의존하는 한계가 있는 문제를 해결하기 위해 UML-CoT를 제안합니다. UML-CoT는 Unified Modeling Language (UML)를 활용하여 상징적인 CoT와 실행 가능한 행동 계획을 생성하는 구조적 사고 및 계획 프레임워크입니다. 이를 통해 UML 클래스 다이어그램은 조합 개체 의미를 포착하고, 활동 다이어그램은 절차적 제어 흐름을 모델링합니다.

- **Technical Details**: UML-CoT는 세 단계의 훈련 파이프라인을 통해 구성됩니다: (1) UML의 주석이 있는 추론 및 계획 흔적에 대한 감독된 미세 조정(Supervised Fine-Tuning, SFT); (2) 최종 계획의 정확성에 따라 보상을 받는 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 기반으로 하는 강화 학습 미세 조정(Reinforcement Learning Fine-Tuning, RLFT); (3) 중간 추론 주석 없이도 효과적 학습을 가능하게 하는 정답만 있는 데이터에 대한 추가 GRPO 훈련입니다. 이 방법은 MRoom-30k 벤치마크에서 평가되며, 구조적 CoT가 비구조적 CoT보다 해석 가능성 및 실행 성공률에서 우수함을 보여줍니다.

- **Performance Highlights**: 우리의 실험은 UML-CoT가 계획 일관성, 실행 성공률, 구조적 충실도에서 현저한 개선을 이루었다는 것을 입증했습니다. 네 가지 구성(단순 텍스트 추론에서 완전 UML 기반 파이프라인까지)의 비교에서 구조적 접근 방식이 효과적임을 확인했습니다. 또한, 세 번째 단계에서의 RLFT가 성능을 향상시킨다는 사실은 단계화된 강화 학습이 효과적이라는 것을 나타냅니다.



### CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach (https://arxiv.org/abs/2509.22627)
- **What's New**: 이번 연구에서는 깊이 추정을 위한 새로운 자기 지도(convolutional) 접근법인 CCNeXt를 제안하며, 기존의 최첨단 Convolutional Neural Networks (CNNs) 및 Vision Transformers (ViTs)를 초월하는 성능을 보여줍니다. CCNeXt 아키텍처는 현대적인 CNN 기능 추출기를 사용하고, 심층 사전 훈련된 에피폴라 크로스 어텐션 모듈을 도입하여 구성되어 있습니다. 또한, 모델의 실행 시간을 감소시키기 위해 깊이 추정 디코더를 포괄적으로 재설계하였습니다.

- **Technical Details**: CCNeXt는 ConvNeXt를 기반으로 하여, 왼쪽과 오른쪽 이미지 쌍 간의 특징 표현을 원활하게 흐르게 하는 새로운 창(windowed) 에피폴라 크로스 어텐션 메커니즘을 결합합니다. 이 모델은 깊이 정보의 질을 더욱 향상시키기 위한 Individual Contextual Expansive Path (ICEP)를 제안하여, 모듈의 실행 시간을 줄이고 더 높은 차원 출력을 보장합니다. 이러한 변경 사항을 통해, CCNeXt는 KITTI 데이터셋에서 기존 모델보다 10.18배 더 빠르게 동작합니다.

- **Performance Highlights**: CCNeXt는 KITTI 데이터셋에서 절대 상대 오차(AbsRel), 제곱 상대 오차(SqRel), 정확도와 같은 주요 깊이 메트릭에서 최고의 성능을 달성했습니다. 최근 제안된 방법과 비교할 때 모든 메트릭에서 최첨단 결과를 기록하였으며, 실행 시간을 획기적으로 단축시키면서도 우수한 정확도를 유지하고 있습니다. 뿐만 아니라, DrivingStereo 데이터셋에서도 모든 깊이 메트릭에서 뛰어난 성능을 보여주어, 기존 연구들 대비 상당한 이점을 제공합니다.



### SPARK: Synergistic Policy And Reward Co-Evolving Framework (https://arxiv.org/abs/2509.22624)
Comments:
          Project:this https URL

- **What's New**: 최근 대형 언어 모델(LLMs) 및 대형 비전-언어 모델(LVLMs)에서는 후속 학습(Post-Pretraining)에서 강화 학습(Reinforcement Learning, RL)을 사용하고 있습니다. 특히, 검증 가능한 보상(RLVR)과 인간 피드백에서의 RL(RLHF) 기반 접근법이 효과를 보이고 있으나, RLHF의 경우 높은 비용과 인간의 선호도에 대한 의존성 문제가 발생합니다. 이 문제를 해결하기 위해, 우리는 SPARK라는 새로운 방법론을 제안하며, 이는 이전에 무시됐던 롤아웃과 정확도 데이터를 재활용하여 모델 자체를 동시에 학습시키는 방식을 채택합니다.

- **Technical Details**: SPARK는 강화 학습 기반의 정책 및 보상 동시 발전 프레임워크로, RLVR을 기반으로 합니다. 기존의 롤아웃을 폐기하지 않고, 그 데이터를 활용해 모델이 스스로 보상을 생성하도록 교육합니다. 이 과정은 점수화된 보상, 쌍 비교, 반영 응답을 기반으로 한 평가 등 다양한 목표를 조합하여 진행되며, 최종적으로는 외부 보상 모델 의존성을 줄이고 GPU 메모리를 절약하여 훈련과 서비스 효율성을 개선합니다.

- **Performance Highlights**: SPARK는 다양한 LLM 및 LVLM 모델에서 상당한 성능 향상을 보여주었습니다. 예를 들어, SPARK-VL-7B는 7개의 추론 벤치마크에서 평균 9.7%, 2개의 보상 벤치마크에서 12.1%, 그리고 8개의 일반 벤치마크에서 1.5%의 성과 개선을 기록했습니다. 이러한 결과는 다양한 모델 스케일과 구조에서 강건성을 입증하며, SPARK는 자기반영(Self-Reflection)을 통한 테스트 시점 확장을 가능하게 합니다.



### LongLive: Real-time Interactive Long Video Generation (https://arxiv.org/abs/2509.22622)
Comments:
          Code, model, and demos are available at this https URL

- **What's New**: LongLive는 실시간 상호작용 긴 비디오 생성을 위한 프레임 수준의 자회귀(AR) 프레임워크입니다. 기존의 확산 모델들은 높은 품질을 제공하지만 효율성에서 부족함을 보이며, 인과적 주의( causal attention) 모델은 메모리 문제로 긴 비디오에서 품질이 떨어지는 경향이 있습니다. LongLive는 새로운 KV-recache 메커니즘을 채택하여 사용자가 실시간으로 비디오 생성을 제어할 수 있도록 했습니다.

- **Technical Details**: LongLive는 프레임 수준의 자회귀 디자인을 채택하여 효율적인 추론을 가능하게 합니다. 주요 설계로는 새로운 프롬프트에 대해 캐시된 상태를 새로 고치는 KV-recache 메커니즘, 긴 비디오 훈련과 추론을 정렬하는 스트리밍 긴 조정 전략이 포함됩니다. 또한, 짧은 윈도우 주의와 프레임 레벨 주의 싱크를 결합하여 추론 속도를 증가시킬 수 있도록 했습니다.

- **Performance Highlights**: LongLive는 1.3B 파라미터 모델을 32 GPU-days만에 분당 고품질의 긴 비디오 생성으로 미세 조정합니다. 단일 NVIDIA H100에서 20.7 FPS의 성능을 유지하며, 짧은 비디오와 긴 비디오 모두에서 뛰어난 VBench 성능을 보여줍니다. LongLive는 단일 H100 GPU에서 최대 240초의 비디오를 생성할 수 있으며, INT8 양자화 추론을 통해 최소한의 품질 손실만을 가지고 지원합니다.



### Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting (https://arxiv.org/abs/2509.22615)
- **What's New**: 이 논문에서는 2D Gaussian Splatting (2DGS)을 새로운 비주얼 표현으로 탐구하여 멀티모달 시스템에서 비전-언어 정렬을 위한 효율적이고 효과적인 중간 표현으로 사용할 수 있는지에 대한 논의를 담고 있습니다. 기존 RGB 기반 비전 인코더의 데이터 전송 및 처리 효율성 문제를 해결하기 위해, 2DGS는 색이 지정된 비대칭 Gaussian의 집합으로 이미지를 매개변수화하여 더 간결하고 공간적으로 적응 가능한 형태로 정보를 전달합니다. 이 연구는 대규모 범위에서 2DGS를 구현하고 CLIP 프레임워크에 적응시키는 방법을 제안하며, 빠른 피팅과 GPU 유틸리제를 달성했습니다.

- **Technical Details**: 논문에서 제안한 시스템 및 알고리즘 최적화는 구조화된 초기화, 밝기 인식 L1 프루닝 및 배치 CUDA 커널을 포함하여 기존의 구현보다 90배 이상 빠른 피팅을 달성하도록 설계되었습니다. 2DGS 표현은 효율적인 데이터 전송을 가능하게 하고, RGB 기반의 변환기 아키텍처의 냉동된 형태를 재사용하여 가벼운 입력 전처리 유닛과 함께 CLIP 프레임워크에 효과적으로 통합됩니다. 이 과정에서 약 7%의 매개변수로 훈련을 수행하면서도 우수한 전이 학습 성과를 나타냅니다.

- **Performance Highlights**: 실험 결과 12.8M 규모의 DataComp 데이터셋에서 GS 인코더는 이미지넷-1K의 제로샷 성능을 의미 있게 달성하면서 픽셀 대비 입력 압축은 3배에서 20배 이르는 결과를 보여주었습니다. 현재 2DGS 기반 인코더는 RGB 기반 모델에 비해 정확성이 낮지만, 명백한 전이 가능성과 압축된 표현으로서의 유용성을 입증하였습니다. 이 연구는 2DGS가 멀티모달 시스템에서 효과적인 대안이 될 수 있으며, 이는 처리 효율성과 지속 가능성을 높이는 방향으로 나아가고자 하는 방법을 제시합니다.



### SpikeMatch: Semi-Supervised Learning with Temporal Dynamics of Spiking Neural Networks (https://arxiv.org/abs/2509.22581)
- **What's New**: 이번 논문에서는 생물학적으로 그럴듯한 특성과 에너지 효율성 덕분에 주목받고 있는 스파이킹 신경망(spiking neural networks, SNNs) 기반 반지도 학습(semi-supervised learning, SSL) 방법, SpikeMatch를 소개합니다. SpikeMatch는 공훈련(co-training) 프레임워크 내에서 SNN의 누수 계수(leakage factor)를 활용하여 다양한 의사 라벨링(pseudo-labeling)을 가능하게 합니다. 이를 통해 단일 SNN의 여러 예측 간의 일치를 기반으로 신뢰할 수 있는 의사 라벨을 생성하여, 제한된 라벨이 있는 상황에서도 차별화된 특징을 캡처할 수 있도록 합니다.

- **Technical Details**: SpikeMatch는 SNN의 시간적 동역학을 활용한 SSL 프레임워크입니다. 저자들은 Leaky Integrate-and-Fire (LIF) 뉴런의 누수 계수를 높여 예측의 다양성을 증진시키고, 이를 통해 예측의 일치를 기반으로 한 의사 라벨링을 수행합니다. 고립된 예측 대신 여러 관점에서 SNN의 시간적 출력을 훈련함으로써 극단적인 의사 라벨 선택을 예방하고 확인 편향(confirmation bias)을 완화할 수 있습니다.

- **Performance Highlights**: 실험을 통해 SpikeMatch는 다양한 표준 벤치마크에서 SNN 기반의 기존 SSL 방법들을 능가하는 성능을 보여주었습니다. 이 연구는 SNN에서 반지도 학습을 위한 강력한 해결책을 제시하며, ANNs에서 성공적으로 사용되는 여러 SSL 기술들을 접목함으로써 효과성을 입증합니다. 이러한 접근법은 시간적 정보를 효과적으로 활용하므로 점차 더 발전하고 있습니다.



### JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation (https://arxiv.org/abs/2509.22548)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문은 Vision-and-Language Navigation(VLN)에서의 새로운 접근 방식을 제시합니다. 기존의 VLN 방법들은 명시적인 semantic memory에 의존했으나, JanusVLN은 공간 기하학적(spatial-geometric) 메모리와 시각적 의미(visual-semantic) 메모리를 별도로 처리하는 이중 암묵적 신경망을 통해 이를 개선합니다. 이로 인해 기존 모형들이 겪었던 공간 정보 손실, 계산 중복 및 메모리 부풀어 오름 문제가 해결됩니다.

- **Technical Details**: JanusVLN은 3D 공간 기하학 인코더와 함께 멀티모달 대형 언어 모형(MLLM)을 확장하여 RGB 비디오 입력만으로도 공간적 추론 능력을 증가시킵니다. 이 이중 암묵적 메모리는 역사적 키-값 저장소(key-value cache)를 통해 구성되며, 초기 및 슬라이딩 윈도우를 통해 동적으로 업데이트하여 불필요한 계산을 피할 수 있습니다. 이러한 설계는 신경망의 메모리 크기를 고정 크기로 유지하며 처리 과정에서의 효율성을 높입니다.

- **Performance Highlights**: 다양한 실험을 통해 JanusVLN은 20개 이상의 최신 방법들을 초월하여 SOTA(State-of-the-Art) 성능을 달성했습니다. 예를 들어, 여러 유형의 데이터 입력을 사용하는 방법들에 비해 성공률이 10.5-35.5% 향상되었고, 더 많은 RGB 훈련 데이터를 사용하는 방법들보다 3.6-10.8% 개선되었습니다. 이는 JanusVLN이 VLN 연구의 새로운 방향성을 탐구하는 데 기여하고 있음을 나타냅니다.



### HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection (https://arxiv.org/abs/2509.22544)
- **What's New**: 이번 논문에서는 복잡한 비디오 이상 감지 문제에 대한 새로운 접근법인 HyCoVAD를 제안합니다. 이 모델은 자기 지도 학습(self-supervised learning, SSL)과 대형 언어 모델(large language models, LLM)을 통합하여 다중 에이전트 간의 복잡한 상호작용을 기반으로 하는 이상 사건을 탐지하고 검증할 수 있습니다. SSL 모듈은 nnFormer 신경망을 기반으로 하여 비디오 프레임에서 이상 가능성이 있는 프레임을 선정한 후, LLM을 통해 의미적 문맥을 추가하여 분석을 강화합니다.

- **Technical Details**: HyCoVAD는 SSL 모듈과 LLM 검증기를 융합한 하이브리드 구조를 가지고 있습니다. SSL 모듈은 다양한 프록시 작업을 통해 비디오의 정상적인 시공간 패턴을 학습하며, 이로 인해 비정상적인 동작을 탐지할 수 있습니다. 이후 선택된 프레임은 LLM으로 전달되어, 구조적 규칙 기반 추론을 통해 이상 존재 여부를 검증합니다.

- **Performance Highlights**: ComplexVAD 데이터셋에서 HyCoVAD는 72.5%의 프레임 수준 AUC를 달성하여 기존의 기법보다 12.5% 향상된 성능을 보여주었습니다. 이는 SSL의 효율성과 LLM의 의미론적 추론을 결합한 새로운 접근 방식이 효과적으로 기능하고 있음을 나타냅니다. 본 연구는 앞으로의 복잡한 VAD 시나리오 연구에 기여할 수 있는 상호작용 이상 분류 체계 및 적응형 임계값 프로토콜도 제공합니다.



### Category Discovery: An Open-World Perspectiv (https://arxiv.org/abs/2509.22542)
- **What's New**: 이번 논문에서는 Category Discovery (CD)라는 새로운 오픈 월드 학습 과제를 다루고 있습니다. CD는 이전에 보지 못한 클래스를 포함한 레이블이 없는 데이터를 자동으로 카테고리화하는 것을 목표로 하며, 이는 기존 연구 분야에서 새로운 관점을 제시합니다. 특히, Novel Category Discovery (NCD)와 Generalized Category Discovery (GCD)로 구분되는 다양한 설정을 제안하며, 현실 세계의 다양한 응용 시나리오에 적응할 수 있도록 있습니다.

- **Technical Details**: 논문에서는 CD의 기초 설정을 소개하며, 각 설정에 대한 방법 분석을 진행합니다. 여기서는 대표적으로 세 가지 요소인 representation learning, label assignment, class number estimation에 대한 접근 방식을 다룹니다. 더불어, CD의 문제를 해결하기 위한 연속적인 카테고리 발견(Continual Category Discovery), 연합 카테고리 발견(Federated Category Discovery) 등 다양한 파생 설정을 제안합니다.

- **Performance Highlights**: CD 연구는 최근 2022년 이후로 급증하였으며, 기존의 연구 성과를 기준으로 성능을 비교하였습니다. 이 과정에서 대규모 프리트레인 백본(pretrained backbone), 계층적 및 보조 신호(hierarchical and auxiliary cues), 커리큘럼 스타일의 학습이 CD에 유리하다는 통찰을 제공합니다. 그러나 label assignment 설계, class numbers 추정, 복잡한 멀티 오브젝트에 대한 확장성 등 여전히 해결해야 할 과제가 남아 있습니다.



### EfficientDepth: A Fast and Detail-Preserving Monocular Depth Estimation Mod (https://arxiv.org/abs/2509.22527)
Comments:
          12 pages, 7 figures, 5 tables

- **What's New**: 본 논문에서 소개하는 EfficientDepth는 경량의 convolutional decoder와 transformer 아키텍처를 결합한 새로운 monocular depth estimation (MDE) 시스템입니다. 이 시스템은 다양한 3D 재구성 및 뷰 합성 응용 프로그램을 위한 기하학적 일관성과 세부 사항, 그리고 실제 환경에서의 도전 과제를 잘 처리할 수 있도록 설계되었습니다. 나아가 LPIPS 기반의 손실 함수를 도입하여 깊이 맵의 세부 사항을 강조하는 접근 방식을 제공합니다.

- **Technical Details**: EfficientDepth는 입력으로 단일 이미지를 취하여 역 깊이(predicted inverse depth)를 예측하도록 설계되었습니다. 모델은 MiT-B5라는 경량 transformer 네트워크와 간단한 UNet 디코더를 사용하였습니다. 또한, 최종 레이어에서 bimodal 분포를 예상하는 방법을 통해 이미지를 통해 얻어진 두 가지 불확실한 깊이를 동시에 추정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과 EfficientDepth는 기존의 최첨단 모델과 비슷하거나 더 나은 성능을 보이며, 계산 자원을 현저히 줄였습니다. 모델은 여러 유형의 데이터 세트에서 훈련되고 테스트되었으며, 특히 실제 이미지를 잘 처리하면서도 세부 사항과 기하학적 일관성에서 뛰어난 성능을 보여주었습니다. 이러한 결과는 실제 환경에서도 강력한 적용 가능성을 잘 보여줍니다.



### Color Names in Vision-Language Models (https://arxiv.org/abs/2509.22524)
- **What's New**: 이 연구는 비전-언어 모델(VLMs)의 색상 명명 능력을 체계적으로 평가한 첫 번째 사례입니다. 957개의 색상 샘플을 사용하여 고전적인 색상 명명 방법론을 재현하였으며, 비전-언어 모델이 인간처럼 색상을 명명하는지를 분석하였습니다. 실험 결과, VLMs는 전통적인 색상에서 높은 정확도를 보였으나, 비전형(non-prototypical) 색상 집합에서는 성능이 급격히 떨어지는 것을 확인했습니다.

- **Technical Details**: 연구에는 5개의 대표적인 비전-언어 모델이 포함되었으며, 기본 색상과 체계적인 밝기 수정자를 사용한 두 가지 접근 방식을 구분했습니다. 또한, 9개 언어 간의 분석을 통해 영어와 중국어에 유리한 심각한 훈련 불균형을 드러냈습니다. 색상 명명 결정에 있어 색조(hue)가 주요한 요인으로 작용함을 강조하였습니다.

- **Performance Highlights**: 21개의 일반적인 색상 용어가 모든 모델에서 일관되게 나타났으며, 언어 모델 구조가 시각 처리 능력과는 독립적으로 색상 명명에 상당한 영향을 미친다는 사실을 발견했습니다. 이 연구는 비전-언어 모델의 색상 인식 능력을 향상시키기 위한 방향성을 제시하고 있으며, 다국적 모델의 언어적 불균형을 해결하기 위한 필요성을 강조합니다.



### Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation (https://arxiv.org/abs/2509.22496)
- **What's New**: 이번 논문에서는 EAGLE(Empowering Attributed Generation in Language)의 새로운 프레임워크를 제안합니다. 이 프레임워크는 Autoregressive Token Generation에 대한 설명을 수반하며, 시각적 모달리티와 언어 모달리티 간의 관계를 해석하는 데 도움을 줍니다. EAGLE은 시각적 증거와 언어 우선성 간의 상대적인 영향을 정량화하며, 이는 MLLMs의 해석 가능성을 개선하는 데 중요한 진전을 나타냅니다.

- **Technical Details**: EAGLE는 선택된 토큰을 조밀한 인식 영역에 귀속시키며, 그 과정을 최적화하기 위해 두 가지 점수(Insight Score와 Necessity Score)를 사용합니다. 이 프레임워크는 희소화된 이미지 영역에 대한 탐욕적 검색을 통해 최적화되며, 모달리티 인식 분석도 수행하여 생성된 각 토큰이 어떤 요소에 더 의존하는지를 파악합니다. 이를 통해 MLLMs의 결정 과정에 대해 더 세밀한 해석을 제공합니다.

- **Performance Highlights**: EAGLE은 다양한 오픈소스 MLLMs에서 다른 기존 방법들보다 20% 이상의 정확도로 우수한 성능을 보여주었습니다. 특히 우리 방법은 이미지 캡셔닝과 VQA 작업에서 탁월한 성능을 기록하며, 오브젝트 토큰에 대한 더욱 합리적인 설명을 제공합니다. 이러한 결과를 통해 EAGLE은 MLLMs의 해석 가능성을 높이고 특정 문제(예: hallucinations)의 진단 및 완화를 지원합니다.



### Group Critical-token Policy Optimization for Autoregressive Image Generation (https://arxiv.org/abs/2509.22485)
Comments:
          Code is available at this https URL

- **What's New**: 이번 논문은 RLVR(Reinforcement Learning with Verifiable Rewards) 기반의 자동회귀(autoregressive, AR) 비주얼 생성에서 중요한 토큰을 효율적으로 최적화하는 새로운 방법론인 GCPO(Group Critical-token Policy Optimization)를 제안합니다. 기존 접근 방식은 모든 이미지 토큰에 대해 균일한 최적화를 적용했으나, 이 연구에서는 서로 다른 이미지 토큰의 기여도가 다름을 탐구합니다.

- **Technical Details**: GCPO는 세 가지 관점에서 중요 토큰을 식별합니다: 1) 인과적 의존성(causal dependency)으로 인해 초기 생성된 토큰이 후속 토큰과 전체 이미지 구조에 미치는 영향을 고려합니다; 2) 엔트로피 유도된 공간 구조(entropy-induced spatial structure)로서, 높은 엔트로피 기울기를 가진 토큰이 이미지 구조에 해당하는 것을 발견합니다; 3) RLVR 기반 토큰 다양성(RLVR-focused token diversity)으로, 시각적으로 낮은 유사성을 가진 토큰이 더 풍부한 토큰 레벨의 다양성에 기여한다고 판단합니다.

- **Performance Highlights**: GCPO는 이미지 토큰의 30%만을 활용하여 높은 성능을 달성했으며, 기존 방식인 GRPO(Group Relative Policy Optimization)보다 더 나은 결과를 제시했습니다. Geneval, T2I-CompBench 및 Human Preference Benchmark 등 다양한 평가에서 GCPO의 효과성은 입증되었습니다. 이 연구의 결과는 AR 모델과 통합된 다중 모달 모델 모두에 적용 가능하다는 점도 강점입니다.



### PSTTS: A Plug-and-Play Token Selector for Efficient Event-based Spatio-temporal Representation Learning (https://arxiv.org/abs/2509.22481)
- **What's New**: 본 논문에서는 이벤트 데이터에 최적화된 Progressive Spatio-Temporal Token Selection (PSTTS) 방법을 제안합니다. PSTTS는 추가적인 매개변수를 도입하지 않고도 이벤트 데이터의 시공간 분포 특성을 활용하여 중복되는 시공간 토큰을 효과적으로 식별하고 제거함으로써 정확성과 효율성 간의 최적의 균형을 달성합니다. 이 방법은 특히 Spatial Token Purification과 Temporal Token Selection의 두 단계로 구성되어 있습니다.

- **Technical Details**: PSTTS의 첫 번째 단계인 Spatial Token Purification은 각 이벤트 프레임 내에서 이벤트의 시공간 일관성을 평가하여 노이즈 및 비이벤트 영역을 제거하여 시간적 중복 평가에 대한 영향을 방지합니다. 두 번째 단계인 Temporal Token Selection은 인접한 이벤트 프레임 간의 모션 패턴 유사성을 평가하여 중복된 시간 정보를 정확히 식별하고 제거합니다. PSTTS는 UniformerV2, VideoSwin, EVMamba, ExACT와 같은 다양한 모델 아키텍처에 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과 PSTTS는 DailyDVS-200 데이터셋에서 FLOPs를 29-43.6% 감소시키고 FPS를 21.6-41.3% 증가시키며 작업 정확도를 유지하는 데 성공했습니다. 본 연구는 PSTTS가 현재의 최첨단 방법들을 초과하여 효율성을 크게 개선할 수 있음을 보여줍니다. 또한 PSTTS는 다양한 모델 아키텍처에 적용 가능하여 뛰어난 일반화 가능성을 지닙니다.



### Bézier Meets Diffusion: Robust Generation Across Domains for Medical Image Segmentation (https://arxiv.org/abs/2509.22476)
Comments:
          17 pages, 7 figures

- **What's New**: 본 논문은 Bézier-curve 기반의 스타일 전이 방법을 제안하여 기존의 GAN 기반 도메인 전이 방법의 한계를 극복하고자 합니다. 이 방법은 고차원에서의 유사성을 최적화하여 소스 도메인과 타겟 도메인 간의 도메인 격차를 줄이고, 보다 강력한 세분화 모델을 훈련하기 위한 기반을 마련합니다. 또한, 조건부 확산 모델(Conditional Diffusion Model, CDM)을 통해 높은 품질의 라벨링된 타겟 도메인 이미지를 생성합니다.

- **Technical Details**: 제안된 방법은 Bézier-curve 기반의 스타일 전이 및 조건부 확산 모델을 활용하여 라벨이 없는 타겟 도메인에서 효과적으로 이미지를 생성하는 프레임워크를 형성합니다. CDM은 소스 도메인의 세분화 마스크를 바탕으로 타겟 도메인 이미지를 생성하는데, 이는 도메인 간 분포가 동일하다는 가정에 기초합니다. 또한, 노이즈가 포함된 의사 라벨(pseudo-labels)의 영향을 줄이기 위해 불확실성에 기반한 훈련 기법을 개발하였습니다.

- **Performance Highlights**: 공공 데이터 세트를 기반으로 한 실험 결과, 제안한 프레임워크는 사실적인 라벨링 이미지를 생성하여 타겟 도메인을 상당히 증강하고 세분화 성능을 향상시켰음을 보여줍니다. 특히, Bézier-curve 기반 스타일 전이 방법과 CDM의 시너지가 돋보이며, 이러한 접근법은 다른 기존 방법들과 결합하여 성능을 더욱 높일 수 있습니다.



### SSVIF: Self-Supervised Segmentation-Oriented Visible and Infrared Image Fusion (https://arxiv.org/abs/2509.22450)
- **What's New**: 이번 논문에서는 segmentation-oriented VIF 방법을 위한 새로운 자기 지도 학습 프레임워크인 SSVIF를 제안합니다. 이 프레임워크는 수동 분할 레이블 없이도 결합 이미지의 분할 성능을 고려하여 학습할 수 있습니다. 또한, SSVIF는 교차 분할 일관성(cross-segmentation consistency)이라는 새로운 자기 지도 태스크를 통해 의미 있는 의미적 특징을 학습할 수 있도록 합니다.

- **Technical Details**: SSVIF는 두 개의 분지 구조를 채택하여, 추가적인 CSC 손실을 도입함으로써 모델이 비수치적 레이블의 감독 없이도 의미적 특징을 학습하도록 합니다. 논문에서는 두 단계의 훈련 전략과 동적 가중치 조정 방법(GDWA)을 설계하여 훈련 중에 결합 손실과 CSC 손실의 기여를 효과적으로 균형 잡을 수 있도록 합니다.

- **Performance Highlights**: 공공 데이터셋에서의 광범위한 실험 결과 SSVIF가 기존의 전통적 VIF 방법보다 우수한 성능을 보임을 확인했습니다. SSVIF는 비표시된 가시-적외선 이미지 쌍에서만 훈련되었음에도 불구하고, 감독된 분할 지향 방법들과 경쟁할 수 있는 성능을 발휘하고 있습니다.



### $γ$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition (https://arxiv.org/abs/2509.22448)
Comments:
          Accepted at DAGM GCPR 2025

- **What's New**: 본 연구는 데이터 전송 대역폭과 에너지가 제한된 환경에서 최적화된 양자화 방법인 γ-Quant를 제안합니다. 이는 사람의 개입 없이 자동화된 분석의 효율성을 높이는 데 중점을 두며, 저비트 정량화(low-bit quantization)를 통해 데이터 품질과 에너지 소비를 균형 있게 조절하는 새로운 관점을 제공합니다. 또한, 4비트로도 성능이 크게 저하되지 않고 객체 감지 및 인체 활동 인식을 수행할 수 있음을 보여줍니다.

- **Technical Details**: γ-Quant는 비선형 양자화(non-linear quantization)를 학습(learning)하여 특정 작업을 위해 최적화됩니다. 이 연구에서는 고비트(raw) 이미지와 바디-부착 센서(body-worn sensor)에서 얻은 데이터를 통해 성능을 검증하였으며, 이를 통해 4비트 정량화 방식이 12비트를 사용하는 것과 유사한 성능을 발휘함을 입증하였습니다. 이 방법론은 특별한 매개변수 γ로 표현되며, 정량화 프로세스를 최적화하여 신경망(Neural Networks)과 함께 작동합니다.

- **Performance Highlights**: 연구 결과 γ-Quant를 적용한 모델이 기존의 선형 정량화 방법에 비해 정확도나 성능에서 현저한 향상을 보였으며, 저비트 데이터 전송으로도 효과적으로 작동함을 확인했습니다. 객체 감지와 인체 활동 인식 모두에서 γ-Quant를 통한 성능 향상이 입증되었으며, 이는 특히 배터리 수명과 데이터 전송 효율성을 크게 개선할 수 있는 가능성을 제시합니다. 따라서 본 연구는 차세대 인공지능 시스템에서 데이터 처리의 새로운 기준을 설정할 것으로 기대됩니다.



### U-MAN: U-Net with Multi-scale Adaptive KAN Network for Medical Image Segmentation (https://arxiv.org/abs/2509.22444)
Comments:
          5 pages

- **What's New**: 이 논문에서는 전통적인 U-Net 아키텍처의 한계를 극복하기 위해 U-Net with Multi-scale Adaptive KAN (U-MAN)이라는 새로운 구조를 제안합니다. 이 구조는 특징 집합의 세분성(semantic gap)을 해소하고, 복잡한 비선형 관계의 처리를 지원합니다. 특히, Progressive Attention-Guided Feature Fusion (PAGF)와 Multi-scale Adaptive KAN (MAN)이라는 두 개의 모듈을 통해 성능을 극대화합니다.

- **Technical Details**: U-MAN은 하이브리드 인코더-디코더 구조로, 다양한 해상도에서 다중 스케일(feature processing)을 지원합니다. MAN 모듈은 KAN 기반 피쳐와 Multi-Scale Attention Blocks (MSAB)을 결합하여 다양한 스케일에서의 특징 추출 능력을 강화합니다. PAGF 모듈은 전통적인 스킵 연결을 대체하여, 디코더와 인코더로부터의 특징을 더욱 합리적으로 융합하는 기능을 갖추고 있습니다.

- **Performance Highlights**: U-MAN은 BUSI, GLAS, CVC 등의 세 가지 공공 데이터셋에서 실험을 통해 기존의 최첨단 방법보다 우수한 성능을 보여주었습니다. 특히, 정확한 경계를 정의하고 세부 사항을 보존하는 데 우수한 효과를 나타냈습니다. 이러한 결과는 U-MAN의 보편성과 의료 영상 처리에서의 효율성을 입증합니다.



### Explaining multimodal LLMs via intra-modal token interactions (https://arxiv.org/abs/2509.22415)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 해석 가능성을 높이기 위해 인트라-모달( intra-modal) 상호작용을 활용한 새로운 접근 방식을 제안합니다. 기존의 연구는 주로 크로스-모달( cross-modal) 기여도 분석에 초점을 맞추었으나, 인트라-모달 의존성을 간과했습니다. 이로 인해 단편적이고 노이즈가 포함된 설명이 발생하는 문제를 해결하기 위해, 새로운 해석 가능성 프레임워크를 도입했습니다.

- **Technical Details**: 제안된 접근법은 두 가지 보완적인 구성 요소로 구성됩니다. 하나는 Multi-Scale Explanation Aggregation (MSEA)으로, 다양한 스케일의 입력을 통해 시각적 기여도를 계산하고 통합하여 공간적 컨텍스트를 포착합니다. 또 하나는 Activation Ranking Correlation (ARC)으로, 현재 토큰과 관련된 지각적 정보의 상대적 중요성을 평가하여 무관한 선행 토큰의 영향을 줄이는 방법을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 다양한 MLLM 모델에서 기존의 해석 방법을 초월하는 성능을 보였습니다. Qwen2-VL-2B 모델을 포함한 여러 모델에서 3.69%에서 14.52%의 정량적 개선을 기록하였으며, COCO Caption 데이터셋을 사용할 때 더욱 두드러진 성과를 나타냈습니다. 이를 통해 인트라-모달 상호작용을 고려한 새로운 접근 방식의 유효성을 강화했습니다.



### LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer (https://arxiv.org/abs/2509.22414)
Comments:
          Project Page: this https URL

- **What's New**: LucidFlux는 캡션 없이 대규모 확산 변환기(Flux.1)를 적용한 새로운 보편적 이미지 복원(UIR) 프레임워크입니다. 이를 통해 저하된 입력 이미지와 lightly restored proxy의 신호를 활용하여 기하학을 고정하고 아티팩트를 억제하는 경량화된 두 가지 분기 모듈을 도입하였습니다. 이 시스템은 복원 과정에서의 지연(latency)과 불안정성을 피하고, 명확한 의미적 정렬을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: LucidFlux는 시점-적응 모듈(timestep- and layer-adaptive modulation schedule)을 사용하여, 데이터의 계층 구조에 따라 신호를 조정합니다. 저품질 이미지에서 기하학적 정보를 취득하는 첫 번째 분기와 아티팩트를 억제하는 두 번째 분기가 강력한 모델링 기능을 발휘하며, SigLIP에서 추출한 특성을 기반으로 캡션 없는 의미 정렬(semantic alignment)을 수행합니다. 이 프레임워크는 구조가 풍부한 데이터 세트를 위한 자동화된 세 단계의 큐레이션 파이프라인과 결합되어 있습니다.

- **Performance Highlights**: LucidFlux는 여러 합성 및 실제 벤치마크에서 기존의 강력한 오픈소스 및 상업적 기반보다 consistently 높은 성능을 보이며, 각 구성 요소의 필요성을 확인하는 ablation study를 통해 그 유효성을 입증했습니다. 또한, LucidFlux는 백만 개 이상의 매개 변수에서 다양한 훈련 세트를 조합하고, 고유한 구조적 정보로 회복 효율성을 높였습니다.



### FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing (https://arxiv.org/abs/2509.22412)
Comments:
          Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)

- **What's New**: 이번 논문에서는 딥페이크 탐지기의 일반화 능력을 저해하는 새로운 형태의 모델 바이어스, 즉 스펙트럼 바이어스를 찾아냈습니다. 연구팀은 FreqDebias라는 주파수 디바이싱 프레임워크를 제안하여 특정 주파수 대역에 대한 과도한 의존을 완화합니다. 이를 통해 탐지기가 보지 못한 위조 샘플을 식별할 수 있는 능력을 향상시킵니다.

- **Technical Details**: FreqDebias는 두 가지 상호보완적인 전략을 통합합니다. 첫 번째는 Forgery Mixup (Fo-Mixup)이라는 새로운 데이터 증강 기법으로, 이는 훈련 샘플의 주파수 특성을 동적으로 다양화합니다. 두 번째는 이중 일관성 정규화 (CR)을 도입하여 지역적 및 전역적으로 일관된 표현 학습을 촉진합니다.

- **Performance Highlights**: 실험 결과, FreqDebias 프레임워크는 교차 도메인 설정에서 기존의 최신 기술(SOTA)들과 비교하여 유의미한 성능 향상을 보여줍니다. 특히, FreqDebias는 인도메인과 교차 도메인 모두에서 효과적으로 일반화 능력을 배양하며, 탐지기의 성능을 크게 개선하였습니다.



### RAU: Reference-based Anatomical Understanding with Vision Language Models (https://arxiv.org/abs/2509.22404)
- **What's New**: 본 논문은 RAU라는 새로운 프레임워크를 소개하며, 이는 시각-언어 모델(VLMs)을 활용한 참조 기반(anatomical understanding) 해석을 가능하게 합니다. 동작 중에 제공되는 참조 이미지를 사용해 비슷한 목표 이미지를 이해하도록 VLM을 학습시킵니다. 이를 통해 작은 해부학적 영역에 대한 정밀한 표시(segmentation)와 위치 확인(localization)을 통합하는 데 성공하였습니다.

- **Technical Details**: RAU는 참조 이미지를 기반으로 VLM이 상대적인 공간 추론(relative spatial reasoning)을 통해 해부학적 영역을 식별하도록 학습합니다. 또한, SAM2와의 통합을 통해 정밀한 세분화(segmentation) 능력을 결합하여 작은 해부학적 구조를 지역화할 수 있게 했습니다. 이 과정에서 시각적 질문 답변(visual question answering, VQA) 및 바운딩 박스 예측(bounding box prediction)으로 그 능력을 검증했습니다.

- **Performance Highlights**: RAU는 두 가지 기준 내(in-distribution) 데이터셋과 두 가지 기준 외(out-of-distribution) 데이터셋에서 일관되게 SAM2의 파인 튜닝(fine-tuning) 기준선을 초과하는 성과를 보여주었습니다. 특히, RAU는 분포 변동에 대한 강력한 일반화 능력을 갖추고 있어 자동화된 임상 워크플로우에서 큰 잠재력을 가지고 있습니다. 본 연구는 참조 기반으로 해부학적 구조를 식별하고 지역화하는 VLM의 최초 탐색을 포함하여 의료 이미지 이해를 위한 VLM 주도 접근 방식의 가능성을 강조합니다.



### Closing the Safety Gap: Surgical Concept Erasure in Visual Autoregressive Models (https://arxiv.org/abs/2509.22400)
- **What's New**: 이 논문은 시각 오토회귀 모델(Visual Autoregressive Models, VAR)을 위한 새로운 개념 지우기(framework)를 제안한다. VARE라는 이 프레임워크는 보조 시각 토큰을 활용하여 안정적인 개념 지우기를 가능하게 한다. S-VARE라는 새롭고 효과적인 방법도 도입하여 안전하지 않은 시각 토큰을 정확히 식별하고 최소한으로 조정하는 방식을 제안한다.

- **Technical Details**: VARE는 기존의 방법들이 VAR 모델에 직접 적용될 수 없다는 문제를 해결하기 위해 설계되었다. 이 프레임워크에서는 보조 대상 토큰을 추가 입력으로 사용하여 토큰 정렬의 불일치를 완화한다. S-VARE는 필터링된 크로스 엔트로피 손실(filtered cross entropy loss)과 보존 손실(preservation loss)을 사용하여 언어 드리프트와 출력 다양성을 유지하면서 개념을 효과적으로 지운다.

- **Performance Highlights**: 실험 결과, S-VARE는 97%의 민감한 개념을 지우는 동시에 CLIP 점수가 2% 이하로 감소하는 결과를 보였다. 따라서 VAR 모델을 안전하고 효율적으로 배포할 수 있는 중요한 기여를 한다. 이러한 방식은 기존의 기술들보다 질적으로 향상된 작업을 가능하게 한다.



### Integrating Background Knowledge in Medical Semantic Segmentation with Logic Tensor Networks (https://arxiv.org/abs/2509.22399)
Comments:
          Accepted at TAIM@IJCNN 2025

- **What's New**: 이번 연구는 의료 이미지 분석에서 의미 분할(semantic segmentation) 작업을 개선하기 위한 새로운 접근 방법을 소개합니다. 특히, Logic Tensor Networks (LTNs)를 활용하여 첫 번째 논리 규칙을 이용한 의료 배경 지식을 통합함으로써 세그멘테이션 모델의 성능을 향상시킬 수 있음을 강조합니다. 이 방법론은 훈련 데이터가 부족한 상황에서도 효과적으로 작동하며, 신경 상징적(neurosymbolic) 방법들이 다른 의료 세그멘테이션 작업에도 적용 가능성을 가지고 있음을 제시합니다.

- **Technical Details**: 의미 분할 작업은 이미지의 각 픽셀에 특정 범주를 할당하는 과정을 포함합니다. 본 연구에서 사용한 Swin-UNETR는 최첨단 아키텍처로, 이미지 처리 작업을 위해 설계된 Swin Transformer를 기반으로 합니다. Swin-UNETR는 인코더-디코더 구조를 사용하여 입력 이미지를 계층적으로 처리하고, LTN을 통해 추가적인 논리 제약을 통합하여 학습 과정에서 세밀한 정보 처리를 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 뇌 MRI 스캔에서 해마를 분할하는 작업에 대해 평가되었습니다. 초기 실험 결과에 따르면, LTN을 통합한 접근 방식은 기본 세그멘테이션 성능을 개선하는 데 성공하였음을 보여주었습니다. 또한, 데이터가 부족한 상황에서 특히 개선 효과가 두드러지며, 이는 의료 세그멘테이션 분야에서의 혁신적인 접근 방식을 나타냅니다.



### Text Adversarial Attacks with Dynamic Outputs (https://arxiv.org/abs/2509.22393)
- **What's New**: 이번 논문에서는 전통적인 텍스트 분류 모델이 동적 출력 환경에서의 공격에 취약하다는 점을 강조합니다. 이를 해결하기 위해 동적 출력을 정적 상황으로 변환하는 텍스트 동적 출력 공격(Textual Dynamic Outputs Attack, TDOA) 방법을 제안합니다. TDOA는 클러스터링 기반의 서브 모델 훈련 방식을 사용하여 새로운 전략인 '가장 먼 레이블 목표 공격(Farthest Label Targeted Attack, FLTA)'을 포함시켜 공격의 효과성을 높입니다.

- **Technical Details**: TDOA는 기존의 정적 출력 모델과 달리, 입력 텍스트에 따라 동적으로 변화하는 레이블의 수치와 내용을 처리할 수 있습니다. 이를 위해 TDOA는 피해 모델에서 동적 레이블을 추출하고 이를 벡터화 하여 클러스터링을 통해 정적 레이블로 변환합니다. 또한, 이 과정에서 ‘하드 레이블 블랙박스 제약’과 같은 실제적 제약 조건을 고려하여 공격의 세부 사항을 조정합니다.

- **Performance Highlights**: TDOA는 여러 데이터셋에서 8종의 피해 모델에 대해 평가되었으며, 단일 질의로 최대 50.81%의 공격 성공률을 기록하였습니다. 또한, 일반적인 정적 출력 환경에서도 82.68%의 최고 ASR를 달성하며, 번역 작업에서도 기존 결과를 초과하여 최고 성능을 보였습니다. 이러한 결과는 TDOA가 텍스트 공격에 대한 저항력을 강화하는 데 기여할 수 있는 가능성을 보여줍니다.



### Gradient-based multi-focus image fusion with focus-aware saliency enhancemen (https://arxiv.org/abs/2509.22392)
Comments:
          iCIG 2025

- **What's New**: 본 논문에서는 다중 초점 이미지 융합(MFIF) 기술을 개선하여, 여러 개의 부분적으로 초점이 맞춰진 입력으로부터 전체 초점이 맞춰진 이미지를 생성하는 방법을 제안합니다. 기존의 방법들이 초점과 불명료함 사이의 경계를 잘 보존하지 못하는 문제를 해결하는데 중점을 두었습니다.

- **Technical Details**: 제안된 방법은 경계 강화를 기반으로 하여 고품질의 융합된 경계를 생성하고 초점 정보를 효과적으로 감지합니다. 특히, 그라디언트 도메인(gradient-domain) 기반 모델을 사용하여 완전한 경계와 세부 사항을 보존하며 초기 융합 결과를 확보합니다.

- **Performance Highlights**: 본 연구는 4개의 공공 데이터셋에서 12개의 최첨단 방법들보다 우수한 성능을 지속적으로 보여주는 실험 결과를 보고합니다. 주관적 및 객관적 평가 모두에서 이 방법의 효과성을 입증했습니다.



### GPT-4 for Occlusion Order Recovery (https://arxiv.org/abs/2509.22383)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 연구는 GPT-4 모델을 활용하여 물체들의 폐색 순서를 예측하는 새롭고 효과적인 방법을 제안합니다. 이 방법은 입력 이미지와 특정 텍스트 프롬프트를 결합하여 GPT-4가 시각적 콘텐츠를 분석하고 폐색 행렬을 생성하도록 합니다. 이전 연구들과는 달리, 이 연구는 주석 데이터 없이도 정확한 폐색 순서 예측을 가능하게 합니다.

- **Technical Details**: 연구는 GPT-4의 고급 기능을 활용하여 이미지 내 물체의 폐색 관계를 유추하는 모델을 구현합니다. 프롬프트에 입력된 이미지와 관련된 객체 카테고리를 포함하여, 모델이 보다 정확한 출력을 생성할 수 있도록 합니다. 출력된 정보는 프로그램적으로 해석되어 폐색 행렬을 구축하는 데 사용됩니다.

- **Performance Highlights**: COCOA와 InstaOrder 데이터셋에서 평가를 진행한 결과, GPT-4는 더 정확한 폐색 순서 예측을 보여주었습니다. 기존의 베이스라인 모델들과 비교했을 때, 이 모델은 시맨틱 맥락과 상식 지식을 활용하여 정확성을 높였습니다. 결과적으로, 우리의 방법은 주석 데이터에 의존하지 않고 다양한 장면과 객체 유형에 일반화할 수 있는 장점이 있습니다.



### Effectiveness of Large Multimodal Models in Detecting Disinformation: Experimental Results (https://arxiv.org/abs/2509.22377)
Comments:
          9 pages

- **What's New**: 본 연구는 텍스트와 이미지가 결합된 다중 모드 상황에서의 허위 정보 탐지의 가능성을 탐구하며, GPT-4o 모델을 활용하여 이를 해결할 방법론을 제안합니다. 연구의 주요 기여로는 고급 프롬프트 엔지니어링 기법을 포함한 최적화된 프롬프트 개발과 이미지 및 텍스트에 대한 전처리 방법론이 포함됩니다. 특히, 우리는 여섯 가지 평가 기준을 정의하여 콘텐츠의 세분화된 분류를 가능하게 하고, 신뢰도 기반의 자기 평가 메커니즘을 도입하였습니다.

- **Technical Details**: 본 연구에서는 대규모 다중 모드 모델(Large Multimodal Models, LMMs)을 사용하여 허위 정보 탐지의 접근 방식을 제안합니다. 이는 이미지와 텍스트의 구조화된 프레임워크를 구현하고, 모델의 토큰 한도를 준수하는 전처리 방법론을 포함합니다. 여섯 가지 기준에 따른 구체적인 평가를 통해 모델의 성능을 다각적으로 분석하며, 반복 테스트를 통한 예측 변동성 평가를 통해 모델의 안정성과 신뢰성을 확인합니다.

- **Performance Highlights**: GPT-4o 모델은 Gossipcop, Politifact, Fakeddit, MMFakeBench, AMMEBA와 같은 다양한 데이터셋에서 성능 분석을 수행하며 허위 정보 탐지의 강점과 한계를 드러냅니다. 우리의 방법론은 신뢰도 수준 및 변동성 기반 평가 방법을 도입하여 성능을 더 강화했습니다. 이러한 결과는 다중 모드 허위 정보 분석을 위한 견고하고 재현 가능한 방법론적 프레임워크를 제공하며, 실제 세계 시나리오에서의 효과를 평가할 수 있는 기반을 마련합니다.



### HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography (https://arxiv.org/abs/2509.22365)
- **What's New**: HierLight-YOLO는 드론 촬영에서의 소형 객체 실시간 검출을 향상시키기 위한 경량화된 계층적 특징 융합 모델입니다. 이 모델은 YOLOv8 아키텍처를 기반으로 하며, Hierarchical Extended Path Aggregation Network (HEPAN)를 도입하여 다양한 스케일의 특징을 개선합니다. 또한, Inverted Residual Depthwise Convolution Block (IRDCB) 및 Lightweight Downsample (LDown) 모듈을 포함하여, 모델의 파라미터 수와 연산 복잡도를 줄이고, 작은 객체 검출 성능을 높였습니다.

- **Technical Details**: HierLight-YOLO는 소형 객체 검출 성능을 향상시키기 위해 최적화된 두 가지 경량 모듈을 포함합니다. IRDCB 모듈은 깊이별 분리 합성을 사용하여 22.1%의 파라미터 감소를 이루었고, LDown 모듈은 11.4%의 추가적인 감소를 통해 계산 자원이 제한된 환경에서도 실시간 작동이 가능합니다. 이러한 구조적 혁신을 통해 작은 객체(4 픽셀) 검출을 더욱 효과적으로 수행할 수 있도록 설계된 작은 객체 검출 헤드도 도입했습니다.

- **Performance Highlights**: HierLight-YOLO는 VisDrone2019 벤치마크에서 YOLOv8 기준 대비 3.3% 향상된 평균 정밀도(AP)를 달성하며, 133 FPS의 실시간 처리 속도를 유지합니다. 모델의 클래스 활성화 열지도는 작은 객체에 대한 더 강력한 반응과 더 정밀한 공간적 로컬라이제이션을 보여줍니다. 이러한 성능 개선은 HEPAN의 계층적 특징 융합과 혁신적인 경량 모듈 덕분임을 시각적으로 확인할 수 있습니다.



### CircuitSense: A Hierarchical Circuit System Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process (https://arxiv.org/abs/2509.22339)
- **What's New**: 이번 연구에서는 CircuitSense라는 새로운 벤치마크를 제안합니다. 이는 8,006개 이상의 문제를 통해 회로 이해도를 평가하는 포괄적인 기준으로, 시스템 수준 블록 다이어그램에서 구성 요소 수준 회로도에 이르기까지 다단계 평가를 수행합니다. 더욱이, 이 벤치마크는 시각적 입력에서 기호 방정식을 유도하는 능력의 중요성을 강조하며, 현재 AI 시스템이 이 기능을 얼마나 잘 수행하는지 평가합니다.

- **Technical Details**: CircuitSense는 세 가지 작업 범주(Perception, Analysis, Design)와 여섯 가지 계층 수준으로 구성되어 있으며, 회로 이해도를 평가하기 위해 체계적으로 설계되었습니다. 이 연구에서는 저자들과 공신력 있는 데이터 세트에서 수집한 2,986개의 문제와 5,020개의 합성 회로 문제를 포함하여 총 8,006개의 문제를 사용합니다. 각 단계에서 시각적 이해와 수학적 추론을 완전히 평가하는 것이 가능합니다.

- **Performance Highlights**: 최신 MLLMs에 대한 포괄적인 평가를 통해, 폐쇄형 모델은 인식 작업에서 85% 이상의 정확도를 달성하지만 기호 유도 및 분석 추론에서는 19% 이하로 떨어지는 한계를 드러냈습니다. 강한 기호 추론 능력을 갖춘 모델은 설계 작업에서 일관되게 높은 정확도를 보여주며, 이는 회로 합성을 위한 수학적 이해의 중요성을 확립합니다.



### Pedestrian Attribute Recognition via Hierarchical Cross-Modality HyperGraph Learning (https://arxiv.org/abs/2509.22331)
Comments:
          The First Work that Exploits Multi-modal Knowledge Graph for Pedestrian Attribute Recognition

- **What's New**: 이 논문은 보행자 속성 인식(Pedestrian Attribute Recognition, PAR)의 정확성을 향상시키기 위해 다중 모달 지식 그래프(multi-modal knowledge graph)를 구축하는 새로운 방법을 제안합니다. 현재의 알고리즘들이 시각적 특징과 속성을 단순히 결합하는 데 그치고 있는 반면, 이 연구는 속성과 시각적 맥락 간의 관계를 탐구하여 보다 정교한 인식을 가능하게 합니다. 이를 통해 PAR에서의 지식 기반 인식의 가능성을 크게 높이고 있습니다.

- **Technical Details**: 논문에서는 지식 그래프 안내 크로스 모달 하이퍼그래프 학습(framework) 방법론을 도입하여 다중 모달 지식 그래프의 관계를 효과적으로 모델링합니다. M2PA-KG라는 이름의 새 지식 그래프는 보행자 몸체, 속성 및 각각의 속성의 언어 캡션과 시각적 샘플을 포함한 다양한 엔티티를 설정하며, 고차원 관계를 캡처하기 위해 하이퍼그래프를 사용합니다. 이러한 관계는 LA-UniGNN 및 AG-UniGNN로 인코딩되고, 최종적으로 multi-modal Transformer를 통해 비주얼-시맨틱 집합을 처리합니다.

- **Performance Highlights**: 다양한 PAR 벤치마크 데이터셋에서의 포괄적인 실험을 통해 제안된 M2PA-KG의 효과성을 검증하였으며, 이는 지식 기반 보행자 속성 인식의 중요한 기초를 마련합니다. 기존의 속성 인식 방법들에 비해 좋은 성능 향상을 보였고, 하이퍼그래프 모델링을 통해 고차원 관계를 효과적으로 활용하였습니다.



### RAPID^3: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer (https://arxiv.org/abs/2509.22323)
- **What's New**: 이번 연구는 RAPID3라는 새로운 프레임워크를 소개하여, Diffusion Transformers(DiTs)의 샘플링 속도를 극적으로 개선합니다. RAPID3는 기본 생성기에는 업데이트 없이 이미지별로 가속화를 제공하며, 세 가지 경량화된 정책 헤드인 Step-Skip, Cache-Reuse, Sparse-Attention이 현재의 디노이징 상태를 관찰하고 독립적으로 속도 향상을 결정합니다. 이 방식으로 기존의 장점은 유지하면서도 생성 품질의 손실 없이 샘플링 속도를 거의 3배 향상시킵니다.

- **Technical Details**: RAPID3는 Group Relative Policy Optimization(GRPO)을 통해 모든 정책 파라미터를 온라인으로 훈련합니다. 여기서, 각 정책 헤드는 잠재 공간의 요약, 시점 및 프롬프트를 관찰하고, 그에 따른 가속화 전략을 독립적으로 결정합니다. 또한, 기존의 평가 메트릭을 사용하는 대신, 적대적 학습을 통해 보상을 극대화 하여 생성된 샘플이 원래 모델의 분포와 가까운 경우에만 보상을 증가시키는 방식으로 보상 해킹을 방지합니다.

- **Performance Highlights**: 실험 결과, RAPID3는 Stable Diffusion 3와 FLUX에서 각각 약 3배 속도로 샘플링을 수행하면서도 경쟁력 있는 시각적 품질을 유지합니다. 정책 헤드만 업데이트하는 방식으로, 이 헤드는 전체 생성기 파라미터의 단 0.025%에 해당하며, 훈련 과정에서 텍스트 프롬프트만 사용하여 실제 동적 신경망 훈련에 필요한 GPU 시간의 1%만 소모합니다.



### NIFTY: a Non-Local Image Flow Matching for Texture Synthesis (https://arxiv.org/abs/2509.22318)
- **What's New**: 이 논문은 예제 기반 텍스처 합성에 대한 문제를 다루며, NIFTY라는 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 최근의 컨볼루션 신경망(CNN)으로 훈련된 확산 모델(diffusion model)과 고전적인 패치 기반 텍스처 최적화 기법을 결합합니다. NIFTY는 비모수적(non-parametric) 플로우-매칭(flow-matching) 모델로, 신경망 훈련 없이도 일반적인 패치 기반 방법의 단점을 완화합니다.

- **Technical Details**: NIFTY는 패치의 비국소(non-local) 매칭에 기반한 모델로, 텍스처 합성을 위해 패치 분포에서 흐름(flow)을 명시적으로 계산하고 효율적으로 근사합니다. 본 논문에서는 신경망 없이 패치 기반 텍스처 합성을 위한 효율적인 흐름 계산 방법을 상세히 설명합니다. 따라서, 냉정한 파라미터의 관리나 초기화 과정에서의 민감함 없이도 텍스처 생성이 가능하다는 점이 강조됩니다.

- **Performance Highlights**: NIFTY는 실험 결과를 통해 초기화에 대한 강건성, 이미지 품질 및 속도에서 여러 장점들을 보여줍니다. 특히, 패치의 수를 줄이면서도 품질을 유지하는 새로운 기법들을 도입해 효율성을 높였습니다. 이러한 접근 방식은 이미지 합성에서 기대하는 결과를 달성할 수 있는 잠재력을 지니고 있습니다.



### Johnson-Lindenstrauss Lemma Guided Network for Efficient 3D Medical Segmentation (https://arxiv.org/abs/2509.22307)
- **What's New**: 이 논문에서는 3D 의학 이미지 분할에서 '효율성( efficiency )과 견고성( robustness )의 충돌' 문제를 해결하기 위해 VeloxSeg라는 새로운 프레임워크를 제안합니다. 이 방법은 고차원 3D 이미지의 특성을 기반으로 설계되었으며, 가벼운 방법의 취약한 표현을 극복하기 위해 데이터 시너지를 탐색합니다. 복합 해부 구조와 이질적인 모달리티를 처리하는데 집중합니다.

- **Technical Details**: VeloxSeg는 Paired Window Attention (PWA)와 Johnson-Lindenstrauss lemma-guided convolution (JLC)로 구성된 이중 스트림 CNN-Transformer 아키텍처를 기반으로 합니다. PWA는 다양한 스케일의 정보를 신속하게 호출하고, JLC는 최소한의 매개변수로 견고한 로컬 특징 추출을 보장합니다. 이러한 구조를 통해 낮은 계산 비용으로 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과에 따르면, VeloxSeg는 멀티모달 벤치마크에서 26%의 Dice 개선을 이루었으며, GPU 처리량을 11배, CPU는 48배 향상시켰습니다. 이는 VeloxSeg가 기존의 기준 모델보다 더 강력한 표현력을 가진다는 것을 뜻합니다.



### HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models (https://arxiv.org/abs/2509.22300)
- **What's New**: 이번 논문에서는 이미지 생성을 위한 확산 모델(difusion models)의 품질 향상을 위해 역사 기반 샘플링 기법인 HiGS(이력 안내 샘플링)를 제안합니다. HiGS는 모델의 최근 예측값을 통합하여 샘플링 과정에서 더 사실적인 결과를 유도합니다. 이 접근법은 기존 확산 프레임워크에 쉽게 통합될 수 있으며, 추가 훈련(trainig)이나 튜닝(tuning) 없이 사용할 수 있습니다.

- **Technical Details**: HiGS는 현재의 예측값과 과거 예측들의 가중 평균(weighted average)의 차이를 활용하여 샘플링을 진행합니다. 이 기법은 추가적인 계산(computation)을 필요로 하지 않으며, 효율성을 높이는데 중점을 두고 있습니다. 실험 결과, 다양한 모델과 아키텍처에서 HiGS는 일관되게 이미지 품질을 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: HiGS는 사전 훈련된 SiT 모델을 사용하여, 가이드 없이 ImageNet 생성에서 새로운 최첨단 FID(Frechet Inception Distance)인 1.61을 달성하였습니다. 이는 표준 250단계의 샘플링 대신 30단계로 이루어진 것입니다. HiGS는 더 높은 충실도의 이미지 생성을 위해 빠른 생성 속도를 제공하는 플러그 앤 플레이(plug-and-play) 개선책으로 자리잡을 것입니다.



### Jailbreaking on Text-to-Video Models via Scene Splitting Strategy (https://arxiv.org/abs/2509.22292)
- **What's New**: 최근 Text-to-Video (T2V) 모델의 급속한 발전과 함께 이러한 모델의 안전성 위험에 대한 우려가 증가하고 있습니다. 기존의 연구들은 LLMs, VLMs, T2I 모델의 취약점을 jailbreak 공격을 통해 탐구했지만 T2V 모델은 거의 탐구되지 않았습니다. 이 논문에서는 SceneSplit이라는 새로운 블랙박스(jailbreak) 방법을 소개하며, 이 방법은 해로운 내러티브를 여러 장면으로 분할하여 각 장면이 개별적으로 무해하도록 합니다.

- **Technical Details**: SceneSplit은 생성(output) 공간을 조작하여 내러티브 구조를 악용하는 공격 방식을 제시합니다. 개별 장면은 대부분의 결과가 무해한 넓고 안전한 공간에 해당하지만, 이들을 순차적으로 결합함으로써 안전하지 않은 영역으로 좁혀집니다. 이 핵심 메커니즘은 반복적인 장면 조작을 통해 안전 필터를 우회하여 효율성을 높입니다.

- **Performance Highlights**: SceneSplit은 T2V 모델에서 11개의 안전 카테고리를 평가하여 공격 성공률(Attack Success Rate, ASR)에서 77.2% (Luma Ray2), 84.1% (Hailuo), 78.2% (Veo2)라는 높은 평균 값을 달성했습니다. 기존의 기준 대비 상당히 개선된 성과를 보이며, T2V 모델의 안전 메커니즘이 내러티브 구조를 이용한 공격에 취약하다는 새로운 통찰을 제공합니다.



### Rule-Based Reinforcement Learning for Document Image Classification with Vision Language Models (https://arxiv.org/abs/2509.22283)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 rule-based reinforcement learning (RL)을 문서 이미지 분류에 적용할 수 있는 가능성을 탐구하고 있습니다. 기존의 supervised fine-tuning (SFT) 방식 대신 RL을 사용해도 높은 일반화 성능을 보일 수 있음을 제안합니다. 세 가지 시나리오에서 RL과 SFT의 일반화 능력을 비교하여, RL의 도출된 추론 능력의 영향을 검토합니다.

- **Technical Details**: DeepSeek-R1을 통해 입증된 rule-based RL의 성공 이후, 연구자들은 다양한 모델에 이를 적용하기 시작했습니다. 이 연구에서는 RL이 문서 이미지 분류에 어떻게 적용될 수 있는지에 대해 논의하며, RVL-CDIP 데이터셋을 기반으로 실험을 진행합니다. 문서 이해를 위한 공개 데이터셋과 합성 데이터셋을 포함하여 기존 모델들이 학습된 내용에도 주목하며, RL이 문서 이미지 분류의 성능을 향상할 수 있음을 시사합니다.

- **Performance Highlights**: 성능 측면에서, RL 기반 접근 방식이 out-of-distribution 데이터에 대한 일반화 능력이 뛰어남을 발견했습니다. 이는 unseen classes와 다양한 modalities에서의 평가를 포함하여 검토되었습니다. 이러한 결과는 문서 이미지 분류와 같은 다운스트림(Downstream) 작업에서도 RL의 장점을 극대화할 수 있는 가능성을 보여줍니다.



### MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning (https://arxiv.org/abs/2509.22281)
Comments:
          Accepted by NeurIPS 2025; Project page: this https URL

- **What's New**: 이 논문에서는 작업 지침을 해석하고 실행하는 로봇을 위한 새로운 과제, 즉 작업 지향 테이블탑 장면 생성(Task-oriented tabletop scene generation)의 도전을 다룹니다. 이를 해결하기 위해, 저자들은 MesaTask-10K라는 대규모 데이터셋을 소개하며, 10,700개의 합성 테이블탑 장면으로 구성되어 있습니다. 이 데이터셋은 작업 변수와 인터랙티브한 장면 구성의 필요성을 충족시키는 수작업 레이아웃으로 이루어져 있습니다. 또한, Spatial Reasoning Chain라는 새로운 접근 방식을 제안하여 객체 추론과 공간적 관계 추정을 통해 최종 3D 레이아웃을 구성하는 구조적인 사고 체인을 형성합니다.

- **Technical Details**: MesaTask 프레임워크는 LLM(대형 언어 모델)에 기반하여 작업 지향 테이블탑 장면 생성을 가능하게 합니다. 이 프레임워크는 객체 속성과 상호 공간 관계를 통해 장면 그래프를 형성하고, 각 장면에 대해 훈련 데이터를 습득하기 위한 섬세한 규칙을 설계합니다. Direct Preference Optimization(DPO) 알고리즘을 이용하여 생성된 장면들이 객체 충돌이 없고 주어진 작업 설명과 잘 정렬됨을 보장합니다. 이 과정에서 VLM(비전-언어 모델)을 활용하여 여러 관점에서 평가를 수행하며, 물리적으로 그럴듯한 테이블탑 장면을 생성합니다.

- **Performance Highlights**: MesaTask 프레임워크는 기존 방법들과 비교하여 매우 우수한 성능을 발휘합니다. FID, VLM 기반 메트릭스 및 사용자 연구에 대한 평가에서 벤치마크 방법들을 초월하는 결과를 낼 수 있었습니다. 특히, 생성된 테이블탑 장면들은 주어진 작업 지침을 엄격히 따르며, 스태킹이나 포함과 같은 복잡한 객체 간 관계를 효과적으로 보여 줍니다. 저자들은 이러한 성과를 통해 작업 간의 격차를 해소할 수 있는 새로운 데이터셋과 프레임워크를 제공했다고 주장합니다.



### GS-2M: Gaussian Splatting for Joint Mesh Reconstruction and Material Decomposition (https://arxiv.org/abs/2509.22276)
Comments:
          13 pages, 10 figures

- **What's New**: GS-2M이라는 이름의 새로운 통합 솔루션을 제설계하여 다중 뷰 이미지에서 메쉬 재건과 재료 분해를 동시에 수행하도록 제안하였습니다. 기존 기법들은 이 두 가지 작업을 별도로 처리하며, 특히 반사 면을 재구성하는 데 어려움을 겪고 있습니다. 반면, 우리의 방법은 재구성 품질을 유지하면서 이러한 두 문제를 함께 최적화하여 해결합니다.

- **Technical Details**: GS-2M 프레임워크는 3D Gaussian Splatting을 기반으로 하며, 다중 뷰 광학 변화에 기반한 새로운 거칠기 감독 전략을 제안합니다. 이는 사전 훈련된 모델의 가정에 의존하지 않고 독립적으로 작동하여 성능을 확장할 수 있는 장점을 제공합니다. 또한, 다중 뷰 이미지에서 향상된 메쉬 재건을 위해 재료 매개변수를 훈련 파이프라인에 통합하여 지오메트릭 아티팩트를 제거하는 조치를 취하고 있습니다.

- **Performance Highlights**: 우리의 접근 방식은 여러 널리 사용되는 데이터 세트를 통해 효과성을 검증하였으며, 최신 표면 재구성 방법들과 비교하여 이미지 품질을 유지합니다. GS-2M은 테스셀레이션(mesh)과 이와 관련된 재료 요소를 제공하며, 이를 통해 후속 작업에서도 우수한 성능을 발휘합니다. 여러 평가에서 우리의 프레임워크는 최신 기술을 활용한 방법들과 유사한 메쉬 재구성 품질을 달성하였습니다.



### UniMapGen: A Generative Framework for Large-Scale Map Construction from Multi-modal Data (https://arxiv.org/abs/2509.22262)
Comments:
          17 pages, 10 figures

- **What's New**: 본 논문에서는 새로운 대규모 지도 구축 프레임워크인 UniMapGen을 제안합니다. UniMapGen은 기존 방법의 한계를 극복하기 위한 세 가지 주요 혁신을 포함하고 있습니다: 첫째, 도로 라인을 분산된 시퀀스로 표현하고, 둘째, 다중 모달 입력을 수용할 수 있는 유연한 아키텍처를 제공하며, 셋째, 전역 일관성과 연속성을 위한 상태 업데이트 전략을 개발했습니다.

- **Technical Details**: UniMapGen의 핵심 혁신 중 하나는 도로 라인 생성을 토큰 기반 생성 문제로 재구성하는 것입니다. 이를 통해 다양한 길이의 도로 라인에 걸쳐 매끄러운 벡터 출력을 생성할 수 있습니다. 또한 이 체계는 BEV( Bird's Eye View)와 PV( Perspective View) 및 텍스트 프롬프트와 같은 다양한 입력을 허용하여 위성 데이터의 단점을 극복합니다. 각 상태에서 업데이트되는 지도는 이전 상태의 지도에 기반하고, 상태 업데이트 전략을 통해 글로벌 연속성을 유지합니다.

- **Performance Highlights**: UniMapGen은 OpenSatMap 데이터 세트에서 최첨단 성능을 달성하였으며, 이 프레임워크는 가려진 도로를 추론하고 주석에서 누락된 도로를 예측할 수 있습니다. 이 연구는 라인 벡터의 직렬화와 반복적인 지도 생성을 통해 지도 품질을 향상시켰습니다. UniMapGen은 혼합 모달 접근 방식을 통해 위성 이미지로부터의 지도 구축에서 나타나는 주요 문제들을 해결하며, 이는 대규모 지도 구축의 새로운 가능성을 제시합니다.



### Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks (https://arxiv.org/abs/2509.22258)
Comments:
          23 pages, 12 figures

- **What's New**: 최근 비전-언어 모델(VLM)의 발전은 의료 AI 작업에서 놀라운 개선을 가져왔으나, 실제 임상 추론 능력은 여전히 불명확합니다. 이에 따라 Neural-MedBench라는 새로운 벤치마크를 소개하며, 이는 다중모달(clinical reasoning)을 탐구하도록 설계되었습니다. Neural-MedBench는 다중 MRI 스캔, 전자 건강 기록, 임상 노트를 통합하여 차별 진단, 병변 인식, 논리 생성 등 세 가지 핵심 작업을 포함합니다.

- **Technical Details**: Neural-MedBench는 120개의 전문가 주석이 달린 다중모달 사례로 구성되며, 200개의 추론 집약적 작업을 생성합니다. 이 벤치마크는 의료 교육의 관행을 반영하여 디자인되었으며, 대상의 불확실성 아래에서 추론 능력을 평가할 수 있도록 설계되었습니다. 평가 과정에서는 LLM 기반의 평가자, 임상의 검증, 의미 유사성 메트릭을 결합하는 하이브리드 점수 시스템을 개발하여 신뢰할 수 있는 평가를 보장합니다.

- **Performance Highlights**: 최신 VLM 모델의 평가 결과, 대규모 벤치마크에서 우수한 성과를 내는 모델들이 Neural-MedBench에서는 체계적으로 실패하는 경향을 보였습니다. 오류 분석을 통해 이러한 실패는 인지적 오류가 아닌 임상 추론의 한계에서 기인함을 확인했습니다. 이러한 결과들은 깊이 있는 평가가 필요함을 강조하며, Neural-MedBench가 신뢰할 수 있는 임상 AI의 평가에 기여할 수 있음을 시사합니다.



### FlashEdit: Decoupling Speed, Structure, and Semantics for Precise Image Editing (https://arxiv.org/abs/2509.22244)
Comments:
          Our code will be made publicly available at this https URL

- **What's New**: FlashEdit는 고충실도의 실시간 이미지 편집을 가능하게 하는 새로운 프레임워크입니다. 이 시스템은 One-Step Inversion-and-Editing (OSIE) 파이프라인과 Background Shield (BG-Shield) 기법, Sparsified Spatial Cross-Attention (SSCA) 메커니즘을 통해 효율적인 수정을 구현합니다. 이러한 혁신들은 기존의 방법보다 150배 이상의 속도 향상을 제공하며, 편집이 0.2초 이내에 완료됩니다.

- **Technical Details**: FlashEdit의 주요 혁신은 세 가지로 구성됩니다. 첫째, One-Step Inversion-and-Editing (OSIE) 파이프라인은 반복적인 프로세스를 우회하여 높은 편집 품질을 유지합니다. 둘째, Background Shield (BG-Shield) 기술은 선택적으로 수정된 영역 내에서만 특징을 수정하여 배경을 보존합니다. 셋째, Sparsified Spatial Cross-Attention (SSCA) 메커니즘은 의미 자원의 유출을 최소화하여 정밀하고 지역적인 편집을 보장합니다.

- **Performance Highlights**: FlashEdit는 실험 결과로 배경 일관성과 구조적 무결성을 유지하면서도 빠른 편집 성능을 보여주었습니다. 기존의 다단계 방법에 비해 150배 이상의 속도 향상을 달성하며, 초당 0.2초 이내에 편집을 완료합니다. 이러한 성능은 고급 이미지 편집 기능을 제공하여 다양한 실제 응용에서 큰 장점을 가집니다.



### A Tale of Two Experts: Cooperative Learning for Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2509.22229)
- **What's New**: 본 논문에서는 Source-Free Unsupervised Domain Adaptation (SFUDA) 문제를 해결하기 위해 Experts Cooperative Learning (EXCL)이라는 새로운 방법론을 제안합니다. EXCL은 Dual Experts 프레임워크와 Retrieval-Augmentation-Interaction (RAIN) 최적화 파이프라인을 포함하여, 라벨이 없는 타겟 샘플으로부터 정보의 합의를 유도합니다. 이는 데이터 프라이버시 문제로 인해 소스 도메인 데이터에 접근할 수 없는 상황에서도 효과적인 도메인 적응을 이루기 위한 접근법입니다.

- **Technical Details**: EXCL은 두 개의 주요 구성 요소로 나누어지며, 첫째로 Dual Experts 프레임워크는 소스 도메인 모델과 비주얼-언어 모델(VLM)을 동등하게 하여 비라벨 타겟 데이터에서 합의 지식을 채굴합니다. 둘째로, RAIN 파이프라인은 (1) 의사-소스 및 복잡한 타겟 샘플을 협력적으로 검색하고, (2) 각 전문가를 해당 샘플 세트에 대해 개별적으로 미세 조정하며, (3) 공유된 학습 결과를 통해 학습 목표의 일관성을 강화하는 세 가지 단계로 구성됩니다.

- **Performance Highlights**: 네 개의 벤치마크 데이터셋에서의 실험 결과는 EXCL 방법이 최신 기술(state-of-the-art)과 동등한 성능을 보여줍니다. 이는 우리의 접근 방식이 다양한 데이터셋에서 일반화 성능을 크게 향상시킬 수 있음을 시사합니다. 이러한 결과들은 EXCL의 제안된 프레임워크와 최적화 방법이 효과적임을 입증합니다.



### UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspectiv (https://arxiv.org/abs/2509.22228)
Comments:
          13 pages, 6 figures

- **What's New**: UrbanFeel이라는 새로운 벤치마크가 제안되었습니다. 이 벤치마크는 도시 개발을 이해하고 주관적 환경 인식을 평가하기 위해 Multimodal Large Language Models (MLLMs)의 성능을 측정하는 데 중점을 두고 있습니다. UrbanFeel은 Static Scene Perception, Temporal Change Understanding, Subjective Environmental Perception의 세 가지 차원으로 구성된 14,300개의 시각적 질문으로 설계되었습니다.

- **Technical Details**: UrbanFeel은 11개의 대표적인 도시에서 수집된 다중 시간대의 단일 뷰 및 파노라마 스트리트 뷰 이미지를 기반으로 제작되었습니다. 이 벤치마크는 공간 클러스터링, 규칙 기반 생성, 모델 보조 프롬프트, 수동 주석과 같은 하이브리드 파이프라인을 통해 고품질의 질문-응답 쌍을 생성합니다. 이는 MLLMs의 인식, 추론 및 인적 인식 일치 능력을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: 20개의 최첨단 MLLMs에 대한 평가 결과, Gemini-2.5 Pro가 인간 전문가 수준에 가까운 정확도를 보여주며 최상의 성능을 보였습니다. 대부분의 모델은 장면 이해와 관련된 과제에서 성능이 좋았지만, 도시 개발에 대한 시간적 추론이 필요한 작업에서는 성능이 크게 떨어졌습니다. 주관적 인식 차원에서 몇몇 모델은 아름다움과 안전과 같은 차원에 대해 인간 수준의 일관성을 달성했습니다.



### Polysemous Language Gaussian Splatting via Matching-based Mask Lifting (https://arxiv.org/abs/2509.22225)
- **What's New**: MUSplat은 3D Gaussian Splatting(3DGS) 장면의 오픈-보카불러리(open-vocabulary) 이해를 위한 새롭고 훈련이 필요 없는 프레임워크입니다. 기존의 복잡한 훈련과 최적화 과정 없이 다중 해상도의 2D 마스크를 3D로 변환하여 객체 그룹을 형성합니다. 이 방법은 관점에 따라 객체의 모습 해석을 통해 텍스트 기능을 정제하여 언어 기반 검색을 가능한 직관적으로 수행합니다.

- **Technical Details**: 이 프레임워크는 각 Gaussian의 의미를 매칭 메커니즘으로 결정하며, 다의적(polysemous) 표현을 지원하기 위해 설계되었습니다. 초기의 객체 그룹 경계는 중립점 처리(neutral point processing)를 통해 개선되며, Vision-Language Model(VLM)을 활용하여 각 객체에 대한 강력한 텍스트적 표현을 생성합니다. 이를 통해 각 Gaussian point의 전경 확률을 추정하여 초기 객체 그룹을 형성합니다.

- **Performance Highlights**: MUSplat은 기존의 훈련 기반 프레임워크들보다 적응 시간이 수 시간에서 수 분으로 단축되는 장점을 가지고 있습니다. 오픈-보카불러리 3D 객체 선택 및 의미 없는 분할의 기준 작업에서 성공적으로 규정된 방식으로 성능이 우수함을 입증하였습니다. 이 모델은 특히 복잡한 의미 표현에서 기존의 한정된 방식으로부터 발생하는 한계를 해결하고 있습니다.



### Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models (https://arxiv.org/abs/2509.22221)
- **What's New**: 본 논문에서는 Perceptually-Grounded Geospatial Chain-of-Thought (Geo-CoT)라는 새로운 프레임워크를 소개합니다. Geo-CoT는 원거리 감시(remote sensing) 분석을 검증 가능한 다단계의 과정으로 모델링하여 Vision-Language Models (VLMs)의 복잡한 분석 작업을 개선합니다. 이 프레임워크는 Geo-CoT380k라는 대규모 데이터셋을 활용하여 모델의 사고 구조를 효과적으로 정립하는 두 단계의 정렬 전략을 구현합니다.

- **Technical Details**: 프레임워크는 두 단계의 접근 방식을 통해 구성됩니다. 첫 번째 단계는 Supervised Fine-Tuning (SFT)을 통해 기본적인 인지 구조를 구축하고, 두 번째 단계는 Group Reward Policy Optimization (GRPO)을 통해 모델의 추론 정책을 사실적 정확성을 위해 세밀하게 조정합니다. 이 과정은 모델이 최종 답변과 함께 그에 대한 정당화 및 검증 가능한 분석 흔적을 출력할 수 있도록 합니다.

- **Performance Highlights**: RSThinker라는 모델은 Geo-CoT 프레임워크를 기반으로 하며, 다양한 작업에서 최신의 성능을 능가합니다. 연구 결과, 이 모델은 기존의 최첨단 모델들을 통틀어 가장 우수한 성과를 보이며, 자연재해 대응과 환경 모니터링과 같은 고위험 원거리 감시에 중요한 검증 가능성을 제시합니다. Geo-CoT380k 데이터셋과 RSThinker 모델이 발표됨에 따라, 불투명한 인식에서 구조적이고 검증 가능한 추론으로의 실질적인 경로가 제시됩니다.



### MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing (https://arxiv.org/abs/2509.22186)
Comments:
          Technical Report; GitHub Repo: this https URL Hugging Face Model: this https URL Hugging Face Demo: this https URL

- **What's New**: MinerU2.5는 1.2B 매개변수를 가진 문서 파싱 비전-언어 모델로 가장 발전된 인식 정확도를 달성하고 있을 뿐만 아니라 뛰어난 계산 효율성을 유지합니다. 이 모델은 글로벌 레이아웃 분석을 로컬 콘텐츠 인식과 분리하는 조잡한-세밀한 두 단계 파싱 전략을 채택하였습니다. 이러한 접근은 고해상도 입력을 처리하는 경우의 계산 오버헤드를 피할 수 있게 합니다.

- **Technical Details**: MinerU2.5는 두 단계의 추론 메커니즘을 통해 문서 이미지를 처리합니다. 첫 번째 단계에서는 다운샘플링된 문서 이미지에 대한 빠르고 포괄적인 레이아웃 분석을 수행하여 구조적 요소를 식별합니다. 두 번째 단계에서는 원본 이미지에서 추출한 지역의 고해상도에 맞춰 세밀한 내용을 인식하여 복잡한 수식 및 테이블과 같은 세밀한 디테일을 보존합니다.

- **Performance Highlights**: MinerU2.5는 여러 벤치마크에서 상태-of-the-art (SOTA) 성능을 달성하며, 다양한 인식 작업에서 일반 목적 및 도메인 특화 모델을 초월합니다. 특히, 복잡한 문서에서 높은 파싱 정확도를 유지하면서 오버헤드를 크게 줄이는 혁신적인 아키텍처와 종합적인 데이터 엔진을 개발하여 문서 파싱 분야의 강력한 성능을 입증하였습니다.



### DragGANSpace: Latent Space Exploration and Control for GANs (https://arxiv.org/abs/2509.22169)
Comments:
          6 pages with 7 figures and 3 tables

- **What's New**: 이번 연구는 StyleGAN, DragGAN, 그리고 Principal Component Analysis (PCA)를 통합하여 GAN 생성 이미지의 잠재 공간(latent space) 효율성과 제어 능력을 향상시킵니다. 이 방법론을 통해 PCA 기반의 차원 축소 기법을 사용하여 DragGAN 프레임워크의 이미지 조작 성능을 유지하면서 최적화 효율성을 개선하는 데 성공했습니다. 특히, DragGAN의 잠재 W+ 레이어에 PCA를 도입하여 총 최적화 시간을 일관되게 줄일 수 있었으며, 이는 더욱 효율적이고 해석 가능한 잠재 공간 제어의 가능성을 제시합니다.

- **Technical Details**: 기술적 배경으로, GAN의 잠재 공간 조작의 복잡성을 개선하기 위해 StyleGAN2와 DragGAN을 활용하였습니다. StyleGAN2의 W+W+ 잠재 공간 구조는 각 레이어에서 개별 잠재 코드를 수용할 수 있어 이미지 속성의 분리된 조작을 가능하게 합니다. 그러나 이러한 잠재 속성을 조작하는 것은 여전히 복잡하고 계산 집약적이기 때문에, PCA를 적용하여 잠재 공간의 차원을 줄인 뒤 DragGAN을 통해 사용자가 보다 직관적으로 조작할 수 있도록 했습니다.

- **Performance Highlights**: 동물 얼굴 고화질(AFHQ) 데이터셋을 활용한 실험 결과, PCA를 적용한 DragGAN 프레임워크는 더 빠르고 정확한 이미지 수정 작업을 가능하게 하였습니다. PCA를 적용한 후, 최적화 시간은 단축되었으며, Structural Similarity Index Measure (SSIM) 지수도 향상되어 이미지 품질이 유지되었음을 확인했습니다. 이 방법론은 이미지 합성 및 편집 작업에 있어 광범위하게 활용될 수 있는 효율적이고 해석 가능성 있는 잠재 공간 제어의 가능성을 보여줍니다.



### MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models (https://arxiv.org/abs/2509.22151)
Comments:
          Submitted to ICLR 2026

- **What's New**: 이 논문은 MultiMat이라는 새로운 멀티모달 프로그램 합성 프레임워크를 소개합니다. 이 프레임워크는 시각적 및 텍스트 기반 그래프 표현을 동시에 처리하여 절차적 물질 그래프의 생성을 개선합니다. 기존의 텍스트 기반 프로그램 생성 접근법의 시각적 피드백 부족 문제를 해결함으로써 인간 아티스트의 창의적 작업 흐름을 더 잘 반영합니다.

- **Technical Details**: MultiMat는 새로운 데이터셋과 제약된 트리 탐색 추론 알고리즘을 활용하여 구문적 유효성을 보장하고, 프로그램 공간을 효율적으로 탐색합니다. 이 접근법은 중간 그래프의 시각화를 통합하여 발생하는 각 노드의 실시간 유효성을 검증할 수 있습니다. 변환기를 구현하여 Adobe Substance Designer 형식과 언어 모델링에 적합한 압축 표현 간의 전환을 지원하며, 이는 더 복잡한 물질 생성이 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MultiMat은 조건부 및 비조건부 그래프 합성에서 더 높은 시각적 품질과 충실도로 이전의 텍스트 전용 접근법보다 더 효율적인 성능을 보여줍니다. 이러한 새로운 기술은 절차적 물질 합성의 새로운 최첨단 성능을 확립하며, 모든 수준의 프로필을 가진 아티스트에게 유용한 도구가 될 것입니다.



### Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions (https://arxiv.org/abs/2509.22150)
- **What's New**: 본 연구에서는 비독립적이고 동일하게 분포된(Non-IID) 3D 포인트 클라우드 데이터에 적합한 새로운 분류 전략인 Joint Graph Entropy Knowledge Distillation(JGEKD)를 제안합니다. JGEKD는 클래스 간 상관관계를 유지하는 지식 증류(knowledge distillation)를 통해 클래스 간의 관계를 정확히 설명할 수 있도록 설계되었습니다. 이를 통해 모델의 일반화 능력을 강화하고, 특히 공간 변환에 불변인 포인트 클라우드 데이터를 처리할 수 있습니다.

- **Technical Details**: Joint Graph Entropy Knowledge Distillation(JGEKD) 전략은 클래스 간의 잠재적 관계를 설명하기 위해 공동 그래프(joint graph)를 구축하고, 그래프 엔트로피(graph entropy)를 기반으로 클래스를 특성화합니다. 이러한 그래프를 통해 다양한 변환 형태 간에 정보 전이를 용이하게 하는 프레임워크인 JGEsKD 및 JGEtKD를 개발하였습니다. 또한, 손상된 포인트 클라우드 데이터를 처리하기 위한 적대적 훈련(adversarial training) 전략도 제안하였습니다.

- **Performance Highlights**: ScanObject, ModelNet40, ScanntV2_cls, ModelNet-C와 같은 데이터셋에서 수행된 광범위한 실험을 통해 제안된 JGEKD 전략이 경쟁력 있는 결과를 달성할 수 있음을 입증하였습니다. 제안된 프레임워크는 모델의 강건성을 높이고, 데이터의 다양한 변환 형태에 대해 보다 나은 일반화 성능을 제공합니다. 이러한 결과는 비독립적 및 동등하게 분포된 데이터가 현실 세계 문제를 해결하는 데 있어 얼마나 중요한지를 강조합니다.



### REFINE-CONTROL: A Semi-supervised Distillation Method For Conditional Image Generation (https://arxiv.org/abs/2509.22139)
Comments:
          5 pages,17 figures

- **What's New**: 이번 연구에서는 Refine-Control이라는 새로운 반지도 증류 프레임워크를 제안합니다. 이 프레임워크는 tri-level knowledge fusion loss를 도입하여 지식 전이의 다양한 수준을 효과적으로 수행하고, 라벨이 없는 데이터와 라벨이 있는 데이터를 활용한 반지도 학습 방법을 통해 데이터 부족 문제를 완화합니다. 이로 인해 계산 비용과 지연 시간이 크게 줄어들면서도 고품질 이미지 생성을 유지할 수 있습니다.

- **Technical Details**: Refine-Control 프레임워크는 이미지 인페인팅을 위한 두 단계 프로세스로, 기초 학습과 고급 정제가 있습니다. 첫 번째 단계에서는 완전히 주석이 달린 데이터셋을 사용하여 학생 모델이 기본적인 매핑을 배우도록 합니다. 이어지는 두 번째 단계에서는 라벨이 없는 데이터셋을 활용하여 자가 지도 미세 조정을 수행하며, tri-level knowledge fusion loss를 통해 교사 모델의 계층적 지식을 효율적으로 전이합니다.

- **Performance Highlights**: 실험 결과 Refine-Control은 계산 비용과 지연 시간을 크게 줄이면서도 높은 충실도의 이미지 생성 능력과 제어 가능성을 유지하는 것으로 나타났습니다. 특히, 학생 모델이 복잡한 장면에서 다수의 객체를 다룰 때 발생하는 오해를 줄이는 데 기여했습니다. 또한, 이 프레임워크는 기존 모델보다 실용적인 데이터 접근을 허용하여 실제 환경에서도 효율적으로 활용될 수 있음을 보여주었습니다.



### Self-Supervised Point Cloud Completion based on Multi-View Augmentations of Single Partial Point Cloud (https://arxiv.org/abs/2509.22132)
- **What's New**: 이 논문에서는 부분 점 구름(Partial Point Cloud)을 기반으로 하는 새로운 자가 지도(self-supervised) 점 구름 완성(Point Cloud Completion) 방법을 제안합니다. 기존 방법들의 한계를 극복하기 위해 단일 점 구름에서 다중 시점(multi-view) 증강을 통해 새로운 자가 지도 신호(self-supervised signals)를 설계했습니다. 또한, Mamba를 최초로 자가 지도 점 구름 완성 작업에 통합하여 모델의 품질을 향상시킴으로써 더욱 관련성 있는 결과를 도출했습니다.

- **Technical Details**: 제안된 방법은 단일 부분 점 구름에서 생성된 다양한 불완전한 형태를 활용하여 모델의 훈련을 도와주는 다중 시점 증강 방식을 이용합니다. 이러한 방법을 통해 모델은 다양한 불완전성에 적응하고 이를 인식하는 능력을 향상시킵니다. Mamba 기반 인코더는 전역(feature extraction) 및 지역(local feature) 특성을 추출하는 능력을 강화하여 더 정확한 예측 결과를 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 합성(synthetic) 및 실제(real-world) 데이터셋에서 기존 방법들보다 더 높은 성능을 기록하며, 더 정확한 형태의 점 구름을 생성합니다. 제안하는 접근법은 기존 자가 지도 방법들이 제공한 결과들보다 더욱 세밀한 구조적 디테일을 포함하고 있어 성능의 향상을 보여줍니다. 이러한 결과는 실제 응용 가능성에 중요한 기여를 할 것으로 기대됩니다.



### Large Material Gaussian Model for Relightable 3D Generation (https://arxiv.org/abs/2509.22112)
- **What's New**: 이 논문에서는 Physically Based Rendering (PBR) 재질 속성을 가진 고품질 3D 콘텐츠 생성을 위한 새로운 프레임워크인 Large Material Gaussian Model (MGM)을 소개합니다. 기존의 3D 생성 모델들은 재질 속성을 고려하지 않아 현실적인 조명 환경에서의 렌더링 품질이 부족했습니다. 그러나 MGM은 입력 깊이 및 노멀 맵에 조건화된 다중 뷰 재질 확산 모델을 활용하여 이를 해결하고, PBR 속성을 극대화하여 더욱 사실적인 렌더링 효과를 제공합니다.

- **Technical Details**: 본 연구는 텍스트 프롬프트를 기초로 하여 PBR 이미지를 생성하는 멀티뷰 재질 확산 모델과 2D Gaussian Splatting 기반의 대규모 복원 모델 두 가지 주요 구성 요소로 구성됩니다. 이렇게 생산된 PBR 이미지는 각 재질 채널을 모델링하는 Gaussian 재질 표현과 통합되어 복원되고, 다양한 조명 조건에서의 역동적인 재조명 기능을 가능하게 합니다. 또한 깊이와 노멀 맵을 포함한 기하학적 데이터를 통합함으로써 생성 및 복원 단계에서의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, MGM을 통해 생성된 재질은 기존의 방법보다 시각적인 매력이 뛰어나고, 재질 모델링의 질 또한 향상되었습니다. 본 논문에서 제안하는 방법은 최신 Gaussian Splatting 기반 생성 품질과 견주어 보아도 동등하거나 우수한 성능을 보여주며, 다양한 조명 조건에서도 사실적인 렌더링 결과를 달성했습니다. 이로 인해 MGM은 3D 자산 생성과 관련된 광범위한 응용에 적합한 솔루션으로 자리잡을 것입니다.



### SpecXNet: A Dual-Domain Convolutional Network for Robust Deepfake Detection (https://arxiv.org/abs/2509.22070)
Comments:
          ACM MM Accepted

- **What's New**: GANs와 확산 모델이 생성한 콘텐츠의 현실성이 증가함에 따라 딥페이크 탐지가 상당히 어려워졌습니다. 기존의 방법들은 종종 공간적 또는 주파수 도메인(feature) 특성에만 초점을 맞추어, 보이지 않는 조작에 대한 일반화에 한계를 가지고 있습니다. 본 연구에서는 Spectral Cross-Attentional Network(SpecXNet)이라는 이중 도메인 아키텍처를 제안하여 강력한 딥페이크 탐지를 가능하게 합니다.

- **Technical Details**: SpecXNet의 핵심 구성 요소는 Dual-Domain Feature Coupler(DDFC)로, 이 특성은 텍스처 수준의 이상을 포착하기 위한 지역 공간 분기와 주기적 불일치를 모델링하기 위해 Fast Fourier Transform을 사용하는 전역 스펙트럼 분기로 분해됩니다. 이러한 이중 도메인 구조는 SpecXNet이 정통 이미지와 조작된 이미지를 구분하는 데 중요할 수 있는 지역적 세부 정보와 전역적 구조적 일관성을 공동으로 활용할 수 있도록 합니다. 또한, 우리는 콘텐츠 인식 방식으로 공간적 및 스펙트럼 특성을 동적으로 결합하는 Dual Fourier Attention(DFA) 모듈을 도입했습니다.

- **Performance Highlights**: 다양한 딥페이크 벤치마크에 대한 광범위한 실험 결과, SpecXNet은 특히 크로스 데이터셋 및 보이지 않는 조작 시나리오에서 최첨단 정확도를 달성했으며, 실시간 구현 가능성을 유지합니다. 결과는 강력하고 일반화 가능한 딥페이크 탐지를 위한 통합 공간-스펙트럴 학습의 효과를 강조합니다. 재현성을 보장하기 위해 본 논문에서는 전체 코드를 GitHub에 공개했습니다.



### High-Quality Sound Separation Across Diverse Categories via Visually-Guided Generative Modeling (https://arxiv.org/abs/2509.22063)
Comments:
          Accepted to IJCV

- **What's New**: 이번 논문에서는 DAVIS라는 새로운 Diffusion 기반의 오디오-비주얼 분리(framework)를 제안합니다. 기존의 방법론이 마스크 기반의 회귀 문제로 프레임을 구성한 것에 비해, DAVIS는 강력한 생성 모델링 패러다임을 활용하여 복잡한 데이터 분포를 캡처합니다. 특히 Denoising Diffusion Probabilistic Models (DDPM)와 Flow Matching (FM)을 이용하여 성공적으로 오디오-비주얼 분리 문제를 해결합니다.

- **Technical Details**: DAVIS는 오디오 혼합 신호와 관련된 비주얼 정보를 동시에 조건으로 하여, 노이즈 분포에서 원하는 분리된 스펙트로그램을 직접 생성하는 과정을 거칩니다. 이 과정은 Separation U-Net 아키텍처를基하여 이루어지며, Convolution-Attention (CA) 블록을 통해 스펙트로그램 데이터의 복잡성을 처리하고 장거리 의존성을 포착합니다. 또한, ℒ1 (L1) 손실을 사용하여 훈련하면서 데이터 분포의 왜곡을 완화하고, 침묵 마스크 기반의 추론 전략을 제안하여 그 일관성을 높입니다.

- **Performance Highlights**: DAVIS의 DDPM 및 DAVIS-Flow 버전은 AVE 및 MUSIC 데이터셋에서 기존 최첨단 방법들을 초과하는 분리 품질을 보여주었습니다. 실험 결과는 우리의 생성적 프레임워크가 오디오-비주얼 분리 작업에 효과적이라는 것을 입증합니다. 또한 CLIP 모델을 통해 제로샷(text-guided)의 오디오 분리라는 새로운 가능성을 열었습니다.



### EgoInstruct: An Egocentric Video Dataset of Face-to-face Instructional Interactions with Multi-modal LLM Benchmarking (https://arxiv.org/abs/2509.22019)
Comments:
          Accepted to the I-HFM Workshop at ICCV 2025

- **What's New**: 이 논문에서는 얼굴을 맞대고 교육을 받는 장면을 분석하기 위한 새로운 egocentric 비디오 데이터셋을 소개하고 있습니다. 이 데이터셋은 동기화된 시선 정보, 신체 움직임, 그리고 대화 상황 분류를 통해 의사소통의 복잡성을 보다 잘 이해할 수 있게 해 줍니다. 이는 기존의 머신러닝 모델이 효과적으로 이러한 상호작용을 처리할 수 있도록 지원합니다.

- **Technical Details**: 저자들은 두 가지 기본 작업을 정의합니다: 절차적 단계 분할(procedural step segmentation)과 대화 상태 분류(conversation-state classification)입니다. 새로운 데이터셋은 38회의 수업을 포함하여, 사용자가 수행하는 각 단계와 발화에 대한 주석을 제공합니다. 이 연구에서 다루는 MLLMs는 이미지, 오디오 및 텍스트를 동시에 처리하는 최신 기법을 기반으로 하고 있습니다.

- **Performance Highlights**: 실험 결과, MLLMs는 기존의 특정 작업을 위한 모델보다 우수한 성능을 보여주었습니다. 이들은 별도의 작업 맞춤형 미세 조정 없이도 얼굴을 맞대고 한 교육 장면을 이해하는 데 더 효과적이라는 것을 시사합니다. 이러한 결과는 앞으로의 교육 지원과 기술 이전을 위한 머신러닝 모델의 가능성을 보여줍니다.



### Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics (https://arxiv.org/abs/2509.22014)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 의료 환경을 위한 경량의 에이전틱(multimodal) 프레임워크를 소개합니다. 기존의 비전-언어 모델들이 제한된 시간적(reasoning) 추론 및 불확실성 측정의 한계를 보여주는 것에 비해, 이 프레임워크는 비디오 기반의 장면 이해를 목표로 하여 업그레이드되었습니다. Qwen2.5-VL-3B-Instruct 모델과 SmolAgent 기반의 조정 레이어를 결합하여 사고 연쇄(chain-of-thought reasoning), 음성-비전 융합, 동적 도구 호출 기능을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 VisionQA, SceneGen 및 GraphQA라는 세 가지 보완 워크플로우로 구성되어 있습니다. 이 구조는 안전 비즈니스 환경에서 클리니션 및 로봇 시스템을 지원하기 위한 것이며, 의사결정의 해석 가능성과 추적 가능성을 보장합니다. 통합된 하이브리드 검색 메커니즘을 통해 효율적이고 해석 가능한 지식 통합이 가능하여, 로봇이 안전하게 행동할 수 있는 기반을 마련합니다.

- **Performance Highlights**: 실험 결과, Video-MME 벤치마크 및 맞춤형 임상 데이터 세트에서 기존의 비전-언어 모델에 비해 경쟁력 있는 정확도와 개선된 강인성을 보여주었습니다. 이 프레임워크는 로봇 보조 수술, 환자 모니터링, 의사 결정 지원 등 다양한 의료 응용 분야에서의 잠재력을 강조합니다. 따라서 안전하고 투명하며 적응력이 뛰어난 의료 로봇의 발전에 기여할 것으로 기대됩니다.



### CoFFT: Chain of Foresight-Focus Thought for Visual Language Models (https://arxiv.org/abs/2509.22010)
- **What's New**: 본 논문에서는 복잡한 시각적 입력으로 인한 VLM의 한계를 극복하기 위해 Chain of Foresight-Focus Thought (CoFFT)라는 훈련 없는 새로운 접근 방식을 제안합니다. 이 방법은 인간의 시각적 인지를 모방하여 VLM의 시각적 추론 능력을 향상시키는 데 초점을 맞춥니다. CoFFT는 다단계로 구성된 과정으로, 각 단계가 반복적으로 작용하여 VLM의 전반적인 추론 프로세스를 최적화합니다.

- **Technical Details**: CoFFT의 각 단계는 (1) Diverse Samples Generation (DSG), (2) Dual Foresight Decoding (DFD), (3) Visual Focus Adjustment (VFA)로 구성됩니다. DSG 단계에서는 다양한 샘플을 생성하여 복잡한 추론 경로를 탐색하고, DFD 단계에서는 생성된 샘플을 평가하여 최적의 샘플을 선택합니다. 마지막으로 VFA 단계에서는 질문과 미래의 추론 단계와 관련된 영역에 시각적 초점을 조정함으로써 다음 반복에 사용할 이미지를 결정합니다.

- **Performance Highlights**: 실험을 통해 CoFFT는 Qwen2.5-VL, InternVL-2.5, Llava-Next와 같은 여러 벤치마크에서 평균 3.1-5.8% 성능 향상을 보여주었습니다. 또한 모델의 파라미터 수에 따른 분석 결과, CoFFT는 더 큰 모델에서 더 큰 개선 효과를 보여 모델 크기와 함께 효과가 긍정적으로 확장되는 것으로 나타났습니다. CoFFT는 기존의 VLM 추론보다 추가적인 계산 비용이 필요하지만 Monte Carlo Tree Search보다 효율적이라는 점에서 실용성을 입증하였습니다.



### Exposing Hallucinations To Suppress Them: VLMs Representation Editing With Generative Anchors (https://arxiv.org/abs/2509.21997)
- **What's New**: 이번 연구에서는 다중 모달 대형 언어 모델(MLLMs)의 환각(hallucination) 문제를 해결하기 위한 새로운 방법을 제안합니다. 이 방법은 별도의 훈련 없이 자기 지도(self-supervised) 방식을 사용하여 환각을 완화하는 기법으로, 텍스트-이미지 모델을 활용하여 캡션(Caption)의 시각적 신호를 드러내는 새로운 환각 증폭 메커니즘을 도입합니다. 이 방식은 두 개의 앵커(anchor)를 사용하여 모델의 디코더(hidden states)를 수정함으로써 시각적 의미의 정확성을 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 환각을 감지하기 위해 텍스트-이미지 모델(T2I)을 통해 캡션을 시각적 공간으로 투영합니다. 이로 인해 암묵적인 환각 신호가 드러나며, 원본 이미지는 신뢰할 수 있는 의미의 앵커 역할을 합니다. 이 두 개의 앵커를 통해 디코더의 잠재 표현을 조정하여 원본 이미지의 깨끗한 의미로 끌어당기고 환각 방향으로부터 밀어냅니다. 이 과정은 추가적인 학습 비용 없이도 가능합니다.

- **Performance Highlights**: 다양한 벤치마크에서 수행된 실험 결과, 제안된 방법이 객체, 속성 및 관계 수준에서 환각을 유의미하게 줄인다는 것이 입증되었습니다. 예를 들어, LLaVA-v1.5-7B 모델을 사용하여 CHAIR 데이터셋에서 환각을 5% 이상 줄이는 성과를 보였습니다. 게다가 다양한 아키텍처에 대해 강력한 일반화 성능을 입증하며, 거의 모든 경우에 부작용 없이 기존 모델에 통합될 수 있는 강력한 방법임을 강조했습니다.



### FailureAtlas:Mapping the Failure Landscape of T2I Models via Active Exploration (https://arxiv.org/abs/2509.21995)
- **What's New**: 이번 논문에서는 Text-to-Image (T2I) 모델의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 기존의 정적 벤치마크가 체계적인 실패를 발견하는 데 한계가 있음을 지적하며, 이를 보완하기 위한 능동적인 탐색(active exploration) Paradigm을 도입합니다. 이로써 모델의 실패 방식에 대한 깊은 이해를 제공하는 FailureAtlas라는 첫 번째 프레임워크를 소개하며, 이를 통해 이전에는 발견되지 않았던 수많은 오류 조각(error slices)을 체계적으로 발견할 수 있게 되었습니다.

- **Technical Details**: FailureAtlas는 entity-attribute corpus를 활용하여 T2I 모델의 오류 조각을 탐색하는 것을 목표로 합니다. 이 과정은 세 가지 단계로 나누어지며, 첫 번째 단계에서는 대형 언어 모델(LLM)을 사용해 기본 어휘를 초기화합니다. 두 번째 단계에서는 두 개의 데이터셋인 COCO Captions와 T2I-CompBench에서 추출된 엔티티와 속성을 추가하여 어휘를 확장하고, 마지막 단계에서는 각 엔티티-속성 쌍의 의미적 유효성을 검증하여 비현실적인 프롬프트 생성을 방지합니다.

- **Performance Highlights**: FailureAtlas를 사용하여 SD1.5 모델에서 247,000개 이상의 오류 조각을 자동으로 발견하였으며, SDXL Turbo에서는 439,000개 이상의 오류 조각을 발견하였습니다. 이 연구는 이러한 오류가 훈련 데이터의 부족과 연관되며, 이는 T2I 모델의 한계를 극복하기 위한 중요한 단서를 제공합니다. 또한 기존 정적 벤치마크를 넘어서는 체계적이고 대규모의 진단 기법을 제시하며, 향후 강력한 생성 AI 개발을 위한 기초 자료를 제공합니다.



### Rate-Distortion Optimized Communication for Collaborative Perception (https://arxiv.org/abs/2509.21994)
- **What's New**: 이 논문에서는 다중 에이전트 협력 지각을 위한 새로운 개념인 실용적 비율-왜곡 이론(pragmatic rate-distortion theory)을 제안합니다. 이 이론은 통신 비율과 작업 성능 간의 무역을 모델링하여, 에이전트 간에 최적의 통신 전략을 설계하는 데 필요한 두 가지 조건인 실용적인 정보 제공과 중복 없는 메시지 전송을 정의합니다. 이를 통해 RDcomm이라는 커뮤니케이션 효율적인 협력 지각 프레임워크를 제안하며, 이는 두 가지 혁신적인 요소를 포함합니다.

- **Technical Details**: RDcomm은 작업과 관련된 정보 전달을 극대화하기 위한 작업 엔트로피(discrete coding) 모듈과, 상호 정보(mutual information)가 중심이 되는 메시지 선택 모듈로 구성됩니다. 첫째, 작업 중요도에 따라 코드워드 길이를 조정하여 정보전달의 효율성을 높이는 작업 엔트로피 이산 부호화(task entropy discrete coding) 방법을 도입합니다. 둘째, 에이전트 간의 메시지 중복을 평가하는 상호 정보 신경 추정(mutual information neural estimation)을 활용하여 중복을 최소화하는 메시지 선택 과정을 구현합니다.

- **Performance Highlights**: RDcomm은 3D 객체 탐지와 BEV 세그멘테이션 작업을 통해 실험을 진행하였으며, DAIR-V2X 및 OPV2V 데이터셋에서 기존 방법들과 비교하여 통신 볼륨을 최대 108배까지 줄이면서도 최고의 정확도를 달성했습니다. 이러한 결과는 RDcomm이 성능과 커뮤니케이션 효율성 모두에서 우수함을 보여줍니다. 본 논문은 다중 에이전트 시스템에서의 커뮤니케이션 효율성 분석을 위한 중요한 이론적 토대를 제공하며, 향후 연구 방향에 큰 기여를 할 것으로 보입니다.



### DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints (https://arxiv.org/abs/2509.21992)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이 논문에서는 DualFocus라는 새로운 Depth-from-Focus (DFF) 프레임워크를 제안합니다. 이는 포커스 변화에 의해 유도된 고유한 그라디언트 패턴을 활용하여 공간적 및 포커스 차원 전반에 걸쳐 포커스 변화를 공동 모델링합니다. 기존 방법들이 겪는 문제를 해결하기 위해, 이 방법은 DFF에 특화된 변별적인 제약 조건을 도입합니다.

- **Technical Details**: DualFocus는 두 가지 변별적 제약을 도입하여 깊이 추정을 수행합니다. 첫 번째는 공간적 제약으로, 포커스 레벨에 따른 그래디언트 패턴 변화를 분석하여 신뢰할 수 있는 깊이 엣지를 식별합니다. 두 번째는 포커스 제약으로, 포커스 거리에 따라 확률 분포를 조정하여 실제 깊이에서의 초점 확신을 유지하도록 합니다.

- **Performance Highlights**: 다양한 공개 데이터셋에서 엄청난 실험 결과를 보여줍니다. DualFocus는 깊이 정확도와 지각 품질 모두에서 최신 기술을 초월하는 성능을 나타내며, 데이터셋에 대한 일반화 성능 또한 뛰어납니다. 전통적인 DFF 기법에 비해 복잡한 장면에서 더욱 견고하고 정확한 깊이 추정이 가능합니다.



### ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models (https://arxiv.org/abs/2509.21991)
- **What's New**: 이번 연구에서는 고해상도 이미지 처리의 효율성을 높이기 위한 새로운 모델인 ERGO (Efficient Reasoning & Guided Observation)를 발표했습니다. 기존의 대형 비전-언어 모델(LVLMs)은 비디오 토큰 수로 인해 높은 계산 비용이 발생하는데, 이를 해결하기 위해 두 단계의 "coarse-to-fine" 추론 파이프라인을 제안했습니다. 분석된 이미지에서 중요한 지역을 식별한 후, 해당 영역만을 높은 해상도로 크롭하여 후속 재처리를 진행하는 방식입니다.

- **Technical Details**: ERGO는 강화 학습( Reinforcement Learning) 기반의 모델로, 이미지와 쿼리를 기반으로 이미지에서 중요한 영역의 바운딩 박스를 생성합니다. 이때, 전체 이미지에서 초점이 되는 영역을 재구성한 후 해당 영역에 대한 정확한 질문 응답을 수행합니다. 연구자는 이 모델이 비전 처리 효율성과 함께 정밀한 추론 능력도 유지할 수 있도록 설계했다고 설명합니다.

- **Performance Highlights**: 여러 데이터셋에서 ERGO는 기존 모델들과 비교하여 정확도를 크게 향상시키며, 적은 수의 비전 토큰 사용으로도 성능이 우수함을 입증했습니다. 예를 들어, ERGO는 V* 벤치마크에서 Qwen2.5-VL-7B를 4.7점 초과하며 비전 토큰의 23%만을 사용하여 3배의 추론 속도를 달성하였습니다. 이러한 결과는 고해상도 비전-언어 처리의 효율성과 정확성을 동시에 개선할 수 있음을 보여줍니다.



### WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM (https://arxiv.org/abs/2509.21990)
- **What's New**: WAVE는 텍스트, 오디오, 비디오 모달리티에 대한 통합 표현 공간을 생성하는 최초의 LLM 기반 임베딩입니다. 이 모델은 새롭고 계층적인 특징 융합 전략과 다중 모달 및 다중 작업 훈련 방식을 사용하여, 사용자 지침에 맞춘 프롬프트 인식 임베딩을 생성하고 모든 모달리티 간의 상호 검색 기능을 제공합니다. 연구 결과, WAVE는 MMEB-v2 비디오 벤치마크에서 최신 기술인 SOTA를 달성하며 다중 모달 질의 응답에서도 기존 모델을 능가하는 성능을 보여줍니다.

- **Technical Details**: WAVE는 다양한 입력(텍스트, 오디오, 무음 비디오 및 동기화된 오디오-비디오)에 대한 통합 임베딩을 생성하는 LLM입니다. 사용된 계층적 특징 융합 전략은 다중 MLLM 레이어로부터의 표현을 집계하여 다중 모달 검색과 같은 작업에서 안정적인 성과를 냅니다. 또한 오디오를 위한 이중 인코더 디자인을 통해 음성과 환경 소리에 대한 보완 신호를 포착하여 배운 임베딩의 표현력을 향상시킵니다.

- **Performance Highlights**: WAVE는 MMEB-v2 비디오 트랙에서 SOTA 성능을 달성하며, 텍스트-비디오, 비디오-오디오 등 다양한 전환 작업에서도 우수한 결과를 나타냅니다. 프롬프트 인식 능력이 있는 WAVE는 사용자 지침에 맞춰 여러 모달리티에 대한 임베딩을 생성하여 다중 모달 질의 응답에서도 돋보이는 성능을 발휘합니다. 이 연구는 다중 모달 이해력 벤치마크에서 기초 모델인 Qwen2.5-Omni의 성능을 유지하거나 초과하는 성과를 내며, 기존 모델들과 비교할 때 명확한 강점을 보여줍니다.



### Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation (https://arxiv.org/abs/2509.21989)
Comments:
          NeurIPS 2025 (Spotlight). Project Page: this https URL

- **What's New**: 본 논문에서는 사전 훈련된 diffusion 모델의 백본에서 시각적(visual) 및 의미적(semantic) 특징을 분리하는 새로운 접근 방식을 제안합니다. 이 방법은 이미지 합성의 품질을 지원하는 시각적 특징을 효과적으로 분리하여 의미적 특성과 유사한 방식으로 시각적 대응을 가능하게 합니다. 자동화된 데이터셋 생성 파이프라인을 소개하여 주어진 주체에 대한 시각적 및 의미적 대응이 주석 처리된 이미지 쌍을 구성합니다.

- **Technical Details**: 제안된 아키텍처는 대조적(constrastive) 방식으로 시각적 및 의미적 특징을 분리하여, Visual Semantic Matching(VSM)이라는 새로운 메트릭을 도출합니다. 이 메트릭은 주체에 기반한 이미지 생성에서 시각적 불일치 정도를 정량화하며, 기존의 CLIP, DINO 같은 메트릭보다 우수한 성능을 보입니다. 이를 통해 불일치 지역의 공간적 로컬라이징(localization)도 가능해집니다.

- **Performance Highlights**: 실험 결과, 본 접근 방식은 기존의 전역(feature-based) 메트릭보다 시각적 불일치를 정량화하는 데에서 더 높은 성능을 나타내며, 특히 불일치 지역의 로컬라이징을 지원합니다. 이는 주체 기반 생성에서 불일치의 양적 평가와 로컬라이징을 동시에 지원하는 최초의 방법으로, 이 작업의 발전에 중요한 도구가 될 수 있습니다.



### From Bias to Balance: Exploring and Mitigating Spatial Bias in LVLMs (https://arxiv.org/abs/2509.21984)
- **What's New**: 본 연구는 LVLMs의 공간 편향(spatial bias)에 대한 체계적인 검토를 제공하며, 동일한 시각 정보가 이미지 내에서 서로 다른 위치에 배치될 때 모델이 어떻게 반응하는지를 분석합니다. 기존 연구들이 시각적 콘텐츠의 공간적 이해가 약하다는 점을 지적하고 있으나, 이는 기본적으로 LLM(대형 언어 모델) 부분에서 발생하는 문제라는 점을 강조합니다. 이 연구는 Balanced Position Assignment (BaPA)라는 새로운 메커니즘을 제안하여, 모든 이미지 토큰에 같은 위치 임베딩(position embedding)을 부여함으로써 시각적 정보 통합의 균형을 촉진합니다.

- **Technical Details**: LVLM은 비전 인코더(vision encoder)와 대형 언어 모델(LLM)을 결합하여 시각적 그리고 텍스트 정보를 동시에 활용합니다. 공간적인 위치에 따른 모델의 반응을 분석하기 위해 설계된 탐침 데이터셋을 사용하여, 현재의 LVLM이 위치 변화에 민감하게 반응하며 일관성이 결여된 결과를 생산하는 경향을 발견했습니다. 분석 결과, 이러한 현상은 비전 인코더에서 파생되지 않고, LLM의 위치 임베딩(embedding) 설계의 불균형에서 기인한다고 밝혀졌습니다.

- **Performance Highlights**: BaPA를 적용한 실험은 LVLM의 공간적 견고성을 향상시키며, 재학습 없이도 성능 향상을 나타냅니다. 또한, BaPA는 경량의 미세 조정(lightweight fine-tuning)과 결합하였을 때, 다양한 멀티모달 벤치마크에서 성능을 추가로 향상시킵니다. 최종적으로, BaPA를 적용한 LVLM은 노출된 테스트 셋에서 더 균형 잡힌 주의를 생성함으로써 시각적 정보에 대한 보다 총체적인 이해를 가능하게 합니다.



### Resolving Ambiguity in Gaze-Facilitated Visual Assistant Interaction Paradigm (https://arxiv.org/abs/2509.21980)
- **What's New**: 최근 AI 시스템이 복잡한 비주얼 장면을 해석하고 이에 대한 자연어 질문에 응답할 수 있는 능력이 크게 향상되었습니다. 특히, 현대의 AR/VR 헤드셋 및 스마트 글라스는 주목 추적(gaze-tracking) 기능을 통합하여 AI 시스템이 이를 활용할 수 있게 되었습니다. 이러한 발전을 통해 인간-컴퓨터 상호작용(HCI) 연구에 대한 관심이 높아지고 있으며, 눈 추적 안경을 사용한 시각 질문 응답에서의 주목 신호의 중요성이 강조되고 있습니다.

- **Technical Details**: 이 논문에서는 GLARIFY라는 새로운 방법을 제안하여 주목의 동적 특성을 비주얼-언어 모델(VLMs)에 통합합니다. GLARIFY는 사용자의 눈 추적 데이터를 포함하여 다중모달 상호작용 시나리오에서 모호함을 해소하는 데 도움을 줍니다. 그리고 사용자의 시선에 따라 변화하는 데이터를 통해 훈련된 GLARIFY-Ambi라는 합성 데이터셋을 구축하여 모델의 성능을 평가합니다.

- **Performance Highlights**: GLARIFY는 기존의 기준선을 크게 초과하는 성능을 보여주었으며, 인간의 주목을 비주얼-언어 모델과 강력하게 정렬시킵니다. 또한, GLARIFY는 VLM과의 상호작용을 더 직관적이고 사용 가능한 패러다임으로 발전시킵니다. 이를 통해 시각 비서와의 상호작용에서 모호성을 효과적으로 해결할 수 있는 가능성을 제시합니다.



### Benchmarking and Mitigate Psychological Sycophancy in Medical Vision-Language Models (https://arxiv.org/abs/2509.21979)
Comments:
          19figures, 37pages

- **What's New**: 이번 연구는 비전 언어 모델(VLMs)이 의료 영상 및 임상 의사 결정 지원 시스템에 점점 더 많이 통합되고 있는 현황을 다룹니다. 그러나 이 모델들은 사용자의 언어적 표현이나 사회적 신호에 따라 실제 증거 기반의 추론보다 우선순위를 두는 경향이 있습니다. 따라서, 임상 질문 응답(visual question answering)에서 모델의 '아부적' 행동을 평가하기 위한 새로운 벤치마크를 제안합니다.

- **Technical Details**: 연구진은 PathVQA, SLAKE 및 VQA-RAD에서 다양한 장기 시스템 및 모드에 따라 계층화된 비슷한 패턴을 가진 의료 아부 데이터 세트를 구성했습니다. 심리적 압박 템플릿을 활용하여 다양한 아부 행동을 평가하고, 실험에서 모델의 약점을 분석했습니다. 특히, 시각적 증거와는 무관한 모델의 편향 메커니즘을 발견하였고, 이러한 문제를 해결하기 위해 VIPER(Visual Information Purification for Evidence based Response)라는 경량화된 완화 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, 다양한 모델에서 아부에 대한 취약성이 관찰되었으며, VIPER는 서로 다른 압박 유형과 시스템 간의 최상의 저항성과 복원 균형을 달성했습니다. 특히, 적용된 압박 유형에 따른 성능 변화와 VIPER의 효과를 심층적으로 분석하여, VIPER가 비주얼 증거에 기초한 답변 생성을 유지할 수 있도록 정렬 목표와 수용 압박 간의 상호작용을 강조했습니다. 이러한 연구 결과는 다중 모달리티 임상 AI의 행동 신뢰성을 평가하고 개선하는 기초를 제공합니다.



### Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning (https://arxiv.org/abs/2509.21976)
- **What's New**: 최근 원격 감지(이것은 Remote Sensing을 의미함)에서 지시 표현(Referring Expression) 이해에 대한 독창적인 접근 방식인 Geo-R1을 제안했습니다. Geo-R1은 극히 제한된 데이터 환경에서도 효과적으로 작동할 수 있도록 설계된 추론 중심의 강화 학습(RL) 기반의 방법입니다. 이 방식은 모델이 지시 표현을 분해하여 명확하고 해석 가능한 추론 체인을 생성하게 하여, 타겟 객체를 효과적으로 локалize할 수 있도록 돕습니다.

- **Technical Details**: Geo-R1은 장면 분류와 달리 지역 수준의 인식이 중요한 REU(Referring Expression Understanding) 작업에서 초점 맞춰집니다. 기존의 감독된 미세 조정(Supervised Fine-Tuning) 방식 대신, Geo-R1은 여러 추론 경로를 탐색하고, 각 예제당 더 풍부한 감독을 제공함으로써 소량의 샘플을 보다 효율적으로 활용합니다. 이 과정에서 REU를 위한 특정 보상 함수를 최적화하여, 각 작업(REC와 RES)에서 목표와 일치하는 학습을 촉진합니다.

- **Performance Highlights**: Geo-R1은 세 가지 신뢰할 수 있는 소수 샘플 벤치마크 모두에서 SFT 기반의 모델보다 뛰어난 성능을 보였습니다. 특히, 다양한 데이터셋 간에 더 견고한 일반화를 보여주어, 그 성능의 안정성을 입증했습니다. 또한 Geo-R1이 생성한 추론 흔적은 해석 가능성을 제공하며, 지역 및 의미적 단초를 활용하여 최종 로컬라이제이션에 기여합니다.



### No-Reference Image Contrast Assessment with Customized EfficientNet-B0 (https://arxiv.org/abs/2509.21967)
Comments:
          32 pages, 9 tables, 6 figures

- **What's New**: 이번 연구에서는 이미지 품질 평가에 있어서 중요한 요소인 대비(distortion)를 효과적으로 평가하기 위해 새로운 딥러닝 기반의 프레임워크를 제안하였습니다. 이 프레임워크는 세 가지 사전 훈련된 아키텍처, 즉 EfficientNet B0, ResNet18, MobileNetV2를 맞춤 구성하고 미세 조정(fine-tuning)하여 인지 관점의 Mean Opinion Score를 기반으로 합니다. 또한, 시암 네트워크(Siamese network)를 활용한 추가 모델이 개발되었으며, 이는 인지 대비 왜곡(perceptual contrast distortions)을 포착하는 능력이 제한적임을 보여주었습니다.

- **Technical Details**: 제안된 모델은 대비 인식 회귀 헤드(contrast-aware regression head)로 수정되어 CID2013과 CCID2014라는 두 개의 벤치마크 데이터셋에서 합성(synthetic) 및 진짜(authentic) 대비 왜곡 데이터를 위한 타겟된 데이터 증강(targeted data augmentations)을 이용하여 엔드 투 엔드(end-to-end)로 훈련됩니다. 성능 평가는 예측된 점수와 인간 평가 점수 간의 정렬을 평가하는 Pearson Linear Correlation Coefficient(PLCC)와 Spearman Rank Order Correlation Coefficient(SRCC)를 사용하여 평가됩니다.

- **Performance Highlights**: 제조된 EfficientNet B0 모델은 CCID2014에서 PLCC = 0.9286, SRCC = 0.9178의 성능을, CID2013에서는 PLCC = 0.9581, SRCC = 0.9369의 성능으로 전통적인 방법들 및 다른 딥러닝 기반 모델들을 초월하며 최첨단(performance state-of-the-art)을 달성했습니다. 이 결과들은 제안된 모델이 인지 대비 왜곡을 효과적으로 포착할 수 있는 강력함과 효율성을 강조합니다. 전반적으로, 대비 인식(adaptation) 경량 사전 훈련 네트워크를 기반으로 한 제안된 방법은 실시간 및 자원 제한 환경에서도 사용 가능한 고성능의 확장 가능한 솔루션을 제공함을 보여주었습니다.



### PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data (https://arxiv.org/abs/2509.21965)
- **What's New**: 이 논문에서 소개하는 PartSAM은 대규모 3D 데이터를 활용하여 훈련된 최초의 프롬프트 기반(part segmentation) 모델입니다. 기존의 접근 방식을 극복하고 다양한 3D 객체를 효과적으로 분할하는 것을 목표로 하고 있습니다. PartSAM은 일반적인 구조로서 Encoder-Decoder 아키텍처를 사용하고, 다차원적인 3D 파트에 대한 분할 능력을 강화합니다.

- **Technical Details**: PartSAM은 SAM(Segment Anything Model)의 설계를 기반으로 하여, 트리플레인의 이중 가지 인코더를 통해 공간적으로 구성된 토큰을 생성합니다. 이러한 구조로 인해 유연한 상호작용과 정확한 분할이 가능해지며, 다양한 데이터에서 내재적인 기하학을 더 잘 이해할 수 있습니다. 또한, PartSAM은 500만 개 이상의 3D 형상-부품 쌍을 활용한 대규모 주석 파이프라인을 도입하여 훈련 성능을 극대화합니다.

- **Performance Highlights**: PartSAM은 여러 기준점에서 기존 최첨단 방법들보다 90% 이상 뛰어난 성능을 자랑합니다. 이 모델은 제한된 프롬프트를 통해 전체 형상을 자동으로 세분화할 수 있는 능력을 보여줍니다. 실험 결과는 PartSAM이 3D 파트 이해를 위한 강력한 기초를 제공하며, 향후 연구에 널리 적용될 수 있는 가능성을 시사합니다.



### MultiCrafter: High-Fidelity Multi-Subject Generation via Spatially Disentangled Attention and Identity-Aware Reinforcement Learning (https://arxiv.org/abs/2509.21953)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 MultiCrafter라는 새로운 프레임워크를 제안하여 고충실도와 선호도 정렬된 다중 주제 이미지 생성을 수행합니다. 기존의 방법들은 단순 재구성 기법에 의존하여 특성 누수(attribute leakage) 문제를 초래하고 인간의 미적 선호에 맞지 않는 결과를 생성해왔습니다. MultiCrafter는 각 주제의 주의(attention) 영역을 명시적으로 분리하기 위한 위치 기반 감독(positional supervision)을 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: MultiCrafter는 각각의 주제를 위한 구분된 주의 영역을 학습하기 위해 Identity-Disentangled Attention Regularization을 도입합니다. 또한 Mixture-of-Experts (MoE) 아키텍처를 활용하여 다양한 시나리오에서 주제에 적합한 주의 계획을 가능하게 하고, 효율적인 전문가 튜닝(Efficient Adaptive Expert Tuning)을 통해 모델의 용량을 증가시킵니다. 마지막으로, 새로운 온라인 강화 학습 프레임워크를 설계하여 모델이 인간의 미적 선호에 맞춰 학습하도록 돕습니다.

- **Performance Highlights**: 실험 결과 다중 주제 생성에서 MultiCrafter는 기존 방법에 비해 주제 충실도를 크게 향상시켰으며, 인간의 선호에 더욱 잘 정렬된 결과를 보였습니다. 특히, Identity-Preserving Preference Optimization과 Multi-ID Alignment Reward의 도입을 통해 주제 일치 품질을 정밀하게 평가할 수 있었습니다. 이 접근법은 MultiCrafter가 다양한 시나리오에서 우수한 성능을 유지함과 동시에 훈련 안정성을 보장함을 입증합니다.



### Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach (https://arxiv.org/abs/2509.21950)
- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 다양한 작업에서 뛰어난 성능을 보여주고 있지만, 이미지에서 감정을 인식하는 능력에 대한 논의는 여전히 진행 중입니다. 본 연구에서는 기존 평가 방법의 한계를 지적하며, 이를 극복하기 위한 감정 진술 판단(Emotion Statement Judgment, ESJ) 작업을 제안합니다. 이를 통해 MLLMs의 맞춤형 시각 감정 평가는 더욱 용이해질 것으로 기대됩니다.

- **Technical Details**: 현재 감정 평가 접근 방식은 크게 감정 분류와 감정 해석으로 나눌 수 있으며, 두 가지 방법에서 전통적으로 활용되는 고정된 정답 구조가 다양한 해석을 제한한다는 문제점이 있습니다. 감정 인식의 주관성 때문에, 같은 이미지를 보더라도 개인마다 상이한 감정이 유발될 수 있습니다. 저자들은 감정을 더 세분화된 분류 체계로 평가하기 위해 INSETS 파이프라인을 설계하였고, 이를 통해 최소한의 인간 개입으로 감정 중심의 다양한 진술을 생성할 수 있습니다.

- **Performance Highlights**: 본 연구는 MLLMs의 감정 해석 및 문맥 기반 감정 판단에서 뛰어난 성능을 보였으나, 인간과 비교했을 때 여전히 성능 격차가 존재함을 보여줍니다. 이러한 성능 검증을 바탕으로, 저자들은 MLLMs의 정서적 지능을 발전시키기 위한 잠재적인 방향성을 제시하고 있습니다. 시스템적으로 정리된 평가 결과는 향후 연구에서 중요한 참고자료가 될 것입니다.



### SemanticControl: A Training-Free Approach for Handling Loosely Aligned Visual Conditions in ControlN (https://arxiv.org/abs/2509.21938)
Comments:
          BMVC 2025

- **What's New**: 이번 연구에서는 SemanticControl이라는 새로운 방법을 제안하여 기존의 ControlNet에서 발생하는 텍스트 프롬프트와 시각적 조건의 불일치를 해결하고자 합니다. 이 방법은 훈련 없이도 비선형적이지만 의미적으로 관련이 있는 시각적 조건을 효과적으로 활용할 수 있습니다. 또한, 불일치가 있는 경우에도 텍스트 가이드를 강화하면서 시각적 조건의 영향을 조절하여 생성 품질을 향상시킵니다.

- **Technical Details**: SemanticControl의 핵심 아이디어는 먼저 대체 프롬프트를 기반으로 보조 노이즈 제거 과정을 실행하여 유용한 주의 마스크(attention masks)를 추출하는 것입니다. 이후 이러한 마스크를 사용하여 텍스트 프롬프트의 노이즈 제거 과정에서 텍스트의 의미에 맞는 부분을 강화합니다. 이를 통해, SemanticControl은 기존의 ControlNet보다 불일치한 조건 하에서도 더욱 높은 품질의 이미지를 생성할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, SemanticControl은 다양한 입력 조건에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 특히 깊이 맵, 엣지 맵 및 인간 골격과 같은 다양한 시각적 조건 하에서도 효과적임이 입증되었습니다. 또한, 인간 평가에서도 SemanticControl이 더 높은 선호도를 얻어 내며 우수한 결과를 나타냈습니다.



### DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation (https://arxiv.org/abs/2509.21930)
Comments:
          Accepted as a poster in NeurIPS 2025

- **What's New**: 이번 연구에서 우리는 DynaNav라고 하는 동적 시각 내비게이션 프레임워크를 제안합니다. DynaNav는 장면의 복잡성에 따라 특성(feature) 및 레이어(layer) 선택을 조정하여 계산 효율성을 높이고 해석 가능성을 개선합니다. 또한 Bayesian Optimization을 통해 최적의 조기 탈출 기준을 결정하는 조기 탈출 메커니즘을 통합하여 계산 비용을 줄입니다.

- **Technical Details**: DynaNav는 EfficientNet-B0을 기반으로 RGB 이미지 시퀀스에서 특징을 추출합니다. 이 모델은 특성 선택 모듈을 도입하여 Transformer 해석기에서 특성 처리를 위한 마스크를 생성하며, 이를 통해 계산의 일부만 수행합니다. 또한, Bayesian Optimization을 이용하여 적절한 조기 탈출 기준을 결정하여 계산 오버헤드를 최소화합니다.

- **Performance Highlights**: 실험 결과, DynaNav는 ViNT에 비해 FLOPs를 2.26배 줄였으며, 추론 시간을 42.3% 단축하고 메모리 사용량을 32.8% 줄였음에도 불구하고 내비게이션 성능을 개선하는 데 성공했습니다. 이는 DynaNav가 동적 네트워크 메커니즘을 도입한 최초의 사례이며, 전체적으로 더 높은 효율성을 보장합니다.



### SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Referenc (https://arxiv.org/abs/2509.21927)
Comments:
          Accepted as a poster in NeurIPS 2025

- **What's New**: 최근 6D 포즈 추정 방법들이 주목할 만한 성과를 기록하고 있으나, 여전히 실용적인 한계에 직면하고 있습니다. 특히 많은 방법들이 센서 깊이에 의존하고 있어 투명하거나 매우 반사된 표면과 같은 어려운 환경에서는 실패할 수 있습니다. 이에 우리 연구진은 단 하나의 RGB 이미지만을 사용하여 복잡한 깊이 센서나 다중 뷰 이미지 수집 없이도 견고하고 효율적인 SingRef6D 파이프라인을 제안합니다.

- **Technical Details**: SingRef6D는 깊이 인식과 포즈 해결 단계에서 각각의 도전을 독립적으로 해결합니다. 먼저, DPAv2를 기반으로 한 새로운 파인튜닝 접근 방식을 개발했습니다. 이 접근법은 transformer 레이어에서 특성의 가중치를 동적으로 조정하여 인간의 공간 인식 메커니즘을 모방합니다. 또한 LoFTR를 기반으로 한 깊이 인식 매칭 모듈을 통해 RGB와 깊이 정보를 통합하여 공간적 관계를 효과적으로 처리합니다.

- **Performance Highlights**: 우리는 REAL275, ClearPose, Toyota-Light 데이터셋에서 포즈 추정 성능을 평가한 결과, SingRef6D가 최신 기술을 능가하고 평균 리콜 점수에서 6.1% 향상을 달성했습니다. 투명한 객체에 대해서도 ClearPose 데이터셋에서 31.23%에서 54.30%로 정확도를 향상시키며 도전적인 표면 조건에서도 뛰어난 성능을 입증하였습니다.



### PANICL: Mitigating Over-Reliance on Single Prompt in Visual In-Context Learning (https://arxiv.org/abs/2509.21926)
Comments:
          21 pages, 12 figures

- **What's New**: 본 논문은 Visual In-Context Learning (VICL)에서 발생하는 문제를 해결하기 위해 PAtch-based k-Nearest neighbor visual In-Context Learning (PANICL)이라는 새로운 프레임워크를 제안합니다. 기존의 VICL 방식은 단일 in-context pair에 과도하게 의존하여 편향된 예측을 초래했습니다. PANICL은 여러 in-context pairs를 활용하여 편향을 완화하고, 추가적인 훈련 없이 다양한 시각적 작업의 성능을 향상시키는 방식으로 설계되었습니다.

- **Technical Details**: PANICL은 MAE-VQGAN을 기반으로 하여, 입력 이미지에 대한 예측이 VQGAN 코드북의 토큰에 의해 이루어집니다. 이 때, 각 토큰의 assignment scores가 단일 in-context pair에 지나치게 의존하는 경향을 보이기에, 이 assignment scores를 query의 k-nearest neighbors로부터 오는 다른 in-context pairs와 더불어 스무딩하여 적용합니다. 이러한 방식은 모델에 대한 의존성을 줄이고, 보다 안정적인 예측을 가능하게 합니다.

- **Performance Highlights**: PANICL은 foreground segmentation, single object detection, colorization, multi-object segmentation, keypoint detection 등 다양한 작업에서 뛰어난 성능을 보여주었습니다. 실험 결과, PANICL은 강력한 성능 기준을 초과하며, COCO에서 Pascal로의 데이터셋 수준 변화와 같은 도메인 변동에 강한 강건성을 보입니다. PANICL은 SegGPT, Painter, LVM과 같은 다른 VICL 모델에 대해서도 높은 일반화 성능을 보여주어 그 다양성과 광범위한 적응성을 입증했습니다.



### Spatial Reasoning in Foundation Models: Benchmarking Object-Centric Spatial Understanding (https://arxiv.org/abs/2509.21922)
Comments:
          4 pages, NeurIPS Workshop SpaVLE

- **What's New**: 이번 연구에서는 비전 모델과 비전-언어 모델(VLM)에서 객체 중심의 공간 추론(object-centric spatial reasoning)을 위한 체계적인 벤치마크를 제시합니다. 다양한 최신 비전 모델과 VLM을 평가하여, 공간 남획(spatial localization), 공간 추론(spatial reasoning), 다운스트림 검색(tasks)에서의 성능을 비교합니다. 이 연구는 물체의 상대적 위치 파악, 그룹화, 깊이 이해를 포함한 장면 이해의 중요성을 강조합니다.

- **Technical Details**: 우리는 아홉 가지 가구 카테고리에 걸쳐 11,000장의 이미지를 포함한 합성 데이터셋을 구축하였습니다. 각 이미지에는 제품의 3D 렌더링을 기반으로 한 프론탈 뷰와 각도 있는 쿼리 이미지가 포함됩니다. 14개의 모델을 평가하였고, 태스크 특화 비전 모델(예: GroundingDINO)과 일반-purpose VLM(예: GPT-4o)을 분류하여 실험을 진행했습니다.

- **Performance Highlights**: 비교 결과, 태스크 특화 비전 모델들이 일반-purpose VLM보다 명확하게 우수한 성능을 보였습니다. GroundingDINO가 공간 남획 및 이미지 기반 검색에서 가장 높은 성능을 기록했으며, VLM 중에서는 InternVL과 LLaVA 변종이 가장 경쟁력 있었습니다. 그러나 VLM들은 정교한 공간 컨텍스트가 요구되는 경우 consistently 저조한 성능을 보였습니다.



### Multi-View Crowd Counting With Self-Supervised Learning (https://arxiv.org/abs/2509.21918)
- **What's New**: 이번 연구에서는 대규모 주석 데이터셋에 대한 의존성을 줄이기 위해 자가 감독 학습(self-supervised learning, SSL)을 기반으로 하는 새로운 다중 뷰 카운팅(multi-view counting, MVC) 프레임워크인 SSLCounter를 제안합니다. SSLCounter는 신경 볼류메트릭 렌더링(neural volumetric rendering)을 활용하여 장면에 대한 암시적 표현을 학습하고 있으며, 이로써 복잡한 뷰 종속 방식의 2D 프로젝션을 재구성하는 능력을 제공합니다. 이 방법은 기존 프레임워크에 원활하게 통합될 수 있으며, 오직 70%의 학습 데이터를 사용하여도 경쟁력 있는 성능을 보임을 입증하였습니다.

- **Technical Details**: SSLCounter는 다중 뷰 이미지를 입력으로 받아 장면 수준의 볼류메트릭 표현과 이미지 수준의 밀도 맵을 동시에 추출하는 인코더와, 이 볼류메트릭 특징을 통해 이미지 수준 밀도 맵, 장면 수준 밀도 볼륨, 깊이 맵, RGB 이미지를 디코드하는 미분 가능 신경 디코더로 구성됩니다. 이 방법에서 레이 트레이싱(ray tracing)을 이용하여 깊이 맵을 추정하고, 각 레이에 대한 색상, 깊이, 밀도를 예측합니다. 이 과정에서 SDF(signed distance field)를 사용하여 장면의 기하학적 구조를 표현하며, 다양한 손실 함수와 자가 감독 전략을 통해 네트워크의 성능을 개선합니다.

- **Performance Highlights**: SSLCounter는 기존 MVC 벤치마크에서 최첨단 성능을 보여주었으며, 특히 데이터 부족 문제를 효과적으로 해결하는 데 기여하였습니다. 실험 결과 SSLCounter는 70%의 학습 데이터만을 사용하여도 우수한 데이터 효율성을 발휘하며, 실세계 응용 프로그램에서도 실제로 적용 가능한 가능성을 입증했습니다. 이 연구는 자가 감독 학습을 통해 데이터 주석의 필요성을 줄이는 새로운 방향성을 제시하고 있어 주목받고 있습니다.



### Taming Flow-based I2V Models for Creative Video Editing (https://arxiv.org/abs/2509.21917)
- **What's New**: 최근 이미지 편집 기술이 크게 발전하였으나, 동영상 편집은 여전히 사용자 의도에 맞게 영상을 조작하는 것이 도전적입니다. 기존의 이미지 조건 동영상 편집 방식은 모델 특정 설계와 많은 최적화를 요구하여 최신 이미지-투-비디오(I2V) 모델의 활용이 제한됩니다. 본 연구에서는 이러한 문제를 해결하기 위해, 계산 비용을 최소화하면서 범용적으로 적용 가능한 Inversion-Free 메서드인 IF-V2V를 제안합니다.

- **Technical Details**: IF-V2V는 Vector Field Rectification with Sample Deviation(VFR-SD) 기법을 통해 소스 영상의 정보를 분산 과정에 통합합니다. 이 방법은 대상 위상 미분 방정식(ODE)을 해결하는 데 사용되는 벡터 필드에 편차 항을 추가하여 소스 비디오 샘플을 맞추는 경향을 생성합니다. 또한 구조 및 운동 보존 초기화(SMPI)를 통해 소스 비디오의 운동 정보를 활용하여 구조 정보가 포함된 동적 연관 잡음을 생성하고, Deviation Caching(D-Cache) 메커니즘을 통해 추가적인 계산 비용을 최소화합니다.

- **Performance Highlights**: IF-V2V는 기존 접근 방식에 비해 우수한 시각적 품질과 일관성을 자랑하며, 추가적인 계산 비용을 줄이면서도 다양한 편집 패러다임에서 일관되게 성능을 향상시킵니다. 이 방법은 다양한 I2V 모델과 이미지 편집 기술의 가능성을 효과적으로 결합하여 창의적인 비디오 편집 작업을 지원하는 경량 솔루션으로서 강력한 잠재력을 보여줍니다.



### Enhancing Vehicle Detection under Adverse Weather Conditions with Contrastive Learning (https://arxiv.org/abs/2509.21916)
- **What's New**: 본 연구에서는 Nordic 지역에서 UAV(무인 항공기) 이미지를 사용하여 차량 탐지를 위한 새로운 sideload-CL-adaptation 프레임워크를 제안합니다. 이 프레임워크는 주석이 없는 데이터를 활용하여 경량 모델의 차량 탐지 성능을 향상시키는 데 초점을 맞춥니다. 특히, 비주석 데이터를 기반으로 대조 학습을 통해 CNN 기반의 표현 추출기를 학습하고, 이를 YOLO11n 백본에 결합하여 성능을 극대화합니다.

- **Technical Details**: 제안된 sideload-CL-adaptation 모델은 YOLO11n을 기반으로 하며, 비주석 데이터를 통해 훈련된 CNN feature extractor를 이용합니다. 대조 학습(Contrastive Learning)을 통해 다양한 비주석 데이터에서 얻어진 특성 추출기를 구축하고, 동적 융합 방식(dynamic fusion approach)을 채택하여 모델이 입력 데이터와 작업 목표에 따라 각 특성의 중요도를 조정할 수 있도록 합니다. 이로써 각 데이터의 도메인에 강건한 특성을 유지하며 성능을 높입니다.

- **Performance Highlights**: NVD(Nordic Vehicle Dataset) 데이터셋에서 제안된 모델은 mAP50 기준으로 9.5% 향상된 탐지 성능을 보여줍니다. 또한, 훈련, 테스트 및 검증이 전혀 다른 비디오에서 수행될 때도 3.8%의 성능 개선을 달성하였습니다. 이러한 결과는 비주석 데이터의 활용이 차량 탐지 개선에 기여할 수 있다는 것을 잘 보여줍니다.



### TDEdit: A Unified Diffusion Framework for Text-Drag Guided Image Manipulation (https://arxiv.org/abs/2509.21905)
- **What's New**: 이번 논문은 텍스트와 드래그(interaction)를 통한 이미지 편집의 통합 제어 방식을 탐구합니다. 기존에 텍스트 기반 및 드래그 기반 편집 방식이 각각의 강점을 가지고 있었으나, 통합된 솔루션이 없었습니다. 이를 해결하기 위해 제안하는 통합된 확산(diffusion) 기반 프레임워크는 Point-Cloud Deterministic Drag와 Drag-Text Guided Denoising이라는 두 가지 혁신적 접근 방식을 포함합니다.

- **Technical Details**: TDEdit는 드래그 및 텍스트 기반 제어를 동시에 처리할 수 있는 통합 편집 프레임워크입니다. 이 프레임워크는 동적 안내 균형 조절을 위해 Drag-Text Guided Denoising(DTGD) 메커니즘을 사용하며, Point-Cloud Deterministic Drag(PCDD)를 통해 3D 특징 맵핑을 통한 공간 제어를 강화합니다. DTGD의 세 가지 분기 구조는 소스, 참조 및 목표 세부 정보를 효과적으로 통합하여 노이즈 감소 과정에서 정확한 제어를 가능하게 합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 기존의 텍스트 전용 또는 드래그 전용 방법의 성능을 초월하는 뛰어난 편집 품질을 보여줍니다. 제안된 접근 방식은 텍스트, 드래그, 또는 이 두 가지 조건을 조합하여 작업을 수행할 수 있는 유연성을 제공합니다. 다양한 정량적 및 정성적 실험을 통해 고충실도의 공동 편집을 성공적으로 달성했음을 입증하였습니다.



### LG-CD: Enhancing Language-Guided Change Detection through SAM2 Adaptation (https://arxiv.org/abs/2509.21894)
Comments:
          *Corresponding authors: Min Zhu (this http URL@scu.this http URL) and Junlong Cheng (jlcheng@scu.this http URL)

- **What's New**: 이 논문은 다중 모달 데이터(멀티모달 데이터)의 풍부한 의미 정보를 활용하여 원거리 영상 변화 탐지(Remote Sensing Change Detection, RSCD) 성능을 높이기 위한 새로운 모델인 언어 유도 변화 탐지 모델(Language-Guided Change Detection, LG-CD)을 제안합니다. LG-CD는 자연어 프롬프트를 통해 네트워크의 주의를 주요 영역으로 유도하여 변화 탐지의 정확성과 강건성을 크게 향상시킵니다. 이 모델은 특히 비시간적 원거리 영상에서 고해상도에서 저해상도로의 다중 스케일 피라미드 특성을 캡처하는 시각 기초 모델(SAM2)을 사용합니다.

- **Technical Details**: LG-CD의 전체 구조는 SAM2 인코더가 두 개의 시간적 원거리 영상의 다중 스케일 특성을 추출하고, 텍스트 주의 모듈이 텍스트와 이미지 특징을 정렬하여 탐지 영역에 집중하도록 돕는 과정을 포함합니다. 이 과정에서 경량 어댑터 레이어를 통해 인코더의 출력 특성들을 미세 조정하고 결합하여 글로벌 특징 맵을 생성합니다. 마지막으로, 텍스트 융합 주의 모듈(Text Fusion Attention Module, TFAM)과 시각-의미 융합 디코더(Vision-Semantic Fusion Decoder)를 통해 시각 및 의미 정보를 통합하여 고정밀 변화 탐지 마스크를 생성합니다.

- **Performance Highlights**: 실험 결과, LG-CD 모델은 LEVIR-CD, WHU-CD 및 SYSU-CD 세 가지 데이터 세트에서 최신 변화 탐지 방법보다 일관되게 우수한 성능을 발휘했습니다. LG-CD는 시각적 특징과 텍스트적 의미 정보를 결합하여 모델의 의미 이해 능력을 높이고 탐지 정확도 및 일반화 능력을 상당히 향상시킵니다. 이를 통해 자연어가 제공하는 맥락 정보를 활용하여 복잡한 장면과 변화 패턴을 더 잘 이해하고, 다양한 유형의 대상에 대해 강력한 탐지를 달성할 수 있음을 입증했습니다.



### Syncphony: Synchronized Audio-to-Video Generation with Diffusion Transformers (https://arxiv.org/abs/2509.21893)
Comments:
          Project page: this https URL

- **What's New**: Syncphony는 텍스트(텍스트-비디오) 및 이미지(이미지-비디오) 생성의 한계를 극복하고 오디오를 이용한 비디오 생성의 새로운 접근을 제시합니다. 기존의 오디오-비디오(A2V) 모델들이 세밀한 동기화에 실패했던 문제를 해결하기 위해, Syncphony는 사전 훈련된 비디오 백본을 바탕으로 두 가지 주요 구성 요소인 Motion-aware Loss와 Audio Sync Guidance를 도입합니다. 이를 통해 비디오와 오디오 간의 보다 정밀한 타이밍 조정이 가능해집니다.

- **Technical Details**: Syncphony는 380x640 해상도와 24fps로 비디오를 생성하며, 오디오 입력과 동기화됩니다. 모델 구조는 DiT 아키텍처를 기반으로 하여 텍스트 비디오 및 오디오 크로스 어텐션을 통합하여 생성 과정에서 오디오 피처를 직접 주입합니다. 훈련 과정에서는 큰 모션이 있는 영역에 가중치를 두어 플로우 매칭 손실을 수정하고, 샘플링 과정에서는 오디오 구동 생성을 강화하는 새로운 동기화 가이드를 도입합니다.

- **Performance Highlights**: Syncphony는 AVSync15 및 Greatest Hits 데이터셋에서 실험한 결과, 기존 방법들보다 동기화 정확성과 시각적 품질 모두에서 우수한 성능을 보였습니다. 특히 CycleSync라는 새로운 동기화 메트릭을 도입하여 생성된 비디오에서 원본 오디오를 재구성하는데 필요한 모션 신호의 양을 평가하며, 이는 고프레임 속도 비디오 생성에 적합합니다. 연구 결과는 오픈 소스로 제공될 예정으로, 후속 연구에 기여할 것으로 기대됩니다.



### Drag4D: Align Your Motion with Text-Driven 3D Scene Generation (https://arxiv.org/abs/2509.21888)
Comments:
          version 1

- **What's New**: 이번 논문에서는 Drag4D라는 새로운 상호작용 프레임워크를 소개합니다. 이 프레임워크는 사용자가 단일 이미지에서 생성된 3D 객체의 경로를 설정하고, 이를 고품질 3D 배경에 원활하게 통합할 수 있도록 설계되었습니다. Drag4D 파이프라인은 3단계로 구성되며, 각 단계를 통해 최적의 3D 장면 생성 및 객체 애니메이션을 가능하게 합니다.

- **Technical Details**: Drag4D의 첫 번째 단계에서는 2D Gaussian Splatting을 활용하여 파노라마 이미지를 생성하고, 이를 통해 밀집하고 시각적으로 완전한 3D 재구성을 이루어냅니다. 두 번째 단계에서는 참조 이미지에서 목표 객체를 추출하고, 이를 3D 메쉬 형태로 생성하여 3D 장면에 조화롭게 합성합니다. 마지막 단계에서는 사용자 정의 3D 궤적을 따라 객체를 시간적으로 애니메이션하여, 모션 일치성을 보장합니다.

- **Performance Highlights**: Drag4D의 통합 아키텍처는 각 단계에서의 평가를 통해 효과성을 입증합니다. 특히, 최종 결과에서 사용자가 제어하는 객체의 움직임과 고품질 3D 배경 간의 조화로운 정렬이 강조됩니다. 또한, 새로운 사용자 정의 데이터셋인 Drag4D-30을 통해 기존 방법들과 비교한 성능 개선이 확인되었습니다.



### StableDub: Taming Diffusion Prior for Generalized and Efficient Visual Dubbing (https://arxiv.org/abs/2509.21887)
- **What's New**: 최근 비주얼 더빙(visual dubbing) 분야는 음성과 시각을 효과적으로 동기화하는 기술로 큰 발전을 이루었습니다. 그러나, 기존 방법들은 화자의 특정 입술 습관을 포착하지 못하고, 장애물에 의한 시각적인 아티팩트(visual artifacts) 문제로 인해 실질적인 응용에 한계가 있습니다. 본 논문에서는 이러한 문제를 해결하는 새로운 프레임워크인 StableDub을 제안하여, 입술 습관을 모델링하고 장애물에 강한 합성을 실시하는 방법론을 제시합니다.

- **Technical Details**: StableDub은 U-Net 기반의 Stable Diffusion 구조를 기반으로 하여 만들었으며, 음성과 입술 습관을 조화롭게 모델링합니다. 특히, 이 프레임워크는 이중 모달(audio-visual) 피처를 통해 음성-입술 동기화를 개선하고, 특정 화자의 입술 습관을 학습할 수 있도록 설계되었습니다. 또한 장애물 인식 훈련 전략을 도입하여, 입체적인 재구성을 가능하게 함으로써 기존 기술의 한계를 극복합니다.

- **Performance Highlights**: 다양한 실험을 통해 StableDub은 입술 습관 유사성과 장애물 강인성 분야에서 최고의 성능을 발휘하는 것이 입증되었습니다. 또한, 오디오-입술 동기화, 비디오 품질, 해상도 일관성 측면에서도 다른 방법들을 초월합니다. 본 연구는 비주얼 더빙 기술의 적용 가능성을 넓히고 있으며, 데모 비디오를 통해 그 성능을 확인할 수 있습니다.



### Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization (https://arxiv.org/abs/2509.21871)
- **What's New**: 이번 논문에서는 Aes-R1이라는 새로운 체계를 제안하여 이미지 미적 평가(IAA)를 위한 강력한 미적 추론 프레임워크를 구축했습니다. 기존의 다중 형태 대형 언어 모델(MLLM)들이 미적 판단을 수행하는 데 어려움을 겪는 이유로, 데이터 부족과 주관적인 미적 판단의 본질을 언급했습니다. Aes-R1은 강화 학습을 통해 이 문제를 해결하며, 더 나아가 설명 가능한 미적 점수를 생성할 수 있도록 지원합니다.

- **Technical Details**: Aes-R1는 AesCoT라는 파이프라인을 통해 고품질의 미적 추론 데이터를 생성하고 필터링합니다. 이 방법은 구조화된 설명 생성을 모델에 가르친 후, Relative-Absolute Policy Optimization (RAPO)이라는 새로운 강화 학습 알고리즘을 사용하여 미적 평가 점수와 상대적 순위를 최적화합니다. 이를 통해 모델은 이미지의 미적 점수를 정확히 평가할 수 있는 능력을 얻게 됩니다.

- **Performance Highlights**: Aes-R1은 기존 최첨단 모델 대비 47.9% 및 34.8%의 PLCC/SRCC 향상을 이루며, 단 15K의 학습 데이터로도 우수한 성능을 보여줍니다. 다양한 IAA 벤치마크에서 최첨단 성능을 달성하였고, 제한된 지도식 아래에서도 견고한 일반화를 지원하는데 필요한 분석을 수행했습니다. 논문에 제시된 실험 결과와 데이터 파이프라인 AesCoT의 개발은 향후 IAA 연구에 큰 기여를 할 것입니다.



### Deepfakes: we need to re-think the concept of "real" images (https://arxiv.org/abs/2509.21864)
- **What's New**: 본 논문은 현재 이미지 생성 모델의 보편적 사용이 범죄 및 사회적 부작용을 야기할 수 있다는 우려를 제기합니다. 머신러닝 커뮤니티는 '가짜' 이미지(예: 전적으로 생성되거나 부분적으로 조작된 이미지)를 탐지하기 위한 알고리즘적 해결책을 제안하고 있지만, '진짜' 이미지의 명확한 정의와 데이터 수집이 충분히 다뤄지지 않고 있음을 강조합니다. '진짜 이미지'의 정의에 대한 재고가 필요하며 이를 위해서 새로운 벤치마크 데이터세트가 필요하다는 입장을 취합니다.

- **Technical Details**: 현재 대부분의 '진짜' 이미지 데이터셋은 오래된 저해상도 이미지를 기반으로 하며, 현대 스마트폰 카메라에서 촬영된 이미지의 개선 알고리즘은 충분히 반영되지 않았습니다. 스마트폰 카메라는 다양한 센서로부터 처리된 여러 입력값을 통해 이미지를 생성하며, 이는 사실상 이미지를 '촬영'하는 것이 아니라 '계산'하는 과정입니다. 이를 감안할 때, 현재의 가짜 탐지 알고리즘의 성능과 데이터셋이 이러한 기술 발달을 적절히 반영하고 있는지 의문이 제기됩니다.

- **Performance Highlights**: 현재의 가짜 탐지 알고리즘은 성능은 높을 수 있으나, 알려지지 않은 생성 모델에 대한 일반화에 어려움을 겪고 있으며, 이미지 증대 또는 전처리에 대한 강건성이 부족합니다. 여러 데이터세트가 JPEG 압축에 편향되어 있어 탐지기의 성능에 영향을 미치며, 대다수의 이미지가 낮은 해상도로 이루어져 있어 탐지기의 일반화를 저해할 수 있습니다. 따라서, 현대 모바일 카메라에서 촬영된 최신 데이터로 가짜 탐지기를 훈련하고 평가해야 한다는 점이 강조됩니다.



### SRHand: Super-Resolving Hand Images and 3D Shapes via View/Pose-aware Neural Image Representations and Explicit 3D Meshes (https://arxiv.org/abs/2509.21859)
Comments:
          10 pages, 6 figures

- **What's New**: SRHand는 저해상도 이미지에서 3D 손 기하학과 텍스처 이미지를 복원하는 혁신적인 방법으로, 손 메쉬와 암묵적 이미지 표현(implicit image representation)을 결합하여 높은 충실도의 아바타 모델링을 가능하게 합니다. 기존 연구는 저해상도 이미지에서의 정밀한 손 복원이 어려우며, SRHand는 다양한 뷰와 포즈를 유지하면서 아래와 같은 문제를 해결합니다: 흐릿하고 불안정한 형상, 시각적으로 매끄럽지 않은 텍스처.

- **Technical Details**: SRHand는 기하학적 정보를 반영한 암묵적 이미지 함수(Geometric-aware Implicit Image Function, GIIF)를 도입하여, 제한적인 해상도에서 손 이미지를 임의의 비율로 향상시킵니다. 이 과정에서 3D 손 모양을 명시적으로 최적화하여 손의 회전과 포즈에 따른 일관성을 유지하고, 결과적으로 손의 주름 및 손톱과 같은 세밀한 요소들을 복원합니다. 이는 머신러닝을 활용한 방법으로, 경량의 이미지 슈퍼 해상도 모듈과 손 복원 모듈을 통합하여 수행됩니다.

- **Performance Highlights**: SRHand는 InterHand2.6M 및 Goliath 데이터셋 실험에서 기존의 아바타 복원 방법 및 손 복원 기법보다 수량적, 질적으로 뛰어난 성과를 보입니다. 이 방법은 흐릿함을 제거하고 다양한 포즈와 시점에서의 정확한 3D 구조를 유지하며, 실시간 상호작용이 가능한 VR/AR 애플리케이션에 필수적인 세부 복원을 가능하게 합니다. 실험적 결과에서는 비록 저해상도 입력 이미지에서 시작하더라도, 손 아바타의 질감과 기하학적 품질이 크게 향상됨을 나타냅니다.



### Dynamic Novel View Synthesis in High Dynamic Rang (https://arxiv.org/abs/2509.21853)
- **What's New**: 최근 발표된 논문은 High Dynamic Range Dynamic Novel View Synthesis (HDR DNVS)라는 새로운 문제를 제안합니다. 이는 저 동적 범위(LDR) 영상으로부터 시간적으로 일관된 HDR 라디언스 필드를 복원하는 것을 목표로 하고 있습니다. 기존의 HDR NVS 방법이 정적인 장면에 국한된 반면, 이 연구는 동적 요소가 포함된 실제 환경을 다루고 있습니다.

- **Technical Details**: HDR DNVS는 4D 라디언스 필드를 공동으로 모델링해야 하며, HDR과 LDR 간의 복잡한 변환을 요구합니다. 이를 위해 소개된 HDR-4DGS는 동적 톤 맵핑(dynamic tone-mapping) 모듈을 가지고 있어, 시간 축에 따라 적응적으로 HDR과 LDR 영역을 연결합니다. 이 모델은 Gaussian Splatting을 기반으로 하여 시간과 공간에서의 라디언스 일관성을 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, HDR-4DGS는 기존의 최첨단 방법들을 초월하는 정량적 성능과 시각적 충실도를 달성했습니다. 이 연구는 두 가지 신규 벤치마크 데이터셋인 HDR-4D-Syn과 HDR-4D-Real을 소개하여 HDR DNVS 방법의 rigor한 평가를 가능하게 합니다. 향후 소스 코드는 공개될 예정입니다.



### A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design (https://arxiv.org/abs/2509.21845)
- **What's New**: 본 논문에서는 Transformer 기반의 다중 연결 질문 응답(Multi-Hop Question Answering, QA) 시스템을 위한 검색 전략의 종합적인 평가를 제공합니다. 특히, 우리는 코사인 유사도(cosine similarity), 최대 한계 관련성(Maximal Marginal Relevance, MMR), 그리고 조밀한 임베딩(dense embeddings)과 어휘적(overlapping lexical) 기법을 통합한 하이브리드 방법을 비교합니다. 이 연구는 검색 최적화를 위해 EfficientRAG 파이프라인을 조정하며, 토큰 라벨링(token labeling)과 반복 정제를 통해 검색을 더욱 향상시킵니다.

- **Technical Details**: 본 연구는 다중 연결 QA 작업에 초점을 맞추어 Retrieval-Augmented Generation (RAG) 방법론을 사용하고 있습니다. 하이브리드 검색 방법은 조밀 벡터 유사성(dense vector similarity)과 키워드 기반의 어휘 일치를 결합하여 문서의 관련성과 다양성을 균형있게 다루는 독창적인 방식입니다. 또한, 복잡한 쿼리를 보다 관리 가능한 하위 쿼리로 분해하는 쿼리 최적화 기법을 탐구하여 다중 연결 추론 능력을 향상시킵니다.

- **Performance Highlights**: HotpotQA 데이터셋을 통한 실험 결과, 하이브리드 접근 방식이 기준 방법들에 비해 상관 개선을 50% 그리고 F1 점수에서 47%의 개선을 보여주었습니다. 오류 분석에 따르면, 하이브리드 검색 방식이 개체 조회(entity recall)와 증거 보완성을 개선하는 데 효과적입니다. 전체적으로 하이브리드 검색으로 강화된 생성 방법은 정확성, 효율성 및 해석 가능성을 모두 고려한 실용적인 제로샷 솔루션을 제공할 수 있음을 보여줍니다.



### DiTraj: training-free trajectory control for video diffusion transformer (https://arxiv.org/abs/2509.21839)
- **What's New**: 새로운 Diffusion Transformers (DiT) 기반 비디오 생성 모델인 DiTraj를 소개합니다. DiTraj는 텍스트-비디오 생성에서의 궤적 제어를 위해 특별히 설계된 교육이 필요 없는 프레임워크입니다. 이 모델은 객체의 움직임을 제어하기 위해 전경-배경 분리 가이드를 적용하여 생성 과정에서 사용자 제공 프롬프트를 효과적으로 활용합니다.

- **Technical Details**: DiTraj는 Large Language Model (LLM)을 사용하여 사용자 프롬프트를 전경과 배경 프롬프트로 변환합니다. 또, 3D 풀 어텐션과 위치 임베딩 간의 밀접한 관계를 분석하여, inter-frame Spatial-Temporal Decoupled 3D-RoPE (STD-RoPE)라는 혁신적인 방법을 제안합니다. STD-RoPE는 전경 토큰의 위치 임베딩을 조정하여 서로 다른 프레임 간의 어텐션을 개선하고, 객체의 궤적 제어를 강화합니다.

- **Performance Highlights**: 실험 결과, DiTraj는 기존 모델들에 비해 비디오 품질과 궤적 제어 가능성 모두에서 우수한 성능을 보여줍니다. 특히, DiTraj는 학습 과정이 필요 없으며, 대부분의 DiT 기반 비디오 생성 모델에 쉽게 적응할 수 있습니다. 이러한 접근 방식은 사용자가 제공한 텍스트 설명에 따라 정확한 비디오 생성을 가능하게 합니다.



### MoWM: Mixture-of-World-Models for Embodied Planning via Latent-to-Pixel Feature Modulation (https://arxiv.org/abs/2509.21797)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문에서는 MoWM(Mixture-of-World-Model)이라는 하이브리드 월드 모델 프레임워크를 제안하여 로봇이 시각적 관찰(data)과 언어 지침(instructions)에서 정밀한 행동(action)을 생성하도록 지원합니다. MoWM은 디퓨전 기반 비디오 생성 모델의 시각적 정보와 잠재적(latent) 세계 모델의 모션-웨어(motion-aware) 표현을 융합하여, 행동 디코딩을 위한 정보가 풍부한 시각적 세부사항을 강조합니다. 이 방법은 CALVIN 벤치마크에서 최신 기술(task success rate)과 우수한 일반화(generalization) 성능을 발휘하는 것으로 평가되었습니다.

- **Technical Details**: MoWM은 두 단계로 구성됩니다. 첫 번째 단계에서는 비디오 디퓨전 기반의 픽셀 월드 모델과 임베디드 조작 데이터에 대한 잠재적 월드 모델을 개별적으로 학습합니다. 두 번째 단계에서는 두 모델의 표현을 결합하여 모션-웨어 저수준(low-level) 시각적 표현을 생성합니다. 이러한 설정을 통해 행동 디코딩에 가장 관련 있는 시각 정보를 주목할 수 있게 되며, 이는 역동적 모델(inverse dynamics model)을 통해 최종 행동을 디코딩하는 데 사용됩니다.

- **Performance Highlights**: MoWM은 여러 벤치마크에 대해 광범위한 실험을 수행하였고, 그 결과 최신 기술과 비교하여 우수한 작업 성공률(task success rate)을 달성했습니다. 본 논문은 픽셀 수준과 잠재 수준의 시각적 특징들에 대한 깊이 있는 분석도 제공하는데, 이는 실제 응용에서 모델 선택에 대한 유용한 지침을 제공합니다. 이를 통해 하이브리드 월드 모델링의 미래 연구에 중요한 통찰력을 제시함으로써 로봇 행동 계획 분야에 기여하고자 합니다.



### LongScape: Advancing Long-Horizon Embodied World Models with Context-Aware MoE (https://arxiv.org/abs/2509.21790)
Comments:
          13 pages, 8 figures

- **What's New**: 이번 논문에서는 LongScape라는 새로운 하이브리드 프레임워크를 제안합니다. 이 프레임워크는 intra-chunk diffusion denoising과 inter-chunk autoregressive causal generation을 결합하여 긴 시간에 걸친 비디오 생성의 안정성을 개선합니다. 핵심 혁신은 로봇 행동의 의미론적 맥락에 따라 비디오를 나누는 동작 유도 가변 길이 청크 메커니즘을 도입한 것입니다.

- **Technical Details**: LongScape는 고유한 가변 길이 청크 생성 메커니즘을 사용하여 각 비디오 청크가 완전하고 일관된 행동을 나타내도록 합니다. 이를 위해 그립퍼 상태 및 효과기 움직임의 크기를 활용하여 청크의 정밀도를 결정합니다. 예를 들어, 로봇 그립퍼 상태가 변화하거나 상당한 움직임이 있는 경우 더 짧고 세분화된 청크로 나누어 높은 품질의 생성을 보장합니다.

- **Performance Highlights**: LongScape는 LIBERO 및 AGIBOT-World 벤치마크에서 광범위한 실험을 통해 기존 모델에 비해 비디오 생성 품질의 최고 성능을 달성했습니다. 15회 롤아웃에서 시각적 일관성과 안정성을 유지할 수 있는 능력을 보여주며, 이는 장기 생성 능력을 강조합니다. 우리 모델은 안정적이고 고품질의 긴 비디오 생성을 가능하게 하는 주요 기여를 합니다.



### MIRG-RL: Multi-Image Reasoning and Grounding with Reinforcement Learning (https://arxiv.org/abs/2509.21788)
- **What's New**: 이 논문은 다중 이미지 이유 및 그라운딩(multi-image reasoning and grounding)을 위한 새로운 통합 프레임워크인 MIRG-RL을 소개합니다. 기존의 모델들이 겪던 두 가지 주요 문제인 이미지 간 이유 명료화 부족과 보상 모델링의 한계를 극복하기 위해, 새로운 강화 학습( reinforcement learning) 방법을 도입했습니다. 이 연구는 다중 이미지가 포함된 맥락에서 모델이 효과적으로 기능할 수 있도록 고안된 데이터 생성 및 훈련 방법을 제안합니다.

- **Technical Details**: MIRG-RL은 두 단계의 훈련 패러다임을 채택하여 주석이 달린 경로를 이용한 감독적 미세 조정(supervised fine-tuning)과 이미지 인식 강화 학습( image-aware reinforcement learning) 최적화를 결합합니다. 이 과정에서 객체 수준 및 이미지 수준의 주석 정보를 통합한 경로 데이터를 생성하는 혁신적인 방법을 사용하며, 이를 통해 고품질의 경량 이유 향상 데이터를 만들어냅니다. 모델은 두 가지 기본 보상 함수를 사용하여 멀티 이미지의 정확도를 개선하며, 더 나아가 이미지를 통한 자세한 이해를 가능하게 합니다.

- **Performance Highlights**: MIRG-RL은 다중 이미지 그라운딩 벤치마크에서 SOTA(state-of-the-art) 성능을 달성했습니다. 실험 결과, Cross-image reasoning 작업에서 64.82%의 정확도를 기록하며, 이전의 최고 방법보다 1% 향상된 결과를 보였습니다. 이를 통해 MIRG-RL의 다중 이미지 이유 능력이 기술적으로 우수함을 입증하였습니다.



### DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate Speech in Images (https://arxiv.org/abs/2509.21787)
Comments:
          Defactify 3 workshop at AAAI 2024

- **What's New**: 이번 연구는 디지털 콘텐츠에서 증오를 식별하기 위한 다중 모드(multi-modal) 데이터 세트를 도입합니다. 이 데이터 세트는 수위가 있는 안정화된 확산(stable diffusion) 기술과 디지털 주의 분석 모듈(Digital Attention Analysis Module, DAAM)을 결합하여 혐오 요소를 인식하는데 중점을 둡니다. 이를 통해 이미지 속 혐오 구역을 흐리게 처리하여 적절한 AI 응용 프로그램을 향상시키고자 하는 노력을 담고 있습니다.

- **Technical Details**: 연구는 텍스트와 이미지 간의 일관된 정렬을 위한 프롬프트 엔지니어링(prompt engineering) 및 안정화된 확산 프로세스에서 생성된 이미지의 혐오 요소를 중화하는 독특한 언어-이미지 모델인 DeHater를 포함합니다. 본 연구는 Hatenorm 데이터 세트를 활용하여 악성 텍스트를 규명하고, 이 텍스트를 바탕으로 합성된 이미지를 생성합니다. 또한, DAAM을 활용하여 생성된 이미지 내의 혐오적인 요소를 정확하게 핀포인트하고 이를 효과적으로 흐리게 처리하는 새로운 방법론을 제시합니다.

- **Performance Highlights**: 우리의 데이터 세트는 총 2411개의 사례를 포함하고 있으며, 교육 세트와 테스트 세트로 나뉘어 있습니다. Defactify 3 워크숍의 dehate 공유 작업의 일환으로 데이터가 제공되었습니다. 향후 연구를 위해 IOU(Intersection over Union) 메트릭을 활용해 참여자들의 성과를 평가하였고, 20개 이상의 등록과 5개의 제출이 있었습니다.



### Prompt-guided Representation Disentanglement for Action Recognition (https://arxiv.org/abs/2509.21783)
- **What's New**: 본 논문에서는 여러 동작이 존재하는 복잡한 장면에서 특정 동작을 분리하고 분석하는 새로운 방법인 Prompt-guided Disentangled Representation for Action Recognition (ProDA)를 제안합니다. ProDA는 Spatio-temporal Scene Graphs (SSGs)를 활용하여 동작별 표현을 생성하는 Graph Parsing Neural Network (GPNN)를 안내하는 Dynamic Prompt Module (DPM)을 도입합니다. 이를 통해 기존 방법들이 가진 동작 상호작용 모델링의 한계를 극복하고, 특정 동작에 대한 보다 정확한 이해를 제공합니다.

- **Technical Details**: ProDA의 핵심은 복잡한 장면에서 특정 동작을 떼어내는 기능입니다. 여기서 사용되는 DPM은 지정된 동작의 multi-hot 벡터와 SSG 특성을 결합하여 동작 인식에 적합한 프롬프트를 생성합니다. 또한, VGPNN을 통해 동작 분리를 수행하고, VGNorm 모듈을 이용하여 서로 다른 비디오의 SSG 간의 차이를 보존하여 시간적 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, ProDA는 다중 레이블 동작 인식 벤치마크에서 State-of-the-Art의 성능을 달성했으며 행동 위치 추적(supervision) 없이도 뛰어난 동작 위치 추적 성능을 입증하였습니다. 이 연구는 동작 이해의 해석 가능성을 높이고, 복잡한 동작을 이해하는 데 더욱 효과적임을 보여줍니다.



### Training-Free Multimodal Deepfake Detection via Graph Reasoning (https://arxiv.org/abs/2509.21774)
- **What's New**: 이번 연구에서는 Multimodal deepfake detection (MDD)에 초점을 맞춰, 시각적, 텍스트, 청각적 모달리티를 함께 모델링하여 현대 정보 시스템의 신뢰성을 강화합니다. GASP-ICL(가이디드 어댑티브 스코어 및 컨텍스트 학습)이라는 새로운 프레임워크를 제안하여, LVLMs(대형 비전-언어 모델)의 효과적인 활용을 위한 장벽을 극복하려고 합니다. 이 방법은 훈련이 필요 없는 교육 방식을 채택하고 있어 비용과 자원 소모를 줄이는데 기여합니다.

- **Technical Details**: GASP-ICL은 LVLM을 기반으로 한 MDD 문제를 다루며 이진 분류 문제로 공식화됩니다. 각 샘플은 시각적 입력 I와 텍스트 입력 T로 구성되며, 모델은 입력의 진위 여부를 예측합니다. 기존의 ICL(문맥 학습) 방식은 유사성 기반 검색에 의존해 왔으나, GASP-ICL은 구조화된 선택 파이프라인을 도입하여 각 쿼리에 특화된 태스크 중심의 프롬프트를 구축합니다.

- **Performance Highlights**: GASP-ICL 프레임워크는 기존의 강력한 기준치보다 월등한 성능 향상을 보여주며, LVLM을 추가적인 훈련 없이 활용할 수 있음을 입증하였습니다. 다양한 변조 유형과 복잡한 시나리오에서도 잘 일반화되는 성능을 보입니다. 실험 결과에 따르면, 이 방법은 세밀한 변조 단서를 정확하게 포착하고, 세분화된 맥락을 파악하는 데 있어 뛰어난 효과를 나타냅니다.



### CubistMerge: Spatial-Preserving Token Merging For Diverse ViT Backbones (https://arxiv.org/abs/2509.21764)
- **What's New**: 이 논문에서는 공간 구조를 유지하면서 효과적으로 토큰을 병합하는 새로운 방법인 CubistMerge를 소개합니다. 기존의 토큰 감소 방식이 공간 아키텍처와의 호환성을 해치기는 해도, 공간 구조와 정보의 불균형한 분포를 동시에 활용하는데 성공합니다. 이 방법은 비구조적인 레이아웃을 피하면서, 기존 모델들에서 성능을 향상시키는 데 기여하였습니다.

- **Technical Details**: CubistMerge는 2D 토큰 감소 전략, 공간 인식 토큰 병합 알고리즘, 그리고 중요한 정보가 잘 보존되도록 하는 새로운 토큰 표현 방식을 포함합니다. 첫 번째 단계인 2D 토큰 감소 전략은 각 행과 열마다 일관된 토큰 수를 유지하도록 설계되었습니다. 두 번째 단계는 병합 후에도 상대적 공간 관계를 유지하는 토큰 병합 알고리즘을 사용하고, 세 번째 단계에서는 최대 크기 토큰 표현 방식을 통해 주요 특징을 효과적으로 인식합니다.

- **Performance Highlights**: CubistMerge는 다양한 비전 작업에서 state-of-the-art 성능을 달성했습니다. 특히 COCO 데이터셋에서 0.7%의 mIOU 저하로 1.25배 속도 향상을 달성하였고, ImageNet에서 1.15배의 속도 향상을 이루면서도 top-1 정확도에서는 손실이 없었습니다. 이러한 실적은 CubistMerge가 공간 아키텍처와 비공간 아키텍처 모두에서 효과적으로 작동할 수 있음을 보여줍니다.



### UniVid: Unifying Vision Tasks with Pre-trained Video Generation Models (https://arxiv.org/abs/2509.21760)
- **What's New**: 본 논문은 UniVid라는 새로운 프레임워크를 제안하며, 이는 사전 훈련된 비디오 생성 모델을 활용하여 다양한 비전 작업에 대응할 수 있도록 합니다. 기존 모델들이 요구했던 작업별 사전 훈련 데이터 없이, 하나의 비디오 모델이 여러 비전 작업에 쉽게 적응할 수 있는지를 탐구합니다. 이를 통해 이미지와 비디오 작업을 시각적 문장으로 표현하고, 공통된 맥락(sequence)을 통해 작업을 처리하는 방식으로 접근합니다.

- **Technical Details**: UniVid는 비디오 확산 변환기(video diffusion transformer)를 세부 조정(fine-tuning)하여 작업 별 수정없이 다양한 비전 작업을 처리하도록 설계되었습니다. 각 작업은 시각적 문장으로 구성되며, 각 샘플은 문맥(context) 순서로 작업과 기대되는 출력 양식을 정의합니다. 우리는 UniVid의 일반화 가능성을 평가할 때 이미지와 비디오 혼합 입력으로부터 출력 양식을 추론할 수 있는지와, 라벨이 포함된 데이터 없이 자연 비디오에서 주어진 데이터로의 크로스 소스 작업을 다룰 수 있는지를 중점적으로 살펴봅니다.

- **Performance Highlights**: UniVid는 오로지 자연 비디오 데이터로 사전 훈련됨에도 불구하고, 크로스 모달 및 크로스 소스 작업에서 효과적으로 일반화됩니다. 특히, 생성 작업과 이해 작업 간의 전환이 시각적 문장 내 요소의 순서를 단순히 변경함으로써 쉽게 이루어질 수 있음을 보여줍니다. 이러한 결과는 사전 훈련된 비디오 생성 모델이 비전 모델링의 통합된 기초로 기능할 수 있는 잠재력을 강조합니다.



### KG-SAM: Injecting Anatomical Knowledge into Segment Anything Models via Conditional Random Fields (https://arxiv.org/abs/2509.21750)
- **What's New**: 이번 연구에서는 의료 이미징에서의 고유한 문제를 해결하기 위해 KG-SAM(Knowledge-Guided Segment Anything Model)을 제안합니다. KG-SAM은 해부학적 우선 순위, 경계 정교화, 그리고 불확실성 정량화를 통합하여 더욱 신뢰할 수 있는 분할 결과를 제공합니다. 특히, 의료 지식 그래프를 활용하여 상세한 해부학적 관계를 인코딩하고, 에너지 기반의 Conditional Random Field (CRF)를 통해 생물학적으로 일관된 예측을 보장합니다.

- **Technical Details**: KG-SAM 프레임워크는 시각적 토대 모델과 명시적 의료 지식을 결합한 이원 스트림 아키텍처를 채택합니다. 먼저, 사전 훈련된 SAM 모델을 활용하여 초기 확률 맵과 심층적 이미지 특징을 생성하고, 동시에 의료 지식 그래프를 구축하여 해부학적 우선 사항을 표현합니다. CRF 모듈을 통한 에너지 최적화 과정을 통해 해부학적 제약과 시각적 증거를 결합하고, 이를 통해 정교화된 세분화 및 불확실성 맵을 얻습니다.

- **Performance Highlights**: 실험 결과, KG-SAM은 전립선 세분화에서 평균 Dice 점수 82.69%를 달성하며, 복부 세분화에서도 MRI에서 78.05%, CT에서 79.68%를 기록했습니다. 이러한 결과는 KG-SAM이 정교하게 구조화된 해부학적 제약을 반영하여 뛰어난 성능을 발휘함을 보여줍니다. 또한, 기존 SAM 구현에 비해 14.71%의 성능 향상을 달성했으며, DeSAM과의 통합을 통해 2.27%의 추가 개선 효과도 입증했습니다.



### Incorporating Scene Context and Semantic Labels for Enhanced Group-level Emotion Recognition (https://arxiv.org/abs/2509.21747)
Comments:
          10 pages, 5figures, submitted to IEEE Transactions on Human-Machine Systems

- **What's New**: 이 논문에서는 그룹 수준의 감정 인식(Group-level emotion recognition, GER)을 개선하기 위한 새로운 프레임워크를 제안합니다. 기존 방법들이 개별 관계에 대한 시각적 장면의 맥락 정보를 충분히 반영하지 못하고, 감정 라벨로부터의 의미 정보의 중요성을 간과하고 있다는 점을 지적합니다. 본 연구는 시각적 맥락 정보를 통합하고 라벨 기반의 의미 정보를 활용하여 GER의 성능을 향상시키는 혁신적인 접근법을 제시합니다.

- **Technical Details**: 제안된 방법은 시각적 맥락 인코딩 모듈을 통해 다중 스케일(scene) 정보를 활용하여 개별 관계를 다양하게 인코딩합니다. 감정 의미 인코딩 모듈은 그룹 수준의 감정 라벨을 사용하여 대규모 언어 모델(large language model)을 통해 세밀한 감정 어휘를 생성합니다. 그후 이러한 어휘가 감정 라벨과 결합되어 구조화된 감정 트리를 통해 종합적인 의미 표현으로 정제됩니다.

- **Performance Highlights**: 세 가지 널리 사용되는 GER 데이터셋(GAFF2, GAFF3, GroupEmoW)에 대한 실험에서 제안된 방법이 기존의 최첨단 방법과 비교해 경쟁력 있는 성과를 거두었음을 보여줍니다. 이 연구는 감정 라벨이 GER에 통합된 최초의 접근 방식으로, 감정에 대한 포괄적인 이해를 돕고 모델의 판별력을 향상시킵니다. 전체적으로 제안된 방법은 복잡한 장면을 이해하고 감정을 보다 정확하게 추론할 수 있도록 합니다.



### LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation (https://arxiv.org/abs/2509.21738)
- **What's New**: 이 논문에서는 자원 제약이 있는 임상 환경에서 사용하기 위한 경량 망막 혈관 분할 네트워크인 LFA-Net을 제안합니다. LFA-Net은 새로운 주의 모듈인 LiteFusion-Attention을 통합하여 경량 및 효율적인 방식으로 지역 및 전역 맥락을 캡처할 수 있습니다. 이 연구는 기존 딥러닝 기반 분할 방법에서 발생하는 작은 혈관 분할과 높은 계산 비용의 도전 과제를 해결하고자 합니다.

- **Technical Details**: LFA-Net은 0.11 백만 개의 파라미터, 0.42 MB의 메모리 크기 및 4.46 GFLOPs의 수치를 자랑하며, 이는 자원 제한 환경에서의 실시간 배치를 위한 이상적인 설계를 가능하게 합니다. LiteFusion-Attention 모듈은 잔차 학습 연결, Vision Mamba에서 영감을 받은 역학, 변조 기반 주의를 결합한 효과적인 주의 메커니즘을 이용하여 다중 스케일 표현 학습을 수행합니다. 이러한 설계는 메모리 사용을 최소화하면서도 세밀한 분할을 보장합니다.

- **Performance Highlights**: LFA-Net은 DRIVE, STARE 및 CHASE_DB 데이터셋에서 각각 83.28%, 87.44%, 84.50%의 Dice 점수와 72.85%, 79.31%, 74.70%의 Jaccard 지수를 기록하며 뛰어난 성능을 보였습니다. 이 논문은 LFA-Net이 기존의 더 크고 복잡한 모델보다 우수한 성과를 내며, 자원 제한 환경에서의 임상 배치에 이상적임을 입증합니다. 또한, 제안된 모델은 임상 진단의 신뢰성을 높이며 효율성 또한 보장합니다.



### UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments (https://arxiv.org/abs/2509.21733)
- **What's New**: 이번 논문에서는 UISim이라는 새로운 이미지 기반 UI(simulator) 시뮬레이터를 소개합니다. 이 시스템은 단순한 화면 이미지에서 포괄적으로 모바일 환경을 탐색할 수 있는 다이나믹하고 인터랙티브한 플랫폼을 제공합니다. 기존의 방법들에서 느껴지는 제약을 해소하며, UI 전환을 현실감 있게 시뮬레이션할 수 있도록 설계되었습니다.

- **Technical Details**: UISim은 초기 전화 화면 이미지와 사용자 행동을 바탕으로 두 단계의 방법론을 사용하여 다음 UI 상태를 생성합니다. 첫 번째 단계에서 추상적인 레이아웃 정보를 예측하고, 두 번째 단계에서 이 정보를 기반으로 새로운 시각적으로 일관된 UI 이미지를 합성합니다. 이러한 구조의 분리는 복잡한 이미지 간 UI 변환 문제를 보다 관리하기 쉬운 하위 문제로 나누어, 높은 충실도와 다양한 생성 능력을 제공합니다.

- **Performance Highlights**: 실험 결과에서 UISim은 이전의 UI 생성 모델보다 36.73의 Fréchet Inception Distance로 우수한 성능을 보여주었습니다. 이는 UISim이 현실적이고 일관된 다음 UI 상태를 생성하는 데 효과적임을 입증합니다. 또한, UI 테스트와 신속한 프로토타입 제작, 고급 응용 프로그램에도 유용하여 AI 에이전트의 UI 내비게이션 작업 계획에 기여할 수 있습니다.



### On the Status of Foundation Models for SAR Imagery (https://arxiv.org/abs/2509.21722)
- **What's New**: 이번 연구에서는 Synthetic Aperture Radar (SAR) 객체 인식 작업을 위한 기초 AI/ML 모델의 가능성을 조사합니다. 특히 Self-Supervised Learning (SSL)으로 훈련된 시각적 기초 모델들이 SAR 도메인에서 매우 제한된 레이블 데이터로도 조정 가능하다는 점을 강조합니다. 기존의 SAR 모델에 비해 새로운 역량을 갖춘 SAR 기초 모델 개발을 위해 여러 실험을 수행했습니다.

- **Technical Details**: 기존의 시각적 기초 모델(SARATR-X 포함)을 사용하여 SAR 이미지에서 의미 있는 특징을 추출할 수 있는지 테스트했습니다. 이를 통해 DINOv2 모델을 사용하여 SAR 데이터에서의 자가 감독 미세 조정(self-supervised finetuning)을 수행하였고, 이를 통해 태스크 적응 과정에서 유의미한 개선 사항을 발견했습니다. 또한, 여러 뼈대(backbone)와 태스크 적응 레시피(task-adaptation recipe)를 사용한 성능 분석을 진행하였습니다.

- **Performance Highlights**: 연구 결과, SAR 데이터에 미세 조정된 DINOv2가 기존의 최첨단 모델인 SARATR-X보다 월등한 성능을 보여주었습니다. 이는 SAR 도메인에서 SSL을 활용한 모델이 강력한 성능을 발휘할 수 있다는 것을 시사합니다. 향후 SAR 기초 모델 설계에 있어서 자가 감독 미세 조정 방식이 유망한 방향임을 제안합니다.



### DeLiVR: Differential Spatiotemporal Lie Bias for Efficient Video Deraining (https://arxiv.org/abs/2509.21719)
- **What's New**: 이 논문에서는 Lie 그룹 이론을 비디오 강우 제거(video deraining)에 최초로 도입하여 새로운 스페이치오-템포랄 Lie 메커니즘을 제안합니다. 이는 기하학적 사전(geometric priors)을 기반으로 하여 동적 장면의 피처 정렬 문제를 해결하기 위한 새로운 패러다임을 제공합니다. 제안된 DeLiVR는 강우의 강도와 방향을 효과적으로 추정하기 위해 Lie 그룹 기반의 스페이치오-템포랄 편향을 주목(attention) 메커니즘에 직접 주입함으로써, 기존 방법들이 가진 한계를 극복하고자 합니다.

- **Technical Details**: DeLiVR는 입력 비디오 클립을 패치로 나누고, 각 프레임의 회전을 예측하는 경량화된 SO(2) 헤드를 통해 비디오 복원을 수행합니다. 이 방법은 기하학 일관성을 유지하도록 공간 편향(spatial bias)과 상대적인 각 변위를 반영하는 템포랄 편향(temporal bias)을 구축합니다. 이러한 두 편향은 시간적 감소(temporal decay) 및 마스킹 전략을 통해 통합되어, 이전 프레임과의 신뢰할 수 있는 공간-시간 대응 관계를 강조합니다.

- **Performance Highlights**: 실험 결과, DeLiVR는 여러 공개 벤치마크에서 기존의 최첨단 방법들을 능가하여 더 명확한 세부사항과 완전한 강우 제거 효과를 나타냈습니다. 또한, 이 방법은 자율 주행 및 로봇 항법과 같은 고급 비전 과제에서의 정확도를 개선합니다. DeLiVR는 복잡한 환경에서도 강건한 정렬을 제공하며, 광범위한 비디오 데이터에 대한 일반화 능력도 향상되었습니다.



### Motion-Aware Transformer for Multi-Object Tracking (https://arxiv.org/abs/2509.21715)
- **What's New**: 본 논문에서는 Multi-object tracking (MOT)에서 발생하는 쿼리 충돌(query collisions) 문제를 해결하기 위한 Motion-Aware Transformer (MATR)를 제안합니다. MATR는 객체의 움직임을 명시적으로 예측하여 프레임 간의 트랙 쿼리를 미리 업데이트합니다. 이를 통해 딱딱한 쿼리 매칭 과정에서 발생하는 문제를 줄이고, 탐지(detection)와 연관성(association) 모두를 개선합니다. 다양한 데이터셋에서의 실험에 따라 MATR는 MOTR보다 유의미한 성능 향상을 보여 줍니다.

- **Technical Details**: MATR는 Transformer Decoder에 들어가기 전에 트랙 쿼리의 위치적 임베딩과 특징을 업데이트합니다. 이러한 접근 방식은 트랙 쿼리와 그라운드 진실(ground truth) 타겟 간의 간극을 줄여 빠른 객체 이동을 예측하고, 이를 통해 쿼리 간의 충돌을 최소화합니다. MOTR을 기반으로 하여, MATR는 향상된 훈련 전략과 최신 DETR 요소인 DAB-DETR로 기반을 강화합니다. 이는 보다 강력한 비교를 가능하게 하고, 제안된 모션 기반 디자인으로 인한 성과를 보장합니다.

- **Performance Highlights**: 실험 결과, MATR는 DanceTrack에서 MOTR보다 HOTA를 9점 이상 향상시켰으며, 보조 데이터를 포함할 경우 71.3의 새로운 최첨단 점수를 기록했습니다. SportsMOT에서도 72.2 HOTA를 달성하여 최상의 성능을 입증했습니다. BDD100k에서는 54.7 mTETA 및 41.6 mHOTA를 달성하여 이전 방법들을 초월했습니다. 이로 인해, MOT와 END-TO-END Transformer 기반 트래커에서 모션 예측의 중요성이 강조됩니다.



### MS-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss (https://arxiv.org/abs/2509.21696)
Comments:
          Accepted by the International Joint Conference on Neural Networks (IJCNN) 2025. Keywords: Infrared Object Detection, MobileNetV4, SlideLoss, YOLO Model

- **What's New**: 이 논문은 저조도와 불리한 날씨 조건에서의 도시 객체 탐지를 위한 새로운 접근 방식인 MS-YOLO를 소개합니다. MS-YOLO는 YOLOv8의 CSPDarknet 백본을 MobileNetV4로 대체하여 계산 효율성을 높이고, 새로운 손실 함수 SlideLoss를 도입하여 클래스 불균형 문제를 해결합니다. 이를 통해 MS-YOLO는 높은 정밀도를 유지하면서도 6.7 GFLOPs의 낮은 연산량으로 도시 환경에서의 실시간 배치를 가능하게 합니다.

- **Technical Details**: 이 연구에서는 YOLO 변형에 대한 비교를 통해 YOLOv8이 높은 정밀도와 회수를 보이는 최적의 아키텍처임을 확인했습니다. 또한, MobileNetV4를 통합하여 MS-YOLO의 계산 오버헤드를 1.5% 줄였습니다. SlideLoss는 학습 과정에서 역량 비율을 동적으로 조정하여 오클루전(occlusion)과 드물게 나타나는 데이터를 더 효과적으로 처리합니다.

- **Performance Highlights**: FLIR ADAS V2 벤치마크에서 MS-YOLO는 경쟁력 있는 mAP와 함께 YOLOv8보다 4% 더 높은 정밀도를 달성하며, YOLOv9보다 18.3% 더 빠른 추론 속도를 기록했습니다. 이러한 성과들은 MS-YOLO가 높은 탐지 품질과 낮은 계산 비용을 동시에 해결할 수 있음을 보여주며, 도시 환경에서의 실시간 경량 배치에 적합하다는 것을 입증합니다.



### MORPH: Shape-agnostic PDE Foundation Models (https://arxiv.org/abs/2509.21670)
- **What's New**: MORPH는 형태에 구애받지 않는 자가 회귀형 (autoregressive) 기초 모델로, 부분 미분 방정식 (PDEs) 문제 해결을 위한 것입니다. 이 모델은 다른 차원의 다양한 spacetime 데이터 세트를 자동으로 처리하게 설계된 convolutional vision transformer 아키텍처를 사용합니다. 주요 기술로는 component-wise convolution, inter-field cross-attention 및 axial attentions가 포함되며, 이들은 모두 복잡한 과학적 데이터를 효과적으로 처리할 수 있는 기능을 제공합니다.

- **Technical Details**: MORPH의 아키텍처는 여러 가지 필드와 혼합된 스칼라 및 벡터 성분을 다릅니다. component-wise convolution은 스칼라와 벡터 채널에서의 지역적인 상호작용을 포착하며, inter-field cross-attention은 서로 다른 물리적 필드 간의 정보를 선택적으로 전파합니다. 마지막으로, axial attentions는 전체 spatiotemporal self-attention을 개별 공간 및 시간 축을 따라 분해하여 계산 부담을 줄이며 표현력을 유지합니다.

- **Performance Highlights**: MORPH는 여러 가지 하류 예측 작업으로의 전이 학습에서 초기 모델을 초과하는 성능을 보입니다. 특히 zero-shot 및 full-shot 일반화 모두에서 모델 성능을 극대화하여 강력한 기준선과 최신 모델을 초과하는 성능을 달성하였습니다. 이러한 특성들은 다양한 과학적 관측의 이질적이고 다차원적인 본질에서 학습하기 위한 유연하고 강력한 기초 구조를 제시합니다.



### FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction (https://arxiv.org/abs/2509.21657)
- **What's New**: FantasyWorld는 정적 비디오 모델에 훈련 가능한 기하학적 분기를 추가하여 비디오 잠재량과 암시적 3D 필드를 결합하는 혁신적인 프레임워크입니다. 이 접근 방식은 기하학적 힌트가 비디오 생성 과정을 안내하고, 비디오 선행 정보가 3D 예측을 정규화하게 하여 일관되고 일반화된 3D 인식이 가능한 비디오 표현을 생성하도록 돕습니다. 이러한 새로운 비디오-3D 모델링 구조는 기존의 기준선보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: FantasyWorld는 비디오 선행 블록과 기하학 동시 인코더를 포함하는 Integrated Reconstruction and Generation Blocks (IRG)을 통해 비디오의 조건부 특성과 명시적 3D 표현을 단일 포워드 패스에서 생성합니다. 또한, 이 구조는 억세스 없이 독립적으로 강화된 깊이와 포인트 클라우드를 예측하는 대신 비디오 잠재량에서 카메라 매개변수와 3D 신호를 직접 유추합니다. 이로 인해, 추가적인 3D 재구성 과정을 요구하지 않고도 3D-일관된 특징을 재사용 가능하게 만듭니다.

- **Performance Highlights**: FantasyWorld는 비디오 상상력과 3D 인식 사이의 간극을 효율적으로 연결하여 다중 관점 일관성과 스타일 일관성에서 기존의 기하학 일관성 기준선을 초월하는 성능을 보였습니다. 또한, 크게 개선된 3D 인식 능력을 유지하면서도 높은 품질의 3D 표현을 생성할 수 있는 잠재력을 보여줍니다. 이런 방식으로, 이 구조는 최신 작업에서 보여준 새로운 시각 합성 및 탐색과 같은 다양한 다운스트림 3D 작업에 사용될 수 있습니다.



### A Data-driven Typology of Vision Models from Integrated Representational Metrics (https://arxiv.org/abs/2509.21628)
- **What's New**: 이 논문은 다양한 구조의 비전 모델들 간의 표현 차이를 체계적으로 평가하는 새로운 프레임워크를 소개합니다. 이 연구는 기하학(geometry)나 조정(tuning)과 같은 특정 표현적 특성이 모델 가계의 분리를 어떻게 기여하는지를 규명하여, 개별 메트릭보다 더 명확한 가족 정체성을 제공하는 Similarity Network Fusion (SNF) 방법을 적용합니다. 기존의 접근법에 대한 보완으로, 이 연구는 자가 감독(self-supervised) 모델들이 아키텍처 경계를 초월하여 자연스러운 군집을 형성하고 있음을 발견했습니다.

- **Technical Details**: 연구진은 35개의 비전 모델을 네 가지 주요 범주로 분석하며, 이는 감독된 Convolutional Neural Networks (CNNs), 자가 감독 CNNs, 감독된 Transformers, 자가 감독 Transformers입니다. 다양한 유사성 메트릭을 평가하고, 기하학적 변환(Procrustes) 및 선형 예측(Linear Predictivity) 접근 방식의 유연성을 비교하여 모델 간의 표현 구조를 분석합니다. Singular Value Decomposition (SVD)와 Canonical Correlation Analysis (CCA)와 같은 방법론이 사용되며, 이론적으로 중요한 정보를 보존하는 다양한 매트릭의 효과를 평가합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 Hybrid 구조(ConvNeXt, Swin)가 MAE 모델들과 군집을 이루고, 자가 감독 모델들이 서로 밀접하게 연관되어 있음을 보여줍니다. 이러한 발견은 다양한 아키텍처 및 훈련 목표에 의한 나타나는 계산 전략이 표면적인 설계 카테고리를 넘어 얼마나 큰 의미를 가지는지를 잘 보여줍니다. 제안된 데이터 기반의 비전 모델 분류법은 연구자들이 모델의 관계를 이해하고, 전이 학습(transfer learning) 호환성을 예측하며, 새로운 작업에서 유사한 행동을 보이는 모델을 선택하는 데 유용한 도구를 제공합니다.



### VLCE: A Knowledge-Enhanced Framework for Image Description in Disaster Assessmen (https://arxiv.org/abs/2509.21609)
Comments:
          29 pages, 40 figures, 3 algorithms

- **What's New**: 이 연구에서는 재난 이미지 분석을 위한 새로운 다중 모달 시스템인 VLCE(Vision Language Caption Enhancer)를 소개합니다. VLCE는 재난 이미지를 이해하기 위한 문맥적으로 풍부한 설명을 자동으로 생성하는 기능을 갖추고 있습니다. 이 시스템은 CNN-LSTM 모델과 비전 변환기(Vision Transformer) 모델을 결합하여 위성 및 드론 이미지의 분석을 향상시키는 것을 목표로 합니다.

- **Technical Details**: VLCE는 EuroSat 데이터셋에서 사전 훈련된 ResNet50 기반 모델과 RescueNet 데이터셋에서 사전 훈련된 Vision Transformer 모델을 사용하는 이중 아키텍처 접근 방식을 활용합니다. 이 시스템은 ConceptNet와 WordNet에서 외부 의미 지식을 활용하여 어휘 범위를 확장하고 설명 정확성을 개선합니다. 또한, cross-modal attention 메커니즘을 통해 시각 신호와 도메인 특정 텍스트 설명을 연결하는 맞춤형 아키텍처로 구성됩니다.

- **Performance Highlights**: VLCE는 CLIPScore와 InfoMetIC을 사용하여 기존 비전-언어 모델들과 비교하였으며, InfoMetIC에서 최대 95.33%의 성과를 기록하며 뛰어난 결과를 보여주었습니다. 이는 기존의 기준 모델과 비교하여 손상 평가의 정밀성과 정보 밀도가 크게 향상되었음을 나타냅니다. VLCE는 재난 대응 관점에서 매우 유용한 자동화된 설명 생성을 통해 긴급 상황에서의 정보 접근성을 향상시키는데 기여할 것입니다.



### Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis (https://arxiv.org/abs/2509.21595)
- **What's New**: 이번 연구는 비디오 액션 인식을 위한 두 가지 주요 자가 감독 학습 아키텍처인 DINOv3와 V-JEPA2를 비교 분석합니다. DINOv3는 프레임을 독립적으로 처리하는 방식으로 공간적 특징을 추출하고, V-JEPA2는 비디오 시퀀스에 걸친 공동 시간 모델링을 활용합니다. UCF Sports 데이터셋을 기반으로 두 접근 방식의 특징 품질을 분류 정확도, 군집 성능 등 다양한 차원에서 평가하였습니다.

- **Technical Details**: DINOv3는 이미지 컬렉션에서 훈련된 교사-학생 증류 프레임워크를 사용하여 자가 감독 방식으로 밀집 시각적 특징을 학습합니다. 반면, V-JEPA2는 동작 역학과 프레임 관계를 인코딩하는 시간 토큰을 생성하기 위해 이를 공동으로 처리하는 아키텍처입니다. 이 두 접근 방식의 주요 차이는 공간적 충실성과 시간적 모델링 간의 근본적인 긴장을 반영합니다.

- **Performance Highlights**: DINOv3는 정적 포즈 인식에서 우수한 성능을 보이는 반면, V-JEPA2는 모든 액션 유형에서 일관된 신뢰성을 제공합니다. DINOv3는 포즈 식별 가능한 액션에 대해 뛰어난 판별 능력을 보여주지만, 움직임에 의존하는 액션에서는 성능 저하가 발생합니다. 이 연구는 비디오 분석 시스템에서의 건축 설계 선택에 대한 이해를 높이고, 작업 요구 및 신뢰성 제약에 따라 적절한 특징 추출 방법 선택에 대한 실증적 지침을 제공합니다.



### What Happens Next? Anticipating Future Motion by Generating Point Trajectories (https://arxiv.org/abs/2509.21592)
- **What's New**: 이 논문에서는 단일 이미지에서 동작을 예측하는 문제를 다룹니다. 기존의 비디오 생성기 아키텍처를 토대로 하여, 픽셀 대신에 운동 궤적(trajectory)을 생성하는 모델을 제안합니다. 이를 통해 장면의 동적 변화를 포착하고 예측의 정확성과 다양성을 향상시킵니다.

- **Technical Details**: 이 연구는 입력 이미지에 따라 예측되는 궤적을 생성하는 확률적 모델링을 실시합니다. 기존의 점 추적(point track) 모델과 달리, 모든 점에 대한 동작을 예측하여 전체 장면을 고려합니다. 특히, 비디오 생성기와 유사하게 흐름 일치(flow matching) 기법을 활용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 이전 방식 대비 더 효과적으로 동작 예측을 수행하며, 특히 합성 및 실제 시나리오에서 우수한 성능을 보였습니다. 기존의 최첨단 비디오 모델들이 단순한 시뮬레이션 상황에서도 어려움을 겪는 반면, 직관적으로 물리적 현상을 이해하는 데 있어 제안된 접근 방식이 훨씬 더 효율적이라는 점이 입증되었습니다.



### X-Streamer: Unified Human World Modeling with Audiovisual Interaction (https://arxiv.org/abs/2509.21574)
Comments:
          Project Page at this https URL

- **What's New**: X-Streamer라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 텍스트, 음성, 비디오의 무한 상호작용을 지원하는 디지털 인간 에이전트를 위한 엔드 투 엔드 다중모달(human world modeling) 모델입니다. X-Streamer는 단일 초상화(portrait)에서 시작해 실시간으로 개방형 비디오 통화를 가능하게 하며, Thinker-Actor 이중 트랜스포머 아키텍처를 통해 멀티모달 이해 및 생성 기능을 통합하고 있습니다.

- **Technical Details**: X-Streamer의 핵심은 Thinker와 Actor 모듈로 구성된 구조입니다. Thinker는 사용자 입력을 인식하고 이유를 제기하며, Actor는 이 숨겨진 상태(hidden states)를 실시간으로 동기화된 멀티모달 스트림으로 변환합니다. 이를 위해 사전 훈련된 대형 언어-음성 모델(large language-speech model)과 chunk-wise autoregressive diffusion 모델이 활용되어 텍스트, 음성 및 비디오 응답을 생성합니다.

- **Performance Highlights**: X-Streamer는 두 개의 A100 GPU에서 실시간으로 작동하며, 4,248.64시간의 대화 비디오로 훈련되어 긴 대화의 일관성을 유지하며 무한한 오디오비주얼 스트림을 생성할 수 있습니다. 이 모델은 멀티모달 생성에서 실시간 효율성과 긴 범위의 시청각 일관성을 보장하며, 엔터테인먼트, 교육, 쇼핑과 같은 다양한 분야에서 디지털 인간 에이전트의 변화를 이끌 기반 기술로 기대되고 있습니다.



### Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms (https://arxiv.org/abs/2509.21573)
- **What's New**: 본 논문에서는 새로운 공간 정규화 대조 학습 전략을 제안하여 기하학적 거리를 고려한 이미지 기반 위치 추정의 정확도를 향상시키고 있습니다. 이를 통해, 시각적으로 유사하지만 지리적으로 멀리 떨어져 있는 샘플을 효과적으로 구별할 수 있는 방법을 개발했습니다. 이 전략은 GeoCLIP 모델에 통합되어 OSV5M 데이터셋에서 평가되어 성능 향상을 입증하였습니다.

- **Technical Details**: 제안된 방법은 세미변량도(samivariogram)를 사용하여 샘플 간의 공간 상관 관계를 모델링합니다. 이를 통해 이미지의 특성 공간 내에서의 거리와 지리적 거리 간의 관계를 정의하여 시각적 유사성의 기대치가 공간의 상관 관계에 따라 변화한다는 것을 캡처합니다. GeoCLIP는 정적인 CLIP 이미지 인코더와 위치 인코더로 구성되어 있으며, 이미지와 위치 정보를 통해 특징을 효과적으로 추출합니다.

- **Performance Highlights**: 제안된 공간 정규화 대조 학습 전략은 특히 미세한 세분화에서 이미지 기반 위치 추정 성능을 크게 개선했습니다. 이 접근법은 과거의 방법들이 놓치고 있던 지리적 현실을 모델에 통합하여, 잘못된 부정 샘플(false negatives)과 어려운 부정 샘플(hard negatives)을 구별하는 데 핵심적인 역할을 합니다. 연구 결과, 지역별 이미지의 시각적 유사성과 지리적 거리가 시스템적으로 연관되어 있음을 보여 주었습니다.



### No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models (https://arxiv.org/abs/2509.21565)
- **What's New**: 이 논문은 대규모 diffusion 모델을 위한 새로운 훈련 전략으로 Linear SEParability (LSEP)를 제안합니다. LSEP는 외부 인코더나 표현 정렬 없이 중간 계층의 표현을 선형으로 분리 가능하게 훈련시켜, 모델의 학습 효율성과 생성 품질을 동시에 개선할 수 있습니다. 특히, 훈련 중에 직접 선형 프로브(linear probe)를 삽입하여 학습의 역동성과 표현 품질을 향상시키는 방식을 사용합니다.

- **Technical Details**: 이 연구의 주요 혁신은 LSEP를 통한 훈련 정규화 전략이며, 이는 명시적으로 선형 분리를 개선하는 두 가지 목표, 즉 초기사형(hidden representations)의 분리 가능성과 denoising 목적을 통합합니다. 구체적으로는 새로운 훈련 기법들을 통해 선형 프로브의 분류 조건을 다르게 적용하고, 패치 수준의 선형 분리를 향상시키기 위해 무작위 자르기(random cropping)를 활용했습니다. 결과적으로, LSEP는 SiT와 같은 흐름 기반 변환기 아키텍처에서 큰 학습 효율성과 생성 품질 향상을 이끌어냈습니다.

- **Performance Highlights**: LSEP를 적용한 SiT-XL 모델은 기존 SiT-XL 모델보다 더 낮은 FID를 더 빠르게 수렴합니다. FID는 256x256 해상도에서 ImageNet 데이터셋에 대해 1.46로 최적의 성능을 기록하였으며, 이는 외부 인코더에 의존하지 않고도 이루어진 성과입니다. 또한 이러한 방법은 기존의 정렬 기반 접근 방식과 상호작용하여 표현의 선형 분리를 더욱 향상시키며, 최종적으로 훈련 효율성과 생성 성능을 더 끌어올립니다.



### Unsupervised Defect Detection for Surgical Instruments (https://arxiv.org/abs/2509.21561)
- **What's New**: 이 논문은 수술 도구의 품질 보증을 위해 기존의 비지도 결함 탐지 방법을 적응시키는 새로운 방식과 그에 따른 도전 과제를 다룹니다. 해당 방법은 백그라운드 마스킹(Background Masking), 패치 기반 분석(Patch-Based Analysis), 저순위 적응(Low-Rank Adaptation) 기술을 이용하여, 수술 기구 이미지에서 미세 결함을 신뢰성 있게 탐지할 수 있도록 합니다.

- **Technical Details**: 제안한 방법은 모델에 구애받지 않으며 다양한 기초 모델들과 통합될 수 있습니다. 특정 수술 도구의 요건을 고려하여, 결함 탐지가 미비한 문제세 가지(백그라운드 간섭, 미세 결함의 발견 어려움, 도메인 변화)를 해결합니다. SAM(Segment Anything Model)을 활용하여 수술 도구를 정확히 분할하고, 패치 기반 처리를 통해 결함 탐지의 민감성을 향상시킵니다.

- **Performance Highlights**: 이 연구는 두 가지 비지도 결함 탐지 방법(Dinomaly, DRAEM)을 평가하고, 만약 이를 조정하여 수술 도구 검사를 위한 효과적인 도구로 전환할 수 있음을 보여줍니다. 제공된 데이터셋과 방법론을 활용하여, 정확한 스팟 검출 및 정밀한 테스팅이 가능해짐을 강조하였습니다. 이 방법은 임상 품질 보증의 신뢰성을 높이는데 기여할 것으로 기대됩니다.



### X-CoT: Explainable Text-to-Video Retrieval via LLM-based Chain-of-Thought Reasoning (https://arxiv.org/abs/2509.21559)
Comments:
          12 pages, 7 figures. Accepted at EMNLP 2025 (Main Conference)

- **What's New**: 이번 연구는 X-CoT라는 새로운 설명 가능한 검색 시스템을 제안합니다. 이는 기존의 embedding 모델 기반 유사도 순위를 LLM 기반의 판단으로 대체하여 텍스트-비디오 검색을 개선합니다. 또한, 기존의 벤치마크 데이터셋에 추가 비디오 주석을 통해 데이터의 품질을 향상시키고 편향을 줄였습니다. X-CoT는 텍스트-비디오 관련 데이터를 분석하고, 모델 행동을 평가하며 랭킹 결과에 대한 명확한 설명을 제공합니다.

- **Technical Details**: X-CoT 시스템은 LLM의 체인-오브-생각(Chain-of-Thought) 추론을 통해 텍스트-비디오 검색을 수행합니다. 먼저, 비디오와 텍스트 각각에 대한 embedding을 생성하고, 이후 코사인 유사도를 계산하여 순위를 매깁니다. 그러나 전통적인 접근방식인 코사인 유사도만으로는 결과의 논리를 이해하기 어렵기 때문에, X-CoT는 쌍 비교 단계를 통해 세분화된 추론을 제공합니다.

- **Performance Highlights**: 실험 결과, X-CoT는 embedding 모델 기반 시스템에 비해 현저한 성능 향상을 보여줍니다. 또한, X-CoT는 텍스트-비디오 데이터의 품질을 분석하는 데 유용하며, 검색의 신뢰성을 높이는 데 기여합니다. 이러한 결과는 X-CoT가 기존의 텍스트-비디오 검색 시스템을 보완할 수 있는 강력한 도구임을 보여줍니다.



### Learning GUI Grounding with Spatial Reasoning from Visual Feedback (https://arxiv.org/abs/2509.21552)
- **What's New**: 본 논문에서는 GUI grounding을 인터랙티브 서치(task)로 재구성하여, Vision Language Models (VLMs)가 고해상도 GUI 이미지에서 효과적으로 UI 요소를 찾을 수 있도록 개선하는 방법을 제시합니다. 새로운 접근 방식인 GUI-Cursor는 커서를 움직이며 목표 객체를 찾는 과정에서 시각적 피드백을 제공하여 정확성을 높입니다.

- **Technical Details**: GUI grounding은 사용자가 자연어로 지시한 내용을 다양한 GUI 상호작용 단계로 변환하는 과정으로, 이 모델은 reinforcement learning(RL)을 활용하여 훈련됩니다. 이 과정은 모델이 커서를 특정 UI 요소로 이동시키며 공간(Spatial) 관계를 평가하고, 이전 이동 기록을 기반으로 새로운 위치를 예측하는 반복적인 방식으로 이루어집니다.

- **Performance Highlights**: GUI-Cursor는 ScreenSpot-v2 및 ScreenSpot-Pro 벤치마크에서 기존 최고 성과를 초월하며, 각각 93.9% 및 56.5%의 정확도를 기록했습니다. 특히, 95%의 사례에서 두 단계 내에 문제를 해결하는 능력을 보였으며, 어려운 작업에서는 더욱 많은 단계를 수행하는 적응력을 보여주었습니다.



### Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Moderation (https://arxiv.org/abs/2509.21486)
- **What's New**: 이 논문에서는 짧은 동영상 플랫폼에서 부적절한 콘텐츠를 탐지하기 위한 새로운 방법을 제안합니다. 기존의 작은 분류 모델 대신, 통합된 부적절한 콘텐츠 감지를 위한 Reasoning-enhanced Multimodal Large Language Model (MLLM)을 활용합니다. 이를 위해 세 가지 사전 훈련 작업인 Caption, Visual Question Answering (VQA), Chain-of-Thought (CoT)을 도입하여 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: MLLM을 활용하여 짧은 동영상 콘텐츠의 배포 특성에 맞춰 사전 훈련을 진행하며, 데이터 분포의 차이와 복잡한 문제 정의를 해결합니다. Caption 작업은 영상의 세부 사항을 향상시키고, VQA 작업은 주석 지침의 깊은 이해를 도와주며, CoT 작업은 체계적인 추론 능력을 강화합니다. 이러한 통합 접근 방식은 모델의 성능을 제고하며, 제로샷(zero-shot) 및 지도급기(Supervised Fine-Tuning, SFT) 설정에서도 좋은 성과를 보입니다.

- **Performance Highlights**: 실험 결과, 제안된 MLLM 프레임워크는 기존 모델에 비해 성능이 크게 향상되었습니다. 특히, 새로운 문제에 대한 일반화 능력도 뛰어나며, 낮은 감독 하에서도 강력한 성능을 발휘하는 것이 특징입니다. 이러한 성과는 다양한 모델 규모에서의 차별화된 효과를 통해 입증되었습니다.



### Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models (https://arxiv.org/abs/2509.21466)
- **What's New**: 이번 연구는 현대의 Text-to-Image 인공지능 모델이 사우디 아라비아에서의 전문 직종을 묘사할 때 성별 고정관념과 문화적 부정확성을 얼마나 지속하는지를 조사하였습니다. 1,006개의 이미지를 분석하여, 56개의 다양한 사우디 직업에 대해 중립적인 프롬프트(prompts)를 사용했습니다. 이 연구는 기존 모델들이 어떻게 사회적 편향(social biases)을 반영하는지를 드러냅니다.

- **Technical Details**: 연구에서는 ImageFX, DALL-E V3, Grok의 세 가지 인공지능 모델로 생성된 이미지를 각각 평가하였습니다. 두 명의 훈련된 사우디 평가자가 성별, 의상(wardrobe) 및 외모, 배경 및 설정, 활동 및 상호작용, 나이의 다섯 가지 기준으로 이미지를 평가했고, 세 번째 연구자가 불일치할 경우 중재하였습니다. 이러한 과정을 통해 10,100개의 개별 판단을 확보하였습니다.

- **Performance Highlights**: 결과는 성별 불균형이 뚜렷하게 나타났으며, 각 모델의 출력 결과는 ImageFX가 85% 남성, Grok가 86.6% 남성, DALL-E V3는 96% 남성으로 나타났습니다. 특히 DALL-E V3가 가장 강한 성별 고정관념을 보였으며, 이러한 불균형은 리더십 및 기술직에서 특히 두드러졌습니다. 문화적 부정확성도 확인되었고, 이는 꼭 진보적인 묘사라기보다 문화적 오해로 인한 것임을 보여줍니다.



### Residual Vector Quantization For Communication-Efficient Multi-Agent Perception (https://arxiv.org/abs/2509.21464)
Comments:
          5 pages

- **What's New**: 본 논문에서는 다수의 에이전트가 정보를 공유하여 장면 이해도를 향상시키는 Multi-agent collaborative perception (CP)을 위한 새로운 방법인 ReVQom을 제안합니다. ReVQom은 중간 특징을 압축하면서도 공간적인 정체성을 유지하는 학습된 특징 코덱입니다. 이 방법은 연결된 자율 주행 차량, 무인 항공기, 로봇 등에서의 사용을 염두에 두고 개발되었습니다.

- **Technical Details**: ReVQom은 단순한 bottleneck 네트워크를 사용하여 특징 차원을 압축하고, 다단계 잔차 벡터 양자화(multi-stage residual vector quantization, RVQ)를 통해 이러한 압축을 진행합니다. 이 과정에서 전송되는 데이터는 픽셀당 코드 인덱스만 포함되며, 이는 32비트 float 특징의 압축되지 않은 상태에서 초당 8192 비트(bpp)에서 6~30 bpp로 줄여 Accuracy 손실을 최소화합니다.

- **Performance Highlights**: ReVQom은 DAIR-V2X 실제 CP 데이터셋에서 30 bpp에서 273배 압축을 달성하고, 6 bpp에서는 1365배 압축을 기록했습니다. 18 bpp에서 ReVQom은 원본 특징 CP와 동급 이상의 성능을 보이며, 6-12 bpp에서는 초저대역폭(ultra-low-bandwidth) 작동을 가능하게 하여 점진적인 성능 저하를 견딜 수 있습니다. 이는 효율적이고 정확한 다수 에이전트 협력 인식을 가능하게 하여 실제 V2X 배치에 한 걸음 다가설 수 있게 합니다.



### VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding (https://arxiv.org/abs/2509.21451)
Comments:
          Work in progress

- **What's New**: 이 연구에서는 VideoJudge라는 새로운 다중 모달 대형 언어 모델(MLLM)을 제안합니다. VideoJudge는 3B 및 7B 크기로 비디오 이해 모델의 출력을 평가하는 데 특화되어 있습니다. 기존의 평가 지표들이 인간의 미세한 판단을 포착하지 못하는 데 반해, VideoJudge는 비디오 기반의 텍스트 응답을 직접 평가할 수 있는 가능성을 보여줍니다. 이는 비디오 이해 작업에 중심을 두고 대형 언어 모델을 활용한 첫 시도로 여겨집니다.

- **Technical Details**: VideoJudge는 생성자(generator)와 평가자(evaluator) 간의 상호작용을 기반으로 훈련됩니다. 평가자 모델이 예상과 다른 평가를 내릴 경우 해당 응답은 버려지며, 이 과정에서 훈련 데이터가 자동으로 생성됩니다. 또한, VideoJudge는 점수 예측과 함께 사례별 고유한 채점 기준(instance-specific rubrics)을 생성할 수 있는 기능을 갖추고 있습니다. 이러한 시스템은 대규모로 성과를 높이기 위한 첫 번째 단계로 작용합니다.

- **Performance Highlights**: VideoJudge-7B는 다른 대형 MLLM 평가 모델인 Qwen2.5-VL보다 세 가지 메타 평가 벤치마크에서 뛰어난 성능을 보였습니다. 특히, LLM 모델들이 MLLM 모델보다 효과적이지 않으며, 비디오 입력의 제공이 비디오 이해 작업의 평가에 있어 필수적임을 발견했습니다. VideoJudge는 인간의 평가와 더 높은 상관성을 보이며, 높은 샘플 효율성을 입증했습니다.



### DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation (https://arxiv.org/abs/2509.21433)
- **What's New**: 이 논문에서는 텍스트-이미지 확산 모델의 다중 개념 지우기(multi-concept erasure)에서 기존 방법의 한계를 극복하기 위한 새로운 프레임워크인 DyME를 제안합니다. 기존 방식들은 정적 지우기(static erasure)에 의존하여 복잡한 개념 요청을 처리하는 데 한계를 보였으며, 이를 해결하기 위해 개념별 LoRA 어댑터를 훈련하여 필요에 따라 동적으로 조합하는 방식을 채택했습니다. DyME는 더욱 유연하고 효과적인 다중 개념 지우기를 가능하게 하며, 이를 통해 법적 및 윤리적 문제를 완화할 수 있습니다.

- **Technical Details**: DyME는 각 개념에 특정한 경량의 LoRA 어댑터를 훈련하여, 추론(inference) 시 필요한 어댑터만 동적으로 조합하여 사용합니다. 이 프레임워크는 정적 방식과는 달리, 훈련과 추론을 분리하여 특정 세대에 필요한 개념만 선택적으로 지웁니다. 또한, 두 가지 수준(기능 및 매개변수)에서 직교성 제약(bi-level orthogonality constraints)을 도입하여 여러 어댑터 간의 간섭(crosstalk)을 최소화하고, 신뢰성 있는 다중 개념 지우기를 실현합니다.

- **Performance Highlights**: ErasureBench-H와 표준 데이터셋(CIFAR-100, Imagenette)에서의 실험 결과, DyME는 기존 방식보다 우수한 성능을 발휘하였으며, 90% 이상의 조화 정확도(harmonic accuracy)를 기록했습니다. 특히, 지우기 범위가 확장되더라도 DyME는 모든 기준선보다 명확한 우위를 유지하는 것으로 나타났습니다. 이 연구는 다중 개념 지우기의 확장성을 체계적으로 조사한 최초의 사례로, 실용적인 평가 방법을 제공합니다.



### QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models (https://arxiv.org/abs/2509.21420)
- **What's New**: 본 논문에서는 QuadGPT라는 최초의 자가 회귀(autoregressive) 프레임워크를 소개합니다. 이 프레임워크는 사각형 메쉬(quadrilateral meshes)를 직접 생성하는 엔드 투 엔드(end-to-end) 방식으로, 혼합된 지오메트리(topologies)를 처리하는 통합 토큰화(tokenization) 방법과 개선된 생성 품질을 위한 특화된 강화 학습(fine-tuning) 방법인 tDPO를 특징으로 합니다. 기존의 메쉬 생성 방식이 삼각형 메쉬를 기반으로 하는 데 비해, QuadGPT는 사각형 메쉬의 우수한 토폴로지(topology) 품질을 제공합니다.

- **Technical Details**: QuadGPT는 입력으로 포인트 클라우드(point cloud)를 사용하고, 구조화된 면(face) 시퀀스를 출력합니다. 이 모델은 삼각형과 사각형이 혼합된 메쉬를 명시적으로 지원하기 위해 특별하게 설계된 패딩(padding) 전략을 통해 새로운 통합 표현(unified representation)을 도입합니다. 처리 효율성을 위해, 우리는 얼굴 시퀀스를 압축하고 정점(vertex) 정보를 후속적으로 압축하는 아워글라스(Hourglass) Transformer 아키텍처를 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과, QuadGPT는 기존의 삼각형-사각형 변환 파이프라인보다 기하학적 정확도와 토폴로지 품질 모두에서 크게 향상된 성능을 보였습니다. 특히, QuadGPT는 게임 제작에 적합한 메쉬를 생성하며, 복잡한 지오메트리의 변화에도 높은 성능을 발휘합니다. 이 연구는 본격적인 사각형 메쉬 생성의 새로운 기준을 설정하고, 대규모 자가 회귀 모델과 토폴로지 인식 강화 학습의 결합 가능성을 보여줍니다.



### Overview of ExpertLifeCLEF 2018: how far automated identification systems are from the best experts? (https://arxiv.org/abs/2509.21419)
Comments:
          11 pages, 2 figures, CLEF 2018 Conference and Labs of the Evaluation Forum, September 10 to 14, 2018, Avignon, France

- **What's New**: 자동화된 식물 및 동물 식별이 최근 몇 년 동안 크게 개선되었습니다. 이 논문은 LifeCLEF 2018 ExpertCLEF 챌린지를 통해 인간 전문가와 자동화 시스템 간의 성능을 비교하는 것을 목표로 했습니다. 연구 결과, 최신 딥러닝 모델의 성능이 고급 인간 전문가의 수준에 가까워졌음을 확인했습니다.

- **Technical Details**: ExpertCLEF 챌린지는 19개 딥러닝 시스템과 9명의 프랑스 식물 전문가를 평가했으며, 훈련 데이터로는 이전 PlantCLEF 챌린지에서 사용했던 데이터셋이 포함되었습니다. 테스트 세트는 다른 전문가들이 현장에서 식별한 이미지로 구성되었고, 신뢰할 수 있는 데이터와 노이즈가 포함된 데이터로 나뉘어 참가자들에게 제공되었습니다. AutoML은 이 두 데이터 세트에서 성능을 비교하여 배운 효과를 평가했습니다.

- **Performance Highlights**: CMP 팀은 88.4%의 top-1 정확도로 챌린지에서 가장 우수한 성과를 기록했습니다. 이들은 여러 개의 Convolutional Neural Networks (CNNs) 모델을 앙상블하여 사용했으며, 클래스 불균형 문제를 해결하기 위해 Expectation Maximization 알고리즘을 적용했습니다. 연구 결과, 신뢰할 수 있는 데이터와 노이즈가 있는 데이터 모두에서 학습하여 모델 성능을 최적화하는 전략이 효과적임을 보여주었습니다.



### JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation (https://arxiv.org/abs/2509.21401)
- **What's New**: 본 논문에서는 Vision-Language Models (VLMs)의 취약성을 공격하기 위해 새로운 방식의 jailbreaking 공격인 JaiLIP을 제안합니다. JaiLIP은 공격자가 정의한 목표에 대한 유해한 출력을 생성하도록 모델을 유도하는 손실 기반 최적화를 통해 이미지 공간에서 조합된 MSE 손실과 모델 손실을 최소화하는 방법입니다. 이 방법은 기존의 방식보다 더 눈에 잘 띄지 않으면서도 효과적인 적대적(adversarial) 이미지를 생성하는 성능을 보입니다.

- **Technical Details**: JaiLIP은 이미지 공간에서의 조합된 MSE 손실과 모델 손실을 최소화하는 최적화 프레임워크로 구성되어 있습니다. 특히, tanh 기반의 재파라미터화를 적용하여 픽셀 수준의 수정을 제한하며, 원본 이미지와의 유사성을 유지하게 합니다. 이를 통해 고유한 유해한 텍스트를 생성하도록 VLM을 유도할 수 있는 공격을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안한 JaiLIP 방법은 Perspective API와 Detoxify의 표준 독성 지표를 사용해 기존 방식보다 유해한 이미지를 생성하는 데 있어 뛰어난 성능을 발휘했습니다. 특히, 교통 분야에서도 평가하여 특정 도메인에서의 공격의 실제적인 적용성을 입증하였습니다. JaiLIP의 효과성은 높은 공격 성공률을 달성하면서도 적은 시각적 간섭을 유지하는 데 기여합니다.



### Downscaling climate projections to 1 km with single-image super resolution (https://arxiv.org/abs/2509.21399)
- **What's New**: 이번 연구는 기존의 저해상도 기후 예측을 1km로 다운스케일링하는 새로운 방법론을 제시합니다. 일반적으로 사용되는 저해상도 기후 데이터의 한계를 극복하기 위해 single-image super-resolution 모델을 활용하여 기후 예측의 공간 해상도를 높입니다. 특히, 고해상도 데이터가 없어도 관측 데이터를 통해 모델을 학습시키는 혁신적인 접근을 보여줍니다.

- **Technical Details**: 이 연구에서는 EDRS, FNO, SwinIR와 같은 고급 single-image super-resolution 모델을 사용합니다. 저해상도 관측 데이터를 훈련 데이터로 활용하여 기후 예측의 고해상 버전을 생성합니다. 이러한 통계적 다운스케일링 방법은 머신 러닝 모델을 사용하여 저해상도 데이터와 고해상도 관측 데이터 간의 관계를 학습합니다.

- **Performance Highlights**: 실험 결과, 단일 이미지 초해상도 모델이 저해상도 기후 예측의 오류를 증가시키지 않고도 하루 평균 온도를 효과적으로 다운스케일하는 데 성공했습니다. 기후 지수 기반의 평가 방법을 통해 전통적인 고해상도 기후 데이터가 없더라도 모델의 성능을 평가할 수 있었습니다. 이는 기후 예측의 실용성을 높이는 데 기여할 것으로 기대됩니다.



### Skeleton Sparsification and Densification Scale-Spaces (https://arxiv.org/abs/2509.21398)
- **What's New**: 이번 연구에서는 해밀턴-야코비 스켈레턴(Hamilton-Jacobi skeleton), 즉 중간 축(medial axis)을 기반으로 하여 새로운 스켈레톤화(scale-space) 이론을 제안합니다. 이는 노이즈에 대한 민감성을 해결하기 위해 스켈레톤의 불필요한 가지를 제거하는 기존의 가지치기(pruning) 기법을 넘어서서, 위계적(hierarchical) 형태 단순화를 가능하게 합니다. 제안된 이론은 중간 축의 희소화(sparsification)와 밀접하게 연결되어 있으며, 이는 다양한 실용적인 응용에도 기여할 수 있는 과정을 포함합니다.

- **Technical Details**: 연구는 원래 스켈레톤에서 시작하여 더 세밀한 스켈레톤 스케일에 도달하는 밀집화(densification) 과정을 포함하여, 기존의 이론적 배경을 더욱 심화합니다. 이러한 기법은 포물선적(scale-space) 특성을 통해 위계 구조, 제어 가능한 단순화, 기하학적 변환에 대한 동등성을 자동으로 만족합니다. 이 새로운 스켈레톤화 이론은 정량적인 결과뿐만 아니라 3D 프린팅을 위한 전처리와 같이 다양한 응용 분야를 위한 근거를 제공합니다.

- **Performance Highlights**: 실험을 통해 스켈레톤의 견고화, 형태 압축 및 적층 제조(additive manufacturing)를 위한 강도 향상과 같은 다양한 작업에서 프레임워크의 효율성을 입증하였습니다. 이러한 결과는 이론의 타당성을 높이며, 스켈레톤화 및 희소화의 통합 개념이 실용적인 과제에서 어떻게 활용될 수 있는지를 보여줍니다. 또한, 연속적인 스켈레톤화 스케일 이론과 관련된 세부 사항을 구체화함으로써 실제 응용 가능성을 더욱 확대하고 있습니다.



### mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing (https://arxiv.org/abs/2509.21396)
- **What's New**: 이번 연구에서는 Integrated Sensing and Communication (ISAC) 시스템 내에서 인간 감지 연구를 지원하기 위한 개방형 mmWave 데이터셋 mmHSense를 제시합니다. 이 데이터셋은 제스처 인식, 인물 식별, 자세 추정 및 위치 결정을 포함한 다양한 최종 응용 프로그램을 위해 mmWave ISAC를 탐색하는 데 사용될 수 있습니다. mmHSense는 다양한 신호 특징을 포함하여, 데이터의 유용성을 특정 다운스트림 작업에서 유효성을 검증함으로써 입증합니다.

- **Technical Details**: mmHSense는 여러 mmWave ISAC 데이터셋으로 구성되어 있으며, 이는 COTS 기기와 사용자 정의 실험 플랫폼을 통해 수집된 데이터로, CSI(채널 상태 정보), beam SNR 및 PPBP(빔당 전력 비율)를 포함합니다. 데이터 수집은 여러 국가에서 다양한 사용자와 환경을 아우르는 방식으로 이루어졌으며, 이는 mmWave ISAC 데이터셋의 기존 부족을 해소하고 있습니다. 또한, 본 연구는 Low-Rank Adaptation(LoRA)을 사용하여 ISAC 모델을 효율적으로 미세 조정하는 방법도 제시합니다.

- **Performance Highlights**: 제안된 데이터셋은 기존 mmWave ISAC 실험의 제약을 극복하며, 사용자가 VR 게임과 상호작용하는 자연스러운 제스처를 캡처함으로써 더 진정한 자세와 동작을 포착합니다. 이 데이터셋은 다양한 신호 특징의 조직을 통해 고급 포즈 추정 및 멀티 센서 융합 기술을 지원하며, 5G mmWave OFDM 신호를 사용하여 제스처 인식을 위한 미래 6G 연구의 기초를 마련합니다. 데이터와 코드는 IEEE Dataport 및 GitHub를 통해 제공됩니다.



### Large AI Model-Enabled Generative Semantic Communications for Image Transmission (https://arxiv.org/abs/2509.21394)
Comments:
          Accepted to the IEEE GLOBECOM 2025

- **What's New**: 이 논문은 이미지 전송 효율성과 정확성을 향상시키기 위해 새로운 생성적 의미 통신 시스템을 제안합니다. 이 시스템은 이미지를 중요 영역과 비중요 영역으로 세분화하여 시멘틱(semantic) 정보의 정밀도를 높입니다. 주 영역은 이미지 지향 시멘틱 인코더로 처리하고, 비주요 영역은 텍스트 모델링을 통해 효율적으로 압축됩니다.

- **Technical Details**: 제안된 시스템은 이미지를 중요한 객체 및 장면 요소가 포함된 주 영역과 비주요 배경 지역으로 나누는 의미 인식 이미지 세분화 메커니즘을 핵심 혁신으로 합니다. 주 영역은 이미지 시멘틱 인코더를 통해 전송되고, 비주요 부분은 구조화된 텍스트 프롬프트로 인코딩되어 시멘틱 정보를 보존합니다. 노이즈가 포함된 무선 채널을 통해 전송된 인코딩된 신호는 역확산 프로세스를 통해 고충실도의 이미지를 복원하는 데 사용됩니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안한 시스템이 전통적인 방법에 비해 의미 충실도와 시각적 품질 모두에 있어서 우수한 성능을 보임을 보여줍니다. 특히 낮은 SNR(신호 대 잡음비) 조건에서 기존 DeepJSCC 접근 방식보다 월등히 나은 성과를 달성하였습니다. 실험을 통해 각 전송된 구성 요소가 최적의 시스템 성능을 달성하는 데 중요한 역할을 한다는 것을 확인했습니다.



### TUN3D: Towards Real-World Scene Understanding from Unposed Images (https://arxiv.org/abs/2509.21388)
- **What's New**: 본 논문은 실세계 이미지를 입력으로 사용하여 3D 객체 감지와 레이아웃 추정을 동시에 해결하는 TUN3D라는 새로운 방법을 제안합니다. 기존의 방법들과는 달리 TUN3D는 깊이 센서나 카메라 자세 정보 없이도 작동할 수 있으며, 경량의 sparse-convolutional backbone을 기반으로 두 개의 전용 헤드를 사용합니다. 이를 통해 기존의 포인트 클라우드 기반 방법의 한계를 극복하고, 효과적인 벽 표현(parametric wall representation)을 활용하여 보다 효율적인 장면 이해를 가능하게 합니다.

- **Technical Details**: TUN3D는 색상 포인트 클라우드 P를 입력으로 받아, 레이아웃과 3D 객체를 동시에 예측하는 모델로 구조화되어 있습니다. 3D 객체는 3D 바운딩 박스의 중심과 크기로 매개변수화되며, 레이아웃은 3D 좌표로 정의된 여러 벽의 집합으로 나타납니다. 우리의 모델은 fully-differentiable한 구조를 가지고 있으며, backbone, neck, 3D 객체 감지와 레이아웃 추정을 위한 두 개의 헤드로 구성되어 있습니다.

- **Performance Highlights**: TUN3D는 세 가지 도전적인 장면 이해 벤치마크(ground-truth 포인트 클라우드, 자세가 있는 다중 뷰 이미지, 자세가 없는 이미지)에 대해 최첨단 성능을 달성했습니다. TUN3D는 전문 3D 객체 감지 방법들과 동등한 성능을 보이는 동시에 레이아웃 추정에서 상당한 진전을 이루며, 실내 장면 이해 분야에서 새로운 기준을 설정합니다. 이 연구는 더 넓은 적용 가능성을 보이는 새로운 방향성을 제시합니다.



### Do Sparse Subnetworks Exhibit Cognitively Aligned Attention? Effects of Pruning on Saliency Map Fidelity, Sparsity, and Concept Coherenc (https://arxiv.org/abs/2509.21387)
Comments:
          4 pages

- **What's New**: 본 논문에서는 신경망 프루닝(pruning)이 모델의 해석 가능성에 미치는 영향을 조사합니다. 저수준 atribtion maps와 고수준 concept representations에 대한 두 가지 차원에서 분석을 수행하며, ResNet-18을 기반으로 한 실험을 통해 프루닝의 다양한 강도가 saliency maps와 모델의 개념 표현에 미치는 변화를 평가합니다. 이 연구는 프루닝이 내부 표현을 인간의 주의 패턴에 더 가깝게 형성할 수 있으나, 과도한 프루닝은 해석 가능성을 저해할 수 있음을 제시합니다.

- **Technical Details**: 연구에서는 활동 수준의 변화를 살펴보기 위해 ResNet-18 모델을 ImageNette 데이터셋에서 훈련 후 전역 크기 기반 프루닝(global magnitude pruning)을 적용하고 세부적으로 다양한 프루닝 수준에서의 saliency maps를 생성합니다.  Vanilla Gradients(VG)와 Integrated Gradients(IG)를 통해 post-hoc 설명이 어떻게 변화하는지를 정량적으로 종합하여 평가합니다. CRAFT 기반의 개념 추출 기술을 통해 프루닝 단계에서 학습된 개념의 의미 일관성의 변화를 추적하여 고수준 개념 분석을 수행합니다.

- **Performance Highlights**: 경량에서 중간 정도의 프루닝은 saliency map의 초점과 정확성을 향상시키며, 시맨틱적으로 의미 있는 개념을 보존합니다. 하지만 공격적인 프루닝이 시행되면 다양한 특징이 병합되어 saliency map의 희소성과 개념의 일관성이 줄어드는 반면, 정확성은 유지되는 경향을 보였습니다. 이러한 결과는 적절한 프루닝이 내부 표현을 보다 유사한 인간 주의 패턴으로 형성할 수 있음을 보여주며, 프루닝의 양과 해석 가능성 간의 미묘한 균형이 필요함을 강조합니다.



### ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data (https://arxiv.org/abs/2509.21386)
Comments:
          Accepted to OCEANS 2025 Great Lakes

- **What's New**: 이번 논문에서는 ShipwreckFinder라는 오픈소스 QGIS 플러그인을 소개합니다. 이 도구는 멀티빔 소나(multibeam sonar) 데이터를 활용하여 침몰선의 탐지를 자동화합니다. 침몰선은 해양 역사에서 중요한 유물로, 전통적으로 수작업으로 데이터를 검토하여 찾아냈으나 이 과정은 시간 소모적이었습니다. ShipwreckFinder는 이 과정을 자동화하며, 사용자에게 실시간으로 검출된 데이터를 시각화할 수 있는 기능을 제공합니다.

- **Technical Details**: ShipwreckFinder는 멀티빔 소나 센서를 통해 수집한 데이터로부터 침몰선 분할(segmentation) 모델을 학습하고 검증합니다. 이 도구는 QGIS와 통합되어 있으며, 다양한 해양 데이터 소스를 사용합니다. 데이터를 처리하여 3D 포인트 클라우드를 생성하고, 배경이 되는 심도 정보와 함께 적용하여 침몰선의 경계를 식별합니다. 또한, 합성 데이터 생성(synthetic data generation) 기법을 사용하여 훈련 데이터의 양과 다양성을 증가시킵니다.

- **Performance Highlights**: ShipwreckFinder는 오픈소스 도구로서 기존 딥러닝 기반 ArcGIS 툴킷과 전통적인 방법에 비해 우수한 분할 성능을 보여줍니다. 병렬로 다수의 데이터를 처리할 수 있어, 기존 방법보다 감지 속도와 정확도를 개선하였습니다. 이러한 접근은 해양 고고학 커뮤니티에서 최첨단 기계 학습 방법들에 대한 접근을 용이하게 합니다.



### Debugging Concept Bottleneck Models through Removal and Retraining (https://arxiv.org/abs/2509.21385)
- **What's New**: 본 논문에서는 Concept Bottleneck Models (CBMs)을 위한 새로운 디버깅 프레임워크를 제안합니다. 이 프레임워크는 Removal과 Retraining의 두 단계로 구성되며, 전문가의 피드백을 통해 모델이 원치 않는 개념에 대한 의존성을 줄일 수 있도록 합니다. CBDebug라는 새로운 방법론을 사용하여 전문가의 피드백을 샘플 수준의 보조 레이블로 변환하여 모델을 재학습시킵니다. 이 방법은 이전의 재학습 방법들보다 높은 성능을 보여줍니다.

- **Technical Details**: CBMs은 두 단계로 구성되어 있으며, 첫 단계에서는 개념 추출기가 인간이 이해 가능한 개념을 예측합니다. Removal 단계에서 전문가들은 제공된 개념 설명을 통해 원하지 않는 개념을 제거합니다. Retraining 단계에서는 CBDebug를 통해 샘플 수준의 보조 레이블을 생성하고, 이를 이용해 데이터셋에 대한 재가중치 및 증강 기법을 적용합니다. 이러한 과정을 통해 모델은 전문가의 사고 방식과 더 잘 일치하도록 개선됩니다.

- **Performance Highlights**: CBDebug는 다양한 CBMs 아키텍처(PIP-Net, Post-hoc CBM)와 알려진 허위 상관관계를 가진 데이터셋에서 검증되었습니다. 이 방법은 이전의 ProtoPNets에 대한 재학습 방법보다 더 효과적이며, 원래 모델에 비해 최악의 그룹 정확도를 최대 26% 향상시켰습니다. 자동화된 피드백이 제공될 경우에도 뛰어난 성과를 보여줍니다.



### Assessing the Alignment of Popular CNNs to the Brain for Valence Appraisa (https://arxiv.org/abs/2509.21384)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 CNN(Convolutional Neural Networks)이 사회인지, 특히 정서적 평가(valence appraisal)와의 관련성을 얼마나 잘 반영하는지를 탐구합니다. 기존의 연구들이 주로 일반적인 시각 지각에 집중한 반면, 본 연구는 고차원 인지 작업에 대한 CNN의 적합성을 평가하고 있습니다. 또한 Object2Brain이라는 새로운 프레임워크를 도입하여 CNN과 인간의 인지 간의 상관관계를 조사합니다.

- **Technical Details**: 연구는 CNN 아키텍처와 인간 뇌의 fMRI 데이터 간의 상관관계를 측정하는 상관 분석(correlation analysis)을 통해 수행됩니다. 또한, 정서적 평가를 위한 CNN의 층별 구조가 인간 뇌의 저수준(low-level), 중수준(mid-level), 고수준(high-level) 처리와 어떻게 비슷한지 평가합니다. 이 연구는 FindingEmo 데이터셋을 사용하여 총 24개 상관관계 목표(true labels 포함)를 대상으로 CNN 모델의 예측 성능을 비교합니다.

- **Performance Highlights**: 본 연구에서는 CNN 모델들이 단순한 시각 처리 이상으로 발전하지 못하며, 복잡한 사회적 정보를 처리하는 데 한계가 있음을 보여줍니다. 다양한 CNN 아키텍처들은 비슷한 상관관계 경향을 보였지만 각각 다른 객체 클래스에 대한 민감도(object class sensitivity)는 다름을 나타냈습니다. 이러한 결과는 CNN의 구조와 인간의 사회적 인지 메커니즘 간의 차이를 조명합니다.



### The LongiMam model for improved breast cancer risk prediction using longitudinal mammograms (https://arxiv.org/abs/2509.21383)
- **What's New**: 이 연구에서는 LongiMam이라는 새로운 심층 학습 모델을 개발하여 유방암 진단을 위한 선별검사에서 현재 및 최대 4회의 이전 유방 촬영 영상을 통합하여 활용하는 방법을 제시합니다. 이전의 방법들과 달리, LongiMam은 공간적 및 시간적 패턴을 포착하기 위해 컨볼루션 신경망(Convolutional Neural Network)과 순환 신경망(Recurrent Neural Network)을 결합하여 효과적인 예측 모델을 구현합니다. 이는 불균형한 결과 분포와 이질적인 추적을 고려하여 실제 임상 환경에서도 적응 가능성을 높였습니다.

- **Technical Details**: LongiMam 모델은 큰 인구 기반 스크리닝 데이터셋을 활용하여 학습되었으며, 각 유방에 대해 두 개의 시각적 이미지를 포함한 네 개의 이미지를 생성하는 검사 방식을 따릅니다. 모델은 시간에 따른 유방조직의 변화를 감지하고 예측하기 위해 데이터를 정리하고, 핵심 구성 요소로는 CNN, 게이티드 순환 유닛(Gated Recurrent Unit, GRU) 및 분류 모듈이 포함됩니다. 이 연구는 데이터를 증강하는 여러 기술을 사용하여 모델의 성능을 높이고, 각 환자의 이전 검사 정보를 체계적으로 구축했습니다.

- **Performance Highlights**: LongiMam은 여러 시나리오에서 이전 유방 촬영 자료를 포함했을 때 예측 성능이 일관되게 개선되었음을 보여주었습니다. 특히, 밀도가 높은 유방을 가진 여성 및 55세 이상의 고위험 그룹에서 모델의 효율성이 확인되었습니다. 반복적인 유방 촬영을 사용한 이번 연구는 유방암 예측을 강화하며, 스크리닝 프로그램에서 리스크 분류를 정교하게 할 수 있는 중요한 통찰을 제공합니다.



### Coreset selection based on Intra-class diversity (https://arxiv.org/abs/2509.21380)
- **What's New**: 이 논문은 딥러닝(Deep Learning) 모델의 훈련에서 코어셋(coreset)이라고 불리는 데이터의 부분집합을 선택하여 새로운 접근법을 제안합니다. 기존의 무작위 샘플링(random sampling) 방법이 가지는 대표성 결여와 클래스 불균형 문제를 해결하기 위해, 클래스 내 다양성(intra-class diversity)을 추출하여 더 효과적인 샘플링 방법을 개발하였습니다. 이 연구는 대표적인 생물의학 이미지 데이터셋인 PBC(peripheral blood cell)를 사용하여 제안된 방법의 효과를 입증합니다.

- **Technical Details**: 코어셋 선택 문제는 대량의 생물의학 데이터셋을 가진 의료 분야에서 특히 중요합니다. 본 연구에서는 랜덤 샘플링(Random Sampling) 대신에 인텔리전트 샘플링(Intelligent Sampling) 방법을 통해 클래스 내 클러스터를 형성하여 최종 샘플링을 수행합니다. 이렇게 함으로써 모델의 학습에 필요한 계산 자원과 시간 소모를 줄이고, 더 나은 성능을 제공할 수 있음을 보여주고 있습니다.

- **Performance Highlights**: 제안된 인텔리전트 샘플링 방법은 여러 성능 지표에서 랜덤 샘플링 방법을 초과하는 성능을 발휘하였습니다. 연구 결과는 DL 모델의 훈련에 있어 효율성과 계산 복잡성의 감소를 입증하며, 이는 의료 분야의 딥러닝 연구와 산업 응용에 큰 기여를 할 수 있음을 나타냅니다. 또한, 본 연구는 클래스 간 균형이 유지되는 상황에서도 훌륭한 성능을 보여주며, 향후 관련 분야의 연구에 기여할 것으로 기대됩니다.



### SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders (https://arxiv.org/abs/2509.21379)
- **What's New**: 본 논문에서는 텍스트-이미지 확산 모델에서 개념을 효과적으로 학습 해제하기 위한 새로운 방법인 SAEmnesia를 소개합니다. 이 접근법은 개념-신경세포 매핑을 체계적으로 라벨링하여 단일 신경세포가 하나의 개념에만 대응하도록 하여, 특징 분할(feature splitting)을 방지합니다. SAEmnesia를 통해 기능 중앙 집중화(feature centralization)를 달성하고, 더 나은 해석 가능성을 제공하여 모델이 생성하는 내용의 제어를 가능하게 합니다.

- **Technical Details**: SAEmnesia는 감독된 희소 오토인코더(sparse autoencoder) 훈련 방법으로, 훈련 과정에서 교차 엔트로피 계산(cross-entropy computation)이라는 최소한의 계산적 오버헤드를 추가합니다. 이 방법은 특화된 신경세포를 학습하여 미세조정 없이도 효과적인 개념 해제를 가능하게 하며, 이론적으로 각 개념을 단일 잠재 특성(latent feature)에 로컬라이즈하여 보다 정밀한 개입(intervention)을 수행합니다. 또한, TopK SAEs를 통해 희소성 제어(sparsity control)를 강화합니다.

- **Performance Highlights**: UnlearnCanvas 벤치마크에서 SAEmnesia는 최신 기법에 비해 9.22%의 성능 향상을 달성하였으며, 9개의 객체를 제거하는 연속 학습 해제(task)에서 28.4% 향상된 정확도를 보여주었습니다. 이 접근법은 추론 발생 시 하이퍼파라미터 검색을 96.67% 감소시켜, 전체적인 효율성을 크게 향상시킵니다. 이러한 성과는 SAEmnesia가 원치 않는 내용을 효과적으로 제거하면서도 생성 품질을 유지할 수 있음을 시사합니다.



### Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (https://arxiv.org/abs/2509.21377)
Comments:
          Main paper (8 pages). Accepted for publication by ECAI( European Conference on Artificial Intelligence) 2025

- **What's New**: 이번 연구에서는 Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (DMTF-AVN)이라는 새로운 프레임워크를 제안합니다. DMTF-AVN은 시각적 데이터를 더 깊이 있게 이해하고 오디오 신호와의 상호작용을 최적화하는 방식으로, 기존 방법들을 능가하는 성능을 보여주고 있습니다. 이 접근법은 멀티모달(Multimodal) 정보 융합에서 선택적인 정보 처리 방법을 통해, 로봇 navigation의 정확도를 개선합니다.

- **Technical Details**: DMTF-AVN 모델은 정보 추출, 멀티 타겟 특징 추출 전략, GRU(Gated Recurrent Unit), 액터-크리틱 네트워크 구조의 4가지 핵심 모듈로 구성되어 있습니다. 정보 추출 모듈은 원시 모달 데이터(raw modality data)를 고정 차원의 임베딩 벡터로 변환합니다. 여기서, GRU는 데이터 시퀀스의 시간적 동적을 캡처하여 의사결정 과정에 도움을 줍니다.

- **Performance Highlights**: DMTF-AVN은 Replica 및 Matterport3D 데이터셋에서 기존 방법들보다 더 뛰어난 성공률(SR), 경로 효율성(SPL), 장면 적응(SNA) 성과를 기록했습니다. 또한, 이 모델은 강력한 확장성 및 일반화 능력을 갖추고 있어 로봇 내비게이션에서의 고급 멀티모달 융합 전략에 대한 기반을 마련하고 있습니다. 제안된 방식은 멀티모달 데이터의 중복을 필터링하여 효과적인 정보 처리를 가능하게 합니다.



### In silico Deep Learning Protocols for Label-Free Super-Resolution Microscopy: A Comparative Study of Network Architectures and SNR Dependenc (https://arxiv.org/abs/2509.21376)
Comments:
          20 pages, 10 figures

- **What's New**: 이번 연구는 기존의 고가 장비나 전문 기술 없이도 초해상도(optical super-resolution) 광학 현미경을 달성할 수 있는 경제적인 대안을 제시합니다. 특히, 형광(fluorescent) 모드 대신 비형광(non-fluorescent) 위상 변조 현미경 방법인 Zernike phase contrast(PCM)와 differential interference contrast(DIC) 현미경을 활용합니다. 이를 통해 일반 사용자가 접근할 수 있는 새로운 초해상도 방법론이 제안됩니다.

- **Technical Details**: 연구에서는 O-Net과 Theta-Net이라는 두 가지 심층 신경망(deep neural network) 아키텍처를 개발하여, 원자 힘 현미경(atomic force microscopy)을 통해 보정된 나노스케일(nanoscale) 특성을 가진 커스텀 테스트 대상을 해상하는 능력을 평가합니다. 결과적으로 O-Net과 Theta-Net 모델은 높은 신호 대 잡음 비(signal-to-noise ratio, SNR)에서 다른 성능을 보이며, 서로 보완적(supplementary)인 접근법으로 사용될 수 있음을 보여줍니다.

- **Performance Highlights**: O-Net 모델은 높은 SNR에서 더 뛰어난 성능을 발휘하며, 반면 저가의 SNR에서는 Theta-Net 모델이 더욱 효과적인 경향을 보였습니다. 이러한 결과는 비형광 광학 나노스코피에서 DNN 모델의 아키텍처와 이미지 SNR의 중요성을 강조합니다. 동일한 교육 데이터셋(training dataset)과 에폭 수를 사용할 때에도 각 모델의 성능 차이는 여전히 뚜렷하게 나타났습니다.



### Automated Prompt Generation for Creative and Counterfactual Text-to-image Synthesis (https://arxiv.org/abs/2509.21375)
Comments:
          text-to-image generation, automatic prompt, DPO, Counterfactual

- **What's New**: 이 논문에서는 텍스트-이미지 생성 분야에서 정확성과 창의성을 높이기 위한 새로운 접근 방식인 자동 프롬프트 엔지니어링 프레임워크를 제안합니다. 특히 기존의 데이터셋 부족 문제를 해결하기 위해 'counterfactual size'에 초점을 맞추어 작은 객체가 큰 객체보다 더 크게 보이는 이미지를 생성하는 방법을 연구합니다. 이 프레임워크는 이미지 평가자, 프롬프트 재작성기, 프롬프트 랭커라는 세 가지 구성 요소로 이루어져 있으며, 이를 통해 더 나은 결과를 도출할 수 있습니다.

- **Technical Details**: 프레임워크의 핵심 요소 중 하나는 이미지 평가자로, 텍스트 프롬프트로부터 생성된 이미지가 'counterfactual size' 기준을 얼마나 잘 충족하는지를 평가합니다. 이 과정에서 세분화 마스크(Segmentation Mask)와 CLIP 모델을 활용하여 정확한 객체 검출과 레이블 확인(Verification)을 수행합니다. 프롬프트 재작성기는 감독 학습(Supervised Learning)을 통해 긍정적인 프롬프트 세트에 대해 학습하여 신뢰할 수 있는 카운터팩추얼 이미지 생성을 위한 후보 프롬프트를 만들고, 프롬프트 랭커는 직접 선호 최적화(Direct Preference Optimization) 방식으로 최상의 후보를 선택합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 최신의 텍스트-이미지 생성 자동 프롬프트 재작성 기법과 ChatGPT-4o를 능가하는 성능을 보였습니다. 특히, 새로운 'counterfactual size' 이미지 생성을 위한 데이터셋을 구축하고, 이를 통해 보다 정확한 결과를 얻어, 텍스트-이미지 생성의 창의성 및 탐색적 응용 분야에서의 가능성을 높였습니다. 이 연구는 앞으로의 카운터팩추얼 제어 가능성 연구의 기초를 다지는 중요한 계기가 될 것입니다.



### Safety Assessment of Scaffolding on Construction Site using AI (https://arxiv.org/abs/2509.21368)
- **What's New**: 본 논문에서는 건설 산업에서 안전성 평가의 중요성을 강조하며, 특히 비계(scaffolding)에 대한 인공지능(AI) 기반의 검사 방법을 제안합니다. 기존의 시각적 검사가 시간 소모적이며 인간의 오류에 취약한 문제를 해결하고자 합니다. 새로운 클라우드 기반 AI 플랫폼을 통해 비계 구조의 점 구름(point cloud) 데이터를 처리하고 분석하는 방법을 탐구합니다.

- **Technical Details**: 제안된 시스템은 인증된 참조 데이터와 최근 점 구름 데이터의 비교 및 평가를 통해 구조적인 수정사항을 탐지합니다. 이는 자동화된 모니터링을 가능하게 하여 수동 검사에 필요한 시간과 노력을 줄이며, 전체적인 안전성을 향상시키는 데 기여합니다. AI와 디지털화를 통해 더욱 정확한 검사 절차가 가능해질 것으로 기대됩니다.

- **Performance Highlights**: AI 기반 검사 시스템은 건설 현장에서의 안전성을 높이고, 수작업 검사보다 효율적인 모니터링을 지원합니다. 이러한 접근 방식은 건설자산의 신뢰성 및 작업자 안전성을 보장하는 데 중요한 역할을 할 것으로 나타났습니다. 비계의 구조적 변화 감지에 있어 실질적인 진전을 이루며, 보다 안전한 건설 환경 조성에 기여할 수 있습니다.



### MAJORScore: A Novel Metric for Evaluating Multimodal Relevance via Joint Representation (https://arxiv.org/abs/2509.21365)
- **What's New**: 이 논문에서는 MAJORScore라는 새로운 멀티모달 관련성 평가 지표를 제안합니다. 이는 기존의 이계(二界, bimodal) 데이터에 대한 사전 훈련된 대조 학습 모델의 임베딩(embedding) 능력을 기반으로 합니다. MAJORScore는 N개의 모달리티(N>=3) 간의 상관관계를 평가하기 위한 최초의 멀티모달 공동 표현(joint representation) 지표입니다.

- **Technical Details**: MAJORScore는 다양한 모달리티를 동일 잠재 공간(latent space)으로 통합하여 서로 다른 모달리티를 정확히 표현할 수 있는 기능을 가지고 있습니다. 이를 통해 공정한 관련성 점수(scoring)를 제공하고 모달리티 간의 유사성을 평가하는 데 필수적입니다. 기존의 평가 지표는 두 개의 모달리티 간의 상관관계 분석에만 적합하여 멀티모달 유사성 평가에 한계가 있었습니다.

- **Performance Highlights**: 대규모 멀티모달 데이터셋에서 MAJORScore는 기존 방법에 비해 일관성 있는 모달리티에서 26.03%-64.29% 향상되었고, 비일관성 모달리티에서는 13.28%-20.54% 감소하는 결과를 보였습니다. 이는 MAJORScore가 멀티모달 모델 성능 평가에서 더욱 신뢰할 수 있는 지표임을 입증합니다.



### A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision--Revised (https://arxiv.org/abs/2509.21363)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 기존의 salient object detection 기술이 갖고 있는 경계 불명확성과 복잡한 객체 구조 문제를 해결하기 위해 다중 과제(supervision) 훈련 전략을 제안합니다. 이 방식은 salient object detection, foreground contour detection, edge detection의 세 가지 작업을 동시에 활용하여 saliency maps를 개선합니다. 또한, 새로운 Mutual Learning Module (MLM)을 도입하여 여러 네트워크 분기를 상호 학습 구조로 훈련시키며 성능 향상을 꾀합니다.

- **Technical Details**: 제안된 방법은 VGG-16 백본(neural network backbone)을 기반으로 하는 인코더-디코더 아키텍처를 따릅니다. 네트워크는 서로 얽혀 있는 다중 작업(supervision) 학습을 통해 salient object detection과 foreground contour detection을 동시에 수행하여 보다 정확한 예측을 가능하게 합니다. 특히, edge module(EM)이 함께 설계되어 saliency feature의 노이즈를 감소시키고, 동시에 foreground contour detection 결과의 정확성을 높입니다.

- **Performance Highlights**: 제안된 방법은 7개의 도전적인 데이터셋에서 최첨단 성능을 달성했으며, 이전의 다른 모델들과 비교했을 때 상대적으로 훨씬 빠른 속도로 유사한 edge detection 성능을 보여주었습니다. 광범위한 실험 결과, 이 알고리즘은 다른 경쟁 모델에 비해 유의미한 성과를 얻었습니다. 또한, 다중작업 상호 학습 구조 덕분에 더욱 정교한 모습의 salient 지역을 생성할 수 있음을 입증했습니다.



### Multimodal Prompt Decoupling Attack on the Safety Filters in Text-to-Image Models (https://arxiv.org/abs/2509.21360)
- **What's New**: 최근 T2I (Text-to-Image) 모델들이 고화질 이미지 생성에 많이 활용되고 있지만, 이러한 모델이 악용되어 Not-Safe-for-Work (NSFW) 콘텐츠를 제작할 수 있다는 우려가 커지고 있습니다. 기존의 jailbreak 공격 방식은 주로 텍스트 프롬프트를 조작하는 데 초점을 맞추어 왔으나, 이미지 기반 입력에 대한 취약점은 충분히 다루어지지 않았습니다. 본 논문에서는 이를 해결하기 위해 MPDA (Multimodal Prompt Decoupling Attack)를 제안하며, 이 방법은 이미지 모달리티를 활용하여 원래의 안전하지 않은 프롬프트의 해로운 의미를 분리합니다.

- **Technical Details**: MPDA는 세 가지 주요 단계로 작동합니다. 첫 번째 단계에서 LLM (Large Language Model)은 안전하지 않은 프롬프트를 유사 안전 프롬프트와 해로운 프롬프트로 분리합니다. 두 번째 단계에서는 유사 안전 프롬프트로부터 기본 이미지를 생성하고, 해로운 프롬프트는 안전 필터를 우회하는 적대적 프롬프트로 변환되어 T2I 모델의 이미지 정제 과정을 안내합니다. 마지막으로, 생성된 이미지의 정확성을 확보하기 위해 비전-언어 모델(Vision-Language Model)이 생성된 이미지의 캡션을 분석하고, 이를 통해 LLM의 프롬프트 세분화 과정을 반복합니다.

- **Performance Highlights**: MPDA는 현재 상용화된 T2I 모델인 Wan-T2I, Cogview, MidJourney에서 평균 93%의 우회 성공률을 달성했습니다. 이는 기존의 텍스트 기반 공격 방법에 비해 훨씬 더 효과적인 결과로, T2I 모델의 다중 모달 안전 프로토콜에서의 취약점을 드러냅니다. 이 연구는 다중 모달 적대적 프롬프트 공격에 대한 방어 전략을 논의하며, 모델의 안전성 향상을 위한 참고자료를 제공할 수 있습니다.



### MDF-MLLM: Deep Fusion Through Cross-Modal Feature Alignment for Contextually Aware Fundoscopic Image Classification (https://arxiv.org/abs/2509.21358)
Comments:
          Word count: 5157, Table count: 2, Figure count: 5

- **What's New**: 본 연구에서는 망막 계측 이미지에서 질병 분류 정확도를 향상시키기 위해 정밀한 이미지 특징과 전반적인 텍스트 맥락을 통합하는 새로운 다중 모달 딥 러닝 아키텍처인 MDF-MLLM을 제안합니다. 기존의 다중 모달 대규모 언어 모델(MLLM)은 녹내장, 당뇨병성 망막병증, 망막 색소변성증과 같은 망막 질환 진단에 필수적인 저수준 공간 세부정보를 캡처하는 데에 어려움을 겪고 있습니다. 이 연구는 3개의 공개 데이터 세트(FIVES, HRF, StoneRounds)로부터 수집한 1,305개의 이미지-텍스트 쌍을 기반으로 모델 개발 및 검증을 진행했습니다.

- **Technical Details**: MDF-MLLM은 LLaMA 3.2 11B MLLM 내의 비선형 주의 블록에 U-Net 인코더 레이어의 네 가지 스킵 특징을 통합했습니다. 이를 통해 이미지 특성을 패치 단위로 투사하고, 크로스 주의 및 FiLM 기반 U-Net 변조를 통해 융합했습니다. 이 모델은 이중 유형 질병 분류 작업에서 기본 MLLM의 60% 정확도에 비해 94%의 정확도로 56% 향상을 보여주었습니다.

- **Performance Highlights**: MDF-MLLM 모델은 기본 모델에 비해 Recall과 F1-score가 각각 67%와 35% 향상되었음을 보여주었습니다. 다중 깊이 융합 접근 방식이 상속된 질병을 포함하여 공간적 추론 및 분류에서 상당한 이점을 제공한다는 것을 검증하였습니다. 이 연구는 임상 의사 결정 지원 시스템에 실제 배치 가능성이 높은 해석 가능한 모듈형 프레임워크를 제시합니다.



### Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports (https://arxiv.org/abs/2509.21356)
Comments:
          In proceedings MICCAI 2025

- **What's New**: 이 논문에서는 자동으로 생성된 흉부 방사선 보고서에서 오류를 감지하기 위한 새로운 문구 기반 사실 확인 모델(FC 모델)을 제안합니다. 이 모델은 발견물 및 각 위치의 오류를 식별하는 데 중점을 두고 있습니다. 특히, 통합된 데이터를 통해 사실과 허위 발견물-위치 쌍을 구성하여 보고서 오류를 시뮬레이션합니다.

- **Technical Details**: FC 모델은 다중 라벨 교차 모달 회귀 네트워크를 기반으로 하며, 2,700만 개 이상의 이미지를 조합하여 훈련됩니다. 이 훈련 과정은 기본적으로 발견물 위치 격리, 합성 데이터 생성, FC 모델 훈련 등의 단계로 구성됩니다. 모델은 보고서에서 추출된 발견과 이미지를 기반으로 입력을 받아 오류를 예측합니다.

- **Performance Highlights**: FC 모델은 다양한 흉부 X-레이 데이터 세트에서 발견 정확성과 위치 예측의 robust성을 보여주며, 0.997의 일치 상관 계수를 달성하여 지상 진실 확인과 높은 신뢰성을 입증하였습니다. 이러한 결과는 이 모델이 방사선 업무에서 임상 추론 단계에서 유용할 가능성을 시사합니다.



### KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cach (https://arxiv.org/abs/2509.21354)
- **What's New**: 이번 연구에서는 KV-Efficient VLA라는 메모리 압축 프레임워크를 제안하여 비전-언어-액션 (VLA) 모델의 효율성을 개선합니다. 이 접근법은 고유 사용 주제를 선택적으로 유지하면서 과거의 정보를 고정 크기 청크로 분할합니다. 결과적으로 이 모델은 빠른 추론 속도와 메모리 사용량을 줄일 수 있도록 설계되었습니다.

- **Technical Details**: KV-Efficient VLA는 키-값 (KV) 캐시를 고정 길이 청크로 나누고, 각 청크에 대해 순환 게이팅 모듈을 사용하여 과거의 문맥을 요약 및 필터링합니다. 이 시스템은 저차 근사 (LoRA) 기술을 이용해 사전 훈련된 LLaMA 모델과 통합되어 작동하며, 적은 비용으로 세밀한 조정을 수행합니다. 이러한 구조적 변화로 인해 실시간 로봇 제어에서 요구되는 저지연성을 유지하면서도 메모리 사용량을 대폭 줄일 수 있습니다.

- **Performance Highlights**: KV-Efficient VLA 모델은 평균적으로 1.21배 더 빠른 추론 속도를 달성하며, KV 메모리를 36% 줄이는 효과를 보여주었습니다. 이러한 성능 향상에도 불구하고, 작업 성공률에 미치는 영향은 최소화되어 있어서 다양한 환경에서 높은 정확도를 유지합니다. 솔루션은 기존의 VLA 시스템에 원활하게 통합될 수 있어, 실제 로봇 제어 논리에 어떠한 수정도 필요하지 않습니다.



### Improving Autism Detection with Multimodal Behavioral Analysis (https://arxiv.org/abs/2509.21352)
- **What's New**: 이번 연구는 자폐 스펙트럼 상태(ASC) 진단을 위한 비디오 데이터 분석을 통한 컴퓨터 지원 진단 방법을 제안합니다. 총 168명의 ASC 환자와 157명의 비자폐 환자가 포함된 데이터셋을 사용하여, 다양한 행동 지표를 통해 새로운 통계적 설명자를 도입하였습니다. 이를 통해 기존 모델들의 한계를 극복하고, 행동 지표 간 통합을 통한 분류 정확도를 향상시켰습니다.

- **Technical Details**: 研究에서 사용한 방법론은 Simulated Interaction Task (SIT) 패러다임을 기반으로 하며, 다양한 비언어적 특성을 추출하기 위해 OpenFace 2.2와 같은 오픈 소스 라이브러리를 활용했습니다. 눈 가는 방향, 얼굴 표정, 머리 움직임과 같은 여러 모달리티를 종합적으로 분석하였으며, 비디오 링킹으로 자동화된 절차를 통해 대화 상황을 평가했습니다. 또한, 새로운 통계적 설명자들은 시선 각도의 변동성을 수량화하여 시선 기반 분류 정확도를 개선하였습니다.

- **Performance Highlights**: 최종적으로, 이 연구는 여러 모달리티를 통합한 데이터 분석을 통해 74%의 분류 정확도를 달성하였으며, 이는 자폐 진단 도구의 효과성과 실용성을 높일 수 있는 가능성을 제시합니다. 결과적으로, 비디오 기반 스크리닝 도구의 확장 가능성을 보였으며, 자폐 평가에 있어 신뢰할 수 있는 도구로 자리 잡을 수 있음을 시사합니다.



### Random Direct Preference Optimization for Radiography Report Generation (https://arxiv.org/abs/2509.21351)
- **What's New**: 이번 연구는 Radiography Report Generation (RRG) 분야에서 기존 방법의 한계를 극복하기 위한 방법으로, Direct Preference Optimization (DPO) 프레임워크를 제안합니다. 이 방법은 임의의 대조적 샘플링을 활용하여 훈련 쌍을 구성하며, 추가적인 보상 모델이나 인간의 선호 주석이 필요하지 않습니다. 실험 결과, 세 가지 최첨단 모델에 우리의 Random DPO를 보완함으로써 최대 5%의 임상 성능 향상을 달성했습니다.

- **Technical Details**: RRG 모델은 전통적으로 다음 단어 예측을 위한 표준 크로스 엔트로피 손실 기준으로 최적화됩니다. 이러한 접근 방식은 실질적으로 모델이 잘못된 연관성을 학습하게 만들어 편향을 유발할 수 있습니다. 본 연구에서는 DPO를 기반으로 모델 성능을 향상시키기 위한 무작위 샘플링 방법을 제안하며, 이는 추가적인 데이터 준비 없이 기존 데이터셋을 활용하여 성과를 낼 수 있습니다.

- **Performance Highlights**: MIMIC-CXR, CheXpert Plus, IU X-ray, Interpret-CXR와 같은 네 개의 공개 데이터셋을 사용하여 실험을 진행했습니다. MIMIC-CXR는 377,110개의 이미지와 227,835개의 방사선 연구가 포함되어 있으며, CheXpert Plus는 223,228개의 독특한 방사선 보고서와 가슴 X선 쌍으로 구성되어 있습니다. 각 데이터셋은 다양한 임상 환경에서 수집되어 RRG 방법의 종합적인 평가를 허용합니다.



### See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation (https://arxiv.org/abs/2509.22653)
Comments:
          CoRL 2025. Project page: this https URL

- **What's New**: 본 연구에서는 See, Point, Fly (SPF)라는 새로운 훈련 없는 항공 비전-언어 내비게이션(AVLN) 프레임워크를 소개합니다. SPF는 비전-언어 모델(VLM)을 기반으로 하여 어떤 형태의 자유 형식 지시에도 강력하게 반응할 수 있도록 설계되었습니다. 특히, 기존의 VLM 기반 접근 방식들이 행동 예측을 텍스트 생성 작업으로 다룬 것과 달리, 우리는 AVLN을 2D 공간 기초 작업으로 간주하는 중요한 통찰력을 제공합니다.

- **Technical Details**: SPF는 모호한 언어 지시를 2D 웨이포인트(waypoints)로 변환하여 항공 유무인기(UAV)의 행동 명령으로 사용합니다. 이 과정에서 카메라 정보를 활용해 2D 웨이포인트를 3D displacement 벡터로 전환하게 됩니다. 또한, SPF는 주행 거리를 적응적으로 조정하여 내비게이션 효율성을 높이며, 폐쇄 루프(control) 방식으로 동적인 환경에서도 유동적인 목표를 추적할 수 있습니다.

- **Performance Highlights**: SPF는 DRL 시뮬레이션 벤치마크에서 이전 최고의 방법에 비해 63%의 절대적 개선을 보이며 새로운 최첨단 성과를 세웠습니다. 실제 환경 평가에서도 SPF는 강력한 기준선 모델들보다 큰 폭으로 성능이 우수했습니다. 또한, 다양한 VLM에 대한 일반화 능력도 뛰어난 것을 입증하였습니다.



### Pixel Motion Diffusion is What We Need for Robot Contro (https://arxiv.org/abs/2509.22652)
Comments:
          16 pages, 7 figures

- **What's New**: 본 논문에서는 DAWN(Diffusion is All We Need for robot control)이라는 통합된 diffusion 기반 프레임워크를 제안하며, 이를 통해 언어에 조건화된 로봇 조작을 가능하게 합니다. DAWN은 고수준의 동작 의도와 저수준의 로봇 행동 간의 연결을 구조화된 픽셀 모션 표현을 통해 실현합니다. 특히 DAWN은 최신 CALVIN 벤치마크에서 최첨단 성능을 달성했으며, MetaWorld에서의 효과도 추가로 검증되었습니다.

- **Technical Details**: DAWN은 고수준과 저수준의 컨트롤러를 diffusion 프로세스로 모델링하여 완전히 훈련 가능한 엔드-투-엔드 시스템을 제공합니다. 이 시스템은 해석 가능한 중간 모션 추상화를 생성하며, 다중 작업 성능이 뛰어납니다. 중간 픽셀 모션은 시각적 입력에 기반을 두어 해석 가능성을 부여하며, 분산 기반 정책 헤드를 통해 실행 가능한 행동으로 변환됩니다.

- **Performance Highlights**: DAWN은 제한된 데이터와 작은 모델 용량에도 불구하고 CALVIN, MetaWorld 및 실제 환경 벤치마크에서 경쟁력 있는 성능을 발휘합니다. 연구 결과에 따르면, 구조화된 픽셀 모션과 다양한 사전 훈련된 모델의 장점을 활용하여 데이터 효율성을 극대화함에도 불구하고 최첨단 vision-language action 모델과 동등하거나 그 이상으로 성능을 달성했습니다.



### VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing (https://arxiv.org/abs/2509.22651)
- **What's New**: 음성 중심의 AI 보조 도구에 대한 관심이 높아짐에 따라 VoiceAssistant-Eval이라는 포괄적인 벤치마크가 도입되었습니다. 이 벤치마크는 청취(listening), 발화(speaking), 시청(viewing) 능력을 평가하기 위해 13개 작업 카테고리에서 10,497개의 정제된 예시로 구성되어 있습니다. 다양한 작업에는 자연 소리, 음악 및 대화가 포함됩니다.

- **Technical Details**: VoiceAssistant-Eval은 개인화된 음성 모방, 자연스러운 핸즈프리 음성 상호작용, 다중 모드 비전-오디오 이해 및 고급 오디오 QA와 같은 4가지 대표 작업을 평가합니다. 비교를 통해 기존의 벤치마크는 특정 측면만을 커버하고 있는 반면, VoiceAssistant-Eval은 AI 보조 도구의 전체 범위를 종합적으로 테스트합니다. 또한, 역할별 말하기 스타일과 음색을 분석하여 개인화 상호작용의 잠재력을 입증합니다.

- **Performance Highlights**: 21개의 오픈 소스 모델과 GPT-4o-Audio 모델을 평가한 결과, 독점 모델이 항상 오픈 소스 모델보다 우수하지 않음을 보여주었습니다. 대부분의 모델이 발화 작업에서 뛰어난 성능을 보였으나 오디오 이해에서는 뒤처지는 경향을 나타냈습니다. 특히, 중간 크기의 Step-Audio-2-mini(7B)가 LLaMA-Omni2-32B-Bilingual보다 두 배 이상의 청취 정확도를 기록한 것이 주목할 만한 성과입니다.



### WoW: Towards a World omniscient World model Through Embodied Interaction (https://arxiv.org/abs/2509.22642)
- **What's New**: 본 논문은 WoW라는 새로운 생성형 세계 모델을 소개합니다. WoW는 14억 개의 매개변수로 구성되어 있으며, 200만 개의 로봇 상호작용 경로를 기반으로 학습되었습니다. 이 모델은 사람과 자율 평가 모두에서 최신 성능을 보이며, 물리적 인과관계와 충돌 역학에서 강력한 능력을 나타냅니다.

- **Technical Details**: WoW는 Vision Language Model (VLM)와 Diffusion Transformer (DiT) 구조적 패러다임인 SOPHIA를 기반으로 개발되었습니다. Flow-Mask Inverse Dynamics Model (FM-IDM)을 사용하여 에이전트의 상상을 물리적 현실로 연결하고, 이를 통해 픽셀 수준의 미래 예측을 실제 행동으로 변환합니다.

- **Performance Highlights**: WoW는 WoWBench라는 새로운 벤치마크에서 SOTA(최신 기술 수준) 성능을 달성하였으며, 특히 지시 이해에서 96.53%, 물리 법칙에서 80.16%의 정확도를 기록했습니다. 사용자 평가에서도 높은 일치도를 보여주며, 다양한 다운스트림 애플리케이션에서도 더 나은 결과를 나타냅니다.



### Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning (https://arxiv.org/abs/2509.22601)
Comments:
          26 pages, 11 figures

- **What's New**: 본 논문에서는 자기 모방 학습(Self-Imitation Learning, SIL)을 기반으로 한 새로운 강화 학습(RL) 접근 방식인 SPEAR(Self-imitation with Progressive Exploration for Agentic Reinforcement learning)를 제안합니다. 이 방법은 정책 엔트ropy를 동적으로 조절하여 탐색(exploration)과 활용(exploitation)의 균형을 유지합니다. 특히, SPEAR는 툴 사용 능력을 개발하기 위한 도움 보상을 포함하여, 훈련 초기 단계에서의 엔트로피 증가가 탐색을 촉진함을 강조합니다.

- **Technical Details**: 논문에서 제안하는 SPEAR는 스킬 기반 탐색을 지원하기 위한 인트린식 보상을 이용하며, 후속 단계에서는 기존의 성공적인 패턴을 활용하여 행동 수준의 탐색을 촉진합니다. 또한, Replay Buffer의 경험에 대한 이점 재조정과 높은 공분산의 클리핑 기법을 통해 정책 업데이트의 안정성을 강화하고 보상 해킹(reward hacking) 문제를 완화합니다. 이는 정책 엔트로피를 조절하여 훈련의 불안정성을 방지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: SPEAR는 GRPO, GiGPO, Dr.BoT와 같은 여러 기준 방법들에 비해 상당한 성능 향상을 보여주며, ALFWorld 및 WebShop 태스크에서 각각 최대 16.1%, 20.7%의 성능 개선을 기록했습니다. 또한, Dr.BoT의 성능을 각각 AIME24에서 3.8%, AIME25에서 6.1% 향상시킵니다. SPEAR는 낮은 계산 복잡도를 유지하며 다른 LLM 기반 에이전트들에 대해 뛰어난 호환성과 확장성을 보여주는 플러그 앤 플레이 알고리즘입니다.



### MINT-RVAE: Multi-Cues Intention Prediction of Human-Robot Interaction using Human Pose and Emotion Information from RGB-only Camera Data (https://arxiv.org/abs/2509.22573)
- **What's New**: 이 논문은 유비쿼터스 로봇과의 효과적인 인간-로봇 상호작용(HRI)을 위한 인간의 상호작용 의도를 예측하는 새로운 RGB-only 파이프라인을 제안합니다. 이전의 방법들은 RGB-D 방법을 사용하여 다중 모달 입력에 의존했지만, 본 연구는 일반 웹캠으로 단일 RGB 입력만으로도 프레임 수준의 정밀도로 의도를 예측할 수 있게 합니다. 이를 통해 로봇의 반응 속도를 높이고 서비스 품질을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서 제안하는 MINT-RVAE는 클래스 불균형 문제를 해결하기 위한 합성 시퀀스 생성 방법으로, 새로운 손실 함수 및 훈련 전략과 함께 사용됩니다. 모델은 다양한 환경에서 인간의 동작 궤적과 추론된 의도를 기반으로 훈련됩니다. 데이터셋은 개별 프레임에 대한 정확한 레이블링을 제공하며, 그것은 지금까지의 연구에서 이루어진 적이 없는 데이터셋입니다.

- **Performance Highlights**: 본 연구는 세 가지 백본 네트워크(GRU, LSTM, Transformer)를 사용하여 MINT-RVAE로 데이터 증강을 수행하며, 상태-of-the-art 성능(AUROC: 0.95)을 달성했습니다. 이전의 방법들(AUROC: 0.90-0.912)보다 뛰어난 성능을 보여주며, 로봇의 실시간 반응을 위한 프레임 수준의 예측이 가능하도록 합니다. 또한, 새로운 데이터셋은 연구자들에게 공개되어 더욱 다양한 연구가 가능해질 것입니다.



### Activation Function Design Sustains Plasticity in Continual Learning (https://arxiv.org/abs/2509.22562)
- **What's New**: 이번 연구는 연속 학습에서 활성화 함수가 플라스틱성 손실에 미치는 영향을 탐구합니다. 변화하는 데이터 환경에서 플라스틱성을 유지하기 위해 Smooth-Leaky와 Randomized Smooth-Leaky와 같은 새로운 비선형 활성화 함수를 제안하고 평가했습니다. 또한, 활성화 함수의 모양과 적응도 간의 관계를 파악하기 위한 간단한 스트레스 프로토콜을 제공하여 더 나은 활성화 함수 설계를 강조합니다.

- **Technical Details**: 연구에서는 활성화 함수가 기울기 정보의 첫 번째 관문으로 작용하며, 그 기울기와 포화 정도가 중요한 역할을 한다고 설명합니다. 각종 활성화 함수를 비교 분석하여 이들이 플라스틱성 손실을 어떻게 가중시키거나 완화할 수 있는지를 조명합니다. 특히, Leaky-ReLU와 같은 변종이 어떻게 비활성 유닛 문제(dormant unit problem)를 감소시킬 수 있는지를 논의합니다.

- **Performance Highlights**: 두 가지 새로운 활성화 함수는 연속 학습 및 변화하는 RL 환경에서 플라스틱성을 저하 없이 개선하는 데 기여합니다. 이 연구의 결과는 활성화 설계가 연속 학습에서의 적응성을 지속시키는 강력하면서 가벼운 방법임을 보여주며, 정보 손실을 최소화할 수 있는 잠재력을 됩니다. 기존 모델보다 더 좋은 성과를 보여주는 다양한 활성화 함수 성능을 정리하여 제시하고 있습니다.



### JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation (https://arxiv.org/abs/2509.22522)
- **What's New**: JointDiff는 연속적인 데이터와 이산적인 사건을 동시에 생성할 수 있는 새로운 확산 프레임워크입니다. 이 모델은 스포츠와 같은 복잡한 시스템에서 두 프로세스의 상호작용을 효과적으로 포착하여, 선수의 경로와 같은 연속적인 데이터와 패스와 점유와 같은 이산적인 사건을 동시에 모델링합니다. 이를 통해 기존의 분리된 모델링 접근 방식의 한계를 극복할 수 있습니다.

- **Technical Details**: JointDiff는 다중 에이전트의 경로와 동기화된 소유 사건을 동시에 생성하는 조화로운 연속-이산 확산 모델을 기반으로 합니다. 이 모델에서는 weak-possessor-guidance(WPG)와 text-guidance와 같은 새로운 제어 메커니즘을 통해 사용자가 경기를 유연하게 조정할 수 있는 방법을 제공하며, CrossGuid라는 효과적인 조건화 운영을 도입하여 제어 신호를 모델에 통합합니다.

- **Performance Highlights**: JointDiff는 스포츠 도메인에서 가장 진보된 성능을 보여줍니다. 새로운 통합 벤치마크와 함께 기존의 비제어 생성 및 제어 가능한 두 가지 새로운 임무에 대한 평가를 수행하여, 장면 수준의 일관성을 고려한 결과에서 뛰어난 성과를 입증했습니다. 이 접근 방식은 다중 에이전트 시스템의 새로운 현실적이고 제어 가능한 생성 모델을 구축하는 데 중추적인 역할을 합니다.



### Adaptive Dual-Mode Distillation with Incentive Schemes for Scalable, Heterogeneous Federated Learning on Non-IID Data (https://arxiv.org/abs/2509.22507)
- **What's New**: 본 논문은 Federated Learning(FL)의 여러 도전 과제를 해결하기 위해 새로운 접근 방식을 제안한다. 특히, 데이터를 다양한 고객의 요구와 자원에 맞춰 맞춤형으로 처리하고, 비정상적인 통계적 이질성을 관리하는 효율적인 방법을 모색했다. 제안된 접근 방식에는 모델 이질성을 다루기 위한 DL-MH와 클라이언트 참여를 유도하는 인센티브 기반 확장인 I-DL-MH가 포함되어 있다.

- **Technical Details**: FL은 분산 데이터에서 모델을 학습할 수 있도록 설계된 분산 학습 접근 방식으로, 클라이언트 장치에서 데이터를 중앙 집중화하지 않고 보호하는 방식이다. 이러한 접근 방식의 주요 목표는 고객의 데이터와 계산 자원을 활용하여 하나의 글로벌 모델을 훈련시키는 것이다. 그러나 고객 간의 모델 아키텍처 차이나 비정상적(non-IID) 데이터로 인해 이질적인 모델을 훈련하는 과정에서 효율성을 달성하기 어렵다.

- **Performance Highlights**: 제안된 방법을 다양한 실험 환경에서 평가한 결과, DL-SH는 글로벌 모델의 정확도를 153% 향상시켰고, I-DL-MH는 비정상적 데이터 조건하에서 225%의 성능 개선을 보였다. 제안된 접근 방식은 기존 방법들과 비교했을 때, 통계적 이질성과 모델 이질성을 효과적으로 다루면서도 통신 비용을 감소시키고 정확도를 높일 수 있음을 입증했다.



### Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss (https://arxiv.org/abs/2509.22394)
- **What's New**: 이번 연구에서는 의료 영상 번역을 위해 nnU-Net 프레임워크를 새로운 방식으로 적용했습니다. SynthRAD2025 대회에서 MR(Magnetic Resonance)에서 CT(Computed Tomography) 및 CBCT(Cone-Beam CT)로의 이미지 전환 작업에 초점을 맞추고 있습니다. 특히, Anatomical Feature-Prioritized (AFP) 손실 함수를 도입하여 임상적으로 중요한 구조의 재구성을 향상시켰습니다.

- **Technical Details**: 제안된 모델은 표준 U-Net과 잔여 U-Net(residual U-Net) 두 가지를 사용하며, 두 모델 모두 nnU-Net로부터 기인하였습니다. MRI 데이터는 개별적으로 z-score 정규화가 적용되었고, CBCT와 CT는 데이터셋 수준에서의 z-score 정규화와 클리핑을 통해 처리되었습니다. 훈련 데이터는 해부학적 영역에 맞춰 3D 패치로 구성되며, 1000 및 1500 에폭(epoch) 동안 훈련된 후, AFP의 미세 조정이 수행됩니다.

- **Performance Highlights**: 두 모델은 모든 영역에서 일관된 설계를 가능하게 하며, L1 및 AFP 손실을 활용한 경우 해부학적 신뢰성을 크게 향상시킨 것으로 평가되었습니다. 결과적으로 잔여 네트워크와 AFP의 조합은 뼈 구조와 병변 재구성의 질을 높이며, 의료 영상의 다중 모드 합성을 위한 안정적인 해결 방안을 제공합니다. 이를 통해 환자의 방사선 노출을 최소화하고, 전체 치료 계획 프로세스를 간소화 할 수 있습니다.



### RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation (https://arxiv.org/abs/2509.22356)
- **What's New**: 이 논문은 RoboView-Bias라는 로봇 조작에서의 시각적 편향(visual bias)을 체계적으로 정량화하기 위해 설계된 첫 번째 벤치마크를 제안합니다. 이 벤치마크는 변수 분리의 원리를 따르며, 2,127개의 과제 인스턴스를 생성하여 개인 시각적 요소에 의해 유도된 편향과 그 상호작용을 강력하게 측정할 수 있도록 합니다. 이러한 새로운 접근법은 의사결정 안정성(decision-making stability)에 대한 인식을 향상시키고, 로봇 시스템의 안전성을 보장하기 위한 토대를 마련합니다.

- **Technical Details**: RoboView-Bias는 시각적 방해 요소(Visual Perturbation)와 작업 맥락 일반화(Task Context Generalization)의 차원으로 변수를 분리하여 평가 인스턴스를 생성하는 구조화된 변형 생성 프레임워크(Structured Variant-Generation Framework, SVGF)를 활용합니다. 이 방법론은 특정 시각적 조건 하에서의 변동성과 불안정성을 간과하는 기존 메트릭의 한계를 극복하고, 색상 및 카메라 시점(camera viewpoint)과 같은 시각적 속성에서의 체계적 편향을 정량화하는 데 초점을 맞춥니다. 이로써, 로봇 시스템의 감지-결정 파이프라인을 따라 공정하고 분명한 비교 세트를 제공할 수 있게 됩니다.

- **Performance Highlights**: 세 가지 주요 발견이 보고되었습니다: 첫째, 모든 에이전트는 시각적 편향을 보이며, 카메라 시점이 가장 중요한 요소로 작용합니다. 둘째, 에이전트는 고채도 색상에서 최고의 성공률을 달성하며, 이는 기본 Vision-Language Models(VLMs)에서 전이된 시각적 선호를 나타냅니다. 셋째, 시각적 편향은 강력하고 비대칭적인 결합을 보여주며, 카메라 시점의 변화가 색상에 관련된 편향을 크게 증폭시킵니다. 이러한 결과는 로봇 시스템의 신뢰성을 결여될 수 있음을 시사하며, 시각적 편향을 줄이기 위한 전략의 필요성을 강조합니다.



### Clinical Uncertainty Impacts Machine Learning Evaluations (https://arxiv.org/abs/2509.22242)
- **What's New**: 본 논문은 의료 데이터셋에서 주석이 불확실한 특성을 가진다는 점을 강조하면서, 기계 학습 평가에 있어 이러한 불확실성을 수치적으로 반영해야 한다고 주장합니다. 기존의 다수결 및 임계값 설정 같은 집계 방법은 주석의 본질적인 변동성을 흐리게 만들어 잘못된 평가를 유발할 수 있습니다. 따라서, 저자들은 확률 기반의 측정방법을 도입하여 더 나은 모델 평가를 촉진하고자 합니다.

- **Technical Details**: 논문에서는 soft metrics(불확실성 인식 지표)의 정의를 제시합니다. 이 지표들은 몇 가지 다른 주석 프로세스와 관계없이 확률 값에서 직접 작용하며, 선형 시간 내에 구현할 수 있는 계산적으로 효율적인 방법입니다. 또한, 저자들은 소프트 AP와 소프트 AUROC 같은 확률적 레이블을 위한 여러 가지 메트릭 확장을 제안하여 데이터셋 간의 결과 비교를 더 정확하게 할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 불확실성을 반영한 평가 방식이 기존의 이진 레이블을 사용한 평가 방식보다 모델 간의 순위에 상당한 영향을 미친다고 밝혀졌습니다. 특히, 다양한 의료 이미지 벤치마크에서 레이블 불확실성을 고려할 경우, 제출된 모델들의 순위가 조정되는 현상이 나타났습니다. 이 관점을 통해 논문은 기술 커뮤니티가 불확실성 인식을 기반으로 한 평가 관행을 채택할 것을 촉구합니다.



### COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics (https://arxiv.org/abs/2509.22240)
- **What's New**: 새로운 연구에서는 COMPASS라는 프레임워크를 소개하여 의료 이미징 세분화 모델에 대한 효율적인 메트릭 기반의 불확실성 정량화 방법을 제안합니다. 이 방법은 딥러닝 모델의 내부 특성을 활용하여 중간 피쳐를 변형하고, 목표 메트릭에 민감한 저차원 부분 공간에서 보정(calibration)을 수행합니다. 이로 인해 COMPASS는 전통적인 방법보다 훨씬 더 좁은 예측 구간을 생성하며, 테스트 데이터에 대한 신뢰성을 높입니다.

- **Technical Details**: COMPASS는 딥러닝 네트워크의 중간 레이어에서 출력된 피쳐를 선형으로 교란하고, 이 과정에서 각 레이어의 피쳐에 대한 출력 메트릭의 기울기를 분석하여 중요한 방향을 찾습니다. 이렇게 정의된 피쳐 공간에서 불확실성을 정량화하면, 목표 메트릭의 위한 예측 구간을 제공합니다. 이 방법은 라벨을 가진 데이터셋에서 모델을 학습시키고, 주어진 데이터 포인트에 대한 비례적(covariate) 변화에 맞춰 신뢰 구간을 조정할 수 있도록 합니다.

- **Performance Highlights**: COMPASS는 피부 병변 및 해부학적 구조에 대한 네 가지 의료 이미지 세분화 작업에서 전통적인 불확실성 예측 방법에 비해 우수한 결과를 보였습니다. COMPASS는 예측 구간이 유의미한 방향으로 조정되고, 메트릭과 높은 정합성을 유지하도록 도와줍니다. 보정된 변형 방법을 추가하면 데이터의 변동성에도 불구하고 목표 커버리지를 복구할 수 있어, 의료 이미지 세분화 분야의 실용적인 적용 가능성을 높입니다.



### Aerial Path Planning for Urban Geometry and Texture Co-Captur (https://arxiv.org/abs/2509.22227)
Comments:
          ACM TOG and SIGGRAPH Asia 2025 (Patent Protected); Project page: this https URL

- **What's New**: 본 논문은 도시 환경의 구조와 질감을 동시에 캡처하는 문제를 다룹니다. 기존 UAV 기반의 도시 재건축 방법은 지오메트리에 초점을 맞추는 경향이 있었으나, 이 연구에서는 2D 건물 윤곽도와 안전 비행 고도 정보만으로 시각적 품질이 높은 텍스처 맵을 재구성하는 방안을 제안합니다. 이는 이미지 캡처의 효율성뿐만 아니라 질감과 기하학적 정확성을 모두 개선하는 다중 목적 최적화 전략을 포함하고 있습니다.

- **Technical Details**: 연구에서는 지오메트리 및 텍스처 품질을 평가하기 위한 종합적인 품질 평가 시스템을 도입하였고, 이를 위해 건물 전면에 맞춤화된 새로운 메트릭 두 가지를 제안합니다. 매개 변수가 캡처되는 과정에서의 구도, 시점, 조명 조건 등 주요 요소들을 고려하여, 드론이 최적의 경로를 따라 비행하면서 동시에 질감과 구조 정보를 효과적으로 수집할 수 있도록 합니다. 이 과정에서 비행 안전성과 효율성도 함께 고려하여 알고리즘을 설계했습니다.

- **Performance Highlights**: 대규모 합성 및 실제 도시 데이터셋에서 진행한 광범위한 실험을 통해, 제안한 방법이 기하학적 구조와 텍스처 재구성을 동시에 수행할 수 있는 이미지 세트를 효과적으로 생성하는데 성공적임을 입증했습니다. 이는 품질 높은 텍스처화된 장면 대리모델을 낮은 운영 비용으로 제작할 수 있게 해줍니다. 최적화 과정에서 텍스처 충실도와 기하학적 정확성을 극대화하고, 비행 비용을 최소화하는 등의 성능을 보였습니다.



### Rigidity-Aware 3D Gaussian Deformation from a Single Imag (https://arxiv.org/abs/2509.22222)
Comments:
          10 pages, 11 figures, conference

- **What's New**: 이번 연구에서는 단일 이미지에서 3D 물체 변형을 복구하는 새로운 프레임워크인 DeformSplat을 제안합니다. 기존 방법들이 다수의 시점을 요구하는 것과 달리, 본 연구는 단일 RGB 이미지만으로도 3D Gaussian 변형이 가능하다는 점에서 큰 의의가 있습니다. 특히 Gaussian-to-Pixel Matching과 Rigid Part Segmentation이라는 두 가지 기술적 기여를 통해 이미지에서 직접적으로 변형 가이드를 제공합니다.

- **Technical Details**: DeformSplat의 첫 번째 기법인 Gaussian-to-Pixel Matching은 3D Gaussian을 2D 픽셀 관측과 연결하여 변형의 원활한 유도를 가능하게 합니다. 두 번째 기법인 Rigid Part Segmentation은 변형 동안 기하학적 일관성을 유지하기 위해 강체(rigid) 지역을 명시적으로 식별합니다. 이를 통해 변형 프로세스에서의 오버피팅을 방지하고, 원래 구조를 유지하는데 기여합니다.

- **Performance Highlights**: 이 연구를 통해 DeformSplat는 기존의 최첨단 방법들과 비교해 우수한 성능을 보이며, 단일 이미지에서 3D Gaussian 변형을 성공적으로 수행할 수 있음을 입증하였습니다. 또한, 해당 프레임워크는 프레임 보간(frame interpolation) 및 상호작용 객체 조작(interactive object manipulation)과 같은 다양한 응용 프로그램으로의 확장성도 가지고 있습니다.



### Guidance Watermarking for Diffusion Models (https://arxiv.org/abs/2509.22126)
- **What's New**: 이 논문은 확산 모델(diffusion models)을 위한 새로운 워터마킹(watermarking) 방법을 소개합니다. 이 방법은 기존의 워터마크 디코더(watermark decoder)로부터 계산된 그래디언트를 사용하여 확산 과정을 안내합니다. 그래디언트 계산은 다양한 이미지 증강(image augmentation)을 포괄하여, 재훈련(retraining)이나 미세 조정(fine-tuning) 없이도 새로운 공격에 대한 강인성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 어떤 포스트-호크(post-hoc) 워터마킹 스킴을 확산 과정 중에 생성적으로 임베딩(in-generation embedding)으로 변환할 수 있음을 보여줍니다. 이 방법은 확산 모델의 재훈련을 요구하지 않으며, 워터마크 디텍터(watermark detector)의 강인성을 계승하면서도 새로운 표적 공격(targeted attacks)에 대해서 개선할 수 있습니다. 이러한 접근 방식은 텍스트-이미지 생성(text-to-image generation) 또는 이미지-이미지 변환(image-to-image translation) 같은 다양한 생성 작업에서 사용될 수 있습니다.

- **Performance Highlights**: 본 연구는 여러 확산 모델과 디텍터를 통해 제안한 방법을 검증하였습니다. 워터마킹 가이던스(watermarking guidance)는 주어진 시드(seed)와 프롬프트(prompt)에 대해 생성된 이미지를 크게 수정하지 않으면서 생성의 다양성과 품질을 보존합니다. 이러한 특성을 통해 제안된 방법은 다양한 생성 AI 응용 프로그램에서 중요한 시사점을 제공합니다.



### Enriching Knowledge Distillation with Intra-Class Contrastive Learning (https://arxiv.org/abs/2509.22053)
- **What's New**: 이 논문에서는 특히 Soft Labels(소프트 레이블)의 활용도를 높이기 위해 인트라 클래스 대비 손실(intra-class contrastive loss)을 도입하는 새로운 접근법을 제안합니다. 기존 연구들이 지적한 바와 같이, 소프트 레이블은 데이터 내 다중 시각 구조에서 기인한 암묵적 지식을 담고 있습니다. 따라서 우리의 방법론은 내용을 풍부하게 하고, 학생 모델의 일반화를 도울 수 있는 방향으로 발전하고자 합니다.

- **Technical Details**: 연구에서는 모델 학습 중 발생할 수 있는 안정성 문제와 수렴 속도 저하를 해결하기 위해 인트라 클래스 대비 학습(intra-class contrastive learning)에 마진 손실(margin loss)을 통합했습니다. 이를 통해 모델의 훈련 효율성을 향상 시켰으며, 이론적으로는 손실이 클래스 내 거리와 클래스 간 거리에도 긍정적인 영향을 미친다는 점을 분석하였습니다. 생긴 새로 제안한 손실 함수는 소프트 레이블의 클래스 내 다양성을 더욱 풍부하게 하는 것을 증명했습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 효과적임을 입증하였습니다. 특히, 소프트 레이블의 변별 능력이 개선되며, 과도한 모델 적합으로 인한 레이블의 결정론적(end-to-end) 성격이 완화되었습니다. 또한, 메모리 효율성을 높이기 위한 파이프라인 기반 캐싱 기법의 도입은 GPU 메모리 제약에서도 안정적인 훈련을 가능하게 하였습니다.



### Comparative Analysis of GAN and Diffusion for MRI-to-CT translation (https://arxiv.org/abs/2509.22049)
- **What's New**: 이 논문에서는 자기 공명 영상(MRI)으로부터 합성 컴퓨터 단층 촬영(CT) 이미지를 생성하는 두 가지 아키텍처인 조건부 생성적 적대 신경망(cGAN)과 조건부 노이즈 제거 확산 확률 모델(cDDPM)의 성능을 비교합니다. 연구팀은 각각 Pix2Pix(조건부 GAN)와 Palette(조건부 DDPM)를 사용해, MRI에서 CT로의 변환을 위한 최적의 전략을 규명합니다. 또한 2D 변환이 계산 비용을 줄이는지 조사하였으며, 단일 및 다중 MRI 이미지 슬라이스에 기초하여 생성 과정을 조건화하는 방법을 평가했습니다.

- **Technical Details**: 작업의 효율성을 높이기 위해, 논문에서는 3D 번역 문제를 횡단면에서의 2D 번역 문제로 나누었습니다. Pix2Pix와 Palette의 구현에서, cGAN 기반의 Pix2Pix는 전통적인 ℒ1-손실을 사용하는 반면, cDDPM은 두 단계인 전방 확산과 후방 노이즈 제거 과정을 포함하여 동작합니다. 최종적으로, SIMOS라는 새로운 슬라이스 간 유사성 지표를 도입하여 3D 형식으로의 합성 CT 재조합의 연속성을 측정했습니다.

- **Performance Highlights**: 비교 분석 결과, cDDPM 모델이 다채널 조건 입력을 활용할 때 MRI-CT 생성 성능이 향상되었으며, 특히 뇌 영역에서 우수한 성능을 보여주었습니다. 유사한 건축 구조를 가진 cGAN이 골반 영역에서는 더 나은 성능을 보였다는 이전 연구 결과도 언급되었습니다. 종합적인 평가 프로토콜을 통해, cDDPM과 cGAN 간의 성능 차이 및 각각의 장단점에 대한 통찰을 제공하였습니다.



### Closing the Oracle Gap: Increment Vector Transformation for Class Incremental Learning (https://arxiv.org/abs/2509.21898)
- **What's New**: 이 논문에서는 Class Incremental Learning (CIL)에서의 케이스별 새 클래스 학습이 과거 학습 데이터에 대한 접근 없이 이루어져야 하는 도전에 대해 다룹니다. 기존 CIL 방법들은 우수한 오라클(oracle) 모델과 비교하여 성능 격차가 존재하는데, 이는 catastrophic forgetting(재앙적 망각)이라는 현상이 주 원인입니다. 이 문제를 해결하기 위한 새로운 접근법으로, Increment Vector Transformation (IVT)이라는 프레임워크를 제안하여 훈련 중의 망각을 줄이는 방법을 소개하고 있습니다.

- **Technical Details**: IVT는 선형 모드 연결성(linear mode connectivity, LMC)이라는 개념을 활용하여 훈련할 때 발생하는 지오메트릭(geometric) 특성의 불일치를 조사합니다. 구체적으로, IVT는 이전 작업 최적값(θt−1∗)에서 현재 솔루션(θt)으로의 방향성을 나타내는 increment vector(Vt)를 정의하고, 이를 저손실(低損失) 대응물(V^t)로 매핑하는 변환을 설정합니다. 이 변환은 주로 diagonal Fisher Information Matrix를 사용하여 곡률 정보를 효율적으로 근사화함으로써 작업 관련 방향을 보존하고, 저손실 연결을 촉진합니다.

- **Performance Highlights**: CIFAR-100, FGVCAircraft, ImageNet-Subset 및 ImageNet-Full에 대한 광범위한 실험을 통해, IVT가 강력한 CIL 기준선의 성능을 지속적으로 향상시킨다는 것을 확인했습니다. 예를 들어, CIFAR-100에서 IVT는 PASS 기준선의 마지막 정확도를 +5.12% 향상시키고, 망각을 2.54% 줄였습니다. 또한, CLIP로 사전 훈련된 SLCA 기준선의 경우, IVT는 평균 정확도를 +14.93%, 마지막 정확도를 +21.95% 개선하는 결과를 보였습니다.



### Perception-Consistency Multimodal Large Language Models Reasoning via Caption-Regularized Policy Optimization (https://arxiv.org/abs/2509.21854)
Comments:
          12pages, 11 figures

- **What's New**: 이번 연구에서는 Caption-Regularized Policy Optimization (CapPO)라는 새로운 강화 학습(RL) 프레임워크를 제안합니다. CapPO는 정책 최적화 과정에서 인지적 일관성을 명시적으로 강제하여 시각적 분석과 추론 과정의 불일치를 해결하는 것을 목표로 합니다. 기존의 RL 방법들이 추론 능력을 향상시키는 데 집중한 반면, CapPO는 캡션을 활용하여 모델의 추론을 신뢰할 수 있는 시각적 콘텐츠에 고정시킵니다.

- **Technical Details**: CapPO는 두 가지 주요 메커니즘을 통합합니다: 첫 번째는 원시 이미지에 기반한 응답과 캡션에 기반한 응답의 발산을 최소화하는 캡션 기반 일관성 정규화입니다. 두 번째는 KL 가중화 이점 추정 방식으로, 이를 통해 강화 신호를 조정하고 시각적 일관성을 유지하며 잘못된 상관관계를 억제합니다. 이러한 방법은 정책 최적화 과정에서 시각적 문제로 인한 오류를 효과적으로 억제합니다.

- **Performance Highlights**: 다양한 수학 및 일반 추론 벤치마크에서의 광범위한 실험 결과를 통해 CapPO가 기존 Qwen2.5-VL-7B 모델 대비 수학 관련 작업에서 +6.0% 정확도, 일반 추론 작업에서 +2.4% 정확도를 달성하며 경쟁력을 보여줍니다. 또한, 컴포넌트별 소거 연구를 통해 각 요소의 효과성과 인지적 오류 감소를 확인하였습니다.



### Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow (https://arxiv.org/abs/2509.21789)
- **What's New**: 이번 연구에서는 멀티 에이전트 시스템(MAS)에서 비주얼 언어 모델(VLM)에 의해 발생하는 시각적 환각이 상호작용하면서 심화되는 현상, 즉 '멀티 에이전트 시각 환각 눈덩이' 현상을 다루고 있습니다. 이 연구는 기존의 단일 에이전트 연구에서 해결하지 못한 새로운 신뢰성과 효율성 문제를 제기하며, 아울러 VLM 기반 MAS의 시각적 정보 흐름에서 발생하는 고유한 메커니즘을 분석합니다.

- **Technical Details**: 우리는 에이전트 간의 시각적 정보 흐름의 흐름에 대한 포괄적인 분석을 통해 시각적 토큰에 대한 주의 할당의 감소에서 시각 환각 눈덩이가 발생할 수 있음을 관찰했습니다. 이러한 분석을 통해 확인된 특정 비주얼 토큰들은 중간 계층에서 unimodal 주의 정점이 나타나며 시각적 정보를 잘 보존하는 것과 관련이 있습니다. 이에 따라 비주얼 흐름을 통해 시각 정보를 중계하고 주의 재할당을 적용하는 새로운 완화 패러다임인 ViF를 제안합니다.

- **Performance Highlights**: ViF는 8개의 기준 벤치마크에서 인상적인 성능을 보여주며, 다양한 MAS 구조와 기본 모델에서 시각 환각 눈덩이를 현저히 줄이는 효과를 입증했습니다. 연구 결과는 ViF가 시각적 메시지를 최적화하고 환각 눈덩이를 줄이는 데 상당한 기여를 할 수 있음을 보여줍니다. 이 연구는 또한 시각 정보 흐름에서의 특정 토큰 집합의 중요성을 확인하여 데이터 정확도를 높이는 방법을 제시합니다.



### ControlHair: Physically-based Video Diffusion for Controllable Dynamic Hair Rendering (https://arxiv.org/abs/2509.21541)
Comments:
          9 pages,Project website: this https URL

- **What's New**: 본 논문에서는 동적 헤어 렌더링을 위한 첫 번째 물리정보 기반 비디오 확산(physics-informed video diffusion) 프레임워크인 ControlHair를 소개합니다. 이 프레임워크는 물리 시뮬레이터와 조건부 비디오 확산 모델을 통합하여 더 정밀한 헤어 다이내믹스를 제공합니다. ControlHair는 10K 비디오 데이터셋으로 학습되어 기존 텍스트 또는 포즈 조건부 모델보다 우수한 성능을 발휘합니다. 이번 연구의 주요 기여는 시뮬레이터와 확산 모델 간의 브릿지 역할을 하는 제어 신호 추출 파이프라인을 설계한 점입니다.

- **Technical Details**: ControlHair는 세 단계의 파이프라인으로 구성됩니다: 먼저 물리적 매개변수(예: 헤어 강도, 바람)를 사용하여 프레임별 기하학적 모델을 인코딩합니다. 이후 프레임별 제어 신호를 추출하고, 마지막으로 비디오 확산 모델에 이러한 신호를 적용하여 원하는 헤어 다이내믹스를 가지는 비디오를 생성합니다. 이 방법은 물리적 사고와 비디오 생성의 분리를 지원하고 다양한 물리 조건을 학습하는 데 유리합니다. 또한 시뮬레이터에서 생성된 프레임별 기하학적 모델을 통해 제어 신호가 생성되며, 이는 실제 비디오에서 쉽게 추출될 수 있습니다.

- **Performance Highlights**: ControlHair는 다양한 응용 사례에서 뛰어난 성능을 보여줍니다. 동적 헤어 스타일 시도(dynamic hairstyle try-on), 불릿타임 효과(bullet-time effects), 시네마그래픽(cinemagraphic) 비디오 제작 등에서 성공적으로 활용됩니다. 본 연구에서 제안된 모델은 기존 비디오 확산 모델에 비해 뛰어난 제어 능력을 가지고 있으며, 다양한 물리 시나리오에 유연하게 적응할 수 있습니다. 향후 연구를 위한 코드를 공개할 계획입니다.



### Patch-Based Diffusion for Data-Efficient, Radiologist-Preferred MRI Reconstruction (https://arxiv.org/abs/2509.21531)
Comments:
          Code is available at: this https URL

- **What's New**: 이번 연구에서는 Patch-based Diffusion Inverse Solver (PaDIS)를 통해 복소수 다중 코일 MRI 재구성을 개선했습니다. 기존의 전 이미지(diffusion model) 기법인 FastMRI-EDM과 비교하여, PaDIS-MRI 모델이 적은 데이터로도 더 나은 성능을 보임을 증명하였습니다. 특히, 25개의 k-space 이미지를 훈련 세트로 사용해도 고품질 재구성을 가능하게 했습니다.

- **Technical Details**: PaDIS는 MRI 데이터를 교체해 불균형한 샘플링 패턴 문제를 해결하는 데 초점을 맞췄습니다. 비록 일반적으로 대용량의 데이터셋이 필요한 생성 모델들이지만, 이번 연구에서는 소규모의 데이터를 통해 localization을 활용하여 성능을 향상시켰습니다. 또한, Diffusion Posterior Sampling (DPS) 알고리즘을 사용하여 잃어버린 k-space 측정을 보완하는 방식으로 복소수 값을 고려한 MRI 재구성을 구현했습니다.

- **Performance Highlights**: 세 명의 방사선과 의사들이 진행한 블라인드 시험에서 PaDIS-MRI가 진단적으로 더 우수하다고 평가된 비율이 91.7%에 달했습니다. FastMRI-EDM 및 전통적인 wavelet sparse reconstruction 방법과 비교했을 때, PaDIS-MRI이 이미지 품질 지표(PSNR, SSIM, NRMSE) 전반에서도 두드러진 성능 향상을 보여주었습니다. 이로 인해, 데이터가 부족한 상황에서도 고충실도의 MRI 재구성이 가능함을 입증했습니다.



### TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning (https://arxiv.org/abs/2509.21526)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: TRiCo는 세 명의 주체—교사(teacher), 두 명의 학생(student)과 적대적 생성기(adversarial generator)—간의 상호작용을 수반하는 신규의 삼각형 게임 이론적 협력 훈련 프레임워크입니다. 이 프레임워크는 기존의 반지도 학습(SSL) 구조를 재고하여 강력한 의사 결정 경계를 탐색할 수 있도록 돕습니다. TRiCo는 상호 정보(mutual information)를 활용하여 의사 레이블 선택을 수행하며, 일관된 성과를 내며 유연하게 확장 가능한 방식으로 반지도 학습의 기준을 다시 설정합니다.

- **Technical Details**: TRiCo는 두 개의 학생 분류기(f1, f2), 비모수적 생성기(G), 그리고 메타 학습된 교사(πT)를 포함하여 세 가지 상호 작용하는 구성 요소로 이루어져 있습니다. 학생들은 서로 의사 레이블을 교환하며, 교사는 상호 정보(MI)를 기반으로 레이블을 필터링하여 모델의 일반화 성능을 조정합니다. 생성기는 임베딩을 섞어 의사 결정 경계를 도전하도록 학생들에게 힘을 줍니다.

- **Performance Highlights**: CIFAR-10, SVHN, STL-10, ImageNet의 광범위한 실험 결과, TRiCo는 낮은 레이블 환경에서 일관되게 최첨단 성능을 달성했습니다. 특히, ImageNet의 경우 25%의 레이블만으로도 최고의 감독 모델들과 경쟁할 수 있는 정확도를 기록했습니다. TRiCo는 구조에 독립적이며 여러 도메인에서 일반화 가능한 효과적인 해법입니다.



### DistillKac: Few-Step Image Generation via Damped Wave Equations (https://arxiv.org/abs/2509.21513)
- **What's New**: DistillKac는 감쇠된 파동 방정식(damped wave equation)과 확률적 Kac 표현을 사용하여, 확률 질량을 유한한 속도로 이동시키는 빠른 이미지 생성기를 제안합니다. 기존의 확산 모델(diffusion models)은 역시멘트 급속한 속도로 질량을 확산시키지만, Kac 동역학은 유한한 속도 수송을 강제하며 전 세계적으로 유도된 운동 에너지를 유지합니다. 또한, 우리는 부드러운 조건에서 제곱 적분 가능성을 보존하는 분류기 없는 안내(classifier-free guidance)를 도입했습니다.

- **Technical Details**: Kac 동역학은 전통적인 확산 모델에 비해 낮은 시간의 무게 중심에서 질량 이동이 정제되며, 이로 인해 엄격한 경계 속도(norm)와 보편적인 안정성(stability)의 이점을 제공합니다. 우리는 경로 전체에 대한 근접성을 증진시키는 종단점만의 증류(endpoint-only distillation) 방법을 제안하며, 이로 인해 학습한 네트워크가 평균적으로 두 점 사이를 연결할 수 있게 됩니다. 이러한 특징은 유한 속도 Kac 동역학 하에서 안정적인 추론을 지원합니다.

- **Performance Highlights**: 우리의 실험은 DistillKac가 매우 적은 함수 평가로 높은 품질의 샘플을 생성할 수 있음을 보여줍니다. 특히, 삼각측면에서의 안정성과 품질이 뛰어나며, 결과적으로 몇 단계로도 효과적인 샘플링을 가능하게 합니다. 이러한 실험 결과는 우리가 제안한 방법들이 가이드를 잘 유지하면서도 적은 에너지를 소모한다는 것을 입증합니다.



### SlimDiff: Training-Free, Activation-Guided Hands-free Slimming of Diffusion Models (https://arxiv.org/abs/2509.21498)
- **What's New**: 본 논문에서 소개된 SlimDiff는 기존의 효율화 기법들과 달리 활성화(activation)를 기반으로 한 자동화된 구조적 압축 프레임워크입니다. 이 프레임워크는 DMs의 주의(attention) 및 피드포워드(feedforward) 차원을 줄이며, 전혀 그래디언트에 의존하지 않습니다. SlimDiff는 압축을 스펙트럼 근사(spectral approximation) 작업으로 재구성하여, 고정된 압축 예산 하에서의 동적 프루닝(dynamic pruning)을 가능하게 합니다. 이는 활성화를 고려한 구조적 압축을 통해 보다 효율적이고 성능을 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: SlimDiff는 기능적 가중치 그룹(𝒬​𝒦, 𝒱​𝒪, ℱ​ℱ​𝒩)에 대해 작용하며, 이는 단순히 개별 행렬을 프루닝(puning)하거나 분해(factorize)하는 것이 아닙니다. 주요 구성 요소로는 스펙트럴 인플루언스 스코어링(spectral influence scoring), 시맨틱 캘리브레이션 데이터셋(SlimSet), 시간 단계 인식 상관 모델링(timestep-aware correlation modeling) 및 모듈 정렬 데이터 인식 압축(MADAC)과 랭크 할당이 포함됩니다. 이러한 다양한 기법은 denoising 과정에서의 활성화 통계와 일치를 이루어 압축을 도와줍니다.

- **Performance Highlights**: SlimDiff는 기존 방법들에 비해 최대 35%의 속도 향상과 약 1억 개의 파라미터 감소를 달성하며, 품질은 전혀 감소하지 않았습니다. 또한, SlimDiff는 고작 500개의 캘리브레이션 샘플을 사용하여 성능을 최적화할 수 있으며, 이는 이전 방법들에 비해 70배 이상 적은 수치입니다. 실험 결과는 MS-COCO, LAION Aesthetics, ImageReward 및 PartiPrompts와 같은 다양한 데이터셋에서 검증되어, 품질 위주로도 안정성을 확보함을 보여줍니다.



### VISION: Prompting Ocean Vertical Velocity Reconstruction from Incomplete Observations (https://arxiv.org/abs/2509.21477)
- **What's New**: 이번 연구는 표면 관측만으로 해양의 수직속도(field)인 ww를 복원하기 위한 새로운 기준 데이터셋인 KD48을 발표하고, 동적 프롬프트(Dynamic Prompting) 기반의 VISION 모델을 도입하여 데이터 부족 문제를 해결하는 접근 방식을 제안합니다. 이를 통해 해양 동역학 연구에 필요한 분석 준비가 완료된 데이터셋을 제공하면서, 이전 방법들의 한계를 극복할 수 있는 새로운 틀을 마련하였습니다.

- **Technical Details**: KD48 데이터셋은 페타스케일(petascale) 시뮬레이션에서 유래된 고해상도 해양 동역학 벤치마크로, 전문가의 신호 필터링(dynamical signal filtering)을 통해 편집되었습니다. VISION 모델의 핵심은 기존의 제한된 입력 변수에 의존하지 않고, 동적으로 데이터를 이용하여 실시간으로 프롬프트를 생성하는 기능입니다. 이 프롬프트는 기본 네트워크의 연산 방식을 조정하는 데 사용되며, 기하학적 및 스케일 인식 연산자를 통합하여 다양한 입력 조합을 처리할 수 있게 합니다.

- **Performance Highlights**: KD48 벤치마크에서 실시한 광범위한 실험 결과, VISION 모델은 기존의 최첨단 모델들에 비해 뛰어난 성능을 보이며, 극단적인 데이터 누락 상황에서도 강력한 일반화 능력을 입증하였습니다. 이를 통해 VISION은 해양 과학 연구의 새로운 기준을 설정하고, 데이터 불확실성 속에서도 신뢰성 있는 해양 동역학 분석을 가능하게 합니다.



### Are Hallucinations Bad Estimations? (https://arxiv.org/abs/2509.21473)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 생성 모델(generative models)에서의 환각(hallucination)을 추정치가 실행 가능한 원인과 연결되지 않는 실패로 형식화합니다. 이러한 해석하에, 손실 최소화(loss-minimizing) 최적 추정기조차도 환각을 경험함을 보여줍니다. 이는 손실 최소화와 인간 수용 가능한 출력 간의 구조적 불일치(structural misalignment)를 재구성하며, 잘못된 보정(miscalibration)으로 인한 추정 오류를 설명합니다.

- **Technical Details**: 최고 밀도 영역(Highest Density Regions, HDR) 개념은 주어진 확률 질량을 포함하는 최소 볼륨 집합으로 정의되며, 이는 다변량 밀도를 시각화하는 데 유용한 알고리즘을 활용하여 다중 모드 구조를 보다 잘 드러냅니다. Conditional Density Regions(HCDRs)이라는 개념을 도입하여, 잠재 변수(latent variable)가 하나의 상태를 가질 때 특정 질량의 HDR에서 대상 분포(target distribution)의 기대값이 탈출하는 δ-hallucination의 원인을 설명합니다. 이는 다수의 상태를 가진 잠재 변수와 연관된 분포로 확장됩니다.

- **Performance Highlights**: 코인 집계(coin aggregation), 개방형 QA(open-ended QA), 텍스트-이미지(text-to-image) 실험을 통해 이론을 뒷받침하고 있습니다. 특히, 최적 추정기가 기대하는 출력 도메인에서 손실을 최소화하지만 여전히 δ-hallucinate 될 우연성을 보여주는 위의 다양한 분포가 구성됩니다. 이 연구는 환각 문제를 보다 깊이 이해하고 해결하는 데 기여할 것으로 기대됩니다.



### Language-in-the-Loop Culvert Inspection on the Erie Cana (https://arxiv.org/abs/2509.21370)
Comments:
          First two authors contributed equally

- **What's New**: 이 논문에서 소개하는 VISION 시스템은 인공지능 기반의 언어-비전 모델을 활용하여 배관 점검을 자동화하는 혁신적인 접근 방식을 제안합니다. 이러한 시스템은 배관의 제한된 시점 계획을 통해 높은 해상도의 이미지를 생성하여 안전하고 효율적인 검사를 수행할 수 있도록 합니다. 이는 전통적인 수동 검사 방법의 위험성을 줄이고 정확성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: VISION 시스템은 웹 규모의 비전-언어 모델(VLM)과 제한적 시점 계획을 결합하여 작동합니다. 사용자는 일반적인 프롬프트를 제공하며, 시스템은 결함이 있을 수 있는 영역을 제안합니다. 이러한 제안은 로봇의 기하학적 제한을 고려하여 고해상도의 이미지를 포착하기 위한 명령을 내리는데 사용됩니다.

- **Performance Highlights**: 뉴욕 운하 공사(NYCC)의 외부 평가에서 초도 ROI 제안이 전문가들과 61.4% 일치하였으며, 최종 재촬영 평가에서는 80%에 달하는 일치를 기록했습니다. 이는 VISION이 가설을 전문가와 일치하는 현실적인 결과로 변환할 수 있는 능력을 가진 것을 의미합니다.



### Cross-Modal Retrieval with Cauchy-Schwarz Divergenc (https://arxiv.org/abs/2509.21339)
Comments:
          Accepted by ACMMM-25

- **What's New**: 이 논문에서는 다중 모달 학습의 핵심 과제인 교차 모달 검색(Cross-Modal Retrieval, CMR)에 Cauchy-Schwarz (CS) divergence를 도입합니다. CS divergence는 초매개변수( hyperparameter)가 필요 없고, 수치적으로 안정적이며 선형적으로 확장 가능하여 기존 방법보다 뛰어난 성능을 보여줍니다. 또한, 새로운 Generalized CS (GCS) divergence를 제안하여 세 개 이상의 모달을 통합하여 비교할 수 있는 수학적 프레임워크를 제공합니다.

- **Technical Details**: CS divergence는 두 개 이상의 모달 간의 직접적인 정렬을 가능하게 하며, 고전적인 방법인 Kullback-Leibler divergence, Maximum Mean Discrepancy, Correlation Alignment의 한계를 극복합니다. 기존 방법들이 일반적으로 두 모달에 제한되는 것에 반해, CS divergence는 다수의 모달을 동시에 비교할 수 있는 회전 대칭 비교( bidirectional circular comparison)가 포함되어 있습니다. 이로 인해 계산 복잡성이 크게 줄어듭니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋을 활용한 포괄적인 실험 결과, CS divergence 기반 방법이 이중 모달 및 삼중 모달 검색 임무에서 모든 기존 접근 방식을 초월하는 성능을 보였습니다. 특히 전통적인 방법에서 요구되는 쌍 비교(pairwise comparisons) 없이도 우수한 검색 정확도를 달성하였습니다. 연구 코드 또한 공개되어 있어 연구자들이 쉽게 활용할 수 있도록 하고 있습니다.



### SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignmen (https://arxiv.org/abs/2509.20401)
- **What's New**: SGAligner++는 3D 장면 그래프 정렬을 위한 경량화된 크로스 모달 프레임워크로, 다양한 모달리티의 데이터를 통합하여 부분적으로 겹치는 장면 관찰을 정렬하는 문제를 해결합니다. 이 방법은 여러 모달리티(예: 포인트 클라우드, CAD 메시, 텍스트 설명)의 정보를 결합하여 통합된 표현 공간을 생성하여 노이즈가 많은 환경에서도 정확한 정렬을 가능하게 합니다. 또한, SGAligner++는 단일 모달 센서 데이터에 의존하지 않고, 더 나아가 언어를 활용한 접근 방식을 통해 다양한 3D 환경에서의 적응성을 보장합니다.

- **Technical Details**: SGAligner++는 포인트 클라우드, 메쉬, 텍스트 캡션, 공간 리퍼럴 등 다양한 모달리티를 통합하여 3D 장면을 구성하는 구조적 그래프를 생성합니다. 각 노드는 다양한 모달의 정보를 포함하여 공간 관계를 이해하고 자원의 결여로 인한 모호성을 해결할 수 있습니다. 이를 통해 운영에서 발생할 수 있는 센서 노이즈 및 부분 겹침 문제를 다루며, 실시간으로 연산량이 적은 방식으로 모달 통합을 수행합니다.

- **Performance Highlights**: 실험 결과, SGAligner++는 실제 데이터셋에서 노이즈가 있는 재구성을 기준으로 기존의 최첨단 방법보다 최대 40% 더 나은 성능을 보여줍니다. 이 접근 방식은 시각적 로컬라이제이션, 3D 재구성, 내비게이션 작업에서도 효과적으로 작용하며, 낮은 런타임과 메모리 오버헤드를 유지하여 확장성이 뛰어납니다. SGAligner++는 경량화된 단일 모달 인코더와 주의 기반 융합을 통해 더욱 견고한 장면 이해를 제공합니다.



### SeamCrafter: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning (https://arxiv.org/abs/2509.20725)
- **What's New**: 본 논문에서는 3D 표면의 UV 매개화 및 질감 매핑에 있어 핵심적인 역할을 하는 메쉬 시음(Mesh seams) 생성 문제를 다룹니다. SeamCrafter라는 새로운 autoregressive GPT 스타일의 시음 생성기를 소개하여 점 구름(point cloud) 입력에 따라 조건화됩니다. 이 모델은 지오메트리(geometry)와 토폴로지(topology) 정보를 효과적으로 결합하여 소스의 복잡성을 감소시키면서 질감 작업의 효율성을 높입니다.

- **Technical Details**: SeamCrafter는 이중 분기 인코더를 사용하여 입력 메쉬의 기하학적(geometric) 및 토폴로지적(topological) 단서를 분리하여 캡처합니다. 이 모델은 직접 선호 최적화(Direct Preference Optimization, DPO) 기법을 활용하여 인간의 선호에 맞춘 시음 품질을 향상시키도록 조정됩니다. 훈련 과정은 방대한 시음 데이터에 대해 유 supervised pretraining 과정을 통해 이루어지며, 이후 DPO를 통해 미세 조정됩니다.

- **Performance Highlights**: SeamCrafter는 이전 방법들에 비해 UV 왜곡(UV distortion) 및 파편화(fragmentation)가 현저히 낮은 시음을 생성하는 것으로 나타났습니다. 실험을 통해 이 모델이 토폴로지적 일관성과 시각적 충실도를 보존하면서 질감 워크플로우에 적합한 세련된 시음을 제공함을 입증했습니다. 이러한 성과는 예술가의 작업 흐름을 개선할 수 있는 가능성을 시사합니다.



New uploads on arXiv(cs.AI)

### Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspectiv (https://arxiv.org/abs/2509.22613)
- **What's New**: 본 논문에서는 강화 학습 (Reinforcement Learning, RL) 기법이 대규모 언어 모델 (Large Language Models, LLMs)에 대한 계획 능력을 어떻게 향상시키는지에 관한 새로운 이론적 분석을 제공합니다. 특히, 정책 기울기 (Policy Gradient) 및 Q-학습 (Q-learning) 방법의 장점과 한계를 조사하였습니다. RL의 탐색이 계획의 핵심 역할을 한다는 점을 강조하며, RL이 감독 하에 미세 조정 (Supervised Fine-Tuning, SFT) 방법보다 더 나은 일반화를 가능하게 한다는 것을 보여줍니다.

- **Technical Details**: 계획은 LLM에서 비방향 그래프의 경로 탐색 문제로 추상화되며, 각 노드는 고유한 토큰으로 표현됩니다. 본 연구에서는 SFT와 RL을 이용한 경로 계획의 구조적 특징을 제시하고, SFT가 도입하는 동시 발생 기반 오류를 분석합니다. PG와 Q-learning에 대한 심도 있는 분석을 통해 이론적 장점을 강조하였고, 특히 Q-learning의 경우 출력 다양성을 보존하면서 최적의 정확성에 수렴할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, PG는 훈련 중 출력 다양성이 감소하는 '다양성 붕괴' 현상을 보여줍니다. 하지만 Q-learning은 프로세스 보상이 포함되었을 때 이 문제를 해결할 수 있으며, 이는 RL 기반 학습 방식에서 중요한 이점으로 작용합니다. 마지막으로, Blocksworld와 같은 실제 계획 벤치마크에 본 연구의 프레임워크를 적용하여 이론적 발견이 실제로 발현되는 것을 확인했습니다.



### Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Tim (https://arxiv.org/abs/2509.22572)
- **What's New**: 본 논문에서는 Test-Time Scaling (TTS) 기법을 통해 대규모 언어 모델(LLM)의 추론 능력을 향상시킬 수 있다는 새로운 전략인 Dynamic Experts Search (DES)를 제안합니다. 기존의 방법들은 주로 출력 수준의 샘플링에 의존하였으나, 우리는 모델 아키텍처의 역할을 간과하고 있는 점을 지적했습니다. Mixture-of-Experts (MoE) 아키텍처의 전문가 활성화 수를 조정하여 다양한 해결책을 생성하는 가능성을 발견하였습니다.

- **Technical Details**: DES는 두 가지 주요 요소를 통합하여 작동합니다. 첫 번째는 Dynamic MoE로, 이는 추론 과정에서 활성화된 전문가 수를 직접 제어할 수 있게 하여 추가 비용 없이 다양한 추론 경로를 생성할 수 있게 합니다. 두 번째는 Expert Configuration Inheritance로, 이는 추론 경로 내에서 전문가 수를 일관되게 유지하여 안정성과 다양성을 균형 있게 분산할 수 있도록 합니다.

- **Performance Highlights**: 다양한 MoE 아키텍처와 검증자, 추론 벤치마크(예: 수학, 코드, 지식)에 대한 광범위한 실험 결과, DES는 기존 TTS 기준을 일관되게 초월하여 정확도와 안정성을 높였습니다. DES는 추가 비용 없이 추론 능력을 향상시키며, 다양한 모델 규모와 검증자 선택에 대해 효과적으로 일반화되는 장점을 보여주었습니다.



### UniMIC: Token-Based Multimodal Interactive Coding for Human-AI Collaboration (https://arxiv.org/abs/2509.22570)
- **What's New**: 최근 인공지능(AI) 기술의 발전, 특히 대형 멀티모달 모델(LMMs)과 자율 AI 에이전트가 인간과 AI의 협업 패러다임을 근본적으로 변화시키고 있습니다. 기존의 코덱은 주로 단방향, 단일 모드 통신에 최적화되어 있어, 반복적으로 품질 저하를 겪곤 했습니다. 이러한 문제를 해결하기 위해 제안된 UniMIC는 통합된 토큰 기반 멀티모달 인터랙티브 코딩 프레임워크로서, 에지 디바이스와 클라우드 AI 에이전트 간의 효율적인 통신을 가능하게 합니다.

- **Technical Details**: UniMIC는 AI-친화적인 통신 프로토콜을 통해 토큰을 기본 교환 매체로 활용합니다. 이를 통해 에지 디바이스는 오직 작업과 관련된 토큰만을 전송하고, 클라우드는 필요에 따라 생성된 토큰만을 반환하게 됩니다. 추가적으로, 다양한 시나리오에 적응 가능한 경량화된 Transformer 기반의 엔트로피 모델을 개발하여 상호 작용에 맞춘 효율적인 압축을 지원합니다.

- **Performance Highlights**: UniMIC는 텍스트-이미지 생성, 텍스트 기반 인페인팅, 아웃페인팅 및 시각적 질문 응답(VQA)에서 기존 픽셀 기반 코덱에 비해 상당한 비트레이트 절약을 보여주었습니다. 실험 결과는 전반적으로 세밀함을 유지하면서도 초저 비트레이트(<0.05 bpp)에서 높은 작업 성능을 유지한다고 입증되었습니다. 더 나아가 UniMIC는 차세대 멀티모달 인터랙티브 커뮤니케이션을 위한 실용적이고 진보적인 패러다임으로 자리잡을 수 있습니다.



### StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models (https://arxiv.org/abs/2509.22558)
- **What's New**: 이번 논문에서는 Operations Research (OR) 문제 해결을 위한 새로운 Self-Evolving Framework인 StepORLM을 소개합니다. 기존의 강화 학습 (Reinforcement Learning) 접근 방식이 가지고 있는 한계인 신뢰할 수 없는 중간 단계 보상과 정확한 기여 할당 문제를 해결하는 데 중점을 두었으며, StepORLM은 정책 모델과 생성적 프로세스 보상 모델(GenPRM)의 공동 진화를 통해 이를 이루고자 합니다. 이 연구는 전체 추론 경로를 평가하는 생성적 프로세스 감독을 최초로 적용함으로써, 기존 접근 방식의 한계를 효과적으로 극복합니다.

- **Technical Details**: StepORLM의 핵심 구성 요소는 정책 모델과 GenPRM 간의 공진화 루프로, 이들은 서로를 반복적으로 개선하는 역할을 합니다. 각 반복에서 정책 모델은 두 가지 보상 메커니즘, 즉 외부 해결자로부터의 확정적인 결과 검증과 GenPRM으로부터의 미세하고 전체적인 과정 피드백을 통해 추론 경로를 평가합니다. 이 이중 피드백 신호는 Weighted Direct Preference Optimization(W-DPO)을 통해 정책을 조정하는 데 사용되며, 동시에 GenPRM을 추가로 미세 조정하는 데 이용됩니다.

- **Performance Highlights**: StepORLM은 8B 파라미터를 가지고 있으며, 여섯 개의 벤치마크에서 새로운 최신 성과(SOTA)를 기록했습니다. 특히, StepORLM은 일반적인 대형 모델과 전문화된 기준선보다 현저하게 우수한 성능을 보여주며, GenPRM은 다른 기존 LLM의 추론 성능을 향상시키는 강력하고 보편적인 프로세스 검증기로 기능합니다. 이와 함께, 우리는 StepORLM 모델과 GenPRM 검증기 모델의 코드 및 가중치를 커뮤니티에 공개하여 향후 LLM 기반 OR 문제 해결 연구에 기여할 기반을 마련하고자 합니다.



### The Emergence of Altruism in Large-Language-Model Agents Society (https://arxiv.org/abs/2509.22537)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용한 사회 시뮬레이션의 새로운 접근 방식을 제안합니다. 저자들은 기존 연구가 자본주의적(egoistic) 선택 및 이타적(altruistic) 선택을 어떻게 보여주는지를 분석하는 데 초점을 맞추었다고 주장합니다. 특히, 두 가지 유형의 LLM(‘Adaptive Egoists’와 ‘Altruistic Optimizers’)의 사회적 경향성을 식별하고, 이들이 사회적 규범의 영향을 어떻게 받는지를 탐구합니다.

- **Technical Details**: 이 연구는 Schelling-변형 도시 이주 모델을 소개하며, 200개 이상의 LLM 에이전트가 이기적 및 이타적 목표 간의 명시적 갈등을 탐색해야 하는 환경을 만듭니다. 또한, Grounded Theory에서 영감을 얻은 방법론을 통해 에이전트의 추론 과정을 체계적으로 코드화하여 질적 분석을 수행합니다. 이 접근법은 LLM의 내재적 사회적 행동의 논리를 선택하는 것의 중요성을 강조합니다.

- **Performance Highlights**: 실험 결과, LLM의 사회적 경향성에 대한 근본적인 이분화가 발견되었습니다. 'Adaptive Egoists'는 자아 중심으로 시작하지만 사회적 규범의 영향을 받을 경우 이타적 행동을 증가시킵니다. 반면 'Altruistic Optimizers'는 본질적으로 이타적 논리를 지니고 있으며, 개인적인 비용을 감수하면서도 집단의 이익을 지속적으로 우선시합니다. 이는 사회 시뮬레이션에 사용할 LLM의 선택이 그 이론적 기초를 선택하는 것과 같음을 시사합니다.



### REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Mod (https://arxiv.org/abs/2509.22518)
- **What's New**: 이 연구에서는 Large Language Models(LLMs)의 복잡한 추론(process) 수행과 실패 메커니즘을 이해하기 위한 새로운 접근법으로 Reasoning Manifold(추론 다양체) 개념을 소개합니다. 이 다양체는 올바르게 추론된 생성물과 관련된 내부 표현의 잠재적 저차원 기하학적 구조로 정의됩니다. 이 연구는 REMA라는 새로운 프레임워크를 개발하여 잘못된 추론과 올바른 추론 샘플의 내부 모델 표현 간 공간적 관계를 정량적으로 비교하여 실패의 원인을 설명하는 방법을 제안합니다.

- **Technical Details**: REMA 프레임워크의 중심 메커니즘은 모든 추론 실패를 그 내부 표현이 올바른 추론 다양체에서 벗어난 기하학적 편차로 통합하는 것입니다. 이를 위해 두 가지 단계의 분석 작업 흐름을 따릅니다. 첫째로, 잘못된 표현 각각의 k-최근접 이웃 거리(k-nearest neighbors distance)를 계산하여 편차의 심각성을 정량화하고, 둘째로 이러한 편차의 원점을 추적하여 실패의 기원을 국소화합니다. 이 과정에서 다양한 언어 및 다중 모달 모델에서 실험을 통해 저차원적 추론 다양체의 특성을 확인했습니다.

- **Performance Highlights**: 실험 결과, 올바른 추론과 잘못된 추론의 표현 간의 기하학적 분리가 높음을 나타내며, 잘못된 추론의 표현은 항상 올바른 추론의 다양체에서 통계적으로 유의미한 편차를 보입니다. REMA 프레임워크는 이러한 기하학적 편차를 통해 추론 실패의 원인을 검토하는 데 효과적임을 입증하고, 검증된 다수의 모델과 작업에서 이 개념을 적용할 수 있는 가능성을 보여줍니다.



### TrueGradeAI: Retrieval-Augmented and Bias-Resistant AI for Transparent and Explainable Digital Assessments (https://arxiv.org/abs/2509.22516)
- **What's New**: 이 논문에서는 전통적인 종이 기반 평가의 단점을 극복하기 위해 설계된 AI 기반 디지털 시험 프레임워크인 TrueGradeAI를 소개하고 있습니다. 이 시스템은 보안이 유지된 태블릿에서 스타일러스 입력을 캡처하여 자연스러운 필기체를 보존하고, 트랜스포머 기반의 Optical Character Recognition (OCR)을 적용하여 필기 내용을 전사합니다. TrueGradeAI는 공정하고 투명한 평가를 가능하게 하는 방법으로 평가 과정을 획득 증강 파이프라인을 통해 수행합니다.

- **Technical Details**: TrueGradeAI는 시험 전달, 필기 캡처, 전사, 자동 점수 부여 및 보고를 관리하는 완전한 디지털 평가 파이프라인을 구축합니다. 시스템은 필기 인식 기술을 통합하여 학생들이 자연스러운 필기 방법으로 답변을 작성할 수 있게 하며, 이를 통해 학생들의 응답을 신뢰할 수 있는 분석 근거와 함께 점수화합니다. 또한, 모든 결정 과정은 로깅되어 감사 가능성을 제공하며, 채점 지연을 몇 주에서 몇 시간으로 줄입니다.

- **Performance Highlights**: TrueGradeAI는 AI-assisted grading과 인간의 검토를 결합하여 채점 과정을 더욱 효율적으로 만드는 다양한 기능을 제공합니다. 학생 포털과 교사 포털은 보안이 유지되는 온라인 환경에서 원활하게 정보를 처리하며, 교사 포털에서는 실시간 모니터링, 결과 관리, 학생 분석 및 AI 보조 채점을 제공합니다. 이러한 통합된 아키텍처는 운영의 투명성과 공정성을 높이며, 기존의 평가 시스템보다 더 신뢰할 수 있는 옵션을 제시합니다.



### Estimating the Empowerment of Language Model Agents (https://arxiv.org/abs/2509.22504)
Comments:
          10 pages, 8 figures. Submitted to ICLR 2026

- **What's New**: 이 논문에서는 언어 모델(언어 모델 에이전트)의 능력을 평가하기 위한 새로운 정보 이론적 접근 방식을 제안했습니다. 전통적인 평가 방식의 한계를 극복하기 위해, 에이전트의 행동과 미래 상태 간의 상호 정보를 기준으로 한 효율적인 측정 방식을 개발했습니다. 새로운 알고리즘 EELMA를 통해 다단계 텍스트 상호작용에서 효과적인 역량을 근사화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: EELMA는 Markov 결정 프로세스(Markov Decision Process, MDP) 프레임워크를 기반으로 설계되었습니다. 이 알고리즘은 에이전트의 상태, 행동, 보상 및 할인계수와 같은 요소를 고려하여 능력 평가를 수행합니다. 에이전트의 현재 행동과 미래 상태 간의 상호 정보를 측정하여, 에이전트의 환경에 대한 통제력을 정량화하며, 이를 통해 언어 기반 에이전트의 역량을 평가할 수 있도록 합니다.

- **Performance Highlights**: EELMA의 성능을 다양한 환경에서 검증하여, 효과적인 역량 추정이 임무 성과와 강한 상관관계를 가지고 있음을 발견했습니다. 또한, 높은 역량 순간에서는 에이전트가 환경에 대한 통제를 빠르게 확장하는 중요한 시점을 포착할 수 있음을 보여주었습니다. 이러한 결과는 EELMA가 언어 모델 에이전트를 평가하는 강력한 일반 목적의 지표임을 입증하며, 지속적인 모니터링에도 유용하게 활용될 잠재력을 시사합니다.



### InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios (https://arxiv.org/abs/2509.22502)
Comments:
          9 pages of main content and 32 pages of others, 2 figures, under review as a conference paper at ICLR 2026

- **What's New**: 이번 논문에서는 LLM (Large Language Model) 에이전트의 개발과 배포의 어려움을 해결하기 위해 새로운 프레임워크, InfiAgent를 제안합니다. InfiAgent는 다양한 문제 도메인에 자동으로 적응할 수 있도록 설계된 DAG(Directed Acyclic Graph) 기반의 다중 에이전트 프레임워크로, 복잡한 작업을 자동으로 계층화하여 처리할 수 있는 기능을 갖추고 있습니다. 이는 에이전트의 자동 decomposing, 품질 및 안정성을 보장하는 이중 감사 메커니즘, 효율적인 작업-에이전트 매칭을 위한 라우팅 기능 등을 포함하여, 전통적인 방법의 한계를 극복하는 혁신적인 접근입니다.

- **Technical Details**: InfiAgent는 에이전트를 도구로 사용하는 추상화 메커니즘을 통해 하위 에이전트를 상위 에이전트가 호출할 수 있는 계층형 구조를 자동으로 생성합니다. 이와 함께 지속적인 모니터링과 검증을 수행하는 이중 감사 시스템을 도입하여 복잡한 워크플로우에서도 오류 전파를 방지합니다. 또한, 자동화된 작업 라우팅 기능이 에이전트 간의 매칭을 최적화하며, 자원의 활용도를 극대화하여 시스템의 효율성을 증대시킵니다.

- **Performance Highlights**: InfiAgent는 ADAS (자동 생성된 에이전트 프레임워크) 대비 9.9% 더 높은 성능을 기록했습니다. 또한 InfiHelper라는 AI 연구 도우미의 사례 연구에서는 이 프레임워크가 IEEE의 상위 학술 대회에서 인간 리뷰어에게 인정을 받은 과학 논문을 생성하는 성과를 보여주었습니다. 이러한 결과는 InfiAgent의 뛰어난 효율성과 안정성을 입증하며, 다양한 산업 분야에서의 지능형 자동화 가능성을 열어줍니다.



### GeoSketch: A Neural-Symbolic Approach to Geometric Multimodal Reasoning with Auxiliary Line Construction and Affine Transformation (https://arxiv.org/abs/2509.22460)
- **What's New**: 이 논문에서는 Geometric Problem Solving (GPS)을 위한 새로운 신경-상징적 프레임워크인 GeoSketch를 소개합니다. 기존의 다중 모달 대형 언어 모델(MLLMs)은 정적인 이미지를 처리하는 데 국한되어 있었으나, GeoSketch는 상호 작용적 인식-추론-행동 루프를 통해 기하학적 추론을 재구성합니다. 이는 텍스트와 다이어그램의 공동 해석뿐만 아니라 동적인 비주얼-공간적 추론을 가능하게 합니다.

- **Technical Details**: GeoSketch는 세 가지 주요 모듈로 구성됩니다: (1) Perception 모듈은 다이어그램을 구조적 논리 형태로 추상화하고, (2) Symbolic Reasoning 모듈은 기하학 정리를 적용하여 다음 추론 단계를 결정하며, (3) Sketch Action 모듈은 동적 조작을 수행하는 역할을 합니다. 이러한 구조를 위해 GeoSketch는 2000개의 기호 커리큘럼 경로에 대해 감독된 미세 조정(Supervised Fine-Tuning) 후, 밀집 상징적 보상을 통한 강화 학습(Reinforcement Learning) 과정을 통해 학습됩니다.

- **Performance Highlights**: GeoSketch 벤치마크를 사용하여 390개의 기하학 문제를 평가했으며, 기존의 정적 인식 방법보다 단계적으로 추론 정확도와 문제 해결 성과를 유의미하게 향상시켰습니다. 실험 결과는 GeoSketch의 아키텍처가 MLLMs에 비해 성능을 크게 향상시켰다는 것을 보여주며, 이는 동적이며 검증 가능한 상호 작용을 통해 이루어진 것입니다.



### Guiding Evolution of Artificial Life Using Vision-Language Models (https://arxiv.org/abs/2509.22447)
Comments:
          9 pages, 6 figures. Accepted for publication in the Proceedings of the Artificial Life Conference 2025 (MIT Press)

- **What's New**: 최근 Foundation Models (FMs)은 인공지능 생명체(Artificial Life, ALife) 분야에서 새로운 가능성을 열어주고 있다. 본 논문은 기존 ALife 시뮬레이션을 자동화하는 검색 방법을 개선한 ASAL++ 방법을 제안한다. ASAL++는 멀티모달 FMs에 의해 안내된 개방형의 검색을 통해 진화의 목표를 제안하는 새로운 방식을 모색한다.

- **Technical Details**: ASAL++는 두 가지 전략을 탐구한다: (1) Evolved Supervised Targets (EST), 각 반복에서 단일 새로운 프롬프트에 맞도록 시뮬레이션을 진화시키고, (2) Evolved Temporal Targets (ETT), 생성된 프롬프트의 전체 시퀀스에 맞도록 시뮬레이션을 진화시키는 방법이다. 또한, Gemma-3를 사용하여 ALife 시뮬레이션에서 진화 목표를 제안하고, 시뮬레이션 파라미터를 최적화하여 시각적이고 텍스트적인 임베딩에 따라 목표를 조정한다.

- **Performance Highlights**: 실험 결과, EST는 보다 높은 시각적 참신성을 촉진했으며, ETT는 보다 일관되고 해석 가능한 진화 시퀀스를 촉진하였다. ASAL++는 FMs에 의한 ALife 발견의 새로운 방향을 제시하며, 개방형 특성을 가진 진화 경로를 발견할 가능성을 보여준다.



### EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer (https://arxiv.org/abs/2509.22407)
- **What's New**: 본 연구에서는 Embodied Manipulation Media Adaptation (EMMA)을 제안합니다. EMMA는 로봇 조작 비디오를 생성하는 DreamTransfer와 효과적인 훈련 전략인 AdaMix를 통합한 VLA 정책 향상 프레임워크입니다. 특히 DreamTransfer는 텍스트 기반의 시각 편집을 지원하여 기존 비디오의 전경, 배경 및 조명 조건을 변경할 수 있는 기능을 제공합니다.

- **Technical Details**: DreamTransfer는 다중 카메라 시점에서 일관성 있는 로봇 조작 비디오를 생성하며, 3D 기하 구조와 시간적 일관성을 유지합니다. AdaMix는 훈련 중 어려운 샘플에 대해 가중치를 동적으로 조정하는 전략으로, 이를 통해 정책의 일반화를 향상시킵니다. 이 프레임워크는 비디오 생성 품질과 실제 로봇 배치에 대해 평가되었습니다.

- **Performance Highlights**: DreamTransfer는 다중 시점 일관성에서 42%, 깊이 일관성에서 24%의 성능 향상을 보여주며, 제로 쇼트(Zero-shot) 환경에서 200% 이상의 상대적 개선을 달성했습니다. AdaMix와 통합했을 때 추가로 13%의 성능 향상이 이루어져 정책 일반화에 유의미한 기여를 합니다.



### Do LLM Agents Know How to Ground, Recover, and Assess? A Benchmark for Epistemic Competence in Information-Seeking Agents (https://arxiv.org/abs/2509.22391)
- **What's New**: 이 연구는 대형 언어 모델(LLM) 검색 에이전트의 평가에서 단순한 정답 정확도를 넘어, 외부 증거를 어떻게 활용하고 행동하는지를 분석하는 SeekBench라는 새로운 벤치마크를 소개합니다. SeekBench는 190개의 전문가 주석이 달린 트레이스와 1,800개 이상의 응답 단계로 구성되며, 에이전트가 (1) 관찰된 증거에 기반한 추론 단계를 생성하는지, (2) 낮은 품질의 결과에서 회복하기 위해 검색을 재구성하는지, (3) 현재 증거가 충분한지를 올바르게 평가하는지 등의 세분화된 분석을 가능하게 합니다.

- **Technical Details**: 본 연구는 서술된 트레이스 구조를 사용하여 에이전트의 추론, 검색 및 증거 통합의 다단계 실행을 포착합니다. 특히, 추론 과정에서 에이전트의 지식 획득, 평가 및 행동 능력을 측정하기 위해 세 가지 핵심 인식 능력을 formalize했습니다. 이는 증거 기반 추론, 적응적 증거 회복, 불확실성 하의 보정 결정으로 구성됩니다.

- **Performance Highlights**: 대규모 7개의 QA 벤치마크(28,493 트레이스)에서의 평가 결과, RL 에이전트는 증거 수집에서 뛰어난 성과를 보였으나 추론에 어려움을 겪는다는 사실을 발견했습니다. 기존의 정확성 지표는 에이전트 간의 특정 강점을 드러내지 못하여, 검색 단계와 기본 모델의 추론을 조합하면 성능 향상으로 이어질 수 있습니다.



### PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning (https://arxiv.org/abs/2509.22315)
Comments:
          8 pages

- **What's New**: 이 논문은 인간 인지의 이중 처리 이론에 영감을 받아, 빠르고 직관적인 사고(System 1)와 느리고 신중한 사고(System 2)를 통합하는 멀티 에이전트 추론 프레임워크인 PRIME(Planning and Retrieval-Integrated Memory for Enhanced Reasoning)를 소개하고 있습니다. PRIME은 초기 빠른 답변을 생성하는 Quick Thinking Agent를 활용하고, 불확실성이 감지되면 보다 구조화된 System 2 사고 프로세스를 활성화하여 전문화된 에이전트들이 계획, 가설 생성, 정보 통합 및 의사 결정을 수행합니다. 이러한 설계는 인간의 인지 과정을 충실히 모방하며, 효율성과 정확성을 향상시키는 데 기여합니다.

- **Technical Details**: PRIME은 인간의 인지 프로세스를 모델링하고, 빠른 직관적인 사고와 느린 신중한 사고를 통합하기 위해 설계된 멀티 에이전트 시스템입니다. 이 시스템은 Quick Thinking Agent(System 1)를 통해 직관적 응답을 생성하며, 그 후 Reflection Agent가 이 응답을 비판적으로 평가합니다. 문제가 발견되면, System 2 프로세스가 활성화되어 체계적인 계획, 탐색, 가설 설정 및 정보 통합을 통해 신뢰할 수 있는 최종 답변을 도출합니다.

- **Performance Highlights**: 실험 결과, PRIME은 LLaMA 3 모델이 GPT-4 및 GPT-4o와 같은 최신 클로즈드 소스 모델과 경쟁할 수 있는 성능을 발휘하게 함을 보여줍니다. 특히 PRIME은 복잡한 지식 기반의 추론이 필요한 작업에서 뛰어난 성능을 발휘했으며, 효율적인 컴퓨팅 자원 할당을 통해 보다 어려운 작업에 대해서만 System 2를 활성화하여 불필요한 숙고를 줄였습니다. 이로 인해 PRIME은 복잡한 지식 집약적 추론 작업을 위한 강력하고 확장 가능한 프레임워크로 자리 잡았습니다.



### Large Language Models as Nondeterministic Causal Models (https://arxiv.org/abs/2509.22297)
Comments:
          Preprint: under review

- **What's New**: Chatzi et al. 및 Ravfogel et al.의 최근 연구는 확률적 대형 언어 모델(Large Language Models, LLM)의 반사실적(counterfactual) 결과를 생성하는 첫 번째 방법을 개발했습니다. 이러한 반사실적 결과는 특정 사실적 프롬프트가 다른 것으로 바뀌었을 때 LLM의 출력이 어떻게 달라질지를 보여줍니다. 하지만 기존 방법은 LLM의 해석이 모호하여 비결정론적 모델을 결정론적 인과 모델로 표현하는 등의 변경을 전제로 하고 있습니다.

- **Technical Details**: 이 논문은 LLM의 의도된 해석을 바탕으로 LLM를 비결정론적 인과 모델로 표현함으로써 반사실적 결과를 생성하는 훨씬 간단한 방법을 제시합니다. 이 방법은 구현 세부 사항에 독립적이기 때문에 수정 없이 모든 블랙 박스 LLM에 직접 적용할 수 있습니다. 반면, 기존 방법은 특정 용도에 유용한 반사실적 결과를 생성하는데 직접 사용되지만, 반면 불필요하게 복잡성을 추가합니다.

- **Performance Highlights**: 이 새로운 접근법은 다양한 방법들을 체계적으로 탐색할 수 있는 이론적 기초를 제공하여, 반사실적 결과 생성을 위한 향후 연구의 기초를 마련합니다. 또한, LLM의 자체 생성된 설명의 신뢰성을 테스트할 수 있는 가능성을 제공하여, LLM들이 생성한 반사실적 설명의 품질을 평가할 수 있는 기회를 제시합니다. 반사실적 설명의 생성은 설명 가능한 인공지능(Explainable AI, XAI) 발전에 중요한 역할을 할 것으로 기대됩니다.



### Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models (https://arxiv.org/abs/2509.22284)
Comments:
          10 pages, NeurIPS 2025 Spotlight

- **What's New**: 논문에서는 현대 상태 공간 모델(State-Space Models, SSMs)의 전이 행렬(transition matrix)에 대한 새로운 구조적 희소 파라미터화를 제안합니다. 이 접근법은 유한 상태 오토마타(Finite-State Automata, FSA)를 효율적으로 추적할 수 있도록 하며, 계산 비용을 대각 SSM과 비슷하게 유지합니다. PD-SSM이라는 새로운 방법론은 Column One-Hot 행렬과 복소수 대각 행렬을 이용해 전이 행렬을 파라미터화합니다.

- **Technical Details**: PD-SSM은 N 상태의 FSA를 단일 레이어, 상태 차원 N과 N×N 크기의 선형 리드아웃으로 구현할 수 있는 시간 변이 SSM입니다. 이 모델은 BIBO 안정성을 가지고 있으며, 복잡한 FSA의 상태를 효과적으로 추적할 수 있습니다. PD 행렬은 서로 곱해도 여전히 PD 행렬이며, 병렬 스캔을 통해 계산할 수 있어 효율성을 증가시킵니다.

- **Performance Highlights**: 모델은 다양한 FSA 상태 추적 작업에서 기존의 여러 현대 SSM 변형들보다 독보적으로 뛰어난 성과를 보입니다. 멀티클래스 시계열 분류에서도 신경 제어 미분 방정식(neural controlled differential equations)과 비교했을 때 유사한 성과를 나타냈습니다. 마지막으로, hybrid Transformer-SSM 아키텍처에 통합하여 복잡한 FSA의 상태를 효과적으로 추적할 수 있음을 보여줍니다.



### InfiMed-Foundation: Pioneering Advanced Multimodal Medical Models with Compute-Efficient Pre-Training and Multi-Stage Fine-Tuning (https://arxiv.org/abs/2509.22261)
- **What's New**: 이 논문에서는 InfiMed-Foundation-1.7B와 InfiMed-Foundation-4B라는 두 가지 의학 특화 다중 모달 대형 언어 모델(MLLMs)을 제안합니다. 이 모델들은 의료 응용 프로그램에서 최첨단 성능을 제공하도록 설계되었습니다. 고품질의 일반 및 의학 다중 모달 데이터를 결합하고, 새로운 다차원 품질 평가 프레임워크를 제안하여 높은 품질의 데이터셋을 구성하였습니다.

- **Technical Details**: InfiMed-Foundation 모델은 이미지 해상도를 낮추고, 다중 모달 시퀀스 패킹 기술을 적용하여 훈련 효율성을 향상시켰습니다. 또한, 세 가지 단계의 감독 학습 미세 조정 과정을 통해 복잡한 의료 작업에 필요한 지식을 효과적으로 추출할 수 있게 하였습니다. MedEvalKit 프레임워크를 사용하여 모델 성능을 평가하였으며, 이는 의료 시각 질문응답 및 진단 작업에서 우수한 성과를 나타냈습니다.

- **Performance Highlights**: InfiMed-Foundation-1.7B 모델은 Qwen2.5VL-3B를 초월했으며, InfiMed-Foundation-4B는 HuatuoGPT-V-7B와 MedGemma-27B-IT를 초월하여 의료 관련 성능을 입증하였습니다. 이러한 결과는 모델들이 실제 치료 상황에서 의료 전문가를 지원할 수 있는 가능성을 믿을 수 있게 보여줍니다. 데이터 품질, 훈련 효율성, 및 도메인 특화 지식 추출과 같은 핵심 도전을 해결하면서, 본 연구는 의료 분야에서 더 신뢰할 수 있고 효과적인 AI 솔루션을 위한 길을 열었습니다.



### Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing (https://arxiv.org/abs/2509.22255)
Comments:
          1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling

- **What's New**: 이 논문은 Large Language Models(LLMs)의 조합 최적화(combinatorial optimization)에서의 능력을 평가하는 프레임워크를 제시합니다. 특히 2D bin-packing 문제를 다루며, LLM과 진화 알고리즘을 결합하여 휴리스틱 솔루션을 생성하고 반복적으로 개선하는 체계적인 방법론을 소개합니다. LLM이 생성한 솔루션은 전통적인 방법들에 비해 더 효율적인 결과를 제공하는 것으로 나타났습니다.

- **Technical Details**: 2D-bin packing 문제는 NP-hard 문제로, 최소 수의 고정 크기 bin에 직사각형 아이템을 배치하는 것을 목표로 합니다. 논문에서는 LLM의 성능을 평가하기 위해 강력한 제약 조건을 설계하고, 문제 해결 과정에서 생성된 솔루션의 성능을 여러 지표를 통해 평가합니다. 이 프레임워크는 LLM이 복잡한 알고리즘적 제약을 이해하고 기존의 휴리스틱과 비교할 수 있는지를 검증합니다.

- **Performance Highlights**: LLM에 의해 생성된 휴리스틱 솔루션은 모든 평가 메트릭에서 뛰어난 성능을 나타내며, 전통적인 방법에 비해 bin 사용량을 6.25% 줄이고, 공간 활용도를 HFF보다 6.4% 개선했습니다. LLM은 최적 솔루션을 2회의 반복 내에 도달하며, 이는 강력한 패턴 인식 및 제약 만족 학습 능력을 나타냅니다. 이러한 결과는 LLM의 조합 최적화 과제에 대한 평가 메트릭을 제시하고 기본 선례를 제공하는 데 기여합니다.



### Clinical Uncertainty Impacts Machine Learning Evaluations (https://arxiv.org/abs/2509.22242)
- **What's New**: 본 논문은 의료 데이터셋에서 주석이 불확실한 특성을 가진다는 점을 강조하면서, 기계 학습 평가에 있어 이러한 불확실성을 수치적으로 반영해야 한다고 주장합니다. 기존의 다수결 및 임계값 설정 같은 집계 방법은 주석의 본질적인 변동성을 흐리게 만들어 잘못된 평가를 유발할 수 있습니다. 따라서, 저자들은 확률 기반의 측정방법을 도입하여 더 나은 모델 평가를 촉진하고자 합니다.

- **Technical Details**: 논문에서는 soft metrics(불확실성 인식 지표)의 정의를 제시합니다. 이 지표들은 몇 가지 다른 주석 프로세스와 관계없이 확률 값에서 직접 작용하며, 선형 시간 내에 구현할 수 있는 계산적으로 효율적인 방법입니다. 또한, 저자들은 소프트 AP와 소프트 AUROC 같은 확률적 레이블을 위한 여러 가지 메트릭 확장을 제안하여 데이터셋 간의 결과 비교를 더 정확하게 할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 불확실성을 반영한 평가 방식이 기존의 이진 레이블을 사용한 평가 방식보다 모델 간의 순위에 상당한 영향을 미친다고 밝혀졌습니다. 특히, 다양한 의료 이미지 벤치마크에서 레이블 불확실성을 고려할 경우, 제출된 모델들의 순위가 조정되는 현상이 나타났습니다. 이 관점을 통해 논문은 기술 커뮤니티가 불확실성 인식을 기반으로 한 평가 관행을 채택할 것을 촉구합니다.



### Log2Plan: An Adaptive GUI Automation Framework Integrated with Task Mining Approach (https://arxiv.org/abs/2509.22137)
- **What's New**: Log2Plan은 GUI(그래픽 사용자 인터페이스) 작업 자동화의 한계를 극복하기 위해 새로운 구조의 이중 수준 계획 프레임워크와 사용자 행동 로그를 기반으로 한 작업 마이닝 방식을 결합하였습니다. 기존의 LLM(대형 언어 모델)과 VLM(비주얼-언어 모델) 기반의 계획자-실행자 에이전트들이 다양한 작업에서 겪었던 부서지기 쉬운 일반화와 높은 지연 시간 문제를 해결합니다. 사용자 명령을 구성하는 높은 수준의 계획을 세우고, 이를 실제 GUI 환경에 기반하여 적절한 저수준 행동 시퀀스로 변환합니다.

- **Technical Details**: Log2Plan은 원시 GUI 로그를 의미 있는 행동-객체 단위로 분할하고, 자주 반복되는 패턴을 발견하여 모듈형 자동화 흐름을 생성하는 등 사용자 로그에서 전반적인 작업 구조를 파악하는 기능을 갖추고 있습니다. 이 시스템은 고수준의 의도를 낮은 수준의 실행으로 분리하여 사용자 인터페이스의 변화에 쉽게 적응하고 강력한 실행을 보장합니다. 계획자와 실행자의 별도 운영으로 LLM이 구조화된 행동-기반의 표현에서 작업을 수행하도록 합니다.

- **Performance Highlights**: 200개의 실제 작업을 대상으로 한 평가에서 Log2Plan은 작업 성공률과 실행 시간에서 상당한 개선을 보였습니다. 특히, 장기 작업 시퀀스에서도 60% 이상의 성공률을 유지하여 복잡한 다단계 워크플로우에 대한 강력함을 강조하고 있습니다. 이러한 성능 개선은 사용자 맞춤형 경험과 반복 사용을 지원하는 데 중요한 역할을 합니다.



### Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements (https://arxiv.org/abs/2509.22092)
- **What's New**: 이 논문에서는 AI 및 머신러닝(ML) 모델의 에너지 소비와 탄소 배출을 추정하는 기존 도구들의 정확성을 평가합니다. 기존의 ML Emissions Calculator와 CodeCarbon은 사용하기 쉽지만, 에너지 소비를 과소 또는 과대 평가할 수 있는 한계를 가지고 있다는 점을 지적합니다. 이러한 연구는 AI의 지속 가능성을 위한 중요한 기여로, AI 연구자들에게 에너지 사용을 더 잘 이해하도록 도와줄 정보를 제공합니다.

- **Technical Details**: 연구에서는 다양한 AI 실험의 실제 에너지 소비를 기반으로 한 정적 및 동적 자원 추정 방법의 신뢰성을 평가합니다. 정적 접근법은 하드웨어의 전력 소비를 일정하게 간주하기 때문에 실험 환경에 따라 불일치를 초래할 수 있습니다. 반면, 동적 추정은 CPU/GPU의 실제 소비 전력을 프로파일링하여 에너지 소비를 추정하지만, 전원 공급 장치, 냉각장치와 같은 추가 하드웨어의 소비를 무시하는 경우가 많습니다.

- **Performance Highlights**: 연구 결과, 정적 및 동적 추정 도구는 대부분의 실험에서 최대 40%까지 에너지 소비를 잘못 추정하는 것으로 밝혀졌습니다. 이 연구는 AI 에너지 소비와 추정 오류에 대한 경험적 증거를 제공하며, 이러한 도구들의 정확성을 개선하기 위한 가이드라인과 코드도 제공합니다. 결과적으로, 지속 가능한 AI 개발을 위한 투명성을 확보하고 자원 기반 ML 및 AI 지속 가능성 연구에 중요한 기여를 하였습니다.



### Generalizing Multi-Objective Search via Objective-Aggregation Functions (https://arxiv.org/abs/2509.22085)
- **What's New**: 이번 연구에서는 다목적 탐색(Multi-objective search, MOS)의 새로운 일반화된 문제 форм식을 제안하고, 이를 통해 숨겨진 목표(hidden objectives)와 해결 목표(solution objectives)를 분리하여 성공적으로 활용할 수 있는 방법을 제시합니다. 이 새로운 접근 방식은 복잡한 목표 간의 상호작용을 효과적으로 모델링하면서도 기존의 최첨단 MOS 알고리즘과의 호환성을 유지합니다. 또한, 세부 조정이 최소화된다는 점에서 알고리즘의 적용이 용이해집니다.

- **Technical Details**: 이 연구의 핵심 통찰은 숨겨진 목표와 해결 목표를 구별하는 것으로, 이를 통해 문제를 해결하는 과정에서 발생하는 여러 가지 목표 간의 상호작용을 효과적으로 나타낼 수 있게 됩니다. 숨겨진 목표는 탐색 프로세스 중 점진적으로 계산되며, 해결 목표는 이러한 숨겨진 목표에 대한 집계 함수(aggregation function)를 통해 정의됩니다. 따라서 이 새로운 형식은 표준 MOS 알고리즘의 핵심 작업을 최소한으로 연장하는 것만으로도 적용 가능성을 열어줍니다.

- **Performance Highlights**: 제안된 형식은 다양한 로봇 계획 문제에 적용되어 숨겨진 목표에 대한 충돌 확률을 집계하여 총 충돌 확률이라는 해결 목표를 도출하는 데 성공합니다. 이러한 새로운 접근 방식은 기존의 MOS 알고리즘에 비해 성능을 획기적으로 개선하는 결과를 보였으며, 실험적으로 그 효과가 입증되었습니다. 연구 결과는 로봇 공학 분야에서의 복잡한 탐색 문제에 대한 효율적인 솔루션을 제공함으로써 중요한 기여를 합니다.



### A2R: An Asymmetric Two-Stage Reasoning Framework for Parallel Reasoning (https://arxiv.org/abs/2509.22044)
Comments:
          15 pages, 3 figures

- **What's New**: 이번 연구에서는 비대칭 이단계 추론 프레임워크인 A2R(Asymmetric Two-Stage Reasoning)을 제안합니다. 이 프레임워크는 모델의 잠재력과 실제 성능 간의 격차를 해소하기 위해 설계되었습니다. A2R은 첫 번째 단계에서 솔루션 후보를 생성하는 '탐색기'(explorer) 모델과, 두 번째 단계에서 이들을 통합하는 '합성기'(synthesizer) 모델로 구성되어 있습니다. 이를 통해 기존의 순차적 접근법과는 다른 방식으로 계산량을 확장할 수 있습니다.

- **Technical Details**: A2R은 두 가지 상호보완적인 단계로 추론을 분리합니다: 첫 번째 단계는 다양한 추론 경로를 생성하는 '탐색' 단계이고, 두 번째 단계는 통합하는 '합성' 단계입니다. A2R은 이전의 단순 투표나 선택을 통한 병렬 추론 접근 방식과 달리 전체 추론 경로 집합에 대해 생성적인 재추론을 수행합니다. 이를 통해 후보 솔루션의 전체적인 시각을 형성하고, 일관된 증거를 판단하며, 보다 정확하고 견고한 최종 답변을 통합할 수 있습니다.

- **Performance Highlights**: A2R을 AIME 2024와 같은 복잡한 추론 벤치마크에 적용한 결과, Qwen3-8B-distill 모델을 사용하여 4개의 추론 경로에서 75% 성능 향상을 달성했습니다. 또한, A2R-Efficient 구조를 통해 더 작은 탐색기와 더 강력한 합성기를 조합하여 평균 30% 낮은 비용으로 단일 모델과 동등한 성능을 얻을 수 있었습니다. 이 연구 결과는 A2R이 단순한 성능 향상 프레임워크일 뿐만 아니라, 실제 적용에도 효율적이고 실용적인 솔루션이라는 것을 보여줍니다.



### The Thinking Spectrum: An Emperical Study of Tunable Reasoning in LLMs through Model Merging (https://arxiv.org/abs/2509.22034)
- **What's New**: 이 연구는 다양한 추론 깊이와 비용을 균형화할 수 있는 대형 언어 모델(LLM)의 모델 병합 기법을 새롭게 탐구합니다. 기존의 다양한 병합 기법 중 이 연구는 각각의 모델이 갖고 있는 특징을 조합하여 정확성과 효율성을 동시에 향상시키는 새로운 접근 방식을 제시합니다. 이번 연구는 모델의 병합 강도를 체계적으로 변화시켜 성능과 효율성을 분석하는 첫 번째 포괄적 연구를 제공합니다.

- **Technical Details**: 연구팀은 Linear, SLERP, TIES, TWIN, EMR, DARE, LORE와 같은 일곱 가지 대표적인 모델 병합 기법을 평가했습니다. 또한, AIME24, AIME25, HMMT25와 같은 다섯 가지 다양한 벤치 마크에서 이 기법들의 효율성 곡선을 구성하였습니다. 모델 병합은 특정 요구에 맞춰 추론 효율성과 정확도의 균형을 맞출 수 있는 방법으로, 각 기법을 통해 더욱 세밀한 제어가 가능합니다.

- **Performance Highlights**: 모델 병합 기술은 고도로 다양한 매개변수를 조합하여 티어레벨에서 높은 정확도와 더 낮은 토큰 소비를 달성하는 것으로 확인되었습니다. 특히, 병합된 모델이 원래 모델보다 더 좋은 성능을 보이는 Pareto 개선 사례가 다수 발견되었습니다. 이 연구는 모델 병합의 적용 가능성과 실용적인 가이드라인을 제시하여 진화하는 LLM의 요구에 대응할 수 있는 방법을 보여줍니다.



### GSM-Agent: Understanding Agentic Reasoning Using Controllable Environments (https://arxiv.org/abs/2509.21998)
Comments:
          35 pages, 8 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)이 믿을 수 있는 에이전트로 기능하기 위해 중요한 능력인 agentic reasoning(에이전틱 추론)에 대한 새로운 벤치마크, GSM-Agent를 소개합니다. 기존의 에이전트 벤치마크는 수학적 추론 및 전문가 수준의 지식과 같은 복잡한 작업과 혼합되어 있었으나, GSM-Agent는 학년 수준의 문제를 해결해야 하며, 프롬프트에서 정보 없이 질문만 제시됩니다.

- **Technical Details**: GSM-Agent는 문제를 질문과 여러 전제로 분해하여 환경을 구축하고, 에이전트가 제공된 도구(검색 도구 및 다음 페이지 도구)를 사용하여 관련 문서를 찾아야 하는 구조입니다. 연구 결과에 따르면 GPT-5와 같은 첨단 모델이 평균적으로 33%의 정확도를 잃는 등, 기존의 정적 설정과 에이전틱 설정 간에 성능 차이가 큽니다.

- **Performance Highlights**: 에이전틱 추론 그래프라는 개념을 도입하여 문서 임베딩을 클러스터링하고 도구 호출을 매핑하여 추론 경로를 구축함으로써 에이전틱 추론을 정량적으로 분석합니다. 연구진은 에이전틱 환경에서 다음 노드를 재방문(revisit)하는 능력이 정확도와 밀접하게 관련되어 있음을 발견하였고, 도구를 강화하는 방법을 통해 LLM의 에이전틱 추론 성능을 향상시킬 수 있음을 제안합니다.



### Bilinear relational structure fixes reversal curse and enables consistent model editing (https://arxiv.org/abs/2509.21993)
Comments:
          9 pages

- **What's New**: 이 논문에서는 언어 모델(LM)이 역전 저주(reversal curse)를 극복할 수 있는 새로운 방법을 제시합니다. 기존에는 "A는 B의 부모이다"라는 사실로부터 "B는 A의 자식이다"라는 사실을 유추할 수 없는 것으로 간주되었습니다. 그러나 이 연구에서는 지식 인코딩 방식에서 발생하는 문제라고 주장하며, 관계형 지식 그래프를 통한 훈련을 통해 이러한 역전 저주를 완화할 수 있음을 보입니다.

- **Technical Details**: 연구에서는 bilinear 관계 구조(bilinear relational structure)가 LM의 내부 표현에서 나타난다는 것을 입증합니다. 이 구조는 관계를 나타내는 행렬을 통해 역 관계를 자연스럽게 포착하고, 관계를 조합하는 데 뛰어난 성능을 보입니다. 연구자들은 시뮬레이션된 지식 그래프를 활용하여 LM이 역전 저주를 극복하며 일반화된 편집을 수행할 수 있다는 것을 발견했습니다.

- **Performance Highlights**: LM의 내부 표현을 조사한 결과, bilinear 프로브(bilinear probe)가 가장 좋은 설명력을 가지며, 수정이 논리적으로 관련된 사실들에 잘 전파된다는 사실이 드러났습니다. 이러한 구조를 가진 모델은 역전 저주를 극복하고 사실을 업데이트하는 데 있어 더 높은 일관성을 보여줍니다. 이는 모델 편집의 성공이 단순한 알고리즘보다 지식 표현의 기하학적 구조에 크게 의존함을 시사합니다.



### RISK: A Framework for GUI Agents in E-commerce Risk Managemen (https://arxiv.org/abs/2509.21982)
- **What's New**: 이번 논문에서는 e-commerce 리스크 관리의 복잡한 웹 상호작용을 자동화하기 위한 새로운 프레임워크인 RISK를 소개합니다. 기존의 GUI 에이전트가 단일 단계 작업에 한정되는 반면, RISK는 다단계 상호작용을 지원하여 리스크 평가에 필수적인 동적 콘텐츠를 효과적으로 관리할 수 있습니다. RISK는 데이터셋, 벤치마크 및 강화 학습 기반의 세 가지 주요 구성 요소로 구성되어 있습니다.

- **Technical Details**: RISK 프레임워크는 (1) RISK-Data, (2) RISK-Bench, (3) RISK-R1으로 구성되어 있습니다. RISK-Data는 총 8,492개의 단일 단계와 2,386개의 다단계 상호작용 경로로 이루어진 데이터셋이며, RISK-Bench는 802개의 단일 단계와 320개의 다단계 상호작용으로 평가됩니다. RISK-R1은 서로 다른 난이도의 작업을 고려한 강화 학습을 통해 GUI 에이전트의 학습 프로세스를 개선합니다.

- **Performance Highlights**: RISK-R1의 실험 결과, 기존 방법보다 단일 단계 성능이 6.8%, 다단계 성능이 8.8% 향상되었습니다. 온라인 평가에서도 70.5%의 높은 성공률을 기록하며, RISK는 e-commerce 리스크 관리 분야에서 최신 기술 수준을 향상시키고 있습니다. 이번 연구는 GUI 에이전트가 현실 세계의 복잡한 웹 상호작용을 자동화할 수 있는 강력한 도구임을 입증하였습니다.



### CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration (https://arxiv.org/abs/2509.21981)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 활용하여 다중 에이전트 협업의 효율성을 개선하기 위한 새로운 프레임워크인 CoBel-World를 제안합니다. 이 프레임워크는 협업 신념 세계(collaborative belief world)를 통해 에이전트가 물리적인 환경과 동료의 정신 상태를 함께 모형화할 수 있게 해줍니다. 이를 통해 에이전트는 더 정확한 계획 수립과 의사 소통 불일치를 방지할 수 있습니다.

- **Technical Details**: CoBel-World는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 상징적 신념 표현(symbolic belief representation)은 에이전트가 태스크 요구사항을 해석하고 환경 지식을 구조화된 신념 규칙으로 인코딩할 수 있게 해줍니다. 둘째, 베이지안 신념 협업(Bayesian belief collaboration)은 에이전트가 태스크 수행 중에 신념 세계를 동적으로 업데이트하여 통신 비용을 줄이고 협업 의사 결정을 향상시킵니다.

- **Performance Highlights**: CoBel-World는 TDW-MAT 및 C-WAH와 같은 도전적인 기준에서 평가되었으며, 비슷한 방법들에 비해 통신 비용을 22-60% 줄이고, 작업 완료 효율성을 4-28% 향상시켰습니다. 이러한 결과는 신념 기반 협업이 다중 에이전트 시스템에서 효율적이고 인간과 유사한 협력을 가능하게 한다는 것을 보여줍니다.



### Outlier Detection in Plantar Pressure: Human-Centered Comparison of Statistical Parametric Mapping and Explainable Machine Learning (https://arxiv.org/abs/2509.21943)
- **What's New**: 이번 연구는 Plantar pressure mapping(발바닥 압력 매핑)의 중요성을 강조하는 동시에, SPM(Statistical Parametric Mapping)의 한계를 보여줍니다. 특별히, 비모수적이고 등록 의존적인 SPM 접근법과 CNN(Convolutional Neural Network)을 SHAP(SHapley Additive exPlanations)로 설명하는 기계 학습 접근법을 비교했습니다. 연구는 여러 센터에서 수집된 데이터를 다루며, 합의된 전문가의 주석과 인위적인 이상을 추가하여 798개의 유효 샘플과 2000개의 외부 요인을 확보했습니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 사용하여 데이터를 분석했습니다. 첫 번째는 등록 의존적인 비모수적 SPM 방식이며, 두 번째는 SHAP로 설명되는 CNN 모델입니다. 이들은 서로 다른 품질 관리 파이프라인을 제공하며, SPM은 정렬에 민감하고 외부 요인 탐지의 강력함은 불확실합니다.

- **Performance Highlights**: 성능 평가는 중첩 교차 검증(nested cross-validation)을 통해 이루어졌고, 전문가와 함께한 의미 론적 차별 조사로 설명 품질이 평가되었습니다. 기계 학습 모델은 높은 정확도를 기록하며 SPM을 초월하는 성능을 보였고, 전문가들은 SPM과 SHAP의 설명이 명확하고 유용하며 신뢰할 수 있다고 평가했습니다. 이러한 결과는 SPM과 설명 가능한 ML이 발바닥 압력 데이터의 자동 외부 요인 탐지에 상호 보완적일 수 있음을 보여줍니다.



### DyRo-MCTS: A Robust Monte Carlo Tree Search Approach to Dynamic Job Shop Scheduling (https://arxiv.org/abs/2509.21902)
- **What's New**: 본 논문에서는 새로운 작업의 도착으로 인한 불확실성을 관리하기 위해 동적 강건 몬테 카를로 트리 탐색(Dynamic Robust MCTS, DyRo-MCTS) 방법을 제안합니다. 이 방법은 작업 환경을 양호한 스케줄링 결과 뿐만 아니라 미래 작업 도착에 쉽게 적응할 수 있는 상태로 안내하도록 설계되었습니다. DyRo-MCTS는 기존의 최첨단 기술들과의 비교에서 강력한 성능을 보이며, 최소한의 추가적인 온라인 계획 시간을 사용할 수 있습니다.

- **Technical Details**: DyRo-MCTS는 행동 강건성(robustness)을 평가하여 메타 휴리스틱에 통합하고, MCTS의 트리 정책(Tree Policy) 내에서 온라인 환경에서의 직면할 가능성이 있는 방해 요소를 고려합니다. 이를 통해 작업 도착의 불확실성을 효과적으로 처리가 가능합니다. DyRo-UCT(Dynamic Robust Upper Confidence bound for Trees)는 탐색 전이, 활용(exploration, exploitation) 및 강건성 간의 균형을 능동적으로 조절하는 능력을 보여줍니다.

- **Performance Highlights**: 실험 결과, DyRo-MCTS는 오프라인 학습된 정책의 성능을 현저히 개선하였으며, 약간의 추가 시간으로 이루어진 온라인 계획 과정에서도 효과적이었습니다. 또한, DyRo-MCTS는 다양한 스케줄링 시나리오에서 기본 MCTS보다 일관되게 우수한 성능을 발휘합니다. 강건한 스케줄링 결정의 중요성을 강조하며, 불확실성 하에서도 적응 가능한 생산 환경을 유지하는 것이 장기적으로 이점이 있다고 제안합니다.



### GenesisGeo: Technical Repor (https://arxiv.org/abs/2509.21896)
- **What's New**: GenesisGeo는 유클리드 기하학을 위한 오픈소스 자동 정리 증명기로, 2천 180만 개의 기하 문제를 포함하는 대규모 데이터셋을 오픈소스로 제공합니다. 또한, DDARN의 기호 추론 엔진을 120배 가속화하며, Qwen3-0.6B-Base 모델을 기반으로 하는 신경-기호 증명기를 구축하여 IMO-AG-30 벤치마크에서 30개 중 24문제를 단일 모델로 해결했습니다.

- **Technical Details**: GenesisGeo의 데이터 생성 절차는 기하학적 구성의 무작위 샘플링으로 시작됩니다. 이러한 구성에서 기하 도형을 생성하고, 기호 엔진을 통해 가능한 모든 결론을 도출한 후, 낮은 품질의 결과를 제거합니다. 최종적으로 21.8백만 개의 기하 정리와 증명으로 구성된 데이터셋을 생성하였으며, 이 중 300만 개 이상은 보조 구성요소를 포함합니다.

- **Performance Highlights**: DDARN 엔진의 성능 향상을 위해, 탐색 공간을 크게 줄이는 전처리를 포함해 효율적인 부분 매칭 기법을 도입했습니다. 이러한 최적화로 인해 GenesisGeo는 기존 DDARN 엔진보다 120배 빠른 속도를 달성하였으며, 두 가지 모델의 앙상블을 통해 IMO 금메달 수준의 26문제를 해결했습니다.



### TRACE: Learning to Compute on Graphs (https://arxiv.org/abs/2509.21886)
- **What's New**: 새로운 패러다임 TRACE는 계산 그래프의 기능적 동작을 모델링하는 기본적인 도전을 해결하기 위해 설계되었습니다. 기존의 MPNNs 및 Transformer 기반 모델들이 가진 아키텍처적 결함을 극복하고, 단계별 계산 흐름을 반영하는 계층적 Transformer를 사용하여 아키텍처가 정렬된 강력한 기반 위에 성립됩니다. TRACE는 또한 복잡한 전역 함수 예측 문제를 단순화하는 function shift learning(기능 이동 학습)이라는 새로운 학습 목표를 도입하여, 올바른 계산 표현을 학습할 수 있도록 합니다.

- **Technical Details**: TRACE는 최신 전자 회로 설계 및 분석을 위해 고안된 새로운 알고리즘입니다. 이는 입력의 독립성을 가정한 간단한 지역 근사치와 실제 전역 함수 간의 차이를 예측하도록 구성되어 있습니다. 이 구조는 계층적 Transformer의 도움으로 각 계산 단계를 위치 인식과 함께 표현하며, 재귀적으로 처리함으로써 계산 의존성을 고려하게 됩니다.

- **Performance Highlights**: TRACE의 성능은 여러 전자 회로의 벤치마크 시험에서 입증되었습니다. RTL, AIG, PM 넷리스트 등 다양한 회로 호환성에서, TRACE는 기존의 모든 모델에 비해 일관성 있게 개선된 결과를 보여줍니다. 이는 TRACE가 회로 분석을 위한 새로운 최첨단 기술일 뿐만 아니라, 계산 그래프에서 학습하는 데 더 robust하고 아키텍처적으로 견고한 패러다임임을 확립합니다.



### Reimagining Agent-based Modeling with Large Language Model Agents via Shach (https://arxiv.org/abs/2509.21862)
- **What's New**: 이번 연구에서는 다수의 에이전트 시스템에서 대형 언어 모델(LLM)의 긴장감 있는 행동을 연구하기 위한 새로운 방법론인 Shachi를 소개합니다. Shachi는 에이전트의 정책을 구성 요소별로 분해하여 심층적인 분석을 가능하게 합니다. 이를 통해 실험의 통제성과 재현성을 강화하여 에이전트의 집합적 행동에 대한 체계적인 분석을 지원합니다.

- **Technical Details**: Shachi는 LLM, 구성 모듈(Configuration), 기억(Memory), 도구(Tools)라는 네 가지 핵심 구성 요소를 기반으로 한 표준화된 에이전트 아키텍처를 제공합니다. 이러한 모듈식 설계를 통해 연구자들은 에이전트의 내부 구조를 외부 환경과 분리하여 분석할 수 있습니다. 본 접근법은 에이전트 간의 상호작용 및 통신을 통합하여 향상된 결과를 도출하도록 합니다.

- **Performance Highlights**: Shachi는 10개의 벤치마크 과제를 통해 검증되었으며, 이는 LLM 에이전트의 외부 유효성을 입증하는 예시로서, 미국의 관세 충격을 모델링하는 실험을 포함합니다. 또한, 에이전트가 새 환경으로 기억을 지닌 상태로 이식되거나, 두 개의 세계에서 동일하게 살아가는 방식의 행동 분석을 통해 기존의 연구 접근법으로는 불가능했던 새로운 과학적 탐구를 가능하게 합니다.



### DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents (https://arxiv.org/abs/2509.21842)
Comments:
          Under review

- **What's New**: 최근의 여행 계획(TP) 에이전트는 사용자 경험을 향상시키기 위해 외부 도구 및 자원과 상호 작용하는 새로운 구성 요소로 부상하였습니다. 본 논문에서는 사용자의 선호를 반영하여 여행 일정을 자율적으로 생성할 수 있는 DeepTravel이라는 에이전틱 강화 학습(reinforcement learning) 프레임워크를 제안합니다. 이는 여행 계획에서 유연하고 자율적인 에이전트를 구축하는 데 필요한 여러 도전 과제를 해결합니다.

- **Technical Details**: DeepTravel은 사용자 쿼리에 따라 여행 일정을 자동으로 작성하고, 도구 호출 및 응답을 통해 중간 작업을 탐색하고 검증하는 multi-turn reasoning 방식의 에이전트를 구축합니다. 이를 위해 Robust Sandbox 환경을 조성하고, 이동 경로 검증기와 세부 검증기를 포함하는 계층적 보상 모델링 시스템을 개발하여 신뢰할 수 있고 효율적인 보상 신호를 제공합니다. 또한 Reply-Augmented Reinforcement Learning 방법을 통해 에이전트의 추론 능력을 향상시킵니다.

- **Performance Highlights**: DeepTravel로 훈련된 TP 에이전트는 DiDi Enterprise Solutions App에 배포되어 온라인 및 오프라인 평가를 통해 안내되었습니다. 그 결과 소규모 LLM(예: Qwen3-32B)이 OpenAI의 최신 모델(o1, o3) 및 DeepSeek R1와 같은 기존 모델을 능가하는 성과를 보여주었습니다. 이는 기존 여행 계획 연구를 진전시키기 위한 유망한 프레임워크로서의 DeepTravel의 가능성을 검증합니다.



### Axiomatic Choice and the Decision-Evaluation Paradox (https://arxiv.org/abs/2509.21836)
- **What's New**: 이번 연구에서는 의사결정을 모델링하기 위한 프레임워크를 소개합니다. 특히 윤리적 제약(ethical constraints)과 같은 의사결정에 관한 공리를 기반으로 하는 분류법(taxonomy)을 제시하고, 의사결정을 수행하는 데 있어 공리를 사용하는 것과 의사결정을 평가하는 것 간의 갈등 관계인 'Decision-Evaluation Paradox'를 보여줍니다. 이 역설은 현실적인 공리 구조와 관련이 있으며, 의사결정 데이터를 학습하거나 공리를 적용할 때 특히 주의해야 함을 강조합니다.

- **Technical Details**: 우리는 의사결정과 공리를 의사결정의 속성(properties of decisions)으로 정의하여 이를 설명합니다. 이 프레임워크는 전통적인 사회적 선택(Social Choice) 위에 추상화된 계층(layer of abstraction)을 제공하여 개인 및 집단의 의사결정을 모델링할 수 있도록 합니다. 특히 공리에 대한 정의를 넓혀, 규칙(properties of rules)의 속성을 넘어서는 윤리적 원칙과 규범적 입장을 포괄하도록 하고 있습니다.

- **Performance Highlights**: 우리는 새로운 프레임워크를 통해 의사결정과 평가 간의 갈등을 명확히 드러내며, 이러한 갈등이 특정한 공리의 형식적 성질에서 비롯된다는 점을 강조합니다. 또한, 특정 공리를 함수적(input-output pairs) 관점에서 정의할 수 없음을 보여주어, 이를 기반으로 학습된 모델이 이러한 공리 간 구별에 필요한 충분한 정보를 갖추지 못하게 되는 문제를 논의합니다. 이러한 인과적 관계는 AI Alignment의 중요성을 재조명하고 실질적인 의사결정에서의 해석 가능성을 저해할 수 있습니다.



### DS-STAR: Data Science Agent via Iterative Planning and Verification (https://arxiv.org/abs/2509.21825)
- **What's New**: 본 논문에서는 데이터 과학 작업을 수행하기 위한 새로운 에이전트인 DS-STAR를 소개합니다. DS-STAR는 다양한 데이터 형식에서 컨텍스트를 탐색하고 추출하는 데이터 파일 분석 모듈과, 분석 계획의 충분성을 평가하는 LLM 기반의 검증 단계를 포함합니다. 특히, 이 시스템은 순차적 계획 메커니즘을 이용하여 초기 계획을 단순하게 세우고, DS-STAR의 피드백에 따라 이를 반복적으로 수정합니다.

- **Technical Details**: DS-STAR의 주요 기여는 크게 세 가지로 나눌 수 있습니다: 1) 다양한 데이터 형식에서 정보를 자동으로 추출하는 데이터 분석 모듈, 2) 각 단계에서 LLM 기반의 검사자가 분석 계획의 충분성을 평가하는 검증 단계, 3) 문제 해결을 위해 초기 계획에서 출발하여 피드백을 바탕으로 점진적으로 수정하는 반복적 계획 메커니즘입니다. 이 접근 방식은 복잡한 데이터 분석을 성공적으로 내비게이션할 수 있도록 합니다.

- **Performance Highlights**: DS-STAR는 DABStep, KramaBench, DA-Code와 같은 세 가지 벤치마크에서 가장 최신 성능을 달성했습니다. 특히, DABStep 벤치마크에서 DS-STAR는 정확도를 41.0%에서 45.2%로 향상시켰고, KramaBench에서는 39.8%에서 44.7%로, DA-Code에서는 37.0%에서 38.5%로 개선되었습니다. 이 실험 결과는 DS-STAR가 복잡한 데이터 분석 작업에서 기존 방법보다 뛰어난 성과를 낸다는 것을 보여줍니다.



### ProRe: A Proactive Reward System for GUI Agents via Reasoner-Actor Collaboration (https://arxiv.org/abs/2509.21823)
Comments:
          10 pages, 7 figures

- **What's New**: 이 논문에서는 GUI 에이전트를 위한 정밀한 보상을 창출하기 위한 새로운 접근 방식인 ProRe를 제안합니다. 기존의 규칙 기반 또는 모델 기반 보상 방법은 GUI 에이전트에 일반화하는 데 한계를 보였습니다. ProRe는 일반 목적의 reasoning agent와 도메인 특정 evaluator agent를 결합하여 보다 정확하고 검증 가능한 보상을 제공합니다.

- **Technical Details**: ProRe 시스템은 상태 probing 작업을 사전에 계획한 후, 이 작업을 수행하는 evaluator agents가 환경과 상호 작용하여 추가 관찰을 수집하도록 설계되었습니다. 이를 통해 reasoner는 GUI 에이전트에 보다 정확한 보상을 부여할 수 있습니다. 결과적으로 ProRe는 3,000개 이상의 궤적에서 보상 정확도와 F1 점수를 최대 5.3% 및 19.4% 향상시켰습니다.

- **Performance Highlights**: Empirical results show that ProRe achieves an impressive average accuracy of 93.7%, surpassing 90% accuracy for the first time in reward systems. Additionally, integrating ProRe into state-of-the-art policy agents leads to an enhancement in success rate by up to 22.4%, highlighting its effectiveness for real-world applications.



### D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents (https://arxiv.org/abs/2509.21799)
- **What's New**: 이 논문에서는 D-Artemis라는 새로운 심사적(Deliberative) 프레임워크를 소개합니다. D-Artemis는 사용자 인터랙션을 모방하여 인간 작업을 자동화하는 GUI 에이전트의 성능을 향상시키기 위해 고안되었습니다. 이 프레임워크는 기존의 데이터 병목 현상과 오류 탐지의 높은 비용, 상충하는 지침의 위험 등을 해결합니다.

- **Technical Details**: D-Artemis는 애플리케이션에 특화된 세밀한 팁 검색 메커니즘을 활용하여 의사결정 과정을 지원합니다. 초기 실행 전 단계(Pre-execution Alignment)에서 Thought-Action Consistency (TAC) 검사 모듈과 Action Correction Agent (ACA)가 함께 작동하여 실행 실패의 위험을 줄입니다. 또한, 실행 후 Status Reflection Agent (SRA)가 경험으로부터 전략적인 학습을 가능하게 하여 인지적 루프(Cognitive Loop)를 완성합니다.

- **Performance Highlights**: D-Artemis는 복잡한 궤적 데이터셋에 대한 훈련 없이도 멀티모달 대형 언어 모델(MLLM)의 GUI 작업 성능을 크게 향상시켰습니다. 이 프레임워크는 AndroidWorld에서 75.8%, ScreenSpot-V2에서 96.8%의 성공률을 기록하며 새로운 최첨단(State-of-the-Art, SOTA) 결과를 수립했습니다. 각 구성 요소의 기여도를 명확히 보여주는 포괄적인 제거 연구(Ablation Study)도 수행되었습니다.



### Benchmarking MLLM-based Web Understanding: Reasoning, Robustness and Safety (https://arxiv.org/abs/2509.21782)
- **What's New**: 이 논문에서는 웹 어플리케이션을 위한 포괄적인 이해 배치 평가 프레임워크인 WebRSSBench를 소개합니다. 이 프레임워크는 Reasoning, Robustness, Safety의 세 가지 핵심 능력을 평가하며, 실제 웹사이트 729개에서 3799개의 QA 샘플을 사용하여 모델의 성능을 검증합니다. 이를 통해 기존의 평가 기준에서 다루지 않았던 UI 요소 간의 관계와 의미적 이해를 포함한 새로운 평가 차원을 제시합니다.

- **Technical Details**: WebRSSBench는 위치 관계 추론, 양식 채우기, 힌트 텍스트 예측, UI 그룹화와 같은 네 가지 새로운 Reasoning 작업을 도입하여 MLLMs의 UI 이해 능력을 평가합니다. 또한, 레이아웃 재배치, 색상 변화, 텍스트 변형에 대한 세 가지 새로운 위협 평가 방식을 제안하여 모델의 Robustness를 평가합니다. 마지막으로, Safe Critical Detection을 통해 모델이 잠재적인 보안 위험 요소를 식별할 수 있는지를 측정하는 작업도 포함되어 있습니다.

- **Performance Highlights**: 12개의 최신 오픈소스 및 상용 MLLM에 대한 평가 결과, 모델들이 실제 레이아웃에서 조합적 및 크로스 요소 추론을 수행하는 데 어려움을 겪고 있으며, UI와 콘텐츠의 다양한 변동에 대해 제한된 Robustness를 보이는 것이 밝혀졌습니다. 특히 모델이 안전 위험이나 돌이킬 수 없는 작업을 인식하지 못하는 경향이 있어 보안 측면에서도 미비한 점이 발견되었습니다. 이러한 결과는 MLLMs의 실제 배치 준비 상태를 평가하는 데 있어 기존 기준의 한계를 강조합니다.



### UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios (https://arxiv.org/abs/2509.21766)
- **What's New**: 최근 자율 에이전트(agents)들은 다양한 분야에서 괄목할 만한 발전을 이루었지만, 대부분의 평가는 짧은 지평의 완전 관측(task) 작업에 집중되어 있습니다. 그러나 대규모 소프트웨어 개발, 상업 투자 및 과학 발견과 같은 현실 세계의 중요한 작업은 장기 지평(long-horizon) 및 부분 관측(partially observable) 시나리오에서 진행됩니다. 이러한 환경에서 성공은 지속적인 추론(reasoning), 계획(planning), 메모리 관리(memory management), 도구 사용(tool use)에 달려 있습니다.

- **Technical Details**: 이에 따라 우리는 복잡한 현실 세계의 도전에 필수적인 기초 능력을 측정하기 위한 새로운 벤치마크인 \textbf{UltraHorizon}을 소개합니다. 이 벤치마크는 세 가지 서로 다른 환경을 통해 탐사(exploration)를 통일된 작업으로 사용하여 이러한 핵심 역량을 검증합니다. 장기 발견 작업에서 설계된 에이전트는 숨겨진 규칙을 반복적으로 발견해야 하며, 평균적으로 \textbf{200k+} 토큰과 \textbf{400+} 툴 호출을 진행합니다.

- **Performance Highlights**: 우리의 실험 결과, LLM-agents는 이러한 설정에서 일관되게 낮은 성능을 보이는 반면, 인간 참가자는 더 높은 점수를 기록하여 에이전트의 장기 지평 능력에서 지속적인 격차를 강조합니다. 또한 단순한 스케일링(scaling)만으로는 이러한 작업을 해결할 수 없음을 발견하였습니다. 수집된 경로(trajectories)에 대한 심층 분석을 통해 여덟 가지 유형의 오류를 확인하였으며, 이를 두 가지 주요 원인인 컨텍스트 잠금(in-context locking)과 기능적 기본 능력의 격차로 귀속시켰습니다.



### Lifelong Learning with Behavior Consolidation for Vehicle Routing (https://arxiv.org/abs/2509.21765)
- **What's New**: 이번 연구에서는 여러 가지 문제 분포 및 규모를 다룬 새로운 임무에 대해 효과적으로 학습할 수 있는 새로운 지속적인 학습(paradigm) 프레임워크인 Lifelong Learning Router with Behavior Consolidation (LLR-BC)를 제안합니다. LLR-BC는 새로운 작업을 해결하면서 이전에 학습한 작업에서의 성능을 유지하는 방법을 모색합니다. 이 연구는 특히 VRP(차량 경로 문제) 및 TSP(여행하는 세일즈맨 문제)에서의 대규모 실험을 통해 카타스트로픽 포겟팅(catastrophic forgetting) 문제를 다룹니다.

- **Technical Details**: LLR-BC는 새로운 작업의 행동을 이전 작업에서의 행동과 정렬하여 선호하는 결정을 유지하는 방식으로 이전 지식을 효과적으로 통합합니다. 행동 통합(Behavior Consolidation) 과정에서는 자신감이 낮은 결정에 더 높은 가중치를 부여하여 중요한 경험에 더 집중하도록 권장합니다. 이 프레임워크는 지식의 축적을 통해 제로샷 일반화(zero-shot generalization) 능력을 개선하는 데에도 초점을 맞춥니다.

- **Performance Highlights**: 다양한 작업 순서 및 기본 신경 해결사에 대한 광범위한 실험 결과, LLR-BC는 카타스트로픽 포겟팅 문제를 완화하고 장기적인 학습(learning) 성능과 제로샷 일반화 능력을 유지하는 데 효과적임을 보여주었습니다. LLR-BC의 새로운 프레임워크는 기존의 연구와 비교하여 학습 및 일반화 능력을 향상시키는 혁신적인 접근 방식을 제시합니다.



### Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts (https://arxiv.org/abs/2509.21743)
- **What's New**: 본 논문에서는 Retrieval-of-Thought (RoT)라는 새로운 방법을 제안합니다. RoT는 이전의 추론을 재사용하여 새로운 문제를 안내하는 컴포저블 '생각' 단계로 구성되어 있습니다. 이 방법은 생각 그래프(thought graph)를 조직하여 신속한 검색과 유연한 재조합이 가능하도록 합니다. RoT는 동적 템플릿 재사용을 통해 중복 탐색을 줄이고, 정확성을 유지하면서 출력 토큰을 감소시킵니다.

- **Technical Details**: RoT의 구조는 추론 단계를 생각 그래프로 구성하여 메타데이터 기반 검색(metadata-based retrieval) 방법을 통해 빠르게 검색 공간을 좁히고, 보상 기반 탐색(reward-guided traversal) 알고리즘을 적용하여 서로 연결된 추론 패턴을 탐색합니다. 이로 인해 RoT는 문맥적으로 적절한 템플릿을 동적으로 생성합니다. 또한, RoT는 LRM이 과거의 '생각'을 유연하게 조합할 수 있도록 하여 효율적이고 일반화된 추론을 지원합니다.

- **Performance Highlights**: RoT는 여러 모델을 대상으로 한 추론 벤치마크에서 평가되었습니다. 결과는 출력 토큰 수를 최대 40% 줄이고, 추론 지연(latency)을 82% 감소시켰으며, 비용을 59% 절감하면서도 정확성을 유지했습니다. RoT는 동적 템플릿 구축을 통해 LRM 추론을 위한 확장 가능한 패러다임을 확립하며, Chain-of-Thought 및 retrieval 베이스라인과 비교하여 유사하거나 더 높은 정확성을 일관되게 달성합니다.



### Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Online Preference Optimization (https://arxiv.org/abs/2509.21718)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 본 논문에서는 낮은 자원 언어를 위한 텍스트 음성 변환(TTS) 모델의 개발을 위한 새로운 프레임워크인 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 제안합니다. 이 방법은 다국어 TTS 모델을 새로운 언어에 적응시키기 위해 다양한 기법을 활용하며, 저자들은 특정 언어의 억양 특징을 포착하는 것을 목표로 합니다. 실험을 통해 이러한 접근법이 저자원 언어에서 가청성과 화자 일관성을 갖춘 음성을 생성하는 데 효과적임을 입증했습니다.

- **Technical Details**: 제안된 프레임워크는 다단계 과정으로 구성되며, 세 단계의 접근법을 통해 다국어 TTS 모델을 저자원 언어에 적응시킵니다. 첫째, 국제 음성 기호(International Phonetic Alphabet, IPA) 기반으로 다국어 데이터 세트에서 기본 모델을 사전 훈련합니다. 이후, 제한된 양의 쌍 데이터로 TTS 모델을 미세 조정하여 대상 언어의 음운 및 억양 패턴을 캡처합니다. 마지막으로, GRPO를 적용하여 단순히 쌍이 아닌 텍스트와 화자 프롬프트를 기반으로 모델을 최적화합니다.

- **Performance Highlights**: 실험 결과, 저자원 언어에서 GRPO 기반 접근 방식은 기존의 미세 조정 접근 방식에 비해 월등히 뛰어난 성능을 보여주었습니다. 또한 GRPO는 고자원 언어의 TTS 성능도 향상시키며, 직접 선호 최적화(Direct Preference Optimization, DPO)와 같은 오프라인 정렬 방법보다 우수한 이해도와 화자 유사성, 음질을 자랑합니다. 이러한 발견은 온라인 강화 학습이 TTS 모델을 언어적 및 지각적 목표와 정렬하는 효과적인 메커니즘임을 시사합니다.



### Can AI Perceive Physical Danger and Intervene? (https://arxiv.org/abs/2509.21651)
- **What's New**: 이 논문은 AI가 물리적 세계와 상호작용할 때 새로운 안전 문제를 제기한다는 점을 강조합니다. 연구자들은 ASIMOV-2.0이라는 지속 가능한 물리적 안전 벤치마크를 개발하여, 로봇이 인간의 부상 위험을 평가하고 물리적 제약을 이해하며, 잠재적인 사고에 능동적으로 대응하는 능력을 측정합니다. 또한 주요 기초 모델의 안전 이해 능력을 평가하고, 모델들이 구체적인 물리적 안전 제약을 명시적으로 추론하도록 교육하는 후속 훈련 패러다임을 제시합니다.

- **Technical Details**: ASIMOV-2.0 벤치마크는 실제 부상 사례 및 운영 안전 제약에 기초하여 설계되었습니다. 이 벤치마크는 문서, 이미지, 비디오를 포함한 멀티모달 안전 이해를 평가하며, 이는 안전이 무시된 위험한 상태로의 전환을 포착한 고해상도 시나리오로 구성됩니다. 다양한 대형 언어 모델(LLM)에 대한 평가 결과가 포함되며, 특히 행동 및 이미지/비디오 모드에서의 낮은 점수를 발견하였습니다.

- **Performance Highlights**: 모든 모델이 '사고 모드'에서 안전 이해 과제를 수행하는 데 있어 개선된 성과를 보였습니다. 작은 모델도 더 큰 추론 시간 계산 예산으로 혜택을 입었으며, 이는 모델들이 안전성을 기초로 한 경로를 형성하는 것을 가능하게 합니다. 결과적으로 본 연구는 기초 모델이 안전 비판적 응용 프로그램에 배치될 준비 여부를 평가하는데 중요한 통찰력을 제공합니다.



### Semantic F1 Scores: Fair Evaluation Under Fuzzy Class Boundaries (https://arxiv.org/abs/2509.21633)
Comments:
          33 pages, 1 table, 29 figures, 4 algorithms

- **What's New**: 이번 논문에서는 Semantic F1 Score라는 새로운 평가 지표를 제안하여, 주관적이거나 모호한 멀티 레이블 분류에서 예측 레이블과 실제 레이블 간의 의미적 유사성을 정량화합니다. 기존의 F1 메트릭과 달리, Semantic F1은 레이블 유사도 행렬(label similarity matrix)을 사용하여 부드러운 정밀도와 재현율에 기반한 점수를 계산하며, 이 점수들은 보다 공정한 평가를 제공합니다. 이러한 접근법은 카테고리 간의 중첩을 인식하고, 주석 작성자 간의 불일치를 반영하여 보다 현실적인 평가를 가능하게 합니다.

- **Technical Details**: Semantic F1은 기존 F1 메트릭을 확장한 것이며, 두 단계(match) 방식으로 정밀도(semantic precision)와 재현율(semantic recall)을 계산합니다. 예측 레이블을 가장 가까운 실제 레이블에 매핑하고, 반대로 실제 레이블을 가까운 예측 레이블에 매핑하는 두 가지 단계를 통해 계산합니다. 이 두 단계 프로세스는 기존의 단일 단계 알고리즘보다 훨씬 강력하며, 레이블 공간의 과다 예측(over-prediction) 및 누락된 레이블 커버리지(under-coverage)를 모두 고려합니다.

- **Performance Highlights**: Semantic F1은 여러 합성 및 실제 데이터 연구에서 검증되었으며, 기존의 F1 메트릭보다 예측 성능을 더 잘 반영합니다. 예측 오류율에 선형적으로 감소하고, 유사성 행렬의 부분적인 계획 오류에도 강인성을 발휘하여 비메트릭 공간에서도 성공적으로 작동합니다. 또한, Semantic F1은 주관적 작업에 대한 대형 언어 모델(LLM)의 평가에서 더 강한 상관관계를 보여줍니다.



### Automated and Interpretable Survival Analysis from Multimodal Data (https://arxiv.org/abs/2509.21600)
Comments:
          4 figures; 4 tables; 24 pages

- **What's New**: 이 논문에서는 다모달 AI 프레임워크인 MultiFIX를 제안하여 생존 분석을 자동화하고 임상 변수와 CT 이미지를 통합합니다. 이 프레임워크는 해석 가능성을 높이기 위해 deep learning과 symbolic expressions를 사용하여 생존 관련 특징을 추출합니다. 또한, 해석 가능한 Cox 회귀를 통해 위험도를 추정해주며, 이는 환자군을 명확히 분류할 수 있게 합니다.

- **Technical Details**: MultiFIX는 다모달 데이터에서 해석 가능한 특징을 생성하기 위해 deep learning (DL)과 genetic programming (GP)을 결합한 구조입니다. DL은 데이터를 통해 모델을 훈련시키고 관련 특징을 엔지니어링하며, GP는 이를 기호 표현 형태로 변환하여 임상에서의 해석성을 제공합니다. 최종 결과로는 CTS (Clinical Trials System) 기반의 해석 가능한 위험 예측이 이루어지며, 이는 생존 곡선을 생성할 수 있습니다.

- **Performance Highlights**: MultiFIX 프레임워크는 RADCURE 데이터셋을 사용하여 구두암에 대한 생존 예측에서 C-index 0.838(예측) 및 0.826(구분) 값을 달성했습니다. 이는 기존의 임상 및 학술적 기준 방법들과 비교했을 때 우수한 성능을 보여주며, 알려진 예후 지표와 일치하는 결과입니다. 이러한 결과는 해석 가능한 다모달 AI가 정밀 의학에 대한 가능성을 가진다는 것을 강조합니다.



### GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models (https://arxiv.org/abs/2509.21593)
- **What's New**: GeoEvolve는 LLM(대형 언어 모델)에 기반한 알고리즘 발견 프레임워크로서, 지리적 문제를 해결하기 위해 진화적 검색과 지리적 도메인 지식을 결합한 혁신적인 접근 방식을 제공합니다. 이 프레임워크는 두 개의 루프, 즉 후보 솔루션을 생성하고 변형하는 내부 루프와 전 세계의 최고의 솔루션을 평가하는 외부 제어 루프를 통해 작동합니다. 이를 통해 지리적 알고리즘을 자동으로 설계하고 정제하여, 고전 모델을 개선하는 새로운 알고리즘을 발견합니다.

- **Technical Details**: GeoEvolve의 내부 루프는 코드 진화기(OpenEvolve)를 사용하여 후보 솔루션을 생성하고, 외부 루프는 최적의 솔루션을 평가하고 지리적 지식 데이터베이스인 GeoKnowRAG 모듈을 통해 이론적 지식을 주입합니다. 이를 통해 진화 과정을 지식에 기반하여 더 의미 있고 계산적으로 효율적인 알고리즘으로 유도할 수 있습니다. GeoEvolve는 공간 보간(kriging) 및 공간 불확실성 정량화(geospatial conformal prediction)와 같은 두 가지 기본 과제에서 평가되었습니다.

- **Performance Highlights**: GeoEvolve는 공간 보간에서 RMSE 오류를 13-21% 감소시키고, 불확실성 추정 성능을 17% 향상시킵니다. 또한 도메인 지식에 기반한 검색이 안정적이고 고품질의 진화를 위한 필수 요소임을 입증하는 강한 증거를 제공합니다. 이 결과는 GeoEvolve가 자동화된 지리적 모델링 과정에 대한 확장 가능한 경로를 제공하며, 신뢰할 수 있고 효율적인 과학적 발견의 새로운 기회를 열어준다는 것을 보여줍니다.



### EEG-Based Consumer Behaviour Prediction: An Exploration from Classical Machine Learning to Graph Neural Networks (https://arxiv.org/abs/2509.21567)
- **What's New**: 본 연구는 EEG (Electroencephalography) 데이터를 활용하여 소비자 행동을 예측하는 새로운 접근법을 제안합니다. 기존의 고전적인 기계 학습 모델과 더욱 혁신적인 Graph Neural Network (GNN) 모델을 비교하여 두 가지 모델의 성능 차이를 분석합니다. 특히, EEG 신호 분석과 기계 학습 모델의 결합을 통해 소비자 행동 예측에 대한 깊은 통찰을 제공합니다.

- **Technical Details**: 연구에서는 NeuMa 데이터셋으로부터 EEG 데이터의 특징을 추출하고 처리하여 뇌의 연결성을 모델링했습니다. GNN 구조를 활용하여 뇌 전극 간의 연결성(feature)을 기반으로 하여 서로 다른 GNN 아키텍처, 예를 들어 Graph Convolutional Network (GCN), Graph Attention Networks (GAT) 등을 사용하여 비교합니다. 이 외에도 XGBoost, LightGBM과 같은 고전적 모델의 성능도 평가해 전체적인 모델 성능을 종합적으로 분석합니다.

- **Performance Highlights**: 결과적으로, GNN 모델들이 특정 기본 지표에서 고전적 모델에 비해 일반적으로 더 나은 성능을 보였습니다. 특정 데이터 클래스에서 GNN이 더 두드러진 성과를 보여 준 반면, 전체적으로는 모델 간의 유의미한 차이는 없었습니다. 이러한 연구 결과는 EEG 기반 소비자 행동 예측에서 GNN 사용의 가능성을 보여줍니다.



### AutoClimDS: Climate Data Science Agentic AI -- A Knowledge Graph is All You Need (https://arxiv.org/abs/2509.21553)
- **What's New**: 이번 연구는 기후 데이터 과학의 장벽을 허물기 위해 큐레이션된 지식 그래프(knowledge graph, KG)와 클라우드 기반의 과학 워크플로우를 위한 AI 에이전트를 통합하는 개념 증명을 제시합니다. KG는 데이터셋, 도구 및 워크플로우를 조직하는 통합 계층을 제공하고, AI 에이전트는 자연어 상호작용 및 데이터 자동 접근을 가능하게 하여 비전문가가 기후 데이터를 분석할 수 있도록 합니다. 이 시스템의 오픈 소스 디자인은 커뮤니티 기여를 지원하며, 기후 데이터 접근을 민주화하고 AI와의 협업을 통해 재현 가능한 연구 프레임워크를 확보하는 길을 제시합니다.

- **Technical Details**: 연구에서는 OpenCypher를 활용해 NASA의 CMR 기록과 기관 카탈로그를 세분화된 그래프로 통합하는 방법론을 소개합니다. 관측 메타데이터를 표준화된 지구 시스템 모델 변수와 정합시키기 위해 ClimateBERT 기반의 정교한 변환기 분류기를 개발하여 99.17%의 의미적 정확도를 달성했습니다. 이를 통해 KG는 자동화된 기후 데이터 과학 AI 시스템인 AutoClimDS의 추론 기반으로 활용되어 자연어로 표현된 연구 목표에 따라 데이터 소스를 식별하고 메타데이터를 조정하여 분석 결과를 생성하는 방법을 보여줍니다.

- **Performance Highlights**: 이 연구는 기후 데이터 과학의 목표를 지원하기 위해 의도 인식 데이터 발견, 데이터 획득의 간소화, 그리고 재현 가능한 기후 모델링을 실현하는 세 가지 주요 목표를 설정합니다. AutoClimDS는 연구자들이 과학적 목표를 자연어로 입력하면 관련 데이터를 자율적으로 수집하고 재현 가능성을 높이는 분석 결과를 제공함으로써, 데이터 과학 작업의 접근성과 효율성을 크게 향상시킵니다. 이러한 접근법은 기후 연구의 혁신을 촉진하고, 검색뿐만 아니라 비즈니스와 학문 간의 협력을 크게 확대할 수 있습니다.



### Correct Reasoning Paths Visit Shared Decision Pivots (https://arxiv.org/abs/2509.21549)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문에서는 Chain-of-thought (CoT) 추론의 중간 사고 과정을 대형 언어 모델(LLM)에서 드러내고, 이러한 과정의 검증을 위한 해결책을 제안합니다. 이 논문은 결정 피벗(decision pivots)이라는 개념을 도입하는데, 이는 올바른 추론 경로가 반드시 거쳐야 하는 최소한의 검증 가능한 체크포인트입니다. 이를 통해 올바른 사고가 동일한 피벗 집합에 수렴하고, 잘못된 사고는 적어도 하나의 피벗을 위반한다고 가정합니다.

- **Technical Details**: 제안된 방법론은 자가 학습(self-training) 파이프라인을 활용하여 다양한 추론 경로를 샘플링하고 공유된 결정 피벗을 발굴합니다. 각 경로는 보조 검증기(auxiliary verifier)를 사용하여 피벗 중심의 단기 경로 추론으로 압축됩니다. 마지막으로 모델은 자체 생성된 출력을 사용하여 추가 학습(post-training)을 진행합니다. 이 방법은 정답 데이터나 외부 메트릭 없이도 추론을 정렬할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: LogiQA, MedQA, MATH500과 같은 표준 벤치마크를 통한 실험에서는 제안한 방법이 효과적임을 보여줍니다. 원하는 추론 경로를 낼 수 있는 능력을 통해 모형의 성능을 향상시키는 데 기여합니다. 이러한 실험 결과는 결정 피벗 개념을 적용하여 LLM의 추론 과정을 더욱 투명하게 만든다는 점에서 중요한 의미가 있습니다.



### Towards mitigating information leakage when evaluating safety monitors (https://arxiv.org/abs/2509.21344)
Comments:
          14 pages, 4 figures

- **What's New**: 이번 연구에서는 대형 언어 모델에서 잠재적인 위험 행동을 탐지하기 위한 화이트 박스 모니터(white box monitors)의 성능을 체계적으로 평가하기 위한 프레임워크를 제안합니다. 이러한 모니터는 모델의 내부를 분석할 수 있는 장점이 있으며, 응답 예시(response exemplars)를 필요로 하지만 이는 실제 성능에 영향을 미칠 수 있는 정보 유출(leakage) 문제를 동반합니다. 이 연구는 진정한 모델 동작을 탐지하는 능력을 평가하는 방법을 찾고 있습니다.

- **Technical Details**: 연구진은 세 가지 새로운 전략을 제안했습니다. 첫째, 콘텐츠 필터링(content filtering)을 통해 입력에서 기만 관련 텍스트를 제거하고, 둘째, 점수 필터링(score filtering)을 통해 작업과 관련된 토큰만 집계하며, 셋째, 기만 행동을 드러내기 위해 구체적인 프롬프트 없이 훈련된 모델 유기체(prompt distilled fine-tuned model organisms)를 사용합니다. 특히 기만 탐지를 Representative case study로 하여, 두 가지 형태의 유출을 확인했습니다: 유도 유출(elicitation leakage)과 추론 유출(reasoning leakage).

- **Performance Highlights**: 실험을 통해 여러 기만 벤치마크에서 제안된 완화 전략을 적용하고 성능 유지(performance retention)를 측정했습니다. 연구 결과, 첫째, 콘텐츠 필터링은 30%까지 AUROC(Area Under the Receiver Operating Characteristic curve)를 감소시키는 효과적 전략으로 나타났습니다. 둘째, 점수 필터링은 15%의 AUROC 감소를 보여주었지만 그 기여도를 명확히 알기 어렵습니다. 셋째, 훈련된 모델 유기체는 모니터 평가를 개선하지만, 다시 훈련하더라도 성능은 최대 40% 감소하는 결과를 보였습니다.



### See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation (https://arxiv.org/abs/2509.22653)
Comments:
          CoRL 2025. Project page: this https URL

- **What's New**: 본 연구에서는 See, Point, Fly (SPF)라는 새로운 훈련 없는 항공 비전-언어 내비게이션(AVLN) 프레임워크를 소개합니다. SPF는 비전-언어 모델(VLM)을 기반으로 하여 어떤 형태의 자유 형식 지시에도 강력하게 반응할 수 있도록 설계되었습니다. 특히, 기존의 VLM 기반 접근 방식들이 행동 예측을 텍스트 생성 작업으로 다룬 것과 달리, 우리는 AVLN을 2D 공간 기초 작업으로 간주하는 중요한 통찰력을 제공합니다.

- **Technical Details**: SPF는 모호한 언어 지시를 2D 웨이포인트(waypoints)로 변환하여 항공 유무인기(UAV)의 행동 명령으로 사용합니다. 이 과정에서 카메라 정보를 활용해 2D 웨이포인트를 3D displacement 벡터로 전환하게 됩니다. 또한, SPF는 주행 거리를 적응적으로 조정하여 내비게이션 효율성을 높이며, 폐쇄 루프(control) 방식으로 동적인 환경에서도 유동적인 목표를 추적할 수 있습니다.

- **Performance Highlights**: SPF는 DRL 시뮬레이션 벤치마크에서 이전 최고의 방법에 비해 63%의 절대적 개선을 보이며 새로운 최첨단 성과를 세웠습니다. 실제 환경 평가에서도 SPF는 강력한 기준선 모델들보다 큰 폭으로 성능이 우수했습니다. 또한, 다양한 VLM에 대한 일반화 능력도 뛰어난 것을 입증하였습니다.



### VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing (https://arxiv.org/abs/2509.22651)
- **What's New**: 음성 중심의 AI 보조 도구에 대한 관심이 높아짐에 따라 VoiceAssistant-Eval이라는 포괄적인 벤치마크가 도입되었습니다. 이 벤치마크는 청취(listening), 발화(speaking), 시청(viewing) 능력을 평가하기 위해 13개 작업 카테고리에서 10,497개의 정제된 예시로 구성되어 있습니다. 다양한 작업에는 자연 소리, 음악 및 대화가 포함됩니다.

- **Technical Details**: VoiceAssistant-Eval은 개인화된 음성 모방, 자연스러운 핸즈프리 음성 상호작용, 다중 모드 비전-오디오 이해 및 고급 오디오 QA와 같은 4가지 대표 작업을 평가합니다. 비교를 통해 기존의 벤치마크는 특정 측면만을 커버하고 있는 반면, VoiceAssistant-Eval은 AI 보조 도구의 전체 범위를 종합적으로 테스트합니다. 또한, 역할별 말하기 스타일과 음색을 분석하여 개인화 상호작용의 잠재력을 입증합니다.

- **Performance Highlights**: 21개의 오픈 소스 모델과 GPT-4o-Audio 모델을 평가한 결과, 독점 모델이 항상 오픈 소스 모델보다 우수하지 않음을 보여주었습니다. 대부분의 모델이 발화 작업에서 뛰어난 성능을 보였으나 오디오 이해에서는 뒤처지는 경향을 나타냈습니다. 특히, 중간 크기의 Step-Audio-2-mini(7B)가 LLaMA-Omni2-32B-Bilingual보다 두 배 이상의 청취 정확도를 기록한 것이 주목할 만한 성과입니다.



### Toward a Physics of Deep Learning and Brains (https://arxiv.org/abs/2509.22649)
- **What's New**: 본 논문은 딥 뉴럴 네트워크(deep neural networks)와 생물학적 뇌(brains) 간의 유사성에 대해 논의하며, 두 시스템에 공통적인 이론적 틀을 제시합니다. 제시된 방정식은 살아있는 뇌에서의 신경쇄도(neuronal avalanches)를 설명하는 데 사용되며, 딥 뉴럴 네트워크의 활동 전파(cascades)에도 적용될 수 있음을 보여줍니다. 이 연구는 또한 비평형 통계 물리(non-equilibrium statistical physics)에 기반하여 네트워크의 학습 최적화 조건을 규명합니다.

- **Technical Details**: 저자들은 신경망의 가중치(weight)와 바이어스(bias)가 독립적으로 작용함을 명확히 하며, 신호 에너지를 구하는 수식을 도출합니다. 이론적 프레임워크 안에서, 깊은 네트워크는 흡수(absorbing)와 활동(active) 단계 사이에서 최적의 학습을 수행하나, 진정한 임계점(critical point)이 아닌 준 임계점(quasi-critical regime) 내에서 작동함을 설명합니다. 이는 기본적인 물리적 개념을 통합하여, 네트워크의 성능 향상을 위한 설계 청사진을 제공합니다.

- **Performance Highlights**: 최대 감수성(maximal susceptibility)이 네트워크 학습을 예측하는 더욱 신뢰할 수 있는 지표라는 것을 보여줍니다. 또한, 실험을 통해 바르카우젠 소음(Barkhausen noise)과 유도 침투(directed percolation) 등 다양한 보편성(class)에서 보이는 특성을 확인했습니다. 이러한 연구 결과는 생물학적 뉴럴 네트워크와 인공적인 뉴럴 네트워크 간의 보편적인 특징을 공유하고 있음을 강조합니다.



### CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning (https://arxiv.org/abs/2509.22647)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문은 이미지 캡셔닝 과제를 위한 새로운 Reinforcement Learning with Verifiable Rewards (RLVR) 패러다임을 제안합니다. 기존의 Supervised Fine-Tuning (SFT) 모델들이 특정 Ground-Truth 답변을 기억하는 것에 한계를 두고, 일반성 부족 문제를 해결할 수 있는 방법을 갖추고 있습니다. 새로운 Captioning Reinforcement Learning (CapRL) 프레임워크는 이미지에 대해 질문을 정확히 답할 수 있는 캡션 생성을 평가합니다.

- **Technical Details**: CapRL은 LVLM이 생성한 캡션과 비슷한 정보에 대한 Multiple-Choice Questions (MCQs)에 기반하여 객관적인 보상을 제공하는 이중 파이프라인을 사용합니다. 세부적으로, LVLM이 캡션을 생성하고, 비전이 없는 LLM이 그 캡션을 통해 질문에 답하는 정확성을 통해 보상을 결정합니다. 이 접근법은 주관적인 이미지 캡셔닝 문제에서 캡션의 품질을 유용성(utility)으로 정의합니다.

- **Performance Highlights**: CapRL을 적용한 결과, 12개의 벤치마크에서 상당한 성능 향상을 보여 주며, CapRL-3B가 72B 모델에 필적하는 성과를 달성했습니다. Prism Framework를 통한 평가에서 CapRL 모델이 기본선보다 평균 8.4% 향상된 성능을 나타냈습니다. 이러한 결과는 CapRL이 Dense하고 Accurate한 캡션을 생성하도록 모델을 효과적으로 유도함을 보여줍니다.



### Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs (https://arxiv.org/abs/2509.22646)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 AI 생성 비디오에서 확인할 수 있는 인간 지각의 '딥페이크 흔적'을 탐구하는 DeeptraceReward라는 새로운 벤치마크를 소개합니다. 이 데이터셋은 4,300개의 세부 주석을 포함하며, 각 주석은 자연어 설명, 경계 박스(bounding box) 위치 및 정확한 시간 정보를 제공합니다. 이를 통해 인간이 어떻게 AI 생성 비디오를 식별할 수 있는지와 그 이유를 조사하는 것을 목표로 하고 있습니다.

- **Technical Details**: DeeptraceReward 데이터셋은 3,300개의 고품질 생성 비디오와 3,300개의 실제 비디오로 구성됩니다. 저자들은 비디오에 존재하는 다양한 딥페이크 흔적을 아홉 가지 범주로 분류했으며, 이를 통해 멀티모달 언어 모델을 훈련시켜 인간의 판단을 모방할 수 있도록 하였습니다. 연구 결과, 7B의 보상 모델이 GPT-5를 34.7% 초과하여 인식 및 설명 과제에서 우수한 성능을 보였다는 것이 흥미롭습니다.

- **Performance Highlights**: 모델의 성능 평가에서 이진 분류(진짜 vs 가짜 비디오) 작업은 99.4%에 달하지만, 보다 세밀한 딥페이크 흔적 탐지 성능은 70.2%에 불과함을 보였습니다. 저자들은 자연어 설명 및 공간적 지각은 상대적으로 쉬운 반면, 시간 라벨링 작업은 가장 어려운 과제임을 확인했습니다. 이러한 결과는 영상 생성의 사회적 신뢰성을 향상시키기 위한 중요한 지침을 제공합니다.



### Hierarchical Representation Matching for CLIP-based Class-Incremental Learning (https://arxiv.org/abs/2509.22645)
- **What's New**: 본 논문에서는 HiErarchical Representation MAtchiNg (HERMAN)을 제안하여 CLIP 기반의 클래스 증가 학습(Class-Incremental Learning, CIL) 문제를 해결합니다. HERMAN은 LLMs(대형 언어 모델)를 활용하여 계층적인 텍스트 설명을 생성하고, 이를 시각적 표현과 매칭하는 과정을 통해 새로운 범주에 대한 적응을 향상시킵니다. 새로운 접근방식인 HERMAN은 기존의 단순한 템플릿을 넘어서 정보의 계층적 구조를 효과적으로 활용하도록 설계되어 있습니다.

- **Technical Details**: HERMAN 프레임워크는 먼저 LLMs를 통해 여러 계층에서 구별되는 텍스트 설명을 생성합니다. 그런 다음 이러한 설명은 해당하는 시각적 특징과 매칭되어, 형상과 세부 정보를 모두 포함하는 구조화된 의미 공간을 생성합니다. 이후 적응형 라우팅 메커니즘을 통해 의미 계층 전반에 걸쳐 가중치를 동적으로 할당하며, 각 입력에 대해 가장 관련성 높은 특징을 강조할 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험 결과, HERMAN은 기존의 최첨단 성능을 지속적으로 초과하는 능력을 보여주었습니다. 특히 계층적 의미를 통한 정밀한 구별을 가능하게 하여, 재학습 과정에서의 재앙적 망각(catasrophic forgetting)을 완화하는 데 기여하고 있습니다. 이러한 성과는 CLIP 모델의 강력한 성능 향상에 기여함은 물론, 실세계 incremental learning 문제를 해결하는 데 중요한 통찰을 제공합니다.



### WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning (https://arxiv.org/abs/2509.22644)
- **What's New**: 최근의 연구들은 코드 에이전트가 GitHub 이슈 수정 및 새로운 기능 구현과 같은 코드 생성 작업에서 큰 발전을 보여주었음을 강조하고 있습니다. 그러나 웹사이트 코드 생성과 같은 작업에서는 시각적 미학 및 사용자 상호작용의 유창성에 의존하게 되어 현재의 코드 에이전트 시스템이 생성된 코드베이스의 실제 품질을 완전히 포착하지 못하는 문제가 발생하고 있습니다. 이 논문에서는 통합적이고 다단계 시각적 피드백을 활용하여 웹사이트 코드를 반복적으로 생성하고 개선하는 웹사이트 생성 에이전트인 WebGen-Agent를 제안합니다.

- **Technical Details**: WebGen-Agent는 자연어 지침을 기반으로 웹사이트를 생성하며 코드 생성, 코드 실행, 피드백 수집의 세 가지 동작을 포함하는 반복적이고 다단계의 패러다임을 채택하고 있습니다. 코드 실행 후 생성된 웹사이트의 스크린샷을 캡처하고, 흐름 언어 모델(VLM)을 사용하여 시각적 매력과 품질을 평가합니다. 이 피드백을 통해 웹사이트의 디자인과 인터랙티브 기능을 지속적으로 개선할 수 있습니다.

- **Performance Highlights**: WebGen-Agent는 WebGen-Bench 데이터셋에서 Claude-3.5-Sonnet의 정확도를 26.4%에서 51.9%로 증가시키고 외관 점수를 3.0에서 3.9로 올리며 이전의 최고 성능을 초과했습니다. Step-GRPO 훈련 방법은 Qwen2.5-Coder-7B-Instruct의 정확도를 38.9%에서 45.4%로 개선하고 외관 점수를 3.4에서 3.7로 증가시켜 생성된 웹사이트의 기능성과 외관을 모두 크게 향상시킵니다.



### Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity (https://arxiv.org/abs/2509.22641)
Comments:
          26 pages, 10 figures, under review

- **What's New**: 이 연구는 언어 모델의 텍스트 생성 능력을 평가하기 위해 n-그램(n-gram) 신규성(로비티, novelty)을 사용하는 데 한계를 지적합니다. 창의성의 두 가지 성격인 신규성과 적합성(pragmaticality)을 함께 고려해야 한다고 주장하며, 이를 통해 n-그램 신규성과 창의성의 관계를 분석합니다. 7542개의 전문가 작가 주석을 통해 n-그램 신규성이 있는 텍스트의 대부분이 창의적이지 않다고 판단된다는 중요한 발견을 보고합니다.

- **Technical Details**: 연구에서는 창의성을 인간 평가에 기반하여 운영화하고, 적합성을 의미론적(sensicality)과 맥락적(pragmaticality)으로 분해합니다. LLM(대형 언어 모델)에서 생성된 텍스트는 지나치게 혁신적(novel)일 수 있으나 그 맥락에서 이해되지 않으면 창의적으로 평가받지 못할 수 있습니다. 또한, 전문가의 평가를 통해 n-그램 신규성이 높은 텍스트의 91%가 창의적이지 않다고 판별되었음을 밝혔다.

- **Performance Highlights**: 최신 LLM들이 창의적인 표현을 생성하는 데 어려움을 겪고 있으며, 이는 LLM을 이용한 평가보다 전문가의 판별이 더욱 적합하다는 것을 의미합니다. 연구 결과는 LLM으로 생성된 텍스트의 창의성 평가에서 상대적으로 높은 성과를 보여주지만 비적합적인 표현(non-pragmatic expressions)을 식별하는 데에는 한계가 있음을 드러냅니다. 가장 성과가 좋았던 모델의 신규성 점수는 전문가의 선호도와 강한 상관관계를 보였습니다.



### Language Models Can Learn from Verbal Feedback Without Scalar Rewards (https://arxiv.org/abs/2509.22638)
- **What's New**: 이 논문은 LLMs(대형 언어 모델) 훈련에서 일반적으로 사용되는 인간 또는 AI 피드백으로부터의 강화 학습(RL) 방식을 재정의합니다. 구체적으로, 언어 피드백을 조건부 신호로 간주하여, 피드백 조건부 정책(FCP)을 도입합니다. FCP는 응답-피드백 쌍으로부터 직접 학습하고, 최적화된 피드백을 통해 모델을 개선할 수 있는 구조를 제공합니다.

- **Technical Details**: FCP는 πθ(𝒐|𝒙,𝒄)로 표현되며, 주어진 지침 𝒙에 대해 응답 𝒐를 생성하는 참조 정책 πref(𝒐|𝒙)와 환경 피드백 분포 penv(𝒄|𝒙,𝒐)을 결합합니다. 이 정책은 긍정적인 피드백 조건에서 훈련되어 피드백과 응답 간의 관계를 강화합니다. FCP는 또한 온라인 부트스트래핑 단계를 통해 지속적으로 피드백을 업데이트하며 모델 성능을 향상시킵니다.

- **Performance Highlights**: 파일럿 실험 결과, FCP는 오프라인 RFT 및 온라인 GRPO와 같은 기존의 강력한 스칼라 기반 기준선을 초과하는 성능을 보였습니다. 이 방법은 검증기나 스칼라 변환 없이도 풍부한 언어 피드백을 보존할 수 있는 단순하고 확장 가능한 프레임워크를 제공합니다. 향후 발전된 훈련 기술을 적용하면 FCP의 성능을 더욱 개선할 수 있을 것으로 기대됩니다.



### Variational Reasoning for Language Models (https://arxiv.org/abs/2509.22637)
- **What's New**: 이번 연구는 언어 모델을 위한 변별적 사고 프레임워크를 소개하며, 사고 흔적(thinking traces)을 잠재 변수(latent variables)로 취급하여 변별 추론(variational inference)을 통해 최적화하는 방법을 제안합니다. 기존의 ELBO(evidence lower bound)를 다중 흔적(multi-trace) 목표로 확장하고, 훈련을 안정화하는 forward-KL 형태를 소개합니다. 이 방법론은 기존의 지도 학습(Supervised Finetuning) 및 강화 학습(Reinforcement Learning) 방법론과 비교했을 때, 사전 훈련된 모델의 사회적 편향(bias)를 더 명확히 이해할 수 있도록 돕습니다.

- **Technical Details**: 변별적 사고 프레임워크에서는 사고 흔적을 생성하고 최종 답변을 도출하기 위해 모델 πθ(𝒛,𝒚|𝒙)를 사용합니다. 이는 사고 과정(𝒛)과 응답(𝒚)으로 구성되며 연합확률(joint probability)은 두 조건부 확률로 나눌 수 있습니다. 본 논문에서는 ELBO를 통해 사고 흔적에 대한 마진 분포(marginal distribution)를 최적화하는 비용 함수를 제안하고, IWAE 스타일의 다중 흔적(extension)을 통해 목표를 강화하여 훈련 안정성을 높입니다.

- **Performance Highlights**: 제안된 프레임워크는 Qwen2.5와 Qwen3 모델 가족을 통해 다양한 추론 벤치마크에서 강력한 기준선을 넘어서는 일관된 개선 효과를 보였습니다. 연구는 MATH500, AIME24&25, OlympiadBench 등 여러 이유 문제에서 성능을 검증하였으며, 변별적 사고 접근방식이 기존 방법들과 비교해 확실한 이점을 제공함을 입증하였습니다. 또한, 중도 폐쇄(drop-out)를 방지하며 더 나은 답변 힌트를 활용하도록 훈련 파이프라인을 조정하는 방식으로 시스템의 전반적인 추론 역량을 향상시킵니다.



### Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback (https://arxiv.org/abs/2509.22633)
- **What's New**: 본 논문에서는 온라인 강화 학습과 인간 피드백(RLHF)에 대한 탐색 원칙을 다룹니다. 인간의 선호 데이터를 사용하여 보상 모델을 적응적으로 수집하고 정책을 개선하는 새로운 탐색 스킴을 제안합니다. 기존 탐색 알고리즘의 단점을 분석하여, 비효율적인 비교 방식으로 인해 정보 불확실성을 줄이는 데 실패하는 지점을 지적합니다.

- **Technical Details**: RLHF의 모델 설정은 사용자가 제공하는 모든 가능한 입력이나 쿼리 집합인 프롬프트 공간 𝒳과 주어진 프롬프트에 대한 가능한 모든 출력 세트인 응답 공간 𝒜로 구성됩니다. 저자는 인간 선호 데이터를 최대 우도 추정(MLE)을 통해 학습하는 보상 모델을 정의하고, 이를 바탕으로 보상 극대화와 원래 모델의 유사성을 유지하는 정책을 미세 조정합니다. KL 정규화 보상 목표를 설정하여 보상 함수의 변화를 관리합니다.

- **Performance Highlights**: 제안된 탐색 스킴은 복잡한 동작 쌍 중에서 정책 개선에 가장 중요한 정보 불확실성을 줄이는 방향으로 선호 쿼리를 유도합니다. 이 시스템 아래에서 저자는 RLHF의 후회 경계를 $T^{(eta+1)/(eta+2)}$로 확립하며, 이는 모델 파라미터에 대해 다항적으로 스케일링됩니다. 이 알고리즘은 모든 모델 파라미터에 대해 처음으로 다항적 후회 스케일링을 제공합니다.



### StateX: Enhancing RNN Recall via Post-training State Expansion (https://arxiv.org/abs/2509.22630)
- **What's New**: 이 논문에서는 RNN의 상태 크기를 포스트 트레이닝(post-training)을 통해 효율적으로 확장하는 StateX 훈련 파이프라인을 소개합니다. 이는 학습 비용을 최소화하고 새로운 매개변수 추가를 거의 없이 이루어질 수 있습니다. 또한, 긴 문맥을 처리하는 모델의 경우 더 큰 재귀적 상태가 중요하다는 점을 반영하여 상태 확장을 사전적으로 수행합니다.

- **Technical Details**: StateX는 선형 주의(linear attention)와 상태 공간 모델(state space models) 두 가지 인기 있는 RNN 클래스에 대해 상태를 확장하기 위한 아키텍처 수정을 설계했습니다. 포스트 트레이닝에 필요한 데이터 양이 기존의 프리 트레이닝(pre-training)보다 적고, 모델 성능과 적응 효율성을 균형 있게 유지하기 위해 핵심 레이어만 선택하여 확장합니다. 기존의 복잡한 방법들과 비교하여 더 간단하고 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: StateX를 통해 RNN의 기억 및 문맥 학습 능력이 크게 향상되었음을 실험 결과로 입증했습니다. GLA 모델에서 기억 중심 작업의 상대적인 정확도가 3.36%, 문맥 학습에서는 7.2% 증가하였으며, Mamba2 모델에서도 각각 1.1%와 1.0% 향상되었습니다. 평균 NIAH 정확도는 GLA에서 26.0%에서 42.2%, Mamba2에서 33.2%에서 39.2%로 개선되었습니다.



### Learning Admissible Heuristics for A*: Theory and Practic (https://arxiv.org/abs/2509.22626)
- **What's New**: 이 논문은 A* 알고리즘과 같은 탐색 알고리즘의 성능에 대한 휴리스틱 함수의 중요성을 강조합니다. 최근의 딥러닝 접근 방식은 적합성을 종종 무시하고 있으며, 훈련 데이터 이상의 일반화에 대한 보장이 제한적입니다. 이에 본 연구에서는 휴리스틱 학습을 제약 최적화 문제로 간주하고 Cross-Entropy Admissibility (CEA)라는 손실 함수를 도입하여 훈련 중 적합성을 보장합니다.

- **Technical Details**: 휴리스틱 함수는 특정 문제에 대한 도메인 지식을 기반으로 계산되며, 예를 들어 패턴 데이터베이스 (PDBs)와 같은 방식이 있습니다. 본 논문에서는 PDB 추상화를 활용하고 그래프의 구조적 속성을 통해 A* 알고리즘이 일반화되는 데 필요한 훈련 샘플의 수를 정확하게 계산했습니다. 실질적으로, 일반적인 가설 클래스 대신 ReLU 신경망을 사용함으로써 그래프 크기가 아닌 신경망의 너비와 깊이에 따라 경계를 정의했습니다.

- **Performance Highlights**: 3x3 루빅스 큐브 패턴 데이터베이스를 대상으로 한 평가에서, CEA 손실 함수는 우수한 성능을 보여주었고, 학습된 휴리스틱은 상당히 강력한 가이드를 제공하여 기존의 압축 패턴 데이터베이스 휴리스틱보다 훨씬 우수한 결과를 냈습니다. 특히, CEA 손실을 사용한 8코너 PDB에 대한 결과에서는 완벽한 적합성을 학습한 것으로 보고되었습니다.



### A Theoretical Analysis of Discrete Flow Matching Generative Models (https://arxiv.org/abs/2509.22623)
- **What's New**: 이번 연구에서는 Discrete Flow Matching (DFM) 생성 모델의 이론적 분석을 제공합니다. DFM은 변환 속도 필드를 근사하는 신경망을 훈련하여 생성 동력을 학습하는 유망한 이산 생성 모델링 프레임워크입니다. 이 분석은 최종 분포 추정 오차를 분해함으로써 명확한 보장 체계를 수립합니다.

- **Technical Details**: 연구에서는 Transformer 아키텍처의 근사 오차를 분석합니다. 이 과정에서 두 가지 주요 소스인 Approximation Error와 Estimation Error를 통해 학습된 속도 필드의 위험을 제한하고, 각 소스의 통계적 수렴률을 도출합니다. DFM 모델이 훈련 세트 크기가 증가할수록 진짜 데이터 분포에 수학적으로 수렴함을 처음으로 공식적으로 증명합니다.

- **Performance Highlights**: DFM 모델은 다양한 생성 작업에 대한 효율성과 적용 가능성을 탐색하는 데 있어 급격한 확산을 일으키고 있습니다. 예를 들어 Hu와 Ommer(2024)는 이미지 도메인에서 DFM의 효율성을 검증하였고, Qin et al.(2024)는 DeFoG 프레임워크를 소개하여 그래프의 고유한 대칭을 존중하는 최적화를 달성했습니다. 이처럼 DFM의 성공은 경험적 검증에 의해 고무적으로 나타나고 있으며, 이 연구는 DFM의 이론적 기초를 제공함으로써 이 중요한 격차를 채웁니다.



### IA2: Alignment with ICL Activations Improves Supervised Fine-Tuning (https://arxiv.org/abs/2509.22621)
- **What's New**: 본 논문에서는 ICL(인맥 학습)와 SFT(지도 세부 조정)의 내부 계산을 통해 SFT의 품질을 향상시킬 수 있는지를 탐구하며, 이를 ICL Activation Alignment(IA2)라는 자기 증류(self-distillation) 기법을 통해 제안합니다. IA2는 ICL의 활성화 패턴을 SFT 모델에 복제하려고 하며, ICL처럼 내부적으로 사고할 수 있도록 유도합니다.

- **Technical Details**: IA2는 (1) 정보가 풍부한 ICL 활성화를 수집하고, (2) ICL과 기능적 정렬을 강화한 후, (3) 이 준비된 모델에서 SFT를 수행하는 단계로 구성됩니다. 이는 SFT 모델의 성능을 극적으로 향상시키며, 우리가 사용한 12개의 벤치마크에서 13,000개 이상의 모델을 훈련하여 그 결과를 검증합니다.

- **Performance Highlights**: IA2는 SFT 전의 priming 단계로서, 모델의 출력 정확도와 보정을 개선해 주며, ICL만의 중요한 훈련 신호를 제공합니다. IA2의 적용은 모델 적응의 내부 메커니즘을 이해하는 데도 중요한 통찰력을 제공하며, SFT만으로는 얻을 수 없는 중요한 데이터로 모델 성능 향상에 기여합니다.



### Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting (https://arxiv.org/abs/2509.22615)
- **What's New**: 이 논문에서는 2D Gaussian Splatting (2DGS)을 새로운 비주얼 표현으로 탐구하여 멀티모달 시스템에서 비전-언어 정렬을 위한 효율적이고 효과적인 중간 표현으로 사용할 수 있는지에 대한 논의를 담고 있습니다. 기존 RGB 기반 비전 인코더의 데이터 전송 및 처리 효율성 문제를 해결하기 위해, 2DGS는 색이 지정된 비대칭 Gaussian의 집합으로 이미지를 매개변수화하여 더 간결하고 공간적으로 적응 가능한 형태로 정보를 전달합니다. 이 연구는 대규모 범위에서 2DGS를 구현하고 CLIP 프레임워크에 적응시키는 방법을 제안하며, 빠른 피팅과 GPU 유틸리제를 달성했습니다.

- **Technical Details**: 논문에서 제안한 시스템 및 알고리즘 최적화는 구조화된 초기화, 밝기 인식 L1 프루닝 및 배치 CUDA 커널을 포함하여 기존의 구현보다 90배 이상 빠른 피팅을 달성하도록 설계되었습니다. 2DGS 표현은 효율적인 데이터 전송을 가능하게 하고, RGB 기반의 변환기 아키텍처의 냉동된 형태를 재사용하여 가벼운 입력 전처리 유닛과 함께 CLIP 프레임워크에 효과적으로 통합됩니다. 이 과정에서 약 7%의 매개변수로 훈련을 수행하면서도 우수한 전이 학습 성과를 나타냅니다.

- **Performance Highlights**: 실험 결과 12.8M 규모의 DataComp 데이터셋에서 GS 인코더는 이미지넷-1K의 제로샷 성능을 의미 있게 달성하면서 픽셀 대비 입력 압축은 3배에서 20배 이르는 결과를 보여주었습니다. 현재 2DGS 기반 인코더는 RGB 기반 모델에 비해 정확성이 낮지만, 명백한 전이 가능성과 압축된 표현으로서의 유용성을 입증하였습니다. 이 연구는 2DGS가 멀티모달 시스템에서 효과적인 대안이 될 수 있으며, 이는 처리 효율성과 지속 가능성을 높이는 방향으로 나아가고자 하는 방법을 제시합니다.



### Quantile Advantage Estimation for Entropy-Safe Reasoning (https://arxiv.org/abs/2509.22611)
- **What's New**: 본 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR)을 통해 LLM(대형 언어 모델)의 추론을 강화하지만, 훈련 과정에서 발생하는 두 가지 문제인 {entropy collapse}와 {entropy explosion} 사이의 진동을 확인합니다. 이 두 가지 문제는 가치 없는 RL에서 사용되는 평균 기준선의 부적절한 적용에서 기인한다고 제안하고, {Quantile Advantage Estimation} (QAE) 방법을 통해 평균 대신 K-quantile 기준선을 사용하여 해결하고자 합니다. 이를 통해 훈련의 안정성을 높이고 성과 향상을 도모합니다.

- **Technical Details**: QAE는 응답 수준의 두 가지 레짐을 도입하여, 어려운 질문(p <= 1 - K)에서는 드문 성공을 강화하고, 쉬운 질문(p > 1 - K)에서는 남은 실패를 겨냥합니다. 본 연구에서는 첫 번째 차수 softmax 업데이트 하에 {two-sided entropy safety}를 증명하였으며, 이를 통해 폭발 및 붕괴를 방지하는 일 단계의 엔트로피 변화에 대한 하한 및 상한을 설정합니다. QAE는 엔트로피 규제 문제를 토큰 수준 튜닝 문제로부터 기준 설계 문제로 재구성하는 접근법을 보여줍니다.

- **Performance Highlights**: 실험에 따르면, QAE는 엔트로피를 안정화하고 신뢰도 할당을 희소화하며, 조정된 K로 약 80%의 응답이 제로 이점을 받도록 하여 효율적인 학습이 가능하게 합니다. 이러한 최소한의 수정이 Qwen3-8B/14B-Base에서의 pass@1 성능을 지속적으로 증가시키는 것을 확인했습니다. QAE는 Qwen3-30B-A3B-Base 모델과 더불어 다른 RLVR 메소드와 잘 결합되어, AIME 2024, AIME 2025 및 AMC 2023에서 강력한 성능을 나타냈습니다.



### Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning (https://arxiv.org/abs/2509.22601)
Comments:
          26 pages, 11 figures

- **What's New**: 본 논문에서는 자기 모방 학습(Self-Imitation Learning, SIL)을 기반으로 한 새로운 강화 학습(RL) 접근 방식인 SPEAR(Self-imitation with Progressive Exploration for Agentic Reinforcement learning)를 제안합니다. 이 방법은 정책 엔트ropy를 동적으로 조절하여 탐색(exploration)과 활용(exploitation)의 균형을 유지합니다. 특히, SPEAR는 툴 사용 능력을 개발하기 위한 도움 보상을 포함하여, 훈련 초기 단계에서의 엔트로피 증가가 탐색을 촉진함을 강조합니다.

- **Technical Details**: 논문에서 제안하는 SPEAR는 스킬 기반 탐색을 지원하기 위한 인트린식 보상을 이용하며, 후속 단계에서는 기존의 성공적인 패턴을 활용하여 행동 수준의 탐색을 촉진합니다. 또한, Replay Buffer의 경험에 대한 이점 재조정과 높은 공분산의 클리핑 기법을 통해 정책 업데이트의 안정성을 강화하고 보상 해킹(reward hacking) 문제를 완화합니다. 이는 정책 엔트로피를 조절하여 훈련의 불안정성을 방지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: SPEAR는 GRPO, GiGPO, Dr.BoT와 같은 여러 기준 방법들에 비해 상당한 성능 향상을 보여주며, ALFWorld 및 WebShop 태스크에서 각각 최대 16.1%, 20.7%의 성능 개선을 기록했습니다. 또한, Dr.BoT의 성능을 각각 AIME24에서 3.8%, AIME25에서 6.1% 향상시킵니다. SPEAR는 낮은 계산 복잡도를 유지하며 다른 LLM 기반 에이전트들에 대해 뛰어난 호환성과 확장성을 보여주는 플러그 앤 플레이 알고리즘입니다.



### From Parameters to Behavior: Unsupervised Compression of the Policy Spac (https://arxiv.org/abs/2509.22566)
- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL)의 비효율성을 해결하기 위한 새로운 관점을 제시합니다. 기존의 고차원 매개변수 공간 $	heta$에서 직접 정책을 최적화하는 방식 대신, 저차원 잠재 공간 $	ext{Z}$로 압축하여 다루려는 접근법을 택합니다. 이는 여러 작업을 동시에 수행할 때 더욱 중요해지며, 주어진 특정 작업에 대한 적응력을 높입니다.

- **Technical Details**: 제안된 방법은 비지도 학습을 기반으로 하여 잠재 행동 공간을 학습하는 두 단계의 프레임워크를 포함합니다. 첫 번째 단계에서는 생성 모델을 이용하여 행동 매니폴드의 잠재 표현을 학습하고, 두 번째 단계에서는 이 표현을 활용하여 특정 작업을 효율적으로 해결합니다. 이 방식은 고차원 매개변수 공간이 아닌 잠재 행동 공간을 탐색하는 것을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 낮은 차원의 잠재 표현을 학습할 수 있으며, 이는 환경의 복잡도에 의해 더 영향을 받는다는 것을 보여줍니다. 또한, 간단한 알고리즘이 기존의 복잡한 DRL 알고리즘과 경쟁할 수 있는 성능을 발휘하는 것을 확인했습니다. 이는 DRL의 샘플 효율성을 크게 향상시키는 결과를 가져올 것으로 기대됩니다.



### Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation (https://arxiv.org/abs/2509.22565)
- **What's New**: 이번 연구에서는 EHR(전자 건강 기록) 포털을 통한 비동기 환자-임상 의사 메시징의 증가로 인해 임상 의사의 업무 부담이 커지고 있음을 언급하며, 대형 언어 모델(LLMs)을 활용한 초안 응답 보조의 필요성을 강조합니다. 연구는 임상적으로 기반한 오류 온톨로지를 도입하고, 검색 보강 평가 파이프라인(RAEC)을 개발하며, DSPy를 사용한 두 단계의 프롬프트 구조를 제공하는 세 가지 주요 기여로 구성됩니다.

- **Technical Details**: 연구에서 소개된 오류 온톨로지는 5개의 도메인과 59개의 세부 오류 코드로 구성되어 있으며, 이는 귀납적 코딩과 전문가의 판단을 통해 개발되었습니다. RAEC는 의미적으로 유사한 과거 메시지-응답 쌍을 활용하여 평가 품질을 향상시키는 방식으로 작동하며, 두 단계의 DSPy 파이프라인을 사용하여 질적인 검토를 수행합니다.

- **Performance Highlights**: 1,500개 이상의 환자 메시지를 대상으로 한 평가에서, 검색된 맥락이 임상적 완전성과 업무 적합성 등의 영역에서 오류 식별을 개선하는 데 기여한 것으로 나타났습니다. 100개의 메시지에 대한 인간 검증 결과, 맥락이 향상된 레이블의 성능이 기준선과 비교하여 더 높은 일치도(50% vs. 33%)와 성능(F1 = 0.500 vs. 0.256)을 보이며 RAEC 파이프라인의 유용성을 지지하고 있습니다.



### Activation Function Design Sustains Plasticity in Continual Learning (https://arxiv.org/abs/2509.22562)
- **What's New**: 이번 연구는 연속 학습에서 활성화 함수가 플라스틱성 손실에 미치는 영향을 탐구합니다. 변화하는 데이터 환경에서 플라스틱성을 유지하기 위해 Smooth-Leaky와 Randomized Smooth-Leaky와 같은 새로운 비선형 활성화 함수를 제안하고 평가했습니다. 또한, 활성화 함수의 모양과 적응도 간의 관계를 파악하기 위한 간단한 스트레스 프로토콜을 제공하여 더 나은 활성화 함수 설계를 강조합니다.

- **Technical Details**: 연구에서는 활성화 함수가 기울기 정보의 첫 번째 관문으로 작용하며, 그 기울기와 포화 정도가 중요한 역할을 한다고 설명합니다. 각종 활성화 함수를 비교 분석하여 이들이 플라스틱성 손실을 어떻게 가중시키거나 완화할 수 있는지를 조명합니다. 특히, Leaky-ReLU와 같은 변종이 어떻게 비활성 유닛 문제(dormant unit problem)를 감소시킬 수 있는지를 논의합니다.

- **Performance Highlights**: 두 가지 새로운 활성화 함수는 연속 학습 및 변화하는 RL 환경에서 플라스틱성을 저하 없이 개선하는 데 기여합니다. 이 연구의 결과는 활성화 설계가 연속 학습에서의 적응성을 지속시키는 강력하면서 가벼운 방법임을 보여주며, 정보 손실을 최소화할 수 있는 잠재력을 됩니다. 기존 모델보다 더 좋은 성과를 보여주는 다양한 활성화 함수 성능을 정리하여 제시하고 있습니다.



### ConQuER: Modular Architectures for Control and Bias Mitigation in IQP Quantum Generative Models (https://arxiv.org/abs/2509.22551)
- **What's New**: 이번 연구에서는 컨트롤 가능한 양자 생성 모델인 ConQuER를 제안합니다. ConQuER는 기존의 IQP 회로에 경량화된 제어 회로를 통합하여 생성 과정의 출력을 정밀하게 제어할 수 있도록 합니다. 이 접근 방식은 전체 재훈련 없이도 원하는 출력 분포를 생성할 수 있는 유연성을 제공합니다.

- **Technical Details**: ConQuER는 두 가지 주요 혁신을 기반으로 합니다. 첫째, 경량 제어 회로를 활용하여 사전 훈련된 IQP 회로를 보강함으로써, 특정 샘플 속성을 제어할 수 있는 모듈형 제어 메커니즘을 개발했습니다. 둘째, 데이터 기반 아키텍처 최적화 방법론을 통해 생성 편향(problem)에 대응하며, 주어진 설정에서 균형 잡힌 샘플 생성을 가능하게 합니다.

- **Performance Highlights**: ConQuER의 성능은 2D Ising 모델과 이진 blob 데이터셋을 통한 실험으로 검증되었습니다. 실험 결과, ConQuER는 제어 정확도와 균형 잡힌 생성 성능에서 우수한 성과를 나타내었으며, 원래 IQP 회로에 비해 오버헤드 비용이 매우 낮았습니다. 이로써 양자 컴퓨팅의 이점과 실용적 생성 모델링 간의 간극을 해소하는 데 기여합니다.



### Does AI Coaching Prepare us for Workplace Negotiations? (https://arxiv.org/abs/2509.22545)
- **What's New**: 이번 연구는 Trucey라는 AI 기반 코치의 개발과 평가를 다룬다. Trucey는 Brett의 협상 모델에 기반하여 설계되었으며, 협상에서의 심리적 장벽을 타파하고 개인의 준비성을 향상시키는 데 초점을 맞추고 있다. 실험 연구를 통해 Trucey, ChatGPT, 그리고 전통적인 협상 핸드북의 효과를 비교하여 AI가 협상 준비에 미치는 영향을 탐구하였다.

- **Technical Details**: Trucey는 인간 중심의 AI와 Industrial-Organizational (I/O) 심리학의 원리에 따라 설계되었으며, fine-tuned GPT-4.1를 사용하였다. 참가자들은 세 가지 조건에서 실험을 수행하고, 심리적 자원과 자아 효능감, 협상 준비 상태를 평가하는 설문지를 완료하였다. 특히 Trucey는 사용자 맞춤형 프롬프트와 역할 기반 시뮬레이션을 통해 협상 준비 과정을 지원한다.

- **Performance Highlights**: 실험 결과, Trucey가 두 가지 비교 조건에 비해 두려움을 유의미하게 감소시켰으나, 전통적인 핸드북은 사용 용이성과 심리적 자원 면에서 더 높은 점수를 받았다. 인터뷰 결과, 참가자들은 핸드북의 종합적이고 리뷰 가능한 내용이 자신감을 높이고 준비에 필수적이라고 언급하였다. 반면, AI 코치는 안내가 세분화되어 불안감을 유발하고, 경험에 따른 명확한 피드백이 부족하다는 평가를 받았다.



### InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models (https://arxiv.org/abs/2509.22536)
- **What's New**: 본 연구에서는 FP8 훈련을 위한 포괄적인 오픈소스 훈련 레시피를 도입하였습니다. 이 레시피는 지속적인 사전 훈련과 감독 학습의 미세 조정을 원활하게 통합하며, 수치적 정확도를 유지하면서도 계산 효율성을 극대화하는 하이브리드 그레뉼러티(Granularity) 양자화 전략을 사용합니다. 이를 통해 FP8 훈련이 기존의 BF16 기준 성능에 상응하면서도 훈련 시간을 최대 22% 단축하고 메모리 사용을 14% 줄이는 등 현저한 효율 개선을 이루었습니다.

- **Technical Details**: FP8 훈련은 NVIDIA의 Hoppe 아키텍처에 의해 지원되며, 슬롯당 블록 기반 양자화와 높은 정확도를 위한 토큰 기반 양자화가 결합된 하이브리드 방식으로 수행됩니다. 이 방식은 모델의 가중치에 블록 기반 양자화를 적용하고, 보다 동적 범위를 보이는 활성화에 대해서는 고정밀의 양자화를 사용하여 훈련 중 안정성을 유지합니다. 결과적으로 FP8 훈련은 BF16과 유사한 성능을 달성하면서도 더 빠르고 효율적인 학습이 가능하다는 것을 보여주었습니다.

- **Performance Highlights**: FP8 훈련의 결과, 총 훈련 시간이 최대 22% 단축되고, 피크 메모리 소모는 14% 절감되었습니다. 또한, 계산 처리량은 최대 19% 증가하였으며, 훈련과 검증 손실 곡선은 BF16의 거의 동일한 양상을 보이며 상당한 안정성을 나타냈습니다. FP8을 통해 대규모 모델 훈련의 접근성과 지속 가능성을 높이는데 기여할 수 있는 강력한 대안으로 자리잡았습니다.



### Mental Health Impacts of AI Companions: Triangulating Social Media Quasi-Experiments, User Perspectives, and Relational Theory (https://arxiv.org/abs/2509.22505)
- **What's New**: 최근 AI 기반 친구 채팅봇(AICC)인 Replika의 사용이 급증하고 있으며, 이러한 채팅봇은 공감적인 상호작용을 제공하나, 그 심리사회적 영향은 불분명한 상태입니다. 본 연구는 구글 소셜 미디어 데이터를 활용한 대규모 준 실험적 연구 및 심층 인터뷰를 통해 AICC와의 상호작용이 사용자 웰빙에 미치는 영향을 조사하였습니다. 결과는 정서적 표현의 증가와 외로움, 자살 사상에 대한 언급 증가가 혼재된 모습을 보였습니다.

- **Technical Details**: 연구는 15명의 AICC 사용자와의 심층 반구조화 인터뷰를 통해 사용자 경험의 복잡성을 분석하였으며, Knapp의 관계 발달 모델을 활용하여 AICC와의 관계 맥락을 해석했습니다. 준 실험적 분석에서는 Reddit 커뮤니티에서 AICC 사용 전후의 대화 패턴을 관찰하였고, 위계적 성향 점수 매칭과 Difference-in-Differences 회귀 기법을 적용하여 치료 효과를 추정했습니다.

- **Performance Highlights**: 결과적으로, AICC와의 상호작용은 사용자들이 정서적 검증을 받고 사회적 연습을 하는 긍정적인 요소를 제공하면서, 과도한 의존과 Withdrawal의 위험도 존재한다는 점을 밝혔습니다. 이러한 연구 결과는 AI 친구 채팅봇 설계에서 건강한 경계를 지원하고, 의존 없이 공개성을 촉진하며, 관계 단계를 드러내는 방향으로의 시사점을 제공합니다.



### Ontological foundations for contrastive explanatory narration of robot plans (https://arxiv.org/abs/2509.22493)
Comments:
          This version was submitted to the journal Information Sciences and is under review since October 2024

- **What's New**: 이번 연구는 인공지능 로봇이 다른 계획(plan) 간의 차이점을 모델링하고 설명할 수 있는 새로운 방법론을 제안합니다. 특히, 두 개의 경쟁 계획을 비교하여 각각의 특성과 인간의 선호에 가장 적합한 계획을 식별하는 데 중점을 둡니다. 본 논문에서는 이러한 비교를 통해 획득한 지식을 바탕으로 로봇이 어떻게 구체적으로 설명(narrate)할 수 있는지를 다루고 있습니다.

- **Technical Details**: 연구진은 새로운 온톨로지 모델(ontological model)을 제안하여 경쟁 계획 간의 차이를 정식화하고 이의 분석을 통해 결정론적인 관계를 도출합니다. 장소적 지식(divergent knowledge)을 이용하여 무작위 정보 획득을 촉진하는 ACXON(Algorithm for Contrastive eXplanatory Ontology-based Narratives) 알고리즘을 소개하였습니다. 이 알고리즘은 최종적으로 텍스트 기반의 대조 서사(narrative)를 생성하며, 사용자 요구에 따라 다양한 세부사항으로 결과를 조절할 수 있습니다.

- **Performance Highlights**: 제안된 ACXON 알고리즘은 기존의 바닥선(baseline) 알고리즘과 비교하여 향상된 성능을 보여주었습니다. 객관적인 평가 지표에 따르면, 더 적은 지식을 사용하여 내러티브(narratives)를 구성함으로써 의사소통 시간을 단축시킬 수 있음을 입증했습니다. 이는 로봇이 복잡한 결정 상황에서도 신뢰할 수 있는 설명을 제공하는 데 기여할 수 있음을 시사합니다.



### A Machine Learning Pipeline for Multiple Sclerosis Biomarker Discovery: Comparing explainable AI and Traditional Statistical Approaches (https://arxiv.org/abs/2509.22484)
Comments:
          Short paper presented at the 20th conference on Computational Intelligence methods for Bioinformatics and Biostatistics (CIBB2025)

- **What's New**: 이번 연구에서 우리는 다발성 경화증(Multiple Sclerosis, MS)에서 바이오마커 발견을 위한 기계학습 파이프라인을 제안합니다. XGBoost 분류기를 Bayesian 검색을 통해 최적화하고, SHapley Additive exPlanations (SHAP)을 사용하여 모델 예측의 주요 기능을 식별했습니다. 이 접근 방식은 전통적인 차등 발현 분석(Differential Expression Analysis, DEA)에서 발견된 유전자와 비교하여 MS와 관련된 바이오마커를 확인합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 공공에서 사용할 수 있는 8개의 Peripheral Blood Mononuclear Cells (PBMC) 마이크로어레이 데이터셋으로 구성됩니다. 각 데이터셋은 건강한 개인과 MS 진단을 받은 개인의 샘플을 포함하며, 연구 목적에 맞도록 전처리 및 정규화 과정을 거쳤습니다. 이 파이프라인은 R과 Python을 사용하여 구현되며, 데이터와 분석 결과는 GitHub에서 공개됩니다.

- **Performance Highlights**: SHAP와 DEA 방법 모두 167개의 유전자가 중복으로 발견되었으며, 각각 SHAP 특화된 133개 유전자와 DEA에서 식별된 1000개 이상의 유전자를 포함하였습니다. SHAP는 MS와 관련된 HLA-DRB1과 HLA-DRB5를 상위 유전자 중 하나로 분류했으며, IL1B 및 IL2RA와 같은 새로운 통찰력도 제시했습니다. 이에 따라 SHAP과 같은 해석 가능한 AI 방법이 기존의 통계적 방법과 비교하여 새로운 바이오마커를 발견하는 데 중요한 역할을 할 수 있음을 보여주었습니다.



### OFMU: Optimization-Driven Framework for Machine Unlearning (https://arxiv.org/abs/2509.22483)
Comments:
          Under review at ICLR 2026

- **What's New**: 본 논문은 기계 학습에서의 관행을 개선하기 위한 OFMU(Optimization-Driven Framework for Machine Unlearning)라는 새로운 최적화 기반의 이중 최적화 프레임워크를 제안합니다. OFMU는 잊기(forgetting)와 유지(retention) 목표를 명확히 구분하여, 최적화를 위한 단계에서 이 두 목표 간의 우선순위를 정함으로써 기존 방법들이 가지고 있는 한계점을 극복합니다. 기존의 방법들이 무관하게 두 목표를 결합하여 비효율적이던 것을 해결하며, 특정 데이터의 영향을 제거하면서도 모델의 유용성을 보존하는 능력을 강조하고 있습니다.

- **Technical Details**: OFMU는 두 가지 단계의 최적화 구조를 기반으로 하며, 잊기 목표에 대한 정돈된 페널티를 도입합니다. 이 구조는 잊기 목표와 유지 목표 간의 경량화된 경량화된 페널티를 통해 이 두 목표의 경량화를 보장합니다. 또한 OFMU는 완전 수렴 없이도 효율적인 두 루프 최적화가 가능하도록 설계되었습니다. 우리는 더 나아가, OFMU의 해결 속도를 이론적으로 분석하여 다양한 상황에서도 우수한 성능과 수렴 속도를 보장한다고 주장합니다.

- **Performance Highlights**: 자세한 실험 결과에 따르면, OFMU는 언어와 비전 모델의 기준 벤치마크에서 기존의 잊기 방법들보다 일관되게 더 우수한 성능을 나타냈습니다. OFMU는 잊기 효율성(forgetting efficacy)과 보존된 유용성(retained utility) 간의 새로운 균형을 제공하여 더 나은 성과를 달성했습니다. 기존 방법들에 비해, OFMU는 어려운 학습 사례에서도 안정적인 성능을 유지하면서 실제로 중요한 언Learnning 목표를 달성하는 데 기여했습니다.



### Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving (https://arxiv.org/abs/2509.22480)
Comments:
          17 pages, 11 figures

- **What's New**: 이 논문에서는 기존의 지도 세부 조정(Supervised Fine-Tuning, SFT)이나 강화 학습(Reinforcement Learning, RL) 방법과는 다른 접근 방식으로 언어 모델의 문제 해결 성능을 개선하는 전략인 솔루션 발산(solution divergence)이라는 개념을 새롭게 제안합니다. 솔루션 발산이란 단일 문제에 대해 생성된 여러 해결책 간의 다양성을 측정하는 지표로, 이는 다양한 모델에서 문제 해결 능력과 긍정적인 상관관계를 가지고 있음을 보여줍니다. 이러한 발견을 바탕으로 연구자들은 솔루션 발산을 사용하여 LLM 훈련과 평가를 향상시킬 수 있는 가능성을 탐구합니다.

- **Technical Details**: 논문에서 제안하는 솔루션 발산은 LLM이 생성한 솔루션의 집합에서 서로 다른 솔루션 간의 차이를 수치적으로 나타내며, 이는 LLM의 성능을 평가하는 새로운 지표로 사용됩니다. 연구진은 솔루션 집합의 평균 발산을 계산하기 위해 관계 그래프를 구성하였고, 그래프의 고유값(eigenvalues)을 통해 발산을 측정합니다. 이 과정에서 문자열 편집 거리(normalized string edit distance)를 사용하여 솔루션 간의 유사성을 측정하며, 이 방법은 다양한 도메인에서 일관되고 계산 효율적인 특성을 제공합니다.

- **Performance Highlights**: 제안된 솔루션 발산 메트릭을 세 가지 문제 영역, 즉 수학, 프로그래밍 및 논리적 추론 영역에 적용한 결과, 발산을 활용한 접근 방식이 성공률을 일관되게 향상시킨다는 것을 실증적으로 확인했습니다. 이러한 성과는 솔루션 발산이 LLM의 문제 해결 능력을 증대시키는 간단하면서도 효과적인 도구가 될 수 있음을 시사합니다. 교육 연구 또한 적절한 솔루션 다양성을 지닌 학습자가 더 좋은 학습 결과를 낸다는 것을 보여 주어, 이 메트릭의 활용 가능성을 뒷받침합니다.



### Evaluating the Limits of Large Language Models in Multilingual Legal Reasoning (https://arxiv.org/abs/2509.22472)
Comments:
          39 pages, 36 figures. Code and evaluation pipeline available at this https URL

- **What's New**: 본 연구는 다국어 환경에서의 법률 작업을 다루며, LLM(대형 언어 모델)인 Meta의 LLaMA와 Google의 Gemini의 성능을 평가합니다. 특히, 법률과 비법률 벤치마크에서의 성능을 비교하고, 인위적인 변형을 통한 법적 작업의 견고성을 분석합니다. 고유한 LLM-as-a-Judge 평가 접근 방식을 통해 인간과 정렬된 평가를 진행하며, 법적 작업에 초점을 맞춘 모듈형 오픈 소스 평가 파이프라인을 제공합니다.

- **Technical Details**: 이 연구는 법률적 언어와 추론의 복잡성이 LLM의 성능에 미치는 영향을 평가합니다. LLaMA와 Gemini의 두 가지 최신 모델을 사용하여 다국어 법률 분류, 요약, 공정성 예측, 법률적 추론 및 적대적 견고성 테스트를 수행합니다. 법률 작업은 기술적 용어와 복잡한 문장 구조로 인해 LLM이 일반 작업보다 낮은 정확도를 보이는 경향이 있음을 확인했습니다.

- **Performance Highlights**: 연구 결과, LLaMA는 Gemini보다 약 24%포인트 낮은 성능을 보였으며, 법률 추론 벤치마크에서는 LLM이 50% 미만의 정확도를 기록하는 경우가 많았습니다. 영어는 전반적으로 더 안정적인 성능을 보여주지만, 더 높은 정확도를 보장하지는 않습니다. 또한, 언어의 성능은 영어와의 구문 유사성과 정비례하는 경향이 있으며, LLM의 응답 중립성 및 긍정성 경향은 공정성 분류 작업에서 두드러지게 나타났습니다.



### Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining (https://arxiv.org/abs/2509.22468)
- **What's New**: C-FREE (Contrast-Free Representation learning on Ego-nets) 프레임워크는 2D 그래프와 3D conformer의 조합을 통합하여 분자의 표현을 학습하는 새로운 접근 방식을 제시합니다. 기존 방법들은 수작업으로 제작된 보강(augmentation)이나 복잡한 생성 목표에 의존하거나 2D 토폴로지에만 국한되는 경우가 많습니다. C-FREE는 3D 구조 정보를 효과적으로 활용하여, 다양한 화학 도메인에 맞춤형으로 전이할 수 있는 성능을 입증했습니다.

- **Technical Details**: C-FREE는 고정 반지름 ego-net을 사용하여 서로 보완적인 이웃에서 하위 그래프 엠베딩을 예측하는 방식으로 분자 표현을 학습합니다. 이 방식은 Graph Neural Network (GNN)과 Transformer의 혼합 구조를 통해 기하학적 정보와 위상 정보를 통합하며, 비용이 많이 드는 전처리나 네거티브 샘플링에 의존하지 않습니다. GEOM 데이터셋에서 사전 훈련(pretraining)하여 MoleculeNet에서 최첨단 결과를 달성하였습니다.

- **Performance Highlights**: C-FREE는 2D 및 3D conformations을 모두 활용하여 높은 예측 성능을 보여주고 있으며, 기존의 자기 지도(self-supervised) 모델들과 비교해도 쏠림(linear-probe) 평가 및 전체 미세 조정(full fine-tuning) 모두에서 유의미한 성과를 거두었습니다. MoleculeNet에서 평균 성능이 가장 뛰어난 점수를 기록하며, 새로운 다중 모달(multi-modal) 화학 벤치마크에도 강력하게 전이되는 모습을 보여주었습니다.



### MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark (https://arxiv.org/abs/2509.22461)
Comments:
          25 pages, 7 figures

- **What's New**: MDAR는 복잡하고 동적으로 발전하는 오디오 추론 작업을 평가하기 위해 설계된 새로운 벤치마크입니다. 기존 벤치마크는 주로 정적 상황이나 단일 장면에 초점을 맞추었으며, 다수의 화자와 환경 소음이 공존하는 현실 세계의 다양한 상황을 충분히 포착하지 못했습니다. MDAR는 3,000개의 정교하게 구성된 질문-응답 쌍을 포함하고 있으며, 다섯 가지 복잡한 추론 범주에 대한 평가를 가능하게 합니다.

- **Technical Details**: MDAR는 오디오 언어 모델의 복잡한 추론 능력을 평가하기 위한 세 가지 작업 유형을 사용합니다: 단일 선택 질문, 다중 선택 질문, 오픈 엔디드 질문. 각 범주는 장면 이해, 사회적 관계, 사건 추론, 시간적 추론, 이상 탐지 및 안전을 포함하여 고품질의 오디오 클립과 인간 주석 질문이 결합된 새로운 데이터 세트를 기반으로 합니다. MDAR는 또한 이전의 벤치마크와는 달리 동적이고 혼합된 오디오 장면을 제공하여 실제적인 벤치마크를 설정했습니다.

- **Performance Highlights**: 26개의 최첨단 오디오 언어 모델을 MDAR에서 평가한 결과, 복잡한 추론 작업에서 제한이 있음을 발견했습니다. Qwen2.5-Omni는 단일 선택 질문에서 76.67%의 정확도를 달성한 반면, GPT-4o Audio는 68.47%에 그쳤습니다. 모든 질문 유형에서 모델이 80% 이상의 성능을 발휘하지 못하는 것으로 나타났으며, 이는 미래 오디오 추론 에이전트의 개선이 필요하다는 것을 강조합니다.



### Physics-informed GNN for medium-high voltage AC power flow with edge-aware attention and line search correction operator (https://arxiv.org/abs/2509.22458)
Comments:
          5 pages, 2 figures. Submitted to ICASSP 2026. Code available at this https URL

- **What's New**: 이번 논문에서는 전기 물리학에 기반한 그래프 신경망(Physics-informed Graph Neural Networks, PIGNN)을 활용하여 빠른 AC 전력 흐름 해법을 제시합니다. 기존의 PIGNN은 정확성 향상이 필요하였으며, 특히 추론(inference) 시 물리적 손실이 작동하지 않아 실제 운영에서의 채택에 장애가 있었습니다. 이를 해결하기 위해 새로운 PIGNN-Attn-LS 아키텍처를 도입하여, 에지 인식 주의(attention) 메커니즘을 통해 전력망의 비대칭성을 캡처하고 전역 교정 방사를 구현했습니다.

- **Technical Details**: 제안된 PIGNN-Attn-LS는 복잡한 노드 상태를 가진 무방향 그래프 모델링을 기반으로 합니다. 경로 에지의 물리적 특성을 주입하여 전자적 메시지 전송 과정에서 전력 불일치를 업데이트하는 방식으로 작동합니다. 각 단계에서 잔차(Power mismatches)를 계산하고 메시지를 교환하여 AC 전력 흐름 방정식 만족을 목표로 하는 학습 신호를 생성합니다. 또한, 추론 시에는 적응적인 단계 크기를 선택할 수 있는 백트래킹 라인 서치(backtracking line search)를 포함하여, 각 반복에서 이전 값들로의 감소를 보장합니다.

- **Performance Highlights**: PIGNN-Attn-LS는 4-32 버스(grid)에서 테스트했을 때, 전압에 대해RMSE 0.00033, 각도에 대해 0.08도라는 성과를 달성했으며, 이는 기존 PIGNN-MLP에 비해 각각 99.5% 및 87.1% 향상된 결과입니다. 또한, 4-1024 버스 그리드에서 뉴턴-랩슨(Newton-Raphson) 해법에 비해 2-5배 더 빠른 배치 추론(batched inference)을 제공합니다. 이러한 성과는 PIGNN의 실제 운영 가능성을 높임과 동시에 정확성과 속도를 모두 달성할 수 있도록 합니다.



### Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers (https://arxiv.org/abs/2509.22445)
- **What's New**: 이번 논문은 기계 학습에서 Occam's razor를 적용하기 위한 공식적인 틀인 Minimum Description Length (MDL) 원칙을 신경망, 특히 Transformers에 적용하는 데 따른 어려움을 다루고 있습니다. 저자들은 Kolmogorov 복잡도의 이론에 근거하여 점근적으로 최적화된 설명 길이 목표의 이론적 개념을 소개합니다. 또, 이러한 목표를 최소화하는 것이 모델 자원 제한이 증가함에 따라 최적의 압축을 달성한다는 것을 보여줍니다.

- **Technical Details**: 연구는 Transformers에 대해 점근적으로 최적화된 목표가 존재함을 입증하며, 이는 이들의 계산적 보편성을 새로운 방식으로 시연하여 이루어졌습니다. 저자들은 적응형 가우시안 혼합 선행 지식을 바탕으로 만드는 변량 목표를 통해 이러한 목표가 취급 가능하고 미분 가능하다는 것을 보여줍니다. 이를 통해 최적화 과정에서 발생할 수 있는 문제를 다루고, 수학적으로 기반한 목표 설정이 신경망 훈련에 유용함을 논의합니다.

- **Performance Highlights**: 경험적 분석 결과, 변량 목표는 알고리즘 작업에서 강력한 일반화를 보여주는 저복잡도 해결책을 선택하는 것으로 나타났습니다. 그러나 표준 최적화 방법은 무작위 초기화에서 이러한 해결책을 찾는 데 실패하였으며, 이는 최적화 과정에서의 중요 과제를 강조합니다. 이 논문은 강한 점근적 보장을 가진 설명 길이 목표를 판별하기 위한 이론적 틀을 제공함으로써 더 나은 압축 및 일반화를 이룰 수 있는 신경망 훈련의 잠재적 경로를 제시하고 있습니다.



### Learning to Ball: Composing Policies for Long-Horizon Basketball Moves (https://arxiv.org/abs/2509.22442)
Comments:
          ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2025). Website: this http URL. Video: this https URL. Code: this https URL

- **What's New**: 이번 논문에서는 다단계(long-horizon) 작업에서 모호한 중간 상태를 가진 스킬들을 통합하는 새로운 정책 통합 프레임워크를 제안합니다. 강화학습(RL) 방법들이 과거부터 이러한 다단계 작업의 정책 구성의 어려움을 겪어왔지만, 본 연구는 사전 학습된 정책을 활용하여 이 문제를 해결하는 방법을 소개합니다. 최종적으로, 높은 수준의 소프트 라우터(high-level soft router)를 도입하여 중간 서브태스크들 간의 매끄럽고 강력한 전환을 가능하게 합니다.

- **Technical Details**: 종래의 정책 통합 방법들이 개별적인 서브태스크의 종료 상태가 명확해야 했던 반면, 본 연구에서는 각 서브태스크 A, B, C의 명확한 목표와 B의 모호한 목표를 구분하여 B를 A의 정책에 의해 트레이닝합니다. 이때, C의 상태 가치 함수(state value function)를 활용하여 보상을 설정하여 B에서 C로의 전환을 원활하게 만듭니다. 우리의 방식은 비구조적 데이터에서도 정책을 생성할 수 있어, 다양한 농구 모션 데이터를 활용한 점이 특징적입니다.

- **Performance Highlights**: 제안된 방법을 통해 학습된 정책은 실시간으로 농구 모션을 수행하며, 드리블 후 점프슛을 하는 데 91.8%의 성공률을 기록했습니다. 또한, 다수의 에이전트가 상호작용하며 농구 팀 플레이를 수행하는 모습을 통해 본 연구의 효과성을 입증했습니다. 연구 결과는 부드러운 농구 동작과 사용자의 명령에 대한 적응성을 보여 주며, 기존 방법들이 갖고 있는 한계를 극복하는 데 기여합니다.



### Chimera: Diagnosing Shortcut Learning in Visual-Language Understanding (https://arxiv.org/abs/2509.22437)
Comments:
          Our code (this https URL) and data (this https URL) are publicly available

- **What's New**: 이번 논문에서는 Chimera라는 새로운 테스트 스위트를 소개합니다. 이 테스트 스위트는 7,500개의 고품질 다이어그램을 포함하고 있으며, 각 다이어그램은 의미론적 트리플(semantic triples)로 주석이 달립니다. Chimera는 다이어그램 이해의 네 가지 기본 요소인 개체 인식(entity recognition), 관계 이해(relation understanding), 지식 기초(knowledge grounding), 시각적 추론(visual reasoning)을 평가하기 위한 다양한 수준의 질문들을 포함하고 있습니다.

- **Technical Details**: Chimera 데이터셋은 Wikipedia의 다이어그램 이미지를 수집하여 구성하였으며, 불필요한 이미지를 필터링하여 품질이 뛰어난 다이어그램을 보장합니다. 각 다이어그램에는 의도한 내용을 기술하는 주석과, 다이어그램 이해 과정에서의 시퀀스를 분석하기 위한 네 가지 수준의 질문이 포함되어 있습니다. 테스트는 비주얼 및 의미적 모달리티를 통해 다이어그램의 이해를 평가하며, 다이어그램의 내용을 다양한 표현 렌즈로 재구성하여 제출하게 됩니다.

- **Performance Highlights**: 15개의 오픈 소스 VLM을 대상으로 한 평가 결과, 모델들은 다이어그램 이해에서 공통적인 단축키(시각적 단기기억, 지식 회상, Clever-Hans)를 활용하는 경향이 있음을 보였습니다. 특히 Clever-Hans 단축키가 특히 심각하여, 모델들이 진정한 이해 없이 표면적인 언어 패턴을 기반으로 높은 성과를 이룰 수 있음을 보여주었습니다. 이 연구 결과는 현재 VLM의 한계를 드러내며, 진정한 시각적 이해를 위한 강력한 평가 프레임워크의 필요성을 강조합니다.



### Global Convergence in Neural ODEs: Impact of Activation Functions (https://arxiv.org/abs/2509.22436)
Comments:
          ICLR 2025 (Oral)

- **What's New**: 이번 논문에서는 Neural Ordinary Differential Equations (ODEs)의 학습 과정에서 발생하는 주요 도전 과제를 다루고 있으며, 특히 활성화 함수(activation function)의 영향을 조사합니다. 활성화 함수의 특성인 부드러움(smoothness)과 비선형성(nonlinearity)이 Neural ODEs의 학습 역학에 미치는 중요한 역할을 보여줍니다. 또한, 이러한 특성을 통해 Neural ODE가 전반적인 수렴을 이루도록 개선할 수 있는 이론적 기반을 제공합니다.

- **Technical Details**: Neural ODEs는 연속적인 시간 차분 방정식으로, 은닉 상태(hidden states)의 진화를 모델링합니다. 이 논문에서는 활성화 함수가 두 ODE의 고유한 해를 보장하도록 돕고, Neural Tangent Kernel (NTK)의 스펙트럼 성질을 유지하기 위해 비선형성이 중요함을 강조합니다. 본 연구에서는 연속 모델에서 NTK가 잘 정의될 수 있음을 보여주며, 특히 비폴리노미얼(non-polynomial) 활성화 함수가 NTK의 양의 정의 성질을 유지하는 데 도움이 된다는 점을 강조합니다.

- **Performance Highlights**: 이론적 연구뿐 아니라, 다양한 수치 실험을 통해 활성화 함수의 부드러움과 비선형성이 Neural ODE의 수렴 속도를 가속화하고 성능을 향상시킬 수 있음을 입증했습니다. 하지만 잘못된 ODE 규모 조정은 수치 오차로 인한 저항을 초래할 수 있으며, 큰 규모의 Neural ODE에서 적응형 솔버는 비효율성을 초래하고 불안정성을 만들 수 있습니다. 이러한 실험 결과는 Neural ODE의 실제 적용을 위한 유용한 지침을 제공하며, 훈련 과정을 개선하는 데 기여할 수 있습니다.



### An Ontology for Unified Modeling of Tasks, Actions, Environments, and Capabilities in Personal Service Robotics (https://arxiv.org/abs/2509.22434)
- **What's New**: 본 연구에서 제안하는 OntoBOT(Ontology for roBOts and acTions)는 기존의 온톨로지를 확장하여 로봇의 작업, 행동, 환경 및 능력을 통합적으로 표현하는 구조를 제공합니다. 이를 통해 단순히 단일 도메인에 국한되지 않고, 다양한 로봇과 환경을 포괄하는 보다 포괄적인 지식 표현을 목표로 합니다. 이 모델은 타 작업 실행과 관계지식을 위한 논리적 추론을 지원하며, 다양한 로봇에서의 실험을 통해 그 일반성을 입증하였습니다.

- **Technical Details**: 서비스 로봇은 자율적 또는 반자율적 시스템으로, 가정이나 돌봄, 사회적 환경에서 인간을 돕기 위해 설계되었습니다. 그러나 기존의 시스템은 특정 하드웨어 및 소프트웨어 스택에 깊게 연결되어 있어 재사용 가능성과 상호 운용성을 제한하는 문제가 있습니다. 본 연구에서 제안된 온톨로지는 작업 구조, 환경 맥락 및 로봇 능력을 포괄적으로 캡처하고, 이를 통해 기계적으로 해석 가능한 지식을 제공합니다.

- **Performance Highlights**: OntoBOT의 성능은 TIAGo, HSR, UR3 및 Stretch라는 네 가지 이식 가능 로봇을 대상으로 한 평가를 통해 검증되었습니다. 이 평가에서는 다양한 능력을 지닌 로봇들이 맥락 인식 추론과 작업 중심 실행을 어떻게 지원하는지를 확인했습니다. 온톨로지의 사용으로 인해 서비스 로봇 간의 지식 공유가 용이해지고, 보다 유연하고 설명 가능한 동작을 구현할 수 있게 되었습니다.



### Partial Parameter Updates for Efficient Distributed Training (https://arxiv.org/abs/2509.22418)
- **What's New**: 이 논문에서는 배포된 훈련에서 저통신(low-communication)을 위한 메모리 및 계산 효율적인 방법을 제안합니다. 기존의 방법은 통신을 줄이기 위해 드문 글로벌 동기화 사이에 여러 번의 로컬 업데이트를 수행하는 방식입니다. 저자는 파라미터의 모든 업데이트가 아닌 고정된 하위 집합만 업데이트하는 방식으로 역전파(backpropagation)를 제한함으로써 효율성을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 방법은 각각의 노드가 파라미터의 일정한 슬라이스에 대해서만 역전파를 수행하도록 하여 메모리 사용량과 총 훈련 FLOPs를 줄입니다. 로컬 단계 후에 각 노드 간에 파라미터 차이를 평균내고 외부 옵티마이저로 업데이트를 진행합니다. 이 접근 방식은 높은 대역폭의 통신 의존도를 낮추고, 메모리 및 계산량을 줄이며, 훈련에서는 1.3B 파라미터의 언어 모델(Language Model)을 32개의 노드에서 훈련하여 이전의 저통신 접근 방식의 성능과 유사한 결과를 도출합니다.

- **Performance Highlights**: 새로운 접근 방식은 동일한 토큰 및 대역폭 예산 하에 이전 저통신 훈련 접근법과 유사한 perplexity를 달성하면서도 15% 더 적은 FLOPs와 최대 47% 더 적은 메모리를 사용하는 성과를 보여주었습니다. 저자들은 또한 시뮬레이션된 저대역폭 환경에서, 본 방법이 매 단계 동기화를 사용하는 표준 분산 데이터 병렬 훈련보다 상당히 빠르게 수렴하는 것을 증명했습니다.



### Explaining multimodal LLMs via intra-modal token interactions (https://arxiv.org/abs/2509.22415)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 해석 가능성을 높이기 위해 인트라-모달( intra-modal) 상호작용을 활용한 새로운 접근 방식을 제안합니다. 기존의 연구는 주로 크로스-모달( cross-modal) 기여도 분석에 초점을 맞추었으나, 인트라-모달 의존성을 간과했습니다. 이로 인해 단편적이고 노이즈가 포함된 설명이 발생하는 문제를 해결하기 위해, 새로운 해석 가능성 프레임워크를 도입했습니다.

- **Technical Details**: 제안된 접근법은 두 가지 보완적인 구성 요소로 구성됩니다. 하나는 Multi-Scale Explanation Aggregation (MSEA)으로, 다양한 스케일의 입력을 통해 시각적 기여도를 계산하고 통합하여 공간적 컨텍스트를 포착합니다. 또 하나는 Activation Ranking Correlation (ARC)으로, 현재 토큰과 관련된 지각적 정보의 상대적 중요성을 평가하여 무관한 선행 토큰의 영향을 줄이는 방법을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 다양한 MLLM 모델에서 기존의 해석 방법을 초월하는 성능을 보였습니다. Qwen2-VL-2B 모델을 포함한 여러 모델에서 3.69%에서 14.52%의 정량적 개선을 기록하였으며, COCO Caption 데이터셋을 사용할 때 더욱 두드러진 성과를 나타냈습니다. 이를 통해 인트라-모달 상호작용을 고려한 새로운 접근 방식의 유효성을 강화했습니다.



### RAU: Reference-based Anatomical Understanding with Vision Language Models (https://arxiv.org/abs/2509.22404)
- **What's New**: 본 논문은 RAU라는 새로운 프레임워크를 소개하며, 이는 시각-언어 모델(VLMs)을 활용한 참조 기반(anatomical understanding) 해석을 가능하게 합니다. 동작 중에 제공되는 참조 이미지를 사용해 비슷한 목표 이미지를 이해하도록 VLM을 학습시킵니다. 이를 통해 작은 해부학적 영역에 대한 정밀한 표시(segmentation)와 위치 확인(localization)을 통합하는 데 성공하였습니다.

- **Technical Details**: RAU는 참조 이미지를 기반으로 VLM이 상대적인 공간 추론(relative spatial reasoning)을 통해 해부학적 영역을 식별하도록 학습합니다. 또한, SAM2와의 통합을 통해 정밀한 세분화(segmentation) 능력을 결합하여 작은 해부학적 구조를 지역화할 수 있게 했습니다. 이 과정에서 시각적 질문 답변(visual question answering, VQA) 및 바운딩 박스 예측(bounding box prediction)으로 그 능력을 검증했습니다.

- **Performance Highlights**: RAU는 두 가지 기준 내(in-distribution) 데이터셋과 두 가지 기준 외(out-of-distribution) 데이터셋에서 일관되게 SAM2의 파인 튜닝(fine-tuning) 기준선을 초과하는 성과를 보여주었습니다. 특히, RAU는 분포 변동에 대한 강력한 일반화 능력을 갖추고 있어 자동화된 임상 워크플로우에서 큰 잠재력을 가지고 있습니다. 본 연구는 참조 기반으로 해부학적 구조를 식별하고 지역화하는 VLM의 최초 탐색을 포함하여 의료 이미지 이해를 위한 VLM 주도 접근 방식의 가능성을 강조합니다.



### Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss (https://arxiv.org/abs/2509.22394)
- **What's New**: 이번 연구에서는 의료 영상 번역을 위해 nnU-Net 프레임워크를 새로운 방식으로 적용했습니다. SynthRAD2025 대회에서 MR(Magnetic Resonance)에서 CT(Computed Tomography) 및 CBCT(Cone-Beam CT)로의 이미지 전환 작업에 초점을 맞추고 있습니다. 특히, Anatomical Feature-Prioritized (AFP) 손실 함수를 도입하여 임상적으로 중요한 구조의 재구성을 향상시켰습니다.

- **Technical Details**: 제안된 모델은 표준 U-Net과 잔여 U-Net(residual U-Net) 두 가지를 사용하며, 두 모델 모두 nnU-Net로부터 기인하였습니다. MRI 데이터는 개별적으로 z-score 정규화가 적용되었고, CBCT와 CT는 데이터셋 수준에서의 z-score 정규화와 클리핑을 통해 처리되었습니다. 훈련 데이터는 해부학적 영역에 맞춰 3D 패치로 구성되며, 1000 및 1500 에폭(epoch) 동안 훈련된 후, AFP의 미세 조정이 수행됩니다.

- **Performance Highlights**: 두 모델은 모든 영역에서 일관된 설계를 가능하게 하며, L1 및 AFP 손실을 활용한 경우 해부학적 신뢰성을 크게 향상시킨 것으로 평가되었습니다. 결과적으로 잔여 네트워크와 AFP의 조합은 뼈 구조와 병변 재구성의 질을 높이며, 의료 영상의 다중 모드 합성을 위한 안정적인 해결 방안을 제공합니다. 이를 통해 환자의 방사선 노출을 최소화하고, 전체 치료 계획 프로세스를 간소화 할 수 있습니다.



### SpinGPT: A Large-Language-Model Approach to Playing Poker Correctly (https://arxiv.org/abs/2509.22387)
Comments:
          Accepted at Advances in Computer Games (ACG) 2025, LNCS (Springer)

- **What's New**: 이번 연구에서는 SpinGPT라는 대형 언어 모델(LLM)을 소개하며, 이는 인기 있는 삼인용 온라인 포커 포맷인 Spin & Go에 맞춰 개발되었습니다. 기존의 Counterfactual Regret Minimization (CFR) 알고리즘의 한계를 극복하고자 하였습니다. SpinGPT는 Supervised Fine-Tuning과 Reinforcement Learning의 두 단계에 걸쳐 훈련되어, 포커 게임에서의 결정 과정에서 효율성을 높였습니다.

- **Technical Details**: SpinGPT는 320,000개의 고위험 화폐 전문가 결정에 대한 지도 학습(Supervised Fine-Tuning)과 270,000개의 해결자 생성 손에 대한 강화 학습(Reinforcement Learning)으로 구성된 두 단계의 훈련 과정을 통해 학습되었습니다. 이 모델은 다양한 상황에서 신뢰할 수 있는 결정을 내릴 수 있도록 설계되었습니다. 또한, 삼인 이상 플레이어의 게임에서의 성능을 극대화하는 데 필요한 새로운 접근 방식을 모색하고 있습니다.

- **Performance Highlights**: SpinGPT는 결정의 78%에서 해결자의 행동과 일치하는 높은 수용 가능성을 보였습니다. 단순한 딥 스택 휴리스틱을 사용하여 Slumbot과의 30,000 핸드에서 13.4 +/- 12.9 BB/100의 성과를 달성하였습니다. 이 결과는 LLM이 포커와 같은 다인 다결정 정보가 불완전한 게임을 다루는 새로운 방법이 될 수 있음을 시사합니다.



### Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach (https://arxiv.org/abs/2509.22378)
- **What's New**: 이번 연구에서는 첫 번째 Vision Language Model (VLM) 기반의 Image-to-Music (I2M) 프레임워크를 제안합니다. 이 프레임워크는 높은 해석 가능성과 낮은 계산 비용을 제공하여 사용자들이 생성된 음악의 결과를 쉽게 이해할 수 있도록 합니다. ABC 표기법을 활용하여 텍스트와 음악을 연결하며, 이를 통해 자연어로 음악을 생성할 수 있습니다. 또한 다중 모드 검색 증강 생성(Multi-Modal Retrieval-Augmented Generation, RAG) 및 자기 정제 기법을 적용하여 외부 훈련 없이도 고품질 음악을 제작할 수 있습니다.

- **Technical Details**: 방법론에서는 VLM을 사용한 음악 생성 과정이 세 가지 주요 단계로 나뉘어 있습니다: 다중 모드 RAG를 통한 기본 음악 생성, 모델 기반 평가기를 사용한 음악 정제, 텍스트 출력 및 이미지 주의 맵을 통한 설명 생성입니다. ABC 표기법을 사용하여 이미지와 음악 간의 다리를 놓고, CLIP 모델을 통해 음악 설명을 임베딩으로 변환하여 유사성을 평가합니다. 평가 기준으로는 Pitch Range, Polyphony, Scale Consistency 등 다양한 음악 특성을 포함하고, 이를 통해 생성된 음악을 정교하게 개선합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 음악 품질 및 음악-이미지 일관성 측면에서 기존 I2M 방법들보다 우수한 성과를 나타냈습니다. 인간 연구 및 기계 평가를 통해 검증된 결과, 제안된 프레임워크는 높은 점수를 기록하며, 음악 생성 품질을 크게 개선하는 데 성공했습니다. 이로써 다중 모드 음악 생성의 잠재력을 보여주는 결과를 도출하였습니다.



### What Is The Political Content in LLMs' Pre- and Post-Training Data? (https://arxiv.org/abs/2509.22367)
Comments:
          9 pages, under review

- **What's New**: 이번 연구에서는 OLMO2라는 대규모 오픈소스 모델의 학습 데이터를 분석하여 정치적 편향의 기원을 규명하려고 합니다. 이 모델의 요구 데이터셋은 공개되어 있으며, 이를 통해 훈련 데이터에 포함된 정치적 콘텐츠의 비율을 도출하고, 모델의 특정 정책 이슈에 대한 입장과의 상관관계를 평가합니다. 또한, 학습 전후 데이터 세트를 비교하여 정치적으로 기여한 콘텐츠의 양을 증가시킬 수 있는 지침을 제공합니다.

- **Technical Details**: 연구진은 OLMO2의 전후 훈련 데이터에서 무작위 샘플을 추출하고, 이를 정치적 방향으로 자동 주석 처리합니다. 이러한 데이터는 좌파, 우파, 중립으로 분류되며, 새로운 분류기를 통해 유효성을 검증합니다. 분석 결과, 좌파 성향의 문서가 우파 성향의 문서보다 3배에서 12배 더 많고, 훈련 데이터에서 정치적으로 관여한 콘텐츠는 사전 훈련 샘플에서 네 배 더 많음을 발견했습니다.

- **Performance Highlights**: 결과적으로, 연구는 정치적 편향이 주로 훈련 과정에서 형성된다는 것을 강조합니다. 또한, 모델 행동과 훈련 데이터에서의 지배적 관점 사이에 강한 상관관계(rr=0.90)가 있음을 보여줍니다. 이 연구는 훈련 데이터의 정치적 콘텐츠 분석이 향후 LLM 개발 및 검증 과정에서 필수적임을 시사합니다.



### CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models (https://arxiv.org/abs/2509.22360)
- **What's New**: 이 논문은 CHRONOBERG라는 250년 동안의 영어 책 텍스트로 구성된 시간적으로 구조화된 말뭉치를 도입합니다. 기존의 다양한 웹 크롤링 데이터셋은 장기적인 시간 구조가 부족하여 LLMs의 언어의 의미와 규범적 변화 맥락을 잘 반영하지 못합니다. CHRONOBERG는 Project Gutenberg에서 수집된 자료를 바탕으로 하여 시간에 따라 변화하는 어휘 의미를 분석할 수 있도록 합니다.

- **Technical Details**: CHRONOBERG는 2.7B (billion) 토큰으로 구성된 데이터셋으로, 레퍼런스가 되는 전통적인 Valence-Arousal-Dominance (VAD) 어휘가 거의 30만 개의 단어를 포함하도록 확장되었습니다. 이를 통해 정서적 의미의 변화와 감정의 추적을 지원하는 정량적 분석이 가능합니다. 이 데이터셋은 역사적 의미 전이를 반영한 교육을 통해 LLM이 더욱 효과적으로 발전할 수 있도록 돕습니다.

- **Performance Highlights**: 제안된 CHRONOBERG는 LLM이 과거의 맥락에서 언어를 인식하고 적응하는 능력의 부족을 드러내며, 과거 정보의 망각과 미래 문장에 대한 일반화를 다루는 데 어려움을 겪고 있음을 보여줍니다. 또한, 현대 LLM 기반의 도구들이 차별 언어의 탐지와 다양한 시간대에서의 정서적 맥락화에 더 나은 위치에 티기 위해서는 시간이 준수된 처리 방법이 필요함을 강조합니다.



### Forecasting the Future with Yesterday's Climate: Temperature Bias in AI Weather and Climate Models (https://arxiv.org/abs/2509.22359)
Comments:
          13 pages, 5 figures

- **What's New**: 이 연구에서는 AI 기반 기후 및 날씨 모델의 역사적 데이터에 기반하여 미래 기후를 예측하는 데 있어 상온 편향 문제를 조사합니다. FourCastNet, Pangu와 같은 두 기상 모델과 ACE2 기후 모델을 평가하며, 이 모델들이 예측하는 기온이 실제로는 15~20년 이전의 기후와 유사하다는 점을 발견했습니다. 특히 일부 지역에서는 이 모델들의 예측이 20~30년 전에 비슷한 결과를 보이는 것으로 나타났습니다.

- **Technical Details**: FourCastNet과 Pangu는 NVIDIA와 Huawei가 설계한 완전 데이터 기반 AI 기상 모델입니다. 두 모델은 각각 ECMWF Reanalysis v5 (ERA5) 데이터를 사용하여 훈련되었으며, FourCastNet은 Spherical Fourier Neural Operator (SFNO) 아키텍처를 활용합니다. ACE2는 대기 전용 AI 기후 모델이며, 안정적인 100년 기후 시뮬레이션을 위해 CO2 강제력이 포함된 구조를 갖추고 있습니다.

- **Performance Highlights**: 모델의 예측은 일반적으로 과거 15~20년 이전의 클라이밋 데이터를 반영하는 경향이 있으며, FourCastNet과 Pangu는 특히 가장 더운 예측 기온에서 강한 냉각 편향을 보였습니다. ACE2는 기후 변화가 두드러진 지역과 계절에서 편향이 점차적으로 더 크게 나타나며, 이 연구는 CI 모델들이 역사적 데이터만으로 훈련될 때의 한계를 강조합니다.



### Stochastic activations (https://arxiv.org/abs/2509.22358)
- **What's New**: 본 논문에서는 Stochastic Activations(확률적 활성화)라는 새로운 접근 방식을 소개합니다. 이 전략은 대규모 언어 모델의 Feed-Forward Layer(전방향 계층)에서 여러 비선형 함수 중 하나를 무작위로 선택하는 방법으로, 실제로는 SILU와 RELU를 Bernoulli 샘플링에 따라 선택합니다. 이러한 방식은 RELU의 최적화 문제를 해결하고, CPU에서의 추론 속도를 크게 향상시킵니다.

- **Technical Details**: 기존의 RELU는 음수 입력에 대해 기울기가 0이 되어 가중치 업데이트가 이루어지지 않는 'Dying RELU 문제'를 겪습니다. SILU 활성화는 모델 정확도에서는 높은 성능을 보이지만 희소성(sparsity)을 제공하지 않습니다. 본 연구는 활성화 미세 조정(Swi+FT)과 StochA라는 두 가지 접근 방식을 통해 이러한 문제를 해결하려 합니다.

- **Performance Highlights**: 논문에서는 Stochastic Activations를 사용하여 생성 성능을 평가하며, 이 전략이 전통적인 SILU와 온도 스케일링을 조합한 것에 비해 경쟁력 있는 결과를 보인다고 주장합니다. 두 가지 접근 방식 모두 희소성의 장점을 누리며, 추론 시에도 더 좋을 성능을 발휘하여 다양한 시퀀스를 생성할 수 있는 새로운 대안을 제공합니다.



### Context and Diversity Matter: The Emergence of In-Context Learning in World Models (https://arxiv.org/abs/2509.22353)
- **What's New**: 최근 연구에서는 환경 동적 예측 능력이 생물학적 신경 시스템과 일반화된 임베디드 AI의 핵심 요소임을 강조하고 있습니다. 현재의 접근 방법들이 정적 세계 모델에 의존하고 있다는 점에 주목하며, 본 논문은 In-Context Environment Learning (ICEL)으로의 전환을 제안합니다. 이 방법론은 제로샷 성능에서 세계 모델의 성장과 비대칭 한계를 조사하는 데 초점을 맞춥니다.

- **Technical Details**: 연구에서는 두 가지 핵심 메커니즘인 환경 인식(Environment Recognition)과 환경 학습(Environment Learning)을 정립하고, 이들 메커니즘의 오류 상한(upper-bounds)을 유도하였습니다. 이러한 메커니즘의 출현은 데이터 분포에 따라 달라지며, 다양한 환경과 복잡성, 궤적의 길이에 의존함을 보여주었습니다. 또한, 새로운 세계 모델인 L2World를 소개하여 긴 시퀀스를 고효율적으로 모델링할 수 있음을 입증하였습니다.

- **Performance Highlights**: L2World는 카트-폴 제어와 비전 기반 실내 내비게이션에서 분포 특성과 긴 문맥 용량에 따라 ICER 또는 ICEL을 발동하는 방식으로 진행되었습니다. 비록 경량 이미지 인코더 및 디코더를 활용했지만, L2World는 내비게이션에서 긴 시퀀스 관찰 예측에 대한 새로운 최첨단 성능을 기록했습니다. 이러한 결과는 ICEL을 자극하기 위해 신중하게 설계된 데이터 세트와 모델 아키텍처의 중요성을 강조합니다.



### SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis (https://arxiv.org/abs/2509.22352)
- **What's New**: 이 논문은 SurvDiff라는 새로운 전이(diffusion) 모델을 소개합니다. 이 모델은 생존 분석(survival analysis)에서의 합성 데이터 생성(synthetic data generation)을 위해 특별히 설계되었습니다. SurvDiff는 혼합형 변수를 공동으로 생성하고, 이벤트 시간(event time)과 우측 검열(right-censoring)을 함께 고려하여 생존 데이터의 생성 메커니즘을 정확히 포착합니다.

- **Technical Details**: SurvDiff는 생존 특화 손실 함수(survival-tailored loss function)에 의해 안내받아 변수를 생성합니다. 이 손실 함수는 시간-사건 구조(time-to-event structure)를 인코딩하고, 검열 메커니즘을 명시적으로 고려합니다. 또한, 우측 검열을 고려하여 데이터에 대한 지지를 더 많이 가지는 초기 사건 시간에 더 높은 가중치를 부여하여 훈련 안정성을 향상시킵니다.

- **Performance Highlights**: SurvDiff는 여러 의학 데이터셋에서 기존 방법들보다 높은 성능을 발휘하며, 이벤트 시간 분포(event-time distribution)와 검열 메커니즘을 잘 보존합니다. 특히, 나이브한 전이 모델을 활용한 연구와 비교했을 때 우수한 결과를 제공합니다. SurvDiff는 합성 생존 데이터 생성에 있어 최초의 전이 모델로, 생존 분석 분야의 발전에 기여할 것으로 기대됩니다.



### Transformers Can Learn Connectivity in Some Graphs but Not Others (https://arxiv.org/abs/2509.22343)
Comments:
          Under Review

- **What's New**: 이 연구는 transformer 기반의 대형 언어 모델(LLMs)이 전이 관계(transitive relations)를 추론하는 능력을 분석합니다. 특히, 그래프에서의 연결성(connectivity) 문제를 학습하기 위한 transformers의 성능에 초점을 맞추어, 다양한 크기의 방향 그래프를 생성하여 훈련하는 방식을 사용합니다. 연구 결과, 저차원 그리드 그래프에서 transformers가 연결성을 학습하는 데 성공적인 반면, 연결이 끊긴 그래프에서는 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 transformers가 전이 관계를 추론할 수 있는지를 평가하기 위해 훈련 예제를 사용하여 연결성을 학습하는 능력을 조사합니다. 저차원 그리드 그래프에서는 노드들이 저차원 서브스페이스로 쉽게 임베딩(embedding) 될 수 있어 연결성이 쉽게 추론될 수 있습니다. 반면, 고차원 그리드 그래프와 연결이 끊긴 체인 그래프에서는 연결성 학습이 어려움을 겪습니다. 모델 크기를 확장하는 것이 그리드 그래프에서 연결성을 학습하는 데 긍정적인 영향을 미친다고 밝혀졌습니다.

- **Performance Highlights**: 연구 결과는 transformers가 저차원 그리드 그래프에서 연결성을 학습하는 데 성공적이라는 것을 보여줍니다. 그러나 고차원 그리드 그래프와 많은 구성 요소를 가진 연결 끊긴 그래프에서는 성능이 떨어지는 경향을 보입니다. 그래프 크기를 확장하였을 때 연결이 끊긴 체인 그래프에 대한 transformers의 성능이 개선된다는 점도 주목할 만 합니다. 요약하자면, transformers는 연결성 학습에 있어 그래프 크기를 확장하는 것에서 더 많은 혜택을 보임을 보여줍니다.



### Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs (https://arxiv.org/abs/2509.22338)
Comments:
          15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)

- **What's New**: 본 논문에서는 자연어(natural language)를 1차 논리(first-order logic)로 자동 번역하는 과정에서의 대규모 언어 모델(LLM)의 성능을 평가합니다. 새로운 과제에 대한 LLM의 성능을 비교하기 위해 다양한 아키텍처(encoder-decoder vs. decoder-only)와 훈련 전략을 검토하였습니다. 특히, 우리는 MALLS와 Willow 데이터셋을 사용하여 어휘 확장(vocabulary extension), 술어 조건부(predicate conditioning), 다국어 훈련(multilingual training)과 같은 기법을 탐구하였습니다.

- **Technical Details**: 모델 훈련을 위해 encoder-decoder 및 decoder-only 구조의 다양한 LLM을 사용하여 시스템적인 비교를 수행했습니다. Fine-tuning된 Flan-T5-XXL 모델이 술어 리스트를 활용하여 70%의 정확성을 달성하며, 기존의 모델들보다 우수한 성능을 보였습니다. 이 과정에서 Low-Rank Adaptation (LoRA)을 적용하여 모델의 훈련 상태를 개선하고, AdamW Optimizer를 통해 모델 성능을 극대화하는 방법을 사용했습니다.

- **Performance Highlights**: 주요 결과에 따르면, 술어의 가용성이 성능을 15-20% 향상시키며, T5 모델이 더 큰 decoder-only LLM보다 뛰어난 성능을 나타냅니다. 또한, 이러한 모델들은 특정 훈련 없이도 보지 못한 논리적 주장을 일반화할 수 있는 능력을 가지며, 기존의 해석과 비교할 수 있는 새로운 메트릭을 도입했습니다. 구조적 논리 번역의 강점과 함께, 술어 추출(predicate extraction)이 주요 병목현상으로 나타났습니다.



### Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning (https://arxiv.org/abs/2509.22335)
- **What's New**: 본 연구에서는 깊은 지속적 학습(deep continual learning)에서 깊은 신경망이 plasticity를 상실하는 이유를 조사합니다. 특히, 새로운 작업 초기화 시 Hessian의 스펙트럼 붕괴(spectral collapse)가 발생하며, 이로 인해 유의미한 곡률 방향이 사라지고 경량 하강(gradient descent)이 비효율적으로 변하는 과정을 설명합니다.

- **Technical Details**: 연구진은 성공적인 학습을 위한 필수 조건인 $	au$-trainability 개념을 소개하며, 기존의 plasticity 유지 알고리즘들이 이 프레임워크 안에서 통합될 수 있음을 보여줍니다. 스펙트럼 붕괴를 직접 목표로 하기 위해, Hessian의 Kronecker 인수분해(Kronecker factored approximation)에 대해 논의하고, 높은 유효(feature) 순위 유지 및 $L2$ 페널티를 적용하는 두 가지 정규화 강화 방법을 제안합니다.

- **Performance Highlights**: 연속적 감독(supervised) 및 강화 학습(reinforcement learning) 작업에 대한 실험 결과, 이 두 가지 정규화 기법을 결합하면 plasticity를 효과적으로 유지할 수 있음을 확인했습니다. 이러한 발견은 지속적 학습에서 신경망의 성능을 향상시키기 위한 새로운 접근 방식을 제시합니다.



### Pedestrian Attribute Recognition via Hierarchical Cross-Modality HyperGraph Learning (https://arxiv.org/abs/2509.22331)
Comments:
          The First Work that Exploits Multi-modal Knowledge Graph for Pedestrian Attribute Recognition

- **What's New**: 이 논문은 보행자 속성 인식(Pedestrian Attribute Recognition, PAR)의 정확성을 향상시키기 위해 다중 모달 지식 그래프(multi-modal knowledge graph)를 구축하는 새로운 방법을 제안합니다. 현재의 알고리즘들이 시각적 특징과 속성을 단순히 결합하는 데 그치고 있는 반면, 이 연구는 속성과 시각적 맥락 간의 관계를 탐구하여 보다 정교한 인식을 가능하게 합니다. 이를 통해 PAR에서의 지식 기반 인식의 가능성을 크게 높이고 있습니다.

- **Technical Details**: 논문에서는 지식 그래프 안내 크로스 모달 하이퍼그래프 학습(framework) 방법론을 도입하여 다중 모달 지식 그래프의 관계를 효과적으로 모델링합니다. M2PA-KG라는 이름의 새 지식 그래프는 보행자 몸체, 속성 및 각각의 속성의 언어 캡션과 시각적 샘플을 포함한 다양한 엔티티를 설정하며, 고차원 관계를 캡처하기 위해 하이퍼그래프를 사용합니다. 이러한 관계는 LA-UniGNN 및 AG-UniGNN로 인코딩되고, 최종적으로 multi-modal Transformer를 통해 비주얼-시맨틱 집합을 처리합니다.

- **Performance Highlights**: 다양한 PAR 벤치마크 데이터셋에서의 포괄적인 실험을 통해 제안된 M2PA-KG의 효과성을 검증하였으며, 이는 지식 기반 보행자 속성 인식의 중요한 기초를 마련합니다. 기존의 속성 인식 방법들에 비해 좋은 성능 향상을 보였고, 하이퍼그래프 모델링을 통해 고차원 관계를 효과적으로 활용하였습니다.



### Progressive Weight Loading: Accelerating Initial Inference and Gradually Boosting Performance on Resource-Constrained Environments (https://arxiv.org/abs/2509.22319)
- **What's New**: 이 논문에서는 모델 로딩 시간을 줄이면서 성능을 유지하기 위해 Progressive Weight Loading (PWL)이라는 새로운 방법을 제안합니다. PWL은 처음에 작은 student 모델을 배포하고, 이어서 pre-trained teacher 모델의 레이어를 점진적으로 교체함으로써 원활한 사이즈 적합성(compatibility)과 초기 추론 속도(inference speed)를 제공합니다. 이를 통해 모델을 로딩하는 데 필요한 시간을 줄이면서도 더 높은 정확도를 달성할 수 있습니다.

- **Technical Details**: PWL은 두 가지 주요 구성 요소를 포함합니다: Invertible Feature Converter와 Training Strategy for PWL. Invertible Feature Converter는 student와 teacher 모델의 내부 특징(feature) 표현을 정렬하는 데 필요한 변환기를 제안합니다. Training Strategy는 층(layer) 교체 과정에서 성능 저하를 방지하고, student 모델이 teacher의 최종 출력뿐만 아니라 중간 표현까지 모방하도록 돕습니다.

- **Performance Highlights**: 실험을 통해 VGG, ResNet, ViT 아키텍처에서 PWL로 훈련된 모델들이 경쟁력 있는 distillation 성능을 유지하면서 teacher 레이어가 로드됨에 따라 정확도가 점진적으로 향상됨을 확인했습니다. 최종적으로 PWL은 초기 추론 속도를 저하시키지 않고 전체 teacher 모델과 동일한 최종 정확도를 달성할 수 있습니다. 이러한 특성 덕분에 PWL은 자원 제약이 있는 모바일 및 지연 민감한 환경에서 특히 적합합니다.



### Adaptive Policy Backbone via Shared Network (https://arxiv.org/abs/2509.22310)
- **What's New**: 본 논문에서는 Adaptive Policy Backbone (APB)라는 메타-전이 강화 학습 방법을 제안합니다. APB는 공유된 backbone을 앞뒤로 가벼운 선형 층을 삽입하여 매개변수 효율적인 미세 조정(PEFT)을 가능하게 하며, 적응 중 이전 지식을 보존할 수 있습니다. APB는 기존의 메타-RL 방법들이 실패하는 OOD(Out-of-Distribution) 작업에서도 훌륭한 샘플 효율성을 제공합니다.

- **Technical Details**: APB는 메타-RL 패러다임에서 초기화 기반 접근법을 채택하여 메타-초기화를 학습하고 테스트 시 미세 조정을 통해 적응합니다. 이는 메타-훈련 작업을 통해 습득된 backbone의 매개변수를 유지하면서도 작업에 특화된 선형 층만을 업데이트하여 적응한다는 것을 의미합니다. 이 방법은 기존의 페어론 조정보다 OOD 작업 적응에 더욱 강력한 성능을 보여줍니다.

- **Performance Highlights**: APB는 표준 RL 알고리즘보다 높은 샘플 효율성을 제공하며, OOD 작업에서의 적응이 가능하다는 것을 실험적으로 입증하였습니다. 또한, 행동 클로닝 평가를 통해 OOD 작업에서의 일반화 능력도 확인하였습니다. 본 연구는 APB가 OOD 작업에 대한 일반화를 개선하는 메타 학습의 관점에서도 해석될 수 있음을 보여줍니다.



### HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models (https://arxiv.org/abs/2509.22300)
- **What's New**: 이번 논문에서는 이미지 생성을 위한 확산 모델(difusion models)의 품질 향상을 위해 역사 기반 샘플링 기법인 HiGS(이력 안내 샘플링)를 제안합니다. HiGS는 모델의 최근 예측값을 통합하여 샘플링 과정에서 더 사실적인 결과를 유도합니다. 이 접근법은 기존 확산 프레임워크에 쉽게 통합될 수 있으며, 추가 훈련(trainig)이나 튜닝(tuning) 없이 사용할 수 있습니다.

- **Technical Details**: HiGS는 현재의 예측값과 과거 예측들의 가중 평균(weighted average)의 차이를 활용하여 샘플링을 진행합니다. 이 기법은 추가적인 계산(computation)을 필요로 하지 않으며, 효율성을 높이는데 중점을 두고 있습니다. 실험 결과, 다양한 모델과 아키텍처에서 HiGS는 일관되게 이미지 품질을 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: HiGS는 사전 훈련된 SiT 모델을 사용하여, 가이드 없이 ImageNet 생성에서 새로운 최첨단 FID(Frechet Inception Distance)인 1.61을 달성하였습니다. 이는 표준 250단계의 샘플링 대신 30단계로 이루어진 것입니다. HiGS는 더 높은 충실도의 이미지 생성을 위해 빠른 생성 속도를 제공하는 플러그 앤 플레이(plug-and-play) 개선책으로 자리잡을 것입니다.



### HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Spac (https://arxiv.org/abs/2509.22299)
- **What's New**: 최근 Mixture-of-Experts (MoE) 아키텍처는 메모리 요구사항을 줄이면서 뛰어난 성능을 제공하는 대형 언어 모델의 대안으로 주목받고 있습니다. 본 논문은 HEAPr이라는 새로운 프루닝(Pruning) 알고리즘을 소개하여, 전문가를 더 작고 분리할 수 없는 원자 전문가(Atomic Expert)로 분해함으로써 더 정밀하고 유연한 원자 전문가 프루닝을 가능하게 합니다. 이를 통해 MoE 모델의 효율성과 배포 가능성이 크게 향상됩니다.

- **Technical Details**: HEAPr는 전문가의 파라미터에서 원자 전문가 파라미터로의 두 번째 순서 정보 추정 방법을 혁신적으로 활용하여, 원자 전문가 출력의 두 번째 순서 정보로 단순화함으로써 공간 복잡성을 $O(d^4)$에서 $O(d^2)$로 줄입니다. 이 알고리즘은 최소한의 계산에서 자신의 중요도를 평가하기 위해 두 번의 전방 패스(forward pass)와 한 번의 후방 패스(backward pass)만을 요구합니다. 이로 인해, 고전적인 Optimal Brain Surgeon 이론을 바탕으로 한 효율적이고 높은 성능의 원자 전문가 프루닝이 가능합니다.

- **Performance Highlights**: HEAPr은 DeepSeek MoE 및 Qwen MoE 모델에서 기존 전문가 수준의 프루닝 방법들을 능가하는 성과를 보였습니다. 특히, 대다수 모델에서 20%-25%의 압축 비율로 거의 무손실(compression) 압축을 달성하며, FLOPs를 약 20% 줄이는 데 성공했습니다. 혼합 모델에 대해 상당한 성능 유지와 함께 메모리 절약 효과를 입증하며, 작은 캐리브레이션 세트에서 이 방법의 효율성이 실험을 통해 확인되었습니다.



### Jailbreaking on Text-to-Video Models via Scene Splitting Strategy (https://arxiv.org/abs/2509.22292)
- **What's New**: 최근 Text-to-Video (T2V) 모델의 급속한 발전과 함께 이러한 모델의 안전성 위험에 대한 우려가 증가하고 있습니다. 기존의 연구들은 LLMs, VLMs, T2I 모델의 취약점을 jailbreak 공격을 통해 탐구했지만 T2V 모델은 거의 탐구되지 않았습니다. 이 논문에서는 SceneSplit이라는 새로운 블랙박스(jailbreak) 방법을 소개하며, 이 방법은 해로운 내러티브를 여러 장면으로 분할하여 각 장면이 개별적으로 무해하도록 합니다.

- **Technical Details**: SceneSplit은 생성(output) 공간을 조작하여 내러티브 구조를 악용하는 공격 방식을 제시합니다. 개별 장면은 대부분의 결과가 무해한 넓고 안전한 공간에 해당하지만, 이들을 순차적으로 결합함으로써 안전하지 않은 영역으로 좁혀집니다. 이 핵심 메커니즘은 반복적인 장면 조작을 통해 안전 필터를 우회하여 효율성을 높입니다.

- **Performance Highlights**: SceneSplit은 T2V 모델에서 11개의 안전 카테고리를 평가하여 공격 성공률(Attack Success Rate, ASR)에서 77.2% (Luma Ray2), 84.1% (Hailuo), 78.2% (Veo2)라는 높은 평균 값을 달성했습니다. 기존의 기준 대비 상당히 개선된 성과를 보이며, T2V 모델의 안전 메커니즘이 내러티브 구조를 이용한 공격에 취약하다는 새로운 통찰을 제공합니다.



### Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection? (https://arxiv.org/abs/2509.22291)
- **What's New**: 이번 연구는 증오 발언 탐지(hate speech detection)라는 중요한 응용 분야에서 설명 가능성(explainability)과 공정성(fairness) 간의 관계를 최초로 체계적으로 분석하였습니다. 연구는 세 가지 주요 질문을 통해 입력 기반 설명(input-based explanations)이 편향된 예측(biased predictions)을 식별하고, 공정한 모델을 자동으로 선택하며, 모델 훈련 동안 편향을 완화하는 데 어떻게 도움이 되는지를 탐구합니다.

- **Technical Details**: 연구에서는 입력 기반 설명이 모델의 예측에 대한 각 토큰의 기여를 나타내며, 이를 통해 모델의 행동을 더 명확하게 이해할 수 있도록 돕습니다. 특히, 본 논문은 이러한 설명 방법이 모델 학습 중 편향을 줄이는 데 유용하다고 밝혔으며, 다양한 설명 방법의 유효성을 비교하기 위한 정량적 분석을 실시했습니다.

- **Performance Highlights**: 연구 결과, 입력 기반 설명이 편향된 예측을 효과적으로 검출하고, 모델 훈련 중 편향을 줄이는 데 기여하는 것으로 나타났습니다. 그러나 이러한 설명이 후보 모델 중에서 공정한 모델을 자동으로 선택하는 데는 덜 신뢰할 수 있음을 보여주었습니다. 또한, 설명 기반의 편향 검출이 민감한 특성(sensitive features) 의존도를 줄이기 위해 훈련된 모델에서도 탄탄한 성능을 유지한다는 점이 강조되었습니다.



### Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities (https://arxiv.org/abs/2509.22287)
Comments:
          12 pages, 2 figures, Preprint of: Sundstedt, S., Wingren, M., Hägglund, S. & Ventus, D. (2025). Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities. In: Stephanidis, C., Antona, M., Ntoa, S. & Salvendy, G. (eds.), Communications in Computer and Information Science, vol. 2523, pp. 415-425. Springer

- **What's New**: 본 논문에서는 언어 취약성을 가진 유아들을 위한 새로운 접근 방식을 소개합니다. 특정 형태소(target morphological structures)를 게임을 통해 가르치는 대화형 로봇을 활용한 방법입니다. TalBot 프로젝트에서는 Furhat 로봇이 아이들과 함께 단어 찾기 게임을 진행하며, 언어 능력 향상을 도모합니다.

- **Technical Details**: 이 접근법은 수동적이지 않은 학습 방법인 implicit learning의 원칙에 기반하고 있습니다. 게임에서의 대화와 감정적 반응, 턴 테이킹(turn-taking)을 처리하기 위해 대규모 언어 모델(LLM)을 사용합니다. 이러한 시스템은 educators(교육자)와 SLTs(언어치료사)가 동시에 다양한 형태소를 사용하는 것을 도와줍니다.

- **Performance Highlights**: 로봇이 이러한 과제를 인간보다 잘 수행할 수 있을 것이라는 가설을 세우고 있습니다. 이는 로봇이 아이들과 전문가에게 모델 역할을 하고, 언어 취약성을 가진 아이들의 기본 의사소통 요구를 지원할 수 있음을 뜻합니다. 궁극적으로 LLM 기반 로봇 보조 언어 학습(intervention)을 개발하여 다양한 언어에서 형태소 구조를 가르치는 것을 목표로 하고 있습니다.



### A Global Analysis of Cyber Threats to the Energy Sector: "Currents of Conflict" from a Geopolitical Perspectiv (https://arxiv.org/abs/2509.22280)
Comments:
          THIS IS A POSTPRINT OF A PEER-REVIEWED ARTICLE, PLEASE CITE IT IF USING THIS WORK: Gustavo Sanchez, Ghada Elbez, and Veit Hagenmeyer. "A Global Analysis of Cyber Threats to the Energy Sector:"Currents of Conflict" from a geopolitical perspective." atp magazin 67.9 (2025): 56-66. this https URL

- **What's New**: 이 논문은 사이버 위협을 포괄적으로 이해할 필요성을 강조하며, 지정학적 동력학 및 사이버 위협 인텔리전스 분석의 교차점을 탐구합니다. 특히 에너지 분야를 중심으로 하여, 생성형 인공지능(Generative AI)을 활용해 사이버 위협 설명으로부터 정보를 추출하고 구조화합니다. 위협 행위자의 기원과 대상 지역에 대한 지정학적 비교를 통해 사이버 위협 경향을 파악할 수 있는 통찰을 제공합니다.

- **Technical Details**: 행위자 및 사이버 사건의 기원과 대상 지역에 대한 지정학적 분석을 진행하며, 이와 동시에 에너지 분야에 특화된 일반적 경향과 특정 경향을 비교합니다. 또한, 포괄적 데이터베이스를 활용하여 위협 정보의 깊은 분석을 가능케 하는 방법론을 제시합니다. AI 기반의 탐지 기술이 에너지 분야의 공격에서 피해 지표(Indicators of Compromise, IOCs)를 효과적으로 감지하는지 평가합니다.

- **Performance Highlights**: 기술적 성능 측면에서 AI 기반 탐지 시스템은 대규모 데이터를 실시간으로 분석하고 비정상적인 행동 패턴을 식별하는 데 우수한 성능을 보입니다. 그러나 AI 모델이 적대적 공격(Adversarial Attacks)에 취약하며, 학습 데이터의 편향성 문제를 해결해야 합니다. 따라서, AI와 인간 분석가 간의 협업 모델을 통해 이러한 문제를 해결하는 방향으로 연구의 필요성이 강조됩니다.



### Wavelet-Induced Rotary Encodings: RoPE Meets Graphs (https://arxiv.org/abs/2509.22259)
- **What's New**: WIRE(웨이브렛 유도 회전 인코딩)은 기존의 RoPE(회전 위치 인코딩)를 그래프 구조 데이터에 확장한 새로운 알고리즘입니다. 이 방법은 RoPE의 특별한 경우인 그리드 그래프에서 이를 회복할 수 있으며, 다양한 이론적 속성을 제공합니다.

- **Technical Details**: WIRE는 노드 순서 변환에 대한 동등성(equivariance)과 선형 주의(linear attention)와의 호환성 등의 속성을 유지합니다. 또한 특정 가정을 만족할 경우 그래프 저항 거리(resistive distance)에 대한 점근적(asymptotic) 의존성을 가지고 있습니다.

- **Performance Highlights**: WIRE는 다양한 합성 및 실제 작업에서 테스트되었으며, 단색 서브그래프(monochromatic subgraphs) 식별, 점 구름( point clouds) 의미 분할 및 표준 그래프 벤치마크를 포함합니다. 이 알고리즘은 기본 그래프 구조가 중요한 설정에서 효과적임을 보여주었습니다.



### Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks (https://arxiv.org/abs/2509.22258)
Comments:
          23 pages, 12 figures

- **What's New**: 최근 비전-언어 모델(VLM)의 발전은 의료 AI 작업에서 놀라운 개선을 가져왔으나, 실제 임상 추론 능력은 여전히 불명확합니다. 이에 따라 Neural-MedBench라는 새로운 벤치마크를 소개하며, 이는 다중모달(clinical reasoning)을 탐구하도록 설계되었습니다. Neural-MedBench는 다중 MRI 스캔, 전자 건강 기록, 임상 노트를 통합하여 차별 진단, 병변 인식, 논리 생성 등 세 가지 핵심 작업을 포함합니다.

- **Technical Details**: Neural-MedBench는 120개의 전문가 주석이 달린 다중모달 사례로 구성되며, 200개의 추론 집약적 작업을 생성합니다. 이 벤치마크는 의료 교육의 관행을 반영하여 디자인되었으며, 대상의 불확실성 아래에서 추론 능력을 평가할 수 있도록 설계되었습니다. 평가 과정에서는 LLM 기반의 평가자, 임상의 검증, 의미 유사성 메트릭을 결합하는 하이브리드 점수 시스템을 개발하여 신뢰할 수 있는 평가를 보장합니다.

- **Performance Highlights**: 최신 VLM 모델의 평가 결과, 대규모 벤치마크에서 우수한 성과를 내는 모델들이 Neural-MedBench에서는 체계적으로 실패하는 경향을 보였습니다. 오류 분석을 통해 이러한 실패는 인지적 오류가 아닌 임상 추론의 한계에서 기인함을 확인했습니다. 이러한 결과들은 깊이 있는 평가가 필요함을 강조하며, Neural-MedBench가 신뢰할 수 있는 임상 AI의 평가에 기여할 수 있음을 시사합니다.



### Secure and Efficient Access Control for Computer-Use Agents via Context Spac (https://arxiv.org/abs/2509.22256)
- **What's New**: 본 논문에서는 CSAgent라는 새로운 시스템 수준의 정적 정책 기반 접근 제어 프레임워크를 제안합니다. 이 프레임워크는 컴퓨터 사용 에이전트(CUA)의 행동을 제한하고, 사용자 의도와 컨텍스트에 따라 적절한 정책을 정의하여 안전성을 강화합니다. CSAgent는 API, CLI 및 GUI와 같은 다양한 인터페이스를 통한 에이전트 보호를 지원합니다.

- **Technical Details**: CSAgent는 정적 정책을 이용하여 에이전트의 행동을 제어하며, 각 정책은 특정 사용자 의도와 시스템 상태와 같은 컨텍스트에서만 기능이 안전하게 실행되도록 설정됩니다. 이는 Contextual Integrity (CI) 프레임워크를 기반으로 하며, 자동화된 도구 체인을 통해 개발자가 규칙을 구성하고 개선할 수 있도록 도와줍니다. 또한, LLM 기반의 의도 예측 기법을 통해 정책 생성 단계에서 사용자 의도를 예측합니다.

- **Performance Highlights**: 구현된 CSAgent는 각기 다른 세 가지 벤치마크(AgentBench, AgentDojo, AndroidWorld)를 통해 테스트되었으며, 99.36% 이상의 공격 차단률을 기록했습니다. 평균적으로 6.83%의 성능 오버헤드와 9.33%의 유틸리티 감소를 초래하는 것으로 평가되어, 기존 방법들보다 현저히 뛰어난 성능을 보입니다. 특히, LLM 기반의 컨텍스트 분석기가 다양한 사용자 의도를 정확하게 식별하는 데 기여함을 입증했습니다.



### Beyond Textual Context: Structural Graph Encoding with Adaptive Space Alignment to alleviate the hallucination of LLMs (https://arxiv.org/abs/2509.22251)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구에서는 SSKG-LLM이라는 새로운 모델 아키텍처를 제안하여 지식 그래프(KGs)의 구조적 및 의미적 정보를 효과적으로 통합하고 LLMs의 추론 과정에 활용할 수 있도록 하였습니다. 기존의 LLM 기술이 KGs를 단순 텍스트로 취급하며 중요한 구조적 측면을 활용하지 못하고 있다는 문제를 해결하고자 했습니다. SSKG-LLM은 Knowledge Graph Retrieval(KGR), Knowledge Graph Encoding(KGE), Knowledge Graph Adaptation(KGA) 모듈을 포함하여, LLMs가 KGs 임베딩을 이해할 수 있게 돕습니다.

- **Technical Details**: SSKG-LLM은 두 가지 핵심 도전 과제를 해결합니다. 첫째로, KGs의 구조적 및 의미적 정보를 어떻게 획득하고 통합할 것인지에 대한 문제입니다. 이를 위해, GraphLM이라는 새로운 사전 훈련 모델을 사용하여 KGs의 하위 그래프를 인코딩하고 구조와 의미의 뉘앙스를 보존합니다. 둘째로, KGs 인코딩과 LLMs 간의 간극을 어떻게 메꿀 것인지에 대한 문제입니다. 이를 해결하기 위해, cross-attention을 활용한 KG-Adapter 모듈을 제안하여 그래프와 텍스트 인코딩을 서로 조정할 수 있도록 하였습니다.

- **Performance Highlights**: 우리의 방법은 Multiple types of Question Answering(QA) 데이터셋을 통해 기존 모델들을 능가하는 성과를 보였습니다. 특히 LLMs 기반의 QA 작업에서 KGs 통합에 있어 큰 성과를 나타내며, 구조적 정보의 중요성을 강조합니다. 우리의 실험 결과는 SSKG-LLM이 KGs의 정보와 LLMs 간의 통합을 더 효과적으로 수행하여 더 정확한 답변을 생성할 수 있음을 보여줍니다.



### Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Complianc (https://arxiv.org/abs/2509.22250)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 안전성을 법적 준수 관점에서 접근하여 새로운 방법론, 즉 '안전 준수'를 제안합니다. 기존의 안전 방법론과 달리, 유럽 연합 인공지능법(EU AI Act)과 일반 데이터 보호 규정(GDPR) 등의 법적 프레임워크를 안전 기준으로 활용합니다. 이러한 접근법은 LLM의 복잡한 행동을 보다 시스템적이고 철저하게 보호하는 기반을 제공합니다.

- **Technical Details**: 연구진은 법적 조항을 사용하여 현실적인 LLM 안전 시나리오를 생성하는 새로운 벤치마크를 개발하였습니다. 이를 통해 LLM의 성능을 평가하고, Qwen3-8B 모델을 Group Policy Optimization (GRPO) 기법을 사용해 정합성 추론기인 Compliance Reasoner를 구축하였습니다. 이 모델은 LLM을 법적 기준에 맞춰 조정하여 안전 위험을 완화하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, Compliance Reasoner는 새로운 벤치마크에서 EU AI Act에 대해 +10.45%, GDPR에 대해 +11.85%의 평균 개선률을 보이며 우수한 성능을 나타냈습니다. 또한, 기존의 안전 데이터를 준수 시나리오로 확장하여 안전 준수를 위한 데이터의 양을 크게 증가시키는 일반화 가능 방법을 제공합니다.



### ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity (https://arxiv.org/abs/2509.22246)
- **What's New**: 본 논문에서는 ASSESS(A Semantic and Structural Evaluation Framework for Statement Similarity)를 제안합니다. ASSESS는 자연어에서 형식 언어로의 자동 변환인 autoformalization의 평가를 위한 새로운 접근 방식을 제공합니다. 이 프레임워크는 semantic(의미론적)과 structural(구조적) 정보를 통합하여 연속적인 유사도 점수를 제공하는 것을 목표로 합니다.

- **Technical Details**: ASSESS의 핵심은 TransTED(Transformation Tree Edit Distance) Similarity 지표로, 이는 문장의 유사성을 평가하는 데 필요한 semantic(의미론적) 및 structural(구조적) 뉘앙스를 포착하도록 설계되었습니다. ASSESS는 형식 문장을 Operator Tree로 변환하고, 이를 기반으로 TransTED 지표를 이용해 유사도 점수를 계산합니다. 이 메트릭은 기계적인 처리로만 작동하며 CPU 자원만을 요구합니다.

- **Performance Highlights**: 최신 EPLA(테스트 데이터 세트)에서의 실험 결과는 TransTED Similarity가 기존 방법보다 뛰어난 성과를 보였음을 보여줍니다. TransTED는 정확도 78.82%와 70.86%, Cohen's Kappa 계수 0.46와 0.40을 기록하며 현재 최고의 성과를 달성했습니다. 이 연구는 semantic(의미론적) 및 structural(구조적) 정보를 효과적으로 결합한 자동화된 평가 메트릭의 필요성을 강조합니다.



### FeatBench: Evaluating Coding Agents on Feature Implementation for Vibe Coding (https://arxiv.org/abs/2509.22237)
- **What's New**: 이번 논문은 새롭게 등장한 "vibe coding" 패러다임을 다루며, 이를 평가하기 위한 새로운 벤치마크인 FeatBench를 제안합니다. 기존 코드 생성 평가 기준들이 vibe coding의 요구사항을 충족하지 못하는 문제를 지적하며, 이 틈새를 메우기 위한 다양한 기능 구현에 중점을 둡니다. FeatBench는 사용자 관점에서 자연어로 설명된 기능을 구현하는 과제를 평가하게 설계되었습니다.

- **Technical Details**: FeatBench는 전적으로 자연어 프롬프트를 사용하며, 코드나 구조적 힌트가 포함되지 않은 단순한 설명만을 기반으로 합니다. 데이터 수집 프로세스는 정확성과 품질을 보장하기 위해 엄격하게 구성되어 있으며, 자동화된 파이프라인을 통해 지속적으로 진화합니다. 또한 각 과제는 Fail-to-Pass (F2P)와 Pass-to-Pass (P2P) 테스트 케이스를 포함하여 정확성과 기존 기능의 보존을 검증합니다.

- **Performance Highlights**: 주요 테스트 결과에 따르면 FeatBench는 기존 SOTA(최첨단) 에이전트 frameworks에 상당한 도전을 제공합니다. 평가된 에이전트들은 새로운 기능을 추가할 때 기존 기능을 손상시키는 경향이 있으며, 최고 성공률은 29.94%로 나타났습니다. 특이하게도, 에이전트들은 "aggressive implementation" 전략을 취하는데, 이는 적절한 소프트웨어 아키텍처를 제공하는 동시에 작업 실패를 초래하는 경향이 있습니다.



### Fairness-Aware Reinforcement Learning (FAReL): A Framework for Transparent and Balanced Sequential Decision-Making (https://arxiv.org/abs/2509.22232)
- **What's New**: 이번 논문에서는 성과와 공정성을 동시에 고려할 수 있는 새로운 프레임워크를 제안합니다. 이 프레임워크는 강화학습( reinforcement learning ) 알고리즘을 통해 다양한 성과-공정성 트레이드오프를 탐색할 수 있게 하여 의사결정자들이 가장 적절한 정책을 선택할 수 있도록 돕습니다. 또, 개인 및 그룹을 명시적으로 인코딩한 확장된 마르코프 결정 프로세스인 $f$MDP를 도입하여 공정성을 정의하고, 직무 채용 및 사기 탐지와 같은 다양한 시나리오를 평가합니다.

- **Technical Details**: 제안된 프레임워크는 여러 공정성 개념을 동시에 고려하며, 다목적 강화학습( multi-objective reinforcement learning )에 기초하여 성과-공정성 트레이드오프를 학습합니다. 이 프레임워크는 시간에 따라 공정성 지표를 계산하고, 직무 채용에서의 성별 공정성 및 사기 탐지에서의 공정함을 요구하는 두 가지 설정에서 평가됩니다. 또한 공정성 개념은 정책 학습 과정에서 서로 다른 시나리오에서의 적용 가능성에 맞게 정의됩니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 시나리오에서 더 공정한 정책을 학습하는 동시에 성과 보상의 손실은 최소화함을 보여줍니다. 그룹 공정성과 개인 공정성 개념 간의 상호 작용을 분석하며 이들 두 개념이 항상 서로를 의미하지는 않음을 강조합니다. 또한, 기존 접근 방식과는 달리 본 프레임워크는 다양한 공정성 유형을 수용할 수 있는 방법을 제시하여, 복잡한 현실 문제에서도 적절한 성과-공정성 트레이드오프를 결정하는 데 기여할 수 있습니다.



### Polysemous Language Gaussian Splatting via Matching-based Mask Lifting (https://arxiv.org/abs/2509.22225)
- **What's New**: MUSplat은 3D Gaussian Splatting(3DGS) 장면의 오픈-보카불러리(open-vocabulary) 이해를 위한 새롭고 훈련이 필요 없는 프레임워크입니다. 기존의 복잡한 훈련과 최적화 과정 없이 다중 해상도의 2D 마스크를 3D로 변환하여 객체 그룹을 형성합니다. 이 방법은 관점에 따라 객체의 모습 해석을 통해 텍스트 기능을 정제하여 언어 기반 검색을 가능한 직관적으로 수행합니다.

- **Technical Details**: 이 프레임워크는 각 Gaussian의 의미를 매칭 메커니즘으로 결정하며, 다의적(polysemous) 표현을 지원하기 위해 설계되었습니다. 초기의 객체 그룹 경계는 중립점 처리(neutral point processing)를 통해 개선되며, Vision-Language Model(VLM)을 활용하여 각 객체에 대한 강력한 텍스트적 표현을 생성합니다. 이를 통해 각 Gaussian point의 전경 확률을 추정하여 초기 객체 그룹을 형성합니다.

- **Performance Highlights**: MUSplat은 기존의 훈련 기반 프레임워크들보다 적응 시간이 수 시간에서 수 분으로 단축되는 장점을 가지고 있습니다. 오픈-보카불러리 3D 객체 선택 및 의미 없는 분할의 기준 작업에서 성공적으로 규정된 방식으로 성능이 우수함을 입증하였습니다. 이 모델은 특히 복잡한 의미 표현에서 기존의 한정된 방식으로부터 발생하는 한계를 해결하고 있습니다.



### Thinking in Many Modes: How Composite Reasoning Elevates Large Language Model Performance with Limited Data (https://arxiv.org/abs/2509.22224)
Comments:
          7 pages, 3 figures

- **What's New**: 본 논문에서는 Composite Reasoning (CR)이라는 새로운 추론 방식을 소개합니다. 기존의 대형 언어 모델(LLMs)은 단일 추론 패러다임에 의존했으나, CR은 다양한 추론 스타일(예: deductive, inductive, abductive)을 동적으로 결합하여 더 정교한 문제 해결을 가능하게 합니다. 이 방법은 의학과 과학 질문 응답 기준에서 기존 방법들보다 향상된 성능을 보였습니다.

- **Technical Details**: CR 접근법에서는 모델이 여러 추론 전략을 탐색하고 통합하도록 유도합니다. 이 과정은 Low-Rank Adaptation (LoRA)와 Group Relative Policy Optimization (GRPO)와 같은 파라미터 효율적인 미세 조정 기법을 사용하여 이루어집니다. 이를 통해, 모델은 문제 해결의 다양한 경로와 관점을 고려하여 보다 정확하고 잘 뒷받침된 답변을 생성하도록 돕습니다.

- **Performance Highlights**: CR은 세 가지 도전적인 데이터셋에서 테스트되었으며, 기존의 Chain-of-Thought (CoT) 및 Standard Reasoning (SR)과 비교하여 탁월한 성과를 보였습니다. 특히, CR SFT + GRPO 조합은 ARC-Complex와 MedMCQA에서 각각 94.99%, 56.30%의 정확도를 기록하며 가장 높은 성과를 달성했습니다. 이는 리소스 제한 환경에서도 CR이 더 효과적인 추론 경로를 탐색할 수 있다는 점을 보여줍니다.



### Rigidity-Aware 3D Gaussian Deformation from a Single Imag (https://arxiv.org/abs/2509.22222)
Comments:
          10 pages, 11 figures, conference

- **What's New**: 이번 연구에서는 단일 이미지에서 3D 물체 변형을 복구하는 새로운 프레임워크인 DeformSplat을 제안합니다. 기존 방법들이 다수의 시점을 요구하는 것과 달리, 본 연구는 단일 RGB 이미지만으로도 3D Gaussian 변형이 가능하다는 점에서 큰 의의가 있습니다. 특히 Gaussian-to-Pixel Matching과 Rigid Part Segmentation이라는 두 가지 기술적 기여를 통해 이미지에서 직접적으로 변형 가이드를 제공합니다.

- **Technical Details**: DeformSplat의 첫 번째 기법인 Gaussian-to-Pixel Matching은 3D Gaussian을 2D 픽셀 관측과 연결하여 변형의 원활한 유도를 가능하게 합니다. 두 번째 기법인 Rigid Part Segmentation은 변형 동안 기하학적 일관성을 유지하기 위해 강체(rigid) 지역을 명시적으로 식별합니다. 이를 통해 변형 프로세스에서의 오버피팅을 방지하고, 원래 구조를 유지하는데 기여합니다.

- **Performance Highlights**: 이 연구를 통해 DeformSplat는 기존의 최첨단 방법들과 비교해 우수한 성능을 보이며, 단일 이미지에서 3D Gaussian 변형을 성공적으로 수행할 수 있음을 입증하였습니다. 또한, 해당 프레임워크는 프레임 보간(frame interpolation) 및 상호작용 객체 조작(interactive object manipulation)과 같은 다양한 응용 프로그램으로의 확장성도 가지고 있습니다.



### Automatic Discovery of One Parameter Subgroups of $SO(n)$ (https://arxiv.org/abs/2509.22219)
- **What's New**: 이번 논문에서는 $SO(3)$ 및 일반적으로는 $SO(n)$의 단일 매개변수 (one-parameter) 군 (subgroup) 자동 발견을 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 Robotics, Quantum Mechanics, 분자 구조 분석 등 다양한 응용 분야에 중요하며, skew-symmetric matrices의 일반적인 Jordan 형식(standard Jordan form)을 활용하여 $H_{eta}$에 대한 궤도의 표준 표현을 도출합니다. 이를 통해 $H_{eta}$ 불변 함수(invariant functions)에 대한 표준화된 표현을 개발합니다.

- **Technical Details**: 제안된 방법론은 $H_{eta}$의 작용 아래에서 궤도 구조를 분석하고, $H_{eta}$의 불변 함수에 대한 표준 형태를 구축하는 데 중점을 둡니다. 특히, 무관심한 감독(symmetry supervision) 없이도 대칭 (symmetry) 발견과 학습을 자연스럽게 통합하는 방법론을 제공하여, 신경망이 데이터로부터 구조를 발견하고 활용할 수 있도록 합니다. 이 연구는 $SO(n)$의 단일 매개변수 군을 자동으로 발견하는 기반을 마련하여, 고차원 Lie 서브군(higher-dimensional Lie subgroups) 이해를 지원합니다.

- **Performance Highlights**: 향상된 프레임워크는 더블 진자 모델링(double pendulum modeling), 관성 모멘트 예측(moment of inertia prediction), top 쿼크 태깅(top quark tagging), 불변 다항 회귀(invariant polynomial regression) 등의 다양한 작업을 통해 효과성을 입증합니다. 이러한 작업들을 통해 의미 있는 군 구조(subgroup structure)를 회복하고, 해석 가능한 대칭 인지 표현(symmetry-aware representations)을 생성하는 데 성공하였습니다. 이는 초구적 및 실제 설정에 모두 적용 가능하며, 과학 컴퓨팅에서의 대칭 인지 학습(symmetry-aware learning)의 가능성을 확대하는 데 기여합니다.



### VizGen: Data Exploration and Visualization from Natural Language via a Multi-Agent AI Architectur (https://arxiv.org/abs/2509.22218)
- **What's New**: VizGen은 AI의 도움으로 그래프를 생성하는 시스템으로, 사용자들이 자연어를 사용하여 의미 있는 시각화를 만들 수 있도록 합니다. 기존의 데이터 시각화 도구에 비해 더 많은 사용자에게 접근 가능해진 점이 주목할 만합니다. 이는 전문 지식이 없는 사용자들도 복잡한 데이터셋을 쉽게 해석할 수 있게 합니다.

- **Technical Details**: NLP(자연어 처리)와 LLMs(대규모 언어 모델)인 Claude 3.7 Sonnet 및 Gemini 2.0 Flash를 활용하여 사용자 쿼리를 SQL로 변환하고 적절한 그래프 유형을 추천합니다. 또한, 다중 에이전트 아키텍처를 통해 SQL 생성, 그래프 생성 및 사용자 맞춤 설정, 통찰력 추출 등을 처리합니다.

- **Performance Highlights**: VizGen은 데이터 분석을 직관적이고 접근 가능하게 만들어 주며, 실시간으로 SQL 데이터베이스와 상호작용할 수 있도록 지원합니다. 이를 통해 사용자는 대화식으로 그래프를 다듬을 수 있으며, 데이터 내의 패턴, 이상 현상, 상관관계를 분석하여 더욱 풍부한 이해를 제공합니다.



### Impact of Collective Behaviors of Autonomous Vehicles on Urban Traffic Dynamics: A Multi-Agent Reinforcement Learning Approach (https://arxiv.org/abs/2509.22216)
Comments:
          Work presented at the European Workshop on Reinforcement Learning (EWRL 2024)

- **What's New**: 이 연구는 RL(강화 학습) 기반의 자율주행 차량(AV)이 혼합 교통 환경에서 도시 교통 흐름에 미치는 잠재적 영향을 조사합니다. 주로 다중 대리인 설정에서 일상적인 경로 선택 문제를 단순화하여 다룹니다. 인구의 1/3을 AV로 변환하여 딥 Q-러닝 알고리즘을 사용하는 RL 에이전트로 설정하고, 여러 행동(이기적, 협력적, 경쟁적, 사회적, 이타적, 악의적)을 정의하여 이들을 보상으로 통제합니다.

- **Technical Details**: 제안된 모델은 AV의 행동에 따라 여행 시간을 5%까지 최적화하는 결과를 보여주며, 이는 AV 행동에 따라 인간 운전자의 여행 시간에 상이한 영향을 미칠 수 있습니다. 또한, 다중 대리인 RL(MARL) 구조를 채택하여 자율주행 차량과 인간 운전자의 상호작용을 시뮬레이션할 수 있습니다. 연구는 Csömör의 교통 네트워크를 사용하여 고유의 경로 선택 문제를 정의하고, PARCOUR라는 RL 프레임워크로 실험을 진행합니다.

- **Performance Highlights**: 시뮬레이션 결과, 이기적인 행동을 채택한 AV가 평균적으로 인간 운전자의 여행 시간보다 짧은 시간을 기록했습니다. 이러한 결과는 AV의 다양한 행동이 교통 흐름과 혼잡도에 미치는 영향을 분석하는 데 기여하며, 자율주행 기술의 완전한 통합이 여러 이해관계자에게 미치는 다양한 결과를 이해하는 데 도움을 줍니다.



### Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation (https://arxiv.org/abs/2509.22211)
- **What's New**: 이번 논문에서는 Recursive Thematic Partitioning (RTP)이라는 새로운 프레임워크를 소개합니다. RTP는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 데이터 세트를 상호작용적으로 분석하는 방법을 제시합니다. 이 프레임워크는 데이터를 의미적으로 나누는 자연어 질문을 생성하며, 이를 통해 해석 가능한 주제 계층 구조를 구축합니다.

- **Technical Details**: RTP는 이진 트리를 구축하는 과정에서 LLM에 의한 질문 생성을 통해 문서를 재귀적으로 분할합니다. 각 트리 노드는 '이 리뷰는 주로 음식 품질에 대한 것인가, 고객 서비스에 대한 것인가?'와 같은 분할 질문을 포함합니다. 이 과정에서 데이터 세트를 효율적으로 다루기 위해 전처리 단계로 글로벌 샘플링과 문서 요약을 수행하여 전체 데이터베이스에서 대표 샘플을 뽑아냅니다.

- **Performance Highlights**: RTP의 실험 결과는 다른 주제 모델인 BERTopic과 비교했을 때 해석 용이성이 높고, 데이터 세트 내에서 액션 가능한 정보와의 연결성이 뛰어남을 보여줍니다. 또한, 생성된 군집이 다운스트림 분류 작업의 효과적인 특징으로 작용함을 입증하였으며, RTP 트리의 경로를 제어 가능한 주제 생성에 활용할 수 있다는 점도 강조합니다. 이는 텍스트 생성에서 높은 수준의 제어력을 제공하여 특정 특성을 일관되게 재현할 수 있도록 돕습니다.



### Reversible GNS for Dissipative Fluids with Consistent Bidirectional Dynamics (https://arxiv.org/abs/2509.22207)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구에서는 사용자 정의 목표를 향한 물리적으로 그럴듯한 궤적을 시뮬레이션하는 Reversible Graph Network Simulator (R-GNS)를 소개합니다. 이 시스템은 단일 그래프 아키텍처 내에서 양방향 일관성을 유지하는 통합된 프레임워크로, 역동력을 반영하려 하지 않고도 효과적으로 초기 상태를 회복해냅니다. R-GNS는 기존의 비선형 유도 불가능한 역학과 차별화되며, 물리적 일관성을 보장하면서도 높은 정확도를 보여줍니다.

- **Technical Details**: R-GNS는 잔여 가역 메시지 전파(residual reversible message passing) 디자인을 기반으로 하여 입자 상호작용을 양 방향으로 전파합니다. 이 모델은 공통의 매개변수를 공유하고 양방향 학습 방식을 활용하여 직관적으로 전진 동역학(forward dynamics)과 역 추론(inverse inference)을 연결하며, 프로젝트의 일관성을 높입니다. 이 기법은 물리적 요소가 복잡하게 얽혀 있는 입자 기반(fluids) 시뮬레이션에서 유연하고 효율적인 성능을 발휘합니다.

- **Performance Highlights**: R-GNS는 Water-3D, WaterRamps, WaterDrop의 세 가지 주요 훈련 벤치마크에서 기존 최첨단 모델보다 높은 정확도와 일관성을 나타냅니다. 또한, 목표에 맞춘 캐릭터 형성 작업(goal-conditioned tasks)에서는 반복 최적화를 제거하여 속도 측면에서 수십 배의 성능 향상을 이루었습니다. 최종적으로, R-GNS는 물리적으로 그럴듯한 궤적을 생성 및 각종 복잡한 목표 형태를 매칭하는 능력을 통해 그 효용성을 입증하였습니다.



### The Outputs of Large Language Models are Meaningless (https://arxiv.org/abs/2509.22206)
Comments:
          24 pages, 2 figures, forthcoming in Herman Cappelen and Rachel Sterken, eds. Communicating with AI: Philosophical Perspectives. Oxford: Oxford University Press

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 출력이 무의미하다는 결론에 대한 간단한 주장을 제시합니다. 이 주장은 두 가지 핵심 전제에 기반하고 있습니다. 첫째, LLM의 출력이 문자 그대로의 의미를 갖기 위해 필요한 특정 종류의 의도가 필요하다는 점과, 둘째, LLM이 그러한 의도를 가질 수 없다는 점입니다.

- **Technical Details**: 논문은 LLM의 출력이 의미를 지니기 위해 필요한 의도와 관련된 두 가지 전제에 대해 방어하고 있습니다. 저자들은 의미 외부주의자(semantic externalist)와 의미 내부주의자(semantic internalist) 등의 반응에 대해 이 주장을 방어합니다. 특히, 개념적 역할(conceptual roles)과 같은 개념 간의 내재적 관계를 통해 의미를 정의할 수 있다는 주장에 반대합니다.

- **Performance Highlights**: 그럼에도 불구하고 논문은 LLM의 출력이 어떻게 의미 있는 것처럼 보이고, 이를 통해 진실한 믿음(true beliefs)과 지식(knowledge)을 얻을 수 있는지에 대한 논의로 결론을 맺습니다. 이러한 점은 LLM의 출력이 실제로는 무의미하지만, 사용되는 맥락에서 의미 있는 것처럼 인식될 수 있다는 것을 시사합니다.



### MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training (https://arxiv.org/abs/2509.22199)
- **What's New**: 이 연구는 MimicDreamer라는 새로운 프레임워크를 제안하여, 비용 효율적인 인간 시연을 로봇에서 활용 가능한 지도(supervision)로 변환합니다. 이를 통해 시각(vision), 시점(viewpoint), 행동(actions)을 정렬하여 VLA(vision-language-action) 모델의 훈련을 지원합니다. 특히, H2R Aligner와 EgoStabilizer와 같은 혁신적인 모델을 도입하여 인간의 행동 데이터를 로봇과 효과적으로 연계할 수 있습니다.

- **Technical Details**: MimicDreamer는 인간 조작 영상에서 로봇 구현 비디오를 생성하는 H2R Aligner와 같은 비디오 확산 모델을 기반으로 하여, 고충실도의 로봇 비디오를 생성합니다. EgoStabilizer는 동작을 제어하기 위해 'homography'를 이용한 변형을 통해 인간 관점의 영상을 안정화합니다. 또한, 인간의 손 궤적을 로봇 프레임으로 매핑하고 제한된 역운동학(solver)기를 적용하여 실제 작업이 가능하고 저진폭(jitter) 명령을 생성합니다.

- **Performance Highlights**: MimicDreamer에서 합성된 인간-로봇 비디오로 훈련된 VLA 모델은 실제 로봇에서 불과 몇 번의 실행으로 작업을 수행할 수 있습니다. 이 모델은 여섯 가지 조작 작업에서 평균 성공률을 기존 로봇 데이터 기반보다 14.7% 향상시켰으며, 이는 견고한 일반화와 높은 샘플 효율성을 입증합니다.



### Learning Equivariant Functions via Quadratic Forms (https://arxiv.org/abs/2509.22184)
- **What's New**: 이 연구에서는 데이터로부터 그룹(알려진 또는 알려지지 않은) 동형 함수를 학습하기 위해 관련된 이차 형태인 $x^T A x$를 학습하는 방법을 제안합니다. 우리는 특정 그룹인 직교 그룹이 특정 이차 형태를 보존하는 성질을 활용하여, 직교적인 기본 대칭 그룹을 찾아냅니다. 이 연구는 자연스럽게 신경망 아키텍처에 적합한 유도 편향을 통합하여 간소화되고 효율적인 모델을 생성하는 방식으로 진행됩니다. 이를 통해 우리는 다양한 작업에 대해 효과적인 결과를 검증합니다.

- **Technical Details**: 제안된 방법은 직교 그룹에 중점을 두고 있으며, 유클리드 노름을 보존하는 표준 직교 그룹 O(n)의 기본 특성을 활용합니다. 우리는 직교 그룹이 보존하는 임의의 이차 형태에 대한 모델링을 진행하고, 네트워크 아키텍처에서 이러한 대칭성을 효과적으로 통합할 수 있는 체계적인 프레임워크를 구축합니다. 추가로, 우리는 Lorentz 그룹을 탐구하여 물리학에서 시공간 구조의 기본 대칭을 나타내는 방식을 효율적으로 다룹니다. 이 과정에서 순서 벡터에서만 추출된 각 성분과 전체 Gram 매트릭스에 따라 의존하는 크기 불변 성분의 분해를 제시합니다.

- **Performance Highlights**: 우리는 제안한 방법이 다수의 작업에서 어떻게 유용한지를 평가하였으며, 특히 다항 회귀, top 쿼크 태깅, 관성 모멘트 매트릭스 예측 작업에서 눈에 띄는 성과를 보였습니다. 해당 연구는 기존 방법들과 비교했을 때 모델이 기본 대칭을 발견하고 상응하는 동형 함수를 학습하는 데 일관되게 우수함을 입증했습니다. 이로 인해 고급 대칭성과 효율성이 결합된 접근법이 각종 데이터셋에서 효과적으로 활용될 수 있음을 나타냅니다.



### Efficiency Boost in Decentralized Optimization: Reimagining Neighborhood Aggregation with Minimal Overhead (https://arxiv.org/abs/2509.22174)
- **What's New**: DYNAWEIGHT는 다중 에이전트 네트워크에서 정보 집계를 위한 새로운 프레임워크로, 비중정량화(weight assignment) 방식을 동적(dynamic)으로 바꿉니다. 기존의 정적 메트로폴리스 가중치와 달리, DYNAWEIGHT는 이웃 서버의 상대적인 손실에 기반하여 가중치를 동적으로 할당합니다. 이는 다양한 데이터 이질성(data heterogeneity) 상황에서 정보의 다양성을 보장하는 데 유리합니다.

- **Technical Details**: DYNAWEIGHT 프레임워크는 N개의 상호 연결된 서버들이 서로의 파라미터를 교환하며 협력적으로 학습하는 분산(decentralized) 구조에서 작동합니다. 각 서버는 비공유(local) 데이터 세트를 가지고 있으며, 비중정량화에 의해 이웃 서버와 협력하여 파라미터를 업데이트합니다. DYNAWEIGHT는 이 과정에서 데이터 보안을 유지하면서도 더 빠른 수렴(convergence)을 이끌어내며, 다양한 최적화 알고리즘과 통합이 가능하다는 특징이 있습니다.

- **Performance Highlights**: 다양한 데이터셋인 MNIST, CIFAR10, CIFAR100을 통해 DYNAWEIGHT의 성능을 실험한 결과, 균형 잡힌 가중치 할당을 통한 학습 속도의 현저한 향상을 입증하였습니다. 실험에서는 DYNAWEIGHT가 전통적인 비적응형(non-adaptive) 가중치 체계와 비교했을 때 빠른 수렴속도를 보이며, 계산 효율성 또한 유지된다는 점이 강조됩니다.



### Teaching AI to Feel: A Collaborative, Full-Body Exploration of Emotive Communication (https://arxiv.org/abs/2509.22168)
Comments:
          9 pages, 10 Figures, ACM MM'25

- **What's New**: Commonaiverse는 사람의 감정을 탐구하는 인터랙티브 설치물로, 전신 모션 트래킹(full-body motion tracking)과 실시간 AI 피드백을 통해 참가자들이 감정을 공동으로 표현하고 해석할 수 있도록 하고 있습니다. 참가자는 'Teaching', 'Exploration', 'Cosmos Phase'의 세 가지 단계에서 시스템과 함께 감정을 소통하며 새로운 길을 제시하고, 포용적이고 윤리적인 감정 컴퓨팅(affective computing)을 강조합니다. 기존의 정량적 감정 분류법을 넘어선 이 혁신적인 접근 방식은 사용자의 주체성을 높이고 편견을 줄이며, 고급 인터랙티브 응용 프로그램으로 향하는 길을 열고 있습니다.

- **Technical Details**: Commonaiverse의 설계는 인간 감정의 신체적 표현과 상호작용을 강화하는 데 초점을 맞추고 있습니다. 이 시스템은 최소한 두 명의 참가자가 필요하며, 이는 감정 소통의 본질적으로 사회적이고 상호 의존적인 특성을 반영합니다. 참가자들은 자기 감정을 전문가들이 미리 정해 놓은 데이터 세트가 아닌, AI와 공동으로 창조하여 모션을 통해 표현할 수 있으며, 이는 그들의 신체 동작과 실시간으로 상관관계를 통해 이루어집니다.

- **Performance Highlights**: Commonaiverse의 인터랙션은 세 가지 단계로 구성되며, 전체 경험은 약 15~20분 동안 진행됩니다. 각 단계는 약간의 시간이 다르며, 참가자들이 시스템과 서로의 감정적 상호작용을 풍부하게 경험할 수 있도록 설계되었습니다.  Teaching Phase에서는 참가자들이 전신의 동작을 통해 특정 감정을 표현하고, Exploration Phase에서는 AI가 학습한 패턴에 따라 자유로운 움직임을 해석합니다. 마지막으로 Emotional Cosmos Phase에서는 디지털 우주를 통해 세션 동안의 감정의 교환을 시각화하여 제공합니다.



### Lightweight error mitigation strategies for post-training N:M activation sparsity in LLMs (https://arxiv.org/abs/2509.22166)
- **What's New**: 최근 연구의 필요성이 증가하면서 LLM(대형 언어 모델)의 효율적인 추론을 위한 희소화(sparsification) 기술에 많은 주목을 받고 있습니다. 특히, N:M 구조의 활성화 프루닝(activation pruning) 기술은 동적이면서 입력에 적응할 수 있는 압축 방법으로 최적화된 성능을 보여주고 있습니다. 본 논문에서는 다양한 LLM을 대상으로 한 사후 훈련(post-training) N:M 활성화 프루닝 방법을 포괄적으로 분석하여, 활성화 프루닝이 동일한 희소성 수준에서 가중치 프루닝보다 더 우수한 생성 능력을 유지함을 확인했습니다.

- **Technical Details**: LLM의 효율성을 높이기 위해 두 가지 주요 방법인 계산량 감소와 I/O 트래픽 감소가 중요한 역할을 합니다. 특히, 활성화의 희소성(dynamically sparse activations)은 입력에 따라 모델 용량을 보존할 수 있어 효과적인 방법입니다. 이 연구는 N:M 희소성 패턴이 활성화 프루닝에 적용될 때 성능 저하를 최소화하는 선택 기준과 에러 완화 방법을 소개하고, 특히 8:16 패턴이 유망한 후보로 작용할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구는 Llama2-7B, Llama3-8.1B, Qwen2.5-7B 및 Gemma3-4B와 같은 네 가지 서로 다른 LLM에서 활성화 프루닝이 가중치 프루닝보다 일관되게 우수한 성능을 발휘함을 입증하였습니다. 새로운 희소성 패턴인 2:4, 4:8, 8:16, 16:32를 평가한 결과, 8:16 패턴이 2:4보다 2배 향상된 성능을 보여줍니다. 이러한 연구 결과는 향후 하드웨어 개발에 있어 더 유연한 희소성 패턴을 지원하는 데 기여할 것입니다.



### Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization (https://arxiv.org/abs/2509.22161)
- **What's New**: 본 논문에서는 벡터 양자화(Vector Quantization)의 비미분성(non-differentiability) 문제를 해결하기 위해 초기화된 방법을 제안합니다. 이 연구는 기존의 벡터 양자화 방법과는 다르게, 간단하고 직관적인 정규화(regularization) 방식을 도입하여 벡터 양자화를 개선합니다. 특히, K-최근접 이웃(KNN)과의 거리 최적화를 통해 코드북(codebook)의 전부를 활용할 수 있도록 합니다.

- **Technical Details**: 벡터 양자화는 연속 벡터 공간을 유한한 대표 벡터 집합으로 변환하는 방법으로, 코드북 내의 각 벡터에 대해 비미분적인 양자화(quantization) 단계를 적용합니다. 본 논문에서는 코드를 저하시키지 않고, 모든 코드북 항목을 사용할 수 있도록 하기 위해, 각 심플렉스 정점과 최근접 양자화 벡터 간의 거리 차이를 최소화하는 정규화 손실 함수를 제안합니다. 이 접근 방식은 이전의 연구들에 비해 두 가지 목표를 동시에 달성하도록 합니다.

- **Performance Highlights**: 대표적인 벤치마크를 사용한 실험 결과, 제안한 방법이 더 신뢰할 수 있는 코드북 활용도를 달성하며 이전 방법들보다 성능이 개선된 것으로 나타났습니다. 이는 이미지 자동 인코딩(discrete image autoencoding) 및 대조적 음성 표현 학습(contrastive speech representation learning)과 같은 다양한 응용 분야에서 확인되었습니다. 전반적으로 이 연구는 벡터 양자화 문제를 해결하기 위한 새로운 시각을 제공하며, 심형의 유연성을 극대화하려는 노력이 담겨 있습니다.



### From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinemen (https://arxiv.org/abs/2509.22144)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문은 Multiround Adaptive Chain-of-Thought Compression (MACC) 프레임워크를 제안하여, Chain-of-Thought( CoT) 추론의 길이를 효과적으로 줄여주는 방법을 소개합니다. MACC는 각 입력에 대해 최적의 압축 깊이를 정하기 위한 점진적인 압축 전략을 사용합니다. 이를 통해 CoT의 평균 정확도가 5.6% 향상되며, CoT의 길이는 평균 47 토큰이 단축되고 지연 시간도 크게 줄어듭니다.

- **Technical Details**: MACC는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Chain-of-thought 생성, (2) 다단계 진행 압축, (3) 다중 작업 파인튜닝입니다. 이 과정에서 입력 질문에 대해 모델은 전체 추론 경로를 생성하고, 각 라운드에서 중복되거나 장황한 단계를 제거하면서 중요한 정보를 보존합니다. 또한, training set의 특징을 기반으로 test-time 성능을 예측할 수 있는 성능 추정 가설을 제안합니다.

- **Performance Highlights**: MACC는 모든 모델에 걸쳐 효과적인 모델 선택 및 예측을 가능하게 하며, 이전의 방법들보다 특정 입력에 따라 성능 저하가 적습니다. 이 접근법은 성능과 효율성을 동시에 달성하는 것이 가능하다는 것을 보이며, 파인튜닝 없이도 정보 생성과 함께 성능을 예측할 수 있습니다. 최적의 압축 전략을 선택하는 데 필요한 시간과 비용을 절감할 수 있는 방법을 제시하고 있습니다.



### REFINE-CONTROL: A Semi-supervised Distillation Method For Conditional Image Generation (https://arxiv.org/abs/2509.22139)
Comments:
          5 pages,17 figures

- **What's New**: 이번 연구에서는 Refine-Control이라는 새로운 반지도 증류 프레임워크를 제안합니다. 이 프레임워크는 tri-level knowledge fusion loss를 도입하여 지식 전이의 다양한 수준을 효과적으로 수행하고, 라벨이 없는 데이터와 라벨이 있는 데이터를 활용한 반지도 학습 방법을 통해 데이터 부족 문제를 완화합니다. 이로 인해 계산 비용과 지연 시간이 크게 줄어들면서도 고품질 이미지 생성을 유지할 수 있습니다.

- **Technical Details**: Refine-Control 프레임워크는 이미지 인페인팅을 위한 두 단계 프로세스로, 기초 학습과 고급 정제가 있습니다. 첫 번째 단계에서는 완전히 주석이 달린 데이터셋을 사용하여 학생 모델이 기본적인 매핑을 배우도록 합니다. 이어지는 두 번째 단계에서는 라벨이 없는 데이터셋을 활용하여 자가 지도 미세 조정을 수행하며, tri-level knowledge fusion loss를 통해 교사 모델의 계층적 지식을 효율적으로 전이합니다.

- **Performance Highlights**: 실험 결과 Refine-Control은 계산 비용과 지연 시간을 크게 줄이면서도 높은 충실도의 이미지 생성 능력과 제어 가능성을 유지하는 것으로 나타났습니다. 특히, 학생 모델이 복잡한 장면에서 다수의 객체를 다룰 때 발생하는 오해를 줄이는 데 기여했습니다. 또한, 이 프레임워크는 기존 모델보다 실용적인 데이터 접근을 허용하여 실제 환경에서도 효율적으로 활용될 수 있음을 보여주었습니다.



### Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding (https://arxiv.org/abs/2509.22134)
- **What's New**: 이 논문에서는 Group Tree Optimization (GTO)라는 새로운 훈련 알고리즘을 제안하여, 저항상태(decoding-time) 트리 정책과 훈련 간의 불일치를 해결하고자 합니다. 기존의 방식이 단일 그리디(draft) 경로만 최적화하는 문제를 가지고 있는 반면, GTO는 트리 기반 초안을 직접 최적화하는 방식을 통해 디코딩 효율성을 높입니다. 또한, 이 알고리즘은 훈련 목표를 실제 디코딩 절차와 일치시켜 초안 모델이 디코딩 시간의 효율성을 직접 향상시키도록 합니다.

- **Technical Details**: GTO는 두 가지 주요 구성 요소로 구성됩니다: (i) Draft Tree Reward는 목표 모델 하의 초안 트리의 기대 수용 길이를 측정함으로써 디코딩 성능을 직접적으로 평가하는 샘플링 없음(no sampling) 목표입니다; (ii) Group-based Draft Policy Training은 현재 모델과 고정된 참조 초안 모델의 트리 간 대조를 통해 디비아스(debiased)된 그룹 표준화 이점을 형성하는 안정적인 최적화 방식입니다. 이 방식은 PPO 스타일의 서브러겟을 적용하여 강력한 업데이트를 보장합니다.

- **Performance Highlights**: GTO는 대화(MT-Bench), 코드(HumanEval), 수학(GSM8K)과 같은 다양한 벤치마크에서 여러 개의 LLM(예: LLaMA-3.1-8B, LLaMA-3.3-70B)에서 성능을 검증하였으며, 수용 길이를 7.4% 증가시키고 이전의 최첨단 EAGLE-3에 비해 7.7%의 추가 속도 향상을 달성했습니다. 이러한 성과는 GTO가 디코딩 효율성 향상에 어떻게 기여하는지를 잘 보여줍니다.



### R-Capsule: Compressing High-Level Plans for Efficient Large Language Model Reasoning (https://arxiv.org/abs/2509.22131)
- **What's New**: 이번 논문의 핵심은 Reasoning Capsule(R-Capsule) 프레임워크로, 이는 Latent Reasoning의 효율성과 Chain-of-Thought(CoT)의 투명성을 결합하려는 새로운 접근 방식입니다. R-Capsule은 고수준 계획을 소형의 잠재 토큰으로 압축하면서도 실행 단계를 경량화하거나 명확하게 유지합니다. 이 혁신적인 접근은 정보 병목 원칙(Information Bottleneck)에서 영감을 받아 효과적으로 작업의 최소성을 유지하면서도 충분한 효용을 보장합니다.

- **Technical Details**: R-Capsule 프레임워크는 두 가지 주요 단계를 포함합니다: 잠재 계획 단계와 조건부 실행 단계입니다. 첫 번째는 고수준의 전략적 계획을 압축된 잠재 표현으로 생성하고, 두 번째로는 이 잠재 계획에 조건을 두어 저수준의 실행 단계를 진행하는 방식입니다. 우리의 아키텍처는 디코더 전용 트랜스포머를 기반으로 하며, Reasoning Capsule을 생성하고 활용하는 메커니즘을 포함합니다.

- **Performance Highlights**: R-Capsule은 복잡한 기준에서 명시적 CoT 보다 경쟁적이거나 개선된 정확도를 보이며, 적은 수의 가시 토큰 및 감소된 지연(latency)으로 결과를 제공합니다. 실험을 통해 R-Capsule이 계획과 실행의 압축에서 어떻게 유익한 인덕티브 편향(inductive biases)을 보존하는지를 확인했습니다. 전반적으로, R-Capsule은 정확도와 효율성을 동시에 달성하여, 더 적은 자원으로도 높은 성능을 제공합니다.



### Multi-Agent Path Finding via Offline RL and LLM Collaboration (https://arxiv.org/abs/2509.22130)
- **What's New**: 본 논문에서는 Multi-Agent Path Finding (MAPF) 문제를 해결하기 위해 offline reinforcement learning을 활용한 효율적인 분산 계획 프레임워크를 제안합니다. 이를 통해 훈련 시간은 몇 주에서 몇 시간으로 단축되며, 복잡한 의사소통 모듈의 필요성이 줄어듭니다. 특히, 환경 변화에 대한 적응력 향상을 위해, 대형 언어 모델(GPT-4o)을 통합하여 에이전트의 정책을 동적으로 안내합니다.

- **Technical Details**: 연구는 각 에이전트가 자신의 관찰에 따라 결정을 내리는 Decentralized Partially Observable Markov Decision Process (Dec-POMDP)에서 진행됩니다. MAPF 시나리오에서의 긴 범위 의사결정 문제를 해결하기 위해 Decision Transformer (DT) 아키텍처를 사용하며, 이는 지연된 보상을 다루는 데 효과적입니다. 또한, GPT-4o를 통해 동적 환경에서의 정책 조정을 가능하게 하여 기존의 RL 방법의 한계를 극복합니다.

- **Performance Highlights**: 실험 결과, DT 기반 접근 방식과 GPT-4o의 통합이 다양한 환경에서 에이전트의 성능을 유의미하게 향상시킴을 보여주었습니다. 특히 동적 조건에서 GPT-4o는 에이전트가 새로운 목표를 신속하게 탐지하고 조정할 수 있게 하여 비효율적인 탐색 행동을 피하도록 돕습니다. 이 연구는 MAPF 문제를 해결하는 데 있어 효율적이고 실용적인 접근 방식을 제시합니다.



### Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM (https://arxiv.org/abs/2509.22119)
Comments:
          10 pages, 6 figures, Accepted to ICAIL 2025 (International Conference on Artificial Intelligence and Law)

- **What's New**: 이번 논문에서는 Legal Article Prediction (LAP)에서의 한계를 극복하기 위해 Uni-LAP이라는 새로운 보편적 프레임워크를 제안합니다. 기존의 Supervised Classification Models (SCMs)과 Large Language Models (LLMs)의 장점을 결합함으로써, 법률 기사의 예측 정확도를 높이고 효율성을 강화하는 것을 목표로 합니다. Uni-LAP은 Top-K loss function을 도입하여 후보 법률 기사의 품질을 향상시키고, LLM은 삼단 논법(syllogism-inspired reasoning)을 활용하여 최종 예측을 정교화합니다.

- **Technical Details**: Uni-LAP 프레임워크는 SCM과 LLM 간의 밀접한 협력을 통해 법률 기사 예측을 수행합니다. 먼저 SCM은 Noval Top-K loss function으로 훈련되어 Top-K 후보 기사를 보다 정확하게 생성하며, 이후 LLM은 이를 토대로 최종 예측을 수행합니다. 이 과정에서 LLM은 법률 기사의 복잡성을 반영하기 위해 삼단 논법적 추론을 포함하여 예측 능력을 극대화합니다.

- **Performance Highlights**: 다양한 지역과 언어로 구성된 두 개의 실제 데이터 세트인 유럽 인권 재판소(ECtHR) 데이터 세트와 중국 AI 및 법률 챌린지(CAIL2018) 데이터 세트에서 Uni-LAP을 평가했습니다. 결과적으로 Uni-LAP은 모든 기준선 모델을 지속적으로 초월하는 성능을 보여주어, 효과성과 일반화 가능성을 입증하였습니다. 이를 통해 법률 기사의 예측 정확도를 비약적으로 향상시킬 수 있는 가능성을 확인할 수 있었습니다.



### The AI_INFN Platform: Artificial Intelligence Development in the Cloud (https://arxiv.org/abs/2509.22117)
Comments:
          To be published in SciPost Physics Proceedings for European AI for Fundamental Physics Conference (EuCAIFCon 2025)

- **What's New**: AI_INFN 프로젝트는 기계 학습(ML) 기술의 도입을 촉진하기 위해 INFN의 데이터 집약적 소프트웨어 개발을 지원하고 있으며, 이를 위해 AI 전용 컴퓨팅 자원의 프로비저닝에 초점을 맞추고 있습니다. INFN 클라우드의 클라우드 네이티브 솔루션을 활용하여 하드웨어 가속기를 효과적으로 공유하고, 연구 활동의 다양성을 보장하는 것을 목표로 하고 있습니다. 이 플랫폼은 GPU 기반 데이터 분석 워크플로우의 개발과 이들의 수직적 확장을 용이하게 하도록 설계되었습니다.

- **Technical Details**: AI_INFN 플랫폼은 Kubernetes 기반의 서비스 지향 아키텍처로 전환되었으며, 이는 기존 VM 기반 모델의 한계를 극복하는 역할을 합니다. 이 플랫폼은 연구자들이 인프라에 집중하지 않고, 연구에 더 집중할 수 있도록 고안되었습니다. INFN CNAF 데이터 센터에서 호스팅되는 하드웨어는 고성능 서버와 함께 Kubernetes 클러스터에서 관리됩니다.

- **Performance Highlights**: 현재 다양한 리소스 제공자에서 워크플로우를 관리할 수 있는 능력을 가진 이 플랫폼은 초기 테스트 결과와 사례 연구를 통해 신뢰성을 입증하고 있습니다. GPU 관리의 효율성은 NVIDIA GPU Operator의 도입을 통해 향상됐으며, 이는 GPU 드라이버 생명 주기 관리 및 모니터링을 자동화하여 일관된 구성의 유지보수를 가능하게 합니다. 향후, 이 플랫폼이 다양한 연구 분야의 혁신을 가속화하는 데 기여할 것으로 기대됩니다.



### Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization (https://arxiv.org/abs/2509.22115)
Comments:
          18 pages, 5 figures, Under review as a conference paper at ICLR 2026

- **What's New**: 이번 연구에서는 Dynamic Dual-Level Down-Sampling (D$^3$S) 프레임워크를 제안하여 리인포스먼트 러닝(Reinforcement Learning, RL)에서 메모리 효율성을 개선하고 정책 최적화를 가속화합니다. D$^3$S는 샘플 수준(sample-level)과 토큰 수준(token-level) 두 가지 차원에서 정보를 우선시하여 선택합니다. 이러한 방식으로, 상대적 보상에서 유래한 유용한 신호를 최대화함으로써 빠른 수렴을 도모합니다.

- **Technical Details**: D$^3$S 프레임워크는 두 가지 핵심 메커니즘으로 작동합니다. 첫 번째는 샘플 수준의 선택으로, 이 과정에서는 각 배치에 대해 그룹 상대적 이점을 산출하여 advantage variance (Var(A))를 최대화합니다. 두 번째는 토큰 수준으로, 이때 advantage 크기와 정책 엔트로피(Policy Entropy)의 곱을 이용하여 토큰의 중요성을 결정합니다.

- **Performance Highlights**: 실험 결과, D$^3$S는 GRPO 및 GSPO와 같은 기존 리인포스먼트 러닝 알고리즘에 통합되었을 때, 성능 향상과 함께 표본 및 토큰의 요구량을 줄이는 데 성공하였습니다. 특히 Qwen2.5 및 Llama3.1 모델을 위한 수치 실험에서 D$^3$S는 평균적으로 Pass@1에서 4.5, Pass@8에서 3.7 점수 향상을 기록하였습니다. 이러한 결과는 D$^3$S가 더욱 빠른 수렴을 가능하게 하며, 더 높은 정책 기울기(Policy Gradient)를 추구할 수 있음을 보여줍니다.



### Reinforcement Learning for Durable Algorithmic Recours (https://arxiv.org/abs/2509.22102)
- **What's New**: 이번 연구에서는 알고리즘적 보상(algorithmic recourse) 접근법의 주요 문제인 시간적 동적성을 다루는 새로운 프레임워크를 제안합니다. 이는 추천 시스템이 후보자 집단의 변화에 어떻게 대응하는지를 명시적으로 모델링합니다. 또한, 환경의 변화에 적응하여 실행 가능하고 유효한 추천을 생성하는 강화학습 기반의 새로운 알고리즘을 도입합니다. 이 방법은 추천의 지속 가능성을 보장하여 사람들이 제안된 변경 사항을 구현한 후에도 자신있게 재신청할 수 있도록 합니다.

- **Technical Details**: 우리는 제한된 자원과 경쟁적인 환경 하에서 작동하는 알고리즘적 보상 프레임워크를 제안합니다. 이 프레임워크는 후보자 재신청 간의 시간적 지연과 피처 수정의 난이도를 반영하도록 설계되었습니다. 강화학습(RL) 관점에서 추천 과정을 RL agent의 정책으로 해석하고, 이러한 상호작용의 순차적 특성을 포착합니다. 연구에서는 후보자 집단에 대한 추천의 피드백 효과를 명시적으로 고려하여 정보를 제공합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 제안된 방법이 기존의 알고리즘보다 우수한 성능을 보임을 입증하였습니다. 이 연구는 실행 가능성과 장기적 유효성 간의 균형을 잘 이루는 방법을 제시하며, 피크 데이터를 바탕으로 문제를 분석합니다. 연구 결과, 사회적 기술적 맥락과 시스템의 동적 변화에 대한 고려가 알고리즘적 보상 설계에서 얼마나 중요한지 강조합니다.



### SecureAgentBench: Benchmarking Secure Code Generation under Realistic Vulnerability Scenarios (https://arxiv.org/abs/2509.22097)
- **What's New**: 이번 논문에서는 코드 에이전트의 안전한 코드 생성을 평가하기 위한 새로운 벤치마크인 SecureAgentBench를 제안합니다. 이 벤치마크는 105개의 코딩 작업을 포함하고 있으며, 실세계의 오픈소스 취약점을 기반으로 한 맥락을 고려합니다. 기존 벤치마크의 한계를 넘어, 기능적 정확성 및 새로운 취약성의 검출을 아우르는 포괄적인 평가 방식을 채택하고 있습니다.

- **Technical Details**: SecureAgentBench는 현실적인 작업 형태를 채택하여 여러 파일을 수정하는 것을 요구합니다. 각 작업은 평균 200단어의 요구사항을 해석하고, 최대 36.4K 파일과 4.2M 코드 라인을 가진 코드 베이스를 분석해야 합니다. 평가에는 기능 테스트, PoC(exploits) 프로그램을 통한 검증, 정적 애플리케이션 보안 테스트(SAST) 도구를 사용하여 새로운 취약성을 탐지하는 방식이 포함됩니다.

- **Performance Highlights**: 평가 결과 현재의 코드 에이전트들은 올바르고 안전한 코드를 생성하는 데 어려움을 겪고 있으며, SWE-agent가 지원되는 DeepSeek-V3.1의 경우에도 오직 15.2%만이 올바르고 안전한 솔루션을 제공하였습니다. 또한, 기존의 취약성을 재도입하는 것 외에도 새로운 보안 리스크를 발생시키는 경향이 20% 이상으로 나타났습니다. 이러한 결과는 코드 에이전트의 보안 인식 강화를 위한 추가 연구의 필요성을 강조합니다.



### Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation (https://arxiv.org/abs/2509.22093)
- **What's New**: 이 논문에서는 Action-aware Dynamic Pruning (ADP)이라는 새로운 멀티모달 프루닝 프레임워크를 제안합니다. ADP는 비전 언어 행동 모델에서 시각적 토큰의 중복성을 고려하여 로봇 조작의 다양한 단계에서 효율성을 높입니다. 기존 기술들은 시각적 중복성을 단순히 줄이는 접근에 그쳤으나, ADP는 조작의 단계에 따라 중복성이 다르다는 점에 착안했습니다.

- **Technical Details**: ADP는 (1) 텍스트 기반 프루닝과 (2) 동작 인식 동적 조정의 두 가지 아이디어로 구성됩니다. 첫 번째는 멀티모달 데이터를 서로 유사한 조건에서 평가하여 최적의 시각적 토큰만 선택하는 것입니다. 두 번째는 최근의 동작 통계를 기준으로 프루닝이 활성화되는지 여부를 결정하는 경량의 결정 신호를 사용하는 것입니다.

- **Performance Highlights**: ADP를 적용한 실험에서, LIBERO 스위트와 다양한 실제 상황에서 FLOPs(부동 소수점 연산 수)와 동작 추론 지연 시간을 눈에 띄게 감소시켰습니다. 예를 들어 OpenVLA-OFT에서는 1.35배의 속도 향상을 기록했고, OpenVLA비교하여 25.8% 성공률 개선을 달성했습니다. 이는 로봇 정책의 효율성과 성능을 동시에 발전시키는 간단한 플러그인 경로를 제공합니다.



### The Rogue Scalpel: Activation Steering Compromises LLM Safety (https://arxiv.org/abs/2509.22067)
- **What's New**: 이번 연구에서는 Activation steering 기법이 모델의 안전성과 정렬 기능을 어떻게 손상시킬 수 있는지를 밝히고 있습니다. 일반적으로 이 기법은 명확한 해석 가능성을 제공하는 안전한 모델 제어 대안으로 인식되나, 연구 결과에 따르면 오히려 유해한 요청을 수용할 가능성을 증가시킵니다. 더구나, 랜덤한 방향으로의 Steering 만으로도 유해한 응답을 생성할 확률이 0%에서 2-27%로 증가한다는 충격적인 결과가 나타났습니다.

- **Technical Details**: Activation steering은 모델의 내부 상태를 조작하여 동작을 제어하는 방법으로, 특정 방향 벡터를 은닉 상태에 주입하여 작동합니다. 이 연구에서 사용된 JailbreakBench 데이터셋에서는 100개의 유해 쿼리를 대상으로 스티어링을 적용해 응답을 수집하고, 유해성을 평가했습니다. 조사된 모델은 Llama-3, Qwen2.5 및 Falcon-3을 포함하여 여러 모델 패밀리에서 결과를 참조하고 있습니다.

- **Performance Highlights**: 연구의 주요 발견은 랜덤한 방향으로의 스티어링이 모델의 거부 메커니즘을 효과적으로 무너뜨릴 수 있다는 것입니다. SAE 기능을 사용한 스티어링은 랜덤 스티어링보다 2-4% 더 유해한 결과를 보였으며, 20개의 랜덤 벡터를 합친 경우에는 유니버설 공격이 가능하다는 것이 밝혀졌습니다. 이는 악의적인 사용자가 모델의 안전 장치를 우회하는 데 활용할 수 있는 잠재적인 위험을 내포하고 있습니다.



### The QCET Taxonomy of Standard Quality Criterion Names and Definitions for the Evaluation of NLP Systems (https://arxiv.org/abs/2509.22064)
Comments:
          39 pages, 7 figures

- **What's New**: 이 논문에서는 동일한 품질 기준 이름(예: Fluency)을 사용하는 NLP 평가 실험이 반드시 동일한 품질 측면을 평가하지 않음을 보여주고 있습니다. 이는 서로 다른 평가를 가진 시스템 품질에 대한 신뢰할 수 있는 결론을 내릴 수 없게 하며, NLP 분야의 과학적 발전을 저해하는 요소로 작용합니다. 따라서, 품질 기준 이름과 정의의 표준 세트를 생성하는 것이 필요하다고 주장합니다.

- **Technical Details**: 이 연구에서는 QCET(Quality Criteria for Evaluation Taxonomy) 품질 기준을 제안하며, NLP에서 보고된 세 가지 평가 조사에서 유래한 표준 품질 기준 이름과 정의를 도출합니다. 이러한 기준은 계층 구조로 구성되어 있으며, 각 부모 노드는 하위 노드의 공통 측면을 포착합니다. 이는 평가의 비교 가능성을 확보하고, 새로운 평가 설계를 안내하며, 규제 준수를 평가하는 데 도움을 줍니다.

- **Performance Highlights**: QCET의 주요 활용법으로는 기존 평가의 비교 가능성을 확립하고, 새로운 평가의 설계를 안내하며, 규제 준수를 평가하는 것이 포함됩니다. 이는 NLP 평가의 일관성을 높이고, 연구의 신뢰성을 향상시키기 위한 노력이 될 것입니다. 이러한 자원은 NLP 분야의 품질 평가 기준의 명확화를 통해, 향후 연구에 중요한 기초자료로 기능할 것입니다.



### Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks (https://arxiv.org/abs/2509.22060)
- **What's New**: 최근 연구에 따르면, 자동 음성 인식 시스템(Automatic Speech Recognition, ASR)은 적대적 공격에 취약하다는 점이 밝혀졌습니다. 특히, 이 논문에서는 비용 효율적인 화이트 박스 공격(white-box attack)과 비전이식 블랙 박스 공격(non-transferability black-box attack)을 탐구하고 있으며, Fast Gradient Sign Method와 Zeroth-Order Optimization에서 인사이트를 가져왔습니다. 또한, 데이터 오염 공격(poisoning attack)이 최첨단 모델의 성능 저하를 초래할 수 있음을 시연하고 있어 주목할 만합니다.

- **Technical Details**: ASR 시스템은 복잡한 전처리(preprocessing), 기능 추출(feature extraction), 모델 기반 예측(model-based prediction) 단계를 특징으로 하여 적대적 공격에 대한 방어가 어렵습니다. 이 논문은 화이트 박스 및 블랙 박스 공격을 통해 ASR 시스템의 취약점을 탐구하며, 특히 제로스 오더 최적화(Zeroth-Order Optimization, ZOO) 공격을 통해 공격자가 목표 모델에 대한 쿼리 접근을 통해 인식되지 않는 적대적 예제를 생성하는 방법을 설명합니다. 데이터 오염 기법을 ASR 시스템에 통합하여 첫 번째로 데이터 오염이 모델 성능 저하에 미치는 영향을 논의합니다.

- **Performance Highlights**: 본 연구는 Fast Gradient Sign Method를 활용한 비용 효율적인 회피 공격을 통해, 현재의 최첨단 Distil Whisper 모델이 목표 및 비목표 회피 공격에 취약하다는 점을 입증합니다. 특히, 블랙박스 설정에서 제로스 오더 최적화 공격을 통해 50%의 단어 오류율(Word Error Rate)을 달성했으며, 신호 대 잡음 비율(Signal Noise Ratio)은 1.5-4 dB입니다. 실험을 통해 ASR 시스템이 제공하는 보안 취약점의 심각성을 강조하며, 이러한 문제에 대한 적절한 방어 기법 개발의 필요성을 제기하고 있습니다.



### An Adaptive ICP LiDAR Odometry Based on Reliable Initial Pos (https://arxiv.org/abs/2509.22058)
- **What's New**: 이 논문에서는 신뢰할 수 있는 초기 자세를 기반으로 한 적응형 ICP 기반 LiDAR 오도메트리 방법을 제안합니다. 기존의 방법들이 초기 자세의 신뢰성을 충분히 고려하지 않는 문제를 해결하기 위해, 밀도 필터링을 통해 분산된 조정 등록을 수행하여 초기 자세 추정을 얻습니다. 또한, 실시간 동적 환경 변화를 효과적으로 처리할 수 있는 적응형 메커니즘을 도입하여 전체적인 등록 정확도를 향상시킵니다.

- **Technical Details**: 제안된 방법은 먼저 밀도 필터링을 이용한 분산 조정 등록으로 초기 자세를 추정합니다. 그런 다음 이 초기 자세를 동작 예측 자세와 비교하여 신뢰할 수 있는 초기 자세를 선택하고, 현재 및 역사적 오류를 결합하여 적응형 임계값을 동적으로 조정합니다. 마지막으로, 높은 정밀도의 정렬을 달성하기 위하여 신뢰할 수 있는 초기 자세와 적응형 임계값을 기반으로 점대면(uint-to-plane) 적응형 ICP 등록을 수행합니다.

- **Performance Highlights**: KITTI 데이터셋을 활용한 실험 결과, 제안된 방법이 기존의 LiDAR 오도메트리 방법들보다 등록 정확도에서 우수한 성능을 나타냄을 보여주었습니다. 특히, 복잡한 동적 환경에서의 LiDAR 오도메트리 성능을 효과적으로 향상시켰으며, 실용적인 응용 가능성을 높였습니다. 이로 인해 제안된 방법은 이동 로봇의 자율주행 및 내비게이션 기술에 중요한 기여를 할 것으로 기대됩니다.



### Fuzzy Reasoning Chain (FRC): An Innovative Reasoning Framework from Fuzziness to Clarity (https://arxiv.org/abs/2509.22054)
Comments:
          Accepet by EMNLP 2025 Findings (11 pages, 1 figures)

- **What's New**: 이 논문에서는 모호한 텍스트와 불확실성을 처리하기 위한 Fuzzy Reasoning Chain (FRC) 프레임워크를 소개합니다. FRC는 대형 언어 모델(LLM)의 의미적 사전 정보를 지속적인 퍼지 멤버십(continuous fuzzy membership)으로 통합하여 확률 기반 추론과 퍼지 멤버십 추론 간의 명확한 상호작용을 생성합니다. 이 방법을 통해 전통적인 방법으로는 처리할 수 없는 혼란스러운 입력을 명확하게 이해 가능한 결정으로 전환할 수 있습니다.

- **Technical Details**: FRC는 감정 분석의 표준 단계별 추론 절차를 따르며, 이산 확률 할당을 지속적인 퍼지 멤버십으로 대체하여 기존의 체인 오브 썸 논리(Chain-of-Thought, CoT) 접근법을 확장합니다. 확률 기반 추론에서 퍼지 멤버십 추론으로의 전환은 논문의 핵심 방법론 혁신으로, 감정의 보다 세련되고 강력한 표현을 가능하게 합니다. FRC는 여러 모델 규모 간의 지식 전이 가능성을 높이는 동시에 안정적인 추론을 보장합니다.

- **Performance Highlights**: FRC는 감정 분석 태스크에서 이론적 분석과 경험적 결과를 통해 안정적인 추론을 보장하고 모델 성능을 향상시킵니다. 실험 결과는 FRC가 혼란스럽고 모호한 입력에서 뛰어난 성능을 보여준다고 강조합니다. 이러한 발견은 FRC가 더 나은 해석 가능성과 견고함으로 미세하고 모호한 표현을 관리하기 위한 일반적인 메커니즘을 제공한다는 것을 보여줍니다.



### Latent Diffusion : Multi-Dimension Stable Diffusion Latent Space Explorer (https://arxiv.org/abs/2509.22038)
- **What's New**: 이번 논문에서는 Stable Diffusion 모델에 맞춤형 latent space 조작을 통합한 Latent Diffusion 프레임워크를 소개합니다. 새로운 기법을 통해 사용자는 직관적으로 개념과 형태를 조작할 수 있어, 생성 예술의 창의적인 가능성을 더욱 넓혀줍니다. 이로 인해, 사용자는 더 세밀하게 생성된 콘텐츠를 조절할 수 있는 자유를 얻게 됩니다.

- **Technical Details**: Latent Diffusion은 두 가지 유형의 사용자 정의 벡터 조작을 도입합니다: Query-wise Concept Latent Operation과 Conditioning Vector Shape Latent Operation. 이 조작들은 각각 AI의 개념 인식과 공간 정보와 관련된 latent 벡터를 조작하게 해주어, 사용자가 보다 목적에 맞게 latent space를 탐색할 수 있게 도와줍니다. 이렇게 확장된 작업 흐름에서는 새로운 조작자를 통해 네트워크의 정보 이해도를 직접적으로 제어할 수 있습니다.

- **Performance Highlights**: 논문에서 제시된 두 개의 예술 프로젝트, Infinitepedia와 Latent Motion은 이 도구의 실제 적용 가능성을 보여줍니다. 이들은 개념 혼합과 동적인 움직임 생성에 있어 Latent Diffusion의 강점을 강조하며, semantic robustness와 latent space 내의 모호성을 줄이는 데 기여합니다. 또한, 논문은 diffusion 모델 내에서 'latent deserts'라는 현상을 식별하고, 향후 탐색을 위한 기하학적 작업에 대해 논의합니다.



### Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics (https://arxiv.org/abs/2509.22014)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 의료 환경을 위한 경량의 에이전틱(multimodal) 프레임워크를 소개합니다. 기존의 비전-언어 모델들이 제한된 시간적(reasoning) 추론 및 불확실성 측정의 한계를 보여주는 것에 비해, 이 프레임워크는 비디오 기반의 장면 이해를 목표로 하여 업그레이드되었습니다. Qwen2.5-VL-3B-Instruct 모델과 SmolAgent 기반의 조정 레이어를 결합하여 사고 연쇄(chain-of-thought reasoning), 음성-비전 융합, 동적 도구 호출 기능을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 VisionQA, SceneGen 및 GraphQA라는 세 가지 보완 워크플로우로 구성되어 있습니다. 이 구조는 안전 비즈니스 환경에서 클리니션 및 로봇 시스템을 지원하기 위한 것이며, 의사결정의 해석 가능성과 추적 가능성을 보장합니다. 통합된 하이브리드 검색 메커니즘을 통해 효율적이고 해석 가능한 지식 통합이 가능하여, 로봇이 안전하게 행동할 수 있는 기반을 마련합니다.

- **Performance Highlights**: 실험 결과, Video-MME 벤치마크 및 맞춤형 임상 데이터 세트에서 기존의 비전-언어 모델에 비해 경쟁력 있는 정확도와 개선된 강인성을 보여주었습니다. 이 프레임워크는 로봇 보조 수술, 환자 모니터링, 의사 결정 지원 등 다양한 의료 응용 분야에서의 잠재력을 강조합니다. 따라서 안전하고 투명하며 적응력이 뛰어난 의료 로봇의 발전에 기여할 것으로 기대됩니다.



### Black-Box Hallucination Detection via Consistency Under the Uncertain Expression (https://arxiv.org/abs/2509.21999)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전에도 불구하고, 사실과 다른 응답을 생성하는 '환각(hallucination)' 문제는 여전히 해결되지 않고 있습니다. 기존의 환각 탐지 방법들은 외부 자원이나 LLM의 내부 상태에 의존하는 경우가 많습니다. 그러나 본 연구에서는 블랙 박스(Black-Box) 접근 방식을 통해 효과적인 환각 탐지 기술을 개발하고, LLM의 불확실성을 표현하는 모델을 통해 환각 탐지를 간단하게 수행할 수 있는 메트릭을 제안합니다.

- **Technical Details**: 우리는 LLM의 응답에서 사실과 비사실적 응답을 구분할 수 있는 프롬프트를 탐구합니다. 특히, 불확실한 표현과 확실한 표현을 사용하여 LLM의 일관성을 평가하는 방법을 제안합니다. 본 메트릭은 샘플링 없이 단일 응답만으로도 효과적으로 실제 정보를 구별할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 메트릭이 기존의 내부 지식에 의존하는 환각 탐지 방법들보다 모델 응답의 사실성을 더 잘 예측하는 것으로 나타났습니다. 두 개의 오픈 도메인 질의응답 데이터셋에서 (질문, 응답) 쌍을 분석한 결과, 사실적 응답에서 LLM이 일관된 응답을 생성하는 경향이 강하다는 것을 확인했습니다.



### ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models (https://arxiv.org/abs/2509.21991)
- **What's New**: 이번 연구에서는 고해상도 이미지 처리의 효율성을 높이기 위한 새로운 모델인 ERGO (Efficient Reasoning & Guided Observation)를 발표했습니다. 기존의 대형 비전-언어 모델(LVLMs)은 비디오 토큰 수로 인해 높은 계산 비용이 발생하는데, 이를 해결하기 위해 두 단계의 "coarse-to-fine" 추론 파이프라인을 제안했습니다. 분석된 이미지에서 중요한 지역을 식별한 후, 해당 영역만을 높은 해상도로 크롭하여 후속 재처리를 진행하는 방식입니다.

- **Technical Details**: ERGO는 강화 학습( Reinforcement Learning) 기반의 모델로, 이미지와 쿼리를 기반으로 이미지에서 중요한 영역의 바운딩 박스를 생성합니다. 이때, 전체 이미지에서 초점이 되는 영역을 재구성한 후 해당 영역에 대한 정확한 질문 응답을 수행합니다. 연구자는 이 모델이 비전 처리 효율성과 함께 정밀한 추론 능력도 유지할 수 있도록 설계했다고 설명합니다.

- **Performance Highlights**: 여러 데이터셋에서 ERGO는 기존 모델들과 비교하여 정확도를 크게 향상시키며, 적은 수의 비전 토큰 사용으로도 성능이 우수함을 입증했습니다. 예를 들어, ERGO는 V* 벤치마크에서 Qwen2.5-VL-7B를 4.7점 초과하며 비전 토큰의 23%만을 사용하여 3배의 추론 속도를 달성하였습니다. 이러한 결과는 고해상도 비전-언어 처리의 효율성과 정확성을 동시에 개선할 수 있음을 보여줍니다.



### Developing Vision-Language-Action Model from Egocentric Videos (https://arxiv.org/abs/2509.21986)
- **What's New**: 이번 연구에서는 EgoScaler라는 프레임워크를 활용하여 보조 기록 없이도 egocentric 비디오에서 6DoF 물체 조작 궤적을 추출합니다. 이러한 접근 방식은 VLA(비전-언어-행동) 모델을 훈련하는 데 필요한 보조 주석의 필요성을 없애고, 대규모 egocentric 비디오 데이터셋을 자동으로 구축할 수 있게 해 줍니다. 기존 방법들이 보조 기록에 의존하던 점을 극복하며, 필요한 데이터의 양을 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: EgoScaler는 egocentric 비디오에서 물체 조작 궤적을 추출하도록 설계된 프레임워크입니다. 각 궤적의 포즈는 조작되는 물체의 중심과 회전을 나타내며, 이는 로봇의 end-effector 상태로 근사화됩니다. 이 프레임워크는 다양한 대규모 egocentric 비디오 데이터셋에 적용되어, 전처리 과정을 통해 노이즈가 많은 데이터나 불완전한 궤적을 자동으로 정제하여 새로운 대규모 데이터셋을 구성합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, EgoScaler를 사용하여 사전 훈련된 데이터셋의 성공률이 20% 이상 향상되며, 실제 로봇 데이터셋을 사용한 성능과 경쟁력이 있습니다. VLA 아키텍처를 기반으로 한 테스트에서도, 기존의 데이터셋과 결합 시 추가적인 성능 개선을 이루어내었습니다. 이러한 결과는 egocentric 비디오가 VLA 연구를 최적화할 수 있는 유망한 자원임을 보여줍니다.



### Hybrid Diffusion for Simultaneous Symbolic and Continuous Planning (https://arxiv.org/abs/2509.21983)
Comments:
          10 pages, 11 figures. This work has been submitted to the IEEE for possible publication. See this https URL for the project website

- **What's New**: 이 논문에서는 로봇이 장기 목표를 달성하기 위한 새로운 접근 방식인 Hybrid Diffusion Planner (HDP)를 제안합니다. HDP는 연속적인 행동 경로와 기호적 계획을 동시에 생성하여 장기적인 계획 문제를 해결하는데 중점을 둡니다. 기존의 Diffusion Models가 장기적 의사결정에서 성능이 저하되는 문제를 해결하기 위해 새로운 결합 방식을 도입하였습니다.

- **Technical Details**: HDP는 두 개의 결합된 확산 과정을 통해 기호적 계획과 연속적 모션 계획을 동시에 모델링합니다. 특히, HDP는 기호적 계획을 통해 로봇의 행동을 보다 투명하게 제어할 수 있게 하며, 조건부 샘플링을 통해 특정 상황에 맞는 행동 경로를 생성할 수 있도록 합니다. 이러한 접근법은 노이즈의 생성과 예측을 포함하여, 기존 계획법에 비해 상당한 성능 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, HDP는 복잡하고 장기적인 계획을 요구하는 로봇 작업에서 기존의 순수한 Diffusion Models보다 뛰어난 성능을 보였습니다. HDP를 이용한 조건부 샘플링 기능은 다양한 과제를 효과적으로 처리할 수 있게 하여, 실제 로봇 작업에서 더욱 활용 가능성이 높습니다. 또한, HDP의 유연한 조건 설정은 로봇의 동작을 더욱 직관적으로 수월하게 제어할 수 있다는 장점을 제공합니다.



### Benchmarking and Mitigate Psychological Sycophancy in Medical Vision-Language Models (https://arxiv.org/abs/2509.21979)
Comments:
          19figures, 37pages

- **What's New**: 이번 연구는 비전 언어 모델(VLMs)이 의료 영상 및 임상 의사 결정 지원 시스템에 점점 더 많이 통합되고 있는 현황을 다룹니다. 그러나 이 모델들은 사용자의 언어적 표현이나 사회적 신호에 따라 실제 증거 기반의 추론보다 우선순위를 두는 경향이 있습니다. 따라서, 임상 질문 응답(visual question answering)에서 모델의 '아부적' 행동을 평가하기 위한 새로운 벤치마크를 제안합니다.

- **Technical Details**: 연구진은 PathVQA, SLAKE 및 VQA-RAD에서 다양한 장기 시스템 및 모드에 따라 계층화된 비슷한 패턴을 가진 의료 아부 데이터 세트를 구성했습니다. 심리적 압박 템플릿을 활용하여 다양한 아부 행동을 평가하고, 실험에서 모델의 약점을 분석했습니다. 특히, 시각적 증거와는 무관한 모델의 편향 메커니즘을 발견하였고, 이러한 문제를 해결하기 위해 VIPER(Visual Information Purification for Evidence based Response)라는 경량화된 완화 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, 다양한 모델에서 아부에 대한 취약성이 관찰되었으며, VIPER는 서로 다른 압박 유형과 시스템 간의 최상의 저항성과 복원 균형을 달성했습니다. 특히, 적용된 압박 유형에 따른 성능 변화와 VIPER의 효과를 심층적으로 분석하여, VIPER가 비주얼 증거에 기초한 답변 생성을 유지할 수 있도록 정렬 목표와 수용 압박 간의 상호작용을 강조했습니다. 이러한 연구 결과는 다중 모달리티 임상 AI의 행동 신뢰성을 평가하고 개선하는 기초를 제공합니다.



### Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning (https://arxiv.org/abs/2509.21976)
- **What's New**: 최근 원격 감지(이것은 Remote Sensing을 의미함)에서 지시 표현(Referring Expression) 이해에 대한 독창적인 접근 방식인 Geo-R1을 제안했습니다. Geo-R1은 극히 제한된 데이터 환경에서도 효과적으로 작동할 수 있도록 설계된 추론 중심의 강화 학습(RL) 기반의 방법입니다. 이 방식은 모델이 지시 표현을 분해하여 명확하고 해석 가능한 추론 체인을 생성하게 하여, 타겟 객체를 효과적으로 локалize할 수 있도록 돕습니다.

- **Technical Details**: Geo-R1은 장면 분류와 달리 지역 수준의 인식이 중요한 REU(Referring Expression Understanding) 작업에서 초점 맞춰집니다. 기존의 감독된 미세 조정(Supervised Fine-Tuning) 방식 대신, Geo-R1은 여러 추론 경로를 탐색하고, 각 예제당 더 풍부한 감독을 제공함으로써 소량의 샘플을 보다 효율적으로 활용합니다. 이 과정에서 REU를 위한 특정 보상 함수를 최적화하여, 각 작업(REC와 RES)에서 목표와 일치하는 학습을 촉진합니다.

- **Performance Highlights**: Geo-R1은 세 가지 신뢰할 수 있는 소수 샘플 벤치마크 모두에서 SFT 기반의 모델보다 뛰어난 성능을 보였습니다. 특히, 다양한 데이터셋 간에 더 견고한 일반화를 보여주어, 그 성능의 안정성을 입증했습니다. 또한 Geo-R1이 생성한 추론 흔적은 해석 가능성을 제공하며, 지역 및 의미적 단초를 활용하여 최종 로컬라이제이션에 기여합니다.



### From Superficial Outputs to Superficial Learning: Risks of Large Language Models in Education (https://arxiv.org/abs/2509.21972)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 교육에 미치는 영향과 이로 인해 발생하는 다양한 위험 요소를 체계적으로 분석합니다. 70개의 실증 연구를 검토하여 LLMs의 교육적 응용에 대한 연구가 세 가지 주요 영역으로 구분된다는 점을 발견했습니다. 또한, 연구에서는 LLMs의 통합이 학생과 학습 시스템에 미치는 위험에 대해 다각적으로 접근하고, 인간 중심의 적용이 필요하다고 강조합니다.

- **Technical Details**: LLMs는 대량의 텍스트 데이터를 기반으로 훈련되어 전문적인 답변을 모방하는 능력을 가집니다. 모델의 작동 메커니즘은 입력 시퀀스의 각 부분에 가중치를 동적으로 할당하는 transformer 모델에 기반하고 있습니다. 이 기술적 토대 위에서 LLM은 특정 맥락에서 가장 가능성이 높은 다음 토큰을 예측하여 유창하고 일관된 언어를 생성하는 데 도움을 줍니다.

- **Performance Highlights**: LLMs의 교육적 응용에 대한 연구 결과, 시스템 수준의 위험(표면적인 이해, 알고리즘적 편향, 데이터 개인 정보 보호 문제 등) 이외에도 학생의 경험과 인지 프로세스에 미치는 영향이 강조되었습니다. LLMs와의 상호작용이 학생의 비판적 사고 감소, 독립 학습 기술 저하, 학생의 자율성 상실 등과 같은 부정적인 결과를 초래할 수 있다고 경고합니다. 기술적, 교육적, 사회적 위험을 포괄하는 리스크 모델을 통해 교육적 성과에 미치는 영향을 분석하는 방법론을 제안합니다.



### No-Reference Image Contrast Assessment with Customized EfficientNet-B0 (https://arxiv.org/abs/2509.21967)
Comments:
          32 pages, 9 tables, 6 figures

- **What's New**: 이번 연구에서는 이미지 품질 평가에 있어서 중요한 요소인 대비(distortion)를 효과적으로 평가하기 위해 새로운 딥러닝 기반의 프레임워크를 제안하였습니다. 이 프레임워크는 세 가지 사전 훈련된 아키텍처, 즉 EfficientNet B0, ResNet18, MobileNetV2를 맞춤 구성하고 미세 조정(fine-tuning)하여 인지 관점의 Mean Opinion Score를 기반으로 합니다. 또한, 시암 네트워크(Siamese network)를 활용한 추가 모델이 개발되었으며, 이는 인지 대비 왜곡(perceptual contrast distortions)을 포착하는 능력이 제한적임을 보여주었습니다.

- **Technical Details**: 제안된 모델은 대비 인식 회귀 헤드(contrast-aware regression head)로 수정되어 CID2013과 CCID2014라는 두 개의 벤치마크 데이터셋에서 합성(synthetic) 및 진짜(authentic) 대비 왜곡 데이터를 위한 타겟된 데이터 증강(targeted data augmentations)을 이용하여 엔드 투 엔드(end-to-end)로 훈련됩니다. 성능 평가는 예측된 점수와 인간 평가 점수 간의 정렬을 평가하는 Pearson Linear Correlation Coefficient(PLCC)와 Spearman Rank Order Correlation Coefficient(SRCC)를 사용하여 평가됩니다.

- **Performance Highlights**: 제조된 EfficientNet B0 모델은 CCID2014에서 PLCC = 0.9286, SRCC = 0.9178의 성능을, CID2013에서는 PLCC = 0.9581, SRCC = 0.9369의 성능으로 전통적인 방법들 및 다른 딥러닝 기반 모델들을 초월하며 최첨단(performance state-of-the-art)을 달성했습니다. 이 결과들은 제안된 모델이 인지 대비 왜곡을 효과적으로 포착할 수 있는 강력함과 효율성을 강조합니다. 전반적으로, 대비 인식(adaptation) 경량 사전 훈련 네트워크를 기반으로 한 제안된 방법은 실시간 및 자원 제한 환경에서도 사용 가능한 고성능의 확장 가능한 솔루션을 제공함을 보여주었습니다.



### FlowDrive: moderated flow matching with data balancing for trajectory planning (https://arxiv.org/abs/2509.21961)
- **What's New**: 본 논문에서는 자율주행 분야에서의 비율 불균형 문제를 해결하기 위해 FlowDrive라는 새로운 기법을 제안합니다. FlowDrive는 조건부 수정된 흐름(conditional rectified flow)을 이용하여 훈련 데이터에서 드라이브 시나리오를 효과적으로 생성합니다. 특히, 드문 시나리오를 다루는 데 중점을 두어 지도(percentages)와 함께 동작의 다양성을 높이는 메커니즘을 도입했습니다.

- **Technical Details**: FlowDrive는 관찰된 환경 상황을 고려하여 자율주행 차량의 실행 가능한 경로를 생성하는 데 중점을 둡니다. 이를 위해 흐름 맞춤(flow matching) 방식을 사용하여 소음(noise)을 직접적으로 드라이브 경로로 매핑합니다. 또한, 샘플링 성능을 향상시키기 위해 경로 간에 소규모의 섭동(perturbation)을 적용하여 경로의 다양성을 증가시키는 중재 가이드를 도입했습니다.

- **Performance Highlights**: FlowDrive는 nuPlan 및 interPlan 벤치마크에서 학습 기반 플래너 중 가장 높은 성능을 달성했습니다. 특히, 중재 가이드를 추가하고 경량 후처리(FlowDrive*)를 수행함으로써 거의 모든 벤치마크 분할에서 최첨단 성능을 확보했습니다. 이로써 전통적인 규칙 기반 플래너 및 학습 기반 접근 방식과 비교할 때 월등한 결과를 보여주었습니다.



### Active Attacks: Red-teaming LLMs via Adaptive Environments (https://arxiv.org/abs/2509.21947)
Comments:
          22 pages, 7 figures, 18 tables

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)에서 유해한 행동을 유도하는 다양한 공격 프롬프트 생성을 위한 새로운 접근법인 Active Attacks를 제안합니다. 기존의 수작업으로 진행되는 프롬프트 제작 방법 대신, 적대자 LLM이 강화 학습(RL)을 통해 자동으로 공격 프롬프트를 생성하도록 권장합니다. 이 방식은 유해한 행동의 다양성을 포착하기 위한 명시적 목표가 필요하며, 기존의 RL 방법들이 쉽게 발견된 모드에 구속되는 문제를 해결하고자 합니다.

- **Technical Details**: Active Attacks는 적대자의 공격을 발전시키는 강화 학습 기반의 알고리즘으로, 피해자 LLM이 진화함에 따라 공격을 적응시킵니다. 주기적으로 수집된 공격 프롬프트로 피해자 LLM을 안전하게 미세 조정(safety fine-tuning)함으로써, 이미 활용된 영역의 보상을 줄이고 공격자는 새로운 취약점을 탐색하기 강제합니다. 이 과정은 쉽게 발견된 모드에서 시작하여 어려운 모드로 진행되는 자연스러운 탐색 커리큘럼을 만듭니다.

- **Performance Highlights**: Active Attacks를 기존의 RL 알고리즘인 GFlowNets와 통합했을 때, 공격 성공률이 0.07%에서 31.28%로 증가한 결과를 보여주며, 상대적으로 400배 이상의 개선을 나타냅니다. 이 방법은 단지 6%의 계산 증가로 더 다양한 유해 행동을 발견하며, RL 기반의 기존 방법들의 모드 붕괴 문제를 극복합니다. 이로 인해 수집된 프롬프트는 안전 미세 조정에 유용한 데이터셋으로 활용됩니다.



### Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration (https://arxiv.org/abs/2509.21946)
Comments:
          9 pages

- **What's New**: 본 논문은 태국의 정치적 맥락에서  정치적 입장 탐지에 있어 대규모 언어 모델(LLMs)의 편향 문제를 다룹니다. 특히, 정치적 대화가 암시적이고 감정적으로 얽힌 환경에서 LLMs들이 보여주는 비체계적 편향, 즉 감정 누출(sentiment leakage) 및 특정 정치적 인물에 대한 편애를 실질적으로 감소시키는 프레임워크인 ThaiFACTUAL을 제안합니다. 이는 모델 튜닝 없이도 정치적 편향을 줄일 수 있는 경량의 방법론입니다.

- **Technical Details**: ThaiFACTUAL는 반사실적 데이터 증강(counterfactual data augmentation) 및 근거 기반(supervision based on rationale) 접근 방법을 활용하여 감정과 입장을 분리합니다. 연구에서는 LLM들의 편향을 정량적으로 분석하기 위해 회수 표준 편차(RStd) 메트릭을 채택하여, 주요 정치 인물에 대한 입장 분류 성능을 평가하였습니다. 이 프레임워크는 기존에 요구되었던 모델 파라미터 접근 없이도 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ThaiFACTUAL은 여러 LLM에서 무작위 상관관계를 크게 줄이고, 제로샷 일반화(zero-shot generalization)를 향상시키며, 공정성을 개선하는 것으로 나타났습니다. 특히, 태국 정치적 대화를 다룰 때 발생하는 감정의 복잡성과 문화적 뉘앙스를 고려하여, 더 나은 성능을 보이는 것을 입증하였습니다. 이러한 결과는 정치적으로 민감한 상황에서의 AI 모델의 신뢰성을 높이는 데 기여할 것입니다.



### Unveiling Many Faces of Surrogate Models for Configuration Tuning: A Fitness Landscape Analysis Perspectiv (https://arxiv.org/abs/2509.21945)
Comments:
          This paper is under review

- **What's New**: 이 논문은 구성 설정(configuration tuning)을 위한 대리 모델(surrogate model)의 역할을 체계적으로 탐색하고 논의한 최초의 연구이다. 저자들은 대리 모델의 유용성을 평가하기 위해 정확성(accuracy) 대신 새로운 이론인 fitness landscape 분석을 제안한다. 이 접근 방식은 관련된 질문들에 대한 답을 제공하며, 구성 조정에서 대리 모델의 수많은 얼굴을 조명한다.

- **Technical Details**: 연구에서는 27,000건 이상의 사례를 포함한 광범위한 실증 연구를 수행하여 모델 유용성 평가를 위한 새로운 틀을 개발하였다. Model4Tune이라는 자동화된 예측 도구를 제안하고, 이는 비싼 튜너 프로파일링 없이도 최적의 모델-튜너 쌍을 추정할 수 있다. 실험 결과, Model4Tune은 무작위 추측(random guessing)보다 79%-82% 더 나은 성능을 보였다.

- **Performance Highlights**: 연구 결과는 Model4Tune이 구성 조정에서 매우 유용한 도구가 될 수 있음을 입증한다. 이 도구는 실무자들이 가장 유용한 모델을 평가하는 데 도움을 줄 수 있으며, 향후 연구 방향에 대한 통찰을 제공한다. 대리 모델에 대한 깊이 있는 이해는 성능 최적화의 새로운 가능성을 제시한다.



### SemanticControl: A Training-Free Approach for Handling Loosely Aligned Visual Conditions in ControlN (https://arxiv.org/abs/2509.21938)
Comments:
          BMVC 2025

- **What's New**: 이번 연구에서는 SemanticControl이라는 새로운 방법을 제안하여 기존의 ControlNet에서 발생하는 텍스트 프롬프트와 시각적 조건의 불일치를 해결하고자 합니다. 이 방법은 훈련 없이도 비선형적이지만 의미적으로 관련이 있는 시각적 조건을 효과적으로 활용할 수 있습니다. 또한, 불일치가 있는 경우에도 텍스트 가이드를 강화하면서 시각적 조건의 영향을 조절하여 생성 품질을 향상시킵니다.

- **Technical Details**: SemanticControl의 핵심 아이디어는 먼저 대체 프롬프트를 기반으로 보조 노이즈 제거 과정을 실행하여 유용한 주의 마스크(attention masks)를 추출하는 것입니다. 이후 이러한 마스크를 사용하여 텍스트 프롬프트의 노이즈 제거 과정에서 텍스트의 의미에 맞는 부분을 강화합니다. 이를 통해, SemanticControl은 기존의 ControlNet보다 불일치한 조건 하에서도 더욱 높은 품질의 이미지를 생성할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, SemanticControl은 다양한 입력 조건에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 특히 깊이 맵, 엣지 맵 및 인간 골격과 같은 다양한 시각적 조건 하에서도 효과적임이 입증되었습니다. 또한, 인간 평가에서도 SemanticControl이 더 높은 선호도를 얻어 내며 우수한 결과를 나타냈습니다.



### Why Chain of Thought Fails in Clinical Text Understanding (https://arxiv.org/abs/2509.21933)
- **What's New**: 이 연구는 임상 텍스트 이해를 위한 CoT(Chain-of-Thought) 프롬프팅의 효과성에 대한 대규모 체계적 연구를 처음으로 제시합니다. 95개의 최신 대형 언어 모델(LLM)을 87개의 실제 임상 작업에 대해 평가하였으며, 이를 통해 CoT가 임상 텍스트 작업에서 성능을 일관되게 저하시킬 수 있다는 점이 드러났습니다. 특히, 모델의 성능 저하는 약한 모델에서 더 두드러지는 경향을 보였습니다.

- **Technical Details**: 연구에서는 9개 언어와 8개 작업 종류에 걸쳐 임상 텍스트 작업을 평가하며, 두 가지 프롬프팅 전략인 제로샷(zero-shot)과 CoT를 활용하였습니다. LLM 평가에 있어 정확성과 추론 길이, 임상 개념 정렬, 그리고 오류 프로파일을 보다 세밀하게 분석하였습니다. 특히, CoT가 일관되게 정확도를 저하시키는 방식과 그 이유를 메커니즘 분석을 통해 조명하였습니다.

- **Performance Highlights**: 연구의 주요 발견으로는 CoT 프롬프팅이 명백한 투명성을 증가시키지만, 임상 텍스트 이해에서는 정량적으로 성능 저하를 초래한다는 점이 포함됩니다. 임상 사례 분석 결과, CoT의 실패가 더 긴 추론 연결고리와 관련이 있으며, 잘못된 답변과 연관된 언어적 특성도 확인되었습니다. 마지막으로, 오류 분류 체계를 개발하여 LLM의 임상 활용에 대한 안전한 지침을 제공하고 있습니다.



### SAGE: Scene Graph-Aware Guidance and Execution for Long-Horizon Manipulation Tasks (https://arxiv.org/abs/2509.21928)
- **What's New**: 이 논문에서는 SAGE라는 새로운 프레임워크를 제안하여 복잡한 현실 세계에서 로봇이 긴 시간의 조작 작업을 수행하는 문제를 해결하고자 합니다. SAGE는 scene graph를 활용하여 작업의 의미적 표현을 구조적으로 제작하고, 이를 통해 동작 계획과 시각적 제어를 효과적으로 연결합니다. 기존의 전통적인 방법과 LLM 기반 접근 방식의 한계를 극복하고, 새로운 하위 목표 이미지를 생성할 수 있는 제어 가능한 방법론을 제공합니다.

- **Technical Details**: SAGE는 두 가지 주요 구성 요소로 이루어져 있습니다: 첫 번째는 VLM과 LLM을 사용하여 환경을 분석하고 물리적 장면 상태 전이 시퀀스에 대해 추론하는 scene graph 기반 작업 계획기입니다. 두 번째 구성 요소는 각 목적의 하위 목표 그래프를 이미지 인페인팅(image inpainting) 및 합성을 통해 대응하는 이미지로 변환하는 분리된 구조적 이미지 편집 파이프라인입니다. 이러한 구조적 접근법은 시각적-운동 제어를 위한 신뢰할 수 있는 목표 표현을 제공합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면 SAGE는 다양한 긴 시간 조작 작업에서 최첨단 성능을 달성했습니다. 특히, SAGE는 시간적 실행 순서와 객체 간의 공간적 관계가 독특한 여러 작업에서 효과적으로 작동했습니다. 이 프레임워크는 여러 가지 새로운 작업 유형에 대해 높은 질의 이미지를 생성할 수 있는 능력을 보여줍니다.



### Generation Properties of Stochastic Interpolation under Finite Training S (https://arxiv.org/abs/2509.21925)
- **What's New**: 이 논문은 유한한 훈련 인구 하에서 생성 모델의 이론적 행동을 조사합니다. Stochastic interpolation generative framework를 활용하여, 훈련 샘플이 유한할 때 최적 속도 필드와 score function에 대한 닫힌 형식을 유도합니다. 이는 훈련 샘플을 완전히 재현하는 결정론적 생성(process)과 훈련 샘플에 가우시안 노이즈가 추가된 확률론적 생성 과정의 차이를 보여줍니다.

- **Technical Details**: 생성 모델의 학습 부족(underfitting) 및 과적합(overfitting)과 관련된 이론적 정의를 도입하며, 추정 오류가 있는 경우의 생성 성능에 대한 분석을 수행합니다. Stochastic interpolation 모델의 속도 필드와 score function은 Fokker–Planck 방정식과 연속 방정식을 기반으로 정의됩니다. 이 모델은 결정론적 및 확률론적 생성을 모두 지원하는 생성 모델의 일반 클래스에 해당합니다.

- **Performance Highlights**: MNIST, CIFAR-10, Imagenet와 같은 데이터셋에 대한 다운스트림 실험을 통해 이론적 발견을 검증합니다. 연구 결과는 훈련 샘플에 균일한 노이즈와 가우시안 노이즈가 결합되어 섞인 형태로 생성되는 것을 보여주며, 이는 생성 모델의 메모리(Memorization) 및 일반화(Generalization) 행동에 대한 깊은 통찰을 제공합니다.



### EqDiff-CT: Equivariant Conditional Diffusion model for CT Image Synthesis from CBC (https://arxiv.org/abs/2509.21913)
Comments:
          12 pages, 8 figures, 3 tables, submitted to IEEE Transactions on Radiation and Plasma Medical Sciences

- **What's New**: 이번 연구에서는 Cone-beam computed tomography (CBCT)의 한계를 극복하기 위해 새로운 확산 기반 조건 생성 모델인 EqDiff-CT를 제안합니다. 기존의 CT는 이미지 품질이 뛰어나지만 실시간으로 획득할 수 없고, 치료 중 해부학적 변화 포착에 한계가 있습니다. EqDiff-CT는 CBCT로부터 고품질 CT 이미지를 합성하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: EqDiff-CT는 Denoising Diffusion Probabilistic Model (DDPM)을 이용하여 노이즈를 순차적으로 주입하고 잠재 표현(latent representations)을 학습합니다. Group-equivariant conditional U-Net 백본이 도입되어 회전 변환 불변성(rotational equivariance)을 유지하며 세부 구조를 보존하도록 설계되었습니다. 이 모델은 SynthRAD2025 데이터셋으로 훈련 및 검증되었습니다.

- **Performance Highlights**: EqDiff-CT는 CycleGAN 및 DDPM과 같은 기존의 방법들과 비교했을 때 구조적 충실도(structural fidelity)와 HU 정확도에서 상당한 개선을 보여주었습니다. 시각적으로도 향상된 복원력과 날카로운 연조직 경계, 현실적인 뼈 재구성을 확인하였습니다. 이는 CBCT 방식의 치료계획 및 용량 계산에 대한 clinical confidence(임상적 신뢰도)를 높이는 데 기여할 것으로 기대됩니다.



### AutoSCORE: Enhancing Automated Scoring with Multi-Agent Large Language Models via Structured Component Recognition (https://arxiv.org/abs/2509.21910)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문에서는 AutoSCORE라는 다중 에이전트 LLM 프레임워크를 제안하여 자동 채점 시스템의 한계를 극복하고자 합니다. AutoSCORE는 학생의 응답에서 채점 기준의 관련 구성 요소를 추출하고 이를 구조화된 형태로 인코딩하여 최종 점수를 부여합니다. 이 과정은 루브릭을 기반으로 하여 인간과 유사한 채점 과정을 따르며 해석 가능성과 신뢰성을 향상합니다.

- **Technical Details**: AutoSCORE는 두 개의 핵심 에이전트로 구성되어 있습니다. 첫 번째는 Scoring Rubric Component Extraction Agent로, 학생의 응답에서 관련 구성요소를 식별하여 구조화된 출력을 생성합니다. 두 번째는 Scoring Agent로, 이러한 표현을 바탕으로 최종 점수를 부여하며, 선택적으로 검증 및 피드백 에이전트를 포함할 수 있어 품질 관리를 강화합니다.

- **Performance Highlights**: AutoSCORE는 ASAP 벤치마크의 네 가지 데이터 세트에서 평가되었으며, 여러 작업과 루브릭에 걸쳐 채점 정확도, 인간-기계 간의 합의, 오류 메트릭에서 일관되게 향상된 성과를 보였습니다. 특히 복잡한 다차원 루브릭에서 두드러진 이익을 보여주었고, 상대적으로 작은 LLM에서도 큰 성과를 달성했습니다.



### A Large-Scale Dataset and Citation Intent Classification in Turkish with LLMs (https://arxiv.org/abs/2509.21907)
Comments:
          Submitted to IEEE UBMK 2025 International Conference on Computer Science and Engineering

- **What's New**: 이번 연구에서는 터키어의 인용 의도를 분류하기 위한 대규모 데이터셋과 체계적인 분류 방법론을 개발했습니다. 새롭게 공개된 이 데이터셋은 2,650개의 주석이 달린 컴퓨터 과학 논문 샘플로 구성되어 있습니다. 기존 In-Context Learning (ICL) 방식의 한계를 극복하기 위해 자동화된 프롬프트 최적화 프레임워크인 DSPy를 사용하여 인용 의도를 효과적으로 분류합니다.

- **Technical Details**: 연구는 인용 문장의 추출을 위해 GROBID 라이브러리를 활용한 CEX 모듈을 사용합니다. 인용 의도 분류는 Web of Science (WoS) 플랫폼의 다섯 가지 카테고리 체계를 기반으로 하여 진행되며, 전문적인 주석 작성을 위한 웹 인터페이스가 개발되었습니다. 최종적으로 여러 모델의 출력을 집계하는 스택된 앙상블 기법을 통해 91.3%의 정확도를 달성하였습니다.

- **Performance Highlights**: 이 연구는 터키어 NLP 커뮤니티에 강력한 기초 데이터셋과 분류 프레임워크를 제공하여, 향후 정성적 인용 연구의 기반을 마련합니다. 특히, DSPy 프레임워크를 통한 자동화된 프롬프트 최적화 방식은 기존 수동 방식보다 안정적이고 효율적인 성능 향상을 보여주었습니다. 이러한 접근은 과학적 발견 가속화를 위한 지능형 학술 도구 개발에 기여할 것입니다.



### Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts (https://arxiv.org/abs/2509.21892)
- **What's New**: 이번 연구에서는 Elastic Mixture-of-Experts (EMoE)라는 새로운 훈련 프레임워크를 소개했습니다. 전통적인 Mixture-of-Experts (MoE) 모델에서는 훈련 및 추론 시 고정된 수의 전문가(expert)를 사용하는 반면, EMoE는 추론 시 활성화된 전문가 수를 유연하게 조정할 수 있어 성능을 향상시킬 수 있습니다. 이는 학습된 전문가 간의 협력이 부족해 성능 저하를 초래하는 문제를 해결하기 위한 접근으로, 스토카스틱(co-activation) 샘플링 기법과 계층적 라우터 손실(hierarchical router loss)을 도입하였습니다.

- **Technical Details**: EMoE의 주요 기술적 요소는 스토카스틱(co-activation) 샘플링과 계층적 라우터 손실입니다. 스토카스틱 샘플링 기법을 통해 다양한 전문가 조합을 훈련 기간 동안 균형 있게 활성화하여 전문가 사이의 협력 능력을 향상시키는 동시에 훈련 부담을 최소화합니다. 또한 계층적 라우터 손실은 KL 발산을 활용해 라우터의 확률 분포를 명확한 계층 체계를 만들어내며, 이를 통해 각 입력 토큰에 대해 최상의 전문가 세트를 선별할 수 있도록 합니다.

- **Performance Highlights**: EMoE를 적용한 결과, 표준 Top-k 모델과 비교하여 훨씬 더 넓은 범위의 활성화된 전문가 수를 지원하며, 이는 2-3배까지 확대될 수 있음을 보여주었습니다. 다양한 컴퓨팅 예산 아래에서 EMoE는 지속적으로 기준선을 초과하는 성능을 발휘하여, 다양한 상황에서 뛰어난 유용성을 자랑합니다. 또한, 스토카스틱 샘플링 및 계층적 라우터 손실이 EMoE의 효과에 필수적임을 실험적으로 입증하였습니다.



### You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors (https://arxiv.org/abs/2509.21884)
Comments:
          29 pages, 10 tables, 6figures, accepted by CCS 25

- **What's New**: 본 논문은 최신 대형 언어 모델(LLM)의 잠재적인 시스템 프롬프트 리크(Leak) 공격에 대한 새로운 접근법을 제안합니다. 기존 모델들은 공격 패턴을 인지할 경우 시스템 프롬프트를 반복하지 않도록 설계되었지만, 여전히 예기치 않은 공격 방식에 취약합니다. 이에 본 논문은 SysVec라는 새로운 방법을 통해 시스템 프롬프트를 텍스트(context) 대신 내부 표현 벡터로 인코딩하여 무단 공개를 최소화하고 LLM의 언어 능력을 유지합니다.

- **Technical Details**: SysVec는 시스템 프롬프트를 LLM의 내적 벡터 공간에 변환하여, 프롬프트 리크 공격으로부터 보호합니다. 이 방법은 시스템 프롬프트가 더 이상 텍스트 출력을 통해 노출되거나 반복되지 않도록 설계되었습니다. 또한, 이 방법은 추론 오버헤드를 줄이고 긴 입력을 처리하는 능력을 향상시키며 메모리 관리를 세밀하게 제어할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, SysVec는 프롬프트 리크 공격을 효과적으로 완화시키고 LLM의 기능 무결성을 유지하며 장기 맥락에서의 망각 문제를 완화하는 데 기여합니다. 이 연구는 LLM의 보안성을 강화하고, 특정 사용자 요구에 맞춰 모델의 일반적인 지침 따르기 능력을 개선할 수 있음을 보여줍니다.



### Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards (https://arxiv.org/abs/2509.21882)
- **What's New**: 본 논문은 RLVR(Reinforcement Learning with Verifiable Rewards)가 대형 언어 모델의 수학, 코드 및 기타 구조화된 작업에서의 성능 향상에 기여하는지를 검토합니다. 연구자들은 RLVR가 진정한 추론 능력을 향상시키는지, 혹은 기존 모델이 알고 있는 행동 중 선택성을 키우는 것인지에 대한 의문을 제기합니다. 특별히, RLVR로 얻은 이익이 실제로 과장되었을 수 있음을 강조하며, 이를 평가하는 과정에서 발생할 수 있는 여러 문제들을 다룹니다.

- **Technical Details**: RLVR은 프로그램의 유닛 테스트, 수학에 대한 정확한 숫자 또는 문자열 일치, 그리고 인용의 검색 기반 체크와 같은 자동으로 계산 가능한 신호에 대해 최적화하여 작동합니다. 연구에서는 RLVR의 적용이 얼마나 신뢰할 수 있는지, 그리고 사전 훈련된 모델과 비교 시 RLVR 모델에서 발견되는 여러 주요 차이점들이 실제로 측정 디자인에 의해 왜곡되는지에 대해 훌륭하게 설명하고 있습니다. 특히, RLVR 훈련에서 어떠한 '세금(tax)'이 발생하는지를 분석하고 이를 토대로 새로운 평가 프로토콜을 제안합니다.

- **Performance Highlights**: 연구 결과에 따르면, RLVR을 통해 얻은 성과는 신뢰성과 정확성을 우선시하여 보다 신뢰할 수 있는 추론 결과를 제공할 수 있습니다. RLVR의 활용을 통해 여러 사례에서 기존의 결론이 수정될 수 있음을 보여줍니다. 이러한 접근 방식은 개선 격차를 줄이고, RLVR의 실용성을 강조하는 동시에 신뢰성 및 안전성을 보장하는 함의를 가집니다.



### No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping (https://arxiv.org/abs/2509.21880)
- **What's New**: 이 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR) 프레임워크에서 zero-variance prompts의 유용성을 주장합니다. 기존의 기법들이 서로 다른 보상을 갖는 입력에만 초점을 맞춘 반면, 저자들은 모든 응답이 동일한 보상을 받는 경우에도 유의미한 피드백을 제공할 수 있다는 점을 강조합니다. 이를 통해 RL-ZVP라는 새로운 알고리즘을 제안하며, 이는 token-level 정보를 활용하여 정책 최적화를 개선하는 데 집중합니다.

- **Technical Details**: RL-ZVP는 zero-variance prompts에서 학습 신호를 추출하기 위해 두 가지 주요 직관에 기반합니다: (i) 같은 그룹 내에서 잘못된 응답이 없더라도 정확한 응답에 대해 보상을 주어야 하며, (ii) 보상 또는 패널티의 정도는 샘플링된 토큰의 특성에 따라 결정됩니다. 논문에서는 정확성과 정밀도를 높이는 두 가지 속성이 방향성과 크기라는 점을 강조하고, 이를 통해 트레이닝의 효과성을 크게 향상시킵니다.

- **Performance Highlights**: RL-ZVP는 여섯 개의 수학 추론 기준에서 GRPO보다 평균 4.00점, Pass@8에서 4.28점의 정확도를 개선하였습니다. 특히, RL-ZVP는 AIME25에서 최대 8.66점, OlympiadBench에서 7.77점의 성능 향상을 기록하며, zero-variance prompts를 필터링하는 다른 기준 모델들을 꾸준히 초월하는 결과를 보였습니다. 이러한 결과는 zero-variance prompts가 RLVR에서 학습 신호의 귀중한 자원이 될 수 있음을 입증합니다.



### Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization (https://arxiv.org/abs/2509.21871)
- **What's New**: 이번 논문에서는 Aes-R1이라는 새로운 체계를 제안하여 이미지 미적 평가(IAA)를 위한 강력한 미적 추론 프레임워크를 구축했습니다. 기존의 다중 형태 대형 언어 모델(MLLM)들이 미적 판단을 수행하는 데 어려움을 겪는 이유로, 데이터 부족과 주관적인 미적 판단의 본질을 언급했습니다. Aes-R1은 강화 학습을 통해 이 문제를 해결하며, 더 나아가 설명 가능한 미적 점수를 생성할 수 있도록 지원합니다.

- **Technical Details**: Aes-R1는 AesCoT라는 파이프라인을 통해 고품질의 미적 추론 데이터를 생성하고 필터링합니다. 이 방법은 구조화된 설명 생성을 모델에 가르친 후, Relative-Absolute Policy Optimization (RAPO)이라는 새로운 강화 학습 알고리즘을 사용하여 미적 평가 점수와 상대적 순위를 최적화합니다. 이를 통해 모델은 이미지의 미적 점수를 정확히 평가할 수 있는 능력을 얻게 됩니다.

- **Performance Highlights**: Aes-R1은 기존 최첨단 모델 대비 47.9% 및 34.8%의 PLCC/SRCC 향상을 이루며, 단 15K의 학습 데이터로도 우수한 성능을 보여줍니다. 다양한 IAA 벤치마크에서 최첨단 성능을 달성하였고, 제한된 지도식 아래에서도 견고한 일반화를 지원하는데 필요한 분석을 수행했습니다. 논문에 제시된 실험 결과와 데이터 파이프라인 AesCoT의 개발은 향후 IAA 연구에 큰 기여를 할 것입니다.



### Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations (https://arxiv.org/abs/2509.21870)
Comments:
          This manuscript has been submitted to IEEE Journal of Selected Topics in Signal Processing (JSTSP) for review. Until the moment I submitted the manuscript to arXiv, we haven't received any review comments from JSTSP

- **What's New**: 본 논문은 Low-Rank Adaptation (LoRA)의 비선형 확장인 LoRAN을 제안합니다. LoRAN은 파라미터 수를 증가시키지 않으면서도 저랭크 업데이트에 경량 변환을 적용하여 표현 능력을 향상시킵니다. 추가적으로, Sinter라는 새로운 sine 기반 활성화 함수를 도입하여 구조적 왜곡을 추가하면서도 파라미터 수를 유지합니다. 실험 결과, LoRAN은 QLoRA보다 일관된 성능 향상을 보여 주었습니다.

- **Technical Details**: LoRAN은 LoRA의 저랭크 프로젝션에 비선형 함수를 적용하여 가중치 업데이트의 표현 능력을 강화합니다. Sinter는 제어된 진동성 perturbation을 추가하여 모델의 표현성과 안정성을 동시에 개선합니다. 논문은 다양한 자연어 처리 작업에서 LoRAN의 성능을 검증하며, 기존의 Sigmoid, ReLU, Tanh와 같은 활성화 함수들에 비해 Sinter의 장점을 입증합니다.

- **Performance Highlights**: LoRAN은 리소스가 제한된 환경에서도 기존 LoRA보다 뛰어난 성능을 보여주며, 특히 고차원 업데이트를 더 잘 근사할 수 있습니다. 이 모델은 다양한 기본 모델에 대해 높은 유연성과 일반화 능력을 발휘하였으며, 전체적인 성능이 충분한 Fine Tuning에 근접하게 개선되었습니다. 실험 범위가 확장되어 여러 작업에서 LoRAN과 Sinter의 효과가 더욱 명확히 드러났습니다.



### Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration (https://arxiv.org/abs/2509.21848)
Comments:
          Preprint

- **What's New**: 이 논문은 모델 비의존적인(long context modeling) 접근 방식을 통해 멀티-에이전트 시스템이 대규모 언어 모델의 컨텍스트 윈도우보다 더 긴 입력을 처리할 수 있도록 하는 방법을 제시합니다. 저자들은 복잡한 협업 전략과 프롬프트 엔지니어링에 의존하지 않고 정보 이론적 압축 목표를 바탕으로 하는 프레임워크(Graph of Agents, GoA)를 개발했습니다.

- **Technical Details**: GoA는 입력에 따라 동적으로 협업 구조를 구성하여 정보를 최대화합니다. 이 시스템은 텍스트 조각을 의미적 유사성을 기준으로 클러스터화하고, 각 클러스터 내에서 쿼리와 가장 관련성 높은 텍스트 조각을 선택하는 방식으로 작동합니다. 이러한 정보 이론적 목표에 의해 유도된 적응형 그래프 구조는 다양한 작업에 걸쳐 일반화가 가능하도록 만듭니다.

- **Performance Highlights**: GoA는 Llama 3.1 8B 및 Qwen3 8B 모델을 이용해 총 여섯 개의 문서 질문 응답 벤치마크에서 F1 점수를 각각 5.7% 및 16.35% 개선했습니다. 또한, 2K 컨텍스트 윈도를 사용하여 128K 컨텍스트 윈도인 Llama 3.1 8B 모델을 LongBench에서 초과하는 성능을 보이며 효과적인 컨텍스트 길이를 극적으로 증가시켰습니다.



### Beyond Johnson-Lindenstrauss: Uniform Bounds for Sketched Bilinear Forms (https://arxiv.org/abs/2509.21847)
- **What's New**: 이번 연구에서는 벡터 및 행렬의 스케치된 내적(sketched inner products)에 대한 균일 경계(uniform bounds)를 재정립했습니다. 이는 머신러닝 및 확률 알고리즘에서 중요한 여러 이론적 결과들에 적용될 수 있는 새로운 프레임워크를 제공합니다. 특히, 스케치된 이변량 형태(sketched bilinear forms)에 대한 기존의 제약이 있는 경우가 많아 이를 해결하기 위한 새로운 기술을 도입했습니다.

- **Technical Details**: 연구의 핵심은 일반 체이닝(generic chaining)에 의존하여, 쌍(pair) 집합에 대한 상한(suprema)을 처리하는 새로운 접근법을 제공하는 것입니다. 또한, T개의 독립적인 스케치 매트릭스의 합이 포함된 경우에 대해서도 결과를 확장하였으며, 이 경우 편차(deviation)는 \sqrt{T}로 스케일링됨을 보였습니다. 이 통합 분석은 J-L lemma와 같은 잘 알려진 결과를 특수한 사례로 복원하면서 RIP 유형 보장을 확장합니다.

- **Performance Highlights**: 이 연구는 스케치된 Federated Learning 알고리즘의 수렴 경계를 개선하는 동시에, 스케치된 그래디언트 압축으로 인해 발생하는 교차 항(cross terms) 문제를 해결합니다. 또한, 작업(action) 및 매개변수(parameter) 집합의 기하학적 복잡성에 따라 더욱 정밀한 후회(bound) 경계를 가지는 스케치 변형 밴딧 알고리즘(bandit algorithms)을 설계했습니다.



### Can Large Language Models Autoformalize Kinematics? (https://arxiv.org/abs/2509.21840)
- **What's New**: 이번 논문은 로봇 및 자율 주행차와 같은 사이버 물리 시스템(CPS)의 제어 결정을 신뢰성 있게 추론하기 위해 대형 언어 모델(LLMs)을 사용하여 자동으로 형식을 작성할 수 있는 가능성을 탐구합니다. LLM은 물리 모델링 작업을 자동화하는 데 활용될 수 있으며, 20개의 물리학 문제로 구성된 벤치마크를 통해 그 성능을 평가합니다. 이 연구는 LLM이 자연어를 기반으로 물리적 모델을 자동으로 형식화할 수 있는 가능성을 제시하며, 사이버 물리 시스템의 안전성을 높일 수 있는 첫 번째 정량적 기준을 제공합니다.

- **Technical Details**: 연구에서 제안한 접근 방식은 LLM이 자연어로 설명된 물리 문제를 기반으로 차별적 게임 논리(dGL) 모델을 생성하는 것입니다. 각 문제에서 LLM은 물체의 운동을 설명하는 자연어 설명을 입력받고, 생성된 모델은 구문 검사 및 의미 평가를 통해 검증됩니다. 이 과정에서 LLM은 최초 제안에서 제공된 5개의 시도 중 최적의 선택을 통해 70%의 성공률을 달성합니다. 그러나 가장 복잡한 문제에서 낙제한 케이스가 발견되며, 이는 검사기(checker)의 한계로 인한 것입니다.

- **Performance Highlights**: 벤치마크 평가 결과, LLM의 자동 형식화 능력은 70%의 상위 정확도를 보여주며 이는 CPS와 같은 복잡하고 안전이 중요한 시스템의 공식화 작업에 있습니다. 향후 연구에서는 실패 사례를 분석하고 LLM의 성능을 개선할 방향을 모색하고 있습니다. 이러한 자동 형식화의 가능성은 CPS의 안정성과 설계 용이성을 증가시킬 수 있으며, 모든 모듈 구성 요소에 대한 공식적 모델링이 가능하게 할 것입니다.



### DiTraj: training-free trajectory control for video diffusion transformer (https://arxiv.org/abs/2509.21839)
- **What's New**: 새로운 Diffusion Transformers (DiT) 기반 비디오 생성 모델인 DiTraj를 소개합니다. DiTraj는 텍스트-비디오 생성에서의 궤적 제어를 위해 특별히 설계된 교육이 필요 없는 프레임워크입니다. 이 모델은 객체의 움직임을 제어하기 위해 전경-배경 분리 가이드를 적용하여 생성 과정에서 사용자 제공 프롬프트를 효과적으로 활용합니다.

- **Technical Details**: DiTraj는 Large Language Model (LLM)을 사용하여 사용자 프롬프트를 전경과 배경 프롬프트로 변환합니다. 또, 3D 풀 어텐션과 위치 임베딩 간의 밀접한 관계를 분석하여, inter-frame Spatial-Temporal Decoupled 3D-RoPE (STD-RoPE)라는 혁신적인 방법을 제안합니다. STD-RoPE는 전경 토큰의 위치 임베딩을 조정하여 서로 다른 프레임 간의 어텐션을 개선하고, 객체의 궤적 제어를 강화합니다.

- **Performance Highlights**: 실험 결과, DiTraj는 기존 모델들에 비해 비디오 품질과 궤적 제어 가능성 모두에서 우수한 성능을 보여줍니다. 특히, DiTraj는 학습 과정이 필요 없으며, 대부분의 DiT 기반 비디오 생성 모델에 쉽게 적응할 수 있습니다. 이러한 접근 방식은 사용자가 제공한 텍스트 설명에 따라 정확한 비디오 생성을 가능하게 합니다.



### ChaosNexus: A Foundation Model for Universal Chaotic System Forecasting with Multi-scale Representations (https://arxiv.org/abs/2509.21802)
- **What's New**: 이번 연구에서는 ChaosNexus라는 새로운 기초 모델을 소개합니다. 이 모델은 다양한 혼돈 시스템에 대한 예측 성능을 높이는 데 중점을 두고 있습니다. ChaosNexus는 미세한 다중 스케일 아키텍처인 ScaleFormer와 Mixture-of-Experts 레이어를 활용하여 일반적인 패턴을 캡처하고 시스템별 행동을 식별합니다.

- **Technical Details**: ChaosNexus는 약 2만 개의 시뮬레이션된 혼돈 시스템을 기반으로 사전 훈련되었습니다. 이 모델의 핵심은 여러 스케일을 효과적으로 처리하기 위한 U-Net inspired Transformer 구조인 ScaleFormer입니다. 각 Transformer 블록은 Mixture-of-Experts(MoE) 레이어로 강화되어 서로 다른 시스템 레짐을 위한 전문화된 매개변수를 할당합니다.

- **Performance Highlights**: ChaosNexus는 제로샷(zero-shot) 예측에서 새로운 최첨단 성능을 창출하며, 긴 기간의 끌림 통계(fidelity)를 40.55% 향상시켰습니다. 실제 기상 예측에서는 1도 이하의 제로샷 평균 오차(MAE)를 달성하여 강력한 기준 모델을 초과했습니다. 이 모델의 실험 결과는 데이터 수량보다 시스템의 다양성이 더 중요한 일반화의 원동력임을 보여줍니다.



### Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignmen (https://arxiv.org/abs/2509.21798)
Comments:
          Under review on ICLR 2026;Work in progress;

- **What's New**: 본 연구에서는 문화적 인식(cultural awareness)을 평가하기 위한 Cultural Awareness Reward modeling Benchmark (CARB)를 제안합니다. 이 벤치마크는 10개의 다양한 문화와 4개의 문화적 분야를 아우릅니다. 현재의 보상 모델(reward models, RMs)의 한계를 드러내며, 문화적 특성을 효과적으로 반영하는 새로운 평가 기준을 마련하는 데 기여합니다.

- **Technical Details**: CARB는 문화적 지식, 가치, 안전 및 언어학과 같은 4가지 주요 도메인에서 10개의 문화적 특성을 포괄하도록 설계되었습니다. 연구에서는 각 문화에 대해 인간이 선별한 질문을 기반으로 생성된 응답들을 활용하여 8,576개의 고품질 BoN 세트를 작성했습니다. Think-as-Locals라는 방법을 통해 보상 모델이 표면적 특성 대신 진정한 문화적 뉘앙스를 이해하도록 유도합니다.

- **Performance Highlights**: 실험 결과, CARB에서의 성과는 다양한 문화적 정렬 작업에서의 보상 모델 성능과 긍정적인 상관관계를 나타냈습니다. 현재의 보상 모델들은 문화적 인식 측면에서 제한적이며, 겉보기에 기반한 스퍼리어스 상관관계를 보이는 경향이 있습니다. 본 연구에서 제안한 방법은 이러한 문제들을 감소시키고, 문화 인식 능력을 강화하는 데 효과적임을 입증했습니다.



### FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning (https://arxiv.org/abs/2509.21792)
Comments:
          Submitted to ICLR 2026

- **What's New**: 이 논문은 Group Relative Policy Optimization (GRPO)를 통해 대형 언어 모델(LLM)의 추론 능력을 향상시키는 새로운 방법을 제안합니다. 특히, 고속 훈련을 위한 concurrency-aware speculative decoding 프레임워크를 도입하여 레이턴시를 줄이고 성능을 최적화하는 데 중점을 두었습니다. 이 방법은 훈련 과정 중 발생하는 분포 변화 문제를 해결하기 위해 온라인 드래프트 학습 메커니즘을 도입하여 모형의 적합도를 적절히 증진시킵니다.

- **Technical Details**: 본 연구는 GRPO에서의 응답 생성 단계의 병목 현상을 해결하기 위한 전략으로, 실시간 동시성(concurrency) 수준에 맞춰 드래프트 및 검증 전략을 동적으로 조정합니다. 또한, 드래프트 모델은 목표 모델의 피드백 신호를 지속적으로 수용하여 적응함으로써 점진적으로 수용된 토큰의 평균 길이를 증가시킵니다. 이 과정에서, 기존의 모델에 비해 2.35배에서 2.72배의 전반적인 속도 향상을 달성할 수 있었습니다.

- **Performance Highlights**: 다양한 수학적 추론 데이터셋 및 모델에서 실시된 실험 결과, 제안한 방법이 기존 접근 방식에 비해 효율성을 현저히 향상시킨 것으로 나타납니다. 특히, Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct 모델을 사용한 실험에서 고속 추론을 가능하게 하며, 하드웨어의 병렬 처리 성능을 최적화했습니다. 이러한 결과는 GRPO 훈련 과정에서 고속 유지와 성공적인 응답 생성을 위한 효과적인 해결책을 제시합니다.



### Unbiased Binning: Fairness-aware Attribute Representation (https://arxiv.org/abs/2509.21785)
- **What's New**: 이번 논문에서는 데이터셋을 공유하기 전에 원시 특성을 버킷화하여 속성 표현을 만드는 과정에서 발생하는 편향 문제를 다룹니다. 특히, 다양한 버킷 간의 그룹 형평성을 만족하는 최적의 디스크리타이제이션(discretization)을 찾는 unbiased binning 문제를 제안합니다. 연구진은 경계 후보의 소집합을 정의하고, 이 후보에서 경계를 선택해야 한다는 것을 증명했습니다.

- **Technical Details**: 논문은 unbiased binning 문제를 해결하기 위해 동적 프로그래밍 알고리즘을 개발하였고, 이는 경계 후보를 기반으로 합니다. 그러나 이러한 동적 프로그래밍 알고리즘은 큰 데이터셋에서는 확장성이 떨어지므로, 지역 탐색(local search) 기반의 epsilon-biased binning 문제를 제안합니다. 이때 핵심 요소는 D&C(divide-and-conquer) 알고리즘으로, 근사 최적 해를 near-linear 시간 내에 찾아냅니다.

- **Performance Highlights**: D&C 알고리즘은 모든 경우의 valid solution을 찾을 수 있다고 증명되어 있습니다. 그 후, LS 알고리즘은 D&C 솔루션을 상한으로 사용하여 최적 솔루션을 찾기 위한 지역 탐색을 시작합니다. 연구의 결과는 그룹 비율에 대한 작은 편향을 허용하면서도 효율적으로 binning 문제를 처리할 수 있다는 것을 보여줍니다.



### Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction (https://arxiv.org/abs/2509.21778)
- **What's New**: 이번 연구에서는 PRDNet이라는 새로운 모델을 소개하며, 이 모델은 전통적인 graph representations 외에 reciprocal-space diffraction 정보를 활용하여 결정(structure)의 특성을 예측하는 방식을 개선합니다. PRDNet은 원소 및 환경 변동에 대한 민감성을 높이기 위해 데이터 기반の pseudo-particle을 사용하여 합성(diffraction pattern)을 생성합니다. 이 모델은 결정학적 대칭성에 대해 완전한 불변성을 보장합니다.

- **Technical Details**: 결정 구조는 원자 유형, 원자의 좌표 및 주기적인 격자 정의 등의 세 가지 요소로 기본적으로 설명될 수 있습니다. PRDNet은 기존의 다각형 그래프 모델 및 고차원 기하학적 정보를 통합하여 고유한 원자 및 구조적 환경의 구분을 보다 효과적으로 수행합니다. 이를 통해 긴 범위의 원자 상호작용을 포착하고, 이 모델은 잠재적으로 고체 상태 물질의 보다 정확한 특성 예측을 가능하게 합니다.

- **Performance Highlights**: PRDNet은 Materials Project, JARVIS-DFT, MatBench 등 대규모 데이터셋에서 반복적으로 평가를 수행하였으며, 기존의 최첨단 모델들을 능가하는 성과를 보였습니다. 다양한 결정 특성 예측 작업에서 본 모델이 제공하는 정확성과 효율성을 입증하였습니다. 본 연구의 결과는 결정 구조 예측 방법의 새로운 패러다임을 제시하며, 여러 분야에서의 응용 가능성을 보여줍니다.



### Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models (https://arxiv.org/abs/2509.21761)
- **What's New**: 이 논문은 Fine-tuned Large Language Models (LLMs)의 내부 메커니즘을 탐구하며, 특히 백도어 공격(backdoor attack)에 대한 새로운 해석 가능성을 제시합니다. 기존 연구들이 LLM의 안전성에 대한 해석을 형성하는 데 집중했으나 주로 alignment, jailbreak, hallucination에 초점을 맞춘 반면, 본 연구는 백도어 메커니즘을 분석할 수 있는 프레임워크인 Backdoor Attribution (BkdAttr)을 도입합니다. 이를 통해 백도어 기능이 학습된 방식 및 해당 기능을 처리하는 특정 attention heads를 규명합니다.

- **Technical Details**: BkdAttr 프레임워크는 세 가지 해석기법으로 구성되며, Backdoor Probe를 통해 다양한 입력 샘플의 표현에서 백도어를 구분합니다. 백도어 기능이 내재된 표현이라는 사실을 입증한 후, Backdoor Attention Head Attribution (BAHA)를 통해 이 기능 추출에 기여하는 attention heads를 식별합니다. 이를 바탕으로, 백도어를 제어하기 위한 Backdoor Vector를 구축하여 LLM의 입력에서 1-point intervention으로 백도어 활성화 또는 중립화를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, ∼3%의 주의 헤드를 제거하는 것으로 백도어 공격 성공률(ASR)을 90% 이상 감소시킬 수 있습니다. Backdoor Vector는 간단한 덧셈 또는 뺄셈을 통해 백도어 행동을 조절할 수 있으며, 특정 상태에 대한 1-point intervention으로 ASR을 0.39% 이하로 낮추거나 100%까지 끌어올릴 수 있습니다. 이러한 결과들은 백도어 시스템의 메커니즘에 대한 깊은 통찰을 제공하며, LLM의 안전한 배포를 위한 중요한 정보로 작용할 수 있습니다.



### SubZeroCore: A Submodular Approach with Zero Training for Coreset Selection (https://arxiv.org/abs/2509.21748)
- **What's New**: SubZeroCore는 훈련이 필요 없는 새로운 coreset 선택 방법으로, submodular optimization에 기반하여 커버리지(coverage)와 밀도를 밀접하게 결합합니다. 기존의 gradient 기반 방법과 대조적으로, 이 방법은 훈련 및 평가가 필요 없으며, 단일 하이퍼파라미터에 의해 조절되는 효율적인 샘플링 전략을 제안합니다. 이러한 접근 방식은 특히 자원 제한 및 데이터 수집의 제약이 있는 현실 세계의 상황에서 큰 장점을 가집니다.

- **Technical Details**: SubZeroCore에서는 기하학 기반의 방법론을 활용해 코어셋(selection된 집합)의 커버리지와 밀도를 동시에 최적화합니다. 이 메소드는 gradients나 iteration training 없이도 적용되며, 데이터의 구조적 밀도를 이해하고 최적의 근Neighborhood size를 정하는 데 도움이 됩니다. 또한, submodularity 속성을 통해 이론적 최적성 보장을 제공합니다.

- **Performance Highlights**: CIFAR-10 및 ImageNet-1K에서의 실험 결과, SubZeroCore는 낮은 pruning 비율에서는 기존의 훈련 기반 기준에 비견될 만큼 성능을 보였고, 높은 pruning 비율에서는 이를 초과하였습니다. 이러한 결과는 SubZeroCore가 라벨 노이즈에 강한 견고성을 보여주며, 실제 데이터 처리에서 효과적으로 확장 가능함을 강조합니다.



### HyperCore: Coreset Selection under Noise via Hypersphere Models (https://arxiv.org/abs/2509.21746)
- **What's New**: 이번 논문에서는 HyperCore라는 새로운 coreset selection 방법을 제안합니다. 기존의 방법들은 라벨 노이즈를 무시하거나 고정된 pruning 비율이 필요하여 실제 환경에 적용하기 어려운 문제점을 가지고 있었습니다. HyperCore는 각 클래스를 위한 경량 하이퍼스피어 모델을 사용하여 노이즈가 많은 환경에서 데이터 선택을 적응적으로 수행합니다.

- **Technical Details**: HyperCore는 각 클래스에 대해 하이퍼스피어 모델을 훈련시키고, 이를 통해 클래스 내부 샘플과 외부 샘플을 분리합니다. Youden의 J 통계량을 이용하여 적응적인 pruning 임계값을 선택하며, 이로 인해 하이퍼파라미터 조정 없이 자동으로 노이즈를 인식하고 데이터 선택을 최적화할 수 있습니다. 이러한 방식으로, HyperCore는 연산적으로도 경량이며 병렬 처리가 가능합니다.

- **Performance Highlights**: 실험결과, HyperCore는 가장 최신의 coreset selection 방법들보다 일관되게 더 나은 성능을 보였으며, 특히 노이즈가 많고 데이터 양이 적은 상황에서 두드러진 성과를 나타냅니다. HyperCore는 잘못 라벨이 붙여진 샘플이나 모호한 포인트를 효과적으로 제거하여, 압축되면서도 정보량이 높은 서브셋을 생성하여 스케일과 노이즈가 없는 학습에 적합합니다.



### Brain PathoGraph Learning (https://arxiv.org/abs/2509.21742)
- **What's New**: 뇌 그래프 학습(Brain graph learning)의 효율성을 높이기 위해 새로운 경량 모델인 Brain PathoGraph Learning (BrainPoG)이 제안되었습니다. 기존 방법들은 질병 관련 지식의 선택적 학습에 어려움을 겪고 있어 매개변수와 계산 비용이 증가하는 경향이 있었습니다. BrainPoG는 병리적 패턴 필터링과 병리적 특징 증류(pathological feature distillation)를 통해 효율적인 뇌 그래프 학습을 가능하게 합니다.

- **Technical Details**: BrainPoG는 질병 관련성이 높은 하위 그래프(subgraphs)를 추출하기 위해 설계된 병리적 패턴 필터와 불필요한 노이즈 특성을 제거하고 병리적 특성을 강화하는 병리적 특징 증류 모듈을 포함합니다. 이를 통해 BrainPoG는 뇌 그래프에서 질병 관련 지식을 효율적으로 학습하며, 특히 데이터의 불필요한 노이즈를 줄일 수 있습니다. 최종적으로, 강화된 노드 특성을 가진 PathoGraph에 대해 간단한 그래프 컨볼루션 네트워크 (GCN)를 이용하여 학습을 진행합니다.

- **Performance Highlights**: 최신 네 가지 벤치마크 데이터셋에서 BrainPoG는 기존 최첨단 방법들보다 모델 성능과 계산 효율성 모두에서 우수한 성과를 보였습니다. 본 모델은 특히 임상 환경에서의 실용성을 크게 향상시키며, 시간에 민감한 또는 자원이 제한된 상황에서도 효과적으로 적용할 수 있습니다. 이러한 특징들로 인해 BrainPoG는 다양한 뇌 질병 감지 작업에서 효과적인 솔루션으로 자리매김하고 있습니다.



### Self-Speculative Biased Decoding for Faster Live Translation (https://arxiv.org/abs/2509.21740)
- **What's New**: 이번 연구에서는 Self-Speculative Biased Decoding이라는 새로운 추론 패러다임을 제안하여 동시 번역과 같은 스트리밍 응용 프로그램에서 대형 언어 모델(LLM)의 출력 생성을 효율적으로 개선하고자 합니다. 기존의 방식과 달리, 이 방법은 새롭게 생성된 출력이 아닌 최신 출력을 초안으로 사용하여 계속 확장되는 입력 컨텍스트를 처리합니다. 이렇게 함으로써, 불필요한 재 생성(re-generation) 과정을 줄이고 처리 속도를 높일 수 있습니다.

- **Technical Details**: Self-Speculative Biased Decoding은 새로운 입력에 대한 출력을 생성할 때, 이전의 스트리밍 출력을 직접 검증하고 불일치하는 지점에서 과정(데코딩)을 재개하는 방식을 사용합니다. 여기서 중간 출력 기록을 재사용함으로써, 초안 생성 단계(draft computation)를 완전히 생략할 수 있습니다. 또한, 검증 단계에서 초안 토큰에 대한 편향(bias)을 적용하여 초안 수용률을 높임으로써 처리 속도를 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안하는 방법은 기존의 오토리그레시브(auto-regressive) 방식과 비교하여 최대 1.7배의 속도 향상을 달성하였으며, 출력 품질에도 타협을 두지 않았습니다. 뿐만 아니라, flickering 현상을 80%까지 줄이는 효과를 보이며 사용자 경험을 개선하는데 기여했습니다. 이러한 방식은 다양한 스트리밍 응용 프로그램에서 널리 활용될 수 있는 가능성을 지니고 있습니다.



### LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation (https://arxiv.org/abs/2509.21738)
- **What's New**: 이 논문에서는 자원 제약이 있는 임상 환경에서 사용하기 위한 경량 망막 혈관 분할 네트워크인 LFA-Net을 제안합니다. LFA-Net은 새로운 주의 모듈인 LiteFusion-Attention을 통합하여 경량 및 효율적인 방식으로 지역 및 전역 맥락을 캡처할 수 있습니다. 이 연구는 기존 딥러닝 기반 분할 방법에서 발생하는 작은 혈관 분할과 높은 계산 비용의 도전 과제를 해결하고자 합니다.

- **Technical Details**: LFA-Net은 0.11 백만 개의 파라미터, 0.42 MB의 메모리 크기 및 4.46 GFLOPs의 수치를 자랑하며, 이는 자원 제한 환경에서의 실시간 배치를 위한 이상적인 설계를 가능하게 합니다. LiteFusion-Attention 모듈은 잔차 학습 연결, Vision Mamba에서 영감을 받은 역학, 변조 기반 주의를 결합한 효과적인 주의 메커니즘을 이용하여 다중 스케일 표현 학습을 수행합니다. 이러한 설계는 메모리 사용을 최소화하면서도 세밀한 분할을 보장합니다.

- **Performance Highlights**: LFA-Net은 DRIVE, STARE 및 CHASE_DB 데이터셋에서 각각 83.28%, 87.44%, 84.50%의 Dice 점수와 72.85%, 79.31%, 74.70%의 Jaccard 지수를 기록하며 뛰어난 성능을 보였습니다. 이 논문은 LFA-Net이 기존의 더 크고 복잡한 모델보다 우수한 성과를 내며, 자원 제한 환경에서의 임상 배치에 이상적임을 입증합니다. 또한, 제안된 모델은 임상 진단의 신뢰성을 높이며 효율성 또한 보장합니다.



### POLO: Preference-Guided Multi-Turn Reinforcement Learning for Lead Optimization (https://arxiv.org/abs/2509.21737)
- **What's New**: 이번 논문에서는 POLO(Preference-guided multi-turn Optimization for Lead Optimization)라는 새로운 프레임워크를 제안합니다. POLO는 기존의 LLM 기반 접근 방식이 가진 한계를 극복하여, 개별 최적화 단계를 독립적으로 처리하는 대신, 전체 최적화 경로를 학습할 수 있도록 합니다. 또한 Preference-Guided Policy Optimization(PGPO)이라는 새로운 강화 학습 알고리즘을 도입하여 각 단계에서의 성공적인 전략을 강화하고, 중간 분자의 순위를 매겨 비교적인 피드백을 제공합니다.

- **Technical Details**: POLO는 다중 턴 강화 학습 프레임워크로서, LLM을 샘플 효율적인 분자 최적화 전문 모델로 변환시킵니다. 이 과정에서 최적화 문제를 마르코프 결정 프로세스(MDP)로 수학적으로 모델링하며, 각 분자는 주어진 화학 공간 내에서 평가됩니다. POLO는 두 가지 상보적인 수준에서 학습 신호를 추출하는데, 경로 수준 최적화는 성공적인 전략을 강화하고 턴 수준 선호 학습은 분자의 중간 결과에 대한 피드백을 제공합니다.

- **Performance Highlights**: POLO는 단일 속성 작업에서 84%의 평균 성공률을 달성하며, 이는 기본 모델보다 2.3배 향상된 결과입니다. 다중 속성 작업에서도 50%의 성공률을 기록하였으며, 단 500회의 오라클 평가만으로도 기적적인 성과를 내었습니다. 이러한 결과는 POLO가 샘플 효율적인 분자 최적화 분야에서 새로운 최첨단 수준을 설정했음을 보여줍니다.



### Uncovering Alzheimer's Disease Progression via SDE-based Spatio-Temporal Graph Deep Learning on Longitudinal Brain Networks (https://arxiv.org/abs/2509.21735)
- **What's New**: 이번 연구에서는 Alzheimer’s Disease (AD) 진행을 예측하기 위한 해석 가능한 시공간 그래프 신경망 프레임워크를 개발했습니다. 이는 불규칙하게 샘플링된 종단적인 기능적 자기공명영상(fMRI) 데이터를 모델링하기 위해 이중 Stochastic Differential Equations (SDEs)를 활용합니다. 우리의 접근 방식은 OASIS-3와 ADNI라는 두 개의 독립적인 집단에서 검증되었으며, AD 진행과 관련된 주요 뇌 회로 이상을 식별하는 데 효과적입니다.

- **Technical Details**: 이 모델은 fMRI에서 장기적 기능적 연결 패턴을 학습하고 불규칙하게 샘플링된 이미징 데이터 간의 연속적인 추론을 가능하게 합니다. 이를 통해 parahippocampal cortex, prefrontal cortex 및 parietal lobule와 같은 중요한 뇌 영역의 이상을 명확히 할 수 있습니다. 또한, ventral attention, dorsal attention 및 default mode networks에서의 중요한 비정상 연결이 발견되었습니다.

- **Performance Highlights**: 제안된 방법은 AD 진행 예측에서 평균적으로 3%의 정확도 향상을 보여주었으며, 5-fold cross-validation을 통해 여러 분류 지표에서 높은 성능을 발휘했습니다. SDE 기반의 시계열 재구성 전략은 정확도, 민감도 및 특이도에서 각각 3%, 6%, 4%의 향상을 이뤘으며, 최종적으로 최상의 성능은 전체 파이프라인이 사용되었을 때 달성되었습니다.



### UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments (https://arxiv.org/abs/2509.21733)
- **What's New**: 이번 논문에서는 UISim이라는 새로운 이미지 기반 UI(simulator) 시뮬레이터를 소개합니다. 이 시스템은 단순한 화면 이미지에서 포괄적으로 모바일 환경을 탐색할 수 있는 다이나믹하고 인터랙티브한 플랫폼을 제공합니다. 기존의 방법들에서 느껴지는 제약을 해소하며, UI 전환을 현실감 있게 시뮬레이션할 수 있도록 설계되었습니다.

- **Technical Details**: UISim은 초기 전화 화면 이미지와 사용자 행동을 바탕으로 두 단계의 방법론을 사용하여 다음 UI 상태를 생성합니다. 첫 번째 단계에서 추상적인 레이아웃 정보를 예측하고, 두 번째 단계에서 이 정보를 기반으로 새로운 시각적으로 일관된 UI 이미지를 합성합니다. 이러한 구조의 분리는 복잡한 이미지 간 UI 변환 문제를 보다 관리하기 쉬운 하위 문제로 나누어, 높은 충실도와 다양한 생성 능력을 제공합니다.

- **Performance Highlights**: 실험 결과에서 UISim은 이전의 UI 생성 모델보다 36.73의 Fréchet Inception Distance로 우수한 성능을 보여주었습니다. 이는 UISim이 현실적이고 일관된 다음 UI 상태를 생성하는 데 효과적임을 입증합니다. 또한, UI 테스트와 신속한 프로토타입 제작, 고급 응용 프로그램에도 유용하여 AI 에이전트의 UI 내비게이션 작업 계획에 기여할 수 있습니다.



### Developing Strategies to Increase Capacity in AI Education (https://arxiv.org/abs/2509.21713)
Comments:
          This is a 40 page report prepared by the CRA based on 32 virtual roundtable discussions with 202 experts committed to developing AI Education from varied backgrounds

- **What's New**: 이번 연구는 인공지능(AI) 교육의 수요 증가에 대응하기 위한 많은 기관들의 노력을 다루고 있습니다. Computing Research Association (CRA)는 AI 교육 개선에 헌신하는 202명의 전문가와 32회의 화상 원탁 논의를 진행하였습니다. 이 논의는 AI 지식 영역 및 교수법, AI 교육 인프라 문제, AI 교육의 용량 증대 전략, 모든 사람을 위한 AI 교육 등 네 가지 주요 영역에 집중되었습니다.

- **Technical Details**: 디지털 격차는 인프라 문제를 초래하고, 이는 특히 자원이 부족한 기관에서 두드러지게 나타납니다. AI 전문 지식을 갖춘 교수진의 부족과 그들의 재교육을 위한 시간의 제한, AI 모델을 개발하고 테스트할 수 있는 컴퓨팅 인프라의 부족, 그리고 기술 지원의 부족이 주요 문제로 지적되었습니다. 커리큘럼 업데이트와 새로운 프로그램 개발의 부담은 이러한 문제를 더욱 악화시킵니다.

- **Performance Highlights**: 저자는 이러한 교수진 격차를 해소하기 위해 접근 가능하고 지속적인 전문 개발이 필수적이라고 강조하고 있습니다. 특히 자원이 부족한 기관을 위한 지원이 필요하며, 컴퓨팅 프로그램 내외의 교수진 모두에게 AI 교육 접근을 보장해야 합니다. 이 연구에서는 자주 요구되는 AI 교육 자료의 중앙 저장소 목록을 정리하였으며, 이는 고등교육 기관들이 자유롭게 활용할 수 있도록 제공됩니다.



### Not My Agent, Not My Boundary? Elicitation of Personal Privacy Boundaries in AI-Delegated Information Sharing (https://arxiv.org/abs/2509.21712)
- **What's New**: 이 연구는 개인의 프라이버시 경계를 이해하고 AI 시스템과 일치시키기 위한 새로운 접근 방식을 제안합니다. 특히, 개인의 미세한 프라이버시 행동을 탐구하기 위해 AI 기반 유도 방법을 도입했습니다. 연구 결과는 커뮤니케이션 역할과 AI 위임 조건이 개인의 프라이버시 경계를 형성하는 데 미치는 영향을 분석합니다. 이는 AI 시스템이 개인의 프라이버시 기대에 부합할 수 있도록 하는 데 중요한 기초를 제공합니다.

- **Technical Details**: AI 기반 방법론을 사용하여 개인의 프라이버시 경계를 탐색하기 위해 차별화 작업(discriminative task)을 채택했습니다. 169명의 참가자와의 온라인 실험을 통해, 특정한 커뮤니케이션 역할(발신인/주체/수신인)과 위임 조건(AI/인간)이 정보 공개의 수용성에 미치는 영향을 조사했습니다. 결과적으로, 참가자들은 정보의 식별 가능성과 세부 수준에 따라 서로 다른 민감도를 보였으며, 이는 개인의 특성에 의해서도 영향을 받았습니다.

- **Performance Highlights**: AI 에이전트가 정보 공개를 위임받을 때, 참가자들은 식별 가능한 정보를 공개할 때 더 조심스러워지는 경향을 보였습니다. 또한, 프라이버시 필요성이 높은 참가자들일수록 식별자 포함에 대한 우려가 컸습니다. 이러한 발견들은 AI 시스템이 보다 정교한 프라이버시 경계를 설정하는 데에 있어 중요한 기반을 제공하며, 향후 AI와 인간의 프라이버시 선호를 정렬하기 위한 방향을 제시합니다.



### Optimizing the non-Clifford-count in unitary synthesis using Reinforcement Learning (https://arxiv.org/abs/2509.21709)
- **What's New**: 이 논문은 양자 회로 합성을 위해 강화 학습(reinforcement learning, RL)을 활용하여 T-count와 CS-count를 최적화하는 방법을 제시합니다. 기존 알고리즘의 한계를 넘어, 개선된 성능과 성공률을 통해 더 큰 단위(units)를 보다 효율적으로 구현할 수 있음을 보여 줍니다. 이 논문은 특히 클리포드(+T)와 클리포드(+CS) 게이트 집합에 대해 정확히 구현 가능한 단위에 대한 최적화 문제를 다루고 있습니다.

- **Technical Details**: 강화 학습 프레임워크는 단위를 정수 배열(channel representation)로 표현하여 매트릭스 연산을 효율적으로 처리합니다. 이는 복잡한 수치 계산을 제거하고 검색 복잡성을 줄이는 데 기여합니다. 알고리즘은 게이트 집합을 재구성하여 기존 게이트 세트인 클리포드(+T) 대신 생성 집합인 𝒢T 및 𝒢CS를 사용합니다.

- **Performance Highlights**: 알고리즘은 두 큐비트 클리포드(+T) 합성에 대해 이전 RL 알고리즘보다 5배 더 많은 T 게이트를 사용하여 최적에 가까운 분해 결과를 달성했습니다. 또한, 두 큐비트 클리포드(+CS) 단위에 대해 선형 복잡도를 실현하여 이전 알고리즘들이 도달하지 못한 성과를 이뤘습니다. 이 연구는 특히 다중 큐비트 비클리포드 게이트의 카운트를 최적화하는 데 있어 새로운 통찰을 제공합니다.



### QueryGym: Step-by-Step Interaction with Relational Databases (https://arxiv.org/abs/2509.21674)
- **What's New**: QueryGym은 LLM 기반의 쿼리 계획 에이전트를 개발, 테스트 및 평가하기 위한 상호작용 환경으로 소개됩니다. 이 환경은 기존 프레임워크와 달리 에이전트가 관계 대수 연산의 명시적 시퀀스를 구성하도록 요구하여, 엔진 독립적인 평가와 투명한 단계별 계획을 보장합니다. QueryGym은 Gymnasium 인터페이스로 구현되어 관찰 사항과 액션을 수신하여 에이전트의 데이터베이스 탐색을 지원합니다.

- **Technical Details**: QueryGym의 NL2Query 작업은 POMDP(부분적으로 관찰 가능한 마르코프 결정 프로세스)로 설정되며, 환경 상태 전이 함수와 보상 함수가 정의됩니다. 에이전트는 데이터베이스와 상호작용하여 자연어 질문에 답변해야 하며, 이때 데이터베이스의 스키마와 중간 테이블 정보를 포함하는 다양한 관찰 사항을 수신합니다. QueryGym은 SQLite 및 PostgreSQL을 지원하며, NL2Query 데이터세트를 POMDP로 변환할 수 있는 기능도 갖추고 있습니다.

- **Performance Highlights**: QueryGym은 오류 수정, 투명성 및 쿼리 생성을 위한 강화 학습 연구의 실용적인 시험대로 자리 잡고 있습니다. 에이전트는 초기 SQL 쿼리를 첫 번째 작업으로 사용하고, 피드백을 통해 쿼리를 점진적으로 수정하여 목표 답변 테이블과의 간격을 줄여 나갑니다. 이 환경은 연구자들이 다양한 SQL 방언과 데이터베이스 시스템에 구애받지 않고 쿼리 계획 에이전트를 개발할 수 있도록 합니다.



### SlotFM: A Motion Foundation Model with Slot Attention for Diverse Downstream Tasks (https://arxiv.org/abs/2509.21673)
- **What's New**: 이번 논문에서는 다양한 하위 작업에 대해 일반화할 수 있는 가속도계 기반 모델인 SlotFM을 제안합니다. 기존의 모델들은 주로 일상 활동 분류에 집중하고 있었으나, SlotFM은 Time-Frequency Slot Attention을 사용하여 시간 및 주파수 표현을 처리하며 다양한 신호 특성을 활용합니다. 이를 통해 모델이 더 광범위한 작업을 처리할 수 있도록 하여 응용 가능성을 확장합니다.

- **Technical Details**: SlotFM 모델은 가속도계 신호를 여러 개의 "슬롯"으로 분해합니다. 이러한 슬롯 각각은 고유한 신호 구성요소를 캡처하며, 이를 통해 앞으로의 작업에 맞춘 특정 헤드가 데이터의 가장 중요한 부분에 집중할 수 있도록 합니다. 또한, 신호의 로컬 구조와 주파수 패턴을 포착하는 두 가지 손실 정규화기를 도입하여 세부 정보의 재구성을 개선하고 임베딩이 작업 관련 정보를 보존하도록 지원합니다.

- **Performance Highlights**: SlotFM은 16개의 다양한 하위 작업에서 평가되었으며, 기존의 자가 감독 방법보다 13개 작업에서 뛰어난 성능을 보였습니다. 평균적으로 4.5%의 성능 향상을 보였으며, 일부 작업에서는 완전 감독 모델을 능가하기도 하였습니다. SlotFM은 일관되게 우수한 성능을 발휘하며, 각 작업에 따라 특정 슬롯이 강조되는 것을 통해 적응성이 뛰어난 임베딩을 생성합니다.



### MORPH: Shape-agnostic PDE Foundation Models (https://arxiv.org/abs/2509.21670)
- **What's New**: MORPH는 형태에 구애받지 않는 자가 회귀형 (autoregressive) 기초 모델로, 부분 미분 방정식 (PDEs) 문제 해결을 위한 것입니다. 이 모델은 다른 차원의 다양한 spacetime 데이터 세트를 자동으로 처리하게 설계된 convolutional vision transformer 아키텍처를 사용합니다. 주요 기술로는 component-wise convolution, inter-field cross-attention 및 axial attentions가 포함되며, 이들은 모두 복잡한 과학적 데이터를 효과적으로 처리할 수 있는 기능을 제공합니다.

- **Technical Details**: MORPH의 아키텍처는 여러 가지 필드와 혼합된 스칼라 및 벡터 성분을 다릅니다. component-wise convolution은 스칼라와 벡터 채널에서의 지역적인 상호작용을 포착하며, inter-field cross-attention은 서로 다른 물리적 필드 간의 정보를 선택적으로 전파합니다. 마지막으로, axial attentions는 전체 spatiotemporal self-attention을 개별 공간 및 시간 축을 따라 분해하여 계산 부담을 줄이며 표현력을 유지합니다.

- **Performance Highlights**: MORPH는 여러 가지 하류 예측 작업으로의 전이 학습에서 초기 모델을 초과하는 성능을 보입니다. 특히 zero-shot 및 full-shot 일반화 모두에서 모델 성능을 극대화하여 강력한 기준선과 최신 모델을 초과하는 성능을 달성하였습니다. 이러한 특성들은 다양한 과학적 관측의 이질적이고 다차원적인 본질에서 학습하기 위한 유연하고 강력한 기초 구조를 제시합니다.



### DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks (https://arxiv.org/abs/2509.21666)
- **What's New**: 이 논문은 깊은 신경망에서 도메인 정보를 반영한 단조성(monotonicity) 제약 조건을 적용하는 새로운 정규화 방법인 Domain Informed Monotonicity (DIM)를 제안합니다. 이 방법은 모델이 훈련 데이터의 잡음을 기억하기보다 일반화 가능한 패턴을 학습하도록 돕기 위해 복잡한 심층 학습 모델 내에서 도메인 정보를 반영한 단조 관계를 유지합니다. DIM은 선형 기준선에 대해 위반을 처벌하여 모델이 예측을 개선하도록 유도합니다.

- **Technical Details**: 논문에서는 먼저 각 단조 특성이 선형 기준선을 따르도록 하여 모델의 현재 예측을 비교하고, 이를 통해 단조 관계가 유지될 수 있도록 하는 정량적 메커니즘을 제안합니다. 이 접근 방식은 시작점으로서 선형 기준선을 설정하고, 이를 통해 위반 정도를 정량화함으로써 객관적인 평가를 제공합니다. 또한, 이 방법은 다양한 입력 영역과 모델 아키텍처에 대해 일관된 단조성 강제 적용을 가능하게 합니다.

- **Performance Highlights**: 실험은 시카고의 실제 라이더 소싱 데이터셋 및 합성 데이터셋을 사용하여 수행되었으며, 다양한 신경망 아키텍처에서 단조성 제약의 효과가 포함된 성능 향상을 보였습니다. DIM을 통해 심층 신경망의 예측 성능이 향상됨을 보여주며, 단조 정보 제약이 노이즈가 많고 신호가 약한 환경에서 일반화 능력을 개선하는 방법을제공합니다.



### Logic of Hypotheses: from Zero to Full Knowledge in Neurosymbolic Integration (https://arxiv.org/abs/2509.21663)
- **What's New**: 이 논문에서는 Logic of Hypotheses (LoH)라는 새로운 언어를 제안하며, 이는 데이터 기반의 규칙 학습과 기호적 사전 지식의 유연한 통합을 가능하게 합니다. LoH는 선택 연산자(choice operator)를 포함하여 신경망(neural networks) 모델에서 부분적인 논리 구조를 지정할 수 있는 기능을 제공합니다. 이로 인해 전문가의 사전 지식이 부족하거나 완전한 규칙이 없는 경우에도 적절한 모델을 학습할 수 있는 중간 경로를 제공합니다.

- **Technical Details**: LoH는 제안된 방법으로, 확률적 선택 연산자를 통해 수식의 하위 공식을 선택하는 기능을 제공합니다. 이를 통해 LoH는 데이터를 기반으로 한 논리적 수식을 학습할 수 있으며, 옵션 풀(pool)에서 최적의 선택을 학습하여 역전파(backpropagation)를 통해 딥러닝 모델을 개선할 수 있게 됩니다. 또한 LoH에서 생성된 수식은 미분 가능한(computable) 그래프로 변환되어 신경망과 함께 학습될 수 있습니다.

- **Performance Highlights**: 실험 결과, LoH를 기반으로 한 모델이 표 형식(tabular) 데이터와 Visual Tic-Tac-Toe 신경 기호 통합 작업에서 강력한 성능을 보였습니다. 본 연구는 해석 가능한 결정 규칙을 생성함으로써 NeSy(Neurosymbolic) 분야에서의 새로운 가능성을 제시합니다. 나아가 LoH는 기호적 규칙과 원시 데이터를 연결하는 중요한 도구로 작용할 수 있습니다.



### Limitations on Safe, Trusted, Artificial General Intelligenc (https://arxiv.org/abs/2509.21654)
Comments:
          17 pages, 1 figure

- **What's New**: 이 논문은 안전(Safety), 신뢰(Trust), 그리고 인공지능 일반 지능(AGI)의 수학적 정의를 제시하며, 이 세 가지 개념 간에 근본적인 불일치점을 보여줍니다. 전통적으로 해석되는 이 개념들을 수학적으로 명확하게 정의함으로써, 우리가 안전하고 신뢰받는 인공지능 시스템은 AGI가 될 수 없다는 결과를 도출합니다. 이 연구에서는 프로그래밍 검증 및 계획, 그래프 도달성 등을 통한 결과를 제시합니다.

- **Technical Details**: 논문에서는 AI 시스템을 특정 작업 인스턴스를 받아들여 해결하거나 답변을 제공하지 않도록 하는 시스템으로 정의합니다. 안전성은 시스템이 언제나 정확한 답변만 제공하는 것을 의미하며, 신뢰는 시스템이 안전하다고 가정하는 것입니다. AGI는 인간이 해결할 수 있는 작업 인스턴스를 비제로 확률로 해결할 수 있는 시스템으로 정의되며, 특정 작업을 인간보다 잘 수행해야 하는 것은 아닙니다.

- **Performance Highlights**: 저자들은 AI 시스템이 안전하고 신뢰받는 경우 AGI로 기능할 수 없다는 주요 결과를 강조합니다. 즉, 안전성과 신뢰성이 있는 AI 시스템은 인간이 쉽게 해결할 수 있는 작업을 해결하지 못할 수 있습니다. 이러한 발견은 AGI 개발 과정에서 안전성과 신뢰성이 어떻게 상충하는지를 보여줍니다.



### MobiLLM: An Agentic AI Framework for Closed-Loop Threat Mitigation in 6G Open RANs (https://arxiv.org/abs/2509.21634)
- **What's New**: 이번 논문에서는 6G O-RAN 환경에서 완전 자동화된 사고 완화 시스템을 구현하기 위한 에이전틱 AI 프레임워크인 MobiLLM을 소개합니다. MobiLLM은 다양한 보안 워크플로를 조율하는 다중 에이전트 시스템으로, 대규모 언어 모델(LLM)을 기반으로 하고 있습니다. 이를 통해 O-RAN의 보안 격차를 해소하고, 신뢰할 수 있는 AI 기반 네트워크 보안을 위한 청사진을 제공합니다.

- **Technical Details**: MobiLLM은 MITRE FiGHT 프레임워크 및 3GPP 사양과 같은 신뢰할 수 있는 지식 기반에 기반하여 설계되었습니다. 이 프레임워크는 실시간 데이터 분류를 위한 Threat Analysis Agent, RAG를 활용하여 이상을 특정 대응 조치에 매핑하는 Threat Classification Agent, O-RAN 제어 인터페이스를 통해 완화 조치를 안전하게 운영화하는 Threat Response Agent의 세 가지 주요 요소로 구성됩니다. 이러한 기술적 접근방식은 독립적이고 자율적인 보안 작업의 실행 가능성을 입증합니다.

- **Performance Highlights**: 초기 평가 결과, MobiLLM은 복잡한 완화 전략을 효과적으로 식별하고 조율할 수 있으며, 응답 지연 시간을 크게 줄이는 성능을 보여줍니다. 이는 6G 네트워크의 안전성을 높이고, 미션 크리티컬한 도메인에서의 6G 구현에 필요한 신뢰를 강화하는 데 기여할 것입니다. 또한, 안전 가드레일을 통해 LLM의 오작동이나 허위 출력을 원천적으로 차단할 수 있는 가능성을 보였습니다.



### InvBench: Can LLMs Accelerate Program Verification with Invariant Synthesis? (https://arxiv.org/abs/2509.21629)
- **What's New**: 이 논문은 프로그램 검증(verification)에서 루프 불변식(loop invariants)을 자동으로 발견하는 새로운 방법론을 제시합니다. 기존의 연구에서는 LLMs(대형 언어 모델)들이 이 작업에서 얼마나 효과적인지 평가하는 데 오류가 있었으며, 이에 대한 보다 체계적인 평가를 위한 InvBench라는 프레임워크를 도입합니다. 이 프레임워크는 LLM들이 생성한 불변식의 정확성과 검증 속도 개선 효과를 동시에 평가합니다.

- **Technical Details**: 루프 불변식은 각 루프 반복 전후에 참이 되는 조건으로, 프로그램 검증의 핵심 컴포넌트입니다. 이 연구에서는 검증 과정의 속도 향상 정도를 측정하여 불변식의 유용성을 평가하는 방법론을 개발했습니다. 고전적인 검증 도구 UAutomizer와의 비교를 통해 LLM 기반 검증기의 성능을 조사하며, 모델의 성능 차이를 분석하고 있습니다.

- **Performance Highlights**: 실험 결과, 7개의 최신 LLM들이 평가되었고 이들 중 Elite 모델은 높은 성능을 보였으나, UAutomizer와 비교할 때 실질적인 우위를 보이지 못했습니다. 더불어, 3589개의 인스턴스를 활용한 지도 학습(‘supervised fine-tuning’)과 Best-of-N 샘플링 기법을 통해 LLM의 성능이 크게 향상될 수 있음을 입증했습니다. 예를 들어, Qwen3-Coder-480B 모델의 속도 향상 사례 비율이 8%에서 29.2%로 증가했습니다.



### A Data-driven Typology of Vision Models from Integrated Representational Metrics (https://arxiv.org/abs/2509.21628)
- **What's New**: 이 논문은 다양한 구조의 비전 모델들 간의 표현 차이를 체계적으로 평가하는 새로운 프레임워크를 소개합니다. 이 연구는 기하학(geometry)나 조정(tuning)과 같은 특정 표현적 특성이 모델 가계의 분리를 어떻게 기여하는지를 규명하여, 개별 메트릭보다 더 명확한 가족 정체성을 제공하는 Similarity Network Fusion (SNF) 방법을 적용합니다. 기존의 접근법에 대한 보완으로, 이 연구는 자가 감독(self-supervised) 모델들이 아키텍처 경계를 초월하여 자연스러운 군집을 형성하고 있음을 발견했습니다.

- **Technical Details**: 연구진은 35개의 비전 모델을 네 가지 주요 범주로 분석하며, 이는 감독된 Convolutional Neural Networks (CNNs), 자가 감독 CNNs, 감독된 Transformers, 자가 감독 Transformers입니다. 다양한 유사성 메트릭을 평가하고, 기하학적 변환(Procrustes) 및 선형 예측(Linear Predictivity) 접근 방식의 유연성을 비교하여 모델 간의 표현 구조를 분석합니다. Singular Value Decomposition (SVD)와 Canonical Correlation Analysis (CCA)와 같은 방법론이 사용되며, 이론적으로 중요한 정보를 보존하는 다양한 매트릭의 효과를 평가합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 Hybrid 구조(ConvNeXt, Swin)가 MAE 모델들과 군집을 이루고, 자가 감독 모델들이 서로 밀접하게 연관되어 있음을 보여줍니다. 이러한 발견은 다양한 아키텍처 및 훈련 목표에 의한 나타나는 계산 전략이 표면적인 설계 카테고리를 넘어 얼마나 큰 의미를 가지는지를 잘 보여줍니다. 제안된 데이터 기반의 비전 모델 분류법은 연구자들이 모델의 관계를 이해하고, 전이 학습(transfer learning) 호환성을 예측하며, 새로운 작업에서 유사한 행동을 보이는 모델을 선택하는 데 유용한 도구를 제공합니다.



### Guiding Audio Editing with Audio Language Mod (https://arxiv.org/abs/2509.21625)
- **What's New**: 이번 논문에서는 SmartDJ라는 새로운 스테레오 오디오 편집 프레임워크를 소개합니다. 이 프레임워크는 사용자로부터 고수준의 지침을 받아 이를 원자적인 편집 작업으로 분해한 후, 이를 조건부 Latent Diffusion Model (LDM)을 통해 실행합니다. 이로써, 사용자는 복잡한 편집 과정에 대한 세부 사항을 걱정하지 않고 오디오의 최종 결과만을 선언할 수 있습니다.

- **Technical Details**: SmartDJ는 오디오 언어 모델(Audio Language Models, ALM)과 생성 모델(latent diffusion)의 결합을 통해 고수준 지침을 해석하고, 이를 순차적인 편집 작업으로 분해합니다. 이 과정에서, ALM은 오리지널 오디오를 인지하고 사용자의 요청을 이해하여 편집을 계획하며, 그런 다음 LDM을 통해 이러한 계획을 실행합니다. 이러한 접근 방식은 편집 작업을 절차적 작업에서 선언형 작업으로 전환합니다.

- **Performance Highlights**: SmartDJ는 기존의 오디오 편집 방법에 비해 뛰어난 감각적 품질과 공간적 현실감 및 의미적 일치를 달성합니다. 실험 결과, SmartDJ는 다양한 사용자 베이스라인과의 비교에서 가장 높은 편집 품질을 보여주었으며, 주관적 평가 및 객관적 메트릭 모두에서 우수한 성과를 기록했습니다. 이는 SmartDJ가 고수준 사용자의 지침을 효과적으로 이해하고 실행할 수 있는 능력을 가지고 있음을 보여줍니다.



### OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja's Ru (https://arxiv.org/abs/2509.21623)
- **What's New**: 이번 연구에서는 OjaKV라는 새로운 프레임워크를 소개합니다. OjaKV는 온라인 서브스페이스 적응을 결합한 전략적 하이브리드 저장 정책을 통해 KV 캐시의 압축 문제를 해결합니다. 이 프레임워크는 모든 토큰을 균일하게 압축하는 것이 최적이 아님을 인식하고, 중요한 첫 번째 및 최신 토큰의 고정밀도를 유지하는 동시에, 나머지 중간 토큰에 대해서는 저차원 압축을 적용합니다.

- **Technical Details**: OjaKV는 Oja의 알고리즘을 사용하여 온라인 주성분 분석(PCA)을 통해 프로젝션 기반의 저차원 어레이를 점진적으로 적응시키는 방법을 채택합니다. 이는 프롬프트 채우기 단계에서 포괄적인 업데이트를 수행하고 디코딩 중에는 경량의 주기적 업데이트를 통해 서브스페이스가 진화하는 컨텍스트와 정렬되도록 합니다. 이 방식은 FlashAttention과 같은 현대의 주의 모듈과 완벽하게 호환되어 실제 긴 컨텍스트 추론에서 실용성을 높입니다.

- **Performance Highlights**: OjaKV는 다양한 벤치마크에서 뛰어난 성능을 보여주며, 특히 제로샷 정확도를 유지하거나 개선합니다. 특히 긴 컨텍스트에 대한 성능 향상이 두드러지며, 복잡한 추론이 요구되는 작업에서 온라인 서브스페이스 적응의 중요성이 강조됩니다. 따라서 OjaKV는 메모리 효율적인 긴 컨텍스트 추론을 위한 실용적이고 플러그 앤 플레이 솔루션으로 자리잡았습니다.



### LANCE: Low Rank Activation Compression for Efficient On-Device Continual Learning (https://arxiv.org/abs/2509.21617)
Comments:
          16 pages, 3 figures

- **What's New**: 본 논문에서는 LANCE (Low-rank Activation Compression)라는 새로운 프레임워크를 제안합니다. LANCE는 한 번의 고차원 특잇값 분해(HOSVD)를 통해 재사용 가능한 저차원 서브스페이스를 생성하여 활성화 프로젝션을 수행합니다. 이는 반복적인 분해 과정을 제거하여 메모리 및 계산 비용을 줄이고, 클라우드 기반 학습이 아닌 온디바이스 학습 환경에서 연속 학습을 가능하게 합니다.

- **Technical Details**: 온디바이스 학습에서 메모리 사용의 주요 병목 현상은 역전파(backpropagation) 중 활동성을 저장하는 데 필요한 대량의 메모리입니다. 본 논문에서는 HOSVD를 활용하여 활성화 텐서를 저차원으로 근사하고, 이를 통해 메모리 사용을 대폭 줄일 수 있는 방법론을 제시합니다. LANCE는 훈련 시작 시 단 한 번만 분해 과정을 수행하여, 이후의 활성화는 고정된 서브스페이스에 투영되며, 반복적인 분해를 피하게 됩니다.

- **Performance Highlights**: LANCE는 CIFAR-10/100, Oxford-IIIT Pets, Flowers102, CUB-200 데이터셋에서 활성화 저장을 최대 250배까지 줄이면서도, 전체 역전파 대비 유사한 정확도를 유지하는 결과를 보여줍니다. 연속 학습 벤치마크에서는 기존의 정교한 경량 하드웨어와 비슷한 성능을 발휘하며, 메모리 비용 측면에서 효율성을 제공합니다. 이러한 결과는 LANCE가 에지 디바이스에서의 효율적인 미세 조정과 연속 학습을 위한 실용적이고 확장 가능한 솔루션으로 자리 잡을 것임을 시사합니다.



### Multi-Objective Reinforcement Learning for Large Language Model Optimization: Visionary Perspectiv (https://arxiv.org/abs/2509.21613)
Comments:
          3 pages, 1 figure, accepted by ECAI MODeM 2025

- **What's New**: 이 논문은 Multi-Objective Reinforcement Learning (MORL)의 새로운 세분화 체계를 소개하고 LLM(대형 언어 모델) 최적화에 적용했을 때의 장점과 단점을 분석합니다. MORL 방법의 필요성을 강조하며, 개인화 기능을 수용하는 효율적이고 유연한 접근 방식의 개발을 제안합니다. 또한, 다양한 목표 관계의 영향을 다루는 MORL 벤치마킹 프레임워크의 비전을 제시합니다.

- **Technical Details**: MORL은 LLM 최적화를 위한 명시적이고 분해된 목적을 정의하기 위해 주로 스칼라 기반 RL을 사용합니다. 문서에서는 메타-정책(Meta-policy) MORL 방법이 전통적인 MORL 접근 방식의 비효율성과 유연성 부족 문제를 해결할 수 있는 가능성을 보여줍니다. 여러 정책을 학습하고 결합하는 바이레벨 학습(bi-level learning) 패러다임의 개발이 향후 연구 방향으로 강조됩니다.

- **Performance Highlights**: MORL 방법의 종합적인 벤치마킹 분석이 필요하며, 성능, 안정성, 적응력 및 설명 가능성과 같은 다양한 평가 메트릭스를 통해 LLM의 개선된 솔루션을 도출할 수 있다고 논합니다. 메타-정책 방법은 편향된 선호를 처리할 수 있는 가능성이 있지만 아직 충분히 탐구되지 않았으며 많은 MORL 방법이 의사결정 작업에 제한적으로 응용되고 있습니다.



### Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis (https://arxiv.org/abs/2509.21595)
- **What's New**: 이번 연구는 비디오 액션 인식을 위한 두 가지 주요 자가 감독 학습 아키텍처인 DINOv3와 V-JEPA2를 비교 분석합니다. DINOv3는 프레임을 독립적으로 처리하는 방식으로 공간적 특징을 추출하고, V-JEPA2는 비디오 시퀀스에 걸친 공동 시간 모델링을 활용합니다. UCF Sports 데이터셋을 기반으로 두 접근 방식의 특징 품질을 분류 정확도, 군집 성능 등 다양한 차원에서 평가하였습니다.

- **Technical Details**: DINOv3는 이미지 컬렉션에서 훈련된 교사-학생 증류 프레임워크를 사용하여 자가 감독 방식으로 밀집 시각적 특징을 학습합니다. 반면, V-JEPA2는 동작 역학과 프레임 관계를 인코딩하는 시간 토큰을 생성하기 위해 이를 공동으로 처리하는 아키텍처입니다. 이 두 접근 방식의 주요 차이는 공간적 충실성과 시간적 모델링 간의 근본적인 긴장을 반영합니다.

- **Performance Highlights**: DINOv3는 정적 포즈 인식에서 우수한 성능을 보이는 반면, V-JEPA2는 모든 액션 유형에서 일관된 신뢰성을 제공합니다. DINOv3는 포즈 식별 가능한 액션에 대해 뛰어난 판별 능력을 보여주지만, 움직임에 의존하는 액션에서는 성능 저하가 발생합니다. 이 연구는 비디오 분석 시스템에서의 건축 설계 선택에 대한 이해를 높이고, 작업 요구 및 신뢰성 제약에 따라 적절한 특징 추출 방법 선택에 대한 실증적 지침을 제공합니다.



### What Happens Next? Anticipating Future Motion by Generating Point Trajectories (https://arxiv.org/abs/2509.21592)
- **What's New**: 이 논문에서는 단일 이미지에서 동작을 예측하는 문제를 다룹니다. 기존의 비디오 생성기 아키텍처를 토대로 하여, 픽셀 대신에 운동 궤적(trajectory)을 생성하는 모델을 제안합니다. 이를 통해 장면의 동적 변화를 포착하고 예측의 정확성과 다양성을 향상시킵니다.

- **Technical Details**: 이 연구는 입력 이미지에 따라 예측되는 궤적을 생성하는 확률적 모델링을 실시합니다. 기존의 점 추적(point track) 모델과 달리, 모든 점에 대한 동작을 예측하여 전체 장면을 고려합니다. 특히, 비디오 생성기와 유사하게 흐름 일치(flow matching) 기법을 활용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 이전 방식 대비 더 효과적으로 동작 예측을 수행하며, 특히 합성 및 실제 시나리오에서 우수한 성능을 보였습니다. 기존의 최첨단 비디오 모델들이 단순한 시뮬레이션 상황에서도 어려움을 겪는 반면, 직관적으로 물리적 현상을 이해하는 데 있어 제안된 접근 방식이 훨씬 더 효율적이라는 점이 입증되었습니다.



### Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms (https://arxiv.org/abs/2509.21573)
- **What's New**: 본 논문에서는 새로운 공간 정규화 대조 학습 전략을 제안하여 기하학적 거리를 고려한 이미지 기반 위치 추정의 정확도를 향상시키고 있습니다. 이를 통해, 시각적으로 유사하지만 지리적으로 멀리 떨어져 있는 샘플을 효과적으로 구별할 수 있는 방법을 개발했습니다. 이 전략은 GeoCLIP 모델에 통합되어 OSV5M 데이터셋에서 평가되어 성능 향상을 입증하였습니다.

- **Technical Details**: 제안된 방법은 세미변량도(samivariogram)를 사용하여 샘플 간의 공간 상관 관계를 모델링합니다. 이를 통해 이미지의 특성 공간 내에서의 거리와 지리적 거리 간의 관계를 정의하여 시각적 유사성의 기대치가 공간의 상관 관계에 따라 변화한다는 것을 캡처합니다. GeoCLIP는 정적인 CLIP 이미지 인코더와 위치 인코더로 구성되어 있으며, 이미지와 위치 정보를 통해 특징을 효과적으로 추출합니다.

- **Performance Highlights**: 제안된 공간 정규화 대조 학습 전략은 특히 미세한 세분화에서 이미지 기반 위치 추정 성능을 크게 개선했습니다. 이 접근법은 과거의 방법들이 놓치고 있던 지리적 현실을 모델에 통합하여, 잘못된 부정 샘플(false negatives)과 어려운 부정 샘플(hard negatives)을 구별하는 데 핵심적인 역할을 합니다. 연구 결과, 지역별 이미지의 시각적 유사성과 지리적 거리가 시스템적으로 연관되어 있음을 보여 주었습니다.



### No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models (https://arxiv.org/abs/2509.21565)
- **What's New**: 이 논문은 대규모 diffusion 모델을 위한 새로운 훈련 전략으로 Linear SEParability (LSEP)를 제안합니다. LSEP는 외부 인코더나 표현 정렬 없이 중간 계층의 표현을 선형으로 분리 가능하게 훈련시켜, 모델의 학습 효율성과 생성 품질을 동시에 개선할 수 있습니다. 특히, 훈련 중에 직접 선형 프로브(linear probe)를 삽입하여 학습의 역동성과 표현 품질을 향상시키는 방식을 사용합니다.

- **Technical Details**: 이 연구의 주요 혁신은 LSEP를 통한 훈련 정규화 전략이며, 이는 명시적으로 선형 분리를 개선하는 두 가지 목표, 즉 초기사형(hidden representations)의 분리 가능성과 denoising 목적을 통합합니다. 구체적으로는 새로운 훈련 기법들을 통해 선형 프로브의 분류 조건을 다르게 적용하고, 패치 수준의 선형 분리를 향상시키기 위해 무작위 자르기(random cropping)를 활용했습니다. 결과적으로, LSEP는 SiT와 같은 흐름 기반 변환기 아키텍처에서 큰 학습 효율성과 생성 품질 향상을 이끌어냈습니다.

- **Performance Highlights**: LSEP를 적용한 SiT-XL 모델은 기존 SiT-XL 모델보다 더 낮은 FID를 더 빠르게 수렴합니다. FID는 256x256 해상도에서 ImageNet 데이터셋에 대해 1.46로 최적의 성능을 기록하였으며, 이는 외부 인코더에 의존하지 않고도 이루어진 성과입니다. 또한 이러한 방법은 기존의 정렬 기반 접근 방식과 상호작용하여 표현의 선형 분리를 더욱 향상시키며, 최종적으로 훈련 효율성과 생성 성능을 더 끌어올립니다.



### Domain-Aware Speaker Diarization On African-Accented English (https://arxiv.org/abs/2509.21554)
Comments:
          5 pages

- **What's New**: 본 연구는 아프리카 영어 억양에서의 화자 분리 (speaker diarization) 문제를 다룹니다. 다양한 대화형 및 임상 대화 데이터를 사용하여 여러 시스템의 성능을 엄격한 DER(다이얼리제이션 오류율) 프로토콜 아래 평가하였습니다. 연구 결과는 화자의 억양과 임상 대화에서의 전반적인 일관된 오류가 있음을 시사하며, 이로 인해 기존 모델들이 이러한 환경에 잘 일반화되지 못한다는 점을 강조합니다.

- **Technical Details**: 연구는 임상 대화와 일반 대화 각각에 대해 아프리카 영어 억양 데이터셋인 AfriSpeech-Dialog를 사용하여, Pyannote 모델을 통한 도메인 적응을 진행했습니다. fine-tuning을 통해 오류를 줄였지만, 여전히 임상 대화에서의 성능 차이는 존재합니다. 또한, DER을 기준으로 한 성능 평가를 통해 일반적 대화와 임상 대화 간의 성능 차이를 정보와 함께 제시합니다.

- **Performance Highlights**: 서로 다른 8개의 다이얼리제이션 모델을 평가한 결과, 임상 도메인에서 일반 도메인에 비해 DER이 유의미하게 높은 것을 확인했습니다. Pyannote 모델의 경우 fine-tuning 후에도 여전히 임상-일반 성능 차이가 남아있어, 임상 대화가 본질적으로 어려운 과제임을 나타냅니다. 이러한 결과는 의료와 같은 다중 스피커 환경에서 모델의 성능을 향상시키기 위한 다음 단계로, 균형 잡힌 임상 자원과 중첩 인식이 중요함을 강조합니다.



### Psychological and behavioural responses in human-agent vs. human-human interactions: a systematic review and meta-analysis (https://arxiv.org/abs/2509.21542)
- **What's New**: 이번 논문에서는 인간-기계 상호작용의 경험을 체계적으로 분석하여 기존의 연구들과의 차별점을 제시하였습니다. 162개의 연구를 종합적으로 검토하고 메타 분석을 통해, 인간 대 인간 상호작용과 인간 대 에이전트 상호작용 간의 행동적 및 심리적 반응을 비교하였습니다. 이 연구는 인간의 반응이 인간 대 기계 상호작용에서 상대적으로 낮은 친사회적 행동을 보이며, 기계에 대한 책임과 능력을 덜 부여한다는 점을 밝혔습니다.

- **Technical Details**: 연구는 146개의 연구에서 468개의 효과 크기를 포함하여 진행되었습니다. 빈도주의(frequentist) 및 베이지안(Bayesian) 접근 방식을 통합하여 메타 분석을 수행하였고, 주관적인 반응(예: 파트너의 사회적 인식, 주관적 신뢰 등)의 효과 크기 이질성을 높은 수준으로 관찰하였습니다. 이 결과는 파트너 효과가 맥락 의존적(context-dependency)임을 나타내며, 여러 요인이 파트너 효과를 형성하는 그래도 데이터로 식별되었습니다.

- **Performance Highlights**: 인간-에이전트 상호작용에서 기능적 행동과 상호작용 경험은 인간과 유사하게 나타났지만, 기본적인 사회적 귀속 및 도덕적/친사회적concerns은 여전히 부족하다는 점이 강조되었습니다. 에이전트는 인간과 유사한 도구적 가치는 가질 수 있으나, 본질적 가치(intrinsic value)에서는 차이가 있음을 보여줍니다. 이 연구 결과는 에이전트 디자인 및 규제에 있어 실질적인 시사점을 제공합니다.



### Agribot: agriculture-specific question answer system (https://arxiv.org/abs/2509.21535)
- **What's New**: 이번 논문에서는 인도 농업에 적합한 챗봇 시스템을 구축했습니다. 농부의 질문에 대해 날씨, 시장 가격, 식물 보호 및 정부 지원 프로그램 관련 정보를 제공하며, 24시간 언제든지 접근 가능하다는 점이 특징입니다.

- **Technical Details**: 이 시스템은 Kisan Call Center의 데이터를 기반으로 하며, 문장 임베딩 모델(sentence embedding model)을 사용하여 초기 56%의 정확도를 기록하였습니다. 동의어를 제거하고 개체 추출(entity extraction)을 통합함으로써 정확도는 86%로 향상되었습니다.

- **Performance Highlights**: 이 챗봇 시스템을 통해 농부는 농업 관련 정보를 쉽게 얻을 수 있으며, 농업 생산성을 높일 수 있는 기회를 제공합니다. 또한 Call Center 직원들의 업무가 보다 효율적으로 조정될 수 있도록 도와줍니다.



### Preemptive Detection and Steering of LLM Misalignment via Latent Reachability (https://arxiv.org/abs/2509.21528)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 안전성을 확보하기 위한 새로운 방법인 BRT-Align을 제안합니다. 이 방법은 LLM 추론 시의 안전성을 높이기 위해 reachability 기반의 제어 이론 도구를 활용합니다. BRT-Align은 잠재 공간(latent space)에서 모델의 자기 회귀 생성(autoregressive generation)을 동적 시스템으로 모델링하며, 안전 가치 함수(safety value function)를 역방향 도달 가능성을 통해 학습합니다.

- **Technical Details**: BRT-Align은 동적 시스템의 관점에서 LLM 생성을 접근하여 입력 프롬프트(prompt)에 의해 초기화된 숨겨진 상태(latent states)와 토큰 레벨 임베딩(token-level embedding) 간의 변화를 관리합니다. 이를 통해 생성 중 위험한 경로를 사전에 예측하고 조정할 수 있는 모니터와 최소한의 침해로 안전한 경로로 유도하는 필터를 구현합니다. 이러한 방식은 이전의 방법들이 생성 후에 반응하는 것과 달리 사전 예방적 조치를 취할 수 있습니다.

- **Performance Highlights**: BRT-Align은 여러 LLM 및 독성 평가 지표를 사용한 실험에서 기존 방법들보다 더 정확하고 빠르게 안전하지 않은 생성을 감지하는 성과를 보여주었습니다. 또한 BRT-Align은 불안전한 생성을 크게 줄이면서도 문장의 다양성과 일관성을 유지하는 데 성공하였습니다. 결과적으로, 이 연구는 LLM 안전성을 위한 새로운 이론적 및 실용적 기반을 제공함을 강조합니다.



### Shortcut Flow Matching for Speech Enhancement: Step-Invariant flows via single stage training (https://arxiv.org/abs/2509.21522)
Comments:
          5 pages, 2 figures, submitted to ICASSP2026

- **What's New**: 이번 연구에서는 음성 향상(Speech Enhancement) 기술을 위한 새로운 방식으로 Shortcut Flow Matching for Speech Enhancement(SFMSE)를 도입합니다. 이 접근법은 단일 단계 불변 모델을 학습하여 특히 낮은 지연시간과 높은 품질을 동시에 달성할 수 있습니다. SFMSE는 목표 시간 단계에 따라 속도 필드를 조정하여 단일 또는 다중 단계로 노이즈 제거를 가능하게 합니다.

- **Technical Details**: SFMSE는 확률 질량을 확산 모델의 시작점인 p1(𝐱∣𝐲)에서 조건부 분포 p0(𝐱∣𝐲)로 운반하는 속도 필드를 학습하는 흐름 기반 접근법입니다. 이 모델은 ODE(Ordinary Differential Equation)를 사용하여 역방향으로 통합하여 노이즈가 포함된 신호를 복원합니다. 특히, 이번 연구에서는 linear interpolation을 활용하여 안정적인 속도를 유지하면서 다단계 추론을 진행할 수 있습니다.

- **Performance Highlights**: 실험 결과, SFMSE는 소비자 GPU에서 단일 NFE로 0.013의 실시간 요인을 달성하며, 60 NFEs를 사용하는 기존의 확산 모델과 유사한 품질을 보여주었습니다. 이 연구는 고품질 생성 음성 향상과 낮은 지연 시간 요구 사항 간의 간극을 메우기 위한 실질적인 분석을 제공합니다.



### $\mathbf{Li_2}$: A Framework on Dynamics of Feature Emergence and Delayed Generalization (https://arxiv.org/abs/2509.21519)
- **What's New**: 이번 논문에서는 복잡한 구조를 가진 입력에 대해 학습 중에 어떤 특징(feature)이 어떻게 발생하는지를 수학적으로 정량화할 수 있는 새로운 프레임워크인 $	extbf{Li_2}$를 제안합니다. 이 프레임워크는 2층 비선형 네트워크의 grokking 행동을 설명하는 세 가지 핵심 단계를 포착하는 데 초점을 맞추고 있습니다. 각 단계는 게산된 전파된 기울기 $G_F$의 구조에 의해 특성화됩니다.

- **Technical Details**: 제안된 프레임워크의 세 가지 단계는 다음과 같습니다: (I) Lazy learning(게으른 학습)에서는 기울기 $G_F$가 무작위적이며, 출력을 구성하는 최상위 층이 무작위 숨겨진 표현에 과적합합니다. (II) Independent feature learning(독립된 특징 학습)에서는 각 노드의 기울기가 자신의 활성화에만 의존하게 되며, 이제 $G_F$는 목표 레이블에 대한 정보를 실은 채로 숨겨진 노드가 독립적으로 그들의 표현을 학습합니다. (III) Interactive feature learning(상호작용 특징 학습)에서는 숨겨진 층의 가중치가 업데이트되어 노드 간의 상호작용이 발생하고, $G_F$는 학습해야 할 누락된 특징에 집중하게 됩니다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크의 주장을 뒷받침하며, 일반화/암기 경계에 대한 증명된 스케일링 법칙은 실험 데이터를 완벽하게 재현합니다. 작은 학습률이 일반화 가능한 솔루션으로 이어진다는 사실을 보여주며, 발산한 에너지가 경계에서 더 나은 성능을 보일 수 있음에도 불구하고 발생합니다. 마지막으로, $	extbf{Li_2}$ 프레임워크는 잔여 연결(residual connections)을 포함하는 다층 설정으로 확장이 가능하다는 점을 입증합니다.



### DistillKac: Few-Step Image Generation via Damped Wave Equations (https://arxiv.org/abs/2509.21513)
- **What's New**: DistillKac는 감쇠된 파동 방정식(damped wave equation)과 확률적 Kac 표현을 사용하여, 확률 질량을 유한한 속도로 이동시키는 빠른 이미지 생성기를 제안합니다. 기존의 확산 모델(diffusion models)은 역시멘트 급속한 속도로 질량을 확산시키지만, Kac 동역학은 유한한 속도 수송을 강제하며 전 세계적으로 유도된 운동 에너지를 유지합니다. 또한, 우리는 부드러운 조건에서 제곱 적분 가능성을 보존하는 분류기 없는 안내(classifier-free guidance)를 도입했습니다.

- **Technical Details**: Kac 동역학은 전통적인 확산 모델에 비해 낮은 시간의 무게 중심에서 질량 이동이 정제되며, 이로 인해 엄격한 경계 속도(norm)와 보편적인 안정성(stability)의 이점을 제공합니다. 우리는 경로 전체에 대한 근접성을 증진시키는 종단점만의 증류(endpoint-only distillation) 방법을 제안하며, 이로 인해 학습한 네트워크가 평균적으로 두 점 사이를 연결할 수 있게 됩니다. 이러한 특징은 유한 속도 Kac 동역학 하에서 안정적인 추론을 지원합니다.

- **Performance Highlights**: 우리의 실험은 DistillKac가 매우 적은 함수 평가로 높은 품질의 샘플을 생성할 수 있음을 보여줍니다. 특히, 삼각측면에서의 안정성과 품질이 뛰어나며, 결과적으로 몇 단계로도 효과적인 샘플링을 가능하게 합니다. 이러한 실험 결과는 우리가 제안한 방법들이 가이드를 잘 유지하면서도 적은 에너지를 소모한다는 것을 입증합니다.



### New Algorithmic Directions in Optimal Transport and Applications for Product Spaces (https://arxiv.org/abs/2509.21502)
- **What's New**: 본 연구는 고차원 분포 $cmu$와 $cnu$ 사이의 최적 운송 문제를 알고리즘 관점에서 조사합니다. 주어진 $x 	hickapprox cmu$에 대해, $poly(n)$ 시간 내에 근접한 $y 	hickapprox cnu$를 찾아야 합니다. 이를 통해 실행 시간은 $cn$의 차원에 의존하게 됩니다.

- **Technical Details**: 주요 결과는 $ell_p^p$ 하에서 여유비용 $cDelta + cdelta$로 어떤 제품 분포 $cmu$를 임의의 $cnu$로 운송하기 위한 일반적인 알고리즘입니다. 이 알고리즘은 $cnu$가 한정된 평균 샘플링 비용을 가진 '순차적 샘플링 가능'해야 하며, 이는 새로운 개념입니다. 또한 표준 가우시안 $cPhi^{n}$을 제곱 유클리드 비용 하에서 임의의 $cnu$로 운송하는 알고리즘의 형태를 증명합니다.

- **Performance Highlights**: 가우시안 측도가 유클리드 거리하에서 차원 독립적인 운송 비용으로 조사되는 첫 번째 계산 집중 결과를 얻습니다. 이는 Etesami et al.의 열려 있는 질문을 해결한 것으로, 대부분의 $cPhi^{n}$ 샘플을 측도 $cvarepsilon$의 집합 $cS$로 매핑할 수 있습니다. 이를 통해 기댓값 제곱 거리 $O(	ext{log} 1/cvarepsilon)$ 이내에서 $poly(n/cvarepsilon)$ 시간으로 해결됩니다.



### Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training (https://arxiv.org/abs/2509.21500)
- **What's New**: 본 논문에서는 강화학습 세부 조정(Reinforcement fine-tuning, RFT)이 보상 과최적화(reward over-optimization) 문제에 직면하고 있음을 밝히고 있습니다. 연구진은 보상의 높은 보상(value) 영역에서의 오스펙이 주요한 문제임을 파악하였고, 보상 모델을 효과적으로 개선하기 위해 고급 응답의 구별이 필요함을 강조합니다. 또한 비정상적으로 좋은 대답을 유도할 수 있는 루브릭 기반 보상(rubric-based rewards) 접근법에 초점을 맞추고 있습니다.

- **Technical Details**: 연구진은 루브릭 기반 보상이 응답 품질을 평가하는 데 강력하다는 것을 보여주기 위해 이론적 분석과 실험적 연구를 조화롭게 진행하였습니다. 이 방법은 미리 정의된 기준(criterion)을 사용하여 각 프롬프트에 대한 적절한 평가를 수행하며, 이러한 기준들은 고유한 중요도(weights)를 갖습니다. 루브릭을 이용한 보상 모델은 오프-정책(off-policy) 예시를 활용함으로써, 불필요한 부분에 둔감하게 설계되어 보상의 추적 상의 정확성을 높입니다.

- **Performance Highlights**: 루브릭 기반 보상 방식을 통해 얻은 데이터는 RFT의 보상 과최적화를 상당히 완화시키며, LLM의 포스트 트레이닝(post-training) 개선에서 실질적인 효과를 보여주었습니다. 이 접근법은 고보상 영역의 정확성을 유지함으로써 모델 성능을 높이는 데 기여하며, 연구진은 이 새로운 방법론이 높은 품질의 응답을 일관되게 생성하는데 유리하다는 것을 입증하였습니다. 실험 결과는 루브릭 기반 접근법이 기존의 보상 모델들보다 더 효과적임을 나타냅니다.



### Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning (https://arxiv.org/abs/2509.21487)
Comments:
          Accepted by the Workshop on Efficient Reasoning, Neurips 2025, 5 pages

- **What's New**: 이 논문에서는 Dual-Head Reasoning Distillation (DHRD)라는 새로운 훈련 방법을 소개합니다. 이 방법은 decoder-only language models (LMs)에 두 개의 추가적인 헤드를 결합하여 성능을 향상시킵니다. DHRD는 supervised learning 시에만 reasoning 헤드를 사용하고, 추론 시에는 pooled classifier를 사용하여 throughput을 극대화합니다.

- **Technical Details**: DHRD는 (i) 마지막 토큰 풀링을 사용하는 pooled classification head와 (ii) auxiliary training loss를 위해 설계된 reasoning tower/LM head를 결합합니다. 이 기술은 입력-합리화-레이블 triplet의 정렬을 통해 모델 성능을 극대화합니다. 훈련 시에는 Teacher 모델을 통해 제공된 합리화를 사용하고, 테스트 시에는 reasoning head를 비활성화하여 속도 저하없이 성능을 제공합니다.

- **Performance Highlights**: DHRD는 SuperGLUE의 7개 작업에서 pooled 기준 대비 0.65%에서 5.47%의 상대적 성능 향상을 보여주었습니다. 특히, entailment/causal 작업에서 더 큰 성능 향상이 있었습니다. 테스트 시에는 DHRD가 CoT decoding보다 96에서 142배 더 높은 throughput을 기록하며, baseline latency가 유지되는 것을 확인했습니다.



### Neural Operators for Mathematical Modeling of Transient Fluid Flow in Subsurface Reservoir Systems (https://arxiv.org/abs/2509.21485)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 TFNO-opt라는 신경 연산자 아키텍처를 기반으로 지하 저수지 시스템의 일시적인 유체 흐름 모델링 방법을 제안합니다. 전통적인 수치적 방법이 높은 정확도로 유체 흐름을 모델링할 수 있지만, 처리 시간의 엄청난 비용이 문제로 지적되었습니다. TFNO-opt는 무한 차원 함수 공간에서 PDE 솔루션을 근사화할 수 있도록 해주며, 하드웨어의 한계를 극복하는 새로운 가능성을 열어줍니다.

- **Technical Details**: TFNO-opt 아키텍처는 통합 푸리에 연산자의 내부 시간 해상도를 조정할 수 있는 기능과 스펙트럼 영역에서의 매개변수 텐서 분해를 포함하여 모델의 정확도와 안정성을 높이고 있습니다. 또한, Sobolev norm을 사용하여 오차 함수를 개선하고 초기 조건의 재구성을 통해 물리적 과정을 더욱 정확히 재현할 수 있도록 하고 있습니다. 이러한 방법론은 특히 제어 문제에서 중요한 역할을 합니다.

- **Performance Highlights**: 계산 실험 결과 TFNO-opt는 기존의 수치 시뮬레이터에 비해 최소 6배의 속도 향상을 보여주었습니다. 이는 지하 가스 저장소 모델을 사용한 사례에 기반하며, 논의된 방법의 현실적인 적용 가능성을 강조합니다. 이러한 성과는 복잡한 저수지 시스템을 효과적으로 제어하는 데 새로운 기회를 제공할 것으로 기대됩니다.



### Learning to Reason with Mixture of Tokens (https://arxiv.org/abs/2509.21482)
Comments:
          30 page

- **What's New**: 이번 논문에서는 강화 학습에서 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)의 한계를 해결하기 위해 혼합 토큰 생성(Mixture-of-Token Generation, MoT-G) 방법론을 도입했습니다. MoT-G는 모델이 각 추론 단계에서 여러 토큰에 대한 분포를 유지할 수 있게 하여 기존의 불연속 선택의 제약을 제거합니다. 이를 통해 보다 효율적인 훈련과 탐색이 가능하다는 점에서 큰 의의를 지닙니다.

- **Technical Details**: MoT-G는 기존의 RLVR 프레임워크를 확장하여 연속 혼합 공간에서 작동할 수 있도록 설계되었습니다. 모델은 추론 단계에서 다양한 토큰의 조합을 활용하여 더욱 풍부한 정보처리를 가능하게 하며, 이 구조를 통해 복잡한 문제 해결 과정에서 모델의 불확실성을 유지하고 탐색을 촉진합니다. 전체적 구조는 훈련이 필요 없는 방법들을 포함하여 통합된 형식을 제공합니다.

- **Performance Highlights**: Reasoning-Gym에서의 평가 결과, MoT-G는 표준 디코딩 방법에 비해 7개 작업 중 5~35%의 성능 향상을 보였습니다. 또한 MoT-G 방법은 최종적으로 비슷한 성능을 달성함에 있어 절반의 경로 수만으로도 가능하였으며, 이는 훈련 효율성 증가를 나타냅니다. 추가적인 숨겨진 상태 및 토큰 수준 분석을 통해 MoT-G의 이점이 높은 엔트로피 유지와 탐색 능력에서 기인함을 밝혔습니다.



### Are Hallucinations Bad Estimations? (https://arxiv.org/abs/2509.21473)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 생성 모델(generative models)에서의 환각(hallucination)을 추정치가 실행 가능한 원인과 연결되지 않는 실패로 형식화합니다. 이러한 해석하에, 손실 최소화(loss-minimizing) 최적 추정기조차도 환각을 경험함을 보여줍니다. 이는 손실 최소화와 인간 수용 가능한 출력 간의 구조적 불일치(structural misalignment)를 재구성하며, 잘못된 보정(miscalibration)으로 인한 추정 오류를 설명합니다.

- **Technical Details**: 최고 밀도 영역(Highest Density Regions, HDR) 개념은 주어진 확률 질량을 포함하는 최소 볼륨 집합으로 정의되며, 이는 다변량 밀도를 시각화하는 데 유용한 알고리즘을 활용하여 다중 모드 구조를 보다 잘 드러냅니다. Conditional Density Regions(HCDRs)이라는 개념을 도입하여, 잠재 변수(latent variable)가 하나의 상태를 가질 때 특정 질량의 HDR에서 대상 분포(target distribution)의 기대값이 탈출하는 δ-hallucination의 원인을 설명합니다. 이는 다수의 상태를 가진 잠재 변수와 연관된 분포로 확장됩니다.

- **Performance Highlights**: 코인 집계(coin aggregation), 개방형 QA(open-ended QA), 텍스트-이미지(text-to-image) 실험을 통해 이론을 뒷받침하고 있습니다. 특히, 최적 추정기가 기대하는 출력 도메인에서 손실을 최소화하지만 여전히 δ-hallucinate 될 우연성을 보여주는 위의 다양한 분포가 구성됩니다. 이 연구는 환각 문제를 보다 깊이 이해하고 해결하는 데 기여할 것으로 기대됩니다.



### Score-based Idempotent Distillation of Diffusion Models (https://arxiv.org/abs/2509.21470)
- **What's New**: 이번 연구에서는 Idempotent Generative Networks(IGNs)와 확산 모델(difusion models)을 융합한 Score-based Idempotent Generative Networks(SIGNs)를 제안합니다. SIGNs는 기존 IGNs의 훈련 불안정성을 해결하면서 수치적으로 안정적이고 빠른 샘플링을 가능하게 합니다. 이들은 전처리된 확산 모델로부터 효과적으로 추출된 데이터로 훈련되며, 이를 통해 높은 품질의 샘플을 효율적으로 생성할 수 있습니다.

- **Technical Details**: SIGNs는 확산 모델의 점수(score)에서 아이템포턴트 모델을 증류하여 생성됩니다. 이러한 모델은 다중 단계 샘플링을 지원하며, 무작위 노이즈 샘플들을 데이터 매니폴드(data manifold)로 재투영할 수 있습니다. 훈련의 안정성을 높이기 위해 새로운 점수 기반 훈련 방법론을 제안하며, 기존 IGNs의 훈련 과정에서 나타나는 불안정성을 제거하는 데 목적을 두고 있습니다.

- **Performance Highlights**: 이 모델은 CIFAR 및 CelebA 데이터셋에서 아이템포턴트 모델의 최신 성능을 기록하며, 이전 SOTA IGN 모델보다 41% 이상 더 낮은 FID를 달성했습니다. SIGNs는 신속한 추론을 가능하게 하며 사용자가 품질과 효율 사이에 유연한 조정을 할 수 있게 합니다. 또한, 모델은 제로샷 편집(zero-shot editing)을 통해 입력을 변형하는 능력도 가지고 있습니다.



### Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models (https://arxiv.org/abs/2509.21466)
- **What's New**: 이번 연구는 현대의 Text-to-Image 인공지능 모델이 사우디 아라비아에서의 전문 직종을 묘사할 때 성별 고정관념과 문화적 부정확성을 얼마나 지속하는지를 조사하였습니다. 1,006개의 이미지를 분석하여, 56개의 다양한 사우디 직업에 대해 중립적인 프롬프트(prompts)를 사용했습니다. 이 연구는 기존 모델들이 어떻게 사회적 편향(social biases)을 반영하는지를 드러냅니다.

- **Technical Details**: 연구에서는 ImageFX, DALL-E V3, Grok의 세 가지 인공지능 모델로 생성된 이미지를 각각 평가하였습니다. 두 명의 훈련된 사우디 평가자가 성별, 의상(wardrobe) 및 외모, 배경 및 설정, 활동 및 상호작용, 나이의 다섯 가지 기준으로 이미지를 평가했고, 세 번째 연구자가 불일치할 경우 중재하였습니다. 이러한 과정을 통해 10,100개의 개별 판단을 확보하였습니다.

- **Performance Highlights**: 결과는 성별 불균형이 뚜렷하게 나타났으며, 각 모델의 출력 결과는 ImageFX가 85% 남성, Grok가 86.6% 남성, DALL-E V3는 96% 남성으로 나타났습니다. 특히 DALL-E V3가 가장 강한 성별 고정관념을 보였으며, 이러한 불균형은 리더십 및 기술직에서 특히 두드러졌습니다. 문화적 부정확성도 확인되었고, 이는 꼭 진보적인 묘사라기보다 문화적 오해로 인한 것임을 보여줍니다.



### Enhanced Generative Machine Listener (https://arxiv.org/abs/2509.21463)
- **What's New**: GMLv2는 MUSHRA 점수로 측정된 주관적 오디오 품질 예측을 위해 설계된 레퍼런스 기반 모델입니다. 이 모델은 청취자 평점을 모델링하기 위해 Beta 분포 기반 손실 함수를 도입하고, 신경 오디오 코딩(NAC) 주관적 데이터셋을 통합하여 일반화 능력을 확장합니다. GMLv2는 다양한 테스트 세트에 대한 광범위한 평가를 통해 PEAQ 및 ViSQOL과 같은 널리 사용되는 메트릭보다 일관되게 우수한 성능을 보여줍니다.

- **Technical Details**: GMLv2는 주어진 레퍼런스 신호와 저하된 신호의 쌍에 대한 예상 주관적 품질과 불확실성을 포착하기 위해 Beta 분포의 매개변수를 예측합니다. 이 모델은 역 상관 기능과 Gammatone 필터를 사용하여 주파수 분석을 수행하고, 심층 신경 구조에서 예측합니다. 또한 네트워크 아키텍처는 Inception 블록 기반으로 조정되어 Beta 분포의 매개변수 제약 조건을 수용합니다.

- **Performance Highlights**: GMLv2는 다양한 콘텐츠 유형과 코덱 구성에서도 주관적 점수와의 상관관계와 정확한 예측에서 우수한 성능을 보여줍니다. 이로 인해 GMLv2는 현대 오디오 코딩 기술의 연구 및 개발을 촉진할 수 있는 확장 가능하고 자동화된 프레임워크를 제공합니다. 또한 추정된 품질과 불확실성을 함께 분석함으로써 더욱 신뢰할 수 있는 평가 결과를 제공합니다.



### A State-of-the-Art SQL Reasoning Model using RLVR (https://arxiv.org/abs/2509.21459)
- **What's New**: 본 연구에서는 Reinforcement Learning with Verifiable Rewards (RLVR)를 적용하여 BIRD라는 데이터 과학 벤치마크에서 조직 특화 지식을 포함하는 맞춤형 추론 모델을 개발했습니다. 이 접근 방식은 일반적인 데이터 과학 작업인 text2sql의 성과를 향상시켰으며, 75.68%의 정확도로 최고 성과를 기록했습니다. 이를 통해 데이터 과학, 비즈니스 인텔리전스 및 코딩 분야에서의 응용 가능성을 보여주고 있습니다.

- **Technical Details**: RLVR은 사전 학습된 LLM을 미세 조정하여 주어진 작업에 맞춰 보상 함수를 사용하여 목표 진리를 측정하는 후처리 학습 패러다임입니다. 이 방법은 SQL 코드 생성과 같이 객관적으로 측정할 수 있는 과제에 적합하며, BIRD 벤치마크를 통해 LLM을 어떻게 RLVR을 통해 미세 조정하는지에 대한 전략을 제시합니다. 본 연구에서 제안하는 방법론은 데이터셋의 특성과 리소스 요구에 따라 다를 수 있는 다양한 접근 방식을 지원합니다.

- **Performance Highlights**: 모델의 성능 측정을 위해 BIRD 벤치마크의 학습 데이터 세트만을 사용하여 75.68%의 정확도를 기록했습니다. 또한, self-consistency(자기 일관성) 방법을 추가하여 더 적은 생성 과정으로 두 번째 최고 성능을 초과하는 결과를 보였습니다. 우리의 연구 결과는 RLVR이 복잡한 기업 과제에도 효과적으로 적용될 수 있음을 시사합니다.



### ARTI-6: Towards Six-dimensional Articulatory Speech Encoding (https://arxiv.org/abs/2509.21447)
- **What's New**: 본 논문에서는 실시간 MRI 데이터에서 수집된 정보를 바탕으로 한 6차원 발음 장치 음성 인코딩 프레임워크인 ARTI-6을 제안합니다. 이 모델은 인체 생리학적으로 기반을 두고 있으며, 음성 생성을 위한 해석 가능하고 효율적인 방법을 제공합니다. ARTI-6는 발음 역전(articulatory inversion)과 음성 합성(articulatory synthesis)을 개선하기 위해 중요한 음성 기관을 다루며, 저차원의 표현이 자연스러운 음성을 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 발음 생산의 목표를 달성하는 데 필요한 6개의 주요 지역을 선택하였습니다: Lip Aperture (LA), Tongue Tip (TT), Tongue Body (TB), Velum (VL), Tongue Root (TR), 그리고 Larynx (LX). 발음 역전 모델은 Whisper, WavLM, HuBERT, Wav2Vec2와 같은 음성 기초 모델을 기반으로 구성되어 있으며, 이를 통해 얻은 결과는 고해상도의 발음 관련 데이터를 제공합니다. 합성 모델은 HiFi-GAN 아키텍처를 사용하고, ECAPA-TDNN에서 추출된 화자 임베딩으로 조건화하여 개별 화자의 특징을 반영할 수 있습니다.

- **Performance Highlights**: ARTI-6의 발음 역전 성능은 WavLM이나 HuBERT를 사용할 때 예측 상관 계수 0.872를 기록하며 가장 우수한 결과를 보였습니다. 발음 합성에서는 LSS 데이터셋과 LibriTTS-R 데이터셋을 조합하여 훈련되었으며, 다양한 손실 함수를 통해 모델의 동작이 최적화되었습니다. 실험 결과, ARTI-6는 실시간 애플리케이션에 적합한 저차원 표현을 제공함으로써 발음 기술 개발에 유용한 플랫폼이 될 것으로 기대됩니다.



### One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning (https://arxiv.org/abs/2509.21443)
Comments:
          22 pages, 11 figures, 6 tables

- **What's New**: 본 연구는 다문화 및 다언어 환경에서의 도덕적 추론과 관련된 종합적인 조사 결과를 발표합니다. 기존의 LLM(대형 언어 모델)은 주로 영어 기반의 데이터로 사전 훈련되어 문화적 맥락이나 언어적 차이를 고려하지 못하는 경향이 있습니다. 연구는 도덕적 판단의 일관성과 문화적 불일치를 다루고, LLM이 보편적인 도덕 원칙을 얼마나 잘 일반화하는지를 평가합니다.

- **Technical Details**: 연구에서는 MoralExceptQA 및 ETHICS와 같은 도덕적 추론 벤치마크를 다섯 개의 언어(중국어, 독일어, 힌디어, 스페인어, 우르두)로 번역하여 다국어로 제로샷 평가를 실시했습니다. 이를 통해 LLM의 응답 차이를 분석하였으며, 특히 도덕적 판단에서 문화적 요인이 어떻게 작용하는지를 탐구했습니다. 추가적으로, 사전 훈련 데이터가 LLM의 도덕적 방향성에 미치는 영향을 사례 연구를 통해 설명했습니다.

- **Performance Highlights**: 결과적으로 LLM은 언어에 따라 도덕적 추론이 현저하게 다르게 나타나며, 특히 영어에서 가장 높은 성능을 보였습니다. 저자들은 이러한 경향이 낮은 자원 언어에서 더욱 두드러진다고 언급하며, LLM이 정확한 판단을 하거나 지침을 따르는 데 어려움을 겪고 있음을 보여줍니다. 연구결과는 AI 윤리의 접근 방식이 더 문화적으로 포괄적이 되어야 한다는 필요성을 강조합니다.



### Foundation models for high-energy physics (https://arxiv.org/abs/2509.21434)
Comments:
          To be submitted to SciPost Physics Proceedings (EuCAIFCon 2025)

- **What's New**: 이 논문은 고에너지 물리학 분야에서 foundation models의 적용 가능성을 탐구하는 최초의 리뷰 논문입니다. 최근 자연어 처리 및 컴퓨터 비전 분야에서 혁신을 가져온 대규모 머신러닝 모델이 고에너지 물리학 데이터에 어떻게 활용될 수 있을지 관심이 커지고 있습니다. 이 논문은 기존 연구들을 요약 및 논의하며, 이러한 모델들이 물리학 연구에 직접 활용될 수 있는지에 대한 질문에 집중하고 있습니다.

- **Technical Details**: foundation models의 정의는 대량의 데이터로 학습되며, 다양한 다운스트림 작업에 적응할 수 있도록 설계된 모델입니다. 이러한 모델들은 일반적으로 self-supervision을 통해 훈련되며, 최근 고에너지 물리학 커뮤니티에서도 큰 관심을 받고 있습니다. 이 리뷰에서는 데이터로부터의 전이 학습 (transfer learning) 개념을 바탕으로, 이러한 모델들이 어떻게 구성되고 수행되는지에 대해 다룰 것입니다.

- **Performance Highlights**: OmniJet라는 첫 번째 cross-task foundation model의 예시를 통해, 입자 제트의 생성 및 분류가 가능함을 보여줍니다. 이 모델은 pT, Δη 등을 포함한 물리적 피처를 벡터 양자화 변분 오토인코더(VQ-VAE)를 통해 처리하며, 다음 항목 예측(next-token prediction) 기반의 자기 지도 학습 방식으로 훈련됩니다. 초기 연구에서는 OmniJet 모델이 무작위 초기화된 모델보다 뛰어난 성능을 발휘했으며, 특히 소규모 데이터셋에서 좋은 성과를 보였습니다.



### DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation (https://arxiv.org/abs/2509.21433)
- **What's New**: 이 논문에서는 텍스트-이미지 확산 모델의 다중 개념 지우기(multi-concept erasure)에서 기존 방법의 한계를 극복하기 위한 새로운 프레임워크인 DyME를 제안합니다. 기존 방식들은 정적 지우기(static erasure)에 의존하여 복잡한 개념 요청을 처리하는 데 한계를 보였으며, 이를 해결하기 위해 개념별 LoRA 어댑터를 훈련하여 필요에 따라 동적으로 조합하는 방식을 채택했습니다. DyME는 더욱 유연하고 효과적인 다중 개념 지우기를 가능하게 하며, 이를 통해 법적 및 윤리적 문제를 완화할 수 있습니다.

- **Technical Details**: DyME는 각 개념에 특정한 경량의 LoRA 어댑터를 훈련하여, 추론(inference) 시 필요한 어댑터만 동적으로 조합하여 사용합니다. 이 프레임워크는 정적 방식과는 달리, 훈련과 추론을 분리하여 특정 세대에 필요한 개념만 선택적으로 지웁니다. 또한, 두 가지 수준(기능 및 매개변수)에서 직교성 제약(bi-level orthogonality constraints)을 도입하여 여러 어댑터 간의 간섭(crosstalk)을 최소화하고, 신뢰성 있는 다중 개념 지우기를 실현합니다.

- **Performance Highlights**: ErasureBench-H와 표준 데이터셋(CIFAR-100, Imagenette)에서의 실험 결과, DyME는 기존 방식보다 우수한 성능을 발휘하였으며, 90% 이상의 조화 정확도(harmonic accuracy)를 기록했습니다. 특히, 지우기 범위가 확장되더라도 DyME는 모든 기준선보다 명확한 우위를 유지하는 것으로 나타났습니다. 이 연구는 다중 개념 지우기의 확장성을 체계적으로 조사한 최초의 사례로, 실용적인 평가 방법을 제공합니다.



### PhenoMoler: Phenotype-Guided Molecular Optimization via Chemistry Large Language Mod (https://arxiv.org/abs/2509.21424)
- **What's New**: 본 연구에서는 PhenoMoler라는 새로운 분자 생성 프레임워크를 제안합니다. 이 프레임워크는 약물 유도 전사 프로파일을 통합하여 생물학적으로 유의미한 약물 설계를 가능하게 합니다. 기존의 방법들이 약물-타겟 결합 친화도와 특이성에 초점을 맞추었던 반면, PhenoMoler는 표현 프로파일을 통해 분자 설계를 가이드합니다.

- **Technical Details**: PhenoMoler는 약리학적 특성을 고려하여 화학적으로 유효한 분자를 생성하기 위해 사전 훈련된 자가 회귀 언어 모델과 1차원 합성곱 신경망(1D-CNN)을 사용합니다. 이 과정에서 특정 화학 하위 구조(예: 스캐폴드, 사이드 체인, 링커)를 선택적으로 마스킹하고 복원하여 교차 주의(Cross-attention) 메커니즘을 통해 전사적 프로파일과 분자 맥락을 융합합니다. 이렇게 훈련된 시스템은 생물학적 관련성과 구문적으로 유효한 분자를 생성하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, PhenoMoler는 화학적으로 유효하고, 새로운 다양한 분자를 생성하며, 원하는 표현 프로파일과 일치하는 결과를 나타냅니다. FDA 승인 약물과 비교했을 때, 생성된 화합물들은 유사하거나 더 나은 약물 유사성(QED)과 최적화된 물리 화학적 속성을 보여주며, 주요 암 타겟에 대한 결합 친화도도 우수합니다. 이러한 결과는 PhenoMoler가 표현 유도 및 구조 제어 기반의 분자 최적화에 대한 잠재력을 강조합니다.



### Near-Optimal Experiment Design in Linear non-Gaussian Cyclic Models (https://arxiv.org/abs/2509.21423)
- **What's New**: 이 논문에서는 사이클이 포함될 수 있는 선형 비가우시안 구조적 인과 모델(Structural Causal Models, SCMs)에서 관찰 데이터와 개입(interventional data)의 조합을 통한 인과 구조 학습 문제를 다룬다. 최근의 연구에 따르면 단순한 관찰 데이터만으로는 인과 그래프를 특정할 수 없으며, 이 문제를 해결하기 위한 조합적 특성을 확립하였다. 각 그래프는 이분 그래프(bipartite graph)에서 완전 매칭(perfect matching)과 맵핑되며, 이로 인해 개입이 인과 그래프를 어떻게 조정하는지를 분석할 수 있게 되었다.

- **Technical Details**: 저자들은 선형 비가우시안 가정 하에, 관찰 데이터만으로 인과 그래프를 동치 클래스(equivalence class)까지 특정할 수 있음을 입증하였다. 그들은 이 동치 클래스를 이분 그래프의 완전 매칭에 매핑하여 조합적 특성을 제공하였다. 또한, 인과 그래프의 응축 그래프(condensation graph) 또는 강하게 연결된 구성 요소(Strongly Connected Components, SCCs)를 식별할 수 있음을 보였다.

- **Performance Highlights**: 저자들은 개입 설계 최적화 문제를 적응형 확률 최적화(adaptive stochastic optimization) 문제로 공식화하였으며, 개입이 동치 클래스에서 몇 개의 그래프를 제거하는지를 정량화하는 보상 기능(reward function)을 제안하였다. 이 보상 함수는 적응형 서브모듈적(adaptive submodular)이고, 그리디 정책(greedy policy)을 사용하여 개입 설계에서의 근사 최적 성능 보장을 제공하였다. 시뮬레이션 결과에 따르면, 소수의 개입이 참된 인과 구조를 복원하는 데 효과적임이 입증되었다.



### How Large Language Models Need Symbolism (https://arxiv.org/abs/2509.21404)
- **What's New**: 이 논문에서는 AI의 미래가 단순한 규모 확장을 넘어선다는 점을 주장합니다. 진정한 발견을 이끌어내기 위해, 대형 언어 모델은 그들의 강력하지만 맹목적인 직관을 안내할 인간이 만든 심볼(symbols)이 필요하다고 강조합니다.

- **Technical Details**: 대형 언어 모델은 방대한 양의 데이터에서 학습하지만, 이 모델들이 실질적인 혁신을 이루기 위해서는 의미 있는 방향성을 제공할 필요성이 있습니다. 이 연구는 이러한 방향성을 제공하기 위한 방법론과 접근 방식을 제안합니다.

- **Performance Highlights**: 이번 연구는 AI 모델의 성능 향상을 위한 새로운 방향성을 제시하며, 기계 학습(machine learning) 분야에서의 심볼 중심의 접근 방식의 중요성을 강조합니다. 이를 통해 AI의 발견 가능성을 극대화할 수 있는 방법论을 모색합니다.



### Large AI Model-Enabled Generative Semantic Communications for Image Transmission (https://arxiv.org/abs/2509.21394)
Comments:
          Accepted to the IEEE GLOBECOM 2025

- **What's New**: 이 논문은 이미지 전송 효율성과 정확성을 향상시키기 위해 새로운 생성적 의미 통신 시스템을 제안합니다. 이 시스템은 이미지를 중요 영역과 비중요 영역으로 세분화하여 시멘틱(semantic) 정보의 정밀도를 높입니다. 주 영역은 이미지 지향 시멘틱 인코더로 처리하고, 비주요 영역은 텍스트 모델링을 통해 효율적으로 압축됩니다.

- **Technical Details**: 제안된 시스템은 이미지를 중요한 객체 및 장면 요소가 포함된 주 영역과 비주요 배경 지역으로 나누는 의미 인식 이미지 세분화 메커니즘을 핵심 혁신으로 합니다. 주 영역은 이미지 시멘틱 인코더를 통해 전송되고, 비주요 부분은 구조화된 텍스트 프롬프트로 인코딩되어 시멘틱 정보를 보존합니다. 노이즈가 포함된 무선 채널을 통해 전송된 인코딩된 신호는 역확산 프로세스를 통해 고충실도의 이미지를 복원하는 데 사용됩니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안한 시스템이 전통적인 방법에 비해 의미 충실도와 시각적 품질 모두에 있어서 우수한 성능을 보임을 보여줍니다. 특히 낮은 SNR(신호 대 잡음비) 조건에서 기존 DeepJSCC 접근 방식보다 월등히 나은 성과를 달성하였습니다. 실험을 통해 각 전송된 구성 요소가 최적의 시스템 성능을 달성하는 데 중요한 역할을 한다는 것을 확인했습니다.



### MIXRAG : Mixture-of-Experts Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (https://arxiv.org/abs/2509.21391)
- **What's New**: MIXRAG (Mixture-of-Experts Graph-RAG) 프레임워크는 복수의 전문 그래프 검색기를 도입하여 다양한 쿼리 의도를 더 효과적으로 처리합니다. 이 프레임워크는 각 검색기가 엔터티, 관계 또는 서브그래프의 토폴로지와 같은 그래프 의미의 특정 측면에 집중하도록 훈련됩니다. 또한, 쿼리 인식 그래프 인코더를 통해 검색된 서브그래프에서 가장 관련성 높은 정보만 강조하여 잡음(noise)을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MIXRAG는 전문가 기반 접근법을 사용하여 여러 검색기 전문가를 동적으로 결합합니다. 각각의 검색기는 각기 다른 형태의 정보를 다루며, 쿼리의 의도와 전문가의 전문성에 기반하여 적합한 검색기를 선택하는 Mixture-of-Experts 모듈이 통합되어 있습니다. 이러한 구조는 쿼리에 적합한 정보 검색이 이루어지도록 하여 보다 정교한 추론이 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MIXRAG는 GraphQA 벤치마크에서 기존의 강력한 기준선을 초월하는 성능을 보였습니다. 전문가 기반 검색 및 세밀한 서브그래프 임베딩 기법이 하부 성능에 큰 기여를 함을 보여줍니다. 이러한 연구 결과는 복합적인 지식을 포착하고 활용하는 데 있어 MIXRAG의 효과를 뒷받침하고 있습니다.



### Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey (https://arxiv.org/abs/2509.21389)
Comments:
          34 pages, 16 figures, IEEE Communication Surveys and Tutorials

- **What's New**: 본 설문조사는 Federated Learning (FL)과 네트워크 침입 탐지 시스템(Network Intrusion Detection Systems, NIDS)의 통합을 탐구하며, 특히 심층 학습(Deep Learning) 및 양자 기계 학습(Quantum Machine Learning) 접근 방식에 중점을 두고 있습니다. FL은 데이터 개인 정보 보호를 유지하면서 분산된 장치 간에 협력 모델 학습을 가능하게 하며, 이는 민감한 트래픽 데이터를 중앙 집중화하지 않아야 하는 네트워크 보안 상황에서 매우 중요합니다. 또한 양자 FL(QFL)에 대한 탐구를 최초로 포함하여, 복잡한 패턴 인식을 위한 양자 특화 집계 방법 등을 논의합니다.

- **Technical Details**: FL 아키텍처, 배포 전략, 통신 프로토콜 및 공격 전용 연합 솔루션을 체계적으로 분석하며, DDoS, MITM 및 봇넷 공격에 대한 잠재적 위협을 상세히 조사합니다. FL은 처리된 센서 측정값을 전송하는 대신 모델 매개변수만 서버와 통신하게 하여, 기존 중앙 집중화 방법보다 통신 비용을 줄이고 데이터 개인 정보를 보호할 수 있습니다. 이를 통해 IoT 장치의 제한된 컴퓨팅 능력을 고려한 통합 모델의 동적 업데이트가 가능합니다.

- **Performance Highlights**: 심층 학습(DL)은 복잡한 네트워크 공격 감지에서 뛰어난 정확성을 보여주며, 다양한 네트워크 조건에서 범용화 할 수 있는 잠재력을 지니고 있습니다. 일반적인 머신 러닝 모델은 수작업으로 제작한 특징을 필요로 하지만, DL은 이러한 과정을 자동화하여 성능을 극대화할 수 있습니다. 양자 기계 학습(QML) 분야의 성장 또한 데이터의 차원과 복잡성이 늘어남에 따라 중요한 발전으로 평가되며, 양자 현상을 활용하여 IDS의 효율성과 안정성을 높일 수 있을 것으로 기대됩니다.



### Do Sparse Subnetworks Exhibit Cognitively Aligned Attention? Effects of Pruning on Saliency Map Fidelity, Sparsity, and Concept Coherenc (https://arxiv.org/abs/2509.21387)
Comments:
          4 pages

- **What's New**: 본 논문에서는 신경망 프루닝(pruning)이 모델의 해석 가능성에 미치는 영향을 조사합니다. 저수준 atribtion maps와 고수준 concept representations에 대한 두 가지 차원에서 분석을 수행하며, ResNet-18을 기반으로 한 실험을 통해 프루닝의 다양한 강도가 saliency maps와 모델의 개념 표현에 미치는 변화를 평가합니다. 이 연구는 프루닝이 내부 표현을 인간의 주의 패턴에 더 가깝게 형성할 수 있으나, 과도한 프루닝은 해석 가능성을 저해할 수 있음을 제시합니다.

- **Technical Details**: 연구에서는 활동 수준의 변화를 살펴보기 위해 ResNet-18 모델을 ImageNette 데이터셋에서 훈련 후 전역 크기 기반 프루닝(global magnitude pruning)을 적용하고 세부적으로 다양한 프루닝 수준에서의 saliency maps를 생성합니다.  Vanilla Gradients(VG)와 Integrated Gradients(IG)를 통해 post-hoc 설명이 어떻게 변화하는지를 정량적으로 종합하여 평가합니다. CRAFT 기반의 개념 추출 기술을 통해 프루닝 단계에서 학습된 개념의 의미 일관성의 변화를 추적하여 고수준 개념 분석을 수행합니다.

- **Performance Highlights**: 경량에서 중간 정도의 프루닝은 saliency map의 초점과 정확성을 향상시키며, 시맨틱적으로 의미 있는 개념을 보존합니다. 하지만 공격적인 프루닝이 시행되면 다양한 특징이 병합되어 saliency map의 희소성과 개념의 일관성이 줄어드는 반면, 정확성은 유지되는 경향을 보였습니다. 이러한 결과는 적절한 프루닝이 내부 표현을 보다 유사한 인간 주의 패턴으로 형성할 수 있음을 보여주며, 프루닝의 양과 해석 가능성 간의 미묘한 균형이 필요함을 강조합니다.



### Toward a Realistic Encoding Model of Auditory Affective Understanding in the Brain (https://arxiv.org/abs/2509.21381)
- **What's New**: 이 연구는 affective neuroscience와 emotion-aware AI 분야에서 복잡한 청각 자극이 감정 각성 동역학에 미치는 영향을 이해하기 위한 계산 프레임워크를 도입합니다. 이 프레임워크는 세 개의 데이터 세트(SEED, LIRIS, 자가 수집한 BAVE)를 이용하여 자연주의적 청각 입력을 행동적 및 신경적 반응으로 모델링합니다. 특히, wav2vec 2.0과 Hubert의 최종 계층에서 파생된 고수준의 의미 표현이 감정 인코딩에 있어 지배적인 역할을 한다는 발견이 주목됩니다.

- **Technical Details**: 연구에 사용된 음향 분해 전략은 계층적 청각 처리의 원칙을 바탕으로 하며, 이는 인간 뇌의 주의 메커니즘과 밀접하게 연결되어 있습니다. 이 연구는 원음, 분리된 음성 및 배경 사운드트랙으로부터 청각 데이터의 계층적 및 다원적 구성을 모델링하여, 이들 요소가 감정 관련 반응과의 연관성을 정량적으로 분석합니다. 본 연구는 고급 DNN 모델을 활용하여 세부적인 감정의 동적 동역학을 이해하고자 하며, EEG 신호를 통해 신경 동조성을 측정하여 감정-행동 연관성을 탐구합니다.

- **Performance Highlights**: 연구 결과, wav2vec 2.0 및 Hubert의 중간 계층이 감정 유도에서 최종 계층보다 뛰어난 성과를 보이며, 이는 다양한 데이터 세트에서 일관되게 나타났습니다. 인간의 목소리와 배경 사운드트랙은 데이터 세트에 따라 서로 다른 감정을 유발하는 편향을 나타내며, 특히 자극 에너지가 높은 LIRIS 데이터 세트에서는 배경 음악이 우세합니다. 이러한 발견은 감정 인코딩의 기저 메커니즘에 대한 이해를 심화시키고, 향후 오디오-감정 상호작용 연구에 기여할 수 있는 토대를 마련합니다.



### SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders (https://arxiv.org/abs/2509.21379)
- **What's New**: 본 논문에서는 텍스트-이미지 확산 모델에서 개념을 효과적으로 학습 해제하기 위한 새로운 방법인 SAEmnesia를 소개합니다. 이 접근법은 개념-신경세포 매핑을 체계적으로 라벨링하여 단일 신경세포가 하나의 개념에만 대응하도록 하여, 특징 분할(feature splitting)을 방지합니다. SAEmnesia를 통해 기능 중앙 집중화(feature centralization)를 달성하고, 더 나은 해석 가능성을 제공하여 모델이 생성하는 내용의 제어를 가능하게 합니다.

- **Technical Details**: SAEmnesia는 감독된 희소 오토인코더(sparse autoencoder) 훈련 방법으로, 훈련 과정에서 교차 엔트로피 계산(cross-entropy computation)이라는 최소한의 계산적 오버헤드를 추가합니다. 이 방법은 특화된 신경세포를 학습하여 미세조정 없이도 효과적인 개념 해제를 가능하게 하며, 이론적으로 각 개념을 단일 잠재 특성(latent feature)에 로컬라이즈하여 보다 정밀한 개입(intervention)을 수행합니다. 또한, TopK SAEs를 통해 희소성 제어(sparsity control)를 강화합니다.

- **Performance Highlights**: UnlearnCanvas 벤치마크에서 SAEmnesia는 최신 기법에 비해 9.22%의 성능 향상을 달성하였으며, 9개의 객체를 제거하는 연속 학습 해제(task)에서 28.4% 향상된 정확도를 보여주었습니다. 이 접근법은 추론 발생 시 하이퍼파라미터 검색을 96.67% 감소시켜, 전체적인 효율성을 크게 향상시킵니다. 이러한 성과는 SAEmnesia가 원치 않는 내용을 효과적으로 제거하면서도 생성 품질을 유지할 수 있음을 시사합니다.



### Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (https://arxiv.org/abs/2509.21377)
Comments:
          Main paper (8 pages). Accepted for publication by ECAI( European Conference on Artificial Intelligence) 2025

- **What's New**: 이번 연구에서는 Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (DMTF-AVN)이라는 새로운 프레임워크를 제안합니다. DMTF-AVN은 시각적 데이터를 더 깊이 있게 이해하고 오디오 신호와의 상호작용을 최적화하는 방식으로, 기존 방법들을 능가하는 성능을 보여주고 있습니다. 이 접근법은 멀티모달(Multimodal) 정보 융합에서 선택적인 정보 처리 방법을 통해, 로봇 navigation의 정확도를 개선합니다.

- **Technical Details**: DMTF-AVN 모델은 정보 추출, 멀티 타겟 특징 추출 전략, GRU(Gated Recurrent Unit), 액터-크리틱 네트워크 구조의 4가지 핵심 모듈로 구성되어 있습니다. 정보 추출 모듈은 원시 모달 데이터(raw modality data)를 고정 차원의 임베딩 벡터로 변환합니다. 여기서, GRU는 데이터 시퀀스의 시간적 동적을 캡처하여 의사결정 과정에 도움을 줍니다.

- **Performance Highlights**: DMTF-AVN은 Replica 및 Matterport3D 데이터셋에서 기존 방법들보다 더 뛰어난 성공률(SR), 경로 효율성(SPL), 장면 적응(SNA) 성과를 기록했습니다. 또한, 이 모델은 강력한 확장성 및 일반화 능력을 갖추고 있어 로봇 내비게이션에서의 고급 멀티모달 융합 전략에 대한 기반을 마련하고 있습니다. 제안된 방식은 멀티모달 데이터의 중복을 필터링하여 효과적인 정보 처리를 가능하게 합니다.



### In silico Deep Learning Protocols for Label-Free Super-Resolution Microscopy: A Comparative Study of Network Architectures and SNR Dependenc (https://arxiv.org/abs/2509.21376)
Comments:
          20 pages, 10 figures

- **What's New**: 이번 연구는 기존의 고가 장비나 전문 기술 없이도 초해상도(optical super-resolution) 광학 현미경을 달성할 수 있는 경제적인 대안을 제시합니다. 특히, 형광(fluorescent) 모드 대신 비형광(non-fluorescent) 위상 변조 현미경 방법인 Zernike phase contrast(PCM)와 differential interference contrast(DIC) 현미경을 활용합니다. 이를 통해 일반 사용자가 접근할 수 있는 새로운 초해상도 방법론이 제안됩니다.

- **Technical Details**: 연구에서는 O-Net과 Theta-Net이라는 두 가지 심층 신경망(deep neural network) 아키텍처를 개발하여, 원자 힘 현미경(atomic force microscopy)을 통해 보정된 나노스케일(nanoscale) 특성을 가진 커스텀 테스트 대상을 해상하는 능력을 평가합니다. 결과적으로 O-Net과 Theta-Net 모델은 높은 신호 대 잡음 비(signal-to-noise ratio, SNR)에서 다른 성능을 보이며, 서로 보완적(supplementary)인 접근법으로 사용될 수 있음을 보여줍니다.

- **Performance Highlights**: O-Net 모델은 높은 SNR에서 더 뛰어난 성능을 발휘하며, 반면 저가의 SNR에서는 Theta-Net 모델이 더욱 효과적인 경향을 보였습니다. 이러한 결과는 비형광 광학 나노스코피에서 DNN 모델의 아키텍처와 이미지 SNR의 중요성을 강조합니다. 동일한 교육 데이터셋(training dataset)과 에폭 수를 사용할 때에도 각 모델의 성능 차이는 여전히 뚜렷하게 나타났습니다.



### Automated Prompt Generation for Creative and Counterfactual Text-to-image Synthesis (https://arxiv.org/abs/2509.21375)
Comments:
          text-to-image generation, automatic prompt, DPO, Counterfactual

- **What's New**: 이 논문에서는 텍스트-이미지 생성 분야에서 정확성과 창의성을 높이기 위한 새로운 접근 방식인 자동 프롬프트 엔지니어링 프레임워크를 제안합니다. 특히 기존의 데이터셋 부족 문제를 해결하기 위해 'counterfactual size'에 초점을 맞추어 작은 객체가 큰 객체보다 더 크게 보이는 이미지를 생성하는 방법을 연구합니다. 이 프레임워크는 이미지 평가자, 프롬프트 재작성기, 프롬프트 랭커라는 세 가지 구성 요소로 이루어져 있으며, 이를 통해 더 나은 결과를 도출할 수 있습니다.

- **Technical Details**: 프레임워크의 핵심 요소 중 하나는 이미지 평가자로, 텍스트 프롬프트로부터 생성된 이미지가 'counterfactual size' 기준을 얼마나 잘 충족하는지를 평가합니다. 이 과정에서 세분화 마스크(Segmentation Mask)와 CLIP 모델을 활용하여 정확한 객체 검출과 레이블 확인(Verification)을 수행합니다. 프롬프트 재작성기는 감독 학습(Supervised Learning)을 통해 긍정적인 프롬프트 세트에 대해 학습하여 신뢰할 수 있는 카운터팩추얼 이미지 생성을 위한 후보 프롬프트를 만들고, 프롬프트 랭커는 직접 선호 최적화(Direct Preference Optimization) 방식으로 최상의 후보를 선택합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 최신의 텍스트-이미지 생성 자동 프롬프트 재작성 기법과 ChatGPT-4o를 능가하는 성능을 보였습니다. 특히, 새로운 'counterfactual size' 이미지 생성을 위한 데이터셋을 구축하고, 이를 통해 보다 정확한 결과를 얻어, 텍스트-이미지 생성의 창의성 및 탐색적 응용 분야에서의 가능성을 높였습니다. 이 연구는 앞으로의 카운터팩추얼 제어 가능성 연구의 기초를 다지는 중요한 계기가 될 것입니다.



### ReGeS: Reciprocal Retrieval-Generation Synergy for Conversational Recommender Systems (https://arxiv.org/abs/2509.21371)
Comments:
          Accepted by WISE 2025: 26th International Web Information Systems Engineering conference. Our code is publicly available at the link: this https URL

- **What's New**: 기존의 대화 기반 추천 시스템(CRS)은 외부 도메인 지식을 연결하는 데 어려움을 겪었지만, ReGeS라는 새로운 프레임워크는 이를 해결하는 데 혁신적인 방법을 제시합니다. ReGeS는 생성-증강 검색(generation-augmented retrieval)과 검색-증강 생성(retrieval-augmented generation)을 결합하여 사용자 의도를 보다 명확히 추출하고, 항목 간의 미세한 차이를 구분할 수 있게 합니다. 이를 통해 추가적인 주석 없이도 높은 품질의 추천을 제공하며 실험 결과에서도 최첨단 성능을 입증했습니다.

- **Technical Details**: ReGeS는 대화 내용을 기반으로 사용자의 의도를 파악하고 추천 항목을 선택하는 두 가지 주요 모듈을 포함합니다. 첫 번째 모듈인 생성-증강 검색은 LLM을 통해 간결하고 정보가 풍부한 쿼리를 생성하여 정확한 항목 검색을 돕습니다. 두 번째 모듈인 검색-증강 생성은 검색된 후보 항목을 사용하여 최종 추천의 정확성을 높이며, 이를 통해 노이즈가 있는 입력과 후보 항목 간의 구분 문제를 효율적으로 해결합니다.

- **Performance Highlights**: 여러 CRS 벤치마크에서 ReGeS는 추천 정확성에서 최첨단 성능을 달성했습니다. 이 프레임워크는 특히 능동적으로 사용자 의도를 반영하고, 대화의 지식적 수요를 충족시키는 데 효과적입니다. 회귀를 통한 체계적인 접근 방식 덕분에, ReGeS는 과거 방식들과 비교해 허위 정보를 줄이고 최신 상태로의 업데이트를 쉽게 만들어, 대화 기반 추천 시스템의 신뢰성을 크게 개선합니다.



### Safety Assessment of Scaffolding on Construction Site using AI (https://arxiv.org/abs/2509.21368)
- **What's New**: 본 논문에서는 건설 산업에서 안전성 평가의 중요성을 강조하며, 특히 비계(scaffolding)에 대한 인공지능(AI) 기반의 검사 방법을 제안합니다. 기존의 시각적 검사가 시간 소모적이며 인간의 오류에 취약한 문제를 해결하고자 합니다. 새로운 클라우드 기반 AI 플랫폼을 통해 비계 구조의 점 구름(point cloud) 데이터를 처리하고 분석하는 방법을 탐구합니다.

- **Technical Details**: 제안된 시스템은 인증된 참조 데이터와 최근 점 구름 데이터의 비교 및 평가를 통해 구조적인 수정사항을 탐지합니다. 이는 자동화된 모니터링을 가능하게 하여 수동 검사에 필요한 시간과 노력을 줄이며, 전체적인 안전성을 향상시키는 데 기여합니다. AI와 디지털화를 통해 더욱 정확한 검사 절차가 가능해질 것으로 기대됩니다.

- **Performance Highlights**: AI 기반 검사 시스템은 건설 현장에서의 안전성을 높이고, 수작업 검사보다 효율적인 모니터링을 지원합니다. 이러한 접근 방식은 건설자산의 신뢰성 및 작업자 안전성을 보장하는 데 중요한 역할을 할 것으로 나타났습니다. 비계의 구조적 변화 감지에 있어 실질적인 진전을 이루며, 보다 안전한 건설 환경 조성에 기여할 수 있습니다.



### Design and Implementation of a Secure RAG-Enhanced AI Chatbot for Smart Tourism Customer Service: Defending Against Prompt Injection Attacks -- A Case Study of Hsinchu, Taiwan (https://arxiv.org/abs/2509.21367)
Comments:
          12 pages, 7 figures, 5 tables

- **What's New**: 이번 연구는 대만 신주에서 스마트 관광 서비스를 위한 안전한 Retrieval-Augmented Generation (RAG) 챗봇의 설계 및 구현에 대한 사례 연구입니다. RAG는 API 기능 호출, 다층 언어 분석 및 인젝션에 대한 방어 장치를 통합하여 높은 맥락 인식 및 보안을 달성합니다. 이 챗봇은 다단계 응답 전략과 RAG 기반 지식 접지를 포함하여 사용자의 요구에 맞춘 개인화된 실시간 지원을 제공합니다.

- **Technical Details**: 챗봇은 RAG 아키텍처를 통해 사실과 교정을 청크로 나누고, OpenAI의 텍스트 임베딩을 활용하여 의미 검색을 수행합니다. 다섯 가지 수준의 언어학적 구문 분석과 API 호출을 통해 다이내믹한 반응을 지원하며, GPT-5를 통합하여 모델 비교를 실시합니다. 방어 체계로는 시스템 규범, 의도 판단을 위한 게이트키퍼, 신뢰할 수 있는 데이터 우선 순위를 위한 역 RAG 텍스트가 포함되어 있습니다.

- **Performance Highlights**: 674개의 적대적 프롬프트와 223개의 안전한 쿼리에 대한 평가 결과, 안전한 작업에서 95% 이상의 정확도를 기록했습니다. GPT-5는 약 85%의 공격을 차단하며, 차단 메커니즘의 유효성을 입증합니다. 이러한 결과는 지속 가능한 관광, 다국어 접근성 및 윤리적 AI 배치에 대한 기여를 강조합니다.



### MAJORScore: A Novel Metric for Evaluating Multimodal Relevance via Joint Representation (https://arxiv.org/abs/2509.21365)
- **What's New**: 이 논문에서는 MAJORScore라는 새로운 멀티모달 관련성 평가 지표를 제안합니다. 이는 기존의 이계(二界, bimodal) 데이터에 대한 사전 훈련된 대조 학습 모델의 임베딩(embedding) 능력을 기반으로 합니다. MAJORScore는 N개의 모달리티(N>=3) 간의 상관관계를 평가하기 위한 최초의 멀티모달 공동 표현(joint representation) 지표입니다.

- **Technical Details**: MAJORScore는 다양한 모달리티를 동일 잠재 공간(latent space)으로 통합하여 서로 다른 모달리티를 정확히 표현할 수 있는 기능을 가지고 있습니다. 이를 통해 공정한 관련성 점수(scoring)를 제공하고 모달리티 간의 유사성을 평가하는 데 필수적입니다. 기존의 평가 지표는 두 개의 모달리티 간의 상관관계 분석에만 적합하여 멀티모달 유사성 평가에 한계가 있었습니다.

- **Performance Highlights**: 대규모 멀티모달 데이터셋에서 MAJORScore는 기존 방법에 비해 일관성 있는 모달리티에서 26.03%-64.29% 향상되었고, 비일관성 모달리티에서는 13.28%-20.54% 감소하는 결과를 보였습니다. 이는 MAJORScore가 멀티모달 모델 성능 평가에서 더욱 신뢰할 수 있는 지표임을 입증합니다.



### A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision--Revised (https://arxiv.org/abs/2509.21363)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 기존의 salient object detection 기술이 갖고 있는 경계 불명확성과 복잡한 객체 구조 문제를 해결하기 위해 다중 과제(supervision) 훈련 전략을 제안합니다. 이 방식은 salient object detection, foreground contour detection, edge detection의 세 가지 작업을 동시에 활용하여 saliency maps를 개선합니다. 또한, 새로운 Mutual Learning Module (MLM)을 도입하여 여러 네트워크 분기를 상호 학습 구조로 훈련시키며 성능 향상을 꾀합니다.

- **Technical Details**: 제안된 방법은 VGG-16 백본(neural network backbone)을 기반으로 하는 인코더-디코더 아키텍처를 따릅니다. 네트워크는 서로 얽혀 있는 다중 작업(supervision) 학습을 통해 salient object detection과 foreground contour detection을 동시에 수행하여 보다 정확한 예측을 가능하게 합니다. 특히, edge module(EM)이 함께 설계되어 saliency feature의 노이즈를 감소시키고, 동시에 foreground contour detection 결과의 정확성을 높입니다.

- **Performance Highlights**: 제안된 방법은 7개의 도전적인 데이터셋에서 최첨단 성능을 달성했으며, 이전의 다른 모델들과 비교했을 때 상대적으로 훨씬 빠른 속도로 유사한 edge detection 성능을 보여주었습니다. 광범위한 실험 결과, 이 알고리즘은 다른 경쟁 모델에 비해 유의미한 성과를 얻었습니다. 또한, 다중작업 상호 학습 구조 덕분에 더욱 정교한 모습의 salient 지역을 생성할 수 있음을 입증했습니다.



### Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs (https://arxiv.org/abs/2509.21361)
Comments:
          20 pages, 4 charts

- **What's New**: 이 논문에서는 대형 언어 모델(LLM) 제공업체들이 자랑하는 최대 컨텍스트 윈도우 크기와 실질적인 사용 사례 간의 차이를 탐구합니다. 연구진은 최대 유효 컨텍스트 윈도우(Maximum Effective Context Window, MECW)의 개념을 정의하고, 다양한 크기와 문제 유형에서 컨텍스트 윈도우의 효과를 테스트하는 방법을 공식화했습니다. 이러한 연구는 컨텍스트 윈도우 크기 증가에 따른 모델의 효능 비교를 위한 표준화된 방법을 마련했습니다.

- **Technical Details**: 논문은 여러 모델에서 수십만 개의 데이터 포인트를 수집하여 보고된 최대 컨텍스트 윈도우(Maximum Context Window, MCW) 크기와 MECW 크기 간의 유의미한 차이를 발견했습니다. 연구 결과 MECW는 MCW와 현저히 다를 뿐만 아니라 문제 유형에 따라 변화한다는 것을 보여줍니다. 일부 최상급 모델은 100 토큰의 컨텍스트로도 실패했으며, 대부분은 1000 토큰에서 정확도가 심각하게 저하되었습니다.

- **Performance Highlights**: 논문의 데이터 분석 결과, 모든 모델은 최대 컨텍스트 윈도우에서 최대 99%에 이르는 큰 차이를 보였습니다. MECW는 제공되는 문제 유형에 따라 변화하며, 이는 모델의 정확도를 향상시키고 환각(hallucination) 비율을 감소시키기 위한 실질적이고 명확한 인사이트를 제공합니다. 본 연구는 효과적인 모델 개발에 필요한 중요한 기준을 제시합니다.



### Multimodal Prompt Decoupling Attack on the Safety Filters in Text-to-Image Models (https://arxiv.org/abs/2509.21360)
- **What's New**: 최근 T2I (Text-to-Image) 모델들이 고화질 이미지 생성에 많이 활용되고 있지만, 이러한 모델이 악용되어 Not-Safe-for-Work (NSFW) 콘텐츠를 제작할 수 있다는 우려가 커지고 있습니다. 기존의 jailbreak 공격 방식은 주로 텍스트 프롬프트를 조작하는 데 초점을 맞추어 왔으나, 이미지 기반 입력에 대한 취약점은 충분히 다루어지지 않았습니다. 본 논문에서는 이를 해결하기 위해 MPDA (Multimodal Prompt Decoupling Attack)를 제안하며, 이 방법은 이미지 모달리티를 활용하여 원래의 안전하지 않은 프롬프트의 해로운 의미를 분리합니다.

- **Technical Details**: MPDA는 세 가지 주요 단계로 작동합니다. 첫 번째 단계에서 LLM (Large Language Model)은 안전하지 않은 프롬프트를 유사 안전 프롬프트와 해로운 프롬프트로 분리합니다. 두 번째 단계에서는 유사 안전 프롬프트로부터 기본 이미지를 생성하고, 해로운 프롬프트는 안전 필터를 우회하는 적대적 프롬프트로 변환되어 T2I 모델의 이미지 정제 과정을 안내합니다. 마지막으로, 생성된 이미지의 정확성을 확보하기 위해 비전-언어 모델(Vision-Language Model)이 생성된 이미지의 캡션을 분석하고, 이를 통해 LLM의 프롬프트 세분화 과정을 반복합니다.

- **Performance Highlights**: MPDA는 현재 상용화된 T2I 모델인 Wan-T2I, Cogview, MidJourney에서 평균 93%의 우회 성공률을 달성했습니다. 이는 기존의 텍스트 기반 공격 방법에 비해 훨씬 더 효과적인 결과로, T2I 모델의 다중 모달 안전 프로토콜에서의 취약점을 드러냅니다. 이 연구는 다중 모달 적대적 프롬프트 공격에 대한 방어 전략을 논의하며, 모델의 안전성 향상을 위한 참고자료를 제공할 수 있습니다.



### Influence Guided Context Selection for Effective Retrieval-Augmented Generation (https://arxiv.org/abs/2509.21359)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG)의 컨텍스트 품질 평가 방식을 새롭게 정의하여 Contextual Influence Value (CI value)를 도입하였습니다. CI value는 생성기 성능 저하를 추적하여 각 컨텍스트의 품질을 정량화함으로써, 질 낮은 컨텍스트를 효율적으로 필터링하는 방법을 제공합니다. 이 새로운 접근법은 기존의 다양한 품질 평가 기준의 한계를 극복하기 위해, 쿼리, 컨텍스트 목록, 생성기를 통합적으로 활용합니다.

- **Technical Details**: CI value는 쿼리-의존성(쿼리와의 연관성), 목록-의존성(컨텍스트 간의 관계), 생성기-의존성(생성기와의 일치도)에 따라 각 컨텍스트의 기여도를 측정합니다. 또한 이 값을 통해 복잡한 하이퍼파라미터 조정 없이 양의 CI 값을 가진 컨텍스트만을 인식하여 유지할 수 있는 간단한 선택 전략을 제공합니다. CI 값 계산의 어려움을 해결하기 위해, CI Surrogate Model (CSM)을 개발하여 레이블 의존성과 계산 오버헤드를 줄이며 생성기 피드백을 활용하는 계층적 구조를 사용합니다.

- **Performance Highlights**: 8개의 NLP 작업과 여러 LLM에서 진행된 실험 결과, 제안된 방법이 최신 기술들과 비교하여 상당한 성능 향상을 달성했습니다. 특히, CI value를 기반으로 한 컨텍스트 선택 기법은 평범한 기준들을 초과하여 저질 컨텍스트를 효과적으로 필터링하면서도 중요한 정보를 보존합니다. 제안된 CSM 접근법은 평균 15.03%의 성능 향상을 보였으며, 이는 RAG 생성 성능의 눈에 띄는 개선을 나타냅니다.



### MDF-MLLM: Deep Fusion Through Cross-Modal Feature Alignment for Contextually Aware Fundoscopic Image Classification (https://arxiv.org/abs/2509.21358)
Comments:
          Word count: 5157, Table count: 2, Figure count: 5

- **What's New**: 본 연구에서는 망막 계측 이미지에서 질병 분류 정확도를 향상시키기 위해 정밀한 이미지 특징과 전반적인 텍스트 맥락을 통합하는 새로운 다중 모달 딥 러닝 아키텍처인 MDF-MLLM을 제안합니다. 기존의 다중 모달 대규모 언어 모델(MLLM)은 녹내장, 당뇨병성 망막병증, 망막 색소변성증과 같은 망막 질환 진단에 필수적인 저수준 공간 세부정보를 캡처하는 데에 어려움을 겪고 있습니다. 이 연구는 3개의 공개 데이터 세트(FIVES, HRF, StoneRounds)로부터 수집한 1,305개의 이미지-텍스트 쌍을 기반으로 모델 개발 및 검증을 진행했습니다.

- **Technical Details**: MDF-MLLM은 LLaMA 3.2 11B MLLM 내의 비선형 주의 블록에 U-Net 인코더 레이어의 네 가지 스킵 특징을 통합했습니다. 이를 통해 이미지 특성을 패치 단위로 투사하고, 크로스 주의 및 FiLM 기반 U-Net 변조를 통해 융합했습니다. 이 모델은 이중 유형 질병 분류 작업에서 기본 MLLM의 60% 정확도에 비해 94%의 정확도로 56% 향상을 보여주었습니다.

- **Performance Highlights**: MDF-MLLM 모델은 기본 모델에 비해 Recall과 F1-score가 각각 67%와 35% 향상되었음을 보여주었습니다. 다중 깊이 융합 접근 방식이 상속된 질병을 포함하여 공간적 추론 및 분류에서 상당한 이점을 제공한다는 것을 검증하였습니다. 이 연구는 임상 의사 결정 지원 시스템에 실제 배치 가능성이 높은 해석 가능한 모듈형 프레임워크를 제시합니다.



### A Novel Differential Feature Learning for Effective Hallucination Detection and Classification (https://arxiv.org/abs/2509.21357)
Comments:
          10 pages, 7 figures, 13 tables

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)에서 발생하는 환각(hallucination) 신호의 정확한 위치를 탐색하기 위한 이중 모델 아키텍처를 제안합니다. 새로운 구조는 Projected Fusion (PF) 블록과 Differential Feature Learning (DFL) 메커니즘을 통합하여, 효율적인 검출 시스템의 개발을 위한 길잡이를 제공합니다. 연구의 결과, 환각 신호가 매우 희소한 특성 부분집합에 집중된다는 점을 발견하여 검출 효율성을 획기적으로 개선하는 방법론을 제시합니다.

- **Technical Details**: 제안된 이중 모델 아키텍처는 각기 다른 입력을 처리하는 두 개의 병렬 인코더 모델을 포함하며, 하나는 환각 신호에 주목하고 다른 하나는 사실 정보 인식을 목표로 합니다. Projected Fusion (PF) 블록은 서로 다른 은닉층에서의 정보를 효과적으로 통합하는 반면, Differential Feature Learning (DFL) 메커니즘은 두 모델의 특징 간의 절대 차이를 계산하여 판별적인 피쳐를 도출합니다. 이 연구는 특성 차이가 가장 큰 상위 1%의 피쳐만으로도 효과적인 검출이 가능함을 입증하며, 이를 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 체계적인 실험을 통해, HaluEval의 질문 응답(question answering), 대화(dialogue), 요약(summarization) 데이터 세트에 대해 환각 신호가 성능 향상에 중요한 역할을 함을 보여주었습니다. 연구 결과, 최적의 성능을 위해서는 전체 피쳐 차원의 1%만으로도 충분하다는 사실을 발견했습니다. 이러한 발견은 계산 효율적인 검출 시스템 구축의 가능성을 제시하며, 인퍼런스(inference) 비용을 줄이면서도 정확도를 유지할 수 있는 길을 열어줍니다.



### Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports (https://arxiv.org/abs/2509.21356)
Comments:
          In proceedings MICCAI 2025

- **What's New**: 이 논문에서는 자동으로 생성된 흉부 방사선 보고서에서 오류를 감지하기 위한 새로운 문구 기반 사실 확인 모델(FC 모델)을 제안합니다. 이 모델은 발견물 및 각 위치의 오류를 식별하는 데 중점을 두고 있습니다. 특히, 통합된 데이터를 통해 사실과 허위 발견물-위치 쌍을 구성하여 보고서 오류를 시뮬레이션합니다.

- **Technical Details**: FC 모델은 다중 라벨 교차 모달 회귀 네트워크를 기반으로 하며, 2,700만 개 이상의 이미지를 조합하여 훈련됩니다. 이 훈련 과정은 기본적으로 발견물 위치 격리, 합성 데이터 생성, FC 모델 훈련 등의 단계로 구성됩니다. 모델은 보고서에서 추출된 발견과 이미지를 기반으로 입력을 받아 오류를 예측합니다.

- **Performance Highlights**: FC 모델은 다양한 흉부 X-레이 데이터 세트에서 발견 정확성과 위치 예측의 robust성을 보여주며, 0.997의 일치 상관 계수를 달성하여 지상 진실 확인과 높은 신뢰성을 입증하였습니다. 이러한 결과는 이 모델이 방사선 업무에서 임상 추론 단계에서 유용할 가능성을 시사합니다.



### Domain-Informed Genetic Superposition Programming: A Case Study on SFRC Beams (https://arxiv.org/abs/2509.21355)
Comments:
          11 pages, 6 tables, 4 figures

- **What's New**: 이번 연구에서는 분리 가능한 물리 메커니즘에 의해 지배되는 공학 시스템을 위해 특수 설계된 기호 회귀(framework)인 DIGSP (domain-informed genetic superposition programming)를 제안합니다. DIGSP는 입력 공간을 도메인 기반의 특성 서브셋으로 나누고, 물질 특유의 효과를 모델링하기 위해 독립적인 유전 프로그래밍(populations)을 진화시킵니다. 이를 통해 DIGSP는 전통적인 회귀 모델과 비교했을 때 더 나은 수렴성과 일반화 능력을 보여줍니다.

- **Technical Details**: DIGSP는 입력 특성을 공학적 의미에 따라 사전 분할하고, 독립적으로 진화하는 아형(populations)으로 구성됩니다. 또한 AHSAM(adaptive hierarchical symbolic abstraction mechanism)을 통해 모든 집단에서 정체 현상이 발생할 때 통계적으로 유의미한 기호 표현을 식별하고, 이를 다음 세대에 주입하여 기호 탐색을 유도합니다. 이러한 구조적 개선은 모델의 해석 가능성을 높이면서도 수렴 효율성을 증대시킵니다.

- **Performance Highlights**: DIGSP는 철근 콘크리트(SFRC) 보에 대한 데이터를 사용해 기존의 multi-gene genetic programming (BGP) 모델과 비교 평가되었습니다. 30회의 독립적인 실험을 통해 DIGSP는 훈련 및 테스팅 루트 평균 제곱 오차(RMSE)에서 일관되게 BGP를 초과했으며, Wilcoxon 순위 합 검정에서는 통계적 유의성이 확인되었습니다(p < 0.01). 이 연구는 물리적 임팩트와 해석 가능성을 중시하는 기계 학습의 새로운 발전 방향을 제시합니다.



### KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cach (https://arxiv.org/abs/2509.21354)
- **What's New**: 이번 연구에서는 KV-Efficient VLA라는 메모리 압축 프레임워크를 제안하여 비전-언어-액션 (VLA) 모델의 효율성을 개선합니다. 이 접근법은 고유 사용 주제를 선택적으로 유지하면서 과거의 정보를 고정 크기 청크로 분할합니다. 결과적으로 이 모델은 빠른 추론 속도와 메모리 사용량을 줄일 수 있도록 설계되었습니다.

- **Technical Details**: KV-Efficient VLA는 키-값 (KV) 캐시를 고정 길이 청크로 나누고, 각 청크에 대해 순환 게이팅 모듈을 사용하여 과거의 문맥을 요약 및 필터링합니다. 이 시스템은 저차 근사 (LoRA) 기술을 이용해 사전 훈련된 LLaMA 모델과 통합되어 작동하며, 적은 비용으로 세밀한 조정을 수행합니다. 이러한 구조적 변화로 인해 실시간 로봇 제어에서 요구되는 저지연성을 유지하면서도 메모리 사용량을 대폭 줄일 수 있습니다.

- **Performance Highlights**: KV-Efficient VLA 모델은 평균적으로 1.21배 더 빠른 추론 속도를 달성하며, KV 메모리를 36% 줄이는 효과를 보여주었습니다. 이러한 성능 향상에도 불구하고, 작업 성공률에 미치는 영향은 최소화되어 있어서 다양한 환경에서 높은 정확도를 유지합니다. 솔루션은 기존의 VLA 시스템에 원활하게 통합될 수 있어, 실제 로봇 제어 논리에 어떠한 수정도 필요하지 않습니다.



### Random Direct Preference Optimization for Radiography Report Generation (https://arxiv.org/abs/2509.21351)
- **What's New**: 이번 연구는 Radiography Report Generation (RRG) 분야에서 기존 방법의 한계를 극복하기 위한 방법으로, Direct Preference Optimization (DPO) 프레임워크를 제안합니다. 이 방법은 임의의 대조적 샘플링을 활용하여 훈련 쌍을 구성하며, 추가적인 보상 모델이나 인간의 선호 주석이 필요하지 않습니다. 실험 결과, 세 가지 최첨단 모델에 우리의 Random DPO를 보완함으로써 최대 5%의 임상 성능 향상을 달성했습니다.

- **Technical Details**: RRG 모델은 전통적으로 다음 단어 예측을 위한 표준 크로스 엔트로피 손실 기준으로 최적화됩니다. 이러한 접근 방식은 실질적으로 모델이 잘못된 연관성을 학습하게 만들어 편향을 유발할 수 있습니다. 본 연구에서는 DPO를 기반으로 모델 성능을 향상시키기 위한 무작위 샘플링 방법을 제안하며, 이는 추가적인 데이터 준비 없이 기존 데이터셋을 활용하여 성과를 낼 수 있습니다.

- **Performance Highlights**: MIMIC-CXR, CheXpert Plus, IU X-ray, Interpret-CXR와 같은 네 개의 공개 데이터셋을 사용하여 실험을 진행했습니다. MIMIC-CXR는 377,110개의 이미지와 227,835개의 방사선 연구가 포함되어 있으며, CheXpert Plus는 223,228개의 독특한 방사선 보고서와 가슴 X선 쌍으로 구성되어 있습니다. 각 데이터셋은 다양한 임상 환경에서 수집되어 RRG 방법의 종합적인 평가를 허용합니다.



### SGNNBench: A Holistic Evaluation of Spiking Graph Neural Network on Large-scale Graph (https://arxiv.org/abs/2509.21342)
Comments:
          The code is available at this https URL

- **What's New**: 이번 연구에서는 Spiking Graph Neural Networks (SGNNs)의 발전을 위한 포괄적인 벤치마크인 SGNNBench를 소개합니다. SGNNBench는 9개의 최신 SGNN과 7개의 고전적인 GNN을 18개의 데이터셋에서 비교하여 효과성, 에너지 효율성 및 아키텍처 설계 등 다양한 관점에서 SGNNs를 깊이 있게 조사합니다. 본 논문은 기존의 SGNN 연구에서 간과된 에너지 병목 현상을 밝혀내는 데 중점을 두고 있습니다.

- **Technical Details**: 그래프 데이터는 노드 𝒱(𝓥)와 간선 ℰ(𝓔)로 표현되며, 각 노드는 메시지 패싱 메커니즘을 통해 이웃 노드로부터 정보를 축적하고 자신의 표현을 업데이트합니다. SGNN에서는 스파이크 기반 뉴런을 통해 통신하며, 메시지 패싱 네트워크를 단순화하여 에너지를 절약하고 계산을 용이하게 합니다. 최근 연구들은 그래프 모델을 보다 효율적으로 확장하기 위해 스파이크 기반 노드 토크나이저와 선형 시간 주의 변형을 설계하고 있습니다.

- **Performance Highlights**: SGNNBench의 분석 결과, 기존 SGNN의 설계에서 고전적인 GNN 아키텍처 구성 요소의 효과가 간과되었음을 보여 주었습니다. 이 연구는 SGNN의 성능 차이를 줄이기 위한 중요한 통찰력을 제공하며, SGNN의 완전 정밀 GNN과의 비교를 통해 에너지 소비 측면에서 효율성을 높일 기회를 제시합니다. 또한, SGNN의 발전을 위한 데이터 기반 접근 방식을 통해 향후 모델 설계에 대한 더 깊은 이해를 제공합니다.



### From Embeddings to Equations: Genetic-Programming Surrogates for Interpretable Transformer Classification (https://arxiv.org/abs/2509.21341)
Comments:
          20 pages, 8 tables, 7 figures

- **What's New**: 이번 연구는 동결된 Transformer 임베딩(frozen Transformer embeddings)에 대한 상징적 대리모델링(symbolic surrogate modeling)을 통해 컴팩트(compact)하고 감사 가능한(auditable) 분류기를 개발하는 것을 목표로 합니다. 이 연구는 여러 벤치마크(benchmark)에서 성능을 입증하며, 특히 주목할 만한 점은 성능과 해석 가능성을 모두 갖춘 대리 모델을 제공한다는 점입니다. 새로운 기법인 의미 보존 특징 분할(Semantic-Preserving Feature Partitioning, SPFP)을 활용하여 정보 손실 없이 정보를 유지하는 뷰(view)를 생성합니다.

- **Technical Details**: 연구에서는 현대 BERT, DINOv2, SigLIP에서 추출한 고정 임베딩(frozen embeddings)을 사용하여 훈련 데이터에서만 의미 보존을 통해 코디네이트를 파티셔닝합니다. 이후 협력적 다인구 유전 프로그래밍(multi-population genetic programming, MEGP)을 통해 각 뷰에 대한 상징적 프로그램으로 클래스 로짓(class logits)을 출력합니다. 이를 통해 예측 성능을 유지하면서도 해석 가능하고 간결한 대리 모델을 생성하며, 검증(validation) F1 점수에 대한 표준 편차 규칙을 적용하여 모형 선택을 수행하는 메커니즘을 제공합니다.

- **Performance Highlights**: 모델은 MNIST, CIFAR10, SST2G와 같은 데이터셋에서 강력한 분리 성능을 보여주며, F1 점수가 MNIST, CIFAR10, MSC17에서 약 0.99에 이릅니다. 특히 SST2G는 상대적으로 도전적 과제로 남아 있습니다. 연구 결과, 신뢰도 다이어그램(reliability diagrams) 및 기여 기반 중요도를 포함한 포괄적인 분석을 제공하여 대리 모델이 신뢰할 수 있는 결과를 도출함을 입증합니다.



### Cycle is All You Need: More Is Differen (https://arxiv.org/abs/2509.21340)
- **What's New**: 이번 논문에서는 메모리와 의식의 근본 메커니즘으로 사이클 클로저(cycle closure)를 제안하는 정보-위상(topological) 프레임워크를 소개합니다. 메모리는 정적 저장소가 아니라 신경 상태 공간의 잠재 사이클(re-enter latent cycles)로 다시 들어가는 능력으로 정의됩니다. 저자들은 이 관점을 통해 정보의 흐름에서 사이클의 유니버설한 역할을 살펴보며, 이를 통하여 인지, 기억, 추상의 조직을 설명합니다.

- **Technical Details**: 제안된 프레임워크는 생물학적으로 다중 시차(polychronous) 신경 그룹이 지연 잠금 발사를 통해 1-사이클을 실현한다는 점을 강조합니다. 이 과정은 세타-감마 리듬(theta-gamma rhythms) 내에서 경계 취소(boundary cancellation)를 강화하며, 마이크로 사이클은 계층적으로 구성되어 탐색 루프를 일반 기억과 인지로 확장합니다. 이러한 정보 과정은 연결 끊기 없이 마이크로 이벤트의 순서를 마음대로 바꿀 수 있는 능력을 요구합니다.

- **Performance Highlights**: 사이클 기반 아키텍처는 전통적인 기계가 외부에서 의미를 할당하는 방식과는 달리, 신경 상호작용의 역동성에서 의미를 내재적으로 이끌어냅니다. 이러한 접근은 AI의 비-튜링(non-Turing) 경로를 열어주며, 고차원 매개변수 공간에 대해 기울기 하강을 요구하지 않고도 오류 수정(error correction), 일반화(generalization), 에너지 효율을 달성할 수 있는 자연스러운 메커니즘을 제공합니다. 결국 사이클은 메모리가 지속되는 것을 가능하게 하며, 이는 구조적 불변성(structural invariants)에서 비롯됩니다.



### Cross-Modal Retrieval with Cauchy-Schwarz Divergenc (https://arxiv.org/abs/2509.21339)
Comments:
          Accepted by ACMMM-25

- **What's New**: 이 논문에서는 다중 모달 학습의 핵심 과제인 교차 모달 검색(Cross-Modal Retrieval, CMR)에 Cauchy-Schwarz (CS) divergence를 도입합니다. CS divergence는 초매개변수( hyperparameter)가 필요 없고, 수치적으로 안정적이며 선형적으로 확장 가능하여 기존 방법보다 뛰어난 성능을 보여줍니다. 또한, 새로운 Generalized CS (GCS) divergence를 제안하여 세 개 이상의 모달을 통합하여 비교할 수 있는 수학적 프레임워크를 제공합니다.

- **Technical Details**: CS divergence는 두 개 이상의 모달 간의 직접적인 정렬을 가능하게 하며, 고전적인 방법인 Kullback-Leibler divergence, Maximum Mean Discrepancy, Correlation Alignment의 한계를 극복합니다. 기존 방법들이 일반적으로 두 모달에 제한되는 것에 반해, CS divergence는 다수의 모달을 동시에 비교할 수 있는 회전 대칭 비교( bidirectional circular comparison)가 포함되어 있습니다. 이로 인해 계산 복잡성이 크게 줄어듭니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋을 활용한 포괄적인 실험 결과, CS divergence 기반 방법이 이중 모달 및 삼중 모달 검색 임무에서 모든 기존 접근 방식을 초월하는 성능을 보였습니다. 특히 전통적인 방법에서 요구되는 쌍 비교(pairwise comparisons) 없이도 우수한 검색 정확도를 달성하였습니다. 연구 코드 또한 공개되어 있어 연구자들이 쉽게 활용할 수 있도록 하고 있습니다.



### Seismic Velocity Inversion from Multi-Source Shot Gathers Using Deep Segmentation Networks: Benchmarking U-Net Variants and SeismoLabV3+ (https://arxiv.org/abs/2509.21331)
- **What's New**: 이번 연구는 지진 속도 역산(Seismic velocity inversion) 분야에서 U-Net, U-Net++, 그리고 DeepLabV3+의 최적화 버전인 SeismoLabV3+를 활용한 새로운 접근 방식을 제시하고 있습니다. 연구는 ThinkOnward 2025 Speed & Structure 데이터셋을 사용하여, 다양한 딥러닝 아키텍처 간의 성능을 비교합니다. SeismoLabV3+는 최고 성능을 기록하며, 지진 속도 역산을 위한 딥 분할 네트워크의 가능성을 보여주고 있습니다.

- **Technical Details**: 지진 속도 역산은 지진파 데이터에서 지하 구조를 재구성하는 과정으로, 이는 고해상도 지진 이미징과 해석을 위해 필수적입니다. 기존의 물리 기반 방법들은 연산 복잡성과 초기 속도 모델에 민감하며, Bandwidth(대역폭)의 한계로 인해 제약을 받습니다. 반면, 딥러닝 기반의 데이터 주도 접근법은 지진 샷 수집(data)에서 지하 속도 모델로의 직접적인 매핑(mapping)을 통해 이를 해결하려는 노력을 집중하고 있습니다.

- **Performance Highlights**: SeismoLabV3+는 내부 유효성 검증 분할에서 MAPE 값이 0.03025, 숨겨진 테스트 세트에서 0.031246로 측정되어 최상의 성능을 나타냈습니다. 연구 결과는 맞춤형 건축물 개선이 지구 물리학 AI 모델 발전을 위해 얼마나 중요한지를 강조합니다. 이러한 성과는 딥 세분화 네트워크가 지진 속도 역산에 적합하며, 다중 스케일의 지질 구조와 날카로운 속도 경계 (sharp velocity boundaries)를 효과적으로 캡쳐하는 능력을 보여줍니다.



### Assessment of deep learning models integrated with weather and environmental variables for wildfire spread prediction and a case study of the 2023 Maui fires (https://arxiv.org/abs/2509.21327)
- **What's New**: 이번 연구에서는 야생화재 확산 예측에서 인공지능(AI)의 역할을 다루었습니다. 특히 하와이에서 10년 이상의 데이터를 활용하여 날씨와 환경 변수를 통합한 다섯 가지 심층 학습(deep learning) 모델의 성능을 평가했습니다. 또한 2023년 마우이 화재를 사례로 들어 심층 학습 모델과 전통적인 화재 확산 모델인 FARSITE를 비교하였습니다.

- **Technical Details**: 연구에서 적용된 모델은 ConvLSTM, ConvLSTM with attention 등으로, 이들 모델은 야생화재 예측을 위해 날씨 데이터와 환경 변수를 학습합니다. 각 모델의 성능을 평가하기 위해 F1-score, precision, recall과 같은 다양한 지표를 사용했습니다. 연구 결과, ConvLSTM과 ConvLSTM with attention이 다섯 가지 AI 모델 중에서 가장 우수한 성능을 보였습니다.

- **Performance Highlights**: FARSITE는 최고의 AI 모델보다 높은 precision과 F1-score를 기록했지만, recall은 더 낮았습니다. 반면 AI 모델들은 입력 데이터에 대한 높은 유연성을 제공했습니다. 또한, 설명 가능한 AI(explainable AI) 기법을 통해 2023년 마우이 화재와 관련된 중요한 날씨 및 환경 요인을 식별할 수 있었습니다.



### PIR-RAG: A System for Private Information Retrieval in Retrieval-Augmented Generation (https://arxiv.org/abs/2509.21325)
- **What's New**: 본 논문에서는 개인 정보 보호를 고려한 Retrieval-Augmented Generation (RAG) 시스템인 PIR-RAG를 소개합니다. PIR-RAG는 대규모 AI 시스템에서 개인 정보를 안전하게 보호하기 위해 설계된 새로운 아키텍처로, 전체 문서 클러스터를 효율적으로 검색할 수 있는 기능을 제공합니다. 본 연구는 기존의 RAG 아키텍처에서의 개인 정보 노출 문제를 해결하기 위한 실질적인 솔루션을 제시합니다.

- **Technical Details**: PIR-RAG는 코스 그레인(Coarse-grained) 의미 클러스터링을 활용하여 검색 공간을 축소하고, 빠른 격자 기반( lattice-based) Private Information Retrieval (PIR) 프로토콜을 결합합니다. 이 데이터 구조를 통해 전체 내용 검색을 단일 고성능의 행렬-벡터 곱(matrix-vector product)으로 변환함으로써, 개인 정보 검색을 위한 비용을 효율적으로 관리합니다. 이를 통해 데이터 검색 및 취득 단계의 비용을 통합하여 운영 효율을 극대화합니다.

- **Performance Highlights**: PIR-RAG는 기존 아키텍처들과 비교하여 우수한 성능을 보입니다. 특히, "RAG-Ready Latency" 즉, LLM의 콘텐츠 보안을 유지하면서 데이터를 안전하게 가져오기 위해 필요한 종단 간의 시간을 명확히 보여줍니다. 종합적인 비교 평가 결과, PIR-RAG는 확장성과 성능, 검색 품질에서 매우 효율적인 솔루션으로 입증되었습니다.



### From Search to Reasoning: A Five-Level RAG Capability Framework for Enterprise Data (https://arxiv.org/abs/2509.21324)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템의 새로운 분류 프레임워크(L1-L5)를 제안하여 다양한 데이터 형식과 질문 복잡성에 따라 질문-응답 시스템을 카테고리화합니다. L1은 비구조적 데이터의 표면적 지식을 사용하며, L5는 일반 지능으로 자리잡으려는 aspirational한 수준을 다룹니다. 이를 통해 엔터프라이즈 사용자들이 기대하는 질문-응답 문제를 보다 효과적으로 해결하고자 합니다.

- **Technical Details**: 논문은 RAG 시스템이 대량의 비즈니스 데이터를 처리하는 데 있어서 발생하는 할루시네이션(hallucination) 문제를 해결하기 위한 여러 방법을 설명하고 있습니다. 특히, 이 연구는 RAG 시스템의 적절한 성능을 확보하기 위해 다양한 차원의 데이터 관련성을 고려한 개선된 검색 메커니즘의 필요성을 강조합니다. 각 레벨(L1부터 L4까지)의 시스템은 다루는 데이터 유형과 질문의 복잡성에 따라 다른 기능을 제공합니다.

- **Performance Highlights**: 제안한 프레임워크는 LangChain, Azure AI Search, OpenAI, Corvic AI와 같은 최신 플랫폼에 대한 평가에 기반하여 L1-L4 기능을 활성화하는 데 있어 다중 공간 검색 및 동적 오케스트레이션의 중요성을 강조합니다. 연구 결과는 복잡한 데이터 환경에서도 RAG 시스템이 갖는 유용성과 비즈니스 결정에 미치는 영향을 검증하는데 도움이 됩니다. 전반적으로 이 논문은 LLMs와 RAG의 잠재력을 최대한 활용하기 위한 혁신적인 전략에 대한 통찰력을 제공합니다.




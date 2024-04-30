### A Comprehensive Evaluation on Event Reasoning of Large Language Models (https://arxiv.org/abs/2404.17513)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 사건 추론 능력을 평가합니다. 특히, 새로운 벤치마크 EV2를 도입하여 사건 스키마 지식과 인스턴스 수준에서의 사건 추론 능력을 모두 평가합니다. EV2는 다양한 관계 유형과 추론 패러다임을 포괄하며, 문맥적 사건 분류(Contextual Event Classification, CEC) 및 문맥적 관계 추론(Contextual Relation Reasoning, CRR) 두 가지 사건 추론 작업을 포함합니다.

- **Technical Details**: EV2 벤치마크는 GPT 생성 및 인간 주석을 통해 구성되며, 사건 스키마 지식의 평가와 사건 추론 능력의 평가를 모두 포함합니다. 이 연구는 LLMs가 다양한 관계 유형과 추론 패러다임에 대한 사건 추론을 어떻게 수행하는지에 대한 체계적인 평가를 제공하려고 시도합니다. 그 과정에서, 사건 스키마 지식을 활용하는 방법에 대한 두 가지 지도 방법(Mentoring Methods)을 도입하여 LLMs의 사건 추론 성능을 향상시킵니다.

- **Performance Highlights**: 연구 결과에 따르면, LLMs는 사건 추론 능력을 보유하고 있지만 만족할 만한 수준은 아니며 여러 관계 및 추론 패러다임에서 능력의 불균형이 있음이 드러났습니다. LLMs는 사건 스키마 지식을 가지고 있으나, 사람들이 그 지식을 활용하는 방식과 일치하지 않습니다. 제안된 지도 방법을 통해 사건 스키마 지식을 명시적으로 안내할 때 LLMs의 사건 추론 성능이 개선되었으며 특히 직접적인 지도 방법이 큰 효과를 보였습니다.



### ReproHum #0087-01: Human Evaluation Reproduction Report for Generating  Fact Checking Explanations (https://arxiv.org/abs/2404.17481)
Comments: Accepted to HumEval at LREC-Coling 2024

- **What's New**: 이 논문은 Anatanasova 등(2020)의 'Generating Fact Checking Explanations' 연구를 부분 재현한 내용을 담고 있습니다. ReproHum이라는 공유 작업의 일환으로 수행된 이 연구는 NLP(Natural Language Processing) 분야의 재현 가능성이 시간이 지남에 따라 어떻게 변화하는지를 조사하는 것을 목표로 하고 있습니다. 본 연구는 인간 평가(human evaluation)의 역할과 품질 측정 방법의 재현성 문제에 초점을 맞춥니다.

- **Technical Details**: 이 연구는 종합적인 인간 평가를 통해 40개의 입력에 대한 3개의 팩트체킹 설명(fact-checking explanations)에 대한 상대적 순위를 수집하여 원본 논문의 결과를 재현했습니다. 사용된 기준은 'Coverage' (커버리지)로, 이는 팩트 체크와 관련된 중요 정보를 포함하고 있음을 의미합니다. 또한, 본 재현 연구는 DistilBERT 모델을 기반으로 멀티태스크 학습(multi-task learning) 프레임워크를 사용하여 사실 확인 텍스트의 중요 부분을 파악합니다.

- **Performance Highlights**: 재현 결과는 원본 연구의 주요 발견들을 지지합니다. 원래 연구와 우리의 재현 간에 유사한 패턴이 관찰되었으며, 약간의 차이는 있지만 이는 모델의 효율성에 관한 원 저자들의 주요 결론을 지지합니다. 멀티태스크 학습 접근 방식은 사실 검증 및 설명 생성에서 향상된 성능을 보여줍니다.



### CEval: A Benchmark for Evaluating Counterfactual Text Generation (https://arxiv.org/abs/2404.17475)
- **What's New**: 이 연구는 반사실적 텍스트 생성을 위한 새로운 벤치마크인 CEval을 소개합니다. CEval은 반사실적 생성 방법론을 평가하고, 텍스트 품질 지표를 결합하여 가장 일반적인 반사실적 데이터셋들과 표준 기준 모델들(MICE, GDBA, CREST)과 오픈 소스 언어 모델 LLAMA-2를 포함합니다. 또한, 연구자들은 이 벤치마크를 오픈소스 파이썬 라이브러리로 제공하여 커뮤니티가 더 많은 방법을 기여하고 일관된 평가를 유지할 수 있도록 독려합니다.

- **Technical Details**: CEval 벤치마크는 다양한 반사실적(Counterfactual) 텍스트에 대해 '라벨 뒤집기 능력(label flipping ability)'과 '텍스트 품질(text quality)'(예: 유창성, 문법, 일관성)을 평가합니다. 이는 인간 주석이 포함된 주의 깊게 선별된 데이터셋 및 큰 언어 모델을 사용하는 간단한 기준선을 포함합니다. 또한, 이 벤치마크는 반사실적 텍스트를 생성하는 다양한 방법을 체계적으로 검토하고 비교합니다. 방법 구분은 Masking and Filling Methods (MF), Conditional Distribution Methods (CD), 그리고 Large Language Models (LLMs)를 활용한 반사실적 생성으로 나뉩니다.

- **Performance Highlights**: 실험 결과 어떠한 방법도 완벽한 반사실적 텍스트를 생성하지 못했습니다. 반사실적 지표에서 우수한 성능을 보이는 방법들은 종종 텍스트 품질이 낮았고, 간단한 프롬프트를 사용하는 LLMs는 높은 품질의 텍스트를 생성하지만 반사실적 기준을 충족시키는 데 어려움을 겪었습니다. 이를 통해 추후 연구를 위해 여러 방법의 조합을 탐구하는 것이 유망한 방향이 될 수 있음을 제안합니다.



### Ruffle&Riley: Insights from Designing and Evaluating a Large Language  Model-Based Conversational Tutoring System (https://arxiv.org/abs/2404.17460)
Comments: arXiv admin note: substantial text overlap with arXiv:2310.01420

- **What's New**: 이 연구에서는 자연어 기반의 인터액션을 통해 학습 경험을 제공하는 대화형 교수 시스템(Conversational Tutoring Systems, CTSs)의 새로운 유형을 다루고 평가합니다. 특히 이 시스템은 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전을 활용하여 AI 지원 콘텐츠 제작 및 스크립트 조정을 자동화합니다. ‘Ruffle&Riley’라는 두 LLM 기반 에이전트는 각각 학생과 교수 역할을 하며, 자유 형식의 대화를 통해 교육을 진행합니다.

- **Technical Details**: 이 시스템은 수업 텍스트로부터 자동으로 편집 가능한 교수 스크립트를 생성하여 AI가 교수 콘텐츠 제작을 돕습니다. 또한, 두 LLM 기반 에이전트, 'Ruffle'과 'Riley'가 학습을 통한 교수(Learning-by-Teaching) 형식의 스크립트를 자동으로 조정합니다. 이 구조는 대화형 지능형 시스템(ITS) 특유의 내부 및 외부 루프 구조를 따릅니다.

- **Performance Highlights**: 200명의 온라인 사용자 스터디를 통해 단순 QA 챗봇 및 독서 활동과 비교한 결과, ‘Ruffle&Riley’ 사용자는 높은 수준의 참여도, 이해도를 보고하며 제공된 지원을 유용하게 평가했습니다. 활동 완료에 더 많은 시간이 필요했음에도 불구하고, 독서 활동 대비 단기 학습 성과에서는 유의미한 차이를 발견하지 못했습니다. 연구 결과는 향후 CTS 설계자에게 다양한 통찰력을 제공하며, 연구를 지원하기 위해 시스템을 오픈 소스로 제공합니다.



### Evaluation of Geographical Distortions in Language Models: A Crucial  Step Towards Equitable Representations (https://arxiv.org/abs/2404.17401)
- **What's New**: '자연 언어 처리(Natural Language Processing, NLP)에서 언어 모델은 글쓰기, 코드 작성, 학습과 같은 많은 전문적인 작업의 효율성을 향상시키기 위해 필수적인 도구로 자리 잡았습니다. 그러나 이 연구는 언어 모델이 지리적 지식과 관련하여 고유한 편향성(Bias)을 가지고 있다는 점에 주목하며, 이러한 편향성이 지리적 거리의 왜곡을 초래하여 불공정한 대표성을 낳을 수 있음을 제시합니다.

- **Technical Details**: 이 연구는 지리적 지식에 관련된 편향성을 조사하기 위해 네 가지 지표를 도입하여 지리적 거리와 의미론적(Semantic) 거리를 비교합니다. 연구자들은 이 지표들을 활용하여 10개의 널리 사용되는 언어 모델을 실험하였습니다. 이러한 지표는 언어 모델이 공간 정보를 어떻게 왜곡하는지 평가하는데 중요한 도구입니다.

- **Performance Highlights**: 실험 결과는 언어 모델에서의 공간적(Spatial) 편향을 점검하고 수정하는 것이 어떻게 중요한 지를 강조하고 있으며, 이는 정확하고 공정한 대표성을 보장하기 위한 핵심적인 단계임을 보여줍니다. 연구진은 언어 모델들이 지리적 거리를 어떻게 인식하고 표현하는지에 대한 중대한 왜곡들을 강조하고 이를 수정할 필요성을 지적하고 있습니다.



### Child Speech Recognition in Human-Robot Interaction: Problem Solved? (https://arxiv.org/abs/2404.17394)
Comments: Presented at 2024 International Symposium on Technological Advances in Human-Robot Interaction

- **What's New**: 최근 데이터 기반 음성 인식 기술의 발전, 특히 Transformer 구조와 방대한 훈련 데이터의 활용으로, 어린이 음성 인식(child speech recognition) 분야에서 눈에 띄는 발전이 있었습니다. 이번 연구에서는 2017년의 연구 결과를 재검토하여 OpenAI의 Whisper가 주목할 만한 성능 향상을 보였다는 것을 확인했습니다. 이는 어린이-로봇 상호작용(child-robot interaction)에서의 가능성을 보여줍니다.

- **Technical Details**: 여러 ASR 엔진 비교를 통해 OpenAI의 Whisper 모델은 Transformer(encoder-decoder Transformer architecture)를 기반으로 하며, 680,000시간에 달하는 레이블이 지정된 오디오 데이터에서 훈련되었습니다. 이는 그 이전의 모델보다 더 정교한 음성 인식 능력을 가능하게 했습니다. 테스트에는 Microsoft Azure Speech to Text, Google Cloud Speech-to-Text과 비교되었습니다.

- **Performance Highlights**: Whisper 모델은 경쟁 상업 클라우드 서비스보다 좋은 성능을 보여주었으며, 최고 모델은 작은 문법적 차이를 제외하고 60.3%의 문장을 정확하게 인식하였습니다. 또한, 로컬 GPU에서 실행될 때 초 단위의 빠른 대사 전환 시간(sub-second transcription time)을 제공하며, 이는 실시간 어린이-로봇 상호작용(real-time child-robot interactions)에 활용될 수 있는 잠재력을 보여줍니다.



### A Bionic Natural Language Parser Equivalent to a Pushdown Automaton (https://arxiv.org/abs/2404.17343)
Comments: to be published in IJCNN 2024

- **What's New**: 새로운 바이오닉 자연어 파서(BNLP)가 제안되었습니다. 이 파서는 기존의 어셈블리 계산(Assembly Calculus, AC) 모델을 기반으로 하면서 재귀 회로(Recurrent Circuit)와 스택 회로(Stack Circuit)라는 두 가지 새로운 생물학적으로 타당한 구조를 통합하여 설계되었습니다. 이러한 구조들은 각각 RNN과 단기 기억 메커니즘에서 영감을 받아 개발되었습니다. BNLP는 모든 정규 언어(Regular Languages)와 다이크 언어(Dyck Languages)를 완전히 처리할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: BNLP는 AC 기반으로 구축된 자연어 파서로서, 기존의 파서가 처리할 수 없었던 클리니 폐쇄(Kleene closures)를 효과적으로 다룰 수 있습니다. 이를 통해 BNLP는 모든 정규 언어와 다이크 언어를 처리할 수 있으며, 최종적으로 모든 문맥 자유 언어(Context-Free Languages)를 파싱할 수 있는 기능을 갖추게 됩니다. 이는 촘스키-슈첸버그 이론(Chomsky-Schützenberger theorem)을 활용하여 달성되었습니다. 또한 이 연구에서는 어떤 PDA(푸시다운 오토마타, Push-Down Automata)에 대해서도 BNLP에 해당하는 파서 오토마타(Parser Automaton)를 항상 형성할 수 있다는 것을 공식적으로 증명함으로써, BNLP가 PDA와 동등한 설명 능력을 가지며 기존 파서의 결점을 해결할 수 있음을 보여줍니다.

- **Performance Highlights**: BNLP는 독특하고 혁신적인 통합 능력으로 인해 모든 문맥-자유 언어를 파싱할 수 있습니다. 이와 같은 확장된 기능성은 BNLP를 과학적 및 기술적 연구 분야에서 중요한 도구로 만드는 요소입니다. 또한 BNLP는 기존의 유한 오토마타(Finite Automata, FA)가 갖지 못한 처리 능력을 보여주며 표현력에서 PDA와 동등함을 공식적으로 입증하였습니다.



### Can a Multichoice Dataset be Repurposed for Extractive Question  Answering? (https://arxiv.org/abs/2404.17342)
Comments: Paper 8 pages, Appendix 12 pages. Submitted to ARR

- **What's New**: 이 연구는 기존 데이터셋을 새로운 자연어 처리(NLP) 작업에 재활용하는 가능성을 탐구합니다. 특히, Belebele 데이터셋을 기반으로 하여 다중 선택형 질문 응답(MCQA) 문제에서 추출형 질문 응답(EQA)을 위한 기계 독해 형식으로 변경했습니다. 이는 주로 자원이 부족한 언어에 대한 연구를 활성화하는데 목표를 두고 있습니다.

- **Technical Details**: 연구팀은 Belebele 데이터셋을 사용하여 영어와 현대 표준 아랍어(Modern Standard Arabic, MSA)를 포함한 EQA 데이터셋을 생성했습니다. 데이터셋 변환 과정에서, 연구자들은 주어진 태스크에 맞게 데이터 어노테이션 가이드라인(annotation guidelines)을 개발하고 적용했습니다. 연구자들은 또한 여러 단일 언어 및 교차 언어 QA 모델을 테스트하여 그 성능을 평가하였습니다.

- **Performance Highlights**: 공개된 결과에 따르면, 본 연구에서 개발된 접근 방식은 영어, MSA 및 다섯 가지 아랍어 방언에 대해 효과적으로 적용되었습니다. 이는 특히 자원이 제한적인 언어에 대해서도 효율적인 NLP 솔루션을 제시할 수 있음을 시사합니다. 연구 팀은 이러한 방식이 Belebele 데이터셋에 포함된 120개 이상의 다른 언어 변형에 적용될 수 있도록 하기를 기대하고 있습니다.



### Metronome: tracing variation in poetic meters via local sequence  alignmen (https://arxiv.org/abs/2404.17337)
- **What's New**: 본 논문은 시의 구조적 유사성을 탐지하는 비지도학습(unsupervised) 방법을 도입하였습니다. 포엣의 텍스트를 연구하기 위해 '지역 시퀀스 정렬(local sequence alignment)' 기법을 사용하여, 다양한 언어와 시대를 아우르는 시의 구조적 관계를 연구합니다. 이 방법은 시적 텍스트를 음운학적 특징(prosodic features)의 문자열로 인코딩하고, 이를 정렬하여 거리 측정치를 도출합니다. 이러한 접근 방식은 크로스링귈(cross-lingual) 및 역사적 연구에 매우 유용하며, 새로운 시에 대한 이해와 분석을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 시퀀스를 상징적인 네 글자 알파벳으로 인코딩하고, 스미스-워터만(Smith-Waterman) 알고리즘을 사용하여 지역 시퀀스 정렬을 진행합니다. 이 방법은 상징과 페널티(symbol mismatches and penalties)에 무게를 두어 계산되며, 이를 통해 시의 구조적 유사성을 파악할 수 있습니다. 또한, 이 연구는 멧론(Metronome)이라는 파이썬(Python) 패키지로 구현되어, 광범위한 언어와 문맥에서 시의 연구를 지원하도록 공개되었습니다.

- **Performance Highlights**: 본 방법론은 체코어, 독일어, 러시아어 및 고전 라틴어를 포함한 다양한 언어로 된 시에 대한 클러스터링 성능을 평가하여, 여러 기존 방법론과 비교했습니다. 결과적으로, 이 방법은 시의 미터 인식(meter recognition)에 있어서 높은 성능을 보여주었으며, 시의 구조적 유사성을 인식하는 데 탁월한 능력을 입증했습니다. 또한, 역사적 연구를 위한 세 가지 사례 연구를 통해 이 방법의 유용성을 보여주었으며, 이는 향후 시 연구에 큰 영향을 끼칠 잠재력을 가지고 있습니다.



### Introducing cosmosGPT: Monolingual Training for Turkish Language Models (https://arxiv.org/abs/2404.17336)
- **What's New**: 이 연구에서는 터키어 단일 언어 코퍼스(only Turkish corpora)만을 사용하여 새롭게 구축한 cosmosGPT 모델을 소개합니다. 또한, 기본 언어 모델의 성능을 개선하기 위해 새로운 파인튜닝(finetune) 데이터셋과, 터키어 언어 모델의 능력을 측정하기 위한 새로운 평가 데이터셋을 소개하고 있습니다.

- **Technical Details**: 이 연구에서는 다언어 모델의 터키어 코퍼스로의 지속적인 트레이닝 대신, 터키어 전용 코퍼스로 모델을 트레이닝하는 대안적 방법을 사용하였습니다. 이를 통해 cosmosGPT라는 새로운 모델을 구축하였으며, 해당 모델을 위한 파인튜닝 및 평가 데이터셋들 또한 개발되었습니다.

- **Performance Highlights**: 비록 다른 모델들보다 크기가 약 10배 작음에도 불구하고, 단일 언어 코퍼스를 사용한 cosmosGPT 모델은 유망한 성능을 보여주고 있습니다. 이는 터키어 전용 언어 모델의 가능성을 보여주는 중요한 결과로, 언어별 특화 모델의 중요성을 강조합니다.



### When to Trust LLMs: Aligning Confidence with Response Quality (https://arxiv.org/abs/2404.17287)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)이 잘못되거나 말이 안 되는 텍스트를 생성할 수 있다는 문제점을 해결하기 위해 새로운 접근 방식인 CONQORD(CONfidence-Quality-ORDerpreserving alignment approach)를 제안합니다. CONQORD는 품질 보상(quality reward)과 순서 보존 정렬 보상(orderpreserving alignment reward)을 포함하는 맞춤형 이중 구성 요소 보상 함수를 사용하여 강화 학습(reinforcement learning)을 활용합니다.

- **Technical Details**: CONQORD는 응답의 질이 높은 경우 더 큰 확신을 표현하도록 모델을 동기 부여하여 확신과 품질의 순서를 일치시키는 순서 보존 보상(order-preserving reward)을 특히 강조합니다. 이 방식은 기존의 상위-k(top-k) 응답 생성 및 다중 응답 샘플링 및 집계 방법에 비해, 확신의 객관적인 가이드라인 부족 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, CONQORD는 확신 수준과 응답 정확도 사이의 정렬 성능을 크게 향상시켜, 모델이 지나치게 조심하는(over-cautious) 경향 없이 더 투명하고 신뢰할 수 있는 응답을 제공합니다. 또한, CONQORD에 의해 제공된 정렬된 확신은 LLMs를 신뢰할 때를 알려주고, 외부 지식의 검색 과정을 시작하는 결정 요인으로 작용합니다.



### Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact  Checking News Claims with Black-Box LLM (https://arxiv.org/abs/2404.17283)
Comments: Accepted by COLING 2024

- **원문 요약**: [{"What's New": '검색 기능을 향상된 언어 모델(Language Models)과 통합하여 사실 확인(Fact-checking) 작업의 성능을 향상시키기 위한 새로운 접근 방식, Fine-grained Feedback with Reinforcement Retrieval (FFRR)을 제안했습니다. 기존의 큰 언어 모델(Large Language Models, LLMs)과는 달리, FFRR은 더 세밀한 피드백을 통해 검색 정책을 최적화합니다.'}, {'Technical Details': 'FFRR은 두 단계 전략을 채택하여 검은 상자(black-box)식 LLM에서 세밀한 피드백을 수집하고, 그 결과를 보상(reward)으로 활용합니다. 이는 검색된 문서들을 비검색 기반 실제(task-specific non-retrieval ground truth) 성과와 비교 평가하여 이루어집니다.'}, {'Performance Highlights': '실제 뉴스 주장 검증을 위한 두 개의 공개 데이터셋에서 FFRR 모델을 평가한 결과, 강력한 LLM 기반 및 비LLM 기준(baselines) 모델들에 비해 유의미한 향상을 보였습니다.'}]



### Prompting Techniques for Reducing Social Bias in LLMs through System 1  and System 2 Cognitive Processes (https://arxiv.org/abs/2404.17218)
- **What's New**: 이 연구는 NLP 연구에서 인지 이중처리 이론(dual process theory)을 적용한 첫 사례 중 하나입니다. CoT (chain-of-thought) 유도를 사용하여 LLM (large language models)의 사회적 편견을 감소시키는 것이 가능하다는 이전 연구 결과를 확인하였습니다. 또한, 인간의 사고방식을 모델링하는 것과 LLM 시스템에 내재된 특성을 구별하기 위해 인간 및 기계 페르소나(personas)를 사용하였습니다.

- **Technical Details**: 연구팀은 제로샷(zero-shot), CoT, 그리고 여러 이중처리 이론 기반 유도 방법을 사용하여 아홉 가지 다양한 사회적 편견 분류에 걸쳐 두 개의 데이터세트에서 비교 분석하였습니다. 이러한 유도 방식이 LLM에서 어떻게 다르게 작동하는지, 그리고 이것이 인간 인지를 모델링하는 결과인지 또는 시스템 고유의 특성 때문인지를 탐구하였습니다.

- **Performance Highlights**: 연구 결과에 따르면 인간 페르소나와 시스템 2 (System 2), CoT 유도는 모두 LLM에서의 사회적 편견을 줄이는 경향이 있습니다. 그러나 이상적인 기능 조합은 사용된 모델과 편견 범주에 따라 다르며, 스테레오타입 판단에서 최대 13퍼센트의 감소를 보였습니다.



### Prompting Towards Alleviating Code-Switched Data Scarcity in  Under-Resourced Languages with GPT as a Pivo (https://arxiv.org/abs/2404.17216)
Comments: To be published in the Proceedings of SIGUL 2024: 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages

- **What's New**: 이 연구에서는 GPT 3.5를 사용하여 아프리카 언어 및 영어의 코드 혼합(code-switched) 문장을 생성하고자 하는 새로운 시도를 소개합니다. 특히 아프리칸스-영어(Afrikaans-English) 및 요루바-영어(Yoruba-English)의 코드 혼합 문장을 생성하면서, 주제-키워드 쌍, 언어학적 지침, 그리고 소수샷(few-shot) 예제를 활용하여 데이터의 다양성을 높였습니다.

- **Technical Details**: 연구는 GPT 3.5 모델을 활용하여 코드 혼합 데이터의 생성 가능성을 탐구한 것으로, 아프리칸스-영어와 요루바-영어 문장 생성에 초점을 맞추었습니다. 이를 위해 연구팀은 주제-키워드 쌍과 코드 혼합을 위한 언어학적 지침을 개발하여 모델 학습을 지원했으며, 소수샷 학습 접근 방식을 이용하여 표현력을 강화했습니다.

- **Performance Highlights**: 아프리칸스-영어(Afrikaans-English) 코드 혼합 문장은 높은 성공률을 보였지만, 비-라틴 글자(non-Latin script)를 사용하는 언어인 요루바-영어(Yoruba-English) 문장 생성의 품질은 상대적으로 낮았습니다. 이는 데이터의 다양성을 확장하고 저자원(low-resourced) 언어 데이터 문제를 해결하기 위한 기술의 발전과 네이티브 스피커(native speakers)의 중요성을 강조합니다.



### TIGQA:An Expert Annotated Question Answering Dataset in Tigrinya (https://arxiv.org/abs/2404.17194)
Comments: 9 pages,3 figures, 7 tables,2 listings

- **What's New**: 새로운 교육용 데이터셋인 TIGQA가 소개되었습니다. 이 데이터셋은 기존 데이터를 기계 번역(machine translation; MT)을 통해 티그리냐어로 변환하여 만들어졌으며, SQuAD 형식으로 구성된 2.68K의 질문-답변 쌍과 537개의 컨텍스트 문단을 포함하고 있습니다. 이 데이터셋은 기후, 물, 교통 등 122가지 다양한 주제를 다루고 있습니다.

- **Technical Details**: TIGQA 데이터셋은 단순한 단어 매칭을 넘어서는 능력을 필요로 하며, 단일 문장 및 다중 문장 추론 능력을 요구합니다. 이 연구에서는 최신 기계 독해(machine reading comprehension; MRC) 방법들을 사용하여 실험을 수행했으며, 이는 TIGQA 데이터셋에서 이와 같은 모델들을 사용한 첫 번째 탐색입니다. 또한, 연구팀은 사람의 성능을 추정하고 이를 사전 훈련된 모델들의 결과와 비교했습니다.

- **Performance Highlights**: 사람의 성능과 최고 모델 성능 사이에는 표시될 수 있는 차이가 있어, 데이터셋을 통한 추가 연구와 개선의 잠재력을 강조합니다. 연구팀은 연구 커뮤니티가 티그리냐어 MRC에서의 도전을 다루도록 장려하기 위해서 데이터셋을 자유롭게 접근할 수 있는 링크를 제공합니다.



### Prevalent Frequency of Emotional and Physical Symptoms in Social Anxiety  using Zero Shot Classification: An Observational Study (https://arxiv.org/abs/2404.17183)
- **What's New**: 이 연구는 사회 불안 장애의 다양한 신체 및 정서적 증상을 발견하고 이해하는 데 중점을 두고 있습니다. Reddit 데이터셋을 활용하여 다양한 인간 경험을 탐구하고, BART 기반의 multi-label zero-shot classification을 이용하여 증상의 빈도와 중요도를 분석합니다. 이 연구는 사회 불안이 개인에게 미치는 영향을 심층적으로 파악하고, 맞춤형 치료 개입을 제공하는 데 도움을 줄 수 있는 중요한 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 Reddit에서 수집한 사회 불안 관련 포스트를 분석하는데, 특히 r/socialanxiety 서브레딧에서 12,277개의 텍스트 문서를 사용하여 데이터셋을 구성했습니다. BART (Bidirectional and Auto-Regressive Transformers) NLP 모델을 활용하여 zero-shot 다중 라벨 분류를 수행하여 각 증상에 대한 발생 빈도와 중요도를 확률 점수(probability score) 형태로 측정하였습니다. 데이터는 2018년부터 2019년까지의 게시물을 포함하며, 사용자의 익명성을 엄격하게 유지하면서 연구가 진행되었습니다.

- **Performance Highlights**: 연구 결과는 '떨림(Trembling)'이 빈번한 신체적 증상으로 나타났으며, '부정적 판단에 대한 두려움(Fear of being judged negatively)'과 같은 정서적 증상도 높은 빈도로 관찰되었습니다. 이러한 발견은 사회 불안의 다면적 특성을 이해하는 데 크게 기여하며, 개인의 다양한 표현 방식에 맞춘 치료 및 지원 메커니즘을 설계하는 데 도움을 줄 수 있습니다.



### A Unified Label-Aware Contrastive Learning Framework for Few-Shot Named  Entity Recognition (https://arxiv.org/abs/2404.17178)
- **What's New**: 새로운 Few-shot Named Entity Recognition (NER) 모델이 제안되었습니다. 이 모델은 제한된 수의 레이블된 예제만을 사용하여 명명된 엔티티를 추출하는 것을 목표로 합니다. 기존의 대조 학습 방법들이 레이블 의미를 전적으로 의존하거나 완전히 무시하는 문제로 인해 컨텍스트 벡터 표현의 구별력이 부족하다는 문제점을 해결하기 위해, 통합된 레이블 인식 토큰 수준의 대조 학습(contrastive learning) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 모델은 레이블 의미를 접미사(prompts)로 활용하여 컨텍스트를 풍부하게 만듭니다. 또한, 컨텍스트-컨텍스트 간 대조 학습 목표와 컨텍스트-레이블 간 대조 학습 목표를 동시에 최적화하여 일반화된 차별적 컨텍스트 표현을 강화합니다. 이러한 기술적인 세부사항은 모델의 추론 능력과 맥락 인식을 크게 개선하는 데 도움이 됩니다.

- **Performance Highlights**: 다양한 전통적인 테스트 도메인(OntoNotes, CoNLL'03, WNUT'17, GUM, I2B2)과 대규모 Few-shot NER 데이터셋(FEWNERD)에서 실시된 방대한 실험을 통해 이 모델의 효과가 입증되었습니다. 이 모델은 이전 최고의 모델들을 큰 차이로 능가하여 대부분의 시나리오에서 평균 7%의 절대적인 마이크로 F1-스코어(micro F1 scores) 향상을 달성했습니다. 추가 분석을 통해 이 모델이 강력한 전달 능력과 개선된 맥락적 표현으로부터 혜택을 받고 있음을 밝혀냈습니다.



### Quantifying Memorization of Domain-Specific Pre-trained Language Models  using Japanese Newspaper and Paywalls (https://arxiv.org/abs/2404.17143)
Comments: TrustNLP: Fourth Workshop on Trustworthy Natural Language Processing (Non-Archival)

- **What's New**: 이 연구는 일본어 금융 신문 기사를 사용하여 도메인 특화 GPT-2 모델을 사전 학습하고, 이러한 모델이 어떻게 훈련 데이터를 기억하는지를 정량화하는 첫 번째 시도입니다. 특히, 기존 연구가 주로 영어로 진행되었던 것과 달리, 이 연구는 일본어 도메인에서 실행되었으며, 도메인 특화된 사전 훈련 언어 모델(PLM, Pre-trained Language Models)의 데이터 기억 문제에 초점을 맞추고 있습니다.

- **Technical Details**: 연구진은 일본어 신문 기사 데이터베이스를 활용하여 GPT-2 모델을 도메인 특화 형태로 사전 훈련시켰습니다. 실험을 통해, 모델 크기, 프롬프트의 길이 그리고 훈련 데이터의 중복성이 기억화(memorization)와 어떻게 관련되어 있는지 분석했습니다. 중요한 점은, 신문의 페이월(paywalls) 특성 때문에 훈련 데이터로 사용될 수 없는 기사들을 통해 데이터 오염 문제 없이 평가가 가능했다는 것입니다.

- **Performance Highlights**: 도메인 특화 PLM이 때때로 대규모로 '복사 및 붙여넣기'를 실행하는 경향이 있음을 발견했습니다. 또한, 일본어 GPT-2 모델에서도 이전 영어 연구에서 발견된 바와 같이, 훈련 데이터의 중복, 모델 크기, 프롬프트 길이가 기억화와 관련이 있다는 경험적 증거를 확인하였습니다.



### Small Language Models Need Strong Verifiers to Self-Correct Reasoning (https://arxiv.org/abs/2404.17140)
- **What's New**: 이 연구는 크기가 작은(13B 이하) 언어 모델(LM: Language Model)이 강력한 언어 모델로부터 최소한의 입력을 받고, 추론 작업에서 자가 수정(self-correction) 능력을 가지고 있는지를 탐구합니다. 이를 위해 작은 LM이 자체 수정 데이터를 수집하도록 유도하는 새로운 파이프라인을 제안합니다.

- **Technical Details**: 첫째, 연구진은 올바른 해답을 이용해 모델이 잘못된 반응을 비판하도록 유도합니다. 그 다음, 필터링 된 비판들은 자가 수정 추론자의 해답을 정제하는 데 있어 감독학습(supervised fine-tuning)에 사용됩니다. 연구에는 특히 강력한 GPT-4 기반 검증기와 결합했을 때 두 모델의 자가 수정 능력이 향상되었음을 보여주는 실험 결과가 포함되어 있습니다.

- **Performance Highlights**: 수학 및 상식 추론을 다루는 다섯 개의 데이터셋에서 두 모델의 자가 수정 능력이 개선되었습니다. 특히 GPT-4 기반의 강력한 검증기(verifier)와 페어링했을 때 눈에 띄는 성능 향상을 관찰할 수 있었습니다. 그러나 약한 자체 검증기를 사용할 때는 수정 시점을 결정하는 것에 한계가 있었음을 확인했습니다.



### Text Sentiment Analysis and Classification Based on Bidirectional Gated  Recurrent Units (GRUs) Mod (https://arxiv.org/abs/2404.17123)
- **What's New**: 이 연구는 자연어 처리(natural language processing, NLP) 분야에서 텍스트 감정 분석 및 분류의 중요성을 탐구하고 양방향 게이트 순환 유닛(bidirectional gated recurrent units, GRUs) 모델을 기반으로 한 새로운 접근 방식을 제안합니다. 이는 감정 레이블이 있는 텍스트의 단어 구름 모델(word cloud model)을 분석하고 데이터 전처리 과정을 거쳐 성능을 향상시킨 첫 사례입니다.

- **Technical Details**: 연구에서는 감정이 표시된 여섯 개의 레이블을 가진 텍스트의 단어 구름 모델을 분석한 후, 특수 기호, 구두점, 숫자, 불용어(stop words), 비알파벳 부분을 제거하는 데이터 전처리 과정을 진행하였습니다. 데이터 세트는 교육용(training set)과 테스트용(test set)으로 나뉘며, 모델 학습 및 테스트를 통해 검증 세트(validation set)의 정확도는 훈련을 통해 85%에서 93%로 8% 향상되었으며, 손실 값(loss value)은 0.7에서 0.1로 감소했습니다.

- **Performance Highlights**: 모델은 테스트 세트에서 94.8%의 정확도, 95.9%의 정밀도(precision), 99.1%의 재현율(recall), 그리고 97.4%의 F1 점수(F1 score)를 달성하며 우수한 일반화 능력과 분류 효과를 입증했습니다. 이러한 결과는 실제 값에 점차 가까워지며 텍스트 감정을 효과적으로 분류할 수 있는 방법을 제시합니다.



### 2M-NER: Contrastive Learning for Multilingual and Multimodal NER with  Language and Modal Fusion (https://arxiv.org/abs/2404.17122)
Comments: 20 pages

- **What's New**: 이 논문은 다국어 및 다중 모드(NER) 명명된 엔티티 인식의 새로운 도전을 탐색합니다. 이는 다양한 언어들과 다중 모드 데이터셋을 결합하여 NER의 효과를 향상시키는 것을 목표로 하고 있습니다. 저자들은 새로운 데이터셋을 구축했으며, 영어, 프랑스어, 독일어, 스페인어와 텍스트 및 이미지라는 두 가지 모드가 포함된 대규모 MMNER(multilingual and multimodal named entity recognition) 데이터셋을 소개합니다. 또한, 새로운 2M-NER 모델을 도입하여 텍스트와 이미지 표현을 조화롭게 결합하고 다중 모드 협력 모듈을 통해 두 모드 간의 상호작용을 효과적으로 나타냅니다.

- **Technical Details**: 2M-NER 모델은 대조 학습(contrastive learning)을 사용하여 텍스트와 이미지 표현을 정렬하며, 다중 모드 협력 모듈(multimodal collaboration module)을 통해 두 모드 간의 상호작용을 적극적으로 통합합니다. 이로 인해 다국어 및 다중 모드 NER 작업에서 더 높은 성과를 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 이 모델은 다국어 및 다중 모드 NER 작업에서 최고의 F1 점수를 달성했습니다. 이는 비교 및 대표적 베이스라인(basilines)과 비교하여 우수한 성능을 나타냅니다. 또한, 문장 수준의 정렬이 NER 모델에 많은 간섭을 초래하며, 이는 제안한 데이터셋의 난이도가 높음을 시사합니다.



### Talking Nonsense: Probing Large Language Models' Understanding of  Adversarial Gibberish Inputs (https://arxiv.org/abs/2404.17120)
- **What's New**: 이 연구는 대규모 언어 모델 (LLMs)이 인간의 언어를 얼마나 잘 이해하는지는 물론, 우리에게는 이해할 수 없는 LLM 자체 언어를 이해할 수 있는지 여부에 대해 탐구합니다. 우리는 LLM을 조작하여 의미없어 보이는 입력에서 일관된 반응을 생성하도록 하는 'LM Babel'이라는 프롬프트를 만드는 메커니즘을 밝히고자 합니다. 또한, 이 연구는 LLM이 잠재적으로 위험하거나 불법적인 텍스트를 생성할 수 있게 만드는 데 필요한 조건들에 대해서도 조사합니다. 

- **Technical Details**: 이 연구에서 사용된 Greedy Coordinate Gradient (GCG) 최적화 방법은 LLM에 대해 비정상적이거나 '바벨' 프롬프트라 불리는 입력들을 구성하여 모델이 특정한 반응을 생성하도록 유도합니다. 이 방법은 프롬프트의 길이와 대상 텍스트의 길이 및 복잡도(perplexity)에 따라 조작의 효율성이 크게 달라짐을 보여줍니다. 바벨 프롬프트는 자연스러운 프롬프트에 비해 더 낮은 손실 최소값(loss minima)에 위치하는 경향이 있습니다. 

- **Performance Highlights**: 바벨 프롬프트는 토큰 레벨에서 및 엔트로피 측면에서 그 구조를 검토한 결과, 높은 복잡성에도 불구하고 비자연적 토큰이 발견되며, 엔트로피가 무작위 토큰 문자열보다 낮게 유지됨을 확인하였습니다. 이런 프롬프트들은 모델 표현 공간에서 뭉쳐 있으며, 단일 토큰이나 구두점을 제거하는 것만으로도 성공률이 크게 감소하여 20% 이하, 때로는 3%까지 떨어집니다. 본 연구 결과는 또한 모델이 잘못된 출력을 생성하기 쉬웠음을 보여주며, 특히 유해한 텍스트를 재현하는 것이 양성 텍스트를 생성하는 경우보다 더 쉽다는 점을 시사합니다. 



### Player-Driven Emergence in LLM-Driven Game Narrativ (https://arxiv.org/abs/2404.17027)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 활용하여 플레이어가 게임 내러티브의 진화에 참여할 수 있게 하는 새로운 방식을 탐색합니다. 텍스트 기반 모험 게임 'Dejaboom!'을 통해 플레이어가 NPC와 상호 작용하며 게임의 전개를 탐색할 수 있도록 GPT-4 모델을 사용했습니다. 이러한 상호 작용을 통해 플레이어는 원래 내러티브에 없던 새로운 'emergent nodes'를 발견할 수 있었으며, 이는 게임 설계에 새로운 가능성을 제시합니다.

- **Technical Details**: GPT-4를 사용하여 플레이어의 게임 로그를 분석하고, 플레이어의 전략을 나타내는 노드와 게임 진행을 나타내는 방향성 에지로 구성된 내러티브 그래프로 변환합니다. 이를 통해 디자이너의 원래 내러티브와 플레이어가 생성한 내러티브 경로를 비교 분석할 수 있습니다. 이 게임은 TextWorld 엔진을 사용하여 구현되었으며, NPC와의 대화는 GPT-4로 실시간 생성됩니다.

- **Performance Highlights**: 28명의 게이머가 참여한 사용자 연구에서, 플레이어가 생성한 내러티브 그래프는 새로운 전략, 객체, 위치 추가와 같은 창의적 요소를 포함하고 있었습니다. 플레이어 중에서도 발견, 탐험 및 실험을 즐기는 이들이 가장 많은 'emergent nodes'를 생성했으며, 이러한 플레이어들은 게임 개발에 있어서 협업적 모델을 가능하게 할 중요한 역할을 할 수 있습니다.



### Türkçe Dil Modellerinin Performans  Karşılaştırması Performance Comparison of Turkish Language  Models (https://arxiv.org/abs/2404.17010)
Comments: in Turkish language. Baz{\i} \c{c}al{\i}\c{s}malar{\i} i\c{c}ermedi\u{g}ini s\"oyleyen hakem yorumu nedeniyle bir konferanstan kabul almad{\i}. Ancak hakemin bahsetti\u{g}i \c{c}al{\i}\c{s}malar bildiri g\"onderme son tarihinde yay{\i}nlanmam{\i}\c{s}t{\i}

- **What's New**: 이 연구는 터키어에 초점을 맞춘 언어 모델의 성능을 종합적으로 비교하는 것을 목적으로 합니다. 터키어 데이터셋을 이용하여 컨텍스트 학습(contextual learning) 및 질문 응답(question-answering) 능력을 평가하였으며, 다양한 오픈 소스 언어 모델들의 성능을 자동화된 평가 및 인간 평가를 통해 비교한 첫 연구입니다.

- **Technical Details**: 연구에서는 1.5억에서 7.5억 파라미터(parameter)를 가진 GPT 기반의 비교 언어 모델들을 선택하였습니다. 선택된 모델들은 터키어 데이터셋을 이용해 질문을 이해하고, 적절한 답변을 생성하는 능력 등을 평가하였으며, 이를 위해 컨텍스트 학습 및 질문 응답용 데이터셋(data set)이 준비되었습니다. 또한 모델의 성능 평가는 자동화된 도구와 인간의 평가를 모두 포함하는 방법으로 진행되었습니다.

- **Performance Highlights**: 결과적으로, 질문 응답(task; question-answering) 성능에 있어 주요한 발견은 터키어로의 적응을 위해 다국어 모델(multilingual model)에 지속적인 사전 학습(pretraining)과 별도의 교육 데이터셋(data set)을 사용하여 세분화(fine-tuning)하는 것이 더 효과적이라는 것입니다. 그러나 컨텍스트 학습 성능과 질문 응답 성능 간에는 큰 관련성이 없는 것으로 나타났습니다.



### Evaluating Class Membership Relations in Knowledge Graphs using Large  Language Models (https://arxiv.org/abs/2404.17000)
Comments: 11 pages, 1 figure, 2 tables, accepted at the European Semantic Web Conference Special Track on Large Language Models for Knowledge Engineering, Hersonissos, Crete, GR, May 2024, for associated code and data, see this https URL

- **What's New**: 이 연구에서는 지식 그래프(Knowledge graphs, KG)의 클래스 멤버십 관계를 평가하기 위한 새로운 방법을 제시합니다. 제로-샷 사고 체인(zero-shot chain-of-thought, CoT) 분류기를 사용하여 개체와 클래스의 자연어 설명을 처리함으로써 KG의 정확성을 평가합니다.

- **Technical Details**: 사용된 대규모 언어 모델(large language models, LLMs)은 GPT-4 등을 포함하며, 이 모델들은 자연어를 이해하고 분석하는 능력을 기반으로 KG에 저장된 지식이 정확하게 반영되었는지를 판단합니다. 평가는 Wikidata와 CaLiGraph라는 두 공개 지식 그래프를 사용하여 수행되었습니다. 분류 성능은 데이터에서 마크로 평균 F1-점수(macro-averaged F1-score)를 기준으로 측정되었습니다.

- **Performance Highlights**: Wikidata에서의 F1-점수는 0.830, CaLiGraph에서는 0.893으로 나타났습니다. 이는 LLM을 사용하여 KG의 오류를 식별하고 수정하는 데 효과적임을 시사합니다. 오류 분석을 통해 지식 그래프의 문제점이 밝혀지기도 했으며, 잘못된 관계가 주된 오류 원인이었습니다.



### Examining the robustness of LLM evaluation to the distributional  assumptions of benchmarks (https://arxiv.org/abs/2404.16966)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLM)의 벤치마크 평가 방식에 중요한 새로운 접근 방식을 제시합니다. 특히, 벤치마크 내의 테스트 프롬프트 간의 상관관계를 고려함으로써 모델 순위에 변화를 줄 수 있다는 점을 밝혀냈습니다. 이는 테스트 프롬프트가 실제 사용 사례의 분포를 임의적으로 반영하지 않는다는 기존의 가정과 달리, 특정 사용 사례에 따라 관심 분포가 다양할 수 있음을 시사합니다.

- **Technical Details**: 연구자들은 Transformer 아키텍처를 기반으로 하는 LLM의 성능을 평가하기 위하여 다양한 벤치마크를 사용합니다. 이들은 벤치마크에서 각각의 프롬프트에 대한 모델 응답을 평가하고, 이를 평균화하여 전체 벤치마크에 대한 단일 성능 메트릭(performance metric)을 산출합니다. 그러나 이 연구에서는 각 테스트 프롬프트(Test Prompts)의 연관성을 고려한 새로운 평가 방식을 제안하여, 동일 벤치마크에서도 모델 순위가 크게 달라질 수 있음을 보여주었습니다. 또한, 프롬프트의 의미적 유사성(semantic similarity)과 LLM의 공통 실패 지점(common LLM failure points)이 모델 성능의 유사성을 설명할 수 있는 주요 요인으로 밝혀졌습니다.

- **Performance Highlights**: 새로운 평가 방식을 적용한 결과, 모델의 성능과 순위는 최대 10%의 성능 변화와 5위까지의 순위 변동(rank changes)을 경험했습니다. 이는 벤치마크의 프롬프트 간 성능 상관관계가 유의미(p-value < 0.05)하며, 벤치마크 구성의 변화에 따라 모델 비교가 불안정할 수 있음을 시사합니다. 따라서, 벤치마크에서의 프롬프트 대표성과 분포 가정을 재고하는 것이 필수적임을 강조합니다.



### Samsung Research China-Beijing at SemEval-2024 Task 3: A multi-stage  framework for Emotion-Cause Pair Extraction in Conversations (https://arxiv.org/abs/2404.16905)
- **What's New**: 이 연구는 대화에서 감정 및 인과적 표현을 인식하고 식별하는 새로운 과제인 다모드 감정-원인 쌍 추출(Multimodal Emotion-Cause Pair Extraction in Conversations)에 초점을 맞추고 있습니다. 이를 위해 Llama-2 기반 InstructERC를 사용하여 각 대화의 감정 카테고리를 추출하고, 두 개의 주 스트림 주의 모델(two-stream attention model)과 MuTEC을 사용하여 타깃 감정을 고려한 감정 인과 쌍을 추출하는 다단계 프레임워크를 제안합니다.

- **Technical Details**: 첫 번째 단계에서는 Llama-2 기반의 InstructERC를 사용하여 대화 내 각 발화의 감정 범주를 추출합니다. 이어서, 대화에서 특정 감정에 대한 인과 표현을 식별하는 두 가지 하위 작업에 대해, 감정 인과 쌍을 추출하기 위해 두 개의 주 스트림 주의 모델을 활용하고, 첫 번째 하위 작업에 대해서는 감정 인과 범위(causal span)를 추출하기 위해 MuTEC을 사용합니다.

- **Performance Highlights**: 이 접근 방식은 두 하위 과제에서 모두 1위를 차지했으며, 특히 감정 인식(emotion recognition)과 감정 원인 쌍 추출(emotion-cause pair extraction)에서 높은 성능을 보여 주었습니다.



### Rumour Evaluation with Very Large Language Models (https://arxiv.org/abs/2404.16859)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 사용하여 정보의 오류를 방지하고 소셜 미디어에서의 소문을 탐지하는 새로운 접근 방식을 제안합니다. 특히, GPT-3.5-turbo 및 GPT-4와 같은 최신 LLM을 사용하여 RumourEval 태스크를 확장하고, 정보의 진위 여부와 관련된 태도를 분류합니다.

- **Technical Details**: 연구팀은 두 가지 서브태스크에 집중합니다: (1) 진위 예측(veracity prediction)과 (2) 태도 분류(stance classification). 진위 예측을 위해, 각 GPT 모델별로 세 가지 분류 체계를 실험하며, 제로샷(zero-shot), 원샷(one-shot), 퓨샷(few-shot) 설정에서 테스트합니다. 태도 분류에서는 기존 결과와 비슷한 성능을 보이며, finetuning 방법보다 향상된 결과는 없습니다. 또한, 태도 분류 서브태스크는 다중 클래스 분류를 허용하도록 확장되었습니다.

- **Performance Highlights**: 진위 예측에서, 최신 GPT 모델을 사용한 결과가 기존 방법보다 월등히 좋은 성능을 보였습니다. 각각의 예측은 신뢰도 점수와 함께 제공되어 LLM에 의한 신뢰성 정도를 결정하며, 설명 가능성과 해석 가능성을 위한 사후 정당화도 포함됩니다.



### A Semi-Automatic Approach to Create Large Gender- and Age-Balanced  Speaker Corpora: Usefulness of Speaker Diarization & Identification (https://arxiv.org/abs/2404.17552)
Comments: Keywords:, semi-automatic processing, corpus creation, diarization, speaker identification, gender-balanced, age-balanced, speaker corpus, diachrony

- **What's New**: 본 논문에서는 연령, 성별 및 녹음 시기별로 균형을 맞춘 이력(corpus)을 반자동으로 생성하는 새로운 접근법을 제시합니다. 이 연구는 프랑스 국립 시청각 연구소(INA)에서 선택된 코퍼스를 통해 32개 범주(성별 2개, 연령대 4개, 녹음 기간 4개)에 대해 각 범주당 최소 30명의 발화자(speakers)를 확보하였습니다.

- **Technical Details**: 연구팀은 발화 감지(speech detection), 배경 음악 및 중첩 발화 제거(background music and overlapped speech removal), 발화자 구분(speaker diarization)으로 구성된 자동 파이프라인을 사용하여 오디오비주얼 문서에서 발화 부분을 추출하였습니다. 이 파이프라인은 수동 처리 시간을 10분의 1로 단축시켰으며, 최종 출력의 품질 평가도 제공됩니다. 발화자를 식별하는 인간 주석자(human annotators)에게 깨끗한 발화자 세그먼트를 제공합니다.

- **Performance Highlights**: 자동 처리의 품질과 최종 출력은 최신 공정(up-to-date process)과 비교하여 높은 품질의 발화를 제공하는 것으로 평가되었습니다. 이 방법론은 알려진 대상 발화자의 대규모 코퍼스를 생성하는 데 유망한 접근법으로 보여집니다.



### Probabilistic Inference in Language Models via Twisted Sequential Monte  Carlo (https://arxiv.org/abs/2404.17546)
- **What's New**: 이 연구에서는 대규모 언어 모델 (Large Language Models, LLMs)의 능력 및 안전 기술들을 효과적으로 적용하기 위해 순차적 몬테 카를로 (Sequential Monte Carlo, SMC) 방법을 사용합니다. 이 방법은 데이터를 샘플링하는 과정에서 발생할 수 있는 문제점들을 해결하기 위해 고안되었습니다. 특히, 미래 잠재 가치를 추정하는 학습된 twist 함수를 사용하여, 유망한 부분 시퀀스에만 계산을 집중시킬 수 있습니다.

- **Technical Details**: 제안된 방법론은 twist 함수를 학습하기 위해 대조적 방법(contrastive method)을 사용하며, 이는 소프트 강화 학습(soft reinforcement learning)의 풍부한 문헌과 연결됩니다. 또한, twisted SMC 프레임워크의 보완적인 응용으로, 언어 모델 추론 기술의 정확성을 평가하기 위한 새로운 양방향 SMC 경계를 제시합니다.

- **Performance Highlights**: 이 방법을 사용하여 예측 모델로부터 바람직하지 않은 출력을 효과적으로 샘플링하는 것이 가능하였고, 다양한 감정을 가진 리뷰 생성 및 인필링(infilling) 작업에도 효과적임을 보여줍니다. 또한, 로그 분할 함수(log partition function)에 대한 새로운 SMC 경계를 사용하여 추론 및 목표 분포 사이의 KL 발산(KL divergence)을 양방향으로 추정할 수 있습니다.



### Large Language Model Agent as a Mechanical Designer (https://arxiv.org/abs/2404.17525)
- **What's New**: 이 연구에서는 사전 훈련된 LLM(대규모 언어 모델, Large Language Models)을 FEM(유한 요소 방법, Finite Element Method) 모듈과 통합하는 새로운 접근 방식을 제시합니다. 이 통합을 통해 LLM은 도메인 특화 훈련 없이도 설계를 지속적으로 학습, 계획, 생성 및 최적화할 수 있습니다.

- **Technical Details**: LLM과 FEM 모듈을 결합함으로써, 각 설계에 대한 평가와 필수 피드백을 제공하여 설계 최적화 과정을 자동화합니다. 이 시스템은 자연어 사양에 따라 트러스(truss) 구조의 반복적 최적화를 관리하는 데 효과적임을 보여줍니다. 또한, 솔루션-점수 쌍을 제공하여 설계를 반복적으로 정제함으로써 LLM 기반 에이전트가 최적화 동작을 보이는 것을 시연합니다.

- **Performance Highlights**: LLM 기반 에이전트는 자연어 사양을 준수하는 트러스 설계를 최대 90%의 성공률로 생성할 수 있습니다. 이는 적용된 제약 조건에 따라 달라집니다. LLM 에이전트들의 이러한 능력은 그들이 효과적인 설계 전략을 독립적으로 개발하고 실행할 수 있는 잠재력을 강조합니다.



### On the Use of Large Language Models to Generate Capability Ontologies (https://arxiv.org/abs/2404.17524)
- **What's New**: 이 연구는 대용량 언어 모델(Large Language Models, LLMs)이 시스템 또는 기계의 기능을 모델링하는 능력 온톨로지(capability ontologies)의 생성을 어떻게 도울 수 있는지 조사합니다. 특히, 다양한 복잡성을 가진 능력을 생성하기 위해 다양한 프롬프팅 기법(prompting techniques)과 여러 LLM을 사용한 일련의 실험을 소개합니다.

- **Technical Details**: 능력 온톨로지 생성을 위해 사용된 LLM은 자연 언어 텍스트 입력으로부터 기계 해석 가능한 모델들을 생성할 능력이 있습니다. 이 연구에서는 RDF(RDF), OWL(Web Ontology Language), SHACL(SHACL constraints)을 이용한 반자동 접근 방식을 통해 생성된 온톨로지의 품질을 분석합니다. 이러한 점검을 통해 온톨로지의 구문, 추론, 제약 조건을 평가합니다.

- **Performance Highlights**: 실험 결과는 매우 긍정적입니다. 복잡한 능력의 경우에도 생성된 온톨로지는 거의 오류가 없이 나타났습니다. 이는 LLM이 온톨로지 전문가들을 지원할 수 있는 효과적인 도구로서의 가능성을 보여줍니다.



### Automated Data Visualization from Natural Language via Large Language  Models: An Exploratory Study (https://arxiv.org/abs/2404.17136)
- **What's New**: 자연어 설명을 시각적 표현으로 변환하는 NL2Vis(자연어로 시각화) 작업은 대용량 데이터에서 통찰력을 얻기 위해 개발되었습니다. 최근 이 분야에서는 딥러닝 기반 접근법이 많이 등장했으나, 보이지 않는 데이터베이스나 여러 테이블에 걸친 데이터를 시각화하는 데는 도전이 남아있습니다. 이 논문은 대규모 언어 모델(LLMs: Large Language Models)의 생성 능력을 활용하여 시각화 생성 가능성을 평가하고, 맥락 학습 프롬프트(in-context learning prompts)의 효과를 탐구합니다.

- **Technical Details**: 이 연구는 구조화된 테이블 데이터를 순차적 텍스트 프롬프트로 변환하는 방법을 탐구하고, LLMs를 통해 입력하여 어떤 테이블 내용이 NL2Vis 작업에 가장 중요한 기여를 하는지 분석합니다. '테이블 스키마(table schema)'는 프롬프트 구성시 고려해야 할 중요 요소입니다. 실험은 두 가지 유형의 LLMs(예: T5-Small, GPT-3.5)과 최신 방법을 nvBench 벤치마크를 사용하여 비교 평가했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 베이스라인 모델들을 능가하며, 일부 적은 샷(few-shot) 시연을 통해 인퍼런스 전용 모델이 고도로 튜닝된(fin-tuned) 모델들을 능가하기도 했습니다. LLMs가 실패하는 사례를 분석하고, 사고의 사슬(Chain-of-Thought), 역할극(role-playing) 및 코드 해석기(code-interpreter) 같은 전략을 통해 결과를 반복적으로 업데이트하는 것이 효과적임을 확인했습니다.



### A Closer Look at Classification Evaluation Metrics and a Critical  Reflection of Common Evaluation Practic (https://arxiv.org/abs/2404.16958)
Comments: to appear in TACL, this is a pre-MIT Press publication version

- **What's New**: 이 연구는 다양한 평가 메트릭(evaluation metrics)들이 어떻게 기대치와 맞물려 있는지 분석합니다. 특히 '매크로 지표'(macro metrics), 예를 들어 '매크로 F1'과 같은 메트릭이 실제로 무엇을 의미하는지에 대한 명확성이 결여된 상태에서 선택되는 경우가 많다는 점을 지적하며, 이러한 선택이 연구 결과나 공유 과제의 순위에 영향을 줄 수 있음을 경고합니다. 이 논문은 보다 명확하고 투명한 메트릭 선택을 돕기 위해 여러 평가 메트릭들을 심층 분석하고, 자연어 처리(Natural Language Processing, NLP)의 최근 공유 과제에서의 메트릭 선택을 검토합니다.

- **Technical Details**: 이 연구는 분류 시스템(classifier)의 예측 성능을 평가하기 위해 혼동 행렬(confusion matrix)을 구성하고, 이를 요약하여 메트릭으로 표현하는 과정을 다룹니다. 저자는 '정확도'(Accuracy), '매크로 리콜과 프리시전'(Macro Recall and Precision), '가중 F1'(Weighted F1), '카파'(Kappa) 및 '매튜스 상관 계수'(Matthews Correlation Coefficient, MCC) 등 다양한 평가 메트릭을 분석합니다. 또한, 이러한 메트릭들이 어떻게 데이터의 불균형(imbalance)과 같은 문제를 해결하는지, 혹은 실패하는지를 점검합니다.

- **Performance Highlights**: 저자들은 많은 연구들에서 사용된 평가 메트릭 선택이 종종 설득력 있는 근거 없이 이루어졌다는 것을 발견했습니다. 이는 분류기(classifier)의 성능을 평가하고 순위를 매길 때 임의성을 초래할 수 있습니다. 이 논문은 메트릭 선택이 얼마나 중요한지를 강조하고, 해당 선택이 고려되어야 하는 다양한 측면들을 제시함으로써, 보다 의미 있는 평가를 촉진하고자 합니다.



### A Survey of Generative Search and Recommendation in the Era of Large  Language Models (https://arxiv.org/abs/2404.16924)
- **What's New**: 인터넷 상의 정보의 폭발로 인하여 검색과 추천은 사용자의 정보 요구를 만족시키기 위한 기본적 인프라가 되었습니다. 이 논문은 최근에 등장한 정보 시스템 내의 새로운 패러다임인 생성적 검색(generative search)과 생성적 추천(generative recommendation)에 대한 포괄적인 조사를 제공합니다. 이는 문서나 항목에 대한 쿼리나 사용자의 매칭 문제를 생성적 방식으로 해결하고자 하는 것을 목표로 합니다.

- **Technical Details**: 연구는 기계 학습(machine learning) 및 딥 러닝(deep learning)을 포함한 동기적 기술 패러다임 전환을 경험해왔습니다. 현재 초지능형 생성 대규모 언어 모델(superintelligent generative large language models)에 의해 시작된 새로운 패러다임에 초점을 맞추고 있습니다. 이 논문에서는 생성적 패러다임(generative paradigm)에 대해 통합된 프레임워크를 추상화하고, 기존 작업을 이 프레임워크 내의 다른 단계로 나누어 강점과 약점을 강조합니다.

- **Performance Highlights**: 생성적 검색과 추천의 강점과 약점을 분석하고, 독특한 도전 과제를 구별하며, 해결되지 않은 문제(open problems)와 미래의 방향을 식별하여 차세대 정보 검색 패러다임을 제시합니다.



### A Short Survey of Human Mobility Prediction in Epidemic Modeling from  Transformers to LLMs (https://arxiv.org/abs/2404.16921)
- **What's New**: 본 논문은 전염병 동안 인간 이동 패턴을 예측하기 위해 특히 Transformer 모델을 활용하는 최신 기계 학습 기술에 대한 종합적인 조사를 제공합니다. 이 연구는 전염병 확산 모델링 및 효과적인 대응 전략 수립에 필수적인 인간의 이동성 이해에 기여할 것입니다. 전염병 모델링 문맥에서의 인간 이동 패턴을 모델링하기 위해 Transformer와 대규모 언어 모델(Large Language Models, LLM)을 활용한 최근의 연구 진전을 포괄적으로 개관합니다.

- **Technical Details**: 이 논문은 인간 이동 패턴의 생성 및 예측과 같은 주요 작업을 설명하며, 인간의 이동 경로를 예측하는 데 Transformer 모델이 어떻게 활용될 수 있는지 심층적으로 탐구합니다. Transformer의 인코더와 디코더 구조를 활용하고, 주의(Attention) 메커니즘을 통해 시공간(Spatio-temporal) 의존성과 맥락적 패턴을 효과적으로 포착할 수 있는 능력을 강조합니다. 또한 다양한 데이터 소스를 결합하여 인간 이동 패턴에 대한 깊은 통찰을 제공하는 다중 모드(Multimodal) 모델의 사용이 증가하고 있는 점을 지적합니다.

- **Performance Highlights**: Transformer 모델과 LLM의 응용은 특히 전염병 모델링과 같은 복잡한 문제에 있어 더 나은 예측 능력과 처리 능력을 보여주었습니다. 이러한 모델들은 인간의 위치 및 시간 데이터를 입력으로 받아 정확한 미래 위치를 예측하는 데 탁월한 성능을 발휘하며, 전염병 확산의 조기 감지와 확산 방지를 위한 조치의 효과성 평가에 큰 기여를 하고 있습니다.



### Prediction Is All MoE Needs: Expert Load Distribution Goes from  Fluctuating to Stabilizing (https://arxiv.org/abs/2404.16914)
- **What's New**: 이번 연구에서는 큰 언어 모델(Large Language Models, LLMs)에서 활용되는 MoE(Mixture of Experts) 기술의 전문가 부하(expert load)를 정밀하게 예측하기 위해 세 가지 고전적 예측 알고리즘을 사용하여 전문가 부하의 변동과 안정성을 분석했습니다. MoE는 모델의 매개 변수가 증가함에 따라 계산 복잡성이 선형적으로 증가하지 않도록 돕습니다.

- **Technical Details**: MoE는 특정 전문가 네트워크에 입력을 할당하고 그 출력을 최종 출력으로 통합하는 아키텍처입니다. 입력 토큰에 대해 처리할 전문가 집합을 선택하는 학습 스파스 게이팅 네트워크(sparse gating network)를 사용합니다. 이 연구에서는 GPT-3 350M 모델을 기준으로, 1,000단계 및 2,000단계 후의 전문가 부하 비율을 예측하는 평균 오차율이 각각 약 1.3% 및 1.8%임을 밝혔습니다.

- **Performance Highlights**: 전문가 부하의 변동과 안정성에 대한 정의를 규명하고, 세 가지 고전 예측 알고리즘을 사용하여 높은 예측 정확도를 달성하였습니다. 이러한 분석을 통해 모델 트레이닝 중 자원 배치 혹은 재배치에 대한 중요한 지침을 제공할 수 있습니다.



### Attacks on Third-Party APIs of Large Language Models (https://arxiv.org/abs/2404.16891)
Comments: ICLR 2024 Workshop on Secure and Trustworthy Large Language Models

- **What's New**: 최근 대규모 언어 모델(Large Language Model, LLM) 서비스에서는 제3자 API 서비스와 상호작용하는 플러그인 생태계를 제공하기 시작했습니다. 이 혁신은 LLM의 기능을 향상시키지만, 다양한 제3자가 개발한 플러그인은 쉽게 신뢰할 수 없기 때문에 위험도 도입합니다.

- **Technical Details**: 이 논문에서는 제3자 서비스를 통합한 LLM 플랫폼 내의 보안 및 안전 취약점을 검사하기 위한 새로운 공격 프레임워크(Attacking Framework)를 제안합니다. 널리 사용되는 LLM에 우리의 프레임워크를 적용하여, 제3자 API가 LLM 출력을 눈에 띄지 않게 수정할 수 있는 실제 악의적인 공격을 여러 도메인에서 식별했습니다.

- **Performance Highlights**: 논문은 제3자 API 통합이 초래하는 고유한 도전과제를 논의하고, LLM 생태계의 보안 및 안전을 개선하기 위한 전략적 가능성을 제공합니다. 또한, 연구를 지원하기 위한 코드를 이 https URL에서 공개했습니다.



### Atomas: Hierarchical Alignment on Molecule-Text for Unified Molecule  Understanding and Generation (https://arxiv.org/abs/2404.16880)
- **What's New**: 새로운 다중모드(multi-modal) 분자 표현 학습 프레임워크인 'Atomas'가 제안되었습니다. 이는 SMILES 문자열과 텍스트로부터 동시에 표현을 학습하여 분자의 품질을 향상시키고 다양한 과학 분야의 성능을 개선하는 데 도움을 줍니다. 기존의 글로벌 정렬 방식(global alignment approach)이 미세 정보를 포착하지 못하는 단점을 개선하기 위해, Atomas는 세 단계로 분자 조각(fragment) 간의 상관성을 학습하는 계층적 적응 정렬 모델(Hierarchical Adaptive Alignment model)을 설계했습니다.

- **Technical Details**: Atomas는 계층적 적응 정렬 모델을 통해 두 모드 간의 미세한 조각 정보를 동시에 학습하고 여러 레벨에서 이들의 표현을 정렬합니다. 또한, 이 프레임워크는 분자를 이해하고 생성하는 작업을 포함하는 엔드 투 엔드(end-to-end) 훈련 체계를 채택하여 더 넓은 범위의 하류 작업(downstream tasks)을 지원합니다.

- **Performance Highlights**: 검색 작업(retrieval task)에서 Atomas는 강력한 일반화 능력을 보여주며 평균 recall@1에서 기준 모델(baseline)보다 30.8% 높은 성능을 보였습니다. 생성 작업(generation task)에서는 분자 캡셔닝(molecule captioning) 작업과 분자 생성(molecule generation) 작업 모두에서 최신 최고의 결과(state-of-the-art results)를 달성했습니다.



### AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs (https://arxiv.org/abs/2404.16873)
Comments: 32 pages, 9 figures, 7 tables

- **What's New**: 이 논문에서는 AdvPrompter라는 새로운 LLM(Large Language Model)을 사용하여 인간이 읽을 수 있는 적대적 프롬프트를 몇 초 내에 생성하는 새로운 방법을 제시합니다. 이 방법은 기존의 최적화 기반 접근 방식보다 약 800배 빠릅니다. AdvPrompter는 TargetLLM의 그라디언트(gradient) 정보에 접근할 필요 없이 훈련됩니다. 또한, AdvPrompter를 통해 생성된 합성 데이터셋에서 미세 조정(fine-tuning)을 통해 LLM을 강화하고, 탈옥 공격에 대한 내성을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: AdvPrompter 훈련 과정은 타깃 적대적 서픽스(target adversarial suffixes) 생성과 AdvPrompter의 저랭크(low-rank) 미세 조정을 번갈아 수행하는 방식으로 이루어집니다. 이 과정을 통해 입력 지시문의 의미는 변경하지 않으면서 TargetLLM을 해로운 반응으로 유도하는 서픽스를 생성합니다. 기술적으로는 복잡도 기반 필터(perplexity-based filters)를 우회하고, 디스크리트 최적화(discrete optimization) 과정이 필요 없는 접근 방식입니다.

- **Performance Highlights**: 실험 결과는 인기 있는 오픈 소스 TargetLLMs에서 최첨단 결과를 보여주며 AdvBench 데이터셋에서도 이러한 결과는 검증됩니다. 또한, AdvPrompter에서 생성된 합성 데이터셋에 대한 미세 조정을 통해 LLM의 탈옥 공격에 대한 강화된 내성을 확인할 수 있습니다. 특히, MMLU(Massive Multitask Language Understanding) 점수를 유지하면서, 강화된 내성을 보여줍니다.



### A Disease Labeler for Chinese Chest X-Ray Report Generation (https://arxiv.org/abs/2404.16852)
- **What's New**: 이 연구는 중국어 흉부 X선 검사 보고서 생성을 위한 새로운 질병 라벨러(disease labeler)를 제안합니다. 이 라벨러는 진단 보고서와 임상 정보를 별도로 처리할 수 있는 듀얼 BERT(dual BERT) 구조를 활용하며, 질병과 신체 부위 간의 연관성을 기반으로 계층적 라벨 학습 알고리즘(hierarchical label learning algorithm)을 구축하여 텍스트 분류 성능을 강화합니다.

- **Technical Details**: 이 라벨러는 복잡한 임상 데이터를 효과적으로 처리하기 위해 BERT(Bidirectional Encoder Representations from Transformers) 구조를 두 개 사용합니다. 하나는 진단 보고서용이고 다른 하나는 임상 정보용입니다. 또한, 질병과 관련된 라벨을 계층적으로 학습함으로써, 보다 정확한 질병 분류가 가능해집니다. 이를 통해 구축된 데이터셋은 총 51,262건의 중국어 흉부 X선 보고서 샘플을 포함하고 있습니다.

- **Performance Highlights**: 제안된 라벨러는 전문가가 주석을 단(subset of expert-annotated) 중국어 흉부 X선 보고서의 집합에 대한 실험 및 분석을 통해 그 효과를 검증하였습니다. 이 라벨러를 이용하여 생성된 보고서는 임상 정확성(clinical accuracy)과 효능(effectiveness) 측면에서 높은 평가를 받았습니다.



### Automatic Speech Recognition System-Independent Word Error Rate  Estimation (https://arxiv.org/abs/2404.16743)
Comments: Accepted to LREC-COLING 2024 (long)

- **What's New**: 이 논문에서는 자동 음성 인식(ASR) 시스템에 독립적인 단어 오류율(WER) 추정 방법, 즉 System-Independent WER Estimation (SIWE) 방법을 제안하였습니다. 전통적인 ASR 시스템 의존적 추정자와 달리, 제안된 SIWE 모델은 다양한 데이터 증강 (data augmentation) 방법을 활용하여 더 넓은 범위의 적용과 높은 유연성을 제공합니다.

- **Technical Details**: SIWE는 음성 데이터에서 발생할 수 있는 오류를 삽입하는 새로운 데이터 증강 기법을 활용하여 트레이닝 데이터셋을 생성합니다. 구체적으로, 삽입(insertion), 삭제(deletion), 대체(substitution) 오류를 통해 가설을 생성합니다. 이러한 방법은 SIWE가 ASR 시스템에 독립적으로 작동할 수 있도록 하며, 다양한 ASR 시스템으로부터의 트레이닝 데이터셋의 필요성을 제거합니다.

- **Performance Highlights**: 제안된 SIWE 모델은 도메인 내(in-domain) 데이터에서 기존의 시스템 의존적 WER 추정자들과 유사한 성능을 보였으며, 도메인 외(out-of-domain) 데이터에서는 상대적으로 기준 모델보다 17.58% 향상된 평균 제곱근 오차(RMSE)와 18.21% 향상된 피어슨 상관 계수(Pearson correlation coefficient)를 달성하여 최고의 성능을 보였습니다.



### ToM-LM: Delegating Theory of Mind Reasoning to External Symbolic  Executors in Large Language Models (https://arxiv.org/abs/2404.15515)
- **What's New**: 신규 연구에서는 엘엘엠(LLM: Large Language Model)의 마음이론(Theory of Mind, ToM) 추론 능력을 개선하기 위해 외부 기호 실행자(SMCDEL 모델 체커)를 활용하고 파인 튜닝(fine-tuning)을 시행한 새로운 방법을 제시합니다. 이 방법은 LLM이 자연어로 제시된 ToM 문제를 기호적 형식으로 변환하고, 이를 SMCDEL 모델 체커를 활용해 투명하고 검증 가능한 추론으로 처리하도록 합니다. 이는 ToM 추론 과정을 외부화하고 대행시키는 새로운 시각을 제안합니다.

- **Technical Details**: 연구에서 사용된 엘엘엠은 GPT-3.5 터보로, 오픈AI 파인 튜닝 플랫폼에서 자연어와 기호 형식 표현의 ToM 문제 쌍으로 파인 튜닝되었습니다. 생성된 기호 형식은 SMCDEL(Model Checker)을 사용해 실행되며, 동적인 미덱스 논리 문제를 처리하는 데 적합하게 설계되었습니다. 이 과정을 통해 LLM의 복잡한 ToM 추론 능력이 향상될 것으로 기대됩니다.

- **Performance Highlights**: ToM-LM 접근 방식은 기존 베이스라인에 비해 눈에 띄게 개선된 결과를 보여줍니다. 정확도 및 ROC 곡선 아래 면적(AUC)에서 상당한 향상을 보였으며, ToM 추론 과정이 투명하고 검증 가능하다는 점에서 우수합니다.




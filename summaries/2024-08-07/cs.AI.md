New uploads on arXiv(cs.CL)

### CoverBench: A Challenging Benchmark for Complex Claim Verification (https://arxiv.org/abs/2408.03325)
- **What's New**: 새로운 연구인 CoverBench는 복잡한 추론 설정에서 언어 모델(LM) 출력의 정확성을 검증하는 데 중점을 둔 도전적인 벤치마크를 소개합니다. CoverBench는 다양한 도메인에서 복잡한 주장 검증을 평가하기 위해 설계되었으며, 데이터 품질을 보장하기 위해 수동으로 검토된 데이터를 포함하고 있습니다.

- **Technical Details**: CoverBench는 위키피디아, 금융, 생의학, 법률 등의 다양한 언어 도메인에서 복잡한 추론을 필요로 하는 9개의 데이터셋을 통합합니다. 복잡한 주장은 정형 데이터, 긴 문맥, 정량적 추론, 도메인 전문 지식, 다단계 추론 등을 포함합니다. 모든 데이터는 표준화된 형식(HTML, JSON, Markdown)으로 변환되었으며, 거짓 주장을 생성하고 도전적인 예제를 선택하기 위해 메타데이터와 모델 기반 선택을 활용했습니다.

- **Performance Highlights**: 최종 벤치마크는 평균 3,500 토큰의 긴 문맥과 이를 기반으로 한 복잡한 주장 733 예제를 포함합니다. 대부분의 모델은 임의 기준치 수준의 성능을 보였으며, 가장 좋은 모델도 65 Macro-F1 점수 이하를 기록했습니다. 이러한 결과는 해당 작업에 상당한 발전 가능성이 있음을 시사합니다.



### Training LLMs to Recognize Hedges in Spontaneous Narratives (https://arxiv.org/abs/2408.03319)
Comments:
          Amie Paige, Adil Soubki, and John Murzaku contributed equally to this study

- **What's New**: 이번 연구는 로드러너 만화 이야기로 구성된 실험적 코퍼스를 기반으로, BERT와 GPT-4o 및 LLaMA-3을 사용하여 헤지(Hedge) 감지 성능을 평가했습니다. 가장 우수한 성능은 BERT 모델의 미세 조정(fine-tuning)을 통해 달성되었으며, 이를 통해 헤지의 애매모호한 사례를 식별하고 향후 연구 방향을 지도하는 데 도움을 줄 수 있는 LLM-in-the-Loop 접근법을 제안합니다.

- **Technical Details**: 헤지는 발화자가 발화를 임시적(provisional)으로 표시하는 다양한 형태의 담화 표지입니다. 연구에서는 BERT 모델을 미세 조정(fine-tuning)하고, GPT-4o와 LLaMA-3를 사용해 zero-shot 및 few-shot 학습을 비교 평가했습니다. 로드러너 헤지 코퍼스라는 표준 데이터를 활용하여, LLM를 사용한 코드의 정확성을 높였으며, 오류 분석을 통해 헤지 감지 실패 원인을 파악하고 LLM-in-the-Loop 접근법을 사용하여 표본 조사를 보강했습니다.

- **Performance Highlights**: BERT 모델을 미세 조정한 방법이 가장 우수한 성능을 보였으며, 그 다음으로 few-shot GPT-4o가 우수한 성능을 보였습니다. 상세한 오류 분석을 통해 헤지 감지의 실패 지점을 식별하고 이를 개선하기 위한 LLM-in-the-Loop 접근법을 도입했습니다.



### KaPO: Knowledge-aware Preference Optimization for Controllable Knowledge Selection in Retrieval-Augmented Language Models (https://arxiv.org/abs/2408.03297)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 지식 선택 능력을 향상시키기 위해 'Knowledge-aware Preference Optimization' (KaPO)을 제안합니다. KaPO는 실제 검색 시나리오에서 발생할 수 있는 오류 타입을 탐색하고, 이를 통해 부정적인 신호를 피하는 방법을 학습합니다. 이를 통해 LLM이 지식 충돌 상황에서 보다 강력한 성능을 발휘할 수 있도록 돕습니다.

- **Technical Details**: KaPO의 핵심 전략은 'preference optimization'을 통해 부정적인 신호를 감소시키는 것입니다. 다채로운 컨텍스트 조합을 통해 발생할 수 있는 오류 타입을 시뮬레이션하고, 이를 통해 LLM이 적절한 답변을 선택하도록 유도합니다. 동시에 응답 길이와 다양한 행동 패턴을 대표하는 preference 데이터의 비율을 조정함으로써 모델의 순응 능력과 잡음에 대한 견고성을 균형 있게 개선합니다.

- **Performance Highlights**: 실험 결과, KaPO는 기존의 방법들에 비해 지식 충돌을 처리하는 능력이 37% 이상 향상되었으며, 다양한 out-of-distribution 데이터셋에서도 강력한 일반화 성능을 보여주었습니다.



### StructEval: Deepen and Broaden Large Language Model Assessment via Structured Evaluation (https://arxiv.org/abs/2408.03281)
Comments:
          ACL 2024;Benchmark at this https URL at this https URL

- **What's New**: StructEval이라는 새로운 평가 프레임워크를 제안하였습니다. 이는 기존의 단일 항목 평가 패러다임의 한계를 극복하고, 대규모 언어 모델(LLM)의 실제 능력을 평가하기 위한 다단계 평가 접근 방식을 채택합니다. StructEval은 여러 인지 수준과 중요한 개념에 대해 구조화된 평가를 통해 LLM의 성능을 포괄적이고 일관되게 평가할 수 있습니다.

- **Technical Details**: StructEval은 하나의 기본 검사 목표에서 시작하여 Blooms Taxonomy Theory와 개념 매핑 이론에 따라 여러 인지 수준과 핵심 개념을 추출합니다. 이를 통해 모델의 각 테스트 목표에 대해 여러 인스턴스를 생성하여 평가합니다. 이는 단일 항목 평가와 달리 모델의 이해 능력을 다각도로 평가할 수 있게 합니다.

- **Performance Highlights**: StructEval은 데이터 오염 위험을 효과적으로 저항하고 잠재적인 편향의 간섭을 줄이며, 모델 간의 일관된 평가 결론을 제공합니다. 실험 결과, 이전의 증강 기반 전략(단어 변형, 패러프레이징(Paraphrasing), 역번역, 옵션 셔플 등)보다 우수한 성능을 보였습니다. 또한, 대규모 벤치마크를 자동으로 생성하고 높은 품질의 평가를 보장할 수 있습니다.



### Synthesizing Text-to-SQL Data from Weak and Strong LLMs (https://arxiv.org/abs/2408.03256)
Comments:
          12 pages, 7 figures, ACL 2024

- **What's New**: 이번 연구에서는 open-source와 closed-source 대형 언어 모델(LLMs) 간의 텍스트-투-SQL(text-to-SQL) 작업의 성능 격차를 줄이기 위한 새로운 방법을 제시합니다. 이 방법은 더 크고 강력한 모델(strong models)이 생성한 데이터를 활용하면서, 작은 모델(weak models)의 에러 데이터를 결합하여 텍스트-투-SQL 모델의 도메인 일반화를 향상시킵니다. 이 방법으로 SENSE라는 특화된 텍스트-투-SQL 모델을 개발했으며, 이는 SPIDER와 BIRD 벤치마크에서 최첨단(state-of-the-art) 성과를 거두어 open-source 모델과 closed-source 모델 간의 성능 격차를 줄였습니다.

- **Technical Details**: 본 연구에서는 첫째, SQL 쿼리를 생성할 때 더 강력한 모델을 사용하여 데이터의 다양성을 촉진합니다. 둘째, 작은 모델이 생성한 에러 데이터를 바탕으로 하여 오류 학습(preference learning)을 통해 모델이 에러 정보를 습득하도록 합니다. 이러한 방법론을 통해 Supervised Fine-Tuning(SFT)을 수행하여 open-source LLMs의 텍스트-투-SQL 성능을 크게 향상시키려 했습니다. 이 실행 과정에서 강력한 LLM인 GPT-4를 활용한 data synthesis를 포함하며, 보다 높은 도메인 다양성과 복잡성을 갖춘 synthetic data를 생성했습니다.

- **Performance Highlights**: SENSE 모델은 SPIDER와 BIRD 벤치마크에서 뛰어난 성능을 발휘하며, 기존의 open-source 모델들과 비교하여 significant한 성능 향상을 보였습니다. 또한 SYN, REALISTIC, DK와 같은 세 가지 robust 데이터셋에서도 SENSE의 우수한 성능이 입증되었습니다. 이러한 연구 결과는 텍스트-투-SQL 커뮤니티의 발전에 기여할 수 있는 가능성을 보여줍니다.



### Unveiling Factual Recall Behaviors of Large Language Models through Knowledge Neurons (https://arxiv.org/abs/2408.03247)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 추론 작업에서 사실적 지식을 내부적으로 기억하고 활용하는지 조사합니다. 연구 결과, LLMs는 특정 상황에서 중요한 사실적 연관성을 잘 활용하지 못하고 오히려 지름길 같은 경로를 선택하는 경향이 있음을 밝혔다. CoT(Chain-of-Thought) 프롬프팅이 사실 지식을 강하게 기억하도록 유도하는 데 효과적임을 보여주었고, 매개 변수 지식의 수동적 조작을 통해 추론 성능을 향상시키는 법을 검토했다.

- **Technical Details**: 사실 기억에서 LLMs의 내부 동적 과정을 조사하기 위해 'Knowledge Neurons(KNs)' 기술을 활용하여 두 단계 추론(2-hop reasoning) 질문 데이터셋을 작성하고, 각 추론 단계에서 새로운 지표인 'KN 점수'를 도입해 평가했다. 추후 사실 지식의 회수 과정에서 CoT 프롬프팅이 통제 조건(조작 전후)에 어떤 영향을 미치는지 분석했다.

- **Performance Highlights**: ['LLMs는 사실 연관성을 잘못 기억하여 추론 오류의 1/3 이상을 차지한다.', 'CoT 프롬프팅을 통해 단계별 추론을 촉진함으로써 사실 지식의 회수를 현저히 개선할 수 있다.', '사실 회수 과정을 강화하거나 억제함으로써 추론 성능에 중요한 영향을 미친다.', '외부 지식의 충돌이 있을 때, 사실 회수 과정이 더 개선될 수 있다.']



### Making Long-Context Language Models Better Multi-Hop Reasoners (https://arxiv.org/abs/2408.03246)
Comments:
          ACL 2024 Main Conference Camera Ready; Dataset, model, and code are available at this https URL

- **What's New**: 장문 문서 이해(long-context modeling)에서 최근 진보를 이루어 다양한 NLP 응용 분야에서 복잡한 작업을 수행하도록 언어 모델(Language Models, LMs)의 성능이 향상되었습니다. 그러나 이러한 진보에도 불구하고, 다중 단계 추론(multi-hop reasoning) 및 노이즈가 있는 문맥에서 성능 저하를 보입니다. 이에 본 논문에서는 언어 모델이 추론 과정에서 각 주장에 대한 근거(attributions)를 제공하도록 유도하는 새로운 접근법인 'Reasoning with Attributions'을 제안합니다.

- **Technical Details**: 우리의 접근법은 다양한 독점 모델(proprietary models)과 오픈 소스 모델(open-source models)을 사용하여 세 가지 다중 단계 추론 데이터셋에서 실험을 통해 검증되었습니다. 우리는 또한 추론 기능을 향상시키기 위한 파인튜닝(fine-tuning) 방법을 탐구하고, 근거 주석 데이터셋(attribution-annotated dataset)과 특화된 훈련 전략(training strategy)을 제공합니다.

- **Performance Highlights**: 파인튜닝된 모델은 다중 단계 추론 벤치마크에서 경쟁력 있는 성능을 보이며, ChatGPT 및 Claude-instant와 같은 독점 언어 모델과 유사한 성능 수준을 달성했습니다.



### A Debiased Nearest Neighbors Framework for Multi-Label Text Classification (https://arxiv.org/abs/2408.03202)
- **What's New**: 이번 논문에서는 다중 라벨 텍스트 분류(MLTC)를 위한 새로운 DEbiased Nearest Neighbors (DENN) 프레임워크를 제안합니다. 이 방법은 기존의 기술들이 간과했던 두 가지 주요 편향인 임베딩 정렬 편향(embedding alignment bias)과 신뢰도 추정 편향(confidence estimation bias)을 완화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DENN 프레임워크는 두 가지 핵심 전략을 도입합니다. 첫째, 임베딩 정렬 편향을 해결하기 위해 디바이즈드 대조 학습(debiased contrastive learning) 전략을 사용하여 라벨 공존 정보를 효과적으로 향상시킵니다. 둘째, 신뢰도 추정 편향을 해소하기 위해 디바이즈드 신뢰도 추정(debiased confidence estimation) 전략을 제안하여 각 타겟 샘플에 대해 더욱 정확하고 적응적인 신뢰도를 제공합니다. 이를 통해 각 타겟 샘플의 맞춤형 라벨 공존 정보를 임베딩 공간에서 효과적으로 추출하고, 신뢰도를 동적으로 추정하여 적응적인 라벨 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 AAPD, RCV1-V2, Amazon-531, EUR-LEX57K 네 가지 공공 벤치마크 데이터셋에서 광범위한 실험을 통해 그 효과를 입증했습니다. 실험 결과, 제안된 DENN 프레임워크는 기존 방법들보다 우수한 성능을 보였으며, 최첨단(State-of-the-Art) 성능을 달성했습니다. 또한, 이 방법은 추가적인 파라미터를 도입하지 않습니다.



### Leveraging Parameter Efficient Training Methods for Low Resource Text Classification: A Case Study in Marath (https://arxiv.org/abs/2408.03172)
Comments:
          Accepted at I2CT 2024

- **What's New**: 이번 연구에서는 인도 저자세 언어(Indic low-resource language) Marathi를 위한 BERT 모델의 파라미터 효율적 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 방법을 분석하였습니다. 이 연구는 MahaSent, MahaHate, 그리고 MahaNews와 같은 주요 텍스트 분류 데이터셋을 사용하여 다양한 모노링구얼(Monolingual) 및 멀티링구얼(Multilingual) Marathi BERT 모델에 PEFT 방법을 적용하고 그 효과를 평가하였습니다.

- **Technical Details**: 이 연구는 LoRA(Low-Rank Adaptation of Large Language Models) 및 Adapter 방법을 사용하여 저자세 텍스트 분류를 수행하였습니다. 이러한 방법들은 모델의 대부분의 파라미터를 동결(freeze)한 채 소수의 파라미터만을 미세 조정(fine-tune)하여, 전체 미세 조정과 비슷한 성능을 유지하면서도 훈련 속도를 크게 향상시킵니다. 사용된 모델에는 MahaBERT, MahaAIBERT, MahaRoBERTa, Muril-BERT, IndicBERT, 그리고 Multilingual-cased BERT가 포함됩니다.

- **Performance Highlights**: PEFT 방법들은 전체 모델을 미세 조정하는 것과 비교하여 정확도를 유지하면서도 훈련 속도와 효율성을 크게 개선하였습니다. 이는 Marathi BERT 모델의 효과성을 입증하며, Marathi와 유사한 인도 언어들의 NLP 능력 발전을 위한 토대를 제공합니다.



### Conditioning LLMs with Emotion in Neural Machine Translation (https://arxiv.org/abs/2408.03150)
Comments:
          6 pages, In Proceedings of the 21st International Conference on Spoken Language Translation (IWSLT), Bangkok, Thailand, 2024

- **What's New**: 이번 연구에서는 감정 정보를 통합하여 번역 품질을 향상시키는 새로운 기계 번역(Machine Translation, MT) 파이프라인을 제안합니다. 음성 감정 인식(Speech Emotion Recognition, SER) 모델에서 추출한 감정 정보를 기존의 대형 언어 모델(Large Language Models, LLMs)에 통합해, 번역 품질을 높이는 방법을 제안했습니다. Libri-trans 데이터셋을 사용하여 다섯 개의 LLM을 미세 조정한 후, 가장 성능이 우수한 모델을 선택하여 다양한 감정 정보를 포함한 프롬프트로 훈련했습니다.

- **Technical Details**: 감정 정보를 프롬프트에 추가하여 번역을 향상시키는 방법을 탐구했습니다. 먼저, 영어-프랑스어 번역을 위해 다섯 개의 LLM (Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2, TowerBase-7B-v0.1, TowerInstruct-7B-v0.2, ALMA-7B-R)을 Libri-trans 데이터셋에서 미세 조정하고, 가장 성능이 좋은 모델을 선택했습니다. 그 후, 선택된 LLM을 다양한 감정 정보로 훈련했습니다. 감정 정보는 최신 SER 모델을 통해 오디오 녹음에서 추출된 감정 차원을 포함합니다.

- **Performance Highlights**: 감정 정보를 프롬프트에 통합한 결과, 특히 '흥분도(arousal)' 정보가 번역 품질에 큰 향상을 가져왔습니다. BLEU 및 COMET 점수가 눈에 띄게 개선되었음을 확인했습니다.



### Inference Optimizations for Large Language Models: Effects, Challenges, and Practical Considerations (https://arxiv.org/abs/2408.03130)
- **What's New**: 이번 논문 리뷰는 대형 언어 모델(LLM)을 효율적으로 압축하고 자원 소모를 줄이기 위한 최신 기법들을 탐구합니다. 여기에는 양자화(quantization), 가지치기(pruning), 지식 증류(knowledge distillation), 구조적 최적화(architectural optimizations) 등이 포함됩니다. 저자들은 이러한 방법들을 분류하여 최적화 지형을 개괄적으로 제시하고, 연구 방향을 보다 잘 이해하는 데 도움을 줍니다.

- **Technical Details**: 이 논문에서는 트랜스포머(transformer) 아키텍처 및 주요 메커니즘인 주의 메커니즘(attention mechanism)에 대해 논의합니다. 또한, 대형 언어 모델을 양자화하는 방법에 대해 상세히 설명합니다. 특히, 정수 양자화(integer quantization)을 중심으로 이야기를 풀어나가며, 이를 대칭 양자화와 비대칭 양자화로 나누어 설명합니다. 양자화의 경우, 모델의 특정 레이어, 요소, 비트 정밀도 등이 어떻게 영향을 받는지를 구체적으로 다룹니다.

- **Performance Highlights**: 양자화 기법을 활용한 성능 최적화 예로는 LLM.Int8()이 8비트 양자화를 도입하여 메모리 소모를 줄이는 방안을 제시했습니다. 또한, GTPQ와 AWQ 방식은 4비트 양자화를 통해 모델의 품질을 유지하며 효율성을 높이는 전략을 사용합니다. 다양한 양자화 기법들이 대형 언어 모델의 연산 속도와 자원 효율성을 크게 향상시킨 사례들을 다룹니다.



### Lisbon Computational Linguists at SemEval-2024 Task 2: Using A Mistral 7B Model and Data Augmentation (https://arxiv.org/abs/2408.03127)
Comments:
          8 pages, 1 figure, submitted and accepted into the "18th International Workshop on Semantic Evaluation (SemEval-2024)"

- **What's New**: 이번 논문은 SemEval-2024의 안전한 생의학적 자연언어추론(NLI4CT) 과제에 대한 접근 방식을 설명합니다. 과제는 임상시험 보고서(CTRs)에 대한 진술을 분류하는 것으로, open-source 대형 언어 모델(LLM)인 Mistral-7B를 사용했습니다. 우리는 NLI4CT 과제에 맞춘 프롬프트를 개발하고, 학습 데이터셋을 증강하여 양자화한 모델을 미세 조정했습니다. 실험 결과, 이 방법이 macro F1-score 측면에서 주목할 만한 결과를 낳을 수 있음을 보여줍니다.

- **Technical Details**: 가장 주목할 만한 점은 Mistral-7B-Instruct-v0.2 모델을 사용한 것입니다. 이 모델은 4비트로 양자화되었고, Low-Rank Adaptation (LoRA) 기법을 이용해 NLI4CT 과제에 맞춰 미세 조정되었습니다. 프롬프트 템플릿은 $task_description, $ctr_description, $statement_description, $option_description 등 4개의 샘플 독립적인 부분과 $primary_evidence, $secondary_evidence, $statement 등 3개의 샘플 의존적인 부분으로 구성됩니다. 625개의 프롬프트 조합을 개발 셋에서 평가하여 최상의 macro F1-score를 얻은 프롬프트를 선택하였습니다.

- **Performance Highlights**: 우리의 최종 제출물은 macro F1-score 0.80(리더보드 1위), 일관성 점수 0.72(15위), 충실도 점수 0.83(11위)를 기록했습니다. 분류 정확도에서는 우수했으나, 진술에 대한 변화에 대해 강건하지 못한 점이 한계로 드러났습니다.



### COMMENTATOR: A Code-mixed Multilingual Text Annotation Framework (https://arxiv.org/abs/2408.03125)
- **What's New**: NLP 커뮤니티가 다국어 문제에 점점 더 많은 관심을 기울이고 있는 가운데, 다국어 데이터셋을 효율적으로 처리하기 위한 견고한 주석 도구가 필요합니다. 본 논문에서는 코드 혼합 텍스트(Code-Mixed Text)를 주석할 수 있는 다국어 텍스트 주석 프레임워크인 COMMENTATOR를 소개합니다. 이 도구는 Hinglish 텍스트에 대한 토큰 수준 및 문장 수준의 언어 주석 작업에 있어 그 효율성을 입증하였습니다.

- **Technical Details**: Hinglish 텍스트의 예시와 같은 코드 혼합은 비공식 대화 및 소셜 미디어에서 흔히 발견됩니다. COMMENTATOR는 토큰 수준의 언어 식별(Language Identification), 품사 태깅(POS Tagging), 문장 수준의 매트릭스 언어 식별(Matrix Language Identification)과 같은 주석 작업을 지원합니다. 이 프레임워크는 간편한 네비게이션, 사용자 생산성 증대, 빠른 클라우드 또는 로컬 설치, 주석자 피드백 통합을 통한 반복적 개선, 간단한 관리 인터페이스, 병렬 주석 등의 기능을 갖추고 있습니다.

- **Performance Highlights**: COMMENTATOR는 기존의 최첨단(SOTA) 주석 도구보다 5배 빠른 주석 속도를 보여줍니다. 이러한 속도 향상은 더욱 발전된 코드 혼합 라이브러리를 통합함으로써 더욱 증대될 수 있습니다. COMMENTATOR는 Hinglish 외의 다른 언어 쌍도 지원할 수 있도록 확장될 수 있습니다.



### Evaluating the Translation Performance of Large Language Models Based on Euas-20 (https://arxiv.org/abs/2408.03119)
Comments:
          15 pages, 8 figures

- **What's New**: 최근 몇 년 동안, BERT와 GPT와 같은 대형 언어 모델(LLMs)이 자연어 처리 작업에서 획기적인 성과를 거두었습니다. 이러한 모델의 발전 덕분에 기계 번역(MT) 성능도 크게 향상되었습니다. 그러나 여전히 해결해야 할 많은 도전 과제들이 있습니다. 이를 평가하기 위해, 우리는 다양한 언어로 구성된 Euas-20 데이터셋을 구축하여 대형 언어 모델의 번역 성능을 검토하고자 했습니다.

- **Technical Details**: 본 논문에서는 Llama2, Falcon, Vicuna, Mistral 등의 LLM들뿐만 아니라 Bloom, Bloomz, Gemma 같은 다국어 LLM을 평가 대상으로 삼았습니다. 데이터 누출을 방지하고 정확한 결과를 얻기 위해 Euas-20이라는 대표적인 20개 언어로 구성된 데이터셋을 구축했습니다. 이 데이터셋은 다양한 서체와 언어 그룹을 포함하며, 다국어 및 다영역 훈련 데이터를 사용해 모델의 일반화 능력을 향상시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: LLM들은 영어 번역에서 우수한 성능을 보였으나, 다른 언어들에서는 성능 차이가 있었습니다. 특히 영어와 유사한 언어에서 뛰어난 성능을 보였으며, zero-resource 언어에도 어느 정도의 번역 능력을 입증했습니다. 그러나 번역 결과가 유창하게 나오면서도 사용자를 오도할 수 있는 문제점이 있습니다. 이는 LLM들이 학습 과정에서 접하지 못한 단어들에 대해 정확하게 이해하거나 처리할 수 없는 경우가 있기 때문입니다.



### Topic Modeling with Fine-tuning LLMs and Bag of Sentences (https://arxiv.org/abs/2408.03099)
Comments:
          This is the submitted journal version of enhanced with the novel fine-tuning part of "Efficient and Flexible Topic Modeling using Pretrained Embeddings and Bag of Sentences'' which appeared at the International Conference on Agents and Artificial Intelligence(ICAART) in 2024

- **What's New**: 이번 논문은 기존의 주제 모델링 기법을 개선하기 위해 대형 언어 모델(LLM: Large Language Models)을 미세 조정(Fine-Tuning)하는 새로운 방법인 FT-Topic을 도입합니다. 이 방법을 통해 비지도 학습(Unsupervised Learning) 방식으로 LLM을 최적화하여 주제 모델링에서 더 우수한 성능을 제공합니다.

- **Technical Details**: FT-Topic 방법론은 크게 두 단계로 구성됩니다. 첫 번째 단계에서는 휴리스틱(Heuristic) 방법을 사용해 같은 주제에 속할 가능성이 높은 문장 그룹의 쌍을 식별합니다. 두 번째 단계에서는 잘못 라벨링된 쌍을 제거하여 정확도를 높입니다. 이렇게 생성된 데이터셋을 사용해 LLM 인코더를 미세 조정하게 되며, 이는 임베딩(Embeddings)을 사용하는 모든 주제 모델링 접근법에 활용될 수 있습니다.

- **Performance Highlights**: 새로운 주제 모델링 방법인 SenClu를 통해 FT-Topic의 효과를 보여줍니다. SenClu는 기대-최대화 알고리즘(Expectation-Maximization Algorithm)을 통해 빠른 추론 속도를 제공하며, 문장 그룹을 단일 주제에 할당하는 하드 어사인먼트(Hard Assignments) 방식을 사용합니다. 또한 사용자가 주제-문서 분포에 대한 사전 지식을 인코딩할 수 있는 구성을 제공합니다. 이 방법론은 기존의 주제 모델링 기법보다 더 높은 정확성과 유연성을 제공합니다.



### 500xCompressor: Generalized Prompt Compression for Large Language Models (https://arxiv.org/abs/2408.03094)
- **What's New**: 이번 연구에서는 긴 자연어 문맥을 최소 한 개의 특수 토큰으로 압축하는 새로운 방법인 '500xCompressor'를 제안합니다. 이 방법은 약 0.3%의 추가 파라미터만을 도입하며, 6배에서 최대 480배까지의 압축 비율을 달성합니다. 원래의 대형 언어 모델(LLM)을 미세 조정 없이 사용할 수 있는 이 방법은 다양한 질문에 답할 수 있는 범용성을 지닙니다.

- **Technical Details**: 500xCompressor는 Arxiv Corpus를 사전 학습 데이터로 사용했으며, ArxivQA 데이터셋으로 미세 조정되었습니다. 모델은 인코더와 디코더로 구성되며, 인코더는 학습 가능한 LoRA 파라미터를 사용하고 디코더는 원래의 고정된 LLM 파라미터를 사용합니다. 인코더는 입력 텍스트를 압축된 토큰으로 인코딩하고, 디코더는 이 압축된 토큰을 사용하여 원래 텍스트를 재생성합니다.

- **Performance Highlights**: 500xCompressor는 압축된 프롬프트를 사용할 때 원래 모델의 성능의 62.26-72.89%를 유지합니다. 최대 480배의 압축 비율을 달성하며, 이는 이전의 연구들이 달성했던 50배 이하의 압축 비율을 크게 초과합니다. 평가 텍스트는 원래 훈련 데이터셋과 겹치지 않는 새로운 도메인별 텍스트를 사용하여 데이터 유출 문제를 방지합니다.



### Extend Model Merging from Fine-Tuned to Pre-Trained Large Language Models via Weight Disentanglemen (https://arxiv.org/abs/2408.03092)
Comments:
          17 pages

- **What's New**: 이번 연구는 기존의 Fine-Tuned (FT) Language Models (LLMs) 통합 방법을 Pre-Trained (PT) LLMs에도 적용할 수 있는 새로운 접근법을 제시합니다. 논문에서는 PT와 FT 모델 간의 다양한 파라미터 변화를 극복하여 LLMs 통합의 범위를 확장하는 WeIght DisENtanglement (WIDEN) 기법을 제안합니다.

- **Technical Details**: 현재 방법이 PT LLMs를 통합하는 데 어려움을 겪는 이유는 파라미터 변화의 범위가 다르기 때문입니다. WIDEN은 모델 가중치를 크기(magnitude)와 방향(direction) 성분으로 분해한 후, 각 성분의 기여도를 고려하여 적응형 통합을 수행합니다. 이는 기존처럼 매뉴얼로 모델 중요도를 지정할 필요 없이 자동으로 모델 기여도를 계산합니다.

- **Performance Highlights**: 실험 결과, WIDEN은 Qwen1.5-Chat (지시 따르기 기능을 가진 FT LLM)와 Sailor (다국어 능력을 가진 PT LLM)를 7B와 14B 모델 스케일에서 통합하여 놀라운 결과를 도출했습니다. 예를 들어, 현재 방법으로는 Sailor의 다국어 능력이 통합되지 않거나, 지시 따르기 기능이 유지되지 않는 반면, WIDEN은 두 능력을 모두 유지하고 향상시켰습니다. 또한 여러 13B FT LLMs (수학적 추론, 코드 생성 능력 등)를 균형 있게 통합해 성능을 입증했습니다.



### Enhancing Complex Causality Extraction via Improved Subtask Interaction and Knowledge Fusion (https://arxiv.org/abs/2408.03079)
Comments:
          NLPCC 2024 Oral

- **What's New**: Event Causality Extraction (ECE)에서 새로운 통합 프레임워크 UniCE가 제안되었습니다. UniCE는 두 개의 주요 모듈, 즉 이벤트 모듈과 관계 모듈로 구성되어 있으며 각 모듈은 여러 레이어로 구성되어 있습니다. 이 프레임워크는 복잡한 인과 관계 추출, 서브태스크 상호작용(Subtask Interaction) 및 지식 융합(Knowledge Fusion)의 문제를 동시에 해결함으로써 성능을 대폭 향상시켜줍니다.

- **Technical Details**: UniCE는 하나의 입력 문장에서 첫 번째 레이어가 외부 지식 그래프(Knowledge Graph, KG)를 사용하여 초기 배경 그래프를 구성하고, 이후 레이어들이 이 배경 그래프와 사전 학습된 언어 모델(Pretrained Language Models, PLMs)을 활용하여 점진적으로 이벤트와 인과 관계를 추출해내는 구조로 되어 있습니다. 각 레이어에서 이벤트 모듈은 시퀀스 레이블링 디코더를 사용하며, 이는 문장에서 각 토큰의 표현을 업데이트합니다. 그런 다음, 관계 모듈이 그래프 신경망(GNNs)을 활용하여 인과 관계를 판별합니다. 지식 융합 메커니즘은 이벤트와 KG를 동적으로 연결하여, 문장에서 추출한 각 요소와 관련된 지식을 효과적으로 통합합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(EventStoryLine, SCIFI, Causal-TimeBank)에서 UniCE는 최첨단 성능(State-of-the-Art, SOTA)을 기록하며, ChatGPT를 최소 30%의 F1 점수 차이로 앞섰습니다. 또한, 서브태스크 상호작용 및 지식 융합 메커니즘이 성능 향상에 크게 기여하는 것을 확인했습니다. 추가 실험에서는 UniCE가 ChatGPT의 성능을 문맥학습(In-Context Learning)을 통해 효과적으로 향상시킬 수 있음을 보여주었습니다.



### Towards an Analysis of Discourse and Interactional Pragmatic Reasoning Capabilities of Large Language Models (https://arxiv.org/abs/2408.03074)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)에서 시험된 실용적 능력(pragmatic abilities)과 이러한 시험이 어떻게 수행되었는지에 대한 개요를 제공합니다. 연구자는 다양하고 이질적인 실용적 현상 및 방법을 두 개의 하위 도메인인 담화 실용적 현상(discourse pragmatics)과 상호작용 실용적 현상(interactional pragmatics)으로 분류해서 제시하고 있습니다.

- **Technical Details**: 이 논문은 담화 실용적 현상과 상호작용 실용적 현상으로 실용적 능력을 나누어 설명합니다. 담화 실용적 현상은 전제(presupposition), 암시(implicature), 은유(figurative speech) 등 텍스트 연속성(text coherence)과 관련된 공식적 추론 과정을 포함합니다. 반면 상호작용 실용적 현상은 예의(politeness), 차례 교환(turn-taking), 수정(repair) 등 대화에서 직접적인 상호작용을 다룹니다. 이러한 상호작용 실용 현상은 종종 대화 분석(conversation analysis)과 관련됩니다.

- **Performance Highlights**: 대형 언어 모델(LLMs)들이 담화 실용적 현상을 이해하는 데는 어느 정도 성공을 거두었으나, 상호작용 실용적 현상을 다루는 데는 제한적이라는 결과가 나왔습니다. 예를 들어, 대화 내 암시(conversational implicatures)를 이해하는 모델로는 OPT, T5, GPT-4가 있습니다. 또한, GPT-3와 GPT-4는 다른 페르소나를 시뮬레이션하는 능력을 보여주었지만, 전반적으로 상호작용적 실용적 현상을 측정하고 분석할 도구가 부족하다는 한계가 지적되었습니다.



### Probing structural constraints of negation in Pretrained Language Models (https://arxiv.org/abs/2408.03070)
- **What's New**: 최근의 연구들에 따르면 사전 학습된 언어 모델들(PLMs)이 부정어의 의미적 영향(semantic impact)을 어떻게 인코딩하는지에 대해 상충되는 결과들이 나오고 있습니다. 이 논문은 특히 부정적 극성 항목(NPI) 인가(licensing) 현상을 통해 PLMs가 부정을 어떻게 인코딩하고 그 형식적인 영향을 어떻게 파악하는지 조사합니다.

- **Technical Details**: 이 논문은 프로브(probes)를 사용하여 문장에서 부정어의 존재 여부와 인접한 마스킹된 극성 항목(masked polarity item)의 극성을 인코딩하는지 확인합니다. 그 결과, 부정어의 범위 내에 있는 토큰의 문맥적 표현이 범위 외부에 있는 토큰들에 비해 'not'의 존재를 보다 잘 예측할 수 있었고, 'not'에 의해 인가된 극성 항목의 올바른 극성을 예측하는 데에도 우수함을 보였습니다. 이 차이는 PLM마다 다를 수 있습니다.

- **Performance Highlights**: 가장 중요한 점은, 거리와 상관없이 이러한 경향이 유지된다는 것입니다. 이는 모델의 임베딩이 부정 범위의 개념을 반영하고 NPI 인가에 대한 부정의 영향을 인코딩한다는 것을 시사합니다. 그러나 추가 실험에서는 같은 구문 내에서 토큰의 문맥적 표현을 사용할 때 다른 어휘 항목의 존재도 더 잘 포착된다는 것이 밝혀져, PLMs이 단순히 구문의 더 일반적인 개념을 포착하는 것일 수 있음을 시사합니다.



### Analysis of Argument Structure Constructions in a Deep Recurrent Language Mod (https://arxiv.org/abs/2408.03062)
- **What's New**: 이 연구는 Argument Structure Constructions (ASCs) 즉, 전이적, 이중전이적, 야기된 운동, 결과문 집합소에 대한 뇌의 처리 방식을 탐구했습니다. Long Short-Term Memory (LSTM) 네트워크를 사용하여 이들 구문을 분류하고 분석했습니다.

- **Technical Details**: GPT-4를 사용해 2000개의 문장을 생성하여 LSTM 네트워크에 학습시키고, 이때 Multidimensional Scaling (MDS)와 t-Distributed Stochastic Neighbor Embedding (t-SNE)를 활용해 모델의 내부 활성화를 시각화했습니다. 또한 Generalized Discrimination Value (GDV)를 계산해 클러스터링 품질을 평가했습니다.

- **Performance Highlights**: 결과적으로, LSTM 모델은 네 가지 ASC 유형을 효과적으로 분류하였으며, 마지막 은닉 계층에서 가장 뚜렷한 클러스터링이 관찰되었습니다. 이는 단순한 재발 신경망도 다양한 문형을 분화할 수 있음을 시사합니다.



### L3iTC at the FinLLM Challenge Task: Quantization for Financial Text Classification & Summarization (https://arxiv.org/abs/2408.03033)
Comments:
          Joint Workshop of the 8th Financial Technology and Natural Language Processing (FinNLP) and the 1st Agent AI for Scenario Planning (AgentScen), 2024

- **What's New**: 이번 뉴스레터에서는 FinLLM Challenge Task 2024에 참여한 팀 L3iTC의 성과를 소개합니다. 이 팀은 금융 텍스트 분류 및 금융 텍스트 요약 두 가지 주요 과제를 다뤘습니다. 이 팀은 대형 언어 모델(LLMs)을 4비트 양자화와 LoRA를 이용해 미세 조정하는 방식으로 모델을 최적화했습니다. 이를 통해 GPU 메모리가 적은 환경에서도 모델을 실행할 수 있었으며, 금융 텍스트 분류에서 F1 점수 0.7543을 기록하며 3위를 차지했고, 금융 요약에서는 6위를 기록했습니다.

- **Technical Details**: L3iTC 팀은 금융 텍스트 분류와 요약을 위해 다양한 LLMs를 사용했습니다. Mistral-7B-Instruct-v0.2 및 Mistral-7B-Instruct-v0.3, 그리고 Meta-Llama-3-8B-Instruct 같은 모델들을 선택하여 4비트 양자화(4-bit quantization)와 LoRA(Low-Rank Adaptation) 기법을 이용해 미세 조정했습니다. Fine-tuning 과정에서 q_proj, k_proj, v_proj 등의 특정 모듈 내 훈련 파라미터를 설정하고, 드롭아웃 드롭아웃 비율을 0.05로 설정했으며, 학습률은 5×10^{-5}, 배치 크기는 4로 설정했습니다. 모델은 총 2,000단계를 거쳐 미세 조정되었습니다.

- **Performance Highlights**: Task 1(금융 텍스트 분류)의 초기 성능에서 가장 좋은 모델은 Mistral-7B-Inst-v0.2로, 정확도 54%와 F1 점수 0.39를 기록했습니다. 그러나 미세 조정 후, 최상의 성능을 보인 모델인 FT-Clas-Mistral-7B-Inst-v0.3가 정확도 78%와 F1 점수 0.78을 달성했습니다. Task 2(금융 텍스트 요약)에서는 눈에 띄는 성능 향상이 없었지만, 전반적으로 L3iTC의 접근 방식은 효율성을 증대시키고 GPU 메모리 요구사항을 낮추는 등 많은 성과를 거두었습니다.



### Fact Finder -- Enhancing Domain Expertise of Large Language Models by Incorporating Knowledge Graphs (https://arxiv.org/abs/2408.03010)
Comments:
          10 pages, 7 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 질문에 대답하는 능력을 보여주었지만, 도메인별 지식의 한계로 인해 정확성이 떨어지는 문제가 있었습니다. 이를 해결하기 위해 지식 그래프(KG)를 결합한 하이브리드 시스템을 제안합니다. 이 시스템은 KG 기반 검색 방식을 사용해 사실적 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 우리는 의료 지식 그래프(KG)를 사용하여 다음과 같은 단계를 포함한 방법론을 제시합니다: (1) 데이터 전처리, (2) 사이퍼(Cypher) 쿼리 생성, (3) 사이퍼 쿼리 처리, (4) KG 검색, (5) LLM을 통한 응답 생성. 우리의 시스템은 PrimeKG을 사용하여 20개의 고품질 리소스를 통합한 17,080개의 질병과 4,050,249개의 관계를 설명합니다. 또한, 69개의 텍스트-사이퍼 쿼리 쌍으로 구성된 데이터셋을 사용해 평가를 수행했습니다.

- **Performance Highlights**: 테스트 결과, 올바른 KG 노드를 검색하는 정확도가 78%에 달하는 것으로 나타났습니다. 이 하이브리드 시스템은 정확성과 완전성에서 단독 LLM을 능가하며, 'LLM-as-a-Judge' 평가 방법에 의해 검증되었습니다. 또한, 직관적인 검색 인터페이스와 몇 초 내에 정확한 응답을 제공하는 능력을 갖추고 있어, 시간과 정확성이 중요한 연구 환경에 적합합니다.



### Empathy Level Alignment via Reinforcement Learning for Empathetic Response Generation (https://arxiv.org/abs/2408.02976)
- **What's New**: 이번 연구는 공감적 응답 생성을 위해 강화 학습을 사용하는 새로운 프레임워크인 EmpRL(Empathetic Response Generation using Reinforcement Learning)을 제안한 것입니다.

- **Technical Details**: EmpRL은 사전 학습된 T5 모델을 생성기로 사용하며, 공감 보상 함수(empathy reward function)를 설계해 강화 학습을 통해 예상 보상을 최대화하여 공감적 응답을 생성합니다. 보상 함수는 세 가지 공감 커뮤니케이션 메커니즘(empathy communication mechanisms)—정서적 반응(emotional reaction), 해석(interpretation), 탐색(exploration)—을 포함합니다. Proximal Policy Optimization (PPO) 알고리즘을 사용해 정책을 추가로 훈련하여 응답의 일관성을 유지합니다.

- **Performance Highlights**: EmpRL 프레임워크는 자동 및 수동 평가 모두에서 생성된 응답의 공감 수준 유사성을 향상시키고, 감정적 및 인지적 공감을 모두 포함하는 공감적 응답을 생성하는 데 우수한 성능을 보였습니다.



### EC-Guide: A Comprehensive E-Commerce Guide for Instruction Tuning and Quantization (https://arxiv.org/abs/2408.02970)
- **What's New**: 본 연구에서는 EC-Guide라는 새로운 e-commerce(이커머스)용 큰 언어 모델(Large Language Models, LLMs) 튜닝과 양자화(quantization)을 위한 종합 가이드를 개발했습니다. 이 접근법은 Amazon KDD Cup'24에서 Track 2에서 2위, Track 5에서 5위를 차지했습니다. EC-Guide는 Chain-of-Thought(COT)를 사용하여 산술 성능을 향상시켰습니다.

- **Technical Details**: 연구팀은 기존 데이터셋을 활용하고 ChatGPT를 사용해 데이터 생성을 통해 EC-Guide라는 튜닝 데이터셋을 구성했습니다. 데이터셋에는 57개의 과제와 약 20,000개의 질문이 포함되어 있습니다. 트랙 2와 트랙 5를 위해 로우 리소스(4× NVIDIA T4 GPU)에 적합하게 양자화 기법을 사용했습니다. 특히 Chain-of-Thought(COT) 기법을 사용해 추가 학습 없이 모델의 추론 성능을 강화했습니다.

- **Performance Highlights**: EC-Guide는 Amazon KDD Cup'24에서 Track 2에서 2위를, Track 5에서 5위를 기록했습니다. 특히 Chain-of-Thought(COT) 기법을 통해 모델의 산술 성능이 크게 향상되었습니다.



### Accuracy and Consistency of LLMs in the Registered Dietitian Exam: The Impact of Prompt Engineering and Knowledge Retrieva (https://arxiv.org/abs/2408.02964)
- **What's New**: 이 논문에서는 최첨단 대규모 언어 모델(Large Language Models, LLMs)인 GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro를 활용하여 영양 및 다이어트 관련 질문에 대한 정확성과 일관성을 평가하기 위해 Registered Dietitian (RD) 시험을 도입했습니다. 이는 기존 연구와 달리 표준화된 평가를 통해 LLMs의 성능을 객관적으로 측정하고자 합니다.

- **Technical Details**: 논문에서는 RD 시험의 다양한 영양 주제와 능력 수준을 포함하는 1050개의 질문을 사용했습니다. 평가에는 Zero-Shot (ZS), Chain of Thought (CoT), Chain of Thought with Self Consistency (CoT-SC), Retrieval Augmented Prompting (RAP) 기법이 포함되었습니다. 이 기법들은 모델의 정확성과 일관성에 어떤 영향을 미치는지 분석했습니다.

- **Performance Highlights**: 평가 결과, 모든 접근법이 88% 이상의 점수를 기록한 가운데, GPT-4o가 CoT-SC 기법을 사용할 때 가장 높은 점수(91%~95%)를 기록했습니다. Gemini 1.5 Pro는 ZS 방식에서 가장 일관적인 성능을 보였으며, GPT-4o와 Claude 3.5에서는 CoT와 CoT-SC가 정확성을 향상시켰습니다. RAP 기법은 특히 GPT-4o에서 전문가 수준의 질문에 대한 답변 정확성을 높이는 데 효과적이었습니다. 총괄적으로, 적절한 LLM 및 프롬프트 기법을 선택하면 영양 및 다이어트 챗봇의 오류와 잠재적 위험을 줄일 수 있습니다.



### Are Female Carpenters like Blue Bananas? A Corpus Investigation of Occupation Gender Typicality (https://arxiv.org/abs/2408.02948)
- **What's New**: 이번 연구에서는 직업과 성별 언급 패턴이 바나나의 색깔 언급 패턴과 유사한지 여부를 정보 이론적 기법과 코퍼스 통계 분석을 통해 조사했습니다. 대규모 코퍼스 두 가지에서 직업과 성별 언급이 바나나와 색깔처럼 나타난다는 강력한 증거는 찾지 못했습니다. 대신, 여성 주도 직업에서 성별 언급이 더 자주 발생하며, 이는 여성이 주로 종사하는 직업이 남성 주도 직업보다 더 '성별화'된 것으로 보인다는 점을 발견했습니다.

- **Technical Details**: 본 연구에서는 정보 이론적 접근법과 대규모 텍스트 코퍼스를 분석하여 직업과 성별 언급의 두 가지 가설을 테스트했습니다. 첫째, 성별 언급이 직업 성별과 무관하다는 가설을 배제했습니다. 둘째, 여성이 주로 종사하는 직업이 성별 언급과 더 관련이 있음을 발견했습니다. 이를 위해 직업의 성별 결정성을 조건부 엔트로피를 기반으로 한 정보 이론적 양을 정의하여 테스트했습니다.

- **Performance Highlights**: 주요 성과로는 여성이 주로 종사하는 직업에서 성별 언급이 더 빈번하다는 점을 확인했습니다. 이는 Reddit과 Wikipedia 코퍼스에서 반복적으로 나타났으며, '푸시쉬프트.io Reddit' 데이터셋에서 더 강하게 나타났습니다. 또한, 성별 언급이 직업의 성별 놀라움, 즉 비전형성,과는 약한 상관관계를 보였으며 이는 '푸른 바나나' 가설을 지지하지 않는다는 결론을 도출했습니다.



### Self-Supervised Learning for Multi-Channel Neural Transducer (https://arxiv.org/abs/2408.02945)
- **What's New**: 자기 지도 학습(self-supervised learning)을 사용하여 다채널(end-to-end) 자동 음성 인식(ASR) 모델을 wav2vec 2.0 프레임워크에 기반하여 개발했습니다. 특히 다채널 신경 트랜스듀서(neural transducer)에 중점을 두어, 다양한 특징 양자화(feature quantization) 방법을 비교하였습니다.

- **Technical Details**: 다채널 신경 트랜스듀서는 채널별(channel-wise) 및 크로스 채널(cross-channel) 자기 주의(self-attention) 레이어를 사용하여 채널 간의 문맥적 관계를 학습합니다. 세 가지 특징 양자화 방법인 합동 양자화(joint quantization), 특징별 양자화(feature-wise quantization), 채널별 양자화(channel-wise quantization)를 비교하였습니다. 예비 훈련(pre-training)과 미세 조정(fine-tuning)은 CHiME-4와 내부 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 특징별 양자화 방법이 가장 효과적이라는 결론을 도출했습니다. 내부 데이터셋에서 사전 훈련을 하지 않은 모델에 비해 문자 오류율이 66% 상대적으로 감소하였고, CHiME-4 데이터셋에서는 4.2% 감소하였습니다.



### Intermediate direct preference optimization (https://arxiv.org/abs/2408.02923)
- **What's New**: 우리는 대규모 언어 모델(LLMs)을 미세 조정하기 위해 중간 직접 선호 최적화(DPO) 방법을 제안했습니다. 기존의 DPO 방법은 최종 층의 로그잇(logits)을 사용하여 DPO 손실을 계산합니다. 그러나 우리의 중간 DPO 접근법에서는 선택한 중간 층에서 DPO 손실을 계산하고 이를 평균하여 중간 DPO 손실을 얻습니다. 특히, 32층 SFT 모델의 22번째 층에서 계산된 중간 DPO 손실을 사용한 모델은 기존 DPO 및 SFT 모델에 비해 각각 52.5% 및 67.5%의 승률을 기록했습니다.

- **Technical Details**: 기존 DPO 모델은 감독된 미세 조정(SFT) 모델을 최종 층의 로그잇을 사용하여 DPO 손실을 계산함으로써 미세 조정 됩니다. 중간 DPO 방식에서는 K개의 선택된 중간 층에서 DPO 손실을 계산하고, 이를 평균하여 중간 DPO 손실로 만듭니다. 최종 손실은 DPO 및 중간 DPO 손실의 가중 합으로 계산되며, 추론 시, 중간 DPO 모델은 최종 층 로그잇을 사용하여 디코딩합니다. 실험에서 사용된 데이터셋은 ultrafeedback 데이터셋이며, GPT-4로 평가되었습니다.

- **Performance Highlights**: 중간 DPO 손실이 32층 SFT 모델의 22번째 층에서 계산된 경우, 중간 DPO 모델은 기존 DPO 모델에 비해 52.5%, SFT 모델에 비해 67.5%의 승률을 기록했습니다. 이는 제안된 방법이 더 효과적임을 보여줍니다. 또한, 중간 층 선택 위치, 층 수, 성능 간의 관계도 분석되었습니다. 중간 DPO 손실 계산 층이 출력 층에 가까울수록 성능이 향상되며, 여러 층에서 손실을 계산하는 방법이 하나의 층에서 계산하는 방법보다 더 나은 성능을 보였습니다.



### Data Checklist: On Unit-Testing Datasets with Usable Information (https://arxiv.org/abs/2408.02919)
Comments:
          17 pages, 4 figures. COLM 2024

- **What's New**: LLM(대형 언어 모델)의 동작을 이해하는데 유용한 모델 체크리스트(체크리스트) 개념을 데이터셋에 적용한 새로운 접근법을 제안합니다.

- **Technical Details**: 이 논문에서는 다양한 데이터셋의 유닛 테스트를 수행하기 위해 𝒱-정보문헌에 기반한 분류 체계를 제안합니다. 이를 데이터 체크리스트(Data Checklist)라고 부르며, SNLI와 같은 잘 알려진 데이터셋에서 이미 알려진 아티팩트를 회복하는 데 성공했고, LLM 정렬을 위한 선호도 데이터셋에서 이전에 알지 못한 아티팩트를 발견했습니다. 이 데이터 체크리스트는 효율성과 데이터 효율성을 개선하기 위한 데이터 필터링에도 사용되었습니다.

- **Performance Highlights**: 이 접근법은 데이터셋 아티팩트를 체계적으로 분석할 수 있게 하였고, 특히 선호도 정렬 데이터셋에서 발견된 아티팩트를 통해 필터링을 적용하여 정렬된 LLM 성능을 향상시켰습니다.



### Leveraging Inter-Chunk Interactions for Enhanced Retrieval in Large Language Model-Based Question Answering (https://arxiv.org/abs/2408.02907)
- **What's New**: 새로운 retrieval framework IIER를 소개합니다. IIER는 Inter-chunk Interactions를 통해 다양한 문서 조각(chunks) 간의 내부 연결을 포착하여 Retrieval을 개선합니다. 기존 연구들이 개별 문서 문단을 독립적으로 처리하면서 발생하는 맥락 부족과 애매한 참조 문제를 해결하고자 설계되었습니다.

- **Technical Details**: IIER는 문서 조각들 사이의 구조적, 키워드, 의미적 상호작용 등을 고려하여 Chunk-Interaction Graph (CIG)를 구성합니다. 이 CIG를 바탕으로 그래프 기반 evidence chain retriever를 설계하여, 각 질문에 대해 관련 문서 조각을 찾아내고, 반복적으로 검색을 통해 증거 체인을 구축합니다. 이 방법은 문서의 로컬 문맥을 넘어서는 추가적인 컨텍스트를 제공하여, 고도의 추론과 정답 생성을 도와줍니다.

- **Performance Highlights**: 네 개의 데이터셋에 걸친 광범위한 실험 결과, IIER가 강력한 baseline 모델들을 능가하는 성능을 보였습니다. 이는 IIER의 Retrieval 및 추론 능력 향상 효과를 입증하는 결과입니다.



### SETN: Stock Embedding Enhanced with Textual and Network Information (https://arxiv.org/abs/2408.02899)
- **What's New**: 이번 연구에서는 텍스트와 네트워크 정보를 강화하여 주식 임베딩(stock embedding)을 수행하는 SETN(Stock Embedding enhanced with Textual and Network information) 모델을 제안합니다. 이 모델은 금융 도메인에 적합하도록 사전 훈련된 트랜스포머(transformer) 기반 모델과 그래프 신경망(Graph Neural Network, GNN) 모델을 결합하여 주식 정보를 학습합니다. 해당 모델은 관련 업종 정보 추출 작업에서 높은 성능을 보였습니다.

- **Technical Details**: SETN은 네 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 목표 주식을 중심으로 서브그래프를 추출하고, 두 번째 단계에서는 도메인 적응이 완료된 트랜스포머 기반 모델을 사용하여 텍스트 정보를 임베딩합니다. 세 번째 단계에서는 GNN 모델을 통해 네트워크 정보를 캡처하고, 마지막 단계에서는 목표 주식의 임베딩을 활용하여 섹터 및 산업 분류기를 훈련시킵니다. BERT 및 RoBERTa 모델을 사용하여 텍스트 정보를 처리하며, 금융 도메인에 특화된 데이터셋으로 사전 훈련을 수행합니다.

- **Performance Highlights**: 제안된 SETN 모델은 일본 주식 시장 데이터를 통해 평가되었으며, 기존 모델에 비해 관련 회사 정보 추출 작업에서 뛰어난 성능을 보였습니다. 특히, 주제별 펀드 생성 작업에서 SETN 모델을 사용한 임베딩은 베이스라인 모델을 능가하는 결과를 나타냈습니다. 또한, 다양한 그래프 유형과 학습 구조를 비교 분석한 결과, 방향성 그래프를 사용하는 것이 더 높은 성능을 보였으며, 트랜스포머와 GNN 모델의 결합 학습이 성능 향상에 기여함을 확인할 수 있었습니다.



### A Framework for Fine-Tuning LLMs using Heterogeneous Feedback (https://arxiv.org/abs/2408.02861)
Comments:
          7 pages, 1 figure

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)을 이종 피드백(heterogeneous feedback) 데이터로 파인튜닝(fine-tuning)하는 프레임워크를 제안합니다. 다양한 형식의 피드백 데이터를 단일 감독 형태로 통합하고, 고품질 및 다양한 하위 집합을 추출하여 모델 성능을 향상시키는 방식을 사용합니다.

- **Technical Details**: 이 프레임워크는 이질적인 감독 데이터(heterogeneous supervision data)를 단일 형식으로 변환합니다. 변환된 데이터셋은 고품질 및 다양성을 기준으로 필터링 되며, 이는 SFT(Supervised Fine-Tuning)와 RLHF(Reinforcement Learning from Human Feedback) 단계를 통해 사용됩니다. 데이터 필터링은 프롬프트의 임베딩(semantic embeddings)을 생성하고, 이를 기반으로 클러스터링(k-means clustering)하여 이루어집니다. 이 과정은 다양한 데이터셋을 결합하여 보다 포괄적인 인간의 선호도를 반영합니다.

- **Performance Highlights**: 제안된 프레임워크는 여러 영역에서 모델의 성능을 동시에 향상시켰습니다. 특히, 명령어 따르기(instruction following)와 편향(bias) 감소에 효과적이었습니다. 고품질 및 다양한 데이터 하위 집합을 사용함으로써 전체 데이터셋보다 더 높은 성능을 보여주었습니다.



### Examining Gender and Power on Wikipedia Through Face and Politeness (https://arxiv.org/abs/2408.02798)
- **What's New**: 이 연구는 사회언어학 이론의 두 가지 중요한 개념인 face acts와 politeness를 결합하여 담론을 분석하는 새로운 프레임워크를 제안합니다. 특히, 현재까지는 리소스가 부족했던 face acts를 다루기 위해 Wikipedia 대화 페이지(Wikipedia Talk Pages)를 주석하여 새로운 말뭉치를 생성하고 이를 토대로 face act 태거(tagger)를 훈련했습니다. 이를 통해 Wikipedia 편집자 간의 성별과 권력에 따른 face와 politeness의 상호작용을 분석했습니다.

- **Technical Details**: Brown과 Levinson의 politeness 이론은 'face' 개념을 기반으로 하며, 이는 긍정적 얼굴(positive face)과 부정적 얼굴(negative face)로 나눕니다. 이러한 이론을 바탕으로, 연구팀은 Wikipedia Talk Pages를 주석하여 face act 데이터를 생성하고 이를 토대로 face act 태거(tagger)를 훈련했습니다. 이 모델을 사용하여 약 130만 개의 Wikipedia 대화 페이지 문장을 분석했습니다.

- **Performance Highlights**: 연구 결과, 여성 Wikipedia 편집자들이 남성보다 더 예의를 지키는 경향이 있다는 이전 연구 결과를 확인할 수 있었습니다. 더욱 흥미로운 점은, 관리 권한이 있는 편집자들 사이에서는 이러한 성별 차이가 거의 사라진다는 것입니다.



### LLM economicus? Mapping the Behavioral Biases of LLMs via Utility Theory (https://arxiv.org/abs/2408.02784)
Comments:
          Accepted to COLM 2024

- **What's New**: 최근 arXiv에 발표된 연구는 대형 언어 모델(LLMs)에서도 인간의 경제적 편향, 즉 비합리적 경제적 행동이 나타나는지 분석하였습니다. 특히 LLM들이 손실 회피, 앵커링, 프레이밍과 같은 행태적 편향을 얼마나 가지고 있는지를 파악하는 것이 이 연구의 주요 목적입니다. 이를 통해 LLM이 인간의 의사결정을 지원하는 도구로서 활용될 가능성을 평가했습니다.

- **Technical Details**: 이 연구는 현대 경제 이론의 중심인 효용 이론(Utility Theory)을 사용하여 LLM의 경제적 편향을 평가하는 방법을 제안합니다. 연구진은 효용 함수를 통해 인간과 LLM 간의 경제적 행동을 양적으로 비교하였습니다. 이를 위해 공개 소스와 비공개 소스 LLM들을 여러 기준에서 분석하고, 다양한 실험 경제학 게임을 통해 그들의 행태적 편향을 측정했습니다. 특히 불공정 회피(Inequity Aversion), 손실 회피(Loss Aversion), 위험 회피(Risk Aversion), 시간 할인(Time Discounting) 등을 중점적으로 다뤘습니다.

- **Performance Highlights**: 연구 결과 LLM의 경제적 행동은 완전히 인간과 같지도, 완전히 경제적이지도 않은 것으로 나타났습니다. 예를 들어, 대부분의 LLM은 자기 자신에 대해 약한 불공정 회피를 보였으나 타인에 대해서는 강한 불공정 회피를 보였습니다. 또한, 손실 회피는 인간보다 약했고, 시간 할인은 인간보다 강했습니다. 다만, LLM은 설정에 따른 일관성을 유지하는 데 어려움을 겪었으며, 프롬프팅 기술(chain-of-thought, few-shot prompting)과 같은 개입 방법이 예상치 않은 결과를 초래하기도 했습니다. 이러한 연구 결과는 LLM의 경제적 편향을 개선하는 데 중요한 로드맵을 제시하며, 더욱 인간과 유사한 LLM 개발의 목표를 설정하는 데 기여할 것입니다.



### LLaVA-OneVision: Easy Visual Task Transfer (https://arxiv.org/abs/2408.03326)
Comments:
          Project Homepage: this https URL

- **What's New**: LLaVA-OneVision은 LLaVA-NeXT 블로그 시리즈에서 얻어진 통찰을 바탕으로 개발된 대형 멀티모달 모델(LMM)입니다. 특히 단일 이미지, 다중 이미지, 비디오 시나리오 등 세 가지 중요한 컴퓨터 비전 시나리오에서 성능 경계를 확장하는 첫 단일 모델로써 중요한 이정표를 세웠습니다. 또한 이미지를 비디오로 전이 학습(task transfer)하여 새로운 능력을 발휘할 수 있습니다.

- **Technical Details**: LLaVA-OneVision의 모델 아키텍처는 LLM과 비전 인코더를 연결하는 간단한 모듈을 사용해 설계되었습니다. 주 사용된 LLM은 Qwen-2이며 비전 인코더로는 SigLIP를 사용합니다. 또한, 이미지 특징을 단어 임베딩 공간으로 투사하는 2층 MLP인 Projector를 사용합니다. 이 모델은 AnyRes 기술을 통해 높은 해상도의 이미지를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLaVA-OneVision은 단일 이미지, 다중 이미지, 비디오 시나리오 모두에서 성능을 발휘하며, 단일 모델이 다양한 시나리오에서 우수한 성능을 보이는 희귀한 사례로 인정받고 있습니다. 특히 영상 이해 및 시나리오 간 전이 학습을 통해 이전에는 가능하지 않았던 새로운 능력을 보여줍니다. 오픈 소스로 공개된 모델로써 관련 멀티모달 지침 데이터, 코드베이스, 모델 체크포인트 및 시각적 채팅 데모가 포함되어 있어 앞으로 더욱 확장 가능성이 높습니다.



### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (https://arxiv.org/abs/2408.03314)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 테스트 시 계산 자원을 추가로 사용하여 수행 능력을 향상시킬 수 있는 방법에 대해 연구합니다. 특히, 도전적인 프로프트(prompt)에서 성능을 얼마나 향상시킬 수 있는지를 밝히는 데 중점을 둡니다. 이를 통해 LLM의 사전 훈련 및 테스트 시 계산 자원 활용 방법에 대한 새로운 방향을 제시합니다.

- **Technical Details**: LLM의 테스트 시 계산 자원 활용을 두 가지 주요 메커니즘을 통해 분석합니다: (1) 밀집하고 과정 기반의 검증자 보상 모델(verifier reward models)을 사용한 검색 및 (2) 프로프트에 따라 모델의 응답 분포를 적응적으로 업데이트 하는 방식. 두 가지 접근 방식 모두 프로프트의 난이도에 따라 효과가 다르다는 것을 발견했습니다. 최적의 테스트 시 계산 자원 할당 전략을 적용하여 성능을 향상시키는 'compute-optimal' 스케일링 전략을 제안합니다.

- **Performance Highlights**: 최적의 계산 전략을 통해, 'best-of-N' 기준 대비 테스트 시 계산 자원의 효율성을 4배 이상 향상시켰습니다. 또한, FLOPs 매치 평가에서 작은 모델이 어느 정도 성공을 거두는 문제에 대해, 테스트 시 계산 자원을 활용하여 14배 큰 모델보다 더 나은 성과를 얻을 수 있었습니다.



### SARA: Singular-Value Based Adaptive Low-Rank Adaption (https://arxiv.org/abs/2408.03290)
- **What's New**: 이 연구에서는 수정된 Low-Rank Adaptation (LoRA) 방법인 SARA(Singular-Value Based Adaptive Low-Rank Adaption)와 Mo-SARA(Mixture-of-SARA)를 도입했습니다. 기존 LoRA 방법이 각 계층(layer)에 동일한 순위(rank)를 적용해야 하는 단점을 보완하고, SVD(Singular Value Decomposition)를 통해 각 계층에 맞는 최적의 순위를 찾아냅니다.

- **Technical Details**: SARA는 초기화 단계에서 사전 훈련된 가중치에 대해 SVD를 수행하여 각 계층의 중요도에 따라 적절한 순위를 자동으로 계산합니다. Mo-SARA는 복수의 병렬적인 singular value 집합을 사용하여 훈련해야 하는 매개변수 수를 크게 줄입니다. 이는 SVD를 통해 계층별로 적절한 순위를 찾아내어 전체 모델의 효율성을 높이는 방법입니다.

- **Performance Highlights**: 다양한 복잡한 작업에서 수행된 실험을 통해, 제안된 SARA와 Mo-SARA 방법이 높은 매개변수 효율성과 더불어 더 나은 성능을 제공함을 증명했습니다. 이는 기존의 LoRA 방법과 비교했을 때 더 적은 매개변수로도 우수한 성능을 가져옵니다.



### Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization (https://arxiv.org/abs/2408.03149)
Comments:
          In ACL-Findings 2024

- **What's New**: 빠르게 증가하는 멀티미디어 데이터를 효과적으로 요약하기 위해 텍스트와 이미지 모두를 결합한 다중 모드 요약(MSMO)의 중요성이 대두되고 있습니다. 이를 위해 저자들은 BART 기반의 Entity-Guided Multimodal Summarization (EGMS)라는 모델을 제안했습니다. EGMS는 이중 멀티모달 인코더와 게이팅 메커니즘을 사용하여 텍스트 및 이미지 데이터를 처리하며, 프리트레인된 비전-언어 모델에서 지식을 증류하여 이미지 선택을 개선합니다.

- **Technical Details**: EGMS 모델은 BART 프레임워크를 기반으로 구축되었습니다. 텍스트 중심 인코더를 수정하여 텍스트-이미지와 엔티티-이미지 정보를 동시에 처리하는 이중 멀티모달 인코더를 도입했습니다. 모델은 또한 게이팅 메커니즘을 통해 시각 데이터를 결합하여 개선된 텍스트 요약을 생성하며, 지식 증류를 통해 이미지를 선택합니다. 이 모든 과정을 통해 모델은 멀티모달 입력과 출력을 효과적으로 통합할 수 있습니다.

- **Performance Highlights**: 공개된 MSMO 데이터셋에서 EGMS 모델의 우수성이 입증되었습니다. 실험 결과는 엔티티 정보를 MSMO 문제에 통합하는 것이 필요함을 보여주었으며, 제안된 접근법이 기존 방법들보다 더 나은 성능을 나타냈습니다.



### OpenOmni: A Collaborative Open Source Tool for Building Future-Ready Multimodal Conversational Agents (https://arxiv.org/abs/2408.03047)
- **What's New**: OpenOmni라는 오픈 소스, 엔드 투 엔드 파이프라인 벤치마킹 도구를 개발했습니다. 이 도구는 Speech2Text, Emotion Detection, Retrieval Augmented Generation(RAG), Large Language Models(LLMs)와 같은 첨단 기술을 통합하여 협업 개발 및 벤치마킹을 지원합니다.

- **Technical Details**: OpenOmni는 로컬 및 클라우드 배포를 지원하여 데이터 프라이버시와 지연 시간 및 정확성 벤치마킹을 보장합니다. 프레임워크는 비디오 및 오디오 데이터를 처리하고 사용자 맞춤형 에이전트 파이프라인을 통해 응답을 생성합니다. 주요 기술 요소는 Speech2Text, Emotion Detection, RAG, LLMs, TTS(Text-to-Speech)입니다.

- **Performance Highlights**: OpenOmni는 기존 시스템들의 성능인 200-250ms의 응답 시간에 비해 데이터 프라이버시와 벤치마킹 도구의 확장성을 제공하여 사용 사례에 맞게 최적화된 애플리케이션 개발을 가능하게 합니다. 이를 통해 시각 장애인을 위한 실내 보조 기능 등 다양한 애플리케이션에서 성능 향상을 도모할 수 있습니다.



### HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection (https://arxiv.org/abs/2408.02927)
- **What's New**: 최근 연구는 개인정보 보호 문제를 해결하면서도 현실적인 가상 표 형식 데이터를 생성할 수 있는 새로운 프레임워크인 HARMONIC을 도입했습니다. 이 프레임워크는 대규모 언어 모델(LLM)을 활용하여 표 데이터를 생성하고, 이 과정에서 개인정보 유출 위험을 최소화합니다.

- **Technical Details**: HARMONIC 프레임워크는 기존의 LLM 기반 방법들과 달리 소규모 모델이 아닌 대규모 모델을 사용하여 표 데이터를 생성합니다. 이 과정에서 k-최근접 이웃 알고리즘(k-nearest neighbors)을 참고하여 유사한 행들 간의 관계를 학습시키고, 데이터를 기억하는 대신 데이터의 형식과 연결성을 기억하도록 미세 조정(fine-tuning)합니다. 또한, DLT(Data Leakage Test)와 LLE(LLM Efficiency)라는 새로운 평가 지표를 도입하여 생성된 데이터의 개인정보 유출 위험과 모델의 성능을 평가합니다.

- **Performance Highlights**: HARMONIC 프레임워크를 사용한 실험 결과, 기존 방법들과 비교했을 때 개인정보 보호가 강화되었으며, 머신 러닝과 다운스트림 작업에서 동등하거나 더 나은 성능을 보였습니다. 특히, 기존의 가상 데이터 생성 방법이 LLM 기반 다운스트림 작업에 적합하지 않음을 보여주며, 사전 학습 기반의 가상 데이터가 심각한 개인정보 유출 위험을 내포하고 있음을 시사합니다.



### Lighthouse: A User-Friendly Library for Reproducible Video Moment Retrieval and Highlight Detection (https://arxiv.org/abs/2408.02901)
Comments:
          6 pages; library tech report

- **What's New**: Lighthouse는 동영상 순간 검색 및 하이라이트 검출 (MR-HD: Moment Retrieval and Highlight Detection)을 위한 통합적이고 재현 가능한 코드베이스를 제공하는 사용자 친화적인 라이브러리입니다. 현재 연구 커뮤니티에서 다양한 방법, 데이터셋, 비디오-텍스트 기능(feature)을 통해 종합적이고 재현 가능한 실험을 수행하지 않는 두 가지 주요 문제가 있습니다. Lighthouse는 이러한 문제를 해결하기 위해 여섯 가지 모델, 세 가지 기능, 다섯 가지 데이터셋을 포함한 통합적이고 재현 가능한 코드베이스를 구현하였습니다. 또한 인퍼런스 API와 웹 데모를 제공하여 연구자와 개발자가 쉽게 접근할 수 있게 합니다.

- **Technical Details**: Lighthouse는 여섯 개의 최신 MR-HD 방법, 세 가지 비디오-텍스트 기능 및 다섯 가지 데이터셋을 지원하는 통합 코드베이스로, YAML 형식의 구성 파일을 통해 매개변수를 지정하여 단일 Python 명령어로 실험을 재현할 수 있습니다. 또한, 웹 데모와 인퍼런스 API 구현을 통해 사용자가 상세한 비디오-텍스트 처리를 이해하지 못해도 MR-HD를 쉽게 사용할 수 있도록 했습니다.

- **Performance Highlights**: Lighthouse는 참조 논문에서 보고된 점수를 대체로 재현하며, 기존의 여러 라이브러리와 달리 사용자 친화적인 디자인으로 사용자가 단일 환경에서 실험을 설정할 수 있도록 지원합니다. 설치 또한 간단하여 적은 노력으로 다양한 설정에서 실험을 수행할 수 있습니다.



### VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledg (https://arxiv.org/abs/2408.02865)
- **What's New**: VisionUnite는 안과 분야를 위한 새로운 비전-언어 융합 대형 모델입니다. 이 모델은 1.24백만 개 이미지-텍스트 쌍의 대규모 데이터셋으로 사전 훈련되고, 추가로 MMFundus 데이터셋(296,379개의 고품질 눈저부 이미지-텍스트 쌍 및 889,137개의 모의 의사-환자 대화 인스턴스)으로 세부 조정되었습니다.

- **Technical Details**: VisionUnite는 Transformer 기반의 비전 인코더(vision encoder), 비전 어댑터(vision adapter) 및 LLaMA 모델로 고유의 목적을 달성하도록 세 가지 훈련 목표(visual text contrastive learning loss(CLIP Loss), visual classification loss(CLS Loss) 및 text generation loss(LLM Loss))로 학습됩니다. VisionUnite는 자연 이미지-텍스트 쌍과 생의학 이미지-텍스트 쌍을 포함한 1.19백만 개의 이미지-텍스트 쌍의 종합적인 데이터셋을 사용하여 사전 훈련됩니다.

- **Performance Highlights**: VisionUnite는 다양한 임상 시나리오에서 뛰어난 성능을 보입니다. 존재하는 GPT-4V 및 Gemini Pro와 같은 기존 생성 모델보다 우수하며, 진단 능력은 초급 안과 전문의와 비슷한 수준입니다. VisionUnite는 다중 질병 진단, 임상 설명 및 환자 상호 작용 등의 다양한 임상 시나리오에서 잘 작동합니다. 또한 초급 안과 의사의 훈련을 돕는 교육적인 도구로 사용할 수 있습니다.



### Interpretation of the Intent Detection Problem as Dynamics in a Low-dimensional Spac (https://arxiv.org/abs/2408.02838)
Comments:
          Camera-Ready version. Accepted paper at 27th European Conference on Artificial Intelligence (ECAI-2024)

- **What's New**: 최근 SNIPS 데이터셋을 기반으로 한 의도 탐지(intention detection) 작업에서 RNN이 데이터를 처리하는 숨겨진 메커니즘을 새로운 관점에서 분석했습니다. 문장은 RNN의 훈련된 네트워크에 입력되며, 이는 숨겨진 상태 공간을 통해 궤적을 형성합니다. 이 공간은 저차원 매니폴드에 제한되어 있으며, 이는 임베딩과 숨겨진 계층의 크기와 관련 있습니다.

- **Technical Details**: 이 연구는 현대의 RNN이 데이터 시퀀스를 처리할 때의 동적 시스템(dynamic systems) 관점에서 RNN이 어떻게 SNIPS 의도 탐지 문제를 해결하는지를 조사합니다. 문장이 훈련된 RNN에 주입될 때 이 문장은 숨겨진 상태 공간을 통과하는 궤적을 형성합니다. 이 상태 공간은 저차원 매니폴드(manifold)에 제한되어 있으며, 이 매니폴드의 내재된 차원성은 임베딩 계층 및 숨겨진 계층의 크기와 관련이 있습니다. RNN은 이 궤적을 출력 계층의 행들이 지시하는 구체적인 지역으로 유도하여 예측을 생성합니다. 또 다른 중요한 발견은 제한된 수의 어트랙터(attractor)를 가진 예기치 않은 고정점 구조가 존재한다는 것입니다.

- **Performance Highlights**: 연구 결과에 따르면, RNN이 의도 탐지 문제를 해결할 때 감정 분석이나 문서 분류와는 다른 고정점 구조가 나타납니다. 네트워크가 학습한 어트랙터와 안장점(saddle point)의 수는 네트워크의 매개변수와 셀의 종류에 따라 달라집니다. 이러한 결과는 SNIPS 데이터셋을 넘어 일반적인 의도 탐지 문제로 확장될 가능성이 있습니다.



### Entity Retrieval for Answering Entity-Centric Questions (https://arxiv.org/abs/2408.02795)
Comments:
          17 pages total, 10 Tables, 4 Figures

- **What's New**: 이번 연구에서는 질문-문서 유사성에 의존하지 않고, 질문 내의 중심 엔티티(salient entities)를 사용하여 문서를 검색하는 새로운 방법인 'Entity Retrieval'을 제안합니다. 기존에 사용된 조밀(dense) 및 희소(sparse) 검색 방법과의 성능 비교를 통해, 이 방법이 엔티티 중심 질문에 대해 더 정확하고 효율적인 답을 제공함을 발견했습니다.

- **Technical Details**: Entity Retrieval은 질문 내의 주요 엔티티를 통해 지식 기반(예: Wikipedia)의 해당 기사를 찾아냅니다. 각 기사는 처음 W(글자수: 첫 문장) 단어로 잘라 문서 세트를 구성하며, 이 문서 세트는 LLM에게 질문을 전달하는 데 사용됩니다. BM25, DPR, ANCE와 같은 기존 검색 방법들과 비교하였으며, 실험은 Pyserini의 기존 인덱스를 활용했습니다. BM25는 확률적 검색 방법으로, DPR과 ANCE는 각각 질문과 텍스트를 임베딩하고 FAISS 인덱스를 사용하여 유사성을 계산합니다.

- **Performance Highlights**: Entity Retrieval은 기존의 BM25, DPR, ANCE 검색 방법과 비교하여 엔티티 중심 질문에 대해 더 정확한 답변을 제공하며, 더 효율적으로 작동합니다. 특히, 희소(retrievers)보다 더 적절한 문서를 찾는 데 효과적임을 보여주었습니다. 실험 결과, 다양한 엔티티 중심 질문 데이터셋에서도 이러한 경향을 관찰할 수 있었습니다.



### SentenceVAE: Enable Next-sentence Prediction for Large Language Models with Faster Speed, Higher Accuracy and Longer Contex (https://arxiv.org/abs/2408.00655)
Comments:
          update the article

- **What's New**: 본 논문에서는 기존 대형 언어 모델(LLMs)의 예측 방식인 다음 토큰(next-token) 예측 방법이 처리 속도를 저하시킨다는 문제를 다룹니다. 이를 해결하기 위해 문장 단위 예측 방식인 'next-sentence prediction'을 제안합니다. 이 새로운 방식은 Sentence Variational Autoencoder (SentenceVAE)를 도입하여 문장 정보를 단일 토큰으로 압축 후 복원하는 과정을 통해 보다 빠르고 효율적인 추론을 가능하게 합니다.

- **Technical Details**: SentenceVAE는 Sentence Encoder와 Sentence Decoder로 구성되며, 문장의 정보를 단일 토큰으로 압축하고 이를 다시 원래의 문장으로 복원하는 기능을 가집니다. 이 SentenceVAE를 LLM의 입력 및 출력 층에 삽입하여 'Sentence-level LLMs (SLLMs)'를 개발하였습니다. SLLMs는 문장 단위 예측 방식을 사용하여 문맥을 문장으로 분할하는 동시에 원래의 의미를 유지합니다. 이를 통해 추론 속도를 크게 향상시키고, 메모리 부담을 줄이며 더 긴 문맥을 처리할 수 있습니다.

- **Performance Highlights**: Wanjuan 데이터셋을 활용한 광범위한 실험 결과, 제안된 방법은 추론 속도를 204~365% 향상시키고, Perplexity (PPL)를 기존 대비 46~75%로 감소시키며, 동등한 문맥 길이에 대해 메모리 오버헤드를 86~91% 줄이는 것으로 나타났습니다. 이러한 결과는 기존의 토큰 단위 예측 방법보다 월등한 성능 향상을 의미합니다.



New uploads on arXiv(cs.IR)

### CADRL: Category-aware Dual-agent Reinforcement Learning for Explainable Recommendations over Knowledge Graphs (https://arxiv.org/abs/2408.03166)
- **What's New**: 지식 그래프(KG)는 데이터 희소성과 콜드 스타트 문제를 완화하는 데 널리 사용되고 있으며, 최근에는 강화학습(RL)을 사용해 설명 가능한 추천 경로를 통해 적절한 아이템을 찾는 방식이 주목받고 있습니다. 하지만 기존의 RL 기반 방법들은 문맥적 의존성을 포착하는 능력과 효율성 때문에 짧은 경로에 과도하게 의존하는 문제로 인해 성능이 제한적입니다. 이를 극복하기 위해 카테고리 인지형 듀얼 에이전트 강화 학습(CADRL) 모델을 제안합니다. 이 모델은 카테고리 인지형 게이티드 그래프 신경망을 사용하여 문맥 인지형 아이템 표현을 캡처하고, 듀얼 에이전트 RL 프레임워크를 통해 긴 경로를 효율적으로 탐색합니다.

- **Technical Details**: CADRL 모델은 두 가지 주요 컴포넌트로 구성됩니다. 첫째, 카테고리 인지형 게이티드 그래프 신경망(Category-aware Gated Graph Neural Network, CGGNN)을 통해 이웃 엔터티와 아이템 카테고리에서 문맥적 의존성을 포착하여 고차원의 아이템 표현을 생성합니다. 둘째, 듀얼 에이전트 강화학습 프레임워크(Dual-Agent Reinforcement Learning, DARL)를 사용해 협력 보상 메커니즘과 공유 정책 네트워크를 통해 긴 경로를 탐색하며 설명 가능한 추천을 수행합니다.

- **Performance Highlights**: CADRL 모델은 대규모 데이터셋에서 최첨단 모델들보다 효과적이고 효율적이라는 실험 결과를 보여주었습니다. 여러 실제 세계의 벤치마크 데이터셋에서 수행된 실험에서 CADRL의 우수성이 입증되었습니다.



### Modeling User Intent Beyond Trigger: Incorporating Uncertainty for Trigger-Induced Recommendation (https://arxiv.org/abs/2408.03091)
- **What's New**: 새롭게 제안된 모델 Deep Uncertainty Intent Network (DUIN)은 전통적인 Trigger-Induced Recommendation (TIR) 문제를 해결하기 위해 개발되었습니다. TIR은 사용자 클릭 이력에 기반한 몰입형 추천 시스템을 구현하는 데 중요한 역할을 하지만, 기존 방법은 트리거 아이템에 지나치게 의존해 사용자 의도의 복잡성을 충분히 반영하지 못하는 문제를 가지고 있습니다. DUIN은 이러한 문제를 극복하고자 설계되었습니다.

- **Technical Details**: DUIN은 세 가지 주요 모듈로 구성됩니다: i) Explicit Intent Exploit Module (EIEM)은 contrastive learning paradigm을 사용해 명시적인 사용자 의도를 추출합니다. ii) Latent Intent Explore Module (LIEM)은 아이템 간의 다중 관점 관계를 활용해 잠재적인 사용자 의도를 탐색합니다. iii) Intent Uncertainty Measurement Module (IUMM)은 사용자 의도의 불확실성을 Gaussian distribution으로 모델링합니다. 이러한 각 모듈은 트리거 아이템의 맥락에서 보다 정밀하고 포괄적인 사용자 의도 모델링을 가능하게 합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 사용한 실험 결과, DUIN은 기존 최첨단 모델들 대비 우수한 성능을 보여주었습니다. 이러한 성과는 Alibaba.com과 같은 상업 플랫폼의 모든 TIR 시나리오에 DUIN이 적용되었고, 온라인 A/B 테스트로도 그 우수성이 검증되었습니다.



### The Crowd in MOOCs: A Study of Learning Patterns at Sca (https://arxiv.org/abs/2408.03025)
Comments:
          16 pages

- **What's New**: 이 논문은 800,000명의 학습자들이 2년간 1,600개 이상의 코스에 등록하면서 발생한 3억 5천만 건의 학습 활동 데이터를 분석했습니다. 시간적 관점과 코스 등록 관점에서 학습 패턴을 식별하여 대규모 학습자 행동을 분석했습니다.

- **Technical Details**: 연구에서는 상호 정보 이론과 순차 패턴 마이닝(sequential pattern mining) 기법을 사용하여 학습자들의 시간 간격 패턴과 코스 등록 패턴을 분석했습니다. 시간적 관점에서는 학습 활동 간의 시간 간격이 파워 법칙과 주기적 코사인 함수 분포를 따르는 것으로 나타났습니다. 또한, 동일 카테고리 또는 동일 대학에서 제공되는 코스들 간의 동시 등록 패턴이 높은 것으로 밝혀졌습니다.

- **Performance Highlights**: 연구에서 제시한 단순 추천 모델은 코스 등록 패턴을 활용하여 베이스라인에 비해 200배 빠른 학습 시간을 기록하면서도 경쟁력 있는 성능을 보여줍니다. 이는 MOOC 플랫폼 운영자와 코스 강사, 학습자들에게 유용한 적용 사례를 제공합니다.



### A Real-Time Adaptive Multi-Stream GPU System for Online Approximate Nearest Neighborhood Search (https://arxiv.org/abs/2408.02937)
Comments:
          Accepted by CIKM'24

- **What's New**: 최근 논문에서는 실시간 벡터 삽입 기능을 지원하는 Real-Time Adaptive Multi-Stream GPU ANNS System (RTAMS-GANNS)를 제안했습니다. 이 시스템은 특히 검색 및 추천 시스템과 같은 온라인 애플리케이션에서 사용자를 위한 최신 데이터를 빠르게 제공하기 위해 설계되었습니다.

- **Technical Details**: RTAMS-GANNS는 세 가지 주요 혁신을 통해 실시간 벡터 삽입을 구현합니다: 1) 기존 GPU ANNS 시스템의 메모리 할당 및 복사 문제를 해결하는 동적 벡터 삽입 알고리즘을 도입했습니다. 2) 단일 스트림이 아닌 멀티 스트림 병렬 실행 모드를 통해 실시간 삽입을 가능하게 했습니다. 이를 통해 여러 스트림이 동시에 실행될 수 있도록 합니다. 3) 다양한 데이터셋에서 실험을 통해 이 접근법이 QPS 수준을 효과적으로 처리하며, 지연 시간을 40%에서 80%까지 줄였습니다.

- **Performance Highlights**: RTAMS-GANNS는 실험 결과 다양한 데이터셋에서 지연 시간을 현저히 줄이며 성능을 입증했습니다. 이 솔루션은 또한 실제 산업 검색 및 추천 시스템에 성공적으로 배포되어 일일 수억 명의 사용자를 효과적으로 지원하고 있습니다.



### Wiping out the limitations of Large Language Models -- A Taxonomy for Retrieval Augmented Generation (https://arxiv.org/abs/2408.02854)
- **What's New**: 최근의 연구는 다양한 학문 분야에서 RAG(Retrieval-Augmented Generation) 관련 기술을 연구하고 있으며, 기술 혁신에 초점을 맞추고 있습니다. 본 연구에서는 RAG 응용의 포괄적인 개요를 개념화하고 IS(정보 시스템) 커뮤니티에서 이 기술의 도입을 촉진하기 위해 분류 체계를 만들기를 목표로 합니다. 현재까지 RAG 응용에 대한 분류 체계는 개발되지 않았습니다.

- **Technical Details**: 우리는 분류 체계를 개발하는 방법론을 설명하고, 논문을 선택하는 기준과 초기 특성을 추출하고 식별하기 위한 대형 언어 모델(LLM)에 대한 지원 접근 방식을 사용한 이유를 설명합니다. 또한 RAG 응용의 핵심 차원을 개념화하기 위한 체계적인 프로세스를 간략하게 개요합니다. 이 체계적인 분류 체계 개발 과정은 네 가지 반복 단계를 포함하며, 이를 통해 RAG의 핵심 차원을 이해하고 표현하는 능력을 강화합니다.

- **Performance Highlights**: 우리는 RAG 응용의 개념을 포괄적으로 포착하기 위해 총 다섯 가지 메타 차원(meta-dimensions)과 열여섯 가지 차원(dimensions)을 개발했습니다. 연구 결과를 논의하면서 우리는 특정 연구 영역을 상세히 설명하고 향후 정보 시스템 연구자들이 RAG 시스템의 새로운 주제를 탐구하기 위한 주요 연구 질문을 제시합니다.



### Entity Retrieval for Answering Entity-Centric Questions (https://arxiv.org/abs/2408.02795)
Comments:
          17 pages total, 10 Tables, 4 Figures

- **What's New**: 이번 연구에서는 질문-문서 유사성에 의존하지 않고, 질문 내의 중심 엔티티(salient entities)를 사용하여 문서를 검색하는 새로운 방법인 'Entity Retrieval'을 제안합니다. 기존에 사용된 조밀(dense) 및 희소(sparse) 검색 방법과의 성능 비교를 통해, 이 방법이 엔티티 중심 질문에 대해 더 정확하고 효율적인 답을 제공함을 발견했습니다.

- **Technical Details**: Entity Retrieval은 질문 내의 주요 엔티티를 통해 지식 기반(예: Wikipedia)의 해당 기사를 찾아냅니다. 각 기사는 처음 W(글자수: 첫 문장) 단어로 잘라 문서 세트를 구성하며, 이 문서 세트는 LLM에게 질문을 전달하는 데 사용됩니다. BM25, DPR, ANCE와 같은 기존 검색 방법들과 비교하였으며, 실험은 Pyserini의 기존 인덱스를 활용했습니다. BM25는 확률적 검색 방법으로, DPR과 ANCE는 각각 질문과 텍스트를 임베딩하고 FAISS 인덱스를 사용하여 유사성을 계산합니다.

- **Performance Highlights**: Entity Retrieval은 기존의 BM25, DPR, ANCE 검색 방법과 비교하여 엔티티 중심 질문에 대해 더 정확한 답변을 제공하며, 더 효율적으로 작동합니다. 특히, 희소(retrievers)보다 더 적절한 문서를 찾는 데 효과적임을 보여주었습니다. 실험 결과, 다양한 엔티티 중심 질문 데이터셋에서도 이러한 경향을 관찰할 수 있었습니다.



### Fact Finder -- Enhancing Domain Expertise of Large Language Models by Incorporating Knowledge Graphs (https://arxiv.org/abs/2408.03010)
Comments:
          10 pages, 7 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 질문에 대답하는 능력을 보여주었지만, 도메인별 지식의 한계로 인해 정확성이 떨어지는 문제가 있었습니다. 이를 해결하기 위해 지식 그래프(KG)를 결합한 하이브리드 시스템을 제안합니다. 이 시스템은 KG 기반 검색 방식을 사용해 사실적 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 우리는 의료 지식 그래프(KG)를 사용하여 다음과 같은 단계를 포함한 방법론을 제시합니다: (1) 데이터 전처리, (2) 사이퍼(Cypher) 쿼리 생성, (3) 사이퍼 쿼리 처리, (4) KG 검색, (5) LLM을 통한 응답 생성. 우리의 시스템은 PrimeKG을 사용하여 20개의 고품질 리소스를 통합한 17,080개의 질병과 4,050,249개의 관계를 설명합니다. 또한, 69개의 텍스트-사이퍼 쿼리 쌍으로 구성된 데이터셋을 사용해 평가를 수행했습니다.

- **Performance Highlights**: 테스트 결과, 올바른 KG 노드를 검색하는 정확도가 78%에 달하는 것으로 나타났습니다. 이 하이브리드 시스템은 정확성과 완전성에서 단독 LLM을 능가하며, 'LLM-as-a-Judge' 평가 방법에 의해 검증되었습니다. 또한, 직관적인 검색 인터페이스와 몇 초 내에 정확한 응답을 제공하는 능력을 갖추고 있어, 시간과 정확성이 중요한 연구 환경에 적합합니다.



### Symmetric Graph Contrastive Learning against Noisy Views for Recommendation (https://arxiv.org/abs/2408.02691)
Comments:
          24 pages, submitted to TOIS

- **What's New**: 이번 연구에서는 새로운 그래프 대비 학습 방법인 SGCL(Symmetric Graph Contrastive Learning)를 제안합니다. 기존의 데이터 증강 기법들이 원래 인터랙션 그래프의 연결을 방해하고, 부정확한 대비 뷰(contrastive view)를 생성해 최적의 성능을 발휘하지 못하는 문제를 해결하고자 합니다. SGCL은 대칭 이론을 도입하여 소음이 많은 보는 것에 저항하는 대칭 형태와 대조 손실을 구현합니다.

- **Technical Details**: SGCL은 모델-애그노스틱(model-agnostic) 방식으로 대칭 이론을 그래프 대비 학습에 통합합니다. 주로 다음을 포함합니다: 원래 그래프와 소음이 많은 보기를 구분하고, 후기의 20% 뷰의 코사인 유사도가 0.1 미만인 경우 노이즈로 간주합니다. SGCL은 이러한 노이즈가 성능을 저하시키는 것을 방지합니다. 또한, 이 방법은 이론적으로 소음이 많은 보기에도 높은 내성을 가진다는 것을 증명했습니다.

- **Performance Highlights**: 실험 결과, SGCL 방식은 세 가지 실제 데이터셋에서 기존 방법들보다 최대 12.25% 더 높은 추천 정확도를 기록했습니다. 이는 이 접근 방식의 효능을 높이 평가합니다.



### Deep Uncertainty-Based Explore for Index Construction and Retrieval in Recommendation System (https://arxiv.org/abs/2408.00799)
Comments:
          accepted by cikm2024

- **What's New**: 새롭게 제안된 UICR(Uncertainty-based explore for Index Construction and Retrieval) 알고리즘은 불확실성 개념을 매칭 단계에 도입하여 추천 시스템의 결과에서 관련성과 참신성을 모두 개선합니다. 이 알고리즘은 모델 불확실성과 인덱스 불확실성의 다중 작업 모델링을 수행합니다.

- **Technical Details**: UICR은 세 가지 주요 컴포넌트로 구성됩니다: UN-Index(불확실성 기반 인덱스 생성), UN-Retrieval(불확실성 기반 검색), 그리고 UN-Model(불확실성 모델링). UN-Model은 사용자-아이템 및 아이템-아이템 불확실성을 활용하여 점수와 불확실성을 추정합니다. UN-Retrieval는 검색 과정에서 관련성과 불확실성을 함께 고려하여 최종 결과를 도출합니다.

- **Performance Highlights**: 실험 결과 UICR 알고리즘은 실제 산업 환경과 여러 공개 데이터셋에서 참신성을 희생하지 않으면서 관련성을 성공적으로 개선하였습니다. 특히, Shopee의 온라인 A/B 테스트 결과에서 UICR 알고리즘의 효과성을 입증하였습니다.



New uploads on arXiv(cs.CV)

### LLaVA-OneVision: Easy Visual Task Transfer (https://arxiv.org/abs/2408.03326)
Comments:
          Project Homepage: this https URL

- **What's New**: LLaVA-OneVision은 LLaVA-NeXT 블로그 시리즈에서 얻어진 통찰을 바탕으로 개발된 대형 멀티모달 모델(LMM)입니다. 특히 단일 이미지, 다중 이미지, 비디오 시나리오 등 세 가지 중요한 컴퓨터 비전 시나리오에서 성능 경계를 확장하는 첫 단일 모델로써 중요한 이정표를 세웠습니다. 또한 이미지를 비디오로 전이 학습(task transfer)하여 새로운 능력을 발휘할 수 있습니다.

- **Technical Details**: LLaVA-OneVision의 모델 아키텍처는 LLM과 비전 인코더를 연결하는 간단한 모듈을 사용해 설계되었습니다. 주 사용된 LLM은 Qwen-2이며 비전 인코더로는 SigLIP를 사용합니다. 또한, 이미지 특징을 단어 임베딩 공간으로 투사하는 2층 MLP인 Projector를 사용합니다. 이 모델은 AnyRes 기술을 통해 높은 해상도의 이미지를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLaVA-OneVision은 단일 이미지, 다중 이미지, 비디오 시나리오 모두에서 성능을 발휘하며, 단일 모델이 다양한 시나리오에서 우수한 성능을 보이는 희귀한 사례로 인정받고 있습니다. 특히 영상 이해 및 시나리오 간 전이 학습을 통해 이전에는 가능하지 않았던 새로운 능력을 보여줍니다. 오픈 소스로 공개된 모델로써 관련 멀티모달 지침 데이터, 코드베이스, 모델 체크포인트 및 시각적 채팅 데모가 포함되어 있어 앞으로 더욱 확장 가능성이 높습니다.



### MDT-A2G: Exploring Masked Diffusion Transformers for Co-Speech Gesture Generation (https://arxiv.org/abs/2408.03312)
- **What's New**: 최신 연구는 확산 변환기(Diffusion Transformers)를 통해 2D 이미지, 3D 비디오 및 3D 모양의 생성 품질을 크게 개선했습니다. 하지만, 동기화된 음성 제스처 생성(co-speech gesture generation)에서 변환기 아키텍처의 효과는 상대적으로 미개발 분야로 남아 있습니다. 이를 해결하기 위해, 새로운 Masked Diffusion Transformer인 MDT-A2G를 도입하여 제스처 시퀀스에서 직접 디노이징(denoising) 과정을 수행합니다. MDT-A2G는 음성, 텍스트, 감정, 정체성을 포함한 멀티모달 정보를 통합하여 다양한 상황에 맞는 제스처를 생성합니다.

- **Technical Details**: MDT-A2G는 음성 기반 제스처 생성(Audio-to-Gesture) 작업을 위한 새로운 Masked Diffusion Transformer 프레임워크입니다. 이 모델은 시퀀스 제스처 간의 시간적 관계 학습을 강화하기 위해 설계된 마스크 모델링 기법을 사용합니다. 또한, 멀티모달 정보(텍스트, 감정, 정체성)를 포함하여 복잡한 피처 결합 프로세스를 단순화하기 위한 멀티모달 피처 결합 모듈을 통합했습니다. 이를 통해 학습 속도를 빠르게 하고, 일관성 있고 현실적인 모션을 생성합니다.

- **Performance Highlights**: MDT-A2G는 전통적인 확산 변환기보다 학습 속도가 6배 빠르고, 표준 확산 모델보다 추론 속도가 5.7배 빠릅니다. 실험 결과, 제안된 MDT-A2G는 상체 및 전신의 제스처 생성에 대한 질적(qualitative) 및 양적(quantitative) 평가에서 최첨단 성능(state-of-the-art)을 보였습니다. 이 모델은 고품질, 다양한 인간 제스처를 효율적으로 생성하여 현재의 동기화된 음성 제스처 생성 방법들을 능가합니다.



### Fusing Forces: Deep-Human-Guided Refinement of Segmentation Masks (https://arxiv.org/abs/2408.03304)
Comments:
          16 pages, accepted at ICPR2024

- **What's New**: 에트루리아 거울에 새겨진 복잡한 그림을 분석하고 문서화하는 과정에서 수동으로 일러스트레이션을 추적하는 작업의 노동력과 비용을 감소시키기 위해 인간 상호작용 기반의 심층 신경망을 제안했습니다. 초기 예측을 바탕으로 인간 안내를 통해 주석(annotations)을 정제(refinement)하도록 훈련된 이 네트워크는 최대 75%의 인력 절감을 가능하게 하며, 수동 라벨링에 비해 최대 26% 더 빠른 품질 향상을 제공합니다.

- **Technical Details**: 이 방법론은 포토메트릭-스테레오 스캐닝(photometric-stereo scanning)과 심층 신경망(deep neural networks)을 결합하여 자동으로 세분화 과정(segmentation process)을 수행합니다. 데이터를 전처리하고, 거울의 깊이 맵(depth maps)을 사용하여 의도적인 선(line)을 인식하도록 모델을 훈련시켰습니다. 그 결과물의 세밀한 정제를 위해 인간의 상호작용(human-in-the-loop approach)을 추가하여 초기 예측에서 시작해 정제 과정에서 팁, 즉 부분적으로 추가하거나 삭제하는 과정을 거쳤습니다.

- **Performance Highlights**: 수동으로 초기 세분화를 정제하는 것과 비교했을 때, 인력 투입이 최대 75% 감소하고, 수동 라벨링에 비해 최대 26% 더 빠른 품질 향상을 달성했습니다. 이는 복잡한 선을 구분하는 작업에서 큰 성과를 보였으며, 에트루리아 거울뿐만 아니라 다양한 응용 분야에 적용이 가능합니다.



### TextIM: Part-aware Interactive Motion Synthesis from Tex (https://arxiv.org/abs/2408.03302)
- **What's New**: 본 논문에서는 텍스트 기반의 인간 상호작용 동작을 합성하는 새로운 프레임워크인 TextIM을 제안합니다. TextIM은 상호작용하는 신체 부위의 세부적인 의미 정렬을 개선하여, 기존 방법들이 간과했던 상호작용의 정확성을 크게 향상시킵니다.

- **Technical Details**: TextIM은 분리된 조건부 확산 (decoupled conditional diffusion) 프레임워크를 사용하여 텍스트 설명에서 상호작용 의도를 이해하고 이를 기반으로 정교하고 복잡한 상호작용 동작을 생성합니다. 텍스트 설명과 정밀하게 맞춰진 상호작용을 위해 대형 언어 모델(large language models)을 활용하며, 상호작용신체 부위의 정제된 움직임을 전체 신체 동작으로 확장합니다. 그래프 구조를 이용한 공간적 일관성 모듈(spatial coherence module)은 신체 부위 간의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, TextIM은 텍스트 설명과 일치하는, 의미적으로 정확한 인간 상호작용 동작을 생성하여 다양한 시나리오에서 현실감을 크게 향상시킵니다. 특히 동적인 객체나 변형 가능한 객체와의 상호작용을 포함한 사례에서도 우수한 성능을 보였습니다. TextIM은 HUMANML3D 데이터셋에서 재라벨링된 상호작용 동작을 사용해 학습 및 평가되었습니다.



### DopQ-ViT: Towards Distribution-Friendly and Outlier-Aware Post-Training Quantization for Vision Transformers (https://arxiv.org/abs/2408.03291)
- **What's New**: 이번 연구에서는 Vision Transformers(ViTs)를 위한 새로운 사후 훈련 양자화(Post-Training Quantization, PTQ) 방법인 DopQ-ViT를 제안했습니다. DopQ-ViT는 기존 양자화 방식의 비효율성을 분석하고, 이를 해결하기 위해 분포 친화적 Tan Quantizer(TanQ)와 최적의 스케일링 팩터를 찾는 SOSF(Search for the Optimal Scaling Factor) 방법을 도입했습니다. 이 방법은 낮은 비트 설정에서도 성능 향상을 입증했습니다.

- **Technical Details**: DopQ-ViT는 포스트-Softmax 활성화의 파워-법칙(power-law) 분포를 더 정확히 유지하는 Tan Quantizer(TanQ)를 도입하여 양자화 효율성을 개선합니다. 또한, 채널-기반에서 계층-기반 양자화로 변경할 때 발생하는 정확도 감소를 보완하기 위해 최적의 스케일링 팩터를 찾는 SOSF(Search for the Optimal Scaling Factor) 방법을 제안했습니다. 이 방법은 이상적인 채널을 보완하며 대부분의 채널에 정확한 양자화를 실현합니다.

- **Performance Highlights**: DopQ-ViT는 이미지 분류 및 객체 탐지 작업에서 다양한 모델 변형과 비트 폭(bit-widths)을 사용하여 평가되었습니다. 기존의 사후 훈련 양자화 방법과 비교하여, DopQ-ViT는 경쟁력 있는 성능을 보여주었습니다. 특히, 낮은 비트 설정에서도 눈에 띄는 성능 향상을 확인할 수 있었습니다.



### Biomedical SAM 2: Segment Anything in Biomedical Images and Videos (https://arxiv.org/abs/2408.03286)
- **What's New**: BioSAM 2은 의료용 데이터에 최적화된 enhanced foundation model로, 기존의 자연 이미지 segmentation model인 SAM 2를 기반으로 개발되었습니다. BioSAM 2는 의료 이미지 및 비디오 segmentation 작업에서 state-of-the-art 모델보다 우수한 성능을 보이며, 전문 모델과도 견줄 만한 또는 이를 초과하는 성능을 나타냅니다.

- **Technical Details**: BioSAM 2는 메모리 메커니즘과 stream processing architecture를 도입하여 multiple frame segmentation 작업에서 과거 예측 정보를 보유함으로써 정확한 예측을 가능하게 합니다. 이 모델은 다양한 프롬프트를 활용해 single-frame 이미지와 multi-frame 비디오 segmentation 평가 파이프라인을 구축하였으며, CNN, Transformer, SSM 기반의 여러 기준 모델과 비교 평가를 통해 성능을 검증하였습니다.

- **Performance Highlights**: 실험 결과, BioSAM 2는 기존의 state-of-the-art foundation 모델을 상당한 차이로 능가하는 동시에, 동일한 의료 데이터 모달리티로 훈련된 전문 모델의 성능과 비슷하거나 이를 초과하는 결과를 보여주었습니다. 이는 BioSAM 2가 다양한 의료 응용 분야에서 사용할 수 있는 새로운 패러다임이 될 가능성을 강조합니다.



### ReSyncer: Rewiring Style-based Generator for Unified Audio-Visually Synced Facial Performer (https://arxiv.org/abs/2408.03284)
Comments:
          Accepted to European Conference on Computer Vision (ECCV), 2024. Project page: this https URL

- **What's New**: ReSyncer는 오디오와 시각적 정보의 동기화를 실현하는 새로운 통합 프레임워크입니다. 기존 모델들이 긴 비디오 클립을 필요로 하거나 시각적 결함을 남기는 반면, ReSyncer는 3D 얼굴 동작을 예측하고 효과적으로 동기화하여 고품질의 립싱크 비디오를 생성합니다.

- **Technical Details**: ReSyncer는 스타일 기반 생성기(Style-based generator)를 재구성하고, 원칙 있는 스타일 주입 트랜스포머(Style-injected Transformer)를 사용하여 3D 얼굴 동작을 예측합니다. 이 프레임워크는 노이즈와 스타일 공간 내 정보 삽입 메커니즘을 다시 구성하여 모션과 외모를 통합 훈련합니다.

- **Performance Highlights**: ReSyncer는 오디오에 따라 고품질 립싱크 비디오를 생성하며, 가상 발표자와 공연자에 적합한 다양한 특성들을 지원합니다. 이는 빠른 개인 맞춤형 튜닝, 비디오 기반 립싱크, 말하기 스타일 전환, 얼굴 교체 등 여러 기능을 포함합니다.



### AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieva (https://arxiv.org/abs/2408.03282)
Comments:
          ECCV 2024

- **What's New**: 이번 연구는 인스턴스 레벨 이미지 검색 재순위를 메모리 효율성을 유지하면서 해결하는 문제를 다룹니다. 1KB의 메모리 사용 제한을 목표로 하여, 성능 향상보다 메모리 요구 사항과 성능 간의 균형을 우선시하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 Transformer 기반 아키텍처를 사용하여 각 이미지의 로컬 디스크립터(Local Descriptor) 간의 상호작용을 포착함으로써 이미지 간 유사도를 추정합니다. 모델의 독특한 특징 중 하나는 비대칭 유사도 추정을 가능하게 하는 것입니다. 데이터베이스 이미지는 쿼리 이미지보다 적은 디스크립터로 표현되며, 메모리 소비를 늘리지 않고 성능 향상을 가능하게 합니다. 테스트 단계에서는 서로 다른 수의 로컬 디스크립터에 적응하는 범용 모델을 소개합니다.

- **Performance Highlights**: 표준 벤치마크 결과, 제안된 접근 방식은 수작업 및 학습된 모델을 모두 능가하는 성능을 입증했습니다. 특히, 메모리 사용량을 고려하지 않는 현존 최첨단 방법들과 비교하여, 메모리 사용량을 크게 줄이면서도 우수한 성과를 거두었습니다. 코드 및 사전 학습된 모델은 공개되었습니다.



### Contrastive Learning for Image Complexity Representation (https://arxiv.org/abs/2408.03230)
- **What's New**: 위 논문에서는 이미지 복잡도(Complexity)를 평가하는 새로운 모델 MoCo v2 프레임워크 기반의 대조 학습(Contrastive Learning) 기법인 CLIC을 소개했습니다. 이 모델은 주석이 없는 데이터를 통해 이미지 복잡도를 학습하여, 주관적 편향과 높은 주석 비용을 피할 수 있게 합니다.

- **Technical Details**: CLIC은 대조 학습(Contrastive Learning)을 이용하여 이미지 복잡도 특징을 학습합니다. 이를 위해 MoCo v2 프레임워크를 사용하며, Random Crop and Mix (RCM)이라는 기법을 도입해 여러 스케일의 지역 이미지를 혼합하여 긍정 샘플을 생성합니다. 이 방법은 추가 데이터 없이도 훈련 데이터 셋을 확장하고 데이터 다양성을 증가시킵니다.

- **Performance Highlights**: 광범위한 실험 결과, CLIC은 최신 감독 학습 방법들과 비교해도 성능이 유사하거나 더 우수함을 보여주었습니다. 특히 적은 수의 샘플만 사용할 때는 감독 학습 방법보다 성능이 뛰어납니다. 또한, CLIC을 기반으로 한 객체 검출 및 의미론적 분할과 같은 컴퓨터 비전 작업의 성능이 향상되었음을 확인했습니다.



### Line-based 6-DoF Object Pose Estimation and Tracking With an Event Camera (https://arxiv.org/abs/2408.03225)
Comments:
          Accepted by IEEE Transactions on Image Processing,2024

- **What's New**: 이번 연구에서는 이벤트 카메라를 활용한 평면 또는 비평면 객체의 자세 추정 및 추적을 위한 라인 기반 방법을 제안합니다. 이벤트 카메라는 고정된 속도로 이미지를 캡처하는 기존 프레임 기반 카메라와는 달리, 픽셀별 밝기 변화를 비동기적으로 기록하는 '이벤트'를 생성합니다. 이 연구에서는 이벤트로부터 직접 객체의 라인을 추출하고, 글로벌 최적화 Branch-and-Bound(BnB) 접근 방식을 사용하여 초기 자세를 제공한 후 이벤트-라인 매칭을 통해 2D 이벤트와 3D 모델 간의 대응을 설정합니다. 또한 이벤트-라인 거리를 최소화하여 객체의 자세를 지속적으로 추적합니다.

- **Technical Details**: 이 방법은 이벤트 클러스터로부터 객체 라인을 직접 추출하고, BnB 알고리즘을 사용하여 초기 객체 자세를 추정합니다. 이후 이벤트-라인 매칭 전략을 사용하여 이벤트와 객체 모델의 투영 라인 간의 연관성을 설정합니다. 자세 최적화 및 추적 모듈에는 강건한 추정 알고리즘이 포함되어 있어 노이즈로부터 이벤트를 효과적으로 구분합니다. 이 방법을 검증하기 위해 이벤트 기반의 객체 움직임 데이터셋을 구성하고 수집했습니다.

- **Performance Highlights**: 제안된 방법은 합성 데이터를 기반으로 한 실험 및 자체 수집한 이벤트 데이터셋에서 최신 방법들과 비교하여 높은 강건성과 정확성을 입증했습니다. 이벤트 클러스터에서 라인을 추출하고 이벤트-라인 거리를 최소화하는 접근 방식은 기존의 방법들보다 우수한 성능을 보였습니다. 또한, 이 연구의 소스 코드는 공개되어 있어 다른 연구자들도 활용할 수 있습니다.



### IPAdapter-Instruct: Resolving Ambiguity in Image-based Conditioning using Instruct Prompts (https://arxiv.org/abs/2408.03209)
Comments:
          17 pages, 10 figures, Project page: this https URL

- **What's New**: 최신 연구는 Diffusion 모델이 이미지 생성의 최첨단을 지속적으로 확장하지만, 세부적인 제어가 어렵다는 문제를 해결함을 목표로 합니다. 이 논문에서는 IPAdapter-Instruct라는 새로운 방법을 통해 자연 이미지와 'Instruct' 프롬프트를 결합하여 동일한 조건 이미지를 다양한 방식으로 해석할 수 있도록 하는 접근법을 제안합니다.

- **Technical Details**: 기존의 ControlNet과 IPAdapter는 이미지 생성을 이미지 기반 조건(conditional)으로 수행하지만, 각각 단일 조건적 사후 확률(posterior)을 모델링하는 데 그칩니다. IPAdapter-Instruct는 다양한 'Instruct' 프롬프트와 결합하여 스타일 전환(style transfer), 객체 추출(object extraction) 등 다중 작업을 배우도록 합니다. 이 방법은 동일한 조건 이미지에 대해 다양한 해석을 가능하게 하는 융통성을 제공합니다.

- **Performance Highlights**: IPAdapter-Instruct는 전용 작업 모델(dedicated per-task models)에 비해 최소한의 품질 손실로 여러 작업을 효과적으로 학습할 수 있으며, 이는 여러 개의 어댑터를 훈련하고 사용하는 복잡성을 줄입니다.



### Personalizing Federated Instrument Segmentation with Visual Trait Priors in Robotic Surgery (https://arxiv.org/abs/2408.03208)
Comments:
          9 pages, 3 figures, under review

- **What's New**: 이 논문은 기존의 개인화 연합 학습(Personalized Federated Learning, PFL) 방법들이 다루지 않았던, 외관 다양성과 도구 모양 유사성에 기반해, 수술 도구 분할(Surgical Instrument Segmentation, SIS)를 강화하는 새로운 PFL 방법인 PFedSIS를 제안합니다. PFedSIS는 세 가지 핵심 요소를 포함합니다: 전역-개인화 분리(Global-Personalized Disentanglement, GPD), 외관 조절 개인화 강화(Appearance-Regulation Personalized Enhancement, APE), 모양 유사성 전역 강화(Shape-Similarity Global Enhancement, SGE). 이를 통해 각 사이트의 SIS 성능을 향상시킵니다.

- **Technical Details**: PFedSIS는 세 가지 주요 컴포넌트를 포함합니다: (1) GPD는 다중 헤드 셀프 어텐션의 헤드 별 및 채널 별 개인화를 처음으로 시도합니다. 이를 통해 각 사이트 간의 상호 차이를 점진적으로 활용하면서 고유한 외관 표현을 유지합니다. (2) APE는 외관 조절을 도입하며, 하이퍼네트워크를 통해 각 사이트의 개인화된 파라미터를 맞춤형으로 집계합니다. (3) SGE는 도구의 상호 모양 정보를 유지하고, 이미지 레벨에서 스타일 간 모양 일관성을 강화하며, 예측 레벨에서 각 사이트의 모양 유사성 기여도를 계산하여 전역 파라미터를 업데이트합니다.

- **Performance Highlights**: PFedSIS는 최신 방법들보다 Dice에서 +1.51%, IoU에서 +2.11%, ASSD에서 -2.79, HD95에서 -15.55 성능 향상을 달성합니다. 이는 세 곳의 공개 벤치마크 데이터셋 실험 결과로 검증되었습니다. 해당 코드와 모델은 추후 공개될 예정입니다.



### Efficient NeRF Optimization -- Not All Samples Remain Equally Hard (https://arxiv.org/abs/2408.03193)
- **What's New**: 본 연구는 Neural Radiance Fields (NeRF)의 효율적인 학습을 위해 온라인 하드 샘플 마이닝(online hard sample mining)을 제안합니다. NeRF 모델은 3D 재구성과 렌더링 작업에서 최첨단 품질을 제공하지만, 많은 계산 자원을 필요로 합니다. 제안된 방법은 네트워크 매개변수 업데이트에 거의 기여하지 않는 이미 학습된 샘플 처리를 줄임으로써 계산 시간과 메모리 사용량을 크게 절감합니다.

- **Technical Details**: 연구는 확률적 샘플링의 역전파(backward pass)가 최적화의 병목 현상이라는 점에 주목합니다. 따라서 첫 번째 순전파(forward pass)를 추론 모드(inference mode)에서 수행하여 하드 샘플을 저비용으로 검색하고, 이 샘플들만을 사용하여 NeRF 네트워크를 업데이트합니다. 이 과정에서 '피크 신호 대 잡음비(PSNR)' 수준을 동일하게 유지하면서 2배의 속도 향상과 약 40% 메모리 절감을 달성했습니다.

- **Performance Highlights**: Instant-NGP 모델에 제안된 방법을 적용한 결과, 기본 모델에 비해 View-synthesis 품질이 평균 1 dB 개선되었으며, 메모리 사용량이 약 40% 절감되었습니다. 또한 훈련 시간 동안 뛰어난 자원 활용을 통해 최소화된 계산 그래프 생성과 최적화 업데이트를 달성하였습니다.



### An Object is Worth 64x64 Pixels: Generating 3D Object via Image Diffusion (https://arxiv.org/abs/2408.03178)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구는 '객체 이미지(Object Images)'라는 새로운 3D 모델 생성 방법을 소개합니다. 이 방법은 3D 모형의 표면 지오메트리, 외관, 패치 구조를 64x64 픽셀 이미지로 캡슐화하여 복잡한 3D 형상을 관리 가능한 2D 형식으로 변환합니다. 이를 통해 다각형 메쉬(polygonal meshes)의 지오메트릭 및 의미론적 불규칙성을 처리할 수 있으며, Diffusion Transformers와 같은 이미지 생성 모델도 3D 형상 생성에 직접 사용할 수 있습니다.

- **Technical Details**: 객체 이미지는 표면 지오메트리를 포함한 메쉬를 12채널 이미지로 래스터화하여 변환합니다. 이 방식은 지오메트릭 및 의미론적 구조를 보존하며 PBR(PBR: Physically Based Rendering) 재질 생성을 자연스럽게 지원합니다. 연구진은 디자이너가 만든 UV-맵이 포함된 ABO 데이터 셋의 형상을 1024x1024 해상도의 객체 이미지로 변환하고, 이를 64x64 해상도로 다운샘플링하여 Diffusion Transformers로 모델링했습니다.

- **Performance Highlights**: 평가 결과, 생성된 형상은 최신 3D 생성 모델과 비교해 포인트 클라우드 FID(Fréchet Inception Distance) 측면에서 유사한 지오메트릭 품질을 달성했고, PBR 재질 생성을 자연스럽게 지원합니다.



### Dilated Convolution with Learnable Spacings makes visual models more aligned with humans: a Grad-CAM study (https://arxiv.org/abs/2408.03164)
Comments:
          Accepted at The Trustworthy AI Workshop, IJCAI 2024

- **What's New**: 최근 발표된 논문에서는 Learnable Spacing (DCLS)을 가진 Dilated Convolution을 통해 수용 영역을 확장하면서도 매개 변수 수를 증가시키지 않아 다수의 컴퓨터 비전 벤치마크에서 표준 및 확장 합성곱을 능가하는 성과를 보여주었습니다. 이번 연구는 DCLS가 모델의 해석 가능성을 증가시킨다는 점에서 더 나아가 정량적 평가를 수행했습니다. 특히, 인간의 시각적 전략과의 일치를 의미하는 해석 가능성에서 Spearman 상관관계를 이용하여 이를 측정했습니다.

- **Technical Details**: GradCAM을 활용하여 모델의 GradCAM 히트맵과 ClickMe 데이터셋 히트맵 간의 Spearman 상관관계를 분석했습니다. ResNet50, ConvNeXt (T, S, B), CAFormer, ConvFormer 및 FastViT (sa 24 및 36) 등의 8개 참조 모델에서 DCLS로 표준 합성곱 레이어를 대체했습니다. 또한, CAFormer와 ConvFormer 모델에서 Grad-CAM이 무작위 히트맵을 생성하는 문제를 Threshold-Grad-CAM을 도입하여 해결했습니다.

- **Performance Highlights**: DCLS로 대체한 8개의 모델 중 7개 모델에서 해석 가능성 점수가 향상되었습니다. 또한 CAFormer와 ConvFormer 모델에서도 Threshold-Grad-CAM을 도입하여 해석 가능성을 크게 향상시켰습니다. 이로써 DCLS가 성능뿐 아니라 해석 가능성에서도 유의미한 개선을 가져오는 것을 확인했습니다.



### User-in-the-loop Evaluation of Multimodal LLMs for Activity Assistanc (https://arxiv.org/abs/2408.03160)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)를 활용한 현대 다중 모드 추론 모델의 역량을 조사하여, 다단계 일상 활동을 위한 비전 기반 어시스턴트를 지원하는 데 초점을 맞추고 있습니다. 특히, 이러한 어시스턴트는 카메라 같은 센서로부터 시각적 히스토리를 인코딩하고, 미래 행동을 예측하며, 사용자가 있는 상황에서 재계획을 할 수 있어야 합니다. 이를 위해 우리는 Socratic 모델과 Vision Conditioned Language Models (VCLMs)라는 두 가지 주요 다중 모드 LLM 접근 방식을 비교했습니다.

- **Technical Details**: 첫째, 우리는 오프라인 데이터셋을 사용하여 비디오 기반 액션 예측 작업에서 Socratic 모델과 VCLMs의 성능을 벤치마킹했습니다. Socratic 접근 방식은 사전 학습된 비전-언어 모델을 사용하여 비주얼 히스토리를 텍스트로 변환합니다. 반면 VCLMs는 연속적인 임베딩 정보와 텍스트 토큰을 결합하여 시각 정보를 임베딩 합니다. 두 번째로, 우리는 18명의 참가자들이 Aria라는 egocentric observation device를 착용하고, 다중 모드 LLM의 도움을 받으며 다단계 요리 활동을 수행한 최초의 사용자 연구를 수행했습니다.

- **Performance Highlights**: 연구 결과에 따르면 Socratic 접근 방식이 오프라인 및 온라인 설정에서 모두 VCLMs보다 우수한 성능을 보였습니다. 특히 긴 시각적 히스토리를 다루는 작업에서 Socratic 모델이 더 효과적이었습니다. 본 오프라인 메트릭은 온라인 성능을 정확히 반영하지 않으며, 온라인 설정에서는 사용자의 실시간 수행 타임을 고려한 계획 변경이 필요합니다.



### Iterative CT Reconstruction via Latent Variable Optimization of Shallow Diffusion Models (https://arxiv.org/abs/2408.03156)
Comments:
          19 pages, 9 figures

- **What's New**: 최신 연구는 노이즈 제거 확산 확률 모델(Denoising Diffusion Probabilistic Model, DDPM)과 반복 CT 재구성을 결합한 새로운 CT 재구성 방법을 제안합니다. 기존 연구와는 달리, 이미지와 모델 파라미터 대신 확산 모델의 잠재 변수에 대한 충실도 손실을 최적화합니다. 특히, 해상도와 역과정(diffusion and reverse processes)을 낮추고, 역과정에서 추가된 노이즈를 고정시킴으로써 추론 과정에서 결정론적으로 만듭니다.

- **Technical Details**: 제안된 방법은 DDPM을 확산 모델로 사용합니다. 일반적인 DDPM 훈련 단계에서 입력 이미지의 노이즈를 점차적으로 증가시키고, 최종 시간 단계(t=T)에서는 노이즈만 남도록 합니다. 제안된 방법에서는 투시 데이터와 확산 모델의 출력을 일치시키는 일관성 손실(consistency loss)을 부과하며, 이는 CT 재구성 과정에서 수집된 환자의 해부학적 구조 정보를 유지하도록 돕습니다. 이러한 접근 방식은 이미지 품질과 구조 보존 사이의 절충을 제어하는 파라미터를 도입할 필요성을 제거합니다.

- **Performance Highlights**: 1/10 뷰 투시 데이터에서 제안된 방법은 반복 재구성(Iterative Reconstruction), 총 변동(Iterative Reconstruction with Total Variation), 단독 확산 모델(Diffusion Model Alone)을 포함한 기존 방법들보다 우수한 성능을 보였습니다. SSIM과 PSNR 등 정량 지표에서 뛰어난 결과를 나타냈습니다. 또한, 동일한 훈련된 확산 모델을 사용한 1/20 뷰 투시 데이터에서도 이미지 품질이 상당히 향상되었습니다. 제안된 방법은 CT 외에도 MRI, PET, SPECT 등 다른 영상 촬영 모달리티에 널리 적용될 수 있습니다.



### Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization (https://arxiv.org/abs/2408.03149)
Comments:
          In ACL-Findings 2024

- **What's New**: 빠르게 증가하는 멀티미디어 데이터를 효과적으로 요약하기 위해 텍스트와 이미지 모두를 결합한 다중 모드 요약(MSMO)의 중요성이 대두되고 있습니다. 이를 위해 저자들은 BART 기반의 Entity-Guided Multimodal Summarization (EGMS)라는 모델을 제안했습니다. EGMS는 이중 멀티모달 인코더와 게이팅 메커니즘을 사용하여 텍스트 및 이미지 데이터를 처리하며, 프리트레인된 비전-언어 모델에서 지식을 증류하여 이미지 선택을 개선합니다.

- **Technical Details**: EGMS 모델은 BART 프레임워크를 기반으로 구축되었습니다. 텍스트 중심 인코더를 수정하여 텍스트-이미지와 엔티티-이미지 정보를 동시에 처리하는 이중 멀티모달 인코더를 도입했습니다. 모델은 또한 게이팅 메커니즘을 통해 시각 데이터를 결합하여 개선된 텍스트 요약을 생성하며, 지식 증류를 통해 이미지를 선택합니다. 이 모든 과정을 통해 모델은 멀티모달 입력과 출력을 효과적으로 통합할 수 있습니다.

- **Performance Highlights**: 공개된 MSMO 데이터셋에서 EGMS 모델의 우수성이 입증되었습니다. 실험 결과는 엔티티 정보를 MSMO 문제에 통합하는 것이 필요함을 보여주었으며, 제안된 접근법이 기존 방법들보다 더 나은 성능을 나타냈습니다.



### SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection (https://arxiv.org/abs/2408.03143)
Comments:
          Accepted to ICPR 2024

- **What's New**: 슈퍼심플넷(SuperSimpleNet)은 기존의 심플넷(SimpleNet)을 개선한 혁신적인 모델로, 표면 결함 검출(Surface Defect Detection, SDD)에서 높은 성능과 신뢰성을 제공합니다. 이 모델은 정상 훈련 이미지로만 작동하는 비지도 학습 방식이면서도, 라벨링된 비정상 훈련 이미지가 있을 경우 이를 활용할 수 있습니다. 슈퍼심플넷은 네 가지 벤치마크 데이터셋에서 각각 지도 및 비지도 학습 맥락에서 최첨단 결과를 달성했습니다.

- **Technical Details**: 슈퍼심플넷은 사전 학습된 CNN(Convolutional Neural Network)을 사용하여 특징을 추출하고 이를 공통 잠재 공간에 맞춥니다. 사전 학습된 네트워크에서 추출된 특징들은 업스케일링과 풀링 과정을 통해 이웃 맥락을 캡슐화한 후, 특징 어댑터를 통해 적응됩니다. 또한, 이 모델은 바이너리화된 페를린 노이즈 마스크(Perlin noise mask)를 사용하여 특징 수준에서 합성 결함 영역을 생성하는 새로운 방법을 도입하여 성능을 크게 향상시켰습니다. 이 특징들은 이후 세그멘테이션 및 분류 모듈로 전달됩니다.

- **Performance Highlights**: 지도 학습에서는 SensumSODF와 KSDD2 데이터셋에서 각각 AUROC 97.8%와 탐지 AP 97.4%를 기록하며 최첨단 결과를 달성했습니다. 비지도 학습에서는 MVTec AD와 VisA 데이터셋에서 각각 AUROC 98.4%와 93.4%를 기록했습니다. 슈퍼심플넷은 초당 268개의 이미지를 처리할 수 있으며, 9.3 밀리초의 추론 시간을 자랑합니다.



### Benchmarking In-the-wild Multimodal Disease Recognition and A Versatile Baselin (https://arxiv.org/abs/2408.03120)
- **What's New**: 기존의 질병 분류 모델은 실험실 환경에서 촬영된 이미지를 인식하는 데 탁월한 성능을 보여주지만, 자연 환경에서의 성능은 현저히 떨어져 왔습니다. 이를 극복하기 위해 본 연구는 최대 질병 수를 포함하는 새로운 다중모드 식물 질병 인식 데이터셋을 제안합니다. 특히, 각 질병에 대한 텍스트 기반 설명을 추가하여 작은 클래스 간 차이(inter-class discrepancy)와 큰 클래스 내 변이(intra-class variance) 문제를 해결하고자 합니다.

- **Technical Details**: 새로운 PlantWild 데이터셋은 건강한 식물 이미지와 여러 질병 텍스트 설명을 포함하고 있습니다. 이미지 데이터는 다양한 인터넷 소스에서 크라우드 소싱되었으며, 각 클래스의 질병 설명은 Wikipedia와 GPT-3.5를 통해 얻어졌습니다. 데이터셋은 18,542개의 식물 이미지를 포함하며, 89개의 질병 타입이 포함되어 있습니다. 또한, CLIP을 활용하여 텍스트 설명과 시각 데이터를 다중 프로토타입으로 모델링하여 작은 클래스 간 차이와 큰 클래스 내 변이 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 제안된 베이스라인 모델은 여러 도전 과제를 제시하면서도 훈련 없이도 질병 인식이 가능하도록 합니다. 본 연구는 다양한 실험을 통해 PlantWild 데이터셋에서의 성능을 평가하고, 기존의 최신 기술들과 비교하여 우수한 성과를 보였습니다.



### Prototype Learning for Micro-gesture Classification (https://arxiv.org/abs/2408.03097)
Comments:
          1st Place in Micro-gesture Classification in MiGA at IJCAI-2024

- **What's New**: IJCAI 2024 MiGA 챌린지의 미세제스처 분류(micro-gesture classification) 트랙에서 1위를 기록한 HFUT-VUT 팀의 솔루션을 소개합니다. 미세제스처 분류는 일반적인 행동 인식보다 더 세밀하고 미세한 신체 움직임을 인식하는 작업으로서, 내재된 복잡성 및 높은 유사성 때문에 기존의 기술들로는 어려운 과제입니다.

- **Technical Details**: 우리 팀은 이 문제를 해결하기 위해 두 가지 혁신적인 모듈을 사용했습니다. 첫째, 교차 모달 융합 모듈(cross-modal fusion module)로 서로 다른 모달리티 간의 상관관계를 탐구합니다. 둘째, 원형 세분화 모듈(prototypical refinement module)로 각 미세제스처 카테고리의 프로토타입을 정의하고, 모호한 샘플들을 보정합니다. 이 두 모듈은 PoseConv3D 백본에 통합되어 RGB 및 Skeletal 데이터를 처리합니다.

- **Performance Highlights**: 제안된 방법은 iMiGUE 테스트 세트에서 Top-1 정확도 70.254를 달성하여 큰 성공을 거두었습니다. 작년 우승 팀의 성능을 6.13%나 초과했습니다. 이러한 실험 결과는 우리 방법이 미세제스처의 미세한 변화를 효과적으로 캡처할 수 있음을 시사합니다.



### BodySLAM: A Generalized Monocular Visual SLAM Framework for Surgical Applications (https://arxiv.org/abs/2408.03078)
Comments:
          15 pages, 7 figures

- **What's New**: 이 연구는 기존의 문제를 해결하기 위해 혁신적인 딥러닝 기반 SLAM 접근법을 소개했습니다. 새로운 방법론은 CycleGAN 아키텍처를 바탕으로 한 무감독 방식의 Monocular Pose Estimation Module (MPEM)과 혁신적인 Zoe 아키텍처를 활용한 Monocular Depth Estimation Module (MDEM), 그리고 3D Reconstruction Module (3DM)을 포함합니다. 특히, Zoe를 통합한 MDEM은 현존하는 타 알고리즘보다 뛰어난 성능을 보이며, MPEM은 최단 추론 시간을 자랑합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성됩니다: 1) MPEM은 시계열 프레임 간 카메라 포즈를 추정하는 무감독 방법을 사용합니다, 2) MDEM은 여러 이미지 컨텍스트에서 우수한 일반화 성능을 보이는 Zoe 아키텍처를 기반으로 합니다, 3) 3DM은 점 구름(point clouds)을 생성하고 포즈를 정제하며 이를 체적 표현으로 변환하는 다단계 과정을 수행합니다.

- **Performance Highlights**: 제안된 방법론은 Hamlyn, EndoSLAM, SCARED 등 세 개의 공개 데이터셋을 활용하여 엄격하게 평가되었으며, EndoSFMLearner와 EndoDepth 두 가지 기존의 최첨단 방법과 비교 평가되었습니다. 그 결과 Zoe를 통합한 MDEM은 깊이 추정 알고리즘에서 뛰어난 성능을 보였으며, MPEM은 경쟁력 있는 성능과 가장 짧은 추론 시간을 기록했습니다. 이 접근법은 복강경, 위내시경, 대장내시경 등의 다양한 내시경 수술 시나리오에서 그 우수성을 입증했습니다.



### SCOPE: A Synthetic Multi-Modal Dataset for Collective Perception Including Physical-Correct Weather Conditions (https://arxiv.org/abs/2408.03065)
- **What's New**: SCOPE 데이터셋은 최초의 합성 멀티모달( multi-modal) 집합적 인지( collective perception) 데이터셋으로 현실적인 LiDAR 및 카메라 모델을 포함하고 있으며, 물리적으로 정확한 날씨 시뮬레이션을 제공한다. 이 데이터셋은 40개 이상의 다양한 시나리오와 17,600 프레임을 포함하여, 자율주행 기술의 개발 및 테스트를 가능하게 한다.

- **Technical Details**: SCOPE 데이터셋은 현실적인 LiDAR 모델과 고정 상태의 LiDAR를 포함하고 있으며, 강우, 안개 등 다양한 기상 조건을 시뮬레이션 할 수 있다. 데이터셋은 업데이트된 LiDAR 모델을 사용하여 기존 CARLA LiDAR 모델의 한계를 극복하며, 카를스루에와 튀빙겐의 두 가지 디지털 트윈맵을 포함하여 더욱 다양한 환경을 제공한다. 데이터를 생성하기 위해 CARLA 0.9.14와 SUMO 교통 시뮬레이터를 사용하며, 센서 데이터는 시간에 맞춰 HDF5 컨테이너에 저장된다.

- **Performance Highlights**: SCOPE는 17,600 프레임과 40개 이상의 시나리오, 최대 24개의 협력 에이전트( vehicles)와 인프라 센서를 포함하여 다양한 환경에서 자율 주행 알고리즘을 테스트할 수 있는 기회를 제공한다. 특히, 보행자와 자전거를 포함한 취약한 도로 사용자들(Vulnerable Road Users, VRU)을 현실적인 시나리오에 포함시킨다.



### MGFs: Masked Gaussian Fields for Meshing Building based on Multi-View Images (https://arxiv.org/abs/2408.03060)
- **What's New**: 이번 논문에서는 건물 표면 재구성을 위한 새로운 프레임워크인 Masked Gaussian Fields (MGFs) 소개되었습니다. 이 프레임워크는 시간 효율성과 높은 정확도를 추구하면서 건물의 정확한 표면을 재구성합니다.

- **Technical Details**: MGFs 프레임워크는 EfficientSAM과 COLMAP을 사용하여 건물의 다중 레벨 마스크(Multi-level masks) 및 대응되는 마스크된 포인트 클라우드(Point clouds)를 생성합니다. 그 다음, 두 가지 혁신적인 손실 함수(loss functions)를 통합하여 마스크된 가우시안 필드(Gaussian fields)를 훈련합니다: 건물 영역을 재구성하는 다중 레벨 인지 마스크 손실(Multi-level perceptual masked loss)과 서로 다른 마스크 사이의 경계 세부사항을 강화하는 경계 손실(Boundary loss)입니다. 마지막으로 마스크된 가우시안 구체(Gaussian spheres)에 기반한 4면체 표면 메쉬(tetrahedral surface mesh) 추출 방법을 개선했습니다.

- **Performance Highlights**: UAV 이미지에 대한 종합적인 실험 결과, 전통적인 방법과 여러 NeRF 및 Gaussian 기반의 최신 기술(SOTA) 솔루션과 비교했을 때 우리의 접근 방식이 건물 표면 재구성의 정확도와 효율성을 크게 향상시켰습니다. 특히, 신형 뷰 합성(novel view synthesis)에서도 추가적인 성능 향상을 보였습니다.



### Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression (https://arxiv.org/abs/2408.03046)
Comments:
          Accepted by ITSC 2024. Code is publicly available at: this https URL

- **What's New**: 새로운 경량화 모델 압축 프레임워크인 CPD(Comb, Prune, Distill)을 소개합니다. 이 프레임워크는 모델과 과제에 의존하지 않고도 효율적으로 모델을 압축하고, 지식 증류(Knowledge Distillation)를 통해 학습된 정보를 보존합니다.

- **Technical Details**: CPD 프레임워크는 세 가지 주요 단계로 구성됩니다: 'Combing', 'Pruning', 'Distilling'. 'Combing' 과정은 계층 간 종속성 문제를 해결하여 아키텍처 독립성을 보장합니다. 'Pruning' 단계에서는 중요도 점수(importance scoring metrics)를 사용해 매개변수를 적응적으로 제거합니다. 마지막으로 'Distilling' 과정에서는 지식 증류를 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: CPD 프레임워크는 ImageNet과 ADE20K 데이터셋을 사용하여 이미지 분류와 의미론적 분할 작업에서 실험을 진행했습니다. 이미지 분류에서 최대 4.3배의 속도 향상과 1.8%의 정확도 손실을 달성했으며, 의미론적 분할에서는 최대 1.89배의 속도 향상과 5.1%의 mIoU 손실을 기록했습니다.



### Targeted Visual Prompting for Medical Visual Question Answering (https://arxiv.org/abs/2408.03043)
Comments:
          Accepted at the MICCAI AMAI Workshop 2024

- **What's New**: 의료 영상 질문 응답(Visual Question Answering, VQA) 분야에서 다중모드 대형 언어 모델(Multimodal Large Language Model, MLLM)을 활용한 새로운 접근법이 제안되었습니다. 이 논문은 MLLM에 지역 기반 질문 기능을 추가하는 '목표 지향 시각 프롬프트(Targeted Visual Prompting)' 방법을 소개하며, 이를 통해 모델의 시각적 이해도를 향상시킬 수 있음을 여러 데이터셋을 통해 증명했습니다.

- **Technical Details**: 기존의 Med-VQA 모델은 시각적 정보와 텍스트 정보를 독립적으로 처리한 후 이를 융합하여 응답을 생성했습니다. 그러나 이 논문에서는 자동 회귀 방식으로 응답을 생성하는 LLM 기반 접근법을 사용했습니다. 특히, 시각적 인코더와 프로젝션 층을 통해 시각적 임베딩을 LLM의 입력 공간으로 투사하여 다중모드 LLM을 구현했습니다. 시각적 프롬프트는 다섯 가지로 구성됩니다: 모델 명령(Instruction), 시각적 맥락(Context), 텍스트 전처리(Textual Prefix), 자른 이미지 영역(Cropped Region), 질문(Question)입니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋을 통해 다양한 성능 실험을 거쳤으며, 기존 방법들에 비해 명확한 성능 향상을 보여주었습니다. 또한, 새로운 모델 파라미터를 추가하지 않고도 이러한 성능 향상을 달성했습니다.



### Nighttime Pedestrian Detection Based on Fore-Background Contrast Learning (https://arxiv.org/abs/2408.03030)
- **What's New**: 기존의 채널 어텐션 메커니즘(channel attention mechanisms) 연구에서 배경 정보의 중요성이 종종 간과되는 문제를 지적하며, 이를 해결하기 위해 Fore-Background Contrast Attention(FBCA)을 제안하였습니다. 이 연구는 저조도 환경에서 단일 스펙트럼 야간 보행자 탐지 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: FBCA는 두 가지 주요 속성을 가지고 있습니다. 첫째, 채널 기술자(channel descriptors)는 글로벌 공간 특징 정보와 원거리 의존성을 형성합니다. 둘째, 배경 정보의 통합은 저조도 보행자 특징에 집중하는 채널과 배경 정보에 집중하는 채널 간의 구별을 강화합니다. 이를 통해 더 높은 수준의 의미론적(semiotic) 및 공간적(spatial) 정확도를 가진 채널 기술자를 획득할 수 있습니다.

- **Performance Highlights**: FBCA는 NightOwls 및 TJU-DHD-pedestrian 데이터셋에서 단일 스펙트럼 야간 보행자 탐지에 있어 기존 방법보다 뛰어난 성능을 보여주었습니다. 또한, 다중 스펙트럼 LLVIP 데이터셋에서도 성능 향상을 이끌어냈습니다. 이러한 결과는 채널 어텐션 메커니즘에 배경 정보를 통합함으로써 야간 시나리오에서 조명 요인으로 인한 탐지 성능 저하를 효과적으로 완화할 수 있음을 나타냅니다.



### CKNN: Cleansed k-Nearest Neighbor for Unsupervised Video Anomaly Detection (https://arxiv.org/abs/2408.03014)
- **What's New**: 이번 연구에서는 무감독 비디오 이상 탐지(Unsupervised Video Anomaly Detection, UVAD) 문제를 다룹니다. UVAD는 테스트 비디오에서 비정상 이벤트를 감지하며, 레이블이 없는 비디오를 학습 데이터로 사용합니다. 특히 '이상 클러스터' 문제를 해결하기 위해 Cleansed k-Nearest Neighbor (CKNN)라는 새로운 방법을 제시합니다. CKNN은 k-최근접 알고리즘을 사용하여 학습 데이터에서 비정상 클러스터를 제거하는 방식으로 작동합니다.

- **Technical Details**: CKNN는 특징 공간에서 비정상 클러스터를 명시적으로 제거함으로써 훈련 데이터세트를 정화(cleansing)합니다. 이 방법은 NN 검색(Nearest Neighbor search)을 통해 비정상 이벤트를 강력하게 탐지할 수 있습니다. 또한, CKNN은 기존 state-of-the-art UVAD 방법과 비교하여 성능을 크게 향상시켰으며, 일부 OCVAD 방법과 유사한 성능을 달성했습니다. CKNN의 주요 기여점으로는 처음으로 UVAD에서의 '이상 클러스터' 문제를 식별하고 이를 극복하기 위한 해결책을 제시한 것입니다.

- **Performance Highlights**: CKNN은 여러 벤치마크 데이터셋에서 기존 최고 성능의 UVAD 방법을 최대 8.5% (AUROC 측정 기준)까지 초과합니다. CKNN의 성능은 비정상 없는 데이터를 사용해 훈련한 state-of-the-art OCVAD 방법과도 유사한 수준입니다. 또한, 하이퍼파라미터에 대한 민감도 테스트를 통해 제안된 방법의 강건성을 검증하였으며, CKNN의 실행 시간 분석 결과 실시간 응용에 적합한 최소한의 계산 오버헤드를 가지는 것으로 나타났습니다.



### Dual-path Collaborative Generation Network for Emotional Video Captioning (https://arxiv.org/abs/2408.03006)
Comments:
          Acccepted by ACM Multimedia 2024, oral

- **What's New**: 감정 기반 비디오 자막 생성(Emotional Video Captioning, EVC) 과제에서 전통적인 비디오 자막 생성 방법이 간과하는 미묘하고 애매한 감정적 단서를 효과적으로 포착하는 새로운 이중 경로 협력 생성 네트워크(Dual-path Collaborative Generation Network)를 소개합니다. 이 네트워크는 비디오 내 감정의 동적 변화에 반응하며, 감정과 사실적 내용을 균형있게 캡션에 반영합니다.

- **Technical Details**: 제안된 네트워크는 두 개의 경로로 구성됩니다. 첫 번째는 '동적 감정 인식 경로'(dynamic emotion perception path)로, 이는 비주얼 기능과 역사적 캡션 기능을 집계하여 전역 감정 단서를 요약하고 각 단계에서 재구성할 감정 단서를 동적으로 선택합니다. 두 번째는 '적응적 캡션 생성 경로'(adaptive caption generation path)로, 감정 강도를 추정하고 필요한 단계에서 감정 관련 단어를 생성하여 사실적 내용과 감정 단서를 균형있게 반영합니다. 이 경로들은 협력 학습(collaborative learning)을 통해 서로 촉진됩니다.

- **Performance Highlights**: 세 개의 도전적인 공개 데이터셋(EVC-MSVD, EVC-VE, EVC-Combined)에서 광범위한 실험을 통해 제안 방법의 우수성을 입증했습니다. 구체적으로, EVC-VE 데이터셋에서 감정 정확도(emotion accuracy)는 +7.2%, CIDEr는 +6.8%, CFS는 +6.5% 향상되었습니다.



### Multitask and Multimodal Neural Tuning for Large Models (https://arxiv.org/abs/2408.03001)
- **What's New**: 최근 몇 년간 대규모 멀티모달 모델은 다양한 영역에서 인상적인 성능을 보여주고 있습니다. 하지만 여러 멀티모달 작업을 동시에 수행하는 데에는 여전히 큰 도전이 남아있습니다. 이를 해결하기 위해, 신경 튜닝(Neural Tuning)이라는 새로운 튜닝 방법을 도입했습니다. 이 방법은 다양한 멀티모달 작업(예: Reasoning Segmentation, Referring Segmentation, Image Captioning, Text-to-Image Generation)을 동시에 처리할 수 있도록 설계되었습니다. 신경 튜닝은 인간 두뇌의 희소 분산 표현(Sparse Distributed Representation)을 모방하여 각 작업에 대해 특정 신경 집합만을 활성화합니다. 또한 MMUD라는 새로운 벤치마크를 제시하며, 이 벤치마크는 각 샘플이 여러 작업 레이블로 주석이 달려있습니다.

- **Technical Details**: 신경 튜닝 전략은 두 가지 주요 개념을 도입합니다. 첫째, 모든 작업을 통합된 토큰 방식으로 공식을 세우며, 새로운 작업을 추가할 때 단순히 새로운 작업 토큰을 추가하면 됩니다. 둘째, 희소 작업 네트워크(Sparse Task Network)를 도입하여 다양한 작업에 대해 특정 신경만을 활성화하여 희소 분산 표상을 모방합니다. 이는 각 작업에 대해 교차 주의 메커니즘이나 복잡한 디코더를 추가하지 않고도 간단하고 효율적인 멀티테스크 튜닝을 가능하게 합니다.

- **Performance Highlights**: MMUD 벤치마크에서 프리트레이닝된 대규모 모델을 신경 튜닝을 활용해 조정하는 과정을 통해, 다중 작업을 효율적으로 처리할 수 있는 성능을 달성했습니다. 특히 Reasoning Segmentation, Referring Segmentation, Image Captioning, Text-to-Image Generation와 같은 다양한 작업에서 최첨단 성능을 입증했습니다.



### DreamLCM: Towards High-Quality Text-to-3D Generation via Latent Consistency Mod (https://arxiv.org/abs/2408.02993)
Comments:
          15 pages, 9 figures, ACM MM 2024

- **What's New**: 최근 SDS(Specialized Diffusion Sampling) 방법이 등장함에 따라 텍스트-3D 생성 작업이 급격히 발전하였으나, 이러한 방법은 항상 품질이 낮은 3D 객체를 생성하는 문제를 가지고 있었습니다. 이 논문에서는 DreamLCM이라는 새로운 접근 방식을 제안합니다. 이는 라텐트 일관성 모델(Latent Consistency Model, LCM)을 통합하여 고품질의 일관된 가이던스를 제공하여, 더 정확하고 세밀한 3D 객체 생성을 가능하게 합니다.

- **Technical Details**: DreamLCM는 LCM의 강력한 이미지 생성 기능을 활용하여 일관된 고품질의 가이던스를 생성합니다. 이를 통해 단일 단계 추론으로도 높은 품질의 3D 객체 생성을 가능하게 합니다. 또한, 두 가지 전략을 제안하여 생성 품질을 더욱 향상시킵니다. 첫째, Euler 솔버를 사용하여 가이던스 분포를 보정하는 '가이던스 보정 전략'을 제안합니다. 둘째, 3D 모델의 기하학적 형태와 외관을 최적화하는 '이중 시간 단계 전략'을 도입합니다.

- **Performance Highlights**: 실험 결과, DreamLCM은 생성 품질과 훈련 효율성에서 최첨단 성과를 달성했습니다. 이 모델은 높은 세부 사항을 유지하면서 고품질의 3D 객체를 생성하며, 엔드 투 엔드 방식으로 훈련이 가능해 훈련 비용을 절감하고 보다 간소화된 훈련 파이프라인을 유지하였습니다.



### Diffusion Model Meets Non-Exemplar Class-Incremental Learning and Beyond (https://arxiv.org/abs/2408.02983)
- **What's New**: 새로운 논문에서는 NECIL(비모델 클래스 증강 학습)에서 나타나는 큰 분포 차이 문제를 해결하기 위해 **DiffFR**라는 효과적이고 간단한 방식을 제안합니다. 이 방식은 기존 클래스 샘플을 저장하지 않고도 **catastrophic forgetting**을 방지하려는 목표를 가지고 있습니다. DiffFR은 **diffusion models**의 강력한 생성 능력을 활용하여 실제 특징과 유사한 클래스 대표 특징을 생성함으로써, 기존 클래스에 대한 지식을 보존할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 기술적으로 DiffFR은 다음과 같은 요소로 구성됩니다: 1) 초기 일반화된 특징을 위해 **Siamese 기반 자가 지도 학습**을 사용합니다. 2) 클래스 대표 특징을 생성하기 위해 **diffusion 모델**을 설계하여, 실제 특징과 매우 유사한 특징을 생성합니다. 3) **프로토타입 교정**을 도입하여 diffusion 모델이 분포의 전체가 아닌 분포 모양을 학습하도록 유도합니다. 이를 통해, 특징 분포의 중앙 집중성을 강화하고, 클래스 인식 특징 생성을 강화할 수 있습니다.

- **Performance Highlights**: 공개된 데이터셋에서 광범위한 실험을 통해 DiffFR이 현존하는 최첨단 NECIL 방법들을 평균 3.0% 성능 향상으로 능가함을 보여주는 주요 성과를 기록했습니다. DiffFR은 CIFAR-100, TinyImageNet, ImageNet-Subset 등 다양한 벤치마크에서 뛰어난 성능을 입증하며, 새롭게 부상하는 클래스에 대한 일반화 성능도 향상되었습니다. 이는 NECIL 방법들 중 새로운 기준점을 세웠습니다.



### Sample-agnostic Adversarial Perturbation for Vision-Language Pre-training Models (https://arxiv.org/abs/2408.02980)
Comments:
          13 pages, 8 figures, published in ACMMM2024

- **What's New**: 최근 VLP(Vision-Language Pre-training) 모델의 보안 취약성에 주목한 연구입니다. 기존 연구는 샘플별로 고유한 교란을 생성하는 방식이었으나, 본 연구는 처음으로 모든 이미지에 적용 가능한 범용적인 샘플 비의존적 교란(universal, sample-agnostic perturbation)을 제안합니다.

- **Technical Details**: 본 연구는 두 가지 설계 전략으로 구체화됩니다. 첫째로, 샘플 포인트를 선형 분류기의 결정 경계(linear classifiers' decision boundaries) 너머로 이동시키는 방식을 탐구하여, top k 정확도(metric)에 기반한 성공적인 공격을 위한 알고리즘을 개선합니다. 두 번째로, 이미지 임베딩이 텍스트로 구성된 결정 경계를 넘도록 하여 텍스트 투 이미지 및 이미지 투 텍스트 전송 간 교차 공격을 사용합니다. 이를 통해 고유한 악의적인 방향을 찾아내어 VLP 모델의 검색 성능을 저하시킵니다.

- **Performance Highlights**: 다양한 데이터셋과 VLP 모델에 적용한 실험에서 본 연구가 제안한 범용 교란이 뛰어난 적응력을 보였습니다. 실험 결과, 데이터, 태스크, 모델 전반에 걸쳐 일관된 성능 저하를 입증하며, 프리트레인된 인코더에서 다운스트림 애플리케이션으로의 전이 가능성을 보여주었습니다.



### Fast Point Cloud Geometry Compression with Context-based Residual Coding and INR-based Refinemen (https://arxiv.org/abs/2408.02966)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 연구에서는 불규칙한 배치의 점 구름(point cloud)을 효과적으로 압축하는 새로운 방법을 제안했습니다. KNN 방법으로 원시 표면 점의 이웃 관계를 결정하고, 이를 통해 3D 점의 잠재 기능(latent features)이 산술 코딩(arithmetic coding)으로 압축됩니다. 또한, 비학습 기반의 기본 레이어가 주 구조를, 학습 기반의 정밀화 레이어가 세부 사항을 보존하는 이중 레이어 아키텍처를 도입하여 모델 복잡성과 코딩 지연 시간을 크게 줄였습니다.

- **Technical Details**: 본 방법은 Context-based Residual Coding과 Implicit Neural Representation (INR) 기반 정밀화 방식을 결합한 CRCIR(compression system)를 사용합니다. 기본 레이어에서 매우 낮은 복잡도로 거친 구조를 재구성하고, 정밀화 레이어에서 KNN을 사용하여 지역적 상호관계를 특징지어 잠재 기능 사이의 상관관계를 활용합니다. 또한, INR을 정밀화 레이어에 통합하여 임의 밀도의 점을 샘플링할 수 있게 하였습니다.

- **Performance Highlights**: 제안된 CRCIR 방식은 기존 최첨단 방법인 3QNet 대비 모델 복잡성과 코딩 지연 시간을 두 자릿수 줄였으며, 우수한 비율-왜곡(rate-distortion) 성능을 달성하였습니다. 또한, 이 방법은 원시 점 구름의 비규칙성을 극복하고, 임의의 확장 비율에서 유연하게 작동할 수 있는 첫 압축 매커니즘입니다.



### Online Temporal Action Localization with Memory-Augmented Transformer (https://arxiv.org/abs/2408.02957)
Comments:
          Accepted to ECCV 2024, Project page: this https URL

- **What's New**: 기존 방법의 단점을 보완하기 위해, 스트리밍 동영상에서 여러 행동 인스턴스를 식별하는 작업인 온라인 시간적 행동 로컬라이제이션(On-TAL)에 적합한 새로운 아키텍처인 Memory-Augmented Transformer(MATR)를 제안합니다. MATR은 메모리 큐(memory queue)를 활용하여 과거 세그먼트 특징을 선택적으로 저장하고 이를 이용하여 장기적인 컨텍스트를 활용합니다. 또한, 새로운 행동 로컬라이제이션 방법을 도입하여 현재 입력 세그먼트를 관찰해 진행 중인 행동의 종료 시간을 예측하고, 메모리 큐를 통해 시작 시간을 추정합니다.

- **Technical Details**: MATR의 핵심은 메모리 큐로, 과거의 세그먼트 특징을 선택적으로 저장하여 모델이 장기적인 컨텍스트를 이용할 수 있게 합니다. 두 Transformer 디코더를 채택하여 하나는 종료 시간 탐지에, 다른 하나는 시작 시간 탐지에 사용합니다. 학습 가능한 쿼리를 디코더에 입력하여 Transformer의 주의 메커니즘(attention mechanism)을 통해 행동 경계를 지역화하도록 학습합니다. 행동 분류와 로컬라이제이션을 분리하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 두 개의 데이터셋, THUMOS14와 MUSES에서 기존 방법보다 뛰어난 성능을 보였습니다. 우리 모델은 온라인 TAL 설정에서 가장 높은 성능을 달성했으며 오프라인 TAL 방법들과 비교해도 비슷한 성능을 보였습니다. 또한, 광범위한 ablation 연구를 통해 제안된 각 구성 요소의 기여도를 입증했습니다.



### WWW: Where, Which and Whatever Enhancing Interpretability in Multimodal Deepfake Detection (https://arxiv.org/abs/2408.02954)
Comments:
          4 pages, 2 figures, 2 tables, Accepted as Oral Presentation at The Trustworthy AI Workshop @ IJCAI 2024

- **What's New**: 현재 멀티모달 딥페이크 감지를 위한 모든 벤치마크는 다양한 생성 기법을 사용하여 전체 프레임을 조작하여 비디오 레벨의 분류 정확도를 94% 이상으로 높였습니다. 그러나 이러한 벤치마크는 실제 시나리오에서 프레임별 변경이 어려운 동적 딥페이크 공격을 탐지하는 데 어려움을 겪습니다. 이러한 한계를 해결하기 위해 우리는 영상 및 오디오 내의 조작된 세그먼트를 식별하고 딥페이크의 출처에 대한 통찰을 제공하는 새로운 클립 레벨 평가 벤치마크인 FakeMix를 소개합니다.

- **Technical Details**: FakeMix는 클립 레벨의 오디오-비디오 멀티모달 딥페이크 감지 벤치마크로, 기존 벤치마크와 달리 특정 조작된 세그먼트를 정밀하게 평가합니다. 이를 위해 우리는 딥페이크 감지 모델의 강건성을 평가하기 위한 새로운 평가 지표인 Temporal Accuracy (TA)와 Frame-wise Discrimination Metric (FDM)를 개발했습니다. 이 지표들을 사용하여 기존 모델을 다양한 딥페이크 벤치마크, 특히 FakeMix를 기준으로 평가한 결과, 비디오 레벨에서는 94.2%의 평균 정밀도(AP)를 달성했으나 클립 레벨 평가에서는 TA와 FDM 지표들이 각각 53.1%와 52.1%로 급락했습니다.

- **Performance Highlights**: FakeMix 벤치마크를 통한 평가 결과, 기존 모델은 전체 비디오 레벨에서는 높은 정확도를 보였지만, 클립 레벨에서는 정확도가 크게 떨어졌습니다. 이는 딥페이크 감지 모델이 세부 조작된 영역을 식별하는 데 아직 한계를 가지고 있음을 시사합니다. 이러한 평가 지표의 도입이 모델의 세밀한 조작 감지 성능을 향상시키는 데 필수적임을 보여줍니다.



### Segmenting Small Stroke Lesions with Novel Labeling Strategies (https://arxiv.org/abs/2408.02929)
- **What's New**: 이 연구에서는 새로운 두 가지 접근법인 Multi-Size Labeling(MSL)과 Distance-Based Labeling(DBL)를 제안합니다. 이 두 접근법은 다양한 네트워크에 무리 없이 통합이 가능하며, 특히 작은 뇌졸중 병변의 분할 정확도를 높이는 데 목적을 두고 있습니다. MSL은 병변의 부피에 따라 병변 마스크를 다양한 카테고리로 나누고, DBL은 병변 경계 부분을 강조합니다.

- **Technical Details**: MSL과 DBL은 이진 분할 마스크를 병변 부피(병변의 부피에 따른 다중 크기 라벨링) 및 비병변 영역까지의 거리(경계 기반 라벨링)에 따라 여러 클래스(Classes)로 분류하는 라벨링 전략입니다. 이 방법들은 U-Net 구조의 feature extractor (특징 추출기)를 수정하지 않고도 적용 가능하며, 네트워크가 병변 유형을 구별할 수 있도록 도와줍니다. 각각 MSL은 병변 부피에 기반한 분할을, DBL은 병변과 비병변 경계의 강조를 목표로 합니다.

- **Performance Highlights**: Anatomical Tracings of Lesions After Stroke (ATLAS) v2.0 데이터셋을 활용한 실험 결과, MSL과 DBL을 결합한 앙상블(ensemble) 접근법이 2022 MICCAI ATLAS 챌린지에서 최고 성적을 기록한 모델과 비교하여 리콜(recall)에서 3.6%와 3.7%, F1 점수에서 2.4%와 1.5%, 그리고 다이스 점수(Dice score)에서 1.3%와 0.0% 더 높은 성능을 보였습니다. 특히, mini-lesion subset에서는 단일 MSL 모델이 이전 최고 앙상블 전략을 F1과 다이스 점수에서 각각 1.0%와 0.3% 앞섰습니다.



### Evaluation of Segment Anything Model 2: The Role of SAM2 in the Underwater Environmen (https://arxiv.org/abs/2408.02924)
- **What's New**: Meta에서 새롭게 출시한 Segment Anything Model 2 (SAM2)는 이전 버전에 비해 실행 속도와 분할 정확도가 크게 향상되었습니다. 이 보고서는 SAM2를 해양 과학 분야에서 평가하고 UIIS 및 USIS10K와 같은 수중 인스턴스 분할 벤치마크 데이터셋에서의 성능을 탐구합니다.

- **Technical Details**: SAM2는 GT Bbox (ground truth bounding box)를 프롬프트로 사용할 때 탁월한 성능을 보였습니다. 그러나 자동 모드에서 포인트(점) 프롬프트를 사용할 때 SAM2의 능력은 크게 저하되었습니다. 실험은 NVIDIA GeForce RTX 4090에서 수행되었으며, 모델의 속도는 초당 프레임(FPS)으로 평가되었습니다. 프롬프트는 1 Point, 3 Point, GT Bbox 세 가지 유형으로 구성되었습니다.

- **Performance Highlights**: UIIS 데이터셋 테스트에서 SAM2는 GT Bbox 프롬프트를 사용하여 mAP에서 4.8 AP 향상과 약 5배 더 빠른 속도를 보여주었습니다. SAM2-Hiera-Tiny 모델은 EfficientSAM-ViT-Tiny에 비해 mAP에서 7.6 AP 향상을 달성했으나 속도는 21% 느렸습니다. 자동 모드에서는 SAM2의 성능이 크게 저하되었으며, USIS10K 데이터셋에서는 더 많은 포인트 프롬프트를 입력할 시 속도가 느려지는 현상이 나타났습니다.

- **Conclusion**: SAM2의 성능은 입력 프롬프트의 유형과 품질에 크게 의존하며, 자동 추론 시 성능 저하가 두드러집니다. 신뢰할 수 있는 객체 탐지 모듈을 프롬프트 생성기로 설계하는 것이 향후 연구의 초점이 될 것입니다. 또한, 수중 비디오 인스턴스 분할 데이터셋의 부족으로 인해 이 작업에 대한 SAM2의 성능은 평가되지 않았으나, 수중 2D 인스턴스 분할에서의 우수한 성능을 기반으로 SAM2는 강력한 주석 도구가 될 가능성이 큽니다.



### Pose Magic: Efficient and Temporally Consistent Human Pose Estimation with a Hybrid Mamba-GCN Network (https://arxiv.org/abs/2408.02922)
- **What's New**: 최신 3D 인간 자세 추정(3D Human Pose Estimation, 3D HPE) 연구에서는 Transformer 기반 방법이 주류를 이루지만, 정확도와 계산 효율성 사이에 딜레마를 겪고 있습니다. 이를 해결하기 위해, 최근의 상태 공간 모델(State Space Models, SSM)을 활용하여 Mamba를 사용해 고품질의 효율적인 장거리 모델링을 도입했습니다. 하지만 Mamba는 관절 간의 국부적인 의존성을 충분히 활용하는 데에 어려움이 있었습니다. 이를 극복하기 위해, 새로운 주의(attention)-없는 하이브리드 시공간 아키텍처로서 Hybrid Mamba-GCN(Pose Magic)을 제안합니다.

- **Technical Details**: Pose Magic은 Mamba와 Graph Convolutional Networks(GCN)을 결합한 하이브리드 아키텍처로, 전역적인 3D 구조를 학습하는 데 뛰어난 성능을 보입니다. GCN은 인접 관절 간의 관계를 포착하여 Mamba의 출력을 보완하는 새로운 표현을 생성합니다. 또한, 전이(receptive field)와 적응형 융합 기술을 통해 Mamba와 GCN의 표현을 통합하여 종합적인 인간 동작을 캡처합니다. 실시간 추론을 위해 완전히 인과적인(causal) 버전도 제공합니다.

- **Performance Highlights**: Pose Magic은 최신의 SOTA 결과를 달성하며 ($ightarrow$ 0.9 mm 줄어듦) FLOPs를 74.1% 절약합니다. 또한, 인간 동작의 자연스러운 일관성을 모델링하며, 보지 못한 시퀀스 길이에 대한 일반화 능력도 보여줍니다. 대규모 실험을 통해, Pose Magic이 최첨단 결과를 유지하면서도 더 적은 파라미터와 낮은 계산 복잡성으로 고효율을 입증하였습니다.



### Dual-View Pyramid Pooling in Deep Neural Networks for Improved Medical Image Classification and Confidence Calibration (https://arxiv.org/abs/2408.02906)
Comments:
          27

- **What's New**: 새로운 이중 체계 프레임워크(dual-view framework)를 제안합니다. 이는 공간 풀링(SP)과 교차 채널 풀링(CCP)의 상호 보완적인 역할을 체계적으로 분석하는 최초의 접근입니다. 이를 기반으로 새로운 풀링 방법인 듀얼 뷰 피라미드 풀링(DVPP)을 도입했습니다. DVPP는 다중 스케일 듀얼 뷰 피처들을 집계하여 의료 이미지 분류 및 신뢰도 보정 성능을 향상시키기 위해 SP와 CCP 연산자의 장점을 최대한 활용합니다.

- **Technical Details**: DVPP는 고수준 피처 맵(feature map)에서 다중 스케일 듀얼 뷰 피처를 집계하는 새로운 풀링 방법입니다. 우리는 파라미터가 없는(parameter-free) 방식으로 DVPP를 구현하는 다섯 가지 방법을 개발했습니다. DVPP는 여러 DNNs의 끝부분에 직접 통합될 수 있으며, 끝-끝(end-to-end) 방식으로 학습됩니다.

- **Performance Highlights**: 6개의 2D/3D 의료 이미지 분류 작업에서 여러 최첨단(state-of-the-art) 풀링 방법 및 보정 방법을 능가하는 성능을 보였습니다. DVPP는 의료 이미지 분류 결과 및 신뢰도 보정 측면에서 두드러진 향상을 이루었으며, 이러한 성능은 우리의 기대와 일치합니다.



### Enabling Intelligent Traffic Systems: A Deep Learning Method for Accurate Arabic License Plate Recognition (https://arxiv.org/abs/2408.02904)
- **What's New**: 이 논문은 정확한 이집트 차량 번호판 인식을 위한 새로운 2단계 프레임워크를 도입합니다(Egyptian Vehicle License Plate Recognition, EVLPR). 첫 번째 단계에서는 이미지 처리 기술을 사용하여 번호판을 신뢰성 있게 위치시키고, 두 번째 단계에서는 견고한 아랍 문자 인식을 위해 맞춤형 딥러닝 모델을 활용합니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 단계를 포함합니다. 첫 번째 단계는 이미지 처리 기술(image processing techniques)을 사용하여 번호판을 정확히 일정한 위치로 로컬라이징(localizing)하고, 두 번째 단계는 맞춤형 딥러닝(designed deep learning) 모델을 적용하여 견고한 아랍 문자 인식(character recognition)을 수행합니다. 이러한 접근 방식을 통해 다양한 데이터셋(daiverse dataset)에서 뛰어난 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 다양한 데이터셋에서 99.3%의 뛰어난 정확도를 달성하여 기존 방법을 능가합니다. 이 시스템은 트래픽 위반 탐지(traffic violation detection) 및 주차 최적화(parking optimization)와 같은 지능형 트래픽 관리(intelligent traffic management)에 잠재적 응용 가능성을 가지고 있습니다.

- **Future Directions**: 향후 연구에서는 시스템의 아키텍처 개선(architectural refinements), 데이터셋 확장(expanded datasets), 시스템 종속성(system dependencies) 문제를 해결하여 시스템의 기능을 향상시키는 데 중점을 둘 것입니다.



### Lighthouse: A User-Friendly Library for Reproducible Video Moment Retrieval and Highlight Detection (https://arxiv.org/abs/2408.02901)
Comments:
          6 pages; library tech report

- **What's New**: Lighthouse는 동영상 순간 검색 및 하이라이트 검출 (MR-HD: Moment Retrieval and Highlight Detection)을 위한 통합적이고 재현 가능한 코드베이스를 제공하는 사용자 친화적인 라이브러리입니다. 현재 연구 커뮤니티에서 다양한 방법, 데이터셋, 비디오-텍스트 기능(feature)을 통해 종합적이고 재현 가능한 실험을 수행하지 않는 두 가지 주요 문제가 있습니다. Lighthouse는 이러한 문제를 해결하기 위해 여섯 가지 모델, 세 가지 기능, 다섯 가지 데이터셋을 포함한 통합적이고 재현 가능한 코드베이스를 구현하였습니다. 또한 인퍼런스 API와 웹 데모를 제공하여 연구자와 개발자가 쉽게 접근할 수 있게 합니다.

- **Technical Details**: Lighthouse는 여섯 개의 최신 MR-HD 방법, 세 가지 비디오-텍스트 기능 및 다섯 가지 데이터셋을 지원하는 통합 코드베이스로, YAML 형식의 구성 파일을 통해 매개변수를 지정하여 단일 Python 명령어로 실험을 재현할 수 있습니다. 또한, 웹 데모와 인퍼런스 API 구현을 통해 사용자가 상세한 비디오-텍스트 처리를 이해하지 못해도 MR-HD를 쉽게 사용할 수 있도록 했습니다.

- **Performance Highlights**: Lighthouse는 참조 논문에서 보고된 점수를 대체로 재현하며, 기존의 여러 라이브러리와 달리 사용자 친화적인 디자인으로 사용자가 단일 환경에서 실험을 설정할 수 있도록 지원합니다. 설치 또한 간단하여 적은 노력으로 다양한 설정에서 실험을 수행할 수 있습니다.



### MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicin (https://arxiv.org/abs/2408.02900)
Comments:
          The project page is at this https URL

- **What's New**: 이번 논문에서는 의료 분야를 위한 대규모 멀티모달 데이터셋인 MedTrinity-25M을 소개합니다. 이 데이터셋은 10가지 형태의 의료 이미지를 포함하며, 65개 이상의 질병에 대한 멀티그레이너 해설(multigranular annotations)을 제공합니다. 이러한 해설은 전역 텍스트 정보(질병/병변 유형, 모달리티, 지역별 설명 및 지역 간 관계)와, 관심 영역(ROIs)에 대한 상세한 로컬 해설(바운딩 박스, 세그멘테이션 마스크 등)을 포함합니다.

- **Technical Details**: MedTrinity-25M 데이터셋은 자동화된 파이프라인을 사용하여 멀티모달 데이터를 확장하는 첫 번째 사례입니다. 90곳 이상의 소스에서 데이터를 수집하여 사전 처리하고, 도메인 전문가 모델을 통해 비정상 영역과 관련된 ROIs를 식별했습니다. 이후, 포괄적인 지식 베이스를 구축한 후, 이러한 ROIs를 가이드로 멀티모달 대형 언어 모델(LLMs)을 활성화하여 텍스트 설명을 생성합니다. 이 데이터를 통해 이미지-ROI-설명 트리플렛을 생성하여 멀티그레이너 해설을 제공합니다.

- **Performance Highlights**: MedTrinity-25M 데이터셋의 프리트레이닝을 통해, 우리 모델은 VQA-RAD와 PathVQA에서 최첨단 성능(state-of-the-art performance)을 달성하여 기존의 멀티모달 대형 언어 모델 및 다른 대표적인 최신 접근 방식을 능가했습니다. 이 데이터셋은 멀티모달 의료 AI 모델의 대규모 프리트레이닝을 지원하며, 미래의 기초 모델 개발에 기여할 수 있습니다.



### Diverse Generation while Maintaining Semantic Coordination: A Diffusion-Based Data Augmentation Method for Object Detection (https://arxiv.org/abs/2408.02891)
Comments:
          15 pages, 7 figures, ICPR2024

- **What's New**: 최근 연구들은 데이터 증강(data augmentation)이 객체 탐지 모델의 성능을 향상시키는 데 중요한 역할을 한다고 강조하고 있습니다. 그러나 기존 방법들은 데이터셋의 다양성과 의미론적 조정을 효과적으로 조화시키는 데 종종 어려움을 겪습니다. 이를 극복하기 위해, 우리는 사전 훈련된 조건부 확산 모델(pre-trained conditional diffusion models)을 활용하여 이 균형을 조정하는 혁신적인 증강 기법을 도입했습니다. 이 접근 방식은 데이터셋 다양성을 높이기 위한 카테고리 친화도 매트릭스(Category Affinity Matrix)와 증강된 이미지에서 의미론적 조정을 유지하기 위한 주변 영역 정렬(Surrounding Region Alignment) 전략을 포함합니다.

- **Technical Details**: 우리의 방법은 조건부 확산 모델을 활용하여 원본 이미지를 수정하며, 다양한 데이터셋을 생성합니다. 카테고리 친화도 매트릭스는 다른 카테고리 간의 시각적 및 의미론적 유사성을 정량화하여, 친화도를 공유하는 오브젝트로 원본 객체를 대체함으로써 데이터셋의 다양성을 적절히 풍부하게 만듭니다. 주변 영역 정렬 전략은 DDIM 역추적(ddim inversion) 및 새로운 확산 프로세스를 결합함으로써 원본 객체와 새로운 객체 간의 의미적 무결성을 유지합니다. 추가적으로, 인스턴스-레벨 필터(instance-level filter)는 저품질 이미지를 걸러내는 데 사용됩니다.

- **Performance Highlights**: 우리의 방법은 세 가지 서로 다른 객체 탐지 모델(Faster R-CNN, Mask R-CNN, YOLOX)에서 각각 +1.4AP, +0.9AP, +3.4AP의 평균 성능 향상을 보여 줍니다. 또한 카테고리 특정 및 세분화된 데이터셋에서도 +3.6AP 및 +4.4AP의 개선을 달성했습니다. 이러한 성과는 제안된 기술이 데이터셋 다양성과 의미론적 조정의 균형을 유지하는 데 있어 탁월한 성능을 발휘한다고 증명합니다.



### VizECGNet: Visual ECG Image Network for Cardiovascular Diseases Classification with Multi-Modal Training and Knowledge Distillation (https://arxiv.org/abs/2408.02888)
Comments:
          Accepted in International Conference on Image Processing (ICIP) 2024

- **What's New**: VizECGNet은 인쇄된 심전도(ECG) 이미지만을 사용해 다양한 심혈관 질환의 예후를 예측하는 새롭고 혁신적인 다중 모달 딥러닝 모델로서, 병원에서 디지털화된 신호 대신 이미지로 데이터를 저장하는 경향을 해결하고자 개발되었습니다.

- **Technical Details**: VizECGNet은 이미지와 신호 두 가지 모달리티의 정보를 통합하기 위해 크로스 모달 주의 모듈(Cross-Modal Attention Modules, CMAM)을 사용하고, 각 모달리티의 데이터를 처리하기 위해 자체 모달리티 주의 모듈(Self-Modality Attention Modules, SMAM)을 사용합니다. 또한, 지식 증류(Knowledge Distillation) 기법을 사용하여 각 모달리티 스트림의 예측 사이의 유사성을 높입니다. 이를 통해 inference 단계에서는 ECG 이미지만으로 높은 성능을 달성할 수 있습니다.

- **Performance Highlights**: VizECGNet은 기존의 신호 기반 ECG 분류 모델에 비해 정밀도(Precision) 3.50%, 재현율(Recall) 8.21%, F1-Score 7.38% 향상을 이루었습니다. 이 모델은 다중 모달 학습과 지식 증류 기법을 이용하여 대규모 12-lead ECG 데이터셋에서 우수한 성능을 보였습니다.



### Body of Her: A Preliminary Study on End-to-End Humanoid Agen (https://arxiv.org/abs/2408.02879)
Comments:
          Technical Report v1; Project Page: this https URL

- **What's New**: 이번 연구는 현실적인 상호작용형 가상 인간 에이전트를 모델링하는 실시간 양방향(end-to-end) 네트워크를 제안합니다. 이 시스템은 음성, 전신 움직임, 시선 접촉, 얼굴 표정, 입술 움직임, 제스처, 물건 다루기 등의 다양한 행동을 종합적으로 모델링할 수 있습니다. 특히, 대화 중 단절(interrupt)할 수 있는 능력을 포함하여, 실시간 쌍방향 소통이 가능한 특성을 갖추고 있습니다.

- **Technical Details**: 이 시스템은 사전 학습된 대형 언어 모델(Pre-trained Large Language Model, LLM)을 확장하여 오디오 및 비디오 입력을 통합한 멀티모달 모델(multi-modal model)입니다. 약 20만 시간의 오디오, 13만 시간의 비디오 데이터, 약 20,000개의 정렬 샘플을 수집하여 모델을 구축했습니다. 오디오 데이터는 Descript Audio Codec (DAC)을 사용하여 시간당 86Hz로 압축됐으며 비디오는 Querying Transformer 기법을 활용해 고품질을 유지하면서 압축되었습니다.

- **Performance Highlights**: 최종 모델은 일반화된 객체 조작과 같은 이전 시스템에서 달성하기 어려운 기능을 보여줍니다. 시스템이 실시간으로 인간의 입력을 받아 대응하고, 때로는 대화를 단절하는 능력까지 지니게 되었으며, 대규모 데이터로 확장하여 복잡한 하위 기능을 공동 최적화할 수 있는 가능성을 입증했습니다.



### GAReT: Cross-view Video Geolocalization with Adapters and Auto-Regressive Transformers (https://arxiv.org/abs/2408.02840)
Comments:
          Accepted at ECCV 2024

- **What's New**: 새로운 연구는 카메라와 주행 정보 없이 크로스 뷰 비디오 지리적 위치 지정(Cross-view video geo-localization; CVGL)을 수행할 수 있는 완전 트랜스포머 기반 방법인 GAReT을 제안합니다. GeoAdapter 모듈을 사용하여 이미지 레벨 표현을 비디오 입력으로 효율적으로 집계하며, TransRetriever를 통해 시간적으로 일관된 GPS 예측을 제공합니다.

- **Technical Details**: GAReT는 GeoAdapter와 TransRetriever라는 두 가지 주요 모듈로 구성됩니다. GeoAdapter는 트랜스포머 어댑터 모듈로, 비디오 입력에 맞게 이미지 레벨 표현을 집계하고 변환합니다. 트랜스포머 인코더는 비디오 프레임과 항공 이미지를 학습한 후 GeoAdapter 모듈을 최적화하여 비디오 레벨 표현을 얻습니다. TransRetriever는 인코더-디코더 트랜스포머 모델로, 각 프레임의 최상위 k개의 최근접 예측을 인코딩하고 이전 프레임의 예측을 기반으로 최적의 이웃을 자동 회귀 방식으로 디코딩합니다.

- **Performance Highlights**: 제안된 방법은 벤치마크 데이터셋에서 최첨단 성능을 보여줍니다. 기존 방법이 필요로 하는 카메라와 주행 정보를 사용하지 않으면서도 시간적 일관성을 유지하며 높은 정확도를 달성하였습니다.



### DaCapo: a modular deep learning framework for scalable 3D image segmentation (https://arxiv.org/abs/2408.02834)
- **What's New**: DaCapo는 대규모, 근등방성(near-isotropic) 이미지 데이터에서 기계 학습 접근 방식을 신속하게 훈련하고 적용할 수 있도록 맞춤화된 딥 러닝 라이브러리입니다. 이 소식에서는 DaCapo의 독특한 기능들, 모듈형 구조, 효율적인 실험 관리 도구, 그리고 확장 가능한 배포 능력에 대해 소개합니다.

- **Technical Details**: DaCapo는 대규모 근등방성 이미지 세분화(image segmentation)에 대한 접근성을 향상시키기 위해 설계되었습니다. 모듈형 구조로 구성되어 있어 사용자들이 다양한 실험을 효율적으로 관리할 수 있도록 돕고, 확장 가능한 배포 능력을 갖추고 있어 다양한 환경에서 쉽게 적용 가능합니다. 또한, 이 오픈 소스 프로젝트에 대해 커뮤니티의 탐색과 기여를 권장하고 있습니다.

- **Performance Highlights**: DaCapo는 기존의 머신 러닝 접근 방식을 대규모 이미지 데이터에 신속하게 적용하고 훈련할 수 있도록 최적화되어 있습니다. 특히, 근등방성 이미지 세분화 작업에서 뛰어난 성능을 발휘할 것으로 기대됩니다.



### Gaussian Mixture based Evidential Learning for Stereo Matching (https://arxiv.org/abs/2408.02796)
- **What's New**: 본 논문에서는 견고한 스테레오 매칭(stereo matching)을 위해 새로운 가우시안 혼합 기반의 증거 학습 솔루션을 소개합니다. 기존의 단일 가우시안 분포에 의존하는 증거 딥러닝 접근법과 달리, 본 프레임워크는 스테레오 매칭에서 개별 이미지 데이터가 가우시안 혼합(Gaussian mixture) 분포를 따른다고 가정합니다. 이를 통해 픽셀 단위의 예측 정밀도를 높이고, 실제 이미지 분포를 더 정확하게 반영할 수 있습니다.

- **Technical Details**: 본 연구는 역감마(inverse-Gamma) 분포를 각 혼합 구성 요소의 중간 사전으로 사용하여, 단일 가우시안 분포와 비교하여 개선된 깊이 추정을 달성합니다. 이는 모델 불확실성을 효과적으로 포착하여 강력한 도메인 간 생성 능력을 가능하게 합니다. Scene Flow 데이터셋을 사용하여 모델을 학습하고, KITTI 2015 및 Middlebury 2014에서 테스트를 수행하였습니다. 본 방법은 신뢰성 있는 방식으로 베이스라인(baseline) 방법들보다 향상된 성능을 보여주었습니다.

- **Performance Highlights**: 우리의 접근 방식은 인-도메인(in-domain) 검증 데이터와 도메인 간(cross-domain) 데이터셋 모두에서 새로운 최첨단(state-of-the-art) 결과를 달성하며, 스테레오 매칭 작업에서의 효과성과 견고성을 입증했습니다.



### Lesion Elevation Prediction from Skin Images Improves Diagnosis (https://arxiv.org/abs/2408.02792)
Comments:
          Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (MICCAI ISIC) 2024; 12 pages, 2 tables, 4 figures

- **What's New**: 이 논문은 피부 병변 분석에 효과적인 또 다른 임상적 특징인 병변의 상승(elevation) 예측을 심층 학습 모델로 구현하고 이를 통해 진단 성능을 향상시키는 가능성을 다루고 있습니다. 주로 2D 피부 병변 이미지를 이용하여 병변 상승 레이블을 예측하고 그 예측 결과를 다양한 데이터셋에 적용하여 성능을 평가하였습니다.

- **Technical Details**: 이 연구에서는 심층 학습 모델을 사용하여 병변의 상승 레이블을 예측하고, 이를 진단 모델의 부가 입력(auxiliary input)으로 활용합니다. 예측 모델의 성능은 derm7pt 데이터셋에서 테스트 되었으며, 이후 ISIC 2016, 2017, 2018 챌린지 데이터셋, MSK, DermoFit와 같은 다른 데이터셋에서도 평가되었습니다. 또, 예측된 상승 레이블을 활용한 교차 도메인 일반화(cross-domain generalization) 성능도 실험되었습니다.

- **Performance Highlights**: 이 논문에서는 진단 모델에 병변 상승 정보를 추가함으로써 그 성능이 현저히 향상된다는 것을 보여줍니다. 특히, 진단 성능이 dermoscopic 이미지에서는 AUROC이 최대 6.29%, clinical 이미지에서는 2.69%까지 개선되었습니다.



### GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths (https://arxiv.org/abs/2408.02788)
Comments:
          To appear in ECCV2024

- **What's New**: GazeXplain을 소개합니다. GazeXplain은 시각적 시선 경로 예측(scanpath prediction)과 설명(explanation)을 결합한 새로운 연구 접근법입니다. 사람들의 주의 과정에 따라 시각 장면을 탐색하는 과정을 설명하고, 기존 모델들이 제공하지 않는 시선 고정(fixation)에 대한 자연어 설명을 제공합니다.

- **Technical Details**: GazeXplain은 다음과 같은 기술적 혁신을 포함합니다: (1) 다양한 시선 추적 데이터셋에서 시선 고정에 대한 자연어 설명을 주석(annotation)으로 추가, (2) 스캔패스와 자연어 설명을 공동으로 예측하는 attention-language decoder를 사용한 일반적 모델 아키텍처 제안, (3) 시선 고정과 설명의 일관성을 향상시키기 위한 독특한 semantic alignment 메커니즘 통합, (4) 데이터셋 간 공동 훈련(cross-dataset co-training) 접근법을 사용하여 모델의 일반화 성능 향상.

- **Performance Highlights**: 다양한 시선 추적 데이터셋에서 GazeXplain의 광범위한 실험 결과, 이 모델은 스캔패스 예측과 설명 생성 모두에서 높은 성능을 보였습니다. 이는 사람들의 시각적 주의(attention) 과정과 인지(cognitive) 과정을 이해하는 데 중요한 통찰을 제공합니다.



### Segmentation Style Discovery: Application to Skin Lesion Images (https://arxiv.org/abs/2408.02787)
Comments:
          Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (MICCAI ISIC) 2024; 13 pages, 2 tables, 3 figures

- **What's New**: 이번 연구에서는 주석자(correspondence)와 상관 없이 이미지-마스크 쌍에서 다수의, 다양한, 의미론적으로 일관된 세분화 스타일을 학습하는 StyleSeg라는 새로운 세분화 방법을 소개합니다. 이 방법은 네 가지 공개된 피부 병변 세분화(Skin Lesion Segmentation, SLS) 데이터셋에서 기존 방법보다 일관되게 우수한 성능을 보였으며, 주석자 대응 정보를 포함한 ISIC-MultiAnnot 데이터셋을 새롭게 구축하였습니다.

- **Technical Details**: StyleSeg은 두 가지 딥러닝 모델로 구성됩니다: 이미지를 세분화하는 세분화 모델(f_s)과 스타일을 분류하는 스타일 분류기 모델(f_c). 세분화 모델은 다수의 세분화 마스크를 예측하며, 스타일 분류기 모델은 주석자의 선호도를 반영합니다. 이번에 제안된 새로운 척도 AS2는 예측된 스타일과 주석자의 선호도 간의 일치도를 측정합니다.

- **Performance Highlights**: StyleSeg는 네 가지 공개된 SLS 데이터셋에서 경쟁 방법을 능가하였으며, 새로운 AS2 척도를 통해 주석자 선호도와의 강력한 일치를 보였습니다.



### LR-Net: A Lightweight and Robust Network for Infrared Small Target Detection (https://arxiv.org/abs/2408.02780)
- **What's New**: 새로운 경량화 및 견고한 네트워크(LR-Net)를 제안했습니다. 이 네트워크는 복잡한 구조를 버리고, 탐지 정확도와 자원 소비 간의 균형을 효과적으로 맞추었습니다. 특히, ICPR 2024 Resource-Limited Infrared Small Target Detection Challenge Track 2에서 3위를 차지했습니다.

- **Technical Details**: LR-Net은 경량 특성 추출 주의 모듈(LFEA)을 사용하여 타겟 특성을 충분히 추출하고 채널 간 정보 상호작용을 강화합니다. 간단한 정제된 특성 전이 모듈(RFT)은 크로스 레이어 연결 방식 대신 사용되어 네트워크의 특성 정제 추출 능력을 개선합니다. 또한, 소형 타겟 손실 문제를 해결하기 위해 저레벨 특성 분포(LFD) 전략과 간소화된 이중 선형 보간 주의 모듈(SBAM)을 도입하여 고레벨 특성에 대한 저레벨 특성의 정보를 보충하고 두 특성의 융합을 촉진합니다.

- **Performance Highlights**: LR-Net은 매우 높은 성능을 입증했으며, 다양한 실험 결과를 통해 최첨단(State-of-the-art, SOTA) 성능을 달성했습니다. 특히, 다중 스케일 샘플이 포함된 데이터셋에서 더 견고한 새 훈련 및 추론 크로핑 전략을 채택했습니다.



### Refined Infrared Small Target Detection Scheme with Single-Point Supervision (https://arxiv.org/abs/2408.02773)
- **What's New**: 이번에 제안된 연구는 단일 지점 지도(supervision) 방식 기반의 혁신적인 적외선 소형 타겟 감지 기법입니다. 기존 방법들이 실제 요구사항을 충족하지 못하는 문제를 해결하기 위해, 우수한 세그멘테이션(분할) 정확도와 탐지율을 제공하는 새로운 감지 기법을 제안하였습니다. 또한 이 프레임워크 기반에서 다양한 적외선 소형 타겟 감지 네트워크의 성능을 탐구하였습니다.

- **Technical Details**: 본 연구에서는 단일 지점 지도 기반의 라벨 진화(label evolution) 프레임워크인 LESPS를 도입하였으며, 이를 강화하기 위해 종합적인 후처리(post-processing) 전략을 개발하였습니다. 우선, 세그멘테이션 정확도를 위해 테스트 시간 증강(test-time augmentation, TTA)과 조건부 랜덤 필드(conditional random field, CRF)를 결합한 방식을 사용합니다. 탐지율을 향상시키기 위해 조정 가능한 민감도(Adjustable Sensitivity, AS) 전략을 도입하여, 다중 탐지 결과의 장점을 살리고 신뢰도가 낮은 영역을 중심점 형태로 세그멘테이션 이미지에 추가합니다. 또한, 다단계 손실(multi-stage loss)을 통해 세밀한 감지를 구현하고, 테스트 샘플에 대해 합리적인 슬라이딩 윈도우(sliding window) 절단 전략을 사용하는 것이 실제 다중 크기 샘플에 대해 더 좋은 성능을 발휘하는 것을 확인했습니다.

- **Performance Highlights**: 제안된 기법은 다양한 실험 결과에서 최신 성능(SOTA)을 달성하였습니다. 특히, ICPR 2024 리소스 제한 적외선 소형 타겟 감지 챌린지 트랙 1에서 제안된 기법이 3위를 차지하였습니다.



### From Recognition to Prediction: Leveraging Sequence Reasoning for Action Anticipation (https://arxiv.org/abs/2408.02769)
Comments:
          Accepted by ACM TOMM

- **What's New**: 새로운 논문에서는 비디오 기반으로 미래 행동을 예측하는 '행동 예측' 문제를 다루기 위해 Anticipation via Recognition and Reasoning (ARR)이라는 혁신적인 엔드투엔드 비디오 모델링 아키텍처를 제안했습니다. ARR은 행동 인식과 순서 추론 과제로 행동 예측 작업을 분해하고, Attention Mechanism(주의 메커니즘)을 활용하여 다음 행동 예측(Next Action Prediction; NAP)을 효과적으로 학습합니다.

- **Technical Details**: ARR은 Vision Transformer와 Causal Decoder(인과 디코더)를 사용하여 주어진 비디오를 시퀀스 입력으로 모델링합니다. 이것은 입력 클립들로부터 더 효과적인 특징을 추출하고 각각의 예측된 행동이 앞선 정보만을 사용할 수 있게 하여 정보 유출을 방지합니다. 또한, NAP를 사용해 행동 간의 통계적 관계를 학습하고 순서 추론을 강화합니다. NAP에 필요한 대규모 학습 데이터를 해결하기 위해 비디오의 고유 시간 동태를 활용한 자율학습 사전 학습 방법을 제안했습니다.

- **Performance Highlights**: Epic-kitchen-100, EGTEA Gaze+, 그리고 50salads 데이터셋에서 수행한 광범위한 실험을 통해 ARR의 효능을 실증했으며, 행동 간의 강한 통계적 상관관계를 입증했습니다.



### ConDL: Detector-Free Dense Image Matching (https://arxiv.org/abs/2408.02766)
- **What's New**: 본 연구에서는 밀집 이미지 대응(estimating dense image correspondences)을 위한 새로운 딥러닝 프레임워크를 소개합니다. 우리의 완전 컨볼루션 모델은 각 픽셀이 매칭 가능한 디스크립터(descriptor)를 가지는 밀집 피처 맵(feature maps)을 생성합니다. 기존 방법과 달리, 우리의 모델은 원근 변형 및 조명 변화와 같은 왜곡이 포함된 합성 데이터를 사용합니다. 대조 학습(contrastive learning)을 통해 이러한 왜곡에 대한 불변성을 높여 안정적인 매칭을 가능하게 합니다. 특히, 키포인트 검출기(keypoint detector)가 필요 없다는 점에서 기존 이미지 매칭 기술과 차별화됩니다.

- **Technical Details**: 우리는 ConDL(Contrastive Descriptor Learning)이라는 프레임워크를 제안합니다. 이 프레임워크는 SIDAR를 사용하여 다양한 이미지 왜곡(perspective distortion, illumination changes, shadows, occlusions)을 포함한 합성 데이터를 학습에 활용합니다. CNN 기반 ResNet을 통해 밀집 이미지 피처를 추출하며, 대조 학습을 통해 이미지 피처의 유사성을 견고하게 학습합니다. ConDL은 키포인트 검출기가 필요 없이, 모든 대응 관계를 한 번에 최적화하는 교차 엔트로피 손실 함수를 활용합니다.

- **Performance Highlights**: ConDL는 다양한 왜곡 환경에서도 강력한 이미지 매칭 성능을 보여줍니다. 50,000개의 이미지 쌍을 학습 데이터로, 4,000개의 이미지 쌍을 테스트 데이터로 사용하여 트레이닝되었습니다. 학습 데이터는 강한 합성 왜곡을 포함하여 기존 이미지 매칭 방법과의 차별성을 확인할 수 있었습니다. 모델은 픽셀 단위의 유사도 매트릭스(similarity matrix)를 통해 매칭 성능을 극대화합니다.



### Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation (https://arxiv.org/abs/2408.02761)
Comments:
          Expansion of "Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation" arXiv:2308.03723 . Submitted to the Journal for Machine Learning in Biomedical Imaging. Code available at this https URL

- **What's New**: 이번 연구에서는 Mahalanobis 거리(MD)와 k-th nearest neighbors distance(KNN)를 사용하여 훈련 데이터 분포 밖의(out-of-distribution, OOD) 이미지를 식별하는 방법을 제안합니다. MD는 기존의 Gaussian 분포 가정을 기반으로 라벨된 훈련 이미지를 기준으로 테스트 이미지를 비교합니다. KNN은 이 Gaussian 가정 없이 분포에 대한 의존성을 줄이고, 그로 인한 성능 향상을 보여줍니다.

- **Technical Details**: 이 연구에서는 네 개의 Swin UNETR 및 nnU-net 모델을 사용하여 간을 T1-강조 MRI와 CT 이미지에서 분할(segmentation)했습니다. bottleneck features의 차원을 주성분 분석(principal component analysis, PCA) 또는 균일 매니폴드 근사 및 투사 방법(uniform manifold approximation and projection, UMAP)을 통해 줄인 후, MD와 KNN을 적용합니다. 이를 통해 모델이 실패한 이미지를 높은 성능으로 검출할 수 있었으며, 계산 부하도 최소화했습니다.

- **Performance Highlights**: 이 연구는 MD와 KNN을 비교했을 때, KNN이 MD에 비해 높은 성능 및 확장성을 갖는 것을 보여줍니다. 특히, KNN은 raw 및 average-pooled bottleneck features에 대해 MD보다 뛰어난 성능을 나타냈습니다. 이를 통해 Gaussian 분포 가정이 의료영상 분할 모델 특징에 유효하지 않을 가능성을 제기합니다.



### Diffusion Models as Data Mining Tools (https://arxiv.org/abs/2408.02752)
Comments:
          Project Page: this https URL Accepted in ECCV 2024

- **What's New**: 이 논문은 이미지 생성용 학습된 생성 모델(generative model)을 시각적 데이터 마이닝 도구로 활용할 수 있는 방법을 논의합니다. 생성 모델이 훈련 데이터의 정확한 표현을 학습한다는 점을 이용하여 데이터 요약 및 시각적 패턴을 마이닝할 수 있는 접근법을 제시합니다. 특히, 조건부 diffusion 모델을 특정 데이터셋에서 이미지를 생성하도록 세부 조정한 후, 이 모델을 사용하여 데이터셋 내에서 시각적 요소의 typicality(전형성)을 측정할 수 있다는 점을 강조합니다.

- **Technical Details**: 논문에서 제안하는 방법은 주로 다음과 같은 단계로 구성됩니다: 1) 조건부 diffusion 모델을 대상 데이터셋에 맞게 세부 조정합니다. 2) 조정된 모델을 사용하여 픽셀 단위로 특정 라벨(위치 정보, 시간 정보, 의미적 라벨 등)이 이미지 재구성에 미치는 영향을 평가합니다. 3) 전형성을 기반으로 비정형 시각 요소를 뽑아내고 이를 클러스터링하여 대표 패턴을 요약합니다. Diffusion 모델은 무작위 노이즈를 목표 분포로 변환하는 과정에서 반복적인 디노이징을 수행합니다.

- **Performance Highlights**: 제안된 방법론은 다양한 콘텐츠와 규모의 데이터셋에서 뛰어난 성능을 보이며, 기존의 방법론보다 효율적으로 규모를 확장할 수 있음을 보여줍니다. 예를 들어, 20세대의 아비에이터 안경 및 40세대의 군모와 같은 시대적 요소의 시각적 대표성을 강조할 수 있습니다. 또한 대규모 거리뷰 데이터에서 전봇대, 볼라드 등 지리적 특징을 요약하는 데에도 성공적이었습니다. 이는 GeoGuessr 게임(인기있는 지리 퀴즈 게임)에서도 볼 수 있는 주요 지리적 요소를 통해 입증되었습니다.



### Privacy-Safe Iris Presentation Attack Detection (https://arxiv.org/abs/2408.02750)
- **What's New**: 이 논문에서는 신원 누설이 없는 합성 아이리스 이미지를 사용하여 프라이버시 안전한 아이리스 위조 공격 탐지(PAD) 방법론을 제안하였습니다. 최신 아이리스 PAD 벤치마크를 통해 방법론을 평가하고, 합성된 ISO/IEC 19794-6 규격의 아이리스 이미지를 생성하는 두 가지 모델을 설계하였습니다. 첫 번째 모델은 신뢰할 수 있는 샘플을 합성하고, 두 번째 모델은 텍스쳐드 콘택트 렌즈(회사의 브랜드에 따라 조건부) 이미지를 생성합니다.

- **Technical Details**: 제안된 방법론은 합성된 데이터만으로 학습된 최초의 아이리스 PAD 솔루션을 도입했습니다. 본 연구는 StyleGAN2-ADA 모델을 활용하여 ISO/IEC 19794-6 규격의 아이리스 이미지를 합성하였습니다. 또한, 신원 누설을 막기 위해 생성 모델의 학습에 사용된 샘플과 '생체 인식 일치' 측면에서 '너무 가까운' 샘플은 제외시켰습니다. 이 프라이버시 안전한 PAD 모델은 기존의 아이리스 PAD 벤치마크와 함께 테스트되었습니다.

- **Performance Highlights**: 합성 데이터만으로 학습된 모델은 인간 샘플로 학습된 모델과 비교했을 때 약간 낮은 성능을 보였지만 여전히 합리적인 성능을 나타냈습니다. 실험 결과, 합성 데이터로만 학습된 모델의 평균 ROC 곡선 하단 면적(AUC)은 0.90에서 0.93 사이였으며, 인간 샘플로 학습된 모델의 AUC는 0.97로 나타났습니다. 이는 합성 데이터만으로도 아이리스 PAD 방법을 학습할 수 있다는 가능성을 시사합니다.



### MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models (https://arxiv.org/abs/2408.02718)
Comments:
          Project Page: this https URL

- **What's New**: 다중 이미지 처리는 대형 비전-언어 모델(LVLMs: Large Vision-Language Models)이 장면에 대한 더 철저하고 정교한 이해를 발전시키는 데 중요합니다. 최근의 다중 이미지 LVLMs가 이 필요성을 해결하기 시작했지만, 그 평가가 발전 속도를 따라가지 못하고 있습니다. 이에 따라 우리는 LVLMs을 다양한 다중 이미지 작업에서 평가하기 위해 '멀티모달 다중 이미지 이해 벤치마크(MMIU: Multimodal Multi-image Understanding Benchmark)'를 도입합니다.

- **Technical Details**: MMIU는 7가지 유형의 다중 이미지 관계, 52개의 작업, 77K 이미지 및 11K 개의 정성스럽게 큐레이팅된 다중 선택 질문으로 구성된 포괄적인 평가 모음입니다. 이는 지금까지의 벤치마크 중 가장 광범위한 것입니다. 우리는 오픈 소스 및 독점 모델을 포함한 24개의 인기 있는 LVLMs을 평가하여, 특히 공간 이해와 관련된 작업에서 다중 이미지 이해에 상당한 어려움이 있음을 발견했습니다.

- **Performance Highlights**: 가장 발전된 모델인 GPT-4o조차도 MMIU에서 55.7%의 정확도만을 달성했습니다. 다양한 분석 실험을 통해 주요 성능 격차 및 제한점을 식별하고, 향후 모델 및 데이터 개선을 위한 귀중한 통찰력을 제공합니다. 우리는 MMIU가 LVLM 연구 및 개발의 최전선을 발전시키고, 정교한 멀티모달 다중 이미지 사용자 상호작용을 달성하는 방향으로 나아가기를 기대합니다.



### Compositional Physical Reasoning of Objects and Events from Videos (https://arxiv.org/abs/2408.02687)
Comments:
          arXiv admin note: text overlap with arXiv:2205.01089

- **What's New**: 이번 논문에서는 객체의 물리적 속성을 추론하고 예측하는 새로운 접근 방식을 제시합니다. 특히, 시각적으로 확인할 수 없는 질량이나 전하 같은 숨겨진 물리적 속성에 초점을 맞추고 있습니다. 이 문제를 해결하기 위해 합성 비디오와 실제 데이터를 포함하는 Compositional Physical Reasoning (ComPhy) 데이터셋을 도입했습니다.

- **Technical Details**: 핵심 기술로 신경-상징적 프레임워크인 Physical Concept Reasoner (PCR)를 제안합니다. 이 모델은 동영상에서 객체를 감지하고 프레임 간에 연관시키며, 숨겨진 물리적 속성들을 학습 및 추론합니다. 또한 미래 예측과 반사실적 (counterfactual) 예측을 수행하고, 이러한 정보를 활용해 복잡한 질문에 답변할 수 있습니다.

- **Performance Highlights**: 최신 비디오 추론 모델들을 ComPhy 데이터셋에서 평가한 결과, 이 모델들이 숨겨진 물리적 속성을 제대로 캡처하지 못해 성능이 떨어지는 것을 확인했습니다. 그러나 PCR은 훈련 후 놀라운 성능을 보여주었으며, 객체 감지 및 물리적 속성 추론에서 뛰어난 능력을 발휘했습니다.



### Segment Anything in Medical Images and Videos: Benchmark and Deploymen (https://arxiv.org/abs/2408.03322)
- **What's New**: 이번 연구에서는 Segment Anything Model 2 (SAM2)의 의료 이미지 및 비디오에서의 성능을 평가한 내용을 다룹니다. SAM2는 다양한 영상 자료에서 뛰어난 분할 성능을 보여주었지만, 의료 데이터를 대상으로 하는 효용성은 아직 불확실합니다. 이를 해결하기 위해 SAM2를 11개의 의료 영상 모달리티에서 벤치마크하고, SAM1 및 MedSAM과 비교하여 장점 및 단점을 분석합니다. 또한, 전이 학습(transfer learning)을 통해 SAM2가 의료 도메인에 신속히 적응할 수 있음을 입증하고, 3D Slicer 플러그인과 Gradio API로 구현하여 효율적인 3D 이미지 및 비디오 분할을 지원합니다.

- **Technical Details**: SAM2는 자연 이미지 및 비디오에서의 분할 성능을 기반으로 한 모델로, 50.9K 비디오 데이터를 통해 훈련되었습니다. 이 모델은 제로샷(Zero-shot) 성능이 뛰어나 다양한 비디오 및 이미지 분할 벤치마크에서 우수한 성적을 보였습니다. 본 연구에서는 SAM2를 CT, MRI, PET, 초음파(ultrasound), 내시경, 안저(fundus), 진균경검(dermoscopy), 유방조영술(Mammography), 광학 단층촬영(OCT) 등 11개 의료 이미지 모달리티에서 평가하고, 필요시 전이 학습을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: 연구 결과, SAM2는 MR, 진균경검, 광학 현미경 이미지에서 SAM1보다 높은 DSC 점수를 기록했으나, PET와 OCT 이미지에서는 낮은 점수를 보였습니다. 전반적으로, MedSAM이 11개 모달리티 중 9개에서 SAM1 및 SAM2보다 우수한 성능을 나타냈으며, 이는 MedSAM이 더 많은 의료 데이터를 통해 훈련되었기 때문입니다. 특히, CT, MR, PET 이미지에서는 중간 슬라이스에서 시작하여 시퀀스 방식으로 나머지 슬라이스를 분석하는 방법을 사용했습니다. SAM2는 비디오 및 3D 의료 이미지 분할을 지원하여 첫 슬라이스에서 시작하여 비디오 분할 기능을 통해 다른 이미지로 전파됩니다.



### LAC-Net: Linear-Fusion Attention-Guided Convolutional Network for Accurate Robotic Grasping Under the Occlusion (https://arxiv.org/abs/2408.03238)
Comments:
          accepted by IROS2024

- **What's New**: 이 논문은 로봇이 복잡한 장면에서 물체의 가려진 부분을 추론할 수 있도록 하는 'amodal segmentation'을 활용한 새로운 프레임워크를 소개합니다. 기존의 물체 분할 알고리즘이 가시 영역의 세그먼트에 집중한 반면, amodal segmentation은 로봇이 물체의 전체 형태를 인식하고 보다 정확하게 집기(grasping) 능력을 개선할 수 있게 합니다.

- **Technical Details**: 논문에서는 기존의 분할 알고리즘을 활용해 목표 물체의 가시 세그먼트를 감지한 후, 이러한 형태 정보(prior)를 이용해 전체 물체 마스크를 완성합니다. 특히, RGB 이미지에서의 Semantic features와 깊이 이미지에서의 Geometric information을 효율적으로 융합하기 위해 Linear-fusion Attention-guided Convolutional Network (LAC-Net)를 제안합니다. LAC-Net은 선형 융합 전략을 사용해 교차 모달 데이터를 효율적으로 융합하고, 알고리즘이 가시 마스크를 주의 맵(attention map)으로서 활용해 목표 특징 위치에 집중하여 전체 마스크를 복원합니다.

- **Performance Highlights**: 다양한 데이터셋에서의 실험 결과, 제안된 방법이 최첨단 성능(state-of-the-art performance)을 달성했습니다. 또한 로봇 실험을 통해 실제 환경에서 방법의 실현 가능성과 높은 튼튼함을 검증했습니다. 코드와 데모는 프로젝트 페이지에서 확인할 수 있습니다.



### Learning to Learn without Forgetting using Attention (https://arxiv.org/abs/2408.03219)
Comments:
          Published at 3rd Conference on Lifelong Learning Agents (CoLLAs), 2024

- **What's New**: 이 논문은 Continual Learning(CL)을 개선하기 위해 Transformer 기반의 meta-learning 최적화 방법을 제안합니다. meta-optimizer는 모델 파라미터 간의 복잡한 상관관계를 학습하며, 새로운 작업에 대한 효과적인 가중치 업데이트를 생성하는 동시에 이전 작업에 대한 망각(catastrophic forgetting)을 방지합니다. 이 접근법은 Meta-Learned Optimizer를 활용하여 CL 프레임워크 내에서 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 Transformer 기반의 meta-optimizer 네트워크를 제안합니다. 이 네트워크는 별도의 classifier 네트워크의 각 파라미터를 학습하면서 작업 간 파라미터 간의 복잡한 관계를 주의(attention)를 통해 학습합니다. task-specific importance score를 계산하고, 중요도가 낮은 가중치를 무시함으로써 새로운 작업에 대한 최적의 가중치 업데이트를 제공합니다. 이를 통해 이전 작업의 지식을 잃지 않으면서 새로운 작업을 빠르게 학습할 수 있습니다.

- **Performance Highlights**: SplitMNIST, RotatedMNIST, SplitCIFAR-100과 같은 벤치마크 데이터셋에서 실험한 결과, 제안된 방법이 forward와 backward transfer 모두에서 뛰어난 성능을 보였습니다. 특히, 소량의 레이블 데이터만 사용할 때도 강력한 일반화 성능을 보이며 효과적인 학습 성과를 냈습니다.



### SGSR: Structure-Guided Multi-Contrast MRI Super-Resolution via Spatio-Frequency Co-Query Attention (https://arxiv.org/abs/2408.03194)
Comments:
          The 15th International Workshop on Machine Learning in Medical Imaging (MLMI 2024)

- **What's New**: 이번 연구에서는 다중-대조 (multi-contrast) 이미지를 활용한 MRI 초해상도(super-resolution, SR) 구조를 제안합니다. 특히, 구조-유도된 (structure-guided) MCSR 프레임워크인 SGSR을 소개하며, 새로운 공간-주파수 공동 쿼리 주의 메커니즘(spatio-frequency co-query attention, CQA)을 사용합니다. 이 메커니즘은 여러 대조 이미지를 공동 구조 쿼리를 통해 주의를 수행하며, 다른 대조에서 공통 구조를 추출하고 융합하며 정제합니다.

- **Technical Details**: SGSR은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 공간 공동 쿼리 주의(spatial co-query attention, SCQA) 모듈은 여러 대조에서 공통 구조를 추출하고 정제합니다. 둘째, 주파수 공동 쿼리 주의(frequency co-query attention, FCQA) 모듈은 더 세밀한 구조-외형 상호작용을 위해 주파수 영역에서 CQA를 수행합니다. 마지막으로, SCQA와 FCQA에서 추출된 특징을 학습하는 인코더-디코더 네트워크가 뒷받침됩니다. SGSR의 혁신적인 점은 다중 대조 이미지를 통해 공통적인 구조 정보를 명시적으로 활용하는 최초의 모델이라는 것입니다.

- **Performance Highlights**: SGSR은 fastMRI 무릎 데이터와 저자장(低磁場) 뇌 MRI 데이터에 대한 실험에서, 수치적으로나 시각적으로 모두 최첨단 MCSR 방법보다 우수한 성능을 보였습니다. SGSR은 특히, 구조 정보를 보다 효과적으로 정제하여 양질의 이미지를 복원하는 데 있어서 기존 모델들보다 강력한 성능을 입증했습니다.



### Training-Free Condition Video Diffusion Models for single frame Spatial-Semantic Echocardiogram Synthesis (https://arxiv.org/abs/2408.03035)
Comments:
          Accepted to MICCAI 2024

- **What's New**: Free-Echo라는 새로운 방법을 소개합니다. 이 방법은 단일 말기 이완기(end-diastolic) 분할 맵(segmentation map)에서 현실적인 심장 초음파(echocardiograms)를 생성할 수 있습니다. 추가적인 학습 데이터 없이도 실현됩니다.

- **Technical Details**: Free-Echo는 3D-Unet 모델과 Temporal Attention Layers 모델에 기반한 방법입니다. SDEdit라는 학습이 필요 없는 조건부 기법(training-free conditioning method)을 사용하여 분할 맵을 기준으로 심장 초음파를 생성합니다. 모델 구조는 3개의 다운샘플(downsample) 블록, 2개의 중간(middle) 블록, 3개의 업샘플(upsample) 블록으로 구성돼 있으며, 주요 요소는 컨볼루션 레이어, 공간 선형 주의 레이어(spatial linear attention layer), 동일 위치의 주의 레이어(attention layer for same pixel position)입니다.

- **Performance Highlights**: 저희 모델을 CAMUS와 EchoNet-Dynamic라는 두 개의 공공 심장 초음파 데이터 세트에서 평가했습니다. 결과는 입력된 분할 맵과 공간적으로 일치하는 허구적인 심장 초음파를 생성했고, 학습 기반 CDMs와 비슷한 성능을 보였습니다. 이 방법은 데이터 증대(data augmentation)와 도메인 적응(domain adaptation) 같은 다양한 의료 영상 응용 분야에서 활용될 수 있습니다.



### ASR-enhanced Multimodal Representation Learning for Cross-Domain Product Retrieva (https://arxiv.org/abs/2408.02978)
Comments:
          10 pages, 5 figures

- **What's New**: E-commerce 환경이 점점 더 멀티미디어적으로 풍부해지면서, 제품들은 이미지, 짧은 비디오, 또는 라이브 스트림과 같은 다양한 형식으로 전시되고 있습니다. 이러한 다양한 형식을 효과적으로 통합하기 위한 통일된 벡터화된 크로스 도메인 제품 표현이 필요합니다. 본 논문은 ASR 텍스트를 활용하여 멀티모달 제품 표현 학습 (AMPere)을 제안하며, 기존의 시각적 표현만으로는 부족했던 부분을 보완합니다.

- **Technical Details**: AMPere는 LLM 기반의 ASR 텍스트 요약기를 사용하여 원시 ASR 텍스트에서 제품 특정 정보를 추출합니다. 요약된 ASR 텍스트와 시각적 데이터를 결합하여 멀티모달 임베딩(embeddings)을 생성할 수 있도록 Multi-branch Network를 활용합니다. 특히 ROPE와 같은 대규모 트라이 도메인 데이터셋에서 실험을 통해 제안된 방법의 유효성을 입증했습니다.

- **Performance Highlights**: 광범위한 실험 결과는 AMPere가 크로스 도메인 제품 검색(CdPR)에서 통일된 멀티모달 제품 표현을 획득하는 데 있어 매우 효과적임을 보여줍니다. LLM 기반 텍스트 요약기를 사용하여 ASR 텍스트의 노이즈를 효과적으로 제거함으로써, 멀티모달 표현 학습에 있어서 큰 성과를 얻었습니다.



### VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledg (https://arxiv.org/abs/2408.02865)
- **What's New**: VisionUnite는 안과 분야를 위한 새로운 비전-언어 융합 대형 모델입니다. 이 모델은 1.24백만 개 이미지-텍스트 쌍의 대규모 데이터셋으로 사전 훈련되고, 추가로 MMFundus 데이터셋(296,379개의 고품질 눈저부 이미지-텍스트 쌍 및 889,137개의 모의 의사-환자 대화 인스턴스)으로 세부 조정되었습니다.

- **Technical Details**: VisionUnite는 Transformer 기반의 비전 인코더(vision encoder), 비전 어댑터(vision adapter) 및 LLaMA 모델로 고유의 목적을 달성하도록 세 가지 훈련 목표(visual text contrastive learning loss(CLIP Loss), visual classification loss(CLS Loss) 및 text generation loss(LLM Loss))로 학습됩니다. VisionUnite는 자연 이미지-텍스트 쌍과 생의학 이미지-텍스트 쌍을 포함한 1.19백만 개의 이미지-텍스트 쌍의 종합적인 데이터셋을 사용하여 사전 훈련됩니다.

- **Performance Highlights**: VisionUnite는 다양한 임상 시나리오에서 뛰어난 성능을 보입니다. 존재하는 GPT-4V 및 Gemini Pro와 같은 기존 생성 모델보다 우수하며, 진단 능력은 초급 안과 전문의와 비슷한 수준입니다. VisionUnite는 다중 질병 진단, 임상 설명 및 환자 상호 작용 등의 다양한 임상 시나리오에서 잘 작동합니다. 또한 초급 안과 의사의 훈련을 돕는 교육적인 도구로 사용할 수 있습니다.



### Multistain Pretraining for Slide Representation Learning in Pathology (https://arxiv.org/abs/2408.02859)
Comments:
          ECCV'24

- **What's New**: Madeleine이라는 새로운 멀티모달 사전학습 전략이 도입되었습니다. Madeleine은 다양한 염색 마커를 사용한 슬라이드를 사용하여 풍부한 학습 신호를 형성하고, H&E 슬라이드와 다른 염색 마커를 같은 공간에 정렬함으로써 단일한 슬라이드 표현을 학습합니다. 이는 컴퓨터 병리학에서 유용한 보편적이고 전이 가능한 표현을 생성하는 데 중점을 둡니다.

- **Technical Details**: Madeleine은 멀티헤드 어텐션 기반의 다중 인스턴스 학습(MIL) 및 듀얼 글로벌 로컬 크로스 스테인 정렬 목적을 사용합니다. H&E 슬라이드와 다른 염색들 간의 슬라이드 수준 및 패치 수준 정렬을 학습합니다. 글로벌 목표는 대칭적 대립 상 손실(symmetric contrastive loss)을 기반으로 하여 슬라이드 간의 전반적인 모폴로지 구성을 캡슐화하고, 지역 목표는 그래프 최적 운송 프레임워크(Graph Optimal Transport framework)를 통해 세밀한 형태학적 특징을 일치시킵니다.

- **Performance Highlights**: Madeleine이 유방암 샘플(N=4,211 WSIs, 5 stains)과 신장 이식 샘플(N=12,070 WSIs, 4 stains)에서 사전학습되었고, 21개의 다양한 다운스트림 작업에서 평가되었습니다. 이 작업은 형태학적 하위 유형(morphological subtyping), 분자 하위 유형(molecular subtyping), 생존 예측(survival prediction), IHC 정량화를 포함하며, 다양한 의료 센터에서 수집된 7,299개의 WSIs를 사용했습니다.



### Analyzing Data Efficiency and Performance of Machine Learning Algorithms for Assessing Low Back Pain Physical Rehabilitation Exercises (https://arxiv.org/abs/2408.02855)
Comments:
          European Conference on Mobile Robots (2023)

- **What's New**: 이번 연구는 물리 재활 프로그램에서 로봇 코치 시스템을 활용한 인간 운동 분석에 집중하였습니다. 특히, RGB-D 카메라 (Microsoft Kinect)와 RGB 비디오 (OpenPose, BlazePose 알고리즘)에서 얻은 운동 데이터를 비교하여 재활 운동을 평가하는 것이 목표입니다.

- **Technical Details**: 연구는 Gaussian Mixture Model (GMM)을 이용하여 위치와 방향 특징을 평가합니다. GMM의 로그-가능도 값을 기반으로 성능 지표를 정의합니다. 중요한 기술적 요소로는 RGB-D 카메라와 RGB 이미지에서 추정된 2D 및 3D 인간 자세 (human pose) 추정이 포함됩니다. 이번 연구는 특히 로우 백 페인 (Low Back Pain) 환자들의 재활 운동을 평가하는 데 중점을 둡니다.

- **Performance Highlights**: 본 연구는 로봇 코치 Poppy가 이전에 코칭한 로우 백 페인 환자들의 의료 데이터베이스를 활용하여 성능을 평가합니다. 이를 통해 RGB-D 카메라와 RGB 비디오에서 추정된 운동 데이터가 재활 평가에 어떻게 활용될 수 있는지 비교 분석합니다.



### Mitigating Malicious Attacks in Federated Learning via Confidence-aware Defens (https://arxiv.org/abs/2408.02813)
- **What's New**: 연합 학습(Federated Learning, FL)은 클라이언트들이 자신의 데이터 공유 없이 글로벌 모델을 공동으로 학습할 수 있는 분산식 머신러닝 패러다임이다. 그러나, 악성 클라이언트들은 데이터 포이즈닝(data poisoning)과 모델 포이즈닝(model poisoning) 공격을 통해 시스템 성능을 저하시키는 위험이 있다. 이를 개선하기 위해, 클라이언트 모델 업데이트의 불확실성을 평가하는 모델 신뢰 점수(confidence scores)를 기반으로 악성 클라이언트를 탐지하고 방어하는 새로운 방법을 제안했다.

- **Technical Details**: 제안된 방법은 클라이언트 모델 업데이트의 불확실성을 평가하여 악성 업데이트를 탐지하고 방어하는 것이다. 구체적으로, 각 클라이언트의 신뢰 점수를 수집하고, 이를 바탕으로 불확실성 경계를 설정, 이후 높은 불확실성을 보이는 업데이트를 식별하여 적절히 처리하는 단계로 이루어진다. 이 방법은 모델 포이즈닝 공격뿐만 아니라 데이터 포이즈닝 공격에도 효과적으로 대응할 수 있으며, 다양한 공격 강도와 유형에 걸쳐 적합하다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 공격 유형에 대해 FL 시스템의 강인성을 크게 향상시키며, 높은 모델 정확도와 안정성을 달성하였다. 신뢰 점수를 통해 악성 공격을 탐지함으로써 기존 방어 방법보다 더 우수한 성능을 보였다. 본 방법은 여러 데이터셋을 통한 광범위한 실험으로 검증되었으며, 다양한 시나리오에서 FL 모델의 보안성과 성능을 강화하는 데 효과적이다.



### SiCo: A Size-Controllable Virtual Try-On Approach for Informed Decision-Making (https://arxiv.org/abs/2408.02803)
- **What's New**: 우리는 SiCo라는 새로운 온라인 가상 착용 시스템(SiCo)을 도입하였습니다. 이 시스템은 사용자들이 자신의 사진을 업로드하고, 다양한 사이즈의 의류를 자신의 몸에 맞춰 시각화 할 수 있도록 도와줍니다. 이를 통해 사용자들이 더 나은 구매 결정을 할 수 있습니다.

- **Technical Details**: SiCo 시스템은 Stable Diffusion과 IP-Adapter를 사용하여 선택한 의류 이미지를 사용자 이미지에 정확하게 겹쳐놓습니다. 또한, DensePose를 사용해 신체 윤곽을 추출하여 ControlNet을 통해 이미지 출력을 반영합니다. 이를 통해 기존 모델의 편향을 피하고, 다양한 신체 형태를 현실적으로 시각화할 수 있게 합니다.

- **Performance Highlights**: 48명의 참가자를 대상으로 한 사용자 연구 결과, SiCo의 사이즈 조절 기능이 사용자들이 자신의 몸에 맞는 옷의 착용 모습을 시각화하고 의류 선택에 대한 자신감을 높이는 데 매우 효과적임을 확인했습니다.



### RCDM: Enabling Robustness for Conditional Diffusion Mod (https://arxiv.org/abs/2408.02710)
- **What's New**: 새롭게 제안된 연구는 고전 제어 이론을 기반으로 한 강력한 조건부 확산 모델(Robust Conditional Diffusion Model, RCDM)을 제시합니다. 이는 노이즈의 영향을 동적으로 줄여 모델의 견고성을 향상시킵니다. RCDM은 두 개의 신경망 간의 협력적 상호작용과 최적 제어 전략을 활용하여 샘플링 과정 중 두 신경망의 가중치를 최적화합니다. 이러한 접근 방식은 기존 기법과 달리 추가적인 계산 오버헤드를 발생시키지 않으면서도 고정 오류와 신경망 가중치 간의 수학적 관계를 마련합니다.

- **Technical Details**: 고전 제어 이론의 최적 제어 전략을 도입하여 두 개의 신경망의 가중치를 동적으로 최적화합니다. 두 신경망(하나는 조건부 신경망, 다른 하나는 무조건 신경망) 간의 협력적 상호작용을 통해 샘플링 과정에서 오류를 최소화합니다. 이 접근 방식을 통해 오류가 누적되어도 최종 생성 결과에 미치는 영향을 제어할 수 있습니다.

- **Performance Highlights**: MNIST와 CIFAR-10 데이터셋을 대상으로 광범위한 실험을 진행한 결과, 제안된 모델이 효과적이고 적응력이 뛰어난 것으로 나타났습니다. 기존의 강건성 향상 기법들에 비해 추가적인 계산 복잡성이 없으면서도 더 높은 성능을 보였습니다.



### Scribble-Based Interactive Segmentation of Medical Hyperspectral Images (https://arxiv.org/abs/2408.02708)
- **What's New**: 최신 연구는 의료용 초분광 이미지(HSI)를 활용한 새로운 '낙서 기반(Interactive Segmentation) 상호작용 세분화' 프레임워크를 공개했습니다. 이 프레임워크는 딥러닝을 이용한 특징 추출과 지오데식 거리 맵을 결합하여 정확한 세분화 결과를 얻습니다.

- **Technical Details**: 제안된 방법은 사용자가 제공한 낙서(scribble)와 딥러닝으로 추출된 특징을 기반으로 지오데식 거리 맵(Geodesic Distance Map)을 생성합니다. 이를 통해 원본 HSI, 재구성된 RGB 이미지, 혹은 유클리드 거리 맵(Euclidean Distance Map)보다 더 나은 세분화 결과를 얻을 수 있습니다. U-Net 기반의 딥러닝 네트워크를 사용하여 HSI로부터 특징을 추출하고, 낙서를 기반으로 지오데식 거리 맵을 생성하여 세분화 결과를 도출합니다.

- **Performance Highlights**: HeiporSPECTRAL 데이터셋을 사용하여 실험을 수행한 결과, 제안된 방법이 최고 Dice 계수에서 다른 전통적인 방법들보다 우수한 성능을 보였습니다. 특히, 딥러닝 기반의 특징 지오데식 거리 맵이 가장 높은 평균 최대 Dice 점수인 0.842를 기록하여, 정확성과 유연성이 크게 향상되었습니다.



### On Biases in a UK Biobank-based Retinal Image Classification Mod (https://arxiv.org/abs/2408.02676)
Comments:
          To appear at MICCAI FAIMI Workshop 2024

- **What's New**: 최근 연구는 의료 분야에서 머신러닝 모델의 성능 차이 문제를 밝혀냈습니다. 이번 연구에서는 UK Biobank의 망막 이미지를 사용해 질병 분류 모델을 훈련, 평가하며 다양한 인구 집단 간의 성능 차이를 조사했습니다. 특히, 특정 평가 센터에서의 불공정한 성능이 확인되었고, 이는 엄격한 데이터 표준화 프로토콜에도 불구하고 발생한 현상입니다.

- **Technical Details**: 연구는 UK Biobank에서 80,966개의 망막 이미지를 사용해 진행되었습니다. 연구진은 InceptionV3 네트워크를 사용해 고혈압 분류 모델을 훈련시켰으며, 다양한 bias mitigation methods(편향 완화 방법)을 적용해 성능 평가 및 차이 원인을 분석했습니다. 사용된 방법에는 Resampling, Group Distributionnally Robust Optimisation (GroupDRO), Orthogonally Disentangled Representations (ODR), Domain-Independent learning (DomainInd), Learning-Not-to-Learn (LNL), SWAD, ReSWAD, 그리고 Recalibration이 포함되었습니다.

- **Performance Highlights**: 기본 InceptionV3 모델은 73±0.01%의 정확도와 71±0.00%의 AUC를 기록했습니다. 하지만 세부 평가에서 나이, 평가 센터, 성별 등 여러 하위 그룹에서 상당한 성능 차이가 발견되었습니다. 예를 들어, 나이 그룹 간 AUC는 15% 이상, 평가 센터 간은 10% 이상 차이가 있었습니다. 또한, 주어진 편향 완화 방법들이 일관되게 모델의 공정성을 향상시키는 데 실패했음을 밝혔습니다. 이는 특정 유형의 편향에 맞춘 더 나은 방법이 필요함을 시사합니다.



### On Feasibility of Intent Obfuscating Attacks (https://arxiv.org/abs/2408.02674)
Comments:
          31 pages, 18 Figures. Includes technical appendix. To be published in AIES 2024

- **What's New**: 이 논문은 기계 학습 시스템에 대한 적대적 공격에서 의도 은폐(intent obfuscation)를 처음으로 제안합니다. 이 방법은 대상 객체(target object)를 방해하기 위해 다른 비중첩 객체(non-overlapping object)를 교란하여 공격자의 의도를 숨기는 것입니다.

- **Technical Details**: 우리는 요즘 널리 사용되는 TOG(Targeted Objectness Gradient) 알고리즘을 사용하여 적대적 예제를 생성했습니다. 이 알고리즘은 YOLOv3, SSD, RetinaNet, Faster R-CNN, Cascade R-CNN과 같은 주요 객체 탐지기(object detectors)에 대해 효과적으로 적용됩니다. TOG는 반복적 그래디언트 기반 방법이며, 목표한 객체를 정확히 지정할 수 있도록 합니다.

- **Performance Highlights**: 모든 모델과 공격 유형에서 성공적으로 의도 은폐 공격을 수행했습니다. 공격의 성공 요인으로는 대상 객체의 신뢰도(confidence)와 교란 객체의 크기가 있습니다. 우리는 이러한 요인을 이용하여 여러 모델과 공격 모드에서 성공률을 증가시킬 수 있음을 증명했습니다.

- **Importance**: 기계 학습 커뮤니티가 이러한 의도 은폐 공격을 이해하고 방어하는 것이 필수적이며, 이는 공격자가 책임 회피를 가능하게 하고 기존의 공격 방지 알고리즘과 차별화되는 새로운 위협을 제기합니다.



New uploads on arXiv(cs.AI)

### Anytime Multi-Agent Path Finding with an Adaptive Delay-Based Heuristic (https://arxiv.org/abs/2408.02960)
Comments:
          arXiv admin note: text overlap with arXiv:2312.16767

- **What's New**: ADDRESS라는 새로운 단일 파괴 휴리스틱(Destroy Heuristic) 변형을 제안하였습니다. 이는 기존의 MAPF-LNS 기법의 성능을 개선하는데 목표를 두고 있으며, 성공 기반 자기 학습(Success-based Self-Learning)을 통해 적응 지연 기반 파괴 및 복구 기법을 적용합니다.

- **Technical Details**: ADDRESS는 제한된 톰슨 샘플링(Restricted Thompson Sampling)을 사용하여 적응형 LNS(Large Neighborhood Search) 이웃 생성을 위한 시드 에이전트를 선택합니다. 이는 가장 지연된 상위-K 에이전트 집합을 기반으로 동작하며, 일반적인 에이전트 기반 파괴 휴리스틱이 성능 병목을 일으킬 수 있는 문제를 해결합니다. MAPF-LNS의 기존 변형과 달리, ADDRESS는 단일 파괴 휴리스틱을 사용하여 성능을 최적화합니다.

- **Performance Highlights**: ADDRESS는 MAPF 벤치마크 세트에서 여러 맵을 대상으로 평가되었습니다. 그 결과, 최대 천 개의 에이전트를 포함하는 대규모 시나리오에서 COST를 최소 50%까지 향상시키는 성능을 보였습니다. 이는 기존의 MAPF-LNS와 기타 최첨단 방법들에 비해 유의미한 개선을 나타냅니다.



### Compromising Embodied Agents with Contextual Backdoor Attacks (https://arxiv.org/abs/2408.02882)
- **What's New**: 이번 논문에서는 'Contextual Backdoor Attack' 개념을 처음으로 도입하여, 코드 기반의 체화된 지능 (embodied intelligence)을 대상으로 하는 LLM(Large Language Model)에 대해 중대한 보안 위협을 밝혀냈습니다. 이를 통해 몇 가지 'poisoned demonstrations' (오염된 시연)만으로 LLM이 결함이 있는 프로그램을 생성하도록 유도할 수 있음을 보여줍니다. 이러한 프로그램은 특정 트리거에 반응하여 예상치 못한 동작을 유도할 수 있습니다.

- **Technical Details**: LLM의 컨텍스트 환경을 손상시키기 위해, 'adversarial in-context generation'(적대적 컨텍스트 내 생성) 기법을 사용합니다. 주어진 목표와 에이전트의 작업에 따라 LLM judge(판사)가 오염된 시연을 평가하고 추가 LLM으로 보고하여, 체인의 사고(reasoning) 방식으로 두 플레이어 간의 적대적 게임으로 시연을 최적화합니다. 다운스트림 에이전트가 컨텍스트 의존 동작을 수행할 수 있도록, 텍스트 트리거와 비주얼 트리거를 결합한 'dual-modality activation strategy'(이중 모달리티 활성화 전략)을 설계하였습니다.

- **Performance Highlights**: 로봇 플래닝(ProgPrompt), 로봇 조작(VoxPoser), 합성 비주얼 추론(Visual Programming)과 같은 다양한 작업에서 포괄적인 실험을 통해 제안된 공격의 효과를 입증하였습니다. 또한, 실제 자율주행 시스템에 대한 성공적인 공격도 포함되었습니다. 이를 통해 실제 환경에서 LLM 응용 프로그램을 대상으로 하는 잠재적 위협을 인지시키는 것을 목표로 합니다.



### Development of REGAI: Rubric Enabled Generative Artificial Intelligenc (https://arxiv.org/abs/2408.02811)
- **What's New**: 이 논문은 새로운 검색 증강 생성(augmentation generation, RAG) 및 대형 언어 모델(large language model, LLM) 기반 인공지능(AI) 기법: 루브릭 지원 생성 인공지능(rubric enabled generative artificial intelligence, REGAI)을 소개하고 평가합니다. REGAI는 평가 목적을 위해 LLM의 성능을 향상시키기 위해 수동으로 또는 시스템에 의해 자동으로 생성될 수 있는 루브릭을 사용합니다.

- **Technical Details**: REGAI는 고전적인 LLM 및 RAG 기반 LLM 기술 모두의 성능을 향상시킵니다. 논문에서는 REGAI의 구조를 설명하고 그 성능에 관한 데이터를 제시하며, 이 기술의 여러 가능한 응용 분야를 논의합니다.

- **Performance Highlights**: REGAI는 고전적 LLM 및 기존 RAG 기반 LLM 기법보다 우수한 성능을 보입니다. 성능 데이터는 논문에서 자세히 다룹니다.



### Recording First-person Experiences to Build a New Type of Foundation Mod (https://arxiv.org/abs/2408.02680)
Comments:
          5 pages, 5 figures, 3 tables. arXiv admin note: substantial text overlap with arXiv:2408.00030

- **What's New**: 최근 몇 년 동안 기초 모델(Foundation Models)이 인공지능(AI) 분야에서 큰 영향을 미치고 있습니다. 그러나 인터넷 데이터를 기반으로 훈련된 기존의 기초 모델들이 곧 데이터 고갈 문제에 직면할 것으로 예상되고 있습니다. 이에 반하여, 인간의 실제 감정 및 생리 반응을 포착하는 새로운 데이터 수집 장비를 개발하여 보다 현실적인 인간 행동을 모사하는 기초 모델을 제안했습니다.

- **Technical Details**: 이 논문에서는 사람이 보고 듣는 것뿐만 아니라 피부 전도율(GSR), 얼굴 표정, 두뇌 상태(14채널 EEG)를 동시에 기록할 수 있는 장치를 개발했습니다. 이 장치를 통해 수집된 데이터는 AI 알고리즘을 사용해 처리되며, 이를 통해 주제의 환경과 내적 상태를 다각도로 분석합니다. 궁극적으로 이러한 데이터를 이용해 기초 모델을 훈련하면, 기존의 개인성 모델들보다 인간 행동을 훨씬 더 정확하게 복제할 수 있습니다.

- **Performance Highlights**: 제안된 첫 번째 기초 모델(FPFM)은 다양한 응용 분야에 활용 가능성이 큽니다. 개인 비서, 추천 엔진, 생성적 적대 신경망(GAN) 시스템, 데이팅 및 채용 분야에 사용될 수 있습니다. 데이터 수집과 모델 훈련이 매우 비용이 많이 들기 때문에, 현재 프로젝트의 다음 단계를 위해 자금을 모으기 위한 스타트업 런칭이 준비 중입니다.



### LLaVA-OneVision: Easy Visual Task Transfer (https://arxiv.org/abs/2408.03326)
Comments:
          Project Homepage: this https URL

- **What's New**: LLaVA-OneVision은 LLaVA-NeXT 블로그 시리즈에서 얻어진 통찰을 바탕으로 개발된 대형 멀티모달 모델(LMM)입니다. 특히 단일 이미지, 다중 이미지, 비디오 시나리오 등 세 가지 중요한 컴퓨터 비전 시나리오에서 성능 경계를 확장하는 첫 단일 모델로써 중요한 이정표를 세웠습니다. 또한 이미지를 비디오로 전이 학습(task transfer)하여 새로운 능력을 발휘할 수 있습니다.

- **Technical Details**: LLaVA-OneVision의 모델 아키텍처는 LLM과 비전 인코더를 연결하는 간단한 모듈을 사용해 설계되었습니다. 주 사용된 LLM은 Qwen-2이며 비전 인코더로는 SigLIP를 사용합니다. 또한, 이미지 특징을 단어 임베딩 공간으로 투사하는 2층 MLP인 Projector를 사용합니다. 이 모델은 AnyRes 기술을 통해 높은 해상도의 이미지를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLaVA-OneVision은 단일 이미지, 다중 이미지, 비디오 시나리오 모두에서 성능을 발휘하며, 단일 모델이 다양한 시나리오에서 우수한 성능을 보이는 희귀한 사례로 인정받고 있습니다. 특히 영상 이해 및 시나리오 간 전이 학습을 통해 이전에는 가능하지 않았던 새로운 능력을 보여줍니다. 오픈 소스로 공개된 모델로써 관련 멀티모달 지침 데이터, 코드베이스, 모델 체크포인트 및 시각적 채팅 데모가 포함되어 있어 앞으로 더욱 확장 가능성이 높습니다.



### Training LLMs to Recognize Hedges in Spontaneous Narratives (https://arxiv.org/abs/2408.03319)
Comments:
          Amie Paige, Adil Soubki, and John Murzaku contributed equally to this study

- **What's New**: 이번 연구는 로드러너 만화 이야기로 구성된 실험적 코퍼스를 기반으로, BERT와 GPT-4o 및 LLaMA-3을 사용하여 헤지(Hedge) 감지 성능을 평가했습니다. 가장 우수한 성능은 BERT 모델의 미세 조정(fine-tuning)을 통해 달성되었으며, 이를 통해 헤지의 애매모호한 사례를 식별하고 향후 연구 방향을 지도하는 데 도움을 줄 수 있는 LLM-in-the-Loop 접근법을 제안합니다.

- **Technical Details**: 헤지는 발화자가 발화를 임시적(provisional)으로 표시하는 다양한 형태의 담화 표지입니다. 연구에서는 BERT 모델을 미세 조정(fine-tuning)하고, GPT-4o와 LLaMA-3를 사용해 zero-shot 및 few-shot 학습을 비교 평가했습니다. 로드러너 헤지 코퍼스라는 표준 데이터를 활용하여, LLM를 사용한 코드의 정확성을 높였으며, 오류 분석을 통해 헤지 감지 실패 원인을 파악하고 LLM-in-the-Loop 접근법을 사용하여 표본 조사를 보강했습니다.

- **Performance Highlights**: BERT 모델을 미세 조정한 방법이 가장 우수한 성능을 보였으며, 그 다음으로 few-shot GPT-4o가 우수한 성능을 보였습니다. 상세한 오류 분석을 통해 헤지 감지의 실패 지점을 식별하고 이를 개선하기 위한 LLM-in-the-Loop 접근법을 도입했습니다.



### Fusing Forces: Deep-Human-Guided Refinement of Segmentation Masks (https://arxiv.org/abs/2408.03304)
Comments:
          16 pages, accepted at ICPR2024

- **What's New**: 에트루리아 거울에 새겨진 복잡한 그림을 분석하고 문서화하는 과정에서 수동으로 일러스트레이션을 추적하는 작업의 노동력과 비용을 감소시키기 위해 인간 상호작용 기반의 심층 신경망을 제안했습니다. 초기 예측을 바탕으로 인간 안내를 통해 주석(annotations)을 정제(refinement)하도록 훈련된 이 네트워크는 최대 75%의 인력 절감을 가능하게 하며, 수동 라벨링에 비해 최대 26% 더 빠른 품질 향상을 제공합니다.

- **Technical Details**: 이 방법론은 포토메트릭-스테레오 스캐닝(photometric-stereo scanning)과 심층 신경망(deep neural networks)을 결합하여 자동으로 세분화 과정(segmentation process)을 수행합니다. 데이터를 전처리하고, 거울의 깊이 맵(depth maps)을 사용하여 의도적인 선(line)을 인식하도록 모델을 훈련시켰습니다. 그 결과물의 세밀한 정제를 위해 인간의 상호작용(human-in-the-loop approach)을 추가하여 초기 예측에서 시작해 정제 과정에서 팁, 즉 부분적으로 추가하거나 삭제하는 과정을 거쳤습니다.

- **Performance Highlights**: 수동으로 초기 세분화를 정제하는 것과 비교했을 때, 인력 투입이 최대 75% 감소하고, 수동 라벨링에 비해 최대 26% 더 빠른 품질 향상을 달성했습니다. 이는 복잡한 선을 구분하는 작업에서 큰 성과를 보였으며, 에트루리아 거울뿐만 아니라 다양한 응용 분야에 적용이 가능합니다.



### Understanding How Blind Users Handle Object Recognition Errors: Strategies and Challenges (https://arxiv.org/abs/2408.03303)
- **What's New**: 이번 연구는 시각 장애인이 객체 인식 시스템을 사용할 때 발생하는 오류를 식별하고 회피하는 데 겪는 어려움을 탐구했습니다. 이를 위해 URCam이라는 객체 인식 시스템을 세밀하게 조정하여 사용자 연구를 수행했습니다. 참가자 12명을 대상으로 심층 인터뷰와 오류 식별 과제를 통해 사용자 경험을 분석하고, 최적의 인터페이스 설계에 대한 제안을 도출했습니다.

- **Technical Details**: 연구는 반구조화된 원격 인터뷰와 참가자 자택에서의 오류 식별 과제 형태로 진행되었습니다. URCam은 15개 객체 자극을 인식하도록 모델이 미세 조정되어, 객체 레이블 또는 인식 확신도가 낮을 경우 '모르겠음' 응답을 제공합니다. 참가자들은 다양한 관점, 배경, 객체 크기를 조정하여 오류를 식별하고 회피하는 전략을 사용했습니다.

- **Performance Highlights**: 참가자들은 평균적으로 절반만 오류를 식별했으며, 대부분이 false positives였습니다. 첫 번째 시도와 두 번째 시도에서 식별된 오류 비율에는 큰 차이가 없었으며, 두 번째 시도에서는 결정 속도가 빨라졌습니다. 인터뷰에서는 대부분의 참가자들이 독립적으로 사진 품질을 검토하고 오류를 식별하는 것을 선호했으며, 시각적 검증이 불가능한 경우 상황 정보를 활용했습니다.



### KaPO: Knowledge-aware Preference Optimization for Controllable Knowledge Selection in Retrieval-Augmented Language Models (https://arxiv.org/abs/2408.03297)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 지식 선택 능력을 향상시키기 위해 'Knowledge-aware Preference Optimization' (KaPO)을 제안합니다. KaPO는 실제 검색 시나리오에서 발생할 수 있는 오류 타입을 탐색하고, 이를 통해 부정적인 신호를 피하는 방법을 학습합니다. 이를 통해 LLM이 지식 충돌 상황에서 보다 강력한 성능을 발휘할 수 있도록 돕습니다.

- **Technical Details**: KaPO의 핵심 전략은 'preference optimization'을 통해 부정적인 신호를 감소시키는 것입니다. 다채로운 컨텍스트 조합을 통해 발생할 수 있는 오류 타입을 시뮬레이션하고, 이를 통해 LLM이 적절한 답변을 선택하도록 유도합니다. 동시에 응답 길이와 다양한 행동 패턴을 대표하는 preference 데이터의 비율을 조정함으로써 모델의 순응 능력과 잡음에 대한 견고성을 균형 있게 개선합니다.

- **Performance Highlights**: 실험 결과, KaPO는 기존의 방법들에 비해 지식 충돌을 처리하는 능력이 37% 이상 향상되었으며, 다양한 out-of-distribution 데이터셋에서도 강력한 일반화 성능을 보여주었습니다.



### Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability (https://arxiv.org/abs/2408.03292)
- **What's New**: 이번 연구에서는 U-Net 신경망 모델에 attention gates를 결합한 AttUNet을 제안합니다. 이 모델은 빠르고 정확한 이미지 기반의 정적 IR drop 예측을 위해 설계되었습니다. 또한, 예측된 IR drop 이미지의 고강하(pixel) 영역을 설명할 수 있는 saliency maps 기법을 도입하여, 특정 입력 픽셀이 IR drop에 얼마나 기여하는지를 빠르게 파악할 수 있게 했습니다.

- **Technical Details**: AttUNet은 U-Net 아키텍처를 기반으로 하며, attention gates를 적용하여 중요한 입력 데이터를 선택적으로 강조할 수 있습니다. 이는 종종 sparse(희소한)한 이류 드롭 지도에 유용합니다. 또한, 다중 이미지에서 단일 이미지로 예측하는 특성을 고려하여 초기 per-image 필터를 도입하여 사전 처리(convolutional block)를 추가했습니다. 두 단계의 훈련 과정 - 인공적으로 생성된 데이터로 초기 훈련을 하고, 실제 디자인의 몇몇 데이터를 사용해 fine-tuning을 합니다. Saliency maps는 각 입력 픽셀의 기여도를 평가하고 최적화를 안내합니다.

- **Performance Highlights**: ICCAD 2023 대회 우승자 및 U-Net과 비교했을 때, AttUNet은 MAE에서 평균 18%(53%) 개선, F1 점수에서 평균 14%(113%) 개선을 보였습니다. 또한, 예측된 높은 IR drop 픽셀 수를 평균 18% 줄이는 성과를 거두었습니다.



### StructEval: Deepen and Broaden Large Language Model Assessment via Structured Evaluation (https://arxiv.org/abs/2408.03281)
Comments:
          ACL 2024;Benchmark at this https URL at this https URL

- **What's New**: StructEval이라는 새로운 평가 프레임워크를 제안하였습니다. 이는 기존의 단일 항목 평가 패러다임의 한계를 극복하고, 대규모 언어 모델(LLM)의 실제 능력을 평가하기 위한 다단계 평가 접근 방식을 채택합니다. StructEval은 여러 인지 수준과 중요한 개념에 대해 구조화된 평가를 통해 LLM의 성능을 포괄적이고 일관되게 평가할 수 있습니다.

- **Technical Details**: StructEval은 하나의 기본 검사 목표에서 시작하여 Blooms Taxonomy Theory와 개념 매핑 이론에 따라 여러 인지 수준과 핵심 개념을 추출합니다. 이를 통해 모델의 각 테스트 목표에 대해 여러 인스턴스를 생성하여 평가합니다. 이는 단일 항목 평가와 달리 모델의 이해 능력을 다각도로 평가할 수 있게 합니다.

- **Performance Highlights**: StructEval은 데이터 오염 위험을 효과적으로 저항하고 잠재적인 편향의 간섭을 줄이며, 모델 간의 일관된 평가 결론을 제공합니다. 실험 결과, 이전의 증강 기반 전략(단어 변형, 패러프레이징(Paraphrasing), 역번역, 옵션 셔플 등)보다 우수한 성능을 보였습니다. 또한, 대규모 벤치마크를 자동으로 생성하고 높은 품질의 평가를 보장할 수 있습니다.



### Compress and Compare: Interactively Evaluating Efficiency and Behavior Across ML Model Compression Experiments (https://arxiv.org/abs/2408.03274)
Comments:
          Accepted to VIS 2024

- **What's New**: 이번 아카이브 논문에서는 기계 학습(Machine learning, ML) 모델의 압축(compression) 실험을 비교 분석할 수 있는 새로운 대화형 시각화 시스템인 'Compress and Compare'를 소개합니다. 이 시스템은 단일 인터페이스에서 압축 모델 간의 관계를 시각화하고, 모델의 예측, 가중치(weights), 활성화(activations)의 변화를 비교하여 최적의 압축 전략을 찾는 데 도움을 줍니다.

- **Technical Details**: 모델 압축은 주로 세 가지 기법으로 나누어집니다: (1) 양자화와 팔레티제이션(quantization and palettization), (2) 가지치기(pruning), (3) 팩터라이제이션과 증류(factorization and distillation). 대표적인 예로, 높은 정밀도의 포맷을 더 낮은 정밀도의 포맷으로 변환하는 양자화와 작은 절대 값을 가지는 가중치를 제거하는 가지치기가 있습니다. 또한, 압축 후 모델을 재조정하기 위해 미세 조정(fine-tuning)과 보정(calibration)을 사용합니다.

- **Performance Highlights**: Compress and Compare 시스템은 두 가지 사례 연구를 통해 압축 분석 작업을 지원하는 방법을 보여줍니다. 하나는 생성적 언어 모델에서의 압축 실패 디버깅이고, 다른 하나는 이미지 분류 모델에서의 압축 아티팩트(artifacts) 식별입니다. 여덟 명의 압축 전문가를 대상으로 한 사용자 연구를 통해 이 시스템이 압축 워크플로우(workflow)에 구조를 제공하고, 실무자가 압축에 대한 직관을 쌓으며, 모델 동작 변화에 대한 철저한 분석을 독려할 수 있음을 부각합니다.



### Unveiling Factual Recall Behaviors of Large Language Models through Knowledge Neurons (https://arxiv.org/abs/2408.03247)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 추론 작업에서 사실적 지식을 내부적으로 기억하고 활용하는지 조사합니다. 연구 결과, LLMs는 특정 상황에서 중요한 사실적 연관성을 잘 활용하지 못하고 오히려 지름길 같은 경로를 선택하는 경향이 있음을 밝혔다. CoT(Chain-of-Thought) 프롬프팅이 사실 지식을 강하게 기억하도록 유도하는 데 효과적임을 보여주었고, 매개 변수 지식의 수동적 조작을 통해 추론 성능을 향상시키는 법을 검토했다.

- **Technical Details**: 사실 기억에서 LLMs의 내부 동적 과정을 조사하기 위해 'Knowledge Neurons(KNs)' 기술을 활용하여 두 단계 추론(2-hop reasoning) 질문 데이터셋을 작성하고, 각 추론 단계에서 새로운 지표인 'KN 점수'를 도입해 평가했다. 추후 사실 지식의 회수 과정에서 CoT 프롬프팅이 통제 조건(조작 전후)에 어떤 영향을 미치는지 분석했다.

- **Performance Highlights**: ['LLMs는 사실 연관성을 잘못 기억하여 추론 오류의 1/3 이상을 차지한다.', 'CoT 프롬프팅을 통해 단계별 추론을 촉진함으로써 사실 지식의 회수를 현저히 개선할 수 있다.', '사실 회수 과정을 강화하거나 억제함으로써 추론 성능에 중요한 영향을 미친다.', '외부 지식의 충돌이 있을 때, 사실 회수 과정이 더 개선될 수 있다.']



### Personalizing Federated Instrument Segmentation with Visual Trait Priors in Robotic Surgery (https://arxiv.org/abs/2408.03208)
Comments:
          9 pages, 3 figures, under review

- **What's New**: 이 논문은 기존의 개인화 연합 학습(Personalized Federated Learning, PFL) 방법들이 다루지 않았던, 외관 다양성과 도구 모양 유사성에 기반해, 수술 도구 분할(Surgical Instrument Segmentation, SIS)를 강화하는 새로운 PFL 방법인 PFedSIS를 제안합니다. PFedSIS는 세 가지 핵심 요소를 포함합니다: 전역-개인화 분리(Global-Personalized Disentanglement, GPD), 외관 조절 개인화 강화(Appearance-Regulation Personalized Enhancement, APE), 모양 유사성 전역 강화(Shape-Similarity Global Enhancement, SGE). 이를 통해 각 사이트의 SIS 성능을 향상시킵니다.

- **Technical Details**: PFedSIS는 세 가지 주요 컴포넌트를 포함합니다: (1) GPD는 다중 헤드 셀프 어텐션의 헤드 별 및 채널 별 개인화를 처음으로 시도합니다. 이를 통해 각 사이트 간의 상호 차이를 점진적으로 활용하면서 고유한 외관 표현을 유지합니다. (2) APE는 외관 조절을 도입하며, 하이퍼네트워크를 통해 각 사이트의 개인화된 파라미터를 맞춤형으로 집계합니다. (3) SGE는 도구의 상호 모양 정보를 유지하고, 이미지 레벨에서 스타일 간 모양 일관성을 강화하며, 예측 레벨에서 각 사이트의 모양 유사성 기여도를 계산하여 전역 파라미터를 업데이트합니다.

- **Performance Highlights**: PFedSIS는 최신 방법들보다 Dice에서 +1.51%, IoU에서 +2.11%, ASSD에서 -2.79, HD95에서 -15.55 성능 향상을 달성합니다. 이는 세 곳의 공개 벤치마크 데이터셋 실험 결과로 검증되었습니다. 해당 코드와 모델은 추후 공개될 예정입니다.



### Adversarial Safety-Critical Scenario Generation using Naturalistic Human Driving Priors (https://arxiv.org/abs/2408.03200)
Comments:
          Published in IEEE Transactions on Intelligent Vehicles, 2023

- **What's New**: 이번 논문에서는 자연스러운 인간 운전 프라이어(priors)와 강화학습 기법(reinforcement learning techniques)을 활용하여 자율주행 차량 평가에서 필수적인 안전 중요한 테스트 시나리오를 생성하는 솔루션을 제안합니다. 이를 통해 우리는 다양하고 현실적인 대규모 테스트 시나리오를 얻을 수 있습니다.

- **Technical Details**: 특히, 자연 교통 상호작용 시나리오를 모방한 시뮬레이션 환경을 구축하였습니다. 이 환경을 바탕으로, 기존의 규칙 기반 모델(e.g., IDM(Intelligent Driver Model), MOBIL(Minimizing Overall Braking Induced by Lane changes) 모델)을 사용하여 실제 데이터셋에서 주요 제어 매개변수를 대략적으로 포착하고 조정하는 첫 번째 단계를 구현하였습니다. 다음으로, 우리는 GAIL(Generative Adversarial Imitation Learning)을 활용하여 운전자 행동을 연속적으로 나타내었습니다. 파생된 GAIL은 PPO(Proximal Policy Optimization)-기반의 actor-critic 네트워크 프레임워크를 설계하여 보상 함수를 미세 조정하고 우리의 자연 적대 시나리오 생성 솔루션을 최적화하는 데 사용되었습니다.

- **Performance Highlights**: NGSIM 데이터셋을 사용하여 3,000대 이상의 차량 궤적을 포함한 광범위한 실험을 수행했습니다. 충돌률, 가속, 스티어링, 차선 변경 횟수 등 주요 교통 매개변수를 기준 여러 베이스라인 모델과 비교하여 측정하였습니다. 우리의 연구 결과는 제안된 모델이 자연스러움과 공격성을 동시에 포괄하는 현실적인 안전 중요한 테스트 시나리오를 생성할 수 있음을 보여줍니다. 이는 자율주행 차량 개발의 기초가 될 수 있습니다.



### Training on the Fly: On-device Self-supervised Learning aboard Nano-drones within 20 mW (https://arxiv.org/abs/2408.03168)
Comments:
          This paper has been accepted for publication in the IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems. Copyright 2024 IEEE

- **What's New**: 이 논문에서는 초소형 자율 무인 항공기(nano-UAV)에서 도메인 이동에 따른 문제를 극복하기 위한 새로운 온디바이스(온디바이스) 파인튜닝(fine-tuning) 접근법을 제안합니다. 초소형 기기이기 때문에 컴퓨팅, 메모리, 감지 자원이 극히 제한적이며, 도메인 이동으로 인해 현실에서 모델 성능이 저하되는 문제가 존재합니다. 이를 해결하기 위해 자체 초저전력 GWT GAP9 시스템 온칩(System-on-Chip)을 활용하며, 자기 감독 학습(self-supervised learning) 방법을 적용하여 데이터 수집 문제를 해결합니다.

- **Technical Details**: 이 연구에서는 약 512장의 이미지를 사용하여 초저전력 GWT GAP9 SoC에서 온디바이스 파인튜닝을 수행합니다. 이 과정은 1MB의 메모리만 사용하며 19mW의 전력을 소모하거나 510ms 동안 작동합니다(38mW 기준). 파인튜닝 전략은 4가지로 나누어 다양한 메모리 및 컴퓨팅 요구 사항을 고려하여 성능을 평가합니다. 또한, 돼지 움직임 일관성(ego-motion consistency)을 기반으로 한 자기 감독 학습 방법을 사용하여 데이터를 수집하고 모델을 파인튜닝하는데 필요한 실제 데이터 레이블이 부족한 문제를 해결합니다.

- **Performance Highlights**: 현장 테스트 결과, 기존의 최첨단 PULP-Frontnet 대비 수평 위치 오류가 최대 26%까지 감소했습니다. 가장 어려운 환경에서 온디바이스 학습 과정은 임무 성공 여부를 결정짓는 차이를 만들어냈습니다. 또한, 제안된 방법을 통해 기존 학습 모델이 따를 수 없는 사용자 경로를 자체 학습 모델이 92%에서 100%까지 성공적으로 추적하는 것을 확인했습니다.



### Dilated Convolution with Learnable Spacings makes visual models more aligned with humans: a Grad-CAM study (https://arxiv.org/abs/2408.03164)
Comments:
          Accepted at The Trustworthy AI Workshop, IJCAI 2024

- **What's New**: 최근 발표된 논문에서는 Learnable Spacing (DCLS)을 가진 Dilated Convolution을 통해 수용 영역을 확장하면서도 매개 변수 수를 증가시키지 않아 다수의 컴퓨터 비전 벤치마크에서 표준 및 확장 합성곱을 능가하는 성과를 보여주었습니다. 이번 연구는 DCLS가 모델의 해석 가능성을 증가시킨다는 점에서 더 나아가 정량적 평가를 수행했습니다. 특히, 인간의 시각적 전략과의 일치를 의미하는 해석 가능성에서 Spearman 상관관계를 이용하여 이를 측정했습니다.

- **Technical Details**: GradCAM을 활용하여 모델의 GradCAM 히트맵과 ClickMe 데이터셋 히트맵 간의 Spearman 상관관계를 분석했습니다. ResNet50, ConvNeXt (T, S, B), CAFormer, ConvFormer 및 FastViT (sa 24 및 36) 등의 8개 참조 모델에서 DCLS로 표준 합성곱 레이어를 대체했습니다. 또한, CAFormer와 ConvFormer 모델에서 Grad-CAM이 무작위 히트맵을 생성하는 문제를 Threshold-Grad-CAM을 도입하여 해결했습니다.

- **Performance Highlights**: DCLS로 대체한 8개의 모델 중 7개 모델에서 해석 가능성 점수가 향상되었습니다. 또한 CAFormer와 ConvFormer 모델에서도 Threshold-Grad-CAM을 도입하여 해석 가능성을 크게 향상시켰습니다. 이로써 DCLS가 성능뿐 아니라 해석 가능성에서도 유의미한 개선을 가져오는 것을 확인했습니다.



### User-in-the-loop Evaluation of Multimodal LLMs for Activity Assistanc (https://arxiv.org/abs/2408.03160)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)를 활용한 현대 다중 모드 추론 모델의 역량을 조사하여, 다단계 일상 활동을 위한 비전 기반 어시스턴트를 지원하는 데 초점을 맞추고 있습니다. 특히, 이러한 어시스턴트는 카메라 같은 센서로부터 시각적 히스토리를 인코딩하고, 미래 행동을 예측하며, 사용자가 있는 상황에서 재계획을 할 수 있어야 합니다. 이를 위해 우리는 Socratic 모델과 Vision Conditioned Language Models (VCLMs)라는 두 가지 주요 다중 모드 LLM 접근 방식을 비교했습니다.

- **Technical Details**: 첫째, 우리는 오프라인 데이터셋을 사용하여 비디오 기반 액션 예측 작업에서 Socratic 모델과 VCLMs의 성능을 벤치마킹했습니다. Socratic 접근 방식은 사전 학습된 비전-언어 모델을 사용하여 비주얼 히스토리를 텍스트로 변환합니다. 반면 VCLMs는 연속적인 임베딩 정보와 텍스트 토큰을 결합하여 시각 정보를 임베딩 합니다. 두 번째로, 우리는 18명의 참가자들이 Aria라는 egocentric observation device를 착용하고, 다중 모드 LLM의 도움을 받으며 다단계 요리 활동을 수행한 최초의 사용자 연구를 수행했습니다.

- **Performance Highlights**: 연구 결과에 따르면 Socratic 접근 방식이 오프라인 및 온라인 설정에서 모두 VCLMs보다 우수한 성능을 보였습니다. 특히 긴 시각적 히스토리를 다루는 작업에서 Socratic 모델이 더 효과적이었습니다. 본 오프라인 메트릭은 온라인 성능을 정확히 반영하지 않으며, 온라인 설정에서는 사용자의 실시간 수행 타임을 고려한 계획 변경이 필요합니다.



### Optimizing Disease Prediction with Artificial Intelligence Driven Feature Selection and Attention Networks (https://arxiv.org/abs/2408.03151)
Comments:
          16 Pages, 4 Figures

- **What's New**: 이 논문에서는 다중 질병 예측(multi-disease prediction)의 새로운 접근 방식을 도입하는 포괄적인 연구를 소개합니다. 이 연구는 전자 건강 기록(EHR) 데이터의 통계적, 딥러닝, 최적화된 특징을 결합한 혁신적인 앙상블 특징 선택 모델(ensemble feature selection model)을 제안합니다. 모델의 핵심에 자리 잡은 안정화 에너지 밸리 최적화와 향상된 경계(Stabilized Energy Valley Optimization with Enhanced Bounds, SEV-EB) 알고리즘은 최적의 특징 선택을 도와줍니다.

- **Technical Details**: 제안된 모델은 통계적 특징(statistical features), 딥러닝 특징(deep features), 최적화된 선택 특징(optimally selected features)을 조합하여 예측 성능을 향상시키며, SEV-EB 알고리즘을 사용합니다. 이 알고리즘은 향상된 경계(enhanced bounds)와 안정화를 통해 모델의 견고성과 정확성을 높입니다. 또한 HSC-AttentionNet 네트워크는 딥 템포럴 컨볼루션(deep temporal convolution) 기능과 LSTM을 결합하여 건강 데이터의 단기 패턴과 장기 종속성을 모두 포착합니다.

- **Performance Highlights**: 제안된 모델은 다양한 질병 예측에서 95%의 정확도(accuracy)와 94%의 F1 스코어(F1-score)를 달성하며, 기존의 방법들을 능가합니다. 이는 질병 예측 정확성에서 중요한 진전을 의미합니다.



### COMMENTATOR: A Code-mixed Multilingual Text Annotation Framework (https://arxiv.org/abs/2408.03125)
- **What's New**: NLP 커뮤니티가 다국어 문제에 점점 더 많은 관심을 기울이고 있는 가운데, 다국어 데이터셋을 효율적으로 처리하기 위한 견고한 주석 도구가 필요합니다. 본 논문에서는 코드 혼합 텍스트(Code-Mixed Text)를 주석할 수 있는 다국어 텍스트 주석 프레임워크인 COMMENTATOR를 소개합니다. 이 도구는 Hinglish 텍스트에 대한 토큰 수준 및 문장 수준의 언어 주석 작업에 있어 그 효율성을 입증하였습니다.

- **Technical Details**: Hinglish 텍스트의 예시와 같은 코드 혼합은 비공식 대화 및 소셜 미디어에서 흔히 발견됩니다. COMMENTATOR는 토큰 수준의 언어 식별(Language Identification), 품사 태깅(POS Tagging), 문장 수준의 매트릭스 언어 식별(Matrix Language Identification)과 같은 주석 작업을 지원합니다. 이 프레임워크는 간편한 네비게이션, 사용자 생산성 증대, 빠른 클라우드 또는 로컬 설치, 주석자 피드백 통합을 통한 반복적 개선, 간단한 관리 인터페이스, 병렬 주석 등의 기능을 갖추고 있습니다.

- **Performance Highlights**: COMMENTATOR는 기존의 최첨단(SOTA) 주석 도구보다 5배 빠른 주석 속도를 보여줍니다. 이러한 속도 향상은 더욱 발전된 코드 혼합 라이브러리를 통합함으로써 더욱 증대될 수 있습니다. COMMENTATOR는 Hinglish 외의 다른 언어 쌍도 지원할 수 있도록 확장될 수 있습니다.



### Evaluating the Translation Performance of Large Language Models Based on Euas-20 (https://arxiv.org/abs/2408.03119)
Comments:
          15 pages, 8 figures

- **What's New**: 최근 몇 년 동안, BERT와 GPT와 같은 대형 언어 모델(LLMs)이 자연어 처리 작업에서 획기적인 성과를 거두었습니다. 이러한 모델의 발전 덕분에 기계 번역(MT) 성능도 크게 향상되었습니다. 그러나 여전히 해결해야 할 많은 도전 과제들이 있습니다. 이를 평가하기 위해, 우리는 다양한 언어로 구성된 Euas-20 데이터셋을 구축하여 대형 언어 모델의 번역 성능을 검토하고자 했습니다.

- **Technical Details**: 본 논문에서는 Llama2, Falcon, Vicuna, Mistral 등의 LLM들뿐만 아니라 Bloom, Bloomz, Gemma 같은 다국어 LLM을 평가 대상으로 삼았습니다. 데이터 누출을 방지하고 정확한 결과를 얻기 위해 Euas-20이라는 대표적인 20개 언어로 구성된 데이터셋을 구축했습니다. 이 데이터셋은 다양한 서체와 언어 그룹을 포함하며, 다국어 및 다영역 훈련 데이터를 사용해 모델의 일반화 능력을 향상시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: LLM들은 영어 번역에서 우수한 성능을 보였으나, 다른 언어들에서는 성능 차이가 있었습니다. 특히 영어와 유사한 언어에서 뛰어난 성능을 보였으며, zero-resource 언어에도 어느 정도의 번역 능력을 입증했습니다. 그러나 번역 결과가 유창하게 나오면서도 사용자를 오도할 수 있는 문제점이 있습니다. 이는 LLM들이 학습 과정에서 접하지 못한 단어들에 대해 정확하게 이해하거나 처리할 수 없는 경우가 있기 때문입니다.



### Learning Provably Robust Policies in Uncertain Parametric Environments (https://arxiv.org/abs/2408.03093)
- **What's New**: 본 논문은 불확실성(불명확한 분포(parameter with an unknown distribution))을 지닌 확률적 환경(stochastic environments)에서도 강건한 MDP (Markov Decision Process) 정책을 학습하는 데이터 기반 방법을 소개합니다. 새로운 보지 못한 환경에서도 아마도 대략적으로 정확한(PAC) 성능 보장을 제공합니다.

- **Technical Details**: 우리의 접근법은 유한 샘플(finite samples)의 MDP 환경으로부터 시작합니다. 각 환경에 대해 생성된 궤적(trajectories)의 탐색을 통해 모델의 근사치(interval MDP)를 구축합니다. 이를 통해 단일 정책을 합성하고, 이 정책이 보지 못한 새로운 환경에서 주어진 요구 사항을 충족하지 못할 위험을 경계합니다. 제안한 절차는 학습한 정책의 보장된 성능과 새로운 환경에서 보장을 충족하지 못할 위험 사이의 균형을 제공합니다. 또한, 환경의 상태 공간(state space) 및 그래프 구조(graph structure)에 대한 지식을 활용하고 추가적인 매개변수(parametric structure)를 최적화하여 학습을 최적화하고 적은 샘플로 더 좁은 보장을 얻는 방법을 제시합니다.

- **Performance Highlights**: 다양한 확립된 벤치마크에서 제안된 접근법을 평가한 결과, 성능이 뛰어나고 강건한 정책을 생성할 수 있으며, 성능과 관련된 위험을 정량적으로 엄격하게 계량화하는 보장을 제공합니다.



### QADQN: Quantum Attention Deep Q-Network for Financial Market Prediction (https://arxiv.org/abs/2408.03088)
Comments:
          Accepted at the 2024 IEEE International Conference on Quantum Computing and Engineering (QCE24), QCRL, September 2024

- **What's New**: 이 논문은 금융 시장 예측과 최적 거래 전략 개발의 어려움을 해결하기 위해, 양자 금융과 강화 학습을 결합한 양자-고전 하이브리드 알고리즘을 제안합니다. 특히, Quantum Attention Deep Q-Network (QADQN)을 도입하여 양자 강화 학습을 통해 시장의 복잡성과 변동성을 다루는 방식을 소개합니다.

- **Technical Details**: QADQN 아키텍처는 일반적인 딥 Q-러닝으로 변이 양자 회로 (Variational Quantum Circuit)를 사용하여 의사결정에서 양자의 이점을 활용합니다. 해당 논문은 Q-learning 알고리즘과 듀얼 쓰러스트 방법론에서 유도된 행동으로 초기화된 리플레이 버퍼를 통해 모방 학습 전략을 사용하여 탐색과 이용의 균형을 유지하려고 합니다. QADQN 에이전트는 부분 관측 가능 마르코프 결정 과정 (POMDP) 프레임워크 내에서 작동하며, 주식 시장의 불확실성과 부분 관찰성을 다루기에 적합합니다.

- **Performance Highlights**: QADQN 에이전트는 S&P 500 등 주요 시장 지수를 사용한 백테스팅을 통해 더 높은 효과를 보였며, 비중첩 및 중첩 테스트 기간 동안 각각 1.28 및 1.19의 Sortino 비율을 달성했습니다. 이는 하락 위험 관리를 효과적으로 수행했음을 나타냅니다.



### Enhancing Complex Causality Extraction via Improved Subtask Interaction and Knowledge Fusion (https://arxiv.org/abs/2408.03079)
Comments:
          NLPCC 2024 Oral

- **What's New**: Event Causality Extraction (ECE)에서 새로운 통합 프레임워크 UniCE가 제안되었습니다. UniCE는 두 개의 주요 모듈, 즉 이벤트 모듈과 관계 모듈로 구성되어 있으며 각 모듈은 여러 레이어로 구성되어 있습니다. 이 프레임워크는 복잡한 인과 관계 추출, 서브태스크 상호작용(Subtask Interaction) 및 지식 융합(Knowledge Fusion)의 문제를 동시에 해결함으로써 성능을 대폭 향상시켜줍니다.

- **Technical Details**: UniCE는 하나의 입력 문장에서 첫 번째 레이어가 외부 지식 그래프(Knowledge Graph, KG)를 사용하여 초기 배경 그래프를 구성하고, 이후 레이어들이 이 배경 그래프와 사전 학습된 언어 모델(Pretrained Language Models, PLMs)을 활용하여 점진적으로 이벤트와 인과 관계를 추출해내는 구조로 되어 있습니다. 각 레이어에서 이벤트 모듈은 시퀀스 레이블링 디코더를 사용하며, 이는 문장에서 각 토큰의 표현을 업데이트합니다. 그런 다음, 관계 모듈이 그래프 신경망(GNNs)을 활용하여 인과 관계를 판별합니다. 지식 융합 메커니즘은 이벤트와 KG를 동적으로 연결하여, 문장에서 추출한 각 요소와 관련된 지식을 효과적으로 통합합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(EventStoryLine, SCIFI, Causal-TimeBank)에서 UniCE는 최첨단 성능(State-of-the-Art, SOTA)을 기록하며, ChatGPT를 최소 30%의 F1 점수 차이로 앞섰습니다. 또한, 서브태스크 상호작용 및 지식 융합 메커니즘이 성능 향상에 크게 기여하는 것을 확인했습니다. 추가 실험에서는 UniCE가 ChatGPT의 성능을 문맥학습(In-Context Learning)을 통해 효과적으로 향상시킬 수 있음을 보여주었습니다.



### BodySLAM: A Generalized Monocular Visual SLAM Framework for Surgical Applications (https://arxiv.org/abs/2408.03078)
Comments:
          15 pages, 7 figures

- **What's New**: 이 연구는 기존의 문제를 해결하기 위해 혁신적인 딥러닝 기반 SLAM 접근법을 소개했습니다. 새로운 방법론은 CycleGAN 아키텍처를 바탕으로 한 무감독 방식의 Monocular Pose Estimation Module (MPEM)과 혁신적인 Zoe 아키텍처를 활용한 Monocular Depth Estimation Module (MDEM), 그리고 3D Reconstruction Module (3DM)을 포함합니다. 특히, Zoe를 통합한 MDEM은 현존하는 타 알고리즘보다 뛰어난 성능을 보이며, MPEM은 최단 추론 시간을 자랑합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성됩니다: 1) MPEM은 시계열 프레임 간 카메라 포즈를 추정하는 무감독 방법을 사용합니다, 2) MDEM은 여러 이미지 컨텍스트에서 우수한 일반화 성능을 보이는 Zoe 아키텍처를 기반으로 합니다, 3) 3DM은 점 구름(point clouds)을 생성하고 포즈를 정제하며 이를 체적 표현으로 변환하는 다단계 과정을 수행합니다.

- **Performance Highlights**: 제안된 방법론은 Hamlyn, EndoSLAM, SCARED 등 세 개의 공개 데이터셋을 활용하여 엄격하게 평가되었으며, EndoSFMLearner와 EndoDepth 두 가지 기존의 최첨단 방법과 비교 평가되었습니다. 그 결과 Zoe를 통합한 MDEM은 깊이 추정 알고리즘에서 뛰어난 성능을 보였으며, MPEM은 경쟁력 있는 성능과 가장 짧은 추론 시간을 기록했습니다. 이 접근법은 복강경, 위내시경, 대장내시경 등의 다양한 내시경 수술 시나리오에서 그 우수성을 입증했습니다.



### Solving QUBO on the Loihi 2 Neuromorphic Processor (https://arxiv.org/abs/2408.03076)
Comments:
          12 pages, 3 figures. Shared first authorship: Alessandro Pierro, Philipp Stratmann, and Gabriel Andres Fonseca Guerra

- **What's New**: 이번 논문에서는 Intel Loihi 2 뉴로모픽 프로세서를 활용한 Quadratic Unconstrained Binary Optimization (QUBO) 문제 해결 알고리즘을 소개합니다. 이 솔버는 Intel의 뉴로모픽 연구 칩 Loihi 2를 위해 개발된 하드웨어 인식의 세밀한 병렬 시뮬레이티드 어닐링(Simulated Annealing, SA) 알고리즘에 기반합니다.

- **Technical Details**: 이 알고리즘은 SA 알고리즘의 전통적인 형태에서 영감을 받아 개발되었으며, 각 뉴런은 이진 변수 변화를 인코딩하고 잠재적인 에너지 변화를 계산합니다. Loihi 2의 아키텍처 및 구현 세부사항과 이 알고리즘의 성능을 CPU에서 실행되는 두 개의 기준 알고리즘과 비교하여 설명합니다.

- **Performance Highlights**: 네이처 결과에 따르면 이 알고리즘은 문제의 크기가 최대 1000개의 변수까지 확장될 때 1ms 안에 실행 가능한 해결책을 생성할 수 있으며, CPU에서 실행되는 두 기준 솔버에 비해 최대 37배 더 에너지 효율적입니다. 이는 크기, 무게 및 전력 제약이 있는 엣지 컴퓨팅(Edge computing) 애플리케이션에 매우 유리할 수 있습니다.



### OpenOmni: A Collaborative Open Source Tool for Building Future-Ready Multimodal Conversational Agents (https://arxiv.org/abs/2408.03047)
- **What's New**: OpenOmni라는 오픈 소스, 엔드 투 엔드 파이프라인 벤치마킹 도구를 개발했습니다. 이 도구는 Speech2Text, Emotion Detection, Retrieval Augmented Generation(RAG), Large Language Models(LLMs)와 같은 첨단 기술을 통합하여 협업 개발 및 벤치마킹을 지원합니다.

- **Technical Details**: OpenOmni는 로컬 및 클라우드 배포를 지원하여 데이터 프라이버시와 지연 시간 및 정확성 벤치마킹을 보장합니다. 프레임워크는 비디오 및 오디오 데이터를 처리하고 사용자 맞춤형 에이전트 파이프라인을 통해 응답을 생성합니다. 주요 기술 요소는 Speech2Text, Emotion Detection, RAG, LLMs, TTS(Text-to-Speech)입니다.

- **Performance Highlights**: OpenOmni는 기존 시스템들의 성능인 200-250ms의 응답 시간에 비해 데이터 프라이버시와 벤치마킹 도구의 확장성을 제공하여 사용 사례에 맞게 최적화된 애플리케이션 개발을 가능하게 합니다. 이를 통해 시각 장애인을 위한 실내 보조 기능 등 다양한 애플리케이션에서 성능 향상을 도모할 수 있습니다.



### Highly Efficient Self-Adaptive Reward Shaping for Reinforcement Learning (https://arxiv.org/abs/2408.03029)
- **What's New**: 이 논문에서는 강화 학습에서 희귀한 보상을 해결하기 위한 새로운 자동 적응적 보상 형성 메커니즘, SASR(Self-Adaptive Success Rate) 방법을 제안합니다. 제안된 방법은 베타 분포(Beta distributions)를 사용하여 성공률을 모델링합니다. 베타 분포는 시간이 지남에 따라 높은 데이터 신뢰도로 발전하며 초기에는 탐험을 촉진하고 나중에는 활용도를 높이는 데 기여합니다.

- **Technical Details**: SASR 방법은 커널 밀도 추정(Kernel Density Estimation, KDE)을 랜덤 푸리에 특징(Random Fourier Features, RFF)과 결합하여 성공 및 실패 횟수를 효율적으로 추정합니다. 이를 통해 비모수적(non-parametric)이며 학습이 필요 없는 방식으로 고차원 연속 상태 공간에서도 효율적인 계산이 가능하게 합니다.

- **Performance Highlights**: 이 방법은 희소하고 지연된 보상이 있는 다양한 연속 제어 작업에서 평가되었으며, 샘플 효율성, 학습 속도 및 수렴 안정성 측면에서 여러 기준보다 상당한 향상을 보여주었습니다.



### Integrating Controllable Motion Skills from Demonstrations (https://arxiv.org/abs/2408.03018)
- **What's New**: 이번 연구는 다기능 동작을 요구하는 다각형 로봇의 기술적 문제를 해결하기 위해 Controllable Skills Integration(CSI)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 복잡한 보상 설계(Reward Engineering) 없이 다양한 동작 기술을 하나의 정책으로 통합할 수 있으며, 고차원 자연어 추론(NLI) 모듈을 통해 초보적인 언어 지시를 통한 기술 제어도 가능합니다.

- **Technical Details**: CSI는 기존의 강화 학습(RL) 기반 접근 방식과 달리 복잡한 보상 설계가 필요 없으며, Generative Adversarial Imitation Learning(GAIL)을 바탕으로 구축되었습니다. 이를 통해 기술 레이블을 제어 인터페이스로 사용하여 통합 동작 기술을 제어할 수 있습니다. 또한, Conditional Imitation Learning 및 Condition-Aware Loss와 같은 핵심 설계 요소를 통합하여 외부 지식(예: 자연어)을 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과 CSI는 다양한 동작 기술을 유연하게 통합하고, 기술 간 전환을 용이하게 하며, 통합해야 할 동작 기술의 수가 증가할 수록 그 확장성 또한 우수하다는 것을 입증했습니다. 또한, 고차원 NLI 모듈을 결합하면 언어 지시를 통한 기술 제어도 가능합니다.



### NeurDB: On the Design and Implementation of an AI-powered Autonomous Databas (https://arxiv.org/abs/2408.03013)
- **What's New**: NeurDB는 AI와 데이터베이스의 깊은 융합을 통한 AI 실시간 적응 데이터베이스를 소개합니다. 데이터 및 워크로드 변화에 대응하는 새로운 in-database AI 생태계를 구축하여 효율적이고 효과적인 in-database AI 분석을 가능하게 합니다.

- **Technical Details**: NeurDB에서는 AI 워크플로를 데이터베이스 내에 심층적으로 통합하여 모델 학습, 추론, 미세 조정 등의 여러 in-database AI 작업을 수행합니다. 이를 위해 데이터 스트리밍 프로토콜과 증분 업데이트 기법을 도입하여 AI 모델의 빠른 적응을 돕습니다. 또한, SQL 구문을 확장하여 PREDICT 구문을 추가함으로써 사용자가 쉽게 복잡한 AI 분석 작업을 제출할 수 있습니다.

- **Performance Highlights**: 경험적 평가 결과, NeurDB는 기존 솔루션들에 비해 AI 분석 작업 관리에서 상당히 뛰어난 성능을 보였습니다. 제안된 학습된 시스템 구성 요소는 최신 접근법보다 환경 변동성을 더 효과적으로 처리합니다. 또한, 학습된 동시성 제어 알고리즘과 학습된 쿼리 옵티마이저가 도입되어 데이터 및 워크로드 변화에 빠르게 적응할 수 있습니다.



### Cross-cultural analysis of pedestrian group behaviour influence on crossing decisions in interactions with autonomous vehicles (https://arxiv.org/abs/2408.03003)
Comments:
          Paper accepted at the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC 2024)

- **What's New**: 이번 연구는 가상 현실(VR) 환경을 활용하여 자율 주행차량(AV)과 보행자 간의 상호작용을 탐구하고, 문화적 배경이 사람들의 도로 횡단 행동에 미치는 영향을 분석합니다. 실험은 스페인과 호주에서 동일한 조건으로 수행되었으며, 두 나라의 다양한 사회적 규범과 상황적 요인을 고려합니다.

- **Technical Details**: 연구는 CARLA 시뮬레이터 내에 현실적으로 구현된 VR 환경을 사용했습니다. 참여자들은 보행자들이 서로 다른 행동을 보이는 횡단보도에서 도로를 건너려 시도하며, AV는 여러 가지 브레이킹 매뉴버(Braking Maneuver)를 적용했습니다. 실험 결과 분석을 위해 질문지와 도로 진입 시점을 측정하는 방법을 사용했습니다. 두 나라에서 동일한 실험 조건을 만들기 위해 Meta Quest 2(스페인)와 HTC Vive(호주)를 활용했습니다.

- **Performance Highlights**: 결과에 따르면, 보행자들은 무리 속에서 교통 틈새를 더 자주 함께 건너는 경향이 있으며, 무리의 부주의한 행동은 신뢰도를 낮추고 상황을 더 복잡하게 생각하게 만들었습니다. 호주 참여자들은 스페인 참여자들보다 AV가 양보할지 확실하지 않은 상황에서 더 신중한 행동을 보였습니다. 이 연구는 문화적 배경이 도로 횡단 행동에 중요한 역할을 한다는 것을 시사합니다.



### LLMs as Probabilistic Minimally Adequate Teachers for DFA Learning (https://arxiv.org/abs/2408.02999)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 이용한 자동자 학습(automata learning)의 새로운 접근법인 확률론적 최소 적절 교사(probabilistic Minimally Adequate Teacher, pMAT) 개념을 소개합니다. pMAT는 결정적 유한 자동자(Deterministic Finite Automata, DFA) 학습에서 멤버십 쿼리에 대해 무작위로 지속적인 오류를 발생시키는 확률론적 오라클을 활용합니다. 이를 통해 LLM의 허구적 콘텐츠 생성을 방지하고 학습된 자동자의 정확성을 보장하는 기법을 개발하였습니다.

- **Technical Details**: 새로 제안된 pMAT 포뮬레이션은 세 가지 주요 이유로 도입되었습니다. 첫째, LLM은 멤버십 쿼리에는 능숙하지만 동등성 쿼리에는 어려움을 겪습니다. 둘째, LLM은 항상 신뢰할 수 없으며 지속적인 오류를 발생시킬 수 있습니다. 셋째, 실질적인 구현에서는 소프트웨어나 환경 시뮬레이션과 같은 대상 모델에 대해 가설을 검증하는 것이 일반적이므로 유효한 반례(counterexamples)를 쉽게 얻을 수 있습니다. 성능을 높이기 위해 체인 오브 사라우드(Chain of Thought) 방식에서 영감을 받아 두 가지 새로운 프롬프트 기법인 Discrimination 프롬프트와 Verification 프롬프트를 개발했습니다. 또한, TTT 알고리즘과 일반적인 활성 학습 알고리즘 사이의 DFA 학습 성능을 비교했습니다.

- **Performance Highlights**: 시험 결과, 제안된 방법이 높은 견고성과 효율성을 보임을 확인했습니다. 특히, 동적 쿼리 캐시 정제 알고리즘을 통해 활성 학습과 수동 학습 알고리즘을 결합하여 쿼리와 반례 간의 충돌을 식별하고 수정할 수 있었습니다. 이는 LLM을 활용한 자동자 학습의 이론적 기반을 제공하며, 학습된 자동자가 목표 모델과 동등성을 유지하도록 보장합니다.



### ASR-enhanced Multimodal Representation Learning for Cross-Domain Product Retrieva (https://arxiv.org/abs/2408.02978)
Comments:
          10 pages, 5 figures

- **What's New**: E-commerce 환경이 점점 더 멀티미디어적으로 풍부해지면서, 제품들은 이미지, 짧은 비디오, 또는 라이브 스트림과 같은 다양한 형식으로 전시되고 있습니다. 이러한 다양한 형식을 효과적으로 통합하기 위한 통일된 벡터화된 크로스 도메인 제품 표현이 필요합니다. 본 논문은 ASR 텍스트를 활용하여 멀티모달 제품 표현 학습 (AMPere)을 제안하며, 기존의 시각적 표현만으로는 부족했던 부분을 보완합니다.

- **Technical Details**: AMPere는 LLM 기반의 ASR 텍스트 요약기를 사용하여 원시 ASR 텍스트에서 제품 특정 정보를 추출합니다. 요약된 ASR 텍스트와 시각적 데이터를 결합하여 멀티모달 임베딩(embeddings)을 생성할 수 있도록 Multi-branch Network를 활용합니다. 특히 ROPE와 같은 대규모 트라이 도메인 데이터셋에서 실험을 통해 제안된 방법의 유효성을 입증했습니다.

- **Performance Highlights**: 광범위한 실험 결과는 AMPere가 크로스 도메인 제품 검색(CdPR)에서 통일된 멀티모달 제품 표현을 획득하는 데 있어 매우 효과적임을 보여줍니다. LLM 기반 텍스트 요약기를 사용하여 ASR 텍스트의 노이즈를 효과적으로 제거함으로써, 멀티모달 표현 학습에 있어서 큰 성과를 얻었습니다.



### Empathy Level Alignment via Reinforcement Learning for Empathetic Response Generation (https://arxiv.org/abs/2408.02976)
- **What's New**: 이번 연구는 공감적 응답 생성을 위해 강화 학습을 사용하는 새로운 프레임워크인 EmpRL(Empathetic Response Generation using Reinforcement Learning)을 제안한 것입니다.

- **Technical Details**: EmpRL은 사전 학습된 T5 모델을 생성기로 사용하며, 공감 보상 함수(empathy reward function)를 설계해 강화 학습을 통해 예상 보상을 최대화하여 공감적 응답을 생성합니다. 보상 함수는 세 가지 공감 커뮤니케이션 메커니즘(empathy communication mechanisms)—정서적 반응(emotional reaction), 해석(interpretation), 탐색(exploration)—을 포함합니다. Proximal Policy Optimization (PPO) 알고리즘을 사용해 정책을 추가로 훈련하여 응답의 일관성을 유지합니다.

- **Performance Highlights**: EmpRL 프레임워크는 자동 및 수동 평가 모두에서 생성된 응답의 공감 수준 유사성을 향상시키고, 감정적 및 인지적 공감을 모두 포함하는 공감적 응답을 생성하는 데 우수한 성능을 보였습니다.



### Few-shot Scooping Under Domain Shift via Simulated Maximal Deployment Gaps (https://arxiv.org/abs/2408.02949)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2303.02893

- **What's New**: 이 논문에서는 적은 시도로 새로운 목표 지형에서 고급 샘플을 채취하는 문제(few-shot scooping problem)와 이를 해결하기 위한 비전 기반 적응 스쿠핑 전략을 제안합니다. 구체적으로 Deep Kernel Gaussian Process 방법을 사용하여 독특한 메타-트레이닝 전략으로 훈련된 모델을 이용해 지형 변화에 적응합니다. 이 접근법은 특히 Deep Kernel Calibration with Maximal Deployment Gaps (kCMD) 전략을 사용하여 큰 도메인 이동(domain shifts)에 적응할 수 있도록 훈련됩니다.

- **Technical Details**: 제안된 kCMD 전략은 훈련 데이터셋을 통해 생성된 최대 배포 간극(maximal deployment gaps)을 모델이 극복하도록 다루어줍니다. 이 과정은 훈련 세트를 mean-training과 kernel-training으로 나누어 최적 운송(optimal transport, OT)을 최대화하면서 도메인 간극을 극대화합니다. 모델은 결과적으로 잔차(residuals)를 학습하여 새로운 도메인 작업에 빨리 적응합니다. Bayesian Optimization 체계를 적용하여, 이 방법은 스쿱 볼륨 예측과 관련된 불확실성을 균형 잡아 액션을 선택합니다.

- **Performance Highlights**: 제안된 kCMD 방법은 UIUC 테스트베드와 NASA JPL Ocean Worlds Lander Autonomy Testbed(OWLAT)에서 실험을 통해 테스트되었습니다. 이 방법은 기존의 메타학습(meta-learning) 및 비적응 스쿱 방법보다 뛰어난 성능을 보였습니다. 특히 OWLAT에서 새로운 지형에 '제로-샷 전이(Zero-shot transfer)'를 성공적으로 수행하며, 적응력이 뛰어난 모델로 프로젝트가 성공할 수 있는 가능성을 보여주었습니다.



### Scaling Laws for Data Poisoning in LLMs (https://arxiv.org/abs/2408.02946)
Comments:
          20 pages

- **What's New**: 최근 연구에 따르면 LLMs (Large Language Models)는 부분적으로 손상되거나 유해한 데이터를 훈련 데이터로 사용할 경우 데이터 중독(data poisoning)에 취약하다는 사실이 밝혀졌습니다. 이러한 데이터 중독은 감지하기 어려우며, 가드레일을 깨뜨리고 바람직하지 않거나 유해한 행위를 초래할 수 있습니다. 대형 연구소들이 점점 더 크고 능력 있는 LLMs를 개발하고 배포하려는 노력에 따라, 데이터 중독의 위험이 규모에 따라 자연스럽게 완화될지 아니면 증가하는 위협이 될지 묻는 것이 중요합니다. 이 논문은 악의적인 미세조정(Malicious fine-tuning), 불완전한 데이터 큐레이션(Imperfect data curation), 의도적인 데이터 오염(Intentional data contamination) 같은 세 가지 위협 모델을 고려하여 데이터를 중독시키는 방법을 평가했습니다.

- **Technical Details**: 세 가지 위협 모델을 통해 LLMs가 훈련될 때 발생할 수 있는 데이터 중독을 다루고 있습니다. 첫째, 악의적인 미세조정은 정렬 조치를 제거하고 유해한 데이터를 포함하여 가드레일을 우회하는 방법입니다. 둘째, 불완전한 데이터 큐레이션은 데이터 큐레이션 과정에서 의도치 않은 유해한 데이터를 포함하는 경우를 말합니다. 셋째, 의도적인 데이터 오염은 악의적인 행위자가 웹 데이터셋을 독성화하는 방법입니다. 이러한 실험은 Gemma, Llama, Qwen 등의 모델 시리즈를 포함한 23가지 LLMs를 대상으로 수행되었습니다.

- **Performance Highlights**: 주요 발견 사항은 더 큰 LLMs가 데이터 중독에 더 취약하며, 작은 모델보다 유해한 행동을 더 빨리 배우는 경향이 있다는 것입니다. 특히, 더 큰 모델이 '잠복 에이전트(sleeper agent)' 행동을 더 쉽게 배우는 것은 우려스럽습니다. 연구 결과에 따르면 데이터 중독 비율이 낮더라도 더 큰 LLMs가 데이터 중독에 상대적으로 더 취약할 수 있음을 시사합니다. 이 결과는 향후 LLMs의 안전성을 보장하는 데 필요한 강력한 방어 장치의 필요성을 강조합니다.



### LLM-Empowered Resource Allocation in Wireless Communications Systems (https://arxiv.org/abs/2408.02944)
Comments:
          submitted to possible IEEE journal

- **What's New**: 이 논문은 무선 통신 시스템에서의 자원 할당 문제를 다루며, 대규모 언어 모델(Large Language Models, LLMs) 기반의 자원 할당 기법을 제안합니다. 특히, 두 송신 쌍 간의 단순 자원 할당 문제를 통해 에너지 효율성(Energy Efficiency) 또는 스펙트럼 효율성(Spectral Efficiency)을 최대화하는 방법을 탐구합니다.

- **Technical Details**: 논문에서는 LLM의 기본 원리와 그것이 자원 할당 문제에 어떻게 적용될 수 있는지 설명합니다. LLM은 다층 뉴럴 네트워크(Deep Neural Networks, DNN)를 기반으로 하며, 변환기(Transformer) 아키텍처와 같은 혁신적인 기법을 사용합니다. LLM은 엄청난 양의 텍스트 데이터를 통해 사전 학습되며, 이를 통해 다양한 과제를 해결할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 시뮬레이션을 통해 LLM 기반 자원 할당의 적용 가능성과 성능을 검증하였으며, 기존의 자원 할당 기법과 결합하여 신뢰도를 보완하는 하이브리드 자원 할당 전략을 제안했습니다. 또한, LLM 기반 접근법은 태스크 전용 모델 설계 및 학습이 필요하지 않다는 점에서 기존 딥러닝 기반 접근법보다 유연성이 뛰어납니다.



### Doubly Stochastic Adaptive Neighbors Clustering via the Marcus Mapping (https://arxiv.org/abs/2408.02932)
- **What's New**: 새로운 연구는 Marcus 정리에 기반하여 희소(doubly stochastic) 행렬을 학습하는 새로운 방법을 제안합니다. 또한, 최적 운송(optimal transport) 문제와의 관계를 탐구하고, 새로운 군집화 알고리즘인 ANCMM (Adaptive Neighbors Clustering based on the Marcus Mapping) 을 도입했습니다. 이를 통해 학습된 그래프가 자연스럽게 원하는 수의 클러스터로 나뉜다는 장점을 지닙니다.

- **Technical Details**: Marcus 정리는 양수 대칭 행렬을 대각 행렬을 사용해 이중 확률 대칭 행렬로 변환할 수 있음을 나타냅니다. 본 연구에서는 이론을 확장하여 일부 희소 행렬도 이중 확률 대칭 행렬로 변환될 수 있음을 증명하는 Marcus 매핑(Marcus mapping)을 제안했습니다. 또한, 최적 운송 문제와의 관계를 탐구하여, Marcus 매핑이 특정 유형의 최적 운송 문제를 더 효율적으로 해결할 수 있음을 보였습니다. NCMM은 이 Marcus 매핑을 활용하여, 대각 행렬을 통해 학습된 유사성 행렬이 바로 이중 확률 행렬의 형태가 되어 군집화 문제를 해결할 수 있도록 합니다.

- **Performance Highlights**: 제안된 ANCMM 알고리즘은 최첨단 알고리즘과의 비교 실험을 통해 그 효과성이 입증되었습니다. 특히, Marcus 매핑을 통해 최적 운송 문제를 직접 적용하는 것보다 더 효율적인 방법임을 보였습니다. 다양한 합성 및 실제 데이터셋에 대한 광범위한 비교를 통해 그 성능이 확인되었습니다.



### The Need for a Big World Simulator: A Scientific Challenge for Continual Learning (https://arxiv.org/abs/2408.02930)
Comments:
          Accepted to the Finding the Frame Workshop at RLC 2024

- **What's New**: 이 논문은 '작은 에이전트, 큰 세상'이라는 개념을 바탕으로 한 지속적 학습(continual learning)의 필요성을 강조하고 있습니다. 기존의 벤치마크는 비현실적인 데이터 분포 변화와 '작은 에이전트, 큰 세상' 프레임을 충실히 반영하지 못하는 한계를 지니고 있습니다. 이를 극복하기 위해 비슷한 개념을 보다 정확하게 반영하는 새로운 유형의 시뮬레이터 개발을 제안합니다.

- **Technical Details**: 논문은 '작은 에이전트, 큰 세상' 프레임워크 하에서 적절한 데이터를 섭취하고 유지하며 버릴 수 있는 에이전트를 설계해야 한다고 주장합니다. 이를 위해 정보 이론적 관점에서 환경과 에이전트를 정의합니다. 에이전트는 제한된 용량을 가지며, 이는 현실 세계의 비정상적이고 복잡한 데이터 변화에 대응해야 합니다. 논문에서는 이러한 변화가 자연스럽고 점진적으로 발생해야 한다고 지적합니다.

- **Performance Highlights**: 현재 벤치마크는 MNIST, CIFAR, ImageNet 같은 기존 데이터셋의 비정상적이고 인위적인 변화(예: 픽셀 치환)를 사용하며, 이는 실제 상황에서 발생하는 점진적이고 미세한 변화를 포착하지 못합니다. 따라서 새로운 벤치마크는 실제 환경의 복잡성을 반영하고, 에이전트의 성능 향상을 지속적으로 평가할 수 있는 환경을 제공해야 합니다.



### HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection (https://arxiv.org/abs/2408.02927)
- **What's New**: 최근 연구는 개인정보 보호 문제를 해결하면서도 현실적인 가상 표 형식 데이터를 생성할 수 있는 새로운 프레임워크인 HARMONIC을 도입했습니다. 이 프레임워크는 대규모 언어 모델(LLM)을 활용하여 표 데이터를 생성하고, 이 과정에서 개인정보 유출 위험을 최소화합니다.

- **Technical Details**: HARMONIC 프레임워크는 기존의 LLM 기반 방법들과 달리 소규모 모델이 아닌 대규모 모델을 사용하여 표 데이터를 생성합니다. 이 과정에서 k-최근접 이웃 알고리즘(k-nearest neighbors)을 참고하여 유사한 행들 간의 관계를 학습시키고, 데이터를 기억하는 대신 데이터의 형식과 연결성을 기억하도록 미세 조정(fine-tuning)합니다. 또한, DLT(Data Leakage Test)와 LLE(LLM Efficiency)라는 새로운 평가 지표를 도입하여 생성된 데이터의 개인정보 유출 위험과 모델의 성능을 평가합니다.

- **Performance Highlights**: HARMONIC 프레임워크를 사용한 실험 결과, 기존 방법들과 비교했을 때 개인정보 보호가 강화되었으며, 머신 러닝과 다운스트림 작업에서 동등하거나 더 나은 성능을 보였습니다. 특히, 기존의 가상 데이터 생성 방법이 LLM 기반 다운스트림 작업에 적합하지 않음을 보여주며, 사전 학습 기반의 가상 데이터가 심각한 개인정보 유출 위험을 내포하고 있음을 시사합니다.



### A Taxonomy of Architecture Options for Foundation Model-based Agents: Analysis and Decision Mod (https://arxiv.org/abs/2408.02920)
Comments:
          Under review

- **What's New**: AI 기술의 빠른 발전으로 인해 다양한 분야에서 에이전트 시스템의 응용이 확산되고 있습니다. 이러한 시스템의 설계와 운영에는 상세한 아키텍처 설계가 필요하며, 이는 큰 도전 과제를 안고 있습니다. 이 논문에서는 기초 모델 기반 에이전트의 아키텍처를 중점으로 하는 분류 체계를 소개하여 기능적 역량과 비기능적 품질을 다루고 있습니다. 설계 단계와 실행 단계에서의 운영을 논의하고 전체적인 아키텍처 설계와 운영 특성을 제공합니다. 이를 통해 기초 모델 기반 에이전트의 설계를 개선할 것을 목표로 하고 있습니다.

- **Technical Details**: 이 논문은 기초 모델 기반 에이전트 시스템의 아키텍처를 체계적으로 분류하는 것이 주 내용입니다. 논문은 먼저 기능적 역량과 비기능적 품질을 기준으로 에이전트를 분류하였고, 설계 시점과 실행 시점의 운영 단계에서도 분류를 제공합니다. 논문의 주요 기여 중 하나는 설계와 런타임 결정에 도움을 줄 수 있는 의사결정 모델을 제시한다는 점입니다. 이러한 의사결정 모델은 기초 모델 기반 에이전트의 개발 과정을 체계적으로 지도하는 데 도움을 줍니다.

- **Performance Highlights**: 논문은 기초 모델 기반 에이전트의 설계 옵션을 체계화하는 것으로, 설계 과정에서 서로 다른 아키텍처 옵션들을 비교하고 평가할 수 있도록 돕습니다. 예를 들어, 입력 모달리티 지원, 기본 모델에 접근, 외부 기능 통합 등 다양한 분류를 통해 에이전트의 조정 및 통신 효율성을 보장합니다. 이를 통해 복잡한 시스템의 설계 과정을 개선하고 견고한 비교 평가를 가능하게 합니다.



### KOI: Accelerating Online Imitation Learning via Hybrid Key-state Guidanc (https://arxiv.org/abs/2408.02912)
Comments:
          Submitted to Corl 2024

- **What's New**: KOI (Key-state guided Online Imitation) 학습 접근법을 도입하여, 시맨틱 및 모션 키 상태를 활용해 효율적인 온라인 탐험을 위한 과제 인지 보상 추정을 개선했습니다. 이를 통해 기존의 온라인 모사 학습 방법에서 발생하는 탐험 공간과 전문가 궤적 간의 큰 격차를 줄이고, 탐험 효율성을 높였습니다.

- **Technical Details**: KOI는 시맨틱 키 상태와 모션 키 상태를 통합하여 Trajectory-matching reward estimation (궤적-일치 보상 추정)을 정교화합니다. Visual-language 모델을 사용해 전문가의 궤적을 '무엇을 할 것인가'(semantic key states)로 분할하고, Optical flow를 활용해 '어떻게 할 것인가'(motion key states)를 이해합니다. 이를 통해 두 유형의 키 상태를 통합하여 OT(Optimal Transport) 기반 보상 추정을 미세 조정합니다.

- **Performance Highlights**: KOI 방법은 Meta-World와 LIBERO 환경에서 더 높은 샘플 효율성을 보여줍니다. 또한, 실제 로봇 조작 실험에서도 KOI 방법이 실용적 적용 가능성을 입증했습니다.



### Enabling Intelligent Traffic Systems: A Deep Learning Method for Accurate Arabic License Plate Recognition (https://arxiv.org/abs/2408.02904)
- **What's New**: 이 논문은 정확한 이집트 차량 번호판 인식을 위한 새로운 2단계 프레임워크를 도입합니다(Egyptian Vehicle License Plate Recognition, EVLPR). 첫 번째 단계에서는 이미지 처리 기술을 사용하여 번호판을 신뢰성 있게 위치시키고, 두 번째 단계에서는 견고한 아랍 문자 인식을 위해 맞춤형 딥러닝 모델을 활용합니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 단계를 포함합니다. 첫 번째 단계는 이미지 처리 기술(image processing techniques)을 사용하여 번호판을 정확히 일정한 위치로 로컬라이징(localizing)하고, 두 번째 단계는 맞춤형 딥러닝(designed deep learning) 모델을 적용하여 견고한 아랍 문자 인식(character recognition)을 수행합니다. 이러한 접근 방식을 통해 다양한 데이터셋(daiverse dataset)에서 뛰어난 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 다양한 데이터셋에서 99.3%의 뛰어난 정확도를 달성하여 기존 방법을 능가합니다. 이 시스템은 트래픽 위반 탐지(traffic violation detection) 및 주차 최적화(parking optimization)와 같은 지능형 트래픽 관리(intelligent traffic management)에 잠재적 응용 가능성을 가지고 있습니다.

- **Future Directions**: 향후 연구에서는 시스템의 아키텍처 개선(architectural refinements), 데이터셋 확장(expanded datasets), 시스템 종속성(system dependencies) 문제를 해결하여 시스템의 기능을 향상시키는 데 중점을 둘 것입니다.



### A Metric Driven Approach to Mixed Precision Training (https://arxiv.org/abs/2408.02897)
- **What's New**: 이번 연구에서는 딥 러닝의 모델 학습 효율성을 높이기 위해 저정밀도 수치 형식을 선택할 수 있는 지표 중심 방법론을 제안했습니다. 이 방법론이 BERT 모델의 훈련을 확장하는 데 어떻게 도움이 되는지 실험을 통해 입증했습니다. 이 기술은 다른 모델 아키텍처에도 일반화될 수 있습니다.

- **Technical Details**: 딥 러닝 모델의 효율성을 높이기 위한 방법으로 양자화(Quantization)를 사용했습니다. 양자화는 수를 표현하는 데 사용하는 비트 수를 줄여서 메모리 사용량과 계산 능력을 줄이는 기법입니다. 다양한 8비트 데이터 형식(INT8, E4M3, E5M2)을 연구하고, 이들 형식을 통해 모델 훈련의 품질을 예상할 수 있는 방법론을 제안했습니다. 특히 BERT 모델에서 이 방법론을 평가하고, 여러 양자화 파라미터와 분포에 따른 오차를 분석했습니다.

- **Performance Highlights**: 실험 결과, INT8 형식이 높은 정밀도로 좁은 범위의 값을 포착하지만, FP8 형식(E4M3, E5M2)은 더 넓은 동적 범위와 트레이드오프를 통해 높은 정밀도를 제공합니다. 오차 분석에서는 FP8 형식이 입력 분포의 꼬리 부분에 민감하게 반응하는 INT8 형식보다 더 안정적인 결과를 나타냈습니다. 특정 분포 및 양자화 방법론에 따라 E4M3가 E5M2보다 작은 오차를 보일 수 있음을 발견했습니다.



### VizECGNet: Visual ECG Image Network for Cardiovascular Diseases Classification with Multi-Modal Training and Knowledge Distillation (https://arxiv.org/abs/2408.02888)
Comments:
          Accepted in International Conference on Image Processing (ICIP) 2024

- **What's New**: VizECGNet은 인쇄된 심전도(ECG) 이미지만을 사용해 다양한 심혈관 질환의 예후를 예측하는 새롭고 혁신적인 다중 모달 딥러닝 모델로서, 병원에서 디지털화된 신호 대신 이미지로 데이터를 저장하는 경향을 해결하고자 개발되었습니다.

- **Technical Details**: VizECGNet은 이미지와 신호 두 가지 모달리티의 정보를 통합하기 위해 크로스 모달 주의 모듈(Cross-Modal Attention Modules, CMAM)을 사용하고, 각 모달리티의 데이터를 처리하기 위해 자체 모달리티 주의 모듈(Self-Modality Attention Modules, SMAM)을 사용합니다. 또한, 지식 증류(Knowledge Distillation) 기법을 사용하여 각 모달리티 스트림의 예측 사이의 유사성을 높입니다. 이를 통해 inference 단계에서는 ECG 이미지만으로 높은 성능을 달성할 수 있습니다.

- **Performance Highlights**: VizECGNet은 기존의 신호 기반 ECG 분류 모델에 비해 정밀도(Precision) 3.50%, 재현율(Recall) 8.21%, F1-Score 7.38% 향상을 이루었습니다. 이 모델은 다중 모달 학습과 지식 증류 기법을 이용하여 대규모 12-lead ECG 데이터셋에서 우수한 성능을 보였습니다.



### Hide and Seek: Fingerprinting Large Language Models with Evolutionary Learning (https://arxiv.org/abs/2408.02871)
- **What's New**: 이 논문은 LLM(대형 언어 모델)에서 생성된 콘텐츠를 식별하고 특정 모델을 판별하는 새로운 블랙박스 접근 방식을 소개합니다. 이 접근 방식은 모델 계열(Llama, Mistral, Gemma 등)을 식별하는 데 있어 72%의 높은 정확도를 달성했습니다.

- **Technical Details**: 이 방법은 'Hide and Seek' 알고리즘을 사용하여, Auditor LLM이 차별적인 프롬프트(discriminative prompts)를 생성하고 Detective LLM이 응답을 분석하여 목표 모델을 식별합니다. 진화적 전략(evolutionary strategy)을 활용하여 하나의 LLM을 통해 다른 LLM을 식별하는 데 가장 두드러지는 특징을 강조합니다. 문맥 학습(in-context learning)을 통해 프롬프트를 반복적으로 정제함으로써 모델 출력 간의 미묘한 차이를 발견합니다.

- **Performance Highlights**: 이 접근 방식은 LLM 기반 모델 식별의 실현 가능성을 입증하며, 다양한 LLM 계열의 의미적 매니폴드(semantic manifolds)에 대한 통찰을 제공합니다. 모델 속성, 보안, AI 투명성 분야에서 중요한 영향을 미치는 강력한 도구가 될 수 있습니다.



### VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledg (https://arxiv.org/abs/2408.02865)
- **What's New**: VisionUnite는 안과 분야를 위한 새로운 비전-언어 융합 대형 모델입니다. 이 모델은 1.24백만 개 이미지-텍스트 쌍의 대규모 데이터셋으로 사전 훈련되고, 추가로 MMFundus 데이터셋(296,379개의 고품질 눈저부 이미지-텍스트 쌍 및 889,137개의 모의 의사-환자 대화 인스턴스)으로 세부 조정되었습니다.

- **Technical Details**: VisionUnite는 Transformer 기반의 비전 인코더(vision encoder), 비전 어댑터(vision adapter) 및 LLaMA 모델로 고유의 목적을 달성하도록 세 가지 훈련 목표(visual text contrastive learning loss(CLIP Loss), visual classification loss(CLS Loss) 및 text generation loss(LLM Loss))로 학습됩니다. VisionUnite는 자연 이미지-텍스트 쌍과 생의학 이미지-텍스트 쌍을 포함한 1.19백만 개의 이미지-텍스트 쌍의 종합적인 데이터셋을 사용하여 사전 훈련됩니다.

- **Performance Highlights**: VisionUnite는 다양한 임상 시나리오에서 뛰어난 성능을 보입니다. 존재하는 GPT-4V 및 Gemini Pro와 같은 기존 생성 모델보다 우수하며, 진단 능력은 초급 안과 전문의와 비슷한 수준입니다. VisionUnite는 다중 질병 진단, 임상 설명 및 환자 상호 작용 등의 다양한 임상 시나리오에서 잘 작동합니다. 또한 초급 안과 의사의 훈련을 돕는 교육적인 도구로 사용할 수 있습니다.



### On The Stability of Moral Preferences: A Problem with Computational Elicitation Methods (https://arxiv.org/abs/2408.02862)
Comments:
          To appear in AIES 2024

- **What's New**: 참여형 윤리 AI 툴 개발의 중요한 부분으로 도덕적 선호도 조사 방법론의 효용성에 대한 연구가 제시되었습니다. 새로운 연구는 참가자의 도덕적 판단이 일관적인지 확인하기 위해 동일한 질문을 반복하는 방식을 사용하여 기존의 단회성 조사 방법에 대한 한계를 탐구했습니다.

- **Technical Details**: 연구에서는 2주 동안 동일한 참가자들에게 10번의 세션을 통해 동일한 도덕적 질문들을 제시하였습니다. 실험은 단순(Study One) 및 복잡(Study Two) 시나리오를 통해 이루어졌으며, 참가자들의 응답 안정성을 조사하였습니다. 주요 조사는 두 환자 중 누가 신장을 받을지에 대한 결정 상황이었으며, 각 세션마다 환자의 특성과 순서를 변경하여 제시하였습니다.

- **Performance Highlights**: 연구 결과, 논란이 많은 시나리오에 대한 응답 불안정성은 약 10-18% 수준으로 나타났습니다. 응답 불안정성은 응답 시간과 결정의 어려움과 긍정적인 상관관계를 보였습니다. 이러한 결과는 도덕적 선호도 조사 방법론의 효율성뿐만 아니라, 응답 불안정성이 이해 관계자와 AI 툴 간의 가치 불일치 문제를 일으킬 수 있음을 시사합니다.



### Multistain Pretraining for Slide Representation Learning in Pathology (https://arxiv.org/abs/2408.02859)
Comments:
          ECCV'24

- **What's New**: Madeleine이라는 새로운 멀티모달 사전학습 전략이 도입되었습니다. Madeleine은 다양한 염색 마커를 사용한 슬라이드를 사용하여 풍부한 학습 신호를 형성하고, H&E 슬라이드와 다른 염색 마커를 같은 공간에 정렬함으로써 단일한 슬라이드 표현을 학습합니다. 이는 컴퓨터 병리학에서 유용한 보편적이고 전이 가능한 표현을 생성하는 데 중점을 둡니다.

- **Technical Details**: Madeleine은 멀티헤드 어텐션 기반의 다중 인스턴스 학습(MIL) 및 듀얼 글로벌 로컬 크로스 스테인 정렬 목적을 사용합니다. H&E 슬라이드와 다른 염색들 간의 슬라이드 수준 및 패치 수준 정렬을 학습합니다. 글로벌 목표는 대칭적 대립 상 손실(symmetric contrastive loss)을 기반으로 하여 슬라이드 간의 전반적인 모폴로지 구성을 캡슐화하고, 지역 목표는 그래프 최적 운송 프레임워크(Graph Optimal Transport framework)를 통해 세밀한 형태학적 특징을 일치시킵니다.

- **Performance Highlights**: Madeleine이 유방암 샘플(N=4,211 WSIs, 5 stains)과 신장 이식 샘플(N=12,070 WSIs, 4 stains)에서 사전학습되었고, 21개의 다양한 다운스트림 작업에서 평가되었습니다. 이 작업은 형태학적 하위 유형(morphological subtyping), 분자 하위 유형(molecular subtyping), 생존 예측(survival prediction), IHC 정량화를 포함하며, 다양한 의료 센터에서 수집된 7,299개의 WSIs를 사용했습니다.



### Training a multilayer dynamical spintronic network with standard machine learning tools to perform time series classification (https://arxiv.org/abs/2408.02835)
Comments:
          7 pages, 4 figures

- **What's New**: 새로운 연구는 스핀트로닉(Spintronic) 진동기를 다이나믹한 뉴런으로 사용하는 하드웨어 구현의 재발견입니다. 이 연구에서는 다층 네트워크를 구축하여 기존 소프트웨어 기반의 네트워크와 동일한 89.83±2.91%의 정확도로 순차적 숫자 분류 작업을 수행할 수 있음을 시뮬레이션을 통해 입증하였습니다.

- **Technical Details**: 스핀트로닉 진동기는 비선형 고속 다이나믹스로 인해 뉴럴 네트워크의 하드웨어 구현에 유망한 요소로 간주됩니다. 이 연구에서는 스핀트로닉 진동기의 임시 다이나믹스를 활용하여 순차적 숫자 분류 작업을 수행하였습니다. PyTorch와 시간 역전파(Backpropagation Through Time, BPTT)를 사용하여 네트워크를 학습시키고, 네트워크 아키텍처와 다층 뉴런의 연결 방식을 상세히 설명하였습니다.

- **Performance Highlights**: 이 연구에서는 기존 소프트웨어 기반 네트워크와 동일하게 89.83±2.91%의 정확도로 순차적인 숫자 분류 작업을 수행할 수 있음을 입증하였습니다. 또한 스핀트로닉 네트워크는 입력의 시간 스케일에 걸쳐 5배 범위 내에서 학습할 수 있음을 보여 주었습니다.



### Examining Gender and Power on Wikipedia Through Face and Politeness (https://arxiv.org/abs/2408.02798)
- **What's New**: 이 연구는 사회언어학 이론의 두 가지 중요한 개념인 face acts와 politeness를 결합하여 담론을 분석하는 새로운 프레임워크를 제안합니다. 특히, 현재까지는 리소스가 부족했던 face acts를 다루기 위해 Wikipedia 대화 페이지(Wikipedia Talk Pages)를 주석하여 새로운 말뭉치를 생성하고 이를 토대로 face act 태거(tagger)를 훈련했습니다. 이를 통해 Wikipedia 편집자 간의 성별과 권력에 따른 face와 politeness의 상호작용을 분석했습니다.

- **Technical Details**: Brown과 Levinson의 politeness 이론은 'face' 개념을 기반으로 하며, 이는 긍정적 얼굴(positive face)과 부정적 얼굴(negative face)로 나눕니다. 이러한 이론을 바탕으로, 연구팀은 Wikipedia Talk Pages를 주석하여 face act 데이터를 생성하고 이를 토대로 face act 태거(tagger)를 훈련했습니다. 이 모델을 사용하여 약 130만 개의 Wikipedia 대화 페이지 문장을 분석했습니다.

- **Performance Highlights**: 연구 결과, 여성 Wikipedia 편집자들이 남성보다 더 예의를 지키는 경향이 있다는 이전 연구 결과를 확인할 수 있었습니다. 더욱 흥미로운 점은, 관리 권한이 있는 편집자들 사이에서는 이러한 성별 차이가 거의 사라진다는 것입니다.



### Diffusion Models as Data Mining Tools (https://arxiv.org/abs/2408.02752)
Comments:
          Project Page: this https URL Accepted in ECCV 2024

- **What's New**: 이 논문은 이미지 생성용 학습된 생성 모델(generative model)을 시각적 데이터 마이닝 도구로 활용할 수 있는 방법을 논의합니다. 생성 모델이 훈련 데이터의 정확한 표현을 학습한다는 점을 이용하여 데이터 요약 및 시각적 패턴을 마이닝할 수 있는 접근법을 제시합니다. 특히, 조건부 diffusion 모델을 특정 데이터셋에서 이미지를 생성하도록 세부 조정한 후, 이 모델을 사용하여 데이터셋 내에서 시각적 요소의 typicality(전형성)을 측정할 수 있다는 점을 강조합니다.

- **Technical Details**: 논문에서 제안하는 방법은 주로 다음과 같은 단계로 구성됩니다: 1) 조건부 diffusion 모델을 대상 데이터셋에 맞게 세부 조정합니다. 2) 조정된 모델을 사용하여 픽셀 단위로 특정 라벨(위치 정보, 시간 정보, 의미적 라벨 등)이 이미지 재구성에 미치는 영향을 평가합니다. 3) 전형성을 기반으로 비정형 시각 요소를 뽑아내고 이를 클러스터링하여 대표 패턴을 요약합니다. Diffusion 모델은 무작위 노이즈를 목표 분포로 변환하는 과정에서 반복적인 디노이징을 수행합니다.

- **Performance Highlights**: 제안된 방법론은 다양한 콘텐츠와 규모의 데이터셋에서 뛰어난 성능을 보이며, 기존의 방법론보다 효율적으로 규모를 확장할 수 있음을 보여줍니다. 예를 들어, 20세대의 아비에이터 안경 및 40세대의 군모와 같은 시대적 요소의 시각적 대표성을 강조할 수 있습니다. 또한 대규모 거리뷰 데이터에서 전봇대, 볼라드 등 지리적 특징을 요약하는 데에도 성공적이었습니다. 이는 GeoGuessr 게임(인기있는 지리 퀴즈 게임)에서도 볼 수 있는 주요 지리적 요소를 통해 입증되었습니다.



### MDM: Advancing Multi-Domain Distribution Matching for Automatic Modulation Recognition Dataset Synthesis (https://arxiv.org/abs/2408.02714)
- **What's New**: 최근 심층 학습 기술이 자동 변조 인식(AMR) 작업에 성공적으로 도입되었습니다. 그러나 심층 학습의 성공은 대규모 데이터셋에 대한 훈련에 의존합니다. 이러한 대규모 데이터는 저장, 전송 및 모델 훈련에 막대한 부담을 주고 있습니다. 이를 해결하기 위해 데이터 증류(Data Distillation) 기법이 제안되었으나, 신호의 고유한 특성으로 인해 이미지 처리와는 다른 접근이 필요합니다. 본 논문에서는 새로운 데이터셋 증류 방법인 다중 도메인 분포 매칭(Multi-domain Distribution Matching, MDM)을 제안합니다.

- **Technical Details**: MDM은 이산 푸리에 변환(Discrete Fourier Transform, DFT)을 사용하여 시간(domain) 신호를 주파수(domain)로 변환한 후, 모델을 사용하여 합성 데이터셋과 실제 데이터셋 간의 분포 매칭 손실을 계산합니다. 이는 시간과 주파수 도메인을 모두 고려하여 최적의 합성 데이터셋을 갱신합니다. MDM은 세 가지 AMR 데이터셋(RML2016.10a-high, RML2016.10a, Sig2019-12-high)에서 실험을 진행했으며, 시간 및 주파수 도메인에서 손실을 결합하여 최종 목표 함수를 최소화하는 방식으로 최적의 합성 데이터셋을 얻습니다.

- **Performance Highlights**: MDM은 Random Select, Forgetting, Dataset Condensation(DC), Distributed Matching(DM) 등 기존의 기준 방법과 비교하여 동일한 압축률에서 더 나은 성능을 입증했습니다. 또한, 여러 모델(Such as AlexNet, VGG16)에 대한 교차 아키텍처 일반화 실험에서는 MDM이 학습한 합성 데이터셋이 이전에 보지 못한 모델 아키텍처에서도 잘 일반화되는 것으로 나타났습니다.



### A Review on Organ Deformation Modeling Approaches for Reliable Surgical Navigation using Augmented Reality (https://arxiv.org/abs/2408.02713)
- **What's New**: 증강 현실(Augmented Reality, AR)은 수술 절차를 혁신할 잠재력을 가지고 있으며, 이는 수술 전에 얻은 장기 모델을 실제 해부학에 중첩시켜 실현됩니다. 수술 중 장기의 동적 변형으로 인해 사전 모델이 적절한 정합을 유지하기 힘들지만, 적절한 변형 모델링 기법을 통해 이를 극복할 수 있습니다. 본 리뷰는 AR 수술에서 장기 변형을 모델링하는 다양한 방법들을 체계적으로 요약하고 분류하고자 합니다.

- **Technical Details**: 장기 변형 모델링(organ deformation modeling)은 수술 중 관찰된 데이터를 기반으로 기존 3D 장기 모델을 실시간으로 업데이트하는 과정을 포함합니다. 이러한 관찰 데이터는 해부학적 랜드마크, 조직 구조 실루엣 또는 3D 디지타이즈드 장기 표면 등의 형태로 존재할 수 있습니다. 본 논문에서는 모델 기반(model-based), 데이터 기반(data-driven), 혼합형(hybrid) 방법 등으로 분류할 수 있는 다양한 알고리즘들을 다룹니다. 모델 기반 알고리즘은 종양 움직임 추적과 호흡 운동 패턴을 이해하는데 중점을 두며, 이는 주로 외부 신호와 연관된 내부 종양의 움직임을 예측하는데 사용됩니다. 데이터 기반 알고리즘은 머신러닝 기법을 사용하여 장기 변형 패턴을 학습하고 예측합니다.

- **Performance Highlights**: 리뷰에서 수집된 문헌들(총 112편)은 수술 중 장기 변형 모델링에 대한 현재의 기술 상태와 임상 적용 사례들에 대해 상세히 다룹니다. 특히, 여러 수술 분야에서의 변형 모델링 도전에 대한 대응 방법들을 제시하며, 향후 연구 방향에 대한 논의도 포함합니다. 기존 모델들의 도전에 대응하기 위해 새로운 데이터 기반 및 하이브리드 기술의 잠재력을 확인할 수 있습니다.



### Automatic Voice Identification after Speech Resynthesis using PPG (https://arxiv.org/abs/2408.02712)
- **What's New**: 이 논문에서는 PPG(Phonetic PosteriorGrams)를 기반으로 한 새로운 음성 재합성(speech resynthesis) 시스템을 소개합니다. 이 시스템은 음성과 관련된 다양한 작업에서 활용될 수 있으며, 특히 목소리 변환(voice conversion)과 음성 편집(speech edition) 분야에서 유용합니다.

- **Technical Details**: PPG는 음소의 프레임 단위 확률 표현으로, 보통 화자 독립적(speaker-independent)으로 간주됩니다. 새로운 시스템은 중간 표현에서 화자와 음소 콘텐츠를 분리(disentangle)하는 데 중점을 둡니다. 특히 PPG를 활용하여 입력된 음성을 합성하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 지각적 평가(perceptive evaluation)를 통해 시스템이 정확한 오디오 품질을 생성할 수 있음을 확인했습니다. 또한, 자동 화자 검증 모델(automatic speaker verification model)은 PPG로 합성된 후 원래의 화자를 인식하지 못한다는 것을 입증했습니다. 이는 모델이 합성 데이터를 기반으로 훈련된 경우에도 마찬가지입니다.



### Text Conditioned Symbolic Drumbeat Generation using Latent Diffusion Models (https://arxiv.org/abs/2408.02711)
- **What's New**: 이 연구는 텍스트 기반의 드럼 비트 생성을 Latent Diffusion Models (LDMs)을 통해 소개합니다. 훈련 데이터 파일명에서 추출한 정보를 텍스트로 사용하여 텍스트와 드럼 비트를 인코딩하는 모델을 Contrastive Learning을 통해 사전 학습하였습니다. 또한, MultiResolutionLSTM이라는 새로운 LSTM 변형을 제안하였으며, 이는 다양한 해상도에서 독립적으로 작동하도록 설계되었습니다.

- **Technical Details**: 이 연구는 텍스트와 드럼 비트를 인코딩하는 네트워크를 CLIP 방식에 따라 정렬합니다. 텍스트 인코더에는 멀티핫(Multihot) 텍스트 인코딩 기반의 대안을 연구했습니다. 모델은 사전 학습된 무조건적 오토 인코더(unconditional autoencoder)가 제공하는 잠재 공간(latent space)에서 확산(diffusion)을 실행하여 생성 속도를 높입니다. 전체적인 시스템 아키텍처는 텍스트 인코딩을 변환하고 LDM을 통해 드럼 비트를 생성하는 여러 단계를 포함합니다.

- **Performance Highlights**: 생성된 드럼 비트의 독창성과 다양성을 측정하기 위해 바이너리 피아노롤(pianoroll)과 잠재 공간에서의 거리 등을 평가했습니다. 청취 테스트를 통해 텍스트 프롬프트에 대한 적합성, 새로움, 품질을 측정한 결과, 생성된 드럼 비트는 인공지능이 아닌 인간 음악가가 만든 것과 비교할 만한 품질을 나타냈습니다. 테스트 결과 생성된 드럼 비트는 text prompt에 잘 맞고 새롭다는 평가를 받았습니다.



### Enhancing Medical Learning and Reasoning Systems: A Boxology-Based Comparative Analysis of Design Patterns (https://arxiv.org/abs/2408.02709)
- **What's New**: 대규모 하이브리드 AI 시스템의 설계 패턴과 임상 의사결정에서의 효과에 대한 새로운 연구가 Boxology 프레임워크를 사용하여 발표되었습니다. 이 연구는 머신 러닝과 규칙 기반 추론 시스템 (rule-based reasoning systems)이 결합된 다양한 아키텍처를 분석하고 비교하여 기존 설계 패턴을 기반으로 이들을 분류하고 새로운 통찰을 제공합니다.

- **Technical Details**: Boxology 프레임워크를 통해 하이브리드 AI 시스템의 구조적 기초를 이해하고 최적화하기 위한 연구입니다. REML, MLRB, RBML, RMLT, PERML의 다섯 가지 주요 아키텍처가 분석되었습니다. 각각의 아키텍처는 특정 임상 작업에 유리한 다양한 강점과 약점을 가지고 있습니다. 예를 들어, REML은 데이터셋이 부족한 경우 정확한 예측에 뛰어나고, MLRB는 대규모 데이터셋과 복잡한 데이터 통합에 효과적입니다. 또한 이 연구는 네 가지 새로운 패턴을 도입하고 다섯 가지 추상적 분류 패턴을 제시하여 Boxology의 분류 체계를 더욱 향상시켰습니다.

- **Performance Highlights**: 각 아키텍처의 성능 하이라이트는 다음과 같습니다: REML은 고정밀 예측, MLRB는 대규모 데이터 처리, RBML은 설명 가능성과 신뢰성, RMLT는 고차원 데이터 관리, PERML은 긴급 치료 시나리오에서 잠재력을 보였습니다. 이는 Boxology의 체계적이고 모듈식 접근 법이 하이브리드 AI 시스템의 개발과 분석에서 큰 장점을 제공하고 재사용 가능한 솔루션을 촉진할 수 있음을 보여줍니다.

- **Conclusion**: 이 연구는 하이브리드 AI 시스템이 의료 분야를 발전시키는 데 중요한 역할을 함을 강조하고 있으며, Boxology가 AI 통합에 따른 혁신을 더욱 촉진하여 임상 결정 지원 및 환자 결과를 향상시킬 수 있는 가능성을 제시합니다.



### SnapE -- Training Snapshot Ensembles of Link Prediction Models (https://arxiv.org/abs/2408.02707)
Comments:
          Accepted at International Semantic Web Conference (ISWC) 2024

- **What's New**: 새로운 연구는 평가 모델 학습 비용 증대 없이 다양한 예측 모델 앙상블을 훈련할 수 있는 스냅샷 앙상블(Snapshot Ensembles)을 지식 그래프(Knowledge Graph)의 링크 예측(Link Prediction) 모델에 적용하는 방법을 제안합니다. 또한, 명시적 부정 예시(negative examples)가 없는 지식 그래프의 특성을 반영하여 이전 스냅샷 모델을 사용해 부정 예시를 생성하는 새로운 훈련 루프도 제안합니다.

- **Technical Details**: 스냅샷 앙상블은 학습 과정에서 각각의 주기 끝에 모델의 스냅샷을 저장하고 이를 앙상블로 사용하여 다양성 있는 모델을 구성하는 방법입니다. 이 연구에서는 크로닉 코사인 기법(cyclic cosine annealing)을 사용해 학습률을 조정하고, 이전 스냅샷에서 부정 예시를 생성하는 방식을 도입했습니다. 이 방식은 단일 모델 학습 시간과 동일한 비용으로 앙상블 성능을 높일 수 있습니다.

- **Performance Highlights**: 네 가지 데이터셋과 네 가지 기반 모델을 대상으로 한 평가 결과, 제안된 방법인 SnapE는 단일 모델 대비 일관되게 우수한 성능을 발휘하는 것이 입증되었습니다. 이는 동일한 학습 자원을 사용하면서도 성능을 향상시킬 수 있음을 보여줍니다.



### Bayesian Kolmogorov Arnold Networks (Bayesian_KANs): A Probabilistic Approach to Enhance Accuracy and Interpretability (https://arxiv.org/abs/2408.02706)
- **What's New**: 이번 연구는 Kolmogorov Arnold Networks와 베이지안 추론(Bayesian inference)을 결합한 새로운 프레임워크인 Bayesian Kolmogorov Arnold Networks (BKANs)을 제안합니다. 이를 통해 예측의 설명 가능성과 불확실성을 고려한 예측을 제공합니다. 이러한 특성은 특히 임상적 의사결정에 있어 중요합니다.

- **Technical Details**: BKANs는 Kolmogorov Arnold Networks의 표현력을 활용하며, 베이지안 추론을 통해 모델의 불확실성을 양자화합니다. 이 시스템은 Pima Indians Diabetes dataset과 Cleveland Heart Disease dataset의 두 의료 데이터셋에 적용되어 평가되었습니다. BKANs는 aleatoric과 epistemic 불확실성을 모두 표현할 수 있는 능력을 가지며, 이는 더 신뢰할 수 있는 의사결정 지원을 제공합니다.

- **Performance Highlights**: BKANs는 예측 정확도에서 기존의 딥러닝 모델을 능가하며, 특히 작은 크기와 불균형을 가진 의료 데이터셋에서 모델의 설명력을 높이고 과적합(overfitting)을 최소화합니다. 또한, BKANs는 예측의 신뢰성과 결정 경계를 시각화하여 유용한 인사이트를 제공하며, 이는 미래 의료 AI 시스템 개발에 있어 중요한 발견입니다.



### PSNE: Efficient Spectral Sparsification Algorithms for Scaling Network Embedding (https://arxiv.org/abs/2408.02705)
- **What's New**: 새로운 논문은 대규모 그래프에서의 네트워크 임베딩(network embedding)을 위한 효율적인 스펙트럼 스파시피케이션(spectral sparsification) 방법인 PSNE를 소개합니다. 이 방법은 구조적 유사성을 잘 보존하는 임베딩 벡터를 빠르게 획득할 수 있도록 설계되었습니다.

- **Technical Details**: PSNE는 먼저 PPR 매트릭스(Personlized PageRank Matrix)를 계산하기 위한 희소한 행렬 다항식을 설계하여 계산을 가속화합니다. 이는 Frobenius Norm 측면에서 이론적 보장을 가집니다. 그 다음, 다중 관점 전략(multiple-perspective strategy)을 통해 얻어진 근사 PPR 매트릭스의 표현력을 더욱 향상시킵니다. 마지막으로 희소화된 다중 관점 PPR 매트릭스에 랜덤화된 특이값 분해(randomized singular value decomposition, SVD) 알고리즘을 적용하여 목표 임베딩 벡터를 얻습니다.

- **Performance Highlights**: 실제 및 합성 데이터셋에서의 실험 평가를 통해 PSNE가 기존의 10가지 경쟁 방법들과 비교하여 효율성, 효과성, 확장성 측면에서 더욱 뛰어남을 확인할 수 있었습니다.



### Spatial-temporal Graph Convolutional Networks with Diversified Transformation for Dynamic Graph Representation Learning (https://arxiv.org/abs/2408.02704)
Comments:
          8 papges, 1 figure

- **What's New**: 이 연구에서는 공간-시간 그래프 신경망을 위한 새로운 접근법인 STGCNDT(Spatial-Temporal Graph Convolutional Networks with Diversified Transformation)를 제안합니다. 이는 현재의 동적 GCN(Dynamic GCN) 모델들이 직면하고 있는 시공간 정보 분리 문제를 해결하고, 복잡한 시간 패턴을 효과적으로 포착할 수 있도록 설계되었습니다.

- **Technical Details**: STGCNDT는 세 가지 주요 요소로 구성됩니다: 1) 스페이터 텐서 컨볼루션 네트워크(GTCN)를 텐서 M-프로덕트(Tensor M-products)를 사용해 빌드하여 시공간 정보를 별도로 표현하지 않아도 됩니다; 2) GTCN에 세 가지 변환 스키마를 도입하여 시간 정보를 집계(Aggregate)하고 복잡한 시간 패턴을 모델링합니다; 3) 다양한 변환 스키마를 앙상블로 구성해 더 높은 표현 능력을 얻습니다.

- **Performance Highlights**: 통신 네트워크에서 나타나는 네 가지 동적 그래프(DGs)에 대한 실증 연구를 통해 제안된 STGCNDT는 링크 가중치 추정(Link Weight Estimation) 작업에서 최신 모델들을 크게 능가하는 성능을 보였습니다.



### Inventory problems and the parametric measure $m_{\lambda}$ (https://arxiv.org/abs/2408.02700)
- **What's New**: 새로운 파라미트릭 척도(parametric measure) $m_{ightarrow}$를 도입하여 리스크 중립 다항목 재고 문제(risk neutral multi-item inventory problem)를 분석하였습니다. 이는 기존의 신뢰성 측도(credibility measure)를 확장한 개념으로, 구매자의 수요가 퍼지 변수(fuzzy variables)로 표현됩니다.

- **Technical Details**: 수요 벡터(demand vector)의 구성 요소는 퍼지 변수(fuzzy variables)로 모델링되며, $m_{ightarrow}$-기대값(expected value)를 이용하여 최대화 문제를 공식화하였습니다. 최적화 문제의 일반 공식을 증명하며, 특히 수요가 사다리꼴(trapezoidal) 및 삼각형(triangular) 퍼지 수(fuzzy numbers)인 경우에 대한 최적 해(solution)를 계산하는 효과적인 공식을 제공합니다. $ightarrow=1/2$일 때, Li와 Liu의 문제에 대한 최적 해의 계산 공식을 얻을 수 있습니다.

- **Performance Highlights**: 제안된 모델은 신뢰성 이론에서 발전된 방식으로 제시되었으며, 제안된 $m_{ightarrow}$-모델을 통해 다양한 수치 데이터를 사용하여 최적화 해결책을 도출할 수 있습니다.



### DeepNetBeam: A Framework for the Analysis of Functionally Graded Porous Beams (https://arxiv.org/abs/2408.02698)
- **What's New**: 본 연구는 기능적 구배(FG) 다공성 보 (porous beams)를 분석하기 위해 과학적 기계 학습 (SciML) 접근법을 조사하고 이를 새로운 프레임워크 하에서 비교합니다. 보의 재료 특성이 연속 함수로 가정되는 가운데, 다양한 SciML 방법을 사용하여 보의 변위장을 근사하고 보의 거동을 지배하는 방정식을 도출합니다. 본 연구에서는 벡터 접근법을 통한 물리 정보 신경망(PINN), 에너지 접근법을 통한 딥 에너지 방법(DEM), 데이터 기반 접근법을 통한 뉴럴 오퍼레이터(Neural Operator) 방식을 제안합니다.

- **Technical Details**: 새로운 프레임워크에서는 다음의 세 가지 접근법을 사용합니다: (a) 벡터 접근법은 물리 정보 신경망(PINN)을 도출하고, (b) 에너지 접근법은 딥 에너지 방법(DEM)을 도출하며, (c) 데이터 기반 접근법은 뉴럴 오퍼레이터(Neural Operator) 방법의 결과를 도출합니다. 각 메서드는 보의 변위장을 근사하고 연속체 공식화에 따라 보의 거동 방정식을 유도합니다. 또한, 뉴럴 오퍼레이터를 훈련하여 다양한 다공 분포 패턴과 외부 차력 조건 하에서 FG 보의 응답을 예측합니다.

- **Performance Highlights**: 결과는 분석적 및 수치적 참조 솔루션과 비교하여 검증되었습니다. 제안된 방법들을 통해 FG 보의 다양한 재료 분포와 하중 조건 하에서 우수한 예측 정확성을 보여줍니다. 따라서, SciML의 잠재력을 활용하여 복잡한 구조적 문제를 더 정교하고 효율적으로 분석할 수 있음을 입증합니다.



### Why Rectified Power Unit Networks Fail and How to Improve It: An Effective Theory Perspectiv (https://arxiv.org/abs/2408.02697)
Comments:
          25 pages, 8 figures

- **What's New**: 이 논문은 Rectified Power Unit (RePU) 활성 함수들이 deep layers을 쌓을 때 발생하는 문제점을 해결하기 위해 새로운 활성 함수를 제안합니다. RePU는 differentiable한 특성을 가지고 있지만, 값이 폭발하거나 사라지는 현상과 같은 중요한 문제로 학습이 실패하는 경우가 발생합니다. 이를 극복하기 위한 Modified RePU (MRePU)를 새롭게 도입하여 RePU의 장점을 유지하면서도 단점을 보완하고자 합니다.

- **Technical Details**: RePU 네트워크는 깊은 레이어를 쌓을 때 값이 폭발하거나 사라지는 현상을 설명합니다. 이를 네트워크의 크리티칼리티 상태에서 이해하기 위해 초기화를 통해 주요 파라미터를 조정합니다. 딥 뉴럴 네트워크가 임의의 활성화 함수 선택에 민감하며, 특히 효과적인 이론의 관점에서 RePU 활성화 함수를 분석하여 왜 이 함수가 크리티칼리티 조건을 만족시킬 수 없는지 이론적으로 도출합니다.

- **Performance Highlights**: MRePU는 RePU와 비교했을 때, 실험적으로 개선된 성능을 보입니다. 네트워크 초기화 단계 및 훈련 과정 동안 경험적 커널을 계산하여 MRePU의 효과성을 검증합니다. 기존 RePU의 문제를 해결하고 더 나은 학습 안정성을 제공합니다.



### Distribution-Level Memory Recall for Continual Learning: Preserving Knowledge and Avoiding Confusion (https://arxiv.org/abs/2408.02695)
- **What's New**: 새로운 지속적 학습(Continual Learning, CL) 방법인 '분포 수준 메모리 리콜 방법(Distribution-Level Memory Recall, DMR)'이 제안되었습니다. 이 방법은 높은 차원의 특징 분포를 정확하게 적합시키기 위해 가우시안 혼합 모델(Gaussian Mixture Model, GMM)을 사용하여 이전 지식의 분포를 특징 공간에서 재현합니다. 이러한 방법은 고유한 메모리를 유지하고 새로운 지식을 학습하면서 기존 지식과의 혼란(confusion)을 최소화합니다.

- **Technical Details**: 이 연구는 기존의 프로토타입 기반 CL 방법이 특징 수준에서의 혼란을 일으킬 수 있다는 문제를 지적하며, 이를 해결하기 위해 GMM을 사용하여 이전 지식의 특징 분포를 정확하게 적합시킵니다. 또한, '상호 모달 가이드 및 내부 모달 채굴(Inter-modal Guidance and Intra-modal Mining, 이하 IGIM)' 이라는 방법을 사용해 여러 모달리티 모델의 불균형 문제를 해결합니다. 최종적으로, 새로운 샘플 특징을 사용해 가짜 특징을 향상시키는 '구성 인덱스(Confusion Index)'와 '인크리멘털 믹스업 특징 향상법(Incremental Mixup Feature Enhancement, 이하 IMFE)' 방법을 제안하여 새로운 지식과 기존 지식을 더 잘 구분할 수 있게 합니다.

- **Performance Highlights**: 이 새롭게 제안된 방법은 CIFAR100, ImageNet100, UESTC-MMEA-CL 데이터셋에서 SOTA(State-Of-The-Art)의 성능을 달성했습니다. 또한, DMR 방법이 실제 분류 경계를 유지하면서 거의 저장 부담을 증가시키지 않고, 모달리티 불균형 문제를 해결하여 일반화할 수 있는 특징을 얻는 데 기여함을 확인했습니다.



### KAN based Autoencoders for Factor Models (https://arxiv.org/abs/2408.02694)
Comments:
          7 pages

- **What's New**: 최근 Kolmogorov-Arnold Networks(KANs)의 진보에 영감을 받아, 우리는 잠재 요인 조건부 자산 가격 모델을 위한 새로운 접근 방식을 도입했습니다. 기존의 자산 가격에서 머신 러닝은 주로 ReLU(렐루) 활성 함수가 포함된 다층 퍼셉트론(MLP)을 사용했으나, 우리의 방법에서는 KAN 기반 오토인코더(autoencoder)를 도입해 MLP 모델을 정확성과 해석 가능성 두 측면에서 뛰어넘습니다.

- **Technical Details**: 우리의 모델은 자산 특징의 비선형 함수로서 노출을 근사하는 데 있어 유연성을 제공하며, 동시에 잠재 요인을 해석하는 직관적인 프레임워크를 사용자에게 제공합니다. KAN 기반 오토인코더는 원래의 데이터 구조를 압축하고 규명하는 데 있어서 MLP보다 뛰어난 성능을 보입니다. 이를 통해 잠재 요인의 노출을 훨씬 정교하게 모델링할 수 있습니다.

- **Performance Highlights**: 실증적인 백테스팅(empirical backtesting) 결과, 우리의 모델은 횡단면적 위험 노출(cross-sectional risk exposures)을 설명하는 데 있어서 뛰어난 성능을 발휘했습니다. 더불어, 우리 모델의 예측을 사용해 구성한 롱-숏 포트폴리오(long-short portfolio)는 더 높은 Sharpe 비율을 기록하였으며, 이는 투자 관리에서 실질적인 가치를 강조합니다.



### Diff-PIC: Revolutionizing Particle-In-Cell Simulation for Advancing Nuclear Fusion with Diffusion Models (https://arxiv.org/abs/2408.02693)
- **What's New**: 전통적인 입자-시뮬레이션(Particle-in-Cell, PIC) 시뮬레이션이 핵융합 연구에서 병목 현상을 초래하는 현실에 대응하여, 최근 발표된 'Diff-PIC'는 조건부 확산 모델을 활용하여 고해상도의 과학 데이터를 생성하는 컴퓨팅 효율적인 대안으로 소개됩니다. Diff-PIC는 PIC 시뮬레이션의 물리적 패턴을 확산 모델에 증류(distillation)함으로써 이론적 및 실용적인 타당성을 모두 입증합니다.

- **Technical Details**: Diff-PIC는 두 가지 주요 도전에 대응합니다: (1) 물리적 조건을 학습하고 생성할 수 있는 물리 정보를 반영한 조건부 확산 모델을 개발하여 연속적인 물리적 조건의 복잡한 관계를 효과적으로 캡처합니다. (2) 보정된 흐름(rectified flow) 방식을 통해 모델을 단일 단계 조건부 확산 모델로 변환하여, 효율성을 높이며 높은 충실성과 물리적 타당성을 유지합니다.

- **Performance Highlights**: 실험 결과에 따르면, Diff-PIC는 기존의 PIC 시뮬레이션보다 16,200배 더 빠른 속도를 보이며, FID, SWD, MMD 측정 지표에서 각각 0.341, 34.3, 8.98e-5의 높은 정밀도와 물리적 타당성을 유지합니다. 이러한 성과는 Fusion Ignition 연구에 중요한 진전을 의미하며, 지속 가능한 에너지 개발에 실질적인 기여를 할 것입니다.



### Attention is all you need for an improved CNN-based flash flood susceptibility modeling. The case of the ungauged Rheraya watershed, Morocco (https://arxiv.org/abs/2408.02692)
- **What's New**: 이번 연구에서는 플래시 홍수(flash flood) 취약성을 예측하기 위한 방법으로 주의 메커니즘(attention mechanism)을 사용하는 것을 탐구합니다. 특히, 컨볼루션 블록 주의 모듈(CBAM, convolutional block attention module)을 사용하여 CNN 모델의 성능을 향상시키고자 했습니다. 주로 ResNet18, DenseNet121, Xception 백본 아키텍처를 사용하였고, CBAM을 다양한 위치에 통합했습니다.

- **Technical Details**: 연구에서는 16개의 조건 변수와 522개의 플래시 홍수 인벤토리 포인트를 포함한 데이터셋을 사용했습니다. 모델의 성능은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1-score), ROC 곡선 아래 면적(AUC) 등으로 평가되었습니다. 주요 변수를 분석한 결과, 하천까지의 거리와 배수 밀도가 중요한 요소로 확인되었습니다. 연구는 CNN의 각 컨볼루션 블록에 CBAM을 통합한 DenseNet121이 가장 높은 성능을 보였음을 입증했습니다.

- **Performance Highlights**: DenseNet121 네트워크에 CBAM을 통합한 모델이 최고 성능을 보여주었습니다(정확도=0.95, AUC=0.98). 이 결과는 주의 메커니즘이 플래시 홍수 취약성 모델링에서 효과적임을 나타내며, 재난 관리에서 중요한 통찰력을 제공합니다.



### Symmetric Graph Contrastive Learning against Noisy Views for Recommendation (https://arxiv.org/abs/2408.02691)
Comments:
          24 pages, submitted to TOIS

- **What's New**: 이번 연구에서는 새로운 그래프 대비 학습 방법인 SGCL(Symmetric Graph Contrastive Learning)를 제안합니다. 기존의 데이터 증강 기법들이 원래 인터랙션 그래프의 연결을 방해하고, 부정확한 대비 뷰(contrastive view)를 생성해 최적의 성능을 발휘하지 못하는 문제를 해결하고자 합니다. SGCL은 대칭 이론을 도입하여 소음이 많은 보는 것에 저항하는 대칭 형태와 대조 손실을 구현합니다.

- **Technical Details**: SGCL은 모델-애그노스틱(model-agnostic) 방식으로 대칭 이론을 그래프 대비 학습에 통합합니다. 주로 다음을 포함합니다: 원래 그래프와 소음이 많은 보기를 구분하고, 후기의 20% 뷰의 코사인 유사도가 0.1 미만인 경우 노이즈로 간주합니다. SGCL은 이러한 노이즈가 성능을 저하시키는 것을 방지합니다. 또한, 이 방법은 이론적으로 소음이 많은 보기에도 높은 내성을 가진다는 것을 증명했습니다.

- **Performance Highlights**: 실험 결과, SGCL 방식은 세 가지 실제 데이터셋에서 기존 방법들보다 최대 12.25% 더 높은 추천 정확도를 기록했습니다. 이는 이 접근 방식의 효능을 높이 평가합니다.



### Spatio-Temporal Partial Sensing Forecast for Long-term Traffic (https://arxiv.org/abs/2408.02689)
- **What's New**: 이번 연구에서는 교통 예측을 위해 일부 위치에만 센서를 설치한 상황을 가정하고, 장기적인 교통 예측 모델을 제안했습니다. 새로운 Spatio-Temporal Partial Sensing (STPS) 모델을 통해 일부 위치에만 설치된 센서 데이터를 바탕으로 센서가 없는 위치의 교통 상황까지 장기적으로 예측합니다. 이는 교통 관리 인프라 비용을 절감하며, 특히 교통 신호 관리, 법 집행, 의료 지원 및 재난 대응 등에서 장점이 있습니다.

- **Technical Details**: STPS 모델에는 여러 혁신적인 기법이 포함되어 있어, 예측의 정확성을 크게 향상시켰습니다. 먼저, 'rank-based embedding(순위 기반 임베딩)' 기법으로 데이터 노이즈를 극복하고 복잡한 공간-시간 상관관계를 포착합니다. 다음으로, 'spatial transfer matrix(공간 전이 행렬)'를 사용해 센서가 있는 위치와 없는 위치 간의 분포 차이를 해결합니다. 마지막으로, 데이터의 모든 가용성을 최대한 활용해 점진적으로 모형 파라미터를 개선하는 'multi-step training process(다단계 학습 과정)'를 도입했습니다.

- **Performance Highlights**: 실제 교통 데이터셋을 활용한 광범위한 실험 결과, STPS 모델이 최첨단 기법보다 뛰어난 성능을 보이며, 일부 센싱만으로도 장기 예측의 정확도를 확보하는 데 성공했습니다.



### A Systematic Review of Intermediate Fusion in Multimodal Deep Learning for Biomedical Applications (https://arxiv.org/abs/2408.02686)
- **What's New**: 생물 의학 연구에서 딥러닝이 어떻게 복잡하고 고차원적인 데이터를 처리하는 혁신적인 접근 방식을 제공하는지에 대한 심층 리뷰가 새로이 발표되었습니다. 다중 모달 딥러닝(MDL, Multimodal Deep Learning)이 이미징, 텍스트 데이터, 유전 정보와 같은 다양한 데이터 유형을 통합해 예측 모델의 정확성과 강건성을 크게 향상시킬 수 있음에 주목하여 이에 관한 중간 융합(intermediate fusion) 기법들을 체계적으로 분석하고, 그 활용을 위한 구조화된 표기법을 도입하였습니다.

- **Technical Details**: 다중 모달 딥러닝의 융합 기법은 초기에 데이터 레벨에서 특징을 결합하는 'early fusion', 결정 레벨에서 결합하는 'late fusion', 그리고 특징 추출 단계에서 데이터를 통합하는 'intermediate fusion'이 있습니다. 중간 융합은 모달리티 특정 특징을 통합하기에 최적화된 접근 방식으로, 이는 더 깊은 단계에서 데이터 상호작용을 가능하게 합니다. 이 논문에서는 데이터 이질성 및 고차원 데이터 특성을 관리하기 위해 중간 융합을 사용하는 방법에 대해 논의합니다.

- **Performance Highlights**: 중간 융합 방법은 특히 생물 의학 데이터에서 더 정확하고 견고한 모델을 제작하는 데 큰 기여를 합니다. 유전 정보와 이미징 데이터 간의 복잡한 상호작용을 이해하고, 다양한 모달리티의 고유한 특성을 보존하면서 데이터의 고차원성을 줄여주는 특징이 있습니다. 이를 통해 예측 모델의 성능 향상뿐만 아니라, 기존 방법으로는 놓칠 수 있는 패턴과 상호작용을 발견할 수 있습니다.



### Artificial Neural Networks for Photonic Applications: From Algorithms to Implementation (https://arxiv.org/abs/2408.02685)
- **What's New**: 이 튜토리얼 리뷰는 인공지능 신경망이 광학(Photonics) 분야에서 어떻게 응용되는지를 다루며, 광학 연구 및 엔지니어링 공동체와 컴퓨터 과학, 응용 수학의 교차점에 있는 연구자들을 대상으로 합니다. 이 논문은 광학과 머신러닝의 상호작용에 중점을 두어 새로운 과학 및 공학 기술을 개발하고 향상시키는 방법을 탐구합니다.

- **Technical Details**: 초기에는 머신러닝 알고리즘이 광학 시스템에 어떻게 적용되는지를 소개하고, 다양한 신경망 구조의 설계와 알고리즘에서 하드웨어 구현으로의 전환 과정을 설명합니다. 특정 뉴럴 네트워크 구조의 복잡성을 줄이는 방법을 다루며, 모델 압축(model compression) 전략과 광학 응용에서의 새로운 기술을 결합하여 설명합니다.

- **Performance Highlights**: 논문의 중요한 부분은 신경망의 복잡성 감소가 실시간 운영에서 신경제(NN)의 에너지 효율성과 신호 처리 속도에 미치는 영향을 평가하는 것입니다. 이는 특히 광통신(optical communications)에서 다른 신호 처리 방법들과 비교함으로써 이루어집니다. 이 논문은 다양한 광학 응용 분야에서 신경망이 어떻게 성공적으로 사용되는지에 대한 최근의 발전 사항을 리뷰합니다.



### Patient-centered data science: an integrative framework for evaluating and predicting clinical outcomes in the digital health era (https://arxiv.org/abs/2408.02677)
- **What's New**: 본 연구는 디지털 헬스 시대에 환자 중심 데이터 과학을 위한 새로운 통합 프레임워크를 제안합니다. 전통적인 임상 데이터와 환자 보고 결과, 건강의 사회적 결정 요인, 다중 오믹 데이터(multi-omic data)를 결합하여 포괄적인 디지털 환자 표현을 생성하는 다차원 모델을 개발했습니다.

- **Technical Details**: 우리의 프레임워크는 다양한 머신 러닝 기술, 특히 대형 언어 모델(large language models)을 활용하여 복잡하고 종단적인 데이터 세트를 분석하는 다중 에이전트 인공지능 접근 방식을 채택하고 있습니다. 이 모델은 다수의 환자 결과를 동시에 최적화하면서 편향을 다루고 일반화를 보장하는 것을 목표로 합니다.

- **Performance Highlights**: 이 프레임워크는 최적의 환자 관리를 위한 전략을 지속적으로 개선하는 학습 의료 시스템(learning healthcare system)을 만드는 방법을 시연합니다. 이 접근 방식은 AI 기반 헬스케어 모델의 현재 한계를 해결하면서, 디지털 헬스 혁신을 실제 임상 혜택으로 번역하는 데 있어 상당한 개선을 가져올 잠재력을 가지고 있습니다.



### On Biases in a UK Biobank-based Retinal Image Classification Mod (https://arxiv.org/abs/2408.02676)
Comments:
          To appear at MICCAI FAIMI Workshop 2024

- **What's New**: 최근 연구는 의료 분야에서 머신러닝 모델의 성능 차이 문제를 밝혀냈습니다. 이번 연구에서는 UK Biobank의 망막 이미지를 사용해 질병 분류 모델을 훈련, 평가하며 다양한 인구 집단 간의 성능 차이를 조사했습니다. 특히, 특정 평가 센터에서의 불공정한 성능이 확인되었고, 이는 엄격한 데이터 표준화 프로토콜에도 불구하고 발생한 현상입니다.

- **Technical Details**: 연구는 UK Biobank에서 80,966개의 망막 이미지를 사용해 진행되었습니다. 연구진은 InceptionV3 네트워크를 사용해 고혈압 분류 모델을 훈련시켰으며, 다양한 bias mitigation methods(편향 완화 방법)을 적용해 성능 평가 및 차이 원인을 분석했습니다. 사용된 방법에는 Resampling, Group Distributionnally Robust Optimisation (GroupDRO), Orthogonally Disentangled Representations (ODR), Domain-Independent learning (DomainInd), Learning-Not-to-Learn (LNL), SWAD, ReSWAD, 그리고 Recalibration이 포함되었습니다.

- **Performance Highlights**: 기본 InceptionV3 모델은 73±0.01%의 정확도와 71±0.00%의 AUC를 기록했습니다. 하지만 세부 평가에서 나이, 평가 센터, 성별 등 여러 하위 그룹에서 상당한 성능 차이가 발견되었습니다. 예를 들어, 나이 그룹 간 AUC는 15% 이상, 평가 센터 간은 10% 이상 차이가 있었습니다. 또한, 주어진 편향 완화 방법들이 일관되게 모델의 공정성을 향상시키는 데 실패했음을 밝혔습니다. 이는 특정 유형의 편향에 맞춘 더 나은 방법이 필요함을 시사합니다.



### Dynamic Language Group-Based MoE: Enhancing Efficiency and Flexibility for Code-Switching Speech Recognition (https://arxiv.org/abs/2407.18581)
- **What's New**: 이 논문에서는 Mixture of Experts (MoE) 접근법을 사용하여 다국어 및 코드 스위칭(code-switching, CS) 문제를 해결하는 DLG-MoE 모델을 소개합니다. 이 모델은 언어 기반 라우터(language router)와 독립적인 비지도 학습 라우터(unsupervised routers)를 통해 언어 외의 속성을 처리합니다. 이 모델은 별도의 사전 학습(pre-training) 없이도 스트리밍 인식(streaming recognition)을 지원하며, 코드도 공개될 예정입니다.

- **Technical Details**: DLG-MoE는 언어 기반 그룹을 활용한 동적 라우터(dynamic router)를 특징으로 하며, 이는 언어 모델링을 보다 명확하게 수행할 수 있도록 합니다. 언어 그룹 내에서는 독립적인 비지도 학습 라우터가 다양한 속성을 처리합니다. 또한, 이 구조는 전문가 확장 및 dynamic top-k training을 지원하여 유연한 추론을 가능하게 합니다.

- **Performance Highlights**: DLG-MoE는 사전 학습 없이도 SOTA(state-of-the-art) 성능을 달성하였으며, 다른 방법들보다 유연성을 크게 향상시켰습니다.



### Explaining Reinforcement Learning: A Counterfactual Shapley Values Approach (https://arxiv.org/abs/2408.02529)
- **What's New**: 이번 논문에서는 강화학습(RL)에서 설명 가능성을 높이기 위해 반사실적 분석(counterfactual analysis)과 샤플리 값(Shapley Values)을 결합한 새로운 접근법인 Counterfactual Shapley Values(CSV)를 소개합니다. 이 접근법은 다양한 행동 선택에 있어 상태의 각 차원이 미치는 영향을 정량화하고 비교하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 'Counterfactual Difference Characteristic Value'와 'Average Counterfactual Difference Characteristic Value'라는 두 가지 새로운 특성 값을 도입하여 최적 행동과 비최적 행동 간의 기여도 차이를 평가하기 위한 샤플리 값을 계산합니다. 이 접근법은 GridWorld, FrozenLake, Taxi와 같은 여러 RL 도메인에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: CSV 방법은 복잡한 RL 시스템에서 투명성을 향상시킬 뿐만 아니라 다양한 결정 사항들 사이의 차이를 정량화하는 데도 효과적인 것으로 나타났습니다. 실험 결과는 이 방법이 에이전트의 행동을 설명하는 데 있어서 효과적임을 보여줍니다.



### SentenceVAE: Enable Next-sentence Prediction for Large Language Models with Faster Speed, Higher Accuracy and Longer Contex (https://arxiv.org/abs/2408.00655)
Comments:
          update the article

- **What's New**: 본 논문에서는 기존 대형 언어 모델(LLMs)의 예측 방식인 다음 토큰(next-token) 예측 방법이 처리 속도를 저하시킨다는 문제를 다룹니다. 이를 해결하기 위해 문장 단위 예측 방식인 'next-sentence prediction'을 제안합니다. 이 새로운 방식은 Sentence Variational Autoencoder (SentenceVAE)를 도입하여 문장 정보를 단일 토큰으로 압축 후 복원하는 과정을 통해 보다 빠르고 효율적인 추론을 가능하게 합니다.

- **Technical Details**: SentenceVAE는 Sentence Encoder와 Sentence Decoder로 구성되며, 문장의 정보를 단일 토큰으로 압축하고 이를 다시 원래의 문장으로 복원하는 기능을 가집니다. 이 SentenceVAE를 LLM의 입력 및 출력 층에 삽입하여 'Sentence-level LLMs (SLLMs)'를 개발하였습니다. SLLMs는 문장 단위 예측 방식을 사용하여 문맥을 문장으로 분할하는 동시에 원래의 의미를 유지합니다. 이를 통해 추론 속도를 크게 향상시키고, 메모리 부담을 줄이며 더 긴 문맥을 처리할 수 있습니다.

- **Performance Highlights**: Wanjuan 데이터셋을 활용한 광범위한 실험 결과, 제안된 방법은 추론 속도를 204~365% 향상시키고, Perplexity (PPL)를 기존 대비 46~75%로 감소시키며, 동등한 문맥 길이에 대해 메모리 오버헤드를 86~91% 줄이는 것으로 나타났습니다. 이러한 결과는 기존의 토큰 단위 예측 방법보다 월등한 성능 향상을 의미합니다.




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




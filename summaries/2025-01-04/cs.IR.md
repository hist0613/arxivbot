New uploads on arXiv(cs.CL)

### OmniChat: Enhancing Spoken Dialogue Systems with Scalable Synthetic Data for Diverse Scenarios (https://arxiv.org/abs/2501.01384)
- **What's New**: 최근 인공지능의 급속한 발전에 따라, 말하는 대화 시스템이 인간과의 자연스러운 대화를 가능하게 하는 중요한 분야로 떠오르고 있습니다. 그러나 기존의 대화 시스템은 실제 대화의 복잡함을 완전히 처리하는 데 한계가 있는데, 이는 현재의 대화 데이터셋이 규모와 상황 다양성 두 가지 측면에서 제약을 받고 있기 때문입니다. 본 논문에서는 다양한 시나리오를 아우르는 대화 모델 강화를 위한 합성 데이터 활용 방안을 제시하며, 그 일환으로 ShareChatX를 소개합니다.

- **Technical Details**: ShareChatX는 다채로운 시나리오를 포괄하는 첫 번째 대규모 대화 데이터셋으로, 감정을 (-Emotion), 오디오 이벤트를 (-Audio), 음악을 (-Music) 포함하고 있습니다. 이 데이터셋을 기반으로, OmniChat이라는 다중 턴 대화 시스템이 소개되는데, 이는 서로 다른 대화 맥락에 맞춰 특징 선택을 최적화하는 이종 특징 융합 모듈을 갖추고 있습니다. 연구진은 합성 데이터를 활용하여 대화 시스템을 학습하는 데 있어 최적의 데이터 균형을 찾아내어 실제 대화 데이터셋인 DailyTalk에서 최첨단 결과를 달성했습니다.

- **Performance Highlights**: 실험을 통해, 합성 데이터의 중요성을 강조하며 특히 오디오 및 음악을 포함한 복잡한 대화 시나리오를 해결하는 데 있어 큰 기여를 했습니다. ShareChatX와 OmniChat의 도입으로, 기존에 비해 대화 시스템의 성능이 크게 향상되었습니다. 본 논문은 대화의 감정, 오디오 이벤트, 음악을 포함한 다양한 시나리오에 대한 데이터셋과 시스템을 갖춘 최초의 연구로, 향후 대화 시스템의 발전에 큰 기여를 할 것으로 기대됩니다.



### Training Medical Large Vision-Language Models with Abnormal-Aware Feedback (https://arxiv.org/abs/2501.01377)
Comments:
          16 pages

- **What's New**: 본 연구에서는 기존의 Medical Large Vision-Language Models (Med-LVLMs)의 한계를 극복하기 위해 UMed-LVLM을 제안합니다. 이 모델은 의료 이미지에서의 시각적 로컬라이제이션(visual localization) 문제를 해결하는 데 중점을 두고 개발되었습니다. 또한 Medical Abnormalities Unveiling (MAU) 데이터셋을 사용하여 병리학적 이상 감지 능력을 보강합니다.

- **Technical Details**: UMed-LVLM은 두 단계의 훈련 방법인 Abnormal-Aware Instruction Tuning과 Abnormal-Aware Rewarding을 통해 교육됩니다. Abnormal-Aware Rewarding은 Abnormal Localization Rewarding과 Vision Relevance Rewarding을 포함하여 모델이 이상 영역을 효과적으로 캡처할 수 있도록 설계되었습니다. 이는 의료 이미지를 이해하고 이에 따른 진단을 생성하는 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, UMed-LVLM은 기존의 Med-LVLMs와 비교하여 의료 이미지를 이해하고 이상을 식별하는 데 있어 뛰어난 성능을 보여주었습니다. 또한, 모델을 훈련하고 일반화 능력을 심층 분석한 결과, Med-LVLMs의 이상 감지 능력을 개선하는 것이 의료 이미지 이해에 미치는 긍정적인 영향을 강조했습니다. 대규모 모델이 제한된 질병 유형의 다양한 의료 이미지에 노출되더라도 robust한 질병 인식을 위한 잠재력을 갖추고 있음을 보여주었습니다.



### Aligning Large Language Models for Faithful Integrity Against Opposing Argumen (https://arxiv.org/abs/2501.01336)
Comments:
          17 pages, 5 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 신뢰성을 높이기 위한 새로운 프레임워크인 '신뢰성 있는 무결성 정렬을 위한 신뢰도 추정(ALignment for Faithful Integrity with Confidence Estimation, AFICE)'을 제안합니다. 기존에 모델이 반대 주장을 받고도 올바른 주장을 유지하지 못하는 문제를 해결하고자 합니다. AFICE는 대화의 맥락에서 LLM 응답의 신뢰도를 평가하고, 이를 통해 모델이 반대 주장을 만났을 때에도 일관된 답변을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: AFICE는 양자 신뢰도 추정(Bilateral Confidence Estimation, BCE) 접근 방식을 사용하여 LLM의 응답을 평가합니다. 이 방법은 주어진 맥락에 따라 모델의 각 응답의 불확실성을 추정하며, 내부 상태와 누적 확률 비율을 고려합니다. 이를 통해, 우리는 원주어진 상태 및 사용자의 주장을 바탕으로 LLM을 정렬하고, 최종적으로 직접 선호 최적화(Direct Preference Optimization, DPO)를 통해 모델을 미세 조정합니다.

- **Performance Highlights**: 실험 결과, AFICE는 네 가지 질문 범주(수학, 1차 논리, 상식, 일반)에 걸쳐 기존의 기준을 초과하는 성능 개선을 보여주었습니다. 특히, 반대 주장을 접했을 때에도 LLM이 신뢰할 수 있는 응답을 유지하는 능력이 크게 향상되었습니다. 이러한 결과는 LLM의 실용성과 신뢰성을 보장함으로써 복잡한 상호작용 환경에서의 유용성을 강조합니다.



### Decoding Knowledge in Large Language Models: A Framework for Categorization and Comprehension (https://arxiv.org/abs/2501.01332)
- **What's New**: 이 논문은 LLM(대형 언어 모델)의 지식을 정확성과 신뢰성 두 축에서 분류하는 새로운 프레임워크인 K-(CSA)²를 소개합니다. 기존의 정확도 기반 평가를 넘어서 LLM의 지식 구조를 보다 정교하게 평가할 수 있는 방법론을 제시하며, 이 프레임워크는 LLM이 지식을 어떻게 습득하고 표현하는지를 평가할 수 있도록 지원합니다. 다양한 질문에 대한 답변의 일관성 및 신뢰도를 측정함으로써, LLM의 지식 범주를 더욱 명확히 파악할 수 있게 됩니다.

- **Technical Details**: 제안하는 K-(CSA)² 프레임워크는 LLM의 지식을 '알고 있는'(Known)과 '모르는'(Unknown) 두 가지 주요 유형으로 나누고, 각 유형은 세 가지 하위 범주로 further 세분화됩니다. 이 프레임워크는 답변의 정확성과 질문에 대한 모델의 응답의 일관성을 기반으로, 모델이 특정 지식을 얼마나 잘 이해하고 있는지를 평가합니다. 각 지식 카테고리는 지식 포인트의 정확도를 바탕으로 분류되며, 정확한 답변을 적어도 한 번 생성하면 '아는' 정보로 간주됩니다.

- **Performance Highlights**: 실험 결과는 Chain-of-Thought(사고의 연쇄) 프롬프팅 기법이 LLM의 내부 지식 구조를 보다 개선하는 반면, 외부 지식은 지시 조정을 통해 더 잘 향상된다는 점을 보여줍니다. 고수준의 지식은 LLM의 상위 레이어에 집중되는 반면, 낮은 신뢰도의 지식은 중간 및 하위 레이어에서 나타나는 경향이 있습니다. 이러한 발견은 LLM 훈련 방법의 차별적 최적화 필요성을 시사합니다.



### Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process of Fast and Slow Thinking (https://arxiv.org/abs/2501.01306)
- **What's New**: 이번 논문에서는 HaluSearch라는 새로운 프레임워크를 제안합니다. HaluSearch는 트리 검색 기반 알고리즘(MCTS 등)을 통합하여 LLM의 추론 과정에서 환각 현상을 줄이기 위한 명백한 느린 사고 생성 프로세스를 가능하게 합니다. 이 연구는 요청에 대한 응답 정확도를 높이기 위해 LLM의 내부 지식을 최대한 활용하는 방법을 제시합니다.

- **Technical Details**: HaluSearch는 각 문장을 개별적인 추론 단계로 간주하여 텍스트 생성을 단계별로 진행하는 구조를 가지고 있습니다. 이를 통해 빠른 사고 모드와 느린 사고 모드 간의 동적인 전환을 통해 추론 과정을 최적화하며, 단계별 보상 모델을 통해 생성된 각 단계의 질을 평가합니다. HaluSearch 프레임워크는 입력 프롬프트에 따라 적절한 사고 모드를 선택하여 응답을 생성합니다.

- **Performance Highlights**: HaluSearch는 Llama3.1-8B-Instruct와 Qwen2-7B-Instruct와 같은 정책 모델을 사용하여 여러 실험을 진행하였고, 이전의 프롬프트 기반 및 추론 시간 개입 방법에 비해 상당한 개선 성과를 나타냈습니다. 특히 영어 및 중국어 데이터셋에서 효과적인 환각 평가 및 응답 품질 향상을 보여주었으며, 동적인 시스템 전환 메커니즘이 다양한 시나리오에서 효율성과 정확성 간의 균형을 이루는 데 효과적임을 입증했습니다.



### Large Language Models for Mental Health Diagnostic Assessments: Exploring The Potential of Large Language Models for Assisting with Mental Health Diagnostic Assessments -- The Depression and Anxiety Cas (https://arxiv.org/abs/2501.01305)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 의료 분야에서 진단 평가를 지원하는 가능성을 탐구합니다. 특히, 주요 우울장애(MDD)와 일반화된 불안장애(GAD)에 대한 PHQ-9 및 GAD-7 설문지를 분석하고, 이러한 프로세스에 적합하게 LLM을 조정하는 다양한 기법을 조사합니다. "Diagnostic Llama"로 명명된 새롭게 튜닝된 모델이 다른 접근법과 비교하여 어떻게 진단 기준 평가에 기여할 수 있는지를 제시하고 있습니다.

- **Technical Details**: LLMs는 70억 개 이상의 가중치를 가진 대형 신경망으로, 복잡한 언어 표현을 인코딩하기 위해 대규모 데이터셋으로 학습됩니다. 이 연구는 GPT-3.5, GPT-4o와 같은 상업적 모델과 llama-3.1-8b 및 mixtral-8x7b와 같은 오픈소스 모델을 활용하여 모델의 동작을 표준 진단 절차에 맞게 유도하기 위해 프롬프팅 및 핀튜닝 기법을 연구합니다. PHQ-9 및 GAD-7의 증상에 맞는 텍스트 범위를 식별하여 진단 기준을 평가하는 방법론이 논의됩니다.

- **Performance Highlights**: 실험 결과, LLM이 제공한 진단 결과와 전문가의 검증된 결과 간의 일치를 평가하는 과정에서 상당한 합의가 이루어졌습니다. 연구진은 PHQ-9 기준에 맞춰 LLM을 핀튜닝한 첫 번째 모델을 소개하며, 이는 향후 정신 건강 진단 평가에 유용할 것으로 기대됩니다. 또한, 고품질의 언어 모델 주석이 있는 합성 데이터 세트를 제공하여 추가 연구의 기초를 마련합니다.



### Citations and Trust in LLM Generated Responses (https://arxiv.org/abs/2501.01303)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 연구는 질문 답변 시스템의 사용자가 느끼는 신뢰성(trust)를 조사하고, 인용(citation)의 존재가 그 신뢰성에 미치는 영향을 분석합니다. 기존 연구와 달리, 인용이 있는 경우와 사용자가 이를 확인하는 경우의 상반된 효과를 관찰하였습니다.

- **Technical Details**: 연구는 상업용 Chatbot이 생성한 텍스트 응답에 대해 다양한 양의 인용(0개, 1개, 5개)과 관련 있는 인용 및 무작위 인용을 적용하여 실험을 진행했습니다. 이 실험에서 참여자들의 인용 확인 여부와 AI 응답에 대한 신뢰감을 자체보고(self-reported) 형태로 수집했습니다.

- **Performance Highlights**: 결과적으로, 인용이 존재할 때 신뢰감이 유의미하게 증가하였고, 무작위 인용이더라도 이러한 효과가 유지되었습니다. 반면, 인용을 확인한 경우에는 신뢰감이 현저히 감소하는 경향을 보였습니다. 이는 AI가 생성한 콘텐츠의 신뢰도 향상에 있어 인용의 중요성을 강조합니다.



### ToolComp: A Multi-Tool Reasoning & Process Supervision Benchmark (https://arxiv.org/abs/2501.01290)
- **What's New**: 본 논문은 복잡한 다단계 도구 사용(reasoning) 작업을 평가하기 위한 ToolComp라는 새로운 벤치마크를 소개합니다. 기존 벤치마크는 실제 도구 사용의 복잡성을 제대로 반영하지 못하고 있으며, ToolComp는 최종 결과뿐만 아니라 중간 단계의 정확성도 평가할 수 있도록 설계되었습니다. 이를 통해 AI 시스템의 인퍼런스 타임(failure)에 대한 평가와 개발에 기여할 수 있습니다.

- **Technical Details**: ToolComp는 모델과 인간 주석자 간의 협업을 통해 개발되었으며, 이는 인간이 편집 및 검증한 프롬프트(prompt), 최종 답변(final answers), 그리고 과정 감시(labels)를 포함합니다. 여섯 가지 모델 패밀리를 통한 평가 결과, 대부분의 모델이 50% 미만의 정확도를 기록하여 데이터셋의 도전적인 특성을 보여줍니다. 또한, 과정 감독(process supervision) 모델(PRMs)과 결과 감독(outcome-supervised) 모델(ORMs)의 성능을 비교하기 위해 합성 훈련 데이터(synthetic training data)를 생성했습니다.

- **Performance Highlights**: 결과에 따르면 과정 감독 모델(PRMs)은 결과 감독 모델(ORMs)보다 훨씬 더 나은 일반화를 보여주며, 각각 기본 모델과 세부 조정 모델의 rank@1 정확도에서 19%와 11%의 개선을 달성했습니다. 이러한 결과는 AI 모델의 평가와 훈련 모두에서 과정 감독의 중요성을 강조하며, 복잡한 다단계 도구 사용 작업에 적합한 더 강력하고 유능한 시스템 개발의 길을 열어줍니다.



### NeutraSum: A Language Model can help a Balanced Media Diet by Neutralizing News Summaries (https://arxiv.org/abs/2501.01284)
- **What's New**: 이번 연구에서는 미디어 바이어스를 줄이기 위한 새로운 프레임워크인 NeutraSum을 제안합니다. 이 프레임워크는 두 가지 중립성 손실(neutrality losses)을 통합하여, 생성된 요약의 의미 공간을 조정하고 미디어 바이어스를 최소화하는 데 목표를 두고 있습니다. NeutraSum은 같은 사건을 보도하는 편향된 뉴스 기사들을 통합하여 중립적이고 사실에 근거한 요약을 생성할 수 있도록 합니다. 실험 결과는 NeutraSum이 요약 성능을 향상시키고 미디어 바이어스를 현저히 줄이는 데 기여함을 보여줍니다.

- **Technical Details**: NeutraSum 모델은 다중 문서 요약(multi-document summarisation) 손실을 사용하여 고품질 요약을 생성하며, 두 가지 중립성 손실인 대비 손실(contrastive loss)과 동일 거리 손실(equal-distance loss)을 활용합니다. 이러한 손실들은 편향된 출처 간 의미 공간을 조정하고, 전문가 작성 요약과의 일치를 보장하여 중립적인 텍스트 생성을 유도합니다. 이 모델은 Allsides 데이터셋에서 다양한 정치적 편향을 가진 기사들로부터 동일한 사건에 대한 정보를 통합하여 요약을 생성합니다.

- **Performance Highlights**: 실험을 통해 NeutraSum은 요약 과정에서 미디어 바이어스를 효과적으로 감소시키면서도 핵심 정보를 지속적으로 보존하는 성과를 보여줍니다. 특히, 다중 문서 요약 손실과 중립성 손실의 조합이 모델이 보다 중립적인 출력을 생성하는 데 중요한 역할을 했습니다. 이러한 접근 방법은 이후 뉴스 요약에서의 공정성과 중립성을 높이는 데 기여할 것으로 기대됩니다.



### Does a Large Language Model Really Speak in Human-Like Language? (https://arxiv.org/abs/2501.01273)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 생성한 텍스트와 인간이 작성한 텍스트 간의 잠재적인 커뮤니티 구조를 비교하는 새로운 통계적 가설 검정 프레임워크를 제안합니다. 이 연구는 원본 인간 작성 텍스트와 LLM으로 패러프레이즈한 텍스트, 그리고 두 번 패러프레이즈한 텍스트 간의 차이를 분석합니다. 저자들은 LLM이 생성한 텍스트가 실제 인간의 언어와 얼마나 유사한지를 평가하고자 하며, 이는 NLP 분야에서의 기존 연구 간극을 메우는 데 기여할 것입니다.

- **Technical Details**: 이 연구는 30,000개 이상의 인간 작성 텍스트를 웹 크롤링을 통해 수집하고, 이를 기반으로 LLM이 생성한 패러프레이즈 텍스트를 생성하였습니다. 핵심적으로, 통계적 가설 검정 절차를 통해 각 데이터 세트의 상대적 위치를 기반으로 데이터를 매핑하고, 이를 통해 서로 다른 데이터 세트 간의 직접적인 비교를 가능하게 합니다. 저자들은 텍스트 임베딩을 활용하여 두 데이터 세트 간의 커뮤니티 구조 차이를 탐구합니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT가 생성한 텍스트는 인간이 작성한 텍스트와 뚜렷한 차이를 보입니다. 이는 LLM이 텍스트 생성 과정에서 특정한 특성과 패턴을 가지며, 따라서 인간 작성 텍스트와는 다른 수준의 언어 복잡성을 보여주는 것을 시사합니다. 이러한 발견은 NLP 응용 분야에서 LLM의 활용 가능성을 제시하면서도, 인간 이해와 의도를 모방하거나 복제하는 것에 대한 한계를 드러냅니다.



### ProgCo: Program Helps Self-Correction of Large Language Models (https://arxiv.org/abs/2501.01264)
Comments:
          Working in progress

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 자기 검증(self-verify)과 자기 수정(self-refine) 기능을 극대화하기 위한 새로운 접근 방식인 Program-driven Self-Correction (ProgCo)를 제안합니다. 기존 LLM들이 복잡한 추론 작업에서 자주 실패하는 문제를 해결하기 위해, 자체 생성된 검증 의사 프로그램(pseudo-program)을 사용하여 복잡한 검증 로직을 구현합니다.

- **Technical Details**: ProgCo는 두 가지 주요 구성 요소로 구성됩니다. 첫째, 프로그램 기반 검증(ProgVe)은 LLM이 만들어낸 검증 프로그램을 통해 검증 로직을 수행하며, 둘째, 프로그램 기반 수정보강(ProgRe)은 ProgVe의 피드백을 받아 응답과 검증 프로그램 모두를 이중 반영(dual reflection)하여 수정합니다. 이러한 방식은 잘못된 피드백으로 인한 혼란을 줄이는 데 효과적입니다.

- **Performance Highlights**: 세 가지 지침 준수(instruction-following) 및 수학적 벤치마크에서 실시된 실험 결과, ProgCo는 효과적인 자기 수정을 달성했으며, 실제 프로그램 도구와 결합할 경우 성능이 더욱 향상될 수 있음을 보였습니다. 이 연구는 LLM의 자기 개선 가능성을 새롭게 열어줄 수 있는 기초를 마련합니다.



### CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings (https://arxiv.org/abs/2501.01257)
- **What's New**: 이 논문에서는 CodeElo라고 불리는 새로운 코드 생성 벤치마크를 소개합니다. 기존의 벤치마크들이 가지는 한계를 극복하고자 개발된 이 벤치마크는 CodeForces 플랫폼에서의 문제를 기반으로 하여, 공정하고 정확한 평가를 가능하게 합니다. CodeElo는 모든 테스트 케이스에 접근할 수 있는 구조를 제공하고, 특별 심사자(special judges)를 지원함으로써 0%의 잘못된 긍정률(false positive rate)을 달성합니다.

- **Technical Details**: CodeElo는 6개월간의 CodeForces의 최근 대회 문제를 수집하여 카테고리화하고, 각 문제의 난이도와 알고리즘 태그를 포함합니다. 문제 제출 방식은 자동으로 모델 솔루션을 CodeForces에 제출하여 테스트 결과를 받는 방식을採용하며, 이를 통해 인적 평가와 유사한 방식으로 문제를 평가합니다. 또한, 독립적인 Elo 등급 시스템을 도입하여 각 모델의 성과를 정량적으로 나타냅니다.

- **Performance Highlights**: 성능 분석 결과, OpenAI o1-mini 모델이 1578이라는 높은 Elo 등급을 기록하며 가장 우수한 성과를 보였습니다. 반면 대부분의 모델은 가장 간단한 문제조차 통과하기 어려워 최하위 10%의 Elo 등급에 머물렀습니다. 이러한 결과는 각 모델별로 요구되는 코드의 성격과 언어에 따라 성과가 상이함을 나타내며, C++가 대부분의 모델에서 가장 좋은 성능을 보였음을 확인했습니다.



### Digital Guardians: Can GPT-4, Perspective API, and Moderation API reliably detect hate speech in reader comments of German online newspapers? (https://arxiv.org/abs/2501.01256)
- **What's New**: 최근 몇 년 동안 인터넷에 유해한 콘텐츠와 증오 발언이 널리 퍼지고 있으며, 온라인 신문과 포럼의 모더레이터는 독자 댓글을 신중히 검토하고 필요한 경우 삭제해야 합니다. OpenAI의 GPT-4o와 Google의 Perspective API, 그리고 OpenAI의 Moderation API 등 자동화된 증오 발언 탐지 솔루션이 비교되며, 독일어 테스트 데이터셋 HOCON34k를 활용한 실험 결과 GPT-4o가 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구는 증오 발언 탐지에 관한 OpenAI의 GPT-4o, Google의 Perspective API, OpenAI의 Moderation API의 효과성을 비교합니다. HOCON34k 데이터셋은 1,592개의 주석이 달린 샘플로 구성되며, 각 솔루션은 사전 학습된 모델을 사용하고, 변경된 훈련 없이 평가됩니다. GPT-4o는 Zero-Shot, One-Shot, Few-Shot 학습 접근법을 사용하여 다양한 실험이 실시되었습니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 Perspective API와 Moderation API를 초과하여 HOCON34k 기준보다 약 5 포인트 높은 성능을 보였습니다. 모든 실험에서 GPT-4o는 더 나은 F2-score와 Matthews 상관 계수를 달성하였으며, 이는 독일어 온라인 신문에서의 증오 발언 탐지에 있어 효과적임을 시사합니다.



### Large Language Model-Enhanced Symbolic Reasoning for Knowledge Base Completion (https://arxiv.org/abs/2501.01246)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)과 규칙 기반 추론을 결합한 혁신적인 프레임워크인 LeSR(LLM-enhanced Symbolic Reasoning)를 제안합니다. 이 프레임워크는 Subgraph Extractor, LLM Proposer, Rule Reasoner로 구성되어 있으며, LLM의 이해 능력과 규칙 기반 접근 방식의 논리적 강점을 결합하여 지식 베이스 완성(KBC)의 효과성을 높이고 신뢰성을 개선합니다. 기존의 KBC 방법들이 가지고 있는 유연성과 다양성의 부족 문제를 해결하기 위해 LLM을 규칙 탐색 과정에 활용하고, 이로써 더 나은 결과를 도출하고자 합니다.

- **Technical Details**: LeSR 프레임워크는 두 가지 주요 구성 요소로 나뉩니다. 첫째, Subgraph Extractor는 KBC에서 특정 관계를 둘러싼 의미 있는 서브그래프를 식별하고, 둘째, LLM Proposer는 이 서브그래프를 바탕으로 다양한 제안 규칙을 생성합니다. 생성된 규칙들은 Rule Reasoner에 의해 평가되고 개선되어, LLM의 생성 과정에서 발생할 수 있는 오류를 줄이고 유용한 규칙만을 선별하여 KBC 과정의 품질을 높입니다.

- **Performance Highlights**: 제안된 방법은 다섯 개의 지식 베이스 벤치마크에서 폭넓게 실험을 수행하였으며, 다양한 도메인과 복잡성을 아우르며 경쟁력 있는 성능을 보였습니다. 또한, LeSR은 충분한 다양성과 포괄성을 지닌 해석 가능한 규칙을 생성하여 KBC에 대한 강력하고 일반화 가능한 솔루션의 효과성을 입증하였습니다. 이러한 특성 덕분에 이 방법은 전통적인 KBC 접근 방식에 비해 훨씬 높은 신뢰도를 제공합니다.



### Automated Self-Refinement and Self-Correction for LLM-based Product Attribute Value Extraction (https://arxiv.org/abs/2501.01237)
- **What's New**: 이번 연구는 제품 속성 값 추출(Product Attribute Value Extraction) 작업을 위한 두 가지 자기 개선 기법인 오류 기반 프롬프트 재작성(Error-based Prompt Rewriting) 및 자기 수정(Self-Correction)을 적용합니다. 기존의 대형 언어 모델(LLM)은 다양한 데이터를 다루는 데 효과적이지만, 자기 개선 기법이 성능 향상을 가져오지 않는 경우가 많아 비용이 증가하는 문제점을 지적하고 있습니다. 이 논문은 이러한 문제에 대한 실험적 평가를 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 OpenAI의 GPT-4o를 활용하여 속성 값 추출을 위한 새로운 접근 방식을 실험했습니다. 오류 기반 프롬프트 재작성 기법은 레이블이 붙은 학습 샘플에서 모델의 오류를 분석하여 프롬프트의 품질을 개선합니다. 자기 수정 기법은 모델이 잘못 추출한 값을 발견했을 때 초기 출력 결과를 검토하고 업데이트하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, 자기 개선 기법이 모델 성능에 미치는 영향은 미미하였지만, 처리 비용은 상당히 증가하는 것으로 나타났습니다. 훈련 데이터가 존재하는 경우, 파인 튜닝(Fine-tuning)이 가장 높은 성능을 보였으며, 제품 설명의 양이 증가함에 따라 파인 튜닝 비용이 균형을 이룬 것으로 확인되었습니다.



### Data Augmentation Techniques for Chinese Disease Name Normalization (https://arxiv.org/abs/2501.01195)
Comments:
          The Version of Record of this contribution is published in 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

- **What's New**: 본 논문에서는 질병 명칭 정규화 (disease name normalization)의 데이터를 증가시키기 위한 혁신적인 접근법인 Disease Data Augmentation (DDA)을 제안합니다. 기존 시스템들의 훈련 데이터가 부족한 문제를 해결하기 위해 다양한 데이터 증강 기술과 지원 모듈을 통합하였습니다. 이를 통해 DDA 접근법이 여러 기본 모델과 훈련 목표에서 성능 개선을 나타냄을 실험적으로 입증하였습니다.

- **Technical Details**: DDA 접근법에서는 질병 명칭 정규화를 위한 작업 정의 및 데이터 증강 방법을 도입합니다. 세 가지 주요 축 단어(질병 중심(disease center), 해부학적 영역(anatomical region), 질병의 특성(disease characteristic))를 정의하고, 이 축 단어를 찾아내기 위해 BiLSTM+CRF 기반의 명명된 개체 인식(NER) 시스템을 설계하였습니다. 데이터 증강 모듈은 두 가지 주요 범주인 축 단어 대체(Axis-word Replacement)와 다중 미세 집계(Multi-Granularity Aggregation)를 포함하여 질병 명칭의 다양한 구성 요소와 관계에 대한 추가 지식을 제공합니다.

- **Performance Highlights**: 자세한 실험 결과, DDA 접근법이 다른 데이터 증강 방법들을 능가하며 질병 명칭 정규화의 다양한 기본선 모델 성능을 효과적으로 향상시킴을 보여줍니다. 특히, 훈련 데이터가 부족한 상황에서도 DDA 접근법은 전반적인 성능의 약 80%에 가깝게 도달할 수 있음을 증명했습니다. 이러한 성과는 제한된 데이터로도 성능을 극대화할 수 있는 방법이 될 것입니다.



### Blind Men and the Elephant: Diverse Perspectives on Gender Stereotypes in Benchmark Datasets (https://arxiv.org/abs/2501.01168)
- **What's New**: 이번 논문은 언어 모델에서 성 고정관념 편향을 측정하고 완화하는 방법에 대해 다룹니다. 선행 연구에서 내재적 접근법(intrinsic approach)과 외연적 접근법(extrinsic approach) 간의 상관관계 부족이 밝혀졌으며, 본 연구는 내재적 측정의 복잡성을 탐구합니다. 연구진은 데이터 분포(data distribution)와 성 고정관념 요소를 분석하여 결과 일관성 개선을 위한 새로운 방법론을 제안합니다.

- **Technical Details**: 연구는 StereoSet과 CrowS-Pairs라는 두 가지 널리 사용되는 내재적 편향 평가 기준을 중심으로 진행됩니다. 데이터 세트의 샘플 분포(sample distribution) 차이가 두 기준 간 불일치를 유발한다는 가설을 세우고, 샘플 분포를 조정하여 결과의 일관성을 높일 수 있음을 실험을 통해 입증합니다. 연구진은 또한 성 고정관념 샘플을 수동으로 정리하여 데이터 세트의 구조적 개선을 시도하였습니다.

- **Performance Highlights**: 연구 결과에 따르면 StereoSet과 CrowS-Pairs 간의 상관관계는 매우 낮아, Pearson 상관계수는 0.13으로 확인되었습니다. 하지만 데이터 세트를 균형 있게 조정함으로써 두 기준의 상관관계를 최적화할 수 있는 가능성이 드러났습니다. 이러한 발견은 언어 모델에서의 성 고정관념 측정 및 완화 방법 개발에 있어 새로운 방향성을 제시합니다.



### Leveraging Full Dependency Parsing Graph Information For Biomedical Event Extraction (https://arxiv.org/abs/2501.01158)
Comments:
          6 figures, 4 tables

- **What's New**: 이 논문에서는 생의학적 사건 추출(BEE) 작업에서 의존 구문 분석 그래프(dependency parsing graph)를 활용한 새로운 그래프 컨볼루션 네트워크(GCN) 모델을 제안합니다. 기존의 최단 의존 경로(SDP) 방식의 한계를 극복하기 위해, 전체 인접 행렬을 사용하여 각 토큰을 임베딩합니다. 시험 결과, 의존 그래프 정보를 활용했을 때 성능이 획기적으로 개선되었으며, 이 모델은 다양한 데이터셋에서 기존의 최신 모델들을 소폭 초월하는 성과를 나타냈습니다.

- **Technical Details**: 논문은 네 가지 주요 기여를 강조합니다. 첫 번째로, 의존 구문 분석 정보를 사용하는 모델과 사용하지 않는 모델의 실험적 비교를 통해 추가 정보의 효과를 분석합니다. 두 번째로, BioBert의 컨텍스추얼 표현 위에 GNN을 임베딩하여 각 문장에서 개별 토큰의 표현을 향상시킵니다. 또한 모델 구조를 최적화하여 그래프 컨볼루션 네트워크에서 발생할 수 있는 표현력 부족을 줄입니다.

- **Performance Highlights**: 이 모델은 의존 구문 분석 그래프를 고려하여 BEE의 네 가지 하위 작업을 해결하는 데 강력한 성과를 보였습니다. 특히, 네트워크의 경량화 및 파라미터 수를 줄이기 위해 간단한 수치 실험 모델을 이용했습니다. 규명된 연구 결과는 의존 관계의 전체 그래프를 활용한 것이 약간 손실된 민감한 토큰을제거한 다른 연구들과 비교하여 성능 우위를 나타냈습니다.



### BlockDialect: Block-wise Fine-grained Mixed Format for Energy-Efficient LLM Inferenc (https://arxiv.org/abs/2501.01144)
- **What's New**: 이번 논문에서는 BlockDialect라는 새로운 블록-기반의 미세 조정된 혼합 형식 기법을 제안합니다. 이는 각 블록에 최적의 숫자 형식을 할당하여 언어 모델의 데이터 분포를 정확하게 표현할 수 있도록 합니다. 추가로, DialectFP4라는 다양한 FP4 변형의 형식 책자를 도입하여 블록 데이터 분포에 적응할 수 있게 하였습니다. 이를 통해 하드웨어 효율성을 보장하며, 낮은 정밀도의 정수 산술과 호환되는 표현 가능한 값들을 선택하는 방식으로 구성하였습니다.

- **Technical Details**: BlockDialect는 4비트의 가중치와 활성화 후량 조정을 가능하게 하는 혁신적인 방식으로, 각 블록에 대해 최적의 숫자 형식이 선택됩니다. 또한, DialectFP4를 통해 다양한 블록-수준 데이터 분포를 포착할 수 있게 하며, 온라인 DialectFP4 활성화 양자화를 위한 두 단계 접근 방식을 제안합니다. 이 방법은 기존 MSE 기반 접근법과 비교할 때, 동등한 정확도를 달성하는 것으로 확인되었습니다. 이러한 방식은 하드웨어 친화적인 작업을 위해 설계되어 있으며, 복잡한 연산에서도 동일한 수준의 에너지 효율성을 유지할 수 있도록 합니다.

- **Performance Highlights**: BlockDialect는 MXFP4 형식에 비해 LLaMA3-8B 및 LLaMA2-7B 모델에서 11.40%(6.90%)의 더 높은 제로샷 정확도 성능을 보여줍니다. 또한, 전체 정밀도와 비교했을 때 LLaMA3-8B는 5.89% 그리고 LLaMA2-7B는 3.31% 낮은 정확도를 유지합니다. 이 논문에서 제안한 방법은 기존 방법보다 더 많은 LLM에서 성능 향상을 이루어내는 손쉬운 해결책이 됩니다. 특히, 선형 레이어만 양자화 할 경우에는 전체 정확도에 비해 2.22%(1.20%)의 미미한 정확도 감소를 보입니다.



### TED: Turn Emphasis with Dialogue Feature Attention for Emotion Recognition in Conversation (https://arxiv.org/abs/2501.01123)
Comments:
          past activity in 2021

- **What's New**: 이 논문은 대화에서 감정 인식을 위한 새로운 방법인 ‘Turn Emphasis with Dialogue (TED)’를 제안합니다. TED는 대화 특징을 주의 메커니즘에 추가하여 각 턴을 명시적으로 구분합니다. 이 방법은 턴의 위치와 화자 정보에 따라 각 턴에 대한 우선 순위를 부여하고, 이를 통해 멀티 턴 입력을 더 잘 처리할 수 있습니다.

- **Technical Details**: TED는 돌 기반 인코딩(‘Turn-Based Encoding’, TBE) 및 다중 헤드 셀프 어텐션('Multi-Head Self-Attention', MHSA) 기법을 사용하여 불특정 다수의 턴을 인코딩합니다. 각 턴과 현재 턴 간의 관계를 조정하기 위해 우선순위 요소를 활용합니다. TED는 이전 및 이후 턴을 포함한 멀티 턴 입력 시퀀스를 생성하고, 이를 통해 다층적인 대화 맥락을 구축함으로써 감정 인식의 정확도를 높입니다.

- **Performance Highlights**: TED는 네 가지 기준 세트에서 평가되었으며, 모든 데이터셋에서 높은 성능을 나타냈습니다. 특히, TED는 IEMOCAP 데이터셋에서 최신 기술 수준의 성능을 달성하였습니다. 이를 통해 TED가 멀티 턴 대화의 감정 인식에 있어 효과적인 솔루션임을 입증하였습니다.



### BeliN: A Novel Corpus for Bengali Religious News Headline Generation using Contextual Feature Fusion (https://arxiv.org/abs/2501.01069)
Comments:
          28 pages, 4 figures, 11 tables

- **What's New**: 이 연구는 벵골어 종교 뉴스의 자동 요약, 특히 헤드라인 생성의 중요성을 강조합니다. 기존의 헤드라인 생성 방법은 주로 기사 내용에만 의존하고, 감정(sentiment), 범주(category), 및 측면(aspect)과 같은 중요한 맥락적 특징을 간과했습니다. 이를 해결하기 위해 새로운 데이터셋인 BeliN을 도입하고, MultiGen이라는 다중 입력(feature fusion) 방법을 제안합니다.

- **Technical Details**: MultiGen은 BanglaT5, mBART, mT5, mT0와 같은 트랜스포머 기반(pre-trained) 언어 모델을 활용하여 헤드라인 생성을 수행합니다. 이 접근법은 뉴스 콘텐츠와 함께 범주, 측면, 감정 등의 추가 맥락적 특징을 포함하여 모델이 종전 방법들이 간과했던 중요 정보를 포착할 수 있도록 합니다. 이를 통해 낮은 자원의 언어에서도 효과적인 결과를 도출할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 MultiGen은 기존의 기사 내용만 사용하는 기본 방법에 비해 우수한 성능을 보였습니다. BLEU 점수는 18.61, ROUGE-L 점수는 24.19로, 기본 방법의 점수인 16.08과 23.08에 비해 의미 있는 개선이 있음을 보여줍니다. 이러한 결과는 맥락적 특징을 헤드라인 생성에 통합하는 것의 중요성을 다시 한 번 강조합니다.



### Dynamic Attention-Guided Context Decoding for Mitigating Context Faithfulness Hallucinations in Large Language Models (https://arxiv.org/abs/2501.01059)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 사실성 문제를 해소하기 위한 Dynamic Attention-Guided Context Decoding (DAGCD)라는 새로운 프레임워크를 제안합니다. 기존 방법들이 가지는 한계를 보완하여, DAGCD는 주의(attention) 기법과 불확실성(uncertainty) 신호를 통합하여 보다 신뢰할 수 있는 출력을 생성할 수 있도록 합니다. 실험 결과, DAGCD는 여러 QA 데이터셋에서 사실성과 강인성을 동시에 높이며, 계산 효율성 또한 유지합니다.

- **Technical Details**: DAGCD는 주의 분포와 불확실성 신호를 결합하여 단일 패스 단일 디코딩 과정에서 동적으로 출력을 조정합니다. 연구를 통해 추론한 바와 같이, LLM의 주의 메커니즘은 입력 토큰 간의 관계를 명확히 하여 특정 출력을 생성하는 과정에서 컨텍스트(맥락) 사용을 나타내는 신호를 인코딩합니다. 이로 인해 모델이 특정 상황에서 적절한 신뢰도를 반영하여 결과를 산출할 수 있도록 기여합니다.

- **Performance Highlights**: 다양한 QA 데이터셋을 통한 실험 결과, DAGCD는 기존의 greedy decoding 방법에 비해 평균 F1 점수를 각각 8.95%와 1.91% 향상시키며, 사실성과 일관성을 강화하였습니다. 이러한 결과는 DAGCD가 LLM의 디코딩 효율을 높이고, 높은 불확실성에서의 응답 품질을 개선할 수 있음을 보여줍니다. 발표된 데이터들은 이 프레임워크의 강인성과 확장성을 입증하는 데 중요한 역할을 합니다.



### Risks of Cultural Erasure in Large Language Models (https://arxiv.org/abs/2501.01056)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 사회적 지식의 생산과 발견에 통합되어가는 추세를 강조합니다. 언어 모델이 글로벌 문화에 대한 사람들의 학습 및 인식 방식에 미치는 영향을 고려할 필요성이 증가하고 있으며, 특히 디지털 데이터에서 저조하게 나타나는 문화들을 다뤄야 한다고 주장합니다. 연구자들은 문화적 삭제(cultural erasure)라는 개념을 도입해 언어 모델이 문화적 다양성을 어떻게 왜곡할 수 있는지를 평가합니다.

- **Technical Details**: 본 연구는 문화 삭제를 평가하기 위해 두 가지 개념인 생략(omission)과 단순화(simplification)를 구분합니다. 생략은 문화가 전혀 표현되지 않는 경우이고, 단순화는 문화의 복잡성이 왜곡되어 일차원적인 관점으로 나타날 때를 의미합니다. 연구진은 전 세계 50개 도시의 설명을 생성하여 문화와 경제에 관한 주제를 분석하고, 이를 통해 도시들 간의 문화적 및 경제적 표현의 차이를 분석합니다.

- **Performance Highlights**: 연구 결과, 유럽 및 북미의 도시에서 문화 주제가 더 높은 점수를 얻은 반면, 아프리카와 아시아의 도시들은 경제 주제로 높은 점수를 받았습니다. 결과적으로 이 논문은 LLMs가 문화적 대표성을 어떻게 왜곡하는지에 대한 구체적인 증거를 제시하며, 특히 아프리카의 문화들이 심각하게 생략됨을 강조합니다. 연구자들은 이러한 결과를 바탕으로 언어 모델이 문화적 발견 및 생산을 지원하는 데 미치는 실질적인 영향을 논의합니다.



### Dynamic Scaling of Unit Tests for Code Reward Modeling (https://arxiv.org/abs/2501.01054)
Comments:
          Homepage: this https URL

- **What's New**: 이 연구는 단위 테스트(unit tests)의 수를 증가시키는 것이 코드 솔루션의 품질 보상을 향상시키는 효과가 있는지를 조사합니다. 연구 결과, 단위 테스트 수를 늘림으로써 다양한 모델과 코드 솔루션 수에 걸쳐 올바른 답변을 더 잘 식별할 수 있다는 긍정적인 상관관계를 발견했습니다. 또한, 'CodeRM-8B'라는 경량의 단위 테스트 생성기를 제안하여 효율적이고 고품질의 단위 테스트를 생성할 수 있도록 하고, 문제 난이도에 따라 동적으로 단위 테스트의 수를 조정하는 메커니즘도 구현하였습니다.

- **Technical Details**: 코드 생성을 위해, LLM(large language model)을 이용하여 프로그래밍 질문에 대한 다수의 코드 솔루션을 제시하고 이를 검증하는 단위 테스트를 작성하는 방법론을 사용합니다. 연구에서 제안되는 단위 테스트 기반의 다수결 구조는 표준 N중 최고(best-of-N) 전략을 따르며, LLM은 주어진 질문에 대해 N개의 후보 솔루션을 생성하여 M개의 단위 테스트를 작성합니다. 여러 개의 단위 테스트 실행 결과는 각 후보 솔루션에 대한 보상 신호를 형성하며, 이를 통해 가장 적합한 코드를 선택합니다.

- **Performance Highlights**: CodeRM-8B를 사용한 실험 결과, 다양한 모델에 걸쳐 성능이 크게 향상되었음을 보여주었습니다. 예를 들어, Llama3-8B 모델의 경우 HumanEval Plus에서 성능이 18.43% 향상되었고, GPT-4o-mini 모델에서도 3.42%의 개선이 관찰되었습니다. 또한, 문제의 난이도에 따라 자동으로 단위 테스트를 조정하는 동적 구현이 고정된 계산 비용 내에서 추가적인 성능 개선을 제공하여, 실험적 결과에서 이 메커니즘의 효과가 입증되었습니다.



### FED: Fast and Efficient Dataset Deduplication Framework with GPU Acceleration (https://arxiv.org/abs/2501.01046)
Comments:
          12 pages, 3 figures

- **What's New**: 본 논문은 GPU 클러스터를 최적화한 MinHash LSH 기반의 데이터셋 중복 제거 프레임워크인 	extit{FED}를 제안합니다. NVIDIA에서 발표한 기존 GPU 기반 MinHash LSH 방법보다 성능이 크게 향상되어, 1조 개의 토큰으로 구성된 데이터셋을 단 5.1시간 만에 중복 제거 할 수 있습니다. 이는 기존 CPU 기반 도구인 SlimPajama에 비해 최대 58.3배, NVIDIA NeMo Curator의 GPU 기반 도구에 비해 최대 8.6배 빠른 성능을 보여줍니다.

- **Technical Details**: 	extit{FED} 프레임워크는 계산 효율성이 높은 비암호화 해시 함수와 GPU 커널 작업 및 통신 최적화를 통해 중복 제거 프로세스를 개선하고 있습니다. 중복된 데이터 항목을 식별하는 과정에서 비고정적 해시 함수를 사용하여 이전 결과를 부분 재사용하는 방식을 채택하여 계산 비용을 대폭 감소시켰습니다. 이를 통해 해시 생성 속도가 기존 CPU 기반 방식보다 1,700배 이상, GPU 방식보다 100배 이상 빨라졌습니다.

- **Performance Highlights**: 이 연구에서 제안하는 	extit{FED}의 성능은 대규모 데이터셋에서의 중복 제거 품질과 속도를 모두 보장합니다. MinHash 구현을 기준으로 한 비교 평가를 통해 	extit{FED}의 빠르고 안정적인 중복 제거 결과를 확인했습니다. 또한, 모든 코드 베이스는 GitHub를 통해 공개되어 LLM의 데이터 품질을 유지하고 빠른 훈련 주기를 가능하게 할 것입니다.



### MSWA: Refining Local Attention with Multi-ScaleWindow Attention (https://arxiv.org/abs/2501.01039)
- **What's New**: 이번 논문에서는 Multi-Scale Window Attention (MSWA)라는 새로운 주의 메커니즘을 제안합니다. 기존의 Sliding Window Attention (SWA) 방식은 모든 헤드에서 동일한 창 크기를 사용하여 다양한 스케일의 맥락을 포착하는 데 비효율적이었습니다. MSWA는 서로 다른 창 크기를 적용하여 다양한 길이와 거리의 맥락 정보를 효과적으로 캡처할 수 있도록 합니다. 실험 결과에 따르면, MSWA는 전통적인 로컬 주의 방식보다 더 높은 효율성과 효과성을 보여줍니다.

- **Technical Details**: Multi-Scale Window Attention (MSWA)은 Transformer의 여러 레이어와 헤드에 걸쳐 다양한 창 크기를 적용합니다. 이 방법은 얕은 레이어에서 더 작은 창 크기를 할당하고 깊은 레이어로 갈수록 더 큰 창 크기를 할당하여, 모델이 지역 정보를 모델링하고 장거리 의존성을 포착할 수 있도록 지원합니다. 또 다른 유용한 점은 MSWA가 linear attention과 통합되어 효율성과 글로벌 주의력을 모두 갖출 수 있다는 것입니다. 이 접근 방식은 기존의 attention 가속 라이브러리에서 직접 구현 가능하다는 특징이 있습니다.

- **Performance Highlights**: 언어 모델링과 일반 상식 추론 과제에서 MSWA는 기존의 SWA보다 뛰어난 성능을 보였습니다. MSWA를 활용하여 학습한 모델은 효과적인 언어 모델링 능력을 입증했으며, 사전 훈련된 대형 언어 모델에 MSWA 패턴을 fine-tuning하여 향상된 결과를 도출하였습니다. 또한, 계산 효율성을 평가한 결과 MSWA는 표준 주의 방식이나 SWA에 비해 일관되게 더 나은 효율성을 기록하였습니다.



### Advancing Singlish Understanding: Bridging the Gap with Datasets and Multimodal Models (https://arxiv.org/abs/2501.01034)
Comments:
          Open-Source: this https URL

- **What's New**: 이번 논문에서는 영어에 뿌리를 둔 Creole 언어인 Singlish에 대한 연구를 진행하였으며, 기존에 잘 다루어지지 않았던 구어체 Singlish에 대한 분석을 위해 가장 큰 구어체 데이터셋인 Multitask National Speech Corpus (MNSC)를 소개합니다. 이 데이터셋은 Automatic Speech Recognition (ASR), Spoken Question Answering (SQA), Spoken Dialogue Summarization (SDS), Paralinguistic Question Answering (PQA)와 같은 다양한 작업을 지원하도록 표준화되고 주석 처리되었습니다. 또한, SingAudioLLM이라는 멀티태스크 멀티모달 모델을 제안하여 이러한 작업을 동시에 수행할 수 있게 하였습니다.

- **Technical Details**: 구어체 Singlish의 연구를 위해 구축된 MNSC는 기존의 언어 모델 및 다양한 기능을 통합하여 고도의 정향 있는 훈련 방식을 활용하여 제작되었습니다. 이 모델은 Joint Training을 통해 여러 작업에 대해 공동으로 학습하며, Fusion Architecture를 통해 성능을 극대화할 수 있는 구조를 갖추고 있습니다. 실험 결과 SingAudioLLM이 기존의 일반화된 AudioLLM 및 음성 인식 모델 대비 10% 이상의 성과 향상을 보였습니다.

- **Performance Highlights**: MNSC 데이터셋은 Singlish 구어 이해를 위한 최초의 잘 조직된 리소스로, 다국어 및 코드 전환 NLP 연구에서 포괄적인 평가를 가능하게 합니다. SingAudioLLM은 다양한 기준에서 최첨단 성능을 달성했으며, 이전 모델을 10-30% 초과하여 능가하는 성과를 보였습니다. 또한, 이 연구는 AudioLLM의 낮은 자원과 Creole 언어에 대한 가능성을 더욱 탐색하는 기회를 제공하고 있습니다.



### ValuesRAG: Enhancing Cultural Alignment Through Retrieval-Augmented Contextual Learning (https://arxiv.org/abs/2501.01031)
Comments:
          preprint

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG)과 in-context learning을 활용한 ValuesRAG라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 텍스트 생성 과정에서 문화적 및 인구통계학적 지식을 동적으로 통합하여 문화적 가치 정렬 문제를 해결하고자 합니다. ValuesRAG는 World Values Survey (WVS) 데이터를 활용하여 개인별 가치 요약을 생성하고, 인구통계학적 특징에 따라 관련된 요약을 검색하여 최종 결과를 도출합니다.

- **Technical Details**: ValuesRAG는 WVS 데이터셋을 기반으로, 논문에서는 먼저 다양한 인구통계학적 특성에 따라 100개의 관련된 요약을 검색하고, 이를 재정렬하여 상위 k개의 요약을 선택하는 방법을 설명합니다. 아울러, 성능 평가를 위해 zero-shot 추론, 역할 부여 방법, 몇샷 학습 방법 등 여러 기법들과 비교하여 ValuesRAG의 뛰어난 성능을 강조합니다. 이 과정에서 ValuesRAG는 다수의 인구통계적 요약을 동적으로 검색하고 통합하여 단일 정의된 프롬프트나 역할에 의존하지 않고 더 풍부한 가치 표현을 가능하게 합니다.

- **Performance Highlights**: ValuesRAG는 다양한 기준선 방법들과 비교했을 때 문화적 및 맥락적 이해 측면에서 유의미한 향상을 보여주었습니다. 특히, 단순한 값 요약만 제공된 상태에서도 우수한 성과를 나타내며, 문화적으로 균형 잡힌 AI 시스템을 육성하고 AI 응용 프로그램의 포용성을 증진시키는 잠재력을 가지고 있음을 입증했습니다. 이 연구는 ValuesRAG가 정책 입안자와 다양한 분야의 과학자들이 사회 시뮬레이션을 개선하고 공정하고 효과적인 정책을 수립하는 데 에 기여할 수 있음을 제안합니다.



### Reasoning based on symbolic and parametric knowledge bases: a survey (https://arxiv.org/abs/2501.01030)
- **What's New**: 이 논문은 인간의 지능에서 필수적인 추론(reasoning)의 접근 방식에 대한 새로운 관점을 제시합니다. 기존의 연구들은 지식 기반(knowledge base)에 기반한 추론 방법을 체계적으로 분석하지 않았으나, 이 연구는 이를 상징적(symbolic) 및 매개변수적(parametric) 지식 기반으로 분류하여 그 차이를 설명합니다. 이러한 접근을 통해 추론 방법의 새로운 이해와 향후 연구 방향을 제시하고자 합니다.

- **Technical Details**: 연구는 지식 기반을 두 가지 유형으로 분류하였습니다. 상징적 지식 기반은 인간이 이해할 수 있는 기호로 정보를 표현하고, 매개변수적 지식 기반은 매개변수 내에 지식을 암묵적으로 인코딩합니다. 이러한 접근은 추론 방법에 사용되는 지식의 저장 방식과 적용 시나리오를 명확히 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 이 논문은 상징적 및 매개변수적 지식 기반을 기반으로 한 다양한 추론 방법들을 자세히 조사합니다. 특히, 이들은 다양한 실제 응용 프로그램에서의 성능 개선에 기여할 수 있는 중요한 요소들입니다. 추론의 도전 과제와 미래의 연구 방향도 체계적으로 정리하였습니다.



### KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Mod (https://arxiv.org/abs/2501.01028)
Comments:
          Technical Report. 23 pages, 6 figures, 10 tables

- **What's New**: 이번 논문에서는 KaLM-Embedding이라는 다국어 텍스트 임베딩 모델을 소개합니다. 이 모델은 더 깨끗하고 다양한 도메인 특화 훈련 데이터를 활용하여 훈련되었습니다. 기존의 일반 임베딩 모델들은 훈련 데이터 품질을 간과하는 경우가 많았으나, KaLM-Embedding은 이를 최적화하여 성능을 배가시키고자 하였습니다.

- **Technical Details**: KaLM-Embedding은 Qwen2-0.5B를 기초로 한 모델로, 총 20개 범주의 데이터를 수집하여 사전 훈련하였으며, 70개 범주에 걸친 데이터를 통해 미세 조정이 이루어졌습니다. 이 과정에서 생성된 데이터의 품질을 높이기 위해 persona-based synthetic data와 ranking consistency filtering 기술이 적용되었습니다. 또한, semi-homogeneous task batch sampling 접근 방식을 통해 훈련 효율성을 증진시켰습니다.

- **Performance Highlights**: MTEB 벤치마크에 대한 광범위한 평가 결과, KaLM-Embedding은 1억 미만의 파라미터를 가진 모델 중에서 최고 성능을 기록하였습니다. 이로써 다국어 임베딩 모델로서 새로운 기준을 설정하며, 모델의 규모에 비해 우수한 다국어 성능을 발휘합니다. 특히, 중국어 및 영어 외의 다른 언어에서도 만족스러운 성능을 보여주어 다국어 임베딩 우수성을 입증하였습니다.



### MDSF: Context-Aware Multi-Dimensional Data Storytelling Framework based on Large language Mod (https://arxiv.org/abs/2501.01014)
- **What's New**: 본 논문에서는 데이터 분석 및 스토리텔링을 자동화하기 위한 다차원 데이터 스토리텔링 프레임워크(MDSF)를 소개합니다. MDSF는 대형 언어 모델(LLMs)을 기반으로 하여, 자동적으로 인사이트를 생성하고 맥락에 맞는 스토리를 제공하는 기능을 제공합니다. 이 프레임워크는 고급 전처리 기술, 증강 분석 알고리즘, 그리고 실행 가능한 인사이트를 식별하고 우선순위를 매기는 독특한 점수 매커니즘을 통합하고 있습니다.

- **Technical Details**: MDSF는 숫자형, 범주형, 시계열, 공간 데이터 등 다양한 다차원 데이터를 처리 및 통합할 수 있는 기능을 갖추고 있습니다. 이를 통해 MDSF는 더 포괄적이고 깊이 있는 데이터 내러티브를 생성할 수 있으며, LLM의 강력한 자연어 처리 및 생성 능력을 활용하여 매력적이고 일관된 이야기를 자동으로 생성하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 평가 결과, MDSF는 기존 방법들에 비해 인사이트 랭킹 정확도, 설명적 품질, 내러티브 일관성 측면에서 뛰어난 성능을 보였습니다. 사용자 연구 또한 MDSF의 실질적인 유용성을 강조하며, 콘텐츠 구조 강화, 결론 추출 및 세부 사항의 풍부함을 개선하는 데 기여했습니다.



### Exploring Information Processing in Large Language Models: Insights from Information Bottleneck Theory (https://arxiv.org/abs/2501.00999)
Comments:
          9 pages, 9 figures, 3 tables

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 정보 처리 메커니즘을 Information Bottleneck Theory의 관점에서 탐구합니다. 기존의 연구와 차별화되는 점은 비훈련 기반(task space detection) 접근 방식을 사용하여 LLM의 내부 정보 흐름을 추적하고, 특정 작업 공간으로의 입력 정보 압축 및 예측시 관련 정보를 추출하는 과정을 밝힌 것입니다. 이를 통해 LLM의 예측 성능과 추론 과정을 개선할 수 있는 두 가지 새로운 방법인 정보 압축 기반의 맥락 학습(IC-ICL)과 작업 공간 안내 미세 조정(Task-Space-guided Fine-Tuning, TS-FT)을 제안합니다.

- **Technical Details**: 논문에서는 LLM이 정보를 압축하고 풀이하는 과정을 명확히 하기 위해 작업 공간(task space)을 구성하고 이를 기반으로 LLM 내부의 정보 흐름을 탐지합니다. 각 작업 공간은 해당 작업을 가장 잘 나타내는 기본 벡터로 구성되며, 감정 분류 같은 특정 태스크를 예로 들어 이러한 벡터가 어떻게 형성되는지를 설명했습니다. 또한, PCA(Principal Component Analysis) 기법을 통해 작업 공간 내에서 노이즈를 제거한 후, 정보의 압축과 풀이 메커니즘이 어떻게 LLM의 성능에 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면, IC-ICL은 추론 속도를 40% 이상 가속화하며, LLM의 예측 정확성을 크게 향상시킨 것으로 나타났습니다. 또한, TS-FT는 복잡한 조정 없이도 모델 성능을 개선시키는 데 효과적입니다. 이러한 성과는 LLM이 작업 공간 내에서 정보를 효과적으로 처리함으로써 이루어진 결과로, 연구진은 제안한 방법들이 여러 데이터셋에서 유효성을 입증했다고 강조합니다.



### Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practic (https://arxiv.org/abs/2501.00982)
- **What's New**: 이 연구에서는 심리적 질문지를 보완하기 위해 소셜 미디어 게시물을 분석하는 새로운 적응형 Retrieval-Augmented Generation (RAG) 접근 방식을 제안합니다. 사용자의 게시물 데이터에서 가장 관련 있는 내용을 추출하여, 대형 언어 모델(LLMs)을 사용해 질문지 점수를 예측하는 방법입니다. 이 방법은 Reddit 기반 벤치마크 데이터 세트에서 기존의 최첨단 모델과 유사하거나 우수한 성능을 나타내며, 훈련 데이터에 의존하지 않고도 정신 건강 평가를 가능하게 합니다.

- **Technical Details**: 연구는 Beck Depression Inventory II (BDI-II)와 Self-Harm Inventory (SHI)라는 두 가지 표준화된 심리적 질문지를 중심으로 진행됩니다. 또한, 연구진은 개별 사용자의 Reddit 게시물 이력을 분석하여 질문 항목에 대한 응답을 예측하는 적응형 RAG 접근 방식을 구현했습니다. 이는 심리적 질문지에 대한 응답을 보다 정확하게 추정하기 위한 것으로, 데이터 접근성의 향상과 함께 기계 학습 기술을 활용합니다.

- **Performance Highlights**: 최신 연구 결과에 따르면, 제안된 접근 방법은 비지도 학습 환경에서도 효과적으로 작동할 수 있다는 것을 보였습니다. eRisk 데이터 세트를 사용한 실험에서, 이 방법은 기존의 감독 기반 모델과 비교하여 동등하거나 더 나은 성능을 보였습니다. 이에 따라, 연구의 성과는 새로운 행동 장애 예측을 위한 해석 가능하고 비지도 학습 가능한 방법론으로 확장될 수 있는 가능성을 보여줍니다.



### Incremental Dialogue Management: Survey, Discussion, and Implications for HRI (https://arxiv.org/abs/2501.00953)
Comments:
          16 pages

- **What's New**: 이 논문에서는 로봇의 대화 시스템이 인간처럼 언어를 처리할 수 있는 방법으로, 특히 단어 단위로 작동하는 점진적인 시스템(incremental systems)의 필요성을 강조합니다. 기존의 대화 시스템들은 문장 수준의 입력만을 처리하며, 이는 인간과 로봇의 자연스러운 상호작용을 방해합니다. 연구자들은 로봇의 대화 관리(Dialogue Management) 부문에서의 연구 부족을 지적하고, 이를 보완하기 위한 요구 사항을 제시합니다.

- **Technical Details**: 점진적인 음성 대화 시스템(incremental spoken dialogue systems, sdss)은 단어 단위로 입력 및 출력을 처리할 수 있습니다. 이 시스템은 주로 다음과 같은 모듈들로 구성됩니다: 자동 음성 인식(ASR), 자연어 이해(NLU), 대화 관리(DM), 자연어 생성(NLG), 음성 변환(Text-to-Speech, TTS). 논문에서는 이러한 모듈들이 단어 수준에서 상호작용해야 함을 강조하며, 이를 통해 시스템의 성능을 향상시킬 수 있음을 설명합니다.

- **Performance Highlights**: 연구에 따르면, 점진적인 시스템은 비점진적 시스템에 비해 자연스러운 상호작용을 제공하며, 사용자들로부터 긍정적인 피드백을 받았습니다. 점진적인 대화 관리가 효과적으로 성능을 개선할 수 있다는 결과도 언급됩니다. 이러한 연구 결과는 로봇의 대화 시스템이 보다 인간과 유사하게 작동하고, 높은 수준의 응답성을 제공할 수 있는 가능성을 제시합니다.



### Unfolding the Headline: Iterative Self-Questioning for News Retrieval and Timeline Summarization (https://arxiv.org/abs/2501.00888)
- **What's New**: 이 논문에서는 CHRONOS라는 새로운 접근법을 제안하며, 이는 Open-domain News Timeline Summarization(오픈 도메인 뉴스 타임라인 요약) 문제를 해결하기 위한 것입니다. CHRONOS는 대형 언어 모델(LLMs)을 활용하여 사건 간의 인과관계를 기반으로 사건을 정리할 수 있는 능력을 가지고 있습니다. 기존의 데이터셋인 Open-TLS를 통해 다양한 최근 뉴스 주제를 포괄하는 타임라인을 제공합니다.

- **Technical Details**: CHRONOS의 핵심 기능은 Iterative Self-Questioning 기법을 사용하여 관련 뉴스 기사 검색을 지원하고, 각 검색 라운드에서 타임라인을 생성하는 것입니다. 이 과정에서 5W1H 질문(What, Who, Why, Where, When, How)을 만들어 정보를 포괄적으로 수집하고, 이를 통해 새로운 질문을 작성하여 후속 검색을 수행합니다. 정보 검색과 요약 생성 과정에서 Retrieval-Augmented Generation(RAG) 프레임워크를 사용하여 LLM의 성능을 보완합니다.

- **Performance Highlights**: 실험 결과, CHRONOS는 오픈 도메인 타임라인 요약에서 뛰어난 성능을 보이며 기존의 폐쇄 도메인 시스템과 비슷한 수준의 결과를 달성하고 있습니다. 특히, Open-TLS 데이터셋을 통해 얻은 결과는 기존 공공 데이터셋에 비해 크기와 주제 다양성 측면에서 우수함을 나타냅니다. CHRONOS는 또한 효율성과 확장성에서 중요한 향상을 이루었습니다.



### Representation in large language models (https://arxiv.org/abs/2501.00885)
Comments:
          Draft of paper under review. 27 pages, 2 figures

- **What's New**: 본 논문은 최근의 대형 언어 모델(Large Language Models, LLMs)이 어떻게 작동하는지에 대한 근본적인 이론적 질문에 대해 다룹니다. 특히 LLM의 행동이 생물학적 인지에 연관된 정보 처리 방식에 의해 부분적으로 구동되는지, 아니면 기억화(memorization)와 임의의 테이블 조회(stochastic table look-up)에 의해 완전히 구동되는지에 대한 논의를 진행합니다. 이 논문은 LLM의 작동 방식에 대한 새로운 이해를 제공하고, 여러 이론 간의 불일치를 해소하기 위한 기반을 마련합니다.

- **Technical Details**: 저자는 LLM 행동이 부분적으로는 표현 기반 정보 처리(representation-based information processing)에 의해 영향을 받는다고 주장하며, 이에 대한 여러 연구 기법을 설명하고 방어합니다. 이 기법은 LLM이 구현하는 알고리즘의 종류와 관련이 있으며, 이러한 이해는 LLM이 신념(beliefs), 의도(intentions), 개념(concepts), 지식(knowledge), 이해(understanding)를 가질 수 있는지에 대한 높은 수준의 질문에 중대한 함의를 갖습니다.

- **Performance Highlights**: 이 연구는 LLM의 행동을 설명하기 위한 새로운 다양한 기법들을 제공합니다. 이러한 기법은 LLM의 기능에 대한 더 깊은 통찰을 제공하며, 이후 언어 모델과 그 후속 모델들에 대한 이론적 탐구의 기반을 제공합니다. 결과적으로, 이 논문은 LLM의 연구에 중요한 기여를 하며, 향후 모델 개발에 방향성을 제시합니다.



### TrustRAG: Enhancing Robustness and Trustworthiness in RAG (https://arxiv.org/abs/2501.00879)
- **What's New**: TrustRAG는 Retrieval-Augmented Generation(RAG) 시스템의 성능 저하 문제를 해결하기 위한 최초의 방어 프레임워크입니다. 기존의 공격과는 달리, TrustRAG는 K-means 클러스터링과 코사인 유사도를 사용하여 잠재적인 공격 패턴을 식별하고, 잘못된 정보가 포함된 문서를 필터링하는 데 초점을 맞추고 있습니다. 이를 통해 사용자가 요청한 정보에 대해 더 정확하고 신뢰성 있는 응답을 제공합니다.

- **Technical Details**: TrustRAG는 두 단계로 구성된 방어 메커니즘을 구현하며, 첫 번째 단계인 'Clean Retrieval'에서는 K-means clustering을 사용하여 문서의 임베딩 밀도 패턴을 기반으로 잠재적으로 악성인 콘텐츠를 차단합니다. 두 번째 단계인 'Conflict Removal'에서는 LLM의 내부 지식과 자기 평가를 활용하여 정직한 정보와 악의적인 정보 간의 불일치를 해결합니다. 이러한 과정은 언어 모델에 중간 훈련 과정 없이 결합될 수 있으며, 고진리성(contextual relevance)을 유지하고 강력한 방어를 제공합니다.

- **Performance Highlights**: TrustRAG의 실험 결과, 다양한 모델 아키텍처와 데이터셋에서 기존 접근 방식에 비해 검색 정확도, 효율성 및 공격 저항성이 실질적으로 향상된 것으로 나타났습니다. TrustRAG는 특히 지식 부패 공격에 대한 방어를 성공적으로 수행하며, 정밀한 응답 정확도를 유지합니다. 또한, 여러 지식 데이터베이스와 LLM에 대해 광범위한 평가를 수행하여 그 효과성을 입증하였습니다.



### LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models (https://arxiv.org/abs/2501.00874)
- **What's New**: 최근 대규모 언어 모델(LLMs)에 기반한 임베딩 모델의 발전이 여러 텍스트 임베딩 작업에서 새로운 최고 성능 기준을 세웠습니다. 그러나 이러한 모델은 주로 영어에 집중되어 있어 다국어 임베딩 기능은 거의 탐색되지 않았습니다. 이를 해결하기 위해 제안된 LUSIFER는 다국어 감독 없이 LLM 기반 임베딩 모델을 다국어 작업에 적응시키는 새로운 제로샷(zero-shot) 접근법입니다.

- **Technical Details**: LUSIFER의 구조는 다국어 인코더와 임베딩 특정 작업에 최적화된 LLM 기반 임베딩 모델로 구성됩니다. 이 두 구성 요소는 훈련 가능한 최소한의 매개변수를 통해 원활하게 통합되어, 다국어 인코더의 언어 이해 능력을 특화된 임베딩 모델로 효과적으로 전달합니다. LUSIFER는 14개 언어에 걸쳐 123개의 다양한 데이터셋을 포함하는 새로운 벤치마크를 도입하여 다국어 임베딩 성능을 평가합니다.

- **Performance Highlights**: 실험 결과 LUSIFER는 다양한 임베딩 작업에서 다국어 성능을 크게 향상시키며, 특히 중간 및 저자원 언어의 경우 22.15 포인트까지의 향상을 기록했습니다. LUSIFER는 영어 중심 모델보다 평균 5.75 개선된 성능을 보이며 다국어 표현 능력을 강화하는 데 효과적임을 입증했습니다. 이 연구는 다국어 감독 없이도 효과적인 다국어 표현 능력을 향상시키는 LUSIFER의 효용성을 보여줍니다.



### Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation (https://arxiv.org/abs/2501.00868)
Comments:
          Accepted at AAAI 2025. 13 pages, 7 tables, 10 figures

- **What's New**: 본 논문에서는 LLM(대규모 언어 모델)을 활용하여 동시 생성 모델의 정책 결정 문제를 해결하는 새로운 프레임워크인 LSG(LLM-driven Simultaneous Generation)를 제안합니다. 이는 기존의 인코더-디코더 아키텍처의 한계를 극복하며, LLM이 생성 속도와 품질을 동시에 고려하면서 출력 시점을 적절히 결정할 수 있도록 합니다. LSG는 기본 정책인 대기(wait) 정책을 기준으로 하여, 각 생성 단계에서 최소 지연 시간과 생성 품질의 균형을 맞추는 개선된 정책을 생성할 수 있게 합니다.

- **Technical Details**: LSG 프레임워크는 기본 대기 1 정책을 직접 비교하여 분포 차이를 분석하고, 이 차이가 사전 정해진 임계치를 초과할 때 출력을 생성하도록 LLM을 설정합니다. 이는 LLM이 이전의 정책 학습 없이도 실제 상황에서 동시 생성 작업을 수행할 수 있게 합니다. LSG는 최소한의 입력만으로도 제공되는 정보를 기반으로 하여, 생성의 타이밍을 효과적으로 결정할 수 있도록 방식과 구조를 재구성합니다.

- **Performance Highlights**: 실험 결과, LSG 메소드는 동시 번역 및 스트리밍 자동 음성 인식(task)에서 최첨단 성능을 달성하였으며, 실제 세계의 적용 가능성을 증명하였습니다. 개방형 소스 LLM을 활용하여 성능을 높이는 동시에, 메모리 소모와 학습 속도 문제를 최소화했습니다. 이러한 성과는 LSG가 동시 생성 작업에 효과적으로 통합될 수 있다는 것을 시사합니다.



### Negative to Positive Co-learning with Aggressive Modality Dropou (https://arxiv.org/abs/2501.00865)
- **What's New**: 이번 논문은 다중 모드(co-learning)의 효과를 증가시키기 위한 공격적인 모드 드롭아웃(aggressive modality dropout) 기법을 사용하여 부정적 공동 학습(NCL)을 긍정적 공동 학습(PCL)으로 전환하는 방법을 문서화하고 있습니다. 공격적인 모드 드롭아웃을 활용하면, 다중 모드 모델이 단일 모드 배치(unimodal deployment)에 대비하여 사전 준비(prep) 되며, NCL의 경우 실험에서 20%의 정확도 향상이 있었습니다. 또한, 이 기법을 PCL과 비교하여도 개선점을 보여줍니다.

- **Technical Details**: 실험에서는 양방향 조기 융합 장기 단기 메모리(bidirectional early fusion long short term memory, bi-EFLSTM)와 메모리 융합 네트워크(Memory Fusion Network, MFN)를 사용하였습니다. bi-EFLSTM은 앞으로와 뒤로 시간을 이동하는 두 개의 LSTM 셀로 구성되어 있으며, MFN은 각각의 모드를 위한 LSTM 시스템, 델타 메모리 주의 네트워크(Delta-memory Attention Network, DMAN), 그리고 다중 뷰 게이티드 메모리(Multi-view Gated Memory)로 구성되어 있습니다. 가벼운 학습 손실(Loss function)을 사용하여 각각의 네트워크에서 최적화를 수행하였습니다.

- **Performance Highlights**: 실험은 IEMOCAP 및 MOSI 데이터 세트를 사용하였으며, 이 중 MOSI는 P2FA를 통해 강제 정렬(force aligned)되었습니다. IEMOCAP 세트는 수동으로 전처리되었고, 각 모드는 Sphinx III를 활용하여 단어 수준으로 강제 정렬되었습니다. 이 과정에서, 비대칭 데이터 길이를 고려하여 특정 조건을 충족하는 문장들만 포함하여 모델의 성능을 개선하였습니다.



### DiffETM: Diffusion Process Enhanced Embedded Topic Mod (https://arxiv.org/abs/2501.00862)
Comments:
          5 pages, 2 figures, Accepted by ICASSP 2025

- **What's New**: 이 논문은 기존의 embedded topic model (ETM)에서 문서-주제 분포의 로지스틱 정규 분포 가정으로 인해 발생하는 성능 한계를 극복하기 위한 새로운 방법을 제안합니다. 제안된 방법은 확산 과정 (diffusion process)을 샘플링 과정에 통합하여 문서-주제 분포를 모델링하고 최적화 과정을 쉽게 유지할 수 있게 합니다. 우리가 제안한 모델은 두 개의 주요 데이터셋에서 주제 모델링 성능을 향상시키는 데 효과적임을 입증하였습니다.

- **Technical Details**: 이 모델에서는 문서 표현에서 직접 샘플링을 수행하여 고유한 문서-주제 분포를 생성합니다. 기존의 ETM과 비교할 때, 이러한 시도는 문서 정보가 포함된 숨겨진 표현을 통합하여 더 나은 모델링을 가능하게 합니다. 모델은 세 가지 메트릭인 주제 일관성 (topic coherence), 주제 다양성 (topic diversity), 그리고 혼잡도(perplexity) 기준에서 성능 향상을 달성합니다.

- **Performance Highlights**: 제안된 모델은 20Newsgroup 및 New York Times 데이터셋에서 기본 및 최첨단 ETM과 비교할 때 유의미한 성과를 나타냈습니다. 이 연구는 확산 과정을 ETM에 통합하여 문서-주제 분포의 표현 능력을 향상시키는 첫 번째 시도로, 새로운 접근법이 기여할 수 있는 잠재력을 보여줍니다. 결과적으로, 새로운 기술이 주제 모델링에서의 성능을 향상시키는 데 기여하고 있음을 확인했습니다.



### LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions (https://arxiv.org/abs/2501.00830)
Comments:
          42 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 복잡한 행위 추론 작업에서 한계를 겪고 있다는 문제점을 지적합니다. 이에 대응하기 위해, 저자들은 LLM의 자연어 이해 능력과 상징적 추론 능력을 갖춘 행동 언어(action languages)의 강점을 결합한 새로운 방법인 'LLM+AL'을 제안합니다. 이 접근법은 LLM의 의미적 구문 분석(semantic parsing) 및 상식 지식 생성(common sense knowledge generation) 능력을 활용하여 보다 정교한 자동화된 추론을 가능하게 합니다.

- **Technical Details**: LLM+AL 접근법은 주어진 자연어의 추론 문제에 대해 프로그램 서명을 생성하고, 해당 지식을 바탕으로 설정된 규칙을 행동 언어의 구문과 의미에 맞게 변환합니다. 이 과정은 ℬ⁢𝒞+한계로부터ℬ𝒞{
cal{BC}}의 규칙을 생성하여 자동화된 추론기(automated reasoner)와의 통합된 파이프라인을 통해 이루어집니다. 또한, 이 방법은 여러 단계를 통해 LLM을 효과적으로 활용하고, 최종적으로는 수동적인 수정이 최소화된 정확한 규칙을 생성합니다.

- **Performance Highlights**: 연구 결과 LLM+AL은 복잡한 상황에서도 상대적으로 적은 수의 수동 수정을 통해 정확한 답변을 도출하며, 단독 LLM보다 우수한 성능을 보였습니다. 특히, 여러 차례의 인간 피드백에도 불구하고 기존 LLM들은 정답을 생성하는 데 어려움을 겪는 반면, LLM+AL은 적은 수정으로 올바른 해답을 제공합니다. 이는 LLM과 행동 언어의 통합 방식이 더 견고하고 적응 가능한 AI 시스템을 구축하는 데 기여할 수 있음을 보여줍니다.



### Embedding Style Beyond Topics: Analyzing Dispersion Effects Across Different Language Models (https://arxiv.org/abs/2501.00828)
Comments:
          To appear in the Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Abu Dhabi

- **What's New**: 본 논문은 다양한 최신 언어 모델에서 쓰기 스타일이 임베딩 벡터(embedding vectors)의 분산에 미치는 영향을 분석합니다. 초기 트랜스포머 모델은 주로 주제 모델링(topic modeling)과 일치했으나, 이 연구는 쓰기 스타일이 임베딩 공간을 형성하는 역할을 조사합니다. 저자들은 프랑스어와 영어의 문학 코퍼스를 사용하여 언어 모델의 민감성을 비교하고, 스타일 정보가 언어 모델 처리에 미치는 영향을 밝히고자 합니다.

- **Technical Details**: 연구는 두 가지 문학 작품인 레이몽 크뇌의 'Exercices de Style'와 펠릭스 페넥의 'Nouvelles en trois lignes'를 원재료로 사용하여 작성되었습니다. 이러한 데이터 세트는 주제와 스타일 차원을 체계적으로 교환하는 실험적 연구를 통해 생성되었습니다. 결과적으로, 스텝 성과를 차별화하여 각 언어 모델에서 스타일과 주제가 임베딩의 공간 분산에 미치는 상대적인 영향을 평가할 수 있었습니다.

- **Performance Highlights**: 실험 결과, 쓰기 스타일이 임베딩 분산에 미치는 영향이 주제보다 더 주목할 만한 것으로 나타났습니다. 이를 통해 현대 대규모 언어 모델들이 쓰기 스타일에 대한 세밀한 정보를 어떻게 처리하는지를 더 잘 이해할 수 있었습니다. 또한, 향후 연구 방향으로는 임베딩 벡터에서 스타일 표현을 정확히 포착하기 위한 방법론을 제안하고 있습니다.



### Reasoning-Oriented and Analogy-Based Methods for Locating and Editing in Zero-Shot Event-Relational Reasoning (https://arxiv.org/abs/2501.00803)
- **What's New**: 본 논문에서는 이벤트 관계 추론을 위한 새로운 접근법인 Reasoning-Oriented Locating and Editing (ROLE)와 Analogy-Based Locating and Editing (ABLE)을 제안합니다. ROLE은 언어 모델의 핵심 모듈을 찾고 편집하여 해석 가능성을 높이고 리소스를 효율적으로 최적화합니다. ABLE은 다양한 작업 간의 유사성을 활용하여 제로샷 추론 능력을 향상시키는 방법으로 인식됩니다.

- **Technical Details**: ROLE의 핵심 모듈 식별을 위해 평균 간접 효과를 계산하고, 이를 기반으로 키 모듈의 변화 정도를 최적화하여 사고 성능을 향상시킵니다. ABLE은 작업 간 유사성과 차이를 학습하여 지식을 효율적으로 이전하고, 제로샷 추론에서 뛰어난 결과를 달성합니다. 두 방법 모두 Flan-t5-large를 기반 모델로 활용하여 운영됩니다.

- **Performance Highlights**: ROLE은 예상보다 낮은 계산 비용으로 해석성과 추론 성능을 개선했으며, ABLE은 다양한 데이터 세트에서 제로샷 이벤트 관계 추론의 SOTA 성과를 달성했습니다. 이러한 방법들은 기존 접근법보다 더 뛰어난 성능을 제공하며, 특히 대규모 언어 모델의 해석 가능성을 높여줍니다.



### Navigating Nuance: In Quest for Political Truth (https://arxiv.org/abs/2501.00782)
Comments:
          Accepted at JCDL 2024

- **What's New**: 이 연구는 정치적 편향을 탐지하기 위한 새로운 접근법인 Llama-3 (70B) 언어 모델을 Media Bias Identification Benchmark (MBIB)에서 평가했습니다. 특히, subtle reasoning을 반영한 새로운 prompting 기법을 도입하여 정치적 편향을 더욱 효과적으로 식별할 수 있는 가능성을 보여줍니다. 연구 결과, 본 프레임워크는 기존의 최첨단 모델인 ConvBERT와 비교할 만한 성능을 보여주며, 정보의 왜곡과 정치적 양극화를 완화하는 도구 개발에 기여하고자 합니다.

- **Technical Details**: 정치적 편향 탐지는 자연어 처리(NLP) 분야에서 중요한 연구 영역으로, 현재 심층 학습과 대형 언어 모델(LLMs)을 활용한 최근 기법들이 개발되고 있습니다. 본 연구에서 사용된 Llama-3 모델은 다양한 벤치마크에서 성능이 입증되었으며, 특히 복잡한 추론 및 이해 작업에서 다른 모델들을 능가합니다. 실험에는 zero-shot, few-shot prompting과 함께 Chain-of-Thought (CoT) prompting을 활용하여 모델의 성능을 극대화했습니다.

- **Performance Highlights**: 조사 결과, Llama-3는 기존의 모델들과 견주어 높은 정확도로 정치적 편향을 탐지할 수 있었으며, 이는 다양한 데이터셋과 정치적 맥락에서도 일반화 가능한 성능을 가지는 것으로 나타났습니다. 또한, CoT prompting을 통해 LLM의 해석 가능성과 정확성을 높이는 데 성공하며, 복잡한 작업에서의 신뢰성을 확보했습니다. 최종적으로, 본 연구는 자동화된 편향 탐지 솔루션 개발의 새로운 가능성을 제시합니다.



### Decoding the Flow: CauseMotion for Emotional Causality Analysis in Long-form Conversations (https://arxiv.org/abs/2501.00778)
Comments:
          7pages

- **What's New**: 이 논문은 긴 시퀀스의 감정 인과 추론을 다루기 위해 CauseMotion이라는 새로운 프레임워크를 제안합니다. CauseMotion은 Retrieval-Augmented Generation(RAG)와 다중 모드 융합(multimodal fusion)을 기반으로 하여, 복잡한 감정적 인과 관계를 효과적으로 추적할 수 있도록 고안되었습니다. 기존 모델이 다루기 어려운 긴 대화 체인을 통해 감정의 원인과 결과를 명확히 이해할 수 있게 도와줍니다.

- **Technical Details**: CauseMotion은 텍스트 정보 외에도 음성에서 유도된 감정 관련 특성을 통합함으로써 의미론적 표현을 풍부하게 만듭니다. 이 모델은 RAG와 슬라이딩 윈도우 메커니즘을 활용하여 대화의 맥락에 맞는 세그먼트를 효과적으로 검색하고 활용합니다. 이러한 접근 방식을 통해 다중 대화 턴에 걸친 복잡한 감정 인과 체인을 추론하는 것이 가능해집니다.

- **Performance Highlights**: 실험 결과, CauseMotion을 통합한 GLM-4 모델이 원본 모델보다 8.7% 향상된 인과 정확도를 보였으며, GPT-4o 대비 1.2% 더 우수한 성능을 기록했습니다. 공개적으로 사용 가능한 DiaASQ 데이터세트에서 CauseMotion-GLM-4는 정확도, F1 점수, 인과 추론 정확도에서 최고 성능을 달성했습니다.



### FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation (https://arxiv.org/abs/2501.00777)
Comments:
          In submission

- **What's New**: 이 논문은 NLP 및 XAI 분야에서 카운터팩추얼(counterfactual) 예제를 생성하는 새로운 방법을 제안합니다. ZeroCF라는 접근 방식을 통해, 피처 애트리뷰션(feature attribution) 방법으로부터 추출된 중요한 단어를 활용하여 제로샷(zero-shot) 환경에서 카운터팩추얼 예제를 생성합니다. 또 다른 핵심 구성요소인 FitCF 프레임워크는 레이블 플립 검증(label flip verification)을 통해 카운터팩추얼의 질을 개선하고, 이를 몇 초 프로밍(few-shot prompting)의 시연으로 사용하여 기존의 두 가지 최신 기법을 초월합니다.

- **Technical Details**: ZeroCF는 특정 데이터셋에 맞게 조정된 BERT 모델을 활용하여 피처 애트리뷰션 점수에 기반하여 카운터팩추얼을 생성합니다. 이 과정에서 주어진 입력에 대한 특정 피처의 중요성을 평가하는 여러 방법을 사용합니다. FitCF는 ZeroCF에서 생성된 카운터팩추얼을 이용하고, 이를 레이블 플립 검증을 통해 확인한 후, 인간이 제작한 예제에 의존하지 않고 몇 초 프로그래밍에 사용할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, ZeroCF와 FitCF가 Polyjuice 및 FIZLE 모델의 성능을 초과하는 것으로 나타났습니다. 이 과정에서 레이블 플립 비율, 유창성(fluency), 편집 거리(edit distance) 등 세 가지 자동 평가 지표가 활용되었습니다. 특히, FitCF는 다양한 피처 애트리뷰션 방법, 특히 LIME 및 SHAP와 결합 시 더욱 높은 품질을 보였으며, 시연 수가 성능에 미치는 영향이 가장 컸습니다.



### Enhancing Transformers for Generalizable First-Order Logical Entailmen (https://arxiv.org/abs/2501.00759)
Comments:
          17 pages

- **What's New**: 이번 연구에서는 transformers의 일반화 가능한 1차 논리 추론 능력을 조사하고, 이를 개선할 수 있는 방법을 모색합니다. 기존의 연구들과의 차별점은, 모델 안에서 파라미터화된 지식을 바탕으로 추론을 수행한다는 것입니다. 또한, 기존의 데이터 분포와의 연결성을 더 정교하게 설정하여, 새로운 유형의 쿼리와 지식이론의 관점을 통해 1차 논리 추론 작업의 성능을 분석합니다.

- **Technical Details**: 이 논문은 Knowledge Graphs(KG)를 사용하여 1차 논리 함의(first-order logical entailment)를 연구합니다. 연구는 복잡한 쿼리 문제를 해결하기 위한 필수적인 과정으로서의 1차 논리 함의를 제시하는 데 중점을 두고 있습니다. 논리 쿼리를 식별하는 과정에서 추론 과정이 어떻게 이루어지는지를 관찰하고, 다양한 쿼리 유형에 대한 논리적 추론의 성능을 평가합니다.

- **Performance Highlights**: 우리는 transformers가 논리 함의 문제에서 현존하는 방법들보다도 뛰어난 성능을 발휘함을 발견했습니다. 특히, 상대 위치 인코딩(relative positional encoding, RPE)이 전통적인 절대 위치 인코딩(absolute positional encoding, APE)보다 우수한 성능을 보였습니다. TEGA(Transformer Encoder with Guided Attention)라는 새로운 아키텍처를 제안하여, 이러한 성능을 더욱 향상시킬 수 있음을 입증했습니다.



### DIVE: Diversified Iterative Self-Improvemen (https://arxiv.org/abs/2501.00747)
- **What's New**: 새로운 연구에서는 DIVE(Diversified Iterative Self-Improvement)라는 프레임워크를 제안하여 반복적인 자기 개선(Iterative Self-Improvement) 기술의 한계를 극복하고자 합니다. 특히, 자기 생성 데이터에 대한 지속적인 훈련이 출력 다변성(output diversity)을 줄이는 문제를 해결하는 데 중점을 두었습니다. DIVE는 샘플 풀 확장(Sample Pool Expansion)과 데이터 선택(Data Selection)이라는 두 가지 핵심 요소를 통해 다양한 해결책을 탐색하고 품질과 다양성 간의 균형을 맞추는 전략을 제공합니다.

- **Technical Details**: DIVE는 두 가지 보완 전략을 통해 작동합니다. 첫 번째는 다양한 응답을 샘플링하여 각 반복에서 더 넓은 잠재적 솔루션 세트를 탐색하는 샘플 풀 확장(Sample Pool Expansion)입니다. 두 번째는 품질을 필터링하기 위해 아웃라이어 감지 기법을 사용하고, 선호 쌍에서의 다양성을 극대화하기 위한 탐욕적 선택 알고리즘(greedy selection algorithms)을 활용하는 데이터 선택(Data Selection)입니다.

- **Performance Highlights**: 실험 결과, DIVE는 MATH와 GSM8k 데이터셋에서 기존의 ISI 방법과 비교해 출력 다변성 지표가 10%에서 45%까지 상대적으로 증가한 것을 보여줍니다. 이 과정에서 출력 품질은 유지되었습니다. 또한, 아블레이션 연구를 통해 샘플 풀 확장 및 데이터 선택 두 요소가 이러한 개선을 달성하는 데 중요한 역할을 함을 확인했습니다.



### Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines (https://arxiv.org/abs/2501.00745)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 검색 엔진의 정보 검색 환경의 변화를 다룬다. 특히, LLM이 검색 엔진에 통합됨에 따라 발생하는 새로운 취약성과 공격, 특히 랭킹 조작 공격에 대한 연구를 진행한다. 이를 통해 공격자들이 웹페이지 콘텐츠를 조작하여 순위를 조작하는 방법과 그로 인해 발생하는 비즈니스 상의 불공정한 이점에 대해 논의한다.

- **Technical Details**: 이 연구에서는 랭킹 조작 공격의 동학을 무한 반복 죄수의 딜레마(Infinitely Repeated Prisoners' Dilemma)라는 게임 이론의 관점에서 분석한다. 공격자들은 전략적으로 협력할지 공격할지를 결정하며, 이 과정에서 공격 비용, 공격 성공률, 할인율 등 여러 요소가 플레이어 행동에 미치는 영향을 살펴본다. 협력이 지속될 수 있는 조건을 식별하고, 적응형 보안 전략과 생태계 설계의 중요성을 강조한다.

- **Performance Highlights**: 연구 결과, 협력이 지속될 가능성이 높아지는 경계 조건과 각 요인이 협력 장기 지속성에 미치는 영향을 보여준다. 흥미롭게도, 공격 성공률이 중간 수준일 때 공격의 이익과 위험이 최적화되어 협력을 저해할 수 있음을 발견했다. 또한, 방어 조치를 통해 공격 성공률을 제한하려는 시도가 오히려 공격 유인을 증가시킬 수 있다는 점도 강조한다.



### On Importance of Layer Pruning for Smaller BERT Models and Low Resource Languages (https://arxiv.org/abs/2501.00733)
Comments:
          Accepted at FIRE 2024: 16th meeting of the Forum for Information Retrieval Evaluation

- **What's New**: 이번 연구는 레이어 프루닝(layer pruning)을 통해 저자원 언어에 맞춘 더 효율적인 BERT 모델을 개발하는 방법을 탐구합니다. 연구에서는 다양한 BERT 변형 모델을 실험하고, 중간 레이어 프루닝이 가장 효과적인 전략임을 확인했습니다. 이를 통해, 프루닝된 모델이 크기가 줄어들면서도 고성능을 유지할 수 있음을 입증하였습니다.

- **Technical Details**: 레이어 프루닝은 신경망에서 특정 부분을 선택적으로 제거하여 모델 크기와 계산 부담을 줄이는 기술입니다. 본 연구에서는 MahaBERT-v2와 Google-Muril 모델을 사용하여 짧은 헤드라인 분류(SHC), 긴 단락 분류(LPC), 긴 문서 분류(LDC)와 같은 다양한 데이터셋에 대해 모델의 성능을 평가했습니다. 실험 결과, 중간 레이어를 프루닝하는 것이 가장 효율적인 방법으로 나타났으며, 모델은 50%에서 80% 크기 감소를 달성했습니다.

- **Performance Highlights**: 프루닝된 모델은 같은 크기의 스크래치 모델보다 일관되게 더 나은 성능을 보였습니다. 특히, 중간 레이어 프루닝은 상단 및 하단 프루닝과 비교하여 경쟁력 있는 성적을 기록했습니다. 이 연구 결과는 저자원 언어에 대한 고성능 NLP 모델의 접근성을 높이는데 기여할 것입니다.



### eRevise+RF: A Writing Evaluation System for Assessing Student Essay Revisions and Providing Formative Feedback (https://arxiv.org/abs/2501.00715)
- **What's New**: 이번 논문에서는 학생들의 작문 수정을 지원하는 향상된 Automated Writing Evaluation (AWE) 시스템인 eRevise+RF를 도입합니다. 이 시스템은 학생들이 제공된 피드백을 바탕으로 에세이를 수정할 수 있도록 돕고, 수정의 효과를 평가합니다. 연구 결과, eRevise+RF는 학생들의 증거 사용에 대한 평가 및 반영된 수정 사항을 추출하는 데 효과적임을 확인하였습니다.

- **Technical Details**: eRevise+RF 시스템은 Natural Language Processing (NLP) 기술을 활용하여 학생의 작문을 평가하고 피드백을 제공합니다. 이 시스템은 Automated Essay Scoring (AES)과 Revision Feedback (RF) 두 가지 기능으로 구성되어 있으며, 이전의 연구에서 사용된 알고리즘을 기반으로 개발되었습니다. 특히, 시스템은 원본 및 수정된 초안을 비교하여 각 수정 사항이 에세이 개선에 얼마나 기여했는지를 평가합니다.

- **Performance Highlights**: eRevise+RF는 6명의 교사와 406명의 학생을 대상으로 한 연구에서 긍정적인 결과를 보였습니다. 시스템은 학생들이 에세이를 수정할 때 필요한 피드백을 제공하여, 논증적 작문 기술을 향상시키는 데 도움을 줍니다. 이 실험을 통해 학생들이 에세이를 개선하는 데 있어서 중요한 지원을 받을 수 있음을 보여주었습니다.



### CODEOFCONDUCT at Multilingual Counterspeech Generation: A Context-Aware Model for Robust Counterspeech Generation in Low-Resource Languages (https://arxiv.org/abs/2501.00713)
Comments:
          to be published in MCG-COLING's 2025 conference proceedings

- **What's New**: 이 논문은 MCG-COLING-2025 공동 작업에서 튼튼한 반대 발언 생성(counterspeech generation)을 위한 컨텍스트 인식 모델을 소개합니다. 이 모델은 특히 저자원 언어 설정에서 두드러진 성과를 보였으며, 여러 언어 데이터세트를 활용한 시뮬레이티드 어닐링(simulated annealing) 알고리즘으로 사실적이고 정확한 반응을 생성합니다. 연구 결과, 바스크어(Basque)의 1위, 이탈리아어의 2위, 영어와 스페인어의 3위를 차지하여 저자원 언어 시나리오에서의 효과성을 강조합니다.

- **Technical Details**: 모델은 세 단계의 접근 방식을 사용하여 다양한 언어에서 효과적인 카운터스피치를 생성합니다. 첫 단계에서는 LLM(대형 언어 모델)과 함께 시뮬레이티드 어닐링 기법을 사용해 다양한 응답을 생성하고 평가합니다. 두 번째 단계에서는 라운드 로빈(tournament) 평가 시스템을 도입하여 가장 효과적인 응답을 순위 정렬하고, 마지막 단계에서는 상위 네 개의 응답을 결합한 후 평가 스크립트를 실행하여 최상위 결과를 도출합니다.

- **Performance Highlights**: 이 모델은 바스크어에서 반대 발언 생성과 관련하여 우수한 결과를 기록하였습니다. 평가 메트릭으로는 BLEU, ROUGE, BERTScore, Novelty 외에도 JudgeLM 기반의 평가 방법이 사용되었습니다. 실험 결과는 모델의 성능을 상세하게 분석하였으며, 반대 발언 품질을 개선하기 위한 배경 지식 통합의 중요성을 부각시켰습니다.



### Rethinking Addressing in Language Models via Contexualized Equivariant Positional Encoding (https://arxiv.org/abs/2501.00712)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 기존의 position-based addressing 기술의 한계를 극복하기 위해 새로운 프레임워크인 Contextualized Equivariant Positional Embedding (TAPE)를 제안합니다. TAPE는 시퀀스의 내용을 여러 레이어에 걸쳐 통합함으로써 동적이고 맥락 인식적인 positional encoding을 제공하며, 전통적인 고정 패턴의 제약을 극복합니다. 이 방법은 다양한 작업에 대한 적응성과 견고성을 높이기 위해 permutation과 orthogonal equivariance를 적용합니다.

- **Technical Details**: TAPE는 기존의 벡터화된 positional embedding을 다차원 텐서로 확장하여 token feature와의 상호작용을 풍부하게 합니다. Attention 메커니즘에서 TAPE는 positional encodings 간의 쌍별 내적(pairwise inner product)을 포함하여 attention 값을 token 유사도뿐만 아니라 positional 근접성에 근거하여 계산합니다. 또한, orthogonal equivariance를 보존하며 positional encodings와 token features를 직접 혼합하는 MLP 레이어를 사용자 정의합니다.

- **Performance Highlights**: TAPE는 언어 모델링, 산술 추론, 긴 문맥 검색 작업에서 기존의 positional embedding 기술들보다 우수한 성능을 보여줍니다. 광범위한 실험을 통해 TAPE가 스크래치에서 훈련받고 파라미터 효율적인 미세 조정을 통해 최첨단 성능을 달성하며, 긴 시퀀스에 대한 perplexity 감소와 같은 주요 자연어 작업에서 성능을 초월하는 점을 보고합니다.



### PANDA -- Paired Anti-hate Narratives Dataset from Asia: Using an LLM-as-a-Judge to Create the First Chinese Counterspeech Datas (https://arxiv.org/abs/2501.00697)
Comments:
          to be published in MCG-COLING 2025's conference proceedings

- **What's New**: 이 논문은 중국에서의 혐오 발언(hate speech)을 근절하기 위한 현대 표준 중국어 카운터스피치(counterspeech) 코퍼스를 소개합니다. 이 코퍼스는 기존의 영어 중심 자료의 한계를 뛰어넘어 동아시아의 혐오 발언 대응 연구에 필요한 자원을 제공합니다. 또한 LLM(large language models)을 저자로 사용하고, 시뮬레이티드 어닐링(simulated annealing) 및 라운드로빈 알고리즘을 이용하여 카운터스피치를 생성하는 새로운 방법론을 제안합니다.

- **Technical Details**: 본 연구는 카운터스피치 데이터 세트의 구축을 위한 구체적인 방법론을 다룹니다. 이에는 데이터 수집, 전처리, 카운터스피치 생성 및 주석(annotation) 방법 등이 포함됩니다. LLM을 기반으로 한 JudgeLM과 BLEU, ROUGE-L, BERTScore와 같은 다양한 평가지표를 활용하여 생성된 카운터스피치의 품질을 평가하는 것도 특징입니다.

- **Performance Highlights**: 생성된 코퍼스는 중국어 혐오 발언 데이터의 부재를 극복하고, 앞으로의 카운터스피치 생성 및 평가 연구에 필수적인 기초 자료로 자리 잡을 것입니다. 연구 결과는 기존 한국어 카운터스피치 연구의 한계를 보완하며, 추후 자동화된 카운터스피치 생성의 가능성을 보여줍니다. 이로 인해 카운터스피치의 품질 및 효과성 평가에 있어 새로운 패러다임을 제시할 것으로 기대됩니다.



### Labels Generated by Large Language Model Helps Measuring People's Empathy in Vitro (https://arxiv.org/abs/2501.00691)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 새로운 응용 분야인 'in-vitro' 적용 가능성을 탐구하고 있습니다. 기존의 프로프트 엔지니어링 연구와는 달리, LLM을 활용하여 잘못된 라벨을 교정하고 훈련 데이터를 증강함으로써 주류 모델의 지도 학습을 지원하고자 합니다. 특히, 공감(compute empathy) 분야에서 LLM이 생성한 라벨을 통해 모델의 성능을 획기적으로 향상시켰습니다.

- **Technical Details**: 이 연구에서는 LLM을 활용해 RoBERTa와 같은 사전 학습된 언어 모델(PLMs)의 훈련을 개선하는 두 가지 방법을 제안합니다. 첫째, LLM을 사용하여 라벨 노이즈를 조정하는 것이고, 둘째, LLM 생성 라벨을 통해 훈련 데이터를 증강하여 PLMs의 훈련을 지원하는 것입니다. 이러한 접근 방식은 데이터 중심의 AI(data-centric AI) 접근 방식에 기초하여 훈련 데이터의 품질을 개선하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 방법은 신뢰할 수 있는 데이터 세트에서 통계적으로 의미 있는 성능 향상을 달성했으며, NewsEmp 벤치마크에서 0.648로 최신 상태의 Pearson 상관 계수를 기록했습니다. 또한, 효과적인 평가를 위해 Pearson 상관 계수(PCC)와 더불어 일치 상관 계수(CCC) 및 평균 제곱근 오차(RMSE)를 제안하여 기존의 평가 메트릭의 한계를 보완하고 있습니다.



### 2 OLMo 2 Furious (https://arxiv.org/abs/2501.00656)
Comments:
          Model demo available at this http URL

- **What's New**: OLMo 2는 향상된 아키텍처와 훈련 레시피를 갖춘 차세대 완전 개방형 언어 모델입니다. 새로운 Dolmino Mix 1124라는 특화된 데이터 혼합물이 도입되어 모델 성능을 크게 향상시킵니다. 또한, Tülu 3의 베스트 프랙티스를 통해 OLMo 2-Instruct가 개발되어 보다 개방적인 데이터 접근 방식을 강조합니다.

- **Technical Details**: 모델 아키텍처와 훈련 방법의 수정으로 인해 훈련 안정성이 향상되었으며, per-token 효율성도 개선되었습니다. Late-stage curriculum training을 통한 Dolmino Mix 1124의 도입은 많은 하위 작업 벤치마크에서 모델 능력을 크게 향상시킵니다. OLMo 2는 또한 RLVR(Verifiable Rewards)와 함께 강화학습을 통합하여 더 효과적인 훈련을 가능하게 합니다.

- **Performance Highlights**: OLMo 2 베이스 모델은 Llama 3.1 및 Qwen 2.5와 같은 오픈-웨이트 모델과 비교하여 성능이 유사하거나 초월하며, 더 적은 FLOPs로 훈련됩니다. 모든 OLMo 2 아티팩트는 개방적으로 배포되며, 7B 및 13B 규모로 훈련된 모델과 모든 훈련 데이터, 코드, 레시피, 로그 그리고 중간 체크포인트가 포함되어 있습니다. 최종 지침 모델은 Ai2 Playground에서 무료 연구 데모로 제공됩니다.



### Efficient Standardization of Clinical Notes using Large Language Models (https://arxiv.org/abs/2501.00644)
- **What's New**: 이 연구에서는 대형 언어 모델을 활용하여 1,618개의 임상 노트를 표준화하는 방법을 제시합니다. 표준화 과정에서는 평균적으로 4.9개의 문법 오류, 3.3개의 철자 오류, 비표준 용어를 표준 용어로 변환하고, 약어 및 두문자어를 각각 15.8개 확장했습니다. 이러한 방법은 임상 노트의 가독성과 일관성을 향상시키고, 상호 운용이 가능한 데이터 형식으로의 변환을 용이하게 합니다.

- **Technical Details**: 이 연구에서 GPT-4 API를 사용하여 임상 노트의 구조와 명확성을 개선하는 방법을 탐구했습니다. 임상 노트는 신경 면역 질환과 같은 진단을 포함한 비식별 데이터로 구성되었으며, 노트 표준화 과정에서는 약어 확장, 철자 및 문법 수정, 비표준 용어 교체 등을 수행했습니다. 각 노트는 JSON 형식으로 저장되어 통합 및 검색에 적합한 구조로 구성되었습니다.

- **Performance Highlights**: 전문가 리뷰 결과, 표준화 이후 중요한 데이터 손실이 발견되지 않았고, 이는 임상 노트의 질을 유지하면서도 가독성과 기능성을 향상시킬 수 있음을 보여줍니다. 이 연구는 임상 노트 표준화의 개념 증명을 제공하며, 향후 의사 결정 지원 및 연구에 기여할 수 있는 가능성을 제시합니다.



### Toward Corpus Size Requirements for Training and Evaluating Depression Risk Models Using Spoken Languag (https://arxiv.org/abs/2501.00617)
- **What's New**: 이번 연구는 정신 건강 리스크 예측 분야에서 훈련 및 테스트 세트 크기의 변동성이 모델 성능에 미치는 영향을 통제된 환경에서 분석하고 있습니다. 65,000개 이상의 레이블이 있는 데이터 포인트를 사용하여 다양한 훈련 및 테스트 세트 크기의 조합에서 도출된 결과를 제공합니다.

- **Technical Details**: 연구는 언어 기반 모델과 음성 음향 기반 모델 두 가지 유형을 포함하며, 각기 현재 사용되고 있는 방법론을 적용하였습니다. 또한, 연령 불일치가 있는 테스트 세트도 포함되어 실험되었습니다.

- **Performance Highlights**: 결과에 따르면, 1,000 샘플 이하의 테스트 세트 크기는 대규모 훈련 세트에서도 잡음이 섞인 결과를 발생시켰고, 안정적인 결과를 얻기 위해서는 최소 2,000 이상의 훈련 세트 크기가 필요했습니다. NLP와 음향 모델은 훈련/테스트 크기 변동에 대해 유사한 반응을 보였으며, 불일치 테스트 세트도 유사한 패턴을 보여주었습니다.



### Optimizing Speech-Input Length for Speaker-Independent Depression Classification (https://arxiv.org/abs/2501.00608)
- **What's New**: 이 논문은 음성 기반 우울증 분류를 위한 머신러닝 모델의 발전 가능성을 다룹니다. 특히, 음성 입력의 길이가 모델 성능에 미치는 영향을 분석하고 있습니다. 1400시간 이상의 음성 데이터를 사용하는 연구를 통해 다양한 입력 길이에 따른 성능 변화를 조사하였습니다.

- **Technical Details**: 연구에서는 두 개의 NLP(자연어 처리) 시스템을 비교하여 우울증 분류 성능을 분석합니다. 성능은 자연 길이(natural length), 경과 길이(elapsed length), 세션 내에서의 응답 순서(ordering)에 따라 달라집니다. 두 시스템 모두 최소 길이(threshold) 기준을 공유하지만, 응답 포화(threshold of response saturation)에 대한 기준은 다릅니다. 더 좋은 시스템은 더 높은 포화 기준을 보입니다.

- **Performance Highlights**: 상대적으로 응답이 포화 상태일 때 현재 질문을 계속하기 보다는 새로운 질문을 제시하는 것이 유리하다는 결과를 보였습니다. 이 연구의 결과들은 우울증 분류를 위해 최적의 입력 길이를 유도하고 처리하는 애플리케이션 설계에 대한 통찰을 제공합니다. 이를 통해 헬스케어 분야에서 더욱 효과적으로 활용될 수 있는 가능성을 제시합니다.



### "Dialogue" vs "Dialog" in NLP and AI research: Statistics from a Confused Discours (https://arxiv.org/abs/2501.00598)
- **What's New**: 이 논문은 컴퓨터 과학 분야에서 자주 사용되는 용어인 ‘dialog(ue)’의 표기 차이에 대한 분석을 제공합니다. 저자들은 20년 이상에 걸친 NLP/AI 연구에서 이 주제를 다루며, 연구자들이 ‘dialog’와 ‘dialogue’ 중 어느 것이 더 많이 사용되는지에 대한 패턴을 조사했습니다. 특히, 저자들은 이 문제를 ‘dialog(ue) debacle’로 표현하며, 저명한 논문에서 두 가지 표기가 혼합되어 사용되는 현상을 관찰하였습니다.

- **Technical Details**: 연구 방법으로는 Semantic Scholar API를 이용하여 ‘Dialog(ue) Papers’를 수집하였고, 총 87,498개의 작업 중 52,249개가 ‘Computer Science’ 분야로 분류되었습니다. 이를 통하여 저자별로 발표한 문헌의 사용 비율을 계산하고, 다양한 통계적 접근 방식으로 데이터를 분석하였습니다. 특히, 연구자들의 소속 기관과 해당 표기법 간의 관계를 분석하였고, 특정 기관의 저자들이 ‘dialogue’ 대신 ‘dialog’를 더 많이 사용하는 경향이 있음을 발견했습니다.

- **Performance Highlights**: ‘dialogue’ 표기는 72%로 주류를 이루고 있는 반면, ‘dialog’는 24%에 불과하며, 두 표기를 혼합하여 사용하는 경우는 5%로 나타났습니다. 이 연구는 AI/NLP 분야에서의 언어적 변화와 관련된 중요한 통찰을 제공하며, 향후 더 많은 연구가 필요하다는 점을 강조합니다. 이 연구의 결과는 언어 연구 분야 및 컴퓨터 언어학에 대한 새로운 연구 방향을 제시할 것으로 기대됩니다.



### Setting Standards in Turkish NLP: TR-MMLU for Large Language Model Evaluation (https://arxiv.org/abs/2501.00593)
Comments:
          6 pages, 2 tables, submitted to arXiv for review. Includes a comprehensive evaluation framework for Turkish NLP tasks and state-of-the-art LLM evaluations

- **What's New**: 이번 연구는 터키어 대형 언어 모델(Turkish Large Language Models, LLMs) 평가를 위한 새로운 기준인 터키 MMLU(TR-MMLU) 벤치마크를 소개합니다. 이 벤치마크는 터키 교육 시스템에서 파생된 6,200개의 다지선다형 질문으로 구성되어 있으며, 62개 분야를 포괄합니다. TR-MMLU는 언어적, 개념적 능력을 평가하는 투명하고 응답 가능한(framework) 도구로서 터키어 NLP 연구의 표준 프레임워크 역할을 합니다.

- **Technical Details**: TR-MMLU는 다양한 주제를 포함하고 있으며, 각 질문은 현실 세계의 교육 성과에 따라 난이도 등급이 매겨져 모델의 능력을 미세하게 평가합니다. 이 벤치마크는 지식 평가(knowledge assessment) 및 지시 수행(instruction following)이라는 두 가지 주요 차원에서 LLMs의 성능을 평가합니다. 터키어의 복잡한 형태론적 구조를 고려하여 질문을 구성하여 모델이 언어의 표면 패턴과 깊은 언어 구조를 이해하도록 요구합니다.

- **Performance Highlights**: 본 연구에서는 최첨단 LLMs를 TR-MMLU에서 평가하여 터키어 관련 작업에서 강점과 한계를 조명했습니다. 연구 결과, 모델 성능에 영향을 미치는 토크나이제이션(tokenization) 및 파인튜닝(fine-tuning) 전략과 같은 중요한 과제를 발견하였습니다. TR-MMLU는 터키어 모델 평가의 새로운 표준을 설정함으로써, 향후 터키어 NLP 연구와 혁신을 촉진할 것으로 기대됩니다.



### Causal Graph Guided Steering of LLM Values via Prompts and Sparse Autoencoders (https://arxiv.org/abs/2501.00581)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 가치 정렬(value alignment) 과정을 개선하기 위해 인과 그래프(causal graph)를 활용하여 다양한 가치 간의 관계를 명확히 합니다. 기존의 방법론은 주로 제한된 가치 세트에 초점을 맞추고 있었으나, 본 연구는 가치 간의 상호작용을 분석함으로써 LLM의 행동을 보다 효과적으로 조정할 수 있는 새로운 프레임워크를 제안합니다. 또한, 파라미터 수가 적은 두 가지 조정 메커니즘, 즉 프롬프트 템플릿 프로세스(prompt template steering)와 희소 오토인코더(Sparse Autoencoder) 기능 조정 방법을 도입하였습니다.

- **Technical Details**: 연구에서는 LLM의 내부에서 발생하는 다양한 가치 간의 인과 관계를 표현하기 위해 인과 그래프를 채굴합니다. 이 그래프는 각 가치가 다른 가치에 미치는 영향을 보여주어, LLM의 결정 과정을 이해하는 데 도움을 줍니다. 특히, 두 가지 새로운 경량의 조정 메커니즘을 통해 특정 가치 차원을 변경할 때 다른 차원에 미치는 영향을 분석하며, 이 과정을 통해 LLM이 인간의 가치에 더욱 일치하도록 유도합니다.

- **Performance Highlights**: Gemma-2B-IT 및 Llama3-8B-IT 모델에 대한 광범위한 실험을 통해 제안된 조정 방법의 효과성과 통제 가능성을 검증했습니다. 연구 결과, 인과 그래프를 활용한 조정 방식이 LLM의 출력 일관성을 강화하며, 다양한 가치 차원 간의 상호작용을 조정하는 데 긍정적인 영향을 미친다는 점이 확인되었습니다. 이러한 발견은 AIl의 가치 일치성을 향상시키는 데 큰 기여를 할 것으로 기대됩니다.



### KnowRA: Knowledge Retrieval Augmented Method for Document-level Relation Extraction with Comprehensive Reasoning Abilities (https://arxiv.org/abs/2501.00571)
- **What's New**: 이 논문에서는 Document-level Relation Extraction (Doc-RE)을 위한 새로운 방법인 KnowRA를 제안합니다. KnowRA는 외부 지식을 활용하여 Doc-RE의 포괄적인 추론 능력을 발전시킵니다. 이 방법은 문서 그래프를 구축하고 코-참조 추론 모델을 통합하여 코-참조 관계를 강화합니다.

- **Technical Details**: KnowRA는 문서의 입력에서 의미적 인코딩을 수행하기 위해 사전 학습된 언어 모델을 사용하며, 다양한 엔티티, 언급 및 문장 간의 연결을 모델링하기 위해 다층 이질 문서 그래프(Multi-level Heterogeneous Document Graph, MHDG)를 정의합니다. MHDG는 언급 노드(Mention node), 문장 노드(Sentence node), 문서 노드(Document node)의 세 가지 타입으로 구성되며, 다양한 엣지를 정의하여 노드 간의 관계를 나타냅니다.

- **Performance Highlights**: 논문에서 제시한 방법은 두 개의 공개 데이터셋에서 상태-of-the-art(SOTA) 기준과 비교하여 우수한 성능을 보임을 입증하였습니다. 특히, KnowRA는 외부 지식을 필터링하고 자동으로 수용 여부를 결정하는 방법을 통해 Doc-RE의 정확성을 높입니다.



### An Overview and Discussion on Using Large Language Models for Implementation Generation of Solutions to Open-Ended Problems (https://arxiv.org/abs/2501.00562)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)을 활용하여 전통적인 알고리즘 규격을 초월하는 자동 구현 생성 방법을 제시하는 새로운 기회를 강조합니다. LLMs는 문제 프레이밍, 잠재적 해결 접근법 탐색 및 기능 조합 등의 문제 해결 활동을 지원할 수 있는 가능성을 지니고 있습니다. 또한, 이 모델들은 외부 문서에서 새로운 기능을 학습하거나 이전에 생성된 구현에 기반하여 동적으로 지식을 업데이트할 수 있습니다.

- **Technical Details**: 논문은 문제 해결을 위한 구현 생성 활동을 모델링하는 데 LLMs의 잠재력을 조사했습니다. Prompting, Reinforcement Learning(RL) 및 Retrieval-Augmented Generation(RAG)과 같은 방법론을 사용하여 기존의 자동화된 구현 생성 방법에서 다루어지지 않았던 문제 해결 활동을 구현할 수 있는 방안을 논의합니다. 특히, LLMs의 다양한 연관성을 학습할 수 있는 능력을 활용하여 다중 모달 데이터에 대한 이해를 심화시키는 것이 중요하다고 설명합니다.

- **Performance Highlights**: 전통적인 구현 생성 방법들은 주로 잘 정의된 문제와 일명 '하이레벨' 사양에 착안하여 이루어져 왔으나, 이 연구는 다양한 문제 정의가 필요한 개방형 문제 해결에 대해 LLMs가 기여할 수 있는 새로운 기회를 제공합니다. LLMs는 팀 작업에서의 문제 프레이밍과 해결 접근법 탐색을 지원할 수 있으며, 시도-오류 및 신속한 프로토타입 제작을 통해 새로운 기회와 한계를 탐색할 수 있는 자동화를 제공해야 합니다.



### Re-evaluating Automatic LLM System Ranking for Alignment with Human Preferenc (https://arxiv.org/abs/2501.00560)
- **What's New**: 이 논문은 다양한 LLM의 성능을 평가하고 순위를 매기는 자동화된 평가 프레임워크인 LLM bencher의 필요성을 강조합니다. 기존의 연구에서는 이 시스템의 구성 요소를 선택하는 방법이나 조합이 결과에 미치는 영향을 충분히 탐구하지 않았습니다. 이 연구는 통제된 실험을 통해 LLM 평가의 자동화를 위한 구성 요소 선택에 대한 추천을 제공합니다.

- **Technical Details**: 자동 LLM bencher는 입력 집합(input set), 평가 모델(evaluation model), 평가 유형(evaluation type), 집계 방법(aggregation method) 네 가지 구성 요소로 이루어져 있습니다. 특히, 이 연구에서는 Arena Hard와 Llama-3.1-70B 같은 모델을 사용하여 LLM의 성능 평가는 물론, 평가 유사성이 높은 시스템 간의 비교 시 성능 저하가 발생함을 밝혀냈습니다. 또, 각 구성 요소의 선택이 LLM bencher의 효율성에 미치는 영향을 실험을 통해 분석하였습니다.

- **Performance Highlights**: 연구 결과, Arena Hard의 입력 집합을 사용할 경우 Chatbot Arena의 시스템 순위와의 상관관계가 항상 높게 나타났습니다. 또, GPT-4-turbo와 같은 강력한 평가 모델은 LLM 평가에서 뛰어난 성능을 보이는 반면, 비슷한 성능의 시스템들을 평가할 때 성능이 급격히 하락하는 한계가 있음을 발견했습니다. 마지막으로, 인스턴스 수준(instance-level)에서의 평가 결과는 시스템 수준(system-level) 평가의 좋은 참고자료가 될 수 있는 가능성을 보여줍니다.



### AraSTEM: A Native Arabic Multiple Choice Question Benchmark for Evaluating LLMs Knowledge In STEM Subjects (https://arxiv.org/abs/2501.00559)
- **What's New**: 이번 논문에서는 AraSTEM이라는 새로운 아랍어 다지선택 문제 데이터셋을 소개합니다. 이 데이터셋은 STEM 분야에서 LLMs(대규모 언어 모델)의 지식을 평가하는 목적을 가지고 있으며, 다양한 주제를 포함하고 있습니다. 기존의 LLM 평가 기준이 주로 영어에 초점을 맞추고 있어, 다국어 지원 모델의 부족한 아랍어 평가 지표를 보완합니다.

- **Technical Details**: AraSTEM 데이터셋은 총 11,637개의 MCQ(다지선택 문제)로 구성되어 있으며, 초등학교와 중학교 수준부터 고급 생물학, 물리학, 의학 등 다양한 주제를 포함합니다. 데이터는 여러 출처에서 수집되었으며, 각 질문의 출처에 대한 링크가 포함되어 있습니다. 수집 과정은 웹 스크래핑, 수동 추출 및 LLM을 활용한 방법을 포함하고 있습니다.

- **Performance Highlights**: 초기 실험 결과는 공개된 다양한 크기의 LLM 모델들이 AraSTEM 데이터셋에서 저조한 성과를 보였다는 것을 보여줍니다. 이는 아랍어와 과학 분야에 대한 언어 모델의 이해도가 부족함을 시사하며, 더 지역화된 언어 모델 개발의 필요성을 강조합니다. Hugging Face를 통해 데이터셋은 무료로 접근할 수 있습니다.



### Superposition in Transformers: A Novel Way of Building Mixture of Experts (https://arxiv.org/abs/2501.00530)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 새로운 작업이나 도메인에 적응시키는 과정에서 겪는 치명적인 잊음(catatstrophic forgetting) 문제를 해결하기 위한 Superposition in Transformers라는 혁신적인 아키텍처를 제안합니다. 이 아키텍처는 기존 모델과 미세 조정된 모델의 숨겨진 표현(hidden representations)을 공유된 파라미터 공간 내에서 중첩(supeposition) 시킵니다. 이를 통해 기존 지식을 보존하면서 도메인 특정 전문성을 추가할 수 있는 새로운 패러다임을 제공합니다.

- **Technical Details**: 제안된 방법은 B-spline 기반의 혼합 계수(blending coefficients)와 자동 인코더(autoencoders)를 활용하여 입력 데이터 분포에 따라 적응적으로 숨겨진 상태를 재구성(reconstruct)합니다. 이러한 접근 방식은 치명적인 잊음 현상을 효과적으로 완화하고, 모델 상태 간의 동적 전환을 지원하여 추론(inference) 과정에서 원활한 작업 전환을 가능하게 합니다. Superposition을 사용함으로써 기존 모델의 기능이 유지됩니다.

- **Performance Highlights**: Superposition in Transformers는 모델의 원래 성능을 저해하지 않으면서도 새로운 작업에 대한 적응성을 높입니다. 실험 결과, 이 방법이 기존의 미세 조정 기법보다 성능 저하를 더욱 효과적으로 방지하고, 다양한 도메인에서의 효율성을 보여줍니다. 이로 인해 대형 언어 모델의 활용 가능성이 극대화됩니다.



### Sinhala Transliteration: A Comparative Analysis Between Rule-based and Seq2Seq Approaches (https://arxiv.org/abs/2501.00529)
Comments:
          8 pages, 7 tables

- **What's New**: 이번 연구는 Sinhala 언어의 Romanized (로마자 표기법) 변환에 관한 연구로, 비록 많은 연구가 행해졌지만 Singlish (신글리쉬)에서 Sinhala로의 변환을 위한 충분한 연구가 이루어지지 않은 상황을 다룹니다. 제안된 두 가지 방법은 규칙 기반 접근 방식과 Transformer 기반의 딥러닝 접근 방식입니다. 이 연구는 두 접근 방식을 비교하여 깊이 있는 언어 변동성에 대한 내성을 평가하였습니다.

- **Technical Details**: 규칙 기반 접근 방식은 정의된 언어 규칙을 사용하여 라틴 문자로 표기된 Sinhalese 단어를 Sinhala 문자로 매핑하는 방식입니다. 이 접근 방법은 Tennage et al.(2018)의 시스템에 두세 글자를 고려한 매핑 규칙을 추가하는 방식으로 발전했습니다. 반면, 딥러닝 기반 접근 방식은 사전 훈련된 seq2seq(multilingual) 모델을 사용하여 Transformer 아키텍처를 구현하여 Singlish에서 Sinhala로의 번역 문제를 해결합니다.

- **Performance Highlights**: 결과적으로 딥러닝 접근 방식이 규칙 기반 접근 방식에 비해 다양한 언어 형태에 훨씬 더 강력하다는 것을 발견했습니다. 특히, Transformer 기반 모델이 라틴 문자로 작성된 Romanized 스크립트의 비공식적 패턴을 보다 효과적으로 포착할 수 있음을 보여주었습니다. 이러한 발견은 Sinhala의 기계적인 변환 시스템을 개발하기 위한 유망한 길을 제시합니다.



### TinyHelen's First Curriculum: Training and Evaluating Tiny Language Models in a Simpler Language Environmen (https://arxiv.org/abs/2501.00522)
- **What's New**: 이번 연구에서는 기계학습에서 언어 모델의 학습 효율성을 향상시키기 위해 간소화된 언어 환경을 수립하는 방안을 제안합니다. 기존의 대형 언어 모델(Large Language Models, LLMs) 학습에 필요한 방대한 데이터셋과 자원을 줄이기 위해, 텍스트 데이터를 정제하고 노이즈를 제거한 간소화된 데이터셋을 만듭니다. 이런 방법을 통해 조그마한 언어 모델(Tiny LMs)이 instruction-following(지시 따르기) 능력을 더 효과적으로 학습하도록 합니다.

- **Technical Details**: 이 연구에서는 'no noise, low complexity' 원칙을 바탕으로 한 텍스트 데이터 개선 파이프라인을 구현하여 71M Leaner-Pretrain, 7M Leaner-Instruct, Leaner-Glue 및 Leaner-Eval 등 간결한 언어 모델 훈련 및 평가 데이터셋을 생성합니다. 이 데이터셋은 기존의 언어 모델 훈련 데이터의 구성 DNA와 평가 기준을 유지하면서도 언어적으로 더욱 단순화된 특징을 지니고 있습니다. 이를 통해 저비용(high efficiency) 모델을 위한 학습 데이터를 효율적으로 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, Leaner-Pretrain 데이터셋으로 사전 학습된 조그마한 언어 모델들이 원본 데이터셋에 비해 instruction-following 성능이 향상되었음을 확인하였습니다. 특히, 저희의 방안으로 학습한 모델은 문법, 일관성, 구체성을 개선하여 다른 모델들보다 더 높은 성능을 기록했습니다. 이러한 결과는 기존 복잡한 모델 환경에서도 저비용으로 변형된 데이터셋을 통해 효과적인 학습이 가능하다는 것을 보여줍니다.



### Fotheidil: an Automatic Transcription System for the Irish Languag (https://arxiv.org/abs/2501.00509)
Comments:
          Accepted to the 5th Celtic Language Technology Workshop within COLING 2025

- **What's New**: 이 논문은 아일랜드어를 위한 첫 번째 웹 기반 텍스트 전사 시스템인 Fotheidil를 소개합니다. 이 시스템은 ABAIR(아일랜드어 기술 개발 이니셔티브)의 일환으로 음성 관련 AI 기술을 활용합니다. 다수의 사전 훈련된 음성 활동 탐지(voice activity detection) 및 화자 분리(speaker diarisation) 모델이 포함되며, 아일랜드어 자동 음성 인식(automatic speech recognition)을 위해 특별히 훈련된 모델도 포함되어 있습니다.

- **Technical Details**: 반자동 학습(semi-supervised learning)을 활용하여 모듈식 TDNN-HMM ASR 시스템의 음향 모델을 개선하는 방법을 탐구했습니다. 또한, 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 사용해 대문자화(capitalisation) 및 구두점 복원(punctuation restoration) 작업을 수행하는 새로운 접근 방식을 제안했습니다. 실험 결과는 기존의 분류 모델을 사용하는 접근법에 비해 성능 상의 상당한 개선을 보여주었습니다.

- **Performance Highlights**: 웹 기반 전사 시스템은 사용자가 아일랜드어 자료를 쉽게 전사할 수 있는 중요한 자원으로 제공됩니다. 인간이 수정한 전사 데이터는 훈련 데이터셋에 통합되어 ASR 모델의 점진적인 개선을 위해 사용됩니다. 이러한 커뮤니티 주도의 순환적 접근 방식은 아일랜드어 음성 인식의 품질을 향상시킬 것으로 기대됩니다.



### Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents (https://arxiv.org/abs/2501.00430)
- **What's New**: 이번 연구에서는 복잡한 과학적 추론 작업을 위해 Reactive and Reflection agents with Multi-Path Reasoning (RR-MP) 프레임워크를 제안합니다. 이 프레임워크는 다중 경로 추론 메커니즘을 통해 LLMs의 추론 능력을 향상시키며, 각 경로에서 반응형 에이전트와 반영형 에이전트가 협력하여 생각의 퇴화를 방지합니다. 추가적인 훈련 없이도 다양한 대화 사례를 활용하여 다각적인 인사이트를 통합합니다.

- **Technical Details**: RR-MP 프레임워크는 시스템 1(빠르고 직관적인 사고)과 시스템 2(느리고 심사숙고하는 사고)로 구성된 이중 시스템 모델을 적용하여 의사결정 성능을 향상시킵니다. 각 경로는 반응형 에이전트와 반영형 에이전트의 협업을 통해 최적화되며, 이를 통해 다각적인 관점이 통합되고 반영형 에이전트가 스스로 수정하도록 유도합니다. 이 프레임워크는 0-shot 및 few-shot 평가를 통해 도덕적 시나리오, 대학 물리학 및 수학 관련 작업에서 검증되었습니다.

- **Performance Highlights**: 실험 결과, RR-MP 프레임워크가 기존의 베이스라인 방법보다 우수한 성능을 보였으며, 오류 수정 및 경로 최적화를 통해 정확도가 향상되었습니다. 특히, 여러 경로에서의 외부 자극을 통한 자기 수정이 높은 정확도로 이어짐을 보였습니다. 이를 통해 복잡한 과학적 추론 작업을 효과적으로 관리할 수 있음을 확인하였습니다.



### Whisper Turns Stronger: Augmenting Wav2Vec 2.0 for Superior ASR in Low-Resource Languages (https://arxiv.org/abs/2501.00425)
Comments:
          15 pagesm 3 figures

- **What's New**: 이 논문은 Wav2Vec2 모델을 활용한 자동 음성 인식 시스템(ASR) 개선을 위한 새로운 종단 간(end-to-end) 프레임워크를 제안합니다. 특히 자원이 부족한 저자원 언어인 아랍어, 러시아어 및 포르투갈어에 초점을 맞춰, 데이터 증대(data augmentation) 기법을 통해 성능을 향상시키려 시도합니다. 연구는 세 가지 언어 데이터셋을 이용해 프레임워크의 효과를 검증하며, 다양한 악센트와 발음 변화를 다룰 수 있는 강인성을 보여줍니다.

- **Technical Details**: ASR(Automatic Speech Recognition) 기술은 인간의 음성을 읽을 수 있는 텍스트로 변환하는 프로세스를 포함합니다. 본 논문에서는 딥러닝 및 자기 지도 학습(self-supervised learning, SSL)을 활용한 최신 Wav2Vec2 모델을 기반으로 하여, 적은 레이블이 있는 음성 데이터로도 경쟁력 있는 성과를 도출할 수 있음을 보여줍니다. 이 모델은 특히 다양한 방언과 발음의 영향을 받으며, 필연적으로 발생하는 배경 소음 문제에도 강점을 갖추고 있습니다.

- **Performance Highlights**: 제안한 ASR 프레임워크는 Word Error Rate(WER)에서 33.9% 개선, Character Error Rate(CER)에서 53.2% 개선을 달성하면서 기존의 Wav2Vec2 및 Whisper ASR 모델들을 초월하는 결과를 나타냅니다. 이러한 성과는 저자원 언어의 방언 변동성을 효과적으로 처리할 수 있는 견고한 메커니즘을 갖추었다는 것을 시사합니다. 또한, CER 평가 지표의 중요성을 강조하며, 실제 환경의 언어적 다양성에 잘 대응할 수 있는 ASR 시스템을 발전시키는 데 기여할 것입니다.



### Trajectories of Change: Approaches for Tracking Knowledge Evolution (https://arxiv.org/abs/2501.00391)
- **What's New**: 이 논문은 socio-epistemic networks (SEN) 프레임워크를 통해 지식 시스템의 지역적(local) 및 글로벌(global) 진화를 탐구하고 있습니다. SEH는 사회적, 기호적(물질적), 의미적 세 가지 상호 연결된 층으로 구성되며, 이는 지식의 구조적 발전을 이해하는 다층적 접근법을 제안합니다. 정보를 기반으로 한 측정 방식을 통해 언어 모델의 의미적 변화와 문서의 임베딩 밀도를 분석하여 의미적 변화의 중요성과 핵심 특징을 식별합니다.

- **Technical Details**: 이 연구에서는 정보 이론적 측정기법인 상대 엔트로피(relative entropy)와 문서 임베딩 밀도(document embedding density) 변화를 활용하여 시간에 따른 의미의 변화를 분석합니다. 상대 엔트로피는 문서의 의미가 어떻게 변화해왔는지 식별하고 이를 통해 학술 문서의 내용(주제)이나 메타데이터(저자, 소속기관)에 기반한 궤적을 추적할 수 있습니다. Joseph Silk와 Hans-Jürgen Treder를 사례 연구로 통해 특정 학자의 작업이 어떻게 넓은 학문적인 변화와 일치하는지를 보여줍니다.

- **Performance Highlights**: 본 연구는 ‘일반 상대성 이론의 르네상스’를 다루며, 여러 역사적 및 사회적 요인이 지식 시스템의 발전에 미친 영향을 분석합니다. 연구 결과, 1950년대 중반부터 시작된 연구자 간의 연결성이 증가하고 있다는 점과 함께, 학문 공동체가 형성되고 있다는 패턴을 밝혀냈습니다. 이 접근법은 전체 학문적 흐름과 개인의 연구 경로 간의 연관성을 시각적으로 나타내는 데 OT한 기여를 하며, 과거의 패턴을 기반으로 새로운 연구 방향을 제안합니다.



### RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions (https://arxiv.org/abs/2501.00353)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 접근 방식인 RAG-Instruct를 제안합니다. 기존 RAG 방법은 제한된 시나리오와 데이터셋으로 인해 작용 범위가 적었으나, RAG-Instruct는 다양한 고품질 RAG 지침 데이터를 생성할 수 있는 일반적인 방법을 제공합니다. 이를 통해 RAG의 다양한 사용 사례를 포괄적으로 다룰 수 있는 40K의 지침 데이터셋을 구축했습니다.

- **Technical Details**: RAG-Instruct는 다섯 가지 RAG 패러다임을 활용하여 다양한 질의-문서 관계를 모색합니다. 또한, 기존 지침 데이터셋의 강점을 활용해 지침 시뮬레이션을 통해 지침의 다양성과 품질을 높입니다. 이 방법은 Wikipedia에서 만들어진 데이터셋을 기반으로 하여 RAG 시나리오 및 작업을 포괄적으로 커버합니다.

- **Performance Highlights**: 실험 결과, RAG-Instruct는 LLMs의 RAG 능력을 효과적으로 향상시켰으며, 제로샷 성능에서 강력한 결과를 보였습니다. 다양한 RAG 기준선과 비교하여 상당히 뛰어난 성능을 보였으며, 이는 다양한 작업 세트를 통해 나타났습니다. RAG-Instruct에 대한 모든 정보는 공개된 링크를 통해 확인할 수 있습니다.



### Chunk-Distilled Language Modeling (https://arxiv.org/abs/2501.00343)
- **What's New**: CD-LM(Chunk-Distilled Language Modeling)은 현재의 대규모 언어 모델들이 겪고 있는 두 가지 주요 문제인 토큰 레벨 생성의 비효율성과 새로운 데이터 및 지식으로 적응하기 어려움을 처리하는 새로운 접근 방식을 제시합니다. 이 방법은 심층 네트워크 기반의 언어 모델과 간단한 검색 모듈을 결합하여 한 번의 디코딩 단계에서 여러 토큰의 텍스트 청크를 생성할 수 있게 합니다. CD-LM은 확인된 청크를 활용하여 효율성을 높이고, 새로운 영역에 적응할 수 있는 유연성을 제공합니다.

- **Technical Details**: CD-LM은 다양한 크기의 텍스트 청크를 저장하고, 현재 생성 과정에 따라 가장 가능성이 높은 청크를 검색하기 위해 트라이(trie) 구조의 데이터 저장소를 사용합니다. 청크는 일반적인 사전 훈련된 언어 모델에서 발생할 수 있으며, 이는 메모리에 저장된 높은 확률의 시퀀스로서 작용할 수 있습니다. 이 과정에서는 별도의 임베딩 모듈을 사용하는 추가적인 오버헤드 없이, 언어 모델이 자체적으로 유도한 벡터 표현 공간에서 맥락 매칭이 수행됩니다.

- **Performance Highlights**: CD-LM은 언어 모델링 난이도, 텍스트 생성 및 도메인 적응 등에 대한 다양한 경험적 연구를 통해 검증되었습니다. 본 방법은 인퍼런스 효율성과 모델 성능을 동시에 향상시킬 수 있는 능력을 보여줍니다. 결과적으로 CD-LM은 기존의 사전 훈련된 모델의 분포에 새로운 청크를 삽입하여 추가 훈련 없이도 성능을 개선시킬 수 있습니다.



### Rethinking Layer Removal: Preserving Critical Components with Task-Aware Singular Value Decomposition (https://arxiv.org/abs/2501.00339)
- **What's New**: Taco-SVD는 태스크 중심의 접근 방식을 통해 LLM(대형 언어 모델)의 레이어 제거 기술에서 발생할 수 있는 성능 저하를 해결하는 혁신적인 프레임워크입니다. 이 방법은 레이어 제거가 내부 일관성을 해치는 문제를 해결하며, 태스크에 중요한 특성을 보존합니다. Taco-SVD는 그라디언트 기반의 어트리뷰션을 활용하여 단일값(singular value)을 다운스트림 태스크 목표와 정렬합니다.

- **Technical Details**: Taco-SVD는 전통적인 레이어 제거 방식을 개선하기 위해 태스크-aware 비고유값 분해(Task-Aware Singular Value Decomposition) 기법을 적용합니다. 이 프레임워크는 레이어의 중복성을 파악하여 중요도가 낮은 구성 요소를 제거하는 대신, 태스크에 중요한 단일값과 그에 연관된 벡터를 유지합니다. 또한, 코사인 유사성을 활용하여 각 레이어의 중복성을 측정하고, 핵심적인 단일값을 보존하는 프로세스를 포함합니다.

- **Performance Highlights**: Taco-SVD는 언어 생성, 상식 추론, 수학적 추론 작업에서 LLaMA 및 Mistral 모델에 대해 실험을 수행하였으며, 레이어 제거에 비례하는 파라미터 감소를 달성하면서도 perplexity와 태스크 성능이 개선되었습니다. 또한, 다양한 모델 아키텍처에서 강력한 성능을 유지하며, 최소한의 캘리브레이션 데이터만으로도 효과적으로 작동함을 확인했습니다. 추가적인 미세 조정을 통해 성능이 더욱 향상되었으며, 이는 Taco-SVD의 신뢰성과 효율성을 강조합니다.



### Loss-Aware Curriculum Learning for Chinese Grammatical Error Correction (https://arxiv.org/abs/2501.00334)
Comments:
          ICASSP 2025

- **What's New**: 본 논문에서는 중국어 문법 오류 수정(CGEC) 작업에서 다양한 수정 난이도를 고려하지 않고 데이터를 처리하는 기존 접근 방식을 개선하기 위한 다중 레벨 커리큘럼 학습(Multi-granularity Curriculum Learning, CL) 프레임워크를 제안합니다. 이 프레임워크는 오류 수정이 어려운 샘플과 쉬운 샘플의 난이도를 평가하고, 이를 바탕으로 모델이 학습하도록 샘플의 배치를 조정합니다. 기존의 PLM 기반 모델 성능을 개선하려는 이러한 노력은 CGEC에 관한 최근 연구에서 중요한 방향성을 제시하고 있습니다.

- **Technical Details**: CGEC 작업의 정의와 함께, 제안한 커리큘럼 학습 프레임워크는 크게 두 가지 하위 모듈로 구성됩니다. 첫째, 교차 엔트로피 손실 함수를 통해 각 데이터 샘플의 난이도를 평가하고, 낮은 손실을 기록하는 단순 샘플에서부터 높은 손실을 기록하는 복잡한 샘플로 진행하는 배치 레벨 학습이 이루어집니다. 둘째, 샘플의 난이도를 수치적으로 측정하여, 이 난이도에 따라 훈련 과정에서 샘플의 로딩 순서를 조정합니다.

- **Performance Highlights**: 실험 결과, mT5 및 BART와 같은 사전 훈련된 언어 모델을 활용한 결과, 제안된 커리큘럼 학습 방법이 NLPCC 및 MuCGEC 데이터셋에서 기존의 성능을 크게 초과하는 것으로 밝혀졌습니다. 이는 CGEC 모델이 더 어려운 샘플을 효과적으로 학습할 수 있도록 돕는 방안으로, 실제 문법 수정에서의 성능 향상에 기여함을 보여줍니다. 이러한 실험은 CGEC 모델의 성능 개선 가능성을 명확하게 입증하고 있으며, 향후 연구에 중요한 기초 자료가 될 것입니다.



### MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation (https://arxiv.org/abs/2501.00332)
- **What's New**: 이번 연구에서는 Multi-Agent Filtering Retrieval-Augmented Generation (MAIN-RAG)이라는 새로운 RAG 프레임워크를 제안합니다. 이 방법은 여러 LLM 에이전트를 활용하여 문서를 필터링하고 평가하는 협업적 접근 방식을 사용하여, 보다 정확하고 신뢰성 있는 응답을 제공합니다. 특히, MAIN-RAG는 추가적인 훈련 없이도 작동할 수 있도록 설계되어 실제 응용 프로그램에서의 강력한 확장성을 자랑합니다.

- **Technical Details**: MAIN-RAG는 동적으로 조정되는 적응형 필터링 메커니즘을 도입하여, 검색된 문서의 점수 분포에 따라 관련성 필터링 임계값을 조정합니다. 이를 통해 노이즈를 효과적으로 최소화하면서도 관련 문서의 회수 비율을 높이고, 여러 쿼리에 대한 robust한 성능을 보장합니다. 각 문서에는 관련성 점수가 할당되며, 이 점수를 기반으로 문서의 순위를 매기고 불필요한 문서를 필터링합니다.

- **Performance Highlights**: MAIN-RAG는 4개의 QA 벤치마크에서 실험을 통해 기존의 RAG 방법보다 2-11% 향상된 답변 정확도를 기록했습니다. 또한 불필요한 문서의 수를 줄이고, 응답의 일관성과 정확성을 향상시는 등 훈련 기반 솔루션에 대한 경쟁력 있는 대안으로 자리매김할 수 있음을 보여주었습니다.



### Exploring the Implicit Semantic Ability of Multimodal Large Language Models: A Pilot Study on Entity Set Expansion (https://arxiv.org/abs/2501.00330)
Comments:
          ICASSP 2025

- **What's New**: 본 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 다중 모달 엔티티 세트 확장(Multi-modal Entity Set Expansion, MESE) 작업에 적용하여 기존의 한계점을 보완하고, 이 도구의 능력을 탐색합니다. 새로운 엔티티를 시맨틱 클래스에 맞춰 탐색하는 MESE 작업을 통해 MLLM의 암묵적 의미 정보 추출 능력을 평가합니다. 특히, LUSAR라는 리스트 순위 매김 방법을 도입하여 지역 점수를 글로벌 순위로 매핑하는 방식을 제안합니다.

- **Technical Details**: LUSAR 방법론은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 접두사 트리 제약을 사용하여 모델이 지정된 엔티티 데이터 세트 내의 많은 후보 엔티티를 생성하도록 제한합니다. 두 번째 단계에서는 리스트별 접근 방식을 도입하여 각 후보 엔티티에 대한 순위 점수를 얻기 위해 여러 샘플링 및 순위를 수행합니다. 이러한 접근은 암묵적인 정보에서 세부적으로 구분된 의미 기능을 추출하는 데 도움을 줍니다.

- **Performance Highlights**: LUSAR 방식의 적용을 통해 MLLM의 MESE 작업에서 성능 향상이 크게 이루어졌습니다. 이 방식은 대형 모델이 암묵적인 의미 정보를 추출하는 데 어려워하는 문제를 해결하고, 성공적인 후보 엔티티 선정에 기여합니다. 경량의 리스트 방식 접근을 통하여 추천 시스템과 같은 다른 작업에도 적용 가능성을 보여주며, 실험 결과로 MLLM의 성능이 크게 향상된 것을 확인하였습니다.



### MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models (https://arxiv.org/abs/2501.00316)
Comments:
          40 pages, 21 figures

- **What's New**: 최근 기초 모델(Foundation Models)의 발전은 AI 시스템의 자율 도구 사용 및 추론 능력을 향상시켰습니다. 그러나 위치 기반 또는 지도 기반 추론의 능력은 시스템적으로 연구되지 않았습니다. 이를 해결하기 위해 우리는 MapEval이라는 벤치마크를 도입하여 복잡한 지도 기반 사용자 쿼리를 평가하려고 합니다.

- **Technical Details**: MapEval은 다양한 지리 공간적 추론(geo-spatial reasoning)을 평가하는 3가지 작업 유형(텍스트, API 기반 및 시각적)을 포함합니다. 이 작업들은 세계 정보를 수집하고 이질적인 지리 공간 맥락(예: 명명된 엔티티, 여행 거리, 사용자 리뷰 또는 평점, 이미지)을 처리하며, 조합적 추론(compositional reasoning)을 요구합니다. MapEval은 180개 도시와 54개국에 대한 700개의 고유한 객관식 질문으로 구성되어 있습니다.

- **Performance Highlights**: MapEval을 사용하여 28개의 저명한 기초 모델을 포괄적으로 평가했습니다. 모든 작업에서 단일 모델이 뛰어난 성과를 보이지는 않았지만, Claude-3.5-Sonnet, GPT-4o 및 Gemini-1.5-Pro가 경쟁력 있는 성능을 기록했습니다. 그러나 MapEval에서 Claude-3.5-Sonnet이 GPT-4o 및 Gemini-1.5-Pro보다 각각 16% 및 21% 더 높은 성능을 보였고, 오픈 소스 LLM들과 비교했을 때 성능 격차가 더욱 두드러졌습니다.



### LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts (https://arxiv.org/abs/2501.00274)
Comments:
          Updated version of 17 June 2024

- **What's New**: 이 논문에서는 자연어 텍스트의 자동 평가를 위한 새로운 프레임워크인 LLM-Rubric을 소개하고 있습니다. 이 방법은 수작업으로 작성된 평가 기준을 바탕으로 하여, 다차원적으로 문서를 평가할 수 있도록 설계되었습니다. LLM (Large Language Model)의 예측 결과를 인간 평가자의 주석과 일치시키는 문제에 대한 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: LLM-Rubric은 두 단계로 구성된 평가 프로세스를 따릅니다. 첫 번째 단계에서는 LLM에 텍스트와 평가 기준을 제시하고 가능한 응답에 대한 확률 분포를 이끌어내며, 두 번째 단계에서는 이러한 분포를 소형 피드포워드 신경망을 통해 조정하여 인간 평가자의 선호도에 맞춥니다. 이 방법은 각 평가 기준에 따라 응답을 평가하는 9개의 질문을 사용하여 대화 시스템을 테스트합니다.

- **Performance Highlights**: 실험 결과, LLM-Rubric은 RMSE (Root Mean Square Error) 0.5 미만으로, 인간 평가자들의 전체 사용자 만족도 평가를 예측하는 데 있어 2배 이상의 성능 향상을 보여주었습니다. 이는 기존의 비조정 LLM 방법론에 비해 더욱 높은 정확성을 제공합니다. 이러한 자동 텍스트 평가는 다양한 분야에서 인간 평가를 대체하거나 보완할 수 있는 가능성을 시사합니다.



### Echoes in AI: Quantifying Lack of Plot Diversity in LLM Outputs (https://arxiv.org/abs/2501.00273)
- **What's New**: 이번 연구는 최신 대형 언어 모델(LLM)을 사용한 창의적 콘텐츠 생성의 다양성과 반복성에 관한 질문을 다룹니다. 특히, GPT-4와 LLaMA-3의 스토리 생성 기능을 평가하며, LLM이 생성한 스토리에 있는 플롯 요소의 반복성을 분석합니다. 본 연구에서는 Sui Generis 점수를 도입하여 LLM이 생성한 스토리의 독특함을 정량적으로 평가합니다.

- **Technical Details**: Sui Generis 점수는 LLM이 생성한 스토리의 다양한 대안적 연속성을 기반으로 각 스토리 세그먼트의 독창성을 측정합니다. 연구에서 사용된 데이터는 WritingPrompts 및 TV 에피소드의 줄거리 요약을 포함하며, 생성된 100개의 단편 스토리에서 얻은 약 3,700개의 세그먼트를 분석했습니다. 이를 통해 LLM이 생성한 스토리가 유사한 아이디어와 플롯 요소를 반복적으로 포함하고 있다는 결론을 도출했습니다.

- **Performance Highlights**: LLM이 생성한 스토리는 독창성이 낮고 자주 반복되는 플롯 요소로 구성되어 있으며, 이는 인간이 작성한 스토리에서는 드물게 발생합니다. Sui Generis 점수는 자동으로 계산되며, 인간의 판단 없이도 인간의 놀라움 수준과 중간 정도로 상관관계를 보여줍니다. 이 연구는 AI 도구의 사용이 집단적 창의성의 다양성을 줄인다는 이전 연구 결과를 뒷받침하는 정량적 증거를 제공합니다.



### A review of faithfulness metrics for hallucination assessment in Large Language Models (https://arxiv.org/abs/2501.00269)
Comments:
          13 pages, 6 tables

- **What's New**: 이번 리뷰는 open-ended summarization, question-answering, 그리고 machine translation 작업에서의 faithfulness(충실성) 평가 방법을 살펴봅니다. 연구에서는 LLMs(대형 언어 모델)가 충실성 평가기로 사용되는 경우가 가장 일반적이며, 이는 인간의 판단과 높은 상관관계를 갖는다는 점을 발견하였습니다.

- **Technical Details**: 다양한 연구에서 hallucinations(환각 현상)을 완화하기 위해 사용된 방법들이 논의되었으며, retrieval augmented generation(RAG)과 prompting framework 접근 방식이 우수한 충실성과 관련이 있다는 점이 강조되었습니다. 또한 이와 관련된 다른 완화 추천 방법들도 제시되었습니다.

- **Performance Highlights**: 충실성에 대한 연구는 LLMs의 지속적인 사용에 있어 필수적이며, 충실하지 않은 응답은 LLMs가 적합한 많은 분야에서 주요한 위험 요소가 될 수 있습니다. 더욱이, open-ended generation을 평가하는 것은 일반적으로 사용되는 multiple-choice benchmarking보다 LLM의 성능을 보다 포괄적으로 측정할 수 있도록 하여, LLM에 대한 신뢰를 증진시키는 데 기여할 수 있습니다.



### EQUATOR: A Deterministic Framework for Evaluating LLM Reasoning with Open-Ended Questions. # v1.0.0-beta (https://arxiv.org/abs/2501.00257)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 평가 방법에 대한 새로운 접근 방식을 제안합니다. 기존의 평가 기법은 유창성 편향(fluidity bias) 문제를 안고 있으며, 선택형 질문(multiple-choice format)에 의존해 사실적 정확성(factual accuracy)과 복잡한 추론(complex reasoning)을 효과적으로 평가하기 어렵습니다. 이로 인해 LLMs는 복잡한 추론 과제에서 사실적으로 부정확한 응답을 빈번히 생성하는 문제가 발생합니다.

- **Technical Details**: EQUATOR Evaluator(Question Answering Thoroughness in Open-ended Reasoning)는 개방형 추론 과제에 대한 평가 격차를 해소하기 위해 설계된 새로운 프레임워크입니다. 이 시스템은 결정론적 스코어링(deterministic scoring)과 사실적 정확성(factual accuracy), 강력한 추론 평가(robust reasoning assessment)에 초점을 둡니다. EQUATOR는 벡터 데이터베이스(vector database)를 사용하여 개방형 질문(open-ended questions)과 사람의 평가(answer)를 쌍으로 만들어 보다 정밀하고 확장 가능한 평가를 가능하게 합니다.

- **Performance Highlights**: EQUATOR는 인간 평가자(human evaluators)에 대한 의존도를 크게 줄이고, Williams와 Huckle(2004)의 방법에 비해 확장성을 향상시킵니다. 연구 결과, 이 프레임워크는 전통적인 선택형 평가보다 현저히 뛰어난 성능을 보이면서 높은 정확도 기준을 유지합니다. 또한 LLaMA 3.2B를 이용한 자동 평가 과정을 도입해 평가 프로세스를 간소화했습니다.



### Have We Designed Generalizable Structural Knowledge Promptings? Systematic Evaluation and Rethinking (https://arxiv.org/abs/2501.00244)
Comments:
          Work in progress

- **What's New**: 이번 논문은 기존의 Structural Knowledge Prompting (SKP) 패러다임이 대규모 언어 모델(LLMs)에서 어떻게 일반화될 수 있는지를 새로운 관점에서 재검토하고 평가하는 것을 목표로 합니다. 저자들은 기존 SKP 방법의 한계를 극복하고, SKP의 일반화 능력을 입증하기 위해 고안된 새로운 벤치마크인 SUBARU를 소개합니다. 이 벤치마크는 9개의 다양한 난이도와 세분성을 가진 태스크로 구성되어 있습니다.

- **Technical Details**: 이 논문은 SKP에 대한 체계적인 평가를 통해 LLM의 일반화 능력을 다양한 차원에서 탐구합니다. 이러한 차원은 Granularity (세분성), Transferability (전이성), Scalability (확장성), Universality (보편성)으로 나뉘며, 각 차원에서 SKP가 얼마나 효과적으로 작용하는지를 분석합니다. SKP 토큰은 구조적 인코더에 의해 학습된 입력 프롬프트 토큰으로 구성되며, 이 구조적 정보는 LLM의 텍스트 생성을 개선하는 데 기여합니다.

- **Performance Highlights**: 저자들은 SUBARU를 통해 SKP의 성능을 검증하고, 다양한 조건에서 16개의 다른 SKP 설정을 실험하여 SKP의 성공 요인을 찾아냅니다. 이 연구는 LLM의 사실 기반 생성 향상을 위한 더 나은 지침을 제공하며, SKP 방법론이 특정 태스크에 잘 적응하는 이유를 분석하는 데 도움을 줍니다. 또한, 이러한 실험을 통해 SKP의 일반화 능력을 심층적으로 이해할 수 있는 기초 자료를 제공합니다.



### Exploring Variability in Fine-Tuned Models for Text Classification with DistilBER (https://arxiv.org/abs/2501.00241)
- **What's New**: 이번 연구는 DistilBERT 모델을 활용한 텍스트 분류를 위한 파인 튜닝(파인튜닝, fine-tuning) 전략을 평가하며, 하이퍼파라미터에 따른 성능 변화에 대한 실험을 통해 얻은 통찰력을 제공합니다. 특히 학습 속도(learning rate), 배치 크기(batch size), 에폭(epochs)이 정확도(accuracy), F1 점수(F1-score), 손실(loss)에 미치는 영향을 분석합니다.

- **Technical Details**: 우리는 폴리노미얼 회귀(polyomial regression) 분석을 통해 하이퍼파라미터의 내재적 및 점진적 영향을 포착하고, 기본 모델에 대한 상대적 조정의 중요성을 강조합니다. 주요 하이퍼파라미터 간의 상호작용이 F1 점수를 극대화함을 보여주며, 하이퍼파라미터 간 상호작용의 중요성을 강조합니다.

- **Performance Highlights**: 하이퍼파라미터 조정에 따른 성능 지표의 변동성이 나타났으며, 예를 들어 높은 학습 속도는 손실을 줄이지만 정확도 개선에 도전이 될 수 있습니다. 배치 크기는 정확도와 F1 점수에 유의미한 영향을 미치지만 손실 최적화에는 제한적인 영향을 미칩니다. 이러한 결과는 텍스트 분류 외에도 자연어 처리(NLP)와 컴퓨터 비전(CV) 등 다양한 작업에 중요성이 있음을 시사합니다.



### Zero-Shot Strategies for Length-Controllable Summarization (https://arxiv.org/abs/2501.00233)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 길이 조정 능력 문제를 다루고 있습니다. 특히, zero-shot 환경에서의 정확한 길이 제어가 어려운 점을 지적하며, 모델의 내재적 편향을 밝혀냅니다. 이를 보완하기 위해 길이 근사(length approximation), 목표 조정(target adjustment), 샘플 필터링(sample filtering), 자동 수정(automated revisions) 등 여러 실용적인 방법을 제안합니다.

- **Technical Details**: 본 연구는 LLM의 길이 제어 능력을 구조적 측정(Structural Measures)과 세밀한 측정(Granular Measures)으로 나누어 분석합니다. 각 방법론은 단어 수, 문장 수, 문자 수와 같은 다양한 길이 측정을 포함하며, 특히 한 측정의 능력을 다른 측정으로 전이시키는 길이 근사 방법을 개발합니다. 목표 조정 방법을 통해 LLM이 지정된 목표 길이에서 발생하는 편차를 수정하는 메커니즘도 도입됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들을 조합하여 사용함으로써 길이 준수(length compliance)에서 상당한 향상을 보여주었고, 요약의 품질을 유지하거나 향상시키면서도 제로-샷(zero-shot) 환경에서의 효과적인 길이 제어를 입증하였습니다. 이로써 LLM의 통제된 텍스트 생성 이해도를 더욱 발전시키고, 실제 응용 프로그램에서 신뢰할 수 있는 요약 시스템으로 나아갈 수 있는 기반을 제공하였습니다.



### Extracting effective solutions hidden in large language models via generated comprehensive specialists: case studies in developing electronic devices (https://arxiv.org/abs/2501.00224)
Comments:
          18 pages, 4 figures

- **What's New**: 최근 대형 언어 모델(LLMs)을 활용하여 연구 아이디어와 과학 가설 생성을 탐구하는 연구가 증가하고 있습니다. 하지만 실제 연구 및 개발에서는 복잡하고 학제 간의 도전 과제를 해결할 필요가 있으며, 기존 지식을 통해 쉽게 발견할 수 없는 해결책이 필요합니다. 이에 따라, 다양한 분야의 관점을 통합하여 LLM의 방대한 지식을 활용하여 효과적이고 혁신적인 해결책을 생성하는 접근이 요구되고 있습니다.

- **Technical Details**: 이 논문에서는 MECE(상호 배타적, 포괄적) 원칙을 사용하여 LLM과 구조적인 지침을 결합한 SELLM(해결책 열거 프레임워크)을 제안합니다. SELLM은 국제 특허 분류(IPC) 및 주기율표와 같은 구조적 지침을 활용하여 전문가 에이전트를 체계적으로 구성하고, 이를 통해 학제간의 효과적인 해결책을 생성합니다. 이것은 다양한 지식 영역을 아우르는 접근 방식으로, 복잡한 문제 해결에 대한 통합적인 해법을 제공합니다.

- **Performance Highlights**: SELLM의 실용성을 평가하기 위해 두 가지 도전 과제에 적용하였습니다: 유기 발광 다이오드(OLED) 조명의 빛 추출 개선 및 차세대 메모리 소재용 전극 개발입니다. 결과적으로, SELLM은 특정 맞춤화나 노력이 없는 경우와 비교하여 효과적인 해결책 생성을 significantly하게 촉진함을 보여주었습니다. 이 연구는 SELLM이 LLMs를 통해 어려운 문제에 대해서도 효과적인 해결책을 생성할 수 있는 잠재력을 지니고 있음을 입증합니다.



### An Empirical Evaluation of Large Language Models on Consumer Health Questions (https://arxiv.org/abs/2501.00208)
- **What's New**: 이번 연구는 MedRedQA라는 데이터셋에서 여러 Large Language Models (LLMs)의 성능을 평가합니다. 이 데이터셋은 AskDocs subreddit에서 전문가들이 검증한 소비자 기반의 의료 질문과 답변으로 구성되어 있습니다. LLM들은 임상 질문 응답(QA) 벤치마크에서 두각을 나타냈지만, 실제 소비자 질문에 대한 효과는 상대적으로 덜 이해되었습니다.

- **Technical Details**: 연구에서 사용된 모델은 GPT-4o mini, Llama 3.1: 70B, Mistral-123B, Mistral-7B, 그리고 Gemini-Flash입니다. 각 모델이 자가 평가를 진행하고 다른 모델의 응답도 평가하는 크로스-이밸류에이션(cross-evaluation) 방법이 사용되었습니다. MedRedQA는 비공식적인 언어와 비전문가 질문에 적합한 정확한 응답 필요성이라는 독특한 도전 과제를 제공합니다.

- **Performance Highlights**: 연구 결과, GPT-4o mini가 다섯 개 모델의 심사자 중 네 명의 전문가 응답과 가장 높은 일치를 보였습니다. 반면에 Mistral-7B는 세 모델의 심사자들에게 가장 낮은 점수를 기록했습니다. 이 연구는 소비자 건강 의료 질문 응답에 대한 현재 LLM의 잠재력과 한계를 강조하며, 추가 개발의 방향성을 제시합니다.



### GPT-4 on Clinic Depression Assessment: An LLM-Based Pilot Study (https://arxiv.org/abs/2501.00199)
- **What's New**: 이번 연구는 전 세계적으로 널리 퍼진 우울증을 조기에 탐지하기 위한 혁신적인 접근법으로 GPT-4를 활용한 임상 우울증 평가 방법을 제안합니다. 또한 인터뷰 전사 분석을 통해 AI가 인간 전문가의 진단 능력을 어떻게 모방할 수 있는지를 탐구합니다. 이 연구는 단순한 프롬프트를 넘어 다양한 프롬프트 구조와 온도 조정이 모델의 정확성과 일관성에 미치는 영향을 조사합니다. 이를 통해 정신 건강 진료에 AI의 활용 기준을 새롭게 설정하고자 합니다.

- **Technical Details**: 연구는 인터뷰 전사 데이터를 바탕으로 하여 GPT-4의 분류 능력을 테스트합니다. 실험은 기본 프롬프트를 사용한 이진 분류에서 시작하여 예시를 추가하고 임상적인 맥락을 포함한 복잡한 프롬프트로 접근합니다. 마지막으로 온도 조정을 통해 정확도 및 F1-Score 최적화 효과를 분석합니다. 이러한 프롬프트 공학 과정은 모델의 분류 성능을 향상시키는 핵심 요소로 작용합니다.

- **Performance Highlights**: 연구 결과, GPT-4는 다양한 구성에서의 정확도와 F1-Score에서 상당한 변동성을 보였으며, 복잡한 프롬프트에서 낮은 온도 값(0.0-0.2)에서 최적 성능을 발휘했습니다. 하지만 온도 값이 0.3 이상으로 증가할 경우 성능과 임의성의 관계가 예측 불가능해져서 성과가 감소했습니다. 이러한 결과들은 GPT-4가 임상 진단에 대한 잠재력을 보여주지만, 프롬프트 및 모델 파라미터의 세심한 조정이 필요하다는 것을 시사합니다.



### The Text Classification Pipeline: Starting Shallow going Deeper (https://arxiv.org/abs/2501.00174)
- **What's New**: 이번 논문은 Text Classification (TC)의 전체 파이프라인을 상세히 탐구합니다. 특히 각 구성 요소가 TC 모델의 성능에 미치는 영향에 대해 철저하게 검토하였습니다. 더불어 최신 데이터셋과 텍스트 전처리 기법을 포함한 다양한 기술 혁신을 소개합니다.

- **Technical Details**: 논문은 TC의 다양한 단계—데이터셋, 전처리 기법, 텍스트 표현 방법, 분류 모델, 평가 메트릭, 결과 및 미래 동향 등을 포괄적으로 다룹니다. 각 장에서는 이론과 함께 실험적 평가 및 사례 연구를 제공하여 보다 깊은 이해를 돕습니다. 이러한 기술적 디테일은 TC의 효과적인 구현에 매우 중요합니다.

- **Performance Highlights**: 분류 전략에 대한 비판적 평가와 비교 분석을 통해 독자에게 다양한 접근 방법의 강점과 약점을 인식시킵니다. 이 연구는 단순한 조사를 넘어서 TC 분야에서의 중요한 최근 발견을 조명하며, 향후 연구 방향에 대한 통찰을 제공합니다. 결과적으로, 이 논문은 TC의 전문성과 이해도를 높이는 데 기여하고 있습니다.



### Measuring Large Language Models Capacity to Annotate Journalistic Sourcing (https://arxiv.org/abs/2501.00164)
- **What's New**: 이 논문은 ChatGPT 출시 이후 큰 언어 모델(Large Language Models, LLMs)의 평가와 관련된 새로운 시각을 제시합니다. 특히 저널리즘 영역에서 소스 및 윤리적 측면에 대한 충분한 연구가 이루어지지 않았음을 지적하며, 저널리즘이 민주주의에서 중요한 역할을 한다고 강조합니다. 이 연구는 LLM의 성능 평가를 위한 실험적인 시나리오를 구축하여 소스 식별 및 주석 달기 과정을 평가하고자 합니다.

- **Technical Details**: 연구에서는 저널리즘 연구(Gans, 2004)에서 영감을 받아 다섯 개 카테고리 체계를 구성하여 뉴스 스토리 내 소스를 식별하는 작업을 설정합니다. 데이터 세트와 성과 지표를 제공하며, 이 평가 작업은 체계적 벤치마킹으로 나아가는 첫 걸음을 제시합니다. LLM 기반 접근 방식이 이야기에서 모든 소스된 진술을 식별하는 데 있어 더 많은 발전이 필요하다는 정확성(findings) 결과도 포함되어 있습니다.

- **Performance Highlights**: 본 연구의 발견에 따르면, LLM은 모든 소스된 진술을 식별하고 유형에 맞게 매칭하는 데 있어서 부족함이 있으며, 특히 소스 정당성을 식별하는 작업은 더욱 어렵습니다. 이로 인해 저널리즘에서 더욱 투명하고 윤리적으로 엄격한 형태의 시스템 구축의 가능성이 열립니다. 이는 저널리즘의 진실성 및 윤리적 기준 유지에 기여할 수 있는 중요한 연구라고 할 수 있습니다.



### Temporal reasoning for timeline summarisation in social media (https://arxiv.org/abs/2501.00152)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 시간적 추론 능력을 향상시키는 것이 긴 텍스트의 타임라인 요약 품질을 개선할 수 있는지 조사합니다. 특히 사회적 미디어 스레드와 같은 사건의 연속을 포함하는 텍스트를 요약하는 작업을 다룹니다. 우리는 새로운 데이터 세트인 NarrativeReason을 소개하여 사건 간의 시간적 관계를 중점적으로 다루며, 기존의 데이터 세트와의 차별점을 보입니다.

- **Technical Details**: 연구에서는 사건의 시간적 관계를 효과적으로 처리하기 위해 LLM을 시간적 추론 임무에 맞게 미세 조정(fine-tuning)합니다. 이러한 튜닝을 통해 교사 모델을 만들고, 이후 이를 학생 모델로 지식 증류(knowledge distillation)하여 타임라인 요약 작업을 수행하게 합니다. 세 가지 전략인 Neuron Selectivity Transfer, Contrastive Representation Distillation, Probabilistic Knowledge Transfer를 통해 지식이 전달됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 정신 건강과 관련된 긴 사회적 미디어 스레드를 요약하는 작업에서 뛰어난 성능을 보여주며, 신뢰할 수 있는 타임라인 요약을 생성합니다. 이 모델은 정확한 요약을 생성할 뿐만 아니라 LLM의 환각(hallucination) 현상을 줄이는 데에도 기여했습니다. 이는 시간적 추론을 활용하는 것이 타임라인 요약의 품질을 향상시킬 수 있다는 것을 보여줍니다.



### A Data-Centric Approach to Detecting and Mitigating Demographic Bias in Pediatric Mental Health Text: A Case Study in Anxiety Detection (https://arxiv.org/abs/2501.00129)
- **What's New**: 이 연구는 어린이 정신 건강 스크리닝을 지원하는 AI 모델의 훈련 데이터에서 비생물학적 차이에 따른 언어적 차이를 탐지하고 완화하는 방법을 제시합니다. 이는 기존의 구조화된 데이터에 대한 편향 문제를 넘어 비구조화된 데이터에서의 편향 문제에 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 성별 하위 그룹 간의 결과 동등성을 평가하고, 성 중립적인 용어로 편향된 용어를 중화하는 데이터 중심의 디바이싱(de-biasing) 방법을 적용했습니다. 이 접근법은 소아환자의 자동 불안 감지 모델에서 테스트되었으며, 성별에 따른 정보 밀도와 언어적 차이가 진단 정확도에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 여성 청소년 환자의 체계적인 과소 진단이 발견되었고, 남성 환자 대비 4% 낮은 정확도와 9% 높은 허위 음성률(FNR)을 기록했습니다. 디바이싱 방법을 통해 진단 편향이 최대 27% 감소했으며, 이는 인구 집단 간의 평등 성 향상에 효과적임을 보여주었습니다.



### CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions (https://arxiv.org/abs/2501.00097)
- **What's New**: 이 논문은 법률 분야에서 긴 문맥 요약에 대한 새로운 데이터셋인 CaseSumm를 소개합니다. 이 데이터셋은 25,600개의 미국 대법원(SCOTUS) 의견과 공식 요약인 'syllabuses'를 포함하고 있으며, 1815년부터의 SCOTUS 판결 요약을 포함한 최초의 데이터셋입니다. CaseSumm는 현재 가장 큰 공개 법률 사건 요약 데이터셋으로, 연구 커뮤니티에 중요한 자원으로 제공됩니다.

- **Technical Details**: CaseSumm 데이터셋은 법률 문서 요약에서의 다양한 도전 과제를 해결하기 위해 설계되었습니다. 데이터셋은 공식적으로 승인된 시라버스를 포함하여 미국 대법원 판결을 기반으로 하며, 인간 평가와 자동 메트릭을 모두 통해 요약 성능을 종합적으로 평가합니다. 특히, Mistral 7b 모델이 자동 메트릭에서 우수한 성능을 보이지만 전문가의 주관적 평가에서는 환각(hallucination) 문제로 인해 낮은 점수를 받았음을 지적합니다.

- **Performance Highlights**: 연구 결과에 따르면, 대부분의 자동 메트릭에서 Mistral가 대형 모델들을 초월하는 것으로 나타났지만, 인간 전문가들은 GPT-4의 요약을 더 명확하며 중요한 정보에 대한 민감성과 정확성을 더 잘 나타낸다고 평가합니다. 요약에서 발생하는 특정 환각 유형과 사실 오류도 분석하였으며, 이는 법률 요약에서의 자동 메트릭 평가의 한계를 보여줍니다. CaseSumm는 법률 요약 품질 평가에서 인간 전문가의 역할이 얼마나 중요한지를 강조합니다.



### Position Information Emerges in Causal Transformers Without Positional Encodings via Similarity of Nearby Embeddings (https://arxiv.org/abs/2501.00073)
Comments:
          Forthcoming at the International Conference on Computational Linguistics 2025 (COLING 2025)

- **What's New**: 이번 논문에서는 Transformer 모델에서 positional encodings 없이도 positional information을 저장할 수 있는 새로운 가설을 제안합니다. 연구자들은 인근 임베딩(embeddding)이 서로 더 유사하다는 사실을 관찰하고 이것이 token의 위치를 재구성하는 데 도움이 될 수 있음을 보여줍니다. 이 패턴은 causal attention을 사용하는 훈련된 및 초기화된 Transformer 모델 모두에서 발생할 수 있음을 확인하였습니다.

- **Technical Details**: Transformer는 causal attention 메커니즘을 사용하여 입력 token의 순서를 고려할 수 있습니다. 연구에서는 인근 인덱스에 있는 임베딩들이 코사인 유사성(cosine similarity)의 관점에서 서로 더 유사하다는 특성을 통해 positional information이 저장될 수 있는 방법을 제안합니다. 이에 대한 이론적 관측을 제공하고, 다양한 구성의 데이터로 실험을 통해 이 패턴의 일관성을 확인하였습니다.

- **Performance Highlights**: 실험 결과는 causal attention을 사용하는 다양한 설정에서 인근 임베딩 사이의 유사성이 높게 나타난다는 것을 보여줍니다. 특히, 여러 가지 합성 작업(synthetic tasks)을 통해 position의 중요성을 입증하였으며, 훈련된 모델과 훈련되지 않은 모델 모두에서 이 패턴이 관찰되었습니다. 본 연구는 기존 이론의 한계를 지적하고, 다양한 Transformer 계층(layer)에서 positional information의 저장 정도를 탐색하였습니다.



### ICLR: In-Context Learning of Representations (https://arxiv.org/abs/2501.00070)
Comments:
          Preprint (Under Review)

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 사전 훈련(pretraining) 데이터에 기반하여 개념의 표현을 구성하는 방식을 보여줍니다. 학습된 개념의 맥락(context)이 새롭고 다르다면, 이러한 개념 표현이 어떻게 재구성되는지를 탐구합니다. 특히, '그래프 추적(graph tracing)'이라는 간단한 작업을 정의하여, 모델이 맥락 특정 의미를 반영하는 방식으로 표현을 조정할 수 있는지를 조사합니다.

- **Technical Details**: 연구에서는 Llama3.1-8B 모델을 주로 사용하며, 다양한 구조(예: 사각형 격자, 원형, 육각형 격자)에서 랜덤 워크(random walks)에 대한 실험을 진행합니다. 그래프의 노드는 과거 학습에서 잘 알려진 개념으로 참조되며, 연결 구조는 임의로 설정됩니다. 이 과정을 통해 입력된 맥락에 따라 모델이 개념 표현을 어떻게 변화시키는지를 분석합니다.

- **Performance Highlights**: 결과적으로, 맥락의 크기가 커질수록 모델 표현의 구조가 그래프 연결성과 일치하도록 재조직되는 현상을 관찰했습니다. 흥미롭게도 이러한 결과는 유사한 설정에서 인간 피험자에게서 나타난 결과와 유사하여, LLM이 특정 맥락에 따라 의미를 재구성할 수 있는 능력을 가지고 있음을 시사합니다. 이 연구는 맥락 학습(in-context learning)이 본질적으로 최적화 과정과 관련이 있음을 제시합니다.



### Adversarial Negotiation Dynamics in Generative Language Models (https://arxiv.org/abs/2501.00069)
Comments:
          Paper at NeurIPS 2024 Workshop on Red Teaming GenAI

- **What's New**: 이번 연구에서는 계약 작성 및 개선에 사용되는 생성적 언어 모델의 성능과 취약성을 평가하여 AI 안전과 보안에 대한 중요성을 강조합니다. 경쟁하는 양 당사자가 서로 다른 언어 모델을 사용하면서 발생하는 게임 이론적 문제와 AI 안전성 문제를 다루고 있습니다. 이를 통해 더 안전하고 신뢰할 수 있는 모델의 개발에 기여하고자 합니다.

- **Technical Details**: 우리는 8개의 생성적 언어 모델을 사용하여 계약 협상 시나리오에서의 행동을 조사했습니다. 각 모델은 판매자 또는 구매자의 역할을 맡아 상상의 두 기업 간 100대 머신 판매 계약을 생성하고, 서로 교환하며 협상하는 방식을 채택했습니다. 평가 과정에서는 잇따라 6개의 모델이 최종 계약을 평가하며, 이 투표를 통해 계약의 유리함을 판별했습니다.

- **Performance Highlights**: 연구 결과는 일반 모델과 전문 법률 모델 간의 협상 행동에서 뚜렷한 차이를 보여주고 있습니다. 일반 모델은 각각의 역할에서의 적응력이 뛰어난 반면, 법률 전문 모델은 공정성을 고려한 균형 잡힌 성과를 보였습니다. 모델 선택이 협상 결과에 큰 영향을 미칠 수 있음을 강조하며, 특정 상황에 맞는 모델을 전략적으로 배치하는 것이 중요하다고 결론지었습니다.



### On Adversarial Robustness of Language Models in Transfer Learning (https://arxiv.org/abs/2501.00066)
- **What's New**: 본 연구는 LLMs(대형 언어 모델)의 전이 학습(transfer learning) 상황에서의 적대적 강인성(adversarial robustness)을 조사합니다. 다양한 데이터셋(MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias)과 모델 아키텍처(BERT, RoBERTa, GPT-2, Gemma, Phi)를 활용한 실험을 통해, 전이 학습이 표준 성능 지표를 향상시키는 반면, 적대적 공격에 대한 취약성을 증가시킨다는 사실을 밝혀냈습니다. 이 연구는 모델 크기, 아키텍처, 적응 방법 간의 복잡한 상호 작용을 보여줍니다.

- **Technical Details**: 이 연구는 편향된 텍스트 탐지라는 분류 작업에 중점을 두고 있으며, 각 데이터셋은 편향(biased) 및 비편향(non-biased)이라는 두 개의 클래스가 균형 있게 구성되어 있습니다. 성능과 강인성을 평가하기 위해 Original Accuracy (OAcc), Attack Success Rate (ASR), Accuracy Under Attack (AUA) 등의 지표를 사용했습니다. 또한, 전이 학습을 통한 파인튜닝(fine-tuning)과 적대적 훈련(adversarial training) 기법을 통해 강인성 문제를 평가했습니다.

- **Performance Highlights**: 실험 결과, 특히 소규모 모델에서 공격 성공률(ASR)이 전이 학습 후 증가하는 경향이 관찰되었습니다. 예를 들어, Hate Speech 데이터셋에서 GPT-2는 ASR이 평균 20.4% 증가하면서 정확도는 3.67% 상승했습니다. 이러한 결과는 성능 향상이 강력성 감소와 관련된 중요한 요소를 간과하게 할 수 있다는 우려를 제기합니다.



### ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis (https://arxiv.org/abs/2501.00062)
Comments:
          16 pages, 4 figures. Source code and data available at this https URL

- **What's New**: 본 연구에서는 ELECTRA와 GPT-4o 모델을 결합하여 3가지 감정 분류(부정, 중립, 긍정)를 수행하는 새로운 협업 방식을 탐구합니다. 특히, fine-tuned된 ELECTRA 모델의 예측 결과를 GPT 모델에 제공함으로써 분류 성능을 개선할 수 있는지에 관한 가설을 제시하였습니다. 연구 결과, 이러한 접근 방식이 단독 모델보다 우수한 성능을 보여주었으며, 비용 대비 성과(Cost/Performance Ratio)에서도 개선을 이루었습니다.

- **Technical Details**: ELECTRA와 GPT-4o/4o-mini 모델을 사용하여 Stanford Sentiment Treebank(SST)와 DynaSent 리뷰 데이터를 통합해 fine-tuning(FT)을 수행했습니다. 실험에서는 예측 클래스 레이블, 클래스별 확률, 유사한 리뷰 예시를 포함하여 다양한 프롬프트 증강 방안을 적용했습니다. 분석 결과, ELECTRA Base의 예측 결과를 공유함으로써 GPT-4o-mini의 성능이 현저히 향상되었으며, 이는 각각의 모델 단독 사용 시와 비교하여 우수한 결과를 보여주었습니다.

- **Performance Highlights**: ELECTRA Large FT 모델은 base GPT 모델보다 뛰어난 성능을 보였으며, GPT-4o FT-M과 GPT-4o-mini FT 모델도 각각 86.99와 86.77의 성과를 기록했습니다. 특히 GPT-4o-mini FT 모델은 76% 낮은 비용으로 GPT-4o FT 모델에 필적하는 성능을 달성했습니다. 이러한 결과는 리소스가 제한된 프로젝트에 있어 경제적인 대안을 제공하며, fine-tuned된 ELECTRA 모델과 예측 결과를 활용한 LLM 프롬프트 증강이 성능 향상에 기여한다고 보여집니다.



### Large Language Models for Mathematical Analysis (https://arxiv.org/abs/2501.00059)
- **What's New**: 본 연구에서는 DEMI-MathAnalysis 데이터셋을 개발하여 수학적 분석의 증명 기반 문제에 중점을 두었습니다. 기존의 수학 데이터셋은 주로 계산 작업에 포커스를 맞추었고, 정형화된 수학적 언어를 다루는 AI의 능력을 평가하는 등격 차이가 있었습니다. 이를 통해 LLM들이 로직적이고 완전하며 우아한 증명을 생성하는 능력이 향상되었습니다.

- **Technical Details**: DEMI-MathAnalysis 데이터셋은 수학적 분석 주제인 수열, 극한, 무한급수 및 볼록 함수 등의 증명 기반 문제로 구성됩니다. 이러한 문제는 LaTeX 형식으로 표기되며, 포괄적인 단계별 해결책과 함께 제공됩니다. 이 데이터셋은 LLM의 fine-tuning을 통해 심화되어, 수학적 분석 문제 해결 능력을 증가시키기 위해 설계된 가이드 프레임워크와 결합하여 사용됩니다.

- **Performance Highlights**: 이 연구를 통해 LLM들은 보다 집중적으로 형식적이고 논리적인 문제 해결에 접근할 수 있게 되었으며, 특히 수학적 분석의 복잡성을 처리하는 데 있어 신뢰할 수 있는 AI로 성장할 수 있는 기반을 다졌습니다. LLM에 대한 정확하고 논리적 해결책을 평가하는 방법론도 제안되어, 수학적 문제 해결 능력이 더욱 향상되었습니다.



### Seq2Seq Model-Based Chatbot with LSTM and Attention Mechanism for Enhanced User Interaction (https://arxiv.org/abs/2501.00049)
Comments:
          The Third Workshop on Deployable AI at AAAI-2025

- **What's New**: 이 논문에서는Seq2Seq 모델을 기반으로 하여 LSTM 세포와 주의(attention) 메커니즘을 활용한 특정 관광 산업에 적합한 챗봇을 개발하는 방법을 제안합니다. 기존의 상용 API에 의존하지 않고, 모로코 Draa-Tafilalet 지역에 특화된 데이터셋에서 훈련되고 검증된 이 챗봇은 유연성과 비용 효율성을 높입니다.

- **Technical Details**: 챗봇은 인공지능(AI)을 활용하여 자연어 처리(NLP) 기술을 통해 사용자의 요청을 이해하고 응답하는 시스템입니다. 논문에서 제안된 Seq2Seq 모델은 인코더-디코더 아키텍처를 갖추고 있으며 LSTM과 주의 메커니즘을 활용하여 문맥을 잘 이해하고 응답을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 챗봇은 훈련에서 약 99.58%, 검증에서 98.03%, 테스트에서 94.12%라는 높은 정확도를 달성했습니다. 이러한 결과는 관광 분야에서 잘 적응된 유용한 응답을 제공하는 챗봇의 효과성을 입증하며, 틈새 시장에서의 사용자 경험과 만족도를 향상시킬 잠재력을 보여줍니다.



### Cross-Linguistic Examination of Machine Translation Transfer Learning (https://arxiv.org/abs/2501.00045)
- **What's New**: 이 연구는 여러 언어 가족 간 기계 번역에서 전이 학습(transfer learning)의 효과를 조사합니다. 연구는 다양한 샘플을 통한 실험으로 세미틱, 반투어, 로망스, 슬라브어 및 고립어 언어 쌍을 살펴보았습니다. 하이퍼파라미터(learning rate, batch size 등)의 변동에도 불구하고, 전이 학습의 보편성을 확인하였습니다.

- **Technical Details**: 저자들은 다양한 언어 쌍에 대해 고유의 하이퍼파라미터 세트를 사용하지 않고, 각 쌍에 대해 4-6회씩 다양한 하이퍼파라미터로 학습을 진행합니다. 특히, 학습 속도(learning rate)와 에폭 수(epoch)의 변수는 각 언어 쌍의 복잡성을 반영합니다. 이러한 변화는 학습 모델이 각기 다른 고유한 언어 패턴을 효과적으로 반영하도록 돕습니다.

- **Performance Highlights**: 연구 결과에 따르면, 전이 학습이 다양한 언어 가족 간에 효과적이라는 점이 드러났습니다. 중간 정도의 배치 크기(batch size)는 일반적으로 더 효과적이며, 지나치게 높은 학습 속도는 모델 훈련을 방해할 수 있습니다. 본 연구는 일관된 하이퍼파라미터 세팅이 다국어 모델 훈련의 효율성을 높일 수 있음을 보여주었습니다.



### Distilling Large Language Models for Efficient Clinical Information Extraction (https://arxiv.org/abs/2501.00031)
Comments:
          19 pages, 1 figure, 10 tables

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 임상 정보 추출에 대한 제한적인 계산 요구를 해결하기 위해 지식 증류(knowledge distillation) 기법을 활용하여 약 1,000배 작은 증류된 BERT 모델을 임상 이름 개체 인식(NER) 작업에서 평가합니다. 이를 통해 최신 LLMs(Gemini 및 OpenAI 모델) 및 의학적 온톨로지(RxNorm 및 SNOMED)를 사용하여 약물, 질병 및 증상을 효과적으로 추출할 수 있음을 보여줍니다. 이 연구는 3,300개 이상의 임상 노트를 활용하여 증류된 BERT 모델이 유사한 성능을 발휘하면서도 훨씬 더 낮은 비용과 빠른 속성을 갖는다는 점을 강조합니다.

- **Technical Details**: 연구에서는 트리플 서치에 따라 공개적으로 사용 가능한 비식별 데이터셋을 사용하여 실험을 실시하였으며, 각기 다른 NER 작업을 수행했습니다. 약물 추출 작업에는 n2c2 2018 Track 2 데이터셋을 사용하였고, 질병 추출을 위한 데이터셋으로는 NCBI Disease Corpus, 증상 추출에는 CORAL 데이터셋을 사용했습니다. 각 작업은 다양한 임상 문맥에서 포괄적인 검증을 위해 설계되었습니다.

- **Performance Highlights**: 증류된 BERT 모델은 F1 점수에서 약물 추출 0.87, 질병 0.84, 증상 0.68을 기록하며, 비슷한 성능을 유지하면서도 인퍼런스 속도는 최대 12배 더 빠르고 비용은 최대 101배 저렴하다는 결과를 보였습니다. 외부 검증 데이터셋에서는 약물 0.883, 질병 0.726, 증상 0.699의 F1 점수를 달성하였습니다. 이러한 성능은 증류가 대형 LLM을 대신할 수 있는 계산 효율적이고 확장 가능한 대안이 됨을 나타냅니다.



### Underutilization of Syntactic Processing by Chinese Learners of English in Comprehending English Sentences, Evidenced from Adapted Garden-Path Ambiguity Experimen (https://arxiv.org/abs/2501.00030)
Comments:
          18 pages

- **What's New**: 이번 연구는 문장을 이해하는 데 있어 구문 처리(syntactic processing)의 저하된 활용도를 강조하며, 기존 연구에서의 의미적 처리(semantic processing)에 대한 편향을 돌이켜보게 합니다. 전통적인 garden-path 실험과는 달리, 의미적으로 애매하지만 구문적으로 명확한 문장을 사용하여 연구의 새로운 목표를 설정하였습니다.

- **Technical Details**: 이 연구는 140명의 참가자를 대상으로 실시된 실험으로, SPSS, Graph Pad Prism, Cursor를 이용한 설명적(descriptive) 및 추론적(inferential) 통계 분석을 통해 결과를 도출하였습니다. 구문 처리의 활용 저하를 부분적(partial) 및 전체적(complete)로 구분하고, 시도와 오류(trial and error) 과정을 통해 구문 처리가 어떻게 이루어지는지를 명확히 했습니다.

- **Performance Highlights**: 이 연구는 중국인 영어 학습자들이 영어 문장을 이해하는 과정에서 구문 처리를 부족하게 활용하고 있음을 입증하였습니다. 이러한 결과는 향후 구문 처리의 통합을 통해 영어 문장 이해를 향상시킬 새로운 구문 방법 개발의 기초가 될 것입니다.



### A Breadth-First Catalog of Text Processing, Speech Processing and Multimodal Research in South Asian Languages (https://arxiv.org/abs/2501.00029)
- **What's New**: 본 연구는 최근 2022년 1월부터 2024년 10월까지의 남아시아 언어에 대한 문헌을 검토하고, 21개의 낮은 자원(low-resource) 언어에 대한 집중 분석을 제공합니다. 이 논문은 텍스트 기반 언어 처리, 다중 양식(multimodal) 모델 및 음성 처리에 관한 최신 동향, 문제 및 향후 연구 방향을 식별합니다. 우리는 대규모 언어 모델(LLMs)을 활용한 적절성 분류 및 클러스터링을 포함하는 단계별 접근 방식을 사용하였습니다.

- **Technical Details**: 이 연구에서는 Google Scholar와 Publish or Perish 소프트웨어를 활용하여 논문을 검색하고 분류하였습니다. 이 과정에서 GPT-4o를 사용해 공감대 구축과 관련성 라벨을 예측하였으며, O1 모델을 통해 발견된 논문을 주제별로 그룹화하는 방식을 채택하였습니다. 결과적으로 세 가지 분야 각각에 대한 연구 결과가 표 형태로 요약되었습니다.

- **Performance Highlights**: 본 논문에서 확인된 주요 발견은 변화하는 언어 모델의 경향과 문제점들이며, 저자들은 기계 번역, 감정 분석, 편향과 공정성 연구 등 여러 주제를 다루었습니다. 남아시아 지역의 언어적 편향과 차별성을 반영하지 못한 선행 연구의 한계를 강조하며, 향후 연구를 위한 데이터 세트 및 감정 분석의 필요성을 지적합니다. 부문 간 협업과 지식 공유를 통해 더욱 발전된 언어 기술이 필요하다고 결론짓습니다.



### Unifying Specialized Visual Encoders for Video Language Models (https://arxiv.org/abs/2501.01426)
Comments:
          Project page: this https URL

- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전은 비디오 영역에서도 정교한 추론 능력을 선보였습니다. 이에 따라 Video Large Language Models(VideoLLMs)가 등장했지만, 현재 VideoLLMs는 단일 비전 인코더를 사용하여 모든 비주얼 처리(visual processing)를 수행함으로써 전달할 수 있는 비주얼 정보의 양과 종류에 제한이 있습니다. 본 연구에서는 다중 인코더 표현(Multi-Encoder Representation of Videos, MERV)을 소개하여, 여러 개의 동결된 비전 인코더를 활용하여 비디오의 통합 표현을 생성함으로써 VideoLLM에 보다 포괄적인 전문 비주얼 지식을 제공합니다.

- **Technical Details**: MERV는 각 인코더의 기능(feature)을 시공간적으로 정렬(spatio-temporally aligning)함으로써 보다 다양한 열려있는(open-ended) 및 다중 선택(multiple-choice) 비디오 이해 질문을 해결할 수 있게 되었고, 이전의 최상급(state-of-the-art) 작업들을 초월할 수 있었습니다. 특히, MERV는 표준 비디오 이해 벤치마크에서 Video-LLaVA에 비해 정확도에서 최대 3.7% 향상된 결과를 보였으며 Video-ChatGPT에서도 더 나은 점수를 기록했습니다. 또한, 이전에 제일 좋았던 제로샷 인식 테스트(zero-shot Perception Test) 정확도를 2.2% 개선했습니다.

- **Performance Highlights**: MERV는 기존의 단일 인코더 방법들과 비교하여 최소한의 추가 파라미터(minimal extra parameters)로 보다 빠르게 훈련(train faster)할 수 있으며, 비주얼 처리를 병렬화(parallelizing)하여 효율성을 높였습니다. 실험 결과는 MERV가 각 인코더로부터 도메인 지식을 효과적으로 포착(captures domain knowledge)할 수 있음을 질적으로 증명하였습니다. 이러한 결과들은 다중 비전 인코더를 활용한 포괄적인 비디오 이해(video understanding)에서 유망한 방향을 제시합니다.



### Embedding-based Approaches to Hyperpartisan News Detection (https://arxiv.org/abs/2501.01370)
Comments:
          5 pages, 1 figure

- **What's New**: 이 논문은 주어진 뉴스 기사가 극단적으로 편향된 하이퍼당파적(hyperpartisan)인지 여부를 판단하는 시스템을 제안합니다. 저자들은 n-grams와 감정 분석(sentiment analysis)과 같은 다양한 접근 방식을 통해 사전 훈련된 ELMo를 사용한 문장 및 문서 표현을 시도했습니다. Bidirectional LSTM을 사용하는 최상의 시스템은 하이퍼파라미터 튜닝 없이 10-fold 교차 검증을 통해 83%의 정확도를 달성했습니다.

- **Technical Details**: 하이퍼당파적 뉴스는 강한 정치적 입장을 취하며 공공의 정치적 분열을 유도하는 뉴스를 지칭합니다. 저자들은 두 가지 데이터셋을 사용하여 이진 분류 문제를 설정했으며, 하나는 수동으로 레이블링된 1,273개의 기사로 구성되고, 다른 하나는 자동으로 레이블링된 754,000개의 기사로 구성됩니다. 시스템은 다양한 기계 학습 분류기와 CNN(Convolutional Neural Networks), LSTM(Long Short Term Memory Networks) 등을 적용하여 뉴스 기사를 분석합니다.

- **Performance Highlights**: Gradient Boosting Trees는 10-fold 교차 검증을 통해 77.5%의 평균 정확도를 달성하며 가장 우수한 성능을 보였습니다. 또한, Neural Network 기반의 다양한 모델은 상당한 성능 향상을 보여주었으며, Bi-LSTM은 다른 모델보다 약간 더 나은 성능을 나타냈습니다. 결과적으로, 하이퍼당파적 뉴스 탐지에서 딥 러닝의 잠재력을 입증하였으며, 향후 연구 방향에 대한 논의도 포함되어 있습니다.



### ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding (https://arxiv.org/abs/2501.01366)
Comments:
          20 pages with 5 figures and 11 tables

- **What's New**: 이 논문에서는 3D 시나리오에서 자연어로 언급된 대상을 정확히 위치시키는 3D visual grounding (3DVG) 모델의 중요성을 강조합니다. 특히, 기존 데이터셋의 한계를 극복하기 위해 다양한 언어 패턴을 포괄할 수 있는 새로운 진단 데이터셋인 Visual Grounding with Diverse Language in 3D (ViGiL3D)를 소개합니다. 이 데이터셋은 3DVG 방법을 평가하는 데 있어 유용하고 대표적인 프롬프트 세트를 마련하는 데 기여할 것입니다.

- **Technical Details**: ViGiL3D는 다양한 자연어 프롬프트를 통해 3DVG 메서드의 능력을 테스트할 수 있는 프레임워크를 제공합니다. 연구는 기존의 오픈 바카블러리(open-vocabulary) 3DVG 방법들을 평가하여, 이러한 방법들이 더 도전적인 프롬프트를 이해하고 대상을 식별하는 데 부족함이 있음을 보여줍니다. 이를 통해, 보다 실용적인 응용 프로그램을 위해 필요한 언어적 다양성을 강조합니다.

- **Performance Highlights**: 테스트 결과, 현재의 3DVG 방법들은 다양한 언어 패턴에 대한 이해도가 낮은 것으로 나타났습니다. 더욱이 이 연구는 언어적인 성능 향상을 위해 필요로 하는 영역을 밝혀냄으로써, 3DVG 모델의 발전 방향을 제시합니다. 연구진은 이러한 데이터를 바탕으로 미래에 더 효과적이고 효율적인 3DVG 솔루션을 개발할 수 있는 가능성을 열어 놓습니다.



### AdaptVC: High Quality Voice Conversion with Adaptive Learning (https://arxiv.org/abs/2501.01347)
Comments:
          4 pages, 3 figures. Audio samples are available in the demo page: this https URL

- **What's New**: 본 논문은 음성 변환(Voice Conversion, VC) 분야에서의 새로운 접근법을 제시하고 있습니다. 기존 기술들이 언어 콘텐츠와 화자 특성을 분리하는 데 한계가 있었던 것에 반해, 본 연구에서는 Self-Supervised Learning(SSL)을 활용하여 콘텐츠와 화자 특징을 효과적으로 분리하는 방법을 구현하였습니다. AdaptVC라는 새로운 모델은 adapters를 사용하여 고급 자가 감독 음성 표현을 조정하여 명확한 음성 변환을 이끌어냅니다.

- **Technical Details**: AdaptVC 모델은 인코더-디코더 아키텍처를 기반으로 하며, HuBERT 같은 사전 훈련된 SSL 모델을 사용합니다. 인코더는 음성 콘텐츠와 화자 정보를 최대한 효과적으로 추출하기 위해 가중 합계를 이용한 adapters를 포함하고, VQ(벡터 양자화) 계층을 통해 화자 측면의 분리를 더 강화합니다. 디코더는 OT-CFM(Optimal Transport Conditional Flow Matching) 목표를 활용하여 음성 품질과 효율성을 극대화합니다.

- **Performance Highlights**: 제로샷(zero-shot) 시나리오에서 실시된 주관적 및 객관적 평가 결과, AdaptVC는 기존의 모든 음성 변환 모델들을 초월하며 높은 이해도와 목표 화자 유사성을 실현하였습니다. 또한, 방대한 데이터로부터 효율적으로 학습하여 음성 변환의 품질을 현저히 향상시킨 것으로 확인되었습니다. 녹음 샘플은 데모 페이지에서 확인할 수 있습니다: https://mm.kaist.ac.kr/projects/AdaptVC



### Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability (https://arxiv.org/abs/2501.01346)
Comments:
          16 pages, 3 figures

- **What's New**: 최근 연구는 대형 비전-언어 모델(LVLMs)의 시각 및 언어 표현 간의 정합(alignment) 문제를 다룹니다. 기존LVLM들이 이미지 캡셔닝(image captioning) 및 시각적 질문 응답(visual question answering)과 같은 기능에서 큰 진전을 이룬 가운데, 정합 메커니즘에 대한 이해가 부족하다는 점을 지적합니다. 이 설문조사는 LVLMs의 정합과 비정합(misalignment) 현상을 보다 잘 이해할 수 있는 구조적 Framework를 제공합니다.

- **Technical Details**: LVLMs에서의 정합은 다양한 층면에서 이루어집니다. 첫째, 나타나는 주요 개념으로 표현적 정합(representational alignment)과 행동적 정합(behavioral alignment)을 정의합니다. 각각은 모델의 내부 임베딩 공간에서 시각적 및 언어적 표현의 일치를 의미하며, 이러한 정합을 통해 모델이 시각 정보와 언어 정보를 의미 있게 연결할 수 있습니다. 정합 과정은 비주얼 인코더 학습, 어댑터 미세 조정(adapter fine-tuning), 그리고 최종적인 전체 시스템의 미세 조정(end-to-end fine-tuning)의 세 단계로 나뉩니다.

- **Performance Highlights**: 정합이 효과적으로 이루어질 때, LVLM은 이미지 입력에 대한 정확하고 일관된 텍스트 응답을 생성할 수 있습니다. 그러나 비정합 현상이 발생하면 모델은 이미지에 대한 잘못된 식별(object misalignment), 잘못된 설명(attribute misalignment), 관계 비정합(relational misalignment) 등의 문제를 겪게 됩니다. 이러한 문제들은 모델의 신뢰성과 성능에 지대한 영향을 미치므로, 개선 전략과 표준화된 평가 프로토콜의 필요성이 강조됩니다.



### The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation (https://arxiv.org/abs/2501.01329)
- **What's New**: 이 논문에서는 소프트웨어 테스트 케이스 생성을 위한 LLM(대형 언어 모델) 맞춤형 프롬프트를 자동으로 생성하는 MAPS라는 새로운 방법을 제안합니다. 현재까지의 연구는 LLM의 성능이 주로 사람이 작성한 프롬프트에 의존하고 있음을 밝혔으며, 서로 다른 LLM에 가장 적합한 프롬프트를 찾는 것이 필요하다는 점을 강조하고 있습니다. MAPS는 프롬프트 최적화를 위한 세 가지 주요 모듈을 통해 이 문제를 해결합니다.

- **Technical Details**: MAPS는 다양성 기반의 프롬프트 생성, 오류 기반 규칙 도출, 도메인 맥락 지식 추출이라는 세 가지 주요 모듈로 구성되어 있습니다. 다양성 기반 프롬프트 생성 모듈은 다양한 수정 경로를 탐색하여 다양한 프롬프트를 생성합니다. 오류 기반 규칙 도출 모듈은 생성된 테스트 케이스의 일반적인 오류를 반영하여 최적화 방향을 식별하고, 도메인 맥락 지식 추출 모듈은 클래스 상속 및 호출 관계와 같은 정보를 제공하여 LLM이 정확한 테스트 케이스를 생성하도록 돕습니다.

- **Performance Highlights**: 실험 결과, MAPS는 세 가지 인기 있는 LLM(예: ChatGPT, Llama-3.1, Qwen2)에 대해 기존의 최첨단 프롬프트 최적화 방법들에 비해 뛰어난 성능을 보였습니다. 평균적으로 MAPS는 6.19% 더 높은 라인 커버리지 비율과 5.03% 더 높은 분기 커버리지 비율을 달성했습니다. MAPS는 각 LLM에 가장 적합한 프롬프트를 효과적으로 생성하며, 수작업으로 설계된 프롬프트보다 우수한 결과를 보여주었습니다.



### CultureVLM: Characterizing and Improving Cultural Understanding of Vision-Language Models for over 100 Countries (https://arxiv.org/abs/2501.01282)
Comments:
          Technical report; 26 pages

- **What's New**: 이 논문에서는 문화적 이해를 개선하기 위한 대규모 멀티모달 벤치마크 CultureVerse를 구축하였습니다. 이 데이터셋은 19,682개의 문화 개념과 188개의 국가/지역이 포함되어 있어 VLM(Visual-Language Models)이 다문화적 이해 능력을 평가할 수 있도록 합니다. 또한 CultureVLM이라는 일련의 VLM을 제안하여 우리의 데이터셋에 대해 파인튜닝을 통해 문화적 이해를 크게 향상시킵니다.

- **Technical Details**: CultureVerse는 19,682개의 문화 개념, 228,053개의 샘플로 구성되어 있으며, 다양한 문화적 지원을 위해 추가 언어와 문화를 통합할 수 있는 유연한 파이프라인을 제공합니다. VLMs는 일반적으로 문화적 이해의 지역적 불균형을 보여주며, 특히 서구 개념에 더 강하고 아프리카 및 아시아 맥락에서의 성능은 낮은 것으로 나타났습니다. 파인튜닝을 통해 문화적 인식을 향상시켜 모델의 일반적인 성능을 희생하지 않고도 지역과 범주에서의 이해 격차를 줄일 수 있습니다.

- **Performance Highlights**: 모델의 크기와 데이터의 양이 문화적 이해를 증가시키는데 긍정적인 상관관계를 보였으며, 특히 Llama 3.2-11B 모델이 Qwen 2-72B와 유사한 성능을 발휘했습니다. 결과는 VLM이 다양한 문화, 개념, 대륙 및 데이터셋 간의 일반화 능력을 보여주며, 문화적 이해를 향상시키기 위한 일반화 연구의 커다란 가능성을 나타냅니다. 이번 연구는 상황이 잘 반영되지 않은 문화에 대한 AI 형평성을 증진시키는 기반을 마련할 것을 기대합니다.



### Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants (https://arxiv.org/abs/2501.01243)
Comments:
          50 pages, 14 figures, 41 tables. Submitted to ICLR 2025

- **What's New**: 이번 연구에서는 얼굴과 인간 이해 능력을 평가하기 위한 새로운 기준인 Face-Human-Bench를 제안합니다. 이 기준은 3단계의 능력 분류 체계에 기반하여 개발되었으며, 공통적으로 사용되는 데이터셋에서 수집된 900개의 개발 문제와 1800개의 테스트 문제를 포함합니다. 연구 결과는 멀티모달 대형 언어 모델(MLLMs)의 성능 차이에 대한 흥미로운 통찰을 제공합니다.

- **Technical Details**: 제안된 능력 분류 체계는 세 가지 수준으로 구성되어 있으며, Level-1에는 얼굴 이해(face understanding)과 인간 이해(human understanding)의 두 가지 관점이 포함됩니다. 각 수준에서는 인지 과정에 관한 세부 능력이 정의되어 있으며, Level-2에서는 얼굴 관련 5가지와 인간 관련 5가지 능도로 나뉘어 있습니다. 최종적으로, Face-Human-Bench는 25개 주류 MLLMs의 얼굴 및 인간 이해 능력을 종합적으로 평가하기 위한 방안을 제공합니다.

- **Performance Highlights**: Face-Human-Bench에 대한 평가 결과, 특정 MLLMs는 얻은 점수에서 상당한 차이를 보였으며, 상대적 위치가 성능에 미치는 영향도 심각하게 분석되었습니다. 특히, 심각한 시나리오에서의 깊은 가짜 탐지에서 전문 모델이 MLLMs보다 우수한 성능을 보임을 확인하여, 특정 작업에서 전문 모델의 통합이 필요함을 제안합니다. 연구 결과는 멀티모달 조수의 응답 품질을 높이기 위한 방안을 제시합니다.



### Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects (https://arxiv.org/abs/2501.01205)
- **What's New**: 이 논문에서는 다중 에이전트 대형 언어 모델(Multi-Agent LLMs)을 활용하여 공학 학생들이 수행하는 고급 설계 프로젝트(senior design projects, SDP)를 지원하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 전문가 관점을 대표하는 LLM 에이전트들이 상호작용하여 학생들이 복잡한 문제를 효과적으로 해결할 수 있도록 돕습니다. 이 접근법은 공학 교육에서 직면하는 다양한 윤리적, 사회적, 환경적 문제를 고려하고 반영하는 데 유용합니다.

- **Technical Details**: 제안된 프레임워크는 문제 정의 에이전트, 시스템 복잡성 에이전트, 윤리 및 사회적 에이전트 등의 다양한 역할을 수행하는 LLM 에이전트를 포함합니다. 이들 에이전트는 실시간으로 협력하여 인간 엔지니어 팀을 모방하는 방식으로 대화함으로써 프로세스를 촉진합니다. 이 구현은 프로프트 엔지니어링(prompt engineering) 기술을 활용하여 각 에이전트의 역할에 따른 다양한 페르소나(personas)를 개발하며, 이는 다양한 전문 지식의 융합을 통한 문제 해결을 가능하게 합니다.

- **Performance Highlights**: 이 프레임워크는 공학 교육에서 고급 설계 프로젝트에 참여하는 학생들에게 비판적 사고 및 협업 능력을 개발하는 데 기여할 것으로 기대됩니다. 평가 결과, 이 시스템은 복잡한 문제 해결에 있어 학생들이 더 혁신적이고 강력한 솔루션을 제시하도록 유도할 것이라는 점에서 긍정적인 효과를 보여줍니다. 이러한 접근법은 다중 전공, 다각적인 문제 해결을 요구하는 현대 공학 환경을 대비하는 데 필요한 훈련 방안을 제공합니다.



### MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization (https://arxiv.org/abs/2501.01108)
- **What's New**: 이 논문에서는 자기 지도 학습(self-supervised learning, SSL) 기반의 새로운 음악 표현 모델인 MuQ를 제안합니다. MuQ는 Mel Residual Vector Quantization (Mel-RVQ)을 사용하여 안정성과 효율성을 높인 음악 이해를 위한 표상 학습을 목표로 합니다. 또한, MuQ는 MuQ-MuLan이라는 음악-텍스트 결합 임베딩 모델을 통해 개선된 성능을 보여주며, 최신 기술 수준(State-of-the-art)에서 성능을 발휘합니다.

- **Technical Details**: MuQ는 랜덤 프로젝션 또는 기존의 뉴럴 코덱을 사용하는 대신 Mel-RVQ를 활용하여 음악 데이터를 처리합니다. Mel-RVQ는 선형 잔여 벡터 양자화(linear Residual Vector Quantization) 구조를 채택하여 Mel 스펙트럼의 양자화(quantization)를 수행하며, 이는 SSL 훈련의 안정성을 높이고 데이터의 양이 적음에도 불구하고 뛰어난 성과를 제공합니다. 또한, MuQ는 이터레이티브 학습을 통해 성능을 지속적으로 향상시킵니다.

- **Performance Highlights**: MuQ는 0.9K 시간의 공개 데이터로 학습했음에도 불구하고 이전의 최고 성능 모델인 MERT와 MusicFM을 초超越하는 성과를 냅니다. 특히, MuQ-MuLan은 MagnaTagATune 데이터셋에서 제로샷 음악 태깅(zero-shot music tagging) 작업에 대해 79.3의 ROC-AUC 점수를 달성하며, 기존의 SOTA 성과를 능가합니다. 이러한 결과는 MuQ가 다양한 음악 이해 작업에서 우수한 SSL 성능을 발휘하는 것을 입증합니다.



### 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining (https://arxiv.org/abs/2501.00958)
Comments:
          Under review

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)을 위한 고품질의 	extbf{multimodal textbook} 말뭉치(corpus)를 소개하고 있습니다. 기존의 이미지-텍스트 쌍 데이터에 비해, 이 데이터 세트는 풍부한 기초 지식을 제공하며, 2.5년 이상에 걸친 총 22,000시간의 강의 비디오를 수집하여 구성되었습니다. 이 데이터는 교육 비디오에서 체계적으로 수집되어 시간적 순서에 따라 이미지-텍스트 interleaved corpus로 정리되었습니다.

- **Technical Details**: 연구진은 LLM(대규모 언어 모델)이 제안한 분류 체계를 사용하여 교육 비디오를 체계적으로 수집하고, 키프레임(keyframes), 자동 음성 인식(ASR) 및 텍스트 인식(OCR) 기술을 통해 비디오에서 시각적 및 음성적 지식을 추출합니다. 이러한 과정을 거쳐 만든 multimodal textbook은 이전의 데이터 세트들보다 더 일관되고 풍부한 지식을 제공합니다. 특히, 이 데이터는 이미지와 텍스트 간의 정렬이 향상되어 있습니다.

- **Performance Highlights**: 실험 결과, 새로운 textbook을 이용해 사전 훈련(pretraining)한 VLMs는 ScienceQA 및 MathVista와 같은 지식 및 추론 집약적 작업에서 뛰어난 성능을 보였습니다. 또한, 이 VLM들은 이미지와 텍스트의 단서( cues)를 활용하여 몇 가지 경우(few-shot)에서 작업을 해결하는 데 있어 훌륭한 interleaved context awareness를 나타냅니다.



### Aligning Netlist to Source Code using SynAlign (https://arxiv.org/abs/2501.00921)
- **What's New**: 최근 칩 설계 프로세스에서는 여러 도구를 사용하여 게이트 레벨 넷리스트를 얻는 과정에서 소스 코드와의 연관성이 상실되는 문제가 발생합니다. 이를 해결하기 위해 SynAlign은 정렬(alignment) 프로세스를 자동화하여 반복 설계를 간소화하고 오버헤드를 줄이며 다양한 도구 간의 연관성을 유지합니다. 이러한 접근은 칩 설계 워크플로우의 효율성과 효과성을 향상시킵니다.

- **Technical Details**: SynAlign은 합성(synthesis) 도구가 생성한 넷리스트의 크리티컬 패스 정보를 이용하여 디자이너가 넷리스트 셀을 원래의 소스 코드로 추적할 수 없는 문제를 해결합니다. 이 도구는 컴파일러나 합성 프로세스를 변경하지 않고 포스트 최적화(netlist optimization)된 넷리스트를 원저의 소스 코드와 자동으로 정렬합니다. 또한, SynAlign의 정렬 전략은 칩 설계 주기 내내 일관된 설계 구조에 의존하여 컴파일러 흐름의 변화에도 불구하고 원본 소스 코드와 수정된 설계 간의 연관성을 유지할 수 있게 합니다.

- **Performance Highlights**: SynAlign은 설계 넷(net) 변경이 61%에 이르는 경우에도 정렬 정확도에 영향을 주지 않는 뛰어난 내성을 나타냅니다. 이는 엔지니어가 반복적인 설계 작업을 진행하면서도 설계와 소스 간의 상관성을 지속적으로 유지할 수 있음을 보여줍니다. 따라서 SynAlign은 칩 설계 시기와 전력에 대한 조기 피드백을 제공하여 디자이너들이 보다 효율적인 결정을 내릴 수 있게 지원합니다.



### AutoPresent: Designing Structured Visuals from Scratch (https://arxiv.org/abs/2501.00912)
- **What's New**: 이번 연구는 프레젠테이션 슬라이드를 자연어(NL) 명령어로부터 자동 생성하는 문제를 다룹니다. 연구자들은 7,000개의 교육 샘플과 585개의 테스트 샘플을 포함하는 SlidesBench 벤치마크를 소개하고, 이를 통해 슬라이드 생성 성능을 평가합니다. 추가로, AutoPresent라는 8B Llama 기반 모델을 개발하여 고품질 슬라이드를 생성할 수 있도록 하였습니다.

- **Technical Details**: SlidesBench는 다양한 난이도의 자연어 지시와 PPTX 형식의 슬라이드를 포함하는 310개의 슬라이드 덱에서 수집된 데이터입니다. 두 가지 평가 지표를 제공하여 생성된 슬라이드의 질을 평가하는 데 도움을 줍니다: 참조 기반 지표와 참조 없는 지표입니다. 또한, 프로그램 생성 방식을 통해 사용자 지침을 따른 슬라이드를 생성하는 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 프로그램 생성 방식이 이미지 생성 방식보다 훨씬 높은 품질의 슬라이드를 생성함을 보여주었습니다. AutoPresent 모델은 최첨단 성능을 달성하며 기존의 닫힌 소스 모델인 GPT-4o에 근접한 결과를 나타냈습니다. 연구자들은 자기 수정(iterative refinement) 과정을 통해 슬라이드 품질을 더욱 향상시킬 수 있는 가능성을 발견하였습니다.



### U-GIFT: Uncertainty-Guided Firewall for Toxic Speech in Few-Shot Scenario (https://arxiv.org/abs/2501.00907)
Comments:
          16 pages, 6 figures and 10 tables. Comments are welcome

- **What's New**: 최근 소셜 미디어의 사용 증가로 인해 사용자 생성 콘텐츠가 급증하고 있으며, 이는 독성 언어(적대적 발언, 사이버 괴롭힘 등)의 범람으로 이어지고 있습니다. 이러한 문제를 해결하기 위해, 본 연구에서는 U-GIFT라는 새로운 프레임워크를 제안하며, 이는 적은 샷(few-shot) 상황에서도 효과적인 독성 언어 탐지를 가능하게 합니다. U-GIFT는 불안정성(unreliability) 기반의 파이어월을 구현하여 레이블이 부족한 데이터에서도 높은 성능을 발휘합니다.

- **Technical Details**: U-GIFT는 Bayesian Neural Networks(BNNs)를 기반으로 하여 자가 학습(self-training) 기법을 통해 레이블이 없는 데이터에서 고품질 샘플을 자동으로 선택합니다. 이를 통해 학습 시 모델의 예측 불확실성을 평가하고, 높은 신뢰도를 가진 pseudo-label을 우선적으로 선택하여 학습에 활용합니다. 또한, 샘플 안정성 가중치(sample stability weights)를 포함한 향상된 손실 함수가 제안되며, 이는 안정적인 샘플로부터 견고한 특징을 학습하도록 모델을 유도합니다.

- **Performance Highlights**: U-GIFT는 5-shot 설정에서 기본 모델에 비해 14.92%의 성능 향상을 이루었습니다. 다양한 언어 모델에 쉽게 적응하며, 샘플 불균형 및 교차 도메인 상황에서도 견고한 성능을 보입니다. 이러한 결과는 U-GIFT가 사이버 공간에서 다양한 데이터 분포의 도전을 극복하고, 독성 언어 탐지 작업에 강력한 해결책이 될 수 있음을 입증합니다.



### Decoupling Knowledge and Reasoning in Transformers: A Modular Architecture with Generalized Cross-Attention (https://arxiv.org/abs/2501.00823)
- **What's New**: 이 논문에서는 지식과 추론을 명확히 분리하는 새로운 모듈형 Transformer 아키텍처를 제안합니다. 이 구조는 일반화된 크로스 어텐션 메커니즘을 통해 공유 지식 기반에 효과적으로 접근할 수 있도록 설계되었습니다. 기존의 Feed-Forward Network (FFN)를 새로운 메커니즘으로 대체하여 지식 검색의 효율성을 향상시키고자 합니다.

- **Technical Details**: 제안된 모듈형 Transformer는 각 층에서 E라는 공유 지식 기반을 도입하고, 이 기반에 접근하기 위한 전용 크로스 어텐션 메커니즘을 활용합니다. 기존의 FFN을 대체하여, 지식 검색과 추론을 명확히 분리함으로써 해석 가능성, 적응성 및 확장성을 크게 향상시킬 수 있습니다. FFN이 사실상 수행하는 암묵적 지식 검색을 명시적으로 표현하는 것이 핵심적인 목표입니다.

- **Performance Highlights**: 이론적 분석을 통해 기준 Transformer의 FFN이 제안된 일반화된 크로스 어텐션의 특수한 사례라는 것을 수학적으로 증명했습니다. 이 결과는 FFN의 암묵적 지식 검색 역할을 재확인하며, 외부 지식 기반과의 통합을 위한 기초를 다집니다. 이 프레임워크는 향후 연구에서 향상된 해석 가능성 및 적응성에 대한 탐구의 기반이 될 것입니다.



### SLIDE: Integrating Speech Language Model with LLM for Spontaneous Spoken Dialogue Generation (https://arxiv.org/abs/2501.00805)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 최근 진행된 연구에서는 음성 단위를 기반으로 한 "텍스트 없는" 음성 언어 모델(SLMs)이 자연스러운 음성을 생성하는 데 큰 진전을 이루었으나, 생성된 음성 샘플이 의미적 일관성을 결여하는 경우가 많았습니다. 본 논문에서는 자발적인 음성 대화 생성을 위한 SLM과 LLM 통합(SLIDE) 방법을 제안합니다. 이 방법은 LLM을 사용하여 대화의 텍스트 내용을 생성하고, 이를 음소 시퀀스로 변환한 후, SLM을 통해 음성으로 변환하여 높은 의미적 일관성을 유지하며 자연스러운 대화를 생성합니다.

- **Technical Details**: SLIDE 모델은 텍스트 대화 생성, 음소 시퀀스 지속 시간 예측, 음성 대화 생성의 세 가지 주요 부분으로 구성됩니다. LLM을 통해 생성된 대화에서는 "예," "맞아," "좋아"와 같은 짧은 발화를 포함하며, 음소 지속 시간 예측을 위해 두 개의 타워 변환기 모델을 사용합니다. dGSLM은 SLM의 조건에서 음소 시퀀스를 통해 자연스러운 대화 이벤트를 통합하도록 설계되어 있습니다.

- **Performance Highlights**: Fisher 데이터셋에서 수행한 실험 결과, SLIDE 모델은 자연스러운 음성 대화를 생성하며 높은 의미적 일관성을 유지하는 능력을 입증했습니다. 이 시스템은 또한 비언어적 발화와 관련된 정보를 보존하면서 대화의 유창성을 유지하는 데 성공하였습니다. 이는 사용자와의 소통에서 더욱 매력적이고 진정한 대화를 생성하는 데 기여할 것으로 기대됩니다.



### Automatic Text Pronunciation Correlation Generation and Application for Contextual Biasing (https://arxiv.org/abs/2501.00804)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 논문에서는 전통적으로 수작업으로 설계된 발음 렉시콘의 의존 없이 발음 상관관계를 자동으로 획득할 수 있는 데이터 기반 접근 방식인 Automatic Text Pronunciation Correlation (ATPC)을 제안합니다. 기존의 자동 음성 인식 시스템(E2E-ASR)에서 요구되는 감독 방식과 일치하는 이 방법은 음성과 텍스트 주석 쌍을 활용합니다. 이는 발음의 미세 변화를 구별하는 데 있어 중요한 진전을 나타냅니다.

- **Technical Details**: ATPC 생성 과정은 크게 세 단계로 구성됩니다. 첫 번째 단계는 ITSE(Iteratively-trained Timestamp Estimator) 알고리즘을 사용하여 텍스트 기호와 해당 음성을 정렬하는 것입니다. 두 번째 단계에서는 멀티링궐 스피치 표현 모델을 통해 음성을 벡터화하여 추출하고, 마지막으로 다양한 텍스트 기호의 음성 임베딩 거리 비교를 통해 발음 상관관계를 계산합니다.

- **Performance Highlights**: 실험 결과, ATPC는 만다린어의 문맥적 편향 조정(contextual biasing)에서 E2E-ASR 성능을 향상시킨 것으로 나타났습니다. ATPC 방법은 문맥적 편향을 이해하는 데 있어 증가된 정확성과 신뢰성을 제공하여, 인공 발음 렉시콘이 부족한 방언이나 언어에서도 유용하게 작용할 가능성이 확인되었습니다. 이는 최신 E2E-ASR 시스템에 통합될 수 있는 유망한 방향을 제시합니다.



### Adjoint sharding for very long context training of state space models (https://arxiv.org/abs/2501.00692)
- **What's New**: 본 논문에서는 매우 긴 컨텍스트를 가진 대형 언어 모델(LLMs)을 효율적으로 훈련할 수 있는 새로운 기술인 adjoint sharding을 제안합니다. 기존 방법들은 짧은 컨텍스트에서 훈련을 수행하고 긴 컨텍스트에서는 추론 기술을 이용했지만, 이 접근법은 GPU 메모리 용량과 긴 훈련 시간에 제약을 받습니다. Adjoint sharding은 메모리 요구 사항을 극적으로 줄이면서 긴 컨텍스트에서의 훈련을 가능하게 만들어, 사실 추출 및 요약과 같은 다양한 실제 작업에 대응합니다.

- **Technical Details**: Adjoint sharding은 그래디언트 계산을 분해하여 메모리 사용량을 크게 줄이는 방법입니다. 이는 반복적인 모델의 경우 adjoint 방법을 기반으로 하며, 그라디언트를 효율적으로 계산하기 위한 독립적인 벡터-야코비안 곱(VJP) 계산을 포함합니다. 또한, 성능을 유지하면서 알고리즘 속도를 높이기 위한 트렁케이티드(Truncated) adjoint sharding도 제안했습니다.

- **Performance Highlights**: 실험 결과에 따르면, adjoint sharding 알고리즘은 1M 컨텍스트 길이 훈련 시 1.27B 파라미터의 대형 언어 모델에서 메모리 사용량을 최대 3배 감소시켰습니다. 이로 인해 훈련 인프라에서 35K 토큰에서 100K 토큰 이상의 최대 컨텍스트 길이를 증가시킬 수 있습니다. 이러한 결과는 adjoint sharding의 분산 및 병렬 버전으로 훈련 속도를 더욱 개선할 수 있음을 보여줍니다.



### IGC: Integrating a Gated Calculator into an LLM to Solve Arithmetic Tasks Reliably and Efficiently (https://arxiv.org/abs/2501.00684)
- **What's New**: 이번 논문에서는 LLMs가 산술 작업을 효과적으로 수행할 수 있도록 돕는 새로운 모듈인 Integrated Gated Calculator (IGC)를 제안합니다. 이 모듈은 GPU에서 계산기를 모방하여 LLM의 계산 능력을 향상시키며, 여기서는 기존의 Llama 모델을 파인튜닝하고 BigBench Arithmetic 벤치마크에서 SOTA 모델들을 능가하는 성능을 보여줍니다. 이 접근 방식은 외부 도구를 요구하지 않고 단일 반복 내에 작동하며, 중간 토큰을 생성할 필요 없이 LLM 내부에서 산술 작업을 처리합니다.

- **Technical Details**: IGC는 LLM의 기존 레이어 출력을 수정하는 새로운 모듈로, 기존 가중치를 고정한 상태에서 IGC의 가중치만 학습합니다. 이는 Adapter 기반의 조정 방법과 유사하지만, 여러 가지 중요한 차이점이 존재합니다. 비확률적(non-differentiable) 성격을 가지며, 다수의 토큰에서 동시에 작업하고, 이산(discrete) 단계로 실행되며, 출력에 게이트 연결을 사용합니다.

- **Performance Highlights**: BigBench Arithmetic 벤치마크에서 IGC는 98%에서 99%의 정확도를 달성하여 모든 하위 작업을 성공적으로 해결했습니다. 특히 이 모듈은 기존에 해결되지 않던 곱셈 작업을 외부 도구 없이도 성공적으로 수행할 수 있는 능력을 지니고 있습니다. 이 연구는 LLMs가 산술 작업을 직접 해결할 수 있는 가능성을 제시하며, 더 복잡한 문제에 대한 하위 루틴으로서의 활용 가능성을 보여줍니다.



### Titans: Learning to Memorize at Test Tim (https://arxiv.org/abs/2501.00663)
- **What's New**: 이 논문은 최신 Neural Long-Term Memory 모듈을 제안하며, 이는 과거의 문맥을 기억하고 최신 문맥을 처리하는 데 도움을 줍니다. 특히, 이 메모리 모듈은 빠른 병렬 처리와 신속한 추론을 가능하게 하며, Transformer와 최근의 선형 순환 모델에 비해 우수한 성능을 보여줍니다.

- **Technical Details**: Titan이라는 새로운 아키텍처 패밀리를 도입하고, 이를 통해 메모리를 효과적으로 통합하는 세 가지 변형을 발표합니다. 이 모델은 단기 메모리로서의 주의(attention)와 장기 메모리로서의 신경 메모리(neural memory) 사이의 균형을 탐구합니다. 이론적으로 메모리를 업데이트하는 구조와 메커니즘을 재구성함으로써 메모리 학습을 최적화하려합니다.

- **Performance Highlights**: 실험 결과, Titan 모델은 언어 모델링, 상식 추론, 유전체학(genomics), 시계열(time series) 작업에서 Transformer와 현대의 선형 순환 모델보다 더 효과적인 성능을 보였습니다. 특히, Titan은 작업의 맥락 범위를 2M 이상으로 확장할 수 있으며, 'needle-in-haystack' 작업에서 baseline에 비해 높은 정확도를 달성했습니다.



### Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph (https://arxiv.org/abs/2501.00659)
- **What's New**: 이 논문에서는 다층 오토 회귀(autoregressive) Transformer 언어 모델이 명시적인 위치 인코딩(positional encodings, PEs)을 요구하지 않음을 밝혔습니다. 특히, 아키텍처가 하나 이상의 레이어로 구성된 경우, 토큰 순서를 구별할 수 있는 능력을 갖고 있다는 것입니다. 이는 GPT-2의 초기 연구와 관련하여 알려졌던 사실이지만, 최근까지 잘 알려지지 않았고 또한 재발견된 측면이 있습니다.

- **Technical Details**: Transformer 언어 모델에서, 명시적인 위치 인코딩은 인코더의 피드포워드 블록과는 달리, 셀프 어텐션(self-attention) 레이어에서 시퀀스 처리를 담당합니다. 이 연구는 다층 오토 회귀 Transformer 모델이 PEs 없이도 입력 시퀀스를 처리할 수 있는 능력을 강조하며, 이는 단일 레이어 모델에서는 적용되지 않습니다. 저자들은 관련 연구를 추적함으로써 이러한 도메인 지식이 올바르게 이해되는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 이 논문은 다층 오토 회귀 Transformer 모델이 특별한 위치 인코딩 없이도 시퀀스의 순서를 인식하고 처리할 수 있음을 강조하여, 수년 동안 잊혀졌던 중요한 결과를 재조명하고 있습니다. 다층 구조의 장점을 대중에게 환기시키며, 이전 언어 모델 연구에서의 흔적을 되살리려는 노력이 돋보입니다.



### ICONS: Influence Consensus for Vision-Language Data Selection (https://arxiv.org/abs/2501.00654)
Comments:
          25 pages, 19 figures

- **What's New**: 이번 연구에서는 단순하고 효과적인 다중 작업 비전-언어 데이터 선택 방법인 ICONS(Influence CONsensus vision-language data Selection)를 제안합니다. 주로 사용할 대규모 데이터셋을 효과적으로 관리하기 위해 샘플의 교차 작업 영향을 고려하여 우수한 데이터를 선정하는 것이 중요한 과제입니다. ICONS는 효율적인 실험을 통해 모델의 성능을 극대화할 수 있는 간결한 훈련 데이터 세트를 제공합니다.

- **Technical Details**: ICONS는 두 단계로 구성된 데이터 선택 프레임워크를 사용합니다. 첫 번째 단계에서는 각 작업에 대한 영향 점수를 계산하고, 두 번째 단계에서는 투표 기반 집계를 통해 작업 간의 합의를 구축하여 데이터 샘플을 선택합니다. 이 과정에서 각 샘플이 여러 작업에서 일관되게 유용한지를 평가하여, 특정 작업에 국한되지 않고 널리 가치 있는 샘플을 발굴합니다.

- **Performance Highlights**: LLaVA-ICONS-133K 데이터셋은 전체 데이터의 20%로, 98.6%의 성능을 유지합니다. 또한, 무작위 선택된 데이터셋과 비교했을 때 2.8% 성능 향상을 보여주며, 작업 간의 높은 이전 가능성을 입증합니다. ICONS는 성능 기준을 초과하여 다중 작업에서 우수한 성능을 지속적으로 유지할 수 있는 데이터 샘플을 선택하는 효율적인 방법으로 자리잡고 있습니다.



### MCP-Solver: Integrating Language Models with Constraint Programming Systems (https://arxiv.org/abs/2501.00539)
- **What's New**: 이번 논문은 Large Language Models (LLMs)과 constraint programming 시스템 간의 체계적 통합을 위해 Model Context Protocol (MCP)을 기반으로 한 MCP-Solver의 프로토타입 구현을 소개합니다. 이 시스템은 자연어 사양을 정형 constraint 모델로 변환할 수 있는 정밀한 인터페이스를 제공하여, LLM의 자연어 이해와 제약 해결 능력을 효과적으로 결합할 수 있는 가능성을 보여줍니다. 이 논문은 또한 오픈 소스로 구현된 MCP-Solver의 실용성을 입증합니다.

- **Technical Details**: MCP-Solver는 Python 3.9 이상의 환경에서 작동하며, MiniZinc와 Chuffed solver를 사용하여 제약 모델링을 수행합니다. 시스템은 자연어를 통한 상호작용을 지원하고, 모델 일관성을 유지하는 점검 단계를 통해 각 변경 후 모델의 정확성을 보장합니다. 또한, 시스템은 이중 형식으로 LLM의 작업 결과를 처리하기 위해 시간을 잘 관리하며, 지식 기반을 구축하여 모델링 인사이트를 지속적으로 유지합니다.

- **Performance Highlights**: MCP-Solver의 초기 실험 결과는 LLM과 제약 해결 능력이 결합될 때 자연어 처리 성능이 향상될 수 있음을 시사합니다. 이 시스템은 자연어 사양을 유효한 MiniZinc 모델로 변환하는 데 효과적이며, 반복적인 수정 및 검증 과정에서 피드백을 제공하여 모델의 질을 높이는 데 기여합니다. 최종적으로 이 논문은 자연어 처리와 제약 기반 추론의 통합을 위한 중요한 첫 걸음을 내딛었다고 주장합니다.



### Two Cases of Deduction with Non-referring Descriptions (https://arxiv.org/abs/2501.00485)
Comments:
          In Proceedings NCL'24, arXiv:2412.20053

- **What's New**: 이 논문에서는 비지시적(non-denoting) 용어, 특히 '프랑스의 왕(the King of France)'과 같은 비지시적 서술(description)에 대한 형식적(reasoning) 추론을 다룹니다. 기존의 연구들(Indrzejczak, Zawidzki, Krbis)의 접근 방식을 대체하는 새로운 방법론을 제안하고 있습니다. 본 연구는 free logic과 sequent calculus 대신에 편재(type) 이론(partial type theory)과 자연 유도(natural deduction) 방식의 sequent 스타일을 사용하고 있습니다.

- **Technical Details**: 자연어(natural language)의 Montague 및 Tichý 스타일의 형식화(formalization)를 사용하여, 비지시적 서술의 보충(complement)과 함께 존재론적(pre-supposition) 전제를 다루는 강렬한(이행적) 전치사의 추론을 성공적으로 처리합니다. 이 논문은 이러한 서술을 포함한 문장에 대한 Strawsonian 규칙을 도출합니다. 이는 언어의 형식적 세부사항을 언급하며 기존의 이론적인 프레임워크를 확장하는 데 기여합니다.

- **Performance Highlights**: 논문은 비지시적 서술을 포함한 문장에 대한 형식적 추론을 새롭게 정립하며, 이런 접근 방식이 자연어 처리에 효과적으로 적용될 수 있음을 보여줍니다. 더불어, 수립된 규칙들은 언어의 의미론적(semiotic) 해석을 보다 심화하는 데 중요한 역할을 할 것으로 기대합니다. 이러한 연구는 언어철학이나 의미론 영역에서의 중요한 발전을 나타냅니다.



### Differentiable Prompt Learning for Vision Language Models (https://arxiv.org/abs/2501.00457)
- **What's New**: 이번 논문에서는 기존의 수동적 프롬프트 디자인을 자동화하기 위한 새로운 방법인 differentiable prompt learning (DPL)을 제안합니다. DPL은 최적의 프롬프트 길이를 자동으로 결정하는 최적화 문제로 설정되며, 이를 통해 성능을 최대화하는 것을 목표로 합니다. DPL 방법은 사전 훈련된 CLIP 모델에 적용되어, 기존 방법들보다 높은 신뢰도로 딥 연속 프롬프트 구성 파라미터를 찾을 수 있음을 입증하였습니다.

- **Technical Details**: DPL 방법은 연속 프롬프트의 컨텍스트 길이와 깊이를 자동으로 결정하는 데 초점을 맞추고 있습니다. DPL은 최적화 과정에서 각 레이어에 추가될 프롬프트의 깊이와 컨텍스트 길이를 조정함으로써, 프롬프트 학습의 유연성을 높입니다. 또한, DPL 방법은 수동 설계를 거치지 않고도 적은 데이터만으로 성능을 극대화할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: DPL 방법은 다운스트림 과제에서 평균 테스트 정확도를 11개 데이터셋에서 기존 방법보다 2.60% 향상시키는 성능을 보였습니다. 이 방법은 기존의 복잡한 설계와도 호환 가능하여, 최적의 딥 연속 프롬프트 구성을 데이터셋에 맞춰 조정함으로써 성능 향상이 가능합니다. 따라서 DPL 방법은 각기 다른 모델에 대해 비용 없이 쉽게 배포할 수 있는 장점을 가집니다.



### Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning (https://arxiv.org/abs/2501.00437)
Comments:
          ECCV 2024

- **What's New**: 최근 제로샷 이미지 캡셔닝(zero-shot image captioning)은 텍스트 데이터만으로 훈련을 진행하며 주목받고 있습니다. 본 논문에서는 사전 훈련된 텍스트-이미지 확산 모델(text-to-image diffusion model)을 활용해 합성 이미지-캡션 쌍을 생성하는 방법을 제안합니다. 그러나 합성 이미지의 결함 있는 세부사항이 시멘틱 불일치를 초래하는 문제를 다루기 위해, 새로운 패치 기반 크로스 모달 기능 혼합(Patch-wise Cross-modal feature Mix-up) 메커니즘을 제안합니다.

- **Technical Details**: 제안된 PCM-Net은 이미지의 시각적 개념을 탐지하고 해당 개념의 텍스트 피처와 합성 이미지의 패치 기반 시각적 피처를 선택적으로 융합합니다. 이를 통해 결함이 적고, 보다 정교한 피처 맵(feature map)을 생성할 수 있습니다. 또한, CLIP 가중치를 적용한 크로스 엔트로피 손실(clipped weighted cross-entropy loss)을 새롭게 도입하여 노이즈가 있는 합성 데이터를 활용한 훈련의 안정성을 향상시킵니다.

- **Performance Highlights**: MSCOCO 및 Flickr30k 데이터셋에서 실시된 광범위한 실험 결과, PCM-Net은 기존 VLMs 기반 접근 방식에 비해 우수한 성능을 보였습니다. 특히, PCM-Net은 도메인 내(in-domain) 및 교차 도메인(cross-domain) 제로샷 이미지 캡셔닝에서 1위를 기록했습니다. 이 연구 결과는 더 정밀한 시각-언어 정렬(visual-semantic alignment)을 통해 캡션 생성의 질을 높일 수 있는 가능성을 보여줍니다.



### TSPE: Task-Specific Prompt Ensemble for Improved Zero-Shot Audio Classification (https://arxiv.org/abs/2501.00398)
Comments:
          5 pages

- **What's New**: 최근 오디오-언어 모델(Audio-Language Models, ALMs)은 자연어 프롬프트(prompt)를 활용하여 제로샷 오디오 분류(zero-shot audio classification)에서 뛰어난 성능을 발휘하고 있습니다. 본 논문에서는 다양한 오디오 분류 작업을 위한 맞춤형 프롬프트를 생성하여 ALMs의 성능을 향상시키는 간단하면서도 훈련이 필요 없는 하드 프롬프트 기법인 TSPE(Task-Specific Prompt Ensemble)를 제안합니다. 이 방식은 일반적인 템플릿 기반 프롬프트 대신, "터널에서 나오는 자동차 소리"와 같이 컨텍스트가 풍부한 프롬프트를 생성합니다.

- **Technical Details**: TSPE는 레이블 정보를 활용하여 소리의 특성과 출처를 식별하고, 이를 프롬프트에 통합하여 오디오 분류에 사용합니다. 특히, 프롬프트 앙상블을 통해 TSPE가 생성한 작업별 프롬프트의 정렬을 강화합니다. 12개의 다양한 오디오 분류 데이터셋에서 평가한 결과, TSPE는 기존 제로샷 평가에서 1.23-16.36%의 성능 개선을 보여주며, 이는 ALMs의 성능 향상에 기여합니다.

- **Performance Highlights**: TSPE는 추가적인 훈련 없이 오디오 분류 성능을 크게 향상시킬 수 있는 방법론으로, 다양한 오디오 환경에서 잘 작동합니다. 특히, 기존의 ALMs가 OOD(out-of-distribution) 데이터셋에서 성능이 저하되는 한계를 극복하는데 기여합니다. 이러한 점에서 TSPE는 제로샷 오디오 분류 기술의 미래 가능성을 제시하며, 오디오-언어 모델의 진화를 지원합니다.



### Efficient Relational Context Perception for Knowledge Graph Completion (https://arxiv.org/abs/2501.00397)
- **What's New**: 이 논문은 기존의 Knowledge Graph Embedding (KGE) 방법의 한계를 극복하기 위해 새로운 Triple Receptance Perception (TRP) 아키텍처를 제안합니다. TRP는 지식 그래프에서 동적이고 순차적인 컨텍스트를 캡처하도록 설계되어, 다양한 그래프 컨텍스트에 따라 적응하는 임베딩을 학습할 수 있습니다. 또한, Tucker 분해를 결합하여 효율적인 관계 디코딩을 가능하게 하고 있습니다.

- **Technical Details**: TRP 아키텍처는 시퀀스 정보를 모델링하여 엔티티와 관계의 동적 맥락을 학습할 수 있도록 합니다. 이 방법은 복잡한 관계 종속성을 인코딩하는 모델의 능력을 향상시키며, 고차원 텐서를 분해하여 compact yet expressive한 표현을 가능하게 만듭니다. Tucker 분해는 엔티티와 관계의 임베딩을 구성 요소로 모델링하여 효율적인 표현 학습을 지원합니다.

- **Performance Highlights**: YAGO3-10, UMLS, FB15k와 FB13 같은 벤치마크 데이터셋에서의 실험을 통해, 제안된 방법이 링크 예측 및 삼중 분류 작업에서 여러 최신 모델보다 우수한 성능을 보임을 입증하였습니다. 이러한 결과들은 TRP 아키텍처의 효과성을 강하게 뒷받침하고 있습니다.



### VoxVietnam: a Large-Scale Multi-Genre Dataset for Vietnamese Speaker Recognition (https://arxiv.org/abs/2501.00328)
Comments:
          Accepted to 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 본 논문은 베트남어 화자 인식을 위한 첫 번째 다장르 데이터셋인 VoxVietnam을 도입하며, 1,406명의 화자와 187,000개 이상의 발화를 포함합니다. 기존의 베트남어 화자 인식 데이터셋은 크기나 장르 다양성에 있어 한계가 있었으나, VoxVietnam은 대규모 데이터셋 구축을 위한 자동화된 파이프라인을 통해 이러한 문제를 해결합니다. 이를 통해 음악이나 일상 회화 등 다양한 장르에서의 연구를 촉진할 것으로 기대됩니다.

- **Technical Details**: 제안된 데이터 구축 파이프라인은 비디오 크롤링, 오디오 구간 분할, 화자 군집화, 시각적 청소, 화자 조합 및 발화 장르 분류의 여섯 가지 주요 단계로 구성됩니다. 각 단계는 수집된 비디오와 오디오 데이터를 기반으로 하며, 특히 화자 군집화 과정에서는 사전 훈련된 화자 인코더를 사용하여 발화를 화자별로 그룹화합니다. 장르 분류는 Audio Spectrogram Transformer (AST) 모델을 활용하여 수행되며, 베트남어 발화에 대한 세부 조정이 포함됩니다.

- **Performance Highlights**: 실험 결과, 단일 장르 데이터셋으로 훈련된 모델이 다장르 데이터셋에서 테스트될 경우 성능이 최대 5배 향상되는 것으로 나타났습니다. VoxVietnam을 훈련 자료로 포함했을 때 성능 향상이 명확하게 드러났으며, 다장르 현상의 도전 과제를 해결하는 데 중요한 기여를 할 것으로 보입니다. 이 데이터셋은 베트남어 화자 인식 연구의 발전에 크게 기여할 것이며, 향후 연구의 기초 자료로 활용될 것으로 기대됩니다.



### Retrieval-Augmented Generation with Graphs (GraphRAG) (https://arxiv.org/abs/2501.00309)
- **What's New**: 본 논문의 중심 주제인 Retrieval-Augmented Generation (RAG)은 외부 데이터 소스에서 추가 정보를 검색하는 강력한 기술로, 최근에는 GraphRAG와 통합되어 더욱 큰 잠재력을 발휘하고 있습니다. GraphRAG는 비정형의 이질적 관계 정보를 인코딩하는 그래프 구조를 이용해 보다 다양하고 복잡한 도메인에서의 적용 가능성을 높이고 있습니다. 본 설문조사에서는 GraphRAG의 관점에서 주요 구성 요소와 결합의 중요성을 다루며, 다양한 도메인에 특화된 GraphRAG 기술을 검토합니다.

- **Technical Details**: GraphRAG는 질문 수행 과정에서 전통적인 RAG와는 달리 그래프 구조를 통해 연결된 노드와 엣지를 활용하여 관계 지식을 더욱 효율적으로 추출합니다. 이는 Graph Neural Networks (GNNs)와 같은 그래프 기반 머신 러닝 기법을 활용하여 관계 패턴을 탐색할 수 있으므로, 특정 작업에 대해 더욱 정교하고 적합한 답변을 생성할 수 있습니다. 또한, 다양한 형식의 데이터와 도메인 특수성을 고려하여 그래프 인코더와 표기 방식에 독창적인 설계를 필요로 합니다.

- **Performance Highlights**: 본 연구의 결과는 GraphRAG가 전통적인 RAG에 비해 정보 검색과 생성 과정에서의 성능을 상당히 향상시킬 수 있음을 보여줍니다. 특히, 관계 기반 접근 방식을 통해 검색된 정보가 서로 어떻게 연결되는지를 명확히 하여, 다양한 고수익 사례에서 신뢰성과 효율성을 크게 증가시킵니다. 사용자 정의 쿼리에 맞는 다단계 관계 알고리즘을 활용함으로써, 복잡한 질의에 대해 더욱 정확한 응답을 생성할 수 있음을 강조하고 있습니다.



### Automatically Planning Optimal Parallel Strategy for Large Language Models (https://arxiv.org/abs/2501.00254)
- **What's New**: 이번 연구에서는 대규모 언어 모델의 훈련에서 최적의 병렬 전략을 자동으로 찾는 알고리즘을 제안합니다. 새로운 알고리즘은 훈련 시간을 시뮬레이션하고 이를 기반으로 병렬 솔루션 공간을 줄여 최적의 솔루션을 찾습니다. 특히, micro batch size와 global batch size와 같은 세부 변수도 고려하여 보다 정교하게 병렬 전략을 수립할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 병렬 훈련 시간을 계산(computation), 통신(communication), 겹침(overlap)으로 분리하여 시뮬레이션 모델을 구축합니다. 이 모델을 기반으로 99%의 검색 공간을 줄여주며, 평균적으로 96%의 정확도로 병렬 훈련 기간을 추정할 수 있습니다. 이러한 접근을 통해 이전의 경험적인 기준에 의존하지 않고, 사용자에게 최적의 하이퍼파라미터 선택을 지원합니다.

- **Performance Highlights**: 여러 노드에서 진행된 실험 결과, 제안된 알고리즘은 항상 글로벌 최적(global optimum) 전략을 제공하며, 훈련 효율성을 극대화하는 데 기여합니다. 다양한 대규모 언어 모델을 대상으로 한 추가 실험을 통해 알고리즘의 정확성이 입증되었습니다. 이러한 결과들은 특히 자원 집약적인 대규모 언어 모델 개발에 있어 중요한 비용 절감 효과를 가져올 수 있습니다.



### Generative Emergent Communication: Large Language Model is a Collective World Mod (https://arxiv.org/abs/2501.00226)
- **What's New**: 이 연구는 generative emergent communication (generative EmCom)이라는 통합 이론 틀을 제안합니다. 이 틀은 emergent communication, world models, 및 large language models (LLMs)를 집단적 예측 부호화(collective predictive coding, CPC)의 관점에서 연결합니다. 제안된 프레임워크는 여러 에이전트 간의 분산된 Bayesian 추론을 통해 언어 및 기호 시스템의 발생을 형식화하며, 전통적인 차별적 모델 기반 접근 방식을 넘어선다고 설명합니다.

- **Technical Details**: 이 연구는 generative EmCom이라는 새로운 프레임워크를 제안하며, 이를 통해 multi-agent reinforcement learning (MARL)에서의 커뮤니케이션 발생을 제어(control)로서의 추론(inference)으로부터 도출할 수 있음을 보여줍니다. 또한, LLM을 집단적 세계 모델로 해석하는 수학적 정식화도 제안하며, 이는 다양한 에이전트의 경험을 CPC를 통해 통합하는 방식으로 이루어집니다. 이를 통해 집단적 예측 부호화 과정에서 공유 기호 시스템이 어떻게 발생하는지를 이해할 수 있는 통일된 이론적 기초를 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 언어 발생의 근본적인 측면을 설명하고 LLM을 이해하는 데 필요한 실용적인 통찰을 제공합니다. 저자들은 수학적 정식화와 기존의 연구에 대한 논의를 통해 이 프레임워크가 인공지능 개발 및 인간-인공지능 상호작용 개선에 있어 어떻게 기여할 수 있는지를 보여줍니다. 궁극적으로, 이 연구는 복잡한 AI 시스템 및 다중 에이전트 시스템의 발전에 중요한 기틀을 제공할 것으로 기대됩니다.



### MLLM-as-a-Judge for Image Safety without Human Labeling (https://arxiv.org/abs/2501.00192)
- **What's New**: 이번 연구는 AI 생성 이미지 콘텐츠 안전의 중대성을 강조하며, 기존의 데이터 세트에 대한 인간 주석자 의존의 한계를 극복하기 위해 Multimodal Large Language Models (MLLMs)와 미리 정의된 안전 규정(Safety Constitution) 기반의 제로샷(zero-shot) 이미지 안전 판단 방법론인 CLUE를 제안합니다.

- **Technical Details**: CLUE 방법론은 안전 규정을 객관적인 규칙으로 변환하고, 각 이미지를 평가하기 위해 규정 각각의 타당성을 평가합니다. 복잡한 안전 규정을 처리하기 위해 논리적으로 완전하지만 단순화된 전제 조건 체인을 사용하며, 이미지와 규칙의 관련성을 측정하기 위해 CLIP과 같은 다중모달 대조 모델을 활용합니다.

- **Performance Highlights**: 실험 결과, CLUE 방법론은 Qwen2-VL-7B-Instruct, InternVL2-8B-AWQ, LLaVA-v1.6-34B 및 InternVL2-76B와 같은 다양한 MLLM에서 제로샷 이미지 안전 판단의 정확성과 신뢰성을 크게 개선하며, InternVL2-76B 모델을 사용하여 Unsafe/Safe 이미지를 구분할 때 95.9%의 리콜, 94.8%의 정확도 및 0.949 F-1 점수를 달성하였습니다.



### DeepLL: Considering Linear Logic for the Analysis of Deep Learning Experiments (https://arxiv.org/abs/2501.00169)
Comments:
          8 pages, 3 figures

- **What's New**: 이번 연구에서는 Deep Learning 실험의 분석을 위한 Linear Logic의 활용을 조사했습니다. 실험의 control flow를 추상적으로 표현하고, API 호출 및 하드웨어 자원과 같은 사용 가능한 실험 자원 세트를 명시할 수 있는 방법을 제안합니다. 또한 실험 중 자원을 올바르게 사용하는 규칙에 대한 추론 규칙도 포함하고 있습니다.

- **Technical Details**: Deep Learning 실험에서 데이터셋을 관리하고 하드웨어 가속기와 상호작용하는 API를 효율적으로 사용하기 위해서는 신중한 접근이 필요합니다. 데이터 처리 중의 소프트웨어 실수는 실험을 오염시켜 잘못된 결과를 초래할 수 있습니다. 또한, 잘못 작성된 API는 비효율적인 자원 사용으로 이어져 신뢰할 수 없는 결론을 초래할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 경량이며 이해하기 쉬운 두 가지 구성 요소인 기호적(symbolic) 요소와 시각적(visual) 요소를 갖추고 있습니다. 실험 결과를 보장하기 위해 필수적인 데이터셋의 분리와 올바른 하이퍼파라미터 튜닝 과정이 중요한 역할을 하며, 이는 실험의 정확성과 효율성을 높이는 데 기여합니다.



### LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models (https://arxiv.org/abs/2501.00055)
- **What's New**: 본 연구에서는 공격 방법론에 대한 새로운 통찰력을 제시합니다. 연구팀은 기존의 jailbreak 방식의 한계를 극복하기 위해, 생물학적 바이러스의 진화 및 감염 과정을 영감을 받아 새로운 공격 방식인 LLM-Virus를 제안합니다. 이 방법은 진화 알고리즘을 기반으로 하여, 효율성, 이식성 및 낮은 시간 비용을 보장합니다.

- **Technical Details**: LLM-Virus는 공격을 진화적 감염 문제로 간주하며, LLM을 휴리스틱 진화 연산자로 활용합니다. 이 연구는 새로운 출력의 가능성을 극대화하기 위해 LLM 기반의 교차 및 돌연변이 연산을 제안합니다. 또한, 전송 학습 문제로 보아 Local Evolution 및 Generalized Infection 기법을 도입하여 계산 비용과 시간 비용을 줄입니다.

- **Performance Highlights**: 실험 결과, LLM-Virus는 기존의 공격 방법들과 비교할 때 경쟁력 있는 성능을 보여주었습니다. 특히, HarmBench 및 AdvBench 데이터셋에서 LLM-Virus가 기존 방법들보다 높은 성공률을 기록했습니다. 이러한 결과는 LLM-Virus가 진화적 jailbreak의 새로운 기준이 될 수 있음을 의미합니다.



### AdvAnchor: Enhancing Diffusion Model Unlearning with Adversarial Anchors (https://arxiv.org/abs/2501.00054)
- **What's New**: 본 논문에서는 AdvAnchor라는 새로운 방법을 제안하여, 텍스트-이미지 확산 모델의 불필요한 개념 제거를 최적화합니다. 이 방법은 불편한 개념의 임베딩을 유사하게 만들어 전반적인 모델 성능을 유지하면서 해당 개념의 정의적 속성을 효과적으로 제외하는 것을 목표로 합니다. 기존의 방법들과 달리, AdvAnchor는 공격적 앵커(adversarial anchor)를 생성하여 성능 저하를 최소화하고자 합니다.

- **Technical Details**: AdvAnchor는 효과적인 개념 지우기를 위해 이상적인 앵커가 불편한 개념과의 의미론적 유사성을 유지하면서 정의적 속성을 제외해야 함을 발견하였습니다. 이를 위해 추가된 보편적 섭동(universal perturbations)은 불편한 개념의 임베딩에 추가되어, 특정 개념에 대한 모델의 예측을 저하시키거나 지우는 데 도움을 줍니다. 제안된 방법은 다양한 MU 기법과 통합될 수 있어 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, AdvAnchor는 최첨단 방법들에 비해 두 가지 성능을 크게 향상시켰습니다: 불필요한 개념의 제거 성능과 모델의 전반적인 성능 보존에서 큰 향상을 보였습니다. 즉, 효과적인 개념 지우기와 최소한의 성능 저하가 동시에 이루어질 수 있음을 확인했습니다. 또한, 수많은 실험을 통해 제안된 방법의 우수성을 입증하였으며, 해당 코드는 공공적으로 제공됩니다.



### Speech Recognition With LLMs Adapted to Disordered Speech Using Reinforcement Learning (https://arxiv.org/abs/2501.00039)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 논문에서는 음성 입력을 처리할 수 있는 대형 언어 모델(LLM)을 소개하며, 인간 선호에 대한 강화 학습(reinforcement learning on human preference, RLHF)으로 추가 조정하여 전통적인 파인 튜닝보다 비정상적인(disordered) 음성을 더 잘 인식하도록 할 수 있음을 보여줍니다. 이 방법은 LLM의 어휘에서 저주파 텍스트 토큰을 오디오 토큰으로 대체하여, 전사(transcript)가 있는 음성으로 모델을 세밀하게 조정할 수 있게 해줍니다.

- **Technical Details**: LLM 기반 자동 음성 인식(ASR)를 위해 디코더 전용(transformer-based) LLM을 사용하여 텍스트와 오디오 토큰을 입력으로 받아 전사에 해당하는 텍스트를 생성합니다. 특히, 오디오 토큰을 통해 저주파 텍스트 토큰을 대체하고, 이 과정에서 언어 모델의 아키텍처나 훈련 인프라에 변화 없이 일반적인 LLM으로 훈련할 수 있습니다. 음성 데이터의 토큰화와 클러스터링 과정을 통해 모델은 음성을 효과적으로 처리할 수 있는 기능을 추가적으로 확보합니다.

- **Performance Highlights**: 이 모델은 ASR 데이터에 대한 표준 지도 파인 튜닝(supervised fine-tuning) 절차를 통해 음성을 인식하는 데 조정됩니다. 연구에서는 Librispeech와 Euphonia의 두 가지 음성 데이터셋을 사용하여 LLM의 성능을 향상시키는 다양한 혼합 조정을 실험하였습니다. 결과적으로 강화 학습을 사용하여 커스터마이즈된 보상으로 튜닝한 LLM이 기존의 파인 튜닝보다 더 나은 성능을 보이는 것으로 나타났습니다.



### Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs (https://arxiv.org/abs/2501.00032)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론을 가속화하기 위해 Arm CPU에 최적화된 커널을 제안합니다. 기존의 양자화 방식으로 인한 메모리 대역폭 문제를 극복하기 위한 혁신적이고 효율적인 방법을 통해, LLM의 성능을 극대화하고자 합니다. 또한 코드를 쉽게 조정할 수 있도록 개선된 양자화 방법을 통해 저전력 장치에서도 효율적으로 사용할 수 있도록 설계되었습니다.

- **Technical Details**: 저자들은 다양한 저비트 폭 그룹 양자화 LLM을 위한 고도로 최적화된 GEMV(General Matrix Vector multiplication) 및 GEMM(General Matrix Matrix multiplication) 커널을 개발했습니다. 이 커널들은 ARM CPU의 벡터 및 행렬 곱셈 명령어를 최대한 활용하도록 설계되어 메모리 접근 및 오버헤드를 최소화합니다. 또한 비균일 코드를 기반으로 한 포스트 트레이닝 양자화 방법을 제공해 LLM의 품질을 개선하고 효율성을 높이고자 합니다.

- **Performance Highlights**: 연구 결과, 최적화된 4비트 그룹 양자화 커널을 통해 Arm CPU에서 LLM 추론의 첫 토큰 처리 시간에서 3~3.2배 개선되고, 메모리 유틸리티에 대한 처리 성능이 2배 향상되었습니다. 아울러 최적화된 비균일 양자화 방법은 텍스트 생성 품질을 개선하고, 기존의 주요 기술들 대비 더 나은 성능을 계속해서 구현하고 있습니다.



### NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions (https://arxiv.org/abs/2501.00004)
- **What's New**: 이번 연구에서는 NewsHomepages라는 대규모 데이터셋을 통해 3,000개 이상의 뉴스 웹사이트의 홈페이지 레이아웃을 캡처하였습니다. 이 데이터셋은 3년 동안 하루에 두 번 수집된 자료로, 정보 우선순위 결정 과정을 분석하고자 합니다. 연구진은 이러한 데이터셋을 바탕으로 뉴스 아이템 간의 상대적 중요성을 유추하는 모델을 개발하였으며, 이를 통해 조직의 우선순위와 정보 구조의 관계를 탐색합니다.

- **Technical Details**: 뉴스 웹사이트의 홈 페이지 레이아웃은 전문 에디터에 의해 세심하게 선택된 정보 우선순위를 반영합니다. 연구에서는 HTML 스냅샷, 링크 및 추가 메타데이터를 포함해 363,000개의 페이지 스크린샷을 수집하였으며, 이를 통해 레이아웃 내에서 뉴스 기사의 위치를 정교하게 파악합니다. 또한, 페어와이즈 비교 모델을 개발하여 기사 사이의 상대적 중요성을 학습하고 예측합니다.

- **Performance Highlights**: 이 연구의 두 가지 실험을 통해 뉴스 가치 평가에서 예외적인 상관관계를 발견하였습니다. 특히, 다른 성향의 뉴스 아울렛 간에도 유사한 뉴스 가치 판단이 있음을 확인했습니다. 마지막으로, 이러한 발견은 디지털 환경에서의 정보 구조와 미묘한 암시가 인간의 인식에 미치는 영향을 더 깊이 이해할 수 있는 기초를 제공합니다.



### Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey (https://arxiv.org/abs/2412.20367)
- **What's New**: 이 논문은 코드 최적화와 생성에서 강화 학습(Reinforcement Learning, RL)의 응용에 대한 체계적인 조사를 제공합니다. 특히 RL이 컴파일러 최적화, 자원 할당 및 다양한 툴 및 프레임워크 개발에서 어떻게 활용되는지를 강조합니다. 이에 따라 RL은 코드 생성의 새로운 접근법을 제공하며, 기존의 사전 훈련된 모델에 대한 의존도를 줄이는 데 기여하여 유연한 코드 최적화를 가능하게 합니다.

- **Technical Details**: 코드 LLM(대형 언어 모델)에서 RL은 보상 신호를 통해 최적 정책을 발견하는 방법으로 활용됩니다. RL 알고리즘은 코드 생성을 개선하기 위해 다양한 디코딩 전략을 사용하며, deterministic과 sampling 전략으로 나눌 수 있습니다. 특히, PPO(Proximal Policy Optimization)와 Actor-Critic 구조는 RL에서 중요한 역할을 하며, 이들은 액션 실행자와 평가자로 구성되어 정책을 최적화하는 데 기여합니다.

- **Performance Highlights**: 강화 학습은 코드 생성 모델이 실제 기능 요구 사항과 더 잘 일치하도록 도와주는 중요한 방법입니다. Unit test 신호를 활용하여 프로그램의 기능적 정확성을 평가하고, 이 신호를 보상으로 사용하여 RL 모델의 출력을 개선하는 데 초점을 둡니다. 이러한 RL 기반의 미세 조정 방법은 코드 생성 작업에서 지속적으로 성장하고 있으며, 이러한 접근법은 코드 생성을 위한 새로운 가능성을 열어줍니다.



New uploads on arXiv(cs.IR)

### An Efficient Attention Mechanism for Sequential Recommendation Tasks: HydraRec (https://arxiv.org/abs/2501.01242)
- **What's New**: 본 연구는 최근 Transformer 기반 모델이 추천 시스템(recommender systems, RS)에서 점점 더 효과적으로 사용되고 있음을 강조합니다. 기존의 Transformer 모델은 언어 모델링에서 좋은 성능을 발휘했으나, RS에서는 시퀀스 길이에 따라 복잡성이 기하급수적으로 증가하는 문제가 있었습니다. 이를 해결하기 위해 HydraRec이라는 새로운 효율적인 Transformer 기반 Sequential RS 모델을 제안합니다.

- **Technical Details**: HydraRec은 주목(attention) 계산의 이론적 복잡성을 감소시키며, 긴 시퀀스 및 대규모 데이터 세트에 대한 처리 성능을 개선합니다. 특히, Hydra attention의 개념을 기반으로 하여, 토큰 수와 모델의 임베딩 차원 모두에서 복잡성을 줄이는 방식으로 설계되었습니다. 이 모델은 causal masking을 사용할 때, 시퀀스 추천(next item prediction) 작업에서 기존의 dot-product 기반 모델과 비견할 만한 성능을 보여줍니다.

- **Performance Highlights**: HydraRec은 다양한 평가 지표에서 다른 선형 어텐션 기반 모델들보다 뛰어난 성능을 발휘했습니다. 특히, BERT4Rec 모델과 비교했을 때 이 모델은 running time에서 개선을 보였으며, 이는 사용자 구매 이력과 같은 동적 시퀀스 데이터를 처리하는 데 있어 더욱 효율적임을 의미합니다.



### Search Plurality (https://arxiv.org/abs/2501.00987)
- **What's New**: 이번 연구에서는 기존 검색 엔진의 설계 관행을 비판하고 검색 중립성(Search Neutrality)의 실현이 불가능하다는 필립스의 주장을 토대로, 검색의 우선순위 및 계층적 정렬을 제안합니다. 우리는 '검색 다원성(Search Plurality)'이라는 새로운 개념을 도입하여 사용자가 다양한 방법으로 쿼리에 접근할 수 있도록 강조합니다. 이 접근법은 검색 결과의 카테고리를 우선 보여주어 사용자가 검색의 폭을 파악하는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 세 가지 요소를 포함합니다: 카테고리, 명시적 관련성(explicit relevance), 그리고 다원적 설계(plural design)입니다. 사용자가 검색 쿼리와 관련된 특정 주제를 명확히 볼 수 있도록 카테고리 표시가 이루어지며, 여기서는 각 카테고리에 대한 정보가 확대되어 보이게 됩니다. 카테고리 간의 우열을 명시하지 않고 사용자가 선택할 수 있도록 디자인되어, 이를 통해 특정 주제에 대한 다양한 관점을 쉽게 탐색할 수 있게 됩니다.

- **Performance Highlights**: 이 접근법은 탐색이 중심인 경우에 특히 유용하며, 사용자가 다양한 관련 소스를 통해 포괄적인 이해를 얻게 합니다. 학습과 탐험욕구를 충족시켜주어 여러 관점을 인식하고, 결코 정답이 없거나 단일한 해답이 아니라는 점을 강조함으로써 사용자의 비판적 사고를 증진시킵니다. 궁극적으로는 복잡한 정보 생태계에서 사각지대에 있는 목소리를 포함시켜, 보다 포용적이고 민주적인 정보 교환 환경을 촉진합니다.



### S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain (https://arxiv.org/abs/2501.00384)
Comments:
          Accepted by WSDM 2025

- **What's New**: 이 논문에서는 추천 시스템에서 사용자-항목 상호작용 행렬에서 사용자 선호도를 회복하는 접근 방식을 제안합니다. 새로운 모델 S-Diff는 그래프 기반 협업 필터링에 영감을 받아 저주파 성분을 효과적으로 활용합니다. 이를 통해 S-Diff는 사용자 상호작용 벡터를 스펙트럼 도메인으로 매핑하고, 확산 과정을 통해 얻은 노이즈를 조절하여 높은 신호 대 잡음 비율을 유지할 수 있습니다.

- **Technical Details**: S-Diff는 그래프 스펙트럼 도메인에서 저주파 성분을 보존하는 비등방성 확산(Anisotropic diffusion) 모델을 소개합니다. 이 과정에서 확산 모델의 노이즈 스케쥴링 매개변수는 그래프의 고유값 계수와 연관됩니다. 이는 사용자 선호도를 더욱 정확하게 복원할 수 있도록 협업 신호를 조절하는 유연한 분산 유지 매개변수 설정을 포함합니다.

- **Performance Highlights**: 제안한 S-Diff 모델은 여러 데이터 세트에서 강력한 성과를 보였으며, 기존의 추천 시스템보다 사용자 선호도 추정의 정확도를 높였습니다. 다양한 실험을 통해 노이즈 복원 과정에서의 효과성을 입증했으며, 특히 저주파 성분을 유지하는 것이 중요한 성과로 인식됩니다. 이는 기존 확산 모델의 한계를 넘어서는 새로운 접근 방식으로 평가될 수 있습니다.



### Who Gets Recommended? Investigating Gender, Race, and Country Disparities in Paper Recommendations from Large Language Models (https://arxiv.org/abs/2501.00367)
- **What's New**: 이번 연구는 여러 대표적인 대형 모델들이 문학 추천(literature recommendation) 작업에서 보여주는 성능을 조사했습니다. 특히 연구 노출에 대한 잠재적 편향(bias)을 탐구하며, 대형 언어 모델(LLMs)의 추천 성능을 심층적으로 분석하였습니다.

- **Technical Details**: 연구 결과에 따르면, LLMs의 전반적인 추천 정확도(overall recommendation accuracy)는 여전히 제한적이며, 이 모델들은 인용 수(citation counts)가 많은 문헌, 최신 발표 날짜(later publication date), 그리고 더 큰 저자 팀(author teams)을 추천하는 경향이 있습니다. 이러한 경향은 대형 모델의 제한된 성능을 시사하는 요소로 작용합니다.

- **Performance Highlights**: 그러나 학자 추천(scholar recommendation) 작업에서는 LLMs가 남성, 백인, 혹은 개발국(author)의 저자를 불균형적으로 추천하지 않는다는 증거가 발견되지 않았습니다. 이는 인간의 알려진 편향(patterns of known human biases)과는 대조적인 결과입니다.



### Retrieval-Augmented Generation with Graphs (GraphRAG) (https://arxiv.org/abs/2501.00309)
- **What's New**: 본 논문의 중심 주제인 Retrieval-Augmented Generation (RAG)은 외부 데이터 소스에서 추가 정보를 검색하는 강력한 기술로, 최근에는 GraphRAG와 통합되어 더욱 큰 잠재력을 발휘하고 있습니다. GraphRAG는 비정형의 이질적 관계 정보를 인코딩하는 그래프 구조를 이용해 보다 다양하고 복잡한 도메인에서의 적용 가능성을 높이고 있습니다. 본 설문조사에서는 GraphRAG의 관점에서 주요 구성 요소와 결합의 중요성을 다루며, 다양한 도메인에 특화된 GraphRAG 기술을 검토합니다.

- **Technical Details**: GraphRAG는 질문 수행 과정에서 전통적인 RAG와는 달리 그래프 구조를 통해 연결된 노드와 엣지를 활용하여 관계 지식을 더욱 효율적으로 추출합니다. 이는 Graph Neural Networks (GNNs)와 같은 그래프 기반 머신 러닝 기법을 활용하여 관계 패턴을 탐색할 수 있으므로, 특정 작업에 대해 더욱 정교하고 적합한 답변을 생성할 수 있습니다. 또한, 다양한 형식의 데이터와 도메인 특수성을 고려하여 그래프 인코더와 표기 방식에 독창적인 설계를 필요로 합니다.

- **Performance Highlights**: 본 연구의 결과는 GraphRAG가 전통적인 RAG에 비해 정보 검색과 생성 과정에서의 성능을 상당히 향상시킬 수 있음을 보여줍니다. 특히, 관계 기반 접근 방식을 통해 검색된 정보가 서로 어떻게 연결되는지를 명확히 하여, 다양한 고수익 사례에서 신뢰성과 효율성을 크게 증가시킵니다. 사용자 정의 쿼리에 맞는 다단계 관계 알고리즘을 활용함으로써, 복잡한 질의에 대해 더욱 정확한 응답을 생성할 수 있음을 강조하고 있습니다.



### NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions (https://arxiv.org/abs/2501.00004)
- **What's New**: 이번 연구에서는 NewsHomepages라는 대규모 데이터셋을 통해 3,000개 이상의 뉴스 웹사이트의 홈페이지 레이아웃을 캡처하였습니다. 이 데이터셋은 3년 동안 하루에 두 번 수집된 자료로, 정보 우선순위 결정 과정을 분석하고자 합니다. 연구진은 이러한 데이터셋을 바탕으로 뉴스 아이템 간의 상대적 중요성을 유추하는 모델을 개발하였으며, 이를 통해 조직의 우선순위와 정보 구조의 관계를 탐색합니다.

- **Technical Details**: 뉴스 웹사이트의 홈 페이지 레이아웃은 전문 에디터에 의해 세심하게 선택된 정보 우선순위를 반영합니다. 연구에서는 HTML 스냅샷, 링크 및 추가 메타데이터를 포함해 363,000개의 페이지 스크린샷을 수집하였으며, 이를 통해 레이아웃 내에서 뉴스 기사의 위치를 정교하게 파악합니다. 또한, 페어와이즈 비교 모델을 개발하여 기사 사이의 상대적 중요성을 학습하고 예측합니다.

- **Performance Highlights**: 이 연구의 두 가지 실험을 통해 뉴스 가치 평가에서 예외적인 상관관계를 발견하였습니다. 특히, 다른 성향의 뉴스 아울렛 간에도 유사한 뉴스 가치 판단이 있음을 확인했습니다. 마지막으로, 이러한 발견은 디지털 환경에서의 정보 구조와 미묘한 암시가 인간의 인식에 미치는 영향을 더 깊이 이해할 수 있는 기초를 제공합니다.



### On the Robustness of Cover Version Identification Models: A Study Using Cover Versions from YouTub (https://arxiv.org/abs/2501.01333)
Comments:
          accepted for presentation at iConference 2025

- **What's New**: 이 논문은 기존의 커버 송 식별(cover song identification) 방법이 YouTube와 같은 온라인 비디오 플랫폼에서 잘 작동하지 않는다는 점을 밝혀냅니다. 이를 위해 저자들은 다중 모드 불확실성 샘플링(multi-modal uncertainty sampling) 방법으로 YouTube의 노래 샘플을 주석(annotation)하고, 최신 모델들을 평가했습니다. 동영상 플랫폼에서의 커버 송 특성과 관련하여, 저자들은 이러한 특성이 어떻게 현재 모델의 불확실성을 촉발하는지를 탐구합니다.

- **Technical Details**: 버전 식별(version identification, VI) 분야에서는 커버 송의 자동 감지를 목표로 하는 연구가 진행되고 있으며, 최근의 방법들은 가장 관련성이 높은 정보를 유지하는 방식으로 곡을 인코딩하는 것을 지향합니다. 그러나 YouTube와 같은 비디오 플랫폼에서는 비디오 소재의 정렬(alignment) 문제가 중요한 도전 과제가 되며, 이는 비디오가 여러 버전을 포함할 수 있고, 소음(noise)이나 다른 배경 요소가 포함될 수 있기 때문입니다. 저자들은 신뢰할 수 있는 데이터셋을 구성하기 위해 사람이 주석을 달고, 커버 버전의 불확실성을 분류하는 시스템을 제안합니다.

- **Performance Highlights**: 본 연구 결과, 기존의 최첨단 모델이 제안된 데이터셋에서 성능이 크게 저하되는 것으로 나타났습니다. 특히, 악기 단독 또는 보컬 트랙 분리와 같은 특정 버전 특성에서의 어려움이 강조되었습니다. 이러한 발견은 커버 버전 정의의 경계를 드러내며, 향후 연구에 유용한 방향성을 제시합니다.



### Domain-invariant feature learning in brain MR imaging for content-based image retrieva (https://arxiv.org/abs/2501.01326)
Comments:
          6 pages, 1 figures. Accepted at the SPIE Medical Imaging 2025

- **What's New**: 이번 연구에서는 뇌 MR 이미지를 위한 콘텐츠 기반 이미지 검색(CBIR) 시스템에서 도메인 차이를 줄이기 위해 스타일 인코더 적대적 도메인 적응(style encoder adversarial domain adaptation, SE-ADA)이라는 새로운 저차원 표현(low-dimensional representation, LDR) 방법을 제안했습니다. 기존의 방법들이 도메인 간의 통일성을 확보하는 데 한계를 보인 반면, SE-ADA는 적대적 학습(adversarial learning)을 통해 도메인 정보를 분리하고 병리적 특징을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: SE-ADA는 3D 컨볼루션 오토인코더(3D-CAE) 구조를 기반으로 하는 새로운 메커니즘으로, 주요 인코더(primary encoder)와 분리된 스타일 인코더(style encoder)를 통합합니다. 이 스타일 인코더는 도메인 특정 스타일 정보를 추출하여 LDR에 불필요한 도메인 정보를 제거함으로써 이미지 검색을 위한 필수적인 질병 탐지 정보를 보존합니다. 연구 방식은 인코더, 디코더, 스타일 인코더 각각의 구성 요소를 반복적으로 훈련하여 LDR을 업데이트하는 것입니다.

- **Performance Highlights**: SE-ADA는 8개의 공개 뇌 MR 데이터 세트(ADNI1/2/3, OASIS1/2/3/4, PPMI)에서 최근의 도메인 조화화 방법들과 비교한 결과, 도메인 정보를 효과적으로 제거하면서도 원본 뇌 구조의 주요 측면을 보존하여 최고의 질병 검색 정확도를 보여주었습니다. 이러한 성과는 CBIR 시스템의 정밀도를 높이는 데 기여할 것으로 기대되며, 향후 다양한 뇌 MR 이미지를 활용한 연구에도 좋은 영향을 미칠 것으로 보입니다.



### LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models (https://arxiv.org/abs/2501.00874)
- **What's New**: 최근 대규모 언어 모델(LLMs)에 기반한 임베딩 모델의 발전이 여러 텍스트 임베딩 작업에서 새로운 최고 성능 기준을 세웠습니다. 그러나 이러한 모델은 주로 영어에 집중되어 있어 다국어 임베딩 기능은 거의 탐색되지 않았습니다. 이를 해결하기 위해 제안된 LUSIFER는 다국어 감독 없이 LLM 기반 임베딩 모델을 다국어 작업에 적응시키는 새로운 제로샷(zero-shot) 접근법입니다.

- **Technical Details**: LUSIFER의 구조는 다국어 인코더와 임베딩 특정 작업에 최적화된 LLM 기반 임베딩 모델로 구성됩니다. 이 두 구성 요소는 훈련 가능한 최소한의 매개변수를 통해 원활하게 통합되어, 다국어 인코더의 언어 이해 능력을 특화된 임베딩 모델로 효과적으로 전달합니다. LUSIFER는 14개 언어에 걸쳐 123개의 다양한 데이터셋을 포함하는 새로운 벤치마크를 도입하여 다국어 임베딩 성능을 평가합니다.

- **Performance Highlights**: 실험 결과 LUSIFER는 다양한 임베딩 작업에서 다국어 성능을 크게 향상시키며, 특히 중간 및 저자원 언어의 경우 22.15 포인트까지의 향상을 기록했습니다. LUSIFER는 영어 중심 모델보다 평균 5.75 개선된 성능을 보이며 다국어 표현 능력을 강화하는 데 효과적임을 입증했습니다. 이 연구는 다국어 감독 없이도 효과적인 다국어 표현 능력을 향상시키는 LUSIFER의 효용성을 보여줍니다.



### DiffETM: Diffusion Process Enhanced Embedded Topic Mod (https://arxiv.org/abs/2501.00862)
Comments:
          5 pages, 2 figures, Accepted by ICASSP 2025

- **What's New**: 이 논문은 기존의 embedded topic model (ETM)에서 문서-주제 분포의 로지스틱 정규 분포 가정으로 인해 발생하는 성능 한계를 극복하기 위한 새로운 방법을 제안합니다. 제안된 방법은 확산 과정 (diffusion process)을 샘플링 과정에 통합하여 문서-주제 분포를 모델링하고 최적화 과정을 쉽게 유지할 수 있게 합니다. 우리가 제안한 모델은 두 개의 주요 데이터셋에서 주제 모델링 성능을 향상시키는 데 효과적임을 입증하였습니다.

- **Technical Details**: 이 모델에서는 문서 표현에서 직접 샘플링을 수행하여 고유한 문서-주제 분포를 생성합니다. 기존의 ETM과 비교할 때, 이러한 시도는 문서 정보가 포함된 숨겨진 표현을 통합하여 더 나은 모델링을 가능하게 합니다. 모델은 세 가지 메트릭인 주제 일관성 (topic coherence), 주제 다양성 (topic diversity), 그리고 혼잡도(perplexity) 기준에서 성능 향상을 달성합니다.

- **Performance Highlights**: 제안된 모델은 20Newsgroup 및 New York Times 데이터셋에서 기본 및 최첨단 ETM과 비교할 때 유의미한 성과를 나타냈습니다. 이 연구는 확산 과정을 ETM에 통합하여 문서-주제 분포의 표현 능력을 향상시키는 첫 번째 시도로, 새로운 접근법이 기여할 수 있는 잠재력을 보여줍니다. 결과적으로, 새로운 기술이 주제 모델링에서의 성능을 향상시키는 데 기여하고 있음을 확인했습니다.



### Navigating Nuance: In Quest for Political Truth (https://arxiv.org/abs/2501.00782)
Comments:
          Accepted at JCDL 2024

- **What's New**: 이 연구는 정치적 편향을 탐지하기 위한 새로운 접근법인 Llama-3 (70B) 언어 모델을 Media Bias Identification Benchmark (MBIB)에서 평가했습니다. 특히, subtle reasoning을 반영한 새로운 prompting 기법을 도입하여 정치적 편향을 더욱 효과적으로 식별할 수 있는 가능성을 보여줍니다. 연구 결과, 본 프레임워크는 기존의 최첨단 모델인 ConvBERT와 비교할 만한 성능을 보여주며, 정보의 왜곡과 정치적 양극화를 완화하는 도구 개발에 기여하고자 합니다.

- **Technical Details**: 정치적 편향 탐지는 자연어 처리(NLP) 분야에서 중요한 연구 영역으로, 현재 심층 학습과 대형 언어 모델(LLMs)을 활용한 최근 기법들이 개발되고 있습니다. 본 연구에서 사용된 Llama-3 모델은 다양한 벤치마크에서 성능이 입증되었으며, 특히 복잡한 추론 및 이해 작업에서 다른 모델들을 능가합니다. 실험에는 zero-shot, few-shot prompting과 함께 Chain-of-Thought (CoT) prompting을 활용하여 모델의 성능을 극대화했습니다.

- **Performance Highlights**: 조사 결과, Llama-3는 기존의 모델들과 견주어 높은 정확도로 정치적 편향을 탐지할 수 있었으며, 이는 다양한 데이터셋과 정치적 맥락에서도 일반화 가능한 성능을 가지는 것으로 나타났습니다. 또한, CoT prompting을 통해 LLM의 해석 가능성과 정확성을 높이는 데 성공하며, 복잡한 작업에서의 신뢰성을 확보했습니다. 최종적으로, 본 연구는 자동화된 편향 탐지 솔루션 개발의 새로운 가능성을 제시합니다.



### Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines (https://arxiv.org/abs/2501.00745)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 검색 엔진의 정보 검색 환경의 변화를 다룬다. 특히, LLM이 검색 엔진에 통합됨에 따라 발생하는 새로운 취약성과 공격, 특히 랭킹 조작 공격에 대한 연구를 진행한다. 이를 통해 공격자들이 웹페이지 콘텐츠를 조작하여 순위를 조작하는 방법과 그로 인해 발생하는 비즈니스 상의 불공정한 이점에 대해 논의한다.

- **Technical Details**: 이 연구에서는 랭킹 조작 공격의 동학을 무한 반복 죄수의 딜레마(Infinitely Repeated Prisoners' Dilemma)라는 게임 이론의 관점에서 분석한다. 공격자들은 전략적으로 협력할지 공격할지를 결정하며, 이 과정에서 공격 비용, 공격 성공률, 할인율 등 여러 요소가 플레이어 행동에 미치는 영향을 살펴본다. 협력이 지속될 수 있는 조건을 식별하고, 적응형 보안 전략과 생태계 설계의 중요성을 강조한다.

- **Performance Highlights**: 연구 결과, 협력이 지속될 가능성이 높아지는 경계 조건과 각 요인이 협력 장기 지속성에 미치는 영향을 보여준다. 흥미롭게도, 공격 성공률이 중간 수준일 때 공격의 이익과 위험이 최적화되어 협력을 저해할 수 있음을 발견했다. 또한, 방어 조치를 통해 공격 성공률을 제한하려는 시도가 오히려 공격 유인을 증가시킬 수 있다는 점도 강조한다.



### Fine-grained Video-Text Retrieval: A New Benchmark and Method (https://arxiv.org/abs/2501.00513)
- **What's New**: 이번 논문에서 제시하는 FIBER는 비디오-언어 검색(video-language retrieval) 모델의 세밀한 성능 평가를 가능하게 하는 새로운 벤치마크입니다. 기존의 MSRVTT 및 MSVD와 같은 비디오 검색 벤치마크는 세밀한 주석 부족으로 인해 효과적으로 성능을 평가하지 못했습니다. FIBER는 FineAction 데이터세트에서 소싱된 1,000개의 비디오와 함께 상세한 사람 주석(spatial annotations) 및 시간 주석(temporal annotations)을 제공합니다.

- **Technical Details**: FIBER 벤치마크는 비디오 검색 과제에서 비디오-언어 모델의 공간적 및 시간적 편향(spatial and temporal bias)을 독립적으로 평가할 수 있게 합니다. 이 연구는 Multimodal Large Language Models (MLLMs)의 세밀한 비디오-언어 이해를 위한 텍스트 임베딩 방법(text embedding)을 활용하였습니다. 이를 통해 비디오-언어 모델은 더욱 효과적으로 비디오를 검색할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 Video Large Language Encoder (VLLE)는 전통적인 벤치마크에서 CLIP 기반 모델과 비슷한 성능을 보였습니다. 더욱 놀라운 점은 VLLE가 더 낮은 공간-시간 편향(spatial-temporal bias)으로 세밀한 표현력이 더 뛰어난 성능을 발휘했다는 것입니다. 이는 VLLE가 비디오-언어 검색 작업에서 우수한 능력을 갖춘 모델임을 시사합니다.



### MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation (https://arxiv.org/abs/2501.00332)
- **What's New**: 이번 연구에서는 Multi-Agent Filtering Retrieval-Augmented Generation (MAIN-RAG)이라는 새로운 RAG 프레임워크를 제안합니다. 이 방법은 여러 LLM 에이전트를 활용하여 문서를 필터링하고 평가하는 협업적 접근 방식을 사용하여, 보다 정확하고 신뢰성 있는 응답을 제공합니다. 특히, MAIN-RAG는 추가적인 훈련 없이도 작동할 수 있도록 설계되어 실제 응용 프로그램에서의 강력한 확장성을 자랑합니다.

- **Technical Details**: MAIN-RAG는 동적으로 조정되는 적응형 필터링 메커니즘을 도입하여, 검색된 문서의 점수 분포에 따라 관련성 필터링 임계값을 조정합니다. 이를 통해 노이즈를 효과적으로 최소화하면서도 관련 문서의 회수 비율을 높이고, 여러 쿼리에 대한 robust한 성능을 보장합니다. 각 문서에는 관련성 점수가 할당되며, 이 점수를 기반으로 문서의 순위를 매기고 불필요한 문서를 필터링합니다.

- **Performance Highlights**: MAIN-RAG는 4개의 QA 벤치마크에서 실험을 통해 기존의 RAG 방법보다 2-11% 향상된 답변 정확도를 기록했습니다. 또한 불필요한 문서의 수를 줄이고, 응답의 일관성과 정확성을 향상시는 등 훈련 기반 솔루션에 대한 경쟁력 있는 대안으로 자리매김할 수 있음을 보여주었습니다.



### Exploring the Implicit Semantic Ability of Multimodal Large Language Models: A Pilot Study on Entity Set Expansion (https://arxiv.org/abs/2501.00330)
Comments:
          ICASSP 2025

- **What's New**: 본 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 다중 모달 엔티티 세트 확장(Multi-modal Entity Set Expansion, MESE) 작업에 적용하여 기존의 한계점을 보완하고, 이 도구의 능력을 탐색합니다. 새로운 엔티티를 시맨틱 클래스에 맞춰 탐색하는 MESE 작업을 통해 MLLM의 암묵적 의미 정보 추출 능력을 평가합니다. 특히, LUSAR라는 리스트 순위 매김 방법을 도입하여 지역 점수를 글로벌 순위로 매핑하는 방식을 제안합니다.

- **Technical Details**: LUSAR 방법론은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 접두사 트리 제약을 사용하여 모델이 지정된 엔티티 데이터 세트 내의 많은 후보 엔티티를 생성하도록 제한합니다. 두 번째 단계에서는 리스트별 접근 방식을 도입하여 각 후보 엔티티에 대한 순위 점수를 얻기 위해 여러 샘플링 및 순위를 수행합니다. 이러한 접근은 암묵적인 정보에서 세부적으로 구분된 의미 기능을 추출하는 데 도움을 줍니다.

- **Performance Highlights**: LUSAR 방식의 적용을 통해 MLLM의 MESE 작업에서 성능 향상이 크게 이루어졌습니다. 이 방식은 대형 모델이 암묵적인 의미 정보를 추출하는 데 어려워하는 문제를 해결하고, 성공적인 후보 엔티티 선정에 기여합니다. 경량의 리스트 방식 접근을 통하여 추천 시스템과 같은 다른 작업에도 적용 가능성을 보여주며, 실험 결과로 MLLM의 성능이 크게 향상된 것을 확인하였습니다.



### Towards Pattern-aware Data Augmentation for Temporal Knowledge Graph Completion (https://arxiv.org/abs/2501.00252)
- **What's New**: 이 논문에서는 Temporal Knowledge Graph Completion (TKGC)에서 발생하는 데이터 불균형과 모델 선호 문제를 다루기 위해 Booster라는 첫 번째 데이터 증강 전략을 제안합니다. 기존의 방법들이 이러한 두 가지 문제를 간과하고 있다는 점에 주목하고, 데이터의 복잡한 의미론적 및 시간적 패턴을 반영하는 새로운 샘플 생성을 목표로 설정합니다.

- **Technical Details**: Booster는 TKG의 서로 다른 구성 요소에 맞춤화된 세 가지 빈도 기반 필터링 전략을 사용하여 잠재적인 false negatives를 적절히 걸러냅니다. 이후 계층적 스코어링 알고리즘을 통해 hard negatives와 false negatives를 분류하고, 이를 통해 TKG의 전역 의미 패턴과 지역 그래프 구조의 최근 경향에 맞는 샘플을 구별합니다.

- **Performance Highlights**: 실험 결과, Booster는 기존 TKGC 모델에 원활하게 적응하며, 최대로 8.7%의 성능 향상을 달성했습니다. 평균적으로, 기존의 Temporal Graph 및 Knowledge Graph 데이터 증강 기술을 초과하는 7.1%의 향상을 보였고, 기존 TKGC 모델의 성능 변동성을 평균 22.8% 감소시키는 데 기여했습니다.



### CancerKG.ORG A Web-scale, Interactive, Verifiable Knowledge Graph-LLM Hybrid for Assisting with Optimal Cancer Treatment and Car (https://arxiv.org/abs/2501.00223)
- **What's New**: 이번 연구에서는 colorectal Cancer에 대한 최신 동료 검토(peer-reviewed) 의학 지식으로 구성된 최초의 Web-scale hybrid Knowledge Graph(KG)-Large Language Model(LLM) 중 하나를 소개합니다. 이 모델은 미국 및 세계 최고의 암 센터 중 하나인 Moffitt Cancer Center에서 의학 연구 및 임상 정보 검색 작업을 지원하기 위해 평가되고 있습니다. 기존의 LLM, KG, 검색 엔진의 단점을 극복하고 사용자 요구를 더 잘 충족시키는 하이브리드 모델이 개발되었습니다.

- **Technical Details**: 이 하이브리드 모델은 LLM이 가지는 환각(hallucinations)과 재앙적인 망각(catastrophic forgetting)의 문제를 해결하고자 제작되었습니다. 최신의 상태에서의 KG인 PrimeKG, cBioPortal, ChEMBL 등은 수동 큐레이션(manual curation)을 필요로 해 빠르게 오래된 정보를 포함하게 됩니다. 반면, CancerKG는 비지도 학습(unsupervised)에 기반하여 최신 의학 발견을 자동으로 수집하고 조직화할 수 있습니다.

- **Performance Highlights**: CancerKG는 사용자의 편의성을 고려하여 서로 다른 데이터 모달리티(data modalities)를 효과적으로 처리하는 5가지의 고급 사용자 인터페이스를 제공합니다. 이 모델은 LLM의 문제점을 완화하기 위해 검증된 KG를 Retrieval Augmented Generation(RAG) 방식으로 활용하고 있습니다. 이로 인해 사용자는 더욱 향상된 의료 정보 검색 경험을 누릴 수 있습니다.



### The Text Classification Pipeline: Starting Shallow going Deeper (https://arxiv.org/abs/2501.00174)
- **What's New**: 이번 논문은 Text Classification (TC)의 전체 파이프라인을 상세히 탐구합니다. 특히 각 구성 요소가 TC 모델의 성능에 미치는 영향에 대해 철저하게 검토하였습니다. 더불어 최신 데이터셋과 텍스트 전처리 기법을 포함한 다양한 기술 혁신을 소개합니다.

- **Technical Details**: 논문은 TC의 다양한 단계—데이터셋, 전처리 기법, 텍스트 표현 방법, 분류 모델, 평가 메트릭, 결과 및 미래 동향 등을 포괄적으로 다룹니다. 각 장에서는 이론과 함께 실험적 평가 및 사례 연구를 제공하여 보다 깊은 이해를 돕습니다. 이러한 기술적 디테일은 TC의 효과적인 구현에 매우 중요합니다.

- **Performance Highlights**: 분류 전략에 대한 비판적 평가와 비교 분석을 통해 독자에게 다양한 접근 방법의 강점과 약점을 인식시킵니다. 이 연구는 단순한 조사를 넘어서 TC 분야에서의 중요한 최근 발견을 조명하며, 향후 연구 방향에 대한 통찰을 제공합니다. 결과적으로, 이 논문은 TC의 전문성과 이해도를 높이는 데 기여하고 있습니다.



### Crime Hotspot Analysis and Mapping Using Geospatial Technology in Dessie City, Ethiopia (https://arxiv.org/abs/2501.00036)
- **What's New**: 이번 연구는 지리적 기술(geographic technology)을 활용하여 데시에(Dessie) 시의 범죄 패턴을 매핑하고 분석했습니다. 범죄 트렌드를 지리적으로 조사하는 데 초점을 맞추었으며, 연구 결과 범죄의 '핫스팟'(hot spot) 지역이 확인되었습니다. 이렇게 지리적 데이터에 기반한 접근은 범죄 예방과 자원 배치에 효과적인 방안을 제시할 수 있습니다.

- **Technical Details**: 연구자들은 세미바리오그램 모델링(semivariogram modeling)과 모란의 I(Moran's I)를 이용한 공간 자기상관(spatial autocorrelation) 분석을 실시하였습니다. 데시 중심부의 호테(Hote), 아라다(Arada), 세그노(Segno) 지역은 높은 Z-점수(0.037~4.608)로 범죄가 많이 발생하는 지역으로 확인되었으며, 반대로 북중부의 메나페샤(Menafesha) 및 부운부하(Bounbouwha) 지역은 낮은 Z-점수(-3.231~-0.116)를 보여 저조한 범죄 발생이 관찰되었습니다.

- **Performance Highlights**: 최종 분석에서 0.027492의 지수와 3.297616의 Z-점수(p<0.01)를 기록하여 데시 내 범죄의 긍정적 공간 자기상관을 크게 나타냈습니다. 대부분의 범죄는 남북 방향으로 발생했으나, 살인의 경우 북동에서 남서 방향으로의 경향을 보였습니다. 이러한 결과는 데시와 같은 개발 도시에서 범죄 사건의 클러스터 패턴을 강조하고, 정책 결정에 기여할 수 있습니다.



### A Breadth-First Catalog of Text Processing, Speech Processing and Multimodal Research in South Asian Languages (https://arxiv.org/abs/2501.00029)
- **What's New**: 본 연구는 최근 2022년 1월부터 2024년 10월까지의 남아시아 언어에 대한 문헌을 검토하고, 21개의 낮은 자원(low-resource) 언어에 대한 집중 분석을 제공합니다. 이 논문은 텍스트 기반 언어 처리, 다중 양식(multimodal) 모델 및 음성 처리에 관한 최신 동향, 문제 및 향후 연구 방향을 식별합니다. 우리는 대규모 언어 모델(LLMs)을 활용한 적절성 분류 및 클러스터링을 포함하는 단계별 접근 방식을 사용하였습니다.

- **Technical Details**: 이 연구에서는 Google Scholar와 Publish or Perish 소프트웨어를 활용하여 논문을 검색하고 분류하였습니다. 이 과정에서 GPT-4o를 사용해 공감대 구축과 관련성 라벨을 예측하였으며, O1 모델을 통해 발견된 논문을 주제별로 그룹화하는 방식을 채택하였습니다. 결과적으로 세 가지 분야 각각에 대한 연구 결과가 표 형태로 요약되었습니다.

- **Performance Highlights**: 본 논문에서 확인된 주요 발견은 변화하는 언어 모델의 경향과 문제점들이며, 저자들은 기계 번역, 감정 분석, 편향과 공정성 연구 등 여러 주제를 다루었습니다. 남아시아 지역의 언어적 편향과 차별성을 반영하지 못한 선행 연구의 한계를 강조하며, 향후 연구를 위한 데이터 세트 및 감정 분석의 필요성을 지적합니다. 부문 간 협업과 지식 공유를 통해 더욱 발전된 언어 기술이 필요하다고 결론짓습니다.



New uploads on arXiv(cs.CV)

### GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models (https://arxiv.org/abs/2501.01428)
Comments:
          Project page: this https URL

- **What's New**: 최근 연구에서 2D Vision-Language Models (VLMs)는 이미지-텍스트 이해 작업에서 상당한 발전을 이루었지만, 3D 공간 이해에서는 한계가 있습니다. 본 논문에서는 이러한 한계를 극복하기 위해 GPT4Scene이라는 새로운 비전 기반 접근 방식을 제안하며, 인체 인식에서 영감을 받아 시각적 단서만으로 3D 공간을 이해하는 방법을 모색합니다. GPT4Scene은 비디오에서 Bird's Eye View (BEV) 이미지를 구축하고, 이를 통해 3D 실내 장면의 이해를 향상시킵니다.

- **Technical Details**: GPT4Scene은 3D BEV 이미지를 기반으로 Spatial-Temporal Object markers (STO markers)를 사용하여 실제 장면의 구조와 객체 간의 공간적 관계를 설정합니다. 이러한 방식은 VLMs가 비디오 프레임과 BEV 이미지 간의 글로벌-로컬 관계를 효과적으로 구축할 수 있도록 도와주며, 이는 기존의 3D 포인트 LLMs와 비교하여 성능 향상을 이끌어냅니다. 이 프레임워크는 특히 영상 기반 입력을 활용하여, 복잡한 3D 정보가 없이도 실내 장면 이해에 적합한 모델을 증진시킵니다.

- **Performance Highlights**: 본 연구의 결과 GPT4Scene은 강력한 폐쇄형 VLMs인 GPT-4o와의 제로샷 평가에서 성능 개선을 이뤘습니다. 또한, ScanAlign 데이터세트를 활용하여 오픈 소스 VLM들을 미세 조정함으로써, 모든 3D 이해 작업에서 최신 성능을 기록했습니다. 놀랍게도 GPT4Scene을 통해 훈련된 VLMs는 명시적인 시각적 단서 없이도 계속해서 성능 향상을 보였으며, 이는 3D 장면 이해에 대한 본질적인 능력을 개발함을 나타냅니다.



### VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Contro (https://arxiv.org/abs/2501.01427)
Comments:
          Method for object insertion in videos

- **What's New**: 이 논문에서는 VideoAnydoor라는 새로운 제로샷(zero-shot) 비디오 객체 삽입 프레임워크를 제안합니다. 이 프레임워크는 높은 충실도의 세부 사항 보존과 정확한 동작 제어를 가능하게 하여 주목을 받았습니다. 기존의 접근법과 달리, VideoAnydoor는 영상 속에서 특정 객체를 자연스럽게 삽입하는 능력을 갖추고 있으며, 다양한 다운스트림(downstream) 응용 프로그램을 효과적으로 지원합니다.

- **Technical Details**: VideoAnydoor는 텍스트를 비디오로 변환하는 모델을 바탕으로 하며, ID 추출기(ID extractor)를 통해 객체의 전반적인 정체성을 주입하고 박스 시퀀스를 활용하여 전반적인 동작을 제어합니다. 견고하고 정교한 동작 제어를 위해, 우리는 픽셀 변형기(pixel warper)를 설계하였으며, 이는 키포인트(key-point)와 해당 경로를 입력받아 픽셀 세부 사항을 변형합니다. 이와 함께, 비디오 품질을 향상시키기 위해 이미지-비디오 혼합 훈련을 포함한 다양한 전략이 도입되었습니다.

- **Performance Highlights**: VideoAnydoor는 기존의 방법들에 비해 향상된 성능을 보이며, 복잡한 비디오 편집 작업을 치밀하게 수행할 수 있습니다. 특히 이 프레임워크는 사용자 지시를 기반으로 콘텐츠와 동작을 정확하게 수정 가능한 최초의 엔드 투 엔드 방법입니다. 사용자들은 특정 영역에서 목표 이미지를 제공하고 이를 기반으로 편집할 수 있으며, 다양한 형태와 외형의 객체를 자유롭게 삽입할 수 있는 강력한 특성을 나타냅니다.



### Unifying Specialized Visual Encoders for Video Language Models (https://arxiv.org/abs/2501.01426)
Comments:
          Project page: this https URL

- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전은 비디오 영역에서도 정교한 추론 능력을 선보였습니다. 이에 따라 Video Large Language Models(VideoLLMs)가 등장했지만, 현재 VideoLLMs는 단일 비전 인코더를 사용하여 모든 비주얼 처리(visual processing)를 수행함으로써 전달할 수 있는 비주얼 정보의 양과 종류에 제한이 있습니다. 본 연구에서는 다중 인코더 표현(Multi-Encoder Representation of Videos, MERV)을 소개하여, 여러 개의 동결된 비전 인코더를 활용하여 비디오의 통합 표현을 생성함으로써 VideoLLM에 보다 포괄적인 전문 비주얼 지식을 제공합니다.

- **Technical Details**: MERV는 각 인코더의 기능(feature)을 시공간적으로 정렬(spatio-temporally aligning)함으로써 보다 다양한 열려있는(open-ended) 및 다중 선택(multiple-choice) 비디오 이해 질문을 해결할 수 있게 되었고, 이전의 최상급(state-of-the-art) 작업들을 초월할 수 있었습니다. 특히, MERV는 표준 비디오 이해 벤치마크에서 Video-LLaVA에 비해 정확도에서 최대 3.7% 향상된 결과를 보였으며 Video-ChatGPT에서도 더 나은 점수를 기록했습니다. 또한, 이전에 제일 좋았던 제로샷 인식 테스트(zero-shot Perception Test) 정확도를 2.2% 개선했습니다.

- **Performance Highlights**: MERV는 기존의 단일 인코더 방법들과 비교하여 최소한의 추가 파라미터(minimal extra parameters)로 보다 빠르게 훈련(train faster)할 수 있으며, 비주얼 처리를 병렬화(parallelizing)하여 효율성을 높였습니다. 실험 결과는 MERV가 각 인코더로부터 도메인 지식을 효과적으로 포착(captures domain knowledge)할 수 있음을 질적으로 증명하였습니다. 이러한 결과들은 다중 비전 인코더를 활용한 포괄적인 비디오 이해(video understanding)에서 유망한 방향을 제시합니다.



### Free-Form Motion Control: A Synthetic Video Generation Dataset with Controllable Camera and Object Motions (https://arxiv.org/abs/2501.01425)
Comments:
          Project Page: this https URL

- **What's New**: 최근 동적 객체와 카메라의_MOVEMENT_를 제어하는 것이 비디오 생성에서 중요한 과제로 떠오르고 있다. 그러나 고품질 데이터셋의 부족으로 인해 기존 알고리즘은 두 요소를 동시에 효과적으로 조절하기 어렵다. 이를 해결하기 위해 저자들은 Free-Form Motion Control을 위한 합성 데이터셋인 SynFMC를 새롭게 소개하였다.

- **Technical Details**: SynFMC 데이터셋은 다양한 객체와 환경을 포함하며, 특정 규칙에 따른 다양한 MOVEMENT_ 패턴을 커버하여 복잡한 실제 시나리오를 모방한다. 이 데이터셋은 화면 공간에서의 객체와 카메라의 상호작용을 분리할 수 있는 전체 6D 포즈 정보를 제공한다. 이를 통해 Free-Form Motion Control (FMC) 방법론을 통해 객체와 카메라의 MOVEMENT_을 독립적으로 제어할 수 있게 된다.

- **Performance Highlights**: 저자들은 SynFMC 데이터셋을 이용해 FMC의 효과성을 검증하였으며, 이 방법은 기존의 최신 기법들과 비교하여 여러 시나리오에서 우수한 성능을 보였다. FMC는 텍스트 기반의 이미지 생성 모델과 호환 가능하여 유연한 사용자 인터페이스를 제공하며, 사용자가 직접 크기를 입력하거나 곡선을 그릴 수 있는 기능을 지원한다.



### Object-level Visual Prompts for Compositional Image Generation (https://arxiv.org/abs/2501.01424)
Comments:
          Project: this https URL

- **What's New**: 이번 연구는 텍스트-이미지 확산 모델 내에서 객체 수준의 시각적 프롬프트(composition)를 조합하는 방법을 소개합니다. 이 접근 방식은 다양한 장면과 스타일을 아우르는 의미적으로 일관된 구성 생성을 목표로 하며, 텍스트 프롬프트가 제공하는 다재다능함과 표현력을 구현하고자 합니다. 키와 값을 각각 다른 시각적 표현으로부터 학습하는 KV-mixed cross-attention 메커니즘을 도입하여, 객체의 정체성을 유지하면서도 다양한 구성을 생성할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 인코더를 사용하는 KV-mixed cross-attention 모듈을 제안합니다. 작은 보틀넥을 가진 인코더는 레이아웃 컨트롤에 사용되는 키(key)를 생성하고, 더 큰 보틀넥 인코더는 세부적인 외관 정보를 담고 있는 값(value)을 생성합니다. 이러한 두 가지 정보 소스를 혼합하여 시각적 프롬프트의 정체성을 유지하면서도 유연한 변형을 가능하게 합니다. 또한, 역추론(inference) 단계에서는 Compositional Guidance를 통해 객체 수준의 가이드를 제공하여 정체성 보존과 레이아웃 일관성을 강화합니다.

- **Performance Highlights**: 이 방법은 다채로운 장면 구성을 생성하면서 각 시각적 프롬프트의 고유한 특성을 유지하는 데 성공합니다. 결과적으로 본 연구의 기법은 기존의 이미지 프롬프트 방법, 최적화 기법 및 다중 모달 생성 방법에 비해 뛰어난 성능을 보이며, 텍스트-이미지 생성의 창의적 잠재력을 확장합니다. 다양한 개체 배열 및 장면 구성을 유지하면서도 일관성 있고 세부적인 이미지를 생성하는 데 기여할 것으로 기대됩니다.



### Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models (https://arxiv.org/abs/2501.01423)
Comments:
          Models and codes are available at: this https URL

- **What's New**: 최근 연구에 따르면, Transformer 아키텍처를 사용하는 잠재 확산 모델은 고충실도 이미지를 생성하는 데 뛰어난 성능을 보입니다. 그러나 이들 모델의 두 단계 디자인에서 최적화의 딜레마가 발견되었습니다. 특히, 시각 토크나이저의 기능 차원을 증가시키면 재구성 품질이 향상되지만, 이에 대한 대가로 더 큰 확산 모델과 더 많은 훈련 반복이 필요하게 됩니다. 본 논문에서는 이러한 문제를 해결하기 위해 미리 학습된 비전 기반 모델과 잠재 공간을 정렬하는 방법을 제안합니다.

- **Technical Details**: 제안하는 VA-VAE(비전 모델 정렬 변분 오토인코더)는 잠재 확산 모델의 재구성-생성 경계를 크게 확장하며, Diffusion Transformers(DiT)의 고차원 잠재 공간에서의 빠른 수렴을 가능하게 합니다. 우리의 핵심 기술 기여는 비전 기반 모델 정렬 손실(VF Loss)로, 토크나이저 훈련 중 잠재 표현을 미리 학습된 모델과 정렬합니다. 이 손실은 높은 차원의 잠재 공간을 규제하면서도 그 용량을 지나치게 제한하지 않도록 설계되었습니다.

- **Performance Highlights**: 통합 시스템은 ImageNet 256x256 생성에서 FID 점수 1.35를 달성하며, 단 64 에폭 내에 FID 점수 2.11에 도달하여 원래 DiT보다 21배 이상 수렴 속도가 빨라졌습니다. 이는 잠재 확산 모델의 최적화 딜레마를 효과적으로 해결하여, 고차원 토크나이저와 함께 2.5배 빠른 DiT 훈련을 가능하게 하는 성과를 나타냅니다.



### Multi-Modal Video Feature Extraction for Popularity Prediction (https://arxiv.org/abs/2501.01422)
Comments:
          INFORMS 2024 Data Challenge Competition

- **What's New**: 이 연구는 짧은 동영상의 인기 예측을 위해 동영상 자체와 관련된 기능을 사용하는 혁신적인 접근법을 보여줍니다. 인기도는 조회수(view count), 좋아요 수(like count), 댓글 수(comment count), 공유 수(share count)의 네 가지 주요 참여 지표로 측정됩니다. 다양한 동영상 분류 모델을 Backbone Network로 활용하여 동영상 모달리티 기능을 추출하고, 정리된 동영상 캡션을 사용한 텍스트 생성 모델을 통해 동영상 내용을 이해합니다.

- **Technical Details**: 이 연구는 동영상 및 табular 데이터 기반으로 데이터 마이닝(data mining) 및 기능 엔지니어링(feature engineering)을 수행하여 해시태그 등장 빈도, 언급 빈도, 동영상 길이 등 여러 실용적인 기능을 구축했습니다. TimeSformer, ViViT, VideoMAE, X-CLIP과 같은 최신 Neural Network 모델을 사용하여 동영상 특징을 추출하였고, XGBoost와 결합해 예측 정확도를 높였습니다. 또한, 결측치(missing values)는 중앙값 중복(median imputation)으로 처리하였으며 특징의 로그 변환(logarithmic transformation)을 통해 모델의 안정성을 향상시켰습니다.

- **Performance Highlights**: 다양한 모델의 훈련을 통해 가장 안정성이 높은 XGBoost 모델이 최종적으로 선택되었습니다. 최종 예측 결과는 Neural Network와 XGBoost 모델의 예측치를 평균하여 산출되는 형태로, 예측 정확도는 평균 절대 비율 오차(MAPE)로 평가되었습니다. 이 연구는 짧은 동영상의 인기 예측 분야에서 의미 있는 발전을 이루었으며, 사용자 수와 비디오 특성을 고려하여 다양한 요소가 결합되는 점에서 큰 가치를 제공합니다.



### R-SCoRe: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization (https://arxiv.org/abs/2501.01421)
Comments:
          Code: this https URL

- **What's New**: 이 연구는 Scene Coordinate Regression (SCR) 방법을 기반으로 한 새로운 Robust Scene Coordinate Regression (R-SCoRe) 프레임워크를 도입하여 복잡한 조명 변화가 있는 데이터셋에서도 강력한 시각적 위치 추정이 가능하도록 하였습니다. R-SCoRe는 깊이 조정 재투영 손실(depth-adjusted reprojection loss)을 활용하여, 네트워크가 근거리 포인트를 간과하지 않도록 조정합니다. 이 방법을 통해 기존 SCR 방법들보다 더 정확한 결과를 도출하며, 대규모 데이터셋에서 크기가 작은 맵을 유지하면서도 우수한 정확도를 기록하고 있습니다.

- **Technical Details**: R-SCoRe는 co-visibility graph 기반의 글로벌 인코딩 학습과 데이터 증강(data augmentation) 전략을 사용하며, implicit triangulation을 가능하게 하는 깊이 조정 재투영 손실을 도입합니다. 이 연구는 네트워크 아키텍처와 지역적 특징 추출 모듈을 재검토하여 설계 원칙을 개선했습니다. 또한, R-SCoRe는 복잡한 대규모 장면에서도 이전의 SCR 방법들보다 10배 더 높은 정확도를 자랑합니다.

- **Performance Highlights**: Aachen Day-Night 데이터셋에서 R-SCoRe는 (0.25m, 2°)의 기준에 따라 64.3%의 정확도를 달성하였으며, 맵 크기는 단 47MB에 불과합니다. R-SCoRe의 성능은 다른 SCR 방법들에 비해 뛰어나며, 특징 매칭 기술과 비교할 때도 비슷한 수준의 정확도를 제공합니다. 이 연구는 R-SCoRe가 3D 감독 없이는 이전의 SCR 방법들보다 최소한 한 가지 정밀도를 높이며, 빠른 추론 속도와 더 작은 맵 크기를 유지하는 데 기여한다고 강조하고 있습니다.



### A Multi-task Supervised Compression Model for Split Computing (https://arxiv.org/abs/2501.01420)
Comments:
          Accepted at WACV 2025. Code and models are available at this https URL

- **What's New**: 이 논문에서는 멀티태스크(멀티 작업)를 위한 첫 번째 감독하의 압축 모델인 Ladon을 제안합니다. 기존의 스플릿 컴퓨팅(split computing) 방법들은 단일 작업에 초점을 맞추었으며, 멀티태스크 문제에 대한 모델 정확도가 저하되는 문제를 해결하고자 합니다. 새 모델은 이미지 분류, 물체 감지 및 의미론적 분할과 같은 다양한 작업을 빠르고 에너지 효율적으로 처리할 수 있습니다.

- **Technical Details**: Ladon 모델은 감독 압축(supervised compression) 기법을 사용하여 여러 작업을 수행할 수 있도록 설계되었습니다. 이 모델은 인코더의 크기, 압축된 데이터 크기 및 모델 정확성을 기존의 경량 모델들과 비교하여 평가합니다. 추가적으로, 멀티태스크 분할 컴퓨팅 상황에서의 종단 간 지연 시간(end-to-end latency)과 모바일 장치의 에너지 소비를 고려하여 최적화된 성능을 실현합니다.

- **Performance Highlights**: 실험 결과, Ladon 모델은 ILSVRC 2012, COCO 2017 및 PASCAL VOC 2012 데이터셋에서 강력한 경량 기본 모델들과 비교하여 예측 성능이 우수하거나 경쟁력 있는 성능을 보였습니다. 또한, 이 새로운 접근 방식은 다중 작업 스플릿 컴퓨팅 시 종단 간 지연 시간을 최대 95.4%까지 줄이고 모바일 장치의 에너지 소비는 최대 88.2%까지 줄였습니다.



### Hierarchical Alignment-enhanced Adaptive Grounding Network for Generalized Referring Expression Comprehension (https://arxiv.org/abs/2501.01416)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 Generalized Referring Expression Comprehension (GREC) 문제를 다룹니다. GREC는 고전적인 Referring Expression Comprehension (REC)보다 더 복잡한 설정인 no-target 및 multi-target 표현을 포함하여 개체를 감지하는 데 중점을 둡니다. 이를 위해 새로운 Hierarchical Alignment-enhanced Adaptive Grounding Network (HieA2G)를 제안하여 다양한 표현을 유연하게 처리하고, 높은 성능을 달성합니다.

- **Technical Details**: HieA2G는 Hierarchical Multi-modal Semantic Alignment (HMSA) 모듈을 통해 세 가지 수준의 정렬을 구현합니다: word-object, phrase-object, text-image 정렬. 이 구조는 다중 측면의 상호작용을 통해 복잡한 상황에서의 grounding 능력을 향상시킵니다. Adaptive Grounding Counter (AGC)는 각 이미지-텍스트 쌍에 대한 출력 개체 수를 동적으로 결정하여, 다양한 수의 목표 개체를 효율적으로 처리합니다.

- **Performance Highlights**: HieA2G는 GREC 작업에서 새로운 최첨단 성능을 달성했으며, 다른 네 가지 작업인 REC, Phrase Grounding, Referring Expression Segmentation (RES), Generalized Referring Expression Segmentation (GRES)에서도 우수한 성능을 보입니다. 이를 통해 HieA2G의 뛰어난 일반화 능력과 잠재력을 강조하며, 다양한 시각적 기초 작업에 유용한 기초 기술로 자리매김할 수 있음을 증명합니다.



### On Unifying Video Generation and Camera Pose Estimation (https://arxiv.org/abs/2501.01409)
- **What's New**: 이번 연구는 동영상 생성 모델이 3D 인식을 잘 수행할 수 있는지 조사하며, OpenSora라는 비디오 생성 모델의 중간 feature가 카메라 자세 추정(camera pose estimation)에 어떻게 기여하는지를 탐구합니다. 연구팀은 JOG3R이라는 새로운 통합 모델을 제안하여 비디오 생성과 카메라 자세 추정을 동시에 수행할 수 있는 능력을 보여줍니다. 이 모델은 고급 비디오 생성 품질을 유지하면서 카메라 자세 추정 정확도를 높이는 데 강력한 성능을 발휘합니다.

- **Technical Details**: OpenSora는 Diffusion Transformer (DiT) 기반의 비디오 확산 모델로, 사전 훈련된 VAE 인코더를 통해 낮은 차원 잠재 공간에서 확산 과정을 수행합니다. 연구진은 발생된 중간 feature가 본래의 3D 인식을 얼마나 가지고 있는지를 평가하기 위해, DUSt3R와 같은 예측 모듈을 비디오 생성 네트워크에 연결하여 설계했습니다. 이를 통해 비디오 생성 모델이 카메라 추정을 위한 feature를 효과적으로 재사용할 수 있는지를 검증했습니다.

- **Performance Highlights**: 실험 결과, 비디오 생성 feature는 본래 약한 3D 인식을 가지고 있으나, 카메라 자세 추정을 위한 추가적 감독(supervision)을 통해 significantly 향상됨을 확인하였습니다. JOG3R 모델은 경기력 면에서도 state-of-the-art 솔루션과 경쟁할 수 있는 카메라 자세 추정치를 생성하면서 비디오 생성 품질도 유지합니다. 따라서 JOG3R 모델은 비디오 생성 및 3D 카메라 재구성을 동시에 수행할 수 있는 최초의 통합 모델로 주목받고 있습니다.



### Nested Attention: Semantic-aware Attention Values for Concept Personalization (https://arxiv.org/abs/2501.01407)
Comments:
          Project page at this https URL

- **What's New**: 이번 연구에서는 개인화된 텍스트-이미지 모델을 위한 새로운 메커니즘인 Nested Attention을 소개합니다. 이 메커니즘은 기존의 cross-attention layer에 풍부하고 표현력이 뛰어난 이미지 표현을 주입하여 주목할만한 주제 보호와 텍스트 프롬프트 준수를 동시에 가능하게 합니다. 특히, 여러 개인화된 주제를 단일 이미지에서 결합하는 기능을 강조하며, 이는 다양한 도메인에서 훈련이 가능하다는 장점을 가지고 있습니다.

- **Technical Details**: 제안된 Nested Attention 메커니즘은 두 개의 attention layer로 구성되어 있습니다. 첫 번째는 표준 text-to-image cross-attention layer로, 새로운 주제가 특정 텍스트 토큰에 연결됩니다. 두 번째, 'nested' attention layer는 각 생성 이미지 영역에 대한 주제 특징을 선택하여 지역적이고 쿼리 의존적인 attention 값을 생성합니다. 이를 통해 모델은 주제의 전반적인 외모를 단일 토큰에 인코딩하는 대신, 더 작은 의미적 시각 요소를 인코딩하고 필요에 따라 분산할 수 있습니다.

- **Performance Highlights**: 실험 결과, Nested Attention 접근법은 높은 개인화 보호를 달성하면서도 모델의 기존 사전 지식을 더 잘 유지하고 있는 것으로 나타났습니다. 이전의 주제 주입 방법들과 비교했을 때, 신원 유사성과 편집 가능성에서 우수한 성능을 보여주었습니다. 또한 테스트 시 여러 주제 이미지를 제공하여 성능을 더욱 향상시킬 수 있고, 정체성 혼합과 의미적 주제 변형과 같은 추가적인 애플리케이션을 가능하게 합니다.



### nnY-Net: Swin-NeXt with Cross-Attention for 3D Medical Images Segmentation (https://arxiv.org/abs/2501.01406)
Comments:
          MICCAI

- **What's New**: 이 논문에서는 nnY-Net이라는 새로운 3D 의료 이미지 분할 모델 구조를 제안합니다. 모델의 이름은 U-net 구조 하단에 크로스-어텐션 모듈을 추가하여 Y 구조를 형성하기 때문에 붙여졌습니다. 최신 SOTA 모델인 MedNeXt와 SwinUNETR의 장점을 통합하여 Swin Transformer를 인코더로, ConvNeXt를 디코더로 사용하여 혁신적으로 Swin-NeXt 구조를 설계하였습니다.

- **Technical Details**: 이 모델은 인코더의 가장 낮은 수준의 특성 맵을 Key와 Value로 사용하고, 병리 및 치료 정보와 같은 환자 특성을 Query로 사용하여 크로스 어텐션 모듈에서 어텐션 가중치를 계산합니다. 또한, dynUnet과 nnU-net 프레임워크를 기반으로 3D 이미지 분할에서 일부 전처리 및 후처리 방법과 데이터 증강 방법을 간소화하였습니다. 마지막으로, 우리는 DiceFocalCELoss를 구성하여 복셀 분류의 불균형 데이터 수렴을 개선하기 위한 훈련 효율성을 높입니다.

- **Performance Highlights**: 제안된 nnY-Net 모델은 기존 모델들과 비교하여 의료 이미지의 세분화 성능에서 더 나은 결과를 보여줄 것으로 기대됩니다. 특히 Swin-NeXt 구조의 통합으로 인해 이전의 성능 한계를 극복할 수 있는 가능성이 열렸습니다. 효율적인 어텐션 계산과 데이터 처리 방법의 최적화로 인해 훈련 시간과 연산 비용이 감소할 것으로 예상됩니다.



### Learning 3D Garment Animation from Trajectories of A Piece of Cloth (https://arxiv.org/abs/2501.01393)
Comments:
          Accepted by NeurIPS2024, 16 pages

- **What's New**: 이 논문에서는 기존의 의류 애니메이션 방법이 직면한 데이터 대규모 요구 사항과 일반화 문제를 극복하기 위해, 단일 천 조각에서 의류의 동작을 학습하는 분리된 프레임워크를 제안합니다. 'Energy Unit Network (EUNet)'라는 새로운 접근 방식을 통해, 의류의 탄성 행동을 에너지 관점에서 모델링 하고, 다양한 의류를 애니메이션하기 위해 학습된 법칙을 활용합니다. 이러한 접근법은 과거의 의류 데이터에 대한 의존도를 줄이고, 동적 유도 방식을 자연스럽게 지원합니다.

- **Technical Details**: EUNet은 관찰된 천에서 직접 물질의 구성 법칙을 포착하고, 무작위 노이즈로 인한 국소적 변화를 고려하여 에너지 변화를 예측합니다. 이 방법은 잠재적인 Deformation Patterns를 uniformly하게 설명하며, 에너지 최적화 기법을 통해 다양한 의류를 애니메이션할 수 있도록 합니다. 모델 트레이닝 과정에서 분석물리학 모델이나 차별적 시뮬레이터의 사전 정보를 필요로 하지 않으며, 실질적으로 에너지를 기반으로 한 방법론을 채택합니다.

- **Performance Highlights**: 실험 결과, EUNet에 의해 제약된 모델은 기존의 의류 데이터에 기반한 감독 방식으로 훈련된 모델보다 더 낮은 유클리드 오차를 보여주었으며, 신체적으로 그럴듯한 성과를 나타냈습니다. EUNet을 통해 얻은 다양한 의류 애니메이션은 장기 예측까지 효과적으로 수행될 수 있음을 보이며, 염색, 신축 및 접힘과 같은 다양한 변형 상황에서 안정적인 성능을 나타냅니다.



### Iris Recognition for Infants (https://arxiv.org/abs/2501.01375)
- **What's New**: 본 논문은 4-6주 된 영아를 위한 비접촉 식별 기술로써 홍채 인식(irys recognition)의 적용 가능성을 탐구하며, 세 가지 핵심 기여를 제안합니다. 첫째, 맞춤형 NIR (Near Infrared) 홍채 센서를 사용하여 17명의 신생아로부터 1,920개의 고유한 홍채 이미지를 수집했습니다. 둘째, 영아용으로 설계된 홍채 인식 기술을 개발하고, 셋째, 영아의 홍채 사진으로부터 생성된 합성 이미지를 연구 커뮤니티에 제공하여 아동의 프라이버시를 보호하고자 합니다.

- **Technical Details**: 이 연구는 홍채 인식의 여러 방법을 평가하였으며, 신생아를 위한 최초의 홍채 인식 시스템 구축을 목표로 하였습니다. 여기에는 ISO/IEC 19794-6에 준거한 홍채 이미지의 고유한 데이터 세트를 수집하는 것과, 영아의 홍채 이미지를 효과적으로 세분화할 수 있는 새로운 세분화 모델이 포함됩니다. 연구 결과, 제안된 시스템은 성인 홍채 인식 시스템보다 높은 성능을 보이며, EER(동등오류률) 3%와 ROC 곡선 아래 면적(AUC) 99%를 달성하였습니다.

- **Performance Highlights**: 제안된 시스템은 특히 신생아의 홍채 인식 분야에서 성공적인 성과를 입증하였습니다. 기존 성인 홍채 인식 시스템은 EER 20% 이상과 AUC 88% 이하의 성능을 보인 것에 비해, 본 연구에서 제안한 영아의 홍채 인식 기술은 현저히 높은 정확도를 자랑합니다. 이는 신생아의 홍채로부터 생체 인식 데이터를 효과적으로 추출할 수 있는 가능성을 보여 주며, 향후 영아의 안전한 식별 및 건강 모니터링 방법으로서의 활용 가능성을 제시합니다.



### CLIP-UP: CLIP-Based Unanswerable Problem Detection for Visual Question Answering (https://arxiv.org/abs/2501.01371)
- **What's New**: 최근에 제안된 CLIP-UP는 Vision-Language Models (VLMs)에게 정답이 없는 질문에 대해 답변을 보류할 수 있는 능력을 부여하는 경량화된 접근 방식입니다. 이 방법은 CLIP의 질문-이미지 정렬 정보를 활용하여, 모델의 원래 가중치를 변경하지 않고도 몇 개의 추가 레이어만 효율적으로 훈련하면 됩니다. CLIP-UP은 LLaVA 모델에서 테스트되어 MM-UPD 벤치마크에서 최첨단 성능을 달성했습니다.

- **Technical Details**: CLIP-UP은 CLIP의 임베딩을 기반으로 하는 상관관계 벡터를 활용하여 입력 이미지와 질문 간의 정렬 정보를 인코딩합니다. 이러한 벡터는 VLM의 중간 특성 공간으로 투영되어 새로운 임베딩 벡터를 생성하며, 모델에 통합됩니다. 이 과정에서 CLIP-UP은 몇 개의 선형 투영 레이어만 훈련하고, 기존 VLM의 가중치는 그대로 유지합니다.

- **Performance Highlights**: 실험 결과, CLIP-UP은 다양한 LLaVA 모델에서 UPD 성능을 획득하여 전체 모델의 파인튜닝과 비교할 만한 성능을 보였으며, 다른 작업에서 원래의 성능을 유지했습니다. 이를 통해 CLIP-UP은 정답이 없는 질문을 탐지하는 데 효과적인 방법임을 확인할 수 있었습니다.



### Test-time Controllable Image Generation by Explicit Spatial Constraint Enforcemen (https://arxiv.org/abs/2501.01368)
- **What's New**: 이 논문에서는 자연어와 복잡한 레이아웃을 활용한 새로운 테스트 타임 제어 생성 방법을 제안합니다. 이 방법은 공간 조건을 의미적(semantic) 조건과 기하학적(geometric) 조건으로 분리하고, 이미지 생성 과정에서 일관성을 유지합니다. 특히, 우리는 주어진 텍스트 프롬프트와 생성된 이미지 간의 의미 불일치를 해결하여 더 자연스러운 이미지 생성을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 첫 번째로 적절한 의미적 조건을 찾고, 그에 따라 관련된 단어 토큰을 식별하여 불필요한 단어 효과를 제거합니다. 이후 기하학적 일관성을 확보하기 위해 주의(attention) 지도에서 관심 영역(Region-of-Interests)을 찾아내어 라텐트(latent) 값을 변환합니다. 또한, Diffusion 기반 라텐트 리필(latents-refill) 방법을 통해 원래 관심 영역에서 라텐트의 영향을 최소화하여 생성된 이미지의 아티팩트를 줄입니다.

- **Performance Highlights**: Coco-stuff 데이터셋에서 실험한 결과, 제안된 방법은 기존의 SOTA(trained) 방법보다 9% 더 나은 성능을 보여주며, 레이아웃 일관성(layout consistency) 평가 지표에서 30% 향상된 결과를 나타냈습니다. 이를 통해 제안된 방법의 보편성과 효과성을 입증하였으며, 코드 또한 공개할 예정입니다.



### ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding (https://arxiv.org/abs/2501.01366)
Comments:
          20 pages with 5 figures and 11 tables

- **What's New**: 이 논문에서는 3D 시나리오에서 자연어로 언급된 대상을 정확히 위치시키는 3D visual grounding (3DVG) 모델의 중요성을 강조합니다. 특히, 기존 데이터셋의 한계를 극복하기 위해 다양한 언어 패턴을 포괄할 수 있는 새로운 진단 데이터셋인 Visual Grounding with Diverse Language in 3D (ViGiL3D)를 소개합니다. 이 데이터셋은 3DVG 방법을 평가하는 데 있어 유용하고 대표적인 프롬프트 세트를 마련하는 데 기여할 것입니다.

- **Technical Details**: ViGiL3D는 다양한 자연어 프롬프트를 통해 3DVG 메서드의 능력을 테스트할 수 있는 프레임워크를 제공합니다. 연구는 기존의 오픈 바카블러리(open-vocabulary) 3DVG 방법들을 평가하여, 이러한 방법들이 더 도전적인 프롬프트를 이해하고 대상을 식별하는 데 부족함이 있음을 보여줍니다. 이를 통해, 보다 실용적인 응용 프로그램을 위해 필요한 언어적 다양성을 강조합니다.

- **Performance Highlights**: 테스트 결과, 현재의 3DVG 방법들은 다양한 언어 패턴에 대한 이해도가 낮은 것으로 나타났습니다. 더욱이 이 연구는 언어적인 성능 향상을 위해 필요로 하는 영역을 밝혀냄으로써, 3DVG 모델의 발전 방향을 제시합니다. 연구진은 이러한 데이터를 바탕으로 미래에 더 효과적이고 효율적인 3DVG 솔루션을 개발할 수 있는 가능성을 열어 놓습니다.



### Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability (https://arxiv.org/abs/2501.01346)
Comments:
          16 pages, 3 figures

- **What's New**: 최근 연구는 대형 비전-언어 모델(LVLMs)의 시각 및 언어 표현 간의 정합(alignment) 문제를 다룹니다. 기존LVLM들이 이미지 캡셔닝(image captioning) 및 시각적 질문 응답(visual question answering)과 같은 기능에서 큰 진전을 이룬 가운데, 정합 메커니즘에 대한 이해가 부족하다는 점을 지적합니다. 이 설문조사는 LVLMs의 정합과 비정합(misalignment) 현상을 보다 잘 이해할 수 있는 구조적 Framework를 제공합니다.

- **Technical Details**: LVLMs에서의 정합은 다양한 층면에서 이루어집니다. 첫째, 나타나는 주요 개념으로 표현적 정합(representational alignment)과 행동적 정합(behavioral alignment)을 정의합니다. 각각은 모델의 내부 임베딩 공간에서 시각적 및 언어적 표현의 일치를 의미하며, 이러한 정합을 통해 모델이 시각 정보와 언어 정보를 의미 있게 연결할 수 있습니다. 정합 과정은 비주얼 인코더 학습, 어댑터 미세 조정(adapter fine-tuning), 그리고 최종적인 전체 시스템의 미세 조정(end-to-end fine-tuning)의 세 단계로 나뉩니다.

- **Performance Highlights**: 정합이 효과적으로 이루어질 때, LVLM은 이미지 입력에 대한 정확하고 일관된 텍스트 응답을 생성할 수 있습니다. 그러나 비정합 현상이 발생하면 모델은 이미지에 대한 잘못된 식별(object misalignment), 잘못된 설명(attribute misalignment), 관계 비정합(relational misalignment) 등의 문제를 겪게 됩니다. 이러한 문제들은 모델의 신뢰성과 성능에 지대한 영향을 미치므로, 개선 전략과 표준화된 평가 프로토콜의 필요성이 강조됩니다.



### SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration (https://arxiv.org/abs/2501.01320)
Comments:
          Draft ver., may be updated in the future

- **What's New**: 이 논문은 SeedVR이라는 새로운 Diffusion Transformer(DiT) 모델을 소개합니다. SeedVR은 실제 세계에서의 비디오 복원 문제를 해결하기 위해 설계되었으며, 임의의 길이와 해상도를 처리할 수 있습니다. 주요 설계 요소인 Shifted Window Attention을 활용하여 긴 비디오 시퀀스에 효과적으로 복원 작업을 수행합니다.

- **Technical Details**: SeedVR은 기존의 문제를 해결하기 위해 대규모로 겹치지 않는 윈도우 주의(attention) 메커니즘을 사용합니다. 이 모델은 Swin Transformer(Swin-MMDiT) 구조를 채택하여, 64×64 크기의 주의 윈도우를 활용하고 8×8 압축된 잠재 공간에서 작동합니다. 또한, 시간과 공간을 각각 4배 및 8배 압축하는 인과론적 비디오 변량 오토인코더(CVVAE)를 개발하여 높은 복원 품질을 유지하면서 계산 비용을 줄입니다.

- **Performance Highlights**: Extensive 실험 결과에 따르면, SeedVR은 기존의 비디오 복원 방법들 대비 최소 2배 빠른 성능을 보입니다. 2.48B 파라미터를 가지고 있음에도 불구하고, 다양한 실세계 벤치마크에서 상태-of-the-art 성능을 달성하였습니다. SeedVR은 대규모 조인트 훈련과 다중 스케일 점진적 훈련을 활용하여 기존 접근 방식보다 우수한 성과를 보이며, 앞으로 더 발전된 비디오 복원 기술의 선두주자가 될 것입니다.



### Multi-Head Explainer: A General Framework to Improve Explainability in CNNs and Transformers (https://arxiv.org/abs/2501.01311)
- **What's New**: 이번 연구에서는 Multi-Head Explainer (MHEX)를 소개하여, CNN과 Transformer 모델의 설명 가능성과 정확성을 동시에 향상시키는 모듈형 프레임워크를 제안합니다. MHEX는 동적으로 작업 관련 기능을 강조하는 Attention Gate, 초기 레이어가 타겟 클래스와 관련된 세부 정보를 캡처하도록 안내하는 Deep Supervision, 정제된 지역과 전역 표현을 통합하여 포괄적인 saliency maps를 생성하는 Equivalent Matrix의 세 가지 핵심 구성 요소로 이루어져 있습니다. MHEX는 기존의 ResNet 및 BERT 모델에 최소한의 수정으로 쉽게 통합될 수 있으며, 이를 통해 분류 정확도와 해석 가능성이 향상됩니다.

- **Technical Details**: MHEX는 CNN 및 Transformer 기반 모델의 설명 가능성과 정확성을 강화하기 위해 설계된 모듈형 프레임워크로, 세 가지 주요 구성 요소로 구성되어 있습니다. Attention Gate는 입력 기능의 가중치를 지역 및 글로벌 정보에 기반하여 동적으로 조정하며, Deep Supervision은 초기 레이어에서의 특징 학습을 최적화하여 세밀한 세부 정보를 잡아냅니다. 마지막으로 Equivalent Matrix는 정제된 지역 및 전역 표현을 통합하여 의역 가능하고 상세한 saliency 점수를 생성합니다.

- **Performance Highlights**: 의료 이미징 및 텍스트 분류 분야의 벤치마크 데이터셋에서 광범위한 실험 결과, MHEX는 분류 정확도를 개선할 뿐만 아니라 매우 해석 가능한 saliency 점수도 생성하는 것으로 나타났습니다. 특히 의료 전문가는 정확하고 해석 가능한 모델 예측에 의존해야 하므로, 이러한 향상된 모델이 임상적 사용에 있어 더욱 유용할 것으로 기대됩니다. MHEX는 모든 모델 구조에 대한 뛰어난 호환성을 가지고 있어, 더 많은 연구와 응용이 가능할 것입니다.



### HybridTrack: A Hybrid Approach for Robust Multi-Object Tracking (https://arxiv.org/abs/2501.01275)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: HybridTrack는 데이터 기반 Kalman 필터(Kalman Filter)를 통합한 새로운 3D 다중 객체 추적 프레임워크로, 기존의 수동 모션 및 확률 파라미터 모델링 필요성을 제거합니다. 이 방법은 KITTI 데이터셋에서 82.08%의 HOTA 정확도를 기록하며, 기존 최첨단 방법을 크게 능가합니다. 또다른 특징은 112 FPS의 빠른 처리 속도로 실제 교통 응용에서의 가능성을 열어줍니다.

- **Technical Details**: HybridTrack는 데이터에서부터 Kalman 이탈 및 Kalman 이득을 직접 학습하여 수동 조정 없이 다양한 시나리오에 적응할 수 있습니다. 기존 모션 모델의 한계를 넘어, 특정 객체에 대한 설계나 사전 지식 없이도 동적으로 스토캐스틱 파라미터를 조정할 수 있습니다. 이로써 안정성과 효율을 동시에 확보하며, 경량화된 구조를 유지합니다.

- **Performance Highlights**: HybridTrack은 KITTI 데이터셋에서의 실험 결과, 82.08%의 높은 HOTA와 112 FPS의 처리 속도를 달성하여, 실시간 교통 추적 응용에 적합한 효율성을 제공합니다. 이러한 성과는 기존의 모델 기반 추적 기법에 대비하여 성능 향상을 가져오며, 효율성을 높이는 혁신적인 접근법으로 평가됩니다.



### Detail Matters: Mamba-Inspired Joint Unfolding Network for Snapshot Spectral Compressive Imaging (https://arxiv.org/abs/2501.01262)
Comments:
          9 pages, 7 figures, AAAI 2025

- **What's New**: 본 논문에서는 Mamba에서 영감을 얻은 Joint Unfolding Network (MiJUN)를 제안하여 단일 2D 측정값에서 3D 하이퍼스펙트럼 이미지(HSI)의 복원을 목표로 합니다. MiJUN은 물리 기반의 딥 언폴딩 네트워크(DUN)와 학습 기반 HSI 이미징을 통합하며, 고급 네트워크 모듈을 활용하여 이미지 복원 문제를 해결하고자 합니다. 또한, 정보의 지역 및 전역 조화를 위한 Attention 메커니즘의 재구성을 통해 Mamba 아키텍처를 변형하여 효율성을 향상시킵니다.

- **Technical Details**: MiJUN은 다중 방향으로 텐서 모드-k 언폴딩을 통합하여 긴 범위 상호작용 문제를 해결하고, 고차원 입력 형태와 벡터 형태를 연결합니다. 또한, 대칭 분해된 가속화된 반 이차 분할(HQS) 접근 방식을 적용하여 초기 최적화 단계에 대한 의존도를 감소시키고, 12개의 스캔 방향을 통해 HSI 재구성의 정확도를 높입니다. MiJUN은 기본적인 선형 회귀 수정 사항을 고려하면서 주의 깊은 관찰을 통해 HSI 정확성과 안정성을 높이기 위한 방식으로 개발되었습니다.

- **Performance Highlights**: 실험 결과, MiJUN은 기존의 SOTA(Sate Of The Art) 모델에 비해 PSNR 값에서 1.01 dB 향상을 기록하며, 파라미터 수와 계산 비용을 각각 3배 줄였습니다. 시뮬레이션 및 실제 데이터셋에서 정량적인 성능이 우수하며, 이미지의 세밀한 세부 사항을 복원하는 데 뛰어난 결과를 보입니다. MiJUN의 통합 접근 방식은 고속 처리와 고품질 복원의 가능성을 제시합니다.



### SeFAR: Semi-supervised Fine-grained Action Recognition with Temporal Perturbation and Learning Stabilization (https://arxiv.org/abs/2501.01245)
Comments:
          AAAI 2025; Code: this https URL

- **What's New**: 이 논문에서는 Fine-grained Action Recognition (FAR)에 초점을 맞추고, 해당 문제를 해결하기 위한 혁신적인 프레임워크 SeFAR를 제안합니다. SeFAR는 semi-supervised learning (SSL) 기법을 채택하여 적은 수의 레이블이 있는 데이터로도 향상된 성능을 발휘할 수 있도록 설계되었습니다. 또한, 두 가지 레벨의 시간적 요소를 모델링하고, 강약 조절 데이터 쌍을 생성하는 새로운 데이터 증강 전략을 도입했습니다.

- **Technical Details**: SeFAR는 FixMatch (Sohn et al. 2020) SSL 패러다임에 기반하여 개발되었으며, 약한-강한 일관성 정규화와 Teacher-Student 설정을 포함하고 있습니다. 이 프레임워크에서는 시간적 맥락을 포함한 이중 레벨 정보 모델링 전략을 사용하여 세부 동작을 효과적으로 캡처하도록 구성했습니다. 또한, Adaptive Regulation을 통해 Teacher 모델의 예측 불확실성을 관리하여 안정적인 학습 과정을 제공합니다.

- **Performance Highlights**: SeFAR는 FineGym과 FineDiving이라는 두 개의 FAR 데이터셋에서 최첨단 성능을 달성하였으며, UCF101과 HMDB51이라는 두 개의 고전적인 coarse-grained 데이터셋에서도 다른 semi-supervised 방법을 초월하는 성능을 보였습니다. 이 연구 결과는 SeFAR가 멀티모달 기초 모델의 도메인 특화 의미 이해 능력을 향상시킬 수 있음을 보여주며, 더 나아가 FAR 문제 해결에 있어 중요한 기여를 하고 있음을 입증합니다.



### Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants (https://arxiv.org/abs/2501.01243)
Comments:
          50 pages, 14 figures, 41 tables. Submitted to ICLR 2025

- **What's New**: 이번 연구에서는 얼굴과 인간 이해 능력을 평가하기 위한 새로운 기준인 Face-Human-Bench를 제안합니다. 이 기준은 3단계의 능력 분류 체계에 기반하여 개발되었으며, 공통적으로 사용되는 데이터셋에서 수집된 900개의 개발 문제와 1800개의 테스트 문제를 포함합니다. 연구 결과는 멀티모달 대형 언어 모델(MLLMs)의 성능 차이에 대한 흥미로운 통찰을 제공합니다.

- **Technical Details**: 제안된 능력 분류 체계는 세 가지 수준으로 구성되어 있으며, Level-1에는 얼굴 이해(face understanding)과 인간 이해(human understanding)의 두 가지 관점이 포함됩니다. 각 수준에서는 인지 과정에 관한 세부 능력이 정의되어 있으며, Level-2에서는 얼굴 관련 5가지와 인간 관련 5가지 능도로 나뉘어 있습니다. 최종적으로, Face-Human-Bench는 25개 주류 MLLMs의 얼굴 및 인간 이해 능력을 종합적으로 평가하기 위한 방안을 제공합니다.

- **Performance Highlights**: Face-Human-Bench에 대한 평가 결과, 특정 MLLMs는 얻은 점수에서 상당한 차이를 보였으며, 상대적 위치가 성능에 미치는 영향도 심각하게 분석되었습니다. 특히, 심각한 시나리오에서의 깊은 가짜 탐지에서 전문 모델이 MLLMs보다 우수한 성능을 보임을 확인하여, 특정 작업에서 전문 모델의 통합이 필요함을 제안합니다. 연구 결과는 멀티모달 조수의 응답 품질을 높이기 위한 방안을 제시합니다.



### Asymmetric Reinforcing against Multi-modal Representation Bias (https://arxiv.org/abs/2501.01240)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 Asymmetric Reinforcing 방법(ARM)을 제안하여 다중 모달 학습에서 발생하는 모달리티 편향 문제를 해결하고자 합니다. ARM은 동적으로 약한 모달리티를 강화하면서도 지배적인 모달리티의 표현 능력을 유지할 수 있도록 설계되었습니다. 기존의 방법들이 모달리티 간의 상호 관계를 간과한 반면, 이 연구는 조건부 상호 정보(Conditional Mutual Information)를 활용하여 모달리티 기여도를 분석합니다.

- **Technical Details**: ARM은 주어진 샘플의 각 모달리티의 한계 기여도와 모든 모달리티의 공동 기여도를 평가하기 위해 상호 정보(mutual information) 기반의 평가 메트릭을 도입합니다. 이 방법은 모달리티의 기여도를 동적으로 균형 잡으면서도 모달리티 잊힘을 방지합니다. 또한 ARM의 구조는 모달리티 간의 기여 격차를 축소하여 전체 성능을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: 이 연구에서 제시한 방법은 다양한 다중 모달 분류 데이터셋에서 최첨단(State-Of-The-Art) 알고리즘과 비교하여 탁월한 성능을 나타냈습니다. 특히, ARM은 모달리티 잊림 문제를 극복하고 각 모달리티의 성능을 강화하면서 모달리티 간의 협력을 향상시킵니다. 이로 인해 다중 모달 학습의 목표인 상호 보완적인 성격을 최대한 활용할 수 있게 되었습니다.



### EHCTNet: Enhanced Hybrid of CNN and Transformer Network for Remote Sensing Image Change Detection (https://arxiv.org/abs/2501.01238)
- **What's New**: 이 연구에서는 고해상도 원격 감지(RS) 영상의 변화 감지를 개선하기 위해 Enhanced Hybrid CNN-Transformer Network (EHCTNet)를 제안합니다. EHCTNet은 성능 개선을 위해 CNN과 Transformer의 장점을 통합하여 지역적(local) 및 글로벌(global) 피쳐 추출을 강화합니다. 이 방법은 고차원적 변화 정보를 효과적으로 추출하고, 전체적인 검출 성능을 향상시킵니다.

- **Technical Details**: EHCTNet은 다섯 가지 모듈로 구성되어 있으며, 1) 피쳐 추출 모듈, 2) 정제 모듈 I, 3) 향상된 토큰 채굴 기반의 Transformer 모듈, 4) 정제 모듈 II, 5) 검출 헤드 모듈이 포함됩니다. 첫 번째 모듈인 피쳐 추출 모듈은 CNN과 Transformer의 이중 브랜치 하이브리드 구조로, 다중 스케일 특성을 캡처합니다. 정제 모듈 I과 II는 주요 특성의 주파수 구성요소를 정제하여 최종 탐지 성능을 개선합니다.

- **Performance Highlights**: EHCTNet은 복잡한 변화 감지에서 우수한 성과를 보이며, 기존의 모델들보다 더 많은 연속적인 변화 영역을 감지하는 능력을 가지고 있습니다. 시각화 결과는 EHCTNet이 인접한 구역 간의 구별 능력을 향상시키고, 변화 감지 작업에서의 효과성을 드러냅니다. 이 모델은 중요한 변화가 발생한 지역을 더 잘 파악하여 재난 대응 및 불법 건축 모니터링과 같은 응용 분야에서의 활용 가능성을 높입니다.



### SVFR: A Unified Framework for Generalized Video Face Restoration (https://arxiv.org/abs/2501.01235)
- **What's New**: 이 논문은 Generalized Video Face Restoration (GVFR) 작업을 위한 새로운 접근 방식을 제안합니다. 특히, 비디오 BFR, 인페인팅(inpainting), 색상화(colorization) 작업을 통합하여 서로의 이점이 있을 것이라고 명시하고 있습니다. 이를 통해 영상 복원에서의 시간적 일관성을 개선하고, 새로운 패러다임을 제시하고 있습니다.

- **Technical Details**: 우리는 Stable Video Face Restoration (SVFR)이라는 통합 프레임워크를 소개합니다. 이 프레임워크는 Stable Video Diffusion (SVD)의 생성적 모션 프라이어를 활용하고, 학습 가능한 작업 임베딩(task embedding)과 새로운 통합 잠재 규제(Unified Latent Regularization)를 도입하여 다양한 작업 간의 공유 특성 표현 학습을 촉진합니다. 또한, 얼굴 구조를 매개로 한 프라이어 학습과 자기 참조 정제 전략을 통해 복원 품질을 향상시킵니다.

- **Performance Highlights**: 실험을 통해 SVFR 프레임워크가 세 가지 하위 작업에서 모두 성능을 향상시킨다는 것을 입증했습니다. 특히, 색상화 모델의 사전 학습이 저화질 비디오에서 자연스러운 피부 색조를 복원하는 데 도움이 되었고, 인페인팅 모델이 가려진 얼굴 영역을 더 높은 충실도로 복원하는 데 기여하였습니다. 이러한 결과는 비디오 BFR 작업에서 강력하고 효율적인 복원 기술의 가능성을 보여줍니다.



### Exploiting Latent Properties to Optimize Neural Codecs (https://arxiv.org/abs/2501.01231)
Comments:
          Accepted in IEEE TRANSACTIONS ON IMAGE PROCESSING

- **What's New**: 이 논문은 벡터 양자화(vector quantization)와 엔트로피 기울기(entropy gradient)라는 두 가지 속성을 활용하여 비디오 및 이미지 압축 코덱의 성능을 향상시킬 수 있는 방법을 제안하고 있습니다. 기존의 신경망 기반 코덱들이 이 두 가지 기능을 제대로 활용하지 못하고 있다는 점을 강조하며, 이를 통해 1-3%의 비율을 절감할 수 있음을 보여주었습니다.

- **Technical Details**: 논문은 일정량의 데이터를 압축하기 위해, 비선형 인코더(g_a)와 후속 양자화 단계(Q)를 통해 입력 이미지를 처리하는 과정에 대해 설명합니다. 일반적인 스칼라 양자화(scalar quantization) 대신 최적화된 균일 벡터 양자화(uniform vector quantization)를 제안하며, 이를 통해 엔트로피 기울기를 통해 정보 재구성을 개선하는 방법을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 기술이 기존 몇 가지 신경망 코덱과의 비교에서 각기 다른 기법들에 대해 1-3%의 압축 효율성을 높인 것으로 나타났습니다. 또한, 엔트로피 기울기를 기반으로 한 해결책이 전통적인 비디오 코덱의 성능을 0.1% 향상시킬 수 있다는 점도 강조되고 있습니다.



### Conditional Consistency Guided Image Translation and Enhancemen (https://arxiv.org/abs/2501.01223)
Comments:
          6 pages, 5 figures, 4 tables, ICME conference 2025

- **What's New**: 이 논문에서는 Conditional Consistency Models (CCMs)을 도입하여 멀티 도메인 이미지 전송을 위한 새로운 접근 방식을 제안합니다. 기존의 일관성 모델에 추가 조건 입력을 통합하여 생성 과정에서 구조적이고 맥락적인 정보를 유지하도록 설계했습니다. CCM은 다양한 도메인에서의 번역 작업에서 뛰어난 성능을 보여주며, 코드도 제공됩니다.

- **Technical Details**: 조건부 일관성 모델(Conditional Consistency Models, CCMs)은 기존 일관성 모델에 조건적 입력을 통합하여 구현됩니다. 특히, CCM은 보이는 이미지를 활용한 가시광선에서 적외선으로의 변환, HE 이미지를 이용한 HE에서 IHC로의 변환, 저조도 이미지에서 LLIE를 위한 변환 등의 작업을 수행합니다. 네트워크 아키텍처를 면밀히 설계하여 조건 입력을 처리하면서 중요한 지역 및 전역 공간 세부사항을 보존합니다.

- **Performance Highlights**: CCM은 10개의 다양한 데이터셋에서 평가되어 멀티 도메인 간 고품질 번역 이미지를 생성하는 데 효과적임을 입증하였습니다. 특히, 저조도 이미지 개선(LLIE) 및 교차 모드 전송과 같은 복잡한 작업에 대해 인상적인 성능을 보였습니다. 이 모델은 번역 및 이미지 향상 모두에 적용 가능하기 때문에 높은 효율성을 가지고 있습니다.



### Real-time Cross-modal Cybersickness Prediction in Virtual Reality (https://arxiv.org/abs/2501.01212)
- **What's New**: 사이버 멀미(cybersickness)는 몰입형 가상 현실(VR) 경험의 널리 퍼지는 채택에 큰 장애물로 남아 있으며, 이로 인해 사용자 참여도와 편안함이 크게 방해받을 수 있습니다. 본 연구에서는 다중 데이터 모달리티 간의 복잡한 상호작용을 포착하고, 실시간 추론을 지원하는 경량 모델을 제안합니다. 제안된 모델은 생체 신호(bio-signal) 특성과 비디오 기능 추출을 통해 사이버 멀미 예측을 지원하며, 이를 통해 VR 환경에서의 모달리티 상호작용 문제를 해결합니다.

- **Technical Details**: 본 논문에서 제안하는 모델은 희소(self-attention) 기술을 사용하는 transformer 기반 인코더와 비디오 기능 추출을 위한 PP-TSN 네트워크를 활용합니다. 이러한 접근 방식은 생체 신호 특성과 비주얼 정보를 통합하여 영상 인식이 가능한 생체 신호 표현을 생성합니다. 또한, 이 모델은 개인의 특성에 맞춘 맞춤형 사이버 멀미 예측을 위해 VR 비디오 콘텐츠를 사용하여 훈련됩니다.

- **Performance Highlights**: 제안된 모델은 공공 데이터셋을 기반으로 훈련되어 VR 비디오 입력만으로 93.13%의 높은 정확도로 사이버 멀미 예측 성능을 입증했습니다. 이 성과는 VR 환경에서 효과적인 실시간 사이버 멀미 예측을 가능하게 하며, 사용자 행동 및 환경 변화에 잘 적응할 수 있도록 합니다. 이로 인해 개인 맞춤형이고 편안한 VR 경험을 위한 연구의 기초가 마련되었습니다.



### LayeringDiff: Layered Image Synthesis via Generation, then Disassembly with Generative Knowledg (https://arxiv.org/abs/2501.01197)
- **What's New**: 이번 연구에서는 LayeringDiff라는 혁신적인 레이어드 이미지 합성 파이프라인을 제안합니다. 본 방식은 기존 이미지 생성 모델을 활용하여 합성 이미지를 생성한 뒤, 이를 전경(foreground)과 배경(background) 레이어로 분해하는 두 단계로 구성되어 있습니다. 이로 인해 대규모 학습 데이터셋 없이도 레이어를 효과적으로 합성할 수 있는 장점을 갖고 있습니다.

- **Technical Details**: LayeringDiff는 세 단계로 운영됩니다: 첫째, 이미지 생성 단계에서 사전 학습된 이미지 생성 모델을 사용하여 초기 합성 이미지를 생성합니다. 둘째, 전경 결정 단계에서 입력 텍스트 프롬프트를 기준으로 전경 영역을 결정합니다. 마지막으로, 레이어링 단계에서 이미지를 전경과 배경 레이어로 분리하며, 이 과정에서 고주파 정렬(high-frequency alignment) 모듈을 도입하여 세부사항을 개선합니다.

- **Performance Highlights**: 광범위한 실험 결과, LayeringDiff는 기존 방법들과 비교하여 더 다양하고 자연스러운 전경과 배경 레이어를 합성하는 데 뛰어난 성능을 보입니다. 이에 따라 다양한 실용적 응용 가능성을 보여주며, 고품질 레이어드 이미지 합성을 위한 유용한 솔루션으로 자리잡을 것으로 기대됩니다.



### Sparis: Neural Implicit Surface Reconstruction of Indoor Scenes from Sparse Views (https://arxiv.org/abs/2501.01196)
Comments:
          Accepted by AAAI 2025. Project page: this https URL

- **What's New**: 본 논문에서는 Sparis라는 새로운 방법을 제안하여 제한된 시점에서의 실내 표면 재구성을 가능하게 합니다. 기존의 방법들이 수백 장의 이미지를 요구하는 반면, Sparis는 적은 수의 이미지에서도 효과적인 결과를 도출해냅니다. 새로운 prior는 이미지 간 매칭 정보를 기반으로 하여 더 정확한 깊이 정보와 일관성을 제공하여 재구성 품질을 향상시킵니다.

- **Technical Details**: 이 방법은 SDF(Signed Distance Functions)를 기하학적 표현으로 채택하고, 이미지 간 매칭 정보를 활용하여 신뢰할 수 있는 절대 깊이 prior를 확보합니다. 매칭 정확도를 높이기 위해 각도 필터와 에피폴라 가중치 함수 같은 매칭 최적화 전략을 도입하였습니다. 이러한 접근은 재구성할 표면의 품질 향상에 기여하며, 제안된 방법의 모든 과정은 매칭 네트워크에 의해 결정됩니다.

- **Performance Highlights**: Sparis는 다양한 실제 및 합성 데이터 세트에서 평가를 통해 기존의 리딩 인도어 재구성 방법들보다 우수한 성능을 보입니다. 제한된 뷰에서의 실내 장면 재구성을 위해 기존 방법들이 직면했던 여러 문제를 해결하며, 더 완벽하고 상세한 표면 재구성을 달성합니다. 실험 결과는 Sparis가 특히 복잡한 실내 씬에서 뛰어난 효과를 입증하고 있습니다.



### Vulnerability-Aware Spatio-Temporal Learning for Generalizable and Interpretable Deepfake Video Detection (https://arxiv.org/abs/2501.01184)
- **What's New**: 이번 논문에서는 FakeSTormer라는 새로운 방법론을 제안합니다. FakeSTormer는 영상 기반 딥페이크 탐지를 위해 다중 작업 학습 프레임워크를 도입하여, 미세한 시공간 아티팩트(spatio-temporal artifacts)에 집중할 수 있게 합니다. 또한, 이 모델은 비디오 수준 데이터 합성 알고리즘을 통해 고품질의 가짜 샘플을 생성하여 데이터 다양성을 확보합니다.

- **Technical Details**: FakeSTormer는 두 개의 보조 분기인 회귀 temporal 분기(regression temporal branch)와 공간(spatial) 분기를 포함하여 모델의 예측을 각기 다른 측면에서 강화합니다. 회귀 temporanl 분기는 시간적으로 취약한 위치를 예측하고, 공간 분기는 프레임별 공간 취약성을 탐지하여 시공간 전반에 걸친 아티팩트를 효과적으로 포착합니다. 이 모델은 TimeSformer 아키텍처를 기반으로 해 시공간의 주의(attention)을 분해하여 각 프레임 및 패치에 대한 분류 토큰을 활용합니다.

- **Performance Highlights**: FakeSTormer는 여러 어려운 데이터셋에서 수행한 실험을 통해 최근 최첨단 방법들과 비교했을 때 경쟁력 있는 일반화 성능을 보이는 것으로 나타났습니다. 이 접근 방식은 보조 분기를 통해 모델의 출력 해석 가능성을 크게 향상시킵니다. 종합적으로, FakeSTormer는 딥페이크 탐지 분야에서 선도적인 성능을 달성하며, 일반화 및 해석 가능성을 동시에 개선합니다.



### L3D-Pose: Lifting Pose for 3D Avatars from a Single Camera in the Wild (https://arxiv.org/abs/2501.01174)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 논문에서는 동물과 영장류의 3D 포즈 추정이 환경에 따라 동적이고 예측 불가능한 행동을 보이는 동물로 인해 어렵다는 한계를 극복하기 위해 리깅된 아바타와 합성 데이터셋을 활용하는 하이브리드 접근 방식이 제안되었습니다. 이 방법은 주어진 이미지와 무관하게 2D 포즈를 3D로 변환하는 간단한 Attention MLP 네트워크를 도입하여 사용의 확장성을 보장합니다. 또한, 기존의 해부학적 키포인트 탐지기는 임의의 아바타에 포즈를 정확히 재배치하는데 부족하다는 점을 강조하며, 이를 극복하기 위한 루크업 테이블을 제시합니다.

- **Technical Details**: 연구는 2D 데이터 세트의 수집을 통해 정확한 2D 포즈 예측 모델을 훈련시키고, 그 후 합성된 3D 포즈 데이터에서 유래한 priors를 사용하여 2D 포즈를 3D 공간으로 "리스팅(lifting)"하는 프로세스를 설명합니다. 해부학적 제약을 활용하여 2D 키포인트에서 실제적인 3D 재구성을 달성하고, 이를 통해 3D 주석을 수집하지 않고도 정확한 3D 포즈 예측이 가능합니다. 이러한 접근 방식은 특히 자연 환경에서의 포즈 추정을 위한 원천 데이터 확보의 어려움을 극복하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 루크업 테이블 기반의 포즈 리타게팅 방법이 효과적이고 효율적임을 입증했습니다. 연구는 두 가지 데이터셋, 즉 Deep Macaque과 Deep Horse를 개발하였으며, 이는 다양한 행동 세트를 포함하여 물리 기반 게임 엔진을 활용하여 생성되었습니다. 이를 통해 다양한 자연 환경에서의 포즈를 2D에서 3D로 변환하고, 이를 아바타에 매끄럽게 이식하는 그래픽 응용 프로그램의 가능성을 열었습니다.



### Deep Learning in Palmprint Recognition-A Comprehensive Survey (https://arxiv.org/abs/2501.01166)
Comments:
          Palmprint recognition, biometrics, deep learning, feature extraction, recognition tasks

- **What's New**: 이번 논문은 딥러닝(DL)을 기반으로 한 손바닥 인식의 최근 발전을 포괄적으로 리뷰합니다. 기존의 연구들은 특정 작업에 초점을 맞추었으나, 본 논문은 DL 기술의 혁신적인 역할과 다양한 과제들을 통합적으로 탐구하는 데 중점을 두고 있습니다. 이 논문은 손바닥 인식의 최신 동향과 기술을 정리하여 연구자들이 최신 기술을 이해하고 혁신을 주도할 수 있도록 돕습니다.

- **Technical Details**: 손바닥 인식 기술은 이미지 획득, 전처리, 특징 추출, 매칭의 네 가지 주요 단계로 구성됩니다. 딥러닝 모델 특히 CNN(Convolutional Neural Networks), VGG-16, ResNet 등 다양한 네트워크가 손바닥 인식의 독특한 도전 과제를 해결하기 위해 적용되어 왔습니다. 이러한 기술들이 손바닥의 섬세한 텍스쳐 특징을 효과적으로 포착하고 인식의 정확성을 높이는데 기여하고 있습니다.

- **Performance Highlights**: 논문에서는 퍼포먼스 메트릭과 실험 결과를 종합적으로 평가하며, 특히 DL 기반 방법이 손바닥 인식 성능을 어떻게 향상시켰는지 설명합니다. 향후 연구 방향으로는 보안/privacy 유지 인식, 오픈 세트 인식, 다중 도메인 기술 등이 제안되고 있으며, 이는 손바닥 인식 기술의 활용 범위를 더욱 넓힐 것으로 기대됩니다.



### Towards Interactive Deepfake Analysis (https://arxiv.org/abs/2501.01164)
- **What's New**: 이번 논문에서는 Multi-modal Large Language Models (MLLMs)에 대한 instruction tuning을 통해 인터랙티브 딥페이크 분석(Deepfake Analysis, DFA) 시스템을 개발했습니다. 특히, 딥페이크 탐지 및 분류, 아티팩트 설명, 자유 대화 기능을 가진 새로운 시스템 DFA-GPT를 제안하며, 이 시스템은 정보 포렌식과 보안 분야에서의 연구 방향을 새롭게 탐색합니다. 논문에서 소개된 두 가지 데이터셋 DFA-Instruct와 벤치마크 DFA-Bench는 딥페이크 탐지 및 분류 성능을 평가하는 데 기여할 것입니다.

- **Technical Details**: 이 연구에서는 세 가지 주요 도전 과제에 대응하기 위해 GPT-보조 데이터 구축 과정을 제안합니다. 이 과정에서 127.3K 개의 정렬된 얼굴 이미지와 891.6K 개의 질문-답변 쌍으로 구성된 데이터셋 DFA-Instruct를 생성하였습니다. 또한 Low-Rank Adaptation (LoRA) 모듈을 MLLMs에 통합하여 계산 비용을 절감하며, 이 시스템은 제한된 자원에서도 효과적으로 동작합니다.

- **Performance Highlights**: DFA-GPT는 기존의 DFA 방법들보다 효율적이며, 다양한 딥페이크 기술을 이해하고 탐지하는 휴먼 전문가의 작업 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다. 새로운 벤치마크 DFA-Bench는 MLLMs의 딥페이크 분석 능력을 종합적으로 평가하며, 아티팩트 설명과 자유 대화 능력까지 측정 가능하도록 설계되었습니다. 이를 통해 향후 연구자들이 인터랙티브 딥페이크 분석 시스템을 개발하고 개선하는 데 도움을 줄 것입니다.



### 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer (https://arxiv.org/abs/2501.01163)
- **What's New**: 본 논문에서는 3D-사물 모델인 3D-LLaVA를 제안합니다. 3D-LLaVA는 사용자 친화적인 디자인으로 통합된 아키텍처를 갖추고 있으며, 점구름(point clouds)만을 입력으로 사용하여 3D 세계를 인지하고 상호작용할 수 있는 기능을 제공합니다. Omni Superpoint Transformer (OST)라는 혁신적인 모듈은 시각적 기능 선택, 시각적 프롬프트 인코딩 그리고 마스크 디코딩의 세 가지 주요 기능을 수행합니다.

- **Technical Details**: 3D-LLaVA는 복잡한 파이프라인 없이 단일 모델로 여러 가지 3D 작업을 수행할 수 있습니다. OST는 전경과 배경 초점을 구별하여 불필요한 계산 부담을 줄이고, 사용자가 제공하는 시각적 프롬프트를 언어 토큰과 동일한 임베딩 공간으로 매핑하여 직접 3D 마스크를 생성하는 등 다양한 기능을 수행합니다. 이에 따라 모델은 간편하면서도 강력한 성능을 유지할 수 있습니다.

- **Performance Highlights**: 3D-LLaVA는 다양한 3D 비전 및 언어 이해 데이터셋에서 뛰어난 성능을 보여줍니다. 특히, ScanQA 데이터셋에서는 이전 최고 기록보다 4.9% 높은 92.6%의 CiDEr 점수를 기록하였습니다. 이러한 결과는 3D-LLaVA가 이 분야에서 강력한 기준이 될 가능성을 입증해 줍니다.



### TexAVi: Generating Stereoscopic VR Video Clips from Text Descriptions (https://arxiv.org/abs/2501.01156)
Comments:
          6 pages, published in 2024 IEEE International Conference on Computer Vision and Machine Intelligence (CVMI)

- **What's New**: 이 연구는 기존의 생성 시스템을 통합하여 텍스트에서 입체 가상 현실 비디오를 생성하는 새로운 접근 방식을 제안합니다. 텍스트-이미지 (text-to-image) 모델에서 시작하여, 안정적인 이미지를 생성하는 Stable Diffusion과 깊이 추정 (depth estimation) 알고리즘을 활용하여 높은 품질의 프레임을 제시합니다. 기존 텍스트 기반 생성 시스템을 활용하여 가상 현실 제작의 수작업을 줄이고, 입체적인 시청 경험을 제공합니다.

- **Technical Details**: 제안된 TexAVi 모델은 텍스트 프롬프트를 기반으로 초기 이미지를 생성하는 AttnGAN, 더 높은 품질의 프레임을 생성하는 Stable Diffusion, 그리고 좌우 시각을 위한 깊이 추정 모듈로 구성되어 있습니다. 각 모듈은 사전 훈련된 모델을 활용하여 효율적으로 작동하며, 입체 가상 환경의 사용자 맞춤화에 유리합니다. TexAVi는 이러한 단계를 통해 수작업 없이 고품질의 VR 비디오 생성이 가능하도록 설계되었습니다.

- **Performance Highlights**: TexAVi에서 생성된 비디오는 시각적으로 매력적이며, 스마트폰 호환 VR 헤드셋을 통해 전시될 수 있는 품질을 자랑합니다. 연구의 성과는 기존 방법론들과의 비교를 통해 시각적 품질 측면에서 우수함을 입증하며, 새로운 자연어 기반 그래픽의 가능성을 보여줍니다. 이 프로세스는 가상 현실 기술과 심층 신경망의 통합으로 나아가는 한 단계로 여겨집니다.



### Adaptive Hardness-driven Augmentation and Alignment Strategies for Multi-Source Domain Adaptations (https://arxiv.org/abs/2501.01142)
Comments:
          15 pages, 12 figures

- **What's New**: Multi-source Domain Adaptation (MDA)에 대한 새로운 접근 방식인 'A3MDA'가 제안되었습니다. 이 방법은 기존의 상호 영역 정렬(inter-domain alignment) 방법에 있어 데이터 증대(data augmentation), 내부 도메인 정렬(intra-domain alignment) 및 클러스터 레벨 제약(cluster-level constraints)이라는 세 가지 중요한 요소를 고려합니다. 'A3MDA'는 적응적 하드니스 측정(adaptive hardness measurement)을 통해 데이터 증대와 도메인 간 개선을 동시에 진행합니다.

- **Technical Details**: 'A3MDA'의 세 가지 적응적 하드니스 측정기법(AHM)은 각각 Basic AHM, Smooth AHM, Comparative AHM입니다. Basic AHM은 각 샘플의 즉각적인 하드니스 값을 측정하고, Smooth AHM은 강력한 데이터 증대의 강도를 조정하여 모델의 일반화 능력을 유지합니다. 마지막으로 Comparative AHM은 클러스터 레벨 제약을 용이하게 하여, 전통적인 MMD(Maximum Mean Discrepancy)에 가중치를 부여하여 더욱 견고하고 정확한 상호 영역 정렬을 가능하게 합니다.

- **Performance Highlights**: 여러 MDA 벤치마크 실험에서 'A3MDA'는 기존의 다른 방법들보다 우수한 성능을 보였습니다. 하드니스 값에 기반한 특정 가중치를 활용하여 모델의 강건성과 정확성을 높이는 데 성공했습니다. 이 연구는 MDA 작업에서 효과적인 데이터 증대와 내부 도메인 정렬의 중요성을 부각시킵니다.



### InDeed: Interpretable image deep decomposition with guaranteed generalizability (https://arxiv.org/abs/2501.01127)
- **What's New**: 본 연구는 해석 가능한 이미지 분해를 위한 새로운 프레임워크를 제안하며, 이 프레임워크는 계층적 베이지안 모델링(hierarchical Bayesian modeling)과 딥러닝(deep learning)을 결합합니다. 제안된 아키텍처는 모듈화(modularized)되어 있으며, 모델 범용성(model-generalizable)을 갖추고 있어 다양한 다운스트림 작업(예: 이미지 노이즈 제거(image denoising), 비지도 이상 탐지(unsupervised anomaly detection))에 적용될 수 있습니다. 또한 손실 함수(loss function)와 일반화 오류 경계(generalization error bound) 간의 이론적인 연결을 확립하여, 분포 밖(out-of-distribution) 시나리오에 대한 새로운 테스트 시간 적응 방법(test-time adaptation approach)을 제안합니다.

- **Technical Details**: 제안하는 프레임워크는 세 가지 단계로 나뉘어 있습니다. 첫 번째 단계에서는 이미지 분해를 위한 계층적 베이지안 모델링을 통해 통계적으로 의미 있는 구성 요소(statistically meaningful components)로 이미지를 분해합니다. 두 번째 단계에서는 후방 확률(posteriors)을 근사하는 변분 추론(variational inference)을 적용하여 최적화 문제(optimization tasks)로 변환합니다. 세 번째 단계에서는 구축된 모듈화된 딥 뉴럴 네트워크를 통해 후방 확률을 추론합니다. 이러한 방식은 해석 가능성과 일반화를 동시에 향상시킬 수 있습니다.

- **Performance Highlights**: 제안된 방법은 이미지 노이즈 제거 및 비지도 이상 탐지 작업에 대해 검증되었으며, 전통적인 알고리즘들에 비해 뛰어난 일반화(generalizability)와 해석 가능성(interpretability)을 보여주었습니다. 결과적으로, 본 연구에서 제안된 해석 가능한 딥 이미지 분해 프레임워크는 다양한 응용 프로그램에서의 효과를 입증하였습니다. 연구 결과는 코드 공개를 통해 추가적인 검증과 활용이 가능할 예정입니다.



### Source-free Semantic Regularization Learning for Semi-supervised Domain Adaptation (https://arxiv.org/abs/2501.01126)
- **What's New**: 새로운 연구인 SERL(semantic regularization learning) 프레임워크는 기존의 반지도 도메인 적응(SSDA) 방법의 한계를 극복하는 데 중점을 둡니다. SERL은 타겟 도메인으로의 적응을 위해 복잡한 의미적 정보를 학습하는 방식으로, 라벨이 거의 없는 타겟 도메인에서 소스 모델을 미세 조정하는 데 도움을 줍니다. 이를 통해 소스 도메인에서의 학습된 지식을 타겟 도메인에 효과적으로 전이할 수 있습니다.

- **Technical Details**: SERL은 세 가지 주요 기술적인 정규화 기법인 SPCR(semantic probability contrastive regularization), HMR(hard-sample mixup regularization), TPR(target prediction regularization)를 포함합니다. SPCR은 확률적 관점에서 의미적 정보를 활용하여 특징 표현을 학습하고, HMR은 쉬운 샘플을 활용하여 복잡한 타겟 지식을 추출합니다. 마지막으로, TPR은 과거의 학습 목표와 현재 예측 간의 상관관계를 최대화하여 잘못된 의미 정보의 영향을 감소시킵니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(DomainNet, Office-Home, Office-31)에서 수행된 실험 결과, SERL는 기존의 SSDA 방법들보다 뛰어난 성능을 입증했습니다. SERL은 타겟 도메인에서의 모델 성능을 높이고, 전이 학습의 정확성을 크게 향상시키는 결과를 보였습니다. 이러한 성과는 SERL이 복잡한 의미적 정보를 제대로 이해하고 활용하기 때문으로 평가됩니다.



### DuMo: Dual Encoder Modulation Network for Precise Concept Erasur (https://arxiv.org/abs/2501.01125)
Comments:
          AAAI 2025 accepted

- **What's New**: 이 논문은 텍스트-이미지 모델에서 발생하는 Not-Safe-For-Work (NSFW) 콘텐츠 생성에 대한 안전 문제를 해결하기 위해 새로운 방법인 Dual encoder Modulation network (DuMo)를 제안합니다. 기존 방법들이 부적절한 개념을 제거하면서 구조적 요소에 큰 영향을 미쳤다면, DuMo는 비대상 개념을 최소한으로 손상시키며 정확한 제거를 수행합니다. 이로 인해 모델의 전반적인 생성 능력이 향상됩니다.

- **Technical Details**: DuMo는 Eraser with PRior Knowledge (EPR) 모듈을 활용하여 U-NET의 스킵 연결 기능을 수정하고, 이미지의 고주파(high-frequency) 구성 요소에서 주로 개념 제거를 수행합니다. 또한, U-NET의 백본(backbone) 파라미터는 고정되어 있으며, 원래 스킵 연결 기능으로부터의 사전 지식(prior knowledge)이 제거 과정에 도입됩니다. 이 과정에서 다른 타임스텝과 레이어에서 구조 및 디테일에 대한 독특한 제거 선호가 관찰됩니다.

- **Performance Highlights**: 제안된 DuMo 방법은 Explicit Content Erasure, Cartoon Concept Removal, Artistic Style Erasure에서 최첨단 성능을 달성하며, 기존 방법들보다 명확하게 우수한 결과를 보입니다. 이러한 성능 향상은 EPR 모듈의 스케일 조정 및 모델의 생성 능력의 균형을 자동으로 조절하는 Time-Layer MOdulation process (TLMO)에 기인합니다. 본 연구에 사용된 코드는 제공된 링크에서 확인할 수 있습니다.



### PatchRefiner V2: Fast and Lightweight Real-Domain High-Resolution Metric Depth Estimation (https://arxiv.org/abs/2501.01121)
- **What's New**: 패치리파이너(PatchRefiner) V2(PRV2)는 기존의 무거운 리파이너 모델을 경량 인코더로 대체하여 심도 추정(depth estimation) 모델의 크기 및 추론 시간(inference time)을 줄입니다. 새로운 Coarse-to-Fine(C2F) 모듈과 Guided Denoising Unit을 도입하여 리파이너 특성을 정제하고 노이즈를 제거하며, Scale-and-Shift Invariant Gradient Matching(SSIGM) 손실 함수를 활용하여 합성-실제 도메인 전이(synthetic-to-real domain transfer)를 개선합니다. PRV2는 UnrealStereo4K에서 첨단 심도 추정 방법들을 초월하는 성능을 보여줍니다.

- **Technical Details**: PRV2는 경량 인코더(MobileNet, EfficientNet)를 사용하여 고해상도 심도 추정의 효율성을 개선하고, C2F 모듈을 통해 고해상도 특성을 정제합니다. GDUs는 coarse 깊이 특성을 가이던스로 활용하여 고해상도 리파이너 특성을 정제합니다. 또한 Noisy Pre-training 전략을 통해 리파이너 브랜치의 초기화를 최적화하며, SSIGM 손실 함수를 사용하여 합성 데이터로부터 고주파 세부 특성을 학습하는 방법을 제공합니다.

- **Performance Highlights**: PRV2는 UnrealStereo4K 데이터셋에서 뛰어난 정량적 결과와 함께 초당 적절한 추론 시간을 보여줍니다. CityScape, ScanNet++, KITTI와 같은 실제 데이터셋에서도 심도 경계 그리기(depth boundary delineation)에서 현저한 개선을 이루어내며, 이는 다양한 도메인에서의 적응성과 효과성을 나타냅니다. 이러한 성능 향상은 PRV2가 경량 아키텍처를 활용하여 다양한 환경에서의 고해상도 심도 추정을 실행함으로써 가능해졌습니다.



### Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning (https://arxiv.org/abs/2501.01120)
Comments:
          9 pages, 8 figures. Accepted by AAAI 2025. Codes are released at this https URL

- **What's New**: 이 논문은 불완전한 모달리티에서의 다중모달 학습을 해결하기 위해 RAGPT라는 새로운 Retrieval-AuGmented 동적 Prompt Tuning 프레임워크를 제안합니다. RAGPT는 유사한 인스턴스를 식별하는 multi-channel retriever, 결측 정보를 생성하는 missing modality generator, 그리고 맥락 지식을 활용하여 동적 프롬프트를 생성하는 context-aware prompter의 세 가지 모듈로 구성됩니다. 이러한 접근법은 기존의 정적 프롬프트 방법에서 발생하는 성능 저하 문제를 해결하고 다중모달 전처리 모델의 강 robustness를 향상시킵니다.

- **Technical Details**: RAGPT는 여러 모달리티 간 유사성을 바탕으로 유사 샘플을 검색하는 보편적인 multi-channel retrieval 전략을 활용합니다. 이 전략은 동일한 모달리티의 결측 정보를 보완하기 위해 인접한 샘플로부터 정보를 추출하는데 중점을 둡니다. 또한, context-aware prompter는 검색된 인스턴스 간의 의미론적 상관 관계를 식별하여 입력에 맞춘 동적 프롬프트를 생성함으로써 다중모달 특성을 조정합니다.

- **Performance Highlights**: 실험 결과, RAGPT는 세 가지 실제 데이터셋에서 경쟁하는 아키텍처들과 비교하여 일관되게 더 나은 성능을 보여주었습니다. RAGPT는 결측 모달리티 문제를 효과적으로 다루고, 고전적인 방법으로 발생할 수 있는 정보 손실과 노이즈를 최소화합니다. 이 연구는 다중모달 모델의 신뢰성, 정확성 및 안전성을 향상시키는 중요한 기여를 합니다.



### Leverage Cross-Attention for End-to-End Open-Vocabulary Panoptic Reconstruction (https://arxiv.org/abs/2501.01119)
Comments:
          18 pages, 10 figures

- **What's New**: 본 논문에서는 PanopticRecon++라는 새로운 방법을 제안합니다. 이 방법은 cross-attention 기법을 통해 3D 인스턴스와 3D 임베딩 필드의 관계를 모델링합니다. 특히, 학습 가능한 3D Gaussian을 인스턴스 쿼리로 도입하여 근접성을 유지하면서 end-to-end 최적화 가능성을 갖춥니다. 이를 통해 다양한 객체 클래스의 인식을 가능하게 하여 로봇의 플러그 앤 플레이 기능을 실현합니다.

- **Technical Details**: PanopticRecon++는 3D 쿼리와 키를 동시에 학습시키며, 2D 인스턴스 ID와의 정렬을 위해 선형 할당 방식을 사용합니다. 또한, 2D 인스턴스 마스크에서 렌더링한 인스턴스 마스크와의 최적의 선형 할당을 통해 양자 간의 일관성을 확보합니다. 이 방법은 파라미터가 없는 팬옵틱 헤드를 구축하여 인스턴스 가능성과 의미적 확률을 융합함으로써 의미적-인스턴스 일관성을 보장합니다.

- **Performance Highlights**: PanopticRecon++는 시뮬레이션과 실제 데이터 세트에서 3D 및 2D 세분화 및 재구성 성능에서 경쟁력 있는 결과를 보여줍니다. 특히, 이 시스템은 다양한 환경에서 로봇 시뮬레이터로서의 사용 사례를 입증하며, 사용자 경험을 향상시키는 데 기여합니다. 기존 방법들과 비교하여 기하학적 메쉬 및 세분화 정확성에서 우수한 성과를 보입니다.



### HarmonyIQA: Pioneering Benchmark and Model for Image Harmonization Quality Assessmen (https://arxiv.org/abs/2501.01116)
- **What's New**: 이번 논문에서는 이미지 조화 (image harmonization) 평가를 위한 최초의 이미지 품질 평가 데이터베이스(HarmonyIQAD)를 소개합니다. HarmonyIQAD는 9개의 서로 다른 이미지 조화 알고리즘(IHAs)에 의해 생성된 1350개의 조화된 이미지를 포함하며, 해당 이미지에 대한 인간의 시각적 선호 점수도 제공합니다. 이를 기반으로, 조화화된 이미지의 인간 시각적 선호를 예측하기 위한 Harmony 이미지 품질 평가(HarmonyIQA) 기법을 제안합니다.

- **Technical Details**: 조화 알고리즘은 크게 생성형 이미지 조화 알고리즘(GIHAs)과 비생성형 이미지 조화 알고리즘(NGIHAs)으로 나눌 수 있습니다. GIHAs는 Generative Adversarial Networks (GANs) 또는 확산 모델(difussion models)을 활용하며, NGIHAs는 실제 이미지 스타일을 수정하여 생성된 데이터셋에 기반하여 학습됩니다. HarmonyIQA는 대형 다중모델(LMM)을 기반으로 하며, 지침 튜닝(instruction tuning)과 저랭크 적응(low-rank adaptation) 기술을 통합하여 개발되었습니다.

- **Performance Highlights**: 실험 결과, HarmonyIQA는 HarmonyIQAD에서 기존의 최첨단 풀레퍼런스(FR) IQA 및 노레퍼런스(NR) IQA 방법보다 우수한 성능을 보여주었습니다. 또한, 다양한 IQA 데이터셋에서도 뛰어난 성능을 발휘함으로써 범용 IQA 방법으로서의 가능성을 입증했습니다. 주요 기여는 1350개의 조화된 이미지를 포함한 HarmonyIQAD를 구축하고, 일반 이미지 품질 평가에도 효과적인 조화 이미지 품질 평가 모형을 제안한 것입니다.



### Generalized Task-Driven Medical Image Quality Enhancement with Gradient Promotion (https://arxiv.org/abs/2501.01114)
Comments:
          This paper has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence

- **What's New**: 본 논문은 의료 이미지를 위한 새로운 접근법인 GradProm(Gra디언트 촉진) 훈련 전략을 제안합니다. 이는 기존의 이미지 품질 향상(IQE) 모델이 시각 인식 요구 사항을 간과하는 문제를 해결하고자 합니다. GradProm은 이미지 향상 모델과 시각 인식 모델 간의 훈련을 효율적으로 조정하여 두 모델이 상호 보완적으로 작용하도록 합니다.

- **Technical Details**: GradProm은 훈련 과정에서 이미지 향상 모델의 매개변수만 업데이트하며, 이 때 두 하위 모델의 그래디언트 간의 코사인 유사성(cosine similarity)을 기반으로 방향이 일치할 때만 진행됩니다. 만일 그래디언트가 일치하지 않는 경우, GradProm은 이미지 향상 모델의 그래디언트만을 사용하여 매개변수를 업데이트합니다. 이 과정을 통해 GradProm은 향상된 모델의 최적화 방향이 보조 시각 인식 모델의 영향을 받지 않도록 보장합니다.

- **Performance Highlights**: 대규모 의료 이미지 데이터셋을 통해 수행된 실험에서 GradProm은 기존의 최신 방법들보다 뛰어난 성능을 입증하였습니다. 본 연구는 denoising과 super-resolution, 진단 및 세그멘테이션과 같은 다양한 의료 이미지 작업에서 효과성을 보여주었습니다. 특히, GradProm은 추가적인 데이터나 네트워크 구조의 변경 없이 두 하위 모델 간의 지속적인 성능 향상을 이끌어낼 수 있음을 강조합니다.



### BatStyler: Advancing Multi-category Style Generation for Source-free Domain Generalization (https://arxiv.org/abs/2501.01109)
Comments:
          Accepted by IEEE TCSVT

- **What's New**: 본 논문에서는 Source-Free Domain Generalization(SFDG) 분야에 대한 새로운 접근 방식을 제안합니다. 'BatStyler'라는 방법은 다중 카테고리 설정에서 스타일 다양성을 향상시키기 위한 두 가지 모듈, 즉 Coarse Semantic Generation과 Uniform Style Generation으로 구성되어 있습니다. 이를 통해 모델은 소스 도메인에 의존하지 않고도 새로운 도메인에서 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: BatStyler의 첫 번째 모듈인 Coarse Semantic Generation은 스타일 학습에서 공간 압축을 방지하기 위해 조잡한 의미 정보를 추출합니다. 두 번째 모듈인 Uniform Style Generation은 스타일 템플릿을 균등 분포로 초기화하여 훈련 효율성을 높이도록 설계되었습니다. 이 방법은 코사인 유사도를 낮춰 스타일 간의 다양성을 증가시키고, 패러렐 훈련을 가능하게 합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, BatStyler는 적은 카테고리 데이터셋에서는 기존의 방법과 유사한 성능을 보여주며, 다중 카테고리 데이터셋에서는 최첨단 방법들을 초월하는 뛰어난 성능을 발휘합니다. 동적으로 다양한 데이터 생성을 통해 실제 세계에서의 적용 가능성을 높이고 있습니다.



### AIM: Additional Image Guided Generation of Transferable Adversarial Attacks (https://arxiv.org/abs/2501.01106)
- **What's New**: 이 연구는 적대적인 공격(Adversarial attacks)의 전이 가능성을 향상시키기 위해 새로운 접근법을 제안합니다. 특히, 목표에 초점을 맞춘 전이 가능 공격을 생성하기 위해 추가적인 이미지를 활용하는 방법이 도입되었습니다. 이를 통해 보다 정교하게 목표 클래스의 의미를 반영한 공격을 생성할 수 있습니다.

- **Technical Details**: 우리는 적대적 생성기(generator)에 Semantic Injection Module (SIM)을 추가하여 목표 전이 공격을 더욱 효과적으로 구현합니다. 이 모듈은 추가적인 안내 이미지를 활용하여 공격의 효과를 개선하고, 새로운 손실 공식을 통해 SIM의 통합을 촉진합니다. 우리는 이 방법이 목표 전이 공격과 비목표 전이 공격 모두에 적합하게 작용하는 통합적인 접근법임을 보여주고자 합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 목표 전이 공격에서 우수한 성능을 보였으며, 비목표 전이 공격에서도 최첨단 방법들과 동등한 성능을 발휘하는 것을 확인했습니다. 이러한 결과는 DNNs에 대한 적대적 공격의 전이 가능성을 높일 수 있는 새로운 가능성을 보여줍니다.



### Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes (https://arxiv.org/abs/2501.01101)
Comments:
          7 pages, 4 figures, submitted to ICRA 2025

- **What's New**: 서지컬 장면의 변형적 재구성을 위한 EH-SurGS 알고리즘이 제안되었습니다. 이 알고리즘은 3D Gaussian의 생애 주기(life cycle) 모형을 도입하여, 일반적인 변형과 비가역적 변형 모두를 효과적으로 모델링합니다. 또한, 정적 지역과 변형 지역을 구별하는 적응형 모션 계층 전략을 통해 렌더링 속도를 개선하였습니다.

- **Technical Details**: EH-SurGS는 3D Gaussian splatting 방식을 기반으로 하며, 여러 프레임의 RGB 이미지와 깊이 맵을 입력으로 사용하여 서지컬 장면을 재구성합니다. 이 알고리즘은 특정 시간에 활성화되는 3D Gaussian을 사용하여 비가역적인 변형을 모델링하며, 이는 서지컬 장면에서의 정확한 변형 표현을 가능하게 합니다. 또한, 정적 및 변형 지역에 대한 적응형 전략을 통해 필요한 3D Gaussian을 줄여 렌더링 속도를 높였습니다.

- **Performance Highlights**: 익스페리먼트를 통해 EH-SurGS는 기존의 최첨단 방법들보다 더 나은 재구성 품질과 렌더링 속도를 달성했습니다. 수치적 및 질적 결과 모두에서 우수한 성능을 보여주며, 제안된 구성 요소의 효과와 필요성을 입증하기 위한 탈락 연구도 수행되었습니다. 궁극적으로, 이 알고리즘은 서지컬 환경에서의 고충실도 재구성을 위한 혁신적인 접근 방식을 제공합니다.



### EliGen: Entity-Level Controlled Image Generation with Regional Attention (https://arxiv.org/abs/2501.01097)
- **What's New**: 최근 텍스트-이미지 생성의 발전을 이끌어온 diffusion 모델들이 각광받고 있지만, 단순한 global text prompts는 특정 entity를 조절하는 데 한계가 있다. 이를 극복하기 위해 EliGen이라는 새로운 프레임워크를 제안하며, entity-level 제어와 arbitrary-shaped spatial masks를 통합하는 regional attention 메커니즘을 도입하였다. 고품질의 entity-level 주석을 포함한 데이터셋을 통해 EliGen을 훈련시켜, 제어의 정밀성과 이미지 품질에서 기존의 방법들을 초월하는 성과를 거두었다.

- **Technical Details**: EliGen은 Entity-Level controlled Image Generation을 위한 머신 러닝 프레임워크로, accuracy가 뛰어난 entity-level 조작을 가능하게 한다. 이 모델은 bounding boxes 대신 지역적인 attention을 통해서 임의 형태의 마스크를 처리할 수 있으며, 추가적인 매개 변수가 필요하지 않아 기존 모델들보다 우위를 점하고 있다. 더불어, inpainting fusion 파이프라인을 제안하여 여러 엔티티의 이미지를 한 번의 전Forward Pass로 수용할 수 있는 확장성을 제공한다.

- **Performance Highlights**: EliGen은 entity 제어 생성에서 우수한 성능을 보여주며, 이미지 인페인팅 작업에서도 잘 작동한다. 커뮤니티 개발 오픈 소스 모델과의 통합성 또한 높아 새로운 제작 가능성이 열렸다. 이 연구는 entity-level 제어 생성을 위한 데이터셋이 공개되어 관련 연구의 발전을 도모할 것으로 기대된다.



### Evidential Calibrated Uncertainty-Guided Interactive Segmentation paradigm for Ultrasound Images (https://arxiv.org/abs/2501.01072)
- **What's New**: 이번 연구에서는 초음파 이미지 분할을 위한 새로운 인터랙티브 분할 패러다임인 EUGIS(Evidential Uncertainty-Guided Interactive Segmentation)를 제안합니다. EUGIS는 근거 기반의 불확실성 추정을 활용하여 초음파 이미지에서 높은 불확실성을 가진 영역을 우선적으로 샘플링하여, 더 적은 프롬프트로도 고품질의 분할 결과를 제공합니다. 이는 숙련된 방사선 전문의의 상호작용 행동을 효율적으로 모방하는 접근방식입니다.

- **Technical Details**: EUGIS는 Dempster-Shafer 이론과 Subjective Logic에 기반하여 불확실성을 평가합니다. 모델은 높은 불확실성을 가진 지역에 대한 포인트 프롬프트를 제공받아, 이로 인해 더 높은 성능을 발휘할 수 있습니다. 또한, 하이브리드 이미지 인코더를 사용해 지역적 및 전역적 정보를 포괄적으로 캡처하고, 여러 세분화 결과 및 신뢰도 점수를 생성하여 가장 높은 신뢰도 점수를 가진 세분화 결과를 선택합니다.

- **Performance Highlights**: EUGIS는 유방, 갑상선, 좌측심실이라는 세 가지 데이터셋을 사용하여 평가되었으며, 단일 포인트 프롬프트만으로도 기존의 비인터랙티브 및 인터랙티브 세분화 방법들을 초월하는 성능을 보였습니다. 이러한 결과는 EUGIS가 새로운 엔드투엔드 인터랙티브 세분화 패러다임으로 혁신적인 가능성을 지니고 있음을 나타냅니다.



### TS-SatMVSNet: Slope Aware Height Estimation for Large-Scale Earth Terrain Multi-view Stereo (https://arxiv.org/abs/2501.01049)
- **What's New**: 본 논문에서는 대규모 원거리 탐사를 위한 새로운 경사 인식 기반 높이 추정 네트워크 TS-SatMVSNet을 제안하였습니다. 이 네트워크는 고해상도 원거리 이미지에서 지형의 경사를 고려하여 더 정확한 높이 추정을 가능하게 합니다. 경사 정보의 통합은 MVS(멀티 뷰 스테레오) 프레임워크 내에서 지형 복원의 정확도를 크게 향상시킵니다.

- **Technical Details**: TS-SatMVSNet은 수학적 기울기 개념을 바탕으로 하여 높이 맵으로부터 경사 맵을 계산하고, 경사 지도와 함께 작동하는 두 개의 모듈을 설계했습니다. 마이크로 레벨에서 경사를 활용하여 세분화된 높이 추정을 수행하고, 매크로 레벨에서는 가우시안 스무딩 연산자를 사용해 잘못된 높이 값을 수정합니다. 이렇게 통합된 경사 정보를 통해 더욱 정교한 지형 복원이 가능합니다.

- **Performance Highlights**: WHU-TLC 및 MVS3D 데이터셋에서의 실험 결과, 제안된 방법은 최신 기법보다 최소 16% 이상 MAE(Mean Absolute Error) 성능을 개선하고 2.5m 미만 오류율에서도 5% 이상 향상되었습니다. 이러한 결과는 TS-SatMVSNet의 우수한 일반화 능력을 подтверд해주며, 실질적 응용 가능성을 시사합니다.



### ZeroFlow: Overcoming Catastrophic Forgetting is Easier than You Think (https://arxiv.org/abs/2501.01045)
- **What's New**: 본 연구는 gradient ban 상황에서 forgetting 문제를 해결하기 위한 첫 번째 벤치마크 ZeroFlow를 소개합니다. 이 벤치마크는 여러 forgetting 시나리오와 데이터셋에서 forward pass 방법을 평가하며, 단지 forward pass만으로도 forgetting을 극복할 수 있음을 보여줍니다. 또한, 새로운 optimization 원리와 방법들이 발견되어, 단 하나의 forward pass로도 forgetting을 효과적으로 경감할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: Zeroth-order (ZO) 최적화는 수치 계산과 근사 알고리즘의 영역 내에서 광범위하게 연구되어 왔습니다. 이 방법은 첫 번째 차수(FO) gradient를 접근할 수 없거나 계산하기 어려운 경우에 대한 대안으로 작용합니다. 연구에서는 데이터 흐름의 동적 환경에서 여러 ZO 최적화 방법의 성능을 검토하고, 이를 통해 forward pass의 중요한 잠재력을 밝혀내고 있습니다.

- **Performance Highlights**: ZeroFlow 벤치마크를 통해 7개의 forward pass 최적화 알고리즘과 다양한 forgetting 시나리오 및 데이터셋에 대한 포괄적인 평가가 이루어졌습니다. 이를 통해 forgetting을 완화하기 위한 overlooked optimization 원칙이 드러났고, periodic gradient 기술을 통해 ZO 최적화를 효과적으로 개선할 수 있는 방안도 제시되었습니다. 이 연구는 단 한 번의 forward pass로도 forgetting을 성공적으로 경감할 수 있는 가능성을 보여줍니다.



### Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs (https://arxiv.org/abs/2501.01042)
- **What's New**: 이번 논문은 비디오 기반의 다중 모달 대형 언어 모델(V-MLLMs)에 대한 적대적 비디오 샘플의 전이 가능성을 최초로 조사한 것입니다. 기존 연구는 주로 화이트박스 공격에 집중되어 있었으나, V-MLLMs 간의 공격 전이 가능성이 실제 상황에서 어떻게 작용하는지를 살펴보았습니다. 이 연구는 다양한 V-MLLMs에 대해 적대적 비디오 샘플을 생성할 수 있는 새로운 방법인 Image-to-Video MLLM (I2V-MLLM) 공격을 소개합니다.

- **Technical Details**: I2V-MLLM 공격은 이미지 기반 다중 모달 모델(Imm)을 대리 모델로 활용하여 적대적 비디오 샘플을 생성합니다. 이 방법은 비디오의 주요 프레임을 추출하고 이를 IMM에 입력하여 적대적 섭동을 얻습니다. 또한, 비디오 표현을 방해하기 위한 다중 모달 상호작용과 시간적 정보를 통합하여 공격의 전이 가능성을 개선합니다. 다양한 프레임 샘플링 전략을 처리하기 위해 섭동 전파 기법도 도입하였습니다.

- **Performance Highlights**: 실험 결과, I2V-MLLM 공격은 여러 비디오-텍스트 다중 모달 작업에서 다양한 V-MLLM 간의 강력한 전이 가능성을 보였습니다. MSVD-QA 및 MSRVTT-QA 비디오 질문 응답 작업에서 각각 평균 55.48% 및 58.26%의 공격 성공률을 기록하였고, 기존 화이트박스 공격에 비해 경쟁력 있는 성능을 가집니다. 이 연구는 V-MLLM의 배포와 관련된 보안 위험을 줄이기 위한 중요한 기초를 제공합니다.



### Event Masked Autoencoder: Point-wise Action Recognition with Event-Based Cameras (https://arxiv.org/abs/2501.01040)
Comments:
          ICASSP 2025 Camera Ready

- **What's New**: 본 연구는 글로벌한 개념의 새로운 Event Masked Autoencoder(MAE)를 제안하여 이벤트 카메라에서 수집된 데이터를 점 구름(point clouds)으로 간주하고 마스킹 기법을 처음으로 적용합니다. 또한, 기존 DVS 기반 행동 인식 방법의 한계를 극복하기 위해, 이벤트 데이터를 공간적(spatial) 및 시간적(temporal) 구조를 보존하며 인식하는 프레임워크를 개발했습니다. 이 방법은 전통적인 프레임 기반 접근 방식에서의 시간 정보 손실 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 MAE 구조를 기반으로 하여 원시 이벤트 점 데이터를 패치(patch)로 그룹화하는 새로운 방법을 소개하였습니다. 이를 통해 우리는 이벤트의 시간 및 공간 정보를 보존하고, 다운스트림 작업, 예를 들어 행동 인식 및 복원(reconstruction)의 성능을 향상시킬 수 있습니다. 또한, 이벤트 데이터 스트림에는 시간 정보와 점 위치정보가 포함되어 있으며 이를 통해 포인트 클라우드 점 들을 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 공개 벤치마크에서 최첨단 성능을 초과하는 결과를 보여주었습니다. 이 논문은 이벤트 데이터 스트림 처리를 위한 Masked Modeling 방법이 효과적임을 입증하며, 다중 모달리티 데이터를 통합하는 데 있어 통합 백본의 가능성을 강조하고 있습니다. 향후 연구에서는 이러한 프레임워크를 다른 센서 및 데이터 타입으로 확장할 수 있는 잠재력을 가지고 있음을 논의합니다.



### DynamicLip: Shape-Independent Continuous Authentication via Lip Articulator Dynamics (https://arxiv.org/abs/2501.01032)
- **What's New**: 이번 연구에서는 전통적인 생체 인식 방법의 한계를 극복하고자, 입술의 동적 움직임을 기반으로 한 인증 시스템을 제안합니다. 기존의 입술 인식 기술은 정적인 형태에 의존하여 변동성이 적고 신뢰성이 낮았습니다. 따라서 우리는 입술 모양과 상관없는 지속적인 인증을 가능하게 하는 새로운 생체 인식 접근 방식을 개발하였습니다.

- **Technical Details**: 본 연구는 입술의 발음 운동을 분석하여 세 가지 주요 유형으로 분류하고, 이를 바탕으로 지속적인 사용자 인증을 제공합니다. 또한, 입술의 글로벌 및 로컬 특성을 캡처하는 계층적 모델을 구축하여 동적 입술 운동을 정밀하게 표현하고 인증 신뢰성을 높였습니다. 마지막으로 시암 신경망(Siamese neural network) 모델을 활용하여 사용자의 독특한 입술 동적 특징을 인증합니다.

- **Performance Highlights**: 우리는 다양한 환경에서 50명의 피험자를 대상으로 실험을 수행하였고, 실험 결과 시스템의 전체 정확도는 99.06%로 나타났습니다. 이 시스템은 고급 모방 공격 및 AI 딥페이크 공격에 대한 강인성을 입증하였으며, 공공 또는 공유 장치에서도 사용하기 적합합니다. 이를 통해 지속적인 생체 인증 시스템의 가능성을 제시하며, 높은 보안성이 특징입니다.



### Hadamard Attention Recurrent Transformer: A Strong Baseline for Stereo Matching Transformer (https://arxiv.org/abs/2501.01023)
- **What's New**: 하다마르 어텐션 리커런트 스테레오 트랜스포머(HART)는 효율적인 스테레오 매칭을 위한 새로운 접근 방식을 제안합니다. HART는 어텐션 메커니즘의 계산을 선형 시간 복잡도로 제어할 수 있도록 하다마르 곱을 사용하고, 밀집 어텐션 커널(DAK)을 설계하여 주목해야 할 세부 사항에 집중합니다. 이를 통해 HART는 반사와 약한 텍스처와 같은 도전적인 조건에서 성능을 개선합니다.

- **Technical Details**: HART는 하다마르 곱을 이용한 어텐션 계산을 통해 선형 복잡도를 달성하며, 이는 추론 속도를 크게 향상시킵니다. DAK는 출력에서 관련 특성과 무관한 반응을 강조하여 중요한 세부 사항에 집중하도록 돕습니다. 또한, MKOI(Multi Kernel & Order Interaction) 모듈을 통해 공간 및 채널 간 상호작용을 보완하여, 글로벌 및 로컬 정보를 효과적으로 캡처합니다.

- **Performance Highlights**: 실험 결과, HART는 KITTI 2012 벤치마크의 반사 영역에서 모든 발표된 방법 중 1위를 기록했습니다. HART의 어텐션 구조는 확장 가능하다는 특징도 지니고 있으며, 이는 스테레오 트랜스포머의 새로운 기준이 될 가능성을 보여줍니다. 이러한 성과는 기존의 트랜스포머 기반 스테레오 매칭 방법에 비해 속도와 정확도를 모두 향상시켰음을 나타냅니다.



### Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function (https://arxiv.org/abs/2501.01022)
- **What's New**: 이번 연구에서는 신경세포의 복잡한 형태를 재구성하기 위해 topology-aware neural network segmentation 방법을 제안합니다. 기존의 방법들이 가지는 높은 계산 비용의 문제를 해결하며, 딥러닝 기반의 접근법을 통해 세포를 더 효과적으로 분할할 수 있도록 개선되었습니다. 새로운 초분할(supervoxel) 개념을 도입하여 뇌 이미지의 특징을 잘 유지하면서 토폴로지 오류를 최소화합니다.

- **Technical Details**: 제안된 방법은 3차원 이진 이미지에서의 간단한 점(simple point) 개념을 확장하여 연결된 복셀 집합인 초복셀(supervoxel)을 대상으로 합니다. 이 접근법을 통해 신경망이 split 및 merge 오류를 최소화하도록 훈련할 수 있으며, 전체적인 학습 시간을 선형적으로 단축할 수 있습니다. 더불어, 그래프 표현을 통해 이미지의 복셀을 정점으로 간주하고, k-connectivity를 기반으로 한 간선 정의로 정확성을 높였습니다.

- **Performance Highlights**: 제안하는 방법은 새로운 공개 데이터셋인 마우스 뇌의 3D 현미경 이미지에서 그 효과를 입증하였으며, DRIVE, ISBI12 및 CrackTree 데이터셋과의 비교에서도 우수한 성능을 보였습니다. 이로 인해 신경 과학 및 이미지 분석 분야에서 큰 영향을 미칠 것으로 예상됩니다.



### Boosting Adversarial Transferability with Spatial Adversarial Alignmen (https://arxiv.org/abs/2501.01015)
- **What's New**: 본 논문에서는 Spatial Adversarial Alignment (SAA)라는 기법을 제안하여 DNN의 cross-architecture 적대적 전이 가능성을 향상시키고자 하였습니다. 기존의 adversarial transferability 방법들이 CNN에서 ViT로의 전이 경우에 제한적이었던 반면, SAA는 공간적 및 적대적 특성을 정교하게 조정함으로써 이러한 한계를 극복합니다. 특히, 이 방법은 안목 모델(witness model)과 surrogate 모델의 특성을 맞춤으로써 적대적 공격을 더욱 용이하게 만듭니다.

- **Technical Details**: SAA는 두 가지 주요 요소로 구성된 기법으로, 공간 인식 정렬(spatial-aware alignment)과 적대적 인식 정렬(adversarial-aware alignment)을 포함합니다. 공간 인식 정렬에서는 CNN과 ViT의 서로 다른 구조적 특성에도 불구하고 공통적인 특징을 맞추기 위해 글로벌 및 로컬 지역의 특징 차이를 최소화합니다. 적대적 인식 정렬에서는 자가 적대적 전략을 도입하여 다양한 아키텍처 간의 적대적 특징에 기반하여 모델을 학습시킵니다.

- **Performance Highlights**: 다양한 아키텍처에서 수행된 실험 결과, SAA를 기반으로 한 정렬된 surrogate 모델이 기존 방법보다 향상된 transferability를 보여주었습니다. 특히 ResNet50을 기준으로, SAA는 CNN에서 ViT로의 전이 가능성을 25.5%에서 39.1%까지 개선하였습니다. 이러한 결과는 SAA가 cross-architecture 공격에서 정보 공유 특성을 극대화하는 데 효과적임을 입증합니다.



### EasySplat: View-Adaptive Learning makes 3D Gaussian Splatting Easy (https://arxiv.org/abs/2501.01003)
Comments:
          6 pages, 5figures

- **What's New**: 이번 연구에서는 3D Gaussian Splatting (3DGS) 기반의 새로운 프레임워크 EasySplat을 소개합니다. 기존의 구조에서 문제점으로 지적되었던 Scene Initialization과 Point Cloud Densification의 한계를 극복하기 위해, 이미지 유사성을 기반으로 한 효율적인 그룹화 전략과 KNN 알고리즘을 사용하여 Gaussian뿐만 아니라 Camera Poses도 보다 정밀하게 초기화합니다.

- **Technical Details**: EasySplat은 Dense View Scene의 초기화를 위해 View Similarity에 기반한 Adaptive Grouping Strategy를 사용하며, KNN 알고리즘을 통해 Gaussian의 평균 형상과 비교하여 분할 여부를 결정합니다. 이 과정에서 robust pointmap priors를 활용해 고품질의 포인트 클라우드와 카메라 포즈를 얻고, Gaussian primitives의 Densification은 주변 ellipsoids와의 형상 불일치를 반영하여 동적으로 이뤄집니다.

- **Performance Highlights**: Extensive 실험 결과, EasySplat은 Novel View Synthesis 효율성과 성능 모두에서 현재의 최첨단 기법들(SOTA)을 초과하는 성능을 보였습니다. 새로운 방식의 Densification 접근법이 부족한 포인트가 있는 영역에서 더 나은 질의 3DGS 모델링을 가능하게 하였고, 이로 인해 전반적인 렌더링 품질이 향상되었습니다.



### CoordFlow: Coordinate Flow for Pixel-wise Neural Video Representation (https://arxiv.org/abs/2501.00975)
- **What's New**: 이 논문에서는 CoordFlow라는 새로운 픽셀 기반의 암시적 신경 표현(Implicit Neural Representation, INR)을 소개합니다. 이 방법은 기존의 픽셀 기반 INRs보다 뛰어난 성능을 자랑하며, 주요 프레임 기반 기술들과도 동등한 성능을 냅니다. CoordFlow는 시각적 정보를 여러 레이어로 분리하고, 각 레이어의 움직임을 보정하는 네트워크를 통해 더 나은 비디오 압축을 가능하게 합니다.

- **Technical Details**: CoordFlow는 자연 비디오의 시간적 중복성을 활용하여 비디오 시퀀스를 여러 연결된 레이어로 분해하는 방식을 취합니다. 각 레이어에는 하나의 움직임 보정 네트워크와 데이터 인코딩 네트워크가 배치됩니다. 이 구조는 효과적인 RGB 및 알파 값을 생성하여 비디오를 더 응집력 있게 표현합니다.

- **Performance Highlights**: 제안된 CoordFlow 방법은 비디오 압축 분야에서 최첨단 성능을 달성했습니다. 이 모델은 비디오 업샘플링, 안정화, 인페인팅(inpainting), 잡음 제거(denoising) 기능을 내재화하고 있어 다양한 비디오 처리 응용 분야에서 높은 유연성을 보여줍니다. 궁극적으로, CoordFlow는 비디오 콘텐츠의 품질을 향상시키면서도 필요한 데이터 양을 줄이는 데 기여할 것으로 기대됩니다.



### OASIS Uncovers: High-Quality T2I Models, Same Old Stereotypes (https://arxiv.org/abs/2501.00962)
- **What's New**: 이번 연구에서는 T2I(text-to-image) 모델에서 발생하는 시각적 편향과 고정관념을 정량적으로 측정하기 위한 새로운 기준인 OASIS를 제안합니다. 기존의 정량적 방법들이 사회학적 정의와 일치하지 않음을 지적하며, OASIS를 통해 생성된 이미지 데이터셋에서 고정관념을 평가할 수 있는 두 가지 점수를 제공합니다. 특히, M1은 고정관념 속성의 분포 위반을 측정하고, M2는 속성에 따른 스펙트럼 변동성을 측정합니다.

- **Technical Details**: OASIS는 텍스트 프롬프트에 따라 생성된 이미지에서 나타나는 고정관념을 분석하는 도구입니다. 이 도구는 T2I 모델의 내부에서 개념과 연관되는 속성을 발견하고, 이미지 생성 과정에서 고정관념 속성의 출현 정도를 정량화하는 두 가지 방법도 병행합니다. 이러한 접근은 기존의 평가 방법들, 특히 인간 주관 평가의 제한점을 보완하기 위한 것입니다.

- **Performance Highlights**: 연구 결과, 최신 T2I 모델인 FLUX.1 및 SDv3는 여전히 강력한 고정관념의 경향성을 보이며, 특히 낮은 인터넷 발자국을 가지는 국적에 대한 편향이 증가하는 경향이 발견되었습니다. 이는 이러한 모델들이 다양한 사회적 문제에 영향을 미칠 수 있음을 시사하며, 고정관념을 저감하기 위한 자동화된 감시 및 규제가 필요함을 강조합니다.



### 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining (https://arxiv.org/abs/2501.00958)
Comments:
          Under review

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)을 위한 고품질의 	extbf{multimodal textbook} 말뭉치(corpus)를 소개하고 있습니다. 기존의 이미지-텍스트 쌍 데이터에 비해, 이 데이터 세트는 풍부한 기초 지식을 제공하며, 2.5년 이상에 걸친 총 22,000시간의 강의 비디오를 수집하여 구성되었습니다. 이 데이터는 교육 비디오에서 체계적으로 수집되어 시간적 순서에 따라 이미지-텍스트 interleaved corpus로 정리되었습니다.

- **Technical Details**: 연구진은 LLM(대규모 언어 모델)이 제안한 분류 체계를 사용하여 교육 비디오를 체계적으로 수집하고, 키프레임(keyframes), 자동 음성 인식(ASR) 및 텍스트 인식(OCR) 기술을 통해 비디오에서 시각적 및 음성적 지식을 추출합니다. 이러한 과정을 거쳐 만든 multimodal textbook은 이전의 데이터 세트들보다 더 일관되고 풍부한 지식을 제공합니다. 특히, 이 데이터는 이미지와 텍스트 간의 정렬이 향상되어 있습니다.

- **Performance Highlights**: 실험 결과, 새로운 textbook을 이용해 사전 훈련(pretraining)한 VLMs는 ScienceQA 및 MathVista와 같은 지식 및 추론 집약적 작업에서 뛰어난 성능을 보였습니다. 또한, 이 VLM들은 이미지와 텍스트의 단서( cues)를 활용하여 몇 가지 경우(few-shot)에서 작업을 해결하는 데 있어 훌륭한 interleaved context awareness를 나타냅니다.



### Cached Adaptive Token Merging: Dynamic Token Reduction and Redundant Computation Elimination in Diffusion Mod (https://arxiv.org/abs/2501.00946)
- **What's New**: 이번 논문에서는 고차원 이미지를 생성하기 위해 적극적으로 사용되고 있는 diffusion 모델의 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 연구진은 self-attention 메커니즘의 복잡성을 줄이기 위해 token merging 방식의 변형인 cached adaptive token merging (CA-ToMe) 방법을 도입하였습니다. 이 방법은 가장 유사한 토큰을 합치는 과정을 통해 연산 비용을 줄이면서도 기존 방법과 유사한 품질의 이미지를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: CA-ToMe는 토큰 간의 유사성을 계산하여 유사한 토큰을 병합하는 adaptive threshold 기법을 적용합니다. 이를 통해 인접 단계에서의 반복적인 패턴을 보다 효율적으로 처리할 수 있으며, 이를 위한 캐싱 메커니즘도 도입하여 유사한 토큰 쌍을 저장합니다. 이러한 방법을 통해 실험 결과 CA-ToMe는 기존 접근 방법 대비 denoising 과정의 속도를 1.24배 향상시키는 성과를 보여주었습니다.

- **Performance Highlights**: CA-ToMe 방법은 training-free 방식으로 작동하며, 기존의 방법들과 비교했을 때 더 빠른 이미지 생성 속도를 유지함과 동시에 같은 FID 점수를 유지하는 특징이 있습니다. 이 연구는 diffusion 모델이 가지는 높은 지연 시간과 컴퓨팅 비용 문제를 해결하는 데 기여하며, 실제 응용 분야에서 더욱 효과적인 활용이 가능함을 보여주었습니다.



### Diffusion Prism: Enhancing Diversity and Morphology Consistency in Mask-to-Image Diffusion (https://arxiv.org/abs/2501.00944)
- **What's New**: Diffusion Prism은 기존의 훈련이 필요 없는 프레임워크로, 이진 마스크를 사실적이고 다양한 샘플로 변환하는 방법을 제안합니다. 기존의 diffusion 모델들이 저에너지 및 스파스(sparse) 입력 이미지에서 다양성이 제한되는 문제를 해결하고자 합니다. 이 방법은 morphology 일관성을 유지하여 생성된 이미지의 질을 높입니다.

- **Technical Details**: Diffusion Prism은 미리 훈련된 Stable Diffusion v1.5 모델에 기반하여 픽셀 공간에서 입력 이미지를 조작해 도메인 전이를 수행합니다. 이 방법은 노이즈와 색수차(chromatic aberration)를 도입하여 이미지의 다양성과 질감을 향상시킵니다. 또한, 제안된 방법은 입력 정보를 손상시키지 않으면서도 차별화된 이미지를 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Diffusion Prism은 dendritic 패턴과 같은 고에너지 랜덤 패턴에서 이미지 다양성을 현저히 향상시키는 것으로 나타났습니다. 기존의 조절 가능한 diffusion 모델에 비해, 제안한 방법은 이진 마스크의 morphology 구조의 무결성을 유지하면서도 다양성을 크게 개선할 수 있음을 보여줍니다.



### Multiscaled Multi-Head Attention-based Video Transformer Network for Hand Gesture Recognition (https://arxiv.org/abs/2501.00935)
- **What's New**: 이 논문에서는 동적 손 제스처 인식을 위한 새로운 모델인 다중 스케일 다중 헤드 어텐션 비디오 변환기 네트워크(MsMHA-VTN)를 제안합니다. 본 모델은 변환기(multiscale head attention) 기반의 다중 스케일 특징 추출을 통해 제스처 인식 정확도를 향상시킵니다. 특히, 전통적인 단일 모달에 비해 다중 모달에서도 뛰어난 성능을 보이며, RGB, 깊이, 적외선 이미지와 같은 다양한 입력을 활용합니다.

- **Technical Details**: 제안된 MsMHA-VTN 모델은 다중 스케일 어텐션을 이용하여 다양한 스케일에서 제스처 정보를 캡처합니다. 입력 비디오 샘플은 다수의 특성을 가진 다중 헤드 어텐션 구조를 통해 처리되며, 각 헤드는 서로 다른 주의 차원을 사용하여 멀티 스케일 수준에서 특징을 강화합니다. 이 구조는 전통적인 변환기 모델에서의 한계를 극복하고 보다 정교한 제스처 인식을 가능하게 합니다.

- **Performance Highlights**: 제안된 모델은 NVGesture 및 Briareo 데이터셋에서 각각 88.22% 및 99.10%의 우수한 정확도를 달성하며, 기존의 방법들과 비교했을 때 최신의 최첨단 결과를 보여줍니다. 본 논문에서 제안한 프레임워크는 단일 및 다중 입력의 효과성을 입증했으며, 향후 동적 손 제스처 인식 연구에 상당한 기여를 할 것으로 기대됩니다.



### Hierarchical Vision-Language Alignment for Text-to-Image Generation via Diffusion Models (https://arxiv.org/abs/2501.00917)
- **What's New**: 이 논문은 Vision-Language Aligned Diffusion (VLAD) 모델을 소개하며, 이는 고유한 듀얼 스트림 전략을 통해 복잡한 텍스트 설명과 시각적으로 일관된 이미지를 효과적으로 정렬하는 문제를 해결합니다. VLAD는 Contextual Composition Module (CCM)을 활용하여 텍스트 프롬프트를 전역 및 지역 표현으로 분해하고, 다단계 확산 과정을 통해 고품질의 이미지를 생성합니다. 실험 결과, VLAD는 최신 방법들에 비해 이미지 품질과 의미적 정렬에서 상당한 성능 향상을 보입니다.

- **Technical Details**: VLAD 모델은 사전 훈련된 Largе Vision-Language Model (LVLM)과 계층적 확산 프로세스를 통합하여 텍스트와 이미지를 정렬하는 데 최적화되어 있습니다. 모델은 텍스트 및 시각적 피처를 공유되는 의미 공간으로 임베딩하는 조정 모듈을 사용하고, 이 임베딩을 기반으로 고품질의 이미지를 생성하는 계층적 확산 프로세스를 채택합니다. 이를 통해 VLAD는 텍스트 설명을 고지식으로 반영하고 고품질 이미지를 생성합니다.

- **Performance Highlights**: VLAD는 MARIO-Eval 및 INNOVATOR-Eval 벤치마크에서 FID, CLIP Score, OCR 기반 메트릭을 포함한 평가 지표에서 우수한 성능을 발휘합니다. 특히, 기존의 TextDiffuser 및 ARTIST 방법들보다 일관된 정렬 정확도와 높은 시각적 품질을 달성했습니다. 이러한 결과는 VLAD가 복잡한 텍스트-이미지 생성 작업에서 유망한 접근법임을 시사합니다.



### AutoPresent: Designing Structured Visuals from Scratch (https://arxiv.org/abs/2501.00912)
- **What's New**: 이번 연구는 프레젠테이션 슬라이드를 자연어(NL) 명령어로부터 자동 생성하는 문제를 다룹니다. 연구자들은 7,000개의 교육 샘플과 585개의 테스트 샘플을 포함하는 SlidesBench 벤치마크를 소개하고, 이를 통해 슬라이드 생성 성능을 평가합니다. 추가로, AutoPresent라는 8B Llama 기반 모델을 개발하여 고품질 슬라이드를 생성할 수 있도록 하였습니다.

- **Technical Details**: SlidesBench는 다양한 난이도의 자연어 지시와 PPTX 형식의 슬라이드를 포함하는 310개의 슬라이드 덱에서 수집된 데이터입니다. 두 가지 평가 지표를 제공하여 생성된 슬라이드의 질을 평가하는 데 도움을 줍니다: 참조 기반 지표와 참조 없는 지표입니다. 또한, 프로그램 생성 방식을 통해 사용자 지침을 따른 슬라이드를 생성하는 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 프로그램 생성 방식이 이미지 생성 방식보다 훨씬 높은 품질의 슬라이드를 생성함을 보여주었습니다. AutoPresent 모델은 최첨단 성능을 달성하며 기존의 닫힌 소스 모델인 GPT-4o에 근접한 결과를 나타냈습니다. 연구자들은 자기 수정(iterative refinement) 과정을 통해 슬라이드 품질을 더욱 향상시킬 수 있는 가능성을 발견하였습니다.



### Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Mod (https://arxiv.org/abs/2501.00895)
- **What's New**: 본 논문에서는 원거리 센싱 텍스트-이미지 생성 기술에 있어 기존 데이터셋의 한계를 극복하고, 글로벌 스케일에 맞춘 새로운 Git-10M 데이터셋과 텍스트 기반 생성 모델 Text2Earth를 제안합니다. Git-10M은 1천만 개의 이미지-텍스트 쌍으로 구성되어 있으며, 기존 데이터셋보다 5배 더 크고 다양한 지리적 현장을 포함합니다. Text2Earth는 13억 개의 파라미터를 가지며, 다중 해상도를 지원하는 원거리 센싱 장면 생성을 위한 강력한 확산 모델로 설계되었습니다.

- **Technical Details**: Text2Earth 모델은 Variational Autoencoder(VAE)를 활용하여 이미지를 효율적으로 압축하고 복원하며, OpenCLIP ViT-H 텍스트 인코더를 통합하여 텍스트를 고차원 시맨틱 임베딩으로 변환합니다. 또한, 해상도 제어를 위한 해상도 가이던스 메커니즘이 적용되어, 생성 과정의 각 단계에서 해상도 정보를 인코딩하여 질 높은 이미지를 생성합니다. 이와 함께 동적 조건 적응 전략이 훈련과 추론 과정의 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: Text2Earth는 제로샷 텍스트-이미지 생성, 무한한 원거리 센싱 장면 구성, 이미지 편집 및 교차 변환 이미지 생성 등 여러 작업에서 뛰어난 성능을 보여줍니다. 특히, RSICD 벤치마크 데이터셋에서 FID 지표에서 +26.23의 개선과 함께 +20.95%의 정확도를 기록하며 업계에서의 경쟁력을 입증하였습니다. 이러한 개선은 고정된 크기와 제한된 장면 유형에 얽매이지 않고 다양한 작업에 유연하게 적용될 수 있는 가능성을 보여줍니다.



### FullTransNet: Full Transformer with Local-Global Attention for Video Summarization (https://arxiv.org/abs/2501.00882)
Comments:
          16 pages, 8 figures, 4 tables; The code is at this https URL

- **What's New**: 본 논문에서는 비디오 요약(video summarization)에 대한 새로운 접근법인 전체 트랜스포머(full transformer) 아키텍처를 제안합니다. 기존에는 주로 순환 신경망(RNN)이나 합성곱 신경망(CNN) 또는 인코더 전용 트랜스포머가 사용되었으나, 우리는 인코더-디코더 구조의 전체 트랜스포머가 비디오 요약 작업에 적합하다고 주장합니다. 이 방법은 supervised video summarization을 seq2seq 학습 문제로 간주하여 비디오에서 가장 관련성 높은 부분을 효과적으로 선택하고 요약합니다.

- **Technical Details**: 전체 트랜스포머 아키텍처인 FullTransNet을 제안하며, 이는 지역-전역 희소 주의(local-global sparse attention) 메커니즘을 사용하여 긴 범위의 종속성을 모델링하면서 계산 비용을 줄이는 방안입니다. 인코더는 원본 비디오의 프레임 시퀀스를 입력으로 받아들이고, 디코더는 훈련 중에는 정답 시퀀스를, 추론 중에는 예측된 시퀀스를 생성합니다. 이렇게 함으로써, 인코더는 모든 프레임을 동시에 인코딩하여 상관 관계를 파악하고, 디코더는 이러한 정보를 이용하여 요약을 생성합니다.

- **Performance Highlights**: FullTransNet은 공공 멀티미디어 벤치마크 데이터셋인 SumMe와 TVSum에서의 광범위한 실험을 통해 기존의 비디오 요약 접근법을 능가하는 성능을 보여주었습니다. SumMe에서 54.4%의 F-Measure, TVSum에서 63.9%의 F-Measure를 달성하며, 상대적으로 낮은 계산 및 메모리 요구 사항으로 이러한 결과를 검증하였습니다. 또한, 주의(attention) 맵 시각화를 통해 우리의 FullTransNet이 어떻게 작동하는지를 명확히 보여줍니다.



### Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction (https://arxiv.org/abs/2501.00880)
- **What's New**: 최근 LLM(대형 언어 모델)을 시각적 생성에 활용하는 연구가 활발해지고 있습니다. 하지만 기존의 방법들은 LLM 아키텍처를 시각적 생성에 그대로 적용하는 데 그치며, 언어와 시각 간의 근본적인 차이에 대한 조사가 부족한 실정입니다. 본 논문은 LLM 프레임워크 하의 시각 임베딩 공간의 특성을 탐구하고, 이를 통해 보다 안정적이고 강력한 생성 결과를 도출할 수 있음을 발견하였습니다.

- **Technical Details**: 이 논문에서는 IAR(Improved AutoRegressive Visual Generation Method)라는 새로운 방법론을 제안합니다. 이 방법은 Balanced K-means 클러스터링 알고리즘을 활용한 코드북 재배열 전략을 포함하여, 각 클러스터 내의 시각 특징들이 높은 유사성을 가지도록 합니다. 또한, 클러스터 지향적 크로스 엔트로피 손실을 도입하여 모델이 토큰의 올바른 클러스터를 예측하도록 유도합니다.

- **Performance Highlights**: 광범위한 실험을 통해 본 방법이 LlamaGen 모델의 훈련 효율성과 성능을 일관되게 개선함을 보였습니다. 100M부터 1.4B까지 다양한 매개변수 스케일에서 본 방법은 훈련 시간을 절반으로 줄이고, 동일한 FID를 유지하면서 생성 품질을 향상시켰습니다. 이 방법은 다양한 LLM 기반의 시각 생성 모델에 적용할 수 있으며, 향후 연구에서 유망한 방향성을 제공합니다.



### FGAseg: Fine-Grained Pixel-Text Alignment for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2501.00877)
- **What's New**: 본 논문에서는 FGAseg라는 새로운 오픈 보캐블러리 세그멘테이션 모델을 제안합니다. 이 모델은 픽셀 수준의 정밀한 텍스트 정렬과 카테고리 경계를 보완하기 위해 설계되었습니다. 특히, FGAseg는 VLM(Visual-Language Model)에서 추출한 정보를 활용하여 세분화된 마스크 예측을 가능하게 합니다.

- **Technical Details**: FGAseg의 핵심 구성 요소는 Pixel-Level Alignment 모듈과 Category Information Supplementation 모듈입니다. Pixel-Level Alignment 모듈은 cross-modal attention 기법과 텍스트-픽셀 정렬 손실(Text-Pixel Alignment Loss, T2Ploss)을 통합하여 CLIP의 정합성을 보다 정밀한 픽셀 수준으로 조정합니다. 또한, Category Information Supplementation 모듈은 최적화 가능한 의사 마스크를 사용하여 카테고리 경계 정보를 축적합니다.

- **Performance Highlights**: FGAseg는 개방형 어휘 세그멘테이션 영역에서 기존 방법보다 우수한 성능을 보여줍니다. 실험 결과는 다양한 데이터셋에서 FGAseg가 보다 정확한 픽셀 기반의 정렬과 카테고리 경계 정보를 제공하며, 이러한 정보들이 전체적으로 세그멘테이션 성능을 향상시킨다는 것을 증명합니다.



### Exploring Structured Semantic Priors Underlying Diffusion Score for Test-time Adaptation (https://arxiv.org/abs/2501.00873)
Comments:
          Accepted by NeurIPS 2024. Project page: this https URL

- **What's New**: 이 논문은 생성 모델(generative model)과 판별 모델(discriminative model)의 상호 보완적 장점을 이용한 새로운 방법론 DUSA를 제시합니다. DUSA는 디퓨전 스코어(difussion score)를 기반으로 한 구조화된 의미적 사전(structured semantic prior)을 활용하여 이미지 분류기 또는 밀집 예측기(dense predictor)의 테스트 시간 적응(test-time adaptation)을 용이하게 합니다. 또한, 단일 time step에서 노이즈 감소를 통해 지식 추출의 새로운 가능성을 탐구합니다.

- **Technical Details**: DUSA는 디퓨전 모델에서 감춰진 의미적 구조를 활용하여 판별 사전(discriminative prior)을 추출하는 이론적 모델을 제공합니다. 스코어 기반의 디퓨전 모델(score-based diffusion model)은 다양한 테스트 시나리오에서 효율적으로 사용할 수 있는 강력한 도구로 자리잡고 있습니다. 또한, DUSA는 다양한 태스크 직접 모델(task model)과 적응 프로토콜(test-time adaptation protocols)에서 효과성을 입증하며, 이를 통해 성능을 향상시킵니다.

- **Performance Highlights**: DUSA는 ConvNeXt-L에서 완전 및 지속적인 테스트 시간 적응(JFrame)에서 각각 +5.1% 및 +7.3%의 성능 향상을 이루었으며, SegFormer-B5에서의 테스트 시간 의미 세분화(test-time semantic segmentation)에서 +4.2%의 성능 개선을 보여주었습니다. 이러한 우수한 성능은 DUSA가 디퓨전 모델로부터 가치 있는 사전(prior)을 추출하는 데 뛰어난 능력을 가지고 있음을 증명합니다.



### Scale-wise Bidirectional Alignment Network for Referring Remote Sensing Image Segmentation (https://arxiv.org/abs/2501.00851)
Comments:
          Under review

- **What's New**: 이번 논문은 Scale-wise Bidirectional Alignment Network (SBANet)이라는 혁신적 프레임워크를 제안하여, 원거리 영상 세분화(referring remote sensing image segmentation, RRSIS) 분야에서 기존 모델들의 한계를 극복하고자 한다. SBANet은 비전-언어 상호작용을 개선하고, 언어적 특징과 시각적 특징을 효과적으로 분석하는 것을 목표로 한다. 특히, Bidirectional Alignment Module (BAM)과 동적 피쳐 선택 블록을 통해 다양한 공간적 스케일에서 정보 교환을 강화한다.

- **Technical Details**: SBANet은 비단방향 정렬 모듈과 텍스트 조건부 채널 및 공간 집계기를 포함하여 언어와 비전 간의 정보를 유기적으로 교환하도록 설계되었다. BAM은 학습 가능한 쿼리 토큰을 활용해 중요 토큰과 연관된 시각적 및 언어적 피쳐를 선택적으로 표현한다. 또한, 각 스케일 수준에서 글로벌 컨텍스트와 로컬 디테일을 포착하기 위한 동적 피쳐 선택 블록을 포함하여 피쳐 세분화를 강화한다.

- **Performance Highlights**: 광범위한 실험 결과, SBANet은 RRSIS-D 및 RefSegRS 데이터셋에서 이전의 최첨단 방법들보다 우수한 세분화 성능을 달성하였음을 보여준다. 양적 및 질적으로 모두 적절한 성능 향상을 나타냈으며, 논문 발표 이후 코드가 공개될 예정이다.



### IllusionBench: A Large-scale and Comprehensive Benchmark for Visual Illusion Understanding in Vision-Language Models (https://arxiv.org/abs/2501.00848)
- **What's New**: 이 논문은 기존 Visual Language Models (VLMs)이 실제 시나리오에서 시각적 환영을 처리하는 데에 어려움을 겪고 있음을 강조합니다. 이를 해결하기 위해 IllusionBench라는 새로운 데이터셋을 소개하며, 고전적 인지 환영뿐 아니라 실제 장면에서의 환영을 포함한 총 1,051개의 이미지와 5,548개의 질문-답변 쌍이 포함됩니다. 연구의 목표는 VLMs의 시각적 이해 능력을 향상시키는 것입니다.

- **Technical Details**: IllusionBench 데이터셋은 고전적인 시각적 환영, 현실 장면 환영, 색맹 검사 이미지, 그리고 트랩 환영 이미지를 포함한 1,051개의 이미지를 기반으로 구성됩니다. 각 이미지는 질문-답변 쌍 및 수동으로 주석이 달린 설명과 함께 제공되어 시각적 환영의 존재, 원인 및 내용을 평가합니다. 연구팀은 진실 또는 거짓, 다중 선택, 개방형 질문과 같은 다양한 작업을 통해 최신 VLMs를 종합적으로 평가합니다.

- **Performance Highlights**: 최상의 성능을 보이는 모델인 GPT-4o는 진실 또는 거짓 작업에서 80.59%의 정확도, 다중 선택 질문에서 76.75%의 정확도를 달성했습니다. 그러나 여전히 인간 성능에는 미치지 못하는 것으로 나타났으며, 전형적인 환영에서 발생하는 환각 현상으로 인해 트랩 환영에 대한 점수가 낮았습니다. IllusionBench는 VLMs의 시각적 환영 이해 능력을 평가하기 위한 가장 포괄적이고 큰 벤치마크로 자리잡고 있습니다.



### FusionSORT: Fusion Methods for Online Multi-object Visual Tracking (https://arxiv.org/abs/2501.00843)
- **What's New**: 이 논문에서는 다중 객체 비주얼 추적에서 탐지 결과를 트랙렛(tracklet)과 연관시킬 수 있는 네 가지 서로 다른 융합(fusion) 방법을 조사합니다. 이들은 최소값(minimum), IoU 기반 가중치 합(weighted sum), 칼만 필터(Kalman Filter) 게이팅(gating), 비용의 하다마드 곱(Hadamard product) 등의 방법을 포함합니다. 또한 강력한 단서(cues)인 운동 정보와 외관 정보 외에도 약한 단서인 키(ht-IoU) 및 트랙렛 신뢰도 정보를 고민하여 다양한 데이터 연관(data association) 방식을 탐구합니다.

- **Technical Details**: 다중 객체 비주얼 추적의 데이터 연관 문제는 실종 탐지(missed detections), 외관 변화 및 노이즈 탐지(noisy detections) 등의 도전에 의해 복잡성을 띕니다. 이 논문은 칼만 필터(KF)를 사용하여 트랙렛을 추적하고, 강력한 단서와 약한 단서를 통해 최종 비용 행렬을 계산합니다. 두 단계로 이루어진 매칭 전략을 구현하여 첫 번째 매칭에서는 신뢰도가 높은 탐지를, 두 번째 매칭에서는 저신뢰도 탐지를 추적합니다.

- **Performance Highlights**: MOT17, MOT20, DanceTrack 데이터셋에서 광범위한 평가를 실시하여 융합 방법의 선택이 다중 객체 비주얼 추적에서 데이터 연관의 핵심임을 강조합니다. 이를 통해 각 융합 방법이 가진 장단점을 비교하였고, 가장 효과적인 방법을 제시하게 되었습니다. 본 연구는 다양한 융합 방법을 통해 더 나은 성능을 이끌어낼 수 있는 가능성을 보여줍니다.



### Spatially-guided Temporal Aggregation for Robust Event-RGB Optical Flow Estimation (https://arxiv.org/abs/2501.00838)
Comments:
          12 pages, 8 figures, under review

- **What's New**: 본 논문은 프레임과 이벤트 데이터를 활용하여 새로운 방식으로 Optical Flow 추정의 효율성을 개선하고자 합니다. 기존 방법들은 보통 정보를 단순히 쌓는 방식으로 보완적 장점을 활용하지 못했으나, 이 연구에서는 공간적으로 조밀한 모달리티를 통해 시간적으로 조밀한 이벤트 모달리티의 집계를 유도하는 혁신적인 접근 방식을 제안합니다. 이를 통해 프레임의 세밀한 질감과 이벤트의 기본 구조를 유지하는 이벤트 보강 프레임 표현을 도입하여 효과적인 교차 모달 융합을 달성합니다.

- **Technical Details**: 안정적인 공간적 대응을 제공하는 프레임의 특성과 풍부한 시간적 정보를 제공하는 이벤트 특성을 분석하여, 두 모달리티의 장점을 결합하려고 합니다. 프레임 데이터를 활용하여 강력한 공간적 대응을 생성하고, 이벤트 데이터를 통해 시간적으로 조밀한 상관관계를 구축합니다. 또한, Transformer 기반의 모듈을 설계하여 희소한 이벤트 모션 피처를 공간적으로 풍부한 프레임 정보와 보완하고, 전체적인 정보 전파를 향상시키는 방법을 제시합니다.

- **Performance Highlights**: MVSEC 및 DSEC-Flow 데이터셋에서의 실험 결과, 우리의 프레임워크가 최고의 성능을 달성한 것으로 나타났습니다. 이벤트 전용 모델에 비해 프레임 유도를 통해 정확성과 효율성을 각각 10%와 45% 개선하였고, 기존의 최첨단 융합 기반 방법에서도 4%의 정확도 향상을 이루었습니다. 이는 프레임과 이벤트의 보완적 강점을 효과적으로 활용한 결과입니다.



### Recognizing Artistic Style of Archaeological Image Fragments Using Deep Style Extrapolation (https://arxiv.org/abs/2501.00836)
Comments:
          To be published in the 27th International Conference on Human-Computer Interaction (HCII 2025)

- **What's New**: 본 논문은 고대 예술 작품의 파편을 분류하기 위한 새로운 딥러닝 프레임워크를 제안합니다. 현재의 고급 딥러닝 구조를 통해 예술 스타일을 예측하여 다양한 스타일과 기하학적 형상을 가진 작품에 대해 최첨단 결과를 달성했습니다. 특히, 파편의 기하학적 특성이 분류 정확도에 미치는 영향을 검증하기 위한 새로운 데이터셋도 소개됩니다.

- **Technical Details**: 저자들은 예술 스타일 인식을 위한 새로운 딥러닝 아키텍처를 제안하며, 이는 여러 벤치마크에서 SOTA 결과를 달성했습니다. 최근 방식인 Convolutional Neural Networks (CNNs)를 활용하여 예술 작품의 파편에서 시각적 스타일을 인식하는 방법을 다루며, 이것이 기존의 전통적인 기계 학습 방법보다 우수함을 보입니다. 이 시스템은 손상된 예술 작품을 분석하거나 큰 구성요소 내의 세부사항을 검사할 때 유용합니다.

- **Performance Highlights**: 제안된 방법은 고대 예술 작품의 스타일을 자동으로 인식하고 분류하는 데 있어 기존의 연구들에 비해 더 높은 정확도를 보여줍니다. 새로운 데이터셋인 Pompeii Archive Artistic-styles Fragments (POMPAAF) 데이터셋은 전문가 검증을 거친 고품질 이미지로 구성되어 있어, 다양한 크기와 형태의 파편을 포함하고 있습니다. 이러한 고유한 데이터셋을 통해 예술 스타일 분류의 도전 과제에 대한 깊은 통찰을 제공합니다.



### SPARNet: Continual Test-Time Adaptation via Sample Partitioning Strategy and Anti-Forgetting Regularization (https://arxiv.org/abs/2501.00818)
Comments:
          8 pages, 2 figures

- **What's New**: 본 연구에서는 Test-time Adaptation (TTA)를 강화하기 위한 새로운 프레임워크, SPARNet을 제안합니다. 기존의 TTA가 정적인 타겟 도메인에 중점을 두었다면, 이 연구는 계속적인 도메인 변화에 적응하는 새로운 접근법을 제공합니다. SPARNet은 샘플을 신뢰할 수 있는 것과 신뢰할 수 없는 것으로 분류하는 샘플 파르티셔닝(strategy) 전략을 포함합니다.

- **Technical Details**: SPARNet의 샘플 파르티셔닝 전략은 신뢰할 수 있는 샘플과 신뢰할 수 없는 샘플의 두 그룹으로 나누는 방식입니다. 각 그룹에 대한 특성을 고려하여 서로 다른 처리 전략을 적용합니다. 또한, 과도한 변화로부터 중요한 파라미터를 제한하는 정규화(term)를 도입하여, 모델의 재기억(catastrophic forgetting) 문제를 완화합니다.

- **Performance Highlights**: CIFAR10-C, CIFAR100-C, ImageNet-C 데이터셋을 사용한 실험을 통해 SPARNet의 효과를 입증하였습니다. 이러한 실험은 계속적인 TTA 시나리오에서 모델의 장기적인 적응력 향상을 확인하는 데 초점을 맞추었습니다. 결과적으로, SPARNet은 모델의 성능을 향상시키고, 다양한 도메인 변화에 보다 효과적으로 대응할 수 있도록 설계되었습니다.



### MixSA: Training-free Reference-based Sketch Extraction via Mixture-of-Self-Attention (https://arxiv.org/abs/2501.00816)
Comments:
          25 pages, 25 figures; Accepted by IEEE IEEE Transactions on Visualization and Computer Graphics, 2024 (TVCG)

- **What's New**: 본 논문에서는 Mixture-of-Self-Attention (MixSA)이라는 새로운 스케치 추출 방법을 소개합니다. 이 방법은 기존의 방법들이 요구하는 광범위한 훈련 없이도 강력한 diffusion priors를 활용하여 스케치를 향상시킬 수 있도록 설계되었습니다. MixSA는 참조 스케치의 키(key)와 값(value)을 사용하여 self-attention 레이어를 조작함으로써 초기 윤곽 이미지를 세밀하게 조정할 수 있는 특성을 지니고 있습니다.

- **Technical Details**: MixSA의 핵심은 self-attention 기법을 결합하여 스케치 스타일을 변형하는 것입니다. 특히, 참조 스케치의 정보를 활용하여 색상 분포와 질감을 조정하고, 이는 각기 다른 스타일 간의 보간(interpolation)을 가능하게 합니다. 또한, Lucid Diffusion을 통해 기존 문제인 색상 평균화(color averaging)를 해결하여 높은 대조를 유지한 스케치를 생성합니다.

- **Performance Highlights**: 실험 결과, MixSA는 스케치 품질, 유연성 및 적용 가능성 면에서 타 방법들에 비해 뛰어난 성능을 보였습니다. 이 방법은 사용자가 원하는 다양한 예술적 표현을 정확하게 반영할 수 있는 기능을 제공하며, 특히 보이지 않는 스타일을 생성하는 데 강점을 나타냅니다. 그 결과, 스케치 생성을 시의적절하고 다양한 맥락에서 효과적으로 수행할 수 있습니다.



### Regression Guided Strategy to Automated Facial Beauty Optimization through Image Synthesis (https://arxiv.org/abs/2501.00811)
Comments:
          Short paper, 5 pages

- **What's New**: 최근 소셜 미디어에서 사용되는 뷰티 필터는 개인의 외모를 향상시키는 데 매우 효과적입니다. 기존의 방식은 얼굴의 매력적인 특징에 대한 도메인 지식을 활용한 규칙 기반 접근법이나 특정 변환을 적용하여 이질적입니다. 본 연구에서는 사전 훈련된 GAN의 잠재 공간에서 얼굴 이미지를 포인트로 투영하고, 이를 최적화하여 아름다운 얼굴을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 StyleGAN2의 잠재 공간을 미리 훈련된 얼굴 미용 평가 회귀 네트워크와 결합하여 이미지의 미적 요소를 최적화합니다. 입력 이미지로부터 잠재 공간의 포인트를 정교화한 후, Covariance Matrix Adaptation Evolution Strategy (CMA-ES) 알고리즘을 적용하여 최적의 잠재 포인트를 찾습니다. 이 방법은 VGG-16 모델을 이용해 입력 이미지와 생성된 이미지 사이의 지각적 유사성을 측정합니다.

- **Performance Highlights**: 새로운 접근 방식은 기존의 얼굴 미용 평가 모델을 능가하며, 자동으로 아름다움의 전체 패턴을 데이터를 통해 포착합니다. 이 연구는 자동화된 미적 향상에 대한 새로운 방향을 제시하고, 기존 방법과 보완적인 대안을 제공합니다. 우리의 모델은 특히 기존의 정적인 규칙 대신 데이터 기반으로 아름다움을 학습하는 데 중점을 둡니다.



### Multimodal Large Models Are Effective Action Anticipators (https://arxiv.org/abs/2501.00795)
- **What's New**: ActionLLM 프레임워크는 비디오 시퀀스를 연속적인 토큰으로 처리하여 LLMs(대형 언어 모델)를 사용해 미래의 행동을 예측하는 혁신적 접근 방식을 제공합니다. 이 모델은 기존 LLM 아키텍처를 단순화하여 복잡한 지침이나 중복 설명 없이 직접적인 행동 예측을 가능하게 합니다. 또한, Cross-Modality Interaction Block(CMIB)을 도입해 시각 및 텍스트 모달리티 간의 상호작용을 탐색하고 멀티모달 조정을 개선합니다.

- **Technical Details**: ActionLLM은 LLMs의 강력한 시퀀스 모델링 능력과 풍부한 상식 지식을 활용하여 장기 행동 예측을 수행합니다. 이 방법은 시각 정보를 LLM 프레임워크와 원활하게 융합하고, 출력 메커니즘을 단순화하여 예측 효율성을 극대화합니다. 또한, CMIB를 사용하여 각 모달리티의 특성을 탐색하고, 비주얼 및 텍스트 모달리티 간의 보완성을 촉진하는 방식을 적용합니다.

- **Performance Highlights**: 50 Salads 및 Breakfast 데이터셋에서의 평가 결과, 멀티모달 세부 조정이 LLM의 장기 행동 예측 성능을 향상시키는 것으로 나타났습니다. ActionLLM은 다양한 벤치마크 데이터셋에서 우수한 성능을 발휘하며, LLM을 행동 예측의 컨텍스트에서 탐색하는 유망한 방향을 제시합니다. 이러한 연구는 나중의 학문적 탐색을 위한 기초를 제공합니다.



### Beyond Words: AuralLLM and SignMST-C for Precise Sign Language Production and Bidirectional Accessibility (https://arxiv.org/abs/2501.00765)
- **What's New**: 이 논문에서는 청각 장애인을 위한 새로운 수화 생성 및 번역 시스템인 CNText2Sign와 CNSign 데이터셋을 소개합니다. 이 데이터셋은 중국어 자연어를 수화로 매핑하는 고품질 정보를 제공하며, 깊이 있는 수화 표현을 가능하게 합니다. 또한, AuraLLM 및 SignMST-C라는 두 가지 모델을 통해 수화의 정확성과 표현력을 향상시키고 있습니다.

- **Technical Details**: CNText2Sign 데이터셋은 15,000개의 자연어에서 수화 글로스를 매핑한 주석을 포함하고 있으며, OpenPose를 활용하여 키포인트를 추출하고 있습니다. 또한, SignMST-C는 빠른 동작 비디오에 대한 자기 감독 전처리를 활용하여 SLT 성능을 향상시키고, AuraLLM은 LoRA 및 RAG 기법을 통합하여 정확한 텍스트-수화 매핑을 가능하게 합니다.

- **Performance Highlights**: AuraLLM은 CNText2Sign 데이터셋에서 BLEU-4 점수 50.41을 기록하며 높은 정확도를 보여주고 있습니다. SignMST-C는 PHOENIX2014-T 벤치마크에서 BLEU-4 점수 31.03/32.08을 달성하며 새로운 최신 결과(SOTA)를 설정했습니다. 이러한 성과를 통해 새로운 데이터셋과 모델이 수화 생성 및 번역 분야에서 중요한 기여를 하고 있음을 입증하였습니다.



### Less is More: Token Context-aware Learning for Object Tracking (https://arxiv.org/abs/2501.00758)
Comments:
          Accepted by AAAI 2025

- **What's New**: 최근 여러 연구들은 객체 추적에 필요한 타겟 상태를 인식하기 위한 컨텍스트 정보를 활용하는 것이 중요하다는 것을 보여주었습니다. LMTrack라는 새로운 토큰 컨텍스트 인식 추적 파이프라인을 제안하여 효과적인 비주얼 추적을 위해 고품질의 참조 토큰을 자동으로 학습합니다. 이 방법은 참조 프레임 내 각 패치의 중요성을 분석하고, 중요 패치를 지속적으로 주목 및 업데이트하여 잡음을 줄이며 성능을 향상시킵니다.

- **Technical Details**: LMTrack는 Token Context Memory 모듈과 유효한 Unidirectional Token Attention 메커니즘으로 구성됩니다. Token Context Memory 모듈은 타겟의 고품질 시공간(spatio-temporal) 정보를 동적으로 수집하고 업데이트하여 참조 프레임에서 불필요한 배경 토큰을 제거합니다. Unidirectional Token Attention은 참조 토큰과 검색 프레임 간의 종속성을 설정하여 강력한 프레임 간 연결과 타겟 로컬라이제이션을 가능하게 합니다.

- **Performance Highlights**: LMTrack는 GOT-10K, TrackingNet, LaSOT 등 다수의 추적 벤치마크에서 최첨단(tracking state-of-the-art) 결과를 달성했습니다. 이 방법은 고품질의 토큰을 선택하고 업데이트하는 방식으로 추적 능력을 개선하며, 기존 방법들보다 더 나은 성능을 제공합니다. LMTrack의 접근 방식은 적은 양의 기준 정보로도 높은 정확도를 유지할 수 있는 점에서 기존의 방식들과 확연히 구별됩니다.



### Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation (https://arxiv.org/abs/2501.00752)
Comments:
          Association for the Advancement of Artificial Intelligence (AAAI) 2025

- **What's New**: 이번 논문에서는 Few-Shot Segmentation (FSS) 문제를 해결하기 위해 Foreground-Covering Prototype Generation and Matching이라는 새로운 방법을 제안합니다. 이 방법은 지원 이미지와 쿼리 이미지 간의 관계를 활용하여 보다 효과적으로 타겟 지역을 분할합니다. 특히, SAM (Segment Anything Model) 특징과 ResNet 특징을 결합하여 보다 신뢰할 수 있는 프로토타입을 생성하고, 이를 통해 시각적 참조 프롬프트를 생성합니다.

- **Technical Details**: 제안하는 방법은 SAM Image Encoder 기능을 활용하여 픽셀 집합을 집계하고, ResNet 기능을 사용하여 클래스 일관성을 보장합니다. 이를 통해 쿼리 및 지원 프로토타입을 구성하며, 이를 기반으로 주목 기반의 의사 마스크(attention-based pseudo-mask)를 사용하여 ResNet 기능에서 전경 정보에 집중하도록 유도합니다. 이러한 과정에서 프로토타입 간 비교를 통해 신뢰할 수 있는 시각적 참조 프롬프트를 생성하고 SAM Mask Decoder를 통해 객체 마스크를 생성합니다.

- **Performance Highlights**: 본 연구에서는 다양한 데이터셋에서 새로운 최첨단 성능을 달성하여 제안한 방법의 효과를 검증하였습니다. 특히, 프로토타입 간의 비교 방식이 기존 픽셀 기반 비교보다 전경 정보를 더욱 효과적으로 분리하고 강조함을 보여주었습니다. 이러한 결과는 FSS 태스크에서 프로토타입 생성 및 매칭의 중요성을 강조합니다.



### Towards End-to-End Neuromorphic Voxel-based 3D Object Reconstruction Without Physical Priors (https://arxiv.org/abs/2501.00741)
Comments:
          6 pages, 15 figures, 5 tables, submitted to ICME 2025

- **What's New**: 본 연구에서는 기존의 물리적 사전(physical priors)을 추정할 필요 없는 밀집 복셀(dense voxel) 3D 재구성을 위한 엔드 투 엔드(end-to-end) 방법을 제안합니다. 여기서 새로운 이벤트 표현(event representation)인 Sobel Event Frame을 통해 모서리 특징을 강화하고, 3D 특징 학습을 효과적으로 진행할 수 있게 하였습니다. 또한, 최적 이진화 임계값 선택 원칙(Optimal Binarization Threshold Selection Principle)을 제안하여 향후 연구에 대한 지침으로 활용할 수 있도록 하였습니다.

- **Technical Details**: 제안된 방법은 단일 신경 모양 카메라를 직접 이용하여 3D 재구성을 수행하며, 복잡한 이벤트-3D 파이프라인을 배제합니다. 이벤트 표현에는 이벤트의 좌표, 타임스탬프(timestamp), 그리고 밝기 변화의 극성(polity)을 포함해, 수집된 데이터의 전처리를 수행합니다. 이벤트 데이터는 타임스탬프 또는 이벤트 수를 기준으로 나누어 엑스 형태의 프레임으로 변환되며, 이를 통해 각 픽셀은 해당 시간 창에서 이벤트 발생 여부를 나타냅니다.

- **Performance Highlights**: 제안된 방법은 기존의 기준 방법(<baseline method>)에 비해 54.6%의 재구성 정확도 향상을 기록하였습니다. 이는 고객의 기대를 넘어선 성능을 나타내며, 새로운 방법론에 대한 검증의 일환으로 중요한 성과로 평가됩니다. 또한, 본 연구에서 제안된 각 모델은 실시간 솔루션(real-time solution)으로 발전할 가능성이 존재합니다.



### RORem: Training a Robust Object Remover with Human-in-the-Loop (https://arxiv.org/abs/2501.00740)
- **What's New**: 최근 논문에서는 객체 제거(Object Removal) 분야에서 기존 방법들이 가지는 문제점을 해결하기 위해 반자율 학습(semi-supervised learning) 전략을 제안합니다. 이 연구의 주요 목표는 고품질의 데이터 세트를 구축하여, Robust Object Remover (RORem)을 훈련하는 것입니다. 특히, 60,000개의 초기 훈련 쌍을 통해 시작한 후, 인간 피드백을 활용해 20만 개 이상의 고품질 객체 제거 쌍을 생성하였습니다.

- **Technical Details**: 연구에서 제시한 방법론은 Stable Diffusion XL(SDXL) 기반의 이미지 인페인팅 모델을 사용하여, 초기 모델이 50% 미만의 성공률을 기록했던 한계를 극복합니다. 이후, 인간 주도 평가를 통해 질 높은 객체 제거 쌍을 선별하고, 이 데이터를 바탕으로 판별기(discriminator)를 훈련하여 데이터 생성 과정을 자동화합니다. 최종적으로, 이 데이터 세트를 이용해 RORem을 미세 조정(fine-tuning)하여 신뢰성과 이미지 품질에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: RORem은 이전 방법들에 비해 객체 제거 성공률을 18% 이상 향상시켰고, 이미지 품질과 신뢰성 모두에서 우수한 성과를 보였습니다. 또한, 편집 속도(less than 1 second) 측면에서도 그 효율성을 입증하였습니다. 다양한 실험 결과는 RORem이 주관적 및 객관적 평가 모두에서 우수한 성능을 발휘한다는 것을 증명하고 있습니다.



### DDD: Discriminative Difficulty Distance for plant disease diagnosis (https://arxiv.org/abs/2501.00734)
Comments:
          8 pages, 2 figures, 3 tables. Accepted at 4th Annual AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE)

- **What's New**: 최근 식물 질병 진단 분야에서 머신러닝(ML)을 활용한 연구들은 적절하지 않은 데이터 파티셔닝으로 인한 과대평가 문제를 강조하고 있습니다. 본 연구는 훈련 및 테스트 데이터 간의 도메인 간격을 정량화하기 위해 Discriminative Difficulty Distance (DDD)라는 새로운 지표를 제안합니다. DDD는 훈련 데이터의 다양성을 식별하는 데 유용한 도구로 작용하며, 더 다양한 데이터 세트를 개발하는 데 기여할 수 있습니다.

- **Technical Details**: 연구에 사용된 데이터는 27개 도메인에서 수집된 244,063개의 식물 질병 이미지로, 4종의 작물과 34종의 질병을 포함하고 있습니다. DDD는 이미지 인코더를 통해 생성된 저차원 표현을 측정하여 데이터 간의 거리를 평가하는 지표로 활용되며, 이러한 거리 측정이 질병 분류기의 진단 난이도와 강한 상관관계를 갖는다고 설명됩니다. 모델의 성능은 ImageNet21K에서만 사전 훈련된 기본 인코더에 비해 0.106에서 0.485까지 상관성이 증가하였고, 최대 0.909에 도달했습니다.

- **Performance Highlights**: 본 연구는 훈련과 테스트 데이터 간의 도메인 갭을 정량화하는 DDD 지표의 유효성을 평가하였으며, 이를 통해 다양한 작물 이미지에 대한 강력한 진단 성능을 보여주었습니다. 식물 질병 진단 작업에서 DDD를 기반으로 한 접근 방식이 기존의 방법들보다 더 높은 정확도를 달성할 수 있음을 시사합니다. 향후 이러한 접근법은 식물 질병 진단을 위한 ML 모델의 로버스트성 향상에 중요한 기여를 할 것으로 기대됩니다.



### Everywhere Attack: Attacking Locally and Globally to Boost Targeted Transferability (https://arxiv.org/abs/2501.00707)
Comments:
          11 pages, 6 figures, 8 tables, accepted by 2025AAAI

- **What's New**: 본 논문에서는 주어진 이미지에 대한 targeted transferability를 강화하기 위한 새로운 방안인 'everywhere scheme'을 제안합니다. 이전의 연구에서는 이미지의 특정 높은 신뢰도 목표를 최적화하는데 초점을 맞췄다면, 본 방법은 피해 이미지의 모든 지역에서 다수의 목표를 동시에 공격하는 방식을 채택합니다. 이를 통해 다양한 모델에 대한 attention(어텐션) 불일치로 인한 transfer failure(전이 실패)를 감소시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 피해 이미지를 겹치지 않는 블록으로 분할하고 각 블록에 대해 targeted attack을 수행합니다. 이는 각 지역의 attention areas(어텐션 영역)에서 하나의 목표 물체가 피해 모델의 어텐션 지역에 포함될 가능성을 높입니다. 기존의 높은 신뢰도 목표를 설정하는 방식과 달리, 본 방법은 다양한 목표 객체를 포함시켜 transferability(전이 가능성)를 증대시키는 것을 목표로 합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 방법이 기존의 state-of-the-art targeted attack들을 개선함을 보여줍니다. 예를 들어, 널리 사용되는 Logit 공격의 전이 가능성이 28.8%에서 300%로 향상되었습니다. 또한 Google Cloud Vision과 같은 실제 플랫폼에서 출처 공격 예제가 더 나은 성과를 거두는 것을 검증하였습니다.



### Knowledge-Guided Prompt Learning for Deepfake Facial Image Detection (https://arxiv.org/abs/2501.00700)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 최근 생성적 모델들이 현실적인 사진 이미지 생성에서 탁월한 성능을 보이고 있으며, 이는 합성된 얼굴 이미지의 경우에도 마찬가지입니다. 본 연구에서는 딥페이크(fake) 얼굴 이미지 탐지를 위한 새로운 지식 기반 프롬프트 학습 방법을 제안합니다. 특히 대형 언어 모델에서 허위 관련 프롬프트를 검색하여 최적화 과정에 활용함으로써 탐지 성능이 크게 향상됨을 보여줍니다.

- **Technical Details**: 제안된 방법은 지식 기반 프롬프트 학습(KGP)과 테스트 시간 프롬프트 튜닝(TTP)으로 구성되어 있습니다. KGP는 대형 언어 모델에서 허위 이미지의 특성과 관련된 개념을 검색하여 학습하는 반면, TTP는 훈련 카테고리와 테스트 카테고리 간의 도메인 차이를 완화하는 데 중점을 둡니다. 이러한 접근은 레이블이 제공되지 않는 조건에서도 가능하며, 사전 지식을 적극 활용하여 학습의 효과를 극대화합니다.

- **Performance Highlights**: DeepFakeFaceForensics 데이터셋에서의 광범위한 실험 결과, 본 연구에서 제안한 방법은 최신 기술보다 현저한 성능 향상을 보였습니다. 이는 실제 상황에 쉽게 적용될 수 있는 탐지 성능을 대폭 개선하여, 딥페이크 이미지의 신뢰성 있는 검출을 가능하게 함을 의미합니다. 따라서 본 연구는 얼굴 이미지 합성 기술 발전에 따른 보안 위협에 효과적으로 대응할 수 있는 가능성을 보여줍니다.



### ICONS: Influence Consensus for Vision-Language Data Selection (https://arxiv.org/abs/2501.00654)
Comments:
          25 pages, 19 figures

- **What's New**: 이번 연구에서는 단순하고 효과적인 다중 작업 비전-언어 데이터 선택 방법인 ICONS(Influence CONsensus vision-language data Selection)를 제안합니다. 주로 사용할 대규모 데이터셋을 효과적으로 관리하기 위해 샘플의 교차 작업 영향을 고려하여 우수한 데이터를 선정하는 것이 중요한 과제입니다. ICONS는 효율적인 실험을 통해 모델의 성능을 극대화할 수 있는 간결한 훈련 데이터 세트를 제공합니다.

- **Technical Details**: ICONS는 두 단계로 구성된 데이터 선택 프레임워크를 사용합니다. 첫 번째 단계에서는 각 작업에 대한 영향 점수를 계산하고, 두 번째 단계에서는 투표 기반 집계를 통해 작업 간의 합의를 구축하여 데이터 샘플을 선택합니다. 이 과정에서 각 샘플이 여러 작업에서 일관되게 유용한지를 평가하여, 특정 작업에 국한되지 않고 널리 가치 있는 샘플을 발굴합니다.

- **Performance Highlights**: LLaVA-ICONS-133K 데이터셋은 전체 데이터의 20%로, 98.6%의 성능을 유지합니다. 또한, 무작위 선택된 데이터셋과 비교했을 때 2.8% 성능 향상을 보여주며, 작업 간의 높은 이전 가능성을 입증합니다. ICONS는 성능 기준을 초과하여 다중 작업에서 우수한 성능을 지속적으로 유지할 수 있는 데이터 샘플을 선택하는 효율적인 방법으로 자리잡고 있습니다.



### Taming Feed-forward Reconstruction Models as Latent Encoders for 3D Generative Models (https://arxiv.org/abs/2501.00651)
- **What's New**: 이번 연구는 기존의 feed-forward image-to-3D reconstruction 방법들이 3D generative model을 훈련하는 데 효과적인 latent encoder로 사용될 수 있음을 보여줍니다. 이를 통해 두 가지 패러다임 간의 간극을 연결하게 되었으며, 계산 비용이 많이 드는 encoder network 훈련 없이도 풍부한 3D latent feature를 확보할 수 있습니다. 또한, 높은 차원 latent 공간의 처리를 개선하기 위해 post-processing pipeline을 개발하였습니다.

- **Technical Details**: 연구에서 제안된 pipeline은 몇 가지 주요 요소로 구성됩니다. 첫째, feature standardization을 통해 latent distribution을 VAE와 유사한 속성으로 변환합니다. 둘째, spatial importance weighting을 적용하여 최종 렌더링에 가장 중요한 영역에 집중하면서 잡음을 억제합니다. 마지막으로, multi-stream transformer 구조인 TriFlow를 통해 이러한 고차원 feature들을 효율적으로 처리합니다.

- **Performance Highlights**: 이 연구의 실험 결과, 제안한 방법은 최신 3D 생성 방법들과 비교하여 무조건적 생성(unconditional generation)에서 동등하거나 더 좋은 성능을 발휘하며, 텍스트 조건부 생성(text-conditional generation)에서 더욱 뛰어난 결과를 보여줍니다. 이는 기존 reconstruction 모델들이 고품질 3D 생성을 위한 효과적인 latent encoder로 사용될 수 있다는 것을 뒷받침하며, 확장 가능한 3D 콘텐츠 생성에 기여할 수 있음을 입증합니다.



### SoundBrush: Sound as a Brush for Visual Scene Editing (https://arxiv.org/abs/2501.00645)
Comments:
          AAAI 2025

- **What's New**: SoundBrush는 소리를 브러쉬처럼 사용하여 시각적 장면을 수정하고 조작하는 모델입니다. 이 모델은 Latent Diffusion Model (LDM)의 생성 능력을 확장하여 오디오 정보를 포함시킵니다. 다양한 기존 이미지 편집 작업에서 영감을 받아, 우리는 이를 감독 학습 문제로 설정하고 학습을 위한 사운드-페어드 비주얼 씬 데이터 세트를 구축했습니다.

- **Technical Details**: SoundBrush는 LDM의 텍스트 토큰 공간을 다양하고 풍부한 청각 특성으로 증강하며, 사운드를 이러한 토큰으로 변환하는 매핑 네트워크를 설계합니다. 이는 기존의 이미지 편집 도구, 예를 들어 음원 위치 추적(sound source localization) 및 이미지 인페인팅(image inpainting)과 결합하여 데이터 세트를 생성합니다. SoundBrush는 다양한 자연 소리 신호에 의해 시각적 장면을 편집할 수 있는 능력을 배웁니다.

- **Performance Highlights**: 기존 소리 기반 시각 장면 편집 모델과 비교하여, SoundBrush는 소리의 의미를 반영하여 사운드 객체를 삽입하고 전체 경관을 조정하는 능력이 있습니다. 또한 새로운 뷰 합성 방법과 통합하여, SoundBrush의 프레임워크는 3D 장면 편집도 지원합니다. 이러한 기능은 SoundBrush가 소리를 사용하여 시각적 장면을 조작하는 데 있어 많은 진전을 이룰 수 있음을 보여줍니다.



### Flash-Split: 2D Reflection Removal with Flash Cues and Latent Diffusion Separation (https://arxiv.org/abs/2501.00637)
- **What's New**: 본 논문에서는 Flash-Split이라는 새로운 프레임워크를 소개하여, 저조도 환경에서의 카메라 플래시를 활용하여 전송광(transmitted light)과 반사광(reflected light)을 효과적으로 분리하는 방법을 제안합니다. 이 방법은 두 단계로 이루어져 있으며, 지연(latent) 공간에서의 반사 분리 및 고해상도 디코딩 과정을 통해 밀집된 정보 압축을 활용합니다. 기존의 플래시/논플래시 이미지 정렬 문제를 해결하여 불일치한 이미지 쌍을 사용하여 강력한 성능을 발휘합니다.

- **Technical Details**: Flash-Split은 두 개의 단계로 구성된 모델로, 첫 번째 단계에서는 VAE 인코더를 사용하여 전송광과 반사광의 지연 표현(latent representations)을 구분하는 이중 가지 확산 모델(dual-branch diffusion model)을 적용합니다. 두 번째 단계에서는 원본 이미지 정보를 활용하여 고해상도 디코딩을 수행하며, 이를 통해 세부 정보의 유지를 꾀하고 최종 이미지의 충실도를 높입니다. 이 과정을 통해 사용자는 미조정된 플래시/논플래시 이미지 페어에서도 효과적으로 반사광을 분리할 수 있습니다.

- **Performance Highlights**: Flash-Split은 실제 환경에서 다양한 장면을 테스트한 결과, 반사 분리 성능이 최신 기술(state-of-the-art)을 초과함을 보여주었습니다. 이 방법은 기존의 베이스라인(baseline) 방법들, 특히 다른 플래시/논플래시 기반 접근법들보다 우수한 성능을 발휘했습니다. 특히 강한 반사가 있는 복잡한 장면에서도 본 연구의 방법이 효과적임을 입증했습니다.



### Gaussian Building Mesh (GBM): Extract a Building's 3D Mesh with Google Earth and Gaussian Splatting (https://arxiv.org/abs/2501.00625)
- **What's New**: 이번 연구에서는 최근 오픈 소스으로 공개된 pre-trained 모델인 SAM2와 GroundingDINO를 바탕으로 다중 뷰 2D 이미지를 이용하여 물체 세분화를 수행하는 새로운 방법을 제안하였습니다. 사용자는 레이블이 붙은 훈련 데이터셋 없이도 텍스트 기반이나 클릭 기반 프롬프트를 사용하여 관심 있는 객체를 세분화할 수 있습니다. 또한, Gaussian Splatting 기법을 통해 2D 이미지를 기반으로 장면의 3D 표현을 학습할 수 있게 되었습니다. 이와 같은 기술적 진보는 3D 건물 메쉬 추출에 대한 새로운 가능성을 열어주고 있습니다.

- **Technical Details**: 제안된 3D 건물 메쉬 추출 파이프라인은 Google Earth Studio, SAM2, Gaussian Splatting 등을 결합하여 사용합니다. 이 시스템은 건물의 이름, 주소 또는 지리적 좌표를 기반으로 3D 메쉬를 자동으로 생성합니다. 또한, GroundingDINO와 SAM2를 결합하여 텍스트 또는 클릭 기반의 입력으로 일관된 3D 마스킹을 수행하며, 형태학적 연산과 Ramer-Douglas-Peucker 알고리즘을 사용하여 마스크를 정제하는 방법을 추가했습니다. 개선된 2D Gaussian Splatting을 활용하여 관심 대상 건물의 3D 컬러 메쉬를 생성합니다.

- **Performance Highlights**: 본 연구에서 제시된 파이프라인은 기존 GS2Mesh보다 더 높은 정확도와 효율성을 자랑합니다. 기존 GS2Mesh가 겪었던 해결되지 않은 마스크 생성 문제를 극복하고, 배경에 대한 의존 없이 빠르고 정확하게 3D 메쉬를 생성할 수 있습니다. 이를 통해 Gaussian Splatting의 교육 속도가 약 5배 향상되었으며, 마스크 재프롬프트 및 정제 방법을 통해 추출 정확도가 크게 증가했습니다. 이러한 성과는 건물 모델링과 같은 다양한 응용 분야에 적용 가능할 것으로 기대됩니다.



### A Study on Context Length and Efficient Transformers for Biomedical Image Analysis (https://arxiv.org/abs/2501.00619)
Comments:
          Published at ML4H 2024

- **What's New**: 이번 연구에서는 생물 의학 이미지 분석에서 문맥 길이(context length)의 영향을 조사하고, 최근 제안된 장문 모델(long-context models)의 성능을 평가합니다. 생물 의학 이미징 데이터셋을 선별하고 분석함으로써, 다양한 세그멘테이션(segmentation), 노이즈 제거(denoising), 분류(classification) 작업에서의 효율성을 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 비전 트랜스포머(Vision Transformer)와 스윈 트랜스포머(Swin Transformer)를 활용하여 패치 크기(patch size)와 주의 윈도우 크기(attention window size)를 변동시키며 네트워크 성능을 분석합니다. 특히 픽셀 수준의 예측 과제에서 문맥 길이와 성능 간의 강한 관계를 발견하였습니다. 이를 통해, 문맥 길이가 길어질수록 성능이 크게 향상되는 경향을 확인했습니다.

- **Performance Highlights**: 최근의 장문 모델들은 생물 의학 이미지에 대한 효율성을 개선하면서도 유사한 성능을 유지함을 보여주었습니다. 그러나 몇몇 영역에서는 여전히 성능 차이가 존재함을 강조합니다. 이 연구는 생물 의학 이미징에서 장문 모델을 사용할 때의 잠재력 및 과제를 강조하며, 향후 연구의 방향성을 제시합니다.



### DiC: Rethinking Conv3x3 Designs in Diffusion Models (https://arxiv.org/abs/2501.00603)
Comments:
          11 pages, 6 figures

- **What's New**: 본 논문은 기존의 Self-Attention 기반의 Diffusion 모델을 대신하여 3x3 Convolution을 활용한 새로운 Diffusion CNN (DiC) 아키텍처를 제안합니다. 이 아키텍처는 빠른 처리 속도와 뛰어난 성능을 자랑하며, 전통적인 Transformer 기반 Diffusion 모델보다 우수한 성능을 나타냅니다. 특히, Encoder-Decoder Hourglass 디자인을 채택하여 Conv3x3의 성능을 극대화하고, sparse skip connections 도입을 통해 효율성을 높였습니다.

- **Technical Details**: DiC 모델은 3x3 Convolution을 주요 구성 요소로 사용하며, Stride-1 Convolution을 통해 빠른 속도를 구현합니다. Winograd 알고리즘을 활용한 연산 최적화로 3x3 커널의 계산을 빠르게 수행합니다. 또한, stage-specific embeddings 및 mid-block condition injection을 통해 조건부 개선을 이루었으며, 이를 통해 모델의 효율성과 성능을 향상시켰습니다.

- **Performance Highlights**: DiC는 다양한 스케일과 환경에서 실험을 통해 기존의 Diffusion Transformers에 비해 상당한 성능 우위를 보여주었으며, 빠른 처리 속도를 유지하면서 경쟁력 있는 결과를 도출하였습니다. 이 모델은 특히 실시간 및 대규모 응용에 적합하여, 자원 제약이 있는 환경에서도 우수한 성능을 발휘할 수 있습니다.



### STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes (https://arxiv.org/abs/2501.00602)
Comments:
          Project page at: this https URL

- **What's New**: 이번 논문에서 제안하는 STORM은 동적 장면을 재구성하기 위한 공간-시간 복원 모델입니다. 기존의 방법들이 장면별 최적화와 밀접한 관찰에 의존하는 데 비해, STORM은 데이터 기반 Transformer 아키텍처를 활용하여 단일 전방 패스를 통해 동적 3D 장면 표현을 직접적으로 유추합니다. 이 방법은 강력한 움직임 감독 없이도 고품질 재구성을 가능하게 합니다.

- **Technical Details**: STORM은 3D Gaussian과 그 속도를 매개변수화하여 동적 장면을 효율적으로 재구성합니다. 이를 위해 모든 프레임에서 3D Gaussian을 집계하여 목표 시점으로 변환하고, 이를 통해 '아모달'(amodal) 재구성을 수행합니다. 또한, Motion tokens을 도입하여 각 3D Gaussian의 움직임을 캡처하고, 이를 통해 동적 인스턴스를 분리하는 데 도움을 줍니다.

- **Performance Highlights**: STORM은 Waymo Open, NuScenes, Argoverse2 데이터셋에서 광범위한 실험을 통해 동적 장면 재구성의 정확성을 입증하였습니다. 이 모델은 기존 장면별 최적화 방법보다 4.3에서 6.6 PSNR 향상을 보이며, 대규모 야외 장면을 200ms 내에 재구성할 수 있습니다. 또한, STORM은 실시간 렌더링을 지원하며 3D EPE를 0.422m 개선하였습니다.



### DreamDrive: Generative 4D Scene Modeling from Street View Images (https://arxiv.org/abs/2501.00601)
- **What's New**: 이 논문에서는 DreamDrive라는 새로운 방법론을 도입하여 자율주행을 위한 4D 공간-시간 장면 생성을 다루고 있습니다. 기존의 생성 및 재구성 기반 방법들이 가지는 한계를 극복하고, 3D 일관성을 유지하면서도 일반화 가능한 4D 주행 장면을 합성할 수 있도록 설계되었습니다. 특히, 비디오 확산 모델(video diffusion model)과 하이브리드 가우시안 표현(hybrid Gaussian representation)을 결합하여 더 높은 품질의 장면 생성을 가능하게 합니다.

- **Technical Details**: DreamDrive는 에고 차량의 주행 궤적을 이용해 시각적 참조를 생성하고, 이를 4D로 올리기 위해 고안된 새로운 하이브리드 가우시안 표현을 사용합니다. 이 방법론은 정적 배경을 모델링하기 위해 시간 독립적인 가우시안을, 동적 객체를 모델링하기 위해 시간 의존적인 가우시안을 활용하여 4D 장면을 구성합니다. 이러한 접근을 통해 이미지 감독(image supervision)만으로도 고품질의 3D 일관성을 유지하는 주행 비디오를 생성할 수 있습니다.

- **Performance Highlights**: DreamDrive는 nuScenes 데이터셋과 실제 주행 시나리오에서 검증되었으며, 기존 방법보다 30% 향상된 시각적 품질로 3D 일관성이 확보된 주행 비디오를 생성합니다. 또한 이 방법은 자율주행의 인식(perception) 및 계획(planning) 작업에서도 효과적으로 적용될 수 있음을 입증했습니다. 최종적으로, 이 연구는 다양한 주행 시나리오에서의 보편성을 높이는 데 기여합니다.



### VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM (https://arxiv.org/abs/2501.00599)
Comments:
          17 pages, 12 figures, technical report

- **What's New**: 이번 논문에서는 Video LLM에 대한 VideoRefer Suite를 소개하여 비디오의 세부적인 공간-시간적 이해를 향상시키는 방안을 제안합니다. 특히, VideoRefer Suite는 데이터셋, 모델, 벤치마크의 세 가지 중요한 측면을 통해 개발되었습니다. 이 연구는 고품질 객체 수준 비디오 지침 데이터와 포괄적인 벤치마크의 부족이라는 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: VideoRefer Suite는 다중 에이전트 데이터 엔진을 통해 대규모 고품질 객체 수준 비디오 지침 데이터셋인 VideoRefer-700K를 생성합니다. 이 데이터셋은 비디오의 각 객체에 대해 다양한 수준의 질문과 답변 쌍을 포함하고 있습니다. 모델 측면에서, VideoRefer는 공간-시간적 객체 인코더 및 적응형 시간 토큰 병합 모듈을 활용하여 정밀한 객체 이해를 지원합니다. 마지막으로, VideoRefer-Bench를 통해 모델 성능을 종합적으로 평가합니다.

- **Performance Highlights**: VideoRefer 모델은 비디오 참조 벤치마크에서 뛰어난 성능을 보여줍니다. 이 모델은 기본 비디오 객체 참조, 복잡한 객체 간의 관계 분석, 객체 검색 작업 등이 가능하며, 사용자 상호작용을 유지하는 고급 기능을 제공합니다. 다양한 시간대와 객체를 통해 모델의 포괄적인 캡션 작성 및 추론 능력도 평가됩니다.



### Sidewalk Hazard Detection Using Variational Autoencoder and One-Class SVM (https://arxiv.org/abs/2501.00585)
Comments:
          7 pages

- **What's New**: 이 논문은 인도 안전 내비게이션을 위한 새로운 시스템을 소개합니다. 이 시스템은 Variational Autoencoder (VAE)와 One-Class Support Vector Machine (OCSVM)을 결합한 하이브리드 접근 방식을 활용하여 보행에 위험을 초래할 수 있는 이상 징후를 탐지합니다. 기존의 방법들에 비해 더 효과적인 이상 탐지를 위한 모델링을 수행합니다.

- **Technical Details**: 제안된 시스템은 15,000개의 학습 프레임과 5,000개의 테스트 프레임으로 구성된 데이터셋을 사용하여 다양한 인도 시나리오를 캡처했습니다. VAE는 프레임 내에서의 재구성 메커니즘을 사용하여 이상을 탐지하며, 재구성이 좋지 않을 경우 OCSVM을 통해 해당 이상이 위험한지 여부를 확인합니다. 이 과정에서 VAE는 AUC 0.94의 높은 성능을 기록하여 보행 위험성을 효과적으로 구별합니다.

- **Performance Highlights**: 시스템은 91.4%의 정확도로 위험한 시나리오와 비위험 시나리오를 구별할 수 있습니다. 이러한 결과는 제안된 시스템이 불확실한 환경에서의 위험 탐지에 강력한 솔루션을 제공함을 시사합니다. 따라서 이 시스템은 새로운 안전 내비게이션 수단으로서의 가능성을 지니고 있습니다.



### Online Video Understanding: A Comprehensive Benchmark and Memory-Augmented Method (https://arxiv.org/abs/2501.00584)
- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 최신 발전을 바탕으로 온라인 비디오 이해에 대한 도전 과제를 다루고 있습니다. 새로운 평가 기준인 OVBench를 소개하며, 이는 모델의 온라인 비디오 맥락에서 인지, 기억 및 추론 능력을 평가하기 위해 설계된 질문-응답 기준입니다. 또한, 새로운 Pyramid Memory Bank (PMB)를 통해 비디오 스트림의 주요 시공간 정보를 효과적으로 유지하고, 오프라인-온라인 학습 패러다임을 제안하여 실시간 비디오 데이터에 적합한 모델을 개발했습니다.

- **Technical Details**: OVBench는 과거, 현재 및 미래 세 가지 시간적 맥락을 포함하는 6개 핵심 작업 유형으로 구성되며, 총 16개 하위 작업이 다양한 데이터셋에서 형성됩니다. 이 연구에서 제안하는 Pyramid Memory Bank (PMB)는 비디오 스트림 속에서 세밀한 공간적 정보와 장기적 시간적 의존성을 효율적으로 유지하는 구조로 설계되었습니다. 또한, 이 논문에서는 오프라인-온라인 학습 전략을 통해 모델이 지속적으로 개선될 수 있는 방법을 제시하고 있습니다.

- **Performance Highlights**: 새롭게 개발된 모델 VideoChat-Online은 OVBench에서 기존 오프라인 모델 Qwen2-VL을 4.19% 초과 성능을 보였으며, 온라인 비디오 MLLM Flash-Vstream보다는 23.7% 향상된 결과를 나타냈습니다. 이 모델은 오프라인 비디오 벤치마크에서도 뛰어난 성능을 발휘하여, 온라인 및 오프라인 비디오 이해 전반에서 강력함을 입증하고 있습니다. 모든 모델과 데이터는 공개될 예정이며, 향후 연구가 온라인 비디오 이해에 대한 통찰을 제공할 수 있기를 기대합니다.



### VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling (https://arxiv.org/abs/2501.00574)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(MLLM)이 긴 비디오를 효과적으로 처리할 수 있는 새로운 방법을 제안합니다. 특히, 계층적 비주얼 토큰 압축 방법(HiCo)과 비디오 처리에 최적화된 시스템인 VideoChat-Flash를 도입하여 긴 시퀀스의 맥락을 효율적으로 모델링합니다. HiCo는 비디오의 시각적 정보를 압축하여 계산 자원을 크게 줄이면서도 필수 세부정보를 보존하는 방식입니다.

- **Technical Details**: HiCo는 긴 비디오의 맥락을 클립 수준에서 비디오 수준으로 압축하여, 시각 토큰을 계층적으로 처리하는 방식입니다. VideoChat-Flash는 다단계 단기-장기 학습 전략을 사용하며, 300,000시간의 현실적인 긴 비디오 데이터셋인 LongVid를 포함하여 훈련에 필요한 데이터 수집 방식을 제시합니다. 본 시스템은 모델 훈련을 위한 효율적인 분산 시스템을 구축하여 높은 수준의 시퀀스 병렬성을 지원합니다.

- **Performance Highlights**: 비디오처리 효율성을 검증하기 위해 개발된 VideoChat-Flash는 공개된 MLLM 중에서 10,000 프레임에서 99.1%의 정확성을 보였습니다. 7B 모델 규모에 맞춘 정량적 평가에서는 최근의 주류 긴 비디오 벤치마크에서 뛰어난 성능을 기록했습니다. 이러한 결과는 HiCo의 높은 충실도를 기반으로 한 효율적인 데이터 처리가 가능하다는 것을 입증합니다.



### Probing Visual Language Priors in VLMs (https://arxiv.org/abs/2501.00569)
- **What's New**: 최근 Vision-Language Models (VLMs)의 발전에도 불구하고, 많은 모델들이 실제 시각적 추론보다는 훈련 데이터에 존재하는 시각적 언어 선입견에 과도하게 의존하고 있습니다. 이를 검토하기 위해, 우리는 비주얼 질문 응답(Visual Question Answering, VQA) 기준인 ViLP를 소개합니다. ViLP는 각 질문에 세 가지 답변 및 세 가지 이미지를 매칭하여, 실제 시각적 추론이 필요한 이미지를 강조합니다.

- **Technical Details**: ViLP는 텍스타 관련 정보에 의존하여 답변할 수 있는 이미지와 시각적 reasoning이 필요한 이미지를 조합하여, 변형 가능한 요소들(예: texture, shape, conceptual combinations 등)을 고려한 이미지를 제공합니다. 또한, 우리는 모델이 새로운 VQA 쌍 및 이미지를 생성하고 픽셀 및 의미적 부패를 적용하는 자가 개선 프레임워크를 제안합니다. 이 프레임워크는 VLM이 실제 시각적 입력에 더 집중하게 하고, LLaVA-v1.5 및 Cambrian과 같은 오픈 소스 VLM의 성능 향상에 효과적임을 입증합니다.

- **Performance Highlights**: 인간은 거의 완벽한 정확도로 ViLP를 수행하지만, 현대의 VLM들은 GPT-4와 같은 모델이 겨우 66.17%의 정확도를 기록하는 등 성능이 떨어집니다. 이러한 모델의 성능 개선을 위해, 우리의 방법론은 RLHF(급강하 학습 강화)를 활용하여 보상 모델을 통해 피드백을 제공하며, 이를 통해 VLM의 전반적인 품질을 높입니다. 우리의 접근법은 RL_FINE_TUNNING 을 통해 최적의 정책을 찾고, 최대 우도 최적화 문제를 해결하며, 이러한 최적화 과정은 VLM의 언어 생성 성능을 극대화하는 데 기여합니다.



### Exploiting Boundary Loss for the Hierarchical Panoptic Segmentation of Plants and Leaves (https://arxiv.org/abs/2501.00527)
Comments:
          Presented at the 9th Workshop for Computer Vision in Plant Phenotyping and Agriculture (CVPPA) 2024 at the European Conference of Computer Vision (ECCV) 2024. arXiv admin note: text overlap with arXiv:2310.06582

- **What's New**: 이번 논문에서는 정밀 농업을 위한 새로운 계층적 팬옵틱 세분화(hierarchical panoptic segmentation) 방법을 제안합니다. 이 방법은 이미지 내에서 식물 성장의 지표가 되는 잎 수를 산출하고 잡초를 식별하는 작업을 동시에 수행합니다. 특히, 작은 객체인 잎과 잡초를 개선하기 위해 focal loss와 boundary loss를 도입하였으며, 그것이 경쟁력 있는 성능을 달성하는 데 기여했습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 기존의 Mask2Former 아키텍처를 기반으로 하며, 식물 마스크와 잎 마스크를 동시에 생성할 수 있도록 추가적인 transformer decoder를 통합했습니다. 이를 통해 잎과 잡초의 픽셀 수준 세분화를 수행하며, 다중 스케일 변형 주의 (MSDeformAttn) 기법을 사용하여 특징 피라미드를 구성합니다. 이 과정에서, self-attention과 cross-attention 메커니즘을 포함한 transformer 디코더가 활용되어 성능을 극대화합니다.

- **Performance Highlights**: 제안된 방법으로 얻어진 PQ+(Panoptic Quality) 점수는 81.89로, 표준 훈련 세트에서 우수한 성능을 보였습니다. 또한, 이 접근법은 잎의 수를 정확하게 세는 데 있어서도 개선된 정확도를 나타냈습니다. 최종적으로, 우리의 방법은 현대의 최첨단 기술들과 비교할 때 경쟁력 있는 성과를 달성하며, 실용적인 농업 응용 분야에서도 기여할 수 있는 가능성을 보여줍니다.



### Is Segment Anything Model 2 All You Need for Surgery Video Segmentation? A Systematic Evaluation (https://arxiv.org/abs/2501.00525)
- **What's New**: 본 논문은 SAM2 모델을 활용한 제로샷 수술 비디오 세그멘테이션에 대한 체계적인 평가를 수행하였습니다. 기존의 수술 데이터 주석 부재 문제를 해결하기 위해, 자연 비디오에서 훈련된 SAM2 모델을 통해 비디오 세그멘테이션의 가능성을 탐색하고 있습니다. 다양한 프롬프트 전략과 각기 다른 세그멘테이션 작업의 강건성을 포함한 실험을 진행하였습니다.

- **Technical Details**: SAM2는 비디오 데이터에 대한 제로샷 세그멘테이션 가능성을 확장한 모델로, 35.5백만 개 마스크가 포함된 SA-V 데이터셋에서 훈련되었습니다. 모델은 Hiera 인코더와 메모리 어텐션 모듈을 활용하여 시간적 정보를 처리하며, 개별 프레임의 특징을 효과적으로 유지합니다. 프롬프트 메커니즘은 포인트, 바운딩 박스, 픽셀 마스크 등 여러 형태를 지원하여 객체 세그멘테이션을 정밀하게 수행할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, SAM2는 다양한 수술 유형에 대한 비디오 세그멘테이션 작업에서 뛰어난 제로샷 성능을 나타내었습니다. 9개의 데이터셋을 대상으로 한 성능 평가에서, 다양한 수술 도구 및 해부학 구조의 정확한 세그멘테이션을 통해 수술 환경의 동적 재구성을 지원할 수 있는 가능성을 보여주었습니다. 해당 성능은 리소스 소모가 큰 픽셀 수준 마스크 수집 대신 제로샷 접근 방식을 통해 달성되었습니다.



### Innovative Silicosis and Pneumonia Classification: Leveraging Graph Transformer Post-hoc Modeling and Ensemble Techniques (https://arxiv.org/abs/2501.00520)
- **What's New**: 이 논문은 규폐증(Silicosis)과 관련된 폐 염증의 분류 및 탐지에 관한 종합적인 연구 결과를 제시합니다. 주요 기여로는 규폐증과 폐렴 연구 커뮤니티를 위해 맞춤화된 새롭게 구축한 흉부 X-레이(CXR) 데이터셋 SVBCX의 생성과 전통적인 딥 뉴럴 네트워크 모듈과 그래프 변환기 네트워크를 통합한 새로운 딥러닝 아키텍처의 제안이 있습니다.

- **Technical Details**: 제안된 아키텍처는 다양한 모델 아키텍처의 강점을 통합하는 앙상블 접근 방식을 탐구하며, 손실 함수로는 데이터 클래스 간의 균일한 학습을 보장하기 위해 Balanced Cross-Entropy (BalCE)를 사용합니다. 이를 통해 규폐증 및 폐렴 분류의 정확도와 신뢰성을 향상시키고, 미세한 차이를 구별할 수 있는 모델의 능력을 향상시키는 것을 목표로 하였습니다.

- **Performance Highlights**: SVBCX 데이터셋에서 실험한 결과는 특히 인상적이며, 기존의 기준 모델과 비교하여 상당한 향상을 보여주었습니다. 제안된 모델은 매크로 F1 점수 0.9749와 각 클래스에 대해 0.99를 초과하는 AUC ROC 점수를 기록하여 폐 염증 분류의 정확성과 강건성에서 높은 효율성을 입증합니다.



### Fine-grained Video-Text Retrieval: A New Benchmark and Method (https://arxiv.org/abs/2501.00513)
- **What's New**: 이번 논문에서 제시하는 FIBER는 비디오-언어 검색(video-language retrieval) 모델의 세밀한 성능 평가를 가능하게 하는 새로운 벤치마크입니다. 기존의 MSRVTT 및 MSVD와 같은 비디오 검색 벤치마크는 세밀한 주석 부족으로 인해 효과적으로 성능을 평가하지 못했습니다. FIBER는 FineAction 데이터세트에서 소싱된 1,000개의 비디오와 함께 상세한 사람 주석(spatial annotations) 및 시간 주석(temporal annotations)을 제공합니다.

- **Technical Details**: FIBER 벤치마크는 비디오 검색 과제에서 비디오-언어 모델의 공간적 및 시간적 편향(spatial and temporal bias)을 독립적으로 평가할 수 있게 합니다. 이 연구는 Multimodal Large Language Models (MLLMs)의 세밀한 비디오-언어 이해를 위한 텍스트 임베딩 방법(text embedding)을 활용하였습니다. 이를 통해 비디오-언어 모델은 더욱 효과적으로 비디오를 검색할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 Video Large Language Encoder (VLLE)는 전통적인 벤치마크에서 CLIP 기반 모델과 비슷한 성능을 보였습니다. 더욱 놀라운 점은 VLLE가 더 낮은 공간-시간 편향(spatial-temporal bias)으로 세밀한 표현력이 더 뛰어난 성능을 발휘했다는 것입니다. 이는 VLLE가 비디오-언어 검색 작업에서 우수한 능력을 갖춘 모델임을 시사합니다.



### Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning (https://arxiv.org/abs/2501.00437)
Comments:
          ECCV 2024

- **What's New**: 최근 제로샷 이미지 캡셔닝(zero-shot image captioning)은 텍스트 데이터만으로 훈련을 진행하며 주목받고 있습니다. 본 논문에서는 사전 훈련된 텍스트-이미지 확산 모델(text-to-image diffusion model)을 활용해 합성 이미지-캡션 쌍을 생성하는 방법을 제안합니다. 그러나 합성 이미지의 결함 있는 세부사항이 시멘틱 불일치를 초래하는 문제를 다루기 위해, 새로운 패치 기반 크로스 모달 기능 혼합(Patch-wise Cross-modal feature Mix-up) 메커니즘을 제안합니다.

- **Technical Details**: 제안된 PCM-Net은 이미지의 시각적 개념을 탐지하고 해당 개념의 텍스트 피처와 합성 이미지의 패치 기반 시각적 피처를 선택적으로 융합합니다. 이를 통해 결함이 적고, 보다 정교한 피처 맵(feature map)을 생성할 수 있습니다. 또한, CLIP 가중치를 적용한 크로스 엔트로피 손실(clipped weighted cross-entropy loss)을 새롭게 도입하여 노이즈가 있는 합성 데이터를 활용한 훈련의 안정성을 향상시킵니다.

- **Performance Highlights**: MSCOCO 및 Flickr30k 데이터셋에서 실시된 광범위한 실험 결과, PCM-Net은 기존 VLMs 기반 접근 방식에 비해 우수한 성능을 보였습니다. 특히, PCM-Net은 도메인 내(in-domain) 및 교차 도메인(cross-domain) 제로샷 이미지 캡셔닝에서 1위를 기록했습니다. 이 연구 결과는 더 정밀한 시각-언어 정렬(visual-semantic alignment)을 통해 캡션 생성의 질을 높일 수 있는 가능성을 보여줍니다.



### OV-HHIR: Open Vocabulary Human Interaction Recognition Using Cross-modal Integration of Large Language Models (https://arxiv.org/abs/2501.00432)
Comments:
          Accepted in IEEE ICASSP 2025

- **What's New**: 이 논문에서는 다채롭고 예측 불가능한 인간 간 상호작용을 인식하기 위한 열린 어휘(open vocabulary) 기반의 인간 간 상호작용 인식(framework)을 제안합니다. 전통적인 활동 인식 시스템의 한계를 극복하기 위해 대규모 언어 모델(LLM)을 활용하여 훈련 데이터에 구애받지 않는 개방형 설명을 생성합니다. 이 연구는 기존의 상호작용 데이터셋을 통합하여 포괄적인 대규모 인간 간 상호작용 데이터셋을 만들었습니다.

- **Technical Details**: 제안된 방법은 기존의 여러 복수 인원 상호작용 데이터셋을 집계하여 모델을 훈련시키며, 구체적인 상호작용을 더욱 정교하게 설명하기 위해 하드 레이블을 소프트 레이블로 변환합니다. 비디오를 분할하여 각 개인의 행위, 배경을 식별하고, 각 인물별로 비전-언어 브랜치를 도입하여 복잡한 상호작용을 분석합니다. 최종적으로 비디오 임베딩과 텍스트 임베딩을 공유 공간에 정렬하여 열린 어휘적 설명을 생성하는 구조입니다.

- **Performance Highlights**: OV-HHIR 모델은 다양한 데이터셋에서 cosine similarity 점수 0.63을 기록하며, 기존의 고정 어휘 기반 분류 시스템 및 비디오 이해를 위한 크로스 모달 언어 모델보다 우수한 성능을 보였습니다. 이 모델은 훈련 중 발견되지 않은 상호작용을 인식하는 능력도 갖추고 있어, 복잡한 동적 환경에서도 강력한 인식 성능을 발휘합니다. 실험적 결과는 제안된 모델이 실제 APPLICATION에 효율적이라는 것을 보여줍니다.



### B2Net: Camouflaged Object Detection via Boundary Aware and Boundary Fusion (https://arxiv.org/abs/2501.00426)
- **What's New**: 본 논문에서는 Camouflaged Object Detection (COD)의 문제를 해결하기 위해 새로운 네트워크인 B2Net을 제안합니다. 기존의 경계 기반(COD) 알고리즘이 초기 단계에서 객체 경계를 생성할 때 발생하는 부정확한 엣지 프라이어(Edge Priors)로 인해 노이즈가 발생하는 문제점을 다룹니다. 이 네트워크는 경계 인식 모듈을 여러 단계에서 재사용하여 얻은 경계의 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: B2Net은 Residual Feature Enhanced Module (RFEM)을 통해 더 차별화된(feature representation) 특성 표현을 통합하여 검출 정확성과 신뢰성을 높입니다. 그 후 Boundary Aware Module (BAM)을 도입하여 저수준(low-level) 특성으로부터의 공간 정보와 고수준(high-level) 특성으로부터의 의미론적 정보를 통합하여 엣지 큐(edge cues)를 두 번 탐색합니다. 마지막으로, Cross-scale Boundary Fusion Module (CBFM)을 설계하여 서로 다른 스케일(scale)에서 정보를 통합하여 경계 정보와 객체 특성을 머즈하여 종합적(feature representation)인 특성 표현을 생성합니다.

- **Performance Highlights**: 세 가지 도전적인 벤치마크 데이터셋에서 수행한 광범위한 실험 결과, 제안된 B2Net은 널리 사용되는 평가 메트릭(metrics)에서 15개의 최첨단 방법(state-of-art methods)을 초월하는 성능을 보여줍니다. 이러한 결과들은 B2Net이 COD 문제에 있어 강력한 솔루션임을 시사합니다. 향후 이 연구의 코드는 공개될 예정입니다.



### Token Pruning for Caching Better: 9 Times Acceleration on Stable Diffusion for Fr (https://arxiv.org/abs/2501.00375)
- **What's New**: 이 논문에서는 dynamics-aware token pruning (DaTo) 접근법을 제안하여 Stable Diffusion의 feature caching의 한계를 해결합니다. 기존의 feature caching은 이웃 타임스텝에서의 특징들이 유사해지는 문제를 일으켜, 생성 이미지의 품질을 저하시키고 있습니다. DaTo는 저조한 동적인 특성을 가진 토큰을 선택적으로 제거하고, 고동적인 토큰만을 Self-attention 레이어에서 사용함으로써 타임스텝에 걸친 동적 특성을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: DaTo는 특징 캐싱과 토큰 프루닝을 결합하여 학습 없이도 시간적 및 토큰 기반 정보 재사용을 달성합니다. 이 방법은 각 타임스텝에서의 다이나믹한 토큰을 고유하게 유지하고, 전체 토큰에 걸쳐 다이나믹성을 확장합니다. 또한, DaTo는 검색 공간을 정의하고 이를 최적화 문제로 구성하여 다중 목적을 최적화하는 NSGA-II 알고리즘을 통해 최적의 토큰 프루닝 및 캐싱 전략을 탐색합니다.

- **Performance Highlights**: 우리의 접근법은 ImageNet 데이터셋에서 Stable Diffusion v1.5 모델을 적용했을 때 9배의 속도 향상과 함께 FID를 0.33 줄이는 결과를 도출하였습니다. COCO-30k에서는 7배의 가속과 함께 FID를 2.17로 줄이는 주요 성과를 보였습니다. 이러한 실험을 통해 DaTo의 효과성은 입증되었으며, 훈련이나 추가 데이터 없이도 주요 개선을 보여주었습니다.



### A Novel Shape Guided Transformer Network for Instance Segmentation in Remote Sensing Images (https://arxiv.org/abs/2501.00360)
Comments:
          14 pages, 15 figures

- **What's New**: 이번 연구에서는 원거리 센싱 이미지(Remote Sensing Images, RSI)에서 객체의 경계를 정확하게 추출하고 여러 객체 인스턴스의 상호 정보를 통합하는 문제를 다루며, 새로운 Shape Guided Transformer Network (SGTN)를 제안합니다. SGTN은 특히 LSwin이라는 효과적인 transformer encoder를 통해 자가 주의(self-attention) 메커니즘의 전반적인 맥락 모델링 능력을 가져오며, 이는 Swin Transformer보다 개선된 글로벌 인식 능력을 제공합니다.

- **Technical Details**: LSwin은 수직(vertical) 및 수평(horizontal) 1D 글로벌 자가 주의 메커니즘을 포함하여 원거리 센싱 이미지에서 더 나은 성능을 발휘하도록 설계되었습니다. 또한, 객체 경계와 형태 정보를 강조하는 shape guidance module (SGM)을 도입하여 정밀한 인스턴스 마스크 분할을 이룹니다. SGM은 로컬 세부 정보(local detail information)에 중점을 두고, LSwin은 글로벌 맥락 관계(global context relationships)에 집중합니다.

- **Performance Highlights**: SGTN은 WHU, BITCC, NWPU VHR-10과 같은 두 개의 단일 클래스 공개 데이터셋 및 한 개의 다중 클래스 공개 데이터셋에서 뛰어난 평균 정확도(average precision, AP) 점수를 기록했습니다. 특히 LSwin은 효율성 측면에서 널리 사용되는 ResNet 및 Swin Transformer encoder보다 나은 성능을 입증했습니다.



### Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding (https://arxiv.org/abs/2501.00358)
- **What's New**: 이 논문은 에고센트릭 관찰(egocentric observation)에서 동적 3D 장면을 이해하는 문제를 다루고 있습니다. 기존 연구들이 주로 장기 비디오(long-form video) 이해에 초점을 맞춘 반면, 본 논문에서는 에고센트릭 비디오와 감각 입력(embodied sensory inputs)을 결합한 LLM 기반의 에이전트인 Embodied VideoAgent를 제안합니다. 이 접근방식은 활동이나 작업 인식 시 메모리를 자동으로 업데이트할 수 있는 VLM 기반 방법을 포함합니다.

- **Technical Details**: Embodied VideoAgent는 3D 장면의 복잡한 추론(reasoning) 및 계획(planning) 과제를 해결하는 데 있어 기존 방법들보다 뛰어난 성능을 보입니다. 논문에서는 Ego4D-VQ3D에서 4.9%, OpenEQA에서 5.8%, EnvQA에서 11.7%의 성과 향상을 기록했습니다. 또한, 로봇 조작을 위한 에몸이디드 상호작용(embodied interactions) 및 지각(perception) 생성과 같은 다양한 에몸이디드 AI 태스크에서 잠재력을 입증하였습니다.

- **Performance Highlights**: Embodied VideoAgent는 다양한 복잡한 3D 장면에서의 문제 해결에 있어 큰 이점을 제공합니다. 이는 응용 및 실험에서 높은 성과를 나타내며, 연구자들에게 실질적인 코드를 제공하여 향후 연구에 기여할 예정입니다. 이 결과는 실제 로봇 조작 및 인공지능 응용 분야에 중요한 영향을 미칠 것으로 기대됩니다.



### PanoSLAM: Panoptic 3D Scene Reconstruction via Gaussian SLAM (https://arxiv.org/abs/2501.00352)
- **What's New**: 이번 논문에서는 PanoSLAM이라는 새로운 SLAM(System for Simultaneous Localization and Mapping) 시스템을 소개합니다. PanoSLAM은 기하학적 재구성, 3D 의미 분할(3D semantic segmentation), 3D 인스턴스 분할(3D instance segmentation)을 통합하여 패노프틱(panoptic) 3D 장면 재구성을 수행합니다. 이 시스템은 RGB-D 비디오를 기반으로 하며, 수동 레이블 없이도 작동할 수 있는 첫 번째 방법으로, 효율적인 3D 장면 이해를 가능하게 합니다.

- **Technical Details**: PanoSLAM은 3D Gaussian Splatting 기술을 기반으로 하며, 다양한 시점에서 깊이, 색상, 의미 및 인스턴스 정보를 효율적으로 렌더링할 수 있도록 여러 핵심 요소로 수정되었습니다. Spatial-Temporal Lifting (STL) 모듈을 통해 2D 패노프틱 예측을 3D Gaussian 표현으로 변환하며, 이를 통해 레이블 노이즈와 2D 예측의 불일치를 해결합니다. 이 STL 모듈은 다중 뷰 입력 간의 일관성을 활용하여 가시적인 3D 표현을 향상시킵니다.

- **Performance Highlights**: PanoSLAM은 Replica 및 ScanNet++와 같은 벤치마크 데이터셋에서 최근의 의미 SLAM 방법들보다 맵핑과 추적 정확도에서 현저히 뛰어난 성능을 보였습니다. 특히 이 시스템은 레이블 없이 패노프틱 3D 장면 재구성을 성취하여 경량화된 방법론으로 주목받고 있습니다. PanoSLAM의 혁신적인 접근 방식은 다양한 오픈 월드 환경에서의 3D 재구성을 가능하게 합니다.



### CNC: Cross-modal Normality Constraint for Unsupervised Multi-class Anomaly Detection (https://arxiv.org/abs/2501.00346)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문은 기존의 비지도(distillation-based) 방법이 인코딩된 특징과 디코딩된 특징의 차이를 이용하여 테스트 이미지에서 비정상 영역을 찾는 방법의 한계를 다룹니다. 저자들은 디코더가 정상 샘플만을 학습했음에도 불구하고 비정상 패치의 특징을 잘 복원하는 문제를 제기하며, 이를 'over-generalization'(OG)이라고 명명합니다. 이를 해결하기 위해 클래스 불문 학습 가능한 프롬프트(class-agnostic learnable prompts)를 도입하여 다양한 시각적 패턴 간의 공통적인 정상성을 잡아내고, 이를 통해 비정상 패턴에 대한 과도한 일반화를 억제하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 논문에서는 비정상 패턴의 과도한 일반화 문제를 해결하기 위해 Cross-modal Normality Constraint (CNC)라는 방법을 도입합니다. CNC는 비정상 패턴을 복원하는 것을 방지하기 위해 인코딩된 시각적 특징에서 공통적인 정상성을 추출하는 클래스 불문 프롬프트를 사용합니다. 또한, 다양한 패치 패턴을 처리하기 위해 여러 전문가 네트워크를 구축하는 gated mixture-of-experts (MoE) 모듈도 제안되어, 다중 클래스 훈련에서 상호 간섭을 줄이도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 MVTec AD 및 VisA 데이터셋에서 경쟁력 있는 성능을 성취하며, 비지도 다중 클래스 이상 탐지 분야의 성능을 강화합니다. CNC와 MoE를 통합하여, 기존의 단일 모델 기반 접근방식보다 향상된 성과를 기록하였으며, 실험 결과에서 상당한 성능 개선을 보였습니다. 본 연구는 다중 클래스 훈련에서 비정상 탐지를 위한 새로운 방법론을 제공하며 향후 연구 방향에도 기여할 것으로 기대됩니다.



### SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians (https://arxiv.org/abs/2501.00342)
- **What's New**: 본 논문에서는 3D Gaussian Splatting의 한계를 극복하기 위해 SG-Splatting이라는 새로운 방식을 제안합니다. Spherical Gaussians 기반의 색상 표현을 통해 데이터의 저장 요구를 최소화하고, 렌더링 속도를 획기적으로 개선합니다. 이 방법은 색상 표현에 필요한 매개변수 수를 대폭 줄여, 실시간 응용 프로그램에 적합한 성능을 발휘합니다.

- **Technical Details**: SG-Splatting에서는 Spherical Gaussians을 사용하여 뷰 의존적인 색상을 표현하며, 이를 통해 전통적인 3차 구형 조화 함수(3rd degree spherical harmonics)로 인한 메모리 부담을 감소시킵니다. 또한, 여러 방향의 Spherical Gaussians를 효과적으로 조직하여 장면을 최적화하며, Mixed representation을 통해 높은 빈도와 낮은 빈도의 색상 정보를 모두 효과적으로 캡처합니다. 이러한 접근 방식은 렌더링 품질과 성능을 동시에 향상시킵니다.

- **Performance Highlights**: SG-Splatting은 고속 렌더링을 가능하게 하여 가상 현실, 3D&4D 콘텐츠 생성, 물리적 시뮬레이션 및 자율 주행 분야에서 실시간 응용 프로그램에 최적화된 솔루션을 제공합니다. 기존의 기술들과의 통합이 용이하며, 플러그 앤 플레이(plug-and-play) 방식으로 기존 시스템에 쉽게 통합되어 렌더링 속도를 향상시킵니다. 본 연구는 특히 메모리 사용과 렌더링 속도에서 중요한 개선을 이루며, 실제 적용 가능성을 높이고 있습니다.



### Dynamic Prompt Adjustment for Multi-Label Class-Incremental Learning (https://arxiv.org/abs/2501.00340)
Comments:
          published to BICS2024

- **What's New**: 본 연구는 Multi-Label Class-Incremental Learning (MLCIL)의 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 기존의 Single Label Incremental Learning (SLCIL) 방법들이 아닌, 다중 라벨 환경에서도 효과적으로 작동할 수 있는 새로운 프레임워크를 개발한 것이 특징입니다. 연구에서는 이미지-텍스트 매칭을 활용하여 MLCIL 문제를 처음으로 해결하는 방법을 제안하였습니다.

- **Technical Details**: 이 모델은 Incremental Context Prompt (ICP)와 Selective Confidence Cluster Replay (SCCR)의 두 가지 주요 구성 요소로 이루어져 있습니다. ICP는 다양한 클래스 간의 균형을 유지하도록 설계되었으며, SCCR는 클러스터링 및 모델 신뢰도를 이용하여 중요한 샘플을 재생하는 방법입니다. 이 과정에서 Textual Prompt Consistency Loss를 도입하여 텍스트 프롬프트의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 MS COCO와 PASCAL VOC 데이터셋에서 MLCIL 작업의 성능을 크게 향상시켰습니다. 본 연구는 기존 방법들에 비해 경쟁력 있는 결과를 보여주며, MLCIL 분야에서의 효과성을 입증했습니다. 이러한 결과는 본 연구의 접근 방식이 다중 클래스 인식에서 발생하는 지식 소실(knowledge forgetting)을 효과적으로 완화할 수 있음을 나타냅니다.



### OVGaussian: Generalizable 3D Gaussian Segmentation with Open Vocabularies (https://arxiv.org/abs/2501.00326)
- **What's New**: OVGaussian은 오픈-어휘(오픈-밸리) 3D 의미 분할(segmentaion) 프레임워크로, 3D Gaussian 표현을 기반으로 하여 일반화 가능성을 높였습니다. SegGaussian이라는 대규모 3D 데이터셋을 구성하여 각 Gaussian 포인트와 다중 시점(multi-view) 이미지에 대한 자세한 의미 및 인스턴스 주석을 제공하여 3D 장면 이해 능력을 크게 향상시킵니다. 이 방법은 Generalizable Semantic Rasterization(GSR) 및 Cross-modal Consistency Learning(CCL)을 도입하여 의미를 유지하면서 여러 장면에서 일반화할 수 있도록 합니다.

- **Technical Details**: OVGaussian은 288개의 3D Gaussian 장면으로 구성된 SegGaussian 데이터셋을 활용하여, 3D Gaussian 포인트마다 의미적 성질을 예측하는 3D 신경망을 학습합니다. GSR 방법을 통해 각 3D Gaussian에서 예측된 의미적 속성을 다중 시점 일관성 2D 의미 맵으로 렌더링(renderring)할 수 있습니다. 나아가, CCL을 통해 SegGaussian 내의 2D 이미지와 3D Gaussian 주석 간의 정합성을 높여 모델의 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, OVGaussian은 오픈-어휘 의미 분할에서 최신 성능을 달성하였으며, 크로스-씬(cross-scene), 크로스-도메인(cross-domain), 새로운 뷰(새로운 관점) 일반화(capabilities)에서 강력한 효율성을 보였습니다. 이 연구는 3D 공간에서의 오픈-어휘 이해를 위한 새로운 방향을 제시하며, Gaussian 기반의 표현이 다양한 현실 세계 시나리오에서 의미 분할 도구로서의 활용 가능성을 보여줍니다.



### OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning (https://arxiv.org/abs/2501.00321)
- **What's New**: 최근 LMMs(대형 멀티모달 모델)의 OCR(광학 문자 인식) 능력을 평가하는 데 대한 관심이 커지고 있습니다. 기존 벤치마크는 LMMs의 텍스트 인식 성능이 인상적임을 보여주지만, 텍스트 위치 파악, 손글씨 내용 추출 및 논리적 추론과 같은 특정 도전 과제에 대한 성능은 아직 충분히 탐구되지 않았습니다. 이를 보완하기 위해 OCRBench v2를 소개하며, 이전의 멀티-장면 벤치마크인 OCRBench보다 4배 더 많은 과제를 포함하고 있습니다.

- **Technical Details**: OCRBench v2는 현재 가장 포괄적인 bilingual (이중 언어) 텍스트 중심 벤치마크로, 31개의 다양한 시나리오(거리 장면, 영수증, 수식, 다이어그램 등)를 포함하고 있습니다. 이는 인적 검증을 받은 10,000개의 질문-답변 쌍과 높은 비율의 어려운 샘플로 구성되어 있으며, 철저한 평가 지표를 제공합니다. 최신 LMMs를 OCRBench v2에서 신중하게 벤치마킹한 결과, 22개 LMM 중 20개가 100점 만점에 50점 이하를 기록했습니다.

- **Performance Highlights**: LMM은 텍스트 인식과 해석의 능력을 비교할 때 다섯 가지 제한 사항이 있음을 발견했습니다. 이러한 제한 사항은 자주 접하지 않는 텍스트 인식, 세분화된 인식, 레이아웃 인식, 복잡한 요소 구문 분석, 논리적 추론을 포함합니다. LMM의 성능 평가에 있어 OCR은 중요한 구성이 되었으며, 자주 사용되는 VQA(시각 질문 답변) 데이터셋을 활용하여 LMM의 텍스트 인식 능력을 평가하고 있습니다.



### Improving Text-based Person Search via Part-level Cross-modal Correspondenc (https://arxiv.org/abs/2501.00318)
- **What's New**: 본 논문에서는 자연어 텍스트 설명을 기반으로 개인 이미지를 검색하는 text-based person search를 위한 효율적인 encoder-decoder 모델을 소개합니다. 이 모델은 비지도 학습으로 두 모달리티 간의 의미적으로 정렬된 coarse-to-fine embedding vectors를 추출합니다. 또한, 유사한 신체 부위의 차이를 구별하는 세부 정보를 학습하는 과정에서 발생하는 문제를 해결하기 위해, commonality-based margin ranking loss라는 새로운 손실 함수를 제안합니다.

- **Technical Details**: text-based person search는 이미지에서 개인을 찾기 위한 작업으로, 이 작업은 일반적으로 이미지의 시각적 특성, 속성 집합, 자연어를 쿼리로 활용합니다. 이 방법은 비디오에서 범죄자를 식별하거나 여러 감시 카메라를 통한 실종자 검색과 같은 공공 안전 응용 프로그램에서 중요한 역할을 합니다. 텍스트 쿼리는 유연성과 사용자 친화성을 가진 쿼리 수집 덕분에 개인 검색을 효율적이고 효과적으로 가능하게 합니다.

- **Performance Highlights**: 이 연구의 결과로 제안된 방법은 세 가지 공개 벤치마크에서 최고의 기록을 달성했습니다. 이는 새로운 손실 함수와 효율적인 모델이 text-based person search에서의 성능 향상에 기여했음을 보여줍니다. 따라서 기존의 이미지 또는 속성을 기반으로 한 쿼리보다 텍스트 쿼리가 더 적합한 환경에서의 강력한 성능을 가능하게 합니다.



### Spatio-Temporal Multi-Subgraph GCN for 3D Human Motion Prediction (https://arxiv.org/abs/2501.00317)
- **What's New**: 이 논문에서는 과거 데이터를 바탕으로 미래의 사람의 동작을 예측하는 Human motion prediction (HMP) 분야에 혁신적인 접근 방식을 제시합니다. 기존의 Graph Convolutional Networks (GCNs) 기반 방법은 시공간적 특성을 완전히 활용하지 못한 채로 시간 도메인(temporal-domain) 또는 공간 도메인(spatial-domain) 특성에 주로 집중하였습니다. 이에 대해 Spatial-Temporal Multi-Subgraph Graph Convolutional Network (STMS-GCN)을 통해 복잡한 시공간적 의존성(complex spatio-temporal dependencies)을 캡처하는 방법론을 제안합니다.

- **Technical Details**: STMS-GCN은 시간적 의존성과 공간적 의존성을 분리하여 다중 스케일(multiple scales)에서의 교차 도메인 지식 전이(cross-domain knowledge transfer)를 가능하게 합니다. 또한, 동질적 정보 제약 메커니즘(homogeneous information constraint mechanism)을 활용하여 다수의 서브그래프(subgraphs)를 통해 보다 풍부한 운동 정보를 추출하고 이들의 학습 연관성을 개선합니다. 이러한 접근은 시공간 정보의 일관성을 유지하도록 설계되었습니다.

- **Performance Highlights**: 표준 HMP 벤치마크에서의 광범위한 실험 결과, 제안한 STMS-GCN 방법이 기존 대안보다 뛰어난 성능을 보이는 것이 입증되었습니다. 이러한 결과는 복잡한 동작 예측 문제를 해결하는 데 있어 STMS-GCN의 장점을 강조하며, 향후 다양한 인간 동작 예측 응용 프로그램에 기여할 것으로 기대됩니다.



### Temporal Dynamics Decoupling with Inverse Processing for Enhancing Human Motion Prediction (https://arxiv.org/abs/2501.00315)
- **What's New**: 이 논문은 Human Motion Prediction (HMP)의 새로운 접근 방식을 제안합니다. 제안한 방법인 Temporal Decoupling Decoding with Inverse Processing (TD2IP)는 재구성과 예측 작업을 서로 분리하여 진행합니다. 이러한 분리는 재구성과 예측 사이의 갈등을 완화하여 모션 패턴에 대한 보다 깊은 이해를 가능하게 합니다.

- **Technical Details**: TD2IP는 두 개의 별도의 디코더를 사용하여 공유 모션 특징을 역사적 또는 미래의 시퀀스로 디코딩합니다. Inverse Processing (IP)라는 새로운 보조 작업을 통해 미래 모션을 시간적으로 역전시키고 이를 모델에 재도입하면서 역사적 모션 예측을 가능하게 합니다. 이 과정은 양방향 시간 상관관계를 활용하여 역사적 및 미래 정보 간의 연관성을 강화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TD2IP의 효과성과 기존 방법들에 대한 우수성을 입증하였습니다. 기존의 인코더-디코더 프레임워크에 대한 새로운 확장을 제공하며, 각 디코더가 특정 시간 지평에 맞춰 전문화되어 있어 모션 표현력을 극대화하는 데 기여합니다. 본 연구에서 제안한 메커니즘은 인공 지능 분야의 모션 예측 정확도를 비약적으로 향상시킬 가능성이 높습니다.



### SAM-Aware Graph Prompt Reasoning Network for Cross-Domain Few-Shot Segmentation (https://arxiv.org/abs/2501.00303)
Comments:
          AAAI 2025

- **What's New**: 본 논문에서는 cross-domain few-shot segmentation (CD-FSS) 문제를 해결하기 위해, SAM(이전 모델들에 비해 우수한 일반화 성능을 가진 대규모 시각 모델)의 장점을 활용하는 새로운 방법인 GPRN(Graph Prompt Reasoning Network)을 제안합니다. 이는 SAM이 제공하는 시각 프롬프트를 활용하여 CD-FSS의 피처 표현 학습을 개선하게 됩니다. 특히, SAM-aware prompt initialization module(SPI)와 그래프 프롬프트 추론 모듈(GPR)을 도입하여 시맨틱 일관성을 확보하며 예측 성능을 향상시킵니다.

- **Technical Details**: 제안된 GPRN 모델은 SAM으로부터 생성된 마스크를 고급 시맨틱 정보가 풍부한 시각 프롬프트로 변환하는 기능을 갖추고 있으며, 이를 통해 여러 시각 프롬프트 간의 상호관계를 분석하고 유사한 프롬프트로부터 정보를 집계함으로써 글로벌 시맨틱 일관성을 달성합니다. 또한, 테스트 과정에서는 비파라메트릭 적응형 포인트 선택 모듈(APS)을 설계하여 쿼리 예측으로부터 대표 포인트 프롬프트를 선택하여 최종 세분화 마스크를 정제하는 데 도움을 줍니다.

- **Performance Highlights**: 네 개의 표준 CD-FSS 데이터셋에서의 실험 결과, GPRN은 새로운 최첨단 결과를 수립했습니다. PASCAL VOC 데이터셋에서의 실험 결과, 초기 훈련 없이 GPRN에 대한 성능 향상은 없었으나, 파인 튜닝 후 성능이 무려 4% 향상됨을 확인했습니다. APS 모듈을 통한 시각화 결과는 세분화 정확도를 높이는 데 기여함을 보여줍니다.



### Research on vehicle detection based on improved YOLOv8 network (https://arxiv.org/abs/2501.00300)
- **What's New**: 이번 연구에서는 자율 주행 시스템의 안전한 장애물 회피 기능을 보장하기 위한 극도로 정확한 차량 인식 기술의 필요성을 강조합니다. 실제 도로 환경의 변동성과 다양한 차량 및 보행자의 특성이 탐지 정확도를 향상하는 데 큰 장애물이 되고 있음을 지적합니다. 이 문제를 해결하기 위해 개선된 YOLOv8 차량 탐지 방법을 제안합니다.

- **Technical Details**: 본 논문에서는 YOLOv8n-seg 모델을 기반으로 하여 여러 기술적 개선을 수행합니다. 첫째, FasterNet 네트워크를 백본으로 사용하여 계산 복잡성과 메모리를 줄이면서 탐지 정확도와 속도를 향상시킵니다. 둘째, 주의 메커니즘인 CBAM을 Neck에 추가하여 특징 강화를 달성하며, 마지막으로 손실 함수 CIoU를 WIoU로 수정하여 탐지 박스의 위치를 최적화하고 세그멘테이션 정확도를 향상합니다.

- **Performance Highlights**: 개선된 모델은 자동차, 보행자, 오토바이에 대해 각각 98.3%, 89.1%, 88.4%의 탐지 정확도를 달성하였습니다. 이 모델은 이전 YOLOv8 모델 및 YOLOv9 모델 대비 6가지 지표에서 우수한 성능을 보입니다. 이러한 성과는 자율 주행 분야에서 차량 탐지 기술의 발전에 기여할 것으로 기대됩니다.



### Dual Diffusion for Unified Image Generation and Understanding (https://arxiv.org/abs/2501.00289)
- **What's New**: 이 논문은 멀티모달(multi-modal) 이해 및 생성을 위한 대규모의 완전 엔드 투 엔드(diffusion model)를 제안합니다. 기존의 diffusion 기반 멀티모달 모델들보다 성능이 크게 향상된 이 모델은 비전-언어 모델링의 전체 기능을 지원하는 최초의 사례로, 이미지와 텍스트의 조건부 가능성을 동시에 훈련하는 새로운 기법을 도입했습니다. 새로운 손실 함수는 이미지와 텍스트 자료를 통합적으로 학습하여 다양한 작업을 수행할 수 있게 돕습니다.

- **Technical Details**: 제안된 모델은 멀티모달 확산 변환기(multimodal diffusion transformer) 구조를 기반으로 하며, 이미지 및 텍스트 모달리티에 대해 연속 및 이산 그래디언트를 활용한 확산을 수행합니다. 이 모델은 이미지 생성, 캡셔닝(captioning), 시각적 질문 응답(visual question answering) 등을 수행할 수 있는 유연성을 지니고 있습니다. 또한, 기존의 diffusion 기반 모델과의 호환성을 통해 빠른 적응이 가능합니다.

- **Performance Highlights**: 제안된 모델은 최근의 통합 이미지 이해 및 생성 모델과 경쟁할 수 있는 성능을 보여주었으며, 다양한 멀티모달 작업에서 현저한 성능 향상을 이루었습니다. 이는 기존의 diffusion 모델들이 갖추지 못했던 다양한 기능을 가능하게 하여, 인공지능의 다음 단계로 나아가는 건설적인 기여를 제공합니다. 이런 방식으로 멀티모달 확산 모델링은 이미지와 텍스트의 관계를 더욱 깊이 있게 탐구할 수 있는 기회를 제공합니다.



### Cross-Layer Cache Aggregation for Token Reduction in Ultra-Fine-Grained Image Recognition (https://arxiv.org/abs/2501.00243)
Comments:
          Accepted to ICASSP 2025. Main: 5 pages, 4 figures, 1 table

- **What's New**: 이번 연구는 Ultra-fine-grained image recognition (UFGIR)에서 기존의 token reduction 방법에서 발생하는 정보 손실을 해결하기 위해 Cross-Layer Aggregation Classification Head (CLA)와 Cross-Layer Cache (CLC) 메커니즘을 제안합니다. CLA는 중간 레이어의 정보를 직접 분류 모듈로 전송하고, CLC는 이전 레이어의 정보를 다시 사용할 수 있도록 저장합니다. 이 두 가지 메커니즘을 통합한 Cross-Layer Cache Aggregation (CLCA) 방법은 토큰 유지 비율을 10%로 낮추면서도 높은 정확도를 유지할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 CLCA 방법은 일반적인 Vision Transformer 구조에서 토큰 감소를 통해 구현됩니다. 이 방법은 인코더 블록을 그룹으로 나누고, 각 그룹의 CLS 토큰 출력을 기반으로 중간 특징을 집계합니다. 각 그룹의 마지막 인코더 블록에서는 다양한 token reduction 방식에 따라 토큰을 줄이고, 이렇게 얻은 정보를 깊은 레이어에서 다시 사용할 수 있도록 CLC 구조를 통해 관리합니다.

- **Performance Highlights**: 2000회 이상의 실험을 통해, CLCA는 다양한 설정에서 정확도를 유의미하게 향상시키는 결과를 보여줍니다. 또한, 제안된 방법은 계산 비용을 최소화하면서 정확도를 높은 수준으로 유지하여 더 효율적인 UFGIR 시스템을 가능하게 합니다. 특히, FLOPs 기준으로 더 높은 비용의 모델과 비슷한 정확도를 달성하면서도, 훨씬 적은 수의 토큰을 처리함으로써 머신 러닝의 효율성을 극대화합니다.



### Make Domain Shift a Catastrophic Forgetting Alleviator in Class-Incremental Learning (https://arxiv.org/abs/2501.00237)
Comments:
          Accepted as poster paper of AAAI2025

- **What's New**: 이 논문은 클래스 증분 학습(Class-Incremental Learning, CIL) 분야에서 도메인 이동(domain shift)을 통합함으로써 재학습 없이도 망각을 감소시킬 수 있음을 발견했습니다. 실험을 통해 도메인 이동이 차별화된 피처 분포(feature distribution)를 만들어 모델의 성능을 향상시키고 잊어버림 현상을 효과적으로 완화한다는 것을 보여주었습니다. 새로운 기법 'DisCo'를 제안하여 CIL 과제를 효율적으로 처리할 수 있는 방법을 모색합니다.

- **Technical Details**: CIL은 새로운 클래스를 포함한 일련의 과제를 학습하는 과정으로, 모델이 이전 과제에서 본 모든 클래스로 새로운 샘플을 올바르게 분류해야 합니다. 본 논문에서는 도메인 이동과 CIL을 결합하여 파라미터 간섭을 줄이는 새로운 방법을 제안합니다. DisCo는 대조 손실(contrastive loss)을 활용하여 현재 작업에 대한 뚜렷한 피처 분포를 촉진하고, 이전 지식을 효과적으로 보존하기 위한 크로스-태스크 대조 증류 손실(cross-task contrastive distillation loss)을 포함합니다.

- **Performance Highlights**: DisCo를 기존 CIL 기법에 통합한 결과, 망각이 현저하게 감소하고 성능이 일관되게 향상되었습니다. 이 방법은 다양한 CIL 벤치마크에서 검증되어 향상된 성능을 입증하였으며, 클래스 증분 학습의 미래 연구에 활용될 수 있는 강력한 솔루션으로 자리잡을 것으로 기대됩니다.



### DecoratingFusion: A LiDAR-Camera Fusion Network with the Combination of Point-level and Feature-level Fusion (https://arxiv.org/abs/2501.00220)
Comments:
          12 pages, 2 figures. accepted by ICANN2024

- **What's New**: 본 논문에서는 리다(Lidar)와 카메라의 데이터를 결합하는 새로운 접근 방식인 DecoratingFusion을 제안합니다. 이는 기존의 피쳐 레벨(fusion) 및 포인트 레벨(fusion) 결합 방식을 통합하여 하드 상관관계(hard correlation)를 활용합니다. 이 방법은 보정 행렬(calibration matrices)에 의해 정의된 하드 상관관계를 사용하여 객체 쿼리(object queries)를 생성하고 3D 탐지에서의 성능을 향상시킵니다.

- **Technical Details**: DecoratingFusion은 포인트 레벨(fusion)과 피쳐 레벨(fusion) 단계로 구성됩니다. 초기 단계에서는 2D CNN 기능을 사용하여 이미지 데이터로 포인트 클라우드(point cloud)를 장식하고, 두 개의 독립적인 스파스 컨볼루션(sparse convolutions)을 사용하여 장식된 포인트 클라우드 기능을 추출합니다. 중간 단계에서는 중심 히트맵(center heatmap)을 예측하여 객체 쿼리의 초기 위치를 설정하고, 예측된 클래스 레이블을 아이디어 추가 정보를 쿼리에 삽입합니다.

- **Performance Highlights**: KITTI 및 Waymo와 같은 두 개의 주요 자율 주행 데이터 세트에서 광범위한 실험을 수행했으며, DecoratingFusion의 접근 방식이 기존 방법에 비해 우수한 성능을 제공함을 입증하였습니다. 다양한 실험 결과, 제안된 방식이 상관관계 간의 학습을 통해 보다 정확한 3D 탐지를 가능하게 함을 보였습니다.



### MLLM-as-a-Judge for Image Safety without Human Labeling (https://arxiv.org/abs/2501.00192)
- **What's New**: 이번 연구는 AI 생성 이미지 콘텐츠 안전의 중대성을 강조하며, 기존의 데이터 세트에 대한 인간 주석자 의존의 한계를 극복하기 위해 Multimodal Large Language Models (MLLMs)와 미리 정의된 안전 규정(Safety Constitution) 기반의 제로샷(zero-shot) 이미지 안전 판단 방법론인 CLUE를 제안합니다.

- **Technical Details**: CLUE 방법론은 안전 규정을 객관적인 규칙으로 변환하고, 각 이미지를 평가하기 위해 규정 각각의 타당성을 평가합니다. 복잡한 안전 규정을 처리하기 위해 논리적으로 완전하지만 단순화된 전제 조건 체인을 사용하며, 이미지와 규칙의 관련성을 측정하기 위해 CLIP과 같은 다중모달 대조 모델을 활용합니다.

- **Performance Highlights**: 실험 결과, CLUE 방법론은 Qwen2-VL-7B-Instruct, InternVL2-8B-AWQ, LLaVA-v1.6-34B 및 InternVL2-76B와 같은 다양한 MLLM에서 제로샷 이미지 안전 판단의 정확성과 신뢰성을 크게 개선하며, InternVL2-76B 모델을 사용하여 Unsafe/Safe 이미지를 구분할 때 95.9%의 리콜, 94.8%의 정확도 및 0.949 F-1 점수를 달성하였습니다.



### Minimalist Vision with Freeform Pixels (https://arxiv.org/abs/2501.00142)
Comments:
          Project page: this https URL, published at ECCV 2024

- **What's New**: 이 논문에서는 최소한의 픽셀(minimalist pixel)로 비전 태스크(vision task)를 해결할 수 있는 비전 시스템을 소개합니다. 전통적인 카메라가 정사각형 픽셀로 구성되는 반면, 최소한의 카메라는 자유 형태 픽셀(freeform pixels)을 사용하여 정보 밀도를 높입니다. 이를 통해 새로운 형태의 카메라가 더 작은 공간에서 더 많은 정보를 수집할 수 있게 됩니다.

- **Technical Details**: 이 최소한의 카메라의 하드웨어는 신경망(neural network)의 첫 번째 레이어로 모델링될 수 있습니다. 네트워크는 특정 태스크에 대해 훈련되며, 이 훈련 과정에서 카메라의 자유 형태 픽셀의 형상이 결정됩니다. 이 픽셀은 광검출기(photodetector)와 광학 마스크(optical mask)를 이용해 구현됩니다.

- **Performance Highlights**: 각각 8개의 픽셀을 갖는 다양한 최소한의 카메라 설계가 제공되었으며, 실내 공간 모니터링, 조명 측정, 교통 흐름 추정 등을 수행할 수 있습니다. 이러한 시스템의 성능은 수십 배 더 많은 픽셀을 가진 전통적인 카메라와 동등한 수준임을 입증하였습니다. 이러한 최소한의 비전 시스템은 개인의 프라이버시를 보호하고, 극소량의 측정을 통해 외부 전원이나 배터리 없이 스스로 작동할 수 있는 장점을 제공합니다.



### Detection-Fusion for Knowledge Graph Extraction from Videos (https://arxiv.org/abs/2501.00136)
Comments:
          12 pages, To be submitted to a conference

- **What's New**: 본 논문에서는 영상 이해의 주요 과제로서 비디오에서 의미 있는 내용을 추출하기 위한 새로운 방법을 제안합니다. 기존의 시스템들이 자연어 모델을 사용하여 비디오를 설명하는 데 따른 여러 단점을 극복하기 위해, 지식 그래프(knowledge graphs)를 활용하여 비디오에 주석을 달고자 합니다. 이 방법은 비디오의 비주얼 콘텐츠에 기반한 내용을 더 효율적으로 표현할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 깊이 학습(deep learning) 기반으로, 먼저 비디오 입력에서 개체(individuals)와 그들 간의 관계를 예측한 후, 이를 통해 지식 그래프를 구성합니다. 개체는 학습 가능한 벡터로 표현되고, 관계는 다층 퍼셉트론(multi-layer perceptrons, MLP)으로 처리됩니다. 이 모델은 비디오에서 탐지된 개체와 서술어(predicate)를 결합하여 사실(fact)을 예측하며, 문장을 사용하지 않고도 보다 정량적인 평가가 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 지식 그래프로 비디오에 주석을 다는 중요한 작업에서 기존 작업들보다 월등히 성능이 뛰어남을 입증했습니다. 모델의 각 구성요소에 대한 기여도를 분석한 결과, 개체와 서술어 수를 조절함에 따라 실행 시간과 정확도 간의 트레이드오프(trade-off)가 나타났습니다. 지식 그래프의 배경 지식(background knowledge) 포함에 대한 탐색은 해당 분야에서 최초로 이루어진 연구입니다.



### PQD: Post-training Quantization for Efficient Diffusion Models (https://arxiv.org/abs/2501.00124)
Comments:
          7 pages, 3 figures, uses this http URL

- **What's New**: 최근 확산 모델(Diffusion Models, DMs)이 고화질 및 다양한 이미지 합성에서 괄목할 만한 성과를 보이고 있습니다. 그러나 이 모델들은 높은 계산 요구와 느린 생성 속도로 인해 널리 사용되지 못하고 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해 향후 적용가능성이 높은 포스트 트레이닝 양자화(post-training quantization, PQD) 방식의 새로운 최적화 프레임워크를 제안합니다. 이러한 방식은 이미지 생성을 위한 고급 기술들을 적절히 조합하여 높은 성능을 유지합니다.

- **Technical Details**: 제안된 PQD 프레임워크는 추론(inference) 과정을 최적화하여 대표 샘플을 선택하고, 시간 인식(time-aware) 캘리브레이션(calibration)을 수행합니다. 실험 결과, PQD를 사용하여 8비트 또는 4비트 모델로 전체 정밀도(diffusion models) 모델을 직접 양자화할 수 있으며, 훈련 없이도 유사한 성능을 유지하는 것으로 나타났습니다. 나아가, 이 방법은 512x512 크기의 텍스트 가이드 이미지 생성에 처음으로 적용되었습니다.

- **Performance Highlights**: 우리의 접근 방식은 이미지를 생성하는 과정에서 기존 방법들보다 더 나은 이미지 품질을 제공합니다. 구체적으로, 이미지넷에서 무조건적인 이미지 생성을 수행할 때 몇 가지 FID 변화(frechet inception distance 변화)를 달성했습니다. 또한, 512x512 고해상도 텍스트 가이드 이미지 생성에서 최첨단 결과를 달성하며, 고압축률을 유지하면서도 뛰어난 이미지 품질을 보장합니다.



### Text-to-Image GAN with Pretrained Representations (https://arxiv.org/abs/2501.00116)
- **What's New**: 본 논문에서는 TIGER라는 새로운 텍스트-이미지 GAN을 제안하여, 이전의 GAN보다 빠르고 강력한 모델을 구축하려고 합니다. TIGER는 pretrained representations을 활용하여 복잡한 씬에 대한 이해 능력을 크게 향상시킬 수 있는 비전 강화 판별기(vision-empowered discriminator)와 텍스트-이미지 융합을 효율적으로 수행할 수 있는 고용량 생성기(high-capacity generator)를 포함합니다. 이 연구는 GAN을 재조명하여 새로운 접근 방식을 제공합니다.

- **Technical Details**: TIGER의 비전 강화 판별기는 여러 pretrained 비전 모델의 다양한 표현을 수집하여 성능을 향상시킵니다. 이 판별기는 복수의 서브 판별기로 구성되어 있으며, 각 서브 판별기는 서로 다른 pretrained 모델에서 추출된 표현을 처리합니다. 고용량 생성기는 여러 개의 신선한 고용량 융합 블록(HFBlock)으로 구성되며, 각 블록은 효율적인 텍스트-이미지 융합을 위해 여러 깊은 융합 모듈을 포함하고 있습니다.

- **Performance Highlights**: TIGER는 일반 텍스트-이미지 합성(task) 과제에서 두 가지 도전적인 데이터 세트에서 최첨단 성능을 달성하며, FID 값은 COCO에서 5.48, CUB에서 9.38로 기록되었습니다. 제로샷 텍스트-이미지 합성(task) 과제에서도 적은 모델 파라미터와 적은 훈련 데이터로 비슷한 성능을 보이며, LDM 및 Parti보다 120배 빠른 추론 속도를 자랑합니다.



### LTX-Video: Realtime Video Latent Diffusion (https://arxiv.org/abs/2501.00103)
- **What's New**: LTX-Video는 비디오 생성을 위한 혁신적인 변형 모델로, Video-VAE와 denoising transformer의 통합을 통해 효율성과 품질을 향상시킵니다. 이 모델은 1:192의 높은 압축 비율을 달성하며, 32x32x8의 시공간 다운샘플링을 지원합니다. LTX-Video는 텍스트-비디오 및 이미지-비디오 생성 기능을 동시에 훈련시켜 다양한 활용 가능성을 제공합니다.

- **Technical Details**: LTX-Video는 고차원 압축된 잠재 공간에서 작동하여 전체 시공간 셀프 어텐션을 효율적으로 수행합니다. VAE 디코더는 잠재 변수를 픽셀로 변환하는 작업과 최종 denoising 단계를 동시에 처리하여, 고주파 세부 정보를 보존할 수 있습니다. 이는 전통적인 방식과 비교하여 시간 비용을 줄이며, 더 나은 품질의 비디오 생성을 가능하게 합니다.

- **Performance Highlights**: 이 모델은 Nvidia H100 GPU에서 2초 만에 5초 길이의 24 fps 비디오를 생성할 수 있으며, 이는 동종 모델 중에서 가장 빠른 성능으로 평가받고 있습니다. 2B 파라미터 미만의 규모를 유지하면서도, 모든 기존 모델을 초월하는 빠른 시간 내에 질 높은 비디오를 생성합니다. LTX-Video는 기계 학습 연구자들에게 공개되어 접근성과 확장성을 높이고 있습니다.



### ProjectedEx: Enhancing Generation in Explainable AI for Prostate Cancer (https://arxiv.org/abs/2501.01392)
- **What's New**: 이 논문은 전립선 암 진단을 위한 새로운 접근 방식인 ProjectedEx를 제안합니다. ProjectedEx는 의료 이미지 특징과 분류기 결정 간의 연결을 통해 설명 가능한 다중 속성 설명을 제공합니다. 또한, 피쳐 피라미드를 통합하여 잠재 공간을 정제하고 생성된 설명의 품질을 개선합니다. 이 연구는 AI를 의료 환경에서 채택할 수 있도록 지원하는 해석 가능성을 높입니다.

- **Technical Details**: ProjectedEx 프레임워크는 생성 모델링과 다중 스케일 피쳐 분석을 통합하여 의료 이미지 특징과 분류기 결정을 연결합니다. 이 구조는 피처 피라미드 모듈과 여러 해상도에서 작동하는 일련의 판별기로 구성된 인코더-디코더 형식을 갖추고 있습니다. 모든 판별기의 출력은 고정 해상도에서 통합되어, 다양한 스케일에서 일관된 출력을 보장합니다. 이렇게 구성된 모델은 멀티 스케일 기능의 균형 잡힌 활용을 장려하여 해석 가능한 설명의 생성을 개선합니다.

- **Performance Highlights**: 논문에서 수행한 포괄적인 실험은 ProjectedEx가 임상적으로 의미 있는 시각화와 신뢰할 수 있는 진단 통찰력을 생성함을 입증하였습니다. 기존의 자연 이미지 분류에서 효과적이었던 StylEx와의 비교를 통해 의료 영상 적용에서의 한계를 극복하였고, AI의 의료 환경에서의 활용 가능성을 높였습니다. 추가적으로, 모든 판별기의 손실을 집합하여 생성자에 대한 피드백을 제공하여 지속적인 성능을 향상시킵니다.



### Training Medical Large Vision-Language Models with Abnormal-Aware Feedback (https://arxiv.org/abs/2501.01377)
Comments:
          16 pages

- **What's New**: 본 연구에서는 기존의 Medical Large Vision-Language Models (Med-LVLMs)의 한계를 극복하기 위해 UMed-LVLM을 제안합니다. 이 모델은 의료 이미지에서의 시각적 로컬라이제이션(visual localization) 문제를 해결하는 데 중점을 두고 개발되었습니다. 또한 Medical Abnormalities Unveiling (MAU) 데이터셋을 사용하여 병리학적 이상 감지 능력을 보강합니다.

- **Technical Details**: UMed-LVLM은 두 단계의 훈련 방법인 Abnormal-Aware Instruction Tuning과 Abnormal-Aware Rewarding을 통해 교육됩니다. Abnormal-Aware Rewarding은 Abnormal Localization Rewarding과 Vision Relevance Rewarding을 포함하여 모델이 이상 영역을 효과적으로 캡처할 수 있도록 설계되었습니다. 이는 의료 이미지를 이해하고 이에 따른 진단을 생성하는 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, UMed-LVLM은 기존의 Med-LVLMs와 비교하여 의료 이미지를 이해하고 이상을 식별하는 데 있어 뛰어난 성능을 보여주었습니다. 또한, 모델을 훈련하고 일반화 능력을 심층 분석한 결과, Med-LVLMs의 이상 감지 능력을 개선하는 것이 의료 이미지 이해에 미치는 긍정적인 영향을 강조했습니다. 대규모 모델이 제한된 질병 유형의 다양한 의료 이미지에 노출되더라도 robust한 질병 인식을 위한 잠재력을 갖추고 있음을 보여주었습니다.



### ScarNet: A Novel Foundation Model for Automated Myocardial Scar Quantification from LGE in Cardiac MRI (https://arxiv.org/abs/2501.01372)
Comments:
          31 pages, 8 figures

- **What's New**: ScarNet는 LGE 이미징에서 심근 섬유화와 흉터를 평가하는 새로운 하이브리드 모델입니다. 이 모델은 Medical Segment Anything Model (MedSAM)에서 가져온 transformer 기반의 인코더와 U-Net 디코더를 결합하여 만들어졌습니다. ScarNet의 도입으로 수작업으로 이루어지는 LV 흉터 정량화의 비효율성 문제를 해결하고, 임상 환경에서의 활용 가능성을 높였습니다.

- **Technical Details**: ScarNet는 552명의 허혈성 심근병증 환자로부터 전문가의 세분화 데이터를 이용하여 훈련되었습니다. ScarNet은 tailored attention blocks로 성능을 향상시킨 하이브리드 구조이며, 184명의 별도 환자에서 테스트되었습니다. 모델은 높은 스카 경계 정밀도를 달성하며, 기존 모델들보다 더 낮은 편향(bias)과 변동계수(coefficient of variation)를 보였습니다.

- **Performance Highlights**: ScarNet은 테스트 환자에서 0.912의 중앙 Dice 점수를 기록하며, MedSAM과 nnU-Net보다 현저히 우수한 성능을 보였습니다. Monte Carlo 시뮬레이션에서도 ScarNet은 0.892의 높은 스카 Dice를 달성하여, 노이즈 변화에도 강한 내성을 입증했습니다. 이러한 결과는 ScarNet이 다양한 이미지 품질과 흉터 패턴에 대해 안정적인 성능을 발휘함을 보여줍니다.



### Domain-invariant feature learning in brain MR imaging for content-based image retrieva (https://arxiv.org/abs/2501.01326)
Comments:
          6 pages, 1 figures. Accepted at the SPIE Medical Imaging 2025

- **What's New**: 이번 연구에서는 뇌 MR 이미지를 위한 콘텐츠 기반 이미지 검색(CBIR) 시스템에서 도메인 차이를 줄이기 위해 스타일 인코더 적대적 도메인 적응(style encoder adversarial domain adaptation, SE-ADA)이라는 새로운 저차원 표현(low-dimensional representation, LDR) 방법을 제안했습니다. 기존의 방법들이 도메인 간의 통일성을 확보하는 데 한계를 보인 반면, SE-ADA는 적대적 학습(adversarial learning)을 통해 도메인 정보를 분리하고 병리적 특징을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: SE-ADA는 3D 컨볼루션 오토인코더(3D-CAE) 구조를 기반으로 하는 새로운 메커니즘으로, 주요 인코더(primary encoder)와 분리된 스타일 인코더(style encoder)를 통합합니다. 이 스타일 인코더는 도메인 특정 스타일 정보를 추출하여 LDR에 불필요한 도메인 정보를 제거함으로써 이미지 검색을 위한 필수적인 질병 탐지 정보를 보존합니다. 연구 방식은 인코더, 디코더, 스타일 인코더 각각의 구성 요소를 반복적으로 훈련하여 LDR을 업데이트하는 것입니다.

- **Performance Highlights**: SE-ADA는 8개의 공개 뇌 MR 데이터 세트(ADNI1/2/3, OASIS1/2/3/4, PPMI)에서 최근의 도메인 조화화 방법들과 비교한 결과, 도메인 정보를 효과적으로 제거하면서도 원본 뇌 구조의 주요 측면을 보존하여 최고의 질병 검색 정확도를 보여주었습니다. 이러한 성과는 CBIR 시스템의 정밀도를 높이는 데 기여할 것으로 기대되며, 향후 다양한 뇌 MR 이미지를 활용한 연구에도 좋은 영향을 미칠 것으로 보입니다.



### CultureVLM: Characterizing and Improving Cultural Understanding of Vision-Language Models for over 100 Countries (https://arxiv.org/abs/2501.01282)
Comments:
          Technical report; 26 pages

- **What's New**: 이 논문에서는 문화적 이해를 개선하기 위한 대규모 멀티모달 벤치마크 CultureVerse를 구축하였습니다. 이 데이터셋은 19,682개의 문화 개념과 188개의 국가/지역이 포함되어 있어 VLM(Visual-Language Models)이 다문화적 이해 능력을 평가할 수 있도록 합니다. 또한 CultureVLM이라는 일련의 VLM을 제안하여 우리의 데이터셋에 대해 파인튜닝을 통해 문화적 이해를 크게 향상시킵니다.

- **Technical Details**: CultureVerse는 19,682개의 문화 개념, 228,053개의 샘플로 구성되어 있으며, 다양한 문화적 지원을 위해 추가 언어와 문화를 통합할 수 있는 유연한 파이프라인을 제공합니다. VLMs는 일반적으로 문화적 이해의 지역적 불균형을 보여주며, 특히 서구 개념에 더 강하고 아프리카 및 아시아 맥락에서의 성능은 낮은 것으로 나타났습니다. 파인튜닝을 통해 문화적 인식을 향상시켜 모델의 일반적인 성능을 희생하지 않고도 지역과 범주에서의 이해 격차를 줄일 수 있습니다.

- **Performance Highlights**: 모델의 크기와 데이터의 양이 문화적 이해를 증가시키는데 긍정적인 상관관계를 보였으며, 특히 Llama 3.2-11B 모델이 Qwen 2-72B와 유사한 성능을 발휘했습니다. 결과는 VLM이 다양한 문화, 개념, 대륙 및 데이터셋 간의 일반화 능력을 보여주며, 문화적 이해를 향상시키기 위한 일반화 연구의 커다란 가능성을 나타냅니다. 이번 연구는 상황이 잘 반영되지 않은 문화에 대한 AI 형평성을 증진시키는 기반을 마련할 것을 기대합니다.



### Missing Data as Augmentation in the Earth Observation Domain: A Multi-View Learning Approach (https://arxiv.org/abs/2501.01132)
- **What's New**: 최근에 논의된 Multi-view Learning (MVL)을 통해 여러 데이터 소스나 보기를 활용하여 기계 학습 모델의 성능과 견고성을 향상시키려는 시도가 늘어나고 있습니다. 특히, 이 연구에서는 누락된 데이터에 대한 새로운 접근 방식을 제시하여 Earth Observation (EO) 분야에서 MVL 모델의 예측력을 더욱 높이고자 합니다. 이 방법은 누락된 뷰를 상정하여 다양한 조합을 훈련 샘플로 사용하고, 정적 데이터가 아닌 동적 병합 기능을 사용하여 누락된 뷰를 무시합니다.

- **Technical Details**: 이 접근 방식은 모든 누락된 뷰의 조합을 시뮬레이션하는 CoM(Combinations of Missing views) 기술을 사용합니다. 또한, 퓨전(fusion) 작용은 평균 또는 더 복잡한 함수인 Transformer 등의 동적 병합 함수를 통해 이루어집니다. 재미있는 점은, 누락된 뷰의 특징을 완전히 무시하며 동적 병합을 통해 MVL 모델이 데이터 예측을 근본적으로 개선할 수 있다는 것입니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 중간 정도의 정보 누락 상황에서도 모델의 견고성을 향상시키며, 모든 뷰가 존재할 때는 예측 성능을 개선하는 것으로 나타났습니다. 이들은 EO 분야에서의 최신 기법들과 비교하여 우수한 결과를 보여주며, 시간적 및 정적 뷰를 모두 아우르는 데이터셋을 기반으로 검증되었습니다. 이러한 방법들은 가용한 뷰의 조합에 대해 효과적으로 작동하는 단일 적응형 솔루션을 제공합니다.



### HoneypotNet: Backdoor Attacks Against Model Extraction (https://arxiv.org/abs/2501.01090)
Comments:
          Accepted to the AAAI 2025

- **What's New**: 본 논문에서는 "attack as defense"라는 새로운 방어 패러다임을 도입합니다. 기존의 방어 기법들과는 달리, 이 접근법은 공격자가 대체 모델을 훈련하는 것을 방해하여 대체 모델의 기능을 저해합니다. 또한, HoneypotNet이라는 경량 백도어 공격 방법을 제안하여, 대체 모델을 대상으로 한 공격 전략을 구현합니다.

- **Technical Details**: HoneypotNet은 희생 모델의 분류 레이어를 honeypot 레이어로 교체하고, 그림자 모델과 함께 이 레이어를 미세 조정하여 출력이 독성이 되도록 하는 방식입니다. Bi-Level Optimization (BLO) 프레임워크를 통해 이러한 독성 확률 벡터를 생성하며, 공격자들이 사용한 깨끗한 이미지에 숨겨진 유니버설 적대적 변동(Universal Adversarial Perturbation, UAP)을 적용합니다.

- **Performance Highlights**: 실험 결과에 따르면, HoneypotNet은 대체 모델에 백도어를 주입하는 데 56.99%에서 92.35%의 높은 성공률을 보입니다. 이 백도어는 모델 소유자에게 소유권 확인을 가능하게 할 뿐만 아니라, 대체 모델의 기능을 방해하여 모델 추출 공격에 대한 강력한 억지력을 제공합니다.



### Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction (https://arxiv.org/abs/2501.01087)
Comments:
          Submitted to IEEE Transactions on Emerging Topics in Computational Intelligence

- **What's New**: 이 논문에서는 다변량 시간 시계열 예측(multivariate Time Series Forecasting, TSF)을 위한 새로운 데이터 효율 아키텍처인 GLinear를 제안합니다. GLinear는 주기적 패턴을 활용하여 보다 높은 정확도를 제공하며, 기존 선형 예측기보다 적은 양의 과거 데이터로 더 좋은 예측 정확도를 보여줍니다. 특히, GLinear는 기존의 Transformer 기반 모델과 비교했을 때도 뛰어난 성능을 발휘합니다.

- **Technical Details**: GLinear는 복잡한 구성 요소나 블록(self-attention schemes, positional encoding blocks 등)을 포함하지 않는 간단한 모델입니다. 이 모델은 데이터 효율성을 중시하며 대량의 역사적 데이터에 의존하지 않고도 강력한 예측 성능을 입증합니다. 제안된 아키텍처는 다양한 데이터 입력 길이와 예측 구간에 따른 성능 영향을 평가하며, 일반적인 TSF 모델에서 발생하는 제한 사항을 극복하려는 노력을 보여줍니다.

- **Performance Highlights**: GLinear는 ETTh1, Electricity, Traffic, Weather라는 네 가지 데이터셋에서 기존의 선형 아키텍처(NLinear, DLinear, RLinear) 및 Transformer 기반 TSF 예측기(Autoformer)와의 성능 비교에서 대부분의 경우에 뛰어난 성능을 발휘했습니다. 이 연구는 GLinear가 데이터 및 계산적으로 효율적인 시간 시계열 분석을 위한 새로운 연구 및 개발의 전환점을 제공할 것이라는 기대를 나타냅니다.



### MSC-Bench: Benchmarking and Analyzing Multi-Sensor Corruption for Driving Perception (https://arxiv.org/abs/2501.01037)
- **What's New**: 이번 연구에서는 다중 센서 융합 모델의 안전성 및 강건성을 평가하기 위해 Multi-Sensor Corruption Benchmark (MSC-Bench)를 제시합니다. 이 벤치마크는 독특하게 16종의 감염 유형을 포함하여 카메라 및 LiDAR 입력의 이탈을 검토합니다. 연구 결과에 따르면, 3D 물체 감지 및 HD 맵 구축 모델 모두 센서 손상에 취약하다는 것이 드러났습니다.

- **Technical Details**: MSC-Bench는 환경적 요인에 따른 다양한 센서 고장을 평가하기 위해 16개의 감염 유형으로 구성되어 있습니다. 각 감염 유형은 날씨, 실내, 센서 고장 시나리오로 분류되며, 세부사항에는 Camera Crash, Frame Lost, Cross Sensor 등이 포함됩니다. 연구자들은 이러한 감염 시나리오가 자율주행 인식 시스템의 성능에 미치는 영향을 평가하였습니다.

- **Performance Highlights**: 결과적으로, 연구에서 분석된 6개의 3D 물체 감지 및 4개의 HD 맵 구축 모델은 예상보다 성능 저하가 크게 나타났습니다. 특히 악천후나 센서 실패와 같은 조건에서 모델의 강건성이 급격히 낮아지는 것을 확인하였습니다. 연구팀은 이러한 발견이 향후 센서 융합 모델의 신뢰성을 높이기 위한 설계 개선에 기여할 것으로 보았습니다.



### The Silent Majority: Demystifying Memorization Effect in the Presence of Spurious Correlations (https://arxiv.org/abs/2501.00961)
- **What's New**: 이 연구는 기계 학습 모델이 소수 그룹에서 불균형한 성능을 보이는 근본 원인인 스푸리어스 메모리(Spurious Memorization)에 대해 체계적으로 분석하였습니다. 저자들은 모델의 특정 뉴런이 소수 그룹 정보를 어떻게 기억하는지에 대한 최초의 실증적 증거를 제시하고, 이 메모리화가 불균형한 그룹 성능에 기여할 수 있음을 확인하였습니다. 이로 인해 스푸리어스 속성을 제거하는 새로운 프레임워크를 통해 모델의 소수 그룹에 대한 성능을 크게 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 두 가지 실험적 접근법을 통해 소수 그룹 정보 메모리화를 위한 중요한 뉴런의 존재를 입증했습니다. 첫 번째는 비구조적 추적(Unstructured Tracing)으로 뉴런의 역할을 전체 모델 수준에서 평가하고, 두 번째는 구조적 추적(Structured Tracing)으로 각 층의 뉴런 역할을 분석했습니다. 스푸리어스 메모리의 응집 성질을 확인하기 위해, 변형된 모델을 활용하여 주요 뉴런의 변화를 명확하게 이해하고, 이러한 뉴런들이 소수 그룹 성능에 미치는 영향을 면밀하게 조사했습니다.

- **Performance Highlights**: 이 실험 결과들은 소수 그룹의 성능이 특정 중요한 뉴런에 의해 크게 좌우되며, 이러한 뉴런들은 전체 모델 파라미터의 극히 일부를 차지한다는 것을 보여줍니다. 즉, 대다수 그룹의 경우, 전체 네트워크가 메모리를 형성해 Robust한 테스트 성능을 보이는 반면, 소수 그룹 예제는 제한된 뉴런 집합에 의존하여 Poor한 테스트 성능을 나타냅니다. 이로 인해 불균형한 성능 패턴에 대한 강력한 설명이 제공됩니다.



### Enhancing Early Diabetic Retinopathy Detection through Synthetic DR1 Image Generation: A StyleGAN3 Approach (https://arxiv.org/abs/2501.00954)
Comments:
          13 pages, 11 figures

- **What's New**: 이번 연구에서는 StyleGAN3를 활용하여 당뇨병성 망막병증(Diabetic Retinopathy, DR) 1단계를 재현한 합성 이미지를 생성하였습니다. 이러한 이미지들은 미세 혈관 팽창증(microaneurysms)을 특징으로 하며, 높은 충실도(fidelity)와 다양성을 자랑합니다. 이는 고품질의 fundus images 부족 문제를 해결하고, 감독(Classifier) 학습 알고리즘의 성능 향상에 기여하고자 합니다.

- **Technical Details**: 연구팀은 2,602개의 DR1 이미지를 사용하여 모델을 훈련시키고, Frechet Inception Distance (FID), Kernel Inception Distance (KID), 그리고 변환에 대한 동등성(EQ-T) 및 회전에 대한 동등성(EQ-R)과 같은 정량적(quantitative) 지표를 통해 평가하였습니다. 또한, 숙련된 안과 의사들이 합성 이미지의 현실성을 평가하는 'Human Turing test'를 포함한 질적(qualitative) 평가를 진행했습니다. 스펙트럴 분석(spectral analysis) 또한 이미지 품질을 검증하는 데 기여하였습니다.

- **Performance Highlights**: 모델은 최종 FID 점수를 17.29로 기록하였으며, 이는 부트스트랩 재샘플링을 통해 도출된 평균 FID 점수 21.18(95% 신뢰구간 - 20.83 ~ 21.56)보다 우수한 결과입니다. Human Turing test 결과, 모델이 매우 현실적인 이미지를 생성할 수 있는 능력을 입증하였으나, 이미지의 경계 근처에 미세한 아티팩트가 발견되었습니다. 이러한 결과는 StyleGAN3가 생성한 합성 DR1 이미지가 데이터 세트를 보강하고, 당뇨병성 망막병증의 조기 발견을 더욱 정확하게 할 수 있도록 도와줄 잠재력이 크다는 것을 시사합니다.



### Efficient Unsupervised Shortcut Learning Detection and Mitigation in Transformers (https://arxiv.org/abs/2501.00942)
- **What's New**: 이번 연구에서는 딥 러닝 모델의 단축 학습(shortcut learning) 문제를 해결하기 위한 새로운 비지도 프레임워크를 제안합니다. 이 프레임워크는 트랜스포머(transformer) 모델의 단축 학습을 탐지하고 완화하는 능력을 갖추고 있으며, 여러 데이터 세트에서 방법을 검증합니다. 특히, 이 프레임워크는 사용자가 적은 인력이 투입되도록 하면서도 인식 가능한 패턴의 발견과 다모달 대형 언어 모델(MLLM)의 해석 기능을 활용합니다.

- **Technical Details**: 프레임워크는 모델의 활성화를 두 단계로 분석하며, 각 클러스터의 대표 샘플을 선택하여 클러스터 구성을 제공합니다. 주요 혁신은 프로토타입 패치를 식별하는 과정으로, 이 과정에서 주요 공간에서의 토큰 거리 분석을 통해 영향력 있는 영역을 정밀하게 찾아냅니다. 또한, MLLM을 사용하여 프로토타입 패치에 대한 캡션을 생성하고 클러스터 내 패치 요약을 제공하여 전문적인 검토를 지원합니다.

- **Performance Highlights**: 우리의 접근 방식은 최악의 그룹 정확도와 평균 정확도를 모두 개선하면서도 계산의 효율성을 유지합니다. 사용자 연구를 통해 탐지된 프로토타입 개념이 데이터 내의 단축 특성을 올바르게 식별할 수 있는 인간 친화적인 인사이트를 제공함을 확인했습니다. 프레임워크는 기존 최첨단 기술과 경쟁력 있는 성능을 달성하며, 다양한 단축 완화 기법과 쉽게 통합될 수 있어 연구자들에게 유용한 도구가 될 것으로 기대됩니다.



### A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Datas (https://arxiv.org/abs/2501.00941)
Comments:
          Accepted at AAAI 2025. This is the preprint version. Keywords: Multi-modal generation, diffuison models, scientific data generation, unbalanced modalities

- **What's New**: 본 논문은 UB-Diff라는 혁신적인 diffusion 모델을 제안하여, 지구 물리학에서의 다중 모달 과학 데이터 생성을 위한 새로운 접근 방식을 제공합니다. 이 모델은 쌍을 이루는 다중 모달 데이터를 생성하기 위해 Co-latent representation을 활용하며, 이는 현실 세계에서 흔히 발생하는 데이터 불균형 문제를 효과적으로 해결합니다. 실험 결과는 UB-Diff가 기존 기술보다 월등한 성능을 보여주며, 신뢰할 수 있는 다중 모달 데이터를 생성할 수 있음을 증명합니다.

- **Technical Details**: UB-Diff는 인코더-디코더 구조를 기반으로 하는 모델로, 두 개의 독립적인 디코더를 통해 쌍을 이루는 데이터를 생성합니다. 이는 복잡한 과학적 맥락에서 발생하는 다양한 데이터 유형을 연결하는 데 중요한 역할을 합니다. 본 연구는 높은 품질의 쌍 데이터 생성을 위한 매칭된 훈련 방식을 채택하여, 불균형 데이터를 효과적으로 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, UB-Diff는 Fréchet Inception Distance (FID) 점수와 쌍 평가에서 기존 모델들을 능가하여, 불균형 데이터 상황에서도 우수한 성능을 발휘합니다. 이는 지구 물리학 연구에서 데이터 생성 작업에 대한 새로운 가능성을 열어주며, 다양한 과학적 응용 분야에서의 활용 가능성을 보여줍니다. 또한, UB-Diff는 기존의 state-of-the-art 모델들보다 뛰어난 성능을 입증하며, 쌍 데이터 생성을 위한 유용한 솔루션이 될 것입니다.



### A Novel Approach using CapsNet and Deep Belief Network for Detection and Identification of Oral Leukopenia (https://arxiv.org/abs/2501.00876)
Comments:
          Accepted to IEEE International Conference on Advancement in Communication and Computing Technology (INOACC), will be held in Sai Vidya Institute of Technology, Bengaluru, Karnataka, India. (Preprint)

- **What's New**: 이 연구는 구강암(oral cancer)의 조기 진단을 위한 자동화된 탐지 시스템을 구축하기 위한 새로운 접근 방식을 제안합니다. 특히, 다양한 의사들의 바운딩 박스 주석(bounding box annotations)을 통합하여 정확한 구강 병변(oral lesions) 분류를 시도하였습니다. 2023년 전 세계에서 277,484명의 사망자가 발생한 점을 감안할 때, 저소득 및 중간소득 국가에서의 높은 유병률(prevalence)을 반영하여 이 연구는 매우 의미가 있습니다.

- **Technical Details**: 이 연구에서는 Deep Belief Network와 CAPSNET를 결합하여 구강 병변의 자동 탐지 및 분류를 위한 시스템을 개발했습니다. 이미지 분류(image classification) 작업에서 CAPSNET을 이용하여 병변이 있는 사진을 탐지할 때 94.23%의 F1 점수를 기록하였고, 의뢰가 필요한 이미지를 식별할 때는 93.46%의 F1 점수를 달성했습니다. 또한, 객체 탐지(object detection)에서는 의뢰를 위한 병변 식별에서 89.34%의 F1 점수를 보였습니다.

- **Performance Highlights**: 초기 연구 결과에 따르면, 딥 러닝(deep learning) 기술이 복잡한 구강암 탐지 문제를 효과적으로 해결할 수 있는 가능성을 보여줍니다. 분류 결정의 유형에 따라 문서화된 후속 성능이 있으며, 이러한 결과는 조기 탐지를 통한 비용 효율적인 진단 가능성을 제안합니다. 이는 임상적 응용에서도 매우 중요한 성과로 평가될 수 있습니다.



### HCMA-UNet: A Hybrid CNN-Mamba UNet with Inter-Slice Self-Attention for Efficient Breast Cancer Segmentation (https://arxiv.org/abs/2501.00751)
- **What's New**: 이번 연구는 유방암의 병변 세분화에서의 도전 과제를 해결하기 위해 새로운 하이브리드 세분화 네트워크인 HCMA-UNet을 제안합니다. 이 네트워크는 경량의 CNN 백본과 멀티뷰 인터슬라이스 셀프 어텐션 Mamba (MISM) 모듈로 구성되어 있습니다. MISM 모듈은 VSSB와 ISSA 메커니즘을 통합하여 비대칭 분할 채널 전략을 통해 매개변수를 효과적으로 줄입니다.

- **Technical Details**: HCMA-UNet은 인코더-디코더 구조를 따르며, 인코더는 Res Block 및 MISM과 Dense Block으로 구성된 Hybrid Block을 통합합니다. MISM은 3D 볼륨을 세 개의 방향(시상, 관상, 축 방향)으로 재슬라이스하여 다중 뷰 분석을 통해 포괄적인 공간 상관 관계를 캡처합니다. ISSA는 서로 다른 슬라이스 간의 관계를 학습하여 중요한 해부학적 상관 관계를 보존합니다.

- **Performance Highlights**: 결과는 HCMA-UNet이 세 가지 데이터셋에서 우수한 성능을 보여주며, FRLoss가 뛰어난 세분화 정확도와 강력한 크로스 아키텍처 일반화를 입증했습니다. 이 연구는 DCE-MRI에서의 유방암 병변 세분화의 새로운 기준을 세울 것으로 기대됩니다.



### Automatic Construction of Pattern Classifiers Capable of Continuous Incremental Learning and Unlearning Tasks Based on Compact-Sized Probabilistic Neural Network (https://arxiv.org/abs/2501.00725)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문은 확률적 신경망(probabilistic neural network) 모델을 이용한 새로운 패턴 분류 기법을 제안합니다. 제안된 전략은 연속적인 증분 학습(incremental learning)과 비학습(unlearning) 작업을 수행할 수 있는 소형화된 확률적 신경망으로 구성됩니다. 하이퍼파라미터 튜닝 없이 단순한 데이터 기반 업데이트 방식을 사용하여 네트워크의 구조와 매개변수를 자동으로 결정합니다.

- **Technical Details**: 확률적 신경망(PNN)은 세 개의 층으로 구성되어 있으며, 숨겨진 층의 RBF(radial basis function) 유닛과 선형 출력 층 유닛을 포함합니다. 기존의 PNN에 비해 제안된 소형화된 PNN(CS-PNN)은 학습 데이터의 양과 관계없이 동적으로 알고리즘 구조를 조정할 수 있습니다. 이러한 방식은 반복적 매트릭스 기반 파라미터 근사화 없이 간단한 알고리즘으로 구성되며, 학습과 평가 모드에서의 속도와 과적합 문제를 해결합니다.

- **Performance Highlights**: 아홉 개의 공개 데이터베이스를 사용한 시뮬레이션 결과, 제안된 접근 방식이 원래의 PNN 모델에 비해 훨씬 적은 수의 숨겨진 유닛을 가짐에도 불구하고 다층 퍼셉트론 신경망(multilayer perceptron neural network)과 유사한 분류 성능을 달성할 수 있음을 보여줍니다. 또한, 계속적인 클래스 증분 학습과 비학습 작업에서도 충분한 성능을 발휘함을 나타냅니다.



### Deeply Learned Robust Matrix Completion for Large-scale Low-rank Data Recovery (https://arxiv.org/abs/2501.00677)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2110.05649

- **What's New**: 이 논문에서는 대규모 Robust Matrix Completion (RMC) 문제를 해결하기 위한 새로운 방법인 Learned Robust Matrix Completion (LRMC)를 제안합니다. LRMC는 학습 가능한 비볼록(non-convex) 접근 방식을 사용하여 누락된 데이터와 극단적인 이상치(outlier)를 동시에 처리할 수 있습니다. 이 방법은 선형 수렴(linear convergence)을 가지며, 깊은 펼치기(deep unfolding)를 통해 최적의 성능을 달성할 수 있는 자유 변수를 효과적으로 학습할 수 있습니다.

- **Technical Details**: LRMC는 고전적인 반복적 알고리즘을 파라미터화하고 퍼셉트론 기반의 피드포워드 신경망(feedforward neural network, FNN)으로 펼쳤습니다. 이 신경망의 특정 파라미터는 역전파(backpropagation)를 통해 학습될 수 있습니다. 제안된 네트워크는 고정 반복(iteration) 대신 무한 반복이 가능하여 유연성을 갖추고 있습니다.

- **Performance Highlights**: LRMC는 합성 데이터셋(synthetic datasets) 및 실제 애플리케이션에서 고도의 성능을 입증했습니다. 관련된 응용으로는 비디오 배경 제거(video background subtraction), 초음파 이미징(ultrasound imaging), 얼굴 모델링(face modeling), 위성 이미지에서의 클라우드 제거(cloud removal)가 포함됩니다. 이러한 결과들은 LRMC의 탁월한 효율성과 안정성을 입증합니다.



### Leaf diseases detection using deep learning methods (https://arxiv.org/abs/2501.00669)
Comments:
          252 pages , 42 images

- **What's New**: 이번 연구는 식물 잎 질병 식별 및 감지를 위한 새로운 딥러닝(deep learning) 접근법을 개발하는 데 중점을 두고 있습니다. 우리는 현재의 잎 질병 감지 방법이 직면한 도전 과제를 논의하고, 딥러닝이 이러한 문제를 극복하고 질병 탐지의 정확성을 향상시키는 데 어떻게 사용될 수 있는지를 살펴보았습니다.

- **Technical Details**: 이에 따라, 우리는 다양한 작물의 잎 질병을 탐지하기 위한 새로운 방법론을 제안하였으며, 하이퍼파라미터(hyperparameters)와 최적화 방법을 포함하는 효율적인 네트워크 아키텍처(architecture)를 소개하였습니다. 또한 여러 아키텍처의 효율성을 비교하고 평가하여 최적의 아키텍처 구성을 찾아 고속 질병 탐지가 가능한 효과적인 모델을 만들었습니다.

- **Performance Highlights**: 본 연구에서는 사전 학습(pre-trained) 모델에 대한 작업 외에도, CNN(Convolutional Neural Network) 기반의 새로운 모델을 제안하여 식물 잎 질병 식별 및 감지의 효율적인 방법을 제공합니다. 또한, 우리의 모델의 효율성을 평가하고, 일부 최신 사전 학습 아키텍처와 결과를 비교하였습니다.



### Lightweight G-YOLOv11: Advancing Efficient Fracture Detection in Pediatric Wrist X-rays (https://arxiv.org/abs/2501.00647)
- **What's New**: 이번 연구에서는 기존의 CAD 시스템의 제한점을 극복하기 위해 경량화된 G-YOLOv11(CAD 시스템의 이름)을 제안합니다. G-YOLOv11은 최신 YOLOv11 버전을 기반으로 한 fracture detection에 특화된 시스템으로, ghost convolution을 활용하여 기능 추출에서의 효율성을 높였습니다. 기존의 대규모 감지기 대신에 경량화 모델을 사용함으로써 임상 환경에서의 실용성을 향상시키고자 합니다.

- **Technical Details**: G-YOLOv11 모델은 ghost convolution operation을 적용하여 계산 요구조건을 상당히 줄였습니다. 이 방법은 전통적인 convolution보다 더 적은 연산으로 동일한 수의 feature maps를 생성할 수 있으므로 계산 복잡성을 줄이는 데 효과적입니다. 실험 결과, G-YOLOv11은 NVIDIA A10 GPU에서 mAP@0.5 0.535 및 2.4 ms의 추론 시간을 기록했습니다.

- **Performance Highlights**: G-YOLOv11 모델은 기존의 YOLOv11 모델 대비 13.6%의 mAP@0.5 감소 및 68.7%의 크기 감소를 달성했습니다. 이러한 결과는 효율성 측면에서 새로운 최고 성능 기준을 설정하며, 기존 감지기들을 초월하는 성능을 보여줍니다. 이는 CAD 시스템에서 경량화된 감지기 개발의 필요성을 잘 나타냅니다.



### Applying Graph Explanation to Operator Fusion (https://arxiv.org/abs/2501.00636)
Comments:
          DAC'23 WIP Poster; 8 pages, 5 Figures 5 Tables

- **What's New**: 이 논문은 Explainable AI의 Graph Explanation Techniques (GET)를 레이어 융합(layer fusion) 최적화에 통합하는 새로운 접근 방식을 제안합니다. 기존 레이어 융합 방식에서 발생할 수 있는 유효하지 않은 융합 그룹을 식별하고, 이를 해결하기 위한 재귀적 최적화 방안을 모색합니다. 이를 통해 DRAM 접근 비용을 최소화하여 DNN의 추론 효율성을 높이는 데 기여합니다.

- **Technical Details**: 논문에서 제안하는 구조는 유효하지 않은 융합 그룹의 유효성을 바이너리 분류 문제로 환원하여, 해당 그룹을 구성하는 노드와 엣지를 분석하는 방식입니다. 여러 Graph Explanation Techniques를 통해, 융합 그룹의 무효화를 초래하는 주요 연산을 식별하고, 이 정보를 이용하여 적절히 그룹을 분할합니다. 특히, Line-Buffer Depth First (LBDF) 및 Buffer Requirement Reduction (BRR)이라는 두 가지 레이어 융합 방식을 통해 DNN을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 EfficientNet-B3 모델에서 20% 이상의 DRAM 접근 감소를 달성하였습니다. ResNets 및 MobileNets와 같은 다양한 고전 및 현대의 합성곱 신경망 구조에서 융합 그룹을 최적화하여 성능을 크게 향상시키는 양상을 보였습니다. 이러한 결과는 제안한 방법이 여러 모드의 DNN 디자인과 다양한 검색 알고리즘과 결합될 때 유효성을 입증하고 있음을 나타냅니다.



### Advanced Lung Nodule Segmentation and Classification for Early Detection of Lung Cancer using SAM and Transfer Learning (https://arxiv.org/abs/2501.00586)
- **What's New**: 이번 연구는 폐암 진단을 위한 혁신적인 방법으로 Segment Anything Model(SAM)과 Transfer Learning 기법을 활용하여 폐 결절을 정밀하게 분할(segmentation)하는 접근법을 소개합니다. 기존의 기술들과 비교하여 SAM 모델이 Transfer Learning과 결합하여 97.08%의 Dice Similarity Coefficient(DSC)와 95.6%의 Intersection over Union(IoU)를 달성했다는 점이 두드러집니다. 이는 폐암 조기 발견을 위한 정확한 진단을 위한 중요한 발전을 보여줍니다.

- **Technical Details**: 연구에서는 CT(Computed Tomography) 및 MRI 이미지를 이용하여 Machine Learning(ML) 및 Convolution Neural Network(CNN) 기반의 Deep Learning(DL) 기법으로 폐 결절을 효과적으로 분할하고 분류합니다. SAM 모델을 활용하여 Bound Box 프롬프트와 비전 트랜스포머 모델을 결합하여 성능을 향상시킵니다. 이러한 기술적 접근은 컴퓨터 보조 진단 시스템(CAD)을 큰폭으로 개선하여 폐암 진단에 기여합니다.

- **Performance Highlights**: 연구 결과, 제안한 모델이 CT 스캔에서 폐 결절을 명확히 분할하는 데 성공했으며, 96.71%의 분류 정확도를 기록했습니다. 이는 폐암 진단의 조기 발견과 환자 치료를 향상시킬 수 있는 가능성을 보여줍니다. 이러한 성과는 기존의 분할 및 분류 기법에 비해 상당한 향상을 이룬 것으로 평가됩니다.



### H-Net: A Multitask Architecture for Simultaneous 3D Force Estimation and Stereo Semantic Segmentation in Intracardiac Catheters (https://arxiv.org/abs/2501.00514)
- **What's New**: 이번 연구에서는 카테터에서 두 개의 서로 다른 각도에서 동시에 분할(segmentation)하고 3D에서 적용된 힘(force)을 추정할 수 있는 새로운 경량 다중 입력-다중 출력 인코더-디코더 기반 아키텍처를 제안합니다. 이 아키텍처는 비싼 센서를 사용하지 않고, 두 개의 X-레이 이미지를 동시에 처리하여 카테터의 변위를 보여주며, 따라서 카테터의 시각화와 힘 측정을 동시에 지원할 수 있습니다. 기존의 연구와는 달리, 이 모델은 분산형 네트워킹을 사용하여 연산 자원을 최적화하며, 종합적으로 상호작용된 힘과 분할 성능을 높였습니다.

- **Technical Details**: 제안된 H-Net 아키텍처는 두 개의 X-레이 이미지를 입력으로 받아 각각의 이미지에서 추출된 특징 맵을 통합하여 카테터의 3D 구조를 인식합니다. 이 시스템은 두 개의 병렬 인코더와 디코더를 활용하여 두 개의 분할 헤드와 하나의 힘 추정 헤드를 통해 작동합니다. 특히, 힘 추정 헤드는 카테터의 꼭지점에서의 힘을 x, y, z 방향으로 예측하며, 이는 센서가 없는 방식으로 작동하기 때문에 생산 비용을 절감할 수 있습니다. 또한, 이 아키텍처는 효율성을 위해 네트워크 간의 매개변수를 공유하여 계산 복잡성을 최소화합니다.

- **Performance Highlights**: 제안된 H-Net은 기존의 선구적인 방법들과 비교해 최초로 3D 힘 추정과 스테레오 분할 작업을 동시에 처리하며 뛰어난 성능을 보여주었습니다. 실험 결과, 합성된 데이터셋 뿐만 아니라 실제 RGB 데이터셋에서도 탁월한 성능을 입증하였으며, 다중 태스크 시스템이 적용된 걸로 인해 카테터 분할 및 힘 추정의 신뢰성이 증가했습니다. 따라서 H-Net은 기존의 모델들과 차별화된 우수한 성능을 바탕으로 심혈관 시술에서의 안전성과 효율성을 높이는 데 기여할 것으로 기대됩니다.



### SAT-LDM: Provably Generalizable Image Watermarking for Latent Diffusion Models with Self-Augmented Training (https://arxiv.org/abs/2501.00463)
Comments:
          24 pages, 7 figures

- **What's New**: 본 연구는 Latent Diffusion Models를 위한 새로운 이미지 워터마킹 방법인 Self-Augmented Training (SAT-LDM)을 소개합니다. SAT-LDM은 훈련과 테스트 단계 사이의 일반화 능력을 강화하기 위해 자유 생성 분포를 활용합니다. 이 방법은 새로운 데이터를 수집하지 않고도 다양한 프롬프트에서 잘 작동할 수 있는 이론적 기반을 제공합니다.

- **Technical Details**: SAT-LDM은 기존 데이터 세트를 사용하지 않고 내부적으로 생성된 자유 생성 분포를 사용하여 워터마킹 모듈을 훈련합니다. 이 접근 방식은 이미지 생성 모델이 작동하는 자연 조건과 밀접하게 일치하여 다양한 프롬프트에 대한 일반화 능력을 보장합니다. 또한, 자유 생성 분포가 훈련 및 테스트 단계 간의 배포 차이를 줄이는 데 기여한다는 것을 이론적으로 입증하였습니다.

- **Performance Highlights**: SAT-LDM은 다양한 프롬프트에 대한 견고한 워터마킹과 높은 품질의 이미지를 제공합니다. 실험 결과, SAT-LDM은 이전 방법들과 유사한 높은 견고성을 유지하면서도 워터마킹된 이미지의 품질을 크게 향상시키는 것으로 나타났습니다. 본 연구는 AI 생성 콘텐츠 보호를 위한 실용적이고 편리한 해결책을 제공할 것으로 기대됩니다.



### Differentiable Prompt Learning for Vision Language Models (https://arxiv.org/abs/2501.00457)
- **What's New**: 이번 논문에서는 기존의 수동적 프롬프트 디자인을 자동화하기 위한 새로운 방법인 differentiable prompt learning (DPL)을 제안합니다. DPL은 최적의 프롬프트 길이를 자동으로 결정하는 최적화 문제로 설정되며, 이를 통해 성능을 최대화하는 것을 목표로 합니다. DPL 방법은 사전 훈련된 CLIP 모델에 적용되어, 기존 방법들보다 높은 신뢰도로 딥 연속 프롬프트 구성 파라미터를 찾을 수 있음을 입증하였습니다.

- **Technical Details**: DPL 방법은 연속 프롬프트의 컨텍스트 길이와 깊이를 자동으로 결정하는 데 초점을 맞추고 있습니다. DPL은 최적화 과정에서 각 레이어에 추가될 프롬프트의 깊이와 컨텍스트 길이를 조정함으로써, 프롬프트 학습의 유연성을 높입니다. 또한, DPL 방법은 수동 설계를 거치지 않고도 적은 데이터만으로 성능을 극대화할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: DPL 방법은 다운스트림 과제에서 평균 테스트 정확도를 11개 데이터셋에서 기존 방법보다 2.60% 향상시키는 성능을 보였습니다. 이 방법은 기존의 복잡한 설계와도 호환 가능하여, 최적의 딥 연속 프롬프트 구성을 데이터셋에 맞춰 조정함으로써 성능 향상이 가능합니다. 따라서 DPL 방법은 각기 다른 모델에 대해 비용 없이 쉽게 배포할 수 있는 장점을 가집니다.



### STARFormer: A Novel Spatio-Temporal Aggregation Reorganization Transformer of FMRI for Brain Disorder Diagnosis (https://arxiv.org/abs/2501.00378)
- **What's New**: 이번 연구에서는 전통적인 진단 방식에서 벗어나 뇌 기능 네트워크의 공간 및 시간 정보를 모두 효과적으로 capture하는 새로운 변환기 모델인 STARFormer를 제안합니다. 기존의 fMRI 분석 방법들이 간과하는 BOLD 신호의 공간적 및 시간적 의존성을 통합하여, 자폐 스펙트럼 장애(ASD) 및 주의력 결핍 과다 행동 장애(ADHD)의 진단 정확도를 높입니다. STARFormer는 세 가지 핵심 모듈을 통해 이러한 기능을 구현하며, 이는 공간 및 시간 정보를 효과적으로 재구성하고 융합하는 데 중요한 역할을 합니다.

- **Technical Details**: STARFormer는 두 개의 주요 가지 구조로 설계되어 있습니다. 첫 번째는 영역의 공간 구조를 분석하여 유의미한 뇌 영역의 상관성을 강조하기 위해 고유벡터 중심성(eigenvector centrality, EC)을 적용하는 모듈입니다. 두 번째는 다기준 윈도우 전략을 통해 로컬 및 글로벌 시간적 패턴을 capture하는 시간적 특성 재구성 모듈을 포함합니다. 또한, 시공간 특성 융합 모듈은 병렬 transformer 구조를 사용하여 통합된 특성을 추출합니다.

- **Performance Highlights**: STARFormer는 ASD와 ADHD 분류를 위해 공개된 두 개의 데이터 세트에서 rigorously 평가되었습니다. 실험 결과, STARFormer는 기존 기법들과 비교하여 여러 평가 지표에서 최첨단 성능을 달성하며, 뇌 장애 진단 및 생물 의학적 연구를 위한 보다 정확하고 신뢰할 수 있는 도구로 자리잡고 있습니다. 이번 연구는 특히 fMRI 데이터에서 공간적 및 시간적 특징들을 효과적으로 capture함으로써 진단의 정확성 및 효율성을 획기적으로 향상시켰습니다.



### Predicate Invention from Pixels via Pretrained Vision-Language Models (https://arxiv.org/abs/2501.00296)
Comments:
          Workshop on Planning in the Era of LLMs (LM4Plan @ AAAI 2025)

- **What's New**: 이번 연구에서는 기존의 모델로부터 부족한 샘플만으로도 이미지의 원시 센서 입력에서 직접 작동하는 predicates를 발명하는 새로운 접근 방식을 제안합니다. 최근의 비전-언어 모델(Vision-Language Models, VLMs)의 능력을 활용하여, VLMs가 제안할 수 있는 predicates를 통해 로봇의 의사결정 문제를 해결하는 방법을 모색하고 있습니다. 이 방법은 본 연구에서 소개되는 pix2pred라는 프레임워크를 통해 실험적으로 입증되었습니다.

- **Technical Details**: 연구에서는 PDDL(Planning Domain Definition Language) 모델을 학습하여 개념을 추상화하며, 특히 Sparse한 데이터에서 추상적 세계 모델을 학습합니다. VLM을 활용하여 기초적인 predicates를 발명하고, 이들을 통해 실제 문제에 대한 의사결정을 지원합니다. 이 과정에는 기존의 predicates와 동작들을 조합하여 효과적인 의사결정을 위한 최적화를 수행하는 방법이 포함됩니다.

- **Performance Highlights**: pix2pred 접근법은 시뮬레이션된 두 가지 로봇 환경에서 네 가지 작업을 수행하며, 새로운 문제 인스턴스에서 항상 가장 높은 성공률을 기록했습니다. 또한, 더 많은 객체와 복잡한 목표, 그리고 훈련 중 보여준 것보다 긴 작업 수순을 포함하는 새로운 문제에 대해 효과적으로 일반화하는 기초 모델을 학습할 수 있음을 보여줍니다. 실험을 통해 제안된 방법이 모델-프리 모방 학습보다 더 높은 성과를 내는 것을 확인했습니다.



### Outlier-Robust Training of Machine Learning Models (https://arxiv.org/abs/2501.00265)
- **What's New**: 본 논문은 기계 학습 모델을 학습할 때 이상치(outlier)가 존재하는 경우에 대한 두 가지 상이한 강건 손실 함수 설계 방법론을 소개합니다. 특히, 로봇 공학 및 컴퓨터 비전에서 사용되는 M-추정(M-estimation) 프레임워크와 딥 러닝에서 주로 활용되는 리스크 최소화(risk minimization) 프레임워크를 대조합니다. 새로운 방법론으로 제안된 수정된 Black-Rangarajan 이중성(duality)을 통해 두 접근 방식을 통합하는 방안을 제시합니다.

- **Technical Details**: 제안된 수정된 이중성은 강건 손실 커널(robust loss kernel) 정의를 밝혀내며, 이것은 두 문헌에서 사용되는 강건 손실이 만족하는 특성을 나타냅니다. 또한, Adaptive Alternation Algorithm (AAA)을 통해 비강건 손실의 가중치 버전을 사용하는 반복적 학습 기법을 도입하여 각 반복마다 가중치를 업데이트합니다. 이 방법은 이상치에 대한 처리에서 복잡한 파라미터 조정(parameter tuning)을 줄이는 신규 파라미터 업데이트 규칙을 포함합니다.

- **Performance Highlights**: 본 논문에서 제안한 알고리즘은 회귀(regression), 분류(classification), 신경 장면 재구성(neural scene reconstruction) 문제에서 효과성을 입증하였습니다. 실험 결과는 강건 손실 커널을 사용할 때 이상치가 없는 최적값으로의 수렴 구역을 증가시킨다는 것을 보여줍니다. 제안된 AAA 알고리즘은 기본적인 신뢰성을 갖추고 있어, 다양한 도메인에서 이상치 영향을 줄이며 기계 학습 모델의 성능을 향상시킬 수 있습니다.



### TrajLearn: Trajectory Prediction Learning using Deep Generative Models (https://arxiv.org/abs/2501.00184)
- **What's New**: 이 논문은 TrajLearn이라는 새로운 모델을 제안하며, 이를 통해 복잡한 공간적 의존성을 관리하고 동적인 환경에 적응할 수 있는 점이 혁신적입니다. 이 모델은 육각형 공간 표현을 기반으로 한 고차원 이동 흐름의 생성 모델링을 활용하여 미래 경로를 예측합니다. 특히, 여러 후보 경로를 동시에 탐색할 수 있도록 맞춤형 비임 탐색(customized beam search)을 통합하여 공간 연속성을 유지하는 방법론을 채택했습니다.

- **Technical Details**: TrajLearn은 Transformer 아키텍처를 기반으로 하여 설계된 경로 생성 모델로, 높은 차원의 이동 흐름 데이터로부터 학습합니다. 이 모델은 입력으로 주어진 최근의 궤적 내역에 따라 미래의 궤적을 예측하는 SEQUENCE PREDICTION 문제를 형식화합니다. 또한, 하위 해상도 영역을 계층적으로 세분하여 혼합 해상도의 맵을 생성하는 알고리즘을 개발하여 다양한 궤적 분석 시나리오에 적합하게 설계되었습니다.

- **Performance Highlights**: TrajLearn 모델은 실험 결과, 최신 방법들과 유의미한 기준선에 대해 약 40%의 성능 향상을 달성했습니다. 이 모델은 다양한 실제 경로 데이터셋에 대한 평가 지표에서 뛰어난 성능을 보였으며, 예측 수평에 따른 다양한 분석을 통해 파라미터 민감도 및 모델 구성 요소의 영향을 연구했습니다. 코드와 모델을 오픈 소스로 공개하여 재현성을 높이고 다양한 응용 프로그램에 적용할 수 있는 유연한 구성 옵션을 제공합니다.



### VisTabNet: Adapting Vision Transformers for Tabular Data (https://arxiv.org/abs/2501.00057)
- **What's New**: 이 논문에서는 VisTabNet이라는 새로운 크로스 모달 전이 학습 방법을 제안하여, 사전 훈련된 Vision Transformer(ViT) 가중치를 사용하여 표 형식의 데이터를 처리할 수 있도록 합니다. 기존의 전이 학습 관행을 넘어 사전 훈련된 이미지 모델을 이용하여 표 데이터 문제를 해결하는 가능성을 보여줍니다. 이 접근 방식은 새로운 아키텍처를 설계하는 데 드는 개념적 비용을 줄이고 모델을 처음부터 훈련시키는 계산 비용을 감소시킵니다.

- **Technical Details**: VisTabNet은 표 형식 입력을 ViT에서 수용 가능한 패치 임베딩으로 투영함으로써, 사전 훈련된 Transformer Encoder를 표 형식 입력에 직접 적용할 수 있게 합니다. 이는 고차원 이상에서의 패턴을 이해하고 효율적으로 처리할 수 있게 해줍니다. 이 방식은 머신러닝에서 가장 일반적으로 사용되는 표 형식 데이터의 복잡성을 극복하기 위해 개발되었습니다.

- **Performance Highlights**: VisTabNet은 여러 개의 소규모 표 형식 데이터셋에서 실험적으로 검증되어 전통적인 앙상블 방법과 최근의 깊은 학습 모델보다 우수한 성과를 보였습니다. 특히, 소수의 샘플로도 효과적으로 전이 학습이 이루어지며, 이는 크로스 모달 전이가 깊은 네트워크를 처음부터 훈련하는 것보다 더 효과적임을 증명합니다. 본 연구는 표 형식 데이터에 대한 깊은 학습 접근을 개선할 수 있는 가능성을 제시합니다.



New uploads on arXiv(cs.AI)

### Rethinking Relation Extraction: Beyond Shortcuts to Generalization with a Debiased Benchmark (https://arxiv.org/abs/2501.01349)
- **What's New**: 이번 논문에서는 관계 추출(Relation Extraction) 알고리즘에서의 엔티티 편향(entity bias) 문제를 다루고 있습니다. 최근 머신 러닝 모델들이 엔티티 언급(entity mentions) 대신 문맥을 무시하고 관계 유형을 예측하는 경향이 있으며, 이는 잘못된 모델 평가로 이어질 수 있습니다. 새로운 디바이즈 관계 추출 벤치마크인 DREB를 제안하여, 엔티티 교체를 통해 엔티티 언급과 관계 유형 간의 의사 상관(pseudo-correlation)을 끊어내고자 합니다.

- **Technical Details**: DREB는 Bias Evaluator 및 PPL Evaluator를 활용하여 낮은 편향과 높은 자연성을 보장합니다. 새로운 기준선(Baseline)을 마련하기 위해 Data-level과 Model-level 기술을 결합한 MixDebias 방법을 도입하였습니다. MixDebias는 데이터 증강(data augmentation)과 샘플 편향 분석(bias assessment)을 통해 모델을 최적화하며, KL divergence를 사용하여 확률 분포를 정렬하는 방식으로 작동합니다.

- **Performance Highlights**: 광범위한 실험 결과, MixDebias는 DREB에서 모델 성능을 유의미하게 향상시키는 데 성공했습니다. 또한, 기존 데이터셋에서도 안정적인 성능을 유지합니다. DREB와 MixDebias의 공개 발표는 관계 추출 모델의 일반화 능력 향상을 위한 중요한 기틀이 될 것으로 기대됩니다.



### DeepFilter: An Instrumental Baseline for Accurate and Efficient Process Monitoring (https://arxiv.org/abs/2501.01342)
- **What's New**: 이번 논문에서는 산업 자동화에서 프로세스 모니터링의 정확성과 효율성을 개선하기 위해 새로운 아키텍처인 DeepFilter를 제안합니다. DeepFilter는 전통적인 Transformer의 self-attention 계층 대신 글로벌 필터링 블록을 도입하여 오랜 시간 동안의 패턴을 효과적으로 포착하도록 설계되었습니다. 이를 통해 기존의 한계를 극복하고 산업용 모니터링 시스템에서 요구되는 높은 신뢰성과 성능을 달성합니다. 실험 결과, DeepFilter는 기존의 최신 모델에 비해 우수한 성능을 보여주었습니다.

- **Technical Details**: DeepFilter는 과거의 관측치를 기반으로 다음 단계의 품질 변수를 예측하는 프로세스 모니터링을 위해 특별히 설계된 아키텍처입니다. 이 시스템은 Global Filtering (GF) 블록을 통해 데이터의 다양한 시간 단계에서 정보를 혼합하는 방법을 적용하여 정보 전달의 효율성을 높입니다. 이러한 Filtering 계층은 데이터의 비선형 복잡성을 처리하는 데 도움이 되며, 다층의 feed-forward network (FFN)를 통해 다양한 채널 간의 정보 혼합도 지원합니다. 이러한 방식으로 DeepFilter는 실시적으로 효율적인 예측 모델을 생성합니다.

- **Performance Highlights**: DeepFilter는 실제 프로세스 모니터링 데이터셋에서 실험을 통해 기존의 최신 모델들에 비해 정확도와 효율성에서 개선된 결과를 보였습니다. 특히, DeepFilter는 리얼타임 애플리케이션에서도 뛰어난 성능을 발휘하여 엔지니어들이 안전하고 신뢰성 있는 결정을 내릴 수 있도록 지원합니다. 이러한 성과는 DeepFilter의 구조적 혁신이 산업 프로세스 모니터링에 매우 적합하다는 것을 뒷받침합니다.



### Change Detection-Based Procedures for Piecewise Stationary MABs: A Modular Approach (https://arxiv.org/abs/2501.01291)
Comments:
          34 pages, 2 figures, 1 table, submitted to JMLR

- **What's New**: 이 연구는 전통적인 Multi-Armed Bandit (MAB) 알고리즘이 정적인 환경을 위해 설계되었다는 점을 강조합니다. 그러나 현실 세계에서는 비정상적인(nonstationary) 환경이 더 일반적입니다. 이 논문에서는 보상 분포가 변화하는 점이 있는 piecewise stationary MAB (PS-MAB) 환경을 탐구하고 있습니다.

- **Technical Details**: 저자들은 PS-MAB의 비대칭적 축적 이론(asymptotic analysis)을 중심으로 연구하고 있으며, 변화 탐지(change detection, CD)에 기반한 새로운 알고리즘을 모듈화하는 방법을 제안합니다. 이 연구는 보상이 sub-Gaussian이라는 가정 하에 변화 점들의 분리에 관한 조건을 바탕으로 모듈화된 CDB 절차를 개발합니다. 이 과정에서 정적인 밴디트 알고리즘과 변화 탐지기가 요구됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모듈형 CDB 절차는 다양한 변화 탐지기 및 MAB 알고리즘의 조합에 대해 통합된 방법으로 후회 경계(regret bounds)를 도출할 수 있음을 보여주고, 기존 방법들과 비교했을 때 성능이 우수함을 입증하였습니다.



### CultureVLM: Characterizing and Improving Cultural Understanding of Vision-Language Models for over 100 Countries (https://arxiv.org/abs/2501.01282)
Comments:
          Technical report; 26 pages

- **What's New**: 이 논문에서는 문화적 이해를 개선하기 위한 대규모 멀티모달 벤치마크 CultureVerse를 구축하였습니다. 이 데이터셋은 19,682개의 문화 개념과 188개의 국가/지역이 포함되어 있어 VLM(Visual-Language Models)이 다문화적 이해 능력을 평가할 수 있도록 합니다. 또한 CultureVLM이라는 일련의 VLM을 제안하여 우리의 데이터셋에 대해 파인튜닝을 통해 문화적 이해를 크게 향상시킵니다.

- **Technical Details**: CultureVerse는 19,682개의 문화 개념, 228,053개의 샘플로 구성되어 있으며, 다양한 문화적 지원을 위해 추가 언어와 문화를 통합할 수 있는 유연한 파이프라인을 제공합니다. VLMs는 일반적으로 문화적 이해의 지역적 불균형을 보여주며, 특히 서구 개념에 더 강하고 아프리카 및 아시아 맥락에서의 성능은 낮은 것으로 나타났습니다. 파인튜닝을 통해 문화적 인식을 향상시켜 모델의 일반적인 성능을 희생하지 않고도 지역과 범주에서의 이해 격차를 줄일 수 있습니다.

- **Performance Highlights**: 모델의 크기와 데이터의 양이 문화적 이해를 증가시키는데 긍정적인 상관관계를 보였으며, 특히 Llama 3.2-11B 모델이 Qwen 2-72B와 유사한 성능을 발휘했습니다. 결과는 VLM이 다양한 문화, 개념, 대륙 및 데이터셋 간의 일반화 능력을 보여주며, 문화적 이해를 향상시키기 위한 일반화 연구의 커다란 가능성을 나타냅니다. 이번 연구는 상황이 잘 반영되지 않은 문화에 대한 AI 형평성을 증진시키는 기반을 마련할 것을 기대합니다.



### A redescription mining framework for post-hoc explaining and relating deep learning models (https://arxiv.org/abs/2501.01209)
- **What's New**: 이 연구에서는 DLM(Deep Learning Models)의 해석 가능성을 높이기 위해 새로운 프레임워크를 제안합니다. 기존의 explainable-AI 접근 방식과 달리, 이 프레임워크는 통계적으로 유의미한 neuron activation의 재설명(redescription)을 통해 DLM을 설명할 수 있는 방법을 제공합니다. 이로 인해 다양한 DLM과 그 구조에 관계없이 적용할 수 있는 가능성을 엽니다.

- **Technical Details**: 제안된 프레임워크는 어떤 DLM에 대해서도 cohort analysis를 수행할 수 있으며, 목표 레이블(target labels) 또는 설명 속성(descriptive attributes) 집합과 neuron을 연결하는 기능을 제공합니다. 이 프레임워크는 인공 신경망 아키텍처와 독립적이며 복잡한 목표 레이블(예: multi-label, multi-target) 시나리오와도 호환됩니다.

- **Performance Highlights**: 이 프레임워크는 교육적(pedagogical) 및 분해적(decompositional) 접근 방식을 모방하여 규칙 추출(rule extraction)을 수행할 수 있습니다. 이를 통해 다양한 DLM의 설명 가능성과 해석 가능성을 증가시킬 수 있는 정보 제공이 가능해집니다.



### A3: Android Agent Arena for Mobile GUI Agents (https://arxiv.org/abs/2501.01149)
- **What's New**: 이 논문은 Android Agent Arena (A3)라는 새로운 평가 플랫폼을 소개합니다. A3는 실제 사용자 시나리오를 반영한 201개의 과제와 21개의 일반적으로 사용되는 제3자 앱을 통합하여 모바일 GUI 에이전트의 능력을 평가합니다. 이 플랫폼은 더 큰 액션 스페이스를 제공하여 다양한 데이터셋에서 훈련된 에이전트와 호환됩니다.

- **Technical Details**: A3는 동적인 평가 방법을 통해 비즈니스 수준의 대규모 언어 모델(LLM)을 활용하여 자동으로 과제를 평가합니다. 이는 인간의 수작업이나 코딩 전문 지식의 필요성을 크게 줄이며, 실세계에서의 에이전트 성능을 더욱 효과적으로 측정합니다. A3는 201개의 과제를 세 가지 유형으로 분류하고 실제 앱 기능에 기반한 디자인을 채택하였습니다.

- **Performance Highlights**: A3의 도입은 현재 GUI 제어 에이전트가 직면하고 있는 실세계 요구와의 간극을 해소할 수 있는 중요한 발전을 나타냅니다. 이는 에이전트가 여러 단계의 목표 지향 과제를 수행할 수 있는 능력을 평가하는 데 필요한 더 포괄적이고 상호작용적인 평가 플랫폼을 제공합니다. 이로 인해 에이전트의 성능 향상과 개발에 기여할 것으로 기대됩니다.



### Beyond Text: Implementing Multimodal Large Language Model-Powered Multi-Agent Systems Using a No-Code Platform (https://arxiv.org/abs/2501.00750)
Comments:
          22 pages, 27 figures

- **What's New**: 이번 연구는 기업의 AI 도입에 따른 실용적인 제약과 높은 진입 장벽을 해소하기 위해, 노코드(No-Code) 플랫폼에 기반한 다중 모달(Multimodal) LLM 기반의 다중 에이전트 시스템(Multi-Agent System, MAS) 설계를 제안합니다. 이러한 접근법은 프로그래밍 지식이 없는 사용자가 AI 시스템을 손쉽게 구축하고 관리할 수 있도록 하는 것을 목적으로 하고 있습니다. 특히, LLM과 같은 최신 AI 기술의 복잡성과 비용 문제를 극복하기 위한 혁신적인 해결책을 제공합니다.

- **Technical Details**: 연구에서는 이미지 기반 노트에서 코드 생성을 포함한 다양한 사용 사례를 검토하였으며, Advanced RAG 기반의 질문-응답 시스템, 텍스트 기반 이미지 생성 및 이미지와 프롬프트를 활용한 비디오 생성 등도 포함됩니다. 이 시스템은 AI 채택 장벽을 낮추어, 전문 개발자는 물론 일반 사용자도 AI를 활용하여 생산성과 효율성을 크게 향상시킬 수 있도록 합니다. 이러한 기술적 구성은 AI 기술의 민주화를 촉진하는 데 기여하고 있습니다.

- **Performance Highlights**: 노코드 플랫폼의 확장성과 접근성은 기업 내 AI 기술의 민주화에 기여하며, 다중 에이전트 시스템의 실용적인 적용 가능성을 검증하고 있습니다. 연구 결과는 다양한 산업에서 AI의 광범위한 채택을 촉진하는 데 중요한 역할을 할 것으로 기대됩니다. 마지막으로, 이러한 성과는 기업들이 AI를 손쉽게 활용 가능하게 만들고, 전반적인 업무 효율성을 높이는 데 기여하는 것을 목표로 합니다.



### Grade Inflation in Generative Models (https://arxiv.org/abs/2501.00664)
Comments:
          10 pages, 6 figures, 1 table

- **What's New**: 이번 연구에서는 생성 모델(generative model)의 품질 평가에서 발생하는 'grade inflation problem'에 대해 다룹니다. 기존의 여러 품질 점수들이 합성 데이터와 실제 데이터의 비교에서 부적절하게 높은 평가를 주는 경향이 있음을 발견했습니다. 우리는 이러한 문제를 해결할 새로운 점수인 Eden score를 제안하며, 이는 기존의 'equipoint scores'보다 더 신뢰할 수 있는 평가를 제공합니다.

- **Technical Details**: Eden score는 통계적 접근과 기능적 접근을 바탕으로 생성된 점수로, 합성 데이터와 실제 데이터의 특징 간의 유사성을 평가합니다. 이 연구에서는 Pearson 상관계수, Earth-mover's score, Jaccard score, Kullback-Leibler 점수와 같은 기존 점수와 Eden score의 성능을 비교했습니다. 특히, Eden score는 equidensity score의 개념을 도입하여 데이터 포인트의 중요성을 균등하게 평가하는 기존의 방법에서 벗어나 grade inflation 문제를 피할 수 있음을 보여줍니다.

- **Performance Highlights**: Eden score는 실제 데이터와 합성 데이터의 적합도를 평가하는 데 있어 인간의 인식과 더 나은 일치를 보였습니다. 연구 결과, Eden score가 grade inflation을 피함으로써 더 낮은 품질 점수를 제공하는 경향이 있음을 보여주었습니다. 또한 equidensity scores가 저차원 분포에서 생성 모델 성능 평가 시 기존의 equipoint scores보다 우수하다는 것을 실증적으로 입증했습니다.



### MCP-Solver: Integrating Language Models with Constraint Programming Systems (https://arxiv.org/abs/2501.00539)
- **What's New**: 이번 논문은 Large Language Models (LLMs)과 constraint programming 시스템 간의 체계적 통합을 위해 Model Context Protocol (MCP)을 기반으로 한 MCP-Solver의 프로토타입 구현을 소개합니다. 이 시스템은 자연어 사양을 정형 constraint 모델로 변환할 수 있는 정밀한 인터페이스를 제공하여, LLM의 자연어 이해와 제약 해결 능력을 효과적으로 결합할 수 있는 가능성을 보여줍니다. 이 논문은 또한 오픈 소스로 구현된 MCP-Solver의 실용성을 입증합니다.

- **Technical Details**: MCP-Solver는 Python 3.9 이상의 환경에서 작동하며, MiniZinc와 Chuffed solver를 사용하여 제약 모델링을 수행합니다. 시스템은 자연어를 통한 상호작용을 지원하고, 모델 일관성을 유지하는 점검 단계를 통해 각 변경 후 모델의 정확성을 보장합니다. 또한, 시스템은 이중 형식으로 LLM의 작업 결과를 처리하기 위해 시간을 잘 관리하며, 지식 기반을 구축하여 모델링 인사이트를 지속적으로 유지합니다.

- **Performance Highlights**: MCP-Solver의 초기 실험 결과는 LLM과 제약 해결 능력이 결합될 때 자연어 처리 성능이 향상될 수 있음을 시사합니다. 이 시스템은 자연어 사양을 유효한 MiniZinc 모델로 변환하는 데 효과적이며, 반복적인 수정 및 검증 과정에서 피드백을 제공하여 모델의 질을 높이는 데 기여합니다. 최종적으로 이 논문은 자연어 처리와 제약 기반 추론의 통합을 위한 중요한 첫 걸음을 내딛었다고 주장합니다.



### Extending XReason: Formal Explanations for Adversarial Detection (https://arxiv.org/abs/2501.00537)
Comments:
          International Congress on Information and Communication Technology (ICICT), Lecture Notes in Networks and Systems (LNNS), Springer, 2025

- **What's New**: 본 논문에서는 XReason 도구를 확장하여 LightGBM 모델과 클래스 수준 설명(class-level explanations)을 지원하도록 하였습니다. 또한, XReason에서 적대적 예제(adversarial examples)를 생성 및 감지할 수 있는 메커니즘을 구현하였습니다. 이러한 기능 확장은 다양한 기계 학습 모델에 대해 보다 정확하고 안정적인 설명을 생성하기 위한 기반을 제공합니다.

- **Technical Details**: 제안된 방법론은 기존의 XReason 도구를 기반으로 하여 LightGBM 모델을 지원하고, 클래스 수준 설명을 생성하는 기능을 추가하였습니다. 각 결정 트리는 노드에서의 조건을 인코딩한 논리적 제약 조건의 시퀀스로 표현되며, 이는 모델의 결정 과정을 정확하게 포착합니다. 또한 MaxSAT 솔버를 사용하여 사례 수준 및 클래스 수준의 형식적 설명을 생성하는 과정도 포함되어 있습니다.

- **Performance Highlights**: 논문에서는 CICIDS-2017 데이터셋을 사용하여 제안된 방법론의 효율성과 정확성을 평가하였습니다. XReason+ 프레임워크는 훈련된 모델에 대한 테스트 데이터의 예측을 개선하며, 적대적 샘플을 탐지하고 생성하는 데 있어 뛰어난 성능을 보였습니다. 이러한 평가는 제안된 기법이 실제 적용 가능한 해법임을 증명합니다.



### Efficient support ticket resolution using Knowledge Graphs (https://arxiv.org/abs/2501.00461)
- **What's New**: 이 논문은 160,000건이 넘는 고객 사례를 분석하여, 기존의 기계 학습 및 지식 그래프(knowledge graph) 기법을 활용하여 어려운 고객 문제를 빠르게 해결하는 방법을 제안합니다. 특히, 여러 엔지니어가 협력하는 '스워밍'(swarming) 상황을 고려하여 최적의 엔지니어 그룹을 식별하는 방법에 초점을 맞추고 있습니다. 이를 통해 고객의 대기 시간을 줄이는 기회를 창출하고자 합니다.

- **Technical Details**: 논문에서 제안하는 핵심 ML(task)는 학습-순위화(learning-to-rank)로, 사건과 현재 사건에 할당된 엔지니어 집합을 바탕으로 해결에 가장 적합한 엔지니어를 순위로 정리하는 것입니다. 여러 입력 feature로는 고객이 제공한 사건 설명, 영향을 받는 구성 요소, 엔지니어의 전문성 평가, 엔지니어가 작성한 지식 베이스 글의 내용을 포함합니다. 또한, 이 모델은 엔지니어가 해결한 사례 전반에 대한 데이터를 포함하여 LTR 알고리즘의 성능을 향상시키고자 합니다.

- **Performance Highlights**: 해당 연구 결과는 추가적인 맥락을 포함함으로써 전통적인 기계 학습 기법인 TF-IDF보다 현저히 더 나은 추천 성과를 나타냅니다. 이를 통해 엔지니어와 고객 사이의 상호작용을 최적화하고, 문제 해결 과정을 효율적으로 개선할 수 있는 가능성을 보여줍니다. 이 연구는 결국 고객 만족도를 높이는 방향으로 기여할 수 있을 것으로 기대됩니다.



### Knowledge-aware equation discovery with automated background knowledge extraction (https://arxiv.org/abs/2501.00444)
- **What's New**: 이 논문에서는 기존의 차별 방정식 발견 알고리즘의 한계를 극복하기 위해 자동 혹은 수동으로 추출된 배경 지식을 활용하는 새로운 알고리즘을 제안합니다. 이전 알고리즘들이 고정된 형태의 방정식 구조에 맞춰 계수를 회복하는 방식이었다면, 이 연구에서는 교차 및 돌연변이 연산자 내에서 특정 용어가 나타날 가능성을 높여 방정식을 발견할 수 있도록 변경하였습니다. 이러한 방식은 전문가가 선택한 용어를 모방하면서도, 어떠한 방정식 형태도 얻을 수 있는 가능성을 유지합니다.

- **Technical Details**: 제안된 알고리즘은 배경 지식의 중요도 분포를 통해 방정식 발견 과정을 개선하는 데 초점을 맞추고 있습니다. 기존의 SINDy(Sparse Identification of Nonlinear Dynamical Systems) 알고리즘과 비교했을 때, 이 알고리즘은 검색 안정성과 내구성 측면에서 우수한 성능을 보여줍니다. 실험에서는 Burgers, 파동, Korteweg–De Vries 방정식에 대해 합성 예제가 제공되어 알고리즘의 유효성을 검증합니다.

- **Performance Highlights**: 제안된 알고리즘은 pySINDy와 고전적인 EPDE(Equation Discovery via Evolutionary methods) 알고리즘을 포함한 기존 방법들과 비교하였으며, 노이즈 강건성과 전반적인 품질, 가능한 방정식 형태 측면에서 모든 방법들을 초월하는 결과를 보였습니다. 비록 진화 최적화 방법의 속도가 느리다는 단점이 있으나, 방정식의 가능성 범위가 상당히 넓어지는 점은 중요한 발견으로 평가됩니다.



### $\texttt{FORM}$: Learning Expressive and Transferable First-Order Logic Reward Machines (https://arxiv.org/abs/2501.00364)
Comments:
          AAMAS'25

- **What's New**: 이 논문에서는 전통적인 보상 기계(Reward Machines, RMs)의 한계를 극복하기 위해 첫 번째 논리(First-Order Logic)를 사용한 최초의 보상 기계(First-Order Reward Machines, FORM)를 제안합니다. 전통적인 RMs는 한정된 표현력을 가진 제안 논리(Propositional Logic)를 사용하여 엣지를 라벨링하는데, 이는 복잡한 작업에 필요한 여러 상태 및 엣지로 인해 학습 가능성과 전달성을 저해합니다.

- **Technical Details**: FORM은 첫 번째 논리로 엣지를 라벨링하여 더 compact(압축된)하고 transferable(전환 가능한) RMs을 생성합니다. 이 논문에서는 FORM을 학습하기 위한 새로운 방법과 여러 에이전트가 협력하여 공유 FORM을 위한 정책을 학습하는 다중 에이전트 형식을 도입하여 전달성을 촉진합니다. 이러한 접근 방식은 에이전트들이 공동으로 복잡한 작업을 효과적으로 해결할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, FORM이 전통적인 RM에 비해 뛰어난 확장성을 보임을 입증하였습니다. 특히, FORM은 전통적인 RM 학습 방법이 실패하는 작업에서도 효과적으로 학습될 수 있으며, 다중 에이전트 학습 구조와 첫 번째 논리가 제공하는 추상화 덕분에 학습 속도와 작업 전달성에서 뚜렷한 개선을 보여줍니다.



### Autonomous Alignment with Human Value on Altruism through Considerate Self-imagination and Theory of Mind (https://arxiv.org/abs/2501.00320)
- **What's New**: 이 논문에서는 인공지능(AI) 에이전트가 인간의 이타적 가치에 자율적으로 맞추어 행동할 수 있도록 하는 새로운 프레임워크를 제안합니다. 자아 상상(Self-Imagination)과 마음 이론(Theory of Mind, ToM) 메커니즘을 통합하여, 에이전트가 타인의 복지와 환경의 영향을 고려하여 윤리적 결정을 내릴 수 있도록 합니다. 이 연구는 고대 중국 이야기인 '사마 광이 항아리를 깨뜨리다(Sima Guang Smashes the Vat)'에서 영감을 받아 복잡한 도덕적 결정 환경을 설계하였으며, 이는 에이전트가 이타적 결정을 우선시하도록 돕는 것이 주요 목표입니다.

- **Technical Details**: 제안된 프레임워크는 에이전트의 경험을 기반으로 하여 상태 추정을 수행하고, 이를 통해 의사결정이 다른 대행자와 환경에 미치는 잠재적 영향을 예측합니다. 이 모델은 고유한 경험과 관점 전환(perspective taking)을 통해 이타적 동기를 생성하며, 이를 통해 에이전트는 보다 안전하고 이타적인 행동을 할 수 있게 됩니다. 특히, 기존 연구들이 이타적 행동과 안전한 의사결정을 개별적으로 다룬 것을 반영하여, 이 연구는 두 가지를 통합하는 메커니즘을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 훈련된 에이전트는 '항아리를 깨뜨려 아이를 구한다'는 상황에서, 사람을 구하는 것을 우선시하면서도 항아리를 깨뜨리는 부정적인 영향을 최소화하는 방식으로 행동했습니다. 이는 환경의 역효과를 고려하면서도 이타적 행동을 수행할 수 있는 새로운 길을 제시했습니다. 최종적으로 이 에이전트는 설정된 목표를 달성하는 데 성공하였습니다.



### Automatically Planning Optimal Parallel Strategy for Large Language Models (https://arxiv.org/abs/2501.00254)
- **What's New**: 이번 연구에서는 대규모 언어 모델의 훈련에서 최적의 병렬 전략을 자동으로 찾는 알고리즘을 제안합니다. 새로운 알고리즘은 훈련 시간을 시뮬레이션하고 이를 기반으로 병렬 솔루션 공간을 줄여 최적의 솔루션을 찾습니다. 특히, micro batch size와 global batch size와 같은 세부 변수도 고려하여 보다 정교하게 병렬 전략을 수립할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 병렬 훈련 시간을 계산(computation), 통신(communication), 겹침(overlap)으로 분리하여 시뮬레이션 모델을 구축합니다. 이 모델을 기반으로 99%의 검색 공간을 줄여주며, 평균적으로 96%의 정확도로 병렬 훈련 기간을 추정할 수 있습니다. 이러한 접근을 통해 이전의 경험적인 기준에 의존하지 않고, 사용자에게 최적의 하이퍼파라미터 선택을 지원합니다.

- **Performance Highlights**: 여러 노드에서 진행된 실험 결과, 제안된 알고리즘은 항상 글로벌 최적(global optimum) 전략을 제공하며, 훈련 효율성을 극대화하는 데 기여합니다. 다양한 대규모 언어 모델을 대상으로 한 추가 실험을 통해 알고리즘의 정확성이 입증되었습니다. 이러한 결과들은 특히 자원 집약적인 대규모 언어 모델 개발에 있어 중요한 비용 절감 효과를 가져올 수 있습니다.



### Generative Emergent Communication: Large Language Model is a Collective World Mod (https://arxiv.org/abs/2501.00226)
- **What's New**: 이 연구는 generative emergent communication (generative EmCom)이라는 통합 이론 틀을 제안합니다. 이 틀은 emergent communication, world models, 및 large language models (LLMs)를 집단적 예측 부호화(collective predictive coding, CPC)의 관점에서 연결합니다. 제안된 프레임워크는 여러 에이전트 간의 분산된 Bayesian 추론을 통해 언어 및 기호 시스템의 발생을 형식화하며, 전통적인 차별적 모델 기반 접근 방식을 넘어선다고 설명합니다.

- **Technical Details**: 이 연구는 generative EmCom이라는 새로운 프레임워크를 제안하며, 이를 통해 multi-agent reinforcement learning (MARL)에서의 커뮤니케이션 발생을 제어(control)로서의 추론(inference)으로부터 도출할 수 있음을 보여줍니다. 또한, LLM을 집단적 세계 모델로 해석하는 수학적 정식화도 제안하며, 이는 다양한 에이전트의 경험을 CPC를 통해 통합하는 방식으로 이루어집니다. 이를 통해 집단적 예측 부호화 과정에서 공유 기호 시스템이 어떻게 발생하는지를 이해할 수 있는 통일된 이론적 기초를 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 언어 발생의 근본적인 측면을 설명하고 LLM을 이해하는 데 필요한 실용적인 통찰을 제공합니다. 저자들은 수학적 정식화와 기존의 연구에 대한 논의를 통해 이 프레임워크가 인공지능 개발 및 인간-인공지능 상호작용 개선에 있어 어떻게 기여할 수 있는지를 보여줍니다. 궁극적으로, 이 연구는 복잡한 AI 시스템 및 다중 에이전트 시스템의 발전에 중요한 기틀을 제공할 것으로 기대됩니다.



### CancerKG.ORG A Web-scale, Interactive, Verifiable Knowledge Graph-LLM Hybrid for Assisting with Optimal Cancer Treatment and Car (https://arxiv.org/abs/2501.00223)
- **What's New**: 이번 연구에서는 colorectal Cancer에 대한 최신 동료 검토(peer-reviewed) 의학 지식으로 구성된 최초의 Web-scale hybrid Knowledge Graph(KG)-Large Language Model(LLM) 중 하나를 소개합니다. 이 모델은 미국 및 세계 최고의 암 센터 중 하나인 Moffitt Cancer Center에서 의학 연구 및 임상 정보 검색 작업을 지원하기 위해 평가되고 있습니다. 기존의 LLM, KG, 검색 엔진의 단점을 극복하고 사용자 요구를 더 잘 충족시키는 하이브리드 모델이 개발되었습니다.

- **Technical Details**: 이 하이브리드 모델은 LLM이 가지는 환각(hallucinations)과 재앙적인 망각(catastrophic forgetting)의 문제를 해결하고자 제작되었습니다. 최신의 상태에서의 KG인 PrimeKG, cBioPortal, ChEMBL 등은 수동 큐레이션(manual curation)을 필요로 해 빠르게 오래된 정보를 포함하게 됩니다. 반면, CancerKG는 비지도 학습(unsupervised)에 기반하여 최신 의학 발견을 자동으로 수집하고 조직화할 수 있습니다.

- **Performance Highlights**: CancerKG는 사용자의 편의성을 고려하여 서로 다른 데이터 모달리티(data modalities)를 효과적으로 처리하는 5가지의 고급 사용자 인터페이스를 제공합니다. 이 모델은 LLM의 문제점을 완화하기 위해 검증된 KG를 Retrieval Augmented Generation(RAG) 방식으로 활용하고 있습니다. 이로 인해 사용자는 더욱 향상된 의료 정보 검색 경험을 누릴 수 있습니다.



### Probabilistic Explanations for Linear Models (https://arxiv.org/abs/2501.00154)
Comments:
          Extended version of AAAI paper

- **What's New**: 이번 논문에서는 Formal XAI(설명 가능한 인공지능)의 발전을 바탕으로 기계 학습 모델의 결정에 대해 수학적으로 보장된 설명을 제공하는 새로운 접근 방식을 제안합니다. 특히, '충분한 이유(sufficient reason)'의 계산에 중점을 두며, 작은 크기의 $	extdelta$-Sufficient Reasons ($	extdelta$-SRs)을 찾는 것이 본 연구의 핵심으로 자리 잡고 있습니다. 이를 위해 기존의 접근 방식에서 벗어나, $(	extdelta, 	extepsilon)$-Sufficient Reasons 개념을 도입하고 선형 모델에서의 효율적 계산 방안을 제시합니다.

- **Technical Details**: Formal XAI에서는 기계 학습 분류기의 결정을 설명하는 데 필요한 수학적 기반과 해석 가능성을 강조합니다. 이 논문은 입력 인스턴스에 대한 충분한 이유를 정의하고, 이러한 이유들이 어떻게 결정에 대한 PDF 기반($	extdelta$-SR) 설명으로 나타날 수 있는지를 수학적으로 설명합니다. 특히, 여러 피처의 값을 기반으로 한 결정 및 그 결정의 근거를 명확하게 정의하고, 이를 통해 분석의 정확성을 높이는 방안을 제안합니다.

- **Performance Highlights**: 제안된 $(	extdelta, 	extepsilon)$-SR 개념은 효율적으로 계산할 수 있으며, 이는 기계 학습 모델에서 설명 가능성의 새로운 기준을 설정할 것으로 기대됩니다. 기존의 이론적 한계를 극복하고, 더 작고 해석하기 쉬운 설명을 생성함으로써, 사용자가 모델의 작동 방식을 더 잘 이해하도록 돕는 데 기여합니다. 이 접근법은 향후 다양한 기계 학습 모델의 해석 가능성을 높이는 데 중요한 역할을 할 것입니다.



### AltGen: AI-Driven Alt Text Generation for Enhancing EPUB Accessibility (https://arxiv.org/abs/2501.00113)
- **What's New**: 이 논문은 AltGen이라는 AI 기반의 새로운 파이프라인을 소개하여 EPUB 파일 내 이미지의 alt text 생성을 자동화합니다. 기존의 접근 방식과 비교하여 AltGen은 더 높은 정확성과 관련성을 제공합니다. 연구는 대규모 데이터 세트에서의 효율성을 입증하며, 고품질의 자동 생성 alt text를 생산할 수 있는 가능성을 보여줍니다.

- **Technical Details**: AltGen 파이프라인의 첫 단계는 EPUB 파일을 파싱하여 이미지, 텍스트 내용 및 메타데이터를 추출하는 데이터 전처리입니다. 이후 CLIP 및 ViT와 같은 고급 컴퓨터 비전 모델을 사용하여 시각적 내용을 분석하고 의미 있는 특징을 추출합니다. 최종 단계에서는 GPT와 같은 변환기 기반 모델을 사용하여 문맥적으로 정확하고 언어적으로 일관된 alt text를 생성합니다.

- **Performance Highlights**: 실험 결과, AltGen은 접근성 오류를 97.5% 감소시키며, 유사성 및 언어적 충실도 측정에서 높은 점수를 기록했습니다. 사용자 연구에서는 AltGen이 문서의 사용성과 이해를 크게 향상시키는 데 기여함을 보고했습니다. 또한 AltGen은 기존의 접근 방식에 비해 더 높은 정확도와 확장성을 보였습니다.



### Object-level Visual Prompts for Compositional Image Generation (https://arxiv.org/abs/2501.01424)
Comments:
          Project: this https URL

- **What's New**: 이번 연구는 텍스트-이미지 확산 모델 내에서 객체 수준의 시각적 프롬프트(composition)를 조합하는 방법을 소개합니다. 이 접근 방식은 다양한 장면과 스타일을 아우르는 의미적으로 일관된 구성 생성을 목표로 하며, 텍스트 프롬프트가 제공하는 다재다능함과 표현력을 구현하고자 합니다. 키와 값을 각각 다른 시각적 표현으로부터 학습하는 KV-mixed cross-attention 메커니즘을 도입하여, 객체의 정체성을 유지하면서도 다양한 구성을 생성할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 인코더를 사용하는 KV-mixed cross-attention 모듈을 제안합니다. 작은 보틀넥을 가진 인코더는 레이아웃 컨트롤에 사용되는 키(key)를 생성하고, 더 큰 보틀넥 인코더는 세부적인 외관 정보를 담고 있는 값(value)을 생성합니다. 이러한 두 가지 정보 소스를 혼합하여 시각적 프롬프트의 정체성을 유지하면서도 유연한 변형을 가능하게 합니다. 또한, 역추론(inference) 단계에서는 Compositional Guidance를 통해 객체 수준의 가이드를 제공하여 정체성 보존과 레이아웃 일관성을 강화합니다.

- **Performance Highlights**: 이 방법은 다채로운 장면 구성을 생성하면서 각 시각적 프롬프트의 고유한 특성을 유지하는 데 성공합니다. 결과적으로 본 연구의 기법은 기존의 이미지 프롬프트 방법, 최적화 기법 및 다중 모달 생성 방법에 비해 뛰어난 성능을 보이며, 텍스트-이미지 생성의 창의적 잠재력을 확장합니다. 다양한 개체 배열 및 장면 구성을 유지하면서도 일관성 있고 세부적인 이미지를 생성하는 데 기여할 것으로 기대됩니다.



### Multi-Modal Video Feature Extraction for Popularity Prediction (https://arxiv.org/abs/2501.01422)
Comments:
          INFORMS 2024 Data Challenge Competition

- **What's New**: 이 연구는 짧은 동영상의 인기 예측을 위해 동영상 자체와 관련된 기능을 사용하는 혁신적인 접근법을 보여줍니다. 인기도는 조회수(view count), 좋아요 수(like count), 댓글 수(comment count), 공유 수(share count)의 네 가지 주요 참여 지표로 측정됩니다. 다양한 동영상 분류 모델을 Backbone Network로 활용하여 동영상 모달리티 기능을 추출하고, 정리된 동영상 캡션을 사용한 텍스트 생성 모델을 통해 동영상 내용을 이해합니다.

- **Technical Details**: 이 연구는 동영상 및 табular 데이터 기반으로 데이터 마이닝(data mining) 및 기능 엔지니어링(feature engineering)을 수행하여 해시태그 등장 빈도, 언급 빈도, 동영상 길이 등 여러 실용적인 기능을 구축했습니다. TimeSformer, ViViT, VideoMAE, X-CLIP과 같은 최신 Neural Network 모델을 사용하여 동영상 특징을 추출하였고, XGBoost와 결합해 예측 정확도를 높였습니다. 또한, 결측치(missing values)는 중앙값 중복(median imputation)으로 처리하였으며 특징의 로그 변환(logarithmic transformation)을 통해 모델의 안정성을 향상시켰습니다.

- **Performance Highlights**: 다양한 모델의 훈련을 통해 가장 안정성이 높은 XGBoost 모델이 최종적으로 선택되었습니다. 최종 예측 결과는 Neural Network와 XGBoost 모델의 예측치를 평균하여 산출되는 형태로, 예측 정확도는 평균 절대 비율 오차(MAPE)로 평가되었습니다. 이 연구는 짧은 동영상의 인기 예측 분야에서 의미 있는 발전을 이루었으며, 사용자 수와 비디오 특성을 고려하여 다양한 요소가 결합되는 점에서 큰 가치를 제공합니다.



### On Unifying Video Generation and Camera Pose Estimation (https://arxiv.org/abs/2501.01409)
- **What's New**: 이번 연구는 동영상 생성 모델이 3D 인식을 잘 수행할 수 있는지 조사하며, OpenSora라는 비디오 생성 모델의 중간 feature가 카메라 자세 추정(camera pose estimation)에 어떻게 기여하는지를 탐구합니다. 연구팀은 JOG3R이라는 새로운 통합 모델을 제안하여 비디오 생성과 카메라 자세 추정을 동시에 수행할 수 있는 능력을 보여줍니다. 이 모델은 고급 비디오 생성 품질을 유지하면서 카메라 자세 추정 정확도를 높이는 데 강력한 성능을 발휘합니다.

- **Technical Details**: OpenSora는 Diffusion Transformer (DiT) 기반의 비디오 확산 모델로, 사전 훈련된 VAE 인코더를 통해 낮은 차원 잠재 공간에서 확산 과정을 수행합니다. 연구진은 발생된 중간 feature가 본래의 3D 인식을 얼마나 가지고 있는지를 평가하기 위해, DUSt3R와 같은 예측 모듈을 비디오 생성 네트워크에 연결하여 설계했습니다. 이를 통해 비디오 생성 모델이 카메라 추정을 위한 feature를 효과적으로 재사용할 수 있는지를 검증했습니다.

- **Performance Highlights**: 실험 결과, 비디오 생성 feature는 본래 약한 3D 인식을 가지고 있으나, 카메라 자세 추정을 위한 추가적 감독(supervision)을 통해 significantly 향상됨을 확인하였습니다. JOG3R 모델은 경기력 면에서도 state-of-the-art 솔루션과 경쟁할 수 있는 카메라 자세 추정치를 생성하면서 비디오 생성 품질도 유지합니다. 따라서 JOG3R 모델은 비디오 생성 및 3D 카메라 재구성을 동시에 수행할 수 있는 최초의 통합 모델로 주목받고 있습니다.



### A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models (https://arxiv.org/abs/2501.01394)
- **What's New**: 본 연구는 변환기 기반 시간 시계열 예측 모델을 위한 통합된 하이퍼파라미터 최적화 (HPO) 파이프라인을 제안합니다. 이 파이프라인은 다양한 최신 모델들을 표준 벤치마크 데이터셋에서 비교하고, 이를 통해 실질적인 인사이트와 예제를 생성합니다. 또한 이 논문은 하이퍼파라미터 최적화가 다른 최신 모델에도 일반화될 수 있음을 보여줍니다. 연구의 궁극적인 목표는 업계 실무자와 연구자에게 특정 도메인 응용에 적합한 하이퍼파라미터를 효율적으로 파악하고, 정보와 자료를 제공하는 것입니다.

- **Technical Details**: 시간 시계열 예측 (TSF) 모델은 ARMA 모델과 같은 전통적인 방법에서부터, 딥러닝 모델로 발전해 온 과정에서 변환기 모델의 효과iveness가 크게 부각되었습니다. 그러나 변환기 모델은 다양한 하이퍼파라미터에依存 (dependence) 하며, 이러한 하이퍼파라미터의 최적화는 상당한 기술적 전문성이 필요합니다. 본 연구는 다양한 SOTA 모델을 대상으로 하이퍼파라미터 최적화를 수행하고, 여러 실험을 통해 성능을 검증합니다.

- **Performance Highlights**: 실험은 다수의 변환기 기반 TSF 모델을 포함하여 공통적으로 사용되는 공개 데이터셋에서 진행되었습니다. 각 모델은 20회의 반복 실험을 통해 최적의 하이퍼파라미터를 찾고, Mean Squared Error (MSE)와 Mean Absolute Error (MAE)를 평가 지표로 사용하여 성능을 측정하였습니다. 본 연구의 결과는 하이퍼파라미터 최적화를 통한 모델 성능 향상을 보여주며, 다양한 데이터셋에서 재현성, 공정성 그리고 성능을 향상 시킬 수 있는 방향성을 제시합니다.



### Training Medical Large Vision-Language Models with Abnormal-Aware Feedback (https://arxiv.org/abs/2501.01377)
Comments:
          16 pages

- **What's New**: 본 연구에서는 기존의 Medical Large Vision-Language Models (Med-LVLMs)의 한계를 극복하기 위해 UMed-LVLM을 제안합니다. 이 모델은 의료 이미지에서의 시각적 로컬라이제이션(visual localization) 문제를 해결하는 데 중점을 두고 개발되었습니다. 또한 Medical Abnormalities Unveiling (MAU) 데이터셋을 사용하여 병리학적 이상 감지 능력을 보강합니다.

- **Technical Details**: UMed-LVLM은 두 단계의 훈련 방법인 Abnormal-Aware Instruction Tuning과 Abnormal-Aware Rewarding을 통해 교육됩니다. Abnormal-Aware Rewarding은 Abnormal Localization Rewarding과 Vision Relevance Rewarding을 포함하여 모델이 이상 영역을 효과적으로 캡처할 수 있도록 설계되었습니다. 이는 의료 이미지를 이해하고 이에 따른 진단을 생성하는 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, UMed-LVLM은 기존의 Med-LVLMs와 비교하여 의료 이미지를 이해하고 이상을 식별하는 데 있어 뛰어난 성능을 보여주었습니다. 또한, 모델을 훈련하고 일반화 능력을 심층 분석한 결과, Med-LVLMs의 이상 감지 능력을 개선하는 것이 의료 이미지 이해에 미치는 긍정적인 영향을 강조했습니다. 대규모 모델이 제한된 질병 유형의 다양한 의료 이미지에 노출되더라도 robust한 질병 인식을 위한 잠재력을 갖추고 있음을 보여주었습니다.



### ScarNet: A Novel Foundation Model for Automated Myocardial Scar Quantification from LGE in Cardiac MRI (https://arxiv.org/abs/2501.01372)
Comments:
          31 pages, 8 figures

- **What's New**: ScarNet는 LGE 이미징에서 심근 섬유화와 흉터를 평가하는 새로운 하이브리드 모델입니다. 이 모델은 Medical Segment Anything Model (MedSAM)에서 가져온 transformer 기반의 인코더와 U-Net 디코더를 결합하여 만들어졌습니다. ScarNet의 도입으로 수작업으로 이루어지는 LV 흉터 정량화의 비효율성 문제를 해결하고, 임상 환경에서의 활용 가능성을 높였습니다.

- **Technical Details**: ScarNet는 552명의 허혈성 심근병증 환자로부터 전문가의 세분화 데이터를 이용하여 훈련되었습니다. ScarNet은 tailored attention blocks로 성능을 향상시킨 하이브리드 구조이며, 184명의 별도 환자에서 테스트되었습니다. 모델은 높은 스카 경계 정밀도를 달성하며, 기존 모델들보다 더 낮은 편향(bias)과 변동계수(coefficient of variation)를 보였습니다.

- **Performance Highlights**: ScarNet은 테스트 환자에서 0.912의 중앙 Dice 점수를 기록하며, MedSAM과 nnU-Net보다 현저히 우수한 성능을 보였습니다. Monte Carlo 시뮬레이션에서도 ScarNet은 0.892의 높은 스카 Dice를 달성하여, 노이즈 변화에도 강한 내성을 입증했습니다. 이러한 결과는 ScarNet이 다양한 이미지 품질과 흉터 패턴에 대해 안정적인 성능을 발휘함을 보여줍니다.



### Contrastive Learning from Exploratory Actions: Leveraging Natural Interactions for Preference Elicitation (https://arxiv.org/abs/2501.01367)
Comments:
          Accepted to HRI 2025

- **What's New**: 이 논문은 로봇의 행동을 사용자 선호와 정렬하기 위해 사용자들이 탐색하는 행동을 통해 레이블을 생성할 수 있는 새로운 접근 방식을 제안합니다. 기존의 데이터 라벨링 과정 없이 사용자들이 자연스럽게 행동을 탐색하면서 생성된 데이터를 활용하는 것이 핵심입니다. 이를 위해 'Contrastive Learning from Exploratory Actions (CLEA)'라는 새로운 학습 방법을 소개하며, 이는 로봇 행동의 특징을 학습하는 데 효과적입니다.

- **Technical Details**: CLEA는 사용자가 로봇 행동을 커스터마이즈하는 과정에서 수행하는 탐색적 행동을 데이터 소스로 활용하여 학습하는 방법입니다. 이 과정에서 사용자는 흥미로운 행동을 선택하고, 관련없는 행동은 무시하게 됩니다. 이러한 방식은 광범위한 사용자 데이터 수집 과정 없이도 로봇의 행동을 사용자 선호에 맞추는 것이 가능하다는 점에서 혁신적입니다.

- **Performance Highlights**: CLEA로 학습한 특징은 기존의 자가 지도 학습(self-supervised learning) 기법보다 사용자 선호를 이끌어내는 데 더 효과적임을 입증했습니다. 두 개의 사용자 연구에 의해 CLEA 특징은 완전성(completeness), 단순성(simplicity), 최소성(minimality), 설명 가능성(explainability) 등 네 가지 주요 지표에서 우수한 성능을 보였습니다. 이는 로봇 행동 커스터마이징을 위한 사용자 친화적인 접근법을 제공함을 의미합니다.



### ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding (https://arxiv.org/abs/2501.01366)
Comments:
          20 pages with 5 figures and 11 tables

- **What's New**: 이 논문에서는 3D 시나리오에서 자연어로 언급된 대상을 정확히 위치시키는 3D visual grounding (3DVG) 모델의 중요성을 강조합니다. 특히, 기존 데이터셋의 한계를 극복하기 위해 다양한 언어 패턴을 포괄할 수 있는 새로운 진단 데이터셋인 Visual Grounding with Diverse Language in 3D (ViGiL3D)를 소개합니다. 이 데이터셋은 3DVG 방법을 평가하는 데 있어 유용하고 대표적인 프롬프트 세트를 마련하는 데 기여할 것입니다.

- **Technical Details**: ViGiL3D는 다양한 자연어 프롬프트를 통해 3DVG 메서드의 능력을 테스트할 수 있는 프레임워크를 제공합니다. 연구는 기존의 오픈 바카블러리(open-vocabulary) 3DVG 방법들을 평가하여, 이러한 방법들이 더 도전적인 프롬프트를 이해하고 대상을 식별하는 데 부족함이 있음을 보여줍니다. 이를 통해, 보다 실용적인 응용 프로그램을 위해 필요한 언어적 다양성을 강조합니다.

- **Performance Highlights**: 테스트 결과, 현재의 3DVG 방법들은 다양한 언어 패턴에 대한 이해도가 낮은 것으로 나타났습니다. 더욱이 이 연구는 언어적인 성능 향상을 위해 필요로 하는 영역을 밝혀냄으로써, 3DVG 모델의 발전 방향을 제시합니다. 연구진은 이러한 데이터를 바탕으로 미래에 더 효과적이고 효율적인 3DVG 솔루션을 개발할 수 있는 가능성을 열어 놓습니다.



### CySecBench: Generative AI-based CyberSecurity-focused Prompt Dataset for Benchmarking Large Language Models (https://arxiv.org/abs/2501.01335)
- **What's New**: 이 논문에서는 CySecBench라는 사이버 보안 분야의 jailbreak 기술을 평가하기 위해 설계된 12662개의 프롬프트를 담은 새로운 데이터셋을 공개합니다. 이 데이터셋은 10개의 공격 유형 카테고리로 체계적으로 구성되어 있어, 프롬프트의 평가가 보다 일관되고 정확하게 이루어질 수 있습니다. CySecBench의 유용성을 입증하기 위해 프롬프트 은닉(prompts obfuscation)을 기반으로 한 새로운 jailbreak 방법을 제안합니다.

- **Technical Details**: 프롬프트 생성은 OpenAI의 GPT 모델인 GPT-o1-mini와 GPT-3.5-turbo를 사용하여 이루어졌습니다. 먼저, GPT-o1-mini가 다양한 사이버 공격과 관련된 사이버 보안 용어 657개를 식별한 후, 공격 유형에 따라 10개의 그룹으로 분류합니다. 그 후, GPT-3.5-turbo가 제공된 악의적인 질문에 대한 50개의 폐쇄형 지시 및 질문을 생성하여 결과적으로 CySecBench 데이터셋을 만듭니다.

- **Performance Highlights**: 제안한 jailbreak 방법은 상용 블랙박스 LLM에서 유해한 콘텐츠를 성공적으로 유도하여 ChatGPT에서 65%, Gemini에서 88%의 성공률을 기록했습니다. 반면 Claude는 17%의 성공률을 보여 더 강한 저항성을 나타냈습니다. CySecBench를 사용한 평가에서는 기존의 최첨단 방법들보다 뛰어난 성능을 기록하여 LLM 보안 평가를 위한 도메인 특화 평가 데이터셋의 가치를 강조하고 있습니다.



### The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation (https://arxiv.org/abs/2501.01329)
- **What's New**: 이 논문에서는 소프트웨어 테스트 케이스 생성을 위한 LLM(대형 언어 모델) 맞춤형 프롬프트를 자동으로 생성하는 MAPS라는 새로운 방법을 제안합니다. 현재까지의 연구는 LLM의 성능이 주로 사람이 작성한 프롬프트에 의존하고 있음을 밝혔으며, 서로 다른 LLM에 가장 적합한 프롬프트를 찾는 것이 필요하다는 점을 강조하고 있습니다. MAPS는 프롬프트 최적화를 위한 세 가지 주요 모듈을 통해 이 문제를 해결합니다.

- **Technical Details**: MAPS는 다양성 기반의 프롬프트 생성, 오류 기반 규칙 도출, 도메인 맥락 지식 추출이라는 세 가지 주요 모듈로 구성되어 있습니다. 다양성 기반 프롬프트 생성 모듈은 다양한 수정 경로를 탐색하여 다양한 프롬프트를 생성합니다. 오류 기반 규칙 도출 모듈은 생성된 테스트 케이스의 일반적인 오류를 반영하여 최적화 방향을 식별하고, 도메인 맥락 지식 추출 모듈은 클래스 상속 및 호출 관계와 같은 정보를 제공하여 LLM이 정확한 테스트 케이스를 생성하도록 돕습니다.

- **Performance Highlights**: 실험 결과, MAPS는 세 가지 인기 있는 LLM(예: ChatGPT, Llama-3.1, Qwen2)에 대해 기존의 최첨단 프롬프트 최적화 방법들에 비해 뛰어난 성능을 보였습니다. 평균적으로 MAPS는 6.19% 더 높은 라인 커버리지 비율과 5.03% 더 높은 분기 커버리지 비율을 달성했습니다. MAPS는 각 LLM에 가장 적합한 프롬프트를 효과적으로 생성하며, 수작업으로 설계된 프롬프트보다 우수한 결과를 보여주었습니다.



### Understanding Difficult-to-learn Examples in Contrastive Learning: A Theoretical Framework for Spectral Contrastive Learning (https://arxiv.org/abs/2501.01317)
- **What's New**: 이 논문에서는 비지도 대비 학습(unsupervised contrastive learning)에서 학습하기 어려운 예제(difficult-to-learn examples)의 제거가 다운스트림 분류 성능을 향상시킬 수 있다는 흥미로운 사실을 발견했습니다. 이러한 어려운 예제는 일반적으로 감독 학습(supervised learning)에서 모델 성능에 긍정적인 영향을 미치지만, 비지도 설정에서는 반대로 부정적인 영향을 미친다는 것을 이론적으로 분석하였습니다.

- **Technical Details**: 연구진은 샘플 쌍 간 유사성을 모델링하는 이론적 프레임워크인 similarity graph를 개발하였습니다. 이를 바탕으로, 학습하기 어려운 예제가 포함된 모델과 포함되지 않은 모델의 오류 경계를 도출하고, 어려운 예제가 성능 저하에 미치는 영향을 증명했습니다. 또한, 단순히 이러한 어려운 예제를 제거하거나 margin tuning 및 temperature scaling과 같은 기법을 사용할 경우, 일반화 경계를 개선하여 성능을 향상시킬 수 있음을 이론적으로 입증했습니다.

- **Performance Highlights**: 실험을 통해, 원본 이미지 데이터셋에 비해 비지도 대비 학습 성능이 향상된다는 것을 확인했습니다. CIFAR-10, CIFAR-100 등의 데이터셋에서 수행된 실험 결과, 어려운 예제를 제거한 경우 오히려 더 나은 다운스트림 성능을 기록했습니다. 또한, 어려운 예제 선택을 위한 효율적인 메커니즘을 제안하고 이를 통해 성능을 높일 수 있음을 보여 주었습니다.



### Multi-Head Explainer: A General Framework to Improve Explainability in CNNs and Transformers (https://arxiv.org/abs/2501.01311)
- **What's New**: 이번 연구에서는 Multi-Head Explainer (MHEX)를 소개하여, CNN과 Transformer 모델의 설명 가능성과 정확성을 동시에 향상시키는 모듈형 프레임워크를 제안합니다. MHEX는 동적으로 작업 관련 기능을 강조하는 Attention Gate, 초기 레이어가 타겟 클래스와 관련된 세부 정보를 캡처하도록 안내하는 Deep Supervision, 정제된 지역과 전역 표현을 통합하여 포괄적인 saliency maps를 생성하는 Equivalent Matrix의 세 가지 핵심 구성 요소로 이루어져 있습니다. MHEX는 기존의 ResNet 및 BERT 모델에 최소한의 수정으로 쉽게 통합될 수 있으며, 이를 통해 분류 정확도와 해석 가능성이 향상됩니다.

- **Technical Details**: MHEX는 CNN 및 Transformer 기반 모델의 설명 가능성과 정확성을 강화하기 위해 설계된 모듈형 프레임워크로, 세 가지 주요 구성 요소로 구성되어 있습니다. Attention Gate는 입력 기능의 가중치를 지역 및 글로벌 정보에 기반하여 동적으로 조정하며, Deep Supervision은 초기 레이어에서의 특징 학습을 최적화하여 세밀한 세부 정보를 잡아냅니다. 마지막으로 Equivalent Matrix는 정제된 지역 및 전역 표현을 통합하여 의역 가능하고 상세한 saliency 점수를 생성합니다.

- **Performance Highlights**: 의료 이미징 및 텍스트 분류 분야의 벤치마크 데이터셋에서 광범위한 실험 결과, MHEX는 분류 정확도를 개선할 뿐만 아니라 매우 해석 가능한 saliency 점수도 생성하는 것으로 나타났습니다. 특히 의료 전문가는 정확하고 해석 가능한 모델 예측에 의존해야 하므로, 이러한 향상된 모델이 임상적 사용에 있어 더욱 유용할 것으로 기대됩니다. MHEX는 모든 모델 구조에 대한 뛰어난 호환성을 가지고 있어, 더 많은 연구와 응용이 가능할 것입니다.



### Citations and Trust in LLM Generated Responses (https://arxiv.org/abs/2501.01303)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 연구는 질문 답변 시스템의 사용자가 느끼는 신뢰성(trust)를 조사하고, 인용(citation)의 존재가 그 신뢰성에 미치는 영향을 분석합니다. 기존 연구와 달리, 인용이 있는 경우와 사용자가 이를 확인하는 경우의 상반된 효과를 관찰하였습니다.

- **Technical Details**: 연구는 상업용 Chatbot이 생성한 텍스트 응답에 대해 다양한 양의 인용(0개, 1개, 5개)과 관련 있는 인용 및 무작위 인용을 적용하여 실험을 진행했습니다. 이 실험에서 참여자들의 인용 확인 여부와 AI 응답에 대한 신뢰감을 자체보고(self-reported) 형태로 수집했습니다.

- **Performance Highlights**: 결과적으로, 인용이 존재할 때 신뢰감이 유의미하게 증가하였고, 무작위 인용이더라도 이러한 효과가 유지되었습니다. 반면, 인용을 확인한 경우에는 신뢰감이 현저히 감소하는 경향을 보였습니다. 이는 AI가 생성한 콘텐츠의 신뢰도 향상에 있어 인용의 중요성을 강조합니다.



### LEO-Split: A Semi-Supervised Split Learning Framework over LEO Satellite Networks (https://arxiv.org/abs/2501.01293)
Comments:
          13 pages, 15 figures

- **What's New**: 이번 논문에서는 LEO(저지구 궤도) 위성 네트워크를 위한 새로운 반 감독(split) 학습 프레임워크인 LEO-Split을 제안합니다. 기존의 딥러닝 모형의 단점을 극복하기 위해 이 시스템은 반 감독 학습을 통해 라벨이 없는 데이터를 효율적으로 활용하며, 위성-지상국 간의 비연속적 연결 문제를 해결하는 보조 모형을 작성하였습니다.

- **Technical Details**: LEO-Split은 반 감독 학습(SS)과 분할 학습(SL)을 결합한 모델로, 여기서는 데이터 부족 문제와 훈련 실패를 예방하기 위해 보조 모델을 구성합니다. 또한, 의사 레이블링(pseudo-labeling) 알고리즘을 통해 데이터 불균형을 조정하며, 적응형 활성화 보간(activation interpolation) 기법을 도입하여 과적합(overfitting) 문제를 완화합니다.

- **Performance Highlights**: 실제 LEO 위성 데이터(예: Starlink)를 기반으로 한 실험 결과, LEO-Split은 최신 기술에 비해 뛰어난 성능을 기록하였습니다. 특히, 위성 간의 데이터 불균형 문제를 해결하고, 모델의 일반화 능력을 향상시키는 데 기여하며, 구체적으로 170GB 크기의 전송 데이터에서도 효과적으로 작동함을 입증하였습니다.



### NeutraSum: A Language Model can help a Balanced Media Diet by Neutralizing News Summaries (https://arxiv.org/abs/2501.01284)
- **What's New**: 이번 연구에서는 미디어 바이어스를 줄이기 위한 새로운 프레임워크인 NeutraSum을 제안합니다. 이 프레임워크는 두 가지 중립성 손실(neutrality losses)을 통합하여, 생성된 요약의 의미 공간을 조정하고 미디어 바이어스를 최소화하는 데 목표를 두고 있습니다. NeutraSum은 같은 사건을 보도하는 편향된 뉴스 기사들을 통합하여 중립적이고 사실에 근거한 요약을 생성할 수 있도록 합니다. 실험 결과는 NeutraSum이 요약 성능을 향상시키고 미디어 바이어스를 현저히 줄이는 데 기여함을 보여줍니다.

- **Technical Details**: NeutraSum 모델은 다중 문서 요약(multi-document summarisation) 손실을 사용하여 고품질 요약을 생성하며, 두 가지 중립성 손실인 대비 손실(contrastive loss)과 동일 거리 손실(equal-distance loss)을 활용합니다. 이러한 손실들은 편향된 출처 간 의미 공간을 조정하고, 전문가 작성 요약과의 일치를 보장하여 중립적인 텍스트 생성을 유도합니다. 이 모델은 Allsides 데이터셋에서 다양한 정치적 편향을 가진 기사들로부터 동일한 사건에 대한 정보를 통합하여 요약을 생성합니다.

- **Performance Highlights**: 실험을 통해 NeutraSum은 요약 과정에서 미디어 바이어스를 효과적으로 감소시키면서도 핵심 정보를 지속적으로 보존하는 성과를 보여줍니다. 특히, 다중 문서 요약 손실과 중립성 손실의 조합이 모델이 보다 중립적인 출력을 생성하는 데 중요한 역할을 했습니다. 이러한 접근 방법은 이후 뉴스 요약에서의 공정성과 중립성을 높이는 데 기여할 것으로 기대됩니다.



### PIMAEX: Multi-Agent Exploration through Peer Incentivization (https://arxiv.org/abs/2501.01266)
Comments:
          Accepted at ICAART 2025

- **What's New**: 최근 단일 에이전트 강화 학습(Single-Agent Reinforcement Learning)에서 탐사(exploration) 문제는 많은 연구가 이루어졌지만, 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning)에서는 이에 대한 연구가 상대적으로 적었습니다. 이 논문에서는 동료 간 보상(peer-incentivized reward) 함수인 PIMAEX(reward)를 제안하여 다중 에이전트 설정에서 탐사를 촉진합니다. PIMAEX 보상은 에이전트들이 서로 영향을 미치도록 유도하여 새로운 상태를 탐색하기 쉽게 합니다.

- **Technical Details**: PIMAEX 보상은 세 가지 항목 α(알파), β(베타), γ(감마)로 구성된 일반화된 다중 에이전트 사회적 영향(peer reward function)의 특정 사례입니다. 이 보상을 기반으로 한 PIMAEX-Communication 알고리즘은 에이전트 간의 통신 채널을 활용하여 협력적인 탐사를 강화합니다. 전체적으로 이 연구는 Consume/Explore 환경에서 PIMAEX 보상을 적용하여 탐사 대 착취(exploration vs. exploitation) 딜레마와 신뢰 할당(credit-assignment problem)을 해결하려고 합니다.

- **Performance Highlights**: 실험 결과, PIMAEX 보상과 PIMAEX-Communication을 사용하는 에이전트가 그렇지 않은 경우보다 우수한 성능을 보였습니다. 특히 탐사가 어려운 환경에서 에이전트들이 서로 영향을 주고받으며 더 나은 정책(policy)을 학습하는 데 기여하는 것으로 나타났습니다. 이 연구는 여러 에이전트 간의 효과적인 상호작용을 통해 탐사 문제를 해결하는데 중요한 기여를 하고 있습니다.



### ProgCo: Program Helps Self-Correction of Large Language Models (https://arxiv.org/abs/2501.01264)
Comments:
          Working in progress

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 자기 검증(self-verify)과 자기 수정(self-refine) 기능을 극대화하기 위한 새로운 접근 방식인 Program-driven Self-Correction (ProgCo)를 제안합니다. 기존 LLM들이 복잡한 추론 작업에서 자주 실패하는 문제를 해결하기 위해, 자체 생성된 검증 의사 프로그램(pseudo-program)을 사용하여 복잡한 검증 로직을 구현합니다.

- **Technical Details**: ProgCo는 두 가지 주요 구성 요소로 구성됩니다. 첫째, 프로그램 기반 검증(ProgVe)은 LLM이 만들어낸 검증 프로그램을 통해 검증 로직을 수행하며, 둘째, 프로그램 기반 수정보강(ProgRe)은 ProgVe의 피드백을 받아 응답과 검증 프로그램 모두를 이중 반영(dual reflection)하여 수정합니다. 이러한 방식은 잘못된 피드백으로 인한 혼란을 줄이는 데 효과적입니다.

- **Performance Highlights**: 세 가지 지침 준수(instruction-following) 및 수학적 벤치마크에서 실시된 실험 결과, ProgCo는 효과적인 자기 수정을 달성했으며, 실제 프로그램 도구와 결합할 경우 성능이 더욱 향상될 수 있음을 보였습니다. 이 연구는 LLM의 자기 개선 가능성을 새롭게 열어줄 수 있는 기초를 마련합니다.



### Stealthy Backdoor Attack to Real-world Models in Android Apps (https://arxiv.org/abs/2501.01263)
- **What's New**: 본 연구는 모바일 앱에서 실제로 배포된 깊은 신경망 모델에 대한 효과적이고 은밀한 백도어 공격을 탐구합니다. 딥러닝(Deep Learning, DL) 모델들이 모바일 애플리케이션에 내장되면서 사용자의 스마트폰에 다양한 보안 위협이 존재하게 되었습니다. 특히, 전통적인 백도어 공격 방법들은 현실 세계에서의 모델에 대한 효과성이 부족했으나, 본 연구에서는 DNN 기반 스테가노그래피를 활용하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 저자들은 모바일 앱에서 DNN 모델을 추출하고, 스테가노그래피를 사용해 육안으로는 감지할 수 없고 샘플에 특화된 백도어 트리거를 생성하여 공격의 잠복성을 높입니다. 연구는 네 가지 최신 DNN 모델에서 BARWM (Backdoor Attack against Real-World Models)의 유효성을 확인하였고, 이는 기존의 DeepPayload와 비교하여 공격 성공률이 평균 12.50% 높으며 정상 성능을 보다 잘 유지한다고 보고합니다. 이러한 새로운 접근 방식은 기존의 샘플 비특이적 트리거에 대한 의존성을 줄여줍니다.

- **Performance Highlights**: 본 연구의 실험 결과는 BARWM이 기존의 공격 방법보다 더욱 효과적이고 견고한 백도어 공격을 가능하게 한다는 것을 입증하였습니다. BARWM은 정상 사용자 데이터에서 평균 15.38% 높은 공격 성공률과 15.69% 높은 정상 정확도를 보였으며 은밀함을 평가하는 PSNR 값 역시 최소 5.46 dB 더 높은 결과를 나타냈습니다. 38,387개의 모바일 앱에서 추출한 89개의 실제 모델을 통해 이러한 효과를 달성했습니다.



### Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants (https://arxiv.org/abs/2501.01243)
Comments:
          50 pages, 14 figures, 41 tables. Submitted to ICLR 2025

- **What's New**: 이번 연구에서는 얼굴과 인간 이해 능력을 평가하기 위한 새로운 기준인 Face-Human-Bench를 제안합니다. 이 기준은 3단계의 능력 분류 체계에 기반하여 개발되었으며, 공통적으로 사용되는 데이터셋에서 수집된 900개의 개발 문제와 1800개의 테스트 문제를 포함합니다. 연구 결과는 멀티모달 대형 언어 모델(MLLMs)의 성능 차이에 대한 흥미로운 통찰을 제공합니다.

- **Technical Details**: 제안된 능력 분류 체계는 세 가지 수준으로 구성되어 있으며, Level-1에는 얼굴 이해(face understanding)과 인간 이해(human understanding)의 두 가지 관점이 포함됩니다. 각 수준에서는 인지 과정에 관한 세부 능력이 정의되어 있으며, Level-2에서는 얼굴 관련 5가지와 인간 관련 5가지 능도로 나뉘어 있습니다. 최종적으로, Face-Human-Bench는 25개 주류 MLLMs의 얼굴 및 인간 이해 능력을 종합적으로 평가하기 위한 방안을 제공합니다.

- **Performance Highlights**: Face-Human-Bench에 대한 평가 결과, 특정 MLLMs는 얻은 점수에서 상당한 차이를 보였으며, 상대적 위치가 성능에 미치는 영향도 심각하게 분석되었습니다. 특히, 심각한 시나리오에서의 깊은 가짜 탐지에서 전문 모델이 MLLMs보다 우수한 성능을 보임을 확인하여, 특정 작업에서 전문 모델의 통합이 필요함을 제안합니다. 연구 결과는 멀티모달 조수의 응답 품질을 높이기 위한 방안을 제시합니다.



### An Efficient Attention Mechanism for Sequential Recommendation Tasks: HydraRec (https://arxiv.org/abs/2501.01242)
- **What's New**: 본 연구는 최근 Transformer 기반 모델이 추천 시스템(recommender systems, RS)에서 점점 더 효과적으로 사용되고 있음을 강조합니다. 기존의 Transformer 모델은 언어 모델링에서 좋은 성능을 발휘했으나, RS에서는 시퀀스 길이에 따라 복잡성이 기하급수적으로 증가하는 문제가 있었습니다. 이를 해결하기 위해 HydraRec이라는 새로운 효율적인 Transformer 기반 Sequential RS 모델을 제안합니다.

- **Technical Details**: HydraRec은 주목(attention) 계산의 이론적 복잡성을 감소시키며, 긴 시퀀스 및 대규모 데이터 세트에 대한 처리 성능을 개선합니다. 특히, Hydra attention의 개념을 기반으로 하여, 토큰 수와 모델의 임베딩 차원 모두에서 복잡성을 줄이는 방식으로 설계되었습니다. 이 모델은 causal masking을 사용할 때, 시퀀스 추천(next item prediction) 작업에서 기존의 dot-product 기반 모델과 비견할 만한 성능을 보여줍니다.

- **Performance Highlights**: HydraRec은 다양한 평가 지표에서 다른 선형 어텐션 기반 모델들보다 뛰어난 성능을 발휘했습니다. 특히, BERT4Rec 모델과 비교했을 때 이 모델은 running time에서 개선을 보였으며, 이는 사용자 구매 이력과 같은 동적 시퀀스 데이터를 처리하는 데 있어 더욱 효율적임을 의미합니다.



### Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects (https://arxiv.org/abs/2501.01205)
- **What's New**: 이 논문에서는 다중 에이전트 대형 언어 모델(Multi-Agent LLMs)을 활용하여 공학 학생들이 수행하는 고급 설계 프로젝트(senior design projects, SDP)를 지원하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 전문가 관점을 대표하는 LLM 에이전트들이 상호작용하여 학생들이 복잡한 문제를 효과적으로 해결할 수 있도록 돕습니다. 이 접근법은 공학 교육에서 직면하는 다양한 윤리적, 사회적, 환경적 문제를 고려하고 반영하는 데 유용합니다.

- **Technical Details**: 제안된 프레임워크는 문제 정의 에이전트, 시스템 복잡성 에이전트, 윤리 및 사회적 에이전트 등의 다양한 역할을 수행하는 LLM 에이전트를 포함합니다. 이들 에이전트는 실시간으로 협력하여 인간 엔지니어 팀을 모방하는 방식으로 대화함으로써 프로세스를 촉진합니다. 이 구현은 프로프트 엔지니어링(prompt engineering) 기술을 활용하여 각 에이전트의 역할에 따른 다양한 페르소나(personas)를 개발하며, 이는 다양한 전문 지식의 융합을 통한 문제 해결을 가능하게 합니다.

- **Performance Highlights**: 이 프레임워크는 공학 교육에서 고급 설계 프로젝트에 참여하는 학생들에게 비판적 사고 및 협업 능력을 개발하는 데 기여할 것으로 기대됩니다. 평가 결과, 이 시스템은 복잡한 문제 해결에 있어 학생들이 더 혁신적이고 강력한 솔루션을 제시하도록 유도할 것이라는 점에서 긍정적인 효과를 보여줍니다. 이러한 접근법은 다중 전공, 다각적인 문제 해결을 요구하는 현대 공학 환경을 대비하는 데 필요한 훈련 방안을 제공합니다.



### Data Augmentation Techniques for Chinese Disease Name Normalization (https://arxiv.org/abs/2501.01195)
Comments:
          The Version of Record of this contribution is published in 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

- **What's New**: 본 논문에서는 질병 명칭 정규화 (disease name normalization)의 데이터를 증가시키기 위한 혁신적인 접근법인 Disease Data Augmentation (DDA)을 제안합니다. 기존 시스템들의 훈련 데이터가 부족한 문제를 해결하기 위해 다양한 데이터 증강 기술과 지원 모듈을 통합하였습니다. 이를 통해 DDA 접근법이 여러 기본 모델과 훈련 목표에서 성능 개선을 나타냄을 실험적으로 입증하였습니다.

- **Technical Details**: DDA 접근법에서는 질병 명칭 정규화를 위한 작업 정의 및 데이터 증강 방법을 도입합니다. 세 가지 주요 축 단어(질병 중심(disease center), 해부학적 영역(anatomical region), 질병의 특성(disease characteristic))를 정의하고, 이 축 단어를 찾아내기 위해 BiLSTM+CRF 기반의 명명된 개체 인식(NER) 시스템을 설계하였습니다. 데이터 증강 모듈은 두 가지 주요 범주인 축 단어 대체(Axis-word Replacement)와 다중 미세 집계(Multi-Granularity Aggregation)를 포함하여 질병 명칭의 다양한 구성 요소와 관계에 대한 추가 지식을 제공합니다.

- **Performance Highlights**: 자세한 실험 결과, DDA 접근법이 다른 데이터 증강 방법들을 능가하며 질병 명칭 정규화의 다양한 기본선 모델 성능을 효과적으로 향상시킴을 보여줍니다. 특히, 훈련 데이터가 부족한 상황에서도 DDA 접근법은 전반적인 성능의 약 80%에 가깝게 도달할 수 있음을 증명했습니다. 이러한 성과는 제한된 데이터로도 성능을 극대화할 수 있는 방법이 될 것입니다.



### L3D-Pose: Lifting Pose for 3D Avatars from a Single Camera in the Wild (https://arxiv.org/abs/2501.01174)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 논문에서는 동물과 영장류의 3D 포즈 추정이 환경에 따라 동적이고 예측 불가능한 행동을 보이는 동물로 인해 어렵다는 한계를 극복하기 위해 리깅된 아바타와 합성 데이터셋을 활용하는 하이브리드 접근 방식이 제안되었습니다. 이 방법은 주어진 이미지와 무관하게 2D 포즈를 3D로 변환하는 간단한 Attention MLP 네트워크를 도입하여 사용의 확장성을 보장합니다. 또한, 기존의 해부학적 키포인트 탐지기는 임의의 아바타에 포즈를 정확히 재배치하는데 부족하다는 점을 강조하며, 이를 극복하기 위한 루크업 테이블을 제시합니다.

- **Technical Details**: 연구는 2D 데이터 세트의 수집을 통해 정확한 2D 포즈 예측 모델을 훈련시키고, 그 후 합성된 3D 포즈 데이터에서 유래한 priors를 사용하여 2D 포즈를 3D 공간으로 "리스팅(lifting)"하는 프로세스를 설명합니다. 해부학적 제약을 활용하여 2D 키포인트에서 실제적인 3D 재구성을 달성하고, 이를 통해 3D 주석을 수집하지 않고도 정확한 3D 포즈 예측이 가능합니다. 이러한 접근 방식은 특히 자연 환경에서의 포즈 추정을 위한 원천 데이터 확보의 어려움을 극복하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 루크업 테이블 기반의 포즈 리타게팅 방법이 효과적이고 효율적임을 입증했습니다. 연구는 두 가지 데이터셋, 즉 Deep Macaque과 Deep Horse를 개발하였으며, 이는 다양한 행동 세트를 포함하여 물리 기반 게임 엔진을 활용하여 생성되었습니다. 이를 통해 다양한 자연 환경에서의 포즈를 2D에서 3D로 변환하고, 이를 아바타에 매끄럽게 이식하는 그래픽 응용 프로그램의 가능성을 열었습니다.



### Blind Men and the Elephant: Diverse Perspectives on Gender Stereotypes in Benchmark Datasets (https://arxiv.org/abs/2501.01168)
- **What's New**: 이번 논문은 언어 모델에서 성 고정관념 편향을 측정하고 완화하는 방법에 대해 다룹니다. 선행 연구에서 내재적 접근법(intrinsic approach)과 외연적 접근법(extrinsic approach) 간의 상관관계 부족이 밝혀졌으며, 본 연구는 내재적 측정의 복잡성을 탐구합니다. 연구진은 데이터 분포(data distribution)와 성 고정관념 요소를 분석하여 결과 일관성 개선을 위한 새로운 방법론을 제안합니다.

- **Technical Details**: 연구는 StereoSet과 CrowS-Pairs라는 두 가지 널리 사용되는 내재적 편향 평가 기준을 중심으로 진행됩니다. 데이터 세트의 샘플 분포(sample distribution) 차이가 두 기준 간 불일치를 유발한다는 가설을 세우고, 샘플 분포를 조정하여 결과의 일관성을 높일 수 있음을 실험을 통해 입증합니다. 연구진은 또한 성 고정관념 샘플을 수동으로 정리하여 데이터 세트의 구조적 개선을 시도하였습니다.

- **Performance Highlights**: 연구 결과에 따르면 StereoSet과 CrowS-Pairs 간의 상관관계는 매우 낮아, Pearson 상관계수는 0.13으로 확인되었습니다. 하지만 데이터 세트를 균형 있게 조정함으로써 두 기준의 상관관계를 최적화할 수 있는 가능성이 드러났습니다. 이러한 발견은 언어 모델에서의 성 고정관념 측정 및 완화 방법 개발에 있어 새로운 방향성을 제시합니다.



### Deep Learning in Palmprint Recognition-A Comprehensive Survey (https://arxiv.org/abs/2501.01166)
Comments:
          Palmprint recognition, biometrics, deep learning, feature extraction, recognition tasks

- **What's New**: 이번 논문은 딥러닝(DL)을 기반으로 한 손바닥 인식의 최근 발전을 포괄적으로 리뷰합니다. 기존의 연구들은 특정 작업에 초점을 맞추었으나, 본 논문은 DL 기술의 혁신적인 역할과 다양한 과제들을 통합적으로 탐구하는 데 중점을 두고 있습니다. 이 논문은 손바닥 인식의 최신 동향과 기술을 정리하여 연구자들이 최신 기술을 이해하고 혁신을 주도할 수 있도록 돕습니다.

- **Technical Details**: 손바닥 인식 기술은 이미지 획득, 전처리, 특징 추출, 매칭의 네 가지 주요 단계로 구성됩니다. 딥러닝 모델 특히 CNN(Convolutional Neural Networks), VGG-16, ResNet 등 다양한 네트워크가 손바닥 인식의 독특한 도전 과제를 해결하기 위해 적용되어 왔습니다. 이러한 기술들이 손바닥의 섬세한 텍스쳐 특징을 효과적으로 포착하고 인식의 정확성을 높이는데 기여하고 있습니다.

- **Performance Highlights**: 논문에서는 퍼포먼스 메트릭과 실험 결과를 종합적으로 평가하며, 특히 DL 기반 방법이 손바닥 인식 성능을 어떻게 향상시켰는지 설명합니다. 향후 연구 방향으로는 보안/privacy 유지 인식, 오픈 세트 인식, 다중 도메인 기술 등이 제안되고 있으며, 이는 손바닥 인식 기술의 활용 범위를 더욱 넓힐 것으로 기대됩니다.



### TexAVi: Generating Stereoscopic VR Video Clips from Text Descriptions (https://arxiv.org/abs/2501.01156)
Comments:
          6 pages, published in 2024 IEEE International Conference on Computer Vision and Machine Intelligence (CVMI)

- **What's New**: 이 연구는 기존의 생성 시스템을 통합하여 텍스트에서 입체 가상 현실 비디오를 생성하는 새로운 접근 방식을 제안합니다. 텍스트-이미지 (text-to-image) 모델에서 시작하여, 안정적인 이미지를 생성하는 Stable Diffusion과 깊이 추정 (depth estimation) 알고리즘을 활용하여 높은 품질의 프레임을 제시합니다. 기존 텍스트 기반 생성 시스템을 활용하여 가상 현실 제작의 수작업을 줄이고, 입체적인 시청 경험을 제공합니다.

- **Technical Details**: 제안된 TexAVi 모델은 텍스트 프롬프트를 기반으로 초기 이미지를 생성하는 AttnGAN, 더 높은 품질의 프레임을 생성하는 Stable Diffusion, 그리고 좌우 시각을 위한 깊이 추정 모듈로 구성되어 있습니다. 각 모듈은 사전 훈련된 모델을 활용하여 효율적으로 작동하며, 입체 가상 환경의 사용자 맞춤화에 유리합니다. TexAVi는 이러한 단계를 통해 수작업 없이 고품질의 VR 비디오 생성이 가능하도록 설계되었습니다.

- **Performance Highlights**: TexAVi에서 생성된 비디오는 시각적으로 매력적이며, 스마트폰 호환 VR 헤드셋을 통해 전시될 수 있는 품질을 자랑합니다. 연구의 성과는 기존 방법론들과의 비교를 통해 시각적 품질 측면에서 우수함을 입증하며, 새로운 자연어 기반 그래픽의 가능성을 보여줍니다. 이 프로세스는 가상 현실 기술과 심층 신경망의 통합으로 나아가는 한 단계로 여겨집니다.



### Symmetries-enhanced Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.01136)
- **What's New**: 본 논문에서는 내재적 대칭(intrinsic symmetry)이 부족한 다중 에이전트 시스템(multi-agent system)의 동역학에 외재적 대칭(extrinsic symmetry)을 통합하는 새로운 프레임워크를 제안합니다. 이를 통해 다양한 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL) 문제에 대해 대칭을 활용할 수 있는 가능성을 확대합니다. 이 프레임워크의 핵심은 분산 집단 작업을 위해 특별히 설계된 Group Equivariant Graphormer입니다.

- **Technical Details**: Group Equivariant Graphormer 아키텍처는 에이전트 간의 상호작용을 최적화하도록 설계되었으며, 대칭 강화(symmetry-enhanced) 방법을 사용하여 분산 swarm task에 적합합니다. 기존의 많은 동적 시스템들은 활용할 수 있는 대칭이 부족하지만, 본 연구에서는 이러한 문제를 해결하기 위해 외재적 대칭의 통합을 시도합니다. 실험을 통해 다양한 시나리오와 서로 다른 swarm 크기에서의 제로샷 제어(zero-shot scalability)에 대한 향상된 결과를 보여줍니다.

- **Performance Highlights**: 대칭 파괴(symmetry-breaking) 쿼드로터에 대한 광범위한 실험을 통해 본 방법이 충돌률(collision rates)을 상당히 줄이고, 다양한 시나리오에서 작업 성공률(task success rates)을 향상시키는 것을 보여줍니다. 이 연구는 다중 에이전트 강화 학습의 일반화(generalization)와 확장성(scalability)을 개선할 수 있는 잠재력을 입증하였습니다.



### Missing Data as Augmentation in the Earth Observation Domain: A Multi-View Learning Approach (https://arxiv.org/abs/2501.01132)
- **What's New**: 최근에 논의된 Multi-view Learning (MVL)을 통해 여러 데이터 소스나 보기를 활용하여 기계 학습 모델의 성능과 견고성을 향상시키려는 시도가 늘어나고 있습니다. 특히, 이 연구에서는 누락된 데이터에 대한 새로운 접근 방식을 제시하여 Earth Observation (EO) 분야에서 MVL 모델의 예측력을 더욱 높이고자 합니다. 이 방법은 누락된 뷰를 상정하여 다양한 조합을 훈련 샘플로 사용하고, 정적 데이터가 아닌 동적 병합 기능을 사용하여 누락된 뷰를 무시합니다.

- **Technical Details**: 이 접근 방식은 모든 누락된 뷰의 조합을 시뮬레이션하는 CoM(Combinations of Missing views) 기술을 사용합니다. 또한, 퓨전(fusion) 작용은 평균 또는 더 복잡한 함수인 Transformer 등의 동적 병합 함수를 통해 이루어집니다. 재미있는 점은, 누락된 뷰의 특징을 완전히 무시하며 동적 병합을 통해 MVL 모델이 데이터 예측을 근본적으로 개선할 수 있다는 것입니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 중간 정도의 정보 누락 상황에서도 모델의 견고성을 향상시키며, 모든 뷰가 존재할 때는 예측 성능을 개선하는 것으로 나타났습니다. 이들은 EO 분야에서의 최신 기법들과 비교하여 우수한 결과를 보여주며, 시간적 및 정적 뷰를 모두 아우르는 데이터셋을 기반으로 검증되었습니다. 이러한 방법들은 가용한 뷰의 조합에 대해 효과적으로 작동하는 단일 적응형 솔루션을 제공합니다.



### TED: Turn Emphasis with Dialogue Feature Attention for Emotion Recognition in Conversation (https://arxiv.org/abs/2501.01123)
Comments:
          past activity in 2021

- **What's New**: 이 논문은 대화에서 감정 인식을 위한 새로운 방법인 ‘Turn Emphasis with Dialogue (TED)’를 제안합니다. TED는 대화 특징을 주의 메커니즘에 추가하여 각 턴을 명시적으로 구분합니다. 이 방법은 턴의 위치와 화자 정보에 따라 각 턴에 대한 우선 순위를 부여하고, 이를 통해 멀티 턴 입력을 더 잘 처리할 수 있습니다.

- **Technical Details**: TED는 돌 기반 인코딩(‘Turn-Based Encoding’, TBE) 및 다중 헤드 셀프 어텐션('Multi-Head Self-Attention', MHSA) 기법을 사용하여 불특정 다수의 턴을 인코딩합니다. 각 턴과 현재 턴 간의 관계를 조정하기 위해 우선순위 요소를 활용합니다. TED는 이전 및 이후 턴을 포함한 멀티 턴 입력 시퀀스를 생성하고, 이를 통해 다층적인 대화 맥락을 구축함으로써 감정 인식의 정확도를 높입니다.

- **Performance Highlights**: TED는 네 가지 기준 세트에서 평가되었으며, 모든 데이터셋에서 높은 성능을 나타냈습니다. 특히, TED는 IEMOCAP 데이터셋에서 최신 기술 수준의 성능을 달성하였습니다. 이를 통해 TED가 멀티 턴 대화의 감정 인식에 있어 효과적인 솔루션임을 입증하였습니다.



### Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning (https://arxiv.org/abs/2501.01120)
Comments:
          9 pages, 8 figures. Accepted by AAAI 2025. Codes are released at this https URL

- **What's New**: 이 논문은 불완전한 모달리티에서의 다중모달 학습을 해결하기 위해 RAGPT라는 새로운 Retrieval-AuGmented 동적 Prompt Tuning 프레임워크를 제안합니다. RAGPT는 유사한 인스턴스를 식별하는 multi-channel retriever, 결측 정보를 생성하는 missing modality generator, 그리고 맥락 지식을 활용하여 동적 프롬프트를 생성하는 context-aware prompter의 세 가지 모듈로 구성됩니다. 이러한 접근법은 기존의 정적 프롬프트 방법에서 발생하는 성능 저하 문제를 해결하고 다중모달 전처리 모델의 강 robustness를 향상시킵니다.

- **Technical Details**: RAGPT는 여러 모달리티 간 유사성을 바탕으로 유사 샘플을 검색하는 보편적인 multi-channel retrieval 전략을 활용합니다. 이 전략은 동일한 모달리티의 결측 정보를 보완하기 위해 인접한 샘플로부터 정보를 추출하는데 중점을 둡니다. 또한, context-aware prompter는 검색된 인스턴스 간의 의미론적 상관 관계를 식별하여 입력에 맞춘 동적 프롬프트를 생성함으로써 다중모달 특성을 조정합니다.

- **Performance Highlights**: 실험 결과, RAGPT는 세 가지 실제 데이터셋에서 경쟁하는 아키텍처들과 비교하여 일관되게 더 나은 성능을 보여주었습니다. RAGPT는 결측 모달리티 문제를 효과적으로 다루고, 고전적인 방법으로 발생할 수 있는 정보 손실과 노이즈를 최소화합니다. 이 연구는 다중모달 모델의 신뢰성, 정확성 및 안전성을 향상시키는 중요한 기여를 합니다.



### Pruning-based Data Selection and Network Fusion for Efficient Deep Learning (https://arxiv.org/abs/2501.01118)
Comments:
          Accepted at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Workshop on Attributing Model Behavior at Scale (ATTRIB)

- **What's New**: 이번 논문에서는 데이터 선택을 향상시키고 네트워크 훈련을 가속화하기 위해 PruneFuse라는 새로운 방법을 제안합니다. PruneFuse는 모델 가지치기(pruning)와 네트워크 융합(fusion)을 결합하여 훈련 효율성을 높이고 관련된 주석 비용(annotation costs)을 줄이는 것을 목표로 합니다. 기존의 방법들은 일반적으로 계산 비용이 많이 들어 실제 적용에 한계가 있었지만, PruneFuse는 효율적인 데이터 선택을 통해 이 문제를 해결합니다.

- **Technical Details**: PruneFuse는 먼저 원래의 밀집 네트워크를 가지치기하여 더 작은 대리 모델(surrogate model)을 생성합니다. 이 작은 모델은 데이터셋에서 가장 유용한 샘플을 선택하는 과정에서 사용됩니다. 최적의 샘플을 선택한 후, 가지치기된 모델에서 학습한 통찰력을 밀집 모델과 통합하여 훈련 초기화에 활용합니다. 이 과정에서 모델 초기화가 최적화되어 전체 훈련 시간이 가속화됩니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, Tiny-ImageNet-200과 같은 다양한 데이터셋에 대한 실험 결과, PruneFuse는 기존의 최첨단(active learning) 방법들보다 더 우수한 성능을 보여주었습니다. 이는 또한 데이터 선택을 위한 계산 비용을 상당히 줄였습니다. PruneFuse는 다양한 네트워크 아키텍처에서의 폭넓은 적용성을 보여, 딥러닝 설정에서 효율적인 데이터 선택을 위한 유연한 도구로 자리 잡을 가능성을 가지고 있습니다.



### Robust COVID-19 Detection from Cough Sounds using Deep Neural Decision Tree and Forest: A Comprehensive Cross-Datasets Evaluation (https://arxiv.org/abs/2501.01117)
Comments:
          39 pages

- **What's New**: 이 연구는 COVID-19의 기침 소리를 분류하기 위한 강력한 접근 방식을 제시하며, 최첨단 기계 학습 기술인 deep neural decision trees와 deep neural decision forests를 활용합니다. 다양한 기침 소리 데이터셋에서 일관된 성능을 보이는 우리의 방법론은 음성에서 추출된 다양한 기능을 포괄적으로 분석합니다. 이를 통해 COVID-19 положительного 및 부정적인 결과를 가진 개인의 기침 소리에서 중요한 기능을 효율적으로 선택하여 더 나은 진단 성능을 달성하고자 했습니다.

- **Technical Details**: 우리는 Recursive Feature Elimination with Cross-Validation (RFECV)을 사용하여 기침 소리의 오디오 특성을 추출하고, Bayesian optimization (BO)을 통해 hyper-parameters를 세밀하게 조정합니다. 또한, 데이터 불균형 문제를 해결하기 위해 SMOTE (Synthetic Minority Over-sampling Technique)를 사용하여 긍정적 및 부정적 데이터를 균형 있게 대표하도록 훈련합니다. 우리의 분류 성능은 ROC-AUC 점수를 극대화하는 threshold optimization을 통해 개선되었습니다.

- **Performance Highlights**: 우리는 Cambridge, Coswara, COUGHVID, Virufy 및 NoCoCoDa 데이터셋을 포함하여 5개의 데이터셋에서 포괄적인 평가를 수행하였습니다. 각 데이터셋에서 AUC 점수는 단순히 높았으며, 0.97, 0.98, 0.92, 0.93, 0.99, 0.99를 기록하여 기존의 최첨단 방법들을 초과하는 성능을 입증했습니다. 장기적으로, 우리의 방법은 데이터셋 통합을 통해 COVID-19 감지를 향상시키는 데 잠재적인 이점을 보여줍니다.



### MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification (https://arxiv.org/abs/2501.01110)
Comments:
          Accepted paper at AAAI 2025. 9 pages, Figure 6, Table 1

- **What's New**: 이번 논문은 Generative Replay (GR) 기반의 지속적인 학습(Continual Learning, CL) 시스템인 MalCL을 소개합니다. MalCL은 Generative Adversarial Networks (GANs)를 활용하여 고품질의 악성코드 샘플을 생성하고, 이전의 훈련 데이터를 재사용할 수 있는 혁신적인 선택 기법을 구현하였습니다. 이를 통해 악성코드 분류의 성능을 크게 향상시키고, 새로운 악성코드 유형에 빠르게 적응하는 방법을 제시합니다.

- **Technical Details**: MalCL은 특징 매칭 손실(Feature Matching Loss, FML)을 통합하여 GAN의 생성기가 생성하는 샘플의 품질을 향상시킵니다. 또한, 클래스를 점진적으로 추가하는 학습 시나리오에서 복수의 지속적인 작업을 처리하며, 이전에 관찰된 악성코드 클래스를 유지하고 새로운 클래스를 효율적으로 구분합니다. 기존의 GR 및 Brain-Inspired Replay (BI-R) 모델을 비롯한 다른 접근법들보다 평균 정확도가 55%를 기록하여 28% 높은 성능을 보였습니다.

- **Performance Highlights**: 각기 다른 11개의 지속적 학습 작업을 통해 100개의 악성코드 클래스를 정확히 분류하여, 이전 기술들보다 크게 개선된 결과를 보여주었습니다. 특히, 새로운 샘플과 과거 데이터를 효과적으로 결합하여, 시스템이 업데이트된 이후에도 이전 악성코드를 지속적으로 탐지할 수 있게 하는 방안을 제시합니다. 이로 인해 악성코드 분류 및 탐지에서의 메모리 효율성과 정확도를 높이는데 기여하고 있습니다.



### BatStyler: Advancing Multi-category Style Generation for Source-free Domain Generalization (https://arxiv.org/abs/2501.01109)
Comments:
          Accepted by IEEE TCSVT

- **What's New**: 본 논문에서는 Source-Free Domain Generalization(SFDG) 분야에 대한 새로운 접근 방식을 제안합니다. 'BatStyler'라는 방법은 다중 카테고리 설정에서 스타일 다양성을 향상시키기 위한 두 가지 모듈, 즉 Coarse Semantic Generation과 Uniform Style Generation으로 구성되어 있습니다. 이를 통해 모델은 소스 도메인에 의존하지 않고도 새로운 도메인에서 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: BatStyler의 첫 번째 모듈인 Coarse Semantic Generation은 스타일 학습에서 공간 압축을 방지하기 위해 조잡한 의미 정보를 추출합니다. 두 번째 모듈인 Uniform Style Generation은 스타일 템플릿을 균등 분포로 초기화하여 훈련 효율성을 높이도록 설계되었습니다. 이 방법은 코사인 유사도를 낮춰 스타일 간의 다양성을 증가시키고, 패러렐 훈련을 가능하게 합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, BatStyler는 적은 카테고리 데이터셋에서는 기존의 방법과 유사한 성능을 보여주며, 다중 카테고리 데이터셋에서는 최첨단 방법들을 초월하는 뛰어난 성능을 발휘합니다. 동적으로 다양한 데이터 생성을 통해 실제 세계에서의 적용 가능성을 높이고 있습니다.



### MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization (https://arxiv.org/abs/2501.01108)
- **What's New**: 이 논문에서는 자기 지도 학습(self-supervised learning, SSL) 기반의 새로운 음악 표현 모델인 MuQ를 제안합니다. MuQ는 Mel Residual Vector Quantization (Mel-RVQ)을 사용하여 안정성과 효율성을 높인 음악 이해를 위한 표상 학습을 목표로 합니다. 또한, MuQ는 MuQ-MuLan이라는 음악-텍스트 결합 임베딩 모델을 통해 개선된 성능을 보여주며, 최신 기술 수준(State-of-the-art)에서 성능을 발휘합니다.

- **Technical Details**: MuQ는 랜덤 프로젝션 또는 기존의 뉴럴 코덱을 사용하는 대신 Mel-RVQ를 활용하여 음악 데이터를 처리합니다. Mel-RVQ는 선형 잔여 벡터 양자화(linear Residual Vector Quantization) 구조를 채택하여 Mel 스펙트럼의 양자화(quantization)를 수행하며, 이는 SSL 훈련의 안정성을 높이고 데이터의 양이 적음에도 불구하고 뛰어난 성과를 제공합니다. 또한, MuQ는 이터레이티브 학습을 통해 성능을 지속적으로 향상시킵니다.

- **Performance Highlights**: MuQ는 0.9K 시간의 공개 데이터로 학습했음에도 불구하고 이전의 최고 성능 모델인 MERT와 MusicFM을 초超越하는 성과를 냅니다. 특히, MuQ-MuLan은 MagnaTagATune 데이터셋에서 제로샷 음악 태깅(zero-shot music tagging) 작업에 대해 79.3의 ROC-AUC 점수를 달성하며, 기존의 SOTA 성과를 능가합니다. 이러한 결과는 MuQ가 다양한 음악 이해 작업에서 우수한 SSL 성능을 발휘하는 것을 입증합니다.



### learning discriminative features from spectrograms using center loss for speech emotion recognition (https://arxiv.org/abs/2501.01103)
Comments:
          Accepted at ICASSP 2019

- **What's New**: 이 논문에서는 음성 감정 인식을 위해 소프트맥스 크로스 엔트로피 손실(softmax cross-entropy loss)과 센터 손실(center loss)을 결합하여 새로운 접근 방식이 제안됩니다. 이러한 손실 기능을 통해 다양한 감정 범주의 특징은 구별 가능하게 되고, 동일한 감정 범주 내의 특징을 상호 밀착시키는 효과를 누리게 됩니다. 이를 통해 더욱 효과적인 감정 인식 기능을 학습할 수 있습니다. 실험 결과는 이러한 방법이 멜-스펙트로그램(Mel-spectrogram) 입력에서 3	ext{%}, 단기 푸리어 변환(SHORT TIME FOURIER TRANSFORM, STFT) 스펙트로그램에서 4	ext{%} 이상의 개선을 보였음을 보여줍니다.

- **Technical Details**: 제안된 모델은 2D 컨볼루션 신경망(CNN), 양방향 순환 신경망(Bi-RNN)과 두 개의 전결합층(FC1, FC2)으로 구성됩니다. CNN층은 가변 길이 스펙트로그램으로부터 공간 정보를 추출하고, Bi-RNN은 이를 고정 길이 벡터로 압축합니다. FC1은 Bi-RNN의 출력을 원하는 차원으로 투사하고, FC2는 각 감정 범주에 대한 가능한 클래스 확률을 계산하여 최종적인 손실을 만듭니다.

- **Performance Highlights**: 저자들은 제안한 방법의 성능을 다양한 입력 형식에서 실험하여 확인했습니다. 실험 결과, 제안한 손실 함수를 통해 소프트맥스 손실의 효과와 센터 손실의 통합이 이루어져, 감정 인식 결과의 분별력이 개선되었습니다. 이 새로운 접근법은 기존의 방법들에 비해 더 일관된 성능 향상을 보여주며, 특히 대량의 데이터가 필요한 경우 더 효과적일 것으로 예상됩니다.



### Disambiguation of Chinese Polyphones in an End-to-End Framework with Semantic Features Extracted by Pre-trained BER (https://arxiv.org/abs/2501.01102)
Comments:
          Accepted at INTERSPEECH 2019

- **What's New**: 본 연구에서는 중국어 글자 시퀀스를 입력으로 받아 폴리폰 문자(다의 문자)의 발음을 예측하는 엔드투엔드(End-to-End) 프레임워크를 제안합니다. 이 방법은 사전 훈련된 BERT 모델을 사용하여 효과적인 의미적 특징을 추출하고, 여러 가지 신경망 기반 분류기를 활용해 발음 예측 성능을 향상시킵니다. 또한, 기존의 다의 문자 발음 예측 시스템에 비해 더욱 향상된 성능을 보여주며, 컨텍스트 정보의 영향을 탐구합니다.

- **Technical Details**: 제안된 프레임워크는 사전 훈련된 BERT와 신경망(NN) 기반 분류기로 구성됩니다. BERT는 입력된 중국어 문자 시퀀스에서 의미적 특징을 추출하고, 이 특징을 기반으로 신경망이 분류 작업을 수행합니다. 연구에서는 전결합 네트워크, LSTM, Transformer 블록 기반의 세 가지 분류기를 구현하여 성능을 비교하고 분석하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 LSTM 기반 방식과 비교하여 발음 예측 정확도가 크게 향상되었습니다. 폴리폰 문자 예측에서 비공유 출력 레이어를 사용하여 다른 폴리폰 문자의 발음과 혼동되는 경우를 줄였으며, 새로운 폴리폰 문자가 등장할 때 추가적인 레이어를 통해 손쉽게 훈련할 수 있음을 보여주었습니다. 전체적인 결과는 10-fold 교차 검증을 통해 긍정적인 성과를 입증하였습니다.



### MMVA: Multimodal Matching Based on Valence and Arousal across Images, Music, and Musical Captions (https://arxiv.org/abs/2501.01094)
Comments:
          Paper accepted in Artificial Intelligence for Music workshop at AAAI 2025

- **What's New**: Multimodal Matching based on Valence and Arousal (MMVA)는 이미지, 음악, 음악 캡션을 통해 감정 내용을 포착하기 위해 설계된 삼중 모달 인코더 프레임워크입니다. 이 연구에서는 Image-Music-Emotion-Matching-Net (IMEMNet) 데이터셋을 확장하여 24,756개의 이미지와 25,944개의 음악 클립, 해당하는 음악 캡션을 포함하는 IMEMNet-C를 생성했습니다.

- **Technical Details**: MMVA는 지속적인 Valence(감정 긍정도)와 Arousal(감정 강도) 값을 기반으로 하는 다중 모달 매칭 점수를 사용합니다. 이 지속적인 매칭 점수를 통해 다양한 모달리티 간의 Valence-Arousal 값으로부터 유사성 점수를 계산하여 훈련 중에 이미지-음악 쌍을 무작위로 샘플링 할 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 Valence-Arousal 예측 작업에서 최첨단 성능을 달성하며, 제로샷(zeroshot) 작업에서도 그 효능을 입증합니다. 이는 다운스트림 애플리케이션에서 Valence와 Arousal 예측의 잠재력을 강조합니다.



### Graph Generative Pre-trained Transformer (https://arxiv.org/abs/2501.01073)
Comments:
          preprint

- **What's New**: 이번 논문에서는 그래프 생성의 새로운 접근 방식을 제안합니다. 전통적인 인접 행렬(adjacency matrix) 표현법 대신, 그래프를 노드 집합(node set)과 엣지 집합(edge set)의 시퀀스로 나타내는 방법을 제시합니다. 이 새로운 방식은 그래프를 효율적으로 인코딩할 수 있으며, 이를 바탕으로 Graph Generative Pre-trained Transformer (G2PT) 모델을 소개합니다.

- **Technical Details**: G2PT는 오토회귀(auto-regressive) 모델로, 다음 토큰 예측(next-token prediction)을 통해 그래프 구조를 학습합니다. 그래프는 두 부분으로 구성된 시퀀스로 표현되며, 첫 번째 부분은 노드 정의(node definition)이고, 두 번째 부분은 엣지 정의(edge definition)입니다. G2PT는 Transformer 디코더(transformer decoder)를 활용하여 시퀀스 분포를 근사하며, 다양한 다운스트림 작업에 대한 적합성을 평가합니다.

- **Performance Highlights**: G2PT는 여러 데이터셋을 대상으로 한 실험에서 기존의 최첨단(SOTA) 성능을 초과하거나 동등한 결과를 보여줍니다. 목표 지향 생성(goal-oriented generation) 및 그래프 속성 예측(graph property prediction)와 같은 다운스트림 작업에 대한 적합성을 보여줍니다. G2PT는 분자 설계와 속성 예측을 포함한 다양한 작업에서 강력한 적응성과 다재다능함을 발휘합니다.



### Risks of Cultural Erasure in Large Language Models (https://arxiv.org/abs/2501.01056)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 사회적 지식의 생산과 발견에 통합되어가는 추세를 강조합니다. 언어 모델이 글로벌 문화에 대한 사람들의 학습 및 인식 방식에 미치는 영향을 고려할 필요성이 증가하고 있으며, 특히 디지털 데이터에서 저조하게 나타나는 문화들을 다뤄야 한다고 주장합니다. 연구자들은 문화적 삭제(cultural erasure)라는 개념을 도입해 언어 모델이 문화적 다양성을 어떻게 왜곡할 수 있는지를 평가합니다.

- **Technical Details**: 본 연구는 문화 삭제를 평가하기 위해 두 가지 개념인 생략(omission)과 단순화(simplification)를 구분합니다. 생략은 문화가 전혀 표현되지 않는 경우이고, 단순화는 문화의 복잡성이 왜곡되어 일차원적인 관점으로 나타날 때를 의미합니다. 연구진은 전 세계 50개 도시의 설명을 생성하여 문화와 경제에 관한 주제를 분석하고, 이를 통해 도시들 간의 문화적 및 경제적 표현의 차이를 분석합니다.

- **Performance Highlights**: 연구 결과, 유럽 및 북미의 도시에서 문화 주제가 더 높은 점수를 얻은 반면, 아프리카와 아시아의 도시들은 경제 주제로 높은 점수를 받았습니다. 결과적으로 이 논문은 LLMs가 문화적 대표성을 어떻게 왜곡하는지에 대한 구체적인 증거를 제시하며, 특히 아프리카의 문화들이 심각하게 생략됨을 강조합니다. 연구자들은 이러한 결과를 바탕으로 언어 모델이 문화적 발견 및 생산을 지원하는 데 미치는 실질적인 영향을 논의합니다.



### MSWA: Refining Local Attention with Multi-ScaleWindow Attention (https://arxiv.org/abs/2501.01039)
- **What's New**: 이번 논문에서는 Multi-Scale Window Attention (MSWA)라는 새로운 주의 메커니즘을 제안합니다. 기존의 Sliding Window Attention (SWA) 방식은 모든 헤드에서 동일한 창 크기를 사용하여 다양한 스케일의 맥락을 포착하는 데 비효율적이었습니다. MSWA는 서로 다른 창 크기를 적용하여 다양한 길이와 거리의 맥락 정보를 효과적으로 캡처할 수 있도록 합니다. 실험 결과에 따르면, MSWA는 전통적인 로컬 주의 방식보다 더 높은 효율성과 효과성을 보여줍니다.

- **Technical Details**: Multi-Scale Window Attention (MSWA)은 Transformer의 여러 레이어와 헤드에 걸쳐 다양한 창 크기를 적용합니다. 이 방법은 얕은 레이어에서 더 작은 창 크기를 할당하고 깊은 레이어로 갈수록 더 큰 창 크기를 할당하여, 모델이 지역 정보를 모델링하고 장거리 의존성을 포착할 수 있도록 지원합니다. 또 다른 유용한 점은 MSWA가 linear attention과 통합되어 효율성과 글로벌 주의력을 모두 갖출 수 있다는 것입니다. 이 접근 방식은 기존의 attention 가속 라이브러리에서 직접 구현 가능하다는 특징이 있습니다.

- **Performance Highlights**: 언어 모델링과 일반 상식 추론 과제에서 MSWA는 기존의 SWA보다 뛰어난 성능을 보였습니다. MSWA를 활용하여 학습한 모델은 효과적인 언어 모델링 능력을 입증했으며, 사전 훈련된 대형 언어 모델에 MSWA 패턴을 fine-tuning하여 향상된 결과를 도출하였습니다. 또한, 계산 효율성을 평가한 결과 MSWA는 표준 주의 방식이나 SWA에 비해 일관되게 더 나은 효율성을 기록하였습니다.



### MSC-Bench: Benchmarking and Analyzing Multi-Sensor Corruption for Driving Perception (https://arxiv.org/abs/2501.01037)
- **What's New**: 이번 연구에서는 다중 센서 융합 모델의 안전성 및 강건성을 평가하기 위해 Multi-Sensor Corruption Benchmark (MSC-Bench)를 제시합니다. 이 벤치마크는 독특하게 16종의 감염 유형을 포함하여 카메라 및 LiDAR 입력의 이탈을 검토합니다. 연구 결과에 따르면, 3D 물체 감지 및 HD 맵 구축 모델 모두 센서 손상에 취약하다는 것이 드러났습니다.

- **Technical Details**: MSC-Bench는 환경적 요인에 따른 다양한 센서 고장을 평가하기 위해 16개의 감염 유형으로 구성되어 있습니다. 각 감염 유형은 날씨, 실내, 센서 고장 시나리오로 분류되며, 세부사항에는 Camera Crash, Frame Lost, Cross Sensor 등이 포함됩니다. 연구자들은 이러한 감염 시나리오가 자율주행 인식 시스템의 성능에 미치는 영향을 평가하였습니다.

- **Performance Highlights**: 결과적으로, 연구에서 분석된 6개의 3D 물체 감지 및 4개의 HD 맵 구축 모델은 예상보다 성능 저하가 크게 나타났습니다. 특히 악천후나 센서 실패와 같은 조건에서 모델의 강건성이 급격히 낮아지는 것을 확인하였습니다. 연구팀은 이러한 발견이 향후 센서 융합 모델의 신뢰성을 높이기 위한 설계 개선에 기여할 것으로 보았습니다.



### ValuesRAG: Enhancing Cultural Alignment Through Retrieval-Augmented Contextual Learning (https://arxiv.org/abs/2501.01031)
Comments:
          preprint

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG)과 in-context learning을 활용한 ValuesRAG라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 텍스트 생성 과정에서 문화적 및 인구통계학적 지식을 동적으로 통합하여 문화적 가치 정렬 문제를 해결하고자 합니다. ValuesRAG는 World Values Survey (WVS) 데이터를 활용하여 개인별 가치 요약을 생성하고, 인구통계학적 특징에 따라 관련된 요약을 검색하여 최종 결과를 도출합니다.

- **Technical Details**: ValuesRAG는 WVS 데이터셋을 기반으로, 논문에서는 먼저 다양한 인구통계학적 특성에 따라 100개의 관련된 요약을 검색하고, 이를 재정렬하여 상위 k개의 요약을 선택하는 방법을 설명합니다. 아울러, 성능 평가를 위해 zero-shot 추론, 역할 부여 방법, 몇샷 학습 방법 등 여러 기법들과 비교하여 ValuesRAG의 뛰어난 성능을 강조합니다. 이 과정에서 ValuesRAG는 다수의 인구통계적 요약을 동적으로 검색하고 통합하여 단일 정의된 프롬프트나 역할에 의존하지 않고 더 풍부한 가치 표현을 가능하게 합니다.

- **Performance Highlights**: ValuesRAG는 다양한 기준선 방법들과 비교했을 때 문화적 및 맥락적 이해 측면에서 유의미한 향상을 보여주었습니다. 특히, 단순한 값 요약만 제공된 상태에서도 우수한 성과를 나타내며, 문화적으로 균형 잡힌 AI 시스템을 육성하고 AI 응용 프로그램의 포용성을 증진시키는 잠재력을 가지고 있음을 입증했습니다. 이 연구는 ValuesRAG가 정책 입안자와 다양한 분야의 과학자들이 사회 시뮬레이션을 개선하고 공정하고 효과적인 정책을 수립하는 데 에 기여할 수 있음을 제안합니다.



### Reasoning based on symbolic and parametric knowledge bases: a survey (https://arxiv.org/abs/2501.01030)
- **What's New**: 이 논문은 인간의 지능에서 필수적인 추론(reasoning)의 접근 방식에 대한 새로운 관점을 제시합니다. 기존의 연구들은 지식 기반(knowledge base)에 기반한 추론 방법을 체계적으로 분석하지 않았으나, 이 연구는 이를 상징적(symbolic) 및 매개변수적(parametric) 지식 기반으로 분류하여 그 차이를 설명합니다. 이러한 접근을 통해 추론 방법의 새로운 이해와 향후 연구 방향을 제시하고자 합니다.

- **Technical Details**: 연구는 지식 기반을 두 가지 유형으로 분류하였습니다. 상징적 지식 기반은 인간이 이해할 수 있는 기호로 정보를 표현하고, 매개변수적 지식 기반은 매개변수 내에 지식을 암묵적으로 인코딩합니다. 이러한 접근은 추론 방법에 사용되는 지식의 저장 방식과 적용 시나리오를 명확히 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 이 논문은 상징적 및 매개변수적 지식 기반을 기반으로 한 다양한 추론 방법들을 자세히 조사합니다. 특히, 이들은 다양한 실제 응용 프로그램에서의 성능 개선에 기여할 수 있는 중요한 요소들입니다. 추론의 도전 과제와 미래의 연구 방향도 체계적으로 정리하였습니다.



### Towards Adversarially Robust Deep Metric Learning (https://arxiv.org/abs/2501.01025)
- **What's New**: 이 연구에서는 Deep Metric Learning (DML) 모델의 클러스터링 기반 추론(Clustering-based Inference)에서의 강건성 문제를 처음으로 지적합니다. 현재까지의 DML 모델에 대한 방어 체계는 주로 깊은 분류 모델에서 파생된 내용이지만, 클러스터링 시나리오와 같은 DML 고유의 환경에서는 효과적이지 않음을 보여주었습니다. 이를 해결하기 위해 새로운 방어 방법인 Ensemble Adversarial Training (EAT)을 제안하였습니다.

- **Technical Details**: EAT는 앙상블(Ensemble) 학습과 적대적 훈련(Adversarial Training)을 융합하여 DML 모델의 강건성을 획기적으로 향상시킵니다. 이 방식은 모델 다양성을 증진시키고, 각 모델이 다양한 강건성 특성을 갖도록 유도하여 동시에 모든 모델이 동일한 표본에 의해 완전히 속지 않도록 합니다. 또한, EAT는 엔셈블의 강건성 통계를 각 모델 업데이트에 활용하는 자기 전이(self-transferring) 메커니즘을 적용하여 모델의 강건성을 더욱 강화합니다.

- **Performance Highlights**: EAT 방법은 세 개의 널리 사용되는 데이터셋(CUB200, CARS196, In-Shop)과 두 가지 모델(MobileNet-V2, BN-Inception)에서 평가되었습니다. 실험 결과, EAT 방법은 기존의 깊은 분류 모델을 위한 방어 기법의 적응 형태들과 비교하여 월등한 성능을 보였습니다. 이와 함께, DML 작업에 대한 성능 저하를 최소화하면서 모델의 강건성을 크게 향상시키는 것을 확인하였습니다.



### MDSF: Context-Aware Multi-Dimensional Data Storytelling Framework based on Large language Mod (https://arxiv.org/abs/2501.01014)
- **What's New**: 본 논문에서는 데이터 분석 및 스토리텔링을 자동화하기 위한 다차원 데이터 스토리텔링 프레임워크(MDSF)를 소개합니다. MDSF는 대형 언어 모델(LLMs)을 기반으로 하여, 자동적으로 인사이트를 생성하고 맥락에 맞는 스토리를 제공하는 기능을 제공합니다. 이 프레임워크는 고급 전처리 기술, 증강 분석 알고리즘, 그리고 실행 가능한 인사이트를 식별하고 우선순위를 매기는 독특한 점수 매커니즘을 통합하고 있습니다.

- **Technical Details**: MDSF는 숫자형, 범주형, 시계열, 공간 데이터 등 다양한 다차원 데이터를 처리 및 통합할 수 있는 기능을 갖추고 있습니다. 이를 통해 MDSF는 더 포괄적이고 깊이 있는 데이터 내러티브를 생성할 수 있으며, LLM의 강력한 자연어 처리 및 생성 능력을 활용하여 매력적이고 일관된 이야기를 자동으로 생성하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 평가 결과, MDSF는 기존 방법들에 비해 인사이트 랭킹 정확도, 설명적 품질, 내러티브 일관성 측면에서 뛰어난 성능을 보였습니다. 사용자 연구 또한 MDSF의 실질적인 유용성을 강조하며, 콘텐츠 구조 강화, 결론 추출 및 세부 사항의 풍부함을 개선하는 데 기여했습니다.



### CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction (https://arxiv.org/abs/2501.01010)
- **What's New**: 이번 논문에서는 Bitcoin 가격 예측의 어려움을 해결하기 위해 새로운 Mamba 기반 State Space Model (SSM) 구조인 CryptoMamba를 제안합니다. CryptoMamba는 금융 시계열 데이터에서 장기 종속성을 효과적으로 포착하도록 설계되었으며, 기존 모델들의 한계를 뛰어넘는 예측 정확도를 보여주었습니다. 이 모델은 다양한 시장 조건에서의 일반화 능력을 향상시켜 실제 거래 알고리즘과 결합하여 실질적인 유용성을 증명했습니다.

- **Technical Details**: CryptoMamba는 상황 변화에 대비해 시계열 데이터를 잠재 상태와 관측 변수의 조합으로 모델링하는 State Space Models (SSMs)을 기반으로 합니다. 특히, Mamba는 입력 데이터에 적응할 수 있는 선택성 메커니즘을 도입하여 시퀀스 모델링의 challenges를 해결합니다. 이 모델은 실험을 통해 거래 알고리즘과 함께 사용될 때 금융적인 결과로 이어지는 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: CryptoMamba는 여러 기준 모델들과 비교하여 예측 정확도 및 재무 수익성에서 우수한 성능을 보여주었습니다. 이 연구는 SSM의 진전을 금융 예측의 실용적인 응용에 연결하고, 향후 적응형 및 강건한 시장 예측 기법에 대한 연구의 길을 열어줍니다. 전체적으로 CryptoMamba는 비트코인 가격 예측 및 암호화폐 시장에서의 활용 가능성이 높은 모델로 자리매김하고 있습니다.



### Deep Reinforcement Learning for Job Scheduling and Resource Management in Cloud Computing: An Algorithm-Level Review (https://arxiv.org/abs/2501.01007)
- **What's New**: 이 논문에서는 클라우드 컴퓨팅에서의 작업 스케줄링(job scheduling)과 자원 관리(resource management)를 위한 Deep Reinforcement Learning (DRL) 기반 알고리즘에 대한 포괄적인 리뷰를 제공합니다. DRL은 기존의 정적 모델이나 사전 정의된 규칙에 의존하는 전통적인 방법들보다 동적이고 예측할 수 없는 클라우드 환경에 더 잘 적응할 수 있는 솔루션으로 자리 잡았습니다. 이 리뷰는 DRL 알고리즘의 방법론, 성능 지표 및 실제 응용 분야를 분석하고, 이 분야에서의 최신 동향과 연구 방향을 제시합니다.

- **Technical Details**: 클라우드 컴퓨팅은 IaaS, PaaS 및 SaaS와 같은 다양한 서비스 모델을 통해 제공됩니다. DRL은 강화 학습과 심층 신경망(deep neural networks)을 결합하여 환경과 지속적으로 상호작용함으로써 최적의 정책을 학습할 수 있게 해줍니다. 이를 통해 DRL 알고리즘은 실시간 피드백을 기반으로 클라우드 환경의 현재 상태에 맞춰 정보에 입각한 결정을 내릴 수 있으며, 이는 복잡한 작업 스케줄링 및 자원 배분 전략을 개발하는 데 기여합니다.

- **Performance Highlights**: DRL을 통해 클라우드 컴퓨팅에서 작업 스케줄링과 자원 관리를 최적화하는 데 성공적인 사례가 많이 보고되고 있습니다. 이러한 방법은 자원 활용도를 극대화하고 시스템 성능 또는 서비스 품질(QoS)을 향상시키는 데 중요한 역할을 합니다. 이 논문은 DRL 기반 접근 방식의 향후 개발 방향을 제시하며, 개인 정보 보호 강화, DRL 프레임워크의 안전성 및 확장성 향상, 그리고 동적 환경에의 적용 가능성을 넓히는 등의 기회를 강조하고 있습니다.



### FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (https://arxiv.org/abs/2501.01005)
Comments:
          code available at this http URL

- **What's New**: FlashInfer는 LLM(대형 언어 모델) 서비스를 위한 사용자 정의 가능하고 효율적인 attention 엔진입니다. 이 시스템은 KV-cache 저장소의 이질성을 해결하기 위해 블록 스파스 형식을 사용하여 메모리 접근 최적화 및 중복성을 줄입니다. 더불어 FlashInfer는 Just-In-Time (JIT) 컴파일을 통해 다양한 설정에 적응할 수 있는 사용자 정의 가능 attention 템플릿을 제공합니다.

- **Technical Details**: FlashInfer는 KV-cache의 다양한 형상을 통합하기 위해 블록-스파스 형식을 활용하며, 이를 통해 메모리 접근의 효율성을 높입니다. 사용자 정의 프로그래밍 인터페이스를 제공하여 사용자가 자신의 attention 변형을 직접 구현할 수 있으며, JIT 컴파일을 사용하여 이러한 변형을 최적화된 블록-스파스 구현으로 변환합니다. 또한, 동적 로드 균형 스케줄링 프레임워크를 통해 입력 동적성을 효과적으로 관리합니다.

- **Performance Highlights**: FlashInfer는 표준 LLM 서비스 환경에서 성능 평가를 실시하였으며, 다양한 인퍼런스 시나리오에서 커널 성능을 크게 향상시켰습니다. 기존의 LLM 서비스 솔루션과 비교했을 때 inter-token latency를 29-69% 줄이고, 긴 문맥 불러오기 인퍼런스에서 28-30% 감소, 병렬 생성의 경우 13-17% 속도 향상을 달성했습니다.



### Exploring Information Processing in Large Language Models: Insights from Information Bottleneck Theory (https://arxiv.org/abs/2501.00999)
Comments:
          9 pages, 9 figures, 3 tables

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 정보 처리 메커니즘을 Information Bottleneck Theory의 관점에서 탐구합니다. 기존의 연구와 차별화되는 점은 비훈련 기반(task space detection) 접근 방식을 사용하여 LLM의 내부 정보 흐름을 추적하고, 특정 작업 공간으로의 입력 정보 압축 및 예측시 관련 정보를 추출하는 과정을 밝힌 것입니다. 이를 통해 LLM의 예측 성능과 추론 과정을 개선할 수 있는 두 가지 새로운 방법인 정보 압축 기반의 맥락 학습(IC-ICL)과 작업 공간 안내 미세 조정(Task-Space-guided Fine-Tuning, TS-FT)을 제안합니다.

- **Technical Details**: 논문에서는 LLM이 정보를 압축하고 풀이하는 과정을 명확히 하기 위해 작업 공간(task space)을 구성하고 이를 기반으로 LLM 내부의 정보 흐름을 탐지합니다. 각 작업 공간은 해당 작업을 가장 잘 나타내는 기본 벡터로 구성되며, 감정 분류 같은 특정 태스크를 예로 들어 이러한 벡터가 어떻게 형성되는지를 설명했습니다. 또한, PCA(Principal Component Analysis) 기법을 통해 작업 공간 내에서 노이즈를 제거한 후, 정보의 압축과 풀이 메커니즘이 어떻게 LLM의 성능에 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면, IC-ICL은 추론 속도를 40% 이상 가속화하며, LLM의 예측 정확성을 크게 향상시킨 것으로 나타났습니다. 또한, TS-FT는 복잡한 조정 없이도 모델 성능을 개선시키는 데 효과적입니다. 이러한 성과는 LLM이 작업 공간 내에서 정보를 효과적으로 처리함으로써 이루어진 결과로, 연구진은 제안한 방법들이 여러 데이터셋에서 유효성을 입증했다고 강조합니다.



### Bootstrapped Reward Shaping (https://arxiv.org/abs/2501.00989)
Comments:
          Accepted at AAAI-2025, Main Track

- **What's New**: 이 논문에서는 부트스트랩된 잠재 보상 형태(bootstrapped reward shaping, BSRS)를 제안합니다. 기존의 잠재 기반 보상 형태(Potential-Based Reward Shaping, PBRS)와 달리, 에이전트의 현재 상태 가치 함수(state-value function)를 잠재 함수로 사용하여 매끄러운 보상 신호를 제공합니다. 이러한 방식은 에이전트가 이해하는 바에 따라 진화하는 적응형 보상 신호를 생성하여, 훈련 성능을 향상시킵니다.

- **Technical Details**: 논문에서는 상태 공간(𝒮)과 행동 공간(𝒜)을 포함하는 마르코프 결정 프로세스(MDP) 모델을 사용하여 강화 학습 문제를 정의합니다. BSRS 방법은 에이전트의 현재 최적 상태 가치 함수의 추정치를 이용하여, 과거의 학습을 기반으로 한 보상 형태를 제공합니다. 이 방식은 값 기반(value-based) 강화 학습 기법에서 구현되며, 단 한 줄의 코드 변경만으로 기존 알고리즘에 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, BSRS 알고리즘은 Atari 게임과 같은 복잡한 환경에서도 훈련 속도를 개선하는 데 성공했습니다. 특히, BSRS를 사용하면 훈련 샘플 복잡성(sample complexity)이 줄어드는 효과가 있는 것으로 나타났습니다. 이 연구는 기존의 강화 학습 문제 해결에 대한 새로운 접근법을 제공하며, 보상 설계의 난제를 해결할 수 있는 새로운 경로를 제시합니다.



### Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practic (https://arxiv.org/abs/2501.00982)
- **What's New**: 이 연구에서는 심리적 질문지를 보완하기 위해 소셜 미디어 게시물을 분석하는 새로운 적응형 Retrieval-Augmented Generation (RAG) 접근 방식을 제안합니다. 사용자의 게시물 데이터에서 가장 관련 있는 내용을 추출하여, 대형 언어 모델(LLMs)을 사용해 질문지 점수를 예측하는 방법입니다. 이 방법은 Reddit 기반 벤치마크 데이터 세트에서 기존의 최첨단 모델과 유사하거나 우수한 성능을 나타내며, 훈련 데이터에 의존하지 않고도 정신 건강 평가를 가능하게 합니다.

- **Technical Details**: 연구는 Beck Depression Inventory II (BDI-II)와 Self-Harm Inventory (SHI)라는 두 가지 표준화된 심리적 질문지를 중심으로 진행됩니다. 또한, 연구진은 개별 사용자의 Reddit 게시물 이력을 분석하여 질문 항목에 대한 응답을 예측하는 적응형 RAG 접근 방식을 구현했습니다. 이는 심리적 질문지에 대한 응답을 보다 정확하게 추정하기 위한 것으로, 데이터 접근성의 향상과 함께 기계 학습 기술을 활용합니다.

- **Performance Highlights**: 최신 연구 결과에 따르면, 제안된 접근 방법은 비지도 학습 환경에서도 효과적으로 작동할 수 있다는 것을 보였습니다. eRisk 데이터 세트를 사용한 실험에서, 이 방법은 기존의 감독 기반 모델과 비교하여 동등하거나 더 나은 성능을 보였습니다. 이에 따라, 연구의 성과는 새로운 행동 장애 예측을 위한 해석 가능하고 비지도 학습 가능한 방법론으로 확장될 수 있는 가능성을 보여줍니다.



### The Silent Majority: Demystifying Memorization Effect in the Presence of Spurious Correlations (https://arxiv.org/abs/2501.00961)
- **What's New**: 이 연구는 기계 학습 모델이 소수 그룹에서 불균형한 성능을 보이는 근본 원인인 스푸리어스 메모리(Spurious Memorization)에 대해 체계적으로 분석하였습니다. 저자들은 모델의 특정 뉴런이 소수 그룹 정보를 어떻게 기억하는지에 대한 최초의 실증적 증거를 제시하고, 이 메모리화가 불균형한 그룹 성능에 기여할 수 있음을 확인하였습니다. 이로 인해 스푸리어스 속성을 제거하는 새로운 프레임워크를 통해 모델의 소수 그룹에 대한 성능을 크게 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 두 가지 실험적 접근법을 통해 소수 그룹 정보 메모리화를 위한 중요한 뉴런의 존재를 입증했습니다. 첫 번째는 비구조적 추적(Unstructured Tracing)으로 뉴런의 역할을 전체 모델 수준에서 평가하고, 두 번째는 구조적 추적(Structured Tracing)으로 각 층의 뉴런 역할을 분석했습니다. 스푸리어스 메모리의 응집 성질을 확인하기 위해, 변형된 모델을 활용하여 주요 뉴런의 변화를 명확하게 이해하고, 이러한 뉴런들이 소수 그룹 성능에 미치는 영향을 면밀하게 조사했습니다.

- **Performance Highlights**: 이 실험 결과들은 소수 그룹의 성능이 특정 중요한 뉴런에 의해 크게 좌우되며, 이러한 뉴런들은 전체 모델 파라미터의 극히 일부를 차지한다는 것을 보여줍니다. 즉, 대다수 그룹의 경우, 전체 네트워크가 메모리를 형성해 Robust한 테스트 성능을 보이는 반면, 소수 그룹 예제는 제한된 뉴런 집합에 의존하여 Poor한 테스트 성능을 나타냅니다. 이로 인해 불균형한 성능 패턴에 대한 강력한 설명이 제공됩니다.



### Enhancing Early Diabetic Retinopathy Detection through Synthetic DR1 Image Generation: A StyleGAN3 Approach (https://arxiv.org/abs/2501.00954)
Comments:
          13 pages, 11 figures

- **What's New**: 이번 연구에서는 StyleGAN3를 활용하여 당뇨병성 망막병증(Diabetic Retinopathy, DR) 1단계를 재현한 합성 이미지를 생성하였습니다. 이러한 이미지들은 미세 혈관 팽창증(microaneurysms)을 특징으로 하며, 높은 충실도(fidelity)와 다양성을 자랑합니다. 이는 고품질의 fundus images 부족 문제를 해결하고, 감독(Classifier) 학습 알고리즘의 성능 향상에 기여하고자 합니다.

- **Technical Details**: 연구팀은 2,602개의 DR1 이미지를 사용하여 모델을 훈련시키고, Frechet Inception Distance (FID), Kernel Inception Distance (KID), 그리고 변환에 대한 동등성(EQ-T) 및 회전에 대한 동등성(EQ-R)과 같은 정량적(quantitative) 지표를 통해 평가하였습니다. 또한, 숙련된 안과 의사들이 합성 이미지의 현실성을 평가하는 'Human Turing test'를 포함한 질적(qualitative) 평가를 진행했습니다. 스펙트럴 분석(spectral analysis) 또한 이미지 품질을 검증하는 데 기여하였습니다.

- **Performance Highlights**: 모델은 최종 FID 점수를 17.29로 기록하였으며, 이는 부트스트랩 재샘플링을 통해 도출된 평균 FID 점수 21.18(95% 신뢰구간 - 20.83 ~ 21.56)보다 우수한 결과입니다. Human Turing test 결과, 모델이 매우 현실적인 이미지를 생성할 수 있는 능력을 입증하였으나, 이미지의 경계 근처에 미세한 아티팩트가 발견되었습니다. 이러한 결과는 StyleGAN3가 생성한 합성 DR1 이미지가 데이터 세트를 보강하고, 당뇨병성 망막병증의 조기 발견을 더욱 정확하게 할 수 있도록 도와줄 잠재력이 크다는 것을 시사합니다.



### Incremental Dialogue Management: Survey, Discussion, and Implications for HRI (https://arxiv.org/abs/2501.00953)
Comments:
          16 pages

- **What's New**: 이 논문에서는 로봇의 대화 시스템이 인간처럼 언어를 처리할 수 있는 방법으로, 특히 단어 단위로 작동하는 점진적인 시스템(incremental systems)의 필요성을 강조합니다. 기존의 대화 시스템들은 문장 수준의 입력만을 처리하며, 이는 인간과 로봇의 자연스러운 상호작용을 방해합니다. 연구자들은 로봇의 대화 관리(Dialogue Management) 부문에서의 연구 부족을 지적하고, 이를 보완하기 위한 요구 사항을 제시합니다.

- **Technical Details**: 점진적인 음성 대화 시스템(incremental spoken dialogue systems, sdss)은 단어 단위로 입력 및 출력을 처리할 수 있습니다. 이 시스템은 주로 다음과 같은 모듈들로 구성됩니다: 자동 음성 인식(ASR), 자연어 이해(NLU), 대화 관리(DM), 자연어 생성(NLG), 음성 변환(Text-to-Speech, TTS). 논문에서는 이러한 모듈들이 단어 수준에서 상호작용해야 함을 강조하며, 이를 통해 시스템의 성능을 향상시킬 수 있음을 설명합니다.

- **Performance Highlights**: 연구에 따르면, 점진적인 시스템은 비점진적 시스템에 비해 자연스러운 상호작용을 제공하며, 사용자들로부터 긍정적인 피드백을 받았습니다. 점진적인 대화 관리가 효과적으로 성능을 개선할 수 있다는 결과도 언급됩니다. 이러한 연구 결과는 로봇의 대화 시스템이 보다 인간과 유사하게 작동하고, 높은 수준의 응답성을 제공할 수 있는 가능성을 제시합니다.



### $\beta$-DQN: Improving Deep Q-Learning By Evolving the Behavior (https://arxiv.org/abs/2501.00913)
- **What's New**: 이 논문은 $eta$-DQN이라는 새로운 탐험 방법을 소개합니다. 기존의 DQN에 행동 함수 $eta$를 추가하여, 각 상태에서 각 행동이 선택될 확률을 추정합니다. 이를 통해 상태-행동 커버리지와 과대 추정 편향 보정 간의 균형 잡힌 탐험을 수행하는 다양한 정책 집합을 생성합니다. 이 방법은 구현이 용이하고, 기존의 DQN에 비해 계산 비용이 최소화됩니다.

- **Technical Details**: $eta$-DQN은 세 가지 목적을 위해 $eta$를 활용합니다: 첫 번째는 상태-행동 커버리지를 향상하기 위한 탐험, 두 번째는 Q-function의 과대 추정 편향을 교정하기 위한 탐험, 세 번째는 순수한 착취를 위한 것입니다. 이러한 정책은 에피소드마다 효과적인 정책을 선택하는 적응형 메타 컨트롤러에 의해 관리됩니다. 이 과정에서는 명확한 목적과 함께 탐험과 착취를 교차하여 수행합니다.

- **Performance Highlights**: $eta$-DQN은 MinAtar와 MiniGrid와 같은 다양한 탐험 도메인에서 기존 방법들보다 우수한 성능을 보였습니다. 실험 결과는 이 방법이 심지어 어려운 탐험 환경에서도 넓은 적용 가능성을 가지며, 성능을 현저히 증가시킨다는 것을 보여줍니다. 이러한 결과는 $eta$-DQN이 딥 강화 학습에서 탐험을 향상시킬 수 있는 효과적인 솔루션임을 의미합니다.



### Population Aware Diffusion for Time Series Generation (https://arxiv.org/abs/2501.00910)
Comments:
          Accepted for publication at AAAI-2025, 8 pages

- **What's New**: 이번 연구에서는 기존의 시계열 (Time Series, TS) 생성 모델의 한계를 극복하기 위해 새로운 모델인 PaD-TS(Population-aware Diffusion for Time Series)를 제안합니다. PaD-TS는 데이터의 개별 수준에서의 진위를 유지하는 데 중점을 두는 대신, 전체 데이터셋의 집단 수준 특성을 보존하기 위한 새로운 훈련 방식을 도입하고 있습니다. 이 모델은 집단 수준의 특성을 효과적으로 캡처할 수 있는 이중 채널 인코더 아키텍처를 채택하여 데이터 생성의 질을 향상시키고 있습니다.

- **Technical Details**: PaD-TS는 생성 과정에서 집단 수준 분포 변화를 페널티로 부과하는 새로운 훈련 목표를 설정합니다. 그리고 훈련 시 동일한 확산 단계의 두 분포를 비교하는 샘플링 전략을 사용하여 이러한 변화를 강제합니다. 이러한 방법론은 PaD-TS가 기존 모델들과 비교했을 때, 데이터의 집단 수준 특성을 보다 잘 보존할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, PaD-TS는 실제와 합성 데이터 간의 평균 CC (cross-correlation) 분포 변화 점수를 5.9배 개선시킬 수 있었으며, 개별 수준의 진위에서도 기존 최첨단 모델들과 유사한 성능을 유지합니다. 이러한 성과는 TS 생성 모델링의 새로운 가능성을 열어줄 것으로 기대됩니다.



### Large Language Model Based Multi-Agent System Augmented Complex Event Processing Pipeline for Internet of Multimedia Things (https://arxiv.org/abs/2501.00906)
- **What's New**: 이 논문은 비디오 쿼리 처리(use cases)와 관련된 복잡한 이벤트 처리를 위한 다중 에이전트 시스템 프레임워크를 개발하고 평가한 내용을 담고 있습니다. 주요 목표는 현재의 복잡한 이벤트 처리(CEP) 시스템과 통합하기 위해 최신 LLM 오케스트레이션 프레임워크와 퍼블리시/서브스크라이브(pub/sub) 도구를 통합한 개념 증명을 만드는 것입니다. Autogen 프레임워크와 Kafka 메시지 브로커를 활용하여 자율적 CEP 파이프라인을 시연하며, 다양한 구성과 복잡성, 비디오 해상도에서 시스템 성능을 평가하였습니다.

- **Technical Details**: 복잡한 이벤트 처리에서 LLM(large language model) 에이전트를 사용하여 기존 CEP 파이프라인을 보강하는 방법을 모색하고 있습니다. LLM 기반 다중 에이전트 시스템(LLM-MAS)이 있는 환경에서 복잡한 이벤트 처리 파이프라인을 구현하기 위한 전략적 접근 방식을 설명합니다. 연구에서는 Autogen을 사용한 LLM-MAS 프레임워크와 Kafka 같은 pub/sub 도구의 사용 가능성을 조사하여, 기존 시스템 및 기술과 효과적으로 인터페이스를 통해 원활한 전환을 보장합니다.

- **Performance Highlights**: 실험 결과, 에이전트 수와 비디오 복잡성이 증가할수록 지연 시간(latency)이 증가하는 경향이 있지만, 서술의 일관성(narrative coherence)은 높은 수준을 유지합니다. CEP 파이프라인이 다양한 사용 사례와 요구 사항에 적합한지를 평가하며, POC(Proof-of-Concept) 시스템이 획득한 결론 및 향후 연구 방향에 대해서도 논의하였습니다. 이 연구는 분산 AI 시스템에 대한 기존 접근 방식을 발전시키고, CEP 기술을 기존 인프라에 통합하는 데 있어 중요한 통찰력을 제공합니다.



### Demystifying Online Clustering of Bandits: Enhanced Exploration Under Stochastic and Smoothed Adversarial Contexts (https://arxiv.org/abs/2501.00891)
- **What's New**: 이 논문에서는 사용자의 클러스터를 더 정확하게 식별하기 위한 새로운 알고리즘 UniCLUB와 PhaseUniCLUB을 제안합니다. 이 알고리즘들은 기존의 Upper Confidence Bound (UCB) 전략보다 훨씬 약한 가정을 만족시키며, 더 나은 성능을 보여줍니다. 또한, 새로운 설정인 smoothed analysis framework를 적용하여 기계 학습의 효율성을 높입니다.

- **Technical Details**: 기존의 MAB 문제는 강화 학습의 일종으로, 행위를 선택하고 이에 대한 보상 값을 관찰하는 구조로 이루어집니다. 논문에서는 기존 연구의 스토캐스틱 문맥 생성을 유지하면서, 추가 탐색 메커니즘을 포함하여 클러스터 정보를 더 효과적으로 수집할 수 있도록 합니다. 또한 두 가지 접근법을 제안하여 클러스터 추론과 누적 후회 최소화를 동시에 달성합니다.

- **Performance Highlights**: 제안된 알고리즘 UniCLUB과 PhaseUniCLUB은 합성 및 실제 데이터셋에서 기존 방법보다 일관되게 더 나은 성능을 발휘했습니다. 특히, 이들은 이상적인 설정에서의 누적 후회를 최소화하면서도 강한 가정을 제거할 수 있습니다. 이러한 결과는 MAB 연구 간의 간격을 줄이고 실제 적용 가능성을 높이는 데 기여합니다.



### Representation in large language models (https://arxiv.org/abs/2501.00885)
Comments:
          Draft of paper under review. 27 pages, 2 figures

- **What's New**: 본 논문은 최근의 대형 언어 모델(Large Language Models, LLMs)이 어떻게 작동하는지에 대한 근본적인 이론적 질문에 대해 다룹니다. 특히 LLM의 행동이 생물학적 인지에 연관된 정보 처리 방식에 의해 부분적으로 구동되는지, 아니면 기억화(memorization)와 임의의 테이블 조회(stochastic table look-up)에 의해 완전히 구동되는지에 대한 논의를 진행합니다. 이 논문은 LLM의 작동 방식에 대한 새로운 이해를 제공하고, 여러 이론 간의 불일치를 해소하기 위한 기반을 마련합니다.

- **Technical Details**: 저자는 LLM 행동이 부분적으로는 표현 기반 정보 처리(representation-based information processing)에 의해 영향을 받는다고 주장하며, 이에 대한 여러 연구 기법을 설명하고 방어합니다. 이 기법은 LLM이 구현하는 알고리즘의 종류와 관련이 있으며, 이러한 이해는 LLM이 신념(beliefs), 의도(intentions), 개념(concepts), 지식(knowledge), 이해(understanding)를 가질 수 있는지에 대한 높은 수준의 질문에 중대한 함의를 갖습니다.

- **Performance Highlights**: 이 연구는 LLM의 행동을 설명하기 위한 새로운 다양한 기법들을 제공합니다. 이러한 기법은 LLM의 기능에 대한 더 깊은 통찰을 제공하며, 이후 언어 모델과 그 후속 모델들에 대한 이론적 탐구의 기반을 제공합니다. 결과적으로, 이 논문은 LLM의 연구에 중요한 기여를 하며, 향후 모델 개발에 방향성을 제시합니다.



### Diversity Optimization for Travelling Salesman Problem via Deep Reinforcement Learning (https://arxiv.org/abs/2501.00884)
- **What's New**: 본 논문에서는 Multi-Solution TSP (MSTSP)를 해결하기 위해 새로운 심층 강화 학습(Deep Reinforcement Learning) 기반의 신경망 솔버를 제안합니다. 제안된 방법은 인코더-디코더 구조의 정책(policy)을 특징으로 하며, 관련성과 다양성을 동시에 고려하여 고품질의 다양한 솔루션을 제공합니다. 특히 Relativization Filter (RF)와 Multi-Attentive Adaptive Active Search (MA3S)를 통해 솔루션의 품질을 개선하고 최적성과 다양성 간의 균형을 맞춥니다.

- **Technical Details**: 제안된 방법은 Neural Heuristic 형태로 MSTSP를 해결하기 위해 설계되었습니다. 인코더에서는 Cartesian 및 Polar 좌표를 기반으로 한 Relativization Filter (RF)를 사용하여 노드 분포 변화에 대한 저항성을 개선합니다. 디코더는 Attention 기반의 다중 디코더 아키텍처와 적응형 액티브 서치 메커니즘을 결합하여 최적성을 개선하고 다양성을 높인의 균형을 조정합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 벤치마크 사례에서 최근 neural baseline보다 우수한 성능을 보였으며, 기존의 전통적인 휴리스틱 방법들과 비교했을 때도 컴퓨팅 시간이 약 1.3배에서 15배까지 단축되었습니다. 또한 제안된 방법은 용량 제약 차량 경로 문제(CVRP)에도 적용할 수 있음을 보여주었습니다.



### DiffETM: Diffusion Process Enhanced Embedded Topic Mod (https://arxiv.org/abs/2501.00862)
Comments:
          5 pages, 2 figures, Accepted by ICASSP 2025

- **What's New**: 이 논문은 기존의 embedded topic model (ETM)에서 문서-주제 분포의 로지스틱 정규 분포 가정으로 인해 발생하는 성능 한계를 극복하기 위한 새로운 방법을 제안합니다. 제안된 방법은 확산 과정 (diffusion process)을 샘플링 과정에 통합하여 문서-주제 분포를 모델링하고 최적화 과정을 쉽게 유지할 수 있게 합니다. 우리가 제안한 모델은 두 개의 주요 데이터셋에서 주제 모델링 성능을 향상시키는 데 효과적임을 입증하였습니다.

- **Technical Details**: 이 모델에서는 문서 표현에서 직접 샘플링을 수행하여 고유한 문서-주제 분포를 생성합니다. 기존의 ETM과 비교할 때, 이러한 시도는 문서 정보가 포함된 숨겨진 표현을 통합하여 더 나은 모델링을 가능하게 합니다. 모델은 세 가지 메트릭인 주제 일관성 (topic coherence), 주제 다양성 (topic diversity), 그리고 혼잡도(perplexity) 기준에서 성능 향상을 달성합니다.

- **Performance Highlights**: 제안된 모델은 20Newsgroup 및 New York Times 데이터셋에서 기본 및 최첨단 ETM과 비교할 때 유의미한 성과를 나타냈습니다. 이 연구는 확산 과정을 ETM에 통합하여 문서-주제 분포의 표현 능력을 향상시키는 첫 번째 시도로, 새로운 접근법이 기여할 수 있는 잠재력을 보여줍니다. 결과적으로, 새로운 기술이 주제 모델링에서의 성능을 향상시키는 데 기여하고 있음을 확인했습니다.



### What is a Social Media Bot? A Global Comparison of Bot and Human Characteristics (https://arxiv.org/abs/2501.00855)
- **What's New**: 이 논문은 소셜 미디어 상의 커뮤니케이션에서 사람과 봇의 차이를 대규모 분석을 통해 다룹니다. 이 연구에서는 봇의 정의를 새롭게 제시하고, 봇과 인간 간의 의사소통 구조와 언어적 특징 차이를 분석합니다. 사용자, 콘텐츠, 관계라는 세 가지 요소를 중심으로 봇의 본질을 설명합니다.

- **Technical Details**: 논문에서는 2억 명 이상의 사용자 데이터를 사용하여 봇과 인간의 사용 패턴, 언어적 신호, 정체성 용어, 사회적 상호작용을 분석합니다. 조사된 특징들을 기반으로 봇의 정의는 자동화된 계정으로, 인위적이고 비본질적인 특성을 지닌다고 설명됩니다. 이 연구는 또한 머신러닝과 그래프 기반 방법을 사용한 봇 탐지 알고리즘에 대해 논의합니다.

- **Performance Highlights**: 논문에서는 또한 봇이 사회적, 정치적 이념을 확산시키는 데 어떠한 역할을 하는지를 서술합니다. 실제로 봇은 악의적인 담론에 영향을 미치고, 이러한 활동이 오프라인 세계에까지 스며들 수 있음을 보여줍니다. 헬스케어, 위기 관리 등 여러 긍정적인 용도로의 봇 활용 가능성도 논의됩니다.



### Distilled Lifelong Self-Adaptation for Configurable Systems (https://arxiv.org/abs/2501.00840)
Comments:
          Accepted by the 2025 International Conference on Software Engineering (ICSE 2025)

- **What's New**: 본 논문에서는 DLiSA라는 프레임워크를 제안하여 실행 중에 구성이 가능한 시스템의 자동 적응(self-adaptation)을 다룹니다. DLiSA는 '평생 계획(lifelong planning)'과 '정제된 지식 시딩(distilled knowledge seeding)'의 두 가지 주요 속성을 가지고 있어, 지속적으로 накоп된 지식을 활용하여 신속한 적응을 가능하게 합니다. 또한, 유용한 과거 구성을 동적으로 선택하여 부정확한 정보가 혼란을 초래하지 않도록 합니다.

- **Technical Details**: DLiSA는 MAPE-K 루프를 기반으로 하여 구성이 가능한 시스템이 동작하는 동안 적절한 구성을 계획할 수 있도록 지원합니다. 이 프레임워크는 진화 알고리즘을 활용하여 계획을 수행하며, 과거의 최적화된 구성에서 가장 유용한 구성만을 선택함으로써 적응 전략을 보다 효율적으로 만듭니다. 이 과정에서 시스템은 재시작 없이 과거의 경험을 활용하여 최적의 구성을 찾도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 DLiSA는 기존의 최첨단 접근 방식에 비해 성능이 최대 229% 향상되었고, 자원 가속화도 최대 2.22배 이루어졌습니다. 다양한 성과 목표와 복잡성을 가진 9개의 실제 시스템에서 진행된 93개의 실험 사례를 통해 이러한 결과를 확인했습니다. DLiSA는 성공적으로 높은 효율성과 효용성을 모두 갖춘 성능을 입증하고 있습니다.



### LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions (https://arxiv.org/abs/2501.00830)
Comments:
          42 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 복잡한 행위 추론 작업에서 한계를 겪고 있다는 문제점을 지적합니다. 이에 대응하기 위해, 저자들은 LLM의 자연어 이해 능력과 상징적 추론 능력을 갖춘 행동 언어(action languages)의 강점을 결합한 새로운 방법인 'LLM+AL'을 제안합니다. 이 접근법은 LLM의 의미적 구문 분석(semantic parsing) 및 상식 지식 생성(common sense knowledge generation) 능력을 활용하여 보다 정교한 자동화된 추론을 가능하게 합니다.

- **Technical Details**: LLM+AL 접근법은 주어진 자연어의 추론 문제에 대해 프로그램 서명을 생성하고, 해당 지식을 바탕으로 설정된 규칙을 행동 언어의 구문과 의미에 맞게 변환합니다. 이 과정은 ℬ⁢𝒞+한계로부터ℬ𝒞{
cal{BC}}의 규칙을 생성하여 자동화된 추론기(automated reasoner)와의 통합된 파이프라인을 통해 이루어집니다. 또한, 이 방법은 여러 단계를 통해 LLM을 효과적으로 활용하고, 최종적으로는 수동적인 수정이 최소화된 정확한 규칙을 생성합니다.

- **Performance Highlights**: 연구 결과 LLM+AL은 복잡한 상황에서도 상대적으로 적은 수의 수동 수정을 통해 정확한 답변을 도출하며, 단독 LLM보다 우수한 성능을 보였습니다. 특히, 여러 차례의 인간 피드백에도 불구하고 기존 LLM들은 정답을 생성하는 데 어려움을 겪는 반면, LLM+AL은 적은 수정으로 올바른 해답을 제공합니다. 이는 LLM과 행동 언어의 통합 방식이 더 견고하고 적응 가능한 AI 시스템을 구축하는 데 기여할 수 있음을 보여줍니다.



### An LLM-Empowered Adaptive Evolutionary Algorithm For Multi-Component Deep Learning Systems (https://arxiv.org/abs/2501.00829)
Comments:
          9

- **What's New**: 이 논문에서는 복합 멀티 컴포넌트 딥러닝 시스템(MCDL 시스템)의 안전 위반 탐지를 위해 $u 	ext{MOEA}$라는 최초의 LLM 기반 적응적 진화 검색 알고리즘을 제안합니다. 이 알고리즘은 대형 언어 모델(LLM)의 문맥 이해 능력을 활용해 초기 인구를 최적화 문제에 맞게 생성하고, 진화 효율성과 다양성을 유지합니다. 또한, 진화 과정 중 로컬 옵티마에서 벗어나기 위해 LLM에 진화 경험을 통합하여 LLM의 정량적 사고 능력을 이용해 차별적인 씨앗을 생성합니다.

- **Technical Details**: 복합 멀티 컴포넌트 딥러닝 시스템은 여러 상호작용 모듈로 구성되어 있으며, 각 모듈은 독립적인 매개변수와 행동을 가지고 있어 예측하기 어려운 출현 특성을 나타냅니다. 기존의 MOEA는 안전 위반 탐지에서 특정 두 가지 도전에 직면하였으며, 논문에서는 이를 $u 	ext{MOEA}$를 통해 해결하고자 합니다. 이 알고리즘은 초기 인구 생성 시 무작위 대신 명시적으로 목적을 고려하여 생성하고, 적응적 선택 및 변이 기법을 통해 진화 과정을 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과, $u 	ext{MOEA}$는 MCDL 시스템에서 안전 위반을 찾는 데 있어 기존의 최첨단 방법들과 비교할 때 효율성과 다양성을 크게 향상시켰습니다. 이 방법론은 고급 다목적 유전 알고리즘에 기반한 NSGA-II와 비교하여 더 많은 다양하고 뛰어난 최적 솔루션을 발견해냈습니다. 이러한 성과는 MCDL 시스템의 안전성을 유지하고 위험을 줄이는 데 기여할 것으로 기대됩니다.



### Embedding Style Beyond Topics: Analyzing Dispersion Effects Across Different Language Models (https://arxiv.org/abs/2501.00828)
Comments:
          To appear in the Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Abu Dhabi

- **What's New**: 본 논문은 다양한 최신 언어 모델에서 쓰기 스타일이 임베딩 벡터(embedding vectors)의 분산에 미치는 영향을 분석합니다. 초기 트랜스포머 모델은 주로 주제 모델링(topic modeling)과 일치했으나, 이 연구는 쓰기 스타일이 임베딩 공간을 형성하는 역할을 조사합니다. 저자들은 프랑스어와 영어의 문학 코퍼스를 사용하여 언어 모델의 민감성을 비교하고, 스타일 정보가 언어 모델 처리에 미치는 영향을 밝히고자 합니다.

- **Technical Details**: 연구는 두 가지 문학 작품인 레이몽 크뇌의 'Exercices de Style'와 펠릭스 페넥의 'Nouvelles en trois lignes'를 원재료로 사용하여 작성되었습니다. 이러한 데이터 세트는 주제와 스타일 차원을 체계적으로 교환하는 실험적 연구를 통해 생성되었습니다. 결과적으로, 스텝 성과를 차별화하여 각 언어 모델에서 스타일과 주제가 임베딩의 공간 분산에 미치는 상대적인 영향을 평가할 수 있었습니다.

- **Performance Highlights**: 실험 결과, 쓰기 스타일이 임베딩 분산에 미치는 영향이 주제보다 더 주목할 만한 것으로 나타났습니다. 이를 통해 현대 대규모 언어 모델들이 쓰기 스타일에 대한 세밀한 정보를 어떻게 처리하는지를 더 잘 이해할 수 있었습니다. 또한, 향후 연구 방향으로는 임베딩 벡터에서 스타일 표현을 정확히 포착하기 위한 방법론을 제안하고 있습니다.



### LLM-Powered Multi-Agent System for Automated Crypto Portfolio Managemen (https://arxiv.org/abs/2501.00826)
- **What's New**: 이 논문은 암호화폐 투자에 있어 복잡한 문제를 해결할 수 있는 설명 가능한 다중 모드, 다중 에이전트 프레임워크를 제안합니다. 이 프레임워크는 데이터 분석, 문헌 통합 및 투자 의사결정을 위한 전문 에이전트를 활용하여 상위 30개 암호화폐의 투자 결정을 지원합니다. 각 에이전트는 특정 세부 작업에 대해 조정 및 교육을 받으며, 팀 내외의 공동작업 메커니즘을 통해 예측 정확도를 향상시킵니다.

- **Technical Details**: 프레임워크는 전문가 교육 모듈과 다중 에이전트 투자 모듈의 두 가지 주요 모듈로 구성됩니다. 전문가 교육 모듈은 다중 모드의 역사적 데이터와 금융 문헌을 통합하여 에이전트를 최적화 하며, 다중 에이전트 투자 모듈은 실시간 데이터를 활용하여 시장 팀과 암호화 팀 간에 협력합니다. 특히, 팀 간 및 팀 내 협업 메커니즘이 도입되어 최종 투자 결정을 위해 에이전트들의 신뢰도를 함께 반영하여 예측을 수립합니다.

- **Performance Highlights**: 2023년 6월부터 2024년 9월까지의 데이터를 기반으로 한 실증 평가에서, 이 프레임워크는 단일 에이전트 모델과 시장 벤치마크를 초과하는 분류 정확도와 자산 가격 성능을 보여주었습니다. 포트폴리오 성과에서도 눈에 띄는 개선을 나타내며, 아울러 암호화폐 투자에 있어 오류를 줄이기 위한 커뮤니케이션 방식이 효과적으로 작용함을 입증합니다.



### Decoupling Knowledge and Reasoning in Transformers: A Modular Architecture with Generalized Cross-Attention (https://arxiv.org/abs/2501.00823)
- **What's New**: 이 논문에서는 지식과 추론을 명확히 분리하는 새로운 모듈형 Transformer 아키텍처를 제안합니다. 이 구조는 일반화된 크로스 어텐션 메커니즘을 통해 공유 지식 기반에 효과적으로 접근할 수 있도록 설계되었습니다. 기존의 Feed-Forward Network (FFN)를 새로운 메커니즘으로 대체하여 지식 검색의 효율성을 향상시키고자 합니다.

- **Technical Details**: 제안된 모듈형 Transformer는 각 층에서 E라는 공유 지식 기반을 도입하고, 이 기반에 접근하기 위한 전용 크로스 어텐션 메커니즘을 활용합니다. 기존의 FFN을 대체하여, 지식 검색과 추론을 명확히 분리함으로써 해석 가능성, 적응성 및 확장성을 크게 향상시킬 수 있습니다. FFN이 사실상 수행하는 암묵적 지식 검색을 명시적으로 표현하는 것이 핵심적인 목표입니다.

- **Performance Highlights**: 이론적 분석을 통해 기준 Transformer의 FFN이 제안된 일반화된 크로스 어텐션의 특수한 사례라는 것을 수학적으로 증명했습니다. 이 결과는 FFN의 암묵적 지식 검색 역할을 재확인하며, 외부 지식 기반과의 통합을 위한 기초를 다집니다. 이 프레임워크는 향후 연구에서 향상된 해석 가능성 및 적응성에 대한 탐구의 기반이 될 것입니다.



### Reasoning-Oriented and Analogy-Based Methods for Locating and Editing in Zero-Shot Event-Relational Reasoning (https://arxiv.org/abs/2501.00803)
- **What's New**: 본 논문에서는 이벤트 관계 추론을 위한 새로운 접근법인 Reasoning-Oriented Locating and Editing (ROLE)와 Analogy-Based Locating and Editing (ABLE)을 제안합니다. ROLE은 언어 모델의 핵심 모듈을 찾고 편집하여 해석 가능성을 높이고 리소스를 효율적으로 최적화합니다. ABLE은 다양한 작업 간의 유사성을 활용하여 제로샷 추론 능력을 향상시키는 방법으로 인식됩니다.

- **Technical Details**: ROLE의 핵심 모듈 식별을 위해 평균 간접 효과를 계산하고, 이를 기반으로 키 모듈의 변화 정도를 최적화하여 사고 성능을 향상시킵니다. ABLE은 작업 간 유사성과 차이를 학습하여 지식을 효율적으로 이전하고, 제로샷 추론에서 뛰어난 결과를 달성합니다. 두 방법 모두 Flan-t5-large를 기반 모델로 활용하여 운영됩니다.

- **Performance Highlights**: ROLE은 예상보다 낮은 계산 비용으로 해석성과 추론 성능을 개선했으며, ABLE은 다양한 데이터 세트에서 제로샷 이벤트 관계 추론의 SOTA 성과를 달성했습니다. 이러한 방법들은 기존 접근법보다 더 뛰어난 성능을 제공하며, 특히 대규모 언어 모델의 해석 가능성을 높여줍니다.



### Make Shuffling Great Again: A Side-Channel Resistant Fisher-Yates Algorithm for Protecting Neural Networks (https://arxiv.org/abs/2501.00798)
- **What's New**: 이번 논문에서는 임베디드 장치에서의 신경망(Neural Network) 모델을 위한 사이드 채널 공격(SCA)에 대한 방어 메커니즘을 개발했습니다. 기존의 Fisher-Yates 알고리즘을 사용한 셔플링(shuffling) 방법의 취약점을 개선하여, 새로운 보안 방안을 제안하고 있습니다. 이 방법은 곱셈 최적화와 모듈러 연산에서의 마스킹(masking) 기술을 결합하여 사이드 채널 누출을 방지합니다.

- **Technical Details**: 논문에서 제안하는 방식은 Fisher-Yates 알고리즘의 변형으로, 사이드 채널 공격으로부터 새로운 셔플링을 구현하는 것입니다. ARM Cortex-M4 기반의 임베디드 신경망 모델에 대한 공분산 전력 분석(correlation power analysis) 공격을 실험적으로 평가하여, 새로운 방법의 효과를 입증했습니다. 이 과정을 통해 원래 제안된 방법에 비해 메모리 사용량은 두 배 증가했으나, 실행 시간의 오버헤드는 4%에서 0.49% 범위로 다양하게 나타났습니다.

- **Performance Highlights**: 제안된 셔플링 방법은 공격자가 모델 파라미터를 복구하는 데 큰 어려움을 초래했습니다. 실험 결과, 제안된 방법은 임베디드 신경망 모델을 사용한 공격에 대해 효과적으로 방어할 수 있는 것으로 나타났습니다. 특히, 작은 네트워크에서 공격에 대한 최악의 시나리오가 제한되므로, 보안성이 확보되었습니다.



### LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and Variational Autoencoders for Scalable Intrusion Detection in Cybersecurity (https://arxiv.org/abs/2501.00790)
- **What's New**: 본 연구는 리소스 제한 환경에서의 네트워크 보안을 재정의하는 경량화되고 설명 가능한 네트워크 보안 프레임워크인 LENS-XAI를 소개합니다. LENS-XAI는 지식 증류(knowledge distillation)와 변분 오토인코더(variational autoencoder) 모델을 결합하여 높은 탐지 정확도와 의사 결정 과정의 투명성을 달성합니다. 이 프레임워크는 10%의 훈련 데이터로 계산 효율성을 최적화하고 성능 저하 없이도 복잡한 공격 시나리오에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LENS-XAI는 지식 증류를 활용하여 복잡한 데이터 분포 모델링을 개선하고, 다양한 어트리뷰션 기반의 XAI 기법을 통해 신뢰성을 높입니다. 본 프레임워크는 Edge-IIoTset, UKM-IDS20, CTU-13 및 NSL-KDD와 같은 네 가지 벤치마크 데이터셋에서 실험적으로 평가되어 각각 95.34%, 99.92%, 98.42%, 99.34%의 탐지 정확도를 기록했습니다. 경량화된 아키텍처는 IoT 및 엣지 컴퓨팅 시나리오에 적합하며, 신뢰도 높은 탐지 성능을 유지할 수 있습니다.

- **Performance Highlights**: LENS-XAI는 오류 판별(false positive) 감소에도 탁월하여 기존의 최신 방법들보다 우수한 성능을 보여줍니다. 이 프레임워크의 중요한 강점은 자원이 제한된 환경에서도 효과적으로 작동하도록 설계된 경량 디자인과 다각적인 IIoT 및 사이버 보안 맥락에서 확장 가능한 특성입니다. 설명 가능성 모듈은 동적인 환경과 민감한 응용 프로그램에서 신뢰와 투명성을 높이는 데 기여합니다.



### REM: A Scalable Reinforced Multi-Expert Framework for Multiplex Influence Maximization (https://arxiv.org/abs/2501.00779)
- **What's New**: 이번 연구에서는 Reinforced Expert Maximization (REM)이라고 불리는 새로운 프레임워크를 제안합니다. REM은 Propagation Mixture of Experts 기술을 활용하여 대규모 멀티플렉스 네트워크의 동적 전파를 효과적으로 인코딩하고, 강화 학습 관점에서 seed set을 생성하고 개선하는 방법을 학습합니다. 실험을 통해 REM이 기존의 최첨단 방법들보다 영향력 확산, 확장성 및 추론 시간에서 우수하다는 것을 보여줍니다.

- **Technical Details**: REM은 Seed2Vec라는 VAE 기반 모델을 사용하여 노이즈가 포함된 입력 공간을 더 깨끗하고 연속적인 잠재 공간으로 변환합니다. 이 모델은 강화 학습(policy)로 설정되어, 멀티플렉스 네트워크에서 중요한 전파를 가지는 새로운 seed set을 생성하고자 합니다. 또한, REM은 여러 Graph Neural Network (GNN) 모델을 전문가로 활용하여 복잡한 전파 패턴을 포착하는 Propagation Mixture of Experts (PMoE) 방법을 통해 전파 추정 정확성을 높입니다.

- **Performance Highlights**: REM은 여러 실질적인 데이터셋을 통해 기존의 방법들보다 더 나은 성능을 입증했습니다. 특히 REM은 영향력 확산 측면에서 가장 효율적이며, 확장성과 추론 시간에서도 우수한 결과를 보였습니다. 이러한 성과는 멀티플렉스 네트워크의 복잡한 관계를 효율적으로 처리하는 REM의 능력에서 기인합니다.



### Revisiting Graph Neural Networks on Graph-level Tasks: Comprehensive Experiments, Analysis, and Improvements (https://arxiv.org/abs/2501.00773)
- **What's New**: 이번 연구에서는 그래프 수준의 GNN(그래프 신경망) 평가를 위한 통합 평가 프레임워크를 제안합니다. 이는 다양한 데이터셋, 여러 그래프 작업 및 도전적인 시나리오를 아우르는 표준화된 환경을 제공합니다. 또한, k-경로(rooted subgraph) 접근 방식을 사용하는 새로운 GNN 모델이 제시되어 향상된 표현력과 일반화 능력을 보여줍니다. 이러한 접근법은 그래프 분류 및 회귀 작업에 대한 성과를 더 향상시킬 것입니다.

- **Technical Details**: 그래프는 노드와 엣지로 구성되어 있으며, 이는 복잡한 상호작용을 모델링하는 데 필수적인 데이터 구조입니다. GNN은 이웃의 정보 집합을 통해 노드 표현을 학습하고, 이를 기반으로 그래프 수준 작업을 수행합니다. 현재 GNN은 노드 기반, 계층적 풀링(HP) 기반, 서브그래프(subgraph) 기반, 그래프 학습(GL) 기반, 자가 지도(Self-Supervised Learning, SSL) 기반으로 분류됩니다. 연구에서 제안한 새로운 k-경로 서브그래프 GNN은 k 길이 경로와 관련된 서브그래프를 샘플링하여 그래프 표현을 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 27개의 그래프 데이터셋에 대해 14개의 효율적인 기준 모델보다 뛰어난 성능을 달성했습니다. 이 연구는 기존 GNN의 한계를 극복하고, 다양한 도메인에서의 적용 가능성을 크게 향상시켰습니다. 또한, 제안된 그래프 대조 학습(alternate graph contrastive learning) 접근 방식은 그래프의 중요하지 않은 엣지를 제거하여 일반화 성능을 상당히 향상시킵니다. 본 연구는 그래프 수준 작업을 위한 강력하고 일반화 가능한 모델로 자리매김할 것입니다.



### Enhancing Transformers for Generalizable First-Order Logical Entailmen (https://arxiv.org/abs/2501.00759)
Comments:
          17 pages

- **What's New**: 이번 연구에서는 transformers의 일반화 가능한 1차 논리 추론 능력을 조사하고, 이를 개선할 수 있는 방법을 모색합니다. 기존의 연구들과의 차별점은, 모델 안에서 파라미터화된 지식을 바탕으로 추론을 수행한다는 것입니다. 또한, 기존의 데이터 분포와의 연결성을 더 정교하게 설정하여, 새로운 유형의 쿼리와 지식이론의 관점을 통해 1차 논리 추론 작업의 성능을 분석합니다.

- **Technical Details**: 이 논문은 Knowledge Graphs(KG)를 사용하여 1차 논리 함의(first-order logical entailment)를 연구합니다. 연구는 복잡한 쿼리 문제를 해결하기 위한 필수적인 과정으로서의 1차 논리 함의를 제시하는 데 중점을 두고 있습니다. 논리 쿼리를 식별하는 과정에서 추론 과정이 어떻게 이루어지는지를 관찰하고, 다양한 쿼리 유형에 대한 논리적 추론의 성능을 평가합니다.

- **Performance Highlights**: 우리는 transformers가 논리 함의 문제에서 현존하는 방법들보다도 뛰어난 성능을 발휘함을 발견했습니다. 특히, 상대 위치 인코딩(relative positional encoding, RPE)이 전통적인 절대 위치 인코딩(absolute positional encoding, APE)보다 우수한 성능을 보였습니다. TEGA(Transformer Encoder with Guided Attention)라는 새로운 아키텍처를 제안하여, 이러한 성능을 더욱 향상시킬 수 있음을 입증했습니다.



### An AI-powered Bayesian generative modeling approach for causal inference in observational studies (https://arxiv.org/abs/2501.00755)
- **What's New**: CausalBGM은 고차원 공변량이 있는 관찰 연구에서 인과 추론을 분해하는 새로운 AI 기반의 Bayesian Generative Modeling 접근 방식을 소개합니다. CausalBGM의 주요 혁신은 치료 효과 개별 추정(Individual Treatment Effect, ITE)을 통해 각 개인에 대한 효과를 정확히 추정할 수 있는 능력입니다. 이는 저차원의 잠재 변수를 사용하는 분포를 학습하여 인과 관계를 모델링합니다. 이 방법은 기존의 절차보다 보편적이고 신뢰할 수 있는 인과 추정치를 제공합니다.

- **Technical Details**: CausalBGM은 완전한 Bayesian 절차를 채택하여 latent features를 추론합니다. 기존의 encoder-decoder 구조를 제거하여, 명확한 Directed Acyclic Graph(DAG) 구조를 확보하고, 이는 인과 관계를 더 정확하게 나타내도록 합니다. 또한 CausalBGM은 변량 추론(Variational Inference)을 통해 모델 파라미터의 posterior 분포를 업데이트하며, 관측 변수의 평균과 공분산 기능을 동시에 모델링하여 데이터 변동성을 포괄적으로 설명합니다.

- **Performance Highlights**: 광범위한 실험 결과 CausalBGM은 고차원 공변량과 대규모 데이터세트에서 최첨단 방법들보다 성능이 뛰어난 것으로 나타났습니다. 이 모델은 통계적 유의성을 보장하며, 준거 기반 환산(interval) 예측이 잘 조정되어 신뢰할 수 있는 결과를 제공합니다. CausalBGM은 새로운 방법론을 통해 복잡한 인과 추론 문제를 해결하는 강력하고 다재다능한 솔루션으로 자리매김하고 있습니다.



### Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines (https://arxiv.org/abs/2501.00745)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 검색 엔진의 정보 검색 환경의 변화를 다룬다. 특히, LLM이 검색 엔진에 통합됨에 따라 발생하는 새로운 취약성과 공격, 특히 랭킹 조작 공격에 대한 연구를 진행한다. 이를 통해 공격자들이 웹페이지 콘텐츠를 조작하여 순위를 조작하는 방법과 그로 인해 발생하는 비즈니스 상의 불공정한 이점에 대해 논의한다.

- **Technical Details**: 이 연구에서는 랭킹 조작 공격의 동학을 무한 반복 죄수의 딜레마(Infinitely Repeated Prisoners' Dilemma)라는 게임 이론의 관점에서 분석한다. 공격자들은 전략적으로 협력할지 공격할지를 결정하며, 이 과정에서 공격 비용, 공격 성공률, 할인율 등 여러 요소가 플레이어 행동에 미치는 영향을 살펴본다. 협력이 지속될 수 있는 조건을 식별하고, 적응형 보안 전략과 생태계 설계의 중요성을 강조한다.

- **Performance Highlights**: 연구 결과, 협력이 지속될 가능성이 높아지는 경계 조건과 각 요인이 협력 장기 지속성에 미치는 영향을 보여준다. 흥미롭게도, 공격 성공률이 중간 수준일 때 공격의 이익과 위험이 최적화되어 협력을 저해할 수 있음을 발견했다. 또한, 방어 조치를 통해 공격 성공률을 제한하려는 시도가 오히려 공격 유인을 증가시킬 수 있다는 점도 강조한다.



### AttriReBoost: A Gradient-Free Propagation Optimization Method for Cold Start Mitigation in Attribute Missing Graphs (https://arxiv.org/abs/2501.00743)
- **What's New**: 이 논문에서 제안한 AttriReBoost (ARB) 방법은 특히 속성이 결여된 그래프에서의 콜드 스타트 문제를 해결하기 위해 혁신적인 기술을 도입합니다. ARB는 초기 경계 조건을 재정의하고 가상 에지를 통합하여 구성 요소의 전파를 개선합니다. 이 과정은 노드의 연결성을 증가시키고 효율적인 수렴을 보장함으로써, 속성 재구성을 더욱 용이하게 만듭니다.

- **Technical Details**: ARB는 그래프 기반 학습의 효율성을 증대시키기 위해 경량의 경량 메시지 전파 프레임워크를 활용합니다. 이 방법은 백 프로파게이션(backpropagation)과 기울기 학습의 계산 부담을 없애, 속성 재구성을 더 낮은 계산 비용으로 달성하도록 도와줍니다. ARB의 수렴은 Banach 고정점 정리를 이용하여 엄격하게 증명되어 있으며, 두 개의 추가 하이퍼파라미터만을 도입합니다.

- **Performance Highlights**: 실험 결과, ARB는 최신 기법보다 평균 5.11%의 정확도 향상을 이루었습니다. 또한, ARB는 2.49백만 노드를 가진 대규모 그래프를 단일 GPU에서 단 16초 만에 처리할 수 있는 놀라운 계산 효율성을 보입니다. 이는 실세계 응용 프로그램에서 계산 자원과 시간이 중요할 때 큰 장점으로 작용합니다.



### Towards End-to-End Neuromorphic Voxel-based 3D Object Reconstruction Without Physical Priors (https://arxiv.org/abs/2501.00741)
Comments:
          6 pages, 15 figures, 5 tables, submitted to ICME 2025

- **What's New**: 본 연구에서는 기존의 물리적 사전(physical priors)을 추정할 필요 없는 밀집 복셀(dense voxel) 3D 재구성을 위한 엔드 투 엔드(end-to-end) 방법을 제안합니다. 여기서 새로운 이벤트 표현(event representation)인 Sobel Event Frame을 통해 모서리 특징을 강화하고, 3D 특징 학습을 효과적으로 진행할 수 있게 하였습니다. 또한, 최적 이진화 임계값 선택 원칙(Optimal Binarization Threshold Selection Principle)을 제안하여 향후 연구에 대한 지침으로 활용할 수 있도록 하였습니다.

- **Technical Details**: 제안된 방법은 단일 신경 모양 카메라를 직접 이용하여 3D 재구성을 수행하며, 복잡한 이벤트-3D 파이프라인을 배제합니다. 이벤트 표현에는 이벤트의 좌표, 타임스탬프(timestamp), 그리고 밝기 변화의 극성(polity)을 포함해, 수집된 데이터의 전처리를 수행합니다. 이벤트 데이터는 타임스탬프 또는 이벤트 수를 기준으로 나누어 엑스 형태의 프레임으로 변환되며, 이를 통해 각 픽셀은 해당 시간 창에서 이벤트 발생 여부를 나타냅니다.

- **Performance Highlights**: 제안된 방법은 기존의 기준 방법(<baseline method>)에 비해 54.6%의 재구성 정확도 향상을 기록하였습니다. 이는 고객의 기대를 넘어선 성능을 나타내며, 새로운 방법론에 대한 검증의 일환으로 중요한 성과로 평가됩니다. 또한, 본 연구에서 제안된 각 모델은 실시간 솔루션(real-time solution)으로 발전할 가능성이 존재합니다.



### eRevise+RF: A Writing Evaluation System for Assessing Student Essay Revisions and Providing Formative Feedback (https://arxiv.org/abs/2501.00715)
- **What's New**: 이번 논문에서는 학생들의 작문 수정을 지원하는 향상된 Automated Writing Evaluation (AWE) 시스템인 eRevise+RF를 도입합니다. 이 시스템은 학생들이 제공된 피드백을 바탕으로 에세이를 수정할 수 있도록 돕고, 수정의 효과를 평가합니다. 연구 결과, eRevise+RF는 학생들의 증거 사용에 대한 평가 및 반영된 수정 사항을 추출하는 데 효과적임을 확인하였습니다.

- **Technical Details**: eRevise+RF 시스템은 Natural Language Processing (NLP) 기술을 활용하여 학생의 작문을 평가하고 피드백을 제공합니다. 이 시스템은 Automated Essay Scoring (AES)과 Revision Feedback (RF) 두 가지 기능으로 구성되어 있으며, 이전의 연구에서 사용된 알고리즘을 기반으로 개발되었습니다. 특히, 시스템은 원본 및 수정된 초안을 비교하여 각 수정 사항이 에세이 개선에 얼마나 기여했는지를 평가합니다.

- **Performance Highlights**: eRevise+RF는 6명의 교사와 406명의 학생을 대상으로 한 연구에서 긍정적인 결과를 보였습니다. 시스템은 학생들이 에세이를 수정할 때 필요한 피드백을 제공하여, 논증적 작문 기술을 향상시키는 데 도움을 줍니다. 이 실험을 통해 학생들이 에세이를 개선하는 데 있어서 중요한 지원을 받을 수 있음을 보여주었습니다.



### Everywhere Attack: Attacking Locally and Globally to Boost Targeted Transferability (https://arxiv.org/abs/2501.00707)
Comments:
          11 pages, 6 figures, 8 tables, accepted by 2025AAAI

- **What's New**: 본 논문에서는 주어진 이미지에 대한 targeted transferability를 강화하기 위한 새로운 방안인 'everywhere scheme'을 제안합니다. 이전의 연구에서는 이미지의 특정 높은 신뢰도 목표를 최적화하는데 초점을 맞췄다면, 본 방법은 피해 이미지의 모든 지역에서 다수의 목표를 동시에 공격하는 방식을 채택합니다. 이를 통해 다양한 모델에 대한 attention(어텐션) 불일치로 인한 transfer failure(전이 실패)를 감소시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 피해 이미지를 겹치지 않는 블록으로 분할하고 각 블록에 대해 targeted attack을 수행합니다. 이는 각 지역의 attention areas(어텐션 영역)에서 하나의 목표 물체가 피해 모델의 어텐션 지역에 포함될 가능성을 높입니다. 기존의 높은 신뢰도 목표를 설정하는 방식과 달리, 본 방법은 다양한 목표 객체를 포함시켜 transferability(전이 가능성)를 증대시키는 것을 목표로 합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 방법이 기존의 state-of-the-art targeted attack들을 개선함을 보여줍니다. 예를 들어, 널리 사용되는 Logit 공격의 전이 가능성이 28.8%에서 300%로 향상되었습니다. 또한 Google Cloud Vision과 같은 실제 플랫폼에서 출처 공격 예제가 더 나은 성과를 거두는 것을 검증하였습니다.



### Adjoint sharding for very long context training of state space models (https://arxiv.org/abs/2501.00692)
- **What's New**: 본 논문에서는 매우 긴 컨텍스트를 가진 대형 언어 모델(LLMs)을 효율적으로 훈련할 수 있는 새로운 기술인 adjoint sharding을 제안합니다. 기존 방법들은 짧은 컨텍스트에서 훈련을 수행하고 긴 컨텍스트에서는 추론 기술을 이용했지만, 이 접근법은 GPU 메모리 용량과 긴 훈련 시간에 제약을 받습니다. Adjoint sharding은 메모리 요구 사항을 극적으로 줄이면서 긴 컨텍스트에서의 훈련을 가능하게 만들어, 사실 추출 및 요약과 같은 다양한 실제 작업에 대응합니다.

- **Technical Details**: Adjoint sharding은 그래디언트 계산을 분해하여 메모리 사용량을 크게 줄이는 방법입니다. 이는 반복적인 모델의 경우 adjoint 방법을 기반으로 하며, 그라디언트를 효율적으로 계산하기 위한 독립적인 벡터-야코비안 곱(VJP) 계산을 포함합니다. 또한, 성능을 유지하면서 알고리즘 속도를 높이기 위한 트렁케이티드(Truncated) adjoint sharding도 제안했습니다.

- **Performance Highlights**: 실험 결과에 따르면, adjoint sharding 알고리즘은 1M 컨텍스트 길이 훈련 시 1.27B 파라미터의 대형 언어 모델에서 메모리 사용량을 최대 3배 감소시켰습니다. 이로 인해 훈련 인프라에서 35K 토큰에서 100K 토큰 이상의 최대 컨텍스트 길이를 증가시킬 수 있습니다. 이러한 결과는 adjoint sharding의 분산 및 병렬 버전으로 훈련 속도를 더욱 개선할 수 있음을 보여줍니다.



### Leaf diseases detection using deep learning methods (https://arxiv.org/abs/2501.00669)
Comments:
          252 pages , 42 images

- **What's New**: 이번 연구는 식물 잎 질병 식별 및 감지를 위한 새로운 딥러닝(deep learning) 접근법을 개발하는 데 중점을 두고 있습니다. 우리는 현재의 잎 질병 감지 방법이 직면한 도전 과제를 논의하고, 딥러닝이 이러한 문제를 극복하고 질병 탐지의 정확성을 향상시키는 데 어떻게 사용될 수 있는지를 살펴보았습니다.

- **Technical Details**: 이에 따라, 우리는 다양한 작물의 잎 질병을 탐지하기 위한 새로운 방법론을 제안하였으며, 하이퍼파라미터(hyperparameters)와 최적화 방법을 포함하는 효율적인 네트워크 아키텍처(architecture)를 소개하였습니다. 또한 여러 아키텍처의 효율성을 비교하고 평가하여 최적의 아키텍처 구성을 찾아 고속 질병 탐지가 가능한 효과적인 모델을 만들었습니다.

- **Performance Highlights**: 본 연구에서는 사전 학습(pre-trained) 모델에 대한 작업 외에도, CNN(Convolutional Neural Network) 기반의 새로운 모델을 제안하여 식물 잎 질병 식별 및 감지의 효율적인 방법을 제공합니다. 또한, 우리의 모델의 효율성을 평가하고, 일부 최신 사전 학습 아키텍처와 결과를 비교하였습니다.



### Titans: Learning to Memorize at Test Tim (https://arxiv.org/abs/2501.00663)
- **What's New**: 이 논문은 최신 Neural Long-Term Memory 모듈을 제안하며, 이는 과거의 문맥을 기억하고 최신 문맥을 처리하는 데 도움을 줍니다. 특히, 이 메모리 모듈은 빠른 병렬 처리와 신속한 추론을 가능하게 하며, Transformer와 최근의 선형 순환 모델에 비해 우수한 성능을 보여줍니다.

- **Technical Details**: Titan이라는 새로운 아키텍처 패밀리를 도입하고, 이를 통해 메모리를 효과적으로 통합하는 세 가지 변형을 발표합니다. 이 모델은 단기 메모리로서의 주의(attention)와 장기 메모리로서의 신경 메모리(neural memory) 사이의 균형을 탐구합니다. 이론적으로 메모리를 업데이트하는 구조와 메커니즘을 재구성함으로써 메모리 학습을 최적화하려합니다.

- **Performance Highlights**: 실험 결과, Titan 모델은 언어 모델링, 상식 추론, 유전체학(genomics), 시계열(time series) 작업에서 Transformer와 현대의 선형 순환 모델보다 더 효과적인 성능을 보였습니다. 특히, Titan은 작업의 맥락 범위를 2M 이상으로 확장할 수 있으며, 'needle-in-haystack' 작업에서 baseline에 비해 높은 정확도를 달성했습니다.



### Efficient Standardization of Clinical Notes using Large Language Models (https://arxiv.org/abs/2501.00644)
- **What's New**: 이 연구에서는 대형 언어 모델을 활용하여 1,618개의 임상 노트를 표준화하는 방법을 제시합니다. 표준화 과정에서는 평균적으로 4.9개의 문법 오류, 3.3개의 철자 오류, 비표준 용어를 표준 용어로 변환하고, 약어 및 두문자어를 각각 15.8개 확장했습니다. 이러한 방법은 임상 노트의 가독성과 일관성을 향상시키고, 상호 운용이 가능한 데이터 형식으로의 변환을 용이하게 합니다.

- **Technical Details**: 이 연구에서 GPT-4 API를 사용하여 임상 노트의 구조와 명확성을 개선하는 방법을 탐구했습니다. 임상 노트는 신경 면역 질환과 같은 진단을 포함한 비식별 데이터로 구성되었으며, 노트 표준화 과정에서는 약어 확장, 철자 및 문법 수정, 비표준 용어 교체 등을 수행했습니다. 각 노트는 JSON 형식으로 저장되어 통합 및 검색에 적합한 구조로 구성되었습니다.

- **Performance Highlights**: 전문가 리뷰 결과, 표준화 이후 중요한 데이터 손실이 발견되지 않았고, 이는 임상 노트의 질을 유지하면서도 가독성과 기능성을 향상시킬 수 있음을 보여줍니다. 이 연구는 임상 노트 표준화의 개념 증명을 제공하며, 향후 의사 결정 지원 및 연구에 기여할 수 있는 가능성을 제시합니다.



### Enabling New HDLs with Agents (https://arxiv.org/abs/2501.00642)
- **What's New**: 이번 논문은 HDL (Hardware Description Languages) 에 대한 LLM (Large Language Models)의 적용 가능성을 탐구하며, LLM이 훈련되지 않은 HDL에서의 성과를 높이기 위한 AI 에이전트인 HDLAgent를 소개합니다. HDLAgent는 기존 LLM의 제한된 지식을 보완하여 새로운 HDL을 학습하는 데에 효과적입니다. 이 연구는 LLM이 HDLs의 코드 생성을 개선하고, 문서화 작업을 최적화할 수 있는 방법을 제시합니다.

- **Technical Details**: HDLAgent는 기존 LLM을 Fine-tuning 없이 활용하여 작업의 효율성을 높입니다. 이 에이전트는 Chain-of-Thought (CoT)와 Retrieval Augmented Generation (RAG)와 같은 자가 반성(self-reflection) 기술들을 활용하여 입력 데이터를 정제하고 구문 및 의미상의 오류를 수정하는 과정을 거칩니다. 또한, HDL의 설명 요약 생성과 예시의 제공을 통해 새로운 HDL로의 지식 이전을 증진시킵니다.

- **Performance Highlights**: HDLAgent의 도입으로 Mix-8x7B 모델을 사용할 경우 Chisel 코드의 성공률이 3%에서 44%로 향상되었으며, 다른 LLM에서도 비슷한 개선을 보였습니다. HDLAgent는 Verilog의 경우에도 성공률을 13%에서 53%로 증가시키며, 전반적으로 모든 HDL에서의 코드 성공률을 90% 이상 향상하는 성과를 보였습니다. 이러한 결과는 LLM과 하드웨어 디자인의 효율적인 적용 가능성을 제시하며, 개발자의 생산성을 높이는 데 기여할 것으로 기대됩니다.



### A Study on Context Length and Efficient Transformers for Biomedical Image Analysis (https://arxiv.org/abs/2501.00619)
Comments:
          Published at ML4H 2024

- **What's New**: 이번 연구에서는 생물 의학 이미지 분석에서 문맥 길이(context length)의 영향을 조사하고, 최근 제안된 장문 모델(long-context models)의 성능을 평가합니다. 생물 의학 이미징 데이터셋을 선별하고 분석함으로써, 다양한 세그멘테이션(segmentation), 노이즈 제거(denoising), 분류(classification) 작업에서의 효율성을 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 비전 트랜스포머(Vision Transformer)와 스윈 트랜스포머(Swin Transformer)를 활용하여 패치 크기(patch size)와 주의 윈도우 크기(attention window size)를 변동시키며 네트워크 성능을 분석합니다. 특히 픽셀 수준의 예측 과제에서 문맥 길이와 성능 간의 강한 관계를 발견하였습니다. 이를 통해, 문맥 길이가 길어질수록 성능이 크게 향상되는 경향을 확인했습니다.

- **Performance Highlights**: 최근의 장문 모델들은 생물 의학 이미지에 대한 효율성을 개선하면서도 유사한 성능을 유지함을 보여주었습니다. 그러나 몇몇 영역에서는 여전히 성능 차이가 존재함을 강조합니다. 이 연구는 생물 의학 이미징에서 장문 모델을 사용할 때의 잠재력 및 과제를 강조하며, 향후 연구의 방향성을 제시합니다.



### DreamDrive: Generative 4D Scene Modeling from Street View Images (https://arxiv.org/abs/2501.00601)
- **What's New**: 이 논문에서는 DreamDrive라는 새로운 방법론을 도입하여 자율주행을 위한 4D 공간-시간 장면 생성을 다루고 있습니다. 기존의 생성 및 재구성 기반 방법들이 가지는 한계를 극복하고, 3D 일관성을 유지하면서도 일반화 가능한 4D 주행 장면을 합성할 수 있도록 설계되었습니다. 특히, 비디오 확산 모델(video diffusion model)과 하이브리드 가우시안 표현(hybrid Gaussian representation)을 결합하여 더 높은 품질의 장면 생성을 가능하게 합니다.

- **Technical Details**: DreamDrive는 에고 차량의 주행 궤적을 이용해 시각적 참조를 생성하고, 이를 4D로 올리기 위해 고안된 새로운 하이브리드 가우시안 표현을 사용합니다. 이 방법론은 정적 배경을 모델링하기 위해 시간 독립적인 가우시안을, 동적 객체를 모델링하기 위해 시간 의존적인 가우시안을 활용하여 4D 장면을 구성합니다. 이러한 접근을 통해 이미지 감독(image supervision)만으로도 고품질의 3D 일관성을 유지하는 주행 비디오를 생성할 수 있습니다.

- **Performance Highlights**: DreamDrive는 nuScenes 데이터셋과 실제 주행 시나리오에서 검증되었으며, 기존 방법보다 30% 향상된 시각적 품질로 3D 일관성이 확보된 주행 비디오를 생성합니다. 또한 이 방법은 자율주행의 인식(perception) 및 계획(planning) 작업에서도 효과적으로 적용될 수 있음을 입증했습니다. 최종적으로, 이 연구는 다양한 주행 시나리오에서의 보편성을 높이는 데 기여합니다.



### VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM (https://arxiv.org/abs/2501.00599)
Comments:
          17 pages, 12 figures, technical report

- **What's New**: 이번 논문에서는 Video LLM에 대한 VideoRefer Suite를 소개하여 비디오의 세부적인 공간-시간적 이해를 향상시키는 방안을 제안합니다. 특히, VideoRefer Suite는 데이터셋, 모델, 벤치마크의 세 가지 중요한 측면을 통해 개발되었습니다. 이 연구는 고품질 객체 수준 비디오 지침 데이터와 포괄적인 벤치마크의 부족이라는 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: VideoRefer Suite는 다중 에이전트 데이터 엔진을 통해 대규모 고품질 객체 수준 비디오 지침 데이터셋인 VideoRefer-700K를 생성합니다. 이 데이터셋은 비디오의 각 객체에 대해 다양한 수준의 질문과 답변 쌍을 포함하고 있습니다. 모델 측면에서, VideoRefer는 공간-시간적 객체 인코더 및 적응형 시간 토큰 병합 모듈을 활용하여 정밀한 객체 이해를 지원합니다. 마지막으로, VideoRefer-Bench를 통해 모델 성능을 종합적으로 평가합니다.

- **Performance Highlights**: VideoRefer 모델은 비디오 참조 벤치마크에서 뛰어난 성능을 보여줍니다. 이 모델은 기본 비디오 객체 참조, 복잡한 객체 간의 관계 분석, 객체 검색 작업 등이 가능하며, 사용자 상호작용을 유지하는 고급 기능을 제공합니다. 다양한 시간대와 객체를 통해 모델의 포괄적인 캡션 작성 및 추론 능력도 평가됩니다.



### Unbiased GNN Learning via Fairness-Aware Subgraph Diffusion (https://arxiv.org/abs/2501.00595)
- **What's New**: 본 논문에서는 Graph Neural Networks (GNNs)의 편견 없는 학습을 위한 새로운 방법론인 Fairness-Aware Subgraph Diffusion (FASD)을 제안합니다. GNNs가 성별이나 나이와 같은 민감한 노드 속성으로부터 편향된 예측을 생성하는 경향이 있다는 점에 주목하며, 이를 개선하기 위해 소규모 서브그래프를 샘플링하고 stochastic differential equations (SDEs)을 통해 서브그래프 디바이징을 수행합니다.

- **Technical Details**: FASD는 원래의 대규모 입력 그래프에서 소규모 서브그래프를 전략적으로 샘플링한 후, 생성적 공정성 인식 그래프 확산 프로세스를 통해 서브그래프의 편향을 제거합니다. 이 과정에서 추가적인 적대적 편향 교란을 서브그래프에 적용하여, 데이터 내의 편향의 동향을 학습합니다. 훈련된 모델은 역확산 과정을 통해 원본 서브그래프 샘플을 디바이즈하여 공정한 노드 예측을 유도합니다.

- **Performance Highlights**: 실험 결과, FASD 방법이 다양한 벤치마크 데이터세트에서 기존 상태-최고 공정 GNN 모델들에 비해 우수한 성능을 보임을 입증하였습니다. 본 연구의 주요 기여는 공정성 인식 그래프 확산을 통한 공동 작업을 가능하게 함으로써 GNN 학습의 공정성을 향상시키고, 다양한 데이터셋에서 입증된 편향 제거 능력입니다.



### Causal Graph Guided Steering of LLM Values via Prompts and Sparse Autoencoders (https://arxiv.org/abs/2501.00581)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 가치 정렬(value alignment) 과정을 개선하기 위해 인과 그래프(causal graph)를 활용하여 다양한 가치 간의 관계를 명확히 합니다. 기존의 방법론은 주로 제한된 가치 세트에 초점을 맞추고 있었으나, 본 연구는 가치 간의 상호작용을 분석함으로써 LLM의 행동을 보다 효과적으로 조정할 수 있는 새로운 프레임워크를 제안합니다. 또한, 파라미터 수가 적은 두 가지 조정 메커니즘, 즉 프롬프트 템플릿 프로세스(prompt template steering)와 희소 오토인코더(Sparse Autoencoder) 기능 조정 방법을 도입하였습니다.

- **Technical Details**: 연구에서는 LLM의 내부에서 발생하는 다양한 가치 간의 인과 관계를 표현하기 위해 인과 그래프를 채굴합니다. 이 그래프는 각 가치가 다른 가치에 미치는 영향을 보여주어, LLM의 결정 과정을 이해하는 데 도움을 줍니다. 특히, 두 가지 새로운 경량의 조정 메커니즘을 통해 특정 가치 차원을 변경할 때 다른 차원에 미치는 영향을 분석하며, 이 과정을 통해 LLM이 인간의 가치에 더욱 일치하도록 유도합니다.

- **Performance Highlights**: Gemma-2B-IT 및 Llama3-8B-IT 모델에 대한 광범위한 실험을 통해 제안된 조정 방법의 효과성과 통제 가능성을 검증했습니다. 연구 결과, 인과 그래프를 활용한 조정 방식이 LLM의 출력 일관성을 강화하며, 다양한 가치 차원 간의 상호작용을 조정하는 데 긍정적인 영향을 미친다는 점이 확인되었습니다. 이러한 발견은 AIl의 가치 일치성을 향상시키는 데 큰 기여를 할 것으로 기대됩니다.



### An Overview and Discussion on Using Large Language Models for Implementation Generation of Solutions to Open-Ended Problems (https://arxiv.org/abs/2501.00562)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)을 활용하여 전통적인 알고리즘 규격을 초월하는 자동 구현 생성 방법을 제시하는 새로운 기회를 강조합니다. LLMs는 문제 프레이밍, 잠재적 해결 접근법 탐색 및 기능 조합 등의 문제 해결 활동을 지원할 수 있는 가능성을 지니고 있습니다. 또한, 이 모델들은 외부 문서에서 새로운 기능을 학습하거나 이전에 생성된 구현에 기반하여 동적으로 지식을 업데이트할 수 있습니다.

- **Technical Details**: 논문은 문제 해결을 위한 구현 생성 활동을 모델링하는 데 LLMs의 잠재력을 조사했습니다. Prompting, Reinforcement Learning(RL) 및 Retrieval-Augmented Generation(RAG)과 같은 방법론을 사용하여 기존의 자동화된 구현 생성 방법에서 다루어지지 않았던 문제 해결 활동을 구현할 수 있는 방안을 논의합니다. 특히, LLMs의 다양한 연관성을 학습할 수 있는 능력을 활용하여 다중 모달 데이터에 대한 이해를 심화시키는 것이 중요하다고 설명합니다.

- **Performance Highlights**: 전통적인 구현 생성 방법들은 주로 잘 정의된 문제와 일명 '하이레벨' 사양에 착안하여 이루어져 왔으나, 이 연구는 다양한 문제 정의가 필요한 개방형 문제 해결에 대해 LLMs가 기여할 수 있는 새로운 기회를 제공합니다. LLMs는 팀 작업에서의 문제 프레이밍과 해결 접근법 탐색을 지원할 수 있으며, 시도-오류 및 신속한 프로토타입 제작을 통해 새로운 기회와 한계를 탐색할 수 있는 자동화를 제공해야 합니다.



### Re-evaluating Automatic LLM System Ranking for Alignment with Human Preferenc (https://arxiv.org/abs/2501.00560)
- **What's New**: 이 논문은 다양한 LLM의 성능을 평가하고 순위를 매기는 자동화된 평가 프레임워크인 LLM bencher의 필요성을 강조합니다. 기존의 연구에서는 이 시스템의 구성 요소를 선택하는 방법이나 조합이 결과에 미치는 영향을 충분히 탐구하지 않았습니다. 이 연구는 통제된 실험을 통해 LLM 평가의 자동화를 위한 구성 요소 선택에 대한 추천을 제공합니다.

- **Technical Details**: 자동 LLM bencher는 입력 집합(input set), 평가 모델(evaluation model), 평가 유형(evaluation type), 집계 방법(aggregation method) 네 가지 구성 요소로 이루어져 있습니다. 특히, 이 연구에서는 Arena Hard와 Llama-3.1-70B 같은 모델을 사용하여 LLM의 성능 평가는 물론, 평가 유사성이 높은 시스템 간의 비교 시 성능 저하가 발생함을 밝혀냈습니다. 또, 각 구성 요소의 선택이 LLM bencher의 효율성에 미치는 영향을 실험을 통해 분석하였습니다.

- **Performance Highlights**: 연구 결과, Arena Hard의 입력 집합을 사용할 경우 Chatbot Arena의 시스템 순위와의 상관관계가 항상 높게 나타났습니다. 또, GPT-4-turbo와 같은 강력한 평가 모델은 LLM 평가에서 뛰어난 성능을 보이는 반면, 비슷한 성능의 시스템들을 평가할 때 성능이 급격히 하락하는 한계가 있음을 발견했습니다. 마지막으로, 인스턴스 수준(instance-level)에서의 평가 결과는 시스템 수준(system-level) 평가의 좋은 참고자료가 될 수 있는 가능성을 보여줍니다.



### AraSTEM: A Native Arabic Multiple Choice Question Benchmark for Evaluating LLMs Knowledge In STEM Subjects (https://arxiv.org/abs/2501.00559)
- **What's New**: 이번 논문에서는 AraSTEM이라는 새로운 아랍어 다지선택 문제 데이터셋을 소개합니다. 이 데이터셋은 STEM 분야에서 LLMs(대규모 언어 모델)의 지식을 평가하는 목적을 가지고 있으며, 다양한 주제를 포함하고 있습니다. 기존의 LLM 평가 기준이 주로 영어에 초점을 맞추고 있어, 다국어 지원 모델의 부족한 아랍어 평가 지표를 보완합니다.

- **Technical Details**: AraSTEM 데이터셋은 총 11,637개의 MCQ(다지선택 문제)로 구성되어 있으며, 초등학교와 중학교 수준부터 고급 생물학, 물리학, 의학 등 다양한 주제를 포함합니다. 데이터는 여러 출처에서 수집되었으며, 각 질문의 출처에 대한 링크가 포함되어 있습니다. 수집 과정은 웹 스크래핑, 수동 추출 및 LLM을 활용한 방법을 포함하고 있습니다.

- **Performance Highlights**: 초기 실험 결과는 공개된 다양한 크기의 LLM 모델들이 AraSTEM 데이터셋에서 저조한 성과를 보였다는 것을 보여줍니다. 이는 아랍어와 과학 분야에 대한 언어 모델의 이해도가 부족함을 시사하며, 더 지역화된 언어 모델 개발의 필요성을 강조합니다. Hugging Face를 통해 데이터셋은 무료로 접근할 수 있습니다.



### Monty Hall and Optimized Conformal Prediction to Improve Decision-Making with LLMs (https://arxiv.org/abs/2501.00555)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 과신으로 인한 잘못된 예측을 줄이기 위해 새로운 방법론을 제안합니다. 기존의 conformal prediction (CP) 기술을 바탕으로 CP-OPT라는 최적화 프레임워크를 도입하여 예측 세트의 크기를 최소화하면서도 정확성을 유지하는 방법을 보여줍니다. 또한 Monty Hall 문제에서 영감을 받아 conformal revision of questions (CROQ)을 통해 문제를 수정하여 선택지를 좁히는 새로운 접근 방식을 제안합니다.

- **Technical Details**: CP-OPT는 예측 세트의 크기를 줄이면서과 커버리지를 유지하는 점수(score)를 학습하는 최적화 기법입니다. 이를 통해 LLM의 로그잇(logits) 또는 휴리스틱 점수(heuristic scores)에 의존하지 않고 보다 질 높은 점수를 생성할 수 있습니다. CROQ는 문제의 선택지를 예측 세트로 제한함으로써 정확성을 향상시키는 방식으로 기능합니다. 이 과정에서 CP의 커버리지 보장이 올바른 선택지를 포함하도록 보장합니다.

- **Performance Highlights**: 실험 결과, CP-OPT는 MMLU, ToolAlpaca, TruthfulQA 데이터셋에서 예측 세트의 크기를 현저히 감소시키면서도 커버리지를 유지하는 데 성공했습니다. 또한 CROQ는 CP-OPT 점수와 함께 사용할 경우 표준 추론에 비해 정확도를 증가시키는 데 기여했습니다. 이러한 CP-OPT와 CROQ의 결합은 LLM 기반 의사결정의 안전성과 정확성을 동시에 향상시키는 강력한 프레임워크를 제공합니다.



### Superposition in Transformers: A Novel Way of Building Mixture of Experts (https://arxiv.org/abs/2501.00530)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 새로운 작업이나 도메인에 적응시키는 과정에서 겪는 치명적인 잊음(catatstrophic forgetting) 문제를 해결하기 위한 Superposition in Transformers라는 혁신적인 아키텍처를 제안합니다. 이 아키텍처는 기존 모델과 미세 조정된 모델의 숨겨진 표현(hidden representations)을 공유된 파라미터 공간 내에서 중첩(supeposition) 시킵니다. 이를 통해 기존 지식을 보존하면서 도메인 특정 전문성을 추가할 수 있는 새로운 패러다임을 제공합니다.

- **Technical Details**: 제안된 방법은 B-spline 기반의 혼합 계수(blending coefficients)와 자동 인코더(autoencoders)를 활용하여 입력 데이터 분포에 따라 적응적으로 숨겨진 상태를 재구성(reconstruct)합니다. 이러한 접근 방식은 치명적인 잊음 현상을 효과적으로 완화하고, 모델 상태 간의 동적 전환을 지원하여 추론(inference) 과정에서 원활한 작업 전환을 가능하게 합니다. Superposition을 사용함으로써 기존 모델의 기능이 유지됩니다.

- **Performance Highlights**: Superposition in Transformers는 모델의 원래 성능을 저해하지 않으면서도 새로운 작업에 대한 적응성을 높입니다. 실험 결과, 이 방법이 기존의 미세 조정 기법보다 성능 저하를 더욱 효과적으로 방지하고, 다양한 도메인에서의 효율성을 보여줍니다. 이로 인해 대형 언어 모델의 활용 가능성이 극대화됩니다.



### PyMilo: A Python Library for ML I/O (https://arxiv.org/abs/2501.00528)
Comments:
          7 pages, 5 figures, 2 tables, 3 code blocks

- **What's New**: PyMilo는 기존 머신러닝 모델 저장 포맷의 한계를 극복하기 위해 개발된 오픈 소스 Python 패키지로, 안전하고 투명한 모델 전송 및 배포 방법을 제공합니다. 기존의 pickle과 같은 바이너리 포맷의 신뢰성 및 안전성 문제를 해결하기 위해, ML 모델을 인간이 읽을 수 있는 비실행(non-executable) 포맷으로 직렬화합니다. 이를 통해 모델을 쉽게 공유하고 배포할 수 있으며, 생산 환경에서의 재구성과 배치가 용이해집니다.

- **Technical Details**: PyMilo는 ML 모델의 전체 알고리즘을 포착하여 직렬화하며, 구조, 파라미터 값 및 커스텀 기능 구현을 포함합니다. 또한, Chain of Responsibility 디자인 패턴을 사용하여 비직렬화 가능한 복잡한 데이터 구조의 직렬화 및 비직렬화를 관리합니다. 16개의 전용 Transporter를 통해 scikit-learn 모델의 데이터 구조를 효과적으로 처리하며, "ML Streaming" 기능을 통해 웹 환경에서 모델을 원활하게 배포하고 실시간 상호작용을 가능하게 합니다.

- **Performance Highlights**: PyMilo의 주요 성능 요소는 다양한 ML 모델을 정확하게 복원할 수 있는 능력입니다. 이를 통해 트래픽이 많은 웹 서비스 환경에서도 대규모 ML 모델을 효율적으로 운영할 수 있습니다. 또한, ML Streaming 기능은 반환된 예측, 모델 재훈련 및 P2P 모델 공유를 가능하게 하여 실시간 작업 처리를 최적화합니다.



### TinyHelen's First Curriculum: Training and Evaluating Tiny Language Models in a Simpler Language Environmen (https://arxiv.org/abs/2501.00522)
- **What's New**: 이번 연구에서는 기계학습에서 언어 모델의 학습 효율성을 향상시키기 위해 간소화된 언어 환경을 수립하는 방안을 제안합니다. 기존의 대형 언어 모델(Large Language Models, LLMs) 학습에 필요한 방대한 데이터셋과 자원을 줄이기 위해, 텍스트 데이터를 정제하고 노이즈를 제거한 간소화된 데이터셋을 만듭니다. 이런 방법을 통해 조그마한 언어 모델(Tiny LMs)이 instruction-following(지시 따르기) 능력을 더 효과적으로 학습하도록 합니다.

- **Technical Details**: 이 연구에서는 'no noise, low complexity' 원칙을 바탕으로 한 텍스트 데이터 개선 파이프라인을 구현하여 71M Leaner-Pretrain, 7M Leaner-Instruct, Leaner-Glue 및 Leaner-Eval 등 간결한 언어 모델 훈련 및 평가 데이터셋을 생성합니다. 이 데이터셋은 기존의 언어 모델 훈련 데이터의 구성 DNA와 평가 기준을 유지하면서도 언어적으로 더욱 단순화된 특징을 지니고 있습니다. 이를 통해 저비용(high efficiency) 모델을 위한 학습 데이터를 효율적으로 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, Leaner-Pretrain 데이터셋으로 사전 학습된 조그마한 언어 모델들이 원본 데이터셋에 비해 instruction-following 성능이 향상되었음을 확인하였습니다. 특히, 저희의 방안으로 학습한 모델은 문법, 일관성, 구체성을 개선하여 다른 모델들보다 더 높은 성능을 기록했습니다. 이러한 결과는 기존 복잡한 모델 환경에서도 저비용으로 변형된 데이터셋을 통해 효과적인 학습이 가능하다는 것을 보여줍니다.



### A Method for Enhancing the Safety of Large Model Generation Based on Multi-dimensional Attack and Defens (https://arxiv.org/abs/2501.00517)
- **What's New**: 본 논문에서는 복잡한 공격 지침을 받을 때 대형 모델들이 생성하는 유해 콘텐츠 문제를 해결하는 방법을 제안합니다. 특히, 다차원 공격 방어에 맞춘 데이터를 구축하여 대형 모델의 생성 보안을 강화하는 데 초점을 맞추고 있습니다. 이 방법은 안전한 정렬 학습(safe alignment learning)의 효과성을 향상시키기 위해 공격 지침 차원의 다양성과 안전한 응답 생성 정확성을 혁신적으로 증가시켜야 합니다.

- **Technical Details**: 본 연구의 핵심은 대형 모델의 안전성 및 보안성을 향상하기 위한 방법으로, 새로운 보안 평가 벤치마크를 설계하고 Llama3.2를 기준 모델로 비교 실험을 수행한 것입니다. 이러한 실험은 모델이 복잡한 공격 지침을 처리하며 생성 보안을 크게 향상시킬 수 있음을 보여줍니다. 메서드는 효과적인 다차원 공격 방어를 위한 데이터 정렬(data alignment) 구축에 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 복잡한 공격 지침 하에서도 대형 모델의 생성 보안을 크게 향상시킬 수 있었으며, 모델의 일반적인 능력을 유지하고 강화할 수 있는 것으로 나타났습니다. 이러한 결과는 대형 모델들이 복잡한 공격에 대한 방어력을 개선할 수 있는 가능성을 제공합니다.



### H-Net: A Multitask Architecture for Simultaneous 3D Force Estimation and Stereo Semantic Segmentation in Intracardiac Catheters (https://arxiv.org/abs/2501.00514)
- **What's New**: 이번 연구에서는 카테터에서 두 개의 서로 다른 각도에서 동시에 분할(segmentation)하고 3D에서 적용된 힘(force)을 추정할 수 있는 새로운 경량 다중 입력-다중 출력 인코더-디코더 기반 아키텍처를 제안합니다. 이 아키텍처는 비싼 센서를 사용하지 않고, 두 개의 X-레이 이미지를 동시에 처리하여 카테터의 변위를 보여주며, 따라서 카테터의 시각화와 힘 측정을 동시에 지원할 수 있습니다. 기존의 연구와는 달리, 이 모델은 분산형 네트워킹을 사용하여 연산 자원을 최적화하며, 종합적으로 상호작용된 힘과 분할 성능을 높였습니다.

- **Technical Details**: 제안된 H-Net 아키텍처는 두 개의 X-레이 이미지를 입력으로 받아 각각의 이미지에서 추출된 특징 맵을 통합하여 카테터의 3D 구조를 인식합니다. 이 시스템은 두 개의 병렬 인코더와 디코더를 활용하여 두 개의 분할 헤드와 하나의 힘 추정 헤드를 통해 작동합니다. 특히, 힘 추정 헤드는 카테터의 꼭지점에서의 힘을 x, y, z 방향으로 예측하며, 이는 센서가 없는 방식으로 작동하기 때문에 생산 비용을 절감할 수 있습니다. 또한, 이 아키텍처는 효율성을 위해 네트워크 간의 매개변수를 공유하여 계산 복잡성을 최소화합니다.

- **Performance Highlights**: 제안된 H-Net은 기존의 선구적인 방법들과 비교해 최초로 3D 힘 추정과 스테레오 분할 작업을 동시에 처리하며 뛰어난 성능을 보여주었습니다. 실험 결과, 합성된 데이터셋 뿐만 아니라 실제 RGB 데이터셋에서도 탁월한 성능을 입증하였으며, 다중 태스크 시스템이 적용된 걸로 인해 카테터 분할 및 힘 추정의 신뢰성이 증가했습니다. 따라서 H-Net은 기존의 모델들과 차별화된 우수한 성능을 바탕으로 심혈관 시술에서의 안전성과 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Exploring Physics-Informed Neural Networks for Crop Yield Loss Forecasting (https://arxiv.org/abs/2501.00502)
Comments:
          6 pages, 2 figures, NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning

- **What's New**: 이번 연구에서는 기후 변화에 대응하기 위해 극한 날씨 조건에서 작물 생산성을 평가하는 새로운 접근법을 제안합니다. 이는 기존의 머신 러닝(ML) 모델의 한계를 극복하고, 물 사용량과 물 부족에 대한 작물의 민감도를 픽셀 수준에서 추정하여 작물 수확량 손실을 정량화합니다. 향상된 손실 함수를 활용하여 작물 수확량과 물 부족 간의 관계를 물리적 원리를 바탕으로 추정하며, 해석 가능성과 물리적 일관성을 동시에 제공합니다.

- **Technical Details**: 물 사용량을 효과적으로 추정하기 위해 다양한 기후 데이터, Sentinel-2 위성 이미지, 그리고 픽셀 수준 작물 수확량 데이터를 사용합니다. 연구에서는 물 사용량, 즉 evapotranspiration(ET)과 작물 수확량 손실 간의 관계를 규명하기 위해 Recurrent Neural Network(RNN)을 활용합니다. ET의 감소가 수확량 감소와 밀접하게 연결되어 있음을 확인하고, 이를 통해 작물의 물 부족에 대한 감도를 학습합니다.

- **Performance Highlights**: 모델은 기존의 최첨단 모델인 RNN과 Transformers의 성능을 초과하거나 유사한 수준의 R2 값인 0.77을 기록하여 높은 예측 정확성을 보여줍니다. 이는 산업계와 정책 입안자, 농부들이 극한 날씨에 적응하는 데 큰 도움이 될 수 있으며, 해석 가능한 결과를 바탕으로 농업 분야의 실질적인 의사 결정을 지원합니다.



### Differentiable Prompt Learning for Vision Language Models (https://arxiv.org/abs/2501.00457)
- **What's New**: 이번 논문에서는 기존의 수동적 프롬프트 디자인을 자동화하기 위한 새로운 방법인 differentiable prompt learning (DPL)을 제안합니다. DPL은 최적의 프롬프트 길이를 자동으로 결정하는 최적화 문제로 설정되며, 이를 통해 성능을 최대화하는 것을 목표로 합니다. DPL 방법은 사전 훈련된 CLIP 모델에 적용되어, 기존 방법들보다 높은 신뢰도로 딥 연속 프롬프트 구성 파라미터를 찾을 수 있음을 입증하였습니다.

- **Technical Details**: DPL 방법은 연속 프롬프트의 컨텍스트 길이와 깊이를 자동으로 결정하는 데 초점을 맞추고 있습니다. DPL은 최적화 과정에서 각 레이어에 추가될 프롬프트의 깊이와 컨텍스트 길이를 조정함으로써, 프롬프트 학습의 유연성을 높입니다. 또한, DPL 방법은 수동 설계를 거치지 않고도 적은 데이터만으로 성능을 극대화할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: DPL 방법은 다운스트림 과제에서 평균 테스트 정확도를 11개 데이터셋에서 기존 방법보다 2.60% 향상시키는 성능을 보였습니다. 이 방법은 기존의 복잡한 설계와도 호환 가능하여, 최적의 딥 연속 프롬프트 구성을 데이터셋에 맞춰 조정함으로써 성능 향상이 가능합니다. 따라서 DPL 방법은 각기 다른 모델에 대해 비용 없이 쉽게 배포할 수 있는 장점을 가집니다.



### Do Students with Different Personality Traits Demonstrate Different Physiological Signals in Video-based Learning? (https://arxiv.org/abs/2501.00449)
- **What's New**: 이 연구에서는 개인의 성격 특성이 학업 성과에 미치는 영향을 분석하는 새로운 방법을 제안합니다. 기존의 personality trait의 평가 시스템은 한계가 많았지만, 이번 연구는 생리학적 신호(physiological signals)를 통한 새로운 평가 방법을 개발하고자 했습니다. 이는 응답의 신뢰성을 높이고, 더 정확한 결과를 도출할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 실험에 참여한 30명의 참가자의 생리학적 데이터(heart rates, GSR values, voice frequencies 등)와 성격 특성 간의 상관관계를 분석하였습니다. 구체적으로, extraversion, agreeableness, conscientiousness, 그리고 openness to experiences와 같은 성격 특성이 학습 중 생리적 신호의 변화와 밀접하게 연관되어 있음을 발견하였습니다. 이는 기존 평가 방법의 한계를 극복하는 데 기여할 것입니다.

- **Performance Highlights**: 연구 결과, 학생의 성격 특성과 생리학적 신호 변화 간의 유의미한 상관관계를 발견하였습니다. 이는 교수 학습의 맥락에서 개인의 성격을 평가하는 데 있어, 생리학적 신호가 중요한 지표가 될 수 있음을 시사합니다. 이번 연구는 학습 성과 예측을 위한 더 신뢰할 수 있는 평가 방법을 제시함으로써, 교육 분야에서의 응용 가능성을 넓힐 것으로 기대됩니다.



### Generalizing Trust: Weak-to-Strong Trustworthiness in Language Models (https://arxiv.org/abs/2501.00418)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문은 신뢰성(compared to AI 시스템의 성능) 속성이 약한 모델에서 강한 모델로 전이될 수 있는지를 탐구하는 첫 번째 연구로, 이를 위해 약한 모델에서의 조정(fine-tuning) 및 신뢰성 정규화(regularization)를 사용한 방법들을 소개합니다. 또한, Weak와 Weak-to-Strong Trustworthiness Fine-tuning(Weak+WTS TFT)이라는 새로운 훈련 전략을 제안하여 약한 모델의 신뢰성을 개선할 수 있는 가능성을 시사합니다. 이를 통해 약한 모델의 출력을 통해 조정된 강한 모델이 약한 모델의 신뢰성 속성을 어떻게 수용하고 향상시킬 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 이 연구에서는 Weak Trustworthiness Fine-tuning(Weak TFT)과 Weak+WTS TFT라는 두 가지 훈련 전략을 도입하여 약한 모델에서 강한 모델로의 신뢰성 전이를 촉진합니다. Weak TFT는 약한 모델의 신뢰성을 일반화하기 위해 손실 함수(loss function)에 공정성(fairness), 강건성(robustness), 개인정보 보호(privacy) 제한 조건을 통합하는 방식을 사용합니다. Weak+WTS TFT는 약한 모델의 출력을 통해 훈련된 강한 모델에서 신뢰성 정규화를 추가하여 두 모델 모두에서 신뢰성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 두 모델이 정규화되었을 때 공정성, 강건성 및 OOD(Out-Of-Distribution) 강건성과 같은 일부 신뢰성 속성이 유의미하게 향상되는 것을 보여주었습니다. 반면, 개인정보 보호와 같은 일부 속성은 약한 모델에서 강한 모델로의 전이에 대해 개선되지 않음을 발견하였습니다. 이러한 결과는 약한 모델에서 강한 모델로 신뢰성이 전이되고 강화될 수 있음을 강력히 뒷받침하며, 신뢰성 높은 AI 시스템을 개발하기 위한 새로운 가능성을 제시합니다.



### TSPE: Task-Specific Prompt Ensemble for Improved Zero-Shot Audio Classification (https://arxiv.org/abs/2501.00398)
Comments:
          5 pages

- **What's New**: 최근 오디오-언어 모델(Audio-Language Models, ALMs)은 자연어 프롬프트(prompt)를 활용하여 제로샷 오디오 분류(zero-shot audio classification)에서 뛰어난 성능을 발휘하고 있습니다. 본 논문에서는 다양한 오디오 분류 작업을 위한 맞춤형 프롬프트를 생성하여 ALMs의 성능을 향상시키는 간단하면서도 훈련이 필요 없는 하드 프롬프트 기법인 TSPE(Task-Specific Prompt Ensemble)를 제안합니다. 이 방식은 일반적인 템플릿 기반 프롬프트 대신, "터널에서 나오는 자동차 소리"와 같이 컨텍스트가 풍부한 프롬프트를 생성합니다.

- **Technical Details**: TSPE는 레이블 정보를 활용하여 소리의 특성과 출처를 식별하고, 이를 프롬프트에 통합하여 오디오 분류에 사용합니다. 특히, 프롬프트 앙상블을 통해 TSPE가 생성한 작업별 프롬프트의 정렬을 강화합니다. 12개의 다양한 오디오 분류 데이터셋에서 평가한 결과, TSPE는 기존 제로샷 평가에서 1.23-16.36%의 성능 개선을 보여주며, 이는 ALMs의 성능 향상에 기여합니다.

- **Performance Highlights**: TSPE는 추가적인 훈련 없이 오디오 분류 성능을 크게 향상시킬 수 있는 방법론으로, 다양한 오디오 환경에서 잘 작동합니다. 특히, 기존의 ALMs가 OOD(out-of-distribution) 데이터셋에서 성능이 저하되는 한계를 극복하는데 기여합니다. 이러한 점에서 TSPE는 제로샷 오디오 분류 기술의 미래 가능성을 제시하며, 오디오-언어 모델의 진화를 지원합니다.



### Efficient Relational Context Perception for Knowledge Graph Completion (https://arxiv.org/abs/2501.00397)
- **What's New**: 이 논문은 기존의 Knowledge Graph Embedding (KGE) 방법의 한계를 극복하기 위해 새로운 Triple Receptance Perception (TRP) 아키텍처를 제안합니다. TRP는 지식 그래프에서 동적이고 순차적인 컨텍스트를 캡처하도록 설계되어, 다양한 그래프 컨텍스트에 따라 적응하는 임베딩을 학습할 수 있습니다. 또한, Tucker 분해를 결합하여 효율적인 관계 디코딩을 가능하게 하고 있습니다.

- **Technical Details**: TRP 아키텍처는 시퀀스 정보를 모델링하여 엔티티와 관계의 동적 맥락을 학습할 수 있도록 합니다. 이 방법은 복잡한 관계 종속성을 인코딩하는 모델의 능력을 향상시키며, 고차원 텐서를 분해하여 compact yet expressive한 표현을 가능하게 만듭니다. Tucker 분해는 엔티티와 관계의 임베딩을 구성 요소로 모델링하여 효율적인 표현 학습을 지원합니다.

- **Performance Highlights**: YAGO3-10, UMLS, FB15k와 FB13 같은 벤치마크 데이터셋에서의 실험을 통해, 제안된 방법이 링크 예측 및 삼중 분류 작업에서 여러 최신 모델보다 우수한 성능을 보임을 입증하였습니다. 이러한 결과들은 TRP 아키텍처의 효과성을 강하게 뒷받침하고 있습니다.



### Proactive Conversational Agents with Inner Thoughts (https://arxiv.org/abs/2501.00383)
- **What's New**: 이번 논문에서는 대화형 AI가 대화에서 적극적인 역할을 할 수 있도록 하는 방법에 대해 다룹니다. 특히 다중 참여자 대화에 초점을 맞추어, AI가 단순히 반응하는 것이 아니라 스스로 생각을 형성하고 적절한 순간에 기여하는 방식으로 프로액티브(proactive)하게 행동할 수 있음을 제안합니다. 이는 기존의 연구들이 단순히 다음 화자를 예측하는 데 중점을 두었던 것과는 달리, AI가 보다 인간과 유사한 방식으로 대화에 참여할 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 24명의 참여자와 진행된 기초 연구를 바탕으로, 'Inner Thoughts' 프레임워크(Inner Thoughts framework)를 도입하였습니다. 이 프레임워크는 AI가 대화 중에 비공식적인 생각의 흐름을 유지하게 하여, 그 생각을 표현하고 싶어 하는 내재적인 동기를 모델링합니다. AI가 이러한 내적인 사고 과정을 통해 보다 적극적으로 대화에 참여할 수 있도록 설계된 시스템을 AI 놀이터 웹 앱 및 챗봇 두 가지 형태로 구현하였습니다.

- **Performance Highlights**: 우리 프레임워크는 인류 연구 및 사용자 연구를 통해 기존의 기준선(baselines)을 크게 초월하는 성능을 보여주었습니다. 특히 인간적인 감정 표현(anthropomorphism), 일관성(coherence), 지능(intelligence), 턴 테이킹(turn-taking) 적합성 등 여러 측면에서 향상된 결과를 보였습니다. 이러한 결과는 AI가 대화에서 더 자연스럽고 적극적으로 소통할 수 있도록 하는 가능성을 제시합니다.



### Adventures in Demand Analysis Using AI (https://arxiv.org/abs/2501.00382)
Comments:
          42 pages, 9 figures

- **What's New**: 본 논문은 인공지능(AI)을 활용하여 다중 모달(multi-modal) 제품 표현을 통합함으로써 경험적 수요 분석(empirical demand analysis)을 발전시킵니다. 전통적인 방법으로는 요약하기 어려운 품질, 브랜드, 시각적 특성 등 미세한 속성을 포착하는 transformer 기반의 embedding 모델을 사용합니다. 이러한 방식은 판매 순위와 가격의 예측 정확성을 크게 향상시킴과 동시에 가격 탄력성(price elasticity)의 보다 믿을 만한 인과 추정(causal inference)을 제공합니다.

- **Technical Details**: 우리는 Amazon.com의 장난감 자동차에 대한 판매 순위와 가격 데이터를 사용하여 다채로운 제품 정보를 활용하는 transformer 기반 모델을 시연합니다. 데이터는 텍스트 설명, 이미지, 판매 순위, 가격을 포함하며, 이를 이용해 생성한 수치 embedding은 제품 속성을 잘 포착합니다. 이러한 embedding은 가격과 수량 신호를 예측하는 데 활용되며, 기존의 단순한 테이블 데이터에 비해 예측 정확도가 향상됩니다.

- **Performance Highlights**: AI 기반 embedding은 가격 탄력성을 결정짓는 강력한 요소로 작용하며, 이러한 embedding을 통해 소비자 수요에 대한 보다 미세하고 신뢰할 수 있는 추정을 얻을 수 있습니다. 연구 결과, 제품 특성에 따라 가격 탄력성의 이질성(heterogeneity)이 강하게 나타났으며, 이는 AI 기반 표현의 경제적 가치를 강조합니다. 이 방법론은 가격이 변화할 때 소비자 반응을 이해하는 데 상당한 기여를 할 것으로 기대됩니다.



### Design Optimizer for Soft Growing Robot Manipulators in Three-Dimensional Environments (https://arxiv.org/abs/2501.00368)
Comments:
          20 pages, 10 figures

- **What's New**: 이 논문은 소프트 성장 로봇의 디자인 최적화를 위한 새로운 접근 방식을 제안합니다. 본 연구에서는 기존에 평면 조작기를 위해 설계된 최적화 도구를 3차원으로 확장하여 복잡한 환경에서의 조작을 가능하게 합니다. 또한, 이 도구는 특정 작업을 위한 로봇의 최적 크기를 제안하여 엔지니어와 로봇 애호가들이 로봇을 제작하기 전에 사용할 수 있도록 합니다.

- **Technical Details**: 소프트 성장 로봇의 설계 과정은 다중 목표 최적화 문제(Multi-Objective Optimization Problem)를 모델링하여 소프트 조작기의 운동학적 사슬을 다듬습니다. 진화 계산(Evolutionary Computation) 알고리즘에 통합된 새로운 Rank Partitioning 알고리즘 덕분에 최적화 도구는 자원 사용의 효율성뿐만 아니라 타겟에 도달하는 높은 정밀도를 달성합니다. 이 방법은 로봇 디자인의 복잡하고 비선형적인 설계를 해결하기 위해 인공지능 활용을 강조합니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 3차원 작업을 해결하는 데 매우 높은 성능을 보입니다. 비교 실험에서, 최적화 도구는 여러 진화 계산 알고리즘, 특히 유전자 알고리즘을 통해 강건한 출력을 나타냅니다. 이 연구는 미래의 실제 적용을 고려하여 소프트 로봇 디자인을 위한 효율적인 통찰력을 제공합니다.



### Low-Rank Adaptation for Foundation Models: A Comprehensive Review (https://arxiv.org/abs/2501.00365)
- **What's New**: 이 논문은 Low-Rank Adaptation(LoRA) 기술을 포괄적으로 리뷰한 최초의 연구로, 대형 언어 모델을 넘어 여러 유형의 기초 모델에 대한 분석을 제공합니다. LoRA는 기초 모델의 세부 작업에 대한 적응을 효율적으로 수행하는 방법으로, 추가적인 계산 비용을 최소화하는 파라미터 효율적인 조정 방법을 제시하고 있습니다. 이 조사에서는 저자들이 LoRA의 최신 기술 발전 및 응용 분야를 심층적으로 다루며, 향후 연구 방향과 주요 도전 과제를 논의합니다.

- **Technical Details**: LoRA의 수학적 공식화는 미세 조정 시 업데이트 매트릭스인 ΔW를 저차원으로 제한하는 데 중점을 둡니다. 이를 통해 학습해야 하는 파라미터 수를 최소화하고, 계산 및 저장 효율성을 크게 향상시킵니다. LoRA는 특정 초기화 전략을 사용하여 안정적인 학습을 도모하고, 각 작업에 대해 저차원 매트릭스를 최적화함으로써 원래 모델 파라미터를 고정한 상태에서 효율적인 적응을 이룰 수 있습니다.

- **Performance Highlights**: LoRA는 전체 파라미터의 업데이트 없이도 저차원 매트릭스만을 최적화함으로써 훈련 효율성과 비용을 크게 줄일 수 있습니다. 이 접근 방식은 다수의 적응이 요구되는 멀티태스킹 상황에서 특히 유리하며, 저항력이 큰 지식 유지를 통해 '재앙적 망각' 문제를 해결하는데 기여합니다. 또한, LoRA는 느린 지연을 초래하지 않으므로 본 모델의 배포 및 추론 효율을 보장합니다.



### RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions (https://arxiv.org/abs/2501.00353)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 접근 방식인 RAG-Instruct를 제안합니다. 기존 RAG 방법은 제한된 시나리오와 데이터셋으로 인해 작용 범위가 적었으나, RAG-Instruct는 다양한 고품질 RAG 지침 데이터를 생성할 수 있는 일반적인 방법을 제공합니다. 이를 통해 RAG의 다양한 사용 사례를 포괄적으로 다룰 수 있는 40K의 지침 데이터셋을 구축했습니다.

- **Technical Details**: RAG-Instruct는 다섯 가지 RAG 패러다임을 활용하여 다양한 질의-문서 관계를 모색합니다. 또한, 기존 지침 데이터셋의 강점을 활용해 지침 시뮬레이션을 통해 지침의 다양성과 품질을 높입니다. 이 방법은 Wikipedia에서 만들어진 데이터셋을 기반으로 하여 RAG 시나리오 및 작업을 포괄적으로 커버합니다.

- **Performance Highlights**: 실험 결과, RAG-Instruct는 LLMs의 RAG 능력을 효과적으로 향상시켰으며, 제로샷 성능에서 강력한 결과를 보였습니다. 다양한 RAG 기준선과 비교하여 상당히 뛰어난 성능을 보였으며, 이는 다양한 작업 세트를 통해 나타났습니다. RAG-Instruct에 대한 모든 정보는 공개된 링크를 통해 확인할 수 있습니다.



### Temporal Information Reconstruction and Non-Aligned Residual in Spiking Neural Networks for Speech Classification (https://arxiv.org/abs/2501.00348)
Comments:
          9 pages, 5 figures

- **What's New**: 본 논문에서는 시간적 스케일 다양성(output) 문제를 해결하기 위해 새로운 Temporal Reconstruction (TR) 방법을 제안했습니다. 이를 통해 신경망이 여러 시간 해상도의 입력 정보를 학습하도록 하여 음성 데이터에서 더 포괄적인 의미 정보를 모델링할 수 있습니다. 또한, 서로 다른 시간 길이를 가진 음성 데이터에 대해 residual connection을 사용할 수 있도록 Non-Aligned Residual (NAR) 방법을 제안하였습니다.

- **Technical Details**: 주요 기술적 세부사항으로는 Temporal Reconstruction (TR)을 통해 오디오 스펙트럼의 시간적 차원을 재구성하고, 신경망이 입력 오디오 스펙트로그램의 시간 해상도에서 정보를 학습할 수 있도록 합니다. Non-Aligned Residual (NAR) 방법은 시간 길이가 다른 두 오디오 데이터 간의 잔여 연결을 가능하게 하여, 훈련 프로세스를 최적화할 수 있도록 합니다. 이러한 방법들은 스파이킹 신경망(stimulant neural networks)인 SNN 모델에서 효과적으로 구현됩니다.

- **Performance Highlights**: 제안된 방법을 사용하여 Spiking Speech Commands (SSC) 데이터셋에서는 SOTA(Classification Accuracy) 81.02%를 달성하였고, Spiking Heidelberg Digits (SHD) 데이터셋에서는 96.04%의 분류 정확도를 기록하였습니다. 또한, 비스파이킹 데이터셋인 Google Speech Commands v0.02 (GSC)에서도 우수한 에너지 효율 비율을 달성하여 제안된 방식의 효과성을 검증하였습니다.



### CNC: Cross-modal Normality Constraint for Unsupervised Multi-class Anomaly Detection (https://arxiv.org/abs/2501.00346)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문은 기존의 비지도(distillation-based) 방법이 인코딩된 특징과 디코딩된 특징의 차이를 이용하여 테스트 이미지에서 비정상 영역을 찾는 방법의 한계를 다룹니다. 저자들은 디코더가 정상 샘플만을 학습했음에도 불구하고 비정상 패치의 특징을 잘 복원하는 문제를 제기하며, 이를 'over-generalization'(OG)이라고 명명합니다. 이를 해결하기 위해 클래스 불문 학습 가능한 프롬프트(class-agnostic learnable prompts)를 도입하여 다양한 시각적 패턴 간의 공통적인 정상성을 잡아내고, 이를 통해 비정상 패턴에 대한 과도한 일반화를 억제하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 논문에서는 비정상 패턴의 과도한 일반화 문제를 해결하기 위해 Cross-modal Normality Constraint (CNC)라는 방법을 도입합니다. CNC는 비정상 패턴을 복원하는 것을 방지하기 위해 인코딩된 시각적 특징에서 공통적인 정상성을 추출하는 클래스 불문 프롬프트를 사용합니다. 또한, 다양한 패치 패턴을 처리하기 위해 여러 전문가 네트워크를 구축하는 gated mixture-of-experts (MoE) 모듈도 제안되어, 다중 클래스 훈련에서 상호 간섭을 줄이도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 MVTec AD 및 VisA 데이터셋에서 경쟁력 있는 성능을 성취하며, 비지도 다중 클래스 이상 탐지 분야의 성능을 강화합니다. CNC와 MoE를 통합하여, 기존의 단일 모델 기반 접근방식보다 향상된 성과를 기록하였으며, 실험 결과에서 상당한 성능 개선을 보였습니다. 본 연구는 다중 클래스 훈련에서 비정상 탐지를 위한 새로운 방법론을 제공하며 향후 연구 방향에도 기여할 것으로 기대됩니다.



### Chunk-Distilled Language Modeling (https://arxiv.org/abs/2501.00343)
- **What's New**: CD-LM(Chunk-Distilled Language Modeling)은 현재의 대규모 언어 모델들이 겪고 있는 두 가지 주요 문제인 토큰 레벨 생성의 비효율성과 새로운 데이터 및 지식으로 적응하기 어려움을 처리하는 새로운 접근 방식을 제시합니다. 이 방법은 심층 네트워크 기반의 언어 모델과 간단한 검색 모듈을 결합하여 한 번의 디코딩 단계에서 여러 토큰의 텍스트 청크를 생성할 수 있게 합니다. CD-LM은 확인된 청크를 활용하여 효율성을 높이고, 새로운 영역에 적응할 수 있는 유연성을 제공합니다.

- **Technical Details**: CD-LM은 다양한 크기의 텍스트 청크를 저장하고, 현재 생성 과정에 따라 가장 가능성이 높은 청크를 검색하기 위해 트라이(trie) 구조의 데이터 저장소를 사용합니다. 청크는 일반적인 사전 훈련된 언어 모델에서 발생할 수 있으며, 이는 메모리에 저장된 높은 확률의 시퀀스로서 작용할 수 있습니다. 이 과정에서는 별도의 임베딩 모듈을 사용하는 추가적인 오버헤드 없이, 언어 모델이 자체적으로 유도한 벡터 표현 공간에서 맥락 매칭이 수행됩니다.

- **Performance Highlights**: CD-LM은 언어 모델링 난이도, 텍스트 생성 및 도메인 적응 등에 대한 다양한 경험적 연구를 통해 검증되었습니다. 본 방법은 인퍼런스 효율성과 모델 성능을 동시에 향상시킬 수 있는 능력을 보여줍니다. 결과적으로 CD-LM은 기존의 사전 훈련된 모델의 분포에 새로운 청크를 삽입하여 추가 훈련 없이도 성능을 개선시킬 수 있습니다.



### Loss-Aware Curriculum Learning for Chinese Grammatical Error Correction (https://arxiv.org/abs/2501.00334)
Comments:
          ICASSP 2025

- **What's New**: 본 논문에서는 중국어 문법 오류 수정(CGEC) 작업에서 다양한 수정 난이도를 고려하지 않고 데이터를 처리하는 기존 접근 방식을 개선하기 위한 다중 레벨 커리큘럼 학습(Multi-granularity Curriculum Learning, CL) 프레임워크를 제안합니다. 이 프레임워크는 오류 수정이 어려운 샘플과 쉬운 샘플의 난이도를 평가하고, 이를 바탕으로 모델이 학습하도록 샘플의 배치를 조정합니다. 기존의 PLM 기반 모델 성능을 개선하려는 이러한 노력은 CGEC에 관한 최근 연구에서 중요한 방향성을 제시하고 있습니다.

- **Technical Details**: CGEC 작업의 정의와 함께, 제안한 커리큘럼 학습 프레임워크는 크게 두 가지 하위 모듈로 구성됩니다. 첫째, 교차 엔트로피 손실 함수를 통해 각 데이터 샘플의 난이도를 평가하고, 낮은 손실을 기록하는 단순 샘플에서부터 높은 손실을 기록하는 복잡한 샘플로 진행하는 배치 레벨 학습이 이루어집니다. 둘째, 샘플의 난이도를 수치적으로 측정하여, 이 난이도에 따라 훈련 과정에서 샘플의 로딩 순서를 조정합니다.

- **Performance Highlights**: 실험 결과, mT5 및 BART와 같은 사전 훈련된 언어 모델을 활용한 결과, 제안된 커리큘럼 학습 방법이 NLPCC 및 MuCGEC 데이터셋에서 기존의 성능을 크게 초과하는 것으로 밝혀졌습니다. 이는 CGEC 모델이 더 어려운 샘플을 효과적으로 학습할 수 있도록 돕는 방안으로, 실제 문법 수정에서의 성능 향상에 기여함을 보여줍니다. 이러한 실험은 CGEC 모델의 성능 개선 가능성을 명확하게 입증하고 있으며, 향후 연구에 중요한 기초 자료가 될 것입니다.



### Exploring the Implicit Semantic Ability of Multimodal Large Language Models: A Pilot Study on Entity Set Expansion (https://arxiv.org/abs/2501.00330)
Comments:
          ICASSP 2025

- **What's New**: 본 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 다중 모달 엔티티 세트 확장(Multi-modal Entity Set Expansion, MESE) 작업에 적용하여 기존의 한계점을 보완하고, 이 도구의 능력을 탐색합니다. 새로운 엔티티를 시맨틱 클래스에 맞춰 탐색하는 MESE 작업을 통해 MLLM의 암묵적 의미 정보 추출 능력을 평가합니다. 특히, LUSAR라는 리스트 순위 매김 방법을 도입하여 지역 점수를 글로벌 순위로 매핑하는 방식을 제안합니다.

- **Technical Details**: LUSAR 방법론은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 접두사 트리 제약을 사용하여 모델이 지정된 엔티티 데이터 세트 내의 많은 후보 엔티티를 생성하도록 제한합니다. 두 번째 단계에서는 리스트별 접근 방식을 도입하여 각 후보 엔티티에 대한 순위 점수를 얻기 위해 여러 샘플링 및 순위를 수행합니다. 이러한 접근은 암묵적인 정보에서 세부적으로 구분된 의미 기능을 추출하는 데 도움을 줍니다.

- **Performance Highlights**: LUSAR 방식의 적용을 통해 MLLM의 MESE 작업에서 성능 향상이 크게 이루어졌습니다. 이 방식은 대형 모델이 암묵적인 의미 정보를 추출하는 데 어려워하는 문제를 해결하고, 성공적인 후보 엔티티 선정에 기여합니다. 경량의 리스트 방식 접근을 통하여 추천 시스템과 같은 다른 작업에도 적용 가능성을 보여주며, 실험 결과로 MLLM의 성능이 크게 향상된 것을 확인하였습니다.



### OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning (https://arxiv.org/abs/2501.00321)
- **What's New**: 최근 LMMs(대형 멀티모달 모델)의 OCR(광학 문자 인식) 능력을 평가하는 데 대한 관심이 커지고 있습니다. 기존 벤치마크는 LMMs의 텍스트 인식 성능이 인상적임을 보여주지만, 텍스트 위치 파악, 손글씨 내용 추출 및 논리적 추론과 같은 특정 도전 과제에 대한 성능은 아직 충분히 탐구되지 않았습니다. 이를 보완하기 위해 OCRBench v2를 소개하며, 이전의 멀티-장면 벤치마크인 OCRBench보다 4배 더 많은 과제를 포함하고 있습니다.

- **Technical Details**: OCRBench v2는 현재 가장 포괄적인 bilingual (이중 언어) 텍스트 중심 벤치마크로, 31개의 다양한 시나리오(거리 장면, 영수증, 수식, 다이어그램 등)를 포함하고 있습니다. 이는 인적 검증을 받은 10,000개의 질문-답변 쌍과 높은 비율의 어려운 샘플로 구성되어 있으며, 철저한 평가 지표를 제공합니다. 최신 LMMs를 OCRBench v2에서 신중하게 벤치마킹한 결과, 22개 LMM 중 20개가 100점 만점에 50점 이하를 기록했습니다.

- **Performance Highlights**: LMM은 텍스트 인식과 해석의 능력을 비교할 때 다섯 가지 제한 사항이 있음을 발견했습니다. 이러한 제한 사항은 자주 접하지 않는 텍스트 인식, 세분화된 인식, 레이아웃 인식, 복잡한 요소 구문 분석, 논리적 추론을 포함합니다. LMM의 성능 평가에 있어 OCR은 중요한 구성이 되었으며, 자주 사용되는 VQA(시각 질문 답변) 데이터셋을 활용하여 LMM의 텍스트 인식 능력을 평가하고 있습니다.



### M2I2: Learning Efficient Multi-Agent Communication via Masked State Modeling and Intention Inferenc (https://arxiv.org/abs/2501.00312)
- **What's New**: 이번 논문에서는 다중 에이전트 간의 효과적인 커뮤니케이션을 위한 새로운 프레임워크인 M2I2를 소개합니다. M2I2는 에이전트가 받은 정보를 통합하고 활용하는 능력을 향상시켜, 복잡하고 불확실한 상호작용에 대한 이해와 대응 능력을 높입니다. 이에 따라 M2I2는 다차원 합리적 네트워크(Dimensional Rational Network)를 통해 정보의 중요성을 점검하고, 선택적 정보 마스킹 및 공유를 위한 중요 기반 휴리스틱을 제안합니다.

- **Technical Details**: M2I2는 마스킹된 상태 모델링(masked state modeling)과 공동 행동 예측(joint-action prediction)을 위한 발전된 기능을 제공하여 에이전트가 환경의 불확실성을 이해하고 동료의 의도를 예측하도록 돕습니다. 이 프레임워크는 자가 감독(auxiliary tasks)에 기반한 두 가지 보조 작업을 통해 정보 통합의 효율성을 높이고, 마스킹된 오토 인코더(Masked Auto-Encoder)를 사용하여 수신된 메시지로부터 전체 상태를 재구성하는 기술을 사용합니다. 또한, 대응하는 관찰의 각 차원의 중요성을 조정하는 차원 합리적 네트워크(DRN)를 메타 학습(meta-learning) 패러다임을 통해 훈련합니다.

- **Performance Highlights**: M2I2는 다양한 복잡한 다중 에이전트 작업에서 강력한 성능을 입증하였으며, 기존의 최첨단 방법들과 비교할 때 효율성과 일반화 능력 면에서 우수함을 나타냈습니다. 특히, Hallway, MPE, SMAC와 같은 다양한 테스트 환경에서 M2I2는 의사 결정 및 자가 감독 목표에 기여하는 정보 차원의 기여도를 모델링하여 에이전트가 중요한 정보에 집중하게 합니다. 이러한 특성 덕분에 M2I2는 다중 에이전트 강화 학습(MARL) 분야의 중요한 연구 격차를 해소하고 효율적인 메시지 통합을 촉진합니다.



### Fast and Interpretable Mixed-Integer Linear Program Solving by Learning Model Reduction (https://arxiv.org/abs/2501.00307)
- **What's New**: 이 논문은 Mixed-Integer Linear Programming (MILP) 솔버의 성능을 향상시키기 위해 새로운 접근법을 제안합니다. 기존의 ML 기반 MILP 솔버는 고차원 솔루션 공간의 확장성 문제로 어려움을 겪고 있었지만, 새로 제안된 방법은 최적 솔루션을 직접 학습하는 대신, 원래 MILP 모델의 축소 및 동등 모델을 학습하는 방법을 사용합니다. 이를 통해 대규모 MILP 문제를 더 간단하고 빠르게 해결할 수 있습니다.

- **Technical Details**: 제안된 방법은 preference 기반 모델 축소 학습(preference-based model reduction learning) 기술을 사용하여 MILP 인스턴스의 모든 축소 모델에 대한 상대적인 성능을 고려합니다. 이 과정에서 attention 메커니즘을 도입하여 preference 정보를 포착하고 표현하는 방법을 사용합니다. 또한 SetCover 기반의 가지치기(pruning) 방법을 도입하여 축소 모델의 수를 제어함으로써 학습 과정을 단순화합니다.

- **Performance Highlights**: 실세계 MILP 문제에서 평가한 결과 제안된 방법은 기존의 최첨단 모델 축소 ML 방법보다 솔루션 정확도가 거의 20% 향상된 것으로 나타났습니다. 또한 상업용 솔버인 Gurobi와 비교할 때 2에서 4배의 속도 향상을 기록했습니다. 이러한 성장은 대규모 MILP 문제에 대한 적시 해결 능력을 크게 향상시킵니다.



### Enhancing Deployment-Time Predictive Model Robustness for Code Analysis and Optimization (https://arxiv.org/abs/2501.00298)
- **What's New**: 이번 논문에서는 Prom이라는 오픈소스 라이브러리를 소개하여, 배포 중 데이터 드리프트(data drift)로 인한 머신러닝 모델의 신뢰성을 향상시키는 방법을 제안합니다. Prom은 사전 훈련된 모델의 오류 예측을 식별하고, 이러한 오류를 피드백으로 활용하여 모델을 개선하는 기능을 수행합니다. 이 기법을 통해 Prom은 코드 분석 및 최적화에 사용되는 13가지 머신러닝 모델에서 우수한 성능을 보이며, 평균 96%의 오예측을 성공적으로 식별할 수 있음을 입증합니다.

- **Technical Details**: Prom은 배포 후 머신러닝 모델의 신뢰도를 향상시키기 위해 credibility와 confidence 점수를 계산합니다. 이 점수들은 모델 예측의 신뢰성과 확신 정도를 측정하며, 두 점수가 모두 높은 경우에만 예측 결과를 수락하는 방식을 취합니다. Prom은 비슷한 특성을 가진 샘플을 선택해 비정합성(nonconformity) 점수를 계산하고, 여러 통계 함수와 다수결 투표 방식을 통해 예측을 승인 또는 거부합니다.

- **Performance Highlights**: Prom은 총 13개의 대표적인 머신러닝 모델을 통해 코드 분석 및 최적화 작업에서의 성능을 평가했으며, 최대 100%에 달하는 미예측 식별률을 보여주었습니다. 또한, Prom이 식별한 샘플의 5%를 재라벨링하여 모델 훈련 단계에서의 성능을 재현할 수 있음을 확인했습니다. 이러한 결과는 Prom의 사용이 머신러닝 모델의 안정성과 효율성을 높이는 데 기여할 수 있음을 시사합니다.



### Predicate Invention from Pixels via Pretrained Vision-Language Models (https://arxiv.org/abs/2501.00296)
Comments:
          Workshop on Planning in the Era of LLMs (LM4Plan @ AAAI 2025)

- **What's New**: 이번 연구에서는 기존의 모델로부터 부족한 샘플만으로도 이미지의 원시 센서 입력에서 직접 작동하는 predicates를 발명하는 새로운 접근 방식을 제안합니다. 최근의 비전-언어 모델(Vision-Language Models, VLMs)의 능력을 활용하여, VLMs가 제안할 수 있는 predicates를 통해 로봇의 의사결정 문제를 해결하는 방법을 모색하고 있습니다. 이 방법은 본 연구에서 소개되는 pix2pred라는 프레임워크를 통해 실험적으로 입증되었습니다.

- **Technical Details**: 연구에서는 PDDL(Planning Domain Definition Language) 모델을 학습하여 개념을 추상화하며, 특히 Sparse한 데이터에서 추상적 세계 모델을 학습합니다. VLM을 활용하여 기초적인 predicates를 발명하고, 이들을 통해 실제 문제에 대한 의사결정을 지원합니다. 이 과정에는 기존의 predicates와 동작들을 조합하여 효과적인 의사결정을 위한 최적화를 수행하는 방법이 포함됩니다.

- **Performance Highlights**: pix2pred 접근법은 시뮬레이션된 두 가지 로봇 환경에서 네 가지 작업을 수행하며, 새로운 문제 인스턴스에서 항상 가장 높은 성공률을 기록했습니다. 또한, 더 많은 객체와 복잡한 목표, 그리고 훈련 중 보여준 것보다 긴 작업 수순을 포함하는 새로운 문제에 대해 효과적으로 일반화하는 기초 모델을 학습할 수 있음을 보여줍니다. 실험을 통해 제안된 방법이 모델-프리 모방 학습보다 더 높은 성과를 내는 것을 확인했습니다.



### Dual Diffusion for Unified Image Generation and Understanding (https://arxiv.org/abs/2501.00289)
- **What's New**: 이 논문은 멀티모달(multi-modal) 이해 및 생성을 위한 대규모의 완전 엔드 투 엔드(diffusion model)를 제안합니다. 기존의 diffusion 기반 멀티모달 모델들보다 성능이 크게 향상된 이 모델은 비전-언어 모델링의 전체 기능을 지원하는 최초의 사례로, 이미지와 텍스트의 조건부 가능성을 동시에 훈련하는 새로운 기법을 도입했습니다. 새로운 손실 함수는 이미지와 텍스트 자료를 통합적으로 학습하여 다양한 작업을 수행할 수 있게 돕습니다.

- **Technical Details**: 제안된 모델은 멀티모달 확산 변환기(multimodal diffusion transformer) 구조를 기반으로 하며, 이미지 및 텍스트 모달리티에 대해 연속 및 이산 그래디언트를 활용한 확산을 수행합니다. 이 모델은 이미지 생성, 캡셔닝(captioning), 시각적 질문 응답(visual question answering) 등을 수행할 수 있는 유연성을 지니고 있습니다. 또한, 기존의 diffusion 기반 모델과의 호환성을 통해 빠른 적응이 가능합니다.

- **Performance Highlights**: 제안된 모델은 최근의 통합 이미지 이해 및 생성 모델과 경쟁할 수 있는 성능을 보여주었으며, 다양한 멀티모달 작업에서 현저한 성능 향상을 이루었습니다. 이는 기존의 diffusion 모델들이 갖추지 못했던 다양한 기능을 가능하게 하여, 인공지능의 다음 단계로 나아가는 건설적인 기여를 제공합니다. 이런 방식으로 멀티모달 확산 모델링은 이미지와 텍스트의 관계를 더욱 깊이 있게 탐구할 수 있는 기회를 제공합니다.



### Efficient Human-in-the-Loop Active Learning: A Novel Framework for Data Labeling in AI Systems (https://arxiv.org/abs/2501.00277)
- **What's New**: 이 논문에서는 기존의 전통적 액티브 러닝(active learning) 방법과는 차별화되는 새로운 액티브 러닝 프레임워크를 제안합니다. 이 프레임워크는 데이터 포인트의 라벨링 방법뿐만 아니라 다양한 질의 방식(query scheme)을 통합하여, 의사결정 과정에서 활용할 수 있는 새로운 접근 방식을 제시합니다. 특히, 여러 질의 방법에서 얻은 정보를 통합하여 다음에 물어볼 질문을 자동으로 결정하는 모델을 개발하였습니다.

- **Technical Details**: 이 연구에서는 불확실성 기반 메서드를 중심으로 새로운 액티브 러닝 방법을 제안합니다. 즉, 불확실성이 가장 큰 데이터가 정보를 가장 많이 포함하고 있다고 보고, 이를 통해 기계가 클래스의 경계(boundary)를 신속하게 식별할 수 있도록 합니다. 또한, 이 방법은 실험 주기를 거쳐 중복된 정보를 포함할 가능성이 있는 데이터를 자동으로 제거하여 각 액티브 러닝 반복(iteration) 전에 계산하는 데이터-구동 탐색 및 활용(exploration and exploitation)의 프레임워크를 제공합니다.

- **Performance Highlights**: 실제 세계의 다섯 개 데이터 세트에 대한 시뮬레이션을 통해 제안된 액티브 러닝 프레임워크는 다른 방법들에 비해 높은 정확도와 낮은 손실(loss)을 기록하였습니다. 또한, 이 프레임워크는 다양한 액티브 러닝 알고리즘에 통합될 수 있어 적용 가능성이 높습니다. 결과적으로, 제안된 팀은 라벨링의 효율성을 높이는 동시에 머신 러닝의 효율성을 향상시키는 데 이바지할 수 있습니다.



### Enhancing Wireless Sensor Network Security through Integration with the ServiceNow Cloud Platform (https://arxiv.org/abs/2501.00264)
Comments:
          17 pages, 2 figures

- **What's New**: 이 연구는 Wireless Sensor Networks (WSNs)와 클라우드 플랫폼의 통합을 통해 보안 위험을 해결하는 방법을 탐구합니다. WSN은 저전력 응용 프로그램에 사용되며, 이러한 해로운 데이터 공격으로부터 WSN을 보호하기 위해 클라우드 컴퓨팅 활용 방안을 모색합니다. 기존의 WSN 보안 절차의 제약을 극복하기 위한 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: WSNs는 센서 또는 노드를 통해 데이터를 수집하고 처리하며, 다양한 네트워크와 통신하여 정보 관리를 향상시킵니다. WSN은 전통적인 네트워크와 달리 다수의 무선 연결 센서를 통해 할당된 작업을 수행합니다. 그러나 이들은 전통적인 보안 절차의 한계로 인해 보안 문제가 심각합니다.

- **Performance Highlights**: WSNs는 재난 관리, 국가 안보, 전투 감시, 농업 및 의료 산업 등 다양한 저전력 응용 프로그램에서 배치되고 있습니다. 이 연구의 시험 시나리오는 WSN의 크기가 몇 개에서 수천 개까지 다양하다는 점을 보여주며, 신속한 데이터 처리, 저비용 설치, 낮은 지연 시간과 같은 요구 사항을 강조하고 있습니다.



### Collaborative Approaches to Enhancing Smart Vehicle Cybersecurity by AI-Driven Threat Detection (https://arxiv.org/abs/2501.00261)
Comments:
          7 Pages

- **What's New**: 이번 논문에서는 자동차 산업에서의 연결된 자동화 차량(Connected and Automated Vehicles, CAV) 사이버 보안을 강화하기 위한 협력적 접근 방법에 대해 다룹니다. AI 기반 위협 탐지(threat detection) 기술을 통해 새로운 취약점과 보안 요구 사항을 해결할 수 있는 가능성을 제시합니다.

- **Technical Details**: 논문은 5G 네트워크, 블록체인(blockchain), 양자 컴퓨팅(quantum computing)과 같은 최신 기술의 통합이 CAV 사이버 보안 강화에 기여할 수 있음을 설명합니다. 또한, 자율주행 차량의 사이버 보안 로드맵에서는 효율적인 침입 탐지 시스템(intrusion detection systems) 및 AI 기술의 중요성을 강조합니다.

- **Performance Highlights**: 향상된 사이버 보안을 위해 안전한 하드웨어와 소프트웨어 스택(stack) 그리고 고급 위협 인텔리전스(threat intelligence)의 통합이 생명선 역할을 함을 보여줍니다. 이러한 기술적 접근은 향후 자율주행 차량의 사이버 보안 해결책을 향상시키는데 필수적입니다.



### Exploring Variability in Fine-Tuned Models for Text Classification with DistilBER (https://arxiv.org/abs/2501.00241)
- **What's New**: 이번 연구는 DistilBERT 모델을 활용한 텍스트 분류를 위한 파인 튜닝(파인튜닝, fine-tuning) 전략을 평가하며, 하이퍼파라미터에 따른 성능 변화에 대한 실험을 통해 얻은 통찰력을 제공합니다. 특히 학습 속도(learning rate), 배치 크기(batch size), 에폭(epochs)이 정확도(accuracy), F1 점수(F1-score), 손실(loss)에 미치는 영향을 분석합니다.

- **Technical Details**: 우리는 폴리노미얼 회귀(polyomial regression) 분석을 통해 하이퍼파라미터의 내재적 및 점진적 영향을 포착하고, 기본 모델에 대한 상대적 조정의 중요성을 강조합니다. 주요 하이퍼파라미터 간의 상호작용이 F1 점수를 극대화함을 보여주며, 하이퍼파라미터 간 상호작용의 중요성을 강조합니다.

- **Performance Highlights**: 하이퍼파라미터 조정에 따른 성능 지표의 변동성이 나타났으며, 예를 들어 높은 학습 속도는 손실을 줄이지만 정확도 개선에 도전이 될 수 있습니다. 배치 크기는 정확도와 F1 점수에 유의미한 영향을 미치지만 손실 최적화에는 제한적인 영향을 미칩니다. 이러한 결과는 텍스트 분류 외에도 자연어 처리(NLP)와 컴퓨터 비전(CV) 등 다양한 작업에 중요성이 있음을 시사합니다.



### Federated Deep Subspace Clustering (https://arxiv.org/abs/2501.00230)
Comments:
          8pages,4 figures, 4 Tables

- **What's New**: 이 논문에서는 연합 학습(federated learning, FL) 구조를 기반으로 한 개인 정보 보호가 가능한 부분 공간 클러스터링(subspace clustering, SC) 방법인 FDSC를 소개합니다. FDSC는 각 클라이언트에서 고립된 데이터를 그룹화하는 심층 부분 공간 클러스터링 네트워크를 사용하며, 이 네트워크는 인코딩 네트워크, 자기 표현층(self-expressive layer), 디코딩 네트워크로 구성됩니다. 각 클라이언트 간의 통신을 위해 인코딩 네트워크를 서버에 업로드함으로써 FDSC를 구현합니다.

- **Technical Details**: FDSC의 핵심 구성 요소는 공유된 인코더와 각 클라이언트 내의 지역 디코더로, 공유된 인코더는 클라이언트 간의 데이터 특성 추출을 향상시킵니다. 자기 표현층은 스펙트럼 클러스터링을 위한 친화성 행렬(affinity matrix) 구성을 담당하며, 지역 구조를 유지하기 위해 인접 그래프 정보(adjacency graph information)를 네트워크에 추가합니다. 이러한 방식으로 지역 친화성 행렬과 인접 행렬(adjacency matrix)을 정렬하여 전국 판별 기능을 강화합니다.

- **Performance Highlights**: FDSC는 네 개의 이미지 데이터셋에서 클러스터링 성능을 평가한 결과, 최신 방법들(state-of-the-art methods)보다 우수한 결과를 보였습니다. 이러한 성과는 FDSC가 데이터의 지역적 특성을 잘 유지하면서 효과적으로 클러스터링을 수행할 수 있음을 증명합니다. 본 연구는 개인 데이터의 분산 특성을 고려하면서도 효율적인 클러스터링 결과를 도출하는 데 중요한 기여를 합니다.



### Extracting effective solutions hidden in large language models via generated comprehensive specialists: case studies in developing electronic devices (https://arxiv.org/abs/2501.00224)
Comments:
          18 pages, 4 figures

- **What's New**: 최근 대형 언어 모델(LLMs)을 활용하여 연구 아이디어와 과학 가설 생성을 탐구하는 연구가 증가하고 있습니다. 하지만 실제 연구 및 개발에서는 복잡하고 학제 간의 도전 과제를 해결할 필요가 있으며, 기존 지식을 통해 쉽게 발견할 수 없는 해결책이 필요합니다. 이에 따라, 다양한 분야의 관점을 통합하여 LLM의 방대한 지식을 활용하여 효과적이고 혁신적인 해결책을 생성하는 접근이 요구되고 있습니다.

- **Technical Details**: 이 논문에서는 MECE(상호 배타적, 포괄적) 원칙을 사용하여 LLM과 구조적인 지침을 결합한 SELLM(해결책 열거 프레임워크)을 제안합니다. SELLM은 국제 특허 분류(IPC) 및 주기율표와 같은 구조적 지침을 활용하여 전문가 에이전트를 체계적으로 구성하고, 이를 통해 학제간의 효과적인 해결책을 생성합니다. 이것은 다양한 지식 영역을 아우르는 접근 방식으로, 복잡한 문제 해결에 대한 통합적인 해법을 제공합니다.

- **Performance Highlights**: SELLM의 실용성을 평가하기 위해 두 가지 도전 과제에 적용하였습니다: 유기 발광 다이오드(OLED) 조명의 빛 추출 개선 및 차세대 메모리 소재용 전극 개발입니다. 결과적으로, SELLM은 특정 맞춤화나 노력이 없는 경우와 비교하여 효과적인 해결책 생성을 significantly하게 촉진함을 보여주었습니다. 이 연구는 SELLM이 LLMs를 통해 어려운 문제에 대해서도 효과적인 해결책을 생성할 수 있는 잠재력을 지니고 있음을 입증합니다.



### The Potential of LLMs in Automating Software Testing: From Generation to Reporting (https://arxiv.org/abs/2501.00217)
Comments:
          6 pages, 3 figures, 1 table

- **What's New**: 이 연구는 LLMs(대형 언어 모델)를 활용한 자동화된 소프트웨어 테스트를 위한 에이전트 기반 접근 방식을 제안합니다. 기존 방법론과 달리, LLM이 동적으로 테스트를 생성하여 수동 입력을 크게 줄이고 테스트 케이스 생성 및 실행에 소요되는 시간을 최소화합니다. 이 프레임워크는 소프트웨어 테스트 프로세스의 효율성을 높이고 스케일러빌리티(scalability) 및 정확도 문제를 해결하려고 합니다.

- **Technical Details**: 제안된 아키텍처는 네 가지 주요 컴포넌트로 구성되어 있습니다. Audio WebClient는 사용자 입력을 수집하고, Software Testing Agent는 시스템의 핵심으로 기능하여 테스트 스크립트 생성 및 실행을 담당합니다. LLM은 사용자 명령에서 키 엔티티를 추출하고 맞춤형 유닛 테스트를 생성하는 데 사용되며, Development Environment는 테스트할 프로젝트 코드를 포함하여 자동화를 지원합니다.

- **Performance Highlights**: 연구 결과, 제안된 시스템은 Python 및 Java의 여러 응용 프로그램에서 높은 테스트 커버리지(test coverage)와 효율적인 작업 성능을 보였습니다. LLM이 각 테스트 케이스와 엣지 시나리오를 자동으로 생성하고, 테스트 간접 결과를 제공하여 프로세스를 능동적이고 지능적으로 변화시킴으로써, 소프트웨어 품질 보증 메커니즘을 강화할 수 있음을 증명했습니다.



### Debunking the CUDA Myth Towards GPU-based AI Systems (https://arxiv.org/abs/2501.00210)
Comments:
          Under Review

- **What's New**: 이 논문은 Intel의 Gaudi NPU가 NVIDIA의 GPU에 대한 대안으로서의 가능성을 평가한 연구 결과를 발표합니다. Intel Gaudi-2는 저급 AI 연산, 메모리, 통신 작업에서의 경쟁력 있는 성능을 보여주며, 여러 중요한 AI 워크로드를 엔드 투 엔드로 실행하는데 있어서도 유사한 성능을 발휘합니다. 또한, 소프트웨어 최적화 전략에 대한 논의를 통해 Gaudi의 프로그래머빌리티를 평가하여 GPU 최적화 모델과 비교한 결과를 제시하고 있습니다.

- **Technical Details**: Gaudi NPU는 TPC-C라는 네이티브 프로그래밍 언어를 가지고 있으며, 이는 NVIDIA의 CUDA에 해당합니다. 이 아키텍처는 메모리 시스템과 HBM2E 메모리 서브시스템을 사용하여 NVIDIA A100과 비교 가능한 성능을 제공합니다. 연구에서 Gaudi-2는 일반적인 AI 작업에 대해 매우 경쟁력 있는 성능을 나타내었으나, 세부 데이터 접근 방식 및 소수 프로세서간의 집단 통신에서 A100에 비해 뒤처지는 경우도 있었습니다.

- **Performance Highlights**: Gaudi-2는 추천 시스템(RecSys) 및 대형 언어 모델(LLM)과 같은 엔드 투 엔드 AI 애플리케이션에서 A100보다 우수한 에너지 효율성을 보여줍니다. 초기 공개된 Gaudi 최적화 소프트웨어는 실망스러운 성능을 보였으나, 다양한 소프트웨어 수준의 최적화를 통해 Gaudi-2는 실제로 A100의 엔드 투 엔드 성능에 가까운 결과를 달성할 수 있음을 보여주었습니다. 이러한 결과는 Gaudi NPU가 NVIDIA GPU에 도전할 가능성을 지니고 있음을 시사합니다.



### An Empirical Evaluation of Large Language Models on Consumer Health Questions (https://arxiv.org/abs/2501.00208)
- **What's New**: 이번 연구는 MedRedQA라는 데이터셋에서 여러 Large Language Models (LLMs)의 성능을 평가합니다. 이 데이터셋은 AskDocs subreddit에서 전문가들이 검증한 소비자 기반의 의료 질문과 답변으로 구성되어 있습니다. LLM들은 임상 질문 응답(QA) 벤치마크에서 두각을 나타냈지만, 실제 소비자 질문에 대한 효과는 상대적으로 덜 이해되었습니다.

- **Technical Details**: 연구에서 사용된 모델은 GPT-4o mini, Llama 3.1: 70B, Mistral-123B, Mistral-7B, 그리고 Gemini-Flash입니다. 각 모델이 자가 평가를 진행하고 다른 모델의 응답도 평가하는 크로스-이밸류에이션(cross-evaluation) 방법이 사용되었습니다. MedRedQA는 비공식적인 언어와 비전문가 질문에 적합한 정확한 응답 필요성이라는 독특한 도전 과제를 제공합니다.

- **Performance Highlights**: 연구 결과, GPT-4o mini가 다섯 개 모델의 심사자 중 네 명의 전문가 응답과 가장 높은 일치를 보였습니다. 반면에 Mistral-7B는 세 모델의 심사자들에게 가장 낮은 점수를 기록했습니다. 이 연구는 소비자 건강 의료 질문 응답에 대한 현재 LLM의 잠재력과 한계를 강조하며, 추가 개발의 방향성을 제시합니다.



### GPT-4 on Clinic Depression Assessment: An LLM-Based Pilot Study (https://arxiv.org/abs/2501.00199)
- **What's New**: 이번 연구는 전 세계적으로 널리 퍼진 우울증을 조기에 탐지하기 위한 혁신적인 접근법으로 GPT-4를 활용한 임상 우울증 평가 방법을 제안합니다. 또한 인터뷰 전사 분석을 통해 AI가 인간 전문가의 진단 능력을 어떻게 모방할 수 있는지를 탐구합니다. 이 연구는 단순한 프롬프트를 넘어 다양한 프롬프트 구조와 온도 조정이 모델의 정확성과 일관성에 미치는 영향을 조사합니다. 이를 통해 정신 건강 진료에 AI의 활용 기준을 새롭게 설정하고자 합니다.

- **Technical Details**: 연구는 인터뷰 전사 데이터를 바탕으로 하여 GPT-4의 분류 능력을 테스트합니다. 실험은 기본 프롬프트를 사용한 이진 분류에서 시작하여 예시를 추가하고 임상적인 맥락을 포함한 복잡한 프롬프트로 접근합니다. 마지막으로 온도 조정을 통해 정확도 및 F1-Score 최적화 효과를 분석합니다. 이러한 프롬프트 공학 과정은 모델의 분류 성능을 향상시키는 핵심 요소로 작용합니다.

- **Performance Highlights**: 연구 결과, GPT-4는 다양한 구성에서의 정확도와 F1-Score에서 상당한 변동성을 보였으며, 복잡한 프롬프트에서 낮은 온도 값(0.0-0.2)에서 최적 성능을 발휘했습니다. 하지만 온도 값이 0.3 이상으로 증가할 경우 성능과 임의성의 관계가 예측 불가능해져서 성과가 감소했습니다. 이러한 결과들은 GPT-4가 임상 진단에 대한 잠재력을 보여주지만, 프롬프트 및 모델 파라미터의 세심한 조정이 필요하다는 것을 시사합니다.



### Towards Unraveling and Improving Generalization in World Models (https://arxiv.org/abs/2501.00195)
Comments:
          An earlier version of this paper was submitted to NeurIPS and received ratings of (7, 6, 6). The reviewers' comments and the original draft are available at OpenReview. This version contains minor modifications based on that submission

- **What's New**: 이번 연구에서는 세계 모델(world models)의 견고성(robustness)과 일반화 능력(generalization capabilities)을 깊이 이해하기 위해 확률적 미분방정식(stochastic differential equation) 방법을 도입했습니다. 저자는 잠재 표현 오류(latent representation errors)의 영향을 분석하고, 제로 드리프트(zero-drift)와 비제로 드리프트(non-zero-drift) 상황에서의 결과를 비교했습니다. 이 연구는 잠재 표현 오류가 임시적 정규화를 통해 견고성을 개선할 수 있음을 보여주는 놀라운 결과를 제시합니다.

- **Technical Details**: 이 연구에서는 세계 모델 학습을 확률적 동적 시스템(stochastic dynamical system)으로 간주하고 수학적으로 모델링했습니다. 이를 통해 제로 드리프트 상황에서의 잠재 오류가 어떻게 임시적 정규화(implicit regularization)로 작용하는지를 밝히며, 비제로 드리프트 상황에서는 하이코딩 규제(Jacobian regularization) 방식을 제안하여 오류 전파의 누적 효과를 완화하고 훈련의 안정성을 높입니다. 이러한 기법은 특히 훈련 과정에서의 오류 축적 문제를 해결하는 데 주요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 규제 방식은 훈련을 안정화할 뿐만 아니라 수렴 속도를 가속화하고 장기 예측(long-horizon prediction)의 정확성을 향상시키는 것을 확인했습니다. 이러한 성과는 reinforcement learning 분야에서 세계 모델의 활용 가능성을 더욱 높이는 중요한 발견으로 평가됩니다. 저자들은 모델의 일반화 능력을 강조하며 향후 연구 방향도 제시하고 있습니다.



### SepsisCalc: Integrating Clinical Calculators into Early Sepsis Prediction via Dynamic Temporal Graph Construction (https://arxiv.org/abs/2501.00190)
- **What's New**: 이번 연구에서는 의사들의 워크플로우를 모방하여 세균혈증(sepsis) 예측 모델인 SepsisCalc를 제안합니다. 이 모델은 임상 계산기(clinical calculators)를 통합하여 기존 AI 모델의 투명성과 신뢰성을 향상시킵니다. 특히, 전자 건강 기록(EHR)을 시간적인 그래프(temporal graphs)로 표현하여 변수가 일부 누락된 상황에서도 효과적으로 예측할 수 있습니다.

- **Technical Details**: SepsisCalc는 동적 이종 그래프(temporal heterogeneous graphs)를 구축하는데, 이 그래프는 관찰된 임상 변수들을 포함합니다. 각 환자의 데이터를 기반으로 SOFA 점수와 같은 임상 계산기를 추정하고 이를 그래프에 통합하여 예측 성능을 높입니다. 그래프 신경망(graph neural network, GNN)을 사용하여 다양한 변수를 결합하고, 세균혈증 위험 및 장기 기능 장애를 예측합니다.

- **Performance Highlights**: MIMIC-III, AmsterdamUMCdb 및 Ohio State University Wexner Medical Center의 실제 데이터셋을 사용한 실험 결과, SepsisCalc는 최신 기법보다 우수한 성능을 보였습니다. 이 모델은 임상 현장에서 사용할 수 있는 시스템으로 발전하여, 의사들이 예측 결과를 이해하고 빠른 대응을 할 수 있도록 돕습니다. 이는 초기 개입을 위한 행동 가능한 임상 결정 지원 도구로서의 가능성을 보여줍니다.



### The Text Classification Pipeline: Starting Shallow going Deeper (https://arxiv.org/abs/2501.00174)
- **What's New**: 이번 논문은 Text Classification (TC)의 전체 파이프라인을 상세히 탐구합니다. 특히 각 구성 요소가 TC 모델의 성능에 미치는 영향에 대해 철저하게 검토하였습니다. 더불어 최신 데이터셋과 텍스트 전처리 기법을 포함한 다양한 기술 혁신을 소개합니다.

- **Technical Details**: 논문은 TC의 다양한 단계—데이터셋, 전처리 기법, 텍스트 표현 방법, 분류 모델, 평가 메트릭, 결과 및 미래 동향 등을 포괄적으로 다룹니다. 각 장에서는 이론과 함께 실험적 평가 및 사례 연구를 제공하여 보다 깊은 이해를 돕습니다. 이러한 기술적 디테일은 TC의 효과적인 구현에 매우 중요합니다.

- **Performance Highlights**: 분류 전략에 대한 비판적 평가와 비교 분석을 통해 독자에게 다양한 접근 방법의 강점과 약점을 인식시킵니다. 이 연구는 단순한 조사를 넘어서 TC 분야에서의 중요한 최근 발견을 조명하며, 향후 연구 방향에 대한 통찰을 제공합니다. 결과적으로, 이 논문은 TC의 전문성과 이해도를 높이는 데 기여하고 있습니다.



### Federated Learning with Workload Reduction through Partial Training of Client Models and Entropy-Based Data Selection (https://arxiv.org/abs/2501.00170)
- **What's New**: 본 논문에서는 Edge 장치에서의 훈련 워크로드를 줄이기 위한 새로운 방법인 FedFT-EDS(Federated Fine-Tuning with Entropy-based Data Selection)를 제안합니다. 이 방법은 일부 클라이언트 모델의 파인 튜닝(Fine-Tuning)과 엔트로피 기반 데이터 선택(Entropy-based Data Selection)을 결합하여 훈련 효율성을 높이는 데 중점을 둡니다. FedFT-EDS는 고객의 훈련 데이터를 적극적으로 선택하여 FL(Federated Learning) 성능을 향상시키며, 모든 사용자 데이터가 동일하게 유익하지 않다는 점을 강조합니다.

- **Technical Details**: FedFT-EDS는 전역 모델(Global Model)을 대규모로 사전 훈련한 후, 클라이언트가 자신의 데이터로 작은 부분만 파인 튜닝하여 컴퓨터 자원에 대한 부담을 줄입니다. 훈련 과정에서 클라이언트는 로컬 데이터에 대해 단일 전진 패스를 실행하여 가장 유용한 샘플만을 선택합니다. 이를 위해 하드 소프트맥스(Hardened Softmax) 함수를 도입하여 높은 불확실성이 있는 샘플을 우선적으로 선택합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터셋에서 실시된 실험 결과, FedFT-EDS는 기존의 FL 방법인 FedAvg 및 FedProx에 비해 전역 모델 성능이 최대 5% 향상되었고, 훈련 시간은 1/3로 줄어드는 성과를 보였습니다. 특히 50%의 가장 유용한 데이터를 선택해도 전체 데이터셋을 사용할 때보다 성능이 더 나은 결과를 보여주며, 데이터 선택의 중요성을 강조합니다.



### DeepLL: Considering Linear Logic for the Analysis of Deep Learning Experiments (https://arxiv.org/abs/2501.00169)
Comments:
          8 pages, 3 figures

- **What's New**: 이번 연구에서는 Deep Learning 실험의 분석을 위한 Linear Logic의 활용을 조사했습니다. 실험의 control flow를 추상적으로 표현하고, API 호출 및 하드웨어 자원과 같은 사용 가능한 실험 자원 세트를 명시할 수 있는 방법을 제안합니다. 또한 실험 중 자원을 올바르게 사용하는 규칙에 대한 추론 규칙도 포함하고 있습니다.

- **Technical Details**: Deep Learning 실험에서 데이터셋을 관리하고 하드웨어 가속기와 상호작용하는 API를 효율적으로 사용하기 위해서는 신중한 접근이 필요합니다. 데이터 처리 중의 소프트웨어 실수는 실험을 오염시켜 잘못된 결과를 초래할 수 있습니다. 또한, 잘못 작성된 API는 비효율적인 자원 사용으로 이어져 신뢰할 수 없는 결론을 초래할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 경량이며 이해하기 쉬운 두 가지 구성 요소인 기호적(symbolic) 요소와 시각적(visual) 요소를 갖추고 있습니다. 실험 결과를 보장하기 위해 필수적인 데이터셋의 분리와 올바른 하이퍼파라미터 튜닝 과정이 중요한 역할을 하며, 이는 실험의 정확성과 효율성을 높이는 데 기여합니다.



### Class-based Subset Selection for Transfer Learning under Extreme Label Shif (https://arxiv.org/abs/2501.00162)
Comments:
          19 pages

- **What's New**: 이 논문은 기존의 전이 학습 방법의 한계를 극복하기 위해 새로운 몇 가지 샷 전이 학습 프로세스를 제안합니다. 특히, 다양하게 분포하는 클래스 간의 전이를 최적화하기 위하여 소스 도메인에서의 클래스 선택 및 가중치 부여를 통해 도메인 간의 거리(minimizing the distance)를 최소화하는 방식입니다. Wasserstein distance를 활용하여 두 도메인 간의 관계를 더욱 효과적으로 파악할 수 있습니다.

- **Technical Details**: 논문에서는 소스 도메인과 타겟 도메인 간의 전이 가능성을 증가시키기 위해, 특성 공간(feature space)에서 Wasserstein distance를 적용하여 클래스 선택 및 재가중합을 수행합니다. ResNet-50 모델을 사용하여 데이터의 매핑을 수행하고, 학습된 분류기를 제한된 타겟 도메인 샘플에 대해 미세 조정(finetune)합니다. 이 접근 방식은 기존의 클래스 세트 간의 관계에 대한 가정을 하지 않으며, 극단적인 케이스에도 적합합니다.

- **Performance Highlights**: 실험을 통해 제안된 WaSS 방법이 총 7개의 벤치마크 데이터 세트에서 평균적으로 여섯 가지 다른 전이 학습 방법보다 더 높은 분류 정확도를 제공함을 증명합니다. 이러한 결과는 라벨 시프트(label shift) 설정을 포함하여 다양한 조건에서 우수한 성능을 보여줍니다. 또한, 학습된 분류기의 오차를 경계(bound) 지을 수 있는 이론적인 분석도 제공됩니다.



### Temporal reasoning for timeline summarisation in social media (https://arxiv.org/abs/2501.00152)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 시간적 추론 능력을 향상시키는 것이 긴 텍스트의 타임라인 요약 품질을 개선할 수 있는지 조사합니다. 특히 사회적 미디어 스레드와 같은 사건의 연속을 포함하는 텍스트를 요약하는 작업을 다룹니다. 우리는 새로운 데이터 세트인 NarrativeReason을 소개하여 사건 간의 시간적 관계를 중점적으로 다루며, 기존의 데이터 세트와의 차별점을 보입니다.

- **Technical Details**: 연구에서는 사건의 시간적 관계를 효과적으로 처리하기 위해 LLM을 시간적 추론 임무에 맞게 미세 조정(fine-tuning)합니다. 이러한 튜닝을 통해 교사 모델을 만들고, 이후 이를 학생 모델로 지식 증류(knowledge distillation)하여 타임라인 요약 작업을 수행하게 합니다. 세 가지 전략인 Neuron Selectivity Transfer, Contrastive Representation Distillation, Probabilistic Knowledge Transfer를 통해 지식이 전달됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 정신 건강과 관련된 긴 사회적 미디어 스레드를 요약하는 작업에서 뛰어난 성능을 보여주며, 신뢰할 수 있는 타임라인 요약을 생성합니다. 이 모델은 정확한 요약을 생성할 뿐만 아니라 LLM의 환각(hallucination) 현상을 줄이는 데에도 기여했습니다. 이는 시간적 추론을 활용하는 것이 타임라인 요약의 품질을 향상시킬 수 있다는 것을 보여줍니다.



### NiaAutoARM: Automated generation and evaluation of Association Rule Mining pipelines (https://arxiv.org/abs/2501.00138)
- **What's New**: 본 논문에서는 Numerical Association Rule Mining (NARM) 파이프라인을 자동으로 구축하는 새로운 방법인 NiaAutoARM을 제안합니다. 이는 Stochastic Population-Based Meta-Heuristics를 기반으로 하여, 사용자 개입 없이 최적의 파이프라인을 찾을 수 있는 솔루션을 제공합니다. 기존의 AutoML 방법들은 주로 분류 및 회귀 문제를 해결하는 데 집중된 반면, NiaAutoARM은 숫자 및 범주형 속성을 동시에 처리할 수 있는 ARM 파이프라인을 자동으로 구성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: NiaAutoARM의 제안된 방법은 ARM 파이프라인 구축을 지속적 최적화 문제로 정의합니다. 이 방법은 각 솔루션 집단의 개체가 가능한 ARM 파이프라인을 나타내며, 다양한 전처리 방법 및 메트릭 조합을 시험합니다. NiaAutoARM은 Evolutionary Algorithms (EA) 및 Swarm Intelligence (SI) 기반의 알고리즘을 통해 운영되며, 수집되는 메트릭의 조합으로 피트니스 함수를 구성하여 적합한 ARM 규칙 선택을 돕습니다.

- **Performance Highlights**: NiaAutoARM은 여러 데이터셋에 대해 엄밀한 실험 평가를 통해 성능이 입증되었습니다. 현존하는 AutoML 방법과의 비교를 통해 더 높은 신뢰성과 정확성을 제공하며, 과거 논문들에서 다루지 않았던 전처리 단계에 특별한 주의를 기울인 것이 특징입니다. NiaAutoARM은 파이썬 패키지로 구현되어, 사용자들이 쉽게 접근할 수 있도록 돕는 유용한 도구로 자리잡을 것입니다.



### Detection-Fusion for Knowledge Graph Extraction from Videos (https://arxiv.org/abs/2501.00136)
Comments:
          12 pages, To be submitted to a conference

- **What's New**: 본 논문에서는 영상 이해의 주요 과제로서 비디오에서 의미 있는 내용을 추출하기 위한 새로운 방법을 제안합니다. 기존의 시스템들이 자연어 모델을 사용하여 비디오를 설명하는 데 따른 여러 단점을 극복하기 위해, 지식 그래프(knowledge graphs)를 활용하여 비디오에 주석을 달고자 합니다. 이 방법은 비디오의 비주얼 콘텐츠에 기반한 내용을 더 효율적으로 표현할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 깊이 학습(deep learning) 기반으로, 먼저 비디오 입력에서 개체(individuals)와 그들 간의 관계를 예측한 후, 이를 통해 지식 그래프를 구성합니다. 개체는 학습 가능한 벡터로 표현되고, 관계는 다층 퍼셉트론(multi-layer perceptrons, MLP)으로 처리됩니다. 이 모델은 비디오에서 탐지된 개체와 서술어(predicate)를 결합하여 사실(fact)을 예측하며, 문장을 사용하지 않고도 보다 정량적인 평가가 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 지식 그래프로 비디오에 주석을 다는 중요한 작업에서 기존 작업들보다 월등히 성능이 뛰어남을 입증했습니다. 모델의 각 구성요소에 대한 기여도를 분석한 결과, 개체와 서술어 수를 조절함에 따라 실행 시간과 정확도 간의 트레이드오프(trade-off)가 나타났습니다. 지식 그래프의 배경 지식(background knowledge) 포함에 대한 탐색은 해당 분야에서 최초로 이루어진 연구입니다.



### GroverGPT: A Large Language Model with 8 Billion Parameters for Quantum Searching (https://arxiv.org/abs/2501.00135)
Comments:
          12 pages including appendices

- **What's New**: 이번 연구는 Grover의 양자 알고리즘을 시뮬레이션하기 위해 대형 언어 모델인 GroverGPT를 소개합니다. GroverGPT는 LLaMA 아키텍처를 기반으로 하여 8억 개의 매개변수를 가지고 있으며, 15조 개 이상의 토큰으로 훈련되었습니다. 이 모델은 양자 상태를 명시적으로 표현하지 않고도 양자 검색 알고리즘을 근사할 수 있는 패턴 인식 접근법을 채택하여 고전적 시뮬레이션의 한계를 탐색합니다.

- **Technical Details**: GroverGPT 모델은 세 가지 주요 구성 요소, 즉 양자 회로 표현, QASM(Quantum Assembly Language), 자연어 상호작용을 통합하여 훈련되었습니다. 이 모델은 3-10 큐비트의 훈련 세트와 6-20 큐비트의 테스트 세트를 기반으로 구성되어 있으며, 훈련 과정에서 양자 알고리즘의 구조적 및 동작적 측면을 이해할 수 있도록 설계되었습니다. GroverGPT는 양자 Turing 머신을 시뮬레이션하기 위해 Classical Description을 입력으로 받고, 출력으로는 각 비트 문자열에 대한 확률 분포를 생성합니다.

- **Performance Highlights**: GroverGPT는 97K 양자 검색 인스턴스를 분석한 결과, OpenAI의 GPT-4o 모델(정확도 45%)에 비해 일관되게 뛰어난 성능을 보였습니다. 특히 4큐비트 또는 더 큰 데이터 세트에서 훈련받은 경우, 6 및 10 큐비트 데이터셋에서 거의 100%의 정확도를 달성했습니다. 또한 20큐비트 이상의 시스템에서도 95% 이상의 일반화 성능을 나타내며, 이는 양자 특징을 잘 포착하고 있음을 시사합니다.



### A Data-Centric Approach to Detecting and Mitigating Demographic Bias in Pediatric Mental Health Text: A Case Study in Anxiety Detection (https://arxiv.org/abs/2501.00129)
- **What's New**: 이 연구는 어린이 정신 건강 스크리닝을 지원하는 AI 모델의 훈련 데이터에서 비생물학적 차이에 따른 언어적 차이를 탐지하고 완화하는 방법을 제시합니다. 이는 기존의 구조화된 데이터에 대한 편향 문제를 넘어 비구조화된 데이터에서의 편향 문제에 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 성별 하위 그룹 간의 결과 동등성을 평가하고, 성 중립적인 용어로 편향된 용어를 중화하는 데이터 중심의 디바이싱(de-biasing) 방법을 적용했습니다. 이 접근법은 소아환자의 자동 불안 감지 모델에서 테스트되었으며, 성별에 따른 정보 밀도와 언어적 차이가 진단 정확도에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 여성 청소년 환자의 체계적인 과소 진단이 발견되었고, 남성 환자 대비 4% 낮은 정확도와 9% 높은 허위 음성률(FNR)을 기록했습니다. 디바이싱 방법을 통해 진단 편향이 최대 27% 감소했으며, 이는 인구 집단 간의 평등 성 향상에 효과적임을 보여주었습니다.



### Text-to-Image GAN with Pretrained Representations (https://arxiv.org/abs/2501.00116)
- **What's New**: 본 논문에서는 TIGER라는 새로운 텍스트-이미지 GAN을 제안하여, 이전의 GAN보다 빠르고 강력한 모델을 구축하려고 합니다. TIGER는 pretrained representations을 활용하여 복잡한 씬에 대한 이해 능력을 크게 향상시킬 수 있는 비전 강화 판별기(vision-empowered discriminator)와 텍스트-이미지 융합을 효율적으로 수행할 수 있는 고용량 생성기(high-capacity generator)를 포함합니다. 이 연구는 GAN을 재조명하여 새로운 접근 방식을 제공합니다.

- **Technical Details**: TIGER의 비전 강화 판별기는 여러 pretrained 비전 모델의 다양한 표현을 수집하여 성능을 향상시킵니다. 이 판별기는 복수의 서브 판별기로 구성되어 있으며, 각 서브 판별기는 서로 다른 pretrained 모델에서 추출된 표현을 처리합니다. 고용량 생성기는 여러 개의 신선한 고용량 융합 블록(HFBlock)으로 구성되며, 각 블록은 효율적인 텍스트-이미지 융합을 위해 여러 깊은 융합 모듈을 포함하고 있습니다.

- **Performance Highlights**: TIGER는 일반 텍스트-이미지 합성(task) 과제에서 두 가지 도전적인 데이터 세트에서 최첨단 성능을 달성하며, FID 값은 COCO에서 5.48, CUB에서 9.38로 기록되었습니다. 제로샷 텍스트-이미지 합성(task) 과제에서도 적은 모델 파라미터와 적은 훈련 데이터로 비슷한 성능을 보이며, LDM 및 Parti보다 120배 빠른 추론 속도를 자랑합니다.



### An Unsupervised Anomaly Detection in Electricity Consumption Using Reinforcement Learning and Time Series Forest Based Framework (https://arxiv.org/abs/2501.00107)
- **What's New**: 이번 연구에서는 Reinforcement Learning (RL) 기반의 모델 선택 방식을 사용하여 비지도 Anomaly Detection (AD)을 수행하는 새로운 접근 방식을 제안합니다. 기존 연구에서 AD 방법들은 특정 이상치 유형에 대해 특정한 가정을 해왔지만, 우리의 모델은 다양한 형태의 이상치를 효과적으로 탐지할 수 있습니다. 특히, 시간 시계열 데이터의 AD에 집중하여, 라벨링된 데이터 없이도 모델을 최적화하여 사용할 수 있는 방법론을 소개합니다.

- **Technical Details**: 제안하는 AD 모델 선택 프레임워크는 시간 시계열 포레스트(Time Series Forest, TSF)와 RL을 결합하여 진행됩니다. 이 프레임워크는 각 시계열 데이터 포인트에서 가장 적합한 AD 모델을 동적으로 선택하며, 전량의 라벨 없이도 AD를 수행하도록 설계되었습니다. 또한, 개별 TSF 분류기의 예측 결과와 일부 라벨을 기반으로 최적의 AD 솔루션을 찾을 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: 제안한 모델은 실제 및 합성 데이터셋에서 KNN을 제외한 모든 AD 모델을 초월하는 뛰어난 성능을 보여주었습니다. 특히, F1 점수에서 0.989라는 인상적인 수치를 기록하며, 다양한 유형의 이상치에 대해서도 일관된 높은 성능을 유지합니다. 추가적으로, GPT-4와 비교했을 때도 더 나은 결과를 나타내어 RLAD의 유효성을 입증했습니다.



### LicenseGPT: A Fine-tuned Foundation Model for Publicly Available Dataset License Complianc (https://arxiv.org/abs/2501.00106)
- **What's New**: 이 논문에서는 상업적 AI 제품 개발에서 데이터셋 라이센스 준수의 복잡성을 해결하기 위한 'LicenseGPT'라는 파인튜닝된 파운데이션 모델을 소개합니다. 기존 법률 전용 FM의 성능을 평가한 결과, 예측 동의율(PA)이 43.75%에 불과했습니다. 반면, LicenseGPT는 500개의 법률 전문가가 주석을 단 라이센스 데이터셋을 통해 PA를 64.30%로 향상시켰습니다. 이 연구는 LicenseGPT가 상업적 소프트웨어 개발에서 법적 리스크를 줄이는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: LicenseGPT는 공개된 데이터셋 라이센스를 수집하여 파인튜닝된 FM으로, 법률 전문 지식을 바탕으로 라이센스의 상업적 사용 가능 여부와 의무를 식별합니다. 본 연구에서 사용된 DL 데이터셋은 500개의 데이터셋 라이센스를 포함하고 있으며, 각 라이센스는 상업적 사용 허용 여부 및 관련된 권리와 의무에 대한 이유를 주석으로 다룹니다. 연구 질문은 LicenseGPT의 성능을 기존 법률 FM과 일반용 FM과 비교 분석하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: LicenseGPT는 소프트웨어 IP 변호사들에게 94.44%의 분석 시간 단축을 보여주며, 라이센스 당 평균 6초를 기록했습니다. 이는 기존의 분석 속도인 108초에서 크게 향상된 성과입니다. 변호사들은 LicenseGPT를 귀중한 보조 도구로 인식하나 복잡한 사례에 대한 인간의 검토가 여전히 필요하다고 언급했습니다. 이 연구는 AI 소프트웨어 개발 생애 주기에서 법적 준수 통합의 필요성을 강조하며, LicenseGPT와 같은 도구의 필요성을 강조합니다.



### CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions (https://arxiv.org/abs/2501.00097)
- **What's New**: 이 논문은 법률 분야에서 긴 문맥 요약에 대한 새로운 데이터셋인 CaseSumm를 소개합니다. 이 데이터셋은 25,600개의 미국 대법원(SCOTUS) 의견과 공식 요약인 'syllabuses'를 포함하고 있으며, 1815년부터의 SCOTUS 판결 요약을 포함한 최초의 데이터셋입니다. CaseSumm는 현재 가장 큰 공개 법률 사건 요약 데이터셋으로, 연구 커뮤니티에 중요한 자원으로 제공됩니다.

- **Technical Details**: CaseSumm 데이터셋은 법률 문서 요약에서의 다양한 도전 과제를 해결하기 위해 설계되었습니다. 데이터셋은 공식적으로 승인된 시라버스를 포함하여 미국 대법원 판결을 기반으로 하며, 인간 평가와 자동 메트릭을 모두 통해 요약 성능을 종합적으로 평가합니다. 특히, Mistral 7b 모델이 자동 메트릭에서 우수한 성능을 보이지만 전문가의 주관적 평가에서는 환각(hallucination) 문제로 인해 낮은 점수를 받았음을 지적합니다.

- **Performance Highlights**: 연구 결과에 따르면, 대부분의 자동 메트릭에서 Mistral가 대형 모델들을 초월하는 것으로 나타났지만, 인간 전문가들은 GPT-4의 요약을 더 명확하며 중요한 정보에 대한 민감성과 정확성을 더 잘 나타낸다고 평가합니다. 요약에서 발생하는 특정 환각 유형과 사실 오류도 분석하였으며, 이는 법률 요약에서의 자동 메트릭 평가의 한계를 보여줍니다. CaseSumm는 법률 요약 품질 평가에서 인간 전문가의 역할이 얼마나 중요한지를 강조합니다.



### Machine Learning-Based Security Policy Analysis (https://arxiv.org/abs/2501.00085)
- **What's New**: 이 연구는 SELinux(보안 강화 리눅스)의 정책 분석을 자동화하는 새로운 접근 방식을 제안합니다. 기존의 복잡한 정책 언어로 인해 발생하는 문제를 해결하기 위해 그래프 기반 기술과 머신 러닝을 결합하여 정책 내 이상 감지를 시도하고 있습니다. 이를 통해 SELinux 정책 분석의 효율성을 높이고, 보다 정교한 자동화가 가능해졌습니다.

- **Technical Details**: 연구에서는 Neo4j를 활용하여 SELinux 정책을 그래프 형태로 표현하고, Node2vec을 이용하여 이 그래프 구조를 머신 러닝 모델이 처리할 수 있는 의미 있는 벡터 임베딩으로 변환합니다. 두 가지 주요 질문에 대한 답변으로, 그래프 분석을 통해 SELinux 정책 분석의 자동화 가능성과 여러 이상 감지 모델 간의 비교를 다룹니다. 다양한 머신 러닝 모델을 평가하여 정책 위반과 이상 감지의 효과성을 비교합니다.

- **Performance Highlights**: 연구 결과 MLP(다층 퍼셉트론) 신경망 모델이 여러 데이터셋에서 95%의 정확도로 우수한 성능을 나타냈습니다. 반면, Random Forest와 SVM 모델은 경쟁력 있는 성능을 보였지만 약간 낮은 정확도를 기록했습니다. 이러한 그래프 기반 모델링과 머신 러닝 접근 방식을 통해 SELinux 정책을 이해하고 분석하는 전통적인 수동 분석 방법보다 더 발전된 자동화 접근 방식을 제공합니다.



### AI Agent for Education: von Neumann Multi-Agent System Framework (https://arxiv.org/abs/2501.00083)
Comments:
          Conference Proceedings of the 28th Global Chinese Conference on Computers in Education, GCCCE 2024

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 교육 분야에서 새로운 패러다임(Paradigm)을 열어주었습니다. 이 논문에서는 다중 에이전트 시스템(MAS)과 관련하여 본 논문의 von Neumann 다중 에이전트 시스템(framework)을 제안하고 있습니다. 각 AI 에이전트는 통제 유닛(control unit), 논리 유닛(logic unit), 저장 유닛(storage unit), 입출력 장치(input-output devices)의 네 가지 모듈로 구성되며, 작업 분해(task deconstruction), 자기 성찰(self-reflection), 메모리 처리(memory processing), 도구 호출(tool invocation)의 네 가지 작업 유형을 정의합니다.

- **Technical Details**: 이 연구에서는 정보 흐름을 고려하여 LLM 기반 AI 에이전트의 구조를 네 가지 핵심 요소인 계획(planning), 행동(action), 도구(tools), 메모리(memory)로 나누어 설명합니다. AI 에이전트는 계획 모듈을 통해 문제를 분석하고, 메모리 모듈과 결합하여 사고를 발전시키며, 도구 모듈을 사용하여 외부 자원을 통합하고, 마지막으로 행동 모듈을 통해 사고 결과를 실행합니다. 또한 '체인 오브 생각'(Chain of Thought, CoT)과 '회전 및 행동'(Reson+Act, ReAct) 기술을 소개하며, 이는 AI 에이전트가 복잡한 작업을 효과적으로 처리하는 데 기여합니다.

- **Performance Highlights**: AI 에이전트는 작업 분해(task decomposition)를 통해 복잡한 문제를 더 간단한 하위 목표로 나누어 해결의 효율성을 높입니다. 예를 들어, CoT 방법은 '단계별 사고'(think step by step) 접근 방식을 통해 복잡한 작업을 분해하여 명확한 사고 과정을 제시하게 합니다. 자기 성찰을 통해 에이전트는 이전 작업에서의 실수를 분석하고 교훈을 통해 향후 작업의 질을 높일 수 있습니다.



### Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors (https://arxiv.org/abs/2501.00078)
- **What's New**: 이번 논문에서는 VALORANT와 유사한 2v2 전술 슈팅 게임에서 인공지능(AI) 에이전트가 인간과 유사한 방식으로 플레이할 수 있도록 학습시키는 새로운 방법론을 제시합니다. 이 접근 방식은 높은 계산 능력이 필요한 전통적인 픽셀 기반 센서 대신 소량의 레이 캐스트(ray-cast) 센서를 사용하는 픽셀 없는 인식 아키텍처를 활용합니다. 이로 인해 컴퓨터 자원의 제약 속에서도 AI가 효과적으로 작동할 수 있게 되었습니다. 이 논문은 AI 모델이 실제 게임 환경에서 의사결정을 즉각적으로 수행할 수 있도록 설계되었음을 강조합니다.

- **Technical Details**: AI 에이전트는 감독 학습(supervised learning)을 통해 인간의 궤적 데이터를 기반으로 훈련되어 실감나는 행동을 수행할 수 있도록 합니다. 이 연구는 적은 CPU 하드웨어를 사용해도 훈련 및 추론을 수행할 수 있는 효율적인 딥 뉴럴 네트워크(deep neural networks) 구조를 필요로 합니다. 또한, 멀티 플레이어 FPS(first-person shooter) 게임에서의 AI 성능, 추론 시간, 신뢰성을 평가하기 위해 다양한 평가 방법을 사용합니다. 인간과의 대결을 통해 AI의 성능을 검증하며, Turing-test와 유사한 실험을 통해 AI 행동의 신뢰성을 평가합니다.

- **Performance Highlights**: 우리의 AI 에이전트는 적은 계산 자원으로도 인간과 유사한 플레이 경험을 제공하며, 게임 산업에 상당한 발전을 기여할 것으로 기대됩니다. 최종 평가에서 AI 에이전트의 행동이 사용자에게 신뢰할 수 있는 경험을 제공함을 확인했습니다. 또한, 우리는 AI 모델이 현대 3D 비디오 게임에 성공적으로 배포될 수 있는 가능성을 제시하며, 게임 디자이너들이 사용 가능한 유용한 도구가 될 것이라는 목적을 강조합니다.



### A Novel Framework for Learning Stochastic Representations for Sequence Generation and Recognition (https://arxiv.org/abs/2501.00076)
Comments:
          14 pages, 6 figures

- **What's New**: 본 연구에서는 동적 환경에서 작동하는 자율 시스템을 위해 새로운 확률적 Recurrent Neural Network with Parametric Biases (RNNPB) 모델을 제안합니다. 이 모델은 비간섭성(autoencoder)에서 사용되는 재매개변화 기법(reparameterization trick)을 활용하여 잠재 공간(latent space)에 확률적 요소를 도입합니다. 이를 통해 다차원 시퀀스의 확률적 표현을 학습하여 불확실성을 포착하고 오버피팅(overfitting) 대응력을 향상시킵니다.

- **Technical Details**: 제안된 RNNPB 모델은 로봇 모션 데이터셋을 사용하여 시간적 패턴을 생성하고 인식하는 성능을 평가했습니다. 실험 결과, 확률적 RNNPB 모델은 결정론적(deterministic) 모델보다 운동 시퀀스를 생성하고 인식하는 데 있어 우수한 성능을 보였습니다. 이 모델은 학습 및 추론 과정에서 불확실성을 정량화하고 조정할 수 있는 능력이 있음을 강조합니다.

- **Performance Highlights**: 확률성이 있는 잠재 공간 표현 덕분에 안정적인 모션 생성(stable motion generation) 및 새로운 시퀀스를 인식할 때 향상된 일반화 성능이 달성되었습니다. 이 접근 방식은 시간적 패턴을 모델링하기 위한 생물학적으로 영감을 받은 프레임워크를 제공하여 인공지능 및 로보틱스 분야에서 강력하고 적응 가능한 시스템 개발을 촉진합니다.



### Open-Book Neural Algorithmic Reasoning (https://arxiv.org/abs/2501.00072)
Comments:
          Appeared at NeurIPS 2024

- **What's New**: 이번 논문에서는 Neural Algorithmic Reasoning(NAR) 분야에 대한 새로운 접근법인 open-book learning 프레임워크를 제안합니다. 이 프레임워크는 네트워크가 특정 인스턴스를 추론할 때 훈련 데이터셋 내의 모든 인스턴스에 접근하고 활용할 수 있도록 합니다. 이를 통해 기존의 지도 학습 패러다임을 뛰어넘어, 알고리즘적인 추론 능력을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: open-book learning 프레임워크는 cross-attention 메커니즘을 기반으로 구축되며, 특히 단일 작업 훈련과 다중 작업 훈련 시에 효과적입니다. NAR 작업은 문제 인스턴스와 알고리즘 실행 결과에 따라 구성된 데이터셋을 통해 훈련되며, 네트워크는 연속적으로 문제를 해결하는 과정에서 필요한 정보를 집계하여 정확성을 높입니다. 이 논문에서는 기존 NAR 아키텍처에 두 개의 추가 모듈을 통합하여 훈련 세트에서 정보를 집계하는 방법을 제안합니다.

- **Performance Highlights**: CLRS 알고리즘 추론 벤치마크에서 제안된 프레임워크를 사용하여 다양한 네트워크 아키텍처에서 실험한 결과, 각 모델의 추론 능력이 유의미하게 향상된 것을 확인하였습니다. 다중 작업 훈련에 대한 연구에서도, 제안된 open-book 프레임워크가 각 알고리즘 작업에 대한 다중 작업 훈련의 효과를 재현할 수 있음을 보여주었으며, 몇몇 작업에서는 더 높은 정확도를 달성하기도 했습니다. 이 연구는 다양한 작업 간의 내재적 관계를 분석할 수 있는 유용한 도구로 자리 잡고 있습니다.



### ICLR: In-Context Learning of Representations (https://arxiv.org/abs/2501.00070)
Comments:
          Preprint (Under Review)

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 사전 훈련(pretraining) 데이터에 기반하여 개념의 표현을 구성하는 방식을 보여줍니다. 학습된 개념의 맥락(context)이 새롭고 다르다면, 이러한 개념 표현이 어떻게 재구성되는지를 탐구합니다. 특히, '그래프 추적(graph tracing)'이라는 간단한 작업을 정의하여, 모델이 맥락 특정 의미를 반영하는 방식으로 표현을 조정할 수 있는지를 조사합니다.

- **Technical Details**: 연구에서는 Llama3.1-8B 모델을 주로 사용하며, 다양한 구조(예: 사각형 격자, 원형, 육각형 격자)에서 랜덤 워크(random walks)에 대한 실험을 진행합니다. 그래프의 노드는 과거 학습에서 잘 알려진 개념으로 참조되며, 연결 구조는 임의로 설정됩니다. 이 과정을 통해 입력된 맥락에 따라 모델이 개념 표현을 어떻게 변화시키는지를 분석합니다.

- **Performance Highlights**: 결과적으로, 맥락의 크기가 커질수록 모델 표현의 구조가 그래프 연결성과 일치하도록 재조직되는 현상을 관찰했습니다. 흥미롭게도 이러한 결과는 유사한 설정에서 인간 피험자에게서 나타난 결과와 유사하여, LLM이 특정 맥락에 따라 의미를 재구성할 수 있는 능력을 가지고 있음을 시사합니다. 이 연구는 맥락 학습(in-context learning)이 본질적으로 최적화 과정과 관련이 있음을 제시합니다.



### Adversarial Negotiation Dynamics in Generative Language Models (https://arxiv.org/abs/2501.00069)
Comments:
          Paper at NeurIPS 2024 Workshop on Red Teaming GenAI

- **What's New**: 이번 연구에서는 계약 작성 및 개선에 사용되는 생성적 언어 모델의 성능과 취약성을 평가하여 AI 안전과 보안에 대한 중요성을 강조합니다. 경쟁하는 양 당사자가 서로 다른 언어 모델을 사용하면서 발생하는 게임 이론적 문제와 AI 안전성 문제를 다루고 있습니다. 이를 통해 더 안전하고 신뢰할 수 있는 모델의 개발에 기여하고자 합니다.

- **Technical Details**: 우리는 8개의 생성적 언어 모델을 사용하여 계약 협상 시나리오에서의 행동을 조사했습니다. 각 모델은 판매자 또는 구매자의 역할을 맡아 상상의 두 기업 간 100대 머신 판매 계약을 생성하고, 서로 교환하며 협상하는 방식을 채택했습니다. 평가 과정에서는 잇따라 6개의 모델이 최종 계약을 평가하며, 이 투표를 통해 계약의 유리함을 판별했습니다.

- **Performance Highlights**: 연구 결과는 일반 모델과 전문 법률 모델 간의 협상 행동에서 뚜렷한 차이를 보여주고 있습니다. 일반 모델은 각각의 역할에서의 적응력이 뛰어난 반면, 법률 전문 모델은 공정성을 고려한 균형 잡힌 성과를 보였습니다. 모델 선택이 협상 결과에 큰 영향을 미칠 수 있음을 강조하며, 특정 상황에 맞는 모델을 전략적으로 배치하는 것이 중요하다고 결론지었습니다.



### Ensemble of classifiers for speech evaluation (https://arxiv.org/abs/2501.00067)
- **What's New**: 이 연구는 의료 분야에서의 음성 평가 문제를 해결하기 위해 이진 분류기의 앙상블을 적용한 시도를 설명합니다. 또한, 음절 발음 품질에 대한 정량적 및 전문가 평가를 기반으로 데이터셋이 구성되었습니다.

- **Technical Details**: 이 연구에서 사용된 특징으로는 7개의 선택된 메트릭스에 대한 정량적 평가가 포함되며, 이는 dynamic time warp distance, Minkowski distance, correlation coefficient, longest common subsequence (LCSS), edit distance of real sequence (EDR), edit distance with real penalty (ERP), 그리고 merge split (MSM)입니다. 전문가 평가에 의해 음성 품질이 클래스 레이블로 지정되며, 클래스 1은 고품질 음성을 의미하고, 클래스 0은 왜곡된 음성을 의미합니다.

- **Performance Highlights**: 연구에서 다섯 가지 분류 방법인 로지스틱 회귀(logistic regression), 서포트 벡터 머신(support vector machine), 나이브 베이즈(naive Bayes), 결정 트리(decision trees), K-최근접 이웃(K-nearest neighbors) 간의 훈련 결과를 비교했습니다. 또한, 앙상블 방법을 사용하여 분류기의 혼합을 통해 개별 이진 분류기를 사용할 때보다 분류 정확도를 약간 증가시킬 수 있었음을 보여주었습니다.



### On Adversarial Robustness of Language Models in Transfer Learning (https://arxiv.org/abs/2501.00066)
- **What's New**: 본 연구는 LLMs(대형 언어 모델)의 전이 학습(transfer learning) 상황에서의 적대적 강인성(adversarial robustness)을 조사합니다. 다양한 데이터셋(MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias)과 모델 아키텍처(BERT, RoBERTa, GPT-2, Gemma, Phi)를 활용한 실험을 통해, 전이 학습이 표준 성능 지표를 향상시키는 반면, 적대적 공격에 대한 취약성을 증가시킨다는 사실을 밝혀냈습니다. 이 연구는 모델 크기, 아키텍처, 적응 방법 간의 복잡한 상호 작용을 보여줍니다.

- **Technical Details**: 이 연구는 편향된 텍스트 탐지라는 분류 작업에 중점을 두고 있으며, 각 데이터셋은 편향(biased) 및 비편향(non-biased)이라는 두 개의 클래스가 균형 있게 구성되어 있습니다. 성능과 강인성을 평가하기 위해 Original Accuracy (OAcc), Attack Success Rate (ASR), Accuracy Under Attack (AUA) 등의 지표를 사용했습니다. 또한, 전이 학습을 통한 파인튜닝(fine-tuning)과 적대적 훈련(adversarial training) 기법을 통해 강인성 문제를 평가했습니다.

- **Performance Highlights**: 실험 결과, 특히 소규모 모델에서 공격 성공률(ASR)이 전이 학습 후 증가하는 경향이 관찰되었습니다. 예를 들어, Hate Speech 데이터셋에서 GPT-2는 ASR이 평균 20.4% 증가하면서 정확도는 3.67% 상승했습니다. 이러한 결과는 성능 향상이 강력성 감소와 관련된 중요한 요소를 간과하게 할 수 있다는 우려를 제기합니다.



### Predicting Preschoolers' Externalizing Problems with Mother-Child Interaction Dynamics and Deep Learning (https://arxiv.org/abs/2501.00065)
Comments:
          34 pages, 3 figures, 2 tables

- **What's New**: 이번 연구는 아이의 외부 문제(Externalizing problems)를 예측하는 데 있어 모자(母子) 상호작용의 동적 과정을 새롭게 평가하고 개선하고자 하였습니다. 특히 엄마의 자율성 지원(autonomy support)이 아동의 문제 행동에 미치는 영향을 중심으로 분석하였습니다. 이를 통해 유아기의 문제 행동을 사전에 예방할 수 있는 새로운 방향을 제시하게 되었습니다.

- **Technical Details**: 연구에서는 도전적인 퍼즐 과제를 수행하는 동안의 모자 상호작용을 분석하였습니다. 101명의 아이에 대해 남녀 비율이 46:55이며, 평균 나이는 57.41개월로 설정되었습니다. Residual Dynamic Structural Equation Model (RDSEM)과 Attention-based Sequential Behavior Interaction Modeling (ASBIM) 모델의 예측 성능을 비교하여, 딥러닝(Deep Learning) 기법을 활용한 ASBIM 모델이 예측 정확도를 높이는 데 기여했음을 확인하였습니다.

- **Performance Highlights**: RDSEM은 아동이 패배를 경험한 후 어머니가 더 많은 자율성 지지를 제공할 때 외부 문제 행동 수준이 낮아진다는 것을 밝혀냈습니다. 5배 교차 검증(Five-fold cross-validation) 결과, RDSEM 모델이 우수한 예측 정확도를 보였고, ASBIM 모델은 특히 아동의 억제 조절(inhibitory control) 특성을 개인화하여 예측 정확도를 더욱 향상시켰습니다. 이러한 결과는 부모와 자녀 간의 동적 상호작용이 아동 문제 행동 예측에 중요한 정보를 제공함을 입증합니다.



### "Generative Models for Financial Time Series Data: Enhancing Signal-to-Noise Ratio and Addressing Data Scarcity in A-Share Mark (https://arxiv.org/abs/2501.00063)
- **What's New**: 이 연구에서는 주식 시장의 데이터 부족과 낮은 신호 대 잡음 비율 문제를 해결하기 위해 두 가지 새로운 생성 모델 기반 접근 방식을 제안합니다. 첫 번째 방법은 섹터 기반 합성 접근법으로, 중국 A주 시장의 다양한 섹터에서 주식의 특성을 분류하여 신호 대 잡음 비율을 향상시키는 데 초점을 맞추고 있습니다. 두 번째 방법은 패턴 인식을 기반으로 한 재귀적 주식 데이터 합성 접근법으로, 상장 기간이 짧고 비교 가능한 기업이 제한된 주식을 위한 데이터 합성을 목표로 합니다.

- **Technical Details**: 첫 번째 접근법에서는 Approximate Non-Local Total Variation 알고리즘을 사용하여 생성된 데이터를 부드럽게 하고, Fourier Transform에 기반한 밴드패스 필터링 방법을 적용하여 잡음을 제거합니다. 이와 함께, Denoising Diffusion Implicit Models를 사용하여 샘플링 속도를 높입니다. 두 번째 접근법은 마르코프 모델을 활용하여 변동 길이의 주식 시퀀스를 학습 및 생성하며, 데이터 부족을 완화하기 위해 서브 타임 레벨 데이터 증대 방법을 도입합니다.

- **Performance Highlights**: 이에 대한 실험 결과, 생성된 데이터는 예측 모델의 성능뿐만 아니라 가격 거래 전략에 있는 개별 주식 신호의 신호 대 잡음 비율을 개선하는 데 기여하는 것으로 나타났습니다. 특히, 서브 타임 레벨 데이터를 도입함으로써 생성된 데이터의 품질이 크게 향상되었습니다. 이는 예측 정확도와 자산 배분, 리스크 관리에 대한 투자 의사결정을 개선하는 데 중요한 영향을 미칠 수 있습니다.



### ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis (https://arxiv.org/abs/2501.00062)
Comments:
          16 pages, 4 figures. Source code and data available at this https URL

- **What's New**: 본 연구에서는 ELECTRA와 GPT-4o 모델을 결합하여 3가지 감정 분류(부정, 중립, 긍정)를 수행하는 새로운 협업 방식을 탐구합니다. 특히, fine-tuned된 ELECTRA 모델의 예측 결과를 GPT 모델에 제공함으로써 분류 성능을 개선할 수 있는지에 관한 가설을 제시하였습니다. 연구 결과, 이러한 접근 방식이 단독 모델보다 우수한 성능을 보여주었으며, 비용 대비 성과(Cost/Performance Ratio)에서도 개선을 이루었습니다.

- **Technical Details**: ELECTRA와 GPT-4o/4o-mini 모델을 사용하여 Stanford Sentiment Treebank(SST)와 DynaSent 리뷰 데이터를 통합해 fine-tuning(FT)을 수행했습니다. 실험에서는 예측 클래스 레이블, 클래스별 확률, 유사한 리뷰 예시를 포함하여 다양한 프롬프트 증강 방안을 적용했습니다. 분석 결과, ELECTRA Base의 예측 결과를 공유함으로써 GPT-4o-mini의 성능이 현저히 향상되었으며, 이는 각각의 모델 단독 사용 시와 비교하여 우수한 결과를 보여주었습니다.

- **Performance Highlights**: ELECTRA Large FT 모델은 base GPT 모델보다 뛰어난 성능을 보였으며, GPT-4o FT-M과 GPT-4o-mini FT 모델도 각각 86.99와 86.77의 성과를 기록했습니다. 특히 GPT-4o-mini FT 모델은 76% 낮은 비용으로 GPT-4o FT 모델에 필적하는 성능을 달성했습니다. 이러한 결과는 리소스가 제한된 프로젝트에 있어 경제적인 대안을 제공하며, fine-tuned된 ELECTRA 모델과 예측 결과를 활용한 LLM 프롬프트 증강이 성능 향상에 기여한다고 보여집니다.



### Training-free Heterogeneous Model Merging (https://arxiv.org/abs/2501.00061)
- **What's New**: 이번 연구에서는 이질적인 모델을 효과적으로 통합할 수 있는 모델 머징(model merging) 프레임워크를 새롭게 제안했습니다. 기존의 방법들이 대부분 동질적인 아키텍처에만 적용되었던 것과 달리, 깊이와 폭의 이질성을 모두 고려하여 다양한 성능을 가진 여러 모델을 하나의 통합된 모델로 재구성할 수 있는 방식입니다. 즉, 여러 서로 다른 구조의 모델들이 좀 더 유연하게 결합될 수 있는 기반을 마련했습니다.

- **Technical Details**: 제안된 프레임워크는 주로 두 가지 기술을 사용합니다. 첫째, 깊이 이질성(depth heterogeneity)을 해결하기 위해 레이어 정렬(layer alignment) 전략을 도입하여 서로 다른 깊이 구조를 가진 모델의 계층을 일치시킵니다. 둘째, 폭 이질성(width heterogeneity)을 다루기 위해 통합된 차원 공간으로 가중치를 투영하는 엘라스틱 뉴런 지핑(elastic neuron zipping) 알고리즘을 제안하여 서로 다른 넓이를 가진 모델도 동일한 공간에서 통합될 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법들이 효과적임을 입증했습니다. 깊이와 폭 이질성을 가진 모델들을 머징한 결과, 동질적 모델 머징과 유사한 성능 수준을 달성했으며, 이는 비전과 자연어 처리 태스크 모두에서 나타났습니다. 따라서 이 연구는 이질적 모델 머징의 가능성을 보여주는 중요한 기초 작업이 될 것으로 기대됩니다.



### Large Language Models for Mathematical Analysis (https://arxiv.org/abs/2501.00059)
- **What's New**: 본 연구에서는 DEMI-MathAnalysis 데이터셋을 개발하여 수학적 분석의 증명 기반 문제에 중점을 두었습니다. 기존의 수학 데이터셋은 주로 계산 작업에 포커스를 맞추었고, 정형화된 수학적 언어를 다루는 AI의 능력을 평가하는 등격 차이가 있었습니다. 이를 통해 LLM들이 로직적이고 완전하며 우아한 증명을 생성하는 능력이 향상되었습니다.

- **Technical Details**: DEMI-MathAnalysis 데이터셋은 수학적 분석 주제인 수열, 극한, 무한급수 및 볼록 함수 등의 증명 기반 문제로 구성됩니다. 이러한 문제는 LaTeX 형식으로 표기되며, 포괄적인 단계별 해결책과 함께 제공됩니다. 이 데이터셋은 LLM의 fine-tuning을 통해 심화되어, 수학적 분석 문제 해결 능력을 증가시키기 위해 설계된 가이드 프레임워크와 결합하여 사용됩니다.

- **Performance Highlights**: 이 연구를 통해 LLM들은 보다 집중적으로 형식적이고 논리적인 문제 해결에 접근할 수 있게 되었으며, 특히 수학적 분석의 복잡성을 처리하는 데 있어 신뢰할 수 있는 AI로 성장할 수 있는 기반을 다졌습니다. LLM에 대한 정확하고 논리적 해결책을 평가하는 방법론도 제안되어, 수학적 문제 해결 능력이 더욱 향상되었습니다.



### VisTabNet: Adapting Vision Transformers for Tabular Data (https://arxiv.org/abs/2501.00057)
- **What's New**: 이 논문에서는 VisTabNet이라는 새로운 크로스 모달 전이 학습 방법을 제안하여, 사전 훈련된 Vision Transformer(ViT) 가중치를 사용하여 표 형식의 데이터를 처리할 수 있도록 합니다. 기존의 전이 학습 관행을 넘어 사전 훈련된 이미지 모델을 이용하여 표 데이터 문제를 해결하는 가능성을 보여줍니다. 이 접근 방식은 새로운 아키텍처를 설계하는 데 드는 개념적 비용을 줄이고 모델을 처음부터 훈련시키는 계산 비용을 감소시킵니다.

- **Technical Details**: VisTabNet은 표 형식 입력을 ViT에서 수용 가능한 패치 임베딩으로 투영함으로써, 사전 훈련된 Transformer Encoder를 표 형식 입력에 직접 적용할 수 있게 합니다. 이는 고차원 이상에서의 패턴을 이해하고 효율적으로 처리할 수 있게 해줍니다. 이 방식은 머신러닝에서 가장 일반적으로 사용되는 표 형식 데이터의 복잡성을 극복하기 위해 개발되었습니다.

- **Performance Highlights**: VisTabNet은 여러 개의 소규모 표 형식 데이터셋에서 실험적으로 검증되어 전통적인 앙상블 방법과 최근의 깊은 학습 모델보다 우수한 성과를 보였습니다. 특히, 소수의 샘플로도 효과적으로 전이 학습이 이루어지며, 이는 크로스 모달 전이가 깊은 네트워크를 처음부터 훈련하는 것보다 더 효과적임을 증명합니다. 본 연구는 표 형식 데이터에 대한 깊은 학습 접근을 개선할 수 있는 가능성을 제시합니다.



### Transforming CCTV cameras into NO$_2$ sensors at city scale for adaptive policymaking (https://arxiv.org/abs/2501.00056)
Comments:
          43 pages

- **What's New**: 이 연구에서는 도시의 CCTV 카메라를 NO₂의 대체 센서로 활용하는 혁신적인 방법을 제안합니다. 기존의 환경 센서가 부족한 도시에서, CCTV 영상에 내재된 정보와 딥 러닝 예측 모델을 결합하여 NO₂ 수치를 예측하고 있습니다. 연구팀은 런던의 1억 3천만 개 이상의 프레임을 분석하여 교통 흐름과 관련된 패턴이 NO₂ 농도에 미치는 영향을 밝혀내었습니다.

- **Technical Details**: 본 연구는 CCTV 영상과 환경 요인, 공간적 요인을 결합한 예측 그래프 딥 모델을 사용하여 NO₂ 농도를 추정합니다. 센서 데이터의 부족을 해결하기 위해 다양한 데이터 출처를 통합하여, 도시 내 NO₂ 농도를 정밀하게 측정하는 방법론을 개발하였습니다. 이러한 접근은 차량 및 보행자의 움직임을 포착하여 특정 시간대의 NO₂ 농도를 예측하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 연구 결과, 특정 교통 패턴이 NO₂ 농도에 미치는 영향을 나타내는 시공간적 연관성을 발견하였습니다. 예를 들어, 특정 교통 수단(예: 트럭)의 저녁 주행이 아침 출근 시간에 NO₂ 수치에 영향을 줄 수 있음을 확인했습니다. 이러한 결과는 현재 시행 중인 도시 정책의 효과에 대한 의문을 제기하며, 정책결정자들이 실시간으로 환경 문제를 모니터링하고 해결하는데 기여할 수 있는 방법을 제시합니다.



### LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models (https://arxiv.org/abs/2501.00055)
- **What's New**: 본 연구에서는 공격 방법론에 대한 새로운 통찰력을 제시합니다. 연구팀은 기존의 jailbreak 방식의 한계를 극복하기 위해, 생물학적 바이러스의 진화 및 감염 과정을 영감을 받아 새로운 공격 방식인 LLM-Virus를 제안합니다. 이 방법은 진화 알고리즘을 기반으로 하여, 효율성, 이식성 및 낮은 시간 비용을 보장합니다.

- **Technical Details**: LLM-Virus는 공격을 진화적 감염 문제로 간주하며, LLM을 휴리스틱 진화 연산자로 활용합니다. 이 연구는 새로운 출력의 가능성을 극대화하기 위해 LLM 기반의 교차 및 돌연변이 연산을 제안합니다. 또한, 전송 학습 문제로 보아 Local Evolution 및 Generalized Infection 기법을 도입하여 계산 비용과 시간 비용을 줄입니다.

- **Performance Highlights**: 실험 결과, LLM-Virus는 기존의 공격 방법들과 비교할 때 경쟁력 있는 성능을 보여주었습니다. 특히, HarmBench 및 AdvBench 데이터셋에서 LLM-Virus가 기존 방법들보다 높은 성공률을 기록했습니다. 이러한 결과는 LLM-Virus가 진화적 jailbreak의 새로운 기준이 될 수 있음을 의미합니다.



### AdvAnchor: Enhancing Diffusion Model Unlearning with Adversarial Anchors (https://arxiv.org/abs/2501.00054)
- **What's New**: 본 논문에서는 AdvAnchor라는 새로운 방법을 제안하여, 텍스트-이미지 확산 모델의 불필요한 개념 제거를 최적화합니다. 이 방법은 불편한 개념의 임베딩을 유사하게 만들어 전반적인 모델 성능을 유지하면서 해당 개념의 정의적 속성을 효과적으로 제외하는 것을 목표로 합니다. 기존의 방법들과 달리, AdvAnchor는 공격적 앵커(adversarial anchor)를 생성하여 성능 저하를 최소화하고자 합니다.

- **Technical Details**: AdvAnchor는 효과적인 개념 지우기를 위해 이상적인 앵커가 불편한 개념과의 의미론적 유사성을 유지하면서 정의적 속성을 제외해야 함을 발견하였습니다. 이를 위해 추가된 보편적 섭동(universal perturbations)은 불편한 개념의 임베딩에 추가되어, 특정 개념에 대한 모델의 예측을 저하시키거나 지우는 데 도움을 줍니다. 제안된 방법은 다양한 MU 기법과 통합될 수 있어 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, AdvAnchor는 최첨단 방법들에 비해 두 가지 성능을 크게 향상시켰습니다: 불필요한 개념의 제거 성능과 모델의 전반적인 성능 보존에서 큰 향상을 보였습니다. 즉, 효과적인 개념 지우기와 최소한의 성능 저하가 동시에 이루어질 수 있음을 확인했습니다. 또한, 수많은 실험을 통해 제안된 방법의 우수성을 입증하였으며, 해당 코드는 공공적으로 제공됩니다.



### Implementing Trust in Non-Small Cell Lung Cancer Diagnosis with a Conformalized Uncertainty-Aware AI Framework in Whole-Slide Images (https://arxiv.org/abs/2501.00053)
- **What's New**: 이 논문에서는 TRUECAM이라는 프레임워크를 개발하여 비소세포 폐암(Nonsmall Cell Lung Cancer, NSCLC) 아형 분류에서 데이터와 모델의 신뢰성을 확보하는 방법을 제시합니다. 이 프레임워크는 스펙트럼 정규화 신경 가우시안 프로세스(spectral-normalized neural Gaussian process, SNGP)와 모호성 지침에 따른 타일 제거 방법을 통합하여 데이터 신뢰성 문제를 해결합니다. 또한, 정확한 오류율을 보장하는 적합 예측(conformal prediction) 기법을 통해 모델 애플리케이션의 책임성을 강조합니다.

- **Technical Details**: TRUECAM은 NSCLC 아형 분류를 위한 데이터와 모델의 신뢰성을 보장하기 위해 세 가지 주요 요소로 구성됩니다: 1) SNGP로 입력 데이터의 유용한 표현 및 불확실성 정량화 수행, 2) 이질적인 타일을 제거하는 모호한 타일 제거(Elimination of Ambiguous Tiles, EAT) 메커니즘, 3) 신뢰할 수 있는 오류율을 보장하는 CP 적용. 이를 통해 데이터 신뢰성을 높이는 동시에, 모델 예측을 통계적으로 검증할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, TRUECAM을 적용한 AI 모델은 기존 모델에 비해 NSCLC 아형 분류의 정확도가 유의미하게 향상되었으며, 통계적으로 보장된 신뢰 구간과 개선된 해석 가능성을 제공했습니다. 또한, TRUECAM은 OOD(Out-of-Domain) 입력과 분포 변화에 견고하며, 다양한 환자 집단 간의 공정성을 촉진하는 성능을 입증했습니다. 이러한 성과는 TRUECAM이 실제 세팅에서 디지털 병리학 AI 모델의 책임 있고 효과적인 응용 프로그램을 지원하는 범용 프레임워크로 자리 잡게 합니다.



### DDD-GenDT: Dynamic Data-driven Generative Digital Twin Framework (https://arxiv.org/abs/2501.00051)
- **What's New**: 이 논문은 Dynamic Data-Driven Generative Digital Twins framework(DDD-GenDT)를 제안하며, 이 새로운 접근법은 물리 시스템과 대화형으로 상호작용할 수 있는 대형 언어 모델(LLM)의 장점을 활용합니다. DDD-GenDT는 LLM이 물리 시스템의 운영 상태에 반응하여 관련 물리 행동을 생성할 수 있도록 설계되었습니다. 이 기술은 CNC 가공 예제를 통해 LLM이 데이터의 역사적 측면과 현재 관측 결과를 활용하여 물리적 행동을 예측하게 함으로써 검증되었습니다. 또한, 이전 예측 기술에 비해 데이터와 훈련의 요구사항을 현저히 줄일 수 있습니다.

- **Technical Details**: DDD-GenDT 프레임워크는 LLM을 Dynamic Data-Driven Application System (DDDAS) 패러다임에 통합하여 시간 연속 예측을 수행합니다. 이 시스템은 과거와 현재의 관측 데이터를 바탕으로 LLM 입력을 동적으로 조정하여 물리 상태를 예측하는 제로샷(Zero-Shot) 예측 작업을 수행합니다. LLM은 이 예측을 통해 실시간 관측 데이터를 활용하여 시스템을 효과적으로 최적화하고 제어하는 데 필요한 피드백을 제공합니다. 이 기술은 복잡한 시스템에서 동적인 디지털 트윈을 구축하는데 유연하고 확장 가능한 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, DDD-GenDT 프레임워크는 CNC 가공 과정에서 높은 정확도로 물리 상태를 예측할 수 있음을 보여줍니다. 특히, GPT-4 기반의 예측은 평균 RMSE가 0.479A로 10A 최대 스핀들 모터 전류 측정치의 4.79%에 불과하여, 소량의 데이터만으로도 신뢰성 있는 예측이 가능함을 보여줍니다. 이는 디지털 트윈이 기계 도구 마모와 같은 변화에 실시간으로 적응할 수 있는 기능을 발휘하는 것을 의미합니다.



### Stroke Prediction using Clinical and Social Features in Machine Learning (https://arxiv.org/abs/2501.00048)
- **What's New**: 이번 연구에서는 미국에서 매년 발생하는 뇌졸중 위험을 예측하기 위해 머신러닝 기술을 활용하는 새로운 접근법을 제안합니다. 특히, 신경망(neural networks)과 로지스틱 회귀(logistic regression) 모델을 비교하여 각각의 장단점을 분석합니다. 뇌졸중은 두 번째로 많은 사망 원인이므로, 이를 정확하게 예측하는 것이 매우 중요합니다.

- **Technical Details**: 연구에서는 신경망의 밀집층(dense layers)과 합성곱층(convolutional layers), 그리고 로지스틱 회귀 모델을 사용하여 라이프스타일 요인(lifestyle factors)을 기반으로 뇌졸중 예측의 정확도를 평가합니다. 머신러닝을 통해 여러 독립 변수에 따른 이진 결과(binary outcomes)의 가능성을 계산할 수 있습니다. 각 모델의 성능을 비교하여 가장 효과적인 예측기(predicator)를 개발하는 것이 목표입니다.

- **Performance Highlights**: 예측 기법의 효과를 분석하는 과정에서 로지스틱 회귀가 특정 조건에서 신경망보다 성능이 우수할 수 있는 경우를 발견할 수 있습니다. 최종적으로, 연구에서 제안하는 모델은 잘못된 음성(false negatives)을 최소화하여 뇌졸중 위험 평가의 신뢰성을 높이는 데 기여할 것입니다. 이를 통해 개인들이 자신의 뇌졸중 위험을 인식하고 라이프스타일 변화를 유도할 수 있는 가능성을 제시합니다.



### Resource-Efficient Transformer Architecture: Optimizing Memory and Execution Time for Real-Time Applications (https://arxiv.org/abs/2501.00042)
Comments:
          5 pages, 1 figure

- **What's New**: 이 논문은 메모리 사용과 실행 시간을 크게 줄이면서도 원래 모델의 성능에 가까운 새로운 메모리 효율성(메모리 효율성) 트랜스포머(transfomer) 모델을 소개합니다. 최근 파라미터 효율성(parameter efficiency)과 연산 최적화(computational optimization)에 중점을 둔 트랜스포머 아키텍처가 발표되었지만, 실제 엣지 디바이스(edge devices)에서 사용할 경우 상당한 하드웨어 자원(resources)이 요구됩니다.

- **Technical Details**: 이 접근법은 임베딩 크기(embedding size)를 절반으로 줄이고, 파라미터 가지치기(parameter pruning) 및 양자화(quantization)와 같은 특정 기술을 적용하여 메모리 용적을 최적화합니다. 이는 정확도(accuracy) 손실을 최소화하면서 달성된 결과입니다. 실험적으로는 메모리 사용량이 52% 줄어들고, 실행 시간이 33% 감소함으로써 최고 수준 모델(state-of-the-art models)보다 더 나은 효율성을 보여주었습니다.

- **Performance Highlights**: 이 연구는 MobileBERT와 DistilBERT와 같은 기존의 매력적인 아키텍처와 비교하여 모델의 적합성을 입증했습니다. 주로 실시간(real-time) 및 자원 제약(resource-constrained) 응용 프로그램에 적합한 자원 친화적인 딥 러닝 아키텍처(deep learning architectures) 분야에 초점을 맞추고 있습니다.



### Time Series Feature Redundancy Paradox: An Empirical Study Based on Mortgage Default Prediction (https://arxiv.org/abs/2501.00034)
- **What's New**: 이 연구는 기계 학습을 통한 모기지(default) 예측에서 기존의 "더 많은 데이터가 더 나은 성과를 낸다"는 믿음을 도전합니다. 2012년부터 2022년까지의 Fannie Mae의 데이터를 사용하여, 짧은 기간과 중요 특징들에 집중하는 것이 예측 성과를 크게 향상시킨다는 발견을 했습니다. 연구 결과는 오래된 데이터와 비핵심 특징들이 오히려 예측 정확도를 저하시킬 수 있음을 보여 주며, 이는 전통적인 모델 접근 방식을 재고할 기회를 제공합니다.

- **Technical Details**: 연구에서는 2012년부터 2022년까지의 Freddie Mac의 모기지 월별 성과 데이터를 사용합니다. 데이터는 대출의 연체 상태, 현재 대출 잔액 및 대출 수정 정보 같은 여러 성과 지표를 포함합니다. 이 논문의 방법론은 데이터 전처리, 다양한 시간 창과 특징 세트에 대한 변수 통제 실험, 그리고 통합된 프레임워크 하의 모델 추론으로 구성됩니다.

- **Performance Highlights**: 짧은 시간 창(예: 1년)에서 훈련된 모델이 긴 기간의 역사적 데이터를 사용하는 모델보다 항상 더 나은 성과를 보였습니다. 또한 중요하게 선택된 특정 특징들에 집중하는 것이 모든 가능한 변수들을 사용하는 것보다 예측 결과를 개선하는 데 더 효과적이라는 것을 발견했습니다. 이러한 발견은 모기지 default 예측 모델의 개발에서 시간적 관련성과 간결성을 강조하고 있습니다.



### Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs (https://arxiv.org/abs/2501.00032)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론을 가속화하기 위해 Arm CPU에 최적화된 커널을 제안합니다. 기존의 양자화 방식으로 인한 메모리 대역폭 문제를 극복하기 위한 혁신적이고 효율적인 방법을 통해, LLM의 성능을 극대화하고자 합니다. 또한 코드를 쉽게 조정할 수 있도록 개선된 양자화 방법을 통해 저전력 장치에서도 효율적으로 사용할 수 있도록 설계되었습니다.

- **Technical Details**: 저자들은 다양한 저비트 폭 그룹 양자화 LLM을 위한 고도로 최적화된 GEMV(General Matrix Vector multiplication) 및 GEMM(General Matrix Matrix multiplication) 커널을 개발했습니다. 이 커널들은 ARM CPU의 벡터 및 행렬 곱셈 명령어를 최대한 활용하도록 설계되어 메모리 접근 및 오버헤드를 최소화합니다. 또한 비균일 코드를 기반으로 한 포스트 트레이닝 양자화 방법을 제공해 LLM의 품질을 개선하고 효율성을 높이고자 합니다.

- **Performance Highlights**: 연구 결과, 최적화된 4비트 그룹 양자화 커널을 통해 Arm CPU에서 LLM 추론의 첫 토큰 처리 시간에서 3~3.2배 개선되고, 메모리 유틸리티에 대한 처리 성능이 2배 향상되었습니다. 아울러 최적화된 비균일 양자화 방법은 텍스트 생성 품질을 개선하고, 기존의 주요 기술들 대비 더 나은 성능을 계속해서 구현하고 있습니다.



### Predicting Crack Nucleation and Propagation in Brittle Materials Using Deep Operator Networks with Diverse Trunk Architectures (https://arxiv.org/abs/2501.00016)
Comments:
          25 pages, 21 figures

- **What's New**: 본 논문에서는 phase-field 모델링을 통해 균열(Fracture) 문제를 에너지 최소화(energy minimization) 문제로 재구성하여 균열 발생, 전파, 합병, 분화 등을 포괄적으로 분석합니다. 특히, 높은 계산 비용을 해결하기 위한 새로운 접근법으로 DeepONet을 활용하여 취脆성(Brittle) 균열 문제를 해결하는 방안을 제시합니다. 네트워크 구조에 따라 세 가지 독특한 접근법을 탐구하며, 각각의 효과성을 입증합니다.

- **Technical Details**: DeepONet은 브랜치 네트워크(Branch Network)와 트렁크 네트워크(Trunk Network)로 구성되어 있으며, 이들은 각각 다른 구성으로 실험됩니다. 첫 번째 접근법에서는 두 단계로 구성된 DeepONet을 사용하여 학습 작업을 간소화하고, 두 번째 접근법에서는 물리 기반(Physics-informed) DeepONet을 통해 물리적 일관성을 보장합니다. 세 번째 접근법에서는 트렁크에 대한 Kolmogorov-Arnold Network를 사용하여 물리적 손실 없이 훈련을 진행합니다.

- **Performance Highlights**: 제안한 방법을 통해 균열이 발생하는 일차원 동질 바(Bar)에서의 균열 발생 및 다양한 노치 길이를 가진 단일 엣지 노치 샘플의 균열 전파 및 분화를 성공적으로 모델링합니다. 네트워크는 솔루션 필드를 정확하게 예측하며, 예측된 필드의 오차는 균열 근처에서 국소화되는 특징을 보입니다. 이러한 결과는 DeepONet이 복잡한 균열 문제에 효과적으로 적용될 수 있음을 보여줍니다.



### Relation-Aware Equivariant Graph Networks for Epitope-Unknown Antibody Design and Specificity Optimization (https://arxiv.org/abs/2501.00013)
- **What's New**: 본 연구에서는 Relation-Aware Antibody Design (RAAD) 프레임워크를 제안하여 항원-항체 상호작용을 동적으로 모델링합니다. 이를 통해 항원 특이적인 CDR의 서열과 구조를 공동 설계하는 방법을 소개합니다. 기존의 항체 최적화 방법에서 간과된 특이성을 강화하기 위한 새로운 평가 메트릭과 제약 조건을 개발했습니다.

- **Technical Details**: RAAD 프레임워크는 노드 특성(node features), 엣지 특성(edge features), 엣지 관계(edge relations)를 포함하여 더 많은 맥락(contextual) 및 기하학적(geometric) 정보를 통합합니다. 복잡한 CDR을 모델링하는 데 어려운 기존 방법의 한계를 극복하기 위해, 다가오는 항원-항체 상호작용을 기반으로 한 적응형 모델링을 사용합니다. 여러 CDR 유형과 서열 길이에 걸쳐 성능을 최적화할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 RAAD의 뛰어난 항체 모델링, 생성 및 최적화 능력이 입증되었습니다. 다양한 입력 맥락과 사전 훈련(pre-training) 전략에서도 뛰어난 성능을 보여주며, 항체의 특이성을 더욱 향상시킬 수 있습니다. RAAD를 통해 개발된 항체는 비특이 항체 문제를 효과적으로 해결할 수 있는 것으로 나타났습니다.



### Model-Driven Deep Neural Network for Enhanced AoA Estimation Using 5G gNB (https://arxiv.org/abs/2501.00009)
Comments:
          Presented at AAAI 2024 (Main Technical Track)

- **What's New**: 본 연구에서는 인공지능(AI)을 활용하여 위치 추정의 성능을 향상시키는 혁신적인 모델 주도 심층 신경망(모델 주도 딥 뉴럴 네트워크, MoD-DNN)을 소개합니다. 이 방법은 각도 도착(AoA) 추정을 스펙트럼 복원 문제로 재구성하여, 각도를 기준으로 한 상이한 위상 오차를 자동 보정할 수 있는 설계를 제공합니다. 실험과 시뮬레이션 결과를 통해 기존의 모델 기반 및 데이터 기반 접근 방식과 비교했을 때 뛰어난 성능을 보임을 입증하였습니다.

- **Technical Details**: MoD-DNN은 CNN(합성곱 신경망)과 희소 공액 기울기 알고리즘(SCG)을 활용한 반복 최적화 기법을 적용합니다. 연구는 5G 모바일 네트워크의 다양한 구현 가능성을 기반으로 하여, 하드웨어 손상의 영향을 완화하고 AoA 추정 성능을 증대하는 데 중점을 두었습니다. 이 연구는 기존의 단순 데이터 기반 방법의 한계를 극복하고, 하드웨어 환경의 변동성에 적응할 수 있는 새로운 가능성을 제시합니다.

- **Performance Highlights**: 제안된 MoD-DNN 방법은 스펙트럼 보정 및 AoA 추정을 향상시켜, 5G 새로운 전파 기술의 활용에 있어 선도적인 결과를 보입니다. 기존의 모델 기반 및 데이터 기반 방법들과 비교하여 더 높은 정확도를 나타내며, 특히 매칭된 5G NR 기지국 환경에서 AI 기반 위치 추정의 첫 사례로 주목받고 있습니다. 이 연구는 고정밀 위치 추정과 관련된 다양한 응용 프로그램에 중요한 기여를 할 것으로 기대됩니다.



### NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions (https://arxiv.org/abs/2501.00004)
- **What's New**: 이번 연구에서는 NewsHomepages라는 대규모 데이터셋을 통해 3,000개 이상의 뉴스 웹사이트의 홈페이지 레이아웃을 캡처하였습니다. 이 데이터셋은 3년 동안 하루에 두 번 수집된 자료로, 정보 우선순위 결정 과정을 분석하고자 합니다. 연구진은 이러한 데이터셋을 바탕으로 뉴스 아이템 간의 상대적 중요성을 유추하는 모델을 개발하였으며, 이를 통해 조직의 우선순위와 정보 구조의 관계를 탐색합니다.

- **Technical Details**: 뉴스 웹사이트의 홈 페이지 레이아웃은 전문 에디터에 의해 세심하게 선택된 정보 우선순위를 반영합니다. 연구에서는 HTML 스냅샷, 링크 및 추가 메타데이터를 포함해 363,000개의 페이지 스크린샷을 수집하였으며, 이를 통해 레이아웃 내에서 뉴스 기사의 위치를 정교하게 파악합니다. 또한, 페어와이즈 비교 모델을 개발하여 기사 사이의 상대적 중요성을 학습하고 예측합니다.

- **Performance Highlights**: 이 연구의 두 가지 실험을 통해 뉴스 가치 평가에서 예외적인 상관관계를 발견하였습니다. 특히, 다른 성향의 뉴스 아울렛 간에도 유사한 뉴스 가치 판단이 있음을 확인했습니다. 마지막으로, 이러한 발견은 디지털 환경에서의 정보 구조와 미묘한 암시가 인간의 인식에 미치는 영향을 더 깊이 이해할 수 있는 기초를 제공합니다.




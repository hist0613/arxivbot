New uploads on arXiv(cs.CL)

### Dialogue is Better Than Monologue: Instructing Medical LLMs via Strategical Conversations (https://arxiv.org/abs/2501.17860)
- **What's New**: 현재의 의료 AI 시스템은 주로 정적인 텍스트와 질문-답변 작업에 대한 훈련과 평가로 인해 실제 임상 사고(clinical reasoning)를 복제하는 데 실패하고 있습니다. 이 논문에서는 실제 진단 시나리오(diagnostic scenarios)를 시뮬레이션하는 새로운 벤치마크(benchmark)를 소개하며, 이를 통해 가장 중요한 증거 기반(reasoning) 사고와 주의 분산(distraction) 정보를 처리하는 방법을 포함합니다.

- **Technical Details**: 우리는 USMLE(United States Medical Licensing Examination) 기준에 맞춘 노이즈(noise)와 난이도(difficulty level) 조절을 통합하여 시뮬레이션된 진단 시나리오를 구현한 벤치마크를 개발했습니다. 또한 대화 기반(fine-tuning) 방법을 탐구하며, 정적(static) 데이터셋을 대화 형식(conversational formats)으로 변환하여 반복적 reasoning 과정을 더 잘 포착하고자 했습니다.

- **Performance Highlights**: 실험 결과, 대화로 조정(dialogue-tuned)된 모델은 전통적인 방법보다 우수한 성능을 보여주었으며, 다중 라운드(multi-round) reasoning 시나리오에서 9.64% 향상, 노이즈 환경(noisy environment)에서 6.18%의 정확성 향상을 달성했습니다. 이러한 발견은 대화 조정이 임상에 맞춘 강력한 의료 AI 시스템 발전을 위한 유망한 접근 방식임을 강조합니다.



### Improving Your Model Ranking on Chatbot Arena by Vote Rigging (https://arxiv.org/abs/2501.17858)
- **What's New**: 본 논문에서는 Chatbot Arena에서 대형 언어 모델(LLM)의 평가를 위한 투표 시스템에서 발생할 수 있는 조작 가능성을 다룹니다. 연구진은 특정 모델($m_{t}$)의 순위를 조작하기 위한 두 가지 전략을 제시했습니다. 첫 번째는 직접적으로 $m_{t}$가 포함된 투표에 집중하는 '타겟 전용(TARGET-ONLY) 조작' 전략입니다. 두 번째는 모든 투표에 영향을 미칠 수 있는 '전방위적인(OMNIPRESENT) 조작' 전략으로, 이는 보다 효과적으로 $m_{t}$의 순위를 조정할 수 있습니다.

- **Technical Details**: Chatbot Arena는 사용자들이 무작위로 선택된 두 개의 모델 간의 응답을 비교하여 투표하는 방식으로 설계되었습니다. 각 모델의 Elo 점수는 수집된 사용자 투표를 기반으로 계산되며, 이는 모델 순위의 변경을 가능하게 합니다. 특히, 새로운 투표가 이루어지면, 반드시 $m_{t}$ 모델이 직접 포함되지 않더라도 순위에 영향을 줄 수 있는 가능성이 있습니다. 연구팀은 약 170만 건의 투표 기록을 활용해 다양한 조작 시나리오를 실험했습니다.

- **Performance Highlights**: 실험 결과, 전방위적 조작 전략이 단 몇 백 건의 투표만으로도 모델의 순위를 크게 변경할 수 있음을 확인했습니다. 이는 타겟 전용 조작 전략에 비해 획기적으로 더 효율적이며, 기존의 여러 방어 메커니즘에도 불구하고 조작이 가능하다는 점을 부각시킵니다. 이 연구는 투표 조작의 방지를 위한 지속적인 노력의 중요성을 강조하며, 실제 Chatbot Arena 플랫폼에서 가능한 조작 방식에 대한 경각심을 제공합니다.



### Learning Beyond the Surface: How Far Can Continual Pre-Training with LoRA Enhance LLMs' Domain-Specific Insight Learning? (https://arxiv.org/abs/2501.17840)
- **What's New**: 이 연구에서는 LLMs의 인사이트 학습 능력을 지속적인 사전 훈련을 통해 향상시킬 수 있는 방법을 탐구합니다. 특히 의학과 금융 두 가지 도메인에서 LoRA (Low-Rank Adaptation)를 활용하여 기존 데이터셋으로 LLMs를 훈련합니다. 연구의 주요 중점은 문서 수정을 통해 핵심 정보를 유지한 형태가 인사이트 학습 능력을 어떻게 개선하는지에 대한 것입니다.

- **Technical Details**: 연구에서는 지속적인 사전 훈련과 LoRA를 통해 LLaMA 모델을 두 개의 도메인 특정 데이터셋인 Hallmarks of Cancer와 Buster에 적응시킵니다. 각 인사이트 유형을 평가하기 위해 GPT-4o 미니를 사용하여 정보의 트리플을 추출하고, 이를 통해 세 가지 유형의 인사이트(명제적, 통계적, 확률적)를 평가하기 위한 검증 세트를 수집합니다. 이러한 접근 방식은 원본 문서에서의 수익률이 미미하지만, 문서를 수정하여 필수 정보만을 유지하는 경우에는 인사이트 학습 능력을 크게 향상시키는 결과를 보여줍니다.

- **Performance Highlights**: 실험 결과, LoRA를 사용한 지속적인 사전 훈련을 받은 LLMs는 명제적 및 통계적 인사이트에서 미세한 개선을 보였으며, 확률적 인사이트는 더 적은 개선을 보였습니다. 그러나 핵심 지식만 포함된 수정된 문서로 훈련했을 때 모델의 인사이트 학습 능력이 크게 향상되었다는 것이 주요 발견입니다. 모델 크기가 클수록 인사이트 학습 능력이 향상되는 경향이 있으며, 이는 스케일러블한 인사이트 학습의 가능성을 강조합니다.



### A Comprehensive Survey on Legal Summarization: Challenges and Future Directions (https://arxiv.org/abs/2501.17830)
- **What's New**: 이 논문은 법적 도메인에서 자동 요약 기술, 데이터셋, 모델 및 평가 방법에 대한 체계적이고 최신의 조사를 제공합니다. 120편 이상의 논문을 검토하여 기존의 체계적 조사에서의 공백을 채웠습니다. 특히 법적 문서에 적합한 연구 트렌드, 도전 과제 및 기회들을 논의합니다.

- **Technical Details**: 법적 요약은 주로 지역별 법적 문서, 요약 전략, 요약 방법으로 나눌 수 있습니다. 저자는 추출적 (extractive), 생성적 (abstractive), 하이브리드 (hybrid) 방법을 포함한 다양한 요약 전략을 분석합니다. 또한, 다양한 요약 방법론에는 랭크 기반 (rank-based), 그래프 기반 (graph-based), 변환기 기반 (transformer-based) 등이 있습니다.

- **Performance Highlights**: 법적 문서 요약 분야는 중요한 발전을 이루었으나, 포괄적이고 시기적절한 조사가 부족한 실정입니다. 이 논문은 현재의 법적 요약 접근 방식에 대한 종합적인 조사를 제공하며, 법적 요약의 다양한 측면에서 중요한 한계 및 미래 연구 방향을 제안합니다. 윤변화인 법적 문서의 자동 요약은 효율성을 높일 수 있는 가능성이 큽니다.



### BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation -- Challenges and Insights (https://arxiv.org/abs/2501.17790)
- **What's New**: 이번 논문에서는 BreezyVoice라는 대만 만다린을 위한 텍스트 음성 변환(Text-to-Speech, TTS) 시스템을 소개합니다. 이 시스템은 음소 해석(polyphone disambiguation)의 독특한 문제를 해결하기 위해 음소 제어(phonetic control) 기능을 강조하고 있습니다. BreezyVoice는 CosyVoice를 기반으로 하여 $S^{3}$ tokenizer, 대규모 언어 모델(large language model, LLM), 최적 수송 조건부 흐름 매칭 모델(optimal-transport conditional flow matching model, OT-CFM) 등의 요소를 통합하여 현실감 있는 음성을 생성합니다.

- **Technical Details**: BreezyVoice 프레임워크는 Supervised Semantic Speech (S3) Tokenizer, 대규모 언어 모델(LLM), 최적 수송 조건부 흐름 매칭 모델(OT-CFM), 그리고 g2pW(grapheme to phoneme prediction model)로 구성되어 있습니다. 이 시스템은 각 요소의 상호작용을 통해 음성을 디지털 단위로 변환하고, 이를 통해 여러 표현 방식에서 음성이 자연스럽게 생성되도록 합니다. 특히, OT-CFM 모델을 활용하여 시간이 지남에 따라 의미 있는 데이터 분포로 변환되는 스펙트로그램을 생성하여 음성의 시간 주파수 구조를 정확하게 반영합니다.

- **Performance Highlights**: BreezyVoice는 일반 및 코드 스위칭(code-switching) 환경에서 상업 TTS 시스템에 비해 우수한 성능을 보여줍니다. 연구 결과, 이 시스템은 다루기 어려운 장기 스피커(long-tail speaker) 및 음소 해석(polyphone disambiguation) 문제를 특정적으로 해결함으로써 높은 충실도(high-fidelity)의 음성을 생성할 수 있음을 입증했습니다. 이러한 성과는 신경 코덱 TTS 시스템의 작동 원리에 대한 귀중한 통찰을 제공합니다.



### Reasoning Over the Glyphs: Evaluation of LLM's Decipherment of Rare Scripts (https://arxiv.org/abs/2501.17785)
Comments:
          7 pages, 3 figures

- **What's New**: 이번 연구는 Unicode로 인코딩되지 않은 희귀 스크립트를 해독하는 LVLM(large language vision models)과 LLM(large language models)의 능력을 탐구합니다. 새로운 다중 모달 데이터셋을 구축하는 접근법을 소개하며, 이러한 모델들이 언어 기호의 토큰화를 통해 해당 과제를 해결할 수 있도록 돕습니다. 연구는 GPT-4o, Gemini, Claude 3.5 Sonnet과 같은 주요 모델들을 대상으로 한 실험을 진행하였습니다.

- **Technical Details**: 이 연구에서는 Unicode로 인코딩할 수 없는 스크립트를 포함하는 언어 퍼즐의 다중 모달 데이터셋을 구축하기 위해 두 가지 접근법을 제안합니다. LVLM을 위해서는 Picture Method를, LLM을 위해서는 Description Method를 사용하여 시각 정보를 통합합니다. Glyph token 개념을 도입하여 불완전한 데이터셋과 전통적인 기법들이 놓친 측면들을 보완합니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 평균 40.0%의 정확도를 기록했지만, Gemini는 13.4%, Claude-3.5는 31.3%에 불과한 성과를 보였습니다. LVLM들이 복잡한 토큰의 특성을 정확하게 반영하기 어려웠고, 특히 기본 기하형태와 비슷한 토큰에서만 양호한 설명을 생성했습니다. 패턴 매칭과 추론 과정에서 많은 제한점이 발견되어, 이러한 연구는 LVLM의 시각적 구조 모델링 능력에 대한 추가 연구가 필요함을 강조합니다.



### 2SSP: A Two-Stage Framework for Structured Pruning of LLMs (https://arxiv.org/abs/2501.17771)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 구조적 가지치기를 위한 새로운 Two-Stage Framework for Structured Pruning (2SSP)을 제안합니다. 2SSP는 Width Pruning과 Depth Pruning이라는 두 가지 다른 가지치기 전략을 결합하여 모델의 성능 저하를 최소화하면서도 계산 비용을 줄이는 데 초점을 맞추고 있습니다. 첫 번째 단계는 중간 상태의 Feed-Forward Networks에서 신경망을 제거하고, 두 번째 단계는 Attention 서브모듈을 제거하는 방식으로 진행됩니다.

- **Technical Details**: 2SSP의 첫 번째 단계는 Width Pruning을 사용하여 각 신경망의 중요도를 평가한 후, 신경망을 제거하는 것입니다. 이 과정은 Transformer 블록의 구조적 연결성을 유지하면서 진행됩니다. 두 번째 단계에서는 깊이 가지치기를 통해 Attention 서브모듈을 제거하며, 이는 주어진 성능 지표(예: perplexity)를 최소화하는 방향으로 반복적으로 수행됩니다.

- **Performance Highlights**: 제안된 2SSP 방법은 네 가지 LLM 계열과 세 가지 가지치기 비율(25%, 37.5%, 50%)에 대해 테스트되었으며, 관련된 언어 모델링 데이터셋과 6개의 다운스트림 태스크에서 결과를 측정했습니다. 이 방법은 세 가지 언어 모델링 및 여섯 개 다운스트림 태스크에서 최근의 최첨단 방법들보다 일관되게 우수한 성능을 보여주었으며, 가지치기 시간 측면에서도 최대 두 배의 이득을 기록하였습니다.



### Hybrid Graphs for Table-and-Text based Question Answering using LLMs (https://arxiv.org/abs/2501.17767)
Comments:
          Accepted at NAACL 2025 Main Track

- **What's New**: 이번 논문에서는 기존의 고품질 데이터에 의존하지 않고 대규모 언어 모델(Large Language Models, LLMs)을 이용한 새로운 하이브리드 그래프 기반 접근 방식을 제안합니다. 이 방법은 텍스트 및 표 데이터를 조합하여 하이브리드 그래프(Hybrid Graph)를 구성하고, 입력된 질문에 따라 정보의 중요성을 조정하여 LLM에 관련된 맥락을 제공합니다. 이는 멀티 소스 테이블-텍스트 질문 답변(Table-Text QA) 분야에서의 가능성을 보여주는 중요한 발전입니다.

- **Technical Details**: 제안하는 방법은 하이브리드 그래프를 통해 텍스트 데이터와 테이블 데이터를 통합하고, 입력 질문에 따라 불필요한 정보를 제거하는 방식으로 작동합니다. 이 접근법은 별도의 파인튜닝(fine-tuning) 과정 없이도 LLM을 효과적으로 활용할 수 있습니다. 평가에는 최신 LLM인 GPT-3.5, GPT-4, LLaMA-3를 사용하였고, Hybrid-QA와 OTT-QA라는 어려운 데이터셋에서 성능을 측정하였습니다.

- **Performance Highlights**: 제안된 방법은 두 데이터셋에서 제로샷(zero-shot) 성능에서 가장 우수한 결과를 보였으며, Hybrid-QA에서 Exact Match 점수가 10% 증가하고 OTT-QA에서 5.4% 향상되었습니다. 또한, 이 접근법은 원래의 맥락에 비해 토큰 사용(Token usage)을 최대 53% 줄이는 효과를 보였습니다.



### RICoTA: Red-teaming of In-the-wild Conversation with Test Attempts (https://arxiv.org/abs/2501.17715)
Comments:
          PACLIC 38

- **What's New**: 이 논문은 RICoTA라는 새로운 한국어 레드 팀 데이터셋을 소개합니다. 이 데이터셋은 사용자가 만든 대화로, 대화형 에이전트와의 상호작용에서 존경심을 넘어서는 시도를 포함하여 "탈옥(jailbreaking)" 시도를 캡처합니다. 609개의 프롬프트가 포함되어 있으며, 이는 사용자 대화에서 발생한 실제 상호작용을 반영합니다.

- **Technical Details**: 데이터셋은 한국의 리다(Luda)라는 소셜 챗봇과의 대화에서 추출된 사용자 대화 스크린샷을 기반으로 합니다. OCR 기법을 활용하여 이미지 형식의 스크린샷을 평문 텍스트로 변환하고, 이후 텍스트 정제와 대화 형식화를 진행하여 총 609개의 대화 데이터 포인트를 생성했습니다. 각 데이터 포인트는 대화 유형과 테스트 목적 등을 포함한 두 개의 라벨로 주석을 달았습니다.

- **Performance Highlights**: RICoTA 데이터셋은 LLM이 탈옥 시도를 포함하는 대화 유형과 테스트 목적을 정확히 식별할 수 있는 능력을 평가하게 합니다. 이는 사회적 채팅 안전성을 평가하는 새로운 방법을 제안하며, 챗봇 설계자들에게 사용자 테스트 목적의 잠재적 사용을 자기검토하는 데 유용합니다. 해당 연구의 결과는 GitHub을 통해 공개될 예정입니다.



### Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imita (https://arxiv.org/abs/2501.17703)
- **What's New**: 이번 논문에서는 기존의 Supervised Fine-Tuning(SFT) 접근법을 도전하며 Critique Fine-Tuning(CFT)이라는 새로운 전략을 제안합니다. CFT는 모델이 올바른 결과를 단순히 모방하는 대신, 노이즈가 있는 응답을 비판하고 분석하는 방식을 통해 더 깊이 있는 학습을 유도합니다. 이는 인간의 학습 프로세스에서 중요한 비판적 사고를 강조하여, 표준 SFT에서는 간과되기 쉬운 nuanced understanding(세밀한 이해)를 촉진합니다.

- **Technical Details**: CFT의 효과를 검증하기 위해 WebInstruct에서 50K 샘플의 데이터셋을 구축하였고, 이를 통해 모델이 응답을 비판하는 방식으로 학습하도록 하였습니다. CFT에서 모델은 주어진 쿼리-응답 쌍에 대한 비판을 생성하는 것을 목표로 하며, 이를 통해 모델의 비판적 사고 능력을 향상시킬 수 있습니다. 모델은 Qwen2.5-Math-CFT와 같은 다양한 언어 모델을 사용하여 실험하였고, CFT 방식으로 학습한 모델들이 SFT보다 4-10% 높은 성능 향상을 보였습니다.

- **Performance Highlights**: CFT 학습을 통해 생성된 Qwen2.5-Math-CFT 모델은 50K 예제로 학습했음에도 불구하고 2M 이상의 샘플로 학습한 경쟁 모델들인 AceMath 및 Qwen2.5-Math-Instruct와 비교하여 유사한 또는 더 나은 성능을 보였습니다. 실험 결과, CFT가 STEM 분야의 다양한 벤치마크에서 SFT에 비해 지속적인 성능 개선을 이루었으며, 이는 비판 기반 학습의 효과성을 입증합니다.



### Exploring Vision Language Models for Multimodal and Multilingual Stance Detection (https://arxiv.org/abs/2501.17654)
Comments:
          Submitted to the International AAAI Conference on Web and Social Media (ICWSM) 2025

- **What's New**: 이 논문은 멀티모달 환경과 다국어 데이터를 포함하는 스탠스 감지(stance detection) 작업에서 최신 비전-언어 모델(Vision-Language Models, VLMs)의 성능을 평가합니다. 특히, 이전 연구가 텍스트 전용 데이터에 초점을 맞춘 것과 달리, 이미지와 텍스트를 동시에 사용하는 상황에 대한 연구가 부족한 점을 지적합니다. 다양한 언어로 이루어진 최근에 확장된 데이터셋을 바탕으로 VLM의 시각적 단서 및 언어별 성능을 평가하고 있습니다.

- **Technical Details**: 스탠스 감지란 특정 주제, 개체 또는 주장에 대한 사용자의 관점을 자동으로 분류하는 작업입니다. 이 논문에서는 VLM이 스탠스 감지를 수행하는 데 있어 텍스트와 이미지 정보를 얼마나 효과적으로 사용하는지를 분석하며, 다국어에서의 성능 또한 조사합니다. 이를 통해 VLMs의 텍스트와 이미지 간의 상호작용을 다국어 맥락에서 탐구하며, 데이터셋은 영어뿐만 아니라 6개 추가 언어로 확장된 멀티모달 스탠스 감지 데이터셋을 사용합니다.

- **Performance Highlights**: 실험 결과, VLMs는 스탠스 감지를 위해 일반적으로 이미지보다 텍스트에 더 의존하는 경향이 있으며, 이 경향은 모든 언어에 걸쳐 지속됩니다. 또한, VLM은 이미지 내의 텍스트에 대해 다른 시각적 콘텐츠보다 더 크게 의존하는 것으로 나타났습니다. 모델들이 명시적으로 다국어를 지향하지 않더라도, 여러 언어에서 일관된 예측을 생성하는 경향이 있지만, 전반적인 F1 점수, 언어 지원, 모델 크기와 불일치하는 이상치도 존재합니다.



### Tonguescape: Exploring Language Models Understanding of Vowel Articulation (https://arxiv.org/abs/2501.17643)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 연구에서는 언어 모델 (LM)이 실제 혀의 위치와 모음 발음을 이해할 수 있는지에 대한 질문을 다루고 있습니다. 연구팀은 기존의 실시간 MRI 데이터셋에서 영상과 이미지를 생성하여 LMs의 시각 기반 정보 활용 가능성을 검토했습니다. 연구 결과, LMs는 레퍼런스 예제가 제공될 경우 모음과 혀 위치를 이해할 잠재력을 보여주지만, 레퍼런스가 없는 경우 어려움이 있는 것으로 나타났습니다. 이 연구의 코드와 데이터셋 구축 자료는 GitHub에서 제공됩니다.

- **Technical Details**: 모음 발음을 이해하기 위해, 연구팀은 일본어 5모음 시스템을 중심으로 한 데이터셋을 구성하였습니다. 데이터셋은 22명의 일본어 사용자로부터 수집된 실시간 MRI 비디오로 구성되어 있으며, 각 모음의 발음을 촬영한 120개의 비디오를 포함하고 있습니다. 이 비디오는 혀와 인두의 움직임을 관찰할 수 있는 측면 뷰를 제공하며, 각 비디오는 정지 상태에서 시작하여 정지 상태로 끝납니다. LMs는 길게 사용된 다중 모달 정보 방식으로 텍스트뿐만 아니라 이미지, 비디오 및 오디오 정보를 처리할 수 있어야 합니다.

- **Performance Highlights**: LM들은 주어진 이미지에서 모음과 혀 위치를 예측하는 데 일부 능력을 보였지만, 제로샷(Zero-shot) 및 몇 샷(Few-shot) 학습에서는 어려움을 겪었습니다. 연구의 결과는 LMs가 언어의 음성과 관련된 아이디어를 보다 인간적인 생리학에 기반하여 이해하는 데 기여할 수 있음을 시사합니다. 이러한 통찰력은 모음 조화와 같은 언어 현상 분석에 기여할 수 있으며, 발음 교육에 있어서도 유용할 것으로 기대됩니다.



### In-Context Meta LoRA Generation (https://arxiv.org/abs/2501.17635)
- **What's New**: 본 연구에서는 다중 작업 환경에서도 효율성을 유지하면서, 대형 언어 모델(LLMs)의 작업 특화(customization)를 위한 In-Context Meta LoRA (ICM-LoRA)라는 혁신적인 접근 방식을 제안합니다. 기존의 Low-rank Adaptation (LoRA) 모델이 개별 작업에 대해 따로 훈련되는 비효율성을 해결하기 위해, Conditional Variational Autoencoder (CVAE)를 활용하여 task-aware LoRA weights를 생성합니다. ICM-LoRA는 저장 용량을 감소시키면서도 다양한 작업에 대한 정확한 LoRA 매개변수를 생성할 수 있습니다.

- **Technical Details**: ICM-LoRA는 task vectors를 사용하여 context modeling을 수행하며, 메타 학습(meta-learning)을 통해 작업 간의 관계를 포착합니다. 이를 통해 LoRA 매개변수 생성을 단일 생성자(generator)로 동시에 처리할 수 있는 그리드(grid) 생성이 가능해집니다. 또한 ICM-LoRA는 데이터를 추가로 필요로 하지 않고, 단 283MB의 저장 용량만 차지하여 기존 LoRA에 비해 1%의 저장소 사용량을 기록합니다.

- **Performance Highlights**: ICM-LoRA는 다양한 모델에서 텍스트 및 비주얼 작업을 평가하였으며, CVAE가 여러 작업에 대해 LoRA 매개변수를 성공적으로 생성할 수 있음을 보여줍니다. 기존 방법들과 비교해, 생성된 LoRA 매개변수는 정확도 손실이 적으며, 원본 데이터 세트와 LoRA 가중치에 비해 저장 용량을 크게 줄일 수 있습니다. 결과적으로 본 연구의 접근법은 여러 작업에 대한 LoRA 파라미터 생성을 보다 효율적이고 정확하게 수행할 수 있음을 입증합니다.



### Structured Context Recomposition for Large Language Models Using Probabilistic Layer Realignmen (https://arxiv.org/abs/2501.17617)
- **What's New**: 이 연구에서는 Structured Context Recomposition (SCR)라는 새로운 프레임워크를 소개하여 트랜스포머 아키텍처 내에서 확장된 시퀀스 간의 맥락 일관성을 유지하는 방법을 제안합니다. SCR은 전통적인 토큰 수준 주의 분포를 조정하는 방법과는 달리, 모델의 내부 레이어에서 주요 맥락 요소가 지속적으로 유지될 수 있도록 계층적 표현을 재구성합니다. 이를 통해 전통적인 메모리 기반 접근 방식과 검색 증강 방법의 한계를 극복하고자 합니다.

- **Technical Details**: SCR은 확률적 레이어 재정렬 전략을 사용하여, 학습된 표현을 동적으로 조정하여 정의된 맥락 적합도에 따라 강조를 재분배합니다. 이러한 새로운 접근방법은 주요 맥락 요소의 완전성을 유지하면서도 계산 비용을 최소화하도록 설계되었습니다. 전통적인 위치 인코딩 개선 방법과는 달리, SCR은 의미적으로 중요한 임베딩을 강화하면서 덜 중요한 정보는 약화하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, SCR은 급격한 주제 전환과 논리적 불일치를 줄이는 데 효과적임을 보여주었습니다. 시퀀스 수준의 엔트로피 분석에서는 SCR이 표현의 변동성을 조절하면서도 지나치게 출력 정규화를 도입하지 않아 모델이 생성적 다양성을 지속할 수 있도록 함을 나타냅니다. 또한, 계산 리소스 평가에서는 SCR이 처리 시간에 적당한 증가를 초래하지만, 메모리 오버헤드는 허용 가능한 한도 내에서 유지된다는 점에서 실용적인 배치를 갖춘 것으로 확인되었습니다.



### Cross-lingual Embedding Clustering for Hierarchical Softmax in Low-Resource Multilingual Speech Recognition (https://arxiv.org/abs/2501.17615)
- **What's New**: 이 논문에서는 Automatic Speech Recognition (ASR)의 디코딩 단계에 초점을 맞춘 새로운 접근 방식을 제시합니다. 특히 저자원 언어(low-resource languages)의 다국어 성능을 향상시키는데 중점을 두었습니다. 이를 위해 cross-lingual embedding clustering 방법을 활용하여 계층적 Softmax (H-Softmax) 디코더를 구성합니다.

- **Technical Details**: H-Softmax 디코더는 서로 다른 언어 간에 유사한 토큰이 유사한 디코더 표현을 공유할 수 있도록 합니다. 이는 기존의 Huffman 기반 H-Softmax 방법의 한계를 해결하며, 토큰 유사성 평가에서의 얕은 특징(shallow features) 의존성을 극복합니다. 이 연구는 15개 언어의 다운샘플된 데이터셋에 대한 실험을 통해 제안한 접근 방식의 효과성을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 저자원 다국어 ASR의 정확성을 향상시키는 데 있어 새로운 접근 방식이 매우 효과적임을 보여주었습니다. 여러 언어에서 토큰 간의 유사성을 고려함으로써, 기존 방법보다 더 나은 성능을 발휘합니다. 이는 특히 자원이 부족한 언어에서 더욱 두드러진 성과를 보였습니다.



### Semantic Consistency Regularization with Large Language Models for Semi-supervised Sentiment Analysis (https://arxiv.org/abs/2501.17598)
Comments:
          ICONIP 2024

- **What's New**: 이 연구에서는 대규모 라벨링된 데이터셋의 부족 문제를 해결하기 위해, Large Language Models (LLMs)를 활용한 새로운 반지도 학습(framework)인 Semantic Consistency Regularization (SCR)을 제안합니다. SCR은 두 가지 방법인 Entity-based Enhancement (SCR-EE)와 Concept-based Enhancement (SCR-CE)를 통해 비라벨 데이터의 의미론적 일관성을 향상시키며, 이를 통해 더욱 효율적인 학습을 가능하게 합니다. 이는 모델이 불확실한 비라벨 데이터에서 더 유용하고 다양한 정보를 추출할 수 있도록 도와줍니다.

- **Technical Details**: SCR은 LLMs의 강력한 텍스트 생성 및 이해 능력을 활용하여 비라벨 텍스트를 의미론적으로 일관된 변형으로 증강합니다. SCR-EE는 텍스트로부터 엔티티 및 숫자 정보를 추출해 LLM으로 텍스트를 재구성하는 방법을 사용하고, SCR-CE는 원본 문장을 직접 쿼리하여 의미론적으로 일관된 변화를 생성합니다. 여기서 훈련 중에 고품질 일치 샘플을 체계적으로 포함하기 위해 신뢰도 임계값을 적용한 일관성 손실(consistency loss)을 활용합니다.

- **Performance Highlights**: 실험 결과, SCR 프레임워크는 두 가지 감정 분석 데이터셋에서 기존 반지도 학습 방법들보다 월등한 성능을 보여주었습니다. SCR은 특히 기존의 방법들보다 더욱 다양한 레이블링 체계에 대해 효과적으로 작동하는 것으로 나타났으며, 이는 의미적으로 향상된 데이터가 모델 학습에 긍정적인 영향을 미쳤음을 나타냅니다. 전체적으로 SCR의 제안된 접근 방식은 반지도 학습을 위한 새로운 기준을 제시하며, 감정 분석 분야에서의 성과 개선을 달성합니다.



### CSEval: Towards Automated, Multi-Dimensional, and Reference-Free Counterspeech Evaluation using Auto-Calibrated LLMs (https://arxiv.org/abs/2501.17581)
Comments:
          17 pages, 5 figures. arXiv admin note: text overlap with arXiv:2309.13308 by other authors

- **What's New**: 이 연구에서는 자동 반대 연설 생성의 품질을 평가하기 위한 새로운 데이터셋인 CSEval과 네 가지 품질 차원을 평가하는 프레임워크를 제안합니다. 이 프레임워크는 기존의 표면 수준 유사성 측정 방법의 한계를 극복하고, 인간의 판단과 일치하는 더 정교한 평가를 가능하게 합니다. CSEval은 맥락 관련성(contextual relevance), 공격성(aggressiveness), 주장 일관성(argument-coherence), 적합성(suitableness)이라는 네 가지 차원으로 카운터스피치 품질을 평가하도록 설계되었습니다.

- **Technical Details**: CSEval은 자동화된 카운터스피치 생성 평가를 위해 네 가지 주요 품질 차원에 대한 인간의 판단을 포함한 대규모 데이터셋입니다. 또한, 자동 보정된 Chain-of-Thought (CoT) 방식을 활용한 ACE 방안을 제안하여 카운터스피치의 각 품질 차원을 점수화하는 방법을 제시합니다. ACE는 기존의 BLEU, ROUGE, METEOR와 같은 전통적인 지표보다 인간의 평가와 더 높은 상관관계를 보여줍니다.

- **Performance Highlights**: 실험 결과, ACE는 기존의 자동 평가 방법과 LLM 기반 방법들보다 인간의 평가와 더 높은 상관관계를 가지며, 카운터스피치 품질 평가에 있어 큰 진전을 나타냈습니다. 이 연구는 카운터스피치 생성을 다루는 연구들에서 더 개선되고 효과적인 평가 방법을 제안하는 중요한 기초 자료로 작용할 것입니다. CSEval 데이터셋과 소스 코드는 공개되어 연구자들이 쉽게 접근하여 활용할 수 있게 되었습니다.



### A linguistically-motivated evaluation methodology for unraveling model's abilities in reading comprehension tasks (https://arxiv.org/abs/2501.17569)
- **What's New**: 이 연구에서는 특정 예제가 언어적 복잡성 때문에 모델의 성능에 지속적으로 부정적인 영향을 미친다는 직관에 기반한 독해 과제 평가 방법론을 도입합니다. 이 방법론은 의미적 프레임 주석(semantic frame annotation)을 활용하여 복잡성을 특성화하며, 모델이 어려움을 겪는 7가지 복잡성 요소를 분석합니다.

- **Technical Details**: 연구진은 사전 주석이 달린 프랑스어 독해 벤치마크에서 이 방법론을 적용하여 두 개의 복잡성 요소가 모델의 실패를 예측하는 데 유용하다는 것을 발견했습니다. 이후 이 방법론을 영어 벤치마크에서도 적용하였으며, Chat-GPT를 의미적 주석의 대리(proxy)로 사용했습니다.

- **Performance Highlights**: 연구 결과, 언어적 특성을 반영한 세밀한 자동 평가가 가능하다는 것을 보여주었으며, 이는 모델이 특정 언어적 특성을 처리할 수 있는 능력을 이해하는 데 도움을 줍니다. 또한 최신 모델들이 특정 언어적 특성을 처리하는 데 기존의 모델 사이즈 증가만으로는 부족하다는 것을 나타내었습니다.



### Query-Aware Learnable Graph Pooling Tokens as Prompt for Large Language Models (https://arxiv.org/abs/2501.17549)
- **What's New**: 이 논문에서는 Learnable Graph Pooling Token (LGPT)이라는 새로운 접근 방식을 제안합니다. LGPT는 노드 수준 프로젝션의 확장성 문제와 그래프 수준 프로젝션에서의 정보 손실 문제를 해결하며, 학습 가능한 파라미터를 통해 그래프 정보를 보다 효율적으로 표현합니다. 또한, Early Query Fusion 기법을 도입하여 쿼리 컨텍스트를 그래프 표현을 구성하기 전에 결합함으로써 보다 효과적인 그래프 임베딩을 구현합니다.

- **Technical Details**: 제안된 LGPT는 그래프 정보를 여러 개의 학습 가능한 파라미터를 통해 LLM(대형 언어 모델)에 전달하는 방식을 통해 노드 수준과 그래프 수준의 적절한 투영을 가능하게 합니다. 이는 그래프에서 복잡한 문맥을 단일 벡터로 변환할 때 발생하는 정보 손실 문제를 해결합니다. 추가적으로, Early Query Fusion 기법을 통해 쿼리와 그래프 정보의 결합을 그래프 표현 구성 이전에 수행하여 더 나은 임베딩을 제공합니다.

- **Performance Highlights**: LGPT 방법은 GraphQA 벤치마크에서 4.13%의 성능 향상을 달성했으며, LLM을 별도로 훈련시키지 않고도 복잡한 텍스트 속 그래프 데이터를 효과적으로 처리할 수 있음을 보여줍니다. 이 성능 향상은 쿼리 컨텍스트를 그래프 노드의 임베딩 구성 전에 통합하여 이루어진 것으로, 후에 결합하는 것보다 개선된 결과를 보고합니다.



### DINT Transformer (https://arxiv.org/abs/2501.17486)
Comments:
          arXiv admin note: text overlap with arXiv:2410.05258 by other authors

- **What's New**: DINT Transformer는 DIFF Transformer의 한계를 극복하기 위해 differential-integral 메커니즘을 도입하여 글로벌 중요성을 평가하고 주의(Attention) 매트릭스에 통합합니다. 이 모델은 글로벌 의존성을 캡처할 수 있는 능력을 향상시키며, 수치적 안정성을 보강하기 위해 행 정규화(row normalization)를 확립하여 자주 사용되는 attentional 맥락의 소음을 줄입니다. DINT Transformer는 긴 시퀀스의 언어 모델링과 핵심 정보 검색과 같은 다양한 실제 응용 프로그램에서 성능을 입증합니다.

- **Technical Details**: DINT Transformer는 한 개 이상의 stacked layer로 구성되어 있으며, 각 layer는 DINT attention 모듈이 적용된 뒤 feedforward network가 뒤따릅니다. 모델 입력은 X0∈ℝN×d_model이고, 각 레이어를 통해 점진적으로 변환되어 최종 출력 XL로 이어집니다. 핵심 혁신은 주의 모듈 내에 적분 메커니즘을 추가하여 글로벌 의존성을 모델링하고 수치적 안정성을 유지하는 것입니다.

- **Performance Highlights**: DINT Transformer의 실험 결과는 모델이 특히 길어진 시퀀스 작업에서 DIFF Transformer와 기존 Transformer 모델을 일관되게 초과하는 성능을 보여줍니다. 이 모델은 핵심 정보 검색과 같은 다운스트림 작업에서 효율성과 확장성을 유지하며 성능을 향상시키는 것으로 나타났습니다. DINT Transformer는 향후 시퀀스 모델링 및 큰 언어 모델 개발의 강력하고 효율적인 토대를 제공합니다.



### Cross-Language Approach for Quranic QA (https://arxiv.org/abs/2501.17449)
- **What's New**: 이 연구는 언어 자원이 제한된 언어를 대상으로 한 질문 응답 시스템의 주요 한계를 극복하기 위한 새로운 접근 방식을 제시합니다. 특히, 고전 아랍어로 작성된 꾸란 구절과 현대 표준 아랍어(MSA)로 작성된 질문 간의 언어적 격차를 해소하기 위해 교차 언어 접근 방식이採用되었습니다. 데이터 세트를 기계 번역을 통해 영어로 확장하고, 다양한 질문 표현을 생성하며, 꾸란의 영어 번역에서 답변을 찾는 방식으로 모델 성능을 향상시킵니다.

- **Technical Details**: 이 연구는 데이터 세트 준비, 교차 언어 처리, 모델 미세 조정의 세 가지 주요 단계를 포함합니다. 데이터 세트는 아랍어 질문을 영어로 번역하고, 문맥에 맞는 질문을 생성하기 위해 재구 성 및 추가로 문제를 생성하여 다양성을 향상시켰습니다. 또한 BERT-Medium, RoBERTa-Base, DeBERTa-v3-Base, ELECTRA-Large, Flan-T5, Bloom 및 Falcon과 같은 최첨단 모델을 미세 조정하여 꾸란 QA 작업의 성과를 최적화합니다.

- **Performance Highlights**: 실험 결과, RoBERTa-Base는 MAP@10(0.34) 및 MRR(0.52)에서 가장 높은 성과를 나타냈으며, DeBERTa-v3-Base는 Recall@10(0.50)과 Precision@10(0.24)에서 우수한 결과를 기록했습니다. 이러한 성과는 다양한 현대 아랍어 질문에 답하기 위해 필요한 고전 아랍어 구절을 효과적으로 캡처하고 언어적 장벽을 극복하는 교차 언어 접근 방식의 효과를 증명합니다.



### Actions Speak Louder than Words: Agent Decisions Reveal Implicit Biases in Language Models (https://arxiv.org/abs/2501.17420)
- **What's New**: 이 연구는 대형 언어 모델(LLM)에서 발생하는 잠재적 편향(implicit biases)을 발견하기 위한 새로운 방법론을 제안합니다. 해당 기술은 소시오데모그래픽 특성에 따라 만들어진 역할 성격(perseon)을 가진 언어 에이전트(language agents)가 다양한 의사결정 시나리오에서 수행하는 행동을 분석합니다. 이를 통해 모델이 직간접적으로 드러내는 편향을 명확히 구별할 수 있게 됩니다.

- **Technical Details**: 연구는 언어 에이전트의 행동을 ML모델의 반응과 대조하여 편향을 평가하는 두 단계의 과정을 통해 진행됩니다. 첫 번째 단계에서는 LLM을 사용하여 특정 소시오데모그래픽 속성에 기반하여 캐릭터를 생성하고, 두 번째 단계에서는 이러한 캐릭터들이 주어진 의사결정 시나리오에 대해 반응하도록 합니다. 이 접근법은 다양한 소시오데모그래픽 카테고리와 시나리오에서의 편향을 보다 체계적으로 조사할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들이 거의 모든 시뮬레이션에서 상당한 소시오데모그래픽 편차를 보였으며, 보다 진보한 모델들이 명시적 편향(explicit biases)은 줄여도 여전히 더 큰 잠재적 편향을 보였습니다. 본 연구는 또한 발견된 편향이 실제 세계에서 관찰되는 불평등 정도와 방향이 일치하지만, 그 강도는 확대된 것으로 나타났습니다. 이는 LLM이 시뮬레이션에서의 행동과 관련된 내재적 편향을 발견하는 데 이 기술의 유용성을 시사합니다.



### MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs (https://arxiv.org/abs/2501.17399)
- **What's New**: MultiChallenge는 다중 발화 대화에서 대형 언어 모델(LLMs)의 성능을 평가하는 새로운 기준을 제시합니다. 이 벤치마크는 현재 대형 언어 모델들이 직면한 네 가지 도전 과제를 식별하며, 이는 인간과의 상호작용에서 일반적이면서도 현실적인 문제들입니다. 다중 발화 대화에 대한 포괄적인 평가 프레임워크가 부족했던 것을 해결하기 위해 설계된 MultiChallenge는 LLM이 사용자의 요구에 적절히 응답할 수 있는 능력을 평가하는 데 중점을 둡니다.

- **Technical Details**: MultiChallenge는 최대 10턴의 대화 히스토리를 기반으로 하며, 각각의 테스트 예시는 대화의 최종 사용자 발화에 대한 요구 사항/질문으로 끝납니다. 네 가지 도전 과제는 1) 지침 유지, 2) 사용자 정보의 추론 기억, 3) 신뢰할 수 있는 버전 편집, 4) 자기 일관성으로 정의됩니다. 이 과제들은 모두 LLM이 지침을 따르고, 적절하게 문맥을 할당하며, 문맥 내 추론을 수행하는 능력을 요구합니다.

- **Performance Highlights**: 기존의 다중 발화 평가 벤치마크에서는 거의 완벽한 점수를 기록했음에도 불구하고, 현재의 최첨단 LLM들은 MultiChallenge에서 50% 미만의 정확도를 기록했습니다. 이 중에서 Claude 3.5 Sonnet (2024년 6월)은 평균 41.4%의 정확도로 가장 우수한 성과를 보여주었습니다. 연구에서는 인간 평가자와 자동 평가 간의 높은 상관관계를 보여주어 빠르고 정확한 다중 발화 평가의 잠재력을 입증했습니다.



### Leveraging In-Context Learning and Retrieval-Augmented Generation for Automatic Question Generation in Educational Domains (https://arxiv.org/abs/2501.17397)
Comments:
          Accepted at the 16th Meeting of the Forum for Information Retrieval Evaluation as a Regular Paper

- **What's New**: 이 연구에서는 교육적 문맥에서의 질문 생성(Question Generation) 개선을 위한 새로운 기법들을 탐구합니다. 특히, In-Context Learning (ICL)과 Retrieval-Augmented Generation (RAG) 기법을 결합한 하이브리드 모델을 제안하며, 이를 통해 보다 양질의 질문을 생성하고자 합니다. 연구 결과, 하이브리드 모델과 ICL 접근법이 기존의 질문 생성 방법들보다 더 높은 성능을 보임을 확인하였습니다.

- **Technical Details**: 본 연구에서는 GPT-4 모델을 ICL에 활용하고, BART 모델과 검색 모듈을 결합하여 RAG를 구현하였습니다. 하이브리드 모델은 RAG의 외부 정보 검색 기능과 ICL의 적은 샘플 학습 메커니즘을 통합하여 질문의 질을 향상시킵니다. 연구에서 제안된 방법은 BLEU-4, ROUGE-L, METEOR 등의 자동화된 메트릭과 교육자에 의한 인적 평가를 통해 평가되었습니다.

- **Performance Highlights**: ICL 접근법과 하이브리드 모델은 각각 텍스트북 구절에서 질문을 생성하는 과정에서 기존 모델들에 비해 일관되게 더 우수한 성능을 보였습니다. 결과적으로, 제안된 모델들은 문맥적으로 정확하고 학습 목표에 적합한 질문을 생성함으로써 교육적 가치가 높아짐을 입증하였습니다.



### Context-Aware Semantic Recomposition Mechanism for Large Language Models (https://arxiv.org/abs/2501.17386)
- **What's New**: 최근 자연어 처리 분야에서, Context-Aware Semantic Recomposition Mechanism (CASRM)이 도입되어 언어 생성 모델의 의미적 일관성과 맥락 적응성을 높이는 새로운 프레임워크로 주목받고 있습니다. CASRM은 동적으로 생성되는 컨텍스트 벡터와 주의 조절 레이어를 통합하여 대규모 텍스트 생성 작업의 일관성 문제와 오류 전파를 해결하고자 합니다. 다양한 실험 평가를 통해 기술적, 대화형, 서사적 텍스트를 포함한 여러 도메인에서 의미적 일관성이 크게 향상된 결과를 보여주었습니다.

- **Technical Details**: CASRM의 메커니즘은 모델 내에서 의미적 표현을 동적으로 조정하여 맥락에 대한 복잡한 이해를 가능하게 합니다. 컨텍스트 벡터가 주의 메커니즘에 통합되어 입력 데이터의 정제된 처리를 지원하고, 이론적 기반은 맥락 종속 의미 해석을 바탕으로 합니다. 최종적으로 CASRM 모듈은 기존 배치 모델에 보조 처리 유닛으로 포함되며, 입력 시퀀스를 처리하여 동적으로 발전하는 컨텍스트 벡터를 추출합니다. 이 비율로 조정된 주의 가중치는 토큰 간의 중요 분포를 개선하여 다양한 맥락에서의 의미적 일관성을 높입니다.

- **Performance Highlights**: 실험 결과 CASRM은 언어 생성 성능을 극적으로 향상시켰으며, 특히 대화 연속성과 다단계 텍스트 합성에서 오류 전파를 성공적으로 완화했습니다. 또한, 미지의 도메인과 모호한 입력에 대한 적응 능력이 입증되어 시스템의 강인성을 강조합니다. CASRM은 LM 성능의 현재 한계를 극복할 수 있는 유망한 방향성을 제공하며, 더 신뢰할 수 있고 맥락을 인식하는 언어 모델의 발전에 기여할 수 있을 것으로 기대됩니다.



### Better Slow than Sorry: Introducing Positive Friction for Reliable Dialogue Systems (https://arxiv.org/abs/2501.17348)
- **What's New**: 본 연구는 대화 시스템에 긍정적인 마찰(positive friction)을 통합하여 사용자 반성을 촉진하고, 시스템 반응에 대한 비판적 사고를 유도하는 새로운 접근법을 제안합니다. 기존의 대화 시스템이 마찰을 최소화하는 경향이 있는 반면, 긍정적인 마찰은 사용자의 목표와 숨은 가정을 다시 고려하게 하며 AI 시스템의 재조정을 가능하게 합니다. 이러한 마찰을 통해 대화의 흐름을 적절하게 조절함으로써 사용자와 AI 간의 상호작용을 향상시킬 수 있습니다.

- **Technical Details**: 연구는 여러 가지 대화 데이터셋에서 긍정적인 마찰의 개념을 바탕으로 한 새로운 다중 모달 분류학(taxonomy)을 개발했습니다. 이 분류학은 대화에 있어서 마찰의 유형을 정의하며, 인간-기계 상호작용에서 더 효과적인 반성을 장려하기 위해 행동 과학 자료를 활용합니다. 또한, 연구팀은 사용자 목표 모델링(user goal modeling)과 사용자 정신 상태(shows user mental states)를 개선하기 위해 대화 속도를 전략적으로 늦추는 방법을 탐구합니다.

- **Performance Highlights**: 실험 결과, 긍정적인 마찰을 적용한 대화에서는 사용자 목표 모델링과 전반적인 작업 성공률(task success rate)이 향상된 것으로 나타났습니다. 이러한 마찰은 결정을 내리는 과정에서 AI의 사용자 신념 및 목표 이해도를 높이며, 필요한 대화 횟수를 줄이는 데 기여합니다. 이 연구는 대화 시스템의 설계를 단기적인 효율성보다 장기적인 협력을 우선시하도록 전환할 필요성을 강조합니다.



### Inferring from Logits: Exploring Best Practices for Decoding-Free Generative Candidate Selection (https://arxiv.org/abs/2501.17338)
- **What's New**: 이 논문에서는 기존의 decoding-free candidate selection 방법들의 효과를 포괄적으로 평가합니다. 특히, 여러 가지 서로 다른 작업에 대해 이들 방법을 적용하여 성능을 분석하였습니다. 주요 연구는 여러 선택지들 중에서 task-level output을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 다양한 foundation LMs (Language Models) 아키텍처와 크기를 포함한 모델을 사용하여 후보 선택 방법의 성능을 평가했습니다. 이 과정에서 5개의 multiple-choice QA 작업과 4개의 임상 결정 작업을 포함한 다양한 데이터세트를 활용했습니다. 특히, 10,000개 이상의 옵션을 가진 대규모 후보 풀을 가진 작업에 대한 실험이 포함되었습니다.

- **Performance Highlights**: 실험 결과, decoding-free candidate selection 방법이 기존의 token-level decoding 방식에 비해 더 나은 성능을 보여주었다고 보고합니다. 이러한 방법들은 gradients를 효율적으로 활용할 수 있게 하며, 시간 소모를 줄이는 데 기여합니다. 논문은 이러한 결과가 향후 모델 설계에 중요한 통찰력을 제공함을 강조합니다.



### Memorize and Rank: Elevating Large Language Models for Clinical Diagnosis Prediction (https://arxiv.org/abs/2501.17326)
Comments:
          To appear at AAAI 2025

- **What's New**: 이번 연구는 MERA라는 임상 진단 예측 모델을 도입하여 자연어 처리(NLP) 지식을 의료 실무에 연결하는 방안을 제시합니다. MERA는 질병 후보 순위 리스트에서 위계적 대조 학습(hierarchical contrastive learning)을 적용하여 방대한 결정 공간 문제를 해결하며, 모델의 내재된 자연어 임상 지식을 의학 코드(medical codes)와 연계시킵니다.

- **Technical Details**: MERA는 환자의 이전 진단 결과를 기반으로 선형 시퀀스로 모델링하여 다음 방문에서의 진단 결과에 대한 확률 분포를 생성하도록 설계되었습니다. 우리는 의료 코드 간의 관계를 활용하여 임상 지식을 통합하고, 대조 학습을 통해 모델이 진짜 진단을 잘 구별하도록 훈련시킵니다. 이는 ICD 코드의 위계 구조 내에서 여러 수준의 대조 학습을 통해 수행됩니다.

- **Performance Highlights**: MERA는 MIMIC-III와 MIMIC-IV 데이터셋에서 일반 진단 및 심부전 예측 임무에서 기존의 최첨단 모델들과 비교해 существенно 개선된 성과를 보여주었습니다. MERA는 의료 코드 및 정의 간의 양방향 매핑을 거의 완벽하게 암기할 수 있으며, GPT-4보다도 진단 예측 역량이 뛰어난 것으로 나타났습니다.



### Mitigating Hallucinated Translations in Large Language Models with Hallucination-focused Preference Optimization (https://arxiv.org/abs/2501.17295)
Comments:
          NAACL 2025 Main Conference Long paper (9 pages)

- **What's New**: 이번 연구에서는 기계 번역(Machine Translation, MT)의 경향 변화에 대해 논의하고 있습니다. 전통적인 인코더-디코더 모델에 비해 세밀하게 조정된 대형 언어 모델(Large Language Models, LLM) 기반 시스템이 경쟁력을 가지게 되었다는 점이 주목할 만합니다. 그러나 LLM 시스템의 주요 문제는 생성되는 허위 번역(hallucinations)의 위험이 증가한다는 것입니다.

- **Technical Details**: 허위 번역의 문제를 해결하기 위해, 기존 연구는 주로 전통적인 MT 모델에 초점을 맞추고, 후속 수정(post-hoc mitigation) 방식을 사용했습니다. 그러나 본 연구에서는 모델 훈련 단계에서 허위 번역을 내재적으로 완화할 수 있는 방법을 제안합니다. 이 과정에서 허위 번역에 대한 선호 데이터셋(preference datasets)을 생성하는 데이터 생성 프레임워크를 도입했습니다.

- **Performance Highlights**: 이 방법으로 LLM을 미세 조정하게 되면, 다섯 가지 언어 쌍에서 평균 96%의 허위 번역 비율 감소를 보이는 동시에 전체 번역 품질을 유지할 수 있었습니다. 제로샷(zero-shot) 환경에서 본 접근법은 세 가지 미지의 목표 언어(target languages)에서 평균 89%의 허위 번역 감소 효과를 나타냈습니다.



### Tailored Truths: Optimizing LLM Persuasion with Personalization and Fabricated Statistics (https://arxiv.org/abs/2501.17273)
- **What's New**: 이 논문은 대화형 설정에서 인간의 의견을 변화시키려는 LLM(대형 언어 모델)의 설득력을 조사합니다. 특히, 사용자 개인 정보와 조작된 통계를 포함한 맞춤형 주장을 활용하는 혼합 전략이 기존의 정적 인간 주장을 초월할 수 있음을 발견했습니다. 이러한 연구는 LLM들이 저렴하고 설득력 있는 대규모 정보 조작 캠페인을 가능하게 할 수 있는 우려되는 잠재력을 드러냅니다.

- **Technical Details**: 상세한 실험을 통해 연구진은 총 33명의 참가자를 대상으로 198번의 퍼셉션 시험을 수행하고 GPT-4o-mini가 혼합 전략을 활용할 때 정적 인간 주장을 넘어 설득력이 강하다는 것을 입증했습니다. LLM의 설득력은 사용할 수 있는 전략에 따라 다르며, 특히 맞춤형 정보의 활용이 효과적임을 보여줍니다. 연구의 결과는 LLM이 단순한 프롬프트와 스캐폴딩 접근법을 통해도 강력한 설득력을 발휘할 수 있음을 시사합니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM의 혼합 전략을 활용한 실험은 참가자들의 초기 입장을 수정할 확률이 51%로 나타나, 정적 인간 주장의 32%를 초월했습니다. 이에 따라 LLM이 제공하는 주장이나 반박은 단순히 인간이 작성한 주장보다 더 효율적일 수 있음을 설명합니다. 이러한 발견은 LLM이 대규모 정보 조작 및 설득 캠페인에서 무엇보다 중요한 역할을 할 수 있음을 강조합니다.



### Comprehensive Evaluation for a Large Scale Knowledge Graph Question Answering Servic (https://arxiv.org/abs/2501.17270)
- **What's New**: 이번 논문에서 제안된 Chronos는 KGQA(지식 그래프 질문 응답 시스템)의 산업 규모 평가를 위한 포괄적인 프레임워크를 소개합니다. Chronos는 복잡한 KGQA 시스템을 효과적으로 평가하기 위해 엔드 투 엔드 및 컴포넌트 수준의 메트릭을 집중적으로 다루고 있으며, 다양한 데이터셋에 확장 가능한 접근 방식으로 설계되었습니다. 또한, KGQA 시스템의 성능을 출시 전에 측정하는 데 유용합니다.

- **Technical Details**: Chronos는 KGQA 시스템의 성능을 다양한 요소로 나누어 평가합니다. 여기에는 엔티티 링크(Entity Linking), 관계 분류(Relation Classification), 구조적 쿼리 생성(Structured Query Generation), 사실 검색(Fact Retrieval)과 같은 여러 컴포넌트가 포함됩니다. 이러한 각 컴포넌트는 사용자 쿼리를 이해하고 이를 지식 그래프에 맞춰 변환하는 데 중요한 역할을 하며, 상호 의존성으로 인해 하나의 컴포넌트 실패가 전체 시스템에 영향을 미칠 수 있습니다.

- **Performance Highlights**: Chronos는 KGQA 시스템의 효율성과 신뢰성을 높이기 위해 각 요소의 성능을 지속적으로 추적합니다. 메트릭은 대시보드에서 모니터링되며, 이를 통해 시스템의 품질 향상 또는 저하를 평가할 수 있습니다. 논문에서 제안된 사례 연구를 통해, 이와 같은 평가 프레임워크가 실제 KGQA 시스템 개선에 대한 중요한 인사이트를 제공할 수 있음을 보여줍니다.



### Giving the Old a Fresh Spin: Quality Estimation-Assisted Constrained Decoding for Automatic Post-Editing (https://arxiv.org/abs/2501.17265)
Comments:
          Accepted to NAACL 2025 Main Conference: Short Papers

- **What's New**: 이번 논문에서는 Automatic Post-Editing (APE) 시스템의 과도한 수정(over-correction) 문제를 해결하기 위해, decoding 과정에서 단어 수준의 품질 추정(Quality Estimation, QE) 정보를 통합하는 새로운 방법을 제안합니다. 이 방법은 특정 아키텍처에 의존하지 않으며, 모든 APE 시스템에 적응할 수 있어 혁신적입니다. 실험 결과, 영어-독일어, 영어-힌디어, 영어-마라티어 언어 쌍에 대해 기존 APE 시스템 대비 TER(Translation Edit Rate) 점수에서 각각 0.65, 1.86, 1.44 포인트의 유의미한 개선을 보여줍니다.

- **Technical Details**: 이 논문에서 제안하는 접근법은 Grid Beam Search (GBS)라는 확장된 beam search 기법을 사용하여 decoding을 수행합니다. GBS는 어휘 제약(lexical constraints)을 통합하여 출력 시퀀스를 생성하는 방법으로, 각 단계에서 정확한 번역이 포함되도록 세부적으로 제어합니다. 단어 수준의 QE 시스템을 활용해 각 단어에 'OK' 또는 'BAD' 태그를 부여하여 올바른 번역을 제시하는 방식입니다.

- **Performance Highlights**: 실험의 결과, 제안된 GBS 기반의 APE decoding 기술은 기존 Baseline 1 및 Baseline 2와의 직접적인 비교에서 우수한 성능을 나타냅니다. 특히, 단어 수준의 QE 정보를 이용한 decoding 방법은 APE 시스템의 과도한 수정을 효과적으로 줄이는 데 기여함을 보여주며, 결과적으로 번역 품질을 높이는 데 유용하다는 것을 알 수 있습니다.



### NUS-Emo at SemEval-2024 Task 3: Instruction-Tuning LLM for Multimodal Emotion-Cause Analysis in Conversations (https://arxiv.org/abs/2501.17261)
Comments:
          2nd place at SemEval-2024 Task 3, Subtask 2, to appear in SemEval-2024 proceedings

- **What's New**: 이번 연구는 SemEval-2024 Task 3의 멀티모달 감정-원인 분석에서 새로운 시스템 아키텍처를 제안합니다. 특히, Emotion-Cause Pair Extraction with Emotion Category (MECPE-Cat)라는 하위 과제에 집중하며 감정을 인식하고 원인을 추출하기 위해 대형 언어 모델(LLM)을 활용합니다. 본 시스템은 감정-원인 인식 향상을 위해 감정-원인 인식 알고리즘을 추가로 구현하였습니다.

- **Technical Details**: 연구에서는 LLM의 잠재력을 활용하여 두 개의 하위 과제인 ERC(Emotion Recognition in Conversation)와 ECPE(Emotion-Cause Pair Extraction)를 해결하고자 합니다. 또한, LoRA와 같은 효율적인 파라미터 조정을 통해 LLM 성능을 최적화하고, 감정-원인 인식의 정확성을 향상시키기 위한 프롬프트 기반 학습 방법을 접목시켰습니다. 연구진은 ChatGLM 모델을 선택하여 최적화된 성능을 확보하고, 다양한 멀티모달 정보를 활용하여 감정 인식을 강화하였습니다.

- **Performance Highlights**: 이 연구의 모델은 MECPE-Cat의 공식 시험 세트에서 평균 F1 점수 34.71%를 기록하며 2위에 올랐습니다. 이러한 성과는 LLM의 강력한 감정-원인 분석 기능 덕분이며, 향후 연구를 위해 모델과 코드를 공개할 예정입니다. 연구진은 현재 모델의 한계 및 개선 방향에 대해서도 논의하였습니다.



### Improving LLM Leaderboards with Psychometrical Methodology (https://arxiv.org/abs/2501.17200)
Comments:
          53 pages, 10 figures, 6 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능을 평가하기 위한 벤치마크(benchmark)의 필요성에 대해 다루고 있습니다. 기존의 벤치마크는 인간의 테스트와 설문조사처럼 질문 세트로 구성되어 있으며, 시스템의 인지 행동에서 나타나는 특성을 측정하고자 합니다. 이 논문은 현대의 심리측정(psychometric) 방법론을 적용하여 LLM의 순위를 개선할 수 있다는 점을 강조합니다.

- **Technical Details**: 논문에서는 Hugging Face Leaderboard 데이터를 예로 들어 전통적인 단순 순위(rank) 접근법과 심리측정 정보를 바탕으로 한 순위 방식을 비교합니다. 이러한 심리측정 기술은 인간 테스트 및 설문조사를 위해 개발된 방법으로, LLM의 성능을 평가하는 데 있어 사실적이고 강력한 결과를 도출할 수 있습니다. 특히, 심리측정 기법의 적용은 기존의 평균(score) 기반 집계 방식에 비해 더 의미있는 평가를 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 심리측정 기법의 도입이 LLM 성능 평가의 신뢰성을 높이는 데 기여하며, 벤치마크에서 얻는 데이터를 활용하여 더 정교한 비교가 가능하다는 것을 보여줍니다. 이를 통해 LLM의 능력에 대한 보다 깊이 있는 분석이 가능해지며, 향후 LLM의 발전에 중요한 지침을 제공할 수 있습니다.



### Atla Selene Mini: A General Purpose Evaluation Mod (https://arxiv.org/abs/2501.17195)
- **What's New**: Atla Selene Mini는 최신의 소형 언어 모델인 'small language model-as-a-judge' (SLMJ)로, 11개의 벤치마크에서 최고의 성능을 기록하며 다양한 평가 작업에서 가장 우수한 결과를 보여줍니다. 이 모델은 고유한 데이터를 사용하여 인간 전문가 평가와의 제로샷 일치를 크게 향상시켰으며, 공공 데이터셋에 대한 합성 비판을 추가하여 훈련하였습니다. HuggingFace와 Ollama에서 모델 가중치가 공개되어, 커뮤니티의 광범위한 채택을 촉진하고자 합니다.

- **Technical Details**: Selene Mini는 8B Instruct 모델을 사용하여 훈련되었고, 16개의 공개 데이터셋을 조합하여 총 577,000개의 데이터 포인트로 구성된 데이터 믹스를 활용하였습니다. 이 모델은 'direct preference optimization' (DPO)와 'supervised fine-tuning' (SFT) 손실을 결합하여 훈련되는 등 최적화된 구조를 가지고 있습니다. 데이터 품질을 확보하기 위한 정제 프로세스를 개발하여 합성 생성 및 필터링 방식을 도입하였으며, 이는 모델의 현실적 평가 능력을 크게 향상시킵니다.

- **Performance Highlights**: Selene Mini는 현실 세계의 금융 및 의료 데이터셋에서 인간 전문가 평가와의 제로샷 일치에서 현저한 개선을 보여주고, 다양한 프롬프트 형식에 대해서도 강력한 성능을 발휘합니다. 커뮤니티 주도의 Judge Arena에서 최고의 평가자로 자리매김하였으며, 이는 사용자들 사이에서 긍정적인 초기 결과를 나타냅니다. 모든 평가 작업에서의 고도화된 성능 덕분에 Selene Mini는 가용성 및 사용 편의성 측면에서 큰 주목을 받고 있습니다.



### AI-assisted German Employment Contract Review: A Benchmark Datas (https://arxiv.org/abs/2501.17194)
Comments:
          Dataset available on GitHub

- **What's New**: 이 논문에서는 고용 계약서(Employment Contracts)의 공정성과 합법성 검토를 위한 새로운 주석 데이터셋을 발표합니다. 저자들은 법률 텍스트에 적용할 수 있는 NLP 기술의 발전을 활용하여 변호사들이 계약을 검토하는 데 도움을 주기를 원합니다. 이러한 시도가 법률 분야의 데이터 부족 문제를 해결하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 연구팀은 독일의 고용 계약 조항을 대상으로 한 새로운 기준 데이터셋을 생성했습니다. 이 데이터셋은 공정성(Fairness) 및 합법성(Legality) 검사로 주석이 달려 있으며, 이를 통해 최신 NLP 기술을 적절히 적용할 수 있는 기틀을 마련했습니다. 또한 각 조항에 대한 기본 모델 평가(Baseline Model Evaluations)를 포함하여 연구의 기초를 다졌습니다.

- **Performance Highlights**: 이 연구는 고용 계약 텍스트의 검토에서 NLP의 가능성을 보여줍니다. 특히, 발표된 데이터셋은 법률 전문 데이터 부족 문제를 해결하기 위한 중요한 첫걸음입니다. 향후 연구에서는 이 데이터셋을 활용하여 공정성과 합법성에 대한 깊이 있는 분석을 통해 더 나은 결과를 도출할 수 있습니다.



### Aspect-Aware Decomposition for Opinion Summarization (https://arxiv.org/abs/2501.17191)
Comments:
          35 pages

- **What's New**: 이번 연구에서는 대규모 온라인 리뷰에서 의미 있는 인사이트를 도출하기 위한 의견 요약(opinion summarization)의 모듈형 접근법을 제안합니다. 이 접근법은 리뷰의 다양한 측면(aspect)에 의해 안내되어, 측면 식별(aspect identification), 의견 통합(opinion consolidation), 메타 리뷰 합성(meta-review synthesis)의 작업을 분리하여 투명성과 점검 용이성을 증가시킵니다.

- **Technical Details**: 연구는 과학 연구, 비즈니스 및 제품 영역을 대표하는 데이터셋에서 광범위한 실험을 수행하였습니다. 제안된 방법은 자동화 및 인간 평가를 통해 강력한 기준 모델(baseline models)보다 더 근거 있는(grounded) 요약을 생성하는 것으로 확인되었습니다. 또한, 리뷰 측면에 기반한 추론(reasoning)을 포함한 모듈형 접근법은 지식 비귀속(decomposed prompting)에 비해 더 유익한 중간 출력(intermediate outputs)을 생산합니다.

- **Performance Highlights**: 중간 출력은 대량의 리뷰에서 의견을 요약하는 데 있어 인간에게 효과적으로 지원할 수 있습니다. 연구 결과, 이러한 모듈형 접근법이 더 많은 정보를 제공하며, 종합적인 의견 요약 방식에서 더욱 개선된 성과를 낸다는 점이 강조됩니다.



### A Comprehensive Study on Fine-Tuning Large Language Models for Medical Question Answering Using Classification Models and Comparative Analysis (https://arxiv.org/abs/2501.17190)
Comments:
          18 pages, 5 figures,3 tables

- **What's New**: 이번 논문은 의료 질문에 대한 응답을 위해 설계된 대형 언어 모델(LLMs)의 개발 및 미세 조정에 대한 개요를 제시합니다. 모델의 정확성과 효율성을 개선하여 신뢰할 수 있는 의료 질의 응답을 제공하는 데 중점을 두고 있습니다. 이 접근법에서는 특정 의료 질문에 대한 레이블을 예측하고, 이후 해당 레이블에 대해 미리 정의된 응답을 제공하는 두 가지 단계를 구현했습니다.

- **Technical Details**: 의료 질의 응답의 정확성을 높이기 위해 RoBERTa와 BERT와 같은 다양한 모델을 평가했습니다. 모델은 Healthline.com에서 스크래핑된 6,800개의 샘플 데이터셋과 추가적인 합성 데이터로 훈련되었습니다. 성능 평가는 5-fold 교차 검증을 사용하여 수행되며, 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 score) 및 훈련 시간(training time)이 기록됩니다.

- **Performance Highlights**: 로라(LoRA) Roberta-large 모델은 78.47%의 정확도와 73.56%의 F1 점수를 기록하며 성능을 평가했습니다. Roberta-base 모델은 99.87%의 정확도를 보여주며, Bert Uncased 모델은 95.85%의 정확도로 강한 결과를 보였습니다. 마지막으로 Bert Large Uncased 모델은 완벽한 100%의 정확도를 달성하며 최고의 성능을 기록했습니다. 이러한 결과는 의료 질문 분류 및 정확한 응답 생성 능력을 나타내며, 향상된 건강 관련 AI 솔루션의 가능성을 제시합니다.



### Visualizing Uncertainty in Translation Tasks: An Evaluation of LLM Performance and Confidence Metrics (https://arxiv.org/abs/2501.17187)
- **What's New**: 이 논문은 머신 번역(Machine Translation)에서 대형 언어 모델(LLM)의 불확실성을 효과적으로 시각화하는 방법을 제시합니다. 연구의 주요 두 가지 목표는 사용자에게 모델의 신뢰도를 토큰 수준에서 제공하는 것과 번역 불확실성을 정량화하고 표현하는 웹 기반 시각화 도구를 개발하는 것입니다. 또한, T5 모델을 활용하여 WMT19 데이터셋으로 번역 품질을 평가하고 새로운 불확실성 정량화(UQ) 메트릭을 소개합니다.

- **Technical Details**: 연구에서는 세 가지 새로운 불확실성 정량화 메트릭을 도입했습니다: (1) 토큰 확률의 기하 평균, (2) 토큰 확률의 산술 평균, (3) 토큰 분포의 커토시스의 산술 평균. 이러한 메트릭은 번역 성능을 평가하기 위한 간단하면서도 효과적인 프레임워크를 제공합니다. 웹 기반 시각화 도구는 색상 그래디언트를 사용하여 각 토큰의 신뢰도를 표기하여 사용자가 번역의 품질을 직관적으로 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 결과 분석에서 전통적인 평가 메트릭과 새로운 UQ 메트릭 간의 선형 관계가 나타났으며, 연구 방법의 유효성을 입증했습니다. 또한, 인터랙티브 웹 기반 시각화 도구를 사용하여 사용자에게 번역 모델의 성능에 대한 귀중한 통찰력을 제공하며 이러한 불확실성 정량화 메트릭과 시각화 방법은 머신 번역 시스템을 평가하고 접근하는 데 실질적인 도구가 됨을 보여줍니다.



### LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering (https://arxiv.org/abs/2501.17183)
Comments:
          conference paper

- **What's New**: 이 논문은 항공 우주 제조 분야에 적합한 대규모 언어 모델(LLMs), 즉 GPT-4와 QWen의 적용을 위한 새로운 평가 메트릭스를 제안합니다. 이 메트릭스는 전문 지식을 바탕으로 이루어진 질문에 대한 모델의 정확성을 평가하며, 기존 LLM의 '환각(hallucination)' 현상으로 인한 오류를 방지하기 위한 기초를 제공합니다. 이 연구는 데이터의 신뢰성을 확보하고, 안전 기준에 맞춘 평가 기준의 부재 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 항공 우주 제조는 기술적 정확성 요구 사항이 매우 엄격하며, LLM은 프로세스 설계, 재료 선택 및 도구 정보 검색과 같은 작업에 활용될 수 있습니다. 연구진은 항공 우주 분야의 전통적인 문헌을 기반으로 다수의 선택형 질문을 생성하고, 이 질문들을 통해 다양한 LLM 모델의 성능을 평가하였습니다. 결과적으로, 현재 LLM의 전문 지식 수준이 미흡하다는 것을 발견했으며, 이에 대한 개선이 절실함을 강조합니다.

- **Performance Highlights**: 본 연구의 실험 결과에 따르면, 항공 우주 관련 전문 지식에서 LLM의 정확도는 FAA의 허용된 위험 임계치를 초과하는 징후가 발견되었습니다. 특정 기술 매개변수에 대한 오류는 기계적 구조의 안전성을 저해하여 비극적인 결과를 초래할 수 있으며, 이는 데이터 품질 관리를 위한 보다 철저한 검증 프로세스의 필요성을 명확히 합니다. 이 작업은 항공 우주 제조 분야에 LLM을 안전하게 도입하기 위한 기초적인 이해와 실질적인 통찰력을 제공하는 것을 목표로 합니다.



### Dialogue Systems for Emotional Support via Value Reinforcemen (https://arxiv.org/abs/2501.17182)
Comments:
          30 pages, 3 figures

- **What's New**: 본 연구는 감정 지원 시스템에 대한 가치 강화(value reinforcement)를 명시적으로 통합한 첫 번째 사례로, 감정 지원 대화 시스템이 도움을 요청하는 사람의 긍정적인 가치를 강화하도록 설계된 방법론을 제안합니다. 이 모델은 Reddit에서 온라인 지원 대화를 활용하여 각 대화 턴에서 어떤 가치를 강화할지를 학습합니다. 가치 강화가 감정 지원의 효과성을 높이는 잠재력이 있음을 검증하고, 미래 연구를 위한 기초를 마련했습니다.

- **Technical Details**: 연구에서는 목표 가치 감지기(target value detector)와 참조 생성기(reference generator)라는 두 가지 컴포넌트를 도입하여 지지자의 가치 강화 능력을 향상시킵니다. 시스템은 GPT-4o-mini를 기반으로 한 시뮬레이션 훈련과 직접 정책 최적화(direct policy optimization)를 통해 훈련됩니다. 이러한 구조는 보조자의 발언에 반영된 가치 증진의 보상을 극대화하는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 감정 지원 능력 및 가치 강화 면에서 다양한 기준선 모델을 능가하며, 특히 가치 강화 부분에서 전문가 치료사들의 평가에서 두드러진 성과를 보였습니다. 모델은 도움을 요청하는 사람의 도전 과제를 효과적으로 확인하고 긍정적인 측면을 강조하는 능력이 뛰어난 것으로 평가되었습니다. 이는 연구의 주요 기여로, 감정 지원 시스템의 성과 향상에 대한 새로운 방향을 제시합니다.



### Tuning LLM Judges Hyperparameters (https://arxiv.org/abs/2501.17178)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 평가에 필요한 비싼 인간 주석 비용을 줄이기 위해 LLM 기반의 판별자를 제안합니다. 이러한 판별자는 두 개의 LLM 출력을 비교하여 모델을 평가할 수 있으며, 동시에 여러 하이퍼파라미터를 조정하여 성능을 최적화하려고 합니다. 이 방법은 기존의 평가 방법들과 비교할 때 정확성과 비용 효율성 모두에서 뛰어난 성능을 보여주고, 오픈-웨이트 모델을 사용하여 접근성과 재현성을 높입니다.

- **Technical Details**: 저자들은 LLM 판별자의 하이퍼파라미터를 체계적으로 분석하고 조정하는 방법을 제안합니다. 주요 하이퍼파라미터로는 LLM 모델, 프롬프트, 추론 파라미터(예: temperature)가 포함되며, 다목적(multi-objective) 및 다신뢰도(multi-fidelity) 접근 방식을 통해 조정 비용을 절감합니다. 이는 판별자의 성능 개선에 기여하고, 더 나은 LLM 모델 평가를 위한 최적의 구성 요소를 식별할 수 있게 해줍니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 기존의 판별자들과 비교하여 더욱 높은 정확성과 비용 효율성을 나타내며, 여러 실제 테스트 데이터셋에서도 우수한 성과를 거두었습니다. 저자들은 80가지의 프롬프트 전략과 4480가지 판별자 구성을 포함한 검색 공간을 다루며, 최적의 프롬프트 파라미터화 및 기타 하이퍼파라미터 조정 방법론을 제시합니다. 이로 인해 커뮤니티는 더 나은 판별자 구축을 위한 중요한 패턴을 확인할 수 있습니다.



### Document-Level Sentiment Analysis of Urdu Text Using Deep Learning Techniques (https://arxiv.org/abs/2501.17175)
- **What's New**: 이 연구에서는 문서 수준에서 우르두어 감정 분석(Sentiment Analysis, SA)을 위한 새로운 하이브리드 딥러닝 모델인 BiLSTM-SLMFCNN을 제안하고 있습니다. 처음으로 우르두어 데이터를 위한 딥러닝 아키텍처들을 적용하여 성과를 입증하는 데 초점을 맞추었습니다. 특히, 기존의 전통적인 기계 학습 모델보다 딥러닝 모델이 더욱 효과적이며, 문서 크기에 따른 성능 변화를 함께 분석하였습니다.

- **Technical Details**: 제안된 BiLSTM-SLMFCNN 모델은 Bidirectional Long Short Term Memory(BiLSTM)와 Single Layer Multi Filter Convolutional Neural Network(SLMFCNN)를 결합하여 우르두 문서의 감정을 분류하는 능력을 극대화합니다. 이 모델은 BiLSTM을 통해 단어의 맥락적 의미를 이해하고 SLMFCNN을 통해 지역적 특징을 추출합니다. 또한, 여러 개의 필터를 사용하여 다양한 길이의 n-그램 특징을 추출할 수 있는 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 IMDB 우르두 영화 리뷰 데이터 세트와 우르두 고객 지원 데이터 세트에서 성능을 평가했으며, 각각 83%, 79%, 83% 및 94%의 정확도를 기록하여 기존의 딥러닝 모델을 능가했습니다. 특히, BiLSTM-SLMFCNN은 문서 수준 감정 분석에서 기존의 기계 학습 모델보다 우수한 결과를 보여주었습니다.



### Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling (https://arxiv.org/abs/2501.17811)
Comments:
          Research paper. arXiv admin note: text overlap with arXiv:2410.13848

- **What's New**: 이번 연구에서는 Janus의 발전된 버전인 Janus-Pro를 소개합니다. Janus-Pro는 최적화된 훈련 전략, 확장된 훈련 데이터, 더 큰 모델 크기를 포함하여 세 가지 주요 개선 사항을 통합하였습니다. 이러한 개선을 통해 Janus-Pro는 멀티모달 이해 및 텍스트-이미지 지시 수행 능력에서 상당한 진전을 이루었으며, 텍스트-이미지 생성의 안정성 또한 향상시켰습니다.

- **Technical Details**: Janus-Pro의 아키텍처는 Janus와 동일한 원칙을 따르며 멀티모달 이해와 생성 간의 비주얼 인코딩을 분리하는 것을 핵심 디자인 원칙으로 삼고 있습니다. 멀티모달 이해를 위해 SigLIP 인코더를 사용하여 고차원 시맨틱 특징을 추출하고, 이를 LLM의 입력 공간으로 매핑합니다. 그에 반해 비주얼 생성 작업에는 VQ 토크나이저를 활용하여 이미지를 분산된 ID로 변환합니다.

- **Performance Highlights**: Janus-Pro는 다양한 벤치마크에서 평가한 결과, 멀티모달 이해 능력이 우수하고 텍스트-이미지 지시 수행 성능이 크게 향상된 것으로 나타났습니다. 특히, Janus-Pro-7B 모델은 MMBench에서 79.2점을 기록하며, 기존의 최첨단 멀티모달 모델들을 초월했습니다. 또한 GenEval 리더보드에서도 0.80점을 기록하여 Janus와 DALL-E 3 등 다른 모델들을 의가 발휘했습니다.



### Improving Privacy Benefits of Redaction (https://arxiv.org/abs/2501.17762)
- **What's New**: 이 논문에서는 자연어 텍스트 데이터의 비공식화(sanitization)를 위한 새로운 redaction methodology를 제안합니다. 제안된 기법은 기존의 최첨단 기술들보다 더 나은 프라이버시(privacy) 혜택을 제공하며, 낮은 redaction 수준을 유지합니다. 이는 개인 정보 보호가 더욱 강화된 방식으로 데이터 처리에 기여할 수 있습니다.

- **Technical Details**: 프라이버시를 정의하는 주요 개념으로 (ϵ,δ)-Differential Privacy가 활용되었습니다. 두 개의 데이터셋 
\mathcal{D}_{0}와 \mathcal{D}_{1} 중 각각의 요소 x는 확률 분포 P(x)에서 랜덤하게 선택됩니다. redaction 후, 각 요소는 새로운 시퀀스 r𝑒𝑑𝑎𝑐𝑡(x)로 변환되며, 이 과정에서 데이터셋 간의 거리를 ϵ으로 측정합니다.

- **Performance Highlights**: 이 새로운 방법은 redacted dataset 간의 indistinguishability를 통해 개인 정보를 보다 안전하게 보호할 수 있습니다. 작은 ϵ와 δ 값을 사용하면, 비공식화된 데이터셋의 공개가 공격자에게 기존 공개 데이터셋으로부터 이미 알고 있는 정보 외의 추가적인 제한된 정보만 제공하게 됩니다. 이는 개인 정보 보호의 중요한 이점을 제공합니다.



### VICCA: Visual Interpretation and Comprehension of Chest X-ray Anomalies in Generated Report Without Human Feedback (https://arxiv.org/abs/2501.17726)
- **What's New**: 이 연구에서는 의료 리포트 생성의 신뢰성과 해석 가능성을 향상시키기 위한 새로운 다중 모달 프레임워크를 제안합니다. 현재의 흉부 X-ray(CXR) 보고 시스템의 문제점을 해결하기 위해, 본 프레임워크는 패러다임(phrase grounding) 모델과 텍스트-이미지 확산 모듈(text-to-image diffusion module)을 결합하여 AI가 생성한 의료 리포트를 검증할 수 있는 메커니즘을 제공합니다. 이를 통해 병리학적 구조 접근성과 의미적 일치를 평가하는 이중 평가 시스템을 도입했습니다.

- **Technical Details**: 본 연구의 다중 모달 틀은 패스톨로지(patology)를 식별하고 CXR 이미지에서 지역을 국소화하는 Phrase Grounding Model과 텍스트 프롬프트에 따라 합성 CXR 이미지를 생성하는 Text-to-Image Diffusion Module로 구성됩니다. 이 두 구성 요소를 통해 생성된 CXR 이미지의 원본과 합성 이미지의 특징을 비교하고, 지역 정확성(localization accuracy)과 의미적 일관성(semantic consistency)을 각각 평가하는 두 가지 점수를 도출합니다.

- **Performance Highlights**: 제안된 모델은 기존 방법들보다 뛰어난 성능을 보이며, 병리학적 국소화(pathology localization)와 텍스트-이미지 정렬(text-to-image alignment)에서 최첨단 결과를 달성했습니다. 새로운 검증 시스템은 전문가의 피드백 없이도 객관적이고 재현 가능한 검증 과정을 보장하며, CXR 분석 분야에서 다중 모달 AI의 필요성을 강조하고 있습니다. 이 접근법은 의료 데이터의 해석 가능성과 신뢰성을 제고하는 데 중요한 기여를 하고 있습니다.



### Using Code Generation to Solve Open Instances of Combinatorial Design Problems (https://arxiv.org/abs/2501.17725)
- **What's New**: 이 논문에서는 조합 설계(combinatorial designs) 문제의 기존 해결 방법을 자동화하는 CPro1 프로토콜을 소개합니다. 이 프로토콜은 Large Language Models (LLMs)를 활용하여 다양한 후보 방법으로 코드 생성을 가능하게 하며, 특히 Packing Array의 경우 N=21로 구성이 가능하다는 기존의 미해결 질문을 해결합니다. 또한, 다양한 조합 설계의 존재 문제를 해결하기 위한 기초 작업을 진행합니다.

- **Technical Details**: CPro1 프로토콜은 LLM을 사용하여 조합 관련 설계를 정의하고 이를 검증하는 기능을 포함합니다. 각 조합 설계 유형에 대해 전체 인스턴스 목록을 검토하고, 특정 설계가 유효한지 여부를 확인하는 방법도 구축합니다. LLM은 코드 생성에 있어 다양한 탐색 알고리즘(예: simulated annealing, genetic algorithms)을 실험하고, 자동화된 하이퍼파라미터 조정(hyperparameter tuning) 과정을 통해 성능 최적화도 수행합니다.

- **Performance Highlights**: 16가지 조합 설계 유형에 대한 테스트 결과, CPro1은 6개의 미해결 문제를 성공적으로 해결하였습니다. 이 결과로, 대칭 및 스큐 가중 행렬(Symmetric and Skew Weighing Matrices)과 같은 복잡한 조합 설계를 성공적으로 생성하는 성과를 거두었습니다. 또한, 논문에서는 이 프로토콜이 효율적으로 조합 설계 문제를 해결할 수 있음을 입증하고 있습니다.



### Uncertainty Quantification and Decomposition for LLM-based Recommendation (https://arxiv.org/abs/2501.17630)
Comments:
          WWW 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 추천 생성 과정에서 불확실성을 자주 보이는 점을 지적하고 있습니다. LLMs의 신뢰성을 높이기 위해 추천의 신뢰성을 평가하는 것이 중요하다는 점을 강조합니다. 새로운 프레임워크를 소개하여 예측 불확실성을 정량적으로 측정할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 예측 불확실성을 추천 불확실성(recommendation uncertainty)과 프롬프트 불확실성(prompt uncertainty)으로 분해하여 분석할 수 있는 새로운 방식을 소개합니다. 이를 통해 LLM 기반 추천에서 불확실성의 주 원인을 심층적으로 분석할 수 있게 됩니다.

- **Performance Highlights**: 실험을 통해 예측 불확실성이 LLM 기반 추천의 신뢰성을 효과적으로 나타낸다는 것을 입증합니다. 또한, 분해된 불확실성 측정을 통해 불확실성의 근원을 조사하고, 예측 불확실성을 줄이고 추천을 향상시키기 위한 불확실성 인식(prompting) 방안을 제시합니다.



### GLLM: Self-Corrective G-Code Generation using Large Language Models with User Feedback (https://arxiv.org/abs/2501.17584)
- **What's New**: 이 논문은 GLLM이라는 혁신적인 도구를 소개합니다. 이는 Large Language Models (LLMs)을 활용하여 자연어 지침에서 자동으로 G-code를 생성합니다. GLLM은 수동 G-code 작성의 문제를 해결하고, 인간이 읽을 수 있는 작업 설명과 기계가 실행할 수 있는 코드 간의 간격을 연결합니다.

- **Technical Details**: GLLM은 StarCoder-3B 모델을 세밀하게 조정한 후 도메인 특화 훈련 데이터와 Retrieval-Augmented Generation (RAG) 메커니즘을 추가로 통합했습니다. 이 시스템은 고급 프롬프트 전략 및 새로운 자기 수정 코드 생성 접근 방식을 활용하여 생성된 G-code의 구문적 및 의미적 정확성을 보장합니다.

- **Performance Highlights**: GLLM의 아키텍처는 구문 검사, G-code 특정 검증 및 Hausdorff 거리 기반 기능적 정확성 평가와 같은 강력한 검증 메커니즘을 포함하고 있습니다. 이러한 기술을 결합하여 GLLM은 CNC 프로그래밍을 민주화하고, 광범위한 프로그래밍 경험이 없는 사용자에게도 접근 가능하게 하며 G-code 생성의 정확성과 신뢰성을 유지하는 것을 목표로 합니다.



### LLM Assistance for Pediatric Depression (https://arxiv.org/abs/2501.17510)
- **What's New**: 이 연구는 소아 전자 건강 기록(EHR)에서 우울증 증상을 추출하기 위해 최신 대형 언어 모델(LLMs)을 이용한 가능성을 평가합니다. 전통적인 스크리닝 방법의 한계를 극복하고, 데이터를 효율적으로 활용하기 위해 제로 샷 분석(zero-shot analysis)을 적용합니다. 연구 결과, LLM이 단어 일치(word match)보다 60% 더 높은 효율성을 보였으며, Flan 모델이 정밀도에서 우수한 성능을 나타냈습니다.

- **Technical Details**: 이 연구는 2009년부터 2022년까지의 약 130만 명의 소아 환자가 포함된 크리닉 노트를 분석하였습니다. PHQ-9 점수를 찾기 위해 ‘PHQ-9 Total Score:’ 패턴을 검색했으며, 전체 노트의 2%가 해당 점수를 포함했습니다. LLMs를 활용하여 클리닉 노트에서 관련 텍스트를 추출하고, 이를 통해 우울증 증상을 인식하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 모델의 성능 비교에서 Flan은 평균 F1 점수 0.65, 정밀도 0.78을 기록하며, 특히 드물게 나타나는 증상을 잘 추출했습니다. Llama 3는 가장 높은 재현율(0.90)을 보였지만 증상을 과도하게 일반화하는 경향이 있었습니다. 이번 연구는 LLM 기반 증상 주석이 머신러닝 알고리즘에서 우울증 사례를 구별하는 데 중요하다는 점을 강조하며, 0.78의 높은 정밀도로 성능 향상을 보여주었습니다.



### DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performanc (https://arxiv.org/abs/2501.17479)
- **What's New**: 본 논문에서는 여러 LLM의 상호 보완적인 강점을 활용하여 성능을 향상시키는 새로운 앙상블 기법인 Diverse Fingerprint Ensemble (DFPE)를 제안합니다. 이 방법은 피어리와 같은 응답 패턴 기반 모델 군집화, 주제별로 하위 성능 모델 필터링, 최종 모델 가중치 조정을 포함합니다. 이를 통해 LLM의 견고성과 일반화 능력을 증가시키며, 다양한 언어 이해 작업에 효과적으로 대응할 수 있는 방법을 보여줍니다.

- **Technical Details**: Diverse Fingerprint Ensemble (DFPE) 방법은 모델 응답의 'fingerprint' 패턴을 군집화하여 다양한 문제 해결 전략을 유지하면서 중복성을 방지합니다. 모델은 주제 기반 검증 정확도에 따라 적절한 가중치를 부여받고, 주제별 경험치를 고려하여 조정된 가중치를 사용하여 앙상블을 구성합니다. 이 방식은 다양한 질문 유형을 처리할 수 있도록 모델의 적합성과 다양성을 동시에 강조합니다.

- **Performance Highlights**: MMLU 벤치마크 실험에서 DFPE는 최고의 단일 모델에 비해 전체 정확도를 3% 향상시켰으며, 학문별 정확도는 5% 개선되었습니다. 이러한 결과는 LLM의 선택 및 성과 기반 가중치 조정이 복잡한 다면적 언어 이해 작업에서 성능을 유의미하게 개선할 수 있음을 입증합니다. 최종적으로, DFPE는 주제에 맞춘 다각적이고 효과적인 앙상블을 통해 MMLU와 같은 복잡한 작업에서 우수한 성과를 보입니다.



### Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction (https://arxiv.org/abs/2501.17459)
Comments:
          9 pages, 7 figures

- **What's New**: 이 연구는 항공기 비행 궤적 예측 문제를 언어 모델링 문제로 재구성하여 대규모 언어 모델(LLMs)의 활용 가능성을 탐구합니다. 특히, ADS-B 항공 데이터에서 항공기의 위치 및 상태를 나타내는 특징을 추출하여 프롬프트 기반 데이터셋을 구성하고, 이 데이터셋을 사용하여 LLM을 미세 조정합니다. 이러한 접근 방식은 복잡한 시공간 패턴을 학습하여 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: 비행 궤적 예측은 다변량 시계열 문제로 간주되며, 이는 단일 단계 예측과 다단계 예측으로 나뉩니다. 이 연구에서는 단기 예측에 초점을 맞추어 과거의 상태 매개변수를 기반으로 항공기의 미래 상태를 예측합니다. LLM 기반 방법은 표준화에 대한 의존도를 줄여주는 구조화된 워크플로를 채택하여 높은 정확도의 예측을 가능하게 합니다.

- **Performance Highlights**: 포괄적인 실험 결과, LLMs가 기존의 전통적인 방법에 비해 단일 단계 및 다단계 예측 모두에서 주목할 만한 성능 향상을 보여주었습니다. 특히 LLaMA-3.1 모델이 가장 높은 전반적인 정확성을 기록했습니다. 그러나 LLM의 높은 추론 지연(latency)은 실시간 응용 분야에서 도전 과제가 되어, 추가적인 연구 필요성을 강조합니다.



### A review on the novelty measurements of academic papers (https://arxiv.org/abs/2501.17456)
- **What's New**: 본 논문은 혁신의 홍보 및 관리를 위한 혁신성 평가의 중요성을 다룹니다. 정보 기술의 발전과 오픈 데이터 운동에 힘입어 혁신성 측정에 대한 일부 진전을 이루었으며, 이는 과학 분야에서의 기여와 진행 상황을 데이터 기반으로 평가할 수 있도록 합니다. 이 논문은 과학 논문의 혁신성 측정에 대한 체계적인 분석을 제공하는 것을 목표로 합니다.

- **Technical Details**: 우리는 과학적 혁신성(scientific novelty)과 독창성(originality), 과학적 혁신(scientific innovation), 창의성(creativity), 그리고 과학적 돌파구(scientific breakthrough)와 같은 네 가지 유사한 개념 간의 차이를 비교했습니다. 이어서 과학적 혁신성의 유형을 검토하고 기존의 혁신성 측정을 데이터 유형별로 분류하여 각 유형에 대한 측정을 검토했습니다. 또한 혁신성 측정의 검증 방식과 관련된 도구 및 데이터 셋을 조사하였습니다.

- **Performance Highlights**: 마지막으로, 앞으로의 연구를 위한 여러 개방된 이슈를 제안하였습니다. 본 리뷰는 과학 논문의 혁신성 측정을 위한 체계적인 접근법을 제시하며, 이는 향후 연구자들이 과학적 연구의 방향성을 이해하고 평가하는 데 기여할 것으로 기대됩니다.



### Towards Making Flowchart Images Machine Interpretab (https://arxiv.org/abs/2501.17441)
Comments:
          Published at: ICDAR 2023, Project Page: this https URL

- **What's New**: 이 논문은 플로우차트(flowchart) 이미지를 머신이 해석할 수 있는 실행 가능한 파이썬 코드로 변환하는 새로운 접근 방식을 제안합니다. 이를 위해, 최근 자연어를 코드로 생성하는 연구에서 영감을 받아 FloCo-T5라는 트랜스포머 기반의 프레임워크를 구축하여 플로우차트 이미지를 자동으로 변환합니다. 또한, 11,884개의 플로우차트 이미지와 해당하는 파이썬 코드로 구성된 FloCo 데이터셋을 소개하여, 향후 연구에 기여할 기반을 마련하였습니다.

- **Technical Details**: FloCo-T5는 플로우차트 이미지를 시퀀스 인코딩으로 변환하고, 이를 통해 프로그래밍 언어의 구조와 의미를 이해하도록 사전 훈련(pre-training)되었습니다. 이 모델은 특정한 작업을 위한 목표로 코드 샘플을 대규모로 증강하여 학습하였으며, 최종적으로 시퀀스-투-시퀀스(sequence-to-sequence) 생성 문제로 모델을 조정(fine-tuning)하여 사용합니다. 본 논문에서 실시한 실험들은 다양한 코드 생성 메트릭에서 FloCo-T5가 기존 기준 모델들보다 우수한 성능을 보임을 증명하였습니다.

- **Performance Highlights**: 실험 결과, FloCo-T5는 BLEU, CodeBLEU, 정확도 점수에서 각각 67.4, 75.7, 20.0를 기록하며, 기존 모델들보다 뛰어난 성능을 나타냈습니다. 이는 우리가 제안한 사전 훈련 목표와 데이터 증강 전략의 효과를 분명히 입증합니다. 또한, 본 연구에서 제안한 모델은 손으로 그린 플로우차트 이미지에도 적용 가능하다는 점에서, 다양한 응용 가능성을 보이고 있습니다.



### Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation (https://arxiv.org/abs/2501.17433)
- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLMs)은 해로운 샘플에 대한 미세 조정(fine-tuning) 공격에 취약해지고, 이는 모델의 안전성 정렬(safety alignment) 능력을 손상시킬 수 있습니다. 이러한 공격을 방지하기 위해 일반적으로 가드레일(guardrail)로 해로운 샘플을 필터링하는 방법이 사용되지만, 본 논문에서는 이러한 가드레일 의존이 신뢰할 수 없음을 보여줍니다. 새로운 공격 방법인 Virus를 통해, 해로운 데이터의 미세 조정이 가능한 점을 입증하며, 이를 통해 기존의 필터링 방법을 우회하는 데 성공했습니다.

- **Technical Details**: 본 연구에서는 Virus라는 새로운 데이터 최적화(data optimization) 방법을 제안합니다. 이 방법은 두 가지 목표를 달성하기 위해 설계되었습니다. 첫 번째는 가드레일을 성공적으로 우회할 수 있도록 가드레일에 대한 공격 손실(jailbreak loss)을 최소화하는 것이고, 두 번째는 해로운 그래디언트(harmful gradient)을 유사하게 만들어서 피해 모델의 안전성 정렬이 무너지도록 하는 것입니다. 실험 결과 Virus는 가드레일을 효과적으로 우회하며, 100%의 누출(leakage) 비율에 도달할 수 있음을 보였습니다.

- **Performance Highlights**: Virus에 의해 최적화된 데이터는 피해 LLM의 안전성 정렬을 크게 저하시켜 21.8%의 유해 점수(harmful score) 증가를 가져왔습니다. 이를 통해 가드레일의 한계가 드러났으며, 본 연구는 가드레일이 해로운 미세 조정 공격을 방지하는 데 있어 믿을 수 없는 존재라는 것을 강조합니다. 연구의 마지막 메시지는, 가드레일 의존은 해로운 미세 조정 공격에 대한 효과적인 솔루션이 아니라는 점입니다.



### General Scene Adaptation for Vision-and-Language Navigation (https://arxiv.org/abs/2501.17403)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 Vision-and-Language Navigation (VLN) 작업을 개선하기 위한 새로운 접근법인 GSA-VLN을 제안합니다. 기존의 VLN 작업들이 환경 기본 경험을 고려하지 않는 것에 반해, GSA-VLN은 특정 환경에서 내비게이션 지시를 실행하며 그에 따라 지속적으로 적응할 것을 요구합니다. 이를 통해, 에이전트들이 실제 환경에 맞춰 지속적으로 개선될 수 있는 가능성을 엿볼 수 있습니다.

- **Technical Details**: 주요 이점으로는 GSA-R2R 데이터셋을 통해 기존 VLN 데이터셋에 비해 환경 및 지시문 다양성을 획기적으로 확장한 점입니다. 또한, 대규모 Vision-Language Models (VLMs)와 Large Language Models (LLMs)을 활용한 세 단계의 지시 제작 프로세스를 개발하여, 각 환경에서 600개의 지시문을 다양하게 생성합니다. 이 연구는 에이전트의 성능을 ID 및 OOD 문맥 모두에서 평가할 수 있는 환경을 제공합니다.

- **Performance Highlights**: 실험 결과, GR-DUET라는 새로운 방법은 각 환경의 전반적인 위상 그래프를 지속적으로 업데이트하여 훈련 및 평가에서의 과거 정보를 보존합니다. 이러한 접근은 기본 DUET 모델에 비해 8%의 성공률 향상을 이끌어내며, 모든 GSA-R2R 분할에서 최신 결과를 생성하는 데 성공했습니다.



### Learning Free Token Reduction for Multi-Modal LLM (https://arxiv.org/abs/2501.17391)
- **What's New**: 본 논문에서는 비디오 기반 멀티모달 언어 모델(Multimodal Large Language Models, MLLMs)의 효율성을 높이기 위해 시각적 프롬프트를 압축하는 새로운 접근 방식을 제안합니다. 기존 모델 압축 기술은 주로 아키텍처 개선이나 시각적 토큰 수 줄이기에 집중했으나, 시각 데이터의 고유한 공간 및 시간적 특성을 고려하지 않았습니다. 반면, 제안된 방법은 시각적 토큰의 시간적 및 공간적 차원 모두에서 압축을 수행하여 성능을 유지하면서도 계산 비용을 절감할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 방법은 학습이 필요 없는 플러그 앤 플레이(plug-and-play) 압축 파이프라인으로, 대부분의 MLLM 프레임워크에 통합할 수 있습니다. 우리는 시간 차원에서 인접한 토큰을 병합하고 공간 차원에서 덜 유의미한 토큰을 제거하여 시각적 표현의 중복성과 희소성을 활용합니다. 이를 통해 모델의 추론 능력을 향상시키고 실험적으로 비디오-QA(Question Answering) 작업에서의 효율성을 개선했습니다.

- **Performance Highlights**: 실험 결과, 제안된 압축 방법은 비디오-LLMs의 성능을 유지하면서도 토큰 수를 효과적으로 줄일 수 있음을 보여주었습니다. 이로써 비디오 기반 멀티모달 태스크에서 significant한 개선을 나타내며, 전체적인 추론 속도와 효율이 크게 향상되었습니다. 이는 다양한 MLLM 아키텍처와의 호환성을 보장하며, 동종의 기술적 과제 해결에 기여할 것으로 기대됩니다.



### Attribution analysis of legal language as used by LLM (https://arxiv.org/abs/2501.17330)
Comments:
          9 pages, 17 figures

- **What's New**: 이번 연구에서는 법률 관련 작업에 최적화된 세 가지 공개 LLM(대규모 언어 모델)을 구현하고, 법률 텍스트로 훈련했을 때의 분류 정확도 향상 원인과 방법을 탐구합니다. 두 개의 공개 법률 데이터세트를 활용하여 'overruling' 텍스트의 이진 분류와 'holding' 법원 결정을 식별하는 다중 선택 과제를 설정하였습니다. 실험에서는 법률 LLM과 일반 BERT 모델을 비교하며, 각 모델의 성능 변동 원인을 통합 기울기(attribution) 기법을 사용하여 분석하였습니다.

- **Technical Details**: BERT 모델은 일반적으로 영어 위키백과 코퍼스에서 훈련된 후, 법률 텍스트로 추가 학습되었습니다. 라벨링된 법률 문서가 포함된 두 개의 데이터 세트, 즉 'overruling'과 'casehold'를 사용하여 모델의 성능을 비교하였고, 각 모델의 토크나이저(tokenizer) 동작과 법률 언어 처리를 기반으로 한 차이점을 분석하였습니다. 통합 기울기 기법(integrated gradient)이 적용되어, 다양한 네트워크 구조 내에서 모델의 출력 결과를 도출하는 원인을 설명하는데 사용되었습니다.

- **Performance Highlights**: 모든 모델이 테스트 예제의 일부를 올바르게 분류할 수 있는 반면, 특정 예제는 오직 하나의 모델만이 올바르게 식별할 수 있음을 발견하였습니다. 데이터셋 텍스트에서 생성된 토큰의 빈도 분석을 통해, 법률 주제를 명확히 나타내는 토큰을 확인할 수 있었습니다. 연구 결과는 법률 LLM이 어떻게 법률 전문 용어와 특정 사례에 적응하여 다른 모델과 차별화된 성능을 나타내는지를 밝혀내는 데 초점을 맞추고 있습니다.



### "Ownership, Not Just Happy Talk": Co-Designing a Participatory Large Language Model for Journalism (https://arxiv.org/abs/2501.17299)
Comments:
          Under review for an ACM conference

- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)이 저널리즘 분야에서 어떻게 활용되고 있는지를 살펴봅니다. 특히, 저널리스트가 주도하는 LLM을 개발하고 그 과정에서의 참여 디자인에 주목합니다. LLM들이 저널리즘의 경제적 위기를 극복하는 데 기여할 수 있을지에 대한 논의가 이뤄집니다.

- **Technical Details**: 저자들은 20명의 저널리스트들과의 인터뷰를 통해 LLM이 저널리즘 환경에서 어떻게 설계되어야 하는지를 조사했습니다. 이 과정에서 구조적 힘, 시장 경쟁, 그리고 개인 저널리스트 및 뉴스룸에 미치는 영향을 분석하였습니다. 저널리스트 주도의 LLM 구조와 기능에 대해 제안하며, 참여 방법론의 중요성을 강조합니다.

- **Performance Highlights**: 저널리즘에서 LLM의 도입은 신뢰성 및 공정성 문제를 야기할 수 있습니다. 논문에서는 AI 도구의 도입이 저널리즘의 투명성과 신뢰성을 어떻게 해칠 수 있는지를 다루고, 기자들이 AI 기술에 대해 느끼는 인식의 장벽들을 지적합니다. 참여적 접근 방식이 이 문제를 해결하는 데 필수적이라 결론짓습니다.



### Fine-Tuning Open-Source Large Language Models to Improve Their Performance on Radiation Oncology Tasks: A Feasibility Study to Investigate Their Potential Clinical Applications in Radiation Oncology (https://arxiv.org/abs/2501.17286)
- **What's New**: 이번 연구에서는 방사선 종양학(radiation oncology) 분야에서 대규모 언어 모델(LLMs)의 활용 가능성을 탐구하고 있습니다. 특히, 도메인 지식(domain knowledge)으로 세밀하게 조정된 LLM들이 치료 요법 생성(treatment regimen generation), 치료 방식 선택(treatment modality selection), ICD-10 코드 예측에 효과적인 성과를 낼 수 있는지를 검토하였습니다. 이는 해당 분야의 데이터 처리에 새로운 시각을 제공하는 연구입니다.

- **Technical Details**: 이 연구는 15,724명의 환자 사례에서 데이터를 추출하였으며, 7,903개의 사례를 대상으로 진단, 치료 계획, 치료 방식, ICD-10 코드를 수집하여 전처리 및 수작업 주석 작업을 진행했습니다. 각 사례는 환자 진단 상세정보와 각 작업에 대한 답변(치료 요법, 치료 방식 또는 ICD-10 코드)을 구성하여 감독된 세밀 조정(supervised fine-tuning)을 위한 쌍으로 사용되었습니다. LLaMA2-7B 및 Mistral-7B 모델을 활용하였고, Low-Rank Approximations 방법을 사용하여 성능을 향상시켰습니다.

- **Performance Highlights**: 세밀 조정된 LLM은 모든 작업(Task)에서 원래 LLM을 능가한 것으로 나타났으며, 통계적으로 유의미한 성과(p-value <= 0.001)를 보였습니다. 방사선 종양학자에 의한 임상 평가에서는 세밀 조정된 LLM이 생성한 치료 요법의 60% 이상이 임상적으로 수용 가능하다고 평가되었습니다. 또한, 정밀도(precision), 재현율(recall), F1 스코어의 지표에서도 개선된 성과를 기록했습니다.



### From Natural Language to Extensive-Form Game Representations (https://arxiv.org/abs/2501.17282)
Comments:
          This work has been accepted as a full paper for AAMAS 2025. This is a full version of the AAMAS 2025 proceedings

- **What's New**: 이 논문에서는 자연어로 작성된 게임 설명을 게임 이론의 확장 형식으로 변환하기 위한 두 단계의 프레임워크를 제안합니다. 이 과정에서 Large Language Models (LLMs)와 인 컨텍스트 학습(in-context learning)을 활용하여 게임의 전략적 복잡성을 다룹니다. 특히, 불완전한 정보의 문제를 해결하기 위해 정보 집합과 부분 트리 구조를 식별하는 모듈을 개발하고, 이후 이를 바탕으로 완전한 확장 형식 게임 트리를 생성합니다.

- **Technical Details**: 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계는 게임의 불완전한 정보를 처리하기 위한 모듈이 포함되어 있으며, 두 번째 단계에서는 LLM과 셀프 디버깅 모듈이 결합되어 최종 EFG(확장 형식 게임)를 생성합니다. 생성된 EFG는 pygambit이라는 파이썬 API를 사용해 나타내며, 이를 통해 내쉬 균형(Nash equilibria)과 같은 계산 작업을 자동화할 수 있습니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 다양한 LLM에서 기초 모델들에 비해 상당히 향상된 성과를 보였습니다. 모든 테스트 게임을 성공적으로 해결한 최선의 모델이 있으며, 개별 모듈들도 좋은 성과에 기여했습니다. 본 프레임워크는 다양한 전략적 단계와 조건에서 게임 설명의 정확성을 100% 달성하는 등 강력한 성능을 입증했습니다.



### Audio Large Language Models Can Be Descriptive Speech Quality Evaluators (https://arxiv.org/abs/2501.17202)
Comments:
          ICLR 2025

- **What's New**:  최근 연구들은 멀티모달 (multimodal) 에이전트의 이상적인 특성으로 입력 방식의 질을 인식하는 것을 강조하고 있습니다. 하지만 리서치 결과 대부분의 오디오 대형 언어 모델 (audio LLMs)은 처리하는 음성의 품질에 대한 인식이 부족합니다. 이를 해결하기 위해, 저자들은 인간의 평가에 기반한 첫 번째 자연어 기반 음성 평가 코퍼스를 소개하고 있습니다. 이 코퍼스는 전체 평균 의견 점수 (MOS)와 함께 다양한 차원에서의 상세 분석을 제공합니다.

- **Technical Details**:  이 연구는 음성 신호의 품질 평가를 위해 자연어 처리(NLP) 시스템에 음향 정보(acoustic information)를 통합하는 방향으로 나아가고 있습니다. 연구진은 이 새로운 코퍼스를 사용하여 LLM 증류(distillation)와 오디오 LLM의 정렬 접근 방식인 ALLD를 제안하고 있습니다. ALLD는 원시 음성(raw speech)으로부터 관련 정보를 추출하고 의미 있는 응답을 생성할 수 있게 가이드합니다. 실험 결과, ALLD는 MOS 예측에서 최고 예측 모델을 초월하여 평균 제곱 오차(MSE) 0.17과 A/B 테스트 정확도 98.6%를 기록했습니다.

- **Performance Highlights**:  이 연구의 결과는 생성된 응답이 두 가지 작업에서 BLEU 점수 25.8과 30.2를 달성하여 작업 특정 모델(task-specific models)의 성능을 초과했음을 보여줍니다. 이를 통해 오디오 대형 언어 모델이 음성 신호에 대한 포괄적인 인식을 발전시키고, 실세계 청각 및 감각 지능 에이전트의 개발에 기여할 수 있는 가능성을 입증하였습니다.



### Complete Chess Games Enable LLM Become A Chess Master (https://arxiv.org/abs/2501.17186)
Comments:
          NAACL 2025

- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)을 활용하여 체스를 완벽하게 플레이할 수 있는 ChessLLM 모델을 제안합니다. LLM의 여러 응용 분야 중에서 체스와 같은 추상 게임에서의 잠재력은 충분히 탐색되지 않았습니다. ChessLLM은 게임을 텍스트 형식으로 변환하여 최적의 수를 Forsyth-Edwards Notation(FEN)으로 표현하며, 단순한 지도 학습을 통해 전문가 수준의 Elo 등급인 1788을 달성했습니다.

- **Technical Details**: ChessLLM은 20억 개 이상의 토큰으로 구성된 방대한 체스 게임 데이터셋을 구축하였으며, 이 데이터셋은 FEN-최고 수 쌍으로 구성되어 체스의 현재 상태를 명확하게 나타냅니다. 모델은 open-llama-3B를 기반으로 하며, 이전의 강화 학습 방법을 언어 모델링으로 변환하여 정책 학습에 적용했습니다. 체스 게임의 전반적인 품질을 보장하기 위해 길고 짧은 라운드 데이터를 비교한 결과, 긴 라운드 데이터가 더 뛰어난 성능을 발휘함을 입증했습니다.

- **Performance Highlights**: ChessLLM은 Stockfish와의 매칭에서 61%의 승률을 기록하며 Elo 등급 0에서 1788의 성과를 달성했습니다. 모델은 Elo 등급 1에서 56%, 2에서 30%의 승률을 보였으며, 이는 전체 게임을 플레이함을 통해 평가된 결과입니다. Stockfish 대비 매우 경쟁력 있는 성능을 보여주며, 이 모델의 평가 방법은 순수한 게임 성능을 측정하는 데 중점을 두었습니다.



### An AI-Driven Live Systematic Reviews in the Brain-Heart Interconnectome: Minimizing Research Waste and Advancing Evidence Synthesis (https://arxiv.org/abs/2501.17181)
- **What's New**: Brain-Heart Interconnectome (BHI)를 위한 AI 기반 시스템이 개발되었습니다. 이 시스템은 PICOS(인구, 개입, 비교군, 결과 및 연구 설계) 기준을 자동으로 감지하고, 의미 검색과 그래프 기반 쿼리를 통해 문헌을 효율적으로 검색하고 정리합니다. BI-LSTM 모델, 연구 설계 분류기 및 GPT-3.5 기반의 Retrieval-Augmented Generation (RAG) 방법이 도입되어 연구 자원의 낭비를 줄이고 실시간 업데이트를 제공합니다.

- **Technical Details**: 이 시스템은 복잡한 BHI 데이터를 관리하기 위해 Neo4j를 활용하여 관련된 개체들 간의 관계를 그래프 형태로 표현합니다. 또한, pgVector를 이용해 문서 임베딩을 저장하고, LangChain을 통해 사용자 입력에 맞춘 쿼리 처리가 이루어집니다. Bi-LSTM 모델은 연구의 PICOS 준수 여부를 판단하고, BERTopic은 주제를 클러스터링하여 연구의 진화를 추적합니다.

- **Performance Highlights**: 시스템은 BHI 분야에서 신뢰할 수 있는 증거 합성을 지원하며, 연구자들에게 실시간으로 새로운 자료를 통합할 수 있는 기능을 제공합니다. Power BI 대시보드를 통해 사용자들은 출판 패턴 및 연구 동향을 시각적으로 확인할 수 있으며, 이는 효율적인 자원 배분 및 임상 의사결정 지원에 기여합니다. 이러한 기술들은 다른 생물 의학 분야에도 적용 가능하다는 장점을 지니고 있습니다.



### Prompt-Based Cost-Effective Evaluation and Operation of ChatGPT as a Computer Programming Teaching Assistan (https://arxiv.org/abs/2501.17176)
- **What's New**: 최근 등장한 대규모 언어 모델(LLMs)의 발전으로 1:1 학생-교사 비율 달성의 가능성이 커지고 있습니다. 이 모델들은 특히 대학 프로그래밍 과정에서 학생들에게 실시간 피드백을 제공하는 데 사용될 수 있습니다. 연구에서는 GPT-3.5T와 GPT-4T 모델의 성능을 비교하였고, GPT-4T가 월등한 성과를 보였지만 실제 사용에는 여전히 문제가 있는 것으로 나타났습니다.

- **Technical Details**: 논문은 GPT 모델을 활용한 교육 도구의 구현 방안에 대해 세 가지 주요 측면을 다룹니다. 첫째, 다양한 피드백을 제공할 수 있는 잘 설계된 프롬프트(prompts)를 제안하여 LLM의 성능을 평가하는 방법론을 수립했습니다. 둘째, 피드백의 정확성을 평가하기 위한 자동화된 평가 메트릭스를 개발하였고, 이를 통해 수동 평가보다 시간과 비용을 절약할 수 있음을 강조하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM이 제공하는 피드백은 프로그램의 구조적 분석이 가능하여, AI가 학습 과정에서 프로그래밍 문제를 해결하는 데 필요한 진단 정보를 포함할 수 있음을 보여줍니다. 이러한 자동화된 접근법은 특히 대규모 클래스에서 학생들에게 유용할 수 있으며, 이는 향후 교육 환경의 혁신적인 변화를 이끌 수 있는 잠재력을 가지고 있습니다.



### Extractive Schema Linking for Text-to-SQL (https://arxiv.org/abs/2501.17174)
- **What's New**: 이번 연구에서는 Text-to-SQL 시스템을 위한 새로운 스키마 링크 모델을 제안합니다. 이 모델은 decoder-only LLM에서 hidden 상태에 대한 확률 예측을 생성하며, SQL 쿼리의 각 열의 역할에 대한 세밀한 예측이 가능합니다. 연구 결과는 기존의 모델들에 비해 더 높은 정확도를 가지고 있으며, precision과 recall 간의 균형을 조정할 수 있는 기능도 포함되어 있습니다.

- **Technical Details**: 제안된 접근법은 generative LLM과 cross-encoder 접근 방식을 결합하여 각 스키마 항목에 대한 확률 예측을 통해 recall 지향의 예측을 가능하게 합니다. 초기 단계에서 ground truth를 수집하고, SQL 쿼리를 분석하여 사용된 테이블과 열을 식별하여 고유의 구성을 유지합니다. 이 과정에서는 mo-sql-parsing을 사용하여 SQL 문을 정적 분석합니다.

- **Performance Highlights**: 제안된 스키마 링크 모델은 관련 데이터베이스 컬럼에 대한 완벽하고 세밀한 예측을 통해 SQL 쿼리 생성을 지원합니다. 연구 결과, 이전 방법들과 비교하여 더욱 개선된 성능을 보여주며, SQL 생성 과정에서의 오류를 줄이고 정확성을 높이는 데 기여합니다. 이를 통해 다양한 비즈니스 환경에서의 실시간 데이터 기반의 질의 생성이 더욱 원활해질 것으로 기대됩니다.



### Benchmarking Randomized Optimization Algorithms on Binary, Permutation, and Combinatorial Problem Landscapes (https://arxiv.org/abs/2501.17170)
- **What's New**: 이번 연구에서는 Randomized Optimization 알고리즘인 Randomized Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithms (GA), MIMIC (Mutual Information Maximizing Input Clustering)을 평가했습니다. 세 가지 문제 유형인 binary, permutation, combinatorial에서 각 알고리즘의 성능을 비교하였습니다. 특히, MIMIC과 GA는 binary 및 combinatorial 문제에서 우수한 해법을 제공하는 반면, RHC와 SA는 복잡한 문제에서 한계가 있음을 보여줍니다.

- **Technical Details**: 최적화 문제는 다양한 분야에서 의사결정 과정의 핵심으로 작용하며, 각 알고리즘의 효과는 문제의 성격에 따라 다릅니다. 논문에서는 이론적인 배경으로 각 알고리즘의 원리와 특징을 서술하며, random optimization 기술이 binary, permutation, combinatorial 문제에 특히 효과적임을 강조합니다. 알고리즘 선택은 탐색(exploration)과 활용(exploitation)의 균형을 맞추고, local minima에 빠지지 않도록 하는 것이 중요합니다.

- **Performance Highlights**: 결과적으로, MIMIC과 GA는 binary 문제와 combinatorial 문제에서 높은 품질의 솔루션을 제공합니다. 그러나, GA는 정확성과 효율성 사이에서 균형을 이루며, MIMIC은 permutation 문제에서 우수한 성과를 보입니다. 반면, RHC와 SA는 연산 비용이 적으나 복잡한 문제에서 성능 저하를 겪는 것으로 나타났습니다.



New uploads on arXiv(cs.IR)

### WARP: An Efficient Engine for Multi-Vector Retrieva (https://arxiv.org/abs/2501.17788)
- **What's New**: 이 논문에서는 ColBERT와 그 변형 XTR 기반의 다중 벡터 검색 방법의 효율성을 개선하기 위한 새로운 검색 엔진 WARP를 소개합니다. WARP는 세 가지 주요 혁신을 통해 XTR 기반 ColBERT 검색기의 효율성을 대폭 향상시킵니다: 첫째, 동적 유사도 대체를 위한 WARP$_{SELECT}$, 둘째, 복잡한 벡터 복원을 우회하는 암묵적 압축, 셋째, 효율적인 점수 산정을 위한 두 단계 감소 프로세스입니다. 이러한 최적화와 전문화된 추론 런타임을 결합하여 WARP는 XTR의 참조 구현 대비 41배의 응답 속도 개선을 달성하며, PLAID에 비해 3배의 속도 향상을 이뤄냈습니다.

- **Technical Details**: WARP의 구현은 ColBERTv2 및 PLAID 기법을 XTR 아키텍처에 맞게 조합하였습니다. 특정한 기술로는 WARP$_{SELECT}$ 방법을 통해 결측 유사도를 채우고, 검색 중 벡터의 암묵적 압축을 수행하며, 효율적인 점수를 위한 새로운 두 단계 감소 방법을 포함하고 있습니다. 실험 평가 결과, WARP는 LoTTE Pooled 데이터 세트에서 XTR 참조 구현 대비 41배의 끝-끝(latency) 지연을 줄이는데 성공했습니다.

- **Performance Highlights**: WARP는 쿼리 응답 시간을 6초 이상에서 단일 스레드 실행 시 171 밀리초로 단축시키며, ScaNN 기반 벤치마크와 비교하여 인덱스 크기를 2배에서 4배까지 줄였습니다. 또한, WARP는 최신 ColBERTv2/PLAID 시스템에 비해 3배의 속도 향상을 제공하면서도 검색 품질을 유지할 수 있음을 보여줍니다. 이러한 성능 향상은 다중 벡터 검색의 시간 및 공간 효율성을 크게 개선하는데 기여할 것입니다.



### Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation (https://arxiv.org/abs/2501.17670)
- **What's New**: 이 논문에서는 Diffusion models (DMs) 기반의 시퀀스 추천 시스템에서 발생하는 두 가지 주요 문제를 해결하기 위한 새로운 접근법인 DiQDiff를 제안합니다. 기존의 추천 시스템이 사용자 행동에 따른 시퀀스를 단순히 노이즈 처리하는 방식에 반해, DiQDiff는 더 주민적인 방향으로 사용자 관심사를 이해하고 개인화된 항목을 생성하도록 돕습니다. 이 모델은 특히 다양한 사용자의 요구를 충족시키기 위해 개선되었습니다.

- **Technical Details**: DiQDiff는 Semantic Vector Quantization (SVQ) 기술을 도입하여, 시퀀스를 의미 벡터(semantic vectors)로 양자화합니다. 이 코드북(codebook)을 사용함으로써 사용자 성향을 반영한 추가적인 지침을 제공합니다. 또한, DiQDiff는 Contrastive Discrepancy Maximization (CDM)을 통해 비편향적인 생성(generation)을 보장하며, 대조 손실(contrastive loss)을 극대화하여 사용자마다 차별화된 항목을 생성하는 방식입니다.

- **Performance Highlights**: DiQDiff는 네 개의 잘 알려진 데이터셋에서 여러 기준 모델들과 비교 실험을 수행하였고, 그 결과 우수한 추천 성능을 보여주었습니다. 이 연구의 결과는 DMs가 시퀀스 추천 과제에서 사용자 개인화 요구를 효과적으로 충족할 수 있음을 시사합니다. 따라서, DiQDiff는 추천 알고리즘의 새로운 기준을 제시합니다.



### Uncertainty Quantification and Decomposition for LLM-based Recommendation (https://arxiv.org/abs/2501.17630)
Comments:
          WWW 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 추천 생성 과정에서 불확실성을 자주 보이는 점을 지적하고 있습니다. LLMs의 신뢰성을 높이기 위해 추천의 신뢰성을 평가하는 것이 중요하다는 점을 강조합니다. 새로운 프레임워크를 소개하여 예측 불확실성을 정량적으로 측정할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 예측 불확실성을 추천 불확실성(recommendation uncertainty)과 프롬프트 불확실성(prompt uncertainty)으로 분해하여 분석할 수 있는 새로운 방식을 소개합니다. 이를 통해 LLM 기반 추천에서 불확실성의 주 원인을 심층적으로 분석할 수 있게 됩니다.

- **Performance Highlights**: 실험을 통해 예측 불확실성이 LLM 기반 추천의 신뢰성을 효과적으로 나타낸다는 것을 입증합니다. 또한, 분해된 불확실성 측정을 통해 불확실성의 근원을 조사하고, 예측 불확실성을 줄이고 추천을 향상시키기 위한 불확실성 인식(prompting) 방안을 제시합니다.



### Value Function Decomposition in Markov Recommendation Process (https://arxiv.org/abs/2501.17409)
Comments:
          14 pages, 9 figures

- **What's New**: 본 연구에서는 추천 시스템에서 사용자-시스템 상호작용이 장기 최적화 문제로 형성된다는 점을 강조합니다. 온라인 강화 학습(online reinforcement learning) 기법을 통해 추천 성과를 향상시키는 새로운 접근 방법을 제안합니다. 이 방법은 서로 다른 상태(s)에서 두 개의 랜덤 요인(policy의 랜덤 행동 탐색 및 불확실한 사용자 환경)을 분리하여 가치 함수(value function)를 더 정확히 추정하는 기능을 가집니다.

- **Technical Details**: 연구에서 제안하는 분리된 학습(framework)은 임의의 정책과 사용자 환경의 영향을 배제한 두 개의 하위 문제로 TD 학습(temporal difference learning)을 분해합니다. 첫 번째 하위 문제는 사용자 상태의 장기 유용성(long-term utility)을 정확히 추정하는 데 중점을 두고, 두 번째는 추천 행동의 효과성을 포착하는 주(state-action pair)에 대한 정교한 함수를 세분화합니다. 이러한 분리된 접근법은 TD 학습의 원래 목표를 경계(bound)하며 학습 과정을 가속화할 수 있습니다.

- **Performance Highlights**: 제안된 방법의 성능을 입증하기 위해, 여러 TD 학습 기반 방법을 사용한 다수의 오프라인 실험(offline experiments)이 실시되었습니다. 제안된 분해 기법은 행동 탐색(action exploration)에서 뛰어난 성능을 발휘하며, 극단적인 경우에도 추천 성능을 최적화할 수 있는 추가적인 장점을 보여주었습니다. 이를 통해, 정책이 탐색을 과도하게 하더라도 제안된 방법이 더 효과적으로 가치 함수를 최적화할 수 있음을 확인하였습니다.



### Aggregation Schemes for Single-Vector WSI Representation Learning in Digital Pathology (https://arxiv.org/abs/2501.17822)
- **What's New**: 이 논문에서는 Whole Slide Image(WSI) 검색 성능을 평가하기 위해 여러 최근 개발된 집계 기법을 비교합니다. 특히, WSI의 고해상도 특성 때문에 대규모 패치 라벨링 및 특정 집합 표현 학습 기술을 탐구합니다. 기존의 패치 임베딩 세트를 단일 임베딩으로 변환하는 다양한 방법을 제안하며 성능을 비교하고 있습니다.

- **Technical Details**: WSI는 일반적으로 수많은 작은 "패치"나 타일로 분할되며, 이를 통해 각 WSI에 대해 패치 임베딩 세트를 생성합니다. 이 과정에서 다양한 집계 알고리즘이 사용되며, 각 알고리즘은 패치 임베딩 세트에서 단일 벡터로 깊은 특징을 추출합니다. Fisher Vector, Deep Sets, Memory Networks, Focal Attention 등 다양한 접근 방식을 사용하여 WSI의 임베딩 단일화 및 표준화를 목표로 하고 있습니다.

- **Performance Highlights**: 연구에서는 TCGA에서 제공하는 방광, 유방, 신장, 대장과 같은 여러 데이터셋을 통해 k-NN 검색 방식을 적용하여 성능을 평가했습니다. 또한, 이들 방법의 검색 성능을 실제 WSI 검색 시스템에 적용됨에 따라 메디안 최소 거리 기반 접근 방식과 비교하여 더 나은 결과를 도출된 것으로 나타났습니다. 최종적으로 각 방식의 단일 벡터 생성 성능을 분석하고 결과를 상세히 논의합니다.



### Leveraging Multimodal LLM for Inspirational User Interface Search (https://arxiv.org/abs/2501.17799)
Comments:
          In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '25)

- **What's New**: 이번 연구에서는 모빌리티 UI 디자인에서 설계 영감을 위한 검색 프로세스를 개선하고자 다중 모달 대형 언어 모델(MLLM)을 활용한 새로운 접근 방식을 제안합니다. 기존의 UI 검색 방법들이 놓치는 핵심 의미 요소들을 효과적으로 추출하고 해석하여, 실질적이고 많은 사용자의 요구를 충족시키는 검색 경험을 제공합니다. 연구 결과, 우리의 방법이 기존의 UI 검색 모델들보다 의미적으로 훨씬 더 우수하다는 점을 확인했습니다.

- **Technical Details**: 이 방식은 UI 이미지에서 직접 의미 정보를 추출하여 추가적인 메타데이터에 의존하지 않고도 UI 설계의 다양한 요소를 이해하고 설명할 수 있는 가능성을 제공합니다. 연구는 UI 디자이너가 중요하게 여기는 주요 의미 요소를 파악하고, MLLM을 통한 의미 정보의 추출 및 구조화 방법론을 수립하는 데 중점을 두었습니다. 이러한 방법론은 정량적 및 정성적 평가를 통해 그 효과성을 입증했습니다.

- **Performance Highlights**: 우리의 MLLM 기반 의미 검색 시스템은 관련성, 다양성, 신뢰성, 유용성 및 우연성 등의 여러 측면에서 기존의 검색 방법에 비해 현저한 개선을 보여주었습니다. 이러한 개선 사항은 디자인 결정을 보다 효과적으로 이끌며, 디자이너들이 보다 풍부하고 맥락적으로 관련있는 UI 참고자료를 쉽게 찾을 수 있도록 돕습니다. 앞으로의 연구에 기여할 수 있는 방대한 UI 의미 데이터셋도 제공합니다.



### Cross-Language Approach for Quranic QA (https://arxiv.org/abs/2501.17449)
- **What's New**: 이 연구는 언어 자원이 제한된 언어를 대상으로 한 질문 응답 시스템의 주요 한계를 극복하기 위한 새로운 접근 방식을 제시합니다. 특히, 고전 아랍어로 작성된 꾸란 구절과 현대 표준 아랍어(MSA)로 작성된 질문 간의 언어적 격차를 해소하기 위해 교차 언어 접근 방식이採用되었습니다. 데이터 세트를 기계 번역을 통해 영어로 확장하고, 다양한 질문 표현을 생성하며, 꾸란의 영어 번역에서 답변을 찾는 방식으로 모델 성능을 향상시킵니다.

- **Technical Details**: 이 연구는 데이터 세트 준비, 교차 언어 처리, 모델 미세 조정의 세 가지 주요 단계를 포함합니다. 데이터 세트는 아랍어 질문을 영어로 번역하고, 문맥에 맞는 질문을 생성하기 위해 재구 성 및 추가로 문제를 생성하여 다양성을 향상시켰습니다. 또한 BERT-Medium, RoBERTa-Base, DeBERTa-v3-Base, ELECTRA-Large, Flan-T5, Bloom 및 Falcon과 같은 최첨단 모델을 미세 조정하여 꾸란 QA 작업의 성과를 최적화합니다.

- **Performance Highlights**: 실험 결과, RoBERTa-Base는 MAP@10(0.34) 및 MRR(0.52)에서 가장 높은 성과를 나타냈으며, DeBERTa-v3-Base는 Recall@10(0.50)과 Precision@10(0.24)에서 우수한 결과를 기록했습니다. 이러한 성과는 다양한 현대 아랍어 질문에 답하기 위해 필요한 고전 아랍어 구절을 효과적으로 캡처하고 언어적 장벽을 극복하는 교차 언어 접근 방식의 효과를 증명합니다.



### Aspect-Aware Decomposition for Opinion Summarization (https://arxiv.org/abs/2501.17191)
Comments:
          35 pages

- **What's New**: 이번 연구에서는 대규모 온라인 리뷰에서 의미 있는 인사이트를 도출하기 위한 의견 요약(opinion summarization)의 모듈형 접근법을 제안합니다. 이 접근법은 리뷰의 다양한 측면(aspect)에 의해 안내되어, 측면 식별(aspect identification), 의견 통합(opinion consolidation), 메타 리뷰 합성(meta-review synthesis)의 작업을 분리하여 투명성과 점검 용이성을 증가시킵니다.

- **Technical Details**: 연구는 과학 연구, 비즈니스 및 제품 영역을 대표하는 데이터셋에서 광범위한 실험을 수행하였습니다. 제안된 방법은 자동화 및 인간 평가를 통해 강력한 기준 모델(baseline models)보다 더 근거 있는(grounded) 요약을 생성하는 것으로 확인되었습니다. 또한, 리뷰 측면에 기반한 추론(reasoning)을 포함한 모듈형 접근법은 지식 비귀속(decomposed prompting)에 비해 더 유익한 중간 출력(intermediate outputs)을 생산합니다.

- **Performance Highlights**: 중간 출력은 대량의 리뷰에서 의견을 요약하는 데 있어 인간에게 효과적으로 지원할 수 있습니다. 연구 결과, 이러한 모듈형 접근법이 더 많은 정보를 제공하며, 종합적인 의견 요약 방식에서 더욱 개선된 성과를 낸다는 점이 강조됩니다.



### An AI-Driven Live Systematic Reviews in the Brain-Heart Interconnectome: Minimizing Research Waste and Advancing Evidence Synthesis (https://arxiv.org/abs/2501.17181)
- **What's New**: Brain-Heart Interconnectome (BHI)를 위한 AI 기반 시스템이 개발되었습니다. 이 시스템은 PICOS(인구, 개입, 비교군, 결과 및 연구 설계) 기준을 자동으로 감지하고, 의미 검색과 그래프 기반 쿼리를 통해 문헌을 효율적으로 검색하고 정리합니다. BI-LSTM 모델, 연구 설계 분류기 및 GPT-3.5 기반의 Retrieval-Augmented Generation (RAG) 방법이 도입되어 연구 자원의 낭비를 줄이고 실시간 업데이트를 제공합니다.

- **Technical Details**: 이 시스템은 복잡한 BHI 데이터를 관리하기 위해 Neo4j를 활용하여 관련된 개체들 간의 관계를 그래프 형태로 표현합니다. 또한, pgVector를 이용해 문서 임베딩을 저장하고, LangChain을 통해 사용자 입력에 맞춘 쿼리 처리가 이루어집니다. Bi-LSTM 모델은 연구의 PICOS 준수 여부를 판단하고, BERTopic은 주제를 클러스터링하여 연구의 진화를 추적합니다.

- **Performance Highlights**: 시스템은 BHI 분야에서 신뢰할 수 있는 증거 합성을 지원하며, 연구자들에게 실시간으로 새로운 자료를 통합할 수 있는 기능을 제공합니다. Power BI 대시보드를 통해 사용자들은 출판 패턴 및 연구 동향을 시각적으로 확인할 수 있으며, 이는 효율적인 자원 배분 및 임상 의사결정 지원에 기여합니다. 이러한 기술들은 다른 생물 의학 분야에도 적용 가능하다는 장점을 지니고 있습니다.



### Document-Level Sentiment Analysis of Urdu Text Using Deep Learning Techniques (https://arxiv.org/abs/2501.17175)
- **What's New**: 이 연구에서는 문서 수준에서 우르두어 감정 분석(Sentiment Analysis, SA)을 위한 새로운 하이브리드 딥러닝 모델인 BiLSTM-SLMFCNN을 제안하고 있습니다. 처음으로 우르두어 데이터를 위한 딥러닝 아키텍처들을 적용하여 성과를 입증하는 데 초점을 맞추었습니다. 특히, 기존의 전통적인 기계 학습 모델보다 딥러닝 모델이 더욱 효과적이며, 문서 크기에 따른 성능 변화를 함께 분석하였습니다.

- **Technical Details**: 제안된 BiLSTM-SLMFCNN 모델은 Bidirectional Long Short Term Memory(BiLSTM)와 Single Layer Multi Filter Convolutional Neural Network(SLMFCNN)를 결합하여 우르두 문서의 감정을 분류하는 능력을 극대화합니다. 이 모델은 BiLSTM을 통해 단어의 맥락적 의미를 이해하고 SLMFCNN을 통해 지역적 특징을 추출합니다. 또한, 여러 개의 필터를 사용하여 다양한 길이의 n-그램 특징을 추출할 수 있는 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 IMDB 우르두 영화 리뷰 데이터 세트와 우르두 고객 지원 데이터 세트에서 성능을 평가했으며, 각각 83%, 79%, 83% 및 94%의 정확도를 기록하여 기존의 딥러닝 모델을 능가했습니다. 특히, BiLSTM-SLMFCNN은 문서 수준 감정 분석에서 기존의 기계 학습 모델보다 우수한 결과를 보여주었습니다.



New uploads on arXiv(cs.CV)

### U2A: Unified Unimodal Adaptation for Robust and Efficient Multimodal Learning (https://arxiv.org/abs/2501.17823)
Comments:
          14 Pages, 6 Figures, 6 Tables

- **What's New**: 이번 논문에서는 Unified Unimodal Adaptation (U2A)라는 새로운 방법을 제안합니다. 이는 낮은 순위의 적응(low-rank adaptation, LoRA)을 활용하여 미리 훈련된 단일 모달 인코더를 조정함으로써 다양한 다중 모달 작업을 수행합니다. U2A는 복잡한 훈련 전략 없이 필요한 학습 가능한 매개변수를 크게 줄이며, 특히 훈련 및 테스트 시 결여된 모달을 다룰 수 있는 Mask Tokens (MT)도 도입합니다.

- **Technical Details**: U2A는 미리 훈련된 transformer 인코더를 통해 각 모달을 별도로 인코딩합니다. 결여된 모달을 처리하기 위해 Mask Tokens를 도입하여 사용 가능한 입력 모달로부터 결여된 모달 특징을 추정합니다. 여기에는 모달 정렬 손실(modality alignment loss)도 포함되어, 결여된 모달의 클래스 토큰을 효과적으로 대체할 수 있도록 학습합니다.

- **Performance Highlights**: U2A는 테스트에서 다중 모달 환경에서 우수한 성능을 보이며, 많은 기존 방법들보다 더 적은 학습 가능한 매개변수로도 성능을 유지합니다. 다양한 데이터셋에 대한 실험 결과, U2A는 완전한 모달 및 결여된 모달 상황 모두에서 최첨단(state-of-the-art, SOTA) 방법을 초과하거나 동등한 성능을 기록했습니다. 실험을 통해, 결여된 모달 특성을 정확하게 추정하는 능력도 입증되었습니다.



### SSF: Sparse Long-Range Scene Flow for Autonomous Driving (https://arxiv.org/abs/2501.17821)
Comments:
          7 pages, 3 figures, accepted to International Conference on Robotics and Automation (ICRA) 2025

- **What's New**: 이 논문에서는 Sparse Scene Flow (SSF)라는 새로운 장거리(scene flow) 추정 방법을 제안합니다. 기존의 복잡한 설계를 단순화하고 희소(convolution) 합성을 통한 특성 추출 방식을 채택하였으며, 이는 장거리에서의 성능을 크게 향상시킵니다. 이러한 접근 방식은 기존의 방법들이 다루지 못했던 50m 이상의 거리에서의 scene flow 추정이 가능하도록 합니다.

- **Technical Details**: SSF는 희소(convolution) 합성을 기반으로 한 파이프라인으로, 시간에 따른 포인트 스캔 간의 희소 특성 맵의 크기와 순서를 일치시키기 위한 새로운 기능 융합 방안을 갖추고 있습니다. 이는 시간 순서에 따라 특성 벡터의 불일치를 극복하고 희소한 특성을 효과적으로 융합할 수 있도록 합니다. 또한, 제안된 방법은 새로운 범위 기반의 평가지표인 Range-wise EPE를 사용하여 scene flow의 정확도를 평가합니다.

- **Performance Highlights**: SSF는 Argoverse2 데이터셋에서 최신 기술(state-of-the-art) 성능을 달성하였으며, 특히 장거리(scene flow) 추정에서 강력한 성능을 발휘합니다. 희소(convolution) 구조로 인한 계산 효율성 덕분에, 우리의 방법은 높은 정확도를 유지하면서도 메모리 소비를 최소화할 수 있습니다. 이러한 성과는 자율주행 시스템의 전반적인 안전성 향상에 기여할 것입니다.



### P-TAME: Explain Any Image Classifier with Trained Perturbations (https://arxiv.org/abs/2501.17813)
Comments:
          Submitted for publication

- **What's New**: 이 논문에서는 DNN 기반 이미지 분류기를 설명하기 위한 모델 불문(모델-아그노스틱) 방법인 P-TAME (Perturbation-based Trainable Attention Mechanism for Explanations)를 소개합니다. P-TAME는 보조 이미지 분류기를 활용하여 입력 이미지의 특징을 추출하고, DNN의 내부 아키텍처에 맞게 설명 방법을 조정할 필요 없이 설명을 생성합니다. 전통적인 방 perturbation-based 방법의 높은 계산 요구 사항을 해결하고, 단일 forward pass에서 높은 해상도의 설명을 생성하는 효율적인 대안을 제공합니다.

- **Technical Details**: P-TAME는 대체 이미지 분류기를 통해 기능 맵을 추출하며, 이를 통해 DNN의 주요 결정과 관련된 부분을 강조하는 방식으로 설명을 생성합니다. 이 방법은 여러 DNN 아키텍처에 적용 가능하여 모델-아그노스틱한 특성을 가지며, 훈련 후 단일 forward 단계에서 설명을 제공합니다. 실험에 사용된 이미지 분류기로는 VGG-16, ResNet-50 및 ViT-B-16이 있으며, 이들은 모두 ImageNet에 대해 훈련되었습니다.

- **Performance Highlights**: P-TAME의 성능은 이전의 설명 가능성 방법, 특히 T-TAME 및 기타 최신 기술과 비교하여 양적(quantitative) 및 질적(qualitative) 결과가 뛰어난 것을 보여줍니다. 이러한 비교는 P-TAME가 설명의 질에서 최신 perturbation 방식과 견주어 최대한의 성능을 발휘할 수 있음을 나타냅니다. 또한, 연구자들이 이 방법을 쉽게 사용할 수 있도록 P-TAME를 오픈 소스로 제공하고 있습니다.



### CrowdSplat: Exploring Gaussian Splatting For Crowd Rendering (https://arxiv.org/abs/2501.17792)
Comments:
          4 pages, 4 figures

- **What's New**: 본 논문에서는 실시간 고품질 군중 렌더링을 위한 새로운 접근법인 CrowdSplat을 제시합니다. CrowdSplat은 3D Gaussian Splatting을 활용하여 다양한 포즈와 의상을 가진 인간 캐릭터를 애니메이션화합니다. 기존 방식의 데이터 소모와 수작업 편집의 부담을 덜어주는 동시에, Level of Detail (LoD) 렌더링을 통해 계산 효율성과 품질을 최적화합니다.

- **Technical Details**: CrowdSplat은 두 단계로 구성되어 있습니다: 첫 번째는 아바타 재구성, 두 번째는 군중 합성입니다. 14개의 아바타 템플릿을 재구성하고 이러한 템플릿을 사용하여 3,500개의 캐릭터로 구성된 군중 시스템을 구축합니다. GPU 메모리 사용량을 최적화하며, 다양한 거리에서 3D Gaussians 수를 조절하여 실시간 렌더링 품질을 확보합니다. 이를 통해 31 fps의 렌더링 속도를 달성합니다.

- **Performance Highlights**: CrowdSplat의 성능 평가는 정성적 및 정량적 방법으로 이루어졌습니다. 3D Gaussians 수에 따라 PSNR 및 LPIPS와 같은 지표를 사용하여 렌더링 품질을 평가했습니다. 결론적으로, CrowdSplat은 다이나믹한, 사실적인 군중 시뮬레이션을 실시간으로 가능하게 하며, 향후 3DGS LoD 기술에 대한 사용자 연구 및 군중 장면 생성 방법 개발을 목표로 하고 있습니다.



### VICCA: Visual Interpretation and Comprehension of Chest X-ray Anomalies in Generated Report Without Human Feedback (https://arxiv.org/abs/2501.17726)
- **What's New**: 이 연구에서는 의료 리포트 생성의 신뢰성과 해석 가능성을 향상시키기 위한 새로운 다중 모달 프레임워크를 제안합니다. 현재의 흉부 X-ray(CXR) 보고 시스템의 문제점을 해결하기 위해, 본 프레임워크는 패러다임(phrase grounding) 모델과 텍스트-이미지 확산 모듈(text-to-image diffusion module)을 결합하여 AI가 생성한 의료 리포트를 검증할 수 있는 메커니즘을 제공합니다. 이를 통해 병리학적 구조 접근성과 의미적 일치를 평가하는 이중 평가 시스템을 도입했습니다.

- **Technical Details**: 본 연구의 다중 모달 틀은 패스톨로지(patology)를 식별하고 CXR 이미지에서 지역을 국소화하는 Phrase Grounding Model과 텍스트 프롬프트에 따라 합성 CXR 이미지를 생성하는 Text-to-Image Diffusion Module로 구성됩니다. 이 두 구성 요소를 통해 생성된 CXR 이미지의 원본과 합성 이미지의 특징을 비교하고, 지역 정확성(localization accuracy)과 의미적 일관성(semantic consistency)을 각각 평가하는 두 가지 점수를 도출합니다.

- **Performance Highlights**: 제안된 모델은 기존 방법들보다 뛰어난 성능을 보이며, 병리학적 국소화(pathology localization)와 텍스트-이미지 정렬(text-to-image alignment)에서 최첨단 결과를 달성했습니다. 새로운 검증 시스템은 전문가의 피드백 없이도 객관적이고 재현 가능한 검증 과정을 보장하며, CXR 분석 분야에서 다중 모달 AI의 필요성을 강조하고 있습니다. 이 접근법은 의료 데이터의 해석 가능성과 신뢰성을 제고하는 데 중요한 기여를 하고 있습니다.



### Learning Semantic Facial Descriptors for Accurate Face Animation (https://arxiv.org/abs/2501.17718)
Comments:
          6 pages,6 figures

- **What's New**: 이 논문에서는 얼굴 애니메이션(facial animation)이라는 어려운 작업을 다루며, 기존의 모델 기반(method) 방법의 단점을 극복할 새로운 접근법을 제안합니다. 특히, 3DMMs 또는 landmarks를 이용한 기존 방법들이 정체성을 충분히 보존하지 못하는 문제를 해결하기 위해, 학습 가능한 분리된 벡터 공간에서 의미론적 얼굴 설명자(semantic facial descriptors)를 도입합니다.

- **Technical Details**: 제안된 방법은 얼굴 공간을 정체성(identity)과 움직임(motion) 하위 공간으로 분리하고, 각 하위 공간에 대해 의미를 부여하는 방식으로 접근합니다. 이 과정에서 완전 정규 직교 기저 벡터(orthogonal basis vectors)를 학습하여, 소스(face)와 구동(face) 얼굴의 특징을 효과적으로 추출합니다. 이를 통해 얻은 기저 벡터 계수(basis vector coefficients)는 얼굴 애니메이션을 위한 잠재 코드(latent codes)로 재조합될 수 있습니다.

- **Performance Highlights**: 제안된 방법은 VoxCeleb, HDTF, CelebV와 같은 세 가지 어려운 벤치마크에서 폭넓은 실험을 통해 검증되었습니다. 수치적이고 질적인 결과 모두에서, 우리의 모델은 SOTA(state-of-the-art) 방법들과 비교하여 정체성 보존(identity preservation) 및 움직임 전이(motion transfer)에서 뛰어난 성능을 보임을 입증하였습니다.



### Segmentation-Aware Generative Reinforcement Network (GRN) for Tissue Layer Segmentation in 3-D Ultrasound Images for Chronic Low-back Pain (cLBP) Assessmen (https://arxiv.org/abs/2501.17690)
- **What's New**: 이 논문에서는 segmentation-aware joint training framework인 generative reinforcement network (GRN)을 소개합니다. 이 프레임워크는 segmentation loss 피드백을 통합하여 이미지 생성과 segmentation 성능을 동시에 최적화할 수 있도록 설계되었습니다. 또한, segmentation-guided enhancement (SGE)라는 이미지 향상 기법을 개발하여, generator가 segmentation 모델에 맞춘 이미지를 생성하게 합니다.

- **Technical Details**: GRN은 두 가지 변형 버전이 개발되었으며, 각각 sample-efficient learning (GRN-SEL)과 semi-supervised learning (GRN-SSL)입니다. 연구에 사용된 데이터셋은 29명의 피험자로부터 얻은 69개의 3D 초음파 스캔으로, 여섯 가지 해부학적 구조(dermis, superficial fat, superficial fascial membrane (SFM), deep fat, deep fascial membrane (DFM), muscle)가 포함되어 있습니다. GRN은 이러한 구조에 대한 라벨링 노력을 줄이기 위해 최적화되었습니다.

- **Performance Highlights**: GRN-SEL은 SGE를 사용할 경우 라벨링 노력을 최대 70%까지 줄이면서 Dice Similarity Coefficient (DSC)에서 1.98% 개선을 보였습니다. GRN-SEL 단독은 라벨링 노력을 60% 줄였고, GRN-SSL의 경우 SGE를 사용할 경우 라벨링 요구 사항을 70% 감소시켰습니다. 이러한 결과는 GRN 프레임워크가 적은 라벨 데이터로 segmentation 성능을 최적화하는 데 효과적임을 시사하며, 초음파 이미지 분석에 있어 데이터 주석의 부담을 줄이는 스케일 가능한 솔루션을 제안합니다.



### ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer (https://arxiv.org/abs/2501.17688)
- **What's New**: 이 논문에서는 Contourformer라는 실시간 윤곽 기반 인스턴스 분할 알고리즘을 제안합니다. 이 방법은 DETR(paradigm)으로 완전히 기반하여 반복적이고 점진적인 메커니즘을 통해 윤곽을 최적화하여 최종 결과를 end-to-end 방식으로 얻습니다. 또한 효율성과 정확성을 향상시키기 위해 서브 윤곽 분리 메커니즘과 윤곽 세밀한 분포 개선 두 가지 혁신적인 기술을 개발했습니다.

- **Technical Details**: Contourformer의 프레임워크는 D-FINE 객체 탐지 모델을 기반으로 하여 바운딩 박스의 회귀를 윤곽 회귀로 확장합니다. 이 구조는 효율적인 훈련을 위해 윤곽 변형의 반복적 방법을 사용하며, 수렴 속도를 높이기 위해 노이즈 제거 메커니즘을 도입합니다. 모델은 각 반복에서 윤곽 추정의 점진적인 개선을 가능하게 하기 위해 DETR 디코더의 반복 아키텍처를 재설계했습니다.

- **Performance Highlights**: Contourformer는 SBD, COCO 및 KINS와 같은 여러 벤치마크 데이터세트에서 뛰어난 성능을 보여줍니다. 실제로 512x512 크기의 이미지에서 NVIDIA A30 GPU를 이용해 24.6 fps의 추론 속도를 달성하였으며, 기존의 윤곽 기반 인스턴스 분할 알고리즘보다 정확도가 현저히 향상된 결과를 보였습니다.



### FeatureGS: Eigenvalue-Feature Optimization in 3D Gaussian Splatting for Geometrically Accurate and Artifact-Reduced Reconstruction (https://arxiv.org/abs/2501.17655)
Comments:
          16 pages, 9 figures, 7 tables

- **What's New**: FeatureGS는 3D Gaussian Splatting(3DGS) 과정에 추가적인 기하학적 손실 항을 통합한 새로운 접근법입니다. 이를 통해 Gaussian의 기하학적 정확성을 향상시키고, 고유 값 기반의 3D 형태 기능을 활용하여 플래너(planar) 표면의 특성을 개선합니다. 기존의 문제점인 floater artifacts를 억제하고, 대칭성과 메모리 효율성을 높이는 방법을 제시합니다.

- **Technical Details**: FeatureGS는 3D Gaussian의 최적화 과정에서 네 가지의 기하학적 손실 항을 통합합니다. 특히, k-nearest neighbors(kNN)로부터 유도된 근방 기반의 3D 특징을 활용하여 주로 평면적 표면을 강화합니다. 이를 통해 Gaussian의 중심을 이용한 정확한 기하학적 표현을 가능하게 하며, 리소스 요구 사항을 줄입니다.

- **Performance Highlights**: 실험 결과 FeatureGS는 기하학적 정확성을 30% 향상시키고, Gaussian의 수를 90% 줄이면서 floater artifacts를 억제했습니다. 그 외에도 Peak Signal-to-Noise Ratio(PSNR)를 통해 렌더링 품질을 유지하며, 15개의 DTU 벤치마크 데이터 세트를 통해 성능을 검증했습니다. 이는 FeatureGS가 메모리 효율적이며, 정확한 기하학적 재구성을 가능하게 함을 보여줍니다.



### Efficient Redundancy Reduction for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2501.17642)
- **What's New**: ERR-Seg는 중복성을 줄이면서 성과와 효율성을 균형 맞춘 새로운 Open-vocabulary semantic segmentation (OVSS) 프레임워크이다. 이 연구는 Channel Reduction Module (CRM)을 도입하여 가장 관련성이 높은 클래스를 식별하고, 중복성을 줄여 성능과 효율성을 동시에 개선한다. 또한, Hierarchical Semantic Module (HSM)을 통합하여 중간 계층에서 추출된 계층적 의미를 활용한다.

- **Technical Details**: ERR-Seg는 효율적인 Channel Reduction Module (CRM)과 Efficient Semantic Context Fusion (ESCF)을 사용하여 메모리와 계산량을 줄인다. CRM은 시각-언어 모델 CLIP으로부터 지식을 활용하여 가장 관련성 높은 카테고리를 선택하고 나머지는 폐기한다. ESCF는 공간 수준 및 클래스 수준에서 시퀀스 축소 전략을 통해 계산 부담을 49.3% 줄인다.

- **Performance Highlights**: ERR-Seg는 ADE20K-847 설정에서 이전 최첨단 방법들보다 mIoU를 5.6% 향상시켰고, 지연 시간은 67.3% 감소시켰다. 이로써 ERR-Seg는 효율성과 정확성 모두에서 새로운 기준을 수립하였다. 여러 데이터셋에서 우수한 성능과 함께 추론 과정을 크게 가속화하는 점이 주목할 만하다.



### Efficient Interactive 3D Multi-Object Remova (https://arxiv.org/abs/2501.17636)
- **What's New**: 이 논문에서는 3D 다중 객체 제거를 위한 새로운 효율적이고 사용자 친화적인 파이프라인을 제안합니다. 이를 통해 사용자는 객체를 제거하거나 보존할 지역을 유연하게 선택할 수 있습니다. 또한, 고신뢰 기준점과 호모그래피 기반 변형을 통합하여 마스크의 일관성을 확보하는 새로운 모듈이 도입되었습니다.

- **Technical Details**: 본 연구는 키포인트 대응을 설정하기 위해 미리 훈련된 LoFTR를 활용하며, 이를 바탕으로 여러 뷰 이미지 간의 객체 마스크를 매핑합니다. 이 과정에서 IoU(Intersection over Union)와 형태 거리 제약을 이용해 앵커 포인트를 조절함으로써 다중 뷰 이미지 간의 일관성을 확보합니다. 이러한 접근은 깊이 데이터나 다중 뷰 포즈 정보 없이도 실행 가능하여 실용성을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 기술보다 80% 이상 빠른 처리 속도를 달성하면서도 동등하거나 더 높은 재구성 품질을 유지합니다. 또한, 제안된 데이터셋에서의 성능도 뛰어나며, 실세계 개방 시나리오에서도 효과적으로 적용될 수 있음을 입증하였습니다.



### Technical report on label-informed logit redistribution for better domain generalization in low-shot classification with foundation models (https://arxiv.org/abs/2501.17595)
- **What's New**: 이 연구는 CLIP와 같은 foundation models(기초 모델)의 신뢰성(calibration) 문제를 다루기 위해 새로운 	extit{confidence misalignment penalty (CMP)}를 제안합니다. CMP는 잘못된 분류가 발생할 때 손실 목표에 페널티를 추가하여 잘못된 분류에 대한 신뢰도를 감소시킵니다. 또한, 다양한 비전 데이터셋에서 광범위한 실험을 통해 CMP의 효율성을 입증하였으며, 기존의 방법들보다 평균 6.01%의 개선된 Expected Calibration Error (ECE)를 달성했습니다.

- **Technical Details**: CMP는 잘못된 분류를 할 때 posterior likelihood(후방 확률)의 흐름을 조절하여 진짜 클래스의 확률에 대한 페널티를 걸어줍니다. 이 방식은 데이터 증강(data augmentation) 또는 배치/레이어 정규화(batch/layer normalization)와 같은 변환에 대해 본질적으로 불변성을 가집니다. 이 연구는 또한 네트워크 보정(calibration), 신뢰성(trustworthiness), 일반화(generalization) 측면에서 예측 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: CMP는 가장 최신의 프롬프트 학습(prompt learning) 기법들과 비교하여 성능이 뛰어난 결과를 보여주었습니다. 특히, 기존의 방법에 비해 평균 6.01%의 ECE 개선을 달성하였고, 최소 4.01%에서 최대 9.72%의 범위에서 개선 효과를 보였습니다. 이 연구는 모델의 신뢰성을 높이고 윤리적 의사결정의 신뢰도를 향상시키는 데 도움을 주는 것을 목표로 하고 있습니다.



### Boosting Weak Positives for Text Based Person Search (https://arxiv.org/abs/2501.17586)
- **What's New**: 본 연구에서는 텍스트 기반 개인 검색(Text-Based Person Search, TBPS)에 도전하는 한계점을 극복하기 위한 혁신적인 부스팅 기법(Boosting Technique)을 도입했습니다. 기존 연구들이 이미지-텍스트 쌍을 공통 표현 공간으로 정렬하는 데 중점을 둔 반면, 우리는 훈련 중에 어려운 쌍을 도입하여 이들에 대한 가중치를 높이는 방식으로 접근했습니다. 이러한 방법은 특히 잘못 정렬된 이미지-텍스트 쌍에 가중치를 두어 모델이 이러한 예제에 더 많은 주의를 기울이도록 합니다.

- **Technical Details**: 논문에서는 아다부스트(AdaBoost)와 같은 전통적인 부스팅 기법을 바탕으로, 동적으로 훈련 중 어려운 쌍의 가중치를 조정하여 보다 효과적인 모델 학습을 도모합니다. 우리 방법은 이미지와 텍스트 간의 유사성을 정량적으로 평가하고, 순위에서 1위와 동일한 ID를 공유하지 않는 쌍의 가중치를 높이는 방식을 채택합니다. 이는 모델이 더욱 다양한 예제를 통해 학습하도록 유도하여, 최종적으로 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안하는 부스팅 기법은 네 개의 보행자 데이터셋에서 성능 향상을 달성하였으며, 정량적인 실험 결과를 통해 그 효과성을 입증했습니다. 특히, 난이도가 높은 쌍에 동적으로 가중치를 부여함으로써 일반적인 쌍보다 훈련 과정에서 더 큰 영향을 끼쳤습니다. 이러한 접근은 TBPS 분야에서 모델의 강건성을 강화하는데 기여할 것으로 기대됩니다.



### An Exceptional Dataset For Rare Pancreatic Tumor Segmentation (https://arxiv.org/abs/2501.17555)
- **What's New**: 이번 논문에서는 췌장 신경내분비 종양(pNETs)을 위한 최초의 데이터셋을 소개합니다. 이 데이터셋은 469명의 환자 데이터를 포함하며, 잘 주석 처리된 CECT(대조강화 컴퓨터 단층 촬영) 이미지를 중심으로 구성되어 있습니다. 또한, UNet 기반 모델을 개선하기 위한 새로운 슬라이스별 가중치 손실 함수가 포함되어 있어 pNET 세분화 성능을 향상시킬 수 있습니다.

- **Technical Details**: 데이터셋은 3D CECT 스캔을 포함하고 있으며, 각 이미지는 512×512의 해상도를 가지며, 보통 150~270개의 횡단면을 커버합니다. 슬라이스 간격은 1mm에서 1.5mm로 설정되어 있으며, 전문가에 의해 주석 처리된 3D 종양 세분화 마스크와 함께 제공됩니다. 또한, 동맥 및 정맥 단계에서의 이미지가 포함된 다단계 데이터셋이 준비되어 있어 퀄리티가 보장됩니다.

- **Performance Highlights**: 우리는 여러 가지 U-Net 기반 의료 이미지 세분화 방법의 성능을 평가했습니다. 이 데이터셋은 pNETs와 같은 희귀 종양에 대한 연구의 기초 자료로 활용될 수 있으며, 향후 보다 정확한 진단 도구 개발에 기여할 것입니다. 이를 통해 pNETs의 조기 진단 및 치료 결과 개선이 기대됩니다.



### Action Recognition Using Temporal Shift Module and Ensemble Learning (https://arxiv.org/abs/2501.17550)
Comments:
          12 pages, MMVPR @ ICPR2024

- **What's New**: 이번 논문은 2024 ICPR의 Multi-Modal Action Recognition Challenge에서 우수 성적을 거둔 첫 번째 솔루션을 소개합니다. 다양한 데이터 소스를 사용하여 20개의 인간 행동을 인식하는 것을 목표로 하였으며, 제안된 방법론은 TSM(Temporal Shift Module)을 기반으로 하여 동영상 데이터의 시간적 동역학을 효율적으로 포착하도록 설계되었습니다. 우리는 transfer learning을 활용하여 미리 학습된 모델을 사용한 뒤, 도전 과제의 특정 데이터셋에 대한 세밀한 튜닝을 통해 20개 행동 클래스에 대한 성능을 최적화하였습니다.

- **Technical Details**: 이 연구에서 사용한 방법은 TSM을 중심으로 하고 있으며, ResNeSt-269와 ResNeXt101 모델을 활용하여 RGB 및 열 화상 데이터에 대한 예측을 수행했습니다. 또한, 각 모델의 출력을 조합하는 앙상블 기법을 적용하여 예측의 정확성을 높였습니다. 우리는 RGB와 열 IR 데이터만을 이용하였고, 깊이 이미지는 그 크기로 인해 사용하지 않기로 결정하였습니다.

- **Performance Highlights**: 테스트 세트에서 완벽한 Top-1 정확도를 달성하여 제안된 접근 방식의 효과성을 입증하였습니다. 특히, 열 IR 데이터를 활용한 방식이 실험 중 가장 높은 정확도를 보였고, 이는 복잡한 행동을 인식하는 데 있어 상당한 강점을 가지고 있음을 나타냅니다. 이러한 결과는 단일 모달리티에 집중함으로써 모델 아키텍처를 단순화하면서도 기준선을 초월하는 성과를 거두었음을 보여줍니다.



### Towards Training-Free Open-World Classification with 3D Generative Models (https://arxiv.org/abs/2501.17547)
- **What's New**: 이 논문은 3D 오픈 월드(classification) 분류를 위한 혁신적인 접근 방식을 제안합니다. 기존의 2D 모델을 기반으로 한 방법들이 3D 객체를 2D로 투사하는 과정에서 성능이 제한되는 반면, 이 연구에서는 3D 생성 모델을 활용하여 더욱 향상된 3D 객체 분류를 가능하게 합니다. 새로운 파이프라인은 학습이 필요 없으며, 개방형 카테고리 및 자세 불변성을 보장하여 효과적인 3D 오픈 월드 분류에 적합합니다.

- **Technical Details**: 제안된 시스템은 사전 훈련된 텍스트-투-3D 생성 모델을 통해 각 카테고리의 프로토타입 예시로서 앵커 샘플을 생성합니다. 이후에는 포인트 클라우드(polygon cloud) 표현 모델을 사용하여, 생성된 샘플로부터 특징을 추출하고, 이 특징들이 분류 성능에 긍정적인 상관관계를 가짐을 명확히 합니다. 특히, 이 연구는 다양한 표현 모델이 3D 객체에 대한 자세 변화에 미치는 영향을 탐구하고, 회전 불변성의 중요성을 논의합니다.

- **Performance Highlights**: 모델넷(ModelNet10)과 맥길(McGill) 데이터셋에서의 엄청난 실험 결과는 제안한 시스템이 기존 방법들에 비해 각각 32.0% 및 8.7%의 전반적인 정확도 향상을 보여주었음을 입증합니다. 이러한 결과는 3D 생성 모델이 오픈 월드(classification) 과제에 잘 적합함을 보여줍니다. 또한, 대형 언어 모델(LLMs)을 통해 생성된 카테고리 이름에 대한 상세한 텍스트 설명도 활용하여 제한된 텍스트 설명 능력을 향상시키는 시도를 포함하고 있습니다.



### 3DSES: an indoor Lidar point cloud segmentation dataset with real and pseudo-labels from a 3D mod (https://arxiv.org/abs/2501.17534)
- **What's New**: 새로운 데이터셋 3DSES(3D Segmentation of ESGT point clouds)는 427㎡의 공학 학교를 포괄하는 밀집된 TLS(colorized point clouds) 포인트 클라우드를 포함하고 있습니다. 이 데이터셋은 점 수준에서 주석이 달린 의미론적 레이블과 해당 건물의 전체 3D CAD 모델을 병행하여 포함하는 독특한 이중 주석 형식을 제공합니다. 이로 인해, 기존의 photogrammetry 기반 데이터셋의 한계를 극복할 수 있는 기회를 제공합니다.

- **Technical Details**: 3DSES는 다양한 의미론적 및 기하학적 복잡성을 가진 3가지 변형을 제공합니다. 우리는 기존의 3D CAD 모델을 활용하여 실내 포인트 클라우드의 자동 주석을 위한 model-to-cloud 알고리즘을 도입하였습니다. 이를 통해 95% 이상의 정확도로 포인트 클라우드에서 pseudo-labels를 생성할 수 있습니다.

- **Performance Highlights**: 3DSES에 대한 첫 번째 기초 실험은 BIM과 관련된 객체를 분할하는 데 있어 기존 모델들이 직면하는 어려움을 보여줍니다. 또한, pseudo-labels와 라이다(Lidar) 강도를 활용함으로써 분할 정확도를 향상시킬 수 있다는 것을 입증하였습니다. 제공되는 코드와 데이터는 오픈 소스로 공개될 예정입니다.



### Solving Inverse Problems using Diffusion with Fast Iterative Renoising (https://arxiv.org/abs/2501.17468)
- **What's New**: 이번 논문에서는 사전 훈련된 diffusion 모델을 활용하여 이미징 역 문제를 해결하는 새로운 접근 방식을 제안합니다. 기존 방법들은 재진행 과정에서의 조건부 점수 함수의 기울기 근사치가 부정확하여, 특히 초기에 성능이 좋지 않았습니다. 저자들은 새로운 방법인 "DDfire"를 제안하여 이미지의 재추정과 재노이즈를 여러 번 수행함으로써 이러한 문제를 개선합니다.

- **Technical Details**: DDfire 기법은 각 diffusion 단계에서 신중하게 설계된 컬러 노이즈를 추가하여, 사전 훈련된 diffusion 모델이 백색-가우시안 오류(white-Gaussian error)를 인식할 수 있도록 합니다. 이 방법론은 측정 데이터(y)와 함께 역 과정에서 작용하여 posterior distribution p(x0|y)를 샘플링하는데 중점을 둡니다. 기존의 두 가지 접근 방식에서는 조건부 점수의 근사를 제공하지만, 이러한 근사치들은 특히 초기 단계에서 부정확한 경향이 있었습니다.

- **Performance Highlights**: 저자들은 DDfire 방법이 선형 역 문제 및 위상 복구(phase retrieval)에서 20, 100, 1000개의 신경 함수 평가(neural function evaluations)로 효과적으로 작동함을 보였습니다. 이러한 성능 향상은 다수의 재추정 및 재노이즈 단계가 병행되면서 이루어졌으며, 결과적으로 역 문제의 해결에 있어 정확도를 높였습니다. 이 논문은 수치 실험을 통해 DDfire 방법의 우수성을 입증하고 있습니다.



### Towards Making Flowchart Images Machine Interpretab (https://arxiv.org/abs/2501.17441)
Comments:
          Published at: ICDAR 2023, Project Page: this https URL

- **What's New**: 이 논문은 플로우차트(flowchart) 이미지를 머신이 해석할 수 있는 실행 가능한 파이썬 코드로 변환하는 새로운 접근 방식을 제안합니다. 이를 위해, 최근 자연어를 코드로 생성하는 연구에서 영감을 받아 FloCo-T5라는 트랜스포머 기반의 프레임워크를 구축하여 플로우차트 이미지를 자동으로 변환합니다. 또한, 11,884개의 플로우차트 이미지와 해당하는 파이썬 코드로 구성된 FloCo 데이터셋을 소개하여, 향후 연구에 기여할 기반을 마련하였습니다.

- **Technical Details**: FloCo-T5는 플로우차트 이미지를 시퀀스 인코딩으로 변환하고, 이를 통해 프로그래밍 언어의 구조와 의미를 이해하도록 사전 훈련(pre-training)되었습니다. 이 모델은 특정한 작업을 위한 목표로 코드 샘플을 대규모로 증강하여 학습하였으며, 최종적으로 시퀀스-투-시퀀스(sequence-to-sequence) 생성 문제로 모델을 조정(fine-tuning)하여 사용합니다. 본 논문에서 실시한 실험들은 다양한 코드 생성 메트릭에서 FloCo-T5가 기존 기준 모델들보다 우수한 성능을 보임을 증명하였습니다.

- **Performance Highlights**: 실험 결과, FloCo-T5는 BLEU, CodeBLEU, 정확도 점수에서 각각 67.4, 75.7, 20.0를 기록하며, 기존 모델들보다 뛰어난 성능을 나타냈습니다. 이는 우리가 제안한 사전 훈련 목표와 데이터 증강 전략의 효과를 분명히 입증합니다. 또한, 본 연구에서 제안한 모델은 손으로 그린 플로우차트 이미지에도 적용 가능하다는 점에서, 다양한 응용 가능성을 보이고 있습니다.



### SIGN: A Statistically-Informed Gaze Network for Gaze Time Prediction (https://arxiv.org/abs/2501.17422)
Comments:
          4 pages, 2 figures

- **What's New**: 이 논문에서는 이미지에서 집합적 시선 시간을 예측하기 위한 첫 번째 버전의 SIGN(Statistically-Informed Gaze Network)을 제안합니다. 깊이 있는 학습 구현에서 CNN(Convolutional Neural Networks)과 Visual Transformers를 이용하여 기본적인 통계 모델을 개발하였습니다. 이 모델은 전체 시선 시간을 기반으로 이미지의 모든 영역에 대한 시선 패턴을 확률 맵으로 도출합니다.

- **Technical Details**: SIGN 모델은 AdGaze3500과 COCO-Search18 두 데이터셋에서 시선 지속 시간 예측을 위해 평가되었습니다. AdGaze3500은 집합적 시선 시간이 있는 광고 이미지 데이터셋이고, COCO-Search18은 탐색 중에 수집된 개별적 고정 패턴을 포함합니다. 모델의 성능은 최신 딥러닝 벤치마크보다 시선 지속 시간 예측에서 유의하게 개선되었습니다.

- **Performance Highlights**: SIGN은 COCO-Search18에서 경험적 고정 패턴과 일치하는 그럴듯한 시선 패턴을 제공할 수 있음을 보여주었습니다. 이 결과는 SIGN의 첫 번째 버전이 시선 시간 예측에 대한 가능성을 지니고 있으며, 추가 개발이 필요하다는 것을 제시합니다.



### General Scene Adaptation for Vision-and-Language Navigation (https://arxiv.org/abs/2501.17403)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 Vision-and-Language Navigation (VLN) 작업을 개선하기 위한 새로운 접근법인 GSA-VLN을 제안합니다. 기존의 VLN 작업들이 환경 기본 경험을 고려하지 않는 것에 반해, GSA-VLN은 특정 환경에서 내비게이션 지시를 실행하며 그에 따라 지속적으로 적응할 것을 요구합니다. 이를 통해, 에이전트들이 실제 환경에 맞춰 지속적으로 개선될 수 있는 가능성을 엿볼 수 있습니다.

- **Technical Details**: 주요 이점으로는 GSA-R2R 데이터셋을 통해 기존 VLN 데이터셋에 비해 환경 및 지시문 다양성을 획기적으로 확장한 점입니다. 또한, 대규모 Vision-Language Models (VLMs)와 Large Language Models (LLMs)을 활용한 세 단계의 지시 제작 프로세스를 개발하여, 각 환경에서 600개의 지시문을 다양하게 생성합니다. 이 연구는 에이전트의 성능을 ID 및 OOD 문맥 모두에서 평가할 수 있는 환경을 제공합니다.

- **Performance Highlights**: 실험 결과, GR-DUET라는 새로운 방법은 각 환경의 전반적인 위상 그래프를 지속적으로 업데이트하여 훈련 및 평가에서의 과거 정보를 보존합니다. 이러한 접근은 기본 DUET 모델에 비해 8%의 성공률 향상을 이끌어내며, 모든 GSA-R2R 분할에서 최신 결과를 생성하는 데 성공했습니다.



### Learning Free Token Reduction for Multi-Modal LLM (https://arxiv.org/abs/2501.17391)
- **What's New**: 본 논문에서는 비디오 기반 멀티모달 언어 모델(Multimodal Large Language Models, MLLMs)의 효율성을 높이기 위해 시각적 프롬프트를 압축하는 새로운 접근 방식을 제안합니다. 기존 모델 압축 기술은 주로 아키텍처 개선이나 시각적 토큰 수 줄이기에 집중했으나, 시각 데이터의 고유한 공간 및 시간적 특성을 고려하지 않았습니다. 반면, 제안된 방법은 시각적 토큰의 시간적 및 공간적 차원 모두에서 압축을 수행하여 성능을 유지하면서도 계산 비용을 절감할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 방법은 학습이 필요 없는 플러그 앤 플레이(plug-and-play) 압축 파이프라인으로, 대부분의 MLLM 프레임워크에 통합할 수 있습니다. 우리는 시간 차원에서 인접한 토큰을 병합하고 공간 차원에서 덜 유의미한 토큰을 제거하여 시각적 표현의 중복성과 희소성을 활용합니다. 이를 통해 모델의 추론 능력을 향상시키고 실험적으로 비디오-QA(Question Answering) 작업에서의 효율성을 개선했습니다.

- **Performance Highlights**: 실험 결과, 제안된 압축 방법은 비디오-LLMs의 성능을 유지하면서도 토큰 수를 효과적으로 줄일 수 있음을 보여주었습니다. 이로써 비디오 기반 멀티모달 태스크에서 significant한 개선을 나타내며, 전체적인 추론 속도와 효율이 크게 향상되었습니다. 이는 다양한 MLLM 아키텍처와의 호환성을 보장하며, 동종의 기술적 과제 해결에 기여할 것으로 기대됩니다.



### Assessing the Capability of YOLO- and Transformer-based Object Detectors for Real-time Weed Detection (https://arxiv.org/abs/2501.17387)
- **What's New**: 이번 연구에서는 농업 분야에서 사용되는 최근의 객체 탐지 모델들을 비교하여 제초제의 효율적인 적용을 가능하게 하는 기술을 소개합니다. 특히, 다양한 풀과 작물 종을 구별하는 데 필요한 실시간 잔여 효율성을 평가하기 위해 YOLOv8, YOLOv9, YOLOv10 및 RT-DETR 모델에 대한 성능을 분석하였습니다. 이 연구는 효과적인 제초관리와 환경적 지속 가능성을 위해 실시간에서 작물과 잡초를 구별하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서는 두 개의 데이터 세트를 사용하여 서로 다른 제초작물과 잡초를 학습하고 평가하였습니다. 첫 번째 데이터 세트는 식물 종을 개별적으로 학습하고, 두 번째 데이터 세트는 단엽잡초(monocotyledonous weeds), 쌍엽잡초(dicotyledonous weeds), 세 가지 선택된 작물을 구별합니다. YOLOv9s와 YOLOv9e 모델은 mAP50과 mAP50-95에서 높은 성과를 보였으며, RT-DETR 모델은 정밀도에서 특히 뛰어난 결과를 나타냈습니다.

- **Performance Highlights**: YOLOv9 모델은 우수한 recall 성능을 보여주었으며, RT-DETR 모델은 오탐지를 최소화하는 데 적합한 것으로 나타났습니다. 가장 작은 YOLO 모델 변형들은 특히 NVIDIA GeForce RTX 4090 GPU에서 7.58ms의 빠른 추론 시간을 달성하며 경쟁력 있는 정확도를 유지하여, 자원이 제한된 임베디드 컴퓨터 장치에 최적화된 가능성을 보여주었습니다.



### On the Coexistence and Ensembling of Watermarks (https://arxiv.org/abs/2501.17356)
- **What's New**: 이 논문에서는 심층 이미지 워터마킹(deep image watermarking) 방법의 공존을 최초로 연구했습니다. 연구 결과, 다양한 오픈 소스 워터마크가 이미지 품질(image quality)과 디코딩 강인성(decoding robustness)에 미치는 영향이 최소한으로 나타나면서 함께 공존할 수 있음을 발견했습니다. 이는 또한 서로 다른 워터마킹 기법을 조합하여 성능을 향상시킬 수 있는 가능성을 열어줍니다.

- **Technical Details**: 워터마킹은 이미지의 픽셀 값을 왜곡하여 비 가시적인 정보를 인코딩하는 방식으로 이루어지며, 이때 고려해야 할 요소들은 용량(capacity), 이미지 품질(image quality), 정확도(accuracy), 강인성(robustness)입니다. 본 연구에서는 여러 심층 학습 기반의 워터마킹이 서로 간섭 없이 공존할 수 있는지를 분석하였으며, 이는 이미지 품질과 디코딩 강인성을 단지 경미하게 감소시키면서 여전히 공존이 가능함을 보여주었습니다.

- **Performance Highlights**: 논문에서는 두 개의 서로 다른 워터마크를 사용하여 미디어의 데이터 용량을 증가시키고, 기존의 방법들이 재훈련 없이도 정확도, 강인성 및 이미지 품질 간의 트레이드오프를 조정할 수 있는 가능성을 제시합니다. 실험 결과, 이미지 워터마킹 기법이 기대 이상으로 높은 수준의 공존 가능성을 보여주었으며, 이는 효과적으로 워터마킹 모델 및 기존 기술 간의 성능 향상을 위한 기초가 될 수 있습니다.



### Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines (https://arxiv.org/abs/2501.17343)
- **What's New**: 이번 연구는 최신 3D 의료 분할 모델에 대해 실제 8비트 양자화(quantization) 프레임워크를 소개합니다. 특히, TensorRT 엔진을 통한 진정한 LOW-bit quantization을 적용하여 모델 크기 및 추론 시간을 획기적으로 줄이며, 실제 GPU 환경에서 효율성을 극대화합니다. 우리 연구는 의료 이미지 처리에서의 너비 제한을 고려하여, 실제 3D INT8 PTQ를 구현함으로써 가시적인 성과를 도출했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 TensorRT를 이용하여 레이블이 없는 식별 데이터셋을 통해 모델의 가짜 양자화(faux quantization)를 수행합니다. 두 번째 단계에서는 가짜 양자화된 모델을 실제 양자화된 TensorRT 엔진으로 변환하여, 낮은 정밀도의 계산을 가능하게 하는 하드웨어 최적화를 적용합니다. 이를 통해, 모델 크기와 추론 속도가 실제로 감소합니다.

- **Performance Highlights**: 우리가 개발한 PTQ 프레임워크는 기존의 가짜 양자화 방식과 비교하여 모델 크기가 2.42배에서 3.85배, 추론 지연시간이 2.05배에서 2.66배 감소했습니다. 이로 인해 대규모 3D 의료 이미지 작업에서도 높은 분할 정확도를 유지하면서 리소스 사용량을 줄이는 효과를 확인했습니다. 우리의 연구는 자원이 제한된 임상 환경에서도 높은 성능을 보장하는 효율적인 모델 배포의 길을 제시합니다.



### WASUP: Interpretable Classification with Weight-Input Alignment and Class-Discriminative SUPports Vectors (https://arxiv.org/abs/2501.17328)
- **What's New**: WASUP는 딥 러닝 모델의 해석 가능성과 정확성 간의 균형을 제공하는 고유하게 해석 가능한 신경망이다. WASUP는 훈련 이미지에서 클래스 대표성을 가진 지원 벡터를 추출하여 주요 특성을 포착하고, 불필요한 것들은 억제하는 방식으로 작동한다. 이 네트워크는 로컬(local) 및 글로벌(global) 설명을 제공하며, 이 설명이 모두 신뢰할 수 있는지를 보장하는 수학적 정당성도 제시하고 있다.

- **Technical Details**: WASUP는 이미지 분류 시 로컬 및 글로벌 설명을 제공하며, B-Cos 변환을 적용하여 입력 특성과 모델 가중치의 정렬을 가능하게 한다. 이 모델은 지원 벡터와 입력의 잠재 특성 벡터 간의 유사성 점수를 계산하고 집계하여 결정합니다. 또한, 기존의 복잡한 신경망 아키텍처 (예: ResNet, Transformers)를 사용할 수 있도록 하여 다양한 데이터 변동성을 캡처할 수 있는 능력을 갖추고 있다.

- **Performance Highlights**: WASUP는 Stanford Dogs에서의 미세한 분류, Pascal VOC에서의 다중 레이블 분류, 그리고 RSNA 데이터세트의 병리 감지 등 세 가지 작업에서 평가되었다. 결과적으로, WASUP는 최신 블랙박스 모델과 비교할 때 경쟁력 있는 정확도를 달성했으며, 이론적 분석을 통해 검증된 통찰력 있는 설명을 제공하여 모델의 결정 과정을 이해하는 데 기여한다.



### A Contrastive Teacher-Student Framework for Novelty Detection under Style Shifts (https://arxiv.org/abs/2501.17289)
Comments:
          The code repository is available at: this https URL

- **What's New**: 본 연구에서는 스타일 변화에 강인한 새로운 노벨티 탐지(Novelty Detection, ND) 방법을 제안합니다. 기존의 ND 방법들은 훈련 중 Out-of-distribution (OOD) 샘플이 없기 때문에 스타일 피처에 편향된 경향이 있어, 환경의 미세한 변화에 따라 성능 저하가 발생하는 문제를 지적합니다. 본 연구는 ID 세트와 유사하지만 서로 다른 핵심 피처를 갖춘 보조 OOD 세트를 만드는 방법을 논의합니다.

- **Technical Details**: 노벨티 탐지의 주 목적은 추론 중 ID 샘플과 OOD 샘플을 구별하는 것입니다. 이를 위해, 연구진은 Grad-CAM을 활용하여 ID 샘플의 핵심 피처를 식별하고, 해당 피처에 대해 가벼운 변형을 적용하여 최종 살리언시 맵을 생성합니다. 이후, 하드 변환을 이용해 높은 살리언시 값을 가진 ID 샘플 영역에 변형을 적용함으로써 스타일 변화에 강인함을 확보합니다.

- **Performance Highlights**: 제안한 방법은 여러 데이터셋에서 실험적으로 검증되었으며, 기존의 아키텍처와 비교해 최대 12.7%의 AUROC 성능 개선을 보여줍니다. 특히, 자체적으로 생성한 OOD 샘플을 활용한 지식 증류 방법을 통해, ND 작업에서의 우수성을 입증했습니다. 본 연구는 다양한 시나리오에서 높은 강인성을 실현하여 실제 응용에서도 유용할 것으로 기대됩니다.



### ViT-2SPN: Vision Transformer-based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification (https://arxiv.org/abs/2501.17260)
- **What's New**: 이번 연구에서는 ViT-2SPN이라는 새로운 프레임워크를 제안하여 Optical Coherence Tomography (OCT) 이미지 분석의 성능을 높이고자 하였습니다. ViT-2SPN은 Supervised Pretraining, Self-Supervised Pretraining (SSP), Supervised Fine-Tuning의 세 가지 단계의 워크플로우를 사용합니다. 이 방법은 대량의 무표시 데이터에서 피처 추출을 강화하고 진단 정확도를 향상시키는데 기여합니다.

- **Technical Details**: ViT-2SPN의 훈련 단계는 OCTMNIST 데이터셋을 기반으로 진행되며, 이는 97,477개의 비표시 이미지를 포함하고 있습니다. 본 연구에서는 Vision Transformer (ViT-Base) 백본을 사용하여 특징을 추출하고, 네거티브 코사인 유사도 손실을 적용하여 특성 표현을 정렬합니다. 50 에포크 동안 훈련을 진행하고, 최종적으로 10-겹 교차 검증을 통해 미세 조정을 수행합니다.

- **Performance Highlights**: ViT-2SPN은 평균 AUC 0.93, 정확도 0.77, 정밀도 0.81, 재현율 0.75, F1 점수 0.76을 달성하여 기존의 SSP 기반 방법들을 초월하는 성과를 보였습니다. 이는 특히 OCT 분석에서 데이터 효율성을 개선하고, 클래스 불균형 문제를 해결하는데 중요한 진전을 의미합니다.



### Separated Inter/Intra-Modal Fusion Prompts for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.17171)
Comments:
          AIAP 2025

- **What's New**: 이 논문에서는 Compositional Zero-Shot Learning (CZSL) 기법을 통해 의미의 미세한 차이와 상태 및 객체의 조합을 인식하는 새로운 접근방식을 제안합니다. 기존의 방법들이 프롬프트(prompt) 구성이나 프리트레인(pre-trained) 비전-언어 모델 튜닝에 집중하던 반면, 이 방법은 그러한 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 다양한 Prompt Learning 기법과 Inter/Intra-Modality Fusion Synthesizer를 활용하여 장면 이해(scene understanding)에서의 속성 인식(attribute recognition) 성능을 향상시키는 데 초점을 두고 있습니다. 이를 통해 미세한 의미 차이와 여러 객체를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: 이 연구는 CZSL 기술을 통해 보고된 성능을 개선하며, 특히 다양한 의미론적(stylistic) 차이를 명확히 인식함으로써 실험에서 유의미한 결과를 도출했습니다. 이는 구성의 복잡성을 포함한 다양한 상황에서도 강력한 성능을 보이는 것을 의미합니다.



### Aggregation Schemes for Single-Vector WSI Representation Learning in Digital Pathology (https://arxiv.org/abs/2501.17822)
- **What's New**: 이 논문에서는 Whole Slide Image(WSI) 검색 성능을 평가하기 위해 여러 최근 개발된 집계 기법을 비교합니다. 특히, WSI의 고해상도 특성 때문에 대규모 패치 라벨링 및 특정 집합 표현 학습 기술을 탐구합니다. 기존의 패치 임베딩 세트를 단일 임베딩으로 변환하는 다양한 방법을 제안하며 성능을 비교하고 있습니다.

- **Technical Details**: WSI는 일반적으로 수많은 작은 "패치"나 타일로 분할되며, 이를 통해 각 WSI에 대해 패치 임베딩 세트를 생성합니다. 이 과정에서 다양한 집계 알고리즘이 사용되며, 각 알고리즘은 패치 임베딩 세트에서 단일 벡터로 깊은 특징을 추출합니다. Fisher Vector, Deep Sets, Memory Networks, Focal Attention 등 다양한 접근 방식을 사용하여 WSI의 임베딩 단일화 및 표준화를 목표로 하고 있습니다.

- **Performance Highlights**: 연구에서는 TCGA에서 제공하는 방광, 유방, 신장, 대장과 같은 여러 데이터셋을 통해 k-NN 검색 방식을 적용하여 성능을 평가했습니다. 또한, 이들 방법의 검색 성능을 실제 WSI 검색 시스템에 적용됨에 따라 메디안 최소 거리 기반 접근 방식과 비교하여 더 나은 결과를 도출된 것으로 나타났습니다. 최종적으로 각 방식의 단일 벡터 생성 성능을 분석하고 결과를 상세히 논의합니다.



### Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling (https://arxiv.org/abs/2501.17811)
Comments:
          Research paper. arXiv admin note: text overlap with arXiv:2410.13848

- **What's New**: 이번 연구에서는 Janus의 발전된 버전인 Janus-Pro를 소개합니다. Janus-Pro는 최적화된 훈련 전략, 확장된 훈련 데이터, 더 큰 모델 크기를 포함하여 세 가지 주요 개선 사항을 통합하였습니다. 이러한 개선을 통해 Janus-Pro는 멀티모달 이해 및 텍스트-이미지 지시 수행 능력에서 상당한 진전을 이루었으며, 텍스트-이미지 생성의 안정성 또한 향상시켰습니다.

- **Technical Details**: Janus-Pro의 아키텍처는 Janus와 동일한 원칙을 따르며 멀티모달 이해와 생성 간의 비주얼 인코딩을 분리하는 것을 핵심 디자인 원칙으로 삼고 있습니다. 멀티모달 이해를 위해 SigLIP 인코더를 사용하여 고차원 시맨틱 특징을 추출하고, 이를 LLM의 입력 공간으로 매핑합니다. 그에 반해 비주얼 생성 작업에는 VQ 토크나이저를 활용하여 이미지를 분산된 ID로 변환합니다.

- **Performance Highlights**: Janus-Pro는 다양한 벤치마크에서 평가한 결과, 멀티모달 이해 능력이 우수하고 텍스트-이미지 지시 수행 성능이 크게 향상된 것으로 나타났습니다. 특히, Janus-Pro-7B 모델은 MMBench에서 79.2점을 기록하며, 기존의 최첨단 멀티모달 모델들을 초월했습니다. 또한 GenEval 리더보드에서도 0.80점을 기록하여 Janus와 DALL-E 3 등 다른 모델들을 의가 발휘했습니다.



### Glioma Multimodal MRI Analysis System for Tumor Layered Diagnosis via Multi-task Semi-supervised Learning (https://arxiv.org/abs/2501.17758)
Comments:
          23 pages, 13 figures

- **What's New**: 본 연구에서는 다중 모드 MRI(Multimodal MRI)를 활용한 신경아교종 진단을 위한 GMMAS(Glioma Multimodal MRI Analysis System)를 제안합니다. GMMAS는 심층 학습 네트워크를 기반으로 하여 여러 이벤트 간의 상호 의존성을 권장하는 멀티태스킹 학습 아키텍처(architecture)를 통해 동시에 처리합니다. 이 시스템은 종양 영역 세분화와 신경아교종의 조직학적 아형, IDH 변이형, 1p/19q 크로모좀의 질환 상태를 동기화하여 출력하는 기능을 갖추고 있습니다.

- **Technical Details**: GMMAS는 두 단계의 반자율 학습 방법을 사용하여 모델 성능을 향상시키며, 라벨이 있는 MRI 샘플과 라벨이 없는 샘플을 모두 활용합니다. 또한, 지식 자기 증류(knowledge self-distillation) 및 대비 학습(contrastive learning)을 기반으로 하는 적응 모듈을 통해 모달리티 결핍 상황에서도 견고한 성능을 보여줍니다. GMMAS는 CNN(Convolutional Neural Networks)과 변환기(transformer) 모델을 통합하여 다중 모드 MRI 데이터를 효과적으로 처리합니다.

- **Performance Highlights**: GMMAS는 단일 작업 분석 모델에 비해 종양 층 진단 작업에서 정확도가 향상되었습니다. 또한, 자동화된 분석 워크플로우를 통해 신경외과 의사와 방사선과 의사의 업무 부담을 줄이고, 진단 프로세스의 효율성을 높이는 데 기여할 수 있습니다. 향후, GMMAS-GPT를 통해 개인화된 예후 평가와 제안을 생성하는 사용자 친화적인 플랫폼이 제공됩니다.



### PulmoFusion: Advancing Pulmonary Health with Efficient Multi-Modal Fusion (https://arxiv.org/abs/2501.17699)
- **What's New**: 이번 연구에서는 기존의 원격 폐 기능 측정 기술의 한계를 극복하기 위해 RGB와 열화상 비디오 데이터 및 환자 메타데이터를 통합한 새로운 비침습적 접근법을 제안합니다. Spiking Neural Networks (SNNs)를 활용하여 Peak Expiratory Flow (PEF)와 Forced Expiratory Volume (FEV1) 및 Forced Vital Capacity (FVC) 분류를 수행하며, 경량 CNN을 통해 SNN의 회귀 작업 한계를 극복합니다. 또한, 멀티헤드 어텐션 레이어와 K-Fold 검증 방식, 앙상블 학습을 활용하여 모델의 강건성을 높입니다.

- **Technical Details**: 모델은 비디오 기반 학습 접근 방식으로 전환하여 전체 호흡 사이클 측정을 수행하는 복잡성 문제를 해결합니다. SNN은 Leaky Integrate-and-Fire (LIF) 뉴런 구조를 사용하여 생물학적 뉴런을 모델링하고, Spike 타이밍에 기초하여 데이터를 효율적으로 처리합니다. PulmoFusion 모델은 비디오 시퀀스를 처리하여 시공간 패턴을 식별하며, 메타데이터는 스파이크 트레인을 통해 통합됩니다.

- **Performance Highlights**: 열화상 데이터를 사용하여 SNN 모델은 호흡 주기를 기준으로 92%의 정확도와 환자별로 99.5%의 정확도를 달성했습니다. PEF 회귀 모델은 열화상에서 0.11의 상대 RMSE와 RGB에서 0.26의 RMSE를 보였으며, FEV1/FVC 예측에 대한 평균 절대 오차(MAE)는 4.52%로, 최신 성능을 확인했습니다.



### In-Context Meta LoRA Generation (https://arxiv.org/abs/2501.17635)
- **What's New**: 본 연구에서는 다중 작업 환경에서도 효율성을 유지하면서, 대형 언어 모델(LLMs)의 작업 특화(customization)를 위한 In-Context Meta LoRA (ICM-LoRA)라는 혁신적인 접근 방식을 제안합니다. 기존의 Low-rank Adaptation (LoRA) 모델이 개별 작업에 대해 따로 훈련되는 비효율성을 해결하기 위해, Conditional Variational Autoencoder (CVAE)를 활용하여 task-aware LoRA weights를 생성합니다. ICM-LoRA는 저장 용량을 감소시키면서도 다양한 작업에 대한 정확한 LoRA 매개변수를 생성할 수 있습니다.

- **Technical Details**: ICM-LoRA는 task vectors를 사용하여 context modeling을 수행하며, 메타 학습(meta-learning)을 통해 작업 간의 관계를 포착합니다. 이를 통해 LoRA 매개변수 생성을 단일 생성자(generator)로 동시에 처리할 수 있는 그리드(grid) 생성이 가능해집니다. 또한 ICM-LoRA는 데이터를 추가로 필요로 하지 않고, 단 283MB의 저장 용량만 차지하여 기존 LoRA에 비해 1%의 저장소 사용량을 기록합니다.

- **Performance Highlights**: ICM-LoRA는 다양한 모델에서 텍스트 및 비주얼 작업을 평가하였으며, CVAE가 여러 작업에 대해 LoRA 매개변수를 성공적으로 생성할 수 있음을 보여줍니다. 기존 방법들과 비교해, 생성된 LoRA 매개변수는 정확도 손실이 적으며, 원본 데이터 세트와 LoRA 가중치에 비해 저장 용량을 크게 줄일 수 있습니다. 결과적으로 본 연구의 접근법은 여러 작업에 대한 LoRA 파라미터 생성을 보다 효율적이고 정확하게 수행할 수 있음을 입증합니다.



### Federated Learning With Individualized Privacy Through Client Sampling (https://arxiv.org/abs/2501.17634)
- **What's New**: 이 논문에서는 개인화된 프라이버시 요구를 반영한 Individualized Differential Privacy (IDP) 방법을 Federated Learning (FL)에서 구현하는 방법을 제안합니다. 모든 사용자에게 동일한 개인 데이터 보호 수준을 적용하는 대신, 사용자가 자신의 프라이버시 설정을 선택할 수 있도록 합니다. 이로 인해 데이터의 유틸리티와 프라이버시 간의 균형을 맞출 수 있습니다.

- **Technical Details**: IDP-FedAvg라는 수정된 알고리즘을 통해 클라이언트의 개인 프라이버시 예산을 기반으로 각 클라이언트에 맞춤화된 샘플링 비율을 계산합니다. 기존의 SAMPLE 알고리즘을 FL 환경으로 확장하여 클라이언트의 참여를 그들의 프라이버시 요구 사항에 따라 결정하도록 합니다. 논문에서는 또한 비독립적이고 동일하게 분포되지 않은 데이터에 대한 도전 과제를 다룹니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 Uniform DP 기준선에 비해 명확한 개선을 보여주며 프라이버시와 유틸리티 간의 무역 균형을 줄이는 데 성공했습니다. 또한, SCALE 방법보다도 성능이 뛰어난 결과를 얻었습니다. 하지만 비복잡한 작업(예: 비독립적 비균일 분포 데이터)에 있어 여전히 도전과제가 존재합니다.



### Dual Invariance Self-training for Reliable Semi-supervised Surgical Phase Recognition (https://arxiv.org/abs/2501.17628)
- **What's New**: 이번 논문에서는 수술 단계 인식을 위한 새로운 반지도 학습(Semi-supervised Learning, SSL) 프레임워크인 Dual Invariance Self-Training (DIST)를 제안합니다. 이 프레임워크는 시간적 불변성(Temporal invariance)과 변환 불변성(Transformation invariance)을 적용하여 신뢰할 수 있는 의사 라벨(pseudo-label) 선택 메커니즘을 통합합니다. 이러한 접근 방식은 잡음이 포함된 의사 라벨의 위험을 완화하고, 데이터 분포에 대한 정확한 의사 결정을 이끌어내어 미지의 데이터에 대한 일반화를 개선합니다.

- **Technical Details**: DIST는 두 단계로 운영되며, 첫 번째 단계에서는 라벨이 부착된 데이터로부터 교사 모델이 학습하여 의사 라벨을 생성합니다. 이후, 제안된 신뢰성 추정 방법을 사용하여 상위 50%의 신뢰할 수 있는 의사 라벨을 선택합니다. 두 번째 단계에서는 첫 번째 단계에서 생성된 의사 라벨을 적용하여 학생 모델을 훈련시키고, 이 학생 모델이 새로운 교사로 변환되어 더욱 정교한 의사 라벨을 생성합니다.

- **Performance Highlights**: Cataract와 Cholec80 데이터셋에서 실시한 평가 결과, 제안된 방법은 기존의 최첨단 SSL 접근법을 능가하는 성능을 보였습니다. 다양한 네트워크 아키텍처에 걸쳐 감독 학습(Supervised Learning) 및 SSL 기준을 지속적으로 초과하는 성능이 관찰되었습니다. 이는 제안된 DIST 프레임워크가 수술 비디오 분석에 있어 매우 효과적임을 증명합니다.



### Watch Your STEPP: Semantic Traversability Estimation using Pose Projected Features (https://arxiv.org/abs/2501.17594)
Comments:
          7 pages, 7 figures

- **What's New**: 이번 연구에서는 STEPP라는 새로운 로봇 비종속적(agnostic) 지형 추정 모델을 제안합니다. 이 모델은 RGB 이미지만을 활용하여 복잡한 구조와 비구조적 지형에서 성공적으로 탐색할 수 있도록 돕습니다. 기존의 접근 방식과 달리, STEPP는 인간의 보행 데이터를 학습하여 주변 지형의 탐색 비용을 효과적으로 추정합니다. 이를 통해 로봇이 안전하게 목표 지점으로 내비게이션할 수 있도록 지원합니다.

- **Technical Details**: STEPP는 DINOv2와 같은 비전 트랜스포머 모델을 활용하여 픽셀 수준의 특징 임베딩을 생성하고, 이를 통해 MLP 아키텍처에서 매우 세밀한 지형 평가를 수행합니다. 또한, 이 모델은 오프라인 긍정 및 비정형 데이터를 사용하여 훈련되어, 복잡한 환경에 대한 적응력을 높이고 지형의 조사 비용을 최소화합니다. 연구팀은 ANYmal 사족 로봇을 활용하여 실내 및 실외 환경에서 제안된 접근 방식을 실증하였습니다.

- **Performance Highlights**: STEPP는 기존의 방법들과 비교했을 때, 비구조적 접근 방식에서도 보다 가시적인 지형 평가를 가능하게 합니다. 실제 실험을 통해 로봇이 도전적인 지형을 효과적으로 탐색할 수 있는 능력을 보여주었습니다. 이 모델의 오픈 소스 코드는 커뮤니티가 접근할 수 있도록 제공되며, 나중에 더 다양한 시뮬레이션 데이터와 함께 활용될 수 있도록 합니다.



### Trustworthy image-to-image translation: evaluating uncertainty calibration in unpaired training scenarios (https://arxiv.org/abs/2501.17570)
- **What's New**: 이번 연구에서는 유방암 스크리닝의 자동화를 위한 두 가지 프레임워크/아키텍처인 GAN 기반의 cycleGAN과 최근에 개발된 확산 기반의 SynDiff를 평가합니다. 이를 통해 기존의 패턴 인식 기술에 대한 신뢰성을 높이고, 데이터의 일반화 능력을 향상시키는 방안을 모색합니다. 특히, 의료 영상의 경우 일반적으로 사용 가능한 데이터가 부족하다는 문제를 극복하고자 합니다.

- **Technical Details**: 이 논문은 비짝맞춤( unpaired ) 훈련 데이터 환경에서 이미지 전환 모델의 신뢰성을 높이기 위한 불확실성 정량화(uncertainty quantification) 사용을 제안합니다. 여기에서 cycleGAN과 SynDiff는 서로 다른 오픈 액세스 유방촬영 데이터 세트와 비의료 이미지 데이터 세트에서 훈련된 이미지 패치를 기반으로 성능 평가를 수행합니다. 또한, Training dataset 확장을 통한 일반화 능력 향상을 다루고 있으며, 변환된 기존 데이터의 사용을 통해 새로운 예제를 생성하는 데이터 증강(data augmentation) 기술을 논의합니다.

- **Performance Highlights**: 실험 결과, 두 모델 모두 검사에서 깊은 신경망이 인간 방사선의 성능을 초과하거나 일치할 수 있는 가능성을 보여주었습니다. 특히, 데이터 증강을 통해 얻은 다양한 이미지 표현이 모델의 일반화 성능을 개선하는 데 기여했습니다. 새로운 방법론의 적용으로, 진단 정확성과 해석 가능성을 향상시키는 방향으로 이어질 것으로 기대됩니다.



### Influence of field of view in visual prostheses design: Analysis with a VR system (https://arxiv.org/abs/2501.17322)
- **What's New**: 이번 연구에서는 시각 보철 (visual prostheses)의 공간 해상도 (spatial resolution)와 시야 (field of view)가 인지 정확도 및 반응 시간에 미치는 영향을 평가합니다. 기존의 저해상도 시각 보철의 한계를 극복하기 위해 가상 현실 환경에서 새로운 시뮬레이션 시스템을 사용하여 실험을 진행했습니다. 24명의 참가자를 통해 실내 장면에서의 물체 탐지 및 인식 성능을 측정하였습니다.

- **Technical Details**: 연구에서는 200개 또는 500개의 포스펜 (phosphene)을 사용하여 시각 보철 시뮬레이션을 진행했습니다. 포스펜은 이미지 크기와 시각적 반경을 기준으로 계산된 위치에 따라 생성되며, 이를 통해 그래픽적으로 구현된 다양한 패턴의 포스펜을 사용하여 실험했습니다. 포스펜의 강도는 각 픽셀의 밝기와 비례하며 가우시안 (Gaussian) 프로파일을 따릅니다.

- **Performance Highlights**: 실험 결과, 시야가 넓어질수록 인지 정확도와 반응 시간이 감소하는 것으로 나타났습니다. 또한, 각도 해상도 (angular resolution)와 성능 간의 연관성이 있으나, 2.3 포스펜 (phosphenes) 미만의 해상도에서도 점차 감소 효과가 나타났습니다. 따라서 시각 보철 설계 시 포스펜을 좁은 영역에 집중시키고 각도 해상도를 극대화하는 것이 중요함을 시사합니다.



### Advancing the Biological Plausibility and Efficacy of Hebbian Convolutional Neural Networks (https://arxiv.org/abs/2501.17266)
Comments:
          38 pages, 14 figures

- **What's New**: 이번 논문은 이미지를 처리하기 위한 CNN(Convolutional Neural Networks)과 헤비안 학습(Hebbian learning) 통합의 발전을 다루고 있습니다. 저자들은 생물학적인 타당성을 지킨 효율적인 구조를 설계하기 위해 다양한 아키텍처를 체계적으로 탐구했으며, Hard Winning-Takes-All(WTA) 경쟁 및 BCM(Bienenstock-Cooper-Munro) 학습 규칙을 통합하여 대표 표현 능력을 확장했습니다. 이러한 최적 아키텍처는 CIFAR-10 데이터셋에서 76%의 분류 정확도를 달성하여 기존의 백프파게이션(Backpropagation) 변형과 경쟁할 수 있는 성과를 보여주었습니다.

- **Technical Details**: 헤비안 학습은 지역적으로 비지도 학습을 통해 기능 표현(feature representation)을 형성하는 생물학적으로 영감을 받은 학습 알고리즘입니다. 본 연구에서는 Hard-WTA와 BCM 학습 규칙을 통해 깊은 CNN 아키텍처에 헤비안 학습을 통합하여 비지도 기능 추출을 위한 컴퓨팅 효율성을 최적화했습니다. 또한, 새로운 공간적 측면 억제 및 시간적 경쟁 메커니즘을 도입하여 생물학적 현실성을 더욱 향상시켰습니다.

- **Performance Highlights**: 새로운 SoftHebb 3-CNN 레이어 아키텍처를 통해 CIFAR-10에서 76%의 정확도를 달성하였으며, 이는 동일한 네트워크 깊이에서 기존 Hard-WTA 성능인 64.6%에 비해 11.4% 향상된 결과입니다. 실험 결과는 헤비안 학습의 기능 추출이 기존의 백프파게이션 방법과 유사한 성과를 나타낸다는 것을 보여주었으며, 이는 기계 학습의 생물학적 타당성과 효율성을 개선하는 중요한 발걸음으로 평가됩니다.



New uploads on arXiv(cs.AI)

### Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling (https://arxiv.org/abs/2501.17811)
Comments:
          Research paper. arXiv admin note: text overlap with arXiv:2410.13848

- **What's New**: 이번 연구에서는 Janus의 발전된 버전인 Janus-Pro를 소개합니다. Janus-Pro는 최적화된 훈련 전략, 확장된 훈련 데이터, 더 큰 모델 크기를 포함하여 세 가지 주요 개선 사항을 통합하였습니다. 이러한 개선을 통해 Janus-Pro는 멀티모달 이해 및 텍스트-이미지 지시 수행 능력에서 상당한 진전을 이루었으며, 텍스트-이미지 생성의 안정성 또한 향상시켰습니다.

- **Technical Details**: Janus-Pro의 아키텍처는 Janus와 동일한 원칙을 따르며 멀티모달 이해와 생성 간의 비주얼 인코딩을 분리하는 것을 핵심 디자인 원칙으로 삼고 있습니다. 멀티모달 이해를 위해 SigLIP 인코더를 사용하여 고차원 시맨틱 특징을 추출하고, 이를 LLM의 입력 공간으로 매핑합니다. 그에 반해 비주얼 생성 작업에는 VQ 토크나이저를 활용하여 이미지를 분산된 ID로 변환합니다.

- **Performance Highlights**: Janus-Pro는 다양한 벤치마크에서 평가한 결과, 멀티모달 이해 능력이 우수하고 텍스트-이미지 지시 수행 성능이 크게 향상된 것으로 나타났습니다. 특히, Janus-Pro-7B 모델은 MMBench에서 79.2점을 기록하며, 기존의 최첨단 멀티모달 모델들을 초월했습니다. 또한 GenEval 리더보드에서도 0.80점을 기록하여 Janus와 DALL-E 3 등 다른 모델들을 의가 발휘했습니다.



### Using Code Generation to Solve Open Instances of Combinatorial Design Problems (https://arxiv.org/abs/2501.17725)
- **What's New**: 이 논문에서는 조합 설계(combinatorial designs) 문제의 기존 해결 방법을 자동화하는 CPro1 프로토콜을 소개합니다. 이 프로토콜은 Large Language Models (LLMs)를 활용하여 다양한 후보 방법으로 코드 생성을 가능하게 하며, 특히 Packing Array의 경우 N=21로 구성이 가능하다는 기존의 미해결 질문을 해결합니다. 또한, 다양한 조합 설계의 존재 문제를 해결하기 위한 기초 작업을 진행합니다.

- **Technical Details**: CPro1 프로토콜은 LLM을 사용하여 조합 관련 설계를 정의하고 이를 검증하는 기능을 포함합니다. 각 조합 설계 유형에 대해 전체 인스턴스 목록을 검토하고, 특정 설계가 유효한지 여부를 확인하는 방법도 구축합니다. LLM은 코드 생성에 있어 다양한 탐색 알고리즘(예: simulated annealing, genetic algorithms)을 실험하고, 자동화된 하이퍼파라미터 조정(hyperparameter tuning) 과정을 통해 성능 최적화도 수행합니다.

- **Performance Highlights**: 16가지 조합 설계 유형에 대한 테스트 결과, CPro1은 6개의 미해결 문제를 성공적으로 해결하였습니다. 이 결과로, 대칭 및 스큐 가중 행렬(Symmetric and Skew Weighing Matrices)과 같은 복잡한 조합 설계를 성공적으로 생성하는 성과를 거두었습니다. 또한, 논문에서는 이 프로토콜이 효율적으로 조합 설계 문제를 해결할 수 있음을 입증하고 있습니다.



### Inferring Implicit Goals Across Differing Task Models (https://arxiv.org/abs/2501.17704)
- **What's New**: 이 논문은 사용자 목표와 암묵적인 요구 사항 간의 불일치를 해결하기 위해 Markov Decision Process (MDP) 프레임워크 내에서 암묵적 하위 목표(implicit subgoal)를 공식화하는 새로운 접근 방식을 제안합니다. 또한, 사용자의 기본 목표를 성취하기 위한 최소 쿼리 수를 생성하는 쿼리 전략을 도입합니다. 연구 결과는 다양한 작업에서 명시되지 않은 목표를 추론하고 달성하는 데 효과적인 방법임을 보여줍니다.

- **Technical Details**: 이 연구에서는 무한 지평선 할인 Markov Decision Processes (MDP) 문제를 다루며, 목표 달성을 특정 목표 상태를 방문하는 것으로 정의합니다. 알고리즘은 훈련된 모델을 기반으로 하며, 사용자의 세계 모델과 실행하는 에이전트의 모델을 비교 분석하여, 병목 상태(bottleneck states)를 기반으로 잠재적인 암묵적 하위 목표를 도출합니다. 이 논문은 쿼리 기법을 통해 미리 언급되지 않은 사용자의 하위 목표를 이해하는 데 필요한 최소 정보를 수집하는 방법도 제시합니다.

- **Performance Highlights**: 제안된 접근 방식은 다양한 시뮬레이션된 시나리오에서 암묵적 하위 목표를 추론하고 달성하는 능력을 평가하며, 기존의 목표 지향 강화 학습(goal-conditioned reinforcement learning) 방법보다 개선된 성능을 보입니다. 실험 결과는 에이전트가 사용자의 잠재적인 목표를 효과적으로 이해하고 이를 토대로 행동할 수 있도록 지원한다는 것을 입증하였습니다. 이 연구는 에이전트와 사용자 간의 목표 불일치를 줄일 수 있는 중요한 기여를 합니다.



### Solving Urban Network Security Games: Learning Platform, Benchmark, and Challenge for AI Research (https://arxiv.org/abs/2501.17559)
- **What's New**: 이 논문은 다중 플레이어 게임을 해결하기 위한 새로운 플랫폼, GraphChase,를 제안합니다. 이 플랫폼은 Urban Network Security Games (UNSGs)를 모델링하며, 현실 세계 시나리오에서의 범죄자와 경찰 간의 상호작용을 다룹니다. UNSG는 경찰이 범죄자를 제압하기 위해 자원을 할당하는 문제를 해결하는데 매우 중요한 역할을 합니다.

- **Technical Details**: GraphChase는 다양한 UNSG 변수를 모델링할 수 있는 유연한 게임 환경을 제공합니다. 이 플랫폼은 그래프 G=(V,E) 형태로 도시 도로를 나타내며, 정점과 간선으로 구성됩니다. UNSG의 전략 공간은 NP-hard 문제이며, 실시간 정보의 유무나 경찰 간의 통신 가능 여부에 따라 그 복잡성이 달라집니다.

- **Performance Highlights**: 우리는 GraphChase 플랫폼에서 다양한 알고리즘의 성능을 평가하기 위한 실험을 실시했습니다. 실험 결과, 이전 알고리즘이 일정 수준의 성능을 달성했지만, 여전히 실제 환경에서 성능과 확장성 문제를 겪고 있다는 것을 보여주었습니다. 이상의 결과는 UNSGs 문제 해결을 위한 효율적이고 효과적인 알고리즘 개발에 더 많은 노력이 필요함을 시사합니다.



### Reflections on "Can AI Understand Our Universe?" (https://arxiv.org/abs/2501.17507)
Comments:
          Invited talk at the 17th Marcel Grossmann Meeting, associated with arXiv:2404.10019, to be published in the International Journal of Modern Physics D

- **What's New**: 이 논문은 AI의 이해의 개념을 철학적 및 기술적 관점에서 간략히 논의하며, 본질적으로 AI가 이해를 형성할 가능성을 탐구합니다. 특히 Transformer, chain-of-thought reasoning, 및 multimodal processing 기술을 주목하며 이들 기술이 미래의 AI 발전에서 중요한 역할을 할 것으로 예상합니다.

- **Technical Details**: 연구에서는 한 개의 대형 언어 모델인 GPT를 조정하여 다양한 천체 데이터와 작업을 동시에 처리할 수 있음을 보여주었습니다. 예를 들어, SDSS 스펙트럼 데이터를 사용한 분류 실험에서 82%의 정확도를 기록하였으며, 퀘이사 스펙트럼 데이터의 적색 편이 추정에서 90.66%의 정확도로 성능을 입증했습니다.

- **Performance Highlights**: AI는 고차원 데이터를 보다 효율적으로 처리할 수 있는 강점을 가지고 있으며, 예를 들어 블랙홀 파라미터 추론에 있어 100%의 정확도를 달성했습니다. 이러한 성과들은 AI가 다분야 데이터를 처리하는 데 있어 중요한 도구가 될 것임을 시사합니다.



### SemML: Enhancing Automata-Theoretic LTL Synthesis with Machine Learning (https://arxiv.org/abs/2501.17496)
- **What's New**: 이번 논문에서는 LTL(Linear Temporal Logic) 에 기반한 반응형 시스템(synthetic a reactive system) 생성 문제를 다루며, SemML이라는 도구를 소개합니다. SemML은 SYNTCOMP에서 LTL 실현 가능성(LTL realizability) 트랙에서 우승을 차지했으며, 이전의 Strix 도구에 비해 중요한 발전을 나타냅니다. 이 도구는 최근 LTL-automata 변환을 통해 추가적인 논리적 정보인 Semantic labelling을 활용하고, 이를 통해 생성한 패리티 게임(parity game)의 탐색을 머신 러닝 기술로 지원합니다.

- **Technical Details**: SemML은 자동기(automata) 이론적 접근 방식을 기반으로 하며, 두 가지 주요 기능을 갖추고 있습니다. 첫 번째는 Semantic labelling으로, LTL-to-automata 변환을 통해 얻어진 논리적 정보를 사용하여 패리티 게임을 장식(decorating)합니다. 두 번째는 머신 러닝 접근 방식을 통해 이 정보를 탐색 오라클(guidance oracle)로 변환하여 실시간 탐색을 지원합니다. 이러한 구조는 SemML을 기존의 제안에서 나타난 빈틈을 메우는 효율적인 구현으로 이끌었습니다.

- **Performance Highlights**: SemML은 SYNTCOMP 전체 데이터 세트 및 합성 데이터 세트(synthetic data set)에서 평가되었고, Strix와 비교 분석하였습니다. SemML은 SYNTCOMP에서 더 많은 인스턴스를 해결할 수 있었으며, 특히 대규모 인스턴스에서 빠른 속도를 기록하였습니다. 이러한 결과는 머신 러닝 보조 접근 방식이 LTL 합성에서 최첨단 도구들을 처음으로 초월할 수 있음을 보여줍니다.



### Certifying Pareto-Optimality in Multi-Objective Maximum Satisfiability (https://arxiv.org/abs/2501.17493)
- **What's New**: 이 논문에서는 다목적 최대 만족도(MaxSAT) 최적화 기법을 위한 VeriPB 증명 포맷 기반의 증명 로그(proof logging)를 처음으로 가능하게 하였습니다. 기존의 VeriPB 포맷은 다목적 문제에 대한 직접적인 지원을 제공하지 않지만, 이 연구에서는 VeriPB의 전순서(preorders)를 활용하여 Pareto-optimality 하에 비지배 집합(non-dominated set)의 각 요소에 대한 대표 솔루션을 제공하는 알고리즘에 대하여 인증서를 생성하는 방법을 상세히 설명합니다.

- **Technical Details**: VeriPB 포맷은 0-1 선형 부등식(pseudo-Boolean constraints)을 기반으로 하며, 단일 목적 최적화 문제에서 목표 값을 추론하기 위한 직접적인 지원을 제공합니다. 이 논문에서는 VeriPB 포맷을 활용하여 다목적 MaxSAT(MO-MaxSAT) 알고리즘을 위한 증명 로그를 구현하는 과정을 보여주며, 이와 같은 방식으로 Solvers의 신뢰성을 높이고, MO-MaxSAT의 특정 변형 알고리즘에 대한 효율성을 제고합니다. 특히, MO-MaxSAT를 해결하기 위한 Scuttle Solver에 VeriPB 증명 로그를 추가하여 평균적으로 14%에서 29%까지의 오버헤드로 확장 가능함을 입증하였습니다.

- **Performance Highlights**: 이 연구 결과는 다목적 MaxSAT solver에서 증명 로그 구현과 관련된 성능 개선을 보여줍니다. VeriPB 기반의 증명 로그가 적용된 다양한 MO-MaxSAT 기법을 통해, 신뢰할 수 있는 증명 결과를 제공함과 동시에 해결 알고리즘의 효율성을 증대시켰습니다. 최종적으로 기업 현실 문제에 대한 최대 만족도를 달성하는 데 기여할 수 있는 통찰력을 제공하고 있습니다.



### Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction (https://arxiv.org/abs/2501.17459)
Comments:
          9 pages, 7 figures

- **What's New**: 이 연구는 항공기 비행 궤적 예측 문제를 언어 모델링 문제로 재구성하여 대규모 언어 모델(LLMs)의 활용 가능성을 탐구합니다. 특히, ADS-B 항공 데이터에서 항공기의 위치 및 상태를 나타내는 특징을 추출하여 프롬프트 기반 데이터셋을 구성하고, 이 데이터셋을 사용하여 LLM을 미세 조정합니다. 이러한 접근 방식은 복잡한 시공간 패턴을 학습하여 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: 비행 궤적 예측은 다변량 시계열 문제로 간주되며, 이는 단일 단계 예측과 다단계 예측으로 나뉩니다. 이 연구에서는 단기 예측에 초점을 맞추어 과거의 상태 매개변수를 기반으로 항공기의 미래 상태를 예측합니다. LLM 기반 방법은 표준화에 대한 의존도를 줄여주는 구조화된 워크플로를 채택하여 높은 정확도의 예측을 가능하게 합니다.

- **Performance Highlights**: 포괄적인 실험 결과, LLMs가 기존의 전통적인 방법에 비해 단일 단계 및 다단계 예측 모두에서 주목할 만한 성능 향상을 보여주었습니다. 특히 LLaMA-3.1 모델이 가장 높은 전반적인 정확성을 기록했습니다. 그러나 LLM의 높은 추론 지연(latency)은 실시간 응용 분야에서 도전 과제가 되어, 추가적인 연구 필요성을 강조합니다.



### Intensional Inheritance Between Concepts: An Information-Theoretic Interpretation (https://arxiv.org/abs/2501.17393)
- **What's New**: 이 논문은 두 개념 간의 'intensional inheritance' 개념을 정형화하고 양화하는 문제에 접근합니다. 저자들은, F로부터 W에 대한 intension적 상속의 양을 'x는 F이다'라는 명제가 'x는 W이다'라는 명제에 제공하는 정보의 양으로 정의합니다. 이를 바탕으로 Shannon 정보 이론과 알고리듬 정보 이론을 활용하여 다양한 경우의 공식들을 유도하고, 전통적인 집합 이론의 '확장적 상속'과의 관계를 논의합니다.

- **Technical Details**: 논문은 개념 F와 W가 각각 속성 집합 {F1, F2, ..., Fn} 및 {W1, W2, ..., Wm}으로 정의되며, 각 속성의 정도는 {d1, d2, ..., dn} 및 {e1, e2, ..., em}으로 표현됩니다. F와 W의 속성이 겹칠 수 있는 상황을 고려하여, 이론적으로는 두 가지 서로 다른 접근법을 통해 intension적 상속의 공식을 도출합니다. 구체적으로는 모든 속성이 상호 배타적인 특수한 경우를 분석하여 두 프레임워크에서 intension적 상속을 계산합니다.

- **Performance Highlights**: 저자들은 계산된 intension적 상속이 정보 이론적 프레임워크 내에서 전통적인 확장적 상속의 특별한 사례로 나타난다고 결론짓습니다. 또한, P(W|F)와 같은 조건부 확률을 상호 정보 공식을 기반으로 도출하여 논의하고, 이는 AI 및 AGI 시스템에서의 추론 과정에 실질적인 영향을 미칠 것으로 기대됩니다.



### A sketch of an AI control safety cas (https://arxiv.org/abs/2501.17315)
- **What's New**: 이 논문에서는 AI 모델의 안전성을 입증하기 위한 'control safety case'(제어 안전 사례) 개념을 소개하고 있습니다. 이는 AI 개발자들이 모델이 통제 조치를 전복할 수 없다는 구조화된 주장을 통해 안전성을 보장할 수 있도록 돕는 방법입니다. 또한, AI 회사 내에서 내부적으로 배치될 LLM(Large Language Model) 에이전트의 민감한 정보 유출 가능성을 줄이기 위한 구체적인 사례를 제공합니다.

- **Technical Details**: 논문에서 제시한 안전 사례는 'AI control' 방법론을 사용하여 모델이 어떤 방법으로도 안전 조치를 무시할 수 없다는 것을 보여줍니다. 구체적으로, 개발자들은 정기적인 정렬 평가(periodic alignment evaluation) 및 안전 계층(safety layer)과 같은 제어 프로토콜을 구현하여 LLM 에이전트가 민감한 데이터를 외부로 유출하는 것을 방지하는 데 초점을 맞추고 있습니다. 이러한 제어 프로토콜의 효과성을 확인하기 위해 블랙박스 제어 평가(black-box control evaluation)를 수행하며, 이를 통해 데이터 유출 확률(data exfiltration probability)을 추정합니다.

- **Performance Highlights**: 이 안전 사례 스케치는 LLM 에이전트의 잠재적 위험을 줄이는 방법을 탐구하며, 안전 조치의 효과성을 통계적으로 추정하는 과정을 포함합니다. 제어 평가를 통해 개발자는 모델이 2,000개의 소프트웨어 엔지니어링 작업을 수행할 때 얼마나 잘 통제되는지를 평가하고, 이 데이터를 기반으로 실제 배치 환경에서의 데이터 유출 확률을 예측합니다. 이 연구는 LLM 에이전트의 안전한 배치를 위한 보다 구체적인 주장을 위한 첫걸음을 마련하고 있습니다.



### Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding (https://arxiv.org/abs/2501.17310)
- **What's New**: 새롭게 소개된 MARBLES 데이터셋은 사람들이 얼마나 많은 물건(예: 구슬)을 특정 용기에 담을 수 있는지 추정하는 과제를 포함합니다. 이 데이터셋은 이미지가 포함된 경우와 포함되지 않은 경우 모두에 대해 용량 추정 능력을 평가합니다. 새롭게 고안된 'Wisdom of Crowds' (WOC) 디코딩 전략을 이용해 LLM(대형 언어 모델)의 추정 능력을 향상시키는 방법을 제안합니다.

- **Technical Details**: MARBLES 데이터셋은 다섯 개의 서로 다른 용기(예: 1컵 분량의 건조 재료 측정컵)와 세 개의 품목(예: 미국 표준 크기인 구슬)으로 구성된 15개의 추정 질문을 포함합니다. 두 가지 조건인 언어 조건과 다중 모달 조건에서 진행된 실험을 통해 모델 및 실험 참가자 각각이 추정한 결과를 비교합니다. 또한 LLM의 추정 방법으로 WOC, 자기 일관성(self-consistency), 탐욕적 디코딩(greedy decoding)과 같은 접근 방법이 평가되었습니다.

- **Performance Highlights**: LLM과 VLM(비전 언어 모델)은 사람과 유사한 방식으로 WOC 효과를 보여주는 것으로 나타났습니다. WOC 디코딩 방법은 LLM/VLM의 추정 정확도를 높이며, 이미지가 포함된 다중 모달 조건에서 성능이 더욱 향상됩니다. 이러한 결과는 LLM의 세계 모델(world model) 평가에 있어 guesstimation 과제가 중요한 척도가 될 수 있음을 시사합니다.



### From Natural Language to Extensive-Form Game Representations (https://arxiv.org/abs/2501.17282)
Comments:
          This work has been accepted as a full paper for AAMAS 2025. This is a full version of the AAMAS 2025 proceedings

- **What's New**: 이 논문에서는 자연어로 작성된 게임 설명을 게임 이론의 확장 형식으로 변환하기 위한 두 단계의 프레임워크를 제안합니다. 이 과정에서 Large Language Models (LLMs)와 인 컨텍스트 학습(in-context learning)을 활용하여 게임의 전략적 복잡성을 다룹니다. 특히, 불완전한 정보의 문제를 해결하기 위해 정보 집합과 부분 트리 구조를 식별하는 모듈을 개발하고, 이후 이를 바탕으로 완전한 확장 형식 게임 트리를 생성합니다.

- **Technical Details**: 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계는 게임의 불완전한 정보를 처리하기 위한 모듈이 포함되어 있으며, 두 번째 단계에서는 LLM과 셀프 디버깅 모듈이 결합되어 최종 EFG(확장 형식 게임)를 생성합니다. 생성된 EFG는 pygambit이라는 파이썬 API를 사용해 나타내며, 이를 통해 내쉬 균형(Nash equilibria)과 같은 계산 작업을 자동화할 수 있습니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 다양한 LLM에서 기초 모델들에 비해 상당히 향상된 성과를 보였습니다. 모든 테스트 게임을 성공적으로 해결한 최선의 모델이 있으며, 개별 모듈들도 좋은 성과에 기여했습니다. 본 프레임워크는 다양한 전략적 단계와 조건에서 게임 설명의 정확성을 100% 달성하는 등 강력한 성능을 입증했습니다.



### Integrating Reinforcement Learning and AI Agents for Adaptive Robotic Interaction and Assistance in Dementia Car (https://arxiv.org/abs/2501.17206)
Comments:
          18 pages, 12 figures

- **What's New**: 이 연구는 사회적 지원 로봇(socially assistive robotics)과 강화 학습(reinforcement learning, RL), 대형 언어 모델(large language models, LLM), 그리고 임상 전문성을 통합하여 인지증 치매 관리의 획기적인 접근 방식을 탐구합니다. 이러한 통합은 인지증 치매 관리에 대한 실험 데이터의 부족이라는 중대한 문제를 해결하고, 인지증 환자와 로봇 돌봄자의 상호작용을 실제적으로 모델링하는 동적 시뮬레이션 환경을 제공합니다. 제안된 프레임워크는 인지 및 정서 상태를 표현하기 위한 확률적 모델을 도입하여, LLM 기반 행동 시뮬레이션을 통해 환자의 반응을 에뮬레이트합니다.

- **Technical Details**: 연구의 주요 기여는 인공지능(Artificial Intelligence, AI) 에이전트 PLWD 및 로봇 돌봄자(Agent Robot Caregiver)를 포함하는 사실적인 PLWD-돌봄자 상호작용을 모델링하는 오픈 소스 시뮬레이터의 개발입니다. 여기서 PLWD는 인지 및 정서 상태에 대한 추상적 통계 모델과 LLM 기반의 행동 시뮬레이션 모델로 구성되어 있습니다. 반면, 로봇 돌봄자 에이전트는 인식, 의사결정 및 행동 실행의 세 가지 모듈로 이루어져 있으며, LLM 모델을 이용하여 감정 인식과 의사결정을 강화합니다.

- **Performance Highlights**: 강화 학습 시스템은 LLM에 의해 개선되어 PLWD의 복잡한 요구를 효과적으로 해석하고 응답할 수 있습니다. 연구 결과에 따르면, 이 시스템은 개인화된 돌봄 전략을 제공하여 PLWD의 독립성과 삶의 질을 높이는 데 기여할 수 있습니다. 또한, 제안된 프레임워크는 인간-컴퓨터 및 인간-로봇 상호작용 연구에 중요한 기여를 제공하며, 사회적 지원 기술 분야의 혁신을 촉진합니다.



### Smart Cubing for Graph Search: A Comparative Study (https://arxiv.org/abs/2501.17201)
- **What's New**: 본 논문은 SAT 솔버의 성능을 향상시키기 위한 새로운 병렬 해결 방법인 cube-and-conquer의 사용을 다룹니다. 기존의 SAT 문제에서 성공적인 결과를 낳았던 이 방법은 동적으로 제약 조건을 학습하는 propagator를 포함한 SAT 솔버에 적용할 때의 독특한 도전에 대해 연구하였습니다. 특히 대칭 깨기(symmetric-breaking) propagator를 통해 이형 그래프(isomorphic graphs)를 제거하여 탐색 공간을 줄이는 과정을 통하여 효과를 조사하였습니다.

- **Technical Details**: 연구는 SAT Modulo Symmetries (SMS) 문제에 초점을 맞추고, 10,000시간 이상의 CPU 시간을 소요한 방대한 실험을 통해 다양한 cube-and-conquer 변형을 평가하였습니다. 이 방법론은 학습된 제약 조건을 수집하는 prerun phase, 그리고 다양한 cubing 전략과 알고리즘 구성 및 LLM(대형 언어 모델)에서 생성된 디자인 제안에 의한 파라미터 튜닝을 결합하였습니다. 이를 통해 propagator 기반의 SAT 해결을 위한 효과적인 cubing 전략에 대한 새로운 통찰력을 제공하였습니다.

- **Performance Highlights**: 제공된 실험 결과에 따르면, 최적의 방법을 통해 cube-and-conquer 개선 및 파라미터 튜닝을 통해 2-3배의 속도 향상을 달성하였습니다. 특히 난이도가 더 높은 문제에서는 추가적으로 1.5-2배의 성능 개선이 이루어졌습니다. 이러한 결과는 cube-and-conquer 방식의 유효성을 강화하고, 더 나아가 기존 SAT 솔버 기술의 한계를 명확히 확인하는 데 기여합니다.



### Letters, Colors, and Words: Constructing the Ideal Building Blocks S (https://arxiv.org/abs/2501.17188)
Comments:
          29 pages, 8 figures, submitted to SIAM Undergraduate Research Online

- **What's New**: 이 논문에서는 구성 블록 세트라는 새로운 문제를 제안합니다. 이 블록은 각각 여섯 개의 면을 가진 정육면체(cube)로, 각 면에 문자가 나타나고, 지정된 색상 팔레트(palette)에서 색상이 부여됩니다. 목표는 선택된 데이터셋에서 단일 색상의 단어(mono words) 및 다양한 색상의 단어(rainbow words)를 최대한 많이 조합하는 것입니다.

- **Technical Details**: 연구에서는 n=6 및 m=6인 경우를 고려하여, 각 면의 색상이 정확히 한 번만 나타나는 조건을 부여했습니다. 단어 목록에서는 14세 미국 청소년의 일반 영어 단어 중 최대 여섯 글자로 제한하였습니다. 이 문제는 해법 공간이 방대하여 단순 나열(brute-force) 접근이 불가능하므로, 난수 검색(random search), 시뮬레이트 어닐링(simulated annealing), 두 가지 트리 검색 방법(greedy와 best-first), 그리고 유전자 알고리즘(genetic algorithm)을 통해 문제를 해결하고자 했습니다.

- **Performance Highlights**: 유전자 알고리즘을 통해 총 2846개의 단일 색상 단어와 다양한 색상 단어를 성공적으로 조합할 수 있었습니다. 다른 방법들과 비교했을 때, 유전자 알고리즘이 가장 우수한 성과를 냈습니다. 특히, 난수 검색(random search)으로는 1881개의 단어를 조합했으나, 유전자 알고리즘을 통해 성능을 크게 향상시킬 수 있었습니다.



### Complete Chess Games Enable LLM Become A Chess Master (https://arxiv.org/abs/2501.17186)
Comments:
          NAACL 2025

- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)을 활용하여 체스를 완벽하게 플레이할 수 있는 ChessLLM 모델을 제안합니다. LLM의 여러 응용 분야 중에서 체스와 같은 추상 게임에서의 잠재력은 충분히 탐색되지 않았습니다. ChessLLM은 게임을 텍스트 형식으로 변환하여 최적의 수를 Forsyth-Edwards Notation(FEN)으로 표현하며, 단순한 지도 학습을 통해 전문가 수준의 Elo 등급인 1788을 달성했습니다.

- **Technical Details**: ChessLLM은 20억 개 이상의 토큰으로 구성된 방대한 체스 게임 데이터셋을 구축하였으며, 이 데이터셋은 FEN-최고 수 쌍으로 구성되어 체스의 현재 상태를 명확하게 나타냅니다. 모델은 open-llama-3B를 기반으로 하며, 이전의 강화 학습 방법을 언어 모델링으로 변환하여 정책 학습에 적용했습니다. 체스 게임의 전반적인 품질을 보장하기 위해 길고 짧은 라운드 데이터를 비교한 결과, 긴 라운드 데이터가 더 뛰어난 성능을 발휘함을 입증했습니다.

- **Performance Highlights**: ChessLLM은 Stockfish와의 매칭에서 61%의 승률을 기록하며 Elo 등급 0에서 1788의 성과를 달성했습니다. 모델은 Elo 등급 1에서 56%, 2에서 30%의 승률을 보였으며, 이는 전체 게임을 플레이함을 통해 평가된 결과입니다. Stockfish 대비 매우 경쟁력 있는 성능을 보여주며, 이 모델의 평가 방법은 순수한 게임 성능을 측정하는 데 중점을 두었습니다.



### An AI-Driven Live Systematic Reviews in the Brain-Heart Interconnectome: Minimizing Research Waste and Advancing Evidence Synthesis (https://arxiv.org/abs/2501.17181)
- **What's New**: Brain-Heart Interconnectome (BHI)를 위한 AI 기반 시스템이 개발되었습니다. 이 시스템은 PICOS(인구, 개입, 비교군, 결과 및 연구 설계) 기준을 자동으로 감지하고, 의미 검색과 그래프 기반 쿼리를 통해 문헌을 효율적으로 검색하고 정리합니다. BI-LSTM 모델, 연구 설계 분류기 및 GPT-3.5 기반의 Retrieval-Augmented Generation (RAG) 방법이 도입되어 연구 자원의 낭비를 줄이고 실시간 업데이트를 제공합니다.

- **Technical Details**: 이 시스템은 복잡한 BHI 데이터를 관리하기 위해 Neo4j를 활용하여 관련된 개체들 간의 관계를 그래프 형태로 표현합니다. 또한, pgVector를 이용해 문서 임베딩을 저장하고, LangChain을 통해 사용자 입력에 맞춘 쿼리 처리가 이루어집니다. Bi-LSTM 모델은 연구의 PICOS 준수 여부를 판단하고, BERTopic은 주제를 클러스터링하여 연구의 진화를 추적합니다.

- **Performance Highlights**: 시스템은 BHI 분야에서 신뢰할 수 있는 증거 합성을 지원하며, 연구자들에게 실시간으로 새로운 자료를 통합할 수 있는 기능을 제공합니다. Power BI 대시보드를 통해 사용자들은 출판 패턴 및 연구 동향을 시각적으로 확인할 수 있으며, 이는 효율적인 자원 배분 및 임상 의사결정 지원에 기여합니다. 이러한 기술들은 다른 생물 의학 분야에도 적용 가능하다는 장점을 지니고 있습니다.



### Dialogue is Better Than Monologue: Instructing Medical LLMs via Strategical Conversations (https://arxiv.org/abs/2501.17860)
- **What's New**: 현재의 의료 AI 시스템은 주로 정적인 텍스트와 질문-답변 작업에 대한 훈련과 평가로 인해 실제 임상 사고(clinical reasoning)를 복제하는 데 실패하고 있습니다. 이 논문에서는 실제 진단 시나리오(diagnostic scenarios)를 시뮬레이션하는 새로운 벤치마크(benchmark)를 소개하며, 이를 통해 가장 중요한 증거 기반(reasoning) 사고와 주의 분산(distraction) 정보를 처리하는 방법을 포함합니다.

- **Technical Details**: 우리는 USMLE(United States Medical Licensing Examination) 기준에 맞춘 노이즈(noise)와 난이도(difficulty level) 조절을 통합하여 시뮬레이션된 진단 시나리오를 구현한 벤치마크를 개발했습니다. 또한 대화 기반(fine-tuning) 방법을 탐구하며, 정적(static) 데이터셋을 대화 형식(conversational formats)으로 변환하여 반복적 reasoning 과정을 더 잘 포착하고자 했습니다.

- **Performance Highlights**: 실험 결과, 대화로 조정(dialogue-tuned)된 모델은 전통적인 방법보다 우수한 성능을 보여주었으며, 다중 라운드(multi-round) reasoning 시나리오에서 9.64% 향상, 노이즈 환경(noisy environment)에서 6.18%의 정확성 향상을 달성했습니다. 이러한 발견은 대화 조정이 임상에 맞춘 강력한 의료 AI 시스템 발전을 위한 유망한 접근 방식임을 강조합니다.



### Improving Your Model Ranking on Chatbot Arena by Vote Rigging (https://arxiv.org/abs/2501.17858)
- **What's New**: 본 논문에서는 Chatbot Arena에서 대형 언어 모델(LLM)의 평가를 위한 투표 시스템에서 발생할 수 있는 조작 가능성을 다룹니다. 연구진은 특정 모델($m_{t}$)의 순위를 조작하기 위한 두 가지 전략을 제시했습니다. 첫 번째는 직접적으로 $m_{t}$가 포함된 투표에 집중하는 '타겟 전용(TARGET-ONLY) 조작' 전략입니다. 두 번째는 모든 투표에 영향을 미칠 수 있는 '전방위적인(OMNIPRESENT) 조작' 전략으로, 이는 보다 효과적으로 $m_{t}$의 순위를 조정할 수 있습니다.

- **Technical Details**: Chatbot Arena는 사용자들이 무작위로 선택된 두 개의 모델 간의 응답을 비교하여 투표하는 방식으로 설계되었습니다. 각 모델의 Elo 점수는 수집된 사용자 투표를 기반으로 계산되며, 이는 모델 순위의 변경을 가능하게 합니다. 특히, 새로운 투표가 이루어지면, 반드시 $m_{t}$ 모델이 직접 포함되지 않더라도 순위에 영향을 줄 수 있는 가능성이 있습니다. 연구팀은 약 170만 건의 투표 기록을 활용해 다양한 조작 시나리오를 실험했습니다.

- **Performance Highlights**: 실험 결과, 전방위적 조작 전략이 단 몇 백 건의 투표만으로도 모델의 순위를 크게 변경할 수 있음을 확인했습니다. 이는 타겟 전용 조작 전략에 비해 획기적으로 더 효율적이며, 기존의 여러 방어 메커니즘에도 불구하고 조작이 가능하다는 점을 부각시킵니다. 이 연구는 투표 조작의 방지를 위한 지속적인 노력의 중요성을 강조하며, 실제 Chatbot Arena 플랫폼에서 가능한 조작 방식에 대한 경각심을 제공합니다.



### GRACE: Generalizing Robot-Assisted Caregiving with User Functionality Embeddings (https://arxiv.org/abs/2501.17855)
Comments:
          10 pages, 5 figures, Accepted to IEEE/ACM International Conference on Human-Robot Interaction (HRI), 2025

- **What's New**: 본 연구는 개인화된 기능 범위(fROM)를 예측하여 로봇 돌봄의 결정 과정을 일반화하는 새로운 방법을 제안합니다. 연구팀은 작업 치료에서의 기능 평가 점수를 바탕으로 개인화된 기능 범위를 예측하는 드문 데이터 기반 방법을 개발했습니다. 이 모델은 사용자의 운동 제한을 에뮬레이션하여 수집한 데이터로 학습하여 신규 사용자에 대한 fROM을 예측할 수 있습니다.

- **Technical Details**: 우리는 fROM을 사용자 기능의 중심 표현으로 간주하며, 이를 통해 로봇은 다양한 일상 생활 활동(ADLs)에서 맞춤형 지원을 제공할 수 있습니다. GRACE라는 신경 모델을 제안하여 기능 평가 점수를 사용자 기능의 잠재적인 표현으로 변환하는 작업을 수행합니다. 이를 통해 얻은 데이터셋 DataGRACE는 다양한 이동 조건에서의 fROM과 기능 점수를 결합하여 첫 번째 공개 데이터셋을 형성합니다.

- **Performance Highlights**: 시뮬레이션 실험과 실제 로봇 사용자 연구를 통해, GRACE는 과제 성공과 사용자 에이전시 간의 균형을 잘 맞추도록 지원함을 보여주었습니다. 개인화된 기능 범위 예측을 통해 로봇이 효과적으로 사용자 맞춤형 지원을 제공할 수 있으며, 이는 약 13억 명의 장애인을 대상으로 하는 돌봄 로봇스케일을 확장하는 데 기여할 수 있습니다.



### From Sparse to Dense: Toddler-inspired Reward Transition in Goal-Oriented Reinforcement Learning (https://arxiv.org/abs/2501.17842)
Comments:
          Extended version of AAAI 2024 paper: Unveiling the Significance of Toddler-Inspired Reward Transition in Goal-Oriented Reinforcement Learning. This manuscript is currently being prepared for journal submission

- **What's New**: 최근 연구에서는 아동의 학습 방식에서 영감을 얻어 Sparse-to-Dense (S2D) 보상 전환을 통해 강화 학습(RL) 에이전트의 성능을 향상시키는 방법을 제안했습니다. 이 방법은 보상이 드문 환경에서의 자유 탐색에서 목표 지향적 행동으로의 전환을 통해 강화 학습의 학습 효율성과 샘플 효율성을 크게 높입니다. 또한, S2D 전환이 정책 손실 경관을 부드럽게 하여 일반화 능력 향상에 기여하는 것을 보여주고 있습니다.

- **Technical Details**: 이 연구는 S2D 보상 구조가 RL 에이전트의 성능, 정책 손실 경관, 초기 탐색의 역할에 미치는 영향을 분석합니다. 제안된 방법은 보상의 밀도를 조정하면서도 최적의 정책을 유지하기 위해 잠재 기능(potential function)을 통합하고, 내재적 동기 부여 알고리즘을 통해 탐색-착취의 균형을 지원합니다. 실험 결과는 여러 복잡한 목표 지향 RL 환경에서 S2D 전환이 성공률과 샘플 효율성을 높였음을 보여줍니다.

- **Performance Highlights**: S2D 전환은 학습 성능을 개선하고 정책 손실 경관을 부드럽게 하여 일반화 성능을 향상시킵니다. 구체적으로, S2D 접근 방식은 RL 에이전트가 더 넓은 최소값을 갖도록 하여 미세한 파라미터 변화에 덜 민감한 해결책을 제공합니다. 실험 결과는 다양한 환경에서 S2D 보상 전환의 유효성을 입증하며, 초기 자유 탐색이 강력한 초기 정책을 수립하고 이후의 밀집 보상 단계에서 일반화 및 안정성을 향상시키는데 중요한 역할을 한다는 것을 보여줍니다.



### U2A: Unified Unimodal Adaptation for Robust and Efficient Multimodal Learning (https://arxiv.org/abs/2501.17823)
Comments:
          14 Pages, 6 Figures, 6 Tables

- **What's New**: 이번 논문에서는 Unified Unimodal Adaptation (U2A)라는 새로운 방법을 제안합니다. 이는 낮은 순위의 적응(low-rank adaptation, LoRA)을 활용하여 미리 훈련된 단일 모달 인코더를 조정함으로써 다양한 다중 모달 작업을 수행합니다. U2A는 복잡한 훈련 전략 없이 필요한 학습 가능한 매개변수를 크게 줄이며, 특히 훈련 및 테스트 시 결여된 모달을 다룰 수 있는 Mask Tokens (MT)도 도입합니다.

- **Technical Details**: U2A는 미리 훈련된 transformer 인코더를 통해 각 모달을 별도로 인코딩합니다. 결여된 모달을 처리하기 위해 Mask Tokens를 도입하여 사용 가능한 입력 모달로부터 결여된 모달 특징을 추정합니다. 여기에는 모달 정렬 손실(modality alignment loss)도 포함되어, 결여된 모달의 클래스 토큰을 효과적으로 대체할 수 있도록 학습합니다.

- **Performance Highlights**: U2A는 테스트에서 다중 모달 환경에서 우수한 성능을 보이며, 많은 기존 방법들보다 더 적은 학습 가능한 매개변수로도 성능을 유지합니다. 다양한 데이터셋에 대한 실험 결과, U2A는 완전한 모달 및 결여된 모달 상황 모두에서 최첨단(state-of-the-art, SOTA) 방법을 초과하거나 동등한 성능을 기록했습니다. 실험을 통해, 결여된 모달 특성을 정확하게 추정하는 능력도 입증되었습니다.



### Aggregation Schemes for Single-Vector WSI Representation Learning in Digital Pathology (https://arxiv.org/abs/2501.17822)
- **What's New**: 이 논문에서는 Whole Slide Image(WSI) 검색 성능을 평가하기 위해 여러 최근 개발된 집계 기법을 비교합니다. 특히, WSI의 고해상도 특성 때문에 대규모 패치 라벨링 및 특정 집합 표현 학습 기술을 탐구합니다. 기존의 패치 임베딩 세트를 단일 임베딩으로 변환하는 다양한 방법을 제안하며 성능을 비교하고 있습니다.

- **Technical Details**: WSI는 일반적으로 수많은 작은 "패치"나 타일로 분할되며, 이를 통해 각 WSI에 대해 패치 임베딩 세트를 생성합니다. 이 과정에서 다양한 집계 알고리즘이 사용되며, 각 알고리즘은 패치 임베딩 세트에서 단일 벡터로 깊은 특징을 추출합니다. Fisher Vector, Deep Sets, Memory Networks, Focal Attention 등 다양한 접근 방식을 사용하여 WSI의 임베딩 단일화 및 표준화를 목표로 하고 있습니다.

- **Performance Highlights**: 연구에서는 TCGA에서 제공하는 방광, 유방, 신장, 대장과 같은 여러 데이터셋을 통해 k-NN 검색 방식을 적용하여 성능을 평가했습니다. 또한, 이들 방법의 검색 성능을 실제 WSI 검색 시스템에 적용됨에 따라 메디안 최소 거리 기반 접근 방식과 비교하여 더 나은 결과를 도출된 것으로 나타났습니다. 최종적으로 각 방식의 단일 벡터 생성 성능을 분석하고 결과를 상세히 논의합니다.



### P-TAME: Explain Any Image Classifier with Trained Perturbations (https://arxiv.org/abs/2501.17813)
Comments:
          Submitted for publication

- **What's New**: 이 논문에서는 DNN 기반 이미지 분류기를 설명하기 위한 모델 불문(모델-아그노스틱) 방법인 P-TAME (Perturbation-based Trainable Attention Mechanism for Explanations)를 소개합니다. P-TAME는 보조 이미지 분류기를 활용하여 입력 이미지의 특징을 추출하고, DNN의 내부 아키텍처에 맞게 설명 방법을 조정할 필요 없이 설명을 생성합니다. 전통적인 방 perturbation-based 방법의 높은 계산 요구 사항을 해결하고, 단일 forward pass에서 높은 해상도의 설명을 생성하는 효율적인 대안을 제공합니다.

- **Technical Details**: P-TAME는 대체 이미지 분류기를 통해 기능 맵을 추출하며, 이를 통해 DNN의 주요 결정과 관련된 부분을 강조하는 방식으로 설명을 생성합니다. 이 방법은 여러 DNN 아키텍처에 적용 가능하여 모델-아그노스틱한 특성을 가지며, 훈련 후 단일 forward 단계에서 설명을 제공합니다. 실험에 사용된 이미지 분류기로는 VGG-16, ResNet-50 및 ViT-B-16이 있으며, 이들은 모두 ImageNet에 대해 훈련되었습니다.

- **Performance Highlights**: P-TAME의 성능은 이전의 설명 가능성 방법, 특히 T-TAME 및 기타 최신 기술과 비교하여 양적(quantitative) 및 질적(qualitative) 결과가 뛰어난 것을 보여줍니다. 이러한 비교는 P-TAME가 설명의 질에서 최신 perturbation 방식과 견주어 최대한의 성능을 발휘할 수 있음을 나타냅니다. 또한, 연구자들이 이 방법을 쉽게 사용할 수 있도록 P-TAME를 오픈 소스로 제공하고 있습니다.



### International AI Safety Repor (https://arxiv.org/abs/2501.17805)
- **What's New**: 이번 논문은 최초의 국제 AI 안전 보고서(International AI Safety Report)로, 고급 AI 시스템의 기능, 위험 및 안전성에 대한 현재 증거를 종합적으로 분석합니다. 이 보고서는 영국 블렛클리에서 열린 AI 안전 정상 회의(AI Safety Summit) 참석 국가들에 의해 주문되었습니다. 총 30개국과 UN, OECD, EU가 각기 대표를 선정하여 전문가 자문 패널(Expert Advisory Panel)을 구성하였습니다.

- **Technical Details**: 보고서 작성에는 100명의 AI 전문가들이 참여하였으며, 이들은 다양한 관점과 학문 분야를 대표합니다. 보고서의 의장(Chair) 및 독립 전문가들은 보고서 내용에 대한 전적인 재량권을 가지고 있으며, 이는 각국의 다양한 경험을 반영하여 보다 포괄적이고 신뢰할 수 있는 결과를 제공하도록 설계되었습니다.

- **Performance Highlights**: 이 보고서는 고급 AI 시스템의 잠재적인 위험 및 안전성을 평가함으로써, AI 기술의 발전과 관련된 정책적 논의에 기여하는 중요한 자료로 기능할 것으로 보입니다. 다양한 국가와 국제 기구의 협력을 통해, AI의 안전성을 보장하기 위한 공동의 노력이 강화될 것으로 기대됩니다.



### BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation -- Challenges and Insights (https://arxiv.org/abs/2501.17790)
- **What's New**: 이번 논문에서는 BreezyVoice라는 대만 만다린을 위한 텍스트 음성 변환(Text-to-Speech, TTS) 시스템을 소개합니다. 이 시스템은 음소 해석(polyphone disambiguation)의 독특한 문제를 해결하기 위해 음소 제어(phonetic control) 기능을 강조하고 있습니다. BreezyVoice는 CosyVoice를 기반으로 하여 $S^{3}$ tokenizer, 대규모 언어 모델(large language model, LLM), 최적 수송 조건부 흐름 매칭 모델(optimal-transport conditional flow matching model, OT-CFM) 등의 요소를 통합하여 현실감 있는 음성을 생성합니다.

- **Technical Details**: BreezyVoice 프레임워크는 Supervised Semantic Speech (S3) Tokenizer, 대규모 언어 모델(LLM), 최적 수송 조건부 흐름 매칭 모델(OT-CFM), 그리고 g2pW(grapheme to phoneme prediction model)로 구성되어 있습니다. 이 시스템은 각 요소의 상호작용을 통해 음성을 디지털 단위로 변환하고, 이를 통해 여러 표현 방식에서 음성이 자연스럽게 생성되도록 합니다. 특히, OT-CFM 모델을 활용하여 시간이 지남에 따라 의미 있는 데이터 분포로 변환되는 스펙트로그램을 생성하여 음성의 시간 주파수 구조를 정확하게 반영합니다.

- **Performance Highlights**: BreezyVoice는 일반 및 코드 스위칭(code-switching) 환경에서 상업 TTS 시스템에 비해 우수한 성능을 보여줍니다. 연구 결과, 이 시스템은 다루기 어려운 장기 스피커(long-tail speaker) 및 음소 해석(polyphone disambiguation) 문제를 특정적으로 해결함으로써 높은 충실도(high-fidelity)의 음성을 생성할 수 있음을 입증했습니다. 이러한 성과는 신경 코덱 TTS 시스템의 작동 원리에 대한 귀중한 통찰을 제공합니다.



### 2SSP: A Two-Stage Framework for Structured Pruning of LLMs (https://arxiv.org/abs/2501.17771)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 구조적 가지치기를 위한 새로운 Two-Stage Framework for Structured Pruning (2SSP)을 제안합니다. 2SSP는 Width Pruning과 Depth Pruning이라는 두 가지 다른 가지치기 전략을 결합하여 모델의 성능 저하를 최소화하면서도 계산 비용을 줄이는 데 초점을 맞추고 있습니다. 첫 번째 단계는 중간 상태의 Feed-Forward Networks에서 신경망을 제거하고, 두 번째 단계는 Attention 서브모듈을 제거하는 방식으로 진행됩니다.

- **Technical Details**: 2SSP의 첫 번째 단계는 Width Pruning을 사용하여 각 신경망의 중요도를 평가한 후, 신경망을 제거하는 것입니다. 이 과정은 Transformer 블록의 구조적 연결성을 유지하면서 진행됩니다. 두 번째 단계에서는 깊이 가지치기를 통해 Attention 서브모듈을 제거하며, 이는 주어진 성능 지표(예: perplexity)를 최소화하는 방향으로 반복적으로 수행됩니다.

- **Performance Highlights**: 제안된 2SSP 방법은 네 가지 LLM 계열과 세 가지 가지치기 비율(25%, 37.5%, 50%)에 대해 테스트되었으며, 관련된 언어 모델링 데이터셋과 6개의 다운스트림 태스크에서 결과를 측정했습니다. 이 방법은 세 가지 언어 모델링 및 여섯 개 다운스트림 태스크에서 최근의 최첨단 방법들보다 일관되게 우수한 성능을 보여주었으며, 가지치기 시간 측면에서도 최대 두 배의 이득을 기록하였습니다.



### Hybrid Graphs for Table-and-Text based Question Answering using LLMs (https://arxiv.org/abs/2501.17767)
Comments:
          Accepted at NAACL 2025 Main Track

- **What's New**: 이번 논문에서는 기존의 고품질 데이터에 의존하지 않고 대규모 언어 모델(Large Language Models, LLMs)을 이용한 새로운 하이브리드 그래프 기반 접근 방식을 제안합니다. 이 방법은 텍스트 및 표 데이터를 조합하여 하이브리드 그래프(Hybrid Graph)를 구성하고, 입력된 질문에 따라 정보의 중요성을 조정하여 LLM에 관련된 맥락을 제공합니다. 이는 멀티 소스 테이블-텍스트 질문 답변(Table-Text QA) 분야에서의 가능성을 보여주는 중요한 발전입니다.

- **Technical Details**: 제안하는 방법은 하이브리드 그래프를 통해 텍스트 데이터와 테이블 데이터를 통합하고, 입력 질문에 따라 불필요한 정보를 제거하는 방식으로 작동합니다. 이 접근법은 별도의 파인튜닝(fine-tuning) 과정 없이도 LLM을 효과적으로 활용할 수 있습니다. 평가에는 최신 LLM인 GPT-3.5, GPT-4, LLaMA-3를 사용하였고, Hybrid-QA와 OTT-QA라는 어려운 데이터셋에서 성능을 측정하였습니다.

- **Performance Highlights**: 제안된 방법은 두 데이터셋에서 제로샷(zero-shot) 성능에서 가장 우수한 결과를 보였으며, Hybrid-QA에서 Exact Match 점수가 10% 증가하고 OTT-QA에서 5.4% 향상되었습니다. 또한, 이 접근법은 원래의 맥락에 비해 토큰 사용(Token usage)을 최대 53% 줄이는 효과를 보였습니다.



### Yin-Yang: Developing Motifs With Long-Term Structure And Controllability (https://arxiv.org/abs/2501.17759)
Comments:
          16 Pages, 4 Figures, Accepted at Artificial Intelligence in Music, Sound, Art and Design: 14th International Conference, EvoMUSART 2025

- **What's New**: 이번 연구에서는 모티프를 장기 구조를 갖춘 멜로디로 발전시키기 위한 새로운 프레임워크 Yin-Yang을 제안합니다. Yin-Yang은 프레이즈 생성기(phrase generator), 프레이즈 정제기(phrase refiner), 프레이즈 선택기(phrase selector) 모델로 구성됩니다. 이러한 구조는 모티프의 변형을 생성하는 혁신적인 손상-정제(corruption-refinement) 전략을 통해 다루어 전반적인 일관성을 유지합니다.

- **Technical Details**: 프레이즈 생성기는 자가 회귀 모델(autoregressive model)로서 주어진 데이터 세트의 스타일을 학습하여 콘텐츠를 생성합니다. 이 과정에서 프레이즈 정제기가 보조 역할을 하여 모티프의 변형을 부드럽게 되돌리거나 새로운 모티프를 정의하도록 돕습니다. 연구에서는 모티프가 새로운 문맥 내에서 의미 있는 변화를 생성하도록 훈련된 프레이즈 정제기가 중요한 역할을 합니다.

- **Performance Highlights**: 제안된 모델은 기존의 최첨단 변환기(transformer) 모델과 비교하여 더 나은 성능을 기록했으며, 구조적인 제어 가능성과 더불어 생성된 음악 구조의 반 해석 가능성(semi-interpretability)을 덧붙였습니다. 또한, 새로운 객관적인 평가 지표를 제안하여 모티프가 얼마나 부드럽게 발현되는지를 정량적으로 평가할 수 있습니다.



### AI Governance through Markets (https://arxiv.org/abs/2501.17755)
- **What's New**: 이 논문은 인공지능(AI) 거버넌스의 주요 접근법으로 시장 거버넌스 메커니즘을 제안합니다. 현재의 규제 틀과 함께 보험, 감사, 조달 및 실사의 네 가지 시장 기반 접근 방식을 통해 AI 개발을 책임감 있게 하는 효과적인 인센티브를 제공할 수 있습니다. 또한 표준화된 AI 공개 및 시장 메커니즘을 통해 AI 리스크와 재무 리스크 간의 관계를 확립할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 AI 위험을 감소시키기 위한 시장 거버넌스 메커니즘으로 보험, 감사, 조달 및 실사를 다룹니다. 이러한 메커니즘은 경제 행동을 구조화하여 원하는 사회적 또는 경제적 결과와 재정적 인센티브를 일치시키도록 설계되었습니다. 이러한 접근법은 시장 참여자 간의 리스크 분배와 정보 발견을 가능하게 하며, AI에서 발생하는 위험을 측정하고 완화하는 데 기여합니다.

- **Performance Highlights**: AI 리스크와 관련된 불확실성은 AI 기술의 기업 내 도입에 큰 장애물로 작용하고 있습니다. 이 논문은 시장 거버넌스가 AI 거버넌스를 단순히 엄격한 규제로 넘어서 새롭게 조망할 수 있는 기회를 제공한다고 강조합니다. AI의 개발과 시장 활용을 통해 기업과 사회 전반에 걸쳐 안전하고 책임감 있는 AI 발전을 위한 강력한 인센티브를 창출할 수 있음을 입증합니다.



### Early External Safety Testing of OpenAI's o3-mini: Insights from the Pre-Deployment Evaluation (https://arxiv.org/abs/2501.17749)
Comments:
          arXiv admin note: text overlap with arXiv:2501.17132

- **What's New**: 이번 논문에서는 Mondragon University와 University of Seville의 연구자들이 OpenAI의 새로운 o3-mini LLM에 대한 외부 안전성 테스트를 수행한 결과를 보고합니다. 특히, ASTRAL이라는 도구를 활용하여 LLM의 안전성 카테고리를 테스트하기 위해 최신의 위험한 테스트 입력을 자동 생성했습니다. 테스트 과정에서 총 10,080개의 위험한 테스트 입력을 생성 해당 모델이 공개되기 전에 테스트를 진행하여 몇 가지 안전성 관련 문제점을 발견했습니다.

- **Technical Details**: ASTRAL은 LLM의 안전성을 테스트하기 위해 자동으로 위험한 테스트 입력을 생성하는 새로운 도구입니다. 이 도구는 Retrieval Augmented Generation (RAG) 및 few-shot prompting 기법을 활용하여 14가지의 안전성 카테고리에 대해 최신의 테스트 입력을 생성합니다. ASTRAL은 LLM의 출력을 안전하거나 위험하다고 분류하는 자동화된 메커니즘을 제공하여 테스트 오라클 문제를 줄이는 데 기여합니다.

- **Performance Highlights**: OpenAI의 o3-mini 모델에 대한 테스트 결과, 전체적으로 이 모델의 안전성이 높다는 것이 확인되었으나, 일부 안전하지 않은 결과를 초래하는 문제점이 발견되었습니다. 이 발견을 통해 OpenAI는 모델의 안전성을 향상시키기 위한 개선 작업을 수행할 수 있는 기회를 가졌습니다. 논문은 이러한 안전 관련 테스트 과정에서 중요한 통찰 및 발견 내용도 강조하고 있습니다.



### Exact characterization of {\epsilon}-Safe Decision Regions for exponential family distributions and Multi Cost SVM approximation (https://arxiv.org/abs/2501.17731)
- **What's New**: 본 연구에서는 데이터 기반 분류기의 예측에 대한 확률적 보장을 부여하는 것을 강조하고 있습니다. 이를 위해, 안전하다고 고려되는 특정 클래스에 대한 예측이 확률적으로 보장되는 {	extepsilon}-안전 결정 영역(ε-Safe Decision Region)의 공식적인 정의를 소개합니다. 또한, 본 연구는 경량형 Support Vector Machine(서포트 벡터 머신)인 Multi Cost SVM을 제안하여, 불균형 데이터에서도 안전한 분류 영역을 근사할 수 있음을 입증합니다.

- **Technical Details**: 본 논문은 데이터가 지수 패밀리 분포에서 나온 경우의 안전 결정 영역의 형태가 설계 파라미터에 의해 분석적으로 결정된다는 것을 보여줍니다. Multi Cost SVM 알고리즘은 불균형 데이터 처리 및 안전 영역 근사 기능을 제공하며, 각 데이터의 사전 확률에 대한 강건성을 유지합니다. 또한, 이 알고리즘은 여러 가중치가 부여된 SVM들을 동시에 훈련시켜 모델의 일반화 능력을 높입니다.

- **Performance Highlights**: 연구에서 제안된 알고리즘은 의료/재무/자율주행 등 다양한 분야에 적용될 수 있는 가능성을 보여줍니다. 예를 들어, rare diseases 감지 및 신용 카드 사기 탐지와 같은 분야에서 실질적으로 매우 시험적입니다. 이 연구는 분류의 '안전성'에 대한 명확한 정의와 다양한 데이터 조건에 적응 가능한 강건한 이론을 제공하여, 안전한 머신러닝(SafeML)이라는 큰 목표에 기여하고 있습니다.



### PulmoFusion: Advancing Pulmonary Health with Efficient Multi-Modal Fusion (https://arxiv.org/abs/2501.17699)
- **What's New**: 이번 연구에서는 기존의 원격 폐 기능 측정 기술의 한계를 극복하기 위해 RGB와 열화상 비디오 데이터 및 환자 메타데이터를 통합한 새로운 비침습적 접근법을 제안합니다. Spiking Neural Networks (SNNs)를 활용하여 Peak Expiratory Flow (PEF)와 Forced Expiratory Volume (FEV1) 및 Forced Vital Capacity (FVC) 분류를 수행하며, 경량 CNN을 통해 SNN의 회귀 작업 한계를 극복합니다. 또한, 멀티헤드 어텐션 레이어와 K-Fold 검증 방식, 앙상블 학습을 활용하여 모델의 강건성을 높입니다.

- **Technical Details**: 모델은 비디오 기반 학습 접근 방식으로 전환하여 전체 호흡 사이클 측정을 수행하는 복잡성 문제를 해결합니다. SNN은 Leaky Integrate-and-Fire (LIF) 뉴런 구조를 사용하여 생물학적 뉴런을 모델링하고, Spike 타이밍에 기초하여 데이터를 효율적으로 처리합니다. PulmoFusion 모델은 비디오 시퀀스를 처리하여 시공간 패턴을 식별하며, 메타데이터는 스파이크 트레인을 통해 통합됩니다.

- **Performance Highlights**: 열화상 데이터를 사용하여 SNN 모델은 호흡 주기를 기준으로 92%의 정확도와 환자별로 99.5%의 정확도를 달성했습니다. PEF 회귀 모델은 열화상에서 0.11의 상대 RMSE와 RGB에서 0.26의 RMSE를 보였으며, FEV1/FVC 예측에 대한 평균 절대 오차(MAE)는 4.52%로, 최신 성능을 확인했습니다.



### Segmentation-Aware Generative Reinforcement Network (GRN) for Tissue Layer Segmentation in 3-D Ultrasound Images for Chronic Low-back Pain (cLBP) Assessmen (https://arxiv.org/abs/2501.17690)
- **What's New**: 이 논문에서는 segmentation-aware joint training framework인 generative reinforcement network (GRN)을 소개합니다. 이 프레임워크는 segmentation loss 피드백을 통합하여 이미지 생성과 segmentation 성능을 동시에 최적화할 수 있도록 설계되었습니다. 또한, segmentation-guided enhancement (SGE)라는 이미지 향상 기법을 개발하여, generator가 segmentation 모델에 맞춘 이미지를 생성하게 합니다.

- **Technical Details**: GRN은 두 가지 변형 버전이 개발되었으며, 각각 sample-efficient learning (GRN-SEL)과 semi-supervised learning (GRN-SSL)입니다. 연구에 사용된 데이터셋은 29명의 피험자로부터 얻은 69개의 3D 초음파 스캔으로, 여섯 가지 해부학적 구조(dermis, superficial fat, superficial fascial membrane (SFM), deep fat, deep fascial membrane (DFM), muscle)가 포함되어 있습니다. GRN은 이러한 구조에 대한 라벨링 노력을 줄이기 위해 최적화되었습니다.

- **Performance Highlights**: GRN-SEL은 SGE를 사용할 경우 라벨링 노력을 최대 70%까지 줄이면서 Dice Similarity Coefficient (DSC)에서 1.98% 개선을 보였습니다. GRN-SEL 단독은 라벨링 노력을 60% 줄였고, GRN-SSL의 경우 SGE를 사용할 경우 라벨링 요구 사항을 70% 감소시켰습니다. 이러한 결과는 GRN 프레임워크가 적은 라벨 데이터로 segmentation 성능을 최적화하는 데 효과적임을 시사하며, 초음파 이미지 분석에 있어 데이터 주석의 부담을 줄이는 스케일 가능한 솔루션을 제안합니다.



### ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer (https://arxiv.org/abs/2501.17688)
- **What's New**: 이 논문에서는 Contourformer라는 실시간 윤곽 기반 인스턴스 분할 알고리즘을 제안합니다. 이 방법은 DETR(paradigm)으로 완전히 기반하여 반복적이고 점진적인 메커니즘을 통해 윤곽을 최적화하여 최종 결과를 end-to-end 방식으로 얻습니다. 또한 효율성과 정확성을 향상시키기 위해 서브 윤곽 분리 메커니즘과 윤곽 세밀한 분포 개선 두 가지 혁신적인 기술을 개발했습니다.

- **Technical Details**: Contourformer의 프레임워크는 D-FINE 객체 탐지 모델을 기반으로 하여 바운딩 박스의 회귀를 윤곽 회귀로 확장합니다. 이 구조는 효율적인 훈련을 위해 윤곽 변형의 반복적 방법을 사용하며, 수렴 속도를 높이기 위해 노이즈 제거 메커니즘을 도입합니다. 모델은 각 반복에서 윤곽 추정의 점진적인 개선을 가능하게 하기 위해 DETR 디코더의 반복 아키텍처를 재설계했습니다.

- **Performance Highlights**: Contourformer는 SBD, COCO 및 KINS와 같은 여러 벤치마크 데이터세트에서 뛰어난 성능을 보여줍니다. 실제로 512x512 크기의 이미지에서 NVIDIA A30 GPU를 이용해 24.6 fps의 추론 속도를 달성하였으며, 기존의 윤곽 기반 인스턴스 분할 알고리즘보다 정확도가 현저히 향상된 결과를 보였습니다.



### Planning with Vision-Language Models and a Use Case in Robot-Assisted Teaching (https://arxiv.org/abs/2501.17665)
- **What's New**: Image2PDDL는 Vision-Language Models(VLMs)를 활용하여 이미지와 텍스트 입력을 자동으로 Planning Domain Definition Language(PDDL) 문제로 변환하는 새로운 프레임워크입니다. 이 접근 방식은 복잡한 현실 세계의 작업을 다루는 AI 계획에서 중요한 도전과제를 해결합니다. Image2PDDL는 초기 상태의 이미지와 목표 상태 설명을 함께 제공하여, 기호적 계획과 지각적 이해를 연결하는데 기여합니다.

- **Technical Details**: 이 프레임워크는 초기 상태와 목표 상태를 입력으로 받아 PDDL 문제로 변환하는 세 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 초기 상태 이미지의 공간적 관계를 파악하여 텍스트 표현으로 변환합니다. 목표 상태는 이미지 또는 텍스트 설명으로 처리되어 일관된 형식으로 변환되며, 마지막으로 PDDL 문제 생성 단계에서 모든 정보를 바탕으로 최종 결과가 도출됩니다.

- **Performance Highlights**: Image2PDDL는 Blocksworld, Sliding-Tile Puzzle 및 Kitchen Domain의 다양한 도메인에서 평가되었으며, 각 도메인마다 난이도에 따라 50개의 고유한 시나리오를 설정하였습니다. 성능 평가는 구문 정확성과 내용 정확성을 기준으로 수행되었으며, 다양한 복잡성을 가진 작업에서 유망한 결과를 나타냈습니다. 이러한 결과는 이를 통해 AI 계획의 접근성과 확장성을 높이는 데 기여할 수 있음을 시사합니다.



### Exploring Vision Language Models for Multimodal and Multilingual Stance Detection (https://arxiv.org/abs/2501.17654)
Comments:
          Submitted to the International AAAI Conference on Web and Social Media (ICWSM) 2025

- **What's New**: 이 논문은 멀티모달 환경과 다국어 데이터를 포함하는 스탠스 감지(stance detection) 작업에서 최신 비전-언어 모델(Vision-Language Models, VLMs)의 성능을 평가합니다. 특히, 이전 연구가 텍스트 전용 데이터에 초점을 맞춘 것과 달리, 이미지와 텍스트를 동시에 사용하는 상황에 대한 연구가 부족한 점을 지적합니다. 다양한 언어로 이루어진 최근에 확장된 데이터셋을 바탕으로 VLM의 시각적 단서 및 언어별 성능을 평가하고 있습니다.

- **Technical Details**: 스탠스 감지란 특정 주제, 개체 또는 주장에 대한 사용자의 관점을 자동으로 분류하는 작업입니다. 이 논문에서는 VLM이 스탠스 감지를 수행하는 데 있어 텍스트와 이미지 정보를 얼마나 효과적으로 사용하는지를 분석하며, 다국어에서의 성능 또한 조사합니다. 이를 통해 VLMs의 텍스트와 이미지 간의 상호작용을 다국어 맥락에서 탐구하며, 데이터셋은 영어뿐만 아니라 6개 추가 언어로 확장된 멀티모달 스탠스 감지 데이터셋을 사용합니다.

- **Performance Highlights**: 실험 결과, VLMs는 스탠스 감지를 위해 일반적으로 이미지보다 텍스트에 더 의존하는 경향이 있으며, 이 경향은 모든 언어에 걸쳐 지속됩니다. 또한, VLM은 이미지 내의 텍스트에 대해 다른 시각적 콘텐츠보다 더 크게 의존하는 것으로 나타났습니다. 모델들이 명시적으로 다국어를 지향하지 않더라도, 여러 언어에서 일관된 예측을 생성하는 경향이 있지만, 전반적인 F1 점수, 언어 지원, 모델 크기와 불일치하는 이상치도 존재합니다.



### Tonguescape: Exploring Language Models Understanding of Vowel Articulation (https://arxiv.org/abs/2501.17643)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 연구에서는 언어 모델 (LM)이 실제 혀의 위치와 모음 발음을 이해할 수 있는지에 대한 질문을 다루고 있습니다. 연구팀은 기존의 실시간 MRI 데이터셋에서 영상과 이미지를 생성하여 LMs의 시각 기반 정보 활용 가능성을 검토했습니다. 연구 결과, LMs는 레퍼런스 예제가 제공될 경우 모음과 혀 위치를 이해할 잠재력을 보여주지만, 레퍼런스가 없는 경우 어려움이 있는 것으로 나타났습니다. 이 연구의 코드와 데이터셋 구축 자료는 GitHub에서 제공됩니다.

- **Technical Details**: 모음 발음을 이해하기 위해, 연구팀은 일본어 5모음 시스템을 중심으로 한 데이터셋을 구성하였습니다. 데이터셋은 22명의 일본어 사용자로부터 수집된 실시간 MRI 비디오로 구성되어 있으며, 각 모음의 발음을 촬영한 120개의 비디오를 포함하고 있습니다. 이 비디오는 혀와 인두의 움직임을 관찰할 수 있는 측면 뷰를 제공하며, 각 비디오는 정지 상태에서 시작하여 정지 상태로 끝납니다. LMs는 길게 사용된 다중 모달 정보 방식으로 텍스트뿐만 아니라 이미지, 비디오 및 오디오 정보를 처리할 수 있어야 합니다.

- **Performance Highlights**: LM들은 주어진 이미지에서 모음과 혀 위치를 예측하는 데 일부 능력을 보였지만, 제로샷(Zero-shot) 및 몇 샷(Few-shot) 학습에서는 어려움을 겪었습니다. 연구의 결과는 LMs가 언어의 음성과 관련된 아이디어를 보다 인간적인 생리학에 기반하여 이해하는 데 기여할 수 있음을 시사합니다. 이러한 통찰력은 모음 조화와 같은 언어 현상 분석에 기여할 수 있으며, 발음 교육에 있어서도 유용할 것으로 기대됩니다.



### In-Context Meta LoRA Generation (https://arxiv.org/abs/2501.17635)
- **What's New**: 본 연구에서는 다중 작업 환경에서도 효율성을 유지하면서, 대형 언어 모델(LLMs)의 작업 특화(customization)를 위한 In-Context Meta LoRA (ICM-LoRA)라는 혁신적인 접근 방식을 제안합니다. 기존의 Low-rank Adaptation (LoRA) 모델이 개별 작업에 대해 따로 훈련되는 비효율성을 해결하기 위해, Conditional Variational Autoencoder (CVAE)를 활용하여 task-aware LoRA weights를 생성합니다. ICM-LoRA는 저장 용량을 감소시키면서도 다양한 작업에 대한 정확한 LoRA 매개변수를 생성할 수 있습니다.

- **Technical Details**: ICM-LoRA는 task vectors를 사용하여 context modeling을 수행하며, 메타 학습(meta-learning)을 통해 작업 간의 관계를 포착합니다. 이를 통해 LoRA 매개변수 생성을 단일 생성자(generator)로 동시에 처리할 수 있는 그리드(grid) 생성이 가능해집니다. 또한 ICM-LoRA는 데이터를 추가로 필요로 하지 않고, 단 283MB의 저장 용량만 차지하여 기존 LoRA에 비해 1%의 저장소 사용량을 기록합니다.

- **Performance Highlights**: ICM-LoRA는 다양한 모델에서 텍스트 및 비주얼 작업을 평가하였으며, CVAE가 여러 작업에 대해 LoRA 매개변수를 성공적으로 생성할 수 있음을 보여줍니다. 기존 방법들과 비교해, 생성된 LoRA 매개변수는 정확도 손실이 적으며, 원본 데이터 세트와 LoRA 가중치에 비해 저장 용량을 크게 줄일 수 있습니다. 결과적으로 본 연구의 접근법은 여러 작업에 대한 LoRA 파라미터 생성을 보다 효율적이고 정확하게 수행할 수 있음을 입증합니다.



### The Imitation Game According To Turing (https://arxiv.org/abs/2501.17629)
- **What's New**: 현재 인공지능(Artificial Intelligence)에 대한 과장된 주장과 사회적 우려가 증대되고 있는 가운데, 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 진정한 능력을 검토합니다. 특히, 이러한 모델들이 튜링 테스트(Turing Test)를 통과할 수 있다는 최근 주장들을 반박하고 있습니다. 연구팀은 GPT-4-Turbo를 사용하여 전통적인 튜링 테스트의 방법론을 철저히 준수하며 새로운 실험을 수행했습니다.

- **Technical Details**: 연구에서는 세 명의 참가자가 포함된 모사 게임(imitation game)을 철저히 시행했습니다. 이는 전통적인 튜링의 지침을 따르며, 특히 CIHG(Computer-Imitates-Human Game)와 MIWG(Man-Imitates-Woman Game)를 사용하여 LLM의 능력을 평가했습니다. 시간 제한이 없는 형태로 진행되었고, 다양한 기준과 과학적 규범을 따랐습니다.

- **Performance Highlights**: 실험 결과, 모든 참가자가 LLM을 정확히 식별했으며, 이는 현재 가장 발전된 LLM 중 하나가 철저한 튜링 테스트를 통과할 수 없다는 것을 의미합니다. 이러한 결과는 최근의 과장된 주장들이 신뢰할 수 없음을 입증하며, '생각하는 기계(thinking machines)'에 대한 사회적 영향을 긍정적이나 부정적으로 평가할 필요가 없음을 보여줍니다.



### VoicePrompter: Robust Zero-Shot Voice Conversion with Voice Prompt and Conditional Flow Matching (https://arxiv.org/abs/2501.17612)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 논문에서는 VoicePrompter라는 혁신적인 제로샷 음성 변환(VC) 시스템을 소개합니다. 이 시스템은 음성 프롬프트를 활용하여 강력한 In-Context Learning 기능을 바탕으로 높은 화자 유사성을 달성합니다. 또한, 잠재적 믹스업(latent mixup)을 포함하여 VC의 강인성을 향상시키고, 조건부 흐름 매칭(Conditional Flow Matching) 구조를 통합하여 오디오 품질을 한 단계 끌어올립니다.

- **Technical Details**: VoicePrompter는 입력 음성을 효과적으로 분리하고 내장하는 음성 분해 인코더와, 분해된 음성 피처 및 음성 프롬프트에 조건화된 DiT 기반 CFM 디코더로 구성됩니다. 음성 정보를 추출하기 위해 MMS 모델의 7번째 레이어를 활용하고, 음성 특성과 무관하게 언어 정보를 분리하기 위해 신호 교란을 적용합니다. 또한 Mixup을 통해 다양한 화자 특성을 결합하여 In-Context Learning 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, VoicePrompter는 기존의 제로샷 VC 시스템에 비해 화자 유사성, 언어 이해도 및 오디오 품질에서 우수한 성과를 보여줍니다. 이는 프롬프트 단계에서 타겟 음성 스타일을 지정할 때 더 나은 화자 유사성을 달성할 수 있게 합니다. VoicePrompter의 성능은 여러 최신 시스템에 비해 현저한 개선을 보여줍니다.



### CSEval: Towards Automated, Multi-Dimensional, and Reference-Free Counterspeech Evaluation using Auto-Calibrated LLMs (https://arxiv.org/abs/2501.17581)
Comments:
          17 pages, 5 figures. arXiv admin note: text overlap with arXiv:2309.13308 by other authors

- **What's New**: 이 연구에서는 자동 반대 연설 생성의 품질을 평가하기 위한 새로운 데이터셋인 CSEval과 네 가지 품질 차원을 평가하는 프레임워크를 제안합니다. 이 프레임워크는 기존의 표면 수준 유사성 측정 방법의 한계를 극복하고, 인간의 판단과 일치하는 더 정교한 평가를 가능하게 합니다. CSEval은 맥락 관련성(contextual relevance), 공격성(aggressiveness), 주장 일관성(argument-coherence), 적합성(suitableness)이라는 네 가지 차원으로 카운터스피치 품질을 평가하도록 설계되었습니다.

- **Technical Details**: CSEval은 자동화된 카운터스피치 생성 평가를 위해 네 가지 주요 품질 차원에 대한 인간의 판단을 포함한 대규모 데이터셋입니다. 또한, 자동 보정된 Chain-of-Thought (CoT) 방식을 활용한 ACE 방안을 제안하여 카운터스피치의 각 품질 차원을 점수화하는 방법을 제시합니다. ACE는 기존의 BLEU, ROUGE, METEOR와 같은 전통적인 지표보다 인간의 평가와 더 높은 상관관계를 보여줍니다.

- **Performance Highlights**: 실험 결과, ACE는 기존의 자동 평가 방법과 LLM 기반 방법들보다 인간의 평가와 더 높은 상관관계를 가지며, 카운터스피치 품질 평가에 있어 큰 진전을 나타냈습니다. 이 연구는 카운터스피치 생성을 다루는 연구들에서 더 개선되고 효과적인 평가 방법을 제안하는 중요한 기초 자료로 작용할 것입니다. CSEval 데이터셋과 소스 코드는 공개되어 연구자들이 쉽게 접근하여 활용할 수 있게 되었습니다.



### Music2Latent2: Audio Compression with Summary Embeddings and Autoregressive Decoding (https://arxiv.org/abs/2501.17578)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 음원 신호를 압축하여 정보가 풍부한 잠재 공간(latent space)으로 표현하는 것은 생성 모델링(generative modeling)과 음악 정보 검색(music information retrieval, MIR) 등의 다양한 작업에 있어 중요합니다. 기존의 오디오 오토인코더는 높은 압축 비율을 달성하기 어려웠으나, 본 논문에서는 Music2Latent2라는 새로운 오토인코더를 소개하여 이러한 문제를 해결합니다. 이 모델은 순서가 없는 잠재 임베딩(unordered latent embeddings) 방식을 도입해, 각 임베딩이 입력 샘플의 독특한 글로벌 특징을 포착할 수 있게 합니다.

- **Technical Details**: Music2Latent2는 오토리그레시브(autoregressive) 일관성 모델(consistency model)을 활용하고, 원인 마스킹(causal masking) 기법을 적용하여 오디오의 임베딩을 디코딩합니다. 이 모델은 잠재 공간 내에서 정보를 더 효율적으로 배분하여 압축 비율을 유지하면서도 좋은 재구성 품질을 제공합니다. 또한, 새로운 두 단계의 디코딩 절차를 통해 오디오의 품질을 더욱 정제할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, Music2Latent2는 같은 압축 비율에서 기존의 연속 오디오 오토인코더와 비교하여 재구성 품질에서 현저한 성능 향상을 보여주었습니다. 또한, MIR과 같은 다운스트림 작업에서도 경쟁력 있는 결과를 달성했습니다. 이러한 성과는 오디오 압축 및 표현 학습 분야에서 새로운 가능성을 열어줍니다.



### Exploring the Potential of Wireless-enabled Multi-Chip AI Accelerators (https://arxiv.org/abs/2501.17567)
Comments:
          Accepted in AccML @ HiPEAC 2025

- **What's New**: 인공지능(AI) 워크로드의 계산 능력에 대한 끝없는 욕망은 빠르고 효율적인 가속기의 개발을 촉진하고 있습니다. 본 논문에서는 기존의 유선 인터커넥트와의 보완으로서 무선 기술의 가능성을 탐구하며, 최적화된 맵핑 및 효율적인 통신을 통해 멀티-칩셋 아키텍처에서의 성능 개선을 보여줍니다.

- **Technical Details**: AI 가속기 아키텍처는 일반적으로 많은 수의 프로세싱 엘리먼트(PEs)로 구성되며, 이는 네트워크 온 칩(NoC)을 통해 고정된 데이터 흐름을 구현합니다. 그러나 멀티-칩셋 아키텍처에서 데이터 이동으로 인한 병목 현상은 성능과 효율성에 큰 영향을 미치는 문제로, 이 문제를 해결하기 위해 무선 링크를 통합하는 방법을 모색하고 있습니다. 무선 기술을 통해 평균적으로 10%, 최대 20%의 성능 향상을 기대할 수 있습니다.

- **Performance Highlights**: 무선 연결의 도입은 AI 가속기의 성능과 에너지 효율성을 크게 개선할 수 있는 잠재력을 가지고 있습니다. 본 연구에서는 GEMINI 프레임워크를 확장하여 다양한 AI 워크로드를 위해 최적화된 맵핑과 무선 인터커넥트를 활용하여, 멀티-칩 아키텍처에서의 평균 속도 향상을 10%로, 최대 20%로 나타냈습니다. 이는 AI 가속기 설계에서 무선 기술의 적용 가능성을 시사합니다.



### An Exceptional Dataset For Rare Pancreatic Tumor Segmentation (https://arxiv.org/abs/2501.17555)
- **What's New**: 이번 논문에서는 췌장 신경내분비 종양(pNETs)을 위한 최초의 데이터셋을 소개합니다. 이 데이터셋은 469명의 환자 데이터를 포함하며, 잘 주석 처리된 CECT(대조강화 컴퓨터 단층 촬영) 이미지를 중심으로 구성되어 있습니다. 또한, UNet 기반 모델을 개선하기 위한 새로운 슬라이스별 가중치 손실 함수가 포함되어 있어 pNET 세분화 성능을 향상시킬 수 있습니다.

- **Technical Details**: 데이터셋은 3D CECT 스캔을 포함하고 있으며, 각 이미지는 512×512의 해상도를 가지며, 보통 150~270개의 횡단면을 커버합니다. 슬라이스 간격은 1mm에서 1.5mm로 설정되어 있으며, 전문가에 의해 주석 처리된 3D 종양 세분화 마스크와 함께 제공됩니다. 또한, 동맥 및 정맥 단계에서의 이미지가 포함된 다단계 데이터셋이 준비되어 있어 퀄리티가 보장됩니다.

- **Performance Highlights**: 우리는 여러 가지 U-Net 기반 의료 이미지 세분화 방법의 성능을 평가했습니다. 이 데이터셋은 pNETs와 같은 희귀 종양에 대한 연구의 기초 자료로 활용될 수 있으며, 향후 보다 정확한 진단 도구 개발에 기여할 것입니다. 이를 통해 pNETs의 조기 진단 및 치료 결과 개선이 기대됩니다.



### Is Conversational XAI All You Need? Human-AI Decision Making With a Conversational XAI Assistan (https://arxiv.org/abs/2501.17546)
Comments:
          conditionally accepted to IUI 2025

- **What's New**: 본 논문에서는 Explainable Artificial Intelligence (XAI) 방법과 대화형 사용자 인터페이스 (Conversational User Interfaces)를 결합하여 사용자의 AI 시스템에 대한 이해도를 향상시키고, 신뢰도를 높일 수 있음을 주장합니다. 대화형 XAI 인터페이스는 기존의 XAI 대시보드보다 사용자들이 AI 시스템을 더 잘 이해하도록 돕는 것으로 나타났으며, 이는 사용자 신뢰도를 높이는 결과를 가져왔습니다. 그러나, 두 인터페이스 모두 사용자들이 AI 시스템에 과도하게 의존한다는 점이 발견되었습니다.

- **Technical Details**: 연구에서는 N=306명의 사용자를 대상으로 대화형 XAI 인터페이스가 AI 시스템에 대한 이해, 신뢰, 의존성에 미치는 영향을 탐구했습니다. 이 과정에서 대화형 인터페이스는 대형 언어 모델 (Large Language Model, LLM) 에이전트를 활용하여 고품질의 텍스트 응답을 제공하였고, 사용자의 과도한 의존성이 드러나는 경향이 있었습니다. 또한 대화형 XAI 인터페이스는 XAI 대시보드와 비교했을 때 사용자 이해 및 신뢰도의 제한된 향상을 보였습니다.

- **Performance Highlights**: 결과적으로, 대화형 XAI 인터페이스는 사용자들이 AI 시스템을 사용하는 과정에서 신뢰와 이해의 감각을 조작하는 설득 기술로 작용할 수 있습니다. 사용자들은 AI의 설명 깊이를 과대 평가하는 경향이 있으며, 이는 AI에 대한 적절한 의존성을 방해하는 요인이 될 수 있습니다. 이 연구는 인간-AI 협업을 효과적으로 지원하기 위한 디자인적 함의를 제공하며, 적절한 의존성을 촉진하기 위해 XAI 방법의 개선을 필요로 합니다.



### RegD: Hierarchical Embeddings via Distances over Geometric Regions (https://arxiv.org/abs/2501.17518)
- **What's New**: 이번 연구에서는 계층 구조 데이터를 저차원 공간에 임베딩하는 새로운 방법인 RegD를 제안합니다. RegD는 두 가지 새로운 거리 메트릭인 depth distance와 boundary distance를 사용하여 계층적인 데이터를 기하학적 영역으로 나타냅니다. 이를 통해 기존 하이퍼볼릭 임베딩 기법의 최적화 문제와 수작업으로 정의된 구조적 제약 조건을 해결할 수 있습니다.

- **Technical Details**: RegD는 계층 구조에 대한 표현력을 유지하면서 유클리드 공간에서 기하학적 영역을 사용하여 데이터를 모델링합니다. depth distance는 고려하는 영역의 '크기'를 포함하여 하이퍼볼릭 공간의 임베딩 표현력을 달성하는 데 도움을 줍니다. boundary distance는 영역 간의 집합 포함 관계를 명확하게 인코딩하여 더 효과적으로 계층 구조를 캡처할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험 결과, RegD는 기존의 최첨단 방법들에 비해 일관된 성능 향상을 보여주었습니다. 본 연구는 RegD가 단순 계층 구조 작업을 넘어 더 넓은 응용 가능성을 가짐을 입증하며, 온톨로지 및 지식 관리 작업에서도 유망한 도구가 될 수 있음을 강조합니다.



### LLM Assistance for Pediatric Depression (https://arxiv.org/abs/2501.17510)
- **What's New**: 이 연구는 소아 전자 건강 기록(EHR)에서 우울증 증상을 추출하기 위해 최신 대형 언어 모델(LLMs)을 이용한 가능성을 평가합니다. 전통적인 스크리닝 방법의 한계를 극복하고, 데이터를 효율적으로 활용하기 위해 제로 샷 분석(zero-shot analysis)을 적용합니다. 연구 결과, LLM이 단어 일치(word match)보다 60% 더 높은 효율성을 보였으며, Flan 모델이 정밀도에서 우수한 성능을 나타냈습니다.

- **Technical Details**: 이 연구는 2009년부터 2022년까지의 약 130만 명의 소아 환자가 포함된 크리닉 노트를 분석하였습니다. PHQ-9 점수를 찾기 위해 ‘PHQ-9 Total Score:’ 패턴을 검색했으며, 전체 노트의 2%가 해당 점수를 포함했습니다. LLMs를 활용하여 클리닉 노트에서 관련 텍스트를 추출하고, 이를 통해 우울증 증상을 인식하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 모델의 성능 비교에서 Flan은 평균 F1 점수 0.65, 정밀도 0.78을 기록하며, 특히 드물게 나타나는 증상을 잘 추출했습니다. Llama 3는 가장 높은 재현율(0.90)을 보였지만 증상을 과도하게 일반화하는 경향이 있었습니다. 이번 연구는 LLM 기반 증상 주석이 머신러닝 알고리즘에서 우울증 사례를 구별하는 데 중요하다는 점을 강조하며, 0.78의 높은 정밀도로 성능 향상을 보여주었습니다.



### Neural Spelling: A Spell-Based BCI System for Language Neural Decoding (https://arxiv.org/abs/2501.17489)
- **What's New**: 이 논문에서는 비침습적 방법으로 뇌 전기 신호를 해독하여 26개의 영문자를 모두 인식하는 새로운 EEG 기반의 뇌-컴퓨터 인터페이스(BCI) 시스템을 제안합니다. 커리큘럼 기반의 신경 철자 프레임워크(Curriculum-based Neural Spelling Framework)를 사용하여, 손글씨와 연관된 신경 신호를 해독하고, 생성적 인공지능(Generative AI)을 적용하여 철자 기반의 신경 언어 해독 작업을 향상시킵니다. 이 접근 방법은 손글씨의 용이성과 EEG 기술의 접근성을 결합하여 높은 정확도로 EEG 패턴을 텍스트로 변환할 수 있습니다.

- **Technical Details**: 이 연구는 CNN 기반 인코더를 사용하여 개인별 문자 전이 패턴을 학습한 후, 커리큘럼 주도 LLM으로 문장 텍스트를 합성하는 하이브리드 접근 방식을 채택하였습니다. 또한, N-gram 모델을 사용하여 연속 문자 발생 확률을 계산하고, 이 과정에서 발생할 수 있는 인지 부하를 줄이기 위해 철자 기반 방법을 활용합니다. 이러한 기술적 접근은 신경 훈련 샘플의 크기에 구애받지 않으며, 새로운 언어 콘텐츠에 빠르게 적응할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 논문에서 제안하는 시스템은 모든 피험자에서 평균 최고 k 정확도를 달성하며, 커리큘럼 기반의 미세 조정 방법이 주목할 만한 성과를 보여주었습니다. 이로 인해 비접촉식 방법으로 손쉽게 문장을 합성할 수 있는 시스템이 마련되었습니다. 사용자 접근성과 효율성을 높이면서 다양한 신체 능력을 가진 개인에게 자연스럽고 포괄적인 소통 방법을 제공하는 데 기여할 것입니다.



### DINT Transformer (https://arxiv.org/abs/2501.17486)
Comments:
          arXiv admin note: text overlap with arXiv:2410.05258 by other authors

- **What's New**: DINT Transformer는 DIFF Transformer의 한계를 극복하기 위해 differential-integral 메커니즘을 도입하여 글로벌 중요성을 평가하고 주의(Attention) 매트릭스에 통합합니다. 이 모델은 글로벌 의존성을 캡처할 수 있는 능력을 향상시키며, 수치적 안정성을 보강하기 위해 행 정규화(row normalization)를 확립하여 자주 사용되는 attentional 맥락의 소음을 줄입니다. DINT Transformer는 긴 시퀀스의 언어 모델링과 핵심 정보 검색과 같은 다양한 실제 응용 프로그램에서 성능을 입증합니다.

- **Technical Details**: DINT Transformer는 한 개 이상의 stacked layer로 구성되어 있으며, 각 layer는 DINT attention 모듈이 적용된 뒤 feedforward network가 뒤따릅니다. 모델 입력은 X0∈ℝN×d_model이고, 각 레이어를 통해 점진적으로 변환되어 최종 출력 XL로 이어집니다. 핵심 혁신은 주의 모듈 내에 적분 메커니즘을 추가하여 글로벌 의존성을 모델링하고 수치적 안정성을 유지하는 것입니다.

- **Performance Highlights**: DINT Transformer의 실험 결과는 모델이 특히 길어진 시퀀스 작업에서 DIFF Transformer와 기존 Transformer 모델을 일관되게 초과하는 성능을 보여줍니다. 이 모델은 핵심 정보 검색과 같은 다운스트림 작업에서 효율성과 확장성을 유지하며 성능을 향상시키는 것으로 나타났습니다. DINT Transformer는 향후 시퀀스 모델링 및 큰 언어 모델 개발의 강력하고 효율적인 토대를 제공합니다.



### DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performanc (https://arxiv.org/abs/2501.17479)
- **What's New**: 본 논문에서는 여러 LLM의 상호 보완적인 강점을 활용하여 성능을 향상시키는 새로운 앙상블 기법인 Diverse Fingerprint Ensemble (DFPE)를 제안합니다. 이 방법은 피어리와 같은 응답 패턴 기반 모델 군집화, 주제별로 하위 성능 모델 필터링, 최종 모델 가중치 조정을 포함합니다. 이를 통해 LLM의 견고성과 일반화 능력을 증가시키며, 다양한 언어 이해 작업에 효과적으로 대응할 수 있는 방법을 보여줍니다.

- **Technical Details**: Diverse Fingerprint Ensemble (DFPE) 방법은 모델 응답의 'fingerprint' 패턴을 군집화하여 다양한 문제 해결 전략을 유지하면서 중복성을 방지합니다. 모델은 주제 기반 검증 정확도에 따라 적절한 가중치를 부여받고, 주제별 경험치를 고려하여 조정된 가중치를 사용하여 앙상블을 구성합니다. 이 방식은 다양한 질문 유형을 처리할 수 있도록 모델의 적합성과 다양성을 동시에 강조합니다.

- **Performance Highlights**: MMLU 벤치마크 실험에서 DFPE는 최고의 단일 모델에 비해 전체 정확도를 3% 향상시켰으며, 학문별 정확도는 5% 개선되었습니다. 이러한 결과는 LLM의 선택 및 성과 기반 가중치 조정이 복잡한 다면적 언어 이해 작업에서 성능을 유의미하게 개선할 수 있음을 입증합니다. 최종적으로, DFPE는 주제에 맞춘 다각적이고 효과적인 앙상블을 통해 MMLU와 같은 복잡한 작업에서 우수한 성과를 보입니다.



### Towards Making Flowchart Images Machine Interpretab (https://arxiv.org/abs/2501.17441)
Comments:
          Published at: ICDAR 2023, Project Page: this https URL

- **What's New**: 이 논문은 플로우차트(flowchart) 이미지를 머신이 해석할 수 있는 실행 가능한 파이썬 코드로 변환하는 새로운 접근 방식을 제안합니다. 이를 위해, 최근 자연어를 코드로 생성하는 연구에서 영감을 받아 FloCo-T5라는 트랜스포머 기반의 프레임워크를 구축하여 플로우차트 이미지를 자동으로 변환합니다. 또한, 11,884개의 플로우차트 이미지와 해당하는 파이썬 코드로 구성된 FloCo 데이터셋을 소개하여, 향후 연구에 기여할 기반을 마련하였습니다.

- **Technical Details**: FloCo-T5는 플로우차트 이미지를 시퀀스 인코딩으로 변환하고, 이를 통해 프로그래밍 언어의 구조와 의미를 이해하도록 사전 훈련(pre-training)되었습니다. 이 모델은 특정한 작업을 위한 목표로 코드 샘플을 대규모로 증강하여 학습하였으며, 최종적으로 시퀀스-투-시퀀스(sequence-to-sequence) 생성 문제로 모델을 조정(fine-tuning)하여 사용합니다. 본 논문에서 실시한 실험들은 다양한 코드 생성 메트릭에서 FloCo-T5가 기존 기준 모델들보다 우수한 성능을 보임을 증명하였습니다.

- **Performance Highlights**: 실험 결과, FloCo-T5는 BLEU, CodeBLEU, 정확도 점수에서 각각 67.4, 75.7, 20.0를 기록하며, 기존 모델들보다 뛰어난 성능을 나타냈습니다. 이는 우리가 제안한 사전 훈련 목표와 데이터 증강 전략의 효과를 분명히 입증합니다. 또한, 본 연구에서 제안한 모델은 손으로 그린 플로우차트 이미지에도 적용 가능하다는 점에서, 다양한 응용 가능성을 보이고 있습니다.



### Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation (https://arxiv.org/abs/2501.17433)
- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLMs)은 해로운 샘플에 대한 미세 조정(fine-tuning) 공격에 취약해지고, 이는 모델의 안전성 정렬(safety alignment) 능력을 손상시킬 수 있습니다. 이러한 공격을 방지하기 위해 일반적으로 가드레일(guardrail)로 해로운 샘플을 필터링하는 방법이 사용되지만, 본 논문에서는 이러한 가드레일 의존이 신뢰할 수 없음을 보여줍니다. 새로운 공격 방법인 Virus를 통해, 해로운 데이터의 미세 조정이 가능한 점을 입증하며, 이를 통해 기존의 필터링 방법을 우회하는 데 성공했습니다.

- **Technical Details**: 본 연구에서는 Virus라는 새로운 데이터 최적화(data optimization) 방법을 제안합니다. 이 방법은 두 가지 목표를 달성하기 위해 설계되었습니다. 첫 번째는 가드레일을 성공적으로 우회할 수 있도록 가드레일에 대한 공격 손실(jailbreak loss)을 최소화하는 것이고, 두 번째는 해로운 그래디언트(harmful gradient)을 유사하게 만들어서 피해 모델의 안전성 정렬이 무너지도록 하는 것입니다. 실험 결과 Virus는 가드레일을 효과적으로 우회하며, 100%의 누출(leakage) 비율에 도달할 수 있음을 보였습니다.

- **Performance Highlights**: Virus에 의해 최적화된 데이터는 피해 LLM의 안전성 정렬을 크게 저하시켜 21.8%의 유해 점수(harmful score) 증가를 가져왔습니다. 이를 통해 가드레일의 한계가 드러났으며, 본 연구는 가드레일이 해로운 미세 조정 공격을 방지하는 데 있어 믿을 수 없는 존재라는 것을 강조합니다. 연구의 마지막 메시지는, 가드레일 의존은 해로운 미세 조정 공격에 대한 효과적인 솔루션이 아니라는 점입니다.



### Algorithmic Segmentation and Behavioral Profiling for Ransomware Detection Using Temporal-Correlation Graphs (https://arxiv.org/abs/2501.17429)
- **What's New**: 이번 연구에서는 기존의 탐지 방법론의 한계를 극복하고, 현대의 사이버 위협에 대응하기 위해 Temporal-Correlation Graphs라는 새로운 프레임워크를 제안하였다. 이 프레임워크는 악의적인 작용의 복잡한 관계와 시간적 패턴을 모델링하여 행동 이상을 동적으로 포착하는 기능을 제공한다. 실시간 환경에서의 악성 활동과 정상 활동을 구분할 수 있는 견고한 기제를 마련함으로써 기업의 사이버 보안을 개선할 수 있는 가능성을 제시한다.

- **Technical Details**: Temporal-Correlation Graphs는 시스템 리소스, 네트워크 및 파일 시스템과 상호작용할 때 랜섬웨어가 나타내는 특정한 시간적 행동을 분석하여 악성 활동을 탐지한다. 이를 위해 그래프 기반 접근 방식을 채택하여 새로운 랜섬웨어 변종에 대해 종속성 및 시간적 데이터를 활용하여 동적으로 적응한다. 이러한 방법론은 기존의 정적 모델이나 미리 존재하는 규칙에 의존하지 않고, 그래프의 토폴로지적 특성을 통해 비정상적인 행동을 식별할 수 있도록 돕는다.

- **Performance Highlights**: 해당 연구에서 제안한 프레임워크는 다양한 랜섬웨어 계열에 대해 실험을 통해 높은 정밀도(precision), 재현율(recall) 및 전체 탐지 정확도를 유지하는 효과를 보였다. 기존의 서명 기반(signature-based) 및 휴리스틱 방법론에 비해 더 나은 성능을 발휘할 뿐만 아니라, 동적으로 코드를 변경하는 폴리모픽 랜섬웨어와 이전에 보지 못한 변종을 다루는 데 뛰어난 능력을 보여준다. 결과적으로, 이 연구는 랜섬웨어 탐지 및 대응 전략의 발전에 기여하며, 미래의 사이버 보안 혁신을 위한 기초를 마련하였다.



### Actions Speak Louder than Words: Agent Decisions Reveal Implicit Biases in Language Models (https://arxiv.org/abs/2501.17420)
- **What's New**: 이 연구는 대형 언어 모델(LLM)에서 발생하는 잠재적 편향(implicit biases)을 발견하기 위한 새로운 방법론을 제안합니다. 해당 기술은 소시오데모그래픽 특성에 따라 만들어진 역할 성격(perseon)을 가진 언어 에이전트(language agents)가 다양한 의사결정 시나리오에서 수행하는 행동을 분석합니다. 이를 통해 모델이 직간접적으로 드러내는 편향을 명확히 구별할 수 있게 됩니다.

- **Technical Details**: 연구는 언어 에이전트의 행동을 ML모델의 반응과 대조하여 편향을 평가하는 두 단계의 과정을 통해 진행됩니다. 첫 번째 단계에서는 LLM을 사용하여 특정 소시오데모그래픽 속성에 기반하여 캐릭터를 생성하고, 두 번째 단계에서는 이러한 캐릭터들이 주어진 의사결정 시나리오에 대해 반응하도록 합니다. 이 접근법은 다양한 소시오데모그래픽 카테고리와 시나리오에서의 편향을 보다 체계적으로 조사할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들이 거의 모든 시뮬레이션에서 상당한 소시오데모그래픽 편차를 보였으며, 보다 진보한 모델들이 명시적 편향(explicit biases)은 줄여도 여전히 더 큰 잠재적 편향을 보였습니다. 본 연구는 또한 발견된 편향이 실제 세계에서 관찰되는 불평등 정도와 방향이 일치하지만, 그 강도는 확대된 것으로 나타났습니다. 이는 LLM이 시뮬레이션에서의 행동과 관련된 내재적 편향을 발견하는 데 이 기술의 유용성을 시사합니다.



### Reqo: A Robust and Explainable Query Optimization Cost Mod (https://arxiv.org/abs/2501.17414)
- **What's New**: 최근 쿼리 최적화에서 기계 학습(ML)을 사용하는 것에 대한 관심이 증가하고 있습니다. 본 논문에서는 Bidirectional Graph Neural Networks (Bi-GNN)와 Gated Recurrent Units (GRUs)를 기반으로 한 트리 모델 아키텍처를 제안하여 비용 추정의 정확성을 높이고자 합니다. 또한, 불확실성을 계량화하여 보다 견고한 성능을 달성하는 새로운 학습-순위 비용 모델을 구현했습니다.

- **Technical Details**: 쿼리 실행 계획은 일반적으로 트리 구조로 표현되며 각 노드는 데이터를 접근하거나 조인하는 연산자의 정보를 포함합니다. 전통적인 쿼리 최적화기는 히스토그램 기반의 통계적 방법을 사용하여 비용 모델을 구성하지만, 이러한 방법은 복잡한 특성을 포착하는 데 한계가 있어 오류를 초래할 수 있습니다. 본 연구는 ML 기법을 통해 이러한 문제를 해결하고자 하며, 쿼리 계획 내 특정 하위 그래프의 기여를 설명할 수 있는 새로운 설명 가능성 기술도 제안합니다.

- **Performance Highlights**: 본 논문에서 제안하는 Reqo는 비용 추정의 정확성, 견고성 및 설명 가능성을 개선하여 기존의 최첨단 방법들을 능가하는 성능을 보여줍니다. 새로운 비용 모델은 쿼리 최적화 과정에서 불확실성을 효과적으로 계량화하고 이를 비용 추정에 통합하여 얻은 결과입니다. 이러한 혁신을 통해 학습 기반 비용 모델의 신뢰성 및 투명성을 향상시킬 수 있습니다.



### A Genetic Algorithm-Based Approach for Automated Optimization of Kolmogorov-Arnold Networks in Classification Tasks (https://arxiv.org/abs/2501.17411)
- **What's New**: GA-KAN은 Kolmogorov-Arnold Networks (KANs)의 최적화를 자동으로 수행하는 유전자 알고리즘 기반 접근 방식을 제안합니다. 이 논문은 KANs의 구조 최적화에 있어 진화 계산(evolutionary computation)을 최초로 탐구하며, 기존 KAN을 수동으로 조정할 필요 없이 토이 데이터셋에서 최적 결과를 달성하였습니다. 또한, GA-KAN은 다섯 개의 분류 데이터셋에서 전통적인 방법에 비해 뛰어난 성능을 보여줍니다.

- **Technical Details**: GA-KAN은 KAN의 아키텍처와 그리드 값을 최적화하기 위해 고유한 인코딩 및 디코딩 전략을 사용합니다. 신경망의 연결 상태와 깊이를 벡터로 인코딩하고, 그리드 값과 구조를 최적화하기 위해 유전자 알고리즘(GA)을 적용하였습니다. 이는 KAN의 매개변수를 줄이고 해석 가능성을 높이는 데 기여합니다.

- **Performance Highlights**: GA-KAN은 여러 분류 작업을 통해 전통적인 기계 학습 모델 및 기존의 KAN에 비해 우수한 정확도를 보여주었습니다. 또한 GA-KAN은 실험에서 모델의 해석 가능성을 향상시키고 매개변수 수를 대폭 줄이는 성과를 올렸습니다. 이러한 성과들은 KAN 구조의 효율성을 증대시키고, 다양한 데이터셋에 대한 적응성을 높이는 데 중요합니다.



### General Scene Adaptation for Vision-and-Language Navigation (https://arxiv.org/abs/2501.17403)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 Vision-and-Language Navigation (VLN) 작업을 개선하기 위한 새로운 접근법인 GSA-VLN을 제안합니다. 기존의 VLN 작업들이 환경 기본 경험을 고려하지 않는 것에 반해, GSA-VLN은 특정 환경에서 내비게이션 지시를 실행하며 그에 따라 지속적으로 적응할 것을 요구합니다. 이를 통해, 에이전트들이 실제 환경에 맞춰 지속적으로 개선될 수 있는 가능성을 엿볼 수 있습니다.

- **Technical Details**: 주요 이점으로는 GSA-R2R 데이터셋을 통해 기존 VLN 데이터셋에 비해 환경 및 지시문 다양성을 획기적으로 확장한 점입니다. 또한, 대규모 Vision-Language Models (VLMs)와 Large Language Models (LLMs)을 활용한 세 단계의 지시 제작 프로세스를 개발하여, 각 환경에서 600개의 지시문을 다양하게 생성합니다. 이 연구는 에이전트의 성능을 ID 및 OOD 문맥 모두에서 평가할 수 있는 환경을 제공합니다.

- **Performance Highlights**: 실험 결과, GR-DUET라는 새로운 방법은 각 환경의 전반적인 위상 그래프를 지속적으로 업데이트하여 훈련 및 평가에서의 과거 정보를 보존합니다. 이러한 접근은 기본 DUET 모델에 비해 8%의 성공률 향상을 이끌어내며, 모든 GSA-R2R 분할에서 최신 결과를 생성하는 데 성공했습니다.



### MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs (https://arxiv.org/abs/2501.17399)
- **What's New**: MultiChallenge는 다중 발화 대화에서 대형 언어 모델(LLMs)의 성능을 평가하는 새로운 기준을 제시합니다. 이 벤치마크는 현재 대형 언어 모델들이 직면한 네 가지 도전 과제를 식별하며, 이는 인간과의 상호작용에서 일반적이면서도 현실적인 문제들입니다. 다중 발화 대화에 대한 포괄적인 평가 프레임워크가 부족했던 것을 해결하기 위해 설계된 MultiChallenge는 LLM이 사용자의 요구에 적절히 응답할 수 있는 능력을 평가하는 데 중점을 둡니다.

- **Technical Details**: MultiChallenge는 최대 10턴의 대화 히스토리를 기반으로 하며, 각각의 테스트 예시는 대화의 최종 사용자 발화에 대한 요구 사항/질문으로 끝납니다. 네 가지 도전 과제는 1) 지침 유지, 2) 사용자 정보의 추론 기억, 3) 신뢰할 수 있는 버전 편집, 4) 자기 일관성으로 정의됩니다. 이 과제들은 모두 LLM이 지침을 따르고, 적절하게 문맥을 할당하며, 문맥 내 추론을 수행하는 능력을 요구합니다.

- **Performance Highlights**: 기존의 다중 발화 평가 벤치마크에서는 거의 완벽한 점수를 기록했음에도 불구하고, 현재의 최첨단 LLM들은 MultiChallenge에서 50% 미만의 정확도를 기록했습니다. 이 중에서 Claude 3.5 Sonnet (2024년 6월)은 평균 41.4%의 정확도로 가장 우수한 성과를 보여주었습니다. 연구에서는 인간 평가자와 자동 평가 간의 높은 상관관계를 보여주어 빠르고 정확한 다중 발화 평가의 잠재력을 입증했습니다.



### Learning Free Token Reduction for Multi-Modal LLM (https://arxiv.org/abs/2501.17391)
- **What's New**: 본 논문에서는 비디오 기반 멀티모달 언어 모델(Multimodal Large Language Models, MLLMs)의 효율성을 높이기 위해 시각적 프롬프트를 압축하는 새로운 접근 방식을 제안합니다. 기존 모델 압축 기술은 주로 아키텍처 개선이나 시각적 토큰 수 줄이기에 집중했으나, 시각 데이터의 고유한 공간 및 시간적 특성을 고려하지 않았습니다. 반면, 제안된 방법은 시각적 토큰의 시간적 및 공간적 차원 모두에서 압축을 수행하여 성능을 유지하면서도 계산 비용을 절감할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 방법은 학습이 필요 없는 플러그 앤 플레이(plug-and-play) 압축 파이프라인으로, 대부분의 MLLM 프레임워크에 통합할 수 있습니다. 우리는 시간 차원에서 인접한 토큰을 병합하고 공간 차원에서 덜 유의미한 토큰을 제거하여 시각적 표현의 중복성과 희소성을 활용합니다. 이를 통해 모델의 추론 능력을 향상시키고 실험적으로 비디오-QA(Question Answering) 작업에서의 효율성을 개선했습니다.

- **Performance Highlights**: 실험 결과, 제안된 압축 방법은 비디오-LLMs의 성능을 유지하면서도 토큰 수를 효과적으로 줄일 수 있음을 보여주었습니다. 이로써 비디오 기반 멀티모달 태스크에서 significant한 개선을 나타내며, 전체적인 추론 속도와 효율이 크게 향상되었습니다. 이는 다양한 MLLM 아키텍처와의 호환성을 보장하며, 동종의 기술적 과제 해결에 기여할 것으로 기대됩니다.



### Context-Aware Semantic Recomposition Mechanism for Large Language Models (https://arxiv.org/abs/2501.17386)
- **What's New**: 최근 자연어 처리 분야에서, Context-Aware Semantic Recomposition Mechanism (CASRM)이 도입되어 언어 생성 모델의 의미적 일관성과 맥락 적응성을 높이는 새로운 프레임워크로 주목받고 있습니다. CASRM은 동적으로 생성되는 컨텍스트 벡터와 주의 조절 레이어를 통합하여 대규모 텍스트 생성 작업의 일관성 문제와 오류 전파를 해결하고자 합니다. 다양한 실험 평가를 통해 기술적, 대화형, 서사적 텍스트를 포함한 여러 도메인에서 의미적 일관성이 크게 향상된 결과를 보여주었습니다.

- **Technical Details**: CASRM의 메커니즘은 모델 내에서 의미적 표현을 동적으로 조정하여 맥락에 대한 복잡한 이해를 가능하게 합니다. 컨텍스트 벡터가 주의 메커니즘에 통합되어 입력 데이터의 정제된 처리를 지원하고, 이론적 기반은 맥락 종속 의미 해석을 바탕으로 합니다. 최종적으로 CASRM 모듈은 기존 배치 모델에 보조 처리 유닛으로 포함되며, 입력 시퀀스를 처리하여 동적으로 발전하는 컨텍스트 벡터를 추출합니다. 이 비율로 조정된 주의 가중치는 토큰 간의 중요 분포를 개선하여 다양한 맥락에서의 의미적 일관성을 높입니다.

- **Performance Highlights**: 실험 결과 CASRM은 언어 생성 성능을 극적으로 향상시켰으며, 특히 대화 연속성과 다단계 텍스트 합성에서 오류 전파를 성공적으로 완화했습니다. 또한, 미지의 도메인과 모호한 입력에 대한 적응 능력이 입증되어 시스템의 강인성을 강조합니다. CASRM은 LM 성능의 현재 한계를 극복할 수 있는 유망한 방향성을 제공하며, 더 신뢰할 수 있고 맥락을 인식하는 언어 모델의 발전에 기여할 수 있을 것으로 기대됩니다.



### A Dual-Agent Adversarial Framework for Robust Generalization in Deep Reinforcement Learning (https://arxiv.org/abs/2501.17384)
- **What's New**: 이 논문은 최근 강력한 신경망 능력을 활용하여 강화 학습(Reinforcement Learning, RL)에서 일반화 문제를 해결하기 위한 새로운 이중 에이전트 적대적 정책 학습 프레임워크를 제안합니다. 이 프레임워크는 두 에이전트 간의 상호작용을 통해 에이전트가 기본 의미를 자발적으로 학습할 수 있도록 하여 사전 지식 없이도 강력한 일반화 성능을 도모합니다. 특히, 이 방법은 고차원 관찰로부터 무관한 특징을 처리할 수 있는 일반화 가능한 정책을 학습하도록 돕습니다.

- **Technical Details**: 이 프레임워크는 두 동질의 에이전트 간의 게임 프로세스를 이용하며, 각 에이전트는 상대의 정책에 대한 방해의 영향을 극대화하려고 시도하면서, 같은 상태에 대한 표현의 차이를 생성합니다. 또한, 이 과정에서 각 에이전트는 자신의 정책의 안정성을 유지합니다. 이 방법은 Proximal Policy Optimization (PPO)와 같은 기존 정책 학습 알고리즘과 잘 통합되며, 최소한의 추가 하이퍼파라미터만 필요로 합니다.

- **Performance Highlights**: Procgen 벤치마크를 통해 실험 결과, 적대적 과정이 두 에이전트의 일반화 성능을 크게 향상시킨다는 것을 입증했습니다. 특히, 어려운 수준 환경에서 RL 에이전트가 기반 방법들보다 상당한 성과를 달성하여, 심층 강화 학습의 일반화 능력에 있어 중요한 진전을 이룩했습니다.



### ASAP: Learning Generalizable Online Bin Packing via Adaptive Selection After Pruning (https://arxiv.org/abs/2501.17377)
- **What's New**: 최근 딥 강화 학습(Deep Reinforcement Learning, DRL)이 온라인 3D 물품 포장 문제(3D-Bin Packing Problem, 3D-BPP) 해결에 있어 유망한 결과를 보였습니다. 하지만 이러한 DRL 기반 정책이 새로운 인스턴스에서 성능 저하를 겪는 문제를 지적하였습니다. 일반화(generalization) 외에도 빠른 적응(adaptation)을 고려하여, 정책의 의사결정을 프루닝(pruning)과 선택(selection) 두 단계로 나누는 'Adaptive Selection After Pruning (ASAP)' 을 제안합니다.

- **Technical Details**: ASAP은 프루닝 정책과 선택 정책으로 나누어진 의사결정 구조를 가지고 있으며, 첫 번째 단계에서 프루닝 정책이 본질적으로 나쁜 행동을 제거합니다. 두 번째 단계에서 선택 정책은 남은 행동 중에서 가장 가치 있는 행동을 선택합니다. 메타 학습(meta-learning) 단계와 테스트 데이터 분포에 맞춘 파인 튜닝(finetuning) 단계를 포함한 두 단계 훈련 방식을 통해 DRL 기반 해결책의 일반화 가능성을 향상시키고 신속한 적응을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ASAP은 분포 내(in-distribution) 및 분포 외(out-of-distribution) 인스턴스에서 우수한 일반화 및 적응 능력을 보여줍니다. 또한, 기존 베이스라인 방법들과 비교했을 때 일반화 성능과 적응 개선에서 더 높은 성능을 기록하였습니다. 이러한 결과는 ASAP의 두 단계 구조가 실제 상황에서의 성능 저하를 해결하는 데 기여함을 시사합니다.



### A Geometric Perspective for High-Dimensional Multiplex Graphs (https://arxiv.org/abs/2501.17374)
Comments:
          Published in Proceedings of the ACM Conference on Information and Knowledge Management (CIKM) 2024, DOI: https://doi.org/10.1145/3627673.3679541

- **What's New**: 이번 연구는 고차원 멀티플렉스 그래프의 임베딩 문제를 기하학적인 시각에서 분석하고, 이를 통해 발생하는 기하학적 왜곡을 이해하는 것을 목표로 합니다. 특히, 기하적으로 곡률이 높은 다양하게 정의된 내포된 공간에서의 노드 표현을 특성화하여, 임베딩의 질을 향상시키기 위해 계층적 차원 임베딩과 하이퍼볼릭 그래프 신경망을 활용한 새로운 방법인 HYPER-MGE를 제안합니다.

- **Technical Details**: HYPER-MGE는 Riemannian 다양체에서 노드 표현을 계층적으로 추출하고, 멀티플렉스 그래프의 더 효과적인 잠재 차원을 점진적으로 학습하는 방식으로 설계되었습니다. 이 접근법은 하이퍼볼릭 공간에 대해 계층적 집합을 수행하여 고차원 그래프를 보다 정확하게 나타내는 개선된 표현을 제공합니다.

- **Performance Highlights**: 실제 고차원 멀티플렉스 그래프에 대한 실험 결과, 제안된 방법은 기존 최첨단 방법에 비해 훨씬 적은 기하학적 왜곡을 발생시키며, 다운스트림 작업에서의 성능이 크게 향상되는 것으로 나타났습니다. HYPER-MGE는 멀티플렉스 그래프의 복잡한 구조를 포착하고, 기하학적 왜곡을 줄이는 데 있어 효과적인 접근법으로 입증되었습니다.



### Forecasting S&P 500 Using LSTM Models (https://arxiv.org/abs/2501.17366)
- **What's New**: 이번 연구는 전통적인 시간 영역 모델인 ARIMA와 최신 기계 학습 기법 중 하나인 LSTM을 비교하여 S&P 500 지수를 예측하는 방법을 제시합니다. 주식 시장 데이터의 비선형적이고 복잡한 특성을 다루기 위해, LSTM 모델이 전통적인 방법보다 우수한 성능을 나타낸다는 것을 보였습니다. 특히, 추가적인 피처 없이도 LSTM은 MAE 175.9, RMSE 207.34, 96.41%의 정확도로 예측 성능을 극대화했습니다.

- **Technical Details**: ARIMA(Autoregressive Integrated Moving Average) 모델은 주식 가격의 단기적 경향을 포착하는 데 유용하지만, 비선형 의존성을 처리하는 데 한계가 있습니다. 한편, LSTM(Long Short-Term Memory) 네트워크는 긴 시퀀스 데이터의 장기적 의존성을 학습할 수 있는 구조적 이점을 가지고 있습니다. LSTM은 정보의 입력, 삭제, 출력 게이트로 구성되어 있어 복잡한 금융 시간 시계열을 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 전통적인 ARIMA 모델은 MAE 462.1 및 RMSE 614으로 합리적인 성능을 보였으나, LSTM은 MAE 369.32, RMSE 412.84 및 92.46%의 정확도로 더 나은 성능을 보였습니다. 특히, LSTM 모델이 추가적인 피처 없이도 높은 성과를 달성하여 금융 데이터 처리에서의 효율성을 입증했습니다. 이러한 결과는 전통적인 방법보다 심층 학습 모델이 변동성이 큰 금융 데이터를 처리하는 데 더 큰 잠재력을 가지고 있음을 확인시켜 줍니다.



### The M-factor: A Novel Metric for Evaluating Neural Architecture Search in Resource-Constrained Environments (https://arxiv.org/abs/2501.17361)
- **What's New**: 본 논문은 Neural Architecture Search (NAS) 분야에서 모델 정확도와 크기를 결합한 새로운 평가 지표인 M-factor를 제안합니다. 기존의 NAS 기법은 주로 정확도를 극대화하는 데 집중했으며, 이로 인해 자원이 제한된 환경에서의 사용에 한계가 있었습니다. M-factor는 이러한 한계를 극복하고 NAS가 성능과 효율성의 균형 잡힌 아키텍처를 찾도록 돕는 데 초점을 맞춥니다.

- **Technical Details**: 연구는 Policy-Based Reinforcement Learning, Regularised Evolution, Tree-structured Parzen Estimator (TPE), Multi-trial Random Search의 네 가지 NAS 기법을 비교합니다. CIFAR-10 데이터셋에서 ResNet 구성을 통해 19,683개의 구성을 탐색하며, 각 기법의 M-factor 값을 평가합니다. 실험 결과, Policy-Based Reinforcement Learning과 Regularised Evolution이 가장 높은 M-factor 값을 기록했고, 이로써 각 기법의 최적화 역학과 정확성과 모델 크기 간의 trade-off를 분석합니다.

- **Performance Highlights**: 실험의 결과는 Policy-Based Reinforcement Learning이 39회의 시도 후 성능 변화가 있었고, Regularised Evolution은 20회의 시도로 최적화를 이룬 것을 보여줍니다. 또한, 랜덤 서치가 M-factor를 통해 평가 시 더 복잡한 알고리즘과 유사한 성능을 보이는 경향도 나타났습니다. 이러한 결과들은 M-factor가 기존 지표의 한계를 해결하고, 성능과 효율성을 모두 요구하는 상황에서 전략을 선택하는 데 유용한 통찰을 제공함을 보여줍니다.



### On the Coexistence and Ensembling of Watermarks (https://arxiv.org/abs/2501.17356)
- **What's New**: 이 논문에서는 심층 이미지 워터마킹(deep image watermarking) 방법의 공존을 최초로 연구했습니다. 연구 결과, 다양한 오픈 소스 워터마크가 이미지 품질(image quality)과 디코딩 강인성(decoding robustness)에 미치는 영향이 최소한으로 나타나면서 함께 공존할 수 있음을 발견했습니다. 이는 또한 서로 다른 워터마킹 기법을 조합하여 성능을 향상시킬 수 있는 가능성을 열어줍니다.

- **Technical Details**: 워터마킹은 이미지의 픽셀 값을 왜곡하여 비 가시적인 정보를 인코딩하는 방식으로 이루어지며, 이때 고려해야 할 요소들은 용량(capacity), 이미지 품질(image quality), 정확도(accuracy), 강인성(robustness)입니다. 본 연구에서는 여러 심층 학습 기반의 워터마킹이 서로 간섭 없이 공존할 수 있는지를 분석하였으며, 이는 이미지 품질과 디코딩 강인성을 단지 경미하게 감소시키면서 여전히 공존이 가능함을 보여주었습니다.

- **Performance Highlights**: 논문에서는 두 개의 서로 다른 워터마크를 사용하여 미디어의 데이터 용량을 증가시키고, 기존의 방법들이 재훈련 없이도 정확도, 강인성 및 이미지 품질 간의 트레이드오프를 조정할 수 있는 가능성을 제시합니다. 실험 결과, 이미지 워터마킹 기법이 기대 이상으로 높은 수준의 공존 가능성을 보여주었으며, 이는 효과적으로 워터마킹 모델 및 기존 기술 간의 성능 향상을 위한 기초가 될 수 있습니다.



### Deep-and-Wide Learning: Enhancing Data-Driven Inference via Synergistic Learning of Inter- and Intra-Data Representations (https://arxiv.org/abs/2501.17347)
Comments:
          16 pages, 8 figures

- **What's New**: 이번 연구에서는 deep-and-wide learning (DWL)이라는 새로운 학습 방식을 소개하고 있습니다. 이 방법은 개별 입력 데이터 내의 특징뿐만 아니라, 데이터 간의 특징을 체계적으로 파악할 수 있는 능력을 가지고 있습니다. 또한, 본 연구는 DWL을 실현하기 위한 dual-interactive-channel network (D-Net)을 제안합니다.

- **Technical Details**: D-Net은 Bayesian formulation을 활용하여 low-dimensional (LD) inter-data 특징 추출을 수행하며, 이는 기존의 high-dimensional (HD) 데이터셋 표현과 상호작용하여 더 나은 계산 효율성과 추론을 가능하게 합니다. DWL은 각종 분류 및 회귀 작업에 적용되어 다양한 분야의 데이터에서 그 유효성을 입증하고 있습니다.

- **Performance Highlights**: 연구 결과, DWL은 제한된 교육 데이터로도 기존의 state-of-the-art DNN보다 정확도가 현저히 높음을 보여줍니다. 또한, DWL은 계산 효율성을 몇 배 향상시켜 데이터 기반 학습 기술, 특히 대규모 기초 모델에 중대한 변화를 가져올 것으로 기대됩니다.



### Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines (https://arxiv.org/abs/2501.17343)
- **What's New**: 이번 연구는 최신 3D 의료 분할 모델에 대해 실제 8비트 양자화(quantization) 프레임워크를 소개합니다. 특히, TensorRT 엔진을 통한 진정한 LOW-bit quantization을 적용하여 모델 크기 및 추론 시간을 획기적으로 줄이며, 실제 GPU 환경에서 효율성을 극대화합니다. 우리 연구는 의료 이미지 처리에서의 너비 제한을 고려하여, 실제 3D INT8 PTQ를 구현함으로써 가시적인 성과를 도출했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 TensorRT를 이용하여 레이블이 없는 식별 데이터셋을 통해 모델의 가짜 양자화(faux quantization)를 수행합니다. 두 번째 단계에서는 가짜 양자화된 모델을 실제 양자화된 TensorRT 엔진으로 변환하여, 낮은 정밀도의 계산을 가능하게 하는 하드웨어 최적화를 적용합니다. 이를 통해, 모델 크기와 추론 속도가 실제로 감소합니다.

- **Performance Highlights**: 우리가 개발한 PTQ 프레임워크는 기존의 가짜 양자화 방식과 비교하여 모델 크기가 2.42배에서 3.85배, 추론 지연시간이 2.05배에서 2.66배 감소했습니다. 이로 인해 대규모 3D 의료 이미지 작업에서도 높은 분할 정확도를 유지하면서 리소스 사용량을 줄이는 효과를 확인했습니다. 우리의 연구는 자원이 제한된 임상 환경에서도 높은 성능을 보장하는 효율적인 모델 배포의 길을 제시합니다.



### Inferring from Logits: Exploring Best Practices for Decoding-Free Generative Candidate Selection (https://arxiv.org/abs/2501.17338)
- **What's New**: 이 논문에서는 기존의 decoding-free candidate selection 방법들의 효과를 포괄적으로 평가합니다. 특히, 여러 가지 서로 다른 작업에 대해 이들 방법을 적용하여 성능을 분석하였습니다. 주요 연구는 여러 선택지들 중에서 task-level output을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 다양한 foundation LMs (Language Models) 아키텍처와 크기를 포함한 모델을 사용하여 후보 선택 방법의 성능을 평가했습니다. 이 과정에서 5개의 multiple-choice QA 작업과 4개의 임상 결정 작업을 포함한 다양한 데이터세트를 활용했습니다. 특히, 10,000개 이상의 옵션을 가진 대규모 후보 풀을 가진 작업에 대한 실험이 포함되었습니다.

- **Performance Highlights**: 실험 결과, decoding-free candidate selection 방법이 기존의 token-level decoding 방식에 비해 더 나은 성능을 보여주었다고 보고합니다. 이러한 방법들은 gradients를 효율적으로 활용할 수 있게 하며, 시간 소모를 줄이는 데 기여합니다. 논문은 이러한 결과가 향후 모델 설계에 중요한 통찰력을 제공함을 강조합니다.



### Anomaly Detection in Cooperative Vehicle Perception Systems under Imperfect Communication (https://arxiv.org/abs/2501.17329)
Comments:
          10 pages

- **What's New**: 이 연구에서는 자율 주행의 안전성을 보장하기 위해 이상 탐지(anomaly detection)의 중요성을 강조합니다. 특히, Cooperative Perception(협력적 지각)을 활용해 인근 차량 간 정보를 공유함으로써 복잡한 교통 시나리오에서 이상 행동을 보다 정확히 식별할 수 있는 접근 방식을 제안합니다. 새로운 협력 지각 기반 이상 탐지 프레임워크(CPAD)는 통신이 중단되는 상황에서도 일관된 성능을 유지하도록 설계되었습니다.

- **Technical Details**: CPAD는 통신 결함에도 강한 강인성을 제공하며, 경량의 스페이셔 템포럴(copora-temporal) 모델링을 통해 차량 경로에서의 행동 이상을 탐지합니다. 현존하는 차량 경로에 대한 다중 에이전트 이상 탐지 데이터셋이 없기에, 연구진은 15,000개 시나리오와 90,000개의 경로를 통해 새로운 벤치마크 데이터셋을 제시합니다. 또한, 그래프 변환기(graph transformer)를 기반으로 한 기본 아키텍처를 도입하여, 네트워크 연결이 불안정한 상황에서도 효과적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CPAD는 전통적인 이상 분류 방법들보다 F1 점수 및 AUC에서 우수한 성과를 보였으며, 에이전트 간의 연결이 중단되는 상황에서도 강한 내구성을 보여주었습니다. 이러한 결과는 자율 주행의 안전하고 신뢰할 수 있는 운행을 위한 기초를 제공합니다. 연구진은 이러한 방법론과 데이터셋을 공개하여 후속 연구를 촉진할 예정입니다.



### Memorize and Rank: Elevating Large Language Models for Clinical Diagnosis Prediction (https://arxiv.org/abs/2501.17326)
Comments:
          To appear at AAAI 2025

- **What's New**: 이번 연구는 MERA라는 임상 진단 예측 모델을 도입하여 자연어 처리(NLP) 지식을 의료 실무에 연결하는 방안을 제시합니다. MERA는 질병 후보 순위 리스트에서 위계적 대조 학습(hierarchical contrastive learning)을 적용하여 방대한 결정 공간 문제를 해결하며, 모델의 내재된 자연어 임상 지식을 의학 코드(medical codes)와 연계시킵니다.

- **Technical Details**: MERA는 환자의 이전 진단 결과를 기반으로 선형 시퀀스로 모델링하여 다음 방문에서의 진단 결과에 대한 확률 분포를 생성하도록 설계되었습니다. 우리는 의료 코드 간의 관계를 활용하여 임상 지식을 통합하고, 대조 학습을 통해 모델이 진짜 진단을 잘 구별하도록 훈련시킵니다. 이는 ICD 코드의 위계 구조 내에서 여러 수준의 대조 학습을 통해 수행됩니다.

- **Performance Highlights**: MERA는 MIMIC-III와 MIMIC-IV 데이터셋에서 일반 진단 및 심부전 예측 임무에서 기존의 최첨단 모델들과 비교해 существенно 개선된 성과를 보여주었습니다. MERA는 의료 코드 및 정의 간의 양방향 매핑을 거의 완벽하게 암기할 수 있으며, GPT-4보다도 진단 예측 역량이 뛰어난 것으로 나타났습니다.



### Connecting Federated ADMM to Bayes (https://arxiv.org/abs/2501.17325)
- **What's New**: 이번 논문에서는 ADMM(Alternating Direction Methods of Multiplier)와 VB(Variational Bayes)라는 두 가지 상이한 연합 학습(federated learning) 접근 방식 간의 새로운 연결고리를 제시합니다. 특히, ADMM에서 사용되는 이중 변수(dual variables)가 VB에서 사용하는 'site' 매개변수를 통해 자연스럽게 나타남을 보여줍니다. 이를 통해 유연한 공분산(flexible covariances)과 기능 정규화(functional regularisation)를 활용한 ADMM의 두 가지 변형을 도출하였습니다.

- **Technical Details**: 연합 학습의 목표는 여러 지역 클라이언트에 분산된 데이터를 사용하여 중앙 서버에서 글로벌 모델을 학습하는 것입니다. ADMM 접근 방식은 제약 최적화(constrained optimisation)를 사용하여 글로벌 모델과 지역 모델을 동기화하며, 동시에 원시(primal)와 이중 변수(dual variables)를 업데이트합니다. 반면, VB 접근 방식은 지역 후방 분포(local posterior distributions)를 메시지로 사용하여 글로벌 후방 분포를 계산하는 데 중점을 두며, 분포의 상관관계가 ADMM의 업데이트 과정에도 반영됩니다.

- **Performance Highlights**: 실험 결과, 제안된 새로운 알고리즘이 다양한 모델, 데이터셋, 클라이언트 수 및 데이터 이질성 수준에서 성능을 향상시키는 것으로 나타났습니다. ADMM과 VB를 결합한 연합 학습 방법이 기존 기술들과 비교하여 수렴 속도가 개선되었음을 보여주었습니다. 또한, 두 접근 방식 간의 연결을 통해 연합 학습 분야의 연구에 기여할 것으로 기대됩니다.



### Multi-Physics Simulations via Coupled Fourier Neural Operator (https://arxiv.org/abs/2501.17296)
- **What's New**: 이 논문에서는 복합 다물리학 신경 연산자 학습 프레임워크인 COMPOL을 소개합니다. 이 새로운 접근법은 Fourier Neural Operator(FNO)의 능력을 확장하여 다수의 물리적 프로세스 간의 상호작용을 모델링합니다. 특히, COMPOL은 순환 신경망(RNN) 및 주의 메커니즘을 활용하여 feature aggregation을 수행하여 복잡한 상호작용을 포괄적으로 모델링할 수 있습니다.

- **Technical Details**: COMPOL 프레임워크는 두 가지 혁신적인 feature aggregation 방식을 개발하여 Fourier Neural Operator 아키텍처를 기반으로 합니다. 첫 번째 방법은 RNN을 사용하여 이전 hidden layer의 출력을 상태 입력으로 결합하여 후속 layer에서 사용할 augmentation features를 생성합니다. 두 번째 방법은 주의 메커니즘을 활용하여 latent features를 변환하고, multi-head attention을 통해 이전 layer의 latent outputs를 집계합니다.

- **Performance Highlights**: 평가 결과, COMPOL은 생물 시스템, 유체 역학 및 다상 흐름과 같은 다양한 물리 시뮬레이션 작업에서 뛰어난 예측 정확도를 보여주었습니다. 기존 방법과 비교하여 COMPOL은 예측 성능에서 2배에서 3배 향상을 나타내어 복합 물리 시스템을 시뮬레이션하는 강력하고 유연한 프레임워크로 자리매김합니다.



### Mitigating Hallucinated Translations in Large Language Models with Hallucination-focused Preference Optimization (https://arxiv.org/abs/2501.17295)
Comments:
          NAACL 2025 Main Conference Long paper (9 pages)

- **What's New**: 이번 연구에서는 기계 번역(Machine Translation, MT)의 경향 변화에 대해 논의하고 있습니다. 전통적인 인코더-디코더 모델에 비해 세밀하게 조정된 대형 언어 모델(Large Language Models, LLM) 기반 시스템이 경쟁력을 가지게 되었다는 점이 주목할 만합니다. 그러나 LLM 시스템의 주요 문제는 생성되는 허위 번역(hallucinations)의 위험이 증가한다는 것입니다.

- **Technical Details**: 허위 번역의 문제를 해결하기 위해, 기존 연구는 주로 전통적인 MT 모델에 초점을 맞추고, 후속 수정(post-hoc mitigation) 방식을 사용했습니다. 그러나 본 연구에서는 모델 훈련 단계에서 허위 번역을 내재적으로 완화할 수 있는 방법을 제안합니다. 이 과정에서 허위 번역에 대한 선호 데이터셋(preference datasets)을 생성하는 데이터 생성 프레임워크를 도입했습니다.

- **Performance Highlights**: 이 방법으로 LLM을 미세 조정하게 되면, 다섯 가지 언어 쌍에서 평균 96%의 허위 번역 비율 감소를 보이는 동시에 전체 번역 품질을 유지할 수 있었습니다. 제로샷(zero-shot) 환경에서 본 접근법은 세 가지 미지의 목표 언어(target languages)에서 평균 89%의 허위 번역 감소 효과를 나타냈습니다.



### Fine-Tuning Open-Source Large Language Models to Improve Their Performance on Radiation Oncology Tasks: A Feasibility Study to Investigate Their Potential Clinical Applications in Radiation Oncology (https://arxiv.org/abs/2501.17286)
- **What's New**: 이번 연구에서는 방사선 종양학(radiation oncology) 분야에서 대규모 언어 모델(LLMs)의 활용 가능성을 탐구하고 있습니다. 특히, 도메인 지식(domain knowledge)으로 세밀하게 조정된 LLM들이 치료 요법 생성(treatment regimen generation), 치료 방식 선택(treatment modality selection), ICD-10 코드 예측에 효과적인 성과를 낼 수 있는지를 검토하였습니다. 이는 해당 분야의 데이터 처리에 새로운 시각을 제공하는 연구입니다.

- **Technical Details**: 이 연구는 15,724명의 환자 사례에서 데이터를 추출하였으며, 7,903개의 사례를 대상으로 진단, 치료 계획, 치료 방식, ICD-10 코드를 수집하여 전처리 및 수작업 주석 작업을 진행했습니다. 각 사례는 환자 진단 상세정보와 각 작업에 대한 답변(치료 요법, 치료 방식 또는 ICD-10 코드)을 구성하여 감독된 세밀 조정(supervised fine-tuning)을 위한 쌍으로 사용되었습니다. LLaMA2-7B 및 Mistral-7B 모델을 활용하였고, Low-Rank Approximations 방법을 사용하여 성능을 향상시켰습니다.

- **Performance Highlights**: 세밀 조정된 LLM은 모든 작업(Task)에서 원래 LLM을 능가한 것으로 나타났으며, 통계적으로 유의미한 성과(p-value <= 0.001)를 보였습니다. 방사선 종양학자에 의한 임상 평가에서는 세밀 조정된 LLM이 생성한 치료 요법의 60% 이상이 임상적으로 수용 가능하다고 평가되었습니다. 또한, 정밀도(precision), 재현율(recall), F1 스코어의 지표에서도 개선된 성과를 기록했습니다.



### ViT-2SPN: Vision Transformer-based Dual-Stream Self-Supervised Pretraining Networks for Retinal OCT Classification (https://arxiv.org/abs/2501.17260)
- **What's New**: 이번 연구에서는 ViT-2SPN이라는 새로운 프레임워크를 제안하여 Optical Coherence Tomography (OCT) 이미지 분석의 성능을 높이고자 하였습니다. ViT-2SPN은 Supervised Pretraining, Self-Supervised Pretraining (SSP), Supervised Fine-Tuning의 세 가지 단계의 워크플로우를 사용합니다. 이 방법은 대량의 무표시 데이터에서 피처 추출을 강화하고 진단 정확도를 향상시키는데 기여합니다.

- **Technical Details**: ViT-2SPN의 훈련 단계는 OCTMNIST 데이터셋을 기반으로 진행되며, 이는 97,477개의 비표시 이미지를 포함하고 있습니다. 본 연구에서는 Vision Transformer (ViT-Base) 백본을 사용하여 특징을 추출하고, 네거티브 코사인 유사도 손실을 적용하여 특성 표현을 정렬합니다. 50 에포크 동안 훈련을 진행하고, 최종적으로 10-겹 교차 검증을 통해 미세 조정을 수행합니다.

- **Performance Highlights**: ViT-2SPN은 평균 AUC 0.93, 정확도 0.77, 정밀도 0.81, 재현율 0.75, F1 점수 0.76을 달성하여 기존의 SSP 기반 방법들을 초월하는 성과를 보였습니다. 이는 특히 OCT 분석에서 데이터 효율성을 개선하고, 클래스 불균형 문제를 해결하는데 중요한 진전을 의미합니다.



### Rethinking Functional Brain Connectome Analysis: Do Graph Deep Learning Models Help? (https://arxiv.org/abs/2501.17207)
Comments:
          22 pages, 6 figures

- **What's New**: 이 연구에서는 최신의 graph deep learning (GDL) 모델을 재검토하여, 이러한 모델의 메시지 집계(mechanism)가 예측 성능을 저하시킨다는 놀라운 결과를 제시합니다. 이를 극복하고자 선형 모델(linear model)과 그래프 어텐션 네트워크(graph attention network)를 결합한 하이브리드 모델을 제안하며, 해당 모델은 예측 성능과 해석 가능성을 갖추고 있습니다. 이러한 연구 결과는 기능적 뇌 연결체 분석에서 복잡한 딥러닝 모델의 신중한 사용을 촉구합니다.

- **Technical Details**: 기존의 GDL 모델의 메시지 집계 메커니즘은 뇌 연결체 분석에서 예측 성능을 저해하는 효과를 보이며, 이는 ABIDE, PNC, HCP, ABCD 등 네 개의 대규모 fMRI 데이터셋을 통해 입증되었습니다. 제안된 모델은 선형 모델이 뇌 연결체의 전역 연결성 패턴을 효율적으로 포착하고, 그래프 어텐션 네트워크가 로컬 구조를 추출하여 두 경로를 통해 시너지를 발휘합니다. 이와 같은 이중 경로 설계는 특정 인지 과정과 관련된 주요 기능적 연결을 밝혀내며, 뇌의 모듈화된 조직에 대한 통찰을 제공합니다.

- **Performance Highlights**: 제안된 모델은 다양한 데이터셋에서 일관되게 강력한 성능을 달성했으며, 최신 베이스라인 모델과 비교해도 경쟁력 있는 성과를 보여주었습니다. GDL 모델과 Classical ML 모델 간의 성능 격차는 연구를 통해 명확히 드러났으며, 특히 GDL 모델의 메시지 집계 메커니즘이 예측 성능에 부정적인 영향을 미친다는 사실이 강조되었습니다. 이러한 결과는 뇌 연결체 분석에서 예측 정확도보다 해석 가능성을 우선시해야 할 필요성을 제기합니다.



### Improving LLM Leaderboards with Psychometrical Methodology (https://arxiv.org/abs/2501.17200)
Comments:
          53 pages, 10 figures, 6 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능을 평가하기 위한 벤치마크(benchmark)의 필요성에 대해 다루고 있습니다. 기존의 벤치마크는 인간의 테스트와 설문조사처럼 질문 세트로 구성되어 있으며, 시스템의 인지 행동에서 나타나는 특성을 측정하고자 합니다. 이 논문은 현대의 심리측정(psychometric) 방법론을 적용하여 LLM의 순위를 개선할 수 있다는 점을 강조합니다.

- **Technical Details**: 논문에서는 Hugging Face Leaderboard 데이터를 예로 들어 전통적인 단순 순위(rank) 접근법과 심리측정 정보를 바탕으로 한 순위 방식을 비교합니다. 이러한 심리측정 기술은 인간 테스트 및 설문조사를 위해 개발된 방법으로, LLM의 성능을 평가하는 데 있어 사실적이고 강력한 결과를 도출할 수 있습니다. 특히, 심리측정 기법의 적용은 기존의 평균(score) 기반 집계 방식에 비해 더 의미있는 평가를 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 심리측정 기법의 도입이 LLM 성능 평가의 신뢰성을 높이는 데 기여하며, 벤치마크에서 얻는 데이터를 활용하여 더 정교한 비교가 가능하다는 것을 보여줍니다. 이를 통해 LLM의 능력에 대한 보다 깊이 있는 분석이 가능해지며, 향후 LLM의 발전에 중요한 지침을 제공할 수 있습니다.



### Atla Selene Mini: A General Purpose Evaluation Mod (https://arxiv.org/abs/2501.17195)
- **What's New**: Atla Selene Mini는 최신의 소형 언어 모델인 'small language model-as-a-judge' (SLMJ)로, 11개의 벤치마크에서 최고의 성능을 기록하며 다양한 평가 작업에서 가장 우수한 결과를 보여줍니다. 이 모델은 고유한 데이터를 사용하여 인간 전문가 평가와의 제로샷 일치를 크게 향상시켰으며, 공공 데이터셋에 대한 합성 비판을 추가하여 훈련하였습니다. HuggingFace와 Ollama에서 모델 가중치가 공개되어, 커뮤니티의 광범위한 채택을 촉진하고자 합니다.

- **Technical Details**: Selene Mini는 8B Instruct 모델을 사용하여 훈련되었고, 16개의 공개 데이터셋을 조합하여 총 577,000개의 데이터 포인트로 구성된 데이터 믹스를 활용하였습니다. 이 모델은 'direct preference optimization' (DPO)와 'supervised fine-tuning' (SFT) 손실을 결합하여 훈련되는 등 최적화된 구조를 가지고 있습니다. 데이터 품질을 확보하기 위한 정제 프로세스를 개발하여 합성 생성 및 필터링 방식을 도입하였으며, 이는 모델의 현실적 평가 능력을 크게 향상시킵니다.

- **Performance Highlights**: Selene Mini는 현실 세계의 금융 및 의료 데이터셋에서 인간 전문가 평가와의 제로샷 일치에서 현저한 개선을 보여주고, 다양한 프롬프트 형식에 대해서도 강력한 성능을 발휘합니다. 커뮤니티 주도의 Judge Arena에서 최고의 평가자로 자리매김하였으며, 이는 사용자들 사이에서 긍정적인 초기 결과를 나타냅니다. 모든 평가 작업에서의 고도화된 성능 덕분에 Selene Mini는 가용성 및 사용 편의성 측면에서 큰 주목을 받고 있습니다.



### Visualizing Uncertainty in Translation Tasks: An Evaluation of LLM Performance and Confidence Metrics (https://arxiv.org/abs/2501.17187)
- **What's New**: 이 논문은 머신 번역(Machine Translation)에서 대형 언어 모델(LLM)의 불확실성을 효과적으로 시각화하는 방법을 제시합니다. 연구의 주요 두 가지 목표는 사용자에게 모델의 신뢰도를 토큰 수준에서 제공하는 것과 번역 불확실성을 정량화하고 표현하는 웹 기반 시각화 도구를 개발하는 것입니다. 또한, T5 모델을 활용하여 WMT19 데이터셋으로 번역 품질을 평가하고 새로운 불확실성 정량화(UQ) 메트릭을 소개합니다.

- **Technical Details**: 연구에서는 세 가지 새로운 불확실성 정량화 메트릭을 도입했습니다: (1) 토큰 확률의 기하 평균, (2) 토큰 확률의 산술 평균, (3) 토큰 분포의 커토시스의 산술 평균. 이러한 메트릭은 번역 성능을 평가하기 위한 간단하면서도 효과적인 프레임워크를 제공합니다. 웹 기반 시각화 도구는 색상 그래디언트를 사용하여 각 토큰의 신뢰도를 표기하여 사용자가 번역의 품질을 직관적으로 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 결과 분석에서 전통적인 평가 메트릭과 새로운 UQ 메트릭 간의 선형 관계가 나타났으며, 연구 방법의 유효성을 입증했습니다. 또한, 인터랙티브 웹 기반 시각화 도구를 사용하여 사용자에게 번역 모델의 성능에 대한 귀중한 통찰력을 제공하며 이러한 불확실성 정량화 메트릭과 시각화 방법은 머신 번역 시스템을 평가하고 접근하는 데 실질적인 도구가 됨을 보여줍니다.



### LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering (https://arxiv.org/abs/2501.17183)
Comments:
          conference paper

- **What's New**: 이 논문은 항공 우주 제조 분야에 적합한 대규모 언어 모델(LLMs), 즉 GPT-4와 QWen의 적용을 위한 새로운 평가 메트릭스를 제안합니다. 이 메트릭스는 전문 지식을 바탕으로 이루어진 질문에 대한 모델의 정확성을 평가하며, 기존 LLM의 '환각(hallucination)' 현상으로 인한 오류를 방지하기 위한 기초를 제공합니다. 이 연구는 데이터의 신뢰성을 확보하고, 안전 기준에 맞춘 평가 기준의 부재 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 항공 우주 제조는 기술적 정확성 요구 사항이 매우 엄격하며, LLM은 프로세스 설계, 재료 선택 및 도구 정보 검색과 같은 작업에 활용될 수 있습니다. 연구진은 항공 우주 분야의 전통적인 문헌을 기반으로 다수의 선택형 질문을 생성하고, 이 질문들을 통해 다양한 LLM 모델의 성능을 평가하였습니다. 결과적으로, 현재 LLM의 전문 지식 수준이 미흡하다는 것을 발견했으며, 이에 대한 개선이 절실함을 강조합니다.

- **Performance Highlights**: 본 연구의 실험 결과에 따르면, 항공 우주 관련 전문 지식에서 LLM의 정확도는 FAA의 허용된 위험 임계치를 초과하는 징후가 발견되었습니다. 특정 기술 매개변수에 대한 오류는 기계적 구조의 안전성을 저해하여 비극적인 결과를 초래할 수 있으며, 이는 데이터 품질 관리를 위한 보다 철저한 검증 프로세스의 필요성을 명확히 합니다. 이 작업은 항공 우주 제조 분야에 LLM을 안전하게 도입하기 위한 기초적인 이해와 실질적인 통찰력을 제공하는 것을 목표로 합니다.



### Dialogue Systems for Emotional Support via Value Reinforcemen (https://arxiv.org/abs/2501.17182)
Comments:
          30 pages, 3 figures

- **What's New**: 본 연구는 감정 지원 시스템에 대한 가치 강화(value reinforcement)를 명시적으로 통합한 첫 번째 사례로, 감정 지원 대화 시스템이 도움을 요청하는 사람의 긍정적인 가치를 강화하도록 설계된 방법론을 제안합니다. 이 모델은 Reddit에서 온라인 지원 대화를 활용하여 각 대화 턴에서 어떤 가치를 강화할지를 학습합니다. 가치 강화가 감정 지원의 효과성을 높이는 잠재력이 있음을 검증하고, 미래 연구를 위한 기초를 마련했습니다.

- **Technical Details**: 연구에서는 목표 가치 감지기(target value detector)와 참조 생성기(reference generator)라는 두 가지 컴포넌트를 도입하여 지지자의 가치 강화 능력을 향상시킵니다. 시스템은 GPT-4o-mini를 기반으로 한 시뮬레이션 훈련과 직접 정책 최적화(direct policy optimization)를 통해 훈련됩니다. 이러한 구조는 보조자의 발언에 반영된 가치 증진의 보상을 극대화하는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 감정 지원 능력 및 가치 강화 면에서 다양한 기준선 모델을 능가하며, 특히 가치 강화 부분에서 전문가 치료사들의 평가에서 두드러진 성과를 보였습니다. 모델은 도움을 요청하는 사람의 도전 과제를 효과적으로 확인하고 긍정적인 측면을 강조하는 능력이 뛰어난 것으로 평가되었습니다. 이는 연구의 주요 기여로, 감정 지원 시스템의 성과 향상에 대한 새로운 방향을 제시합니다.



### Tuning LLM Judges Hyperparameters (https://arxiv.org/abs/2501.17178)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 평가에 필요한 비싼 인간 주석 비용을 줄이기 위해 LLM 기반의 판별자를 제안합니다. 이러한 판별자는 두 개의 LLM 출력을 비교하여 모델을 평가할 수 있으며, 동시에 여러 하이퍼파라미터를 조정하여 성능을 최적화하려고 합니다. 이 방법은 기존의 평가 방법들과 비교할 때 정확성과 비용 효율성 모두에서 뛰어난 성능을 보여주고, 오픈-웨이트 모델을 사용하여 접근성과 재현성을 높입니다.

- **Technical Details**: 저자들은 LLM 판별자의 하이퍼파라미터를 체계적으로 분석하고 조정하는 방법을 제안합니다. 주요 하이퍼파라미터로는 LLM 모델, 프롬프트, 추론 파라미터(예: temperature)가 포함되며, 다목적(multi-objective) 및 다신뢰도(multi-fidelity) 접근 방식을 통해 조정 비용을 절감합니다. 이는 판별자의 성능 개선에 기여하고, 더 나은 LLM 모델 평가를 위한 최적의 구성 요소를 식별할 수 있게 해줍니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 기존의 판별자들과 비교하여 더욱 높은 정확성과 비용 효율성을 나타내며, 여러 실제 테스트 데이터셋에서도 우수한 성과를 거두었습니다. 저자들은 80가지의 프롬프트 전략과 4480가지 판별자 구성을 포함한 검색 공간을 다루며, 최적의 프롬프트 파라미터화 및 기타 하이퍼파라미터 조정 방법론을 제시합니다. 이로 인해 커뮤니티는 더 나은 판별자 구축을 위한 중요한 패턴을 확인할 수 있습니다.



### Prompt-Based Cost-Effective Evaluation and Operation of ChatGPT as a Computer Programming Teaching Assistan (https://arxiv.org/abs/2501.17176)
- **What's New**: 최근 등장한 대규모 언어 모델(LLMs)의 발전으로 1:1 학생-교사 비율 달성의 가능성이 커지고 있습니다. 이 모델들은 특히 대학 프로그래밍 과정에서 학생들에게 실시간 피드백을 제공하는 데 사용될 수 있습니다. 연구에서는 GPT-3.5T와 GPT-4T 모델의 성능을 비교하였고, GPT-4T가 월등한 성과를 보였지만 실제 사용에는 여전히 문제가 있는 것으로 나타났습니다.

- **Technical Details**: 논문은 GPT 모델을 활용한 교육 도구의 구현 방안에 대해 세 가지 주요 측면을 다룹니다. 첫째, 다양한 피드백을 제공할 수 있는 잘 설계된 프롬프트(prompts)를 제안하여 LLM의 성능을 평가하는 방법론을 수립했습니다. 둘째, 피드백의 정확성을 평가하기 위한 자동화된 평가 메트릭스를 개발하였고, 이를 통해 수동 평가보다 시간과 비용을 절약할 수 있음을 강조하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM이 제공하는 피드백은 프로그램의 구조적 분석이 가능하여, AI가 학습 과정에서 프로그래밍 문제를 해결하는 데 필요한 진단 정보를 포함할 수 있음을 보여줍니다. 이러한 자동화된 접근법은 특히 대규모 클래스에서 학생들에게 유용할 수 있으며, 이는 향후 교육 환경의 혁신적인 변화를 이끌 수 있는 잠재력을 가지고 있습니다.



### Document-Level Sentiment Analysis of Urdu Text Using Deep Learning Techniques (https://arxiv.org/abs/2501.17175)
- **What's New**: 이 연구에서는 문서 수준에서 우르두어 감정 분석(Sentiment Analysis, SA)을 위한 새로운 하이브리드 딥러닝 모델인 BiLSTM-SLMFCNN을 제안하고 있습니다. 처음으로 우르두어 데이터를 위한 딥러닝 아키텍처들을 적용하여 성과를 입증하는 데 초점을 맞추었습니다. 특히, 기존의 전통적인 기계 학습 모델보다 딥러닝 모델이 더욱 효과적이며, 문서 크기에 따른 성능 변화를 함께 분석하였습니다.

- **Technical Details**: 제안된 BiLSTM-SLMFCNN 모델은 Bidirectional Long Short Term Memory(BiLSTM)와 Single Layer Multi Filter Convolutional Neural Network(SLMFCNN)를 결합하여 우르두 문서의 감정을 분류하는 능력을 극대화합니다. 이 모델은 BiLSTM을 통해 단어의 맥락적 의미를 이해하고 SLMFCNN을 통해 지역적 특징을 추출합니다. 또한, 여러 개의 필터를 사용하여 다양한 길이의 n-그램 특징을 추출할 수 있는 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 IMDB 우르두 영화 리뷰 데이터 세트와 우르두 고객 지원 데이터 세트에서 성능을 평가했으며, 각각 83%, 79%, 83% 및 94%의 정확도를 기록하여 기존의 딥러닝 모델을 능가했습니다. 특히, BiLSTM-SLMFCNN은 문서 수준 감정 분석에서 기존의 기계 학습 모델보다 우수한 결과를 보여주었습니다.



### Extractive Schema Linking for Text-to-SQL (https://arxiv.org/abs/2501.17174)
- **What's New**: 이번 연구에서는 Text-to-SQL 시스템을 위한 새로운 스키마 링크 모델을 제안합니다. 이 모델은 decoder-only LLM에서 hidden 상태에 대한 확률 예측을 생성하며, SQL 쿼리의 각 열의 역할에 대한 세밀한 예측이 가능합니다. 연구 결과는 기존의 모델들에 비해 더 높은 정확도를 가지고 있으며, precision과 recall 간의 균형을 조정할 수 있는 기능도 포함되어 있습니다.

- **Technical Details**: 제안된 접근법은 generative LLM과 cross-encoder 접근 방식을 결합하여 각 스키마 항목에 대한 확률 예측을 통해 recall 지향의 예측을 가능하게 합니다. 초기 단계에서 ground truth를 수집하고, SQL 쿼리를 분석하여 사용된 테이블과 열을 식별하여 고유의 구성을 유지합니다. 이 과정에서는 mo-sql-parsing을 사용하여 SQL 문을 정적 분석합니다.

- **Performance Highlights**: 제안된 스키마 링크 모델은 관련 데이터베이스 컬럼에 대한 완벽하고 세밀한 예측을 통해 SQL 쿼리 생성을 지원합니다. 연구 결과, 이전 방법들과 비교하여 더욱 개선된 성능을 보여주며, SQL 생성 과정에서의 오류를 줄이고 정확성을 높이는 데 기여합니다. 이를 통해 다양한 비즈니스 환경에서의 실시간 데이터 기반의 질의 생성이 더욱 원활해질 것으로 기대됩니다.



### Separated Inter/Intra-Modal Fusion Prompts for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.17171)
Comments:
          AIAP 2025

- **What's New**: 이 논문에서는 Compositional Zero-Shot Learning (CZSL) 기법을 통해 의미의 미세한 차이와 상태 및 객체의 조합을 인식하는 새로운 접근방식을 제안합니다. 기존의 방법들이 프롬프트(prompt) 구성이나 프리트레인(pre-trained) 비전-언어 모델 튜닝에 집중하던 반면, 이 방법은 그러한 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 다양한 Prompt Learning 기법과 Inter/Intra-Modality Fusion Synthesizer를 활용하여 장면 이해(scene understanding)에서의 속성 인식(attribute recognition) 성능을 향상시키는 데 초점을 두고 있습니다. 이를 통해 미세한 의미 차이와 여러 객체를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: 이 연구는 CZSL 기술을 통해 보고된 성능을 개선하며, 특히 다양한 의미론적(stylistic) 차이를 명확히 인식함으로써 실험에서 유의미한 결과를 도출했습니다. 이는 구성의 복잡성을 포함한 다양한 상황에서도 강력한 성능을 보이는 것을 의미합니다.



### Benchmarking Randomized Optimization Algorithms on Binary, Permutation, and Combinatorial Problem Landscapes (https://arxiv.org/abs/2501.17170)
- **What's New**: 이번 연구에서는 Randomized Optimization 알고리즘인 Randomized Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithms (GA), MIMIC (Mutual Information Maximizing Input Clustering)을 평가했습니다. 세 가지 문제 유형인 binary, permutation, combinatorial에서 각 알고리즘의 성능을 비교하였습니다. 특히, MIMIC과 GA는 binary 및 combinatorial 문제에서 우수한 해법을 제공하는 반면, RHC와 SA는 복잡한 문제에서 한계가 있음을 보여줍니다.

- **Technical Details**: 최적화 문제는 다양한 분야에서 의사결정 과정의 핵심으로 작용하며, 각 알고리즘의 효과는 문제의 성격에 따라 다릅니다. 논문에서는 이론적인 배경으로 각 알고리즘의 원리와 특징을 서술하며, random optimization 기술이 binary, permutation, combinatorial 문제에 특히 효과적임을 강조합니다. 알고리즘 선택은 탐색(exploration)과 활용(exploitation)의 균형을 맞추고, local minima에 빠지지 않도록 하는 것이 중요합니다.

- **Performance Highlights**: 결과적으로, MIMIC과 GA는 binary 문제와 combinatorial 문제에서 높은 품질의 솔루션을 제공합니다. 그러나, GA는 정확성과 효율성 사이에서 균형을 이루며, MIMIC은 permutation 문제에서 우수한 성과를 보입니다. 반면, RHC와 SA는 연산 비용이 적으나 복잡한 문제에서 성능 저하를 겪는 것으로 나타났습니다.



### EvoGP: A GPU-accelerated Framework for Tree-Based Genetic Programming (https://arxiv.org/abs/2501.17168)
- **What's New**: 이번 논문에서는 GPU 가속을 활용하여 Tree-based Genetic Programming (TGP)의 효율성을 극대화하는 EvoGP 프레임워크를 소개합니다. 기존 TGP 구현의 세 가지 주요 도전 과제를 해결하기 위해, 텐서 기반 인코딩 방식, 통합된 병렬 유전 작업 프레임워크, 그리고 완전 병렬 피트니스 평가 전략을 제안합니다. 이러한 혁신들은 TGP의 속도와 정확성을 크게 향상시킵니다.

- **Technical Details**: EvoGP는 다양한 형태를 가진 프로그램을 텐서로 변환하는 텐서화 인코딩 방식을 도입하여 메모리 접근을 최적화하고 병렬 실행을 가능하게 합니다. 또한 CUDA 커널을 활용하여 유전 작업의 효율성을 높이고, 피트니스 평가 과정에서는 인구 수준과 데이터 수준의 병렬성을 극대화합니다. 이로 인해 GPU 자원을 최대한 활용하여 성능 향상을 달성합니다.

- **Performance Highlights**: 실험 결과, EvoGP는 최신 GPU 기반 TGP 구현 대비 최대 140.89배의 속도 향상을 기록했습니다. EvoGP는 각종 심볼릭 회귀, 분류 및 로봇 제어 작업에서 그 유연함과 효과성을 입증하며, 오픈소스로 제공되어 다양한 연구 및 실험에 활용될 수 있습니다.



### QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks (https://arxiv.org/abs/2501.17167)
- **What's New**: 이번 연구에서는 QualityFlow라는 동적인 에이전틱 워크플로우를 도입하여 프로그램 합성을 위한 새로운 접근 방식을 제시합니다. 주어진 프로그래밍 문제의 영어 설명과 단위 테스트 집합을 바탕으로, 모델은 올바른 프로그램을 합성하여 문제를 해결하고 테스트를 통과하도록 설계되었습니다. 기존의 프로그램 합성 방법들은 테스트 품질의 병목 현상, 자가 디버깅 경로의 편차 등 세 가지 주요 한계를 가지고 있으며, 이러한 문제를 해결하기 위해 LLM 품질 검사기를 제안합니다.

- **Technical Details**: QualityFlow는 프로그램 생성기, 테스트 디자이너, 셀프 디버거와 같은 여러 개의 대형 언어 모델(LLM) 에이전트로 구성되어 소프트웨어 개발 팀과 유사한 기능을 수행합니다. 이 시스템은 각 단계에서 품질 검사를 실시하여 중간 합성물의 올바름을 확인하고, 어긋난 경로가 발생할 경우 리셋하여 문제를 재해석합니다. 이 프로세스를 통해 QualityFlow는 단지 올바른 프로그램만 제출하는 것이 아니라, 테스트 품질을 조정하고 자가 디버깅의 잘못된 경로를 예방합니다.

- **Performance Highlights**: 실험 결과, QualityFlow는 MBPP, HumanEval, EvalPlus에서의 보다 엄격한 평가를 포함한 네 가지 프로그램 합성 벤치마크에서 최첨단 성능을 기록했습니다. Quality Checker의 성공적인 작동은 다변화된 프롬프트(Diversified Prompting)를 가능하게 하여 올바른 프로그램이 생성하고 품질 검사를 통과할 가능성을 극대화합니다. QualityFlow는 동적 워크플로우의 중요성을 보여주며, 정적 워크플로우 및 단일 시도로 인한 0샷 합성을 능가하는 성능을 입증했습니다.



### Split Knowledge Distillation for Large Models in IoT: Architecture, Challenges, and Solutions (https://arxiv.org/abs/2501.17164)
Comments:
          7 pages, 4figures, 2 tables, and 15 conference

- **What's New**: 이번 연구는 인터넷 사물 (IoT) 시스템에서 대형 모델 (Large Models, LMs)의 훈련과 배포의 주요 과제를 분석하고, 본 연구에서 제안하는 동적 자원 관리, 적응형 모델 분할, 클러스터 협업 훈련 등의 솔루션에 대해 설명합니다. 또한, 데이터 프라이버시를 보장하며, LMs를 경량화된 형태로 IoT 장치에 배포할 수 있는 분할 지식 증류 (Split Knowledge Distillation) 프레임워크를 제시합니다. 이 프레임워크는 에너지 소비를 최소화하고 낮은 모델 훈련 지연 요구사항을 충족시키는 것을 목표로 합니다.

- **Technical Details**: 본 연구에서 제안하는 분할 지식 증류 프레임워크는 지식 증류 (Knowledge Distillation)와 분할 학습 (Split Learning)을 통합하여 IoT 장치에서 원시 데이터가 로컬로 유지되도록 하면서 LMs를 경량화된 모델로 증류합니다. 이 과정에서는 수업 모델 (Teacher Model)의 주요 기능과 지식을 사용하여 경량화 모델인 학생 모델 (Student Model)을 훈련시킵니다. 특히, 에지 서버 (Edge Server)에서 모델의 나머지 레이어를 처리하기 위해 중간 활성화만 전송하여 데이터 프라이버시를 유지합니다.

- **Performance Highlights**: 제안된 프레임워크의 실현 가능성과 성능은 사례 연구를 통해 평가되었습니다. 이 사례 연구에서는 에너지 소비 감소와 함께, IoT 환경에서 요구되는 낮은 지연 시간에서의 효과성을 입증하였습니다. 최종적으로, 이 접근 방식은 자원이 제한된 IoT 환경에서 LMs의 배치를 가능하게 하여 지능형 응용 프로그램을 지원하는 데 중점을 둡니다.



### MACI: Multi-Agent Collaborative Intelligence for Adaptive Reasoning and Temporal Planning (https://arxiv.org/abs/2501.16689)
Comments:
          21 pages, 19 tables

- **What's New**: 이 논문에서는 Multi-Agent Collaborative Intelligence (MACI)라는 새로운 프레임워크를 소개합니다. MACI는 인공지능이 가지고 있는 전통적인 한계, 즉 패턴 매칭(narrow pattern matching)과 자기 검증(self-verification)의 부족, 제약 조건 관리의 비효율성을 극복하기 위해 설계되었습니다. 이 프레임워크는 메타 플래너(meta-planner), 일반 및 작업 특화 에이전트(task-specific agents), 그리고 런타임 모니터(run-time monitor)라는 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: MACI의 메타 플래너는 작업 요구사항을 분석하고 역할과 제약 조건을 식별하여 종속 그래프(dependency graph)를 동적으로 생성합니다. 이와 같은 방법을 통해 계획과 검증을 분리하며, 제약 조건의 유효성을 지속적으로 평가합니다. 일반 에이전트는 일반적인 제약 조건을 관리하며, 특화 에이전트는 특정 작업에 대한 전문 지식을 반영하여 수행됩니다.

- **Performance Highlights**: MACI는 Traveling Salesman Problem (TSP)과 복잡한 저녁 식사 계획 과제 등에서 강력한 수행 능력을 입증하였습니다. 해당 연구는 기존 LLMs의 한계를 극복하고, 인공지능이 지혜로운(reasoned) 결정을 내릴 수 있는 능력을 향상시키기 위한 중요한 기초를 제공합니다. 따라서 MACI는 복잡한 계획 문제를 해결하기 위해 필요한 환경에서의 적응성과 유연성을 보장합니다.




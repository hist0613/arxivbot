New uploads on arXiv(cs.CV)

### TEST-V: TEst-time Support-set Tuning for Zero-shot Video Classification (https://arxiv.org/abs/2502.00426)
- **What's New**: 이 논문에서는 제로샷(Zero-shot) 비디오 분류를 위한 새로운 프레임워크인 TEst-time Support-set Tuning(TeST-V)을 제안합니다. 기존의 Test-time Prompt Tuning(TPT)과 지원 집합(support-set) 기반 접근 방식의 장점을 결합하였으며, 지원 집합의 확장과 축소를 통해 제로샷 일반화를 강화합니다. 이 연구는 특히 다중 프롬프트(Multi-prompt)와 시간 인식(Temporal-aware) 가중치를 학습하여 핵심 단서를 동적으로 추출하는 방식을 강조합니다.

- **Technical Details**: TeST-V 프레임워크는 두 가지 주요 모듈로 구성됩니다. 첫 번째는 다중 프롬프트 지원 집합 확장(Multi-prompting Support-set Dilation, MSD)으로, 클래스 이름에서 다수의 텍스트 프롬프트를 생성하여 비디오 생성을 다양화합니다. 두 번째는 시간 인식 지원 집합 침식(Temporal-aware Support-set Erosion, TSE)으로, 수렴적인 특성에 기반하여 각 비디오 프레임의 기여도를 동적으로 조정합니다.

- **Performance Highlights**: 테스트 결과, TeST-V는 CLIP, BIKE 및 VIFi-CLIP 모델 등 최신 사전 훈련된 비전-언어 모델에 비해 대폭 향상된 성능을 보였으며, 각각 2.98%, 2.15%, 1.83%의 절대 평균 정확도를 기록했습니다. 이 논문은 향상된 감지 성능과 함께 지원 집합의 해석 가능성을 강조하고 있습니다.




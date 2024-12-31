New uploads on arXiv(cs.CV)

### DrivingWorld: Constructing World Model for Autonomous Driving via Video GP (https://arxiv.org/abs/2412.19505)
- **What's New**: 최근 자율주행 분야에서 비디오 기반 세계 모델(video-based world models)에 대한 연구가 증가하고 있습니다. GPT(Generative Pre-trained Transformer) 시리즈의 오토회귀 생성 모델을 참고하여, DrivingWorld라는 새로운 GPT 스타일의 세계 모델이 제안되었습니다. 이 모델은 공간-시간 융합 메커니즘(spatial-temporal fusion mechanisms)을 통해 비디오 생성의 질을 크게 향상시킵니다.

- **Technical Details**: DrivingWorld는 세 가지 주요 이니셔티브를 기반으로 하고 있습니다: 1) Temporal-Aware Tokenization으로 비디오 프레임을 시간적으로 일관성 있는 토큰으로 변환합니다. 2) Hybrid Token Prediction 전략을 사용하여 인접한 상태 간의 시간적 일관성을 더 잘 모델링합니다. 3) Long-time Controllable Strategies를 통해 랜덤 토큰 드롭아웃(random token dropout) 및 균형 잡힌 주의(attention) 방법을 적용하여 긴 비디오 생성 시 정확한 제어를 가능하게 합니다.

- **Performance Highlights**: DrivingWorld는 40초 이상의 고충실도 비디오를 생성할 수 있으며, 이는 최신 자율주행 세계 모델에 비해 두 배 이상 긴 것입니다. 실험 결과, 제안된 모델은 뛰어난 시각적 품질과 더 높은 정확도의 제어 가능한 미래 비디오 생성을 보여줍니다. 이를 통해 자율주행 시스템의 일반화 능력을 향상시키고 안전성을 높일 수 있습니다.




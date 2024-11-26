New uploads on arXiv(cs.CV)

### ClickTrack: Towards Real-time Interactive Single Object Tracking (https://arxiv.org/abs/2411.13183)
- **What's New**: 본 논문에서는 실시간 상호작용 시나리오에서 단일 물체 추적기의 초기화 방법을 재평가하고, ClickTrack이라는 새로운 패러다임을 제안합니다. ClickTrack은 클릭 입력을 사용하여 정확한 바운딩 박스를 생성하며, 이 바운딩 박스는 추적기에 입력으로 사용됩니다. 기존의 접근법들이 가지는 한계를 극복하기 위해, Guided Click Refiner(GCR)라는 새로운 모듈을 설계하였습니다.

- **Technical Details**: GCR은 포인트 입력 및 선택적 텍스트 정보를 받아들여, 사용자가 기대하는 바운딩 박스를 생성하는 구조입니다. 이 구조는 기존 시각적 피처와 지침적 피처를 결합하여 사용자의 기대에 부합하는 예측 바운딩 박스를 생성하게 됩니다. ClickTrack 접근 방식은 실시간 상호작용 시나리오에서 단일 물체 추적기와 쉽게 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: LaSOT와 GOT-10k 벤치마크에서 GCR과 STARK 트래커를 결합한 실험 결과, 단일 포인트 및 텍스트 입력으로 우수한 추적 정확도를 달성했습니다. 이는 정밀한 초기 주석과 유사한 성능을 보였습니다. GCR 모델은 실시간 상호작용 요구사항을 충족할 수 있는 처리 속도를 자랑하며, 단일 포인트 주석으로 인한 모호성을 효과적으로 완화하는 데 기여합니다.




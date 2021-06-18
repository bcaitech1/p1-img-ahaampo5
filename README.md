# Source code 설명

- data.py
  - 데이터 전처리를 label별로 나누어주는 함수가 있습니다.
  - test & validation set 구성과 그에 필요한 데이터 관련 보조 코드가 있습니다.

- ensemble.py
  - 여러 모델을 ensemble하여 추론 할 수 있도록 만든 코드

- infer.py
  - 한 가지 모델을 추론하여 csv파일로 저장해주는 파일입니다.

- infer_sep.py
  - 모델 구조를 mask, gender, age 3가지로 나누어 만든 모델 추론 코드

- loss.py
  - 이미지 분류에 사용되는 다양한 Loss를 정의한 파일입니다.

- model.py
  - ResizingNetwork, Efficient Net, ResNet이 구현된 코드가 있습니다.

- train.py
  - 학습에 사용되는 main코드입니다.
  - 이곳에 Augmentation코드가 구현되어 있습니다.
  - cross entropy, binary cross entropy를 평하가는 evaluation 코드가 있습니다.

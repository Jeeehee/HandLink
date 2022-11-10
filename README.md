# HandLink

## 소개
인공지능을 활용해 실시간 수어 번역 App을 제작하기위한, DataSet 추출 및 인공지능 학습 프로그램입니다.

- Python Mediapipe를 사용해 손의 21개 3D 랜드마크를 추출 -> 각도를 활용해 Dataset을 추출
- 딥러닝 뉴럴 네트워크 중 **RNN의 LSTM(Long Short Term Memory)** 으로 인공지능 학습

<br>

## 개발자
|[Jee.e](https://github.com/Jeeehee)|황지희|<img width="130" alt="사진" src="https://user-images.githubusercontent.com/92635121/200990518-49c850d3-91b9-4818-8666-f0f0cc85479a.png">|
|--|--|--|


<br>

## Trouble Shooting
- [주피터 노트북 설치 에러](https://github.com/Jeeehee/HandLink/issues/1)
- [MediaPipe의 한 손 인식으로 정확도 저하](https://github.com/Jeeehee/HandLink/issues/2)
- [LSTM 정확도 개선](https://github.com/Jeeehee/HandLink/issues/3)

<br>

#### Dependency
- Python 3
- TensorFlow
- numpy
- OpenCV
- MediaPipe

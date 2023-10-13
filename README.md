# myCV
졸업과제
## 실행 환경
* python 3.10.0
* requirements.txt 참조
* CUDA toolkit 11.2
* cnDNN 8.1
## 목적
* rtsp 프로토콜을 통해서 받아온 영상을 실시간으로 딥러닝에 적용시킨다.
* LOL Dataset으로 학습시킨 mirnet을 사용해서 화면을 밝게 변경한다.
## 모델 설명
#### mirnet
* 하이퍼 파라미터 : rrg=1, mrb=1, channel=64
### mirnet2
* 하이퍼 파라미터 : rrg=1 mrb=2 channel=64
### mirnet3 
* 하이퍼 파라미터 : rrg=1 mrb=1 channel=64
* 모델 변경 (논문참고) :Learning Enriched Features for Fast Image Restoration and Enhancement
### mirnet4
* 하이퍼 파라미터 : rrg=1 mrb=1 channel=64
* 모델 변경 (논문참고) :Learning Enriched Features for Fast Image Restoration and Enhancement
* parallel multi-resolution convolution stream 개수 3 -> 2 줄임


## 추가사항
* 빛번짐 모델 추가(homomorphic filter로 대체)
* 그래픽카드 인식

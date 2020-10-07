---
layout: post
title: < CS231n 정리 > 12. Detection and Segmentation
category: dev
permalink: /MLDL/:year/:month/:day/:title/
tags: dev mldl cs231n Stanford
comments: true
---

## 소개
- 이 글은 단지 CS231n를 공부하고 정리하기 위한 글입니다.
- Machine Learning과 Deep Learning에 대한 지식이 없는 초보입니다.
- 내용에 오류가 있는 부분이 있다면 조언 및 지적 언제든 환영입니다!

---
<br><br>


## 참조
- [CS231n 2017 Lecture 11. 유튜브 강의](https://www.youtube.com/watch?v=nDPWywWRIRo&ab_channel=StanfordUniversitySchoolofEngineering)

- [CS231n 2017 Lecture 11. 강의 노트](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)

- [CS231n 2020 Lecture 12. 강의 노트](http://cs231n.stanford.edu/slides/2020/lecture_12.pdf)

- [CS231 2017 Lecture 11. 한국어 자막](https://github.com/visionNoob/CS231N_17_KOR_SUB/blob/master/kor/Lecture%2011%20%20%20Detection%20and%20Segmentation.ko.srt)

- [https://leechamin.tistory.com/112](https://leechamin.tistory.com/112)

- [https://wordbe.tistory.com/entry/cs231n-12-Object-detection-Segmentation](https://wordbe.tistory.com/entry/cs231n-12-Object-detection-Segmentation)

- [https://lsjsj92.tistory.com/416](https://lsjsj92.tistory.com/416)

- [https://devkor.tistory.com/entry/CS231n-11-Detection-and-Segmentation](https://devkor.tistory.com/entry/CS231n-11-Detection-and-Segmentation)

- [https://m.blog.naver.com/wpxkxmfpdls/221878233486](https://m.blog.naver.com/wpxkxmfpdls/221878233486)

- [https://younghk.netlify.app/posts/cs231n-lec12-detection-and-segmentation/](https://younghk.netlify.app/posts/cs231n-lec12-detection-and-segmentation/)

- [https://bigdatadiary0819.tistory.com/58](https://bigdatadiary0819.tistory.com/58)

- [https://zzsza.github.io/data/2018/05/30/cs231n-detection-and-segmentation/](https://zzsza.github.io/data/2018/05/30/cs231n-detection-and-segmentation/)

- [https://curt-park.github.io/2017-03-26/yolo/](https://curt-park.github.io/2017-03-26/yolo/)

- [https://mylifemystudy.tistory.com/82](https://mylifemystudy.tistory.com/82)

---
<br><br>

## 개요
### < Detection and Segmentation >

0. [Detection and Segmentation](#detection-and-segmentation)
1. [Semantic Segmentation](#semantic-segmentation)
    1. [Semantic Segmentation Sliding Window](#semantic-segmentation-sliding-window)
    2. [Semantic Segmentation Fully Convolutional](#semantic-segmentation-fully-convolutional)
    3. [Downsampling and Upsampling](#downsampling-and-upsampling)
        1. [Nearest Neighbor and Bed of Nails](#nearest-neighbor-and-bed-of-nails)
        2. [Max Unpooling](#max-unpooling)
        3. [Transpose Convolution](#transpose-convolution)
2. [Object Detection](#object-detection)
    1. [Object Detection Sliding Window](#object-detection-sliding-window)
    2. [Object Detection Region Proposals](#object-detection-region-proposals)
    3. [Object Detection R CNN](#object-detection-r-cnn)
    4. [Object Detection Fast R CNN](#object-detection-fast-r-cnn)
    5. [Object Detection Faster R CNN](#object-detection-faster-r-cnn)
    6. [Object Detection YOLO and SSD](#object-detection-yolo-and-ssd)
3. [Dense Captioning](#dense-captioning)
4. [Instance Segmentation](#instance-segmentation)
5. [Mask R CNN](#mask-r-cnn)
    1. [RoI Pooling vs RoI Align](#roi-pooling-vs-roi-align)
6. [Aside](#aside)

---

## Detection and Segmentation

컴퓨터 비전에는 다양한 문제가 존재합니다.

전에 배웠던 CNN에서는 하나의 사진에 대해서 하나의 물체로 분류하는 작업을 했습니다.

하지만 이 이미지에 어떤 영역에 물체가 있는지, 어느 위치에 있는지, 여러 종류의 물체가 있는 경우와 같은 문제에 대해서는 아직 다루지 않았습니다.

이번 챕터에서는 물체의 영역을 나누는 `Semantic Segmentation`, 여러 물체의 위치가 어디에 있는지 찾아내는 `Object Detection`, 이에 이어서 각 물체마다 Semantic Segmentation를 진행하는 `Instance Segmentation`에 대해서 다루겠습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-001-ComputerVisionTasks.png)

---

## Semantic Segmentation

먼저 Semantic Segmentation에 대해서 다루도록 하겠습니다,

Semantic Segmentation 은 아래 그림과 같이 한 그림 안에서 각 영역을 나누는 분야입니다.

이전에는 한 이미지에 대해서 물체를 분류하는 작업을 했다면, 이는 pixel 단위로 각 pixel이 어떤 물체를 나타내는지 분류하고 나누는 작업입니다.

그렇다면 이것을 머신러닝으로 학습을 시키기 위한 방법을 생각해봅시다.

우선 아주 간단한 방법으로는 모든 training 데이터가 각 pixel에 대한 segmentation이 모두 되어 있는 경우입니다.

이러한 데이터를 만들기는 정말 어렵기 때문에 다른 방법이 필요합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-002-SemanticSegmentation.png)

---

### Semantic Segmentation Sliding Window

위에 문제점을 해결하기 위한 다른 방법으로 Sliding Window 방법이 있습니다.

이 방법은 작은 크기에 Window를 준비하고 돋보기로 훝는 것과 같이 이미지를 분류하는 작업입니다.

이 방법은 기존에 만들어 놓은 GoogLeNet과 ReNet를 이용하면 쉽게 해결될 것처럼 보입니다.

하지만 이 방법에는 문제가 있습니다.

일단 이 모든 영역을 작은 영역으로 잘라서 입력으로 넣어주어 계산 비용이 엉청납니다.

또한 만약 서로 다른 영역이 인접해 있는 경우에는 그 특징을 공유를 하기 되어 분류하는데 있어서 안좋습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-003-SegmentSlidingWindow.png)

---

### Semantic Segmentation Fully Convolutional

이러한 문제를 해결하기 위해서 나온 방법은 하나의 이미지 입력에 대해서 한번에 결과를 얻는 방법입니다.

이 방법은 좋아 보이지만, 높은 해상도의 이미지와 같은 경우 굉장하게 많은 계산량과 파라미터 수를 요구합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-004-SegmentFullyConvolutional.png)

---

### Downsampling and Upsampling

위에 방법은 너무 많은 계산량을 요구했습니다.

이러한 문제점을 완화시키기 위해서 downsampling과 upsampling 방법이 제시되었습니다.

이미지의 크기를 그대로 유지하여 CNN 작업을 하던 전 방법과 다르게, 이미지의 크기를 한번 Downsampling을 통해서 크기를 줄입니다.
이후에 Upsampling 으로 원래의 이미지 크기로 복원하여 계산 효율성을 높였습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-005-UpsamplingDownSampling.png)

대표적인 Donwsampling 방법은 Max Pooling과 ConV의 Stride를 2를 주는 방식입니다.

먼저 Max Pooling이후에 Unpooling(Upsampling)을 하는 방법은 여러가지 방법이 있싑니다. 이에 대한 설명은 아래와 같습니다.

---

### Nearest Neighbor and Bed of Nails

첫번째 방법은 그냥 주변에 이웃값으로 채우는 것입니다. Max Pooling stride값이 2인 경우 아래와 같이 같은 값으로 채워줍니다.

이와 달리 Bed of Nails는 그냥 가장자리 한쪽에만 채우고 나머지는 0으로 채웁니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-006-UpSamplingMethod_01.png)

---

### Max Unpooling

다음으로는 기존에 Max Pooling 된 위치 정보를 기억하는 방법입니다.

Unpooling 시에 기존에 최대 값이었던 곳으로 Unpooling 하고 나머지는 0으로 사용하는 방법입니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-007-UpSamplingMethod_02.png)

---

### Transpose Convolution

이제 Max Pooling의 방법이 아니라 Conv의 stride값을 이용해 Downsampling한후 Upsampling하는 방법을 알아보도록 하겠습니다.

Convolution과 같은 경우 아래 그림과 같이 빨간색과 파란색 Convolution 영역이 겹치는 부분이 생깁니다.

이러한 점에서 Upsampling 시에 이를 반영하여 Upsampling 될 필요성이 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-008-UpSamplingMethod_03.png)

이러한 방법에서 이를 2차원이 아니라 1차원으로 이러한 복원을 나타낸 그림은 아래와 같습니다.

아래에는 입력에 대해서 Transpose Convolution 으로 Upsampling 하는 방법을 나타냅니다.

아래는 가로 방향의 성분에 대해서만 생각한 그림입니다.

여기서 Convolution의 pad가 1이라고 가정하고 생각합시다.

이때에 az+bx로 3번째 요소가 겹치는 부분을 표현합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-009-UpSamplingMethod_04.png)

아래는 1D에서 Convolution과 Transpose Convolution을 행렬곱 형태로 나타낸 식입니다.

아래의 Convolution은 필터크기 3, stride 2, pad 1인 경우입니다.

여기서 x,y,z 값은 Filter의 값이고, a,b,c,d는 입력값입니다.

여기서 pad를 표현하기 위해서 a의 위와 d의 아래에 0의 값으로 채워 넣은 것을 볼 수 있습니다.

여기서 이러한 방식으로 행렬곱으로 transpose convolution을 할 수 있다는 것만 알면 됩니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-010-UpSamplingMethod_05.png)

---

## Object Detection

물체 인식(Object Detection)은 Classification + Location 두 가지를 모두 필요로 합니다.

아래와 그림과 같이 한 마리의 개를 찾아내고(Classification) 이 개의 위치를 찾아(Location)를 찾아 Bounding Box를 찾습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-011-ObjectDetection.png)

이와 같은 방법은 앞에서 배운 이미지 분류기의 마지막 단에 두가지 갈래로 나누어 작업을 진행합니다.

한가지는 이미지에 대해서 어떤 물체가 있는지 알아내는 FC와 나머지 하나는 이 물체의 위치를 찾아내는 FC로 여러가지 일을 해야합니다.

여기서 일반적으로 앞에서 아용하는 이미지 분류기는 pre-training 된 모델을 사용하고 나머지 2개의 task 에 대해서 학습을 진행합니다.

보통 위에서 classification과 같은 경우 Softmax loss를, Location은 L2 loss로 다른 종류의 loss를 사용합니다.

여기서 한개의 loss가 아니라 2개의 대한 loss가 발생하여 이 Hyperparameter를 조정하는데 어려움이 있다고 합니다.

그래서 이 loss 값으로 성능을 비교하는 것이 아니라 다른 지표(실제 모델 성능 지표 등)을 본다고도 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-012-ObjDetectSingleObject.png)

이제 까지는 하나의 물체에 대한 Detection과 Location에 대해서 배웠습니다.

하지만 실제 상황에서는 한 이미지에 물체가 몇 개인지 예측할 수 없습니다.

이러한 문제를 해결하기 위한 방법이 필요합니다.

---

### Object Detection Sliding Window

> - 이미지를 임의에 크기로 잘라 탐색하며 물체인지, 배경(아무것도 아닌지) 탐색합니다.
> - 단점
>     - 어디에 존재하는지 알지 못함
>     - 몇개 인지 알지 못함
>     - 크기도 알 수 없음
>     - 하나씩 CNN으로 하기에 시간이 부족함

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-013-ObjDetectSlidingWindow.png)

---

### Object Detection Region Proposals

> - 물체가 있을법한 후보(Region proposals)을 찾아내는 것입니다.
> - 물체가 '뭉친'곳을 찾아내 region을 selective search 합니다.
> - 이러한 selective search 방법은 딥러닝 방식이 아닙니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-014-ObjDetectRegionProposals.png)

---

### Object Detection R CNN

> - region proposal 를 이용하는 방법을 바로 R-CNN 이라고 합니다.
> - region proposal network(RPN)으로 region proposal을 얻어냅니다.
> - region proposal은 region proposal of interest(ROI)라고 부르기도 합니다.
> - 여기서 구한 region proposal을 동일한 크기로 warping 해줍니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-015-ObjDetectRCNN_01.png)

> - 이후 Warping 된 이미지 각각을 CNN에 넣어줍니다.
> - 여기서 최종 분류로 R-CNN은 SVM을 사용했습니다.
> - BBox reg는 regional proposal을 보정하기 위한 과정입니다.
> - BBox를 보정할 수 있는 4개의 offset 값을 예측하고 이를 multi-task loss로 한번에 학습합니다.
> - supervised 학습으로 이미지 내의 모든 객체의 BBox가 있어야 학습이 가능하다고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-016-ObjDetectRCNN_02.png)

- R-CNN 문제점

> - multi-task loss로 하이퍼파라미터 설정이 어렵습니다.
> - 학습시간이 오래걸립니다.
> - 계산 비용이 많습니다.
> - region proposal이 학습되지 않습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-017-ObjDetectRCNN_03.png)

---

### Object Detection Fast R CNN

위에 R-CNN의 문제점을 해결하기 위해서 Fast R-CNN이 등장합니다.

> - 모든 image 영역을 CNN에 넣고 feature map을 만듭니다.
> - 전체 이미지에 대한 고해상도 feature map을 얻어낼 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-018-ObjDetectFastRCNN_01.png)

> - 위에서 구해진 feature map 에서 region proposal을 계산합니다.
> - 이 방법은 하나의 CNN에서 나온 feature를 여러 ROI들이 공유하는 효과가 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-019-ObjDetectFastRCNN_02.png)

RoI Pooling 방법은 아래와 같다고 합니다.

찾아낸 Proposal region을 512 x 18 x 8 이라고 한다면

RoI Pooling으로 depth은 변하지 않고, WxH의 크기가 7x7로 줄어드는 것을 볼 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-020-ObjDetectFastRCNN_03.png)

> - 이후 각 ROI을 Roi Pooling layer에 넣어 크기를 조정합니다.
> - 이 크기가 조정된 각 ROI들을 Fully-connected layer에 넣습니다.
> - 이 FC의 출력을 classification 점수(softmax)와 Bbox reg의 linear regression의 값을 얻을 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-021-ObjDetectFastRCNN_04.png)

> - 여기서 나온 2개의 loss 값으로 training 과정을 진행합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-022-ObjDetectFastRCNN_05.png)

> - 제안된 다른 방법보다 Fast R-CNN이 훨씬 빠른 속도를 가진 것을 볼 수 있습니다.
> - 다른 방법 보다는 현저히 빠르지만 Fast R-CNN에 많은 시간을 Region Proposal을 계산하는데 사용됩니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-023-ObjDetectFastRCNN_06.png)

---

### Object Detection Faster R CNN

위에서 Fast R-CNN의 Region Proposal 계산 병목 현상을 제거하기 위해서 Fast**er** R-CNN이 나옵니다.

> - Region Proposal 방법을 딥러닝으로 대체했습니다.
> - 이를 Region Proposal Network(RPN)이라고 합니다.
> - CNN의 feature map을 입력으로 하고 region proposal을 계산합니다.
> - 이후 과정은 fast R-CNN과 같습니다.
> - 위에서는 2개의 loss를 사용했지만 여기는 RPN을 위해 추가적으로 2개의 loss를 사용합니다.
> - RPN 2개의 loss중 1개는 `이곳에 객체가 있는지 판단`합니다.
> - RPN 나머지 loss는 `BBox`에 관한 것입니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-024-ObjDetectFasterRCNN_01.png)

> - 결과적으로 Region Proposal의 계산 병목 현상을 제거한 것을 볼 수 있습니다.
> - 이러한 R-CNN 계열의 네트워크들을 region-based method라고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-025-ObjDetectFasterRCNN_02.png)

---

### Object Detection YOLO and SSD

> - R-CNN의 region-based method의 반해 grid cell 접근 방식으로 region proposal을 사용하지 않았습니다.
> - 예시로 YOLO (You Only Look Once), SSD(Single Shot Detection)가 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-026-ObjDetectYOLOSSD_01.png)

이 아이디어는 각 task의 region proposal에 대해서 따로 계산하는 것이 아니라 하나의 regression 문제로 풀어보자는 것입니다.

먼저 이미지가 주어지면 일정한 간격으로 나눕니다. 아래 그림은 7x7 grid로 나누었습니다.

여기서 하나의 grid cell에 대해서는 base bbox가 있습니다. 아래에는 3가지의 base bbox를 가집니다.

이를 기반으로 실제 위치가 되려면 base box가 얼마나 옮겨져야 하는지 예측합니다.

그리고 각 bbox에 대해서 classification score와 BBox의 offset를 계산합니다.

결국 네트워크는 하나의 이미지에 대해서 7x7 grid cell 마다 (5B + C)개의 값을 가지게 됩니다.

B는 base box의 offset과 박스의 신뢰성을 나타내는 값인 5입니다.

C는 클래스의 개수입니다.

YOLO에 대한 자세한 내용을 알고 싶다면 [박진우님 YOLO 분석](https://curt-park.github.io/2017-03-26/yolo/)을 참고하시길 바랍니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-027-ObjDetectYOLOSSD_02.png)

- 결론
> - YOLO와 SDD는 Single Shot method 라고 한다.
> - 후보 base BBox와 GT(정답 박스)를 매칭하는 것이다.
> - R-CNN은 Regression과 Classification를 따로 풀어나갑니다.
> - YOLO는 이 두가지 문제를 한번의 forward pass로 풀어냅니다. 고로 빠릅니다.

---

### Dense Captioning

그리고 아래는 Object detection과 Image captioning을 조합한 논문입니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-028-DenseCaptioning_01.png)

이를 위해서는 각 Region에 caption이 있는 데이터 셋이 필요합니다.

네트워크는 Faster R-CNN과 비슷하다고 합니다.

Region proposal state가 있고 예측한 BBox 단위로 추가적으로 처리를 한다고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-029-DenseCaptioning_02.png)

아래는 이 Dense Captioning의 한 예시라고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-030-DenseCaptioning_03.png)

---

## Instance Segmentation

마지막으로는 Instance Segmentation 입니다.

이는 Semantic segmentation과 Object detection을 섞은 것입니다.

아래 그림처럼 이미지 내의 두 마리의 개가 있으면 이 두 마리를 구분해내야 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-031-InstanceSegmentation_01.png)

---

### Mask R CNN

이에 대한 방법으로 Mask R-CNN을 소개해줍니다.

> - 위에서 배운 Faster R-CNN과 거의 유사합니다.
> - 마지막에 classification + localization에 추가로 Mask Prediction 관련 네트워크를 병렬로 추가합니다.
> - RoI Pooling 대신 RoI Align을 사용합니다.
> - Mask prediction과 Class prediction을 decouple(Class 상관 없이 Masking)합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-032-MaskRCNN_01.png)

구조는 아래와 같습니다.

> - 이미지를 CNN과 RPN 통과시켜 region proposed 된 feature map을 얻습니다.
> - RoI Align을 통해 256 x 14 x 14의 feature map으로 만듭니다.
> - 각각의 Class C에 대한 mask를 얻습니다. (28x28 크기)

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-033-MaskRCNN_02.png)

---

### RoI Pooling vs RoI Align

- [orisland님](https://mylifemystudy.tistory.com/82) 블로그를 참조했습니다.

위에서 Fast R-CNN 방법에서 사용한 RoI Pooling 방법은 아래 그림과 같습니다.

이 과정에서는 RoI 예측 지점이 소수점으로 나온 경우 각 좌표를 `반올림`하여 사용했습니다.

Fast R-CNN의 주 목적은 Classification이었기 때문에 정확한 위치 정보를 담는 것이 중요하지 않았습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-034-MaskRCNN_03.png)

하지만 Mask R-CNN과 같이 Semantic segmentation과 같은 경우 pixel-by-pixel로 detection이 진행됩니다.

이러한 경우 `정확한 위치 정보`가 필요합니다.

이러한 점에서 소수점으로 나오는 경우 주변의 픽셀정보를 함께 담아 내는 작업이 필요합니다.

이에대한 개선 방안이 바로 RoI Align 입니다.

RoI Align은 아래 그림과 같이 주변의 4개에 픽셀에 대한 정보를 모두 사용합니다.

여기서 bilinear interpolation 방법으로 거리에 따라 다른 비율로 샘플링하는 방법입니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-035-MaskRCNN_04.png)

예시는 아래와 같습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-036-MaskRCNN_05.png)

위에서 기본적인 CNN에서 더 나아가 ResNet과 Feature Pyramid Network 방법을 이용한 Mask 추출을 보여줍니다.

이러한 방법이 좀더 좋은 성능을 보였다고 합니다.

Feature Pyramid Network는 CNN이 깊어질수록 receptive field가 커지고, 이는 좀더 큰 물체에 중점을 두는 것으로 볼 수 있습니다.

이러한 부분에서 중간 중간에 미리 feature에 대한 mask를 구해 다양한 정보를 볼 수 있는 장점을 살린 것입니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-037-MaskRCNN_06.png)

이 Mask RNN의 결과는 아래와 같습니다.

각 물체에 대한 마스크는 다음과 같이 의자, 사람, 침대 등으로 나타낼 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-038-MaskRCNN_07.png)

먼 물체에 대해서도 상당히 좋은 성능을 가지는 것을 볼 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-039-MaskRCNN_08.png)

더 나아가서 관절정보도 찾을 수 있다고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-040-MaskRCNN_09.png)

---

## Aside

이 Masking 외에도 다양하게 발전하고 있습니다.

자세한 내용은 저도 공부가 필요하네요 ㅠㅠ 참고하시길 바랍니다.

- Scene Graphs = Objects + Relationships

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-041-SceneGraphs_01.png)

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-042-SceneGraphs_02.png)

- 3D Object Detection

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-043-3D_Object_Detection_01.png)

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-044-3D_Object_Detection_02.png)

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-045-3D_Object_Detection_03.png)

- 3D Shape Prediction

![](/assets/img/dev/mldl/cs231n/lecture12/cs231n-12-046-3D_Shape_Prediction.png)

---
layout: post
title: < CS213n 정리 > 3. Loss Functions and Optimization
category: dev
permalink: /MLDL/:year/:month/:day/:title/
tags: dev mldl CS213n Stanford
comments: true
---

## 소개
- 이 글은 단지 CS231n를 공부하고 정리하기 위한 글입니다.
- Machine Learning과 Deep Learning에 대한 지식이 없는 초보입니다.
- 내용에 오류가 있는 부분이 있다면 조언 및 지적 언제든 환영입니다!

---
<br><br>


## 참조
- [CS231n Lecture 3. 유튜브 강의](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4&t=0s)

- [Cs231n Lecture 3. 강의 노트](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf)

- [CS231 Lecture 3. 한국어 자막](https://github.com/insurgent92/CS231N_17_KOR_SUB/blob/master/kor/Lecture%203%20%20%20Loss%20Functions%20and%20Optimization.ko.srt)

- [https://cding.tistory.com/2?category=670644](https://cding.tistory.com/2?category=670644)

- [https://wonsang0514.tistory.com/17?category=813399](https://wonsang0514.tistory.com/17?category=813399)

- [https://doromi.tistory.com/111?category=849309](https://doromi.tistory.com/111?category=849309)

- [https://leechamin.tistory.com/85?category=830805](https://leechamin.tistory.com/85?category=830805)

---
<br><br>

## 개요
### < Loss Functions and Optimization >
1. [Image Classification 소개](#image-classification-소개)
2. [Data-Driven Approach](#data-driven-approach)

    2-1) [Nearest Neighbor](#nearest-neighbor)
      + [K-Nearest Neighbor 소개](#k-nearest-neighbor-소개)
      + [K-Nearest Neighbors: Distance Matrix](#distance-matrix)
      + [Hyperparameter](#hyperparameter)
      + [Curse of dimension](#curse-of-dimension)

    2-2) [Linear classification](#linear-classification)
      + [Parametric Approach](#parametric-approach)
      + [Hard cases for a linear classification](#hard-cases-for-a-linear-classification)

---

## Reminder Previous Lecture : Linear Classification

Loss function과 Optimization 에 대한 이야기를 시작하기 전에 간단하게 배운 내용을 상기합시다.

Linear Classification은 아래와 같이 입력 이미지와 W값에 대한 곱으로 생각할 수 있습니다.

또한 이 결과는 Class의 갯수와 같은 Dimension 값을 가집니다.

보통 이 중 값이 가장 큰 값으로 예측(Predict)합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-001-previous_lecture_Linear_Classification.png)

<br><br>

아래 결과는 가중치 W 값을 랜덤으로 설정한 뒤에 3개의 이미지에서 얻은 결과입니다.

이중에 가장 큰 값으로 예측하므로 예측 결과는 다음과 같습니다.

1. 고양이 사진 = 개(dog: 8.02) : `Wrong`
2. 자동차 사진 = 자동차(automobile: 6.04) : `Correct`
3. 개구리 사진 = 트럭(truck: 6.14) : `Wrong`

자, 이제 어떻게 올바른 가중치 W 값을 찾을 것인지에 대해서 알아봅시다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-002-previous_lecture_Linear_Classification_2.png)

---
<br><br>

## Loss Function

지금 제가 가지고 있는 model이 잘 되고 있는지 어떻게 알 수 있을까요?

이것에 대한 해답은 바로 Loss Function에 있습니다.

Loss Function(손실 함수)란, 현재 사용하고 있는 이미지 분류기가 얼마 만큼의 손실을 가지고 있는지를 나타내는 함수입니다.

그림의 우측 아래 부분에 간단한 Loss Function의 식을 확인할 수 있습니다.

어떤 Loss Function를 L<sub>i</sub> 라고 합시다.

여기서 각 N개의 데이터에서 구해지는 손실 값들을 더한 뒤 N으로 나눈, 즉 평균 값을 Loss 값으로 사용하는 것을 볼 수 있습니다.

이제 이러한 L<sub>i</sub> 함수가 어떤 종류가 있는지 살펴보고자 합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-003-Loss_Functions.png)

## Multiclass SVM loss



![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-004-Multiclass_SVM_loss.png)

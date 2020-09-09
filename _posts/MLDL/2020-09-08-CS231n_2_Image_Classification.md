---
layout: post
title: < CS213n 정리 > 2. Image Classification
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

## 참조
- [CS231n Lecture 2. 유튜브 강의](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Cs231n Lecture 2. 강의 노트](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf)
- [CS231 Lecture 2. 한국어 자막](https://github.com/insurgent92/CS231N_17_KOR_SUB/blob/master/kor/Lecture%202%20%20%20Image%20Classification.ko.srt)
- [https://zzsza.github.io/data/2018/05/06/image-classification/](https://zzsza.github.io/data/2018/05/06/image-classification/)
- [https://leechamin.tistory.com/65?category=830805](https://leechamin.tistory.com/65?category=830805)
- [https://doromi.tistory.com/110](https://doromi.tistory.com/110)

---

## 개요
### < Image Classification >
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

## Image Classification 소개

**Image Classification** 이란 말 그대로 **이미지를 입력**으로 받아 이를 **분류**하는 것을 말한다.

사람에게는 이러한 이미지 분류가 쉬워 보이지만 **컴퓨터에게는 생각보다 어려운 작업**이다.

아래 그림은 컴퓨터가 그림을 바라보는 관점을 보여준다.

우리가 보기에는 고양이의 그림이지만, 컴퓨터에게는 그저 <strong><span style="color:red">숫자의 집합</span></strong>이다.

또한 **고양이의 자세, 카메라의 각도, 빛의 세기 등** 과 같은 작은 변화에 숫자 값들은 민감하게 반응하며 우리는 **이러한 다양한 상황을 이겨낼 수 있는 "확장성"을 가진 이미지 분류 알고리즘**을 찾아야한다.

우리는 이러한 것을 해결하는 방법으로 <strong><span style="color:red"> "데이터 중심 접근방법(Data-Driven Approach)" </span></strong>에 대해 알아보도록 하자.

![](/assets/img/dev/mldl/cs231n/lecture02/cs231n-02-001-explain_image_data.png)

---

## Data-Driven Approach

Data-Driven Approach 란 단순히 한 사진 만을 가지고 결정을 하는 것이 아니라 **무수히 많은 사진들과 그 사진에 대한 정보** 를 이용하여 **이미지 분류기를 만들고** 새로운 사진에 대해 분류하는 접근방식을 말합니다.

이러한 접근방식을 위해서 먼저 필요한 것은 다음과 같습니다.
1. **데이터 셋 (Dataset)**
2. **데이터 학습 (Train)**
3. **데이터 판단 (Predict)**

<strong><span style="color:red"> "데이터 셋(Dataset)" </span></strong> 이란 무수히 많은 **데이터** 뿐 아니라 각 데이터에 대한 정보를 나타내는 **라벨(Label)** 이 함께 달린 데이터를 말한다.

<strong><span style="color:red"> "데이터 학습(Train)" </span></strong> 이란 새로운 이미지를 판단하기 위해 데이터 셋을 이용하여 미리 **분류기를 만들고 학습하는 과정**을 말합니다.

<strong><span style="color:red"> "데이터 판단(Predict)" </span></strong> 이란 위에서 학습된 분류기를 이용하여 **새로운 이미지**에 대해 **판단, 분류 및 예측**을 하는 과정입니다.

이제 어떻게 데이터를 이용하여 분류기를 만드는지 소개하도록 하겠다.

![](/assets/img/dev/mldl/cs231n/lecture02/cs231n-02-002-Data_Driven_Approach.png)

---

### Nearest Neighbor

먼저 가장 간단하고 기본적인 분류방법인 **Nearest Neighbor(NN)** 를 먼저 다루도록 하겠다.

<strong><span style="color:red"> "Nearest Neighbor(NN)" </span></strong> 이란 즉 **"가장 가까운 이웃 찾기"**입니다.

이는 정말 직관적인데 **새로운 이미지**와 **이미 알고 있던 이미지** 를 비교하여 <strong><span style="color:red"> 가장 비슷하게 생긴 것을 찾아 내는 것 </span></strong> 을 말합니다.

![](/assets/img/dev/mldl/cs231n/lecture02/cs231n-02-003-Nearest_Neighbor_ex.png)

이러한 작업을 하기 위해서는 다음과 같은 작업이 필요하다.
1. **데이터 학습(Train)**
    + NN 에서는 Train에서는 나중에 새로운 이미지와의 비교를 위해 **단순히 이미지를 저장**한다. `O(1)`
2. **데이터 판단(Predict)**
    + Train에서 가지는 **학습 된 이미지**와 **새로운 이미지**와 **가장 비슷한 이웃을 찾고 그에 따라 분류**한다. `O(N), N: 학습된 데이터의 총 개수`
    + **주변 이웃인지 판단하는 방법**은 뒤에서 더 다루도록 하겠다.

![](/assets/img/dev/mldl/cs231n/lecture02/cs231n-02-004-Nearest_Neighbor_code.png)

<strong><span style="color:red"> @문제점 </span></strong>
  + 이미 알아보신 분도 있겠지만, **데이터 학습은 빠르지만, 새로운 데이터를 판단하는데 있어서 걸리는 시간이 많이 필요하다.**
  + 한 사진에 대해서 너무 많은 시간이 걸린다면 이를 이용하는 사람은 없을 것이다.
  + 이 분류기가 언듯보기에 **쓰지 않을 것을 왜 알아야 하지?** 에 대한 의문을 가질 수 있을 것이다. 하지만 이는 **머신러닝의 기초가 되는 내용 알아가기에 좋은 예** 로 한번 살펴 보도록하자.

---

#### K-Nearest Neighbor 소개

아래 그림은 위에서 보여준 **Nearest Neighbor의 한 예**이다.
이래 그림을 보면 **초록색 부분에 한개의 노란색이 존재하는 것**을 볼 수 있다.
이러한 **1개의 노란색으로 저렇게 판단하는 것이 과연 좋은 선택일까?**
이러한 것은 오히려 새로운 데이터에 대해서 좋지 않은 결과를 가져을 수 있다.
이러한 점에서 이제 **K-Nearest Neighbor** 에 대해 다룰 것이다.

![](/assets/img/dev/mldl/cs231n/lecture02/cs231n-02-003-Nearest_Neighbor_ex.png)

<strong><span style="color:red"> K-Nearest Neighbor </span></strong> 은 **가까운 이웃을 K**개 만큼 찾고, **이웃끼리 투표**를 하는 방법이다.



---

#### Distance Matrix

#### hyperparameter

#### Curse of dimension

### Linear Classification

#### Parametric Approach

#### Hard cases for a linear classification

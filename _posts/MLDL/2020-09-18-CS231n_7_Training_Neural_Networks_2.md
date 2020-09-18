---
layout: post
title: < CS231n 정리 > 7. Training Neural Networks 2
category: dev
permalink: /MLDL/:year/:month/:day/:title/
tags: dev mldl CS231n Stanford
comments: true
---

## 소개
- 이 글은 단지 CS231n를 공부하고 정리하기 위한 글입니다.
- Machine Learning과 Deep Learning에 대한 지식이 없는 초보입니다.
- 내용에 오류가 있는 부분이 있다면 조언 및 지적 언제든 환영입니다!

---
<br><br>


## 참조
- [CS231n Lecture 7. 유튜브 강의](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=7)

- [Cs231n Lecture 7. 강의 노트](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)

- [CS231 Lecture 7. 한국어 자막](https://github.com/visionNoob/CS231N_17_KOR_SUB/blob/master/kor/Lecture%207%20%20%20Training%20Neural%20Networks%20II.ko.srt)


---
<br><br>

## 개요
### < Training Neural Networks 1 >
1. [Activation Functions](#activation-functions)
    1. [Sigmoid 함수](#sigmoid-함수)
    2. [tanh 함수](#tanh-함수)
    3. [ReLU 함수](#relu-함수)
    4. [Leaky ReLU](#leaky-relu)
    5. [PReLU](#prelu)
    6. [ELU](#elu)
    7. [Maxout](#maxout)
    8. [Activation 결론](#activation-결론)
2. [Data Preprocessing](#data-preprocessing)
3. [Weight Initialization](#weight-initialization)
    1. [작은 랜덤값 초기화](#작은-랜덤값-초기화)
    2. [Xavier initialization](#xavier-initialization)
4. [Batch Normalization](#batch-normalization)
5. [Hyperparameter Optimization](#hyperparameter-optimization)

---

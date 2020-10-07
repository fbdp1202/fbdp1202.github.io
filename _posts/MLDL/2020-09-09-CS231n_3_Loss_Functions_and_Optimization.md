---
layout: post
title: < CS231n 정리 > 3. Loss Functions and Optimization
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
- [CS231n Lecture 3. 유튜브 강의](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4&t=0s)

- [Cs231n Lecture 3. 강의 노트](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf)

- [CS231 Lecture 3. 한국어 자막](https://github.com/insurgent92/CS231N_17_KOR_SUB/blob/master/kor/Lecture%203%20%20%20Loss%20Functions%20and%20Optimization.ko.srt)

- [https://cding.tistory.com/2?category=670644](https://cding.tistory.com/2?category=670644)

- [https://wonsang0514.tistory.com/17?category=813399](https://wonsang0514.tistory.com/17?category=813399)

- [https://doromi.tistory.com/111?category=849309](https://doromi.tistory.com/111?category=849309)

- [https://doromi.tistory.com/112?category=849309](https://doromi.tistory.com/112?category=849309)

- [https://leechamin.tistory.com/85?category=830805](https://leechamin.tistory.com/85?category=830805)

- [https://www.stand-firm-peter.me/2018/09/24/l1l2/](https://www.stand-firm-peter.me/2018/09/24/l1l2/)

---
<br><br>

## 개요
### < Loss Functions and Optimization >
0. [Reminder Previous Lecture](#reminder-previous-lecture)
1. [Loss Function](#loss-function)
    1. [Multiclass SVM loss](#multiclass-svm-loss)
    2. [Regularization](#regularization)
    3. [Softmax Classifier](#softmax-classifier)
2. [Optimization](#optimization)
    1. [Random search 임의 탐색](#random-search-임의-탐색)
    2. [local geometry 경사 하강법](#local-geometry-경사-하강법)
    3. [Stochastic Gradient Descent](#stochastic-gradient-descent)
3. [특징변환](#특징변환)
    1. [컬러 히스토그램](#컬러-히스토그램)
    2. [Histogram of Oriented Gradient](#histogram-of-oriented-gradient)
    3. [Bag of Words](#bag-of-words)
4. [결론](#결론)

---

## Reminder Previous Lecture

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

한가지 간단한 예시로 Multiclass SVM loss에 대해서 알아봅시다

Multiclass SVM loss에 대한 설명을 하자면 아래와 같습니다.

- Linear Classification인 **f(x<sub>i</sub>, W)** 에서 나온 결과를 **s**라고 합시다.

- 자 정답인 클래스의 s 값을 **s<sub>y<sub>i</sub></sub>** 라고 하고 나머지 클래스와 이 값을 비교합시다.

- 정답인 클래스와 나머지를 비교했을때, 정답보다 다른 클래스의 점수가 더 높다면 이 차이 만큼이 Loss 라고 정합시다.

- 위에서 구한 loss 에서 **safety margin** 이라는 값을 추가합시다. 이는 정답 클래스가 적어도 다른 클래스보다 **safety margin** 값 만큼은 커야 한다는 이야기이며, 여기서는 `safety margin = 1` 입니다.

- 이 loss 값이 0보다 작은 음수 값인 경우에는 포함하지 않습니다.

- 아래에 그래프에 이를 표현하였고, 가로축은 **s<sub>y<sub>i</sub></sub>** 값, 세로축은 **L<sub>i</sub>** 의 값인 loss 값을 표현하였습니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-004-Multiclass_SVM_loss.png)

<br><br>

이제 수식이 아닌 간단한 예로 설명하도록 하겠습니다.

아래 그림에서 3개에 그림에 대해서 Linear classification 결과가 아래와 같다고 합시다.

이 중에 먼저 고양이 사진에 대한 Loss Function를 계산해봅시다.

일단 고양이는 3.2의 값을 가지고 있고 나머지 자동차와 개구리에 대한 Loss 값은 아래와 같습니다.

- Loss(class: `cat`) = 0

- Loss(class: `car`) = max(0, 5.1 - 3.2 + 1) = 2.9

- Loss(class: `frog`) = max(0, -1.7 - 3.2 + 1) = 0

- 고양이 사진의 전체 Loss = 0 + 2.9 + 0 = 2.9

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-005-Multiclass_SVM_loss_example_1.png)

<br><br>

위와 같은 방법을 나머지 두 사진에 대해서 반복하면 아래와 같은 결과를 가집니다.

이 Loss 값을 모두 더하고 사진 개수만큼 나눈 값을 Multiclass SVM loss 입니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-006-Multiclass_SVM_loss_example_2.png)

<br><br>


<strong><span style="color:red">@Multiclass SVM loss의 특징</span></strong>

- 정답 스코어가 `safety margin=1값`을 만족하는 범위라면 loss값에 변화를 주지 않습니다.

- 만약 loss 값이 0이라고 할때, `W: weight parameter` 값에 임의에 상수를 곱해도 loss값은 0입니다.

- 위 점에서 Multiclass SVM loss는 유일하지 않습니다.

- 모든 S 값이 0이라고 하면 `loss = Class수 - 1`값을 가집니다. 이러한 특성은 loss function의 작동을 Debug 하기에 적절합니다.

- loss의 최솟값은 0 이며, 최대값은 무한대입니다.


---
<br><br>

## Regularization

자 이제 위에서 배운 loss function에 대해서 생각해 봅시다.

과연 위에 있는 loss function 값이 줄어들 수록 model이 좋은 성능을 가지는 걸까요?

정답은 좋을 수도 있고 아닐 수도 있습니다. 무슨말이냐고요?

loss가 줄어준다는 것은 `Train`데이터에 대해서는 좋은 성능을 가지게 됩니다.

하지만 이것이 `test`데이터에서 좋은 성능을 가질 지는 알 수 없습니다.

이것에 대한 이야기가 바로 <strong><span style="color:red">overfitting(과적합)</span></strong>입니다.

**overfitting(과적합)**이란 `train`에 대해서는 좋은 성능을 가지도록 학습되지만, `test`데이터에 대해서는 오히려 성능이 떨어지는 현상을 말합니다.

딱 `train`데이터에 대해서만 과적합 현상이 발생한 것이죠.

자 이것을 이제 어떻게 해결할 것인가? 그것이 바로 Regularization입니다.

아래 그림의 수식을 보시면 Data loss 부분 옆에 `Regularization` 수식이 추가된 것을 확인 할 수 있습니다.

이 Regularization은 W 값에 대해서 제약을 가합니다.

아래 그림의 그래프를 보시면 `파란색은 train` 데이터이며, `초록색은 test` 데이터 입니다.

여기서 우리가 원하는 함수의 형태는 초록색 선이지만 예측한 파란색 선은 `train`데이터에 overfitting 되어 있는 것을 볼 수 있습니다.

이러한 부분에서 우리는 W의 값이 복잡한 값을 가지지 않게 하기 위해서 `W 값에 대해서 평가하는 함수를 추가`하는 것이며 이것을 `Regularization`이라고 합니다.

- "weight decay" : 특정 가중치가 비이상적으로 커지는 것을 방해하는것 이다.
- "local noise"와 outlier(특이점)의 영향을 적게 받도록 한다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-007-Regularization.png)

<br><br>

아래에는 Regularization 에서 사용하는 function의 예시를 보여줍니다.

먼저 어느정도로 Regularization 값을 사용할지를 정하는 `λ: Regularization strength`

이후 `R(W)의 R`이 바로 Regularization 함수입니다.

종류는 `L2 regularization`, `L1 regularization`, `Elastic net`, `Dropout` 등 다양한 Regularization 방법이 있습니다.

하지만 여기서는 깊게 다루지 않고 간단하게 L2와 L1에 대해서만 살펴보고자 합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-008-Regularization_types.png)

<br><br>

자 아래는 `L1 regularization`과 `L2 regularization`의 차이를 설명하기 좋은 예시입니다.

어떤 Regularization를 사용하는지에 따라서 weight의 모습이 다른 모습을 가집니다.

아래 2개의 weight 값 예시를 봅시다.

ω<sub>1</sub>와 ω<sub>2</sub>는 x에 대해서 같은 Score 값을 가집니다.

하지만 weight의 분포 형태는 다릅니다.

여기서 L1 regularization를 적용 했을 때에는 두 weight 값은 같은 loss 값을 가집니다.

하짐나 L2 regularization를 적용하면 ω<sub>1</sub>와 같은 경우는 1의 값을 가지지만,

ω<sub>2</sub>와 같은 경우 `4 x (1/4)^2 = 1/4`의 loss 값을 가집니다.

보통 이 L1 regularization과 L2 regularization의 형태를 간단하게 설명하면 아래와 같습니다.

- L1 regularization
    + weight 값이 0으로 수렴하는 것이 많은 형태이다. 이를 Sparse matrix(희소 행렬)이라 부릅니다.
    + 위에서 0의 값이 많다는 이야기는 어떤 특징들은 무시하겠다는 이야기로 볼 수 있습니다.

- L2 regularization
    + weight의 값이 큰 값은 점점 줄이며 대부분의 값들이 0의 가까운 값을 가지는 가우시안 분포를 가집니다.
    + weight이 0이 아니라는 점에서 모든 특징들을 무시하지 않고 조금씩은 참고 하겠다라고 볼 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-009-Regularization_example.png)

- Solution uniqueness & Computational efficiency
    + Taxicab geometry
![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-009-Regularization_Computational_efficiency.png)
<br><br>

- Sparsity & Feature Selection
    + Feature Selection
    + Convex Optimization에서 유용하게 사용

L1과 L2의 특징
- L1
    + Unstable Solution
    + Always on Solution
    + Sparse Solution
    + Feature Selection
- L2
    + Stable Solution
    + Only one Solution
    + Non-sparse Solution

---
<br><br>

## Softmax Classifier

이제 Multi SVM 이외에 다른 loss function인 Softmax Classifier에 대해서 알아보도록 합시다.

Softmax Classifier는 Multinomial Logistic Regression 이라고도 부릅니다.

Softmax Classifier는 위 Multi SVM과 다르게 logistic 한 값을 사용하며

Multi SVM에서는 정답 score의 값이 다른 값보다 safety margin 값 보다 크면 됬다고 한다면,

이 Softmax는 `정답 값의 분포 비중`을 보고 loss 값을 결정합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-010-Softmax_Classifier.png)
<br><br>

자 이제 Softmax Classifier가 어떻게 작동하는지 간단한 예시를 봅시다.

아래는 Softmax Classifier의 loss 값을 가지는 과정을 보여줍니다.

1. score 값에 `exponental`를 취합니다.
2. 이 값에 대한 `확률 분포 값`을 구합니다.
3. 이후 정답 값을 제외한 나머지 class에 대해서 `-log`를 취하여 더해줍니다.
    - 여기서 -log를 취하는 이유는 먼저 log 값을 취하면 점수가 높을 작은 값을 가집니다.
    - 하지만 우리는 loss 는 얼마나 잘못 되었는지를 확인하는 작업으로 - 를 곱하여 사용합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-011-Softmax_Classifier_example.png)
<br><br>

아래는 SVM과 Softmax의 차이를 보여줍니다.

여기서 SVM과 같은 형태를 `hinge loss`라고 부르며 (그래프의 형태가 hingle 모양이여서 붙은 이름이라고 합니다.)

Softmax와 같은 경우 `cross-entropy loss`라고 부른다. (아마도 전체적인 entropy 값을 보고 결정하는 loss 값이기 때문에 이러한 이름이 붙은 것 같습니다.)

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-012-Softmax_Classifier_example_2.png)
<br><br>

## Optimization

자 위에서 가지고 있는 weight 값이 얼마나 잘못 되어 있는지에 대해서 SVM과 Softmax와 같은 loss function으로 확인한다고 했습니다.

이제 얼마나 잘못 되었는지 알았다면 어떻게 좋은 weight 값을 찾아 가야 할지를 다루는 Optimization에 대해서 다루려고 합니다.

일단 Optimization에 앞서 간단한 예시를 봅시다.

아래 그림은 사람이 산에서 아래로 내려가기 위한 길을 찾고 있는 상황입니다.

하지만 평소와 다르게 `눈을 감은 상태에서 산을 내려오려면` 어떻게 해야할까요?

한번 생각해봅시다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-013-Optimization_desc_fig.png)
<br><br>

### Random search 임의 탐색

첫번째 방법은 아무 생각없이 Random search 하는 방법입니다.

그저 발이 아무런 기준없이 weight 값을 변경해보며 가장 좋은 성능을 가지는 weight를 찾는 것이죠.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-014-Optimization_random_1.png)
<br><br>

이와 같은 방법을 사용하면 약 15.5%의 성능을 가지는 것을 볼 수 있습니다.

그냥 찍는 것보다는 나쁘지 않아 보이지만, 현대 최고 성능인 SOTA가 95%인 점을 보면

많이 문제가 있어 보이네요.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-015-Optimization_random_2.png)
<br><br>

### local geometry 경사 하강법

자 다음 방법으로는 경사를 이용해 보는 것입니다.

땅 주변에 발을 가져다 대 보면서, 좀 더 낮은 곳으로 조금씩 움직이며 더욱 낮은 곳으로 내려가 보는 것입니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-015-slope_optimization_1.png)
<br><br>

자 이제 실제로 weight 값에 기울기를 어떻게 알 수 있을까요?

바로 `미분`입니다.

수학이 나와서 어지러우시다고요?

저희가 이것을 실제로 구현할 일은 없고, `이러한 방법으로 기울기를 구하는 것이구나` 정도만 이해하셔도 됩니다.

이제 대충 미분을 쓸 것이고 아래 식 `( f(x+h) - f(x) )/h`라는 공식을 써볼 거다 라는 것만 알아둡시다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-016-slope_optimization_2.png)
<br><br>

자 먼저 이 f(x+h)를 구하기 위해서 `h의 값을 0.0001` 이라고 합시다.

이 W의 한 값을 변화 시켰을때 얻은 loss 값이 1.25322로 변화했다고 합시다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-017-slope_optimization_3.png)
<br><br>

이 loss 값의 차이를 h로 나눈 값이 `gradient dW`의 값입니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-018-slope_optimization_4.png)
<br><br>

이러한 방식으로 모든 W에 대해서 이를 반복하면 모든 gradient dW 값을 구할 수 있습니다.

이 방식을 `Numerical gradient`라고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-019-slope_optimization_5.png)
<br><br>

이러한 방식은 각 모든 W 성분에 대해서 계산되어야 하기 때문에 굉장히 느리다는 단점이 있습니다.

이러한 문제를 뉴턴과 라이프니치가 수학적으로 저 위에 미분을 간단하게 할 수 있게 해주었습니다.

이러한 수학적 접근 방식을 `Analytic gradient`라고 합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-020-slope_optimization_6.png)
<br><br>

자 이제 이 두 가지의 차이점을 정리해봅시다.

- `Numerical gradient`
    + 근사값을 구함, 느리다, 코드짜기 쉽다.
- `Anaytic gradient`
    + 정확한 값을 구함, 빠르다, 코드짜기 어렵다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-021-slope_optimization_7.png)
<br><br>

이제 gradient dW 값을 구했다고 합시다.

우리는 올라가는 방향이 아니라 내려가는 방향을 찾아야 하기 때문에 `-` 값을 곱해줍시다.

또한 여기서 `step_size(learning rate)`라는 새로운 `Hyperparameter`를 사용합니다.

이 `step_size(learning rate)`란 산을 내려가는 사람으로 비유하자면 `걸음의 폭`이라고 생각하시면 됩니다.

방향을 정하고 1m를 갈지, 10m를 갈지를 정하는 것이라고 생각하시면 됩니다.

여기서 이 값을 너무 작게 주면 학습이 너무 느리게 된다는 단점이 있으며,

너무 큰 값을 주게 되면 제대로 수렴하지 못하게 됩니다.

이러한 점에서 이 `step_size(learning rate)`값을 잘 정해주어야 합니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-022-learning_rate.png)
<br><br>

## Stochastic Gradient Descent

이제 `Stochastic Gradient Descent(SGD)`에 대해서 알아보도록 합시다.

우리가 Gradient Descent를 구하기 위해서는 loss function이 필요합니다.

여기서 이 loss function은 전체 트레이닝셋 loss의 평균으로 사용했습니다.

그렇다면 시간이 매우 오래 걸리게 되겠죠.

그래서 실제로는 **Stochastic Gradient Descent (SGD)**라는 방식을 사용합니다.

이 방식은 전체 데이터 셋의 gradient를 구하는 것이 아니라 **Minibatch**라는 작은 트레이닝 샘플로 나누어서 학습하는 방식입니다.

보통 2의 승수로 정하며 **32,64,128**로 쓰는 편입니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-023-Stochastic_Gradient_Descent.png)
<br><br>

## 특징변환

사실 Linear Classification은 이미지에서는 그리 좋은 방법이 아닙니다.

그래서 DNN이 유행하기 전에는 Linear Classifier를 이용하기 위해서는 두가지 스테이지를 거쳐서 사용했습니다.

1. 이미지의 여러가지 특징표현을 계산
    + 모양새, 컬러 히스토그램, edge 형태와 같은 특징표현을 연결한 특징벡터
2. 이 특징벡터를 Linear Classifier에 입력값으로 사용

특징변환에 간단한 예를 보도록 하겠다.

<br><br>

### 컬러 히스토그램

각 이미지에서 `Hue`값만 뽑아서 모든 픽셀을 양동이에 넣고 `각 양동이에 담긴 픽셀의 갯수`를 세는 것입니다.

개구리와 같은 경우 초록색이 많은 것을 알 수 있습니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-024-Color_Histogram.png)
<br><br>

### Histogram of Oriented Gradient

이번에는 이미지를 `8*8`픽셀로 나눠서 각 픽셀의 지배적인 edge 방향을 계산하고 각 edge들에 대해서 양동이에 넣는 것입니다.

그럼 edge에 대한 히스토그램이 되는 것입니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-025-HoG.png)
<br><br>

### Bag of Words

이 방법은 NLP에서 영감을 받은 방식으로, 어떤 문장에서 여러 `단어들의 발생빈도`를 세서 특징벡터로 사용하는 방식을 이미지에 적용한것입니다.

우리는 이미지들을 `임의대로 조각내고`, 각 조각을 `K-means`와 같은 알고리즘으로 군집화 합니다.

다양하게 구성된 각 군집들은 다양한 색과 다양한 방향에 대한 edge도 포착할 수 있습니다.

이러한 것들을 시각 단어(visual words) 라고 부릅니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-025-BoW.png)
<br><br>

## 결론

- Weight 값을 판단하는 `loss function`에 대해서 배웠습니다.
    + SVM 과 Softmax 와 같은 loss function을 이해했습니다.
- Loss function을 Optimization하는 `Gradient Descent`을 배웠습니다.
    + Gradient Descent의 의미와 `learning rate`에 대해 이해했습니다.
- 이 Gradient Descent의 단점을 보안하는 `Stochastic Gradient Descent (SGD)`에 대해서 배웠습니다.
    + 이를 이용해 `Minibatch`에 대한 개념을 이해했습니다.
- Linear Classification에서 사용하는 `특징추출`에 대해 배웠습니다.
    + `컬러 히스토그램`, `Histogram of Oriented Gradient`, `Bag of Words`에 대해 배웠습니다.


이제 우리는 실제로 이미지에서 많이 사용하는 DNN 과 CNN에 대해서 배울 것입니다.

이는 매우 비슷하지만 다른점이 있다면 **이미 만들어 놓은 특징을 쓰기 보다는 데이터로부터 특징을 직접 학습**하려 한다는 점입니다.

다음시간에는 CNN에 대해 살펴볼 것이고 역전파(Backpropagation)에 대해서 알아보겠습니다.

![](/assets/img/dev/mldl/cs231n/lecture03/cs231n-03-026-Difference_Conv_Image_features.png)

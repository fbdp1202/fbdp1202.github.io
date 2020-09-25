## FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks


### 0. Abstract

< Major contributions >
- we focus  on  the  training  data  and  show  that  the  schedule  of presenting data during training is very important.
> 우리는 트레이닝 데이터에 초점을 맞췄고, 트레이닝 중에 보여지는 데이터의 스케줄을 정해주는 것이 매우 중요하다.

- we  develop  a  stacked  architecture  that  includes  warping of the second image with intermediate optical flow.
> 우리는 두 번째 이미지의 뒤틀림을 포함하는 스택형 아키텍처를 개발한다.

- we elaborate on small displacements by introducing a subnetwork specializing on small motions.
> 우리는 작은 동작들을 전문으로 하는 서브 네트워크를 도입함으로써 작은 변위들에 대해 상세히 기술한다.


### 1. Introduction

기존의 FlowNet은 간단한 CNN을 이용하여 optical flow의 개념을 바로 데이터로 학습하는 혁명적인 아이디어였다.

하지만, 이 아이디어는 이미 잘 만들어진 기존 기법들 보다 안좋았다.

동시에 연속적 통합(Successive consolidation)으로 부정적인 영향을 해소하고 새로운 사고방식의 이점을 높였다.

결국 작은 결함과 인공물의 잡음 문제를 해결하였다.

이러한 FlowNet 2.0의 등작으로 실제로 동작 인식과 모션의 Segmentation에 많은 발전을 이끌었다.

이 FlowNet2.0은 몇몇의 발전을 도모했습니다.

1.  First, we evaluate the influence of dataset schedules.  Interestingly, the more sophisticated training data provided by Mayeret al. [19] leads to inferior results if used in isolation.  However, a learning schedule consisting of multiple datasets improves results significantly. In this scope, we also found that the FlowNet version with  an  explicit  correlation  layer  outperforms  the  version without such layer. This is in contrast to the results reported in Dosovitskiyet al. [11].
> 첫번째로, influence 시에 데이터 셋를 평가한다. 여기서 Mayeret al가 제공한 데이터는 좋은 데이터지만 이것만 사용하면 오히려 고립되어 열등한 결과를 초래합니다. 반면, 여러 데이터 세트를 학습 스케줄하여 좋은 성능을 가질 수 있었다. 또한 FlowNet은 layer간에 명시적 상관관계(explicit correlation) 계층을 가진 버전이 좋은 서능을 뛰었다.

2. As a second contribution, we introduce a warping operation and show how stacking multiple networks using this operation can significantly improve the results.  By varying the depth of the stack and the size of individual components we  obtain  many  network  variants  with  different  size  and runtime. This allows us to control the trade-off between accuracy and computational resources.  We provide networks for the spectrum between 8fps and 140fps.
> 두번째로, 우리는 비틀림 작업을 도입하여, 여러 네트워크를 쌓아올리는 방법을 도입했다. stack의 깊이와 각각의 요소들의 크기를 변화시켜 실시간으로 다른 크기와 네트워크의 변형을 얻는다.

3. Finally,  we  focus  on  small,  sub-pixel  motion  and  real-world data. To this end, we created a special training dataset and a specialized network.  We show that the architecture trained  with  this  dataset  performs  well  on  small  motions typical for real-world videos. To reach optimal performance on arbitrary displacements, we add a network that learns to fuse  the  former  stacked  network  with  the  small  displacement network in an optimal manner.The final network outperforms the previous FlowNet by a large  margin  and  performs  on  par  with  state-of-the-art methods on the Sintel and KITTI benchmarks.   It can estimate small and large displacements with very high level of detail while providing interactive frame rates.
> 마지막으로 우리는 작거나, 보조 픽셀 음직임과 실상 데이터에 초점을 맞추었다. 먼저 실시간의 동영상을 통한 작은 음직임 학습한 소변위 네트워크와 이전 내트워크를 융합하는 방법을 추가하였다.

> 결국 이전 FlowNet 성능을 대폭 향상하고, Sintel과 KITI과 같은 첨단 기술과 대등한 성능을 가진다. 또한 소통가능한 frame rates(8fps~120fps)로 작은 움직이 변위와 큰 움직임에 대해서 측정이 가능하였다.

### Related Work
생략

### Dataset Schedules
우리는 학습에 있어서 데이터의 순서가 성능의 영향을 주는 것을 확인하였다. 이러한 점에서 우리는 양질의 데이터를 가지고 이에 대한 순서를 정해 주었다.

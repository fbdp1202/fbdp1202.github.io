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

1.  First, we evaluate the influenceof dataset schedules.  Interestingly, the more sophisticatedtraining data provided by Mayeret al. [19] leads to inferior results if used in isolation.  However, a learning schedule consisting of multiple datasets improves results significantly. In this scope, we also found that the FlowNet versionwith  an  explicit  correlation  layer  outperforms  the  versionwithout such layer. This is in contrast to the results reportedin Dosovitskiyet al. [11].

2. As a second contribution, we introduce a warping operation and show how stacking multiple networks using thisoperation can significantly improve the results.  By varyingthe depth of the stack and the size of individual componentswe  obtain  many  network  variants  with  different  size  andruntime. This allows us to control the tradeoff between accuracy and computational resources.  We provide networksfor the spectrum between 8fps and 140fps.

3. Finally,  we  focus  on  small,  subpixel  motion  and  realworld data. To this end, we created a special training datasetand a specialized network.  We show that the architecturetrained  with  this  dataset  performs  well  on  small  motionstypical for realworld videos. To reach optimal performanceon arbitrary displacements, we add a network that learns tofuse  the  former  stacked  network  with  the  small  displacement network in an optimal manner.The final network outperforms the previous FlowNet bya  large  margin  and  performs  on  par  with  stateoftheartmethods on the Sintel and KITTI benchmarks.   It can estimate small and large displacements with very high levelof detail while providing interactive frame rates.

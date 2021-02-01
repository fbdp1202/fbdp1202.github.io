---
layout: post
title: End-to-End Neural Speaker Diarization with Self-Attention
category: dev
permalink: /PaperToMath/:year/:month/:day/:title/
tags: dev papertomath Diarization EEND
comments: true
---

## 소개
- 이 글은 논문을 읽고 정리하기 위한 글입니다.
- 내용에 오류가 있는 부분이 있다면 조언 및 지적 언제든 환영입니다!

---

2019 ASRU에 올라온 논문입니다. ([Paper](https://ieeexplore.ieee.org/abstract/document/9003959/), [github](https://github.com/hitachi-speech/EEND))

## Citation
- Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Yawen Xue, Kenji Nagamatsu, Shinji Watanabe, " End-to-End Neural Speaker Diarization with Self-attention," Proc. ASRU, pp. 296-303, 2019

---

## Introduction

#### # 기존 speaker diarization system
- 먼저 짧은 단위에 음성 segment를 만듭니다.
- 이 segment를 neural network에 입력으로 넣어 `speaker embedding`를 생성합니다.
- 위 speaker embedding을 `clustering algorithms`으로 speaker를 clustering 합니다.
> Clustering Algorihtm은 Gaussian mixture models, Agglomerative Hierarchical Clustering(AHC), k-means Clustering 등이 있음

#### # 기존 방식의 한계점
`1. Diarization error를 곧바로 최적화 하지 못한다!`

> 많은 경우 clustering은 Unsupervised learning methods 이다.

`2. Speaker overlap 상황을 다루지 못한다!`

> Segment 단위로 speaker embedding이 생성하기 때문에, clustering algorithm은 발화자가 한명이라 가정한다.

#### # Self-Attentitive End-to-End Neural Diarization (SA-EEND)
이 논문에서는 이전 논문이었던 BLSTM 기반 EEND를 Self-Attentitive 기법을 이용해 개선한 SA-EEND를 제안합니다.
이 방법은 Clustering 기법을 사용하지 않으며, Permutation-free loss를 이용하여 end-to-end로 speaker diarization를 학습합니다.
이에 대한 자세한 내용은 차즘 다루도록 하겠습니다.
결과적으로 이 방법은 기존에 가장 좋은 성능을 가지던 x-vector 보다 좋은 성능을 보입니다.

---

## Related Work

#### # x-vector speaker diarization system

1. 입력 소리에 대한 MFCC feature를 생성합니다.
2. Speech activity detection (SAD) Neural에 MFCC 입력을 주어 SAD 영역을 찾아냅니다.
3. SAD network에서 구해진 영역을 short segment 한 뒤, x-vector network로 speaker embedding를 추출합니다.
4. 화자들에 대한 Covariance Matrices 를 이용하여 PLDA(Probabilitic Linear Discriminant Analysis) Scoring을 진행합니다.
5. 마지막으로 AHC(Agglomerative Hierarchical Clustering)으로 diarization error를 계산해냅니다.

#### # EEND speaker diarization system

- 뉴럴넷 기반 end-to-end 접근으로 처음부터 끝까지 한번에 출력과 학습이 진행됩니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--001-Speaker_Diarization_System.png)

---

## Proposed Method

#### # Self-Attention EEND 구조

`1. 입력 소리에 대한 log mel feature를 추출합니다.`
> - Window size: 25ms, Hop size: 10ms, 23-dim log Mel-filterbanks
> - Subsample: 이전 7개, 현재 1개, 미래 7개 frame인 logmel feature를 사용하여 23 x 15 dim
> - 100ms 인 10 frame 마다 network의 subsample를 입력으로 넣어줌

`2. Linear (Fully connected layer)`
> - logmel dim F=23를 Encoder의 D=256 dim으로 project 시킨다.
> - 결국, F x T (23 x 15) -> D x T (256 x 15)

`3. p개의 Encoder Block (논문에선 2개)`

`3-1. Layer Normization`
> - 보통 학습 속도를 빠르게 해준다고 합니다.

`3-2. Multi-head self-attention`
> - Transformer에 사용하는 Multi-head Self-attention의 encoder 부분과 같습니다.
> - 하지만 위 Transformer에서 사용한 positioning encodering은 하지 않았습니다.
> - Query와 Key 값을 같은 encoder 입력값으로 넣어줍니다.
> - Header는 4개 사용하였습니다.
> - 출력과 입력은 residual 해줍니다.

`3-3. Position-wise FF`
> - 2개의 Linear 함수가 들어가있는 형태입니다.
> - 기존 D=256 dim에서 d_ff=1024로 projection 한 뒤에 다시 D=256 dim 형태로 다시 바꿔줍니다.
> - 마지막 출력은 이전과 같이 residual를 해줍니다.

`3-4. p개 encoder 반복`
> - encoder에서 나온 출력을 다음 encoder의 입력으로 넣어주며, 이를 p번 반복합니다.

`4. Linear + Sigmoid`
> - Layer Normization 이후, Linear 와 Sigmoid를 취해줍니다.
> - Linear는 Encoder dim인 D에서 speaker 개수인 S로 project 합니다.
> - 이후 0~1 사이의 speaker 확률 값을 표현하는 sigmoid함수를 사용합니다.

`5. Permutation-free loss`
> - `label ambiguity`: 화자의 순서가 맞은 상태라면 맞은 것으로 하자! 입니다.
> - 아래 모델 구조의 Permutation-free loss 부분을 살펴보면 Permutation 1과 2를 볼 수 있을 것입니다.
> - 각 두 화자에 대한 만들수 있는 모든 permutation 들 중 정답과의 최소 Binary Cross Entropy(BCE) loss 값을 사용하는 것을 Permutation-free loss 라고 합니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--002-Permutation_Free_Loss.png)

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--003-SA_EEND_System.png)

---

## Experimental Setup

#### # Data

`1. Simulated mixtures`
> - 여기서는 두 화자에 대한 overlap 상황을 mixture 기법을 이용하였습니다.
> - mixture란 간단히 두 음성 데이터를 더하는 방식을 말합니다.
> - 여기서는 Beta 값을 이용하여 한 화자가 말한 발화 간의 거리를 random으로 분포시킵니다.
> - 이때 Beta 값이 작으면 작은 간격을 가져, 두 화자를 합쳤을때 겹쳐질 확률일 올라가 전체 overlap 값이 증가합니다.

> ##### 데이터 종류
>> - Switchboard-2 (Phase I, II, III), Switchboard Cellular (Part 1, Part 2) `(SWBD)`
>> - NIST Speaker Recognition Evaluation datasets (2004-5-6-8) `(SRE)`
>> All dataset `telephone speech` sampled 8kH
>> 전체 화자수 6,381명

> ##### Method
>> - 37개 background noises `MUSAN` dataset
>> - 10,000개의 Room Impulse Responses (RIRs)
>> - SNR 10dB, 15dB, 20dB
>> - 화자수 2명, 총 발화 수 최소 10개, 최대 20개 (10~20 사이 random 선택)

`2. Real datasets`
> - 26,172개의 두화자간 대화 셋
> - SWBD + SRE
> - CALLHOME
> - Corpus of Spontaneous Japanese (CSJ)

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--004-Datsets.png)

#### # SA EEND system

`- Model Hyper Parameter`
> - 오디오 음성 최대 길이: 50s
> ##### Encoder 속성
>> - Encoder 개수: 2개
>> - Encoder dimension: 256 dim
>> - Multi-header 개수: 4개
> ##### position wise feed-forward layer
>> - d_ff = 1024
> ##### Training
>> - batch Size: 64
>> - warm up learning rate scheduler: 25k
>> - First 100 echo - 일반적인 학습
>> - After 100 echo - 이전 10 epoch 있는 model parameter 값을 averaging (BLSTM-EEND의 11-frame median filtering과 비슷하게 만들고자). 사실 같은 의미인지 모르겠다... 그냥 성능 더 좋은듯
>> - learning rate: 10^-3, CALLHOME adaptation set: 10^-5
>> - Optimizer: Adam optimizer

---

## Results

- Diarization Error Rate (DER) 기준으로 평가하였습니다.
- 기존 Miss 와 False alarm 에 대한 결과는 사전에 연구에도 다루지 않았습니다.
> 이는 유성/무성음 라벨을 사용하기 때문에 라고함... 자세히 모름

#### # 전체 Test 평가

- Beta 값인 overlap 정도에 따라서 성능차이를 볼 수 있습니다.
- 기존 SOTA 모델인 x-vector 보다 모든 데이터에 대해서 좋은 결과를 가지는 것을 볼 수 있습니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--005-Total_Results.png)

#### # CALLHOME domain adaptation 결과

- 모든 학습 이후에 CALLHOME 데이터를 이용한 domain adaptation 결과를 보여줍니다.
- domain adaptation 이 성능에 많은 영향을 미치며 BLSTM 모델보다 좋은 성능을 보입니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--006-CALLHOME_Adapation_Results.png)

#### # 최종 Test 결과

- DER과 SAD error를 모두 가진 결과를 보여줍니다.
- 기존 방식인 i-vector와 x-vector의 성능과, 제안된 SA-EEND의 성능을 보여줍니다.
- SA-EEND는 DER 관점에서는 가장 좋은 성능을 보이며 MI 와 CF 에 대해서 좋은 성능을 보입니다.
- SA-EEND는 overlap 를 찾다보니 FA이 기존 model 보다 늘어난 모습을 보였습니다.
- SAD 관점에서는 아직 SA-EEND가 부족한 모습을 보입니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--007-Final_Results.png)

#### # Visualization of self-attention

- 이 SA-EEND의 2번 encoder block은 4개의 head 를 가지고 있습니다.
- 이 head는 각 query에 대해서 다른 key 값 형태를 보이는 것을 아래에서 볼 수 있습니다.
- Header 1, 2
> - 먼저 1,2 head 같은 경우 각 쿼리 위치에 대해서 같은 key 값을 가지며 head 1은 spk 1의 발성 부분에, head 2는 spk 2의 발성 부분의 weight 값을 가집니다.
> - 이 1,2 head는 각 query에 대한 같은 key값으로 weighted mean과 같이 작동합니다.
> - 이는 `global speaker의 특징`을 대표하고, 두 화자를 분리하는 것으로 볼 수 있습니다.

- Header 3, 4
> - 3,4 head와 같은 경우 Query와 Key 값에 대해서 `identity matrices`와 같이 보이며, 이는 `position-independent linear transforms`과 같이 작용합니다.
> - 여기서는 이 head의 역활을 `speech/non-speech detection`으로 유추하지만, 정확한 실험이 더 필요해 보입니다.

![](/assets/img/dev/papertomath/summary/SA_EEND/summary--008-Visualization_Self_Attention.png)

#### # Future Work

- Multi-head self-attention 의 head 수와 화자 수 간에 관계성 실험!
- 다양한 Encoder 개수에 대한 추가 실험 필요! (다음 연구에서 다뤄짐)

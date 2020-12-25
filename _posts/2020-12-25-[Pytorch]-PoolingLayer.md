---
layout: post
title: "[Pytorch] - 폴링 계층(Pooling Layer) 종류"
date: 2020-12-25 19:00:00
category: Pytorch
use_math: true
---

이번 포스터에서는 Convolution Layer을 거쳐서 나온 Feature maps을 Resizing 하여 새로운 Layer를 얻는 **폴링(Pooling)** 에 대해 알아보자.

# 폴링 계층(Pooling Layer)
<br>

CNN 알고리즘을 구성하는 필수적인 layer 중 하나인 Pooling Layer는 주로 Convolution Layer 연산을 거쳐 나온 Feature map이나 마지막 출력을 할 때 사용이 된다.

<center>
<img  src="../public/img/pytorch/pooling_layer.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진1. Pooling Layer </figcaption>
</center>

<br>

Pooling Layer에서 사용하는 필터는 값이 따로 존재하지 않고 단지 '기능'을 수행한다. 즉, Matrix 연산을 사용하지 않고 이미지의 픽셀로부터 값을 뽑아내는 역할을 한다. 다른 Filter들과 동일하게 Size와 Stride을 설정할 수 있꼬 보통 Size는 2x2를 사용하고 Stride는 Size에 맞게 2로 설정한다.(stride를 1로 설정하여 overlapping을 할 때도 있다.)

<br>

### Pooling Layer 특징
Pooling Layer 특징은 다음과 같다.
- Convolution Layer 다음에 사용이 된다.
- Feature Map의 Size를 줄인다.(DownSampling)
- 전체적으로 파라미터의 수와 연산량을 줄인다.(Computation 효율 증가)
- 작게나마 Overfitting에도 효과가 있다.(어느 정도 있다고는 한다....)

<br>

# Pooling Layer의 종류
<hr>

Pooling Layer읠 종류로는 약 4가지가 있다. 물론 Mini

Pooling Layer는 각 Feature map에 대해 독립적으로 적용이 되고, map의 Size 또한 감소시킨다. Size가 감소되면 자연스럽게 다른 필터들의 파마리터 수가 감소하고 연산량 또한 감소하게 된다.

<br>



- 큰 효과는 없지만 overfit에도 효과기 있다. 원리는 feature들을 pool해서 분석해야 하는 feature 개수를 낮춘다. 하지만 이걸로 overfit 막을 바에는 그냥 drop-out 써라
- feature map의 size을 줄인다.
- 파라미터의 수와 연산량을ㅈ ㅜㄹ인다.
- 각 feature map에 대해 독립적으로 적용된다.
- overlapping 방법을 쓰기도 한다.
- 종류에는 max, mini, average, global average 등이 있다.


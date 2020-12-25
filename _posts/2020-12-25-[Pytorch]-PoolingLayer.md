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

Pooling Layer는 각 Feature map에 대해 독립적으로 적용이 되고, map의 Size 또한 감소시킨다. Size가 감소되면 자연스럽게 다른 필터들의 파마리터 수가 감소하고 연산량 또한 감소하게 된다.

<br>

 


- 큰 효과는 없지만 overfit에도 효과기 있다. 원리는 feature들을 pool해서 분석해야 하는 feature 개수를 낮춘다. 하지만 이걸로 overfit 막을 바에는 그냥 drop-out 써라
- feature map의 size을 줄인다.
- 파라미터의 수와 연산량을ㅈ ㅜㄹ인다.
- 각 feature map에 대해 독립적으로 적용된다.
- overlapping 방법을 쓰기도 한다.
- 종류에는 max, mini, average, global average 등이 있다.


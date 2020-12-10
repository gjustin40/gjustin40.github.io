---
layout: post
title: [Pytorch] - Tensor
date: 2020-12-10 22:00:00 +0900
category: Pytorch
---
AI기술이 향상되면서 많은 사람들이 기계가 학습하는 기술, Machine Learning에 대해 관심을 가지게 되었다. 나도 그 기술에 매료되어 공부를 시작했고, 그 도구로 Pytorch를 이용했다. 앞으로 Pytorch category에 올리는 포스트에는 Pytorch로 공부한 내용을 업로드 할 예정이다.

# Pytorch란?<hr>
Python 기반의 과학 연산 패키지로 기계학습(딥러닝)을 가능하도록 도와주는 프레임워크이다. Classification, Object Detection 등 기계학습 테스크들을 쉽고 빠르게 구현할 수 있도록 여러 기능들을 제공해주는 도구라고 할 수 있다. 기계학습 프레임워크에는 Tensorflow, Caffe2 등이 있지만 Pytorch가 최근 연구에 많이 사용되어지고 초보자도 쉽게 할 수 있다고 소문이 났다. 구현속도가 다른 프레임워크들에 비해 비교적 빠르기 때문에 연구용으로 많이 사용이 된다.

# Tensor<hr>
다차원의 행열로 여러 Type을 포함하는 데이터의 형태 중 하나이다. 쉽게 표현하면 그냥 '데이터' 그 자체이다. 보통 선형대수학에서 선형관계를 나타내는 다중선형대수학의 대상으로 정의되는데, 기계학습에서 보통 데이터를 표현할 때 자주 사용한다.

**지금부터는 실제 Pytorch을 이용해 Tensor을 다뤄보도록 하자.**

1. Tensor
```python
import torch
print(torch.__vserion__) # 1.4.0
x = torch.Tensor(5, 3)
y = torch.empty(5, 3)
print(x, '\n')
print(y, '\n')
print(type(x), x.dtype, type(y), y,dtype)
```
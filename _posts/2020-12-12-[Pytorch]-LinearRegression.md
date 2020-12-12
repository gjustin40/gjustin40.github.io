---
layout: post
title: "[Pytorch] - 선형회귀(Linear Regression)"
date: 2020-12-12 15:00:00
category: Pytorch
use_math: true
---

이전 포스터에서는 Tensor와 Autograd등에 대해 알아보았다. 이번 포스터에서는 Pytorch을 이용해 간단한 연산을 하는 방법에 대해 알아보고 선형회귀를 코드로 구현을 해서 어떤 원리가 적용되고 있는지, 또한 실제로 간단한 데이터를 생성하여 모델을 학습시켜 데이터에 맞는 적절한 모델을 찾는 실습을 해보기로 하자. 

# 선형 회귀(Linear Regression)
<hr>

| Priority apples | Second priority | Third priority |
|-------|--------|---------|
| ambrosia | gala | red delicious |
| pink lady | jazz | macintosh |
| honeycrisp | granny smith | fuji |


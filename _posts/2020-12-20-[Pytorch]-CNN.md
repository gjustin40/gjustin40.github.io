---
layout: post
title: "[Pytorch] - 합성곱 신경망(Convolutional Neural Network, CNN)"
date: 2020-12-18 19:00:00
category: Pytorch
use_math: true
---

이번 포스터에서는 인공신경망 분야 중 하나인 ComputerVision의 기본이 되는 알고리즘인 **"Convolutional Neural Network"**에 대해 알아볼 것이다. CNN의 개념은 다른 분야에서도 많이 사용되고 있긴 하지만 이미지와 관련된 분야에서 획기적인 발전이 이루어졌다. CNN이 무엇인지 알아보고 Pytorch로 어떻게 구현하는지에 대해서도 다뤄보자.

# 이미지 필터링(Filtering)
<hr>

사진을 찍고 편집을 해 본 경험은 누구나 다 있을 것이다. 얼굴에 잡티를 제거하거 좀 더 예쁘고 잘생기게 만들기 위해 턱을 깍고, 피부의 톤을 바꾸는 등이 있다. 사실은 이 작업들은 모두 **'필터(Filter)`**에 의해 작동이 된다. **필터링**이란 필터를 이용해 이미지를 이루고 있는 픽셀행렬에 여러가지 수식을 대입해 다른값으로 바꿔서 이미지를 변형하는 것을 말한다. 주로 잡티를 제거할 때 가장 많이 사용을 하는 필터인 블러(blur) 필터로 예를 들어보자.

<img  src="../public/img/pytorch/blur.JPG" width="" style='margin: 0px auto;'/>

위 사진은 **Blur Filter**를 사용한 결과이다. 이미지를 흐릿하게 만들어서 각중 노이즈를 제거하는 용도로 많이 사용된다. 이 필터를 사용자가 원하는 부위에 적용을 하면 잡티제거가 된다. 필터가 적용되는 방식은 다음과 같다.

<center><img  src="../public/img/pytorch/filtering.gif" width="" style='margin: 0px auto;'/></center>
---
layout: post
title: "[Pytorch] - 활성화 함수(Activation Function)"
date: 2020-12-23 19:00:00
category: Pytorch
use_math: true
---

이번 포스터에서는 활성화 함수(Activation Function)에 대해 얘기해 볼 것이다. 딥러닝 모델을 설계할 때 각 층(Linear, CNN 등)에서 연산이 이루어지고 난 후 출력할 때 주로 활성화 함수를 거친 후에 출력이 된다. 딥러닝에서 필수적으로 사용되는 활성화 함수에 대해 알아보도록 하자.

# 활성화 함수(Function)
<hr>

우리 몸에는 신경을 전달하는 뉴런이 있다. 어느 부위에서 자극이 있으면 뉴런들은 그 자극을 뇌로 전달해서 우리가 느낄 수 있는 것이다. 하지만 뉴런들도 어느정도 강한 자극이 있거나 특정 임계점의 에너지가 작용해야 자극을 전달할 수 있다.

<br>

<center>
<img  src="../public/img/pytorch/activation_intro.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 뉴런과 딥러닝 </figcaption>
<figcaption> 출처 : https://cs231n.github.io/neural-networks-1/ </figcaption>
</center>

<br>

뉴런의 원리가 딥러닝에서도 똑같이 적용이 되는데, 이처럼 특정한 임계값을 넘겨야만 옆으로 자극(연산결과)을 전달하는 것을 **활성화 함수(Activation Function)** 이라고 한다. 즉, 입력 신호의 총합을 출력 신호로 변환할 때 신호의 총합이 활성화를 일으키는지를 정하는 역할을 한다.

<br>

수식으로 보면 다음과 같다.

$$
\begin{aligned}
a &= b + w_1x_1 + w_2x_2\\
y &= h(a)\\
\end{aligned}
$$

위 식에서 가중치가 달린 입력 신호와 편향의 총합이 $a$이고 활성화 함수가 $h(x)$이다. $h(x)$가 임계값을 경계로 출력값을 바꾸기 때문에 '활성화'를 한다고 볼 수 있는 것이다.

<br>

활성화 함수의 가장 큰 특징 중 하나는 **비선형 함수** 를 이용한다는 것이다. 다시 말하면 **선형 함수**를 사용해서는 안 된다. 그 이유는 신겸앙의 층을 깊게 하는 의미가 없어지기 때문이다. 아래는 선형함수인 $h(x) = cx$를 사용했을 때의 수식이다.

<br>

$$
y = h(h(h(x))) = c*c*c*x = c^3x = ax
$$

<br>

위의 식처럼 은닉층이 없는 네트워크로 표시할 수 있다. 다시 말해 딥러닝 모델을 아무리 깊게 쌓아도 결국 1층 뿐인 것과 같은 효과가 나타나는 것이다. 따라서 모델의 층을 깊게 쌓는 혜택을 얻기 위해서는 활성화 함수를 **비선형 함수**로 사용해야 한다.




<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


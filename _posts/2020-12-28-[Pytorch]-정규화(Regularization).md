---
layout: post
title: "[Pytorch] - 정규화(Regularization)"
date: 2020-12-28 19:00:00
category: Pytorch
use_math: true
---

이번 포스터에서는 모델의 과적합을 방지하는 테크닉 중 하나인 정규화(Regularization)에 대해 알아보도록 하자.

<br>

# 과적합
<hr>

대부분의 테크닉들이 모두 과적합을 방지하기 위해 만들어진 것 같다. 뭐만 하면 과적합을 막을 수 있다고 하니까....

<br>

과적합이란 학습 데이터만 지나치게 학습한 나머지 모델이 **일반화**되지 못 해서 새로운 데이터에 대해 예측을 잘 하지 못 하는 상태를 말한다. 학습할 때는 Loss가 감소하지만, 실제 Test에 대해서는 Loss가 감소하지 않는 현상을 보인다. "Training Datasets에 대해서만 매우 적합한 모델'이 된 셈이다. 과적합에 대한 자세한 내용은 [여기](https://gjustin40.github.io/pytorch/2020/12/28/Pytorch-Overfitting.html)에서 볼 수 있다.

<br>

<center>
<img  src="../public/img/pytorch/overfit_new.png" width="" style='margin: 0px auto;'/>
</center>

위 사진을 보면 '정확도'를 기준으로 판단을 한다면 당연히 (2)가 좋아보인다. 하지만 보통 (2)처럼 학습이 된 모델에 **새로운 데이터**를 입력하게 되면 예측을 잘 하지 못 하는 경우가 더 많다. 그 이유는 현재 가지고 있는 **학습 데이터**에 대해서만 학습이 되어버렸기 때문이다. 따라서 (2)와 같은 모델을 (1)과 같이 **일반화**를 해 줄 필요가 있다. 이와 같은 문제를 해결하기 위해 **정규화(Regularization)**을 이용한다.

<br>

# 정규화(Regularization)
<hr>

**정규화(Regularization)**은 보통 '일반화'라고 번역되기도 하는데, 모델에 '제약'을 걸어서 모델의 복잡도(Complexity)를 줄여 **일반화(Generalization)**를 개선하는 기법을 말한다. 딥러닝에서는 Weight값에 패널티를 가해서 과도하게 커지는 것을 방지한다. 

<br>

<center>
<img  src="../public/img/pytorch/regularization.png" width="" style='margin: 0px auto;'/>
<figcaption> Regularization </figcaption>
<figcaption> 출처 : https://m.blog.naver.com/laonple/220527647084 </figcaption>
</center>

<br>

기계학습을 하는 과정에서 모델이 '학습'을 한다는 것은 Weight을 조정하는 것, 다시 말해 손실(Loss)을 줄이는 방향으로 Weight을 갱신하는 것을 말한다. 학습 데이터를 이용해 단순히 Loss가 최소가 되는 방향으로 진행을 하다 보면, 특정 Weight값이 다른 weight에 비해 상대적으로 커지면서 모델의 성능을 악화시키는 경우가 있다. 모델의 **복잡도**를 증가시켜서 과적합을 야기시킨다. 따라서 Loss를 감소시켠서 동시에 **Regularization**을 통해 Weight값이 커지지 않도록 제약을 걸어줘야한다. 

<br>

아래 사진을 참고하면 위 설명을 직관적으로 이해할 수 있다.

<br>

<center>
<img  src="../public/img/pytorch/model_complex.png" width="" style='margin: 0px auto;'/>
<figcaption> Regularization </figcaption>
<figcaption> 출처 :https://kimlog.me/machine-learning/2016-01-30-4-Regularization </figcaption>
</center>

<br>

Regularization을 통해 $\theta$값에 제약을 걸어줘서 모델이 복잡해지는 현상을 막을 수 있다.

<br>

> 결론적으로 Regularization은 모델의 Generalization을 개선하는 기법

<br>

### Regularization 수학적 표현

<br>


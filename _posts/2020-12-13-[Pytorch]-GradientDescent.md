---
layout: post
title: "[Pytorch] - 경사하강법(Gradient Descent)"
date: 2020-12-13 15:00:00
category: Pytorch
use_math: true
---

앞 포스터에서 선형회귀모델과 두 좌표를 이용한 방식과 정규방정식을 이용하여 손실함수J($\theta$)를 최소화하는 $\theta$값을 구하는 방법에 대해 알아보았다. 물론 Pytorch를 이용하긴 했지만, 앞 두 가지 방법은 numpy나 scipy 등 다른 라이브러리로도 충분히 구할 수 있었다. 따라서 이번 포스터는 Pytorch를 적극 활용하여 $\theta$값을 찾을 수 있는 경사하강법에 대해 알아보자.

# 경사하강법(Gradient-Descent)
<hr>

사실 위에서 했던 방법들은 굳이 Pytorch를 사용하지 않아도 numpy, scipy 등을 이용해서 충분히 구현할 수 있다. 이번에는 Pytorch만의 기능을 이용해서 구해볼 예정인데, '경사하강법'을 이용해 손실함수를 최소로 하는 $\theta$를 구해보도록 하자.

<br>

**경사하강법**은 최적화 알고리즘으로 손실함수를 최소화하기 위해 반복해서 파라미터(계수)들을 조정해가는 것을 말한다. 처음에는 무작위로 초기화된 $\theta$로 시작해 손실함수(여기서는 MSE를 사용)가 감소되는 방향으로 진행하여 모델이 최솟값에 도달할 수 있도록 $\theta$를 갱신하는 알고리즘이다. 그렇다면 왜 이름에 '경사'가 들어갔을까? 바로 **미분**을 이용하기 때문이다.

<br>

<img  src="/public/img/pytorch/gradient-descent.png" width="400" style='margin: 0px auto;'/>

<br>

**미분**은 '기울기'를 뜻한다. 손실함수를 정하고 손실함수가 최소가 되도록 하는 매개변수를 찾는 것이 학습의 목표인데, 손실함수의 미분값이 0 되는 부분을 찾으면 그 때의 매개변수가 손실함수를 최소로 하는 매개변수이다. 하지만 신경망은 식이 매우 복잡하고 미분계수를 계산하기 어려워서 손실함수의 미분값이 0이 되는 부분을 찾기에는 쉽지 않다. 그래서 '기울기'를 이용해서 경사가 낮은 방향으로 매개변수를 변화시켜 최소인 지점을 찾는 것이다.

- 보통 손실함수(Loss)는 비선형(Non-linear)이기 때문에 방정식의 해를 구할 수 없다.
- 방정식의 해가 있다고 하더라도 **경사하강법**의 계산이 더 효율적이다.

<br>
경사하강법을 설명할 수 있는 아주 좋은 예시가 있다.

<br>

> 앞이 보이지 않는 어두운 밤에 산을 내려온다고 가정하자. 등산객은 사방으로 발을 더듬으면서 높이가 낮아지는 방향으로 나아가게 된다.

<br>

### 경사하강법 수식
미분값을 이용해 매개변수($\theta$)를 갱신하는데, 갱신하는 식은 다음과 같다.

<br>

$$
\theta_{n+1} = \theta_n - \eta\frac{\partial f(\theta_n)}{\partial \theta_n}
$$

- $\theta_n$ : 현재 매개변수의 값
- $\theta_{n+1}$ : 다음 매개변수의 값
- $f(\theta)$ : 손실함수(Loss Function)
- $\eta$ : 학습률(Learning Rate)

<br>

위의 식과 아래의 사진을 참고하여 설명을 하면 다음과 같다.

<img  src="/public/img/pytorch/gradient-new.png" width="400" style='margin: 0px auto;'/>

미분의 결과로는 양수 또는 음수의 기울기가 나올 수 있다. 기울기가 양수일 때, $\theta$값을 감소키기는 방향으로 가면 전체 손실함수의 값은 더 작아진다. 따라서 식에서와 같이 기존 매개변수에 학습률과 기울기를 곱한 값을 빼준다. 기울기가 음수일 때는 반대로 $\theta$값을 증가시키는 방향으로 가면 손실함수의 값이 작아진다. 이처럼 기울기를 이용해서 손실함수의 값이 작아지는 방향으로 매개변수를 갱신한다.

<br>

여기서 **학습률($\eta$)**의 역할은 '갱신을 얼마나 해줄 것인가'이다. 즉, 갱신을 하는 '정도'를 정해주는 하이퍼 파라미터(Hyper Parameter)이다. 이 값은 하이퍼 파라미터이기에 사람이 직접 설정을 해줘야한다. 여러 실험을 통해 적절한 학습률($\eta$)을 정할 수도 있고, Learning-Rate-Range-Test(LTTR)을 통해 찾아줄 수도 있다.(LRRT에 대해서는 나중에 다뤄보기로 하겠다.)

<br>

경사하강법에도 여러가지가 있는데, 주로 많이 쓰는 것들에는 무엇이 있는지 알아보자.(대부분의 방법들은 경사하강법 수식을 기반으로 한다.)

### 배치 경사하강법(Batch Gradient Descent)

자료들을 보다보면 굉장히 혼동이 될 소지가 있다. 그냥 GD라고 쓰는 곳도 있고 Batch라는 단어를 포함하는 경우도 있다. 보통 사용하는 방법은 전체 데이터 중 몇 개만 error를 계산하고 기울기를 구한 후 갱신을 하는 mini-batch를 말하는데, mini-batch와 batch를 혼용해서 사용하기 때문에 의문을 가질 수 있다. BGD는 **학습에 이용하는 모든 데이터에 대한 error를 구하고 기울기를 한 번에 계산하여 가중치를 업데이트하는 방법이다.**

<br>

예를 들어 100개의 학습데이터가 있다고 하면, 1 iter 마다 100개의 데이터에 대한 error값을 계산하고 한 번에 기울기를 구한 후 갱신을 한다. 1 Epoch 100개의 데이터를 보는 것이다.
<br>

- BGD의 장점은 다음과 같다.
  - 전체 데이터에 대해 error와 기울기를 구하기 때문에 손실함수의 최솟값을 찾아가는 과정이 매우 안정적이다.(속도와는 무관하다.)
  - 가중치를 업데이트하는 횟수가 상대적으로 적다.
<br>
- BGD의 단점은 다음과 같다.
  - 모든 데이터에 대한 계산을 하기 때문에 학습이 오래 걸린다.
  - 모든 데이터에 대한 계산을 하기 때문에 많은 메모리가 필요하다.
  - 사실상 GPU의 강력한 기능을 사용하지 않기 때문에 매우 비효율이다.

<center><img  src="/public/img/pytorch/GD.JPG" width="400" style='margin: 0px auto;'/></center>

<br>

GD의 코드는 다음과 같다.(앞, 뒤 모두 생략하고 학습하는 부분만 작성하겠다.)

```python
# 데이터 정의
# 모델 정의
# 손실 및 최적화 함수 정의

for e in range(EPOCH):
    
    train_loss = 0
    for data in dataloader: # batch_size=1인 dataloader
        image, label = data # image = 1개
        
        output = model(image)
        loss = loss_func(output, label)
        
        train_loss = train_loss + loss
        ### error 계산 완료

    optimizer.zero_grad()    
    train_loss.backward()
    optimizer.step()   
```

<br>

### 확률적 경사하강법(Stochastic Gradient Descent)

이름 그대로 **확률적으로 가중치를 갱신하는 방법이다.** 학습 데이터 중 무작위로 1개의 데이터를 선택한 후에 error와 기울기를 구하고 가중치를 갱신한다. 즉, 1 iter 당 1개의 데이터에 error와 기울기를 구하고 가중치를 업데이트르 진행한다. 1 Epoch 당 1개의 데이터를 보는 것이다.
<br>

- SGD의 장점은 다음과 같다.
  - 무작위로 데이터를 선택하는 것이기 때문에 Shooting 효과가 일어나서 local minimum에서 벗어날 기회가 있다.
  - 갱신하는 주기가 짧기 때문에 손실함수의 최솟값으로 찾아가는 속도가 빠르다.
<br>
- SGD의 단점은 다음과 같다.
  - 아무래도 확률적으로 찾는 것이기 때문에 반대로 global minimum에 빠질 확률이 있다. 
  - GD와 마찬가지로 데이터를 1개씩 처리하기 때문에 GPU의 강력한 기능을 사용할 수 없다.

<img  src="/public/img/pytorch/SGD-GD.png" width="" style='margin: 0px auto;'/>

SGD의 코드는 다음과 같다.

```python
# 데이터 정의
# 모델 정의
# 손실 및 최적화 함수 정의

for e in range(EPOCH):
    
    for data in dataloader: # batch_size=1인 dataloader
        image, label = data # image = 1개
        
        optimizer.zero_grad()
        
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        
        optimizer.step()      
```

<br>

### 미니 배치 경사하강법(mini-batch Gradient Descent)

대부분의 학습에서 사용하는 방법이다. 한 번 학습을 할 때 설정했던 batch_size 만큼의 이미지만 추출해서 error을 계산하고 기울기 값을 구한 후 가중치를 갱신한다. 예를 들어 100개의 학습 데이터가 있고 batch_size=10일 때, 한 번 학습을 할 때(1 iter) 10개의 이미지를 사용해서 학습하고, 그것을 10번 바복하면 1 epoch이 되는 것이다.

<br>

보통 미니배치경사하강법(MGD)에다가 SGD을 합친 방법인 MSGD(mini-batch Stochastic Gradient Descent)을 주로 사용한다. 설정된 batch_size만큼의 이미지를 선택할 때 무작위로 선정을 하기 때문이다. 물론 MSGD와 MGD는 다른 알고리즘이긴 하지만 혼용해서 사용할 때가 많다.

- MGD(MSGD)의 장점은 다음과 같다.
  - local minimum에 빠질 확률이 현저히 적다.
  - batch 단위로 학습을 하기 때문에 GPU의 강력한 기능인 병렬처리에 유리하다.
  - batch size 만큼의 이지만 학습에 이용하기 때문에 메모리 사용이 상대적으로 적다.
<br>
- MSD(MSGD)의 단점은 다음과 같다.
  - SGD보다는 상대적으로 메모리 사용이 많다.
  - batch_size을 사용자가 직접 지정해줘야하는 번거로움?이 있다.

<img  src="/public/img/pytorch/SGD-MSGD.png" width="" style='margin: 0px auto;'/>

<br>

MSGD의 코드는 다음과 같다.

```python
# 데이터 정의
# 모델 정의
# 손실 및 최적화 함수 정의

BATCH_SIZE = 10
dataloader = dataloader(batch_size = BATCH_SIZE)

for e in range(EPOCH):
    
    for data in dataloader: # batch_size=10인 dataloader
        image, label = data # image = 10개
        
        optimizer.zero_grad()
        
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        
        optimizer.step()      
```
SGD와 비슷하지만 batch_size를 설정해야하는 부분에서 차이가 있다. 또한 error값을 계산할 때 batch_size 개수만큼의 error가 발생하지만, 손실함수를 정의하는 과정에서 모두 합치는 옵션이 기본값으로 설정되어 있어서 자동으로 sum()이 돼서 출력이 된다. 자세한 내용은 [여기](https://gjustin40.github.io/pytorch/2020/12/15/Pytorch-LossFunction.html)를 참고하면 된다.

<br>

지금까지 경사하강법과 그 종류들에 대해 알아보았다. 사실 위에 언급한 알고리즘 말고도 모멘텀, adagrad, adam 등이 있다. 물론 지금은 모멘텀을 추가한 MSGD가 많이 사용되고 있지만, 대부분은 이번 포스터에서 언급한 알고리즘들이 많이 사용된다. 기회가 된다면 다른 최적화 함수들도 다뤄볼 예정이다.

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**
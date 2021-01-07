---
layout: post
title: "[Pytorch] - model.eval() vs torch.no_grad()"
date: 2021-01-07 19:00:00
category: Pytorch
use_math: true
---

오늘은 비교적 가벼운 주제이고 Pytorch만의 특성인 'model.eval()과 torch.no_grad()의 차이'에 대해 알아보자.(코드는 [여기](https://github.com/gjustin40/Pytorch-Cookbook/blob/master/Advanced/model.eval_vs_torch.no_grad.ipynb)를 참고)

# Pytorch 코드
<hr>

<br>

두 메소드 모두 보통 Validation 또는 Test할 때 사용을 한다. 아래는 학습을 할 때의 코드이다.
```python
EPOCH = 20
train_loss_list, val_loss_list = [], []

for e in range(EPOCH):
    
    model.train()
    
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # ................................................
        # ..............생략..................................
        # ................................................
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
        # ................................................
        # ..............생략..................................
        # ................................................
    
    
    torch.save(model.state_dict(), '../../data/weight')
```
- epoch에 대한 for구문 안에 2개의 for구문이 있다.(각각 train과 validation)
- validation 부분을 보면 `model.eval()`와 `with torch.no_grad()` 메소드가 있다.

<br>

### model.eval()

<br>

이 메소드는 train모드와 evaluation(test)모드를 전환할 때 사용한다. 사실 train모드와 eval모드 둘 다 학습(optimizer.step() 과 loss.backward() 등)이 가능하지만, eval모드에서는 **Dropout**이나 **Batch Normalization**을 off하는 기능을 가지고 있다. 이 두 테크닉(Dropout, BN)은 학습을 좋게 하기 위해서만 사용을 하고 실제 test를 할 때는 사용하지 않는다.



<br>

### torch.no_grad()
<br>

이 메소드는 train, test 등과 상관없이 단순히 자동미분(Autograd)를 off하는 기능을 가지고 있다. 모델을 test할 때는 학습을 하지 않아 역전파가 일어나지 않기 때문에 굳이 자동미분을 사용할 필요가 없다. 따라서 test할 때는 거의 세트로 사용한다. 해당 메소드를 사용하면 자동미분을 통해 데이터를 남기지 않기 때문에 당연히 메모리를 절약할 수 있고 실제로 연산 속도도 더 빨라진다.

<br>

정리하면 다음과 같다.
- `model.eval()` : 모델에게 eval모드로 전환된다고 알리고,  Dropout이나 BN 등의 테크닉을 off한다.
- `torch.no_grad()` : 자동미분(Autograd)를 비활성화하고, 메모리와 연산속도를 증가시킨다.(단, 오차역전파 불가능)

<br>

### 코드로 보기

<br>

아래 코드는 dropout기능을 구현한 임의의 모델이다.
```python
class MyModel(nn.Module):
    
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.fc1 = nn.Linear(4,3)
        self.fc2 = nn.Linear(3,2)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x0 = x
        print('x0 == ', x0)
        
        x1 = self.fc1(x0)
        print('FC1 == ', x1)
        
        x2 = self.dropout(x1)
        print('Dropout == ', x2)
        
        x3 = self.fc2(x2)
        print('FC2 == ', x3)
        
        return x
    
model = MyModel()
```
- 각각의 결과를 확인하기 위해 print() 메소드를 중간중간 사용했다.

<br>

`model.train()`과 `model.eval()`을 비교하면 다음과 같다.
```python
model.train()
model(a)
# Before Dropout ==  tensor([[0.1675, 0.4831, 0.2952]], grad_fn=<AddmmBackward>)
# After Dropout ==  tensor([[0.0000, 0.9662, 0.5904]], grad_fn=<MulBackward0>)

model.eval()
model(a)
# Before Dropout ==  tensor([[0.1675, 0.4831, 0.2952]], grad_fn=<AddmmBackward>)
# After Dropout ==  tensor([[0.1675, 0.4831, 0.2952]], grad_fn=<AddmmBackward>)
```
- `train()` 모드에서는 노드가 0값으로 변했지만 `eval()`모드에서는 변화가 없다.
- `grad_fn`을 보면 자동미분이 적용되고 있다는 것을 알 수 있다.
- `trian()` 모드를 보면 값이 살짝 다른데(2배), 이 부분에 대해서는 [여기](https://gjustin40.github.io/pytorch/2020/12/30/Pytorch-DropOut.html)에 맨 밑 scaling 부분을 참고하면 된다.

<br>

Dropout은 off됐지만 자동미분은 off되지 않았다.

<br>

`no_grad()`를 사용하면 다음과 같다.
```python
with torch.no_grad():
    model.train()
    model(a)
# Before Dropout ==  tensor([[0.1675, 0.4831, 0.2952]])
# After Dropout ==  tensor([[0.0000, 0.0000, 0.5904]])

with torch.no_grad():
    model.eval()
    model(a)
# Before Dropout ==  tensor([[0.1675, 0.4831, 0.2952]])
# After Dropout ==  tensor([[0.1675, 0.4831, 0.2952]])
```
- 결과(노드값)는 train모드과 eval모드 비교 코드와 같다.
- 하지만 `grad_fn`이 없는 걸로 보아 자동미분이 적용되고 있지 않다.

<br>

과거에는 생각없이 두 개가 세트라고 생각해서 습관적으로 사용했지만, 지금은 둘의 차이를 알았기에 좀 더 전략적으로 사용하면 더 좋을 것 같다.

<br>

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**
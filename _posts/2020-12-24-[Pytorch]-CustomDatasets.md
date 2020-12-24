---
layout: post
title: "[Pytorch] - 나만의 Datasets 만들기"
date: 2020-12-24 19:00:00
category: Pytorch
use_math: true
---

딥런닝을 하기 위해서는 데이터가 필수이다. Pytorch에서 자체로 제공하는 dataset도 있지만, 현실적인 문제를 다룰 때는 직접 사용자가 수집한 데이터를 이용하는 경우가 더 많다. Pytorch에서 제공하는 데이터는 단순히 `torchvision` 라이브러리를 이용해 download 받고 이용하면 된다. 하지만 직접 수집한 데이터들은 어떻게 다뤄야할까? Custom한 datasets을 다루는 방법에 대해 알아보도록 하자.

# torchvision 내장 데이터셋 >> 수정이 필요함
<hr>

Pytorch로 데이터를 다룰 때는 보통 `torchvision.datasets()`을 이용한다. Pytorch에서 제공하는 데이터를 불러오는 방법은 다음과 같다.

```python
import torchvision

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=1)
```
- `CIFAR10` : CIFAR10 데이터셋을 정의한다.(train=학습 데이터,)
- `DataLoader()` : 정의한 datasets을 각종 옵션을 적용해 불러온다.

<br>

CIFAR10 데이터셋을 다운받고 `train=true`옵션을 통해 학습데이터를 불러온다. 사실 `datasets.CIFAR10`도 다운받은 파일을 `root`옵션을 통해 불러오는 방식이라, 실제로 데이터파일이 로컬에 설치가 되어있는 상태다. 이후에 데이터셋이 정의가 되면 `dataLoader()`을 이용해 실제 학습에 이용할 수 있도록 데이터를 호출한다. 위 코드에서 알 수 있듯이 뭔가 `datasets`의 코드를 바꾸면 우리가 직접 수집한 데이터를 정의할 수 있을 것 같다.

# 사용자 정의 데이터셋(Custom Dataset)
<hr>

Pytorch에서는 사용자가 직접 데이터를 정의할 수 있도록 하는 모듈인 `torch.utils.data.Dataset`을 제공한다. Python의 Class를 이용해 `Dataset`을 상속받고 메소드(Dataset 내에 있는 함수들)를 오버라이드하여 데이터셋을 만든다.

<br>

기본 프로토타입은 다음과 같다.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset): 
  def __init__(self):

  def __len__(self):

  def __getitem__(self, idx): 
```
- `__init__(self)` : 데이터의 경로를 설정하거나 전처리하는 부분
- `__len__(self)` : 데이터의 개수를 출력하는 부분
- `__getitem__(self, idx)` : 데이터셋에서 데이터를 호출하는 부분

<br>



Class 내에 여러가지 목적에 의해 함수가 나눠져있지만, 사실 `getitem`에서도 전처리가 가능하다.(그냥 함수라 전처리를 언제 하냐의 차이)
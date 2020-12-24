---
layout: post
title: "[Pytorch] - 나만의 Datasets 만들기(feat. DataLoader)"
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

### 1. torch.utils.data.Dataset

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

<center><figcaption> 코드1. 프로토타입 </figcaption> </center>

- `__init__(self)` : 데이터의 경로를 설정하거나 전처리하는 부분
- `__len__(self)` : 데이터의 개수를 출력하는 부분
- `__getitem__(self, idx)` : 데이터셋에서 데이터를 호출하는 부분

<br>

`__init__` 메소드에서는 보통 데이터를 다루는데 필요한 여러 요소들을 정의하고 데이터의 경로를 설정하거나 전처리를 한다. 주로 `__getitem__`에서 사용할 변수들을 정의하는 목적을 가지고 있다. `__getitem__`에는 데이터가 설계된 모델과 학습에 맞게 호출되도록 코드를 작성한다.

<br>

### custom dataset 정의하기

<br>

선형 회귀(Linear Regression)에 사용할 데이터를 정의하는 코드를 간단히 작성해보면 다음과 같다.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.x = [[100,80,100,70],
                  [70, 95, 92, 67],
                  [60, 88, 73, 92],
                  [80, 62, 91, 79]]
        self.y = [[150],[200],[200],[300]]

    def __len__(self):
        data_len = len(self.x)

        return data_len

    def __getitem__(self, idx):

        x = torch.Tensor(self.x)[idx]
        y = torch.Tensor(self.y)[idx]

        return x, y

mydata = MyDataset()
>>> print(mydata)
# <__main__.MyDataset object at 0x000001DFF7C83C40>
```

<br>

데이터셋을 정의했으면, 데이터셋 안에 있는 데이터들을 호출해야한다. 보통 `Dataset`은 `DataLoader`와 단짝을 이루어서 작동을 한다.

- `Dataset`을 이용해 데이터셋을 정의한다.
- `DataLoader`를 이용해 데이터셋 내의 데이터를 호출한다.
- 호출된 데이터로 모델을 학습한다.

<br>

`DataLoader`을 이용해 데이터를 호출하면 다음과 같다.

```python
from torch.utils.data import DataLoader

mydata = MyDataset()
data = DataLoader(mydata, batch_size=1)

>>> print(iter(data).next())
# [tensor([[100.,  80., 100.,  70.]]), tensor([[150.]])]
```

- `DataLoader` : `Dataset`을 입력값으로 받는다. batch_size옵션을 통해 한 번에 몇 개의 데이터를 불러올지 설정 가능
- `iter()` : DataLoader의 내장함수인 `__iter__()`을 실행시켜 iterator로 만든다.
- `next()` : `__next__()`을 실행시켜 첫 데이터를 가져온 것이다.

<br>

### 2. torchvision.datasets.ImageFolder

<br>

위에서 다뤘던 `Dataset`은 단순히 데이터들을 하나로 묶어주는 형식이었다. 하지만 각 이미지 데이터들이 파일명으로 레이블이 되어있는 상태일 때 사용하면 매우 유용한 함수가 있다.  `ImageFolder`을 이용하면 데이터와 레이블을 한 번에 만들 수 있다.

<br>

<center>
<img  src="../public/img/pytorch/data_fruit.jpg" width="" style='margin: 0px auto;'/>
<figcaption> 사진1. Label된 파일</figcaption> </center>

<br>

사진(1)과 같이 각 데이터들이 Label의 이름을 가진 파일로 나눠져있을 때 `ImageFolder`는 매우 강력한 힘을 발휘한다. 아래는 `ImageFolder`을 이용해 데이터셋을 정의하는 코드이다.

```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

datasets = ImageFolder(root='../../data/fruit', transform=transform)
print(datasets)
# Dataset ImageFolder
#     Number of datapoints: 8513
#     Root location: ../../data/fruit

data = DataLoader(datasets, batch_size=1)
>>> print(iter(data).next())
# [tensor([[[[0.3412, 0.3451, 0.3451,  ..., 0.6275, 0.6235, 0.6196],
#           [0.3569, 0.3569, 0.3451,  ..., 0.6275, 0.6235, 0.6157],
#           [0.3686, 0.3647, 0.3490,  ..., 0.6235, 0.6196, 0.6157],
#           ...,
#           [0.5137, 0.4980, 0.4902,  ..., 0.3451, 0.3490, 0.3490],
#           [0.5373, 0.5294, 0.5176,  ..., 0.3294, 0.3412, 0.3412],
#           [0.5490, 0.5412, 0.5412,  ..., 0.3098, 0.3216, 0.3255]],

#          [[0.4392, 0.4431, 0.4431,  ..., 0.7176, 0.7137, 0.7098],
#           [0.4510, 0.4510, 0.4392,  ..., 0.7176, 0.7137, 0.7059],
#           [0.4627, 0.4588, 0.4431,  ..., 0.7137, 0.7098, 0.7059],
#           ...,
#           [0.6392, 0.6235, 0.6039,  ..., 0.4980, 0.4941, 0.4863],
#           [0.6510, 0.6431, 0.6314,  ..., 0.4824, 0.4784, 0.4667],
#           [0.6627, 0.6549, 0.6471,  ..., 0.4549, 0.4549, 0.4471]],

#          [[0.3647, 0.3686, 0.3686,  ..., 0.7412, 0.7373, 0.7333],
#           [0.3882, 0.3882, 0.3765,  ..., 0.7412, 0.7373, 0.7294],
#           [0.4078, 0.4039, 0.3882,  ..., 0.7373, 0.7333, 0.7294],
#           ...,
#           [0.3137, 0.2980, 0.2745,  ..., 0.0784, 0.0902, 0.0941],
#           [0.3294, 0.3216, 0.3020,  ..., 0.0667, 0.0784, 0.0863],
#           [0.3490, 0.3333, 0.3294,  ..., 0.0510, 0.0745, 0.0824]]]]), tensor([0])]
```

<br>

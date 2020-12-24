---
layout: post
title: "[Pytorch] - 나만의 Datasets 만들기"
date: 2020-12-24 19:00:00
category: Pytorch
use_math: true
---

딥런닝을 하기 위해서는 데이터가 필수이다. Pytorch에서 자체로 제공하는 dataset도 있지만, 현실적인 문제를 다룰 때는 직접 사용자가 수집한 데이터를 이용하는 경우가 더 많다. Pytorch에서 제공하는 데이터는 단순히 `torchvision` 라이브러리를 이용해 download 받고 이용하면 된다. 하지만 직접 수집한 데이터들은 어떻게 다뤄야할까? Custom한 datasets을 다루는 방법에 대해 알아보도록 하자.

# torchvision.datasets()
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

ㅇ
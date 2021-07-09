---
layout: post
title: "[Pytorch] - Normalization을 역으로! Inverse Normalization"
date: 2021-07-09 15:00:00
category: Pytorch
use_math: true
---

데이터 분석을 할 때 정규화(Normalization)는 많은 이로움을 준다. 
- 스케일이 다른 수치 데이터를 비슷한 범위로 변환
- 종속성을 제거하여 데이터의 일관성과 무결성을 보장
- 이상치 제거 및 완화
- 연산 효율 증가

<br>

이미지 데이터셋에도 Normalization을 적용한다. Pytorch에서는 Normalization을 쉽게 적용할 수 있도록 ```Datasets```을 생성할 때 ```torchvision.transforms```메소드를 이용한다.

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
```

## 정규화 된 이미지 시각화
<hr>

하지만 정규화가 된 이미지를 보기 위해 시각화를 하면 원본 사진과는 약간의 괴리가 있다.
(아래 코드는 이미지를 시각화하는 함수를 정의했다.)

```python
def img_show(dataloader):
    images, labels = iter(dataloader).next()      
    grid = torchvision.utils.make_grid(images, padding=1)
    grid = grid.permute(1,2,0)

    plt.imshow(grid)
```

<img  src="/public/img/pytorch/normalized_image.jpg" width="400" style='margin: 0px auto;'/>

- ```mark_grid``` : Pytorch에서 제공하는 메소드로, batch로 묶인 이미지를 grid로 변환한다.



1. Noramlization 하는 이유 0
2. Noramlization으로 인한 문제점
3. 해결 방법
 - 계산식
4. 유용한 utils는 없는지 검색
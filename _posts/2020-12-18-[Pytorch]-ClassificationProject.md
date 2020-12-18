---
layout: post
title: "[Pytorch] - CIFAR10 데이터를 이용한 Classification(분류) 프로젝트 "
date: 2020-12-18 19:00:00
category: Pytorch
use_math: true
---

인공신경망에 대한 전반적인 내용은 앞 글에서 충분히 다루었다고 생각한다. 따라서 이번 글에서는 실제 데이터를 이용해 분류(Classification)를 하는 프로젝트를 진행해보기로 하자.(배움을 위해 모델은 직접 정의한 후 진행하기로 하자.)

<br>

진행 순서는 다음과 같다.
1. 데이터 정의 및 불러오기(CIFAR10)
2. 모델 정의
3. GPU 사용여부 확인
4. 손실 함수(Loss Function) 및 최적화 함수(Optimizer) 정의
5. 하이퍼 파라미터(Hyper Parameter) 정의
6. 학습 진행
7. 모델 평가

<br>

# 데이터 정의 및 불러오기(CIFAR-10)
<br>

### 데이터

이번 프로젝트에서 사용할 데이터는 CIFAR-10 Datasets이다. 32x32 크기의 이미지가 약 6만개 정도 포함되어 있는 데이터셋이며, 각 이미지는 10개의 클래스 중 하나로 라벨링이 되어 있다. 머신러닝을 연구할 때 가장 많이 사용되는 데이터셋 중 하나이다.

<img  src="../public/img/pytorch/cifar10.jpg" width="400" style='margin: 0px auto;'/>

6만개의 이미지 중 5만개는 학습(Train)에 이용하고 나머지 1만개는 평가(Test)에 사용하도록 하겠다. 그렇다면 이 데이터를 어떻게 불러올까?

### 데이터 정의
<br>

Pytorch에서는 이러한 데이터셋을 쉽게 불러올 수 있도록 하는 라이브러리인 torchvision을 제공한다. 이 라이브러리는 여러 딥러닝 분야 중 Vision분야를 다룰 때 매우 유용하게 사용되고 있다. 데이터를 불러오는 코드는 다음과 같다.

```python
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transforms)
```

`torchvision.datasets`에 있는 데이터 중 `CIFAR10`을 정의했다.
- `root` : 데이터의 위치(./data = 현재 디렉토리 안에 data폴더)
- `train` : CIFAR10 데이터셋 내부적으로 train용과 test용으로 나눠져있다. 따라서 True로 설정 시 train용, False로 설정 시 test용 데이터셋을 정의한다.
- `downlaod` : `root`에 데이터가 없으면 자동으로 다운로드를 한다.
- `transform` : 데이터를 전처리하거나 부풀리기 등의 기능을 추가



<br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>

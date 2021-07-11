---
layout: post
title: "[Pytorch] - 전이학습(Transfer Learning)"
date: 2021-07-11 22:00:00
category: Pytorch
use_math: true
---

딥러닝을 하기 위해서는 수많은 데이터가 필요하다. 이름에서도 알 수 있듯이 딥하게 학습을 하기 위해서는 수백만개의 파라미터(가중치값)를 학습해야하고, 따라서 그만큼 데이터도 많이 필요하다. 물론 지금은 빅데이터 시대이기 때문에 실시간으로 엄청난 양의 데이터가 생성되고 있다. 초당 56만 GB의 데이터가 만들어지고 있고, 유뷰브에서는 1분당 400시간 분량의 동영상이, 페이스북에는 매일 수억장의 이미지가 새로 등록되고 있다.

<br>

하지만 이 수많은 데이터를 딥러닝에 이용하기 위해서는 가공을 해야하는데, 사실 그게 쉽지가 않고 결국 사람이 직접 검수를 해야하는 상황이다. 물론 기술이 좋아져서 사람이 직접 하지 않고도 원하는 데이터를 수집하는 여러 기술들이 개발되고 있지만, 여전히 사람이 직접 해야하는 부분이 많다. 딥러닝을 하는 업체들은 질 좋은 데이터를 수집하기 위해 많은 노력을 하고 있는데, 사실 작은 회사가 대기업 만큼의 데이터를 수집하는 건 사실상 불가능하다. 또한 기존에 하던 연구분야가 아닌 새로운 분야에서 딥러닝을 적용하기 위해서는 또 다시 데이터를 수집해야하는데, 이런 비효율을 해결하기 위해 똑똑한 분들이 '전이학습(Transfer Learning)'이라는 학습 방법을 고안해냈다.

<br>

전이학습이란 무엇일까~?

<br>

# 전이학습(Transfer Learning)
<hr>

**전이**라는 단어를 인터넷에 검색해보았다. 물론 대부분 비슷했지만 의학에서 사용한 **전이**의 뜻이 딥러닝에서의 **전이**와 가장 비슷하다고 생각했다.(개인적인 생각이다.....)

<br>

> 내담자가 과거에 중요하게 생각했던 사람에게 느꼈던 감정을 상담자에게 옮겨서 생각하는 것 [나무위키](https://namu.wiki/w/%EC%A0%84%EC%9D%B4)

<br>

**전이 학습(Transfer Learning)** 이란 **특정 분야(Task)에서 학습된 신경망(Model)의 일부 능력을 비슷하거나 새로운 분야에 활용하여 학습하는 것**이다. 즉, 데이터가 많은 분야에서 학습이 된 모델을 새로운 분야에 재사용하는 것이다. 기존에 사용하던 모델을 이용하여 새로운 분야에 대해 빠르게 학습하고 예측을 높이는 방법이다. 보통은 완전 새로운 분야에서 사용하는 것 보다는 비슷한 테스크를 수행하는 분야에서 활용된다. 예를 들면 **강아지의 종류**를 분류(Classification)하는데 뛰어난 모델을 **고양이의 종류**를 분류하는 부분에 활용하거나 **특정 물건을 탐지**하는 모델을 **사람을 탐지**하는 모델로 사용하는 등 다양하다.(컴퓨터 비젼 분야에서 활용한다고 가정하자.)


<center>
<img  src="/public/img/pytorch/transfer-learning.jpeg" width="800" style='margin: 0px auto;'/>
<figcaption> 출처 : https://sagarsonwane230797.medium.com/transfer-learning-from-pre-trained-model-for-image-facial-recognition-8b0c2038d5f0</figcaption>
</center>


<br>

**전이 학습**을 가장 많이 활용하는 부분이 **특징 추출(Feature Extraction)**이다. 딥러닝 모델은 보통 데이터로부터 특징을 추출하는 부분과 탐지 또는 분류를 하는 부분으로 나누는데, 특징 추출을 하는 부분은 보통 수백만개의 이미지 데이터를 통해 학습된 성능이 좋은 모델을 활용한다. 이미 ImageNet에서 검증된 모델인 VGG나 ResNet 등의 모델을 특징 추출에 사용을 하고, 탐지나 분류하는 부분만 학습을 진행해서 쉽고 빠르게 다른 분야에 적용할 수 있는 모델을 만든다.

<br>

전이학습의 장점은 다음과 같다.

- 이미 학습된 모델을 활용하기 때문에 데이터의 소요가 적다.
- 모델의 일부만 학습하기 때문에 학습에 필요한 시간과 연산량을 줄일 수 있다.
- 새로운 분야에 작동하는 모델을 쉽고 빠르게 만들 수 있다.



# Pytorch로 실습하기
<hr>

실습에는 CIFAR10데이터셋을 이용하고, 모델은 Classification의 대표 중 하나인 VGG16을 이용할 예정이다. 만약 Classification에 대해 알고 싶다면 [CNN을 이용한 CIFAR10 데이터 Classification(분류)](https://gjustin40.github.io/pytorch/2020/12/18/Pytorch-ClassificationProject.html)를 참고하자.

<br>

**전이 학습**을 하는 방법은 다음과 같다.
1. 데이터 불러오기 및 transfer 적용
2. 전이학습에 사용할 모델(VGG16) 불러오기(```Trained=True```)
3. 특징 추출하는 부분은 학습이 되지 않도록 얼리기(Freeze)
4. 특징들을 분류하는 부분만 학습이 될 수 있도록 설정
5. 목표로 하는 테스크에 맞게 모델의 출력 사이즈 재설정(class의 수)

<br>

**최대한 전체 코드를 첨부할 예정이지만, 포스트 주제에 맞지 않는 부분은 생략했다.**

### 1. 데이터 로드 및 transfer 적용

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='../data', train = True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train = False,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch,
                                          shuffle = True, num_workers = 1)
testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch,
                                          shuffle = True, num_workers = 1)
```
- Import하는 부분이 있으면 너무 난잡해보여서 생략했다.

<br>

### 2. 전이학습에 사용할 모델(VGG16) 불러오기(```Trained=True```)

```python
from torchvision import models
vgg = models.vgg16(pretrained=True)
vgg

# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
```
- ```VGG16``` : 뒤에 붙은 숫자 16은 모델의 깊이를 말한다. 16말고도 13, 19 등이 있다.
- ```pretrained``` : 학습 완료 여부를 설정하는 부분, VGG16모델의 껍데기만 사용할 예정이라면 False를 설정하면 된다.
- ```VGG16```모델의 속을 보면 크게 ```features```와 ```avgpool```, ```classifier```부분으로 나눠져 있다.
- ```features``` : 전이학습에 활용할 **특징 추출**부분
- ```avgpool``` : 입력 이미지의 사이즈에 상관없이 사용자가 원하는 output size로 출력
- ```classifier``` : 우리가 학습하고자 하는 부분으로, 특징들을 이용해 분류를 하고 예측을 하는 부분.

<br>

### 3. 특징 추출하는 부분은 학습이 되지 않도록 얼리기(Freeze)
Tensor의 자동미분을 off해서 역전파가 이루어지지 않도록 하는 부분[(자동미분(Autograd))](https://gjustin40.github.io/pytorch/2020/12/11/Pytorch-Autograd.html)

```python
for p in vgg.features.parameters():
    p.requires_grad = False
```
- ```vgg.features.parameters``` : features부분에 있는 파라미터(가중치)를 불러오는 부분
- ```requires_grad``` : 파라미터들이 학습되지 않도록 미분을 off하는 부분
- ```features```부분의 파라미터들은 loss를 계산하지 않기 때문에 역전파가 이루어지지 않아 학습이 되지 않는다.
- 모델을 불러올 때 이미 ```requires_grad```의 기본값은 ```True```이기 때문에 다른 부분은 그냥 두면 된다.

<br>

### 5. 목표로 하는 테스크에 맞게 모델의 출력 사이즈 재설정(class의 수)
ImageNet의 데이터셋으로 학습된 모델들은 보통 1000개의 class를 가지고 있기 때문에 마지막 ```classifier```의 output size는 1000이다. 따라서 전이학습에 사용하는 데이터셋에 맞도록 output size를 설정해야한다.

```python
# (6): Linear(in_features=4096, out_features=1000, bias=True)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
vgg.classifier[6]
>>> 
Linear(in_features=4096, out_features=10, bias=True)
```
- ```vgg.classifier[6]``` : classifier부분에서 가장 마지막 layer
- ```nn.Linear``` : 가장 마지막 부분의 ```out_features```크기를 CIFAR10에 맞도록 1000에서 10으로 변경

<br>

이제 나머지 부분은 [CNN을 이용한 CIFAR10 데이터 Classification(분류)](https://gjustin40.github.io/pytorch/2020/12/18/Pytorch-ClassificationProject.html)를 참고하면 된다. ```Model``` 부분만 ```VGG16```으로 대체되고, GPU설정 및 loss function, optimizer, 학습 등은 동일하다.

<br>

### 유의할 점
사실 이 부분은 필자가 멍청해서 발생한 문제였던 것 같은데, 나와 같은 실수를 하지 않았으면 하는 마음에 공유를 하고자 한다.

<br>

**5. 목표로 하는 테스크에 맞게 모델의 출력 사이즈 재설정(class의 수)** 코드를 다시 보면 

```python
# (6): Linear(in_features=4096, out_features=1000, bias=True)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
vgg.classifier[6]
>>> 
# Linear(in_features=4096, out_features=10, bias=True)
```

마지막 출력 layer를 새로운 Linear layer를 이용해 대체하였다. 하지만 필자는 처음에 아래와 같은 방법을 사용했고, 출력값에 변화가 없는 이유에 대해 하루종일 고민했다.

```python
# (6): Linear(in_features=4096, out_features=1000, bias=True)
vgg.classifier[6].out_features = 10
>>>
# Linear(in_features=4096, out_features=10, bias=True)
```

```classifier[6]```을 확인했을 때는 out_features의 값이 10으로 바뀌었는데, 실제 연산을 진행하면 자꾸 원래의 크기인 1000이 나오는 현상이 발생했다. 그래서 해당 layer의 weight값의 shape을 확인해봤더니....바뀌지 않았다.

```python
vgg16.classifier[6].weight.shape
>>>
# torch.Size([1000, 4096])
```

이유를 알고 싶어서 class에 대한 개념부터 소스코드를 분석해보았는데, 아마도 필자가 생각하는 이유는 아래 코드에서 확인할 수 있는 것 같다.(뇌피셜)

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
```

위 코드는 ```nn.Linear```에서 초기화하는 소스코드이다. VGG16모델을 로드할 때 이미 초기화 함수가 호출되어 ```self.weight```부분에 있는 ```Parameter```클래스가 실행된다. 그때 이미 ```self.out_features```의 기본값은 1000이었기 때문에 ```nn.Linear(out_features=1000)```가 생성되었다. 하지만 필자가 한 행동은 단지 인스턴스 변수(Instance Variable)을 바꿨을 뿐, ```Parameter```클래스를 실행하지 않았기 때문에 겉으로 ```self.out_features```는 10이었지만 실제 적용은 안 되었던 것이다. 사실 뻔한 얘기인데 당시에는 이해가 되지 않아서 한 참을 찾고 고민했던 기억이 있다. **또 바보같은 짓을 했다.**

<br>

**전이 학습(Transfer Learning)**에 대해 알아보았다. 사실 딥러닝을 공부하고 있는, 아직은 기업에 입사하지 않은, 큰 데이터가 없는 회사에 다니고 있는 사람이라면 전이학습은 필수이다. 개인적인 생각이지만, 앞으로 생성되고 개발되는 대부분의 모델은 전이 학습을 이용할 것 같은 기분이 든다. 적은 데이터로도 성능이 좋은 모델을 만들 수 있다는 사실이 너무 매력적이다. 개인 프로젝트에서도 많이 활용을 했지만, 아직 뚜렷한 결과를 보인 적이 없기 때문에 꾸준히 노력해야겠다는 생각이 들었다. 

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**

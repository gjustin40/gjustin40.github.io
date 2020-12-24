---
layout: post
title: "[Pytorch] - 나만의 Datasets 만들기(feat. DataLoader)"
date: 2020-12-24 19:00:00
category: Pytorch
use_math: true
---

딥런닝을 하기 위해서는 데이터가 필수이다. Pytorch를 사용하는 개발자라면 연구목적으로 pytorch가 자체로 제공하는 dataset를 사용하지만, 현실적인 문제를 다룰 때는 직접 사용자가 수집한 데이터를 이용하는 경우가 더 많다. 다행히 pytorch에서 제공하는 데이터는 단순히 `torchvision` 라이브러리를 이용해 다운을 받고 이용하면 된다. 하지만 직접 수집한 데이터들은 어떻게 다뤄야할까? Custom한 datasets을 다루는 방법에 대해 알아보도록 하자.

# Pytorch에서 제공하는 데이터

<hr>

Pytorch로 데이터를 다룰 때는 보통 `torchvision.datasets()`을 이용한다. Pytorch에서 제공하는 데이터를 불러오는 방법은 다음과 같다.

```python
import torchvision

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False)
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False)
trainset = torchvision.datasets.FakeData(root='./data', train=True,
                                        download=False)                                        
####################################################################
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=1)
```

- `CIFAR10` : CIFAR10 데이터셋을 정의한다.(train=학습 데이터,)
- `DataLoader()` : 정의한 datasets을 각종 옵션을 적용해 불러온다.

<br>

CIFAR10 데이터셋을 다운받고 `train=true`옵션을 통해 학습데이터를 불러온다. 사실 `datasets.CIFAR10`도 다운받은 파일을 `root`옵션을 통해 불러오는 방식이라, 실제로 데이터파일이 로컬에 설치가 되어있는 상태다. 이후에 데이터셋이 정의가 되면 `dataLoader()`을 이용해 실제 학습에 이용할 수 있도록 데이터를 호출한다. 위 코드에서 알 수 있듯이 `datasets`의 코드를 바꾸면 우리가 직접 수집한 데이터를 정의할 수 있을 것 같다.

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

### 이미지 데이터셋 정의
<br>

필자가 주로 사용하는 데이터는 이미지이기 때문에 이미지를 불러오는 코드를 간단하게 작성하면 다음과 같다.
```python
dog1.jpg / dog2.jpg / dog3.jpg / cat1.jpg / cat2.jpg / cat3.jpg / .... # 이미지 데이터 파일명
```
```python
import glob
import torchvision.transforms as transforms
import re

from torch.utils.data import Dataset
from PIL import Image


class MyImageData(Dataset):
    
    def __init__(self, data_path, class_to_label, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.class_to_label = class_to_label
        self.image_list = glob.glob(self.data_path + '/*.jpg')
        
    def __len__(self):
        
        return len(self.image_list)
    
    def __getitem__(self, idx):
        
        file_name = self.image_list[idx]
        
        img = Image.open(file_name)

        class_name = re.findall('[a-zA-Z]+', file_name)[-2]
        label = self.class_to_label[class_name]
        label = torch.tensor(label)
        
        img = self.transform(img)

        return img, label
    
    
data_path = 'data/image'
class_to_label = {'dog' : 1, 'cat' : 2}
transform = transforms.ToTensor()

mydataset = MyImageData(data_path, class_to_label, transform=transform)

data = DataLoader(mydataset)

>>> print(iter(data).next())
# [tensor([[[[0.5843, 0.5843, 0.5843,  ..., 0.5882, 0.5882, 0.5882],
#           [0.5843, 0.5843, 0.5843,  ..., 0.5922, 0.5922, 0.5922],
#           [0.5882, 0.5882, 0.5882,  ..., 0.5922, 0.5922, 0.5922],
#           ...,
#           [0.1686, 0.1686, 0.1843,  ..., 0.0784, 0.0863, 0.0902],
#           [0.1608, 0.1490, 0.1529,  ..., 0.0902, 0.0980, 0.1020],
#           [0.1529, 0.1294, 0.1255,  ..., 0.1059, 0.1137, 0.1137]],

#          [[0.6941, 0.6941, 0.6941,  ..., 0.6980, 0.6980, 0.6980],
#           [0.6941, 0.6941, 0.6941,  ..., 0.7020, 0.7020, 0.7020],
#           [0.6980, 0.6980, 0.6980,  ..., 0.7020, 0.7020, 0.7020],
#           ...,
#           [0.2549, 0.2510, 0.2706,  ..., 0.1451, 0.1529, 0.1569],
#           [0.2471, 0.2314, 0.2431,  ..., 0.1569, 0.1647, 0.1686],
#           [0.2392, 0.2157, 0.2157,  ..., 0.1725, 0.1804, 0.1804]],

#          [[0.7882, 0.7882, 0.7882,  ..., 0.7843, 0.7843, 0.7843],
#           [0.7882, 0.7882, 0.7882,  ..., 0.7882, 0.7882, 0.7882],
#           [0.7922, 0.7922, 0.7922,  ..., 0.7882, 0.7882, 0.7882],
#           ...,
#           [0.4157, 0.4196, 0.4549,  ..., 0.2627, 0.2706, 0.2745],
#           [0.4078, 0.4000, 0.4157,  ..., 0.2745, 0.2824, 0.2863],
#           [0.3922, 0.3765, 0.3882,  ..., 0.2824, 0.2902, 0.2902]]]]), tensor([2])]
```
- `re` : 이미지 파일의 파일명에서 class의 이름을 추출하기 위해 사용한 라이브러리(정규표현식)
- `re.findall()` : 정규표현식에 해당하는 문자열 모두 추출('[a-zA-Z]' == 영어단어)<br>
- `glob.glob()` : 폴더 내에 파일명이 .jpg로 끝나는 데이터 모두 호출

<br>

사용자가 보유하고 있는 데이터들이 어떤 형식으로 구성되어있냐에 따라 코드는 다양하게 변경될 수 있다. 위 코드는 한 폴더 내에 여러 class의 이미지들이 섞여있을 때 사용한 코드이다.

<br>

### 2. torchvision.datasets.ImageFolder

<br>

위에서 다뤘던 `Dataset`은 단순히 데이터들을 하나로 묶어주는 형식이었다. 하지만 각 이미지 데이터들이 파일명으로 레이블이 되어있는 상태일 때 사용하면 매우 유용한 함수가 있다.  `ImageFolder`을 이용하면 데이터와 레이블을 한 번에 만들 수 있다.

<br>

<center>
<img  src="/public/img/pytorch/data_fruit.JPG" width="" style='margin: 0px auto;'/>
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

- Class를 따로 정의할 필요없이 바로 데이터셋이 정의된다.
- `DataLoader`와 연계하여 바로 학습에 필요한 데이터를 호출할 수 있다.
- 파일명이 각 레이블이 되고, 각 레이블은 0~n까지의 숫자로 자동 치환이 된다.(SoftMax을 사용할 수 있도록)
- 채널이 1개인 이미지에 대해서는 자동으로 3개의 채널로 확장한 후에 호출이 된다.<br>
(따라서 흑백 사진들은 1채널을 복사해서 3채널까지 자동으로 같은 값을 할당한다.)

<br>

# 실제 데이터를 이용한 실습
<hr>

실제로 필자가 Pytorch로 작은 딥러닝 프로젝트를 하기 위해 직접 데이터를 수집하고 `FasterRCNN` 모델을 이용해 학습을 한 경험이 있다. 따라서 이 데이터를 이용해서 `Dataset`을 어떻게 활용하는지에 대해 알아보겠다.([여기](https://github.com/gjustin40/DeepLearning-Project/blob/master/Voca_Detection_Project_with_FasterRCNN(Pytorch).ipynb)을 참고)

<br>

수집했던 데이터는 다음과 같다.(TOEIC 영어책)
<center>
<img  src="/public/img/pytorch/customdata.jpg" width="" style='margin: 0px auto;'/>
<figcaption> 사진2. 수집한 데이터</figcaption> </center>

<br>

필자는 영어공부를 할 때 모르는 단어를 밑줄로 쳐놓고 나중에 그 단어들만 다시 본다. 그래서 그 단어들만 추출하기 위해 Object Detection 분야 중 하나인 OCR(Optical Character Recognition) 프로젝트를 진행했었다. 입력값은 이미지 데이터가 되고, 예측해야 하는 건 아래 사진과 같이 bbox의 좌표값이다.(x1, y1, x2, y2)

<center>
<img  src="/public/img/pytorch/data_label.jpg" width="" style='margin: 0px auto;'/>
<figcaption> 사진2. 데이터와 bbox 레이블(표 == dataframe)</figcaption> </center>

<br>

따라서 데이터셋을 만들 때 호출해야하는 요소는 **이미지**와 각 단어들에 대응하는 **bbox의 좌표값(4개)**이다. 아래는 데이터셋을 만드는 코드이다.
```python
from torch.uilts.data import Dataset
import pandas as pd

data_path = '../data/voca_detection/rename/'

class voca_dataset(Dataset):
    
    def __init__(self, data_path, dataframe, train=True, transform=None):
        self.data_path = data_path # 데이터의 경로
        self.dataframe = dataframe # dataframe 데이터(bbox 좌표모음)
        self.transform = transform # 미리 정의한 transform
        self.image_ids = self.dataframe['image_id'].unique() # 각 이미지의 고유값(id 값)
        self.train = train # 학습용일 때 사용

    def __len__(self):
        
        return len(self.image_ids)
    
    def __getitem__(self, idx):        
        image_id = self.image_ids[idx] # 이미지의 id값으로 1개씩 호출
        
        # 이미지 불러오기
        img = Image.open(self.data_path + image_id + '.jpg')
        
        # 각 이미지에 있는 bbox 좌표값 호출
        boxes = self.dataframe.loc[self.dataframe['image_id'] == image_id, ['x1', 'y1', 'x2', 'y2']]
        boxes = torch.from_numpy(np.array(boxes)).type(torch.FloatTensor)
        
        # 각 bbox들은 분류가 필요 없기 때문에 label은 전부 1로 통일
        labels = torch.ones((boxes.shape[0]), ).type(torch.int64)
        
        targets = {}
        targets['boxes'] = boxes
        targets['labels'] = labels
        
        # 학습용 transform을 적용
        if self.train:
            t = self.transform['train']
            img = t(img)
            
        # 테스트용 transform을 적용
        else:
            img = self.transform['valid'](img)
            
        return img, targets
```
- `dataframe` : 데이터분석 라이브러리인 `pandas`에서 제공하는 표 형식의 데이터
- 각 요소(bbox, label 등)에 따라 사용되는 데이터의 타입(int, float 등)이 다르다.

<br>

지금까지 Custom 데이터셋을 정의하고 불러오는 방법에 대해 알아보았다. 본인이 보유하고 있는 데이터를 자유자재로 정의하고 불러오며 수정하는 작업은 딥러닝을 연구하는데 있어서 매우 필수적인 테크닉이다. 따라서 다양한 데이터들을 이용해 데이터셋을 customize하는 연습을 해야할 필요가 있다.

<br>

이번 포스터에서 사용된 코드의 풀버젼은 [여기](https://github.com/gjustin40/Pytorch-Cookbook/blob/master/Beginner/Pytorch4_2_DataLoader(Folder%2C%20Custom).ipynb)에서 볼 수 있다.

<br>

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**


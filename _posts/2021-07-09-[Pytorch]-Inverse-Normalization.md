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

이미지 데이터셋에도 Normalization을 적용하면 많은 이득을 얻을 수 있다. Pytorch에서는 Normalization을 쉽게 적용할 수 있도록 ```Datasets```을 생성할 때 ```torchvision.transforms```메소드를 이용한다.

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
```
- ```transforms.Compose``` : 이미지 데이터에 적용할 여러 메소드들을 한 번에 묶는 메소드(이미지 변환 및 Augmentation)
- ```transforms.ToTenr``` : array 자료형이었던 이미지 데이터를 tensor 자료형으로 변환
- ```mean&std``` : 각각 RGB에 적용되는 수치

<br>

## 정규화 된 이미지 시각화
<hr>

하지만 정규화가 된 이미지를 보기 위해 시각화를 하면 원본 사진과는 약간의 괴리가 있다.<br>
(아래 코드는 이미지를 시각화하는 함수를 정의했다.)

```python
def img_show(dataloader):
    images, labels = iter(dataloader).next()      
    grid = torchvision.utils.make_grid(images, padding=1)
    grid = grid.permute(1,2,0)

    plt.imshow(grid)
```
- ```mark_grid``` : Pytorch에서 제공하는 메소드로, batch로 묶인 이미지를 grid로 변환한다.

<br>

<img  src="/public/img/pytorch/normalized_image.jpg" width="400" style='margin: 0px auto;'/>



<br>

위 사진을 보면 원본 사진을 보지 않더라도 뭔가 좀 어색하다는 것을 느낄 수 있다.

<br>
정규화가 적용된 이미지의 픽셀값들은 0과 1사이로 변환되기 때문에 Float형으로 바뀐다. 하지만 이미지의 픽셀값은 int형(0~255)이라 ```plt.imshow()```메소드가 호출될 때 float형을 int형으로 바꿔주는 작업이 필요하다. 다행히 ```plt.imshow()```메소드가 자동으로 변환을 해주지만, 이 과정에서 0~1의 값을 0~255값으로 비율에 맞게 변환하기 때문에 원본과는 다소 차이가 나는 결과가 나온다.

<br>

따라서 ```plt.imshow()```메소드의 자동 변환에 의지하는 것 보다는 정규화를 역으로 적용하여 원본 이미지의 픽셀값으로 바꿔준 후 시각화를 해야한다.

<img  src="/public/img/pytorch/inverse_normalized_image.jpg" width="400" style='margin: 0px auto;'/>


<br>

## Inverse Normalization(역정규화)
<hr>

역정규화를 변역하면 보통 **DeNormalization**이라는 표현을 쓰는데, Pytorch에서 검색을 하다보면 **Inverse Normalization**이라는 표현을 더 많이 쓰는 것을 알 수 있다. 그럼 **Inverse Normalization**을 하는 방법에 대해 알아보자.

<br>

역으로 가기 위해서는 정방향이 어떻게 이루어졌는데 알아야하기에, 정규화 식을 보면 다음과 같다.

<br>

$$
z = \frac{x - \mu}{\sigma}
$$

<br>

이미지 데이터는 보통 3개의 채널(RBG)로 이루어져있기 때문에 channel요소가 포함된다. 위 식은 정규화식을 포현한 것이고, 이미지 데이터에 적용하면 다음과 같이 표시할 수 있다. 물론 흑백 이미지일 경우 channel은 1개 뿐이다.([공식 문서 참고](https://pytorch.org/vision/stable/transforms.html))

<br>

$$
output[channel] = \frac{input[channel] - mean[channel]}{std[channel]}
$$

$$
output = z,\quad input = x,\quad mean = \mu, \quad std = \sigma
$$


<br>

다른 포스터에서도 자주 언급했지만 필자가 매우 멍청이라 역으로 정규화를 하는 과정을 여러 번 시도 끝에 이해할 수 있었다. 이해를 쉽게 하기 위해 코드와 수식을 같이 표시할 예정이다.

<br>

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                             std=[1/0.229, 1/0.224, 1/0.225])

inv_normalized_images = inv_normalize(normalized_image)
```

코드를 보면 **inverse Normalization**의 경우 mean값과 std값에 다양한 수치가 적용된 것을 알 수 있는데, Normalization된 데이터에 위와 같은 값으로 Normalization을 진행하면 원본 이미지를 얻을 수 있다. 수치를 자세히 보면 Normalization을 적용했던 mean과 std의 조합으로 구할 수 있다.

$$
\begin{align}
\mu_{inverse} & = -\frac{\mu}{\sigma} \\
\sigma_{inverse} & = \frac{1}{\sigma} \\
x_{normalized} & = \frac{x - \mu}{\sigma} \\
\end{align}
$$

mean과 std가 각각 위와 같은 식으로 변환되고, **Inverse-Normalization**에서 input(x)으로 들어가는 값은 Normalization이 된 값이다. 따라서 구해야 하는 값(원본 이미지 데이터 $x$값)은 다음과 같다.

$$
\quad \\
z_{inverse} = \frac{x_{normalized} - \mu_{inverse}}{\sigma_{inverse}} = x
\quad \\
\quad 
$$


위 식에 $(1)$,$(2)$,$(3)$을 대입하면 다음과 같다.

$$
\begin{aligned}
\frac{\frac{x-\mu}{\sigma} - (-\frac{\mu}{\sigma})}{\frac{1}{\sigma}} & = \frac{\frac{x-\mu+\mu}{\sigma}}{\frac{1}{\sigma}} \
= \frac{\frac{x}{\sigma}}{\frac{1}{\sigma}} = x
\end{aligned}
$$

원본 이미지 데이터인 $x$로 복구를 했다.

<br>

역정규화를 하는 방법에 대해 알았으니, 코드에 적용시켜 원본 이미지를 시각화하면 다음과 같다.

```python
def img_show(dataloader, inv_norm=True):
    images, labels = iter(dataloader).next()

    if inv_norm:
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                             std=[1/0.229, 1/0.224, 1/0.225])
        images = inv_normalize(images)

    grid = torchvision.utils.make_grid(images, padding=1)
    grid = grid.permute(1,2,0)
    
    plt.axis('off')
    plt.imshow(grid)

img_show(trainloader, inv_norm=True)
```

<img  src="/public/img/pytorch/inverse_normalized_image.jpg" width="400" style='margin: 0px auto;'/>

<br>

지금까지 정규화(Normalization)된 이미지를 역정규화(Inverse Normalization)을 하는 방법에 대해 알아보았다. 이미지로 딥러닝 프로젝트를 진행한다면 필수로 사용하는 방법이라 Pytorch 자체에서 utils로 제공할 줄 알았지만, 아무리 찾아봐도 보이지가 않았다. 여러 Pytorch 커뮤니티에서도 위 식처럼 직접 대입해서 푸는 방법이 대부분이었다. 이제 방법을 알았으니 유용하게 사용할 일만 남았다!

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**
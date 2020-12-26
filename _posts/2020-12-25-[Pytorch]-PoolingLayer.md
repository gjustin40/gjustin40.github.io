---
layout: post
title: "[Pytorch] - 폴링 계층(Pooling Layer) 종류"
date: 2020-12-25 19:00:00
category: Pytorch
use_math: true
---

이번 포스터에서는 Convolution Layer을 거쳐서 나온 Feature maps을 Resizing 하여 새로운 Layer를 얻는 **폴링(Pooling)** 에 대해 알아보자.

# 폴링 계층(Pooling Layer)
<br>

CNN 알고리즘을 구성하는 필수적인 layer 중 하나인 Pooling Layer는 주로 Convolution Layer 연산을 거쳐 나온 Feature map이나 마지막 출력을 할 때 사용이 된다.

<center>
<img  src="../public/img/pytorch/pooling_layer.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진1. Pooling Layer </figcaption>
</center>

<br>

Pooling Layer에서 사용하는 필터는 값이 따로 존재하지 않고 단지 '기능'을 수행한다. 즉, Matrix 연산을 사용하지 않고 이미지의 픽셀로부터 값을 뽑아내는 역할을 한다. 다른 Filter들과 동일하게 Size와 Stride을 설정할 수 있꼬 보통 Size는 2x2를 사용하고 Stride는 Size에 맞게 2로 설정한다.(stride를 1로 설정하여 overlapping을 할 때도 있다.)

<br>

### Pooling Layer 특징
Pooling Layer 특징은 다음과 같다.
- Convolution Layer 다음에 사용이 된다.
- Feature Map의 Size를 줄인다.(DownSampling)
- 전체적으로 파라미터의 수와 연산량을 줄인다.(Computation 효율 증가)
- 작게나마 Overfitting에도 효과가 있다.(어느 정도 있다고는 한다....)

<br>

# Pooling Layer의 종류
<hr>

Pooling Layer읠 종류로는 약 4가지가 있다. 물론 Min-Pooling은 거의 사용하지 않지만, 그래도 Max가 있으면 그 반대도 있기 마련이기에 개념은 존재한다. 

<br>

### Min-Pooling
제목 그대로 2x2 Filter를 이용해 최솟값을 추출하는 방법이다. 물론 max가 있다면 min이 있는 건 당연하지만, Google에 검색을 아무리 해도 Min-Pooling을 사용하는 이유에 대해서는 정확히 언급된 사항이 없다. 최솟값을 추출하기 때문에 이미지가 전체적으로 어두어지는 효과가 있다. 

> 사용을 너무 안 하는 나머지 pytorch에서도 따로 메소드를 제공하지 않는다....나중에 구현해봐야겟네

<br>

### Max-Pooling
Min과 반대로 최대값을 추출하는 방법이다. 대부분의 pooling Layer들은 Size와 Stride을 2로 고정한다. 픽셀값 중 최대값을 추출하기 때문에 적용이 된 이미지는 보통 튀는 부분이 부각된다. Edge Filter와 비슷한 기능을 한다고 생각할 수 있다. 

<br>

Pytorch 코드는 다음과 같다.

```python
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('../data/example.jpg')
maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

img_tensor = transforms.ToTensor()(img)
img_maxpool = maxpool(img_tensor)

img_size = img.size
maxpool_size = np.array(img_maxpool).shape[1:]

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.title(f'Original {img_size}')

plt.subplot(1,2,2)
plt.imshow(np.array(img_maxpool.permute(1,2,0)))
plt.axis('off')
plt.title(f'MaxPool {maxpool_size}')

plt.savefig('maxpool_result.jpg', bbox_inches='tight')
```
- `nn.MaxPool2d()` : 2d라는 건 2차 Tensor에서 사용하는 MaxPooling Filter라는 뜻이다.
- `kernel_size`와 `stride` 옵션을 통해 크기를 설정할 수 있다.

<br>

<center>
<img  src="../public/img/pytorch/maxpool_result.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진2. Original vs Maxpool </figcaption>
</center>

<br>

사실 위 사진에서는 튀는 부분을 부각?했다는 느낌은 들지 않지만 그래도 Maxpool이 적용된 사진을 보면 주위에 비해 유난히 튀는 부위만 남아있는 모습을 볼 수 있다. 

<br>

Max-Pooling Layer의 가장 큰 장점은 **Translation Invariance**의 효과이다. 이미지 내에 있는 물체가 이동을 해도 결과에는 변화가 없다는 뜻이다. 즉, 물체의 '고유'한 특징을 잘 잡아낼 수 있다.

<br>

아래 사진은 Translation Invariance을 가장 잘 보여주는 예시이다.

<center>
<img  src="../public/img/pytorch/benefit_maxpool.JPG" width="400" style='margin: 0px auto;'/>
<figcaption> 사진3. MaxPooling Layer의 장점(Translation Invariance)
<figcaption> 출처 : https://www.quora.com/How-exactly-does-max-pooling-create-translation-invariance</figcaption>
</center>

<br>

### Average-Pooling
<br>

최솟값과 최댓값을 추출했다면, 마지막으로 **평균** 값을 추출하는 Pooling Layer도 있다. 느낌적으로 알 수 있듯이 주위 픽셀값의 평균은 곧 '불러' 효과를 준다. 노이즈를 줄이는 효과가 있다고 한다.

```python
img = Image.open('../data/example.jpg')
avgpool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

img_tensor = transforms.ToTensor()(img)
img_avgpool = avgpool(img_tensor)

img_size = img.size
avgpool_size = np.array(img_avgpool).shape[1:]

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.title(f'Original {img_size}')

plt.subplot(1,2,2)
plt.imshow(np.array(img_avgpool.permute(1,2,0)))
plt.axis('off')
plt.title(f'AvgPool {avgpool_size}')

plt.savefig('avgpool_result.jpg', bbox_inches='tight')
```
- `nn.AvgPool2d()` : Max Layer와 동일하게 각종 옵션을 통해 크기를 설정할 수 있다.

<center>
<img  src="../public/img/pytorch/avgpool_result.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진3. Original vs Avgpool </figcaption>
</center>

사진을 보면 아주 미세하게 흐려진 것을 알 수 있다. Blur 효과가 적용되었다는 뜻인데, 보통 노이즈를 줄이는 효과가 있다.


### Global Average-Pooling
<br>

사실 Average-Pooling Layer도 잘 사용하지 않는 추세이다. 단지 Blur의 효과를 얻기 위해서 사용하기에는 효율적이지 못 하기 때문이다. 하지만 **Global Average-Pooling Layer**는 여러 유명한 모델에서 사용되고 있다. Average-pooling Layer와 비슷한 개념이지만, 좀 더 넓은 범위에 대해 적용이 되기 때문에 **Global**이라는 단어가 추가되었다. 

<br>

**Global Average-Pooling**의 원리는 간단하다. 2x2, 3x3 등의 Filter를 이용하는 것이 아니라 Feature Map 전체에 대해 평균을 계산한다. 결국 Feature Map 1개당 1개의 벡터값이 나온다고 할 수 있다. 

<center>
<img  src="../public/img/pytorch/globalpooling.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진4. Global Average Pooling Layer </figcaption>
<figcaption> 출처 : https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/</figcaption>
</center>

<br>

**Global Averge Pooling Layer(GAP)**의 가장 큰 장점은 기존에 CNN에서 사용하던 Fully-connected Layer를 대체할 수 있다는 것이다. 대부분의 모델에서 파라미터의 수에 가장 큰 영향을 주던 부분이 Classifier를 하던 FC Layer이다. 따라서 이 부분을 대체함으로서 파라미터의 수를 급격하게 줄일 수 있었다.

또한 또 다른 장점으로는 **GAP**를 사용하면 Model을 좀 더 'Interpretable`하게 만들 수 있다는 것이다. 인간이 보이게 좀 더 직관적으로 이해할 수 있도록 해주는 효과가 있다는 뜻이다. 

<center>
<img  src="../public/img/pytorch/globalpooling_benefit.JPG" width="" style='margin: 0px auto;'/>
<figcaption> 사진5. GAP의 장점 </figcaption>
<figcaption> 출처 : https://strutive07.github.io/2019/04/21/Global-average-pooling.html</figcaption>
</center>

Feature Map과 카테고리의 관계에 직접적으로 영향을 줘서 '공간정보'에 대한 손실을 최소화할 수 있다.

<br>

이 밖에도 여러가지 장점이 있다.

- 고정된 입력 size가 필요없다.(FC의 경우 고정된 input size가 필요하다.)
- 결론적으로 파라미터의 수를 감소시키기 때문에 과적합도 방지한다.

<br>


# pooling Layer을 사용하는 이유


- adaptive pooling 오..... 출력값을 설정하면 값에 맞게 pooling의 size와 stride가 자동으로 설정된다.



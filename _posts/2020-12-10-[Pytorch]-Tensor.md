---
layout: post
title: '[Pytorch] - Tensor'
date: 2020-12-10 22:00:00 
category: Pytorch
---
AI기술이 향상되면서 많은 사람들이 기계가 학습하는 기술, Machine Learning에 대해 관심을 가지게 되었다. 나도 그 기술에 매료되어 공부를 시작했고, 그 도구로 Pytorch를 이용했다. 앞으로 Pytorch category에 올리는 포스트에는 Pytorch로 공부한 내용을 업로드 할 예정이다.

# Pytorch란?
<hr>

Python 기반의 과학 연산 패키지로 기계학습(딥러닝)을 가능하도록 도와주는 프레임워크이다. Classification, Object Detection 등 기계학습 테스크들을 쉽고 빠르게 구현할 수 있도록 여러 기능들을 제공해주는 도구라고 할 수 있다. 기계학습 프레임워크에는 Tensorflow, Caffe2 등이 있지만 Pytorch가 최근 연구에 많이 사용되어지고 초보자도 쉽게 할 수 있다고 소문이 났다. 구현속도가 다른 프레임워크들에 비해 비교적 빠르기 때문에 연구용으로 많이 사용이 된다.

# Tensor
<hr>

다차원의 행열로 여러 Type을 포함하는 데이터의 형태 중 하나이다. 쉽게 표현하면 그냥 '데이터' 그 자체이다. 보통 선형대수학에서 선형관계를 나타내는 다중선형대수학의 대상으로 정의되는데, 기계학습에서 보통 데이터를 표현할 때 자주 사용한다.
<br>
**지금부터는 실제 Pytorch을 이용해 Tensor을 다뤄보도록 하자.**

### 1. Tensor
- Pytorch에서 가장 기본이 되는 데이터 형태
- torch.Tensor() : 가장 기본 형태의 Tensor 생성이 가능하다.

```python
import torch
print(torch.__vserion__) # 1.4.0

x = torch.Tensor(5, 3) # 5x3 행렬
>>> print(x, '\n')
# tensor([[1.0653e-38, 4.2246e-39, 1.0286e-38],
#         [1.0653e-38, 1.0194e-38, 8.4490e-39],
#         [1.0469e-38, 9.3674e-39, 9.9184e-39],
#         [8.7245e-39, 9.2755e-39, 8.9082e-39],
#         [9.9184e-39, 8.4490e-39, 9.6429e-39]]) 

>>> print(type(x), x.dtype, type(y), y,dtype)
# <class 'torch.Tensor'> torch.float32 <class 'torch.Tensor'> torch.float32
```

- torch.rand() : 무작위로 초기화 된 행렬을 생성할 수 있다.

```python
x = torch.rand(5, 3) # 5x3 행렬
>>> print(x)
# tensor([[0.1592, 0.9484, 0.2205],
#         [0.0017, 0.5281, 0.5612],
#         [0.4098, 0.3284, 0.6591],
#         [0.3243, 0.4148, 0.8893],
#         [0.1024, 0.3662, 0.1407]])

>>> print(type(x), x.dtype)
# <class 'torch.Tensor'> torch.float32

>>> print(x.size())
# torch.Size([5, 3])
```

<br>
- torch.zeors() : 값이 0인 행렬 생성

```python
x = torch.zeros(5, 3)
>>> print(x)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

>>> print(x.dtype)
# torch.float32
```

<br>
- dtype 옵션을 통해 data type을 설정할 수 있다.

```python
x = torch.zeros(5, 3, dtype = torch.long) # 값은 0이고 data type이 long인 행렬 생성
>>> print(x)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

>>> print(x.dtype)
# torch.int64
```

<br>
- 직접 값을 입력하여 행렬을 생성할 수 있다.

```python
x = torch.Tensor([1, 5])
>>> print(x)
# tensor([1., 5.])
```

<br>
- torch.zeros_like() : 입력한 Tensor와 동일한 shape, dtype을 가진 Tensor 생성
- 이미 정의한 Tensor을 이용해서 동일한 shape의 행렬을 생성할 수 있다.<br>
(사용자가 새로운 값을 적용하지 않는 한 입력한 인자들의 속성을 복사한다.)

```python
x = torch.zeros(5, 3) # 값이 0인 5x3행렬
>>> print(x, x.dtype)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]) 
# torch.float32

y = torch.zeors_like(x)
print(y, y.dtype)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]) 
# torch.float32 
```

### 2. Operations
- 행렬의 사칙연산 기능을 제공한다.
- Numpy와 동일한 기능을 가지고 있다.

```python
x = torch.rand(5, 3)
y = torch.rand(5, 3)

>>> print(x, '\n')
>>> print(y, '\n')
>>> print(x + y)

# tensor([[0.2714, 0.3528, 0.7742],
#         [0.7774, 0.2155, 0.9337],
#         [0.5101, 0.5995, 0.5894],
#         [0.7326, 0.6084, 0.7885],
#         [0.3641, 0.9129, 0.0668]]) 

# tensor([[0.5241, 0.7094, 0.9346],
#         [0.5110, 0.6513, 0.8371],
#         [0.9473, 0.5164, 0.5135],
#         [0.4330, 0.2406, 0.3173],
#         [0.3300, 0.7381, 0.4619]]) 

# tensor([[0.7955, 1.0622, 1.7088],
#         [1.2884, 0.8667, 1.7708],
#         [1.4573, 1.1159, 1.1029],
#         [1.1656, 0.8489, 1.1057],
#         [0.6942, 1.6510, 0.5287]])
```

### 3. Indexing
- 행렬을 다루는 대부분의 프로그래밍 언어에는 Indexing 기능이 있다.
- 행렬의 특정 부분을 출력하거나 구간을 출력할 수 있다.

```python
x = torch.Tensor(5, 3)

>>> print(x)
>>> print(x[:, 1]) # 1열 모든 행
>>> print(x[1, :]) # 1행 모든 열

# tensor([[1.8024, 0.7773, 1.1118],
#         [1.2870, 1.3277, 1.5792],
#         [1.1237, 0.9229, 0.4539],
#         [1.2986, 1.5044, 1.3334],
#         [1.1523, 1.3030, 1.1117]]) 

# tensor([0.7773, 1.3277, 0.9229, 1.5044, 1.3030]) # 1열
# tensor([1.2870, 1.3277, 1.5792]) # 1행
```

### 4. Reshape
- 행렬의 shape을 자유롭게 변경할 수 있다.
- 대표적으로 **view()** 와 **permute()** 가 존재하는데, 약간의 차이가 있다.
- view() : 기존의 데이터와 같은 메모리를 공유하며 모양(shape)만 바꿔준다.
- permute() : 축을 기준으로 데이터를 바꿔준다.<br>
- 데이터를 다시 나열해서 reshape을 한다. -> view()
- 데이터의 형태를 보존한 상태로 축을 이용해 reshape 한다. -> permute()

```python
a = torch.tensor([[1,2], [3,4], [5,6]]) # 3x2 행렬
>>> print(a)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

>>> a.view(2, 3) # 1~6의 데이터를 1자로 펴서 다시 reshape
# tensor([[1, 2, 3],
#         [4, 5, 6]])

>>> a.permute(1,0) # 데이터의 형태를 보존한 상태에서 축을 이용해 대칭
# tensor([[1, 3, 5],
#         [2, 4, 6]])
```
<br>

- view()에서 -1 값을 이용하면 원하는 행 또는 열을 기준으로 reshape을 할 수 있다.

```python

x = torch.randn(4, 4)
y = x.view(16) # 1x16 행렬
z = x.view(-1, 8) # 8열, 나머지는 행으로(2x8)
r = x.view(8, -1) # 8행, 나머지는 열로(8x2)
```

### 5. Numpy와 Tensor의 변환
- Numpy와 Tensor는 '행렬'이라는 동일한 개념을 사용하기 때문에 서로 변환이 가능하다.

```python
import numpy as np
a = np.ones(5)
>>> print(a, type(a))
# [1. 1. 1. 1. 1.] <class 'numpy.ndarray'> 

b = torch.from_numpy(a) # 주소를 공유하는 복사이기 때문에 a값이 변하면 같이 변한다.
>>> print(a, type(a))
# [1. 1. 1. 1. 1.] <class 'numpy.ndarray'> 

np.add(a, 1, out=a)   # np.add를 사용하면 b값도 같이 변한다 <<< 이유가...(깊은 복사 vs 얕은 복사)
>>> print(a, type(a))
>>> print(b, type(b))
# [2. 2. 2. 2. 2.] <class 'numpy.ndarray'> 
# tensor([2., 2., 2., 2., 2.], dtype=torch.float64) <class 'torch.Tensor'>
```
<br>
간단하게 Tensor을 다루는 방법에 대해 알아보았다. Pytorch을 이용해서 딥러닝을 할 때 가장 기본이 되는 요소이기 때문에 충분한 연습을 통해 Tensor를 능숙하게 다룰 줄 알아야 한다. 이 밖에도 여러가지 기능이 있지만, 모든 기능을 한 번에 다 공부하는 것 보다는 프로젝트를 하거나 공부를 할 때 궁금하거나 필요하게 되었을 때 찾아보는 것이 더 효율적이다. 다음 장에는 이러한 Tensor을 이용해서 무엇을 할 수 있는지에 대해 알아보자.

### **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**
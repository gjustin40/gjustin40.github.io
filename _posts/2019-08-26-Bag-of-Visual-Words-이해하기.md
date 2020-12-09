---
layout: post
title:  "Bag-of-Visual-Words 이해하기"
date:   2019-08-26
categories: ComputerVision
tags: cv, computer vision, representation
description: Bag of Visual Words에 대하여
---

논문리뷰를 하던 도중 글에서 Bag-of-visual-words라는 개념이 나왔다. 기존에 Bag-of-words라는 개념은 NLP에서 공부한 적이 있어서 알고 있었지만, visual이 들어간 용어는 처음 봤기에 호기심을 못 참고 찾아봤다.<br>
(느낌상 비슷할 것 같긴 했지만 그래도 정확히....)

<br>



## 분류(Classification)
우리가 보통 이미지를 분류하는 문제를 다룰 때 어떻게 해왔는가? 물론 지금은 딥러닝이 많이 발전해서 end-to-end모델에 이미지를 원본(또는 약간의 수정) 그대로 넣어 분류문제를 해결하지만, 과거에 머신러닝(ML, 딥러닝이 아닌 머신러닝)을 이용하던 때에는 이미지를 컴퓨터(모델)가 이해할 수 있도록 변환시킨 후 모델에 대입했었다. 따라서 이미지를 표현(Represent)하는 여러 방법들이 연구되었고, 그 중 하나가 바로 Bag-of-Visual-Words기법이었다. computer vision분야에서 좋은 성과를 선보였던 Bag-of-Visual-Words기법에 대해 알아보자.<br>
(편의상 BOVW로 칭하자. 너무 길어요ㅠㅠ)

<br>



## Bag-of-Words(BOW)
BOVW를 알기 전에 먼저 BOW(Bag-of-Words)에 대해 알아보자. BOW는 자연어처리(NLP, Natural Language Processing)에서 흔히 사용되는 기법으로 문서(Document)들을 분류할 때 자주 사용한다. 문서들의 뭉치에 포함되어 있는 단어들의 분포를 파악하여 해당 문서가 어떤 문서인지 분류하는 기법을 말한다. 예를 들어 논문(문서)에 '세포', '바이러스', 'DNA' 등의 단어들이 자주 나오면 이 논문은 '생물학'과 관련된 내용으로 분류하게 되고, 'CO2', 'NH4' , 'O2' 등의 화학기호가 많이 나오면 '화학'과 관련된 내용으로 분류하게 된다. 문서들의 뭉치에서 나온 단어들을 가방(Bag)에 넣고 단어들의 사용 횟수(분포)를 파악하여 각 클래스별로 분류를 짓는 방식이다.

![사진-BOW](https://i.imgur.com/HlaJZ51.png)

<br>

BOW와 마찬가지로 BOVW 또한 이러한 방식을 모방한 기법인데, NLP에서는 '단어'이용해 문서를 표현했다면 Computer Vision(CV)에서는 이미지를 통해 추출되는 '특징(feature)'을 이용하여 이미지를 표현한다는 차이가 있다.
(물론 마지막에는 BOW나 BOVW 둘 다 컴퓨터가 이해할 수 있도록 벡터vector로 변환한다)

![사진-벡터로 변환하는](https://i.imgur.com/q8dO8ru.png)

<br>



## Bag-of-Visual-Words(BOVW)
<br>

### 가정(Hypothesis)
본 개념의 이해를 돕기 위해 가정을 한 가지 세워보자.<br>
지금부터 우리는 (강아지, 사람, 자동차) 이 3가지의 클래스에 대해 '분류'작업을 실시한다고 가정하자. 그러면 한 이미지에 대해 해당 이미지가 3가지 클래스 중 어느곳에 속하는지 알아내야 하는데, 그 과정을 생각하면서 개념을 이해해보자.
<br>

이미지를 분류하기 위해서는 각 클래스별로 특징이 필요하다. 즉, 클래스를 구별할 수 있는 '기준'이 있어야한다.

![사진-강아지,사람,자동차 사진](https://i.imgur.com/LVaI8ZN.png)

1. 강아지 : 4개의 다리가 있고(1), 귀가 비교적 크고(2), 털이 많다(3)
2. 사람   : 2개의 다리가 있고(1), 2개의 팔이 있으며(2), 직럽보행의 모양을 띈다(3)
3. 자동차 : 4개의 바퀴가 있고(1), 창문이 있으며(2), 모서리가 많다(3)
<br>
(물론 이 밖에도 각 클래스의 특징들은 더 많다.)

이제 각 클래스별로 드러나는 특징들을 모두 합친 후 가방(bag)에 넣어보자.

![사진-가방에 특징들이 담겨져있는 모습](https://i.imgur.com/Uscz9ij.png)

Bag-of-Viusal-Words가 완성되었다. 참 간단하죠~?
한 가방에 모든 특징들을 합치긴 했지만 클래스별로 특징들의 분포가 다르게 나타날텐데, 이것이 BOVW의 핵심이다.
<br>

### 과정(Processing)
이제 BOVW가 어떻게 만들어지는지 좀 더 구체적으로 알아보자.
(위 예시에서는 간단하게 다리, 바퀴 등 눈으로 봤을 때 구분이 되는 큰 특징들을 말했지만, 사실은 좀 더 복잡하다.(~~컴퓨터는 의외로 우리보다 똑똑하지 못하다....~~)
BOVW는 세 가지의 과정을 거쳐서 완성된다.

1. 이미지에서 Keypoints를 추출하고(feature detection)
2. 추출한 Keypoints을 설명하는 vector로 표현하고(feature description)
3. 표현된 수많은 description(vector) 중 대표할 수 있는 몇 개의 대표값을 설정(codebook generation)
4. 각 이미지들에 대한 대표값들의 히스토그램 만들기(Histogram)

<br>

**1. Feature Detection**
 - 이미지들로부터 특징점(keypoints)을 추출한다. 다양한 feature detection 알고리즘이 있지만 흔히 SIFT 알고리즘을 이용한다.(SIFT 알고리즘은 특허문제로 더이상 opencv-python에서 사용 불가...)
 - 알고리즘을 적용하면 특징점(x,y)을 얻을 수 있다.
 - 여기서 특징점(Keypoints)는 픽셀(pixel)이 될 수도 있고, 더 넓은 구역(patch)가 될 수도 있다.
 - 모든 사진에 대해 Keypoints를 추출한다.
![사진-feature 결과](https://i.imgur.com/oz0Za9Q.png)

<br>

**2. Feature Description**
 - Keypoints를 추출했다면 위에 설명한 바와 같이 대부분 (x,y)좌표로 된 값들이다.(이미지는 격자로 되어있음)
 - (x,y)좌표로는 해당 점이 무엇을 나타내는지 컴퓨터가 알 수 없다(이미지에서 단순히 위치정보만 포함).
 - 하나의 '기준'을 토대로 해당 점을 설명하는 vector를 만든다(의미부여). --> Description
 - 다양한 descriptor가 있지만 대표적으로 SIFT나 HOG등이 있다.(SIFT는 detector와 descriptor 둘 다 가능)
![사진-feature description 결과](https://i.imgur.com/9KsxIXg.png)

<br>

**3. Codebook Generation**
 - 모든 Keypoints에 대해 Description을 만들면 비슷한 부분들이 많다. 따라서 그 중 대표적인 Description들만 추출하기 위해 비지도학습 알고리즘을 적용하여 k개의 대표값(Codebook)을 추출한다.
 - 비지도학습으로 [K-means Clustering][k-means-docs] 알고리즘을 많이 사용한다.
 - k값은 결국 대표값(Codebook)의 size을 뜻하고, 결국 BOVW의 Size가 된다.
![사진-codebook 사진](https://i.imgur.com/XxXvtr5.png)

<br>

**4. Histogram**
 - 완성된 Codebook을 토대로 각 이미지마다 히스토그램을 만든다.
 - Codebook의 크기가 k라면, 히스토그램의 x축의 크기(bin의 개수)도 k가 된다.
![사진-Histogram 사진](https://i.imgur.com/xzDKDZW.png)

<br>

### 완성(Complete)
![사진-여러 이미지들에 BOVW 적용한 결과](https://i.imgur.com/KpisiSw.png)
<br>

짜란~
각 이미지들에 Bag-of-Visual-Words를 적용한 결과다. 물론 BOVW가 단순히 이미지를 다르게 표현하는 방법 중 하나지만, 이미지의 차원(Dimension)을 축소시키는 효과도 있다. 즉, 이미지에서의 특징이 더 잘 드러나도록 도와주고 기계학습을 진행할 떄 계산량도 대폭 감소시켜준다.

<br>



### BOVW 적용
그렇다면 BOVW를 위에서 가정했던 '분류'문제에 어떻게 적용시킬까? 원리는 간단하다. 이미지마다 BOVW으로 표현을 하면 Codebook의 분포가 다양하게 나타날텐데, 같은 클래스라면(강아지, 사람, 자동차) 각 클래스마다 Codebook의 분포가 비슷한 양상을 보일 것이다. 따라서 분류기(베이지안, SVM 등)를 이용해 해당 클래스의 Codebook 분포 양상을 파악하고 새로운 이미지가 들어오면 유사도를 측정해 분류를 실시한다.(실제 적용은 [다음][다음-docs] 포스트에...ㅎ)

<br>



## 요약
![사진-요약](https://i.imgur.com/V5079iu.png)
<br>

### **긴 글 읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영!)**


[k-means-docs]: https:
[다음-docs]: https:

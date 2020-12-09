---
layout: post
title:  "논문리뷰 : Character Recognition in Natural Images(2009)"
date:   2019-08-24
categories: 논문Review
tags: OCR, 논문리뷰
description: 논문리뷰
---

오늘 리뷰할 논문은 [Character Recognition in Natural Images(2009)][paper-url]이다. 우선 제목만 봐도 매우 간단하고 어떤 문제를 다루는지 직관적으로 알 수 있다. 이 논문은 VISAPP(International Conference on Computer Vision Theory and Applications)에 소개된 바 있다.

<br>



# Abstract
이 논문은 제목 그대로 일반적인 사진에서 글자를 인식하는 문제에 대해 다뤘다. 특히 당시에 사용하던 OCR 기술로 인식이 잘 되지 않았던 이미지들에 중점을 두었다. 데이터는 크게 영어와 [칸나다어][kannada-docs]이고 Bangalore, India 지역의 길거리를 일반적인 카메라로 찍은 사진이다. 이미지들은 보통 [**Bag-of-Visual-Words**][bag-of-visual-words-docs]로 표현되고 SVM과 Nearest Neighbour 분류기로 그 성능을 평가한다. 논문에서 제시된 방법에 의하면 **'15개 정도의 적은 이미지로도 상업적으로 사용되는 보통 OCR시스템보다 더 좋은 성능을 발휘한다.'**고 한다. 그럼 어떻게 이러한 결과가 나왔는지 알아보자.

<br>

# Introduction
일반 사진에서 단어를 인식하는 문제는 매우 어려운데, 그 이유에는 여러가지가 있다.
- 글자의 모양과 두께의 다양성
- 배경과 글자를 이루는 색깔과 질감(Texture)
- 카메라의 구도 및 왜곡
- 조명과 해상도

아래 사진을 보면 직감적으로 알 수 있다.

![1](https://i.imgur.com/FMCa11s.png)

당시에 사용되는 OCR System은 위와 같은 요소들 때문에 일반적인 이미지에 대해서는 성능이 좋지 않았고 소수의 언어에서만 사용이 가능하다는 문제점이 있었다. 따라서 이 논문에서는 이와 같은 문제를 해결하는데에 중점을 두었다. 또한 이미지에서 글자를 읽기 위해서는 Localization(글자 위치찾기), Segmentation(이미지에서 글자 분리하기), Recognition(글자 읽기), Integration Language Models and Context(읽어낸 문자들 합치기)등의 문제들을 해결해야 하지만, 논문에서는 Recognition에 집중했다.

**Abstract**에서 언급했듯이 논문에서 제시한 방법을 평가하기 위해서 bag-of-visual-words을 기반한 feature을 이용했다.

<br>
# Datasets
Bangalore, India의 길거리에서 찍은 *영어*와 *칸나다어*를 기본 데이터로 사용했다. 하지만 해당 사진들을 얻기 위해서는 많은 비용과 시간이 들기 때문에 hand-printed characters(손글씨)와 computer fonts(컴퓨터에 있는 폰트들)을 Trainging 데이터에 추가했다.
###### 칸나다어
칸나다어 같은 경우는 dot(점) 하나에 따라 단어가 달라지기 때문에 다른 언어에 비해 시각적으로 비슷한 단어들이 많다. 또한 일반적인 사진에서 흔히 보이는 단어들과는 달리 빈도가 매우 낮은 단어들이 있어서 글자를 인식하는데 어려움이 많다. 따라서 이와 같은 문제를 해결하기 위해 hand-printed characters와 font로 만든 이미지를 대용으로 사용했다.(*결과에 의하면 추가된 이미지들에 의해 성능이 높아졌다고 한다*.)

*영어*의 경우 대문자와 소문자(각각 26개씩) 그리고 10개의 숫자를 합쳐서 총 62개의 class를 이용했고 *칸나다어*의 경우 기본 49개의 alpha-syllabary(알파벳 자음표)와 자음과 모음의 조합으로 만들 수 있는 600개의 단어들을 합쳐 총 657개의 class를 이용했다.

1. **Natural Images Datasets(*Img*)**
 - 간판이나 광고판, 슈퍼마켓 등에서 직접 촬영한 이미지 1922장
 - 이미지에 있는 단어들을 각각의 글자 단위(a,b,c 등)로 분할
 - 분할은 rectangle bounding box와 polygonal segments 두 가지로 진행<br>(**실험 결과에 의하면 polygonal은 아무 효과가 없어서 bounding box만 사용**)
 - *영어*는 총 12503개 중 7705개의 단어 사용(상태불량인 단어 제외)
 - *칸나다어*는 총 4194개 중 3345개의 단어 사용(상태불량인 단어 제외)
![2](https://i.imgur.com/I7GKGWn.png)

2. **Hand-Printed Datasets(*Hnd*)**
 - 테블릿PC를 이용하여 다양한 두께의 펜으로 작성된 글씨
 - *영어*는 55명의 지원자들로부터 3410개의 글자 생성
 - *칸나다어*는 25명의 지원자들로부터 16425개의 글자 생성


3. **Font Datasets(*Fnt*)**
 - *영어*만 이용함
 - 254개의 폰트와 4개의 스타일(normal, bold, italic, bold+italic) 적용
 - 총 62992개의 글자 생성<br>
![3](https://i.imgur.com/gXVj2al.png)

논문에서는 위 3가지의 데이터셋을 각각 *Img*, *Hnd*, *Fnt*로 표기했다.


<br>
# Feature Extraction and Representation
이미지는 차원이 매우 크다. 이미지의 크기를 보통 200x200, 32x32 등으로 표현하는데, 픽셀로 따지만 40000개, 600개 등이 된다. 이러한 이미지들을 '분류'하기 위해서는 분류기에 넣어야 하는데, 차원이 크면 계산량도 많아지고 결과도 좋지 않다. 따라서 보통 '분류'작업을 해주기 위해서 이미지를 다르게 한다. 이 논문에서는 **'Bag-of-Visual-Words'**기법을 사용했다.

[**Bag-of-Visual-Words**][bag-of-visual-words-docs]는 이미지를 표현하는 방법 중 하나로 descriptor로 추출한 feautre들을 모아 비지도학습 알고리즘을 이용하여 대표값(visual words)을 결정하고, 이미지에서 각각의 대표값들의 히스토그램을 만들어 표현하는 방법이다.(자세한 내용은 블로그 포스트 확인!)


이 실험에서는 *영어*의 경우 310개의 단어들로부터 class별 **5개**의 대표값들을 설정하였고 *칸다나어*의 경우 1971개의 단어들로부터 class별 **3개**의 대표값들을 설정했다. 또한 feature를 추출할 때는 모양(Shape)과 엣지(Edge) 뿐만 아니라 질감(Texture)도 함께 추출했다. 사용한 descriptor는 다음과 같다.

- [Shape Contexts(SC)][descriptor-docs] : Point set과 binary image을 추출하는 descriptor, theta = 15와 r = 1로 설정
- [Geometric Blur(GB)][descriptor-docs] : Textrue를 추출하는 샘플링 기법
- [Scale Invariant Feature Transform(SIFT)][descriptor-docs]: Harris Hessian-Laplace detector로 추출되는 점
- [Spin Image][descriptor-docs] : 2차원 히스토그램으로 이미지의 밝기를 이용한 descriptor
- [Maximum Response of filters(MR8)][descriptor-docs] : texture를 추출하는 descriptor
- [Patch Descriptor{PCH}][descriptor-docs] : 가장 심플한 grid기반 추출기

<br>

# Experiments and Results
3가지 분류기를 사용하여 작업을 수행했다. [Nearest Neighbor Classification(NN)][nn-classification-docs], [Support Vector Machine(SVM))][svm-docs], [Multiple Kernel Learning(MKL)][mkl-docs]. 또한 OCR System 중 하나인 ABBYY FineReader 8.0^2의 결과도 같이 보였다. 추가적으로 벤치마크를 위해 ICDAR Robust Reading Competition 2003의 데이터셋도 사용했다.

실험은 대부분 *영어*로 진행되었고, 크게 3종류의 실험을 진행했다. 첫 번째 *영어* 영역에서는 같은 종류의 샘플로 학습과 평가를 진행했다(Fnt-Fnt, Hnd-Hnd, Img-Img). 두 번째는 평가를 Img데이터로 고정하고 학습을 Fnt과 Hnd을 이용했다. 마지막은 *칸나다어*를 각각 Hnd-Hnd, Hnd-Img로 짝지어서 실험했다.

**1. 같은 종류(Homogeneous set) - 영어**<br>
가장 먼저 [Nearest Neighbor Classification(NN)][nn-classification-docs]를 분류기로 사용했을 때 도출된 결과다. 클래스별 15장의 학습데이터와 테스트데이터를 사용했다.

![4](https://i.imgur.com/RzLCZll.png)


결과에 따르면 상업적으로 사용되는 OCR 엔진인 ABBYY가 Img로 학습한 모델보다 성능이 안 좋다는 것이다. 이 뿐만 아니라 ICDAR 데이터셋과 Img로 각각 학습한 후 테스트했을 때 결과도 선보였다.(학습하는 데이터의 수가 많아지면 정확도 더 오른다.)

![5](https://i.imgur.com/NtxSGth.png)

**2. 다른 종류(Hybrid set) - 영어**<br>
이 실험에서는 Fnt와 Hnd를 학습한 후 Img로 평가를 진행했다. 이 실험에서의 핵심은 **'쉽게 얻을 수 있는 데이터인 Fnt와 Hnd를 이용해 학습하면 Img로 학습한 경우와 비슷한 성능을 발휘한다.'이다. 즉, Font와 Hand-Printed Image를 Natural Image에서 글씨를 인식하는 모델을 만들기 위한 데이터셋으로 사용할 수 있다는 뜻이다.**

![6](https://i.imgur.com/w303ySv.png)

아래의 표는 *영어*로 진행한 모든 실험의 총 결과다.
(여러 Descriptor로 진행했지만 가장 성능이 좋은 GB와 SC만 표시)

![7](https://i.imgur.com/qm7goEB.png)

또 다른 insight중 하나는 **'GB와 SC로 추출한 feature를 NN Classifier로 분류했을 때 Img 15장을 학습한 후 분류한 모델보다 더 좋은 성능을 발휘한다.'**이다.

**3. 칸나다어(Kannda Datasets)**<br>
이 언어의 경우에는 어떠한 실험에도 다 좋지 않은 결과가 도출되었기 때문에 이 논문에서 딱히 강조하지 않았다. 성능이 좋지 않은 이유 중 하나로 너무 닯은 글자가 많기 때문이라고 설명했다.

<br>
## Conclusions

우선 실험결과 중 가능 우수한 성능을 발휘한 경우는 MKL 모델을 이용해 Img데이터를 학습하고 Img데이터로 평가했을 때이다(15장 per class). 또한 대소문자를 구별하지 않는다면 더 좋은 성능을 발휘한다. 두 번째로 Hnd 데이터셋은 성능에 영향을 주지 않는다. 그 이유는 손글씨로 작성된 글씨들의 style을 잡아내기에 그 데이터수가 너무 적기 때문이다. 세 번째로 Fnt데이터를 이용해 GB로 feature를 추출한 후 NN으로 분류했을 때 Hnd보다 더 좋은 성능을 발휘했다. 더 나아가 2009년 기준 최고의 성능을 자랑했던 MKL모델과 비슷한 성능을 발휘했다. **특히 GB나 SC의 경우 글씨의 모양(Shape)을 바탕으로 feature 추출하는 방법이기 때문에 해당 feature들이 성능 향상에 많은 영향을 미친다.** 정리하자면

1. Fnt로 학습했을 때와 Img로 학습했을 때 비슷한 성능
2. Shape base인 GB나 SC descriptor가 성능에 좋은 영향을 줌
3. *칸나다어*는 아직 글씨를 읽어내는데 한계가 있음

```
실제 데이터 대신 Font같이 구하기가 쉽고 저렴한 데이터로 충분히 Natural Image에서 글씨를 인식할 수 있는 모델을 만들 수 있다.
```


<br>
이상으로 [Character Recognition in Natural Images(2009)][paper-url]에 대한 논문리뷰를 마치겠습니다.<br>
비록 그대로 해석한 것처럼 방대?하지만 나름 많이 요약한거라 많은 도움이 됐으면 좋겠습니다.

### **긴 글 읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영!)**


[paper-url]: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
[kannada-docs]: https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=kannada&oquery=isolated&tqi=UTZ2idp0JywssKuMh1hssssssbh-114897
[bag-of-visual-words-docs]: https://gjustin40.github.io/computervision/2019/08/26/Bag-of-Visual-Words-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0.html
[descriptor-docs]: http
[nn-classification-docs]: http
[svm-docs]: http
[mkl-docs]: http








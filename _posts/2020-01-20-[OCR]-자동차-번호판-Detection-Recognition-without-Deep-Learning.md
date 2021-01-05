---
layout: post
title:  "자동차 번호판 Detection and Recognition without Deep Learning"
date:   2020-01-20 21:00:00
categories: ComputerVision
---

오늘 포스팅할 내용은 **'자동차 번호판 인식'**이다. 제목에는 Detection, Recognition 등 영어로 멋진 척을 좀 했지만 사실 주변에 많이 사용되고 있는 번호판 인식이다. 물론 번호판 인식 기술은 매우 오래되어서 이미 인공지능 모델도 많이 나온 상황이다. 하지만 단순히 모델 하나를 딱 써서 하는 것 보다는 뭔가 바닥부터 코딩하는? 느낌을 느껴보고 싶어서 딥러닝 모델 없이 조건을 이용해 번호판 인식을 해봤다. 다양한 방법을 시도하기 위해 여러 블로그, 사이트 등에서 참고했다. 

<br>

# How to
<hr>

자동차 번호판을 보면 똑같은 모양을 가지고 있다. 모서리가 살짝 둥글거나 색깔이 조금 다를 수 있어도 사각형 안에 숫자로 구성되어 있다. 번호판의 모양, 숫자의 크기,폰트 등 하나의 포맷으로 통일되어 있어서 판과 숫자의 특징들만 잘 찾아낸다면 어렵지 않게 번호판을 찾을 수 있다.

번호판의 특징을 나열해보면
- **사각형 모양**
- **사각형의 비율**
- **글씨의 구성**
- **글씨의 비율**
- **글씨의 배열상태**

사진만 봐도 쉽게 생각해낼 수 있는 특징들이다.

각 특징들을 조건으로 설정하여 차근차근 찾아가보자.
<br>
# Coding
<hr>

모든 코딩은 Python으로 작성했다. 사용한 라이브러리는 다음과 같다.
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
```
이번 포스팅에 사용할 사진은 다음과 같다.~~(색깔이 왜 이러지....)~~
```python
image = cv2.imread('data/image/example_car2.png')
plt.imshow(image)
plt.axis('off')
height, width, channels = image.shape

print('Shape : ', image.shape)
```
![1](https://i.imgur.com/G62Ou5o.png)
## 1. 이미지 전처리
<hr>

Image을 다루는 작업을 하면 필수적으로 필요한 단계가 바로 전처리이다. 특수한 상황을 연출하여 촬영한 사진이 아닌 Scene Text 이미지를 다룰 때는 노이즈와 왜곡 등이 많아서 전처리가 반드시 필요하다.

가장 먼저 **RGB인 사진을 GRAY로 변환**했다. 번호판을 판별하는데 색깔은 크게 작용하지 않다고 판단했기 때문이다. 이 작업으로 인해 '모양'에 더 집중할 수 있다.
```python
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap = 'gray')
plt.axis('off') # 그래프의 x,y 좌표 제거
```
![1](https://i.imgur.com/Giajntt.png)

OpenCV의 특징 중 하나가 Image을 불러오면 RGB가 아닌 BGR로 적용된다. 따라서 변환 옵션을 BGR2GRAY로 설정했다.

다음은 Image 내에 있는** 잡티(노이즈)을 제거하기 위해 Blur처리**를 해주었다.
```python
image_blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)

plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
plt.imshow(image_gray, cmap = 'gray')
plt.title('Before')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_blur, cmap = 'gray')
plt.title('After')
plt.axis('off')
```
![1](https://i.imgur.com/wpFW8lv.png)
두 사진을 비교하면 살짝 뿌옇게 변하는 특징이 있다. 특정 잡티(픽셀값이 주변과 유난히 다른 부분)를 주변에 있는 픽셀값과 비슷하게 만들어주었기 때문이다. 보통 5x5또는 7x7의 필터를 적용해 평균값을 구하는 방법을 많이 사용한다. 여기서는 5x5 필터를 사용했다.

마지막으로 **Threshold을 적용**했다. 특정 임계값을 기준으로 픽셀값들을 변환하는 작업이다. Threshold을 실시하면 물체의 윤곽선을 추출할 수 있다.
```python
image_thresh = cv2.adaptiveThreshold(
    image_blur,
    maxValue=255, # 임계값
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 판단 알고리즘
    thresholdType=cv2.THRESH_BINARY,
    blockSize=7,
    C = 3
)
#image_thesh = cv2.GaussianBlur(image_thresh, ksize=(5,5), sigmaX=0)

plt.imshow(image_thresh, cmap='gray')
plt.axis('off')
# 두 사진을 비교하는 코드는 따로 작성하지 않았다.
# 위 코드를 참고하여 만들 수 있다.(Blur 적용)
```
![1](https://i.imgur.com/fPBsUPa.png)

Threshold을 하는 방법에는 여러가지가 있다. 단순히 특정 임계값(MaxValue)을 기준으로 반전을 실시하는 방법도 있고, 주변 픽셀의 평균값을 계산하여 해당 픽셀을 반전을 할지 판단을 한 번 이상 실시한 후에 반전을 실시하는 방법 등이 있다. 여기서는 2번째 방법인 **AdaptiveThreshold**을 사용했다.

두 사진을 비교하면 Blur의 필요성을 크게 느낄 수 있다. 보이는 것과 같이 오른쪽 사진이 Blur을 적용한  것이다.
<br>
## 2. 윤곽선 추출하기(Contour)
<hr>

Threshold을 통해 물체의 윤곽선을 선명하게 만들어줬다면, 지금부터는 윤곽선을 실제로 추출하는 작업을 실시할 것이다. OpenCV가 윤곽선을 추출할 때 매우 유용한 함수를 제공한다.

먼저 Image 속에서 **윤곽선을 찾는 작업**을 실시하자. FindContours 함수를 이용하면 편리하다.
```python
contours, _ = cv2.findContours(
    image=image_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

len(contours) # 1038
```
위 코드에서 선언한 contours 안에 각 윤곽선들의 좌표가 포함되어 있다. 각 옵션들은 OpenCV 홈페이지에서 확인할 수 있다.

**추출한 윤곽선을 그려보았다.**
```python
image_temp = np.zeros_like(image)
cv2.drawContours(image_temp, contours, contourIdx=-1, color=(255,255,255))
# contourIdx : index(-1인 경우 모두 그리기)
plt.figure(figsize=(10,10))
plt.imshow(image_temp)
plt.axis('off')
```
![1](https://i.imgur.com/7Jj8O0d.png)

사진의 번호판 부분을 자세히 보면 글씨들의 윤곽선이 그대로 추출된 모습을 볼 수 있다. 이제 글씨를 둘러 싼 윤곽선을 활용하면 좋은 일이 벌어질 것 같은 좋은 느낌이 든다.

이제 저 **윤곽선을 사각형으로 감싸보자**. 즉, 각각의 윤곽선을 감싸는 사각형을 추출하는 작업을 진행할 것이다. 이 또한 OpenCV에서 제공하는 함수를 활용하면 쉽게 구할 수 있다.
```python
image_temp = np.zeros_like(image)

contours_list = []
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont) # 사각형 추출
    rect_image = cv2.rectangle(image_temp, (x,y), (x+w, y+h), (255,255,255), 1)
    
    contours_list.append({
        'contour' : cont,
        'x' : x,
        'y' : y,
        'w' : w,
        'h' : h,
        'cx' : x+w/2,
        'cy' : y+h/2
    })
    
print(len(contours_list)) # 1038

plt.figure(figsize=(10,10))
plt.imshow(rect_image)
plt.axis('off')
```
![1](https://i.imgur.com/IIqHdOQ.png)

지금까지 우리는 조건을 적용하기 위해 필요한 재료들을 추출했다. 이제 저 사각형을 이용해 번호판의 글씨들에 해당하는 부분을 찾고, 해당 문자를 인식하는 과정을 진행해보자.
<br>
## 3. 조건에 맞는 사각형 찾기
<hr>

과연 저기에 보이는 수많은 사각형 중 우리가 필요한 사각형(번호판 속 문자가 포함되어 있는)은 어떤 것일까? 사실 우리는 번호판의 위치를 이미 알고 있기 때문에 해당 부분을 자세히 보면 문자들을 감싸고 있는 사각형들의 모양을 대충 알 수 있다. 그럼 이제부터 우리가 원하는 사각형이 다른 사각형과 무엇이 다른지 알아보고, 차근차근 제거하면서 목표로 다가가보자.

본격적으로 진행하기 전에 시각화를 위해 항상 반복해야 하는 작업을 함수로 만들어놓았다. 이 함수는 제거된 사각형들을 시각화하기 위해 만들었다.
```python
def DrawRectangle(img_temp, cont_dict):
    x = cont_dict['x']
    y = cont_dict['y']
    w = cont_dict['w'] # 가로
    h = cont_dict['h'] # 세로
    rect_image = cv2.rectangle(img_temp, (x,y), (x+w,y+h), (255,255,255), 1)
    
    return rect_image
```

우선 **문자의 비율**에서 특징이 있는 것 같다. 가로/세로 비율이 유난히 다른 사각형들이 보이기 때문에 이 특징을 이용하면 해당사항이 없는 사각형을 빠르게 제거할 수 있을 것 같다. 여기서 저자는 가로/세로 비율을 0.3과 1 사이로 한정했다. 사실 지금과 같이 앞으로 나오는 모든 수치들은 단순히 **느낌**으로만 설정한 값들이다. 직접 돌려가면서 적당한 수치를 찾는 것이 중요하다.
```python
matched_contours1 = []

for i, cont in enumerate(contours_list):
    if (cont['w']/cont['h'] < 1) and (cont['w']/cont['h'] > 0.3): # 0.3과 1 사이
        matched_contours1.append(contours_list[i])

print(len(matched_contours1)) # 224
```
```python
image_temp = np.zeros_like(image)
for cont in matched_contours1:
    rect_image = DrawRectangle(image_temp, cont)

plt.figure(figsize=(10,10))
plt.imshow(rect_image)
plt.axis('off')
```
![1](https://i.imgur.com/YnOXrJS.png)

비정상적으로 상이했던 사각형들이 대충 제거가 되었다.(~~너무 때려맞추는 느낌이 있긴 하다....~~)

다음 조건으로 **가로, 세로의 최소 길이**를 선정했다. 가로는 5, 세로는 15로 최소길이를 한정했다.
```python
matched_contours2 = []

for i, cont in enumerate(matched_contours1):
    if (cont['w'] > 5) & (cont['h'] > 15): # 가로는 5, 세로는 15
        matched_contours2.append(matched_contours1[i])
        
print(len(matched_contours2)) # 52
```
```python
image_temp = np.zeros_like(image)
for cont in matched_contours2:
    rect_image = DrawRectangle(image_temp, cont)

plt.figure(figsize=(10,10))
plt.imshow(rect_image)
plt.axis('off')
```
![1](https://i.imgur.com/FcjNxqy.png)

첫 번째 조건에서는 큰 놈들이 제거되었다면, 이 번에는 작은 놈들이 제거가 되었다. 점점 번호판이 보이는 느낌이 든다.
<br>
## 4. 사각형의 배열을 통해 후보 선별
지금까지는 문자를 감싸고 있는 사각형을 찾기 위한 노력을 했다. 즉, 각 사각형에 대해 매우 개인주의적?인 경향이 있었다. 이제는 저 **사각형의 배열**을 이용해 번호판에 있는 일련의 문자들의 집합을 구해보자. 번호판 속에 있는 **문자들의 집합이 가지는 특징**을 나열하면
1. **두 사각형 사이의 중심 거리**
2. **두 사각형의 각도 차이(y축을 기준으로 높낮이)**
3. **면적의 차이(너무 크면 다른 놈)**
4. **가로, 세로의 길이 차이**
5. **위 조건을 통해 나온 집합의 원소의 개수가 7개가 포함되어 있는지(번호판은 7글자)**

집합군을 구하기 전에 각 사각형들을 쉽게 구별하기 위해 각 사각형에 id을 부여했다. 
```python
contour_list = []

for i, cont in enumerate(matched_contours2):
    cont['idx'] = i
    contour_list.append(cont)
```
각 조건들을 하나하나 나눠서 차근차근 진행하고 싶었지만 코딩실력 부족으로 도저히 표현을 하지 못했다. 따라서 1~4번까지 조건들을 for구문 하나로 합쳐놓았다.
```python
matched_idx_list = []

MAX_DIST = 80 # 사이의 중심 거리(최대)
MAX_ANGLE = 12 # 각도 차이(최대)
MAX_AREA = 0.3 # 면적 차이(최대)
MAX_WIDTH = 0.3 가로 길이 차
MAX_HEIGHT = 0.3 세로 길이 차

for cont1 in contour_list:
    matched_idx = []
    matched_idx.append(cont1['idx'])
    for cont2 in contour_list:
        if cont1['idx'] == cont2['idx']:
            continue
            
        dx = abs(cont1['cx'] - cont2['cx'])
        dy = abs(cont1['cy'] - cont2['cy'])
        
        # 두 사각형의 중심의 거리
        dist = np.sqrt(dx**2 + dy**2)
        
        # 두 사각형의 각도 차
        if dx == 0:
            angle_dif = 90
        else:
            arctan = np.arctan(dy/dx)
            angle_dif = np.degrees(arctan)
            
        # 두 사각형의 면적 차이 비율
        area1 = cont1['w'] * cont1['h']
        area2 = cont2['w'] * cont2['h']
        
        area_dif = abs((area1 - area2)/area1)
        
        # 두 사각형 가로, 세로의 차
        width_dif = abs((cont1['w'] - cont2['w'])/cont1['w'])
        height_dif = abs((cont1['h'] - cont2['h'])/cont2['h'])
        
        if dist < MAX_DIST \
        and angle_dif < MAX_ANGLE \
        and area_dif < MAX_AREA \
        and width_dif < MAX_WIDTH \
        and height_dif < MAX_HEIGHT:
            matched_idx.append(cont2['idx'])
            
    matched_idx_list.append(matched_idx)

len(matched_idx_list) # 52
```
코드를 간단하게 설명하면 **1개의 사각형(R1)을 나머지 사각형(R2~n)과 한 번씩 비교하면서 위 조건을 만족하는 사각형과 짝을 이루도록 설계**했다. 이 때 모든 사각형에 대해 이와 같은 방식을 반복한다. 이 작업을 실시하면 다양한 원소의 개수를 가진 집합군이 형성된다.

이제 형성된 집합군 중 **원소의 개수가 7개인 집합군만 추출**하면 우리가 원하는 후보가 나온다.
```python
final_matched = []
for m in matched_idx_list:
    if len(m) == 7:
        final_matched.append(m)
        
len(final_matched) # 3
```
상식적으로 생각했을 때 원소의 개수가 7개인 집합군은 최소 7개가 나와야하는데 왜 3개만 나온지 도무지 모르겠다. 아직 코딩초보라 어떤 오류를 범했는지 찾을 수가 없다.(~~도와줘요 코딩변태형들~~)

아무튼 이제는 저 3개의 집합군들이 같은 사각형들의 집합으로 이루어져 있을 수 있기 때문에 **중복되는 집합군을 제거**했다.
```python
license_idx_list = np.unique(final_matched)
len(license_idx_list) # 1
```
다행히 최종적으로 1개의 집합군만 나왔다. 이제 이 **집합군의 id를 이용해 contour로 다시 변환**한 후 우리가 찾던 번호판의 문자들이 맞는지 그림을 통해 확인했다.
```python
image_temp = np.zeros_like(image)

if type(license_list[0]) == dict: # 집합군이 1개
    for cont in license_list:
        rect_image = DrawRectangle(image_temp, cont)
    plt.figure(figsize=(10,10))
    plt.imshow(rect_image)
    plt.axis('off')
    
else:
    for cont_list in license_list: # 집합군이 2개 이상
        for cont in cont_list:
            rect_image = DrawRectangle(image_temp, cont)
    plt.figure(figsize=(10,10))
    plt.imshow(rect_image)
    plt.axis('off')
```
![1](https://i.imgur.com/Id7tXox.png)
<br>
## 5. 번포한 부분 자르기(Crop)
<hr>

이제 최종적으로 선정된 저 **집합군의 위치를 실제 사진에서 잘라보았다**. 사각형 모양 그대로 자르게되면 문자모양 그대로만 보일 뿐 번호판을 확인할 수가 없다. 따라서 약간에 padding을 추가해 사각형보다 더 크게 넉넉히 잘랐다.
```python
PLATE_PADDING = 1.1
# 번호판 중심, 가로, 세로 구하기
sorted_plate = sorted(license_list, key=lambda x: x['cx'])

plate_cx = (sorted_plate[-1]['cx'] + sorted_plate[0]['cx']) / 2
plate_cy = (sorted_plate[0]['cy'] + sorted_plate[-1]['cy']) / 2

plate_width = (sorted_plate[-1]['x'] - sorted_plate[0]['x'] + sorted_plate[-1]['w'])
plate_height = sorted_plate[3]['h'] * PLATE_PADDING
```
```python
# 자르기
image_crop = cv2.getRectSubPix(
    image_thresh,
    patchSize=(int(plate_width), int(plate_height)),
    center=(int(plate_cx), int(plate_cy))
)

plt.imshow(image_crop, cmap='gray')
plt.axis('off')
```
![1](https://i.imgur.com/Bu2R9dJ.png)

정확히 번호판 부분만 추출한 것을 알 수 있다. Detection은 끝났고, 마지막 Recognition만 남았다.

## 6. 문자 인식
<hr>

마지막으로 추출한 번호판 부분에서 문자만 가져와 인식하는 과정을 실시했다. 문자 인식은 OCR엔진 중 하나인 Pytesseract을 사용했다. 사용이 간단하고 한글도 지원하기 때문에 해당 엔진을 선정했다. 인식을 하기 전에 **인식률을 향상시키기 위해 전처리**를 진행했다.

```python
_, image_final = cv2.threshold(image_crop, thresh=125, maxval=255, type=cv2.THRESH_BINARY)
plt.imshow(image_final, cmap = 'gray')
plt.axis('off')
```
![1](https://i.imgur.com/Bu2R9dJ.png)

차이는 없어보이지만 우연히 깨끗하게 추출되어서 그런 것 같다. 이제 마지막 단계인 **문자인식**을 진행했다.
```python
result = pytesseract.image_to_string(image_final, lang='kor', config='--psm 7 --oem 0')

plt.imshow(image_final, cmap = 'gray')
plt.axis('off')
print(result) # 57버 4830
```
![1](https://i.imgur.com/Bu2R9dJ.png)

다행히 결과가 잘 나왔다. 사용한 Image의 번호판이 너무 정직하게 있었기 때문에 결과가 잘 나온 것 같긴 하지만 아무튼 분석이 제대로 이루어진 것 같다.

최종적으로 원본 이미지에서 **번호판의 위치를 표시**했다.
```python
image_copy = image.copy()

image_copy = image.copy()
x = sorted_plate[0]['x']
y = sorted_plate[1]['y']


image_visual = cv2.rectangle(
    image_copy, 
    (x,y), 
    (x+int(plate_width), y+int(plate_height)), 
    color=(0,0,255),
    thickness=2
)

plt.figure(figsize=(10,10))
plt.imshow(image_visual)
plt.axis('off')
```
![1](https://i.imgur.com/IMdWK04.png)

매번 딥러닝 모델을 이용해 비교적 쉽고 간단하게 이미지를 분석하는 프로젝트만 진행했었기 때문에 이렇게 밑바닥부터 찾아가는 코딩은 많이 해 본 경험이 없었다. 이 미니프로젝트를 하면서 느낀점은 딥러닝 모델이 계산하는 과정이 결국 위 과정과 동일한 것이 아닐까라는 생각이 들었다. 물론 짧게 끝났지만 꽤 재미있는 프로젝트였고 앞으로 블로그에 이와 같은 프로젝트를 많이 남길 예정이다. 다음에 또 재미있는 프로젝트를 진행해봐야겠다.

# [코드 Full버젼으로 보기](https://github.com/gjustin40/gjustin40-blog/blob/master/Car_License_Recognition.ipynb)
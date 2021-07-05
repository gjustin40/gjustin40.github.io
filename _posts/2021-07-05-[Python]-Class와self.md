---
layout: post
title: "[Python] - Class에서 사용하는 self 이해하기"
date: 2021-07-05 13:00:00
category: Python
use_math: true
---

Python의 핵심 중 하나인 Class에서 사용되고 있는 self에 대해 알아보자.

<br>

# 클래스(Class)는 무엇인가
<hr>

Python으로 코딩을할 때 보통 모든 것은 '객체'라는 표현을 쓴다. 객체는 특정한 개념이나 모양으로 존재하는 것을 말하는데, 이 것을 만들 때 사용하는 것이 클래스(Class)이다.
    
- 변수와 함수를 묶어서 새로운 객체를 만들어준다.
- 복잡한 코드를 쉽게 호출하고 사용할 수 있도록 도와준다.

```python
class MyClass:

    def method1():
        pass

    def method2():
        pass
```

self를 이해하기 위한 포스터이기 때문에 Class에 대해 더 자세히 알고 싶으면 [다음](https://blog.naver.com/PostView.nhn?blogId=kids_power&logNo=221908169295&categoryNo=54&parentCategoryNo=32) 링크를 참고하자!

<br>

## 인스턴스(Instance) 생성
정의된 Class를 이용해 하나의 객체로 저장하는 것을 '인스턴스 생성'으로 생각하면 된다. Class란 여러 객체를 찍어내기 위한 '틀'이기 때문에 한 가지 Class로 여러개의 인스턴스를 만들 수 있다.

```python
my_instance = MyClass()

print(type(my_instance))
# <class '__main__.MyClass'>

instance1 = MyClass()
instance2 = MyClass()
instance3 = MyClass()
```

<br>

## 클래스 변수와 인스턴스 변수(Class & Instance Variables)
Class 안에서 사용하는 변수는 **클래스 변수**와 **인스턴스 변수**가 있다.
<br>
**클래스 변수**는 모든 인스턴스에서 동일한 데이터를 사용할 수 있는 변수이다. 즉, 같은 Class로 각각 다른 객체를 생성해도 안에 있는 변수의 값은 언제나 동일하다.

```python
class MyClass:
    language = 'English'

person1 = MyClass()
person2 = MyClass()

print(person1.language, person2.language)
# English English
```
- English를 통해 대화하는 사용하는 사람(객체)을 생성
- 모든 객체에 대해 동일한 변수를 적용할 때 사용한다.

<br>

**인스턴스 변수**는 class를 이용하여 객체를 생성할 때 각각의 객체에 대해 다른 종류의 변수를 적용할 때 사용한다. 즉, 각 객체들을 구별할 수 있는 변수이다.

```python
class MyClass:
    language = 'English'

    def __init__(self, country):
        self.country = country
```

언어는 English로 동일하지만, English를 사용하는 국가는 다양하기 때문에 사람(객체)별로 본인이 해당하는 국가를 설정하면 그것은 각 객체를 구별할 수 있는 **인스턴스 변수**가 된다.

``python




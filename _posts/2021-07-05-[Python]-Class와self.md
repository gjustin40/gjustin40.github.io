---
layout: post
title: "[Python] - Class에 대한 전반적인 이야기"
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

Class에 대해 더 자세히 알고 싶으면 [다음](https://blog.naver.com/PostView.nhn?blogId=kids_power&logNo=221908169295&categoryNo=54&parentCategoryNo=32) 링크를 참고하자!

<br>

##### 인스턴스(Instance) 생성
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

##### 클래스 변수와 인스턴스 변수(Class & Instance Variables)

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

<br>

언어는 English로 동일하지만, English를 사용하는 국가는 다양하기 때문에 사람(객체)별로 본인이 해당하는 국가를 설정하면 그것은 각 객체를 구별할 수 있는 **인스턴스 변수**가 된다.

```python
person1 = MyClass('America')
person2 = MyClass('England')

print(person1.country, person2.country)
# America England
```
- 국가(인스턴스 변수)가 다른 사람(객체)를 생성
- 하지만 두 사람(객체) 모두 같은 언어(클래스 변수)를 사용한다.

<br>

##### 생성자(Constructor)
Python을 다뤄본 사람이라면 ```__init__```이라는 함수를 본 적이 있을 것이다. ```___init___```이란 Class를 호출할 때 자동으로 실행되는 함수이다.

```python
class MyClass():
    def __init__(self, 변수1, 변수2):
        self.속성1 = 변수1
        self.속성2 = 변수2
        ...
```
- 객체를 만들 때 호출되는 특별한 메서드
- 변수를 정의할 때는 ```self.속성(변수)``` 형태를 이용하면 된다.
- 여기서 사용하는 변수와 속성은 같은 표현이다.

<br>

class를 정의할 때 항상 나타나는 **self**는 과연 무엇일까?

<br>

##### self는 나 자신이다!

class로 객체를 생성할 당시 '객체'에 해당하는 부분이 self인 것이다. class 내부에 있는 메소드나 변수, 속성들은 객체가 생성된 후에 **객체.메소드**나 **객체.속성** 처럼 호출이 되기 때문에 self 인자를 사용하는 것이다.<br>
(반드시 self일 필요는 없고 다른 글자를 사용해도 가능하다.) 

```python
class MyClass:
    
    intro = '제 이름은 {}이고 나이는 {}입니다.'
    
    def __init__(aaa, name, age): # self 말고 aaa도 가능
        aaa.name = name
        aaa.age = age
        pass
    
    def introduce(aaa):
        return aaa.intro.format(aaa.name, aaa.age)
        
person1 = MyClass('justin', 25)
person2 = MyClass('sakong', 20)

print(person1.introduce(), person2.introduce())
```
- 인스턴스를 생성할 때 **생성자**를 이용하여 인스턴스 변수를 만들면 관리가 쉽다.
- 객체로부터 호출할 때 매우 편리하다.

<br>

Class에 대해 전반적으로 알아보았다. Python을 다루면서 많이 사용했는데, 내부 요소들을 이해하고 사용한 적은 없었던 것 같다. 제3자가 물어봤을 때 대답을 못 할 것 같다는 생각이 많았는데, 지금은 어느정도 설명은 할 수 있을 것 같다는 자신감이 샘솟는다. 생각없이 코딩을 한 자신을 반성하게 되는 시간이었다.

<br>

## **읽어주셔서 감사합니다.(댓글과 수정사항은 언제나 환영입니다!)**
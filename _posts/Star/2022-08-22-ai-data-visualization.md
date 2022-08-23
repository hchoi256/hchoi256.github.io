---
layout: single
title: "PART 1: Data Visualization Techniques"
categories: Study
tag: [Data Visualization, Python]
toc: true
toc_sticky: true
toc_label: "쭌스log"
# author_profile: false
header:
    teaser: /assets/images/posts/data-visual.jpg
sidebar:
    nav: "docs"
---

```python
import matplotlib.pyplot as plt
import numpy as np
```

****
# 단순 Plot

```python
fig, ax = plt.subplots() # plot() 보다 세부적으로 plot 조정 가능
ax.plot([1, 2, 3, 4], [1, 4, 5, 7])
```

![image](https://user-images.githubusercontent.com/39285147/186052561-e00911e9-6995-475c-bba1-6d3d6da9a6f4.png)


```python
x = np.linspace(0, 2, 100) # 0 ~ 2 사이는 균등하게 100개로 쪼개라
```


```python
# Objected Oriented Style

fig, ax = plt.subplots()
ax.plot(x, x, label = "linear")
ax.plot(x, x**2, label = "quadratic")
ax.plot(x, x**3, label = "cubic")
ax.set_title("Simple plot")
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.legend() # 범주 박스 추가
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186053025-a9f5e2a8-ccdc-4dbb-a991-41c2ba547217.png)


****
# Plot Style

```python
# 임의 데이터
data1, data2, data3, data4 = np.random.randn(4, 100)
data1.shape, data2.shape, data3.shape, data4.shape
```


        ((100,), (100,), (100,), (100,))


## Pyplot Style (Plot 배치 형태)

```python
fig, (ax1, ax2) = plt.subplots(1, 2) # Plot 배치 유형
ax1.plot(data1, data2)
ax1.set_title("first plot")
ax2.plot(data3, data4)
ax2.set_title("second plot")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054017-c5971905-4a6a-416e-ac1c-29d855f482a2.png)


```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (6, 10))
ax1.plot(data1, data2)
ax1.set_title("first plot")
ax2.plot(data3, data4)
ax2.set_title("second plot")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054163-b74fb20f-16c3-44ed-99fb-949475181255.png)

## Object Oriented Style (Plot 모양)

```python
# 임의 데이터
x = np.arange(100)
x, type(x)
```

        (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
        numpy.ndarray)

        

```python
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(x, np.cumsum(x), c = "#891414", alpha = 0.2, lw = 5, ls = "--")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054258-08c1f521-079f-473b-ad17-4bc8c1a0c099.png)



```python
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(x, np.cumsum(x), c = "#891414", alpha = 0.2, lw = 5, ls = ":")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054285-22ba8fcc-04b7-4bf2-91d8-f6772b4be8dd.png)


```python
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(x, np.cumsum(x), c = "#891414", alpha = 0.2, lw = 5, ls = "-.")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054302-0580976e-3532-409f-acaf-38cbd1ef2e36.png)


```python
fig, ax = plt.subplots(figsize = (10, 8))
ax.scatter(data1, data2)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054406-a2704024-bef7-4281-b685-4b625245b83d.png)


```python
plt.rcParams["figure.dpi"] = 200
fig, ax = plt.subplots()
ax.scatter(data1, data2, marker = "^", label = "First figure")
ax.scatter(data3, data4, marker = "3", label = "Second figure")
ax.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186054489-b79f3339-795d-4e02-8e32-288c182306b7.png)

****
# 폰트 지정

```python
# 1. 윈도우 폰트 저장위치에서 폰트 불러오기
import matplotlib.font_manager
font_list = matplotlib.font_manager.findSystemFonts(fontpaths = None, fontext = "ttf")
font_list
```

        ['C:\\Windows\\Fonts\\PALSCRI.TTF',
        'C:\\WINDOWS\\Fonts\\trebucbi.ttf',
        'C:\\WINDOWS\\Fonts\\ERASDEMI.TTF',
        'C:\\Windows\\Fonts\\BOOKOSBI.TTF',
        'C:\\WINDOWS\\Fonts\\LATINWD.TTF',
        'C:\\WINDOWS\\Fonts\\HARLOWSI.TTF',
        ...
        'C:\\WINDOWS\\Fonts\\MOD20.TTF',
        'C:\\WINDOWS\\Fonts\\georgiaz.ttf',
        'C:\\WINDOWS\\Fonts\\wingding.ttf',
        'C:\\Windows\\Fonts\\LFAXD.TTF',
        'C:\\Windows\\Fonts\\BOD_BLAI.TTF']

        
```python
# 2. 원하는 위치에서 폰트 불러오기
from pathlib import Path

fpath2 = Path("./fonts/SANGJU Haerye.ttf")
```

****
# 사인/코사인 Plot

```python
# 사인/코사인
x = np.linspace(0, 10, 101)
y1 = np.sin(x)
y2 = np.cos(x)
```


```python
plt.rcParams["figure.dpi"] = 200 # 도형 해상도(크기) 2배 확대
plt.plot(x, y1, lw = 4, color= "red", label = "sin")
plt.plot(x, y2, lw = 4, color= "blue", label = "cos")
plt.xlabel("x-축 라벨", font=fpath)
plt.xticks(fontsize = 16)
plt.ylabel("y-축 라벨", font=fpath)
plt.yticks(fontsize = 16)
plt.title("사인, 코사인 그래프", font=fpath2)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186053633-b8fbff6e-9996-497b-95df-d84db86a888e.png)

****
# 다양한 함수 형태

```python
fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2)

ax1.plot(x, y)
ax1.set_yscale("linear")

ax2.plot(x, y)
ax2.set_yscale("log")

ax3.plot(x, y)
ax3.set_yscale("symlog")

ax4.plot(x, y)
ax4.set_yscale("logit")

plt.style.use("Solarize_Light2")
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186150020-68c0880e-fe5c-4308-954f-f3c00eb1df07.png)

다음 데이터 시각화 실습 [바로가기]()
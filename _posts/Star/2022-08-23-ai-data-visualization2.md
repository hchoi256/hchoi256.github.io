---
layout: single
title: "PART 2: Data Visualization Techniques"
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

****
# 복습
이전 [챕터](https://hchoi256.github.io/study/ai-data-visualization/)에서 다룬 기술들을 간단히 복기해보자.

```python
plt.rcParams["figure.dpi"] = 100
data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots(figsize = (6, 3)) # 단위가 inch
ax.scatter(data1, data2, s = 100, fc = "aqua", marker="^", label = "first datas")
ax.scatter(data3, data4, s = 100, fc = "red", marker="*", label = "second datas")
ax.set_title("산점도", fontsize = 20, color="red", loc = "left", font="Gulim")
ax.set_xlabel("x 값", fontsize = 15, color= "green", loc = "left", labelpad = 10, font="Gulim")
ax.set_ylabel("y values", fontsize = 15, loc = "bottom", color= "g", labelpad = 10)
ax.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186143432-a53d3239-aadb-4935-b1f4-544530f49e63.png)

****
# Plot 글자 추가

```python
fig, ax = plt.subplots()
ax.text(
    0.4, 
    0.4,  
    "안녕하세요", 
    ha="right", # horizontal align 
    va="top", # vertical align
    rotation=45, 
    color="red",
    font = "Gulim",
    fontsize = 15,
    style = "italic",
    bbox = dict(
        boxstyle="larrow", fc = "yellow", ec = "red", lw=2, alpha = 0.4
    )
)
ax.text(
    0.4, 
    0.2,  
    "반갑습니다.", 
    ha="right", # horizontal align 
    va="top", # vertical align
    color="red",
    font = "Gulim",
    fontsize = 15,
    style = "oblique"
)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186147842-21051aa7-2376-4641-99ce-ca19abf4e595.png)


```python
fig, ax = plt.subplots(figsize = (6, 3))
x = np.arange(0.0, 5.0, 0.01)
y = np.cos(2*np.pi*x)
ax.plot(x, y, lw = 3)
ax.text(
    1.1, 
    1.5,  
    "반갑습니다.", 
    ha="right", # horizontal align 
    va="top", # vertical align
    color="red",
    font = "Gulim",
    fontsize = 15,
)
ax.annotate("최대값", xy=(3, 1), xytext=(4, 1.5), arrowprops=dict(fc="red", ec="blue"), font="Gulim")
ax.annotate("최소값", xy=(2.5, -1.0), xytext=(1, -1.5), arrowprops=dict(fc="red", ec="blue"), font="Gulim")
ax.set_ylim(-2.0, 2.0)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186148099-06f607f1-e924-4b9b-9462-1a140d059977.png)

****
# X, Y Tick 수정

```python
fig, axs = plt.subplots(2, 1)
axs[0].plot(xdata, data)
axs[0].set_title("Automatic ticks")

axs[1].plot(xdata, data)
axs[1].set_xticks(np.arange(0, 100, 30), ["zero", "thrity", "sixty", "ninety"])
axs[1].set_yticks([-1.5, 0, 1.5])
axs[1].set_title("Manual ticks")

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186148329-8b00f0ea-7ae1-4ffb-b0e8-86facfd1f36a.png)

****
# 직선 그리기

```python
fig, ax = plt.subplots(figsize = (8, 8))
ax.axhline(0.4, ls="--", color="b", lw= 3)
ax.axvline(0.6, ls=":", color="r", lw= 3)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186148502-7f4b8023-790b-4091-92bb-1615c1ef54cd.png)

****
# Legend

```python
x = np.arange(5)

fig, ax = plt.subplots()

ax.plot(x, x, label = "Linear plot")
ax.plot(x, x ** 2, label = "Quadratic plot")
ax.plot(x, x ** 3, label = "Cubic plot")
plt.style.use("fast")
plt.legend(loc = "lower left", title="Various plots", title_fontsize = 15,
    edgecolor = "#ED412B",
    facecolor ="#C6EDD0",
    shadow="True"
)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186150271-e4d1bfcc-e2dd-4b62-8bd2-96ee730502dc.png)



```python
fig, ax = plt.subplots()
line1, = ax.plot([1, 2, 3], label = "Line up", linestyle="--")
line2, = ax.plot([3, 2, 1], label = "Line down", linewidth = 4)

plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186150494-7c5f6d96-907e-4e05-a9e4-0ab2120c3340.png)

****
# 도형

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
plt.rcParams["figure.dpi"] = 150
```

```python
path = Path(verts, codes)
patch = patches.PathPatch(path, facecolor="orange", edgecolor="black")
patch_rect = patches.Rectangle( (0.5, 0.5), 1, 0.6, facecolor = "white", edgecolor="black", hatch="*")
patch_polygon = patches.Polygon(polyPath, hatch = ".")
patch_circle = patches.Circle( (0.3, -0.5), radius = 0.5, hatch = "o")

fig, ax = plt.subplots()
ax.add_patch(patch)
ax.add_patch(patch_rect)
ax.add_patch(patch_polygon)
ax.add_patch(patch_circle)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186153689-8ad4cd39-9030-468b-8b68-a7abaeeede06.png)

## Bar

```python
people = ["A", "B", "C", "D"]
x_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

fig, ax = plt.subplots()
ax.bar(x_pos, performance, yerr = error)

ax.set_xlabel("Category")
ax.set_ylabel("Category Value")
ax.set_title("Catergory Performance")

ax.set_xticks(x_pos)
ax.set_xticklabels(people)

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186161723-34319d8e-1dd3-4d3e-a367-2717ff65b7f7.png)


```python
people = ["A", "B", "C", "D"]
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

fig, ax = plt.subplots()
ax.barh(y_pos, performance, xerr = error)

ax.set_xlabel("Category")
ax.set_ylabel("Category Value")
ax.set_title("Catergory Performance")

ax.set_yticks(x_pos)
ax.set_yticklabels(people)

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186159643-bcb3767b-4ad0-48ae-98e1-65ce01b7c5e7.png)

```python
data = {
    "Seoul": 10000,
    "Busan": 15000,
    "Incheon": 8000,
    "Gwangju": 20000
}

comp_names = list(data.keys())
comp_value = list(data.values())
```

```python
fig, ax = plt.subplots()
ax.barh(comp_names, comp_value, edgecolor = "black")
ax.set(xlim=[0, 25000], xlabel="Total Value", ylabel="City", title="City and Value")
labels = ax.get_xticklabels()
plt.setp(labels, rotation = 45)

def currency(x, pos):
    if x >= 100000:
        s = "${:1.1f}M".format(x / 100000)
    else:
        s = "${:1.0f}K".format(x / 1000)
    return s

ax.xaxis.set_major_formatter(currency)

ax.axvline(np.mean(comp_value), ls = "--", color= "red")

for annotation in [1, 3]:
    ax.text(23000, annotation, "평균 이상", fontsize = 16, font="Gulim")

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186550240-2b16da9f-c7bd-4b50-b0d2-f9b8976c51bc.png)


```python
x = np.arange(1, 5)
y1 = np.arange(1, 5)
y2 = np.ones(y1.shape) * 4
```

```python
fig = plt.figure()
axs = fig.subplot_mosaic([ ["bar1", "bar2", "bar3"], ["bar4", "bar2", "bar6"]])

axs["bar1"].bar(x, y1, edgecolor = "black", facecolor = "blue", hatch = "/", label = "blue")
axs["bar1"].bar(x, y2, edgecolor = "black", facecolor = "orange", hatch = ".", label = "orange", bottom = y1)
axs["bar1"].legend()

axs["bar4"].bar(x, y1, edgecolor="black", hatch = ["--", "+", "O", "\\"])
axs["bar4"].bar(x, y2, edgecolor="black", hatch = ["*", "o", "x", "."], bottom = y1)

axs["bar2"].bar(x, x * 2, hatch = ".") 

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186550128-c3596a6a-813f-4b15-800e-627440516165.png)

```python
plt.style.available
```


        ['Solarize_Light2',
        '_classic_test_patch',
        '_mpl-gallery',
        '_mpl-gallery-nogrid',
        'bmh',
        'classic',
        'dark_background',
        'fast',
        'fivethirtyeight',
        'ggplot',
        'grayscale',
        'seaborn',
        'seaborn-bright',
        'seaborn-colorblind',
        'seaborn-dark',
        'seaborn-dark-palette',
        'seaborn-darkgrid',
        'seaborn-deep',
        'seaborn-muted',
        'seaborn-notebook',
        'seaborn-paper',
        'seaborn-pastel',
        'seaborn-poster',
        'seaborn-talk',
        'seaborn-ticks',
        'seaborn-white',
        'seaborn-whitegrid',
        'tableau-colorblind10']

        

```python
data1 = 4 + np.random.normal(0, 1.5, 200)
data2 = np.random.randn(1000, 3)
```

```python
plt.style.use("fivethirtyeight")

fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2)
colors = ["blue", "green", "red"]

ax1.hist(data1, bins= 5, edgecolor="black", facecolor="pink")

ax2.hist(data1, histtype="step", linewidth = 3, density = True)

ax3.hist(data2, color=colors)

ax4.hist(data2, color=colors, stacked = True, edgecolor="black")

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186550309-845cc82e-4ebf-4138-b2bc-b4abbe8023ff.png)

```python
data = pd.read_csv("020 five-us-states.csv")
data
```

![image](https://user-images.githubusercontent.com/39285147/186997234-cad8d04a-429e-42ab-89da-312188b057f3.png)


```python
states = data.iloc[:, 0]
pop2010 = data.iloc[:, 1]
pop2020 = data.iloc[:, 2]
```

```python
fig, ax = plt.subplots()
x = np.arange(len(states))
width = 0.35

bar1 = ax.bar(x - width/2, pop2010, width, label="2010")
bar2 = ax.bar(x + width/2, pop2020, width, label="2020")

ax.set_xticks(x)
ax.set_xticklabels(states)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186997243-4f58e134-2cf4-4974-9780-e374b7d75b1f.png)

# 원형
```python
labels = "Russia", "Canada", "United States", "China", "India"
sizes = [17, 10, 10, 9, 3]

fig, ax = plt.subplots()
ax.pie(
    sizes, 
    autopct= "%1.1f%%", 
    textprops = dict(color="white"),
    shadow = True,
    counterclock = False,
    startangle = 90,
    explode = [0, 0, 0.2, 0, 0]
)
ax.legend(labels, loc = "center right", title="Land Sizes", bbox_to_anchor=(1., 0., 0.5, 1))

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186553248-860be279-20a0-429e-94b5-23859fab3d05.png)


# 이미지

```python
import matplotlib.image as mpimg
```

```python
img = mpimg.imread("<강아지 이미지>.png")

fig, ax = plt.subplots()
ax.imshow(img)
rect = patches.Rectangle( (650, 120), 300, 300, linewidth=1, edgecolor="red", alpha=0.5)
ax.add_patch(rect)
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186553258-a667a720-76bf-450c-9fce-c06efb9cc5d9.png)

```python
plt.figure()
plt.imshow(img[:,:,1], cmap="nipy_spectral")
plt.colorbar()
```

![image](https://user-images.githubusercontent.com/39285147/186553280-974cd4f7-9190-45dc-ac49-672b0d18a8ab.png)

# 3D 공간

```python
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

x = np.random.randn(20)
y = np.random.randn(20)
z = np.random.randn(20)

ax.scatter(x, y, z, marker = "*", color= "red")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/186997306-6988e280-0a5d-4749-8056-64033dc0d9c7.png)

다음 데이터 시각화 실습 [바로가기](https://hchoi256.github.io/study/ai-data-visualization3/).
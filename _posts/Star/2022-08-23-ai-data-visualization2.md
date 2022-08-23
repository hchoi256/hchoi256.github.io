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
    teaser: /assets/images/posts/data-thumbnail.jpg
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


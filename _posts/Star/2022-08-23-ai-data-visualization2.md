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


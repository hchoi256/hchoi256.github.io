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

# 

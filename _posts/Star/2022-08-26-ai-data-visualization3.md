---
layout: single
title: "PART 3: Data Visualization Techniques"
categories: AIStudy
tag: [Data Visualization, Bokeh, Seaborn, Python]
toc: true
toc_sticky: true
toc_label: "쭌스log"
# author_profile: false
header:
    teaser: /assets/images/posts/data-visual.jpg
sidebar:
    nav: "docs"
---

# Seaborn


# Bokeh

```python
import bokeh
from bokeh.plotting import figure
print(bokeh.__version__)
```

        2.4.3

## Line

```python
x = [1, 2, 3, 4, 5]
y = [6, 5, 3, 6, 5]
```

```python
# 선 생성
p = figure(
    plot_width = 600, 
    plot_height = 300,
    title = "기본 그래프"
)
p.line(x, y)
```

```python
from bokeh.io import show # 윈도우 브라우저에서 보고자 할 때
show(p)
```


```python
from bokeh.io import output_notebook # 노트북 상에서 보고자 할 댸
output_notebook()
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998198-8509c5ac-cba8-4cf5-b3df-972487bd65fd.png)


```python
from bokeh.io import output_file # 파일로 내보낼 때
from bokeh.io import reset_output # 초기화
output_file("<파일이름>.html")
show(p)
```
```python
reset_output() # 초기화
output_file("<파일이름2>.html")
```

```python
nan = float("nan")

p = figure(plot_width = 600, plot_height = 300)
p.line(
    x = [1, 2, 3, 4, 5],
    y = [4, 6, nan, 3, 5],
    line_width = 4,
    color = "rgb(144, 238, 144)",
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998455-a2db103e-ecb7-48ef-b5ed-9575de982732.png)

## Step

```python
p = figure(plot_width = 600, plot_height = 300)
p.line(
    x = [1, 2, 3, 4, 5],
    y = [4, 6, 5, 3, 5],
    line_width = 4,
    color = "rgb(144, 238, 144)",
    alpha = 0.6,
    line_dash = "dashed"
)
p.step(
    x = [1, 2, 3, 4, 5],
    y = [4, 6, 5, 3, 5],
    line_width = 3,
    color = "gold",
    mode = "center"  # after or center
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998510-92bdd785-4cec-4634-8d42-a0ca9aae5ec4.png)


## Multiple Lines

```python
p = figure(plot_width = 600, plot_height = 300)
p.multi_line(
    [[1, 3, 2], [3, 4, 6, 6]],
    [[2, 1, 4], [4, 7, 8, 5]],
    color = ["green", "orange"],
    alpha = [0.8, 1],
    line_width = [4, 3],
    line_dash = "dotted"
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998423-50536987-49f1-47b1-ab38-354e40487621.png)

# 도형

```python
p = figure(plot_width = 600, plot_height = 300)
p.asterisk(
    x = [1, 2, 3, 4, 5],
    y = [2, 5, 1, 9, 5],
)
p.diamond(
    x = [1, 2, 3, 4, 5],
    y = [6, 7, 3, 4, 5],
    size = 15,
    color = "#FF1493",
    angle = 0.8
)
p.square(
    x = [1, 2, 3, 4, 5],
    y = [4, 6, 5, 3, 5],
    size = 20,
    fill_color = "rgb(144, 238, 144)",
    line_color = "rgb(0, 100, 0)"
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998557-6aac0254-4efb-4379-9be2-d10abdbce811.png)


```python
p = figure(plot_width = 600, plot_height = 300)
p.asterisk(
    x = [1, 2, 3, 4, 5],
    y = [6, 7, 3, 4, 5],
    size = 15,
    color = "green",
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998613-8b47559c-5639-469a-a92d-4c8c39912533.png)

```python
p = figure(plot_width = 600, plot_height = 300)
p.circle(
    x = [1, 2, 3, 4, 5],
    y = [6, 7, 3, 4, 5],
    size = 20,
    color = "maroon",
    alpha = 0.5
)
p.line(
    x = [1, 2, 3, 4, 5],
    y = [4, 3, 5, 6, 5]
)

show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998596-18a67ed1-e144-4638-a1dc-f662a80bfb79.png)

# Custom

```python
p = figure(
    plot_width = 600,
    plot_height = 400,
    x_range = Range1d(-2, 10),
    y_range = Range1d(-1, 7)
)
p.patch(
    x = [2, 3, 5, 8, 6],
    y = [4, 6, 4, 5, 3],
    color = "violet",
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998665-13f35675-0fa8-4021-b579-e2cb903d5572.png)


```python
p = figure(
    plot_width = 600,
    plot_height = 400,
    x_range = Range1d(-2, 10),
    y_range = Range1d(-1, 7)
)
p.ellipse(
    x = [2, 5],
    y = [3, 6],
    width = [1, 3],
    height = 2,
    color = "tomato",
    line_width = 4,
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998732-7e6ecab1-6443-4e78-8784-9d445a650d16.png)

```python
p = figure(
    plot_width = 600,
    plot_height = 400,
    x_range = Range1d(-2, 10),
    y_range = Range1d(-1, 7)
)
p.hex(
    x = [2, 5],
    y = [3, 6],
    size = [60, 80],
    fill_color = None,
    line_color = "maroon",
    line_width = 4,
    angle = 0.2
)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998757-1cbf9cba-c599-4ea8-9a82-9e160c5d3390.png)

```python
p = figure()
p.rect(
    x = [4],
    y = [3],
    width = [6],
    height = [5],
    fill_color = "lightblue",
    line_color = "navy",
    line_width = 2,
    angle = 0.2
)
p.plot_width = 600
p.plot_height = 400
p.x_range = Range1d(-2, 10)
p.y_range = Range1d(-1, 7)
show(p)
```

![image](https://user-images.githubusercontent.com/39285147/186998779-186b6614-0820-4cd6-a8dd-06151c972218.png)
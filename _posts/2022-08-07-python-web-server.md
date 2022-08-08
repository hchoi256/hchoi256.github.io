---
layout: single
title: "Python: Web Application without Server"
categories: Python
tag: [python, webserver]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
# author_profile: false
header:
    teaser: /assets/images/posts/python-thumbnail.png
sidebar:
    nav: "docs"
---

파이썬으로 웹 서버 구동없이 웹 어플리케이션을 만들기 위해 '**streamlit**' 라이브러리가 활용된다. <span style="color: blue"> To create a web application without a server, we use the '**streamlit**' library. </span>


# Loading the library

```python
import streamlit as st
```

# How to write texts

```python
st.write("Hello, World!") # print out the message to the web page

st.markdown("""This is an H1=============""") # markdown
```

```python
st.title("title", anchor="title")
st.header("Header 1", anchor="header")
st.subheader("Subheader 1-1", anchor="subheader")
st.caption("caption")

st.code("""def myFunction()
    print("Hello, World!")
""", language="python")

st.latex(r"""a + ar + a r^2 + a r^3 + \cdots +  a r^{n-1} = \sum_{k=0}^{n-1} ar^k = a \left(\frac{1-r^{n}}{1-r}\right)""")
```

# How to load images, audio, and videos

```python
# load an image
st.image(
    "office_view.jpg",
    caption = "선유도 어떤 사무실에서"
    )

# load an audio
st.audio("demo.wav", start_time=30, format="audio/wav")

# load a video
st.video("office_view.mp4", start_time=5)
st.video("https://www.youtube.com/watch?v=myhDXSXetWU")
```

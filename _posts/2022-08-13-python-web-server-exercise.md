---
layout: single
title: "Python: PART 2: Web Application without Server"
categories: Python
tag: [python, webserver, streamlit, opencv]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
# author_profile: false
header:
    teaser: /assets/images/posts/streamlit-thumbnail.png
sidebar:
    nav: "docs"
---

****
# 강아지 품종 분류 AI 웹페이지

```python
% pip install opencv-python
```

```python
# Loading the libraries
from distutils.command.install_egg_info import to_filename
import numpy as np
import streamlit as st
import cv2 # *opencv
import tensorflow as tf
from tensorflow import keras
```

상기 라이브러리 중에 눈에 띄는 것이 있다; **cv2**

cv2 라이브러리는 opencv 패키지를 설치해서 불러올 수 있다.

*OpenCV(Open Source Computer Vision)는 다양한 영상/동영상 처리에 사용할 수 있는 실시간 처리에 능한 오픈소스 라이브러리이다.

실시간 웹서버와 같은 환경에서 활용성이 좋다!

자, 이제 여러 장의 서로 다른 강아지 품종 이미지를 학습한 사전 학습 모델 'dog_breed.h5'를 전이학습을 위해 불러오자.

> 해당 사전모델은 인터넷에서 쉽게 찾을 수 있다.

```python
model = keras.models.load_model("dog_breed.h5")
```

이후, 우리는 세 가지 강아지 품종으로 입력으로 들어온 이미지를 처리해주고 싶다고 해보자.

이 작업을 위해 우리는 최종 Layer에서 개 품종 카테고리를 세 가지로 축소해줘야 한다.

```python
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']
```

상기 세 가지 강아지 품종 중에서 그나마 가장 유사한 종으로 인풋 이미지를 분류한다.

자, 이제 이전 시간에 배웠던 웹 페이지를 디자인할 시간이다.

```python
st.title("업로드한 개의 품종를 알아보자!") # title
st.markdown("개 이미지를 업로드 해주세요!")
st.markdown("---") # division

# file uploader
dog_image = st.file_uploader(
    label = "이미지를 선택해주세요...", 
    type = ["jpg", "png"]
)
submit = st.button("예측해주세요!") # button object
```

```python
# button event handler
if submit:
    if dog_image is not None:
        # opencv의 imdecode 인자로 사용하기 위해 byte로 변환한다
        file_bytes = np.asarray(
            bytearray(dog_image.read()), 
            dtype=np.uint8
    )
    opencv_image = cv2.imdecode(file_bytes, 1) # save the input image

    st.image(opencv_image, channels = "BGR") # opencv는 RGB가 아닌 BGR로 포맷을 저장한다

    # adjusting the size
    opencv_image = cv2.resize(opencv_image, (224, 224))
    opencv_image.shape = (1, 224, 224, 3)

    Y_pred = model.predict(opencv_image) # fine-tuned model 예측 수행

    # st.write(np.argmax(Y_pred))
    st.title(str("판단한 개의 품종은 " + CLASS_NAMES[np.argmax(Y_pred)]))
```

![image](https://user-images.githubusercontent.com/39285147/184471368-46e7b321-db37-41d1-acd6-717e9478f320.png)


****
# Dataframe 분석하기

```python
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
```

어떠한 dataset을 df 객체에 저장했다고 가정하자.

```python
data_types = df.dtypes
cat_cols = tuple(data_types[data_types == "object"].index) # 범주형 컬럼 저장
```

```python
# container
with st.container():
    st.write("컨테이너")
    value_counts = df["Sex"].value_counts()
    st.write(value_counts.index)
    col1, col2 = st.columns(2)

    with col1:
        # 남녀 성비 분포 (Pie)
        st.subheader("서브 컬럼 1")
        fig, ax = plt.subplots()
        ax.pie(value_counts, autopct="%0.2f%%", labels = ["Male", "Female"]) # 원형으로 성비 표현
        st.pyplot(fig)

    with col2:
        # 남녀 성비 분포 (Bar)
        st.subheader("서브 컬럼 2")
        fig, ax = plt.subplots()
        ax.bar(value_counts.index, value_counts) # bar로 성비 표현
        st.pyplot(fig)

    with st.expander("Results of Sex Ratio"):
        st.dataframe(value_counts)
```

![image](https://user-images.githubusercontent.com/39285147/184472017-70a9dc48-5908-4104-9d56-167c49260c7f.png)


당신은 웹페이지 내에서 다른 plot 유형을 선택해서 데이터 분포를 확인하고 싶을지도 모른다.

상기 목표 역시 아래처럼 손쉽게 구현 가능하다.

```python
# 다른 plot 유형 선택
with st.container():
    st.write("asdfasdfasdf")

    chart = ("box", "violin", "kdeplot", "histogram")
    chart_selection = st.selectbox("asefaefa", chart)
    fig, ax = plt.subplots()

    if chart_selection == "box":
        sns.boxplot(x = "Sex", y="Fare", data=df, ax = ax)
    elif chart_selection == "violin":
        sns.violinplot(x = "Sex", y="Fare", data=df, ax = ax)
    elif chart_selection == "kdeplot":
        sns.kdeplot(x = df["Fare"], hue=df["Sex"], ax = ax, shade=True)
    else :
        sns.histplot(x = "Fare", hue="Sex", data=df, ax = ax)

    st.pyplot(fig)
```

![image](https://user-images.githubusercontent.com/39285147/184472175-c98abfb5-55c4-4af4-a688-521659032de1.png)

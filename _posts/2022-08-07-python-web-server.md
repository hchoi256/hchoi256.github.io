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

# Texts

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

# 이미지, 오디오, 비디오, 카메라

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

```python
# 이미지 업로드

uploaded_file = st.file_uploader(
    label = "원하는 이미지를 업로드 해주세요",
    type = ["csv"],
    accept_multiple_files = False
)
```

```python
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.shape)
    st.write(df)

st.camera_input(
    label = "카메라 입력"
)
```

```python
# 색상 선택
st.write(
    st.color_picker(
        label = "원하는 색상을 자세히 선택하세요",
        value = "#00FFAA"
    )
)
```


# Table

```python
# csv to table
df = pd.read_csv("pokemon_40.csv")

st.write(df)
```

```python
# dataframe to table
st.dataframe(
    data = df,
    width = 250,
    height = 200
)

st.metric(
    label = "강우량", 
    value = "100 mm",
    delta = "-80mm",
    help = "상황해제"
)
```

```python
# json to table
data_json2 = {
    "컬럼 A": 1,
    "컬럼 B": [{
        "A" : "11",
        "B" : "22",
        "C" : "33"
    }, {
        "D" : "44",
        "E" : "55",
        "F" : "66"
    }],
    "컬럼 C": 3,
    "컬럼 D": 4,
}

st.json(data_json2, expanded = False)
```

# Button, Checkbox, Radio, Selectbox, Slider,

```python
# 일반
if st.button(
    "첫 버튼", 
    help="내용을 확인",
    on_click = lambda : st.write("축하합니다")
):
    st.write("버튼이 눌림")
else:
    st.write("대기중")

st.button(
    label = "미사용",     
    disabled = True
)
```


```python
# 다운로드 버튼
st.download_button(
    label="현재 DataFrame 저장",
    data = df.to_csv(index=False).encode("utf-8"),
    file_name = "export_df.csv"
)
```


```python
# Checkbox
st.write(
    st.checkbox(
        label="승인 여부를 선택해주세요",
        value = True,
        help = "실제 실행할지 확인",
        key = "isConfirmed",
        on_change = lambda : st.write("실행 가능합니다!"),
        disabled = True
    )
)
```

```python
# radio
st.radio(
    label = "색상을 골라주세요",
    options = ["빨", "주", "노", "초", "파", "남", "보"],
    help = "원하는 색을 골라주세요",
    index = 3,
    horizontal=True,
    on_change = lambda : st.write("색이 이쁘네요!"),
    disabled = True
)
```

```python
# 일반 selectbox
st.selectbox(
    label = "색상을 골라주세요",
    options = ["빨", "주", "노", "초", "파", "남", "보", 1, 2, 3, 4, 6, 7, 8],
    help = "원하는 색을 골라주세요",
    index = 5,
    on_change = lambda : st.write("색을 잘 선택했습니다!"),
    disabled = False
)

# 다중 selectbox
st.multiselect(
    label = "색상을 골라주세요",
    options = ["빨", "주", "노", "초", "파", "남", "보"],
    help = "원하는 색을 골라주세요",
    on_change = lambda : st.write("여러개 고르는 중!"),
    disabled = False
)
```

**on_click**와 **on_change**는 하는 기능이 똑같다.

widget의 기능에 따라서 on_click과 on_change 각각 적용된다.

'버튼'의 경우 click이 중요하기 때문에 on_click.

'radio', 'slider'나 'selectslider'의 경우 변경이 중요하기 때문에 on_change.

```python
# slider
st.slider(
    label = "나이를 입력해주세요", 
    help = "입장가능한 나이를 확인 중",
    min_value = 18.,
    max_value = 65.,
    value = 30.,
    step = .5
)

def change_range(args):
    if args == 6:
        st.write("확인")

my_select_slide = st.select_slider(
    label = "나이를 입력해주세요", 
    help = "입장가능한 나이를 확인 중",
    options = [1, 2, 3, 4, 5, 6, 7, 8],
    key = "check_range",
    value = [4, 5],
)
```

# Text Edit

```python
# 문자 입력받기
st.text_input(
    label = "이름 입력",
    # value = "name", # text_input의 초기값
    max_chars = 500,
    help = "이름은 최대 3글자만 입력해 주세요",
    placeholder="최대 3글자까지 입력해주세요" # 처음에 입력 설명
)
```

```python
# 텍스트 박스 생성
inputvalue_addres = st.text_area(
    label = "주소 입력",
    # value = "name", # text_input의 초기값
    max_chars = 500,
    height = 200,
    help = "주소를 자세히 입력해 주세요",
    placeholder="주소를 자세히 입력해 주세요" # 처음에 입력 설명
)

st.write("입력받은 주소 : ", inputvalue_addres)
```

```python
# 숫자 인풋 받기
st.number_input(
    label = "원하는 숫자를 입력해주세요",
    min_value = 0.,
    max_value = 10.,
    value = 5.,
    step = 2.
)
```

```python
import datetime

# 날짜 입력
st.date_input(
    label = "원하는 날짜를 입력해주세요",
    help = "특정일자를 정확히 입력해주세요",
    min_value = datetime.date(2022, 1, 1),
    max_value = datetime.date(2022, 8, 31),
    value = datetime.date(2022, 7, 15)
)

# 시간 입력
st.time_input(
    label = "시간을 입력해주세요",
    # value = datetime.time(8, 45)
)
```

# Dashboard 생성하기
![image](https://user-images.githubusercontent.com/39285147/184014561-a79a7f04-db1e-4adc-9568-aebfe8f677ed.png)


```python
 # 레이아웃 열
 
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = "강수량",
        value = "200mm",
        delta="50mm"
    )

with col2:
    st.metric(
        label = "온도",
        value = "28도",
        delta="3도"
    )

with col3:
    st.metric(
        label = "습도",
        value = "40%",
        delta="-3%"
    )
```

```python
# 레이아웃 탭

tab1, tab2, tab3 = st.tabs(["이미지", "오디오", "비디오"])

with tab1:
    st.image(
        "office_view.jpg",
        caption = "선유도 어떤 사무실에서"
    )

with tab2:
    st.audio("demo.wav")

with tab3:
    st.video("office_view.mp4")
```

```python

with st.expander("어떤 설명"):
    st.write("""뭔가 자세히 설명하고 싶지만,

    지면 관계상 자세히 넣을 수는 없다.

    streamlit은 그래도 강력하고 사용하기 쉬워서
    쉽게 대시보드를 만들 수가 있다.
    """)

```

```python
# 

with st.container():
    st.radio(
        label = "값을 선택해주세요",
        options = ["R", "G", "B"]
    )

    st.text_input(
        label = "이름을 입력해 주세요"  
    )
```

```python

```

```python

```

```python

```

```python

```




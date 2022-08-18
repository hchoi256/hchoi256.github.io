---
layout: single
title: "Python: Data Crawling"
categories: Python
tag: [Python, Data Crawling]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
# author_profile: false
header:
    teaser: /assets/images/posts/streamlit-thumbnail.png
sidebar:
    nav: "docs"
---

Python으로 정부 사이트와 같은 정보 제공 사이트에서 데이터를 불러와 엑셀 파일로 변환해보자.

# 실습 예제 1: Excel

## 라이브러리

```python
from urllib import request # 웹 연동
from bs4 import BeautifulSoup # dataframe 조정
```

## 데이터 불러오기

```python
URL = "http://singsing.sejong.go.kr/pages/sub02_01.do?pageIndex=%PAGE_NUMBER%&tmpcls2=&searchMenu=&searchMenu2=&searchKeyword1=" # 세종특별자치시 공공급식지원센터

request.urlopen(url), type(request.urlopen(url)) # 해당 웹링크 접근 권한 획득

html = request.urlopen(url) # 해당 사이트 html 추출
# html.read().decode("utf-8") # 인코딩 필요시
```

```python
target_page = BeautifulSoup(html, "html5lib")
target_page
```

        <!DOCTYPE html>
        <html lang="ko"><head>
        <meta charset="utf-8"/>
        <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
        <meta content="IE=edge" http-equiv="X-UA-Compatible"/><!-- IE에서만 -->
        <meta content="telephone=no,email=no,address=no" name="format-detection"/>
        <meta content="" name="author"/>
        <meta content="세종특별자치시 로컬푸드 학교급식 지원센터" name="Keywords"/>
        <meta content="" name="description"/>


상기 출력값처럼 해당 사이트의 html을 긁어왔다.

이제, 사이트에 나와있는 데이터 테이블을 파이썬에서 dataframe으로 변환해보자.

## DataFrame 변환

해당 사이트에 나와있는 테이블의 열을 보고, Dataframe 뼈대를 작성해보자.

```python
df = pd.DataFrame(
    columns = ["NO", "대분류", "중분류", "NEIS식품명/상세식품명", "식품설명", "포장중량", "중량단위", "포장단위"]
)
```

이제 직접 상기 Dataframe에 웹 페이지의 데이터를 넣어보자.

여기서, 해당 사이트의 각 페이지에는 테이블이 딱 하나씩 있어서 테이블을 나타내는 태그인 **tbody**를 곧바로 가져와도 무방하다.

해당 사이트의 모든 페이지를 다 조회해서 나타난 데이터를 Dataframe으로 받아오자.

```python
for PAGE_NUMBER in range(9, 0, -1):
    html = request.urlopen(URL.replace("%PAGE_NUMBER%", str(PAGE_NUMBER))) # URL에서 '%PAGE_NUMBER%'과 치환
    target_page = BeautifulSoup(html, "html5lib")
    for tr in reversed(target_page.select("tbody tr")): # 행
        list_td = tr.find_all("td") # 열

        temp_df = pd.DataFrame({
            "NO" : list_td[0].text, 
            "대분류" : list_td[1].text, 
            "중분류" : list_td[2].text, 
            "NEIS식품명/상세식품명" : list_td[3].text, 
            "식품설명" : list_td[4].text, 
            "포장중량" : list_td[5].text, 
            "중량단위" : list_td[6].text, 
            "포장단위" : list_td[7].text
        }, index = [0])

        df = pd.concat([df, temp_df]) 
    
    time.sleep(1)    
```

![image](https://user-images.githubusercontent.com/39285147/185467114-08016845-f5ea-4296-9065-f3cf1e507d8e.png)

## 엑셀 저장하기

```python
now = datetime.datetime.now() # 현재 시간

df.to_excel("식재료_종류_" + now.strftime("%Y%m%d%H%M%S")  + ".xlsx", index = False)
```

# 실습 예제 2: sqlite3

여러분은 엑셀로 직접 저장하는 것이 아니라 데이터베이스에 넣고싶을 수도 있다.

이를 위해, 크게 세 가지 방법이 존재한다. 
- SQLite3 
- MySQL
- PostgreSQL

> SQL : DML (Manipulate) + DDL (Define) + DCL (Control)

> CRUD : Create (INSERT in SQL) + READ (SELECT in SQL) + UPDATE (UPDAETE in SQL) + DELETE (DELETE in SQL)

## 라이브러리

```python
import sqlite3  # 파일형 데이터베이스
```

## Connection 획득하기

```python
con = sqlite3.connect("crawling.db") # Connection 오브젝트 생성 
cur = con.cursor() # Cursor 오브젝트 생성
```

## SQL 문 실행
```python
# 테이블 생성
execute_SQL = """CREATE TABLE singsingfood (
    NO interger, 
    CategoryA text, 
    CategoryB text, 
    FoodName text, 
    FoodDesc text, 
    unit_A text, 
    unit_B text, 
    unit_C text
)
"""
cur.execute(execute_SQL)
```

```python
# 값 추가
execute_SQL = """INSERT INTO singsingfood VALUES (
    1, 
    "A", 
    "B",
    "C",
    "D", 
    "E",
    "F",
    "G"
)
"""
cur.execute(execute_SQL)
result_set.fetchone() # 하나의 행 값
```


        (1, 'A', 'B', 'C', 'D', 'E', 'F', 'G')


```python
con.commit() # 작업 내역 적용
```

```python
con.close() # Connection 종료
```

## 데이터베이스 접근 & 수정하기

앞서 'crawling.db' 이름으로 데이터 베이스를 하나 만들었는데, 여기로 접근해보자.

```python
# 데이터베이스 접근
con = sqlite3.connect("crawling.db") # DB 접근 권한
cur = con.cursor() # Cursor 획득
result_set = cur.execute("SELECT * FROM singsingfood") # SQL문 실행
result_set.fetchall() # 모든 값 보여주기
```


        [('16,304', '수산물', '어란류', '대구알', '1000g/kg/외국산/일반/대구알/냉동', '1000', 'g', 'kg'),
        ('16,305',
        '축산물',
        '한우',
        '한우(채끝)/일반',
        '1000g/kg/국산/일반/HACCP/1등급이상/한우/채끝/냉장',
        '1000',
        'g',
        ...
        '돼지고기(갈비)/친환경',
        '1000g/kg/국산/친환경/HACCP/1등급이상/돼지고기/무항생제/갈비/냉장',
        '1000',
        'g',
        'kg')]


```python
# 데이터베이스 추가
df_list = df.values.tolist() # 판다스 DataFrame을 리스트형태로 변환

# SQL문 작성
execute_SQL = """INSERT INTO singsingfood VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
cur.executemany(execute_SQL, df_list)
```

```python
# 데이터베이스 삭제
cur.execute("DELETE FROM singsingfood") # 입력한 데이터 1번에 모두 삭제
```

```python
con.commit() # 실제 DB에 반영
con.close() # 연결 종료
```


# 실습 예제 3: DB-API (SQLAlchemy)

앞서 언급했던 모든 과정을 단 몇 줄의 코드로 더 간편하게 해결 가능하다.

이를 도와주는 DB-API가 **'SQLAlchemy'**이다.

## 라이브러리

```python
from sqlalchemy import create_engine
```

앞선 예제에서 만들었던 'crawling.db'에 접근해보자.

```python
engine = create_engine("sqlite:///crawling.db", echo=True, future=True)
```

```python
df.to_sql("singsingfood2", con = engine) # 'crawling.db' 복사하여 새로운 DB 생성 
```

## *TBD*

.

.

.


---
layout: single
title: "Python: FastAPI, Openpyxl 웹 페이지"
categories: Python
tag: [Python Web App, FastAPI, Openpyxl, DB]
toc: true
toc_sticky: true
toc_label: "쭌스log"
# author_profile: false
header:
    teaser: /assets/images/posts/fastapi.png
sidebar:
    nav: "docs"
---

# FastAPI
```python
from fastapi import FastAPI

app = FastAPI()

# get : 일반적으로 웹페이지에 접속하는 방법
# post : 내가 제출한 데이터가 있으니 받아줘!

@app.get("/")
async def root():
    return {"message": "환영합니다."}
```

## 영화 추가 웹페이지 with DB 연동 
```python
from fastapi import FastAPI, HTTPException
import psycopg2
from pydantic import BaseModel
from Movie import Movie

pgdb = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
cur = pgdb.cursor()

app = FastAPI()

# movies = [{"title":"Batman", "year":2021}]
movies = [
    {"title":"", "year":0}, 
    {"title":"Batman", "year":2021}, 
    {"title":"Joker", "year":2022},
    {"title":"Lion King", "year":1999},
    {"title":"Snow white", "year":1998}, 
    {"title":"Ice age", "year":2012}, 
]

@app.get("/")
async def root():
    return {"message": "환영합니다"}

@app.get("/movies")
def get_movies():
    sql = "SELECT * FROM movies"
    cur.execute(sql)
    movies = cur.fetchall()
    return movies

@app.get("/movie/{movie_id}")
def get_movie(movie_id:int):
    sql = "SELECT * FROM movies WHERE id = %s"
    val = (movie_id, )
    cur.execute(sql, val)
    movies = cur.fetchall()
    return movies[0]

@app.get("/movie_by_title/{movie_title}")
def get_mvoie_by_title(movie_title:str):
    sql = "SELECT * FROM movies WHERE title = %s"
    val = (movie_title, )
    cur.execute(sql, val)
    movie = cur.fetchall()
    if len(movie) == 0:
        raise HTTPException(status_code=500, detail="영화가 존재하지 않습니다")
    return movie[0]

@app.delete("/movie/{movie_id}")
def delete_movie(movie_id:int):
    sql = "DELETE FROM movies WHERE id = %s"
    val = (movie_id, )
    cur.execute(sql, val)
    pgdb.commit()
    return {"message":"영화가 성공적으로 삭제되었습니다."}

@app.post("/create_movie")
def create_movie(movie:Movie):
    sql = "INSERT INTO movies (title, year, sstoryline) VALUES (%s, %s, %s)"
    val = (movie.title, movie.year, movie.storyline)
    cur.execute(sql, val)
    pgdb.commit()
    return movie

@app.post("/update_movie")
def update_movie(movie:Movie, movie_id: int):
    sql = "UPDATE movies SET title = %s, year = %s, storyline = %s WHERE id = %s"
    val = (movie.title, movie.year, movie.storyline, movie_id)
    cur.execute(sql, val)
    pgdb.commit()
    return movie
```

# Openpyxl

```python
# 라이브러리
import openpyxl 
wb = openpyxl.load_workbook("salaries.xlsx") # 데이터 불러오기

print(wb.sheetnames) # 엑셀 sheet 이름 가져오기
```

        ['그룹1', '그룹2']


```python
# 시트 셀 데이터
b2_cell = sheet["B2"] # B열 2행
c2_cell = sheet["c2"] # c열 2행

print(b2_cell.value, c2_cell.value)
print(sheet.cell(row=4, column=2).value)
print(c2_cell.row, c2_cell.column)
print(sheet["A5"].parent) # 시트 이름 가져오기
```

        40000 USA

        45000

        2 3

        <Worksheet "그룹1">


```python
print(f"Sheet Dimentions: {sheet.dimensions}")
print(sheet.max_row, sheet.max_column)
for a, b, c in sheet[sheet.dimensions]:
    print(a.value, b.value, c.value)
```

        Sheet Dimentions: A1:C14
        14 3
        Name Salary Country
        Marry  40000 USA
        Elizabeth 32000 Brazil
        Andreas 45000 Germany
        Diana 54000 USA
        Mark 72000 USA
        Dan 62000 Brazil
        Eduard 81000 Germany
        Julia 55000 USA
        Antonio 35000 Germany
        Ben 48000 Brazil
        John 61000 UK
        George 71000 UK
        Oliver 49000 UK



```python
cell_range = sheet["B2:C11"]
for a, b in cell_range:
    print(f"A: {a.value} B:{b.value}")
```

        A: 40000 B:USA
        A: 32000 B:Brazil
        A: 45000 B:Germany
        A: 54000 B:USA
        A: 72000 B:USA
        A: 62000 B:Brazil
        A: 81000 B:Germany
        A: 55000 B:USA
        A: 35000 B:Germany
        A: 48000 B:Brazil


```python
print(sheet["A5"].data_type)
print(sheet["B5"].data_type)
print(sheet["A5"].encoding)
print(sheet["B5"].encoding)
```

        s
        n
        utf-8
        utf-8

```python
```

```python
```

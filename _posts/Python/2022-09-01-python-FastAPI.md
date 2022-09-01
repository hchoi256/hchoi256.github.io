---
layout: single
title: "Python: FastAPI 웹 페이지"
categories: Python
tag: [Python Web App, FastAPI, DB]
toc: true
toc_sticky: true
toc_label: "쭌스log"
# author_profile: false
header:
    teaser: /assets/images/posts/fastapi.png
sidebar:
    nav: "docs"
---

# 사용법
```python
from fastapi import FastAPI

app = FastAPI()

# get : 일반적으로 웹페이지에 접속하는 방법
# post : 내가 제출한 데이터가 있으니 받아줘!

@app.get("/")
async def root():
    return {"message": "환영합니다."}
```

# 영화 추가 웹페이지 with DB 연동 
```python
from fastapi import FastAPI
import sqlite3

app = FastAPI()

# http method
# get : 일반적으로 웹페이지에 접속하는 방법
# post : 내가 제출한 데이터가 있으니 받아줘!
# delete : 삭제

movies = [ 
    {"title":"Batman", "year":2021}, 
    {"title":"Joker", "year":2022},
    {"title":"Lion King", "year":1999},
    {"title":"백설공주", "year":1998}
]

@app.get("/")
async def root():
    return {"message": "DB 예제!!!"}

@app.get("/movies")
def get_movies():
    connection = sqlite3.connect("datafile.db")
    cursor = connection.cursor()
    sql = "SELECT * FROM movies"
    cursor.execute(sql)
    movies = cursor.fetchall()
    connection.close()
    return movies

@app.get("/movie/{movie_id}")
def get_movie(movie_id:int):
    return movies[movie_id]

@app.delete("/movie/{movie_id}")
def delete_movie(movie_id:int):
    movies.pop(movie_id)
    return {"message": "영화가 성공적으로 삭제되었습니다."}

@app.post("/movie")
def create_movie(movie:dict):
    connection = sqlite3.connect("datafile.db")
    cursor = connection.cursor()
    sql = "INSERT INTO movies (title, year) VALUES (%s, %s)"
    val = ( movie["title"], movie["year"] )
    cursor.execute(sql, val)
    connection.commit()
    connection.close()
    return movie

@app.post("/update_movie")
def update_movie(movie_id:int, movie:dict):
    movie_to_updated = movies[movie_id]
    movie_to_updated["title"] = movie["title"]
    movie_to_updated["year"] = movie["year"]
    movies[movie_id] = movie_to_updated
    return movie_to_updated
```

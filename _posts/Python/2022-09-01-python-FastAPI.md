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

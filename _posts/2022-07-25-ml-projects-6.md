---
layout: single
title: "ML Project 6: 사용자 기반 협업 필터링 (Collaborative Filtering) - 영화 추천 시스템 (Movie Recommender Systems)"
categories: ML
tag: [machine learning, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/ml-thumbnail.jpg
sidebar:
    nav: "docs"
---

# Code
**[Notice]** [download here](https://github.com/hchoi256/machine-learning-development)
{: .notice--danger}

# Description
추천 시스템이란 가령, 아마존에서 제품을 구매하면 이 제품을 구매하는 고객들이 다른 상품에도 관심이 있을 것이라 판단해 추천해주는 방법이다. <span style="color: blue"> Providing suggestions for items that are most pertinent to a particular user</span>

'아이템'을 기반으로 사용자에게 추천을 제공한다. <span style="color: blue"> Item-based recommendation systems, not user-based</span>

이게 무슨 말인고 하니, 세상 인구 80억 명을 전부 조사하여 단 몇 편의 영화를 기호에 따라 추천해주는 것은 지극히 비효율적인 문제일 것이다. <span style="color: blue"> It will be not effective to recommend movies after investigating huge volumne of populations</span>

따라서, 사람이 아닌 아이템, 즉 영화와 같은 시간이 지나도 한결같은 것들에 기준을 두고 추천 시스템을 적용한다. <span style="color: blue"> Thus, we need to focus on consistent data like movies</span>

가령, 영화 '타이타닉'을 본 서로 다른 두 사람이 비슷한 장르의 로맨스 장르의 '어바웃 타임'을 시청했다면, 추천 시스템은 타이타닉을 본 새로운 사용자에게 어바웃 타임을 추천할 것이다. <span style="color: blue"> For example, if two people watched 'Titanic' and 'About Time', the model would suggest 'About Time' for new people who watched 'Titanic' </span>

# Loading the dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```python
movie_titles_df = pd.read_csv("Movie_Id_Titles")
movie_titles_df.head(20)
```

![image](https://user-images.githubusercontent.com/39285147/180880109-84645976-273c-4788-84bd-d7acd7986f70.png)



```python
movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating'])
movies_rating_df.head(10)
```

![image](https://user-images.githubusercontent.com/39285147/180880472-49cbbb8f-d2f9-4de9-93cd-0fb847df79fc.png)


```python
# user_id를 기준으로 두 테이블 합치기 merge two tables by 'user_id'
movies_rating_df = pd.merge(movies_rating_df, movie_titles_df, on = 'item_id') 
movies_rating_df
```

![image](https://user-images.githubusercontent.com/39285147/180881533-77bc2c71-4ac8-4d9b-b3ae-f5685614393c.png)

```python
movies_rating_df.shape
```

        (100003, 4)


# Data Visualization

```python
ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean'] # title을 기준으로 rating을 정렬한 평균값 저장 saving the average of ratings aligned by 'title'
ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']
ratings_df_count
```


        title
        'Til There Was You (1997)                                     9.0
        1-900 (1994)                                                  5.0
        101 Dalmatians (1996)                                       109.0
        12 Angry Men (1957)                                         125.0
        187 (1997)                                                   41.0
        2 Days in the Valley (1996)                                  93.0
        20,000 Leagues Under the Sea (1954)                          72.0
        2001: A Space Odyssey (1968)                                259.0
        ...
        Young Poisoner's Handbook, The (1995)                        41.0
        Zeus and Roxanne (1997)                                       6.0
        unknown                                                       9.0
        Á köldum klaka (Cold Fever) (1994)                            1.0
        Name: count, Length: 1664, dtype: float64


```python
ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)
ratings_mean_count_df.reset_index()
```

![image](https://user-images.githubusercontent.com/39285147/180882121-0e696587-3dec-4078-b2a6-52e5d3aca6b9.png)


```python
ratings_mean_count_df['mean'].plot(bins=100, kind='hist', color = 'r') 
```

![image](https://user-images.githubusercontent.com/39285147/180882141-f4292103-5f94-40c2-88a6-233e34978d3e.png)


상기 분포도에서 별점 평균이 3인 경우가 가장 많이 차지하는 것을 확인할 수 있다. <span style="color: blue"> Star rate 3 appears the most in the distribution above</span>

```python
ratings_mean_count_df['count'].plot(bins=100, kind='hist', color = 'r') 
```

![image](https://user-images.githubusercontent.com/39285147/180882216-a837effd-7434-4b2b-a8f1-a31350bb8041.png)


분포도에서 확인할 수 있듯이, 대다수의 영화가 대략 한 두번 정도 이내로 평가받은 것을 볼 수 있다. <span style="color: blue"> As shown in the distribution, most movies are rated at most two times</span>

# 아이템 기반 협력 필터링 (item-based collaborative filtering)

```python
userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')
```

상기 코드로 만들어진 테이블은 요소값으로 'rating', 즉 별점을 나타낸다. <span style="color: blue">'rating' = star rate</span>

열은 'title', 그리고 행들은 'user_id'이다. <span style="color: blue">column = 'title' and row = 'user_id'</span>

```python
titanic = userid_movietitle_matrix['Titanic (1997)']

# Let's calculate the correlations
titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])
titanic_correlations.dropna(inplace=True)

# Let's sort the correlations vector
titanic_correlations.sort_values('Correlation', ascending=False)
```

![image](https://user-images.githubusercontent.com/39285147/180883922-2164ef1f-d80f-4089-afef-55fc56e551ed.png)


영화 '타이타닉'이 다른 영화들과 연관성이 얼마만큼 있는지 *Correlation* 열의 요소값들을 통하여 확인 가능하다. <span style="color: blue"> Can check how much 'Titanic' is related to other movies using *Correlation*</span>


```python
titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation',ascending=False).head()
```

![image](https://user-images.githubusercontent.com/39285147/180883994-d9fe6bc6-509b-42a5-a077-20386409ad8b.png)

결과 테이블에서 우리는 '타이타닉'을 본 사람에게 추천해줄 영화로써 가장 적절한 영화는 타이타닉 자기자신을 제외하고 'River Wild, The (1994)'인 것을 확인한다. <span style="color: blue"> The resultant table tells us the second most appropriate movie for people who watched 'Titanic' is 'River Wild, The (1994)'</span>

그렇다면, 타이타닉처럼 특정 영화와 연관성이 높은 타영화가 아니라 전체 데이터를 대상으로 한 번에 연관성을 확인해볼 수는 없을까? <span style="color: blue"> Can't we check the correlation for all the data at once?</span>

```python
# Obtain the correlations between all movies in the dataframe
movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)
```

> *pearson* : standard correlation coefficient

```python
myRatings = pd.read_csv("My_Ratings.csv")
myRatings
```


![image](https://user-images.githubusercontent.com/39285147/180887705-dbfd782e-92d3-4f04-afc8-0078208bb656.png)



```python
similar_movies_list = pd.Series()
for i in range(0, 2):
    similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna() # Get same movies with same ratings
    similar_movie = similar_movie.map(lambda x: x * myRatings['Ratings'][i]) # Scale the similarity by your given ratings
    similar_movies_list = similar_movies_list.append(similar_movie)
```


'movie_correlations'의 데이터 수치들은 [0, 1] 사이 값으로, 서로 다른 타이틀 간의 상관관계를 나타내는 지표이다. <span style="color: blue"> 'movie_correlations' is in range [0, 1] and represents the correlation among titles</span>

따라서, '상관관계'가 가장 클 경우 그 값은 1일 것이고, 우리가 다루는 '별점'의 총점은 5점이다. <span style="color: blue"> The max **corrlation** and **star rate** is 1 and 5, respectively

따라서, [0, 1]의 상관계수에 별점을 곱하여, 그 값이 높은 순서로 나열한다면, 그것은 해당 사용자에게 가장 알맞은 추천 리스트가 도출될 것이다.<span style="color: blue">  Recommendation systems based on the value of **corrlation** * **star rate**</span>


```python
similar_movies_list.sort_values(inplace = True, ascending = False)
print (similar_movies_list.head(10))
        ```

        Liar Liar (1997)                             5.000000
        Con Air (1997)                               2.349141
        Pretty Woman (1990)                          2.348951
        Michael (1996)                               2.210110
        Indiana Jones and the Last Crusade (1989)    2.072136
        Top Gun (1986)                               2.028602
        G.I. Jane (1997)                             1.989656
        Multiplicity (1996)                          1.984302
        Grumpier Old Men (1995)                      1.953494
        Ghost and the Darkness, The (1996)           1.895376
        dtype: float64


상기 결과는 'Liar Liar'이라는 영화에 높은 별점을 선사한 해당 사용자에게 추천할만한 타영화 리스트이다. <span style="color: blue"> Recommending the shown movies for people who watched 'Liar Liar'</span>

별점으로써 표현된 가장 높은 연관성을 나타내는 'Liar Liar' 자기자신을 제외하고 'Con Air'라는 영화가 두 번째로 해당 사용자에게 가장 추천해줄 영화일 것이다. <span style="color: blue"> The movie 'Con Air' will be the best recommendation for the people. </span>
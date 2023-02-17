---
layout: single
title: "ANN - Regression: Bike Rental Prediction"
categories: ML
tag: [ANN, Regression]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/dl-thumbnail.jpg
sidebar:
    nav: "docs"
search: false
---

# Background
ANN을 이용한 자전거 대여량 예측 <span style="color: yellow">Prediction of bicycle rental volume using ANN </span>

# Loading the dataset

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
bike = pd.read_csv("bike-sharing-daily.csv")

bike.isnull().sum() # 결측치 확인 missing values
```

        instant       0
        dteday        0
        season        0
        yr            0
        mnth          0
        holiday       0
        weekday       0
        workingday    0
        weathersit    0
        temp          0
        hum           0
        windspeed     0
        casual        0
        registered    0
        cnt           0
        dtype: int64

상기 결과는 결측치가 없음을 보여준다. <span style="color: yellow">The results show that there are no missing values. </span>

```python
# 불필요한 열 제거 Removing unnecessary columns
bike.drop(labels = ["instant"], axis = 1, inplace=True) # inplace: apply changes to 'bike'

# 시계열 time series
bike.dteday = pd.to_datetime(bike.dteday, format="%m/%d/%Y") # formatting datetime
bike.index = pd.DatetimeIndex(bike.dteday) # indexing the datetime
bike.drop(labels=["dteday"], axis = 1, inplace=True) # removing the duplicate 'dteday' column
```

# Visualizing the dataset

```python
bike["cnt"].asfreq("W").plot(linewidth = 3) # by week
plt.title("Bike Usage Per Week")
plt.xlabel("Week")
plt.ylabel("Bike Rental")
```

![image](https://user-images.githubusercontent.com/39285147/182334062-7da5e907-cfbb-459f-952e-175ee471dbaa.png)



```python
bike["cnt"].asfreq("M").plot(linewidth = 3) # by month
plt.title("Bike Usage Per Month")
plt.xlabel("Month")
plt.ylabel("Bike Rental")
```

![image](https://user-images.githubusercontent.com/39285147/182334125-6875404b-a15c-4272-8ca8-303bef79478a.png)


```python
bike["cnt"].asfreq("Q").plot(linewidth = 3) # by quarter
plt.title("Bike Usage Per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Bike Rental")
```

![image](https://user-images.githubusercontent.com/39285147/182334180-d63a71ea-861c-45de-9ad1-002937fbc639.png)


```python
# 한 눈에 여러 시각화 확인 Using the visualization tool
sns.pairplot(bike)
```


![image](https://user-images.githubusercontent.com/39285147/182334415-553ca6bb-35bb-4412-92db-b88a60130547.png)


# Building the ANN

## EDA
```python
X_numerical = bike[ ["temp", "hum", "windspeed", "cnt"] ]
X_numerical
```

![image](https://user-images.githubusercontent.com/39285147/182334541-957387c3-1bb5-496b-a702-201249a31694.png)


```python
sns.pairplot(X_numerical) # correlation between independent variables
```


![image](https://user-images.githubusercontent.com/39285147/182334687-6d9a0ca1-3b4c-49d0-8e2e-645effddf397.png)


```python
X_numerical.corr() # correlation analysis
```


![image](https://user-images.githubusercontent.com/39285147/182334770-7d4ae794-a646-4ab3-9a5b-aec76cd6fca2.png)



```python
sns.heatmap(X_numerical.corr(), annot = True) # confusion matrix
```


![image](https://user-images.githubusercontent.com/39285147/182334975-f15f4f95-6213-45f3-b5b6-7dbaed6466f5.png)


*annot*: 수치 표시 <span style="color: yellow"> showing numerical values </span> 

## Preprocessing

```python
X_cat = bike[ ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"] ]
X_cat
```

![image](https://user-images.githubusercontent.com/39285147/182335361-63b27c61-e7f3-4eb8-8b1b-366bb93fef9a.png)


상기 언급된 독립변수를 신경망 학습에 사용한다. <span style="color: yellow"> We are going to train an ANN with the independent variables listed above.</span>

```python
# converting categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat.shape
```

        (731, 32)
            

```python
X_cat = pd.DataFrame(X_cat) # 테이블 형태로 데이터 확인 converting to dataframe for visualization
```

```python
X_cat
```


![image](https://user-images.githubusercontent.com/39285147/182336299-c0b2915a-a373-4a06-84b2-7234b4941cf1.png)



```python
X_numerical = X_numerical.reset_index() # 이전에 datetime이 인덱스로 지정되있었음 previously datetime set to index
X_numerical
```

![image](https://user-images.githubusercontent.com/39285147/182336665-6388c298-eb8c-4721-bbc3-b6d58af4680a.png)


```python
# integrating all the X candidates
X_all = pd.concat( [X_cat, X_numerical], axis = 1)

# removing unnecessary variables
X_all.drop("dteday", axis = 1, inplace = True)
```

```python
X = X_all.iloc[:, :-1].values
y = X_all.iloc[:, -1:].values
```

```python
X.shape, type(X)
```

        ((731, 35), numpy.ndarray)


```python
y.shape, type(y)
```


        ((731, 1), numpy.ndarray)        


## Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y)
```


## Splitting the dataset into Training set and Test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

## Designing the model

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation = "relu", input_shape = (35, )))
model.add(tf.keras.layers.Dense(units = 100, activation = "relu"))
model.add(tf.keras.layers.Dense(units = 100, activation = "relu"))
model.add(tf.keras.layers.Dense(units = 1, activation="linear"))

model.summary()
```


        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        dense_2 (Dense)             (None, 100)               3600      
                                                                        
        dense_3 (Dense)             (None, 100)               10100     
                                                                        
        dense_4 (Dense)             (None, 100)               10100     
                                                                        
        dense_5 (Dense)             (None, 1)                 101       
                                                                        
        =================================================================
        Total params: 23,901
        Trainable params: 23,901
        Non-trainable params: 0
        _________________________________________________________________


## Training the model

```python
model.compile(optimizer="Adam", loss="mean_squared_error")
epochs_hist = model.fit(X_train, y_train, epochs= 50, batch_size = 50, validation_split=0.2)
```


        Output exceeds the size limit. Open the full output data in a text editor
        Epoch 1/50
        10/10 [==============================] - 1s 43ms/step - loss: 0.1554 - val_loss: 0.0687
        Epoch 2/50
        10/10 [==============================] - 0s 13ms/step - loss: 0.0343 - val_loss: 0.0355
        Epoch 3/50
        10/10 [==============================] - 0s 13ms/step - loss: 0.0188 - val_loss: 0.0208
        Epoch 4/50
        ...
        Epoch 49/50
        10/10 [==============================] - 0s 21ms/step - loss: 0.0019 - val_loss: 0.0129
        Epoch 50/50
        10/10 [==============================] - 0s 19ms/step - loss: 0.0025 - val_loss: 0.0115



```python
epochs_hist.history.keys()
```


        dict_keys(['loss', 'val_loss'])


모델 성능 평가 피라미터로 'loss'와 'val_loss'가 있다. <span style="color: yellow">We have two evaluating parameters, 'loss' and 'val_loss'. </span>

'loss'는 테스트셋을 대상으로 학습한 손실값, 'val_loss'는 학습 데이터의 검증셋으로 도출한 손실값 분포이다.  <span style="color: yellow"> 'loss' is based on the test set, and 'val_loss' is based on the validation set that is part of the training set. </span>

## Visualizing the training and test results

```python
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["val_loss"])
plt.title("Model Loss Progress During Traning")
plt.xlabel("Epoch")
plt.ylabel("Traning Loss and Validation Loss")
plt.legend(["Traning Loss", "Validation Loss"])
```

![image](https://user-images.githubusercontent.com/39285147/182337724-49a1d046-4d1e-42d5-9737-09d950947a16.png)


```python
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = "r")
plt.xlabel("Model Predictions")
plt.ylabel("True Values")
```


![image](https://user-images.githubusercontent.com/39285147/182338196-2dc8b1c6-41b5-4b98-93f8-812ab3abb882.png)


상기 분포의 단위가 정규화된 것을 볼 수 있다. <span style="color: yellow"> It can be seen that the units of the distribution are normalized. </span>

따라서, 원래 단위로 다시 변환해주도록 하자. <span style="color: yellow"> So, let's convert it back to the original unit. </span>

```python
y_predict_org = scaler.inverse_transform(y_predict)
y_test_org = scaler.inverse_transform(y_test)
```


![image](https://user-images.githubusercontent.com/39285147/182338615-e41b09a8-0200-4e32-8995-c82dfbe1b79e.png)


## Evaluating the model

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
```

```python
RMSE = float(format(np.sqrt(mean_squared_error(y_test_org, y_predict_org)), ".3f"))
MSE = mean_squared_error(y_test_org, y_predict_org)
MAE = mean_absolute_error(y_test_org, y_predict_org)
r2 = r2_score(y_test_org, y_predict_org)
adj_r2 = 1 - (1-r2)*(n-1)/(n-147-1)
```

모델의 성능을 검증하기 위한 손실/비용 함수의 종류에는 여러 가지가 존재한다. <span style="color: yellow"> There are several types of loss/cost functions for verifying the performance of the model. </span>

각 손실함수 사용에 따른 모델 성능을 확인해보자. <span style="color: yellow"> Let's check the model performance according to the use of each loss function. </span>

```python
print(f"RMSE = {RMSE}, MSE = {MSE}, MAE = {MAE}, R2 = {r2}, Adjusted R2 = {adj_r2}")
```

        RMSE = 1070.871, 
        MSE = 1146764.4761727168, 
        MAE = 807.9534366633617, 
        R2 = 0.7237822988637166, 
        Adjusted R2 = 41.32778436589738

공통적으로 모두 수치가 낮을수록 모델의 좋은 성능을 의미한다. <span style="color: yellow">In general, the lower the number, the better the model's performance. </span>

각각에 대한 보다 자세한 설명은 생략한다. <span style="color: yellow"> A more detailed description of each will be omitted. </span>
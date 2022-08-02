---
layout: single
title: "ANN - Regression: House Sales Prediction"
categories: DL
tag: deep learning, ann, regression
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/dl-thumbnail.jpg
sidebar:
    nav: "docs"
search: false
---

# Learning Goals
ANN을 이용한 주택 가격 예측 <span style="color: blue">House Price Prediction Using ANN </span>

# Loading the dataset

**[Notice]** [Download Dataset (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
{: .notice--danger}

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

house_df = pd.read_csv("kc-house-data.csv")
house_df.info()
```


                <class 'pandas.core.frame.DataFrame'>
                RangeIndex: 21613 entries, 0 to 21612
                Data columns (total 21 columns):
                #   Column         Non-Null Count  Dtype  
                ---  ------         --------------  -----  
                0   id             21613 non-null  int64  
                1   date           21613 non-null  object 
                2   price          21613 non-null  float64
                3   bedrooms       21613 non-null  int64  
                4   bathrooms      21613 non-null  float64
                5   sqft_living    21613 non-null  int64  
                6   sqft_lot       21613 non-null  int64  
                7   floors         21613 non-null  float64
                8   waterfront     21613 non-null  int64  
                9   view           21613 non-null  int64  
                10  condition      21613 non-null  int64  
                11  grade          21613 non-null  int64  
                12  sqft_above     21613 non-null  int64  
                13  sqft_basement  21613 non-null  int64  
                14  yr_built       21613 non-null  int64  
                15  yr_renovated   21613 non-null  int64  
                16  zipcode        21613 non-null  int64  
                17  lat            21613 non-null  float64
                18  long           21613 non-null  float64
                19  sqft_living15  21613 non-null  int64  
                20  sqft_lot15     21613 non-null  int64  
                dtypes: float64(5), int64(15), object(1)
                memory usage: 3.5+ MB


# EDA

```python
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(house_df.corr(), annot=True)
```

![image](https://user-images.githubusercontent.com/39285147/182361111-97d0802d-acd4-45bb-aa0b-94e1b1bab205.png)


```python
sns.scatterplot(x = "sqft_living", y="price", data=house_df)
```

맛보기로 'sqft_living' 변수와 종속변수('Price')의 상관관계 분포도를 만들었다. <span style="color: blue"> For testing, I created a plot of the correlation between 'sqft_living' and 'Price'. </span>


![image](https://user-images.githubusercontent.com/39285147/182361658-ed16e620-ce21-42b2-8f1c-c5fda874f1a4.png)


상기 그리드는 모든 변수들 간의 상관관계 분포도를 나타낸다. <span style="color: blue"> The grid represents correlation distributions between all variables.</span>


```python
house_df.hist(bins=20, figsize=(20, 20))
```

![image](https://user-images.githubusercontent.com/39285147/182361909-5dfe48f9-90b0-49ba-bb1d-584a4c7cdb5e.png)


# Preprocessing

```python
selected_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 
'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = house_df[selected_features]
y = house_df["price"]
```

# Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
```

# Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2)
```

# Building the model

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation="relu", input_shape=(19, )))
model.add(tf.keras.layers.Dense(units = 100, activation="relu"))
model.add(tf.keras.layers.Dense(units = 100, activation="relu"))
model.add(tf.keras.layers.Dense(units = 1, activation="linear"))

model.compile(optimizer = "Adam", loss = "mean_squared_error")
epochs_hist = model.fit(X_train, y_train, epochs = 50, batch_size = 50, validation_split = 0.2)
```

                Epoch 1/50
                277/277 [==============================] - 1s 2ms/step - loss: 8.2607e-04 - val_loss: 4.8727e-04
                Epoch 2/50
                277/277 [==============================] - 0s 1ms/step - loss: 5.2934e-04 - val_loss: 4.1871e-04
                Epoch 3/50
                277/277 [==============================] - 0s 1ms/step - loss: 4.4549e-04 - val_loss: 4.3053e-04
                ...
                Epoch 49/50
                277/277 [==============================] - 0s 2ms/step - loss: 1.9294e-04 - val_loss: 2.6583e-04
                Epoch 50/50
                277/277 [==============================] - 0s 1ms/step - loss: 1.9350e-04 - val_loss: 2.8596e-04



'epoch' 개수를 증가시키면 모델이 적은 손실값과 함께 더 정확한 예측을 해낼 것이다. <span style="color: blue"> Increasing the number of 'epochs' will cause the model to make more accurate predictions with fewer losses. </span>


하지만, 이것은 **과적합** 현상을 불러올 수 있으니, 적절한 개수 설정을 수동으로 바꿔가며 확인해볼 필요가 있다. <span style="color: blue"> However, this can lead to **overfitting**, so you need to manually change the appropriate number setting to check.</span>


```python
epochs_hist.history.keys()
```

                dict_keys(['loss', 'val_loss'])

# Visualizing the results

```python
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])
```


![image](https://user-images.githubusercontent.com/39285147/182364127-10aef50f-9e03-4abf-9d80-4520885520cc.png)


만약, 학습 과정에서 epoch' 개수를 50이 아니라 100으로 했다면, 'validation loss' 분산이 더 적게 나타났을 것이다. <span style="color: blue"> If the number of epochs was 100 instead of 50 in the training process, the 'validation loss' variance would be smaller. </span>

참고로, 입력 피처의 개수를 적게 설정하면 하기 분포처럼 'Validation loss'가 굉장히 튀는 모습을 보일 것이다. <span style="color: blue"> For reference, if the number of input features is set to be smaller, the 'Validation loss' will be very bouncing like the following distribution. </span>

![image](https://user-images.githubusercontent.com/39285147/182363306-02a5eef0-a504-47c2-a4e3-054ab9fac40a.png)


이것은 마치 우리가 시험 성적에 영향을 미치는 요인 중에, 시험 전날 수면 시간이 주요한 입력 피처임에도 무시하고 있다가, 뒤늦게 모델 학습에 반영하여 모델이 새로운 데이터들에 대해 더 정확한 예측을 해내는 것과 비슷한 이치이다. <span style="color: blue"> For example, if you had an exam tomorrow but hadn't slept yesterday, you would likely screw up the exam. Similarly, we see that sleep time before the exam is a pivotal feature in training our AI model. </span>


# Predicting the Test set

```python
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/182363900-7d34d060-edd2-47e6-a2eb-95834311003b.png)


역시 시각화에서 단위가 정규화된 모습이다. <span style="color: blue"> As shown in the image, the units are normalized. </span>

원래 단위로 변환시켜보자. <span style="color: blue"> Let's convert to the original unit</span>


```python
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)

y_predict = model.predict(X_test)
plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()
```

![image](https://user-images.githubusercontent.com/39285147/182363915-2201e582-10bb-45b2-82a9-ac35d94754cd.png)

# Evaluating the model performance

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
```

                RMSE = 135724.235 
                MSE = 18421068040.530396 
                MAE = 79133.06868421813 
                R2 = 0.8656443648009154 
                Adjusted R2 = 0.8654264066441615



상기 다양한 종류의 손실값을 활용하여 모델 성능 평가가 가능하다. <span style="color: blue">It is possible to evaluate the model performance by using the above various types of loss values.</span>

좋은 모델일수록 각 손실값이 작을 것이다. <span style="color: blue">The better the model, the smaller each loss value will be.</span>

앞서 언급한 'epoch'와 같은 요인들을 수동으로 바꿔가며 각 손실값들의 변화를 관찰하여 최적의 설정값을 구하자.<span style="color: blue"> You need to manually change factors such as 'epoch' and find the optimal combination of parameters by observing the change in each loss value.</span>


또한, 혼동 행렬 같은 기법으로 성능 분석이 가능할 것이다. <span style="color: blue"> You might want to use 'confusion matrix' for performance analysis.</span>


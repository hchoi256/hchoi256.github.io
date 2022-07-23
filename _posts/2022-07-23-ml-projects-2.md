---
layout: single
title: "ML Project 2: Deep Learning - CIFAR-10 Classification"
categories: Machine Learning
tag: [ML, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/ml-thumbnail.jpg
sidebar:
    nav: "docs"
---

# 코드
**[Notice]** [download here](https://github.com/hchoi256/machine-learning-development)
{: .notice--danger}

# Learning Goals
1. 합성곱 신경망 모델 설계하여 케라스로 이미지 분류
2. Adam 옵티마이저로 신경망 가중치 최적화
3. 드롭아웃을 통한 과적합 개선
4. 모델 평가 진행 (*confusion matrix*)
5. *Image Augmentation*으로 신경망 일반화 성능 개선
6. 훈련 신경망 가중치 조작 방법

# 배경지식

## CIFAR-10 데이터 세트
![image](https://user-images.githubusercontent.com/39285147/180509114-3b492055-56eb-42c6-85b8-80fb8e077546.png)


10가지 클래스로 나누어져 있는 **6만 개의** **컬러(RGB 채널)** 이미지로 구성된다 (airplanes, cars, birds, cats, etc.).
- 이미지 해상도가 **32x32** 픽셀로 매우 낮다.
- 각 클래스마다 6천개의 이미지가 존재한다 (클래스 별 매우 균등한 이미지 분포).

이번 프로젝트에서 주어진 입력 이미지가 10개의 클래스 중 어디에 속하는지 분류 모델을 학습시켜보자.

## [Convolutional Neural Network (CNN) 기초](https://github.com/hchoi256/ai-boot-camp/blob/main/ai/deep-learning/cnn.md)
![image](https://user-images.githubusercontent.com/39285147/180512062-48c118e4-c9d4-4ea6-8281-958201289626.png)

CNN 관련 배경지식은 상기 링크를 통해 숙지해주세요!

![image](https://user-images.githubusercontent.com/39285147/180513584-f47d8136-4cc3-473b-b85f-437ddd376101.png)

'sharpen' 커널 필터를 적용하면 인풋 이미지의 3x3 픽셀 범위에 대해 가운데 값에 가중치를 높게줘서 출력 이미지에서 가운데 픽셀을 뚜렷하게 강조한다.

## 성능지표: Key Performance Indicators (KPI)
![image](https://user-images.githubusercontent.com/39285147/180517136-7b390f93-0f67-4e21-9217-a482e74a1f41.png)

**Precision**: 암이 없는 환자에게 있다고 오진할 확률 50%
**Recall**: 암이 있는 환자에게 없다고 오진할 확률이 11%

> ![image](https://user-images.githubusercontent.com/39285147/180517030-eedfd66d-7cd8-4109-9fb2-87f6a29d3c7c.png)

# 구현

## 데이터 관찰

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from keras.datasets import cifar10
(X_train, y_train) , (X_test, y_test) = cifar10.load_data() # 훈련, 테스트 데이터 생성
X_train.shape
```

        (50000, 32, 32, 3)


-* 이미지 개수*: 50000개
- *이미지 해상도*: 32x32
- *컬러 RGB*: 3


## 데이터 시각화

```python
i = 30009
plt.imshow(X_train[i])
print(y_train[i])
```

        1

![image](https://user-images.githubusercontent.com/39285147/180518417-37e04fc2-210e-40db-b111-e219b5d66a9b.png)        


클래스 리스트에서 인덱스가 1인 클래스, 'Cars'에 속하는 이미지인 것을 확인할 수 있다.

```python
# 한 번에 여러 이미지 배출하여 비교하기

W_grid = 4 # 그리드 가로
L_grid = 4 # 그리드 세로

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel() # 4x4 행렬을 16개의 요소를 가진 선형배열로 변환한다

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off') # 축 안 보이게 만들기
    
plt.subplots_adjust(hspace = 0.4) # 이미지 사이 공간 벌리기
```

![image](https://user-images.githubusercontent.com/39285147/180518577-9f9298f7-ff70-4bf5-ba08-b353fa610a6a.png)

## 데이터 전처리

### 이미지 포맷 설정 (float32)
```python
# 이미지 포맷은 'float32'여야 한다
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_cat = 10 # 클래스 개수
y_train

```

array([[6],
    [9],
    [9],
    ...,
    [9],
    [1],
    [1]], dtype=uint8)

클래스 인덱스로 종속변수가 표현된 것을 확인해볼 수 있다.

이를 인덱스 번호가 아니라 'One-Hot Encoding' 방법을 활용하여 범주화 시켜보자.

### Categorical Data 범주화시키기

```python
import keras
y_train = keras.utils.to_categorical(y_train, number_cat) # one-hot encoding 범주형으로 변환하기
y_test = keras.utils.to_categorical(y_test, number_cat)
```

array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 1.],
       [0., 0., 0., ..., 0., 0., 1.],
       ...,
       [0., 0., 0., ..., 0., 0., 1.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.]], dtype=float32)


### 정규화
독립변수들은 픽셀에 할당된 [0, 255] 사이의 수치이므로, [0, 1] 사이 값들로 *정규화*를 거칠 필요가 있다.

```python
X_train = X_train/255
X_test = X_test/255
```

## 모델 훈련하기

```python
from keras.models import Sequential # 신경망을 순차적으로 쌓는다
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout # CNN 관련 클래스
from keras.optimizers import Adam # Adam 최적화 옵티마이저
from keras.callbacks import TensorBoard # TensorFlow 시각화 도구
```


```python
cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))


cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 1024, activation = 'relu'))

cnn_model.add(Dense(units = 1024, activation = 'relu'))

cnn_model.add(Dense(units = 10, activation = 'softmax')) # 최종 출력 클래스 개수 10개

cnn_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr = 0.001), metrics = ['accuracy']) # 범주형 크로스엔트로피 손실함수, rmsprop 옵티마이저와 accuracy를 척도로 사용한다.

history = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 1, shuffle = True) # 이미지 순서를 섞어(shuffle) 모델 학습을 시작한다
```

> ReLU: 회귀 작업 관련 함수로 연속적인 출력값을 생성한다

> Softmax: 분류에 사용된다.

> CPU vs. GPU
> 학습이 오래 지속된다면 GPU가 아닌 CPU를 사용한다.

> ANN 은닉층 *뉴런개수*와 CNN *필터개수*를 증가시키면, 모델 복잡도가 증가하여 학습 시간이 늘어난다.

## 모델 평가

```python
evaluation = cnn_model.evaluate(X_test, y_test) # 실제값과 예측값을 비교하여 정확도를 도출한다

predicted_classes = cnn_model.predict_classes(X_test) # 모델 예측값을 도출한다

y_test = y_test.argmax(1) # one hot 인코딩의 이진수로 표현된 값을 십진수로 바꿔준다

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    
```

![image](https://user-images.githubusercontent.com/39285147/180562014-e6d8d90f-7dc5-449f-a7cb-233e50bd6fa3.png)


```python
# 혼동행렬로 평가지표 표현하기

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predicted_classes)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True) # 해당 데이터가 많거나 높은 경우 색을 사용해 시각화하는 그래프
```

![image](https://user-images.githubusercontent.com/39285147/180563602-b9035fea-733c-4514-a223-63c1e99608a6.png)

혼동행렬에서 큰 수치를 띄는 값들은 오류(False Negative, False Positive)를 의미한다.


## 모델 저장하기

```python
import os 
directory = os.path.join(os.getcwd(), 'saved_models') # getcwd: get current working directory

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn_model.save(model_path)
```


## Image Augmentation 활용 모델 개선

Image Augmentation
- 과적합 문제를 해소하고 정확도를 높이기 위해 변화를 적용하는 전처리 기법. 주로, 기존 인풋 이미지에 조금씩 변화(뒤집기, 회전, etc.)를 주어 학습 데이터량을 증가시켜 데이터 차원이 모델 복잡도를 웃돌게 만든다.

> *ImageDataGenerator* 클래스를 통해 변화를 끌어낸다.

### 이미지 증강으로 새로운 데이터 만들어내기

```python
import keras
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

n = 8 
X_train_sample = X_train[:n] # 8개의 샘플 이미지를 가져온다
```

```python
# 이미지 변화주기
from keras.preprocessing.image import ImageDataGenerator

# dataget_train = ImageDataGenerator(rotation_range = 90) # 이미지 회전
# dataget_train = ImageDataGenerator(vertical_flip=True) # 뒤집기/반전
# dataget_train = ImageDataGenerator(height_shift_range=0.5) ## 
dataget_train = ImageDataGenerator(brightness_range=(1,3)) # 밝기 조정

dataget_train.fit(X_train_sample) # 이미지 생성기 적용
```

```python
# 이미지에 변화 적용하기
from scipy.misc import toimage # 배열을 이미지로 변환한다

fig = plt.figure(figsize = (20,2)) # 새로 생성될 이미지 사이즈 조정

# 훈련 샘플을 가져와 이미지 flow를 생성한다.
# 변형된 이미지를 배치 단위로 불러올 수 있는 Generator(*flow()*)을 생성해 준다 (8개 요소 포함하는 리스트 형태로 한 번에 가져오게 된다)
for x_batch in dataget_train.flow(X_train_sample, batch_size = n):
     for i in range(0,n):
            ax = fig.add_subplot(1, n, i+1)
            ax.imshow(toimage(x_batch[i]))
     fig.suptitle('Augmented images (rotated 90 degrees)')
     plt.show()
     break;
```

![image](https://user-images.githubusercontent.com/39285147/180567864-9bef861d-2973-4367-9344-436ae78ee7c9.png)

결과에서 보이는 것처럼 기존 인풋 이미지의 밝기를 수정하여 새로운 데이터를 만들어냈다! 

### 이미지 증강으로 만들어낸 데이터로 모델 새로 학습하기

```python
from keras.preprocessing.image import ImageDataGenerator

# 이미지 생성기 특성 정의
datagen = ImageDataGenerator(
                            rotation_range = 90,
                            width_shift_range = 0.1,
                            horizontal_flip = True,
                            vertical_flip = True
                             )

datagen.fit(X_train) # 이미지 생성기 특성 적용

cnn_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 2) # 이미지 생성기로 만든 데이터 학습에 이용할 때 fit_generator() 함수를 사용한다

score = cnn_model.evaluate(X_test, y_test) # 모델 성능평가
print('Test accuracy', score[1])

# save the model
directory = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model_Augmentation.h5')
cnn_model.save(model_path)
```
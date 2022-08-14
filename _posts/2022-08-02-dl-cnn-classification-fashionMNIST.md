---
layout: single
title: "CNN - Classification: fashionMNIST"
categories: DL
tag: [deep learning, cnn, classification]
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

Tensorflow & PyTorch 활용 CNN 모델의 fashionMNIST 이미지 분류 <span style="color: blue"> fashionMNIST image classification using CNN w/ Tensorflow and PyTorch </span>

# Code
**[Notice]** [fasionMNIST(PyTorch)](https://github.com/hchoi256/cs540-AI/tree/main/introduction-to-pytorch)
{: .notice--danger}

# Loading the dataset

**[Notice]** [Download Dataset (Kaggle)](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
{: .notice--danger}

Fashion MNIST Dataset: 28×28 픽셀의 이미지 70,000개 <span style="color: blue"> fashionMNIST: 70,000 images (28x28) </span>

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fashion_train_df = pd.read_csv("fashion-mnist-train.csv")
fashion_test_df = pd.read_csv("fashion-mnist-test.csv")

fashion_train_df.shape, fashion_test_df.shape
```


        (60000, 785) (10000, 785)



```python
training = np.array( fashion_train_df, dtype = "float32" )
test = np.array( fashion_test_df, dtype = "float32" )

type(training), type(test)
```

        (numpy.ndarray, numpy.ndarray)


신경망 학습을 위해 dataframe 에서 ndarray로 변환한다. <span style="color: blue"> Converting 'dataframe' to 'ndarray' for neural network training. </span>



# Observing the dataset

```python
import random

i = random.randint(1, 60000)
plt.imshow(training[i, 1:].reshape( (28, 28) ))
```

![image](https://user-images.githubusercontent.com/39285147/182385271-f1ff8063-f94e-4429-a4b4-297a03614bc3.png)



맛보기로 한 이미지를 가져와서 시각화했다. <span style="color: blue"> For testing, I randomly extracted an image and visualized it. </span>


```python
plt.imshow(training[i, 1:].reshape( (28, 28) ), cmap = "gray")
```

![image](https://user-images.githubusercontent.com/39285147/182385314-997733f2-2858-4d25-a767-37387a3d2033.png)

흑백 사진으로 변환하려면, 'cmap' 속성만 추가하면 된다. <span style="color: blue"> To convert to a dark image, you can simply add the 'cmap' attribute. </span>


```python
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot

label = training[i, 0]
label
```

        7.0


7은 Sneaker 클래스의 인덱스이다. <span style="color: blue"> This image is classified as 'sneaker'.</span>


# Splitting the dataset into the Training set and Test set

```python
# normalizing the images
X_train = training[:, 1:]/255
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
y_train = training[:, 0] 

X_test = test[:, 1:]/255
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
y_test = test[:, 0]

X_train.shape, y_train.shape
```

        (60000, 28, 28, 1) (60000,)


흑백사진의 RGB 차원(1)도 추가해주기 위해 reshape하였다. <span style="color: blue"> Adding a dimension for the RGB of the dark images. </span>

# Building the model

```python
from tensorflow.keras import datasets, layers, models

cnn = models.Sequential()

cnn.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)))
cnn.add(layers.MaxPooling2D(2, 2))
cnn.add(layers.BatchNormalization())

cnn.add(layers.Conv2D(64, (3, 3), activation = "relu"))
cnn.add(layers.MaxPooling2D(2, 2))
cnn.add(layers.BatchNormalization())

cnn.add(layers.Conv2D(64, (3, 3), activation = "relu"))

cnn.add(layers.Flatten())

cnn.add(layers.Dense(64, activation = "relu"))
cnn.add(layers.Dense(10, activation = "softmax"))

cnn.summary()
```


        Model: "sequential_2"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)           (None, 26, 26, 32)        320       
                                                                        
        max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
        )                                                               
                                                                        
        conv2d_2 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                        
        max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
        2D)                                                             
                                                                        
        conv2d_3 (Conv2D)           (None, 3, 3, 64)          36928     
                                                                        
        flatten (Flatten)           (None, 576)               0         
                                                                        
        dense (Dense)               (None, 64)                36928     
                                                                        
        dense_1 (Dense)             (None, 10)                650       
                                                                        
        =================================================================
        Total params: 93,322
        Trainable params: 93,322
        Non-trainable params: 0
        _________________________________________________________________


```python
cnn.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
```

이전 글에서도 꾸준히 사용했던 것처럼, 이번에도 'Adam'과 크로스 엔트로피를 활용해서 분류 문제를 해결해보자.<span style="color: blue"> As we used to do in the previous posts, we also use 'Adam' and 'Cross Entrophy'. </span>

이전 글에서 지속적으로 모델 형성 공식을 언급했으므로, 이 부분에 대한 추가적인 설명은 생략한다. <span style="color: blue"> Since the model formation has been continuously mentioned in the previous posts, additional explanations on this part will be omitted. </span>
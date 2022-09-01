---
layout: single
title: "Data Preprocessing Techniques"
categories: Study
tag: [Data Preprocessing, Python]
# toc: false
# toc_sticky: false
toc_label: "쭌스log"
# author_profile: false
header:
    teaser: /assets/images/posts/data-thumbnail.jpg
sidebar:
    nav: "docs"
---


'raw_data' is the temporary dataset, and we are going to address various preprocessing tasks with it.


****
# 데이터 확인 (Describing the dataset)

```python
raw_data.info()
```

        <class 'pandas.core.frame.DataFrame'>
        Int64Index: 23 entries, 0 to 29
        Data columns (total 8 columns):
        #   Column          Non-Null Count  Dtype         
        ---  ------          --------------  -----         
        0   customer_id     23 non-null     int64         
        1   date            23 non-null     datetime64[ns]
        2   age             23 non-null     int32         
        3   gender          23 non-null     object        
        4   country         23 non-null     object        
        5   item purchased  23 non-null     float64       
        6   value           23 non-null     int64         
        7   monthly visits  23 non-null     float64       
        dtypes: datetime64[ns](1), float64(2), int32(1), int64(2), object(2)
        memory usage: 1.5+ KB


```python
raw_data["country"].unique(), raw_data["country"].nunique()
```

        array(['US', 'India', 'France', 'Sweden', 'USA', 'Germany', 'Chile',
            'Saudi Arabia', 'Japan', 'Norway', 'Spain', 'United Kingdom',
            'Switzerland', 'Russia'], dtype=object)

        14

```python
raw_data = ["value", "monthly visits", "item purchased"]
raw_data[ out_list ] # show the columns only
```

![image](https://user-images.githubusercontent.com/39285147/183245866-6fb2efd2-0baa-428a-8d28-60ca50f0eac8.png)


```python
from scipy import stats
np.abs(stats.zscore(raw_data[ out_list ])) # convert data in zscore
```

![image](https://user-images.githubusercontent.com/39285147/183245894-aa7abed2-7b18-4c5b-bb05-7c1de3c925c9.png)


```python
raw_data.drop( ["customer_id", "date"], axis = 1, inplace = True ) # remove columns
```


****
# 결측치 (Missing Values)

```python
raw_data.isna().sum() # return the number of missing values in the dataset

raw_data.dropna(inplace=True) # remove rows including 'NaN'
```

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy= "mean") # usually replaced with mean value

raw_data = imputer.fit_transform(raw_data) # apply the tranformation
```

****
# 데이터 정리 (Organizing the data)
```python
raw_data.replace("ERR", np.nan, inplace = True) # replace 'ERR' to 'NaN'

raw_data["gender"].str.strip() # remove blanks at edge
```


```python
raw_data["gender"] = raw_data["gender"].map({"Male": 0, "Female": 1}) # categorical data
```


        0     0
        1     1
        4     0
        6     0
        7     0
        8     1
        9     0
        11    1
        12    0
        13    0
        14    1
        15    0
        16    1
        18    0
        19    1
        20    0
        21    1
        23    0
        24    1
        25    0
        27    0
        28    0
        29    1
        Name: gender, dtype: int64


```python
raw_data["age"] = raw_data["age"].astype(int) # convert type
```


```python
def classify(label):
    if label < 500:
        return "Normal"
    else:
        return "Active"

raw_data["label"] = raw_data["monthly visits"].apply( lambda x: classify(x) )
```

```python
raw_data = pd.get_dummies(raw_data, columns = ["country"]) # make new columns with the data in the 'country' column
```

![image](https://user-images.githubusercontent.com/39285147/183245822-467b1cd4-2b08-4554-8ed5-ffcacedd9d92.png)


# Concat, Merge, Join
```python
df_1, df_2
```

![image](https://user-images.githubusercontent.com/39285147/187830604-72195444-1e11-4bd6-aa9b-eadca6a7796c.png) ![image](https://user-images.githubusercontent.com/39285147/187830639-195cdb22-4a75-4b6e-b6cc-dc7169771246.png)

```python
# 비어 있는 항목에는  NaN 으로 채워진다.
df_cat = pd.concat([df_1, df_2, df_3]) # 아래로 합치기
df_cat = pd.concat([df_1, df_2, df_3], axis = 1) # 옆으로 합치기
```

```python
pd.merge(df_1, df_2, on = "Customer Name", how = "inner")
```

![image](https://user-images.githubusercontent.com/39285147/187830696-daac193c-a038-4104-b0a9-a42f41e041b1.png)


```python
pd.merge(df_2, df_1, on = "Customer Name", how = "outer")
```

![image](https://user-images.githubusercontent.com/39285147/187830744-09e60a4a-4d20-413c-a3c6-883d0e67d6fe.png)


```python
pd.merge(df_1, df_2, on = "Customer Name", how = "outer").drop_duplicates()
```

![image](https://user-images.githubusercontent.com/39285147/187830721-2e3f227b-0004-472f-8f84-cef46a04a7cd.png)

```python
df_1.join(df_2, how="left"), df_1.join(df_2, how="right"), df_1.join(df_2, how="inner"), df_1.join(df_2, how="outer")
```

![image](https://user-images.githubusercontent.com/39285147/187833998-3344532e-91b9-4ab7-97e0-57798e808dbe.png) ![image](https://user-images.githubusercontent.com/39285147/187834052-47868fbd-1fd9-4e3a-8b3e-a981578b39aa.png) ![image](https://user-images.githubusercontent.com/39285147/187834115-b144a12a-5caf-4c02-bfbe-31d0aca03aab.png) ![image](https://user-images.githubusercontent.com/39285147/187834134-703cd536-7de6-4cde-a468-2ae459ea7833.png)

```python

```
```python

```

# Indexing

```python
df_2.set_index( ["Customer Name"], inplace = True)
```

![image](https://user-images.githubusercontent.com/39285147/187833974-500ef442-27a5-4e3e-8584-5f01ee8937a6.png)


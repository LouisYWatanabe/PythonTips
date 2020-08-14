# DataFrameの作成


```python
import pandas as pd
# データフレームの作成
pd.DataFrame()

```

### 書式

	pd.DataFrame( 辞書型データ, index=...)

### 引数

- <b>columns=[リスト]</b>
ソート順の指定
第二引数以降に`columns=[リスト]`とし、
リスト内の順番で表示する

- <b>index=</b>
インデックスの指定
第二引数以降に`index=[リスト]`とし、
インデックス名をリスト順にする

### 例

```python
import pandas as pd

# 辞書型を使いDataFrameを作成
df = pd.DataFrame({"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]})

print(df)
print()
print(df.head())
```

```python
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3

       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3

```

### 説明

辞書型のデータ（`{key1: value1, key2: value2, ...}`）を
`columns`を指定しないで渡すと`key`で昇順にソートされる。


```python
import numpy as np
# NumPyからDataFrameを作成
ndarray = np.arange(0, 10).reshape(2, 5)
ndarray
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
pd.DataFrame(ndarray)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# columnsとindexを指定して作成する
columns = ['a', 'b', 'c', 'd', 'e']
index = ['index1', 'index2']
pd.DataFrame(ndarray, index=index, columns=columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>index1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>index2</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dictionaryから作成
data1 = {
    'name': 'John',
    'sex': 'male',
    'age': 22
}
data2 = {
    'name': 'Zack',
    'sex': 'male',
    'age': 30
}
data3 = {
    'name': 'Emily',
    'sex': 'female',
    'age': 32
}
pd.DataFrame([data1, data2, data3])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>male</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zack</td>
      <td>male</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emily</td>
      <td>female</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dictionaryの各値がリストになっているケース
data = {
    'name': ['John', 'Zack', 'Emily'],
    'sex': ['male', 'male', 'female'],
    'age': [22, 30, 32]
}
# dfという変数をよく使う
df = pd.DataFrame(data)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>male</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zack</td>
      <td>male</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emily</td>
      <td>female</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>

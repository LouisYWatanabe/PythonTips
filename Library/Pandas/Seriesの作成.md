# Seriesの作成


```python
import pandas as pd
# データフレームの作成
pd.Series()
```

### 書式

	pd.Series( 辞書型データ, index=...)

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

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

# dataとindexを含むSeriesを生成しdfに代入
df = pd.Series(data, index=index)

print(df)
```
```python
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
dtype: int64
```

```python
# dictionaryからSeriesを作る
data = {
    'name': 'John',
    'sex': 'male',
    'age': 22
}
john_series = pd.Series(data)
john_series
```

```python
name    John
sex     male
age       22
dtype: object
```

```python
# NumPy Arrayからも作成可能
array = np.array([100, 200, 300])
pd.Series(array)
```

```python
0    100
1    200
2    300
dtype: int64
```

```python
# index引数で，indexにラベルをつけることが可能
labels = ['a', 'b', 'c']
series = pd.Series(array, index=labels)
series
```

```python
a    100
b    200
c    300
dtype: int64
```

```python
# 値を取り出す
john_series['name']
```

```python
'John'
```

### .values

```python
# valuesで，値をnumpyとして扱える
series.values
```

```python
array([100, 200, 300])
```

```python
# NumPyと同じように統計量を計算できる
series.mean()
```

```python
200.0
```

### 説明

辞書型のデータ（`{key1: value1, key2: value2, ...}`）を
`columns`を指定しないで渡すと`key`で昇順にソートされる。


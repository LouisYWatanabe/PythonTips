# フィルタリング

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.DataFrame({"fruits":data}, index=index, )

conditions = [True, True, False, True, False]
print(series[conditions])
```

### 書式

	condition = [True, False]
	# データフレームのTrueのみを抽出する
	df[condition]

### 例

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
df = pd.Series(data, index=index)

# 値が5以上10未満の要素を含むSeriesを抽出
df = df[5 <= df][df < 10]
# df = df[(5 <= df) & (df < 10)] でもよい

print(df)
```

```
orange    5
banana    8
dtype: int64
```

```python
import numpy as np
import pandas as pd

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrameを生成し、列を追加
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# フィルタリングを用いて、dfの"apple"列が5以上かつ
# "kiwifruit"列が5以上の値をもつ行を含むDataFrameをdfに代入
df = df[(df["apple"] >= 5) & (df["kiwifruit"] >= 5)]

print(df)
```

```
   apple  orange  banana  strawberry  kiwifruit
1      6       8       6           3         10
5      8       2       5           4          8
8      6       8       4           8          8
```

### 説明

SeriesやDataFrameは条件式に従い`bool`型のシーケンスを返す

条件を複数つけたい時は[][]のように[]を並べる
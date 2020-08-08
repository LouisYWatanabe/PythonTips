# データ・インデックスの抽出

```python
import pandas as pd
# データフレームの作成
df = pd.Series(...)
# データ値の取得
print(df.values)
# インデックスの取得
print(df.index)
```

### 書式

	1. データ値の取得
		df.values
	2. インデックスの取得
		df.index

### 例

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
df = pd.Series(data, index=index)

# データ値の取得
df_values = df.values

# インデック値の取得
df_index = df.index

print(df_values)
print()
print(df_index)
```

```python
[10  5  8 12  3]

Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')
```

### 説明

データ値は`values`、インデックスは`index`でそれぞれ取得できる


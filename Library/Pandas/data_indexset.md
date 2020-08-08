# インデックス・カラムの取得・設定

```python
df = pd.DataFrame(...)
df.index = [1, ..]
df.columns = ["", ""]
```

### 書式

	1. インデックスの取得
		df.values
	2. カラムの取得
		df.columns
		
### 例

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
df = pd.DataFrame([series1, series2])

# dfのインデックスとカラムを再設定する
df.index = [1, 2]
df.columns = ["a", "o", "b", "s", "k"]

print(df)
print(df.columns)
```
```
    a   o   b   s  k
1  10   5   8  12  3
2  30  25  12  10  8
Index(['a', 'o', 'b', 's', 'k'], dtype='object')
```

### 説明
df.indexでインデックス、df.columsで列数のリストを取得・設定できる

行数または列数を設定するときは、元のデータフレームと同じ長さである必要がある
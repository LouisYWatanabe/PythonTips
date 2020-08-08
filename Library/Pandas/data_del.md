# 要素の削除

```python
import pandas as pd
# データフレームの作成
df = pd.Series(...)
df.drop("削除したいインデックス値")
```

### 書式

	drop(削除したいインデックス値)

### 例

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

# Seriesデータの定義
df = pd.Series(data, index=index)

# インデックスがstrawberryの要素を削除
df = df.drop("strawberry")

print(df)
```

```python
apple        10
orange        5
banana        8
kiwifruit     3
dtype: int64
```

### 説明

引数に指定した削除したインデックス値を持つ要素を削除できる


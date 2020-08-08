# 欠損値の有無の確認

```python
import pandas as pd
# 欠損値の有無の確認
df.isnull().sum()
```

### 書式

	df: データフレーム

### 例

```python
import pandas as pd

df = pd.read_csv("../data/iris.data", header=None)

# 欠損値の有無の確認
df.isnull().sum()
```

```python
0    0
1    0
2    0
3    0
4    0
dtype: int64
```

### 説明


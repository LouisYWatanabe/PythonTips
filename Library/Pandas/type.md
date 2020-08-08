# データ型の確認

```python
import pandas as pd
# データ型の確認
df.dtypes
```

### 書式

	df: データフレーム

### 例

```python
import pandas as pd

df = pd.read_csv("../data/iris.data", header=None)

# データの型の確認
df.dtypes
```

```python
0    float64
1    float64
2    float64
3    float64
4     object
dtype: object
```

### 説明


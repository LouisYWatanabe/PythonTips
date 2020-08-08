# 要素の個数、頻度（出現回数）をカウント（Value_Counts()）

[Five ways to use value_counts()](https://www.kaggle.com/parulpandey/five-ways-to-use-value-counts)

```python
import pandas as pd

train['Sex'].value_counts()
```

### 書式

	normalize=True
	相対的なパーセンテージを取得

	sort: bool, default True
	Sort by frequencies.

	ascending: bool, default False
	Sort in ascending order.

	bins: int, optional
	Rather than count values, group them into half-open bins, a convenience for pd.cut, only works with numeric data.

	dropna: bool, default True
	Don’t include counts of NaN.
### 例

```python
import pandas as pd
# Reading in the data
train = pd.read_csv('../input/titanic/train.csv')

train['Sex'].value_counts()


train['Embarked'].value_counts(normalize=True)
```

```
male      577
female    314
Name: Sex, dtype: int64

S    0.724409
C    0.188976
Q    0.086614
Name: Embarked, dtype: float64
```

### 説明
データフレームの要約統計量を表示する
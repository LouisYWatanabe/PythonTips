# 欠損値の削除

```python
import pandas as pd
# 欠損値の削除
df.dropna()
```

### 書式

	df: データフレーム

	axis = 'columns': 欠損値を含む列を削除する
	how = 'all' : 全ての要素が欠損している行を削除する
	thresh = 2 : 欠損していない要素数が2以上の行のみを保持する
	subset = ['列1', '列2'] : 列1、列2で欠損値を含む行を削除する。

### 例

```python
import pandas as pd

df = pd.read_csv("../data/iris.data", header=None)

# 欠損値の有無の確認
df.isnull().sum()
# isnull()は、欠損値に対しTrueを返し、欠損値以外にはFalseを返す
# sum()は、Trueを1、Falseを0として合計する
# よってdf.isnull().sum()で欠損値を算出することができる
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


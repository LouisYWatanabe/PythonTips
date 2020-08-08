# 欠損値の補完

```python
import pandas as pd
# 欠損値の埋める
df.fillna('FILL')
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

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# trainとtestを縦に連結
data = pd.concat([train, test], axis=0, sort=False)

data['Age'].fillna(20)                   # 列Ageの欠損値を20で穴埋め
data['Age'].fillna(data['Age'].mean())   # 列Ageの欠損値をAgeの平均値で穴埋め
data['Age'].fillna(data['Age'].median()) # 列Ageの欠損値をAgeの中央値で穴埋め
data['Age'].fillna(data['Age'].mode())   # 列Ageの欠損値をAgeの最頻値で穴埋め
```

### 説明


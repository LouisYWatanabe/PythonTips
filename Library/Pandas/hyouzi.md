# データの表示確認（リスト）

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

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# trainとtestを縦に連結
df = pd.concat([train, test], axis=0, sort=False)

train['Name'][0:5]


```

```python
0                              Braund, Mr. Owen Harris
1    Cumings, Mrs. John Bradley (Florence Briggs Th...
2                               Heikkinen, Miss. Laina
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
4                             Allen, Mr. William Henry
Name: Name, dtype: object
```

### 説明

